"""Verify the source-only package; this does not audit scientific evidence."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "LICENSE_SCOPE.md",
    "NOTICE",
    "OMISSIONS.md",
    "README.md",
    "SOURCE-MANIFEST.json",
    "config.yaml",
    "pyproject.toml",
    "uv.lock",
}
DOCS = {"BLOG_POST.md", "PAPER.md", "RESULTS_INDEX.md", "ARCHITECTURE.md"}
PATTERNS = {
    "personal home path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "Windows home path": re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    "API token": re.compile(rb"sk-[A-Za-z0-9_-]{30,}"),
}


def allowed(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or "__pycache__" in p.parts:
        return False
    if len(p.parts) == 1:
        return name in ROOT_FILES
    return (p.parts[0] in {"src", "scripts", "tests"} and p.suffix == ".py") or (
        p.parts[0] == "docs" and len(p.parts) == 2 and p.name in DOCS
    )


def check_content(name, raw):
    if not allowed(name):
        raise ValueError(f"Unapproved source path: {name}")
    raw.decode("utf-8")
    for label, pattern in PATTERNS.items():
        if pattern.search(raw):
            raise ValueError(f"{label} detected in {name}")


def verify(root=ROOT):
    manifest = json.loads((root / "SOURCE-MANIFEST.json").read_text())
    expected = manifest["files"]
    if "SOURCE-MANIFEST.json" in expected:
        raise ValueError("Manifest must not hash itself")
    # Include staged, unstaged and nonignored new files; ignore runtime caches only.
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    actual = {x.decode() for x in output.split(b"\0") if x}
    if actual != set(expected) | {"SOURCE-MANIFEST.json"}:
        raise ValueError(
            f"File set differs: {sorted(actual ^ (set(expected) | {'SOURCE-MANIFEST.json'}))}"
        )
    for name in sorted(actual):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular source file: {name}")
        raw = path.read_bytes()
        check_content(name, raw)
        if name in expected and hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError(f"Source digest changed: {name}")
    return len(actual)


if __name__ == "__main__":
    print(f"{verify()} source-only files verified. Scientific evidence audits were not run.")
