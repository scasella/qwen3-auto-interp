"""Build a deterministic source-only archive without reading research evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = Path("release/source-only-2026-10-21")
ARCHIVE_NAME = "qwen3-auto-interp-source-apache-2.0.tar.gz"
PREFIX = "qwen3-auto-interp-source/"
METADATA = {"pyproject.toml", "uv.lock", "config.yaml"}
NOTICES = {"LICENSE", "NOTICE", "LICENSE_SCOPE.md"}
PATTERNS = {
    "local_home_path": rb"/(?:Users|home)/[^\s/\"\\]+",
    "email": rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "credential": (
        rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|"
        rb"github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|"
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def payload() -> tuple[dict[str, bytes], dict]:
    """Select committed source paths, plus this new builder and approved notices."""
    base = git("rev-parse", "HEAD").decode().strip()
    tracked = git("ls-tree", "-r", "--name-only", base).decode().splitlines()
    paths = {
        p for p in tracked
        if (p.startswith(("src/", "scripts/", "tests/")) and p.endswith(".py"))
        or p in METADATA
    }
    paths.add("scripts/prepare_source_release.py")
    files: dict[str, bytes] = {}
    changed = {}
    for name in sorted(paths | NOTICES):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular file: {name}")
        files[name] = path.read_bytes()
        if name in tracked:
            original = git("show", f"{base}:{name}")
            if original != files[name]:
                changed[name] = {"base_sha256": sha(original), "release_sha256": sha(files[name])}
        else:
            changed[name] = {"base_sha256": None, "release_sha256": sha(files[name])}
    unexpected = set(changed) - NOTICES - {"pyproject.toml", "scripts/prepare_source_release.py"}
    if unexpected:
        raise ValueError(f"Unreviewed source edits: {sorted(unexpected)}")
    files["README.md"] = (ROOT / PACKAGING / "PUBLIC_README.md").read_bytes()
    files["OMISSIONS.md"] = (ROOT / PACKAGING / "OMISSIONS.md").read_bytes()
    files[".gitignore"] = (
        b".venv/\n__pycache__/\n.pytest_cache/\n.ruff_cache/\n.DS_Store\n.env\n.env.*\n"
        b"data/\nreports/\n.worktrees/\n"
    )
    for name, data in files.items():
        for category, pattern in PATTERNS.items():
            if re.search(pattern, data):
                raise ValueError(f"Release privacy pattern {category}: {name}")
    provenance = {
        "schema": 1,
        "source_base_commit": base,
        "project_code_license": "Apache-2.0",
        "publication_status": "PREPARED_LOCALLY_NOT_PUBLISHED",
        "scope": "source-only; no evidence, corpus or historical Git objects",
        "changes_from_base": changed,
        "files": {p: sha(b) for p, b in sorted(files.items())},
    }
    files["SOURCE-MANIFEST.json"] = (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode()
    return files, provenance


def build(output: Path) -> dict:
    output = output.resolve()
    if output == ROOT or output.is_relative_to(ROOT / "data") or output.is_relative_to(ROOT / "reports"):
        raise ValueError("Output must be a separate directory outside protected evidence")
    files, provenance = payload()
    output.mkdir(parents=True, exist_ok=True)
    for name in (ARCHIVE_NAME, "candidate-manifest.json", "SHA256SUMS"):
        target = output / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Release output must be a regular file: {name}")
    archive = output / ARCHIVE_NAME
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo(PREFIX + name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
    # Verify archive paths, exact membership, file types and hashes independently of extraction.
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(m.isfile() and m.name.startswith(PREFIX) for m in members)
        restored = {m.name.removeprefix(PREFIX): tar.extractfile(m).read() for m in members}
    assert len(restored) == len(members) and restored == files
    manifest = {
        "source_base_commit": provenance["source_base_commit"],
        "archive": ARCHIVE_NAME,
        "archive_sha256": sha(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "file_count": len(files),
        "project_code_license": "Apache-2.0",
        "publication_status": "PREPARED_LOCALLY_NOT_PUBLISHED",
        "files": {p: sha(b) for p, b in sorted(files.items())},
    }
    (output / "candidate-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    (output / "SHA256SUMS").write_text(f"{manifest['archive_sha256']}  {ARCHIVE_NAME}\n")
    return {k: v for k, v in manifest.items() if k != "files"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / PACKAGING)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
