"""Read-only adapters for the consolidated v9–v12 campaign.

Registered source reads use preserved versions. Evidence reads use verified local
copies. Git registration checks use hash-checked raw commit/tree/blob proofs;
no private Git history is imported into a release repository.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "consolidation/2026-10-22"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_catalog():
    catalog = json.loads((RECORDS / "audit-catalog.json").read_text())
    assert digest(RECORDS / "git-proofs.json") == catalog["git_proofs_sha256"]
    return catalog


def verify_references(catalog):
    """Verify each reference, hashing duplicate physical destinations only once."""
    protected = json.loads((ROOT / "data/bounded-poc-v12-cross-scale-3hop/protected.json")
                           .read_text())["files"]
    assert len(protected) == catalog["protected_references"]
    for name, expected in protected.items():
        assert catalog["aliases"][name]["sha256"] == expected, name
    measured = {}
    for record in catalog["aliases"].values():
        name = record["path"]
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), name
        if name not in measured:
            measured[name] = digest(path)
        assert measured[name] == record["sha256"], f"Consolidated reference changed: {name}"
    for study in catalog["studies"].values():
        for record in study["sources"].values():
            assert digest(ROOT / record["path"]) == record["sha256"]
    return catalog["protected_references"]


def git_registration(proof, commit):
    """Validate the actual Git object chain, then return its blob and timestamp."""
    objects = {}
    for oid, record in proof["objects"].items():
        value = base64.b64decode(record["data"], validate=True)
        header = f"{record['kind']} {len(value)}\0".encode()
        assert hashlib.sha1(header + value).hexdigest() == oid, "Git object digest mismatch"
        objects[oid] = (record["kind"], value)
    kind, value = objects[commit]
    assert kind == "commit"
    lines = value.split(b"\n")
    tree = lines[0].decode().split()[1]
    timestamp = next(line for line in lines if line.startswith(b"committer ")).split()[-2]
    for component in Path(proof["path"]).parts:
        kind, value = objects[tree]
        assert kind == "tree"
        offset, matches = 0, []
        while offset < len(value):
            end = value.index(0, offset)
            entry = value[offset:end].split(b" ", 1)[1].decode()
            oid = value[end + 1:end + 21].hex()
            offset = end + 21
            if entry == component:
                matches.append(oid)
        assert len(matches) == 1, component
        tree = matches[0]
    kind, blob = objects[tree]
    assert kind == "blob"
    return blob, timestamp + b"\n"


@contextlib.contextmanager
def registered_view(study, catalog):
    """Adapt paths and registration proofs without changing numerical decisions."""
    config = catalog["studies"][study]
    sources = {str(ROOT / k): ROOT / v["path"] for k, v in config["sources"].items()}
    aliases = {k: ROOT / v["path"] for k, v in catalog["aliases"].items()}
    proofs = json.loads((RECORDS / "git-proofs.json").read_text())
    verified = {key: git_registration(value, key) for key, value in proofs.items()}
    real_open, real_glob, real_check = io.open, Path.rglob, subprocess.check_output

    def read_only(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, mode, *args, **kwargs)
        if any(flag in mode for flag in "wax+"):
            raise PermissionError(f"Uncaptured audit write: {file}")
        key = str(Path(file).absolute())
        return real_open(sources.get(key, aliases.get(key, file)), mode, *args, **kwargs)

    def historical_datasets(path, pattern, *args, **kwargs):
        values = real_glob(path, pattern, *args, **kwargs)
        if study == "v9" and path.absolute() == ROOT / "data" and pattern == "dataset.json":
            allowed = set(config["datasets"])
            return (p for p in values if str(p.relative_to(ROOT)) in allowed)
        return values

    def git_show(command, *args, **kwargs):
        if command[:2] != ["git", "show"]:
            return real_check(command, *args, **kwargs)
        if len(command) == 3 and ":" in command[2]:
            commit, path = command[2].split(":", 1)
            assert proofs[commit]["path"] == path
            return verified[commit][0]
        if command[2:4] == ["-s", "--format=%ct"] and len(command) == 5:
            return verified[command[4]][1]
        raise ValueError(f"Unregistered Git audit request: {command}")

    with (patch("builtins.open", read_only), patch("io.open", read_only),
          patch.object(Path, "rglob", historical_datasets),
          patch.object(subprocess, "check_output", git_show)):
        yield


def run_v9():
    from scripts import audit_bounded_poc_v9_json_hop0 as auditor

    captured = {}

    def capture(name, value):
        assert name not in captured
        captured[name] = json.loads(json.dumps(value, allow_nan=False))

    with patch.object(auditor, "write", capture), contextlib.redirect_stdout(io.StringIO()):
        auditor.audit()
    assert set(captured) == {"results.json", "facts.json", "audit.json"}
    for name, value in captured.items():
        assert value == json.loads((auditor.OUT / name).read_text()), name
    facts = captured["facts.json"]
    claims = json.loads((auditor.OUT / "review-results.json").read_text())["numeric_claims"]
    checks = v9_references(facts, claims)
    adjudication = json.loads((auditor.OUT / "review-adjudication.json").read_text())
    assert checks == adjudication["claims"]
    assert all(c["supported"] for c in checks) == adjudication["all_numeric_claims_supported"]
    result = captured["results.json"]
    return dict(verification_passed=True,
                scientific_passed=result["track1"] and result["track2"],
                repeat_file_pairs=result["repeat_pairs"], repeat_arrays=result["repeat_arrays"],
                maximum_delta_error=captured["audit.json"]["maximum_delta_error"])


def v9_references(facts, claims):
    """V9 retained the value/supported schema, before v10's claimed/actual schema."""
    return [dict(metric=c["metric"], value=c["value"],
                 supported=c["metric"] in facts and c["value"] == facts[c["metric"]])
            for c in claims]


def run_one(study, catalog, model):
    with registered_view(study, catalog):
        if study == "v9":
            result = run_v9()
        else:
            suffix = "_v2" if study == "v10" else ""
            wrapper = importlib.import_module(f"scripts.verify_bounded_poc_{study}_release{suffix}")
            filename = ("bounded-poc-v10-qwen17b-release2-sha256.json" if study == "v10"
                        else f"{catalog['studies'][study]['folder']}-sha256.json")
            inventory = ROOT / "release" / filename
            result = wrapper.verify(model, inventory) if study == "v10" else wrapper.verify(inventory)
    return dict(experiment=catalog["studies"][study]["folder"], **result)
