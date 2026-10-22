"""Read-only campaign verification using the original standalone NumPy auditors.

Historical paths and versioned source reads resolve to hash-verified local bytes.
Auditor output writes are captured in memory, never committed to data/ or reports/.
This is an adapter for trusted local audit code, not a sandbox for arbitrary code.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import locale
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Bound BLAS concurrency before NumPy imports, for portable runtime/memory use.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDITS = (
    ("bounded-poc-v1", "audit_bounded_poc", "main"),
    ("lookup-followup-v1", "audit_lookup_followup", "main"),
    ("calibration-diagnosis-v1", "audit_calibration_diagnosis", "main"),
    ("bounded-poc-v2", "audit_bounded_poc_v2", "main"),
    ("bounded-poc-v3", "audit_bounded_poc_v3", "main"),
    ("bounded-poc-upstream", "audit_bounded_poc_upstream", "main"),
    ("bounded-poc-v4-confirm", "audit_bounded_poc_v4_confirm", "audit"),
    ("bounded-poc-v5-pointer", "audit_bounded_poc_v5_pointer", "audit"),
    ("bounded-poc-v6-final", "audit_bounded_poc_v6_final", "audit"),
    ("bounded-poc-v7-capstone", "audit_bounded_poc_v7_capstone", "audit"),
    ("bounded-poc-v8-repair", "audit_bounded_poc_v8_repair", "audit"),
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inventory(root=ROOT):
    inventory = json.loads((root / "release/protected-sha256.json").read_text())["files"]
    for name, expected in inventory.items():
        if digest(root / name) != expected:
            raise ValueError(f"Protected bytes changed: {name}")
    # Verify the original 1,416-entry inventory independently of the new release inventory.
    old = json.loads((root / "v8-protected.json").read_text())
    for item in old:
        if digest(root / item["copy"]) != item["sha256"]:
            raise ValueError(f"Original v8 inventory mismatch: {item['copy']}")
    for name, expected in json.loads((root / "release/adapter-sha256.json").read_text()).items():
        if digest(root / name) != expected:
            raise ValueError(f"Release verification dependency changed: {name}")
    return len(inventory), len(old)


class ReadOnlyAuditIO:
    """Translate only registered reads; hold attempted output writes in an overlay."""

    def __init__(self, root, sources, aliases, checkpoint=None):
        self.root = root
        self.sources = sources
        self.aliases = aliases
        self.checkpoint = checkpoint
        self.outputs = {}
        self.real_open = io.open

    def key(self, file):
        return str(Path(file).absolute())

    def read_path(self, file):
        key = self.key(file)
        path = self.root / self.aliases[key] if key in self.aliases else Path(key)
        try:
            name = str(path.relative_to(self.root))
        except ValueError as exc:
            raise FileNotFoundError(f"Unmapped external audit path: {path}") from exc
        if name in self.sources:
            return self.root / "release/historical-source" / self.sources[name]
        if name.startswith("data/hf-cache/") and self.checkpoint:
            return self.checkpoint / Path(name).name
        return path

    def open(self, file, mode="r", buffering=-1, encoding=None, errors=None,
             newline=None, closefd=True, opener=None):
        if isinstance(file, int):
            return self.real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
        if encoding == "locale":
            encoding = locale.getpreferredencoding(False)
        key = self.key(file)
        writing = any(c in mode for c in "wax+")
        if writing:
            path = Path(key)
            if not any(path.is_relative_to(self.root / d) for d in ("data", "reports")):
                raise PermissionError(f"Unexpected audit write: {path}")
            if not path.is_file():
                raise PermissionError(f"Unexpected new audit output: {path}")
            # All original auditors write text JSON/Markdown outputs.
            if "b" in mode or "+" in mode or "a" in mode:
                raise PermissionError(f"Unsupported audit write mode: {mode}")
            owner = self

            class Capture(io.StringIO):
                def close(self):
                    if not self.closed:
                        owner.outputs[key] = self.getvalue().encode(encoding or "utf-8")
                    super().close()

            return Capture()
        if key in self.outputs:
            value = self.outputs[key]
            return io.BytesIO(value) if "b" in mode else io.StringIO(value.decode(encoding or "utf-8"))
        return self.real_open(self.read_path(file), mode, buffering, encoding, errors,
                              newline, closefd, opener)

    @contextlib.contextmanager
    def active(self):
        with patch("builtins.open", self.open), patch("io.open", self.open):
            yield


def source_map(root, experiment, catalog):
    sources = {}
    for registration in catalog["registrations"].get(experiment, []):
        record = json.loads((root / registration).read_text())
        for name, expected in record.get("sha256", {}).items():
            if name.startswith(("data/", "reports/")):
                continue
            if name in sources and sources[name] != expected:
                raise ValueError(f"Conflicting source registration: {name}")
            blob = root / "release/historical-source" / expected
            if digest(blob) != expected:
                raise ValueError(f"Historical source blob mismatch: {name}")
            sources[name] = expected
    return sources


def nested_numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from nested_numbers(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from nested_numbers(child, f"{prefix}.{i}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def summary(root, experiment, captured):
    folder = root / "data" / experiment
    audit = json.loads(captured.get(str(folder / "audit.json"), (folder / "audit.json").read_bytes()))
    manifest = folder / "manifest.json"
    if not manifest.exists():
        manifest = folder / "plan-lock.json"
    # Original scripts perform their own exact-array assertions. Count the same named repeats.
    pairs = []
    for first in folder.glob("*.npz"):
        name = first.name
        other = None
        for a, b in (("confirmation-1", "confirmation-2"), ("confirm1", "confirm2")):
            if a in name:
                other = folder / name.replace(a, b, 1)
                break
        if other and other.exists():
            pairs.append((first, other))
        elif name.startswith("repeat-") and (folder / name.removeprefix("repeat-")).exists():
            pairs.append((folder / name.removeprefix("repeat-"), first))
    numeric_records = [audit]
    for filename, contents in captured.items():
        if filename.endswith(".json"):
            numeric_records.append(json.loads(contents))
    errors = [v for record in numeric_records for k, v in nested_numbers(record)
              if "maximum_independent_delta_error" in k or "maximum_delta_error" in k
              or "maximum_delta_audit_error" in k or "maximum_arithmetic_error" in k]
    return dict(experiment=experiment, manifest_sha256=digest(manifest),
                array_pairs=len(pairs), maximum_delta_error=max(errors) if errors else None,
                audit_passed=True)


def run_one(root, experiment, script, entry, catalog, checkpoint=None):
    module = importlib.import_module(f"scripts.{script}")
    sources = source_map(root, experiment, catalog)
    adapter = ReadOnlyAuditIO(root, sources, catalog["original_aliases"], checkpoint)
    # Redirect the old author's machine paths without changing audit source or comparisons.
    with contextlib.redirect_stdout(io.StringIO()), adapter.active():
        getattr(module, entry)()
    # Every regenerated numeric artifact must also agree with the preserved one.
    for filename, contents in adapter.outputs.items():
        if filename.endswith(".json"):
            regenerated = json.loads(contents)
            preserved = json.loads(Path(filename).read_bytes())
            # This historical auditor stamps wall-clock audit time on every invocation.
            # Only that metadata field is excluded; all scientific values remain exact.
            if experiment == "bounded-poc-upstream" and filename.endswith("/results.json"):
                regenerated.pop("audited_utc")
                preserved.pop("audited_utc")
            if regenerated != preserved:
                raise ValueError(f"Recomputed JSON differs from preserved artifact: {filename}")
    return summary(root, experiment, adapter.outputs)


def main():
    from src.poc.source_paths import model17_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Pinned HF snapshot directory, if outside data/hf-cache")
    parser.add_argument("--json", type=Path, help="Optional NEW summary path outside data/reports")
    parser.add_argument("--historical-only", action="store_true",
                        help="Run only the frozen v1–v8 audit population")
    parser.add_argument("--model17", type=Path, default=model17_path(),
                        help="Pinned Qwen3-1.7B checkpoint for v10")
    args = parser.parse_args()
    if args.json:
        target = args.json.absolute()
        if target.exists() or any(target.is_relative_to(ROOT / p) for p in ("data", "reports")):
            parser.error("--json must be a new file outside protected data/ and reports/")
    if not (ROOT / "release/protected-sha256.json").is_file():
        parser.error("BLOCKED: private evidence and provenance inventories are omitted from "
                     "this source-only repository; no scientific audit was run. "
                     "Use scripts/verify_source_only.py for package integrity.")
    start = time.monotonic()
    failures, results = [], []
    count, old_count = verify_inventory()
    catalog = json.loads((ROOT / "release/audit-paths.json").read_text())
    from scripts.verify_historical_release_v9 import historical_dataset_view
    print(f"Protected SHA-256: {count}/{count}; original v8 inventory: {old_count}/{old_count}")
    print("Experiment                       Pairs  Max NumPy delta  Audit  Manifest SHA-256")
    for experiment, script, entry in AUDITS:
        try:
            with historical_dataset_view(ROOT):
                result = run_one(ROOT, experiment, script, entry, catalog, args.checkpoint)
            results.append(result)
            error = result["maximum_delta_error"]
            text = "n/a" if error is None else f"{error:.9g}"
            print(f"{experiment:32} {result['array_pairs']:5}  {text:>15}  PASS   "
                  f"{result['manifest_sha256']}", flush=True)
        except Exception as exc:
            failures.append(dict(experiment=experiment, error=f"{type(exc).__name__}: {exc}"))
            print(f"{experiment:32} FAIL: {type(exc).__name__}: {exc}", flush=True)
    campaign_count = 0
    if not args.historical_only:
        from scripts import campaign_audit

        campaign = campaign_audit.load_catalog()
        campaign_count = campaign_audit.verify_references(campaign)
        print(f"V12 protected references: {campaign_count}/{campaign_count}", flush=True)
        for study in ("v9", "v10", "v11", "v12"):
            try:
                result = campaign_audit.run_one(study, campaign, args.model17)
                results.append(result)
                print(f"{result['experiment']:36} "
                      f"{result['repeat_file_pairs']:3} pairs / {result['repeat_arrays']:4} arrays  "
                      f"delta {result['maximum_delta_error']:.9g}  AUDIT PASS  "
                      f"scientific {'PASS' if result['scientific_passed'] else 'FAIL'}", flush=True)
            except Exception as exc:
                failures.append(dict(experiment=study, error=f"{type(exc).__name__}: {exc}"))
                print(f"{study}: AUDIT FAIL: {type(exc).__name__}: {exc}", flush=True)
        campaign_audit.verify_references(campaign)
    # Independent post-verification detects accidental writes by this adapter or any auditor.
    verify_inventory()
    elapsed = time.monotonic() - start
    report = dict(passed=not failures, protected_files=count, original_v8_files=old_count,
                  campaign_protected_references=campaign_count,
                  elapsed_seconds=elapsed, audits=results, failures=failures,
                  scope="CPU/NumPy integrity verification; scientific gate failures remain failures")
    if args.json:
        with args.json.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    total = len(AUDITS) + (0 if args.historical_only else 4)
    print(f"{len(results)}/{total} audits passed in {elapsed:.2f}s; "
          "protected bytes unchanged. Audit success does not mean scientific success.")
    if failures:
        print("Checkpoint missing? Download the exact revision listed in README.md, "
              "then use --checkpoint PATH. No GPU or model forward is needed.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
