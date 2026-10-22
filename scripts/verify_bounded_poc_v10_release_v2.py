"""Read-only v10 release verification; capture the frozen auditor's writes in memory."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

from src.poc.source_paths import model17_path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(model, inventory):
    from scripts import audit_bounded_poc_v10_qwen17b_restart2 as auditor

    record = json.loads(inventory.read_text())
    for field, name in (
        ("historical_inventory_sha256", "release/protected-sha256.json"),
        ("historical_adapter_inventory_sha256", "release/adapter-sha256.json"),
    ):
        assert record[field] == digest(ROOT / name), name

    def check_files():
        for name, expected in record["files"].items():
            assert digest(ROOT / name) == expected, f"Release bytes changed: {name}"

    check_files()
    captured = {}

    def capture(name, value):
        assert name not in captured, name
        captured[name] = json.loads(json.dumps(value, allow_nan=False))

    original_write = auditor.write
    try:
        auditor.write = capture
        with contextlib.redirect_stdout(io.StringIO()):
            auditor.audit(model)
    finally:
        auditor.write = original_write
    assert set(captured) == {"results.json", "facts.json", "audit.json"}
    for name, value in captured.items():
        assert value == json.loads((auditor.OUT / name).read_text()), name
    result = captured["results.json"]
    review = json.loads((auditor.OUT / "review-results.json").read_text())
    adjudication = json.loads((auditor.OUT / "review-adjudication.json").read_text())
    facts = captured["facts.json"]
    expected_claims = []
    for claim in review["numeric_claims"]:
        key = claim["metric"]
        expected_claims.append(
            dict(
                metric=key,
                claimed=claim["value"],
                actual=facts.get(key),
                supported=key in facts and claim["value"] == facts[key],
            )
        )
    assert adjudication["claims"] == expected_claims
    supported = bool(expected_claims) and all(c["supported"] for c in expected_claims)
    assert adjudication["all_numeric_claims_supported"] == supported
    check_files()
    return dict(
        verification_passed=True,
        scientific_passed=result["passed"],
        gates=result["gates"],
        release_files=len(record["files"]),
        historical_files=result["protected_files"],
        frozen_external_files=len(auditor.read("pre-calibration-manifest.json")["external_sha256"]),
        repeat_file_pairs=result["repeat_file_pairs"],
        repeat_arrays=result["repeat_arrays"],
        maximum_delta_error=result["maximum_delta_audit_error"],
        numeric_review_references=len(expected_claims),
        all_numeric_review_references_supported=supported,
        inventory_sha256=digest(inventory),
        mode="Read-only; all frozen auditor writes captured and compared in memory",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=model17_path())
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "release/bounded-poc-v10-qwen17b-release2-sha256.json"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.model, args.inventory), indent=2, allow_nan=False))
