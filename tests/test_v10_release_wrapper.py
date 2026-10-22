"""Frozen auditors communicate by writes and need not return their result."""

import json

from scripts import audit_bounded_poc_v10_qwen17b_restart2 as auditor
from scripts import verify_bounded_poc_v10_release_v2 as verifier


def test_read_only_wrapper_accepts_no_return_auditor(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(auditor, "OUT", tmp_path / "data")
    auditor.OUT.mkdir()
    (tmp_path / "release").mkdir()
    results = dict(passed=True, gates={"gate4": True}, protected_files=1, repeat_file_pairs=1, repeat_arrays=1, maximum_delta_audit_error=0.0)
    values = {"results.json": results, "facts.json": {"x": 1}, "audit.json": {"passed": True}}
    extra = {
        "pre-calibration-manifest.json": {"external_sha256": {}},
        "review-results.json": {"numeric_claims": [{"metric": "x", "value": 1}]},
        "review-adjudication.json": {"claims": [{"metric": "x", "claimed": 1, "actual": 1, "supported": True}], "all_numeric_claims_supported": True},
    }
    for name, value in {**values, **extra}.items():
        (auditor.OUT / name).write_text(json.dumps(value))
    for name in ("protected-sha256.json", "adapter-sha256.json"):
        (tmp_path / "release" / name).write_text("{}")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "historical_inventory_sha256": verifier.digest(tmp_path / "release/protected-sha256.json"),
        "historical_adapter_inventory_sha256": verifier.digest(tmp_path / "release/adapter-sha256.json"),
        "files": {"data/results.json": verifier.digest(auditor.OUT / "results.json")},
    }))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def write_only_audit(model):
        for name, value in values.items():
            auditor.write(name, value)
        # No return value, matching the frozen scientific auditor.

    monkeypatch.setattr(auditor, "audit", write_only_audit)
    result = verifier.verify(tmp_path / "unused-model", inventory)
    assert result["verification_passed"] and result["scientific_passed"]
    assert result["numeric_review_references"] == 1
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
