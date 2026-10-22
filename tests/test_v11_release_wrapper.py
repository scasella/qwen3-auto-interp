"""Publication verification must neither rewrite evidence nor hide unsupported references."""

import json

import pytest

from scripts import audit_bounded_poc_v11_question_hop0 as auditor
from scripts import verify_bounded_poc_v11_release as verifier


@pytest.mark.parametrize("supported", [True, False])
def test_read_only_wrapper(tmp_path, monkeypatch, supported):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(auditor, "OUT", tmp_path / "data")
    auditor.OUT.mkdir()
    results = dict(
        passed=False,
        gates={"gate3": False, "gate4": True},
        protected_files=1,
        repeat_file_pairs=1,
        repeat_arrays=1,
        maximum_delta_error=0.0,
    )
    facts = {"observed": 1}
    metric = "observed" if supported else "unsupported"
    claim = facts.get(metric)
    values = {"results.json": results, "facts.json": facts, "audit.json": {"passed": True}}
    extras = {
        "review-payload.json": {"facts": facts},
        "review-results.json": {"numeric_claims": [{"metric": metric, "value": claim}]},
        "review-adjudication.json": {
            "claims": [dict(metric=metric, claimed=claim, actual=claim, supported=supported)],
            "all_numeric_claims_supported": supported,
        },
    }
    for name, value in {**values, **extras}.items():
        (auditor.OUT / name).write_text(json.dumps(value))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"files": {"data/results.json": verifier.digest(auditor.OUT / "results.json")}})
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def write_only_audit():
        for name, value in values.items():
            auditor.write(name, value)

    monkeypatch.setattr(auditor, "audit", write_only_audit)
    result = verifier.verify(inventory)
    assert result["verification_passed"]
    assert not result["scientific_passed"]
    assert result["all_review_references_supported"] is supported
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_corrected_reference_attempt_keeps_original(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(auditor, "OUT", tmp_path / "data")
    auditor.OUT.mkdir()
    second = auditor.OUT / "review-attempt2"
    second.mkdir()
    facts = {"x": 1}
    result = dict(
        passed=False,
        gates={"gate3": False},
        protected_files=1,
        repeat_file_pairs=1,
        repeat_arrays=1,
        maximum_delta_error=0.0,
    )
    values = {"results.json": result, "facts.json": facts, "audit.json": {"passed": True}}
    for name, value in values.items():
        (auditor.OUT / name).write_text(json.dumps(value))
    for directory, metric, value, supported in [
        (auditor.OUT, "facts.x", None, False),
        (second, "x", 1, True),
    ]:
        (directory / "review-payload.json").write_text(json.dumps({"facts": facts}))
        (directory / "review-results.json").write_text(
            json.dumps({"numeric_claims": [{"metric": metric, "value": value}]})
        )
        (directory / "review-adjudication.json").write_text(
            json.dumps(
                {
                    "claims": [
                        dict(metric=metric, claimed=value, actual=value, supported=supported)
                    ],
                    "all_numeric_claims_supported": supported,
                }
            )
        )
    (auditor.OUT / "final-review.json").write_text(
        json.dumps(
            {
                "directory": "review-attempt2",
                "facts_sha256": verifier.digest(auditor.OUT / "facts.json"),
            }
        )
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"files": {"data/results.json": verifier.digest(auditor.OUT / "results.json")}})
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def audit():
        for name, value in values.items():
            auditor.write(name, value)

    monkeypatch.setattr(auditor, "audit", audit)
    check = verifier.verify(inventory)
    assert check["all_review_references_supported"]
    assert not check["review_attempts"][0]["all_supported"]
    assert len(check["review_attempts"]) == 2
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
