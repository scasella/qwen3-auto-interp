"""Consolidation must preserve actual historical bytes and failed verdicts."""

import copy
import json
from pathlib import Path

import pytest

from scripts import campaign_audit as campaign


def test_git_registration_proofs_bind_blobs_and_detect_tampering():
    proofs = json.loads((campaign.RECORDS / "git-proofs.json").read_text())
    assert len(proofs) == 7
    for commit, proof in proofs.items():
        blob, timestamp = campaign.git_registration(proof, commit)
        assert blob == (campaign.ROOT / proof["path"]).read_bytes()
        assert int(timestamp) > 0
        altered = copy.deepcopy(proof)
        altered["objects"][commit]["data"] = "YQ=="
        with pytest.raises(AssertionError, match="Git object digest mismatch"):
            campaign.git_registration(altered, commit)


def test_registered_view_preserves_source_versions_and_rejects_writes(tmp_path):
    catalog = campaign.load_catalog()
    source = campaign.ROOT / "scripts/verify_all_audits.py"
    current = source.read_bytes()
    expected = catalog["studies"]["v10"]["sources"][str(source.relative_to(campaign.ROOT))]
    with campaign.registered_view("v10", catalog):
        assert source.read_bytes() == (campaign.ROOT / expected["path"]).read_bytes()
        assert source.read_bytes() != current
        with pytest.raises(PermissionError, match="Uncaptured audit write"):
            (tmp_path / "forbidden.json").write_text("{}")
    assert source.read_bytes() == current
    assert not (tmp_path / "forbidden.json").exists()


def test_reference_verifier_does_not_treat_missing_as_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    protected = tmp_path / "data/bounded-poc-v12-cross-scale-3hop/protected.json"
    protected.parent.mkdir(parents=True)
    protected.write_text(json.dumps({"files": {"old": "0" * 64}}))
    catalog = dict(aliases={"old": {"path": "missing", "sha256": "0" * 64}},
                   studies={}, protected_references=1)
    with pytest.raises(FileNotFoundError):
        campaign.verify_references(catalog)


def test_v9_reference_schema_preserves_unsupported_null():
    assert campaign.v9_references({"observed": 1.25}, [
        {"metric": "observed", "value": 1.25},
        {"metric": "missing", "value": None},
    ]) == [dict(metric="observed", value=1.25, supported=True),
           dict(metric="missing", value=None, supported=False)]


def test_v9_dataset_scope_is_registered_and_restores_glob():
    catalog = campaign.load_catalog()
    data = campaign.ROOT / "data"
    before = set(data.rglob("dataset.json"))
    with campaign.registered_view("v9", catalog):
        names = {str(p.relative_to(campaign.ROOT)) for p in data.rglob("dataset.json")}
        assert names == set(catalog["studies"]["v9"]["datasets"])
        assert all("v12" not in str(Path(p)) for p in names)
    assert set(data.rglob("dataset.json")) == before
