"""The release adapter must fail closed without touching preserved evidence."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_all_audits import ReadOnlyAuditIO, source_map, verify_inventory


def test_original_paths_and_frozen_sources_resolve_without_author_worktree(tmp_path):
    data = tmp_path / "data/run/raw.json"
    data.parent.mkdir(parents=True)
    data.write_text('{"a": 1}')
    blob = tmp_path / "release/historical-source/abc"
    blob.parent.mkdir(parents=True)
    blob.write_text("old source\n")
    source = tmp_path / "src/tool.py"
    source.parent.mkdir()
    source.write_text("new source\n")
    adapter = ReadOnlyAuditIO(tmp_path, {"src/tool.py": "abc"},
                             {"/absent/old/worktree/data/run/raw.json": "data/run/raw.json"})
    with adapter.active():
        assert Path("/absent/old/worktree/data/run/raw.json").read_text() == '{"a": 1}'
        assert source.read_text() == "old source\n"
        with pytest.raises(FileNotFoundError, match="Unmapped external"):
            Path("/absent/unregistered/file").read_text()
        with data.open("x") as stream:
            json.dump({"a": 2}, stream)
        assert json.loads(data.read_text()) == {"a": 2}
    assert data.read_text() == '{"a": 1}'
    assert source.read_text() == "new source\n"


def test_writes_fail_closed_and_context_restores_after_failure(tmp_path):
    data = tmp_path / "data/old.txt"
    data.parent.mkdir()
    data.write_text("protected")
    adapter = ReadOnlyAuditIO(tmp_path, {}, {})
    with pytest.raises(RuntimeError), adapter.active():
        data.write_text("temporary")
        raise RuntimeError("audit fails")
    assert data.read_text() == "protected"
    with adapter.active():
        for target in (tmp_path / "data/new.txt", tmp_path / "source.py"):
            with pytest.raises(PermissionError):
                target.write_text("forbidden")
        with pytest.raises(PermissionError):
            data.write_bytes(b"forbidden")


def test_hash_mismatch_cannot_be_hidden_by_path_translation(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    expected = hashlib.sha256(b"original").hexdigest()
    (tmp_path / "evidence").write_bytes(b"tampered")
    (release / "protected-sha256.json").write_text(json.dumps({"files": {"evidence": expected}}))
    with pytest.raises(ValueError, match="Protected bytes changed"):
        verify_inventory(tmp_path)


def test_source_archive_requires_real_matching_bytes(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {"src/x.py": "abc"}}))
    folder = tmp_path / "release/historical-source"
    folder.mkdir(parents=True)
    (folder / "abc").write_text("unverified")
    with pytest.raises(ValueError, match="Historical source blob mismatch"):
        source_map(tmp_path, "x", {"registrations": {"x": ["manifest.json"]}})
