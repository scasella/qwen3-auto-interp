import json

from scripts.verify_historical_release_v9 import historical_dataset_view


def test_historical_view_hides_only_unregistered_future_dataset_glob(tmp_path):
    old = tmp_path / "data/bounded-poc-v8-repair/dataset.json"
    new = tmp_path / "data/bounded-poc-v9-json-hop0/dataset.json"
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_text("{}")
    (tmp_path / "release").mkdir()
    (tmp_path / "release/protected-sha256.json").write_text(
        json.dumps({"files": {str(old.relative_to(tmp_path)): "frozen digest unchanged"}})
    )
    data = tmp_path / "data"
    assert set(data.glob("bounded-poc-*/dataset.json")) == {old, new}
    with historical_dataset_view(tmp_path):
        assert set(data.glob("bounded-poc-*/dataset.json")) == {old}
        assert set(data.glob("*/dataset.json")) == {old, new}
        assert new.read_text() == "{}"
    assert set(data.glob("bounded-poc-*/dataset.json")) == {old, new}
