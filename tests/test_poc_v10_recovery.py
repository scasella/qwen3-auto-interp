"""Recovery keeps scientific code fixed and excludes all previous key sets."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v10-qwen17b-restart1"


def test_recovery_changes_only_namespace_and_retirement():
    for kind in ("run", "audit"):
        original = (ROOT / f"scripts/{kind}_bounded_poc_v10_qwen17b.py").read_text()
        original = original.replace("bounded-poc-v10-qwen17b", "bounded-poc-v10-qwen17b-restart1")
        recovered = (ROOT / f"scripts/{kind}_bounded_poc_v10_qwen17b_restart1.py").read_text()
        trees = [ast.parse(source) for source in (original, recovered)]
        for tree in trees:
            tree.body = [
                node
                for node in tree.body
                if not (isinstance(node, ast.FunctionDef) and node.name == "retired_rows")
            ]
        assert ast.dump(trees[0]) == ast.dump(trees[1])


def test_recovery_freshness_and_independent_grammar():
    from scripts.audit_bounded_poc_v10_qwen17b_restart1 import validate_data
    from src.poc.v10_templates import validate

    data = json.loads((OUT / "dataset.json").read_text())
    validate_data(data)
    validate(data["rows"])
    retired = set(json.loads((OUT / "retired-keys.json").read_text())["keys"])
    previous = json.loads((ROOT / "data/bounded-poc-v10-qwen17b/dataset.json").read_text())
    assert set(previous["keys"]) <= retired
    assert not set(data["keys"]) & retired
    assert data["seed"] == previous["seed"] == 20261026
    assert len(data["keys"]) == 64 and len(data["rows"]) == 256


def test_registered_v9_source_restoration_is_exact():
    record = json.loads((ROOT / "incidents/v10-recovery-20261021/incident.json").read_text())
    assert len(record["restored_v9_sources"]) == 4
    for item in record["restored_v9_sources"]:
        assert hashlib.sha256((ROOT / item["restored_path"]).read_bytes()).hexdigest() == item["expected"]
