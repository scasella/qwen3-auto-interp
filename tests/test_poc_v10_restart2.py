"""Second recovery changes arithmetic backend without changing scientific gates."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_second_recovery_preserves_runner_and_auditor_science():
    for kind in ("run", "audit"):
        original = (ROOT / f"scripts/{kind}_bounded_poc_v10_qwen17b.py").read_text()
        original = original.replace("bounded-poc-v10-qwen17b", "bounded-poc-v10-qwen17b-restart2")
        original = original.replace("from src.poc.v10_harness import", "from src.poc.v10_stable_harness import")
        if kind == "run":
            original = original.replace("from src.poc.v10_stable_harness import V10Harness\n", "")
            original = original.replace("from src.poc.v10_templates import", "from src.poc.v10_stable_harness import V10Harness\nfrom src.poc.v10_templates import")
        actual = (ROOT / f"scripts/{kind}_bounded_poc_v10_qwen17b_restart2.py").read_text()
        trees = [ast.parse(source) for source in (original, actual)]
        for tree in trees:
            tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == "retired_rows")]
        assert ast.dump(trees[0]) == ast.dump(trees[1])


def test_second_recovery_retires_all_previous_attempts():
    from scripts.audit_bounded_poc_v10_qwen17b_restart2 import validate_data

    p = ROOT / "data/bounded-poc-v10-qwen17b-restart2"
    data = json.loads((p / "dataset.json").read_text())
    retired = set(json.loads((p / "retired-keys.json").read_text())["keys"])
    for name in ("bounded-poc-v10-qwen17b", "bounded-poc-v10-qwen17b-restart1"):
        previous = json.loads((ROOT / "data" / name / "dataset.json").read_text())
        assert set(previous["keys"]) <= retired
    assert not set(data["keys"]) & retired
    assert data["seed"] == 20261026
    validate_data(data)
