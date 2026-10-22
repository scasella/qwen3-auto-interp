import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_bounded_poc_v6_final import audit_controls, recompute, validate_data
from src.poc.v6_analysis import LIMITS, evaluate
from src.poc.v6_controls import planted, run_toy
from src.poc.v6_templates import build, parse, validate


def evidence():
    e = run_toy()["evidence"]

    def convert(x):
        return {k: np.asarray(v, dtype=np.float64) for k, v in x.items()}

    return (
        e["rows"],
        convert(e["base"]),
        {k: convert(v) for k, v in e["paths"].items()},
        {k: convert(v) for k, v in e["upstream"].items()},
    )


def test_planted_controls_and_exact_forty_criteria():
    controls = planted()
    assert controls["passed"]
    audit_controls(controls)
    for kind, result in controls.items():
        if kind != "passed":
            assert len(result["criteria"]) == 40
    c = controls["literal_query_only_control"]
    assert c["gates"]["upstream_B"] and not c["gates"]["upstream_C"]
    assert c["effects"]["C_upstream_Q"]["median"] == 0
    assert not controls["direct_k_color_only_control"]["gates"]["C"]
    assert not controls["direct_k_color_only_control"]["gates"]["upstream_C"]


@pytest.mark.parametrize(
    "name,threshold",
    [
        ("A_ov_color", 2.25),
        ("B_q_final", 2.0),
        ("C_q_final", 1.5),
        ("B_upstream_Q", 1.4),
        ("C_upstream_Q", 1.0),
    ],
)
def test_effect_thresholds_do_not_round_up(name, threshold):
    rows, base, paths, upstream = evidence()
    labels = np.array([r["label"] for r in rows])
    base["logits"][:] = 0
    target = upstream[name[0]] if "upstream" in name else paths[name]
    target["logits"][:] = 0
    for value, passes in [(threshold, True), (np.nextafter(threshold, -np.inf), False)]:
        target["logits"][:, 0] = -labels * value
        prod = evaluate(rows, base, paths, upstream)
        audit = recompute(rows, base, paths, upstream)
        assert prod == audit
        assert prod["criteria"][name + "_effect"] == passes


def test_signed_value_invariance_has_both_bounds():
    rows, base, paths, upstream = evidence()
    labels = np.array([r["label"] for r in rows])
    base["logits"][:] = 0
    target = paths["C_ov_color"]
    target["logits"][:] = 0
    for x, passes in [(-0.25, True), (0.25, True), (-0.250001, False), (0.250001, False)]:
        target["logits"][:, 0] = -labels * x
        assert evaluate(rows, base, paths, upstream)["criteria"]["C_ov_signed"] == passes
        assert recompute(rows, base, paths, upstream)["criteria"]["C_ov_signed"] == passes


def test_frozen_numeric_limits_match_requested_envelope():
    assert LIMITS["accuracy"] == 0.85
    assert LIMITS["A_direction"] == 0.78
    assert LIMITS["C_dominance_ratio"] == 2.5
    assert LIMITS["C_dominance_direction"] == 0.75
    assert (LIMITS["C_shift"], LIMITS["C_shift_direction"]) == (0.08, 0.90)
    assert (LIMITS["UB_shift"], LIMITS["UB_first_shift"], LIMITS["UB_joint_direction"]) == (
        0.08,
        0.11,
        0.90,
    )
    assert (LIMITS["UC_shift"], LIMITS["UC_first_shift"], LIMITS["UC_shift_direction"]) == (
        0.06,
        0.09,
        0.90,
    )
    assert (LIMITS["UB_direction"], LIMITS["UC_direction"]) == (0.95, 0.95)


def test_all_inline_tokens_donors_and_disjoint_keys_reproduce_without_model():
    from transformers import AutoTokenizer

    from scripts.run_bounded_poc_v6_final import MODEL, ROOT

    data = json.loads((ROOT / "data/bounded-poc-v6-final/dataset.json").read_text())
    validate(data["rows"])
    validate_data(data)
    for row in data["rows"]:
        assert "\n" not in row["text"] and "\r" not in row["text"]
        assert row["text"].count("code ") == 3
        assert row["text"].endswith("Answer:")
    retired = [
        row
        for v in ("v1", "v2", "v3", "upstream", "v4-confirm", "v5-pointer")
        for row in json.loads((ROOT / f"data/bounded-poc-{v}/dataset.json").read_text())["rows"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    assert build(tokenizer, retired) == data
    assert data["seed"] == 20261022
    bad = copy.deepcopy(data["rows"])
    bad[0]["donors"]["C"] = bad[0]["donors"]["A"]
    with pytest.raises(AssertionError):
        validate(bad)
    for text in (
        data["rows"][0]["text"].replace("Record 2:", "\nRecord 2:"),
        data["rows"][0]["text"].replace("code ", "tag ", 1),
    ):
        with pytest.raises(ValueError):
            parse(text)


def test_v4_layout_claim_is_not_assumed_true():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "data/bounded-poc-v4-confirm/dataset.json").read_text())
    assert all("\n" in r["text"] for r in data["rows"])
