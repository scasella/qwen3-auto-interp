import copy
import json
import os
from dataclasses import fields

import numpy as np
import pytest

from scripts.audit_bounded_poc_v8_repair import (
    adjudicate_A,
    adjudicate_B,
    arithmetic,
    audit_controls,
    validate_data,
)
from scripts.run_bounded_poc_v8_repair import MODEL, ROOT, constants, pack
from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.v8_controls import planted
from src.poc.v8_harness import RepairHarness, RepairProbe, RepairSpec
from src.poc.v8_repair import (
    A_GRID,
    B_GRID,
    IDENTITY,
    build,
    fit_probe,
    grade_A,
    grade_B,
    select,
    stats,
    validate,
)


def test_planted_models_and_independent_stage_audit():
    c = planted()
    assert c["passed"]
    audit_controls(c)
    assert c["A_positive"]["mean_replace_fixed"] == 0
    assert c["A_positive"]["result"]["stats"]["accuracy"] == 1
    assert c["B_positive"]["result"]["stats"]["accuracy"] == 1
    assert (
        c["B_positive"]["result"]["naive"]["fixed"] < c["B_positive"]["result"]["naive"]["broken"]
    )


def test_dataset_tokenizer_reproduction_and_no_retired_keys():
    from transformers import AutoTokenizer

    data = json.loads((ROOT / "data/bounded-poc-v8-repair/dataset.json").read_text())
    retired = [
        r
        for p in (ROOT / "data").glob("bounded-poc-*/dataset.json")
        if p.parent.name != "bounded-poc-v8-repair"
        for r in json.loads(p.read_text())["rows"]
    ]
    retired += json.loads((ROOT / "data/lookup-followup-v1/dataset.json").read_text())["rows"]
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    assert build(tok, retired) == data
    validate(data["rows"])
    validate_data(data)
    for row in data["rows"]:
        ids = tok(row["text"], add_special_tokens=False, return_offsets_mapping=True)
        assert ids["input_ids"] == row["token_ids"]
        for pos, key in zip(row["key_positions"], row["ordered_keys"]):
            a, b = ids["offset_mapping"][pos]
            assert row["text"][a:b].strip() == key
    bad = copy.deepcopy(data["rows"])
    bad[0]["opposite_id"] = bad[0]["id"]
    with pytest.raises(AssertionError):
        validate(bad)


def test_group_sign_and_empty_error_subsets_fail():
    c = planted()
    e = c["A_positive"]["evidence"]
    rows = e["rows"]
    base = np.asarray(e["repaired"])
    result = grade_A(rows, base, base, e["probe"])
    assert result["probe_error_accuracy"] is None and not result["A1"] and not result["A2"]
    b = c["B_positive"]["evidence"]
    clean = np.asarray(b["repaired"])
    result = grade_B(b["rows"], clean, clean, clean)
    assert not result["criteria"]["fewer_regressions"] and result["stats"]["recency_sign_p"] == 1
    assert not result["passed"]


def test_selection_tie_breaks_and_frozen_grid():
    assert len(A_GRID) == 12 and len(B_GRID) == 24
    e = planted()["A_positive"]["evidence"]
    rows = e["rows"]
    clean = np.asarray(e["base"])
    fixed = np.asarray(e["repaired"])
    i, _ = select(rows, clean, [fixed, fixed, clean])
    assert i == 0
    z = stats(rows, clean, fixed)
    assert z["fixed"] > z["broken"]
    assert grade_A(rows, clean, fixed, e["probe"]) == adjudicate_A(rows, clean, fixed, e["probe"])
    b = planted()["B_positive"]["evidence"]
    assert grade_B(b["rows"], b["base"], b["repaired"], b["naive"]) == adjudicate_B(
        b["rows"], b["base"], b["repaired"], b["naive"]
    )


def test_probe_leaves_entire_groups_out():
    e = planted()["A_positive"]["evidence"]
    rows = e["rows"]
    h = np.asarray(e["h"])
    y = np.array([r["label"] for r in rows])
    w, b, lgo = fit_probe(rows, h)
    assert np.all(y * lgo > 0)
    assert np.all(y * (h.reshape(len(h), -1) @ w + b) > 0)
    # Change one group's labels: its own held-out predictions cannot change.
    altered = copy.deepcopy(rows)
    for row in altered:
        if row["group"] == "0":
            row["label"] *= -1
    _, _, other = fit_probe(altered, h)
    ix = [i for i, r in enumerate(rows) if r["group"] == "0"]
    np.testing.assert_array_equal(other[ix], lgo[ix])


def test_repair_contract_excludes_labels_and_donors():
    names = {f.name for f in fields(RepairProbe)}
    assert not names & {"label", "answer_color", "query_record", "opposite_id"}
    # Inherited donor is never populated by the v8 boundary.
    assert {"color_positions", "key_positions", "token_ids"} <= names


@pytest.mark.parametrize("device", ["cpu", "mps"])
async def test_single_pass_repairs_identity_and_numpy_arithmetic(tiny_harness, device, monkeypatch):
    import torch
    from transformers import Qwen3ForCausalLM

    import scripts.run_bounded_poc_v8_repair as runner

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in tiny Metal model")
    cfg = copy.deepcopy(tiny_harness.model.config)
    cfg.num_hidden_layers = 4
    cfg.layer_types = ["full_attention"] * 4
    torch.manual_seed(41)
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    targets = ((0, 0), (1, 1), (2, 2), (3, 3))
    monkeypatch.setattr(runner, "TARGETS", targets)
    wc = WorkerConfig(batch_size=4)
    h = RepairHarness(
        model, tiny_harness.tokenizer, ModelConfig(device=device, dtype="bfloat16"), wc
    )
    tokens = [(2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 7, 7), (2, 3, 4, 5, 8, 9, 7), (2, 3, 4, 5, 9, 9, 7)]
    rows = [
        dict(token_ids=list(t), color_positions=[0, 1, 2, 3], key_positions=[0, 1, 2, 3])
        for t in tokens
    ]

    def probes(rule):
        spec = RepairSpec(
            (),
            targets=targets,
            mode=rule["mode"],
            alpha=rule["alpha"],
            beta=rule["beta"],
            alpha22=rule["alpha22"],
        )
        return [
            RepairProbe(
                "", t, (2, 3), spec, color_positions=(0, 1, 2, 3), key_positions=(0, 1, 2, 3)
            )
            for t in tokens
        ]

    async with MPSBatchWorker(h, wc) as worker:
        base = pack(await worker.map(probes(IDENTITY)))
        con = constants(rows, base)
        assert con["mean_s"].shape == (3, 4)
        for head in range(3):
            np.testing.assert_array_equal(
                con["mean_p"][head], base["p"][:, head, :4].mean(0, dtype=float).astype(np.float32)
            )
        h.constants = con
        ident = pack(await worker.map(probes(IDENTITY)))
        np.testing.assert_array_equal(base["logits"], ident["logits"])
        for rule in [
            dict(IDENTITY, alpha=2),
            dict(IDENTITY, mode="score", beta=1.25, alpha22=1.5),
            dict(IDENTITY, mode="probability", beta=1.0, alpha22=2.0),
        ]:
            a = pack(await worker.map(probes(rule)))
            assert arithmetic(rows, a, con, rule) <= 0.00003
            assert np.any(a["delta"][:, :3] != 0) and np.all(a["delta"][:, 3] == 0)
        bad = copy.deepcopy(a)
        bad["delta"][0, 0, 0] += 0.001
        with pytest.raises(AssertionError):
            arithmetic(rows, bad, con, rule)
    for layer in model.model.layers:
        assert (
            not layer.self_attn._forward_pre_hooks and not layer.self_attn.o_proj._forward_pre_hooks
        )


def test_margin_boundary_no_rounding_and_ties_not_recency_evidence():
    e = planted()["A_positive"]["evidence"]
    rows = e["rows"]
    y = np.array([r["label"] for r in rows])
    base = np.zeros((len(rows), 2))
    for gain, passes in ((0.20, True), (np.nextafter(0.20, -np.inf), False)):
        repaired = base.copy()
        repaired[:, 0] = y * gain
        result = grade_A(rows, base, repaired, y)
        assert result == adjudicate_A(rows, base, repaired, y)
        assert result["criteria"]["margin_gain"] == passes
    e = planted()["B_positive"]["evidence"]
    zeros = np.zeros((len(e["rows"]), 2))
    result = grade_B(e["rows"], zeros, zeros, zeros)
    assert result["stats"]["baseline_errors"] == len(zeros)
    assert result["stats"]["baseline_recency_errors"] == 0
    assert not result["passed"]
