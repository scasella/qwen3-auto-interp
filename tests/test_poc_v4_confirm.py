import os

import numpy as np
import pytest

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.v4_analysis import effect, positive
from src.poc.v4_controls import planted, toy_rows
from src.poc.v4_harness import V4Harness, V4Probe, V4Spec


def test_planted_three_stage_and_negative_controls():
    assert planted()["passed"]


def test_no_rounding_and_group_dependence():
    rows = toy_rows()
    assert effect(rows, np.array([r["label"] for r in rows]), np.zeros(64))["sign_p"] == 1 / 256
    stat = dict(median=3.999999999, direction=1.0, sign_p=1 / 256)
    assert not positive(stat, 4.0)
    stat.update(median=4.0, direction=60 / 64)
    assert not positive(stat, 4.0)


@pytest.mark.parametrize("device", ["cpu", "mps"])
async def test_v4_q_k_v_masks_identity_mediation(tiny_harness, device):
    import torch

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in tiny-model Metal regression; no target checkpoint forward")
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = tiny_harness.model.to(device=device, dtype=dtype)
    wc = WorkerConfig(batch_size=4)
    h = V4Harness(
        model,
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        wc,
    )
    targets = ((1, 1), (2, 2))
    ids = [(2, 3, 4, 5, 6), (2, 3, 4, 7, 6), (3, 4, 5, 6, 7, 8), (3, 4, 5, 7, 7, 8)]

    def probes(mode="capture", payload=None, active=(), upstream=()):
        spec = V4Spec((0,), targets=targets, active=active, upstream=upstream, mode=mode)
        return [
            V4Probe("test", x, (2, 3), spec, payload=None if payload is None else payload[i])
            for i, x in enumerate(ids)
        ]

    async with MPSBatchWorker(h, wc) as w:
        base = await w.map(probes())

        def payload(self_patch=False):
            return [
                dict(
                    clean=r.routing,
                    donor=base[i if self_patch else i ^ 1].routing,
                    upstream={"0:0": base[i if self_patch else i ^ 1].heads["0:0"]},
                    context_end=3,
                    color_positions=[1, 2],
                )
                for i, r in enumerate(base)
            ]

        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            out = await w.map(probes(mode, payload(True), targets))
            np.testing.assert_array_equal([x.logits for x in out], [x.logits for x in base])
        for mode in ("k_context", "ov_context"):
            out = await w.map(probes(mode, payload(), targets))
            np.testing.assert_array_equal([x.logits for x in out], [x.logits for x in base])
        for mode in ("q_final", "k_color"):
            data = payload()
            out = await w.map(probes(mode, data, targets))
            for i, r in enumerate(out):
                for key in r.routing:
                    c, d = data[i]["clean"][key], data[i]["donor"][key]
                    q = np.array(d["q"] if mode == "q_final" else c["q"])
                    k = np.array(c["k"])
                    v = np.array(c["v"])
                    if mode == "k_color":
                        k[[1, 2]] = np.array(d["k"])[[1, 2]]
                    score = k @ q / (k.shape[1] ** 0.5)
                    pp = np.exp(score - score.max())
                    pp /= pp.sum()
                    np.testing.assert_allclose(r.routing[key]["routed_p"], pp, atol=2e-6)
                    expected = ((pp - np.array(c["p"]))[:, None] * v).sum(0)
                    np.testing.assert_allclose(r.routing[key]["delta"], expected, atol=2e-6)
                    assert r.routing[key]["sibling_error"] == 0
        mid = await w.map(probes("mediate", payload(), upstream=((0, 0),)))
        for i, r in enumerate(mid):
            for key in r.routing:
                np.testing.assert_array_equal(r.routing[key]["applied"], base[i].routing[key]["h"])
                np.testing.assert_array_equal(
                    r.routing[key]["k"][:3], base[i].routing[key]["k"][:3]
                )
    for layer in model.model.layers:
        assert not layer.self_attn._forward_pre_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks


def test_registered_dataset_and_invalid_donors():
    import copy
    import json
    from pathlib import Path

    from src.poc.v4_templates import parse, validate

    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "data/bounded-poc-v4-confirm/dataset.json").read_text())["rows"]
    validate(rows)
    bad = copy.deepcopy(rows)
    bad[0]["donors"]["B"] = bad[0]["id"]
    with pytest.raises(AssertionError):
        validate(bad)
    bad = copy.deepcopy(rows)
    bad[0]["target_position"] += 1
    with pytest.raises(AssertionError):
        validate(bad)
    with pytest.raises(ValueError):
        parse(rows[0]["text"].replace("Answer:", "Answer: red"))


def test_independent_numpy_grader_matches_planted():
    from scripts.audit_bounded_poc_v4_confirm import assert_same, recompute

    data = planted()
    for name in ("positive", "nuisance", "reversed", "vocabulary"):
        ev = data[name]["evidence"]
        def conv(a):
            return {k: np.asarray(v) for k, v in a.items()}
        actual = recompute(
            ev["rows"],
            conv(ev["base"]),
            {k: conv(v) for k, v in ev["paths"].items()},
            conv(ev["mediator"]),
            conv(ev["mediator"]),
        )
        for key in ("effects", "gates", "upstream_attention", "C_attention"):
            assert_same(actual[key], data[name][key])
