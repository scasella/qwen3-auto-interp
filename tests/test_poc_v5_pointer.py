import copy
import json
import os

import numpy as np
import pytest

from scripts.audit_bounded_poc_v5_pointer import audit_controls, recompute, validate_data
from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.v4_harness import V4Harness, V4Probe, V4Spec
from src.poc.v5_analysis import attention, effect, evaluate
from src.poc.v5_controls import planted, run_toy
from src.poc.v5_templates import build, parse, validate


def test_pointer_controls_and_independent_stage_recomputation():
    controls = planted()
    assert controls["passed"]
    audit_controls(controls)
    for kind, result in controls.items():
        if kind == "passed":
            continue
        ev = result["evidence"]

        def convert(x):
            return {k: np.asarray(v, dtype=np.float64) for k, v in x.items()}

        a = recompute(
            ev["rows"],
            convert(ev["base"]),
            {k: convert(v) for k, v in ev["paths"].items()},
            {k: convert(v) for k, v in ev["upstream"].items()},
        )
        assert a["criteria"] == result["criteria"]
        assert a["effects"] == result["effects"]
        assert a["attention"] == result["attention"]


def test_evaluator_rejects_erased_context_mediation_and_no_rounding():
    ev = run_toy()["evidence"]

    def convert(x):
        return {k: np.asarray(v, dtype=np.float64) for k, v in x.items()}

    base = convert(ev["base"])
    paths = {k: convert(v) for k, v in ev["paths"].items()}
    upstream = {k: convert(v) for k, v in ev["upstream"].items()}
    upstream["C"] = base
    r = evaluate(ev["rows"], base, paths, upstream)
    assert r["gates"]["upstream_B"] and not r["gates"]["upstream_C"]
    labels = np.array([r["label"] for r in ev["rows"]])
    paths["B_q_final"]["logits"] = base["logits"].copy()
    paths["B_q_final"]["logits"][:, 0] -= labels * 3.249999
    assert not evaluate(ev["rows"], base, paths, upstream)["criteria"]["B_q_final_effect"]
    assert effect(ev["rows"], np.ones(64))["sign_p"] == 1 / 256
    # Ties must fail direction, not count as weak support.
    assert effect(ev["rows"], np.zeros(64))["direction"] == 0


def test_float64_attention_and_separate_direction_statistics():
    rows = run_toy()["evidence"]["rows"]
    pc = np.zeros((64, 3, 2), dtype=np.float32)
    pd = pc.copy()
    for i, row in enumerate(rows):
        c = row["target_position"]
        pc[i, :, c] = 0.8
        pd[i, :, c] = 0.6
        pd[i, :, 1 - c] = [0.1, 0.2, 0.3]
    stat = attention(rows, pc, pd, "C")["coalition"]
    assert stat["gain"] == float(
        np.array([0.1, 0.2, 0.3], dtype=np.float32).astype(np.float64).mean()
    )
    assert stat["gain_direction"] == stat["loss_direction"] == stat["joint_direction"] == 1.0


def test_frozen_dataset_roles_invariants_and_tokenizer_only_reproduction():
    from transformers import AutoTokenizer

    from scripts.run_bounded_poc_v5_pointer import MODEL, ROOT

    data = json.loads((ROOT / "data/bounded-poc-v5-pointer/dataset.json").read_text())
    validate(data["rows"])
    validate_data(data)
    for row in data["rows"]:
        assert parse(row["text"])[0] == row["answer_color"]
        assert row["text"].endswith("Answer:")
    retired = [
        row
        for version in ("v1", "v2", "v3", "upstream", "v4-confirm")
        for row in json.loads((ROOT / f"data/bounded-poc-{version}/dataset.json").read_text())[
            "rows"
        ]
    ]
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    assert build(tok, retired) == data
    bad = copy.deepcopy(data["rows"])
    bad[0]["donors"]["C"] = bad[0]["donors"]["A"]
    with pytest.raises(AssertionError):
        validate(bad)
    with pytest.raises(ValueError):
        parse(data["rows"][0]["text"].replace("Code ", "Bare "))


@pytest.mark.parametrize("device", ["cpu", "mps"])
async def test_two_axis_pure_q_mediation_keeps_all_k_and_v_clean(tiny_harness, device):
    import torch

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in tiny-model MPS regression, not Qwen checkpoint inference")
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    wc = WorkerConfig(batch_size=4)
    h = V4Harness(
        tiny_harness.model.to(device=device, dtype=dtype),
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        wc,
    )
    targets = ((1, 1), (2, 2))
    # Same final question token. Swapped context-key positions with unchanged colors.
    ids = [(2, 3, 4, 5, 6, 7, 8), (4, 3, 2, 5, 6, 7, 8)]

    def probes(mode="capture", payload=None, active=(), upstream=()):
        spec = V4Spec((0,), targets=targets, mode=mode, active=active, upstream=upstream)
        return [
            V4Probe("toy", x, (2, 3), spec, payload=None if payload is None else payload[i])
            for i, x in enumerate(ids)
        ]

    async with MPSBatchWorker(h, wc) as w:
        base = await w.map(probes())
        payload = [
            dict(
                clean=r.routing,
                donor=base[i ^ 1].routing,
                upstream={"0:0": base[i ^ 1].heads["0:0"]},
                color_positions=[1, 3],
                context_end=5,
            )
            for i, r in enumerate(base)
        ]
        mid = await w.map(probes("mediate", payload, upstream=((0, 0),)))
        # Separate copies: donors and recipients in this toy list alias each
        # other's baseline dicts; deepcopy of the whole list preserves aliases.
        replay = [{key: copy.deepcopy(value) for key, value in item.items()} for item in payload]
        for i, item in enumerate(replay):
            for key in item["donor"]:
                item["donor"][key]["q"] = mid[i].routing[key]["q"]
                # Deliberately corrupt donor K/V; pure Q must ignore these entirely.
                item["donor"][key]["k"] = np.full_like(item["donor"][key]["k"], 17).tolist()
                item["donor"][key]["v"] = np.full_like(item["donor"][key]["v"], -9).tolist()
        out = await w.map(probes("q_final", replay, targets))
        for i, r in enumerate(out):
            for key, result in r.routing.items():
                clean = base[i].routing[key]
                q = np.asarray(mid[i].routing[key]["q"], dtype=np.float32)
                k = np.asarray(clean["k"], dtype=np.float32)
                v = np.asarray(clean["v"], dtype=np.float32)
                scores = (q * k).sum(-1) / np.sqrt(k.shape[1])
                p = np.exp(scores - scores.max())
                p /= p.sum()
                np.testing.assert_allclose(result["routed_p"], p, atol=2e-6)
                delta = ((p - np.asarray(clean["p"]))[:, None] * v).sum(0)
                np.testing.assert_allclose(result["delta"], delta, atol=2e-6)
                np.testing.assert_array_equal(mid[i].routing[key]["applied"], clean["h"])
                np.testing.assert_array_equal(mid[i].routing[key]["k"][:-1], clean["k"][:-1])
        assert all(not layer.self_attn.o_proj._forward_pre_hooks for layer in h.model.model.layers)


def test_canonical_review_numeric_reference_binding():
    x = float(np.float64(np.float32(0.031094551)))
    canonical = json.loads(json.dumps({"facts": {"metric": x}}, allow_nan=False))
    claim = {"metric": "metric", "value": canonical["facts"]["metric"]}
    assert claim["value"] == x
    # No decimal value needs to be transcribed by the reviewing model.
    assert json.loads(json.dumps(claim))["value"] == x
