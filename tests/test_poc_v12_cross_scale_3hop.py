import copy
import os

import numpy as np
import pytest

from src.poc.v12_controls import planted, toy
from src.poc.v12_cross_scale_3hop import CANDIDATES, evaluate, internal_delta, pattern


def evidence(kind="question_span"):
    t = toy(kind)
    return (
        t["rows"],
        {"logits": np.array(t["baseline"])},
        {k: {"logits": np.array(v)} for k, v in t["paths"].items()},
    )


def test_planted_spatial_circuits():
    assert planted()["passed"]


@pytest.mark.parametrize(
    "name,threshold",
    [("B_U_v_qspan", 1.2), ("C_U_v_qspan", 0.9), ("B_hop0_qspan", 0.4), ("C_hop0_qspan", 0.25)],
)
def test_exact_effect_boundary(name, threshold):
    rows, base, paths = evidence()
    base["logits"][:] = 0
    signs = np.array([r["label"] for r in rows])
    for v, expected in [(threshold, True), (np.nextafter(threshold, -np.inf), False)]:
        paths[name]["logits"][:, 0] = -signs * v
        paths[name]["logits"][:, 1] = 0
        assert evaluate(rows, base, paths)["criteria"][name + "_effect"] is expected


def test_frozen_coordinates_budget_and_identical_gates():
    from src.poc.v11_question_hop0 import LIMITS as OLD_LIMITS
    from src.poc.v11_question_hop0 import evaluate as old_evaluate
    from src.poc.v12_cross_scale_3hop import H0, LIMITS, ORIGINAL_U, D, U

    assert U == ((16, 4), (17, 1), (18, 7)) and ORIGINAL_U == ((17, 1), (16, 4), (15, 4))
    assert D == ((19, 5), (21, 13), (22, 9))
    assert H0 == ((13, 6), (12, 6), (13, 14), (14, 12), (11, 9), (13, 15))
    assert len(CANDIDATES) == 240 and LIMITS == OLD_LIMITS
    assert (3 + 26 + 13) * 4 == 168
    assert (2 + 2 * (2 + 3) + 3) * 4 == 60
    for kind in ("question_span", "context_kv_only", "final_token_only", "nuisance", "reversed"):
        assert evaluate(*evidence(kind)) == old_evaluate(*evidence(kind))


def test_spatial_delta_identity_and_partition():
    rng = np.random.default_rng(2)
    c = {
        k: rng.normal(size=s).astype("f")
        for k, s in [("q", (8,)), ("k", (7, 8)), ("v", (7, 8)), ("h", (8,))]
    }
    c["p"] = pattern(c["q"], c["k"])
    d = copy.deepcopy(c)
    d["v"] += 1
    for mode in ("full", "v_qspan", "kv_context", "q_final"):
        assert np.array_equal(internal_delta(c, c, 4, mode), np.zeros(8, "f"))
    actual = internal_delta(c, d, 4, "v_qspan")
    expected = np.sum(c["p"][4:, None] * (d["v"][4:] - c["v"][4:]), axis=0, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "mps"])
async def test_question_span_isolation_padding_and_self(tiny_harness, device):
    import torch

    from src.config import ModelConfig, WorkerConfig
    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.v4_harness import V4Probe, V4Spec
    from src.poc.v12_harness import SpanSpec, V12Harness

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in Metal")
    wc = WorkerConfig(batch_size=4)
    model = copy.deepcopy(tiny_harness.model).to(
        device=device, dtype=torch.bfloat16 if device == "mps" else torch.float32
    )
    h = V12Harness(
        model,
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        wc,
    )
    ids = [(2, 3, 4, 5, 6), (3, 2, 4, 5, 6), (2, 3, 4, 5, 6, 7), (3, 2, 4, 5, 6, 7)]
    targets = ((1, 0), (2, 1))
    async with MPSBatchWorker(h, wc) as w:
        ub = await w.map([V4Probe("", x, (2, 3), V4Spec((), targets=targets)) for x in ids])
        spec = SpanSpec((), targets=targets, capture=True, last_layer=2)
        baseline = await w.map(
            [V4Probe("", x, (2, 3), spec, payload={"query_position": 2}) for x in ids]
        )
        payload = []
        for i, r in enumerate(baseline):
            payload.append(
                dict(
                    query_position=2,
                    residual=r.spatial["residual"],
                    head_values={"0:0": r.spatial["hop0"][0]},
                    clean_v=[ub[i].routing[f"{layer_id}:{a}"]["v"] for layer_id, a in targets],
                )
            )
        spec = SpanSpec((), targets=targets, heads=((0, 0),), last_layer=2)
        same = await w.map([V4Probe("", x, (2, 3), spec, payload=p) for x, p in zip(ids, payload)])
        for a, b in zip(baseline, same):
            np.testing.assert_array_equal(a.spatial["vspan"], b.spatial["vspan"])
            assert np.all(b.spatial["errors"] == 0)
        for p in payload:
            p["head_values"]["0:0"] = p["head_values"]["0:0"] + 2
        changed = await w.map(
            [V4Probe("", x, (2, 3), spec, payload=p) for x, p in zip(ids, payload)]
        )
        assert any(
            np.any(a.spatial["vspan"] != b.spatial["vspan"]) for a, b in zip(baseline, changed)
        )
        assert all(np.all(r.spatial["errors"] == 0) for r in changed)


def test_independent_grader_controls_and_full_logit_identity():
    from scripts.audit_bounded_poc_v12_cross_scale_3hop import audit_controls, recompute

    audit_controls(planted())
    rows, base, paths = evidence()
    assert evaluate(rows, base, paths) == recompute(rows, base, paths)
    # Equal shifts cancel in the margin but must fail the full-logit zero gate.
    paths["B_U_kv_context"]["logits"] += 1
    assert not evaluate(rows, base, paths)["criteria"]["B_context_exact"]
    assert evaluate(rows, base, paths) == recompute(rows, base, paths)


def test_direction_group_sign_and_conservation_boundaries():
    from scripts.audit_bounded_poc_v12_cross_scale_3hop import recompute

    rows, base, paths = evidence()
    labels = np.array([r["label"] for r in rows])
    for positives, expected in [(58, True), (57, False)]:
        delta = np.ones(64) * 2
        delta[positives:] = -2
        paths["B_U_v_qspan"]["logits"] = base["logits"].copy()
        paths["B_U_v_qspan"]["logits"][:, 0] -= labels * delta
        assert evaluate(rows, base, paths)["criteria"]["B_U_v_qspan_direction"] is expected
        assert evaluate(rows, base, paths) == recompute(rows, base, paths)
    base["logits"][:] = 0
    paths["B_upstream_Q"]["logits"][:] = 0
    for value, expected in [(0.05, True), (np.nextafter(0.05, np.inf), False)]:
        paths["B_U_full"]["logits"][:] = 0
        paths["B_U_full"]["logits"][:, 0] = value
        assert evaluate(rows, base, paths)["criteria"]["B_conservation"] is expected
    from src.poc.v12_cross_scale_3hop import effect

    values = np.ones(64)
    values[:8] = -1
    assert effect(rows, values)["sign_p"] > 0.01
    assert effect(rows, np.ones(64))["sign_p"] == 1 / 256


@pytest.mark.parametrize("device", ["cpu", "mps"])
async def test_complete_three_stage_numpy_audit(tiny_harness, tmp_path, monkeypatch, device):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from scripts import run_bounded_poc_v12_cross_scale_3hop as runner
    from scripts.audit_bounded_poc_v12_cross_scale_3hop import audit_baseline, audit_path
    from src.config import ModelConfig, WorkerConfig
    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.v12_harness import V12Harness

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in Metal")
    # BF16 anchors also on CPU make the applied-vector audit meaningful on both devices.
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=8,
        max_position_embeddings=128,
    )
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16, device=device)
    wc = WorkerConfig(batch_size=4)
    harness = V12Harness(model, tiny_harness.tokenizer, ModelConfig(device=device), wc)
    ids = [(2, 3, 4, 5, 6), (3, 2, 4, 5, 6), (2, 3, 4, 5, 6, 7), (3, 2, 4, 5, 6, 7)]
    rows = [
        dict(
            text="",
            token_ids=x,
            query_position=2,
            context_end=2,
            id=str(i),
            donors={"B": str(i ^ 1), "C": str(i ^ 1)},
            color_positions=[0, 1],
        )
        for i, x in enumerate(ids)
    ]
    monkeypatch.setattr(runner, "OUT", tmp_path)
    async with MPSBatchWorker(harness, wc) as worker:
        base, ub, sb = await runner.baselines(worker, rows, (2, 3), "test")
        audit_baseline(rows, base, ub, sb)
        from scripts.run_bounded_poc_v10_qwen17b_restart2 import execute

        du = await execute(worker, rows, (2, 3), runner.ORIGINAL_U, full=True)
        ds = await runner.spatial(worker, rows, (2, 3), upstream=runner.ORIGINAL_U)
        audit_baseline(rows, base, du, ds)
        for mode in ("v_qspan", "hop0_qspan"):
            for self_patch in (True, False):
                a = await runner.path(
                    worker,
                    rows,
                    (2, 3),
                    base,
                    du,
                    ds,
                    "C",
                    mode,
                    runner.H0,
                    self_patch,
                    runner.ORIGINAL_U,
                )
                assert audit_path(rows, base, du, ds, a, "C", mode, runner.H0, self_patch) == 0
        for mode in (*runner.MODES, "upstream_Q", "hop0_qspan"):
            for self_patch in (True, False):
                a = await runner.path(
                    worker, rows, (2, 3), base, ub, sb, "C", mode, runner.H0, self_patch
                )
                assert audit_path(rows, base, ub, sb, a, "C", mode, runner.H0, self_patch) == 0
                if mode == "hop0_qspan" and not self_patch:
                    bad = {k: v.copy() for k, v in a.items()}
                    bad["u_delta"][0, 0, 0] += 1
                    with pytest.raises(AssertionError):
                        audit_path(rows, base, ub, sb, bad, "C", mode, runner.H0)
