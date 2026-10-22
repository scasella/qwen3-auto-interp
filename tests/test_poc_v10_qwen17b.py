import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_bounded_poc_v10_qwen17b import EXPECTED_LIMITS, audit_controls, recompute
from src.poc.v10_controls import planted, run_toy
from src.poc.v10_qwen17b import (
    DOWNSTREAM_CANDIDATES,
    LIMITS,
    evaluate,
    select_extension,
    upstream_candidates,
)
from src.poc.v10_templates import validate


def evidence(kind="planted_pointer_model"):
    e = run_toy(kind)["evidence"]

    def convert(a):
        return {k: np.asarray(v, dtype=np.float64) for k, v in a.items()}

    return (
        e["rows"],
        convert(e["base"]),
        {k: convert(v) for k, v in e["paths"].items()},
        {k: convert(v) for k, v in e["upstream"].items()},
    )


def test_controls_and_independent_grader():
    controls = planted()
    assert controls["passed"]
    assert LIMITS == EXPECTED_LIMITS
    audit_controls(controls)
    for kind in (
        "planted_pointer_model",
        "literal_query_only_control",
        "direct_k_color_only_control",
        "nuisance",
        "reversed",
    ):
        args = evidence(kind)
        assert evaluate(*args) == recompute(*args)


@pytest.mark.parametrize(
    "name,threshold",
    [
        ("A_ov_color", 1.5),
        ("B_q_final", 1.5),
        ("C_q_final", 1.0),
        ("B_upstream_Q", 1.0),
        ("C_upstream_Q", 0.75),
    ],
)
def test_effect_exact_boundary(name, threshold):
    rows, base, paths, up = evidence()
    base["logits"][:] = 0
    target = up[name[0]] if "upstream" in name else paths[name]
    signs = np.array([r["label"] for r in rows])
    for value, expected in [(threshold, True), (np.nextafter(threshold, -np.inf), False)]:
        target["logits"][:, 0] = -signs * value
        target["logits"][:, 1] = 0
        actual = evaluate(rows, base, paths, up)
        assert actual["criteria"][name + "_effect"] is expected
        assert actual == recompute(rows, base, paths, up)


def test_gap_specificity_and_signed_cancellation():
    rows, base, paths, up = evidence()
    signs = np.array([r["label"] for r in rows])
    # Median clean-to-A-donor signed gap is exactly 8; quarter-gap gate is 2.
    base["logits"][:, 0] = signs * 4
    base["logits"][:, 1] = 0

    def set_effect(name, values):
        paths[name]["logits"][:, 0] = base["logits"][:, 0] - signs * values
        paths[name]["logits"][:, 1] = 0

    set_effect("A_ov_color", 2.0)
    set_effect("A_ov_other", 0.2)
    set_effect("A_qk", np.tile([-2.0, 2.0], len(rows) // 2))
    a = evaluate(rows, base, paths, up)
    assert a["clean_counterfactual_gap"] == 8
    assert a["criteria"]["A_gap_fraction"]
    assert a["criteria"]["A_qk_signed"]
    assert a["effects"]["A_qk"]["median_abs"] == 2
    set_effect("A_ov_color", 2.0 - 1e-9)
    assert not evaluate(rows, base, paths, up)["criteria"]["A_gap_fraction"]
    assert evaluate(rows, base, paths, up) == recompute(rows, base, paths, up)


def test_selector_budget_ties_and_no_hidden_layer_constraint():
    assert len(DOWNSTREAM_CANDIDATES) == 176
    assert select_extension({(15, 10): 2.0, (15, 2): 2.0, (16, 0): 1.0}) == (15, 2)
    assert upstream_candidates(((15, 0), (15, 1), (25, 9))) == tuple(
        (layer, h) for layer in range(10, 15) for h in range(16)
    )
    with pytest.raises(ValueError, match="ABSTAIN"):
        select_extension({})
    with pytest.raises(ValueError, match="Nonfinite"):
        select_extension({(15, 0): np.nan})


def test_no_prompt_dropping_or_donor_reassignment():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "data/bounded-poc-v7-capstone/dataset.json").read_text())["rows"]
    validate(rows)
    with pytest.raises(AssertionError):
        validate(rows[8:])
    altered = copy.deepcopy(rows)
    altered[0]["donors"]["C"] = altered[0]["donors"]["B"]
    with pytest.raises(AssertionError):
        validate(altered)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("residual_width", [16, 32])
async def test_same_layer_gqa_heads_and_pure_q_replay(tiny_harness, device, residual_width):
    import torch

    from src.config import ModelConfig, WorkerConfig
    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.v4_harness import V4Probe, V4Spec
    from src.poc.v10_harness import V10Harness

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in tiny-model Metal regression")
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    wc = WorkerConfig(batch_size=4)
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = tiny_harness.model.config.to_dict()
    config.update(hidden_size=residual_width, intermediate_size=residual_width * 2)
    model = Qwen3ForCausalLM(Qwen3Config(**config)).to(device=device, dtype=dtype)
    h = V10Harness(
        model,
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        wc,
    )
    targets = ((1, 0), (1, 1), (2, 2))  # shared GQA KV plus another layer
    ids = [(2, 3, 4, 5, 6), (4, 3, 2, 5, 6), (2, 3, 4, 5, 6, 7), (4, 3, 2, 5, 6, 7)]

    def probes(mode="capture", payload=None, active=(), upstream=()):
        spec = V4Spec((0,), targets=targets, mode=mode, active=active, upstream=upstream)
        return [
            V4Probe("toy", x, (2, 3), spec, payload=None if payload is None else payload[i])
            for i, x in enumerate(ids)
        ]

    async with MPSBatchWorker(h, wc) as w:
        base = await w.map(probes())
        from scripts.run_bounded_poc_v10_qwen17b import pack, payloads

        packed = pack(base, targets, ((0, 0),), full=True)
        rows = [
            dict(
                id=str(i),
                token_ids=list(x),
                donors={a: str(i ^ 1) for a in "ABC"},
                color_positions=[1, 3],
                context_end=4,
            )
            for i, x in enumerate(ids)
        ]
        bundles = payloads(rows, packed, targets, ((0, 0),), axis="C")
        for i, bundle in enumerate(bundles):
            for key in bundle["clean"]:
                for field in ("h", "p", "q", "k", "v"):
                    np.testing.assert_array_equal(
                        bundle["clean"][key][field], base[i].routing[key][field]
                    )
                    np.testing.assert_array_equal(
                        bundle["donor"][key][field], base[i ^ 1].routing[key][field]
                    )
            np.testing.assert_array_equal(bundle["upstream"]["0:0"], base[i ^ 1].heads["0:0"])

        def payload(self_patch=False):
            return [
                dict(
                    clean=r.routing,
                    donor=base[i if self_patch else i ^ 1].routing,
                    upstream={"0:0": base[i if self_patch else i ^ 1].heads["0:0"]},
                    color_positions=[1, 3],
                    context_end=4,
                )
                for i, r in enumerate(base)
            ]

        invalid = [{key: copy.deepcopy(value) for key, value in item.items()} for item in payload()]
        invalid[0]["donor"]["1:0"]["q"][0] = float("nan")
        with pytest.raises(ValueError, match="Invalid intervention values"):
            await w.map(probes("q_final", invalid, targets))

        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            out = await w.map(probes(mode, payload(True), targets))
            np.testing.assert_array_equal([r.logits for r in out], [r.logits for r in base])
        for mode in ("k_color", "ov_color", "ov_other", "qk"):
            changed = await w.map(probes(mode, payload(), targets))
            for i, result in enumerate(changed):
                for key, routed in result.routing.items():
                    clean = base[i].routing[key]
                    donor = base[i ^ 1].routing[key]
                    pc = np.asarray(clean["p"], dtype=np.float32)
                    vc = np.asarray(clean["v"], dtype=np.float32)
                    if mode == "k_color":
                        k = np.asarray(clean["k"], dtype=np.float32)
                        k[[1, 3]] = np.asarray(donor["k"], dtype=np.float32)[[1, 3]]
                        q = np.asarray(clean["q"], dtype=np.float32)
                        scores = (q * k).sum(-1) / np.sqrt(k.shape[-1])
                        pd = np.exp(scores - scores.max())
                        pd /= pd.sum()
                        expected = ((pd - pc)[:, None] * vc).sum(0)
                    elif mode == "qk":
                        expected = ((np.asarray(donor["p"]) - pc)[:, None] * vc).sum(0)
                    else:
                        mask = np.zeros(len(pc), dtype=np.float32)
                        mask[[1, 3]] = 1
                        if mode == "ov_other":
                            mask = 1 - mask
                        expected = (
                            pc[:, None] * (np.asarray(donor["v"]) - vc) * mask[:, None]
                        ).sum(0)
                    np.testing.assert_allclose(routed["delta"], expected, atol=2e-6, rtol=0)
                    assert routed["sibling_error"] == 0

        mid = await w.map(probes("mediate", payload(), upstream=((0, 0),)))
        replay = [{k: copy.deepcopy(v) for k, v in item.items()} for item in payload()]
        for i, item in enumerate(replay):
            for key in item["donor"]:
                item["donor"][key]["q"] = mid[i].routing[key]["q"]
                item["donor"][key]["k"] = np.full_like(item["donor"][key]["k"], 17)
                item["donor"][key]["v"] = np.full_like(item["donor"][key]["v"], -9)
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
                assert result["sibling_error"] == 0
        # A single member must not edit its shared-KV sibling at the same layer.
        solo = await w.map(probes("q_final", payload(), active=((1, 0),)))
        for i, r in enumerate(solo):
            np.testing.assert_array_equal(r.routing["1:1"]["applied"], base[i].routing["1:1"]["h"])
    assert all(not layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers)


def test_context_controls_check_logits_not_only_margin():
    rows, base, paths, upstream = evidence()
    paths["B_k_context"]["logits"] = base["logits"] + 1.0
    result = evaluate(rows, base, paths, upstream)
    assert not result["criteria"]["B_k_context_exact"]
    assert result == recompute(rows, base, paths, upstream)


@pytest.mark.parametrize(
    "name,minimum",
    [
        ("A_ov_color", 52),
        ("B_q_final", 58),
        ("C_q_final", 58),
        ("B_upstream_Q", 58),
        ("C_upstream_Q", 58),
    ],
)
def test_direction_uses_all_64_prompts(name, minimum):
    rows, base, paths, upstream = evidence()
    base["logits"][:] = 0
    out = upstream[name[0]] if "upstream" in name else paths[name]
    labels = np.array([r["label"] for r in rows])
    for count, expected in ((minimum, True), (minimum - 1, False)):
        reductions = np.r_[np.ones(count), np.zeros(64 - count)]
        out["logits"][:, 0] = -labels * reductions
        out["logits"][:, 1] = 0
        result = evaluate(rows, base, paths, upstream)
        assert result["criteria"][name + "_direction"] is expected
        assert result == recompute(rows, base, paths, upstream)


@pytest.mark.parametrize(
    "name,threshold", [("C_q_final", 0.05), ("B_upstream_Q", 0.05), ("C_upstream_Q", 0.04)]
)
def test_attention_magnitude_exact_boundary(name, threshold):
    rows, base, paths, upstream = evidence()
    index = {r["id"]: r for r in rows}
    out = upstream[name[0]] if "upstream" in name else paths[name]
    base["p"] = np.zeros((64, 3, 3))
    out["routed_p"] = np.zeros((64, 3, 3))
    for value, expected in ((threshold, True), (np.nextafter(threshold, -np.inf), False)):
        for i, row in enumerate(rows):
            c = row["target_position"]
            d = index[row["donors"][name[0]]]["target_position"]
            base["p"][i, :, c] = threshold
            base["p"][i, :, 2] = 1 - threshold
            out["routed_p"][i, :, c] = threshold - value
            out["routed_p"][i, :, d] = value
            out["routed_p"][i, :, 2] = 1 - threshold
        result = evaluate(rows, base, paths, upstream)
        for pole in ("gain", "loss"):
            assert result["criteria"][name + "_" + pole] is expected
        assert result == recompute(rows, base, paths, upstream)


@pytest.mark.parametrize("name", ["A_qk", "C_ov_color"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_specificity_exact_bounds(name, sign):
    rows, base, paths, upstream = evidence()
    base["logits"][:] = 0
    labels = np.array([r["label"] for r in rows])
    criterion = "A_qk_signed" if name == "A_qk" else "C_ov_signed"
    for value, expected in ((sign * 0.35, True), (np.nextafter(sign * 0.35, sign * np.inf), False)):
        paths[name]["logits"][:, 0] = -labels * value
        paths[name]["logits"][:, 1] = 0
        result = evaluate(rows, base, paths, upstream)
        assert result["criteria"][criterion] is expected
        assert result == recompute(rows, base, paths, upstream)


def test_ratio_and_other_abs_exact_boundaries():
    rows, base, paths, upstream = evidence()
    base["logits"][:] = 0
    labels = np.array([r["label"] for r in rows])

    def effect(name, value):
        paths[name]["logits"][:, 0] = -labels * value
        paths[name]["logits"][:, 1] = 0

    effect("A_ov_other", 0.20)
    effect("A_ov_color", 8 * 0.20)
    effect("C_k_color", 1.0)
    effect("C_q_final", 1.75)
    result = evaluate(rows, base, paths, upstream)
    assert all(
        result["criteria"][k] for k in ("A_other_abs", "A_specificity_ratio", "C_dominance_median")
    )
    effect("A_ov_other", np.nextafter(0.20, np.inf))
    effect("C_q_final", np.nextafter(1.75, -np.inf))
    result = evaluate(rows, base, paths, upstream)
    assert all(
        not result["criteria"][k]
        for k in ("A_other_abs", "A_specificity_ratio", "C_dominance_median")
    )
    assert result == recompute(rows, base, paths, upstream)


def test_group_sign_test_keeps_zero_groups():
    rows, base, paths, upstream = evidence()
    base["logits"][:] = 0
    labels = np.array([r["label"] for r in rows])
    paths["B_q_final"]["logits"][:, 0] = -labels * 2.0
    paths["B_q_final"]["logits"][:, 1] = 0
    good = evaluate(rows, base, paths, upstream)
    assert good["effects"]["B_q_final"]["sign_p"] == 1 / 256
    first_group = rows[0]["group"]
    paths["B_q_final"]["logits"][[r["group"] == first_group for r in rows]] = 0
    result = evaluate(rows, base, paths, upstream)
    assert result["effects"]["B_q_final"]["sign_p"] == 9 / 256
    assert not result["criteria"]["B_q_final_sign_p"]
    assert result == recompute(rows, base, paths, upstream)
