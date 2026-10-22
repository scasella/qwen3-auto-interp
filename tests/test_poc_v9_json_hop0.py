import copy
import json
import os

import numpy as np
import pytest

from scripts.audit_bounded_poc_v9_json_hop0 import (
    audit_controls,
    recompute,
    recompute2,
    validate_data,
)
from src.poc.v9_analysis import evaluate, select_hop0, track2
from src.poc.v9_controls import planted, three_stage
from src.poc.v9_json_hop0 import build, validate


def test_controls_and_independent_grading():
    c = planted()
    assert c["passed"]
    audit_controls(c)
    assert c["three_literal"]["effects"]["C_hop0"]["median"] == 0
    assert not c["three_reversed"]["passed"]


def test_no_dropped_screen_candidates_and_tie_break():
    screen = [
        dict(head=f"{layer}:{h}", B={"median": 1.0}, C={"median": 1.0})
        for layer in range(15)
        for h in range(16)
    ]
    assert select_hop0(screen[::-1]) == ["0:0", "0:1", "0:2"]
    with pytest.raises(ValueError):
        select_hop0(screen[:-1])
    screen[0]["C"]["median"] = float("nan")
    with pytest.raises(ValueError):
        select_hop0(screen)


def test_actual_tokenizer_complete_dataset_and_valid_json():
    from transformers import AutoTokenizer

    from scripts.run_bounded_poc_v9_json_hop0 import MODEL, retired

    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = build(tok, retired())
    assert len(data["rows"]) == 256
    validate(data["rows"])
    validate_data(data)
    for r in data["rows"]:
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == r["token_ids"]
        for p in r["key_positions"] + r["color_positions"] + [r["query_position"]]:
            a, b = enc["offset_mapping"][p]
            word = r["text"][a:b]
            assert word.startswith(" ") and tok.encode(word, add_special_tokens=False) == [
                r["token_ids"][p]
            ]
        if r["format"] == "h0":
            for line in r["text"].splitlines()[1:3]:
                assert isinstance(json.loads(line), dict)
    bad = copy.deepcopy(data["rows"])
    bad.pop()
    with pytest.raises(AssertionError):
        validate(bad)
    bad = copy.deepcopy(data)
    bad["rows"][0]["donors"]["C"] = bad["rows"][0]["donors"]["A"]
    with pytest.raises(AssertionError):
        validate_data(bad)


@pytest.mark.parametrize(
    "name,t",
    [
        ("A_ov_color", 2.0),
        ("B_q_final", 2.0),
        ("C_q_final", 1.5),
        ("B_upstream_Q", 1.25),
        ("C_upstream_Q", 1.0),
    ],
)
def test_t1_effect_boundaries(name, t):
    e = planted()["planted_pointer_model"]["evidence"]

    def cv(a):
        return {k: np.asarray(v, dtype=float) for k, v in a.items()}

    rows = e["rows"]
    base = cv(e["base"])
    paths = {k: cv(v) for k, v in e["paths"].items()}
    up = {k: cv(v) for k, v in e["upstream"].items()}
    labels = np.array([r["label"] for r in rows])
    base["logits"][:] = 0
    target = up[name[0]] if "upstream" in name else paths[name]
    target["logits"][:] = 0
    for value, passes in [(t, True), (np.nextafter(t, -np.inf), False)]:
        target["logits"][:, 0] = -labels * value
        result = evaluate(rows, base, paths, up)
        assert result == recompute(rows, base, paths, up)
        assert result["criteria"][name + "_effect"] == passes


@pytest.mark.parametrize("axis,t", [("B", 0.5), ("C", 0.35)])
def test_t2_effect_boundaries_and_conservation_absolute_difference(axis, t):
    e = three_stage()["evidence"]
    rows = e["rows"]
    labels = np.array([r["label"] for r in rows])
    base = {"logits": np.zeros((len(rows), 2))}
    direct = {a: {"logits": np.zeros((len(rows), 2))} for a in "BC"}
    ins = {
        a + "_U_" + m: dict(logits=np.zeros((len(rows), 2)))
        for a in "BC"
        for m in ("full", "q_final", "kv_context", "k_all", "v_all")
    }
    hops = {a + "_hop0": dict(logits=np.zeros((len(rows), 2))) for a in "BC"}
    for value, passes in [(t, True), (np.nextafter(t, -np.inf), False)]:
        hops[axis + "_hop0"]["logits"][:, 0] = -labels * value
        result = track2(rows, base, direct, ins, hops, 0.0)
        assert result == recompute2(rows, base, direct, ins, hops, 0.0)
        assert result["criteria"][axis + "_hop0_effect"] == passes
    # Opposite signs must not cancel a reconstruction error.
    ins["B_U_full"]["logits"][:, 0] = labels * 0.051
    assert not track2(rows, base, direct, ins, hops, 0.0)["criteria"]["B_conservation"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
async def test_three_stage_harness_identity_and_isolation(tiny_harness, device):
    import torch

    from src.config import ModelConfig, WorkerConfig
    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.v4_harness import V4Harness, V4Probe, V4Spec

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in tiny Metal")
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    h = V4Harness(
        tiny_harness.model.to(device=device, dtype=dtype),
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        WorkerConfig(batch_size=4),
    )
    ids = [(2, 3, 4, 5, 6), (2, 3, 4, 7, 6), (2, 4, 5, 6, 7, 8), (2, 4, 5, 7, 7, 8)]

    def probes(target, mode="capture", payload=None, upstream=()):
        spec = V4Spec(
            (0, 1),
            targets=(target,),
            active=(target,) if mode == "q_final" else (),
            mode=mode,
            upstream=upstream,
        )
        return [
            V4Probe("tiny", x, (2, 3), spec, payload=None if payload is None else payload[i])
            for i, x in enumerate(ids)
        ]

    def payload(base, other, source=None, key=None):
        return [
            dict(
                clean=r.routing,
                donor=other[i].routing,
                context_end=3,
                color_positions=[1, 2],
                upstream={} if source is None else {key: source[i]},
            )
            for i, r in enumerate(base)
        ]

    async with MPSBatchWorker(h, WorkerConfig(batch_size=4)) as w:
        ub = await w.map(probes((1, 1)))
        db = await w.map(probes((2, 2)))
        for self_patch in (True, False):
            source = (
                [r.heads["0:0"] for r in ub]
                if self_patch
                else [ub[i ^ 1].heads["0:0"] for i in range(4)]
            )
            um = await w.map(probes((1, 1), "mediate", payload(ub, ub, source, "0:0"), ((0, 0),)))
            ur = await w.map(probes((1, 1), "q_final", payload(ub, um)))
            dm = await w.map(
                probes(
                    (2, 2),
                    "mediate",
                    payload(db, db, [r.routing["1:1"]["applied"] for r in ur], "1:1"),
                    ((1, 1),),
                )
            )
            end = await w.map(probes((2, 2), "q_final", payload(db, dm)))
            for batch, key in [(um, "1:1"), (ur, "1:1"), (dm, "2:2"), (end, "2:2")]:
                assert all(r.routing[key]["sibling_error"] == 0 for r in batch)
            for i in range(4):
                np.testing.assert_array_equal(
                    um[i].routing["1:1"]["applied"], ub[i].routing["1:1"]["h"]
                )
                np.testing.assert_array_equal(
                    dm[i].routing["2:2"]["applied"], db[i].routing["2:2"]["h"]
                )
            if self_patch:
                np.testing.assert_array_equal([r.logits for r in end], [r.logits for r in db])
    for layer in h.model.model.layers:
        assert (
            not layer.self_attn._forward_pre_hooks and not layer.self_attn.o_proj._forward_pre_hooks
        )


@pytest.mark.parametrize("value,passes", [(0.75, True), (np.nextafter(0.75, -np.inf), False)])
def test_internal_pathway_boundary(value, passes):
    e = three_stage()["evidence"]
    rows = e["rows"]
    labels = np.array([r["label"] for r in rows])
    base = {"logits": np.zeros((len(rows), 2))}
    direct = {a: copy.deepcopy(base) for a in "BC"}
    ins = {
        a + "_U_" + m: copy.deepcopy(base)
        for a in "BC"
        for m in ("full", "q_final", "kv_context", "k_all", "v_all")
    }
    hops = {a + "_hop0": copy.deepcopy(base) for a in "BC"}
    ins["C_U_q_final"]["logits"][:, 0] = -labels * value
    result = track2(rows, base, direct, ins, hops, 0.0)
    assert result == recompute2(rows, base, direct, ins, hops, 0.0)
    assert result["pathways"]["C_U_q_final"] == passes
    assert not track2(rows, base, direct, ins, hops, np.nextafter(0.0, 1.0))["criteria"][
        "hop0_identity"
    ]
    for error, ok in [(0.05, True), (np.nextafter(0.05, np.inf), False)]:
        ins["B_U_full"]["logits"][:, 0] = error
        assert track2(rows, base, direct, ins, hops, 0.0)["criteria"]["B_conservation"] == ok


def test_retired_key_exclusion_does_not_skip_worktree_ancestor(tmp_path, monkeypatch):
    from scripts import run_bounded_poc_v9_json_hop0 as runner

    root = tmp_path / "bounded-poc-v9-json-hop0" / "project"
    historical = root / "data" / "historical"
    current = root / "data" / "bounded-poc-v9-json-hop0-restart1"
    historical.mkdir(parents=True)
    current.mkdir(parents=True)
    row = dict(keys=["maple", "cedar"], text="retired")
    (historical / "dataset.json").write_text(json.dumps(dict(rows=[row])))
    (current / "dataset.json").write_text(json.dumps(dict(rows=[dict(keys=["other"])])))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUT", current)
    assert runner.retired() == [row]


def test_prepare_independently_rejects_empty_retirement_pool(tmp_path, monkeypatch):
    from scripts import run_bounded_poc_v9_json_hop0 as runner

    monkeypatch.setattr(runner, "OUT", tmp_path / "invalid-preparation")
    monkeypatch.setattr(runner, "protected", lambda: 1691)
    monkeypatch.setattr(runner, "retired", lambda: [])
    with pytest.raises(AssertionError):
        runner.prepare()
    assert not (runner.OUT / "dataset.json").exists()
