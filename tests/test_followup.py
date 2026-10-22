"""Executable path controls: source selector and upstream relay must separate."""

import os

import numpy as np
import pytest
import torch

from src.config import ModelConfig, WorkerConfig
from src.followup.harness import MODES, PathHarness, PathProbe, PathSpec, intervene
from src.harness.mps_queue import MPSBatchWorker
from src.poc.harness import CausalProbe, CausalSpec


def test_planted_source_and_relay():
    # Three positions: correct value, other value, upstream answer at query.
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[[8.0, 0.0], [-8.0, 0.0], [0.0, 0.0]]])
    v = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [2.0, 0.0]]])
    p = dict(
        value_positions=[0, 1],
        query_record=0,
        suffix_start=2,
        q=[-1.0, 0.0],
        k=k[0].tolist(),
        v=[[-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
    )

    def output(q, k, v):
        return ((q[:, None] * k).sum(-1).softmax(-1)[:, :, None] * v).sum(1)[0, 0].item()

    assert output(q, k, v) > 1.9
    assert output(*intervene(q, k, v, "source_swap", [p])) < -1.9
    assert output(*intervene(q, k, v, "q_query", [p])) < -1.9
    assert output(*intervene(q, k, v, "v_suffix", [p])) > 1.9
    relay_k = torch.tensor([[[-8.0, 0.0], [-8.0, 0.0], [8.0, 0.0]]])
    assert output(*intervene(q, relay_k, v, "source_swap", [p])) > 1.9
    assert output(*intervene(q, relay_k, v, "v_suffix", [p])) < -1.9
    assert torch.equal(v, torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [2.0, 0.0]]]))


async def run_path_control(tiny_harness, device):
    wc = WorkerConfig(batch_size=4)
    model = tiny_harness.model.to(
        device=device, dtype=torch.bfloat16 if device == "mps" else torch.float32
    )
    harness = PathHarness(
        model,
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        wc,
    )
    texts = ("abcde", "vwxyz", "abcdefg", "vwxzzzz")
    ids = [tuple(harness.tokenizer.encode(t)) for t in texts]

    def probes(mode, payloads=None):
        return [
            PathProbe(
                t,
                tok,
                (2, 3),
                PathSpec((1,), mode=mode, layer=1, head=1),
                payload=None if payloads is None else payloads[i],
            )
            for i, (t, tok) in enumerate(zip(texts, ids))
        ]

    async with MPSBatchWorker(harness, wc) as worker:
        base = await worker.map(probes("capture"))
        same = await worker.map(probes("reconstruct", [{}] * 4))
        assert max(abs(a.margin - b.margin) for a, b in zip(base, same)) < 0.05
        payloads = []
        for i in range(4):
            donor = base[i ^ 1].path
            payloads.append(
                dict(
                    q=donor["q"],
                    k=donor["k"],
                    v=donor["v"],
                    value_positions=[1, 2],
                    query_record=0,
                    suffix_start=3,
                )
            )
        for mode in MODES:
            out = await worker.map(probes(mode, payloads))
            for original, patched in zip(base, out):
                before, after = (
                    np.array(patched.path["original_heads"]),
                    np.array(patched.path["patched_heads"]),
                )
                np.testing.assert_array_equal(before[[0, 2, 3]], after[[0, 2, 3]])
                # Independent ordinary head-output donor patch must reproduce the path patch.
            check = await worker.map(
                [
                    CausalProbe(
                        t,
                        tok,
                        (2, 3),
                        CausalSpec((), patch_head=(1, 1)),
                        tuple(r.path["patched_heads"][1]),
                    )
                    for t, tok, r in zip(texts, ids, out)
                ]
            )
            np.testing.assert_allclose(
                [r.logits for r in out], [r.logits for r in check], atol=1e-5
            )
    assert not harness.model.model.layers[1].self_attn._forward_pre_hooks
    assert not harness.model.model.layers[1].self_attn.o_proj._forward_pre_hooks


async def test_path_patches_cpu(tiny_harness):
    await run_path_control(tiny_harness, "cpu")


@pytest.mark.hardware
@pytest.mark.skipif(os.getenv("QWEN_HARDWARE_TEST") != "1", reason="opt-in Metal test")
async def test_path_patches_metal(tiny_harness):
    await run_path_control(tiny_harness, "mps")


def test_frozen_classifier_distinguishes_source_relay_and_mixed():
    from src.followup.analysis import mechanisms

    rows = [
        dict(group=f"g{g}", label=y, query_record=q, value_positions=[0, 1])
        for g in range(16)
        for q, y in ((0, 1), (0, -1), (1, -1), (1, 1))
    ]
    y = np.array([r["label"] for r in rows])
    base = y * 4.0
    attention = [[0.8, 0.1] if r["query_record"] == 0 else [0.1, 0.8] for r in rows]
    paths = {mode: base.copy() for mode in MODES}
    for name in ("source_swap", "q_query", "drop_relevant"):
        paths[name] = base - y * 2
    result = mechanisms(rows, base, paths, attention, 8.0)
    assert result["classification"] == "source routing supported; exclusive computation unresolved"
    paths["v_suffix"] = base - y * 2
    assert mechanisms(rows, base, paths, attention, 8.0)["classification"] == "mixed"
    paths["source_swap"] = base.copy()
    assert (
        mechanisms(rows, base, paths, attention, 8.0)["classification"]
        == "upstream-query relay supported"
    )
    # No significance does not suffice to infer a relay: source magnitude must be small.
    paths["source_swap"] = base - y * 0.2
    assert mechanisms(rows, base, paths, attention, 8.0)["classification"] == "unresolved"
