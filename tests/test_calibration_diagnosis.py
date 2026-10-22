import os

import numpy as np
import pytest
import torch

from src.config import ModelConfig, WorkerConfig
from src.diagnosis.harness import TraceHarness, TraceSpec
from src.harness.mps_queue import MPSBatchWorker
from src.poc.harness import CausalProbe, CausalSpec


async def trace_control(tiny_harness, device):
    wc = WorkerConfig(batch_size=4)
    dtype = "bfloat16" if device == "mps" else "float32"
    model = tiny_harness.model.to(device=device, dtype=getattr(torch, dtype))
    harness = TraceHarness(
        model, tiny_harness.tokenizer, ModelConfig(device=device, dtype=dtype), wc
    )
    texts = ["abcdefg", "vwxyz", "the cat", "a tree"]
    ids = [tuple(harness.tokenizer.encode(t)) for t in texts]

    def probes(spec, donor=None):
        return [
            CausalProbe(t, tok, (2, 3), spec, None if donor is None else tuple(donor[i]))
            for i, (t, tok) in enumerate(zip(texts, ids))
        ]

    async with MPSBatchWorker(harness, wc) as worker:
        ordinary = await worker.map(probes(CausalSpec((1,))))
        trace = await worker.map(probes(TraceSpec((1,))))
        np.testing.assert_array_equal([r.logits for r in ordinary], [r.logits for r in trace])
        for r in trace:
            before = np.array([r.trace[f"{j}:before"] for j in range(3)])
            attn = np.array([r.trace[f"{j}:after_attention"] for j in range(3)])
            after = np.array([r.trace[f"{j}:after_mlp"] for j in range(3)])
            # Exact telescoping of projected realized residual increments.
            summed = (
                before[0, 0] + (attn[:, 0] - before[:, 0]).sum() + (after[:, 0] - attn[:, 0]).sum()
            )
            np.testing.assert_allclose(summed, after[-1, 0], atol=1e-6)
            assert abs(after[-1, 0] / after[-1, 1] - r.margin) < 0.05
        # Capture explicit module outputs with an independent hook, then self-patch them.
        captured = []
        positions = torch.tensor([len(x) - 1 for x in ids], device=device)
        batch = torch.arange(len(ids), device=device)
        from src.poc.harness import cpu_snapshot

        def capture(module, args, value):
            captured.append(cpu_snapshot(value[batch, positions]))

        handle = model.model.layers[1].mlp.register_forward_hook(capture)
        await worker.map(probes(CausalSpec(())))
        handle.remove()
        donors = captured[0].numpy()
        same = await worker.map(probes(TraceSpec((), trace=False, component=(1, "mlp")), donors))
        np.testing.assert_array_equal([r.logits for r in ordinary], [r.logits for r in same])

        # Independent zero hook must match our scoped module intervention.
        def zero(module, args, value):
            out = value.clone()
            out[batch, positions] = 0
            return out

        handle = model.model.layers[1].mlp.register_forward_hook(zero)
        reference = await worker.map(probes(CausalSpec(())))
        handle.remove()
        actual = await worker.map(
            probes(TraceSpec((), trace=False, component=(1, "mlp")), np.zeros_like(donors))
        )
        np.testing.assert_array_equal([r.logits for r in reference], [r.logits for r in actual])
    assert not model.model.layers[1].mlp._forward_hooks
    assert not model.model.layers[1]._forward_hooks


async def test_trace_and_module_interventions_cpu(tiny_harness):
    await trace_control(tiny_harness, "cpu")


@pytest.mark.hardware
@pytest.mark.skipif(os.getenv("QWEN_HARDWARE_TEST") != "1", reason="Metal opt-in")
async def test_trace_and_module_interventions_metal(tiny_harness):
    await trace_control(tiny_harness, "mps")
