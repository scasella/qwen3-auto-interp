import asyncio

import numpy as np
import pytest
from conftest import FakeHarness

from src.config import WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.harness.qwen_model import Probe, ProbeSpec
from src.schemas.contracts import UnitRef


async def test_queue_batches_and_preserves_order(unit):
    harness = FakeHarness()
    config = WorkerConfig(batch_size=4, queue_size=2, purge_every=2)
    spec = ProbeSpec((unit,))
    async with MPSBatchWorker(harness, config) as worker:
        results = await asyncio.gather(
            *[worker.submit(Probe(f"OOD_POSITIVE-{i}", spec)) for i in range(13)]
        )
        assert len(results) == 13
        assert all(r.values[unit.key] == [10000] for r in results)
    assert sum(map(len, harness.calls)) == 13
    assert max(map(len, harness.calls)) <= 4
    assert harness.purges == len(harness.calls) // 2
    assert worker.queue.empty()
    with pytest.raises(RuntimeError):
        await worker.submit(Probe("late", spec))


async def test_queue_recovers_after_failed_batch(unit):
    harness = FakeHarness()
    async with MPSBatchWorker(harness, WorkerConfig()) as worker:
        with pytest.raises(ValueError, match="invalid probe"):
            await worker.submit(Probe("error", ProbeSpec((unit,))))
        result = await worker.submit(Probe("valid", ProbeSpec((unit,))))
        assert result.values[unit.key] == [0]


async def test_cancellation_does_not_poison_worker(unit):
    async with MPSBatchWorker(FakeHarness(), WorkerConfig(queue_size=1)) as worker:
        task = asyncio.create_task(worker.submit(Probe("cancel", ProbeSpec((unit,)))))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        result = await worker.submit(Probe("OOD_POSITIVE", ProbeSpec((unit,))))
        assert result.values[unit.key] == [10000]


async def test_real_qwen_hooks_padding_clamps_and_cleanup(tiny_harness):
    mlp = UnitRef(kind="mlp", layer=0, index=3)
    head = UnitRef(kind="attention", layer=0, index=2)
    spec = ProbeSpec((mlp, head), all_tokens=True)
    async with MPSBatchWorker(tiny_harness, tiny_harness.worker) as worker:
        batched = await worker.map([Probe("abcd", spec), Probe("xy", spec)])
        single = await worker.submit(Probe("xy", spec))
        for u in (mlp, head):
            assert len(batched[1].values[u.key]) == 2
            np.testing.assert_allclose(batched[1].values[u.key], single.values[u.key], atol=1e-6)
        clamped = await worker.submit(
            Probe("abc", ProbeSpec((mlp, head), clamps=((mlp, 3.0), (head, -2.0)), all_tokens=True))
        )
        assert clamped.values[mlp.key] == [3] * 3
        assert clamped.values[head.key] == [-2] * 3
        with pytest.raises(ValueError, match="max_seq_len"):
            await worker.submit(Probe("x" * 129, spec))
    for module in tiny_harness.model.modules():
        assert not module._forward_pre_hooks


async def test_gradient_patch_matches_small_finite_difference(tiny_harness):
    source = UnitRef(kind="mlp", layer=0, index=4)
    target = UnitRef(kind="mlp", layer=1, index=5)
    async with MPSBatchWorker(tiny_harness, tiny_harness.worker) as worker:
        # A one-token context gives a single source coordinate for an exact local derivative.
        measured = await worker.submit(Probe("x", ProbeSpec((source, target))))
        original = measured.values[source.key][0]
        baseline = measured.values[target.key][0]
        delta = 0.0001
        gradient = await worker.submit(
            Probe(
                "x",
                ProbeSpec(
                    (target,),
                    attribution_target=target,
                    attribution_baselines=((source, original + delta),),
                ),
            )
        )
        patched = await worker.submit(
            Probe("x", ProbeSpec((target,), clamps=((source, original + delta),)))
        )
        difference = patched.values[target.key][0] - baseline
        assert np.isfinite(gradient.grad_act[source.key])
        np.testing.assert_allclose(
            gradient.attribution[source.key], difference, rtol=0.08, atol=1e-8
        )
    assert not any(m._forward_pre_hooks for m in tiny_harness.model.modules())


async def test_head_layout_uses_explicit_head_dim(tiny_harness):
    # hidden_size / nheads is 4; the actual head_dim is 8 in this fixture.
    unit = UnitRef(kind="attention", layer=0, index=3)
    assert tiny_harness._slice(unit) == slice(24, 32)
    with pytest.raises(RuntimeError, match="MPSBatchWorker"):
        tiny_harness._execute([Probe("a", ProbeSpec((unit,)))], object())
