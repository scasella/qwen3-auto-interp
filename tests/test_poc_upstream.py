import os

import numpy as np
import pytest

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.upstream_controls import planted
from src.poc.upstream_harness import RoutingHarness, RoutingProbe, RoutingSpec
from src.poc.upstream_templates import build_upstream


def test_two_stage_controls():
    assert planted()["passed"]


def test_axes_and_fresh_data():
    from tests.test_poc_v2 import OffsetTokenizer

    data = build_upstream(OffsetTokenizer(), [])
    assert len(data["rows"]) == 256
    assert {r["split"] for r in data["rows"]} == {"discovery", "primary", "vocabulary"}


async def exercise(tiny_harness, device):
    import torch

    model = tiny_harness.model.to(
        device=device, dtype=torch.bfloat16 if device == "mps" else torch.float32
    )
    wc = WorkerConfig(batch_size=4)
    h = RoutingHarness(
        model,
        tiny_harness.tokenizer,
        ModelConfig(device=device, dtype="bfloat16" if device == "mps" else "float32"),
        wc,
    )
    targets = ((1, 1), (2, 2))
    ids = [(2, 3, 4, 5, 6), (2, 3, 4, 7, 6), (3, 4, 5, 6, 7, 8), (3, 4, 5, 7, 7, 8)]

    def probes(spec, payloads=None):
        return [
            RoutingProbe("test", x, (2, 3), spec, payload=None if payloads is None else payloads[i])
            for i, x in enumerate(ids)
        ]

    async with MPSBatchWorker(h, wc) as worker:
        base = await worker.map(probes(RoutingSpec((0,), targets=targets)))
        payloads = []
        for i, r in enumerate(base):
            payloads.append(
                dict(
                    clean={k: {f: r.routing[k][f] for f in ("h", "p", "v")} for k in r.routing},
                    donor={
                        k: {f: base[i ^ 1].routing[k][f] for f in ("h", "p", "v")}
                        for k in r.routing
                    },
                    upstream={"0:0": base[i ^ 1].heads["0:0"]},
                    color_positions=[1, 2],
                    context_end=3,
                )
            )
        ov = await worker.map(
            probes(RoutingSpec((), targets=targets, active=targets, mode="ov_context"), payloads)
        )
        np.testing.assert_array_equal([r.logits for r in base], [r.logits for r in ov])
        mediated = await worker.map(
            probes(RoutingSpec((), targets=targets, mode="mediate", upstream=((0, 0),)), payloads)
        )
        for i, r in enumerate(mediated):
            for key in r.routing:
                np.testing.assert_array_equal(r.routing[key]["applied"], base[i].routing[key]["h"])
                payloads[i]["donor"][key]["p"] = r.routing[key]["p"]
        routed = await worker.map(
            probes(RoutingSpec((), targets=targets, active=targets, mode="qk"), payloads)
        )
        for i, r in enumerate(routed):
            for key in r.routing:
                assert r.routing[key]["sibling_error"] == 0
                c, donor = payloads[i]["clean"][key], payloads[i]["donor"][key]
                expected = np.sum((np.array(donor["p"]) - c["p"])[:, None] * c["v"], axis=0)
                np.testing.assert_allclose(r.routing[key]["delta"], expected, atol=2e-7)
        # Replaying precisely the same absolute head outputs must give identical logits.
        for i, r in enumerate(routed):
            for key in r.routing:
                payloads[i]["donor"][key]["h"] = r.routing[key]["applied"]
        replay = await worker.map(
            probes(RoutingSpec((), targets=targets, active=targets, mode="replace"), payloads)
        )
        np.testing.assert_array_equal([r.logits for r in replay], [r.logits for r in routed])
    for layer in model.model.layers:
        assert not layer.self_attn._forward_pre_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks


async def test_routing_cpu(tiny_harness):
    await exercise(tiny_harness, "cpu")


@pytest.mark.hardware
@pytest.mark.skipif(os.getenv("QWEN_HARDWARE_TEST") != "1", reason="opt-in Metal")
async def test_routing_metal(tiny_harness):
    await exercise(tiny_harness, "mps")
