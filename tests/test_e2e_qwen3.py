"""Offline end-to-end tests plus explicitly enabled hardware and authenticated Codex CLI checks."""

import os

import pytest
from conftest import FakeHarness, FakeScientist

from src.agents.client import ScientistClient
from src.config import CodexConfig, ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.harness.qwen_model import Probe, ProbeSpec, QwenHarness
from src.orchestrator.graph_db import GraphDB
from src.orchestrator.swarm import Swarm
from src.schemas.contracts import PIReport, UnitRef


@pytest.mark.parametrize(
    "fail_holdout,fail_discovery,state,attempts",
    [
        (False, False, "verified", 2),
        (True, False, "falsified", 2),
        (False, True, "falsified", 3),
    ],
)
async def test_end_to_end_and_resume(
    indexed, unit, settings, fail_holdout, fail_discovery, state, attempts
):
    db = GraphDB(settings.paths.circuits)
    run = db.start_run({"test": 1})
    client = FakeScientist()
    try:
        async with MPSBatchWorker(
            FakeHarness(fail_holdout, fail_discovery), settings.worker
        ) as worker:
            result = await Swarm(settings, client, worker, indexed, db, run).run_units([unit])
            assert not result["errors"]
            assert db.states(run)[unit.key] == state
            assert len(db.attempts(run, unit)) == attempts
            calls = len(client.calls)
            await Swarm(settings, client, worker, indexed, db, run).run_units([unit])
            assert len(client.calls) == calls
            assert len(db.verified(run)) == int(state == "verified")
        assert db.start_run({"test": 1}, resume=run) == run
        with pytest.raises(ValueError, match="Resume"):
            db.start_run({"test": 2}, resume=run)
    finally:
        db.close()


async def test_resume_failed_discovery_uses_remaining_strikes(indexed, unit, settings):
    from conftest import make_hypothesis, make_suite

    from src.agents.falsifier import grade

    db = GraphDB(settings.paths.circuits)
    run = db.start_run({"test": "crash"})
    hypothesis = make_hypothesis(unit)
    evaluation = grade(
        make_suite(), [0] * 45, indexed.quantiles(unit), settings.thresholds, "discovery"
    )
    db.record_attempt(run, unit, 1, hypothesis, evaluation)
    client = FakeScientist()
    try:
        async with MPSBatchWorker(FakeHarness(fail_discovery=True), settings.worker) as worker:
            await Swarm(settings, client, worker, indexed, db, run).run_units([unit])
        assert len(db.attempts(run, unit)) == 3
        assert sum(role == "explorer" for role, _ in client.calls) == 6
    finally:
        db.close()


@pytest.mark.hardware
@pytest.mark.skipif(os.getenv("QWEN_HARDWARE_TEST") != "1", reason="Set QWEN_HARDWARE_TEST=1")
async def test_full_qwen_on_mps():
    import torch

    assert torch.backends.mps.is_available(), "Requested hardware test requires MPS"
    harness = QwenHarness.load(
        ModelConfig(local_files_only=True, revision=os.getenv("QWEN_TEST_REVISION", "main")),
        WorkerConfig(),
    )
    mlp = UnitRef(kind="mlp", layer=1, index=100)
    head = UnitRef(kind="attention", layer=0, index=15)
    target = UnitRef(kind="mlp", layer=2, index=100)
    async with MPSBatchWorker(harness, harness.worker) as worker:
        result = await worker.submit(
            Probe("If A precedes B, then B follows A.", ProbeSpec((mlp, head, target)))
        )
        assert all(torch.isfinite(torch.tensor(v)).all() for v in result.values.values())
        gradient = await worker.submit(
            Probe(
                "A implies B.",
                ProbeSpec(
                    (target,), attribution_target=target, attribution_baselines=((mlp, 0.0),)
                ),
            )
        )
        assert mlp.key in gradient.attribution
        clamped = await worker.submit(
            Probe("A implies B.", ProbeSpec((mlp,), clamps=((mlp, 0.0),)))
        )
        assert clamped.values[mlp.key] == [0]
        full_batch = await worker.map(
            [
                Probe("", ProbeSpec((mlp, head)), token_ids=tuple(result.token_ids * 128)[:128])
                for _ in range(64)
            ]
        )
        assert len(full_batch) == 64
        assert all(len(r.token_ids) == 128 for r in full_batch)


@pytest.mark.codex
@pytest.mark.skipif(
    os.getenv("QWEN_CODEX_TEST") != "1", reason="Set QWEN_CODEX_TEST=1 for a live Codex request"
)
async def test_live_structured_scientist():
    client = ScientistClient(CodexConfig(max_attempts=1, timeout_seconds=60))
    try:
        report = await client.ask(
            "pi",
            PIReport,
            {
                "verified_units": [],
                "pending_units": [],
                "edges": [],
                "instruction": "Return empty clusters and priorities; no evidence exists yet.",
            },
        )
        assert not report.clusters and not report.prioritized_units
        assert client.metrics.input_tokens > 0
    finally:
        await client.close()


@pytest.mark.hardware
@pytest.mark.codex
@pytest.mark.skipif(os.getenv("QWEN_FULL_SMOKE") != "1", reason="Set QWEN_FULL_SMOKE=1")
async def test_full_hardware_scientist_pipeline(tmp_path):
    from src.config import Settings
    from src.harness.indexer import ActivationIndex, calibrate

    settings = Settings()
    settings.model.local_files_only = True
    settings.model.revision = os.getenv("QWEN_TEST_REVISION", "main")
    settings.swarm.pipelines = 1
    settings.swarm.max_strikes = 1
    settings.swarm.circuit_workers = 1
    settings.calibration.min_tokens = 1000
    settings.codex.max_attempts = 3
    settings.codex.timeout_seconds = 180
    settings.paths.activations = tmp_path / "activations.duckdb"
    settings.paths.circuits = tmp_path / "circuits.sqlite"
    corpus = tmp_path / "plumbing-only.txt"
    corpus.write_text(
        "\n".join(
            f"Case {i}: If A precedes B and B precedes C, A precedes C. "
            f"Compare [{i}, ({i + 1}, {i + 2})] with [({i + 2}, {i + 1}), {i}]. "
            "The opening and closing positions delimit a nested sequence."
            for i in range(64)
        )
    )
    unit = UnitRef(kind="mlp", layer=0, index=0)
    harness = QwenHarness.load(settings.model, settings.worker)
    index = ActivationIndex(settings.paths.activations)
    graph = GraphDB(settings.paths.circuits)
    client = ScientistClient(settings.codex)
    try:
        async with MPSBatchWorker(harness, settings.worker) as worker:
            await calibrate(worker, index, corpus, [unit], settings)
            run = graph.start_run(
                {"purpose": "synthetic plumbing smoke; not scientific validation"}
            )
            result = await Swarm(settings, client, worker, index, graph, run).run_units([unit])
            assert not result["errors"], result["errors"]
            assert graph.states(run)[unit.key] in {"verified", "falsified"}
            assert len(graph.attempts(run, unit)) >= 1
            assert client.metrics.calls >= 4
            print(
                f"\nSmoke result: {graph.states(run)[unit.key]}; "
                f"Codex calls={client.metrics.calls}; input tokens={client.metrics.input_tokens}"
            )
    finally:
        await client.close()
        index.close()
        graph.close()
