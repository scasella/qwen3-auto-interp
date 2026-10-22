import pytest
from conftest import FakeHarness, FakeScientist, make_hypothesis, make_suite

from src.agents.circuit_tracer import CircuitTracer, classify_gate
from src.agents.falsifier import grade
from src.harness.indexer import Quantiles
from src.harness.mps_queue import MPSBatchWorker
from src.harness.qwen_model import ProbeResult
from src.orchestrator.graph_db import GraphDB
from src.schemas.contracts import Gate, UnitRef


class GateIndex:
    def quantiles(self, unit):
        return Quantiles(10000, 0, -1.2, -1, 0, 0.8, 1, 1.2)

    def examples(self, unit):
        return [
            {"band": "positive", "text": f"context {i}", "token_ids": [i + 1]} for i in range(8)
        ]


class GateHarness(FakeHarness):
    def _execute(self, probes, owner):
        results = []
        for p in probes:
            clamps = dict(p.spec.clamps)
            value = 1.0 if all(v > 0.5 for v in clamps.values()) else -1.0
            values = {u.key: [value] for u in p.spec.units}
            attribution = {u.key: -1.0 for u, _ in p.spec.attribution_baselines}
            grad_act = {u.key: 1.0 for u, _ in p.spec.attribution_baselines}
            results.append(ProbeResult(values, [1], ["x"], attribution, grad_act))
        return results


async def test_joint_causal_gate_persists_with_shared_group(settings):
    index = GateIndex()
    graph = GraphDB(settings.paths.circuits)
    run = graph.start_run({"test": "causal"})
    sources = [UnitRef(kind="mlp", layer=0, index=i) for i in range(2)]
    target = UnitRef(kind="mlp", layer=1, index=0)
    try:
        for unit in [*sources, target]:
            discovery = grade(
                make_suite("discovery"),
                [1] * 15 + [-1] * 30,
                index.quantiles(unit),
                settings.thresholds,
                "discovery",
            )
            holdout = grade(
                make_suite("holdout"),
                [1] * 15 + [-1] * 30,
                index.quantiles(unit),
                settings.thresholds,
                "holdout",
            )
            graph.verify(run, make_hypothesis(unit), discovery, holdout)
        async with MPSBatchWorker(GateHarness(), settings.worker) as worker:
            tracer = CircuitTracer(FakeScientist(), worker, index, graph, run, settings)
            evidence = await tracer.trace(make_hypothesis(target))
            assert len(evidence) == 1
            assert evidence[0].gate == Gate.AND
            assert set(evidence[0].truth_table) == {"00", "01", "10", "11"}
            assert evidence[0].grad_act
            assert await tracer.trace(make_hypothesis(target)) == []
        exported = graph.graph(run)
        assert exported.number_of_edges() == 2
        assert len({d["group"] for _, _, d in exported.edges(data=True)}) == 1
        assert all(len(d["joint_sources"]) == 2 for _, _, d in exported.edges(data=True))
        invalid = evidence[0].model_copy(
            update={"sources": [UnitRef(kind="mlp", layer=0, index=10)]}
        )
        with pytest.raises(ValueError, match="verified"):
            graph.add_circuit(run, invalid)
    finally:
        graph.close()
    reopened = GraphDB(settings.paths.circuits)
    try:
        assert reopened.graph(run).number_of_edges() == 2
    finally:
        reopened.close()


def test_incomplete_truth_tables_are_not_boolean_evidence():
    with pytest.raises(ValueError, match="complete"):
        classify_gate({"00": [0] * 5, "11": [1] * 5}, GateIndex().quantiles(None), 0.8)
