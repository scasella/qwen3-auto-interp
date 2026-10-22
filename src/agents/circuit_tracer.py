import hashlib
from itertools import product

import numpy as np

from src.harness.qwen_model import Probe, ProbeSpec
from src.schemas.contracts import CircuitEvidence, CircuitInterpretation, Gate


def classify_gate(table, quantiles, agreement_threshold):
    """A truth-table match requires every cell for the same context, including low cells."""
    keys = sorted(table)
    if keys not in (["0", "1"], ["00", "01", "10", "11"]):
        raise ValueError("A gate requires the complete unary or binary truth table")
    signatures = (
        {Gate.NOT: [1, 0]}
        if len(keys) == 2
        else {
            Gate.AND: [0, 0, 0, 1],
            Gate.OR: [0, 1, 1, 1],
        }
    )
    values = np.array([table[k] for k in keys]).T
    if values.ndim != 2 or len(values) < 5 or not np.isfinite(values).all():
        raise ValueError("A gate needs at least five finite, aligned contexts")
    states = np.where(values >= quantiles.p80, 1, np.where(values <= quantiles.p50, 0, -1))
    for gate, signature in signatures.items():
        agreement = float(np.all(states == signature, axis=1).mean())
        if agreement >= agreement_threshold:
            return gate, agreement
    return Gate.MODULATES, 0.0


class CircuitTracer:
    def __init__(self, client, worker, index, graph, run, settings):
        self.client, self.worker, self.index, self.graph = client, worker, index, graph
        self.run, self.settings = run, settings
        self._seen = set()

    async def trace(self, hypothesis):
        target = hypothesis.unit
        candidates = sorted(
            [h.unit for h in self.graph.verified(self.run) if h.unit.order < target.order],
            key=lambda u: (-u.layer, u.key),
        )[: self.settings.swarm.max_upstream_candidates]
        identity = (target.key, tuple(u.key for u in candidates))
        if not candidates or identity in self._seen:
            return []
        examples = [e for e in self.index.examples(target) if e["band"] == "positive"][:8]
        if len(examples) < 5:
            return []
        contexts = [(e["text"], tuple(e["token_ids"])) for e in examples]
        target_q = self.index.quantiles(target)
        if target_q.span < self.settings.thresholds.min_span:
            return []
        means = tuple((u, self.index.quantiles(u).mean) for u in candidates)
        spec = ProbeSpec((target,), attribution_target=target, attribution_baselines=means)
        gradients = await self.worker.map(
            [Probe(text, spec, token_ids=ids) for text, ids in contexts]
        )
        attribution = {
            u.key: float(np.mean([g.attribution.get(u.key, 0) for g in gradients]))
            for u in candidates
        }
        grad_act = {
            u.key: float(np.mean([g.grad_act.get(u.key, 0) for g in gradients])) for u in candidates
        }
        candidates.sort(key=lambda u: (-abs(attribution[u.key]), u.key))
        sources = candidates[:2]

        async def measure(clamps=()):
            plan = ProbeSpec((target,), clamps=tuple(clamps))
            values = await self.worker.map([Probe(t, plan, token_ids=ids) for t, ids in contexts])
            return [r.values[target.key][0] for r in values]

        baseline = await measure()
        zero, mean = {}, {}
        for source in sources:
            zero[source.key] = await measure(((source, 0.0),))
            mean[source.key] = await measure(((source, self.index.quantiles(source).mean),))
        effects = {
            u.key: max(float(np.mean(np.abs(np.array(v[u.key]) - baseline))) for v in (zero, mean))
            / target_q.span
            for u in sources
        }
        sources = [u for u in sources if effects[u.key] >= self.settings.thresholds.circuit_effect]
        if not sources:
            self._seen.add(identity)
            return []
        table = {}
        for bits in product((0, 1), repeat=len(sources)):
            clamps = [
                (u, self.index.quantiles(u).p95 if bit else self.index.quantiles(u).p05)
                for u, bit in zip(sources, bits)
            ]
            table["".join(map(str, bits))] = await measure(clamps)
        gate, agreement = classify_gate(table, target_q, self.settings.thresholds.gate_agreement)
        # A two-input label must include a measurable response to both inputs.
        if len(sources) == 2 and gate in (Gate.AND, Gate.OR):
            arr = {key: np.asarray(value) for key, value in table.items()}
            individual = [
                max(float(np.abs(arr[a] - arr[b]).mean()) for a, b in pairs) / target_q.span
                for pairs in ((("00", "10"), ("01", "11")), (("00", "01"), ("10", "11")))
            ]
            if min(individual) < self.settings.thresholds.circuit_effect:
                gate, agreement = Gate.MODULATES, 0.0
        payload = {
            "target": target.key,
            "sources": [u.key for u in sources],
            "deterministic_gate": gate.value,
            "truth_table": table,
            "attribution": attribution,
            "grad_act": grad_act,
            "baseline": baseline,
            "zero_ablation": zero,
            "mean_ablation": mean,
            "scope": "all source positions; final-token target; empirical low/high clamps",
        }
        interpretation = await self.client.ask("tracer", CircuitInterpretation, payload)
        if interpretation.gate != gate:
            # An agent may withhold a boolean claim, but it cannot upgrade measured evidence.
            interpretation = interpretation.model_copy(update={"gate": Gate.MODULATES})
            gate, agreement = Gate.MODULATES, 0.0
        eid = hashlib.sha256(
            (target.key + "|" + "|".join(u.key for u in sources)).encode()
        ).hexdigest()
        evidence = CircuitEvidence(
            id=eid,
            sources=sources,
            target=target,
            gate=gate,
            attribution={u.key: attribution[u.key] for u in sources},
            grad_act={u.key: grad_act[u.key] for u in sources},
            truth_table=table,
            zero_ablation={u.key: zero[u.key] for u in sources},
            mean_ablation={u.key: mean[u.key] for u in sources},
            baseline=baseline,
            effect=min(effects[u.key] for u in sources),
            agreement=agreement,
            interpretation=interpretation,
        )
        self.graph.add_circuit(self.run, evidence)
        self._seen.add(identity)
        return [evidence]
