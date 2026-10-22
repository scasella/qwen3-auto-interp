import asyncio
import json
from collections import deque

from src.agents.circuit_tracer import CircuitTracer
from src.agents.client import AgentConfigurationError
from src.agents.explorer import Explorer
from src.agents.falsifier import Falsifier
from src.schemas.contracts import Evaluation, PIReport, PrimitiveHypothesis


class Swarm:
    def __init__(self, settings, client, worker, index, graph, run):
        self.settings, self.client, self.worker = settings, client, worker
        self.index, self.graph, self.run = index, graph, run

        def record(kind, value):
            graph.event(run, kind, json.dumps(value))

        self.explorer = Explorer(client, index, record)
        self.falsifier = Falsifier(client, worker, index, settings.thresholds, record)
        self.tracer = CircuitTracer(client, worker, index, graph, run, settings)
        self.pending = deque()
        self.trace_queue = asyncio.Queue(maxsize=settings.swarm.pipelines)
        self.pi_event = asyncio.Event()
        self.completed = 0
        self.errors = []

    async def _unit(self, unit):
        q = self.index.quantiles(unit)
        if q.span < self.settings.thresholds.min_span:
            self.graph.state(self.run, unit, "inconclusive", "Degenerate calibration span")
            return
        examples = self.index.examples(unit)
        excluded = {e["text"] for e in examples} | {"".join(e["token_pieces"]) for e in examples}
        history = self.graph.attempts(self.run, unit)
        failures = []
        discovery = None
        hypothesis = None
        strike = 0
        # Replay measured evidence, so resume never grants fresh scientific strikes.
        for strike, raw_hypothesis, raw_evaluation in history:
            hypothesis = PrimitiveHypothesis.model_validate(raw_hypothesis)
            evaluation = Evaluation.model_validate(raw_evaluation)
            excluded.update(r.case.text for r in evaluation.results)
            if evaluation.stage == "holdout":
                if evaluation.passed and discovery is not None:
                    self.graph.verify(self.run, hypothesis, discovery, evaluation)
                    await self.trace_queue.put(hypothesis)
                else:
                    self.graph.state(self.run, unit, "falsified", "Held-out confirmation failed")
                return
            if evaluation.passed:
                discovery = evaluation
            else:
                failures.append({"hypothesis": raw_hypothesis, "evaluation": raw_evaluation})
        while discovery is None and strike < self.settings.swarm.max_strikes:
            strike += 1
            self.graph.state(self.run, unit, "exploring", f"Strike {strike}")
            candidates = await self.explorer.explore(unit, failures)
            hypothesis = candidates[0]
            self.graph.state(self.run, unit, "falsifying", f"Strike {strike}")
            evaluation = await self.falsifier.evaluate(hypothesis, "discovery", excluded)
            self.graph.record_attempt(self.run, unit, strike, hypothesis, evaluation)
            excluded.update(r.case.text for r in evaluation.results)
            if evaluation.passed:
                discovery = evaluation
            else:
                failures.append(
                    {
                        "hypothesis": hypothesis.model_dump(mode="json"),
                        "evaluation": evaluation.model_dump(mode="json"),
                    }
                )
        if discovery is None:
            self.graph.state(
                self.run,
                unit,
                "falsified",
                f"{self.settings.swarm.max_strikes}-strike discovery budget exhausted",
            )
            return
        self.graph.state(self.run, unit, "confirming")
        holdout = await self.falsifier.evaluate(hypothesis, "holdout", excluded)
        self.graph.record_attempt(self.run, unit, strike, hypothesis, holdout)
        if holdout.passed:
            self.graph.verify(self.run, hypothesis, discovery, holdout)
            await self.trace_queue.put(hypothesis)
        else:
            self.graph.state(self.run, unit, "falsified", "Held-out confirmation failed")

    async def _pipeline(self):
        while self.pending:
            unit = self.pending.popleft()
            try:
                await self._unit(unit)
            except AgentConfigurationError as exc:
                self.graph.state(self.run, unit, "error", str(exc))
                self.errors.append(f"{unit.key}: {exc}")
                self.pending.clear()
            except Exception as exc:
                self.graph.state(self.run, unit, "error", f"{type(exc).__name__}: {exc}")
                self.errors.append(f"{unit.key}: {exc}")
            finally:
                self.completed += 1
                if self.completed % self.settings.swarm.pi_every == 0:
                    self.pi_event.set()

    async def _trace_worker(self):
        while True:
            hypothesis = await self.trace_queue.get()
            try:
                if hypothesis is None:
                    return
                await self.tracer.trace(hypothesis)
            except Exception as exc:
                self.errors.append(f"Circuit trace {hypothesis.unit.key}: {exc}")
                self.graph.event(self.run, "trace_error", f"{hypothesis.unit.key}: {exc}")
            finally:
                self.trace_queue.task_done()

    async def _investigate_once(self):
        graph = self.graph.graph(self.run)
        verified = list(graph.nodes)[-128:]
        pending = [u.key for u in list(self.pending)[:128]]
        if not verified:
            return
        edges = [
            {"source": a, "target": b, "gate": d["gate"]}
            for a, b, d in graph.edges(data=True)
            if a in verified and b in verified
        ][-256:]

        async def investigate(view):
            try:
                report = await self.client.ask(
                    "pi",
                    PIReport,
                    {
                        "view": "cluster" if view == 0 else "prioritize",
                        "verified_units": verified,
                        "pending_units": pending,
                        "edges": edges,
                        "hypotheses": [graph.nodes[u]["hypothesis"] for u in verified],
                    },
                )
                if not set(report.prioritized_units) <= set(pending):
                    raise ValueError("PI invented pending units")
                if any(not set(c.units) <= set(verified) for c in report.clusters):
                    raise ValueError("PI cluster includes unverified units")
                self.graph.add_report(self.run, report)
                rank = {key: i for i, key in enumerate(report.prioritized_units)}
                self.pending = deque(sorted(self.pending, key=lambda u: rank.get(u.key, len(rank))))
            except Exception as exc:
                self.errors.append(f"PI: {exc}")
                self.graph.event(self.run, "pi_error", str(exc))

        await asyncio.gather(*(investigate(view) for view in range(self.settings.swarm.pi_workers)))

    async def _investigate(self):
        while True:
            await self.pi_event.wait()
            self.pi_event.clear()
            await self._investigate_once()

    async def run_units(self, units):
        previous = self.graph.states(self.run)
        self.pending = deque(
            u for u in units if previous.get(u.key) not in {"verified", "falsified", "inconclusive"}
        )
        tasks = [
            asyncio.create_task(self._trace_worker())
            for _ in range(self.settings.swarm.circuit_workers)
        ]
        pi_task = asyncio.create_task(self._investigate())
        pipelines = [
            asyncio.create_task(self._pipeline()) for _ in range(self.settings.swarm.pipelines)
        ]
        try:
            await asyncio.gather(*pipelines)
            await self.trace_queue.join()
            # Later completions can create new upstream candidates for earlier verified targets.
            for hypothesis in self.graph.verified(self.run):
                await self.trace_queue.put(hypothesis)
            await self.trace_queue.join()
            for _ in tasks:
                await self.trace_queue.put(None)
            await asyncio.gather(*tasks)
            pi_task.cancel()
            await asyncio.gather(pi_task, return_exceptions=True)
            if self.completed:
                await self._investigate_once()
        finally:
            for task in [*pipelines, *tasks, pi_task]:
                task.cancel()
            await asyncio.gather(*pipelines, *tasks, pi_task, return_exceptions=True)
        return self.graph.summary(self.run) | {"errors": self.errors}
