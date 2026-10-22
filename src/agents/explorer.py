import asyncio
import hashlib

from src.agents.client import OutputContractError
from src.harness.indexer import compact_example, split_evidence
from src.schemas.contracts import HypothesisScreen, Lens, PrimitiveHypothesis


class Explorer:
    def __init__(self, client, index, record=None):
        self.client, self.index = client, index
        self.record = record or (lambda *_: None)

    async def explore(self, unit, failures):
        distribution = self.index.distribution(unit)
        examples, held_examples = split_evidence(distribution)
        distribution = distribution | {"examples": [compact_example(e) for e in examples]}
        evidence = {e["id"]: e["band"] for e in examples}
        if set(evidence.values()) != {"positive", "boundary", "inhibitory"}:
            raise ValueError("All three empirical bands are required")

        async def one(lens):
            def validate(hypothesis):
                if hypothesis.unit != unit or hypothesis.lens != lens:
                    raise OutputContractError(
                        f"Use exactly the assigned unit {unit.key} and lens {lens.value}"
                    )
                if any(eid not in evidence for eid in hypothesis.evidence_ids):
                    raise OutputContractError("Cite only supplied evidence identifiers")
                if {evidence[eid] for eid in hypothesis.evidence_ids} != set(evidence.values()):
                    raise OutputContractError(
                        "Cite evidence from each of the three activation bands"
                    )

            hypothesis = await self.client.ask(
                "explorer",
                PrimitiveHypothesis,
                {
                    "unit": unit.model_dump(),
                    "lens": lens.value,
                    "distribution": distribution,
                    "prior_discovery_failures": failures,
                },
                validate_result=validate,
            )
            return hypothesis

        tasks = [asyncio.create_task(one(lens)) for lens in Lens]
        try:
            hypotheses = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Opaque IDs prevent the band encoded in calibration evidence IDs leaking labels.
        held_examples = sorted(
            held_examples,
            key=lambda e: hashlib.sha256(("screen-v3/" + e["id"]).encode()).hexdigest(),
        )
        blinded = [
            {
                "id": f"E{i:02}",
                "text": e["text"],
                "target_token": e["target_token"],
                "token_count": len(e["token_ids"]),
            }
            for i, e in enumerate(held_examples)
        ]
        truth = {e["id"]: original["band"] for e, original in zip(blinded, held_examples)}
        required = {(i, e["id"]) for i in range(3) for e in blinded}

        def validate_screen(result):
            keys = [(p.candidate, p.example_id) for p in result.predictions]
            if len(keys) != len(required) or set(keys) != required:
                raise OutputContractError(
                    "Predict exactly one band for every candidate/example combination"
                )

        screened = await self.client.ask(
            "screen",
            HypothesisScreen,
            {
                "candidates": [
                    {
                        "candidate": i,
                        "structural_rule": h.structural_rule,
                        "positive_prediction": h.positive_prediction,
                        "boundary_prediction": h.boundary_prediction,
                        "inhibitory_prediction": h.inhibitory_prediction,
                    }
                    for i, h in enumerate(hypotheses)
                ],
                "unlabeled_examples": blinded,
            },
            validate_result=validate_screen,
        )
        scores = [
            sum(p.candidate == i and p.band == truth[p.example_id] for p in screened.predictions)
            for i in range(3)
        ]
        self.record(
            "hypothesis_screen",
            {
                "unit": unit.key,
                "candidates": [h.model_dump(mode="json") for h in hypotheses],
                "correct": scores,
                "total": len(blinded),
                "screen": screened.model_dump(mode="json"),
                "truth": truth,
            },
        )
        # This is repeated discovery screening, never independent confirmation.
        return [
            hypotheses[i]
            for i in sorted(range(3), key=lambda i: (-scores[i], hypotheses[i].lens.value))
        ]
