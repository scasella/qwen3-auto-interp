"""Bounded advisory scientist review via codex exec, with numeric adjudication."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from src.agents.client import ScientistClient
from src.agents.prompts import SYSTEM_PROMPTS
from src.config import CodexConfig
from src.poc.review import PROMPT, ReviewV2


async def review_v3(root, payload):
    SYSTEM_PROMPTS["poc_review"] = (
        PROMPT
        + """
This is bounded-poc-v3: a prospective causal rescue experiment with the coalition
19:5 + 22:9 + 21:13 and late head 27:14 carried forward without head reselection.
Four bindings cross all four queries and both opposite balanced assignments: eight
related observations per group. All v1/v2 evaluation prompts are retired. Compare
fixed-intervention rescue to the late-only comparator before crediting the coalition.
Preserve every frozen baseline and rescue gate. Do not count fewer errors if any
already-correct prompt breaks. If the selected rule is identity, that is abstention,
not causal rescue. No held-out rule search is permitted. Computational repeats use
the same frozen prompts and are not independent samples.
Every numeric factual claim must also appear in numeric_claims, copied exactly from
the supplied facts mapping with its exact metric key and value. Never round values
to promote a failed gate. State explicitly whether formal rescue succeeded, while
distinguishing a causal information-transmission effect from useful task repair.
"""
    )
    client = ScientistClient(CodexConfig(max_processes=1, max_attempts=2))
    try:
        answer = await client.ask("poc_review", ReviewV2, payload)
        record = {"review": answer.model_dump(), "usage": asdict(client.metrics)}
        with (root / "review-results.json").open("x") as handle:
            json.dump(record, handle, indent=2)
        checks = [
            {
                "metric": c.metric,
                "value": c.value,
                "supported": c.metric in payload["facts"]
                and abs(payload["facts"][c.metric] - c.value) <= 1e-9,
            }
            for c in answer.numeric_claims
        ]
        adjudication = {
            "numeric_claims": checks,
            "all_supported": all(c["supported"] for c in checks),
            "meaning": "Numeric claims adjudicated; narrative remains advisory. Frozen numerical gates decide success.",
        }
        with (root / "review-adjudication.json").open("x") as handle:
            json.dump(adjudication, handle, indent=2)
        return record, adjudication
    finally:
        await client.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = root / "data/bounded-poc-v3"
    payload = {
        "facts": json.loads((out / "facts.json").read_text()),
        "audit": json.loads((out / "audit.json").read_text()),
        "protocol": (root / "reports/bounded-poc-v3-protocol.md").read_text(),
    }
    asyncio.run(review_v3(out, payload))
