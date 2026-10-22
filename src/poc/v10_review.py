"""V10 prompt adapter using the canonical review.py CLI client and reference schema."""

import json
from dataclasses import asdict
from pathlib import Path

from src.poc.review import SYSTEM_PROMPTS, CodexConfig, ReviewV5, ScientistClient


async def review_v10(root: Path, payload):
    """Reference-only numeric claims: hydrate audited float64 JSON without transcription."""
    path = root / "review-results.json"
    if path.exists():
        raise ValueError("V10 review already recorded")
    # One canonical JSON roundtrip is shared by the prompt, saved evidence, and adjudicator.
    canonical = json.dumps(payload, allow_nan=False, sort_keys=True)
    payload = json.loads(canonical)
    with (root / "review-payload.json").open("x") as handle:
        handle.write(canonical + "\n")
    SYSTEM_PROMPTS["poc_review"] = """You are a skeptical reviewer of bounded-poc-v10-qwen17b.
Use only the prospective protocol, independent NumPy audit and canonical numeric facts.
No tools, filesystem, browsing, or model runs. Return the structured schema.
This is a two-freeze Qwen3-1.7B cross-scale study. Both three-head coalitions are selected
on calibration by the frozen greedy rules, then frozen before held-out primary and
vocabulary inference. Calibration is in-sample and cannot be described as independent
confirmation. All gates are conjunctive across the specified splits; do not waive,
round, or substitute thresholds. A failed prerequisite prevents full confirmation.
Pure-Q replay holds all original K/V clean, including final K. Interpret Axis C as
context sensitivity of the controlled Q route, not proof of a unique symbolic code,
universality, exclusive heads, complete lookup algorithm or additive mediation.
Effects concern two-token logit preferences. Planted controls validate software.
Two fresh processes establish numerical repeatability, not independent samples.
Attention is the median of per-prompt coalition means; only registered marginal
attention directions gate. Preserve scientific failures separately from software
or integrity failures. Your narrative cannot change a deterministic gate.
Select exact numeric metric keys in metric_refs; never transcribe numeric measurements
in narrative. The caller binds your references to canonical float64 facts. Include
references for conclusions and failed gates, and do not invent keys.
"""
    client = ScientistClient(CodexConfig(max_processes=1, max_attempts=2))
    try:
        response = await client.ask("poc_review", ReviewV5, payload)
        claims = [dict(metric=key, value=payload["facts"].get(key)) for key in response.metric_refs]
        record = dict(
            review=response.model_dump(),
            numeric_claims=claims,
            numeric_binding="Metric references selected by Luna; exact numeric values bound by code from audited float64 facts",
            usage=asdict(client.metrics),
        )
        with path.open("x") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
        checks = [
            dict(
                metric=c["metric"],
                claimed=c["value"],
                actual=payload["facts"].get(c["metric"]),
                supported=c["metric"] in payload["facts"]
                and c["value"] == payload["facts"][c["metric"]],
            )
            for c in claims
        ]
        with (root / "review-adjudication.json").open("x") as handle:
            json.dump(
                dict(
                    claims=checks,
                    all_numeric_claims_supported=bool(checks)
                    and all(c["supported"] for c in checks),
                    method="Reference binding, not model decimal transcription; narrative remains advisory",
                    verdict_source="Independent NumPy arrays; frozen gates",
                ),
                handle,
                indent=2,
                allow_nan=False,
            )
        return record
    finally:
        await client.close()
