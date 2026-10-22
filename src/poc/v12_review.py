"""V12 prompt adapter using the canonical review.py CLI client and reference schema."""

import json
from dataclasses import asdict
from pathlib import Path

from src.poc.review import SYSTEM_PROMPTS, CodexConfig, ReviewV5, ScientistClient


async def review_v12(root: Path, payload):
    """Reference-only numeric claims: hydrate audited float64 JSON without transcription."""
    path = root / "review-results.json"
    if path.exists():
        raise ValueError("V12 review already recorded")
    # One canonical JSON roundtrip is shared by the prompt, saved evidence, and adjudicator.
    canonical = json.dumps(payload, allow_nan=False, sort_keys=True)
    payload = json.loads(canonical)
    with (root / "review-payload.json").open("x") as handle:
        handle.write(canonical + "\n")
    SYSTEM_PROMPTS[
        "poc_review"
    ] = """You are a skeptical reviewer of bounded-poc-v12-cross-scale-3hop.
Use only the prospective protocol, independent NumPy audit and canonical numeric facts.
No tools, filesystem, browsing, or model runs. Return the structured schema.
This is a one-freeze Qwen3-1.7B study with zero discovery. H0 coordinates come
from v11 0.6B; gating U was previously selected on 1.7B in v10. D coordinates
match both. A non-gating original-0.6B-U diagnostic is separate; do not call the
main circuit an unchanged twelve-head zero-shot transfer or infer homology.
Only H0 outputs at question positions before T change. The first pass clamps T
residuals clean; only U V at the query/suffix span is exported, then U output is
replayed through D Q_T with original D K/V. Direct donor U V_qspan includes T;
H0 mediation clamps that position. Do not conflate these interventions.
Calibration is only a historical split name; no v12 selection occurs; all scientific gates apply to all three splits. Do not
waive thresholds, failed accuracy or conservation. Interpret positive controlled
paths narrowly: no equality comparator, exclusivity, additive mediated fraction,
universality or complete lookup algorithm follows. Scientific and software
failures must remain separate. Frozen numeric audit verdicts are authoritative.
Effects concern two-token logit preferences. Planted controls validate software.
Two fresh processes establish numerical repeatability, not independent samples.
Attention is the median of per-prompt coalition means; only registered marginal
attention directions gate. Preserve scientific failures separately from software
or integrity failures. Your narrative cannot change a deterministic gate.
Use only literal keys from allowed_metric_refs, e.g. gates.gate0, never facts.gates.gate0.
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
