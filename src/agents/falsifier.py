import numpy as np

from src.agents.client import OutputContractError
from src.harness.indexer import compact_example, split_evidence
from src.harness.qwen_model import Probe, ProbeSpec
from src.schemas.contracts import (
    BatchReview,
    Battery,
    CaseResult,
    Evaluation,
    SuiteReview,
    TestBatch,
    TestSuite,
)


def grade(suite, activations, quantiles, thresholds, stage):
    if quantiles.span < thresholds.min_span:
        raise ValueError("Calibration is degenerate; this unit cannot be verified")
    by_id = {c.id: float(a) for c, a in zip(suite.cases, activations, strict=True)}
    if not np.isfinite(list(by_id.values())).all():
        raise ValueError("Non-finite test measurement")
    results = []
    predictions = []
    for case in suite.cases:
        value = by_id[case.id]
        positive = case.battery == Battery.OOD_POSITIVE
        passed = value >= quantiles.p95 if positive else value <= quantiles.p05
        if case.battery == Battery.SEMANTIC_STRIPPING:
            passed = passed and (
                by_id[case.paired_positive_id] - value >= thresholds.paired_drop * quantiles.span
            )
        results.append(CaseResult(case=case, activation=value, passed=passed))
        predictions.append(quantiles.p95 if positive else quantiles.p05)
    rates = {b.value: sum(r.passed for r in results if r.case.battery == b) / 15 for b in Battery}
    observed = np.array(activations)
    variance = float(np.square(observed - observed.mean()).sum())
    r_squared = (
        1 - float(np.square(observed - predictions).sum()) / variance if variance > 1e-12 else None
    )
    return Evaluation(
        stage=stage,
        passed=all(r >= thresholds.battery_pass_rate for r in rates.values()),
        rates=rates,
        r_squared=r_squared,
        results=results,
    )


class Falsifier:
    def __init__(self, client, worker, index, thresholds, record=None):
        self.client, self.worker, self.index, self.thresholds = client, worker, index, thresholds
        self.record = record or (lambda *_: None)

    async def evaluate(self, hypothesis, stage, excluded):
        distribution = self.index.distribution(hypothesis.unit)
        examples, _ = split_evidence(distribution)
        tokenizer = self.worker.harness.tokenizer
        minimums = {
            b: max(
                1, int(np.quantile([len(e["token_ids"]) for e in examples if e["band"] == b], 0.25))
            )
            for b in ("positive", "inhibitory")
        }
        cases = []
        reviews = []
        excluded = set(excluded)
        for start in range(1, 16, 3):
            numbers = list(range(start, start + 3))
            required = {f"{prefix}{i:02}" for prefix in ("P", "S", "N") for i in numbers}
            accepted_review = None

            async def validate(batch):
                nonlocal accepted_review
                if {c.id for c in batch.cases} != required:
                    raise OutputContractError(f"Use exactly these case IDs: {sorted(required)}")
                reused = [c.id for c in batch.cases if c.text in excluded]
                if reused:
                    raise OutputContractError(f"Replace reused text in cases {reused}")
                tokens = {
                    c.id: tokenizer.encode(c.text, add_special_tokens=False) for c in batch.cases
                }
                errors = []
                for case in batch.cases:
                    ids = tokens[case.id]
                    if not ids or len(ids) > self.worker.config.max_seq_len:
                        errors.append(f"{case.id}: invalid token length {len(ids)}")
                    if case.battery == Battery.SEMANTIC_STRIPPING:
                        other = tokens[case.paired_positive_id]
                        if not ids or not other or ids[-1] != other[-1] or len(ids) != len(other):
                            errors.append(
                                f"{case.id}/{case.paired_positive_id}: preserve identical final target token and position. Got {tokenizer.decode(ids[-1:])!r}/{tokenizer.decode(other[-1:])!r}, lengths {len(ids)}/{len(other)}. Change preceding context with equal-token substitutions."
                            )
                    else:
                        band = "positive" if case.battery == Battery.OOD_POSITIVE else "inhibitory"
                        if len(ids) < minimums[band]:
                            errors.append(
                                f"{case.id}: length {len(ids)} below minimum {minimums[band]}"
                            )
                if errors:
                    self.record(
                        "suite_rejected",
                        {
                            "unit": hypothesis.unit.key,
                            "stage": stage,
                            "batch": numbers,
                            "reason": "token_controls",
                            "errors": errors,
                            "suite": batch.model_dump(mode="json"),
                        },
                    )
                    raise OutputContractError("\n".join(errors))

                def validate_review(review):
                    keys = [c.case_id for c in review.cases]
                    if len(set(keys)) != 9 or set(keys) != set(tokens):
                        raise OutputContractError(
                            "Review each of the nine supplied case IDs exactly once"
                        )

                review = await self.client.ask(
                    "auditor",
                    BatchReview,
                    {
                        "hypothesis": hypothesis.model_dump(mode="json"),
                        "cases": [
                            c.model_dump(mode="json")
                            | {
                                "actual_target_token": tokenizer.decode(tokens[c.id][-1:]),
                                "actual_token_count": len(tokens[c.id]),
                            }
                            for c in batch.cases
                        ],
                    },
                    validate_result=validate_review,
                )
                self.record(
                    "batch_review",
                    {
                        "unit": hypothesis.unit.key,
                        "stage": stage,
                        "batch": numbers,
                        "review": review.model_dump(mode="json"),
                        "suite": batch.model_dump(mode="json"),
                    },
                )
                invalid = [f"{c.case_id}: {c.reason}" for c in review.cases if not c.valid]
                if invalid:
                    raise OutputContractError(
                        "Independent semantic review rejected cases. Repair without changing the hypothesis:\n"
                        + "\n".join(invalid)
                    )
                accepted_review = review

            batch = await self.client.ask(
                "falsifier",
                TestBatch,
                {
                    "hypothesis": hypothesis.model_dump(mode="json"),
                    "stage": stage,
                    "requested_cases": 9,
                    "case_numbers": numbers,
                    "instruction": "Produce ONLY three positives, their three stripping partners, and three opposing cases for the assigned numbers. This is one of five independently checked batches forming a 45-case study.",
                    "excluded_strings": sorted(excluded),
                    "max_tokens_per_string": self.worker.config.max_seq_len,
                    "calibration_evidence": [compact_example(e) for e in examples],
                    "minimum_tokens": minimums,
                    "paired_control": "Each P/S pair must end at exactly the SAME token ID at exactly the SAME token position. Use small context substitutions, never padding or gratuitous repetition.",
                },
                validate_result=validate,
            )
            assert accepted_review is not None
            cases.extend(batch.cases)
            reviews.extend(accepted_review.cases)
            excluded.update(c.text for c in batch.cases)
        suite = TestSuite(cases=cases)
        self.record(
            "suite_review",
            {
                "unit": hypothesis.unit.key,
                "stage": stage,
                "review": SuiteReview(cases=reviews).model_dump(mode="json"),
                "suite": suite.model_dump(mode="json"),
                "reviewed_in_batches": True,
            },
        )
        responses = await self.worker.map(
            [Probe(c.text, ProbeSpec((hypothesis.unit,)), c.token_position) for c in suite.cases]
        )
        return grade(
            suite,
            [r.values[hypothesis.unit.key][0] for r in responses],
            self.index.quantiles(hypothesis.unit),
            self.thresholds,
            stage,
        )
