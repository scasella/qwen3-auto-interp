"""Registered executable control for head 3:3; diagnostic only, never promotion."""

import argparse
import ast
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from src.agents.falsifier import grade
from src.config import Settings
from src.harness.indexer import Quantiles
from src.harness.mps_queue import MPSBatchWorker
from src.harness.qwen_model import Probe, ProbeSpec, QwenHarness
from src.schemas.contracts import TestSuite, UnitRef

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/targeted-diagnosis"
PLAN = OUT / "delimiter-control-plan.json"
MODEL = (
    ROOT
    / "data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
)


def prepare(bag_preserving=False):
    tok = Tokenizer.from_file(str(MODEL))
    cases = []
    checked = []
    labels = [
        "rainfall",
        "inventory",
        "pressure",
        "latency",
        "temperature",
        "distance",
        "duration",
        "volume",
        "brightness",
        "altitude",
        "velocity",
        "demand",
        "capacity",
        "humidity",
        "frequency",
    ]
    for i, label in enumerate(labels):
        prefix = (
            f"def prepare_{label}(values, weight):\n"
            f"    usable = [value for value in values if value >= {i}]\n"
            "    scaled = [value * weight for value in usable]\n"
            "    total = sum(scaled)\n"
            "    count = len(scaled)\n"
            "    mean = total / max(count, 1)\n"
            "    lower = min(scaled, default=0)\n"
            "    upper = max(scaled, default=0)\n"
            "    return [count, mean, "
        )
        positive = prefix + "[lower, upper]"
        stripped = prefix + "(lower, upper]"
        negative = prefix + "[lower, upper,"
        if bag_preserving:
            positive = prefix + "( [ lower, upper]"
            stripped = prefix + "[ ( lower, upper]"
            negative = prefix + "( [ lower, upper,"
        pids = tok.encode(positive, add_special_tokens=False).ids
        sids = tok.encode(stripped, add_special_tokens=False).ids
        assert len(pids) == len(sids) <= 128 and pids[-1] == sids[-1]
        assert sum(a != b for a, b in zip(positive, stripped)) == (
            2 if bag_preserving else 1
        ) and len(positive) == len(stripped)
        if bag_preserving:
            assert Counter(pids) == Counter(sids), (
                "Order control must preserve the exact token multiset"
            )
        ast.parse(positive + (")]" if bag_preserving else "]"))
        try:
            ast.parse(stripped + "]")
        except SyntaxError:
            pass
        else:
            raise AssertionError("Stripping did not disrupt delimiter match")
        ast.parse(negative + ("0])]" if bag_preserving else "0]]"))
        checked.append({"pair": i, "tokens": len(pids), "target_id": pids[-1]})
        for battery, text, case_id, partner, reason in [
            (
                "OOD_POSITIVE",
                positive,
                f"P{i:02}",
                None,
                "The final ] closes the inner [lower, upper component while the outer return list remains open.",
            ),
            (
                "SEMANTIC_STRIPPING",
                stripped,
                f"S{i:02}",
                f"P{i:02}",
                "The two opening delimiters swap order, preserving the exact token multiset; the same final ] no longer correctly closes the current nested component."
                if bag_preserving
                else "Only the matching inner opener changes from [ to (; the same final ] no longer closes that component correctly.",
            ),
            (
                "INHIBITORY_POLE",
                negative,
                f"N{i:02}",
                None,
                "The final comma continues the same inner component instead of closing it.",
            ),
        ]:
            assert len(tok.encode(text, add_special_tokens=False).ids) <= 128
            cases.append(
                {
                    "id": case_id,
                    "battery": battery,
                    "text": text,
                    "token_position": -1,
                    "paired_positive_id": partner,
                    "rationale": reason,
                }
            )
    suite = TestSuite(cases=cases)
    record = {
        "unit": "attention:3:3",
        "purpose": "Manual executable falsification of the strongest original closure candidate. Not a swarm survivor or holdout; code-only templated scope. All numeric gates unchanged.",
        "suite": suite.model_dump(mode="json"),
        "checks": checked,
    }
    if bag_preserving:
        record["purpose"] += (
            " Preserves exact token multiset; changes opener order only. Registered after the identity-change diagnostic, before measuring this order control."
        )
    encoded = json.dumps(record, indent=2)
    if PLAN.exists():
        assert PLAN.read_text() == encoded, "Frozen diagnostic plan differs"
    else:
        PLAN.write_text(encoded)
    print(
        {
            "prepared_cases": 45,
            "token_lengths": [r["tokens"] for r in checked],
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }
    )


async def measure():
    raw = PLAN.read_text()
    plan = json.loads(raw)
    suite = TestSuite.model_validate(plan["suite"])
    settings = Settings.load(ROOT / "data/targeted-real-v1/config.yaml")
    original = json.loads((ROOT / "data/targeted-real-v1/results.json").read_text())
    q = next(u["quantiles"] for u in original["units"] if u["unit"] == plan["unit"])
    unit = UnitRef.parse(plan["unit"])
    harness = await asyncio.to_thread(QwenHarness.load, settings.model, settings.worker)
    async with MPSBatchWorker(harness, settings.worker) as worker:
        measured = await worker.map([Probe(c.text, ProbeSpec((unit,))) for c in suite.cases])
    evaluation = grade(
        suite,
        [r.values[unit.key][0] for r in measured],
        Quantiles(**q),
        settings.thresholds,
        "discovery",
    )
    result = {
        "plan_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "purpose": plan["purpose"],
        "quantiles": q,
        "evaluation": evaluation.model_dump(mode="json"),
    }
    PLAN.with_name(PLAN.name.replace("-plan", "-results")).write_text(json.dumps(result, indent=2))
    print(evaluation.rates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--bag-preserving", action="store_true")
    args = parser.parse_args()
    if args.bag_preserving:
        PLAN = OUT / "delimiter-order-plan.json"
    if args.measure:
        asyncio.run(measure())
    else:
        prepare(args.bag_preserving)
