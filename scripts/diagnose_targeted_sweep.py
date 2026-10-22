"""Audit the frozen v1 evidence; optional exact-token replay on Metal."""

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
from transformers import AutoTokenizer

from src.config import Settings
from src.harness.mps_queue import MPSBatchWorker
from src.harness.qwen_model import Probe, ProbeSpec, QwenHarness
from src.schemas.contracts import UnitRef

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/targeted-real-v1"
OUT = ROOT / "data/targeted-diagnosis"
MODEL = (
    ROOT
    / "data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)


def audit():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = json.loads((OLD / "results.json").read_text())
    db = duckdb.connect(str(OLD / "activations.duckdb"), read_only=True)
    records = []
    for unit in data["units"]:
        cal = db.execute(
            "SELECT e.band,e.token_position,e.activation,d.token_ids,d.pieces FROM examples e JOIN documents d ON e.doc_id=d.id WHERE e.unit=? ORDER BY e.id",
            [unit["unit"]],
        ).fetchall()
        attempts = []
        for attempt in unit["attempts"]:
            results = attempt["evaluation"]["results"]
            by_id = {r["case"]["id"]: r for r in results}
            pairs = []
            for r in results:
                c = r["case"]
                if c["battery"] != "SEMANTIC_STRIPPING":
                    continue
                positive = by_id[c["paired_positive_id"]]
                pids = tokenizer.encode(positive["case"]["text"], add_special_tokens=False)
                nids = tokenizer.encode(c["text"], add_special_tokens=False)
                pairs.append(
                    {
                        "positive": positive["case"]["text"],
                        "stripped": c["text"],
                        "positive_target": tokenizer.decode(pids[-1:]),
                        "stripped_target": tokenizer.decode(nids[-1:]),
                        "same_target_token": pids[-1] == nids[-1],
                        "length_ratio": len(nids) / len(pids),
                        "drop_fraction": (positive["activation"] - r["activation"])
                        / (unit["quantiles"]["p95"] - unit["quantiles"]["p05"]),
                        "below_p05": r["activation"] <= unit["quantiles"]["p05"],
                    }
                )
            lengths = [
                len(tokenizer.encode(r["case"]["text"], add_special_tokens=False)) for r in results
            ]
            attempts.append(
                {
                    "strike": attempt["strike"],
                    "rule": attempt["hypothesis"]["structural_rule"],
                    "rates": attempt["evaluation"]["rates"],
                    "pairs": pairs,
                    "lengths": lengths,
                    "delta_only_strip_passes": sum(p["drop_fraction"] >= 0.25 for p in pairs),
                    "mean_activation_by_battery": {
                        b: float(
                            np.mean([r["activation"] for r in results if r["case"]["battery"] == b])
                        )
                        for b in attempt["evaluation"]["rates"]
                    },
                }
            )
        bands = {}
        for band in ("positive", "boundary", "inhibitory"):
            rows = [r for r in cal if r[0] == band]
            bands[band] = {
                "lengths": [r[1] + 1 for r in rows],
                "targets": Counter(tokenizer.decode([r[3][r[1]]]) for r in rows).most_common(),
                "retokenization_mismatches": sum(
                    tokenizer.encode("".join(r[4][: r[1] + 1]), add_special_tokens=False)
                    != r[3][: r[1] + 1]
                    for r in rows
                ),
            }
        records.append({"unit": unit["unit"], "calibration": bands, "attempts": attempts})
    pairs = [p for r in records for a in r["attempts"] for p in a["pairs"]]
    lengths = [v for r in records for a in r["attempts"] for v in a["lengths"]]
    cal_lengths = [v for r in records for b in r["calibration"].values() for v in b["lengths"]]
    summary = {
        "pairs": len(pairs),
        "changed_target_token": sum(not p["same_target_token"] for p in pairs),
        "probe_median_tokens": float(np.median(lengths)),
        "example_median_tokens": float(np.median(cal_lengths)),
        "stripping_drop_ge_25pct": sum(p["drop_fraction"] >= 0.25 for p in pairs),
        "stripping_below_p05": sum(p["below_p05"] for p in pairs),
        "retokenization_mismatches": sum(
            b["retokenization_mismatches"] for r in records for b in r["calibration"].values()
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps({"summary": summary, "units": records}, indent=2))
    print(json.dumps(summary, indent=2))
    db.close()


async def replay():
    settings = Settings.load(OLD / "config.yaml")
    db = duckdb.connect(str(OLD / "activations.duckdb"), read_only=True)
    harness = await asyncio.to_thread(QwenHarness.load, settings.model, settings.worker)
    output = []
    async with MPSBatchWorker(harness, settings.worker) as worker:
        for (key,) in db.execute("SELECT unit FROM quantiles ORDER BY unit").fetchall():
            unit = UnitRef.parse(key)
            q = db.execute("SELECT p05,p95 FROM quantiles WHERE unit=?", [key]).fetchone()
            rows = db.execute(
                "SELECT e.id,e.band,e.token_position,e.activation,d.token_ids FROM examples e JOIN documents d ON e.doc_id=d.id WHERE e.unit=? ORDER BY e.id",
                [key],
            ).fetchall()
            spec = ProbeSpec((unit,))
            probes = [
                Probe("", spec, token_ids=tuple(ids[: pos + 1])) for _, _, pos, _, ids in rows
            ]
            measured = await worker.map(probes)
            for row, result in zip(rows, measured):
                eid, band, pos, old, _ = row
                new = result.values[key][0]
                output.append(
                    {
                        "id": eid,
                        "unit": key,
                        "band": band,
                        "stored": old,
                        "replayed": new,
                        "error_fraction_of_span": abs(new - old) / (q[1] - q[0]),
                        "tail_still_passes": new >= q[1]
                        if band == "positive"
                        else new <= q[0]
                        if band == "inhibitory"
                        else None,
                    }
                )
    summary = {
        "observations": len(output),
        "max_error_fraction": max(r["error_fraction_of_span"] for r in output),
        "median_error_fraction": float(np.median([r["error_fraction_of_span"] for r in output])),
        "tail_failures": sum(r["tail_still_passes"] is False for r in output),
    }
    (OUT / "replay.json").write_text(
        json.dumps({"summary": summary, "observations": output}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    assert summary["tail_failures"] == 0, (
        "Stored extreme examples no longer meet original P05/P95 gates"
    )
    assert summary["max_error_fraction"] < 0.05, "Replay error exceeds 5% of calibration span"


async def controls():
    settings = Settings.load(OLD / "config.yaml")
    db = duckdb.connect(str(OLD / "activations.duckdb"), read_only=True)
    harness = await asyncio.to_thread(QwenHarness.load, settings.model, settings.worker)
    output = []
    async with MPSBatchWorker(harness, settings.worker) as worker:
        for (key,) in db.execute("SELECT unit FROM quantiles ORDER BY unit").fetchall():
            unit = UnitRef.parse(key)
            spec = ProbeSpec((unit,))
            q = db.execute("SELECT p05,p95 FROM quantiles WHERE unit=?", [key]).fetchone()
            rows = db.execute(
                "SELECT e.id,e.band,e.token_position,e.activation,d.token_ids FROM examples e JOIN documents d ON e.doc_id=d.id WHERE e.unit=? ORDER BY e.id",
                [key],
            ).fetchall()
            probes = []
            for eid, band, pos, value, ids in rows:
                prefix = ids[:pos]
                random.Random(int(hashlib.sha256(eid.encode()).hexdigest()[:16], 16)).shuffle(
                    prefix
                )
                probes.extend(
                    [
                        Probe("", spec, token_ids=(ids[pos],)),
                        Probe("", spec, token_ids=tuple(prefix + [ids[pos]])),
                    ]
                )
            measured = await worker.map(probes)
            for i, (eid, band, pos, value, ids) in enumerate(rows):
                token_only = measured[2 * i].values[key][0]
                shuffled = measured[2 * i + 1].values[key][0]
                output.append(
                    {
                        "id": eid,
                        "unit": key,
                        "band": band,
                        "original": value,
                        "token_only": token_only,
                        "shuffled_prefix": shuffled,
                        "context_length": pos + 1,
                        "target_token": harness.tokenizer.decode([ids[pos]]),
                        "token_only_same_tail": token_only >= q[1]
                        if band == "positive"
                        else token_only <= q[0]
                        if band == "inhibitory"
                        else None,
                        "shuffled_same_tail": shuffled >= q[1]
                        if band == "positive"
                        else shuffled <= q[0]
                        if band == "inhibitory"
                        else None,
                    }
                )
    summary = {
        band: {
            "n": sum(r["band"] == band for r in output),
            "token_only_same_tail": sum(
                r["band"] == band and r["token_only_same_tail"] is True for r in output
            ),
            "shuffled_same_tail": sum(
                r["band"] == band and r["shuffled_same_tail"] is True for r in output
            ),
        }
        for band in ["positive", "inhibitory"]
    }
    (OUT / "controls.json").write_text(
        json.dumps({"summary": summary, "observations": output}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    if args.replay:
        asyncio.run(replay())
    elif args.controls:
        asyncio.run(controls())
    else:
        audit()
