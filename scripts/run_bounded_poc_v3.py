"""Prospective fixed-coalition rescue test. All target forwards use one MPS queue."""

import argparse
import asyncio
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.analysis import rescue_stats_v2, task_stats
from src.poc.harness import HeadAction, V2Harness, V2Probe, V2Spec
from src.poc.rescue_v3 import (
    COALITION,
    HEADS,
    INTERFERER,
    LIMITS,
    RULES,
    block_stats,
    control_stats,
    known_controls,
    margin,
    select_rule,
)
from src.poc.templates_v3 import build_v3_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v3"
PROTOCOL = ROOT / "reports/bounded-poc-v3-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
LAYERS = tuple(sorted({int(h.split(":")[0]) for h in HEADS}))


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def arrays(name, **values):
    with (OUT / name).open("xb") as handle:
        np.savez_compressed(handle, **values)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected():
    record = read("retirement.json")
    for filename, expected in record["protected_sha256"].items():
        assert digest(ROOT / filename) == expected, f"Retired artifact changed: {filename}"
    for version in ("bounded-poc-v1", "bounded-poc-v2"):
        actual = {
            str(p.relative_to(ROOT)) for p in (ROOT / "data" / version).rglob("*") if p.is_file()
        }
        expected = {p for p in record["protected_sha256"] if p.startswith(f"data/{version}/")}
        assert actual == expected, "Retired artifact inventory changed"
    return record


def runtime_files():
    return [
        ROOT / "src/poc/harness.py",
        ROOT / "src/poc/analysis.py",
        ROOT / "src/poc/templates_v3.py",
        ROOT / "src/poc/rescue_v3.py",
        ROOT / "src/poc/review_v3.py",
        ROOT / "src/poc/review.py",
        ROOT / "src/schemas/contracts.py",
        ROOT / "src/config.py",
        ROOT / "src/harness/qwen_model.py",
        ROOT / "src/harness/mps_queue.py",
        ROOT / "src/agents/client.py",
        ROOT / "src/agents/prompts.py",
        Path(__file__).resolve(),
        ROOT / "scripts/audit_bounded_poc_v3.py",
        ROOT / "tests/test_poc_v3.py",
    ]


def prepare():
    from transformers import AutoTokenizer

    protected()
    retired = [
        r
        for v in ("bounded-poc-v1", "bounded-poc-v2")
        for r in json.loads((ROOT / f"data/{v}/dataset.json").read_text())["rows"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / f"data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/{REVISION}"),
        local_files_only=True,
    )
    payload = build_v3_dataset(tokenizer, retired)
    save("dataset.json", payload)
    save("sanity.json", payload["sanity"])
    controls = known_controls(payload)
    assert payload["sanity"]["passed"] and controls["passed"]
    save("known-controls.json", controls)
    print(
        {"rows": len(payload["rows"]), "sanity": payload["sanity"], "controls": controls["passed"]},
        flush=True,
    )


def registration(final=False):
    protected()
    assert read("sanity.json")["passed"] and read("known-controls.json")["passed"]
    files = runtime_files() + [
        PROTOCOL,
        OUT / "dataset.json",
        OUT / "sanity.json",
        OUT / "known-controls.json",
        OUT / "retirement.json",
    ]
    if final:
        verify_registration("discovery-registration.json")
        assert not list(OUT.glob("confirm*-started.json")), "Held-out execution already started"
        assert read("validation.json")["all_passed"]
        files += [
            OUT / "discovery-registration.json",
            OUT / "discovery.json",
            OUT / "validation.json",
        ]
        files += sorted(OUT.glob("discovery-*.npz"))
    manifest = {
        "version": "bounded-poc-v3",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "final_pre_heldout" if final else "pre_discovery",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "device": "mps",
        "dtype": "bfloat16",
        "seed": 42,
        "batch_size": 64,
        "max_seq_len": 128,
        "limits": LIMITS,
        "head_index": HEADS,
        "coalition": COALITION,
        "interferer": INTERFERER,
        "rules": {
            "candidates": RULES,
            "selection": "zero discovery regressions; max correct count, median gain, lexicographic id; else identity",
            "heads_reselected": False,
            "late_only_comparator_eligible": False,
            "heldout_runs": 2,
            "groups": {"discovery": 8, "primary": 16, "vocabulary": 8},
            "baseline_primary_required": 116,
            "baseline_discovery_required": 58,
            "baseline_vocabulary_required": 58,
            "overall": "discovery and primary rescue pass, all unassisted prerequisites, primary coalition, vocabulary transfer, controls and exact repeatability",
        },
        "selected": read("discovery.json")["selection"] if final else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
    }
    name = "manifest.json" if final else "discovery-registration.json"
    save(name, manifest)
    print(name, digest(OUT / name), flush=True)


def verify_registration(name):
    manifest = read(name)
    assert manifest["limits"] == LIMITS
    assert tuple(manifest["coalition"]) == COALITION and manifest["interferer"] == INTERFERER
    for filename, expected in manifest["sha256"].items():
        assert digest(ROOT / filename) == expected, f"Frozen file changed: {filename}"
    protected()
    return manifest


def rows_for(split):
    return [r for r in read("dataset.json")["rows"] if r["split"] == split]


def partners(rows, same=False):
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    return np.array([by_id[r["same_answer_id" if same else "opposite_id"]] for r in rows])


def probes(rows, spec, donors=None, residual=None):
    outputs = read("dataset.json")["output_ids"]
    return [
        V2Probe(
            text=r["text"],
            token_ids=tuple(r["token_ids"]),
            output_ids=tuple(outputs["vocabulary" if r["split"] == "vocabulary" else "primary"]),
            spec=spec,
            donors={h: values[i] for h, values in (donors or {}).items()},
            residual_donor=residual[i] if residual is not None else None,
        )
        for i, r in enumerate(rows)
    ]


async def capture(worker, rows):
    result = await worker.map(probes(rows, V2Spec(LAYERS)))
    return {
        "logits": np.array([r.logits for r in result]),
        "heads": np.array([[r.heads[h] for h in HEADS] for r in result], dtype=np.float32),
        "residual": np.array([r.residual for r in result], dtype=np.float32),
    }


async def patch(
    worker, rows, base, heads=COALITION, mode="opposite", means=None, alpha=1.0, late=False
):
    actions, donors = [], {}
    for h in heads:
        layer, head = map(int, h.split(":"))
        actions.append(HeadAction(layer, head, "scale" if mode == "scale" else "replace", alpha))
        vector = base["heads"][:, HEADS.index(h)]
        if mode == "scale":
            donors[h] = np.tile(means[h], (len(rows), 1))
        elif mode == "self":
            donors[h] = vector
        else:
            donors[h] = vector[partners(rows, same=mode == "same")]
    if late:
        actions.append(HeadAction(27, 14, "mean"))
        donors[INTERFERER] = np.tile(means[INTERFERER], (len(rows), 1))
    result = await worker.map(probes(rows, V2Spec(LAYERS, actions=tuple(actions)), donors))
    return {
        "logits": np.array([r.logits for r in result]),
        "heads": np.array([[r.heads[h] for h in HEADS] for r in result], dtype=np.float32),
        "deltas": np.array(
            [[r.applied_deltas.get(h, np.zeros(128)) for h in HEADS] for r in result],
            dtype=np.float32,
        ),
    }


async def instrumentation(worker, rows, base):
    self_patch = await patch(worker, rows, base, HEADS, "self")
    donor = partners(rows)
    result = await worker.map(
        probes(rows, V2Spec(patch_residual=True), residual=base["residual"][donor])
    )
    return {
        "self": self_patch["logits"],
        "residual": np.array([r.logits for r in result]),
        "donor_logits": base["logits"][donor],
    }


async def discover(worker):
    rows = rows_for("discovery")
    base = await capture(worker, rows)
    arrays("discovery-baseline.npz", **base)
    controls = await instrumentation(worker, rows, base)
    arrays("discovery-controls.npz", **controls)
    means = {h: base["heads"][:, i].mean(0).tolist() for i, h in enumerate(HEADS)}
    candidates, measured = [], []
    for rule in RULES:
        raw = await patch(
            worker, rows, base, mode="scale", means=means, alpha=rule["alpha"], late=rule["late"]
        )
        measured.append(raw)
        candidates.append(
            rule | {"stats": rescue_stats_v2(rows, margin(base["logits"]), margin(raw["logits"]))}
        )
    comparator = await patch(worker, rows, base, (), "scale", means, late=True)
    arrays(
        "discovery-rescue.npz",
        logits=np.array([m["logits"] for m in measured]),
        deltas=np.array([m["deltas"] for m in measured]),
        heads=np.array([m["heads"] for m in measured]),
        late_only=comparator["logits"],
    )
    selection = {
        "coalition": COALITION,
        "interferer": INTERFERER,
        "rescue": select_rule(candidates),
    }
    record = {
        "registration_sha256": digest(OUT / "discovery-registration.json"),
        "task": task_stats(rows, margin(base["logits"])),
        "controls": control_stats(base, controls),
        "mean_vectors": means,
        "rescue_candidates": candidates,
        "selection": selection,
        "late_only": rescue_stats_v2(rows, margin(base["logits"]), margin(comparator["logits"])),
    }
    save("discovery.json", record)
    print("Discovery:", record["task"]["correct"], "/64; frozen rule:", selection, flush=True)
    for c in candidates:
        print(
            c["id"],
            c["stats"]["errors_fixed"],
            "fixed",
            c["stats"]["regressions"],
            "broken",
            c["stats"]["median_gain"],
            "gain",
            flush=True,
        )


async def confirm(worker, rep):
    discovery = read("discovery.json")
    rule, means = discovery["selection"]["rescue"], discovery["mean_vectors"]
    summaries = {}
    for split in ("primary", "vocabulary"):
        rows = rows_for(split)
        prefix = f"confirmation-{rep}-{split}"
        base = await capture(worker, rows)
        arrays(prefix + "-baseline.npz", **base)
        controls = await instrumentation(worker, rows, base)
        arrays(prefix + "-controls.npz", **controls)
        solos = [await patch(worker, rows, base, (h,)) for h in COALITION]
        effects = {"solos": np.array([s["logits"] for s in solos])}
        for key, mode in (("joint", "opposite"), ("same", "same"), ("self", "self")):
            effects[key] = (await patch(worker, rows, base, mode=mode))["logits"]
        arrays(prefix + "-effects.npz", **effects)
        rescue = await patch(
            worker,
            rows,
            base,
            COALITION if rule["alpha"] != 1 else (),
            "scale",
            means,
            rule["alpha"],
            rule["late"],
        )
        rescue["late_only"] = (await patch(worker, rows, base, (), "scale", means, late=True))[
            "logits"
        ]
        arrays(prefix + "-rescue.npz", **rescue)
        summaries[split] = block_stats(rows, base, controls, effects, rescue)
        s = summaries[split]
        print(
            f"Run{rep} {split}: baseline {s['task']['correct']}/{len(rows)}, rescue {s['rescue']['after']['correct']}/{len(rows)}; {s['rescue']['errors_fixed']} fixed/{s['rescue']['regressions']} broken; gain {s['rescue']['median_gain']:.9f}; coalition fraction {s['coalition']['effects']['fraction']:.6f}",
            flush=True,
        )
    save(
        f"confirmation-{rep}.json",
        {
            "manifest_sha256": digest(OUT / "manifest.json"),
            "selection_sha256": digest(OUT / "discovery.json"),
            "blocks": summaries,
        },
    )


async def execute(stage):
    name = "discovery-registration.json" if stage == "discover" else "manifest.json"
    verify_registration(name)
    if stage == "confirm2":
        assert (OUT / "confirmation-1.json").exists()
    save(
        stage + "-started.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "registration_sha256": digest(OUT / name),
        },
    )
    lock = OUT / "running.lock"
    with lock.open("x") as handle:
        handle.write(stage)
    try:
        cfg = WorkerConfig(batch_size=64)
        harness = await asyncio.to_thread(
            V2Harness.load, ModelConfig(revision=REVISION, local_files_only=True), cfg
        )
        async with MPSBatchWorker(harness, cfg) as worker:
            if stage == "discover":
                await discover(worker)
            else:
                await confirm(worker, int(stage[-1]))
            print("Microbatches:", worker.batches, flush=True)
    finally:
        lock.unlink()


async def main(stage):
    if stage == "prepare":
        prepare()
    elif stage == "register":
        registration()
    elif stage == "freeze":
        registration(True)
    else:
        await execute(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "register", "discover", "freeze", "confirm1", "confirm2"]
    )
    asyncio.run(main(parser.parse_args().stage))
