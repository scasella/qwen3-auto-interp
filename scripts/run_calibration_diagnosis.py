"""Retrospective downstream localization with all old success gates preserved."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import ModelConfig, WorkerConfig
from src.diagnosis.harness import TraceHarness, TraceSpec
from src.followup.harness import PathProbe, PathSpec
from src.harness.mps_queue import MPSBatchWorker
from src.poc.harness import CausalProbe, CausalSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/calibration-diagnosis-v1"
OLD = ROOT / "data/lookup-followup-v1"
REV = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = ROOT / "data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots" / REV
COMPONENTS = [(19, "mlp"), *[(layer, k) for layer in range(20, 28) for k in ("attention", "mlp")]]
LAYERS = tuple(range(19, 28))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(name):
    return json.loads((OUT / name).read_text())


def load(name):
    with np.load(OUT / name) as f:
        return {k: f[k] for k in f.files}


def arrays(name, **value):
    with (OUT / name).open("xb") as f:
        np.savez_compressed(f, **value)


def prepare():
    paths = [p for p in OLD.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    sources = [
        *sorted((ROOT / "src/diagnosis").glob("*.py")),
        ROOT / "src/followup/harness.py",
        ROOT / "src/poc/harness.py",
        ROOT / "src/poc/analysis.py",
        ROOT / "src/harness/qwen_model.py",
        ROOT / "src/harness/mps_queue.py",
        ROOT / "src/config.py",
        ROOT / "tests/test_calibration_diagnosis.py",
        ROOT / "scripts/analyze_calibration_diagnosis.py",
        Path(__file__).resolve(),
        ROOT / "reports/calibration-diagnosis-plan.md",
    ]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    snapshots = {}
    for p in sources:
        rel = p.relative_to(ROOT)
        target = OUT / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(p.read_bytes())
        snapshots[str(rel)] = sha(p)
    save(
        "plan-lock.json",
        dict(
            recorded_utc=datetime.now(timezone.utc).isoformat(),
            status="retrospective diagnostic; not confirmatory preregistration",
            original_evidence=hashes,
            source=snapshots,
        ),
    )
    print("Diagnostic plan and source snapshots locked; old evidence hashed.", flush=True)


def verify():
    reg = read("plan-lock.json")
    for name, digest in reg["original_evidence"].items():
        assert sha(ROOT / name) == digest, f"Old evidence changed: {name}"
    for name, digest in reg["source"].items():
        assert sha(ROOT / name) == digest, f"Diagnostic source changed: {name}"


def rows_for(split):
    data = json.loads((OLD / "dataset.json").read_text())
    return [r for r in data["rows"] if r["split"] == split], tuple(data["output_ids"])


def opposite(rows):
    idx = {r["id"]: i for i, r in enumerate(rows)}
    return np.array([idx[r["opposite_id"]] for r in rows])


def margin(base):
    return base["logits"][:, 0] - base["logits"][:, 1]


def probes(rows, outputs, spec, donors=None):
    return [
        CausalProbe(
            r["text"],
            tuple(r["token_ids"]),
            outputs,
            spec,
            None if donors is None else tuple(donors[i]),
        )
        for i, r in enumerate(rows)
    ]


async def trace(worker, rows, outputs):
    results = await worker.map(probes(rows, outputs, TraceSpec(LAYERS)))
    return dict(
        logits=np.array([r.logits for r in results]),
        states=np.array(
            [
                [
                    [r.trace[f"{layer}:{s}"] for s in ("before", "after_attention", "after_mlp")]
                    for layer in range(28)
                ]
                for r in results
            ]
        ),
        head_dot=np.array([[r.trace[f"{layer}:head_dot"] for layer in range(28)] for r in results]),
        heads=np.array(
            [[r.heads[f"{layer}:{h}"] for layer in LAYERS for h in range(16)] for r in results]
        ),
        components=np.array(
            [[r.components[f"{layer}:{kind}"] for layer, kind in COMPONENTS] for r in results]
        ),
    )


async def intervention(worker, rows, outputs, vectors, component=None, head=None):
    if component:
        spec = TraceSpec((), trace=False, component=tuple(component))
    else:
        spec = CausalSpec((), patch_head=tuple(head))
    results = await worker.map(probes(rows, outputs, spec, vectors))
    return np.array([r.margin for r in results])


def patch_vectors(base, rows, family, index, mean):
    vectors = base[family][:, index]
    return dict(
        zero=np.zeros_like(vectors),
        mean=np.tile(mean, (len(rows), 1)),
        opposite=vectors[opposite(rows)],
    )


async def source_test(worker, rows, outputs, mode):
    payloads = [
        dict(
            value_positions=r["value_positions"],
            query_record=r["query_record"],
            suffix_start=r["suffix_start"],
        )
        for r in rows
    ]
    results = await worker.map(
        [
            PathProbe(r["text"], tuple(r["token_ids"]), outputs, PathSpec((), mode=mode), payload=p)
            for r, p in zip(rows, payloads)
        ]
    )
    return np.array([r.margin for r in results])


def head_index(layer, head):
    return (layer - 19) * 16 + head


async def first(worker):
    baselines = {}
    for split in ("calibration", "confirmation"):
        rows, outputs = rows_for(split)
        base = await trace(worker, rows, outputs)
        arrays(f"{split}-baseline.npz", **base)
        old_name = "calibration-baseline.npz" if split == "calibration" else "confirm1-baseline.npz"
        old = np.load(OLD / old_name)
        error = float(np.max(np.abs(base["logits"] - old["logits"])))
        save(f"{split}-replay.json", dict(max_logit_error=error, passed=error <= 0.05))
        assert error <= 0.05, "Baseline replay failed"
        baselines[split] = base
        print(f"{split}: replay maximum logit error {error}", flush=True)
    rows, outputs = rows_for("calibration")
    base = baselines["calibration"]
    y = np.array([r["label"] for r in rows])
    m = margin(base)
    errors = y * m <= 0
    assert errors.sum() == 9
    cal_old = json.loads((OLD / "calibration.json").read_text())
    measured = {}
    for mode, vectors in patch_vectors(
        base, rows, "heads", head_index(19, 5), cal_old["readout"]["mean"]
    ).items():
        measured[f"head19:{mode}"] = await intervention(
            worker, rows, outputs, vectors, head=(19, 5)
        )
    for mode in ("reconstruct", "source_swap", "drop_relevant", "drop_distractor"):
        measured[f"source:{mode}"] = await source_test(worker, rows, outputs, mode)
    # One exact measured-vector self replacement for each module; no labels used.
    sham_errors = {}
    for j, (layer, kind) in enumerate(COMPONENTS):
        key = f"{layer}:{kind}"
        vectors = base["components"][:, j]
        sham = await intervention(worker, rows, outputs, vectors, component=(layer, kind))
        sham_errors[key] = float(np.max(np.abs(sham - m)))
        for mode, values in patch_vectors(base, rows, "components", j, vectors.mean(0)).items():
            measured[f"{key}:{mode}"] = await intervention(
                worker, rows, outputs, values, component=(layer, kind)
            )
        print(f"calibration: localized {key}", flush=True)
    arrays("calibration-interventions.npz", **measured)
    save("module-shams.json", sham_errors)
    assert max(sham_errors.values()) <= 0.05
    # Top three late heads by signed direct contribution, not intervention success.
    final_rms = base["states"][:, -1, -1, 1]
    contributions = y[:, None, None] * base["head_dot"] / final_rms[:, None, None]
    ranked_heads = sorted(
        [(layer, h) for layer in range(20, 28) for h in range(16)],
        key=lambda x: (float(np.mean(contributions[errors, x[0], x[1]])), *x),
    )
    chosen_heads = ranked_heads[:3]
    for layer, head in chosen_heads:
        j = head_index(layer, head)
        for mode, values in patch_vectors(
            base, rows, "heads", j, base["heads"][:, j].mean(0)
        ).items():
            measured[f"{layer}:head{head}:{mode}"] = await intervention(
                worker, rows, outputs, values, head=(layer, head)
            )
    arrays("calibration-heads.npz", **{k: v for k, v in measured.items() if ":head" in k})
    selected_modules = sorted(
        COMPONENTS,
        key=lambda x: (-float(np.median((y * (measured[f"{x[0]}:{x[1]}:mean"] - m))[errors])), *x),
    )[:2]
    selection = dict(
        heads=[list(x) for x in chosen_heads],
        components=[list(x) for x in selected_modules],
        rationale="Retrospective calibration selection under the locked diagnostic plan; no confirmatory status.",
    )
    save("selection.json", selection)
    rows2, outputs2 = rows_for("confirmation")
    b2 = baselines["confirmation"]
    confirm = {}
    for layer, kind in selected_modules:
        j = COMPONENTS.index((layer, kind))
        for mode, values in patch_vectors(
            b2, rows2, "components", j, base["components"][:, j].mean(0)
        ).items():
            confirm[f"{layer}:{kind}:{mode}"] = await intervention(
                worker, rows2, outputs2, values, component=(layer, kind)
            )
    for layer, head in chosen_heads:
        j = head_index(layer, head)
        for mode, values in patch_vectors(
            b2, rows2, "heads", j, base["heads"][:, j].mean(0)
        ).items():
            confirm[f"{layer}:head{head}:{mode}"] = await intervention(
                worker, rows2, outputs2, values, head=(layer, head)
            )
    arrays("confirmation-interventions.npz", **confirm)
    save(
        "run1-complete.json",
        dict(completed_utc=datetime.now(timezone.utc).isoformat(), batches=worker.batches),
    )


async def repeat(worker):
    selection = read("selection.json")
    cal = load("calibration-baseline.npz")
    for split in ("calibration", "confirmation"):
        rows, outputs = rows_for(split)
        base = await trace(worker, rows, outputs)
        arrays(f"repeat-{split}-baseline.npz", **base)
        measured = {}
        for layer, kind in selection["components"]:
            j = COMPONENTS.index((layer, kind))
            values = np.tile(cal["components"][:, j].mean(0), (len(rows), 1))
            measured[f"{layer}:{kind}:mean"] = await intervention(
                worker, rows, outputs, values, component=(layer, kind)
            )
        for layer, head in selection["heads"]:
            j = head_index(layer, head)
            values = np.tile(cal["heads"][:, j].mean(0), (len(rows), 1))
            measured[f"{layer}:head{head}:mean"] = await intervention(
                worker, rows, outputs, values, head=(layer, head)
            )
        if split == "calibration":
            mean = json.loads((OLD / "calibration.json").read_text())["readout"]["mean"]
            measured["head19:mean"] = await intervention(
                worker, rows, outputs, np.tile(mean, (len(rows), 1)), head=(19, 5)
            )
            measured["source:source_swap"] = await source_test(worker, rows, outputs, "source_swap")
        arrays(f"repeat-{split}-interventions.npz", **measured)
    save(
        "run2-complete.json",
        dict(completed_utc=datetime.now(timezone.utc).isoformat(), batches=worker.batches),
    )


def check_gpu_processes():
    # This read needs the same escalated local context as MPS, and must precede model load.
    listing = subprocess.check_output(["ps", "-axo", "pid,command"], text=True)
    import os
    import re

    for line in listing.splitlines():
        if not line.strip():
            continue
        pid, _, command = line.strip().partition(" ")
        if (
            pid.isdigit()
            and int(pid) != os.getpid()
            and re.match(r"(?:\S*/)?python(?:\d+(?:\.\d+)*)?\s", command)
            and any(
                x in command
                for x in ("run_bounded_poc", "run_lookup_followup", "run_calibration_diagnosis")
            )
        ):
            raise RuntimeError(f"Another Qwen execution process is active: {pid}")


async def run(stage):
    verify()
    check_gpu_processes()
    save(f"{stage}-started.json", dict(started_utc=datetime.now(timezone.utc).isoformat()))
    lock = OUT / "running.lock"
    with lock.open("x") as f:
        f.write(stage)
    try:
        wc = WorkerConfig(batch_size=64)
        harness = await asyncio.to_thread(
            TraceHarness.load, ModelConfig(name=str(MODEL), revision=REV, local_files_only=True), wc
        )
        async with MPSBatchWorker(harness, wc) as worker:
            await (first(worker) if stage == "run1" else repeat(worker))
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run1", "run2"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
    else:
        asyncio.run(run(stage))
