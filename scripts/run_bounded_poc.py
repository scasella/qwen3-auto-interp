"""Freeze, execute, and audit the bounded proof of concept without adaptive reruns."""

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
from src.poc.analysis import LIMITS, cross_validate, evaluate, fit_readout
from src.poc.controls import known_controls
from src.poc.harness import CausalHarness, CausalProbe, CausalSpec
from src.poc.review import review
from src.poc.templates import baseline_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v1"
PROTOCOL = ROOT / "reports/bounded-poc-protocol.md"
LAYERS = (3, 11, 19, 27)
HEADS = [f"{layer}:{h}" for layer in LAYERS for h in range(16)]
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def save(name, value):
    path = OUT / name
    if path.exists():
        raise ValueError(f"Refusing to overwrite recorded evidence: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_files():
    return [
        *sorted((ROOT / "src/poc").glob("*.py")),
        ROOT / "src/harness/qwen_model.py",
        ROOT / "src/harness/mps_queue.py",
        ROOT / "src/config.py",
        ROOT / "src/agents/client.py",
        ROOT / "src/agents/prompts.py",
        Path(__file__).resolve(),
    ]


def freeze():
    assert not (OUT / "discovery.json").exists(), "Already measured"
    controls = read("known-controls.json")
    assert controls["passed"]
    assert controls == known_controls(read("dataset.json")["rows"])
    plan_review = read("review-plan.json")
    # Critical design issues must be resolved before freezing, never waived silently.
    review_files = []
    if plan_review["review"]["critical_issues"]:
        adjudication = read("review-plan-adjudication.json")
        assert adjudication["issues"] == plan_review["review"]["critical_issues"]
        assert adjudication["review_sha256"] == sha(OUT / "review-plan.json")
        for item in adjudication["factual_checks"]:
            actual = controls
            for key in item["path"]:
                actual = actual[key]
            assert actual is item["expected"]
        assert adjudication["resolved"] is True
        review_files = [OUT / "review-plan-adjudication.json"]
    files = (
        runtime_files()
        + [
            PROTOCOL,
            OUT / "dataset.json",
            OUT / "known-controls.json",
            OUT / "review-plan.json",
        ]
        + review_files
    )
    if (OUT / "execution-repair.json").exists():
        files.append(OUT / "execution-repair.json")
    manifest = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "device": "mps",
        "dtype": "bfloat16",
        "seed": 42,
        "layers": LAYERS,
        "candidate_limit": 3,
        "discovery_batch_size": 16,
        "confirmation_batch_size": 64,
        "max_seq_len": 128,
        "limits": LIMITS,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
    }
    save("manifest.json", manifest)
    print("Frozen manifest SHA256:", sha(OUT / "manifest.json"), flush=True)


def verify_frozen():
    manifest = read("manifest.json")
    for name, expected in manifest["sha256"].items():
        assert sha(ROOT / name) == expected, f"Frozen input changed: {name}"
    assert manifest["limits"] == LIMITS
    return manifest


def dataset(split):
    data = read("dataset.json")
    return [r for r in data["rows"] if r["split"] == split], tuple(data["output_ids"])


def probes(rows, outputs, spec, donors=None):
    return [
        CausalProbe(
            r["text"],
            tuple(r["token_ids"]),
            outputs,
            spec,
            tuple(donors[i]) if donors is not None else None,
        )
        for i, r in enumerate(rows)
    ]


async def capture(worker, rows, outputs):
    results = await worker.map(probes(rows, outputs, CausalSpec(LAYERS)))
    return {
        "logits": np.array([r.logits for r in results]),
        "residual": np.array([r.residual for r in results]),
        "heads": np.array([[r.heads[h] for h in HEADS] for r in results]),
    }


def margin(base):
    return base["logits"][:, 0] - base["logits"][:, 1]


def indices(rows, donor_field):
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    return np.array([by_id[r[donor_field]] for r in rows])


async def patched(worker, rows, outputs, donor_vectors, head=None, residual=False):
    spec = CausalSpec((), tuple(map(int, head.split(":"))) if head is not None else None, residual)
    result = await worker.map(probes(rows, outputs, spec, donor_vectors))
    return np.array([r.logits for r in result])


async def instrument_controls(worker, rows, outputs, base):
    opposite = indices(rows, "opposite_id")
    full = await patched(worker, rows, outputs, base["residual"][opposite], residual=True)
    sham = await patched(worker, rows, outputs, base["heads"][:, 0], HEADS[0])
    residual_error = float(np.max(np.abs(full - base["logits"][opposite])))
    sham_error = float(np.max(np.abs((sham[:, 0] - sham[:, 1]) - margin(base))))
    return {
        "passed": residual_error <= LIMITS["residual_max"] and sham_error <= LIMITS["sham_max"],
        "residual_patch_max_logit_error": residual_error,
        "sham_max_margin_error": sham_error,
    }, {"residual_patch": full, "sham_patch": sham}


async def discover(worker):
    rows, outputs = dataset("discovery")
    baseline_path = OUT / "discovery-baseline.npz"
    if baseline_path.exists():
        repair = read("execution-repair.json")
        assert repair["retained_baseline_sha256"] == sha(baseline_path)
        with np.load(baseline_path) as saved:
            base = {k: saved[k] for k in saved.files}
        print("Resuming from the retained, hash-checked discovery baseline.", flush=True)
    else:
        base = await capture(worker, rows, outputs)
        np.savez_compressed(baseline_path, **base)
    control, raw = await instrument_controls(worker, rows, outputs, base)
    np.savez_compressed(OUT / "discovery-controls.npz", **raw)
    labels = np.array([r["label"] for r in rows])
    behavior = float(np.mean(labels * margin(base) > 0))
    scores = []
    scan_rows = rows[:16]  # exactly the first four registered groups
    scan_opposite = indices(scan_rows, "opposite_id")
    patch_outputs = []
    for i, head in enumerate(HEADS):
        x = base["heads"][:, i]
        cv = cross_validate(x, rows)
        model = fit_readout(x, labels)
        effect = None
        if control["passed"] and behavior >= LIMITS["model_accuracy"]:
            logit = await patched(worker, scan_rows, outputs, x[:16][scan_opposite], head)
            patch_outputs.append(logit)
            effect = float(
                np.median(labels[:16] * (margin(base)[:16] - (logit[:, 0] - logit[:, 1])))
            )
        scores.append(
            {
                "head": head,
                "cv_accuracy": cv,
                "discovery_patch_median": effect,
                "readout": model,
            }
        )
        if (i + 1) % 16 == 0:
            print(f"Screened {i + 1}/64 heads; task accuracy={behavior:.3f}", flush=True)
    if patch_outputs:
        np.savez_compressed(OUT / "discovery-patches.npz", logits=np.array(patch_outputs))
    eligible = [
        s
        for s in scores
        if s["cv_accuracy"] >= LIMITS["discovery_cv"] and s["discovery_patch_median"] is not None
    ]
    ranked = sorted(
        eligible,
        key=lambda s: (-s["discovery_patch_median"], *map(int, s["head"].split(":"))),
    )
    result = {
        "manifest_sha256": sha(OUT / "manifest.json"),
        "instrument_controls": control,
        "model_accuracy": behavior,
        "baselines": baseline_scores(rows),
        "all_candidates": scores,
        "selected": ranked[:3],
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "meaning": "Selection uses discovery only. Readouts and mean vectors are now frozen.",
    }
    save("discovery.json", result)
    print(
        json.dumps(
            {
                "behavior": behavior,
                "controls": control,
                "selected": [
                    {k: s[k] for k in ("head", "cv_accuracy", "discovery_patch_median")}
                    for s in ranked[:3]
                ],
                "selection_sha256": sha(OUT / "discovery.json"),
            },
            indent=2,
        ),
        flush=True,
    )


async def confirm(worker, replicate):
    selection = read("discovery.json")
    rows, outputs = dataset("confirmation")
    base = await capture(worker, rows, outputs)
    np.savez_compressed(OUT / f"confirmation-{replicate}-baseline.npz", **base)
    control, raw = await instrument_controls(worker, rows, outputs, base)
    np.savez_compressed(OUT / f"confirmation-{replicate}-controls.npz", **raw)
    labels = np.array([r["label"] for r in rows])
    behavior = float(np.mean(labels * margin(base) > 0))
    opposite_idx, same_idx = indices(rows, "opposite_id"), indices(rows, "same_answer_id")
    results = []
    for chosen in selection["selected"]:
        head = chosen["head"]
        vectors = base["heads"][:, HEADS.index(head)]
        donors = {
            "opposite": vectors[opposite_idx],
            "same_answer": vectors[same_idx],
            "mean": np.tile(chosen["readout"]["mean"], (len(rows), 1)),
            "sham": vectors,
        }
        measured = {}
        for name, values in donors.items():
            logits = await patched(worker, rows, outputs, values, head)
            measured[name] = logits[:, 0] - logits[:, 1]
        filename = f"confirmation-{replicate}-head-{head.replace(':', '-')}.npz"
        np.savez_compressed(OUT / filename, **measured)
        evaluation = evaluate(
            rows,
            vectors,
            chosen["readout"],
            margin(base),
            measured["opposite"],
            measured["mean"],
            measured["same_answer"],
            measured["sham"],
            baseline_scores(rows),
        )
        results.append({"head": head, "evaluation": evaluation, "raw": filename})
        print(
            f"Confirmation {replicate}, head {head}: "
            f"passed={evaluation['passed']}, failed="
            f"{[k for k, v in evaluation['checks'].items() if not v]}",
            flush=True,
        )
    save(
        f"confirmation-{replicate}.json",
        {
            "manifest_sha256": sha(OUT / "manifest.json"),
            "selection_sha256": sha(OUT / "discovery.json"),
            "model_accuracy": behavior,
            "instrument_controls": control,
            "candidates": results,
        },
    )


async def execute(stage):
    verify_frozen()
    filename = "discovery.json" if stage == "discover" else f"confirmation-{stage[-1]}.json"
    if (OUT / filename).exists():
        raise ValueError("This registered execution already completed")
    if stage != "discover":
        assert (OUT / "discovery.json").exists(), "Selection must precede confirmation"
    save(
        f"{stage}-started.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha(OUT / "manifest.json"),
            "meaning": "Single registered execution; do not silently rerun this stage.",
        },
    )
    # Exclusive process lock prevents accidental overlapping GPU workers.
    lock = OUT / "running.lock"
    with lock.open("x") as handle:
        handle.write(stage)
    try:
        batch = 16 if stage == "discover" else 64
        config = ModelConfig(revision=REVISION, local_files_only=True)
        worker_config = WorkerConfig(batch_size=batch)
        harness = await asyncio.to_thread(CausalHarness.load, config, worker_config)
        async with MPSBatchWorker(harness, worker_config) as worker:
            if stage == "discover":
                await discover(worker)
            else:
                await confirm(worker, int(stage[-1]))
            print(f"Completed {worker.batches} microbatches.", flush=True)
    finally:
        lock.unlink()


def report():
    verify_frozen()
    d = read("discovery.json")
    a, b = read("confirmation-1.json"), read("confirmation-2.json")
    for run in (a, b):
        assert run["selection_sha256"] == sha(OUT / "discovery.json")
    base1 = np.load(OUT / "confirmation-1-baseline.npz")
    base2 = np.load(OUT / "confirmation-2-baseline.npz")
    base_error = float(np.max(np.abs(margin(base1) - margin(base2))))
    controls_pass = (
        read("known-controls.json")["passed"]
        and all(r["instrument_controls"]["passed"] for r in (d, a, b))
        and all(r["model_accuracy"] >= LIMITS["model_accuracy"] for r in (d, a, b))
    )
    candidates = []
    for first, second in zip(a["candidates"], b["candidates"], strict=True):
        assert first["head"] == second["head"]
        x, y = np.load(OUT / first["raw"]), np.load(OUT / second["raw"])
        error = max(base_error, max(float(np.max(np.abs(x[k] - y[k]))) for k in x.files))
        pred1 = first["evaluation"]["statistics"]["predictions"]
        pred2 = second["evaluation"]["statistics"]["predictions"]
        agreement = float(np.mean(np.array(pred1) == pred2))
        reproducible = (
            error <= LIMITS["replication_max"] and agreement >= LIMITS["replication_agreement"]
        )
        candidates.append(
            {
                "head": first["head"],
                "passed": controls_pass
                and first["evaluation"]["passed"]
                and second["evaluation"]["passed"]
                and reproducible,
                "replication_max_margin_error": error,
                "prediction_agreement": agreement,
                "replicates": [first["evaluation"], second["evaluation"]],
            }
        )
    result = {
        "manifest_sha256": sha(OUT / "manifest.json"),
        "selection_sha256": sha(OUT / "discovery.json"),
        "all_controls_passed": controls_pass,
        "model_accuracy": [r["model_accuracy"] for r in (d, a, b)],
        "candidates": candidates,
        "survivors": [c["head"] for c in candidates if c["passed"]],
        "scope": "Synthetic two-color lookup; final-query head answer-information transmission, not a complete lookup circuit or universal head function.",
    }
    save("results.json", result)
    lines = [
        "# Bounded causal proof of concept",
        "",
        f"**Verified heads: {result['survivors'] or 'none'}.** All required controls passed: {controls_pass}.",
        "",
        "The criteria, executable examples, source files, and checkpoint were frozen before any Qwen forward pass. "
        "We screened 64 heads on 64 discovery prompts, froze at most three candidates, and ran 128 held-out prompts "
        "with new keys and two new formats in two separately loaded model processes.",
        "",
        f"Two-answer task accuracy (discovery / confirmation / repeat): {result['model_accuracy']}. "
        "This compares the logits of ` red` and ` blue`; it does not measure unrestricted generation.",
        "",
        "| Head | Held-out readout accuracy | Opposite patch median | Mean ablation median | Specificity median | Failed gates (first / repeat) | Verified |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for c in candidates:
        s = c["replicates"][0]["statistics"]
        failed = [[k for k, v in r["checks"].items() if not v] for r in c["replicates"]]
        lines.append(
            f"| {c['head']} | {s['readout_accuracy']:.1%} | {s['patch_median']:.3f} | "
            f"{s['mean_median']:.3f} | {s['specificity_median']:.3f} | {failed} | {c['passed']} |"
        )
    lines += [
        "",
        "Patch and ablation values are changes in the correct-answer logit margin; positive means the intervention "
        "weakened the correct answer. Specificity is the difference between opposite-answer and same-answer donor effects.",
        "",
        "The registered token-count, final-token, position, and final-eight-token baselines cannot distinguish opposite "
        "assignments within a matched pair and have a 50% accuracy ceiling. First/last-mentioned-value baselines also "
        "score 50% by the crossed-query design. The executable lookup parser scores 100% as a positive control. "
        "These are limited surface baselines; no claim is made to beat every whole-prompt or nonlinear baseline.",
        "",
        "Analytic planted controls require the true lookup signal to pass and nuisance-only/reversed explanations to fail. "
        "Qwen controls independently check final-residual donor replay and self-patch invariance. "
        "Group-level sign and label-randomization tests account for the maximum three selected candidates.",
        "",
        "A passing head supports a narrow causal account: it transmits information about the value bound to the requested key "
        "at the query position and contributes to that answer's output preference. This does not prove that the head performs "
        "the lookup itself, identify the source tokens it reads, or establish a complete circuit. Mean ablation is off-manifold; "
        "donor patches can also create unusual combinations of internal states.",
        "",
        "Repeated confirmation uses the same held-out examples. It demonstrates computational repeatability, not independent "
        "replication on a second dataset. The examples are synthetic and use only two fixed answer tokens. "
        "This experiment does not rescue the original bipolar hypotheses or establish generalization to natural corpora.",
        "",
        "[Frozen protocol](bounded-poc-protocol.md) · "
        "[Manifest](../data/bounded-poc-v1/manifest.json) · "
        "[Raw results and replication statistics](../data/bounded-poc-v1/results.json) · "
        "[Discovery selection](../data/bounded-poc-v1/discovery.json) · "
        "[Executable dataset](../data/bounded-poc-v1/dataset.json)",
    ]
    (ROOT / "reports/bounded-poc-v1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


async def main(stage):
    if stage == "review-plan":
        rows, _ = dataset("discovery")
        result = await review(
            OUT,
            "plan",
            {
                "protocol": PROTOCOL.read_text(),
                "discovery_examples": rows[:4],
                "known_controls": read("known-controls.json"),
                "analysis_source": (ROOT / "src/poc/analysis.py").read_text(),
            },
        )
        print(json.dumps(result, indent=2))
    elif stage == "freeze":
        freeze()
    elif stage == "report":
        report()
    elif stage == "review-results":
        result = await review(
            OUT,
            "results",
            {
                "protocol": PROTOCOL.read_text(),
                "results": read("results.json"),
            },
        )
        print(json.dumps(result, indent=2))
    else:
        await execute(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "review-plan",
            "freeze",
            "discover",
            "confirm1",
            "confirm2",
            "report",
            "review-results",
        ],
    )
    asyncio.run(main(parser.parse_args().stage))
