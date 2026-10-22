"""Independent frozen follow-up. No old experiment files are modified."""

import argparse
import asyncio
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from src.agents.client import ScientistClient
from src.agents.prompts import SYSTEM_PROMPTS
from src.config import CodexConfig, ModelConfig, WorkerConfig
from src.followup.analysis import mechanisms
from src.followup.controls import known_controls
from src.followup.harness import MODES, PathHarness, PathProbe, PathSpec
from src.followup.templates import baselines, build
from src.harness.mps_queue import MPSBatchWorker
from src.poc.analysis import LIMITS, cross_validate, evaluate, fit_readout, predict
from src.poc.harness import CausalSpec
from src.poc.review import PROMPT, Review

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/lookup-followup-v1"
PROTOCOL = ROOT / "reports/lookup-followup-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_PATH = ROOT / "data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots" / REVISION


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x") as file:
        file.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arrays(name, **values):
    with (OUT / name).open("xb") as file:
        np.savez_compressed(file, **values)


def read(name):
    return json.loads((OUT / name).read_text())


def load(name):
    with np.load(OUT / name) as data:
        return {k: data[k] for k in data.files}


def now():
    return datetime.now(timezone.utc).isoformat()


def verify():
    m = read("manifest.json")
    for name, digest in m["sha256"].items():
        assert sha(ROOT / name) == digest, f"Frozen file changed: {name}"
    assert m["limits"] == LIMITS
    return m


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(MODEL_PATH), revision=REVISION, local_files_only=True
    )
    save("dataset.json", build(tokenizer))
    controls = known_controls()
    assert controls["passed"]
    save("known-controls.json", controls)
    print("Prepared", len(read("dataset.json")["rows"]), "tokenizer-validated examples", flush=True)


async def review(stage):
    SYSTEM_PROMPTS["poc_review"] = (
        PROMPT
        + "\nCURRENT FOLLOW-UP: The supplied separately registered protocol supersedes the previous task description. Review source-specific Q/K/V interventions on fixed head 19:5 with gold/silver values. Preserve all previous numeric gates and distinguish source routing from upstream computation. Source K/V are contextual; never equate routing with independently performing the whole lookup. Assess only supplied evidence."
    )
    payload = dict(stage=stage, protocol=PROTOCOL.read_text())
    if stage == "plan":
        payload.update(
            known_controls=read("known-controls.json"),
            examples=read("dataset.json")["rows"][:4],
            analysis_source=(ROOT / "src/followup/analysis.py").read_text(),
        )
    else:
        payload["results"] = read("results.json")
    client = ScientistClient(CodexConfig(max_processes=1, max_attempts=2))
    try:
        answer = await client.ask("poc_review", Review, payload)
        value = dict(review=answer.model_dump(), usage=asdict(client.metrics))
        save(f"review-{stage}.json", value)
        print(json.dumps(value, indent=2), flush=True)
    finally:
        await client.close()


def freeze():
    assert not list(OUT.glob("*-started.json"))
    assert read("known-controls.json") == known_controls()
    assert known_controls()["passed"]
    issues = read("review-plan.json")["review"]["critical_issues"]
    extra_files = []
    if issues:
        adjudication = read("review-plan-adjudication.json")
        assert adjudication["issues"] == issues
        assert adjudication["review_sha256"] == sha(OUT / "review-plan.json")
        data = read("dataset.json")
        for row in data["rows"]:
            assert sorted(row["token_ids"][p] for p in row["value_positions"]) == sorted(
                data["output_ids"]
            )
        first = data["rows"][0]
        assert first["value_positions"] == adjudication["actual_positions"] == [12, 20]
        assert all(first["token_ids"][p] not in data["output_ids"] for p in [10, 18])
        assert (
            adjudication["resolution"]
            == "Reviewer token-index claim disproved by tokenizer IDs and decoded offsets; no dataset change."
        )
        extra_files = [OUT / "review-plan-adjudication.json"]
    files = [
        *sorted((ROOT / "src/followup").glob("*.py")),
        ROOT / "src/poc/harness.py",
        ROOT / "src/poc/analysis.py",
        ROOT / "src/poc/review.py",
        ROOT / "src/harness/qwen_model.py",
        ROOT / "src/harness/mps_queue.py",
        ROOT / "src/config.py",
        ROOT / "src/agents/client.py",
        ROOT / "src/agents/prompts.py",
        ROOT / "tests/test_followup.py",
        Path(__file__).resolve(),
        PROTOCOL,
        OUT / "dataset.json",
        OUT / "known-controls.json",
        OUT / "review-plan.json",
    ]
    files += (
        extra_files + sorted(MODEL_PATH.glob("*.json")) + sorted(MODEL_PATH.glob("*.safetensors"))
    )
    save(
        "manifest.json",
        dict(
            registered_utc=now(),
            model="Qwen/Qwen3-0.6B",
            revision=REVISION,
            candidate="19:5",
            device="mps",
            dtype="bfloat16",
            seed=42,
            batch_size=64,
            max_seq_len=128,
            limits=LIMITS,
            versions=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                transformers=transformers.__version__,
            ),
            sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        ),
    )
    print("Frozen:", sha(OUT / "manifest.json"), flush=True)


def indices(rows, field):
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    return np.array([by_id[r[field]] for r in rows])


def margins(base):
    return base["logits"][:, 0] - base["logits"][:, 1]


def make_probes(rows, outputs, spec, donors=None, payloads=None):
    return [
        PathProbe(
            r["text"],
            tuple(r["token_ids"]),
            outputs,
            spec,
            None if donors is None else tuple(donors[i]),
            None if payloads is None else payloads[i],
        )
        for i, r in enumerate(rows)
    ]


def pack(results):
    def pad(value, width=128):
        arr = np.array(value)
        return np.pad(arr, ((0, width - len(arr)), *((0, 0) for _ in arr.shape[1:])))

    data = dict(
        logits=np.array([r.logits for r in results]),
        residual=np.array([r.residual for r in results]),
        heads=np.array([r.heads["19:5"] for r in results]),
    )
    for key in ("q", "k", "v", "attention", "upstream"):
        data[key] = np.array(
            [pad(r.path[key]) if key in ("k", "v", "attention") else r.path[key] for r in results]
        )
    assert all(np.isfinite(v).all() for v in data.values())
    return data


async def ordinary(worker, rows, outputs, donors, residual=False):
    spec = CausalSpec((), patch_head=None if residual else (19, 5), patch_residual=residual)
    result = await worker.map(make_probes(rows, outputs, spec, donors=donors))
    return np.array([r.logits for r in result])


async def instrument(worker, rows, outputs, base, name):
    opposite = indices(rows, "opposite_id")
    full = await ordinary(worker, rows, outputs, base["residual"][opposite], residual=True)
    sham = await ordinary(worker, rows, outputs, base["heads"])
    rebuilt = await worker.map(
        make_probes(rows, outputs, PathSpec((), mode="reconstruct"), payloads=[{} for _ in rows])
    )
    rebuild_logits = np.array([r.logits for r in rebuilt])
    sibling_error = 0.0
    head_error = 0.0
    for r in rebuilt:
        a, b = np.array(r.path["original_heads"]), np.array(r.path["patched_heads"])
        siblings = np.arange(16) != 5
        sibling_error = max(sibling_error, float(np.max(np.abs(a[siblings] - b[siblings]))))
        head_error = max(head_error, float(np.max(np.abs(a[5] - b[5]))))
    stats = dict(
        residual_max=float(np.max(np.abs(full - base["logits"][opposite]))),
        sham_max=float(np.max(np.abs(sham[:, 0] - sham[:, 1] - margins(base)))),
        reconstruction_max=float(
            np.max(np.abs(rebuild_logits[:, 0] - rebuild_logits[:, 1] - margins(base)))
        ),
        sibling_max=sibling_error,
        reconstructed_head_max=head_error,
    )
    stats["passed"] = (
        stats["residual_max"] <= LIMITS["residual_max"]
        and stats["sham_max"] <= LIMITS["sham_max"]
        and stats["reconstruction_max"] <= LIMITS["sham_max"]
        and sibling_error == 0.0
    )
    arrays(f"{name}-controls.npz", residual=full, sham=sham, reconstruction=rebuild_logits)
    return stats


async def run(stage):
    verify()
    assert not (OUT / f"{stage}-started.json").exists(), "Registered stage already attempted"
    calibration_hash = None if stage == "calibrate" else sha(OUT / "calibration.json")
    save(
        f"{stage}-started.json",
        dict(
            started_utc=now(),
            manifest_sha256=sha(OUT / "manifest.json"),
            calibration_sha256=calibration_hash,
        ),
    )
    split = "calibration" if stage == "calibrate" else "confirmation"
    data = read("dataset.json")
    rows = [r for r in data["rows"] if r["split"] == split]
    outputs = tuple(data["output_ids"])
    name = "calibration" if stage == "calibrate" else stage
    lock = OUT / "running.lock"
    with lock.open("x") as file:
        file.write(stage)
    try:
        wc = WorkerConfig(batch_size=64)
        harness = await asyncio.to_thread(
            PathHarness.load,
            ModelConfig(name=str(MODEL_PATH), revision=REVISION, local_files_only=True),
            wc,
        )
        async with MPSBatchWorker(harness, wc) as worker:
            base = pack(await worker.map(make_probes(rows, outputs, PathSpec((19,)))))
            arrays(f"{name}-baseline.npz", **base)
            control = await instrument(worker, rows, outputs, base, name)
            y = np.array([r["label"] for r in rows])
            behavior = float(np.mean(y * margins(base) > 0))
            record = dict(
                completed_utc=now(),
                manifest_sha256=sha(OUT / "manifest.json"),
                calibration_sha256=calibration_hash,
                model_accuracy=behavior,
                model_correct=int(np.sum(y * margins(base) > 0)),
                n=len(rows),
                model_by_format={
                    str(f): float(
                        np.mean(
                            [
                                y[i] * margins(base)[i] > 0
                                for i, r in enumerate(rows)
                                if r["format"] == f
                            ]
                        )
                    )
                    for f in sorted({r["format"] for r in rows})
                },
                instrument_controls=control,
                baselines=baselines(rows),
            )
            if stage == "calibrate":
                record.update(
                    readout=fit_readout(base["heads"], y),
                    upstream_readout=fit_readout(base["upstream"], y),
                    cv_accuracy=cross_validate(base["heads"], rows),
                )
                save("calibration.json", record)
                print(
                    json.dumps({k: v for k, v in record.items() if "readout" not in k}, indent=2),
                    flush=True,
                )
                return
            cal = read("calibration.json")
            opposite, same = indices(rows, "opposite_id"), indices(rows, "same_answer_id")
            measured = {}
            for kind, vectors in dict(
                opposite=base["heads"][opposite],
                same=base["heads"][same],
                mean=np.tile(cal["readout"]["mean"], (len(rows), 1)),
                sham=base["heads"],
            ).items():
                value = await ordinary(worker, rows, outputs, vectors)
                measured[kind] = value[:, 0] - value[:, 1]
            arrays(f"{name}-head.npz", **measured)
            record["evaluation"] = evaluate(
                rows,
                base["heads"],
                cal["readout"],
                margins(base),
                measured["opposite"],
                measured["mean"],
                measured["same"],
                measured["sham"],
                baselines(rows),
            )
            record["upstream_readout_accuracy"] = float(
                np.mean(predict(base["upstream"], cal["upstream_readout"]) == y)
            )
            paths, siblings = {}, {}
            query = indices(rows, "query_swap_id")
            for mode in MODES:
                payloads = []
                for i, row in enumerate(rows):
                    j = query[i] if mode == "q_query" else opposite[i]
                    n = len(row["token_ids"])
                    payloads.append(
                        dict(
                            q=base["q"][j].tolist(),
                            k=base["k"][j, :n].tolist(),
                            v=base["v"][j, :n].tolist(),
                            value_positions=row["value_positions"],
                            query_record=row["query_record"],
                            suffix_start=row["suffix_start"],
                        )
                    )
                outputs_path = await worker.map(
                    make_probes(rows, outputs, PathSpec((), mode=mode), payloads=payloads)
                )
                paths[mode] = np.array([r.margin for r in outputs_path])
                siblings[mode] = max(
                    float(
                        np.max(
                            np.abs(
                                np.array(r.path["original_heads"])[np.arange(16) != 5]
                                - np.array(r.path["patched_heads"])[np.arange(16) != 5]
                            )
                        )
                    )
                    for r in outputs_path
                )
                print(f"{stage}: measured {mode}", flush=True)
            arrays(f"{name}-paths.npz", **paths)
            record["path_sibling_max"] = siblings
            record["mechanism"] = mechanisms(
                rows,
                margins(base),
                paths,
                base["attention"],
                record["evaluation"]["statistics"]["clean_counterfactual_gap"],
            )
            save(f"{stage}.json", record)
            print(
                json.dumps(
                    dict(
                        behavior=behavior,
                        controls=control,
                        head_passed=record["evaluation"]["passed"],
                        mechanism=record["mechanism"],
                    ),
                    indent=2,
                ),
                flush=True,
            )
            print("Microbatches:", worker.batches, flush=True)
    finally:
        lock.unlink()


def report():
    verify()
    cal, a, b = read("calibration.json"), read("confirm1.json"), read("confirm2.json")
    assert all(r["calibration_sha256"] == sha(OUT / "calibration.json") for r in (a, b))
    comparisons = {}
    for suffix in ("baseline", "head", "paths", "controls"):
        first, second = load(f"confirm1-{suffix}.npz"), load(f"confirm2-{suffix}.npz")
        comparisons[suffix] = {k: float(np.max(np.abs(first[k] - second[k]))) for k in first}
    # Apply the margin requirement to every measured logit pair, not just each logit.
    margin_errors = []
    for suffix in ("baseline", "head", "paths", "controls"):
        first, second = load(f"confirm1-{suffix}.npz"), load(f"confirm2-{suffix}.npz")
        for key in first:
            if suffix == "baseline" and key != "logits":
                continue
            x, z = first[key], second[key]
            if x.ndim == 2 and x.shape[1] == 2:
                x, z = x[:, 0] - x[:, 1], z[:, 0] - z[:, 1]
            margin_errors.append(float(np.max(np.abs(x - z))))
    agreement = float(
        np.mean(
            np.array(a["evaluation"]["statistics"]["predictions"])
            == np.array(b["evaluation"]["statistics"]["predictions"])
        )
    )
    criteria = dict(
        calibration_cv=cal["cv_accuracy"] >= LIMITS["discovery_cv"],
        behavior=all(r["model_accuracy"] >= LIMITS["model_accuracy"] for r in (cal, a, b)),
        instrumentation=all(r["instrument_controls"]["passed"] for r in (cal, a, b)),
        siblings=all(max(r["path_sibling_max"].values()) == 0.0 for r in (a, b)),
        head=all(r["evaluation"]["passed"] for r in (a, b)),
        repeat=max(margin_errors) <= LIMITS["replication_max"]
        and agreement >= LIMITS["replication_agreement"],
        mechanism=a["mechanism"]["classification"]
        == b["mechanism"]["classification"]
        != "unresolved",
    )
    result = dict(
        manifest_sha256=sha(OUT / "manifest.json"),
        calibration_sha256=sha(OUT / "calibration.json"),
        criteria=criteria,
        formal_survivor=all(criteria.values()),
        head_survives_unchanged_gates=all(v for k, v in criteria.items() if k != "mechanism"),
        model_accuracy=[r["model_accuracy"] for r in (cal, a, b)],
        calibration_cv=cal["cv_accuracy"],
        repeat_max_margin_error=max(margin_errors),
        prediction_agreement=agreement,
        raw_array_repeat_errors=comparisons,
        confirmation=[a, b],
    )
    save("results.json", result)
    s = a["evaluation"]["statistics"]
    m = a["mechanism"]
    lines = [
        "# Head 19:5 separately preregistered lookup follow-up",
        "",
        f"**Formal causal explanation survives: {result['formal_survivor']}.**",
        f"Mechanism classification: **{m['classification']}**.",
        "",
        "New values gold/silver, new keys, and four new record formats; 64 calibration prompts and "
        "128 held-out prompts, repeated in two fresh model processes. The new classifier was fitted "
        "on calibration only; this is not zero-shot transfer of the old red/blue classifier.",
        "",
        f"Model correct: {cal['model_correct']}/{cal['n']} calibration; {a['model_correct']}/{a['n']} "
        f"confirmation; {b['model_correct']}/{b['n']} repeat. Original accuracy gate: 90%, without rounding.",
        f"Frozen head readout: {s['readout_accuracy']:.4%}; upstream pre-attention query readout: "
        f"{a['upstream_readout_accuracy']:.4%} (decodability only).",
        f"Whole-head opposite donor effect: {s['patch_median']:.6f} logits, "
        f"{s['patch_fraction']:.3%} of full counterfactual gap; direction {s['patch_direction_rate']:.3%}.",
        f"Mean ablation: {s['mean_median']:.6f}; same-answer specificity: {s['specificity_median']:.6f}.",
        f"Failed original head gates: {[k for k, v in a['evaluation']['checks'].items() if not v]}.",
        f"Overall gates: `{criteria}`.",
        "",
        "| Head-local path | Median correct-margin decline | Direction | Full-gap fraction | Group p | Path gates pass | Negligible |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, p in m["paths"].items():
        lines.append(
            f"| {name} | {p['median']:.6f} | {p['direction']:.2%} | {p['fraction']:.2%} | "
            f"{p['group_p']:.6g} | {p['passed']} | {p['negligible']} |"
        )
    lines += [
        "",
        f"Routing statistics: `{m['routing']}`.",
        "",
        f"Reconstruction maximum margin error: {a['instrument_controls']['reconstruction_max']:.6f}. "
        f"Sibling-head change: {max(a['path_sibling_max'].values())}. "
        f"Fresh-process maximum margin difference: {max(margin_errors):.6f}; "
        f"prediction agreement: {agreement:.2%}.",
        "",
        "A source-routing result identifies local selection and use of stored-value positions. "
        "It does not show that all lookup computation happens in this head: its Q/K/V inputs are already "
        "contextual. A query-mediated result identifies an upstream path, not a complete upstream circuit. "
        "Hybrid patches can create unusual internal states. Negative significance alone is not proof of absence. "
        "These are two-token next-token preferences on synthetic prompts, not unrestricted generation or natural-corpus generalization.",
        "",
        "[Frozen protocol](lookup-followup-protocol.md) · [Manifest](../data/lookup-followup-v1/manifest.json) · "
        "[Machine-readable results](../data/lookup-followup-v1/results.json) · "
        "[Executable dataset](../data/lookup-followup-v1/dataset.json)",
    ]
    (ROOT / "reports/lookup-followup-v1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "confirmation"}, indent=2))


async def main(stage):
    if stage == "prepare":
        prepare()
    elif stage == "freeze":
        freeze()
    elif stage == "report":
        report()
    elif stage.startswith("review-"):
        await review(stage.removeprefix("review-"))
    else:
        await run(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "review-plan",
            "freeze",
            "calibrate",
            "confirm1",
            "confirm2",
            "report",
            "review-results",
        ),
    )
    asyncio.run(main(parser.parse_args().stage))
