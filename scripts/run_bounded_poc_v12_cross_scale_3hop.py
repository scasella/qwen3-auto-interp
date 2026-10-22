"""Exclusive v12 evidence, one committed freeze, sequential FIFO MPS execution."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from scripts.audit_bounded_poc_v12_cross_scale_3hop import audit_baseline, audit_path
from scripts.run_bounded_poc_v10_qwen17b_restart2 import execute, payloads
from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.source_paths import model17_path
from src.poc.v4_harness import V4Probe
from src.poc.v12_controls import planted
from src.poc.v12_cross_scale_3hop import (
    CANDIDATES,
    H0,
    LIMITS,
    MODES,
    ORIGINAL_U,
    REVISION,
    D,
    U,
    effect,
    evaluate,
    internal_delta,
    margin,
)
from src.poc.v12_harness import SPAN, SpanSpec, V12Harness
from src.poc.v12_templates import LEXICON, build, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v12-cross-scale-3hop"
PROTOCOL = ROOT / "docs/bounded-poc-v12-cross-scale-3hop-protocol.md"
V9 = Path(os.environ.get("QWEN_V9_WORKSPACE", str(ROOT)))
V10 = Path(os.environ.get("QWEN_V10_WORKSPACE", str(ROOT)))
V11 = Path(os.environ.get("QWEN_V11_WORKSPACE", str(ROOT)))
MODEL = model17_path()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read(name):
    return json.loads((OUT / name).read_text())


def arrays(name, value):
    assert all(np.isfinite(x).all() for x in value.values())
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez_compressed(f, **value)


def load(name):
    with np.load(OUT / (name + ".npz"), allow_pickle=False) as z:
        return dict(z)


def verify_files(files):
    for name, expected in files.items():
        assert sha(name) == expected, f"Protected bytes changed: {name}"


def prepare():
    sources = {}
    for root, name in [
        (V9, "bounded-poc-v9-json-hop0-sha256.json"),
        (V10, "bounded-poc-v10-qwen17b-release2-sha256.json"),
        (ROOT, "protected-sha256.json"),
        (V11, "bounded-poc-v11-question-hop0-sha256.json"),
    ]:
        inv = root / "release" / name
        record = json.loads(inv.read_text())
        for rel, digest in record["files"].items():
            sources[str(root / rel)] = digest
        sources[str(inv)] = sha(inv)
    manifest = json.loads(
        (V10 / "data/bounded-poc-v10-qwen17b-restart2/pre-calibration-manifest.json").read_text()
    )
    sources.update(manifest["external_sha256"])
    # Freeze all existing local evidence and reports, including retained failed attempts.
    for folder in ("data", "reports", "release", "incidents"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "hf-cache" not in p.parts:
                sources[str(p)] = sha(p)
    for root in (V9, V10, V11):
        for p in sorted((root / "release").rglob("*")):
            if p.is_file():
                sources[str(p)] = sha(p)
    verify_files(sources)
    retired = []

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("keys"), list) and all(isinstance(k, str) for k in x["keys"]):
                retired.append(dict(keys=x["keys"], text=x.get("text", "")))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for root in (ROOT, V9, V10, V11):
        for p in sorted((root / "data").rglob("*.json")):
            if "hf-cache" not in p.parts and "frozen-source" not in p.parts:
                walk(json.loads(p.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = build(tokenizer, retired)
    assert all(len(r["token_ids"]) - r["query_position"] <= SPAN for r in data["rows"])
    controls = planted()
    assert controls["passed"]
    OUT.mkdir(exist_ok=False)
    save("protected.json", dict(files=sources))
    save("retired-keys.json", dict(keys=sorted({k for r in retired for k in r["keys"]})))
    save("whitelist.json", sorted(LEXICON))
    save("dataset.json", data)
    save("known-controls.json", controls)
    print(
        "Prepared",
        len(data["keys"]),
        "keys; retired",
        len(read("retired-keys.json")["keys"]),
        "protected",
        len(sources),
        flush=True,
    )


def freeze():
    verify_files(read("protected.json")["files"])
    assert read("validation.json")["passed"]
    assert not list(OUT.glob("*started.json"))
    files = [p for folder in ("src", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock", PROTOCOL]
    files += list(OUT.glob("*.json")) + list(OUT.glob("*.log"))
    files += [p for p in MODEL.iterdir() if p.is_file()]
    save(
        "manifest.json",
        dict(
            utc=now(),
            revision=REVISION,
            upstream=U,
            original_upstream_diagnostic=ORIGINAL_U,
            downstream=D,
            hop0=H0,
            discovery_candidates=[],
            discovery_forwards=0,
            limits=LIMITS,
            main_microbatches=168,
            diagnostic_microbatches=60,
            python=platform.python_version(),
            torch=torch.__version__,
            sha256={str(p): sha(p) for p in sorted(files)},
        ),
    )


def verify(stage):
    verify_files(read("protected.json")["files"])
    verify_files(read("manifest.json")["sha256"])
    validate(read("dataset.json")["rows"])
    commit = read("registration-commit.json")["commit"]
    name = "data/bounded-poc-v12-cross-scale-3hop/manifest.json"
    assert (
        subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        == (OUT / "manifest.json").read_bytes()
    )


def indices(rows, axis, self_patch=False):
    index = {r["id"]: i for i, r in enumerate(rows)}
    return (
        np.arange(len(rows)) if self_patch else np.array([index[r["donors"][axis]] for r in rows])
    )


async def spatial(
    worker, rows, outputs, ub=None, sb=None, heads=(), axis="B", self_patch=False, upstream=U
):
    spec = SpanSpec(
        (),
        heads=heads,
        targets=upstream,
        last_layer=max(layer for layer, h in upstream),
        capture=sb is None,
    )
    donor = indices(rows, axis, self_patch)
    probes = []
    for i, r in enumerate(rows):
        payload = dict(query_position=r["query_position"])
        if sb is not None:
            payload.update(
                residual=sb["residual"][i],
                clean_v=ub["v"][i],
                head_values={
                    f"{layer_id}:{h}": sb["hop0"][donor[i], CANDIDATES.index((layer_id, h))]
                    for layer_id, h in heads
                },
            )
        probes.append(
            V4Probe(r["text"], tuple(r["token_ids"]), tuple(outputs), spec, payload=payload)
        )
    rs = await worker.map(probes)
    return {key: np.stack([r.spatial[key] for r in rs]) for key in rs[0].spatial}


async def baselines(worker, rows, outputs, prefix):
    base = await execute(worker, rows, outputs, D, full=True)
    ub = await execute(worker, rows, outputs, U, full=True)
    sb = await spatial(worker, rows, outputs)
    assert np.array_equal(base["logits"], ub["logits"])
    for i, r in enumerate(rows):
        q, n = r["query_position"], len(r["token_ids"])
        assert np.array_equal(sb["vspan"][i, :, : n - q], ub["v"][i, :, q:n])
    for name, a in [("D", base), ("U", ub), ("spatial", sb)]:
        arrays(prefix + "-" + name, a)
    audit_baseline(rows, base, ub, sb)
    return base, ub, sb


def u_contrast(rows, ub, axis, mode, self_patch=False, vspan=None):
    donor = indices(rows, axis, self_patch)
    delta = np.empty_like(ub["h"])
    for i, r in enumerate(rows):
        n, q = len(r["token_ids"]), r["query_position"]
        for h in range(3):
            c = {
                f: ub[f][i, h, :n] if f in ("p", "k", "v") else ub[f][i, h]
                for f in ("h", "p", "q", "k", "v")
            }
            d = {
                f: ub[f][donor[i], h, :n] if f in ("p", "k", "v") else ub[f][donor[i], h]
                for f in ("h", "p", "q", "k", "v")
            }
            if vspan is not None:
                d["v"] = c["v"].copy()
                d["v"][q:] = vspan[i, h, : n - q]
            delta[i, h] = internal_delta(c, d, q, mode)
    target = ub["h"] + delta
    return delta, target


async def relay(worker, rows, outputs, base, target, upstream=U):
    pl = payloads(rows, base, D)
    for i, p in enumerate(pl):
        p["upstream"] = {
            f"{layer_id}:{h}": target[i, k] for k, (layer_id, h) in enumerate(upstream)
        }
    mid = await execute(
        worker, rows, outputs, D, "mediate", pl, upstream=upstream, capture_u=upstream
    )
    end = await execute(
        worker, rows, outputs, D, "q_final", payloads(rows, base, D, donor_q=mid["q"]), active=D
    )
    result = {"D_" + k: v for k, v in end.items() if k != "logits"}
    result.update(
        logits=end["logits"],
        mediator_q=mid["q"],
        upstream_applied=mid["upstream"],
        mediator_applied=mid["applied"],
        mediator_sibling_error=mid["sibling_error"],
    )
    assert np.all(mid["sibling_error"] == 0) and np.all(end["sibling_error"] == 0)
    assert np.array_equal(mid["applied"], base["h"])
    expected = torch.from_numpy(target).to(torch.bfloat16).float().numpy()
    assert np.array_equal(mid["upstream"], expected)
    return result


async def path(
    worker, rows, outputs, base, ub, sb, axis, mode, heads=(), self_patch=False, upstream=U
):
    extra = {}
    if mode == "hop0_qspan":
        m = await spatial(worker, rows, outputs, ub, sb, heads, axis, self_patch, upstream)
        extra = {"H0_" + k: v for k, v in m.items()}
        delta, target = u_contrast(rows, ub, axis, "v_qspan", self_patch, m["vspan"])
    elif mode == "upstream_Q":
        target = ub["h"][indices(rows, axis, self_patch)].copy()
        delta = target - ub["h"]
    else:
        delta, target = u_contrast(rows, ub, axis, mode, self_patch)
    result = await relay(worker, rows, outputs, base, target, upstream)
    result.update(extra, u_delta=delta, u_target=target)
    if self_patch:
        assert np.array_equal(result["logits"], base["logits"])
        assert np.all(delta == 0) and np.all(result["D_delta"] == 0)
    return result


async def confirmation(worker, data, run):
    heads = H0
    summaries = {}
    diagnostics = {}
    for split in ("calibration", "primary", "vocabulary"):
        rows = [r for r in data["rows"] if r["split"] == split]
        outputs = data["output_ids"][split]
        prefix = f"confirm{run}-{split}"
        base, ub, sb = await baselines(worker, rows, outputs, prefix + "-base")
        paths = {}
        for axis in "BC":
            for mode in (*MODES, "upstream_Q", "hop0_qspan"):
                name = axis + ("_" + mode if mode in ("upstream_Q", "hop0_qspan") else "_U_" + mode)
                result = await path(worker, rows, outputs, base, ub, sb, axis, mode, heads)
                arrays(prefix + "-" + name, result)
                audit_path(rows, base, ub, sb, result, axis, mode, heads)
                paths[name] = result
        for mode in (*MODES, "upstream_Q", "hop0_qspan"):
            result = await path(worker, rows, outputs, base, ub, sb, "C", mode, heads, True)
            arrays(prefix + "-self-" + mode, result)
            audit_path(rows, base, ub, sb, result, "C", mode, heads, True)
        summaries[split] = evaluate(rows, base, paths)
        print(split, json.dumps(summaries[split]), flush=True)
        # Separate pre-registered original-0.6B-U diagnostic: 15 batches per 64 rows.
        du = await execute(worker, rows, outputs, ORIGINAL_U, full=True)
        ds = await spatial(worker, rows, outputs, upstream=ORIGINAL_U)
        arrays(prefix + "-diagnostic-U", du)
        arrays(prefix + "-diagnostic-spatial", ds)
        audit_baseline(rows, base, du, ds)
        labels = np.array([r["label"] for r in rows])
        diagnostic = {}
        for axis in "BC":
            for mode in ("v_qspan", "hop0_qspan"):
                result = await path(
                    worker, rows, outputs, base, du, ds, axis, mode, heads, upstream=ORIGINAL_U
                )
                arrays(prefix + "-diagnostic-" + axis + "-" + mode, result)
                audit_path(rows, base, du, ds, result, axis, mode, heads)
                diagnostic[axis + "_" + mode] = effect(
                    rows, labels * (margin(base) - margin(result))
                )
        result = await path(
            worker, rows, outputs, base, du, ds, "C", "hop0_qspan", heads, True, ORIGINAL_U
        )
        arrays(prefix + "-diagnostic-self-hop0_qspan", result)
        audit_path(rows, base, du, ds, result, "C", "hop0_qspan", heads, True)
        diagnostics[split] = diagnostic
        print("non_gating_original_U", split, json.dumps(diagnostic), flush=True)
    assert worker.batches == 228
    save(
        f"confirmation-{run}.json",
        dict(
            utc=now(),
            pid=os.getpid(),
            batches=worker.batches,
            main_batches=168,
            diagnostic_batches=60,
            splits=summaries,
            diagnostics=diagnostics,
        ),
    )


async def run(stage):
    verify(stage)
    assert os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0.70"
    assert os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") == "0.50"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    with open("/tmp/interp-swarm-mps.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(stage + "-started.json", dict(utc=now(), pid=os.getpid()))
        wc = WorkerConfig(batch_size=64, max_seq_len=128)
        harness = V12Harness.load(
            ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True), wc
        )
        async with MPSBatchWorker(harness, wc) as worker:
            await confirmation(worker, read("dataset.json"), stage[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "confirm1", "confirm2"])
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action.startswith("freeze"):
        freeze()
    else:
        asyncio.run(run(action))
