"""Bounded, fixed-membership prospective Q/K/V confirmation. No screening."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.v4_harness import TARGETS, V4Harness, V4Probe, V4Spec
from src.poc.v7_analysis import LIMITS, PATHS, evaluate
from src.poc.v7_controls import planted
from src.poc.v7_templates import build, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v7-capstone"
PROTOCOL = ROOT / "reports/bounded-poc-v7-capstone-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = ROOT / f"data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/{REVISION}"
TARGET_KEYS = ["19:5", "21:13", "22:9"]
UPSTREAM = ["17:1", "16:4", "15:4"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def arrays(name, data):
    with (OUT / name).open("xb") as f:
        np.savez_compressed(f, **data)


def margin(a):
    return a["logits"][:, 0].astype(float) - a["logits"][:, 1].astype(float)


def protected():
    items = json.loads((ROOT / "v7-protected.json").read_text())
    for item in items:
        for path in (Path(item["original"]), ROOT / item["copy"]):
            assert sha(path) == item["sha256"], f"Protected file changed: {path}"
    # Also verify the original pre-v4 paths, not merely the isolated copies.
    for inventory in ("v4-protected.json", "v5-protected.json", "v6-protected.json"):
        for item in json.loads((ROOT / inventory).read_text()):
            assert sha(Path(item["original"])) == item["sha256"]
    return len(items)


def prepare():
    OUT.mkdir(exist_ok=True)
    protected()
    retired = [
        r
        for version in ("v1", "v2", "v3", "upstream", "v4-confirm", "v5-pointer", "v6-final")
        for r in json.loads((ROOT / f"data/bounded-poc-{version}/dataset.json").read_text())["rows"]
    ]
    tok = transformers.AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = build(tok, retired)
    save("dataset.json", data)
    controls = planted()
    assert controls["passed"]
    save("known-controls.json", controls)
    print(
        "Prepared",
        len(data["rows"]),
        "prompts;",
        len(data["keys"]),
        "single-token keys",
        flush=True,
    )
    print("Lengths", sorted({len(r["token_ids"]) for r in data["rows"]}), flush=True)


def register():
    protected()
    assert read("validation.json")["passed"] and read("known-controls.json")["passed"]
    assert not any(OUT.glob("confirm*-started.json"))
    files = []
    for folder in ("src", "scripts", "tests"):
        files += sorted((ROOT / folder).rglob("*.py"))
    files += [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "v7-protected.json",
        ROOT / "v6-protected.json",
        ROOT / "v5-protected.json",
        ROOT / "v4-protected.json",
        PROTOCOL,
        OUT / "dataset.json",
        OUT / "known-controls.json",
        OUT / "validation.json",
    ]
    files += sorted(OUT.glob("*validation.log"))
    files += sorted(MODEL.glob("*.json")) + sorted(MODEL.glob("*.safetensors"))
    record = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        revision=REVISION,
        downstream=TARGET_KEYS,
        upstream=UPSTREAM,
        limits=LIMITS,
        seed=42,
        batch_size=64,
        max_seq_len=128,
        device="mps",
        dtype="bfloat16",
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
        ),
        axis_B_exception="User approved one query-key token substitution; context/length/value positions exact",
        sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
    )
    save("manifest.json", record)
    for p in files:
        if p.suffix in (".py", ".md"):
            dest = OUT / "frozen-source" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    print("Manifest SHA256", sha(OUT / "manifest.json"), flush=True)


def verify():
    m = read("manifest.json")
    assert m["limits"] == LIMITS
    assert m["upstream"] == UPSTREAM and m["downstream"] == TARGET_KEYS
    for name, expected in m["sha256"].items():
        assert sha(ROOT / name) == expected, name
    protected()
    validate(read("dataset.json")["rows"])


def pack(results, full=False, keep_k=False):
    out = dict(logits=np.array([r.logits for r in results], dtype=np.float32))
    fields = [
        "h",
        "p",
        "routed_p",
        "q",
        "applied",
        "reconstruction_error",
        "sibling_error",
        "delta",
    ]
    if full or keep_k:
        fields += ["k"]
    if full:
        fields += ["v"]
    for field in fields:
        values = []
        for r in results:
            heads = []
            for key in TARGET_KEYS:
                value = np.asarray(r.routing[key].get(field, np.zeros(128)), dtype=np.float32)
                if field in ("p", "routed_p", "k", "v"):
                    pad = ((0, 128 - len(value)),) + ((0, 0),) * (value.ndim - 1)
                    value = np.pad(value, pad)
                heads.append(value)
            values.append(heads)
        out[field] = np.array(values)
    if full:
        out["upstream"] = np.array(
            [[r.heads[k] for k in UPSTREAM] for r in results], dtype=np.float32
        )
    assert all(np.isfinite(a).all() for a in out.values())
    return out


def payloads(rows, base, axis="B", self_patch=False, pattern=None, donor_q=None):
    ids = {r["id"]: i for i, r in enumerate(rows)}
    result = []
    for i, r in enumerate(rows):
        j = i if self_patch else ids[r["donors"][axis]]
        n = len(r["token_ids"])

        def bundle(idx):
            return {
                key: {
                    f: base[f][idx, h, :n] if f in ("p", "k", "v") else base[f][idx, h]
                    for f in ("h", "p", "q", "k", "v")
                }
                for h, key in enumerate(TARGET_KEYS)
            }

        clean, donor = bundle(i), bundle(j)
        if pattern is not None:
            for h, key in enumerate(TARGET_KEYS):
                donor[key]["p"] = pattern[i, h, :n]
        if donor_q is not None:
            for h, key in enumerate(TARGET_KEYS):
                donor[key]["q"] = donor_q[i, h]
        result.append(
            dict(
                clean=clean,
                donor=donor,
                upstream={k: base["upstream"][j, h] for h, k in enumerate(UPSTREAM)},
                color_positions=r["color_positions"],
                context_end=r["context_end"],
            )
        )
    return result


async def execute(
    worker, rows, outputs, mode="capture", payload=None, active=(), upstream=(), full=False
):
    spec = V4Spec((15, 16, 17) if full else (), mode=mode, active=active, upstream=upstream)
    results = await worker.map(
        [
            V4Probe(
                r["text"],
                tuple(r["token_ids"]),
                tuple(outputs),
                spec,
                payload=None if payload is None else payload[i],
            )
            for i, r in enumerate(rows)
        ]
    )
    return pack(results, full, mode == "mediate")


async def confirm(worker, run):
    data = read("dataset.json")
    summary = dict(manifest_sha256=sha(OUT / "manifest.json"), splits={})
    members = tuple(tuple(map(int, k.split(":"))) for k in UPSTREAM)
    for split in ("calibration", "primary", "vocabulary"):
        rows = [r for r in data["rows"] if r["split"] == split]
        outputs = data["output_ids"][split]
        prefix = f"confirm{run}-{split}"
        base = await execute(worker, rows, outputs, full=True)
        arrays(prefix + "-baseline.npz", base)
        control = {}
        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            out = await execute(
                worker, rows, outputs, mode, payloads(rows, base, self_patch=True), TARGETS
            )
            arrays(prefix + "-self-" + mode + ".npz", out)
            control["self_" + mode] = float(np.max(abs(margin(out) - margin(base))))
        midself = await execute(
            worker,
            rows,
            outputs,
            "mediate",
            payloads(rows, base, self_patch=True),
            upstream=members,
        )
        endself = await execute(
            worker,
            rows,
            outputs,
            "q_final",
            payloads(rows, base, self_patch=True, donor_q=midself["q"]),
            TARGETS,
        )
        arrays(prefix + "-self-mediator.npz", midself)
        arrays(prefix + "-self-upstream.npz", endself)
        control["self_upstream_margin"] = float(np.max(abs(margin(endself) - margin(base))))
        control["self_upstream_pattern"] = float(np.max(abs(endself["routed_p"] - base["p"])))
        result = dict(
            accuracy=float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0)),
            controls=control,
        )
        if split != "calibration":
            paths = {}
            for axis, modes in PATHS.items():
                for mode in modes:
                    tag = f"{axis}_{mode}"
                    paths[tag] = await execute(
                        worker, rows, outputs, mode, payloads(rows, base, axis), TARGETS
                    )
                    arrays(prefix + "-" + tag + ".npz", paths[tag])
                diagnostic = {"A": "ov_color", "B": "q_final", "C": "q_final"}[axis]
                for head in TARGETS:
                    out = await execute(
                        worker, rows, outputs, diagnostic, payloads(rows, base, axis), (head,)
                    )
                    arrays(prefix + f"-solo-{head[0]}-{head[1]}-{axis}.npz", out)
            upstream_paths = {}
            for axis in ("B", "C"):
                mid = await execute(
                    worker, rows, outputs, "mediate", payloads(rows, base, axis), upstream=members
                )
                arrays(prefix + f"-{axis}-mediator.npz", mid)
                end = await execute(
                    worker,
                    rows,
                    outputs,
                    "q_final",
                    payloads(rows, base, axis, donor_q=mid["q"]),
                    TARGETS,
                )
                arrays(prefix + f"-{axis}_upstream_Q.npz", end)
                upstream_paths[axis] = end
            result["scientific"] = evaluate(rows, base, paths, upstream_paths)
        summary["splits"][split] = result
        print(f"Completed run {run} {split}: " + json.dumps(result), flush=True)
    summary["microbatches"] = worker.batches
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(f"confirmation-{run}.json", summary)


async def run(stage):
    verify()
    with open("/tmp/interp-swarm-mps.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            f"{stage}-started.json",
            dict(
                utc=datetime.now(timezone.utc).isoformat(),
                pid=os.getpid(),
                manifest_sha256=sha(OUT / "manifest.json"),
            ),
        )
        wc = WorkerConfig(batch_size=64)
        harness = await asyncio.to_thread(
            V4Harness.load,
            ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True),
            wc,
        )
        async with MPSBatchWorker(harness, wc) as worker:
            await confirm(worker, int(stage[-1]))
    verify()


async def main(stage):
    if stage == "prepare":
        prepare()
    elif stage == "freeze":
        register()
    elif stage == "review":
        from src.poc.review import review_v7

        await review_v7(
            OUT,
            dict(
                protocol=PROTOCOL.read_text(),
                results=read("results.json"),
                facts=read("facts.json"),
            ),
        )
    else:
        await run(stage)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "freeze", "confirm1", "confirm2", "review"])
    asyncio.run(main(p.parse_args().stage))
