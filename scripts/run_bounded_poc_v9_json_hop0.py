"""New v9 namespace, two committed freezes, bounded sequential target processes."""

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

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.v4_harness import V4Harness, V4Probe, V4Spec
from src.poc.v9_analysis import PATHS, effect, evaluate, margin, select_hop0, track2
from src.poc.v9_json_hop0 import build, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v9-json-hop0-restart1"
PROTOCOL = ROOT / "reports/bounded-poc-v9-json-hop0-restart1-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = ROOT / f"data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/{REVISION}"
D = ((19, 5), (21, 13), (22, 9))
U = ((17, 1), (16, 4), (15, 4))
H0 = tuple((layer, h) for layer in range(15) for h in range(16))


def keys(heads):
    return [f"{layer}:{h}" for layer, h in heads]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read(name):
    return json.loads((OUT / name).read_text())


def arrays(name, a):
    assert all(np.isfinite(v).all() for v in a.values())
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez_compressed(f, **a)


def protected():
    inv = json.loads((ROOT / "release/protected-sha256.json").read_text())["files"]
    for name, expected in inv.items():
        assert sha(ROOT / name) == expected, name
    aborted = json.loads(
        (ROOT / "incidents/v9-key-exclusion/protected-aborted-sha256.json").read_text()
    )["files"]
    for name, expected in aborted.items():
        assert sha(ROOT / name) == expected, name
    return len(inv)


def retired():
    out = []
    for p in sorted((ROOT / "data").rglob("dataset.json")):
        if p.parent == OUT:
            continue
        obj = json.loads(p.read_text())
        out.extend(r for r in obj.get("rows", []) if "keys" in r)
    return out


def prepare():
    protected()
    OUT.mkdir(exist_ok=False)
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = build(tok, retired())
    from scripts.audit_bounded_poc_v9_json_hop0 import validate_data

    validate_data(data)
    save("dataset.json", data)
    from src.poc.v9_controls import planted

    controls = planted()
    assert controls["passed"]
    save("known-controls.json", controls)


def freeze():
    protected()
    assert read("validation.json")["passed"]
    assert not list(OUT.glob("*started.json"))
    files = [p for folder in ("src", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
    files += [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        PROTOCOL,
        OUT / "dataset.json",
        OUT / "known-controls.json",
        OUT / "validation.json",
    ]
    files += [
        ROOT / "incidents/v9-key-exclusion/protected-aborted-sha256.json",
        ROOT / "data/bounded-poc-v9-json-hop0/dataset.json",
    ]
    files += list(OUT.glob("*validation.log"))
    files += list(MODEL.glob("*.json")) + list(MODEL.glob("*.safetensors"))
    save(
        "stage1.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            revision=REVISION,
            upstream=keys(U),
            downstream=keys(D),
            candidates=keys(H0),
            batch_size=64,
            max_seq_len=128,
            seed=42,
            python=platform.python_version(),
            torch=torch.__version__,
            sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        ),
    )


def verify(stage):
    protected()
    for name, digest in read("stage1.json")["sha256"].items():
        assert sha(ROOT / name) == digest, name
    validate(read("dataset.json")["rows"])
    from scripts.audit_bounded_poc_v9_json_hop0 import validate_data

    validate_data(read("dataset.json"))
    commit_file = "stage1-commit.json" if stage == "calibration" else "stage2-commit.json"
    commit = read(commit_file)["commit"]
    frozen = "stage1.json" if stage == "calibration" else "stage2.json"
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:data/bounded-poc-v9-json-hop0-restart1/{frozen}"], cwd=ROOT
    )
    assert committed == (OUT / frozen).read_bytes()
    if stage != "calibration":
        for name, digest in read("stage2.json")["sha256"].items():
            assert sha(OUT / name) == digest, name


def pack(results, targets, capture_h0=False):
    out = {"logits": np.asarray([r.logits for r in results], dtype=np.float32)}
    for field in (
        "h",
        "p",
        "q",
        "k",
        "v",
        "routed_p",
        "applied",
        "delta",
        "sibling_error",
        "reconstruction_error",
    ):
        values = []
        for r in results:
            heads = []
            for key in keys(targets):
                x = np.asarray(
                    r.routing[key].get(field, np.zeros_like(r.routing[key]["h"])), dtype=np.float32
                )
                if field in ("p", "k", "v", "routed_p"):
                    x = np.pad(x, ((0, 128 - len(x)),) + ((0, 0),) * (x.ndim - 1))
                heads.append(x)
            values.append(heads)
        out[field] = np.asarray(values)
    if capture_h0:
        out["hop0"] = np.asarray(
            [[r.heads[k] for k in keys(H0)] for r in results], dtype=np.float32
        )
    return out


async def execute(
    worker, rows, output_ids, targets=D, mode="capture", payload=None, upstream=(), capture_h0=False
):
    layers = tuple(sorted(set(range(15) if capture_h0 else ()) | {layer for layer, _ in upstream}))
    spec = V4Spec(
        layers,
        targets=targets,
        active=targets if mode not in ("capture", "mediate") else (),
        mode=mode,
        upstream=upstream,
    )
    rs = await worker.map(
        [
            V4Probe(
                r["text"],
                tuple(r["token_ids"]),
                tuple(output_ids),
                spec,
                payload=None if payload is None else payload[i],
            )
            for i, r in enumerate(rows)
        ]
    )
    packed = pack(rs, targets, capture_h0)
    if upstream:
        packed["upstream_applied"] = np.asarray(
            [[r.heads[k] for k in keys(upstream)] for r in rs], dtype=np.float32
        )
    return packed


def payloads(rows, base, targets, axis="C", self_patch=False, q=None, outputs=None, heads=()):
    index = {r["id"]: i for i, r in enumerate(rows)}
    result = []
    for i, r in enumerate(rows):
        j = i if self_patch else index[r["donors"][axis]]
        n = len(r["token_ids"])

        def bundle(idx):
            return {
                key: {
                    f: base[f][idx, h, :n] if f in ("p", "k", "v") else base[f][idx, h]
                    for f in ("h", "p", "q", "k", "v")
                }
                for h, key in enumerate(keys(targets))
            }

        clean, donor = bundle(i), bundle(j)
        if q is not None:
            for h, key in enumerate(keys(targets)):
                donor[key]["q"] = q[i, h]
        result.append(
            dict(
                clean=clean,
                donor=donor,
                color_positions=r["color_positions"],
                context_end=r["context_end"],
                upstream={}
                if outputs is None
                else {key: outputs[i, h] for h, key in enumerate(keys(heads))},
            )
        )
    return result


async def downstream(worker, rows, outputs, base, u_outputs, axis, prefix):
    pl = payloads(rows, base, D, axis, outputs=u_outputs, heads=U)
    mid = await execute(worker, rows, outputs, mode="mediate", payload=pl, upstream=U)
    arrays(prefix + "-D-mediator", mid)
    end = await execute(
        worker, rows, outputs, mode="q_final", payload=payloads(rows, base, D, axis, q=mid["q"])
    )
    arrays(prefix + "-end", end)
    return end


async def hop(worker, rows, outputs, base, ub, coalition, axis, prefix, self_patch=False):
    index = {r["id"]: i for i, r in enumerate(rows)}
    donor = np.arange(len(rows)) if self_patch else [index[r["donors"][axis]] for r in rows]
    h = ub["hop0"][donor][:, [H0.index(x) for x in coalition]]
    mid = await execute(
        worker,
        rows,
        outputs,
        U,
        "mediate",
        payloads(rows, ub, U, axis, outputs=h, heads=coalition),
        upstream=coalition,
    )
    arrays(prefix + "-U-mediator", mid)
    replay = await execute(
        worker, rows, outputs, U, "q_final", payloads(rows, ub, U, axis, q=mid["q"])
    )
    arrays(prefix + "-U-replay", replay)
    return await downstream(worker, rows, outputs, base, replay["applied"], axis, prefix)


async def internal(worker, rows, outputs, base, ub, axis, mode, prefix, self_patch=False):
    pl = payloads(rows, ub, U, axis, self_patch=self_patch)
    action = mode
    if mode in ("kv_context", "k_all", "v_all", "full"):
        # Construct mixed donors only from frozen clean/donor captures. V4 applies
        # the resulting FP32 contrast to the measured BF16 clean anchor.
        for p, r in zip(pl, rows):
            n = len(r["token_ids"])
            for key in keys(U):
                c, d = p["clean"][key], p["donor"][key]
                k, v = c["k"].copy(), c["v"].copy()
                stop = r["context_end"] if mode == "kv_context" else n
                if mode in ("kv_context", "k_all", "full"):
                    k[:stop] = d["k"][:stop]
                if mode in ("kv_context", "v_all", "full"):
                    v[:stop] = d["v"][:stop]
                q = d["q"] if mode == "full" else c["q"]
                score = (q[None] * k).sum(-1) / np.sqrt(np.float32(q.shape[-1]))
                prob = np.exp(score - score.max())
                prob /= prob.sum()
                if np.array_equal(q, c["q"]) and np.array_equal(k, c["k"]):
                    prob = c["p"].copy()
                d.update(q=q, k=k, v=v, p=prob)
        action = "full"
    replay = await execute(worker, rows, outputs, U, action, pl)
    arrays(prefix + "-U-replay", replay)
    return await downstream(worker, rows, outputs, base, replay["applied"], axis, prefix)


async def calibration(worker, data):
    rows = [r for r in data["rows"] if r["split"] == "calibration"]
    outputs = data["output_ids"]["calibration"]
    base = await execute(worker, rows, outputs)
    ub = await execute(worker, rows, outputs, U, capture_h0=True)
    arrays("calibration-baseline", base)
    arrays("calibration-U-baseline", ub)
    screen = []
    labels = np.array([r["label"] for r in rows])
    for head in H0:
        record = {"head": keys((head,))[0]}
        for axis in ("B", "C"):
            end = await hop(
                worker,
                rows,
                outputs,
                base,
                ub,
                (head,),
                axis,
                f"calibration-{head[0]}-{head[1]}-{axis}",
            )
            record[axis] = effect(rows, labels * (margin(base) - margin(end)))
        screen.append(record)
        print("Screen", record, flush=True)
    save("screen.json", screen)
    save(
        "calibration-complete.json",
        dict(utc=datetime.now(timezone.utc).isoformat(), batches=worker.batches),
    )


def freeze2():
    verify("calibration")
    assert (OUT / "calibration-complete.json").exists()
    selected = select_hop0(read("screen.json"))
    files = sorted(OUT.glob("calibration-*")) + [
        OUT / "screen.json",
        OUT / "stage1.json",
        OUT / "stage1-commit.json",
    ]
    save(
        "stage2.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            selected=selected,
            sha256={p.name: sha(p) for p in files},
        ),
    )


async def confirm(worker, data, run):
    coalition = tuple(tuple(map(int, k.split(":"))) for k in read("stage2.json")["selected"])
    summary = {}
    for split in ("calibration", "primary", "vocabulary"):
        rows = [r for r in data["rows"] if r["split"] == split]
        outputs = data["output_ids"][split]
        prefix = f"confirm{run}-{split}"
        base = await execute(worker, rows, outputs)
        ub = await execute(worker, rows, outputs, U, capture_h0=True)
        arrays(prefix + "-baseline", base)
        arrays(prefix + "-U-baseline", ub)
        labels = np.array([r["label"] for r in rows])
        result = dict(accuracy=float(np.mean(labels * margin(base) > 0)))
        self_end = await hop(
            worker, rows, outputs, base, ub, coalition, "C", prefix + "-self-hop0", True
        )
        identity = float(np.max(abs(margin(self_end) - margin(base))))
        result["identity"] = identity
        if split != "calibration":
            paths, direct, ins, hops = {}, {}, {}, {}
            index = {r["id"]: i for i, r in enumerate(rows)}
            for axis, modes in PATHS.items():
                for mode in modes:
                    end = await execute(
                        worker, rows, outputs, mode=mode, payload=payloads(rows, base, D, axis)
                    )
                    paths[f"{axis}_{mode}"] = end
                    arrays(prefix + f"-{axis}_{mode}", end)
            for axis in ("B", "C"):
                donor = [index[r["donors"][axis]] for r in rows]
                direct[axis] = await downstream(
                    worker,
                    rows,
                    outputs,
                    base,
                    ub["h"][donor],
                    axis,
                    prefix + f"-{axis}_upstream_Q",
                )
                for mode in ("full", "q_final", "kv_context", "k_all", "v_all"):
                    ins[axis + "_U_" + mode] = await internal(
                        worker, rows, outputs, base, ub, axis, mode, prefix + f"-{axis}_U_{mode}"
                    )
                hops[axis + "_hop0"] = await hop(
                    worker, rows, outputs, base, ub, coalition, axis, prefix + f"-{axis}_hop0"
                )
            result["track1"] = evaluate(rows, base, paths, direct)
            result["track2"] = track2(rows, base, direct, ins, hops, identity)
        summary[split] = result
        print(prefix, json.dumps(result), flush=True)
    save(
        f"confirmation-{run}.json",
        dict(splits=summary, utc=datetime.now(timezone.utc).isoformat(), batches=worker.batches),
    )


async def run(stage):
    verify(stage)
    assert os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0.70"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    assert os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") == "0.50"
    with open("/tmp/interp-swarm-mps.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            stage + "-started.json",
            dict(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()),
        )
        wc = WorkerConfig(batch_size=64)
        h = await asyncio.to_thread(
            V4Harness.load,
            ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True),
            wc,
        )
        async with MPSBatchWorker(h, wc) as worker:
            if stage == "calibration":
                await calibration(worker, read("dataset.json"))
            else:
                await confirm(worker, read("dataset.json"), int(stage[-1]))
    verify(stage)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=["prepare", "freeze", "calibration", "freeze2", "confirm1", "confirm2", "review"],
    )
    stage = p.parse_args().stage
    if stage == "review":
        from src.poc.review import review_v9

        asyncio.run(
            review_v9(
                OUT,
                dict(
                    protocol=PROTOCOL.read_text(),
                    results=read("results.json"),
                    facts=read("facts.json"),
                ),
            )
        )
    elif stage == "prepare":
        prepare()
    elif stage == "freeze":
        freeze()
    elif stage == "freeze2":
        freeze2()
    else:
        asyncio.run(run(stage))
