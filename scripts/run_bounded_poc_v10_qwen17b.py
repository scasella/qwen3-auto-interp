"""Exclusive-write, two-freeze v10 study. No historical runner is invoked."""

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
import transformers

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.source_paths import model17_path
from src.poc.v4_harness import V4Probe, V4Spec
from src.poc.v10_controls import planted
from src.poc.v10_harness import V10Harness
from src.poc.v10_qwen17b import (
    DOWNSTREAM_CANDIDATES,
    LIMITS,
    PATHS,
    REVISION,
    SEED,
    evaluate,
    margin,
    select_extension,
    upstream_candidates,
)
from src.poc.v10_templates import LEXICON, build, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v10-qwen17b"
PROTOCOL = ROOT / "docs/bounded-poc-v10-qwen17b-protocol.md"
MODEL = model17_path()


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def arrays(name, value):
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez_compressed(f, **value)


def load(name):
    with np.load(OUT / (name + ".npz"), allow_pickle=False) as z:
        return dict(z)


def protected():
    from scripts.verify_all_audits import verify_inventory

    return verify_inventory(ROOT)


def verify_hashes(record):
    for name, expected in record["sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen bytes changed: {name}")
    for name, expected in record.get("model_sha256", {}).items():
        if sha(MODEL / name) != expected:
            raise ValueError(f"Checkpoint bytes changed: {name}")
    for name, expected in record.get("external_sha256", {}).items():
        if sha(name) != expected:
            raise ValueError(f"External protected bytes changed: {name}")


def retired_rows(v9_root):
    """Scan all preserved JSON for keys/text, including diagnostic/gold/silver rows."""
    roots = [ROOT / "data", v9_root]
    if v9_root is None or not (v9_root / "dataset.json").is_file():
        raise FileNotFoundError("V9 frozen dataset and manifest are required before preparation")
    registration = v9_root / "stage1.json"
    if not registration.is_file():
        raise FileNotFoundError("V9 stage1.json is required; draft source is insufficient")
    commit_file = v9_root / "stage1-commit.json"
    commit = json.loads(commit_file.read_text())["commit"]
    relative = registration.relative_to(v9_root.parent.parent)
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=v9_root.parent.parent
    )
    if committed != registration.read_bytes():
        raise ValueError("V9 registration does not match its retained commit")
    v9_manifest = json.loads(registration.read_text())
    if not v9_manifest.get("sha256"):
        raise ValueError("V9 manifest has no retained SHA-256 inventory")
    retired, sources = [], {}
    for name, expected in v9_manifest["sha256"].items():
        path = Path(name)
        if not path.is_absolute():
            path = v9_root.parent.parent / path
        if sha(path) != expected:
            raise ValueError(f"V9 protected bytes changed: {path}")
        sources[str(path.resolve())] = expected

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("keys"), list) and all(isinstance(k, str) for k in x["keys"]):
                retired.append(dict(keys=x["keys"], text=x.get("text", "")))
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)

    for root in roots:
        for path in sorted(root.rglob("*.json")):
            if (
                path.is_relative_to(OUT)
                or "hf-cache" in path.parts
                or "frozen-source" in path.parts
            ):
                continue
            walk(json.loads(path.read_text()))
    for path in (registration, commit_file):
        sources[str(path.resolve())] = sha(path)
    for path in sorted((v9_root.parent.parent / "reports").glob("*v9*")):
        if path.is_file():
            sources[str(path.resolve())] = sha(path)
    return retired, sources


def prepare(v9_root):
    protected()
    retired, external = retired_rows(v9_root)
    tok = transformers.AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = build(tok, retired)
    OUT.mkdir(exist_ok=True)
    save(
        "retired-keys.json",
        dict(keys=sorted({k for r in retired for k in r["keys"]}), external_sha256=external),
    )
    save("whitelist.json", sorted(LEXICON))
    save("dataset.json", data)
    controls = planted()
    assert controls["passed"]
    save("known-controls.json", controls)
    print("Prepared 256 prompts, 64 keys; no target forwards", flush=True)


def freeze(stage):
    protected()
    if stage == "pre":
        assert read("validation.json")["passed"]
        assert read("known-controls.json")["passed"]
        assert not list(OUT.glob("*started.json"))
        record = dict(
            model="Qwen/Qwen3-1.7B",
            revision=REVISION,
            key_seed=SEED,
            runtime_seed=42,
            limits=LIMITS,
            batch_size=64,
            max_seq_len=128,
            downstream_candidates=DOWNSTREAM_CANDIDATES,
            upstream_rule="layers 10 through min(downstream layer)-1; all 16 heads",
            selector="sum of axis medians; numeric layer/head tie break; 3 greedy rounds",
            downstream_solo_direction=0.8,
            failure="abstain if fewer than 3 eligible heads",
            registered_utc=now(),
            external_sha256=read("retired-keys.json")["external_sha256"],
            model_sha256={p.name: sha(p) for p in sorted(MODEL.iterdir()) if p.is_file()},
            versions=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                transformers=transformers.__version__,
            ),
        )
        files = [p for folder in ("src", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
        files += [ROOT / "pyproject.toml", ROOT / "uv.lock", PROTOCOL]
        files += list(OUT.glob("*.json")) + list(OUT.glob("*validation.log"))
        filename = "pre-calibration-manifest.json"
    else:
        verify("pre")
        selection = read("selection.json")
        if selection["status"] != "selected":
            raise ValueError("Discovery abstained; confirmation is prohibited")
        record = dict(
            registered_utc=now(),
            downstream=selection["downstream"],
            upstream=selection["upstream"],
            limits=LIMITS,
            revision=REVISION,
            pre_manifest_sha256=sha(OUT / "pre-calibration-manifest.json"),
            model_sha256=read("pre-calibration-manifest.json")["model_sha256"],
        )
        files = [p for p in OUT.iterdir() if p.is_file()]
        filename = "manifest.json"
    record["sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    save(filename, record)
    print(filename, sha(OUT / filename), flush=True)


def verify(stage):
    protected()
    pre = read("pre-calibration-manifest.json")
    assert pre["limits"] == LIMITS and pre["key_seed"] == SEED and pre["revision"] == REVISION
    verify_hashes(pre)
    validate(read("dataset.json")["rows"])
    stamp = read("pre-calibration-registration.json")
    assert stamp["manifest_sha256"] == sha(OUT / "pre-calibration-manifest.json")
    subprocess.run(["git", "cat-file", "-e", stamp["commit"] + "^{commit}"], cwd=ROOT, check=True)
    blob = subprocess.check_output(
        [
            "git",
            "show",
            stamp["commit"] + ":data/bounded-poc-v10-qwen17b/pre-calibration-manifest.json",
        ],
        cwd=ROOT,
    )
    assert hashlib.sha256(blob).hexdigest() == stamp["manifest_sha256"]
    if stage != "pre":
        verify_hashes(read("manifest.json"))
        stamp = read("confirmation-registration.json")
        assert stamp["manifest_sha256"] == sha(OUT / "manifest.json")
        blob = subprocess.check_output(
            ["git", "show", stamp["commit"] + ":data/bounded-poc-v10-qwen17b/manifest.json"],
            cwd=ROOT,
        )
        assert hashlib.sha256(blob).hexdigest() == stamp["manifest_sha256"]


def pack(results, targets, upstream=(), full=False, keep_k=False):
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
            for layer, h in targets:
                value = np.asarray(
                    r.routing[f"{layer}:{h}"].get(field, np.zeros(128)), dtype=np.float32
                )
                if field in ("p", "routed_p", "k", "v"):
                    value = np.pad(value, ((0, 128 - len(value)),) + ((0, 0),) * (value.ndim - 1))
                heads.append(value)
            values.append(heads)
        out[field] = np.array(values)
    if upstream:
        out["upstream"] = np.array(
            [[r.heads[f"{layer}:{h}"] for layer, h in upstream] for r in results], dtype=np.float32
        )
    assert all(np.isfinite(a).all() for a in out.values())
    return out


def payloads(rows, base, targets, upstream=(), axis="B", self_patch=False, donor_q=None):
    ids = {r["id"]: i for i, r in enumerate(rows)}
    result = []
    for i, r in enumerate(rows):
        j = i if self_patch else ids[r["donors"][axis]]
        n = len(r["token_ids"])

        def bundle(idx):
            return {
                f"{layer}:{h}": {
                    f: base[f][idx, k, :n] if f in ("p", "k", "v") else base[f][idx, k]
                    for f in ("h", "p", "q", "k", "v")
                }
                for k, (layer, h) in enumerate(targets)
            }

        clean, donor = bundle(i), bundle(j)
        if donor_q is not None:
            for k, (layer, h) in enumerate(targets):
                donor[f"{layer}:{h}"]["q"] = donor_q[i, k]
        result.append(
            dict(
                clean=clean,
                donor=donor,
                upstream={
                    f"{layer}:{h}": base["upstream"][j, k] for k, (layer, h) in enumerate(upstream)
                },
                color_positions=r["color_positions"],
                context_end=r["context_end"],
            )
        )
    return result


async def execute(
    worker,
    rows,
    outputs,
    targets,
    mode="capture",
    payload=None,
    active=(),
    upstream=(),
    capture_u=(),
    full=False,
):
    spec = V4Spec(
        tuple(sorted({layer for layer, h in capture_u})),
        targets=targets,
        mode=mode,
        active=active,
        upstream=upstream,
    )
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
    return pack(results, targets, capture_u, full, mode == "mediate")


def subset(base, indices):
    return {k: a if k == "logits" else a[:, indices] for k, a in base.items()}


async def discovery(worker):
    data = read("dataset.json")
    rows = [r for r in data["rows"] if r["split"] == "calibration"]
    outputs = data["output_ids"]["calibration"]
    labels = np.array([r["label"] for r in rows])
    solos, records = {}, []
    # Capture 16 heads at a time; no 176-head sequence cache resides on MPS.
    for layer in range(15, 26):
        targets = tuple((layer, h) for h in range(16))
        full = await execute(worker, rows, outputs, targets, full=True)
        arrays(f"discovery-layer-{layer}", full)
        for h in range(16):
            head = (layer, h)
            base = subset(full, [h])
            solos[head] = base
            effects = {}
            for axis, mode in [("A", "ov_color"), ("B", "q_final")]:
                out = await execute(
                    worker,
                    rows,
                    outputs,
                    (head,),
                    mode,
                    payloads(rows, base, (head,), axis=axis),
                    (head,),
                )
                tag = f"discovery-solo-{layer}-{h}-{axis}"
                arrays(tag, out)
                effect = labels * (margin(base) - margin(out))
                effects[axis] = dict(
                    median=float(np.median(effect)), direction=float(np.mean(effect > 0)), array=tag
                )
            records.append(dict(head=head, effects=effects))
        print("Downstream solo layer", layer, "complete", flush=True)
    eligible = [
        tuple(r["head"])
        for r in records
        if all(r["effects"][a]["direction"] >= 0.8 for a in ("A", "B"))
    ]
    save("downstream-solos.json", records)
    if len(eligible) < 3:
        save(
            "selection.json",
            dict(
                status="abstained",
                reason="fewer than 3 solo-eligible downstream heads",
                eligible=eligible,
            ),
        )
        return
    selected = []
    trace = []
    for step in range(3):
        scores = {}
        for head in eligible:
            if head in selected:
                continue
            targets = tuple(sorted(selected + [head]))
            base = {
                k: solos[targets[0]][k]
                if k == "logits"
                else np.concatenate([solos[t][k] for t in targets], axis=1)
                for k in solos[targets[0]]
            }
            values = []
            tags = []
            for axis, mode in [("A", "ov_color"), ("B", "q_final")]:
                out = await execute(
                    worker,
                    rows,
                    outputs,
                    targets,
                    mode,
                    payloads(rows, base, targets, axis=axis),
                    targets,
                )
                tag = f"discovery-D-step{step}-{head[0]}-{head[1]}-{axis}"
                arrays(tag, out)
                tags.append(tag)
                values.append(float(np.median(labels * (margin(base) - margin(out)))))
            scores[head] = sum(values)
            trace.append(
                dict(stage=step, head=head, coalition=targets, score=scores[head], arrays=tags)
            )
        selected.append(select_extension(scores))
        print("Downstream selected", selected, flush=True)
    targets = tuple(sorted(selected))
    del solos
    save("downstream-selection.json", dict(downstream=targets, eligible=eligible, trace=trace))
    candidates = upstream_candidates(targets)
    base = await execute(worker, rows, outputs, targets, capture_u=candidates, full=True)
    arrays("discovery-upstream-baseline", base)
    chosen = []
    trace = []
    for step in range(3):
        scores = {}
        for head in candidates:
            if head in chosen:
                continue
            members = tuple(sorted(chosen + [head]))
            values = []
            tags = []
            for axis in ("B", "C"):
                p = payloads(rows, base, targets, candidates, axis)
                mid = await execute(worker, rows, outputs, targets, "mediate", p, upstream=members)
                tag = f"discovery-U-step{step}-{head[0]}-{head[1]}-{axis}"
                arrays(tag + "-mediator", mid)
                end = await execute(
                    worker,
                    rows,
                    outputs,
                    targets,
                    "q_final",
                    payloads(rows, base, targets, axis=axis, donor_q=mid["q"]),
                    targets,
                )
                arrays(tag, end)
                tags.append(tag)
                values.append(float(np.median(labels * (margin(base) - margin(end)))))
            scores[head] = sum(values)
            trace.append(
                dict(stage=step, head=head, coalition=members, score=scores[head], arrays=tags)
            )
        chosen.append(select_extension(scores))
        print("Upstream selected", chosen, flush=True)
    save(
        "selection.json",
        dict(
            status="selected",
            downstream=targets,
            upstream=sorted(chosen),
            trace=trace,
            completed_utc=now(),
        ),
    )


async def confirm(worker, run):
    data = read("dataset.json")
    manifest = read("manifest.json")
    targets = tuple(map(tuple, manifest["downstream"]))
    members = tuple(map(tuple, manifest["upstream"]))
    summary = dict(manifest_sha256=sha(OUT / "manifest.json"), splits={})
    for split in ("calibration", "primary", "vocabulary"):
        rows = [r for r in data["rows"] if r["split"] == split]
        outputs = data["output_ids"][split]
        prefix = f"confirm{run}-{split}"
        base = await execute(worker, rows, outputs, targets, capture_u=members, full=True)
        arrays(prefix + "-baseline", base)
        controls = {}
        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            out = await execute(
                worker,
                rows,
                outputs,
                targets,
                mode,
                payloads(rows, base, targets, self_patch=True),
                targets,
            )
            arrays(prefix + "-self-" + mode, out)
            controls["self_" + mode] = float(np.max(abs(margin(out) - margin(base))))
        mid = await execute(
            worker,
            rows,
            outputs,
            targets,
            "mediate",
            payloads(rows, base, targets, members, self_patch=True),
            upstream=members,
        )
        end = await execute(
            worker,
            rows,
            outputs,
            targets,
            "q_final",
            payloads(rows, base, targets, self_patch=True, donor_q=mid["q"]),
            targets,
        )
        arrays(prefix + "-self-mediator", mid)
        arrays(prefix + "-self-upstream", end)
        controls["self_upstream_margin"] = float(np.max(abs(margin(end) - margin(base))))
        controls["self_upstream_pattern"] = float(np.max(abs(end["routed_p"] - base["p"])))
        paths = {}
        upstream = {}
        for axis, modes in PATHS.items():
            for mode in modes:
                tag = f"{axis}_{mode}"
                paths[tag] = await execute(
                    worker,
                    rows,
                    outputs,
                    targets,
                    mode,
                    payloads(rows, base, targets, axis=axis),
                    targets,
                )
                arrays(prefix + "-" + tag, paths[tag])
        for axis in ("B", "C"):
            mid = await execute(
                worker,
                rows,
                outputs,
                targets,
                "mediate",
                payloads(rows, base, targets, members, axis),
                upstream=members,
            )
            end = await execute(
                worker,
                rows,
                outputs,
                targets,
                "q_final",
                payloads(rows, base, targets, axis=axis, donor_q=mid["q"]),
                targets,
            )
            arrays(prefix + f"-{axis}-mediator", mid)
            arrays(prefix + f"-{axis}_upstream_Q", end)
            upstream[axis] = end
        summary["splits"][split] = dict(
            accuracy=float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0)),
            controls=controls,
            scientific=evaluate(rows, base, paths, upstream),
        )
        print("Completed", run, split, flush=True)
    summary.update(completed_utc=now(), microbatches=worker.batches)
    save(f"confirmation-{run}.json", summary)


async def run(stage):
    verify("pre" if stage == "discover" else "confirmation")
    for name, value in [
        ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.70"),
        ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.60"),
        ("TOKENIZERS_PARALLELISM", "false"),
    ]:
        if os.environ.get(name) != value:
            raise RuntimeError(f"Required environment: {name}={value}")
    with open("/tmp/interp-swarm-mps.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if stage == "confirm2":
            read("confirmation-1.json")
        save(
            stage + "-started.json",
            dict(
                utc=now(),
                pid=os.getpid(),
                manifest_sha256=sha(
                    OUT
                    / ("pre-calibration-manifest.json" if stage == "discover" else "manifest.json")
                ),
            ),
        )
        wc = WorkerConfig(batch_size=64, purge_every=1)
        harness = None
        try:
            harness = await asyncio.to_thread(
                V10Harness.load,
                ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True),
                wc,
            )
            async with MPSBatchWorker(harness, wc) as worker:
                if stage == "discover":
                    await discovery(worker)
                    save(
                        "discover-complete.json",
                        dict(utc=now(), pid=os.getpid(), microbatches=worker.batches),
                    )
                else:
                    await confirm(worker, int(stage[-1]))
        except Exception as exc:
            save(stage + "-failed.json", dict(utc=now(), type=type(exc).__name__, error=str(exc)))
            raise
        finally:
            del harness
            torch.mps.empty_cache()
    verify("pre" if stage == "discover" else "confirmation")


async def main(args):
    if args.stage == "prepare":
        prepare(args.v9_root)
    elif args.stage == "freeze-pre":
        freeze("pre")
    elif args.stage == "freeze-confirm":
        freeze("confirmation")
    elif args.stage == "review":
        from src.poc.v10_review import review_v10

        await review_v10(
            OUT,
            dict(
                protocol=PROTOCOL.read_text(),
                results=read("results.json"),
                facts=read("facts.json"),
            ),
        )
    else:
        await run(args.stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "freeze-pre",
            "discover",
            "freeze-confirm",
            "confirm1",
            "confirm2",
            "review",
        ],
    )
    parser.add_argument("--v9-root", type=Path)
    asyncio.run(main(parser.parse_args()))
