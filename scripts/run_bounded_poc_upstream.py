"""Isolated, prospective upstream screening and two-axis confirmation."""

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
from src.poc.review import review
from src.poc.upstream_analysis import (
    LIMITS,
    axis_gate,
    effects,
    rank_candidates,
    routing,
    upstream_gate,
)
from src.poc.upstream_controls import planted
from src.poc.upstream_harness import TARGETS, RoutingHarness, RoutingProbe, RoutingSpec
from src.poc.upstream_templates import build_upstream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-upstream"
PROTOCOL = ROOT / "reports/bounded-poc-upstream-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = ROOT / f"data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/{REVISION}"
TARGET_KEYS = [f"{layer}:{head}" for layer, head in TARGETS]
UPSTREAM = [f"{layer}:{head}" for layer in range(19) for head in range(16)]


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def arrays(name, **values):
    with (OUT / name).open("xb") as handle:
        np.savez_compressed(handle, **values)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected():
    inventory = json.loads((ROOT / "upstream-protected.json").read_text())
    for name, expected in inventory["sha256"].items():
        assert sha(ROOT / name) == expected, f"Copied protected artifact changed: {name}"
        assert sha(Path(inventory["original_root"]) / name) == expected, (
            f"Original artifact changed: {name}"
        )
    return inventory


def prepare():
    OUT.mkdir(exist_ok=True)
    protected()
    retired = [
        r
        for version in ("v1", "v2")
        for r in json.loads((ROOT / f"data/bounded-poc-{version}/dataset.json").read_text())["rows"]
    ]
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = build_upstream(tokenizer, retired)
    save("dataset.json", data)
    controls = planted()
    assert controls["passed"]
    save("known-controls.json", controls)
    print(data["sanity"], flush=True)


def runtime_files():
    return [
        *sorted((ROOT / "src/poc").glob("upstream*.py")),
        ROOT / "src/poc/harness.py",
        ROOT / "src/poc/review.py",
        ROOT / "src/poc/templates.py",
        ROOT / "src/harness/qwen_model.py",
        ROOT / "src/harness/mps_queue.py",
        ROOT / "src/config.py",
        ROOT / "src/agents/client.py",
        ROOT / "src/agents/prompts.py",
        ROOT / "tests/test_poc_upstream.py",
        Path(__file__).resolve(),
        ROOT / "scripts/audit_bounded_poc_upstream.py",
        ROOT / "upstream-protected.json",
        PROTOCOL,
        OUT / "dataset.json",
        OUT / "known-controls.json",
        OUT / "validation.json",
        *sorted(MODEL.glob("*.json")),
        *sorted(MODEL.glob("*.safetensors")),
    ]


def register(final=False):
    protected()
    assert read("known-controls.json")["passed"] and read("validation.json")["passed"]
    files = runtime_files()
    if final:
        verify("discovery-registration.json")
        assert not any(OUT.glob("confirm*-started.json"))
        files += [OUT / "discovery-registration.json", OUT / "discovery.json"] + sorted(
            OUT.glob("discovery-*.npz")
        )
    name = "manifest.json" if final else "discovery-registration.json"
    record = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        phase="pre_heldout" if final else "pre_discovery",
        revision=REVISION,
        device="mps",
        dtype="bfloat16",
        seed=42,
        batch_size=64,
        max_seq_len=128,
        targets=TARGET_KEYS,
        upstream_candidates=UPSTREAM,
        limits=LIMITS,
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
        ),
        sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
    )
    save(name, record)
    # Preserve exact source bytes alongside the registration, even if later work changes files.
    if not final:
        for p in files:
            if "hf-cache" not in p.parts and p.suffix in (".py", ".md"):
                dest = OUT / "frozen-source" / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
    print(name, sha(OUT / name), flush=True)


def verify(name="manifest.json"):
    for file, expected in read(name)["sha256"].items():
        assert sha(ROOT / file) == expected, f"Registered input changed: {file}"
    assert read(name)["limits"] == LIMITS
    protected()


def dataset(split):
    data = read("dataset.json")
    return [r for r in data["rows"] if r["split"] == split], tuple(
        data["output_ids"]["vocabulary" if split == "vocabulary" else "primary"]
    )


def donor_indices(rows, axis):
    index = {r["id"]: i for i, r in enumerate(rows)}
    field = "opposite_id" if axis == "A" else "query_opposite_id"
    return np.array([index[r[field]] for r in rows])


def margin(base):
    return base["logits"][:, 0] - base["logits"][:, 1]


def pack(results, full=False):
    data = dict(logits=np.array([r.logits for r in results]))
    for field in ("h", "p", "applied", "reconstruction_error", "sibling_error", "delta"):
        values = []
        for r in results:
            heads = []
            for key in TARGET_KEYS:
                value = np.array(r.routing[key].get(field, np.zeros(128)), dtype=np.float32)
                if field == "p":
                    value = np.pad(value, (0, 128 - len(value)))
                heads.append(value)
            values.append(heads)
        data[field] = np.array(values)
    if full:
        data["v"] = np.array(
            [
                [
                    np.pad(
                        np.array(r.routing[k]["v"], dtype=np.float32),
                        ((0, 128 - len(r.routing[k]["v"])), (0, 0)),
                    )
                    for k in TARGET_KEYS
                ]
                for r in results
            ]
        )
        data["upstream"] = np.array(
            [[r.heads[k] for k in UPSTREAM] for r in results], dtype=np.float32
        )
    assert all(np.isfinite(v).all() for v in data.values())
    return data


def payloads(rows, base, axis="B", pattern=None, upstream=(), self_donor=False):
    donors = np.arange(len(rows)) if self_donor else donor_indices(rows, axis)
    result = []
    for i, r in enumerate(rows):
        n, j = len(r["token_ids"]), donors[i]
        clean = {
            k: dict(h=base["h"][i, h], p=base["p"][i, h, :n], v=base["v"][i, h, :n])
            for h, k in enumerate(TARGET_KEYS)
        }
        other = {
            k: dict(
                h=base["h"][j, h],
                p=base["p"][j, h, :n] if pattern is None else pattern[i, h, :n],
                v=base["v"][j, h, :n],
            )
            for h, k in enumerate(TARGET_KEYS)
        }
        result.append(
            dict(
                clean=clean,
                donor=other,
                upstream={k: base["upstream"][j, UPSTREAM.index(k)] for k in upstream},
                color_positions=r["color_positions"],
                context_end=r["context_end"],
            )
        )
    return result


async def execute_probes(worker, rows, outputs, spec, payload=None, full=False):
    result = await worker.map(
        [
            RoutingProbe(
                r["text"],
                tuple(r["token_ids"]),
                outputs,
                spec,
                payload=None if payload is None else payload[i],
            )
            for i, r in enumerate(rows)
        ]
    )
    return pack(result, full=full)


async def baseline(worker, rows, outputs, prefix):
    result = await execute_probes(worker, rows, outputs, RoutingSpec(tuple(range(19))), full=True)
    arrays(f"{prefix}-baseline.npz", **result)
    return result


async def mediated(worker, rows, outputs, base, heads, self_donor=False):
    members = tuple(tuple(map(int, k.split(":"))) for k in heads)
    mid = await execute_probes(
        worker,
        rows,
        outputs,
        RoutingSpec((), upstream=members, mode="mediate"),
        payloads(rows, base, upstream=heads, self_donor=self_donor),
    )
    end = await execute_probes(
        worker,
        rows,
        outputs,
        RoutingSpec((), active=TARGETS, mode="qk"),
        payloads(rows, base, pattern=mid["p"], self_donor=self_donor),
    )
    return mid, end


async def controls(worker, rows, outputs, base, prefix, selected=()):
    measured = {}
    identity = await execute_probes(
        worker,
        rows,
        outputs,
        RoutingSpec((), active=TARGETS, mode="qk"),
        payloads(rows, base, self_donor=True),
    )
    arrays(f"{prefix}-identity.npz", **identity)
    measured["identity_margin_error"] = float(np.max(abs(margin(identity) - margin(base))))
    measured["clean_reconstruction_error"] = float(base["reconstruction_error"].max())
    measured["sibling_error"] = float(identity["sibling_error"].max())
    context_error = 0.0
    for i, j in enumerate(donor_indices(rows, "B")):
        context_error = max(
            context_error,
            float(
                np.max(
                    abs(
                        base["v"][i, :, : rows[i]["context_end"]]
                        - base["v"][j, :, : rows[i]["context_end"]]
                    )
                )
            ),
        )
    measured["axis_B_context_V_error"] = context_error
    replay_error = 0.0
    for axis in ("A", "B"):
        for mode in ("full", "replace"):
            out = await execute_probes(
                worker,
                rows,
                outputs,
                RoutingSpec((), active=(TARGETS[0],), mode=mode),
                payloads(rows, base, axis),
            )
            arrays(f"{prefix}-{axis}-{mode}-control.npz", **out)
            measured[f"{axis}_{mode}_margin"] = margin(out).tolist()
        replay_error = max(
            replay_error,
            float(
                np.max(
                    abs(
                        np.array(measured[f"{axis}_full_margin"])
                        - measured[f"{axis}_replace_margin"]
                    )
                )
            ),
        )
    measured["full_replay_error"] = replay_error
    if selected:
        mid, end = await mediated(worker, rows, outputs, base, selected, True)
        arrays(f"{prefix}-upstream-self-mediator.npz", **mid)
        arrays(f"{prefix}-upstream-self-output.npz", **end)
        measured["upstream_self_pattern_error"] = float(np.max(abs(mid["p"] - base["p"])))
        measured["upstream_self_margin_error"] = float(np.max(abs(margin(end) - margin(base))))
    measured["passed"] = (
        measured["identity_margin_error"] == 0
        and context_error == 0
        and measured["sibling_error"] == 0
        and measured["clean_reconstruction_error"] <= 0.05
        and replay_error <= 0.10
        and measured.get("upstream_self_pattern_error", 0) == 0
        and measured.get("upstream_self_margin_error", 0) == 0
    )
    return measured


async def discover(worker):
    rows, outputs = dataset("discovery")
    base = await baseline(worker, rows, outputs, "discovery")
    control = await controls(worker, rows, outputs, base, "discovery")
    scores = []
    patterns = []
    outputs_all = []
    deltas = []
    applied = []
    clamp = []
    sibling = []
    for index, key in enumerate(UPSTREAM):
        mid, end = await mediated(worker, rows, outputs, base, [key])
        shifts = routing(rows, base["p"], mid["p"])
        score = dict(
            head=key,
            minimum_transfer=min(min(s["donor_gain"], s["clean_loss"]) for s in shifts),
            effect=effects(rows, margin(base), margin(end)),
            targets=shifts,
        )
        scores.append(score)
        patterns.append(mid["p"])
        outputs_all.append(end["logits"])
        deltas.append(end["delta"])
        applied.append(end["applied"])
        clamp.append(mid["applied"])
        sibling.append(max(float(mid["sibling_error"].max()), float(end["sibling_error"].max())))
        if (index + 1) % 16 == 0:
            print(
                f"Screened {index + 1}/304 upstream heads; latest layer={index // 16}", flush=True
            )
    arrays(
        "discovery-screen.npz",
        patterns=np.array(patterns),
        logits=np.array(outputs_all),
        delta=np.array(deltas),
        applied=np.array(applied),
        clamped=np.array(clamp),
        sibling_error=np.array(sibling),
    )
    selected = [s["head"] for s in rank_candidates(scores)]
    joint = None
    if selected:
        mid, end = await mediated(worker, rows, outputs, base, selected)
        arrays("discovery-joint-mediator.npz", **mid)
        arrays("discovery-joint-output.npz", **end)
        joint = upstream_gate(rows, margin(base), margin(end), base["p"], mid["p"])
        control.update(await controls(worker, rows, outputs, base, "discovery-selected", selected))
    result = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        model_accuracy=float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0)),
        controls=control,
        candidates=scores,
        selected=selected,
        joint=joint,
        registration_sha256=sha(OUT / "discovery-registration.json"),
    )
    save("discovery.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2), flush=True)


async def confirm(worker, run):
    selected = read("discovery.json")["selected"]
    result = dict(manifest_sha256=sha(OUT / "manifest.json"), selected=selected, splits={})
    for split in ("primary", "vocabulary"):
        prefix = f"confirm{run}-{split}"
        rows, outputs = dataset(split)
        base = await baseline(worker, rows, outputs, prefix)
        control = await controls(worker, rows, outputs, base, prefix, selected)
        measures = {}
        sibling_error = 0.0
        for axis in ("A", "B"):
            for name, active in (("single", (TARGETS[0],)), ("coalition", TARGETS)):
                for mode in ("ov_color", "qk") if axis == "A" else ("qk", "ov_context"):
                    tag = f"{axis}_{name}_{mode}"
                    out = await execute_probes(
                        worker,
                        rows,
                        outputs,
                        RoutingSpec((), active=active, mode=mode),
                        payloads(rows, base, axis),
                    )
                    arrays(f"{prefix}-{tag}.npz", **out)
                    measures[tag] = margin(out)
                    sibling_error = max(sibling_error, float(out["sibling_error"].max()))
        arrays(f"{prefix}-axis-margins.npz", **measures)
        joint = None
        if selected:
            mid, end = await mediated(worker, rows, outputs, base, selected)
            arrays(f"{prefix}-mediator.npz", **mid)
            arrays(f"{prefix}-upstream-output.npz", **end)
            joint = upstream_gate(rows, margin(base), margin(end), base["p"], mid["p"])
            sibling_error = max(
                sibling_error, float(mid["sibling_error"].max()), float(end["sibling_error"].max())
            )
        control["all_interventions_sibling_error"] = sibling_error
        control["passed"] = control["passed"] and sibling_error == 0
        result["splits"][split] = dict(
            model_accuracy=float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0)),
            controls=control,
            axis=axis_gate(rows, margin(base), measures),
            upstream=joint,
        )
        print(
            f"Confirmation {run} {split}: "
            + json.dumps({k: v for k, v in result["splits"][split].items() if k != "controls"}),
            flush=True,
        )
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(f"confirmation-{run}.json", result)


async def run(stage):
    verify("discovery-registration.json" if stage == "discover" else "manifest.json")
    lock = OUT / "running.lock"
    # Cross-process advisory guard, in addition to each worker's single consumer.
    with open("/tmp/interp-swarm-mps.lock", "a+") as global_lock:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with lock.open("x") as file:
            file.write(str(os.getpid()))
        try:
            save(
                f"{stage}-started.json",
                dict(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()),
            )
            wc = WorkerConfig(batch_size=64)
            harness = await asyncio.to_thread(
                RoutingHarness.load,
                ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True),
                wc,
            )
            async with MPSBatchWorker(harness, wc) as worker:
                if stage == "discover":
                    await discover(worker)
                else:
                    await confirm(worker, int(stage[-1]))
                print("Microbatches", worker.batches, flush=True)
        finally:
            lock.unlink()
    protected()


async def main(stage):
    if stage == "prepare":
        prepare()
    elif stage == "register":
        register()
    elif stage == "freeze":
        register(True)
    elif stage == "review":
        await review(
            OUT, "upstream", dict(protocol=PROTOCOL.read_text(), results=read("results.json"))
        )
    else:
        await run(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "register", "discover", "freeze", "confirm1", "confirm2", "review"),
    )
    asyncio.run(main(parser.parse_args().stage))
