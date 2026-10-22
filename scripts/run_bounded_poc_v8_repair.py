"""Two-stage preregistration and calibration-only selection; no test-label inputs to hooks."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.v8_controls import planted
from src.poc.v8_harness import TARGETS, RepairHarness, RepairProbe, RepairSpec
from src.poc.v8_repair import (
    A_GRID,
    B_GRID,
    HEADS,
    IDENTITY,
    LIMITS,
    NAIVE,
    UPSTREAM,
    fit_probe,
    grade_A,
    grade_B,
    margin,
    output_center,
    select,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v8-repair"
PROTOCOL = ROOT / "reports/bounded-poc-v8-repair-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = ROOT / f"data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/{REVISION}"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(n):
    return json.loads((OUT / n).read_text())


def save(n, x):
    with (OUT / n).open("x") as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write("\n")


def arrays(n, x):
    with (OUT / n).open("xb") as f:
        np.savez_compressed(f, **x)


def load(n):
    with np.load(OUT / n, allow_pickle=False) as f:
        return dict(f)


def protected():
    inventory = json.loads((ROOT / "v8-protected.json").read_text())
    for item in inventory:
        assert sha(ROOT / item["copy"]) == item["sha256"], item["copy"]
        assert sha(Path(item["original"])) == item["sha256"], item["original"]
    for version in (4, 5, 6, 7):
        for item in json.loads((ROOT / f"v{version}-protected.json").read_text()):
            assert sha(Path(item["original"])) == item["sha256"]
    return len(inventory)


def verify(stage=1):
    names = ["pre-calibration-manifest.json"] + (["manifest.json"] if stage == 2 else [])
    for name in names:
        m = read(name)
        for file, digest in m["sha256"].items():
            assert sha(ROOT / file) == digest, file
        assert m["heads"] == HEADS and m["upstream"] == UPSTREAM and m["late_reference"] == "24:7"
        assert m["limits"] == LIMITS
    validate(read("dataset.json")["rows"])
    protected()


def freeze(stage):
    protected()
    assert read(f"validation-stage{stage}.json")["passed"]
    if stage == 1:
        assert not list(OUT.glob("*started.json"))
        assert read("known-controls.json")["passed"]
        files = [
            p
            for folder in ("src", "scripts", "tests")
            for p in sorted((ROOT / folder).rglob("*.py"))
        ]
        files += [
            PROTOCOL,
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            OUT / "dataset.json",
            OUT / "known-controls.json",
        ]
        files += (
            sorted(ROOT.glob("v*-protected.json"))
            + sorted(MODEL.glob("*.json"))
            + sorted(MODEL.glob("*.safetensors"))
        )
        name = "pre-calibration-manifest.json"
    else:
        verify(1)
        assert not list(OUT.glob("confirm*-started.json"))
        assert read("calibration.json")["completed_utc"]
        files = (
            [OUT / "pre-calibration-manifest.json"]
            + sorted(OUT.glob("cal-*.npz"))
            + [OUT / "calibration.json", OUT / "calibration-started.json", OUT / "calibration.log"]
        )
        name = "manifest.json"
    files += [OUT / f"validation-stage{stage}.json"] + sorted(
        OUT.glob(f"*-stage{stage}-validation.log")
    )
    record = dict(
        registered_utc=now(),
        heads=HEADS,
        upstream=UPSTREAM,
        late_reference="24:7",
        limits=LIMITS,
        grid_A=A_GRID,
        grid_B=B_GRID,
        identity=IDENTITY,
        naive=NAIVE,
        model_revision=REVISION,
        device="mps",
        dtype="bfloat16",
        batch_size=64,
        seed=42,
        stage=stage,
        sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
    )
    if stage == 2:
        record["selected"] = read("calibration.json")["selected"]
    save(name, record)
    if stage == 1:
        for p in files:
            if p.suffix in (".py", ".md"):
                dest = OUT / "frozen-source" / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
    print(name, sha(OUT / name), flush=True)


def pack(results):
    data = dict(logits=np.array([r.logits for r in results], np.float32))
    for field in (
        "h",
        "q",
        "k",
        "v",
        "s",
        "p",
        "routed",
        "delta",
        "applied",
        "reconstruction_error",
        "sibling_error",
    ):
        values = []
        for r in results:
            hs = []
            for layer, h in TARGETS:
                a = np.asarray(r.repair[f"{layer}:{h}"][field], np.float32)
                if field in ("k", "v", "s", "p", "routed"):
                    a = np.pad(a, ((0, 128 - len(a)),) + ((0, 0),) * (a.ndim - 1))
                hs.append(a)
            values.append(hs)
        data[field] = np.asarray(values, np.float32)
    return data


async def execute(worker, rows, rule, constants=None):
    # Only syntax positions, own tokens, output vocabulary and frozen constants cross this boundary.
    worker.harness.constants = constants
    spec = RepairSpec(
        (), mode=rule["mode"], alpha=rule["alpha"], beta=rule["beta"], alpha22=rule["alpha22"]
    )
    probes = [
        RepairProbe(
            r["text"],
            tuple(r["token_ids"]),
            tuple(r["output_ids"]),
            spec,
            color_positions=tuple(r["color_positions"]),
            key_positions=tuple(r["key_positions"]),
        )
        for r in rows
    ]
    return pack(await worker.map(probes))


def constants(rows, base):
    def slot_means(field, positions):
        return (
            np.stack([base[field][i, :3, r[positions]].T for i, r in enumerate(rows)])
            .mean(0, dtype=np.float64)
            .astype(np.float32)
        )

    return dict(
        mu=base["h"][:, :3].mean(0, dtype=np.float64).astype(np.float32),
        mean_s=slot_means("s", "color_positions"),
        mean_p=slot_means("p", "color_positions"),
        key_mean_s=slot_means("s", "key_positions"),
        key_mean_p=slot_means("p", "key_positions"),
        mean_logits=base["logits"].mean(0, dtype=np.float64),
    )


def projection_weights(harness, ids):
    model = harness.model
    # Static decoder weights, no example labels or test-derived statistics.
    output = model.lm_head.weight[list(ids)].detach().float().cpu().numpy()
    norm = model.model.norm.weight.detach().float().cpu().numpy()
    wo = np.stack(
        [
            model.model.layers[layer]
            .self_attn.o_proj.weight[:, h * harness.head_dim : (h + 1) * harness.head_dim]
            .detach()
            .float()
            .cpu()
            .numpy()
            for layer, h in TARGETS
        ]
    )
    directions = np.einsum(
        "hid,i->hd",
        wo.astype(np.float64),
        norm.astype(np.float64) * (output[0].astype(float) - output[1].astype(float)),
    )
    return dict(wo=wo, norm=norm, unembedding=output, directions=directions)


async def calibrate(worker):
    dataset = read("dataset.json")["rows"]
    summary = dict(
        selected={},
        candidate_scores={},
        pre_manifest_sha256=sha(OUT / "pre-calibration-manifest.json"),
    )
    for track, grid in [("A", A_GRID), ("B", B_GRID)]:
        rows = [r for r in dataset if r["block"] == track + "_calibration"]
        base = await execute(worker, rows, IDENTITY)
        arrays(f"cal-{track}-baseline.npz", base)
        con = constants(rows, base)
        arrays(f"cal-{track}-constants.npz", con)
        identity = await execute(worker, rows, IDENTITY, con)
        arrays(f"cal-{track}-identity.npz", identity)
        assert np.array_equal(base["logits"], identity["logits"])
        candidates = []
        if track == "A":
            for alpha in (1.0, 1.5, 2.0, 2.5):
                raw = await execute(worker, rows, dict(IDENTITY, alpha=alpha), con)
                arrays(f"cal-A-alpha{alpha}.npz", raw)
                candidates += [
                    output_center(raw["logits"], con["mean_logits"], gamma)
                    for gamma in (0.0, 0.5, 1.0)
                ]
            w, b, lgo = fit_probe(rows, base["h"][:, :3])
            weights = projection_weights(worker.harness, rows[0]["output_ids"])
            arrays("cal-A-weights.npz", weights)
            arrays("cal-A-probe.npz", dict(w=w, b=np.asarray(b), lgo=lgo))
            summary["probe_lgo_accuracy"] = float(
                np.mean(np.array([r["label"] for r in rows]) * lgo > 0)
            )
        else:
            for i, rule in enumerate(grid):
                a = await execute(worker, rows, rule, con)
                arrays(f"cal-B-candidate{i:02}.npz", a)
                candidates.append(a["logits"])
            naive = await execute(worker, rows, NAIVE, con)
            arrays("cal-B-naive.npz", naive)
        index, scores = select(rows, base["logits"], candidates)
        arrays(f"cal-{track}-candidate-logits.npz", dict(logits=np.stack(candidates)))
        summary["selected"][track] = dict(index=index, rule=grid[index])
        summary["candidate_scores"][track] = scores
        print(
            "Calibration", track, "selected", summary["selected"][track], scores[index], flush=True
        )
    summary["completed_utc"] = now()
    summary["microbatches"] = worker.batches
    save("calibration.json", summary)


async def confirm(worker, run):
    rows_all = read("dataset.json")["rows"]
    selection = read("manifest.json")["selected"]
    summary = dict(manifest_sha256=sha(OUT / "manifest.json"), blocks={})
    for block in ("A_primary", "B_primary", "B_vocabulary"):
        rows = [r for r in rows_all if r["block"] == block]
        track = block[0]
        con = load(f"cal-{track}-constants.npz")
        rule = selection[track]["rule"]
        prefix = f"confirm{run}-{block}"
        base = await execute(worker, rows, IDENTITY)
        arrays(prefix + "-baseline.npz", base)
        ident = await execute(worker, rows, IDENTITY, con)
        arrays(prefix + "-identity.npz", ident)
        repaired = await execute(worker, rows, rule, con)
        arrays(prefix + "-repaired.npz", repaired)
        z = output_center(repaired["logits"], con["mean_logits"], rule["gamma"])
        derived = dict(centered_logits=z)
        if track == "A":
            probe = load("cal-A-probe.npz")
            h = base["h"][:, :3].astype(float)
            score = h.reshape(len(h), -1) @ probe["w"] + probe["b"]
            directions = load("cal-A-weights.npz")["directions"]
            ov = ((h - con["mu"]) * directions[:3]).sum((1, 2))
            late = (base["h"][:, 3].astype(float) * directions[3]).sum(1)
            derived.update(probe=score, ov_readout=ov, late_reference_projection=late)
            result = grade_A(rows, base["logits"], z, score)
            result["unsupervised_ov_accuracy"] = float(
                np.mean(np.array([r["label"] for r in rows]) * ov > 0)
            )
            result["unsupervised_ov_on_errors"] = (
                float(
                    np.mean(
                        (np.array([r["label"] for r in rows]) * ov > 0)[
                            np.array([r["label"] for r in rows]) * margin(base["logits"]) <= 0
                        ]
                    )
                )
                if result["stats"]["baseline_errors"]
                else None
            )
        else:
            naive = await execute(worker, rows, NAIVE, con)
            arrays(prefix + "-naive.npz", naive)
            result = grade_B(rows, base["logits"], z, naive["logits"])
        arrays(prefix + "-derived.npz", derived)
        result["identity_error"] = float(
            np.max(abs(margin(base["logits"]) - margin(ident["logits"])))
        )
        summary["blocks"][block] = result
        print(block, json.dumps(result), flush=True)
    summary["completed_utc"] = now()
    summary["microbatches"] = worker.batches
    save(f"confirmation-{run}.json", summary)


async def run(stage):
    level = 1 if stage == "calibration" else 2
    verify(level)
    with open("/tmp/interp-swarm-mps.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            stage + "-started.json",
            dict(
                utc=now(),
                pid=os.getpid(),
                manifest_sha256=sha(
                    OUT / ("pre-calibration-manifest.json" if level == 1 else "manifest.json")
                ),
            ),
        )
        wc = WorkerConfig(batch_size=64)
        h = await asyncio.to_thread(
            RepairHarness.load,
            ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True),
            wc,
        )
        async with MPSBatchWorker(h, wc) as worker:
            if stage == "calibration":
                await calibrate(worker)
            else:
                await confirm(worker, int(stage[-1]))
    verify(level)


async def main(stage):
    if stage == "controls":
        save("known-controls.json", planted())
    elif stage in ("freeze1", "freeze2"):
        freeze(int(stage[-1]))
    elif stage == "review":
        from src.poc.review import review_v8

        await review_v8(
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["controls", "freeze1", "calibration", "freeze2", "confirm1", "confirm2", "review"],
    )
    asyncio.run(main(parser.parse_args().stage))
