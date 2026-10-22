"""Post hoc Axis C Q-only diagnostics; never writes existing v4 evidence."""

import asyncio
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.audit_bounded_poc_v4_confirm import arithmetic, attention_stats, statistics
from scripts.run_bounded_poc_v4_confirm import (
    MODEL,
    OUT,
    REVISION,
    ROOT,
    TARGETS,
    UPSTREAM,
    ModelConfig,
    MPSBatchWorker,
    V4Harness,
    WorkerConfig,
    execute,
    margin,
    payloads,
    verify,
)

DEST = ROOT / "data/bounded-poc-v4-axis-c-q-diagnostic"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    with (DEST / name).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def save_arrays(name, obj):
    with (DEST / name).open("xb") as f:
        np.savez_compressed(f, **obj)


async def main():
    verify()
    inventory = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(
            list(OUT.rglob("*")) + list((ROOT / "reports").glob("bounded-poc-v4-confirm*.md"))
        )
        if p.is_file()
    }
    DEST.mkdir(exist_ok=False)
    write("existing-v4-sha256.json", inventory)
    write(
        "diagnostic-plan.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "post hoc diagnostic; no regrading of v4-confirm",
            "script_sha256": digest(Path(__file__)),
            "upstream": UPSTREAM,
            "downstream": [f"{layer}:{head}" for layer, head in TARGETS],
            "axis": "C: context-key swap, same query and color positions",
            "device": "mps",
            "dtype": "bfloat16",
            "batch_size": 64,
            "metrics": "median signed margin reduction; positive fraction; median of per-prompt three-head mean donor gain and clean loss",
            "C_q_final": "Donor Q_final with recipient clean K, V and measured H anchors",
            "C_upstream_Q": "Axis C upstream donor outputs at final token; downstream outputs clamped clean in mediator; replay mediator Q_final onto original clean K and V",
            "processes": 1,
            "no_new_success_thresholds": True,
        },
    )
    data = json.loads((OUT / "dataset.json").read_text())
    summary = {"post_hoc": True, "splits": {}, "existing_v4_files": len(inventory)}
    members = tuple(tuple(map(int, key.split(":"))) for key in UPSTREAM)
    with open("/tmp/interp-swarm-mps.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write("started.json", {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
        wc = WorkerConfig(batch_size=64)
        harness = await asyncio.to_thread(
            V4Harness.load,
            ModelConfig(name=str(MODEL), revision=REVISION, local_files_only=True),
            wc,
        )
        async with MPSBatchWorker(harness, wc) as worker:
            for split in ("primary", "vocabulary"):
                rows = [r for r in data["rows"] if r["split"] == split]
                outputs = data["output_ids"][split]
                base = await execute(worker, rows, outputs, full=True)
                save_arrays(f"{split}-baseline.npz", base)
                with np.load(OUT / f"confirm1-{split}-baseline.npz", allow_pickle=False) as old:
                    assert set(base) == set(old.files)
                    for key in base:
                        assert base[key].dtype == old[key].dtype
                        assert base[key].shape == old[key].shape
                        assert base[key].tobytes() == old[key].tobytes(), (split, key)
                identity = await execute(
                    worker,
                    rows,
                    outputs,
                    "q_final",
                    payloads(rows, base, axis="C", self_patch=True),
                    TARGETS,
                )
                save_arrays(f"{split}-self-Q.npz", identity)
                np.testing.assert_array_equal(identity["logits"], base["logits"])
                direct = await execute(
                    worker, rows, outputs, "q_final", payloads(rows, base, axis="C"), TARGETS
                )
                save_arrays(f"{split}-C_q_final.npz", direct)
                mid = await execute(
                    worker,
                    rows,
                    outputs,
                    "mediate",
                    payloads(rows, base, axis="C"),
                    upstream=members,
                )
                save_arrays(f"{split}-C-mediator.npz", mid)
                np.testing.assert_array_equal(mid["applied"], base["h"])
                for i, row in enumerate(rows):
                    n = len(row["token_ids"])
                    np.testing.assert_array_equal(mid["k"][i, :, : n - 1], base["k"][i, :, : n - 1])
                mediated = await execute(
                    worker,
                    rows,
                    outputs,
                    "q_final",
                    payloads(rows, base, axis="C", donor_q=mid["q"]),
                    TARGETS,
                )
                save_arrays(f"{split}-C_upstream_Q.npz", mediated)
                measured = {}
                labels = np.array([r["label"] for r in rows])
                for name, arr, donor_q in (
                    ("C_q_final", direct, None),
                    ("C_upstream_Q", mediated, mid["q"]),
                ):
                    error = arithmetic(rows, base, arr, "C", "q_final", donor_q=donor_q)
                    effect = statistics(rows, labels * (margin(base) - margin(arr)))
                    # Measure the pure-Q replay pattern, never the mediator's unrestricted QK pattern.
                    attention = attention_stats(rows, base["p"], arr["routed_p"], "C")
                    measured[name] = {
                        "effect": effect,
                        "attention": attention,
                        "numpy_delta_max_error": error,
                    }
                    coal = attention["coalition"]
                    print(
                        f"{split} {name}: median reduction={effect['median']:.9f} logits; "
                        f"direction={effect['direction'] * 100:.6f}%; "
                        f"coalition donor gain={coal['gain'] * 100:.6f} pp; "
                        f"clean loss={coal['loss'] * 100:.6f} pp; "
                        f"joint attention direction={coal['joint_direction'] * 100:.6f}%",
                        flush=True,
                    )
                summary["splits"][split] = {
                    "n": len(rows),
                    "baseline_bit_identical_to_v4": True,
                    "identity_max_logit_error": 0.0,
                    "measures": measured,
                }
            summary["microbatches"] = worker.batches
    verify()
    assert set(inventory) == {
        str(p.relative_to(ROOT))
        for p in list(OUT.rglob("*")) + list((ROOT / "reports").glob("bounded-poc-v4-confirm*.md"))
        if p.is_file()
    }
    for name, expected in inventory.items():
        assert digest(ROOT / name) == expected, name
    summary.update(
        existing_v4_unchanged=True,
        protected_artifacts_unchanged=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    write("results.json", summary)
    print(
        f"Verified {len(inventory)} existing v4 files unchanged; all prior protection hashes unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
