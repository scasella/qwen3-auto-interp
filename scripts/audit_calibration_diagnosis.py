"""Independent checks of raw replay, error accounting, and immutable old evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/calibration-diagnosis-v1"
OLD = ROOT / "data/lookup-followup-v1"


def read(path):
    return json.loads(path.read_text())


def main():
    lock = read(OUT / "plan-lock.json")
    for name, digest in lock["original_evidence"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in lock["source"].items():
        assert hashlib.sha256((OUT / "source" / name).read_bytes()).hexdigest() == digest
    original = read(OLD / "manifest.json")
    for name, digest in original["sha256"].items():
        if name.startswith("data/hf-cache/"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    data = read(OLD / "dataset.json")["rows"]
    analysis = read(OUT / "analysis.json")
    extra = {}
    assert original["limits"]["model_accuracy"] == 0.9
    assert analysis["original_accuracy_gate"] == 0.9
    assert max(read(OUT / "module-shams.json").values()) == 0
    for split in ["calibration", "confirmation"]:
        rows = [r for r in data if r["split"] == split]
        y = np.array([r["label"] for r in rows])
        b = np.load(OUT / f"{split}-baseline.npz")
        rep = np.load(OUT / f"repeat-{split}-baseline.npz")
        old = np.load(
            OLD
            / ("calibration-baseline.npz" if split == "calibration" else "confirm1-baseline.npz")
        )
        np.testing.assert_array_equal(b["logits"], old["logits"])
        np.testing.assert_array_equal(b["heads"][:, 5], old["heads"])
        for k in b.files:
            np.testing.assert_array_equal(b[k], rep[k])
        patches = dict(np.load(OUT / f"{split}-interventions.npz"))
        if split == "calibration":
            patches.update(dict(np.load(OUT / "calibration-heads.npz")))
        rep_patch = np.load(OUT / f"repeat-{split}-interventions.npz")
        for k in rep_patch.files:
            np.testing.assert_array_equal(patches[k], rep_patch[k])
        m = b["logits"][:, 0] - b["logits"][:, 1]
        before = y * m > 0
        s = analysis["splits"][split]
        assert int(before.sum()) == s["model_correct"]
        for key, patched in patches.items():
            after = y * patched > 0
            stat = s["interventions"][key]
            assert int((~before & after).sum()) == stat["rescued"]
            assert int((before & ~after).sum()) == stat["regressed"]
            assert int(after.sum()) == stat["correct"]
            np.testing.assert_allclose(
                np.median((y * (patched - m))[~before]),
                stat["gain_errors_median"],
                rtol=0,
                atol=1e-12,
            )
        # Direct projection reconstruction includes actual residual-rounding increments.
        d = b["states"][..., 0]
        rms = b["states"][..., -1]
        raw = d[:, 0, 0].copy()
        for layer in range(28):
            raw += d[:, layer, 1] - d[:, layer, 0]
            raw += d[:, layer, 2] - d[:, layer, 1]
        np.testing.assert_allclose(raw, d[:, -1, -1], rtol=0, atol=1e-10)
        residual_error = np.max(abs(raw / rms[:, -1, -1] - m))
        assert residual_error < 0.05
        raw_head = b["head_dot"][:, 24, 7] / rms[:, -1, -1]
        extra[split] = dict(
            head24_7_gold_positive=int((raw_head > 0).sum()),
            n=len(rows),
            head24_7_projection_range=[float(raw_head.min()), float(raw_head.max())],
            original_gold_correct=int(np.sum(before & (y == 1))),
            original_silver_correct=int(np.sum(before & (y == -1))),
            final_norm_rounding_max=float(residual_error),
        )
        if split == "calibration":
            assert int(before.sum()) == 55
            assert np.all(y[~before] == -1)
            assert max(abs(patches["source:reconstruct"] - m)) <= 0.05
            assert np.all((y * (patches["head19:mean"] - m))[~before] < 0)
            assert np.all((y * (patches["source:source_swap"] - m))[~before] < 0)
            # Independent leave-group-out nearest-centroid classification.
            x = b["heads"][:, 5]
            pred = np.zeros(len(rows))
            for group in sorted({r["group"] for r in rows}):
                test = np.array([r["group"] == group for r in rows])
                positive = x[(~test) & (y == 1)].mean(0)
                negative = x[(~test) & (y == -1)].mean(0)
                direction = positive - negative
                direction /= np.linalg.norm(direction)
                threshold = ((positive + negative) / 2) @ direction
                pred[test] = np.sign(x[test] @ direction - threshold)
            assert np.all(pred[~before] == y[~before])
            assert int((pred == y).sum()) == 62
    result = dict(
        passed=True,
        old_evidence_files_unchanged=len(lock["original_evidence"]),
        old_gate_still_failed=True,
        checkpoint_hashes_verified=True,
        controls="Exact original logits and head vectors; zero module self-patch error; exact baseline and planned intervention repeats; independently counted rescues/regressions; leave-group-out decoding; residual accounting; exact source snapshots.",
        note="Only the planned mean interventions and calibration head19 source swap were repeated. Zero and opposite-donor interventions were not repeated.",
        additional_descriptive_checks=extra,
    )
    with (OUT / "audit.json").open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
