"""Independent raw-evidence arithmetic audit; never changes registered decisions."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/lookup-followup-v1"


def read(name):
    return json.loads((OUT / name).read_text())


def load(name):
    with np.load(OUT / name) as file:
        return {k: file[k] for k in file.files}


def sign_probability(effects, rows):
    means = [
        np.mean([e for e, r in zip(effects, rows) if r["group"] == g])
        for g in sorted({r["group"] for r in rows})
    ]
    n, k = len(means), sum(e > 0 for e in means)
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n


def near(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def main():
    manifest = read("manifest.json")
    for name, digest in manifest["sha256"].items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            # Later repository work may append to shared modules. Use only a
            # byte-exact snapshot matching the original preregistration hash.
            path = OUT / "frozen-source" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    old_limits = json.loads((ROOT / "data/bounded-poc-v1/manifest.json").read_text())["limits"]
    assert manifest["limits"] == old_limits, "Original numerical bar changed"
    all_rows = read("dataset.json")["rows"]
    outputs = read("dataset.json")["output_ids"]
    old_rows = json.loads((ROOT / "data/bounded-poc-v1/dataset.json").read_text())["rows"]
    assert not {k for r in old_rows for k in r["keys"]} & {k for r in all_rows for k in r["keys"]}
    assert not {r["text"] for r in old_rows} & {r["text"] for r in all_rows}
    for r in all_rows:
        assert sorted(r["token_ids"][i] for i in r["value_positions"]) == sorted(outputs)
        assert r["value_positions"][-1] < r["suffix_start"]
    detail = {}
    cal = read("calibration.json")
    for name, split in (
        ("calibration", "calibration"),
        ("confirm1", "confirmation"),
        ("confirm2", "confirmation"),
    ):
        rows = [r for r in all_rows if r["split"] == split]
        saved, raw = read(f"{name}.json"), load(f"{name}-baseline.npz")
        controls = load(f"{name}-controls.npz")
        y = np.array([r["label"] for r in rows])
        index = {r["id"]: i for i, r in enumerate(rows)}
        opposite = [index[r["opposite_id"]] for r in rows]
        for r, j in zip(rows, opposite):
            assert Counter(r["token_ids"]) == Counter(rows[j]["token_ids"])
            assert r["token_ids"][-8:] == rows[j]["token_ids"][-8:]
            assert r["label"] == -rows[j]["label"]
        margin = raw["logits"][:, 0] - raw["logits"][:, 1]
        near(np.mean(y * margin > 0), saved["model_accuracy"])
        assert int(np.sum(y * margin > 0)) == saved["model_correct"]
        near(
            np.max(np.abs(controls["residual"] - raw["logits"][opposite])),
            saved["instrument_controls"]["residual_max"],
        )
        for key, check in (("sham", "sham_max"), ("reconstruction", "reconstruction_max")):
            near(
                np.max(np.abs(controls[key][:, 0] - controls[key][:, 1] - margin)),
                saved["instrument_controls"][check],
            )
        if name == "calibration":
            continue
        scores = raw["heads"] @ cal["readout"]["direction"] - cal["readout"]["threshold"]
        predicted = np.sign(scores)
        stats = saved["evaluation"]["statistics"]
        near(predicted, stats["predictions"])
        near(np.mean(predicted == y), stats["readout_accuracy"])
        gap = float(np.median(y * (margin - margin[opposite])))
        near(gap, stats["clean_counterfactual_gap"])
        head = load(f"{name}-head.npz")
        effect = y * (margin - head["opposite"])
        mean_effect = y * (margin - head["mean"])
        same_effect = y * (margin - head["same"])
        near(np.median(effect), stats["patch_median"])
        near(np.mean(effect > 0), stats["patch_direction_rate"])
        near(np.median(effect) / gap if gap > 0 else 0.0, stats["patch_fraction"])
        near(np.median(mean_effect), stats["mean_median"])
        near(np.mean(mean_effect > 0), stats["mean_direction_rate"])
        near(np.median(effect - same_effect), stats["specificity_median"])
        near(sign_probability(effect, rows), stats["causal_group_sign_p"])
        for group_key, value in stats["subgroup_accuracy"].items():
            key, v = group_key.split("=")
            subset = [i for i, r in enumerate(rows) if str(r[key]) == v]
            near(np.mean(predicted[subset] == y[subset]), value)
        # Independent recreation of frozen group-flip null.
        groups = sorted({r["group"] for r in rows})
        gi = np.array([groups.index(r["group"]) for r in rows])
        flips = np.random.default_rng(20261020).choice([-1, 1], (4095, len(groups)))
        null = np.mean(predicted[None] == y[None] * flips[:, gi], axis=1)
        near((1 + np.sum(null >= np.mean(predicted == y))) / 4096, stats["readout_label_flip_p"])
        paths = load(f"{name}-paths.npz")
        for path, patched in paths.items():
            e = y * (margin - patched)
            p = saved["mechanism"]["paths"][path]
            near(np.median(e), p["median"])
            near(np.mean(e > 0), p["direction"])
            near(np.median(e) / gap if gap > 0 else 0.0, p["fraction"])
            near(sign_probability(e, rows), p["group_p"])
            near(np.median(abs(e)), p["absolute_median"])
            near(np.quantile(abs(e), 0.9), p["absolute_p90"])
            assert p["passed"] == (
                np.mean(e > 0) >= 0.8
                and np.median(e) >= 0.25
                and gap > 0
                and np.median(e) / gap >= 0.10
                and sign_probability(e, rows) <= 0.05 / 8
            )
            assert p["negligible"] == (
                np.median(abs(e)) <= 0.1 and np.quantile(abs(e), 0.9) <= 0.25
            )
        masses = dict(relevant=[], distractor=[], suffix=[], remaining_prefix=[])
        for r, attn in zip(rows, raw["attention"]):
            n, start = len(r["token_ids"]), r["suffix_start"]
            positions = r["value_positions"]
            masses["relevant"].append(attn[positions[r["query_record"]]])
            masses["distractor"].append(attn[positions[1 - r["query_record"]]])
            masses["suffix"].append(attn[start:n].sum())
            masses["remaining_prefix"].append(attn[:start].sum() - attn[positions].sum())
            np.testing.assert_allclose(attn.sum(), 1.0, atol=1e-6)
        delta = np.array(masses["relevant"]) - masses["distractor"]
        routing = saved["mechanism"]["routing"]
        near(np.median(delta), routing["attention_difference_median"])
        near(np.mean(delta > 0), routing["attention_direction"])
        de = y * (paths["drop_distractor"] - paths["drop_relevant"])
        near(np.median(de), routing["deletion_specificity_median"])
        near(sign_probability(de, rows), routing["deletion_group_p"])
        p = saved["mechanism"]["paths"]
        source = (
            p["source_swap"]["passed"]
            and p["q_query"]["passed"]
            and np.median(delta) >= 0.1
            and np.mean(delta > 0) >= 0.8
            and np.median(de) >= 0.1
            and sign_probability(de, rows) <= 0.05 / 8
        )
        upstream = p["q_assignment"]["passed"] or p["v_suffix"]["passed"]
        classification = (
            "mixed"
            if source and upstream
            else "source routing supported; exclusive computation unresolved"
            if source
            else "upstream-query relay supported"
            if upstream and p["source_swap"]["negligible"]
            else "unresolved"
        )
        assert classification == saved["mechanism"]["classification"]
        detail[name] = dict(
            attention_mass_descriptive={
                k: dict(mean=float(np.mean(v)), median=float(np.median(v)))
                for k, v in masses.items()
            },
            model_wrong_count=int(np.sum(y * margin <= 0)),
            head_readout_correct_when_model_wrong=int(np.sum((predicted == y) & (y * margin <= 0))),
        )
    for suffix in ("baseline", "head", "paths", "controls"):
        a, b = load(f"confirm1-{suffix}.npz"), load(f"confirm2-{suffix}.npz")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    result = read("results.json")
    assert result["formal_survivor"] == all(result["criteria"].values())
    assert not result["criteria"]["behavior"]  # observed failed control, preserved explicitly
    record = dict(
        passed=True,
        checks="Unchanged old limits; all frozen hashes; donor matching; actual value indices; raw control errors; predictive/causal/path statistics; group tests; mechanism classification; bit-identical fresh-process arrays.",
        raw_result_sha256=hashlib.sha256((OUT / "results.json").read_bytes()).hexdigest(),
        descriptive_only=detail,
    )
    if (OUT / "audit.json").exists():
        assert record == read("audit.json"), "Raw evidence changed since the original audit"
    else:
        with (OUT / "audit.json").open("x") as file:
            file.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
