"""Frozen statistics. No thresholds are estimated from confirmation data."""

import math

import numpy as np

LIMITS = {
    "model_accuracy": 0.90,
    "discovery_cv": 0.75,
    "readout_accuracy": 0.85,
    "subgroup_accuracy": 0.80,
    "baseline_advantage": 0.15,
    "patch_direction_rate": 0.80,
    "patch_median": 0.25,
    "patch_fraction": 0.10,
    "mean_direction_rate": 0.70,
    "mean_median": 0.10,
    "specificity_median": 0.10,
    "p": 0.05 / 3,
    "sham_max": 0.05,
    "residual_max": 0.05,
    "replication_max": 0.15,
    "replication_agreement": 0.95,
}


def fit_readout(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    plus, minus = x[y == 1].mean(0), x[y == -1].mean(0)
    direction = plus - minus
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    return {
        "direction": direction.tolist(),
        "threshold": float(((plus + minus) / 2) @ direction),
        "mean": x.mean(0).tolist(),
    }


def predict(x, readout):
    scores = np.asarray(x) @ np.asarray(readout["direction"]) - readout["threshold"]
    # Exact ties abstain and count as wrong.
    return np.sign(scores)


def cross_validate(x, rows):
    x = np.asarray(x)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    predicted = np.zeros(len(rows))
    for g in sorted(set(groups)):
        test = groups == g
        model = fit_readout(x[~test], y[~test])
        predicted[test] = predict(x[test], model)
    return float(np.mean(predicted == y))


def grouped_values(values, rows):
    return np.array(
        [
            np.mean([v for v, r in zip(values, rows) if r["group"] == g])
            for g in sorted({r["group"] for r in rows})
        ]
    )


def sign_p(values):
    n = len(values)
    positives = sum(v > 0 for v in values)
    return sum(math.comb(n, k) for k in range(positives, n + 1)) / 2**n


def label_flip_p(predicted, rows):
    y = np.array([r["label"] for r in rows])
    groups = sorted({r["group"] for r in rows})
    group_idx = np.array([groups.index(r["group"]) for r in rows])
    rng = np.random.default_rng(20261020)
    signs = rng.choice([-1, 1], size=(4095, len(groups)))
    null = np.mean(predicted[None, :] == y[None, :] * signs[:, group_idx], axis=1)
    observed = np.mean(predicted == y)
    return float((1 + np.sum(null >= observed)) / 4096)


def evaluate(rows, vectors, readout, margins, opposite, mean, same, sham, baselines):
    y = np.array([r["label"] for r in rows])
    base = np.asarray(margins)
    pred = predict(vectors, readout)
    accuracy = float(np.mean(pred == y))
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    donor_base = np.array([base[by_id[r["opposite_id"]]] for r in rows])
    gap = float(np.median(y * (base - donor_base)))
    effects = y * (base - np.asarray(opposite))
    mean_effects = y * (base - np.asarray(mean))
    same_effects = y * (base - np.asarray(same))
    subgroup = {
        f"{key}={v}": float(np.mean([p == r["label"] for p, r in zip(pred, rows) if r[key] == v]))
        for key in ("label", "format")
        for v in sorted({r[key] for r in rows})
    }
    best_baseline = max(v for k, v in baselines.items() if "positive_control" not in k)
    stats = {
        "readout_accuracy": accuracy,
        "subgroup_accuracy": subgroup,
        "baseline_advantage": accuracy - best_baseline,
        "patch_direction_rate": float(np.mean(effects > 0)),
        "patch_median": float(np.median(effects)),
        "clean_counterfactual_gap": gap,
        "patch_fraction": float(np.median(effects) / gap) if gap > 0 else 0,
        "mean_direction_rate": float(np.mean(mean_effects > 0)),
        "mean_median": float(np.median(mean_effects)),
        "specificity_median": float(np.median(effects - same_effects)),
        "causal_group_sign_p": sign_p(grouped_values(effects, rows)),
        "readout_label_flip_p": label_flip_p(pred, rows),
        "sham_max": float(np.max(np.abs(base - sham))),
        "predictions": pred.tolist(),
    }
    checks = {
        key: stats[key] >= LIMITS[key]
        for key in (
            "readout_accuracy",
            "baseline_advantage",
            "patch_direction_rate",
            "patch_median",
            "patch_fraction",
            "mean_direction_rate",
            "mean_median",
            "specificity_median",
        )
    }
    checks["subgroups"] = min(subgroup.values()) >= LIMITS["subgroup_accuracy"]
    checks["causal_significance"] = stats["causal_group_sign_p"] <= LIMITS["p"]
    checks["predictive_significance"] = stats["readout_label_flip_p"] <= LIMITS["p"]
    checks["sham"] = stats["sham_max"] <= LIMITS["sham_max"]
    return {"passed": all(checks.values()), "checks": checks, "statistics": stats}


# These gates belong only to v2; v1's LIMITS and evaluator remain unchanged.
V2_LIMITS = {
    "model_accuracy": 0.90,
    "discovery_cv": 0.75,
    "readout_accuracy": 0.85,
    "subgroup_accuracy": 0.80,
    "baseline_advantage": 0.15,
    "solo_direction": 0.80,
    "solo_median": 0.25,
    "solo_fraction": 0.10,
    "coalition_direction": 0.90,
    "coalition_fraction": 0.35,
    "coalition_median": 0.25,
    "coalition_gain_over_best_solo": 0.10,
    "transfer_direction": 0.80,
    "transfer_median": 0.25,
    "rescue_median_gain": 0.10,
    "rescue_min_errors_fixed": 1,
    "rescue_max_regressions": 0,
    "identity_max": 0.0,
    "residual_max": 0.05,
    "path_reconstruction_max": 0.05,
    "path_full_margin_error_max": 0.10,
    "delta_closure_max": 2e-5,
    "p": 0.05 / 4,
    "replication_max": 0.0,
}


def _v2_margins(rows, margins):
    values = np.asarray(margins, dtype=float)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("Expected one finite margin per row")
    if not rows:
        raise ValueError("A v2 measurement needs at least one row")
    return values


def effect_stats(rows, base_margin, patched_margin):
    """Correct-answer margin reductions, with whole-group sign inference."""
    base = _v2_margins(rows, base_margin)
    patched = _v2_margins(rows, patched_margin)
    labels = np.array([row["label"] for row in rows])
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate measurement row identifiers")
    donor = np.array([by_id[row["opposite_id"]] for row in rows])
    if not np.array_equal(labels[donor], -labels):
        raise ValueError("Counterfactual donors must have opposite answers")
    effects = labels * (base - patched)
    gap = float(np.median(labels * (base - base[donor])))
    median = float(np.median(effects))
    return {
        "direction_rate": float(np.mean(effects > 0)),
        "median": median,
        "clean_gap": gap,
        "fraction": median / gap if gap > 0 else 0.0,
        "group_sign_p": sign_p(grouped_values(effects, rows)),
        "effects": effects.tolist(),
    }


def task_stats(rows, margin):
    values = _v2_margins(rows, margin)
    labels = np.array([row["label"] for row in rows])
    correct_margins = labels * values
    correct = correct_margins > 0

    def counts(mask):
        n = int(np.sum(mask))
        k = int(np.sum(correct & mask))
        return {"correct": k, "total": n, "accuracy": k / n}

    result = counts(np.ones(len(rows), dtype=bool))
    for field, name in (
        ("format", "by_format"),
        ("query_record", "by_query"),
        ("answer_color", "by_color"),
    ):

        def value(row, key=field):
            return row.get(key, "positive" if row["label"] == 1 else "negative")

        result[name] = {
            str(category): counts(np.array([value(row) == category for row in rows]))
            for category in sorted({value(row) for row in rows}, key=str)
        }
    return result | {
        "median_correct_margin": float(np.median(correct_margins)),
        "correct_answer_margins": correct_margins.tolist(),
        "error_ids": [row["id"] for row, ok in zip(rows, correct, strict=True) if not ok],
    }


def readout_metrics_v2(rows, vectors, readout):
    """Frozen readout gates; no refitting on held-out examples or new values."""
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(rows) or not np.isfinite(x).all():
        raise ValueError("Expected one finite head vector per measurement row")
    labels = np.array([row["label"] for row in rows])
    predicted = predict(x, readout)
    accuracy = float(np.mean(predicted == labels))
    subgroup = {
        f"{field}={category}": float(
            np.mean(
                [
                    guess == row["label"]
                    for guess, row in zip(predicted, rows, strict=True)
                    if row[field] == category
                ]
            )
        )
        for field in ("label", "format")
        for category in sorted({row[field] for row in rows})
    }
    p = label_flip_p(predicted, rows)
    statistics = {
        "readout_accuracy": accuracy,
        "subgroup_accuracy": subgroup,
        "baseline_advantage": accuracy - 0.5,
        "label_flip_p": p,
        "predictions": predicted.tolist(),
    }
    checks = {
        "readout_accuracy": accuracy >= V2_LIMITS["readout_accuracy"],
        "subgroups": min(subgroup.values()) >= V2_LIMITS["subgroup_accuracy"],
        "baseline_advantage": accuracy - 0.5 >= V2_LIMITS["baseline_advantage"],
        "predictive_significance": p <= V2_LIMITS["p"],
    }
    return {"passed": all(checks.values()), "checks": checks, "statistics": statistics}


def decomposition_metrics_v2(rows, base_margin, modes):
    """Classify measured source/routing contrasts conservatively.

    Components overlap through QK-by-V interaction. Ratios serve classification
    rules only and must not be described as percentages of causal mediation.
    """
    required = {"source_color", "source_other", "ov_all", "qk", "qkov"}
    if required - set(modes):
        raise ValueError(f"Missing decomposition measurements: {sorted(required - set(modes))}")
    statistics = {name: effect_stats(rows, base_margin, values) for name, values in modes.items()}
    full = statistics["qkov"]["median"]

    def classify(first, second, first_name, second_name):
        a, b = statistics[first], statistics[second]
        positive_a = a["median"] >= 0.25 and a["direction_rate"] >= 0.80
        positive_b = b["median"] >= 0.25 and b["direction_rate"] >= 0.80
        if positive_a and a["median"] >= 0.5 * full and a["median"] >= b["median"] + 0.10:
            return first_name
        if positive_b and b["median"] >= 0.5 * full and b["median"] >= a["median"] + 0.10:
            return second_name
        if positive_a and positive_b:
            return "mixed"
        return "unresolved"

    source = classify("source_color", "source_other", "direct", "relay")
    routing = classify("ov_all", "qk", "ov", "pattern")
    return {
        "source_classification": source,
        "routing_classification": routing,
        "classifiable": source != "unresolved" and routing != "unresolved",
        "statistics": statistics,
        "meaning": "Overlapping intervention contrasts; no percentage-mediation claim.",
    }


def rescue_stats_v2(rows, base_margin, rescued_margin):
    base = _v2_margins(rows, base_margin)
    rescued = _v2_margins(rows, rescued_margin)
    labels = np.array([row["label"] for row in rows])
    before, after = labels * base, labels * rescued
    fixed = int(np.sum((before <= 0) & (after > 0)))
    regressions = int(np.sum((before > 0) & (after <= 0)))
    gain = float(np.median(after - before))
    checks = {
        "errors_fixed": fixed >= V2_LIMITS["rescue_min_errors_fixed"],
        "no_regressions": regressions <= V2_LIMITS["rescue_max_regressions"],
        "median_gain": gain >= V2_LIMITS["rescue_median_gain"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "before": task_stats(rows, base),
        "after": task_stats(rows, rescued),
        "errors_fixed": fixed,
        "regressions": regressions,
        "median_gain": gain,
    }
