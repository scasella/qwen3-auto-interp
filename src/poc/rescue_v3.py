"""Frozen v3 rescue rules. No head search and no held-out rule selection."""

import numpy as np

from src.poc.analysis import V2_LIMITS, effect_stats, rescue_stats_v2, task_stats

HEADS = ("19:5", "22:9", "21:13", "27:14")
COALITION = HEADS[:3]
INTERFERER = HEADS[-1]
LIMITS = dict(V2_LIMITS)
RULES = tuple(
    {"id": f"center_{alpha}_late_{int(late)}", "alpha": alpha, "late": late}
    for alpha in (1.5, 2.0)
    for late in (False, True)
)
IDENTITY = {"id": "identity", "alpha": 1.0, "late": False}


def margin(logits):
    x = np.asarray(logits)
    return x[..., 0] - x[..., 1]


def select_rule(candidates):
    """Zero discovery regressions first; no rescue of a disqualified candidate."""
    safe = [c for c in candidates if c["stats"]["regressions"] == 0]
    if not safe:
        return dict(IDENTITY)
    chosen = min(
        safe,
        key=lambda c: (-c["stats"]["after"]["correct"], -c["stats"]["median_gain"], c["id"]),
    )
    return {key: chosen[key] for key in ("id", "alpha", "late")}


def control_stats(base, control):
    self_error = float(np.max(np.abs(base["logits"] - control["self"])))
    residual_error = float(np.max(np.abs(control["residual"] - control["donor_logits"])))
    return {
        "self_max": self_error,
        "residual_max": residual_error,
        "passed": self_error == LIMITS["identity_max"] and residual_error <= LIMITS["residual_max"],
    }


def block_stats(rows, base, controls, effects, rescue):
    bm = margin(base["logits"])
    solo = {h: effect_stats(rows, bm, margin(effects["solos"][i])) for i, h in enumerate(COALITION)}
    joint = effect_stats(rows, bm, margin(effects["joint"]))
    signed_solos = np.array([s["effects"] for s in solo.values()])
    gain = float(np.median(np.array(joint["effects"]) - np.max(signed_solos, axis=0)))
    self_error = float(np.max(np.abs(effects["self"] - base["logits"])))
    checks = {
        "direction": joint["direction_rate"] >= LIMITS["coalition_direction"],
        "median": joint["median"] >= LIMITS["coalition_median"],
        "fraction": joint["fraction"] >= LIMITS["coalition_fraction"],
        "gain_over_best_solo": gain >= LIMITS["coalition_gain_over_best_solo"],
        "significance": joint["group_sign_p"] <= LIMITS["p"],
        "self_exact": self_error == LIMITS["identity_max"],
    }
    transfer_checks = {
        key: {
            "direction": stat["direction_rate"] >= LIMITS["transfer_direction"],
            "median": stat["median"] >= LIMITS["transfer_median"],
            "significance": stat["group_sign_p"] <= LIMITS["p"],
        }
        for key, stat in (("19:5", solo["19:5"]), ("coalition", joint))
    }
    return {
        "task": task_stats(rows, bm),
        "controls": control_stats(base, controls),
        "single_effects": solo,
        "coalition": {
            "effects": joint,
            "gain_over_best_solo": gain,
            "checks": checks,
            "passed": all(checks.values()),
            "self_error": self_error,
            "same_answer": effect_stats(rows, bm, margin(effects["same"])),
        },
        "rescue": rescue_stats_v2(rows, bm, margin(rescue["logits"])),
        "late_only": rescue_stats_v2(rows, bm, margin(rescue["late_only"])),
        "incremental_over_late_only": rescue_stats_v2(
            rows, margin(rescue["late_only"]), margin(rescue["logits"])
        ),
        "transfer": {
            "checks": transfer_checks,
            "passed": all(all(v.values()) for v in transfer_checks.values()),
        },
    }


def known_controls(payload):
    """Planted rescue, nuisance, reversed and damaging interventions, no model calls."""
    from src.poc.templates_v3 import parse_v3_lookup

    blocks = {}
    for split in ("discovery", "primary", "vocabulary"):
        rows = [r for r in payload["rows"] if r["split"] == split]
        y = np.array([1 if parse_v3_lookup(r["text"]) in {"red", "green"} else -1 for r in rows])
        assert np.array_equal(y, [r["label"] for r in rows])
        # One group has four mistakes (6.25% discovery, 3.125% primary).
        first = min(r["group"] for r in rows)
        prior = np.array([-5.0 if r["group"] == first else 0 for r in rows])
        bm = 4 * y + prior
        positive = rescue_stats_v2(rows, bm, bm + 1.8 * y)
        nuisance = rescue_stats_v2(rows, bm, bm.copy())
        reverse = rescue_stats_v2(rows, bm, bm - 1.8 * y)
        damaged = bm + 1.8 * y
        good_i = next(i for i, value in enumerate(y * bm) if value > 0)
        damaged[good_i] = -y[good_i]
        damaging = rescue_stats_v2(rows, bm, damaged)
        signal = effect_stats(rows, bm, bm - 3.6 * y)
        checks = {
            "baseline": float(np.mean(y * bm > 0)) >= LIMITS["model_accuracy"],
            "positive_rescue": positive["passed"],
            "nuisance_rejected": not nuisance["passed"],
            "reverse_rejected": not reverse["passed"],
            "regression_rejected": not damaging["passed"],
            "identity_exact": np.array_equal(bm, bm.copy()),
            "coalition": signal["fraction"] >= LIMITS["coalition_fraction"]
            and signal["direction_rate"] >= LIMITS["coalition_direction"],
        }
        blocks[split] = {
            "checks": {k: bool(v) for k, v in checks.items()},
            "positive": positive,
            "nuisance": nuisance,
            "reverse": reverse,
            "damaging": damaging,
        }
    return {"passed": all(all(b["checks"].values()) for b in blocks.values()), "blocks": blocks}
