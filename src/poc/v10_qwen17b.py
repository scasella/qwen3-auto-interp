"""Prospective v10 cross-scale statistics: float64 reductions, strict signs, group-level sign tests."""

import numpy as np

from src.poc.v4_analysis import sign_p

LIMITS = dict(
    accuracy=0.85,
    sign_p=0.01,
    A_effect=1.50,
    A_gap_fraction=0.25,
    A_direction=0.80,
    A_other_abs=0.20,
    A_qk_signed_abs=0.35,
    A_specificity_ratio=8.0,
    B_effect=1.50,
    B_direction=0.90,
    B_context=0.0,
    C_effect=1.00,
    C_direction=0.90,
    C_shift=0.05,
    C_shift_direction=0.85,
    C_dominance_ratio=1.75,
    C_dominance_direction=0.70,
    C_ov_signed_abs=0.35,
    UB_effect=1.00,
    UB_direction=0.90,
    UB_shift=0.05,
    UB_shift_direction=0.85,
    UC_effect=0.75,
    UC_direction=0.90,
    UC_shift=0.04,
    UC_shift_direction=0.85,
    identity=0.0,
    numpy_delta=0.00003,
)
DOWNSTREAM_CANDIDATES = tuple((layer, head) for layer in range(15, 26) for head in range(16))
SEED = 20261026
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def upstream_candidates(downstream):
    return tuple(
        (layer, head)
        for layer in range(10, min(layer for layer, _ in downstream))
        for head in range(16)
    )


def select_extension(scores):
    """Maximum sum of two coalition medians; ties use numeric layer/head order."""
    if not scores:
        raise ValueError("ABSTAIN: insufficient eligible candidates")
    if not all(np.isfinite(score) for score in scores.values()):
        raise ValueError("Nonfinite discovery score")
    return min(scores, key=lambda head: (-scores[head], head))


PATHS = {
    "A": ["ov_color", "ov_other", "qk"],
    "B": ["q_final", "k_context", "ov_context"],
    "C": ["q_final", "k_color", "ov_color"],
}


def margin(a):
    return a["logits"][:, 0].astype(np.float64) - a["logits"][:, 1].astype(np.float64)


def effect(rows, values):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        median=float(np.median(values)),
        direction=float(np.mean(values > 0)),
        median_abs=float(np.median(abs(values))),
        sign_p=sign_p(rows, values),
    )


def attention(rows, clean, patched, axis):
    clean, patched = np.asarray(clean, dtype=np.float64), np.asarray(patched, dtype=np.float64)
    index = {r["id"]: r for r in rows}
    gains, losses = [], []
    for i, row in enumerate(rows):
        c, d = row["target_position"], index[row["donors"][axis]]["target_position"]
        gains.append(patched[i, :, d] - clean[i, :, d])
        losses.append(clean[i, :, c] - patched[i, :, c])
    gains, losses = np.asarray(gains), np.asarray(losses)

    def summarize(g, loss):
        return dict(
            gain=float(np.median(g)),
            loss=float(np.median(loss)),
            gain_direction=float(np.mean(g > 0)),
            loss_direction=float(np.mean(loss > 0)),
            joint_direction=float(np.mean((g > 0) & (loss > 0))),
        )

    return dict(
        coalition=summarize(gains.mean(1), losses.mean(1)),
        heads=[summarize(gains[:, h], losses[:, h]) for h in range(3)],
    )


def evaluate(rows, base, paths, upstream):
    labels = np.array([r["label"] for r in rows])
    reductions = {name: labels * (margin(base) - margin(a)) for name, a in paths.items()}
    reductions.update(
        {f"{axis}_upstream_Q": labels * (margin(base) - margin(a)) for axis, a in upstream.items()}
    )
    effects = {name: effect(rows, x) for name, x in reductions.items()}
    shifts = {"C_q_final": attention(rows, base["p"], paths["C_q_final"]["routed_p"], "C")}
    shifts.update(
        {
            f"{axis}_upstream_Q": attention(rows, base["p"], a["routed_p"], axis)
            for axis, a in upstream.items()
        }
    )
    criteria = {}

    def positive(name, threshold, direction):
        stat = effects[name]
        criteria[name + "_effect"] = stat["median"] >= threshold
        criteria[name + "_direction"] = stat["direction"] >= direction
        criteria[name + "_sign_p"] = stat["sign_p"] <= 0.01

    positive("A_ov_color", 1.50, 0.80)
    criteria["A_other_abs"] = effects["A_ov_other"]["median_abs"] <= 0.20
    criteria["A_qk_signed"] = -0.35 <= effects["A_qk"]["median"] <= 0.35
    criteria["A_specificity_ratio"] = (
        effects["A_ov_color"]["median"] >= 8 * effects["A_ov_other"]["median_abs"]
    )
    positive("B_q_final", 1.50, 0.90)
    for name in ("B_k_context", "B_ov_context"):
        criteria[name + "_exact"] = bool(np.array_equal(paths[name]["logits"], base["logits"]))
    positive("C_q_final", 1.00, 0.90)
    dominance = float(np.mean(reductions["C_q_final"] > reductions["C_k_color"]))
    criteria["C_dominance_median"] = (
        effects["C_q_final"]["median"] >= 1.75 * effects["C_k_color"]["median"]
    )
    criteria["C_dominance_direction"] = dominance >= 0.70
    criteria["C_ov_signed"] = -0.35 <= effects["C_ov_color"]["median"] <= 0.35
    for name, shift, direction in (
        ("C_q_final", 0.05, 0.85),
        ("B_upstream_Q", 0.05, 0.85),
        ("C_upstream_Q", 0.04, 0.85),
    ):
        stat = shifts[name]
        for pole in ("gain", "loss"):
            criteria[name + "_" + pole] = stat["coalition"][pole] >= shift
            criteria[name + "_" + pole + "_direction"] = (
                stat["coalition"][pole + "_direction"] >= direction
            )
    positive("B_upstream_Q", 1.00, 0.90)
    positive("C_upstream_Q", 0.75, 0.90)
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    donors = [lookup[r["donors"]["A"]] for r in rows]
    clean_gap = float(np.median(labels * (margin(base) - margin(base)[donors])))
    criteria["A_gap_fraction"] = effects["A_ov_color"]["median"] >= 0.25 * clean_gap
    assert len(criteria) == 36
    gates = dict(
        A=all(v for k, v in criteria.items() if k.startswith("A_")),
        B=all(
            v for k, v in criteria.items() if k.startswith("B_") and not k.startswith("B_upstream")
        ),
        C=all(
            v for k, v in criteria.items() if k.startswith("C_") and not k.startswith("C_upstream")
        ),
        upstream_B=all(v for k, v in criteria.items() if k.startswith("B_upstream")),
        upstream_C=all(v for k, v in criteria.items() if k.startswith("C_upstream")),
    )
    return dict(
        clean_counterfactual_gap=clean_gap,
        effects=effects,
        attention=shifts,
        C_dominance_direction=dominance,
        criteria={k: bool(v) for k, v in criteria.items()},
        gates=gates,
        passed=all(gates.values()),
    )
