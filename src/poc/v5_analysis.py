"""Frozen v5 statistics: float64 reductions, strict signs, group-level sign tests."""

import numpy as np

from src.poc.v4_analysis import sign_p

LIMITS = dict(
    accuracy=0.90,
    sign_p=0.01,
    A_effect=3.0,
    A_direction=0.85,
    A_other_abs=0.15,
    A_qk_signed_abs=0.25,
    A_specificity_ratio=10.0,
    B_effect=3.25,
    B_direction=0.95,
    B_context=0.0,
    C_effect=2.25,
    C_direction=0.95,
    C_shift=0.10,
    C_shift_direction=0.90,
    C_dominance_ratio=3.0,
    C_dominance_direction=0.85,
    C_ov_signed_max=0.25,
    UB_effect=2.50,
    UB_direction=0.95,
    UB_shift=0.12,
    UB_first_shift=0.20,
    UB_joint_direction=0.95,
    UC_effect=2.00,
    UC_direction=0.95,
    UC_shift=0.10,
    UC_first_shift=0.15,
    UC_shift_direction=0.95,
    identity=0.0,
    numpy_delta=0.00003,
)
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

    positive("A_ov_color", 3.0, 0.85)
    criteria["A_other_abs"] = effects["A_ov_other"]["median_abs"] <= 0.15
    criteria["A_qk_signed"] = -0.25 <= effects["A_qk"]["median"] <= 0.25
    criteria["A_specificity_ratio"] = (
        effects["A_ov_color"]["median"] >= 10 * effects["A_ov_other"]["median_abs"]
    )
    positive("B_q_final", 3.25, 0.95)
    for name in ("B_k_context", "B_ov_context"):
        criteria[name + "_exact"] = bool(np.all(reductions[name] == 0))
    positive("C_q_final", 2.25, 0.95)
    dominance = float(np.mean(reductions["C_q_final"] > reductions["C_k_color"]))
    criteria["C_dominance_median"] = (
        effects["C_q_final"]["median"] >= 3 * effects["C_k_color"]["median"]
    )
    criteria["C_dominance_direction"] = dominance >= 0.85
    criteria["C_ov_signed"] = effects["C_ov_color"]["median"] <= 0.25
    for name, shift, direction, first, joint in (
        ("C_q_final", 0.10, 0.90, None, False),
        ("B_upstream_Q", 0.12, 0.95, 0.20, True),
        ("C_upstream_Q", 0.10, 0.95, 0.15, False),
    ):
        stat = shifts[name]
        for pole in ("gain", "loss"):
            criteria[name + "_" + pole] = stat["coalition"][pole] >= shift
            criteria[name + "_" + pole + "_direction"] = (
                stat["coalition"][pole + "_direction"] >= direction
            )
            if first is not None:
                criteria[name + "_first_" + pole] = stat["heads"][0][pole] >= first
        if joint:
            criteria[name + "_joint_direction"] = stat["coalition"]["joint_direction"] >= direction
    positive("B_upstream_Q", 2.5, 0.95)
    positive("C_upstream_Q", 2.0, 0.95)
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
        effects=effects,
        attention=shifts,
        C_dominance_direction=dominance,
        criteria={k: bool(v) for k, v in criteria.items()},
        gates=gates,
        passed=all(gates.values()),
    )
