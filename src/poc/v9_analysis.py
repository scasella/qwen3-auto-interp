"""Prospective v9 statistics: float64 reductions, strict signs, group-level sign tests."""

import numpy as np

from src.poc.v4_analysis import sign_p

LIMITS = dict(
    accuracy=0.80,
    sign_p=0.01,
    A_effect=2.00,
    A_direction=0.75,
    A_other_abs=0.15,
    A_qk_signed_abs=0.25,
    A_specificity_ratio=10.0,
    B_effect=2.00,
    B_direction=0.95,
    B_context=0.0,
    C_effect=1.50,
    C_direction=0.95,
    C_shift=0.06,
    C_shift_direction=0.90,
    C_dominance_ratio=2.0,
    C_dominance_direction=0.75,
    C_ov_signed_max=0.25,
    C_ov_signed_min=-0.25,
    UB_effect=1.25,
    UB_direction=0.95,
    UB_shift=0.06,
    UB_first_shift=0.06,
    UB_shift_direction=0.90,
    UC_effect=1.00,
    UC_direction=0.95,
    UC_shift=0.05,
    UC_first_shift=0.05,
    UC_shift_direction=0.90,
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

    positive("A_ov_color", 2.00, 0.75)
    criteria["A_other_abs"] = effects["A_ov_other"]["median_abs"] <= 0.15
    criteria["A_qk_signed"] = -0.25 <= effects["A_qk"]["median"] <= 0.25
    criteria["A_specificity_ratio"] = (
        effects["A_ov_color"]["median"] >= 10 * effects["A_ov_other"]["median_abs"]
    )
    positive("B_q_final", 2.00, 0.95)
    for name in ("B_k_context", "B_ov_context"):
        criteria[name + "_exact"] = bool(np.all(reductions[name] == 0))
    positive("C_q_final", 1.50, 0.95)
    dominance = float(np.mean(reductions["C_q_final"] > reductions["C_k_color"]))
    criteria["C_dominance_median"] = (
        effects["C_q_final"]["median"] >= 2.0 * effects["C_k_color"]["median"]
    )
    criteria["C_dominance_direction"] = dominance >= 0.75
    criteria["C_ov_signed"] = -0.25 <= effects["C_ov_color"]["median"] <= 0.25
    for name, shift, direction, first, joint in (
        ("C_q_final", 0.06, 0.90, None, False),
        ("B_upstream_Q", 0.06, 0.90, 0.06, False),
        ("C_upstream_Q", 0.05, 0.90, 0.05, False),
    ):
        stat = shifts[name]
        for pole in ("gain", "loss"):
            criteria[name + "_" + pole] = stat["coalition"][pole] >= shift
            criteria[name + "_" + pole + "_direction"] = (
                stat["coalition"][pole + "_direction"] >= direction
            )
            if first is not None:
                criteria[name + "_first_" + pole] = stat["heads"][0][pole] >= first
                criteria[name + "_first_" + pole + "_direction"] = (
                    stat["heads"][0][pole + "_direction"] >= direction
                )
        if joint:
            criteria[name + "_joint_direction"] = stat["coalition"]["joint_direction"] >= direction
    positive("B_upstream_Q", 1.25, 0.95)
    positive("C_upstream_Q", 1.00, 0.95)
    assert len(criteria) == 43
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


def track2(rows, base, direct, internal, hop0, identity):
    labels = np.array([r["label"] for r in rows])
    stats = {
        k: effect(rows, labels * (margin(base) - margin(a)))
        for k, a in {**internal, **hop0}.items()
    }
    checks = {}
    conservation = {}
    for axis in ("B", "C"):
        # Absolute paired differences, not absolute difference of medians.
        error = float(np.median(abs(margin(internal[axis + "_U_full"]) - margin(direct[axis]))))
        conservation[axis] = error
        checks[axis + "_conservation"] = error <= 0.05
    pathways = {}
    for name in ("C_U_q_final", "C_U_kv_context"):
        s = stats[name]
        pathways[name] = s["median"] >= 0.75 and s["direction"] >= 0.90 and s["sign_p"] <= 0.01
    checks["C_internal_pathway"] = any(pathways.values())
    for axis, threshold in (("B", 0.50), ("C", 0.35)):
        s = stats[axis + "_hop0"]
        checks[axis + "_hop0_effect"] = s["median"] >= threshold
        checks[axis + "_hop0_direction"] = s["direction"] >= 0.85
        checks[axis + "_hop0_sign_p"] = s["sign_p"] <= 0.01
    checks["hop0_identity"] = identity == 0
    return dict(
        effects=stats,
        conservation=conservation,
        pathways=pathways,
        criteria={k: bool(v) for k, v in checks.items()},
    )


def select_hop0(screen):
    if len(screen) != 240 or {x["head"] for x in screen} != {
        f"{layer}:{head}" for layer in range(15) for head in range(16)
    }:
        raise ValueError("Screen must contain all 240 heads exactly once")

    def rank(x):
        b, c = x["B"]["median"], x["C"]["median"]
        if not np.isfinite([b, c]).all():
            raise ValueError("Nonfinite screen")
        layer, head = map(int, x["head"].split(":"))
        return (-min(b / 0.50, c / 0.35), -(b + c), layer, head)

    return [x["head"] for x in sorted(screen, key=rank)[:3]]
