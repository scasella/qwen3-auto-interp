"""Frozen confirmatory statistics. Groups, not prompts, are sign-test units."""

import math

import numpy as np

LIMITS = dict(
    accuracy=0.90,
    direction=0.95,
    sign_p=0.01,
    A_effect=4.0,
    A_specificity=0.25,
    B_effect=4.0,
    C_effect=3.5,
    C_shift=0.35,
    C_ov_max=0.50,
    upstream_effect=4.0,
    upstream_mean_shift=0.30,
    upstream_first_shift=0.45,
    identity=0.0,
)


def sign_p(rows, values):
    groups = sorted({r["group"] for r in rows})
    positives = sum(
        np.mean([x for r, x in zip(rows, values) if r["group"] == g]) > 0 for g in groups
    )
    return sum(math.comb(len(groups), k) for k in range(positives, len(groups) + 1)) / 2 ** len(
        groups
    )


def effect(rows, clean, patched):
    values = np.array([r["label"] for r in rows]) * (clean - patched)
    return dict(
        median=float(np.median(values)),
        direction=float(np.mean(values > 0)),
        sign_p=sign_p(rows, values),
        median_abs=float(np.median(abs(values))),
    )


def positive(stat, threshold):
    return stat["median"] >= threshold and stat["direction"] >= 0.95 and stat["sign_p"] <= 0.01


def shifts(rows, clean, patched, axis):
    by_id = {r["id"]: r for r in rows}
    gain, loss = [], []
    for i, r in enumerate(rows):
        donor = by_id[r["donors"][axis]]
        p, d = r["target_position"], donor["target_position"]
        gain.append(patched[i, :, d] - clean[i, :, d])
        loss.append(clean[i, :, p] - patched[i, :, p])
    gain, loss = np.array(gain), np.array(loss)

    def stats(g, losses):
        return dict(
            gain=float(np.median(g)),
            loss=float(np.median(losses)),
            gain_direction=float(np.mean(g > 0)),
            loss_direction=float(np.mean(losses > 0)),
            joint_direction=float(np.mean((g > 0) & (losses > 0))),
        )

    return dict(
        coalition=stats(gain.mean(1), loss.mean(1)),
        heads=[stats(gain[:, h], loss[:, h]) for h in range(3)],
    )


def shift_pass(s, minimum):
    return s["gain"] >= minimum and s["loss"] >= minimum and s["joint_direction"] >= 0.95


def evaluate(rows, base, paths, mediator, upstream):
    def margin(x):
        return x["logits"][:, 0].astype(float) - x["logits"][:, 1].astype(float)

    clean = margin(base)
    stats = {name: effect(rows, clean, margin(a)) for name, a in paths.items()}
    c_shift = shifts(rows, base["p"], paths["C_k_color"]["routed_p"], "C")
    u_shift = shifts(rows, base["p"], mediator["p"], "B")
    u_effect = effect(rows, clean, margin(upstream))
    gates = {
        "A": positive(stats["A_ov_color"], 4.0)
        and stats["A_ov_other"]["median_abs"] <= 0.25
        and stats["A_qk"]["median_abs"] <= 0.25,
        "B": positive(stats["B_q_final"], 4.0)
        and np.array_equal(paths["B_k_context"]["logits"], base["logits"])
        and np.array_equal(paths["B_ov_context"]["logits"], base["logits"]),
        "C": positive(stats["C_k_color"], 3.5)
        and shift_pass(c_shift["coalition"], 0.35)
        and stats["C_ov_color"]["median"] <= 0.50,
        "upstream": positive(u_effect, 4.0)
        and shift_pass(u_shift["coalition"], 0.30)
        and shift_pass(u_shift["heads"][0], 0.45)
        and all(s["joint_direction"] >= 0.95 for s in u_shift["heads"]),
    }
    return dict(
        effects=stats,
        C_attention=c_shift,
        upstream_effect=u_effect,
        upstream_attention=u_shift,
        gates={k: bool(v) for k, v in gates.items()},
        passed=bool(all(gates.values())),
    )
