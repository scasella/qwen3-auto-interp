"""Prospective upstream gates; no fitting to held-out data."""

import math

import numpy as np

LIMITS = dict(
    axis_A_ov_median=2.0,
    axis_A_direction=0.90,
    axis_A_qk_low=-0.25,
    axis_A_qk_high=0.25,
    axis_B_single=1.5,
    axis_B_coalition=3.5,
    axis_B_direction=0.90,
    axis_B_ov_exact=0.0,
    upstream_mass=0.30,
    upstream_margin=1.0,
    upstream_direction=0.85,
    model_accuracy=0.90,
    reconstruction_max=0.05,
    full_replay_max=0.10,
    identity_max=0.0,
    replication_max=0.0,
    p=0.05 / 3,
)


def group_sign(values, rows):
    groups = sorted({r["group"] for r in rows})
    means = [np.mean([v for v, r in zip(values, rows) if r["group"] == g]) for g in groups]
    n, k = len(means), sum(v > 0 for v in means)
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n


def effects(rows, base, patched):
    e = np.array([r["label"] for r in rows]) * (np.asarray(base) - patched)
    return dict(
        median=float(np.median(e)),
        direction=float(np.mean(e > 0)),
        group_p=group_sign(e, rows),
        absolute_max=float(np.max(abs(e))),
    )


def routing(rows, clean, changed):
    gains, losses = [], []
    for i, r in enumerate(rows):
        c = r["color_positions"][r["query_record"]]
        d = r["color_positions"][1 - r["query_record"]]
        gains.append(changed[i, :, d] - clean[i, :, d])
        losses.append(clean[i, :, c] - changed[i, :, c])
    gains, losses = np.array(gains), np.array(losses)
    return [
        dict(
            donor_gain=float(np.median(gains[:, h])),
            clean_loss=float(np.median(losses[:, h])),
            direction=float(np.mean((gains[:, h] > 0) & (losses[:, h] > 0))),
            gain_group_p=group_sign(gains[:, h], rows),
            loss_group_p=group_sign(losses[:, h], rows),
        )
        for h in range(clean.shape[1])
    ]


def upstream_gate(rows, base, margin, clean_p, mediated_p):
    e, shifts = effects(rows, base, margin), routing(rows, clean_p, mediated_p)
    output = e["median"] >= 1.0 and e["direction"] >= 0.85 and e["group_p"] <= LIMITS["p"]
    attention = all(
        x["donor_gain"] >= 0.30
        and x["clean_loss"] >= 0.30
        and x["direction"] >= 0.85
        and x["gain_group_p"] <= LIMITS["p"]
        and x["loss_group_p"] <= LIMITS["p"]
        for x in shifts
    )
    return dict(
        passed=bool(output and attention),
        margin=e,
        targets=shifts,
        checks=dict(output=bool(output), attention=bool(attention)),
    )


def axis_gate(rows, base, measures):
    details, checks = {}, {}
    for set_name in ("single", "coalition"):
        aov = effects(rows, base, measures[f"A_{set_name}_ov_color"])
        aqk = effects(rows, base, measures[f"A_{set_name}_qk"])
        bqk = effects(rows, base, measures[f"B_{set_name}_qk"])
        bov = effects(rows, base, measures[f"B_{set_name}_ov_context"])
        details[set_name] = dict(A_ov_color=aov, A_qk=aqk, B_qk=bqk, B_ov_context=bov)
        checks[set_name] = bool(
            aov["median"] >= 2.0
            and aov["direction"] >= 0.90
            and aov["group_p"] <= LIMITS["p"]
            and -0.25 <= aqk["median"] <= 0.25
            and bqk["median"] >= (1.5 if set_name == "single" else 3.5)
            and bqk["direction"] >= 0.90
            and bqk["group_p"] <= LIMITS["p"]
            and bov["absolute_max"] == 0.0
        )
    return dict(passed=all(checks.values()), checks=checks, statistics=details)


def rank_candidates(scores):
    eligible = [s for s in scores if s["minimum_transfer"] > 0 and s["effect"]["median"] > 0]
    return sorted(
        eligible,
        key=lambda s: (
            -s["minimum_transfer"],
            -s["effect"]["median"],
            tuple(map(int, s["head"].split(":"))),
        ),
    )[:3]
