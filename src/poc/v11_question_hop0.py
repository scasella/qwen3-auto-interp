"""Prospective spatial relay gates, independent groups, and bounded greedy selection."""

import math

import numpy as np

U = ((17, 1), (16, 4), (15, 4))
D = ((19, 5), (21, 13), (22, 9))
CANDIDATES = tuple((layer, head) for layer in range(15) for head in range(16))
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SEED = 20261027
MODES = ("full", "v_qspan", "kv_context", "q_final")
LIMITS = dict(
    accuracy=0.8,
    conservation=0.05,
    B_v=1.2,
    C_v=0.9,
    relay_direction=0.9,
    ratio=3.0,
    dominance=0.8,
    B_hop=0.4,
    C_hop=0.25,
    hop_direction=0.8,
    sign_p=0.01,
    identity=0.0,
    delta=0.0,
)


def margin(a):
    x = np.asarray(a["logits"], dtype=np.float64)
    return x[:, 0] - x[:, 1]


def effect(rows, values):
    v = np.asarray(values, dtype=np.float64)
    assert len(v) == len(rows) and np.isfinite(v).all()
    groups = sorted({r["group"] for r in rows})
    sums = [sum(v[i] for i, r in enumerate(rows) if r["group"] == g) for g in groups]
    k = sum(s > 0 for s in sums)
    p = sum(math.comb(len(groups), j) for j in range(k, len(groups) + 1)) / 2 ** len(groups)
    return dict(median=float(np.median(v)), direction=float(np.mean(v > 0)), sign_p=p)


def select(scores):
    if not scores or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("ABSTAIN: empty or nonfinite candidate scores")
    return min(scores, key=lambda h: (-scores[h], h))


def evaluate(rows, base, paths):
    labels = np.array([r["label"] for r in rows])
    reduction = {k: labels * (margin(base) - margin(a)) for k, a in paths.items()}
    stats = {k: effect(rows, v) for k, v in reduction.items()}
    criteria = {}
    diagnostics = {}
    for axis, threshold, hop_threshold in [("B", 1.2, 0.4), ("C", 0.9, 0.25)]:
        full, direct = paths[f"{axis}_U_full"], paths[f"{axis}_upstream_Q"]
        difference = float(np.median(abs(margin(full) - margin(direct))))
        diagnostics[axis + "_conservation"] = difference
        criteria[axis + "_conservation"] = difference <= 0.05
        name = axis + "_U_v_qspan"
        for k, yes in [
            ("effect", stats[name]["median"] >= threshold),
            ("direction", stats[name]["direction"] >= 0.9),
            ("sign_p", stats[name]["sign_p"] <= 0.01),
        ]:
            criteria[name + "_" + k] = yes
        for competitor in ["q_final"] if axis == "B" else ["q_final", "kv_context"]:
            other = axis + "_U_" + competitor
            criteria[name + "_over_" + competitor + "_median"] = (
                stats[name]["median"] >= 3 * stats[other]["median"]
            )
            fraction = float(np.mean(reduction[name] > reduction[other]))
            diagnostics[name + "_over_" + competitor] = fraction
            criteria[name + "_over_" + competitor + "_direction"] = fraction >= 0.8
        hop = axis + "_hop0_qspan"
        criteria[hop + "_effect"] = stats[hop]["median"] >= hop_threshold
        criteria[hop + "_direction"] = stats[hop]["direction"] >= 0.8
        criteria[hop + "_sign_p"] = stats[hop]["sign_p"] <= 0.01
    error = float(
        np.max(
            abs(np.asarray(paths["B_U_kv_context"]["logits"], dtype=np.float64) - base["logits"])
        )
    )
    diagnostics["B_context_max_logit_error"] = error
    criteria["B_context_exact"] = error == 0
    accuracy = float(np.mean(labels * margin(base) > 0))
    g1 = all(v for k, v in criteria.items() if "hop0" not in k)
    g2 = all(v for k, v in criteria.items() if "hop0" in k)
    return dict(
        accuracy=accuracy,
        correct=int(np.sum(labels * margin(base) > 0)),
        n=len(rows),
        effects=stats,
        diagnostics=diagnostics,
        criteria={k: bool(v) for k, v in criteria.items()},
        gate0=accuracy >= 0.8,
        gate1=g1,
        gate2=g2,
        passed=accuracy >= 0.8 and g1 and g2,
    )


def pattern(q, k):
    scores = np.sum(k * q, axis=-1, dtype=np.float32) * np.float32(q.shape[-1] ** -0.5)
    w = np.exp(scores - np.max(scores))
    return w / np.sum(w, dtype=np.float32)


def internal_delta(clean, donor, query, mode):
    """FP32 anchored contrast; sequence arrays must be unpadded."""
    q, k, v = (clean[x].copy() for x in ("q", "k", "v"))
    if mode in ("full", "q_final"):
        q = donor["q"].copy()
    if mode == "full":
        k, v = donor["k"].copy(), donor["v"].copy()
    elif mode == "kv_context":
        k[:query], v[:query] = donor["k"][:query], donor["v"][:query]
    elif mode == "v_qspan":
        v[query:] = donor["v"][query:]
    elif mode != "q_final":
        raise ValueError(mode)
    p = pattern(q, k)
    if mode == "v_qspan":
        terms = clean["p"][:, None] * (v - clean["v"])
    elif mode == "q_final":
        terms = (p - clean["p"])[:, None] * clean["v"]
    else:
        terms = p[:, None] * v - clean["p"][:, None] * clean["v"]
    delta = np.sum(terms, axis=0, dtype=np.float32)
    return delta
