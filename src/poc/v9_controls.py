"""Explicit three-stage attention control and mechanism alternatives."""

import numpy as np

from src.poc.v7_controls import KINDS, run_toy
from src.poc.v9_analysis import evaluate, margin, track2


def three_stage(kind="positive", groups=8):
    e = run_toy(groups=groups)["evidence"]
    rows = e["rows"]
    labels = np.array([r["label"] for r in rows])
    slots = np.array([r["q_key"] ^ r["k_order"] for r in rows])
    h0 = np.eye(2)[slots]
    if kind == "literal":
        h0 = np.eye(2)[[r["q_key"] for r in rows]]
    if kind == "nuisance":
        h0[:] = 0.5
    value = np.array([[1.0, -1.0] if r["c_order"] == 0 else [-1.0, 1.0] for r in rows])

    def attention(q):
        score = 12 * q
        p = np.exp(score - score.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def stages(x, order):
        # Literal identity must be resolved against the recipient context.
        if kind == "literal":
            x = np.eye(2)[x.argmax(1) ^ order]
        u = attention(x) @ np.eye(2)
        d = attention(u) @ np.eye(2)
        val = (attention(d) * value).sum(1) * 3
        if kind == "reversed":
            val = -val
        return u, d, np.stack((val, -val), 1)

    order = np.array([r["k_order"] for r in rows])
    u, d, logits = stages(h0, order)
    base = dict(logits=logits)
    ix = {r["id"]: i for i, r in enumerate(rows)}
    hops, intermediates = {}, {}
    for axis in ("B", "C"):
        donor = [ix[r["donors"][axis]] for r in rows]
        uu, dd, ll = stages(h0[donor], order)
        hops[axis + "_hop0"] = dict(logits=ll)
        intermediates[axis] = dict(u=uu.tolist(), d=dd.tolist(), logits=ll.tolist())
    direct = {a: hops[a + "_hop0"] for a in ("B", "C")}
    internal = {
        a + "_U_" + m: direct[a]
        for a in ("B", "C")
        for m in ("full", "q_final", "kv_context", "k_all", "v_all")
    }
    result = track2(
        rows, base, direct, internal, hops, float(np.max(abs(stages(h0.copy(), order)[2] - logits)))
    )
    result["evidence"] = dict(
        rows=rows,
        kind=kind,
        h0=h0.tolist(),
        u=u.tolist(),
        d=d.tolist(),
        value=value.tolist(),
        logits=logits.tolist(),
        stages=intermediates,
    )
    result["accuracy"] = float(np.mean(labels * margin(base) > 0))
    result["passed"] = all(result["criteria"].values())
    return result


def planted():
    out = {}
    for kind in KINDS:
        e = run_toy(kind)["evidence"]

        def convert(a):
            return {k: np.asarray(v) for k, v in a.items()}

        out[kind] = evaluate(
            e["rows"],
            convert(e["base"]),
            {k: convert(v) for k, v in e["paths"].items()},
            {k: convert(v) for k, v in e["upstream"].items()},
        )
        out[kind]["evidence"] = e
    for kind in ("positive", "literal", "nuisance", "reversed"):
        out["three_" + kind] = three_stage(kind)
    out["passed"] = (
        out["planted_pointer_model"]["passed"]
        and all(not out[k]["passed"] for k in KINDS[1:])
        and out["three_positive"]["passed"]
        and all(not out["three_" + k]["passed"] for k in ("literal", "nuisance", "reversed"))
    )
    return out
