"""Executable pointer, identity-copy, K-stamping, nuisance and reversed mechanisms."""

import numpy as np

from src.poc.v4_controls import toy_rows
from src.poc.v7_analysis import PATHS, evaluate, margin

KINDS = (
    "planted_pointer_model",
    "literal_query_only_control",
    "direct_k_color_only_control",
    "nuisance",
    "reversed",
)


def run_toy(kind="planted_pointer_model", groups=8, vocabulary=("red", "blue")):
    rows = toy_rows(groups)
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    query = np.array([r["q_key"] for r in rows])
    order = np.array([r["k_order"] for r in rows])
    slot = query ^ order
    # Upstream and downstream representations are explicit. A pointer encodes the
    # resolved slot, not the surface key. Identity-only requires a separate clean
    # context-dependent resolver downstream of the copied literal query identity.
    u = np.eye(2)[
        query if kind in ("literal_query_only_control", "direct_k_color_only_control") else slot
    ]
    if kind == "nuisance":
        u[:] = np.eye(2)[0]
    q = np.eye(2)[query] if kind == "direct_k_color_only_control" else np.eye(2)[slot]
    k = np.repeat(np.eye(2)[None], len(rows), axis=0)
    if kind == "direct_k_color_only_control":
        k = np.array([np.eye(2)[[o, 1 - o]] for o in order])
    if kind == "nuisance":
        q = u.copy()
    v = np.array([[1.0, -1.0] if r["c_order"] == 0 else [-1.0, 1.0] for r in rows])

    def pattern(qq, kk):
        scores = np.sum(qq[:, None, :] * kk, axis=-1) * 12
        pp = np.exp(scores - scores.max(1, keepdims=True))
        return pp / pp.sum(1, keepdims=True)

    def output(qq, kk, vv):
        pp = pattern(qq, kk)
        val = np.sum(pp * vv, axis=1) * 3 * (-1 if kind == "reversed" else 1)
        return dict(
            logits=np.stack([val, -val], axis=1),
            p=np.repeat(pp[:, None, :], 3, axis=1),
            routed_p=np.repeat(pp[:, None, :], 3, axis=1),
        )

    base = output(q, k, v)
    paths, upstream, replay_q = {}, {}, {}
    for axis, modes in PATHS.items():
        ix = [lookup[r["donors"][axis]] for r in rows]
        for mode in modes:
            qq = q[ix] if mode in ("q_final", "qk") else q
            kk = k[ix] if mode in ("k_color", "qk") else k
            vv = v[ix] if mode == "ov_color" else v
            paths[f"{axis}_{mode}"] = output(qq, kk, vv)
        if axis in ("B", "C"):
            uq = u[ix]
            if kind == "literal_query_only_control":
                uq = np.eye(2)[uq.argmax(1) ^ order]
            replay_q[axis] = uq
            upstream[axis] = output(uq, k, v)
    result = evaluate(rows, base, paths, upstream)
    result["accuracy"] = float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0))
    result["passed"] &= result["accuracy"] >= 0.80
    result["self_error"] = float(
        np.max(abs(output(q.copy(), k.copy(), v.copy())["logits"] - base["logits"]))
    )
    result["vocabulary"] = list(vocabulary)

    def serialize(x):
        return {f: a.tolist() for f, a in x.items()}

    result["evidence"] = dict(
        rows=rows,
        q=q.tolist(),
        k=k.tolist(),
        v=v.tolist(),
        u=u.tolist(),
        base=serialize(base),
        paths={n: serialize(a) for n, a in paths.items()},
        upstream={n: serialize(a) for n, a in upstream.items()},
        replay_q={n: a.tolist() for n, a in replay_q.items()},
    )
    return result


def planted():
    out = {kind: run_toy(kind) for kind in KINDS}
    out["positive_primary"] = run_toy(groups=16)
    out["positive_vocabulary"] = run_toy(vocabulary=("green", "yellow"))
    out["passed"] = bool(
        all(
            out[k]["passed"]
            for k in ("planted_pointer_model", "positive_primary", "positive_vocabulary")
        )
        and out["literal_query_only_control"]["gates"]["upstream_B"]
        and not out["literal_query_only_control"]["gates"]["upstream_C"]
        and out["literal_query_only_control"]["effects"]["C_upstream_Q"]["median"] == 0
        and not out["direct_k_color_only_control"]["gates"]["C"]
        and not out["direct_k_color_only_control"]["gates"]["upstream_C"]
        and all(not out[k]["passed"] for k in KINDS[1:])
        and all(r["self_error"] == 0 for r in out.values())
    )
    return out
