"""Executable planted three-stage router / key-stamper / value-mover."""

import itertools

import numpy as np

from src.poc.v4_analysis import evaluate


def toy_rows(groups=8):
    rows = []
    for g, k, c, q in itertools.product(range(groups), range(2), range(2), range(2)):
        def uid(a, b, d):
            return f"{g}-{a}-{b}-{d}"
        rows.append(
            dict(
                id=uid(k, c, q),
                group=str(g),
                k_order=k,
                c_order=c,
                q_key=q,
                label=1 if q ^ k ^ c == 0 else -1,
                target_position=q ^ k,
                donors=dict(A=uid(k, 1 - c, q), B=uid(k, c, 1 - q), C=uid(1 - k, c, q)),
            )
        )
    return rows


def run_toy(kind="positive", vocabulary=("red", "blue")):
    rows = toy_rows()
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    # Stage 1 moves the question identity into Q_final; stage 2 stamps each
    # preceding record key onto its following value position's K vector.
    q = np.array([np.eye(2)[r["q_key"]] for r in rows])
    k = np.array([np.eye(2)[[r["k_order"], 1 - r["k_order"]]] for r in rows])
    v = np.array([[1.0, -1.0] if r["c_order"] == 0 else [-1.0, 1.0] for r in rows])
    if kind == "nuisance":
        q[:] = np.eye(2)[0]

    def pattern(q, k):
        a = np.einsum("bd,bsd->bs", q, k) * 12
        a = np.exp(a - a.max(1, keepdims=True))
        return a / a.sum(1, keepdims=True)

    def output(p, v):
        value = (p * v).sum(1) * 3
        if kind == "reversed":
            value = -value
        return dict(
            logits=np.stack([value, -value], axis=1),
            p=np.repeat(p[:, None, :], 3, axis=1),
            routed_p=np.repeat(p[:, None, :], 3, axis=1),
        )

    p = pattern(q, k)
    base = output(p, v)
    paths = {}
    for axis, modes in dict(
        A=["ov_color", "ov_other", "qk"],
        B=["q_final", "k_context", "ov_context"],
        C=["k_color", "ov_color"],
    ).items():
        ix = [lookup[r["donors"][axis]] for r in rows]
        for mode in modes:
            pp, vv = p, v
            if mode == "ov_color":
                vv = v[ix]
            elif mode == "q_final":
                pp = pattern(q[ix], k)
            elif mode == "k_color":
                pp = pattern(q, k[ix])
            elif mode == "qk":
                pp = pattern(q[ix], k[ix])
            paths[f"{axis}_{mode}"] = output(pp, vv)
    ix = [lookup[r["donors"]["B"]] for r in rows]
    moved_q = q[ix] if kind != "nuisance" else q
    mid = output(pattern(moved_q, k), v)
    result = evaluate(rows, base, paths, mid, mid)
    result["accuracy"] = float(
        np.mean(
            np.array([r["label"] for r in rows]) * (base["logits"][:, 0] - base["logits"][:, 1]) > 0
        )
    )
    result["passed"] = result["passed"] and result["accuracy"] >= 0.90
    result["vocabulary"] = list(vocabulary)
    result["self_error"] = float(np.max(abs(output(pattern(q, k), v)["logits"] - base["logits"])))
    result["evidence"] = dict(
        rows=rows,
        q=q.tolist(),
        k=k.tolist(),
        v=v.tolist(),
        base={f: a.tolist() for f, a in base.items()},
        paths={tag: {f: a.tolist() for f, a in arr.items()} for tag, arr in paths.items()},
        mediator={f: a.tolist() for f, a in mid.items()},
    )
    return result


def planted():
    out = {name: run_toy(name) for name in ("positive", "nuisance", "reversed")}
    out["vocabulary"] = run_toy(vocabulary=("green", "yellow"))
    out["passed"] = (
        out["positive"]["passed"]
        and out["vocabulary"]["passed"]
        and not out["nuisance"]["passed"]
        and not out["reversed"]["passed"]
        and all(v["self_error"] == 0 for v in out.values() if isinstance(v, dict))
    )
    return out
