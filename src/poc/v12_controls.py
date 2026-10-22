"""Executable three-stage spatial toy with nonlinear downstream value retrieval."""

import itertools

import numpy as np

from src.poc.v12_cross_scale_3hop import evaluate


def toy(kind="question_span"):
    rows = []
    for group in range(8):
        for k, c, q in itertools.product(range(2), repeat=3):
            rows.append(dict(group=str(group), k=k, c=c, q=q, label=(-1) ** (k ^ c ^ q)))
    # Two context slots, queried-key position 2, suffix position 3, T=4.
    values = np.zeros((64, 5), dtype=np.float64)
    for i, r in enumerate(rows):
        location = 0 if kind == "context_kv_only" else 4 if kind == "final_token_only" else 2
        values[i, location] = (-1) ** (r["k"] ^ r["q"])
    readout = np.zeros(5)
    readout[0 if kind == "context_kv_only" else 4 if kind == "final_token_only" else 2] = 1

    def logits(u):
        # U -> D query; original D K=[+1,-1], V=[slot1color,slot2color].
        result = []
        for z, r in zip(u, rows):
            scores = np.array([z, -z]) * 2
            p = np.exp(scores - scores.max())
            p /= p.sum()
            color = np.array([(-1) ** r["c"], -((-1) ** r["c"])])
            result.append([float(2 * p @ color), float(-2 * p @ color)])
        return dict(logits=np.asarray(result))

    base = logits(values @ readout)
    paths = {}
    for axis in "BC":
        donor = np.array([i ^ (1 if axis == "B" else 4) for i in range(64)])
        vd = values[donor]
        if kind == "nuisance":
            vd = values.copy()
        elif kind == "reversed":
            vd = 3 * values
        for mode in ("full", "v_qspan", "kv_context", "q_final", "upstream_Q", "hop0_qspan"):
            mixed = values.copy()
            if mode in ("full", "upstream_Q"):
                mixed = vd.copy()
            elif mode == "v_qspan":
                mixed[:, 2:] = vd[:, 2:]
            elif mode == "kv_context":
                mixed[:, :2] = vd[:, :2]
                # B cannot alter prequery positions: toy context values are query-
                # dependent only in this deliberately incorrect negative mechanism.
                if axis == "B":
                    mixed = values.copy()
            elif mode == "hop0_qspan":
                mixed[:, 2:4] = vd[:, 2:4]
            name = axis + ("_" + mode if mode in ("upstream_Q", "hop0_qspan") else "_U_" + mode)
            paths[name] = logits(mixed @ readout)
    result = evaluate(rows, base, paths)
    return dict(
        kind=kind,
        rows=rows,
        values=values.tolist(),
        readout=readout.tolist(),
        baseline=base["logits"].tolist(),
        paths={k: v["logits"].tolist() for k, v in paths.items()},
        result=result,
    )


def planted():
    cases = {
        k: toy(k)
        for k in ("question_span", "final_token_only", "context_kv_only", "nuisance", "reversed")
    }
    passed = (
        cases["question_span"]["result"]["passed"]
        and not cases["final_token_only"]["result"]["gate2"]
        and not cases["context_kv_only"]["result"]["gate1"]
        and not cases["nuisance"]["result"]["passed"]
        and not cases["reversed"]["result"]["passed"]
    )
    return dict(cases=cases, passed=passed)
