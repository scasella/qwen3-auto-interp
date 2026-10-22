"""Planted two-stage selector -> three value movers, with nuisance/reversed failures."""

import numpy as np

from src.poc.upstream_analysis import axis_gate, upstream_gate


def planted():
    rows = [
        dict(
            group=f"g{g}",
            query_record=q,
            assignment=a,
            color_positions=[0, 1],
            label=1 if q == a else -1,
        )
        for g in range(16)
        for q in range(2)
        for a in range(2)
    ]
    # Stage 1: query/key comparison creates a positional one-hot code, no value dependence.
    q = np.array([np.eye(2)[r["query_record"]] for r in rows])
    p = np.repeat(q[:, None], 3, axis=1)
    # Stage 2: each head reads its selected value. A vocabulary swap changes the values,
    # not the selector. The output measurement uses that vocabulary's corresponding axis.
    v = np.array([[1.0, -1.0] if r["assignment"] == 0 else [-1.0, 1.0] for r in rows])
    y = np.array([r["label"] for r in rows])
    h = np.sum(p * v[:, None], axis=-1)
    base = 2 * h.sum(-1)
    a = np.arange(len(rows)) ^ 1
    b = np.arange(len(rows)) ^ 2
    measures = {}
    for set_name, active in (("single", [0]), ("coalition", [0, 1, 2])):
        for axis, donor in (("A", a), ("B", b)):
            qk = np.sum((p[donor] - p) * v[:, None], axis=-1)
            ov = np.sum(p * (v[donor] - v)[:, None], axis=-1)
            measures[f"{axis}_{set_name}_qk"] = base + 2 * qk[:, active].sum(-1)
            measures[
                f"{axis}_{set_name}_ov_color" if axis == "A" else f"{axis}_{set_name}_ov_context"
            ] = base + 2 * ov[:, active].sum(-1)
    # Upstream query code is replaced; only the induced pattern is replayed into stage 2.
    changed_p = p[b]
    mediated = 2 * np.sum(changed_p * v[:, None], axis=(1, 2))
    positive = upstream_gate(rows, base, mediated, p, changed_p)
    nuisance = upstream_gate(rows, base, base, p, p)
    reversed_output = upstream_gate(rows, base, base + 12 * y, p, changed_p)
    reverse_routing = upstream_gate(rows, base, mediated, p, 2 * p - changed_p)
    axes = axis_gate(rows, base, measures)
    # Re-express the same two-stage computation in disjoint two-dimensional value axes.
    vocabulary_values = v[..., None] * np.array([0.0, 1.0])
    vocabulary_margin = (
        2 * np.sum(changed_p[..., None] * vocabulary_values[:, None], axis=(1, 2))[:, 1]
    )
    transfer = upstream_gate(rows, base, vocabulary_margin, p, changed_p)
    return dict(
        passed=axes["passed"]
        and positive["passed"]
        and transfer["passed"]
        and not nuisance["passed"]
        and not reversed_output["passed"]
        and not reverse_routing["passed"],
        axis=axes,
        selector=positive,
        vocabulary=transfer,
        nuisance=nuisance,
        reversed_output=reversed_output,
        reversed_routing=reverse_routing,
    )
