"""Analytic source-selection and upstream-relay positive/negative controls."""

import torch

from src.followup.harness import intervene


def known_controls():
    q = torch.tensor([[1.0, 0.0]])
    source_k = torch.tensor([[[8.0, 0.0], [-8.0, 0.0], [0.0, 0.0]]])
    relay_k = torch.tensor([[[-8.0, 0.0], [-8.0, 0.0], [8.0, 0.0]]])
    v = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [2.0, 0.0]]])
    p = dict(
        value_positions=[0, 1],
        query_record=0,
        suffix_start=2,
        q=[-1.0, 0.0],
        v=[[-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
    )

    def measure(q, k, v):
        return float((((q[:, None] * k).sum(-1)).softmax(-1)[:, :, None] * v).sum(1)[0, 0])

    result = {}
    for name, k in (("source", source_k), ("relay", relay_k)):
        result[name] = {"base": measure(q, k, v)}
        for mode in ("source_swap", "q_query", "v_suffix"):
            result[name][mode] = measure(*intervene(q, k, v, mode, [p]))
    a, b = result["source"], result["relay"]
    result["passed"] = (
        a["base"] > 1.9
        and a["source_swap"] < -1.9
        and a["q_query"] < -1.9
        and a["v_suffix"] > 1.9
        and b["base"] > 1.9
        and b["source_swap"] > 1.9
        and b["v_suffix"] < -1.9
    )
    return result
