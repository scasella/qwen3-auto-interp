"""Real activation regression for the registered FP32 delta tolerance."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_bounded_poc_v4_confirm import softmax
from src.poc.v10_stable_harness import contribution_delta, routing_pattern


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_saved_large_value_q_patch_meets_fixed_tolerance(device):
    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Opt-in Metal arithmetic replay; no target forward")
    root = Path(__file__).resolve().parents[1] / "data/bounded-poc-v10-qwen17b-restart1"
    rows = [r for r in json.loads((root / "dataset.json").read_text())["rows"] if r["split"] == "calibration"]
    ids = {r["id"]: i for i, r in enumerate(rows)}
    with np.load(root / "discovery-layer-22.npz") as z:
        base = dict(z)
    n = len(rows[0]["token_ids"])
    donor = [ids[r["donors"]["B"]] for r in rows]

    def t(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    q = t(base["q"][donor, 9])
    k = t(base["k"][:, 9, :n])
    pc = t(base["p"][:, 9, :n])
    vc = t(base["v"][:, 9, :n])
    valid = torch.ones((len(rows), n), dtype=torch.bool, device=device)
    pd = routing_pattern(q, k, valid, 128**-0.5)
    delta = contribution_delta("qk", pc, vc, pd, vc, torch.ones_like(pc))
    actual = delta.detach().cpu().numpy()
    expected = []
    for i, j in enumerate(donor):
        p = softmax(base["q"][j, [9]], base["k"][i, [9]], n)[0]
        expected.append(((p - base["p"][i, 9, :n])[:, None] * base["v"][i, 9, :n]).sum(0, dtype=np.float32))
    assert actual.dtype == np.float32
    error = float(np.max(np.abs(actual - np.asarray(expected))))
    assert error <= 3e-5, error
