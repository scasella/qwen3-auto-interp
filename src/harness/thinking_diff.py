"""Template-mode sensitivity pre-screen, not a proof of reasoning causality."""

import numpy as np

from src.harness.qwen_model import Probe, ProbeSpec


async def thinking_diff(worker, prompts, units, shard_size=128):
    if not prompts:
        raise ValueError("Thinking screen requires prompts")
    scores = []
    for start in range(0, len(units), shard_size):
        shard = tuple(units[start : start + shard_size])
        modes = {}
        for thinking in (True, False):
            spec = ProbeSpec(shard, thinking=thinking)
            modes[thinking] = await worker.map([Probe(t, spec) for t in prompts])
        for unit in shard:
            diffs = np.array(
                [
                    a.values[unit.key][0] - b.values[unit.key][0]
                    for a, b in zip(modes[True], modes[False])
                ]
            )
            scores.append(
                {
                    "unit": unit.key,
                    "mean_difference": float(diffs.mean()),
                    "mean_absolute_difference": float(np.abs(diffs).mean()),
                    "standard_error": float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
                    if len(diffs) > 1
                    else None,
                }
            )
    return sorted(scores, key=lambda s: s["mean_absolute_difference"], reverse=True)
