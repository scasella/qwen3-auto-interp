"""Portable locations for the source-only derivative; no implicit downloads."""

import os
from pathlib import Path


def model17_path():
    root = Path(__file__).resolve().parents[2]
    default = (
        root
        / "data/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    return Path(os.environ.get("QWEN17_CHECKPOINT", str(default)))
