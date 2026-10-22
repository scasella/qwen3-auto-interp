# Qwen3 automated mechanistic interpretability

Project-owned code is licensed under **Apache-2.0**. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

This source-only release contains the implementation, experiment runners,
independent graders, and tests for a bounded mechanistic interpretability campaign
on `Qwen/Qwen3-0.6B`, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`.

The campaign investigates a six-head lookup pathway: upstream heads
`15:4`, `16:4`, `17:1` influence downstream final-position Q vectors of
`19:5`, `21:13`, `22:9`, which transport bound values toward the output.
The code retains failed experiments and their original criteria; availability
of an implementation does not establish a scientific gate pass.

## Install and inspect

```sh
uv sync
uv run --no-sync ruff check
uv run --no-sync pytest -q tests/test_release_verification.py tests/test_poc.py tests/test_poc_upstream.py
```

Code lives in `src/harness/`, `src/agents/`, and versioned `src/poc/` modules.
Experiment state machines are in `scripts/run_bounded_poc*.py`; independent
NumPy audits are in `scripts/audit_bounded_poc*.py`; executable mechanism
controls are in `tests/test_poc*.py`. Scientist subprocesses use
`codex exec --model gpt-6-luna`.

## Verification scope

This package omits all experimental evidence, third-party corpus data, archived
source versions, release-adapter records, and Git history. Full campaign audits
and evidence-dependent tests require those separately cleared inputs. The
canonical campaign's 11-audit result and full-suite test count do not describe
this package. A missing dependency or evidence file is an unavailable check,
never a scientific or integrity pass. Consult `OMISSIONS.md`.

`SOURCE-MANIFEST.json` records the exact included source bytes, base commit,
and locally introduced licensing/packaging changes. Distribution archives have
an external SHA-256 checksum. No model weights or dependency packages are bundled.
The model must be obtained separately under its upstream license.

## Safe experiment execution

Historical runners use fixed output paths. Run new experiments in an isolated
new directory with a new protocol and frozen registration, and never overwrite
canonical evidence. Use a single FIFO `MPSBatchWorker`, one target process at
a time, and the common `/tmp/interp-swarm-mps.lock`; batch size is at most 64.
Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.70` and
`TOKENIZERS_PARALLELISM=false`. Use BF16 weights, FP32 intervention deltas,
and synchronized CPU snapshots. Register selection, thresholds, prompts,
tokens, donors, source hashes and model revision before target inference.

## Distribution

The corpus remains private pending licensing and privacy clearance. The project
license does not relicense third-party data. A future evidence release requires
its own rights review, asset manifest, and explicit verification scope.
