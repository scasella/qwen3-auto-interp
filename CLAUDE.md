# Source-only research agent instructions

## Read and verify

Read `docs/RESULTS_INDEX.md` before interpreting results. Read `docs/ARCHITECTURE.md` before changing intervention or audit code. This is a source-only derivative: raw evidence, corpus, frozen reports, historical source snapshots and private Git history are omitted.

Before changes, run `uv run python scripts/verify_source_only.py`. After changes, run `uv run pytest` and `uv run --no-sync ruff check`; report exact passes/skips. A changed source manifest requires explicit release maintenance, not silently replacing expected hashes. The full campaign verifier `scripts/verify_all_audits.py` requires the private evidence bundle and returns BLOCKED when absent. Package integrity and planted-test success are not scientific audit passes.

## Evidence and distribution guardrails

Keep existing protected artifacts byte-identical if a private evidence bundle is mounted. Historical auditors can write outputs; run them only through the read-only wrapper in a validated evidence workspace. Historical runners use fixed output namespaces. Use new isolated namespaces for new work. Resolve any digest mismatch from an identified exact source; retain an incident outside protected paths.

Keep `data/`, `reports/`, `incidents/`, `release/`, `consolidation/`, corpus, checkpoints, credentials and local logs out of this Git repository. A future upload requires a source allowlist, full tracked-file scan and all-history scan. Keep the repository private and create no release unless the user explicitly authorizes publication. Apache-2.0 covers project-owned code under LICENSE_SCOPE.md; it does not clear third-party material or excluded research artifacts.

## Code map

- Scientist subprocess contracts: `src/agents/`, `src/schemas/`, `src/poc/review.py`.
- Hook ownership and FIFO: `src/harness/qwen_model.py`, `src/harness/mps_queue.py`.
- Early experiments: `src/poc/harness.py`, `analysis.py`, `controls.py`, `templates.py`, `templates_v3.py`, `rescue_v3.py`, `upstream_*`.
- Prospective routes and repairs: `src/poc/v4_*` through `v8_*`.
- Structured formats, cross-scale and question-span relay: `src/poc/v9_*` through `v12_*`; `v10_stable_harness.py` owns the later FP32 routing contract.
- Experiment state machines: `scripts/run_bounded_poc*.py`; independent graders: `scripts/audit_bounded_poc*.py`.
- Read-only evidence adapters: `scripts/verify_all_audits.py`, `scripts/campaign_audit.py` and versioned wrapper scripts. Their private inventories are omitted.
- Executable controls: `tests/test_poc*.py`. Only explicitly named archived-input tests skip when those inputs are absent.

## Execution contract

One FIFO `MPSBatchWorker` owns model forwards. Acquire `/tmp/interp-swarm-mps.lock`; execute target processes sequentially, batch size ≤64, max length 128 unless newly registered otherwise. Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.70`, compatible low watermark `0.50`, and `TOKENIZERS_PARALLELISM=false`. Use BF16 model weights and FP32 deltas. Synchronize CPU snapshots before reading them.

V10 recovery 2 and v11/v12 use synchronized CPU NumPy FP32 routing and intervention sums with BF16 MPS forwards. Preserve operation order, padding, measured BF16 anchors and original clean K/V. Keep GQA query-head patches isolated from shared KV projections. Numerical changes require the bounded tiny-model Metal regressions in ARCHITECTURE.md, not historical target inference.

Invoke advisory scientists using `codex exec --model gpt-6-luna` through the versioned reviewers. Numeric references and deterministic gates decide results. Do not substitute a reviewer narrative for an array audit.

## New experiments

1. Use a new unambiguous namespace, isolated workspace and output/report paths. Preserve failures and preregistration boundaries.
2. Define the claim, fixed/selected heads, donor invariants, budget, tie-breaks, exact gates and failure rules before inference. Write planted positive and mechanism-specific negative controls first.
3. The next campaign seed starts at 20261029. At least 1,429 identifiers were retired by v12; obtain the complete private registry including concurrent work before claiming disjointness. This package alone cannot establish the retired-key union. Validate tokens in actual prompt contexts without examining target answers.
4. Run controls, CPU tests, bounded Metal tests and Ruff. Freeze prompts, tokens, pairings, thresholds, source hashes and model revision before the first target forward. For calibrated selection, use two freezes and only calibration between them.
5. Freeze selection before held-out inference; retain all prompts and failures. Run two fresh processes and compare arrays bit-for-bit. Repeats are numerical replication, not new independent samples.
6. Independently audit saved arrays with NumPy, then run reference-bound review on canonical float64 JSON. Publish a new report distinguishing full conjunctions, favorable components and software failures.

## Scientific limits

V7, operative v10 recovery and v12 pass their registered full conjunctions. V9 fails both tracks; v11 passes 63/63 mechanism criteria but fails its behavioral prerequisite; v8 fails its repair conjunction. See RESULTS_INDEX.md for the full ledger. Coordinates matching across related scales do not establish homology, exclusivity or universality; ratios of median contrasts are not causal mediation fractions.

Further H0 decomposition, cross-family replication and selective 19:5 recency centering are research proposals, not authorization to run them.
