# Harness architecture and mathematics

> Source-only documentation: frozen reports, arrays, manifests and private registration history are omitted. Reported findings are summarized from the private archive.

Read [the result ledger](RESULTS_INDEX.md) for the frozen decisions and [AGENTS.md](../AGENTS.md) before changing code. All layer/head indices are zero-based. Pinned targets are Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca` and Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

## Execution boundaries

`src/agents/client.py` invokes the Codex CLI scientist engine. Scientists receive bounded evidence and strict schemas. Their output is a candidate explanation or an advisory review, never permission to bypass an experimental gate. The initial broad swarm uses `src/agents/explorer.py`, `falsifier.py`, and `circuit_tracer.py`; the bounded PoCs use explicit executable templates and versioned graders.

`src/harness/mps_queue.py::MPSBatchWorker` owns the harness, bounds the producer queue, and executes compatible deterministic microbatches through a single executor thread. A probe cannot invoke a model outside the claimed worker. Each registered target runner also acquires `/tmp/interp-swarm-mps.lock`. Batch size is at most 64, maximum sequence length 128, weights BF16, and intervention arithmetic FP32. Different experiment processes run sequentially.

Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.70` and `TOKENIZERS_PARALLELISM=false`. Snapshots use `cpu_snapshot` in `src/poc/harness.py`; conversion and synchronization occur before values are read as NumPy/Python objects. Earlier CPU-copy/donor-dtype failures and their renewed runtime manifests are preserved in v1. Never consume an asynchronous BF16 CPU snapshot before completion. The queue periodically clears the MPS allocation cache.

## Anchored fused-kernel deltas

For query head h at final position T, define

```
P = softmax(Q_T K^T / sqrt(d_head) + causal_mask)
H_explicit = sum_j P_j V_j
H_clean = measured BF16 head output immediately before o_proj
```

Fused BF16 SDPA and explicit FP32 attention need not give the same output. Reconstruction differences reached roughly 0.03–0.045 in the relevant instrumentation checks. This discrepancy is distinct from a self-patch error and is not silently treated as zero.

The patch applies a contrast to the measured anchor:

```
delta_QK = sum_j (P_donor,j - P_clean,j) V_clean,j
delta_OV = sum_j P_clean,j (V_donor,j - V_clean,j) mask_j
H_patched = BF16(FP32(H_clean) + FP32(delta))
```

A self donor makes delta exactly zero; rounding an already representable BF16 anchor then recovers the measured vector exactly. The implementation avoids assignment when the target equals the current head. It does **not** replace the measured head with a numerically different explicit reconstruction. Registered identity margin errors were exactly `0.00000000`.

Qwen3 has 16 query heads and 8 KV heads: query head h reads KV head `floor(h/2)`. `src/poc/v4_harness.py` hooks post-normalization Q/K, applies RoPE, captures V, and replaces only the h-th slice of the input to `o_proj` at T. It never overwrites the shared `k_proj` or `v_proj` output. The paired GQA sibling and other positions remain intact; saved sibling-error arrays and planted tests check this isolation.

V2 path reconstruction is in `src/poc/harness.py`; upstream QK replay is in `src/poc/upstream_harness.py`; Q-only/K-only interventions are in `src/poc/v4_harness.py`. V5–v7 reuse the latter harness with separate registered templates, criteria, and controls.

## Three counterfactual axes

An eight-prompt group crosses context-key order, value order, and requested key. The three opposite-answer donors are:

| Axis | Changed tokens | Controlled quantity |
|---|---|---|
| A | Swap the two bound values | OV value transport at the two value positions |
| B | Substitute the requested key once | Final-position query routing with identical preceding context |
| C | Swap the two context keys | Context-sensitive routing with fixed question and fixed value tokens |

Axes A/C preserve the full token multiset, length, value positions, and final eight tokens. Axis B preserves context, length, and value positions but **cannot** preserve the full multiset when only one query token changes. That precise invariant was explicitly approved before v4 registration. Parser and tokenizer assertions enforce these distinctions.

Q/K/V are contextual activations, not token identities. Identical value tokens do not by themselves guarantee identical V under changed preceding context. Only the causal mask guarantees that changing a later question cannot alter preceding K/V.

## Two-pass pure-Q_final mediation

1. Capture clean recipient Q, K, V, head outputs and donor upstream outputs.
2. In the mediator pass, replace upstream `17:1, 16:4, 15:4` outputs at T with donor values. At each downstream target (`19:5, 21:13, 22:9`), capture the resulting Q and clamp that target's output to its original clean output. This prevents target-to-target propagation through their changed outputs.
3. In the output pass, replay **only the captured Q_T**. Use original recipient K at every position, including T, and original recipient V everywhere. Recompute P and apply the anchored QK contrast above.

This measures the controlled upstream-to-downstream-Q route. Full QK-pattern replay, which can include changes in K, is a different intervention; upstream and v4 recorded it separately. `B_upstream_Q` and `C_upstream_Q` refer to pure-Q replay in v5–v7 and the specified v4 diagnostics.

Attention gain/loss summaries in v5–v7 are medians across prompts of each prompt's coalition mean. Directionality refers to positive signed reductions or attention changes as separately specified. Group sign tests aggregate within factorial groups; related prompts are not independent Bernoulli samples.

Patch effects are overlapping nonlinear contrasts. Their medians need not add to a full-patch median, and their ratios are not automatically mediation percentages. Axis C rules out a literal-query-identity-only account for the measured route; it does not prove a unique symbolic slot code.

## Planted controls before inference

| Control | Executable mechanism | Expected verdict |
|---|---|---|
| `planted_pointer_model` | Upstream context key matching produces a final-Q slot pointer, then downstream V dereferencing | Pass registered pointer tests |
| `literal_query_only_control` | Upstream copies only literal question identity | Pass B mediation; zero C mediation; fail C gate |
| `direct_k_color_only_control` | Context identity carried at K_color without upstream Q resolution | Fail Q-dominance and upstream C tests |
| `planted_vocab_offset_model` | Correct early retrieval plus constant late gold offset | Mean replacement preserves offset; centered repair and probe succeed |
| `planted_qk_recency_model` | Last-slot prior in attention with variable value assignment | Naive output amplification worsens errors; slot centering rescues them |
| Nuisance / reversed | No relevant relation, or a contrast with the wrong sign | Fail |

These are software and evaluator tests, not evidence that Qwen implements the planted mechanism. See `v5_controls.py` through `v8_controls.py` and `tests/test_poc_*.py`.

## V8: calibration-only selection and single-pass repair

V8 Stage 1 freezes all 448 prompts, candidate grids, thresholds, and source. Only 64 calibration prompts per track may then be executed. Stage 2 freezes their arrays, calibration means, centroid probe, and selected rules before held-out inference.

Track A selects among α ∈ {1, 1.5, 2, 2.5} and γ ∈ {0, .5, 1}. The head intervention is `(alpha−1)(H_current−mu_H_cal)`, and output centering subtracts γ times the baseline calibration mean two-token logits. This final-output adjustment is not a localized causal removal of head 24:7. The centroid probe uses concatenated final outputs of the three heads; leave-group-out calibration evaluation precedes fitting the final probe.

Track B selects score-space or probability-space centering, β ∈ {.5, .75, 1, 1.25}, and 22:9 amplification ∈ {1, 1.5, 2}. For the four value slots:

```
excess_j = mean_cal(P_j) - mean_over_four_slots(mean_cal(P))
P_adjusted,j = P_current,j - beta * excess_j
```

The probability variant permits negative coefficients; it is a linear value-weight intervention, not necessarily a valid probability distribution. The score variant re-softmaxes adjusted scores. Keys are recorded but only value slots are edited in this grid. The frozen comparator uses α=2 centered output amplification on all three heads without QK adjustment.

`v8_harness.py` uses the current prompt's activations at each layer, so earlier interventions can alter later inputs. This differs deliberately from historical clean-anchor path mediation. Calibration labels select the rule and fit the probe, as preregistered; calibration means themselves are unlabeled. Held-out interventions receive no answer labels, donor prompts, or test-split means. Both tracks use deterministic tie-breaking: net fixes, fewer regressions, median signed gain, enumeration order.

## Reference-bound scientist review

`src/poc/review.py` sends one canonical JSON serialization of audited float64 facts to `codex exec --model gpt-6-luna`. Later review schemas request **metric references**, not decimal values. Code binds each returned reference to the exact same numeric object used by the adjudicator. Missing keys are unsupported even when a lookup returns null.

V7 had **83/83** supported references. V8 had **19/20**: one invalid metric path was preserved and explicitly flagged. Earlier decimal-transcription errors also remain in the evidence. Binding prevents numeric transcription drift but does not validate the prose's interpretation, and the reviewer cannot change a gate.

## Read-only campaign verification

`verify_all_audits.py` imports and executes the 11 original v1–v8 NumPy auditors, then the four v9–v12 audits through the consolidation adapter described below. It does not copy their graders into a new implementation. A small I/O adapter:

- resolves preregistered source reads to content-addressed historical bytes;
- translates recorded absolute author-workspace paths to the corresponding local evidence;
- captures attempted JSON/Markdown writes in memory;
- checks regenerated JSON against preserved JSON (excluding only upstream's regenerated `audited_utc` metadata);
- verifies the release inventory, original v8 inventory, auditor/source dependencies before and after execution.

No old file is opened for writing. Assertions still execute against actual local bytes. Checkpoint reads require the real pinned weights, optionally supplied with `--checkpoint`. The adapter is for trusted repository audit code, not a security sandbox for arbitrary plugins. `n/a` in the summary means an auditor did not expose an independent delta-error aggregate, not zero error. Same-dataset array identity is repeatability, not independent sampling.

The Axis C follow-up and early targeted sweeps have protected bytes but no separate standalone NumPy audit entry point. Their integrity checks must not be described as a new scientific reanalysis. Original absolute paths remain inside the evidence; portability is supplied outside it.

## Fresh target replays

Historical runners have fixed output paths and assume exclusive creation. Never invoke `freeze`, `calibration`, or `confirm*` in the published evidence tree. To replay:

1. Run the master verification and retain its summary. Make an isolated copy of the source plus read-only reference evidence.
2. Create a **new** replay directory and a small runner adapter that redirects OUT there; preserve reference manifests, dataset, controls and calibration/discovery artifacts byte-for-byte. Redirect ROOT/source-hash resolution to the registered historical snapshot, not current shared modules. Record adapter source and the distinction between replay and the original experiment.
3. Inspect the selected historical runner's `--help` and registration checks before executing. V8 requires both frozen stages and calibration arrays; v7 requires its fixed dataset/manifest and controls. Do not refit or rerun selection.
4. With the model revision installed and a single process lock held, execute confirmation 1, exit the process, then confirmation 2. Use the registered batch shape, dtypes and environment. Record new start/process metadata separately.
5. Compare corresponding arrays within the new runs and against the published reference. Run the independent NumPy arithmetic against the new output through a separate read-only adapter. A mismatch is a recorded result, not permission to overwrite references.

This procedure requires a replay adapter because changing fixed paths or source resolution changes runtime provenance. The repository does not claim that a historical one-line confirmation command safely reproduces into an already populated release tree. For a new scientific experiment rather than a replay, follow the next-experiment checklist in AGENTS.md.


## V10 synchronized CPU FP32 numerical contract

The original 1.7B recovery produced a saved delta error 0.000034332275390625, above the unchanged 0.00003 limit. Large value vectors amplified small attention-rounding differences. The failed attempt is preserved; it was not fixed by increasing tolerance or editing arrays.

`src/poc/v10_stable_harness.py` computes routing and intervention contrasts using synchronized CPU NumPy FP32 values. The target transformer still runs BF16 on MPS. For each valid sequence, scaled scores are explicitly accumulated in FP32; max-subtracted exponentials are normalized in FP32; contrasts use explicit FP32 sums. Padding positions are excluded from the score normalization and retained consistently in saved/application shapes. The original clean K/V and measured head-output anchor remain authoritative. Applied vectors are rounded back to BF16, and the auditor independently reconstructs that rounding.

V11 and v12 inherit this arithmetic. Their independent auditor reproduces the saved U and D deltas, original-K/V isolation and actual applied BF16 vectors with exact zero error. Production and audit share NumPy primitives; independence refers to code, saved-array reconstruction, selection and grading, not independent arithmetic libraries. Zero arithmetic error is separate from full donor reconstruction error, whose paired-logit gate remains ≤0.05.

## V11/V12: H0 → U question-span V → D final Q

H0 = `(13:6, 12:6, 13:14, 14:12, 11:9, 13:15)` is selected in v11 and fixed in v12. V11 uses U = `(17:1, 16:4, 15:4)`; v12’s main relay uses v10-selected U = `(16:4, 17:1, 18:7)`. Both use D = `(19:5, 21:13, 22:9)`.

1. Capture original clean recipient U Q/K/V, U outputs, downstream Q/K/V and measured outputs, plus donor H0 slices.
2. Patch H0 output slices at positions `[p_query,T−1]`. At every decoder layer through layer 17 (v11) or 18 (v12 main), clamp the input residual at T clean. Verify preceding context U V and final-position U V are unchanged. Export only the resulting U V question span; discard other changes from this pass.
3. Reconstruct U outputs at T from the exported V with original U Q/K. Add the explicit FP32 contrast to the measured clean U output, round to BF16 and mediate into D Q_T while clamping D target outputs clean.
4. Replay only the captured D Q_T with original D K/V, including original K at T. Record logits, attention, FP32 deltas, BF16 applied vectors, clamps and sibling errors.

Direct donor `U_v_qspan` includes `[p_query,T]`; the H0 intervention modifies `[p_query,T−1]` while clamping T. These scopes must not be conflated. The full-U donor QKV reconstruction is a fidelity comparison to donor U-output mediation, not a claim that Q/K/V effects add. The median absolute paired margin discrepancy prevents opposite-signed errors canceling.

Planted question-span relay, final-token-only, context-KV-only, nuisance and reversed controls distinguish these routes. The independent graders reconstruct their raw values and logits. A positive planted control validates an implementation; it does not identify the model’s algorithm.

V11’s six greedy rounds use 1,425 coalitions and 2,850 B/C trials, all audited. V12 performs no search: 168 main plus 60 diagnostic microbatches per process. Capturing all candidate activations is instrumentation and does not authorize candidate evaluation or reselection.

## Source-only boundary and portability

The private campaign archive uses read-only source/path adapters and verifies 21,073 historical references. It retains registered source versions, local registration proofs and original array pairs. None of that evidence or private Git history is included in this repository. Full audit commands require those separately managed inputs. A missing input is a blocked audit, never a pass.

This derivative replaces local worktree paths with `QWEN_V9_WORKSPACE`, `QWEN_V10_WORKSPACE` and `QWEN_V11_WORKSPACE` (default: repository root). The 1.7B checkpoint location is controlled by `QWEN17_CHECKPOINT`, defaulting to a repository-relative pinned snapshot. The generic `/tmp/interp-swarm-mps.lock` is an inter-process lock, not an author-specific path. Scientific graders, gates and intervention math are unchanged.

Use `scripts/verify_source_only.py` to check this package. `tests/conftest.py` skips only individually named tests whose required private archived input is absent. It preserves the historical v8 dataset scope when its inventory is present. These skips do not claim evidence verification or scientific reproduction.

## Metal regressions

This bounded suite loads tiny synthetic transformer configurations and replays saved arithmetic; it does not run a new target experiment or invoke Luna. Hold the common process lock, set both allocator watermarks and keep test processes sequential:

```sh
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.70 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.50 TOKENIZERS_PARALLELISM=false \
QWEN_HARDWARE_TEST=1 uv run pytest tests/test_poc.py tests/test_poc_upstream.py \
  tests/test_poc_v2.py tests/test_poc_v3.py tests/test_poc_v4_confirm.py \
  tests/test_poc_v5_pointer.py tests/test_poc_v6_final.py tests/test_poc_v7_capstone.py \
  tests/test_poc_v8_repair.py tests/test_poc_v9_json_hop0.py \
  tests/test_poc_v10_numerics.py tests/test_poc_v10_qwen17b.py \
  tests/test_poc_v10_recovery.py tests/test_poc_v10_restart2.py \
  tests/test_poc_v10_stable_harness.py tests/test_poc_v11_question_hop0.py \
  tests/test_poc_v12_cross_scale_3hop.py
```

The command assumes `/tmp/interp-swarm-mps.lock` is held externally; hold the lock in the process launching pytest. Never run an unrestricted hardware suite against the archived target runners.
