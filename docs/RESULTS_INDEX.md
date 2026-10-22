# Chronological results ledger

> Source-only documentation: frozen reports, arrays, manifests and private registration history are omitted. Reported findings are summarized from the private archive.

This ledger distinguishes scientific verdicts from integrity checks. Every failed gate remains failed. Counts refer to unique prompts, not repeated fresh-process measurements. Early runs were timestamped locally before inference; a Git commit made later is not retroactively called preregistration. Frozen reports, registration commits, manifests and raw arrays remain in the private evidence archive and are not included here. This is a narrative ledger, not an independently reproducible evidence bundle.

## Early real-corpus sweeps

The initial targeted campaign tested 12 units on 768 real excerpts across six domains (seed 20261019). V1 measured 36 hypotheses / 1,620 counterfactual cases with no survivor. Later runs investigated invalid or unmatched test generation and retained construction failures as inconclusive rather than promoting them. These exploratory runs predate the bounded circuit protocol. There are no standalone NumPy campaign auditors for these sweeps; the release inventory checks all saved bytes.


## V1 post-mortem

The v1 post-mortem (private record: `bounded-poc-v1-postmortem.md`) reuses retired v1 data; it is descriptive follow-up, not a new confirmation. Its analysis (private record: `bounded-poc-v1-postmortem`) and report are retained privately. The 13/13 error readout result was not a selection or promotion criterion. There is no separate prospective manifest, preregistration commit, or standalone NumPy audit for this follow-up.

## Bounded experiments and diagnostics

### V1: bounded head discovery

Full proof failed: model prerequisite 115/128 < 116/128. Head 19:5 passed all head-level gates; its readout recovered 13/13 model errors descriptively. Two infrastructure failures preceded screening; all original manifests and repair records remain. Lesson: a strong component cannot waive a global prerequisite.

**Key construction:** Deterministic literal identifier stems plus group numbers; no lexical sampling seed. Runtime seed 42.

**Dataset:** discovery: 16 groups / 64 prompts (4 per group); confirmation: 32 groups / 128 prompts (4 per group).




### Gold/silver: preregistered head 19:5 follow-up

No formal survivor: calibration 55/64 failed 90%. Source-value and query swaps had 128/128 predicted direction on held-out prompts. Helpful local retrieval did not establish a complete independent lookup algorithm.

**Key construction:** Fixed maple/oak and willow/hazel identifier stems plus group numbers; no lexical RNG.

**Dataset:** calibration: 16 groups / 64 prompts (4 per group); confirmation: 32 groups / 128 prompts (4 per group).




### Gold/silver: downstream post-mortem

Post-hoc diagnosis, not prospective confirmation. Head 19:5 helped on all nine calibration errors. Head 24:7 projected toward gold on 192/192 prompts; magnitude was not constant. Zeroing it fixed 4/0 calibration but 3/4 other-format cases; mean replacement fixed 1/3 on each. Only the specified baseline/mean/source-swap subset was repeated. Lesson: a helpful early representation and a harmful later contribution do not imply a safe universal repair.

**Key construction:** Reuses retired gold/silver follow-up data; no new key draw.

**Dataset:** reused 64 calibration + 128 diagnostic confirmation prompts, four prompts per group.




No additional scientist adjudication is recorded for this post-mortem. Its standalone NumPy audit is included in the master verification.

### V2: direct OV value transport

Full proof failed: discovery 56/64 below 90%; primary was 128/128, so the rescue claim had no baseline errors to fix. Direct-source/OV, single/coalition and vocabulary gates passed. Discovery selected the downstream coalition 19:5 + 22:9 + 21:13. Lesson: assignment swaps mainly test value movement when the target position is unchanged.

**Key construction:** Deterministic sequoia/tundra, willow/topaz, quartz/umber stems and group indices; runtime seed 42.

**Dataset:** discovery: 16 groups / 64 prompts (4 per group); primary: 32 groups / 128 prompts (4 per group); vocabulary: 16 groups / 64 prompts (4 per group).




### V3: four-record rescue rejected

Every discovery amplification candidate broke correct answers. Identity was frozen; held-out primary 104/128 and vocabulary 44/64 remained unchanged. Coalition directions 85.15625% / 78.125% missed required bounds. All 17 discovery errors preferred the last-record color, but that observation did not localize a QK or RoPE cause. Lesson: a positive median margin can hide worsening existing errors.

**Key construction:** Deterministic new identifier stems and group indices; runtime seed 42.

**Dataset:** discovery: 8 groups / 64 prompts (8 per group); primary: 16 groups / 128 prompts (8 per group); vocabulary: 8 groups / 64 prompts (8 per group).




### Upstream: opposite-query QK mediation

Full proof failed: single-head Axis A direction, per-head attention-shift requirements and discovery accuracy failed. Coalition double dissociation passed; mediated QK reductions were 5.161254 / 5.303111 logits at 100% direction. Discovery screened 304 heads and froze 17:1 + 16:4 + 15:4 before held-out inference. Lesson: query sensitivity alone does not distinguish identity copying from context-dependent matching.

**Key construction:** Deterministic routeDA/DB, routeHA/HB, routeVA/VB stems; runtime seed 42.

**Dataset:** discovery: 16 groups / 64 prompts (4 per group); primary: 32 groups / 128 prompts (4 per group); vocabulary: 16 groups / 64 prompts (4 per group).




### V4: fixed six-head Q/K/V test

Gates 0–3 failed; Gate 4 passed. Accuracy 62/64, 113/128, 55/64. Axis C K_color effect was only 0.7706 primary logits; the strong K-stamping prediction failed. V4 used multiline templates, despite later briefs describing them as inline. No heads were reselected.

**Key construction:** 20261020; 64 disjoint single-token keys.

**Dataset:** calibration: 8 groups / 64 prompts (8 per group); primary: 16 groups / 128 prompts (8 per group); vocabulary: 8 groups / 64 prompts (8 per group).




### V4 Axis C: post-hoc Q-routing diagnostic

Post-hoc, not a new prospective gate pass. Primary C_q_final 3.797235 logits and C_upstream_Q 3.360059, both 100% direction. Vocabulary C_q_final 2.692898 with 98.4375% direction; C_upstream_Q 2.479070 with 100%. This motivated prospective Q dominance. It did not prove additive logit conservation.

**Key construction:** Reuses frozen v4 primary/vocabulary prompts and donors; no new keys.

**Dataset:** reused 128 primary + 64 vocabulary v4 prompts; eight prompts per factorial group.



audited results (private record: `results.json`)

No separate final Markdown report, `audit.json`, or Luna adjudication exists for this diagnostic. Its NumPy delta checks are in `results.json`; the master CLI verifies protected hashes rather than executing the inference diagnostic script.

### V5: prospective pointer interpretation

Gates 0–3 failed; Gate 4 passed. Accuracy 47/64, 95/128, 41/64. All four direct/upstream Q intervention directions were 100%. 655 arrays repeated exactly; reference review 39/39. Lesson: reliable direction does not guarantee magnitude or behavior prerequisites. Layout/role-label explanations for failure were not separately causally established.

**Key construction:** 20261021; 64 fresh neutral single-token keys.

**Dataset:** calibration: 8 groups / 64 prompts (8 per group); primary: 16 groups / 128 prompts (8 per group); vocabulary: 8 groups / 64 prompts (8 per group).




### V6: new inline templates

Gates 1 and 4 passed; Gates 0, 2 and 3 failed. Primary 37/40 scientific subcriteria, vocabulary 40/40. Accuracy 56/64, 98/128, 52/64. Three primary head-19 attention magnitudes missed thresholds despite 100% Q-effect directions. These illustrative inline templates were not the actual v4 generator.

**Key construction:** 20261022; 64 fresh neutral single-token keys.

**Dataset:** calibration: 8 groups / 64 prompts (8 per group); primary: 16 groups / 128 prompts (8 per group); vocabulary: 8 groups / 64 prompts (8 per group).




### V7: exact v4 family, prospective revised thresholds

All gates passed: 40/40 primary + 40/40 vocabulary scientific subcriteria, accuracy 61/64, 117/128, 52/64. Accuracy prerequisite revised prospectively to 80%; first-head B/C attention thresholds to 7/6 pp. Other 36 criteria unchanged from v6. 655 arrays exact; review 83/83. Lesson: bounded confirmation under revised prospective criteria does not retroactively validate prior failed gates.

**Key construction:** 20261023; 64 fresh neutral single-token keys.

**Dataset:** calibration: 8 groups / 64 prompts (8 per group); primary: 16 groups / 128 prompts (8 per group); vocabulary: 8 groups / 64 prompts (8 per group).




### V8: two-track calibrated repairs

A1, A2, B1 failed; C passed. A: baseline/repaired 122/128, probe 120/128 and 2/6 model errors; 2 fixes / 2 breaks. B primary: 99→102/128, 4/1; B vocabulary: 46→48/64, 2/0. Primary error-case margin test failed; vocabulary net rescue tied naive amplification. 138 arrays exact. Review rejected; 19/20 references supported, invalid reference preserved. Lesson: modest net gains and successful software controls do not satisfy a failed repair conjunction.

**Key construction:** 20261024; 176 fresh single-token keys from an explicitly extended, response-blind frozen whitelist.

**Dataset:** Track A calibration 8×8=64, primary 16×8=128; Track B calibration 8×8=64, primary 16×8=128, vocabulary 8×8=64. A crosses key order × assignment × query; B crosses four queries × two complementary assignments. Total 448 prompts.




## V9–V12: structured formats, cross-scale routing and the question-span relay

These entries extend the frozen v1–v8 ledger. Scientific rejection remains distinct from software/integrity success. The reported 21,073 references include physical copies and repeated historical references; they are not 21,073 independent observations.

| Operative study | Dataset and key seed | Registered verdict | Arrays / file pairs | Review |
|---|---|---|---|---|
| v9-json-hop0-restart1 | 64 calibration + 128 Primary + 64 Vocabulary; 8-prompt groups; seed 20261025 | Track 1 FAIL, Track 2 FAIL; G4 PASS | 1,361 / 120 | reject; 27/27 references |
| v10-qwen17b-restart2 | Same 64/128/64 group counts; seed 20261026 | Gates 0–4 PASS; 108/108 (36 calibration included) | 612 / 66 | support; 109/109 |
| v11-question-hop0 | Same counts; seed 20261027 | Gates 1/2 PASS, 63/63; Gates 0/3 FAIL; Gate 4 PASS | 915 / 63 | reject; corrected attempt 29/29; initial 26 unsupported retained |
| v12-cross-scale-3hop | Same counts; seed 20261028 | Gates 0–4 PASS; 63/63 | 1,212 / 84 | support; 31/31 |

### V9 — a failed structured-format confirmation

The operative restart uses YAML calibration and JSON/Markdown held-out groups. Accuracy was 53/64, 83/128, 43/64. Both scientific tracks fail. Primary C-Q/C-K medians are 0.337667/0.029295 logits (11.53×); Vocabulary is 0.309563/0.039843 (7.77×). Ratios do not rescue failed absolute-effect or direction gates.

Primary B/C U-V-all effects are 0.282709/0.251401 logits versus full-U effects 0.332915/0.301762 (84.9%/83.3% ratios of medians). C-U-V-all direction is 95.3125%; C-U-final-Q and C-U-context-KV medians are 0.007771 and 0.002110. All-position V includes context, question and answer suffix. V9 therefore motivates spatial localization but does not itself prove a question-span route or an additive mediation fraction.

The calibration-selected final-position H0 trio 9:1, 14:1, 11:13 fails held-out confirmation. An earlier invalid-key attempt and its 1,695 files remain intact. Protocol (private record: `bounded-poc-v9-json-hop0-restart1-protocol.md`) · Report (private record: `bounded-poc-v9-json-hop0.md`) · Stage 1 (private record: `stage1.json`) · Stage 2 (private record: `stage2.json`) · Results (private record: `results.json`) · Audit (private record: `audit.json`) · Review adjudication (private record: `review-adjudication.json`).

### V10 — one additional scale, with explicit numerical recovery

The final recovery uses Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Blind-to-held-out calibration selection searches 176 downstream and 144 upstream heads. It recovers D = 22:9, 19:5, 21:13 and U = 17:1, 18:7, 16:4 in greedy order. Discovery costs 2,134 microbatches; each confirmation costs 88. Accuracy is 62/64, 120/128, 62/64. All 108 registered criteria and Gates 0–4 pass. Primary B/C upstream-Q medians are 13.116610/7.501224; Vocabulary 9.767340/6.989500; all directions 100%.

The original interrupted attempt and first recovery remain incomplete, with no held-out verdict. Recovery 1 exceeded the unchanged delta tolerance: 0.000034332275390625 > 0.00003. Recovery 2 freezes explicit synchronized CPU FP32 routing/delta arithmetic, fresh keys, and a new registration. Its maximum delta error is zero. This is prospective recovery after a numerical failure, not one uninterrupted experiment.

The matching downstream coordinates are descriptive evidence, not proof of head homology. The uniform-triplet combinatorial probability 1.12e−6 is not a valid p-value for the nonuniform greedy selection procedure. Operative protocol (private record: `bounded-poc-v10-qwen17b-restart2-protocol.md`) · Corrected final report (private record: `bounded-poc-v10-qwen17b-final.md`) · Original report retained (private record: `bounded-poc-v10-qwen17b.md`) · Manifest (private record: `manifest.json`) · Results (private record: `results.json`) · Audit (private record: `audit.json`) · Adjudication (private record: `review-adjudication.json`).

### V11 — mechanism passes with a failed behavioral prerequisite

Six exhaustive greedy rounds over layers 0–14 evaluate 1,425 coalitions / 2,850 B/C trials / 8,553 calibration microbatches. H0 = 13:6, 12:6, 13:14, 14:12, 11:9, 13:15 is frozen before held-out inference. The intervention patches question-span outputs while clamping final-position residual inputs clean. Only resulting U-question-span V is relayed through U outputs into downstream Q.

All 63 mechanism criteria pass; Primary/Vocabulary H0 B medians are 3.495205/4.450511 and C medians 2.997966/3.495610, all 100% direction. Overall confirmation **fails** because Primary accuracy is 101/128, below 103/128. Calibration 54/64 and Vocabulary 58/64 pass. Changing templates and positions together relative to v9 does not isolate why v9 failed.

Two protected metadata mismatches were restored from retained Git bytes, with failed checks and incidents preserved. Review attempt 1 has 26 unsupported references; attempt 2 has 29/29 supported references and still rejects full confirmation. Protocol (private record: `bounded-poc-v11-question-hop0-protocol.md`) · Report (private record: `bounded-poc-v11-question-hop0.md`) · Stage 1 (private record: `stage1.json`) · Stage 2 (private record: `stage2.json`) · Results (private record: `results.json`) · Audit (private record: `audit.json`) · Final adjudication (private record: `review-adjudication.json`).

### V12 — H0 transfer through the previously selected 1.7B relay

There is no discovery. H0 transfers from v11; U uses v10's 16:4, 17:1, 18:7; D retains 19:5, 21:13, 22:9. All 63 mechanism criteria and Gates 0–4 pass. Accuracy is 64/64, 127/128, 62/64. Main H0 C medians are 2.116088, 1.151426, 1.391571 with 96.875%, 92.1875%, 100% direction. Two processes execute 228 microbatches each, including the predeclared original-U diagnostic. All 1,212 arrays / 84 pairs match, with zero delta and self errors.

The original twelve-coordinate circuit is a separate non-gating diagnostic. Its calibration C-V-span direction is 85.9375%, below the main 90% threshold, and it does not receive a full unchanged-circuit PASS. H0 transfer is not head homology, universality or proof of an equality comparator. The 21,073 reference count includes copies/external provenance. Protocol (private record: `bounded-poc-v12-cross-scale-3hop-protocol.md`) · Report (private record: `bounded-poc-v12-cross-scale-3hop.md`) · Manifest (private record: `manifest.json`) · Results (private record: `results.json`) · Audit (private record: `audit.json`) · Review adjudication (private record: `review-adjudication.json`).

### Consolidation and distribution boundary

The root now contains byte-identical copies of v9–v12 evidence, including unsuccessful attempts. Existing `release/`, `incidents/`, reports and data remain unchanged except the separately user-authorized restoration of `data/.DS_Store` to its retained digest. The additive copy inventory and restoration record live in `consolidation/2026-10-22/`. A different concurrent experiment named `v12-query-update` remains in its separate original workspace; it is not the completed v12 cross-scale study and is not merged into this campaign or its scientific totals.

This source-only ledger omits frozen artifacts and private Git history. See [OMISSIONS.md](../OMISSIONS.md) for the verification boundary.
