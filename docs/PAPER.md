# Controlled query routing and question-span value relays in Qwen3 at two scales

> Source-only documentation: frozen reports, arrays, manifests and private registration history are omitted. Reported findings are summarized from the private archive.

**Manuscript draft for release preparation — 22 October 2026.** Author list, affiliations, archival identifier and submission venue are to be supplied by the research owners. This document synthesizes frozen experimental records; it does not replace their protocols or revise their verdicts.

## Abstract

We investigate in-context key–value retrieval in Qwen3-0.6B and Qwen3-1.7B using controlled activation interventions on a single 24 GB Apple M4 Pro system. An agentic scientist proposes and reviews hypotheses, while prospective registrations, executable negative controls, fixed statistical gates and independently implemented saved-array audits determine outcomes. A three-axis factorial design separates value swaps, query swaps and context-key swaps. At 0.6B, v7 confirms a six-head upstream-to-final-query/downstream-value route under prospectively revised criteria (80/80 scientific subcriteria). Calibration discovery at 1.7B recovers the same downstream coordinates and a partially overlapping upstream coalition; v10 passes 108/108 criteria. A subsequent question-span study at 0.6B passes 63/63 mechanism criteria but fails its behavioral prerequisite. Transferring its six earlier H0 coordinates through the previously target-selected 1.7B relay yields a v12 full pass, including 63/63 mechanism criteria and exact repetition of 1,212 arrays. Structured-format generalization and broad repair experiments fail their registered conjunctions. The results support bounded, context-sensitive controlled routes rather than a complete lookup algorithm, additive mediation decomposition, head homology or cross-family universality.

## 1. Research question and evidential scope

A component can carry information favoring the correct value when the full model favors the wrong output. Our campaign asks which controlled attention-head routes carry that information and how those routes depend on a requested key and its context binding. The motivating alternative explanations are literal query-identity copying, context-key information represented at downstream color-position K, a question-span value relay, output-vocabulary offsets, and positional attention bias.

We distinguish three outcomes throughout: (i) passing a preregistered scientific conjunction; (ii) a favorable component observation within a failed study; and (iii) software/provenance verification. These are not interchangeable. The early real-corpus swarm, which produced no verified survivor, is exploratory background. Its subsequent bounded studies retain failed runs, diagnostics, numerical recoveries and reviewer rejections. The [chronological ledger](RESULTS_INDEX.md) summarizes every study; original reports, registration digests and private commits are retained separately and omitted from this source-only package.

All accuracy measurements concern preference between two specified next-token logits. They are not unrestricted generated-answer accuracy. Group-related factorial prompts and repeated processes are not independent observations.

## 2. Models, tasks and counterfactuals

### 2.1 Pinned models and execution

| Model | Revision | Layers / Q heads / KV heads | Residual width |
|---|---|---|---:|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 28 / 16 / 8 | 1,024 |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 28 / 16 / 8 | 2,048 |

Registered target forwards use BF16 weights on MPS, one FIFO `MPSBatchWorker`, a shared process lock, batch size ≤64 and sequence length ≤128. FP32 intervention arithmetic is synchronized before CPU consumption. Later v10 recovery and v11/v12 use explicit CPU NumPy FP32 routing/delta operations, while transformer forwards remain MPS computations. The original and revised numerical contracts are versioned separately.

### 2.2 Factorial datasets

Later two-record studies contain 64 calibration, 128 Primary and 64 Vocabulary prompts, organized as 8, 16 and 8 complete eight-prompt groups. Each group crosses context-key order, value order and requested key. Calibration and Primary use red/blue; Vocabulary uses green/yellow. Gold/silver and four-record studies have separately registered designs. Fresh identifiers are selected by deterministic response-blind lexical and tokenization rules, with previous keys retired, including incomplete attempts.

For prompt coordinates `(k,c,q)`, donors are A=`(k,1−c,q)`, B=`(k,c,1−q)` and C=`(1−k,c,q)`. A swaps values with keys and question fixed; B changes the requested key with preceding context fixed; C swaps context keys with values and question fixed. A/C preserve exact token multisets, lengths, value positions and, in registered later templates, the final eight tokens. B changes one query token and consequently does not preserve the full multiset. That exception is explicit in the registrations.

V9 uses YAML calibration and quoted JSON/Markdown confirmation; v7, v10, v11 and v12 use the v4-derived natural-language template family. V5 and v6 use different constructions. Between-study accuracy changes cannot be attributed solely to layout, keys or scale because these factors are not independently randomized across the entire campaign.

## 3. Controlled intervention methods

### 3.1 Anchored attention contrasts

For a recipient query head at final position T, let `P=softmax(Q_T Kᵀ/√d)` over valid causal positions and let `H_clean` be the measured BF16 head output immediately before the output projection. The interventions apply contrasts to that measured anchor:

\[
\Delta_{QK}=\sum_j(P'_j-P_j)V_j,\qquad
\Delta_{OV}=\sum_jP_j(V'_j-V_j),\qquad
H'=\mathrm{BF16}(\mathrm{FP32}(H_{clean})+\Delta).
\]

Masks restrict OV changes to color or other positions as registered. Pure-Q changes recompute routing from a donor/mediator Q_T with original recipient K and V at every position. Context K/V equality under B follows from causal masking and is checked over both saved logits, where specified, rather than inferred from a canceling margin.

Anchoring avoids replacing a fused BF16 output with a numerically different explicit reconstruction. A zero contrast recovers the measured anchor exactly. Fused-versus-explicit differences and full donor-reconstruction discrepancies are separately measured; neither is silently rounded to zero. Query-head h uses shared KV head `floor(h/2)`, but patches modify only h’s output slice, with sibling/isolation checks.

### 3.2 Upstream-to-downstream mediation

The 0.6B upstream coalition U=`{15:4,16:4,17:1}` is patched at T. A mediator pass captures Q_T at downstream D=`{19:5,21:13,22:9}` while clamping every D target output to its clean anchor. An output pass replays only those Q vectors onto original clean D K/V. This discards other mediator changes and prevents propagation through changed D outputs. V10’s selected 1.7B U is `{16:4,17:1,18:7}`.

The v11/v12 H0 experiment adds an earlier controlled stage. It patches H0 outputs over `[p_query,T−1]` and clamps residual inputs at T clean through the relevant upstream layers. Only resulting U question-span V is exported. Original U Q/K and context V are retained, and the resulting anchored U outputs enter the pure-Q mediation route above. Direct donor `U_v_qspan` includes T; H0-mediated changes keep T clean. Those scopes differ and their effects are not additive mediated fractions.

### 3.3 Selection, statistics and controls

Selection is confined to registered calibration budgets. V10’s operative recovery screens 176 downstream heads and 144 upstream candidates with fixed greedy selection and numeric tie-breaking, consuming 2,134 microbatches. V11 evaluates six rounds of 240, 239, 238, 237, 236 and 235 extensions: 1,425 coalitions, 2,850 B/C trials and 8,553 calibration microbatches. V12 fixes H0 and both relays before inference and performs no search.

The signed effect is the original correct-answer margin minus the patched margin. Margins and summary statistics use saved logits converted to float64. Direction counts strictly positive effects. Later exact one-sided group sign tests use each complete group’s summed effect, count zero groups as nonpositive and retain every prompt. Magnitude, direction, specificity, attention and behavioral criteria are conjoined according to each frozen protocol; thresholds differ prospectively between studies. Exact process repetition tests numerical repeatability without increasing statistical n.

Planted positive mechanisms are paired with literal-query-only, direct-K-color-only, final-token-only, context-KV-only, nuisance and reversed alternatives as appropriate. Repair controls separately model a constant vocabulary offset and positional recency bias. These establish that the implementation can discriminate specified mechanisms; they are not target-model observations.

## 4. Campaign results

### 4.1 Discovery, OV transport and early failed confirmations

V1 selected 19:5 but failed the full prerequisite at 115/128 correct (89.84375% < 116/128 required). Its separate readout correctly classified all 13 model-error cases descriptively. The gold/silver follow-up also failed calibration, despite 128/128 held-out directional query/value-swap effects. A diagnostic found helpful 19:5 retrieval on 9/9 calibration errors and a gold-favoring 24:7 projection on 192/192 prompts. The projection’s magnitude varied; consistent sign is not a constant vector.

V2 provides direct OV evidence. Its frozen three-head coalition has a Primary median reduction of 5.564765 logits, 61.617% of the median clean/counterfactual gap, and 100% direction. Green/yellow causal transfer passes, with a 5.978752-logit coalition median and 100% direction. V2’s full conjunction still fails. In the upstream study, the coalition’s A-OV and B-QK effects are both 100% directional; the study fails because its single-head, attention and discovery prerequisites also matter.

### 4.2 Context-key swaps and prospective Q dominance

V4 rejects the preregistered K-color-stamping account: Primary C-K median reduction is 0.7706 logits. Its post-hoc Q diagnostic measures C-Q=3.7972 and C-upstream-Q=3.3601. Under C, the question is fixed, making mere query-identity copying insufficient for the observed context-sensitive route.

The exploratory prediction `4.5601−0.7706≈3.7895` and the descriptive ratio `3.7972/(3.7972+0.7706)≈83.1%` are not additive conservation or mediation laws. They combine nonlinear contrasts and medians; the subtraction also crosses axes. The prospective evidence is comparative Q dominance in later registrations.

V5’s four Q-routing interventions attain 100% direction on both confirmation splits but fail the full gates. V6 passes Gate 1 and instrumentation; Primary passes 37/40 scientific criteria and Vocabulary 40/40, with behavioral and single-head magnitude misses. V7 changes templates and thresholds before new inference and passes all Gates 0–4 with 80/80 scientific subcriteria. Its calibration/Primary/Vocabulary accuracies are 61/64, 117/128 and 52/64. These prospective revisions do not regrade earlier failures.

### 4.3 Repair failures and descriptive partial rescue

V3’s pure centered α=2 output amplification increases median correct-answer margin by 1.4676 logits yet worsens 15/17 existing errors and breaks four correct answers. A 1.5565-logit variant additionally mean-ablates 27:14. All discovery candidates violate the selection constraints, so identity is frozen.

V8’s 19:5 calibration profile allocates 13.3824% mean attention to Slot 4 versus 1.6590% to Slot 1 (8.07×). Slot-4 excesses are +5.04, −4.30 and −1.65 percentage points for 19:5, 21:13 and 22:9. These profiles suggest head-specific positional biases but do not causally identify RoPE. Slot-probability centering plus 22:9 weighting yields six fixes and one break over the two held-out four-record splits; the registered B1 conjunction fails its error-margin/superiority requirements. Gold/silver probe/repair gates A1/A2 also fail: the readout scores 120/128 versus baseline 122/128 and recovers 2/6 baseline errors. Across all three held-out splits the repair yields eight fixes and three breaks. Lexical/template changes coincide with changed vocabulary offsets but were not an isolated key-pool causal test.

### 4.4 Structured-format results and upstream decomposition

V9 fails both scientific tracks, with accuracies 53/64, 83/128 and 43/64. Primary C-Q/C-K medians are 0.337667/0.029295 (11.53×); Vocabulary gives 7.77×. Large relative dominance does not satisfy failed absolute-effect or behavioral gates.

Primary upstream B/C V-all effects are 0.282709/0.251401 logits, approximately 84.9%/83.3% of their full-U median effects. C-U-V-all has 95.3125% positive direction, while C-U-final-Q and C-U-context-KV medians are 0.007771 and 0.002110. V-all covers all positions, including the question and answer suffix. It motivates a spatial follow-up rather than proving a localized question-span mechanism. The selected final-position H0 trio does not confirm.

### 4.5 Cross-scale confirmation and question-span transfer

| Study | Model | Calibration | Primary | Vocabulary | Scientific decision |
|---|---|---:|---:|---:|---|
| V10 final recovery | 1.7B | 62/64 | 120/128 | 62/64 | Gates 0–4 PASS; 108/108 criteria |
| V11 | 0.6B | 54/64 | 101/128 | 58/64 | 63/63 mechanism PASS; overall FAIL |
| V12 | 1.7B | 64/64 | 127/128 | 62/64 | Gates 0–4 PASS; 63/63 mechanism criteria |

V10 recovers downstream coordinates `{22:9,19:5,21:13}` through held-out-blind calibration selection, with upstream `{17:1,18:7,16:4}`. The coordinate match is descriptive. `1/choose(176,3)≈1.12×10⁻⁶` assumes uniformly sampled triplets, unlike the actual greedy search, and is not reported as an inferential p-value. Original interrupted and recovery-1 numerical failures remain separate incomplete attempts. Recovery 2 retains the 0.00003 tolerance and passes with exact-zero saved-delta error.

V11 selects H0=`{13:6,12:6,13:14,14:12,11:9,13:15}`. Its direct U-question-span and H0 mechanisms pass on all splits, but Primary misses the 80% prerequisite by two correct prompts. Its overall verdict and final advisory review remain rejection.

V12 transfers H0 unchanged through the v10-selected target-scale U relay. It does not transfer all twelve original coordinates unchanged. The original-U route is a separately registered non-gating diagnostic, whose calibration C-V-span direction is 85.9375%, below the main 90% requirement.

| Study / held-out route | Primary median / direction | Vocabulary median / direction |
|---|---|---|
| V10 B upstream-Q | 13.116610 / 100% | 9.767340 / 100% |
| V10 C upstream-Q | 7.501224 / 100% | 6.989500 / 100% |
| V11 B H0 question-span | 3.495205 / 100% | 4.450511 / 100% |
| V11 C H0 question-span | 2.997966 / 100% | 3.495610 / 100% |
| V12 B H0 question-span | 4.342432 / 99.21875% | 3.783139 / 100% |
| V12 C H0 question-span | 1.151426 / 92.1875% | 1.391571 / 100% |

Each route in this table passes its registered group sign threshold. The selected historical v4–v7 plus v10 series has 960/960 positive held-out directions for each upstream B/C Q route. It includes a post-hoc v4 component and excludes failed-format v9; it is not an all-campaign rate. B/C reuse the same prompts.

## 5. Verification, review and reproducibility

| Study | Exact paired arrays / NPZ pairs | Maximum saved-delta error | Supported scientist references / disposition |
|---|---:|---:|---|
| V7 | 655 / 71 | 0.0000100135803 | 83/83; support |
| V9 operative restart | 1,361 / 120 | 0.0000120252371 | 27/27; reject |
| V10 final recovery | 612 / 66 | 0 | 109/109; support |
| V11 | 915 / 63 | 0 | 29/29 final attempt; reject |
| V12 | 1,212 / 84 | 0 | 31/31; support |

Array identity compares dtype, shape and element bytes, not compressed NPZ container bytes. V11’s original 26 unsupported references are retained alongside the corrected-reference attempt. Review is performed via `codex exec --model gpt-6-luna`; literal metric identifiers bind to canonical audited float64 values. Correct numeric references do not validate every narrative claim and cannot override deterministic rejection.

The v12 protected inventory contains 21,073 references, including physical copies and external historical provenance. Byte-exact metadata restorations and failed checks are disclosed in the original reports and additive consolidation record. Seven raw Git commit/tree/blob proofs preserve later registration bindings without importing private history into a clean source release. Git time and SHA-256 consistency are provenance evidence, not third-party time certification.

The root master executes 15 audits with writes captured in memory and historical source/path resolution outside frozen artifacts. The historical auditor implementations remain unchanged. Production and audit share NumPy primitives in later experiments; independence does not extend to separate arithmetic-library implementations. Full audit reproduction requires private arrays, manifests, source snapshots, registration proofs and pinned checkpoint bytes. See [ARCHITECTURE.md](ARCHITECTURE.md) and the [source-only verification scope](../OMISSIONS.md).

## 6. Limitations and interpretation

The data are bounded synthetic lookup tasks, with selected calibration circuits, registered template populations and two-token readouts. Neither exact repeatability nor group sign significance establishes broader task generalization. Studies revise hypotheses, templates and thresholds prospectively; the campaign is an adaptive research program, not one unchanged confirmatory experiment. Failures on structured formats, gold/silver repair and four-record repair constrain its claims.

The observed paths are controlled interventions, not exclusive decompositions of normal computation. Effect ratios are not additive mediated fractions. The work does not prove an equality comparator, unique slot code, universal truth probe, complete algorithm, head homology or cross-family universality. H0 transfer in v12 uses a relay already selected on the target scale. Its original-coordinate diagnostic has narrower coverage and a failed descriptive threshold.

No corpus, protected artifact or private Git history is cleared for public upload by this manuscript. Apache-2.0 covers project-owned software under the repository’s license scope. Corpus licensing/privacy and evidence distribution remain separate release decisions.

## 7. Conclusion

The campaign establishes reproducible, task-bounded Q-routing and value-transport effects at two related model scales and a controlled earlier question-span value relay. It also records where stronger claims fail. Agentic exploration is useful when its authority ends at executable tests: frozen designs, explicit alternatives, controlled interventions and independent saved-array audits determine what survives.

## Evidence references

The named frozen records are the private primary sources for this manuscript. They are omitted here. The narrative [results ledger](RESULTS_INDEX.md) summarizes the findings; this package cannot independently substantiate them from raw arrays.

1. V1 discovery (private record: `bounded-poc-v1.md`) and post-mortem (private record: `bounded-poc-v1-postmortem.md`).
2. Gold/silver follow-up (private record: `lookup-followup-v1.md`) and downstream diagnostic (private record: `calibration-diagnosis-v1.md`).
3. V2 OV and coalition (private record: `bounded-poc-v2.md`); V3 four-record repair (private record: `bounded-poc-v3.md`).
4. Upstream dissociation (private record: `bounded-poc-upstream.md`); V4 (private record: `bounded-poc-v4-confirm.md`); Axis C Q diagnostic (private record: `results.json`).
5. V5 (private record: `bounded-poc-v5-pointer.md`); V6 (private record: `bounded-poc-v6-final.md`); V7 (private record: `bounded-poc-v7-capstone.md`).
6. V8 repair (private record: `bounded-poc-v8-repair.md`); V9 structured formats (private record: `bounded-poc-v9-json-hop0.md`).
7. V10 corrected final report (private record: `bounded-poc-v10-qwen17b-final.md`).
8. V11 question-span study (private record: `bounded-poc-v11-question-hop0.md`).
9. V12 fixed-coordinate transfer (private record: `bounded-poc-v12-cross-scale-3hop.md`).
