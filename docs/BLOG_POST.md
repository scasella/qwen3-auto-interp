# Following a lookup circuit backward—and keeping the failures

> Source-only documentation: frozen reports, arrays, manifests and private registration history are omitted. Reported findings are summarized from the private archive.

*Release-preparation draft. Results are drawn from the frozen reports linked below. The full evidence remains private; this text does not announce a public data release.*

A small language model can retrieve the right value internally and still prefer the wrong answer. That observation changed our project. We began by asking an automated scientist swarm to name neurons. We ended with a much narrower question: when a model reads two key–value records, which controlled routes carry the requested value to its next-token preference?

On a 24 GB M4 Pro MacBook Pro, we traced a route in Qwen3-0.6B, tested a related route in Qwen3-1.7B, and followed it backward into an earlier question-span relay. The strongest studies passed their prospective gates. Several others failed. The failures are necessary to understand what the positive results establish.

Our approach combined Codex CLI scientist calls using `gpt-6-luna` with executable counterfactuals, planted controls, frozen datasets and thresholds, two fresh target processes, and independently implemented NumPy audits. The scientist could suggest an explanation. It could not change a failed gate into a pass.

## The first useful result was a refusal to confirm

The initial real-corpus sweep produced no survivor among 36 hypotheses. Broad descriptions and synthetic examples were not enough to establish a mechanism. We moved to a bounded lookup task whose counterfactuals could be executed exactly.

V1 selected attention head **19:5**, using zero-based coordinates. Its head-level causal tests passed, but the model preferred the right answer on only **115/128 prompts, or 89.84375%**—below the registered 116/128 minimum. The full proof therefore failed. This was the evaluator refusing confirmation, not a measurement of the model’s refusal behavior.

The post-mortem contained the puzzle: a separate readout of 19:5 was correct on **all 13 model errors**. A helpful component could coexist with a wrong final answer. That did not make it a universal truth detector; it made later contributions worth investigating. V1 report (private record: `bounded-poc-v1.md`) · Post-mortem (private record: `bounded-poc-v1-postmortem.md`)

## Gold and silver separated retrieval from answer preference

A fresh gold/silver follow-up again found helpful retrieval in 19:5, including **9/9 calibration errors**. Head **24:7** had a gold-favoring direct projection on **192/192** diagnostic prompts. Its sign was consistent, but its magnitude varied. Calling this a constant vector would overstate the measurement.

Replacing a head with its dataset mean preserves any component common to the dataset. It cannot remove a strictly constant offset in a planted model. In the real diagnostic, mean-replacing 24:7 fixed one error and broke three on each split. Zeroing it also failed to generalize safely. The lesson was not “delete the late head.” It was that local retrieval and the full model’s answer preference are different quantities, and a repair must be tested on new prompts. Follow-up (private record: `lookup-followup-v1.md`) · Bias diagnostic (private record: `calibration-diagnosis-v1.md`)

## Three counterfactual axes made the circuit testable

Each later two-record group crosses two context-key orders, two value orders, and two queried keys. That creates eight related prompts and three opposite-answer donors:

| Axis | Change | What stays fixed |
|---|---|---|
| A | Swap the two colors | Context keys and question |
| B | Substitute the queried key | Every preceding context token |
| C | Swap context keys | Color positions and question |

Axes A and C preserve the token multiset. B cannot: changing one key occurrence changes the multiset. We registered that precise exception before inference.

V2 showed that 19:5’s effect under value swaps was carried mainly through **OV value transport**. A coalition of **19:5, 21:13, 22:9** reduced the primary margin by **5.5648 logits**, or **61.617% of the median clean/counterfactual gap**, with 100% positive direction. The ratio compares two medians; it is not a fraction of every answer’s computation. The unchanged coalition transferred causally to green/yellow with 100% direction. V2 still did not pass its full conjunction: its discovery accuracy failed a prerequisite and its rescue requirement was not met. V2 (private record: `bounded-poc-v2.md`)

The upstream study used opposite-query donors. At coalition level, value-swap effects were dominated by OV while query-swap effects were dominated by QK, with 100% directional consistency. It selected **17:1, 16:4, 15:4** as an upstream coalition. Yet the full study failed single-head, attention-magnitude and discovery-accuracy requirements. Strong coalition evidence did not waive those gates. Upstream report (private record: `bounded-poc-upstream.md`)

## Swapping context keys changed the explanation

A plausible story was that context keys were stamped onto the attention keys at the color positions. V4 tested that story. Its primary `C_k_color` effect was only **0.7706 logits**, below the registered routing requirement.

A post-hoc diagnostic instead patched only downstream final-position queries. It measured **3.7972 logits** for `C_q_final` and **3.3601 logits** for the upstream-to-final-Q route. The question itself was unchanged. The measured route therefore carried context-dependent information, not only the literal requested key.

An exploratory subtraction predicted about **3.7895 logits**, close to the observed 3.7972. The Q contrast also represented about 83.1% of the sum of the Q and K median contrasts. Those coincidences motivated new experiments, but they are not a conservation law: attention and downstream computation are nonlinear, medians do not add, and the subtraction mixed axes. The defensible claim is comparative Q dominance under specified interventions. V4 (private record: `bounded-poc-v4-confirm.md`) · Axis C diagnostic (private record: `results.json`)

## Confirmation required keeping near misses

V5 introduced executable alternatives: a literal-query-only model should pass query-swap mediation but fail context-swap mediation; a direct-K-color model should fail the Q-pointer claim. These controls passed their intended software checks. The real model’s four Q-routing interventions had 100% direction, but behavioral and magnitude gates failed.

V6 passed its complete direct-dissociation Gate 1 and its instrumentation Gate 4. Vocabulary passed 40/40 scientific criteria; Primary passed 37/40. Three single-head attention magnitudes and the task prerequisites prevented full confirmation. Changes in templates and key populations coincided with accuracy changes; the campaign did not isolate layout as their sole cause.

V7 reused the actual v4 template generator and prospectively changed the accuracy and direct-head attention thresholds. On fresh keys it passed **all Gates 0–4 and 80/80 scientific subcriteria**. Its reference-bound Luna review supported **83/83 numeric references**. That was a successful new experiment, not a retroactive rescue of v4–v6. V5 (private record: `bounded-poc-v5-pointer.md`) · V6 (private record: `bounded-poc-v6-final.md`) · V7 (private record: `bounded-poc-v7-capstone.md`)

## The repair experiments exposed a second failure regime

Four records were harder than two. In v3, pure centered output amplification at α=2 increased the median correct-answer margin by **1.4676 logits**, while worsening **15/17 existing errors** and breaking four correct answers. The often-quoted **1.5565-logit** variant also mean-ablated head 27:14. A favorable aggregate margin was a poor proxy for repairing errors. The preregistered selector abstained. V3 (private record: `bounded-poc-v3.md`)

V8 measured a head-specific positional profile. Head 19:5 placed **13.3824%** mean attention on Slot 4 and **1.6590%** on Slot 1: an **8.07× ratio**, not a fitted slope or a causal demonstration of RoPE. Its Slot-4 excess was +5.04 percentage points; 21:13 and 22:9 had negative excesses, −4.30 and −1.65 points.

Calibration-selected slot-probability centering plus 22:9 weighting fixed **six** and broke **one** across the two held-out four-record splits. That 6:1 ratio is promising, but the registered repair conjunction failed: the Primary error-case margin test failed and Vocabulary net rescue tied the naive comparator. Including gold/silver, all three held-out splits had eight fixes and three breaks. The gold/silver probe and repair gates also failed on the new lexical/template population. V8 established neither a general repair nor a general truth probe. V8 (private record: `bounded-poc-v8-repair.md`)

## Structured formats suggested an earlier value relay—and also failed

V9 moved to YAML calibration and quoted JSON/Markdown held-out formats. Both scientific tracks failed. Primary accuracy was **83/128**, and the frozen final-position Hop-0 coalition had near-zero held-out effects.

Within that failure, upstream all-position V patches were more effective than upstream final-Q or context-KV patches. Primary C-U-V-all had **95.3125%** positive direction and a **0.2514-logit** median, compared with **0.0078** for final Q and **0.0021** for context KV. B/C V-all/full-U ratios were roughly 85%/83%. These were overlapping contrasts, not mediated percentages. Crucially, V-all included the question and answer suffix; it did not localize the effect to those positions by itself.

Direct C-Q/C-K dominance remained large—**11.53× on Primary**, 7.77× on Vocabulary—but did not compensate for failed absolute-effect and task gates. V9’s useful output was a sharper follow-up question. V9 (private record: `bounded-poc-v9-json-hop0.md`)

## A larger model recovered the same downstream coordinates

V10 performed calibration discovery on **Qwen3-1.7B**, using 2,134 microbatches in the operative recovery. It selected downstream **22:9, 19:5, 21:13**, exactly the earlier coordinates, and upstream **17:1, 18:7, 16:4**. The upstream set changed one head.

All **108/108 criteria** and Gates 0–4 passed, with Primary accuracy 120/128 and Vocabulary 62/64. All 612 paired arrays repeated exactly, and 109/109 Luna references were supported. Matching coordinates are striking, but selection was greedy and nonuniform. The number `1 / choose(176,3) ≈ 1.12e−6` is not a valid discovery p-value for that procedure.

This success followed two retained incomplete attempts. The first recovery exceeded the fixed delta-error limit by a small amount. A new versioned harness used synchronized CPU FP32 routing and delta arithmetic, while BF16 model forwards remained on MPS. Fresh keys and a fresh registration preceded the successful recovery. No tolerance was relaxed. V10 final report (private record: `bounded-poc-v10-qwen17b-final.md`)

Across the selected v4–v7 and v10 held-out datasets, each B/C upstream-Q route had **960/960 positive directions**. That includes a post-hoc v4 component and omits failed-format v9; it is not an all-campaign generalization rate.

## The third stage lives on the question span

V11 searched earlier heads at question-span positions, clamping the final-position residual clean. Only the resulting U-value information could flow through the controlled downstream route. Six greedy rounds selected:

**13:6, 12:6, 13:14, 14:12, 11:9, 13:15.**

All **63 mechanism criteria** passed on calibration, Primary and Vocabulary. Nevertheless, Primary accuracy was **101/128**, two answers short of the 103/128 prerequisite. V11’s overall verdict remains **FAIL**, and Luna’s corrected-reference review still rejected full confirmation. V11 (private record: `bounded-poc-v11-question-hop0.md`)

V12 transferred those six H0 coordinates to 1.7B without a new search. Its main relay used the upstream heads already selected on 1.7B in v10. It passed **63/63 mechanism criteria and Gates 0–4**, with accuracies **64/64, 127/128 and 62/64**. All **1,212 arrays across 84 pairs** repeated exactly, and the audit checked **21,073 protected references**.

This is a fixed H0 transfer through a previously target-selected relay. It is not unchanged transfer of all twelve original heads. The exact original relay was tested separately as a non-gating diagnostic and did not receive a full-conjunction pass. V12 (private record: `bounded-poc-v12-cross-scale-3hop.md`)

## What made the automation useful

The result is a controlled route:

```text
H0 question-span outputs
    → U question-span values
    → U output at the final position
    → downstream final-position queries
    → downstream value transport
    → two-token preference
```

We have not proved an equality comparator, a unique symbolic pointer, exclusive circuit membership, head homology, or a complete lookup algorithm. Two related Qwen scales do not establish universality.

The automation helped because it could propose many concrete tests and execute a bounded plan. The trust came from the parts it could not negotiate: frozen prompts, fixed budgets, planted alternatives, exact identities, saved arrays, independent graders, retained failures, and numeric references that bind a reviewer to the evidence.

The [manuscript](PAPER.md), [complete ledger](RESULTS_INDEX.md), and [implementation specification](ARCHITECTURE.md) separate those claims and their limits. The current public-release boundary remains source-only and private; publication and corpus clearance are separate decisions.
