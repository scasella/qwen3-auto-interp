# Zero-Trust Automated Mechanistic Interpretability

A falsification-first study of in-context key–value retrieval in **Qwen3-0.6B and Qwen3-1.7B**, conducted on one **Apple M4 Pro MacBook Pro with 24 GB unified memory**. Codex CLI scientist calls (`gpt-6-luna`) propose and review explanations; executable counterfactuals, frozen gates, raw arrays, and independent NumPy audits determine the verdict.

**V7 confirmed a bounded six-head route at 0.6B; v10 confirmed a related route at 1.7B; v12 confirmed transfer of six earlier heads through the 1.7B relay. V9 and v11 retain failed overall verdicts, and v8’s broader repair claim failed.** This source-only repository describes the unsuccessful sweeps, near misses, post-hoc diagnostics, and rejected claims alongside the positive result. Their frozen artifacts remain private.

Model: `Qwen/Qwen3-0.6B`, revision **`c1899de289a04d12100db370d81485cdf75e47ca`**. Head indices are zero-based. “Accuracy” means preference between two specified next-token logits, not unrestricted answer generation.

[Blog post](docs/BLOG_POST.md) · [Manuscript](docs/PAPER.md) · [Results ledger](docs/RESULTS_INDEX.md) · [Harness and mathematics](docs/ARCHITECTURE.md) · [Agent onboarding](AGENTS.md)

## Five findings

### 1. A two-stage route connects six attention heads

The upstream coalition **15:4, 16:4, 17:1** changes downstream final-position query vectors. The downstream coalition **19:5, 21:13, 22:9** uses the resulting routing to transport information from bound value positions through OV.

Across v4–v7, **each** of `B_upstream_Q` and `C_upstream_Q` had positive correct-answer margin reduction on **768/768 prompts (100%)**: 128 primary + 64 vocabulary prompts in each of four datasets. The v4 Axis C component comes from the **post-hoc diagnostic**, not v4's preregistered gates. These are two intervention series on 768 prompts, not 1,536 independent observations; fresh-process repeats are not counted twice.

V7 (private record: `bounded-poc-v7-capstone.md`) passed **40/40 scientific subcriteria on primary and 40/40 on vocabulary**, plus its behavior and instrumentation gates. Its accuracy threshold was prospectively revised to 80%; primary accuracy was 117/128 and vocabulary accuracy 52/64. Earlier failed experiments were not regraded. This supports context-sensitive Q routing and value transport on these tasks; it does not establish an exclusive circuit or a literal symbolic pointer representation.

### 2. Swapping context keys exposed Q dominance

Axis C keeps the question and value tokens fixed and swaps only the context keys. On v4 primary prompts, `C_k_color` reduced the margin by **0.7706 logits**, while the later `C_q_final` diagnostic measured **3.7972 logits**. This rejects the registered claim that context-key information is carried predominantly by K at the color positions.

The exploratory subtraction `4.5601 − 0.7706 ≈ 3.7895` was close to the measured Q effect. Likewise, `3.7972 / (3.7972 + 0.7706) ≈ 83.1%` describes the ratio of two median contrasts. **Neither is a conservation law or a causal mediation percentage:** softmax, downstream computation, and medians are nonlinear, and the subtraction mixes counterfactual axes. The prospective evidence is the later fixed-Q-versus-K dominance tests, not additive logit closure. V4 (private record: `bounded-poc-v4-confirm.md`) · Axis C diagnostic arrays and results (private record: `results.json`)

### 3. Transfer is real, but vocabulary and task scope matter

The fixed coalition transferred from **red/blue to green/yellow**, including all v7 gates. A separately preregistered **gold/silver** follow-up found query-dependent source-value retrieval in head 19:5: source-value and query swaps both had the predicted direction on **128/128** held-out prompts. That experiment still failed its calibration prerequisite. Gold/silver was not a successful full six-head confirmation, and four-record lookup exposed further failures. Gold/silver follow-up (private record: `lookup-followup-v1.md`) · V3 (private record: `bounded-poc-v3.md`)

### 4. Helpful retrieval can coexist with a wrong final answer

Head 19:5's separate readout was correct on **13/13 v1 model errors**, and on **9/9 gold/silver calibration errors**; on those nine, removing or corrupting its value contribution worsened the correct-answer margin. Head **24:7** had a gold-favoring direct projection on **192/192** diagnostic prompts. Its magnitude varied: this was a consistent sign, **not a constant vector or universal gold bias**.

Mean replacement retains a component common across examples. It therefore cannot remove a strictly constant offset in a planted model; in the real diagnostic, mean-replacing 24:7 fixed one error but broke three on each split. Zeroing it also failed to generalize safely. In v8 (private record: `bounded-poc-v8-repair.md`), fresh single-token keys changed the observed regime: the calibration mean favored silver, and baseline held-out accuracy was already 122/128. The coalition probe scored **120/128**, recovered **2/6** baseline errors, and the repair fixed **2** while breaking **2**. A changed lexical/template population coincided with a changed offset; a controlled key-pool-only causal comparison was not performed. Post-mortem (private record: `calibration-diagnosis-v1.md`)

### 5. Four-record positional bias is head-specific, and partial rescue is not confirmation

In v8 calibration, head 19:5 assigned mean attention **13.3824%** to Slot 4 versus **1.6590%** to Slot 1: an **8.07× ratio**, not a fitted slope. Its Slot-4 excess over its four-slot mean was **+5.04 percentage points**. Heads 21:13 and 22:9 instead had excesses **−4.30** and **−1.65 points**, respectively. A mean positional profile alone does not establish a RoPE-specific cause.

In v3, centered output amplification at α=2 increased the median correct-answer margin by **1.4676 logits** yet worsened **15/17 existing errors** and broke four correct answers. The **1.5565-logit** variant also included mean ablation of head 27:14. The preregistered selector abstained.

V8's calibration-selected slot-probability centering plus 22:9 weighting fixed **6** and broke **1** across the two held-out four-record splits: a **6:1** ratio and net +5/192. Across all three held-out splits, including gold/silver, the total was **8 fixes / 3 breaks**, not 8/1. Primary error-case margin improvement failed, and vocabulary net rescue tied the naive comparator. Thus **v8 Gates A1, A2, and B1 failed; Gate C passed**. QK centering and 22:9 weighting were combined, so their separate contributions remain unresolved. V8 complete report (private record: `bounded-poc-v8-repair.md`)

## From six heads to a controlled three-stage route

| Study | Model | Registered result | What it establishes |
|---|---|---|---|
| V9 (private record: `bounded-poc-v9-json-hop0.md`) | 0.6B | Both tracks FAIL; instrumentation PASS | Structured-format effects weaken; all-position upstream V carries larger effects than upstream final Q or context KV, but the final-position Hop-0 search fails |
| V10 final recovery (private record: `bounded-poc-v10-qwen17b-final.md`) | 1.7B | Gates 0–4 PASS; 108/108 criteria | Calibration discovery recovers downstream 19:5, 21:13, 22:9 and upstream 16:4, 17:1, 18:7; held-out pure-Q routing confirms |
| V11 (private record: `bounded-poc-v11-question-hop0.md`) | 0.6B | 63/63 mechanism criteria PASS; overall FAIL | Six layer-11–14 heads influence the question-span V relay; Primary accuracy 101/128 misses 103/128 |
| V12 (private record: `bounded-poc-v12-cross-scale-3hop.md`) | 1.7B | Gates 0–4 PASS; 63/63 mechanism criteria | The six fixed H0 coordinates transfer through v10’s target-selected relay, with no new search |

V10 used **2,134 calibration microbatches** in its operative recovery, then two fresh confirmations. Its downstream coordinates match v7, but selection was not uniform random sampling: `1 / choose(176, 3) ≈ 1.12e−6` is an illustrative combinatorial reference, **not a calibrated discovery p-value**. Two earlier interrupted/numerically invalid attempts remain preserved. V10’s 108 criteria include 36 calibration criteria; calibration is in-sample.

Across v4–v7 plus v10, **each** B/C upstream pure-Q intervention had positive margin reduction on **960/960 held-out prompts**. This selected historical series includes v4’s post-hoc Axis C diagnostic, omits the failed-format v9 series, and is not an all-campaign success rate. B/C reuse the same prompts and the paired processes do not double the sample.

V11 selects H0 = **13:6, 12:6, 13:14, 14:12, 11:9, 13:15** on calibration. V12 fixes these six coordinates before any target inference. Its Primary and Vocabulary accuracies are **127/128** and **62/64**. H0’s context-swap mediated effects are **1.1514** and **1.3916 logits**, with **92.1875%** and **100%** positive direction. This is transfer of H0 through a previously target-selected relay, not proof that all twelve original coordinates transferred unchanged or that heads are homologous.

## Circuit and verification architecture

```text
Question-span context-dependent computation (H0; v11 discovery / v12 transfer)
    13:6  12:6  13:14  14:12  11:9  13:15
                 | outputs at [query, T−1]; residual input at T clamped clean
                 v
          U value vectors on the question span
                 | original U Q/K; anchored U output at T
                 v
    Qwen3-0.6B U: 15:4 16:4 17:1     (v7 / v11)
    Qwen3-1.7B U: 16:4 17:1 18:7     (v10 / v12)
                 | controlled pure Q_T replay; original D K/V
                 v
    Both scales D: 19:5 21:13 22:9
                 | attention-weighted V at bound color positions
                 v
             residual stream -> two-token logits
```

```text
1. Scientist proposals       Codex CLI + gpt-6-luna; strict contracts
2. Prospective registration  prompts, tokens, pairings, gates, source SHA-256
3. Controlled execution     planted controls; one MPS FIFO; two fresh processes
4. Independent verification NumPy reconstruction + frozen decisions + exact arrays
5. Reference-bound review   literal metric IDs -> canonical numbers; advisory prose
```

“Zero trust” means no scientist narrative can override executable criteria. Hashes establish consistency with retained bytes, not scientific truth or independently certified wall-clock time. Registrations were local Git commits. See [the numerical and provenance contracts](docs/ARCHITECTURE.md).

## Source-only installation and verification

This private repository contains project-owned source and documentation through **v12-cross-scale-3hop**. It contains no corpus, model assets, frozen experiment artifacts, historical source archives, or private Git history. Numerical results above are reported research findings; the underlying evidence is omitted. See [OMISSIONS.md](OMISSIONS.md).

```sh
uv sync --locked
uv run python scripts/verify_source_only.py
uv run pytest
uv run --no-sync ruff check
```

The package verifier checks the source manifest, allowed file paths and common private-path/credential patterns. Tests requiring omitted evidence are explicitly skipped. Available planted controls, independent graders and CPU tiny-model tests still execute. Neither these tests nor the package verifier assert a scientific campaign PASS.

`scripts/verify_all_audits.py` and the versioned auditors are included for inspection. Full campaign verification requires the separately retained evidence, historical source versions, registration proofs and pinned checkpoints. The master exits with **BLOCKED** when that evidence is absent. Do not invoke historical runners as a quick start: they require archived inputs and use fixed experiment output namespaces.

## Models and portable paths

| Model | Pinned revision |
|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

The exported v11/v12 runners use `QWEN_V9_WORKSPACE`, `QWEN_V10_WORKSPACE` and `QWEN_V11_WORKSPACE` for prerequisite workspaces (default: this root). `QWEN17_CHECKPOINT` selects a local pinned 1.7B snapshot; the default is the repository-relative Hugging Face cache layout. No model download is automatic in these adapters. These are disclosed source-only portability changes, not replacements for the frozen registered source bytes.

## Navigation and licensing

- [Research narrative](docs/BLOG_POST.md), [manuscript draft](docs/PAPER.md), and [results ledger](docs/RESULTS_INDEX.md)
- [Numerical contracts and replay limitations](docs/ARCHITECTURE.md)
- [Agent instructions](AGENTS.md), also available as [CLAUDE.md](CLAUDE.md)
- `src/poc/`: versioned interventions, controls, templates and reviewers through v12
- `scripts/`: historical runners, independent auditors and package verification
- `tests/`: planted controls, exact-boundary tests, CPU and opt-in Metal regressions

Project-owned code is Apache-2.0 under [LICENSE_SCOPE.md](LICENSE_SCOPE.md); see [LICENSE](LICENSE) and [NOTICE](NOTICE). The research narrative/manuscript documents are included for review, without extending the software license to excluded research material. Corpus and protected evidence remain private. This update keeps the repository private and does not publish a GitHub release.
