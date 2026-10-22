"""Audit the controlled rerun, preserving construction failures as inconclusive."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/targeted-real-v1"
DIAG = ROOT / "data/targeted-diagnosis"
BANDS = ["OOD_POSITIVE", "SEMANTIC_STRIPPING", "INHIBITORY_POLE"]


def counts(e):
    return [sum(r["passed"] for r in e["results"] if r["case"]["battery"] == b) for b in BANDS]


def main(directory="targeted-real-v3"):
    NEW = ROOT / "data" / directory
    old = json.loads((OLD / "results.json").read_text())
    diagnosis = json.loads((DIAG / "audit.json").read_text())
    replay = json.loads((DIAG / "replay.json").read_text())
    controls = json.loads((DIAG / "controls.json").read_text())
    delimiter = json.loads((DIAG / "delimiter-control-results.json").read_text())
    order_path = DIAG / "delimiter-order-results.json"
    order = json.loads(order_path.read_text()) if order_path.exists() else None
    order_description = "The order-swap diagnostic has not been measured."
    if order:
        order_rows = {r["case"]["id"]: r for r in order["evaluation"]["results"]}
        span = order["quantiles"]["p95"] - order["quantiles"]["p05"]
        drops = [
            (order_rows[r["case"]["paired_positive_id"]]["activation"] - r["activation"]) / span
            for r in order_rows.values()
            if r["case"]["battery"] == "SEMANTIC_STRIPPING"
        ]
        medians = {
            b: float(
                np.median(
                    [r["activation"] for r in order_rows.values() if r["case"]["battery"] == b]
                )
            )
            for b in BANDS
        }
        order_description = (
            f"The exact-token-multiset order control scored {counts(order['evaluation'])} of 15 "
            f"(positive / stripping / opposing). Median positive activation was {medians[BANDS[0]]:.6f}; "
            f"median stripped activation was {medians[BANDS[1]]:.6f}. The median paired drop was "
            f"{np.median(drops):.1%} of the calibration span: the malformed sequence activated "
            "more strongly than the correctly nested one, opposite to the closure prediction. "
            "This falsifies that directional prediction in this registered code family. It does "
            "not identify the head's complete function. Neither diagnostic promotes a unit."
        )
    plan = json.loads((NEW / "plan.json").read_text())
    db = sqlite3.connect(f"file:{NEW}/circuits.sqlite?mode=ro", uri=True)
    run, manifest = db.execute(
        "SELECT id,manifest FROM runs ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    manifest = json.loads(manifest)
    assert manifest["settings"]["thresholds"] == old["run_manifest"]["settings"]["thresholds"]
    cal = duckdb.connect(str(NEW / "activations.duckdb"), read_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(
            ROOT
            / "data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
        ),
        local_files_only=True,
    )
    events = []
    for kind, raw in db.execute("SELECT kind,detail FROM events WHERE run=? ORDER BY id", [run]):
        try:
            detail = json.loads(raw)
        except ValueError:
            detail = raw
        events.append({"kind": kind, "detail": detail})
    approved = {
        json.dumps(e["detail"]["suite"], sort_keys=True)
        for e in events
        if e["kind"] == "suite_review" and all(c["valid"] for c in e["detail"]["review"]["cases"])
    }
    states = {
        u: (s, d)
        for u, s, d in db.execute("SELECT unit,state,detail FROM states WHERE run=?", [run])
    }
    assert len(states) == 12 and not any(
        s in {"exploring", "falsifying", "confirming"} for s, d in states.values()
    ), "Run is still active"
    units = []
    all_lengths = []
    paired = 0
    for previous in old["units"]:
        key = previous["unit"]
        q = previous["quantiles"]
        state, detail = states[key]
        actual = cal.execute(
            "SELECT count,mean,p001,p05,p50,p80,p95,p999 FROM quantiles WHERE unit=?", [key]
        ).fetchone()
        assert list(actual) == [
            q[n] for n in ["count", "mean", "p001", "p05", "p50", "p80", "p95", "p999"]
        ]
        attempts = []
        seen = set()
        for strike, h, e in db.execute(
            "SELECT strike,hypothesis,evaluation FROM attempts WHERE run=? AND unit=? ORDER BY id",
            [run, key],
        ):
            h, e = json.loads(h), json.loads(e)
            suite = {"cases": [r["case"] for r in e["results"]]}
            assert json.dumps(suite, sort_keys=True) in approved, (key, "No semantic approval")
            by_id = {r["case"]["id"]: r for r in e["results"]}
            toks = {
                cid: tokenizer.encode(r["case"]["text"], add_special_tokens=False)
                for cid, r in by_id.items()
            }
            texts = [r["case"]["text"] for r in e["results"]]
            assert len(set(texts)) == 45 and not seen.intersection(texts)
            seen.update(texts)
            for r in e["results"]:
                case, value = r["case"], r["activation"]
                ids = toks[case["id"]]
                assert 0 < len(ids) <= 128
                all_lengths.append(len(ids))
                passed = (
                    value >= q["p95"] if case["battery"] == "OOD_POSITIVE" else value <= q["p05"]
                )
                if case["battery"] == "SEMANTIC_STRIPPING":
                    partner = by_id[case["paired_positive_id"]]
                    other = toks[case["paired_positive_id"]]
                    assert ids[-1] == other[-1] and len(ids) == len(other)
                    paired += 1
                    passed = passed and partner["activation"] - value >= 0.25 * (
                        q["p95"] - q["p05"]
                    )
                assert passed == r["passed"]
            assert all(sum(r["case"]["battery"] == b for r in e["results"]) == 15 for b in BANDS)
            assert e["passed"] == all(n >= 12 for n in counts(e))
            attempts.append({"strike": strike, "hypothesis": h, "evaluation": e})
        screens = [
            e["detail"]
            for e in events
            if e["kind"] == "hypothesis_screen" and e["detail"]["unit"] == key
        ]
        units.append(
            {
                "unit": key,
                "state": state,
                "detail": detail,
                "attempts": attempts,
                "screens": screens,
            }
        )
    cal.close()
    db.close()
    summary = dict(Counter(u["state"] for u in units))
    measured = sum(len(a["evaluation"]["results"]) for u in units for a in u["attempts"])
    verified = [u for u in units if u["state"] == "verified"]
    batch_reviews = [e["detail"] for e in events if e["kind"] == "batch_review"]
    reviewed_cases = [c for b in batch_reviews for c in b["review"]["cases"]]
    screen_best = [max(s["correct"]) for u in units for s in u["screens"]]
    combined = {
        "run_id": run,
        "manifest": manifest,
        "plan": plan,
        "diagnosis": diagnosis,
        "replay": replay,
        "context_controls": controls,
        "delimiter_identity_diagnostic": delimiter,
        "delimiter_order_diagnostic": order,
        "summary": summary,
        "units": units,
        "events": events,
        "audit": {
            "measured_cases": measured,
            "matched_pairs": paired,
            "measured_probe_median_tokens": float(np.median(all_lengths)) if all_lengths else None,
            "same_numeric_thresholds": True,
            "same_quantiles": True,
            "all_measured_suites_semantically_approved": True,
        },
    }
    (NEW / "results.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    n_attempts = sum(len(u["attempts"]) for u in units)
    lines = [
        "# Why the first sweep failed, and the controlled rerun",
        "",
        f"**Controlled rerun: {len(verified)} verified units; {n_attempts} measured suites and {measured:,} probes.** Run `{run}`.",
        "",
        f"Final states: {summary}. Construction failures are inconclusive, not measured falsifications.",
        "No full valid suite means the rerun did not reach numerical falsification; it must not be presented as twelve newly falsified hypotheses."
        if n_attempts == 0
        else "Only completed, approved suites contribute scientific pass/fail evidence.",
        "",
        "## Diagnosis",
        "",
        "| Question | Evidence | Conclusion |",
        "| --- | --- | --- |",
        "| Measurement bug? | 432 saved calibration prefixes replayed exactly on Metal; 0 tail-threshold failures. | No replay error detected. The numerical gates are attainable. |",
        "| Confounded test pairs? | 335/540 original stripping pairs (62.0%) changed the measured token. Median synthetic length was 13 tokens versus 49 in sampled calibration examples. | The original tests did not adequately control token identity, position, or context. |",
        "| Unsupported structural stories? | Original rules repeatedly invoked completion/closure. For head 3:3, 11/12 high-tail examples remain high after prefix shuffling. For MLP 27:564, 11/12 remain high with the target token alone. | Specific controls contradict an automatic syntax-first interpretation for these samples. These are diagnostics, not universal feature labels. |",
        "| Is stripping P05 the only bottleneck? | 108/540 stripping pairs drop by at least 25% of the span, but only 47 reach P05. Yet all 36 original candidates fail positive synthesis independently; its maximum is 11/15. | Requiring an opposite extreme is stronger than feature removal. Relaxing only stripping cannot rescue any original candidate. |",
        "",
        "The aggregate controls retain the same tail for 11/144 positive and 12/144 inhibitory observations using the target token alone; after prefix shuffling, 62/144 and 63/144 remain in the same tail. Token order and context matter differently across units. Shuffling is off-distribution, so these observations constrain explanations rather than identify complete computations.",
        "",
        "## Executable delimiter controls",
        "",
        "A separately registered diagnostic tested head `attention:3:3` on 15 parser-checked code pairs. Only one opening delimiter changed; final token identity and position were identical. It scored **15/15 positives, 0/15 stripping, and 0/15 opposing cases** under the unchanged gates. Nevertheless, every stripping case exceeded the 25%-span drop requirement: the median drop was **96.0% of the calibration span**. The stripped median was −0.004946, above P05 = −0.022808; the positive median was 0.022451, above P95 = 0.005729.",
        "",
        "Thus the strict lower-tail gate can reject a substantial controlled activation drop. This is evidence of sensitivity to the opener edit, not yet proof of a nesting computation: that edit also changes delimiter identity. The diagnostic is templated, code-only, and not a survivor or independent holdout. A further registered order-swap test preserves the exact token multiset as well as target identity and position.",
        "",
        order_description,
        "",
        "## Corrections and frozen standards",
        "",
        "The same checkpoint, corpus, 12 units, calibration quantiles, three measured discovery strikes, and fresh 45-case confirmation requirement were retained. **No numerical gate was lowered:** P95 for positives; P05 for stripping/opposing cases; at least 25% span drop for paired stripping; at least 12/15 per battery. These remain a strict test of bipolar selectivity, not a universal definition of interpretability.",
        "",
        "Explorers now receive measured token-only and shuffled-context controls. Six examples per band are reserved for a blinded ranking screen with opaque shuffled IDs; candidate selection uses predictive accuracy rather than self-confidence. The screen is repeated discovery selection within the same corpus, not independent confirmation.",
        "",
        "The generator receives calibration prefixes and actual target tokens. Every stripping pair must preserve the final tokenizer ID and its exact position. Evidence-based minimum context lengths prevent reducing all long-context claims to tiny probes. A separate activation-blind reviewer must accept all 45 cases before measurement. Rejections trigger bounded construction repair without revealing activations. Hypotheses, rankings, rejected suites, reviews and measurements are retained.",
        "",
        "The intermediate 45-case generation run produced no acceptable full suites and remains recorded as construction-inconclusive. The final generator constructs five nine-case batches, checks each with the executor-bound CPU tokenizer and a separate reviewer, and assembles all 45 before grading. The case count, paired controls, context floors, numerical thresholds and confirmation requirements were not relaxed. The tokenizer helper cannot execute the model. See [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) for the per-session tool configuration mechanism.",
        "",
        "## Results by unit",
        "",
        "| Unit | Final state | Measured suites | Best valid positive / stripping / opposing | Best ranking screen |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for u in units:
        attempts = [a for a in u["attempts"] if a["evaluation"]["stage"] == "discovery"]
        best = (
            max(
                attempts, key=lambda a: (min(counts(a["evaluation"])), sum(counts(a["evaluation"])))
            )
            if attempts
            else None
        )
        score = (
            " / ".join(f"{n}/15" for n in counts(best["evaluation"]))
            if best
            else "No valid measured suite"
        )
        screen = max((max(s["correct"]) / s["total"] for s in u["screens"]), default=None)
        label = "inconclusive: output/test construction" if u["state"] == "error" else u["state"]
        lines.append(
            f"| `{u['unit']}` | {label} | {len(u['attempts'])} | {score} | {screen:.0%} |"
            if screen is not None
            else f"| `{u['unit']}` | {label} | {len(u['attempts'])} | {score} | — |"
        )
    lines += [
        "",
        "“Best” uses highest worst-battery count, then highest total. No cases are omitted from a measured suite. A rejected construction receives no activation score and cannot count as either a win or a scientific failure.",
        "",
        f"Audit: {paired} measured pairs preserve target identity and position; all measured pass/fail decisions were independently recomputed. Median valid probe length: {combined['audit']['measured_probe_median_tokens']}. Every measured suite has a matching all-valid semantic review."
        if measured
        else "Audit: the checkpoint configuration, numerical thresholds, and all 12 calibration tables match the original run. There are no completed suites to audit numerically in this rerun; only the separately labeled diagnostics contain new activation measurements.",
        "",
        "## Candidate details",
        "",
        f"The best candidate in {sum(n <= 6 for n in screen_best)}/{len(screen_best)} screens did no better than the 6/18 constant-band baseline. These small, selected screens are descriptive; they are not significance tests or independent holdouts.",
        "",
        f"The auditor inspected {len(batch_reviews)} batches ({len(reviewed_cases)} case reviews, including repairs); it accepted {sum(c['valid'] for c in reviewed_cases)} individual cases. Individual approval is not a complete, balanced evaluation and cannot promote a hypothesis. Rejected cases often left the hypothesized local relation intact or measured a different final token than their rationale described.",
        "",
    ]
    for u in units:
        lines += [f"### {u['unit']} — {u['state']}", "", u["detail"], ""]
        for a in u["attempts"]:
            lines += [
                f"Attempt {a['strike']} ({a['evaluation']['stage']}): {a['hypothesis']['structural_rule']}",
                "",
                f"Passing counts: {counts(a['evaluation'])} of 15 per battery.",
                "",
            ]
        if not u["attempts"] and u["screens"]:
            s = u["screens"][-1]
            i = min(range(3), key=lambda i: (-s["correct"][i], s["candidates"][i]["lens"]))
            lines += [
                "Unmeasured candidate from the final screen:",
                s["candidates"][i]["structural_rule"],
                "",
            ]
    usage = [e["detail"] for e in events if e["kind"] == "usage"]
    if usage:
        inp = sum(x["input_tokens"] for x in usage)
        cache = sum(x["cached_tokens"] for x in usage)
        lines += [
            "## Execution",
            "",
            f"Recorded Codex usage: {inp:,} input tokens, {cache / inp:.1%} cached; {sum(x['output_tokens'] for x in usage):,} output tokens; {sum(x.get('tokenizer_calls', 0) for x in usage)} tokenizer-tool calls. No dollar estimate is inferred from CLI token counters.",
            "",
        ]
    lines += [
        "## Limits and evidence",
        "",
        "The remaining implementation limitation is that the explorer contract still requires a hypothesis even when its own text says no usable structural rule exists. There is no structured abstention route, and the generator/reviewer loop cannot reliably operationalize these weak rules. The repaired controls prevent false claims but have not made the autonomous discovery workflow scientifically productive in this pilot.",
        "",
        "No negative result here proves that a unit is uninterpretable. The scalar head mean loses vector information; the bipolar criteria exclude useful unipolar features; the corpus and units are limited. A same-target control tests contextual structural claims; it cannot validate a purely lexical feature whose defining manipulation necessarily changes the target token. The semantic auditor is another Luna call, not human ground truth; some reviewer rationales were demonstrably mistaken, although the corresponding suites also contained genuine invalid manipulations. Its rejection can itself be wrong, so construction failures remain inconclusive. Multiple corrections changed together in the rerun; score differences cannot identify the isolated causal contribution of each correction.",
        "",
        f"[Frozen rerun plan](../data/{NEW.name}/plan.json) · [Full rerun evidence](../data/{NEW.name}/results.json) · [Original results](targeted-real-v1.md) · [Original test audit](../data/targeted-diagnosis/audit.json) · [Exact replay](../data/targeted-diagnosis/replay.json) · [Context controls](../data/targeted-diagnosis/controls.json) · [Delimiter identity control](../data/targeted-diagnosis/delimiter-control-results.json) · [Delimiter order control](../data/targeted-diagnosis/delimiter-order-results.json)",
    ]
    path = ROOT / f"reports/{NEW.name}.md"
    path.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"run": run, "states": summary, "measured_cases": measured, "report": str(path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", default="targeted-real-v3", choices=["targeted-real-v2", "targeted-real-v3"]
    )
    main(parser.parse_args().directory)
