"""Render measured results without promoting failed or untested hypotheses."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/targeted-real-v1"
REPORT = ROOT / "reports/targeted-real-v1.md"
BATTERIES = ["OOD_POSITIVE", "SEMANTIC_STRIPPING", "INHIBITORY_POLE"]


def counts(evaluation):
    return [
        sum(r["passed"] for r in evaluation["results"] if r["case"]["battery"] == b)
        for b in BATTERIES
    ]


def main():
    db = sqlite3.connect(f"file:{DATA / 'circuits.sqlite'}?mode=ro", uri=True)
    run, raw_manifest = db.execute(
        "SELECT id,manifest FROM runs ORDER BY started DESC, rowid DESC LIMIT 1"
    ).fetchone()
    manifest = json.loads(raw_manifest)
    corpus = json.loads((DATA / "corpus-manifest.json").read_text())
    plan = json.loads((DATA / "plan.json").read_text())
    token_stats = json.loads((DATA / "corpus-token-stats.json").read_text())
    token_count = sum(x["tokens_used"] for x in token_stats.values())
    calibration = duckdb.connect(str(DATA / "activations.duckdb"), read_only=True)
    cursor = calibration.execute("SELECT * FROM quantiles")
    columns = [c[0] for c in cursor.description]
    quantiles = {row[0]: dict(zip(columns[1:], row[1:])) for row in cursor.fetchall()}
    calibration.close()
    states = {
        u: (s, d)
        for u, s, d in db.execute("SELECT unit,state,detail FROM states WHERE run=?", [run])
    }
    units = []
    for unit in plan["units"]:
        attempts = [
            {"strike": strike, "hypothesis": json.loads(h), "evaluation": json.loads(e)}
            for strike, h, e in db.execute(
                "SELECT strike,hypothesis,evaluation FROM attempts WHERE run=? AND unit=? ORDER BY id",
                [run, unit],
            )
        ]
        state, detail = states.get(unit, ("pending", ""))
        # Recompute all case decisions from stored measurements and frozen thresholds.
        q = quantiles[unit]
        assert q["count"] == token_count, (unit, "calibration count mismatch")
        seen_texts = set()
        for a in attempts:
            e = a["evaluation"]
            texts = [r["case"]["text"] for r in e["results"]]
            assert len(texts) == len(set(texts)) == 45
            assert not seen_texts.intersection(texts), (unit, "reused measured text")
            seen_texts.update(texts)
            assert all(
                sum(r["case"]["battery"] == b for r in e["results"]) == 15 for b in BATTERIES
            )
            measured = {r["case"]["id"]: r["activation"] for r in e["results"]}
            for r in e["results"]:
                case, value = r["case"], r["activation"]
                passed = (
                    value >= q["p95"] if case["battery"] == "OOD_POSITIVE" else value <= q["p05"]
                )
                if case["battery"] == "SEMANTIC_STRIPPING":
                    passed = passed and measured[case["paired_positive_id"]] - value >= 0.25 * (
                        q["p95"] - q["p05"]
                    )
                assert passed == r["passed"], (unit, a["strike"], case["id"])
            assert e["passed"] == all(n >= 12 for n in counts(e))
        units.append(
            {"unit": unit, "state": state, "detail": detail, "quantiles": q, "attempts": attempts}
        )
    summary = dict(Counter(u["state"] for u in units))
    verified = [u for u in units if u["state"] == "verified"]
    stages = Counter(a["evaluation"]["stage"] for u in units for a in u["attempts"])
    measured_cases = sum(len(a["evaluation"]["results"]) for u in units for a in u["attempts"])
    circuit_count = db.execute("SELECT count(*) FROM circuits WHERE run=?", [run]).fetchone()[0]
    events = [
        {"kind": k, "detail": json.loads(d) if k in ("usage", "completion") else d}
        for k, d in db.execute("SELECT kind,detail FROM events WHERE run=? ORDER BY id", [run])
    ]
    records = {
        "run_id": run,
        "summary": summary,
        "measured_stages": dict(stages),
        "measured_cases": measured_cases,
        "circuit_count": circuit_count,
        "corpus": corpus,
        "corpus_token_stats": token_stats,
        "run_manifest": manifest,
        "plan": plan,
        "units": units,
        "events": events,
    }
    notes_path = DATA / "execution-notes.json"
    if notes_path.exists():
        notes = json.loads(notes_path.read_text())
        original = db.execute(
            "SELECT id,unit,strike,hypothesis,evaluation FROM attempts WHERE run=? ORDER BY id LIMIT ?",
            [run, notes["pre_resume_attempts_count"]],
        ).fetchall()
        assert (
            hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
            == notes["pre_resume_attempts_sha256"]
        )
        records["execution_notes"] = notes
    (DATA / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    lines = [
        "# Targeted Qwen3 sweep on real text",
        "",
        f"Run: `{run}`. Model revision: `{manifest['calibration']['resolved_revision']}`.",
        "",
        f"**{len(verified)} of {len(units)} units have hypotheses that passed discovery and fresh synthetic holdout.**",
        "",
        "Status counts: " + ", ".join(f"{n} {s}" for s, n in sorted(summary.items())) + ".",
        "",
        f"Measured {stages['discovery']} discovery hypotheses and {stages['holdout']} holdout suites: **{measured_cases:,} synthetic test cases**. Verified circuit records: {circuit_count}. Holdout testing and causal tracing are gated on passing discovery and confirmation respectively.",
        "",
        f"Calibration corpus: {corpus['documents']} real documents/excerpts, 128 each from encyclopedia, news, reviews, scientific abstracts, Python code, and literature. There are {token_count:,} token observations per unit after the 128-token document cap. Documents are balanced by domain; token counts are not. English/code coverage only.",
        "",
        "Units were chosen with seed 20261019 before calibration: two MLP coordinates and one query head in each of layers 3, 11, 19 and 27 (zero-based). Thresholds were not tuned to the results.",
        "",
        "Pass rule: at least **12/15 in each battery**; positives ≥ calibration P95; removals/opposing-pole tests ≤ P05; paired removals also need a ≥25% P95–P05-span drop. At most three discovery attempts; a passing discovery requires a new 45-case holdout, with no holdout-driven revisions.",
        "",
        "| Unit | Status | Measured attempts | Best discovery: OOD / stripping / opposing | Holdout |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for u in units:
        discoveries = [a for a in u["attempts"] if a["evaluation"]["stage"] == "discovery"]
        best = (
            max(
                discoveries,
                key=lambda a: (min(counts(a["evaluation"])), sum(counts(a["evaluation"]))),
            )
            if discoveries
            else None
        )
        holds = [a for a in u["attempts"] if a["evaluation"]["stage"] == "holdout"]
        score = " / ".join(f"{n}/15" for n in counts(best["evaluation"])) if best else "—"
        hold = (
            " / ".join(f"{n}/15" for n in counts(holds[-1]["evaluation"]))
            if holds
            else "Not reached"
        )
        lines.append(f"| `{u['unit']}` | {u['state']} | {len(u['attempts'])} | {score} | {hold} |")
    lines.extend(
        [
            "",
            "“Best” means the attempt with the highest worst-battery count, then highest total count. This is a descriptive ranking, not a changed pass criterion. All attempts are listed below.",
            "",
            "## Surviving hypotheses",
            "",
        ]
    )
    if not verified:
        lines.append(
            "No surviving hypotheses in this run. Failed hypotheses are not evidence that the units have no interpretable function."
        )
    else:
        for u in verified:
            last = u["attempts"][-1]
            lines.extend(
                [
                    f"### {u['unit']}",
                    "",
                    last["hypothesis"]["structural_rule"],
                    "",
                    f"Holdout counts: {counts(last['evaluation'])}; R²: {last['evaluation']['r_squared']}.",
                    "",
                ]
            )
    usage = [e["detail"] for e in events if e["kind"] == "usage"]
    if usage:
        inp = sum(x["input_tokens"] for x in usage)
        cached = sum(x["cached_tokens"] for x in usage)
        out = sum(x["output_tokens"] for x in usage)
        lines.extend(
            [
                "",
                "## Execution",
                "",
                f"Measured Codex usage: {inp:,} input tokens ({cached / inp:.1%} cached) and {out:,} output tokens across recorded run segments. These are account token counters, not a dollar cost estimate.",
            ]
        )
    if notes_path.exists():
        lines.extend(
            [
                "",
                "The first execution segment encountered invalid scientist outputs (assignment/evidence identifiers or repeated test strings). Contract validation now sends those assignment/evidence errors back through the existing bounded repair loop. The same run was resumed with every measured attempt preserved; no scientific thresholds, role prompts, unit selection, or strike budgets changed. Original errors, source hashes, and recovery steps are retained in [execution-notes.json](../data/targeted-real-v1/execution-notes.json).",
            ]
        )
    lines.extend(["", "## Full hypothesis history", ""])
    for u in units:
        lines.extend([f"### {u['unit']} — {u['state']}", "", u["detail"], ""])
        for a in u["attempts"]:
            h, e = a["hypothesis"], a["evaluation"]
            lines.extend(
                [
                    f"**Attempt {a['strike']}, {e['stage']} ({h['lens']})**",
                    "",
                    h["structural_rule"],
                    "",
                    f"Pass counts (OOD / stripping / opposing): {' / '.join(map(str, counts(e)))} out of 15 each. R²: {e['r_squared']}.",
                    "",
                    f"Competing explanation: {h['alternative_explanation']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Scope and reproducibility",
            "",
            "The corpus is real text; falsification and confirmation strings are generated by Luna. Passing this protocol would support a candidate on those tests, not prove independent real-world generalization or a causal circuit. A lower-tail head mean is a signed coordinate summary, not evidence of inhibition by itself.",
            "",
            "Failures reject a proposed hypothesis together with its generated test operationalization under the frozen thresholds. They do not prove the abstract rule false in every context. Generated cases can change the final token or place it after the intended feature; this protocol does not independently certify those semantic judgments. Synthetic probes measure their final token, whereas calibration covers all unpadded token positions.",
            "",
            "The report independently recomputed every pass/fail decision from the stored activations and calibration thresholds. It did not regenerate or discard measured cases.",
            "",
            "Every measured case, activation, battery result, hypothesis and settings manifest is in [results.json](../data/targeted-real-v1/results.json). The SQLite and DuckDB databases retain the underlying evidence. [Frozen plan](../data/targeted-real-v1/plan.json). [Corpus provenance](../data/targeted-real-v1/corpus-manifest.json).",
            "",
            "The dataset viewer served unpinned dataset revisions. The sampled source row identities, downloaded text, and SHA-256 hashes preserve exactly what was used. WikiText includes corpus-specific spacing; short prefixes and this limited unit selection constrain the conclusions.",
            "",
            "Sources:",
            "",
        ]
    )
    for source in corpus["sources"]:
        lines.append(
            f"- [{source['dataset']}]({source['url']}): `{source['config']}/{source['split']}`, field `{source['field']}`."
        )
    lines.append(
        "- [Project Gutenberg](https://www.gutenberg.org/): 16 naturally occurring paragraphs per book, IDs "
        + ", ".join(map(str, corpus["literature_books"]))
        + "."
    )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "run": run,
                "states": summary,
                "report": str(REPORT),
                "total_measured_cases": sum(
                    len(a["evaluation"]["results"]) for u in units for a in u["attempts"]
                ),
            },
            indent=2,
        )
    )
    db.close()


if __name__ == "__main__":
    main()
