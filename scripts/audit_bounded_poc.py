"""Independent arithmetic audit of saved arrays; never changes experiment decisions."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v1"


def read(name):
    return json.loads((OUT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fitted(x, y):
    a, b = np.mean(x[y == 1], axis=0), np.mean(x[y == -1], axis=0)
    w = a - b
    length = float(np.sqrt(np.sum(w * w)))
    if length:
        w /= length
    return w, float(np.dot((a + b) / 2, w))


def main():
    manifest = read("manifest.json")
    for name, expected in manifest["sha256"].items():
        assert digest(ROOT / name) == expected, name
    original = read("aborted-start-1/manifest.json")
    assert original["limits"] == manifest["limits"]
    for name in ("reports/bounded-poc-protocol.md", "data/bounded-poc-v1/dataset.json"):
        assert original["sha256"][name] == manifest["sha256"][name]
    assert (
        digest(OUT / "discovery-baseline.npz")
        == read("execution-repair.json")["retained_baseline_sha256"]
    )
    limits = manifest["limits"]
    data = read("dataset.json")["rows"]
    train = [r for r in data if r["split"] == "discovery"]
    held = [r for r in data if r["split"] == "confirmation"]
    labels = np.array([r["label"] for r in held])
    train_labels = np.array([r["label"] for r in train])
    heads = [f"{layer}:{h}" for layer in manifest["layers"] for h in range(16)]
    selected = read("discovery.json")
    baseline = np.load(OUT / "discovery-baseline.npz")
    scan = np.load(OUT / "discovery-patches.npz")["logits"]
    train_margin = baseline["logits"][:, 0] - baseline["logits"][:, 1]
    scores = []
    for i, c in enumerate(selected["all_candidates"]):
        assert c["head"] == heads[i]
        x = baseline["heads"][:, i]
        w, threshold = fitted(x, train_labels)
        np.testing.assert_allclose(w, c["readout"]["direction"], atol=1e-12)
        np.testing.assert_allclose(threshold, c["readout"]["threshold"], atol=1e-12)
        np.testing.assert_allclose(x.mean(0), c["readout"]["mean"], atol=1e-12)
        cv = []
        for group in sorted({r["group"] for r in train}):
            test = np.array([r["group"] == group for r in train])
            direction, cut = fitted(x[~test], train_labels[~test])
            cv.extend(np.sign(x[test] @ direction - cut) == train_labels[test])
        accuracy = float(np.mean(cv))
        effect = float(
            np.median(train_labels[:16] * (train_margin[:16] - (scan[i, :, 0] - scan[i, :, 1])))
        )
        assert accuracy == c["cv_accuracy"]
        assert np.isclose(effect, c["discovery_patch_median"])
        if accuracy >= limits["discovery_cv"]:
            scores.append((c["head"], effect))
    ranked = sorted(scores, key=lambda x: (-x[1], *map(int, x[0].split(":"))))[:3]
    assert [h for h, _ in ranked] == [c["head"] for c in selected["selected"]]
    result = read("results.json")
    audits = []
    behaviors = []
    correct_predictions = {}
    for rep in (1, 2):
        summary = read(f"confirmation-{rep}.json")
        b = np.load(OUT / f"confirmation-{rep}-baseline.npz")
        assert all(np.isfinite(b[k]).all() for k in b.files)
        margins = b["logits"][:, 0] - b["logits"][:, 1]
        correct_predictions[rep] = labels * margins > 0
        behavior_count = int(np.sum(correct_predictions[rep]))
        assert behavior_count / len(held) == summary["model_accuracy"]
        controls = np.load(OUT / f"confirmation-{rep}-controls.npz")
        ids = {r["id"]: i for i, r in enumerate(held)}
        donor_idx = np.array([ids[r["opposite_id"]] for r in held])
        residual_error = float(np.max(np.abs(controls["residual_patch"] - b["logits"][donor_idx])))
        sham_error = float(
            np.max(np.abs(controls["sham_patch"][:, 0] - controls["sham_patch"][:, 1] - margins))
        )
        assert residual_error <= limits["residual_max"]
        assert sham_error <= limits["sham_max"]
        behaviors.append(
            {
                "replicate": rep,
                "correct": behavior_count,
                "total": len(held),
                "required": math.ceil(limits["model_accuracy"] * len(held)),
                "format_correct": {
                    str(fmt): sum(
                        bool(ok)
                        for ok, r in zip(correct_predictions[rep], held)
                        if r["format"] == fmt
                    )
                    for fmt in (2, 3)
                },
                "residual_error": residual_error,
                "sham_error": sham_error,
            }
        )
        group_names = sorted({r["group"] for r in held})
        group_indices = np.array([group_names.index(r["group"]) for r in held])
        for c, chosen in zip(summary["candidates"], selected["selected"], strict=True):
            assert c["head"] == chosen["head"]
            raw = np.load(OUT / c["raw"])
            vector = b["heads"][:, heads.index(c["head"])]
            predicted = np.sign(
                vector @ np.array(chosen["readout"]["direction"]) - chosen["readout"]["threshold"]
            )
            stats = c["evaluation"]["statistics"]
            np.testing.assert_array_equal(predicted, stats["predictions"])
            assert float(np.mean(predicted == labels)) == stats["readout_accuracy"]
            effects = labels * (margins - raw["opposite"])
            mean_effects = labels * (margins - raw["mean"])
            specificity = labels * (raw["same_answer"] - raw["opposite"])
            gap = np.median(labels * (margins - margins[donor_idx]))
            for key, observed in {
                "patch_median": np.median(effects),
                "patch_fraction": np.median(effects) / gap,
                "mean_median": np.median(mean_effects),
                "specificity_median": np.median(specificity),
                "patch_direction_rate": np.mean(effects > 0),
                "mean_direction_rate": np.mean(mean_effects > 0),
            }.items():
                assert np.isclose(observed, stats[key], atol=1e-12), (c["head"], key)
            group_effects = [np.mean(effects[group_indices == i]) for i in range(32)]
            positives = sum(v > 0 for v in group_effects)
            p_sign = sum(math.comb(32, k) for k in range(positives, 33)) / 2**32
            assert p_sign == stats["causal_group_sign_p"]
            rng = np.random.default_rng(20261020)
            signs = rng.choice([-1, 1], size=(4095, 32))
            null_accuracy = np.mean(
                predicted[None, :] == labels[None, :] * signs[:, group_indices], axis=1
            )
            p_prediction = (1 + np.sum(null_accuracy >= np.mean(predicted == labels))) / 4096
            assert p_prediction == stats["readout_label_flip_p"]
            checks = {
                key: stats[key] >= limits[key]
                for key in (
                    "readout_accuracy",
                    "baseline_advantage",
                    "patch_direction_rate",
                    "patch_median",
                    "patch_fraction",
                    "mean_direction_rate",
                    "mean_median",
                    "specificity_median",
                )
            }
            subgroups = {
                f"{key}={v}": float(
                    np.mean([p == r["label"] for p, r in zip(predicted, held) if r[key] == v])
                )
                for key in ("format", "label")
                for v in sorted({r[key] for r in held})
            }
            assert subgroups == stats["subgroup_accuracy"]
            assert stats["baseline_advantage"] == stats["readout_accuracy"] - 0.5
            checks.update(
                subgroups=min(subgroups.values()) >= limits["subgroup_accuracy"],
                causal_significance=p_sign <= limits["p"],
                predictive_significance=p_prediction <= limits["p"],
                sham=float(np.max(np.abs(raw["sham"] - margins))) <= limits["sham_max"],
            )
            assert checks == c["evaluation"]["checks"]
            assert all(checks.values()) == c["evaluation"]["passed"]
            audits.append(
                {
                    "replicate": rep,
                    "head": c["head"],
                    "passes_all_head_gates": all(checks.values()),
                    "failed_head_gates": [k for k, v in checks.items() if not v],
                    "correct_readout": int(np.sum(predicted == labels)),
                    "correct_readout_on_model_errors": int(
                        np.sum((predicted == labels) & ~correct_predictions[rep])
                    ),
                    "positive_patch_effects": int(np.sum(effects > 0)),
                    "positive_mean_effects": int(np.sum(mean_effects > 0)),
                    "statistics": {k: v for k, v in stats.items() if k != "predictions"},
                }
            )
    baseline1 = np.load(OUT / "confirmation-1-baseline.npz")
    baseline2 = np.load(OUT / "confirmation-2-baseline.npz")
    replicated_exactly = all(np.array_equal(baseline1[k], baseline2[k]) for k in baseline1.files)
    for c in result["candidates"]:
        h = c["head"].replace(":", "-")
        a = np.load(OUT / f"confirmation-1-head-{h}.npz")
        b = np.load(OUT / f"confirmation-2-head-{h}.npz")
        replicated_exactly &= all(np.array_equal(a[k], b[k]) for k in a.files)
    assert replicated_exactly
    task_pass = all(b["correct"] >= b["required"] for b in behaviors)
    assert not task_pass and not result["survivors"]
    audit = {
        "all_numeric_checks_match": True,
        "criteria_and_dataset_match_original_registration": True,
        "discovery_selection_recomputed": True,
        "replication_arrays_bit_identical": bool(replicated_exactly),
        "behavior": behaviors,
        "candidates": audits,
        "formal_success": False,
        "reason": "Held-out model task performance 115/128 is below the frozen minimum 116/128.",
        "component_gates_passed_both_runs": [
            h
            for h in heads
            if sum(a["head"] == h and a["passes_all_head_gates"] for a in audits) == 2
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    best = next(a for a in audits if a["head"] == "19:5" and a["replicate"] == 1)
    lines = [
        "# Bounded causal proof of concept",
        "",
        "**Formal outcome: inconclusive; do not scale yet.** Head `19:5` passed every registered "
        "head-level predictive and causal gate in both runs. The experiment missed its global task "
        "performance prerequisite: Qwen answered 115/128 (89.84375%) held-out comparisons correctly, "
        "while the frozen ≥90% rule required 116/128. No rounding, dropped cases, or threshold changes "
        "were used to promote it.",
        "",
        "## What the experiment established",
        "",
        "The candidate explanation is: **at the final query position, head 19:5 transmits information "
        "about the color bound to the requested key and contributes to the model's preference for that "
        "color's next token.** It is a reproducible causal candidate, but does not receive the experiment's "
        "formal verified status because the global prerequisite failed.",
        "",
        "| Head 19:5 measure | Result, in both runs | Frozen requirement |",
        "| --- | ---: | ---: |",
        "| Held-out readout accuracy | 126/128 = 98.4375% | ≥85% |",
        "| Lowest class/format readout accuracy | 96.875% | ≥80% |",
        "| Advantage over registered surface baselines | 48.4375 percentage points | ≥15 points |",
        "| Counterfactual patch direction | 128/128 | ≥80% |",
        "| Median correct-answer margin reduction | 0.950489 logits | ≥0.25 logits |",
        "| Reduction / median clean-counterfactual gap | 18.0307% | ≥10% |",
        "| Mean-ablation direction | 125/128 | ≥70% |",
        "| Median mean-ablation margin reduction | 0.536531 logits | ≥0.10 logits |",
        "| Opposite-answer versus same-answer patch specificity | 0.902844 logits | ≥0.10 logits |",
        "| Group-level causal sign-test p | 2.3283×10⁻¹⁰ | ≤0.05/3 |",
        "| Group-label randomization p | 1/4096 | ≤0.05/3 |",
        "| Self-patch error | 0 | ≤0.05 logits |",
        "| Fresh-process maximum margin difference | 0 | ≤0.15 logits |",
        "",
        "These are changes in the actual red-versus-blue logit comparison, with signs oriented toward "
        "the correct answer. They are not changes in unrestricted generation accuracy. The 18.0% ratio "
        "is a normalized intervention effect, not a claim that the head performs 18% of the computation.",
        "",
        f"As a descriptive follow-up, the frozen head readout was correct on "
        f"{best['correct_readout_on_model_errors']}/13 examples where the model's final two-token preference "
        "was wrong. This was not a selection or promotion criterion.",
        "",
        "## Bounded design and failed candidates",
        "",
        "We used 16 discovery groups (64 prompts) and 32 held-out groups (128 prompts). Every group "
        "crossed both queries with both color assignments. Counterfactuals preserve token multiset, "
        "sequence length, target token, final eight tokens, and query. Held-out keys and two prompt "
        "formats were disjoint from discovery. The answer vocabulary was always red/blue.",
        "",
        "All layer and head indices in this report are zero-based.",
        "",
        "All 64 heads in layers 3, 11, 19, and 27 were screened. A readout was evaluated with "
        "leave-one-group-out validation. The three eligible heads with strongest discovery patch "
        "effects were frozen before any held-out measurement. Selection and every reported gate "
        "were independently recomputed from saved arrays.",
        "",
        "| Candidate | Held-out readout | Head-level result in both runs |",
        "| --- | ---: | --- |",
        "| `27:15` | 85.9375% | Failed relative patch size, mean-ablation direction, and class balance |",
        "| `19:5` | 98.4375% | Passed every head-level gate; global task prerequisite failed |",
        "| `27:13` | 55.46875% | Failed predictive, effect-size, specificity, and other gates |",
        "",
        "The model's discovery task score was 59/64 (92.1875%). The held-out task score was 115/128 "
        "in each fresh process. Token-count, position, last-token, and recent-context baselines have "
        "an exact 50% ceiling on matched pairs; first/last-mentioned-value rules also score 50%. "
        "The executable lookup parser scores 100% as the positive explanatory control. This does "
        "not establish superiority to every whole-prompt or nonlinear baseline.",
        "",
        "## Controls, repeatability, and execution repairs",
        "",
        "The planted toy lookup mechanism passed the same evaluator. A nuisance-only mechanism "
        "and a deliberately reversed explanation failed. These are software controls, not Qwen "
        "discoveries. In the real model, replacing the final normalized residual reproduced the "
        "donor's two-token logits exactly; self-patches changed no margins. Both freshly loaded "
        "confirmation processes produced bit-identical saved baseline and intervention arrays.",
        "",
        "The scientific criteria and dataset were frozen before the first Qwen forward pass. Two "
        "infrastructure failures were retained and repaired: an unsafe CPU conversion during an "
        "asynchronous BF16 copy, and an inferred float64 donor unsupported by Metal. The first failed "
        "before saving a baseline; the second failed before candidate scanning. Its valid baseline "
        "was retained and hash-checked on continuation. Source manifests were renewed for these "
        "repairs; the original criteria, dataset, hypothesis, candidate budget, and selection rules "
        "remained byte-identical. Dedicated Metal regressions passed before continuation.",
        "",
        "The pre-run Luna reviewer incorrectly conflated the overall successful control-suite status "
        "with the nested nuisance unit's correctly false pass status. A saved factual adjudication "
        "checked the serialized values and recomputed the controls. Reviewer prose did not change "
        "the experiment's grading.",
        "",
        "Validation: 44 tests passed with five opt-in tests skipped in the standard run; "
        "both dedicated Metal regressions passed separately, and lint passed. The final Luna "
        "review agreed that the head-level evidence supports a narrow causal candidate while "
        "the missed global prerequisite prevents formal promotion. "
        "[Execution summary](../data/bounded-poc-v1/execution-summary.json) · "
        "[Final scientist review](../data/bounded-poc-v1/review-results.json)",
        "",
        "## Interpretation and stopping decision",
        "",
        "This supports a narrowly scoped causal information-transmission account. It does not show "
        "that head 19:5 itself searches for the key or reads the value directly; it may transmit "
        "information computed upstream. It does not identify a complete circuit, prove monosemanticity, "
        "or establish generalization to natural corpora or new value vocabularies. Measured donor "
        "patches can create unusual internal combinations; mean ablation is off-manifold.",
        "",
        "The repeated confirmation establishes computational repeatability on the same held-out set, "
        "not a second independent generalization sample. The two successful task formats are still "
        "synthetic. Old P95/P05 sweep outcomes have not been regraded.",
        "",
        "**Stop here under the registered budget.** The result is a strong candidate for a separately "
        "registered follow-up, not authorization to scale the swarm or waive the missed prerequisite.",
        "",
        "[Frozen protocol](bounded-poc-protocol.md) · "
        "[Final runtime manifest](../data/bounded-poc-v1/manifest.json) · "
        "[Original registration](../data/bounded-poc-v1/aborted-start-1/manifest.json) · "
        "[Independent audit](../data/bounded-poc-v1/audit.json) · "
        "[Raw result index](../data/bounded-poc-v1/results.json) · "
        "[Discovery selection](../data/bounded-poc-v1/discovery.json) · "
        "[Data](../data/bounded-poc-v1/dataset.json) · "
        "[Repair provenance](../data/bounded-poc-v1/execution-repair.json)",
    ]
    (ROOT / "reports/bounded-poc-v1.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "audit": "passed",
                "head_level_candidate": audit["component_gates_passed_both_runs"],
                "formal_success": False,
                "model_task": "115/128; required 116/128",
                "bit_identical_replication": bool(replicated_exactly),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
