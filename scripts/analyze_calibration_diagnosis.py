"""Raw-array audit and retrospective scientific report; no Qwen forward calls."""

import json

import numpy as np

from scripts.run_calibration_diagnosis import OLD, OUT, load, margin, rows_for
from src.poc.analysis import fit_readout, predict


def outcome(rows, base, patched):
    y = np.array([r["label"] for r in rows])
    correct = y * base > 0
    after = y * patched > 0
    gain = y * (patched - base)
    return dict(
        correct=int(after.sum()),
        n=len(rows),
        rescued=int(np.sum(~correct & after)),
        regressed=int(np.sum(correct & ~after)),
        gain_errors_median=float(np.median(gain[~correct])) if np.any(~correct) else None,
        errors_improved=int(np.sum(gain[~correct] > 0)),
        gold_correct=int(np.sum(after & (y == 1))),
        silver_correct=int(np.sum(after & (y == -1))),
        gain_gold_median=float(np.median(gain[y == 1])),
        gain_silver_median=float(np.median(gain[y == -1])),
        rescued_ids=[r["id"] for i, r in enumerate(rows) if not correct[i] and after[i]],
        regressed_ids=[r["id"] for i, r in enumerate(rows) if correct[i] and not after[i]],
    )


def main():
    saved = json.loads((OUT / "saved-evidence.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    result = dict(
        status="Retrospective diagnosis. Original outcome remains no formal survivor.",
        original_calibration_accuracy=55 / 64,
        original_accuracy_gate=0.9,
        selection=selection,
        splits={},
    )
    for split in ("calibration", "confirmation"):
        rows, _ = rows_for(split)
        y = np.array([r["label"] for r in rows])
        base = load(f"{split}-baseline.npz")
        m = margin(base)
        bad = y * m <= 0
        patch = load(f"{split}-interventions.npz")
        if split == "calibration":
            patch.update(load("calibration-heads.npz"))
        again = load(f"repeat-{split}-baseline.npz")
        again_patch = load(f"repeat-{split}-interventions.npz")
        repeat_errors = {
            key: float(np.max(np.abs(value - again[key]))) for key, value in base.items()
        }
        patch_errors = {
            key: float(np.max(np.abs(value - patch[key]))) for key, value in again_patch.items()
        }
        assert max(repeat_errors.values()) <= 0.15
        assert max(patch_errors.values()) <= 0.15
        states = base["states"]
        dot = states[:, :, :, 0]
        rms = states[:, :, :, 1]
        # Projected actual residual increments telescope, including BF16 residual adds.
        attn = (dot[:, :, 1] - dot[:, :, 0]) / rms[:, -1, -1, None]
        mlp = (dot[:, :, 2] - dot[:, :, 1]) / rms[:, -1, -1, None]
        initial = dot[:, 0, 0] / rms[:, -1, -1]
        reconstructed = initial + attn.sum(-1) + mlp.sum(-1)
        np.testing.assert_allclose(reconstructed, dot[:, -1, -1] / rms[:, -1, -1], atol=1e-10)
        np.testing.assert_array_equal(dot[:, 1:, 0], dot[:, :-1, 2])
        rounding = m - reconstructed
        lens = dot / rms
        head_dla = y[:, None, None] * base["head_dot"] / rms[:, -1, -1, None, None]
        cv = np.zeros(len(rows))
        if split == "calibration":
            for g in sorted({r["group"] for r in rows}):
                test = np.array([r["group"] == g for r in rows])
                fit = fit_readout(base["heads"][~test, 5], y[~test])
                cv[test] = predict(base["heads"][test, 5], fit)
        else:
            fit = json.loads((OLD / "calibration.json").read_text())["readout"]
            cv = predict(base["heads"][:, 5], fit)
        source = np.load(
            OLD
            / ("calibration-baseline.npz" if split == "calibration" else "confirm1-baseline.npz")
        )
        by_id = {r["id"]: i for i, r in enumerate(rows)}
        cases = []
        for i, r in enumerate(rows):
            if not bad[i]:
                continue
            j = by_id[r["opposite_id"]]
            plus, minus = (i, j) if y[i] == 1 else (j, i)
            offset = (m[plus] + m[minus]) / 2
            signal = (m[plus] - m[minus]) / 2
            a = source["attention"][i]
            relevant, distractor = (
                r["value_positions"][r["query_record"]],
                r["value_positions"][1 - r["query_record"]],
            )
            trajectory = []
            for layer in range(19, 28):
                for idx, name in ((0, "before"), (1, "after_attention"), (2, "after_mlp")):
                    trajectory.append(
                        dict(
                            stage=f"{layer}:{name}",
                            correct_margin=float(y[i] * lens[i, layer, idx]),
                        )
                    )
            flips = [
                trajectory[k]["stage"]
                for k in range(1, len(trajectory))
                if trajectory[k - 1]["correct_margin"] > 0 and trajectory[k]["correct_margin"] <= 0
            ]
            case = dict(
                id=r["id"],
                text=r["text"],
                format=r["format"],
                query_record=r["query_record"] + 1,
                truth="gold" if y[i] == 1 else "silver",
                correct_margin=float(y[i] * m[i]),
                out_of_group_or_frozen_head_prediction_correct=bool(cv[i] == y[i]),
                source_attention_correct=float(a[relevant]),
                source_attention_other=float(a[distractor]),
                pair_gold_offset=float(offset),
                pair_answer_signal=float(signal),
                head19_direct_signed_contribution=float(head_dla[i, 19, 5]),
                signed_margin_after_layer19=float(y[i] * lens[i, 19, 2]),
                trajectory=trajectory,
                positive_to_negative_transitions=flips,
                intervention_gain={k: float(y[i] * (v[i] - m[i])) for k, v in patch.items()},
            )
            cases.append(case)
        component_stats = []
        for layer in range(28):
            for kind, effects in [("attention", attn), ("mlp", mlp)]:
                signed = y * effects[:, layer]
                component_stats.append(
                    dict(
                        layer=layer,
                        kind=kind,
                        mean_signed_errors=float(np.mean(signed[bad])),
                        median_signed_errors=float(np.median(signed[bad])),
                        mean_raw_gold_direction=float(np.mean(effects[:, layer])),
                        mean_raw_gold_direction_errors=float(np.mean(effects[bad, layer])),
                        mean_signed_correct_silver=float(np.mean(signed[(~bad) & (y == -1)])),
                    )
                )
        pair_layers = []
        for layer in range(28):
            offsets = []
            signals = []
            for i, r in enumerate(rows):
                if y[i] != 1:
                    continue
                j = by_id[r["opposite_id"]]
                offsets.append((lens[i, layer, 2] + lens[j, layer, 2]) / 2)
                signals.append((lens[i, layer, 2] - lens[j, layer, 2]) / 2)
            pair_layers.append(
                dict(
                    layer=layer,
                    median_gold_offset=float(np.median(offsets)),
                    median_binding_signal=float(np.median(signals)),
                )
            )
        result["splits"][split] = dict(
            model_correct=int(np.sum(~bad)),
            n=len(rows),
            errors=len(cases),
            head_readout_correct_on_errors=int(np.sum(cv[bad] == y[bad])),
            readout_total_correct=int(np.sum(cv == y)),
            final_norm_rounding_max=float(np.max(abs(rounding))),
            baseline_repeat_errors=repeat_errors,
            intervention_repeat_errors=patch_errors,
            all_baseline_arrays_identical=all(v == 0 for v in repeat_errors.values()),
            all_repeated_interventions_identical=all(v == 0 for v in patch_errors.values()),
            interventions={k: outcome(rows, m, v) for k, v in patch.items()},
            cases=cases,
            component_accounting=component_stats,
            pair_layer_summary=pair_layers,
            late_residual_signed_increment_errors_mean=float(
                np.mean((y[:, None] * (attn[:, 20:] + mlp[:, 20:]))[bad].sum(-1))
            ),
            positive_after19_errors=int(np.sum((y * lens[:, 19, 2] > 0) & bad)),
        )
    assert result["splits"]["calibration"]["model_correct"] == saved["model_correct"] == 55
    with (OUT / "analysis.json").open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    cal = result["splits"]["calibration"]
    conf = result["splits"]["confirmation"]
    print(
        json.dumps(
            dict(
                calibration={
                    k: v
                    for k, v in cal.items()
                    if k
                    not in ("cases", "component_accounting", "pair_layer_summary", "interventions")
                },
                worst_components=sorted(
                    cal["component_accounting"], key=lambda x: x["mean_signed_errors"]
                )[:10],
                head19={
                    k: v
                    for k, v in cal["interventions"].items()
                    if k.startswith(("head19", "source:"))
                },
                selected=selection,
                confirmation_interventions=conf["interventions"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
