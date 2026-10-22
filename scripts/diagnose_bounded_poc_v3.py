"""Posthoc error diagnostics only; never consulted by the frozen v3 selector."""

import json
from pathlib import Path

import numpy as np


def main():
    out = Path(__file__).resolve().parents[1] / "data/bounded-poc-v3"
    payload = json.loads((out / "dataset.json").read_text())
    result = {"status": "posthoc descriptive; no role in selection or frozen gates", "blocks": {}}
    for split in ("discovery", "primary", "vocabulary"):
        rows = [r for r in payload["rows"] if r["split"] == split]
        prefix = "discovery" if split == "discovery" else f"confirmation-1-{split}"
        with np.load(out / f"{prefix}-baseline.npz") as saved:
            logits = saved["logits"]
        y = np.array([r["label"] for r in rows])
        margin = logits[:, 0] - logits[:, 1]
        signed = y * margin
        wrong = signed <= 0
        pred = np.sign(margin)
        last = np.array([r["last_value"] for r in rows])
        index = {r["id"]: i for i, r in enumerate(rows)}
        pairs = [
            (i, index[r["opposite_id"]]) for i, r in enumerate(rows) if i < index[r["opposite_id"]]
        ]
        both_wrong = sum(bool(wrong[i] and wrong[j]) for i, j in pairs)
        block = {
            "errors": int(wrong.sum()),
            "errors_predicting_last_record_color": int(np.sum(wrong & (pred == last))),
            "target_matches_last": {
                "correct": int(np.sum((y == last) & ~wrong)),
                "total": int(np.sum(y == last)),
            },
            "target_differs_last": {
                "correct": int(np.sum((y != last) & ~wrong)),
                "total": int(np.sum(y != last)),
            },
            "opposite_pairs_both_wrong": both_wrong,
            "opposite_pairs_total": len(pairs),
            "failed_correct_margin_min": float(signed[wrong].min()),
            "failed_correct_margin_median": float(np.median(signed[wrong])),
            "failed_correct_margin_max": float(signed[wrong].max()),
            "errors_below_minus_one": int(np.sum(signed[wrong] < -1)),
        }
        if split == "discovery":
            candidates = json.loads((out / "discovery.json").read_text())["rescue_candidates"]
            with np.load(out / "discovery-rescue.npz") as saved:
                interventions = saved["logits"]
            diagnostic = []
            for rule, after in zip(candidates, interventions, strict=True):
                after_margin = y * (after[:, 0] - after[:, 1])
                gain = after_margin - signed
                fixed = wrong & (after_margin > 0)
                broken = ~wrong & (after_margin <= 0)
                diagnostic.append(
                    {
                        "id": rule["id"],
                        "errors_improved": int(np.sum(wrong & (gain > 0))),
                        "errors_worsened": int(np.sum(wrong & (gain < 0))),
                        "correct_improved": int(np.sum(~wrong & (gain > 0))),
                        "fixed_ids": [r["id"] for r, mask in zip(rows, fixed, strict=True) if mask],
                        "broken": [
                            {
                                "id": r["id"],
                                "format": r["format"],
                                "query_record": r["query_record"],
                                "color": r["answer_color"],
                                "before_margin": float(signed[i]),
                                "after_margin": float(after_margin[i]),
                            }
                            for i, r in enumerate(rows)
                            if broken[i]
                        ],
                    }
                )
            block["steering"] = diagnostic
        else:
            with np.load(out / f"{prefix}-effects.npz") as saved:
                joint = saved["joint"]
            effect = y * (margin - (joint[:, 0] - joint[:, 1]))
            block["coalition_counterfactual"] = {
                "wrong_cases_correct_direction": int(np.sum(wrong & (effect > 0))),
                "wrong_cases_wrong_direction": int(np.sum(wrong & (effect < 0))),
                "correct_cases_correct_direction": int(np.sum(~wrong & (effect > 0))),
            }
        result["blocks"][split] = block
    with (out / "diagnostics.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
