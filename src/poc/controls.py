"""Analytic planted-mechanism controls, explicitly separate from Qwen evidence."""

import numpy as np

from src.poc.analysis import evaluate, fit_readout
from src.poc.templates import baseline_scores, parse_lookup


def known_controls(rows):
    train = [r for r in rows if r["split"] == "discovery"]
    test = [r for r in rows if r["split"] == "confirmation"]

    def latent(items):
        # Parse the actual text, independently of the stored answer metadata.
        return np.array(
            [
                [2.0 if parse_lookup(r["text"]) == "red" else -2.0, len(r["token_ids"]) / 100]
                for r in items
            ]
        )

    x_train, x_test = latent(train), latent(test)
    y = np.array([r["label"] for r in train])
    readout = fit_readout(x_train, y)
    margins = 2 * x_test[:, 0]
    scores = baseline_scores(test)
    positive = evaluate(
        test,
        x_test,
        readout,
        margins,
        -margins,
        np.zeros(len(test)),
        margins,
        margins,
        scores,
    )
    reverse = readout | {
        "direction": (-np.array(readout["direction"])).tolist(),
        "threshold": -readout["threshold"],
    }
    false_rule = evaluate(
        test,
        x_test,
        reverse,
        margins,
        -margins,
        np.zeros(len(test)),
        margins,
        margins,
        scores,
    )
    nuisance_readout = fit_readout(x_train[:, 1:], y)
    nuisance = evaluate(
        test,
        x_test[:, 1:],
        nuisance_readout,
        margins,
        margins,
        margins,
        margins,
        margins,
        scores,
    )
    return {
        "passed": positive["passed"] and not false_rule["passed"] and not nuisance["passed"],
        "planted_lookup_unit": positive,
        "reversed_explanation": false_rule,
        "nuisance_only_unit": nuisance,
        "meaning": "Analytic planted controls validate evaluator discrimination; they are not Qwen discoveries.",
    }


def known_controls_v2(payload):
    """Executable planted attention/value mechanism; never calls Qwen.

    The toy reads actual token identities at offset-derived source positions.
    Its direct values and fixed attention pattern make the expected QK/OV split
    known. Three additive heads transmit a registered coalition, while a planted
    class prior creates errors repaired by deterministic amplification.
    """
    from src.poc.analysis import (
        V2_LIMITS,
        decomposition_metrics_v2,
        effect_stats,
        readout_metrics_v2,
        rescue_stats_v2,
    )
    from src.poc.templates import parse_v2_lookup

    rows = payload["rows"]
    train = [row for row in rows if row["split"] == "discovery"]
    readout = fit_readout(
        np.array([[1.0 if parse_v2_lookup(row["text"]) == "red" else -1.0] for row in train]),
        np.array([row["label"] for row in train]),
    )
    blocks = {}
    for split in ("discovery", "primary", "vocabulary"):
        items = [row for row in rows if row["split"] == split]
        output_ids = payload["output_ids"]["vocabulary" if split == "vocabulary" else "primary"]
        by_id = {row["id"]: i for i, row in enumerate(items)}
        labels = np.array([row["label"] for row in items])
        values, patterns, masks = [], [], []
        source_masks_exact = True
        for row in items:
            value = np.zeros(len(row["token_ids"]), dtype=float)
            positions = row["color_positions"]
            for position in positions:
                value[position] = 1.0 if row["token_ids"][position] == output_ids[0] else -1.0
            pattern = np.zeros_like(value)
            pattern[positions[row["query_record"]]] = 1.0
            mask = np.zeros(len(value), dtype=bool)
            mask[positions] = True
            source_masks_exact &= set(np.flatnonzero(mask)) == set(positions)
            source_masks_exact &= not np.any(mask & ~mask) and np.all(mask | ~mask)
            values.append(value)
            patterns.append(pattern)
            masks.append(mask)
        signal = np.array(
            [pattern @ value for pattern, value in zip(patterns, values, strict=True)]
        )
        assert np.array_equal(signal, labels)
        # A negative class prior in one group produces two red/green errors.
        first_group = min(row["group"] for row in items)
        prior = np.array([-5.0 if row["group"] == first_group else 0.0 for row in items])
        base = 4 * signal + prior
        coefficients = (0.7, 0.6, 0.5)
        modes = {
            name: []
            for name in (
                "source_color",
                "source_other",
                "ov_all",
                "ov_color",
                "ov_other",
                "qk",
                "qkov",
                "interaction",
            )
        }
        identity = []
        for i, row in enumerate(items):
            j = by_id[row["opposite_id"]]
            p, v, pd, vd, mask = patterns[i], values[i], patterns[j], values[j], masks[i]
            clean = p @ v
            delta_color = (pd * vd - p * v) * mask
            delta_other = (pd * vd - p * v) * ~mask
            candidates = {
                "source_color": clean + np.sum(delta_color),
                "source_other": clean + np.sum(delta_other),
                "ov_all": p @ vd,
                "ov_color": p @ np.where(mask, vd, v),
                "ov_other": p @ np.where(~mask, vd, v),
                "qk": pd @ v,
                "qkov": pd @ vd,
                "interaction": clean + (pd - p) @ (vd - v),
            }
            for name, candidate in candidates.items():
                modes[name].append(base[i] + coefficients[0] * (candidate - clean))
            identity.append(base[i] + coefficients[0] * ((p @ v) - clean))
        modes = {key: np.array(value) for key, value in modes.items()}
        decomposition = decomposition_metrics_v2(items, base, modes)
        solo_margins = [base - 2 * coefficient * signal for coefficient in coefficients]
        solo = [effect_stats(items, base, margin) for margin in solo_margins]
        coalition_margin = base - 2 * sum(coefficients) * signal
        coalition = effect_stats(items, base, coalition_margin)
        transfer = {
            "solo": solo[0],
            "coalition": coalition,
            "passed": all(
                effect["direction_rate"] >= V2_LIMITS["transfer_direction"]
                and effect["median"] >= V2_LIMITS["transfer_median"]
                for effect in (solo[0], coalition)
            ),
        }
        rescue = rescue_stats_v2(items, base, base + sum(coefficients) * signal)
        positive_readout = readout_metrics_v2(items, signal[:, None], readout)
        reversed_model = readout | {
            "direction": (-np.array(readout["direction"])).tolist(),
            "threshold": -readout["threshold"],
        }
        reversed_readout = readout_metrics_v2(items, signal[:, None], reversed_model)
        nuisance_x = np.array([[len(row["token_ids"]) / 100] for row in items])
        nuisance_readout = readout_metrics_v2(items, nuisance_x, fit_readout(nuisance_x, labels))
        nuisance_effect = effect_stats(items, base, base)
        reversed_effect = effect_stats(items, base, base + 2 * coefficients[0] * signal)
        null_decomposition = decomposition_metrics_v2(
            items, base, {name: base.copy() for name in modes}
        )
        reversed_decomposition = decomposition_metrics_v2(
            items, base, {name: 2 * base - measured for name, measured in modes.items()}
        )
        identity_error = float(np.max(np.abs(base - np.array(identity))))
        coalition_checks = {
            "fraction": coalition["fraction"] >= V2_LIMITS["coalition_fraction"],
            "direction": coalition["direction_rate"] >= V2_LIMITS["coalition_direction"],
            "median": coalition["median"] >= V2_LIMITS["coalition_median"],
            "gain_over_best_solo": float(
                np.median(
                    np.array(coalition["effects"])
                    - np.max(np.array([single["effects"] for single in solo]), axis=0)
                )
            )
            >= V2_LIMITS["coalition_gain_over_best_solo"],
        }
        checks = {
            "planted_readout": positive_readout["passed"],
            "source_direct": decomposition["source_classification"] == "direct",
            "routing_ov": decomposition["routing_classification"] == "ov",
            "source_mask_exact": bool(source_masks_exact),
            "identity_exact": identity_error == V2_LIMITS["identity_max"],
            "coalition": all(coalition_checks.values()),
            "rescue": rescue["passed"],
            "vocabulary_transfer": transfer["passed"],
            "nuisance_fails": not nuisance_readout["passed"]
            and not null_decomposition["classifiable"]
            and nuisance_effect["direction_rate"] < V2_LIMITS["solo_direction"]
            and nuisance_effect["fraction"] < V2_LIMITS["coalition_fraction"]
            and nuisance_effect["median"] < V2_LIMITS["transfer_median"],
            "reversed_fails": not reversed_readout["passed"]
            and not reversed_decomposition["classifiable"]
            and reversed_effect["direction_rate"] < V2_LIMITS["solo_direction"]
            and reversed_effect["fraction"] < V2_LIMITS["coalition_fraction"]
            and reversed_effect["median"] < V2_LIMITS["transfer_median"],
        }
        blocks[split] = {
            "passed": all(checks.values()),
            "checks": checks,
            "planted_lookup": positive_readout,
            "decomposition": decomposition,
            "coalition": coalition,
            "coalition_checks": coalition_checks,
            "rescue": rescue,
            "vocabulary_transfer": transfer,
            "nuisance": {
                "passed": nuisance_readout["passed"],
                "readout": nuisance_readout,
                "effect": nuisance_effect,
                "decomposition": null_decomposition,
            },
            "reversed": {
                "passed": reversed_readout["passed"],
                "readout": reversed_readout,
                "effect": reversed_effect,
                "decomposition": reversed_decomposition,
            },
            "identity_error": identity_error,
        }
    return {
        "passed": all(block["passed"] for block in blocks.values()),
        "blocks": blocks,
        "meaning": "Planted executable attention/value controls validate software gates, not a Qwen discovery.",
    }
