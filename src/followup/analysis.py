"""Frozen path-specific thresholds supplement, never replace, the original gates."""

import numpy as np

from src.poc.analysis import LIMITS, grouped_values, sign_p

PATH_P = 0.05 / 8


def path_stats(rows, base, patched, gap):
    y = np.array([r["label"] for r in rows])
    effect = y * (base - patched)
    median = float(np.median(effect))
    stats = dict(
        direction=float(np.mean(effect > 0)),
        median=median,
        fraction=median / gap if gap > 0 else 0.0,
        group_p=sign_p(grouped_values(effect, rows)),
        absolute_median=float(np.median(np.abs(effect))),
        absolute_p90=float(np.quantile(np.abs(effect), 0.9)),
    )
    stats["passed"] = (
        stats["direction"] >= LIMITS["patch_direction_rate"]
        and median >= LIMITS["patch_median"]
        and stats["fraction"] >= LIMITS["patch_fraction"]
        and stats["group_p"] <= PATH_P
    )
    stats["negligible"] = stats["absolute_median"] <= 0.10 and stats["absolute_p90"] <= 0.25
    return stats


def mechanisms(rows, base, paths, attention, gap):
    stats = {name: path_stats(rows, base, value, gap) for name, value in paths.items()}
    delta = np.array(
        [
            a[r["value_positions"][r["query_record"]]]
            - a[r["value_positions"][1 - r["query_record"]]]
            for r, a in zip(rows, attention)
        ]
    )
    y = np.array([r["label"] for r in rows])
    deletion = y * (paths["drop_distractor"] - paths["drop_relevant"])
    routing = dict(
        attention_difference_median=float(np.median(delta)),
        attention_direction=float(np.mean(delta > 0)),
        deletion_specificity_median=float(np.median(deletion)),
        deletion_group_p=sign_p(grouped_values(deletion, rows)),
    )
    source = (
        stats["source_swap"]["passed"]
        and stats["q_query"]["passed"]
        and routing["attention_difference_median"] >= 0.10
        and routing["attention_direction"] >= 0.80
        and routing["deletion_specificity_median"] >= 0.10
        and routing["deletion_group_p"] <= PATH_P
    )
    upstream = stats["q_assignment"]["passed"] or stats["v_suffix"]["passed"]
    if source and upstream:
        classification = "mixed"
    elif source:
        classification = "source routing supported; exclusive computation unresolved"
    elif upstream and stats["source_swap"]["negligible"]:
        classification = "upstream-query relay supported"
    else:
        classification = "unresolved"
    return dict(
        paths=stats,
        routing=routing,
        source_routing=source,
        upstream_query=upstream,
        classification=classification,
    )
