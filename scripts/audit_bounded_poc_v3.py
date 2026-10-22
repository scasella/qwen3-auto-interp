"""Independent array-only audit of the frozen, fixed-coalition rescue experiment.

This module imports no production graders, selectors, templates or model harness.
The only outputs are exclusive-write audit and numeric-facts artifacts. Old PoCs
and every experiment input remain read-only.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v3"
COALITION = ("19:5", "22:9", "21:13")
INTERFERER = "27:14"
HEADS = (*COALITION, INTERFERER)
LIMITS = {
    "model_accuracy": 0.90,
    "discovery_cv": 0.75,
    "readout_accuracy": 0.85,
    "subgroup_accuracy": 0.80,
    "baseline_advantage": 0.15,
    "solo_direction": 0.80,
    "solo_median": 0.25,
    "solo_fraction": 0.10,
    "coalition_direction": 0.90,
    "coalition_fraction": 0.35,
    "coalition_median": 0.25,
    "coalition_gain_over_best_solo": 0.10,
    "transfer_direction": 0.80,
    "transfer_median": 0.25,
    "rescue_median_gain": 0.10,
    "rescue_min_errors_fixed": 1,
    "rescue_max_regressions": 0,
    "identity_max": 0.0,
    "residual_max": 0.05,
    "path_reconstruction_max": 0.05,
    "path_full_margin_error_max": 0.10,
    "delta_closure_max": 2e-5,
    "p": 0.05 / 4,
    "replication_max": 0.0,
}
RULES = tuple(
    {"id": f"center_{alpha}_late_{int(late)}", "alpha": alpha, "late": late}
    for alpha in (1.5, 2.0)
    for late in (False, True)
)
IDENTITY = {"id": "identity", "alpha": 1.0, "late": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def arrays(name):
    with np.load(OUT / name, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    assert result and all(np.isfinite(v).all() for v in result.values()), name
    return result


def write(name, payload):
    with (OUT / name).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare(actual, expected, label="value", tolerance=1e-10):
    """Recursively compare every independently recomputed factual field."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), label
        for key, value in expected.items():
            assert key in actual, (label, key)
            compare(actual[key], value, f"{label}.{key}", tolerance)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, (first, second) in enumerate(zip(actual, expected, strict=True)):
            compare(first, second, f"{label}[{i}]", tolerance)
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected, (label, actual, expected)
    else:
        assert np.isclose(actual, expected, rtol=0, atol=tolerance), (label, actual, expected)


def margins(logits):
    value = np.asarray(logits, dtype=float)
    assert value.shape[-1] == 2
    return value[..., 0] - value[..., 1]


def orientation(rows):
    return np.asarray([row["label"] for row in rows])


def donor_indices(rows, field="opposite_id"):
    index = {row["id"]: i for i, row in enumerate(rows)}
    assert len(index) == len(rows)
    return np.asarray([index[row[field]] for row in rows])


def group_means(values, rows):
    names = sorted({row["group"] for row in rows})
    return np.asarray(
        [np.asarray(values)[[row["group"] == group for row in rows]].mean() for group in names]
    )


def sign_test(values):
    n = len(values)
    positive = int(np.sum(np.asarray(values) > 0))
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / 2**n


def task(rows, margin):
    correct_margin = orientation(rows) * np.asarray(margin)
    correct = correct_margin > 0

    def count(mask):
        total, valid = int(np.sum(mask)), int(np.sum(mask & correct))
        return {"correct": valid, "total": total, "accuracy": valid / total}

    result = count(np.ones(len(rows), dtype=bool))
    for field, name in (
        ("format", "by_format"),
        ("query_record", "by_query"),
        ("answer_color", "by_color"),
    ):
        result[name] = {
            str(value): count(np.asarray([row[field] == value for row in rows]))
            for value in sorted({row[field] for row in rows}, key=str)
        }
    return result | {
        "median_correct_margin": float(np.median(correct_margin)),
        "correct_answer_margins": correct_margin.tolist(),
        "error_ids": [row["id"] for row, valid in zip(rows, correct, strict=True) if not valid],
    }


def effects(rows, baseline, patched):
    baseline, patched = np.asarray(baseline), np.asarray(patched)
    signed = orientation(rows) * (baseline - patched)
    gap = float(np.median(orientation(rows) * (baseline - baseline[donor_indices(rows)])))
    median = float(np.median(signed))
    return {
        "direction_rate": float(np.mean(signed > 0)),
        "median": median,
        "clean_gap": gap,
        "fraction": median / gap if gap > 0 else 0.0,
        "group_sign_p": sign_test(group_means(signed, rows)),
        "effects": signed.tolist(),
    }


def rescue(rows, baseline, assisted):
    before, after = orientation(rows) * baseline, orientation(rows) * assisted
    fixed = (before <= 0) & (after > 0)
    regressed = (before > 0) & (after <= 0)
    gain = float(np.median(after - before))
    checks = {
        "errors_fixed": int(fixed.sum()) >= LIMITS["rescue_min_errors_fixed"],
        "no_regressions": int(regressed.sum()) <= LIMITS["rescue_max_regressions"],
        "median_gain": gain >= LIMITS["rescue_median_gain"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "before": task(rows, baseline),
        "after": task(rows, assisted),
        "errors_fixed": int(fixed.sum()),
        "regressions": int(regressed.sum()),
        "median_gain": gain,
    }


def select(rows, baseline, candidate_logits):
    assert np.asarray(candidate_logits).shape == (4, len(rows), 2)
    candidates = [
        rule | {"stats": rescue(rows, baseline, margins(logits))}
        for rule, logits in zip(RULES, candidate_logits, strict=True)
    ]
    eligible = [candidate for candidate in candidates if candidate["stats"]["regressions"] == 0]
    chosen = (
        min(
            eligible,
            key=lambda item: (
                -item["stats"]["after"]["correct"],
                -item["stats"]["median_gain"],
                item["id"],
            ),
        )
        if eligible
        else IDENTITY
    )
    return {key: chosen[key] for key in IDENTITY}, candidates


def controls(rows, baseline, measured):
    identity = float(np.abs(measured["self"] - baseline["logits"]).max())
    replay = float(np.abs(measured["residual"] - baseline["logits"][donor_indices(rows)]).max())
    return {
        "self_max": identity,
        "residual_max": replay,
        "passed": identity == 0.0 and replay <= LIMITS["residual_max"],
    }


def audit_retirement():
    frozen = read("retirement.json")
    protected = frozen["protected_sha256"]
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, f"Protected old file changed: {name}"
    for version in ("v1", "v2"):
        actual = {
            str(path.relative_to(ROOT))
            for path in (ROOT / f"data/bounded-poc-{version}").rglob("*")
            if path.is_file()
        }
        expected = {name for name in protected if name.startswith(f"data/bounded-poc-{version}/")}
        assert actual == expected, f"Protected {version} inventory changed"
        assert f"reports/bounded-poc-{version}.md" in protected
    return {"passed": True, "protected_file_count": len(protected), "heldout_retired": ["v1", "v2"]}


def audit_dataset(payload):
    rows = payload["rows"]
    assert len(rows) == 256
    assert len({row["id"] for row in rows}) == 256
    retired = [
        row
        for version in ("v1", "v2")
        for row in json.loads((ROOT / f"data/bounded-poc-{version}/dataset.json").read_text())[
            "rows"
        ]
    ]
    retired_keys = {key for row in retired for key in row["keys"]}
    retired_text = {row["text"] for row in retired}
    split_keys = {}
    for split, count, groups in (("discovery", 64, 8), ("primary", 128, 16), ("vocabulary", 64, 8)):
        cohort = [row for row in rows if row["split"] == split]
        assert len(cohort) == count and len({row["group"] for row in cohort}) == groups
        index = {row["id"]: row for row in cohort}
        split_keys[split] = {key for row in cohort for key in row["keys"]}
        assert not split_keys[split] & retired_keys
        assert not {row["text"] for row in cohort} & retired_text
        output = payload["output_ids"]["vocabulary" if split == "vocabulary" else "primary"]
        for row in cohort:
            audit_grammar(row)
            other = index[row["opposite_id"]]
            assert other["opposite_id"] == row["id"]
            assert row["label"] == -other["label"]
            assert row["query_record"] == other["query_record"]
            assert row["keys"] == other["keys"] and len(set(row["keys"])) == 4
            assert len(row["token_ids"]) == len(other["token_ids"]) <= 128
            assert Counter(row["token_ids"]) == Counter(other["token_ids"])
            assert row["token_ids"][-8:] == other["token_ids"][-8:]
            assert row["color_positions"] == other["color_positions"]
            assert len(set(row["color_positions"])) == 4
            source_tokens = [row["token_ids"][i] for i in row["color_positions"]]
            assert sorted(source_tokens) == sorted(output * 2)
            selected_token = source_tokens[row["query_record"]]
            assert selected_token == output[0 if row["label"] == 1 else 1]
            assert row["label"] == index[row["same_answer_id"]]["label"]
        for group in {row["group"] for row in cohort}:
            selected = [row for row in cohort if row["group"] == group]
            assert len(selected) == 8
            assert sorted(row["label"] for row in selected) == [-1] * 4 + [1] * 4
            assert sorted(row["query_record"] for row in selected) == [0, 0, 1, 1, 2, 2, 3, 3]
            assert sorted((row["query_record"], row["assignment"]) for row in selected) == [
                (query, assignment) for query in range(4) for assignment in range(2)
            ]
        for fmt in {row["format"] for row in cohort}:
            assert {row["base_pattern_index"] for row in cohort if row["format"] == fmt} == {
                0,
                1,
                2,
            }
    for first, second in (
        ("discovery", "primary"),
        ("discovery", "vocabulary"),
        ("primary", "vocabulary"),
    ):
        assert not split_keys[first] & split_keys[second]
    discovery_formats = {row["format"] for row in rows if row["split"] == "discovery"}
    held_formats = {row["format"] for row in rows if row["split"] != "discovery"}
    assert not discovery_formats & held_formats
    assert payload["sanity"]["passed"] is True
    return {"passed": True, "rows": 256, "disjoint_from_v1_v2": True, "bindings_per_prompt": 4}


def audit_grammar(row):
    """Parse frozen strings independently of the construction implementation."""
    lines = row["text"].split("\n")
    assert len(lines) == 9
    assert lines[0] == "The registry contains four independent key-to-color bindings."
    assert lines[5] == ""
    assert lines[6] == (
        "Find the requested key among all four records and return only its stored color."
    )
    assert lines[8] == "Answer:"
    key, color = r"(?P<key>[A-Za-z]+[0-9]+)", r"(?P<color>red|blue|green|yellow)"
    patterns = {
        "d0": rf"Entry {key} stores {color}\.",
        "d1": rf"Stored value for {key}: {color}\.",
        "h0": rf"Binding for {key} contains value {color}\.",
        "h1": rf"The registry stores {color} under key {key}\.",
    }
    parsed = [re.fullmatch(patterns[row["format"]], line) for line in lines[1:5]]
    assert all(parsed), row["id"]
    assert [match["key"] for match in parsed] == row["keys"]
    query = re.fullmatch(r"Requested registry key: ([A-Za-z]+[0-9]+)", lines[7])
    assert query and query[1] == row["keys"][row["query_record"]]
    assert parsed[row["query_record"]]["color"] == row["answer_color"]
    assert row["label"] == (1 if row["answer_color"] in ("red", "green") else -1)
    pattern_index = int(row["group"].rsplit("-", 1)[1]) % 3
    pattern = ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0))[pattern_index]
    assert row["base_pattern_index"] == pattern_index
    assert row["base_pattern"] == list(pattern)
    colors = ("green", "yellow") if row["split"] == "vocabulary" else ("red", "blue")
    assert [match["color"] for match in parsed] == [
        colors[(value + row["assignment"]) % 2] for value in pattern
    ]
    cursor = len(lines[0]) + 1
    for i, match in enumerate(parsed):
        expected_span = [cursor + value for value in match.span("color")]
        assert row["color_spans"][i] == expected_span
        cursor += len(lines[i + 1]) + 1


def audit_planted(payload):
    saved = read("known-controls.json")
    for split in ("discovery", "primary", "vocabulary"):
        rows = [row for row in payload["rows"] if row["split"] == split]
        y = orientation(rows)
        first = min(row["group"] for row in rows)
        prior = np.asarray([-5.0 if row["group"] == first else 0.0 for row in rows])
        baseline = 4 * y + prior
        positive = rescue(rows, baseline, baseline + 1.8 * y)
        nuisance = rescue(rows, baseline, baseline)
        reverse = rescue(rows, baseline, baseline - 1.8 * y)
        damaged = baseline + 1.8 * y
        first_correct = next(i for i, value in enumerate(y * baseline) if value > 0)
        damaged[first_correct] = -y[first_correct]
        damaging = rescue(rows, baseline, damaged)
        signal = effects(rows, baseline, baseline - 3.6 * y)
        checks = {
            "baseline": float(np.mean(y * baseline > 0)) >= LIMITS["model_accuracy"],
            "positive_rescue": positive["passed"],
            "nuisance_rejected": not nuisance["passed"],
            "reverse_rejected": not reverse["passed"],
            "regression_rejected": not damaging["passed"],
            "identity_exact": True,
            "coalition": signal["fraction"] >= LIMITS["coalition_fraction"]
            and signal["direction_rate"] >= LIMITS["coalition_direction"],
        }
        expected = {
            "checks": checks,
            "positive": positive,
            "nuisance": nuisance,
            "reverse": reverse,
            "damaging": damaging,
        }
        compare(saved["blocks"][split], expected, f"planted.{split}")
        assert all(checks.values())
    assert saved["passed"] is True
    return {"passed": True, "positive_and_negative_evaluator_controls_recomputed": True}


def bf16(value):
    import torch

    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def audit_deltas(raw, baseline, means, rule):
    """Check CPU-derived means and the actual FP32/BF16 intervention arithmetic.

    Earlier interventions can change later head inputs. Later input vectors are
    reconstructed from the saved delta and then checked against the BF16 lattice
    and saved post-intervention vectors. They are not equated to clean inputs.
    """
    delta = np.asarray(raw["deltas"], dtype=np.float32)
    target = np.asarray(raw["heads"], dtype=np.float32)
    assert delta.shape == target.shape == baseline["heads"].shape
    if rule["id"] == "identity":
        assert not np.any(delta)
        np.testing.assert_array_equal(target, baseline["heads"])
        return {"passed": True, "maximum_arithmetic_error": 0.0}
    means = np.asarray([means[head] for head in HEADS], dtype=np.float32)
    error = 0.0
    for index, head in enumerate(HEADS):
        if index == 3:
            if not rule["late"]:
                assert not np.any(delta[:, index])
                continue
            expected = np.broadcast_to(bf16(means[index]), target[:, index].shape)
            np.testing.assert_array_equal(target[:, index], expected)
            current = expected - delta[:, index]
        else:
            factor = rule["alpha"] - 1
            current = delta[:, index] / factor + means[index]
            if head == "19:5":
                # This is the first intervened layer and must match clean exactly.
                difference = float(np.abs(current - baseline["heads"][:, index]).max())
                error = max(error, difference)
                assert difference <= LIMITS["delta_closure_max"]
        difference = float(np.abs(current - bf16(current)).max())
        error = max(error, difference)
        assert difference <= LIMITS["delta_closure_max"]
        current = bf16(current)
        if index < 3:
            if head == "19:5":
                np.testing.assert_array_equal(current, baseline["heads"][:, index])
            expected_delta = (rule["alpha"] - 1) * (current - means[index])
            difference = float(np.abs(expected_delta - delta[:, index]).max())
            error = max(error, difference)
            assert difference <= LIMITS["delta_closure_max"]
            np.testing.assert_array_equal(target[:, index], bf16(current + delta[:, index]))
    return {"passed": True, "maximum_arithmetic_error": error}


def audit_discovery(payload):
    rows = [row for row in payload["rows"] if row["split"] == "discovery"]
    base = arrays("discovery-baseline.npz")
    assert base["logits"].shape == (64, 2)
    assert base["heads"].shape == (64, 4, 128)
    saved = read("discovery.json")
    average = {head: base["heads"][:, i].mean(0).tolist() for i, head in enumerate(HEADS)}
    compare(saved["mean_vectors"], average, "discovery.mean_vectors", tolerance=0)
    measured = arrays("discovery-controls.npz")
    if "donor_logits" in measured:
        np.testing.assert_array_equal(measured["donor_logits"], base["logits"][donor_indices(rows)])
    control = controls(rows, base, measured)
    compare(saved["controls"], control, "discovery.controls")
    raw = arrays("discovery-rescue.npz")
    chosen, candidates = select(rows, margins(base["logits"]), raw["logits"])
    for recorded, expected in zip(saved["rescue_candidates"], candidates, strict=True):
        compare(recorded, expected, f"discovery.candidate.{expected['id']}")
    selection = {"coalition": list(COALITION), "interferer": INTERFERER, "rescue": chosen}
    compare(saved["selection"], selection, "discovery.selection")
    behavior = task(rows, margins(base["logits"]))
    compare(saved["task"], behavior, "discovery.task")
    comparator = rescue(rows, margins(base["logits"]), margins(raw["late_only"]))
    compare(saved["late_only"], comparator, "discovery.late_only")
    chosen_stats = (
        rescue(rows, margins(base["logits"]), margins(base["logits"]))
        if chosen["id"] == "identity"
        else next(item["stats"] for item in candidates if item["id"] == chosen["id"])
    )
    arithmetic = [
        audit_deltas({"deltas": raw["deltas"][i], "heads": raw["heads"][i]}, base, average, rule)
        for i, rule in enumerate(RULES)
    ]
    return {
        "task": behavior,
        "controls": control,
        "selection": selection,
        "rescue_candidates": candidates,
        "selected_rescue": chosen_stats,
        "task_prerequisite": behavior["accuracy"] >= LIMITS["model_accuracy"],
        "rescue_passed": chosen_stats["passed"],
        "actual_patch_arithmetic": arithmetic,
        "late_only": comparator,
    }


def audit_block(rep, split, payload, discovery):
    rows = [row for row in payload["rows"] if row["split"] == split]
    prefix = f"confirmation-{rep}-{split}"
    base = arrays(prefix + "-baseline.npz")
    measured = arrays(prefix + "-controls.npz")
    raw = arrays(prefix + "-effects.npz")
    rescued = arrays(prefix + "-rescue.npz")
    assert base["logits"].shape == (len(rows), 2)
    assert base["heads"].shape == (len(rows), 4, 128)
    assert raw["solos"].shape == (3, len(rows), 2)
    for key in ("joint", "same", "self"):
        assert raw[key].shape == (len(rows), 2)
    baseline = margins(base["logits"])
    solos = {
        head: effects(rows, baseline, margins(raw["solos"][i])) for i, head in enumerate(COALITION)
    }
    joint = effects(rows, baseline, margins(raw["joint"]))
    gain = float(
        np.median(
            np.asarray(joint["effects"])
            - np.max([item["effects"] for item in solos.values()], axis=0)
        )
    )
    identity = float(np.abs(raw["self"] - base["logits"]).max())
    coalition_checks = {
        "direction": joint["direction_rate"] >= LIMITS["coalition_direction"],
        "median": joint["median"] >= LIMITS["coalition_median"],
        "fraction": joint["fraction"] >= LIMITS["coalition_fraction"],
        "gain_over_best_solo": gain >= LIMITS["coalition_gain_over_best_solo"],
        "significance": joint["group_sign_p"] <= LIMITS["p"],
        "self_exact": identity == LIMITS["identity_max"],
    }
    transfer_checks = {
        name: {
            "direction": item["direction_rate"] >= LIMITS["transfer_direction"],
            "median": item["median"] >= LIMITS["transfer_median"],
            "significance": item["group_sign_p"] <= LIMITS["p"],
        }
        for name, item in (("19:5", solos["19:5"]), ("coalition", joint))
    }
    summary = {
        "task": task(rows, baseline),
        "controls": controls(rows, base, measured),
        "single_effects": solos,
        "coalition": {
            "effects": joint,
            "gain_over_best_solo": gain,
            "checks": coalition_checks,
            "passed": all(coalition_checks.values()),
            "self_error": identity,
            "same_answer": effects(rows, baseline, margins(raw["same"])),
        },
        "rescue": rescue(rows, baseline, margins(rescued["logits"])),
        "late_only": rescue(rows, baseline, margins(rescued["late_only"])),
        "incremental_over_late_only": rescue(
            rows, margins(rescued["late_only"]), margins(rescued["logits"])
        ),
        "transfer": {
            "checks": transfer_checks,
            "passed": all(all(check.values()) for check in transfer_checks.values()),
        },
    }
    recorded = read(f"confirmation-{rep}.json")["blocks"][split]
    assert set(recorded) == set(summary)
    compare(recorded, summary, f"confirmation.{rep}.{split}")
    arithmetic = audit_deltas(
        rescued,
        base,
        read("discovery.json")["mean_vectors"],
        discovery["selection"]["rescue"],
    )
    if "donor_logits" in measured:
        np.testing.assert_array_equal(measured["donor_logits"], base["logits"][donor_indices(rows)])
    return summary | {"actual_patch_arithmetic": arithmetic}


def audit_registration(discovery_only=False):
    initial = read("discovery-registration.json")
    assert initial["limits"] == LIMITS
    assert initial["stage"] == "pre_discovery" and initial["selected"] is None
    registrations = (initial,) if discovery_only else (initial, read("manifest.json"))
    for registration in registrations:
        assert registration["limits"] == LIMITS
        assert registration["model"] == "Qwen/Qwen3-0.6B"
        assert registration["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
        assert registration["device"] == "mps" and registration["dtype"] == "bfloat16"
        assert registration["seed"] == 42
        assert registration["batch_size"] == 64 and registration["max_seq_len"] == 128
        assert registration["head_index"] == list(HEADS)
        assert registration["coalition"] == list(COALITION)
        assert registration["interferer"] == INTERFERER
        assert registration["rules"]["candidates"] == list(RULES)
        assert registration["rules"]["heads_reselected"] is False
        assert registration["rules"]["late_only_comparator_eligible"] is False
        assert registration["rules"]["heldout_runs"] == 2
        assert registration["rules"]["groups"] == {"discovery": 8, "primary": 16, "vocabulary": 8}
        assert registration["rules"]["baseline_primary_required"] == 116
        assert registration["rules"]["baseline_discovery_required"] == 58
        assert registration["rules"]["baseline_vocabulary_required"] == 58
        for filename, expected in registration["sha256"].items():
            assert digest(ROOT / filename) == expected, f"Frozen file changed: {filename}"
    if not discovery_only:
        final = registrations[1]
        assert final["stage"] == "final_pre_heldout"
        assert final["selected"] == read("discovery.json")["selection"]
        assert final["rules"] == initial["rules"]
        for stage in ("confirm1", "confirm2"):
            start = read(stage + "-started.json")
            assert start["registration_sha256"] == digest(OUT / "manifest.json")
            assert final["registered_utc"] < start["utc"]
        for rep in (1, 2):
            recorded = read(f"confirmation-{rep}.json")
            assert recorded["manifest_sha256"] == digest(OUT / "manifest.json")
            assert recorded["selection_sha256"] == digest(OUT / "discovery.json")
    started = read("discover-started.json")
    assert started["registration_sha256"] == digest(OUT / "discovery-registration.json")
    assert initial["registered_utc"] < started["utc"]
    assert read("discovery.json")["registration_sha256"] == digest(
        OUT / "discovery-registration.json"
    )
    assert read("validation.json")["all_passed"] is True
    return {"passed": True, "registrations_checked": len(registrations)}


def repeatability():
    first = {
        path.name.removeprefix("confirmation-1-"): path for path in OUT.glob("confirmation-1-*.npz")
    }
    second = {
        path.name.removeprefix("confirmation-2-"): path for path in OUT.glob("confirmation-2-*.npz")
    }
    assert first and first.keys() == second.keys()
    files = {}
    for name in sorted(first):
        a, b = arrays(first[name].name), arrays(second[name].name)
        assert a.keys() == b.keys()
        fields = {
            key: {
                "bit_identical": bool(np.array_equal(a[key], b[key])),
                "maximum_absolute_difference": float(np.abs(a[key] - b[key]).max()),
            }
            for key in a
        }
        files[name] = {
            "passed": all(field["bit_identical"] for field in fields.values()),
            "fields": fields,
        }
    return {
        "passed": all(file["passed"] for file in files.values()),
        "files": files,
        "meaning": "Fresh-process repetition of one frozen dataset, not independent-data replication.",
    }


def numeric_facts(result):
    facts = {
        "formal_success": int(result["formal_success"]),
        "discovery.correct": result["discovery"]["task"]["correct"],
        "discovery.total": result["discovery"]["task"]["total"],
        "discovery.accuracy": result["discovery"]["task"]["accuracy"],
        "discovery.rescue.errors_fixed": result["discovery"]["selected_rescue"]["errors_fixed"],
        "discovery.rescue.regressions": result["discovery"]["selected_rescue"]["regressions"],
        "discovery.rescue.median_gain": result["discovery"]["selected_rescue"]["median_gain"],
        "replication.bit_identical": int(result["repeatability"]["passed"]),
        "old_versions.protected_files_unchanged": result["retirement"]["protected_file_count"],
    }
    for split, block in result["confirmation"]["1"].items():
        for key in ("correct", "total", "accuracy", "median_correct_margin"):
            facts[f"{split}.task.{key}"] = block["task"][key]
        for key in ("errors_fixed", "regressions", "median_gain"):
            facts[f"{split}.rescue.{key}"] = block["rescue"][key]
            facts[f"{split}.late_only.{key}"] = block["late_only"][key]
            facts[f"{split}.incremental_over_late_only.{key}"] = block[
                "incremental_over_late_only"
            ][key]
        facts[f"{split}.rescue.correct"] = block["rescue"]["after"]["correct"]
        facts[f"{split}.control.self_max"] = block["controls"]["self_max"]
        facts[f"{split}.control.residual_max"] = block["controls"]["residual_max"]
        for name, stat in list(block["single_effects"].items()) + [
            ("coalition", block["coalition"]["effects"])
        ]:
            for key in ("direction_rate", "median", "fraction", "group_sign_p"):
                facts[f"{split}.{name}.{key}"] = stat[key]
        facts[f"{split}.coalition.gain_over_best_solo"] = block["coalition"]["gain_over_best_solo"]
    return facts


def main(discovery_only=False):
    retirement = audit_retirement()
    payload = read("dataset.json")
    dataset = audit_dataset(payload)
    registration = audit_registration(discovery_only)
    planted = audit_planted(payload)
    discovery = audit_discovery(payload)
    if discovery_only:
        result = {
            "retirement": retirement,
            "dataset": dataset,
            "registration": registration,
            "known_controls": planted,
            "discovery": discovery,
        }
        write("discovery-audit.json", result)
        print(
            json.dumps(
                {"discovery_selection_recomputed": True, "selection": discovery["selection"]}
            )
        )
        return
    confirmation = {
        str(rep): {
            split: audit_block(rep, split, payload, discovery)
            for split in ("primary", "vocabulary")
        }
        for rep in (1, 2)
    }
    repetition = repeatability()
    prerequisites = {
        "discovery": discovery["task_prerequisite"],
        "primary": all(
            confirmation[str(rep)]["primary"]["task"]["accuracy"] >= LIMITS["model_accuracy"]
            for rep in (1, 2)
        ),
        "vocabulary": all(
            confirmation[str(rep)]["vocabulary"]["task"]["accuracy"] >= LIMITS["model_accuracy"]
            for rep in (1, 2)
        ),
    }
    gates = {
        "discovery_rescue": discovery["rescue_passed"],
        "primary_rescue": all(
            confirmation[str(rep)]["primary"]["rescue"]["passed"] for rep in (1, 2)
        ),
        "primary_coalition": all(
            confirmation[str(rep)]["primary"]["coalition"]["passed"] for rep in (1, 2)
        ),
        "vocabulary_transfer": all(
            confirmation[str(rep)]["vocabulary"]["transfer"]["passed"] for rep in (1, 2)
        ),
    }
    control_passed = discovery["controls"]["passed"] and all(
        block["controls"]["passed"] and block["coalition"]["checks"]["self_exact"]
        for run in confirmation.values()
        for block in run.values()
    )
    formal = bool(
        all(prerequisites.values())
        and all(gates.values())
        and control_passed
        and planted["passed"]
        and repetition["passed"]
    )
    result = {
        "version": "bounded-poc-v3",
        "manifest_sha256": digest(OUT / "manifest.json"),
        "retirement": retirement,
        "dataset": dataset,
        "registration": registration,
        "known_controls": planted,
        "discovery": discovery,
        "confirmation": confirmation,
        "prerequisites": prerequisites,
        "gates": gates,
        "controls_passed": bool(control_passed),
        "repeatability": repetition,
        "formal_success": formal,
        "all_production_numeric_fields_match": True,
        "raw_sha256": {path.name: digest(path) for path in sorted(OUT.glob("*.npz"))},
    }
    facts = numeric_facts(result)
    write("audit.json", result)
    write("facts.json", facts)
    print(
        json.dumps(
            {
                "audit_complete": True,
                "formal_success": formal,
                "prerequisites": prerequisites,
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-only", action="store_true")
    main(parser.parse_args().discovery_only)
