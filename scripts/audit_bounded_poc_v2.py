"""Independent NumPy audit of frozen v2 arrays; never calls the target model.

This module intentionally does not import the production analysis, controls, runner,
or harness. Decisions and numerical checks below are recomputed from saved evidence.
Only v2 derived outputs are written; every protected v1 file is hash-checked.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v2"
HEADS = [f"{layer}:{head}" for layer in (*range(15, 23), 27) for head in range(16)]
SCREEN_HEADS = HEADS[:128]
MODES = (
    "source_color",
    "source_other",
    "ov_all",
    "ov_color",
    "ov_other",
    "qk",
    "qkov",
    "interaction",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def arrays(name):
    with np.load(OUT / name, allow_pickle=False) as payload:
        result = {key: payload[key] for key in payload.files}
    assert all(np.isfinite(value).all() for value in result.values()), name
    return result


def margins(logits):
    values = np.asarray(logits, dtype=float)
    assert values.shape[-1] == 2
    return values[..., 0] - values[..., 1]


def orientation(rows):
    return np.array([row["label"] for row in rows])


def donor_indices(rows, field="opposite_id"):
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    assert len(by_id) == len(rows)
    return np.array([by_id[row[field]] for row in rows])


def groups(rows):
    names = sorted({row["group"] for row in rows})
    return np.array([names.index(row["group"]) for row in rows]), names


def group_means(values, rows):
    index, names = groups(rows)
    values = np.asarray(values)
    return np.array([np.mean(values[index == i]) for i in range(len(names))])


def sign_test(values):
    n = len(values)
    positive = int(np.sum(np.asarray(values) > 0))
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / 2**n


def fitted(x, labels):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    positive, negative = x[labels == 1].mean(0), x[labels == -1].mean(0)
    direction = positive - negative
    length = float(np.linalg.norm(direction))
    if length:
        direction /= length
    return {
        "direction": direction.tolist(),
        "threshold": float(((positive + negative) / 2) @ direction),
        "mean": x.mean(0).tolist(),
    }


def predicted(x, readout):
    return np.sign(np.asarray(x, dtype=float) @ readout["direction"] - readout["threshold"])


def cv_accuracy(x, rows):
    index, names = groups(rows)
    labels = orientation(rows)
    guess = np.zeros(len(rows))
    for i in range(len(names)):
        test = index == i
        guess[test] = predicted(np.asarray(x)[test], fitted(np.asarray(x)[~test], labels[~test]))
    return float(np.mean(guess == labels))


def label_randomization(guess, rows):
    labels = orientation(rows)
    index, names = groups(rows)
    random = np.random.default_rng(20261020)
    signs = random.choice([-1, 1], size=(4095, len(names)))
    observed = np.mean(guess == labels)
    null = np.mean(guess[None, :] == labels[None, :] * signs[:, index], axis=1)
    return float((1 + np.sum(null >= observed)) / 4096)


def effects(rows, base, intervention):
    base, intervention = np.asarray(base), np.asarray(intervention)
    labels = orientation(rows)
    values = labels * (base - intervention)
    gap = float(np.median(labels * (base - base[donor_indices(rows)])))
    median = float(np.median(values))
    return {
        "direction_rate": float(np.mean(values > 0)),
        "median": median,
        "clean_gap": gap,
        "fraction": median / gap if gap > 0 else 0.0,
        "group_sign_p": sign_test(group_means(values, rows)),
        "effects": values.tolist(),
    }


def task(rows, margin):
    labels = orientation(rows)
    values = labels * np.asarray(margin)
    correct = values > 0

    def count(mask):
        n, k = int(np.sum(mask)), int(np.sum(correct & mask))
        return {"correct": k, "total": n, "accuracy": k / n}

    result = count(np.ones(len(rows), dtype=bool))
    for field, name in (
        ("format", "by_format"),
        ("query_record", "by_query"),
        ("answer_color", "by_color"),
    ):
        categories = sorted({row[field] for row in rows}, key=str)
        result[name] = {
            str(value): count(np.array([row[field] == value for row in rows]))
            for value in categories
        }
    return result | {
        "median_correct_margin": float(np.median(values)),
        "correct_answer_margins": values.tolist(),
        "error_ids": [row["id"] for row, valid in zip(rows, correct, strict=True) if not valid],
    }


def readout(rows, vectors, model, limits):
    labels = orientation(rows)
    guess = predicted(vectors, model)
    accuracy = float(np.mean(guess == labels))
    subgroup = {
        f"{field}={value}": float(
            np.mean(
                [
                    prediction == row["label"]
                    for prediction, row in zip(guess, rows, strict=True)
                    if row[field] == value
                ]
            )
        )
        for field in ("label", "format")
        for value in sorted({row[field] for row in rows})
    }
    p = label_randomization(guess, rows)
    checks = {
        "readout_accuracy": accuracy >= limits["readout_accuracy"],
        "subgroups": min(subgroup.values()) >= limits["subgroup_accuracy"],
        "baseline_advantage": accuracy - 0.5 >= limits["baseline_advantage"],
        "predictive_significance": p <= limits["p"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "statistics": {
            "readout_accuracy": accuracy,
            "subgroup_accuracy": subgroup,
            "baseline_advantage": accuracy - 0.5,
            "label_flip_p": p,
            "predictions": guess.tolist(),
        },
    }


def rescue(rows, base, assisted, limits):
    before, after = orientation(rows) * base, orientation(rows) * assisted
    fixed = (before <= 0) & (after > 0)
    regressed = (before > 0) & (after <= 0)
    median_gain = float(np.median(after - before))
    checks = {
        "errors_fixed": int(fixed.sum()) >= limits["rescue_min_errors_fixed"],
        "no_regressions": int(regressed.sum()) <= limits["rescue_max_regressions"],
        "median_gain": median_gain >= limits["rescue_median_gain"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "before": task(rows, base),
        "after": task(rows, assisted),
        "errors_fixed": int(fixed.sum()),
        "regressions": int(regressed.sum()),
        "median_gain": median_gain,
        "fixed_ids": [row["id"] for row, ok in zip(rows, fixed, strict=True) if ok],
        "regressed_ids": [row["id"] for row, ok in zip(rows, regressed, strict=True) if ok],
    }


def assert_close(actual, expected, label="value", tolerance=1e-10):
    """Strict recursive factual comparison; expected may contain fewer fields."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), label
        for key, value in expected.items():
            assert key in actual, (label, key)
            assert_close(actual[key], value, f"{label}.{key}", tolerance)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_close(a, e, f"{label}[{i}]", tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, (label, actual, expected)
    else:
        assert np.isclose(actual, expected, rtol=0, atol=tolerance), (label, actual, expected)


def protected_v1():
    snapshot = json.loads((ROOT / "reports/bounded-poc-v1-immutability.json").read_text())
    for name, expected in snapshot.items():
        assert digest(ROOT / name) == expected, f"Protected v1 file changed: {name}"
    actual = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "data/bounded-poc-v1").rglob("*")
        if path.is_file()
    }
    assert actual == {name for name in snapshot if name.startswith("data/bounded-poc-v1/")}
    return {"passed": True, "protected_file_count": len(snapshot)}


def audit_dataset(payload):
    rows = payload["rows"]
    retired = json.loads((ROOT / "data/bounded-poc-v1/dataset.json").read_text())["rows"]
    assert len(rows) == 256
    retired_keys = {key for row in retired for key in row["keys"]}
    retired_text = {row["text"] for row in retired}
    key_sets = {}
    for split, n, ngroups in (("discovery", 64, 16), ("primary", 128, 32), ("vocabulary", 64, 16)):
        selected = [row for row in rows if row["split"] == split]
        assert len(selected) == n
        assert len({row["group"] for row in selected}) == ngroups
        by_id = {row["id"]: row for row in selected}
        assert len(by_id) == n
        key_sets[split] = {key for row in selected for key in row["keys"]}
        assert not key_sets[split] & retired_keys
        assert not {row["text"] for row in selected} & retired_text
        for row in selected:
            other = by_id[row["opposite_id"]]
            assert other["opposite_id"] == row["id"]
            assert row["label"] == -other["label"]
            assert row["query_record"] == other["query_record"]
            assert row["keys"] == other["keys"]
            assert row["color_positions"] == other["color_positions"]
            assert len(row["token_ids"]) == len(other["token_ids"]) <= 128
            assert row["token_ids"][-8:] == other["token_ids"][-8:]
            assert Counter(row["token_ids"]) == Counter(other["token_ids"])
            assert len(set(row["color_positions"])) == 2
            output = payload["output_ids"][split if split == "vocabulary" else "primary"]
            assert sorted(row["token_ids"][i] for i in row["color_positions"]) == sorted(output)
            selected_token = row["token_ids"][row["color_positions"][row["query_record"]]]
            assert selected_token == output[0 if row["label"] == 1 else 1]
            assert row["label"] == by_id[row["same_answer_id"]]["label"]
        assert all(
            sum(row["group"] == g for row in selected) == 4 for g in {r["group"] for r in selected}
        )
    splits = list(key_sets)
    for i, first in enumerate(splits):
        for second in splits[i + 1 :]:
            assert not key_sets[first] & key_sets[second]
    discovery_formats = {r["format"] for r in rows if r["split"] == "discovery"}
    held_formats = {r["format"] for r in rows if r["split"] != "discovery"}
    assert not discovery_formats & held_formats
    assert payload["sanity"]["passed"] is True
    return {"passed": True, "rows": 256, "groups": 64, "disjoint_from_retired_v1": True}


def audit_planted(payload, limits):
    saved = read("known-controls.json")
    audited = {}
    for split in ("discovery", "primary", "vocabulary"):
        rows = [row for row in payload["rows"] if row["split"] == split]
        labels = orientation(rows).astype(float)
        first = min(row["group"] for row in rows)
        prior = np.array([-5.0 if row["group"] == first else 0.0 for row in rows])
        base = 4 * labels + prior
        solo = effects(rows, base, base - 1.4 * labels)
        joint = effects(rows, base, base - 3.6 * labels)
        recovered = rescue(rows, base, base + 1.8 * labels, limits)
        model = fitted(labels[:, None], labels)
        positive = readout(rows, labels[:, None], model, limits)
        reversed_model = model | {"direction": [-1.0], "threshold": -model["threshold"]}
        reversed_readout = readout(rows, labels[:, None], reversed_model, limits)
        empty = effects(rows, base, base)
        wrong = effects(rows, base, base + 1.4 * labels)
        block = saved["blocks"][split]
        assert_close(block["planted_lookup"], positive, f"toy.{split}.readout")
        assert_close(block["coalition"], joint, f"toy.{split}.coalition")
        assert_close(
            block["rescue"],
            {k: v for k, v in recovered.items() if not k.endswith("_ids")},
            f"toy.{split}.rescue",
        )
        assert_close(block["vocabulary_transfer"]["solo"], solo, f"toy.{split}.transfer")
        assert_close(block["nuisance"]["effect"], empty, f"toy.{split}.nuisance")
        assert_close(block["reversed"]["effect"], wrong, f"toy.{split}.reversed")
        assert_close(block["reversed"]["readout"], reversed_readout, f"toy.{split}.reverse_readout")
        assert block["decomposition"]["source_classification"] == "direct"
        assert block["decomposition"]["routing_classification"] == "ov"
        for mode in MODES:
            expected = solo if mode in {"source_color", "ov_all", "ov_color", "qkov"} else empty
            assert_close(
                block["decomposition"]["statistics"][mode], expected, f"toy.{split}.{mode}"
            )
        assert block["identity_error"] == 0
        assert all(block["checks"].values()) and block["passed"] is True
        assert not block["nuisance"]["passed"] and not block["reversed"]["passed"]
        assert positive["passed"] and not reversed_readout["passed"]
        assert recovered["passed"] and joint["fraction"] >= limits["coalition_fraction"]
        assert joint["direction_rate"] >= limits["coalition_direction"]
        assert solo["direction_rate"] >= limits["transfer_direction"]
        audited[split] = {"passed": True, "positive_readout": 1.0, "reversed_readout": 0.0}
    assert saved["passed"] is True
    return {"passed": True, "blocks": audited}


def audit_discovery(payload, limits):
    rows = [row for row in payload["rows"] if row["split"] == "discovery"]
    base = arrays("calibration-baseline.npz")
    assert base["heads"].shape == (64, 144, 128)
    baseline_margin = margins(base["logits"])
    labels = orientation(rows)
    selected = read("discovery.json")
    scan = arrays("discovery-screen.npz")["logits"]
    assert scan.shape == (128, 64, 2)
    screen = []
    for i, head in enumerate(SCREEN_HEADS):
        vector = base["heads"][:, i]
        model = fitted(vector, labels)
        assert_close(selected["readouts"][head], model, f"readout.{head}")
        score = {
            "head": head,
            "cv": cv_accuracy(vector, rows),
            "median": effects(rows, baseline_margin, margins(scan[i]))["median"],
        }
        assert_close(selected["screen"][i], score, f"screen.{head}")
        screen.append(score)
    for head in HEADS[128:]:
        model = fitted(base["heads"][:, HEADS.index(head)], labels)
        assert_close(selected["readouts"][head], model, f"readout.{head}")
    eligible = [row for row in screen if row["cv"] >= limits["discovery_cv"]]
    ranked = sorted(eligible, key=lambda row: (-row["median"], *map(int, row["head"].split(":"))))
    best = ranked[0]["head"] if ranked else None
    assert selected["selection"]["best_head"] == best
    coalition = ["19:5"]
    greedy = []
    for size in (2, 3):
        candidates = [head for head in SCREEN_HEADS if head not in coalition]
        raw = arrays(f"discovery-greedy-{size}.npz")["logits"]
        assert raw.shape == (len(candidates), 64, 2)
        medians = [effects(rows, baseline_margin, margins(logit))["median"] for logit in raw]
        choice = min(
            range(len(candidates)), key=lambda i: (-medians[i], *map(int, candidates[i].split(":")))
        )
        coalition.append(candidates[choice])
        expected = {
            "candidate_heads": candidates,
            "chosen": candidates[choice],
            "coalition": coalition.copy(),
            "scores": medians,
        }
        assert_close(selected["greedy"][size - 2], expected, f"greedy.{size}")
        greedy.append(expected)
    assert selected["selection"]["coalition"] == coalition
    late = arrays("discovery-late.npz")["logits"]
    assert late.shape == (16, 64, 2)
    late_screen = []
    for i in range(16):
        result = rescue(rows, baseline_margin, margins(late[i]), limits)
        item = {
            "head": f"27:{i}",
            "stats": {k: v for k, v in result.items() if not k.endswith("_ids")},
        }
        assert_close(selected["late_screen"][i], item, f"late.{i}")
        late_screen.append(item)
    interferer = min(
        late_screen,
        key=lambda item: (
            -item["stats"]["after"]["correct"],
            item["stats"]["regressions"],
            -item["stats"]["median_gain"],
            int(item["head"].split(":")[1]),
        ),
    )["head"]
    assert selected["selection"]["interferer"] == interferer
    rescued = arrays("discovery-rescue.npz")["logits"]
    assert rescued.shape == (5, 64, 2)
    rescue_candidates = []
    for item, logits in zip(selected["rescue_candidates"], rescued, strict=True):
        result = rescue(rows, baseline_margin, margins(logits), limits)
        expected = {"stats": {k: v for k, v in result.items() if not k.endswith("_ids")}}
        assert_close(item, expected, f"rescue_selection.{item['id']}")
        rescue_candidates.append(item | expected)
    assert sorted((item["alpha"], item["late"]) for item in rescue_candidates) == sorted(
        [(1.5, False), (1.5, True), (2.0, False), (2.0, True), (1.0, True)]
    )
    eligible_rescues = [item for item in rescue_candidates if item["stats"]["regressions"] == 0]
    chosen = (
        min(
            eligible_rescues,
            key=lambda item: (
                -item["stats"]["after"]["correct"],
                -item["stats"]["median_gain"],
                item["id"],
            ),
        )
        if eligible_rescues
        else {"id": "identity", "alpha": 1.0, "late": False}
    )
    assert_close(
        selected["selection"]["rescue"], {key: chosen[key] for key in ("id", "alpha", "late")}
    )
    behavior = task(rows, baseline_margin)
    return {
        "passed": True,
        "task": behavior,
        "task_prerequisite": behavior["accuracy"] >= limits["model_accuracy"],
        "selection": selected["selection"],
        "screen": screen,
        "greedy": greedy,
        "late_screen": late_screen,
        "rescue_candidates": rescue_candidates,
        "readouts_recomputed": 144,
    }


def instrumentation(rows, base, measured, limits):
    residual = float(np.abs(measured["residual"] - base["logits"][donor_indices(rows)]).max())
    sham = float(np.abs(margins(measured["sham"]) - margins(base["logits"])).max())
    reconstructed = float(base["path_error"].max())
    return {
        "residual_max": residual,
        "sham_max": sham,
        "path_reconstruction_max": reconstructed,
        "passed": residual <= limits["residual_max"]
        and sham == 0
        and reconstructed <= limits["path_reconstruction_max"],
    }


def bf16_round(value):
    """CPU-only BF16 casting checks the actually applied GPU donor arithmetic."""
    import torch

    value = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(value).to(torch.bfloat16).float().numpy()


def audit_path_arrays(rows, base, measured, limits):
    pattern, values = base["pattern"], base["values"]
    assert pattern.shape == values.shape[:2] and values.shape[-1] == 128
    assert pattern.shape[0] == len(rows)
    assert np.all(pattern >= 0)
    np.testing.assert_allclose(pattern.sum(1), 1, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(base["lengths"], [len(row["token_ids"]) for row in rows])
    source = np.zeros_like(pattern, dtype=bool)
    for i, row in enumerate(rows):
        n = len(row["token_ids"])
        source[i, row["color_positions"]] = True
        assert not np.any(pattern[i, n:]) and not np.any(values[i, n:])
    donor = donor_indices(rows)
    pd, vd = pattern[donor], values[donor]
    terms = pd[..., None] * vd - pattern[..., None] * values
    ov = pattern[..., None] * (vd - values)
    expected = {
        "source_color": (terms * source[..., None]).sum(1),
        "source_other": (terms * (~source)[..., None]).sum(1),
        "ov_all": ov.sum(1),
        "ov_color": (ov * source[..., None]).sum(1),
        "ov_other": (ov * (~source)[..., None]).sum(1),
        "qk": ((pd - pattern)[..., None] * values).sum(1),
        "qkov": terms.sum(1),
        "interaction": ((pd - pattern)[..., None] * (vd - values)).sum(1),
    }
    reconstructed = (pattern[..., None] * values).sum(1)
    np.testing.assert_allclose(reconstructed, base["reconstructed"], rtol=0, atol=2e-5)
    head = base["heads"][:, HEADS.index("19:5")]
    errors = np.abs(head - base["reconstructed"]).max(1)
    np.testing.assert_allclose(errors, base["path_error"], atol=2e-7, rtol=0)
    max_delta_error = 0.0
    for mode, delta in expected.items():
        saved_delta = measured[mode + "_delta"]
        error = float(np.abs(delta - saved_delta).max())
        max_delta_error = max(max_delta_error, error)
        np.testing.assert_allclose(delta, saved_delta, rtol=0, atol=limits["delta_closure_max"])
        np.testing.assert_array_equal(measured[mode + "_self_delta"], np.zeros_like(saved_delta))
        applied = bf16_round(head + saved_delta.astype(np.float32))
        np.testing.assert_array_equal(applied, measured[mode + "_applied"])
    return {
        "passed": True,
        "maximum_independent_delta_error": max_delta_error,
        "applied_vectors_match_bf16_arithmetic": True,
    }


def classify_paths(statistics):
    full = statistics["qkov"]["median"]

    def choose(first, second, first_name, second_name):
        a, b = statistics[first], statistics[second]
        valid_a = a["median"] >= 0.25 and a["direction_rate"] >= 0.80
        valid_b = b["median"] >= 0.25 and b["direction_rate"] >= 0.80
        if valid_a and a["median"] >= 0.5 * full and a["median"] >= b["median"] + 0.10:
            return first_name
        if valid_b and b["median"] >= 0.5 * full and b["median"] >= a["median"] + 0.10:
            return second_name
        return "mixed" if valid_a and valid_b else "unresolved"

    return choose("source_color", "source_other", "direct", "relay"), choose(
        "ov_all", "qk", "ov", "pattern"
    )


def audit_block(rep, split, payload, selected, limits):
    rows = [row for row in payload["rows"] if row["split"] == split]
    prefix = f"confirmation-{rep}-{split}"
    base = arrays(prefix + "-baseline.npz")
    assert base["heads"].shape == (len(rows), 144, 128)
    control = instrumentation(rows, base, arrays(prefix + "-controls.npz"), limits)
    selection = selected["selection"]
    wanted = sorted(
        set(
            selection["coalition"]
            + ["19:5"]
            + ([selection["best_head"]] if selection["best_head"] else [])
        ),
        key=lambda h: tuple(map(int, h.split(":"))),
    )
    solos = {head: arrays(prefix + f"-solo-{head.replace(':', '-')}.npz") for head in wanted}
    coalition = arrays(prefix + "-coalition.npz")
    assisted = arrays(prefix + "-rescue.npz")
    paths = arrays(prefix + "-decomposition.npz")
    path_audit = audit_path_arrays(rows, base, paths, limits)
    bm = margins(base["logits"])
    solo_effects = {
        head: effects(rows, bm, margins(raw["opposite"])) for head, raw in solos.items()
    }
    readouts = {
        head: readout(rows, base["heads"][:, HEADS.index(head)], selected["readouts"][head], limits)
        for head in wanted
    }
    joint = effects(rows, bm, margins(coalition["opposite"]))
    components = np.array([solo_effects[head]["effects"] for head in selection["coalition"]])
    joint_effect = np.asarray(joint["effects"])
    gain = float(np.median(joint_effect - components.max(0)))
    interaction = joint_effect - components.sum(0)
    interaction_groups = group_means(interaction, rows)
    joint_checks = {
        "direction": joint["direction_rate"] >= limits["coalition_direction"],
        "median": joint["median"] >= limits["coalition_median"],
        "fraction": joint["fraction"] >= limits["coalition_fraction"],
        "gain_over_best_solo": gain >= limits["coalition_gain_over_best_solo"],
        "significance": joint["group_sign_p"] <= limits["p"],
        "self_exact": bool(np.array_equal(margins(coalition["sham"]), bm)),
    }
    path_statistics = {mode: effects(rows, bm, margins(paths[mode + "_logits"])) for mode in MODES}
    source, routing = classify_paths(path_statistics)
    source_closure = float(
        np.abs(
            paths["source_color_delta"] + paths["source_other_delta"] - paths["qkov_delta"]
        ).max()
    )
    qkov_closure = float(
        np.abs(
            paths["ov_all_delta"]
            + paths["qk_delta"]
            + paths["interaction_delta"]
            - paths["qkov_delta"]
        ).max()
    )
    full_error = float(
        np.abs(margins(paths["qkov_logits"]) - margins(solos["19:5"]["opposite"])).max()
    )
    self_error = max(
        float(np.abs(margins(paths[mode + "_self_logits"]) - bm).max()) for mode in MODES
    )
    path_checks = {
        "reconstruction": float(base["path_error"].max()) <= limits["path_reconstruction_max"],
        "source_closure": source_closure <= limits["delta_closure_max"],
        "qkov_closure": qkov_closure <= limits["delta_closure_max"],
        "full_margin_closure": full_error <= limits["path_full_margin_error_max"],
        "self_exact": self_error == 0,
        "classifiable": source != "unresolved" and routing != "unresolved",
    }
    rescue_summary = rescue(rows, bm, margins(assisted["logits"]), limits)
    transfer_checks = {}
    for head, value in (("19:5", solo_effects["19:5"]), ("coalition", joint)):
        transfer_checks[head] = {
            "direction": value["direction_rate"] >= limits["transfer_direction"],
            "median": value["median"] >= limits["transfer_median"],
            "significance": value["group_sign_p"] <= limits["p"],
        }
    attention = {"relevant": [], "other_value": [], "keys": [], "delimiters": []}
    for i, row in enumerate(rows):
        pattern = base["pattern"][i]
        attention["relevant"].append(float(pattern[row["color_positions"][row["query_record"]]]))
        attention["other_value"].append(
            float(pattern[row["color_positions"][1 - row["query_record"]]])
        )
        key_positions = np.array(
            [p for positions in row["key_positions"] for p in positions], dtype=int
        )
        attention["keys"].append(float(pattern[key_positions].sum()))
        attention["delimiters"].append(float(pattern[row["delimiter_positions"]].sum()))
    self_controls = (
        control["passed"]
        and all(np.array_equal(margins(raw["sham"]), bm) for raw in solos.values())
        and joint_checks["self_exact"]
        and self_error == 0
    )
    summary = {
        "task": task(rows, bm),
        "controls": control | {"all_self_controls_passed": bool(self_controls)},
        "single_effects": solo_effects,
        "readouts": readouts,
        "decomposition": {
            "source_classification": source,
            "routing_classification": routing,
            "classifiable": source != "unresolved" and routing != "unresolved",
            "statistics": path_statistics,
            "meaning": "Overlapping intervention contrasts; no percentage-mediation claim.",
            "checks": path_checks,
            "passed": all(path_checks.values()),
            "source_closure": source_closure,
            "qkov_closure": qkov_closure,
            "full_margin_error": full_error,
            "self_error": self_error,
            "attention_mass": attention,
        },
        "coalition": {
            "effects": joint,
            "checks": joint_checks,
            "passed": all(joint_checks.values()),
            "gain_over_best_solo": gain,
            "interaction_median": float(np.median(interaction)),
            "interaction_group_direction": float(np.mean(interaction_groups > 0)),
            "synergy_supported": bool(
                np.median(interaction) >= 0.10 and np.mean(interaction_groups > 0) >= 0.80
            ),
        },
        "rescue": {key: value for key, value in rescue_summary.items() if not key.endswith("_ids")},
        "transfer": {
            "checks": transfer_checks,
            "passed": all(all(checks.values()) for checks in transfer_checks.values()),
        },
    }
    recorded = read(f"confirmation-{rep}.json")["blocks"][split]
    assert set(recorded) == set(summary)
    assert_close(recorded, summary, f"confirmation.{rep}.{split}")
    controls = {
        name: {
            "mean": effects(rows, bm, margins(raw["mean"])),
            "same": effects(rows, bm, margins(raw["same"])),
        }
        for name, raw in solos.items()
    }
    controls["coalition"] = {
        mode: effects(rows, bm, margins(coalition[mode])) for mode in ("mean", "same")
    }
    best = selection["best_head"]
    solo_checks = (
        {}
        if best is None
        else {
            "readout": readouts[best]["passed"],
            "direction": solo_effects[best]["direction_rate"] >= limits["solo_direction"],
            "median": solo_effects[best]["median"] >= limits["solo_median"],
            "fraction": solo_effects[best]["fraction"] >= limits["solo_fraction"],
        }
    )
    gates = {
        "gate1_source_and_routing": summary["decomposition"]["passed"],
        "gate2_single_and_coalition": best is not None
        and all(solo_checks.values())
        and summary["coalition"]["passed"],
        "gate3_rescue": summary["task"]["accuracy"] >= limits["model_accuracy"]
        and rescue_summary["passed"],
        "gate4_vocabulary_transfer": summary["transfer"]["passed"],
    }
    return summary | {
        "replicate": rep,
        "split": split,
        "independent_path_audit": path_audit,
        "solo_selection_checks": solo_checks,
        "control_effects": controls,
        "fixed_ids": rescue_summary["fixed_ids"],
        "regressed_ids": rescue_summary["regressed_ids"],
        "gates": gates,
    }


def audit_registration():
    final, initial = read("manifest.json"), read("discovery-registration.json")
    assert final["stage"] == "final_pre_heldout" and initial["stage"] == "pre_discovery"
    assert final["limits"] == initial["limits"]
    assert final["rules"] == initial["rules"]
    for registration in (initial, final):
        for filename, expected in registration["sha256"].items():
            assert digest(ROOT / filename) == expected, f"Frozen file changed: {filename}"
        assert registration["rules"]["v1_heldout_retired"] is True
        assert registration["rules"]["heldout_runs"] == 2
    assert final["selected"] == read("discovery.json")["selection"]
    for stage in ("confirm1", "confirm2"):
        started = read(stage + "-started.json")
        assert started["registration_sha256"] == digest(OUT / "manifest.json")
        assert final["registered_utc"] < started["utc"]
    for rep in (1, 2):
        summary = read(f"confirmation-{rep}.json")
        assert summary["manifest_sha256"] == digest(OUT / "manifest.json")
        assert summary["selection_sha256"] == digest(OUT / "discovery.json")
    assert read("validation.json")["all_passed"] is True
    return final


def replication():
    first = {p.name.replace("confirmation-1-", ""): p for p in OUT.glob("confirmation-1-*.npz")}
    second = {p.name.replace("confirmation-2-", ""): p for p in OUT.glob("confirmation-2-*.npz")}
    assert first and first.keys() == second.keys()
    results = {}
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
        results[name] = {
            "passed": all(v["bit_identical"] for v in fields.values()),
            "fields": fields,
        }
    return {
        "passed": all(v["passed"] for v in results.values()),
        "files": results,
        "meaning": "Two fresh processes on the same frozen dataset; computational repeatability only.",
    }


def percentage(value):
    return f"{100 * value:.3f}%"


def scientific_report(result):
    primary = result["confirmation"]["1"]["primary"]
    vocabulary = result["confirmation"]["1"]["vocabulary"]
    discovery = result["discovery"]
    selection = discovery["selection"]
    ptask = primary["task"]
    coalition = primary["coalition"]
    path = primary["decomposition"]
    rescued = primary["rescue"]
    best = selection["best_head"]
    passed = result["formal_success"]
    outcome = (
        "all frozen success criteria passed"
        if passed
        else "one or more frozen gates failed or remain unresolved"
    )
    lines = [
        "# Bounded PoC v2: causal confirmation and post-mortem follow-up",
        "",
        f"**Formal outcome: {outcome}.** No v1 file or its final report was changed. "
        "V1 held-out examples were permanently retired before this experiment. V2 used new keys, "
        "two new held-out syntax formats and a separate green/yellow vocabulary block. No prompt "
        "was dropped and no numerical gate was rounded, weakened or changed after confirmation.",
        "",
        "The output is the correct-answer preference between two actual next-token logits, "
        "computed from the normalized final residual. It is not unrestricted generation accuracy.",
        "",
        "## Frozen design and discovery",
        "",
        "Discovery used 16 groups / 64 red-blue prompts; primary confirmation used 32 groups / "
        "128 prompts; vocabulary confirmation used 16 groups / 64 green-yellow prompts. Each group "
        "crosses both queried records and both assignments. Opposite-assignment pairs retain the "
        "token multiset, length, query, final eight tokens and exact offset-derived source positions. "
        "Every bound value is a single token. All indices are zero-based.",
        "",
        f"Discovery unassisted task accuracy was {discovery['task']['correct']}/64 "
        f"({percentage(discovery['task']['accuracy'])}); its ≥90% prerequisite "
        f"{'passed' if discovery['task_prerequisite'] else 'failed'}. All 128 heads in layers 15–22 "
        "were screened on all 64 prompts with leave-one-group-out readout validation and matched "
        "causal replacement. The strongest eligible median patch effect selected "
        f"`{best}`. The fixed reference was `19:5`.",
        "",
        f"The greedy coalition was **{', '.join('`' + h + '`' for h in selection['coalition'])}**. "
        "Its two additions were selected exclusively by discovery joint effect. The discovery-selected "
        f"late interferer was `{selection['interferer']}` and the frozen rescue rule was "
        f"`{selection['rescue']['id']}` (alpha={selection['rescue']['alpha']}, "
        f"late mean ablation={selection['rescue']['late']}). Centered amplification uses "
        "x+(alpha−1)(x−discovery_mean), without access to the correct answer during execution.",
        "",
        "## Registered verdicts",
        "",
        "| Required item | Result, both runs checked | Verdict |",
        "| --- | --- | --- |",
        f"| Discovery unassisted prerequisite | {discovery['task']['correct']}/64; ≥90% required | {'PASS' if result['prerequisites']['discovery'] else 'FAIL'} |",
        f"| Primary unassisted prerequisite | {ptask['correct']}/128; ≥116/128 required | {'PASS' if result['prerequisites']['primary'] else 'FAIL'} |",
    ]
    descriptions = {
        "gate1_source_and_routing": f"Source={path['source_classification']}, routing={path['routing_classification']}",
        "gate2_single_and_coalition": f"Best single={best}; coalition gap fraction={percentage(coalition['effects']['fraction'])}",
        "gate3_rescue": f"{rescued['errors_fixed']} errors fixed, {rescued['regressions']} correct answers broken",
        "gate4_vocabulary_transfer": f"19:5 direction={percentage(vocabulary['single_effects']['19:5']['direction_rate'])}; coalition={percentage(vocabulary['coalition']['effects']['direction_rate'])}",
    }
    for gate, valid in result["gates"].items():
        lines.append(
            f"| {gate} | {descriptions[gate]} | {'PASS' if valid else 'FAIL / UNRESOLVED'} |"
        )
    lines.extend(
        [
            f"| Real-model, planted and validation controls | All required controls | {'PASS' if result['controls_passed'] else 'FAIL'} |",
            f"| Fresh-process repeatability | Every saved baseline/intervention array | {'BIT-IDENTICAL' if result['repeatability']['passed'] else 'FAILED'} |",
            "",
            "## Source-position and QK/OV decomposition",
            "",
            "Only head 19:5 at the final query position is modified. Reconstructed FP32 contributions "
            "are added as deltas to the measured SDPA head vector. The shared GQA value projection is "
            "never mutated. Donor patterns and value sequences come from aligned clean counterfactuals. "
            "Explicit reconstructed probabilities may differ numerically from the fused BF16 kernel.",
            "",
            "| Primary intervention | Positive direction | Median margin reduction |",
            "| --- | ---: | ---: |",
        ]
    )
    for mode, stats in path["statistics"].items():
        lines.append(
            f"| {mode} | {percentage(stats['direction_rate'])} | {stats['median']:.6f} logits |"
        )
    lines.extend(
        [
            "",
            f"Maximum measured-head reconstruction error: {primary['controls']['path_reconstruction_max']:.8f}; "
            f"source delta closure error: {path['source_closure']:.8f}; QK/OV delta closure error: "
            f"{path['qkov_closure']:.8f}; reconstructed-full versus whole-head patch margin error: "
            f"{path['full_margin_error']:.8f}. Every identity margin error was {path['self_error']:.8f}. "
            "These are overlapping intervention contrasts; ratios are not percentages of causal mediation.",
            "",
            "Reconstructed mean attention mass: "
            + ", ".join(
                f"{name}={float(np.mean(values)):.6f}"
                for name, values in path["attention_mass"].items()
            )
            + ". Attention mass alone does not establish a causal route.",
            "",
            "## Single heads, coalition and interaction",
            "",
            "| Primary head | Frozen readout | Patch direction | Median reduction | Gap fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for head, stats in primary["single_effects"].items():
        accuracy = primary["readouts"][head]["statistics"]["readout_accuracy"]
        lines.append(
            f"| `{head}` | {percentage(accuracy)} | {percentage(stats['direction_rate'])} | {stats['median']:.6f} | {percentage(stats['fraction'])} |"
        )
    joint = coalition["effects"]
    lines.extend(
        [
            "",
            f"The coalition's median reduction was {joint['median']:.6f} logits, "
            f"{percentage(joint['fraction'])} of the median clean/counterfactual gap, with "
            f"{percentage(joint['direction_rate'])} directional consistency. The registered requirements "
            "were ≥0.25 logits, ≥35%, ≥90%, group sign p≤0.0125, exact self identity and a median "
            f"gain over the strongest per-prompt constituent ≥0.10 logits. The measured gain was "
            f"{coalition['gain_over_best_solo']:.6f} logits and sign p={joint['group_sign_p']:.8g}.",
            "",
            f"Joint-minus-sum interaction had median {coalition['interaction_median']:.6f} logits and "
            f"positive group means in {percentage(coalition['interaction_group_direction'])} of groups. "
            f"The separate frozen superadditivity criterion {'passed' if coalition['synergy_supported'] else 'did not pass'}. "
            "Joint coverage and greedy selection alone do not establish synergy. Same-answer and "
            "discovery-mean interventions are retained in the machine-readable results.",
            "",
            "## Prospective task rescue",
            "",
            f"Unassisted primary accuracy was {rescued['before']['correct']}/128; the frozen intervention "
            f"produced {rescued['after']['correct']}/128. It corrected {rescued['errors_fixed']} errors "
            f"and broke {rescued['regressions']} previously correct preferences. Median correct-margin "
            f"increase was {rescued['median_gain']:.6f} logits (required ≥0.10). Rescue never substitutes "
            "for the unassisted ≥90% prerequisite. At a baseline ceiling with no errors, successful "
            "error rescue is untestable under the frozen rule.",
            "",
            "Fixed cases: " + (", ".join(primary["fixed_ids"]) or "none") + ".",
            "",
            "Regressed cases: " + (", ".join(primary["regressed_ids"]) or "none") + ".",
            "",
            "## Vocabulary transfer",
            "",
            f"Green/yellow unassisted task accuracy was {vocabulary['task']['correct']}/64. This score "
            "is descriptive; causal transfer requires both 19:5 and the unchanged coalition to reach "
            "≥80% direction, ≥0.25-logit median reduction and group sign p≤0.0125.",
            "",
            "| Green/yellow intervention | Direction | Median reduction | Group sign p |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, stats in (
        ("19:5", vocabulary["single_effects"]["19:5"]),
        ("coalition", vocabulary["coalition"]["effects"]),
    ):
        lines.append(
            f"| {name} | {percentage(stats['direction_rate'])} | {stats['median']:.6f} | {stats['group_sign_p']:.8g} |"
        )
    lines.extend(
        [
            "",
            "Transfer to a second color pair supports that specific vocabulary generalization. "
            "It does not prove a universal abstract bound-value mover. The red/blue readout was never "
            "refitted on green/yellow labels; any such readout results are descriptive diagnostics.",
            "",
            "## Controls, independent audit and scope",
            "",
            "The independent audit uses its own NumPy arithmetic and does not call the target model "
            "or import production graders. It re-fits discovery readouts, recomputes every group CV "
            "score, single-head rank, greedy addition, late-head rank, rescue selection and held-out "
            "statistic. It independently checks source masks, contribution algebra and the actual BF16 "
            "patched vectors, and compares every production summary against the arrays. It verifies "
            "all frozen hashes and every protected v1 hash.",
            "",
            "Planted lookup controls pass the known value-routing, coalition, rescue and transfer gates; "
            "nuisance and reversed controls fail. These validate the evaluator and are not Qwen discoveries. "
            "Final-residual donor replay and exact self/coalition identity validate the intervention path. "
            "The validation artifact records unit tests, dedicated Metal regressions and lint.",
            "",
            "The two confirmation processes reload weights independently and use the same frozen data "
            "and batching plan. Repeatability is computational reproducibility on one held-out dataset, "
            "not an independent dataset replication. Synthetic keys, full-sentence formats, two value "
            "pairs and one checkpoint bound all conclusions. Mean replacement and steering can produce "
            "internal states outside the unmodified model's distribution. Source routing at one head "
            "does not by itself explain the full key lookup algorithm.",
            "",
            "The registered budget ends here. No additional sweep, threshold change, replacement "
            "candidate or third confirmation run is authorized by this result.",
            "",
            "[Frozen v2 protocol](bounded-poc-v2-protocol.md) · "
            "[Pre-held-out manifest](../data/bounded-poc-v2/manifest.json) · "
            "[Discovery registration](../data/bounded-poc-v2/discovery-registration.json) · "
            "[Independent audit](../data/bounded-poc-v2/audit.json) · "
            "[All audited results](../data/bounded-poc-v2/results.json) · "
            "[Frozen dataset](../data/bounded-poc-v2/dataset.json) · "
            "[V1 post-mortem](bounded-poc-v1-postmortem.md)",
        ]
    )
    review = OUT / "review-results.json"
    if review.exists():
        lines.extend(
            [
                "",
                "[Final gpt-6-luna scientist review](../data/bounded-poc-v2/review-results.json) · "
                "[Numeric review adjudication](../data/bounded-poc-v2/review-adjudication.json). "
                "Reviewer prose cannot override a frozen gate.",
            ]
        )
    (ROOT / "reports/bounded-poc-v2.md").write_text("\n".join(lines) + "\n")


def numeric_facts(result):
    """Flat exact array-derived claims supplied to the advisory scientist review."""
    facts = {
        "discovery.correct": result["discovery"]["task"]["correct"],
        "discovery.total": result["discovery"]["task"]["total"],
        "discovery.accuracy": result["discovery"]["task"]["accuracy"],
        "v1.protected_files_unchanged": result["v1_immutable"]["protected_file_count"],
        "formal_success": int(result["formal_success"]),
        "replication.bit_identical": int(result["repeatability"]["passed"]),
    }
    for split, block in result["confirmation"]["1"].items():
        for key in ("correct", "total", "accuracy", "median_correct_margin"):
            facts[f"{split}.task.{key}"] = block["task"][key]
        for key in ("errors_fixed", "regressions", "median_gain"):
            facts[f"{split}.rescue.{key}"] = block["rescue"][key]
        facts[f"{split}.rescue.correct"] = block["rescue"]["after"]["correct"]
        for key in ("residual_max", "sham_max", "path_reconstruction_max"):
            facts[f"{split}.control.{key}"] = block["controls"][key]
        for head, stats in block["single_effects"].items():
            for key in ("direction_rate", "median", "fraction", "clean_gap", "group_sign_p"):
                facts[f"{split}.head.{head}.{key}"] = stats[key]
            facts[f"{split}.head.{head}.readout_accuracy"] = block["readouts"][head]["statistics"][
                "readout_accuracy"
            ]
        for key in ("direction_rate", "median", "fraction", "clean_gap", "group_sign_p"):
            facts[f"{split}.coalition.{key}"] = block["coalition"]["effects"][key]
        for key in ("gain_over_best_solo", "interaction_median", "interaction_group_direction"):
            facts[f"{split}.coalition.{key}"] = block["coalition"][key]
        facts[f"{split}.coalition.synergy_supported"] = int(block["coalition"]["synergy_supported"])
        for mode, stats in block["decomposition"]["statistics"].items():
            for key in ("direction_rate", "median", "fraction", "group_sign_p"):
                facts[f"{split}.decomposition.{mode}.{key}"] = stats[key]
        for key in ("source_closure", "qkov_closure", "full_margin_error", "self_error"):
            facts[f"{split}.decomposition.{key}"] = block["decomposition"][key]
    return facts


def main(discovery_only=False):
    immutable = protected_v1()
    payload = read("dataset.json")
    dataset_check = audit_dataset(payload)
    registration = read("discovery-registration.json")
    limits = registration["limits"]
    planted = audit_planted(payload, limits)
    discovery = audit_discovery(payload, limits)
    calibration = arrays("calibration-baseline.npz")
    calibration_rows = [row for row in payload["rows"] if row["split"] == "discovery"]
    control = instrumentation(
        calibration_rows, calibration, arrays("calibration-controls.npz"), limits
    )
    assert_close(read("discovery.json")["controls"], control, "discovery.controls")
    assert_close(read("discovery.json")["task"], discovery["task"], "discovery.task")
    for i, head in enumerate(HEADS):
        np.testing.assert_array_equal(
            read("discovery.json")["mean_vectors"][head], calibration["heads"][:, i].mean(0)
        )
    if discovery_only:
        write(
            "discovery-audit.json",
            {
                "v1_immutable": immutable,
                "dataset": dataset_check,
                "planted": planted,
                "discovery": discovery,
                "controls": control,
            },
        )
        print(
            json.dumps(
                {"discovery_selection_recomputed": True, "selection": discovery["selection"]}
            )
        )
        return
    manifest = audit_registration()
    assert limits == manifest["limits"]
    selected = read("discovery.json")
    confirmation = {
        str(rep): {
            split: audit_block(rep, split, payload, selected, limits)
            for split in ("primary", "vocabulary")
        }
        for rep in (1, 2)
    }
    repeatability = replication()
    prerequisites = {
        "discovery": discovery["task_prerequisite"],
        "primary": all(
            confirmation[str(rep)]["primary"]["task"]["accuracy"] >= limits["model_accuracy"]
            for rep in (1, 2)
        ),
    }
    gates = {
        gate: all(confirmation[str(rep)][split]["gates"][gate] for rep in (1, 2))
        for gate, split in (
            ("gate1_source_and_routing", "primary"),
            ("gate2_single_and_coalition", "primary"),
            ("gate3_rescue", "primary"),
            ("gate4_vocabulary_transfer", "vocabulary"),
        )
    }
    controls_passed = control["passed"] and all(
        block["controls"]["all_self_controls_passed"]
        for run in confirmation.values()
        for block in run.values()
    )
    formal = bool(
        all(prerequisites.values())
        and all(gates.values())
        and controls_passed
        and planted["passed"]
        and repeatability["passed"]
    )
    result = {
        "version": "bounded-poc-v2",
        "manifest_sha256": digest(OUT / "manifest.json"),
        "v1_immutable": immutable,
        "dataset": dataset_check,
        "known_controls": planted,
        "discovery": discovery,
        "confirmation": confirmation,
        "prerequisites": prerequisites,
        "gates": gates,
        "controls_passed": bool(controls_passed),
        "repeatability": repeatability,
        "formal_success": formal,
        "survivors": [selected["selection"]["best_head"]] if formal else [],
        "meaning": "All outcomes retained; formal status requires all preregistered gates.",
    }
    result["facts"] = numeric_facts(result)
    write("results.json", result)
    audit = {
        "all_numeric_checks_match": True,
        "frozen_inputs_match": True,
        "discovery_selection_recomputed": True,
        "readouts_recomputed": True,
        "decomposition_and_actual_bf16_patches_recomputed": True,
        "coalition_rescue_and_transfer_recomputed": True,
        "v1_immutable": immutable,
        "dataset": dataset_check,
        "replication_arrays_bit_identical": repeatability["passed"],
        "prerequisites": prerequisites,
        "gates": gates,
        "controls_passed": bool(controls_passed),
        "formal_success": formal,
        "raw_sha256": {path.name: digest(path) for path in sorted(OUT.glob("*.npz"))},
        "result_sha256": digest(OUT / "results.json"),
    }
    write("audit.json", audit)
    scientific_report(result)
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
