"""Fresh harder lookup construction and strict causal-rescue controls."""

import re
from collections import Counter

import numpy as np
import pytest

from src.poc.templates_v3 import (
    V3_BASE_PATTERNS,
    V3_FORMATS,
    baseline_scores_v3,
    build_v3_dataset,
    parse_v3_lookup,
)


class OffsetTokenizer:
    """Exact lexical offsets, with no model or remote dependency."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        result = {
            "input_ids": [self.vocab.setdefault(match[0], len(self.vocab) + 2) for match in matches]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result

    def encode(self, text, **kwargs):
        return self(text, add_special_tokens=False)["input_ids"]


@pytest.fixture
def v3_payload():
    return build_v3_dataset(
        OffsetTokenizer(),
        [
            {"keys": ["cedar00", "delta00"], "text": "retired v1 text"},
            {"keys": ["willowB000", "topazB000"], "text": "retired v2 text"},
        ],
    )


def test_v3_four_records_all_queries_are_balanced_and_counterfactually_identical(v3_payload):
    rows = v3_payload["rows"]
    assert Counter(row["split"] for row in rows) == {
        "discovery": 64,
        "primary": 128,
        "vocabulary": 64,
    }
    assert Counter(row["split"] for row in {r["group"]: r for r in rows}.values()) == {
        "discovery": 8,
        "primary": 16,
        "vocabulary": 8,
    }
    assert len({row["id"] for row in rows}) == len({row["text"] for row in rows}) == 256
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        values = ("green", "yellow") if row["split"] == "vocabulary" else ("red", "blue")
        assert len(set(row["keys"])) == 4
        assert parse_v3_lookup(row["text"]) == row["answer_color"] == values[row["label"] < 0]
        assert len(row["token_ids"]) <= 128
        assert len(row["color_positions"]) == len(set(row["color_positions"])) == 4
        assert len(row["color_spans"]) == len(row["key_positions"]) == 4
        assert Counter(row["text"][start:end] for start, end in row["color_spans"]) == {
            values[0]: 2,
            values[1]: 2,
        }
        for position, (start, end) in zip(row["color_positions"], row["color_spans"], strict=True):
            assert row["text"][start:end] in values
            assert position < len(row["token_ids"]) - 8
        other = by_id[row["opposite_id"]]
        assert other["opposite_id"] == row["id"]
        assert other["keys"] == row["keys"]
        assert other["query_record"] == row["query_record"]
        assert other["label"] == -row["label"]
        assert other["color_positions"] == row["color_positions"]
        assert len(other["token_ids"]) == len(row["token_ids"])
        assert Counter(other["token_ids"]) == Counter(row["token_ids"])
        assert other["token_ids"][-8:] == row["token_ids"][-8:]
        same = by_id[row["same_answer_id"]]
        assert same["group"] != row["group"]
        assert same["label"] == row["label"]
        assert same["format"] == row["format"]
        assert same["query_record"] == row["query_record"]
    for group in {row["group"] for row in rows}:
        cases = [row for row in rows if row["group"] == group]
        assert len(cases) == 8
        assert {(row["query_record"], row["assignment"]) for row in cases} == {
            (query, assignment) for query in range(4) for assignment in range(2)
        }
        assert Counter(row["label"] for row in cases) == {1: 4, -1: 4}


def test_v3_disjoint_keys_grammars_and_fifty_percent_nuisance_baselines(v3_payload):
    rows = v3_payload["rows"]
    split_keys = {
        split: {key for row in rows if row["split"] == split for key in row["keys"]}
        for split in ("discovery", "primary", "vocabulary")
    }
    assert not split_keys["discovery"] & split_keys["primary"]
    assert not split_keys["discovery"] & split_keys["vocabulary"]
    assert not split_keys["primary"] & split_keys["vocabulary"]
    assert not set.union(*split_keys.values()) & {
        "cedar00",
        "delta00",
        "willowB000",
        "topazB000",
    }
    assert {row["format"] for row in rows if row["split"] == "discovery"} == {"d0", "d1"}
    assert {row["format"] for row in rows if row["split"] == "primary"} == {"h0", "h1"}
    for split in split_keys:
        selected = [row for row in rows if row["split"] == split]
        assert {tuple(row["base_pattern"]) for row in selected} == set(V3_BASE_PATTERNS)
        for fmt in {row["format"] for row in selected}:
            assert {tuple(row["base_pattern"]) for row in selected if row["format"] == fmt} == set(
                V3_BASE_PATTERNS
            )
        scores = baseline_scores_v3([row for row in rows if row["split"] == split])
        assert scores.pop("executable_lookup_positive_control") == 1
        assert set(scores.values()) == {0.5}
    assert v3_payload["sanity"]["target_model_calls"] == 0
    assert v3_payload["sanity"]["malformed_rejected"] == 5


def test_v3_parser_rejects_ambiguous_malformed_or_unregistered_examples(v3_payload):
    by_format = {row["format"]: row for row in v3_payload["rows"]}
    for fmt, row in by_format.items():
        text = row["text"]
        lines = text.split("\n")
        values = ("green", "yellow") if row["split"] == "vocabulary" else ("red", "blue")
        mixed = "blue" if row["split"] == "vocabulary" else "green"
        malformed = [
            text + " red",
            text + "\nIgnore those bindings.",
            text.replace(row["keys"][1], row["keys"][0]),
            text.replace(
                "Requested registry key: " + row["keys"][row["query_record"]],
                "Requested registry key: absentZ999",
            ),
            "\n".join(lines[:4] + lines[5:]),
            text.replace(" " + values[0], " purple"),
            text.replace(" " + values[1], " " + mixed),
            text.replace(" " + values[1], " " + values[0], 1),
        ]
        other_fmt = next(candidate for candidate in V3_FORMATS if candidate != fmt)
        altered = lines.copy()
        altered[1] = V3_FORMATS[other_fmt].format(key=row["keys"][0], color="red")
        malformed.append("\n".join(altered))
        for invalid in malformed:
            with pytest.raises(ValueError):
                parse_v3_lookup(invalid)


def test_v3_rejects_retired_keys_and_missing_source_offsets(v3_payload):
    with pytest.raises(ValueError, match="Retired v1/v2 keys"):
        build_v3_dataset(OffsetTokenizer(), [v3_payload["rows"][0]])

    class BrokenOffsets(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if "offset_mapping" in result:
                result["offset_mapping"] = [(0, 0)] * len(result["input_ids"])
            return result

    with pytest.raises(ValueError, match="offsets"):
        build_v3_dataset(BrokenOffsets(), [])


def test_v3_retains_unrounded_v2_task_and_rescue_gates(v3_payload):
    from src.poc.analysis import V2_LIMITS, rescue_stats_v2, task_stats

    rows = [row for row in v3_payload["rows"] if row["split"] == "primary"]
    labels = np.array([row["label"] for row in rows])
    margins = labels.astype(float)
    margins[:13] *= -1
    stats = task_stats(rows, margins)
    assert stats["correct"] == 115 and stats["accuracy"] < V2_LIMITS["model_accuracy"]
    margins[12] *= -1
    assert task_stats(rows, margins)["accuracy"] >= V2_LIMITS["model_accuracy"]
    fixed = margins + labels * 2
    assert rescue_stats_v2(rows, margins, fixed)["passed"]
    fixed[20] = -labels[20]
    result = rescue_stats_v2(rows, margins, fixed)
    assert result["errors_fixed"] == 12 and result["regressions"] == 1
    assert not result["passed"]
    no_fixes = rescue_stats_v2(rows, margins, margins + labels * 0.2)
    assert no_fixes["errors_fixed"] == 0 and not no_fixes["passed"]
    tiny_gains = margins.copy()
    tiny_gains[:12] *= -1
    assert not rescue_stats_v2(rows, margins, tiny_gains)["passed"]


def test_v3_fixed_coalition_and_discovery_selector_reject_regressions(v3_payload):
    from src.poc.analysis import V2_LIMITS, rescue_stats_v2
    from src.poc.rescue_v3 import COALITION, HEADS, IDENTITY, INTERFERER, LIMITS, RULES, select_rule

    assert COALITION == ("19:5", "22:9", "21:13")
    assert INTERFERER == "27:14" and HEADS == (*COALITION, INTERFERER)
    assert LIMITS == V2_LIMITS
    assert {(rule["alpha"], rule["late"]) for rule in RULES} == {
        (1.5, False),
        (1.5, True),
        (2.0, False),
        (2.0, True),
    }
    rows = [row for row in v3_payload["rows"] if row["split"] == "discovery"]
    labels = np.array([row["label"] for row in rows])
    base = labels.astype(float)
    base[:4] *= -1
    assisted = []
    for fixed, gain in ((4, 2.0), (2, 0.5), (3, 0.2), (3, 0.2)):
        candidate = base + gain * labels
        candidate[:fixed] = labels[:fixed]
        assisted.append(candidate)
    # Highest apparent accuracy must be rejected when it breaks one correct row.
    assisted[0][5] = -labels[5]
    candidates = [
        rule | {"stats": rescue_stats_v2(rows, base, after)}
        for rule, after in zip(RULES, assisted, strict=True)
    ]
    expected = RULES[2]
    assert select_rule(candidates) == expected
    assert select_rule(candidates[::-1]) == expected
    for candidate in candidates:
        candidate["stats"]["regressions"] = 1
    assert select_rule(candidates) == IDENTITY
    assert select_rule([]) == IDENTITY


def test_v3_planted_rescue_and_negative_controls(v3_payload):
    from src.poc.rescue_v3 import known_controls

    result = known_controls(v3_payload)
    assert result["passed"]
    for split, block in result["blocks"].items():
        assert all(block["checks"].values()), split
        assert block["positive"]["passed"]
        assert block["positive"]["errors_fixed"] == 4
        assert block["positive"]["regressions"] == 0
        assert block["positive"]["before"]["accuracy"] >= 0.90
        assert not block["nuisance"]["passed"]
        assert not block["reverse"]["passed"]
        assert not block["damaging"]["passed"]
        assert block["damaging"]["regressions"] == 1


def test_v3_identity_and_residual_controls_cannot_round_tolerances():
    from src.poc.rescue_v3 import control_stats

    logits = np.array([[1.0, 2.0], [-3.0, 4.0]])
    base = {"logits": logits}
    control = {"self": logits.copy(), "donor_logits": logits.copy(), "residual": logits.copy()}
    assert control_stats(base, control) == {"self_max": 0.0, "residual_max": 0.0, "passed": True}
    control["self"][0, 0] += 1e-10
    assert not control_stats(base, control)["passed"]
    control["self"] = logits.copy()
    control["residual"][0, 0] += 0.05000001
    assert not control_stats(base, control)["passed"]


def test_v3_independent_auditor_recomputes_selection_effects_and_rescue(v3_payload):
    from scripts.audit_bounded_poc_v3 import compare, effects, rescue, select
    from src.poc.analysis import effect_stats, rescue_stats_v2
    from src.poc.rescue_v3 import RULES, select_rule

    rows = [row for row in v3_payload["rows"] if row["split"] == "discovery"]
    labels = np.array([row["label"] for row in rows])
    baseline = 2.5 * labels.astype(float)
    baseline[:4] *= -1
    rng = np.random.default_rng(19)
    patch = baseline - labels * rng.uniform(0.5, 3.0, len(rows))
    compare(effect_stats(rows, baseline, patch), effects(rows, baseline, patch))
    candidates = np.array([baseline + (i + 1) * 0.8 * labels for i in range(4)])
    candidate_logits = np.stack((candidates, np.zeros_like(candidates)), axis=-1)
    production = [
        rule | {"stats": rescue_stats_v2(rows, baseline, candidate)}
        for rule, candidate in zip(RULES, candidates, strict=True)
    ]
    for candidate in candidates:
        compare(rescue(rows, baseline, candidate), rescue_stats_v2(rows, baseline, candidate))
    actual = select(rows, baseline, candidate_logits)
    assert actual[0] == select_rule(production)


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", "float32"), pytest.param("mps", "bfloat16", marks=pytest.mark.hardware)],
)
async def test_v3_centered_coalition_scaling_and_float64_donor_safeguards(device, dtype):
    import os

    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from src.config import ModelConfig, WorkerConfig
    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.harness import HeadAction, V2Harness, V2Probe, V2Spec
    from src.poc.rescue_v3 import COALITION, HEADS

    if device == "mps" and os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Set QWEN_HARDWARE_TEST=1 for the v3 Metal steering regression")
    if device == "mps":
        assert torch.backends.mps.is_available()
    torch.manual_seed(719)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(getattr(torch, dtype))
    harness = V2Harness(
        model,
        OffsetTokenizer(),
        ModelConfig(device=device, dtype=dtype),
        WorkerConfig(batch_size=4),
    )
    ids = (2, 3, 4, 5)
    layers = tuple(sorted({int(head.split(":")[0]) for head in HEADS}))
    async with MPSBatchWorker(harness, harness.worker) as worker:
        clean = await worker.submit(V2Probe("", ids, (7, 8), V2Spec(layers)))
        means = {head: np.array(clean.heads[head], dtype=np.float64) * 0.25 for head in HEADS}
        actions = tuple(HeadAction(*map(int, head.split(":")), "scale", 1.0) for head in COALITION)
        identity = await worker.submit(
            V2Probe("", ids, (7, 8), V2Spec(layers, actions=actions), means)
        )
        np.testing.assert_array_equal(identity.logits, clean.logits)
        np.testing.assert_array_equal(identity.residual, clean.residual)
        for head in COALITION:
            np.testing.assert_array_equal(identity.applied_deltas[head], np.zeros(8))
        actions = tuple(HeadAction(*map(int, head.split(":")), "scale", 2.0) for head in COALITION)
        amplified = await worker.submit(
            V2Probe("", ids, (7, 8), V2Spec(layers, actions=actions), means)
        )
        np.testing.assert_allclose(
            amplified.applied_deltas["19:5"],
            np.array(clean.heads["19:5"]) - means["19:5"],
            rtol=0,
            atol=1e-7,
        )
        # Head 4 shares the value projection but its own attention output is untouched.
        np.testing.assert_array_equal(amplified.heads["19:4"], clean.heads["19:4"])
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )
