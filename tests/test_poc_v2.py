"""Controls for the preregistered v2 experiment, independent of its Qwen outcome."""

import re
from collections import Counter

import numpy as np
import pytest


class OffsetTokenizer:
    """Small deterministic lexical fixture with exact source spans."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        ids = [self.vocab.setdefault(match[0], len(self.vocab) + 2) for match in matches]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result

    def encode(self, text, **kwargs):
        return self(text, add_special_tokens=False)["input_ids"]


@pytest.fixture
def v2_payload():
    from src.poc.templates import build_v2_dataset

    return build_v2_dataset(
        OffsetTokenizer(),
        [{"keys": ["cedar00", "delta00"], "text": "retired v1 template text"}],
    )


def test_v2_dataset_fresh_splits_exact_counterfactuals_and_source_spans(v2_payload):
    from src.poc.templates import parse_v2_lookup

    rows = v2_payload["rows"]
    assert Counter(row["split"] for row in rows) == {
        "discovery": 64,
        "primary": 128,
        "vocabulary": 64,
    }
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows) == 256
    split_keys = {
        split: {key for row in rows if row["split"] == split for key in row["keys"]}
        for split in ("discovery", "primary", "vocabulary")
    }
    assert not split_keys["discovery"] & split_keys["primary"]
    assert not split_keys["discovery"] & split_keys["vocabulary"]
    assert not split_keys["primary"] & split_keys["vocabulary"]
    assert not set.union(*split_keys.values()) & {"cedar00", "delta00"}
    discovery_formats = {row["format"] for row in rows if row["split"] == "discovery"}
    held_formats = {row["format"] for row in rows if row["split"] != "discovery"}
    assert len(discovery_formats) == len(held_formats) == 2
    assert not discovery_formats & held_formats
    assert len({row["text"] for row in rows}) == 256
    for row in rows:
        expected_values = ("green", "yellow") if row["split"] == "vocabulary" else ("red", "blue")
        assert parse_v2_lookup(row["text"]) == expected_values[row["label"] < 0]
        donor = by_id[row["opposite_id"]]
        assert donor["split"] == row["split"]
        assert donor["keys"] == row["keys"]
        assert donor["query_record"] == row["query_record"]
        assert donor["label"] == -row["label"]
        assert len(donor["token_ids"]) == len(row["token_ids"]) <= 128
        assert Counter(donor["token_ids"]) == Counter(row["token_ids"])
        assert donor["token_ids"][-8:] == row["token_ids"][-8:]
        assert donor["color_positions"] == row["color_positions"]
        assert len(set(row["color_positions"])) == 2
        for position, (start, end) in zip(row["color_positions"], row["color_spans"]):
            assert row["text"][start:end] in expected_values
            assert 0 <= position < len(row["token_ids"]) - 8
        # Crossing both queries and assignments rules out record-order shortcuts.
        group = [other for other in rows if other["group"] == row["group"]]
        assert {(r["query_record"], r["assignment"]) for r in group} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }


def test_v2_parser_rejects_ambiguous_or_wrong_queries(v2_payload):
    from src.poc.templates import parse_v2_lookup

    row = v2_payload["rows"][0]
    assert parse_v2_lookup(row["text"]) in {"red", "blue"}
    with pytest.raises(ValueError):
        parse_v2_lookup(row["text"].replace(row["keys"][1], row["keys"][0]))
    with pytest.raises(ValueError):
        parse_v2_lookup(row["text"] + "\nAnother answer: blue")


def test_path_decomposition_source_masks_and_qk_ov_interaction():
    import torch

    from src.poc.harness import path_delta

    # Color occupies sources 1 and 3. Donor values at other sources differ hugely:
    # they must have no effect under the color-only source intervention.
    pc = torch.tensor([[0.1, 0.5, 0.1, 0.3], [0.0, 0.2, 0.0, 0.8]])
    pd = torch.tensor([[0.2, 0.2, 0.2, 0.4], [0.0, 0.7, 0.0, 0.3]])
    vc = torch.tensor([[[1.0, 0], [4, 2], [3, 1], [-4, -2]]] * 2)
    vd = torch.tensor([[[1000.0, 99], [-4, -2], [-900, 19], [4, 2]]] * 2)
    mask = torch.tensor([[False, True, False, True]] * 2)
    terms = pd[:, :, None] * vd - pc[:, :, None] * vc
    color = path_delta("source_color", pc, vc, pd, vd, mask)
    other = path_delta("source_other", pc, vc, pd, vd, mask)
    torch.testing.assert_close(color, (terms * mask[:, :, None]).sum(1))
    torch.testing.assert_close(other, (terms * (~mask)[:, :, None]).sum(1))
    full = path_delta("full", pc, vc, pd, vd, mask)
    torch.testing.assert_close(color + other, full)
    qk = path_delta("qk", pc, vc, pd, vd, mask)
    ov = path_delta("ov_all", pc, vc, pd, vd, mask)
    interaction = path_delta("interaction", pc, vc, pd, vd, mask)
    torch.testing.assert_close(full, qk + ov + interaction)
    torch.testing.assert_close(
        ov,
        path_delta("ov_color", pc, vc, pd, vd, mask) + path_delta("ov_other", pc, vc, pd, vd, mask),
    )
    # Altering the non-color donor values cannot alter the color-only result.
    donor_without_nuisance = vd.clone()
    donor_without_nuisance[:, ~mask[0]] = -123456
    torch.testing.assert_close(
        color, path_delta("source_color", pc, vc, pd, donor_without_nuisance, mask)
    )
    for mode in (
        "source_color",
        "source_other",
        "ov_all",
        "ov_color",
        "ov_other",
        "qk",
        "qkov",
        "full",
        "interaction",
    ):
        assert torch.equal(path_delta(mode, pc, vc, pc, vc, mask), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="shape"):
        path_delta("full", pc, vc, pd[:, :2], vd, mask)
    with pytest.raises(ValueError, match="Unknown"):
        path_delta("unsupported", pc, vc, pd, vd, mask)


def _v2_harness(*, device="cpu", dtype="float32", paths=False):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from src.config import ModelConfig, WorkerConfig
    from src.poc.harness import V2Harness

    torch.manual_seed(719)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=28 if paths else 3,
        num_attention_heads=16 if paths else 4,
        num_key_value_heads=8 if paths else 2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(getattr(torch, dtype))
    return V2Harness(
        model,
        OffsetTokenizer(),
        ModelConfig(device=device, dtype=dtype),
        WorkerConfig(batch_size=4),
    )


async def test_v2_coalition_identity_and_donors_are_isolated_at_final_token():
    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.harness import HeadAction, V2Probe, V2Spec

    harness = _v2_harness()
    ids = (2, 3, 4, 5)
    layers = (0, 1, 2)
    coalition = ((0, 1), (1, 2), (2, 0))
    async with MPSBatchWorker(harness, harness.worker) as worker:
        clean = await worker.submit(V2Probe("", ids, (7, 8), V2Spec(layers)))
        donor = await worker.submit(V2Probe("", (5, 4, 3, 2), (7, 8), V2Spec(layers)))
        actions = tuple(HeadAction(layer, head) for layer, head in coalition)
        identity = await worker.submit(
            V2Probe("", ids, (7, 8), V2Spec(layers, actions=actions), clean.heads)
        )
        np.testing.assert_array_equal(identity.logits, clean.logits)
        np.testing.assert_array_equal(identity.residual, clean.residual)
        for delta in identity.applied_deltas.values():
            np.testing.assert_array_equal(delta, np.zeros(harness.head_dim))
        # A same-layer two-head coalition must assign each individual donor and
        # leave its GQA-sharing neighbor untouched at that layer.
        actions = (HeadAction(1, 1), HeadAction(1, 2))
        changed = await worker.submit(
            V2Probe("", ids, (7, 8), V2Spec(layers, actions=actions), donor.heads)
        )
        for head in (1, 2):
            np.testing.assert_array_equal(changed.heads[f"1:{head}"], donor.heads[f"1:{head}"])
        for head in (0, 3):
            np.testing.assert_array_equal(changed.heads[f"1:{head}"], clean.heads[f"1:{head}"])
        for head in range(4):
            np.testing.assert_array_equal(changed.heads[f"0:{head}"], clean.heads[f"0:{head}"])
        # numpy float64 arrays must be explicitly cast before creating MPS tensors.
        residual = await worker.submit(
            V2Probe(
                "",
                ids,
                (7, 8),
                V2Spec(patch_residual=True),
                residual_donor=np.array(donor.residual, dtype=np.float64),
            )
        )
        np.testing.assert_array_equal(residual.logits, donor.logits)
        for invalid in ({"1:2": [1.0]}, {"1:2": [float("nan")] * 8}, {}):
            with pytest.raises(ValueError, match="donor"):
                await worker.submit(
                    V2Probe("", ids, (7, 8), V2Spec(actions=(HeadAction(1, 2),)), invalid)
                )
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in harness.model.modules()
    )
    with pytest.raises(RuntimeError, match="MPSBatchWorker"):
        harness._execute([V2Probe("", ids, (7, 8), V2Spec())], object())


async def _check_paths_and_padding(harness, *, reconstruction_atol):
    import torch

    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.harness import HeadAction, V2Probe, V2Spec, path_delta

    short, long = (2, 3, 4, 5), (5, 2, 6, 4, 3, 7)
    colors = (1, 2)
    capture = V2Spec((18, 19, 20), paths=True)
    modes = (
        "source_color",
        "source_other",
        "ov_all",
        "ov_color",
        "ov_other",
        "qk",
        "qkov",
        "full",
        "interaction",
    )
    async with MPSBatchWorker(harness, harness.worker) as worker:
        # The shorter row must never expose a padded position as an attention source.
        base, long_base = await worker.map(
            [V2Probe("", ids, (7, 8), capture, color_positions=colors) for ids in (short, long)]
        )
        assert len(base.pattern) == len(base.values) == len(short)
        assert len(long_base.pattern) == len(long)
        assert sum(base.pattern) == pytest.approx(1.0, abs=2e-6)
        np.testing.assert_allclose(
            np.array(base.pattern) @ np.array(base.values), base.reconstructed, atol=1e-7
        )
        np.testing.assert_allclose(base.reconstructed, base.heads["19:5"], atol=reconstruction_atol)
        alone = await worker.submit(V2Probe("", short, (7, 8), capture, color_positions=colors))
        np.testing.assert_allclose(alone.pattern, base.pattern, atol=reconstruction_atol)
        np.testing.assert_allclose(alone.logits, base.logits, atol=reconstruction_atol)
        # Exact identity is compared at an identical execution shape; the padding
        # comparison above separately checks shape-dependent numerical stability.
        base = alone
        for mode in modes:
            result = await worker.submit(
                V2Probe(
                    "",
                    short,
                    (7, 8),
                    V2Spec((19,), paths=True, actions=(HeadAction(19, 5, mode),)),
                    pattern_donor=np.array(base.pattern, dtype=np.float64),
                    values_donor=np.array(base.values, dtype=np.float64),
                    color_positions=colors,
                )
            )
            np.testing.assert_array_equal(result.logits, base.logits)
            np.testing.assert_array_equal(result.applied_deltas["19:5"], np.zeros(8))
        # A changed source-specific donor acts only on query head 5, despite
        # query heads 4 and 5 sharing a value projection in GQA.
        donor = await worker.submit(
            V2Probe("", (2, 6, 3, 5), (7, 8), capture, color_positions=colors)
        )
        changed = await worker.submit(
            V2Probe(
                "",
                short,
                (7, 8),
                V2Spec((19,), paths=True, actions=(HeadAction(19, 5, "source_color"),)),
                pattern_donor=donor.pattern,
                values_donor=donor.values,
                color_positions=colors,
            )
        )
        mask = torch.tensor([[i in colors for i in range(len(short))]])
        expected = path_delta(
            "source_color",
            torch.tensor([base.pattern]),
            torch.tensor([base.values]),
            torch.tensor([donor.pattern]),
            torch.tensor([donor.values]),
            mask,
        )
        np.testing.assert_allclose(
            changed.applied_deltas["19:5"], expected[0].numpy(), rtol=0, atol=1e-7
        )
        np.testing.assert_array_equal(changed.heads["19:4"], base.heads["19:4"])
        for head in range(16):
            if head != 5:
                np.testing.assert_array_equal(changed.heads[f"19:{head}"], base.heads[f"19:{head}"])
        for field, value in (("pattern_donor", [1.0]), ("values_donor", [[1.0]])):
            kwargs = dict(pattern_donor=donor.pattern, values_donor=donor.values)
            kwargs[field] = value
            with pytest.raises(ValueError, match="donor"):
                await worker.submit(
                    V2Probe(
                        "",
                        short,
                        (7, 8),
                        V2Spec(actions=(HeadAction(19, 5, "full"),)),
                        color_positions=colors,
                        **kwargs,
                    )
                )
        for invalid_positions in ((), (1, 1), (len(short),)):
            with pytest.raises(ValueError, match="source position|color positions"):
                await worker.submit(
                    V2Probe(
                        "",
                        short,
                        (7, 8),
                        V2Spec(actions=(HeadAction(19, 5, "source_color"),)),
                        pattern_donor=donor.pattern,
                        values_donor=donor.values,
                        color_positions=invalid_positions,
                    )
                )
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in harness.model.modules())


async def test_v2_real_tiny_transformer_qk_ov_paths_gqa_masks_and_padding():
    await _check_paths_and_padding(_v2_harness(paths=True), reconstruction_atol=1e-6)


@pytest.mark.hardware
async def test_v2_metal_identity_coalition_and_reconstructed_paths_are_exact():
    import os

    import torch

    from src.harness.mps_queue import MPSBatchWorker
    from src.poc.harness import HeadAction, V2Probe, V2Spec

    if os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Set QWEN_HARDWARE_TEST=1 for v2 Metal regression")
    assert torch.backends.mps.is_available()
    harness = _v2_harness(device="mps", dtype="bfloat16", paths=True)
    await _check_paths_and_padding(harness, reconstruction_atol=0.002)
    # The prior worker owns its harness. A fresh model/harness tests a whole-head
    # identity coalition independently of the decomposition's zero-delta path.
    harness = _v2_harness(device="mps", dtype="bfloat16")
    async with MPSBatchWorker(harness, harness.worker) as worker:
        ids = (2, 3, 4, 5)
        clean = await worker.submit(V2Probe("", ids, (7, 8), V2Spec((0, 1, 2))))
        actions = (HeadAction(0, 1), HeadAction(1, 2), HeadAction(2, 0))
        result = await worker.submit(
            V2Probe(
                "",
                ids,
                (7, 8),
                V2Spec(actions=actions),
                donors={
                    key: np.array(value, dtype=np.float64) for key, value in clean.heads.items()
                },
            )
        )
        np.testing.assert_array_equal(result.logits, clean.logits)
        np.testing.assert_array_equal(result.residual, clean.residual)


def test_v2_planted_mechanism_passes_all_gates_and_negative_controls_fail(v2_payload):
    from src.poc.controls import known_controls_v2

    controls = known_controls_v2(v2_payload)
    assert controls["passed"]
    assert set(controls["blocks"]) == {"discovery", "primary", "vocabulary"}
    for block in controls["blocks"].values():
        assert block["passed"]
        assert all(block["checks"].values())
        assert block["planted_lookup"]["statistics"]["readout_accuracy"] == 1
        assert block["decomposition"]["source_classification"] == "direct"
        assert block["decomposition"]["routing_classification"] == "ov"
        assert block["identity_error"] == 0
        assert block["coalition"]["fraction"] >= 0.35
        assert block["coalition"]["direction_rate"] == 1
        assert block["vocabulary_transfer"]["passed"]
        assert block["rescue"]["errors_fixed"] > 0
        assert block["rescue"]["regressions"] == 0
        assert not block["nuisance"]["passed"]
        assert block["nuisance"]["effect"]["direction_rate"] == 0
        assert not block["nuisance"]["decomposition"]["classifiable"]
        assert not block["reversed"]["passed"]
        assert block["reversed"]["readout"]["statistics"]["readout_accuracy"] == 0
        assert block["reversed"]["effect"]["direction_rate"] == 0
        assert not block["reversed"]["decomposition"]["classifiable"]


def test_v2_rescue_cannot_hide_regressions_or_round_task_accuracy(v2_payload):
    from src.poc.analysis import V2_LIMITS, effect_stats, rescue_stats_v2, task_stats

    rows = [row for row in v2_payload["rows"] if row["split"] == "primary"]
    labels = np.array([row["label"] for row in rows])
    margins = labels.astype(float)
    margins[:13] *= -1
    stats = task_stats(rows, margins)
    assert stats["correct"] == 115 and stats["total"] == 128
    assert stats["accuracy"] < V2_LIMITS["model_accuracy"]
    assert len(stats["error_ids"]) == 13
    rescued = margins + 2 * labels
    assert rescue_stats_v2(rows, margins, rescued)["passed"]
    # Fixing thirteen errors and introducing one error is still unacceptable.
    rescued[20] = -labels[20]
    failed = rescue_stats_v2(rows, margins, rescued)
    assert failed["errors_fixed"] == 13
    assert failed["regressions"] == 1
    assert not failed["passed"] and not failed["checks"]["no_regressions"]
    assert not rescue_stats_v2(rows, margins, margins)["passed"]
    reverse = effect_stats(rows, labels, 2 * labels)
    assert reverse["median"] == -1 and reverse["direction_rate"] == 0
    for invalid in (np.array([1.0]), np.full(128, np.nan)):
        with pytest.raises(ValueError, match="finite margin"):
            effect_stats(rows, invalid, invalid)
    bad_rows = [row | {"opposite_id": row["id"]} for row in rows]
    with pytest.raises(ValueError, match="opposite answers"):
        effect_stats(bad_rows, margins, margins)


def test_v2_builder_rejects_retired_keys_and_inaccurate_offsets(v2_payload):
    from src.poc.templates import build_v2_dataset

    reused = v2_payload["rows"][0]
    with pytest.raises(ValueError, match="Retired v1 keys"):
        build_v2_dataset(OffsetTokenizer(), [reused])

    class WrongOffsets(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            value = super().__call__(text, **kwargs)
            if "offset_mapping" in value:
                value["offset_mapping"] = [(0, 0)] * len(value["input_ids"])
            return value

    with pytest.raises(ValueError, match="offsets"):
        build_v2_dataset(WrongOffsets(), [])
    outputs = v2_payload["output_ids"]
    assert len(set(outputs["primary"])) == len(set(outputs["vocabulary"])) == 2
    assert not set(outputs["primary"]) & set(outputs["vocabulary"])
