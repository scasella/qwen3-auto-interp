import numpy as np
import pytest

from src.harness.mps_queue import MPSBatchWorker
from src.poc.analysis import cross_validate, evaluate, fit_readout
from src.poc.controls import known_controls
from src.poc.harness import CausalHarness, CausalProbe, CausalSpec, cpu_snapshot
from src.poc.review import Review
from src.poc.templates import baseline_scores, build_dataset, parse_lookup


class WordTokenizer:
    """Template-control fixture; actual tokenizer checked again in preparation."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, **kwargs):
        import re

        return [self.vocab.setdefault(w, len(self.vocab)) for w in re.findall(r"\w+|[^\w\s]", text)]


def test_templates_and_planted_controls():
    rows = build_dataset(WordTokenizer())
    assert len(rows) == 192
    assert len({r["group"] for r in rows}) == 48
    assert all(parse_lookup(r["text"]) in {"red", "blue"} for r in rows)
    scores = baseline_scores([r for r in rows if r["split"] == "confirmation"])
    assert scores.pop("executable_lookup_positive_control") == 1
    assert set(scores.values()) == {0.5}
    control = known_controls(rows)
    assert control["passed"]
    assert control["planted_lookup_unit"]["statistics"]["readout_accuracy"] == 1
    assert control["reversed_explanation"]["statistics"]["readout_accuracy"] == 0
    assert not control["nuisance_only_unit"]["checks"]["causal_significance"]


def test_review_contract_has_no_optional_structured_output_fields():
    schema = Review.model_json_schema()
    assert set(schema["properties"]) == set(schema["required"])


@pytest.mark.hardware
def test_metal_snapshots_wait_without_reading_unfinished_bf16_transfer():
    import os

    import torch

    if os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Set QWEN_HARDWARE_TEST=1 for Metal transfer regression")
    assert torch.backends.mps.is_available()
    expected = torch.arange(131072, dtype=torch.float32).remainder(113).to(torch.bfloat16)
    source = expected.to("mps")
    copies = [cpu_snapshot(source.reshape(64, 16, 128)) for _ in range(25)]
    torch.mps.synchronize()
    for copy in copies:
        torch.testing.assert_close(copy.flatten(), expected.float(), rtol=0, atol=0)


def test_group_readout_and_causal_failure_cannot_be_overridden():
    rows = build_dataset(WordTokenizer())
    train = [r for r in rows if r["split"] == "discovery"]
    test = [r for r in rows if r["split"] == "confirmation"]
    x = np.array([[r["label"], 0] for r in train])
    assert cross_validate(x, train) == 1
    model = fit_readout(x, [r["label"] for r in train])
    y = np.array([r["label"] for r in test])
    result = evaluate(
        test,
        np.stack([y, y * 0], axis=1),
        model,
        4 * y,
        4 * y,
        4 * y,
        4 * y,
        4 * y,
        baseline_scores(test),
    )
    assert result["statistics"]["readout_accuracy"] == 1
    assert not result["passed"]
    assert not result["checks"]["patch_median"]


async def test_real_transformer_vector_patches_and_output_controls(tiny_harness):
    harness = CausalHarness(
        tiny_harness.model, tiny_harness.tokenizer, tiny_harness.config, tiny_harness.worker
    )
    rows = [(2, 3, 4), (5, 6)]

    def probe(ids, spec, donor=None):
        return CausalProbe("", ids, (7, 8), spec, donor)

    async with MPSBatchWorker(harness, harness.worker) as worker:
        base = await worker.map([probe(ids, CausalSpec((0, 1, 2))) for ids in rows])
        sham = await worker.submit(
            probe(rows[0], CausalSpec((0, 1, 2), (1, 2)), tuple(base[0].heads["1:2"]))
        )
        np.testing.assert_allclose(sham.logits, base[0].logits, atol=1e-7)
        changed = await worker.submit(
            probe(rows[0], CausalSpec((0, 1, 2), (1, 2)), tuple(base[1].heads["1:2"]))
        )
        np.testing.assert_allclose(changed.heads["1:2"], base[1].heads["1:2"])
        for h in (0, 1, 3):
            np.testing.assert_allclose(changed.heads[f"1:{h}"], base[0].heads[f"1:{h}"], atol=1e-7)
        np.testing.assert_allclose(changed.heads["0:0"], base[0].heads["0:0"], atol=1e-7)
        residual = await worker.submit(
            probe(rows[0], CausalSpec((), patch_residual=True), tuple(base[1].residual))
        )
        np.testing.assert_allclose(residual.logits, base[1].logits, atol=1e-7)
        with pytest.raises(ValueError, match="donor"):
            await worker.submit(probe(rows[0], CausalSpec((), (1, 2)), (1.0,)))
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in harness.model.modules())
    with pytest.raises(RuntimeError, match="MPSBatchWorker"):
        harness._execute([probe(rows[0], CausalSpec(()))], object())


@pytest.mark.hardware
async def test_metal_numpy_donor_patches(tiny_harness):
    import os

    import torch

    from src.config import ModelConfig

    if os.getenv("QWEN_HARDWARE_TEST") != "1":
        pytest.skip("Set QWEN_HARDWARE_TEST=1 for Metal patch integration")
    harness = CausalHarness(
        tiny_harness.model.to(torch.bfloat16),
        tiny_harness.tokenizer,
        ModelConfig(device="mps", dtype="bfloat16"),
        tiny_harness.worker,
    )
    async with MPSBatchWorker(harness, harness.worker) as worker:
        base = await worker.map(
            [
                CausalProbe("", (2, 3, 4), (7, 8), CausalSpec((0, 1, 2))),
                CausalProbe("", (5, 6, 7), (7, 8), CausalSpec((0, 1, 2))),
            ]
        )
        # np.load returns float64 in this experiment; never infer an MPS float64 donor.
        donor = tuple(np.array(base[1].residual, dtype=np.float64))
        patched = await worker.submit(
            CausalProbe("", (2, 3, 4), (7, 8), CausalSpec((), patch_residual=True), donor)
        )
        np.testing.assert_allclose(patched.logits, base[1].logits, atol=1e-6)
        head_donor = tuple(np.array(base[1].heads["1:2"], dtype=np.float64))
        patched = await worker.submit(
            CausalProbe("", (2, 3, 4), (7, 8), CausalSpec((1,), (1, 2)), head_donor)
        )
        np.testing.assert_array_equal(patched.heads["1:2"], head_donor)
