import numpy as np
import pytest
from conftest import make_suite
from pydantic import ValidationError

from src.agents.circuit_tracer import classify_gate
from src.agents.falsifier import grade
from src.config import Thresholds
from src.harness.indexer import Quantiles
from src.schemas.contracts import Gate, UnitRef, all_units


def test_unit_inventory_and_constraints():
    assert len(all_units()) == 86464
    with pytest.raises(ValidationError):
        UnitRef(kind="attention", layer=0, index=16)


def test_exact_quantiles_and_three_bands(indexed, unit):
    q = indexed.quantiles(unit)
    expected = np.quantile(np.arange(10000), [0.001, 0.05, 0.5, 0.8, 0.95, 0.999])
    np.testing.assert_allclose([q.p001, q.p05, q.p50, q.p80, q.p95, q.p999], expected)
    assert q.count == 10000
    examples = indexed.examples(unit)
    assert {e["band"] for e in examples} == {"positive", "boundary", "inhibitory"}
    assert all(len(e["token_ids"]) == e["token_position"] + 1 for e in examples)
    assert all(len(e["token_pieces"]) == len(e["token_ids"]) for e in examples)


def test_batteries_and_fixed_predictions():
    suite = make_suite()
    q = Quantiles(10000, 0, -3, -2, 0, 1, 2, 3)
    result = grade(suite, [2] * 15 + [-2] * 30, q, Thresholds(), "discovery")
    assert result.passed and result.r_squared == 1
    constant = grade(suite, [2] * 45, q, Thresholds(), "discovery")
    assert not constant.passed and constant.r_squared is None
    with pytest.raises(ValueError, match="Non-finite"):
        grade(suite, [float("nan")] * 45, q, Thresholds(), "discovery")


def test_duplicate_or_unpaired_suite_rejected():
    data = make_suite().model_dump()
    data["cases"][15]["paired_positive_id"] = "not-a-real-case"
    with pytest.raises(ValidationError):
        type(make_suite()).model_validate(data)


def test_generated_probes_cannot_guess_tokenizer_offsets():
    data = make_suite().model_dump()
    data["cases"][0]["token_position"] = 3
    with pytest.raises(ValidationError):
        type(make_suite()).model_validate(data)


@pytest.mark.parametrize("gate,signature", [(Gate.AND, [0, 0, 0, 1]), (Gate.OR, [0, 1, 1, 1])])
def test_boolean_gate_requires_joint_table(gate, signature):
    q = Quantiles(10000, 0, -3, -2, 0, 1, 2, 3)
    table = {key: [value * 2] * 5 for key, value in zip(["00", "01", "10", "11"], signature)}
    assert classify_gate(table, q, 0.8) == (gate, 1.0)
    table["00"] = [0.5] * 5
    assert classify_gate(table, q, 0.8)[0] == Gate.MODULATES


def test_unary_inversion_and_inconclusive_band():
    q = Quantiles(10000, 0, -3, -2, 0, 1, 2, 3)
    assert classify_gate({"0": [2] * 5, "1": [-1] * 5}, q, 0.8)[0] == Gate.NOT
    assert classify_gate({"0": [0.2] * 5, "1": [0.7] * 5}, q, 0.8)[0] == Gate.MODULATES


def test_calibration_manifest_cannot_be_mixed(indexed):
    indexed.set_manifest({"corpus": "a"})
    indexed.set_manifest({"corpus": "a"})
    with pytest.raises(ValueError, match="provenance"):
        indexed.set_manifest({"corpus": "b"})
