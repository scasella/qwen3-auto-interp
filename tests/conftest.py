import inspect

import pytest
import torch

from src.config import ModelConfig, Settings, WorkerConfig
from src.harness.indexer import ActivationIndex
from src.harness.qwen_model import ProbeResult, QwenHarness
from src.schemas.contracts import (
    Battery,
    Counterfactual,
    Lens,
    PrimitiveHypothesis,
    TestSuite,
    UnitRef,
)


@pytest.fixture
def unit():
    return UnitRef(kind="mlp", layer=0, index=0)


def make_suite(stage="discovery"):
    cases = []
    for battery in Battery:
        for i in range(15):
            cases.append(
                Counterfactual(
                    id=f"{dict(OOD_POSITIVE='P', SEMANTIC_STRIPPING='S', INHIBITORY_POLE='N')[battery.value]}{i + 1:02}",
                    battery=battery,
                    text=f"{stage}:{battery.value}:{i}".ljust(110, "x") + f"{i:02}",
                    token_position=-1,
                    paired_positive_id=f"P{i + 1:02}"
                    if battery == Battery.SEMANTIC_STRIPPING
                    else None,
                    rationale="A matched structural intervention",
                )
            )
    return TestSuite(cases=cases)


def make_hypothesis(unit, evidence_ids=None, lens=Lens.GEOMETRIC_STRUCTURAL):
    return PrimitiveHypothesis(
        unit=unit,
        lens=lens,
        structural_rule="A nested dependency closes in reverse order",
        positive_prediction="Nested closure",
        inhibitory_prediction="Unmatched opening",
        boundary_prediction="Partially nested closure",
        alternative_explanation="Delimiter token",
        falsification_condition="Removing nesting preserves the measured response",
        evidence_ids=evidence_ids or ["positive", "boundary", "inhibitory"],
        confidence=0.7,
    )


@pytest.fixture
def indexed(tmp_path, unit):
    index = ActivationIndex(tmp_path / "activations.duckdb")
    # Many contexts, with tails distributed across documents.
    for start in range(0, 10000, 100):
        values = list(map(float, range(start, start + 100)))
        result = ProbeResult({unit.key: values}, [1] * 100, ["x"] * 100)
        index.add_batch([f"calibration {start}"], [result], [unit])
    index.finalize([unit], 12, 10000)
    yield index
    index.close()


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(c) % 29 for c in text]

    def decode(self, ids):
        return " ".join(str(t) for t in ids)

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + ("T" if kwargs["enable_thinking"] else "F")


@pytest.fixture
def tiny_harness():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(cfg)
    return QwenHarness(
        model,
        TinyTokenizer(),
        ModelConfig(device="cpu", dtype="float32"),
        WorkerConfig(batch_size=4),
    )


class FakeHarness:
    revision = "test-revision"

    class tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(text.encode())

        @staticmethod
        def decode(ids):
            return bytes(ids).decode()

    def __init__(self, fail_holdout=False, fail_discovery=False):
        self.claimed = None
        self.calls = []
        self.fail_holdout = fail_holdout
        self.fail_discovery = fail_discovery
        self.purges = 0

    def claim(self, owner):
        if self.claimed:
            raise RuntimeError("Already claimed")
        self.claimed = owner

    def _execute(self, probes, owner):
        assert owner is self.claimed
        self.calls.append(probes)
        results = []
        for p in probes:
            if p.text == "error":
                raise ValueError("invalid probe")
            high = "OOD_POSITIVE" in p.text
            value = 10000.0 if high else 0.0
            if (self.fail_holdout and p.text.startswith("holdout")) or self.fail_discovery:
                value = 0.0
            results.append(ProbeResult({u.key: [value] for u in p.spec.units}, [1], ["x"]))
        return results

    def purge(self):
        self.purges += 1

    def memory_bytes(self):
        return 0


class FakeScientist:
    def __init__(self):
        self.calls = []
        self.suites = 0

    async def ask(self, role, schema, payload, *, validate_result=None):
        result = await self._answer(role, schema, payload)
        if validate_result is not None:
            validation = validate_result(result)
            if inspect.isawaitable(validation):
                await validation
        return result

    async def _answer(self, role, schema, payload):
        self.calls.append((role, payload))
        if role == "explorer":
            examples = payload["distribution"]["examples"]
            ids = [
                next(e["id"] for e in examples if e["band"] == b)
                for b in ("positive", "boundary", "inhibitory")
            ]
            return make_hypothesis(
                UnitRef.model_validate(payload["unit"]), ids, Lens(payload["lens"])
            )
        if role == "falsifier":
            self.suites += 1
            suite = make_suite(f"{payload['stage']}-{self.suites}")
            if "case_numbers" in payload:
                return schema(
                    cases=[c for c in suite.cases if int(c.id[1:]) in payload["case_numbers"]]
                )
            return suite
        if role == "screen":
            return schema(
                predictions=[
                    {"candidate": i, "example_id": e["id"], "band": "boundary"}
                    for i in range(3)
                    for e in payload["unlabeled_examples"]
                ]
            )
        if role == "auditor":
            return schema(
                cases=[
                    {"case_id": c["id"], "valid": True, "reason": "Valid fixture"}
                    for c in payload["cases"]
                ]
            )
        if role == "pi":
            return schema(clusters=[], prioritized_units=[], rationale="No additional priority")
        if role == "tracer":
            return schema(
                gate=payload["deterministic_gate"],
                explanation="Measured local effect",
                confounds=[],
            )
        raise AssertionError(role)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.paths.activations = tmp_path / "activations.duckdb"
    s.paths.circuits = tmp_path / "circuits.sqlite"
    s.swarm.pipelines = 2
    s.swarm.circuit_workers = 1
    return s
