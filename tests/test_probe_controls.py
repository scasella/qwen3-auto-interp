import inspect
from types import SimpleNamespace

import pytest
from conftest import make_hypothesis, make_suite

from src.agents.client import OutputContractError
from src.agents.falsifier import Falsifier
from src.config import Thresholds
from src.harness.qwen_model import ProbeResult


@pytest.mark.parametrize(
    "positive,stripped",
    [
        ("Readings[day + 1]", "Readings day + 1"),
        ("A long context ends here.", "Short."),
    ],
)
async def test_stripping_cannot_change_target_or_position(indexed, unit, positive, stripped):
    suite = make_suite()
    suite.cases[0].text = positive
    suite.cases[15].text = stripped
    from src.schemas.contracts import TestBatch as CaseBatch

    suite = CaseBatch(cases=suite.cases[:3] + suite.cases[15:18] + suite.cases[30:33])

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

        def decode(self, ids):
            return bytes(ids).decode()

    class Client:
        async def ask(self, role, schema, payload, *, validate_result=None):
            if validate_result:
                result = validate_result(suite)
                if inspect.isawaitable(result):
                    await result
            return suite

    class Worker:
        config = SimpleNamespace(max_seq_len=128)
        harness = SimpleNamespace(tokenizer=Tokenizer())
        calls = 0

        async def map(self, probes):
            self.calls += 1
            return [ProbeResult({unit.key: [0]}, [1], ["x"]) for _ in probes]

    worker = Worker()
    with pytest.raises(OutputContractError, match="target|position|length"):
        await Falsifier(Client(), worker, indexed, Thresholds()).evaluate(
            make_hypothesis(unit), "discovery", set()
        )
    assert worker.calls == 0


async def test_semantic_rejection_prevents_measurement(indexed, unit):
    from conftest import FakeHarness, FakeScientist

    class RejectingScientist(FakeScientist):
        async def _answer(self, role, schema, payload):
            if role == "auditor":
                return schema(
                    cases=[
                        {
                            "case_id": c["id"],
                            "valid": False,
                            "reason": "Does not remove the hypothesized relation",
                        }
                        for c in payload["cases"]
                    ]
                )
            return await super()._answer(role, schema, payload)

    async def forbidden_map(_):
        pytest.fail("Invalid semantic tests must not reach model execution")

    worker = SimpleNamespace(
        config=SimpleNamespace(max_seq_len=128), harness=FakeHarness(), map=forbidden_map
    )
    with pytest.raises(OutputContractError, match="semantic review"):
        await Falsifier(RejectingScientist(), worker, indexed, Thresholds()).evaluate(
            make_hypothesis(unit), "discovery", set()
        )


async def test_blinded_ranking_overrides_confidence(indexed, unit):
    import hashlib

    from conftest import FakeScientist

    from src.agents.explorer import Explorer
    from src.harness.indexer import split_evidence
    from src.schemas.contracts import Lens

    _, held = split_evidence(indexed.distribution(unit))
    held = sorted(held, key=lambda e: hashlib.sha256(("screen-v3/" + e["id"]).encode()).hexdigest())
    truth = {f"E{i:02}": e["band"] for i, e in enumerate(held)}

    class Scientist(FakeScientist):
        async def _answer(self, role, schema, payload):
            if role == "screen":
                assert all(
                    set(e) == {"id", "text", "target_token", "token_count"}
                    for e in payload["unlabeled_examples"]
                )
                assert all(e["id"].startswith("E") for e in payload["unlabeled_examples"])
                return schema(
                    predictions=[
                        {
                            "candidate": i,
                            "example_id": e["id"],
                            "band": truth[e["id"]] if i == 2 else "boundary",
                        }
                        for i in range(3)
                        for e in payload["unlabeled_examples"]
                    ]
                )
            answer = await super()._answer(role, schema, payload)
            if role == "explorer":
                answer.confidence = 0.01 if answer.lens == Lens.POLAR_AXIS_CONTRAST else 0.99
            return answer

    events = []
    candidates = await Explorer(Scientist(), indexed, lambda *args: events.append(args)).explore(
        unit, []
    )
    assert candidates[0].lens == Lens.POLAR_AXIS_CONTRAST
    boundary_count = sum(e["band"] == "boundary" for e in held)
    assert events[0][1]["correct"] == [boundary_count, boundary_count, len(held)]


def test_stripping_threshold_not_relaxed():
    from src.agents.falsifier import grade
    from src.harness.indexer import Quantiles

    result = grade(
        make_suite(),
        [2] * 15 + [0] * 30,
        Quantiles(10000, 0, -3, -2, 0, 1, 2, 3),
        Thresholds(),
        "discovery",
    )
    assert result.rates["OOD_POSITIVE"] == 1
    assert result.rates["SEMANTIC_STRIPPING"] == 0  # A 50% span drop still does not reach P05.
    assert not result.passed
