from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class UnitRef(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["mlp", "attention"]
    layer: int = Field(ge=0, lt=28)
    index: int = Field(ge=0, lt=3072)

    @model_validator(mode="after")
    def valid_head(self):
        if self.kind == "attention" and self.index >= 16:
            raise ValueError("Qwen3-0.6B has 16 query heads")
        return self

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.layer}:{self.index}"

    @property
    def order(self) -> tuple[int, int]:
        return self.layer, int(self.kind == "mlp")

    @classmethod
    def parse(cls, key: str) -> "UnitRef":
        kind, layer, index = key.split(":")
        return cls(kind=kind, layer=int(layer), index=int(index))


def all_units() -> list[UnitRef]:
    return [
        UnitRef(kind=kind, layer=layer, index=index)
        for layer in range(28)
        for kind, count in (("attention", 16), ("mlp", 3072))
        for index in range(count)
    ]


class Lens(StrEnum):
    GEOMETRIC_STRUCTURAL = "GEOMETRIC_STRUCTURAL"
    RELATIONAL_DYNAMIC = "RELATIONAL_DYNAMIC"
    POLAR_AXIS_CONTRAST = "POLAR_AXIS_CONTRAST"


class PrimitiveHypothesis(Contract):
    unit: UnitRef
    lens: Lens
    structural_rule: str = Field(min_length=20, max_length=2000)
    positive_prediction: str
    inhibitory_prediction: str
    boundary_prediction: str
    alternative_explanation: str
    falsification_condition: str
    evidence_ids: list[str] = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)


class Battery(StrEnum):
    OOD_POSITIVE = "OOD_POSITIVE"
    SEMANTIC_STRIPPING = "SEMANTIC_STRIPPING"
    INHIBITORY_POLE = "INHIBITORY_POLE"


class Counterfactual(Contract):
    id: str
    battery: Battery
    text: str = Field(min_length=1, max_length=1500)
    token_position: Literal[-1]
    paired_positive_id: str | None
    rationale: str


class TestSuite(Contract):
    per_battery: ClassVar[int] = 15
    cases: list[Counterfactual] = Field(min_length=45, max_length=45)

    @model_validator(mode="after")
    def balanced(self):
        ids = [c.id for c in self.cases]
        if len(set(ids)) != self.per_battery * 3:
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            raise ValueError(f"Every case ID must be unique; repeated IDs: {duplicates}")
        texts = [c.text for c in self.cases]
        if len(set(texts)) != self.per_battery * 3:
            duplicates = [c.id for c in self.cases if texts.count(c.text) > 1]
            raise ValueError(
                f"All {self.per_battery * 3} texts must differ; rewrite repeated text in cases: {duplicates}"
            )
        positives = {c.id for c in self.cases if c.battery == Battery.OOD_POSITIVE}
        for battery in Battery:
            if sum(c.battery == battery for c in self.cases) != self.per_battery:
                raise ValueError(f"Need exactly {self.per_battery} cases per battery")
        pairs = []
        for c in self.cases:
            if c.battery == Battery.SEMANTIC_STRIPPING:
                if c.paired_positive_id not in positives:
                    raise ValueError("Each stripping case needs a positive partner")
                pairs.append(c.paired_positive_id)
            elif c.paired_positive_id is not None:
                raise ValueError("Only stripping cases have a partner")
        if len(set(pairs)) != self.per_battery:
            raise ValueError("Stripping pairs must be one-to-one")
        return self


class TestBatch(TestSuite):
    per_battery: ClassVar[int] = 3
    cases: list[Counterfactual] = Field(min_length=9, max_length=9)


class CaseResult(Contract):
    case: Counterfactual
    activation: float
    passed: bool


class Evaluation(Contract):
    stage: Literal["discovery", "holdout"]
    passed: bool
    rates: dict[str, float]
    r_squared: float | None
    results: list[CaseResult]


class BandPrediction(Contract):
    candidate: int = Field(ge=0, le=2)
    example_id: str
    band: Literal["positive", "boundary", "inhibitory"]


class HypothesisScreen(Contract):
    predictions: list[BandPrediction]


class CaseReview(Contract):
    case_id: str
    valid: bool
    reason: str


class SuiteReview(Contract):
    cases: list[CaseReview] = Field(min_length=45, max_length=45)


class BatchReview(SuiteReview):
    cases: list[CaseReview] = Field(min_length=9, max_length=9)


class Gate(StrEnum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    MODULATES = "MODULATES"


class CircuitInterpretation(Contract):
    gate: Gate
    explanation: str
    confounds: list[str]


class CircuitEvidence(Contract):
    id: str
    sources: list[UnitRef] = Field(min_length=1, max_length=2)
    target: UnitRef
    gate: Gate
    attribution: dict[str, float]
    grad_act: dict[str, float]
    truth_table: dict[str, list[float]]
    zero_ablation: dict[str, list[float]]
    mean_ablation: dict[str, list[float]]
    baseline: list[float]
    effect: float
    agreement: float
    interpretation: CircuitInterpretation


class MacroCircuit(Contract):
    name: str
    units: list[str]
    explanation: str


class PIReport(Contract):
    clusters: list[MacroCircuit]
    prioritized_units: list[str]
    rationale: str
