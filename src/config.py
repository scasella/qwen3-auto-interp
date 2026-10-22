from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictConfig):
    name: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    device: Literal["mps", "cpu"] = "mps"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    local_files_only: bool = False
    seed: int = 42


class WorkerConfig(StrictConfig):
    batch_size: int = Field(64, ge=1, le=64)
    attribution_batch_size: int = Field(1, ge=1, le=4)
    max_seq_len: int = Field(128, ge=4, le=128)
    queue_size: int = Field(256, ge=1)
    purge_every: int = Field(25, ge=1)


class CalibrationConfig(StrictConfig):
    units_per_pass: int = Field(128, ge=1, le=512)
    examples_per_band: int = Field(12, ge=1, le=100)
    min_tokens: int = Field(10000, ge=1000)
    max_documents: int = Field(10000, ge=1)
    duckdb_memory_limit: str = "2GB"


class SwarmConfig(StrictConfig):
    pipelines: int = Field(64, ge=1, le=64)
    max_agent_calls: int = Field(256, ge=1, le=256)
    circuit_workers: int = Field(16, ge=1, le=32)
    pi_workers: int = Field(2, ge=1, le=2)
    max_strikes: int = Field(3, ge=1, le=3)
    max_upstream_candidates: int = Field(16, ge=1, le=64)
    pi_every: int = Field(25, ge=1)


class Thresholds(StrictConfig):
    battery_pass_rate: float = Field(0.8, gt=0.5, le=1)
    paired_drop: float = Field(0.25, gt=0, le=1)
    min_span: float = Field(1e-6, gt=0)
    circuit_effect: float = Field(0.1, gt=0)
    gate_agreement: float = Field(0.8, gt=0.5, le=1)


class RoleConfig(StrictConfig):
    effort: Literal["low", "medium", "high", "xhigh"]


class CodexConfig(StrictConfig):
    executable: str = "codex"
    model: str = "gpt-6-luna"
    max_processes: int = Field(8, ge=1, le=32)
    tokens_per_minute: int = Field(1000000, ge=1000)
    output_token_reserve: int = Field(8192, ge=512)
    harness_token_reserve: int = Field(16000, ge=0)
    max_attempts: int = Field(3, ge=1, le=10)
    timeout_seconds: float = Field(600, gt=0)
    max_capture_bytes: int = Field(2097152, ge=1024)
    explorer: RoleConfig = RoleConfig(effort="high")
    falsifier: RoleConfig = RoleConfig(effort="low")
    tracer: RoleConfig = RoleConfig(effort="medium")
    pi: RoleConfig = RoleConfig(effort="xhigh")
    screen: RoleConfig = RoleConfig(effort="medium")
    auditor: RoleConfig = RoleConfig(effort="medium")
    poc_review: RoleConfig = RoleConfig(effort="high")


class Paths(StrictConfig):
    activations: Path = Path("data/activations.duckdb")
    circuits: Path = Path("data/circuits.sqlite")


class Settings(StrictConfig):
    model: ModelConfig = Field(default_factory=ModelConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    codex: CodexConfig = Field(default_factory=CodexConfig)
    paths: Paths = Field(default_factory=Paths)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Settings":
        path = Path(path).resolve()
        settings = cls.model_validate(yaml.safe_load(path.read_text()) or {})
        for field in ("activations", "circuits"):
            value = getattr(settings.paths, field)
            if not value.is_absolute():
                setattr(settings.paths, field, path.parent / value)
        return settings
