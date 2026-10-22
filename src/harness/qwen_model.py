"""Hook the post-SwiGLU intermediate and each query head's pre-o_proj output.

Only MPSBatchWorker may call _execute. No vocabulary-sized logits are materialized.
Attention observations are signed component means, not inherently semantic axes.
"""

import threading
from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import ModelConfig, WorkerConfig
from src.schemas.contracts import UnitRef


@dataclass(frozen=True)
class ProbeSpec:
    units: tuple[UnitRef, ...]
    clamps: tuple[tuple[UnitRef, float], ...] = ()
    attribution_target: UnitRef | None = None
    attribution_baselines: tuple[tuple[UnitRef, float], ...] = ()
    all_tokens: bool = False
    thinking: bool | None = None
    allow_truncate: bool = False


@dataclass(frozen=True)
class Probe:
    text: str
    spec: ProbeSpec
    position: int = -1
    token_ids: tuple[int, ...] | None = None


@dataclass
class ProbeResult:
    values: dict[str, list[float]]
    token_ids: list[int]
    token_pieces: list[str]
    attribution: dict[str, float] = field(default_factory=dict)
    grad_act: dict[str, float] = field(default_factory=dict)


class _TargetReached(Exception):
    pass


class QwenHarness:
    def __init__(self, model, tokenizer, config: ModelConfig, worker: WorkerConfig):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.config, self.worker = config, worker
        self.device = torch.device(config.device)
        self._owner: int | None = None
        self._claim: object | None = None
        self.revision = getattr(model.config, "_commit_hash", None) or config.revision
        self.head_dim = model.config.head_dim
        self.model.to(self.device)

    @classmethod
    def load(cls, config: ModelConfig, worker: WorkerConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable. Use --device cpu only for development tests.")
        torch.manual_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.name, revision=config.revision, local_files_only=config.local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            local_files_only=config.local_files_only,
            torch_dtype=getattr(torch, config.dtype),
            attn_implementation="sdpa",
        )
        expected = dict(
            num_hidden_layers=28,
            hidden_size=1024,
            intermediate_size=3072,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        )
        for name, value in expected.items():
            if getattr(model.config, name, None) != value:
                raise ValueError(f"Unexpected Qwen architecture: {name}")
        return cls(model, tokenizer, config, worker)

    def claim(self, owner: object):
        if self._claim is not None:
            raise RuntimeError("Harness already belongs to a worker")
        self._claim = owner

    def _slice(self, unit: UnitRef):
        if unit.kind == "mlp":
            return slice(unit.index, unit.index + 1)
        return slice(unit.index * self.head_dim, (unit.index + 1) * self.head_dim)

    def _encode(self, probe: Probe) -> list[int]:
        if probe.token_ids is not None:
            ids = list(probe.token_ids)
        else:
            text = probe.text
            if probe.spec.thinking is not None:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=probe.spec.thinking,
                )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Empty token sequence")
        if len(ids) > self.worker.max_seq_len:
            if not probe.spec.allow_truncate:
                raise ValueError("Probe exceeds max_seq_len; shorten it, do not silently truncate")
            ids = ids[: self.worker.max_seq_len]
        return ids

    def _execute(self, probes: list[Probe], owner: object) -> list[ProbeResult]:
        if owner is not self._claim:
            raise RuntimeError("All model execution must pass through MPSBatchWorker")
        ident = threading.get_ident()
        if self._owner is None:
            self._owner = ident
        if self._owner != ident:
            raise RuntimeError("Model execution moved off its single consumer thread")
        if not probes or len(probes) > self.worker.batch_size:
            raise ValueError("Invalid micro-batch size")
        spec = probes[0].spec
        if any(p.spec != spec for p in probes):
            raise ValueError("Only identical hook plans can share a micro-batch")
        ids = [self._encode(p) for p in probes]
        lengths = [len(row) for row in ids]
        positions = [n - 1 if p.position == -1 else p.position for p, n in zip(probes, lengths)]
        if any(pos < 0 or pos >= n for pos, n in zip(positions, lengths)):
            raise ValueError("Target token position is outside the encoded sequence")
        width = max(lengths)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        input_ids = torch.tensor(
            [row + [pad] * (width - len(row)) for row in ids], device=self.device
        )
        mask = (
            torch.arange(width, device=self.device)[None, :]
            < torch.tensor(lengths, device=self.device)[:, None]
        )
        rows = torch.arange(len(probes), device=self.device)
        pos = torch.tensor(positions, device=self.device)
        observed, patch, grad_act = {}, {}, {}
        objective = []
        handles = []
        baselines = dict(spec.attribution_baselines)
        clamps = dict(spec.clamps)
        target = spec.attribution_target
        if target and any(u.order >= target.order for u in baselines):
            raise ValueError("Attribution sources must precede their target")
        wanted = set(spec.units) | set(clamps) | set(baselines)
        if target:
            wanted.add(target)
        groups = {}
        for u in sorted(wanted, key=lambda u: u.key):
            groups.setdefault((u.layer, u.kind), []).append(u)

        def cpu(tensor):
            # Only reduced observations leave the GPU; graph tensors stay in autograd.
            return tensor.detach().to("cpu", non_blocking=True).float()

        def make_hook(units):
            def hook(module, args):
                x = args[0]
                local_sources = [u for u in units if u in baselines]
                if local_sources and not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                local_clamps = [u for u in units if u in clamps]
                if local_clamps:
                    x = x.clone()
                    for unit in local_clamps:
                        section = self._slice(unit)
                        x[..., section] = torch.where(
                            mask[..., None], clamps[unit], x[..., section]
                        )
                for unit in units:
                    values = x[..., self._slice(unit)].float().mean(-1)
                    if unit in spec.units:
                        observed[unit.key] = cpu(values if spec.all_tokens else values[rows, pos])
                    if unit == target:
                        objective.append(values[rows, pos].sum())
                if local_sources:
                    # Keep only the selected source coordinates for grad*act and grad*delta.
                    activations = {u: x[..., self._slice(u)].detach() for u in local_sources}

                    def backward(gradient):
                        for u, act in activations.items():
                            grad = gradient[..., self._slice(u)].float()
                            valid = mask[..., None]
                            grad_act[u.key] = cpu((grad * act * valid).sum((1, 2)))
                            patch[u.key] = cpu((grad * (baselines[u] - act) * valid).sum((1, 2)))

                    handles.append(x.register_hook(backward))
                if target in units:
                    raise _TargetReached()
                return (x, *args[1:])

            return hook

        try:
            for (layer, kind), units in groups.items():
                block = self.model.model.layers[layer]
                module = block.mlp.down_proj if kind == "mlp" else block.self_attn.o_proj
                handles.append(module.register_forward_pre_hook(make_hook(units)))
            with torch.set_grad_enabled(target is not None):
                try:
                    self.model.model(
                        input_ids=input_ids, attention_mask=mask.long(), use_cache=False
                    )
                except _TargetReached:
                    pass
                if target is not None:
                    if not objective or not objective[0].requires_grad:
                        raise RuntimeError("No differentiable path to attribution target")
                    objective[0].backward()
            if self.device.type == "mps":
                torch.mps.synchronize()  # Complete non-blocking CPU copies before reading.
            results = []
            for i, row in enumerate(ids):
                values = {
                    key: (value[i, : len(row)].tolist() if spec.all_tokens else [float(value[i])])
                    for key, value in observed.items()
                }
                arrays = list(values.values())
                if any(not np.isfinite(a).all() for a in arrays):
                    raise RuntimeError("Non-finite activation encountered")
                results.append(
                    ProbeResult(
                        values,
                        row,
                        [self.tokenizer.decode([t]) for t in row],
                        {key: float(value[i]) for key, value in patch.items()},
                        {key: float(value[i]) for key, value in grad_act.items()},
                    )
                )
            return results
        finally:
            for handle in handles:
                handle.remove()
            objective.clear()
            self.model.zero_grad(set_to_none=True)

    def purge(self):
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def memory_bytes(self) -> int:
        return torch.mps.current_allocated_memory() if self.device.type == "mps" else 0
