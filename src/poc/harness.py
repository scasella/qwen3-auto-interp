"""Final-position measured-vector patches and two-token output measurements.

Uses the existing single-consumer queue. At most a few MB of selected final-position
vectors are copied to CPU per batch. No sequence-wide activation cache or full-vocab
logit tensor is allocated. Scalar reductions and snapshots are detached immediately.
"""

import threading
from dataclasses import dataclass

import numpy as np
import torch

from src.harness.qwen_model import QwenHarness


def cpu_snapshot(x):
    # Convert on the source device. Converting a nonblocking CPU destination before
    # synchronization can read its unfinished BF16 transfer and freeze garbage.
    return x.detach().float().to("cpu", non_blocking=True)


@dataclass(frozen=True)
class CausalSpec:
    layers: tuple[int, ...]
    patch_head: tuple[int, int] | None = None
    patch_residual: bool = False
    attribution_target: None = None
    allow_truncate: bool = False


@dataclass(frozen=True)
class CausalProbe:
    text: str
    token_ids: tuple[int, ...]
    output_ids: tuple[int, int]
    spec: CausalSpec
    donor: tuple[float, ...] | None = None


@dataclass
class CausalResult:
    logits: list[float]
    heads: dict[str, list[float]]
    residual: list[float]

    @property
    def margin(self):
        return self.logits[0] - self.logits[1]


class CausalHarness(QwenHarness):
    def _execute(self, probes, owner):
        if owner is not self._claim:
            raise RuntimeError("All execution must pass through MPSBatchWorker")
        ident = threading.get_ident()
        if self._owner is None:
            self._owner = ident
        if ident != self._owner:
            raise RuntimeError("Execution moved off the consumer thread")
        if not probes or len(probes) > self.worker.batch_size:
            raise ValueError("Invalid microbatch")
        spec = probes[0].spec
        if any(p.spec != spec for p in probes):
            raise ValueError("Mixed causal hook plans")
        if spec.patch_head is not None and spec.patch_residual:
            raise ValueError("Only one intervention per probe")
        lengths = [len(p.token_ids) for p in probes]
        if any(not 0 < n <= self.worker.max_seq_len for n in lengths):
            raise ValueError("Invalid token length")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        width = max(lengths)
        ids = torch.tensor(
            [list(p.token_ids) + [pad] * (width - n) for p, n in zip(probes, lengths)],
            device=self.device,
        )
        rows = torch.arange(len(probes), device=self.device)
        pos = torch.tensor(lengths, device=self.device) - 1
        mask = torch.arange(width, device=self.device)[None, :] <= pos[:, None]
        if spec.patch_head:
            layer, head = spec.patch_head
            if not (0 <= layer < len(self.model.model.layers)):
                raise ValueError("Invalid patch layer")
            if not (0 <= head < self.model.config.num_attention_heads):
                raise ValueError("Invalid patch head")
        patch_width = self.head_dim if spec.patch_head else self.model.config.hidden_size
        donor = None
        if spec.patch_head or spec.patch_residual:
            if any(p.donor is None or len(p.donor) != patch_width for p in probes):
                raise ValueError("Invalid donor vector")
            donor = torch.tensor([p.donor for p in probes], device=self.device, dtype=torch.float32)
            if not torch.isfinite(donor).all():
                raise ValueError("Non-finite donor")
        captured = {}
        handles = []

        def cpu(x):
            return cpu_snapshot(x)

        def hook_for(layer):
            def hook(module, args):
                x = args[0]
                if spec.patch_head and layer == spec.patch_head[0]:
                    x = x.clone()
                    h = spec.patch_head[1]
                    x[rows, pos, h * self.head_dim : (h + 1) * self.head_dim] = donor.to(x.dtype)
                if layer in spec.layers:
                    captured[layer] = cpu(x[rows, pos].reshape(len(probes), -1, self.head_dim))
                return (x, *args[1:])

            return hook

        def residual_hook(module, args, output):
            patched = output.clone()
            patched[rows, pos] = donor.to(output.dtype)
            return patched

        try:
            layers = set(spec.layers)
            if spec.patch_head:
                layers.add(spec.patch_head[0])
            for layer in sorted(layers):
                handles.append(
                    self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
                        hook_for(layer)
                    )
                )
            if spec.patch_residual:
                handles.append(self.model.model.norm.register_forward_hook(residual_hook))
            with torch.inference_mode():
                hidden = self.model.model(
                    input_ids=ids, attention_mask=mask.long(), use_cache=False
                ).last_hidden_state[rows, pos]
                weights = self.model.lm_head.weight[
                    torch.tensor([p.output_ids for p in probes], device=self.device)
                ]
                logits = cpu((hidden.float()[:, None, :] * weights.float()).sum(-1))
                residual = cpu(hidden)
            if self.device.type == "mps":
                torch.mps.synchronize()
            result = []
            for i in range(len(probes)):
                heads = {
                    f"{layer}:{h}": values[i, h].tolist()
                    for layer, values in captured.items()
                    for h in range(values.shape[1])
                }
                out = CausalResult(logits[i].tolist(), heads, residual[i].tolist())
                if not np.isfinite(out.logits + out.residual).all():
                    raise RuntimeError("Non-finite output")
                if any(not np.isfinite(v).all() for v in heads.values()):
                    raise RuntimeError("Non-finite head vector")
                result.append(out)
            return result
        finally:
            for handle in handles:
                handle.remove()


# V2 keeps the original Causal* contract above intact. Each action affects one
# query head at the final real token, including when several layers are patched.
@dataclass(frozen=True)
class HeadAction:
    layer: int
    head: int
    mode: str = "replace"
    alpha: float = 1.0


@dataclass(frozen=True)
class V2Spec:
    layers: tuple[int, ...] = ()
    paths: bool = False
    actions: tuple[HeadAction, ...] = ()
    patch_residual: bool = False
    attribution_target: None = None


@dataclass
class V2Probe:
    text: str
    token_ids: tuple[int, ...]
    output_ids: tuple[int, int]
    spec: V2Spec
    donors: dict[str, list[float] | np.ndarray] | None = None
    pattern_donor: np.ndarray | list[float] | None = None
    values_donor: np.ndarray | list[list[float]] | None = None
    color_positions: tuple[int, ...] = ()
    residual_donor: np.ndarray | list[float] | None = None


@dataclass
class V2Result:
    logits: list[float]
    heads: dict[str, list[float]]
    residual: list[float]
    pattern: list[float] | None = None
    values: list[list[float]] | None = None
    reconstructed: list[float] | None = None
    path_error: float = 0.0
    applied_deltas: dict[str, list[float]] | None = None

    @property
    def margin(self):
        return self.logits[0] - self.logits[1]


_PATH_MODES = frozenset(
    {
        "source_color", "source_other", "ov_all", "ov_color", "ov_other",
        "qk", "qkov", "full", "interaction",
    }
)
_VECTOR_MODES = frozenset({"replace", "mean", "scale"})


def path_delta(mode, clean_pattern, clean_values, donor_pattern, donor_values, source_mask):
    """FP32 contribution algebra shared by execution and small planted controls.

    All arguments are tensors on the same device. Source masks select positions,
    not value token identities. No attention normalization changes under a mask.
    """
    pc, vc = clean_pattern.float(), clean_values.float()
    pd, vd = donor_pattern.float(), donor_values.float()
    if pc.shape != pd.shape or vc.shape != vd.shape or vc.shape[:-1] != pc.shape:
        raise ValueError("Incompatible pattern/value shapes")
    if source_mask.shape != pc.shape:
        raise ValueError("Source mask shape does not match attention pattern")
    if mode in {"source_color", "source_other", "qkov", "full"}:
        terms = pd[..., None] * vd - pc[..., None] * vc
    elif mode in {"ov_all", "ov_color", "ov_other"}:
        terms = pc[..., None] * (vd - vc)
    elif mode == "qk":
        terms = (pd - pc)[..., None] * vc
    elif mode == "interaction":
        terms = (pd - pc)[..., None] * (vd - vc)
    else:
        raise ValueError(f"Unknown decomposition mode: {mode}")
    if mode.endswith("_color"):
        terms = terms * source_mask[..., None]
    elif mode.endswith("_other"):
        terms = terms * (~source_mask)[..., None]
    return terms.sum(-2)


class V2Harness(CausalHarness):
    """Measured-vector coalitions and explicitly reconstructed final-row paths.

    SDPA's measured output is retained as the numerical anchor. Reconstructed
    pattern/value deltas are added to that anchor, so an identity delta preserves
    the fused BF16 computation exactly. We never mutate a shared GQA value head.
    """

    def _execute(self, probes, owner):
        if owner is not self._claim:
            raise RuntimeError("All execution must pass through MPSBatchWorker")
        ident = threading.get_ident()
        if self._owner is None:
            self._owner = ident
        if ident != self._owner:
            raise RuntimeError("Execution moved off the consumer thread")
        if not probes or len(probes) > self.worker.batch_size:
            raise ValueError("Invalid microbatch")
        spec = probes[0].spec
        if any(p.spec != spec for p in probes):
            raise ValueError("Mixed causal hook plans")
        config = self.model.config
        nlayers, nheads = len(self.model.model.layers), config.num_attention_heads
        nkv = config.num_key_value_heads
        if nheads % nkv:
            raise ValueError("Query heads do not divide into KV groups")
        if spec.attribution_target is not None:
            raise ValueError("V2 interventions do not support backward attribution")
        if len(set(spec.layers)) != len(spec.layers):
            raise ValueError("Duplicate capture layer")
        if any(not 0 <= layer < nlayers for layer in spec.layers):
            raise ValueError("Invalid capture layer")
        if spec.patch_residual and spec.actions:
            raise ValueError("Residual and head interventions must be separate")
        action_keys = [(a.layer, a.head) for a in spec.actions]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("Only one action per head is allowed")
        for action in spec.actions:
            if not 0 <= action.layer < nlayers or not 0 <= action.head < nheads:
                raise ValueError("Invalid action head")
            if action.mode not in _PATH_MODES | _VECTOR_MODES:
                raise ValueError(f"Unknown action mode: {action.mode}")
            if not np.isfinite(action.alpha):
                raise ValueError("Non-finite action alpha")
            if action.mode in _PATH_MODES and (action.layer, action.head) != (19, 5):
                raise ValueError("Path actions are registered only for head 19:5")
        paths = spec.paths or any(a.mode in _PATH_MODES for a in spec.actions)
        if paths and (nlayers <= 19 or nheads <= 5):
            raise ValueError("Path capture requires head 19:5")
        lengths = [len(p.token_ids) for p in probes]
        if any(not 0 < n <= self.worker.max_seq_len for n in lengths):
            raise ValueError("Invalid token length")
        if any(len(p.output_ids) != 2 for p in probes):
            raise ValueError("Each probe requires exactly two output tokens")
        for p, n in zip(probes, lengths):
            if len(set(p.color_positions)) != len(p.color_positions):
                raise ValueError("Duplicate color source position")
            if any(not isinstance(j, int) or not 0 <= j < n for j in p.color_positions):
                raise ValueError("Invalid color source position")
            if any(a.mode.endswith(("_color", "_other")) for a in spec.actions):
                if not p.color_positions:
                    raise ValueError("Source-specific action requires color positions")
        batch, width = len(probes), max(lengths)

        def validated_array(value, shape, name):
            if value is None:
                raise ValueError(f"Missing {name}")
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} shape or values")
            return array

        vector_donors = {}
        for action in spec.actions:
            if action.mode not in _VECTOR_MODES:
                continue
            key = f"{action.layer}:{action.head}"
            arrays = [
                validated_array((p.donors or {}).get(key), (self.head_dim,), f"donor {key}")
                for p in probes
            ]
            vector_donors[key] = torch.tensor(
                np.stack(arrays), device=self.device, dtype=torch.float32
            )
        pd = vd = None
        if any(a.mode in _PATH_MODES for a in spec.actions):
            patterns, values = [], []
            for p, n in zip(probes, lengths):
                pattern = validated_array(p.pattern_donor, (n,), "donor pattern")
                if np.any(pattern < 0) or not np.isclose(pattern.sum(), 1.0, atol=2e-5):
                    raise ValueError("Donor pattern must be a probability distribution")
                value = validated_array(p.values_donor, (n, self.head_dim), "donor values")
                patterns.append(np.pad(pattern, (0, width - n)))
                values.append(np.pad(value, ((0, width - n), (0, 0))))
            pd = torch.tensor(np.stack(patterns), device=self.device, dtype=torch.float32)
            vd = torch.tensor(np.stack(values), device=self.device, dtype=torch.float32)
        residual_donor = None
        if spec.patch_residual:
            residual_donor = torch.tensor(
                np.stack([
                    validated_array(p.residual_donor, (config.hidden_size,), "residual donor")
                    for p in probes
                ]),
                device=self.device,
                dtype=torch.float32,
            )
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = torch.tensor(
            [list(p.token_ids) + [pad] * (width - n) for p, n in zip(probes, lengths)],
            device=self.device,
        )
        rows = torch.arange(batch, device=self.device)
        pos = torch.tensor(lengths, device=self.device) - 1
        valid = torch.arange(width, device=self.device)[None, :] <= pos[:, None]
        source_mask = torch.zeros((batch, width), device=self.device, dtype=torch.bool)
        for i, p in enumerate(probes):
            if p.color_positions:
                source_mask[i, list(p.color_positions)] = True
        captured, qkv, path_cpu, delta_cpu = {}, {}, {}, {}
        handles = []
        actions_by_layer = {}
        for action in spec.actions:
            actions_by_layer.setdefault(action.layer, []).append(action)

        def capture_position(module, args, kwargs):
            embeddings = kwargs.get("position_embeddings")
            if embeddings is None and len(args) > 1:
                embeddings = args[1]
            if embeddings is None:
                raise RuntimeError("Attention position embeddings unavailable")
            qkv["position_embeddings"] = embeddings

        def capture_projection(name):
            def hook(module, args, output):
                qkv[name] = output
            return hook

        def reconstruct_path(x):
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            q = qkv["q"].transpose(1, 2)
            k = qkv["k"].transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *qkv["position_embeddings"])
            kv_head = 5 // (nheads // nkv)
            final_q = q[rows, 5, pos].float()
            keys = k[:, kv_head].float()
            values = qkv["v"].reshape(batch, width, nkv, self.head_dim)[:, :, kv_head].float()
            scaling = self.model.model.layers[19].self_attn.scaling
            scores = (keys * final_q[:, None, :]).sum(-1) * scaling
            scores = scores.masked_fill(~valid, -torch.inf)
            pattern = torch.softmax(scores, dim=-1)
            reconstructed = (pattern[..., None] * values).sum(1)
            measured = x[rows, pos, 5 * self.head_dim : 6 * self.head_dim].float()
            path_cpu["error"] = cpu_snapshot((measured - reconstructed).abs().amax(-1))
            if spec.paths:
                path_cpu["pattern"] = cpu_snapshot(pattern)
                path_cpu["values"] = cpu_snapshot(values)
                path_cpu["reconstructed"] = cpu_snapshot(reconstructed)
            return pattern, values

        def hook_for(layer):
            def hook(module, args):
                x = args[0]
                pattern = values = None
                if paths and layer == 19:
                    pattern, values = reconstruct_path(x)
                changed = False
                for action in actions_by_layer.get(layer, ()):
                    key = f"{layer}:{action.head}"
                    section = slice(action.head * self.head_dim, (action.head + 1) * self.head_dim)
                    current = x[rows, pos, section]
                    if action.mode in {"replace", "mean"}:
                        target = vector_donors[key].to(x.dtype)
                        delta = target.float() - current.float()
                    elif action.mode == "scale":
                        delta = (action.alpha - 1.0) * (current.float() - vector_donors[key])
                        target = (current.float() + delta).to(x.dtype)
                    else:
                        delta = path_delta(action.mode, pattern, values, pd, vd, source_mask)
                        target = (current.float() + delta).to(x.dtype)
                    delta_cpu[key] = cpu_snapshot(delta)
                    # Avoid even a new assignment for an exact identity intervention.
                    if not torch.equal(target, current):
                        if not changed:
                            x = x.clone()
                            changed = True
                        x[rows, pos, section] = target
                if layer in spec.layers:
                    captured[layer] = cpu_snapshot(
                        x[rows, pos].reshape(batch, nheads, self.head_dim)
                    )
                return (x, *args[1:])
            return hook

        def residual_hook(module, args, output):
            target = residual_donor.to(output.dtype)
            if torch.equal(output[rows, pos], target):
                return output
            patched = output.clone()
            patched[rows, pos] = target
            return patched

        try:
            layers = set(spec.layers) | set(actions_by_layer)
            if paths:
                layers.add(19)
                attention = self.model.model.layers[19].self_attn
                handles.append(attention.register_forward_pre_hook(capture_position, with_kwargs=True))
                handles.append(attention.q_norm.register_forward_hook(capture_projection("q")))
                handles.append(attention.k_norm.register_forward_hook(capture_projection("k")))
                handles.append(attention.v_proj.register_forward_hook(capture_projection("v")))
            for layer in sorted(layers):
                handles.append(
                    self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
                        hook_for(layer)
                    )
                )
            if spec.patch_residual:
                handles.append(self.model.model.norm.register_forward_hook(residual_hook))
            with torch.inference_mode():
                hidden = self.model.model(
                    input_ids=ids, attention_mask=valid.long(), use_cache=False
                ).last_hidden_state[rows, pos]
                weights = self.model.lm_head.weight[
                    torch.tensor([p.output_ids for p in probes], device=self.device)
                ]
                logits = cpu_snapshot((hidden.float()[:, None, :] * weights.float()).sum(-1))
                residual = cpu_snapshot(hidden)
            if self.device.type == "mps":
                torch.mps.synchronize()
            results = []
            for i, n in enumerate(lengths):
                heads = {
                    f"{layer}:{h}": snapshot[i, h].tolist()
                    for layer, snapshot in captured.items()
                    for h in range(nheads)
                }
                pattern = path_cpu["pattern"][i, :n].tolist() if spec.paths else None
                values = path_cpu["values"][i, :n].tolist() if spec.paths else None
                reconstructed = path_cpu["reconstructed"][i].tolist() if spec.paths else None
                result = V2Result(
                    logits=logits[i].tolist(), heads=heads, residual=residual[i].tolist(),
                    pattern=pattern, values=values, reconstructed=reconstructed,
                    path_error=float(path_cpu["error"][i]) if paths else 0.0,
                    applied_deltas={key: value[i].tolist() for key, value in delta_cpu.items()},
                )
                arrays = [result.logits, result.residual, *heads.values()]
                if spec.paths:
                    arrays.extend([pattern, values, reconstructed])
                arrays.extend(result.applied_deltas.values())
                if any(not np.isfinite(array).all() for array in arrays):
                    raise RuntimeError("Non-finite causal result")
                results.append(result)
            return results
        finally:
            for handle in handles:
                handle.remove()
            qkv.clear()
