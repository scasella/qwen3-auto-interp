"""Head-local Q/K/V interventions retaining the installed SDPA backend and GQA siblings."""

from dataclasses import dataclass

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from src.poc.harness import CausalHarness, CausalProbe, CausalSpec, cpu_snapshot


@dataclass(frozen=True)
class PathSpec(CausalSpec):
    mode: str = "capture"
    layer: int = 19
    head: int = 5


@dataclass(frozen=True)
class PathProbe(CausalProbe):
    payload: dict | None = None


MODES = (
    "source_swap",
    "q_query",
    "q_assignment",
    "v_prefix",
    "v_suffix",
    "drop_relevant",
    "drop_distractor",
    "k_assignment",
)


def intervene(q, k, v, mode, payloads):
    """Per-example component substitution; recipient RoPE positions remain aligned."""
    q, k, v = q.clone(), k.clone(), v.clone()
    for i, p in enumerate(payloads):

        def donor(name):
            return torch.as_tensor(p[name], device=q.device, dtype=torch.float32).to(q.dtype)

        if mode in ("q_query", "q_assignment"):
            q[i] = donor("q")
        elif mode == "k_assignment":
            k[i, : len(p["k"])] = donor("k")
        elif mode == "source_swap":
            a, b = p["value_positions"]
            v[i, a], v[i, b] = v[i, b].clone(), v[i, a].clone()
        elif mode == "v_prefix":
            stop = p["suffix_start"]
            v[i, :stop] = donor("v")[:stop]
        elif mode == "v_suffix":
            start, n = p["suffix_start"], len(p["v"])
            v[i, start:n] = donor("v")[start:n]
        elif mode in ("drop_relevant", "drop_distractor"):
            which = p["query_record"] if mode == "drop_relevant" else 1 - p["query_record"]
            v[i, p["value_positions"][which]] = 0
        elif mode not in ("capture", "reconstruct"):
            raise ValueError(f"Unknown intervention {mode}")
    return q, k, v


class PathHarness(CausalHarness):
    def _execute(self, probes, owner):
        if owner is not self._claim:
            raise RuntimeError("Only the single consumer may execute the model")
        spec = probes[0].spec
        if not isinstance(spec, PathSpec):
            return super()._execute(probes, owner)
        module = self.model.model.layers[spec.layer].self_attn
        h, d = spec.head, self.head_dim
        kv = h // (self.model.config.num_attention_heads // self.model.config.num_key_value_heads)
        lengths = [len(p.token_ids) for p in probes]
        pos = torch.tensor(lengths, device=self.device) - 1
        batch = torch.arange(len(probes), device=self.device)
        cache, snapshots, handles = {}, {}, []

        def pre(module, args, kwargs):
            cache["rope"] = tuple(
                t.expand(len(probes), -1, -1) for t in kwargs["position_embeddings"]
            )
            snapshots["upstream"] = cpu_snapshot(kwargs["hidden_states"][batch, pos])

        def q_hook(module, args, output):
            cos, sin = cache["rope"]
            x = output[batch, pos, h]
            cache["q"] = x * cos[batch, pos] + rotate_half(x) * sin[batch, pos]

        def k_hook(module, args, output):
            cos, sin = cache["rope"]
            x = output[:, :, kv]
            cache["k"] = x * cos + rotate_half(x) * sin

        def v_hook(module, args, output):
            cache["v"] = output.reshape(*output.shape[:-1], -1, d)[:, :, kv]

        def output_hook(module, args):
            x = args[0]
            q, k, v = (cache[name] for name in ("q", "k", "v"))
            for name, value in zip(("q", "k", "v"), (q, k, v), strict=True):
                snapshots[name] = cpu_snapshot(value)
            valid = torch.arange(k.shape[1], device=self.device)[None, :] <= pos[:, None]
            weights = (q.float()[:, None, :] * k.float()).sum(-1) * module_parent.scaling
            weights = weights.masked_fill(~valid, float("-inf")).softmax(-1)
            snapshots["attention"] = cpu_snapshot(weights)
            original = x[batch, pos].reshape(len(probes), -1, d)
            snapshots["original_heads"] = cpu_snapshot(original)
            if spec.mode != "capture":
                q, k, v = intervene(q, k, v, spec.mode, [p.payload for p in probes])
                rebuilt = scaled_dot_product_attention(
                    q[:, None, None, :],
                    k[:, None, :, :],
                    v[:, None, :, :],
                    attn_mask=valid[:, None, None, :],
                    dropout_p=0.0,
                    scale=module_parent.scaling,
                )[:, 0, 0]
                x = x.clone()
                x[batch, pos, h * d : (h + 1) * d] = rebuilt
            snapshots["patched_heads"] = cpu_snapshot(x[batch, pos].reshape(len(probes), -1, d))
            cache.clear()
            return (x, *args[1:])

        module_parent = module
        try:
            handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(module.q_norm.register_forward_hook(q_hook))
            handles.append(module.k_norm.register_forward_hook(k_hook))
            handles.append(module.v_proj.register_forward_hook(v_hook))
            # Runs before the parent harness capture hook, and changes only this Q-head.
            handles.append(module.o_proj.register_forward_pre_hook(output_hook))
            results = super()._execute(probes, owner)
            # Parent synchronizes the device before any CPU reads.
            for i, r in enumerate(results):
                r.path = {
                    name: value[i, : lengths[i]].tolist()
                    if name in ("k", "v", "attention")
                    else value[i].tolist()
                    for name, value in snapshots.items()
                }
            return results
        finally:
            for handle in handles:
                handle.remove()
