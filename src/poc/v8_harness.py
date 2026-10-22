"""Single-pass repairs from current activations plus frozen calibration constants."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from src.poc.harness import CausalHarness, CausalProbe, CausalSpec, cpu_snapshot

TARGETS = ((19, 5), (21, 13), (22, 9), (24, 7))


@dataclass(frozen=True)
class RepairSpec(CausalSpec):
    targets: tuple = TARGETS
    mode: str = "amplify"
    alpha: float = 1.0
    beta: float = 0.0
    alpha22: float = 1.0


@dataclass(frozen=True)
class RepairProbe(CausalProbe):
    color_positions: tuple = ()
    key_positions: tuple = ()


def delta_torch(h, scores, p, v, positions, mu, mean_s, mean_p, mode, alpha, beta, alpha22):
    """Signed post-softmax coefficients are not clipped or renormalized."""
    routed = p
    qk = torch.zeros_like(h)
    if beta != 0:
        row = torch.arange(len(h), device=h.device)[:, None]
        excess = mean_s - mean_s.mean() if mode == "score" else mean_p - mean_p.mean()
        if mode == "score":
            edited = scores.clone()
            edited[row, positions] -= beta * excess
            routed = edited.softmax(-1)
        elif mode == "probability":
            routed = p.clone()
            routed[row, positions] -= beta * excess
        else:
            raise ValueError("Nonzero beta needs registered routing mode")
        qk = ((routed - p)[..., None] * v).sum(1)
    weight = alpha if mode == "amplify" else alpha22
    delta = qk + (weight - 1) * (h + qk - mu)
    return delta, routed


class RepairHarness(CausalHarness):
    # Constants are set once per candidate on the consumer's serial queue, never per label.
    constants = None

    def _execute(self, probes, owner):
        if owner is not self._claim:
            raise RuntimeError("Single worker required")
        spec = probes[0].spec
        if not isinstance(spec, RepairSpec):
            return super()._execute(probes, owner)
        if spec.mode not in ("amplify", "score", "probability"):
            raise ValueError("Unknown repair")
        b, d = len(probes), self.head_dim
        lengths = [len(p.token_ids) for p in probes]
        width = max(lengths)
        rows = torch.arange(b, device=self.device)
        last = torch.tensor(lengths, device=self.device) - 1
        valid = torch.arange(width, device=self.device)[None, :] <= last[:, None]
        positions = torch.tensor([p.color_positions for p in probes], device=self.device)
        saved = {}
        handles = []
        if len({t[0] for t in spec.targets}) != len(spec.targets):
            raise ValueError("One target per layer")
        constants = self.constants

        def install(hi, layer, head):
            attn = self.model.model.layers[layer].self_attn
            state = {}
            snap = {}
            saved[f"{layer}:{head}"] = snap
            kv = head // (
                self.model.config.num_attention_heads // self.model.config.num_key_value_heads
            )

            def pre(module, args, kwargs):
                state["rope"] = tuple(x.expand(b, -1, -1) for x in kwargs["position_embeddings"])

            def qhook(module, args, out):
                c, s = state["rope"]
                q = out[rows, last, head]
                state["q"] = q * c[rows, last] + rotate_half(q) * s[rows, last]

            def khook(module, args, out):
                c, s = state["rope"]
                k = out[:, :, kv]
                state["k"] = k * c + rotate_half(k) * s

            def vhook(module, args, out):
                state["v"] = out.reshape(b, width, -1, d)[:, :, kv]

            def ohook(module, args):
                x = args[0]
                q, k, v = (state[n].float() for n in ("q", "k", "v"))
                scores = ((q[:, None] * k).sum(-1) * attn.scaling).masked_fill(~valid, -torch.inf)
                p = scores.softmax(-1)
                section = slice(head * d, (head + 1) * d)
                h = x[rows, last, section].float()
                delta = torch.zeros_like(h)
                routed = p
                if hi < 3 and constants is not None:

                    def t(name):
                        return torch.as_tensor(
                            constants[name][hi], device=self.device, dtype=torch.float32
                        )

                    delta, routed = delta_torch(
                        h,
                        scores,
                        p,
                        v,
                        positions,
                        t("mu"),
                        t("mean_s"),
                        t("mean_p"),
                        spec.mode,
                        spec.alpha,
                        spec.beta,
                        spec.alpha22 if hi == 2 else 1.0,
                    )
                applied = (h + delta).to(x.dtype)
                snap.update(
                    {
                        n: cpu_snapshot(value)
                        for n, value in dict(
                            q=q,
                            k=k,
                            v=v,
                            s=scores,
                            p=p,
                            h=h,
                            routed=routed,
                            delta=delta,
                            applied=applied,
                        ).items()
                    }
                )
                snap["reconstruction_error"] = cpu_snapshot(
                    ((p[..., None] * v).sum(1) - h).abs().amax(-1)
                )
                before = x[rows, last].reshape(b, -1, d)
                if not torch.equal(applied, x[rows, last, section]):
                    x = x.clone()
                    x[rows, last, section] = applied
                after = x[rows, last].reshape(b, -1, d)
                others = [j for j in range(before.shape[1]) if j != head]
                snap["sibling_error"] = cpu_snapshot(
                    (before[:, others] - after[:, others]).abs().flatten(1).amax(-1)
                )
                state.clear()
                return (x, *args[1:])

            handles.extend(
                [
                    attn.register_forward_pre_hook(pre, with_kwargs=True),
                    attn.q_norm.register_forward_hook(qhook),
                    attn.k_norm.register_forward_hook(khook),
                    attn.v_proj.register_forward_hook(vhook),
                    attn.o_proj.register_forward_pre_hook(ohook),
                ]
            )

        try:
            for hi, (layer, head) in enumerate(spec.targets):
                install(hi, layer, head)
            result = super()._execute(probes, owner)
            for i, r in enumerate(result):
                r.repair = {
                    key: {
                        field: value[i, : lengths[i]].tolist()
                        if field in ("k", "v", "s", "p", "routed")
                        else value[i].tolist()
                        for field, value in snapshot.items()
                    }
                    for key, snapshot in saved.items()
                }
                assert all(
                    np.isfinite(v).all() for snap in r.repair.values() for v in snap.values()
                )
            return result
        finally:
            for handle in handles:
                handle.remove()
