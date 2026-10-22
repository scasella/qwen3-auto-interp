"""Spatial H0 pass; synchronized v10 arithmetic and original 0.6B loader."""

from dataclasses import dataclass

import numpy as np
import torch

from src.harness.qwen_model import QwenHarness
from src.poc.harness import CausalHarness, CausalSpec, cpu_snapshot
from src.poc.v10_stable_harness import V10Harness
from src.poc.v11_question_hop0 import U

SPAN = 8


@dataclass(frozen=True)
class SpanSpec(CausalSpec):
    heads: tuple = ()
    targets: tuple = U
    capture: bool = False
    last_layer: int = 17


class V11Harness(V10Harness):
    @classmethod
    def load(cls, config, worker):
        if config.device != "mps" or config.dtype != "bfloat16":
            raise ValueError("Target requires MPS BF16")
        return QwenHarness.load.__func__(cls, config, worker)

    def _execute(self, probes, owner):
        spec = probes[0].spec
        if not isinstance(spec, SpanSpec):
            return super()._execute(probes, owner)
        if owner is not self._claim:
            raise RuntimeError("FIFO owner required")
        assert spec.last_layer < len(self.model.model.layers)
        assert all(
            0 <= layer_id < min(x[0] for x in spec.targets)
            and 0 <= h < self.model.config.num_attention_heads
            for layer_id, h in spec.heads
        )
        b, d = len(probes), self.head_dim
        lengths = [len(p.token_ids) for p in probes]
        query = [p.payload["query_position"] for p in probes]
        assert all(0 <= q < n - 1 and n - q <= SPAN for q, n in zip(query, lengths))
        width = max(lengths)
        rows = torch.arange(b, device=self.device)
        final = torch.tensor(lengths, device=self.device) - 1
        handles = []
        state = {"residual": {}, "heads": {}, "v": {}, "applied": {}, "errors": []}
        group = self.model.config.num_attention_heads // self.model.config.num_key_value_heads

        def tensor(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

        def padded(x, stop_final=False):
            result = torch.zeros((b, SPAN, d), device=self.device, dtype=x.dtype)
            for i, (q, n) in enumerate(zip(query, lengths)):
                end = n - 1 if stop_final else n
                result[i, : end - q] = x[i, q:end]
            return cpu_snapshot(result)

        def layer_pre(layer):
            def hook(module, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                if spec.capture:
                    state["residual"][layer] = cpu_snapshot(x[rows, final])
                else:
                    clean = tensor([p.payload["residual"][layer] for p in probes]).to(x.dtype)
                    x = x.clone()
                    x[rows, final] = clean
                    state["errors"].append(cpu_snapshot((x[rows, final] - clean).abs().amax(-1)))
                    if args:
                        args = (x, *args[1:])
                    else:
                        kwargs = dict(kwargs, hidden_states=x)
                return args, kwargs

            return hook

        def head_hook(layer):
            def hook(module, args):
                x = args[0]
                if spec.capture:
                    for h in range(self.model.config.num_attention_heads):
                        state["heads"][(layer, h)] = padded(x[:, :, h * d : (h + 1) * d], True)
                else:
                    before = x
                    x = x.clone()
                    allowed = torch.zeros_like(x, dtype=torch.bool)
                    for layer_id, h in spec.heads:
                        if layer_id != layer:
                            continue
                        patch = tensor(
                            [p.payload["head_values"][f"{layer_id}:{h}"] for p in probes]
                        ).to(x.dtype)
                        for i, (q, n) in enumerate(zip(query, lengths)):
                            x[i, q : n - 1, h * d : (h + 1) * d] = patch[i, : n - 1 - q]
                            allowed[i, q : n - 1, h * d : (h + 1) * d] = True
                        state["applied"][(layer_id, h)] = padded(x[:, :, h * d : (h + 1) * d], True)
                    error = (x - before).abs().masked_fill(allowed, 0).flatten(1).amax(-1)
                    state["errors"].append(cpu_snapshot(error))
                return (x, *args[1:])

            return hook

        def value_hook(layer, head, index):
            def hook(module, args, output):
                v = output.reshape(b, width, -1, d)[:, :, head // group]
                state["v"][index] = padded(v)
                if not spec.capture:
                    errors = []
                    for i, (q, n) in enumerate(zip(query, lengths)):
                        clean = tensor(probes[i].payload["clean_v"][index])
                        errors.append(
                            torch.maximum(
                                (v[i, :q].float() - clean[:q]).abs().max(),
                                (v[i, n - 1].float() - clean[n - 1]).abs().max(),
                            )
                        )
                    state["errors"].append(cpu_snapshot(torch.stack(errors)))

            return hook

        try:
            for layer in range(spec.last_layer + 1):
                module = self.model.model.layers[layer]
                handles.append(module.register_forward_pre_hook(layer_pre(layer), with_kwargs=True))
            layers = (
                range(min(x[0] for x in spec.targets))
                if spec.capture
                else sorted({layer_id for layer_id, h in spec.heads})
            )
            for layer in layers:
                handles.append(
                    self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
                        head_hook(layer)
                    )
                )
            for i, (layer, head) in enumerate(spec.targets):
                handles.append(
                    self.model.model.layers[layer].self_attn.v_proj.register_forward_hook(
                        value_hook(layer, head, i)
                    )
                )
            result = CausalHarness._execute(self, probes, owner)
            # CausalHarness synchronizes every CPU copy before this point.
            for i, r in enumerate(result):
                r.spatial = {
                    "vspan": np.stack([state["v"][h][i].numpy() for h in range(len(spec.targets))]),
                    "errors": np.array([e[i].item() for e in state["errors"]], dtype=np.float32),
                }
                if spec.capture:
                    r.spatial["residual"] = np.stack(
                        [
                            state["residual"][layer_id][i].numpy()
                            for layer_id in range(spec.last_layer + 1)
                        ]
                    )
                    r.spatial["hop0"] = np.stack(
                        [state["heads"][h][i].numpy() for h in sorted(state["heads"])]
                    )
                else:
                    r.spatial["applied"] = np.stack(
                        [state["applied"][h][i].numpy() for h in spec.heads]
                    )
                    assert np.all(r.spatial["errors"] == 0), (
                        "Context/final clamp or head isolation violation"
                    )
            return result
        finally:
            for handle in handles:
                handle.remove()
