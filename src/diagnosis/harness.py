"""Final-position residual accounting and isolated module-output interventions."""

from dataclasses import dataclass

import torch

from src.followup.harness import PathHarness
from src.poc.harness import CausalSpec, cpu_snapshot


@dataclass(frozen=True)
class TraceSpec(CausalSpec):
    trace: bool = True
    component: tuple[int, str] | None = None


class TraceHarness(PathHarness):
    def _execute(self, probes, owner):
        if owner is not self._claim:
            raise RuntimeError("All forwards must use MPSBatchWorker")
        spec = probes[0].spec
        if not isinstance(spec, TraceSpec):
            return super()._execute(probes, owner)
        batch = torch.arange(len(probes), device=self.device)
        pos = torch.tensor([len(p.token_ids) - 1 for p in probes], device=self.device)
        ids = torch.tensor([p.output_ids for p in probes], device=self.device)
        # Direction independent of residual RMS. Divide by each final RMS on CPU.
        unembedding = self.model.lm_head.weight[ids].float()
        direction = (unembedding[:, 0] - unembedding[:, 1]) * self.model.model.norm.weight.float()
        snapshots, vectors, handles = {}, {}, []
        head_weights = {}
        if spec.trace:
            for layer, block in enumerate(self.model.model.layers):
                head_weights[layer] = direction @ block.self_attn.o_proj.weight.float()
        donor = None
        if spec.component:
            assert not spec.patch_head and not spec.patch_residual
            if any(
                p.donor is None or len(p.donor) != self.model.config.hidden_size for p in probes
            ):
                raise ValueError("Module donor width mismatch")
            donor = torch.tensor([p.donor for p in probes], device=self.device, dtype=torch.float32)
            if not torch.isfinite(donor).all():
                raise ValueError("Nonfinite module donor")

        def state(name, x):
            final = x[batch, pos].float()
            eps = self.model.model.norm.variance_epsilon
            snapshots[name] = cpu_snapshot(
                torch.stack(
                    ((final * direction).sum(-1), (final.square().mean(-1) + eps).sqrt()), -1
                )
            )

        def before(layer):
            def hook(module, args, kwargs):
                if spec.trace:
                    state(f"{layer}:before", kwargs.get("hidden_states", args[0] if args else None))

            return hook

        def after_attention(layer):
            def hook(module, args):
                if spec.trace:
                    state(f"{layer}:after_attention", args[0])

            return hook

        def after_block(layer):
            def hook(module, args, output):
                if spec.trace:
                    state(f"{layer}:after_mlp", output)

            return hook

        def heads(layer):
            def hook(module, args):
                x = args[0][batch, pos].float().reshape(len(probes), -1, self.head_dim)
                w = head_weights[layer].reshape(len(probes), -1, self.head_dim)
                snapshots[f"{layer}:head_dot"] = cpu_snapshot((x * w).sum(-1))

            return hook

        def component(layer, kind):
            def hook(module, args, output):
                value = output[0] if kind == "attention" else output
                if spec.component == (layer, kind):
                    value = value.clone()
                    value[batch, pos] = donor.to(value.dtype)
                if spec.trace and (layer >= 20 or (layer == 19 and kind == "mlp")):
                    vectors[f"{layer}:{kind}"] = cpu_snapshot(value[batch, pos])
                if kind == "attention":
                    return (value, *output[1:])
                return value

            return hook

        try:
            for layer, block in enumerate(self.model.model.layers):
                if spec.trace:
                    handles.extend(
                        [
                            block.register_forward_pre_hook(before(layer), with_kwargs=True),
                            block.post_attention_layernorm.register_forward_pre_hook(
                                after_attention(layer)
                            ),
                            block.register_forward_hook(after_block(layer)),
                            block.self_attn.o_proj.register_forward_pre_hook(heads(layer)),
                        ]
                    )
                for kind, module in (("attention", block.self_attn), ("mlp", block.mlp)):
                    if spec.trace or spec.component == (layer, kind):
                        handles.append(module.register_forward_hook(component(layer, kind)))
            results = super()._execute(probes, owner)
            for i, result in enumerate(results):
                result.trace = {key: value[i].tolist() for key, value in snapshots.items()}
                result.components = {key: value[i].tolist() for key, value in vectors.items()}
            return results
        finally:
            for handle in handles:
                handle.remove()
