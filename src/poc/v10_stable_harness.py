"""Controlled upstream -> QK path patches; no shared GQA projections are mutated."""

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from src.poc.harness import CausalHarness, cpu_snapshot
from src.poc.v4_harness import V4Spec


def fp32_arrays(*tensors):
    """Finish all asynchronous CPU copies before NumPy consumes their storage."""
    snapshots = [cpu_snapshot(value) for value in tensors]
    if any(value.device.type == "mps" for value in tensors):
        torch.mps.synchronize()
    return [value.numpy() for value in snapshots]


def contribution_delta(mode, pc, vc, pd, vd, mask):
    """Explicit FP32 CPU reduction; return the FP32 intervention to its device."""
    p, v, donor_p, donor_v, weights = fp32_arrays(pc, vc, pd, vd, mask)
    if mode == "qk":
        terms = (donor_p - p)[..., None] * v
    elif mode in ("ov_color", "ov_context", "ov_other"):
        terms = p[..., None] * (donor_v - v) * weights[..., None]
    elif mode == "full":
        terms = donor_p[..., None] * donor_v - p[..., None] * v
    else:
        raise ValueError(mode)
    delta = np.sum(terms, axis=-2, dtype=np.float32)
    return torch.from_numpy(delta).to(device=pc.device, dtype=torch.float32)


def routing_pattern(q, k, valid, scaling):
    """Reconstruct each unpadded attention row with explicit FP32 operations."""
    queries, keys, validity = fp32_arrays(q, k, valid)
    validity = validity.astype(bool)
    result = np.zeros(validity.shape, dtype=np.float32)
    for i, keep in enumerate(validity):
        scores = np.sum(keys[i, keep] * queries[i], axis=-1, dtype=np.float32)
        scores *= np.float32(scaling)
        weights = np.exp(scores - np.max(scores))
        result[i, keep] = weights / np.sum(weights, dtype=np.float32)
    return torch.from_numpy(result).to(device=q.device, dtype=torch.float32)


class V10Harness(CausalHarness):
    @classmethod
    def load(cls, config, worker):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config.device != "mps" or config.dtype != "bfloat16":
            raise ValueError("Target execution requires MPS and BF16")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.manual_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(config.name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        expected = dict(
            num_hidden_layers=28,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        )
        for key, value in expected.items():
            if getattr(model.config, key, None) != value:
                raise ValueError(f"Unexpected 1.7B architecture: {key}")
        if model.__class__.__name__ != "Qwen3ForCausalLM":
            raise ValueError("Unexpected model class")
        return cls(model, tokenizer, config, worker)

    def _execute(self, probes, owner):
        if owner is not self._claim:
            raise RuntimeError("Single-consumer worker required")
        spec = probes[0].spec
        if not isinstance(spec, V4Spec):
            return super()._execute(probes, owner)
        if spec.mode not in (
            "capture",
            "mediate",
            "qk",
            "ov_color",
            "ov_context",
            "full",
            "replace",
            "q_final",
            "k_color",
            "k_context",
            "ov_other",
            "identity",
        ):
            raise ValueError("Unknown routing mode")
        if spec.mode == "mediate" and spec.active:
            raise ValueError("Mediator runs clamp all targets; no final path action")
        if not spec.targets or len(set(spec.targets)) != len(spec.targets):
            raise ValueError("Targets must be nonempty and unique")
        for layer_id, head in spec.targets + spec.upstream:
            if not (
                0 <= layer_id < len(self.model.model.layers)
                and 0 <= head < self.model.config.num_attention_heads
            ):
                raise ValueError("Head outside model architecture")
        if not set(spec.active) <= set(spec.targets):
            raise ValueError("Unknown target")
        if any(layer_id >= min(t[0] for t in spec.targets) for layer_id, _ in spec.upstream):
            raise ValueError("Upstream interventions must precede targets")
        b, d = len(probes), self.head_dim
        lengths = [len(p.token_ids) for p in probes]
        width = max(lengths)
        rows = torch.arange(b, device=self.device)
        pos = torch.tensor(lengths, device=self.device) - 1
        valid = torch.arange(width, device=self.device)[None, :] <= pos[:, None]
        hooks, cache, snapshots = [], {}, {}
        input_tensors = {}

        def tensor(values):
            values = np.asarray(values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Invalid intervention values")
            return torch.tensor(values, device=self.device, dtype=torch.float32)

        if spec.mode != "capture" or spec.upstream:
            for target in spec.targets:
                key = f"{target[0]}:{target[1]}"
                input_tensors[key] = {"h": tensor([p.payload["clean"][key]["h"] for p in probes])}
                if spec.mode not in ("mediate", "replace"):
                    for name in ("p", "v", "k"):
                        vals = []
                        donor = []
                        for p, n in zip(probes, lengths):
                            x = np.asarray(p.payload["clean"][key][name], dtype=np.float32)[:n]
                            y = np.asarray(p.payload["donor"][key][name], dtype=np.float32)[:n]
                            pad = ((0, width - n),) + ((0, 0),) * (x.ndim - 1)
                            vals.append(np.pad(x, pad))
                            donor.append(np.pad(y, pad))
                        input_tensors[key][name] = tensor(vals)
                        input_tensors[key]["d" + name] = tensor(donor)
                if spec.mode not in ("mediate", "replace"):
                    input_tensors[key]["q"] = tensor([p.payload["clean"][key]["q"] for p in probes])
                    input_tensors[key]["dq"] = tensor(
                        [p.payload["donor"][key]["q"] for p in probes]
                    )
                if spec.mode == "replace":
                    input_tensors[key]["dh"] = tensor(
                        [p.payload["donor"][key]["h"] for p in probes]
                    )
            for layer, head in spec.upstream:
                key = f"{layer}:{head}"
                input_tensors["u" + key] = tensor([p.payload["upstream"][key] for p in probes])

        def upstream_hook(layer):
            def hook(module, args):
                x = args[0].clone()
                for layer_id, h in spec.upstream:
                    if layer_id == layer:
                        x[rows, pos, h * d : (h + 1) * d] = input_tensors[f"u{layer_id}:{h}"].to(
                            x.dtype
                        )
                return (x, *args[1:])

            return hook

        def install(layer, head):
            module = self.model.model.layers[layer].self_attn
            key = f"{layer}:{head}"
            kv = head // (
                self.model.config.num_attention_heads // self.model.config.num_key_value_heads
            )
            state, saved = {}, {}
            cache[key], snapshots[key] = state, saved

            def pre(module, args, kwargs):
                state["rope"] = tuple(x.expand(b, -1, -1) for x in kwargs["position_embeddings"])

            def qhook(module, args, output):
                cos, sin = state["rope"]
                q = output[rows, pos, head]
                state["q"] = q * cos[rows, pos] + rotate_half(q) * sin[rows, pos]

            def khook(module, args, output):
                cos, sin = state["rope"]
                k = output[:, :, kv]
                state["k"] = k * cos + rotate_half(k) * sin

            def vhook(module, args, output):
                state["v"] = output.reshape(b, width, -1, d)[:, :, kv]

            def ohook(oproj, args):
                x = args[0]
                q, k, v = state["q"].float(), state["k"].float(), state["v"].float()
                pattern = routing_pattern(q, k, valid, module.scaling)
                h = x[rows, pos, head * d : (head + 1) * d]
                saved["q"], saved["k"] = cpu_snapshot(q), cpu_snapshot(k)
                saved["routed_p"] = cpu_snapshot(pattern)
                saved["p"], saved["v"], saved["h"] = (
                    cpu_snapshot(pattern),
                    cpu_snapshot(v),
                    cpu_snapshot(h),
                )
                saved["reconstruction_error"] = cpu_snapshot(
                    ((pattern[..., None] * v).sum(1) - h.float()).abs().amax(-1)
                )
                before = x[rows, pos].reshape(b, -1, d).clone()
                # Only one head is assigned, never its GQA sibling or another position.
                target = None
                if spec.mode == "mediate":
                    target = input_tensors[key]["h"].to(x.dtype)
                elif (layer, head) in spec.active:
                    t = input_tensors[key]
                    if spec.mode == "replace":
                        target = t["dh"].to(x.dtype)
                        delta = target.float() - t["h"]
                    else:
                        # Construct discrete masks on CPU; one transfer avoids a
                        # scalar MPS assignment for every prompt and selected head.
                        mask_values = np.zeros((b, width), dtype=np.float32)
                        for i, probe in enumerate(probes):
                            if spec.mode in ("ov_context", "k_context"):
                                mask_values[i, : probe.payload["context_end"]] = 1.0
                            else:
                                mask_values[i, probe.payload["color_positions"]] = 1.0
                        mask = tensor(mask_values)
                        if spec.mode == "ov_other":
                            mask = valid.float() - mask
                        if spec.mode in ("q_final", "k_color", "k_context", "identity"):
                            pq, pk = t["q"], t["k"]
                            if spec.mode == "q_final":
                                pq = t["dq"]
                            elif spec.mode in ("k_color", "k_context"):
                                pk = torch.where(mask[..., None].bool(), t["dk"], pk)
                            pd = routing_pattern(pq, pk, valid, module.scaling)
                            delta = contribution_delta("qk", t["p"], t["v"], pd, t["dv"], mask)
                            saved["routed_p"] = cpu_snapshot(pd)
                        else:
                            delta = contribution_delta(
                                spec.mode, t["p"], t["v"], t["dp"], t["dv"], mask
                            )
                            saved["routed_p"] = cpu_snapshot(
                                t["dp"] if spec.mode == "qk" else t["p"]
                            )
                        target = (t["h"] + delta).to(x.dtype)
                    saved["delta"] = cpu_snapshot(delta)
                if target is not None and not torch.equal(target, h):
                    x = x.clone()
                    x[rows, pos, head * d : (head + 1) * d] = target
                saved["applied"] = cpu_snapshot(x[rows, pos, head * d : (head + 1) * d])
                after = x[rows, pos].reshape(b, -1, d)
                others = [j for j in range(before.shape[1]) if j != head]
                saved["sibling_error"] = cpu_snapshot(
                    (before[:, others] - after[:, others]).abs().flatten(1).amax(-1)
                )
                state.clear()
                return (x, *args[1:])

            hooks.extend(
                (
                    module.register_forward_pre_hook(pre, with_kwargs=True),
                    module.q_norm.register_forward_hook(qhook),
                    module.k_norm.register_forward_hook(khook),
                    module.v_proj.register_forward_hook(vhook),
                    module.o_proj.register_forward_pre_hook(ohook),
                )
            )

        try:
            for layer_id in sorted({layer_id for layer_id, _ in spec.upstream}):
                hooks.append(
                    self.model.model.layers[layer_id].self_attn.o_proj.register_forward_pre_hook(
                        upstream_hook(layer_id)
                    )
                )
            for layer_id, h in spec.targets:
                install(layer_id, h)
            result = super()._execute(probes, owner)
            for i, r in enumerate(result):
                r.routing = {
                    key: {
                        name: value[i, : lengths[i]].tolist()
                        if name in ("p", "v", "k", "routed_p")
                        else value[i].tolist()
                        for name, value in snapshot.items()
                    }
                    for key, snapshot in snapshots.items()
                }
                if any(
                    not np.isfinite(v).all() for snap in r.routing.values() for v in snap.values()
                ):
                    raise RuntimeError("Nonfinite routing capture")
            return result
        finally:
            for handle in hooks:
                handle.remove()
            cache.clear()
