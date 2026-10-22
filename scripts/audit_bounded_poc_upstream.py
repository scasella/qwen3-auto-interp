"""Independent NumPy audit. No production graders, torch, or model execution."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-upstream"
HEADS = [f"{layer}:{head}" for layer in range(19) for head in range(16)]
TARGETS = ["19:5", "21:13", "22:9"]


def read(name):
    return json.loads((OUT / name).read_text())


def load(name):
    with np.load(OUT / name) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def m(data):
    return data["logits"][..., 0] - data["logits"][..., 1]


def eq(a, b):
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def pvalue(e, rows):
    g = sorted({r["group"] for r in rows})
    k = sum(np.mean([x for x, r in zip(e, rows) if r["group"] == group]) > 0 for group in g)
    return sum(math.comb(len(g), i) for i in range(k, len(g) + 1)) / 2 ** len(g)


def effect(rows, base, changed):
    e = np.array([r["label"] for r in rows]) * (base - changed)
    return dict(
        median=float(np.median(e)),
        direction=float(np.mean(e > 0)),
        group_p=pvalue(e, rows),
        absolute_max=float(np.max(abs(e))),
    )


def transfers(rows, base, changed):
    result = []
    for h in range(3):
        gain = np.array(
            [
                changed[i, h, r["color_positions"][1 - r["query_record"]]]
                - base[i, h, r["color_positions"][1 - r["query_record"]]]
                for i, r in enumerate(rows)
            ]
        )
        loss = np.array(
            [
                base[i, h, r["color_positions"][r["query_record"]]]
                - changed[i, h, r["color_positions"][r["query_record"]]]
                for i, r in enumerate(rows)
            ]
        )
        result.append(
            dict(
                donor_gain=float(np.median(gain)),
                clean_loss=float(np.median(loss)),
                direction=float(np.mean((gain > 0) & (loss > 0))),
                gain_group_p=pvalue(gain, rows),
                loss_group_p=pvalue(loss, rows),
            )
        )
    return result


def upstream_verdict(rows, base, end, pattern):
    e = effect(rows, m(base), m(end))
    shifts = transfers(rows, base["p"], pattern)
    out = e["median"] >= 1.0 and e["direction"] >= 0.85 and e["group_p"] <= 0.05 / 3
    att = all(
        s["donor_gain"] >= 0.30
        and s["clean_loss"] >= 0.30
        and s["direction"] >= 0.85
        and s["gain_group_p"] <= 0.05 / 3
        and s["loss_group_p"] <= 0.05 / 3
        for s in shifts
    )
    return dict(
        passed=bool(out and att),
        margin=e,
        targets=shifts,
        checks=dict(output=bool(out), attention=bool(att)),
    )


def axis_verdict(rows, base, values):
    checks = {}
    stats = {}
    for size in ("single", "coalition"):
        av = effect(rows, m(base), values[f"A_{size}_ov_color"])
        aq = effect(rows, m(base), values[f"A_{size}_qk"])
        bq = effect(rows, m(base), values[f"B_{size}_qk"])
        bv = effect(rows, m(base), values[f"B_{size}_ov_context"])
        stats[size] = dict(A_ov_color=av, A_qk=aq, B_qk=bq, B_ov_context=bv)
        checks[size] = bool(
            av["median"] >= 2.0
            and av["direction"] >= 0.90
            and av["group_p"] <= 0.05 / 3
            and -0.25 <= aq["median"] <= 0.25
            and bq["median"] >= (1.5 if size == "single" else 3.5)
            and bq["direction"] >= 0.90
            and bq["group_p"] <= 0.05 / 3
            and bv["absolute_max"] == 0.0
        )
    return dict(passed=all(checks.values()), checks=checks, statistics=stats)


def donors(rows, axis):
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    return np.array(
        [lookup[r["opposite_id" if axis == "A" else "query_opposite_id"]] for r in rows]
    )


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(
        np.float32
    )


def vector_audit(rows, base, out, mode, axis, active, pattern=None):
    idx = donors(rows, axis)
    pc, vc = base["p"], base["v"]
    pd = base["p"][idx] if pattern is None else pattern
    vd = base["v"][idx]
    if mode == "qk":
        delta = ((pd - pc)[..., None] * vc).sum(2)
    elif mode == "full":
        delta = (pd[..., None] * vd - pc[..., None] * vc).sum(2)
    elif mode == "replace":
        delta = base["h"][idx] - base["h"]
    else:
        mask = np.zeros(pc.shape, dtype=np.float32)
        for i, r in enumerate(rows):
            if mode == "ov_color":
                mask[i, :, r["color_positions"]] = 1
            else:
                mask[i, :, : r["context_end"]] = 1
        delta = (pc[..., None] * (vd - vc) * mask[..., None]).sum(2)
    for h in active:
        np.testing.assert_allclose(delta[:, h], out["delta"][:, h], atol=2e-5, rtol=0)
        np.testing.assert_array_equal(
            bf16(base["h"][:, h] + out["delta"][:, h]), out["applied"][:, h]
        )
    assert np.max(out["sibling_error"]) == 0


def controls(rows, base, prefix, selected, saved):
    identity = load(f"{prefix}-identity.npz")
    err = float(np.max(abs(m(identity) - m(base))))
    eq(err, saved["identity_margin_error"])
    recon = ((base["p"][..., None] * base["v"]).sum(2) - base["h"]).__abs__().max(-1)
    np.testing.assert_allclose(recon, base["reconstruction_error"], atol=2e-5, rtol=0)
    eq(float(base["reconstruction_error"].max()), saved["clean_reconstruction_error"])
    ce = 0.0
    for i, j in enumerate(donors(rows, "B")):
        ce = max(
            ce,
            float(
                np.max(
                    abs(
                        base["v"][i, :, : rows[i]["context_end"]]
                        - base["v"][j, :, : rows[i]["context_end"]]
                    )
                )
            ),
        )
    eq(ce, saved["axis_B_context_V_error"])
    full_error = 0.0
    for axis in ("A", "B"):
        f = load(f"{prefix}-{axis}-full-control.npz")
        r = load(f"{prefix}-{axis}-replace-control.npz")
        vector_audit(rows, base, f, "full", axis, [0])
        vector_audit(rows, base, r, "replace", axis, [0])
        eq(m(f), saved[f"{axis}_full_margin"])
        eq(m(r), saved[f"{axis}_replace_margin"])
        full_error = max(full_error, float(np.max(abs(m(f) - m(r)))))
    eq(full_error, saved["full_replay_error"])
    self_p = self_m = 0.0
    if selected:
        mid = load(f"{prefix}-upstream-self-mediator.npz")
        end = load(f"{prefix}-upstream-self-output.npz")
        self_p = float(np.max(abs(mid["p"] - base["p"])))
        self_m = float(np.max(abs(m(end) - m(base))))
        eq(self_p, saved["upstream_self_pattern_error"])
        eq(self_m, saved["upstream_self_margin_error"])
        np.testing.assert_array_equal(mid["applied"], base["h"])
    passed = (
        err == 0
        and ce == 0
        and full_error <= 0.10
        and base["reconstruction_error"].max() <= 0.05
        and identity["sibling_error"].max() == 0
        and self_p == 0
        and self_m == 0
    )
    assert bool(passed) == saved["passed"]


def main():
    for manifest in ("discovery-registration.json", "manifest.json"):
        reg = read(manifest)
        for name, want in reg["sha256"].items():
            assert digest(ROOT / name) == want, name
    protect = json.loads((ROOT / "upstream-protected.json").read_text())
    for name, want in protect["sha256"].items():
        assert digest(ROOT / name) == want
        assert digest(Path(protect["original_root"]) / name) == want
    data = read("dataset.json")
    by_id = {r["id"]: r for r in data["rows"]}
    for r in data["rows"]:
        b = by_id[r["query_opposite_id"]]
        assert b["assignment"] == r["assignment"] and b["label"] == -r["label"]
        assert b["color_positions"] == r["color_positions"] and b["context_end"] == r["context_end"]
        assert len(b["token_ids"]) == len(r["token_ids"])
        assert b["token_ids"][: r["context_end"]] == r["token_ids"][: r["context_end"]]
        diffs = {i for i, (x, y) in enumerate(zip(b["token_ids"], r["token_ids"])) if x != y}
        assert diffs and diffs <= set(r["query_positions"]) & set(b["query_positions"])
    discovery = read("discovery.json")
    rows = [r for r in data["rows"] if r["split"] == "discovery"]
    base = load("discovery-baseline.npz")
    screen = load("discovery-screen.npz")
    scores = []
    for i, head in enumerate(HEADS):
        shifts = transfers(rows, base["p"], screen["patterns"][i])
        e = effect(rows, m(base), screen["logits"][i, :, 0] - screen["logits"][i, :, 1])
        s = dict(
            head=head,
            minimum_transfer=min(min(x["donor_gain"], x["clean_loss"]) for x in shifts),
            effect=e,
            targets=shifts,
        )
        assert s == discovery["candidates"][i]
        scores.append(s)
        np.testing.assert_array_equal(screen["clamped"][i], base["h"])
        vector_audit(
            rows,
            base,
            dict(
                delta=screen["delta"][i],
                applied=screen["applied"][i],
                sibling_error=screen["sibling_error"][i],
            ),
            "qk",
            "B",
            [0, 1, 2],
            screen["patterns"][i],
        )
    eligible = [s for s in scores if s["minimum_transfer"] > 0 and s["effect"]["median"] > 0]
    chosen = sorted(
        eligible,
        key=lambda s: (
            -s["minimum_transfer"],
            -s["effect"]["median"],
            tuple(map(int, s["head"].split(":"))),
        ),
    )[:3]
    selected = [s["head"] for s in chosen]
    assert selected == discovery["selected"]
    accuracy = float(np.mean(np.array([r["label"] for r in rows]) * m(base) > 0))
    eq(accuracy, discovery["model_accuracy"])
    controls(
        rows,
        base,
        "discovery-selected" if selected else "discovery",
        selected,
        discovery["controls"],
    )
    if selected:
        mid = load("discovery-joint-mediator.npz")
        end = load("discovery-joint-output.npz")
        assert upstream_verdict(rows, base, end, mid["p"]) == discovery["joint"]
        vector_audit(rows, base, end, "qk", "B", [0, 1, 2], mid["p"])
    confirmations = []
    for run in (1, 2):
        saved = read(f"confirmation-{run}.json")
        assert (
            saved["manifest_sha256"] == digest(OUT / "manifest.json")
            and saved["selected"] == selected
        )
        for split in ("primary", "vocabulary"):
            rows = [r for r in data["rows"] if r["split"] == split]
            prefix = f"confirm{run}-{split}"
            base = load(f"{prefix}-baseline.npz")
            report = saved["splits"][split]
            eq(
                float(np.mean(np.array([r["label"] for r in rows]) * m(base) > 0)),
                report["model_accuracy"],
            )
            controls(rows, base, prefix, selected, report["controls"])
            values = load(f"{prefix}-axis-margins.npz")
            for axis in ("A", "B"):
                for size, active in (("single", [0]), ("coalition", [0, 1, 2])):
                    for mode in ("ov_color", "qk") if axis == "A" else ("qk", "ov_context"):
                        name = f"{axis}_{size}_{mode}"
                        out = load(f"{prefix}-{name}.npz")
                        eq(m(out), values[name])
                        vector_audit(rows, base, out, mode, axis, active)
            assert axis_verdict(rows, base, values) == report["axis"]
            if selected:
                mid = load(f"{prefix}-mediator.npz")
                end = load(f"{prefix}-upstream-output.npz")
                np.testing.assert_array_equal(mid["applied"], base["h"])
                vector_audit(rows, base, end, "qk", "B", [0, 1, 2], mid["p"])
                assert upstream_verdict(rows, base, end, mid["p"]) == report["upstream"]
        confirmations.append(saved)
    repeat = {}
    for first in sorted(OUT.glob("confirm1-*.npz")):
        second = OUT / first.name.replace("confirm1-", "confirm2-", 1)
        a, b = load(first.name), load(second.name)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        repeat[first.name] = True
    a, b = confirmations
    gate1 = all(
        r["splits"][s]["axis"]["passed"] for r in confirmations for s in ("primary", "vocabulary")
    )
    gate2 = all(
        r["splits"]["primary"]["upstream"] is not None
        and r["splits"]["primary"]["upstream"]["passed"]
        for r in confirmations
    )
    gate3 = all(
        r["splits"]["vocabulary"]["upstream"] is not None
        and r["splits"]["vocabulary"]["upstream"]["passed"]
        for r in confirmations
    )
    behavior = accuracy >= 0.90 and all(
        r["splits"][s]["model_accuracy"] >= 0.90
        for r in confirmations
        for s in ("primary", "vocabulary")
    )
    instruments = discovery["controls"]["passed"] and all(
        r["splits"][s]["controls"]["passed"]
        for r in confirmations
        for s in ("primary", "vocabulary")
    )
    gates = dict(
        gate1_double_dissociation=gate1,
        gate2_upstream_primary=gate2,
        gate3_upstream_vocabulary=gate3,
        model_behavior=behavior,
        instrumentation=instruments,
        bit_identical_replication=bool(repeat),
        planted_controls=read("known-controls.json")["passed"],
        protected_artifacts=True,
    )
    result = dict(
        audited_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=digest(OUT / "manifest.json"),
        selected=selected,
        selected_discovery_scores=chosen,
        discovery_accuracy=accuracy,
        discovery_joint=discovery["joint"],
        gates=gates,
        passed=all(gates.values()),
        confirmation=confirmations,
        replicated_files=len(repeat),
        protected_files=len(protect["sha256"]),
        audit="Independent NumPy: hashes, donor invariants, 304 rankings, path algebra/applied BF16 vectors, controls, gate arithmetic, group tests and exact repeatability.",
    )
    with (OUT / "results.json").open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    lines = [
        "# bounded-poc-upstream: QK routing and upstream mediation",
        "",
        f"**All preregistered requirements passed: {result['passed']}.**",
        "",
        f"Frozen upstream coalition: {selected or 'abstained'}. Downstream targets remained 19:5, 21:13, 22:9.",
        f"Discovery accuracy: {accuracy:.3%}. The screen tested all 304 heads in layers 0–18 at the final query token, using two-pass QK-only mediation.",
        "",
        "| Requirement | Verdict |",
        "| --- | --- |",
    ]
    lines.extend(f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in gates.items())
    lines += [
        "",
        "All effect values below are median reductions in correct-answer two-token logit margin. They are not necessarily answer flips. No criterion was relaxed, no examples were removed, and no downstream head was reselected.",
        "",
        "| Split / target set | Axis A OV-color | Axis A QK | Axis B QK | Axis B context OV max absolute |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("primary", "vocabulary"):
        for size, s in a["splits"][split]["axis"]["statistics"].items():
            lines.append(
                f"| {split} / {size} | {s['A_ov_color']['median']:.6f} ({s['A_ov_color']['direction']:.2%}) | {s['A_qk']['median']:.6f} | {s['B_qk']['median']:.6f} ({s['B_qk']['direction']:.2%}) | {s['B_ov_context']['absolute_max']:.6f} |"
            )
    for split in ("primary", "vocabulary"):
        s = a["splits"][split]
        u = s["upstream"]
        lines += ["", f"{split.capitalize()} unassisted model accuracy: {s['model_accuracy']:.3%}."]
        if u:
            lines += [
                f"Upstream QK-only mediated effect: **{u['margin']['median']:.6f} logits**, direction **{u['margin']['direction']:.3%}**, group sign p={u['margin']['group_p']:.6g}.",
                "",
                "| Downstream target | Donor-target gain | Clean-target loss | Both changes in predicted direction |",
                "| --- | ---: | ---: | ---: |",
            ]
            lines.extend(
                f"| {h} | {t['donor_gain']:.3%} | {t['clean_loss']:.3%} | {t['direction']:.3%} |"
                for h, t in zip(TARGETS, u["targets"])
            )
    lines += [
        "",
        f"All arrays in {len(repeat)} confirmation file pairs were bit-identical across fresh model processes. All {len(protect['sha256'])} protected v1/v2 files matched both the original folder and isolated snapshot.",
        "",
        "The upstream perturbation is applied in a mediator pass with downstream target outputs clamped clean; only the resulting attention patterns are replayed into clean value vectors in the output pass. Thus the measured effect isolates this controlled QK path and excludes unrestricted upstream effects. Context K/V remain query-independent by causality. This is not a full proof of an abstract key-matching algorithm: an upstream head can convey query identity, and matching may be distributed. A negative result does not exclude upstream context-position paths, larger coalitions, or other architectures.",
        "",
        "New identifiers and frozen v2 grammar were used for 64 discovery, 128 primary, and 64 green/yellow prompts. Same-data repeatability is not independent dataset replication. Output is restricted next-token preference, not unrestricted generation. The named requirements are a conjunction; a successful subtest cannot override a failed global gate. The registered budget ends here.",
        "",
        "[Protocol](bounded-poc-upstream-protocol.md) · [Pre-discovery registration](../data/bounded-poc-upstream/discovery-registration.json) · [Pre-held-out manifest](../data/bounded-poc-upstream/manifest.json) · [Audited results](../data/bounded-poc-upstream/results.json)",
    ]
    (ROOT / "reports/bounded-poc-upstream.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("confirmation", "selected_discovery_scores")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
