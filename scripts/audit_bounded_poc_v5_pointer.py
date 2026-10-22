"""Independent NumPy v5 audit; no torch, production grader, or target execution."""

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.audit_bounded_poc_v4_confirm import arithmetic, softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v5-pointer"
HEADS = ["19:5", "21:13", "22:9"]
MODES = {
    "A": ["ov_color", "ov_other", "qk"],
    "B": ["q_final", "k_context", "ov_context"],
    "C": ["q_final", "k_color", "ov_color"],
}


def read(name):
    return json.loads((OUT / name).read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def load(prefix, name):
    with np.load(OUT / f"{prefix}-{name}.npz", allow_pickle=False) as z:
        return dict(z)


def margin(a):
    return np.asarray(a["logits"], dtype=np.float64) @ np.array([1.0, -1.0])


def statistics(rows, values):
    values = np.asarray(values, dtype=np.float64)
    groups = sorted({r["group"] for r in rows})
    positives = sum(sum(x for r, x in zip(rows, values) if r["group"] == g) > 0 for g in groups)
    return dict(
        median=float(np.median(values)),
        direction=float(np.mean(values > 0)),
        median_abs=float(np.median(np.abs(values))),
        sign_p=sum(math.comb(len(groups), k) for k in range(positives, len(groups) + 1))
        / 2 ** len(groups),
    )


def attention_stats(rows, pc, pd, axis):
    index = {r["id"]: r for r in rows}
    pc, pd = np.asarray(pc, dtype=np.float64), np.asarray(pd, dtype=np.float64)
    changes = np.empty((len(rows), 3, 2), dtype=np.float64)
    for i, r in enumerate(rows):
        c, d = r["target_position"], index[r["donors"][axis]]["target_position"]
        changes[i, :, 0] = pd[i, :, d] - pc[i, :, d]
        changes[i, :, 1] = pc[i, :, c] - pd[i, :, c]

    def summary(x):
        return dict(
            gain=float(np.median(x[:, 0])),
            loss=float(np.median(x[:, 1])),
            gain_direction=float(np.mean(x[:, 0] > 0)),
            loss_direction=float(np.mean(x[:, 1] > 0)),
            joint_direction=float(np.mean(np.all(x > 0, axis=1))),
        )

    return dict(
        coalition=summary(changes.mean(1)), heads=[summary(changes[:, h]) for h in range(3)]
    )


def recompute(rows, base, paths, upstream):
    signs = np.array([r["label"] for r in rows])
    values = {name: signs * (margin(base) - margin(a)) for name, a in paths.items()}
    for axis in ("B", "C"):
        values[f"{axis}_upstream_Q"] = signs * (margin(base) - margin(upstream[axis]))
    effects = {name: statistics(rows, x) for name, x in values.items()}
    attention = {"C_q_final": attention_stats(rows, base["p"], paths["C_q_final"]["routed_p"], "C")}
    for axis in ("B", "C"):
        attention[f"{axis}_upstream_Q"] = attention_stats(
            rows, base["p"], upstream[axis]["routed_p"], axis
        )
    checks = {}
    for name, threshold, direction in [
        ("A_ov_color", 3.0, 0.85),
        ("B_q_final", 3.25, 0.95),
        ("C_q_final", 2.25, 0.95),
        ("B_upstream_Q", 2.5, 0.95),
        ("C_upstream_Q", 2.0, 0.95),
    ]:
        checks[name + "_effect"] = effects[name]["median"] >= threshold
        checks[name + "_direction"] = effects[name]["direction"] >= direction
        checks[name + "_sign_p"] = effects[name]["sign_p"] <= 0.01
    dominance = float(np.mean(values["C_q_final"] > values["C_k_color"]))
    checks.update(
        A_other_abs=effects["A_ov_other"]["median_abs"] <= 0.15,
        A_qk_signed=abs(effects["A_qk"]["median"]) <= 0.25,
        A_specificity_ratio=effects["A_ov_color"]["median"]
        >= 10 * effects["A_ov_other"]["median_abs"],
        B_k_context_exact=bool(np.all(values["B_k_context"] == 0)),
        B_ov_context_exact=bool(np.all(values["B_ov_context"] == 0)),
        C_dominance_median=effects["C_q_final"]["median"] >= 3 * effects["C_k_color"]["median"],
        C_dominance_direction=dominance >= 0.85,
        C_ov_signed=effects["C_ov_color"]["median"] <= 0.25,
    )
    for name, magnitude, direction, first in [
        ("C_q_final", 0.1, 0.9, None),
        ("B_upstream_Q", 0.12, 0.95, 0.2),
        ("C_upstream_Q", 0.1, 0.95, 0.15),
    ]:
        for pole in ("gain", "loss"):
            checks[f"{name}_{pole}"] = attention[name]["coalition"][pole] >= magnitude
            checks[f"{name}_{pole}_direction"] = (
                attention[name]["coalition"][pole + "_direction"] >= direction
            )
            if first is not None:
                checks[f"{name}_first_{pole}"] = attention[name]["heads"][0][pole] >= first
    checks["B_upstream_Q_joint_direction"] = (
        attention["B_upstream_Q"]["coalition"]["joint_direction"] >= 0.95
    )
    gates = {}
    for gate, prefix, excluded in [
        ("A", "A_", "never"),
        ("B", "B_", "B_upstream"),
        ("C", "C_", "C_upstream"),
        ("upstream_B", "B_upstream", "never"),
        ("upstream_C", "C_upstream", "never"),
    ]:
        gates[gate] = all(
            v for k, v in checks.items() if k.startswith(prefix) and not k.startswith(excluded)
        )
    return dict(
        effects=effects,
        attention=attention,
        C_dominance_direction=dominance,
        criteria={k: bool(v) for k, v in checks.items()},
        gates=gates,
        passed=all(gates.values()),
    )


def validate_data(data):
    rows = data["rows"]
    index = {r["id"]: r for r in rows}
    assert len(rows) == 256 and len(index) == 256
    keys = []
    patterns = [
        r"Code ([a-z]+) has color (red|blue|green|yellow)\.",
        r"Item code ([a-z]+): color (red|blue|green|yellow)\.",
        r"Record tag ([a-z]+) is assigned color (red|blue|green|yellow)\.",
    ]
    for split, ng in [("calibration", 8), ("primary", 16), ("vocabulary", 8)]:
        block = [r for r in rows if r["split"] == split]
        groups = {r["group"] for r in block}
        assert len(block) == ng * 8 and len(groups) == ng
        for g in groups:
            members = [r for r in block if r["group"] == g]
            assert len(members) == 8
            assert {(r["k_order"], r["c_order"], r["q_key"]) for r in members} == set(
                itertools.product(range(2), repeat=3)
            )
            keys += members[0]["keys"]
        for r in block:
            lines = r["text"].split("\n")
            bindings = {}
            for line in lines[1:3]:
                matches = [m for p in patterns if (m := re.fullmatch(p, line))]
                assert len(matches) == 1
                bindings[matches[0][1]] = matches[0][2]
            query = lines[5].removeprefix("Requested code: ")
            assert bindings[query] == r["answer_color"]
            assert query == r["keys"][r["q_key"]]
            colors = ("green", "yellow") if split == "vocabulary" else ("red", "blue")
            assert r["label"] == (1 if bindings[query] == colors[0] else -1)
            assert r["target_position"] == r["color_positions"][r["q_key"] ^ r["k_order"]]
            ids = r["token_ids"]
            assert len(ids) <= 128
            for axis in ("A", "B", "C"):
                donor = index[r["donors"][axis]]
                assert donor["group"] == r["group"] and donor["label"] == -r["label"]
                assert donor["donors"][axis] == r["id"] and len(donor["token_ids"]) == len(ids)
                assert donor["color_positions"] == r["color_positions"]
                changes = [j for j, (a, b) in enumerate(zip(ids, donor["token_ids"])) if a != b]
                expected = (
                    r["color_positions"]
                    if axis == "A"
                    else r["key_positions"]
                    if axis == "C"
                    else [r["query_position"]]
                )
                assert changes == expected
                if axis == "C":
                    assert ids[-8:] == donor["token_ids"][-8:]
                if axis != "B":
                    assert Counter(ids) == Counter(donor["token_ids"])
                else:
                    assert ids[: r["context_end"]] == donor["token_ids"][: r["context_end"]]
            assert len(r["key_positions"]) == 2 and r["query_position"] >= r["context_end"]
    assert len(keys) == len(set(keys)) == 64
    old = [
        r
        for v in ("v1", "v2", "v3", "upstream", "v4-confirm")
        for r in json.loads((ROOT / f"data/bounded-poc-{v}/dataset.json").read_text())["rows"]
    ]
    assert not set(keys) & {k for r in old for k in r["keys"]}
    assert not {r["text"] for r in rows} & {r["text"] for r in old}
    return rows


EXPECTED_LIMITS = dict(
    accuracy=0.90,
    sign_p=0.01,
    A_effect=3.0,
    A_direction=0.85,
    A_other_abs=0.15,
    A_qk_signed_abs=0.25,
    A_specificity_ratio=10.0,
    B_effect=3.25,
    B_direction=0.95,
    B_context=0.0,
    C_effect=2.25,
    C_direction=0.95,
    C_shift=0.10,
    C_shift_direction=0.90,
    C_dominance_ratio=3.0,
    C_dominance_direction=0.85,
    C_ov_signed_max=0.25,
    UB_effect=2.50,
    UB_direction=0.95,
    UB_shift=0.12,
    UB_first_shift=0.20,
    UB_joint_direction=0.95,
    UC_effect=2.00,
    UC_direction=0.95,
    UC_shift=0.10,
    UC_first_shift=0.15,
    UC_shift_direction=0.95,
    identity=0.0,
    numpy_delta=0.00003,
)


def audit_controls(controls):
    for kind, control in controls.items():
        if kind == "passed":
            continue
        e = control["evidence"]
        rows = e["rows"]

        def convert(x):
            return {k: np.asarray(v, dtype=np.float64) for k, v in x.items()}

        base = convert(e["base"])
        paths = {k: convert(v) for k, v in e["paths"].items()}
        upstream = {k: convert(v) for k, v in e["upstream"].items()}
        computed = recompute(rows, base, paths, upstream)
        for field in ("effects", "attention", "criteria", "gates", "C_dominance_direction"):
            assert computed[field] == control[field], (kind, field)
        q, k, v, u = (np.asarray(e[f], dtype=np.float64) for f in ("q", "k", "v", "u"))
        query = np.array([r["q_key"] for r in rows])
        order = np.array([r["k_order"] for r in rows])
        slot = query ^ order
        expected_u = np.eye(2)[
            query
            if kind in ("query_identity_only_control", "direct_k_color_only_control")
            else slot
        ]
        expected_q = np.eye(2)[query if kind == "direct_k_color_only_control" else slot]
        if kind == "nuisance":
            expected_q[:] = expected_u[:] = [1, 0]
        np.testing.assert_array_equal(u, expected_u)
        np.testing.assert_array_equal(q, expected_q)
        expected_k = (
            np.array([np.eye(2)[[o, 1 - o]] for o in order])
            if kind == "direct_k_color_only_control"
            else np.repeat(np.eye(2)[None], len(rows), axis=0)
        )
        np.testing.assert_array_equal(k, expected_k)
        np.testing.assert_array_equal(
            v, [[1.0, -1.0] if r["c_order"] == 0 else [-1.0, 1.0] for r in rows]
        )

        def forward(qq, kk, vv):
            logits = (qq[:, None, :] * kk).sum(-1) * 12
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            value = (p * vv).sum(1) * 3 * (-1 if kind == "reversed" else 1)
            return np.stack([value, -value], 1), np.repeat(p[:, None, :], 3, axis=1)

        ll, pp = forward(q, k, v)
        np.testing.assert_array_equal(base["logits"], ll)
        np.testing.assert_array_equal(base["p"], pp)
        index = {r["id"]: i for i, r in enumerate(rows)}
        for axis, modes in MODES.items():
            ix = [index[r["donors"][axis]] for r in rows]
            for mode in modes:
                ll, pp = forward(
                    q[ix] if mode in ("q_final", "qk") else q,
                    k[ix] if mode in ("k_color", "qk") else k,
                    v[ix] if mode == "ov_color" else v,
                )
                np.testing.assert_array_equal(paths[f"{axis}_{mode}"]["logits"], ll)
                np.testing.assert_array_equal(paths[f"{axis}_{mode}"]["routed_p"], pp)
            if axis in ("B", "C"):
                uq = u[ix]
                if kind == "query_identity_only_control":
                    uq = np.eye(2)[uq.argmax(1) ^ order]
                np.testing.assert_array_equal(e["replay_q"][axis], uq)
                ll, pp = forward(uq, k, v)
                np.testing.assert_array_equal(upstream[axis]["logits"], ll)
                np.testing.assert_array_equal(upstream[axis]["routed_p"], pp)
        accuracy = float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0))
        assert accuracy == control["accuracy"]
        assert control["passed"] == (accuracy >= 0.9 and computed["passed"])
        assert control["self_error"] == 0
    assert controls["passed"]
    assert all(
        controls[k]["passed"]
        for k in ("planted_pointer_model", "positive_primary", "positive_vocabulary")
    )
    assert all(
        not controls[k]["passed"]
        for k in (
            "query_identity_only_control",
            "direct_k_color_only_control",
            "nuisance",
            "reversed",
        )
    )
    assert controls["query_identity_only_control"]["gates"]["upstream_B"]
    assert controls["query_identity_only_control"]["effects"]["C_upstream_Q"]["median"] == 0
    assert not controls["direct_k_color_only_control"]["gates"]["C"]
    assert not controls["direct_k_color_only_control"]["gates"]["upstream_C"]


def audit():
    manifest = read("manifest.json")
    mh = sha(OUT / "manifest.json")
    assert manifest["downstream"] == HEADS and manifest["upstream"] == ["17:1", "16:4", "15:4"]
    assert manifest["limits"] == EXPECTED_LIMITS
    for file, digest in manifest["sha256"].items():
        assert sha(ROOT / file) == digest, file
    inventory = json.loads((ROOT / "v5-protected.json").read_text())
    for item in inventory:
        assert sha(ROOT / item["copy"]) == item["sha256"], item["copy"]
        assert sha(Path(item["original"])) == item["sha256"], item["original"]
    for item in json.loads((ROOT / "v4-protected.json").read_text()):
        assert sha(Path(item["original"])) == item["sha256"]
    rows = validate_data(read("dataset.json"))
    runs = [read(f"confirmation-{n}.json") for n in (1, 2)]
    starts = [read(f"confirm{n}-started.json") for n in (1, 2)]
    assert starts[0]["pid"] != starts[1]["pid"]
    assert all(r["manifest_sha256"] == mh for r in starts + runs)
    assert datetime.fromisoformat(manifest["registered_utc"]) < datetime.fromisoformat(
        starts[0]["utc"]
    )
    assert datetime.fromisoformat(runs[0]["completed_utc"]) < datetime.fromisoformat(
        starts[1]["utc"]
    )
    files = sorted(OUT.glob("confirm1-*.npz"))
    assert len(files) == 71
    assert {p.name.replace("confirm1-", "") for p in files} == {
        p.name.replace("confirm2-", "") for p in OUT.glob("confirm2-*.npz")
    }
    array_count = 0
    for file in files:
        with (
            np.load(file, allow_pickle=False) as a,
            np.load(
                file.with_name(file.name.replace("confirm1-", "confirm2-")), allow_pickle=False
            ) as b,
        ):
            assert a.files == b.files
            for key in a.files:
                assert np.isfinite(a[key]).all()
                assert (a[key].dtype, a[key].shape, a[key].tobytes()) == (
                    b[key].dtype,
                    b[key].shape,
                    b[key].tobytes(),
                ), (file.name, key)
                array_count += 1
            assert a["sibling_error"].max() == 0
    max_error = 0.0
    results, diagnostics = {}, {}
    identities = True
    for split in ("calibration", "primary", "vocabulary"):
        block = [r for r in rows if r["split"] == split]
        prefix = f"confirm1-{split}"
        base = load(prefix, "baseline")
        labels = np.array([r["label"] for r in block])
        controls = {}
        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            a = load(prefix, "self-" + mode)
            controls["self_" + mode] = float(np.max(abs(margin(a) - margin(base))))
        ms, es = load(prefix, "self-mediator"), load(prefix, "self-upstream")
        np.testing.assert_array_equal(ms["applied"], base["h"])
        controls["self_upstream_margin"] = float(np.max(abs(margin(es) - margin(base))))
        controls["self_upstream_pattern"] = float(np.max(abs(es["routed_p"] - base["p"])))
        identities &= all(x == 0 for x in controls.values())
        correct = int(np.sum(labels * margin(base) > 0))
        result = dict(accuracy=correct / len(block), controls=controls)
        diagnostics[split] = dict(
            correct=correct,
            total=len(block),
            fused_reconstruction_max=float(base["reconstruction_error"].max()),
        )
        for i, row in enumerate(block):
            n = len(row["token_ids"])
            np.testing.assert_allclose(
                base["p"][i, :, :n], softmax(base["q"][i], base["k"][i], n), atol=3e-6, rtol=0
            )
        if split != "calibration":
            paths = {
                f"{a}_{m}": load(prefix, f"{a}_{m}") for a, modes in MODES.items() for m in modes
            }
            for axis, modes in MODES.items():
                for mode in modes:
                    max_error = max(
                        max_error, arithmetic(block, base, paths[f"{axis}_{mode}"], axis, mode)
                    )
                for h, key in enumerate(HEADS):
                    solo = load(prefix, f"solo-{key.replace(':', '-')}-{axis}")
                    mode = "ov_color" if axis == "A" else "q_final"
                    max_error = max(max_error, arithmetic(block, base, solo, axis, mode, (h,)))
                    diagnostics[split][f"solo_{key}_{axis}"] = statistics(
                        block, labels * (margin(base) - margin(solo))
                    )
            index = {r["id"]: i for i, r in enumerate(block)}
            for i, row in enumerate(block):
                j = index[row["donors"]["B"]]
                for field in ("k", "v"):
                    np.testing.assert_array_equal(
                        base[field][i, :, : row["context_end"]],
                        base[field][j, :, : row["context_end"]],
                    )
            upstream = {}
            for axis in ("B", "C"):
                mid, end = load(prefix, f"{axis}-mediator"), load(prefix, f"{axis}_upstream_Q")
                np.testing.assert_array_equal(mid["applied"], base["h"])
                for i, row in enumerate(block):
                    n = len(row["token_ids"])
                    np.testing.assert_array_equal(mid["k"][i, :, : n - 1], base["k"][i, :, : n - 1])
                    np.testing.assert_allclose(
                        mid["p"][i, :, :n], softmax(mid["q"][i], mid["k"][i], n), atol=3e-6, rtol=0
                    )
                max_error = max(
                    max_error, arithmetic(block, base, end, axis, "q_final", donor_q=mid["q"])
                )
                upstream[axis] = end
            result["scientific"] = recompute(block, base, paths, upstream)
        for run in runs:
            # All statistics must use the same canonical float64 JSON values, exactly.
            assert result == run["splits"][split], split
        results[split] = result
    audit_controls(read("known-controls.json"))
    assert read("validation.json")["passed"]
    gate0 = all(v["accuracy"] >= 0.9 for v in results.values())
    gate1 = all(
        results[s]["scientific"]["gates"][g]
        for s in ("primary", "vocabulary")
        for g in ("A", "B", "C")
    )
    gate2 = all(
        results[s]["scientific"]["gates"][g]
        for s in ("primary", "vocabulary")
        for g in ("upstream_B", "upstream_C")
    )
    gates = dict(
        gate0=gate0,
        gate1=gate1,
        gate2=gate2,
        gate3=gate0 and gate1 and gate2,
        gate4=bool(identities and max_error <= 0.00003),
    )
    record = dict(
        splits=results,
        diagnostics=diagnostics,
        gates=gates,
        passed=all(gates.values()),
        manifest_sha256=mh,
        protected_files=len(inventory),
        repeat_file_pairs=len(files),
        repeat_arrays=array_count,
        maximum_delta_audit_error=max_error,
    )
    facts = {}

    def flatten(value, key=""):
        if isinstance(value, dict):
            for k, x in value.items():
                flatten(x, f"{key}.{k}" if key else k)
        elif isinstance(value, list):
            for k, x in enumerate(value):
                flatten(x, f"{key}.{k}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            facts[key] = value

    flatten(record)
    write("results.json", record)
    write("facts.json", facts)
    write(
        "audit.json",
        dict(
            passed=True,
            manifest_sha256=mh,
            protected_files=len(inventory),
            repeat_file_pairs=len(files),
            repeat_arrays=array_count,
            maximum_delta_error=max_error,
            production_summaries_recomputed=True,
            scope="NumPy only; independent grading and control-stage recomputation",
        ),
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    audit()
