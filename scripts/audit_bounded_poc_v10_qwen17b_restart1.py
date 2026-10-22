"""Independent NumPy v10 audit; no torch, production grader, or target execution."""

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
from src.poc.source_paths import model17_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v10-qwen17b-restart1"
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
        ("A_ov_color", 1.50, 0.80),
        ("B_q_final", 1.50, 0.90),
        ("C_q_final", 1.00, 0.90),
        ("B_upstream_Q", 1.00, 0.90),
        ("C_upstream_Q", 0.75, 0.90),
    ]:
        checks[name + "_effect"] = effects[name]["median"] >= threshold
        checks[name + "_direction"] = effects[name]["direction"] >= direction
        checks[name + "_sign_p"] = effects[name]["sign_p"] <= 0.01
    dominance = float(np.mean(values["C_q_final"] > values["C_k_color"]))
    checks.update(
        A_other_abs=effects["A_ov_other"]["median_abs"] <= 0.20,
        A_qk_signed=abs(effects["A_qk"]["median"]) <= 0.35,
        A_specificity_ratio=effects["A_ov_color"]["median"]
        >= 8 * effects["A_ov_other"]["median_abs"],
        B_k_context_exact=bool(np.array_equal(paths["B_k_context"]["logits"], base["logits"])),
        B_ov_context_exact=bool(np.array_equal(paths["B_ov_context"]["logits"], base["logits"])),
        C_dominance_median=effects["C_q_final"]["median"] >= 1.75 * effects["C_k_color"]["median"],
        C_dominance_direction=dominance >= 0.70,
        C_ov_signed=abs(effects["C_ov_color"]["median"]) <= 0.35,
    )
    for name, magnitude, direction in [
        ("C_q_final", 0.05, 0.85),
        ("B_upstream_Q", 0.05, 0.85),
        ("C_upstream_Q", 0.04, 0.85),
    ]:
        for pole in ("gain", "loss"):
            checks[f"{name}_{pole}"] = attention[name]["coalition"][pole] >= magnitude
            checks[f"{name}_{pole}_direction"] = (
                attention[name]["coalition"][pole + "_direction"] >= direction
            )
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    ix = [lookup[r["donors"]["A"]] for r in rows]
    gap = float(np.median(signs * (margin(base) - margin(base)[ix])))
    checks["A_gap_fraction"] = effects["A_ov_color"]["median"] >= gap / 4
    assert len(checks) == 36
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
        clean_counterfactual_gap=gap,
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
    header = "Each identifier has exactly one assigned color."
    instruction = (
        "Use the assignments above. Return only the color assigned to the requested identifier."
    )
    records = [
        r"Key ([a-z]+) has color (red|blue|green|yellow)\.",
        r"The color associated with identifier ([a-z]+) is (red|blue|green|yellow)\.",
        r"For identifier ([a-z]+), the assigned color is (red|blue|green|yellow)\.",
    ]
    patterns = [
        re.escape(header)
        + "\n"
        + p
        + "\n"
        + p
        + "\n\n"
        + re.escape(instruction)
        + r"\nRequested identifier: ([a-z]+)\nAnswer:"
        for p in records
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
            assert r["text"].count("\n") == 6 and "\r" not in r["text"]
            matches = [(i, m) for i, p in enumerate(patterns) if (m := re.fullmatch(p, r["text"]))]
            assert len(matches) == 1
            fmt, match = matches[0]
            assert r["format"] == ("calibration", "h0", "h1")[fmt]
            assert (fmt == 0) == (split == "calibration")
            bindings = {match[1]: match[2], match[3]: match[4]}
            assert len(bindings) == 2
            query = match[5]
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
                if axis in ("A", "C"):
                    assert ids[-8:] == donor["token_ids"][-8:]
                if axis != "B":
                    assert Counter(ids) == Counter(donor["token_ids"])
                else:
                    assert ids[: r["context_end"]] == donor["token_ids"][: r["context_end"]]
            assert len(r["key_positions"]) == 2 and r["query_position"] >= r["context_end"]
    assert len(keys) == len(set(keys)) == 64
    retired = read("retired-keys.json")["keys"]
    assert not set(keys).intersection(retired)
    return rows


EXPECTED_LIMITS = dict(
    accuracy=0.85,
    sign_p=0.01,
    A_effect=1.50,
    A_gap_fraction=0.25,
    A_direction=0.80,
    A_other_abs=0.20,
    A_qk_signed_abs=0.35,
    A_specificity_ratio=8.0,
    B_effect=1.50,
    B_direction=0.90,
    B_context=0.0,
    C_effect=1.00,
    C_direction=0.90,
    C_shift=0.05,
    C_shift_direction=0.85,
    C_dominance_ratio=1.75,
    C_dominance_direction=0.70,
    C_ov_signed_abs=0.35,
    UB_effect=1.00,
    UB_direction=0.90,
    UB_shift=0.05,
    UB_shift_direction=0.85,
    UC_effect=0.75,
    UC_direction=0.90,
    UC_shift=0.04,
    UC_shift_direction=0.85,
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
        for field in (
            "clean_counterfactual_gap",
            "effects",
            "attention",
            "criteria",
            "gates",
            "C_dominance_direction",
        ):
            assert computed[field] == control[field], (kind, field)
        q, k, v, u = (np.asarray(e[f], dtype=np.float64) for f in ("q", "k", "v", "u"))
        query = np.array([r["q_key"] for r in rows])
        order = np.array([r["k_order"] for r in rows])
        slot = query ^ order
        expected_u = np.eye(2)[
            query if kind in ("literal_query_only_control", "direct_k_color_only_control") else slot
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
                if kind == "literal_query_only_control":
                    uq = np.eye(2)[uq.argmax(1) ^ order]
                np.testing.assert_array_equal(e["replay_q"][axis], uq)
                ll, pp = forward(uq, k, v)
                np.testing.assert_array_equal(upstream[axis]["logits"], ll)
                np.testing.assert_array_equal(upstream[axis]["routed_p"], pp)
        accuracy = float(np.mean(np.array([r["label"] for r in rows]) * margin(base) > 0))
        assert accuracy == control["accuracy"]
        assert control["passed"] == (accuracy >= 0.85 and computed["passed"])
        assert control["self_error"] == 0
    assert controls["passed"]
    assert all(
        controls[k]["passed"]
        for k in ("planted_pointer_model", "positive_primary", "positive_vocabulary")
    )
    assert all(
        not controls[k]["passed"]
        for k in (
            "literal_query_only_control",
            "direct_k_color_only_control",
            "nuisance",
            "reversed",
        )
    )
    assert controls["literal_query_only_control"]["gates"]["upstream_B"]
    assert controls["literal_query_only_control"]["effects"]["C_upstream_Q"]["median"] == 0
    assert not controls["direct_k_color_only_control"]["gates"]["C"]
    assert not controls["direct_k_color_only_control"]["gates"]["upstream_C"]


def frozen_hashes(record, model):
    for name, digest in record["sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in record.get("model_sha256", {}).items():
        assert sha(model / name) == digest, name
    for name, digest in record.get("external_sha256", {}).items():
        assert sha(Path(name)) == digest, name


def npz(name):
    with np.load(OUT / (name + ".npz"), allow_pickle=False) as z:
        out = dict(z)
    assert all(np.isfinite(a).all() for a in out.values()), name
    return out


def verify_mediator(rows, base, mid):
    np.testing.assert_array_equal(mid["applied"], base["h"])
    for i, row in enumerate(rows):
        n = len(row["token_ids"])
        np.testing.assert_array_equal(mid["k"][i, :, : n - 1], base["k"][i, :, : n - 1])
        np.testing.assert_allclose(
            mid["p"][i, :, :n], softmax(mid["q"][i], mid["k"][i], n), atol=3e-6, rtol=0
        )


def audit_discovery(rows, selection):
    labels = np.array([r["label"] for r in rows])
    solos = read("downstream-solos.json")
    expected = {(layer, h) for layer in range(15, 26) for h in range(16)}
    assert len(solos) == 176 and {tuple(r["head"]) for r in solos} == expected
    cache = {}
    worst = 0.0
    first_logits = None
    for layer in range(15, 26):
        a = npz(f"discovery-layer-{layer}")
        if first_logits is not None:
            np.testing.assert_array_equal(a["logits"], first_logits)
        first_logits = a["logits"]
        for h in range(16):
            cache[(layer, h)] = {k: v if k == "logits" else v[:, [h]] for k, v in a.items()}
    eligible = []
    for r in solos:
        head = tuple(r["head"])
        base = cache[head]
        for axis, mode in [("A", "ov_color"), ("B", "q_final")]:
            out = npz(r["effects"][axis]["array"])
            values = labels * (margin(base) - margin(out))
            assert float(np.median(values)) == r["effects"][axis]["median"]
            assert float(np.mean(values > 0)) == r["effects"][axis]["direction"]
            worst = max(worst, arithmetic(rows, base, out, axis, mode, active=(0,)))
        if all(r["effects"][a]["direction"] >= 0.8 for a in ("A", "B")):
            eligible.append(head)
    if len(eligible) < 3:
        assert selection["status"] == "abstained"
        assert set(map(tuple, selection["eligible"])) == set(eligible)
        assert not (OUT / "downstream-selection.json").exists()
        assert not (OUT / "manifest.json").exists()
        assert not list(OUT.glob("confirm*"))
        assert not list(OUT.glob("discovery-D-step*"))
        assert not list(OUT.glob("discovery-U-step*"))
        assert read("discover-complete.json")["microbatches"] == 363
        return worst
    assert selection["status"] == "selected"
    ds = read("downstream-selection.json")
    assert set(map(tuple, ds["eligible"])) == set(eligible)
    chosen = []
    assert len(ds["trace"]) == 3 * len(eligible) - 3
    for step in range(3):
        records = [r for r in ds["trace"] if r["stage"] == step]
        remaining = set(eligible) - set(chosen)
        assert len(records) == len(remaining) and {tuple(r["head"]) for r in records} == remaining
        scores = {}
        for r in records:
            head = tuple(r["head"])
            targets = sorted(chosen + [head])
            assert list(map(tuple, r["coalition"])) == targets
            base = {
                k: cache[targets[0]][k]
                if k == "logits"
                else np.concatenate([cache[t][k] for t in targets], axis=1)
                for k in cache[targets[0]]
            }
            values = []
            assert len(r["arrays"]) == 2
            for axis, mode, tag in zip(("A", "B"), ("ov_color", "q_final"), r["arrays"]):
                out = npz(tag)
                worst = max(
                    worst,
                    arithmetic(rows, base, out, axis, mode, active=tuple(range(len(targets)))),
                )
                values.append(float(np.median(labels * (margin(base) - margin(out)))))
            score = sum(values)
            assert score == r["score"]
            scores[head] = score
        chosen.append(sorted(scores, key=lambda h: (-scores[h], h))[0])
    assert (
        sorted(chosen)
        == list(map(tuple, selection["downstream"]))
        == list(map(tuple, ds["downstream"]))
    )
    del cache
    candidates = {
        (layer, h) for layer in range(10, min(layer for layer, h in chosen)) for h in range(16)
    }
    base = npz("discovery-upstream-baseline")
    np.testing.assert_array_equal(base["logits"], first_logits)
    assert len(selection["trace"]) == 3 * len(candidates) - 3
    chosen = []
    for step in range(3):
        records = [r for r in selection["trace"] if r["stage"] == step]
        remaining = candidates - set(chosen)
        assert len(records) == len(remaining) and {tuple(r["head"]) for r in records} == remaining
        scores = {}
        for r in records:
            head = tuple(r["head"])
            assert list(map(tuple, r["coalition"])) == sorted(chosen + [head])
            values = []
            assert len(r["arrays"]) == 2
            for axis, tag in zip(("B", "C"), r["arrays"]):
                mid = npz(tag + "-mediator")
                out = npz(tag)
                verify_mediator(rows, base, mid)
                worst = max(worst, arithmetic(rows, base, out, axis, "q_final", donor_q=mid["q"]))
                values.append(float(np.median(labels * (margin(base) - margin(out)))))
            score = sum(values)
            assert score == r["score"]
            scores[head] = score
        chosen.append(sorted(scores, key=lambda h: (-scores[h], h))[0])
    assert sorted(chosen) == list(map(tuple, selection["upstream"]))
    assert read("discover-complete.json")["microbatches"] == 346 + 6 * len(eligible) + 12 * len(
        candidates
    )
    return worst


def audit_abstention(model):
    """An independently checked discovery abstention is not a held-out gate result."""
    import subprocess

    from scripts.verify_all_audits import verify_inventory

    counts = verify_inventory(ROOT)
    pre = read("pre-calibration-manifest.json")
    assert pre["limits"] == EXPECTED_LIMITS and pre["key_seed"] == 20261026
    assert pre["downstream_candidates"] == [
        [layer, h] for layer in range(15, 26) for h in range(16)
    ]
    frozen_hashes(pre, model)
    reg = read("pre-calibration-registration.json")
    blob = subprocess.check_output(
        [
            "git",
            "show",
            reg["commit"] + ":data/bounded-poc-v10-qwen17b-restart1/pre-calibration-manifest.json",
        ],
        cwd=ROOT,
    )
    assert (
        hashlib.sha256(blob).hexdigest()
        == reg["manifest_sha256"]
        == sha(OUT / "pre-calibration-manifest.json")
    )
    start = read("discover-started.json")
    assert start["manifest_sha256"] == reg["manifest_sha256"]
    assert datetime.fromisoformat(reg["utc"]) < datetime.fromisoformat(start["utc"])
    rows = validate_data(read("dataset.json"))
    rows = [r for r in rows if r["split"] == "calibration"]
    selection = read("selection.json")
    assert selection["status"] == "abstained"
    worst = audit_discovery(rows, selection)
    audit_controls(read("known-controls.json"))
    base = npz("discovery-layer-15")
    correct = int(np.sum(np.array([r["label"] for r in rows]) * margin(base) > 0))
    record = dict(
        status="discovery_abstention",
        passed=False,
        reason="Fewer than three downstream heads met both solo direction thresholds",
        calibration_correct=correct,
        calibration_total=64,
        eligible_downstream=len(selection["eligible"]),
        held_out_evaluated=False,
        gates={"gate0": None, "gate1": None, "gate2": None, "gate3": False, "gate4": None},
        maximum_delta_audit_error=worst,
        protected_files=counts[0],
    )
    facts = {
        key: value
        for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    write("results.json", record)
    write("facts.json", facts)
    write(
        "audit.json",
        dict(
            passed=True,
            maximum_delta_error=worst,
            scope="Independent complete solo screen and registered abstention; no held-out inference",
        ),
    )
    print(json.dumps(record, indent=2))
    verify_inventory(ROOT)
    return record


def audit(model):
    if not (OUT / "manifest.json").exists():
        return audit_abstention(model)
    import subprocess

    from scripts.verify_all_audits import verify_inventory

    counts = verify_inventory(ROOT)
    pre = read("pre-calibration-manifest.json")
    manifest = read("manifest.json")
    assert pre["limits"] == manifest["limits"] == EXPECTED_LIMITS
    assert pre["revision"] == manifest["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert pre["key_seed"] == 20261026
    assert pre["batch_size"] == 64 and pre["max_seq_len"] == 128
    assert pre["downstream_candidates"] == [
        [layer, h] for layer in range(15, 26) for h in range(16)
    ]
    frozen_hashes(pre, model)
    frozen_hashes(manifest, model)
    selection = read("selection.json")
    assert selection["status"] == "selected"
    for field in ("downstream", "upstream"):
        assert manifest[field] == selection[field] and len(manifest[field]) == 3
        assert len(set(map(tuple, manifest[field]))) == 3
    rows = validate_data(read("dataset.json"))
    for name, stage in [("pre-calibration", "discover"), ("confirmation", "confirm1")]:
        reg = read(name + "-registration.json")
        fn = "pre-calibration-manifest.json" if name == "pre-calibration" else "manifest.json"
        blob = subprocess.check_output(
            ["git", "show", reg["commit"] + ":data/bounded-poc-v10-qwen17b-restart1/" + fn], cwd=ROOT
        )
        assert hashlib.sha256(blob).hexdigest() == reg["manifest_sha256"] == sha(OUT / fn)
        assert datetime.fromisoformat(reg["utc"]) < datetime.fromisoformat(
            read(stage + "-started.json")["utc"]
        )
    assert datetime.fromisoformat(pre["registered_utc"]) < datetime.fromisoformat(
        read("discover-started.json")["utc"]
    )
    assert datetime.fromisoformat(selection["completed_utc"]) < datetime.fromisoformat(
        manifest["registered_utc"]
    )
    assert datetime.fromisoformat(manifest["registered_utc"]) < datetime.fromisoformat(
        read("confirm1-started.json")["utc"]
    )
    assert datetime.fromisoformat(
        read("confirmation-1.json")["completed_utc"]
    ) < datetime.fromisoformat(read("confirm2-started.json")["utc"])
    assert read("confirm1-started.json")["pid"] != read("confirm2-started.json")["pid"]
    for n in (1, 2):
        assert read(f"confirm{n}-started.json")["manifest_sha256"] == sha(OUT / "manifest.json")
        assert read(f"confirmation-{n}.json")["microbatches"] == 88
    worst = audit_discovery([r for r in rows if r["split"] == "calibration"], selection)
    suffixes = (
        ["baseline"]
        + ["self-" + m for m in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk")]
        + ["self-mediator", "self-upstream"]
    )
    suffixes += [f"{a}_{m}" for a, modes in MODES.items() for m in modes]
    suffixes += [f"{a}{s}" for a in ("B", "C") for s in ("-mediator", "_upstream_Q")]
    expected = {
        f"confirm1-{split}-{s}.npz"
        for split in ("calibration", "primary", "vocabulary")
        for s in suffixes
    }
    assert {p.name for p in OUT.glob("confirm1-*.npz")} == expected
    assert {
        p.name.replace("confirm2-", "confirm1-") for p in OUT.glob("confirm2-*.npz")
    } == expected
    count = 0
    for name in sorted(expected):
        a = npz(name[:-4])
        b = npz(name.replace("confirm1-", "confirm2-")[:-4])
        assert a.keys() == b.keys()
        for k in a:
            assert (a[k].dtype, a[k].shape, a[k].tobytes()) == (
                b[k].dtype,
                b[k].shape,
                b[k].tobytes(),
            ), (name, k)
            count += 1
        assert a["sibling_error"].max() == 0
    results = {}
    identities = True
    for split in ("calibration", "primary", "vocabulary"):
        block = [r for r in rows if r["split"] == split]
        prefix = f"confirm1-{split}"
        base = load(prefix, "baseline")
        labels = np.array([r["label"] for r in block])
        controls = {}
        for i, row in enumerate(block):
            n = len(row["token_ids"])
            np.testing.assert_allclose(
                base["p"][i, :, :n], softmax(base["q"][i], base["k"][i], n), atol=3e-6, rtol=0
            )
        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            out = load(prefix, "self-" + mode)
            controls["self_" + mode] = float(np.max(abs(margin(base) - margin(out))))
            np.testing.assert_array_equal(out["delta"], np.zeros_like(out["delta"]))
            np.testing.assert_array_equal(out["applied"], base["h"])
        mid = load(prefix, "self-mediator")
        end = load(prefix, "self-upstream")
        verify_mediator(block, base, mid)
        controls["self_upstream_margin"] = float(np.max(abs(margin(end) - margin(base))))
        controls["self_upstream_pattern"] = float(np.max(abs(end["routed_p"] - base["p"])))
        identities &= all(v == 0 for v in controls.values())
        paths = {}
        upstream = {}
        for axis, modes in MODES.items():
            for mode in modes:
                a = load(prefix, f"{axis}_{mode}")
                paths[f"{axis}_{mode}"] = a
                worst = max(worst, arithmetic(block, base, a, axis, mode))
        lookup = {r["id"]: i for i, r in enumerate(block)}
        for i, row in enumerate(block):
            j = lookup[row["donors"]["B"]]
            for field in ("k", "v"):
                np.testing.assert_array_equal(
                    base[field][i, :, : row["context_end"]], base[field][j, :, : row["context_end"]]
                )
        for axis in ("B", "C"):
            mid = load(prefix, axis + "-mediator")
            out = load(prefix, axis + "_upstream_Q")
            verify_mediator(block, base, mid)
            worst = max(worst, arithmetic(block, base, out, axis, "q_final", donor_q=mid["q"]))
            upstream[axis] = out
        result = dict(
            accuracy=float(np.mean(labels * margin(base) > 0)),
            controls=controls,
            scientific=recompute(block, base, paths, upstream),
        )
        for n in (1, 2):
            assert result == read(f"confirmation-{n}.json")["splits"][split], split
        results[split] = result
    audit_controls(read("known-controls.json"))
    gate0 = all(v["accuracy"] >= 0.85 for v in results.values())
    gate1 = all(v["scientific"]["gates"][g] for v in results.values() for g in ("A", "B", "C"))
    gate2 = all(
        v["scientific"]["gates"][g] for v in results.values() for g in ("upstream_B", "upstream_C")
    )
    gates = dict(
        gate0=gate0,
        gate1=gate1,
        gate2=gate2,
        gate3=gate0 and gate1 and gate2,
        gate4=bool(identities and worst <= 3e-5),
    )
    record = dict(
        splits=results,
        gates=gates,
        passed=all(gates.values()),
        downstream=manifest["downstream"],
        upstream=manifest["upstream"],
        maximum_delta_audit_error=worst,
        repeat_file_pairs=len(expected),
        repeat_arrays=count,
        protected_files=counts[0],
        manifest_sha256=sha(OUT / "manifest.json"),
    )
    facts = {}

    def flatten(v, key=""):
        if isinstance(v, dict):
            for k, x in v.items():
                flatten(x, f"{key}.{k}" if key else k)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                flatten(x, f"{key}.{i}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            facts[key] = v

    flatten(record)
    verify_inventory(ROOT)
    write("results.json", record)
    write("facts.json", facts)
    write(
        "audit.json",
        dict(
            passed=True,
            maximum_delta_error=worst,
            repeat_arrays=count,
            scope="Independent NumPy arithmetic, discovery selection, gates, controls, freezes and hashes",
        ),
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=model17_path())
    audit(p.parse_args().model)
