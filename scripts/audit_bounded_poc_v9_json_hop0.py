"""Standalone NumPy audit: no production grader, torch or target execution."""

import hashlib
import itertools
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.audit_bounded_poc_v4_confirm import arithmetic, bf16, softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v9-json-hop0-restart1"


def read(name):
    return json.loads((OUT / name).read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name):
    with np.load(OUT / (name + ".npz"), allow_pickle=False) as z:
        return dict(z)


def write(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


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
        ("A_ov_color", 2.00, 0.75),
        ("B_q_final", 2.00, 0.95),
        ("C_q_final", 1.50, 0.95),
        ("B_upstream_Q", 1.25, 0.95),
        ("C_upstream_Q", 1.00, 0.95),
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
        C_dominance_median=effects["C_q_final"]["median"] >= 2.0 * effects["C_k_color"]["median"],
        C_dominance_direction=dominance >= 0.75,
        C_ov_signed=abs(effects["C_ov_color"]["median"]) <= 0.25,
    )
    for name, magnitude, direction, first in [
        ("C_q_final", 0.06, 0.90, None),
        ("B_upstream_Q", 0.06, 0.90, 0.06),
        ("C_upstream_Q", 0.05, 0.90, 0.05),
    ]:
        for pole in ("gain", "loss"):
            checks[f"{name}_{pole}"] = attention[name]["coalition"][pole] >= magnitude
            checks[f"{name}_{pole}_direction"] = (
                attention[name]["coalition"][pole + "_direction"] >= direction
            )
            if first is not None:
                checks[f"{name}_first_{pole}"] = attention[name]["heads"][0][pole] >= first
                checks[f"{name}_first_{pole}_direction"] = (
                    attention[name]["heads"][0][pole + "_direction"] >= direction
                )
    assert len(checks) == 43
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
    assert len(rows) == 256 and len({r["id"] for r in rows}) == 256
    assert data["seed"] == 20261025 and len(set(data["keys"])) == 64
    by_id = {r["id"]: r for r in rows}
    key_ids = {}
    for split, size, ng in [("calibration", 64, 8), ("primary", 128, 16), ("vocabulary", 64, 8)]:
        block = [r for r in rows if r["split"] == split]
        assert len(block) == size and len({r["group"] for r in block}) == ng
        for group in {r["group"] for r in block}:
            members = [r for r in block if r["group"] == group]
            assert len(members) == 8
            assert {(r["k_order"], r["c_order"], r["q_key"]) for r in members} == set(
                itertools.product(range(2), repeat=3)
            )
        for r in block:
            order = r["keys"][:: 1 if r["k_order"] == 0 else -1]
            colors = ["red", "blue"] if split != "vocabulary" else ["green", "yellow"]
            values = colors[:: 1 if r["c_order"] == 0 else -1]
            lines = r["text"].split("\n")
            assert len(lines) == 7 and lines[3] == "" and lines[6] == "Answer:"
            assert lines[5] == "Requested code: " + r["keys"][r["q_key"]]
            if r["format"] == "h0":
                assert lines[0] == "JSON records:" and lines[4] == "Read the JSON records above."
                for i in (0, 1):
                    assert json.loads(lines[i + 1]) == dict(
                        slot=i + 1, code=" " + order[i] + " ", color=" " + values[i] + " "
                    )
            elif r["format"] == "h1":
                assert lines[0] == "Table records:" and lines[4] == "Read the table records above."
                assert lines[1:3] == [
                    f"| slot {i + 1} | code {order[i]} | color {values[i]} |" for i in (0, 1)
                ]
            else:
                assert r["format"] == "calibration" and split == "calibration"
                assert lines[0] == "Config records:" and lines[4] == "Use the config records above."
                assert lines[1:3] == [
                    f"record_{i + 1}: {{ code: {order[i]} , color: {values[i]} }}" for i in (0, 1)
                ]
            assert (r["format"] == "calibration") == (split == "calibration")
            assert r["label"] == (1 if r["k_order"] ^ r["c_order"] ^ r["q_key"] == 0 else -1)
            assert r["answer_color"] == colors[0 if r["label"] == 1 else 1]
            assert r["target_position"] == r["color_positions"][r["q_key"] ^ r["k_order"]]
            ids = r["token_ids"]
            assert 8 < len(ids) <= 128
            assert [ids[p] for p in r["color_positions"]] == data["output_ids"][split][
                :: 1 if r["c_order"] == 0 else -1
            ]
            for word, pos in zip(
                order + [r["keys"][r["q_key"]]], r["key_positions"] + [r["query_position"]]
            ):
                key_ids.setdefault(word, ids[pos])
                assert key_ids[word] == ids[pos]
            for axis in "ABC":
                d = by_id[r["donors"][axis]]
                assert d["group"] == r["group"] and d["donors"][axis] == r["id"]
                assert d["label"] == -r["label"] and len(d["token_ids"]) == len(ids)
                assert r["color_positions"] == d["color_positions"]
                changed = {i for i, (x, y) in enumerate(zip(ids, d["token_ids"])) if x != y}
                assert changed == set(
                    r["color_positions"]
                    if axis == "A"
                    else r["key_positions"]
                    if axis == "C"
                    else [r["query_position"]]
                )
                if axis != "B":
                    assert (
                        Counter(ids) == Counter(d["token_ids"]) and ids[-8:] == d["token_ids"][-8:]
                    )
                else:
                    assert ids[: r["context_end"]] == d["token_ids"][: r["context_end"]]
    assert set(key_ids) == set(data["keys"]) and len(set(key_ids.values())) == 64
    for p in (ROOT / "data").rglob("dataset.json"):
        if p.parent == OUT:
            continue
        for r in json.loads(p.read_text()).get("rows", []):
            assert not set(r.get("keys", [])) & set(data["keys"])
    return rows


def recompute2(rows, base, direct, internal, hop0, identity):
    signs = np.array([r["label"] for r in rows])
    effects = {
        name: statistics(rows, signs * (margin(base) - margin(a)))
        for name, a in {**internal, **hop0}.items()
    }
    errors = {
        axis: float(np.median(np.abs(margin(direct[axis]) - margin(internal[axis + "_U_full"]))))
        for axis in ("B", "C")
    }
    checks = {a + "_conservation": errors[a] <= 0.05 for a in ("B", "C")}
    pathways = {
        name: effects[name]["median"] >= 0.75
        and effects[name]["direction"] >= 0.90
        and effects[name]["sign_p"] <= 0.01
        for name in ("C_U_q_final", "C_U_kv_context")
    }
    checks["C_internal_pathway"] = any(pathways.values())
    for a, t in [("B", 0.50), ("C", 0.35)]:
        e = effects[a + "_hop0"]
        checks[a + "_hop0_effect"] = e["median"] >= t
        checks[a + "_hop0_direction"] = e["direction"] >= 0.85
        checks[a + "_hop0_sign_p"] = e["sign_p"] <= 0.01
    checks["hop0_identity"] = identity == 0
    return dict(
        effects=effects,
        conservation=errors,
        pathways=pathways,
        criteria={k: bool(v) for k, v in checks.items()},
    )


def reconstruction(rows, base, out, axis, mode):
    index = {r["id"]: i for i, r in enumerate(rows)}
    worst = 0.0
    for i, r in enumerate(rows):
        j = index[r["donors"][axis]]
        n = len(r["token_ids"])
        k, v = base["k"][i, :, :n].copy(), base["v"][i, :, :n].copy()
        q = base["q"][j] if mode == "full" else base["q"][i]
        stop = r["context_end"] if mode == "kv_context" else n
        if mode in ("full", "kv_context", "k_all"):
            k[:, :stop] = base["k"][j, :, :stop]
        if mode in ("full", "kv_context", "v_all"):
            v[:, :stop] = base["v"][j, :, :stop]
        pd = softmax(q, k, n)
        delta = (pd[:, :, None] * v - base["p"][i, :, :n, None] * base["v"][i, :, :n]).sum(
            1, dtype=np.float32
        )
        worst = max(worst, float(np.max(abs(delta - out["delta"][i]))))
        np.testing.assert_array_equal(out["applied"][i], bf16(base["h"][i] + out["delta"][i]))
    assert worst <= 0.00003, (mode, worst)
    return worst


def check_mediator(mid, base, upstream):
    np.testing.assert_array_equal(mid["applied"], base["h"])
    np.testing.assert_array_equal(mid["upstream_applied"], upstream)
    assert np.max(mid["sibling_error"]) == 0


def audit_route(rows, base, ub, coalition, axis, prefix, self_patch=False):
    ix = {r["id"]: i for i, r in enumerate(rows)}
    donor = np.arange(len(rows)) if self_patch else [ix[r["donors"][axis]] for r in rows]
    h0_indices = [int(k.split(":")[0]) * 16 + int(k.split(":")[1]) for k in coalition]
    um = load(prefix + "-U-mediator")
    ur = load(prefix + "-U-replay")
    check_mediator(um, ub, ub["hop0"][donor][:, h0_indices])
    err = arithmetic(rows, ub, ur, axis, "q_final", donor_q=um["q"])
    dm = load(prefix + "-D-mediator")
    end = load(prefix + "-end")
    check_mediator(dm, base, ur["applied"])
    err = max(err, arithmetic(rows, base, end, axis, "q_final", donor_q=dm["q"]))
    return end, err


def audit_controls(controls):
    for name, c in controls.items():
        if name == "passed":
            continue
        e = c["evidence"]
        if not name.startswith("three_"):

            def cv(a):
                return {k: np.asarray(v) for k, v in a.items()}

            r = recompute(
                e["rows"],
                cv(e["base"]),
                {k: cv(v) for k, v in e["paths"].items()},
                {k: cv(v) for k, v in e["upstream"].items()},
            )
            assert r == {k: v for k, v in c.items() if k != "evidence"}
            continue
        h0 = np.asarray(e["h0"])
        value = np.asarray(e["value"])
        rows = e["rows"]
        order = np.array([r["k_order"] for r in rows])
        index = {r["id"]: i for i, r in enumerate(rows)}

        def forward(x):
            if e["kind"] == "literal":
                x = np.eye(2)[x.argmax(1) ^ order]

            def step(y):
                exp = np.exp(12 * y - (12 * y).max(1, keepdims=True))
                return exp / exp.sum(1, keepdims=True)

            u = step(x)
            d = step(u)
            y = (step(d) * value).sum(1) * 3
            if e["kind"] == "reversed":
                y = -y
            return u, d, np.stack((y, -y), 1)

        u, d, ll = forward(h0)
        np.testing.assert_array_equal(u, e["u"])
        np.testing.assert_array_equal(d, e["d"])
        np.testing.assert_array_equal(ll, e["logits"])
        hops = {}
        for a in ("B", "C"):
            u, d, y = forward(h0[[index[r["donors"][a]] for r in rows]])
            for field, x in [("u", u), ("d", d), ("logits", y)]:
                np.testing.assert_array_equal(x, e["stages"][a][field])
            hops[a + "_hop0"] = dict(logits=y)
        direct = {a: hops[a + "_hop0"] for a in ("B", "C")}
        ins = {
            a + "_U_" + m: direct[a]
            for a in ("B", "C")
            for m in ("full", "q_final", "kv_context", "k_all", "v_all")
        }
        result = recompute2(rows, dict(logits=ll), direct, ins, hops, 0.0)
        for k, v in result.items():
            assert c[k] == v
        assert c["passed"] == all(result["criteria"].values())
    assert controls["passed"] and controls["three_positive"]["passed"]
    assert all(not controls["three_" + k]["passed"] for k in ("literal", "nuisance", "reversed"))


def audit():
    assert read("validation.json")["passed"]
    inventory = json.loads((ROOT / "release/protected-sha256.json").read_text())["files"]
    for name, digest in inventory.items():
        assert sha(ROOT / name) == digest, name
    aborted = json.loads(
        (ROOT / "incidents/v9-key-exclusion/protected-aborted-sha256.json").read_text()
    )["files"]
    for name, digest in aborted.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read("stage1.json")["sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read("stage2.json")["sha256"].items():
        assert sha(OUT / name) == digest, name
    for stage in ("stage1", "stage2"):
        commit = read(stage + "-commit.json")["commit"]
        assert (
            subprocess.check_output(
                ["git", "show", f"{commit}:data/bounded-poc-v9-json-hop0-restart1/{stage}.json"],
                cwd=ROOT,
            )
            == (OUT / (stage + ".json")).read_bytes()
        )
    timeline = [
        read("stage1.json")["utc"],
        read("calibration-started.json")["utc"],
        read("calibration-complete.json")["utc"],
        read("stage2.json")["utc"],
        read("confirm1-started.json")["utc"],
        read("confirmation-1.json")["utc"],
        read("confirm2-started.json")["utc"],
        read("confirmation-2.json")["utc"],
    ]
    assert all(
        datetime.fromisoformat(a) < datetime.fromisoformat(b)
        for a, b in zip(timeline, timeline[1:])
    )
    assert read("confirm1-started.json")["pid"] != read("confirm2-started.json")["pid"]
    rows = validate_data(read("dataset.json"))
    audit_controls(read("known-controls.json"))
    assert len(list(OUT.glob("calibration-*.npz"))) == 1922
    screen = read("screen.json")
    assert len(screen) == 240
    assert {x["head"] for x in screen} == {f"{layer}:{h}" for layer in range(15) for h in range(16)}
    selected = [
        x["head"]
        for x in sorted(
            screen,
            key=lambda x: (
                -min(x["B"]["median"] / 0.50, x["C"]["median"] / 0.35),
                -(x["B"]["median"] + x["C"]["median"]),
                *map(int, x["head"].split(":")),
            ),
        )[:3]
    ]
    assert selected == read("stage2.json")["selected"]
    worst = 0.0
    block = [r for r in rows if r["split"] == "calibration"]
    base = load("calibration-baseline")
    ub = load("calibration-U-baseline")
    labels = np.array([r["label"] for r in block])
    for cell in screen:
        for a in ("B", "C"):
            end, err = audit_route(
                block,
                base,
                ub,
                [cell["head"]],
                a,
                "calibration-" + cell["head"].replace(":", "-") + "-" + a,
            )
            worst = max(worst, err)
            assert cell[a] == statistics(block, labels * (margin(base) - margin(end)))
    pairs = sorted(OUT.glob("confirm1-*.npz"))
    count = 0
    assert {p.name.replace("confirm1-", "") for p in pairs} == {
        p.name.replace("confirm2-", "") for p in OUT.glob("confirm2-*.npz")
    }
    for p in pairs:
        a = load(p.stem)
        b = load(p.stem.replace("confirm1-", "confirm2-"))
        assert a.keys() == b.keys()
        for k in a:
            assert np.isfinite(a[k]).all()
            assert (a[k].dtype, a[k].shape, a[k].tobytes()) == (
                b[k].dtype,
                b[k].shape,
                b[k].tobytes(),
            ), (p.name, k)
            count += 1
        assert np.max(a["sibling_error"]) == 0
    results = {}
    for split in ("calibration", "primary", "vocabulary"):
        block = [r for r in rows if r["split"] == split]
        prefix = "confirm1-" + split
        base = load(prefix + "-baseline")
        assert base["logits"].shape == (len(block), 2)
        ub = load(prefix + "-U-baseline")
        end, err = audit_route(block, base, ub, selected, "C", prefix + "-self-hop0", True)
        worst = max(worst, err)
        identity = float(np.max(abs(margin(end) - margin(base))))
        result = dict(
            accuracy=float(np.mean(np.array([r["label"] for r in block]) * margin(base) > 0)),
            identity=identity,
        )
        if split != "calibration":
            modes = {
                "A": ["ov_color", "ov_other", "qk"],
                "B": ["q_final", "k_context", "ov_context"],
                "C": ["q_final", "k_color", "ov_color"],
            }
            paths = {}
            for a, ms in modes.items():
                for m in ms:
                    paths[a + "_" + m] = load(prefix + "-" + a + "_" + m)
                    worst = max(worst, arithmetic(block, base, paths[a + "_" + m], a, m))
            direct, ins, hops = {}, {}, {}
            ix = {r["id"]: i for i, r in enumerate(block)}
            for a in ("B", "C"):
                pre = prefix + "-" + a + "_upstream_Q"
                mid = load(pre + "-D-mediator")
                end = load(pre + "-end")
                check_mediator(mid, base, ub["h"][[ix[r["donors"][a]] for r in block]])
                worst = max(worst, arithmetic(block, base, end, a, "q_final", donor_q=mid["q"]))
                direct[a] = end
                for m in ("full", "q_final", "kv_context", "k_all", "v_all"):
                    pre = prefix + "-" + a + "_U_" + m
                    ur = load(pre + "-U-replay")
                    mid = load(pre + "-D-mediator")
                    end = load(pre + "-end")
                    err = (
                        arithmetic(block, ub, ur, a, "q_final")
                        if m == "q_final"
                        else reconstruction(block, ub, ur, a, m)
                    )
                    check_mediator(mid, base, ur["applied"])
                    worst = max(
                        worst, err, arithmetic(block, base, end, a, "q_final", donor_q=mid["q"])
                    )
                    ins[a + "_U_" + m] = end
                end, err = audit_route(block, base, ub, selected, a, prefix + "-" + a + "_hop0")
                worst = max(worst, err)
                hops[a + "_hop0"] = end
            result["track1"] = recompute(block, base, paths, direct)
            result["track2"] = recompute2(block, base, direct, ins, hops, identity)
        results[split] = result
        for run in (1, 2):
            assert result == read(f"confirmation-{run}.json")["splits"][split], split
    assert len(pairs) == 120, len(pairs)
    t1 = all(r["accuracy"] >= 0.80 for r in results.values()) and all(
        results[s]["track1"]["passed"] for s in ("primary", "vocabulary")
    )
    t2 = all(
        all(results[s]["track2"]["criteria"].values()) for s in ("primary", "vocabulary")
    ) and any(
        all(results[s]["track2"]["pathways"][p] for s in ("primary", "vocabulary"))
        for p in ("C_U_q_final", "C_U_kv_context")
    )
    g4 = worst <= 0.00003 and all(r["identity"] == 0 for r in results.values())
    result = dict(
        splits=results,
        track1=bool(t1),
        track2=bool(t2),
        software_checks=bool(g4),
        maximum_delta_error=worst,
        protected_files=len(inventory),
        aborted_files_verified=len(aborted),
        repeat_pairs=len(pairs),
        repeat_arrays=count,
        selected=selected,
        review_pending=True,
    )
    facts = {}

    def flatten(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                flatten(v, f"{key}.{k}" if key else k)
        elif isinstance(x, list):
            for k, v in enumerate(x):
                flatten(v, f"{key}.{k}")
        elif isinstance(x, (float, int)) and not isinstance(x, bool):
            facts[key] = x

    flatten(result)
    write("results.json", result)
    write("facts.json", facts)
    write("audit.json", dict(passed=True, maximum_delta_error=worst))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
