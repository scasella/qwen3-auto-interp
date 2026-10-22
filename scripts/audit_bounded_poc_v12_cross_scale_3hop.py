"""Standalone NumPy v12 arithmetic, fixed coordinates, replication and gate audit."""

import hashlib
import json
import math
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.audit_bounded_poc_v4_confirm import validate_data
from src.poc.source_paths import model17_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v12-cross-scale-3hop"
U = ((16, 4), (17, 1), (18, 7))
ORIGINAL_U = ((17, 1), (16, 4), (15, 4))
H0 = ((13, 6), (12, 6), (13, 14), (14, 12), (11, 9), (13, 15))
D = ((19, 5), (21, 13), (22, 9))
CANDIDATES = tuple((layer_id, h) for layer_id in range(15) for h in range(16))
MODES = ("full", "v_qspan", "kv_context", "q_final")


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, x):
    with (OUT / name).open("x") as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write("\n")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_hashes(files):
    for path, digest in files.items():
        assert sha(path) == digest, path


def load(name):
    with np.load(OUT / (name + ".npz"), allow_pickle=False) as z:
        a = dict(z)
    assert all(np.isfinite(x).all() for x in a.values()), name
    return a


def margin(a):
    x = np.asarray(a["logits"], dtype=np.float64)
    return x[:, 0] - x[:, 1]


def statistics(rows, v):
    v = np.asarray(v, dtype=np.float64)
    totals = {
        g: sum(x for x, r in zip(v, rows) if r["group"] == g) for g in {r["group"] for r in rows}
    }
    n = len(totals)
    k = sum(t > 0 for t in totals.values())
    return dict(
        median=float(np.median(v)),
        direction=float(np.count_nonzero(v > 0) / len(v)),
        sign_p=sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n,
    )


def recompute(rows, base, paths):
    signs = np.array([r["label"] for r in rows])
    values = {k: signs * (margin(base) - margin(a)) for k, a in paths.items()}
    stats = {k: statistics(rows, v) for k, v in values.items()}
    tests = {}
    diag = {}
    for a in "BC":
        diag[a + "_conservation"] = float(
            np.median(np.abs(margin(paths[a + "_U_full"]) - margin(paths[a + "_upstream_Q"])))
        )
        tests[a + "_conservation"] = diag[a + "_conservation"] <= 0.05
        name = a + "_U_v_qspan"
        s = stats[name]
        tests[name + "_effect"] = s["median"] >= (1.2 if a == "B" else 0.9)
        tests[name + "_direction"] = s["direction"] >= 0.9
        tests[name + "_sign_p"] = s["sign_p"] <= 0.01
        for comp in ("q_final", "kv_context"):
            if a == "B" and comp == "kv_context":
                continue
            key = name + "_over_" + comp
            tests[key + "_median"] = s["median"] >= 3 * stats[a + "_U_" + comp]["median"]
            diag[key] = float(np.mean(values[name] > values[a + "_U_" + comp]))
            tests[key + "_direction"] = diag[key] >= 0.8
        name = a + "_hop0_qspan"
        s = stats[name]
        tests[name + "_effect"] = s["median"] >= (0.4 if a == "B" else 0.25)
        tests[name + "_direction"] = s["direction"] >= 0.8
        tests[name + "_sign_p"] = s["sign_p"] <= 0.01
    diag["B_context_max_logit_error"] = float(
        np.max(
            np.abs(np.asarray(paths["B_U_kv_context"]["logits"], dtype=np.float64) - base["logits"])
        )
    )
    tests["B_context_exact"] = diag["B_context_max_logit_error"] == 0
    correct = int(np.sum(signs * margin(base) > 0))
    accuracy = correct / len(rows)
    gate1 = all(v for k, v in tests.items() if "hop0" not in k)
    gate2 = all(v for k, v in tests.items() if "hop0" in k)
    return dict(
        accuracy=accuracy,
        correct=correct,
        n=len(rows),
        effects=stats,
        diagnostics=diag,
        criteria={k: bool(v) for k, v in tests.items()},
        gate0=accuracy >= 0.8,
        gate1=gate1,
        gate2=gate2,
        passed=accuracy >= 0.8 and gate1 and gate2,
    )


def bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def probability(query, keys):
    scores = np.sum(keys * query, axis=-1, dtype=np.float32)
    scores *= np.float32(query.size**-0.5)
    weights = np.exp(scores - np.max(scores))
    return weights / np.sum(weights, dtype=np.float32)


def audit_path(rows, base, ub, sb, a, axis, mode, heads=(), self_patch=False):
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    donor = [i if self_patch else lookup[r["donors"][axis]] for i, r in enumerate(rows)]
    expected = np.empty_like(ub["h"])
    if mode == "hop0_qspan":
        assert np.all(a["H0_errors"] == 0)
        np.testing.assert_array_equal(
            a["H0_applied"], sb["hop0"][donor][:, [CANDIDATES.index(h) for h in heads]]
        )
    for i, row in enumerate(rows):
        n, qpos = len(row["token_ids"]), row["query_position"]
        # The FIFO batches 64, including mixed template lengths on heldout splits.
        batch_start = (i // 64) * 64
        width = max(len(r["token_ids"]) for r in rows[batch_start : batch_start + 64])
        for h in range(3):
            cq = ub["q"][i, h]
            ck = ub["k"][i, h, :n]
            cv = ub["v"][i, h, :n]
            cp = ub["p"][i, h, :n]
            dq = ub["q"][donor[i], h]
            dk = ub["k"][donor[i], h, :n]
            dv = ub["v"][donor[i], h, :n]
            if mode == "upstream_Q":
                delta = ub["h"][donor[i], h] - ub["h"][i, h]
            else:
                v = cv.copy()
                k = ck.copy()
                query = cq
                if mode in ("full", "q_final"):
                    query = dq
                if mode == "full":
                    k = dk.copy()
                    v = dv.copy()
                elif mode == "kv_context":
                    k[:qpos] = dk[:qpos]
                    v[:qpos] = dv[:qpos]
                elif mode == "v_qspan":
                    v[qpos:] = dv[qpos:]
                elif mode == "hop0_qspan":
                    v[qpos:] = a["H0_vspan"][i, h, : n - qpos]
                    np.testing.assert_array_equal(v[n - 1], cv[n - 1])
                    assert np.all(a["H0_vspan"][i, h, n - qpos :] == 0)
                pp = probability(query, k)
                if mode in ("v_qspan", "hop0_qspan"):
                    terms = cp[:, None] * (v - cv)
                elif mode == "q_final":
                    terms = (pp - cp)[:, None] * cv
                else:
                    terms = pp[:, None] * v - cp[:, None] * cv
                delta = np.sum(terms, axis=0, dtype=np.float32)
            expected[i, h] = delta
            p = probability(a["mediator_q"][i, h], base["k"][i, h, :n])
            np.testing.assert_array_equal(a["D_routed_p"][i, h, :n], p)
            assert np.all(a["D_routed_p"][i, h, n:] == 0)
            padded = np.pad(p, (0, width - n))
            terms = (padded - base["p"][i, h, :width])[:, None] * base["v"][i, h, :width]
            d_delta = np.sum(terms, axis=0, dtype=np.float32)
            np.testing.assert_array_equal(a["D_delta"][i, h], d_delta)
            np.testing.assert_array_equal(a["D_applied"][i, h], bf16(base["h"][i, h] + d_delta))
    np.testing.assert_array_equal(a["u_delta"], expected)
    target = ub["h"] + expected if mode != "upstream_Q" else ub["h"][donor]
    np.testing.assert_array_equal(a["u_target"], target)
    np.testing.assert_array_equal(a["upstream_applied"], bf16(target))
    np.testing.assert_array_equal(a["mediator_applied"], base["h"])
    assert np.all(a["mediator_sibling_error"] == 0) and np.all(a["D_sibling_error"] == 0)
    if self_patch:
        np.testing.assert_array_equal(a["logits"], base["logits"])
        assert np.all(a["u_delta"] == 0) and np.all(a["D_delta"] == 0)
    return 0.0


def audit_baseline(rows, base, ub, sb):
    np.testing.assert_array_equal(base["logits"], ub["logits"])
    for a in (base, ub):
        for i, r in enumerate(rows):
            n = len(r["token_ids"])
            for h in range(3):
                np.testing.assert_array_equal(
                    a["p"][i, h, :n], probability(a["q"][i, h], a["k"][i, h, :n])
                )
    for i, r in enumerate(rows):
        q, n = r["query_position"], len(r["token_ids"])
        np.testing.assert_array_equal(sb["vspan"][i, :, : n - q], ub["v"][i, :, q:n])
        assert np.all(sb["hop0"][i, :, n - 1 - q :] == 0)


def audit_controls(controls):
    for kind, t in controls["cases"].items():
        rows = t["rows"]
        values = np.array(t["values"])
        readout = np.array(t["readout"])

        def forward(v):
            result = []
            for u, r in zip(v @ readout, rows):
                s = np.array([u, -u]) * 2
                p = np.exp(s - max(s))
                p /= sum(p)
                colors = np.array([(-1) ** r["c"], -((-1) ** r["c"])])
                result.append([float(2 * p @ colors), float(-2 * p @ colors)])
            return np.array(result)

        np.testing.assert_array_equal(forward(values), t["baseline"])
        for axis in "BC":
            donor = np.arange(64) ^ (1 if axis == "B" else 4)
            vd = values[donor]
            if kind == "nuisance":
                vd = values
            if kind == "reversed":
                vd = 3 * values
            for mode in (*MODES, "upstream_Q", "hop0_qspan"):
                v = values.copy()
                if mode in ("full", "upstream_Q"):
                    v = vd.copy()
                if mode == "v_qspan":
                    v[:, 2:] = vd[:, 2:]
                if mode == "kv_context" and axis == "C":
                    v[:, :2] = vd[:, :2]
                if mode == "hop0_qspan":
                    v[:, 2:4] = vd[:, 2:4]
                name = axis + ("_" + mode if mode in ("upstream_Q", "hop0_qspan") else "_U_" + mode)
                np.testing.assert_array_equal(t["paths"][name], forward(v))
        result = recompute(
            rows,
            {"logits": np.array(t["baseline"])},
            {k: {"logits": np.array(v)} for k, v in t["paths"].items()},
        )
        assert result == t["result"]
    c = controls["cases"]
    assert c["question_span"]["result"]["passed"]
    assert not c["final_token_only"]["result"]["gate2"]
    assert not c["context_kv_only"]["result"]["gate1"]
    assert not c["nuisance"]["result"]["passed"] and not c["reversed"]["result"]["passed"]
    assert controls["passed"]


def audit_data():
    data = read("dataset.json")
    validate_data(data)
    assert data["seed"] == 20261028 and data["target_forwards"] == 0
    keys = read("retired-keys.json")["keys"]
    assert len(set(keys)) >= 1357
    assert not set(data["keys"]) & set(keys)
    pool = data["admissible_candidates"]
    assert len(pool) >= 300 and pool == sorted(set(pool))
    assert set(pool) <= set(read("whitelist.json")) and not set(pool) & set(keys)
    shuffled = pool.copy()
    random.Random(20261028).shuffle(shuffled)
    assert shuffled[:64] == data["keys"]
    from tokenizers import Tokenizer

    model = model17_path()
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    excluded = (
        set(
            re.findall(
                "[a-z]+",
                "Each record assigns a color to a code or tag. Use the records above. Return only the color for the requested code.".lower(),
            )
        )
        | {"red", "blue", "green", "yellow"}
        | set(keys)
    )
    expected_pool = sorted(
        w
        for w in read("whitelist.json")
        if re.fullmatch("[a-z]{4,9}", w)
        and w not in excluded
        and len(tok.encode(" " + w, add_special_tokens=False).ids) == 1
    )
    assert pool == expected_pool

    for r in data["rows"]:
        enc = tok.encode(r["text"], add_special_tokens=False)
        assert enc.ids == r["token_ids"]
        q = r["query_position"]
        n = len(enc.ids)
        assert r["context_end"] == q and 0 < q < n - 1 and n - q <= 8
        for p, word in [
            (q, r["keys"][r["q_key"]]),
            *zip(r["key_positions"], r["keys"] if r["k_order"] == 0 else r["keys"][::-1]),
        ]:
            assert tok.encode(" " + word, add_special_tokens=False).ids == [enc.ids[p]]
    return data


def audit():
    check_hashes(read("protected.json")["files"])
    manifest = read("manifest.json")
    check_hashes(manifest["sha256"])
    assert manifest["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    for name, coordinates in [
        ("upstream", U),
        ("downstream", D),
        ("hop0", H0),
        ("original_upstream_diagnostic", ORIGINAL_U),
    ]:
        assert manifest[name] == [list(x) for x in coordinates]
    assert manifest["discovery_candidates"] == [] and manifest["discovery_forwards"] == 0
    assert manifest["main_microbatches"] == 168 and manifest["diagnostic_microbatches"] == 60
    data = audit_data()
    audit_controls(read("known-controls.json"))
    commit = read("registration-commit.json")["commit"]
    rel = "data/bounded-poc-v12-cross-scale-3hop/manifest.json"
    assert (
        subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
        == (OUT / "manifest.json").read_bytes()
    )
    committed = int(
        subprocess.check_output(["git", "show", "-s", "--format=%ct", commit], cwd=ROOT)
    )
    assert committed <= datetime.fromisoformat(read("confirm1-started.json")["utc"]).timestamp()
    selected = H0
    first_start = read("confirm1-started.json")
    second_start = read("confirm2-started.json")
    assert first_start["pid"] != second_start["pid"]
    assert datetime.fromisoformat(read("confirmation-1.json")["utc"]) < datetime.fromisoformat(
        second_start["utc"]
    )
    arrays_count = 0
    pairs = 0
    for p in sorted(OUT.glob("confirm1-*.npz")):
        a = load(p.stem)
        b = load(p.stem.replace("confirm1-", "confirm2-", 1))
        assert a.keys() == b.keys()
        for k in a:
            assert (
                a[k].dtype == b[k].dtype
                and a[k].shape == b[k].shape
                and a[k].tobytes() == b[k].tobytes()
            ), (p.name, k)
            arrays_count += 1
        pairs += 1
    assert pairs == 84
    summaries = {}
    diagnostics = {}
    for split in ("calibration", "primary", "vocabulary"):
        rows = [r for r in data["rows"] if r["split"] == split]
        prefix = "confirm1-" + split
        base, ub, sb = [load(prefix + "-base-" + x) for x in ("D", "U", "spatial")]
        audit_baseline(rows, base, ub, sb)
        paths = {}
        for axis in "BC":
            for mode in (*MODES, "upstream_Q", "hop0_qspan"):
                name = axis + ("_" + mode if mode in ("upstream_Q", "hop0_qspan") else "_U_" + mode)
                a = load(prefix + "-" + name)
                audit_path(rows, base, ub, sb, a, axis, mode, tuple(selected))
                paths[name] = a
        for mode in (*MODES, "upstream_Q", "hop0_qspan"):
            audit_path(
                rows, base, ub, sb, load(prefix + "-self-" + mode), "C", mode, tuple(selected), True
            )
        summaries[split] = recompute(rows, base, paths)
        du = load(prefix + "-diagnostic-U")
        ds = load(prefix + "-diagnostic-spatial")
        audit_baseline(rows, base, du, ds)
        diagnostic = {}
        for axis in "BC":
            for mode in ("v_qspan", "hop0_qspan"):
                a = load(prefix + "-diagnostic-" + axis + "-" + mode)
                audit_path(rows, base, du, ds, a, axis, mode, H0)
                diagnostic[axis + "_" + mode] = statistics(
                    rows, np.array([r["label"] for r in rows]) * (margin(base) - margin(a))
                )
        audit_path(
            rows,
            base,
            du,
            ds,
            load(prefix + "-diagnostic-self-hop0_qspan"),
            "C",
            "hop0_qspan",
            H0,
            True,
        )
        diagnostics[split] = diagnostic
    for run in ("1", "2"):
        assert read("confirmation-" + run + ".json")["splits"] == summaries
        assert read("confirmation-" + run + ".json")["batches"] == 228
        assert read("confirmation-" + run + ".json")["main_batches"] == 168
        assert read("confirmation-" + run + ".json")["diagnostic_batches"] == 60
        assert read("confirmation-" + run + ".json")["diagnostics"] == diagnostics
    gates = {f"gate{k}": all(s[f"gate{k}"] for s in summaries.values()) for k in (0, 1, 2)}
    gates["gate3"] = all(gates.values())
    gates["gate4"] = True
    result = dict(
        splits=summaries,
        gates=gates,
        passed=all(gates.values()),
        selected=selected,
        maximum_delta_error=0.0,
        identity_error=0.0,
        repeat_arrays=arrays_count,
        repeat_file_pairs=pairs,
        protected_files=len(read("protected.json")["files"]),
        discovery_forwards=0,
        diagnostic_original_upstream=diagnostics,
    )
    facts = {}

    def flatten(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                flatten(v, (key + "." if key else "") + k)
        elif isinstance(x, (int, float, bool)):
            facts[key] = x

    flatten(result)
    write("results.json", result)
    write("facts.json", facts)
    write(
        "audit.json",
        dict(
            passed=True,
            maximum_delta_error=0.0,
            repeat_file_pairs=pairs,
            repeat_arrays=arrays_count,
            scientific_passed=result["passed"],
        ),
    )
    check_hashes(read("protected.json")["files"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    audit()
