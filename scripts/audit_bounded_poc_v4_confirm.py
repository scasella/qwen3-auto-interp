"""Independent NumPy-only audit: no torch, target execution, or production graders."""

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v4-confirm"
HEADS = ["19:5", "21:13", "22:9"]
MODES = {
    "A": ["ov_color", "ov_other", "qk"],
    "B": ["q_final", "k_context", "ov_context"],
    "C": ["k_color", "ov_color"],
}


def read(name):
    return json.loads((OUT / name).read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(prefix, name):
    with np.load(OUT / f"{prefix}-{name}.npz", allow_pickle=False) as z:
        return dict(z)


def margin(a):
    return a["logits"][:, 0].astype(float) - a["logits"][:, 1].astype(float)


def write(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def statistics(rows, values):
    groups = sorted({r["group"] for r in rows})
    positive = sum(sum(x for r, x in zip(rows, values) if r["group"] == g) > 0 for g in groups)
    p = sum(math.comb(len(groups), n) for n in range(positive, len(groups) + 1)) / 2 ** len(groups)
    return dict(
        median=float(np.median(values)),
        direction=float(np.count_nonzero(values > 0) / len(values)),
        sign_p=p,
        median_abs=float(np.median(abs(values))),
    )


def attention_stats(rows, pc, pd, axis):
    index = {r["id"]: r for r in rows}
    changes = np.empty((len(rows), 3, 2))
    for i, r in enumerate(rows):
        clean = r["target_position"]
        donor = index[r["donors"][axis]]["target_position"]
        changes[i, :, 0] = pd[i, :, donor] - pc[i, :, donor]
        changes[i, :, 1] = pc[i, :, clean] - pd[i, :, clean]

    def summary(x):
        return dict(
            gain=float(np.median(x[:, 0])),
            loss=float(np.median(x[:, 1])),
            gain_direction=float(np.mean(x[:, 0] > 0)),
            loss_direction=float(np.mean(x[:, 1] > 0)),
            joint_direction=float(np.mean(np.all(x > 0, axis=1))),
        )

    # Keep float32 means to match the explicitly frozen intervention statistic.
    return dict(
        coalition=summary(changes.astype(np.float32).mean(1)),
        heads=[summary(changes[:, h]) for h in range(3)],
    )


def recompute(rows, base, paths, mid, end):
    labels = np.array([r["label"] for r in rows])
    clean = margin(base)
    effects = {k: statistics(rows, labels * (clean - margin(v))) for k, v in paths.items()}
    upstream = statistics(rows, labels * (clean - margin(end)))
    ca = attention_stats(rows, base["p"], paths["C_k_color"]["routed_p"], "C")
    ua = attention_stats(rows, base["p"], mid["p"], "B")

    def ok(s, n):
        return s["median"] >= n and s["direction"] >= 0.95 and s["sign_p"] <= 0.01

    def shift(s, n):
        return s["gain"] >= n and s["loss"] >= n and s["joint_direction"] >= 0.95

    gates = dict(
        A=ok(effects["A_ov_color"], 4.0)
        and effects["A_ov_other"]["median_abs"] <= 0.25
        and effects["A_qk"]["median_abs"] <= 0.25,
        B=ok(effects["B_q_final"], 4.0)
        and np.array_equal(paths["B_k_context"]["logits"], base["logits"])
        and np.array_equal(paths["B_ov_context"]["logits"], base["logits"]),
        C=ok(effects["C_k_color"], 3.5)
        and shift(ca["coalition"], 0.35)
        and effects["C_ov_color"]["median"] <= 0.50,
        upstream=ok(upstream, 4.0)
        and shift(ua["coalition"], 0.30)
        and shift(ua["heads"][0], 0.45)
        and all(h["joint_direction"] >= 0.95 for h in ua["heads"]),
    )
    return dict(
        effects=effects,
        C_attention=ca,
        upstream_effect=upstream,
        upstream_attention=ua,
        gates={k: bool(v) for k, v in gates.items()},
        passed=bool(all(gates.values())),
    )


def assert_same(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, path + f"/{i}")
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert abs(a - b) <= 2e-6, (path, a, b)
    else:
        assert a == b, (path, a, b)


def validate_data(data):
    rows = data["rows"]
    index = {r["id"]: r for r in rows}
    assert len(rows) == 256 and len(index) == 256
    keys = []
    patterns = [
        r"Key ([a-z]+) has color (red|blue|green|yellow)\.",
        r"The color associated with identifier ([a-z]+) is (red|blue|green|yellow)\.",
        r"For identifier ([a-z]+), the assigned color is (red|blue|green|yellow)\.",
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
            query = lines[5].removeprefix("Requested identifier: ")
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
                if axis != "B":
                    assert Counter(ids) == Counter(donor["token_ids"])
                else:
                    assert ids[: r["context_end"]] == donor["token_ids"][: r["context_end"]]
            assert len(r["key_positions"]) == 2 and r["query_position"] >= r["context_end"]
    assert len(keys) == len(set(keys)) == 64
    old = [
        r
        for v in ("v1", "v2", "v3", "upstream")
        for r in json.loads((ROOT / f"data/bounded-poc-{v}/dataset.json").read_text())["rows"]
    ]
    assert not set(keys) & {k for r in old for k in r["keys"]}
    assert not {r["text"] for r in rows} & {r["text"] for r in old}
    return rows


def softmax(q, k, n):
    scores = (q[:, None, :] * k[:, :n, :]).sum(-1, dtype=np.float32) * np.float32(128**-0.5)
    exp = np.exp(scores - scores.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(
        np.float32
    )


def arithmetic(rows, base, out, axis, mode, active=(0, 1, 2), pattern=None, donor_q=None):
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    worst = 0.0
    for i, r in enumerate(rows):
        j = lookup[r["donors"][axis]]
        n = len(r["token_ids"])
        pc, vc = base["p"][i, :, :n], base["v"][i, :, :n]
        pd = base["p"][j, :, :n] if pattern is None else pattern[i, :, :n]
        mask = np.zeros(n, dtype=np.float32)
        mask[r["color_positions"]] = 1
        if mode == "ov_other":
            mask = 1 - mask
        if mode in ("k_context", "ov_context"):
            mask[:] = 0
            mask[: r["context_end"]] = 1
        if mode in ("q_final", "k_color", "k_context"):
            q = base["q"][j] if mode == "q_final" else base["q"][i]
            if donor_q is not None:
                q = donor_q[i]
            k = base["k"][i, :, :n].copy()
            if mode in ("k_color", "k_context"):
                k[:, mask.astype(bool)] = base["k"][j, :, :n][:, mask.astype(bool)]
            pd = softmax(q, k, n)
            expected = ((pd - pc)[..., None] * vc).sum(1, dtype=np.float32)
        elif mode == "qk":
            expected = ((pd - pc)[..., None] * vc).sum(1, dtype=np.float32)
        else:
            expected = (pc[..., None] * (base["v"][j, :, :n] - vc) * mask[None, :, None]).sum(
                1, dtype=np.float32
            )
            pd = pc
        for h in active:
            worst = max(worst, float(np.max(abs(out["delta"][i, h] - expected[h]))))
            np.testing.assert_allclose(out["routed_p"][i, h, :n], pd[h], rtol=0, atol=3e-6)
            np.testing.assert_array_equal(
                out["applied"][i, h], bf16(base["h"][i, h] + out["delta"][i, h])
            )
    assert worst <= 3e-5, ("Intervention delta arithmetic", axis, mode, worst)
    assert out["sibling_error"].max() == 0
    return worst


def audit():
    manifest = read("manifest.json")
    mh = sha(OUT / "manifest.json")
    assert manifest["downstream"] == HEADS and manifest["upstream"] == ["17:1", "16:4", "15:4"]
    expected_limits = dict(
        accuracy=0.90,
        direction=0.95,
        sign_p=0.01,
        A_effect=4.0,
        A_specificity=0.25,
        B_effect=4.0,
        C_effect=3.5,
        C_shift=0.35,
        C_ov_max=0.50,
        upstream_effect=4.0,
        upstream_mean_shift=0.30,
        upstream_first_shift=0.45,
        identity=0.0,
    )
    assert manifest["limits"] == expected_limits
    for file, digest in manifest["sha256"].items():
        assert sha(ROOT / file) == digest, file
    inventory = json.loads((ROOT / "v4-protected.json").read_text())
    for item in inventory:
        assert (
            sha(ROOT / item["copy"]) == item["sha256"]
            and sha(Path(item["original"])) == item["sha256"]
        )
    rows = validate_data(read("dataset.json"))
    runs = [read(f"confirmation-{n}.json") for n in (1, 2)]
    starts = [read(f"confirm{n}-started.json") for n in (1, 2)]
    assert starts[0]["pid"] != starts[1]["pid"]
    assert all(x["manifest_sha256"] == mh for x in starts + runs)
    assert datetime.fromisoformat(manifest["registered_utc"]) < datetime.fromisoformat(
        starts[0]["utc"]
    )
    assert datetime.fromisoformat(runs[0]["completed_utc"]) < datetime.fromisoformat(
        starts[1]["utc"]
    )
    files = sorted(OUT.glob("confirm1-*.npz"))
    pairs = 0
    array_count = 0
    assert {p.name.replace("confirm1-", "") for p in files} == {
        p.name.replace("confirm2-", "") for p in OUT.glob("confirm2-*.npz")
    }
    assert len(files) == 67, ("Unexpected execution coverage", len(files))
    for p in files:
        with (
            np.load(p, allow_pickle=False) as a,
            np.load(p.with_name(p.name.replace("confirm1-", "confirm2-")), allow_pickle=False) as b,
        ):
            assert a.files == b.files
            for k in a.files:
                assert np.isfinite(a[k]).all() and (
                    a[k].dtype == b[k].dtype
                    and a[k].shape == b[k].shape
                    and a[k].tobytes() == b[k].tobytes()
                ), (p.name, k)
                array_count += 1
            assert a["sibling_error"].max() == 0, p.name
        pairs += 1
    results = {}
    max_arithmetic = 0.0
    controls = True
    diagnostics = {}
    for split in ("calibration", "primary", "vocabulary"):
        block = [r for r in rows if r["split"] == split]
        prefix = f"confirm1-{split}"
        base = load(prefix, "baseline")
        cm = margin(base)
        labels = np.array([r["label"] for r in block])
        correct = int(np.sum(labels * cm > 0))
        accuracy = correct / len(block)
        self_errors = {}
        for mode in ("identity", "q_final", "k_color", "ov_color", "ov_other", "qk"):
            out = load(prefix, "self-" + mode)
            self_errors["self_" + mode] = float(np.max(abs(margin(out) - cm)))
            controls &= self_errors["self_" + mode] == 0
        ms = load(prefix, "self-mediator")
        es = load(prefix, "self-upstream")
        self_errors["self_upstream_margin"] = float(np.max(abs(margin(es) - cm)))
        self_errors["self_upstream_pattern"] = float(np.max(abs(ms["p"] - base["p"])))
        controls &= all(x == 0 for x in self_errors.values())
        result = dict(accuracy=accuracy, controls=self_errors)
        diagnostics[split] = dict(
            correct=correct,
            total=len(block),
            fused_reconstruction_max=float(base["reconstruction_error"].max()),
        )
        for i, r in enumerate(block):
            np.testing.assert_allclose(
                base["p"][i, :, : len(r["token_ids"])],
                softmax(base["q"][i], base["k"][i], len(r["token_ids"])),
                atol=3e-6,
                rtol=0,
            )
        if split != "calibration":
            paths = {
                f"{a}_{m}": load(prefix, f"{a}_{m}") for a, mods in MODES.items() for m in mods
            }
            mid, end = load(prefix, "mediator"), load(prefix, "upstream")
            np.testing.assert_array_equal(mid["applied"], base["h"])
            np.testing.assert_array_equal(ms["applied"], base["h"])
            index = {r["id"]: i for i, r in enumerate(block)}
            for i, r in enumerate(block):
                j = index[r["donors"]["B"]]
                stop = r["context_end"]
                for field in ("k", "v"):
                    np.testing.assert_array_equal(
                        base[field][i, :, :stop], base[field][j, :, :stop]
                    )
                np.testing.assert_array_equal(
                    mid["k"][i, :, : len(r["token_ids"]) - 1],
                    base["k"][i, :, : len(r["token_ids"]) - 1],
                )
                np.testing.assert_allclose(
                    mid["p"][i, :, : len(r["token_ids"])],
                    softmax(mid["q"][i], mid["k"][i], len(r["token_ids"])),
                    rtol=0,
                    atol=3e-6,
                )
            for a, mods in MODES.items():
                for mode in mods:
                    max_arithmetic = max(
                        max_arithmetic, arithmetic(block, base, paths[f"{a}_{mode}"], a, mode)
                    )
                for h, key in enumerate(HEADS):
                    solo = load(prefix, f"solo-{key.replace(':', '-')}-{a}")
                    mode = dict(A="ov_color", B="q_final", C="k_color")[a]
                    max_arithmetic = max(
                        max_arithmetic, arithmetic(block, base, solo, a, mode, (h,))
                    )
                    diagnostics[split][f"solo_{key}_{a}"] = statistics(
                        block, labels * (cm - margin(solo))
                    )
            max_arithmetic = max(
                max_arithmetic, arithmetic(block, base, end, "B", "qk", pattern=mid["p"])
            )
            pureq = load(prefix, "upstream-q-only")
            max_arithmetic = max(
                max_arithmetic, arithmetic(block, base, pureq, "B", "q_final", donor_q=mid["q"])
            )
            diagnostics[split]["upstream_Q_only"] = statistics(block, labels * (cm - margin(pureq)))
            diagnostics[split]["upstream_Q_vs_QK_margin_max"] = float(
                np.max(abs(margin(pureq) - margin(end)))
            )
            result["scientific"] = recompute(block, base, paths, mid, end)
        for run in runs:
            assert_same(result, run["splits"][split], split)
        results[split] = result
    planted = read("known-controls.json")
    for kind in ("positive", "vocabulary", "nuisance", "reversed"):
        control = planted[kind]
        ev = control["evidence"]
        rr = ev["rows"]

        def convert(arr):
            return {k: np.array(v) for k, v in arr.items()}

        bb = convert(ev["base"])
        pp = {k: convert(v) for k, v in ev["paths"].items()}
        mm = convert(ev["mediator"])
        computed = recompute(rr, bb, pp, mm, mm)
        for k in ("effects", "C_attention", "upstream_effect", "upstream_attention", "gates"):
            assert_same(computed[k], control[k], f"planted/{kind}/{k}")
        q, k, v = np.array(ev["q"]), np.array(ev["k"]), np.array(ev["v"])

        def plant_output(qq, kk, vv):
            scores = np.sum(qq[:, None, :] * kk, axis=-1) * 12
            weights = np.exp(scores - scores.max(1, keepdims=True))
            weights /= weights.sum(1, keepdims=True)
            values = np.sum(weights * vv, axis=1) * 3 * (-1 if kind == "reversed" else 1)
            return np.stack([values, -values], axis=1)

        np.testing.assert_array_equal(bb["logits"], plant_output(q, k, v))
        indices = {r["id"]: i for i, r in enumerate(rr)}
        for axis, modes in MODES.items():
            ix = [indices[r["donors"][axis]] for r in rr]
            for mode in modes:
                qq = q[ix] if mode in ("q_final", "qk") else q
                kk = k[ix] if mode in ("k_color", "qk") else k
                vv = v[ix] if mode == "ov_color" else v
                np.testing.assert_array_equal(
                    pp[f"{axis}_{mode}"]["logits"], plant_output(qq, kk, vv)
                )
        ix = [indices[r["donors"]["B"]] for r in rr]
        np.testing.assert_array_equal(
            mm["logits"], plant_output(q if kind == "nuisance" else q[ix], k, v)
        )
        accuracy = float(np.mean(np.array([r["label"] for r in rr]) * margin(bb) > 0))
        assert control["passed"] == (computed["passed"] and accuracy >= 0.90)
    assert planted["passed"] and planted["positive"]["passed"] and planted["vocabulary"]["passed"]
    assert not planted["nuisance"]["passed"] and not planted["reversed"]["passed"]
    assert read("validation.json")["passed"]
    gates = dict(
        gate0=all(v["accuracy"] >= 0.90 for v in results.values()),
        gate1=all(results["primary"]["scientific"]["gates"][x] for x in ("A", "B", "C")),
        gate2=results["primary"]["scientific"]["gates"]["upstream"],
        gate3=results["vocabulary"]["scientific"]["passed"],
        gate4=bool(controls),
    )
    record = dict(
        splits=results,
        diagnostics=diagnostics,
        gates=gates,
        passed=all(gates.values()),
        manifest_sha256=mh,
        protected_files=len(inventory),
        repeat_file_pairs=pairs,
        repeat_arrays=array_count,
        maximum_delta_audit_error=max_arithmetic,
    )
    facts = {}

    def flatten(v, key=""):
        if isinstance(v, dict):
            for k, x in v.items():
                flatten(x, f"{key}.{k}" if key else k)
        elif isinstance(v, list):
            for k, x in enumerate(v):
                flatten(x, f"{key}.{k}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            facts[key] = v

    flatten(record)
    write("results.json", record)
    write("facts.json", facts)
    write(
        "audit.json",
        dict(
            passed=True,
            manifest_sha256=mh,
            protected_files=len(inventory),
            repeat_file_pairs=pairs,
            repeat_arrays=array_count,
            maximum_delta_error=max_arithmetic,
            production_summaries_recomputed=True,
            scope="NumPy only; no production graders imported",
        ),
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    audit()
