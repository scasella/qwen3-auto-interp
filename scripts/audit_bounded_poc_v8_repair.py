"""Independent NumPy-only arithmetic, selection, grading, provenance and repeat audit."""

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v8-repair"
A_GRID = [
    dict(mode="amplify", alpha=a, gamma=g, beta=0.0, alpha22=1.0)
    for a in (1.0, 1.5, 2.0, 2.5)
    for g in (0.0, 0.5, 1.0)
]
B_GRID = [
    dict(mode=m, beta=b, alpha22=a, alpha=1.0, gamma=0.0)
    for m in ("score", "probability")
    for b in (0.5, 0.75, 1.0, 1.25)
    for a in (1.0, 1.5, 2.0)
]
ZERO = dict(mode="amplify", alpha=1.0, gamma=0.0, beta=0.0, alpha22=1.0)
NAIVE = dict(ZERO, alpha=2.0)
LIMITS = dict(
    A_probe_accuracy=0.92,
    A_error_probe=0.75,
    A_accuracy=0.90,
    A_fix_fraction=0.50,
    A_margin=0.20,
    sign_p=0.01,
    delta=0.00003,
    identity=0.0,
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(n):
    return json.loads((OUT / n).read_text())


def load(n):
    with np.load(OUT / n, allow_pickle=False) as z:
        return dict(z)


def save(n, x):
    with (OUT / n).open("x") as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write("\n")


def m(x):
    return np.asarray(x, dtype=np.float64)[:, 0] - np.asarray(x, dtype=np.float64)[:, 1]


def pvalue(rows, gain, mask):
    ids = sorted({r["group"] for r, b in zip(rows, mask) if b})
    positives = 0
    for g in ids:
        values = [x for r, x, b in zip(rows, gain, mask) if r["group"] == g and b]
        positives += np.mean(values) > 0
    n = len(ids)
    return float(sum(math.comb(n, j) for j in range(int(positives), n + 1)) / 2**n) if n else 1.0


def metrics(rows, base, after):
    y = np.asarray([r["label"] for r in rows])
    clean = m(base) * y
    new = m(after) * y
    good = clean > 0
    fixed = new > 0
    gains = new - clean
    eligible = np.asarray(
        [
            r["track"] == "B" and r["query_record"] != 3 and r["label"] != r["last_value"]
            for r in rows
        ]
    )
    bad = eligible & (clean < 0)
    return dict(
        accuracy=float(fixed.mean()),
        baseline_accuracy=float(good.mean()),
        fixed=int((~good & fixed).sum()),
        broken=int((good & ~fixed).sum()),
        baseline_errors=int((~good).sum()),
        median_gain=float(np.median(gains)),
        group_sign_p=pvalue(rows, gains, np.ones(len(rows), bool)),
        minority_accuracy=float(fixed[y == -1].mean()),
        baseline_minority_accuracy=float(good[y == -1].mean()),
        recency_eligible=int(eligible.sum()),
        recency_errors=int((eligible & (new < 0)).sum()),
        baseline_recency_errors=int(bad.sum()),
        recency_gain_median=float(np.median(gains[bad])) if bad.any() else None,
        recency_sign_p=pvalue(rows, gains, bad),
        recency_error_groups=len({r["group"] for r, b in zip(rows, bad) if b}),
    )


def adjudicate_A(rows, base, after, probe):
    z = metrics(rows, base, after)
    y = np.asarray([r["label"] for r in rows])
    bad = y * m(base) <= 0
    hits = y * np.asarray(probe) > 0
    acc = float(hits.mean())
    err = float(hits[bad].mean()) if bad.any() else None
    c = {
        "probe_overall": acc >= 0.92,
        "probe_exceeds_baseline": acc > z["baseline_accuracy"],
        "probe_on_errors": err is not None and err >= 0.75,
        "repair_accuracy": z["accuracy"] >= 0.90,
        "net_rescue": z["fixed"] > z["broken"],
        "fixes_half": z["baseline_errors"] < 4 or z["fixed"] / z["baseline_errors"] >= 0.5,
        "margin_gain": z["median_gain"] >= 0.20,
        "group_sign": z["group_sign_p"] <= 0.01,
        "silver_non_decrease": z["minority_accuracy"] >= z["baseline_minority_accuracy"],
    }
    return dict(
        stats=z,
        probe_accuracy=acc,
        probe_error_accuracy=err,
        criteria=c,
        A1=all(c[k] for k in ("probe_overall", "probe_exceeds_baseline", "probe_on_errors")),
        A2=all(v for k, v in c.items() if not k.startswith("probe_")),
    )


def adjudicate_B(rows, base, after, naive):
    z = metrics(rows, base, after)
    other = metrics(rows, base, naive)
    c = dict(
        net_rescue=z["fixed"] > z["broken"],
        recency_reduced=z["recency_errors"] < z["baseline_recency_errors"],
        beats_naive_net=z["fixed"] - z["broken"] > other["fixed"] - other["broken"],
        fewer_regressions=z["broken"] < other["broken"],
        error_margin_positive=z["recency_gain_median"] is not None and z["recency_gain_median"] > 0,
        error_sign=z["recency_sign_p"] <= 0.01,
    )
    return dict(stats=z, naive=other, criteria=c, passed=all(c.values()))


def softmax(s):
    exp = np.exp(s - s.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def bf16(x):
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(
        np.float32
    )


def arithmetic(rows, a, con, rule):
    worst = 0.0
    for i, row in enumerate(rows):
        n = len(row["token_ids"])
        slots = row["color_positions"]
        for h in range(4):
            q = a["q"][i, h]
            k = a["k"][i, h, :n]
            v = a["v"][i, h, :n]
            p = a["p"][i, h, :n]
            s = a["s"][i, h, :n]
            np.testing.assert_allclose(
                s, (q[None, :] * k).sum(-1) * (q.shape[-1] ** -0.5), atol=2e-5, rtol=0
            )
            np.testing.assert_allclose(p, softmax(s), atol=3e-6, rtol=0)
            routed = p.copy()
            delta = np.zeros_like(a["h"][i, h])
            hh = a["h"][i, h]
            if h < 3 and con is not None:
                beta = rule["beta"]
                if beta:
                    field = "mean_s" if rule["mode"] == "score" else "mean_p"
                    mean = con[field][h]
                    excess = mean - np.mean(mean, dtype=np.float32)
                    if rule["mode"] == "score":
                        edit = s.copy()
                        edit[slots] -= np.float32(beta) * excess
                        routed = softmax(edit)
                    else:
                        routed[slots] -= np.float32(beta) * excess
                qk = ((routed - p)[:, None] * v).sum(0) if beta else np.zeros_like(hh)
                weight = (
                    rule["alpha"]
                    if rule["mode"] == "amplify"
                    else rule["alpha22"]
                    if h == 2
                    else 1.0
                )
                delta = qk + np.float32(weight - 1) * (hh + qk - con["mu"][h])
            err = float(np.max(abs(delta - a["delta"][i, h])))
            worst = max(worst, err)
            np.testing.assert_allclose(a["routed"][i, h, :n], routed, atol=3e-6, rtol=0)
            np.testing.assert_array_equal(a["applied"][i, h], bf16(hh + a["delta"][i, h]))
    assert np.max(a["sibling_error"]) == 0
    assert worst <= 0.00003, worst
    return worst


def validate_data(data):
    rows = data["rows"]
    assert len(rows) == 448 and len({r["id"] for r in rows}) == 448
    expected = {
        "A_calibration": 64,
        "A_primary": 128,
        "B_calibration": 64,
        "B_primary": 128,
        "B_vocabulary": 64,
    }
    assert Counter(r["block"] for r in rows) == expected
    keys = []
    index = {r["id"]: r for r in rows}
    patterns = {
        "a0": r"(?P<key>[a-z]+) has value (?P<color>gold|silver)\.",
        "a1": r"(?P<key>[a-z]+) \| (?P<color>gold|silver)",
        "d0": r"Entry (?P<key>[a-z]+) stores (?P<color>red|blue)\.",
        "d1": r"Stored value for (?P<key>[a-z]+): (?P<color>red|blue)\.",
        "h0": r"Binding for (?P<key>[a-z]+) contains value (?P<color>red|blue|green|yellow)\.",
        "h1": r"The registry stores (?P<color>red|blue|green|yellow) under key (?P<key>[a-z]+)\.",
    }
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group]
        assert len(members) == 8
        keys += members[0]["keys"]
        assert len(set(members[0]["keys"])) == len(members[0]["keys"])
        n = 2 if members[0]["track"] == "A" else 4
        assert Counter(r["query_record"] for r in members) == {i: 8 // n for i in range(n)}
        assert Counter(r["assignment"] for r in members) == {0: 4, 1: 4}
    for r in rows:
        n = 2 if r["track"] == "A" else 4
        lines = r["text"].split("\n")
        assert len(lines) == n + 5
        if n == 2:
            assert (
                lines[0] == "Each key has one value."
                and lines[n + 2]
                == "Find the value assigned to the requested key. Return only that value."
                and lines[-1] == "Value:"
            )
            query = re.fullmatch("Requested key: ([a-z]+)", lines[-2])
        else:
            assert (
                lines[0] == "The registry contains four independent key-to-color bindings."
                and lines[n + 2]
                == "Find the requested key among all four records and return only its stored color."
                and lines[-1] == "Answer:"
            )
            query = re.fullmatch("Requested registry key: ([a-z]+)", lines[-2])
        assert lines[n + 1] == "" and query
        parsed = [re.fullmatch(patterns[r["format"]], line) for line in lines[1 : n + 1]]
        assert all(parsed)
        kk = [x["key"] for x in parsed]
        vv = [x["color"] for x in parsed]
        assert kk == r["ordered_keys"] and vv == r["values"] and query[1] == r["query"]
        assert kk.index(query[1]) == r["query_record"]
        assert r["label"] == (1 if vv[kk.index(query[1])] == r["colors"][0] else -1)
        assert len(r["token_ids"]) <= 128 and len(r["color_positions"]) == n
        for j, color in zip(r["color_positions"], vv):
            assert r["token_ids"][j] == r["output_ids"][r["colors"].index(color)]
        other = index[r["opposite_id"]]
        assert other["label"] == -r["label"] and other["opposite_id"] == r["id"]
        assert (
            Counter(r["token_ids"]) == Counter(other["token_ids"])
            and r["token_ids"][-8:] == other["token_ids"][-8:]
        )
        assert (
            r["color_positions"] == other["color_positions"]
            and r["key_positions"] == other["key_positions"]
        )
    assert len(keys) == len(set(keys)) == 176 and set(keys) == set(data["keys"])
    old = [
        r
        for p in (ROOT / "data").glob("bounded-poc-*/dataset.json")
        if p.parent != OUT
        for r in json.loads(p.read_text())["rows"]
    ]
    old += json.loads((ROOT / "data/lookup-followup-v1/dataset.json").read_text())["rows"]
    assert not set(keys) & {k for r in old for k in r["keys"]}
    return rows


def audit_controls(records):
    for track in ("A", "B"):
        for kind in ("positive", "nuisance", "reversed"):
            entry = records[f"{track}_{kind}"]
            e = entry["evidence"]
            rows = e["rows"]
            y = np.array([r["label"] for r in rows])

            def ll(x):
                return np.stack([x / 2, -x / 2], 1)

            if track == "A":
                signal = y.astype(float) * (
                    0 if kind == "nuisance" else -1 if kind == "reversed" else 1
                )
                np.testing.assert_array_equal(e["h"], np.repeat(signal[:, None, None], 3, 1))
                base = ll(signal + 3)
                after = ll(2.5 * signal + 3) - base.mean(0)
                np.testing.assert_array_equal(e["base"], base)
                np.testing.assert_array_equal(e["repaired"], after)
                assert entry["mean_replace_fixed"] == 0
                result = adjudicate_A(rows, base, after, e["probe"])
                assert entry["passed"] == (result["A1"] and result["A2"])
            else:
                values = np.array(e["values"])
                p = np.array(e["p"])
                h = (p * values).sum(1)
                bypass = np.array(e["bypass"])
                sign = -1 if kind == "reversed" else 1
                prior = np.array([0.15, 0.15, 0.15, 0.55])
                expected = np.stack(
                    [
                        prior
                        if kind == "nuisance"
                        else prior + 0.2 * (np.eye(4)[r["query_record"]] - 0.25)
                        for r in rows
                    ]
                )
                np.testing.assert_array_equal(p, expected)
                np.testing.assert_array_equal(h, e["h"])
                routed = p - (p.mean(0) - p.mean(0).mean())
                np.testing.assert_array_equal(e["routed"], routed)
                base = ll(sign * (h + bypass))
                after = ll(sign * ((routed * values).sum(1) + bypass))
                naive = ll(sign * (2 * h - h.mean() + bypass))
                np.testing.assert_array_equal(base, e["base"])
                np.testing.assert_allclose(naive, e["naive"], atol=1e-15, rtol=0)
                np.testing.assert_array_equal(after, e["repaired"])
                result = adjudicate_B(rows, e["base"], e["repaired"], e["naive"])
                assert result["passed"] == entry["passed"]
            assert result == entry["result"]
    assert records["passed"]
    for t in ("A", "B"):
        assert records[f"{t}_positive"]["passed"]
        assert not records[f"{t}_nuisance"]["passed"] and not records[f"{t}_reversed"]["passed"]
    assert (
        records["A_positive"]["probe_accuracy"] == 1.0
        and records["A_positive"]["result"]["stats"]["accuracy"] == 1.0
    )
    b = records["B_positive"]
    assert np.isclose(b["mean_p"][3], 0.55) and b["result"]["stats"]["accuracy"] == 1.0
    assert b["result"]["naive"]["fixed"] < b["result"]["naive"]["broken"]


def audit():
    first, second = read("pre-calibration-manifest.json"), read("manifest.json")
    for record in (first, second):
        assert (
            record["heads"] == ["19:5", "21:13", "22:9"]
            and record["upstream"] == ["17:1", "16:4", "15:4"]
            and record["late_reference"] == "24:7"
        )
        assert (
            record["grid_A"] == A_GRID and record["grid_B"] == B_GRID and record["limits"] == LIMITS
        )
        for file, digest in record["sha256"].items():
            assert sha(ROOT / file) == digest, file
    inventory = json.loads((ROOT / "v8-protected.json").read_text())
    for item in inventory:
        assert sha(ROOT / item["copy"]) == item["sha256"]
        assert sha(Path(item["original"])) == item["sha256"]
    for v in (4, 5, 6, 7):
        for item in json.loads((ROOT / f"v{v}-protected.json").read_text()):
            assert sha(Path(item["original"])) == item["sha256"]
    rows = validate_data(read("dataset.json"))
    audit_controls(read("known-controls.json"))
    for stage in (1, 2):
        assert read(f"validation-stage{stage}.json")["passed"]
    cal = read("calibration.json")
    starts = [read(x + "-started.json") for x in ("calibration", "confirm1", "confirm2")]
    runs = [read(f"confirmation-{i}.json") for i in (1, 2)]
    dates = [
        first["registered_utc"],
        starts[0]["utc"],
        cal["completed_utc"],
        second["registered_utc"],
        starts[1]["utc"],
        runs[0]["completed_utc"],
        starts[2]["utc"],
        runs[1]["completed_utc"],
    ]
    assert all(
        datetime.fromisoformat(a) < datetime.fromisoformat(b) for a, b in zip(dates, dates[1:])
    )
    assert len({x["pid"] for x in starts}) == 3
    assert (
        starts[0]["manifest_sha256"]
        == cal["pre_manifest_sha256"]
        == sha(OUT / "pre-calibration-manifest.json")
    )
    assert all(x["manifest_sha256"] == sha(OUT / "manifest.json") for x in starts[1:] + runs)
    worst = 0.0
    for track, grid in [("A", A_GRID), ("B", B_GRID)]:
        block = [r for r in rows if r["block"] == track + "_calibration"]
        base = load(f"cal-{track}-baseline.npz")
        con = load(f"cal-{track}-constants.npz")
        np.testing.assert_array_equal(
            con["mu"], base["h"][:, :3].astype(float).mean(0).astype(np.float32)
        )
        np.testing.assert_array_equal(con["mean_logits"], base["logits"].astype(float).mean(0))
        for positions, prefix in [("color_positions", ""), ("key_positions", "key_")]:
            for field in ("s", "p"):
                extracted = np.array(
                    [
                        [base[field][i, h, r[positions]] for h in range(3)]
                        for i, r in enumerate(block)
                    ]
                )
                np.testing.assert_array_equal(
                    con[prefix + "mean_" + field],
                    extracted.mean(0, dtype=np.float64).astype(np.float32),
                )
        identity = load(f"cal-{track}-identity.npz")
        np.testing.assert_array_equal(base["logits"], identity["logits"])
        worst = max(
            worst, arithmetic(block, base, None, ZERO), arithmetic(block, identity, con, ZERO)
        )
        candidates = []
        if track == "A":
            for alpha in (1.0, 1.5, 2.0, 2.5):
                a = load(f"cal-A-alpha{alpha}.npz")
                worst = max(worst, arithmetic(block, a, con, dict(ZERO, alpha=alpha)))
                candidates += [
                    a["logits"].astype(float) - g * con["mean_logits"] for g in (0.0, 0.5, 1.0)
                ]
            x = base["h"][:, :3].astype(float).reshape(len(block), -1)
            y = np.array([r["label"] for r in block])

            def fit(x, y):
                a = x[y == 1].mean(0)
                b = x[y == -1].mean(0)
                return a - b, float(-0.5 * ((a * a).sum() - (b * b).sum()))

            w, b = fit(x, y)
            probe = load("cal-A-probe.npz")
            np.testing.assert_array_equal(w, probe["w"])
            assert b == probe["b"]
            lgo = np.empty(len(block))
            for group in sorted({r["group"] for r in block}):
                ix = np.array([r["group"] == group for r in block])
                ww, bb = fit(x[~ix], y[~ix])
                lgo[ix] = x[ix] @ ww + bb
            np.testing.assert_array_equal(lgo, probe["lgo"])
            assert float(np.mean(y * lgo > 0)) == cal["probe_lgo_accuracy"]
            weights = load("cal-A-weights.npz")
            direction = np.einsum(
                "hid,i->hd",
                weights["wo"].astype(float),
                weights["norm"].astype(float)
                * (
                    weights["unembedding"][0].astype(float)
                    - weights["unembedding"][1].astype(float)
                ),
            )
            np.testing.assert_array_equal(direction, weights["directions"])
        else:
            for i, rule in enumerate(grid):
                a = load(f"cal-B-candidate{i:02}.npz")
                worst = max(worst, arithmetic(block, a, con, rule))
                candidates.append(a["logits"])
            worst = max(worst, arithmetic(block, load("cal-B-naive.npz"), con, NAIVE))
        np.testing.assert_array_equal(
            load(f"cal-{track}-candidate-logits.npz")["logits"], np.stack(candidates)
        )
        scores = [metrics(block, base["logits"], a) for a in candidates]
        assert scores == cal["candidate_scores"][track]
        winner = sorted(
            range(len(grid)),
            key=lambda i: (
                -(scores[i]["fixed"] - scores[i]["broken"]),
                scores[i]["broken"],
                -scores[i]["median_gain"],
                i,
            ),
        )[0]
        assert (
            cal["selected"][track]
            == dict(index=winner, rule=grid[winner])
            == second["selected"][track]
        )
    files = sorted(OUT.glob("confirm1-*.npz"))
    assert len(files) == 14
    assert {p.name.replace("confirm1-", "") for p in files} == {
        p.name.replace("confirm2-", "") for p in OUT.glob("confirm2-*.npz")
    }
    arrays = 0
    for path in files:
        a = load(path.name)
        b = load(path.name.replace("confirm1-", "confirm2-"))
        assert a.keys() == b.keys()
        for key, value in a.items():
            assert np.isfinite(value).all()
            assert (value.shape, value.dtype, value.tobytes()) == (
                b[key].shape,
                b[key].dtype,
                b[key].tobytes(),
            ), (path.name, key)
            arrays += 1
    results = {}
    for blockname in ("A_primary", "B_primary", "B_vocabulary"):
        block = [r for r in rows if r["block"] == blockname]
        track = blockname[0]
        con = load(f"cal-{track}-constants.npz")
        rule = second["selected"][track]["rule"]
        prefix = "confirm1-" + blockname
        base = load(prefix + "-baseline.npz")
        after = load(prefix + "-repaired.npz")
        identity = load(prefix + "-identity.npz")
        derived = load(prefix + "-derived.npz")
        worst = max(
            worst,
            arithmetic(block, base, None, ZERO),
            arithmetic(block, after, con, rule),
            arithmetic(block, identity, con, ZERO),
        )
        z = after["logits"].astype(float) - rule["gamma"] * con["mean_logits"]
        np.testing.assert_array_equal(z, derived["centered_logits"])
        if track == "A":
            h = base["h"][:, :3].astype(float)
            probe = load("cal-A-probe.npz")
            score = h.reshape(len(block), -1) @ probe["w"] + probe["b"]
            np.testing.assert_array_equal(score, derived["probe"])
            directions = load("cal-A-weights.npz")["directions"]
            ov = ((h - con["mu"]) * directions[:3]).sum((1, 2))
            np.testing.assert_array_equal(ov, derived["ov_readout"])
            np.testing.assert_array_equal(
                (base["h"][:, 3].astype(float) * directions[3]).sum(1),
                derived["late_reference_projection"],
            )
            result = adjudicate_A(block, base["logits"], z, score)
            y = np.array([r["label"] for r in block])
            bad = y * m(base["logits"]) <= 0
            result["unsupervised_ov_accuracy"] = float(np.mean(y * ov > 0))
            result["unsupervised_ov_on_errors"] = (
                float(np.mean((y * ov > 0)[bad])) if bad.any() else None
            )
        else:
            naive = load(prefix + "-naive.npz")
            worst = max(worst, arithmetic(block, naive, con, NAIVE))
            result = adjudicate_B(block, base["logits"], z, naive["logits"])
        result["identity_error"] = float(np.max(abs(m(base["logits"]) - m(identity["logits"]))))
        assert result["identity_error"] == 0
        for run in runs:
            assert run["blocks"][blockname] == result, blockname
        results[blockname] = result
    gates = dict(
        A1=results["A_primary"]["A1"],
        A2=results["A_primary"]["A2"],
        B1=all(results[b]["passed"] for b in ("B_primary", "B_vocabulary")),
        C=True,
    )
    record = dict(
        gates=gates,
        passed=all(gates.values()),
        blocks=results,
        selected=second["selected"],
        protected_files=len(inventory),
        repeat_file_pairs=len(files),
        repeat_arrays=arrays,
        maximum_delta_error=worst,
        manifest_sha256=sha(OUT / "manifest.json"),
        pre_calibration_manifest_sha256=sha(OUT / "pre-calibration-manifest.json"),
    )
    facts = {}

    def flatten(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                flatten(v, f"{key}.{k}" if key else k)
        elif isinstance(x, (float, int)) and not isinstance(x, bool):
            facts[key] = x

    flatten(record)
    save("results.json", record)
    save("facts.json", facts)
    save(
        "audit.json",
        dict(
            passed=True,
            protected_files=len(inventory),
            repeat_file_pairs=len(files),
            repeat_arrays=arrays,
            maximum_delta_error=worst,
            independent_selection_and_grading=True,
        ),
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    audit()
