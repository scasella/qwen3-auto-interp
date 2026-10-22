"""Response-blind data, frozen calibration selection and prospective repair grading."""

import itertools
import math
import random
import re
from collections import Counter

import numpy as np

from src.poc.v5_templates import LEXICON

# Fixed neutral extension, approved before tokenizer filtering or any v8 response.
EXTENSION = """
ability absence academy access accord account achieve acquire action active actor actual
adapt addition address adjust admin admit adopt advance advice advise affair affect affirm
agency agree ahead allow alter amount answer appear appoint approve archive arrange arrive
article aside asset assign attend avenue avoid aware basic basis become before begin behave
being beyond branch brief bring broad build bureau calendar campaign capable capital capture
carry category central centre certain channel charge chief citizen clarify clerk colleague
collect college command comment commerce compact company complete complex comply compose
concept conclude concrete connect consider consist consult continue contract convene convert
correct counsel course court credit culture current daily data decide declare decline default
deliver deputy design desire desk detect device devise dialogue differ digest discuss dismiss
display dispute divide document doubt drive during early economy educate editor either element
email emerge enforce engage enhance enjoy enough enroll entire entry episode estate ethics
evaluate examine example exceed execute exhibit exist expand expense express extra faculty
feature fellow figure fiscal focus format forward freedom fulfil further gallery gender global
grant greater ground growth guard guest guidance hearing hence higher history holding humble
ideal identify improve include increase indeed initial inquire insight inspect install instance
instead instruct integral interest interim internal invite involve joint journal journey justice
keeping knowledge later leader legacy library likely listing local locate maintain majority
master meaning measure meeting message middle migrate minute missing modify moment motion
nation native neither normal object observe occur operate opinion ordinary outline overall
package passage perform perhaps personal platform plural portion possible practice predict
prepare present preserve press prevent primary proceed process produce project promote prompt
proper propose protect provide publish purpose qualify quality quarter question queue reader
ready receive recent recover reflect regular reject release relevant relief remind remote renew
repair replace respond restore return reveal revenue routine salary satisfy schedule scholar
scheme science second section secure segment sequence series service several shape should
similar since single situation social society solve special specify standard start station
steady strategy strict subject succeed suggest summary supply support suppose system teach
teacher temporal terminal their thing together transfer travel treaty trial trust tutor typical
under union unique unite update usage useful usual valid validate value various verbal verify
version viable visit voice volume voter warrant weekly welfare whether while willing within
working writer writing yield youth absolute accepted achieved admitted adoption advisory
analysis approved arranged arrival assessed assigned assisted assumed attached attempted
attitude awarded becoming belonging calling capacity caution chairman checking clearly
commence commerce committee composed computer concepts conclude consider continual contracts
corrected counted coverage creation customer decision declared delivery dependent described
designed directed division economic eligible emphasis employer enclosed entitled establish
estimate evidence exchange existing expanded expected external familiar federal feedback
finished flexible founding frequent friendly fulfill graduate helpful holiday identify incoming
increase indicate indirect informed intended involved limited location movement national
necessary official opening optional original planning position prepared presence pressure
priority private problem provided purchase reaction received recovery regarded register
relation release reminder required reserved resolve resource response returned revision
sensible separate setting sharing showing software solution speaking specific stating
statement storage student suitable supplied teaching technical temporary tendency thinking
training transfer tutorial universal unknown updated utility various viewing visible visited
voluntary waiting warning welcome western whoever writing zone abroad accept accountancy
activation advocate aftermath algebra analogy anthem appoint archive argument arithmetic
assembly associate autonomy bachelor behavior biography briefing candidate ceremony charter
citizenship classroom coalition commentary compiler composer consensus consultant convention
coordination correspondence credibility delegation democracy discourse discovery discretion
education election enquiry equality exception exercise expertise federation fellowship
formation framework friendship governance hypothesis identity inclusion inference initiative
innovation inquiry intention interview introduction judgement judgment leadership legislation
membership ministry narrative negotiation objective observation operation opposition ownership
participation partnership perspective petition philosophy planning population precedent
precision preference principle procedure profession proposal publication qualification
reasoning recognition recommendation regulation representation requirement resolution
responsibility scholarship secretary selection seminar signature significance simulation
speaker specialist specification statistics subscriber suggestion supervision symposium
technology testimony tradition transaction translation understanding university variation
verification vocabulary volunteer workshop workspace
""".split()

FORMATS_A = {"a0": "{key} has value {color}.", "a1": "{key} | {color}"}
FORMATS_B = {
    "d0": "Entry {key} stores {color}.",
    "d1": "Stored value for {key}: {color}.",
    "h0": "Binding for {key} contains value {color}.",
    "h1": "The registry stores {color} under key {key}.",
}
HEADER_A = "Each key has one value."
INSTRUCTION_A = "Find the value assigned to the requested key. Return only that value."
HEADER_B = "The registry contains four independent key-to-color bindings."
INSTRUCTION_B = "Find the requested key among all four records and return only its stored color."
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
IDENTITY = dict(mode="amplify", alpha=1.0, beta=0.0, alpha22=1.0, gamma=0.0)
NAIVE = dict(IDENTITY, alpha=2.0)
HEADS = ["19:5", "21:13", "22:9"]
UPSTREAM = ["17:1", "16:4", "15:4"]
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


def render(track, fmt, keys, values, query):
    templates = FORMATS_A if track == "A" else FORMATS_B
    records = "\n".join(templates[fmt].format(key=k, color=v) for k, v in zip(keys, values))
    if track == "A":
        return f"{HEADER_A}\n{records}\n\n{INSTRUCTION_A}\nRequested key: {query}\nValue:"
    return f"{HEADER_B}\n{records}\n\n{INSTRUCTION_B}\nRequested registry key: {query}\nAnswer:"


def build(tokenizer, retired):
    old = {k for row in retired for k in row["keys"]}
    whitelist = set(LEXICON) | set(EXTENSION)
    reserved = set(
        re.findall(
            "[a-z]+",
            (
                " ".join(FORMATS_A.values())
                + " ".join(FORMATS_B.values())
                + HEADER_A
                + HEADER_B
                + INSTRUCTION_A
                + INSTRUCTION_B
            ).lower(),
        )
    )
    words = sorted(
        w
        for w in whitelist - old - reserved
        if re.fullmatch("[a-z]{4,12}", w)
        and all(len(tokenizer.encode(x, add_special_tokens=False)) == 1 for x in (w, " " + w))
    )
    keys = words.copy()
    random.Random(20261024).shuffle(keys)
    keys = keys[:176]
    assert len(keys) == 176, len(words)
    rows = []
    cursor = 0
    for block, groups, track, colors in [
        ("A_calibration", 8, "A", ("gold", "silver")),
        ("A_primary", 16, "A", ("gold", "silver")),
        ("B_calibration", 8, "B", ("red", "blue")),
        ("B_primary", 16, "B", ("red", "blue")),
        ("B_vocabulary", 8, "B", ("green", "yellow")),
    ]:
        output_ids = [tokenizer.encode(" " + c, add_special_tokens=False) for c in colors]
        assert all(len(x) == 1 for x in output_ids)
        for g in range(groups):
            n = 2 if track == "A" else 4
            kk = keys[cursor : cursor + n]
            cursor += n
            fmt = (
                f"a{g % 2}" if track == "A" else f"{'d' if 'calibration' in block else 'h'}{g % 2}"
            )
            group = f"v8-{block}-{g:02}"
            cases = (
                itertools.product(range(2), range(2), range(2))
                if track == "A"
                else ((0, c, q) for c in range(2) for q in range(4))
            )
            pattern = (0, 1) if track == "A" else ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0))[g % 3]
            for order, assignment, q in cases:
                ordered = kk if order == 0 else kk[::-1]
                values = [colors[i ^ assignment] for i in pattern]
                query = kk[q]
                query_record = ordered.index(query)
                text = render(track, fmt, ordered, values, query)
                enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                ids = enc["input_ids"]
                offsets = enc["offset_mapping"]

                def pos(start, end):
                    matches = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
                    assert len(matches) == 1, (text, start, end, matches)
                    return matches[0]

                keypos = []
                valuepos = []
                start = len((HEADER_A if track == "A" else HEADER_B)) + 1
                for key, color, line in zip(ordered, values, text.split("\n")[1 : n + 1]):
                    template = (FORMATS_A if track == "A" else FORMATS_B)[fmt]
                    ks = start + len(template.split("{key}")[0].format(color=color))
                    vs = start + len(template.split("{color}")[0].format(key=key))
                    keypos.append(pos(ks, ks + len(key)))
                    valuepos.append(pos(vs, vs + len(color)))
                    start += len(line) + 1
                qs = text.rindex(query)
                qp = pos(qs, qs + len(query))
                assert len(ids) <= 128
                rows.append(
                    dict(
                        id=f"{group}-k{order}-a{assignment}-q{q}",
                        group=group,
                        block=block,
                        track=track,
                        format=fmt,
                        keys=kk,
                        ordered_keys=ordered,
                        values=values,
                        colors=list(colors),
                        query=query,
                        q_key=q,
                        query_record=query_record,
                        assignment=assignment,
                        k_order=order,
                        label=1 if values[query_record] == colors[0] else -1,
                        last_value=1 if values[-1] == colors[0] else -1,
                        token_ids=ids,
                        text=text,
                        key_positions=keypos,
                        color_positions=valuepos,
                        query_position=qp,
                        output_ids=[x[0] for x in output_ids],
                        opposite_id=f"{group}-k{order}-a{1 - assignment}-q{q}",
                    )
                )
    validate(rows)
    assert not set(keys) & old
    return dict(
        rows=rows,
        keys=keys,
        admissible_candidates=words,
        whitelist=sorted(whitelist),
        seed=20261024,
        target_forwards=0,
    )


def validate(rows):
    byid = {r["id"]: r for r in rows}
    assert len(rows) == len(byid) == 448
    keys = []
    for group in sorted({r["group"] for r in rows}):
        block = [r for r in rows if r["group"] == group]
        assert len(block) == 8
        keys += block[0]["keys"]
        assert Counter(r["query_record"] for r in block) == (
            {0: 4, 1: 4} if block[0]["track"] == "A" else {0: 2, 1: 2, 2: 2, 3: 2}
        )
    assert len(keys) == len(set(keys)) == 176
    for r in rows:
        assert (
            render(r["track"], r["format"], r["ordered_keys"], r["values"], r["query"]) == r["text"]
        )
        assert r["label"] == (1 if r["values"][r["query_record"]] == r["colors"][0] else -1)
        other = byid[r["opposite_id"]]
        assert other["label"] == -r["label"] and other["group"] == r["group"]
        assert Counter(r["token_ids"]) == Counter(other["token_ids"])
        assert (
            r["color_positions"] == other["color_positions"]
            and r["key_positions"] == other["key_positions"]
        )
        assert (
            len(r["token_ids"]) == len(other["token_ids"])
            and r["token_ids"][-8:] == other["token_ids"][-8:]
        )
        assert r["values"].count(r["colors"][0]) == len(r["values"]) // 2


def margin(logits):
    z = np.asarray(logits, dtype=np.float64)
    return z[:, 0] - z[:, 1]


def group_p(rows, changes, mask=None):
    mask = np.ones(len(rows), bool) if mask is None else np.asarray(mask, bool)
    means = [
        np.mean([v for r, v, m in zip(rows, changes, mask) if r["group"] == g and m])
        for g in sorted({r["group"] for r, m in zip(rows, mask) if m})
    ]
    n = len(means)
    k = sum(x > 0 for x in means)
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n if n else 1.0


def stats(rows, clean, repaired):
    y = np.array([r["label"] for r in rows])
    a = y * margin(clean)
    b = y * margin(repaired)
    gain = b - a
    before = a > 0
    after = b > 0
    rec = np.array(
        [
            r["track"] == "B" and r["query_record"] != 3 and r["label"] != r["last_value"]
            for r in rows
        ]
    )
    errors = rec & (a < 0)
    return dict(
        accuracy=float(after.mean()),
        baseline_accuracy=float(before.mean()),
        fixed=int((~before & after).sum()),
        broken=int((before & ~after).sum()),
        baseline_errors=int((~before).sum()),
        median_gain=float(np.median(gain)),
        group_sign_p=group_p(rows, gain),
        minority_accuracy=float(after[y == -1].mean()),
        baseline_minority_accuracy=float(before[y == -1].mean()),
        recency_eligible=int(rec.sum()),
        recency_errors=int((rec & (b < 0)).sum()),
        baseline_recency_errors=int(errors.sum()),
        recency_gain_median=float(np.median(gain[errors])) if errors.any() else None,
        recency_sign_p=group_p(rows, gain, errors),
        recency_error_groups=len({r["group"] for r, m in zip(rows, errors) if m}),
    )


def grade_A(rows, clean, repaired, probe):
    s = stats(rows, clean, repaired)
    y = np.array([r["label"] for r in rows])
    correct = y * margin(clean) > 0
    pc = y * np.asarray(probe) > 0
    probe_accuracy = float(pc.mean())
    error_accuracy = float(pc[~correct].mean()) if (~correct).any() else None
    checks = dict(
        probe_overall=probe_accuracy >= 0.92,
        probe_exceeds_baseline=probe_accuracy > s["baseline_accuracy"],
        probe_on_errors=error_accuracy is not None and error_accuracy >= 0.75,
        repair_accuracy=s["accuracy"] >= 0.90,
        net_rescue=s["fixed"] > s["broken"],
        fixes_half=s["baseline_errors"] < 4 or s["fixed"] / s["baseline_errors"] >= 0.5,
        margin_gain=s["median_gain"] >= 0.20,
        group_sign=s["group_sign_p"] <= 0.01,
        silver_non_decrease=s["minority_accuracy"] >= s["baseline_minority_accuracy"],
    )
    return dict(
        stats=s,
        probe_accuracy=probe_accuracy,
        probe_error_accuracy=error_accuracy,
        criteria=checks,
        A1=all(checks[k] for k in ("probe_overall", "probe_exceeds_baseline", "probe_on_errors")),
        A2=all(v for k, v in checks.items() if not k.startswith("probe_")),
    )


def grade_B(rows, clean, repaired, naive):
    s = stats(rows, clean, repaired)
    c = stats(rows, clean, naive)
    checks = dict(
        net_rescue=s["fixed"] > s["broken"],
        recency_reduced=s["recency_errors"] < s["baseline_recency_errors"],
        beats_naive_net=s["fixed"] - s["broken"] > c["fixed"] - c["broken"],
        fewer_regressions=s["broken"] < c["broken"],
        error_margin_positive=s["recency_gain_median"] is not None and s["recency_gain_median"] > 0,
        error_sign=s["recency_sign_p"] <= 0.01,
    )
    return dict(stats=s, naive=c, criteria=checks, passed=all(checks.values()))


def select(rows, clean, candidates):
    scored = [stats(rows, clean, x) for x in candidates]
    # Final tie is the first tuple in the registered grid.
    index = max(
        range(len(scored)),
        key=lambda i: (
            scored[i]["fixed"] - scored[i]["broken"],
            -scored[i]["broken"],
            scored[i]["median_gain"],
            -i,
        ),
    )
    return index, scored


def centroid(h, y):
    x = np.asarray(h, dtype=np.float64).reshape(len(h), -1)
    y = np.asarray(y)
    pos = x[y == 1].mean(0)
    neg = x[y == -1].mean(0)
    return pos - neg, float(-0.5 * ((pos * pos).sum() - (neg * neg).sum()))


def fit_probe(rows, h):
    y = np.array([r["label"] for r in rows])
    x = np.asarray(h, dtype=np.float64).reshape(len(h), -1)
    predictions = np.empty(len(rows))
    for group in sorted({r["group"] for r in rows}):
        test = np.array([r["group"] == group for r in rows])
        w, b = centroid(x[~test], y[~test])
        predictions[test] = x[test] @ w + b
    w, b = centroid(x, y)
    return w, b, predictions


def output_center(logits, mean, gamma):
    return np.asarray(logits, dtype=np.float64) - gamma * np.asarray(mean, dtype=np.float64)
