"""Prospective single-line lookup factorial; tokenizer-only neutral identifier draw."""

import itertools
import random
import re
from collections import Counter

from src.poc.v5_templates import HEADER, INSTRUCTION, LEXICON

FORMATS = {
    "calibration": "Record 1: code {k1} has color {c1}. Record 2: code {k2} has color {c2}. Question: What color is code {q}? Answer:",
    "h0": "Entry 1 states code {k1} is color {c1}. Entry 2 states code {k2} is color {c2}. Query: Which color is code {q}? Answer:",
    "h1": "First pair: code {k1} equals color {c1}. Second pair: code {k2} equals color {c2}. Task: State the color for code {q}. Answer:",
}


def match_prompt(text):
    if "\n" in text or "\r" in text:
        raise ValueError("Prompt must be a single line")
    matches = []
    for fmt, template in FORMATS.items():
        pattern = re.escape(template)
        for key in ("k1", "k2", "q", "c1", "c2"):
            values = "red|blue|green|yellow" if key.startswith("c") else "[a-z]+"
            pattern = pattern.replace(re.escape("{" + key + "}"), f"(?P<{key}>{values})")
        match = re.fullmatch(pattern, text)
        if match is not None:
            matches.append((fmt, match))
    if len(matches) != 1:
        raise ValueError("Unregistered or ambiguous inline grammar")
    return matches[0]


def parse(text):
    fmt, m = match_prompt(text)
    records = [
        dict(
            key=m[f"k{i}"],
            color=m[f"c{i}"],
            key_span=list(m.span(f"k{i}")),
            color_span=list(m.span(f"c{i}")),
        )
        for i in (1, 2)
    ]
    if len({r["key"] for r in records}) != 2 or {r["color"] for r in records} not in (
        {"red", "blue"},
        {"green", "yellow"},
    ):
        raise ValueError("Ambiguous bindings")
    table = {r["key"]: r["color"] for r in records}
    if m["q"] not in table:
        raise ValueError("Unknown query key")
    return table[m["q"]], records, fmt, m["q"]


def validate(rows):
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(rows)
    groups = {r["group"] for r in rows}
    for group in groups:
        members = [r for r in rows if r["group"] == group]
        assert len(members) == 8
        assert {(r["k_order"], r["c_order"], r["q_key"]) for r in members} == set(
            itertools.product(range(2), repeat=3)
        )
    for r in rows:
        answer, records, fmt, query = parse(r["text"])
        assert answer == r["answer_color"] and fmt == r["format"]
        assert query == r["keys"][r["q_key"]]
        assert [p["key"] for p in records] == (r["keys"] if r["k_order"] == 0 else r["keys"][::-1])
        assert r["target_position"] == r["color_positions"][r["q_key"] ^ r["k_order"]]
        assert r["label"] == (1 if r["q_key"] ^ r["k_order"] ^ r["c_order"] == 0 else -1)
        for axis in ("A", "B", "C"):
            d = by_id[r["donors"][axis]]
            assert d["donors"][axis] == r["id"] and d["group"] == r["group"]
            assert d["label"] == -r["label"]
            assert (
                len(d["token_ids"]) == len(r["token_ids"])
                and d["color_positions"] == r["color_positions"]
            )
            changes = {i for i, (a, b) in enumerate(zip(r["token_ids"], d["token_ids"])) if a != b}
            expected = set(
                r["color_positions"]
                if axis == "A"
                else r["key_positions"]
                if axis == "C"
                else [r["query_position"]]
            )
            assert changes == expected
            if axis in ("A", "C"):
                assert r["token_ids"][-8:] == d["token_ids"][-8:]
            if axis != "B":
                assert Counter(r["token_ids"]) == Counter(d["token_ids"])
            else:
                assert r["token_ids"][: r["context_end"]] == d["token_ids"][: r["context_end"]]
                assert r["k_order"] == d["k_order"] and r["c_order"] == d["c_order"]


def build(tokenizer, retired):
    retired_keys = {k for r in retired for k in r["keys"]}
    # Deterministic lexical/tokenizer admissibility only. No response-based filtering.
    excluded = (
        set(re.findall("[a-z]+", (HEADER + " " + INSTRUCTION).lower()))
        | {"red", "blue", "green", "yellow"}
        | retired_keys
    )
    words = set()
    for token_id in tokenizer.get_vocab().values():
        text = tokenizer.decode([token_id])
        if re.fullmatch(r" [a-z]{4,8}", text) and text[1:] not in excluded and text[1:] in LEXICON:
            if tokenizer.encode(text, add_special_tokens=False) == [token_id]:
                words.add(text[1:])
    candidates = sorted(words)
    random.Random(20261022).shuffle(candidates)
    keys = candidates[:64]
    assert len(keys) == 64 and len(set(keys)) == 64 and not set(keys) & retired_keys
    rows = []
    cursor = 0
    output_ids = {}
    for split, groups, colors in [
        ("calibration", 8, ("red", "blue")),
        ("primary", 16, ("red", "blue")),
        ("vocabulary", 8, ("green", "yellow")),
    ]:
        output_ids[split] = [tokenizer.encode(" " + v, add_special_tokens=False)[0] for v in colors]
        assert all(len(tokenizer.encode(" " + v, add_special_tokens=False)) == 1 for v in colors)
        for g in range(groups):
            pair = keys[cursor : cursor + 2]
            cursor += 2
            fmt = "calibration" if split == "calibration" else f"h{g % 2}"
            group = f"v6-{split}-{g:02}"

            def uid(k, c, q):
                return f"{group}-k{k}-c{c}-q{q}"

            for k, c, q in itertools.product(range(2), repeat=3):
                order = pair if k == 0 else pair[::-1]
                values = colors if c == 0 else colors[::-1]
                text = FORMATS[fmt].format(
                    k1=order[0], c1=values[0], k2=order[1], c2=values[1], q=pair[q]
                )
                answer, parsed, _, _ = parse(text)
                enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                ids, offsets = enc["input_ids"], enc["offset_mapping"]

                def position(span):
                    found = [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0]]
                    assert len(found) == 1, (text, span, found)
                    return found[0]

                color_pos = [position(r["color_span"]) for r in parsed]
                key_pos = [position(r["key_span"]) for r in parsed]
                qp = position(match_prompt(text)[1].span("q"))
                assert 8 < len(ids) <= 128
                assert [ids[p] for p in key_pos] == [
                    tokenizer.encode(" " + a, add_special_tokens=False)[0] for a in order
                ]
                assert ids[qp] == tokenizer.encode(" " + pair[q], add_special_tokens=False)[0]
                assert sorted(ids[p] for p in color_pos) == sorted(output_ids[split])
                rows.append(
                    dict(
                        id=uid(k, c, q),
                        group=group,
                        split=split,
                        format=fmt,
                        text=text,
                        keys=pair,
                        k_order=k,
                        c_order=c,
                        q_key=q,
                        label=1 if answer == colors[0] else -1,
                        answer_color=answer,
                        token_ids=ids,
                        color_positions=color_pos,
                        key_positions=key_pos,
                        query_position=qp,
                        context_end=qp,
                        target_position=color_pos[q ^ k],
                        donors=dict(A=uid(k, 1 - c, q), B=uid(k, c, 1 - q), C=uid(1 - k, c, q)),
                    )
                )
    validate(rows)
    assert not {r["text"] for r in rows} & {r["text"] for r in retired}
    return dict(
        rows=rows,
        output_ids=output_ids,
        keys=keys,
        seed=20261022,
        admissible_candidates=sorted(words),
        lexical_policy="Closed neutral word whitelist; single-token filter; exclude all retired keys; fixed seed shuffle",
        approved_axis_B_invariant="One query-token substitution; identical context, length and value positions",
        target_forwards=0,
    )
