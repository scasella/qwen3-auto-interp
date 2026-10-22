"""Executable lookup counterfactuals; no target-model calls or generated prose."""

import re
from collections import Counter

FORMATS = (
    "{a} = {va}\n{b} = {vb}",
    "{a}: {va}\n{b}: {vb}",
    "{a} -> {va}\n{b} -> {vb}",
    "({a}, {va})\n({b}, {vb})",
)
SPACERS = (
    "Use the assignments above to find the requested value. Give only the color.",
    "Look up the requested key in the table above. Answer with its color only.",
    "Read the table carefully and return only the value attached to the requested key.",
    "Choose the color assigned to the requested key, using the records above.",
)


def parse_lookup(text):
    pairs = re.findall(r"([A-Za-z]+\d+)\s*(?:=|:|->|,)\s*(red|blue)", text)
    query = re.search(r"Requested key: ([A-Za-z]+\d+)\nValue:$", text)
    if len(pairs) != 2 or query is None or len(dict(pairs)) != 2:
        raise ValueError("Not a valid two-record lookup template")
    return dict(pairs)[query.group(1)]


def build_dataset(tokenizer):
    data = []
    for split, count, offset in (("discovery", 16, 0), ("confirmation", 32, 2)):
        for g in range(count):
            keys = (
                (f"atlas{g:02}", f"birch{g:02}")
                if split == "discovery"
                else (f"cedar{g:02}", f"delta{g:02}")
            )
            fmt = offset + g % 2
            # Deterministically vary record order, independently of query and assignment.
            if (g // 2) % 2:
                keys = keys[::-1]
            group = f"{split}-{g:02}"
            for query in range(2):
                for assignment in range(2):
                    colors = ("red", "blue") if assignment == 0 else ("blue", "red")
                    records = FORMATS[fmt].format(a=keys[0], b=keys[1], va=colors[0], vb=colors[1])
                    text = (
                        "Each key has one color.\n"
                        + records
                        + "\n\n"
                        + SPACERS[(g // 4) % len(SPACERS)]
                        + f"\nRequested key: {keys[query]}\nValue:"
                    )
                    answer = parse_lookup(text)
                    assert answer == colors[query]
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    assert 8 < len(ids) <= 128
                    data.append(
                        {
                            "id": f"{group}-q{query}-a{assignment}",
                            "group": group,
                            "split": split,
                            "format": fmt,
                            "query_record": query,
                            "assignment": assignment,
                            "keys": list(keys),
                            "text": text,
                            "token_ids": ids,
                            "label": 1 if answer == "red" else -1,
                            "first_value": 1 if colors[0] == "red" else -1,
                            "last_value": 1 if colors[1] == "red" else -1,
                            "opposite_id": f"{group}-q{query}-a{1 - assignment}",
                        }
                    )
    by_id = {r["id"]: r for r in data}
    for row in data:
        other = by_id[row["opposite_id"]]
        assert row["label"] == -other["label"]
        assert len(row["token_ids"]) == len(other["token_ids"])
        assert Counter(row["token_ids"]) == Counter(other["token_ids"])
        assert row["token_ids"][-8:] == other["token_ids"][-8:]
        candidates = [
            r
            for r in data
            if r["split"] == row["split"]
            and r["format"] == row["format"]
            and r["group"] != row["group"]
            and r["label"] == row["label"]
            and r["query_record"] == row["query_record"]
        ]
        row["same_answer_id"] = min(candidates, key=lambda r: r["id"])["id"]
    assert len({r["text"] for r in data}) == len(data)
    train_keys = {k for r in data if r["split"] == "discovery" for k in r["keys"]}
    test_keys = {k for r in data if r["split"] == "confirmation" for k in r["keys"]}
    assert not train_keys & test_keys
    return data


def baseline_scores(rows):
    """Invariance gives an exact upper bound for any classifier using these features."""
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        other = by_id[r["opposite_id"]]
        assert Counter(r["token_ids"]) == Counter(other["token_ids"])
        assert r["token_ids"][-8:] == other["token_ids"][-8:]
        assert len(r["token_ids"]) == len(other["token_ids"])
        assert r["label"] == -other["label"]
    n = len(rows)
    return {
        "constant": max(sum(r["label"] == y for r in rows) for y in (-1, 1)) / n,
        "last_token_identity_upper_bound": 0.5,
        "token_count_vector_upper_bound": 0.5,
        "position_length_upper_bound": 0.5,
        "last_eight_tokens_upper_bound": 0.5,
        "first_mentioned_value": sum(r["first_value"] == r["label"] for r in rows) / n,
        "last_mentioned_value": sum(r["last_value"] == r["label"] for r in rows) / n,
        "executable_lookup_positive_control": sum(
            (1 if parse_lookup(r["text"]) == "red" else -1) == r["label"] for r in rows
        )
        / n,
    }


# Version 2 is deliberately separate: the v1 grammars and registered examples remain intact.
V2_FORMATS = {
    "d0": "Key {a} has color {va}.",
    "d1": "{a} is assigned the color {va}.",
    "h0": "The color associated with identifier {a} is {va}.",
    "h1": "For identifier {a}, the assigned color is {va}.",
}
_V2_RECORD_PATTERNS = {
    "d0": re.compile(r"Key (?P<key>[A-Za-z]+[0-9]+) has color (?P<color>red|blue|green|yellow)\."),
    "d1": re.compile(
        r"(?P<key>[A-Za-z]+[0-9]+) is assigned the color (?P<color>red|blue|green|yellow)\."
    ),
    "h0": re.compile(
        r"The color associated with identifier (?P<key>[A-Za-z]+[0-9]+) is (?P<color>red|blue|green|yellow)\."
    ),
    "h1": re.compile(
        r"For identifier (?P<key>[A-Za-z]+[0-9]+), the assigned color is (?P<color>red|blue|green|yellow)\."
    ),
}
_V2_HEADER = "Each identifier has exactly one assigned color."
_V2_INSTRUCTION = (
    "Use the assignments above. Return only the color assigned to the requested identifier."
)
_V2_QUERY = re.compile(r"Requested identifier: (?P<key>[A-Za-z]+[0-9]+)")


def _v2_parse(text):
    lines = text.split("\n")
    if (
        len(lines) != 7
        or lines[0] != _V2_HEADER
        or lines[3] != ""
        or lines[4] != _V2_INSTRUCTION
        or lines[6] != "Answer:"
    ):
        raise ValueError("Not a valid v2 two-record lookup grammar")
    records = []
    offset = len(lines[0]) + 1
    formats = []
    for line in lines[1:3]:
        matches = [(fmt, pattern.fullmatch(line)) for fmt, pattern in _V2_RECORD_PATTERNS.items()]
        matches = [(fmt, match) for fmt, match in matches if match is not None]
        if len(matches) != 1:
            raise ValueError("Malformed v2 record")
        fmt, match = matches[0]
        formats.append(fmt)
        records.append(
            {
                "key": match.group("key"),
                "color": match.group("color"),
                "key_span": tuple(offset + p for p in match.span("key")),
                "color_span": tuple(offset + p for p in match.span("color")),
                "delimiter_span": (offset + len(line) - 1, offset + len(line)),
            }
        )
        offset += len(line) + 1
    query = _V2_QUERY.fullmatch(lines[5])
    if query is None or len({r["key"] for r in records}) != 2 or len(set(formats)) != 1:
        raise ValueError("Missing query, duplicate keys, or mixed v2 grammars")
    if {r["color"] for r in records} not in ({"red", "blue"}, {"green", "yellow"}):
        raise ValueError("Values must cross exactly one registered color vocabulary")
    lookup = {r["key"]: r["color"] for r in records}
    if query.group("key") not in lookup:
        raise ValueError("Requested identifier is absent")
    return lookup[query.group("key")], records, formats[0]


def parse_v2_lookup(text):
    """Parse the actual v2 text, rejecting malformed, duplicate, or missing bindings."""
    return _v2_parse(text)[0]


def _v2_token_positions(offsets, span):
    start, end = span
    positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not positions:
        raise ValueError("Tokenizer offsets do not cover a source span")
    return positions


def build_v2_dataset(tokenizer, retired_rows):
    """Build all new examples without target-model execution or response rejection."""
    output_ids = {}
    for block, colors in (("primary", ("red", "blue")), ("vocabulary", ("green", "yellow"))):
        encoded = [tokenizer.encode(" " + color, add_special_tokens=False) for color in colors]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("Each registered answer color must be one token with a leading space")
        output_ids[block] = [int(ids[0]) for ids in encoded]
    rows = []
    for split, count, prefixes, formats, colors, outputs in (
        (
            "discovery",
            16,
            ("sequoiaA", "tundraA"),
            ("d0", "d1"),
            ("red", "blue"),
            output_ids["primary"],
        ),
        (
            "primary",
            32,
            ("willowB", "topazB"),
            ("h0", "h1"),
            ("red", "blue"),
            output_ids["primary"],
        ),
        (
            "vocabulary",
            16,
            ("quartzC", "umberC"),
            ("h0", "h1"),
            ("green", "yellow"),
            output_ids["vocabulary"],
        ),
    ):
        for group_index in range(count):
            keys = tuple(f"{prefix}{group_index:03}" for prefix in prefixes)
            if (group_index // 2) % 2:
                keys = keys[::-1]
            fmt = formats[group_index % 2]
            group = f"v2-{split}-{group_index:02}"
            for query in range(2):
                for assignment in range(2):
                    values = colors if assignment == 0 else colors[::-1]
                    records = "\n".join(
                        V2_FORMATS[fmt].format(a=key, va=value)
                        for key, value in zip(keys, values, strict=True)
                    )
                    text = (
                        f"{_V2_HEADER}\n{records}\n\n{_V2_INSTRUCTION}\n"
                        f"Requested identifier: {keys[query]}\nAnswer:"
                    )
                    answer, parsed, parsed_format = _v2_parse(text)
                    if answer != values[query] or parsed_format != fmt:
                        raise AssertionError("Text parser disagrees with generated binding")
                    tokenized = tokenizer(
                        text, add_special_tokens=False, return_offsets_mapping=True
                    )
                    ids = list(map(int, tokenized["input_ids"]))
                    offsets = [tuple(map(int, pair)) for pair in tokenized["offset_mapping"]]
                    if len(ids) != len(offsets) or not 8 < len(ids) <= 128:
                        raise ValueError("Invalid v2 tokenizer offsets or sequence length")
                    color_positions = []
                    for record in parsed:
                        positions = _v2_token_positions(offsets, record["color_span"])
                        expected = outputs[colors.index(record["color"])]
                        if len(positions) != 1 or ids[positions[0]] != expected:
                            raise ValueError(
                                "Bound value must occupy its registered single-token source"
                            )
                        color_positions.append(positions[0])
                    label = 1 if answer == colors[0] else -1
                    rows.append(
                        {
                            "id": f"{group}-q{query}-a{assignment}",
                            "group": group,
                            "split": split,
                            "format": fmt,
                            "query_record": query,
                            "assignment": assignment,
                            "keys": list(keys),
                            "text": text,
                            "token_ids": ids,
                            "label": label,
                            "answer_color": answer,
                            "first_value": 1 if values[0] == colors[0] else -1,
                            "last_value": 1 if values[1] == colors[0] else -1,
                            "opposite_id": f"{group}-q{query}-a{1 - assignment}",
                            "color_positions": color_positions,
                            "color_spans": [list(record["color_span"]) for record in parsed],
                            "key_positions": [
                                _v2_token_positions(offsets, record["key_span"])
                                for record in parsed
                            ],
                            "delimiter_positions": [
                                p
                                for record in parsed
                                for p in _v2_token_positions(offsets, record["delimiter_span"])
                            ],
                        }
                    )
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        other = by_id[row["opposite_id"]]
        if not (
            row["label"] == -other["label"]
            and row["query_record"] == other["query_record"]
            and row["color_positions"] == other["color_positions"]
            and len(row["token_ids"]) == len(other["token_ids"])
            and row["token_ids"][-8:] == other["token_ids"][-8:]
            and Counter(row["token_ids"]) == Counter(other["token_ids"])
        ):
            raise AssertionError("Counterfactual invariance failed")
        candidates = [
            candidate
            for candidate in rows
            if candidate["split"] == row["split"]
            and candidate["format"] == row["format"]
            and candidate["group"] != row["group"]
            and candidate["label"] == row["label"]
            and candidate["query_record"] == row["query_record"]
        ]
        row["same_answer_id"] = min(candidates, key=lambda candidate: candidate["id"])["id"]
    split_keys = {
        split: {key for row in rows if row["split"] == split for key in row["keys"]}
        for split in ("discovery", "primary", "vocabulary")
    }
    retired_keys = {key for row in retired_rows for key in row["keys"]}
    if any(keys & retired_keys for keys in split_keys.values()):
        raise ValueError("Retired v1 keys cannot appear in v2")
    if any(
        split_keys[a] & split_keys[b]
        for a, b in (
            ("discovery", "primary"),
            ("discovery", "vocabulary"),
            ("primary", "vocabulary"),
        )
    ):
        raise AssertionError("Split keys overlap")
    if {row["text"] for row in rows} & {row["text"] for row in retired_rows}:
        raise ValueError("Retired v1 text cannot appear in v2")
    malformed_rejected = 0
    for text in (
        rows[0]["text"].replace("Answer:", "Answer: red"),
        rows[0]["text"].replace(rows[0]["keys"][1], rows[0]["keys"][0]),
        rows[0]["text"].replace(
            "Requested identifier: " + rows[0]["keys"][0], "Requested identifier: absentZ999"
        ),
        rows[0]["text"].replace(" has color ", " might have color ", 1),
    ):
        try:
            parse_v2_lookup(text)
        except ValueError:
            malformed_rejected += 1
    if malformed_rejected != 4:
        raise AssertionError("Parser sanity negative controls failed")
    return {
        "rows": rows,
        "output_ids": output_ids,
        "sanity": {
            "passed": True,
            "parsed_rows": len(rows),
            "malformed_rejected": malformed_rejected,
            "counterfactual_invariance": True,
            "source_positions_from_offsets": True,
            "keys_disjoint_from_retired_v1": True,
            "keys_disjoint_across_splits": True,
            "new_discovery_and_heldout_grammars": V2_FORMATS,
            "target_model_calls": 0,
        },
    }
