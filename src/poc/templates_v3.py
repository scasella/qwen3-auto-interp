"""Fresh four-record lookup tasks, constructed without target-model responses."""

import re
from collections import Counter

V3_FORMATS = {
    "d0": "Entry {key} stores {color}.",
    "d1": "Stored value for {key}: {color}.",
    "h0": "Binding for {key} contains value {color}.",
    "h1": "The registry stores {color} under key {key}.",
}
V3_BASE_PATTERNS = ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0))
_COLORS = "red|blue|green|yellow"
_KEY = "[A-Za-z]+[0-9]+"
_PATTERNS = {
    "d0": re.compile(rf"Entry (?P<key>{_KEY}) stores (?P<color>{_COLORS})\."),
    "d1": re.compile(rf"Stored value for (?P<key>{_KEY}): (?P<color>{_COLORS})\."),
    "h0": re.compile(rf"Binding for (?P<key>{_KEY}) contains value (?P<color>{_COLORS})\."),
    "h1": re.compile(rf"The registry stores (?P<color>{_COLORS}) under key (?P<key>{_KEY})\."),
}
_HEADER = "The registry contains four independent key-to-color bindings."
_INSTRUCTION = "Find the requested key among all four records and return only its stored color."
_QUERY = re.compile(rf"Requested registry key: (?P<key>{_KEY})")


def _parse(text):
    lines = text.split("\n")
    if (
        len(lines) != 9
        or lines[0] != _HEADER
        or lines[5] != ""
        or lines[6] != _INSTRUCTION
        or lines[8] != "Answer:"
    ):
        raise ValueError("Not a valid v3 four-record lookup grammar")
    records, formats = [], []
    offset = len(lines[0]) + 1
    for line in lines[1:5]:
        matches = [(fmt, pattern.fullmatch(line)) for fmt, pattern in _PATTERNS.items()]
        matches = [(fmt, match) for fmt, match in matches if match is not None]
        if len(matches) != 1:
            raise ValueError("Malformed v3 binding record")
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
    query = _QUERY.fullmatch(lines[7])
    if query is None or len({record["key"] for record in records}) != 4:
        raise ValueError("Missing query or duplicate v3 keys")
    if len(set(formats)) != 1:
        raise ValueError("Mixed v3 binding grammars")
    values = Counter(record["color"] for record in records)
    if values not in (Counter(red=2, blue=2), Counter(green=2, yellow=2)):
        raise ValueError("Values must use a registered balanced four-record vocabulary")
    lookup = {record["key"]: record["color"] for record in records}
    if query.group("key") not in lookup:
        raise ValueError("Requested registry key is absent")
    return lookup[query.group("key")], records, formats[0]


def parse_v3_lookup(text):
    """Execute the complete grammar, rejecting ambiguous or malformed bindings."""
    return _parse(text)[0]


def _positions(offsets, span):
    start, end = span
    positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not positions:
        raise ValueError("Tokenizer offsets do not cover a source span")
    return positions


def build_v3_dataset(tokenizer, retired_rows):
    """Freeze all cases; never filter or alter examples using model behavior."""
    output_ids = {}
    for block, colors in (("primary", ("red", "blue")), ("vocabulary", ("green", "yellow"))):
        encoded = [tokenizer.encode(" " + color, add_special_tokens=False) for color in colors]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("Each registered answer color must be one leading-space token")
        output_ids[block] = [int(ids[0]) for ids in encoded]
    rows = []
    for split, count, prefixes, formats, colors, outputs in (
        (
            "discovery",
            8,
            ("amberD", "berylD", "cobaltD", "dahliaD"),
            ("d0", "d1"),
            ("red", "blue"),
            output_ids["primary"],
        ),
        (
            "primary",
            16,
            ("elmE", "flintE", "garnetE", "hazelE"),
            ("h0", "h1"),
            ("red", "blue"),
            output_ids["primary"],
        ),
        (
            "vocabulary",
            8,
            ("irisF", "jadeF", "kelpF", "larchF"),
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
            group = f"v3-{split}-{group_index:02}"
            base_pattern_index = group_index % len(V3_BASE_PATTERNS)
            base_pattern = V3_BASE_PATTERNS[base_pattern_index]
            for query in range(4):
                for assignment in range(2):
                    values = tuple(colors[(value + assignment) % 2] for value in base_pattern)
                    bindings = "\n".join(
                        V3_FORMATS[fmt].format(key=key, color=value)
                        for key, value in zip(keys, values, strict=True)
                    )
                    text = (
                        f"{_HEADER}\n{bindings}\n\n{_INSTRUCTION}\n"
                        f"Requested registry key: {keys[query]}\nAnswer:"
                    )
                    answer, parsed, parsed_format = _parse(text)
                    if answer != values[query] or parsed_format != fmt:
                        raise AssertionError("Executable parser disagrees with generated binding")
                    tokenized = tokenizer(
                        text, add_special_tokens=False, return_offsets_mapping=True
                    )
                    ids = list(map(int, tokenized["input_ids"]))
                    offsets = [tuple(map(int, pair)) for pair in tokenized["offset_mapping"]]
                    if len(ids) != len(offsets) or not 8 < len(ids) <= 128:
                        raise ValueError("Invalid v3 tokenizer offsets or sequence length")
                    color_positions = []
                    for record in parsed:
                        positions = _positions(offsets, record["color_span"])
                        expected = outputs[colors.index(record["color"])]
                        if len(positions) != 1 or ids[positions[0]] != expected:
                            raise ValueError(
                                "Bound color must occupy its registered single-token source"
                            )
                        color_positions.append(positions[0])
                    rows.append(
                        {
                            "id": f"{group}-q{query}-a{assignment}",
                            "group": group,
                            "split": split,
                            "format": fmt,
                            "query_record": query,
                            "assignment": assignment,
                            "base_pattern": list(base_pattern),
                            "base_pattern_index": base_pattern_index,
                            "keys": list(keys),
                            "text": text,
                            "token_ids": ids,
                            "label": 1 if answer == colors[0] else -1,
                            "answer_color": answer,
                            "first_value": 1 if values[0] == colors[0] else -1,
                            "last_value": 1 if values[-1] == colors[0] else -1,
                            "opposite_id": f"{group}-q{query}-a{1 - assignment}",
                            "color_positions": color_positions,
                            "color_spans": [list(record["color_span"]) for record in parsed],
                            "key_positions": [
                                _positions(offsets, record["key_span"]) for record in parsed
                            ],
                            "delimiter_positions": [
                                p
                                for record in parsed
                                for p in _positions(offsets, record["delimiter_span"])
                            ],
                        }
                    )
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len({row["text"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate v3 IDs or prompt text")
    for row in rows:
        other = by_id[row["opposite_id"]]
        if not (
            row["label"] == -other["label"]
            and row["keys"] == other["keys"]
            and row["query_record"] == other["query_record"]
            and row["color_positions"] == other["color_positions"]
            and len(row["token_ids"]) == len(other["token_ids"])
            and row["token_ids"][-8:] == other["token_ids"][-8:]
            and Counter(row["token_ids"]) == Counter(other["token_ids"])
        ):
            raise AssertionError("Four-record counterfactual invariance failed")
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
        raise ValueError("Retired v1/v2 keys cannot appear in v3")
    if any(
        split_keys[a] & split_keys[b]
        for a, b in (
            ("discovery", "primary"),
            ("discovery", "vocabulary"),
            ("primary", "vocabulary"),
        )
    ):
        raise AssertionError("V3 split keys overlap")
    if {row["text"] for row in rows} & {row["text"] for row in retired_rows}:
        raise ValueError("Retired v1/v2 text cannot appear in v3")
    example = rows[0]
    invalid = (
        example["text"] + " red",
        example["text"].replace(example["keys"][1], example["keys"][0]),
        example["text"].replace(
            "Requested registry key: " + example["keys"][0], "Requested registry key: absentZ999"
        ),
        example["text"].replace(" stores ", " might store ", 1),
        example["text"].replace(" stores blue.", " stores red.", 1),
    )
    rejected = 0
    for text in invalid:
        try:
            parse_v3_lookup(text)
        except ValueError:
            rejected += 1
    if rejected != len(invalid):
        raise AssertionError("V3 executable parser negative controls failed")
    return {
        "rows": rows,
        "output_ids": output_ids,
        "sanity": {
            "passed": True,
            "parsed_rows": len(rows),
            "malformed_rejected": rejected,
            "records_per_prompt": 4,
            "queries_per_group": 4,
            "assignments_per_query": 2,
            "balanced_base_patterns": [list(pattern) for pattern in V3_BASE_PATTERNS],
            "counterfactual_invariance": True,
            "source_positions_from_offsets": True,
            "keys_disjoint_from_retired_v1_v2": True,
            "keys_disjoint_across_splits": True,
            "new_discovery_and_heldout_grammars": V3_FORMATS,
            "max_sequence_length": max(len(row["token_ids"]) for row in rows),
            "target_model_calls": 0,
        },
    }


def baseline_scores_v3(rows):
    """Counterfactual invariance rules out token-bag, suffix and length classifiers."""
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        other = by_id[row["opposite_id"]]
        if not (
            Counter(row["token_ids"]) == Counter(other["token_ids"])
            and row["token_ids"][-8:] == other["token_ids"][-8:]
            and len(row["token_ids"]) == len(other["token_ids"])
            and row["label"] == -other["label"]
        ):
            raise ValueError("Invalid baseline counterfactual pair")
    count = len(rows)
    return {
        "constant": max(sum(row["label"] == label for row in rows) for label in (-1, 1)) / count,
        "last_token_identity_upper_bound": 0.5,
        "token_count_vector_upper_bound": 0.5,
        "position_length_upper_bound": 0.5,
        "last_eight_tokens_upper_bound": 0.5,
        "first_mentioned_value": sum(row["first_value"] == row["label"] for row in rows) / count,
        "last_mentioned_value": sum(row["last_value"] == row["label"] for row in rows) / count,
        "executable_lookup_positive_control": sum(
            parse_v3_lookup(row["text"]) == row["answer_color"] for row in rows
        )
        / count,
    }
