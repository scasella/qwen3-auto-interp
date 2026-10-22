"""New vocabulary and formats, generated without any target-model inference."""

import re
from collections import Counter

FORMATS = (
    "{a} has value {va}.\n{b} has value {vb}.",
    "{a} | {va}\n{b} | {vb}",
    "entry {a} stores {va};\nentry {b} stores {vb};",
    "[{a} => {va}]\n[{b} => {vb}]",
)
SPACERS = (
    "Find the value assigned to the requested key. Return only that value.",
    "Use the two records to answer the request. Give the value alone.",
    "Match the requested key to its record above. Answer with its assigned value.",
    "Retrieve the requested entry from the records above. Return its value only.",
)


def parse(text):
    pairs = re.findall(r"([a-z]+\d+) (?:has value |\| |stores |=> )(gold|silver)", text)
    query = re.search(r"Requested key: ([a-z]+\d+)\nValue:$", text)
    assert len(pairs) == 2 and len(dict(pairs)) == 2 and query
    return dict(pairs)[query[1]]


def build(tokenizer):
    outputs = [tokenizer.encode(" " + v, add_special_tokens=False) for v in ("gold", "silver")]
    assert all(len(v) == 1 for v in outputs)
    outputs = [v[0] for v in outputs]
    rows = []
    for split, count, offset, stems in (
        ("calibration", 16, 0, ("maple", "oak")),
        ("confirmation", 32, 2, ("willow", "hazel")),
    ):
        for g in range(count):
            keys = [f"{stem}{g:02}" for stem in stems]
            if (g // 2) % 2:
                keys.reverse()
            fmt, group = offset + g % 2, f"{split}-{g:02}"
            for query in range(2):
                for assignment in range(2):
                    values = ("gold", "silver") if assignment == 0 else ("silver", "gold")
                    records = FORMATS[fmt].format(a=keys[0], b=keys[1], va=values[0], vb=values[1])
                    text = "Each key has one value.\n" + records + "\n\n" + SPACERS[(g // 4) % 4]
                    suffix_char = len(text)
                    text += f"\nRequested key: {keys[query]}\nValue:"
                    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    ids, offsets = enc["input_ids"], enc["offset_mapping"]
                    assert 8 < len(ids) <= 128
                    slots = [i for i, token in enumerate(ids) if token in outputs]
                    assert len(slots) == 2
                    assert [ids[i] for i in slots] == [
                        outputs[(assignment + j) % 2] for j in range(2)
                    ]
                    suffix_start = next(i for i, (a, b) in enumerate(offsets) if b > suffix_char)
                    assert slots[-1] < suffix_start
                    label = 1 if parse(text) == "gold" else -1
                    assert label == (1 if values[query] == "gold" else -1)
                    rows.append(
                        dict(
                            id=f"{group}-q{query}-a{assignment}",
                            group=group,
                            split=split,
                            format=fmt,
                            keys=keys,
                            query_record=query,
                            assignment=assignment,
                            text=text,
                            token_ids=ids,
                            label=label,
                            value_positions=slots,
                            suffix_start=suffix_start,
                            first_value=1 if values[0] == "gold" else -1,
                            last_value=1 if values[1] == "gold" else -1,
                            opposite_id=f"{group}-q{query}-a{1 - assignment}",
                            query_swap_id=f"{group}-q{1 - query}-a{assignment}",
                        )
                    )
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        other = by_id[r["opposite_id"]]
        assert Counter(r["token_ids"]) == Counter(other["token_ids"])
        assert len(r["token_ids"]) == len(other["token_ids"])
        assert r["token_ids"][-8:] == other["token_ids"][-8:]
        assert r["value_positions"] == other["value_positions"]
        # Identical pre-query records for the query intervention, including position.
        q = by_id[r["query_swap_id"]]
        assert len(r["token_ids"]) == len(q["token_ids"])
        assert r["token_ids"][: r["suffix_start"]] == q["token_ids"][: q["suffix_start"]]
        candidates = [
            s
            for s in rows
            if s["split"] == r["split"]
            and s["format"] == r["format"]
            and s["group"] != r["group"]
            and s["label"] == r["label"]
            and s["query_record"] == r["query_record"]
        ]
        r["same_answer_id"] = min(candidates, key=lambda s: s["id"])["id"]
    assert len({r["text"] for r in rows}) == len(rows)
    return {"output_ids": outputs, "rows": rows}


def baselines(rows):
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        other = by_id[r["opposite_id"]]
        assert Counter(r["token_ids"]) == Counter(other["token_ids"])
        assert r["token_ids"][-8:] == other["token_ids"][-8:]
        assert r["label"] == -other["label"]
    return {
        "constant": 0.5,
        "token_counts_upper_bound": 0.5,
        "last_token_upper_bound": 0.5,
        "position_upper_bound": 0.5,
        "last_eight_tokens_upper_bound": 0.5,
        "first_value": sum(r["label"] == r["first_value"] for r in rows) / len(rows),
        "last_value": sum(r["label"] == r["last_value"] for r in rows) / len(rows),
        "parser_positive_control": sum(
            (1 if parse(r["text"]) == "gold" else -1) == r["label"] for r in rows
        )
        / len(rows),
    }
