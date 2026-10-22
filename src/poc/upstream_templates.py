"""New identifiers, unchanged v2 grammar, exact within-group donor axes."""

from copy import deepcopy

from src.poc.templates import _v2_parse, _v2_token_positions, build_v2_dataset


def build_upstream(tokenizer, retired):
    # Reuse the validated grammar, then regenerate every identifier and token span.
    payload = deepcopy(build_v2_dataset(tokenizer, []))
    stems = {
        "discovery": ("routeDA", "routeDB"),
        "primary": ("routeHA", "routeHB"),
        "vocabulary": ("routeVA", "routeVB"),
    }
    for row in payload["rows"]:
        old = row["keys"]
        # The original group order reversals remain deterministic and balanced.
        original = {
            "discovery": ("sequoiaA", "tundraA"),
            "primary": ("willowB", "topazB"),
            "vocabulary": ("quartzC", "umberC"),
        }[row["split"]]
        keys = [
            stems[row["split"]][next(i for i, s in enumerate(original) if k.startswith(s))] + k[-3:]
            for k in old
        ]
        text = row["text"]
        for a, b in zip(old, keys):
            text = text.replace(a, b)
        row["keys"], row["text"] = keys, text
        row["id"] = row["id"].replace("v2-", "upstream-")
        row["group"] = row["group"].replace("v2-", "upstream-")
        row["opposite_id"] = row["opposite_id"].replace("v2-", "upstream-")
        row["same_answer_id"] = row["same_answer_id"].replace("v2-", "upstream-")
        row["query_opposite_id"] = f"{row['group']}-q{1 - row['query_record']}-a{row['assignment']}"
        answer, parsed, _ = _v2_parse(text)
        assert answer == row["answer_color"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        row["token_ids"] = encoded["input_ids"]
        row["color_positions"] = [_v2_token_positions(offsets, p["color_span"])[0] for p in parsed]
        assert all(len(_v2_token_positions(offsets, p["color_span"])) == 1 for p in parsed)
        row["color_spans"] = [list(p["color_span"]) for p in parsed]
        row["key_positions"] = [_v2_token_positions(offsets, p["key_span"]) for p in parsed]
        row["delimiter_positions"] = [
            i for p in parsed for i in _v2_token_positions(offsets, p["delimiter_span"])
        ]
        start = text.index("Requested identifier: ") + len("Requested identifier: ")
        query_span = (start, start + len(keys[row["query_record"]]))
        row["query_positions"] = _v2_token_positions(offsets, query_span)
        row["context_end"] = row["query_positions"][0]
        assert len(row["token_ids"]) <= 128
        ids = payload["output_ids"]["vocabulary" if row["split"] == "vocabulary" else "primary"]
        assert sorted(row["token_ids"][p] for p in row["color_positions"]) == sorted(ids)
    validate(payload["rows"])
    assert not {r["text"] for r in retired} & {r["text"] for r in payload["rows"]}
    assert not {k for r in retired for k in r["keys"]} & {
        k for r in payload["rows"] for k in r["keys"]
    }
    payload["sanity"] = {
        "passed": True,
        "rows": len(payload["rows"]),
        "fresh_identifiers": True,
        "axis_B_equal_length_and_context": True,
        "target_forwards": 0,
    }
    return payload


def validate(rows):
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(rows)
    for r in rows:
        a, b = by_id[r["opposite_id"]], by_id[r["query_opposite_id"]]
        assert a["label"] == b["label"] == -r["label"]
        assert b["assignment"] == r["assignment"] and b["query_record"] != r["query_record"]
        assert a["query_record"] == r["query_record"]
        assert len(r["token_ids"]) == len(b["token_ids"]) == len(a["token_ids"])
        assert r["color_positions"] == b["color_positions"] == a["color_positions"]
        assert r["context_end"] == b["context_end"]
        assert r["token_ids"][: r["context_end"]] == b["token_ids"][: b["context_end"]]
        changed = {i for i, (x, y) in enumerate(zip(r["token_ids"], b["token_ids"])) if x != y}
        assert changed and changed <= set(r["query_positions"]) & set(b["query_positions"])
