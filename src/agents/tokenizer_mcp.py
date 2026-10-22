"""Read-only stdio MCP service: Rust tokenization only, no torch/model execution."""

import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

TOOL = {
    "name": "check_cases",
    "description": "Check Qwen token lengths and final tokens before returning a synthetic suite. Call on the entire requested batch, repair errors, and recheck until errors is empty. No activations or semantic judgements are available.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "minItems": 1,
                "maxItems": 45,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string", "maxLength": 1500},
                        "battery": {"type": "string"},
                        "paired_positive_id": {"type": ["string", "null"]},
                    },
                    "required": ["id", "text", "battery", "paired_positive_id"],
                },
            },
            "minimum_positive_tokens": {"type": "integer", "minimum": 1, "maximum": 128},
            "minimum_inhibitory_tokens": {"type": "integer", "minimum": 1, "maximum": 128},
        },
        "required": ["cases"],
        "additionalProperties": False,
    },
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


def check_cases(tokenizer, args, minimums=None):
    cases = args["cases"]
    if not 1 <= len(cases) <= 45 or any(
        not isinstance(c.get("text"), str) or len(c["text"]) > 1500 for c in cases
    ):
        raise ValueError("Provide 1–45 cases with at most 1500 characters each")
    ids = [c["id"] for c in cases]
    if any("battery" not in c or "paired_positive_id" not in c for c in cases):
        raise ValueError(
            "Every case requires battery and paired_positive_id (null for unpaired cases)"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique")
    encoded = {c["id"]: tokenizer.encode(c["text"], add_special_tokens=False).ids for c in cases}
    output = []
    errors = []
    expected = (minimums or {}).get("expected_cases", 45)
    if len(cases) == expected and Counter(c["battery"] for c in cases) != Counter(
        {
            "OOD_POSITIVE": expected // 3,
            "SEMANTIC_STRIPPING": expected // 3,
            "INHIBITORY_POLE": expected // 3,
        }
    ):
        errors.append("A complete request requires equal numbers in each battery")
    for c in cases:
        tokens = encoded[c["id"]]
        band = c.get("battery", "")
        row = {
            "id": c["id"],
            "count": len(tokens),
            "last_token_id": tokens[-1] if tokens else None,
            "last_token_text": tokenizer.decode(tokens[-1:], skip_special_tokens=False),
            "token_pieces": [tokenizer.decode([i], skip_special_tokens=False) for i in tokens],
        }
        output.append(row)
        if not 1 <= len(tokens) <= 128:
            errors.append(f"{c['id']}: length {len(tokens)} outside 1–128")
        supplied_minimum = (
            args.get("minimum_positive_tokens", 1)
            if band == "OOD_POSITIVE"
            else args.get("minimum_inhibitory_tokens", 1)
            if band == "INHIBITORY_POLE"
            else 1
        )
        minimum = (
            (
                minimums["positive"]
                if band == "OOD_POSITIVE"
                else minimums["inhibitory"]
                if band == "INHIBITORY_POLE"
                else 1
            )
            if minimums is not None
            else supplied_minimum
        )
        if len(tokens) < minimum:
            errors.append(f"{c['id']}: {len(tokens)} tokens; minimum {minimum}")
        partner = c.get("paired_positive_id")
        if band == "SEMANTIC_STRIPPING" and partner is None:
            errors.append(f"{c['id']}: stripping requires its positive partner")
        if partner is not None:
            if partner not in encoded:
                errors.append(f"{c['id']}: include partner {partner} in this check")
            elif (
                not tokens
                or not encoded[partner]
                or len(tokens) != len(encoded[partner])
                or tokens[-1] != encoded[partner][-1]
            ):
                errors.append(
                    f"{c['id']}/{partner}: final token ID and total token count must both match; see counts and target tokens"
                )
    texts = [c["text"] for c in cases]
    if len(set(texts)) != len(texts):
        errors.append("Every text must be unique")
    return {
        "cases": output,
        "errors": errors,
        "valid_token_controls": not errors,
        "complete_request": len(cases) == expected,
        "required_minimums": minimums,
    }


def dispatch(tokenizer, request, minimums=None):
    method = request.get("method")
    if method == "initialize":
        return {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "qwen-tokenizer", "version": "1.0"},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [TOOL]}
    if method == "tools/call":
        if request["params"]["name"] != "check_cases":
            raise ValueError("Unknown tool")
        result = check_cases(tokenizer, request["params"]["arguments"], minimums)
        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            "isError": False,
        }
    raise ValueError("Unsupported method")


def main():
    tokenizer = Tokenizer.from_file(sys.argv[1])
    minimums = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    for line in sys.stdin:
        if len(line) > 1048576:
            raise ValueError("Oversized MCP request")
        request = json.loads(line)
        if "id" not in request:
            continue
        try:
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": dispatch(tokenizer, request, minimums),
            }
        except (KeyError, TypeError, ValueError) as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32602, "message": str(exc)},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
