import json
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from src.agents.tokenizer_mcp import check_cases


def tokenizer():
    t = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "red": 1, "blue": 2, "target": 3, "other": 4}, unk_token="[UNK]"
        )
    )
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return t


def test_token_controls_detect_and_accept_matched_replacements():
    cases = [
        {"id": "P", "text": "red target", "battery": "OOD_POSITIVE", "paired_positive_id": None},
        {
            "id": "S",
            "text": "blue target",
            "battery": "SEMANTIC_STRIPPING",
            "paired_positive_id": "P",
        },
    ]
    assert check_cases(tokenizer(), {"cases": cases})["valid_token_controls"]
    cases[1]["text"] = "blue other"
    assert not check_cases(tokenizer(), {"cases": cases})["valid_token_controls"]
    cases[1]["text"] = "red blue target"
    assert not check_cases(tokenizer(), {"cases": cases})["valid_token_controls"]


def test_stdio_protocol_exposes_only_readonly_tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer().save(str(path))
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "check_cases",
                "arguments": {
                    "cases": [
                        {
                            "id": "P",
                            "text": "red target",
                            "battery": "OOD_POSITIVE",
                            "paired_positive_id": None,
                        }
                    ]
                },
            },
        },
    ]
    script = Path(__file__).resolve().parents[1] / "src/agents/tokenizer_mcp.py"
    result = subprocess.run(
        [sys.executable, str(script), str(path)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    replies = [json.loads(line) for line in result.stdout.split("\n") if line]
    assert len(replies) == 3
    assert [t["name"] for t in replies[1]["result"]["tools"]] == ["check_cases"]
    assert replies[1]["result"]["tools"][0]["annotations"]["readOnlyHint"]
    checked = json.loads(replies[2]["result"]["content"][0]["text"])
    assert checked["cases"][0]["count"] == 2
    assert checked["cases"][0]["last_token_id"] == 3


def test_trusted_minimum_cannot_be_omitted_or_lowered_by_generator():
    case = {"id": "P", "text": "red target", "battery": "OOD_POSITIVE", "paired_positive_id": None}
    for supplied in ({}, {"minimum_positive_tokens": 1}):
        result = check_cases(
            tokenizer(), {"cases": [case], **supplied}, minimums={"positive": 20, "inhibitory": 20}
        )
        assert not result["valid_token_controls"]
        assert any("minimum 20" in error for error in result["errors"])
