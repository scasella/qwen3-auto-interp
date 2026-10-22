"""The packaging boundary must reject private paths and non-source payloads."""

import pytest

from scripts.verify_source_only import allowed, check_content


@pytest.mark.parametrize(
    "name",
    [
        "data/corpus.jsonl",
        "reports/result.md",
        "release/proof.json",
        "incidents/log.txt",
        "consolidation/record.json",
        "../outside.py",
        "src/checkpoint.safetensors",
        ".env",
        ".git/objects/object",
    ],
)
def test_package_rejects_unapproved_paths(name):
    assert not allowed(name)


def test_package_detects_personal_path_and_binary():
    home = "/".join(["", "Users", "example-person", "private", "file"])
    with pytest.raises(ValueError, match="home path"):
        check_content("src/example.py", home.encode())
    with pytest.raises(UnicodeDecodeError):
        check_content("src/example.py", b"\xff")


def test_package_allows_source_and_generic_lock():
    check_content("src/example.py", b'lock = "/tmp/interp-swarm-mps.lock"')
    assert allowed("docs/PAPER.md")
