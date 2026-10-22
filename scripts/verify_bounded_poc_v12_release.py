"""Read-only publication verification; frozen auditor writes are captured in memory."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(inventory):
    from scripts import audit_bounded_poc_v12_cross_scale_3hop as auditor

    record = json.loads(inventory.read_text())
    for name, expected in record["files"].items():
        assert digest(ROOT / name) == expected, name
    captured = {}

    def capture(name, value):
        assert name not in captured
        captured[name] = json.loads(json.dumps(value, allow_nan=False))

    original = auditor.write
    try:
        auditor.write = capture
        with contextlib.redirect_stdout(io.StringIO()):
            auditor.audit()
    finally:
        auditor.write = original
    assert set(captured) == {"results.json", "facts.json", "audit.json"}
    for name, value in captured.items():
        assert value == json.loads((auditor.OUT / name).read_text()), name
    facts = captured["facts.json"]
    directories = [auditor.OUT]
    selection_path = auditor.OUT / "final-review.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        assert selection["facts_sha256"] == digest(auditor.OUT / "facts.json")
        selected = (auditor.OUT / selection["directory"]).resolve()
        assert selected.is_relative_to(auditor.OUT.resolve()) and selected != auditor.OUT.resolve()
        directories.append(selected)
    attempts = []
    for directory in directories:
        review = json.loads((directory / "review-results.json").read_text())
        adjudication = json.loads((directory / "review-adjudication.json").read_text())
        payload = json.loads((directory / "review-payload.json").read_text())
        assert payload["facts"] == facts
        checks = []
        for claim in review["numeric_claims"]:
            key = claim["metric"]
            checks.append(
                dict(
                    metric=key,
                    claimed=claim["value"],
                    actual=facts.get(key),
                    supported=key in facts and claim["value"] == facts[key],
                )
            )
        assert checks == adjudication["claims"]
        supported = bool(checks) and all(x["supported"] for x in checks)
        assert supported == adjudication["all_numeric_claims_supported"]
        attempts.append(
            dict(
                directory=str(directory.relative_to(auditor.OUT)),
                references=len(checks),
                all_supported=supported,
            )
        )
    for name, expected in record["files"].items():
        assert digest(ROOT / name) == expected, name
    result = captured["results.json"]
    return dict(
        verification_passed=True,
        scientific_passed=result["passed"],
        gates=result["gates"],
        release_files=len(record["files"]),
        protected_files=result["protected_files"],
        repeat_arrays=result["repeat_arrays"],
        repeat_file_pairs=result["repeat_file_pairs"],
        maximum_delta_error=result["maximum_delta_error"],
        review_attempts=attempts,
        review_references=len(checks),
        all_review_references_supported=supported,
        inventory_sha256=digest(inventory),
        mode="Read-only: auditor writes captured and compared",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "release/bounded-poc-v12-cross-scale-3hop-sha256.json"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.inventory), indent=2, allow_nan=False))
