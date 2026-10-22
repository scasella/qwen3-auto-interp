"""Run unchanged legacy auditors against their frozen release dataset universe.

Only the v8 auditor's broad dataset-discovery glob is scoped. Reads, hashes,
output capture and all scientific arithmetic remain the original verifier's.
Later (including invalid) experiments are independently checked by the v9 audit.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def historical_dataset_view(root):
    root = Path(root).absolute()
    allowed = set(json.loads((root / "release/protected-sha256.json").read_text())["files"])
    original_glob = Path.glob

    def scoped_glob(path, pattern, *args, **kwargs):
        values = original_glob(path, pattern, *args, **kwargs)
        if path.absolute() == root / "data" and pattern == "bounded-poc-*/dataset.json":
            return (p for p in values if str(p.absolute().relative_to(root)) in allowed)
        return values

    with patch.object(Path, "glob", scoped_glob):
        yield


def main():
    from scripts import verify_all_audits as legacy

    print(
        "Historical dataset discovery scoped to release/protected-sha256.json; "
        "original auditors and hashes unchanged.",
        flush=True,
    )
    with historical_dataset_view(legacy.ROOT):
        return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
