"""Exact empirical token quantiles, compact stratified examples, bounded unit shards."""

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from src.config import Settings
from src.harness.qwen_model import Probe, ProbeSpec
from src.schemas.contracts import UnitRef


@dataclass
class Quantiles:
    count: int
    mean: float
    p001: float
    p05: float
    p50: float
    p80: float
    p95: float
    p999: float

    @property
    def span(self):
        return self.p95 - self.p05


def corpus_stream(path: Path, maximum: int):
    """One document per text line, or JSONL objects with a string `text` field."""
    with path.open() as stream:
        for line in islice((s for s in stream if s.strip()), maximum):
            text = json.loads(line)["text"] if path.suffix == ".jsonl" else line.rstrip("\n")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Corpus text must be a nonempty string")
            yield text


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ActivationIndex:
    def __init__(self, path: Path, memory_limit="2GB"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(path))
        self.db.execute("SET memory_limit = ?", [memory_limit])
        self.db.execute("SET threads = 2")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        self.db.execute("""CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR PRIMARY KEY, text VARCHAR, token_ids INTEGER[], pieces VARCHAR[])""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS quantiles (
            unit VARCHAR PRIMARY KEY, count BIGINT, mean DOUBLE,
            p001 DOUBLE, p05 DOUBLE, p50 DOUBLE, p80 DOUBLE, p95 DOUBLE, p999 DOUBLE)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS examples (
            id VARCHAR PRIMARY KEY, unit VARCHAR, band VARCHAR, doc_id VARCHAR,
            token_position INTEGER, activation DOUBLE)""")
        self.db.execute("""CREATE TEMP TABLE raw (
            unit VARCHAR, doc_id VARCHAR, token_position INTEGER, activation DOUBLE)""")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS evidence_controls (id VARCHAR PRIMARY KEY, data VARCHAR)"
        )

    def close(self):
        self.db.close()

    def metadata(self):
        return {k: json.loads(v) for k, v in self.db.execute("SELECT * FROM metadata").fetchall()}

    def set_manifest(self, manifest: dict):
        existing = self.metadata().get("manifest")
        if existing is not None and existing != manifest:
            raise ValueError(
                "Calibration provenance differs; use a new data path for this corpus/model"
            )
        self.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('manifest', ?)",
            [json.dumps(manifest, sort_keys=True)],
        )

    def units(self) -> list[UnitRef]:
        return [
            UnitRef.parse(row[0])
            for row in self.db.execute("SELECT unit FROM quantiles ORDER BY unit").fetchall()
        ]

    def quantiles(self, unit: UnitRef) -> Quantiles:
        row = self.db.execute(
            "SELECT * EXCLUDE(unit) FROM quantiles WHERE unit=?", [unit.key]
        ).fetchone()
        if row is None:
            raise KeyError(f"Unit is not calibrated: {unit.key}")
        return Quantiles(*row)

    def examples(self, unit: UnitRef) -> list[dict]:
        rows = self.db.execute(
            """SELECT e.id, e.band, e.token_position, e.activation,
            d.text, d.token_ids, d.pieces FROM examples e JOIN documents d ON e.doc_id=d.id
            WHERE unit=? ORDER BY band, activation, e.id""",
            [unit.key],
        ).fetchall()
        controls = dict(self.db.execute("SELECT id,data FROM evidence_controls").fetchall())
        return [
            dict(
                id=r[0],
                band=r[1],
                token_position=r[2],
                activation=r[3],
                text="".join(r[6][: r[2] + 1]),
                token_ids=r[5][: r[2] + 1],
                token_pieces=r[6][: r[2] + 1],
                target_token=r[6][r[2]],
                controls=json.loads(controls[r[0]]) if r[0] in controls else None,
            )
            for r in rows
        ]

    def distribution(self, unit: UnitRef):
        return {
            "quantiles": asdict(self.quantiles(unit)),
            "examples": self.examples(unit),
            "observable": "post-SwiGLU scalar"
            if unit.kind == "mlp"
            else "signed mean of one query head output before o_proj",
        }

    def add_batch(self, texts, results, units):
        doc_ids = []
        for text, result in zip(texts, results):
            doc_id = hashlib.sha256(text.encode()).hexdigest()
            doc_ids.append(doc_id)
            self.db.execute(
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?)",
                [doc_id, text, result.token_ids, result.token_pieces],
            )
        docs = [doc_id for doc_id, r in zip(doc_ids, results) for _ in r.token_ids]
        positions = [i for r in results for i in range(len(r.token_ids))]
        for unit in units:
            values = np.concatenate([r.values[unit.key] for r in results])
            if not np.isfinite(values).all():
                raise ValueError("Non-finite calibration data")
            table = pa.table(
                {
                    "unit": [unit.key] * len(values),
                    "doc_id": docs,
                    "token_position": positions,
                    "activation": values,
                }
            )
            self.db.register("activation_batch", table)
            try:
                self.db.execute("INSERT INTO raw SELECT * FROM activation_batch")
            finally:
                self.db.unregister("activation_batch")

    def finalize(self, units: list[UnitRef], examples_per_band: int, minimum: int):
        self.db.execute("BEGIN TRANSACTION")
        try:
            for unit in units:
                row = self.db.execute(
                    """SELECT count(*), avg(activation),
                    quantile_cont(activation, [0.001, 0.05, 0.5, 0.8, 0.95, 0.999])
                    FROM raw WHERE unit=?""",
                    [unit.key],
                ).fetchone()
                if row[0] < minimum:
                    raise ValueError(f"{unit.key}: need {minimum} token observations, got {row[0]}")
                self.db.execute(
                    "INSERT OR REPLACE INTO quantiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [unit.key, row[0], row[1], *row[2]],
                )
                self.db.execute("DELETE FROM examples WHERE unit=?", [unit.key])
                q = Quantiles(row[0], row[1], *row[2])
                for band, lower, upper in (
                    ("positive", q.p999, float("inf")),
                    ("boundary", q.p50, q.p80),
                    ("inhibitory", float("-inf"), q.p001),
                ):
                    # Stable hash sampling avoids selecting only one end of the boundary band.
                    examples = self.db.execute(
                        """SELECT DISTINCT doc_id, token_position, activation
                        FROM raw WHERE unit=? AND activation>=? AND activation<=?
                        ORDER BY hash(doc_id, token_position) LIMIT ?""",
                        [unit.key, lower, upper, examples_per_band],
                    ).fetchall()
                    for doc_id, position, activation in examples:
                        eid = f"{unit.key}/{band}/{doc_id[:16]}/{position}"
                        self.db.execute(
                            "INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?)",
                            [eid, unit.key, band, doc_id, position, activation],
                        )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        finally:
            self.db.execute("DELETE FROM raw")


async def calibrate(
    worker,
    index: ActivationIndex,
    corpus: Path,
    units: list[UnitRef],
    settings: Settings,
    progress=None,
):
    index.set_manifest(
        {
            "model": settings.model.model_dump(),
            "resolved_revision": worker.harness.revision,
            "max_seq_len": settings.worker.max_seq_len,
            "corpus_sha256": file_digest(corpus),
            "max_documents": settings.calibration.max_documents,
            "observable_version": "swiglu-and-head-component-mean/v1",
            "population": "all non-padding tokens in truncated raw-text documents",
        }
    )
    done = {u.key for u in index.units()}
    todo = [u for u in units if u.key not in done]
    shard_size = settings.calibration.units_per_pass
    for start in range(0, len(todo), shard_size):
        shard = todo[start : start + shard_size]
        spec = ProbeSpec(tuple(shard), all_tokens=True, allow_truncate=True)
        stream = corpus_stream(corpus, settings.calibration.max_documents)
        while texts := list(islice(stream, settings.worker.batch_size)):
            results = await worker.map([Probe(t, spec) for t in texts])
            index.add_batch(texts, results, shard)
        index.finalize(
            shard, settings.calibration.examples_per_band, settings.calibration.min_tokens
        )
        if progress:
            progress(min(start + shard_size, len(todo)), len(todo))


def split_evidence(distribution):
    """Fixed per-band discovery/screen split; screen labels never reach explorers."""
    training, screen = [], []
    for band in ("positive", "boundary", "inhibitory"):
        rows = sorted(
            (e for e in distribution["examples"] if e["band"] == band), key=lambda e: e["id"]
        )
        if len(rows) < 4:
            raise ValueError("Need at least four examples per band for independent screening")
        training.extend(rows[::2])
        screen.extend(rows[1::2])
    return training, screen


def compact_example(example):
    return {k: v for k, v in example.items() if k not in {"token_ids", "token_pieces"}} | {
        "target_token_id": example["token_ids"][-1],
        "token_count": len(example["token_ids"]),
    }
