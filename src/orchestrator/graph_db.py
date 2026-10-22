import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import networkx as nx

from src.schemas.contracts import PrimitiveHypothesis, UnitRef


class GraphDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, manifest TEXT NOT NULL,
                                          started TEXT DEFAULT CURRENT_TIMESTAMP);
          CREATE TABLE IF NOT EXISTS states (run TEXT, unit TEXT, state TEXT, detail TEXT,
                                            PRIMARY KEY(run, unit));
          CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, run TEXT, unit TEXT,
                strike INTEGER, hypothesis TEXT, evaluation TEXT);
          CREATE TABLE IF NOT EXISTS primitives (run TEXT, unit TEXT, hypothesis TEXT,
                discovery TEXT, holdout TEXT, r_squared REAL, PRIMARY KEY(run, unit));
          CREATE TABLE IF NOT EXISTS circuits (run TEXT, id TEXT, evidence TEXT,
                                               PRIMARY KEY(run, id));
          CREATE TABLE IF NOT EXISTS pi_reports (id INTEGER PRIMARY KEY, run TEXT, report TEXT);
          CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run TEXT, kind TEXT,
                detail TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

    def close(self):
        self.db.close()

    def event(self, run, kind, detail):
        with self.db:
            self.db.execute(
                "INSERT INTO events (run, kind, detail) VALUES (?, ?, ?)", [run, kind, detail]
            )

    def start_run(self, manifest, resume=None):
        encoded = json.dumps(manifest, sort_keys=True)
        if resume:
            row = self.db.execute("SELECT manifest FROM runs WHERE id=?", [resume]).fetchone()
            if row is None or row[0] != encoded:
                raise ValueError(
                    "Resume requires the exact calibration, model, and experiment settings"
                )
            return resume
        run = uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO runs (id, manifest) VALUES (?, ?)", [run, encoded])
        return run

    def state(self, run, unit, state, detail=""):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?)", [run, unit.key, state, detail]
            )

    def states(self, run):
        return {
            u: s for u, s in self.db.execute("SELECT unit, state FROM states WHERE run=?", [run])
        }

    def record_attempt(self, run, unit, strike, hypothesis, evaluation):
        with self.db:
            self.db.execute(
                "INSERT INTO attempts (run, unit, strike, hypothesis, evaluation) VALUES (?, ?, ?, ?, ?)",
                [run, unit.key, strike, hypothesis.model_dump_json(), evaluation.model_dump_json()],
            )

    def attempts(self, run, unit):
        return [
            (s, json.loads(h), json.loads(e))
            for s, h, e in self.db.execute(
                "SELECT strike, hypothesis, evaluation FROM attempts WHERE run=? AND unit=? ORDER BY id",
                [run, unit.key],
            )
        ]

    def verify(self, run, hypothesis, discovery, holdout):
        if (
            not discovery.passed
            or not holdout.passed
            or holdout.stage != "holdout"
            or discovery.stage != "discovery"
        ):
            raise ValueError("Verification requires passing discovery and a fresh holdout")
        earlier = {r.case.text for r in discovery.results}
        if earlier & {r.case.text for r in holdout.results}:
            raise ValueError("Discovery and holdout overlap")
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO primitives VALUES (?, ?, ?, ?, ?, ?)",
                [
                    run,
                    hypothesis.unit.key,
                    hypothesis.model_dump_json(),
                    discovery.model_dump_json(),
                    holdout.model_dump_json(),
                    holdout.r_squared,
                ],
            )
            self.db.execute(
                "INSERT OR REPLACE INTO states VALUES (?, ?, 'verified', '')",
                [run, hypothesis.unit.key],
            )

    def verified(self, run):
        return [
            PrimitiveHypothesis.model_validate_json(r[0])
            for r in self.db.execute(
                "SELECT hypothesis FROM primitives WHERE run=? ORDER BY unit", [run]
            )
        ]

    def add_circuit(self, run, evidence):
        verified = {h.unit for h in self.verified(run)}
        if evidence.target not in verified or not set(evidence.sources) <= verified:
            raise ValueError("Circuit endpoints must be verified in this run")
        if any(s.order >= evidence.target.order for s in evidence.sources):
            raise ValueError("Circuit edges must follow causal execution order")
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO circuits VALUES (?, ?, ?)",
                [run, evidence.id, evidence.model_dump_json()],
            )

    def add_report(self, run, report):
        with self.db:
            self.db.execute(
                "INSERT INTO pi_reports (run, report) VALUES (?, ?)",
                [run, report.model_dump_json()],
            )

    def graph(self, run):
        graph = nx.MultiDiGraph(run=run)
        for h in self.verified(run):
            graph.add_node(h.unit.key, hypothesis=h.model_dump(mode="json"))
        for row in self.db.execute("SELECT evidence FROM circuits WHERE run=?", [run]):
            evidence = json.loads(row[0])
            target = UnitRef.model_validate(evidence["target"]).key
            sources = [UnitRef.model_validate(s).key for s in evidence["sources"]]
            for source in sources:
                graph.add_edge(
                    source,
                    target,
                    key=evidence["id"],
                    gate=evidence["gate"],
                    group=evidence["id"],
                    joint_sources=sources,
                    evidence=evidence,
                )
        return graph

    def summary(self, run):
        counts = dict(
            self.db.execute("SELECT state, count(*) FROM states WHERE run=? GROUP BY state", [run])
        )
        return {
            "run": run,
            "states": counts,
            "circuits": self.db.execute(
                "SELECT count(*) FROM circuits WHERE run=?", [run]
            ).fetchone()[0],
        }
