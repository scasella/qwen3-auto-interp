"""Fetch deterministic, provenance-preserving samples through HF Dataset Viewer."""

import concurrent.futures
import hashlib
import json
import random
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/targeted-real-v1"
SOURCES = [
    ("encyclopedia", "Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    ("news", "fancyzhx/ag_news", "default", "text"),
    ("reviews", "stanfordnlp/imdb", "plain_text", "text"),
    ("science", "ccdv/arxiv-summarization", "section", "abstract"),
    ("python", "google-research-datasets/mbpp", "full", "code"),
]


def fetch(endpoint, params):
    url = (
        "https://datasets-server.huggingface.co/" + endpoint + "?" + urllib.parse.urlencode(params)
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def inspect(source):
    domain, dataset, config, field = source
    cached = OUT / f"{domain}-metadata.json"
    if cached.exists():
        return json.loads(cached.read_text())
    splits = fetch("splits", {"dataset": dataset})
    available = [(s["config"], s["split"]) for s in splits.get("splits", [])]
    if (config, "train") not in available:
        raise ValueError(f"{dataset}: expected {config}/train; available={available}")
    sample = fetch("first-rows", {"dataset": dataset, "config": config, "split": "train"})
    if field not in sample["rows"][0]["row"]:
        raise ValueError(f"{dataset}: fields={list(sample['rows'][0]['row'])}")
    first = fetch(
        "rows", {"dataset": dataset, "config": config, "split": "train", "offset": 0, "length": 1}
    )
    info = {
        "domain": domain,
        "dataset": dataset,
        "config": config,
        "field": field,
        "split": "train",
        "num_rows": first["num_rows_total"],
        "url": f"https://huggingface.co/datasets/{dataset}",
    }
    (OUT / f"{domain}-metadata.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return info


def sample_source(info):
    cache = OUT / (info["domain"] + "-sample.json")
    if cache.exists():
        return json.loads(cache.read_text())
    rng = random.Random(20261019 + sum(map(ord, info["domain"])))
    n = info["num_rows"]
    candidates = {}
    if n < 500:
        offsets = list(range(0, n, 100))
        page_size = 100
    else:
        offsets = [int((i + rng.random()) / 8 * (n - 32)) for i in range(8)]
        offsets += [rng.randrange(n - 32) for _ in range(16)]
        page_size = 32
    for page, offset in enumerate(offsets):
        result = fetch(
            "rows",
            {
                "dataset": info["dataset"],
                "config": info["config"],
                "split": info["split"],
                "offset": offset,
                "length": min(page_size, n - offset),
            },
        )
        for item in result["rows"]:
            row = item["row"]
            value = row[info["field"]]
            if isinstance(value, list):
                value = "\n".join(value)
            text = value.strip()
            if len(text) < (70 if info["domain"] == "python" else 160):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            candidates[digest] = {
                "text": text,
                "domain": info["domain"],
                "dataset": info["dataset"],
                "config": info["config"],
                "split": info["split"],
                "row_index": item["row_idx"],
                "source_url": info["url"],
                "field": info["field"],
                "text_sha256": digest,
                "label": row.get("label"),
                "viewer_truncated_cells": item.get("truncated_cells", []),
            }
        if len(candidates) >= 160 and (page >= 7 or n < 500):
            break
    records = list(candidates.values())
    rng.shuffle(records)
    if len(records) < 128:
        raise ValueError(f"Insufficient eligible rows: {info['domain']}: {len(records)}")
    records = records[:128]
    cache.write_text(json.dumps(records, ensure_ascii=False))
    print(f"Sampled {len(records)} {info['domain']} documents", flush=True)
    return records


def literature():
    cache = OUT / "literature-sample.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import re

    rng = random.Random(20261019)
    records = []
    for book_id in [11, 1342, 84, 1661, 2701, 98, 35, 345]:
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        raw_path = OUT / f"gutenberg-{book_id}.txt"
        if not raw_path.exists():
            subprocess.run(
                [
                    "/usr/bin/curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--max-time",
                    "60",
                    url,
                    "--output",
                    str(raw_path),
                ],
                check=True,
            )
        text = raw_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        starts = list(re.finditer(r"\*\*\* START OF[^\n]+", text))
        ends = list(re.finditer(r"\*\*\* END OF[^\n]+", text))
        body = text[starts[0].end() if starts else 0 : ends[-1].start() if ends else len(text)]
        paragraphs = [
            (i, p.strip())
            for i, p in enumerate(re.split(r"\n\s*\n", body))
            if 200 <= len(p.strip()) <= 2000
        ]
        for index, paragraph in rng.sample(paragraphs, 16):
            records.append(
                {
                    "text": paragraph,
                    "domain": "literature",
                    "dataset": "Project Gutenberg",
                    "book_id": book_id,
                    "paragraph_index": index,
                    "source_url": url,
                    "text_sha256": hashlib.sha256(paragraph.encode()).hexdigest(),
                }
            )
    cache.write_text(json.dumps(records, ensure_ascii=False))
    print(f"Sampled {len(records)} literary paragraphs across 8 books", flush=True)
    return records


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        sources = list(pool.map(inspect, SOURCES))
        samples = list(pool.map(sample_source, sources))
    samples.append(literature())
    records = [r for sample in samples for r in sample]
    random.Random(20261019).shuffle(records)
    corpus = OUT / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    manifest = {
        "sources": sources,
        "literature_books": [11, 1342, 84, 1661, 2701, 98, 35, 345],
        "documents": len(records),
        "per_domain": {
            d: sum(r["domain"] == d for r in records)
            for d in sorted({r["domain"] for r in records})
        },
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "seed": 20261019,
        "sampling": "Stratified random dataset page offsets followed by fixed-seed row selection; code from MBPP human-written training functions; 16 real paragraphs per each of 8 Gutenberg books.",
        "notes": [
            "English and Python only",
            "128-token document prefixes used for calibration",
            "Viewer dataset revisions are not pinned; sampled text hashes preserve the actual downloaded corpus",
            "PG19 viewer was unavailable (HTTP 501); literary passages obtained directly from Gutenberg instead",
        ],
    }
    (OUT / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps({k: manifest[k] for k in ("documents", "per_domain", "corpus_sha256")}),
        flush=True,
    )
