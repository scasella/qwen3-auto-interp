"""Executable planted retrieval/offset and recency models, with saved stage evidence."""

import numpy as np

from src.poc.v8_repair import centroid, grade_A, grade_B, output_center


def logits(m):
    return np.stack([m / 2, -m / 2], 1)


def planted():
    records = {}
    for track in ("A", "B"):
        for kind in ("positive", "nuisance", "reversed"):
            rows = []
            for g in range(8):
                for q in range(4):
                    for flip in range(2):
                        v = np.array([1, -1] if track == "A" else [1, 1, -1, -1]) * (-1) ** flip
                        query = ((q % 2) ^ (q // 2)) if track == "A" else q
                        rows.append(
                            dict(
                                group=f"{g}",
                                track=track,
                                label=int(v[query]),
                                query_record=query,
                                last_value=int(v[-1]),
                                values=v.tolist(),
                            )
                        )
            y = np.array([r["label"] for r in rows])
            values = np.array([r["values"] for r in rows], float)
            if track == "A":
                retrieval = y.astype(float) if kind != "nuisance" else np.zeros(len(y))
                if kind == "reversed":
                    retrieval = -retrieval
                h = np.repeat(retrieval[:, None, None], 3, 1)
                w, b = centroid(h, y)
                # Negative controls use an uninformative/wrong causal repair even if a fitted decoder can invert.
                base = logits(retrieval + 3)
                repaired = output_center(logits(2.5 * retrieval + 3), base.mean(0), 1)
                probe = h.reshape(len(y), -1) @ w + b
                result = grade_A(rows, base, repaired, probe)
                record = dict(
                    result=result,
                    mean_replace_fixed=int(
                        np.sum(
                            (y * (base[:, 0] - base[:, 1]) <= 0)
                            & (y * (retrieval + np.full(len(y), 3.0).mean()) > 0)
                        )
                    ),
                    probe_accuracy=result["probe_accuracy"],
                    evidence=dict(
                        rows=rows,
                        h=h.tolist(),
                        bias=3.0,
                        base=base.tolist(),
                        repaired=repaired.tolist(),
                        probe=probe.tolist(),
                    ),
                )
                record["passed"] = result["A1"] and result["A2"]
            else:
                prior = np.array([0.15, 0.15, 0.15, 0.55])
                p = []
                bypass = []
                for row, v in zip(rows, values):
                    q = row["query_record"]
                    target = np.eye(4)[q]
                    pattern = prior + 0.2 * (target - 0.25)
                    if kind == "nuisance":
                        pattern = prior.copy()
                    p.append(pattern)
                    opposing = np.where(v != v[3])[0]
                    bypass.append(
                        0.0 if kind == "nuisance" else (0.1 if q == opposing[0] else 0.3) * v[q]
                    )
                p = np.array(p)
                bypass = np.array(bypass)
                h = (p * values).sum(1)
                sign = -1 if kind == "reversed" else 1
                base = logits(sign * (h + bypass))
                naive = logits(sign * (h + (h - h.mean()) + bypass))
                mean = p.mean(0)
                fixed = p - (mean - mean.mean())
                repair = logits(sign * ((fixed * values).sum(1) + bypass))
                result = grade_B(rows, base, repair, naive)
                record = dict(
                    result=result,
                    mean_p=p.mean(0).tolist(),
                    passed=result["passed"],
                    evidence=dict(
                        rows=rows,
                        p=p.tolist(),
                        values=values.tolist(),
                        bypass=bypass.tolist(),
                        h=h.tolist(),
                        base=base.tolist(),
                        repaired=repair.tolist(),
                        naive=naive.tolist(),
                        routed=fixed.tolist(),
                    ),
                )
            records[f"{track}_{kind}"] = record
    records["passed"] = all(records[f"{t}_positive"]["passed"] for t in ("A", "B")) and all(
        not records[f"{t}_{k}"]["passed"] for t in ("A", "B") for k in ("nuisance", "reversed")
    )
    assert np.isclose(records["B_positive"]["mean_p"][3], 0.55)
    assert (
        records["B_positive"]["result"]["naive"]["fixed"]
        < records["B_positive"]["result"]["naive"]["broken"]
    )
    return records
