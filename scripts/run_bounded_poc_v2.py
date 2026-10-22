"""Prospective v2 execution. V1 is read-only and all target calls use one queue."""

import argparse
import asyncio
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from src.config import ModelConfig, WorkerConfig
from src.harness.mps_queue import MPSBatchWorker
from src.poc.analysis import (
    V2_LIMITS,
    cross_validate,
    decomposition_metrics_v2,
    effect_stats,
    fit_readout,
    grouped_values,
    readout_metrics_v2,
    rescue_stats_v2,
    task_stats,
)
from src.poc.controls import known_controls_v2
from src.poc.harness import HeadAction, V2Harness, V2Probe, V2Spec
from src.poc.templates import build_v2_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bounded-poc-v2"
PROTOCOL = ROOT / "reports/bounded-poc-v2-protocol.md"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
LAYERS = tuple(range(15, 23)) + (27,)
HEADS = [f"{layer}:{h}" for layer in LAYERS for h in range(16)]
SCREEN = HEADS[:128]
REFERENCE = "19:5"
MODES = (
    "source_color",
    "source_other",
    "ov_all",
    "ov_color",
    "ov_other",
    "qk",
    "qkov",
    "interaction",
)


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def arrays(name, **values):
    with (OUT / name).open("xb") as handle:
        np.savez_compressed(handle, **values)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_v1():
    snapshot = json.loads((ROOT / "reports/bounded-poc-v1-immutability.json").read_text())
    for name, expected in snapshot.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Protected v1 artifact changed: {name}")
    actual = {
        str(p.relative_to(ROOT)) for p in (ROOT / "data/bounded-poc-v1").rglob("*") if p.is_file()
    }
    assert actual == {n for n in snapshot if n.startswith("data/bounded-poc-v1/")}
    return snapshot


def runtime_files():
    return [
        *sorted((ROOT / "src/poc").glob("*.py")),
        ROOT / "src/config.py",
        ROOT / "src/harness/qwen_model.py",
        ROOT / "src/harness/mps_queue.py",
        ROOT / "src/agents/client.py",
        ROOT / "src/agents/prompts.py",
        Path(__file__).resolve(),
        ROOT / "scripts/audit_bounded_poc_v2.py",
        ROOT / "tests/test_poc_v2.py",
    ]


def prepare():
    from transformers import AutoTokenizer

    OUT.mkdir(exist_ok=True)
    immutable_v1()
    retired = json.loads((ROOT / "data/bounded-poc-v1/dataset.json").read_text())["rows"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / f"data/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/{REVISION}"),
        local_files_only=True,
    )
    payload = build_v2_dataset(tokenizer, retired)
    save("dataset.json", payload)
    save("sanity.json", payload["sanity"])
    controls = known_controls_v2(payload)
    assert controls["passed"], "Planted controls failed"
    save("known-controls.json", controls)
    print(
        {
            "rows": len(payload["rows"]),
            "sanity": payload["sanity"]["passed"],
            "controls": controls["passed"],
        },
        flush=True,
    )


def registration(final=False):
    immutable_v1()
    assert read("sanity.json")["passed"] and read("known-controls.json")["passed"]
    paths = runtime_files() + [
        PROTOCOL,
        OUT / "dataset.json",
        OUT / "known-controls.json",
        OUT / "sanity.json",
    ]
    if final:
        verify_registration("discovery-registration.json")
        assert not any(OUT.glob("confirm[12]-started.json")), (
            "Held-out execution already started"
        )
        paths += [
            OUT / "discovery-registration.json",
            OUT / "discovery.json",
            OUT / "validation.json",
        ]
        assert read("validation.json")["all_passed"]
        paths += sorted(OUT.glob("discovery-*.npz")) + [
            OUT / "calibration-baseline.npz",
            OUT / "calibration-controls.npz",
        ]
    manifest = {
        "version": "bounded-poc-v2",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "final_pre_heldout" if final else "pre_discovery",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "device": "mps",
        "dtype": "bfloat16",
        "seed": 42,
        "batch_size": 64,
        "max_seq_len": 128,
        "limits": V2_LIMITS,
        "head_index": HEADS,
        "reference": REFERENCE,
        "rules": {
            "single": "max median signed donor effect among CV>=.75; layer/head tie break",
            "coalition": "19:5 plus two exhaustive greedy additions maximizing median joint effect",
            "interferer": "accuracy descending, regressions ascending, median gain descending, index",
            "rescue": "zero discovery regressions, max correct count, median gain, rule id; else identity",
            "alpha_grid": [1.5, 2.0],
            "heldout_runs": 2,
            "v1_heldout_retired": True,
            "baseline_primary_required": 116,
            "overall": "all four gates, unassisted task prerequisite, controls, bit-identical repeatability",
        },
        "selected": read("discovery.json")["selection"] if final else None,
        "protected_v1": immutable_v1(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
    }
    name = "manifest.json" if final else "discovery-registration.json"
    save(name, manifest)
    print(name, digest(OUT / name), flush=True)


def verify_registration(name):
    manifest = read(name)
    assert manifest["limits"] == V2_LIMITS
    for filename, expected in manifest["sha256"].items():
        assert digest(ROOT / filename) == expected, f"Frozen file changed: {filename}"
    immutable_v1()
    return manifest


def rows_for(split):
    payload = read("dataset.json")
    return [r for r in payload["rows"] if r["split"] == split]


def margin(logits):
    return np.asarray(logits)[..., 0] - np.asarray(logits)[..., 1]


def partner_indices(rows, field="opposite_id"):
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    return np.array([by_id[r[field]] for r in rows])


def make_probes(rows, spec, donors=None, pattern=None, values=None, residual=None):
    output_map = read("dataset.json")["output_ids"]
    return [
        V2Probe(
            text=r["text"],
            token_ids=tuple(r["token_ids"]),
            output_ids=tuple(output_map["vocabulary" if r["split"] == "vocabulary" else "primary"]),
            spec=spec,
            donors={h: v[i] for h, v in (donors or {}).items()},
            pattern_donor=pattern[i, : len(r["token_ids"])] if pattern is not None else None,
            values_donor=values[i, : len(r["token_ids"])] if values is not None else None,
            color_positions=tuple(r["color_positions"]),
            residual_donor=residual[i] if residual is not None else None,
        )
        for i, r in enumerate(rows)
    ]


async def capture(worker, rows):
    result = await worker.map(make_probes(rows, V2Spec(LAYERS, paths=True)))
    width = max(len(r["token_ids"]) for r in rows)
    return {
        "logits": np.array([r.logits for r in result]),
        "heads": np.array([[r.heads[h] for h in HEADS] for r in result], dtype=np.float32),
        "residual": np.array([r.residual for r in result], dtype=np.float32),
        "pattern": np.array(
            [np.pad(r.pattern, (0, width - len(r.pattern))) for r in result], dtype=np.float32
        ),
        "values": np.array(
            [np.pad(r.values, ((0, width - len(r.values)), (0, 0))) for r in result],
            dtype=np.float32,
        ),
        "reconstructed": np.array([r.reconstructed for r in result], dtype=np.float32),
        "path_error": np.array([r.path_error for r in result]),
        "lengths": np.array([len(r["token_ids"]) for r in rows]),
    }


def action(head, mode="replace", alpha=1.0):
    layer, h = map(int, head.split(":"))
    return HeadAction(layer, h, mode, alpha)


async def patch(worker, rows, base, heads, mode="opposite", means=None, alpha=1.0, late=None):
    partners = partner_indices(rows, "same_answer_id" if mode == "same" else "opposite_id")
    donors, actions = {}, []
    for h in heads:
        vector = base["heads"][:, HEADS.index(h)]
        if mode in {"mean", "scale"}:
            donors[h] = np.tile(means[h], (len(rows), 1))
            actions.append(action(h, mode, alpha))
        else:
            donors[h] = vector if mode == "sham" else vector[partners]
            actions.append(action(h))
    if late is not None:
        donors[late] = np.tile(means[late], (len(rows), 1))
        actions.append(action(late, "mean"))
    result = await worker.map(make_probes(rows, V2Spec(actions=tuple(actions)), donors=donors))
    return np.array([r.logits for r in result])


async def instrumentation(worker, rows, base):
    donor = base["residual"][partner_indices(rows)]
    result = await worker.map(make_probes(rows, V2Spec(patch_residual=True), residual=donor))
    residual = np.array([r.logits for r in result])
    sham = await patch(worker, rows, base, [REFERENCE], "sham")
    raw = {"residual": residual, "sham": sham}
    stats = {
        "residual_max": float(np.max(np.abs(residual - base["logits"][partner_indices(rows)]))),
        "sham_max": float(np.max(np.abs(margin(sham) - margin(base["logits"])))),
        "path_reconstruction_max": float(base["path_error"].max()),
    }
    stats["passed"] = (
        stats["residual_max"] <= V2_LIMITS["residual_max"]
        and stats["sham_max"] == 0
        and stats["path_reconstruction_max"] <= V2_LIMITS["path_reconstruction_max"]
    )
    return stats, raw


def rescue_key(candidate):
    s = candidate["stats"]
    return (-s["after"]["correct"], -s["median_gain"], candidate["id"])


async def discover(worker):
    rows = rows_for("discovery")
    labels = np.array([r["label"] for r in rows])
    base = await capture(worker, rows)
    arrays("calibration-baseline.npz", **base)
    controls, raw = await instrumentation(worker, rows, base)
    arrays("calibration-controls.npz", **raw)
    task = task_stats(rows, margin(base["logits"]))
    print("Discovery task", task["correct"], "/", task["total"], "controls", controls, flush=True)
    means = {h: base["heads"][:, i].mean(0).tolist() for i, h in enumerate(HEADS)}
    readouts = {h: fit_readout(base["heads"][:, i], labels) for i, h in enumerate(HEADS)}
    screen, raw_screen = [], []
    for i, h in enumerate(SCREEN):
        logits = await patch(worker, rows, base, [h])
        raw_screen.append(logits)
        stats = effect_stats(rows, margin(base["logits"]), margin(logits))
        screen.append(
            {"head": h, "cv": cross_validate(base["heads"][:, i], rows), "median": stats["median"]}
        )
        if (i + 1) % 16 == 0:
            print(f"Dense screen {i + 1}/128", flush=True)
    arrays("discovery-screen.npz", logits=np.array(raw_screen))
    eligible = [s for s in screen if s["cv"] >= V2_LIMITS["discovery_cv"]]
    eligible.sort(key=lambda s: (-s["median"], *map(int, s["head"].split(":"))))
    best = eligible[0]["head"] if eligible else None
    coalition, greedy = [REFERENCE], []
    for size in (2, 3):
        candidates = [h for h in SCREEN if h not in coalition]
        measured, scores = [], []
        for j, h in enumerate(candidates):
            logits = await patch(worker, rows, base, coalition + [h])
            measured.append(logits)
            scores.append(effect_stats(rows, margin(base["logits"]), margin(logits))["median"])
            if (j + 1) % 32 == 0:
                print(f"Greedy size {size}: {j + 1}/{len(candidates)}", flush=True)
        chosen_i = min(
            range(len(candidates)), key=lambda j: (-scores[j], *map(int, candidates[j].split(":")))
        )
        coalition.append(candidates[chosen_i])
        arrays(f"discovery-greedy-{size}.npz", logits=np.array(measured))
        greedy.append(
            {
                "candidate_heads": candidates,
                "chosen": candidates[chosen_i],
                "coalition": coalition.copy(),
                "scores": scores,
            }
        )
    late_screen, late_arrays = [], []
    for h in HEADS[128:]:
        logits = await patch(worker, rows, base, [h], "mean", means)
        late_arrays.append(logits)
        late_screen.append(
            {"head": h, "stats": rescue_stats_v2(rows, margin(base["logits"]), margin(logits))}
        )
    arrays("discovery-late.npz", logits=np.array(late_arrays))
    late_ranked = sorted(
        late_screen,
        key=lambda c: (
            -c["stats"]["after"]["correct"],
            c["stats"]["regressions"],
            -c["stats"]["median_gain"],
            int(c["head"].split(":")[1]),
        ),
    )
    interferer = late_ranked[0]["head"]
    rules = [{"id": "late_only", "alpha": 1.0, "late": True}]
    rules += [
        {"id": f"center_{alpha}_late_{int(use_late)}", "alpha": alpha, "late": use_late}
        for alpha in (1.5, 2.0)
        for use_late in (False, True)
    ]
    rescue_arrays, rescue_candidates = [], []
    for rule in rules:
        logits = await patch(
            worker,
            rows,
            base,
            coalition if rule["alpha"] != 1 else [],
            "scale",
            means,
            rule["alpha"],
            interferer if rule["late"] else None,
        )
        rescue_arrays.append(logits)
        rescue_candidates.append(
            rule | {"stats": rescue_stats_v2(rows, margin(base["logits"]), margin(logits))}
        )
    arrays("discovery-rescue.npz", logits=np.array(rescue_arrays))
    safe = [r for r in rescue_candidates if r["stats"]["regressions"] == 0]
    rule = (
        {k: v for k, v in min(safe, key=rescue_key).items() if k != "stats"}
        if safe
        else {"id": "identity", "alpha": 1.0, "late": False}
    )
    selection = {
        "best_head": best,
        "coalition": coalition,
        "interferer": interferer,
        "rescue": rule,
    }
    save(
        "discovery.json",
        {
            "registration_sha256": digest(OUT / "discovery-registration.json"),
            "task": task,
            "controls": controls,
            "screen": screen,
            "readouts": readouts,
            "mean_vectors": means,
            "greedy": greedy,
            "late_screen": late_screen,
            "rescue_candidates": rescue_candidates,
            "selection": selection,
        },
    )
    print("Frozen selection:", json.dumps(selection), flush=True)


async def decomposition(worker, rows, base):
    opposite = partner_indices(rows)
    raw = {}
    for mode in MODES:
        spec = V2Spec(layers=(19,), actions=(action(REFERENCE, mode),))
        for identity in (False, True):
            idx = np.arange(len(rows)) if identity else opposite
            result = await worker.map(
                make_probes(rows, spec, pattern=base["pattern"][idx], values=base["values"][idx])
            )
            suffix = "_self" if identity else ""
            raw[f"{mode}{suffix}_logits"] = np.array([r.logits for r in result])
            raw[f"{mode}{suffix}_delta"] = np.array([r.applied_deltas[REFERENCE] for r in result])
            if not identity:
                raw[f"{mode}_applied"] = np.array([r.heads[REFERENCE] for r in result])
    return raw


def evaluate_block(rows, base, control, solos, coalition, rescue, decomp, selection, readouts):
    bm = margin(base["logits"])
    ef = {h: effect_stats(rows, bm, margin(a["opposite"])) for h, a in solos.items()}
    readout = {
        h: readout_metrics_v2(rows, base["heads"][:, HEADS.index(h)], readouts[h]) for h in solos
    }
    joint = effect_stats(rows, bm, margin(coalition["opposite"]))
    y = np.array([r["label"] for r in rows])
    constituent = np.array(
        [y * (bm - margin(solos[h]["opposite"])) for h in selection["coalition"]]
    )
    joint_effect = y * (bm - margin(coalition["opposite"]))
    gain = float(np.median(joint_effect - np.max(constituent, axis=0)))
    interaction = joint_effect - constituent.sum(0)
    synergy_groups = grouped_values(interaction, rows)
    joint_checks = {
        "direction": joint["direction_rate"] >= V2_LIMITS["coalition_direction"],
        "median": joint["median"] >= V2_LIMITS["coalition_median"],
        "fraction": joint["fraction"] >= V2_LIMITS["coalition_fraction"],
        "gain_over_best_solo": gain >= V2_LIMITS["coalition_gain_over_best_solo"],
        "significance": joint["group_sign_p"] <= V2_LIMITS["p"],
        "self_exact": bool(np.array_equal(margin(coalition["sham"]), bm)),
    }
    path_summary = decomposition_metrics_v2(
        rows, bm, {m: margin(decomp[f"{m}_logits"]) for m in MODES}
    )
    source_closure = float(
        np.max(
            np.abs(
                decomp["source_color_delta"] + decomp["source_other_delta"] - decomp["qkov_delta"]
            )
        )
    )
    qkov_closure = float(
        np.max(
            np.abs(
                decomp["ov_all_delta"]
                + decomp["qk_delta"]
                + decomp["interaction_delta"]
                - decomp["qkov_delta"]
            )
        )
    )
    full_error = float(
        np.max(np.abs(margin(decomp["qkov_logits"]) - margin(solos[REFERENCE]["opposite"])))
    )
    self_error = max(float(np.max(np.abs(margin(decomp[f"{m}_self_logits"]) - bm))) for m in MODES)
    path_checks = {
        "reconstruction": float(base["path_error"].max()) <= V2_LIMITS["path_reconstruction_max"],
        "source_closure": source_closure <= V2_LIMITS["delta_closure_max"],
        "qkov_closure": qkov_closure <= V2_LIMITS["delta_closure_max"],
        "full_margin_closure": full_error <= V2_LIMITS["path_full_margin_error_max"],
        "self_exact": self_error == 0,
        "classifiable": path_summary["classifiable"],
    }
    rescue_summary = rescue_stats_v2(rows, bm, margin(rescue["logits"]))
    transfer_checks = {
        h: {
            "direction": ef[h]["direction_rate"] >= V2_LIMITS["transfer_direction"],
            "median": ef[h]["median"] >= V2_LIMITS["transfer_median"],
            "significance": ef[h]["group_sign_p"] <= V2_LIMITS["p"],
        }
        for h in [REFERENCE]
    }
    transfer_checks["coalition"] = {
        "direction": joint["direction_rate"] >= V2_LIMITS["transfer_direction"],
        "median": joint["median"] >= V2_LIMITS["transfer_median"],
        "significance": joint["group_sign_p"] <= V2_LIMITS["p"],
    }
    attention = {"relevant": [], "other_value": [], "keys": [], "delimiters": []}
    for i, r in enumerate(rows):
        attention["relevant"].append(
            float(base["pattern"][i, r["color_positions"][r["query_record"]]])
        )
        attention["other_value"].append(
            float(base["pattern"][i, r["color_positions"][1 - r["query_record"]]])
        )
        attention["keys"].append(
            float(base["pattern"][i, np.array(sum(r["key_positions"], []), dtype=int)].sum())
        )
        attention["delimiters"].append(float(base["pattern"][i, r["delimiter_positions"]].sum()))
    controls_pass = (
        control["passed"]
        and all(np.array_equal(margin(v["sham"]), bm) for v in solos.values())
        and joint_checks["self_exact"]
        and self_error == 0
    )
    return {
        "task": task_stats(rows, bm),
        "controls": control | {"all_self_controls_passed": bool(controls_pass)},
        "single_effects": ef,
        "readouts": readout,
        "decomposition": path_summary
        | {
            "checks": path_checks,
            "passed": all(path_checks.values()),
            "source_closure": source_closure,
            "qkov_closure": qkov_closure,
            "full_margin_error": full_error,
            "self_error": self_error,
            "attention_mass": attention,
        },
        "coalition": {
            "effects": joint,
            "checks": joint_checks,
            "passed": all(joint_checks.values()),
            "gain_over_best_solo": gain,
            "interaction_median": float(np.median(interaction)),
            "interaction_group_direction": float(np.mean(synergy_groups > 0)),
            "synergy_supported": bool(
                np.median(interaction) >= 0.10 and np.mean(synergy_groups > 0) >= 0.80
            ),
        },
        "rescue": rescue_summary,
        "transfer": {
            "checks": transfer_checks,
            "passed": all(all(c.values()) for c in transfer_checks.values()),
        },
    }


async def confirm(worker, rep):
    discovery = read("discovery.json")
    selection = discovery["selection"]
    heads = sorted(
        set(
            selection["coalition"]
            + [REFERENCE]
            + ([selection["best_head"]] if selection["best_head"] else [])
        ),
        key=lambda h: tuple(map(int, h.split(":"))),
    )
    summaries = {}
    for split in ("primary", "vocabulary"):
        rows = rows_for(split)
        prefix = f"confirmation-{rep}-{split}"
        base = await capture(worker, rows)
        arrays(prefix + "-baseline.npz", **base)
        controls, raw = await instrumentation(worker, rows, base)
        arrays(prefix + "-controls.npz", **raw)
        solos = {}
        for head in heads:
            solos[head] = {
                m: await patch(worker, rows, base, [head], m, discovery["mean_vectors"])
                for m in ("opposite", "sham", "mean", "same")
            }
            arrays(prefix + f"-solo-{head.replace(':', '-')}.npz", **solos[head])
        coalition = {
            m: await patch(worker, rows, base, selection["coalition"], m, discovery["mean_vectors"])
            for m in ("opposite", "sham", "mean", "same")
        }
        arrays(prefix + "-coalition.npz", **coalition)
        rule = selection["rescue"]
        rescue = {
            "logits": await patch(
                worker,
                rows,
                base,
                selection["coalition"] if rule["alpha"] != 1 else [],
                "scale",
                discovery["mean_vectors"],
                rule["alpha"],
                selection["interferer"] if rule["late"] else None,
            )
        }
        arrays(prefix + "-rescue.npz", **rescue)
        decomp = await decomposition(worker, rows, base)
        arrays(prefix + "-decomposition.npz", **decomp)
        summaries[split] = evaluate_block(
            rows, base, controls, solos, coalition, rescue, decomp, selection, discovery["readouts"]
        )
        s = summaries[split]
        print(
            f"Run{rep} {split}: task {s['task']['correct']}/{len(rows)}, source {s['decomposition']['source_classification']}, routing {s['decomposition']['routing_classification']}, coalition fraction {s['coalition']['effects']['fraction']:.3f}, rescue {s['rescue']['errors_fixed']} fixed/{s['rescue']['regressions']} regressed",
            flush=True,
        )
    save(
        f"confirmation-{rep}.json",
        {
            "manifest_sha256": digest(OUT / "manifest.json"),
            "selection_sha256": digest(OUT / "discovery.json"),
            "blocks": summaries,
        },
    )


async def execute(stage):
    verify_registration("discovery-registration.json" if stage == "discover" else "manifest.json")
    if stage == "confirm2":
        assert (OUT / "confirmation-1.json").exists()
    save(
        stage + "-started.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "registration_sha256": digest(
                OUT / ("discovery-registration.json" if stage == "discover" else "manifest.json")
            ),
        },
    )
    lock = OUT / "running.lock"
    with lock.open("x") as f:
        f.write(stage)
    try:
        model = ModelConfig(revision=REVISION, local_files_only=True)
        worker_config = WorkerConfig(batch_size=64)
        harness = await asyncio.to_thread(V2Harness.load, model, worker_config)
        async with MPSBatchWorker(harness, worker_config) as worker:
            if stage == "discover":
                await discover(worker)
            else:
                await confirm(worker, int(stage[-1]))
            print("Microbatches:", worker.batches, flush=True)
    finally:
        lock.unlink()


async def main(stage):
    if stage == "prepare":
        prepare()
    elif stage == "register":
        registration()
    elif stage == "freeze":
        registration(final=True)
    else:
        await execute(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "register", "discover", "freeze", "confirm1", "confirm2"]
    )
    asyncio.run(main(parser.parse_args().stage))
