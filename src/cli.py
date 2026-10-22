import argparse
import asyncio
import json
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import networkx as nx
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.agents.client import ScientistClient
from src.agents.prompts import PROMPT_VERSION
from src.config import Settings
from src.harness.indexer import ActivationIndex, calibrate, corpus_stream
from src.harness.mps_queue import MPSBatchWorker
from src.harness.qwen_model import QwenHarness
from src.harness.thinking_diff import thinking_diff
from src.orchestrator.graph_db import GraphDB
from src.orchestrator.swarm import Swarm
from src.schemas.contracts import UnitRef, all_units

console = Console()


def select_units(value, layers=None):
    if value == "all":
        units = all_units()
    else:
        units = [UnitRef.parse(item.strip()) for item in value.split(",")]
    if layers:
        wanted = {int(item) for item in layers.split(",")}
        if not wanted <= set(range(28)):
            raise ValueError("Layers must be in [0, 27]")
        units = [u for u in units if u.layer in wanted]
    if not units or len(set(units)) != len(units):
        raise ValueError("Select at least one unique unit")
    return units


def dashboard(client, worker, graph, run):
    table = Table(title=f"Qwen3 interpretability | run {run}")
    table.add_column("Metric")
    table.add_column("Value")
    metrics = client.metrics
    for name, value in {
        "MPS allocated RAM": f"{worker.harness.memory_bytes() / 2**30:.2f} GiB",
        "Local micro-batches": worker.batches,
        "Active Codex processes": client.active_processes,
        "Queued probes": worker.queue.qsize(),
        "TPM reserved (conservative)": client.bucket.reserved,
        "Measured input / output tokens": f"{metrics.input_tokens} / {metrics.output_tokens}",
        "Cached input share": f"{metrics.cache_hit_percent:.1f}%",
        "Codex retries / failures": f"{metrics.retries} / {metrics.failures}",
        **graph.summary(run)["states"],
        "Verified circuits": graph.summary(run)["circuits"],
    }.items():
        table.add_row(str(name), str(value))
    return table


async def async_main(args, settings):
    if args.command == "export":
        graph = GraphDB(settings.paths.circuits)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(nx.node_link_data(graph.graph(args.run_id)), indent=2)
            )
            console.print(f"Saved {args.output}")
        finally:
            graph.close()
        return 0
    units = select_units(args.units, args.layers)
    index = ActivationIndex(settings.paths.activations, settings.calibration.duckdb_memory_limit)
    graph = None
    client = None
    try:
        if args.command == "run":
            missing = set(units) - set(index.units())
            if missing:
                raise ValueError(
                    f"Calibrate {len(missing)} selected units before running the swarm"
                )
            for unit in units:
                if index.quantiles(unit).count < settings.calibration.min_tokens:
                    raise ValueError(f"Insufficient calibration observations for {unit.key}")
        console.print("Loading Qwen3; all execution will use the single-consumer worker.")
        harness = await asyncio.to_thread(QwenHarness.load, settings.model, settings.worker)
        async with MPSBatchWorker(harness, settings.worker) as worker:
            if args.command == "calibrate":
                await calibrate(
                    worker,
                    index,
                    args.corpus,
                    units,
                    settings,
                    lambda done, total: console.print(f"Calibrated {done}/{total} new units"),
                )
                return 0
            if args.command == "screen":
                prompts = list(corpus_stream(args.corpus, args.prompts))
                scores = await thinking_diff(
                    worker, prompts, units, settings.calibration.units_per_pass
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "scope": "Chat-template sensitivity at the final prompt token; no generated reasoning",
                            "resolved_revision": harness.revision,
                            "scores": scores,
                        },
                        indent=2,
                    )
                )
                console.print(f"Saved {args.output}")
                return 0
            manifest = index.metadata().get("manifest")
            if (
                manifest is None
                or manifest["model"] != settings.model.model_dump()
                or manifest["resolved_revision"] != harness.revision
                or manifest["max_seq_len"] != settings.worker.max_seq_len
            ):
                raise ValueError("Active model settings do not match calibration provenance")
            graph = GraphDB(settings.paths.circuits)
            run = graph.start_run(
                {
                    "calibration": manifest,
                    "settings": settings.model_dump(mode="json"),
                    "prompt_version": PROMPT_VERSION,
                    "units": sorted(u.key for u in units),
                },
                args.resume,
            )
            console.print(f"Run ID: {run}")
            client = ScientistClient(
                settings.codex,
                settings.swarm.max_agent_calls,
                tokenizer_json=harness.tokenizer.backend_tokenizer.to_str(),
            )
            swarm = Swarm(settings, client, worker, index, graph, run)
            task = asyncio.create_task(swarm.run_units(units))
            try:
                display = (
                    nullcontext(None)
                    if args.no_dashboard
                    else Live(
                        dashboard(client, worker, graph, run), console=console, refresh_per_second=1
                    )
                )
                with display as live:
                    while not task.done():
                        await asyncio.wait({task}, timeout=1)
                        if live:
                            live.update(dashboard(client, worker, graph, run))
                result = await task
                graph.event(run, "usage", json.dumps(asdict(client.metrics)))
                graph.event(run, "completion", json.dumps(result))
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            console.print_json(data=result)
            return int(bool(result["errors"]))
    finally:
        if client:
            await client.close()
        if graph:
            graph.close()
        index.close()


def parser():
    result = argparse.ArgumentParser(description="Falsification-first Qwen3 interpretability")
    result.add_argument("--config", type=Path, default=Path("config.yaml"))
    result.add_argument("--device", choices=["cpu", "mps"])
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Check local hardware and configuration without loading weights"
    )
    for name in ("calibrate", "run", "screen"):
        command = commands.add_parser(name)
        command.add_argument(
            "--units", default="mlp:0:0", help="kind:layer:index comma-list, or all"
        )
        command.add_argument(
            "--layers", help="Restrict an all-unit sweep to comma-separated layers"
        )
        if name != "run":
            command.add_argument("--corpus", type=Path, required=True)
        if name == "screen":
            command.add_argument("--prompts", type=int, default=32)
            command.add_argument("--output", type=Path, default=Path("data/thinking_diff.json"))
        if name == "run":
            command.add_argument("--pipelines", type=int, choices=range(1, 65), metavar="1..64")
            command.add_argument(
                "--codex-processes", type=int, choices=range(1, 33), metavar="1..32"
            )
            command.add_argument("--resume", help="Resume a run ID without resetting strikes")
            command.add_argument("--no-dashboard", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path, default=Path("data/circuit_graph.json"))
    return result


def main():
    args = parser().parse_args()
    try:
        settings = Settings.load(args.config)
        if args.device:
            settings.model.device = args.device
            if args.device == "cpu":
                settings.model.dtype = "float32"
        if getattr(args, "pipelines", None):
            settings.swarm.pipelines = args.pipelines
        if getattr(args, "codex_processes", None):
            settings.codex.max_processes = args.codex_processes
        if args.command == "doctor":
            import torch

            console.print_json(
                data={
                    "mps_available": torch.backends.mps.is_available(),
                    "torch": torch.__version__,
                    "model": settings.model.model_dump(),
                    "scientist_model": settings.codex.model,
                    "codex_executable": shutil.which(settings.codex.executable),
                    "codex_max_processes": settings.codex.max_processes,
                    "unit_count": len(all_units()),
                }
            )
            return
        code = asyncio.run(async_main(args, settings))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        code = 1
    except KeyboardInterrupt:
        console.print("Interrupted. Measured attempts are saved; resume with the printed run ID.")
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
