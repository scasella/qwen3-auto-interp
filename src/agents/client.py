"""Structured scientist calls through authenticated, noninteractive `codex exec`."""

import asyncio
import hashlib
import inspect
import json
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from src.agents.prompts import SYSTEM_PROMPTS
from src.config import CodexConfig


class AgentError(RuntimeError):
    pass


class AgentConfigurationError(AgentError):
    """An installation/auth/model error that must stop new pipeline submissions."""


class OutputContractError(ValueError):
    """A structurally valid response violates its assigned evidence contract."""


class TokenBucket:
    """Local rolling-window throttle, not an assertion about the account's quota."""

    def __init__(self, capacity: int, window=60.0):
        self.capacity, self.window = capacity, window
        self.entries = deque()
        self.lock = asyncio.Lock()

    def _expire(self):
        now = time.monotonic()
        while self.entries and self.entries[0][0] <= now - self.window:
            self.entries.popleft()

    @property
    def reserved(self):
        self._expire()
        return sum(n for _, n in self.entries)

    async def acquire(self, tokens: int):
        if tokens > self.capacity:
            raise AgentError(f"Request reservation {tokens} exceeds TPM budget {self.capacity}")
        while True:
            async with self.lock:
                self._expire()
                if self.reserved + tokens <= self.capacity:
                    self.entries.append((time.monotonic(), tokens))
                    return
                delay = self.entries[0][0] + self.window - time.monotonic()
            await asyncio.sleep(max(0.001, min(delay, 60)))

    def record_overage(self, tokens):
        if tokens > 0:
            self.entries.append((time.monotonic(), tokens))


@dataclass
class UsageMetrics:
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    failures: int = 0
    tokenizer_calls: int = 0

    @property
    def cache_hit_percent(self):
        return 100 * self.cached_tokens / self.input_tokens if self.input_tokens else 0.0


def parse_events(stdout: bytes):
    events = []
    for line in stdout.decode("utf-8").splitlines():
        if line.strip():
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise AgentError("Codex stdout was not JSONL") from exc
            if not isinstance(event, dict):
                raise AgentError("Invalid Codex event")
            events.append(event)
    return events


class ScientistClient:
    def __init__(self, config: CodexConfig, concurrency=256, *, tokenizer_json=None):
        self.config = config
        executable = shutil.which(config.executable)
        if executable is None:
            raise AgentConfigurationError("Codex CLI not found. Install it and run `codex login`.")
        self.executable = executable
        self.limit = asyncio.Semaphore(min(concurrency, config.max_processes))
        self.bucket = TokenBucket(config.tokens_per_minute)
        self.metrics = UsageMetrics()
        self._temp = tempfile.TemporaryDirectory(prefix="qwen-scientists-")
        self.root = Path(self._temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.tokenizer_path = self.root / "qwen-tokenizer.json" if tokenizer_json else None
        if self.tokenizer_path is not None:
            self.tokenizer_path.write_text(tokenizer_json)
        self._processes = set()
        self._closed = False
        self._fatal = None

    @property
    def active_processes(self):
        return len(self._processes)

    async def close(self):
        self._closed = True
        await asyncio.gather(*(self._stop(p) for p in tuple(self._processes)))
        self._temp.cleanup()

    async def _stop(self, process):
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    def command(self, role, schema_path, output_path):
        args = [
            self.executable,
            "exec",
            "--model",
            self.config.model,
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--json",
            "--color",
            "never",
            "--cd",
            str(self.workspace),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
        ]
        overrides = {
            "model_reasoning_effort": getattr(self.config, role).effort,
            "developer_instructions": SYSTEM_PROMPTS[role],
            "approval_policy": "never",
            "web_search": "disabled",
            "features.shell_tool": False,
            "features.apps": False,
            "features.multi_agent": False,
            "features.shell_snapshot": False,
        }
        if role == "falsifier" and self.tokenizer_path is not None:
            overrides.update(
                {
                    "mcp_servers.qwen_tokenizer.command": sys.executable,
                    "mcp_servers.qwen_tokenizer.args": [
                        str(Path(__file__).with_name("tokenizer_mcp.py")),
                        str(self.tokenizer_path),
                    ],
                    "mcp_servers.qwen_tokenizer.required": True,
                    "mcp_servers.qwen_tokenizer.enabled_tools": ["check_cases"],
                    "mcp_servers.qwen_tokenizer.default_tools_approval_mode": "approve",
                }
            )
            overrides["developer_instructions"] += (
                "\nThe sole exception to the no-tools rule is the supplied read-only qwen_tokenizer.check_cases tool. It performs CPU tokenization, never model inference. You MUST check all requested cases (nine for a batch) with the supplied minimum lengths, repair token-count/target errors by coherent context edits, and recheck until errors is empty BEFORE returning the final suite. Do not guess tokenizer counts. This tool grants no permission to access files, shell, network, activations, or other tools."
            )
        for key, value in overrides.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        return [*args, "-"]

    async def _read(self, stream):
        captured = bytearray()
        while chunk := await stream.read(65536):
            captured.extend(chunk)
            if len(captured) > self.config.max_capture_bytes:
                raise AgentError("Codex output exceeded the configured capture limit")
        return bytes(captured)

    async def _invoke(self, command, prompt):
        env = os.environ.copy()
        env.pop("CODEX_THREAD_ID", None)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self._processes.add(process)

        async def write_input():
            try:
                process.stdin.write(prompt.encode())
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

        jobs = [
            asyncio.create_task(self._read(process.stdout)),
            asyncio.create_task(self._read(process.stderr)),
            asyncio.create_task(write_input()),
            asyncio.create_task(process.wait()),
        ]
        try:
            stdout, stderr, _, code = await asyncio.wait_for(
                asyncio.gather(*jobs), self.config.timeout_seconds
            )
            return code, stdout, stderr
        finally:
            await self._stop(process)
            for task in jobs:
                task.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            self._processes.discard(process)

    async def ask(self, role, schema, payload, *, validate_result=None):
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        encoded_schema = json.dumps(schema.model_json_schema(), sort_keys=True)
        last = None
        for attempt in range(self.config.max_attempts):
            reservation = len((SYSTEM_PROMPTS[role] + prompt + encoded_schema).encode())
            reservation += self.config.output_token_reserve + self.config.harness_token_reserve
            if self._fatal:
                raise self._fatal
            if self._closed:
                raise AgentError("Scientist client is closed")
            schema_hash = hashlib.sha256(encoded_schema.encode()).hexdigest()[:16]
            schema_path = self.root / f"{schema.__name__}-{schema_hash}.schema.json"
            if not schema_path.exists():
                schema_path.write_text(encoded_schema)
            output_path = self.root / f"{uuid4().hex}.json"
            limits_path = output_path.with_suffix(".limits.json")
            command = self.command(role, schema_path, output_path)
            if role == "falsifier" and self.tokenizer_path is not None:
                limits_path.write_text(
                    json.dumps(
                        payload.get("minimum_tokens", {"positive": 1, "inhibitory": 1})
                        | {"expected_cases": payload.get("requested_cases", 45)}
                    )
                )
                command[-1:-1] = [
                    "-c",
                    "mcp_servers.qwen_tokenizer.args="
                    + json.dumps(
                        [
                            str(Path(__file__).with_name("tokenizer_mcp.py")),
                            str(self.tokenizer_path),
                            str(limits_path),
                        ]
                    ),
                ]
            try:
                async with self.limit:
                    if self._fatal:
                        raise self._fatal
                    if self._closed:
                        raise AgentError("Scientist client is closed")
                    await self.bucket.acquire(reservation)
                    if self._fatal:
                        raise self._fatal
                    if self._closed:
                        raise AgentError("Scientist client is closed")
                    code, stdout, stderr = await self._invoke(command, prompt)
                events = parse_events(stdout)
                self.metrics.tokenizer_calls += sum(
                    e.get("type") == "item.completed"
                    and e.get("item", {}).get("type") == "mcp_tool_call"
                    and e.get("item", {}).get("server") == "qwen_tokenizer"
                    for e in events
                )
                completed = [e for e in events if e.get("type") == "turn.completed"]
                actual_tokens = 0
                for event in completed:
                    usage = event.get("usage") or {}
                    self.metrics.input_tokens += usage.get("input_tokens", 0)
                    self.metrics.output_tokens += usage.get("output_tokens", 0)
                    self.metrics.cached_tokens += usage.get("cached_input_tokens", 0)
                    actual_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                self.bucket.record_overage(actual_tokens - reservation)
                if (
                    code != 0
                    or not completed
                    or any(e.get("type") == "turn.failed" for e in events)
                ):
                    messages = [
                        str(e.get("error", e.get("message", "")))
                        for e in events
                        if e.get("type") in ("turn.failed", "error")
                    ]
                    detail = ("; ".join(messages) or stderr.decode(errors="replace"))[-2000:]
                    if any(
                        word in detail.lower()
                        for word in (
                            "not supported",
                            "unsupported",
                            "unauthorized",
                            "401",
                            "not logged in",
                            "authentication",
                            "unexpected argument",
                            "unknown variant",
                            "invalid value",
                            "invalid schema",
                            "model_not_found",
                        )
                    ):
                        self._fatal = AgentConfigurationError(
                            f"Codex configuration/authentication error: {detail}"
                        )
                        raise self._fatal
                    raise AgentError(f"codex exec failed (exit {code}): {detail}")
                if not output_path.exists():
                    raise AgentError("Codex completed without the requested structured output file")
                if output_path.stat().st_size > self.config.max_capture_bytes:
                    raise AgentError("Structured output exceeded the capture limit")
                result = schema.model_validate_json(output_path.read_text())
                if validate_result is not None:
                    validation = validate_result(result)
                    if inspect.isawaitable(validation):
                        await validation
                self.metrics.calls += 1
                return result
            except AgentConfigurationError:
                self.metrics.failures += 1
                raise
            except (ValidationError, OutputContractError) as exc:
                last = exc
                # Semantic schema constraints (such as unique strings) need actionable feedback.
                # Keep the immutable role prefix; only the per-call evidence changes.
                prompt = json.dumps(
                    {
                        "request": payload,
                        "repair": {
                            "instruction": "Correct these validation failures and return the complete object.",
                            "errors": exc.errors(
                                include_input=False, include_context=False, include_url=False
                            )
                            if isinstance(exc, ValidationError)
                            else [{"type": "output_contract", "message": str(exc)}],
                            "previous_output": output_path.read_text()[
                                : self.config.max_capture_bytes
                            ],
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except (AgentError, TimeoutError, OSError) as exc:
                last = exc
            finally:
                output_path.unlink(missing_ok=True)
                limits_path.unlink(missing_ok=True)
            if attempt + 1 < self.config.max_attempts:
                self.metrics.retries += 1
                await asyncio.sleep(min(30, 2**attempt) + random.random())
        self.metrics.failures += 1
        raise AgentError(
            f"Codex scientist failed after {self.config.max_attempts} attempts: {last}"
        ) from last
