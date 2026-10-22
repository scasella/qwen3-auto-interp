import asyncio
import json
import sys
from pathlib import Path

import pytest

from src.agents.client import (
    AgentConfigurationError,
    AgentError,
    OutputContractError,
    ScientistClient,
    TokenBucket,
)
from src.agents.prompts import SYSTEM_PROMPTS
from src.config import CodexConfig
from src.schemas.contracts import PIReport


@pytest.fixture
def fake_codex(tmp_path):
    path = tmp_path / "fake-codex"
    path.write_text(
        f"#!{sys.executable}\n"
        + """
import json, pathlib, sys, time
args = sys.argv
payload = json.load(sys.stdin)
mode = payload.get('mode', 'ok')
if mode == 'auth':
    print(json.dumps({'type': 'turn.failed', 'error': {'message': 'Unauthorized 401'}}))
    sys.exit(1)
if mode == 'slow':
    time.sleep(30)
if mode == 'large':
    print('x' * 5000)
    sys.exit(0)
output = pathlib.Path(args[args.index('--output-last-message')+1])
if mode != 'missing':
    output.write_text('not-json' if mode == 'invalid' else json.dumps({
        'clusters': [], 'prioritized_units': [], 'rationale': 'No evidence'}))
print(json.dumps({'type': 'turn.completed', 'usage': {
    'input_tokens': 2000, 'cached_input_tokens': 1500, 'output_tokens': 100}}))
"""
    )
    path.chmod(0o755)
    return str(path)


async def test_structured_settings_prefix_and_usage(fake_codex):
    client = ScientistClient(CodexConfig(executable=fake_codex))
    try:
        command = client.command("pi", Path("schema.json"), Path("out.json"))
        assert command[1] == "exec"
        assert command[command.index("--model") + 1] == "gpt-6-luna"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert 'model_reasoning_effort="xhigh"' in command
        assert "--output-schema" in command and "--json" in command
        assert "--ignore-user-config" in command and "--ephemeral" in command
        assert command[-1] == "-"
        assert not any("temperature" in c for c in command)
        developer = next(c for c in command if c.startswith("developer_instructions="))
        assert json.loads(developer.split("=", 1)[1]) == SYSTEM_PROMPTS["pi"]
        result = await client.ask("pi", PIReport, {"units": []})
        assert result.clusters == []
        assert client.metrics.cache_hit_percent == 75
        assert client.metrics.calls == 1
        assert all(len(prompt.split()) > 1024 for prompt in SYSTEM_PROMPTS.values())
    finally:
        await client.close()
    assert not client.root.exists()


async def test_auth_failure_stops_following_requests(fake_codex):
    client = ScientistClient(CodexConfig(executable=fake_codex))
    try:
        with pytest.raises(AgentConfigurationError, match="401"):
            await client.ask("pi", PIReport, {"mode": "auth"})
        with pytest.raises(AgentConfigurationError, match="401"):
            await client.ask("pi", PIReport, {})
        assert client.metrics.retries == 0
    finally:
        await client.close()


@pytest.mark.parametrize("mode", ["missing", "invalid", "large"])
async def test_bad_output_never_becomes_evidence(fake_codex, mode):
    client = ScientistClient(
        CodexConfig(executable=fake_codex, max_attempts=1, max_capture_bytes=1024)
    )
    try:
        with pytest.raises(AgentError):
            await client.ask("pi", PIReport, {"mode": mode})
        assert not client._processes
    finally:
        await client.close()


async def test_timeout_and_cancellation_reap_children(fake_codex):
    client = ScientistClient(
        CodexConfig(executable=fake_codex, max_attempts=1, timeout_seconds=0.1)
    )
    try:
        with pytest.raises(AgentError):
            await client.ask("pi", PIReport, {"mode": "slow"})
        assert not client._processes
        client.config.timeout_seconds = 30
        task = asyncio.create_task(client.ask("pi", PIReport, {"mode": "slow"}))
        for _ in range(100):
            if client.active_processes:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not client._processes
    finally:
        await client.close()


async def test_subprocess_ceiling(fake_codex):
    client = ScientistClient(CodexConfig(executable=fake_codex, max_processes=2))
    maximum = 0
    jobs = [asyncio.create_task(client.ask("pi", PIReport, {})) for _ in range(6)]
    try:
        while not all(t.done() for t in jobs):
            maximum = max(maximum, client.active_processes)
            await asyncio.sleep(0.001)
        await asyncio.gather(*jobs)
        assert 1 <= maximum <= 2
    finally:
        await client.close()


async def test_tpm_reservations_expire_and_large_request_fails():
    bucket = TokenBucket(100, window=0.01)
    await bucket.acquire(80)
    await asyncio.wait_for(bucket.acquire(80), timeout=0.2)
    assert bucket.reserved == 80
    with pytest.raises(AgentError, match="exceeds TPM"):
        await bucket.acquire(101)


async def test_invalid_structure_gets_repair_feedback(fake_codex, monkeypatch):
    client = ScientistClient(CodexConfig(executable=fake_codex, max_attempts=2))
    prompts = []
    original_invoke = client._invoke

    async def record(command, prompt):
        prompts.append(json.loads(prompt))
        return await original_invoke(command, prompt)

    async def no_backoff(_):
        return None

    monkeypatch.setattr(client, "_invoke", record)
    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    try:
        result = await client.ask("pi", PIReport, {"mode": "invalid"})
        assert result.clusters == []
        assert "repair" in prompts[1]
        assert prompts[1]["request"] == {"mode": "invalid"}
        assert client.metrics.retries == 1
        assert client.metrics.input_tokens == 4000
    finally:
        await client.close()


async def test_contract_repair_precedes_acceptance(fake_codex, monkeypatch):
    client = ScientistClient(CodexConfig(executable=fake_codex, max_attempts=2))
    prompts = []
    original_invoke = client._invoke

    async def record(command, prompt):
        request = json.loads(prompt)
        prompts.append(request)
        result = await original_invoke(command, prompt)
        if "repair" in request:
            output = Path(command[command.index("--output-last-message") + 1])
            data = json.loads(output.read_text())
            data["prioritized_units"] = ["mlp:3:1321"]
            output.write_text(json.dumps(data))
        return result

    def validate(result):
        if result.prioritized_units != ["mlp:3:1321"]:
            raise OutputContractError("Use exactly the assigned unit mlp:3:1321")

    async def no_backoff(_):
        return None

    monkeypatch.setattr(client, "_invoke", record)
    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    try:
        result = await client.ask("pi", PIReport, {}, validate_result=validate)
        assert result.prioritized_units == ["mlp:3:1321"]
        assert prompts[1]["repair"]["errors"][0]["type"] == "output_contract"
        assert client.metrics.calls == 1 and client.metrics.retries == 1
        assert client.metrics.input_tokens == 4000
        with pytest.raises(AgentError, match="assigned unit"):
            await client.ask(
                "pi",
                PIReport,
                {},
                validate_result=lambda _: validate(
                    PIReport(clusters=[], prioritized_units=[], rationale="Still invalid")
                ),
            )
    finally:
        await client.close()


async def test_async_contract_validation_is_awaited(fake_codex, monkeypatch):
    client = ScientistClient(CodexConfig(executable=fake_codex, max_attempts=2))
    checked = []

    async def validate(result):
        checked.append(result)
        if len(checked) == 1:
            raise OutputContractError("Independent reviewer rejected first output")

    async def no_backoff(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    try:
        await client.ask("pi", PIReport, {}, validate_result=validate)
        assert len(checked) == 2
        assert client.metrics.calls == 1 and client.metrics.retries == 1
    finally:
        await client.close()
