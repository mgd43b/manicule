"""The local command generator's process boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.config.settings import LlmSettings
from manicule.core.errors import ConfigError
from manicule.core.generation import FinishReason
from manicule.generation.cli_provider import CliGenerator
from tests.generation.fakes import context, query


def configured(provider: str = "codex", **overrides: object) -> LlmSettings:
    values: dict[str, object] = {
        "provider": provider,
        "generator": "cli",
        "model": "default",
        "context_window": 32_768,
    }
    values.update(overrides)
    return LlmSettings.model_validate(values)


async def test_setup_requires_one_of_the_two_supported_commands() -> None:
    generator = CliGenerator(configured("ollama"), executable="/bin/false")

    with pytest.raises(ConfigError, match="claude, codex"):
        await generator.setup()


async def test_setup_requires_the_cli_models_served_window() -> None:
    generator = CliGenerator(
        configured(context_window=None),
        executable="/bin/false",
    )

    with pytest.raises(ConfigError, match=r"llm\.context_window"):
        await generator.setup()


async def test_setup_refuses_an_endpoint_the_command_would_ignore() -> None:
    generator = CliGenerator(
        configured(),
        base_url="http://localhost:1234",
        executable="/bin/false",
    )

    with pytest.raises(ConfigError, match=r"does not use llm\.base_url"):
        await generator.setup()


def test_codex_runs_ephemerally_in_a_read_only_sandbox(tmp_path: Path) -> None:
    generator = CliGenerator(configured(model="gpt-test"), executable="codex")
    command = generator._command(  # pyright: ignore[reportPrivateUsage]
        "codex", tmp_path / "answer.txt", "system authority"
    )

    assert command[:2] == ("codex", "exec")
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "system authority" in command[command.index("--config") + 1]
    assert command[command.index("--disable") + 1] == "shell_tool"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-test"
    assert command[-1] == "-"


def test_claude_disables_tools_and_session_persistence(tmp_path: Path) -> None:
    generator = CliGenerator(configured("claude"), executable="claude")
    command = generator._command(  # pyright: ignore[reportPrivateUsage]
        "claude", tmp_path / "unused.txt", "system authority"
    )

    assert "--safe-mode" in command
    assert command[command.index("--system-prompt") + 1] == "system authority"
    assert "--no-session-persistence" in command
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[command.index("--tools") + 1] == ""
    assert "--model" not in command, "model 'default' leaves the CLI's own selection in force"


async def test_codex_last_message_becomes_the_generator_stream(tmp_path: Path) -> None:
    command = tmp_path / "codex-fake"
    command.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

prompt = sys.stdin.read()
assert "chat transcript" in prompt
output = pathlib.Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.write_text("Use the rollback command. [[cite:1]]", encoding="utf-8")
""",
        encoding="utf-8",
    )
    command.chmod(0o755)
    generator = CliGenerator(configured(), executable=str(command))
    await generator.setup()

    tokens = [token async for token in generator.generate(query(), context())]

    assert tokens[0].text == "Use the rollback command. [[cite:1]]"
    assert tokens[-1].finish_reason is FinishReason.STOP


async def test_claude_stdout_becomes_the_generator_stream(tmp_path: Path) -> None:
    command = tmp_path / "claude-fake"
    command.write_text(
        """#!/usr/bin/env python3
import sys

assert "chat transcript" in sys.stdin.read()
print("Use the rollback command. [[cite:1]]")
""",
        encoding="utf-8",
    )
    command.chmod(0o755)
    generator = CliGenerator(configured("claude"), executable=str(command))
    await generator.setup()

    tokens = [token async for token in generator.generate(query(), context())]

    assert tokens[0].text == "Use the rollback command. [[cite:1]]"
    assert tokens[-1].finish_reason is FinishReason.STOP


async def test_cli_output_is_bounded_and_reports_length(tmp_path: Path) -> None:
    command = tmp_path / "claude-long-fake"
    command.write_text(
        """#!/usr/bin/env python3
import sys

sys.stdin.read()
print("one two three four five six seven eight nine ten")
""",
        encoding="utf-8",
    )
    command.chmod(0o755)
    generator = CliGenerator(
        configured("claude", max_tokens=3),
        executable=str(command),
    )
    await generator.setup()

    tokens = [token async for token in generator.generate(query(), context())]

    assert tokens[0].text != "one two three four five six seven eight nine ten"
    assert tokens[-1].finish_reason is FinishReason.LENGTH
