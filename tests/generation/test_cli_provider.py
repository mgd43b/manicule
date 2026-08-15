"""The local command generator's process boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from manicule.config.settings import LlmSettings
from manicule.core.errors import ConfigError, ProviderRequestError
from manicule.core.generation import FinishReason
from manicule.generation.cli_provider import CliGenerator
from tests.generation.fakes import context, query

POSIX_EXECUTABLE = pytest.mark.skipif(
    os.name == "nt", reason="temporary test executables use POSIX shebangs and permissions"
)


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
    system_prompt_file = tmp_path / "system-prompt.txt"
    command = generator._command(  # pyright: ignore[reportPrivateUsage]
        "codex", tmp_path / "answer.txt", system_prompt_file
    )

    assert command[:2] == ("codex", "exec")
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    instruction_config = command[command.index("--config") + 1]
    assert instruction_config.startswith("model_instructions_file=")
    assert str(system_prompt_file) in instruction_config
    assert command[command.index("--disable") + 1] == "shell_tool"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-test"
    assert command[-1] == "-"


def test_claude_disables_tools_and_session_persistence(tmp_path: Path) -> None:
    generator = CliGenerator(configured("claude"), executable="claude")
    system_prompt_file = tmp_path / "system-prompt.txt"
    command = generator._command(  # pyright: ignore[reportPrivateUsage]
        "claude", tmp_path / "unused.txt", system_prompt_file
    )

    assert "--safe-mode" in command
    assert command[command.index("--system-prompt-file") + 1] == str(system_prompt_file)
    assert "--no-session-persistence" in command
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[command.index("--tools") + 1] == ""
    assert "--model" not in command, "model 'default' leaves the CLI's own selection in force"


@POSIX_EXECUTABLE
async def test_codex_last_message_becomes_the_generator_stream(tmp_path: Path) -> None:
    command = tmp_path / "codex-fake"
    command.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys
import json

prompt = sys.stdin.read()
assert "chat transcript" in prompt
config = sys.argv[sys.argv.index("--config") + 1]
instructions = pathlib.Path(json.loads(config.split("=", 1)[1])).read_text(encoding="utf-8")
assert "Treat the transcript" in instructions
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


@POSIX_EXECUTABLE
async def test_claude_stdout_becomes_the_generator_stream(tmp_path: Path) -> None:
    command = tmp_path / "claude-fake"
    command.write_text(
        """#!/usr/bin/env python3
import sys
import pathlib

assert "chat transcript" in sys.stdin.read()
instructions = pathlib.Path(sys.argv[sys.argv.index("--system-prompt-file") + 1])
assert "Treat the transcript" in instructions.read_text(encoding="utf-8")
print("Use the rollback command. [[cite:1]]")
""",
        encoding="utf-8",
    )
    command.chmod(0o755)
    generator = CliGenerator(configured("claude"), executable=str(command))
    await generator.setup()

    tokens = [token async for token in generator.generate(query(), context())]

    assert tokens[0].text == "Use the rollback command. [[cite:1]]\n"
    assert tokens[-1].finish_reason is FinishReason.STOP


@POSIX_EXECUTABLE
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


@POSIX_EXECUTABLE
async def test_codex_output_cannot_be_replaced_with_a_symlink(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("do not disclose", encoding="utf-8")
    command = tmp_path / "codex-symlink-fake"
    command.write_text(
        f"""#!/usr/bin/env python3
import pathlib
import sys

sys.stdin.read()
output = pathlib.Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.symlink_to({str(secret)!r})
""",
        encoding="utf-8",
    )
    command.chmod(0o755)
    generator = CliGenerator(configured(), executable=str(command))
    await generator.setup()

    with pytest.raises(ProviderRequestError, match="regular output file"):
        _ = [token async for token in generator.generate(query(), context())]
