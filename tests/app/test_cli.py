"""What the command line does that the service does not: parsing, piping and exit status.

Only the adapter's own behaviour is asserted here. Anything about *what an operation means*
belongs to :mod:`tests.app.test_service`, and asserting it twice would make the command line a
second place a rule lives — which is the thing this design exists to prevent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app import results as r
from manicule.app.dispatch import run_op
from manicule.app.results import succeeded
from manicule.app.service import ApplicationService
from manicule.cli.shell import SHELLS, completion_script
from manicule.core.errors import ConfigError
from tests.app.fakes import FakeBackend, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.app.results import Envelope


@pytest.fixture
def service() -> ApplicationService:
    backend = FakeBackend()
    document = make_document(backend.workspace)
    backend.store.add(document, make_chunk(document))
    return ApplicationService(backend)


def _bind(monkeypatch: pytest.MonkeyPatch, service: ApplicationService) -> None:
    """Point the command line at an already-built service.

    One function is substituted — the one that would otherwise read the suite machine's
    configuration and open a database. Everything above it, including argument parsing,
    dispatch, rendering and the exit status, is the real thing.
    """

    async def execute(op: str, call: Any) -> Envelope:
        return await run_op(op, service.workspace, lambda: call(service))

    monkeypatch.setattr(cli, "_execute", execute)


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch, service: ApplicationService) -> ApplicationService:
    """The command line, wired to a service the test controls."""
    _bind(monkeypatch, service)
    return service


def run(argv: Sequence[str], stdin: str | None = None) -> Any:
    return CliRunner().invoke(cli.app, list(argv), input=stdin)


# --- --json ------------------------------------------------------------------------------------


def test_json_output_is_the_only_thing_on_stdout(bound: ApplicationService) -> None:
    """So that a pipe is parseable without anybody filtering a banner out of it."""
    del bound
    result = run(["--json", "document", "list"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is True
    assert parsed["op"] == "document_list"


def test_a_failure_still_emits_a_parseable_envelope_and_exits_non_zero(
    bound: ApplicationService,
) -> None:
    """A script branching on the exit status and a script parsing the output both work."""
    del bound
    result = run(["--json", "document", "get", "missing"])
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is False
    assert parsed["error"]["type"] == "UnknownEntityError"


def test_a_human_readable_failure_goes_to_stderr(bound: ApplicationService) -> None:
    """``manicule search x | jq`` on a failure reads an empty stream rather than prose."""
    del bound
    result = CliRunner().invoke(cli.app, ["document", "get", "missing"])
    assert result.exit_code == 1
    assert result.stdout == ""


# --- stdin -------------------------------------------------------------------------------------


def test_a_query_can_arrive_on_stdin(bound: ApplicationService) -> None:
    """``echo "..." | manicule search`` needs no flag: an absent argument and a pipe agree."""
    del bound
    result = run(["--json", "search"], stdin="retry policy\n")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["data"]["query"] == "retry policy"


def test_an_empty_stdin_is_a_refusal_rather_than_an_empty_query(
    bound: ApplicationService,
) -> None:
    """An empty query would otherwise be a search for nothing, reported as no results."""
    del bound
    result = run(["--json", "search"], stdin="")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["type"] == "ConfigError"


# --- guarded commands ---------------------------------------------------------------------------


def test_reset_index_refuses_without_an_explicit_confirmation(
    bound: ApplicationService,
) -> None:
    """The one irreversible operation, and it takes a flag rather than a prompt.

    A prompt cannot be answered by a script and is skipped by a pipe; a required flag is the
    same refusal whether a person or a cron job typed it.
    """
    del bound
    result = run(["reset-index"])
    assert result.exit_code != 0
    assert "--yes" in result.output


def test_reset_index_runs_when_confirmed(bound: ApplicationService) -> None:
    del bound
    result = run(["--json", "reset-index", "--yes"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["op"] == "reset_index"


def test_backup_refuses_a_request_that_is_both_a_backup_and_a_restore(
    bound: ApplicationService,
) -> None:
    """Two opposite operations in one invocation is a typo, not a plan."""
    del bound
    result = run(["backup", "--output", "/tmp/a", "--restore", "/tmp/b"])  # noqa: S108 - never opened
    assert result.exit_code != 0


def test_backup_needs_somewhere_to_write(bound: ApplicationService) -> None:
    del bound
    result = run(["backup"])
    assert result.exit_code != 0
    assert "--output" in result.output


# --- completion ---------------------------------------------------------------------------------


@pytest.mark.parametrize("shell", SHELLS)
def test_a_completion_script_is_produced_for_every_supported_shell(shell: str) -> None:
    """Generated from the command tree, so a command added tomorrow completes tomorrow."""
    script = completion_script(shell)
    assert "_MANICULE_COMPLETE" in script


def test_an_unsupported_shell_is_named_in_the_refusal() -> None:
    with pytest.raises(ConfigError) as caught:
        completion_script("csh")
    assert "bash" in str(caught.value)


def test_completion_needs_no_configuration_at_all() -> None:
    """It runs before ``init``, on a machine with no config file and no database.

    A completion script that required a working installation would be unavailable in exactly
    the situation somebody is setting one up.
    """
    result = run(["completion", "--shell", "bash"])
    assert result.exit_code == 0
    assert "_MANICULE_COMPLETE" in result.stdout


# --- the version option -------------------------------------------------------------------------


def test_version_is_an_option_rather_than_a_twentieth_command() -> None:
    from manicule.core.version import CORE_VERSION  # noqa: PLC0415

    result = run(["--version"])
    assert result.exit_code == 0
    assert CORE_VERSION in result.stdout


# --- streaming ------------------------------------------------------------------------------


def _answer(text: str) -> Envelope:
    return succeeded("ask", "default", r.AnswerResultPayload(question="q", text=text))


def test_a_streamed_answer_is_not_printed_a_second_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ask`` at a terminal writes the answer once, as it arrives — and then leaves it alone.

    The renderer knows nothing about how the payload reached it, so without being told it
    would print the whole answer again underneath the streamed copy. A reader seeing the same
    paragraph twice has no way to know they are the same paragraph.
    """
    monkeypatch.setattr(cli.STATE, "json_output", False)
    monkeypatch.setattr(cli.STATE, "text_already_streamed", True)
    cli.print_envelope(_answer("The client retries twice."))
    assert capsys.readouterr().out.count("retries twice") == 0


def test_an_answer_that_was_not_streamed_is_printed_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The positive control: without streaming there is nothing on screen yet, so it prints."""
    monkeypatch.setattr(cli.STATE, "json_output", False)
    monkeypatch.setattr(cli.STATE, "text_already_streamed", False)
    cli.print_envelope(_answer("The client retries twice."))
    assert capsys.readouterr().out.count("retries twice") == 1


def test_the_streaming_flag_is_cleared_between_invocations(bound: ApplicationService) -> None:
    """It records what reached the screen, so a stale one would hide the next answer entirely."""
    del bound
    cli.STATE.text_already_streamed = True
    run(["--json", "document", "list"])
    assert cli.STATE.text_already_streamed is False
