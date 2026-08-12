"""What the command line does that the service does not: parsing, piping and exit status.

Only the adapter's own behaviour is asserted here. Anything about *what an operation means*
belongs to :mod:`tests.app.test_service`, and asserting it twice would make the command line a
second place a rule lives — which is the thing this design exists to prevent.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app import results as r
from manicule.app.dispatch import run_op
from manicule.app.results import succeeded
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.cli.shell import SHELLS, completion_script
from manicule.core.errors import ConfigError
from manicule.core.version import CORE_VERSION
from manicule.mcp.server import TOOL_NAMES
from manicule.storage.backup import BackupError
from tests.app.fakes import FakeBackend, FakeMaintenance, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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

    Asserted against the message and the backend, never against the rendered box. Terminal
    width, colour and elision differ by machine, and a test that reads them is testing the
    terminal.
    """
    assert "--yes" in cli.RESET_NEEDS_CONFIRMATION
    result = run(["reset-index"])
    assert result.exit_code != 0
    maintenance = asyncio.run(bound.backend.maintenance())
    assert isinstance(maintenance, FakeMaintenance)
    assert maintenance.resets == 0, "the index was reset by a command that refused"


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
    assert "--output" in cli.BACKUP_IS_NOT_A_RESTORE
    assert "--restore" in cli.BACKUP_IS_NOT_A_RESTORE
    result = run(["backup", "--output", "/tmp/a", "--restore", "/tmp/b"])  # noqa: S108 - never opened
    assert result.exit_code != 0


def test_backup_needs_somewhere_to_write(bound: ApplicationService) -> None:
    """A backup with nowhere to go is a command that would report success and write nothing."""
    del bound
    assert "--output" in cli.BACKUP_NEEDS_A_TARGET
    result = run(["backup"])
    assert result.exit_code != 0


def test_a_backup_consents_to_nothing_unless_the_flag_is_typed(
    bound: ApplicationService,
) -> None:
    """Refusing an exposed target is the default, and defaults are what get exercised."""
    result = run(["--json", "backup", "--output", "/tmp/somewhere"])  # noqa: S108 - the fake never writes
    assert result.exit_code == 0
    maintenance = asyncio.run(bound.backend.maintenance())
    assert isinstance(maintenance, FakeMaintenance)
    assert maintenance.backups == [(Path("/tmp/somewhere"), False)]  # noqa: S108


def test_allow_insecure_target_reaches_the_layer_that_acts_on_it(
    bound: ApplicationService,
) -> None:
    """Four layers between the flag and the ``stat`` that decides.

    Asserted at the backend rather than at the exit status, because a flag that parses,
    renders and exits zero while never arriving looks exactly like one that works — which is
    the defect (#60) this option was added to close.
    """
    result = run(
        ["--json", "backup", "--output", "/tmp/somewhere", "--allow-insecure-target"]  # noqa: S108
    )
    assert result.exit_code == 0
    maintenance = asyncio.run(bound.backend.maintenance())
    assert isinstance(maintenance, FakeMaintenance)
    assert maintenance.backups == [(Path("/tmp/somewhere"), True)]  # noqa: S108


def test_a_refused_backup_reaches_the_operator_as_a_result_not_a_traceback(
    bound: ApplicationService,
) -> None:
    """``BackupError`` is a ``ManiculeError``, and this is what that buys.

    Only that hierarchy becomes an envelope; anything else propagates as a defect. A security
    refusal delivered as a stack trace is read as a crash, and the path it names — the whole
    point of naming it — arrives buried in one.
    """
    maintenance = asyncio.run(bound.backend.maintenance())
    assert isinstance(maintenance, FakeMaintenance)
    maintenance.backup_error = BackupError(
        "backup target /srv/share carries group or other permissions (055)"
    )

    result = run(["--json", "backup", "--output", "/srv/share"])

    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert error["type"] == "BackupError"
    assert "/srv/share" in error["message"]


def test_allow_insecure_target_is_refused_on_a_restore_rather_than_ignored(
    bound: ApplicationService,
) -> None:
    """A security flag accepted where it does nothing is the shape of the original bug."""
    del bound
    assert "--allow-insecure-target" in cli.INSECURE_TARGET_IS_A_BACKUP_OPTION
    result = run(["backup", "--restore", "/tmp/b", "--allow-insecure-target"])  # noqa: S108 - never opened
    assert result.exit_code != 0


def test_an_export_consents_to_nothing_unless_the_flag_is_typed(
    bound: ApplicationService,
) -> None:
    """The same default as `backup`, because it is the same corpus in the same danger."""
    result = run(["--json", "export", "--output", "/tmp/somewhere"])  # noqa: S108 - the fake never writes
    assert result.exit_code == 0
    maintenance = asyncio.run(bound.backend.maintenance())
    assert isinstance(maintenance, FakeMaintenance)
    assert maintenance.exports == [(Path("/tmp/somewhere"), False)]  # noqa: S108


def test_export_gets_the_same_escape_hatch_under_the_same_name(
    bound: ApplicationService,
) -> None:
    """One flag spelled one way across both commands that write a copy of the corpus.

    An operator who learned it on `backup` should not discover that `export` calls it
    something else, or has nothing.
    """
    result = run(
        ["--json", "export", "--output", "/tmp/somewhere", "--allow-insecure-target"]  # noqa: S108
    )
    assert result.exit_code == 0
    maintenance = asyncio.run(bound.backend.maintenance())
    assert isinstance(maintenance, FakeMaintenance)
    assert maintenance.exports == [(Path("/tmp/somewhere"), True)]  # noqa: S108


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


# --- values survive a colouring terminal ---------------------------------------------------


def _laid_bare(text: str) -> str:
    """Rendered output with the layout taken back out: no escapes, no padding, no borders.

    What is left is the characters that were actually printed, which is the thing an
    identifier assertion is about.
    """
    without_escapes = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"[\s\u2502\u2500]", "", without_escapes)


def test_an_identifier_is_not_mangled_when_the_terminal_wants_colour(
    monkeypatch: pytest.MonkeyPatch, bound: ApplicationService
) -> None:
    """Rich's automatic highlighter puts escape codes *inside* a token.

    With highlighting on, a document id prints in pieces — styled around each run of digits —
    and nobody can copy it out of a pipe or paste it into the next command. The console turns
    that off; this asserts it stays off, in the one environment where it shows.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = run(["document", "list"])
    assert result.exit_code == 0
    identifier = asyncio.run(bound.document_list()).documents[0].id
    assert identifier in result.stdout, "a highlighted identifier reached the terminal in pieces"


@pytest.mark.parametrize("width", ["30", "40", "80", "200"])
def test_an_identifier_is_never_elided_however_narrow_the_terminal(
    monkeypatch: pytest.MonkeyPatch, bound: ApplicationService, width: str
) -> None:
    """A table cell elides by default, and an elided id reads exactly like a complete one.

    The id column folds instead, so a narrow terminal costs a line break rather than eight
    characters — and the prose columns give up their width first. The assertion is over the
    characters that were printed, with the layout removed, because a folded id is complete
    and a truncated one is not.
    """
    monkeypatch.setenv("COLUMNS", width)
    result = run(["document", "list"])
    assert result.exit_code == 0
    identifier = asyncio.run(bound.document_list()).documents[0].id
    assert identifier in _laid_bare(result.stdout), (
        f"at {width} columns the id was truncated rather than folded"
    )


def test_the_version_is_one_plain_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read by scripts far more often than by people."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = run(["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"{CORE_VERSION}\n"


FINGERPRINT = '{"dimension":1024,"model_id":"BAAI/bge-m3","normalized":true}'
"""A canonical embedding fingerprint: JSON, with numbers, strings and braces in it.

The shape of value this output is full of, and the one Rich's highlighter has the most to say
about.
"""


def _status_output(capsys: pytest.CaptureFixture[str]) -> str:
    render.render_index_status(
        render.console(),
        r.IndexStatus(documents=1, chunks=2, embed_fingerprint=FINGERPRINT),
    )
    return capsys.readouterr().out


def test_a_fingerprint_is_not_styled_through(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rich's highlighter styles JSON element by element, and this value is an identity.

    A URI it styles *around*, so that token survives. A fingerprint it takes apart: the
    braces, each key and every number get their own escape codes, so what reaches the terminal
    is a dozen fragments. That string is what a re-embed compares, and it is printed so
    somebody can compare it — highlighting is off for this, not for taste.

    The width is pinned wide on purpose. The claim here is about *styling*, and letting the
    ambient terminal decide whether the line also wraps would make the test's subject depend
    on the machine running it. Wrapping is the next test's business.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    assert FINGERPRINT in _status_output(capsys), (
        "a highlighted fingerprint reached the terminal in fragments"
    )


@pytest.mark.parametrize("width", ["40", "80", "200"])
def test_a_fingerprint_is_never_truncated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], width: str
) -> None:
    """However narrow the terminal, every character of it is printed.

    It is on its own line rather than in a table cell for exactly this reason: a cell elides,
    and half a fingerprint compares unequal to the thing it is half of while looking like a
    fingerprint.
    """
    monkeypatch.setenv("COLUMNS", width)
    assert FINGERPRINT in _laid_bare(_status_output(capsys)), (
        f"at {width} columns the fingerprint was truncated rather than wrapped"
    )


# --- the two registries the command line dispatches through -----------------------------------


CLI_ONLY_OPS: frozenset[str] = frozenset(
    {
        "auth_create_key",
        "auth_list_keys",
        "auth_revoke_key",
        "backup",
        "completion",
        "export",
        "import",
        "index_changes",
        "init",
        "reset_index",
        "restore",
        "start",
        "stop",
        "upgrade",
    }
)
"""Operations the command line carries that the MCP surface deliberately does not.

Named here so the emptiness check below covers both halves of ``PAYLOADS``. ``TOOL_NAMES``
alone would leave the command-line-only operations unproven, and those are exactly the ones a
new command is most likely to join — the ones with no tool to keep them honest.
"""

RENDERER_LANDMARKS: tuple[type[r.Payload], ...] = (
    r.AnswerResultPayload,
    r.SearchResult,
    r.Diagnosis,
    r.IngestReport,
)
"""Renderers that must exist, named individually.

A count is satisfied by any table of the right size. These four are the results of the
operations somebody actually runs — ask, search, doctor, index — so a table that has lost its
way fails here rather than passing on how many entries it happens to have.
"""


def _registries() -> tuple[Mapping[str, type[r.Payload]], Mapping[type[r.Payload], object]]:
    """The two tables the command line dispatches through, having proved they are populated.

    An invariant over two empty collections holds vacuously, and this repository has shipped
    that shape before. So the emptiness check lives here, where both directions of the
    invariant go through it, rather than beside them where one could skip it.

    The floor is *derived* rather than written down: every MCP tool's operation is also a
    command-line operation, so :data:`~manicule.mcp.server.TOOL_NAMES` is a lower bound that
    nobody has to maintain — and one that fails loudly if the tables are read from the wrong
    module.
    """
    payloads = cli.PAYLOADS
    renderers = render.RENDERERS

    missing_tools = sorted(set(TOOL_NAMES) - set(payloads))
    assert missing_tools == [], (
        f"PAYLOADS is missing {missing_tools}, which are MCP tool operations. Every tool's "
        f"operation is one the command line can run too, so this is either a table read from "
        f"the wrong place or a surface that has genuinely diverged."
    )
    missing_cli = sorted(CLI_ONLY_OPS - set(payloads))
    assert missing_cli == [], (
        f"PAYLOADS is missing the command-line-only operation(s) {missing_cli}. If one was "
        f"deliberately removed, delete it from CLI_ONLY_OPS in the same change."
    )
    absent = [kind.__name__ for kind in RENDERER_LANDMARKS if kind not in renderers]
    assert absent == [], (
        f"RENDERERS has no entry for {absent}. Whatever table this is, it is not the one the "
        f"command line renders through."
    )
    return payloads, renderers


def test_every_operation_the_command_line_can_emit_has_a_renderer() -> None:
    """A payload with no renderer is a ``KeyError`` on the operation's **success** path.

    ``print_envelope`` parses the envelope into the type ``PAYLOADS`` names, then hands it to
    ``render``, which looks it up in ``RENDERERS``. A missing entry raises — and it raises on
    the path least likely to be exercised by hand, because whoever adds a command checks the
    error case first and sees the failure envelope render perfectly well.

    This is the check that would have caught it with nobody in the loop.
    """
    payloads, renderers = _registries()
    unrenderable = sorted(
        f"{op} -> {kind.__name__}" for op, kind in payloads.items() if kind not in renderers
    )
    assert unrenderable == [], (
        f"these operations produce a payload no renderer handles: {unrenderable}. Each one is "
        f"a KeyError the moment the operation succeeds. Add the payload type to RENDERERS in "
        f"manicule.cli.render."
    )


def test_every_renderer_is_reachable_from_some_operation() -> None:
    """The other direction, and it catches the mistake that reads as coverage.

    A renderer for a payload no operation produces is dead code that looks exactly like a
    handled case — so the table appears complete while the operation somebody actually wanted
    renders through nothing. It is also what is left behind when an operation is removed and
    its view is not.
    """
    payloads, renderers = _registries()
    produced = set(payloads.values())
    unreachable = sorted(kind.__name__ for kind in renderers if kind not in produced)
    assert unreachable == [], (
        f"these renderers cannot be reached by any operation: {unreachable}. Either the "
        f"operation that produced them was removed and its view was not, or a payload type "
        f"is missing from PAYLOADS."
    )


EMITTED_OP_LANDMARKS: frozenset[str] = frozenset({"ask", "search", "document_list", "doctor"})
"""Operation names the scan below must have found in the source.

The scan reads string literals out of an AST, which is the kind of derivation that returns an
empty set when it is pointed at the wrong thing — and an empty set satisfies a subset check.
These four are emitted by commands nobody is going to delete.
"""


def _emitted_ops() -> set[str]:
    """Every operation name passed to ``emit`` as a literal, read from the source.

    The op is a string argument inside a lambda, so it exists nowhere a type checker or an
    import can reach it — but it is what ``print_envelope`` looks ``PAYLOADS`` up by, so a
    command that emits an operation the table does not name is a ``KeyError`` the first time
    it succeeds. Reading the source is the only way to see them.

    Deliberately only literals. A computed op name would not be found here, and a scan that
    guessed at one would report a name nothing emits.
    """
    source = Path(cli.__file__)
    assert source.is_file(), f"{source} is not a file; the scan below would read nothing"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "emit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def test_every_operation_the_command_line_emits_has_a_payload_type() -> None:
    """The other half of the same ``KeyError``, and the half a new command actually hits.

    ``print_envelope`` does ``PAYLOADS[envelope.op]`` before it renders anything. Adding a
    command means writing an op name in one file and an entry in another, and nothing but this
    connects them — which is exactly the step somebody adding a command forgets, because the
    failure path renders perfectly well and the success path is the one that raises.
    """
    emitted = _emitted_ops()
    missing_landmarks = sorted(EMITTED_OP_LANDMARKS - emitted)
    assert missing_landmarks == [], (
        f"the scan did not find {missing_landmarks} among the operations this module emits. "
        f"Whatever it parsed, it was not the command line."
    )
    unknown = sorted(emitted - set(cli.PAYLOADS))
    assert unknown == [], (
        f"these operations are emitted but named in no PAYLOADS entry: {unknown}. Each is a "
        f"KeyError the first time the operation succeeds. Add it to PAYLOADS in "
        f"manicule.cli.main."
    )
