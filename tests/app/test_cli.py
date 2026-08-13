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
from typing import TYPE_CHECKING, Any, cast

import pytest
from typer._click.exceptions import NoSuchOption, UsageError
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
from tests.conftest import CLEARED_TERMINAL_VARIABLES

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

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


EMITTED_OP_LANDMARKS: frozenset[str] = frozenset(
    {"ask", "search", "document_list", "doctor", "start", "stop", "completion", "index_changes"}
)
"""Operation names the scan below must have found in the source.

The scan reads string literals out of an AST, which is the kind of derivation that returns an
empty set when it is pointed at the wrong thing — and an empty set satisfies a subset check.

``start``, ``stop``, ``completion`` and ``index_changes`` are why this is not just the obvious
commands. Each reaches ``print_envelope`` through a call shape that is not ``emit``, from a
file that is not ``main.py`` in three of the four cases — and a scan that lost all four would
still find the large majority of the operations and look perfectly healthy.
"""

MINIMUM_CLI_MODULES = 5
"""A floor on how many modules the surface has, far below the real count.

Present to catch a scan reading the wrong directory, not to track the package's size.
"""

OP_TAKING_CALLS: frozenset[str] = frozenset({"emit", "run_op", "succeeded", "failed"})
"""Functions whose **first positional argument** is an operation name.

``Envelope(op=...)`` is the fifth shape and is a keyword rather than a positional, which is
why it is handled separately below. It is also the one a scan written from a quick reading of
``main.py`` misses, because ``completion`` is the only operation that uses it.
"""


def _op_literals(tree: ast.AST) -> set[str]:
    """Operation names appearing as literals in one module's AST.

    Deliberately only literals. A computed op name would not be found, and a scan that guessed
    at one would report a name nothing emits.

    Deliberately only *bare-name* calls, too — ``emit(...)`` and not ``something.emit(...)``.
    That narrowing is safe in the direction that matters: an operation reached through an
    attribute call is one this scan does not find, which makes its ``PAYLOADS`` entry
    unaccounted for and fails :func:`test_the_scan_accounts_for_every_payload_entry`. A missed
    shape is a loud failure rather than a quiet hole, which is the whole reason that test
    exists.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.id if isinstance(node.func, ast.Name) else None
        if called in OP_TAKING_CALLS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add(first.value)
        if called == "Envelope":
            for keyword in node.keywords:
                value = keyword.value
                if (
                    keyword.arg == "op"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    found.add(value.value)
    return found


def _emitted_ops() -> set[str]:
    """Every operation name the command line names, read from every module of the surface.

    The op is a string argument — frequently inside a lambda — so it exists nowhere a type
    checker or an import can reach it. But it is what ``print_envelope`` looks ``PAYLOADS`` up
    by, so an operation the table does not name is a ``KeyError`` the first time it succeeds.
    Reading the source is the only way to see them.

    **The whole package, not just ``main.py``.** Four operations reach ``print_envelope`` from
    ``watch.py`` and ``serving.py``, on three call shapes that are not ``emit`` — which is
    exactly the blind spot a narrower scan leaves while its name promises otherwise.
    """
    package = Path(cli.__file__).parent
    assert package.is_dir(), f"{package} is not a directory; the scan would read nothing"
    modules = sorted(package.rglob("*.py"))
    assert len(modules) >= MINIMUM_CLI_MODULES, (
        f"the scan found {len(modules)} module(s) under {package}, below the floor of "
        f"{MINIMUM_CLI_MODULES}. It is reading the wrong package."
    )
    found: set[str] = set()
    for module in modules:
        found |= _op_literals(ast.parse(module.read_text(encoding="utf-8")))
    return found


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
        f"the scan did not find {missing_landmarks} among the operations this surface names. "
        f"Whatever it parsed, it was not the command line."
    )
    unknown = sorted(emitted - set(cli.PAYLOADS))
    assert unknown == [], (
        f"these operations are named but appear in no PAYLOADS entry: {unknown}. Each is a "
        f"KeyError the first time the operation succeeds. Add it to PAYLOADS in "
        f"manicule.cli.main."
    )


def test_the_scan_accounts_for_every_payload_entry() -> None:
    """The other half of the same accounting, and what keeps the blind spot from reopening.

    The check above proves everything the scan *found* is in the table. This proves the scan
    found everything the table names — so an operation reaching ``print_envelope`` through a
    sixth call shape, or from a module nobody thought to look in, shows up here as a
    ``PAYLOADS`` entry the scan cannot account for.

    That is the failure this test exists for. The previous version scanned one function in one
    file, covered twenty-nine of thirty-three operations, and was named as though it covered
    them all; nothing said otherwise because the four it missed were all legitimate. A
    derived total says otherwise.

    Set equality is deliberately asserted as two separate checks rather than one. "Something
    is emitted that the table does not name" and "the table names something the scan cannot
    find" are different defects with different fixes, and one message for both would describe
    neither.
    """
    unaccounted = sorted(set(cli.PAYLOADS) - _emitted_ops())
    assert unaccounted == [], (
        f"PAYLOADS names {unaccounted}, which the scan could not find anywhere in the command "
        f"line's source. Either the operation is dead and its entry should go, or it reaches "
        f"print_envelope through a call shape this scan does not know about — in which case "
        f"add the shape to OP_TAKING_CALLS, or the whole surface loses the guarantee."
    )


# --- `--json` reaches the same place from either side of the command name ---------------------


ENVELOPE_PRODUCING: frozenset[str] = frozenset({"emit", "print_envelope"}) | OP_TAKING_CALLS
"""Bare-name calls that put a result envelope in front of the caller.

``emit`` and ``print_envelope`` are the two that write one; the four in
:data:`OP_TAKING_CALLS` are the ones that build one. A command reaching any of them, directly
or through a helper, is a command that emits data — which is the set ``--json`` has to cover.
"""

MINIMUM_DATA_EMITTING_COMMANDS = 25
"""A floor on the derivation below, well under the real count.

Present for one reason: a subset check over an empty set passes, and a derivation that walked
the wrong tree or matched no call shape would return one. This repository has shipped that
shape more than once — a scan that read no files, a contract test that enumerated no routes —
so the number is asserted before anything is asserted *about* the set.
"""

JSON_LANDMARKS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("doctor",),
        ("search",),
        ("ask",),
        ("index",),
        ("document", "list"),
        ("config", "set"),
        ("auth", "create-key"),
        ("completion",),
        ("stop",),
    }
)
"""Commands the derivation must have found, named individually.

Not the obvious ones only. ``completion`` and ``stop`` reach an envelope without going through
``emit`` at all, and ``auth create-key`` and ``config set`` live two levels down on
sub-applications — so a walk that only descended one level, or only recognised ``emit``, would
still return the large majority of commands and look perfectly healthy.
"""


def _called_names(node: ast.AST) -> set[str]:
    """Every bare-name call inside one function, nested definitions included.

    Bare names only — ``emit(...)`` and not ``something.emit(...)`` — matching
    :func:`_op_literals`. The narrowing is safe in the direction that matters: a call this
    misses makes its command look non-emitting, and
    :func:`test_the_derivation_covers_every_command_the_interface_offers` fails loudly rather
    than quietly shrinking what ``--json`` is asserted over.
    """
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def _call_graph() -> dict[str, set[str]]:
    """Every function the command line defines, and the bare names each one calls.

    One flat namespace across the package, deliberately. A call is a bare name resolved through
    an import — ``main.py`` names ``serve_forever``, which ``serving.py`` defines — so
    qualifying by module would break the cross-module hop this graph exists to follow.

    The cost is that two modules defining the same function name would be merged, and a
    non-emitting one would inherit its namesake's verdict. That is a false *positive*: it can
    only make a command look like it emits data, which is the direction that adds an assertion
    rather than dropping one. There are no such collisions today.
    """
    package = Path(cli.__file__).parent
    graph: dict[str, set[str]] = {}
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                graph.setdefault(node.name, set()).update(_called_names(node))
    return graph


def _emitting_functions() -> set[str]:
    """Functions that reach an envelope, following calls until the set stops growing.

    Transitive rather than one-hop, and derived rather than listed. Four commands hand off to a
    runner in another module — ``start`` to ``serve_forever``, ``stop`` to ``stop_running``,
    ``ask --repl`` to ``run_repl``, ``index --watch`` to ``watch_path`` — and a list of those
    hand-offs is exactly the kind of thing that is edited when a fifth appears rather than
    heeded.
    """
    graph = _call_graph()
    emitting = {name for name, calls in graph.items() if calls & ENVELOPE_PRODUCING}
    while True:
        grown = {
            name for name, calls in graph.items() if calls & emitting or calls & ENVELOPE_PRODUCING
        }
        if grown == emitting:
            return emitting
        emitting = grown


def _leaf_commands() -> dict[tuple[str, ...], Any]:
    """Every command that can be typed, by the argv that reaches it.

    Read from the **built** command tree rather than from the source, so a command registered
    on a sub-application and never attached is absent here exactly as it is absent from the
    interface.
    """
    from typer.core import TyperGroup  # noqa: PLC0415 - only this derivation walks the tree
    from typer.main import get_command  # noqa: PLC0415 - only this derivation builds the tree

    def walk(group: Any, prefix: tuple[str, ...]) -> dict[tuple[str, ...], Any]:
        found: dict[tuple[str, ...], Any] = {}
        for name, command in group.commands.items():
            if isinstance(command, TyperGroup):
                found |= walk(command, (*prefix, name))
            else:
                found[*prefix, name] = command
        return found

    return walk(get_command(cli.app), ())


def _data_emitting_commands() -> dict[tuple[str, ...], Any]:
    """The commands ``--json`` has to work on, derived from what each one actually calls."""
    emitting = _emitting_functions()
    return {
        path: command
        for path, command in _leaf_commands().items()
        if getattr(command.callback, "__name__", "") in emitting
    }


def _parse(command: Any, name: str, args: list[str]) -> None:
    """Run the **real** parser over one command's arguments, and throw away the result.

    Parsing rather than invoking, deliberately. ``start`` and ``stop`` are in this set and
    running them would bind a socket or hunt for a daemon; ``make_context`` stops at the point
    the defect lived, which is argument parsing and nothing after it.

    ``resilient_parsing`` is deliberately **off**. With it on Click ignores unknown options
    entirely, so the assertion below would pass for ``--nonsense`` just as readily as for
    ``--json`` — a guard that cannot fail, which is worse than no guard.

    ``STATE`` is put back afterwards. Parsing ``--json`` *sets* it — that side effect is the
    mechanism working, and it is how the value escapes an option Click never passes to the
    command — but a helper that left the flag on would hand the next test a command line
    already in JSON mode.
    """
    was = cli.STATE.json_output
    try:
        command.make_context(name, args, parent=None)
    finally:
        cli.STATE.json_output = was


def test_the_derivation_covers_every_command_the_interface_offers() -> None:
    """Before anything is asserted *about* the set, that the set is the right size and shape.

    Three separate claims, because they fail for three different reasons. The floor catches a
    walk that found nothing. The landmarks catch a walk that found plenty and missed the
    awkward ones. The equality catches the case that would quietly weaken every assertion
    below: a command the derivation decided does not emit data.

    That last one is the interesting assertion. Every command manicule has emits an envelope on
    some path, so "the commands that emit data" and "the commands" are the same set today. If
    somebody adds one that genuinely emits nothing, this fails and asks them to say so on
    purpose — rather than ``--json`` silently ceasing to be a promise about the whole interface.
    """
    emitting = _data_emitting_commands()
    assert len(emitting) >= MINIMUM_DATA_EMITTING_COMMANDS, (
        f"the derivation found {len(emitting)} data-emitting command(s), below the floor of "
        f"{MINIMUM_DATA_EMITTING_COMMANDS}. It is walking the wrong tree, or recognising none "
        f"of the calls that produce an envelope."
    )
    missing = sorted(JSON_LANDMARKS - set(emitting))
    assert missing == [], (
        f"the derivation did not find {missing} among the commands that emit data. Whatever it "
        f"walked, it was not the whole command tree."
    )
    silent = sorted(set(_leaf_commands()) - set(emitting))
    assert silent == [], (
        f"these commands were not found to emit any envelope: {silent}. Either one of them "
        f"reaches print_envelope by a route this scan cannot follow — in which case the scan "
        f"needs the shape, or --json stops being asserted for it — or a command that genuinely "
        f"emits nothing has been added, which is worth saying out loud."
    )


SAMPLE_VALUES: dict[str, list[str]] = {"--json": [], "--workspace": ["scratch"]}
"""Argv for each shared option, since one is a flag and the other takes a value.

Keyed by the same names :data:`~manicule.cli.main.SHARED_OPTIONS` uses, and checked against it
by :func:`test_every_shared_option_has_a_way_to_be_typed` — so an option added to that table
without a sample here fails rather than being quietly skipped by every test below.
"""


def test_every_shared_option_has_a_way_to_be_typed() -> None:
    """The parametrisation below is only as complete as this mapping.

    A shared option with no entry would be silently absent from every assertion in this
    section, which is the failure mode where coverage is reported for an option nothing typed.
    """
    assert set(SAMPLE_VALUES) == set(cli.SHARED_OPTIONS), (
        "SAMPLE_VALUES and SHARED_OPTIONS have diverged, so some shared option is either "
        "untested or tested and not declared."
    )


@pytest.mark.parametrize("option", sorted(cli.SHARED_OPTIONS))
def test_every_data_emitting_command_takes_a_shared_option_after_the_command_name(
    option: str,
) -> None:
    """The defect: ``manicule doctor --json`` was ``No such option: --json``, exit 2.

    Both shared options were declared once, on the root callback, so each worked only in front
    of the command name. That is the position nobody types first, and for ``--json`` the
    restriction had been written up twice as though it were a decision — once by correcting the
    example in ``docs/surfaces.md`` that used the natural order, and once by describing the rule
    in the README.

    Asserted for every command and every shared option rather than for the one in the bug
    report, because a fix that special-cased either leaves the same trap everywhere else.
    """
    for path, command in sorted(_data_emitting_commands().items()):
        try:
            _parse(command, path[-1], [option, *SAMPLE_VALUES[option]])
        except NoSuchOption as unknown:  # pragma: no cover - the assertion is the report
            pytest.fail(
                f"`manicule {' '.join(path)} {option}` is rejected: {unknown.option_name} is "
                f"not an option of this command. {option} has to work on either side of the "
                f"command name."
            )
        except UsageError:
            # A required argument this parse did not supply. A different failure entirely, and
            # not the one under test: `document get --json` still has to be told which document.
            continue


@pytest.mark.parametrize("option", sorted(cli.SHARED_OPTIONS))
def test_every_data_emitting_command_still_takes_a_shared_option_before_the_command_name(
    option: str,
) -> None:
    """The documented position, which the fix must not have traded away.

    ``manicule --json <command>`` is in the README, in ``docs/deployment.md``'s health-gate
    recipe and in every envelope-parity assertion. Moving an option rather than sharing it
    would have broken all three while making the reported bug go away.
    """
    for path in sorted(_data_emitting_commands()):
        # `--help` stops the run at the point this test is about, so `start` and `stop` can be
        # asserted alongside the rest without one of them binding a socket.
        result = run([option, *SAMPLE_VALUES[option], *path, "--help"])
        assert result.exit_code == 0, (
            f"`manicule {option} {' '.join(path)}` no longer parses: {result.output}"
        )


ROOT_ONLY_OPTIONS: dict[str, str] = {
    "--version": (
        "replaces the invocation rather than modifying it: eager, prints one line and exits "
        "before any command runs. `manicule doctor --version` would be asking for the version "
        "*of doctor*, which is not a thing that exists, and thirty-one ways to ask one "
        "question is not a fix."
    ),
}
"""Root options deliberately **not** shared with the commands, and why for each.

A reason per entry rather than a bare list, because "this one is different" is a claim that
needs stating: the next reader has to be able to tell a decision from an oversight, and that is
exactly what could not be told about ``--json`` — its restriction was written up in the README
as though somebody had chosen it, and nobody had.
"""


def _root_option_names() -> set[str]:
    """The long name of every option ``manicule`` itself declares, from the built tree."""
    from typer.main import get_command  # noqa: PLC0415 - only this derivation builds the tree

    root = get_command(cli.app)
    # The longest spelling, so `--workspace`/`-w` is keyed by the name the tables use.
    return {max(option.opts, key=len) for option in root.params}


def test_every_root_option_is_either_shared_with_the_commands_or_deliberately_not() -> None:
    """The accounting that stops this defect recurring under a different option's name.

    ``--json`` and ``--workspace`` were the same bug twice: an option declared on the root
    callback alone, so it could not be typed where people type it. Fixing the second without
    this test would leave the door open for a third, and the third would be found the same way
    the first two were — by somebody hitting it.

    So the root callback's **real** parameters are read from the built tree and every one must
    be accounted for: either it is shared with the commands, or it carries a written reason for
    not being. An option added tomorrow belongs to neither set and fails here, which forces
    whoever adds it to decide rather than to default into the trap.

    Asserted as set equality in both directions. "A root option nobody classified" and "a
    classification for an option that no longer exists" are different mistakes — the second is
    what is left behind when an option is removed — and one message for both would describe
    neither.
    """
    declared = _root_option_names()
    classified = set(cli.SHARED_OPTIONS) | set(ROOT_ONLY_OPTIONS)

    unclassified = sorted(declared - classified)
    assert unclassified == [], (
        f"manicule declares {unclassified}, which is neither shared with the commands nor "
        f"listed as deliberately root-only. Decide which it is: if it modifies the operation, "
        f"add it to SHARED_OPTIONS in manicule.cli.main; if it replaces the invocation, add it "
        f"to ROOT_ONLY_OPTIONS here with the reason."
    )
    stale = sorted(classified - declared)
    assert stale == [], (
        f"{stale} is classified here but the root callback no longer declares it. Either the "
        f"option was removed and its entry was not, or it has been renamed."
    )
    assert all(ROOT_ONLY_OPTIONS.values()), "a root-only option with no reason is an oversight"


def test_the_root_option_derivation_reads_the_real_command_line() -> None:
    """The floor under the accounting above, which is vacuous over an empty set.

    ``--version`` is the landmark: it is the one root option that is *not* shared, so a
    derivation returning only the shared ones would satisfy the equality above while proving
    nothing about the case the exclusion list exists for.
    """
    declared = _root_option_names()
    assert "--version" in declared, "the derivation is not reading manicule's own options"
    assert set(cli.SHARED_OPTIONS) <= declared, (
        "a shared option is not declared on the root callback at all, so the two positions "
        "cannot be the same option"
    )


def test_an_unknown_option_is_still_rejected_by_every_one_of_those_commands() -> None:
    """The control. Without it the test above proves nothing.

    Every command now carries an option the parser did not previously know, added by a
    mechanism that reaches inside the built command tree. If that mechanism had instead made
    the commands accept *anything* — by widening the parser rather than adding one option —
    the assertion above would pass on a command line that had stopped checking its arguments,
    which is a far worse defect than the one being fixed.
    """
    for path, command in sorted(_data_emitting_commands().items()):
        with pytest.raises(NoSuchOption):
            _parse(command, path[-1], ["--not-an-option"])


def test_json_before_and_after_the_command_name_are_the_same_invocation(
    bound: ApplicationService,
) -> None:
    """Not merely both accepted — both producing the same bytes.

    An option that parsed in the new position and reached nothing would satisfy every
    assertion above while printing the human table to stdout, which is the exact shape of the
    ``--allow-insecure-target`` defect this repository has already been bitten by.
    """
    del bound
    before = run(["--json", "document", "list"])
    after = run(["document", "list", "--json"])
    assert before.exit_code == 0
    assert after.exit_code == 0
    assert json.loads(after.stdout) == json.loads(before.stdout)
    assert json.loads(after.stdout)["op"] == "document_list"


ESCAPE = "\x1b"
"""The first byte of every ANSI sequence Rich emits.

Asserted against the characters that were actually printed. The mistake this replaces is
asserting that colour was *configured* off — a check that passes while the library it is about
carries on writing escapes, because the setting was read at import and the test never looked at
the output.
"""

SGR = re.compile(r"\x1b\[([0-9;]*)m")
"""Every "select graphic rendition" sequence, with its parameters.

The parameters are the point. ``\\x1b[1m`` is bold and ``\\x1b[32m`` is green, and the
difference decides whether the control below can fail.
"""

COLOUR_PARAMETERS = frozenset(
    [*(str(code) for code in range(30, 38)), *(str(code) for code in range(90, 98)), "38"]
)
"""SGR parameters that set a foreground colour: the 8, the bright 8, and 256/true-colour.

Backgrounds are not listed because this output sets none, and a set naming codes nothing emits
would be a guess about what to expect.
"""


def _colours(text: str) -> set[str]:
    """The foreground colours actually present in some output.

    Separate from "are there escapes at all", and the separation is the whole reason this
    exists. Rich's ``no_color`` strips **colour** and keeps everything else, so a run with
    colour genuinely turned off still emits ``\\x1b[1m`` for bold — and a control asserting
    only that *an* escape appeared would pass while reporting on a stream that had no colour in
    it. That is the shape of vacuous check this control was written to prevent, so it must not
    be the shape of the control.
    """
    return {
        parameter
        for sequence in SGR.findall(text)
        for parameter in sequence.split(";")
        if parameter in COLOUR_PARAMETERS
    }


def test_a_colouring_terminal_puts_no_escape_sequences_in_the_json(
    monkeypatch: pytest.MonkeyPatch, bound: ApplicationService
) -> None:
    """``manicule doctor --json | jq`` has to work from an interactive shell.

    ``FORCE_COLOR`` is the environment a terminal actually presents, and it is the one where
    this fails: Rich decides to colour, and a single escape sequence anywhere in the stream
    makes the whole document unparseable — for ``jq``, and for every JSON library there is.

    The output is captured and read. Nothing here asserts that a flag was set, because the
    flag being set is not the claim; the claim is about what came out.
    """
    del bound
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = run(["doctor", "--json"])
    assert result.exit_code == 0
    assert ESCAPE not in result.stdout, "an ANSI escape sequence reached stdout under --json"
    assert json.loads(result.stdout)["op"] == "doctor"


def test_the_same_terminal_does_colour_the_human_output(
    monkeypatch: pytest.MonkeyPatch, bound: ApplicationService
) -> None:
    """The positive control, and without it the test above proves nothing.

    ``FORCE_COLOR`` has to actually reach Rich for the absence of escapes under ``--json`` to
    mean anything. If it did not — a typo in the variable, a console built before the
    environment was set — the assertion above would hold on a stream that was never going to be
    coloured, and would go on holding after somebody reintroduced a banner.

    **Asserted on colour rather than on escapes**, which the first version of this control got
    wrong. Rich's ``no_color`` strips colour and leaves bold, so a console built with colour
    genuinely disabled still writes ``\\x1b[1m`` — and "an escape appeared" passed on output
    with no colour in it. The control could not fail for the reason it existed to detect, which
    is the same defect it was guarding the neighbouring test against.

    The colour environment is the fixture's, not the caller's. This test failed under
    ``TERM=dumb`` before ``colour_environment`` existed, because ``TERM`` declares what the
    stream can *render* and overrides ``FORCE_COLOR``, which only declares that it is a
    terminal.
    """
    del bound
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = run(["doctor"])
    assert result.exit_code == 0
    assert _colours(result.stdout), (
        f"no foreground colour reached stdout, so the no-escapes assertion under --json is "
        f"vacuous. Sequences present: {sorted(set(SGR.findall(result.stdout)))}"
    )


def test_the_human_diagnosis_is_unchanged_when_json_is_absent(
    bound: ApplicationService,
) -> None:
    """The regression bug6 asks for by name: the table a person reads still prints.

    Asserted with the layout taken back out, because the subject is the characters that were
    printed rather than how Rich chose to box them on this machine.
    """
    del bound
    result = run(["doctor"])
    assert result.exit_code == 0
    bare = _laid_bare(result.stdout)
    assert "overall:" in bare
    assert "configuration" in bare
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_naming_json_in_both_positions_at_once_is_not_an_error(
    bound: ApplicationService,
) -> None:
    """``manicule --json doctor --json`` says the same thing twice, which is not a conflict.

    It is what a script that already passed the flag produces when somebody adds it to the
    other end, and refusing it would be a usage error over an unambiguous request. The
    per-command option **or**\\ s into the root's value rather than assigning, so the second,
    defaulted-to-``False`` position cannot cancel the first.
    """
    del bound
    both = run(["--json", "document", "list", "--json"])
    assert both.exit_code == 0
    assert json.loads(both.stdout)["ok"] is True


# --- what a reader is told when the number alone would mislead -------------------------------


def _search_output(
    capsys: pytest.CaptureFixture[str], *, band: str, reason: str, hits: int = 1
) -> str:
    """Render one search result and hand back what reached the terminal."""
    render.render_search(
        render.console(),
        r.SearchResult(
            query="how do I fix a carburettor on a 1974 Norton",
            profile="balanced",
            count=hits,
            hits=tuple(
                r.SearchHit(
                    document_id="d",
                    chunk_id=f"c{index}",
                    uri="file:///x",
                    title="x.md",
                    score=0.0,
                    text="something",
                )
                for index in range(hits)
            ),
            confidence=0.0,
            confidence_band=band,
            confidence_reason=reason,
        ),
    )
    return capsys.readouterr().out


REASON = (
    "every passage retrieved sits at or below the level this corpus returns for a question "
    "it has no answer to"
)


def test_a_search_the_corpus_cannot_answer_says_why_rather_than_only_how_much(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The failure this closes is a reader trusting three passages because they look like prose.

    Retrieval computes the reason and the payload has always carried it; the browser surface
    renders it and this one dropped it, so the same query explained itself in one place and
    printed a bare ``0.00 (none)`` above plausible-looking excerpts in the other.
    """
    out = _search_output(capsys, band="none", reason=REASON)
    assert REASON in " ".join(out.split())


def test_a_confident_search_does_not_explain_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reason nobody needed, printed every time, is how a reader learns to skip the last line.

    The scores beside each hit already say why a high-confidence result is high-confidence.
    """
    out = _search_output(capsys, band="high", reason="the passages scored well")
    assert "the passages scored well" not in " ".join(out.split())


def test_a_low_confidence_search_explains_itself_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``low`` and ``none`` are both bands where the number needs a sentence beside it."""
    out = _search_output(capsys, band="low", reason=REASON)
    assert REASON in " ".join(out.split())


def test_a_pending_model_download_is_announced_before_the_command_that_pays_for_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``init`` recommends ``index``, and on a fresh machine that command downloads a gigabyte.

    Asserted on the rendered output rather than the payload, because the whole failure being
    fixed is one of *emphasis*: the fact was reachable in ``notes`` all along and nobody read
    it there.
    """
    render.render_init(
        render.console(),
        r.InitReport(
            path="/x/config.toml",
            data_dir="/x/data",
            embedding_provider="mlx",
            embedding_model="BAAI/bge-m3",
            llm_provider="ollama",
            llm_model="qwen2.5:14b",
            weights_pending=True,
        ),
    )
    out = " ".join(capsys.readouterr().out.split())
    assert "not on this machine yet" in out
    assert "Expect minutes, once." in out


def test_an_install_with_its_weights_already_here_is_not_warned_about_a_download(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Second ``init`` on a working machine, where the warning would be a lie."""
    render.render_init(
        render.console(),
        r.InitReport(
            path="/x/config.toml",
            data_dir="/x/data",
            embedding_provider="mlx",
            embedding_model="BAAI/bge-m3",
            llm_provider="ollama",
            llm_model="qwen2.5:14b",
            weights_pending=False,
        ),
    )
    out = " ".join(capsys.readouterr().out.split())
    assert "not on this machine yet" not in out
    assert "next:" in out


def _ingest_output(capsys: pytest.CaptureFixture[str], *, ingested: int, error: str = "") -> str:
    render.render_ingest(
        render.console(),
        r.IngestReport(connector="local", discovered=13, ingested=ingested, error=error),
    )
    return " ".join(capsys.readouterr().out.split())


def test_the_longest_command_in_a_first_run_says_what_comes_after_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A first ingest can take minutes and used to end on a table with nothing to do about it."""
    assert "next: manicule search <query>" in _ingest_output(capsys, ingested=13)


def test_a_run_that_indexed_nothing_does_not_suggest_searching_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After a no-op re-run, "now search it" is advice about somebody else's corpus."""
    assert "next:" not in _ingest_output(capsys, ingested=0)


def test_a_run_that_failed_says_how_to_resume_rather_than_what_to_do_next(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The next command after a broken ingest is the same ingest, and the error already says so."""
    out = _ingest_output(capsys, ingested=4, error="the connector stopped")
    assert "running it again resumes" in out
    assert "next:" not in out


def _address_output(capsys: pytest.CaptureFixture[str], *, transport: str, stopped: bool) -> str:
    render.render_address(
        render.console(),
        r.ServerAddress(transport=transport, host="127.0.0.1", port=8765, tools=19),
        stopped=stopped,
    )
    return " ".join(capsys.readouterr().out.split())


def test_stopping_a_server_does_not_read_like_starting_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An address reads the same whichever direction the server was going; the op does not.

    `manicule stop` printed the start banner — the bind line and then the URL of a browser
    surface that had just gone away.
    """
    out = _address_output(capsys, transport=render.API_TRANSPORT, stopped=True)

    assert "stopped the HTTP API" in out
    assert "/ui" not in out
    assert "API documentation" not in out


def test_stopping_an_mcp_server_says_which_kind_it_was(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _address_output(capsys, transport="http", stopped=True)

    assert "stopped the MCP server" in out


def test_starting_still_announces_where_it_is_listening(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The banner the `stopped` branch must not have taken away."""
    out = _address_output(capsys, transport=render.API_TRANSPORT, stopped=False)

    assert "HTTP API on http://127.0.0.1:8765" in out
    assert "API documentation" in out


# --- --workspace, which carries a value and so can contradict itself -------------------------


def _run_recording_workspace(argv: list[str]) -> tuple[Any, str | None]:
    """Run one invocation and report which workspace the runtime would have been opened for.

    Read at the **override** rather than at the exit status. An option that parses, renders and
    exits zero while never arriving looks exactly like one that works — the ``#60``
    ``--allow-insecure-target`` defect — and for a *tenancy* option that failure mode would run
    the operation against the wrong tenant while reporting success, which is the one outcome
    worth the most to catch.

    ``STATE`` is put back afterwards. The root callback resets it at the start of every real
    invocation, so this matters only for whatever runs next in the same process.
    """
    was_workspace, was_overrides = cli.STATE.workspace, dict(cli.STATE.overrides)
    cli.STATE.workspace = None
    cli.STATE.overrides = {}
    try:
        result = run(argv)
        return result, cli.STATE.overrides.get("workspace")
    finally:
        cli.STATE.workspace = was_workspace
        cli.STATE.overrides = was_overrides


def _asked_workspace(argv: list[str]) -> str | None:
    """The workspace one successful invocation would have run in."""
    result, workspace = _run_recording_workspace(argv)
    assert result.exit_code == 0, f"{argv} did not run: {result.output}"
    return workspace


def test_a_workspace_named_after_the_command_reaches_the_runtime(
    bound: ApplicationService,
) -> None:
    """The defect: ``manicule doctor --workspace other`` was exit 2, an unknown option.

    The same shape as ``--json``'s, one option along, and the reason it is worth fixing rather
    than documenting: the position people type is the position that failed.
    """
    del bound
    assert _asked_workspace(["doctor", "--workspace", "other"]) == "other"


def test_the_short_form_reaches_the_runtime_too(bound: ApplicationService) -> None:
    """``-w`` is half the option's interface, and an operator who learned it expects it here."""
    del bound
    assert _asked_workspace(["doctor", "-w", "other"]) == "other"


def test_a_workspace_named_before_the_command_still_reaches_the_runtime(
    bound: ApplicationService,
) -> None:
    """The documented position. Sharing the option must not have moved it."""
    del bound
    assert _asked_workspace(["--workspace", "other", "doctor"]) == "other"


def test_naming_the_same_workspace_in_both_positions_is_accepted(
    bound: ApplicationService,
) -> None:
    """Saying the same thing twice is not a contradiction.

    This is the case that makes ``--json`` twice acceptable, and it arrives the same way: a
    script that already passes the option, and somebody adding it at the other end. Refusing it
    would be a usage error over an unambiguous request.
    """
    del bound
    assert _asked_workspace(["--workspace", "same", "doctor", "--workspace", "same"]) == "same"


@pytest.mark.parametrize("second", ["--workspace", "-w"])
def test_naming_two_different_workspaces_is_refused_rather_than_resolved(
    bound: ApplicationService, second: str
) -> None:
    """The decision this option needed and ``--json`` did not.

    A flag cannot disagree with itself; a value can. Last-wins would be defensible if the two
    positions meant "general" then "specific", but by construction this is the same option in
    two places, so there is nothing to appeal to — and picking one silently would run the
    operation in a workspace the operator also named, with the envelope reporting the winner as
    though it were the request. A wrong-tenant run that looks exactly like a correct one is the
    worst outcome available here.

    Asserted through both spellings, because ``-w`` reaching a different code path than
    ``--workspace`` is precisely how a refusal ends up half-implemented.
    """
    del bound
    result, recorded = _run_recording_workspace(
        ["--workspace", "tenant-a", "doctor", second, "tenant-b"]
    )

    assert result.exit_code == 2, "a contradiction has to be a usage error, not a run"
    assert recorded != "tenant-b", "the second workspace was recorded despite the refusal"


def test_the_refusal_names_both_workspaces_so_the_typo_is_visible() -> None:
    """ "You gave it twice" without the values sends somebody to re-read their own shell history.

    Asserted against the message constant rather than the rendered box, which wraps, colours
    and elides differently on every machine.
    """
    message = cli.WORKSPACE_NAMED_TWICE.format(before="tenant-a", after="tenant-b")
    assert "tenant-a" in message
    assert "tenant-b" in message
    assert "--workspace" in message


def test_recording_a_workspace_keeps_any_other_override_already_set() -> None:
    """The bag of overrides is splatted into ``Runtime.open``, so it can hold more than one key.

    Nothing else puts a key in it today, which is exactly why this is worth pinning: the next
    value-carrying shared option will be written by analogy with ``_accept_workspace``, and a
    line that *replaces* the bag rather than adding to it would drop the workspace on the way
    past — silently, and only for the invocations that used both.

    Driven through the option's own callback, taken from the shared table rather than imported
    by name — so it is the function the command line actually reaches, not one that merely
    still exists. There is no argv that produces a second override today, and inventing a
    command-line spelling to reach it would be testing something this change did not add.
    """
    recorder = cli.SHARED_OPTIONS["--workspace"]().callback
    assert recorder is not None, "--workspace records nothing, so it reaches nothing"
    record = cast("Callable[[object, object, str | None], object]", recorder)

    was = dict(cli.STATE.overrides)
    try:
        cli.STATE.workspace = None
        cli.STATE.overrides = {"already": "set"}

        record(None, None, "other")

        assert cli.STATE.overrides == {"already": "set", "workspace": "other"}
    finally:
        cli.STATE.workspace = None
        cli.STATE.overrides = was


# --- what the colour variables mean, pinned rather than assumed --------------------------------


COLOUR_ENVIRONMENTS: tuple[tuple[str, dict[str, str], bool], ...] = (
    ("forced", {"FORCE_COLOR": "1"}, True),
    ("forced, and NO_COLOR set", {"FORCE_COLOR": "1", "NO_COLOR": "1"}, False),
    ("forced, on a dumb terminal", {"FORCE_COLOR": "1", "TERM": "dumb"}, False),
    ("forced, dumb and NO_COLOR", {"FORCE_COLOR": "1", "NO_COLOR": "1", "TERM": "dumb"}, False),
    ("NO_COLOR alone", {"NO_COLOR": "1"}, False),
    ("nothing set", {}, False),
)
"""Each colour environment, and whether manicule's human output should carry colour in it.

Read from the behaviour of Rich 14 rather than from a rule manicule imposes, because manicule
imposes none: :func:`manicule.cli.render.console` passes no colour arguments at all, so both
conventions are honoured by the library and this table is what that delegation *means*.

Two entries are the ones worth having. ``FORCE_COLOR`` with ``NO_COLOR`` is the combination
bug3 asked about, and the answer is that ``NO_COLOR`` wins — though not by overriding, which is
why the question is worth answering precisely: they are separate mechanisms, ``FORCE_COLOR``
saying the stream is a terminal and ``NO_COLOR`` stripping the colour back out of what is
written to it. ``FORCE_COLOR`` on a dumb terminal is the one that actually broke the suite, and
it is not a colour switch at all: ``TERM=dumb`` says the stream cannot render ANSI, and Rich
believes the capability over the request.
"""


@pytest.mark.parametrize(("label", "environment", "coloured"), COLOUR_ENVIRONMENTS)
def test_the_human_output_honours_both_colour_conventions(
    monkeypatch: pytest.MonkeyPatch,
    bound: ApplicationService,
    *,
    label: str,
    environment: dict[str, str],
    coloured: bool,
) -> None:
    """The precedence, pinned by what comes out rather than documented and hoped for.

    manicule delegates this entirely — it builds a console with no colour arguments — so what
    is pinned here is a dependency's behaviour, deliberately. A Rich upgrade that changed any
    row would change what an operator's ``NO_COLOR`` does, silently and without any manicule
    code being touched. This is the test that would say so.
    """
    del bound
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    result = run(["doctor"])

    assert result.exit_code == 0
    assert bool(_colours(result.stdout)) is coloured, (
        f"with {label}, colour was expected {'present' if coloured else 'absent'} and was not. "
        f"Sequences present: {sorted(set(SGR.findall(result.stdout)))}"
    )


@pytest.mark.parametrize(("label", "environment", "coloured"), COLOUR_ENVIRONMENTS)
def test_json_carries_no_ansi_in_any_colour_environment(
    monkeypatch: pytest.MonkeyPatch,
    bound: ApplicationService,
    *,
    label: str,
    environment: dict[str, str],
    coloured: bool,
) -> None:
    """The property the whole section exists to protect, across every environment above.

    ``--json`` is not a colour setting and must not behave like one: the envelope goes to
    stdout through ``sys.stdout.write`` rather than through Rich, so no colour variable can put
    a byte in it. Asserted for the coloured rows especially — those are the ones where a
    regression would actually show, and the ones a developer on a ``TERM=dumb`` terminal would
    never reproduce.
    """
    del bound, coloured
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    result = run(["doctor", "--json"])

    assert result.exit_code == 0
    assert ESCAPE not in result.stdout, f"an ANSI escape reached stdout under --json, with {label}"
    assert json.loads(result.stdout)["op"] == "doctor"


# --- the colour isolation, checked against what Rich actually reads ---------------------------


SIZE_AND_JUPYTER: dict[str, str] = {
    "COLUMNS": "terminal width, not colour. Pinned per case by the tests that assert layout; a "
    "width chosen here would quietly become the one every such assertion was written against.",
    "LINES": "terminal height, and nothing this surface prints depends on it.",
    "JUPYTER_COLUMNS": "size, and only consulted when Rich believes it is inside Jupyter.",
    "JUPYTER_LINES": "size, and only consulted when Rich believes it is inside Jupyter.",
}
"""Variables Rich reads that the colour fixture deliberately leaves alone, and why for each.

A reason apiece rather than a bare list, because "this one does not matter" is a claim. The
alternative — clearing everything Rich reads — would make this fixture decide the terminal
*size* as well, which is a different subject with its own tests already pinning it.
"""

RICH_ENVIRONMENT_FLOOR = 8
"""How many variables the scan below must find in Rich's console module.

Rich reads ten there today. The floor is under that and far above zero, because the failure
this guards against is a scan that matches nothing and reports a fixture as complete.
"""


def _variables_rich_reads() -> set[str]:
    """Every environment variable Rich's console module consults, read from its source.

    A third-party library's source, deliberately. What decides whether this suite's output is
    coloured is not manicule's code — ``render.console`` passes no colour arguments — so the
    only honest place to derive the list is the library that does decide.
    """
    import rich.console  # noqa: PLC0415 - only this derivation reads Rich's source

    source = Path(rich.console.__file__).read_text(encoding="utf-8")
    return set(re.findall(r"environ(?:\.get)?\(\s*[\"']([A-Z_0-9]+)[\"']", source))


def test_the_colour_isolation_accounts_for_every_variable_rich_reads() -> None:
    """The guard that would have caught both misses, instead of a person catching one of them.

    The fixture's list has been wrong twice. ``TERM`` was absent, which is the bug that started
    this — and it is not a colour switch, so no amount of thinking about colour would have
    surfaced it. Then the first version of the fix added ``TERM`` and still missed
    ``TTY_COMPATIBLE``, which Rich checks *before* ``FORCE_COLOR``: ``TTY_COMPATIBLE=0``
    reproduced the original failure exactly, through a fixture written to prevent it.

    Twice is a pattern, and the pattern is that a hand-written list of somebody else's
    environment variables goes stale silently. So the list is checked against Rich's source:
    every variable it consults is either cleared, pinned, or carries a written reason for being
    left alone. A Rich upgrade that starts reading an eleventh fails here until somebody says
    which it is.
    """
    read = _variables_rich_reads()

    assert len(read) >= RICH_ENVIRONMENT_FLOOR, (
        f"the scan found {len(read)} environment variable(s) in Rich's console module, below "
        f"the floor of {RICH_ENVIRONMENT_FLOOR}. It is reading the wrong file, or Rich has "
        f"restructured and this derivation no longer sees what decides colour."
    )
    for landmark in ("FORCE_COLOR", "NO_COLOR", "TERM", "TTY_COMPATIBLE"):
        assert landmark in read, (
            f"the scan did not find {landmark}, which Rich certainly reads. Whatever it "
            f"parsed, it was not the module that decides whether output is coloured."
        )

    classified = CLEARED_TERMINAL_VARIABLES | {"TERM"} | set(SIZE_AND_JUPYTER)
    unaccounted = sorted(read - classified)
    assert unaccounted == [], (
        f"Rich reads {unaccounted}, which the colour fixture neither controls nor excuses. If "
        f"it can change whether output is a terminal or is coloured, add it to "
        f"CLEARED_TERMINAL_VARIABLES in tests/conftest.py; if it cannot, add it to "
        f"SIZE_AND_JUPYTER here with the reason. Leaving it unclassified is how this suite "
        f"went back to depending on the caller's shell twice already."
    )
    assert all(SIZE_AND_JUPYTER.values()), "a variable left alone without a reason is an oversight"


# --- connector sidecar ---------------------------------------------------------------------------


ENRICHED = """<!doctype html><html><head><title>Retry Runbook</title></head><body>
<section data-source-metadata>
<p><strong>Page ID:</strong> 1002</p>
<p><strong>Source:</strong> <a href="https://docs.example.test/pages/1002">canonical page</a></p>
</section><main data-document-representation="storage"><p>Retry with backoff.</p></main>
</body></html>"""


def test_connector_sidecar_runs_end_to_end_and_renders(
    bound: ApplicationService, tmp_path: Path
) -> None:
    """The success path, exercised.

    Every other test of this feature calls the module directly, and two registrations sit between
    a working module and a working command — ``PAYLOADS`` and ``RENDERERS``. Both were missing on
    the first attempt, and both raise ``KeyError`` only when the operation *succeeds*, which is
    the path a unit test of the extractor never reaches.
    """
    del bound
    (tmp_path / "1002.html").write_text(ENRICHED, encoding="utf-8")

    result = run(["connector", "sidecar", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "1002.html.source.json").is_file()


def test_connector_sidecar_names_a_skipped_page_relative_to_the_root(
    bound: ApplicationService, tmp_path: Path
) -> None:
    """Absolute paths repeat the root on every row and push the reason off the table."""
    del bound
    (tmp_path / "plain.html").write_text("<html><body>hi</body></html>", encoding="utf-8")

    result = run(["--json", "connector", "sidecar", str(tmp_path)])

    payload = json.loads(result.output)["data"]
    assert payload["skipped"][0]["path"] == "plain.html"


def test_connector_sidecar_reports_a_missing_directory_as_a_failure(
    bound: ApplicationService, tmp_path: Path
) -> None:
    del bound
    result = run(["--json", "connector", "sidecar", str(tmp_path / "nowhere")])

    assert result.exit_code != 0
    assert json.loads(result.output)["ok"] is False


def test_connector_sidecar_accepts_a_source_and_no_root(
    bound: ApplicationService, tmp_path: Path
) -> None:
    """The flag reaches the service, which is the registration a unit test never exercises.

    ``--source`` made the positional argument optional, and an argument that stopped being
    required is exactly the kind of change Typer accepts and then fails on at call time. The
    source here is not configured, so the *refusal* is the assertion: reaching a refusal about
    the source proves the parse succeeded and the value arrived.
    """
    del bound, tmp_path

    result = run(["--json", "connector", "sidecar", "--source", "docs"])

    assert result.exit_code != 0
    body = json.loads(result.output)
    assert body["ok"] is False
    assert body["error"]["type"] == "UnknownEntityError"
    assert "configured instance" in body["error"]["message"]


def test_connector_sidecar_with_neither_a_root_nor_a_source_says_what_to_pass(
    bound: ApplicationService,
) -> None:
    """Refused rather than defaulting to the working directory, and the message names the flag."""
    del bound

    result = run(["--json", "connector", "sidecar"])

    assert result.exit_code != 0
    body = json.loads(result.output)
    assert body["ok"] is False
    assert "--source" in body["error"]["message"]


def test_connector_sidecar_reports_which_profiles_ran(
    bound: ApplicationService, tmp_path: Path
) -> None:
    """The rendered line an operator reads when a run adapted nothing.

    Without it "adapted no pages at all" is the same sentence whether the run used the built-in
    default or a source's own profiles, and those send them opposite ways.
    """
    del bound
    (tmp_path / "plain.html").write_text("<html><body>hi</body></html>", encoding="utf-8")

    result = run(["connector", "sidecar", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "standalone-storage" in result.output
    assert "no configured source" in result.output
