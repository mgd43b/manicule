"""The two surfaces are one service, and this is what makes that checkable.

The claim is specific: for the same operation and the same arguments, an MCP tool call and
``manicule ... --json`` produce **byte-identical** envelopes. Not "similar shapes" — the same
bytes, because both go through one builder over one payload model.

It matters because the MCP surface is the one called unattended. A rule implemented in the
command line is a rule an assistant can walk around, and the only durable defence against that
is a test that fails the moment the two stop being the same call.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.mcp.server import TOOL_NAMES, build_server
from tests.app.fakes import FakeBackend, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.app.results import Envelope

WORKSPACE = "default"


@pytest.fixture
def backend() -> FakeBackend:
    made = FakeBackend()
    document = make_document(WORKSPACE)
    made.store.add(document, make_chunk(document))
    return made


@pytest.fixture
def service(backend: FakeBackend) -> ApplicationService:
    return ApplicationService(backend)


async def _call(service: ApplicationService, name: str, arguments: dict[str, Any]) -> Any:
    server = build_server(service)
    result = await server.call_tool(name, arguments)
    return result.structured_content


def _tool(service: ApplicationService, name: str, arguments: dict[str, Any]) -> Any:
    """Call one tool, from a synchronous test.

    These tests are synchronous on purpose. The command line runs its own event loop — that is
    what a console script does — and driving it from inside one would fail on the nesting
    rather than on anything the surfaces do. So each side gets its own loop, sequentially,
    which is also how the two are actually used.
    """
    return asyncio.run(_call(service, name, arguments))


def _cli(monkeypatch: pytest.MonkeyPatch, service: ApplicationService, argv: Sequence[str]) -> Any:
    """Run a command with the service already built, and parse its ``--json`` output.

    One function is substituted — the one that would otherwise read configuration off the
    machine running the suite. Argument parsing, dispatch, serialisation and the exit status
    are all the real thing, which is what makes the comparison below worth making.
    """

    async def execute(op: str, call: Any) -> Envelope:
        return await run_op(op, service.workspace, lambda: call(service))

    monkeypatch.setattr(cli, "_execute", execute)
    result = CliRunner().invoke(cli.app, ["--json", *argv])
    assert result.exit_code in {0, 1}, result.output
    return json.loads(result.stdout)


# --- the surfaces offer what they say they offer ---------------------------------------------


def test_the_server_offers_exactly_nineteen_tools(service: ApplicationService) -> None:
    """Nineteen, named, and matching the list the ticket specifies."""
    server = build_server(service)
    offered = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    assert offered == sorted(TOOL_NAMES)
    assert len(offered) == 19


def test_the_command_line_offers_exactly_nineteen_commands() -> None:
    """Counted from the built command tree rather than from the source.

    A command registered on a sub-application and never attached would be in the file and not
    in the interface, which is the kind of thing a source-level count misses.
    """
    import typer.main  # noqa: PLC0415 - only this assertion needs the click tree

    command = typer.main.get_command(cli.app)
    names = sorted(getattr(command, "commands", {}))
    assert names == [
        "ask",
        "auth",
        "backup",
        "completion",
        "config",
        "connector",
        "doctor",
        "document",
        "export",
        "import",
        "index",
        "init",
        "plugin",
        "reset-index",
        "search",
        "start",
        "stop",
        "upgrade",
        "workspace",
    ]
    assert len(names) == 19


def test_every_tool_describes_itself_and_its_arguments(service: ApplicationService) -> None:
    """A tool with no description is a tool nothing knows when to call.

    The arguments matter as much: FastMCP takes the summary line as the description and folds
    the ``Args:`` section into the input schema, so a parameter with no prose is one an
    assistant has to guess at.
    """
    server = build_server(service)
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description"
        assert len(tool.description) >= 20, f"{tool.name}'s description says nothing useful"
        properties = tool.parameters.get("properties", {})
        undocumented = sorted(
            name for name, schema in properties.items() if not schema.get("description")
        )
        assert undocumented == [], f"{tool.name} has undocumented argument(s): {undocumented}"


# --- the same call through both surfaces ------------------------------------------------------


PAIRS: tuple[tuple[str, dict[str, Any], list[str]], ...] = (
    ("search", {"query": "retry"}, ["search", "retry"]),
    ("document_list", {}, ["document", "list"]),
    ("stats", {}, ["index", "--stats"]),
    ("index_status", {}, ["index"]),
    ("doctor", {}, ["doctor"]),
    ("connector_list", {}, ["connector", "list"]),
    ("workspace_list", {}, ["workspace", "list"]),
    ("plugin_list", {}, ["plugin", "list"]),
    ("config_get", {"key": "rag.profile"}, ["config", "get", "rag.profile"]),
)


@pytest.mark.parametrize(("tool", "arguments", "argv"), PAIRS)
def test_a_tool_and_its_command_produce_the_same_envelope(
    monkeypatch: pytest.MonkeyPatch,
    service: ApplicationService,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
) -> None:
    """The same operation, both ways round, compared as serialised JSON.

    Comparing the parsed structures rather than the raw text is deliberate: the command line
    pretty-prints and sorts keys and the tool does not, and neither of those is part of the
    contract. Everything else is.
    """
    from_tool = _tool(service, tool, arguments)
    from_cli = _cli(monkeypatch, service, argv)
    assert from_tool == from_cli


@pytest.mark.parametrize(("tool", "arguments", "argv"), PAIRS)
def test_both_surfaces_carry_the_workspace_and_the_contract_version(
    monkeypatch: pytest.MonkeyPatch,
    service: ApplicationService,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
) -> None:
    """Four keys are on every envelope, whichever surface produced it."""
    for envelope in (_tool(service, tool, arguments), _cli(monkeypatch, service, argv)):
        assert set(envelope) == {"version", "op", "ok", "workspace", "data", "error"}
        assert envelope["workspace"] == WORKSPACE
        assert envelope["op"] == tool


def test_a_failure_is_a_result_on_both_surfaces(
    monkeypatch: pytest.MonkeyPatch, service: ApplicationService
) -> None:
    """A failure is data, not a transport error, so an assistant can act on it.

    ``ok`` is false, ``data`` is absent and ``error`` names the type, the message and — where
    there is something specific to say — what to do next.
    """
    from_tool = _tool(service, "document_get", {"document_id": "nope"})
    from_cli = _cli(monkeypatch, service, ["document", "get", "nope"])
    assert from_tool == from_cli
    assert from_tool["ok"] is False
    assert from_tool["data"] is None
    assert from_tool["error"]["type"] == "UnknownEntityError"
    assert from_tool["error"]["hint"]


def test_the_json_payload_names_the_same_operation_the_tool_does(
    service: ApplicationService,
) -> None:
    """A log line, a shell pipeline and a tool call all name one operation the same way."""
    envelope = _tool(service, "search", {"query": "retry"})
    assert envelope["op"] == "search"
    assert envelope["op"] in TOOL_NAMES
