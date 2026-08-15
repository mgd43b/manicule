"""What each tool tells a client it will do, checked against what it does.

Read over the protocol, from ``tools/list``, rather than off the decorator arguments. A test
that read the arguments would pass on a server that computed its annotations correctly and then
failed to publish them, which is the one failure that matters here: an annotation a client never
receives buys nothing, and an annotation a client receives is acted on without asking.

**The load-bearing test in this file is not the one that checks the four hints against a table.**
It is :func:`test_a_tool_that_says_it_reads_leaves_the_installation_as_it_found_it`, which calls
every tool that claims to be read-only and compares the whole backend before and after. That one
cannot be satisfied by a plausible-looking annotation, and it is the reason the annotations are
worth publishing at all: a client that auto-approves ``search`` on the strength of
``readOnlyHint`` is trusting this assertion, not the decorator.

Two fields are excluded from that comparison **by name**, and each is a decision rather than a
tolerance. ``FakeRetriever.seen`` is test instrumentation — a record that a query reached a
retriever, which is how other suites prove one did *not*. ``FakeTelemetry.queries`` is the query
log, and it is the single write retrieval performs; ``manicule.app.dispatch`` has carried that
exception since before this file existed, ``manicule.mcp.server.hints`` restates it where the
annotation is made, and ``docs/surfaces.md`` §4.1 says what it costs. Anything else moving fails
here.
"""

from __future__ import annotations

import copy
import inspect
import json
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client
from fastmcp.tools.function_tool import FunctionTool

from manicule.app.service import ApplicationService
from manicule.mcp.server import TOOL_NAMES, build_server
from tests.app.fakes import FakeBackend, make_chunk, make_document
from tests.mcp.qualification import COLLECTION, build_fixture

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import Tool

WORKSPACE = "default"

HINTS: tuple[str, ...] = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")

TICKETED: tuple[str, ...] = (
    "collection_list",
    "collection_counts",
    "collection_documents",
    "search",
)
"""The four operations #01 requires by name. Everything else was audited, not assumed."""

MUTATIONS: tuple[str, ...] = (
    "index_path",
    "document_delete",
    "document_reindex",
    "connector_sync",
    "config_set",
    "workspace_switch",
    "plugin_add",
    "plugin_remove",
    "collection_create",
    "collection_rename",
    "collection_update",
    "collection_delete",
    "collection_add",
    "collection_remove",
    "ask",
)
"""Every tool that changes something, named so the negative is a list rather than a leftover.

``ask`` is here and it is the entry worth pausing on. It reads the corpus like ``search`` does,
and it is not read-only for two independent reasons: given a ``conversation_id`` it persists the
turn (``manicule.generation.answering``), and the model it calls may be a provider on another
machine. A classification taken from the name, or from
:data:`~manicule.app.dispatch.READ_ONLY_OPS` — which answers a different question, "does this
need the writer's lock", and answers *yes it is read-only* for ``ask``, ``backup`` and ``init``
— would have got this one wrong.
"""


async def _tools(service: ApplicationService) -> list[Tool]:
    """``tools/list``, as a client receives it."""
    async with Client(build_server(service)) as client:
        return list(await client.list_tools())


async def _instructions(service: ApplicationService) -> str:
    """Server guidance from the initialization result, as a client receives it."""
    async with Client(build_server(service)) as client:
        result = client.initialize_result
    assert result is not None, "the client never completed initialization"
    return result.instructions or ""


@pytest.fixture
def service() -> ApplicationService:
    backend = FakeBackend()
    document = make_document(WORKSPACE)
    backend.store.add(document, make_chunk(document))
    backend.organization_.documents[document.id] = document
    return ApplicationService(backend)


async def test_the_four_operations_the_ticket_names_report_themselves_read_only(
    service: ApplicationService,
) -> None:
    """All four hints, in the protocol, with the values #01 specifies."""
    found = {tool.name: tool.annotations for tool in await _tools(service)}
    for name in TICKETED:
        annotations = found[name]
        assert annotations is not None, f"{name} publishes no annotations at all"
        assert annotations.readOnlyHint is True, name
        assert annotations.destructiveHint is False, name
        assert annotations.idempotentHint is True, name
        assert annotations.openWorldHint is False, name


async def test_every_registered_tool_answers_all_four_questions(
    service: ApplicationService,
) -> None:
    """A tool with no annotations is a tool a client has to guess about.

    Asserted over the whole surface rather than over a list, so a tool added tomorrow fails here
    until somebody decides what it does. That is the entire mechanism keeping the annotations
    honest: they live on the registrations, and there is no second table to update.
    """
    published = await _tools(service)
    assert sorted(tool.name for tool in published) == sorted(TOOL_NAMES)
    for tool in published:
        assert tool.annotations is not None, f"{tool.name} publishes no annotations"
        undecided = [hint for hint in HINTS if getattr(tool.annotations, hint) is None]
        assert undecided == [], f"{tool.name} leaves {undecided} unanswered"


async def test_the_protocol_explains_how_to_judge_and_recover_from_retrieval(
    service: ApplicationService,
) -> None:
    """A fresh client can act on weak or partial evidence without guessing from a float.

    Read the initialization result rather than :data:`manicule.mcp.server.INSTRUCTIONS`, because
    guidance assembled in the process and never sent across the protocol buys the client
    nothing. The assertions name machine fields rather than pinning paragraphs: wording may
    improve, but losing one of the decisions would make a plausible-looking result ambiguous.
    """
    instructions = " ".join((await _instructions(service)).split())
    for field in (
        "confidence_band",
        "confidence_reason",
        "collections",
        "truncated",
        "corpus_consulted",
        "ungrounded",
        "context_truncated",
        "dropped",
    ):
        assert field in instructions, f"server instructions do not explain {field}"
    assert "not probabilities" in instructions
    assert "Treat `none` and `low` as insufficient support" in instructions
    assert "not that the corpus does not" in instructions
    assert "Never remove a scope" in instructions
    assert "one retry with `precise`" in instructions


async def test_retrieval_tools_publish_their_result_decisions_in_tools_list(
    service: ApplicationService,
) -> None:
    """Tool selection can be correct even when a client ignores server-wide instructions.

    Some clients present only ``tools/list`` metadata to the model choosing a call. Checking the
    published descriptions keeps the recovery contract at that boundary instead of proving
    only that the Python docstrings contain it.
    """
    descriptions = {
        tool.name: " ".join((tool.description or "").split()) for tool in await _tools(service)
    }
    for field in ("confidence_band", "confidence_reason", "collections", "truncated"):
        assert field in descriptions["search"], f"search does not explain {field}"
    for field in ("corpus_consulted", "ungrounded", "context_truncated", "dropped"):
        assert field in descriptions["ask"], f"ask does not explain {field}"
    assert "not the probability" in descriptions["search"]
    assert "not the probability" in descriptions["ask"]


async def test_no_tool_that_changes_something_reports_itself_read_only(
    service: ApplicationService,
) -> None:
    """The negative, named. A wrong ``true`` here is worse than no annotation at all."""
    found = {tool.name: tool.annotations for tool in await _tools(service)}
    for name in MUTATIONS:
        annotations = found[name]
        assert annotations is not None, name
        assert annotations.readOnlyHint is False, f"{name} claims to be read-only and is not"


async def test_the_mutations_and_the_reads_together_are_the_whole_surface(
    service: ApplicationService,
) -> None:
    """No tool is in neither list, so :data:`MUTATIONS` cannot go stale quietly.

    Without this, adding a tool and forgetting to classify it leaves every assertion above
    passing — each is a statement about the tools it names, and an unnamed one is named nowhere.
    """
    reads = {
        tool.name
        for tool in await _tools(service)
        if tool.annotations is not None and tool.annotations.readOnlyHint
    }
    assert reads | set(MUTATIONS) == set(TOOL_NAMES)
    assert reads & set(MUTATIONS) == set()


def _installation(backend: FakeBackend) -> FakeBackend:
    """Everything the fakes hold, with the two moving parts named in this module's docstring
    cleared so that two snapshots of an unchanged installation compare equal."""
    copied = copy.deepcopy(backend)
    copied.retriever_.seen.clear()
    copied.telemetry_.queries.clear()
    return copied


async def test_a_tool_that_says_it_reads_leaves_the_installation_as_it_found_it() -> None:
    """Every read-only tool called for real, and the whole backend compared across the call.

    The arguments below are written down, and a read-only tool missing from them fails rather
    than being skipped — a tool this cannot call is a tool whose annotation nothing has checked.

    **What this does not catch, said plainly.** The comparison is over the *backend*, so a tool
    that wrote to the filesystem rather than through a store would pass it. Two tools reach
    outside the fakes today and both were read instead: ``config_get`` resolves a path and
    redacts settings already in memory, and ``doctor`` stats the data directory without
    creating anything the runtime had not already created by opening it. A third case is
    excluded by argument rather than by luck — ``plugin_list`` is called without ``registry``,
    which is the only branch of it that opens a connection.
    """
    service, backend = await build_fixture()
    document_id = next(iter(backend.store.documents))
    collection_id = next(iter(backend.organization_.collections))

    arguments: dict[str, dict[str, Any]] = {
        "search": {"query": "admission control", "collections": [COLLECTION], "limit": 3},
        "document_list": {},
        "document_get": {"document_id": document_id, "chunks": True},
        "index_status": {},
        "stats": {},
        "doctor": {},
        "connector_list": {},
        "config_get": {},
        "workspace_list": {},
        # `registry` left off deliberately. It is why this tool's `openWorldHint` is true, and
        # a test that fetched a community listing would be a test with a network dependency.
        "plugin_list": {},
        "collection_list": {},
        "collection_documents": {"collection_id": collection_id},
        "collection_counts": {"collection_id": collection_id},
    }

    async with Client(build_server(service)) as client:
        published = list(await client.list_tools())
        reads = [
            tool.name
            for tool in published
            if tool.annotations is not None and tool.annotations.readOnlyHint
        ]
        unexercised = sorted(set(reads) - set(arguments))
        assert unexercised == [], f"no arguments recorded for read-only tool(s): {unexercised}"

        for name in reads:
            before = _installation(backend)
            envelope = (await client.call_tool(name, arguments[name])).structured_content or {}
            assert envelope.get("ok") is True, f"{name} failed: {envelope.get('error')}"
            assert _installation(backend) == before, (
                f"{name} says it reads and it changed something"
            )


async def test_retrieval_writes_exactly_the_one_row_the_annotation_admits_to() -> None:
    """The exception, watched rather than asserted in prose.

    ``search`` carries ``readOnlyHint: true`` and does append to the query log. Nailing that
    down means proving the write happens — so the exception is real and documented rather than
    theoretical — and that it is the *only* one, which is what the comparison above establishes.
    """
    service, backend = await build_fixture()
    assert backend.telemetry_.queries == []

    async with Client(build_server(service)) as client:
        await client.call_tool("search", {"query": "admission control", "limit": 3})

    assert len(backend.telemetry_.queries) == 1, (
        "the query-log append is what `hints(reads=...)` names as retrieval's one exception; "
        "if it has gone, the exception in the documentation should go with it"
    )
    assert backend.telemetry_.audits == []


async def test_no_hint_leaked_into_an_input_schema(service: ApplicationService) -> None:
    """An annotation describes a tool; it is not part of the call.

    Both halves are checked: no hint name appears anywhere in a schema, and each schema's
    properties are exactly the tool function's own parameters — so nothing was added, and
    nothing was dropped either. The *result* envelope is held to the same claim one file over,
    where ``tests/app/test_surface_parity.py`` compares an MCP call byte for byte against the
    command line and the HTTP route for ``search``, ``collection_list``, ``collection_counts``
    and ``collection_documents``.
    """
    server = build_server(service)
    async with Client(server) as client:
        published = list(await client.list_tools())

    for tool in published:
        serialized = json.dumps(tool.inputSchema)
        for hint in HINTS:
            assert hint not in serialized, f"{tool.name}'s input schema mentions {hint}"
        registered = await server.get_tool(tool.name)
        assert isinstance(registered, FunctionTool)
        expected = sorted(inspect.signature(registered.fn).parameters)
        assert sorted(tool.inputSchema.get("properties", {})) == expected, tool.name


async def test_nothing_on_this_server_is_gated_on_its_own_annotation(
    service: ApplicationService,
) -> None:
    """Authorization stays with the operator, so a mutation runs when it is called.

    The hints are a description. A server that started refusing calls on the strength of them
    would have invented an approval mechanism inside application code — which is the thing #01
    forbids, and which would also be trivially bypassed by the HTTP surface and the command
    line, since neither has annotations at all.
    """
    async with Client(build_server(service)) as client:
        created = (
            await client.call_tool("collection_create", {"name": "Engineering Architecture"})
        ).structured_content or {}
    assert created["ok"] is True
    assert created["op"] == "collection_create"


def test_the_server_ships_no_blanket_approval_setting() -> None:
    """No ``approve``, no ``trusted``, no auto-approval default anywhere in the MCP package.

    Read out of the source because the claim is an absence, and an absence has no runtime
    behavior to observe. What a client approves is the client's decision and the operator's
    policy; a server that shipped a default answer to it would be answering for both.
    """
    from pathlib import Path  # noqa: PLC0415 - only this assertion reads the tree

    import manicule.mcp  # noqa: PLC0415 - located rather than imported for behavior

    package = Path(str(next(iter(manicule.mcp.__path__))))
    banned = ("auto_approve", "autoapprove", "always_allow", "trusted_tools")
    for module in sorted(package.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        for word in banned:
            assert word not in source, f"{module.name} names {word!r}"


async def test_a_client_can_tell_the_reads_from_the_writes_without_reading_a_document(
    service: ApplicationService,
) -> None:
    """The whole point, stated as the thing an operator wanted and could not have.

    Before this, ``tools/list`` said nothing about side effects, so approving ``search`` meant
    approving the server — and the server also deletes documents, rewrites configuration and
    enables plugins. One pass over the published metadata now separates them.
    """
    published = await _tools(service)
    reads: Sequence[str] = [
        tool.name
        for tool in published
        if tool.annotations is not None and tool.annotations.readOnlyHint
    ]
    assert "search" in reads
    assert "collection_list" in reads
    assert "collection_counts" in reads
    assert "document_delete" not in reads
    assert "config_set" not in reads
    assert "plugin_add" not in reads
