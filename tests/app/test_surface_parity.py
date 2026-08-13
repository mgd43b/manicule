"""The three surfaces are one service, and this is what makes that checkable.

The claim is specific: for the same operation and the same arguments, an MCP tool call, an
HTTP request and ``manicule ... --json`` produce **byte-identical** envelopes. Not "similar
shapes" — the same bytes, because all three go through one builder over one payload model.

It matters because two of the three are called unattended. A rule implemented in the command
line is a rule an assistant can walk around; a rule implemented in a route is one the MCP tool
does not have. The only durable defence is a test that fails the moment they stop being the
same call — so when a third surface arrived, this file grew a third column rather than a
parallel file with its own idea of what parity means.

**The browser surface is the fourth column**, and it is a different kind of claim. A page is
HTML, so it cannot be compared byte for byte with an envelope; what *is* asserted is that the
page's content came from that envelope — a value the tool reported is found in the page, and a
failure the tool reports is the failure the page shows, with the same type and the same message.
That is the property that would break if a page ever computed something of its own, which is the
only thing this file has ever been for.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

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


COLLECTION = "runbooks-alpha"
"""The collection the fixture seeds, and the marker the browser column looks for.

Named here because three columns need the same string and a page that renders a *different*
collection would still contain the word "collection". Distinctive enough that finding it in
the HTML means the page rendered this envelope rather than coincidence.
"""


@pytest.fixture
def backend() -> FakeBackend:
    made = FakeBackend()
    document = make_document(WORKSPACE)
    made.store.add(document, make_chunk(document))
    # Seeded so the collections page has something to render. Two of the rows below carry no
    # page precisely because this fixture leaves their surface empty; collections deliberately
    # does not join them — a group whose page is never asserted against is a page free to stop
    # showing what the envelope says.
    made.organisation_.documents[document.id] = document
    asyncio.run(made.organisation_.create_collection(COLLECTION, description="worked examples"))
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


def test_the_server_offers_exactly_twenty_eight_tools(service: ApplicationService) -> None:
    """Twenty-eight, named, and matching the list the ticket specifies."""
    server = build_server(service)
    offered = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    assert offered == sorted(TOOL_NAMES)
    assert len(offered) == 28


def test_no_tool_moves_documents_out_of_the_corpus_wholesale() -> None:
    """``collection_orphans`` is command line only, and this is what keeps it there.

    It moves every document outside every collection into the trash. That is the one
    collection operation that destroys data, so it stays on the surface where a person is
    present — the rule ``reset-index``, ``backup`` and ``import`` are already held to. An
    absence with no test is an absence that comes back.
    """
    assert "collection_orphans" not in TOOL_NAMES


def test_no_tool_writes_files_into_the_users_corpus() -> None:
    """``connector_sidecar`` is command line only, by the same rule.

    It is the one operation that writes to the corpus *directory* rather than to the index — a
    manifest beside every page under a root the caller names. Everything else manicule does to a
    corpus is read-only, so an unattended surface that could write into one is a new kind of
    authority rather than a new operation, and it stays where a person is present.
    """
    assert "connector_sidecar" not in TOOL_NAMES


def test_the_command_line_offers_exactly_twenty_commands() -> None:
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
        "collection",
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
    assert len(names) == 20


def test_only_the_command_line_can_ask_doctor_to_repair_anything(
    service: ApplicationService,
) -> None:
    """Parity is about the same call producing the same answer, not about every surface being
    able to do everything — and this is the one asymmetry, deliberately.

    ``doctor --fix`` seeds the declared code grammars: it writes to the machine and may fetch
    from the network. That is a thing an operator asks for. An assistant holding a tool call and
    an authenticated HTTP client both reach ``doctor`` as a *report*, and a health endpoint that
    could be made to start an 80 MB download by adding a query parameter is a health endpoint
    with a side effect nobody asked for. So the argument exists on the service and is passed by
    exactly one surface.
    """
    import inspect  # noqa: PLC0415 - only this assertion reads a signature

    import typer.main  # noqa: PLC0415 - only this assertion needs the click tree

    from manicule.api.routes import health  # noqa: PLC0415 - keeps FastAPI out of the CLI path

    assert "fix" in inspect.signature(service.doctor).parameters

    offered = asyncio.run(build_server(service).list_tools())
    tool = next(tool for tool in offered if tool.name == "doctor")
    assert "fix" not in tool.parameters.get("properties", {})
    assert "fix" not in inspect.signature(health.health).parameters

    commands: dict[str, object] = getattr(typer.main.get_command(cli.app), "commands", {})
    params: list[object] = getattr(commands["doctor"], "params", [])
    assert "--fix" in {str(opt) for parameter in params for opt in getattr(parameter, "opts", [])}


def test_the_repair_flag_reaches_the_service_rather_than_merely_existing(
    service: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half a declaration cannot prove: that ``--fix`` is *wired*, not just offered.

    The assertion above establishes that exactly one surface offers the flag. It stays green
    if the option is declared and never passed — a flag an operator types, that reports a
    healthy-looking diagnosis, and repairs nothing. Found by disabling the pass-through and
    watching that test stay green, which is the only way this kind of gap is ever found.

    Through the real Typer command and the real dispatch, with only the service's own method
    recorded, so what is observed is the argument arriving rather than a mock being called.
    """
    from manicule.app import results  # noqa: PLC0415 - only this assertion needs the model

    asked: list[bool] = []

    async def record(*, fix: bool = False) -> results.Diagnosis:
        asked.append(fix)
        return results.Diagnosis(state="ok")

    monkeypatch.setattr(service, "doctor", record)

    _cli(monkeypatch, service, ["doctor", "--fix"])
    _cli(monkeypatch, service, ["doctor"])

    assert asked == [True, False], "the flag is declared and does not reach the service"


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


type Envelopes = dict[str, Any]
"""One serialised envelope, as any of the three surfaces produced it."""

type HttpCall = tuple[str, str, dict[str, Any]] | None
"""A method, a path and request keyword arguments — or ``None`` for an operation with no route."""

type WebPage = tuple[str, tuple[str | int, ...]] | None
"""A page of the browser surface and a key path into the payload it must render.

``None`` where this operation has no page, or where the fixture produces nothing for a page to
show — an empty connector list renders "no connectors are configured", which is the right page
and carries no value from the envelope to assert on.
"""


def _http(service: ApplicationService, method: str, path: str, **kwargs: Any) -> Envelopes:
    """Run one request against the **real** application and parse its envelope.

    The production `build_app`, over the same service the other two surfaces are driving. A
    helper that called the service directly would compare the service with itself.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415 - only the HTTP column needs it

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI out of the CLI path

    with TestClient(build_app(service), client=("127.0.0.1", 41234)) as client:
        body: Envelopes = client.request(method, path, **kwargs).json()
        return body


PAIRS: tuple[tuple[str, dict[str, Any], list[str], HttpCall, WebPage], ...] = (
    (
        "search",
        {"query": "retry"},
        ["search", "retry"],
        ("GET", "/api/v1/search", {"params": {"q": "retry"}}),
        ("/ui/search?q=retry", ("confidence_reason",)),
    ),
    (
        "document_list",
        {},
        ["document", "list"],
        ("GET", "/api/v1/documents", {}),
        ("/ui/documents", ("documents", 0, "id")),
    ),
    ("stats", {}, ["index", "--stats"], ("GET", "/api/v1/stats", {}), ("/ui", ("by_media_type",))),
    (
        "index_status",
        {},
        ["index"],
        ("GET", "/api/v1/admin/stats", {}),
        ("/ui/settings", ("data_dir",)),
    ),
    (
        "doctor",
        {},
        ["doctor"],
        ("GET", "/api/v1/health", {}),
        ("/ui/health", ("checks", 1, "detail")),
    ),
    ("connector_list", {}, ["connector", "list"], ("GET", "/api/v1/admin/connectors", {}), None),
    (
        "workspace_list",
        {},
        ["workspace", "list"],
        ("GET", "/api/v1/workspaces", {}),
        ("/ui/workspaces", ("workspaces", 0, "mode")),
    ),
    ("plugin_list", {}, ["plugin", "list"], ("GET", "/api/v1/plugins", {}), None),
    ("config_get", {"key": "rag.profile"}, ["config", "get", "rag.profile"], None, None),
    (
        "collection_list",
        {},
        ["collection", "list"],
        ("GET", "/api/v1/collections", {}),
        ("/ui/collections", ("collections", 0, "name")),
    ),
    (
        "collection_counts",
        {"collection_id": "col-0"},
        ["collection", "counts", "col-0"],
        ("GET", "/api/v1/collections/col-0/counts", {}),
        None,
    ),
)
"""One row per operation: the MCP tool, the command, the HTTP request, and the page.

``config_get`` has neither an HTTP column nor a page, and that is a decision rather than an
omission: reading and writing configuration over the network is how an installation gets
repointed at a different data directory by something holding a key. It stays on the command line
and the MCP tool, and the browser surface's settings area shows the installation's *posture*
from ``doctor`` and ``index_status`` instead — which is why those two rows have pages.

``connector_list`` and ``plugin_list`` have pages and no page column: this fixture configures no
connectors and installs no plugins, so their pages correctly render "there are none" and there
is no value from the envelope to assert on. ``tests/web/test_pages.py`` covers that they answer.
"""

VOLATILE: frozenset[str] = frozenset({"elapsed_ms", "checked_at"})
"""The fields that legitimately differ between two runs of the same operation.

Excluded **by name** rather than by tolerance, and the distinction is the whole value of this
harness. A comparison that ignored whatever happened to differ would ignore a real divergence
too — it would pass on a surface that had quietly stopped reporting a field, which is the
defect this test exists to catch.

Both entries are clocks. ``elapsed_ms`` is how long the operation took; ``checked_at`` is when
``doctor`` took the diagnosis. Neither can be equal across three separate runs of the same
operation, and neither is a thing the surfaces could disagree about *in kind*: whatever they
report, they report the same field with the same meaning, which is what parity is about.
"""


def _comparable(envelope: Envelopes) -> Envelopes:
    """One envelope, with the clocks removed, so two runs of one operation can be equal."""
    payload = envelope.get("data")
    if not isinstance(payload, dict):
        return envelope
    typed = cast("dict[str, Any]", payload)
    return {**envelope, "data": {key: value for key, value in typed.items() if key not in VOLATILE}}


@pytest.mark.parametrize(("tool", "arguments", "argv", "request_", "page"), PAIRS)
def test_a_tool_and_its_command_produce_the_same_envelope(
    monkeypatch: pytest.MonkeyPatch,
    service: ApplicationService,
    *,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
    request_: HttpCall,
    page: WebPage,
) -> None:
    """The same operation, every way round, compared as serialised JSON.

    Comparing the parsed structures rather than the raw text is deliberate: the command line
    pretty-prints and sorts keys and the others do not, and neither of those is part of the
    contract. Everything else is.
    """
    del request_, page
    from_tool = _tool(service, tool, arguments)
    from_cli = _cli(monkeypatch, service, argv)
    assert _comparable(from_tool) == _comparable(from_cli)


@pytest.mark.parametrize(("tool", "arguments", "argv", "request_", "page"), PAIRS)
def test_the_http_surface_produces_the_same_envelope_as_the_tool(
    service: ApplicationService,
    *,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
    request_: HttpCall,
    page: WebPage,
) -> None:
    """The third column. A route that decided anything of its own fails here."""
    del argv, page
    if request_ is None:
        pytest.skip("this operation is deliberately not on the HTTP surface")
    method, path, kwargs = request_
    from_tool = _tool(service, tool, arguments)
    from_http = _http(service, method, path, **kwargs)
    assert _comparable(from_tool) == _comparable(from_http)


@pytest.mark.parametrize(("tool", "arguments", "argv", "request_", "page"), PAIRS)
def test_every_surface_carries_the_workspace_and_the_contract_version(
    monkeypatch: pytest.MonkeyPatch,
    service: ApplicationService,
    *,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
    request_: HttpCall,
    page: WebPage,
) -> None:
    """Four keys are on every envelope, whichever surface produced it."""
    del page
    produced = [_tool(service, tool, arguments), _cli(monkeypatch, service, argv)]
    if request_ is not None:
        method, path, kwargs = request_
        produced.append(_http(service, method, path, **kwargs))
    for envelope in produced:
        assert set(envelope) == {"version", "op", "ok", "workspace", "data", "error"}
        assert envelope["workspace"] == WORKSPACE
        assert envelope["op"] == tool


def _web(service: ApplicationService, path: str) -> str:
    """Render one page of the browser surface, through the **real** application.

    The production ``build_app`` again, so the page goes through the same routing, the same
    principal resolution and the same middleware a browser would meet.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415 - only this column needs it

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI out of the CLI path

    with TestClient(build_app(service), client=("127.0.0.1", 41234)) as client:
        response = client.get(path)
        assert response.status_code == 200, response.text
        return response.text


def _leaf(payload: Any, keys: tuple[str | int, ...]) -> str:
    """One value out of a payload, as the page would render it.

    A counter table is rendered as its **keys** — ``by_media_type`` shows ``text/markdown``, not
    the number beside it — so a path ending at a mapping yields its first key. Anything else is
    stringified.
    """
    value: Any = payload
    for key in keys:
        value = value[key]
    if isinstance(value, dict):
        return str(next(iter(cast("dict[str, Any]", value))))
    return str(value)


@pytest.mark.parametrize(("tool", "arguments", "argv", "request_", "page"), PAIRS)
def test_the_browser_surface_renders_the_tools_envelope(
    service: ApplicationService,
    *,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
    request_: HttpCall,
    page: WebPage,
) -> None:
    """The fourth column: a value the tool reported is on the page.

    Not a byte comparison — a page is HTML — but the same claim in the form HTML can carry. The
    marker is read out of the tool's envelope at run time rather than written down, so a page
    that started computing its own version of a field fails here rather than passing against a
    literal somebody updated to match.
    """
    del argv, request_
    if page is None:
        pytest.skip("this operation has no page, or this fixture gives its page nothing to show")
    path, keys = page
    marker = _leaf(_tool(service, tool, arguments)["data"], keys)
    assert len(marker) >= 4, (
        f"the marker for {tool} is {marker!r}, which is short enough to appear in a page by "
        f"coincidence. Choose a field whose value is distinctive."
    )
    trail = ".".join(str(key) for key in keys)
    assert marker in _web(service, path), f"{path} does not render {tool}'s {trail}"


def test_a_page_reports_a_failure_exactly_as_the_tool_does(service: ApplicationService) -> None:
    """The same operation, the same failure, on a surface that renders HTML.

    A page that caught the error and wrote its own sentence would be a second description of
    what went wrong — and the hint is usually the thing that fixes it, so losing it costs
    something real.
    """
    from markupsafe import escape  # noqa: PLC0415 - only this column renders HTML

    envelope = _tool(service, "document_get", {"document_id": "nope"})
    body = _web_status(service, "/ui/documents/nope")
    assert envelope["ok"] is False
    assert envelope["error"]["type"] in body
    # Escaped, because the page escapes everything — including an error message, which quotes
    # the identifier the caller sent and is therefore attacker-controlled text like any other.
    assert str(escape(envelope["error"]["message"])) in body
    assert str(escape(envelope["error"]["hint"])) in body


def _web_status(service: ApplicationService, path: str) -> str:
    """A page that is expected to fail, with the status the error's type implies."""
    from fastapi.testclient import TestClient  # noqa: PLC0415 - only this column needs it

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI out of the CLI path

    with TestClient(build_app(service), client=("127.0.0.1", 41234)) as client:
        response = client.get(path)
        assert response.status_code == 404, response.status_code
        return response.text


def test_a_failure_is_a_result_on_every_surface(
    monkeypatch: pytest.MonkeyPatch, service: ApplicationService
) -> None:
    """A failure is data, not a transport error, so an assistant can act on it.

    ``ok`` is false, ``data`` is absent and ``error`` names the type, the message and — where
    there is something specific to say — what to do next. The HTTP surface additionally
    carries a status code, and the **body is still the envelope**, so a client that reads
    ``ok`` first never has two shapes to parse.
    """
    from_tool = _tool(service, "document_get", {"document_id": "nope"})
    from_cli = _cli(monkeypatch, service, ["document", "get", "nope"])
    from_http = _http(service, "GET", "/api/v1/documents/nope")
    assert from_tool == from_cli == from_http
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


# --- source metadata, through every surface ---------------------------------------------------


SOURCE_PAIRS: tuple[tuple[str, dict[str, Any], list[str], HttpCall, tuple[str | int, ...]], ...] = (
    (
        "search",
        {"query": "retry"},
        ["search", "retry"],
        ("GET", "/api/v1/search", {"params": {"q": "retry"}}),
        ("hits", 0, "provenance"),
    ),
    (
        "document_list",
        {},
        ["document", "list"],
        ("GET", "/api/v1/documents", {}),
        ("documents", 0, "provenance"),
    ),
)
"""The operations that report a document's source metadata, and where it sits in each payload.

``ask`` is deliberately absent: the fake backend answers without a model, so there is no answer
envelope to compare across surfaces here. ``tests/app/test_service.py`` covers the citation's own
copy of this block.
"""


def _service_ranking_one_document(*, with_record: bool) -> ApplicationService:
    """A service whose single document is also the one thing retrieval ranks.

    The module's own ``backend`` fixture leaves the fake retriever with no candidates, so
    ``search`` there renders no hits at all — which is fine for comparing envelopes and useless
    for asserting anything about a hit's contents. Both source-metadata cases need a hit, so both
    come from here and differ in one argument.
    """
    from manicule.core.provenance import LocalSnapshot, Provenance, SourceMetadata  # noqa: PLC0415
    from manicule.core.retrieval import Candidate  # noqa: PLC0415 - only this helper ranks one

    record = (
        Provenance(
            source=SourceMetadata(
                title="Retry policy",
                canonical_uri="https://docs.example.test/pages/123456/retry-policy",
                source_id="123456",
                version="7",
                section_path=("Engineering", "Runbooks"),
            ),
            snapshot=LocalSnapshot(path="mirror/123456.html"),
        )
        if with_record
        else None
    )
    made = FakeBackend()
    document = make_document(
        WORKSPACE, source_id="123456.html", title="Retry policy", provenance=record
    )
    chunk = make_chunk(document)
    made.store.add(document, chunk)
    made.retriever_.candidates = [Candidate(chunk=chunk, score=0.5)]
    return ApplicationService(made)


@pytest.fixture
def sourced() -> ApplicationService:
    """A service whose one document carries authoritative source metadata.

    A fixture of its own rather than an extension of the module's, so that every other test in
    this file keeps running against a document with **no** record — that is the case which has to
    stay unchanged, and a shared fixture would quietly stop covering it.
    """
    return _service_ranking_one_document(with_record=True)


@pytest.mark.parametrize(("tool", "arguments", "argv", "request_", "keys"), SOURCE_PAIRS)
def test_source_metadata_serialises_identically_on_every_surface(
    monkeypatch: pytest.MonkeyPatch,
    sourced: ApplicationService,
    *,
    tool: str,
    arguments: dict[str, Any],
    argv: list[str],
    request_: HttpCall,
    keys: tuple[str | int, ...],
) -> None:
    """One record, four ways out, and the same bytes each time.

    The block is nested inside a payload, which is the thing worth checking rather than assuming:
    a surface that flattened it, dropped it because it was optional, or serialised a
    :class:`~datetime.datetime` its own way would produce a consumer-visible difference that the
    envelope comparison at the top of this file would catch only if some row happened to carry
    one. None did until this fixture.

    The three timestamp fields are the specific hazard. They are ``str | None`` on the payload and
    ``datetime | None`` on the record, so exactly one place converts them; a second converter
    anywhere would give one surface an offset and another a ``Z``.
    """
    from_tool = _tool(sourced, tool, arguments)
    from_cli = _cli(monkeypatch, sourced, argv)
    produced = [from_tool, from_cli]
    if request_ is not None:
        method, path, kwargs = request_
        produced.append(_http(sourced, method, path, **kwargs))

    blocks = [_at(envelope["data"], keys) for envelope in produced]
    assert blocks[0] is not None, (
        f"{tool} reported no source metadata at {'.'.join(str(key) for key in keys)}, so this "
        f"test is comparing four copies of null"
    )
    assert blocks[0]["canonical_uri"] == "https://docs.example.test/pages/123456/retry-policy"
    assert blocks[0]["source_id"] == "123456"
    assert blocks[0]["version"] == "7"
    assert blocks[0]["section_path"] == ["Engineering", "Runbooks"]
    assert blocks[0]["snapshot_path"] == "mirror/123456.html"
    assert blocks[0]["snapshot_checksum"], "the local snapshot's digest is part of the citation"
    for other in blocks[1:]:
        assert other == blocks[0]


def test_the_canonical_identity_and_the_local_one_are_both_on_the_wire(
    sourced: ApplicationService,
) -> None:
    """Neither identity is dropped in favour of the other on the way out.

    ``title`` and ``uri`` on the hit are the canonical ones, because that is what a citation
    should show. The local snapshot is still reachable in the same payload — its path in the
    record, its digest beside it, and the file it was fetched by in ``source_id`` on the document
    summary. A consumer auditing what was actually read never has to go and ask a second
    question.
    """
    hit = _tool(sourced, "search", {"query": "retry"})["data"]["hits"][0]
    summary = _tool(sourced, "document_list", {})["data"]["documents"][0]

    assert hit["title"] == "Retry policy"
    assert hit["provenance"]["snapshot_path"] == "mirror/123456.html"
    assert summary["source_id"] == "123456.html", "the local artefact it was fetched by"
    assert summary["provenance"]["canonical_uri"].startswith("https://docs.example.test/")


def test_a_document_with_no_source_metadata_reports_null_rather_than_an_empty_block() -> None:
    """Absent, not blank. "There is no canonical address" is not "the address is empty".

    A consumer branching on presence gets one thing to look at rather than a shape it has to
    inspect field by field to find out whether anything is in it. This is also the assertion that
    keeps the ordinary local file — the overwhelming majority of any corpus — visibly on the old
    path rather than on a new one that merely produces blanks.
    """
    plain = _service_ranking_one_document(with_record=False)
    hit = _tool(plain, "search", {"query": "retry"})["data"]["hits"][0]
    summary = _tool(plain, "document_list", {})["data"]["documents"][0]

    assert hit["provenance"] is None
    assert summary["provenance"] is None
    assert hit["title"] == "Retry policy", "the fixture must still have produced a real hit"


def _at(payload: Any, keys: tuple[str | int, ...]) -> Any:
    """One nested value out of a payload, or ``None`` if the path does not resolve."""
    value: Any = payload
    for key in keys:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            return None
    return value
