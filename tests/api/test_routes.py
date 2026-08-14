"""The surface offers exactly what it says it offers, and nothing more.

Two kinds of assertion live here.

**Coverage.** Every one of the twelve route groups is mounted and answers, checked from the
generated OpenAPI document rather than from a list somebody keeps in their head — a route
registered on a router that was never included is in the file and not in the interface. Two of
the twelve describe themselves in no schema and are driven instead: the websocket, and the MCP
endpoint.

**Absence.** Destructive operations exist on the command line and are deliberately not
reachable here. Absence is the easiest property to lose by accident and the hardest to notice,
so each one is asserted against the **route table** — see :data:`ABSENT` — rather than by
sending the request and accepting a 404 or a 405. Those two statuses are what an absent
operation returns and also what several present ones return, so the probe could not tell the
two apart, and for one entry it was not telling them apart.

**And the same absence over MCP**, which is now served from the same process on the same port —
see :data:`ABSENT_TOOLS`. It is here rather than in ``tests/mcp/`` because it is one boundary
rather than two: these are the operations this process will not let a network reach, and a list
of them kept in two files is a list that gets extended in one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.exceptions import ToolError

from manicule.api.app import MCP_PATH, ROUTE_GROUPS
from manicule.mcp.server import TOOL_NAMES
from tests.api.live import mounted
from tests.api.support import app_for, backend_with_a_document, client_for, envelope
from tests.routing_support import Reach, classify, walk_routes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mcp.types import Tool

    from tests.app.fakes import FakeBackend

NOT_FOUND = 404
UNPROCESSABLE = 422


def _paths() -> dict[str, dict[str, Any]]:
    backend, _ = backend_with_a_document()
    document: dict[str, Any] = app_for(backend).openapi()
    return document["paths"]


def _operations() -> Iterator[tuple[str, str]]:
    for path, methods in _paths().items():
        for method in methods:
            yield method.upper(), path


# --- coverage ---------------------------------------------------------------------------------


def test_every_route_group_is_mounted() -> None:
    """Twelve groups, each with at least one route that answers.

    Checked against the OpenAPI document, which is built from the routes that were actually
    included — so a router written and never mounted fails here rather than being discovered
    by a client.

    Two of the twelve are not in that document and are named here as the exceptions rather than
    left out of the comparison: a websocket is not describable by OpenAPI, and ``/mcp`` is a
    mounted ASGI application rather than a route. Each has its own test below, because a group
    an OpenAPI-driven check cannot see is exactly the one it would report as present.
    """
    paths = set(_paths())
    expected = {
        "health": "/healthz",
        "documents": "/api/v1/documents",
        "chat": "/api/v1/chat",
        "conversations": "/api/v1/conversations",
        "collections": "/api/v1/collections",
        "tags": "/api/v1/tags",
        "admin": "/api/v1/admin/stats",
        "plugins": "/api/v1/plugins",
        "auth": "/auth/session",
        "workbench": "/api/v1/workbench",
    }
    assert set(expected) | {"websocket-chat", "mcp"} == set(ROUTE_GROUPS)
    missing = sorted(group for group, path in expected.items() if path not in paths)
    assert missing == [], f"route groups with no mounted route: {missing}"


def test_the_websocket_channel_is_mounted() -> None:
    """Not in the OpenAPI document — websockets are not described by it — so asserted directly.

    A group that no schema can describe is exactly the one an OpenAPI-driven check would
    silently report as present.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client, client.websocket_connect("/api/v1/chat/ws") as socket:
        socket.send_text('{"question": "does the client retry"}')
        assert socket.receive_json()["event"]


def test_the_mcp_endpoint_is_mounted() -> None:
    """The other group no schema describes, asserted by asking it for something.

    A bare ``GET`` rather than a protocol exchange, which the tool assertions below do: all this
    has to establish is that *something is mounted there*, and the cheapest honest way to
    establish that is a request that reaches the mount rather than the 404 handler.

    The status is deliberately not pinned. MCP's HTTP transport answers a bare ``GET`` with
    whatever it thinks of a request carrying no session and no ``Accept: text/event-stream`` —
    405 today — and that is a fact about the library. What this asserts is that the request did
    not fall through to this application, which is what an unmounted path does.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get(f"{MCP_PATH}/")
    assert response.status_code != NOT_FOUND, (
        f"{MCP_PATH}/ answered 404, so nothing is mounted there and every tool assertion below "
        f"is a statement about a surface that is not being served"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/healthz",
        "/readyz",
        "/api/v1/health",
        "/api/v1/stats",
        "/api/v1/workspaces",
        "/api/v1/documents",
        "/api/v1/documents/trash",
        "/api/v1/conversations",
        "/api/v1/collections",
        "/api/v1/tags",
        "/api/v1/plugins",
        "/api/v1/admin/stats",
        "/api/v1/admin/query-logs",
        "/api/v1/admin/audit-logs",
        "/api/v1/admin/search-quality",
        "/api/v1/admin/plugins",
        "/api/v1/admin/connectors",
        "/auth/session",
        "/auth/providers",
    ],
)
def test_every_read_route_answers(path: str) -> None:
    """Each one, with the default fake backend. A 500 here is a route nobody ever called."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get(path)
    assert response.status_code == 200, response.text


def test_every_envelope_route_returns_the_same_six_keys() -> None:
    """One shape for every route, including the failures.

    ``/healthz`` and ``/readyz`` are the two exceptions and are excluded by name: they answer
    a probe rather than a person, and a liveness check that has to parse JSON reports
    unhealthy when the serializer changes.

    ``/widget`` and ``/ui`` are excluded because they are documents rather than data — the
    browser surface has its own suites, and a page that returned an envelope would be a page
    nobody could read.

    ``/`` is the third kind: a signpost. It is a redirect to the browser surface on a whole
    server and a plain-text list of what is served on the two modes that have no browser
    surface, and neither is data a client parses. ``tests/app/test_front_door.py`` is its suite.
    """
    backend, _ = backend_with_a_document()
    probes = {"/healthz", "/readyz"}
    signposts = {"/"}
    documents = ("/widget", "/ui")
    with client_for(backend) as client:
        for method, path in _operations():
            if method != "GET" or "{" in path or path in probes or path in signposts:
                continue
            if path.startswith(documents):
                continue
            if path.startswith("/api/docs") or path.endswith("openapi.json"):
                continue
            envelope(client.get(path))


# --- statuses ---------------------------------------------------------------------------------


def test_an_unknown_document_is_a_404_carrying_an_envelope() -> None:
    """The status is derived from the error's type; the body is the same shape as a success."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/api/v1/documents/nope")
    body = envelope(response)
    assert response.status_code == NOT_FOUND
    assert body["ok"] is False
    assert body["error"]["type"] == "UnknownEntityError"
    assert body["error"]["hint"]


def test_a_duplicate_collection_name_is_a_409() -> None:
    """A collection is a deliberate object: handing back somebody else's under the same name
    merges two people's sets."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        assert client.post("/api/v1/collections", json={"name": "Runbooks"}).status_code == 200
        again = client.post("/api/v1/collections", json={"name": "Runbooks"})
    assert again.status_code == 409
    assert envelope(again)["error"]["type"] == "NameInUseError"


def test_an_unknown_profile_is_a_400() -> None:
    """A caller error, and the message lists the profiles that exist."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/api/v1/search", params={"q": "x", "profile": "telepathic"})
    assert response.status_code == 400
    assert "fast" in envelope(response)["error"]["message"]


def test_a_closed_request_body_rejects_an_unknown_field() -> None:
    """A field silently ignored looks exactly like one that worked.

    The refusal is the **ordinary envelope**, not FastAPI's own ``{"detail": [...]}``. That is
    the single most common failure a client hits, and it is the one place a second response
    shape would otherwise appear on a surface whose whole contract is that there is one.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post("/api/v1/chat", json={"question": "hello", "temperature": 0.9})
    assert response.status_code == UNPROCESSABLE
    body = envelope(response)
    assert body["ok"] is False
    assert body["error"]["type"] == "RequestValidationError"


def test_a_missing_required_parameter_is_also_the_ordinary_envelope() -> None:
    """The other half of the same property: a query parameter, not a body."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/api/v1/search")
    assert response.status_code == UNPROCESSABLE
    assert envelope(response)["ok"] is False


# --- the destructive boundary -----------------------------------------------------------------

ABSENT: tuple[tuple[str, str, Reach, str], ...] = (
    (
        "DELETE",
        "/api/v1/index",
        Reach.UNROUTED,
        "reset-index empties the whole workspace with no restore path",
    ),
    (
        "POST",
        "/api/v1/admin/reset-index",
        Reach.UNROUTED,
        "the same operation under an admin path",
    ),
    (
        "POST",
        "/api/v1/admin/restore",
        Reach.UNROUTED,
        "restore overwrites the live data directory",
    ),
    ("POST", "/api/v1/admin/backup", Reach.UNROUTED, "a backup writes wherever the caller names"),
    (
        "POST",
        "/api/v1/documents/upload",
        Reach.SHADOWED,
        "an ingest path with no filesystem permission check",
    ),
    ("POST", "/api/v1/admin/upgrade", Reach.UNROUTED, "an upgrade fetches and executes code"),
    (
        "POST",
        "/api/v1/admin/connectors",
        Reach.SIBLING,
        "declaring a connector points the index somewhere new",
    ),
    (
        "GET",
        "/api/v1/admin/benchmark",
        Reach.UNROUTED,
        "a benchmark on request is a denial of service",
    ),
    (
        "POST",
        "/api/v1/connectors/sidecar",
        Reach.UNROUTED,
        "sidecar generation writes files into the corpus directory",
    ),
    (
        "POST",
        "/api/v1/admin/sidecar",
        Reach.UNROUTED,
        "the same operation under an admin path",
    ),
    (
        "POST",
        "/api/v1/connectors/login",
        Reach.UNROUTED,
        "browser sign-in opens a window on the host and writes a credential to the keychain",
    ),
    (
        "POST",
        "/api/v1/admin/connectors/login",
        Reach.UNROUTED,
        "the same operation under an admin path",
    ),
    (
        "POST",
        "/api/v1/admin/reindex",
        Reach.UNROUTED,
        "a corpus-wide re-parse runs the embedder over everything a parser bump touched",
    ),
    (
        "POST",
        "/api/v1/documents/reindex",
        Reach.SHADOWED,
        "the same sweep where the per-document verb lives",
    ),
)
"""Operations that exist elsewhere in manicule and are deliberately not routes here.

Each is named with the reason, and with **why the request does not reach an operation** — which
is a fact about the route table and is what :func:`test_a_destructive_operation_has_no_route`
checks. The three are not interchangeable, and writing the expected one down is the point:

* :attr:`~tests.routing_support.Reach.UNROUTED` — nothing matches. Absence is structural.
* :attr:`~tests.routing_support.Reach.SIBLING` — the literal path is published for another verb.
  ``GET /api/v1/admin/connectors`` lists connectors; adding ``POST`` to it would be adding the
  declaring operation, which is exactly the change this should fail on.
* :attr:`~tests.routing_support.Reach.SHADOWED` — **a latent defect, recorded rather than
  fixed.** ``POST /api/v1/documents/upload`` is refused only because ``/documents/{document_id}``
  declares no ``POST`` *today*. Nothing about upload is being checked; the day a document-update
  verb is added there, this path starts executing it with ``document_id='upload'``. The entry is
  declared ``SHADOWED`` so that day turns it into ``EXECUTES`` and fails here, loudly, instead of
  passing on in silence.

``POST /api/v1/plugins/install`` is deliberately **not** in this list. It matches
``/api/v1/plugins/{name}`` and *runs* ``plugin_add`` with ``name='install'``; its 404 comes from
inside that handler, so probing the path never demonstrated an absence at all. What is actually
absent is the operation, and that is asserted by name in
:func:`test_no_route_installs_a_plugin`.

An absence with no test is an absence that comes back.
"""


@pytest.mark.parametrize(("method", "path", "expected", "why"), ABSENT)
def test_a_destructive_operation_has_no_route(
    method: str, path: str, expected: Reach, why: str
) -> None:
    """Not reachable, asserted over the route table, and the reason travels with it.

    Deliberately **not** a request whose status code is inspected. 404 and 405 are what an
    absent operation returns and also what several present ones return, so a probe cannot tell
    "there is no such operation" from "a handler ran and did not find the entity you named" —
    and this list contained one of the latter wearing the former's costume.
    """
    reached = classify(method, path, walk_routes())
    assert reached.reach is not Reach.EXECUTES, (
        f"{method} {path} runs {reached.route}. It is deliberately absent because {why}, but "
        f"this request reaches a handler — any 404 comes from inside it, and a request naming "
        f"an entity that exists would not get one."
    )
    assert reached.reach is expected, (
        f"{method} {path} is {reached.reach.value} ({reached.route}), not {expected.value}. The "
        f"operation is deliberately absent because {why}; what changed is *why* it is absent, "
        f"which is what this list records. Update the declared reach if the new reason is "
        f"intended."
    )


def test_no_route_installs_a_plugin() -> None:
    """Asserted by name over the route table, because no path probe can assert it.

    ``POST /api/v1/plugins/install`` looks like the check and is not one: it matches
    ``/api/v1/plugins/{name}`` and **executes** ``plugin_add`` with ``name='install'``, returning
    404 only because nothing is installed under that name. Install a plugin actually called
    ``install`` and the same request returns 200 and enables it — with a status-code assertion
    about that path still green.

    The boundary itself holds, and this is about the assertion rather than about the boundary:
    ``plugin_add`` requires an admin principal and has no branch that fetches or executes code,
    which :func:`test_enabling_a_plugin_never_installs_one` covers from the behavioral side.
    What was missing was anything that would notice an *install* route being added, so that is
    what this is.
    """
    offending = sorted(
        f"{route.name} {route.path}"
        for route in walk_routes()
        if "install" in route.path.lower() or "install" in route.name.lower()
    )
    assert offending == [], (
        f"a route installs plugins: {offending}. Installing a plugin fetches and executes code, "
        f"so it stays on the command line; this surface only enables one that an operator has "
        f"already put on disk."
    )


def test_no_route_signs_a_connector_in() -> None:
    """Asserted by name as well as by path, because a path probe cannot cover what nobody named.

    ``connector login`` was already command-line only, and what changed is how much that matters:
    it now **opens a browser window on the host** and waits for a person, on top of writing a
    credential to the keychain. A request that launches a GUI on the server is a new kind of
    authority rather than a new operation — and an unattended caller reaching it would hang for
    the length of the timeout with a window nobody is sitting at.

    The two ``ABSENT`` entries above cover the paths somebody would guess. This covers the one
    they would not, which is the gap ``tests/routing_support`` names in as many words.

    **Matched on the operation rather than on the route's name**, and the difference is not
    academic: the first version of this test looked for ``login`` in the path or the name, and a
    route mounted at ``/admin/sources/authenticate`` called ``sign_in_connector`` passed it
    while calling ``connector_login``. A word list is a guess about what somebody will call
    their route; the service operation is what the route actually reaches, and there is exactly
    one name for it.
    """
    import inspect  # noqa: PLC0415 - only this assertion reads a handler's source

    offending: list[str] = []
    for route in walk_routes():
        try:
            source = inspect.getsource(route.endpoint)
        except (OSError, TypeError):  # pragma: no cover - a handler with no readable source
            continue
        if "connector_login" in source:
            offending.append(f"{route.name} {route.path}")
    assert sorted(offending) == [], (
        f"a route signs a connector in: {sorted(offending)}. Browser sign-in opens a window on "
        f"this machine and stores a credential; both belong on the surface where a person is "
        f"present."
    )


def test_no_route_generates_sidecar_manifests() -> None:
    """Asserted by name over the route table as well as by path, and both are needed.

    The two ``ABSENT`` entries above say that the paths somebody would *guess* are unrouted.
    They say nothing about a route mounted somewhere nobody guessed, and "an operation
    reappearing under a name nobody predicted" is exactly what ``routing_support`` says a path
    probe cannot catch — which is why ``plugins/install`` has a test of this shape too.

    Sidecar generation is the one operation that writes into the corpus *directory* rather than
    into the index: a manifest beside every page under a root the caller names. Everything else
    manicule does to a corpus is read-only, so an unattended surface able to write into one is a
    new kind of authority rather than a new operation. It stays where a person is present.
    ``tests/app/test_surface_parity.py`` holds the same line for MCP; this holds it for HTTP.
    """
    offending = sorted(
        f"{route.name} {route.path}"
        for route in walk_routes()
        if "sidecar" in route.path.lower() or "sidecar" in route.name.lower()
    )
    assert offending == [], (
        f"a route generates sidecar manifests: {offending}. That writes files into the "
        f"operator's corpus directory at a path the request names, so it stays on the command "
        f"line where a person is present."
    )


def test_deleting_a_document_is_soft_and_there_is_no_hard_variant() -> None:
    """The route takes no ``hard`` parameter, and passing one changes nothing.

    Asserted through the store's own record rather than through the response, because a
    response saying ``mode: soft`` while the store performed a hard delete is exactly the
    failure worth catching.
    """
    backend, document = backend_with_a_document()
    with client_for(backend) as client:
        client.delete(f"/api/v1/documents/{document.id}", params={"hard": "true"})
    assert backend.store.deleted == [(document.id, "soft")]


def test_enabling_a_plugin_never_installs_one() -> None:
    """The route exists and refuses a plugin that is not installed, reporting the command."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post("/api/v1/plugins/not-installed")
    body = envelope(response)
    assert body["ok"] is False
    assert body["error"]["type"] == "UnknownEntityError"


# --- the same boundary, on the MCP endpoint mounted at the same address ------------------------

ABSENT_TOOLS: tuple[tuple[str, str], ...] = (
    ("index_path", "an ingest path that walks any directory this process can read"),
    ("document_delete", "removing a document, with `hard` there is no restore from"),
    ("document_reindex", "a re-parse that holds the embedder for as long as the document takes"),
    ("connector_sync", "starting a sync, which #113 refused a route for on the same grounds"),
    ("config_set", "rewriting the configuration file the server is running from"),
    ("workspace_switch", "changing which tenant the next start serves"),
    ("plugin_add", "enabling code that runs with this process's full authority"),
    ("plugin_remove", "disabling it again, which is the same authority in reverse"),
    ("collection_create", "creating a grouping"),
    ("collection_rename", "renaming one"),
    ("collection_update", "overwriting a description the call does not carry"),
    ("collection_delete", "deleting a grouping"),
    ("collection_add", "changing what a grouping holds"),
    ("collection_remove", "changing what a grouping holds"),
    ("ask", "it persists a turn given a conversation, and calls a model that may be elsewhere"),
)
"""Every mutating tool, named with what it would let an unattended caller do from the network.

The MCP twin of :data:`ABSENT`, kept in the same file because it is the same boundary: these two
lists are the whole of what this process refuses to let a network reach, and splitting them
across two files is how one of them gets extended and the other does not.

**Named rather than derived, deliberately.** ``manicule.mcp.server`` derives the offered set from
each registration's ``readOnlyHint`` — that is the mechanism, and a test that re-derived it would
assert the mechanism against itself and pass however the mechanism was wrong. So the expectation
here is written out, and :func:`test_the_absent_tools_and_the_offered_ones_are_the_whole_surface`
holds the list to being complete rather than merely true.

It is the same set as ``tests/mcp/test_annotations.py``'s ``MUTATIONS``, arrived at from the
other end: that one asserts each of these reports itself as writing, and this one asserts each is
therefore not served over a socket. Two files, one classification, and the pair is what makes the
classification worth having.
"""

MINIMUM_TOOLS = 8
"""A floor on how many tools the network surface must offer, for the reason MINIMUM_ROUTES exists.

Every assertion in :func:`test_a_mutating_tool_is_absent_from_the_network_mcp_surface` is a
statement about the published list, so a surface that published **nothing** would satisfy all
fifteen of them — and a mount that failed to start, a lifespan that was not run or a filter that
excluded everything all produce exactly that. Far below the real count on purpose: this is here
to catch a collapse, not to track the size of the surface.
"""


async def _network_tools(backend: FakeBackend) -> dict[str, Tool]:
    """``tools/list`` as a client of the mounted endpoint receives it.

    Over the protocol rather than off ``manicule.mcp.server``'s registrar, because what is under
    test is what a caller on the socket can reach. A surface computed correctly and mounted
    wrongly is the failure this is for, and reading the registrar would report it as fine.
    """
    async with mounted(backend) as client:
        return {tool.name: tool for tool in await client.list_tools()}


@pytest.mark.parametrize(("name", "why"), ABSENT_TOOLS)
async def test_a_mutating_tool_is_absent_from_the_network_mcp_surface(name: str, why: str) -> None:
    """Not published, and — the next test — not callable either.

    Absence rather than refusal is the whole property. A tool that was published and then said no
    would put the decision in a check, and a check is something a caller can be granted an
    exception to by a setting, a middleware or a header. There is no handler behind these names.
    """
    backend, _ = backend_with_a_document()
    published = await _network_tools(backend)
    assert name not in published, (
        f"{name} is published on the MCP endpoint served over the network. It is deliberately "
        f"absent because it is {why} — see manicule.mcp.serve.NETWORK_SURFACE_IS_READ_ONLY."
    )


async def test_calling_an_absent_tool_over_the_socket_finds_no_tool() -> None:
    """The second half: the name is not a handler that refuses, it is not a handler.

    Asserted on the *kind* of failure rather than only on there being one, because "the tool
    exists and declined" and "there is no such tool" are the two answers this boundary is the
    difference between — and only the second is a property nothing can grant an exception to.
    """
    backend, document = backend_with_a_document()
    async with mounted(backend) as client:
        with pytest.raises(ToolError, match="Unknown tool"):
            await client.call_tool("document_delete", {"document_id": document.id})
    assert backend.store.deleted == [], "the call reached a handler after all"


async def test_every_read_only_tool_is_offered_on_the_network_mcp_surface() -> None:
    """The mirror, without which every absence above passes on an empty surface.

    The expectation is ``TOOL_NAMES`` minus the list above rather than a second literal, because
    *that* subtraction is the claim: the two lists together are the surface, so a tool added
    tomorrow lands in one of them or fails the test after this one.
    """
    backend, _ = backend_with_a_document()
    published = await _network_tools(backend)
    expected = sorted(set(TOOL_NAMES) - {name for name, _ in ABSENT_TOOLS})
    assert sorted(published) == expected
    assert len(published) >= MINIMUM_TOOLS, (
        f"the network MCP surface published {len(published)} tool(s), below the floor of "
        f"{MINIMUM_TOOLS}. Every absence assertion above is a statement about the published "
        f"list, so a surface that published nothing would pass all of them."
    )


async def test_the_absent_tools_and_the_offered_ones_are_the_whole_surface() -> None:
    """No tool is in neither list, so :data:`ABSENT_TOOLS` cannot go stale quietly.

    The same guard ``tests/mcp/test_annotations.py`` puts on its own classification, applied to
    this one. Without it, adding a mutating tool and forgetting to name it above leaves every
    assertion here green — each is a statement about the tools it names, and an unnamed one is
    named nowhere.
    """
    backend, _ = backend_with_a_document()
    published = set(await _network_tools(backend))
    absent = {name for name, _ in ABSENT_TOOLS}
    assert published | absent == set(TOOL_NAMES)
    assert published & absent == set()


async def test_every_published_tool_says_it_reads() -> None:
    """The surface and the classification agree, checked over the protocol at the far end.

    ``manicule.mcp.server`` builds the read-only surface *from* these hints, so this is the round
    trip: what a client is told about a tool it can reach on the socket is that the tool reads.
    A published tool answering ``readOnlyHint: false`` would mean the filter and the annotation
    had come apart between the registration and the wire.
    """
    backend, _ = backend_with_a_document()
    for name, tool in (await _network_tools(backend)).items():
        assert tool.annotations is not None, f"{name} publishes no annotations"
        assert tool.annotations.readOnlyHint is True, f"{name} is served on a socket and writes"


async def test_the_instructions_tell_a_client_the_write_tools_are_not_here() -> None:
    """So that "I cannot do that" is available before a turn is spent discovering it.

    Read off the initialization result rather than off the constant, because instructions the
    server computes and does not send buy nothing.
    """
    backend, _ = backend_with_a_document()
    async with mounted(backend) as client:
        result = client.initialize_result
    assert result is not None, "the client never completed initialization"
    instructions = result.instructions or ""
    assert "read-only" in instructions, instructions
    assert "manicule serve" in instructions, instructions
    assert "## Scope every question to a collection" in instructions, (
        "the read-only notice replaced the ordinary instructions instead of being added to them"
    )
