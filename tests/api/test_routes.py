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

from typing import TYPE_CHECKING, Any, cast

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from manicule.api.app import MCP_PATH, ROUTE_GROUPS, build_app
from manicule.app.bind import Bind
from manicule.app.service import ApplicationService
from manicule.config.settings import AuthMode
from manicule.core.errors import PolicyError
from manicule.mcp.server import NETWORK_AUTHORING, TOOL_NAMES, build_server
from tests.api.live import mounted
from tests.api.support import app_for, backend_with_a_document, client_for, envelope
from tests.mcp.test_annotations import MUTATIONS
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


def test_collection_rule_create_show_replace_and_clear_round_trip() -> None:
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        created = envelope(
            client.post(
                "/api/v1/collections",
                json={"name": "Team A", "rule": {"sources": ["wiki-team-a"]}},
            )
        )
        collection_id = created["data"]["id"]
        assert created["data"]["rule"]["sources"] == ["wiki-team-a"]

        replaced = envelope(
            client.put(
                f"/api/v1/collections/{collection_id}/rule",
                json={"rule": {"sources": ["wiki-team-a-archive", "wiki-team-a"]}},
            )
        )
        assert replaced["data"]["rule"]["sources"] == [
            "wiki-team-a",
            "wiki-team-a-archive",
        ]
        shown = envelope(client.get(f"/api/v1/collections/{collection_id}/rule"))
        assert shown["data"] == replaced["data"]

        cleared = envelope(client.delete(f"/api/v1/collections/{collection_id}/rule"))
        assert cleared["data"]["rule"] is None


def test_collection_rule_http_body_refuses_empty_blank_and_workspace_scope() -> None:
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        made = envelope(client.post("/api/v1/collections", json={"name": "Team A"}))
        path = f"/api/v1/collections/{made['data']['id']}/rule"
        for rule in (
            {},
            {"sources": [""]},
            {"sources": ["wiki"], "workspace": "other"},
            {"sources": ["wiki"], "workspace_ids": ["other"]},
        ):
            response = client.put(path, json={"rule": rule})
            assert response.status_code == UNPROCESSABLE

        # Explicit DELETE is the sole clear spelling: omission and null cannot erase a rule.
        assert client.put(path, json={}).status_code == UNPROCESSABLE
        assert client.put(path, json={"rule": None}).status_code == UNPROCESSABLE


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


def test_reembed_network_surface_has_operator_action_and_status_parity() -> None:
    """Authenticated admins receive the same explicit run lifecycle as other surfaces."""
    reembed = sorted(
        (
            route.name,
            route.path,
            sorted(cast("set[str] | None", getattr(route, "methods", None)) or ()),
        )
        for route in walk_routes()
        if "reembed" in route.name.lower() or "reembed" in route.path.lower()
    )
    assert reembed == [
        ("reembed_abandon", "/api/v1/admin/reembed/{run_id}/abandon", ["POST"]),
        ("reembed_cleanup", "/api/v1/admin/reembed/{run_id}", ["DELETE"]),
        ("reembed_plan", "/api/v1/admin/reembed", ["GET"]),
        ("reembed_resume", "/api/v1/admin/reembed/{run_id}/resume", ["POST"]),
        ("reembed_start", "/api/v1/admin/reembed/{run_id}/start", ["POST"]),
        ("reembed_status", "/api/v1/admin/reembed/{run_id}", ["GET"]),
        ("ui_reembed", "/ui/reembed", ["GET"]),
    ]


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
    ("collection_rule_set", "overwriting a membership rule the call cannot restore"),
    ("collection_rule_clear", "removing a membership rule the call cannot restore"),
    ("collection_delete", "deleting a grouping"),
    ("collection_add", "changing what a grouping holds"),
    ("collection_remove", "changing what a grouping holds"),
    (
        "research",
        "several model calls and several retrievals for one call, which is `ask`'s reasons "
        "over again and more of them",
    ),
    ("ask", "it persists a turn given a conversation, and calls a model that may be elsewhere"),
)
"""Every mutating tool the network may not reach, named with what it would let a caller do.

**``document_create`` is deliberately not in this list, and it is the only write that is not.**
Every entry above is excluded for the *authority* it carries rather than for being a write:
``index_path`` walks any directory this process can read, ``config_set`` rewrites the
configuration the server is running from, ``plugin_add`` enables code with this process's full
authority, ``collection_delete`` removes a grouping the call cannot restore. Authoring writes one
document, to one workspace, in one configured collection, beneath one configured connector's
root, at a path the caller never supplies — bounded by settings an operator wrote rather than by
arguments a caller sends. That is a different kind of thing from its neighbors here, and the list
is only worth having because every entry explains itself, so its absence has to explain itself
too.

The care is not proportional to the tool's size, and that is the point of saying it out loud: the
corpus this exists for is read as *instructions*. Guidance recalled from it is treated as standing
direction by whatever recalls it, so writing into it is the ability to place text in front of
future sessions. That is why the default is off — an installation that configures no authoring
source has no authoring, including over a socket — and why
:func:`~manicule.app.bind.require_authoring_authentication` refuses to start a socket that would
serve it unauthenticated.

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


EVERYWHERE = "0.0.0.0"  # noqa: S104 - named once, so the literal is explained once

AUTHENTICATED = {"security": {"auth": {"mode": "api_key"}}}
"""Settings for a socket that carries its one write.

**Named because it is now a precondition rather than a detail.**
``manicule.mcp.server.network_authoring`` returns nothing when ``security.auth.mode`` is
``none``, so a surface built over default settings is the read-only set and *only* that. Every
assertion below about ``document_create`` being present is an assertion about an authenticated
socket, and passing these overrides is what makes it one.
"""


async def _network_tools(
    backend: FakeBackend, *, credential: dict[str, str] | None = None
) -> dict[str, Tool]:
    """``tools/list`` as a client of the mounted endpoint receives it.

    Over the protocol rather than off ``manicule.mcp.server``'s registrar, because what is under
    test is what a caller on the socket can reach. A surface computed correctly and mounted
    wrongly is the failure this is for, and reading the registrar would report it as fine.

    ``credential`` is needed on an authenticated mount and must be absent from an unauthenticated
    one — the guard refuses an anonymous caller wherever a key is configured, so a test that
    forgot it would see an empty surface and read it as an absence.
    """
    async with mounted(backend, credential=credential) as client:
        return {tool.name: tool for tool in await client.list_tools()}


async def _authenticated_socket() -> tuple[FakeBackend, dict[str, str]]:
    """A backend whose socket carries its one write, and a viewer key that may list it.

    A *viewer* deliberately: the surface a socket publishes is the same for every role — the
    mount's guard asks for a viewer because it carries the read surface — so listing as the
    least-privileged caller is the honest way to ask what is published. Who may *call*
    ``document_create`` is a separate question, and ``_authored_as`` is where it is asked.
    """
    backend, _ = backend_with_a_document(**AUTHENTICATED)
    issued = await ApplicationService(backend).api_key_create("viewer-key", role="viewer")
    return backend, {"X-API-Key": issued.secret}


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
    backend, credential = await _authenticated_socket()
    published = await _network_tools(backend, credential=credential)
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
    backend, credential = await _authenticated_socket()
    published = set(await _network_tools(backend, credential=credential))
    absent = {name for name, _ in ABSENT_TOOLS}
    assert published | absent == set(TOOL_NAMES)
    assert published & absent == set()


async def test_every_published_tool_reads_or_is_the_one_named_write() -> None:
    """The surface and the classification agree, checked over the protocol at the far end.

    ``manicule.mcp.server`` builds the read-only surface *from* these hints, so this is the round
    trip: what a client is told about a tool it can reach on the socket is that the tool reads —
    unless it is named in ``NETWORK_AUTHORING``. A published tool answering ``readOnlyHint: false``
    and *not* in that set would mean the filter and the annotation had come apart between the
    registration and the wire.

    The exception is read from the constant rather than written out here, because a literal in a
    test is a second place the set is decided and the two would agree until somebody edited one.
    What this file writes out instead is the *equality* below, which is the claim worth pinning.
    """
    backend, credential = await _authenticated_socket()
    for name, tool in (await _network_tools(backend, credential=credential)).items():
        assert tool.annotations is not None, f"{name} publishes no annotations"
        if name in NETWORK_AUTHORING:
            assert tool.annotations.read_only_hint is False, (
                f"{name} is in NETWORK_AUTHORING, which exists for tools that write. A read-only "
                f"tool there is admitted by a rule it does not need."
            )
            continue
        assert tool.annotations.read_only_hint is True, f"{name} is served on a socket and writes"


async def test_the_network_surface_is_the_reads_plus_exactly_one_named_write() -> None:
    """Asserted as a set operation, so nothing else can drift onto a socket behind authoring.

    **This is the test that makes ``NETWORK_AUTHORING`` safe to have at all.** Every assertion
    above names a tool; a second write tool admitted tomorrow would be named by none of them and
    would pass all of them. This one names no tool: the published set *is* the read-only set plus
    that constant, and a third member — or a write tool let through by some other route — fails
    here with both sides printed.

    The read-only set is derived from the published annotations of the **whole** surface rather
    than from ``ABSENT_TOOLS``, so this does not compare one hand-maintained list against another.
    """
    backend, credential = await _authenticated_socket()
    published = set(await _network_tools(backend, credential=credential))
    assert published == await _reads_on_the_whole_surface(backend) | NETWORK_AUTHORING
    assert {"document_create"} == NETWORK_AUTHORING, (
        "the network surface grew a second write tool. That is a decision with its own threat "
        "model — see docs/surfaces.md — not a line to update until this passes."
    )


async def test_the_network_surface_is_the_same_set_without_authentication() -> None:
    """The surface a socket carries is decided by the transport, never by the credential.

    **Asserted as the same set operation as the test above**, over an installation with
    ``auth.mode = none``, because the pair is the claim: authentication decides *who may call*
    ``document_create`` and never *whether it is there*. Those were briefly one question, and the
    answer made the deployment authoring exists for impossible — assistants on machines running
    no manicule of their own could search a memory corpus and not write to it.

    The risk that comes with it is real and is accepted elsewhere: an anonymous caller on an
    unauthenticated installation resolves to an administrator, so ``require_network_member``'s
    floor admits everybody. That is what ``--no-authentication`` buys, and what the startup
    banner and ``doctor`` say out loud; it is not something this surface silently prevents.
    """
    backend, _ = backend_with_a_document()
    assert backend.settings.security.auth.mode is AuthMode.NONE

    published = set(await _network_tools(backend))

    assert published == await _reads_on_the_whole_surface(backend) | NETWORK_AUTHORING
    assert published & (set(MUTATIONS) - NETWORK_AUTHORING) == set(), sorted(
        published & (set(MUTATIONS) - NETWORK_AUTHORING)
    )


async def _reads_on_the_whole_surface(backend: FakeBackend) -> set[str]:
    """Every tool that reports itself read-only, read off the **whole** surface.

    Derived from the published annotations rather than from ``ABSENT_TOOLS``, so the tests above
    do not compare one hand-maintained list against another. Over stdio, because that is the
    surface that carries everything and is therefore the only one that can be asked what the
    complete classification is.
    """
    async with Client(build_server(ApplicationService(backend))) as client:
        everything = {tool.name: tool.annotations for tool in await client.list_tools()}
    return {
        name
        for name, annotations in everything.items()
        if annotations is not None and annotations.read_only_hint is True
    }


def test_an_application_serving_authoring_without_authentication_refuses_to_be_built() -> None:
    """The second half of the same rule, and the half that covers somebody else's server.

    ``manicule.api.serve`` is not the only thing that puts this application on a port: a
    container entry point or a production ASGI server builds it and does the listening itself.
    That is the reason ``_require_auth_for_wide_bind`` lives here rather than beside the bind,
    and authoring's refusal is here for it too — stricter, because it refuses a **loopback** bind
    as well. ``/mcp/`` is mounted on this application, so an unauthenticated local port would
    otherwise carry a tool that writes into a corpus.
    """
    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI out of the CLI path
    from manicule.config.settings import AuthoringSettings  # noqa: PLC0415

    backend, _ = backend_with_a_document()
    backend.settings = backend.settings.model_copy(
        update={"authoring": AuthoringSettings(source="memories", collections=("memory",))}
    )

    with pytest.raises(PolicyError, match="authoring"):
        build_app(ApplicationService(backend))


def test_the_no_authentication_flag_buys_an_unauthenticated_authoring_application() -> None:
    """The HTTP half of the waiver, and it is the same decision taken once.

    ``document_create`` reaches this surface twice — as the MCP tool on the mount, and as
    ``POST /api/v1/documents``, whose ``MemberPrincipal`` floor an anonymous administrator
    clears. Serving one and refusing the other would be a distinction no operator asked for and
    no threat model supports, so the flag opens both or neither.

    Without the flag it is still refused, which is the half that matters for an installation that
    never asked: a corpus is not exposed by forgetting a setting.
    """
    from manicule.config.settings import AuthoringSettings  # noqa: PLC0415

    backend, _ = backend_with_a_document()
    backend.settings = backend.settings.model_copy(
        update={"authoring": AuthoringSettings(source="memories", collections=("memory",))}
    )

    with pytest.raises(PolicyError, match="authoring"):
        build_app(ApplicationService(backend))

    assert build_app(ApplicationService(backend), allow_unauthenticated=True) is not None


def test_an_unauthenticated_wide_application_is_built_without_authoring_on_either_surface() -> None:
    """The positive control for the pair above, and the shape of what the flag actually buys.

    An application that refused to be built however it was asked would satisfy both refusal
    tests, so this is the one that says the escape hatch opens.

    **What it opens is not a read-only application, and this test does not claim it is.** It was
    called "can write nowhere", which was false and is the kind of name that becomes evidence:
    `build_app` still mounts the admin route table, and ``auth.mode = none`` makes an anonymous
    caller an administrator, so `config_set`, `plugin_add` and the rest stay callable. The
    narrowing is scoped to one write — the one into a corpus read back as standing instructions
    — and the name now says exactly that much. Whether the rest of the HTTP surface should
    shrink too is an open decision, not something asserted here.
    """
    backend, _ = backend_with_a_document()
    assert backend.settings.authoring.configured is False

    wide = Bind(EVERYWHERE, 8765, loopback=False, every_interface=True)
    app = build_app(ApplicationService(backend), bind=wide, allow_unauthenticated=True)

    assert app.title == "manicule"


def test_an_application_without_authoring_configured_is_built_unauthenticated() -> None:
    """The condition is authoring being *configured*, not the tool existing.

    Without this, the refusal above would make every loopback installation require an API key —
    a change to how manicule is served, imposed by a feature that installation does not use.
    """
    backend, _ = backend_with_a_document()
    assert backend.settings.authoring.configured is False
    assert build_app(ApplicationService(backend)) is not None


AUTHORING = {
    "security": {"auth": {"mode": "api_key"}},
    "authoring": {"source": "memories", "collections": ["memory"]},
}
"""An installation that has configured authoring, with authentication on.

Both, because one without the other cannot be served: ``require_authoring_authentication``
refuses to build an application whose authoring is reachable unauthenticated.
"""


async def _authored_as(role: str) -> dict[str, Any]:
    """Call ``document_create`` over the mount with a key of this role, and return the envelope.

    Through the mounted endpoint rather than the in-memory server, because the floor is a
    property of *a call that arrived over HTTP* — the check reads the request out of the FastMCP
    context, and a server driven in memory has none.
    """
    backend, _ = backend_with_a_document(**AUTHORING)
    issued = await ApplicationService(backend).api_key_create(f"{role}-key", role=role)
    async with mounted(backend, credential={"X-API-Key": issued.secret}) as client:
        result = await client.call_tool(
            "document_create",
            {"collection": "memory", "slug": "retry-policy", "body": "# Retry\n"},
        )
    return dict(result.structured_content or {})


async def test_a_viewer_key_cannot_author_over_the_mounted_surface() -> None:
    """The mount admits viewers because it carries the read surface. Authoring is not a read.

    The guard's floor is ``Role.VIEWER`` — right for `search`, `collection_list` and the rest of
    what a socket offers — so admitting one write tool to that mount without a floor of its own
    handed a read-only key an authority ``POST /api/v1/documents`` denies it. The two surfaces
    must not disagree about who may write into a corpus.
    """
    envelope = await _authored_as("viewer")

    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "ForbiddenError"


async def test_a_member_key_clears_the_floor() -> None:
    """The other half, without which the refusal above passes on a tool that refuses everybody.

    A member gets *past* the floor and then meets this installation's own configuration — the
    fake backend has no filesystem connector to author into — so what is asserted is that the
    failure is no longer an authorization one.
    """
    envelope = await _authored_as("member")

    assert envelope["ok"] is False
    assert envelope["error"]["type"] not in {"ForbiddenError", "UnauthenticatedError"}


async def test_the_instructions_tell_a_client_the_write_tools_are_not_here() -> None:
    """So that "I cannot do that" is available before a turn is spent discovering it.

    Read off the negotiation result rather than off the constant, because instructions the
    server computes and does not send buy nothing. ``client.instructions`` answers from whichever
    handshake the connection used — MCP SDK v2's ``server/discover`` leaves ``initialize_result``
    unset — so this stays a claim about what a client receives.
    """
    backend, credential = await _authenticated_socket()
    async with mounted(backend, credential=credential) as client:
        instructions = client.instructions
    assert instructions is not None, "the server sent no instructions"
    assert "read-only" in instructions, instructions
    assert "manicule serve" in instructions, instructions
    assert "document_create" in instructions, (
        "the notice lists what a socket does not carry, and authoring is the one write it does. "
        "A client told the server is read-only and not told about the exception will not call it."
    )
    assert "## Scope every question to a collection" in instructions, (
        "the read-only notice replaced the ordinary instructions instead of being added to them"
    )


async def test_an_unauthenticated_socket_is_told_the_same_thing_as_any_other() -> None:
    """One notice, because there is one surface. A client is told authoring is here, and it is.

    A second notice existed briefly, saying ``document_create`` was absent, for a surface that no
    longer differs. A client given the wrong one of those spends a turn either calling a tool that
    is not there or declining to call one that is.
    """
    backend, _ = backend_with_a_document()
    assert backend.settings.security.auth.mode is AuthMode.NONE
    async with mounted(backend) as client:
        instructions = client.instructions

    assert instructions is not None, "the server sent no instructions"
    assert "read-only" in instructions, instructions
    assert "document_create" in instructions, (
        "an unauthenticated socket carries authoring and its instructions do not say so, so an "
        "assistant deployed to write memories will not call the tool that is there"
    )
