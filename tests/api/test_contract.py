"""Properties that hold for every route, asserted over every route.

A per-route test proves a property of the route somebody remembered to write a test for. These
enumerate the mounted routes instead, so a route added later is covered by construction.

The one to read carefully is the operation vocabulary. A **refused** request never reaches its
service call, so the ``op`` on its envelope comes from the matched route rather than from the
call — and without an explicit name that is the handler's Python function name, which is a
different vocabulary from the one every successful envelope uses. An access log of refusals
would then be unjoinable to one of successes, quietly.

**The enumeration proves it ran before it reports what it found.** ``app.routes`` is not a flat
list of routes on every FastAPI: from 0.13x an included router appears as a wrapper object, and
a walk that only recognised :class:`~fastapi.routing.APIRoute` therefore found **nothing at
all** — and reported success, because "no route is misnamed" is trivially true of no routes.
That is the failure mode this file exists to not have, so :func:`_routes` descends into
whatever shape the framework used and :data:`MINIMUM_ROUTES` is a floor below which the walk is
assumed to have collapsed rather than the surface to have shrunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute

from manicule.api.app import build_app
from manicule.app.service import ApplicationService
from manicule.mcp.server import TOOL_NAMES
from tests.api.support import backend_with_a_document, client_for, envelope

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

UNAUTHORIZED = 401

OPERATIONS: frozenset[str] = frozenset(TOOL_NAMES) | {
    # Operations this surface adds. Each is a service method name, exactly as the MCP tool
    # names are — the vocabulary is one list, not one per surface.
    "audit_log",
    "auth_providers",
    "auth_session",
    "api_key_create",
    "api_key_list",
    "api_key_revoke",
    "chat_feedback",
    "collection_add",
    "collection_create",
    "collection_delete",
    "collection_documents",
    "collection_list",
    "collection_remove",
    "conversation_create",
    "conversation_delete",
    "conversation_list",
    "conversation_messages",
    "conversation_rename",
    "conversation_share",
    "conversation_unshare",
    "document_restore",
    "document_tag",
    "document_trash",
    "document_untag",
    "plugin_health",
    "query_logs",
    "search_quality",
    "shared_conversation",
    "tag_create",
    "tag_delete",
    "tag_list",
    "workbench",
}
"""Every ``op`` this surface may emit.

The MCP tool names plus the operations only the HTTP surface reaches. Written out so that a
route named something new fails here rather than inventing a word for the contract.
"""

NOT_OPERATIONS: frozenset[str] = frozenset(
    {
        # The two probes answer a supervisor rather than a person and emit no envelope, so
        # they have no operation to name.
        "healthz",
        "readyz",
        # The widget is a script and a page, not an operation.
        "widget_script",
        "widget_demo",
        # FastAPI's own, mounted by the framework.
        "swagger_ui_html",
        "swagger_ui_redirect",
        "openapi",
        # The browser surface (#12). A page is not an operation: it runs several of them and
        # renders HTML, and naming it after one of them would put a page's name on an envelope
        # that some other operation produced. Its refusals are rendered as pages rather than
        # as envelopes, so none of these ever reaches the `op` field this file is about.
        "ui_admin",
        "ui_auth",
        "ui_chat",
        "ui_collections",
        "ui_connectors",
        "ui_conversation",
        "ui_dashboard",
        "ui_document",
        "ui_documents",
        "ui_health",
        "ui_plugins",
        "ui_script",
        "ui_search",
        "ui_settings",
        "ui_shared",
        "ui_stylesheet",
        "ui_trash",
        "ui_workspaces",
    }
)

MINIMUM_ROUTES = 40
"""A floor on how many routes the walk below must find.

Far below the real count, and present to catch a walk that collapsed rather than to track the
size of the surface. It has caught one: on FastAPI 0.141 an included router is a wrapper object
rather than its routes, and the previous walk found **zero** — with every assertion in this
file passing, because each of them is a statement about every route and there were none.
"""


def _descend(routes: Iterable[object]) -> Iterator[APIRoute | APIWebSocketRoute]:
    """Every route, whichever shape this FastAPI put them in.

    A router included with ``include_router`` may appear on ``app.routes`` as its routes or as
    one object standing for them, depending on the version. Both are followed, so this file
    keeps enumerating the surface across an upgrade instead of quietly enumerating none of it.
    """
    for route in routes:
        if isinstance(route, (APIRoute, APIWebSocketRoute)):
            yield route
            continue
        inner = getattr(route, "routes", None)
        if inner is None:
            inner = getattr(getattr(route, "original_router", None), "routes", None)
        if inner:
            yield from _descend(cast("Iterable[object]", inner))


def _routes() -> list[APIRoute | APIWebSocketRoute]:
    backend, _ = backend_with_a_document()
    found = list(_descend(build_app(ApplicationService(backend)).routes))
    assert len(found) >= MINIMUM_ROUTES, (
        f"the walk found {len(found)} route(s), below the floor of {MINIMUM_ROUTES}. Every "
        f"assertion in this file is a statement about *every* route, so a walk that found "
        f"none passes them all. Fix the walk rather than the floor."
    )
    return found


def test_every_route_is_named_for_the_operation_it_runs() -> None:
    """The vocabulary is one list across three surfaces, including on a refusal.

    A route with no explicit name takes its handler's function name — ``list_documents``
    rather than ``document_list`` — and that name is what a 401 or a 403 puts on the envelope,
    because a refused request never reaches the service call that would have named it.
    """
    unknown = sorted(
        route.name
        for route in _routes()
        if route.name not in OPERATIONS and route.name not in NOT_OPERATIONS
    )
    assert unknown == [], (
        f"these routes are named something that is not an operation: {unknown}. Give the route "
        f"`name=` matching the service method it calls, or add it to OPERATIONS."
    )


def test_a_refusal_names_the_operation_the_request_was_heading_for() -> None:
    """End to end: a 401 on the document listing says ``document_list``.

    Asserted through a real refused request rather than through the route table, because the
    property is about what reaches the client.
    """
    backend, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    with client_for(backend) as client:
        response = client.get("/api/v1/documents")
    body = envelope(response)
    assert response.status_code == UNAUTHORIZED
    assert body["op"] == "document_list"
    assert body["op"] in OPERATIONS


def test_a_refusal_and_a_success_name_the_same_operation() -> None:
    """The point of the property: two log lines about one endpoint are joinable."""
    keyed, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    open_, _ = backend_with_a_document()
    with client_for(keyed) as refused_client, client_for(open_) as allowed_client:
        refused = envelope(refused_client.get("/api/v1/documents"))
        allowed = envelope(allowed_client.get("/api/v1/documents"))
    assert refused["ok"] is False
    assert allowed["ok"] is True
    assert refused["op"] == allowed["op"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/v1/admin/query-logs", "query_logs"),
        ("/api/v1/conversations", "conversation_list"),
        ("/api/v1/workbench?document_id=x", "workbench"),
        ("/api/v1/collections", "collection_list"),
    ],
)
def test_refusals_across_the_groups_name_their_operation(path: str, expected: str) -> None:
    """One per group that a viewer-or-better reaches, so a group added later is not the gap."""
    backend, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    with client_for(backend) as client:
        body = envelope(client.get(path))
    assert body["ok"] is False
    assert body["op"] == expected


def test_every_route_describes_itself() -> None:
    """A route with no summary is an OpenAPI entry a client has to guess at."""
    undescribed = sorted(
        route.path
        for route in _routes()
        if isinstance(route, APIRoute) and not (route.summary or route.description)
    )
    assert undescribed == [], f"routes with no summary: {undescribed}"


def test_no_route_exposes_the_orphan_cleanup() -> None:
    """``collection_orphans`` deletes documents, so it is not on this surface at all.

    Asserted over the walked route table, and it belongs in this file rather than beside the
    other deliberate absences for a reason worth writing down.

    The obvious version asks for ``POST /api/v1/collections/orphans`` and accepts 404 or 405.
    Both come back and **neither means what it looks like**: ``/collections/orphans`` matches
    ``/collections/{collection_id}``, so the 405 is "wrong verb on a path that exists", and the
    ``DELETE`` is a 404 from ``collection_delete`` *running* with ``collection_id='orphans'``.
    Create a collection actually called ``orphans`` and that request deletes it and returns
    200, while the assertion goes on passing.

    The second attempt read ``app.routes`` directly and was worse — it found no routes at all,
    for the reason this module's docstring already gives, and reported success. Hence
    :func:`_routes`, whose floor fails a walk that collapsed.
    """
    offending = sorted(
        f"{route.name} {route.path}"
        for route in _routes()
        if "orphan" in route.path.lower() or "orphan" in route.name.lower()
    )
    assert offending == [], (
        f"the orphan cleanup is reachable over HTTP: {offending}. It moves every document "
        f"outside every collection into the trash, which in a corpus where collections are "
        f"optional is most of it. It stays on the command line as `collection orphans "
        f"--confirm`, with the other operations that destroy data."
    )
