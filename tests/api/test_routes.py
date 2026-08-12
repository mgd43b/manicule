"""The surface offers exactly what it says it offers, and nothing more.

Two kinds of assertion live here.

**Coverage.** Every one of the eleven route groups is mounted and answers, checked from the
generated OpenAPI document rather than from a list somebody keeps in their head — a route
registered on a router that was never included is in the file and not in the interface.

**Absence.** Seven destructive operations exist on the command line and are deliberately not
reachable here. Absence is the easiest property to lose by accident and the hardest to notice,
so each one is asserted by name: a route added later that reintroduces one fails this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from manicule.api.app import ROUTE_GROUPS
from tests.api.support import app_for, backend_with_a_document, client_for, envelope

if TYPE_CHECKING:
    from collections.abc import Iterator

NOT_FOUND = 404
METHOD_NOT_ALLOWED = 405
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
    """Eleven groups, each with at least one route that answers.

    Checked against the OpenAPI document, which is built from the routes that were actually
    included — so a router written and never mounted fails here rather than being discovered
    by a client.
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
    assert set(expected) | {"websocket-chat"} == set(ROUTE_GROUPS)
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
    unhealthy when the serialiser changes.

    ``/widget`` and ``/ui`` are excluded because they are documents rather than data — the
    browser surface has its own suites, and a page that returned an envelope would be a page
    nobody could read.
    """
    backend, _ = backend_with_a_document()
    probes = {"/healthz", "/readyz"}
    documents = ("/widget", "/ui")
    with client_for(backend) as client:
        for method, path in _operations():
            if method != "GET" or "{" in path or path in probes or path.startswith(documents):
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

ABSENT: tuple[tuple[str, str, str], ...] = (
    (
        "DELETE",
        "/api/v1/index",
        "reset-index empties the whole workspace with no restore path",
    ),
    ("POST", "/api/v1/admin/reset-index", "the same operation under an admin path"),
    ("POST", "/api/v1/admin/restore", "restore overwrites the live data directory"),
    ("POST", "/api/v1/admin/backup", "a backup writes wherever the caller names"),
    ("POST", "/api/v1/documents/upload", "an ingest path with no filesystem permission check"),
    ("POST", "/api/v1/plugins/install", "installing a plugin fetches and executes code"),
    ("POST", "/api/v1/admin/upgrade", "an upgrade fetches and executes code"),
    ("POST", "/api/v1/admin/connectors", "declaring a connector points the index somewhere new"),
    ("GET", "/api/v1/admin/benchmark", "a benchmark on request is a denial of service"),
    (
        "POST",
        "/api/v1/collections/orphans",
        "deleting every document outside every collection is most of a corpus where "
        "collections are optional, and `collection orphans --confirm` is the only way to it",
    ),
    (
        "DELETE",
        "/api/v1/collections/orphans",
        "the same operation spelled as a delete",
    ),
)
"""Operations that exist elsewhere in manicule and are deliberately not routes here.

Each is named with the reason. An absence with no test is an absence that comes back.
"""


@pytest.mark.parametrize(("method", "path", "why"), ABSENT)
def test_a_destructive_operation_has_no_route(method: str, path: str, why: str) -> None:
    """Not reachable, and the reason travels with the assertion.

    404 or 405 both mean "there is no such operation here". What must not happen is a 200, a
    401 or a 403 — each of those says the route exists and something else stopped it, which is
    one configuration change away from not stopping it.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.request(method, path, json={})
    assert response.status_code in {NOT_FOUND, METHOD_NOT_ALLOWED}, (
        f"{method} {path} exists. It is deliberately absent because {why}."
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
