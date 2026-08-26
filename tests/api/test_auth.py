"""Authentication and authorization, on the surface an unattended caller reaches.

Two properties, and each needs both halves of a pair to mean anything.

**A credential is required when one is configured.** Asserted with a positive control — the
same request with a valid key succeeds — because a surface that refused everything would pass
every negative assertion here and be useless.

**A role is a floor.** A viewer may read and may not write; a member may write and may not
administer. Each is asserted in both directions for the same reason.

The unauthenticated case is asserted too, and it is the one that is easy to get wrong: with
``security.auth.mode = none`` the caller is the operator at a loopback socket and gets admin,
which is only tolerable because the two bind refusals make that address loopback-only. Both of
those refusals are asserted here as well, because they are what this decision rests on.
"""

from __future__ import annotations

import json

import pytest

from manicule.api.app import build_app
from manicule.api.security import WEBSOCKET_SUBPROTOCOL_PREFIX
from manicule.app.bind import Bind
from manicule.app.service import ApplicationService
from manicule.core.errors import PolicyError
from tests.api.live import mounted
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

AUTHENTICATED = {"security": {"auth": {"mode": "api_key"}}}

UNAUTHORIZED = 401
FORBIDDEN = 403
OK = 200


async def _issue(backend: FakeBackend, name: str, role: str) -> str:
    """Mint a key through the service, and return its secret."""
    service = ApplicationService(backend)
    issued = await service.api_key_create(name, role=role)
    return issued.secret


@pytest.fixture
def keyed() -> tuple[FakeBackend, dict[str, str]]:
    """A backend with authentication on, and one secret per role."""
    import asyncio  # noqa: PLC0415 - the fixture is synchronous by design

    backend, _ = backend_with_a_document(**AUTHENTICATED)
    secrets = {
        role: asyncio.run(_issue(backend, f"{role}-key", role))
        for role in ("viewer", "member", "admin")
    }
    return backend, secrets


def test_a_request_with_no_key_is_refused_when_authentication_is_configured(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    """401, in the ordinary envelope, naming both header forms it would have accepted."""
    backend, _ = keyed
    with client_for(backend) as client:
        response = client.get("/api/v1/documents")
    body = envelope(response)
    assert response.status_code == UNAUTHORIZED
    assert body["ok"] is False
    assert body["error"]["type"] == "UnauthenticatedError"
    assert "X-API-Key" in body["error"]["message"]


def test_a_valid_key_is_admitted(keyed: tuple[FakeBackend, dict[str, str]]) -> None:
    """The positive control. Without it, "refuses everything" would pass the test above."""
    backend, secrets = keyed
    with client_for(backend) as client:
        response = client.get("/api/v1/documents", headers={"X-API-Key": secrets["viewer"]})
    assert response.status_code == OK
    assert envelope(response)["ok"] is True


def test_both_header_forms_are_accepted(keyed: tuple[FakeBackend, dict[str, str]]) -> None:
    """``Authorization: Bearer`` and ``X-API-Key``. Neither is a query parameter."""
    backend, secrets = keyed
    with client_for(backend) as client:
        bearer = client.get(
            "/api/v1/documents", headers={"Authorization": f"Bearer {secrets['viewer']}"}
        )
        query = client.get("/api/v1/documents", params={"api_key": secrets["viewer"]})
    assert bearer.status_code == OK
    assert query.status_code == UNAUTHORIZED, (
        "a key in the query string was accepted. It would then be in the access log, the "
        "browser history and every Referer the page sends."
    )


def test_a_viewer_may_read_and_may_not_write(keyed: tuple[FakeBackend, dict[str, str]]) -> None:
    """Both halves, because either alone is satisfied by a surface that always says the same."""
    backend, secrets = keyed
    headers = {"X-API-Key": secrets["viewer"]}
    with client_for(backend) as client:
        read = client.get("/api/v1/documents", headers=headers)
        write = client.post("/api/v1/tags", json={"name": "runbook"}, headers=headers)
    assert read.status_code == OK
    assert write.status_code == FORBIDDEN
    assert envelope(write)["error"]["type"] == "ForbiddenError"


def test_a_member_may_write_and_may_not_administer(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    backend, secrets = keyed
    headers = {"X-API-Key": secrets["member"]}
    with client_for(backend) as client:
        write = client.post("/api/v1/tags", json={"name": "runbook"}, headers=headers)
        administer = client.get("/api/v1/admin/query-logs", headers=headers)
    assert write.status_code == OK
    assert administer.status_code == FORBIDDEN


def test_an_admin_may_do_both(keyed: tuple[FakeBackend, dict[str, str]]) -> None:
    backend, secrets = keyed
    headers = {"X-API-Key": secrets["admin"]}
    with client_for(backend) as client:
        assert client.post("/api/v1/tags", json={"name": "x"}, headers=headers).status_code == OK
        assert client.get("/api/v1/admin/query-logs", headers=headers).status_code == OK


def test_a_revoked_key_stops_working(keyed: tuple[FakeBackend, dict[str, str]]) -> None:
    """Revocation is immediate, and the refusal is the same one an unknown key gets.

    Telling a caller that their key is merely *revoked* confirms it was once real, which is a
    fact worth having if you are collecting them.
    """
    import asyncio  # noqa: PLC0415 - the revocation runs outside the client's loop

    backend, secrets = keyed
    with client_for(backend) as client:
        headers = {"X-API-Key": secrets["viewer"]}
        assert client.get("/api/v1/documents", headers=headers).status_code == OK
        asyncio.run(ApplicationService(backend).api_key_revoke("viewer-key"))
        after = client.get("/api/v1/documents", headers=headers)
    assert after.status_code == UNAUTHORIZED
    assert "revoked" not in envelope(after)["error"]["message"].lower()


def test_an_unrecognized_key_is_refused(keyed: tuple[FakeBackend, dict[str, str]]) -> None:
    backend, _ = keyed
    with client_for(backend) as client:
        response = client.get("/api/v1/documents", headers={"X-API-Key": "mnk_not_a_key"})
    assert response.status_code == UNAUTHORIZED


def test_with_no_authentication_configured_the_local_caller_is_an_operator() -> None:
    """``auth.mode = none`` means loopback, and a loopback caller already has the CLI.

    Stated as a test because it is the one place this surface grants authority without a
    credential, and the next two tests are what make it safe.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        assert client.get("/api/v1/admin/query-logs").status_code == OK
        identity = envelope(client.get("/auth/session"))["data"]
    assert identity["authenticated"] is False
    assert identity["mode"] == "none"


def test_an_unauthenticated_application_is_refused_on_a_non_loopback_configured_host() -> None:
    """The application is refused, not merely the bind.

    ``resolve_bind`` refuses the *address* before a socket exists. This refuses the
    *application*, so a deployment that starts the ASGI app under somebody else's server —
    a container entry point, a production server, a hand-written uvicorn call — hits the same
    wall.
    """
    backend, _ = backend_with_a_document(security={"transport": {"bind_host": "192.0.2.10"}})
    with pytest.raises(PolicyError, match=r"security\.auth\.mode"):
        build_app(ApplicationService(backend))


def test_a_decided_bind_is_what_the_refusal_reads_when_one_is_given() -> None:
    """``--host`` can name an address configuration does not, so the decided one wins.

    Both directions: a wide decided bind is refused even though configuration says loopback,
    and a loopback decided bind is allowed even though configuration says otherwise.
    """
    backend, _ = backend_with_a_document()
    with pytest.raises(PolicyError):
        build_app(
            ApplicationService(backend),
            bind=Bind(host="192.0.2.10", port=8765, loopback=False),
        )
    wide_config, _ = backend_with_a_document(security={"transport": {"bind_host": "192.0.2.10"}})
    assert build_app(
        ApplicationService(wide_config), bind=Bind(host="127.0.0.1", port=8765, loopback=True)
    )


def test_an_authenticated_application_may_be_built_on_a_wide_host() -> None:
    """The positive control for the refusal above."""
    backend, _ = backend_with_a_document(
        security={"transport": {"bind_host": "192.0.2.10"}, "auth": {"mode": "api_key"}}
    )
    assert build_app(ApplicationService(backend))


def test_the_websocket_refuses_an_unauthenticated_handshake(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    """Closed **before** ``accept``, so no question is ever queued on the connection."""
    from starlette.websockets import WebSocketDisconnect  # noqa: PLC0415 - only this test needs it

    backend, _ = keyed

    def connect() -> None:
        with client_for(backend) as client, client.websocket_connect("/api/v1/chat/ws"):
            pass  # pragma: no cover - the handshake is refused before this runs

    with pytest.raises(WebSocketDisconnect):
        connect()


def test_the_websocket_accepts_a_key_offered_as_a_subprotocol(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    """The positive control, and the browser's only way to present one.

    A browser cannot set headers on a ``WebSocket``. The usual workaround puts the key in the
    query string, where it lands in the access log; the subprotocol header does not.
    """
    backend, secrets = keyed
    protocol = f"{WEBSOCKET_SUBPROTOCOL_PREFIX}{secrets['member']}"
    with (
        client_for(backend) as client,
        client.websocket_connect("/api/v1/chat/ws", subprotocols=[protocol]) as socket,
    ):
        socket.send_text('{"question": "does the client retry"}')
        first = socket.receive_json()
    assert first["event"] in {"delta", "citation", "final"}


def test_the_websocket_accepts_a_key_in_the_authorization_header(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    """For a non-browser client, which can set headers."""
    backend, secrets = keyed
    with (
        client_for(backend) as client,
        client.websocket_connect(
            "/api/v1/chat/ws", headers={"Authorization": f"Bearer {secrets['member']}"}
        ) as socket,
    ):
        socket.send_text('{"question": "does the client retry"}')
        assert socket.receive_json()["event"]


# --- the mounted MCP surface ------------------------------------------------------------------


async def test_the_mcp_mount_refuses_a_caller_the_routes_would_refuse(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    """Two surfaces on one port cannot disagree about who is admitted.

    **A mount is not a route, and that is the whole of how this was missed.** ``identify``
    resolves a principal for every request, but :func:`~manicule.api.security.resolve` never
    raises for a bad credential — the anonymous routes are reached through the same resolution
    — and ``require`` was reached only through a FastAPI ``Depends``. A Starlette ``Mount`` is
    an opaque ASGI application, so no dependency of this application ran beneath it, and
    nothing in :mod:`manicule.mcp` checked for itself.

    So with ``auth.mode = api_key``, ``GET /api/v1/documents`` answered 401 while an anonymous
    ``tools/call`` on the same process answered ``{"ok": true}`` over the same corpus. Reads
    only — the mount carries the reading and dry-running tools — but that is the whole corpus
    and the whole configuration, including ``data_dir``, to anyone who can route a packet.

    The configuration is not a corner: ``_require_auth_for_wide_bind`` *forces* a mode other
    than ``none`` for a non-loopback bind, so the exposed case is exactly the one an operator
    is made to configure before publishing the port.
    """
    backend, _ = keyed
    with pytest.raises(BaseException, match=r"401|Unauthorized|TaskGroup"):
        async with mounted(backend) as client:
            await client.list_tools()


async def test_the_mcp_mount_admits_a_caller_the_routes_would_admit(
    keyed: tuple[FakeBackend, dict[str, str]],
) -> None:
    """The positive control. Without it, "refuses everything" would pass the test above.

    A viewer key is the floor the read routes ask for, and the mount carries the read surface,
    so the same key that reads a document over HTTP must drive a tool call here.
    """
    backend, secrets = keyed
    async with mounted(backend, credential={"X-API-Key": secrets["viewer"]}) as client:
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "retry policy"})

    assert tools, "an authenticated caller must still see the tool surface"
    assert json.loads(result.content[0].text)["ok"] is True


async def test_the_mcp_mount_is_open_when_nothing_is_configured() -> None:
    """The shipped posture is loopback with no credential, and it must stay reachable.

    The guard asks :func:`~manicule.api.security.require`, which admits anybody when the mode is
    ``none`` — so this is the assertion that the fix did not quietly make the default install
    require a key nobody has issued.
    """
    backend, _ = backend_with_a_document()
    async with mounted(backend) as client:
        assert await client.list_tools()
