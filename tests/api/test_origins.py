"""A page on another origin may not change anything here.

The threat is specific and it arrived with the browser surface. manicule's shipped posture is
loopback with ``security.auth.mode = none``: there is no credential, so *any* request that
reaches the port is the operator's. A page on the internet cannot read the response — there is
no CORS header to let it — but with a "simple" request it does not need to. The effect happens.

Three halves, which is one more than the obvious two.
:func:`~manicule.api.origins.permitted` is a pure function of four header values and is
exercised directly, including the cases a browser would send; the middleware is driven through
the real application, because a decision function nothing consults is a decision nobody makes;
and **the websocket is driven separately**, because an HTTP middleware never sees a websocket
scope. That last one is not a completeness exercise: a browser applies no cross-origin policy to
a ``WebSocket`` — no preflight, no CORS — so the page **reads** every frame, which makes the
socket the one place a cross-origin connection gets the corpus rather than only an effect.

The control runs through every one of these: a request from a **program** — which sends neither
header — is admitted. Refusing those would break every non-browser client to defend against a
threat only browsers create, and a guard that refuses everything is not a guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.api.origins import (
    UNSAFE_METHODS,
    handshake_permitted,
    host_of,
    permitted,
)
from manicule.api.routes.sockets import POLICY_VIOLATION
from tests.api.support import backend_with_a_document, client_for, envelope

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

FORBIDDEN = 403
HOST = "127.0.0.1:8765"
SAME = "http://127.0.0.1:8765"
EVIL = "https://evil.example"
EMBEDDER = "https://docs.example.com"


# --- the decision, on its own -------------------------------------------------------------------


def test_a_safe_method_is_never_refused() -> None:
    """Reads are not on the list. A ``GET`` that changed state would be the defect, and putting
    reads behind this check would break every ordinary link into the browser surface."""
    assert "GET" not in UNSAFE_METHODS
    assert permitted("GET", fetch_site="cross-site", origin=EVIL, host=HOST, allowed_origins=())


@pytest.mark.parametrize("method", sorted(UNSAFE_METHODS))
def test_a_cross_site_write_is_refused_by_the_unforgeable_header(method: str) -> None:
    """``Sec-Fetch-Site`` is a forbidden header name: page script cannot set it."""
    assert not permitted(
        method, fetch_site="cross-site", origin=EVIL, host=HOST, allowed_origins=()
    )


@pytest.mark.parametrize("site", ["same-origin", "none"])
def test_the_browsers_own_signal_admits_a_first_party_request(site: str) -> None:
    """``same-origin`` is the page's own fetch; ``none`` is a typed URL or a bookmark."""
    assert permitted("POST", fetch_site=site, origin=SAME, host=HOST, allowed_origins=())


def test_a_sibling_subdomain_is_not_the_same_site() -> None:
    """``same-site`` is deliberately not admitted: on a shared domain it is the neighbour this
    check exists for, and a different origin either way."""
    assert not permitted(
        "POST",
        fetch_site="same-site",
        origin="https://other.example.com",
        host=HOST,
        allowed_origins=(),
    )


def test_a_configured_origin_may_still_write() -> None:
    """The widget is the reason that list exists, and a widget asks questions — which is a POST.

    Without this, switching the guard on would break the one part of manicule that is
    deliberately cross-origin.
    """
    assert permitted(
        "POST", fetch_site="cross-site", origin=EMBEDDER, host=HOST, allowed_origins=(EMBEDDER,)
    )


def test_a_request_with_neither_header_is_admitted() -> None:
    """`curl`, a script, an assistant holding a key. None has ambient authority to abuse."""
    assert permitted("POST", fetch_site=None, origin=None, host=HOST, allowed_origins=())


def test_an_origin_matching_the_host_is_admitted_without_the_newer_header() -> None:
    """The fallback for a browser too old to send ``Sec-Fetch-Site``."""
    assert permitted("POST", fetch_site=None, origin=SAME, host=HOST, allowed_origins=())
    assert not permitted("POST", fetch_site=None, origin=EVIL, host=HOST, allowed_origins=())


def test_the_scheme_is_deliberately_not_compared() -> None:
    """Behind a TLS-terminating proxy the request arrives as ``http`` and the browser's origin
    says ``https``. A check that failed there is a policy operators switch off."""
    assert permitted(
        "POST", fetch_site=None, origin="https://127.0.0.1:8765", host=HOST, allowed_origins=()
    )


@pytest.mark.parametrize(
    "value", ["127.0.0.1:8765", "https://", "https://host/path", "https://host?q=1", "null"]
)
def test_something_that_is_not_an_origin_never_matches(value: str) -> None:
    """A browser's ``Origin`` is scheme, host and port. ``null`` is what a sandboxed frame or a
    ``file://`` page sends, and it must not be able to match a host."""
    assert host_of(value) == "" or host_of(value) != HOST
    assert not permitted("POST", fetch_site=None, origin=value, host=HOST, allowed_origins=())


def test_an_empty_origin_header_reads_as_no_header_at_all() -> None:
    """Empty is treated as absent, and that is not a hole.

    No browser sends an empty ``Origin``: an opaque origin — a sandboxed frame, a ``file://``
    page — sends the literal ``null``, which is covered above and matches no host. A blank value
    comes from a client that constructed the header itself, which is a program, which is the
    carve-out this guard is written around.
    """
    assert permitted("POST", fetch_site=None, origin="", host=HOST, allowed_origins=())
    assert not permitted("POST", fetch_site="cross-site", origin="", host=HOST, allowed_origins=())


# --- the same decision, through the application -------------------------------------------------


def test_a_cross_site_post_is_refused_by_the_running_application() -> None:
    """The envelope, the status and the reason — the ordinary shape, not a bare 403."""
    backend, document = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post(
            f"/api/v1/documents/{document.id}/restore",
            headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site", "Host": HOST},
        )
    body = envelope(response)
    assert response.status_code == FORBIDDEN
    assert body["ok"] is False
    assert body["error"]["type"] == "PolicyError"
    assert EVIL in body["error"]["message"]


def test_a_cross_site_form_post_is_refused_even_with_no_unusual_header() -> None:
    """The case CORS does **not** cover.

    A form submission is a "simple" request: the browser sends it and only hides the response.
    Under ``auth.mode = none`` that is a state change caused by a page the operator merely
    visited, which is exactly what a loopback service with no credential is exposed to.
    """
    backend, document = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post(
            f"/api/v1/documents/{document.id}/reindex",
            headers={"Origin": EVIL, "Host": HOST},
            data={"anything": "1"},
        )
    assert response.status_code == FORBIDDEN
    assert backend.store.documents[document.id].status.value == "indexed"


def test_the_browser_surfaces_own_calls_are_admitted() -> None:
    """The control, through the application: a same-origin write works.

    Without this the test above would pass for a build that refused every write, which is not
    a browser surface at all.
    """
    backend, document = backend_with_a_document()
    backend.organisation_.trash[document.id] = document
    with client_for(backend) as client:
        response = client.post(
            f"/api/v1/documents/{document.id}/restore",
            headers={"Origin": SAME, "Sec-Fetch-Site": "same-origin", "Host": HOST},
        )
    assert response.status_code == 200
    assert envelope(response)["ok"] is True


def test_a_program_with_no_browser_headers_is_unaffected() -> None:
    """Every existing client keeps working. This is the compatibility control."""
    backend, document = backend_with_a_document()
    backend.organisation_.trash[document.id] = document
    with client_for(backend) as client:
        response = client.post(f"/api/v1/documents/{document.id}/restore")
    assert response.status_code == 200


def test_a_read_from_anywhere_is_still_answered() -> None:
    """Reads are unaffected, so a link into the browser surface from anywhere still works."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get(
            "/api/v1/documents", headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site"}
        )
    assert response.status_code == 200


# --- the websocket, which middleware never sees -------------------------------------------------


def test_a_websocket_handshake_from_another_origin_is_refused() -> None:
    """The worst case on this surface, and the one middleware cannot cover.

    A browser applies no cross-origin policy to a ``WebSocket``: no preflight, no CORS, and the
    page **reads** every frame. On the shipped posture — loopback, no credential — an unchecked
    handshake is any page the operator visits asking the corpus questions and getting answers.

    Refused before ``accept``, so the connection never reaches a state in which a question could
    be queued.
    """
    from starlette.websockets import WebSocketDisconnect  # noqa: PLC0415 - only this test needs it

    def ask(client: TestClient) -> None:
        """Open the socket and try to use it, which is what a hostile page would do."""
        with client.websocket_connect("/api/v1/chat/ws", headers={"Origin": EVIL}) as socket:
            socket.send_text('{"question": "what is in the corpus"}')
            socket.receive_json()

    backend, _ = backend_with_a_document()
    with client_for(backend) as client, pytest.raises(WebSocketDisconnect) as refused:
        ask(client)
    assert refused.value.code == POLICY_VIOLATION
    assert "websocket" in refused.value.reason


def test_a_same_origin_websocket_handshake_is_accepted() -> None:
    """The control. Without it the refusal above would pass for a socket nobody can open."""
    backend, _ = backend_with_a_document()
    with (
        client_for(backend) as client,
        client.websocket_connect(
            "/api/v1/chat/ws", headers={"Origin": SAME, "Host": HOST}
        ) as socket,
    ):
        socket.send_text('{"question": "does the client retry"}')
        assert socket.receive_json()["event"]


def test_a_websocket_client_that_is_not_a_browser_is_unaffected() -> None:
    """No ``Origin`` at all: a script, or an assistant holding a key."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client, client.websocket_connect("/api/v1/chat/ws") as socket:
        socket.send_text('{"question": "does the client retry"}')
        assert socket.receive_json()["event"]


def test_a_configured_origin_may_still_open_a_websocket() -> None:
    """An operator who listed an origin for the widget gets the socket too."""
    backend, _ = backend_with_a_document(security={"transport": {"allowed_origins": [EMBEDDER]}})
    with (
        client_for(backend) as client,
        client.websocket_connect("/api/v1/chat/ws", headers={"Origin": EMBEDDER}) as socket,
    ):
        socket.send_text('{"question": "does the client retry"}')
        assert socket.receive_json()["event"]


def test_the_handshake_decision_is_the_same_one_the_middleware_makes() -> None:
    """One rule, reached by two names, so the two cannot drift apart."""
    assert handshake_permitted(origin=None, host=HOST, allowed_origins=())
    assert handshake_permitted(origin=SAME, host=HOST, allowed_origins=())
    assert not handshake_permitted(origin=EVIL, host=HOST, allowed_origins=())
    assert handshake_permitted(origin=EMBEDDER, host=HOST, allowed_origins=(EMBEDDER,))
    assert not handshake_permitted(origin="null", host=HOST, allowed_origins=())


def test_a_refused_request_still_carries_the_surfaces_headers() -> None:
    """A refusal is a response like any other, and the middleware dresses it the same way."""
    backend, document = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post(
            f"/api/v1/documents/{document.id}/restore",
            headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site"},
        )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
