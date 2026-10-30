"""A browser signs in with a key a person owns, and a program keeps sending the key.

The pairs that give each property its meaning: the form post that succeeds beside the one with a
wrong key, the cookie that works on the surface beside the same cookie refused under ``/mcp``, the
session that ends beside the key that revoked it. A sign-in that accepted every key, or a cookie
that opened every door, would pass any one of the negatives alone.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from manicule.app.service import ApplicationService
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

SECRET = "k" * 40
OK = 200
SEE_OTHER = 303
UNAUTHORIZED = 401
FORBIDDEN = 403
TOO_MANY = 429

BROWSER = {"accept": "text/html"}


def _backend(*, session_secret: str | None = SECRET, **auth: Any) -> FakeBackend:
    configured: dict[str, Any] = {"mode": "api_key", **auth}
    if session_secret is not None:
        configured["session_secret"] = session_secret
    backend, _ = backend_with_a_document(
        security={
            "auth": configured,
            "transport": {"enforce_https": False},
            "audit": {"enabled": True},
        }
    )
    return backend


async def _person(backend: FakeBackend, handle: str = "alice", role: str = "member") -> str:
    """Create a local person through the service and return their first key's secret."""
    created = await ApplicationService(backend).user_create(handle, role=role)
    return created.secret


@pytest.fixture
def keyed() -> tuple[FakeBackend, str]:
    backend = _backend()
    return backend, asyncio.run(_person(backend))


def _sign_in(client: TestClient, key: str, **headers: str) -> Any:
    return client.post(
        "/auth/key", data={"key": key}, headers=headers or BROWSER, follow_redirects=False
    )


def test_a_browser_signs_in_with_a_key_and_is_sent_to_the_front_door(
    keyed: tuple[FakeBackend, str],
) -> None:
    backend, key = keyed
    with client_for(backend) as client:
        response = _sign_in(client, key)
        assert response.status_code == SEE_OTHER
        assert response.headers["location"] == "/"
        assert "manicule_session" in response.cookies
        who = client.get("/auth/session")
    body = envelope(who)["data"]
    assert body["authenticated"] is True
    assert body["via"] == "session"
    assert body["role"] == "member"


def test_the_session_cookie_is_httponly_and_strict(keyed: tuple[FakeBackend, str]) -> None:
    backend, key = keyed
    with client_for(backend) as client:
        header = _sign_in(client, key).headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=strict" in header


def test_a_wrong_key_is_refused_as_a_page_and_sets_no_cookie(
    keyed: tuple[FakeBackend, str],
) -> None:
    backend, _ = keyed
    with client_for(backend) as client:
        response = _sign_in(client, "mnk_not_a_key")
    assert response.status_code == FORBIDDEN
    assert "text/html" in response.headers["content-type"]
    assert "was not accepted" in response.text
    assert "set-cookie" not in response.headers


def test_a_program_gets_the_envelope_without_the_session_token(
    keyed: tuple[FakeBackend, str],
) -> None:
    backend, key = keyed
    with client_for(backend) as client:
        response = client.post("/auth/key", json={"key": key})
    assert response.status_code == OK
    data = envelope(response)["data"]
    assert data["user"]["name"] == "alice"
    assert "token" not in data, "the session secret is for the cookie alone"
    assert "manicule_session" in response.cookies


def test_a_key_in_the_url_is_not_a_key(keyed: tuple[FakeBackend, str]) -> None:
    backend, key = keyed
    with client_for(backend) as client:
        response = client.post("/auth/key", params={"key": key}, headers=BROWSER)
    assert response.status_code == FORBIDDEN


def test_an_unowned_key_cannot_start_a_session_and_the_refusal_says_what_to_do() -> None:
    backend = _backend()
    unowned = asyncio.run(ApplicationService(backend).api_key_create("operator")).secret
    with client_for(backend) as client:
        response = _sign_in(client, unowned)
    assert response.status_code == FORBIDDEN
    assert "auth add-user" in response.text


def test_the_session_ends_when_its_key_is_revoked(keyed: tuple[FakeBackend, str]) -> None:
    backend, key = keyed
    with client_for(backend) as client:
        _sign_in(client, key)
        assert envelope(client.get("/auth/session"))["data"]["authenticated"] is True
        asyncio.run(ApplicationService(backend).api_key_revoke("alice-key"))
        assert envelope(client.get("/auth/session"))["data"]["authenticated"] is False


def test_a_viewer_key_gives_a_viewer_session_whatever_its_owner_is() -> None:
    backend = _backend()
    service = ApplicationService(backend)
    asyncio.run(service.user_create("root", role="admin"))
    viewer_key = asyncio.run(service.api_key_create("read-only", role="viewer", user="root")).secret
    with client_for(backend) as client:
        _sign_in(client, viewer_key)
        read = client.get("/api/v1/documents")
        write = client.post("/api/v1/tags", json={"name": "runbook"})
    assert read.status_code == OK
    assert write.status_code == FORBIDDEN


def test_the_cookie_is_not_a_credential_under_the_mcp_mount(
    keyed: tuple[FakeBackend, str],
) -> None:
    """A browser's ambient cookie must not reach a surface a program uses with a header."""
    backend, key = keyed
    with client_for(backend) as client:
        _sign_in(client, key)
        response = client.post("/mcp/", json={})
    assert response.status_code == UNAUTHORIZED


def test_a_header_key_still_wins_over_a_cookie(keyed: tuple[FakeBackend, str]) -> None:
    backend, key = keyed
    with client_for(backend) as client:
        _sign_in(client, key)
        response = client.get("/auth/session", headers={"X-API-Key": "mnk_wrong"})
    assert envelope(response)["data"]["authenticated"] is False


def test_nothing_is_offered_without_a_session_secret() -> None:
    """Existing key-only installations keep starting; the browser sign-in just is not there."""
    backend = _backend(session_secret=None)
    key = asyncio.run(_person(backend))
    with client_for(backend) as client:
        response = _sign_in(client, key)
        providers = envelope(client.get("/auth/providers"))["data"]
    assert response.status_code == 404
    assert "session_secret" in response.text
    assert providers["key_login_path"] == ""


def test_the_login_page_offers_a_key_form_when_it_can_work(
    keyed: tuple[FakeBackend, str],
) -> None:
    backend, _ = keyed
    with client_for(backend) as client:
        page = client.get("/ui/login")
        refused = client.get("/ui/documents")
    assert 'action="/auth/key"' in page.text
    assert 'type="password"' in page.text
    assert 'href="/ui/login"' in refused.text, "the refusal does not lead to the sign-in"


def test_guessing_keys_at_the_form_is_throttled_like_guessing_them_in_a_header(
    keyed: tuple[FakeBackend, str],
) -> None:
    backend, _ = keyed
    statuses: list[int] = []
    with client_for(backend) as client:
        for attempt in range(60):
            statuses.append(_sign_in(client, f"mnk_guess_{attempt}").status_code)
    assert TOO_MANY in statuses, "sixty wrong keys were all merely refused"


def test_a_disabled_person_cannot_sign_in_and_loses_their_session() -> None:
    backend = _backend()
    service = ApplicationService(backend)
    asyncio.run(service.user_create("root", role="admin"))
    key = asyncio.run(_person(backend, "bob"))
    with client_for(backend) as client:
        _sign_in(client, key)
        asyncio.run(service.user_disable("bob"))
        assert envelope(client.get("/auth/session"))["data"]["authenticated"] is False
        assert _sign_in(client, key).status_code == FORBIDDEN


def test_only_an_administrator_creates_a_person_over_http() -> None:
    backend = _backend()
    admin_key = asyncio.run(_person(backend, "root", "admin"))
    member_key = asyncio.run(_person(backend, "alice"))
    with client_for(backend) as client:
        refused = client.post(
            "/api/v1/auth/users",
            json={"handle": "carol"},
            headers={"X-API-Key": member_key},
        )
        created = client.post(
            "/api/v1/auth/users",
            json={"handle": "carol", "role": "viewer"},
            headers={"X-API-Key": admin_key},
        )
        again = client.post(
            "/api/v1/auth/users", json={"handle": "carol"}, headers={"X-API-Key": admin_key}
        )
        invalid = client.post(
            "/api/v1/auth/users", json={"handle": "Not A Handle"}, headers={"X-API-Key": admin_key}
        )
    assert refused.status_code == FORBIDDEN
    assert created.status_code == OK
    data = envelope(created)["data"]
    assert data["user"]["role"] == "viewer"
    assert data["secret"].startswith("mnk_")
    assert again.status_code == 409
    assert invalid.status_code == 400
