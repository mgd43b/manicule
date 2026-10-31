"""A person signs in with a handle and password, and manages their own credentials after.

Pairs again: the sign-in that works beside every way it can fail, and the failures beside each
other, because the property that matters is that they are *indistinguishable*. The other
property is that a password is never in anything the server says back.
"""

# ruff: noqa: S105 - the passwords in this file are test fixtures, not credentials

from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient

from manicule.app.service import ApplicationService
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

SECRET = "k" * 40
PASSWORD = "a long enough passphrase"
OTHER = "another long passphrase!"
OK = 200
SEE_OTHER = 303
BAD_REQUEST = 400
FORBIDDEN = 403
TOO_MANY = 429

BROWSER = {"accept": "text/html"}


def _backend(**rate_limit: Any) -> FakeBackend:
    backend, _ = backend_with_a_document(
        security={
            "auth": {"mode": "api_key", "session_secret": SECRET},
            "transport": {"enforce_https": False},
            "audit": {"enabled": True},
            "rate_limit": rate_limit,
        }
        if rate_limit
        else {
            "auth": {"mode": "api_key", "session_secret": SECRET},
            "transport": {"enforce_https": False},
            "audit": {"enabled": True},
        }
    )
    return backend


async def _admin(backend: FakeBackend) -> str:
    created = await ApplicationService(backend).user_create("root", role="admin")
    return created.secret


def _people(backend: FakeBackend, **people: str) -> None:
    service = ApplicationService(backend)
    for handle, password in people.items():
        asyncio.run(service.user_create(handle, password=password))


def _sign_in(client: TestClient, handle: str, password: str, **headers: str) -> Any:
    return client.post(
        "/auth/password",
        data={"handle": handle, "password": password},
        headers=headers or BROWSER,
        follow_redirects=False,
    )


def test_a_browser_signs_in_with_a_handle_and_password() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    with client_for(backend) as client:
        response = _sign_in(client, "Alice", PASSWORD)
        assert response.status_code == SEE_OTHER
        assert response.headers["location"] == "/"
        header = response.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=strict" in header
        who = envelope(client.get("/auth/session"))["data"]
    assert who["authenticated"] is True
    assert who["via"] == "session"
    assert who["role"] == "member"


def test_a_program_gets_the_envelope_without_the_session_token() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    with client_for(backend) as client:
        response = client.post("/auth/password", json={"handle": "alice", "password": PASSWORD})
    assert response.status_code == OK
    data = envelope(response)["data"]
    assert data["user"]["name"] == "alice"
    assert "token" not in data
    assert "manicule_session" in response.cookies


def test_every_way_of_failing_looks_the_same() -> None:
    """Unknown handle, wrong password, no password, an identity-provider shape, disabled."""
    backend = _backend()
    _people(backend, alice=PASSWORD, dave=PASSWORD)
    asyncio.run(ApplicationService(backend).user_create("keyonly"))
    asyncio.run(ApplicationService(backend).user_disable("dave"))
    attempts = [
        ("alice", OTHER),
        ("nobody", PASSWORD),
        ("keyonly", PASSWORD),
        ("dave", PASSWORD),
        ("", ""),
    ]
    seen: set[tuple[int, str]] = set()
    with client_for(backend) as client:
        for handle, password in attempts:
            response = _sign_in(client, handle, password)
            assert "set-cookie" not in response.headers
            seen.add((response.status_code, response.text))
    assert len(seen) == 1, f"the refusals differ: {seen}"
    assert next(iter(seen))[0] == FORBIDDEN


def test_a_password_is_never_in_what_comes_back() -> None:
    backend = _backend()
    admin = asyncio.run(_admin(backend))
    secret = "this one is sensitive, truly"
    with client_for(backend) as client:
        responses = [
            _sign_in(client, "alice", secret),
            client.post("/auth/password", json={"handle": "alice", "password": secret}),
            client.post("/auth/password", json={"handle": "alice", "password": [secret]}),
            client.post("/auth/password", content=secret * 400, headers={"content-type": "json"}),
            client.post(
                "/api/v1/auth/users",
                json={"handle": "Not A Handle", "password": secret},
                headers={"X-API-Key": admin},
            ),
            client.post(
                "/api/v1/auth/users",
                json={"handle": "bob", "password": "short"},
                headers={"X-API-Key": admin},
            ),
            client.put(
                "/api/v1/auth/users/root/password",
                json={"password": secret + "x" * 300},
                headers={"X-API-Key": admin},
            ),
        ]
    for response in responses:
        assert secret not in response.text


def test_a_person_created_with_a_password_has_no_key_and_a_short_password_creates_nobody() -> None:
    backend = _backend()
    admin = asyncio.run(_admin(backend))
    with client_for(backend) as client:
        short = client.post(
            "/api/v1/auth/users",
            json={"handle": "bob", "password": "short"},
            headers={"X-API-Key": admin},
        )
        created = client.post(
            "/api/v1/auth/users",
            json={"handle": "bob", "password": PASSWORD},
            headers={"X-API-Key": admin},
        )
        listed = client.get("/api/v1/auth/users", headers={"X-API-Key": admin})
    assert short.status_code == BAD_REQUEST
    data = envelope(created)["data"]
    assert data["key"] is None
    assert data["secret"] == ""
    assert data["user"]["has_password"] is True
    people = {u["name"]: u for u in envelope(listed)["data"]["users"]}
    assert people["bob"]["has_password"] is True
    assert "password_hash" not in listed.text
    assert "scrypt" not in listed.text, "a hash must never be in a result"


def test_a_signed_in_member_makes_and_revokes_their_own_keys_and_no_one_elses() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD, bob=PASSWORD)
    with client_for(backend) as alice, client_for(backend) as bob:
        _sign_in(alice, "alice", PASSWORD)
        _sign_in(bob, "bob", PASSWORD)
        minted = alice.post("/api/v1/auth/keys", json={"name": "laptop", "role": "member"})
        other = bob.post("/api/v1/auth/keys", json={"name": "bobs", "role": "member"})
        mine = envelope(alice.get("/api/v1/auth/keys"))["data"]["keys"]
        foreign = alice.delete(f"/api/v1/auth/keys/{envelope(other)['data']['key']['id']}")
        revoked = alice.delete("/api/v1/auth/keys/laptop")
        above = alice.post("/api/v1/auth/keys", json={"name": "grand", "role": "admin"})
    assert minted.status_code == OK
    assert {key["name"] for key in mine} == {"laptop"}, "a member sees only their own keys"
    assert foreign.status_code == 404, "someone else's key is indistinguishable from a missing one"
    assert revoked.status_code == OK
    assert above.status_code == FORBIDDEN, "a member cannot mint above their own role"


def test_changing_a_password_needs_the_current_one_and_ends_every_other_browser() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    with client_for(backend) as first, client_for(backend) as second:
        _sign_in(first, "alice", PASSWORD)
        _sign_in(second, "alice", PASSWORD)
        wrong = first.post(
            "/api/v1/auth/password", json={"current_password": OTHER, "new_password": OTHER}
        )
        weak = first.post(
            "/api/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": "short"},
        )
        changed = first.post(
            "/api/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": OTHER},
        )
        assert envelope(first.get("/auth/session"))["data"]["authenticated"] is True, (
            "the browser that changed it is signed in again, on a fresh session"
        )
        assert envelope(second.get("/auth/session"))["data"]["authenticated"] is False
    assert wrong.status_code == FORBIDDEN
    assert weak.status_code == BAD_REQUEST
    assert changed.status_code == OK
    assert "token" not in envelope(changed)["data"]
    with client_for(backend) as client:
        assert _sign_in(client, "alice", PASSWORD).status_code == FORBIDDEN
        assert _sign_in(client, "alice", OTHER).status_code == SEE_OTHER


def test_a_key_cannot_change_a_password_however_it_is_presented() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    created = asyncio.run(ApplicationService(backend).user_create("keyed"))
    with client_for(backend) as client:
        with_header = client.post(
            "/api/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": OTHER},
            headers={"X-API-Key": created.secret},
        )
        signed = client.post("/auth/key", json={"key": created.secret})
        assert signed.status_code == OK
        from_key_session = client.post(
            "/api/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": OTHER},
        )
    assert with_header.status_code == FORBIDDEN
    assert from_key_session.status_code == FORBIDDEN


def test_an_administrator_sets_a_password_and_the_persons_sessions_end() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    admin = asyncio.run(_admin(backend))
    with client_for(backend) as alice, client_for(backend) as admin_client:
        _sign_in(alice, "alice", PASSWORD)
        refused = alice.put("/api/v1/auth/users/alice/password", json={"password": OTHER})
        done = admin_client.put(
            "/api/v1/auth/users/alice/password",
            json={"password": OTHER},
            headers={"X-API-Key": admin},
        )
        missing = admin_client.put(
            "/api/v1/auth/users/nobody/password",
            json={"password": OTHER},
            headers={"X-API-Key": admin},
        )
        assert envelope(alice.get("/auth/session"))["data"]["authenticated"] is False
        assert _sign_in(alice, "alice", PASSWORD).status_code == FORBIDDEN
        assert _sign_in(alice, "alice", OTHER).status_code == SEE_OTHER
    assert refused.status_code == FORBIDDEN
    assert done.status_code == OK
    assert missing.status_code == 404


def test_a_disabled_person_cannot_sign_in_even_with_the_right_password() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    asyncio.run(_admin(backend))
    asyncio.run(ApplicationService(backend).user_disable("alice"))
    with client_for(backend) as client:
        assert _sign_in(client, "alice", PASSWORD).status_code == FORBIDDEN


def test_guessing_passwords_is_refused_before_the_password_is_checked() -> None:
    """After the failures a minute allows, even the *right* password waits."""
    backend = _backend(failed_auth_per_minute=3)
    _people(backend, alice=PASSWORD)
    statuses: list[int] = []
    with client_for(backend) as client:
        for attempt in range(8):
            statuses.append(_sign_in(client, "alice", f"wrong password {attempt}").status_code)
        right = _sign_in(client, "alice", PASSWORD)
    assert TOO_MANY in statuses
    assert right.status_code == TOO_MANY, "a spent address is refused whatever it presents"
    assert "retry-after" in right.headers


def test_a_key_is_still_accepted_from_an_address_that_has_spent_its_password_guesses() -> None:
    backend = _backend(failed_auth_per_minute=3)
    _people(backend, alice=PASSWORD)
    key = asyncio.run(ApplicationService(backend).user_create("keyed")).secret
    with client_for(backend) as client:
        for attempt in range(8):
            _sign_in(client, "alice", f"wrong password {attempt}")
        works = client.get("/api/v1/auth/keys", headers={"X-API-Key": key})
    assert works.status_code == OK, "a correct key is never refused by the failure bucket"


def test_the_login_page_offers_both_forms_and_the_providers_say_where() -> None:
    backend = _backend()
    with client_for(backend) as client:
        page = client.get("/ui/login")
        providers = envelope(client.get("/auth/providers"))["data"]
    assert 'action="/auth/password"' in page.text
    assert 'autocomplete="username"' in page.text
    assert 'autocomplete="current-password"' in page.text
    assert 'action="/auth/key"' in page.text
    assert providers["password_login_path"] == "/auth/password"


def test_password_sign_in_is_not_offered_without_a_session_secret() -> None:
    backend, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    with client_for(backend) as client:
        response = client.post("/auth/password", json={"handle": "a", "password": PASSWORD})
        providers = envelope(client.get("/auth/providers"))["data"]
    assert response.status_code == 404
    assert providers["password_login_path"] == ""


def test_a_cross_site_password_post_is_refused_before_routing() -> None:
    backend = _backend()
    _people(backend, alice=PASSWORD)
    with client_for(backend) as client:
        response = client.post(
            "/auth/password",
            data={"handle": "alice", "password": PASSWORD},
            headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://attacker.example.com"},
        )
    assert response.status_code == FORBIDDEN
    assert "set-cookie" not in response.headers


def test_the_people_page_offers_a_password_to_the_administrator_who_adds_a_person() -> None:
    backend = _backend()
    service = ApplicationService(backend)
    asyncio.run(service.user_create("root", role="admin", password=PASSWORD))
    asyncio.run(service.user_create("alice", password=PASSWORD))
    with client_for(backend) as client:
        _sign_in(client, "root", PASSWORD)
        page = client.get("/ui/users")
    assert page.status_code == OK
    assert 'name="password"' in page.text
    assert 'autocomplete="new-password"' in page.text
    assert "data-user-password=" in page.text, "no control to set a person's password"
    assert "scrypt" not in page.text


def test_only_a_password_session_is_offered_the_change_password_form() -> None:
    backend = _backend()
    service = ApplicationService(backend)
    asyncio.run(service.user_create("alice", password=PASSWORD))
    keyed = asyncio.run(service.user_create("keyed"))
    with client_for(backend) as with_password, client_for(backend) as with_key:
        _sign_in(with_password, "alice", PASSWORD)
        with_key.post("/auth/key", json={"key": keyed.secret})
        offered = with_password.get("/ui/auth")
        withheld = with_key.get("/ui/auth")
    assert "data-change-password" in offered.text
    assert 'autocomplete="current-password"' in offered.text
    assert "data-change-password" not in withheld.text


def test_a_lone_surrogate_in_a_password_is_a_refusal_not_a_crash() -> None:
    backend = _backend()
    admin = asyncio.run(_admin(backend))
    body = b'{"handle": "bob", "password": "\\ud800 not text but long enough"}'
    with client_for(backend) as client:
        created = client.post(
            "/api/v1/auth/users",
            content=body,
            headers={"content-type": "application/json", "X-API-Key": admin},
        )
        signed = client.post(
            "/auth/password",
            content=body.replace(b'"bob"', b'"alice"'),
            headers={"content-type": "application/json"},
        )
    assert created.status_code == BAD_REQUEST
    assert signed.status_code == FORBIDDEN


def test_a_password_that_is_not_a_string_is_refused_not_ignored() -> None:
    backend = _backend()
    admin = asyncio.run(_admin(backend))
    with client_for(backend) as client:
        created = client.post(
            "/api/v1/auth/users",
            json={"handle": "bob", "password": ["a long enough passphrase"]},
            headers={"X-API-Key": admin},
        )
    assert created.status_code == BAD_REQUEST
    assert "passphrase" not in created.text
