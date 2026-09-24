"""Administering people over HTTP, and the installations the application refuses to be.

The routes are thin — each calls one service method — so what is asserted here is the part a
surface owns: the admin floor, the operation each route names on a refusal, and that the
service's refusals arrive in the ordinary envelope at the status their type implies. The rules
themselves are ``tests/app/test_people.py``'s.

The second half is ``build_app``'s own refusals: an installation whose sign-in could never
complete, and a team installation without authentication, are not built at all — because an
ASGI application can be started by something that never went through ``manicule serve``.
"""

from __future__ import annotations

from typing import Any

import pytest

from manicule.api.app import build_app
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.core.errors import PolicyError
from tests.api.identity_provider import (
    FakeIdentityProvider,
    client_with,
    google,
    oauth_backend,
    sign_in,
)
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

OK = 200
UNAUTHORIZED = 401
FORBIDDEN = 403
NOT_FOUND = 404


def _admin_backend() -> FakeBackend:
    """An installation whose provider makes the first person to sign in an administrator."""
    backend = oauth_backend(google(role="admin"))
    backend.users_.add_member("bob", role="member", email="bob@example.org")
    return backend


def test_an_administrator_lists_the_members_of_this_workspace() -> None:
    backend = _admin_backend()
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        listed = envelope(client.get("/api/v1/auth/users"))
    assert listed["ok"] is True
    assert listed["op"] == "user_list"
    assert {user["email"] for user in listed["data"]["users"]} == {
        "alice@example.org",
        "bob@example.org",
    }
    assert listed["data"]["admins"] == 1


def test_a_member_may_not_administer_people() -> None:
    """The floor is admin, as it is for keys: both are the workspace's identities."""
    backend = oauth_backend()  # the provider's default role is member
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        listed = client.get("/api/v1/auth/users")
        changed = client.patch("/api/v1/auth/users/anybody", json={"role": "admin"})
        ended = client.post("/api/v1/auth/users/anybody/sign-out")
    for response, op in ((listed, "user_list"), (changed, "user_update"), (ended, "user_sign_out")):
        assert response.status_code == FORBIDDEN
        body = envelope(response)
        assert body["op"] == op
        assert "you have 'member'" in body["error"]["message"]


def test_an_anonymous_caller_is_refused_naming_the_operation() -> None:
    with client_with(_admin_backend(), FakeIdentityProvider()) as client:
        refused = client.get("/api/v1/auth/users")
    assert refused.status_code == UNAUTHORIZED
    body = envelope(refused)
    assert body["op"] == "user_list"
    assert "/auth/providers" in body["error"]["message"], "the refusal did not say how to sign in"


def test_an_administrator_changes_a_role_by_address_and_a_standing_by_id() -> None:
    backend = _admin_backend()
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        demoted = envelope(
            client.patch("/api/v1/auth/users/bob@example.org", json={"role": "viewer"})
        )
        disabled = envelope(client.patch("/api/v1/auth/users/bob", json={"disabled": True}))
    assert demoted["data"]["user"]["role"] == "viewer"
    assert demoted["data"]["previous_role"] == "member"
    assert disabled["data"]["user"]["disabled"] is True


def test_the_last_administrator_is_refused_over_http_as_it_is_everywhere() -> None:
    """A 403 in the ordinary envelope, with the service's own words about what to do instead."""
    backend = _admin_backend()
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        me = envelope(client.get("/auth/session"))["data"]["user_id"]
        refused = client.patch(f"/api/v1/auth/users/{me}", json={"role": "member"})
    assert refused.status_code == FORBIDDEN
    body = envelope(refused)
    assert body["error"]["type"] == "PolicyError"
    assert "set-role" in body["error"]["message"]


def test_an_empty_change_and_an_unknown_member_are_refused_by_the_service() -> None:
    """The route decides neither; the service's refusals arrive at the status their type implies."""
    with client_with(_admin_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        empty = client.patch("/api/v1/auth/users/bob", json={})
        unknown = client.patch("/api/v1/auth/users/carol", json={"role": "viewer"})
        extra = client.patch("/api/v1/auth/users/bob", json={"role": "viewer", "owner": True})
    assert empty.status_code == 400
    assert "nothing to change" in envelope(empty)["error"]["message"]
    assert unknown.status_code == NOT_FOUND
    assert extra.status_code == 422, "a field the body does not have was silently ignored"


def test_signing_a_member_out_everywhere_ends_every_browser_they_are_signed_in_with() -> None:
    """Two browsers, one person: signing them out from one ends the other's session too."""
    backend = _admin_backend()
    with (
        client_with(backend, FakeIdentityProvider()) as laptop,
        client_with(backend, FakeIdentityProvider()) as phone,
    ):
        sign_in(laptop)
        sign_in(phone)
        alice = envelope(phone.get("/auth/session"))["data"]["user_id"]
        assert phone.get("/api/v1/documents").status_code == OK

        ended = envelope(laptop.post(f"/api/v1/auth/users/{alice}/sign-out"))
        assert ended["data"]["sessions_revoked"] == 2
        assert phone.get("/api/v1/documents").status_code == UNAUTHORIZED


def test_the_people_routes_answer_under_no_authentication_with_nobody_in_them() -> None:
    """The shipped posture: the operator at this machine is an administrator; nobody signs in."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        listed = envelope(client.get("/api/v1/auth/users"))
    assert listed["ok"] is True
    assert listed["data"]["count"] == 0


# --- applications that are not built ------------------------------------------------------------


def _oauth(**auth: Any) -> Settings:
    fields: dict[str, Any] = {
        "mode": "oauth",
        "session_secret": "k" * 40,
        "providers": [google()],
    }
    fields.update(auth)
    return Settings(security={"auth": fields, "transport": {"enforce_https": True}})  # pyright: ignore[reportArgumentType]


def test_a_sign_in_that_could_not_complete_is_not_served() -> None:
    """Refused where the application is made, naming the setting, rather than at a callback."""
    with pytest.raises(PolicyError, match="session_secret"):
        build_app(ApplicationService(FakeBackend(settings=_oauth(session_secret=None))))
    with pytest.raises(PolicyError, match="no OAuth provider applies"):
        build_app(ApplicationService(FakeBackend(settings=_oauth(providers=[]))))
    with pytest.raises(PolicyError, match="admits nobody"):
        build_app(
            ApplicationService(FakeBackend(settings=_oauth(providers=[google(allowed_domains=[])])))
        )


def test_a_sign_in_that_can_complete_is_served() -> None:
    """The control."""
    assert build_app(ApplicationService(FakeBackend(settings=_oauth())))


def test_team_mode_without_authentication_is_not_served_even_on_loopback() -> None:
    """The second of team mode's two refusals, for an application something else will serve."""
    settings = Settings(mode="team")  # pyright: ignore[reportArgumentType]
    with pytest.raises(PolicyError, match="team mode"):
        build_app(ApplicationService(FakeBackend(settings=settings)))


def test_team_mode_refuses_no_authentication_rather_than_letting_it_waive_the_rule() -> None:
    settings = Settings(mode="team", security={"auth": {"mode": "api_key"}})  # pyright: ignore[reportArgumentType]
    assert build_app(ApplicationService(FakeBackend(settings=settings))), "control failed"
    with pytest.raises(PolicyError, match="single-operator"):
        build_app(ApplicationService(FakeBackend(settings=settings)), allow_unauthenticated=True)
