"""Who may sign in, who may change a membership, and what a session is — decided in the service.

Every surface reaches these rules through :class:`~manicule.app.service.ApplicationService`, so
they are asserted here against the fakes rather than through a route: a rule that held only on
the route somebody wrote a test for would be a rule the command line does not have.

Three families, each with its negative half:

* **Admission.** Re-checked on every sign-in; a verified address is required unless the provider
  admits anybody; a domain matches exactly, never as a suffix; a disabled membership is refused
  whatever the allowlist says; and the provider's role is used once, on the first sign-in.
* **Administration.** A workspace that has an enabled administrator keeps one — asserted with a
  store that does *not* enforce it, so the service's own check is what is seen to fire — and
  nobody disables themselves.
* **Sessions.** A session authenticates only in its own workspace, and a store that forgot its
  workspace predicate is refused on the way out, asserted with a store written without it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from manicule.app.caller import Caller, acting_as
from manicule.app.people import (
    Profile,
    Refusal,
    admission,
    applicable_providers,
    serving_problems,
)
from manicule.app.service import ApplicationService
from manicule.app.tenancy import CrossWorkspaceError
from manicule.config.settings import OAuthProvider, Role, Settings
from manicule.core.errors import (
    AmbiguousHandleError,
    ConfigError,
    PolicyError,
    SignInRefusedError,
    UnknownEntityError,
)
from tests.app.fakes import FakeBackend, FakeUsers, LeakyUsers

SECRET = "x" * 40
"""A fixture signing key, long enough to be accepted."""

CALLBACK = "http://127.0.0.1:8765/auth/callback/google"


def _provider(**overrides: Any) -> OAuthProvider:
    fields: dict[str, Any] = {
        "type": "google",
        "client_id": "client-id",
        "client_secret": SecretStr("client-secret"),
        "redirect_uri": CALLBACK,
        "allowed_domains": ("example.org",),
    }
    fields.update(overrides)
    return OAuthProvider(**fields)


def _settings(*providers: dict[str, Any], **overrides: Any) -> Settings:
    """Settings for an installation where people sign in through ``providers``."""
    listed = providers or (
        {
            "type": "google",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "redirect_uri": CALLBACK,
            "allowed_domains": ["example.org"],
        },
    )
    security: dict[str, Any] = {
        "auth": {"mode": "oauth", "providers": list(listed), "session_secret": SECRET},
        "transport": {"enforce_https": True},
        "audit": {"enabled": True},
    }
    return Settings(security=security, **overrides)  # pyright: ignore[reportArgumentType]


def _profile(email: str = "alice@example.org", *, verified: bool = True, **extra: Any) -> Profile:
    fields: dict[str, Any] = {
        "provider": "google",
        "subject": f"sub-{email}",
        "email": email,
        "email_verified": verified,
        "name": email.split("@", 1)[0].title(),
    }
    fields.update(extra)
    return Profile(**fields)


class _Observed(ApplicationService):
    """The service, holding a typed handle on the fake behind it for the assertions to read."""

    def __init__(self, backend: FakeBackend) -> None:
        super().__init__(backend)
        self.fake = backend


def _service(settings: Settings | None = None, *, users: FakeUsers | None = None) -> _Observed:
    backend = FakeBackend(settings=settings or _settings())
    backend.users_ = users or FakeUsers(workspace=backend.settings.workspace)
    backend.keys_.workspace = backend.settings.workspace
    return _Observed(backend)


# --- admission, as a pure rule ------------------------------------------------------------------


def test_a_verified_address_on_the_allowlist_is_admitted() -> None:
    """The positive control for every refusal below."""
    provider = _provider(allowed_emails=("alice@example.com",), allowed_domains=())
    assert admission(provider, _profile("Alice@Example.com")) is None


def test_an_unverified_address_is_refused_even_when_it_is_listed() -> None:
    """An unverified address is a claim somebody typed into a form, not an identity.

    Admitting it would let anybody who can create an account at the provider under somebody
    else's address sign in as a person the allowlist names.
    """
    provider = _provider(allowed_emails=("alice@example.org",))
    assert admission(provider, _profile(verified=False)) is Refusal.EMAIL_NOT_VERIFIED


def test_a_domain_matches_exactly_and_never_as_a_suffix() -> None:
    """``team.example.org`` and ``evil-example.org`` are other domains, whoever owns them.

    A suffix match would admit anybody who registers a domain ending in the listed one — which is
    anybody with a few dollars.
    """
    provider = _provider(allowed_domains=("example.org",))
    assert admission(provider, _profile("bob@example.org")) is None
    assert admission(provider, _profile("bob@team.example.org")) is Refusal.NOT_ADMITTED
    assert admission(provider, _profile("bob@evil-example.org")) is Refusal.NOT_ADMITTED
    assert admission(provider, _profile("example.org@elsewhere.test")) is Refusal.NOT_ADMITTED


def test_allow_any_user_admits_an_account_with_no_verified_address() -> None:
    """The switch means what it says, which is why it is a switch rather than an empty list."""
    provider = _provider(allowed_domains=(), allow_any_user=True)
    assert admission(provider, _profile(verified=False)) is None


def test_a_provider_with_no_allowlist_admits_nobody() -> None:
    """Empty is not "everyone" — for GitHub, everyone is anybody on the internet."""
    provider = _provider(allowed_domains=())
    assert not provider.admits_anybody
    assert admission(provider, _profile()) is Refusal.NOT_ADMITTED


def test_a_provider_for_another_workspace_does_not_apply_here() -> None:
    """A process serving ``beta`` offers no sign-in configured for ``alpha``."""
    settings = _settings(
        {
            "type": "google",
            "client_id": "a",
            "client_secret": "s",
            "redirect_uri": CALLBACK,
            "allowed_domains": ["example.org"],
            "workspace": "alpha",
        },
        workspace="beta",
    )
    assert applicable_providers(settings) == ()


def test_no_provider_applies_unless_people_sign_in() -> None:
    """A provider list under another mode is configuration nobody reads, and offers nothing."""
    settings = Settings(
        security={  # pyright: ignore[reportArgumentType]
            "auth": {
                "mode": "api_key",
                "providers": [
                    {
                        "type": "google",
                        "client_id": "a",
                        "client_secret": "s",
                        "allow_any_user": True,
                    }
                ],
            }
        }
    )
    assert applicable_providers(settings) == ()
    assert serving_problems(settings) == []


# --- what a served installation needs before anybody can sign in ------------------------------


def test_a_complete_sign_in_configuration_has_no_serving_problems() -> None:
    """The control. Every refusal below is one field away from this."""
    assert serving_problems(_settings()) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"session_secret": None}, "session_secret"),
        ({"session_secret": "too-short"}, "session_secret"),
        ({"providers": []}, "no OAuth provider applies"),
        ({"redirect_uri": None}, "has no redirect_uri"),
        ({"redirect_uri": "http://127.0.0.1:8765/auth/cb"}, "does not end with"),
        ({"redirect_uri": "http://manicule.example.org/auth/callback/google"}, "plain http"),
        ({"redirect_uri": "not a url"}, "not an absolute"),
        ({"allowed_domains": []}, "admits nobody"),
        ({"duplicate": True}, "two google providers"),
    ],
)
def test_each_condition_a_sign_in_needs_is_reported(change: dict[str, Any], expected: str) -> None:
    """Each is a server that starts, sends a person to a provider, and fails when they return.

    One field at a time from a configuration that has no problems, so each assertion is about
    the one condition it names rather than about whichever happened to be checked first.
    """
    provider: dict[str, Any] = {
        "type": "google",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "redirect_uri": CALLBACK,
        "allowed_domains": ["example.org"],
    }
    auth: dict[str, Any] = {"mode": "oauth", "session_secret": SECRET}
    for key in ("redirect_uri", "allowed_domains"):
        if key in change:
            provider[key] = change[key]
    providers = [provider, dict(provider)] if change.get("duplicate") else [provider]
    auth["providers"] = change.get("providers", providers)
    if "session_secret" in change:
        auth["session_secret"] = change["session_secret"]
    settings = Settings(
        security={"auth": auth, "transport": {"enforce_https": True}}  # pyright: ignore[reportArgumentType]
    )
    problems = serving_problems(settings)
    assert any(expected in problem for problem in problems), problems


def test_a_loopback_redirect_may_be_plain_http() -> None:
    """A developer's own machine is the one place a code in the clear goes nowhere."""
    assert serving_problems(_settings()) == []  # CALLBACK is http://127.0.0.1


def test_plain_http_is_accepted_everywhere_once_https_is_not_enforced() -> None:
    """``enforce_https = false`` is the operator saying the network is theirs, and it is read."""
    settings = Settings(
        security={  # pyright: ignore[reportArgumentType]
            "auth": {
                "mode": "oauth",
                "session_secret": SECRET,
                "providers": [
                    {
                        "type": "github",
                        "client_id": "a",
                        "client_secret": "s",
                        "redirect_uri": "http://manicule.lan/auth/callback/github",
                        "allowed_emails": ["alice@example.org"],
                    }
                ],
            },
            "transport": {"enforce_https": False},
        }
    )
    assert serving_problems(settings) == []


# --- signing in ---------------------------------------------------------------------------------


async def test_a_first_sign_in_makes_a_member_in_the_providers_role_and_starts_a_session() -> None:
    """Everything a first sign-in does, in order, including the audit row naming the person."""
    service = _service(
        _settings(
            {
                "type": "google",
                "client_id": "a",
                "client_secret": "s",
                "redirect_uri": CALLBACK,
                "allowed_domains": ["example.org"],
                "role": "admin",
            }
        )
    )
    signed = await service.sign_in("google", _profile())

    assert signed.user.role == "admin"
    assert signed.user.email == "alice@example.org"
    assert signed.token
    identity = await service.authenticate_session(signed.token)
    assert identity.authenticated
    assert identity.via == "session"
    assert identity.user_id == signed.user.id
    audit = service.fake.telemetry_.audits[-1]
    assert audit["event_type"] == "auth.login"
    assert audit["actor"] == signed.user.id
    assert signed.token not in str(audit["details"]), "the audit trail quoted the session token"


async def test_a_later_sign_in_does_not_overwrite_the_role_an_administrator_gave() -> None:
    """Otherwise every demotion would last until the demoted person next signed in."""
    service = _service(
        _settings(
            {
                "type": "google",
                "client_id": "a",
                "client_secret": "s",
                "redirect_uri": CALLBACK,
                "allowed_domains": ["example.org"],
                "role": "admin",
            }
        )
    )
    first = await service.sign_in("google", _profile())
    service.fake.users_.add_member("other-admin", role="admin")
    await service.user_set_role(first.user.id, "viewer")

    again = await service.sign_in("google", _profile())
    assert again.user.role == "viewer"


async def test_an_address_removed_from_the_allowlist_is_refused_at_its_next_sign_in() -> None:
    """Admission is re-checked every time, not remembered from the first.

    Two services over one people store, differing only in configuration — which is what an
    operator editing the allowlist and restarting is.
    """
    users = FakeUsers()
    before = _service(
        _settings(
            {
                "type": "google",
                "client_id": "a",
                "client_secret": "s",
                "redirect_uri": CALLBACK,
                "allowed_emails": ["alice@example.org", "bob@example.org"],
            }
        ),
        users=users,
    )
    await before.sign_in("google", _profile())

    after = _service(
        _settings(
            {
                "type": "google",
                "client_id": "a",
                "client_secret": "s",
                "redirect_uri": CALLBACK,
                "allowed_emails": ["bob@example.org"],
            }
        ),
        users=users,
    )
    with pytest.raises(SignInRefusedError) as caught:
        await after.sign_in("google", _profile())
    assert caught.value.reason == Refusal.NOT_ADMITTED.value


async def test_a_disabled_member_is_refused_whatever_the_allowlist_says() -> None:
    """Disabling somebody is an administrator's decision, and signing in again does not undo it."""
    service = _service()
    signed = await service.sign_in("google", _profile())
    service.fake.users_.add_member("admin", role="admin")
    await service.user_disable(signed.user.id)

    with pytest.raises(SignInRefusedError) as caught:
        await service.sign_in("google", _profile())
    assert caught.value.reason == Refusal.DISABLED.value
    assert "disabled" in str(caught.value)


async def test_a_refused_sign_in_is_audited_without_the_address_it_was_refused_for() -> None:
    """The address belongs to somebody who is not a member and never agreed to be recorded."""
    service = _service()
    with pytest.raises(SignInRefusedError):
        await service.sign_in("google", _profile("mallory@elsewhere.test"))

    audit = service.fake.telemetry_.audits[-1]
    assert audit["event_type"] == "auth.login_refused"
    assert audit["details"] == {"provider": "google", "reason": "not_admitted"}
    assert "mallory" not in str(audit)


async def test_a_refusal_never_says_which_allowlist_check_failed() -> None:
    """A stranger told "your domain is not listed" knows what to register next."""
    service = _service()
    with pytest.raises(SignInRefusedError) as caught:
        await service.sign_in("google", _profile("mallory@elsewhere.test"))
    message = str(caught.value).lower()
    assert "domain" not in message
    assert "elsewhere.test" not in message
    assert "not admitted" in message


async def test_a_sign_in_through_a_provider_that_does_not_apply_here_is_unknown() -> None:
    """The same refusal for "not configured" and "configured for another workspace"."""
    service = _service()
    with pytest.raises(UnknownEntityError):
        await service.sign_in("github", _profile(provider="github"))


async def test_an_identity_from_one_provider_is_not_accepted_by_another() -> None:
    """A GitHub account id presented to the Google sign-in would be recorded as a Google one."""
    service = _service()
    with pytest.raises(ConfigError):
        await service.sign_in("google", _profile(provider="github"))


# --- sessions -----------------------------------------------------------------------------------


async def test_a_session_authenticates_nothing_when_people_do_not_sign_in_here() -> None:
    """A cookie left from before a configuration change is not a credential under ``api_key``."""
    users = FakeUsers()
    signed = await _service(users=users).sign_in("google", _profile())
    keyed = _service(Settings(security={"auth": {"mode": "api_key"}}), users=users)  # pyright: ignore[reportArgumentType]

    identity = await keyed.authenticate_session(signed.token)
    assert not identity.authenticated
    assert users.resolves == 0, "the store was asked about a session this mode does not accept"


async def test_a_role_change_is_carried_by_the_very_next_request() -> None:
    """The membership is read with the session, not remembered from the sign-in."""
    service = _service()
    signed = await service.sign_in("google", _profile())
    service.fake.users_.add_member("admin", role="admin")
    await service.user_set_role(signed.user.id, "viewer")

    identity = await service.authenticate_session(signed.token)
    assert identity.role == "viewer"


async def test_signing_out_revokes_the_session_so_a_copy_of_the_cookie_stops_working() -> None:
    service = _service()
    signed = await service.sign_in("google", _profile())

    assert (await service.sign_out(signed.token)).ended
    assert not (await service.authenticate_session(signed.token)).authenticated
    assert not (await service.sign_out(signed.token)).ended, "signing out twice found a session"


async def test_another_workspaces_session_is_refused_by_the_service_when_the_store_leaks() -> None:
    """The negative half: a store that ignores its workspace predicate, and the check that holds.

    :class:`~tests.app.fakes.LeakyUsers` resolves any session it holds, whichever workspace
    minted it. The service reads the membership it was handed and refuses one that is not this
    workspace's — the check that would still fire if the store's ``WHERE`` were deleted.
    """
    leaky = LeakyUsers(workspace="beta")
    leaky.add_member("alice", workspace="alpha")
    leaky.sessions["alpha-token"] = {
        "id": "s-1",
        "user_id": "alice",
        "workspace": "alpha",
        "expires_at": _later(),
        "revoked": False,
    }
    service = _service(_settings(workspace="beta"), users=leaky)

    assert (await leaky.resolve_session("alpha-token")) is not None, "the fake stopped leaking"
    assert not (await service.authenticate_session("alpha-token")).authenticated


async def test_a_correct_store_refuses_another_workspaces_session_on_its_own() -> None:
    """The control for the test above: the fake that is not broken does not leak."""
    users = FakeUsers(workspace="beta")
    users.add_member("alice", workspace="alpha")
    users.sessions["alpha-token"] = {
        "id": "s-1",
        "user_id": "alice",
        "workspace": "alpha",
        "expires_at": _later(),
        "revoked": False,
    }
    assert (await users.resolve_session("alpha-token")) is None


async def test_a_member_list_holding_another_workspaces_member_is_refused_whole() -> None:
    """Refused rather than filtered: "the members of this workspace you may see" is a lie."""
    leaky = LeakyUsers(workspace="beta")
    leaky.add_member("alice", workspace="alpha")
    leaky.add_member("bob", workspace="beta")
    service = _service(_settings(workspace="beta"), users=leaky)

    with pytest.raises(CrossWorkspaceError) as caught:
        await service.user_list()
    assert "alice" not in str(caught.value)


def _later() -> Any:
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    return datetime.now(UTC) + timedelta(hours=1)


# --- administering people -----------------------------------------------------------------------


async def test_the_last_enabled_administrator_cannot_be_demoted_or_disabled() -> None:
    """Checked by the service, so a store that forgot the rule still cannot break it.

    ``guards_last_admin=False`` is the fake breaking its half of the bargain on purpose: the
    real store refuses too, in the same transaction as the write, and a test against it could not
    tell which of the two refused.
    """
    users = FakeUsers(guards_last_admin=False)
    users.add_member("root", role="admin")
    service = _service(users=users)

    with pytest.raises(PolicyError, match="last enabled administrator"):
        await service.user_set_role("root", "member")
    with pytest.raises(PolicyError, match="last enabled administrator"):
        await service.user_disable("root")
    assert users.memberships[("default", "root")]["role"] == "admin"


async def test_an_administrator_may_be_demoted_while_another_remains() -> None:
    """The control: the rule is about the last one, not about administrators."""
    users = FakeUsers(guards_last_admin=False)
    users.add_member("root", role="admin")
    users.add_member("deputy", role="admin")
    service = _service(users=users)

    updated = await service.user_set_role("root", "member")
    assert updated.user.role == "member"
    assert updated.previous_role == "admin"


async def test_nobody_disables_their_own_membership() -> None:
    """A disable would end the session making the request, with nobody left to undo it."""
    users = FakeUsers()
    users.add_member("alice", role="admin")
    users.add_member("bob", role="admin")
    service = _service(users=users)

    with acting_as(Caller(role=Role.ADMIN, user_id="alice")), pytest.raises(PolicyError):
        await service.user_disable("alice")
    with acting_as(Caller(role=Role.ADMIN, user_id="bob")):
        assert (await service.user_disable("alice")).user.disabled


async def test_an_administrator_may_hand_administration_over_and_demote_themselves() -> None:
    """Handing over is legitimate, and the last-administrator rule already stops it leaving none."""
    users = FakeUsers()
    users.add_member("alice", role="admin")
    users.add_member("bob", role="admin")
    service = _service(users=users)

    with acting_as(Caller(role=Role.ADMIN, user_id="alice")):
        assert (await service.user_set_role("alice", "member")).user.role == "member"


async def test_disabling_a_member_ends_their_sessions_and_revokes_the_keys_they_minted() -> None:
    """In one change: there is no moment where a disabled person holds a working credential."""
    service = _service()
    backend = service.fake
    signed = await service.sign_in("google", _profile())
    backend.users_.add_member("admin", role="admin")
    with acting_as(Caller(role=Role.MEMBER, user_id=signed.user.id)):
        theirs = await service.api_key_create("alice-laptop", role="member")
    operators = await service.api_key_create("ci", role="viewer")
    backend.keys_.memberships[signed.user.id] = "member"
    assert theirs.key.user_id == signed.user.id
    assert await backend.keys_.verify(theirs.secret) is not None, "the control failed"

    with acting_as(Caller(role=Role.ADMIN, user_id="admin", address="192.0.2.7")):
        updated = await service.user_disable(signed.user.id)

    assert updated.sessions_revoked == 1
    assert updated.keys_revoked == 1
    assert not (await service.authenticate_session(signed.token)).authenticated
    assert await backend.keys_.verify(theirs.secret) is None
    assert await backend.keys_.verify(operators.secret) is not None, "a key nobody owned went too"
    audit = backend.telemetry_.audits[-1]
    assert audit["event_type"] == "user.disabled"
    assert audit["actor"] == "admin"
    assert audit["ip_address"] == "192.0.2.7"


async def test_enabling_a_member_restores_the_membership_and_not_the_credentials() -> None:
    service = _service()
    signed = await service.sign_in("google", _profile())
    service.fake.users_.add_member("admin", role="admin")
    await service.user_disable(signed.user.id)

    enabled = await service.user_enable(signed.user.id)
    assert not enabled.user.disabled
    assert not (await service.authenticate_session(signed.token)).authenticated


async def test_every_membership_change_is_audited_under_the_caller() -> None:
    users = FakeUsers()
    users.add_member("alice", role="member")
    users.add_member("root", role="admin")
    service = _service(users=users)

    with acting_as(Caller(role=Role.ADMIN, user_id="root")):
        await service.user_set_role("alice", "viewer")
        await service.user_disable("alice")
        await service.user_enable("alice")
        await service.user_sign_out("alice")

    audits = service.fake.telemetry_.audits
    events = [(entry["event_type"], entry["actor"]) for entry in audits]
    assert events == [
        ("user.role_changed", "root"),
        ("user.disabled", "root"),
        ("user.enabled", "root"),
        ("user.signed_out", "root"),
    ]


async def test_signing_a_member_out_everywhere_ends_every_session_they_hold() -> None:
    service = _service()
    first = await service.sign_in("google", _profile())
    second = await service.sign_in("google", _profile())

    ended = await service.user_sign_out(first.user.id)
    assert ended.sessions_revoked == 2
    for signed in (first, second):
        assert not (await service.authenticate_session(signed.token)).authenticated


async def test_a_change_with_nothing_in_it_is_refused() -> None:
    users = FakeUsers()
    users.add_member("alice")
    with pytest.raises(ConfigError, match="nothing to change"):
        await _service(users=users).user_update("alice")


async def test_a_role_manicule_does_not_have_is_refused_by_name() -> None:
    users = FakeUsers()
    users.add_member("alice")
    with pytest.raises(ConfigError, match="no such role 'owner'"):
        await _service(users=users).user_set_role("alice", "owner")


async def test_an_address_names_a_member_only_when_it_names_exactly_one() -> None:
    """Two accounts at two providers can report the same address; a guess would pick one."""
    users = FakeUsers()
    users.add_member("alice-google", email="alice@example.org", provider="google")
    users.add_member("alice-github", email="alice@example.org", provider="github")
    users.add_member("bob", email="bob@example.org")
    users.add_member("root", role="admin")
    service = _service(users=users)

    assert (await service.user_set_role("BOB@example.org", "viewer")).user.id == "bob"
    with pytest.raises(AmbiguousHandleError) as caught:
        await service.user_set_role("alice@example.org", "viewer")
    assert "alice-google" in str(caught.value)
    assert "alice-github" in str(caught.value)
    with pytest.raises(UnknownEntityError):
        await service.user_set_role("carol@example.org", "viewer")


async def test_a_member_of_another_workspace_cannot_be_named_here() -> None:
    """A lookup that found somebody elsewhere would confirm who is a member elsewhere."""
    users = FakeUsers(workspace="beta")
    users.add_member("alice", email="alice@example.org", workspace="alpha")
    service = _service(_settings(workspace="beta"), users=users)
    with pytest.raises(UnknownEntityError):
        await service.user_disable("alice@example.org")


# --- the installation's mode --------------------------------------------------------------------


async def test_switching_workspace_with_a_mode_writes_both_in_one_edit(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomllib  # noqa: PLC0415

    config = manicule_environment / "config.toml"
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(config))
    service = _service(Settings(security={"auth": {"mode": "api_key"}}))  # pyright: ignore[reportArgumentType]

    switched = await service.workspace_switch("team-a", create=True, mode="team")

    written = tomllib.loads(await asyncio.to_thread(config.read_text))
    assert written["workspace"] == "team-a"
    assert written["mode"] == "team"
    assert switched.mode == "team"
    assert switched.detail == ""


async def test_switching_to_team_mode_with_no_authentication_says_it_cannot_be_served(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorded, because the file is valid; said out loud, because serving it is refused."""
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(manicule_environment / "config.toml"))
    switched = await _service(Settings()).workspace_switch("team-a", create=True, mode="team")
    assert "cannot be served" in switched.detail


async def test_a_mode_manicule_does_not_have_is_refused_before_anything_is_written(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = manicule_environment / "config.toml"
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(config))
    with pytest.raises(ConfigError, match="no such mode"):
        await _service(Settings()).workspace_switch("team-a", create=True, mode="company")
    assert not await asyncio.to_thread(config.exists)


async def test_the_active_workspace_reports_the_configured_mode_and_others_their_record() -> None:
    """Configuration decides how this process behaves; the record is all there is for the rest."""
    backend = FakeBackend(settings=Settings(mode="team"))  # pyright: ignore[reportArgumentType]
    backend.maintenance_.workspace_rows = [
        ("default", "default", "personal"),
        ("other", "other", "team"),
    ]
    listed = await ApplicationService(backend).workspace_list()
    modes = {summary.id: summary.mode for summary in listed.workspaces}
    assert modes == {"default": "team", "other": "team"}

    backend.maintenance_.workspace_rows = [
        ("default", "default", "team"),
        ("other", "other", "personal"),
    ]
    backend.settings = Settings()
    listed = await ApplicationService(backend).workspace_list()
    assert {summary.id: summary.mode for summary in listed.workspaces} == {
        "default": "personal",
        "other": "personal",
    }


# --- what doctor says ---------------------------------------------------------------------------


async def test_doctor_reports_a_sign_in_that_could_not_complete() -> None:
    settings = Settings(security={"auth": {"mode": "oauth"}})  # pyright: ignore[reportArgumentType]
    diagnosis = await ApplicationService(FakeBackend(settings=settings)).doctor()
    check = next(check for check in diagnosis.checks if check.name == "sign_in")
    assert check.state == "failing"
    assert "session_secret" in check.detail
    assert "no OAuth provider applies" in check.detail


async def test_doctor_reports_a_sign_in_that_would_work() -> None:
    diagnosis = await _service().doctor()
    check = next(check for check in diagnosis.checks if check.name == "sign_in")
    assert check.state == "ok"
    assert "google" in check.detail


async def test_doctor_reports_team_mode_without_authentication_even_on_loopback() -> None:
    """Loopback does not help: there is no operator-at-this-machine for a team to be."""
    settings = Settings(mode="team")  # pyright: ignore[reportArgumentType]
    diagnosis = await ApplicationService(FakeBackend(settings=settings)).doctor()
    check = next(check for check in diagnosis.checks if check.name == "transport")
    assert check.state == "failing"
    assert "team" in check.detail
    assert check.facts["installation_mode"] == "team"
