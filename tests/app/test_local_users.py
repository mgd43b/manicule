"""Creating a person here, and minting keys for people who exist.

A person an administrator creates has a handle and either a password or keys. The service
decides who may create one, who may mint a key for somebody else, and what a handle may be; the
HTTP and command-line surfaces only carry the request.
"""

# ruff: noqa: S106 - the passwords in this file are test fixtures, not credentials

from __future__ import annotations

import pytest

from manicule.app.caller import Caller, acting_as
from manicule.app.people import honors_sessions
from manicule.app.service import ApplicationService
from manicule.config.settings import Role, Settings
from manicule.core.errors import (
    ConfigError,
    NameInUseError,
    PolicyError,
    RateLimitedError,
    SignInRefusedError,
    UnknownEntityError,
)
from tests.app.fakes import FakeBackend

KEYED = {"security": {"auth": {"mode": "api_key", "session_secret": "s" * 40}}}

ADMIN = Caller(role=Role.ADMIN, key_id="admin-key", user_id="u-admin")
MEMBER = Caller(role=Role.MEMBER, key_id="member-key", user_id="u-member")


def _service(**settings: object) -> ApplicationService:
    return ApplicationService(FakeBackend(settings=Settings(**(settings or KEYED))))  # pyright: ignore[reportArgumentType]


async def test_a_created_person_is_a_member_holding_one_key_in_their_role() -> None:
    service = _service()
    created = await service.user_create("Alice", role="member")
    assert created.user.name == "alice", "the handle is folded to lower case"
    assert created.user.provider == "local"
    assert created.user.role == "member"
    assert created.key is not None
    assert created.key.user_id == created.user.id
    assert created.key.role == "member"
    assert created.secret.startswith("mnk_")
    listed = await service.user_list()
    assert [member.id for member in listed.users] == [created.user.id]


async def test_the_first_key_authenticates_as_the_person() -> None:
    service = _service()
    created = await service.user_create("alice", role="admin")
    identity = await service.authenticate(created.secret)
    assert identity.authenticated
    assert identity.user_id == created.user.id
    assert identity.role == "admin"


@pytest.mark.parametrize("handle", ["", "  ", "has space", "a@b.org", "-leading", "x" * 65, "ü"])
async def test_a_handle_that_could_be_mistaken_for_something_else_is_refused(
    handle: str,
) -> None:
    with pytest.raises(ConfigError, match="not a usable handle"):
        await _service().user_create(handle)


async def test_a_handle_is_taken_once() -> None:
    service = _service()
    await service.user_create("alice")
    with pytest.raises(NameInUseError):
        await service.user_create("ALICE")


async def test_an_unknown_role_creates_nobody() -> None:
    service = _service()
    with pytest.raises(ConfigError):
        await service.user_create("alice", role="root")
    assert (await service.user_list()).count == 0


async def test_a_person_is_named_by_handle_afterwards() -> None:
    service = _service()
    created = await service.user_create("alice")
    await service.user_create("root", role="admin")
    changed = await service.user_set_role("alice", "viewer")
    assert changed.user.id == created.user.id
    assert changed.user.role == "viewer"


async def test_an_administrator_mints_another_key_for_a_person() -> None:
    service = _service()
    person = await service.user_create("alice")
    with acting_as(ADMIN):
        second = await service.api_key_create("laptop", role="viewer", user="alice")
    assert second.key.user_id == person.user.id
    assert second.key.role == "viewer"
    keys = [key.name for key in (await service.api_key_list()).keys]
    assert sorted(keys) == ["alice-key", "laptop"]


async def test_the_local_operator_may_mint_for_a_person_too() -> None:
    service = _service()
    person = await service.user_create("alice")
    with acting_as(Caller()):
        issued = await service.api_key_create("ci", user="alice")
    assert issued.key.user_id == person.user.id


async def test_a_member_cannot_mint_a_key_for_somebody_else() -> None:
    service = _service()
    await service.user_create("alice")
    with acting_as(MEMBER), pytest.raises(PolicyError, match="only an administrator"):
        await service.api_key_create("sneaky", user="alice")


async def test_a_key_cannot_be_minted_for_a_stranger_or_a_disabled_person() -> None:
    service = _service()
    await service.user_create("root", role="admin")
    await service.user_create("alice")
    await service.user_disable("alice")
    with acting_as(ADMIN):
        with pytest.raises(UnknownEntityError):
            await service.api_key_create("ghost", user="nobody")
        with pytest.raises(PolicyError, match="disabled"):
            await service.api_key_create("late", user="alice")


async def test_a_key_sign_in_returns_a_session_bound_to_the_key() -> None:
    service = _service()
    created = await service.user_create("alice")
    signed_in = await service.sign_in_with_key(created.secret)
    assert signed_in.user.id == created.user.id
    identity = await service.authenticate_session(signed_in.token)
    assert identity.authenticated
    assert identity.via == "session"
    assert identity.user_id == created.user.id


async def test_a_wrong_key_does_not_sign_in() -> None:
    service = _service()
    await service.user_create("alice")
    with pytest.raises(SignInRefusedError) as refused:
        await service.sign_in_with_key("mnk_nope")
    assert refused.value.reason == "key_not_usable"


async def test_an_unowned_key_is_told_how_to_get_its_own() -> None:
    service = _service()
    unowned = await service.api_key_create("operator")
    with pytest.raises(SignInRefusedError, match="add-user") as refused:
        await service.sign_in_with_key(unowned.secret)
    assert refused.value.reason == "key_has_no_owner"


async def test_key_sign_in_is_not_available_without_a_session_secret_or_under_other_modes() -> None:
    for security in (
        {"auth": {"mode": "api_key"}},
        {"auth": {"mode": "none"}},
    ):
        service = _service(security=security)
        with pytest.raises(UnknownEntityError, match="not available"):
            await service.sign_in_with_key("mnk_whatever")


@pytest.mark.parametrize(
    ("auth", "honored"),
    [
        ({"mode": "api_key", "session_secret": "s" * 40}, True),
        ({"mode": "api_key"}, False),
        ({"mode": "api_key", "session_secret": "short"}, False),
        ({"mode": "none", "session_secret": "s" * 40}, False),
    ],
)
async def test_a_session_cookie_is_a_credential_only_where_a_browser_can_sign_in(
    auth: dict[str, str], *, honored: bool
) -> None:
    """A cookie left over from before ``session_secret`` was removed is just a string."""
    service = _service(security={"auth": auth})
    assert honors_sessions(service.settings) is honored
    identity = await service.authenticate_session("any-token")
    assert identity.authenticated is False


async def test_repeated_wrong_keys_from_one_address_are_throttled() -> None:
    service = _service()
    await service.user_create("alice")
    outcomes: list[type[Exception]] = []
    with acting_as(Caller(address="198.51.100.7")):
        for attempt in range(200):
            try:
                await service.sign_in_with_key(f"mnk_guess_{attempt}")
            except (SignInRefusedError, RateLimitedError) as exc:
                outcomes.append(type(exc))
    assert outcomes[0] is SignInRefusedError, "the very first wrong key was throttled"
    assert RateLimitedError in outcomes, "two hundred wrong keys were all merely refused"


async def test_creating_and_signing_in_are_audited_without_the_secret() -> None:
    backend = FakeBackend(
        settings=Settings(
            security={  # pyright: ignore[reportArgumentType]
                "auth": {"mode": "api_key", "session_secret": "s" * 40},
                "audit": {"enabled": True},
            }
        )
    )
    service = ApplicationService(backend)
    created = await service.user_create("alice")
    await service.sign_in_with_key(created.secret)
    rows, _ = await backend.telemetry_.audit_logs(limit=50)
    events = {str(row["event_type"]) for row in rows}
    assert {"user.created", "api_key.created", "auth.login"} <= events
    assert created.secret not in repr(rows)


# --- passwords --------------------------------------------------------------------------------

PASSWORD = "a long enough passphrase"  # noqa: S105 - a test fixture
OTHER = "another long passphrase!"
WEB = Caller(role=Role.MEMBER, user_id="u-web")


async def test_a_person_created_with_a_password_has_no_key_and_signs_in_with_it() -> None:
    service = _service()
    created = await service.user_create("Alice", password=PASSWORD)
    assert created.key is None
    assert created.secret == ""
    assert created.user.has_password is True
    with acting_as(Caller(address="10.0.0.1")):
        signed = await service.sign_in_with_password(" ALICE ", PASSWORD)
    assert signed.user.id == created.user.id
    assert signed.token


async def test_an_unacceptable_password_creates_nobody() -> None:
    service = _service()
    with pytest.raises(ConfigError, match="at least 12"):
        await service.user_create("alice", password="short")
    assert (await service.user_list()).count == 0


async def test_every_failure_to_sign_in_with_a_password_is_one_refusal() -> None:
    service = _service()
    await service.user_create("alice", password=PASSWORD)
    await service.user_create("keyonly")
    messages: set[str] = set()
    for handle, password in [("alice", OTHER), ("ghost", PASSWORD), ("keyonly", PASSWORD)]:
        with acting_as(Caller(address="10.0.0.2")), pytest.raises(SignInRefusedError) as raised:
            await service.sign_in_with_password(handle, password)
        messages.add(str(raised.value))
        assert raised.value.reason == "bad_credentials"
    assert len(messages) == 1
    assert PASSWORD not in "".join(messages)


async def test_password_sign_in_is_not_available_without_a_session_secret() -> None:
    service = _service(security={"auth": {"mode": "api_key"}})
    with pytest.raises(UnknownEntityError):
        await service.sign_in_with_password("alice", PASSWORD)


async def test_wrong_passwords_are_charged_to_the_bucket_keys_are_and_then_refused_up_front() -> (
    None
):
    service = _service(
        security={
            "auth": {"mode": "api_key", "session_secret": "s" * 40},
            "rate_limit": {"failed_auth_per_minute": 3},
        }
    )
    await service.user_create("alice", password=PASSWORD)
    outcomes: list[type[Exception]] = []
    with acting_as(Caller(address="10.0.0.3")):
        for attempt in range(6):
            with pytest.raises((SignInRefusedError, RateLimitedError)) as raised:
                await service.sign_in_with_password("alice", f"wrong password {attempt}")
            outcomes.append(type(raised.value))
        assert RateLimitedError in outcomes
        with pytest.raises(RateLimitedError):
            await service.sign_in_with_password("alice", PASSWORD)
    with acting_as(Caller(address="10.0.0.4")):
        assert (await service.sign_in_with_password("alice", PASSWORD)).token, (
            "another address is not affected by this one's failures"
        )


async def test_an_administrator_sets_a_password_for_a_local_person_only() -> None:
    service = _service()
    created = await service.user_create("alice")
    result = await service.user_set_password("alice", PASSWORD)
    assert result.user.has_password is True
    with acting_as(Caller(address="10.0.0.5")):
        assert (await service.sign_in_with_password("alice", PASSWORD)).user.id == created.user.id
    with pytest.raises(ConfigError, match="at least 12"):
        await service.user_set_password("alice", "short")
    with pytest.raises(UnknownEntityError):
        await service.user_set_password("ghost", PASSWORD)


async def test_setting_a_password_ends_the_sessions_and_leaves_the_keys() -> None:
    service = _service()
    created = await service.user_create("alice", password=PASSWORD)
    with acting_as(Caller(address="10.0.0.6")):
        signed = await service.sign_in_with_password("alice", PASSWORD)
    minted = await service.api_key_create("laptop", role="member", user=created.user.id)
    assert (await service.authenticate_session(signed.token)).authenticated
    await service.user_set_password("alice", OTHER)
    assert not (await service.authenticate_session(signed.token)).authenticated
    assert (await service.authenticate(minted.secret)).authenticated


async def test_changing_your_own_password_needs_the_old_one_and_a_session_not_a_key() -> None:
    service = _service()
    created = await service.user_create("alice", password=PASSWORD)
    me = Caller(role=Role.MEMBER, user_id=created.user.id, address="10.0.0.7")
    with acting_as(me):
        with pytest.raises(SignInRefusedError):
            await service.password_change(OTHER, OTHER)
        with pytest.raises(ConfigError, match="at least 12"):
            await service.password_change(PASSWORD, "short")
        renewed = await service.password_change(PASSWORD, OTHER)
    assert renewed.token
    with acting_as(Caller(address="10.0.0.8")):
        assert (await service.sign_in_with_password("alice", OTHER)).token
    for refused in (
        Caller(role=Role.MEMBER, user_id=created.user.id, key_id="a-key"),
        Caller(role=Role.MEMBER),
        Caller(),
    ):
        with acting_as(refused), pytest.raises(PolicyError):
            await service.password_change(OTHER, PASSWORD)


async def test_a_provider_persons_password_is_not_ours_to_set() -> None:
    from manicule.app.people import Profile  # noqa: PLC0415

    backend = FakeBackend(settings=Settings(**KEYED))  # pyright: ignore[reportArgumentType]
    service = ApplicationService(backend)
    users = await backend.users()
    google = await users.admit(Profile(provider="google", subject="g", name="g"), role="member")
    with pytest.raises(PolicyError, match="google"):
        await service.user_set_password(google.id, PASSWORD)


async def test_disabling_a_person_takes_their_password_and_enabling_does_not_return_it() -> None:
    service = _service()
    await service.user_create("root", role="admin")
    created = await service.user_create("alice", password=PASSWORD)
    await service.user_disable(created.user.id)
    await service.user_enable(created.user.id)
    assert (await service.user_list()).users[-1].has_password is False
    with acting_as(Caller(address="10.0.0.9")), pytest.raises(SignInRefusedError):
        await service.sign_in_with_password("alice", PASSWORD)
    await service.user_set_password("alice", OTHER)
    with acting_as(Caller(address="10.0.0.9")):
        assert (await service.sign_in_with_password("alice", OTHER)).token
