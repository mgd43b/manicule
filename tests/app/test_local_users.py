"""Creating a person here, and minting keys for people who exist.

A person an administrator creates has a handle and keys and no password. The service decides who
may create one, who may mint a key for somebody else, and what a handle may be; the HTTP and
command-line surfaces only carry the request.
"""

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
