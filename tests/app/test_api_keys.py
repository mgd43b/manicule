"""API key ownership, the role cap, allowed IPs and per-key rate limits.

Two layers are tested separately. The **service** layer (``ApplicationService.api_key_*``)
decides *who may mint, see and revoke what* from :func:`~manicule.app.caller.current` — those
tests run against :class:`~tests.app.fakes.FakeBackend`, fast and with no database. The
**store** layer (:class:`~manicule.app.runtime._Keys`) decides whether a presented secret
*resolves* at all — allowed IPs and the membership-capped effective role — and those tests run
against a real :class:`~manicule.app.runtime.Runtime`, because the predicates live in one SQL
statement and a fake would only prove the fake agrees with itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.app.caller import Caller, acting_as
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Role
from manicule.core.errors import ConfigError, PolicyError, UnknownEntityError
from tests.app.fakes import FakeBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

ADMIN = Caller(role=Role.ADMIN, key_id="admin-key")
MEMBER_ALICE = Caller(role=Role.MEMBER, key_id="alice-key", user_id="alice")
MEMBER_BOB = Caller(role=Role.MEMBER, key_id="bob-key", user_id="bob")
VIEWER_NO_OWNER = Caller(role=Role.VIEWER, key_id="orphan-key")
"""A non-admin caller authenticated by a key that is itself unowned. Owns nothing and may mint
nothing — see :func:`~manicule.app.service._key_owner_for`."""


@pytest.fixture
def service() -> ApplicationService:
    return ApplicationService(FakeBackend())


# --- minting: ownership and the role cap -------------------------------------------------------


async def test_the_local_operator_mints_any_role_unowned(service: ApplicationService) -> None:
    with acting_as(Caller()):  # role=None is the local operator
        issued = await service.api_key_create("root", role="admin")
    assert issued.key.user_id is None


async def test_an_admin_mints_any_role_unowned(service: ApplicationService) -> None:
    with acting_as(ADMIN):
        issued = await service.api_key_create("for-a-teammate", role="admin")
    assert issued.key.role == "admin"
    assert issued.key.user_id is None


async def test_a_member_mints_a_key_at_their_own_role_owned_by_themselves(
    service: ApplicationService,
) -> None:
    with acting_as(MEMBER_ALICE):
        issued = await service.api_key_create("my-script", role="member")
    assert issued.key.user_id == "alice"


async def test_a_member_may_mint_below_their_own_role(service: ApplicationService) -> None:
    with acting_as(MEMBER_ALICE):
        issued = await service.api_key_create("read-only-script", role="viewer")
    assert issued.key.role == "viewer"
    assert issued.key.user_id == "alice"


async def test_a_member_may_not_mint_a_key_with_more_authority_than_they_hold(
    service: ApplicationService,
) -> None:
    """The guard: a caller can never hand out more than they themselves hold.

    This is the negative test — without :func:`~manicule.app.service._key_owner_for`'s role
    check, a member could mint an admin key for themselves and immediately be one.
    """
    with acting_as(MEMBER_ALICE), pytest.raises(PolicyError, match="more authority"):
        await service.api_key_create("privilege-escalation", role="admin")


async def test_a_caller_with_no_user_id_and_no_admin_authority_may_not_mint_at_all(
    service: ApplicationService,
) -> None:
    """The guard on the case with nobody to own the new key.

    A key minted here would outlive nothing — there is no membership to demote or revoke it
    with — so minting is refused rather than producing an ownerless key from a non-admin
    authority, which is what the simplest broken version of this rule would do.
    """
    with acting_as(VIEWER_NO_OWNER), pytest.raises(PolicyError, match="signed in"):
        await service.api_key_create("orphan-attempt", role="viewer")


async def test_an_empty_name_is_refused(service: ApplicationService) -> None:
    with acting_as(ADMIN), pytest.raises(ConfigError, match="name"):
        await service.api_key_create("   ", role="member")


async def test_an_unknown_role_is_refused(service: ApplicationService) -> None:
    with acting_as(ADMIN), pytest.raises(ConfigError, match="no such role"):
        await service.api_key_create("x", role="superuser")


# --- allowed_ips and rate_limit validation ------------------------------------------------------


async def test_a_malformed_cidr_is_refused_naming_the_bad_entry(
    service: ApplicationService,
) -> None:
    with acting_as(ADMIN), pytest.raises(ConfigError, match="not-a-cidr"):
        await service.api_key_create(
            "x", role="member", allowed_ips=["203.0.113.0/24", "not-a-cidr"]
        )


async def test_a_bare_host_address_is_accepted_as_a_single_host_range(
    service: ApplicationService,
) -> None:
    """``ip_network(..., strict=False)``, the same rule ``trusted_proxies`` uses."""
    with acting_as(ADMIN):
        issued = await service.api_key_create("x", role="member", allowed_ips=["203.0.113.5"])
    assert issued.key.allowed_ips == ("203.0.113.5",)


async def test_a_rate_limit_of_zero_or_less_is_refused(service: ApplicationService) -> None:
    with acting_as(ADMIN), pytest.raises(ConfigError, match="at least 1"):
        await service.api_key_create("x", role="member", rate_limit=0)


async def test_a_positive_rate_limit_is_accepted(service: ApplicationService) -> None:
    with acting_as(ADMIN):
        issued = await service.api_key_create("x", role="member", rate_limit=42)
    assert issued.key.rate_limit == 42


# --- listing: scoped to the caller unless they are an admin -------------------------------------


async def test_an_admin_lists_every_key_in_the_workspace(service: ApplicationService) -> None:
    with acting_as(ADMIN):
        await service.api_key_create("alice-key", role="member")
    with acting_as(MEMBER_ALICE):
        await service.api_key_create("alices-own-key", role="viewer")
    with acting_as(ADMIN):
        listed = await service.api_key_list()
    assert listed.count == 2


async def test_a_member_lists_only_keys_they_own(service: ApplicationService) -> None:
    with acting_as(MEMBER_ALICE):
        await service.api_key_create("alices-key", role="viewer")
    with acting_as(MEMBER_BOB):
        await service.api_key_create("bobs-key", role="viewer")
        listed = await service.api_key_list()
    assert listed.count == 1
    assert listed.keys[0].name == "bobs-key"


async def test_a_caller_with_no_user_id_sees_no_keys(service: ApplicationService) -> None:
    """The guard: without this branch, a non-admin caller with ``user_id is None`` would pass
    ``owner=None`` to the store, which means "no restriction" — every key in the workspace."""
    with acting_as(MEMBER_ALICE):
        await service.api_key_create("alices-key", role="viewer")
    with acting_as(VIEWER_NO_OWNER):
        listed = await service.api_key_list()
    assert listed.count == 0


# --- revoking: scoped the same way as listing ----------------------------------------------------


async def test_an_admin_revokes_any_key(service: ApplicationService) -> None:
    with acting_as(MEMBER_ALICE):
        issued = await service.api_key_create("alices-key", role="viewer")
    with acting_as(ADMIN):
        revoked = await service.api_key_revoke(issued.key.id)
    assert revoked.revoked is True


async def test_a_member_may_not_revoke_a_key_they_do_not_own(service: ApplicationService) -> None:
    """The negative test: revoking somebody else's key by id must fail exactly as an unknown
    id would, so a non-admin cannot use this to discover what keys other people hold."""
    with acting_as(MEMBER_ALICE):
        issued = await service.api_key_create("alices-key", role="viewer")
    with acting_as(MEMBER_BOB), pytest.raises(UnknownEntityError):
        await service.api_key_revoke(issued.key.id)


async def test_a_member_revokes_their_own_key(service: ApplicationService) -> None:
    with acting_as(MEMBER_ALICE):
        issued = await service.api_key_create("alices-key", role="viewer")
        revoked = await service.api_key_revoke(issued.key.id)
    assert revoked.revoked is True


async def test_a_caller_with_no_user_id_may_not_revoke_anything(
    service: ApplicationService,
) -> None:
    with acting_as(ADMIN):
        issued = await service.api_key_create("someones-key", role="viewer")
    with acting_as(VIEWER_NO_OWNER), pytest.raises(UnknownEntityError):
        await service.api_key_revoke(issued.key.id)


# --- the real store: allowed_ips and the membership-capped effective role -----------------------


@pytest.fixture
async def runtime(manicule_environment: Path) -> AsyncIterator[Runtime]:
    opened = Runtime.open(data_dir=manicule_environment / "data")
    async with opened:
        yield opened


async def _add_user(runtime: Runtime, *, user_id: str) -> None:
    """A person, with no membership anywhere yet — the row ``api_keys.user_id`` requires."""
    from manicule.storage import models  # noqa: PLC0415
    from manicule.storage.engine import session_factory  # noqa: PLC0415

    await runtime.documents()  # opens the engine; `require_engine` raises before this runs
    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        session.add(
            models.User(
                id=user_id, provider="google", subject=user_id, email=f"{user_id}@example.org"
            )
        )


async def _add_member(runtime: Runtime, *, user_id: str, role: str, disabled: bool = False) -> None:
    """A person and their membership in this workspace, at ``role`` and optionally disabled."""
    from datetime import datetime  # noqa: PLC0415

    from manicule.storage import models  # noqa: PLC0415
    from manicule.storage.engine import session_factory  # noqa: PLC0415
    from manicule.storage.types import utcnow  # noqa: PLC0415

    await _add_user(runtime, user_id=user_id)
    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        disabled_at: datetime | None = utcnow() if disabled else None
        session.add(
            models.WorkspaceMember(
                workspace_id=runtime.workspace, user_id=user_id, role=role, disabled_at=disabled_at
            )
        )


async def test_a_key_with_allowed_ips_is_refused_from_outside_them(runtime: Runtime) -> None:
    store = await runtime.keys()
    _summary, secret = await store.issue("scoped", role="member", allowed_ips=["203.0.113.0/24"])
    assert await store.verify(secret, address="203.0.113.5") is not None
    assert await store.verify(secret, address="198.51.100.5") is None, (
        "a key scoped to one range authenticated from outside it"
    )


async def test_a_key_with_allowed_ips_is_refused_from_an_unknown_address(runtime: Runtime) -> None:
    store = await runtime.keys()
    _summary, secret = await store.issue("scoped", role="member", allowed_ips=["203.0.113.0/24"])
    assert await store.verify(secret, address="") is None


async def test_a_key_with_no_allowed_ips_verifies_from_anywhere(runtime: Runtime) -> None:
    store = await runtime.keys()
    _summary, secret = await store.issue("unscoped", role="member")
    assert await store.verify(secret, address="203.0.113.5") is not None
    assert await store.verify(secret, address="") is not None


async def test_an_owned_keys_effective_role_is_capped_by_current_membership(
    runtime: Runtime,
) -> None:
    """A demoted owner's keys are demoted with them, without a second write to the key row."""
    await _add_member(runtime, user_id="alice", role="viewer")
    store = await runtime.keys()
    _summary, secret = await store.issue("alices-key", role="admin", user_id="alice")
    verified = await store.verify(secret)
    assert verified is not None
    assert verified.role == "viewer", "the key's own role must not outrank its owner's membership"


async def test_an_owned_key_is_unusable_once_its_owner_is_disabled(runtime: Runtime) -> None:
    await _add_member(runtime, user_id="alice", role="member", disabled=True)
    store = await runtime.keys()
    _summary, secret = await store.issue("alices-key", role="member", user_id="alice")
    assert await store.verify(secret) is None


async def test_an_owned_key_is_unusable_if_the_owner_has_no_membership_at_all(
    runtime: Runtime,
) -> None:
    """A real person, referenced by the key's ``user_id`` — required by its foreign key — but
    never actually admitted to *this* workspace: a state a broken caller could try to construct
    directly against the store, bypassing the service's own ownership rule."""
    await _add_user(runtime, user_id="nobody")
    store = await runtime.keys()
    _summary, secret = await store.issue("orphaned", role="member", user_id="nobody")
    assert await store.verify(secret) is None


async def test_an_unowned_key_is_unaffected_by_membership_entirely(runtime: Runtime) -> None:
    await _add_member(runtime, user_id="alice", role="viewer")
    store = await runtime.keys()
    _summary, secret = await store.issue("installation-key", role="admin")
    verified = await store.verify(secret)
    assert verified is not None
    assert verified.role == "admin"
