"""A session a key started, against a real, migrated database.

The properties that make the session only as good as its key are predicates of one statement
in :meth:`~manicule.app.runtime._Users.resolve_session`, plus one in Python for ``allowed_ips``.
Each is made false here on its own — the key revoked, the key expired, the address wrong, the
role lower — with the identity-provider session as the control that none of it applies to.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import update

from manicule.app.people import LOCAL, Profile
from manicule.app.runtime import Runtime
from manicule.storage import models
from manicule.storage.engine import session_factory
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from manicule.app.ports import Keys, Users


@pytest.fixture
async def runtime(manicule_environment: Path) -> AsyncIterator[Runtime]:
    opened = Runtime.open(data_dir=manicule_environment / "data")
    async with opened:
        yield opened


async def _person_with_key(
    users: Users,
    keys: Keys,
    *,
    role: str = "member",
    key_role: str = "member",
    allowed_ips: Sequence[str] = (),
) -> tuple[str, str, str]:
    """A local person, a key they own, and a session that key started.

    Returns the person's id, the key's id and the session token.
    """
    member = await users.admit(Profile(provider=LOCAL, subject="alice", name="alice"), role=role)
    issued, _ = await keys.issue(
        "alice-key", role=key_role, user_id=member.id, allowed_ips=allowed_ips
    )
    _, _, token = await users.begin_session(member.id, max_age_s=3600, key_id=issued.id)
    return member.id, issued.id, token


async def test_a_session_resolves_while_its_key_is_usable(runtime: Runtime) -> None:
    users, keys = await runtime.users(), await runtime.keys()
    user_id, _, token = await _person_with_key(users, keys)
    member = await users.resolve_session(token)
    assert member is not None
    assert member.id == user_id
    assert member.provider == LOCAL


async def test_revoking_the_key_ends_the_session_it_started(runtime: Runtime) -> None:
    """A key lost and revoked must not leave a browser signed in behind it."""
    users, keys = await runtime.users(), await runtime.keys()
    _, _, token = await _person_with_key(users, keys)
    await keys.revoke("alice-key")
    assert await users.resolve_session(token) is None


async def test_an_expired_key_ends_the_session_it_started(runtime: Runtime) -> None:
    users, keys = await runtime.users(), await runtime.keys()
    _, key_id, token = await _person_with_key(users, keys)
    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        await session.execute(
            update(models.ApiKey)
            .where(models.ApiKey.id == key_id)
            .values(expires_at=utcnow() - _ONE_SECOND)
        )
    assert await users.resolve_session(token) is None


async def test_a_session_carries_the_lesser_of_the_key_and_the_membership(
    runtime: Runtime,
) -> None:
    """An administrator's viewer key signs a browser in as a viewer, not an administrator."""
    users, keys = await runtime.users(), await runtime.keys()
    _, _, token = await _person_with_key(users, keys, role="admin", key_role="viewer")
    member = await users.resolve_session(token)
    assert member is not None
    assert member.role == "viewer"


async def test_a_demotion_reaches_a_session_a_key_started(runtime: Runtime) -> None:
    users, keys = await runtime.users(), await runtime.keys()
    user_id, _, token = await _person_with_key(users, keys, role="admin", key_role="admin")
    other = await users.admit(Profile(provider=LOCAL, subject="bob", name="bob"), role="admin")
    await users.update_member(user_id, role="viewer")
    member = await users.resolve_session(token)
    assert member is not None
    assert member.role == "viewer"
    assert other.role == "admin"


async def test_a_session_is_refused_from_an_address_its_key_is_not(runtime: Runtime) -> None:
    users, keys = await runtime.users(), await runtime.keys()
    _, _, token = await _person_with_key(users, keys, allowed_ips=["10.0.0.0/8"])
    assert await users.resolve_session(token, address="10.1.2.3") is not None
    assert await users.resolve_session(token, address="192.0.2.1") is None
    assert await users.resolve_session(token) is None, "an unknown address was trusted"


async def test_a_session_an_identity_provider_started_has_no_key_to_end_it(
    runtime: Runtime,
) -> None:
    """The control: none of the above applies to a session with no ``key_id``."""
    users = await runtime.users()
    member = await users.admit(
        Profile(provider="google", subject="g", email="g@example.org", email_verified=True),
        role="member",
    )
    _, _, token = await users.begin_session(member.id, max_age_s=3600)
    resolved = await users.resolve_session(token, address="192.0.2.1")
    assert resolved is not None
    assert resolved.role == "member"


async def test_a_local_person_is_found_by_their_handle_and_not_by_a_prefix(
    runtime: Runtime,
) -> None:
    users = await runtime.users()
    await users.admit(Profile(provider=LOCAL, subject="alice", name="alice"), role="member")
    assert [m.name for m in await users.find_members("alice")] == ["alice"]
    assert [m.name for m in await users.find_members("ALICE")] == ["alice"]
    assert await users.find_members("alic") == []


async def test_a_local_person_cannot_be_created_twice(runtime: Runtime) -> None:
    """One ``(provider, subject)`` is one person: a second admit finds the first."""
    users = await runtime.users()
    first = await users.admit(Profile(provider=LOCAL, subject="alice", name="alice"), role="member")
    second = await users.admit(Profile(provider=LOCAL, subject="alice", name="alice"), role="admin")
    assert first.id == second.id
    assert second.role == "member", "a second admit rewrote the role an administrator set"


_ONE_SECOND = timedelta(seconds=1)
