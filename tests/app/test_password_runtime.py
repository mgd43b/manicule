"""Passwords against a real, migrated database.

What the fake cannot say: that the hash is read only for a local, enabled member of this
workspace, that setting one ends the person's sessions in the same transaction, and that it
reaches no result a surface serializes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.app.people import LOCAL, Profile
from manicule.app.runtime import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

HASH = "scrypt$16$8$1$AAAA$AAAA"


@pytest.fixture
async def runtime(manicule_environment: Path) -> AsyncIterator[Runtime]:
    opened = Runtime.open(data_dir=manicule_environment / "data")
    async with opened:
        yield opened


async def test_a_hash_is_stored_and_read_back_for_a_local_member(runtime: Runtime) -> None:
    users = await runtime.users()
    member = await users.admit(
        Profile(provider=LOCAL, subject="alice", name="alice"), role="member"
    )
    assert await users.password_hash_for(member.id) is None
    assert await users.set_password(member.id, HASH)
    assert await users.password_hash_for(member.id) == HASH


async def test_setting_a_password_ends_the_persons_sessions_in_the_same_step(
    runtime: Runtime,
) -> None:
    users = await runtime.users()
    member = await users.admit(
        Profile(provider=LOCAL, subject="alice", name="alice"), role="member"
    )
    _, _, token = await users.begin_session(member.id, max_age_s=3600)
    assert await users.resolve_session(token) is not None
    await users.set_password(member.id, HASH)
    assert await users.resolve_session(token) is None


async def test_a_provider_person_has_no_password_to_read_or_set(runtime: Runtime) -> None:
    users = await runtime.users()
    google = await users.admit(
        Profile(provider="google", subject="g1", email="g@example.org", name="g"), role="member"
    )
    assert not await users.set_password(google.id, HASH)
    assert await users.password_hash_for(google.id) is None


async def test_a_disabled_member_has_no_readable_hash(runtime: Runtime) -> None:
    users = await runtime.users()
    admin = await users.admit(Profile(provider=LOCAL, subject="root", name="root"), role="admin")
    member = await users.admit(
        Profile(provider=LOCAL, subject="alice", name="alice"), role="member"
    )
    await users.set_password(member.id, HASH)
    await users.update_member(member.id, disabled=True)
    assert await users.password_hash_for(member.id) is None
    assert admin.id != member.id


async def test_the_hash_is_in_no_result_a_surface_serializes(runtime: Runtime) -> None:
    users = await runtime.users()
    member = await users.admit(
        Profile(provider=LOCAL, subject="alice", name="alice"), role="member"
    )
    await users.set_password(member.id, HASH)
    listed = await users.list_members()
    found = await users.find_members("alice")
    for summary in (*listed, *found):
        assert summary.has_password is True
        assert "scrypt" not in summary.model_dump_json()


async def test_an_unknown_person_has_nothing_to_set(runtime: Runtime) -> None:
    users = await runtime.users()
    assert not await users.set_password("nobody", HASH)
    assert await users.password_hash_for("nobody") is None


async def test_disabling_a_member_clears_their_hash_for_good(runtime: Runtime) -> None:
    users = await runtime.users()
    await users.admit(Profile(provider=LOCAL, subject="root", name="root"), role="admin")
    member = await users.admit(
        Profile(provider=LOCAL, subject="alice", name="alice"), role="member"
    )
    await users.set_password(member.id, HASH)
    await users.update_member(member.id, disabled=True)
    await users.update_member(member.id, disabled=False)
    assert await users.password_hash_for(member.id) is None
    assert (await users.find_members("alice"))[0].has_password is False
