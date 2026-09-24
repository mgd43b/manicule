"""The people store against a real, migrated database.

The fakes in ``tests/app/fakes.py`` are written to the store's contract; these assert the
contract is what the SQL actually does. Each property that makes a session a credential is a
predicate of one statement in :meth:`~manicule.app.runtime._Users.resolve_session`, and a
predicate that is never made false by a test is indistinguishable from one that is absent — so
each is made false here, on its own, by writing the row the ordinary path cannot produce.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import select, update

from manicule.app.people import Profile
from manicule.app.runtime import Runtime
from manicule.core.errors import PolicyError, UnknownEntityError
from manicule.storage import models
from manicule.storage.engine import session_factory
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from manicule.app.ports import Users
    from manicule.app.results import UserSummary


def _profile(email: str = "alice@example.org", *, subject: str = "alice") -> Profile:
    return Profile(
        provider="google", subject=subject, email=email, email_verified=True, name="Alice"
    )


@pytest.fixture
async def runtime(manicule_environment: Path) -> AsyncIterator[Runtime]:
    opened = Runtime.open(data_dir=manicule_environment / "data")
    async with opened:
        yield opened


async def _signed_in(
    users: Users, *, role: str = "member", subject: str = "alice"
) -> tuple[str, str]:
    """Admit one person and start a session. Returns their id and the session token."""
    profile = _profile(subject=subject, email=f"{subject}@example.org")
    member = await users.admit(profile, role=role)
    _, _, token = await users.begin_session(member.id, max_age_s=3600)
    return member.id, token


async def _set_session(runtime: Runtime, **values: object) -> None:
    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        await session.execute(update(models.AuthSession).values(**values))


async def test_a_session_resolves_to_its_member(runtime: Runtime) -> None:
    """The control for every refusal below."""
    users = await runtime.users()
    user_id, token = await _signed_in(users)
    member = await users.resolve_session(token)
    assert member is not None
    assert member.id == user_id
    assert member.workspace == runtime.workspace


async def test_an_unknown_token_resolves_to_nobody(runtime: Runtime) -> None:
    users = await runtime.users()
    await _signed_in(users)
    assert await users.resolve_session("not-a-session") is None
    assert await users.resolve_session("") is None


async def test_a_revoked_session_resolves_to_nobody(runtime: Runtime) -> None:
    users = await runtime.users()
    _, token = await _signed_in(users)
    await _set_session(runtime, revoked_at=utcnow())
    assert await users.resolve_session(token) is None


async def test_an_expired_session_resolves_to_nobody(runtime: Runtime) -> None:
    """Written past its expiry directly, because the ordinary path only writes a future one."""
    users = await runtime.users()
    _, token = await _signed_in(users)
    await _set_session(runtime, expires_at=utcnow() - timedelta(seconds=1))
    assert await users.resolve_session(token) is None


async def test_a_disabled_members_session_resolves_to_nobody(runtime: Runtime) -> None:
    """Asserted by writing ``disabled_at`` alone, not through a disable that also revokes.

    A disable revokes the session as well, so a test that went through it would pass on the
    revocation and never make the membership predicate false.
    """
    users = await runtime.users()
    _, token = await _signed_in(users)
    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        await session.execute(update(models.WorkspaceMember).values(disabled_at=utcnow()))
    assert await users.resolve_session(token) is None


async def test_a_session_minted_in_one_workspace_does_not_authenticate_in_another(
    manicule_environment: Path,
) -> None:
    """Two runtimes over one data directory, differing only in workspace.

    The person is installation-wide and would be found from either; the session and the
    membership are not. Without this, the workspace predicate on the session could be deleted
    and every other test here would still pass — the fixtures hold one tenant.
    """
    data_dir = manicule_environment / "data"
    async with Runtime.open(data_dir=data_dir, workspace="alpha") as alpha:
        user_id, token = await _signed_in(await alpha.users(), role="admin")
        assert await (await alpha.users()).resolve_session(token) is not None, "control failed"

    async with Runtime.open(data_dir=data_dir, workspace="beta") as beta:
        users = await beta.users()
        assert await users.resolve_session(token) is None, "alpha's session authenticated in beta"
        assert await users.list_members() == []
        assert await users.find_members(user_id) == []
        assert await users.find_members("alice@example.org") == []
        assert await users.end_sessions(user_id) == 0
        with pytest.raises(UnknownEntityError):
            await users.update_member(user_id, role="viewer")

    async with Runtime.open(data_dir=data_dir, workspace="alpha") as alpha:
        assert await (await alpha.users()).resolve_session(token) is not None, (
            "a sign-out from beta reached alpha's session"
        )


async def test_standing_names_only_the_enabled_memberships_among_those_asked_about(
    manicule_environment: Path,
) -> None:
    """The one unscoped read in the people store answers one narrow question, and only that.

    A person admitted to beta and gamma, disabled in gamma, and never admitted to delta. Asked
    from alpha about all three, the answer is beta alone: a disabled membership is no standing,
    and a workspace nobody asked about is never reported.
    """
    data_dir = manicule_environment / "data"
    profile = _profile(subject="ada", email="ada@example.org")
    user_id = ""
    for workspace in ("beta", "gamma"):
        async with Runtime.open(data_dir=data_dir, workspace=workspace) as opened:
            user_id = (await (await opened.users()).admit(profile, role="viewer")).id
    async with Runtime.open(data_dir=data_dir, workspace="gamma") as gamma:
        await (await gamma.users()).update_member(user_id, disabled=True)

    async with Runtime.open(data_dir=data_dir, workspace="alpha") as alpha:
        users = await alpha.users()
        assert await users.standing_in(user_id, ["beta", "gamma", "delta"]) == {"beta"}
        assert await users.standing_in(user_id, ["gamma"]) == frozenset()
        assert await users.standing_in(user_id, []) == frozenset()
        assert await users.standing_in("somebody-else", ["beta"]) == frozenset()


async def test_a_role_change_is_what_the_next_resolve_reports(runtime: Runtime) -> None:
    users = await runtime.users()
    await _signed_in(users, role="admin", subject="root")
    user_id, token = await _signed_in(users, role="member")

    await users.update_member(user_id, role="viewer")
    member = await users.resolve_session(token)
    assert member is not None
    assert member.role == "viewer"


async def test_a_sign_in_never_overwrites_the_role_a_membership_already_has(
    runtime: Runtime,
) -> None:
    users = await runtime.users()
    user_id, _ = await _signed_in(users, role="member")
    await users.update_member(user_id, role="viewer")
    again = await users.admit(_profile(), role="admin")
    assert again.role == "viewer"


async def test_disabling_revokes_every_session_and_every_key_the_person_minted_at_once(
    runtime: Runtime,
) -> None:
    """One transaction, and a key nobody owns is left alone."""
    users = await runtime.users()
    keys = await runtime.keys()
    await _signed_in(users, role="admin", subject="root")
    user_id, token = await _signed_in(users)
    _, _, other_token = await users.begin_session(user_id, max_age_s=3600)
    theirs, their_secret = await keys.issue("alice-laptop", role="member")
    _, operator_secret = await keys.issue("ci", role="viewer")
    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        await session.execute(
            update(models.ApiKey).where(models.ApiKey.id == theirs.id).values(user_id=user_id)
        )

    change = await users.update_member(user_id, disabled=True)

    assert change.member.disabled
    assert change.sessions_revoked == 2
    assert change.keys_revoked == 1
    assert await users.resolve_session(token) is None
    assert await users.resolve_session(other_token) is None
    assert await keys.verify(their_secret) is None
    assert await keys.verify(operator_secret) is not None


async def test_the_store_refuses_to_remove_the_last_enabled_administrator(
    runtime: Runtime,
) -> None:
    """Inside the transaction that would make the change, and the change does not happen."""
    users = await runtime.users()
    user_id, _ = await _signed_in(users, role="admin")

    with pytest.raises(PolicyError, match="last enabled administrator"):
        await users.update_member(user_id, role="member")
    with pytest.raises(PolicyError, match="last enabled administrator"):
        await users.update_member(user_id, disabled=True)
    (member,) = await users.list_members()
    assert member.role == "admin"
    assert not member.disabled


async def test_the_store_lets_an_administrator_go_while_another_remains(runtime: Runtime) -> None:
    users = await runtime.users()
    first, _ = await _signed_in(users, role="admin", subject="first")
    await _signed_in(users, role="admin", subject="second")

    change = await users.update_member(first, role="member")
    assert change.previous_role == "admin"
    assert await users.count_admins() == 1


async def test_a_listing_counts_only_live_sessions(runtime: Runtime) -> None:
    users = await runtime.users()
    user_id, token = await _signed_in(users)
    await users.begin_session(user_id, max_age_s=3600)
    await users.end_session(token)

    (member,) = await users.list_members()
    assert member.sessions == 1


async def test_an_unverified_address_is_not_recorded_as_the_persons(runtime: Runtime) -> None:
    """Admitted through ``allow_any_user`` or not, an unverified address is a claim, not a fact.

    And a later sign-in that stops reporting it verified clears the one recorded earlier, so
    the address a member list shows is one the provider vouches for today.
    """
    users = await runtime.users()
    unverified = Profile(provider="github", subject="7", email="who@example.org", name="Who")
    member = await users.admit(unverified, role="viewer")
    assert member.email == ""

    verified = Profile(provider="github", subject="7", email="who@example.org", email_verified=True)
    assert (await users.admit(verified, role="viewer")).email == "who@example.org"
    assert (await users.admit(unverified, role="viewer")).email == ""


async def test_a_session_token_is_stored_only_as_a_digest(runtime: Runtime) -> None:
    """A copy of the database is not a copy of anybody's session."""
    users = await runtime.users()
    _, token = await _signed_in(users)
    sessions = session_factory(runtime.require_engine())
    async with sessions() as session:
        stored = (await session.execute(select(models.AuthSession.token_hash))).scalar_one()
    assert token not in stored
    assert len(stored) == 64


async def test_a_writer_records_the_mode_it_opened_the_workspace_under(
    manicule_environment: Path,
) -> None:
    """The record ``workspace list`` reads for a workspace no process is serving."""
    data_dir = manicule_environment / "data"
    async with Runtime.open(data_dir=data_dir, workspace="alpha", mode="team") as alpha:
        await alpha.documents()
    async with Runtime.open(data_dir=data_dir, workspace="beta") as beta:
        await beta.documents()
        recorded = {row[0]: row[2] for row in await (await beta.maintenance()).workspaces()}
    assert recorded == {"alpha": "team", "beta": "personal"}


async def test_a_reader_does_not_rewrite_the_recorded_mode(manicule_environment: Path) -> None:
    """A ``search`` under a different configuration file has not used the workspace for anything."""
    data_dir = manicule_environment / "data"
    async with Runtime.open(data_dir=data_dir, workspace="alpha", mode="team") as alpha:
        await alpha.documents()
    async with Runtime.open(data_dir=data_dir, workspace="alpha", writer=False) as reader:
        await reader.documents()
        recorded = {row[0]: row[2] for row in await (await reader.maintenance()).workspaces()}
    assert recorded["alpha"] == "team"


async def test_a_first_sign_in_that_loses_the_race_to_create_the_person_still_succeeds(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two tabs, one account, neither of which has signed in before.

    Both find no row and the second insert loses to the unique constraint. That race cannot be
    produced on demand through one engine, which serializes the two, so it is staged: the first
    attempt lets a rival sign-in complete and then fails exactly as the losing insert does. The
    retry must find the rival's row, so both sign-ins are one person.
    """
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    from manicule.app import runtime as runtime_module  # noqa: PLC0415

    users = await runtime.users()
    store = runtime_module._Users  # pyright: ignore[reportPrivateUsage] - the store under test
    real = store._admit_once  # pyright: ignore[reportPrivateUsage] - the seam the race is at
    attempts: list[str] = []

    async def losing_once(self: Any, profile: Profile, *, role: str) -> UserSummary:
        attempts.append(profile.subject)
        if len(attempts) == 1:
            await real(self, profile, role=role)
            raise IntegrityError("INSERT INTO users", {}, Exception("UNIQUE constraint failed"))
        return await real(self, profile, role=role)

    monkeypatch.setattr(store, "_admit_once", losing_once)
    member = await users.admit(_profile(), role="member")

    assert len(attempts) == 2
    sessions = session_factory(runtime.require_engine())
    async with sessions() as session:
        people = (await session.execute(select(models.User.id))).scalars().all()
    assert people == [member.id]


async def test_two_administrators_demoted_at_once_cannot_leave_the_workspace_with_none(
    runtime: Runtime,
) -> None:
    """The last-administrator check and the change it guards are one step, however they race.

    Two demotions started together each count two administrators if the count is read before
    either writes — and then both succeed. Writes queue in the engine's one writer admission, so
    the second counts after the first has committed and is refused.
    """
    import asyncio  # noqa: PLC0415 - only this test races
    import contextlib  # noqa: PLC0415

    users = await runtime.users()
    ada, _ = await _signed_in(users, role="admin", subject="ada")
    bob, _ = await _signed_in(users, role="admin", subject="bob")

    # Hold each demotion at its count until both have counted — the interleaving that loses an
    # administrator when nothing orders the two, forced rather than left to timing. With the
    # writes queued, the second never reaches its count while the first is open, so the barrier
    # is released by a timeout instead of by both arriving.
    counted = 0
    both = asyncio.Event()
    original = type(users)._enabled_admins  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]

    async def count_then_wait(self: object, session: object) -> int:
        nonlocal counted
        result = int(await original(self, session))  # pyright: ignore[reportUnknownArgumentType]
        counted += 1
        if counted == 2:
            both.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(both.wait(), timeout=0.5)
        return result

    type(users)._enabled_admins = count_then_wait  # pyright: ignore[reportAttributeAccessIssue]
    try:
        outcomes = await asyncio.gather(
            users.update_member(ada, role="member"),
            users.update_member(bob, role="member"),
            return_exceptions=True,
        )
    finally:
        type(users)._enabled_admins = original  # pyright: ignore[reportAttributeAccessIssue]

    refused = [outcome for outcome in outcomes if isinstance(outcome, PolicyError)]
    assert len(refused) == 1, outcomes
    assert await users.count_admins() == 1
