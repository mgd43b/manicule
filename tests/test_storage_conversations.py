"""Conversations, feedback and share links against the real schema.

Every read here carries a workspace predicate and ``deleted_at IS NULL``, and the one read
that cannot carry a workspace — the anonymous one — resolves the share token itself instead.
Both are corrections of the same shape of bug: a public read that is *almost* scoped.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import text as sql
from sqlalchemy import update

from manicule.core.anchors import PageAnchor
from manicule.core.generation import FinishReason
from manicule.generation.answers import Citation, Verification
from manicule.generation.ports import Feedback, FeedbackReason, SharedTurn, StoredMessage
from manicule.generation.sharing import ShareLink, hash_token, new_share
from manicule.storage import models
from manicule.storage.conversations import SqliteConversationStore, UnknownConversationError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
CEILING_S = 30 * 24 * 3600
"""``security.sharing.link_ttl_s``, as a route would pass it."""


def citation(slot: int = 1) -> Citation:
    return Citation(
        slot=slot,
        document_id="doc-1",
        uri="https://intranet.invalid/runbook",
        title="Deploy runbook",
        heading_path=("Operations", "Rollback"),
        anchor=PageAnchor(page=4),
        chunk_id="chunk-secret-1",
        quote="Roll back with `deploy --rollback`.",
        verification=Verification.RESOLVED,
    )


@pytest_asyncio.fixture
async def conversations(engine: AsyncEngine) -> SqliteConversationStore:
    built = SqliteConversationStore(engine)
    await built.ensure_workspace()
    return built


async def a_conversation(store: SqliteConversationStore) -> tuple[str, str]:
    """A conversation with one question and one answer. Returns both ids."""
    conversation_id = await store.create_conversation()
    await store.append(
        StoredMessage(conversation_id=conversation_id, role="user", content="how do we roll back?")
    )
    message_id = await store.append(
        StoredMessage(
            conversation_id=conversation_id,
            role="assistant",
            content="Roll back with the runbook.[[cite:1]]",
            citations=(citation(),),
            finish_reason=FinishReason.STOP,
        )
    )
    return conversation_id, message_id


async def read(
    store: SqliteConversationStore,
    token_hash: str,
    *,
    now: datetime | None = None,
    enabled: bool = True,
) -> Sequence[SharedTurn]:
    """The anonymous read, as a route would make it."""
    return await store.shared_conversation(token_hash, now=now or NOW, sharing_enabled=enabled)


async def share(
    store: SqliteConversationStore, conversation_id: str, *, now: datetime | None = None
) -> ShareLink:
    # Minted at real "now" so the snapshot boundary falls *after* the turns above. `NOW` is a
    # fixed instant used for the expiry arithmetic in `read`, and using it here would put the
    # boundary before the conversation — which the store would then correctly serve as empty.
    del now
    link = new_share(conversation_id, ttl_s=3600, maximum_ttl_s=CEILING_S)
    await store.create_share(link, maximum_ttl_s=CEILING_S)
    return link


# --- history and feedback ------------------------------------------------------------------


async def test_history_returns_whole_turns_with_the_citations_they_were_stored_with(
    conversations: SqliteConversationStore,
) -> None:
    conversation_id, _ = await a_conversation(conversations)

    turns = await conversations.history(conversation_id)

    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[1].citations[0].chunk_id == "chunk-secret-1"
    assert turns[1].citation_for(1) is not None


async def test_feedback_attaches_to_a_message_and_reports_whether_a_row_matched(
    conversations: SqliteConversationStore,
) -> None:
    """A user rates an *answer*, and an answer is a message. Reporting success without
    checking the row count means feedback on a mistyped id is silently accepted."""
    _, message_id = await a_conversation(conversations)

    assert await conversations.record_feedback(
        message_id,
        feedback=Feedback.NEGATIVE,
        reason=FeedbackReason.CITATION_WRONG,
        comment="the runbook says the opposite",
    )
    assert not await conversations.record_feedback("msg-does-not-exist", feedback=Feedback.POSITIVE)


# --- tenancy -------------------------------------------------------------------------------


async def test_feedback_on_another_workspaces_message_is_not_accepted(
    engine: AsyncEngine,
) -> None:
    mine = SqliteConversationStore(engine)
    await mine.ensure_workspace()
    _, message_id = await a_conversation(mine)
    theirs = SqliteConversationStore(engine, workspace_id="someone-else")

    assert not await theirs.record_feedback(message_id, feedback=Feedback.POSITIVE)


async def test_a_turn_cannot_be_appended_to_another_workspaces_conversation(
    engine: AsyncEngine,
) -> None:
    """The foreign key says the conversation exists; it does not say who owns it.

    Paired with a share link — which renders a conversation at an unauthenticated URL — an
    unscoped write is content injection into somebody else's public page.
    """
    mine = SqliteConversationStore(engine)
    await mine.ensure_workspace()
    conversation_id, _ = await a_conversation(mine)
    theirs = SqliteConversationStore(engine, workspace_id="someone-else")
    await theirs.ensure_workspace()

    with pytest.raises(UnknownConversationError, match="someone-else"):
        await theirs.append(
            StoredMessage(conversation_id=conversation_id, role="user", content="injected")
        )

    assert [turn.content for turn in await mine.history(conversation_id)] == [
        "how do we roll back?",
        "Roll back with the runbook.[[cite:1]]",
    ]


async def test_a_turn_cannot_be_appended_to_a_deleted_conversation(
    conversations: SqliteConversationStore,
) -> None:
    conversation_id, _ = await a_conversation(conversations)
    await conversations.soft_delete_conversation(conversation_id)

    with pytest.raises(UnknownConversationError):
        await conversations.append(
            StoredMessage(conversation_id=conversation_id, role="user", content="after the delete")
        )


async def test_a_conversation_is_created_in_the_handles_own_workspace(
    engine: AsyncEngine,
) -> None:
    """There is no workspace parameter to get wrong: a scope you can pass is one you can pass
    incorrectly."""
    theirs = SqliteConversationStore(engine, workspace_id="someone-else")
    await theirs.ensure_workspace()
    conversation_id = await theirs.create_conversation()

    mine = SqliteConversationStore(engine)
    await mine.ensure_workspace()

    assert await mine.history(conversation_id) == []
    assert await theirs.history(conversation_id) == []  # empty, but the row is theirs
    assert not await mine.soft_delete_conversation(conversation_id)
    assert await theirs.soft_delete_conversation(conversation_id)


# --- sharing -------------------------------------------------------------------------------


async def test_a_share_link_resolves_only_while_it_is_live(
    conversations: SqliteConversationStore,
) -> None:
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    assert await read(conversations, link.token_hash)
    assert not await read(
        conversations, link.token_hash, now=datetime.now(UTC) + timedelta(hours=2)
    )
    assert not await read(conversations, "some-other-hash")


async def test_the_anonymous_read_resolves_the_token_rather_than_a_conversation_id(
    conversations: SqliteConversationStore,
) -> None:
    """Holding an id must not be enough.

    An earlier shape took a conversation id and checked only ``deleted_at IS NULL``, so a
    revoked link, an expired link and another tenant's conversation all still rendered — and
    ``shared_at`` is deliberately left set on revocation, so "once shared" meant "readable
    forever".
    """
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    assert len(await read(conversations, link.token_hash)) == 2
    assert await read(conversations, "not-a-token") == []
    assert await read(conversations, "") == [], "an empty token must match nothing, not NULL"

    # There is deliberately no method that turns a token into a conversation id. One would
    # rebuild the two-step this replaced: a store handle bound to the owning workspace turns
    # an id into full citations — quote, uri, chunk id — through `history`.
    assert not hasattr(conversations, "find_shared")


async def revoked(store: SqliteConversationStore, conversation_id: str) -> bool:
    return await store.revoke_share(conversation_id)


async def deleted(store: SqliteConversationStore, conversation_id: str) -> bool:
    return await store.soft_delete_conversation(conversation_id)


@pytest.mark.parametrize("withdraw", [revoked, deleted], ids=["revoked", "deleted"])
async def test_a_revoked_or_deleted_conversation_stops_rendering(
    conversations: SqliteConversationStore,
    withdraw: Callable[[SqliteConversationStore, str], Awaitable[bool]],
) -> None:
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    assert await withdraw(conversations, conversation_id)

    assert await read(conversations, link.token_hash) == []


async def test_an_expired_link_stops_rendering(conversations: SqliteConversationStore) -> None:
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    assert (
        await read(conversations, link.token_hash, now=datetime.now(UTC) + timedelta(days=1)) == []
    )


async def test_switching_sharing_off_stops_existing_links_rendering(
    conversations: SqliteConversationStore,
) -> None:
    """An operator who turns the switch off is telling the system to stop disclosing.

    A mint-time-only check would leave every link that already exists live, which makes the
    setting a statement about the future only — and the operator has just decided the past
    was the problem.
    """
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    assert await read(conversations, link.token_hash, enabled=False) == []


async def test_an_anonymous_viewer_never_receives_passage_text_or_an_identifier(
    conversations: SqliteConversationStore,
) -> None:
    """The projection happens in the store, on the only path that serves an unauthenticated
    reader — not in a helper a route has to remember to call."""
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    turns = await read(conversations, link.token_hash)

    labels = [label for turn in turns for label in turn.citations]
    assert [label.title for label in labels] == ["Deploy runbook"]
    assert [label.location for label in labels] == ["page 4"]
    rendered = str([turn.model_dump() for turn in turns])
    assert "deploy --rollback" not in rendered
    assert "chunk-secret-1" not in rendered
    assert "intranet.invalid" not in rendered


async def test_a_share_is_a_snapshot_so_later_turns_are_not_exposed(
    conversations: SqliteConversationStore,
) -> None:
    """Somebody shares after turn 2 and keeps using the conversation; turn 7 would otherwise
    be public the moment it is written, and nobody re-reads a link they already sent."""
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id)
    await conversations.append(
        StoredMessage(conversation_id=conversation_id, role="user", content="and after that?")
    )

    exposed = await read(conversations, link.token_hash, now=datetime.now(UTC))

    assert len(exposed) == 2
    assert all("after that" not in turn.content for turn in exposed)


async def test_re_sharing_replaces_the_previous_token(
    conversations: SqliteConversationStore,
) -> None:
    """There is never more than one live link per conversation to reason about."""
    conversation_id, _ = await a_conversation(conversations)
    first = await share(conversations, conversation_id, now=NOW)
    second = await share(conversations, conversation_id, now=NOW)

    assert await read(conversations, first.token_hash) == []
    assert len(await read(conversations, second.token_hash)) == 2


# --- schema ---------------------------------------------------------------------------------


async def test_the_migrated_database_has_moved_feedback_onto_messages(
    engine: AsyncEngine,
) -> None:
    """Asserted against the migrated database rather than the model classes.

    ``hasattr`` on a model passes whether or not the migration that gives it a column exists,
    which makes it a test of the ORM's declaration and not of the schema anyone runs.
    """
    async with engine.connect() as connection:
        columns = {
            table: {
                row[1] for row in (await connection.execute(sql(f"PRAGMA table_info({table})")))
            }
            for table in ("messages", "conversations", "query_logs")
        }

    assert {"feedback", "feedback_reason", "query_log_id", "finish_reason"} <= columns["messages"]
    assert "feedback" not in columns["query_logs"], "two homes for one fact"
    assert "share_token_hash" in columns["conversations"]
    assert "share_token" not in columns["conversations"], "plaintext tokens are not stored"
    assert {"share_expires_at", "shared_at"} <= columns["conversations"]
    assert hasattr(models.Message, "feedback")


# --- the consolidated fourth pass -----------------------------------------------------------


def test_a_share_link_cannot_outlive_the_policy_ceiling() -> None:
    """The ceiling was an optional keyword no production caller passed, so the clamp existed
    and never ran. A link expiring in **2126** minted cleanly."""
    link = new_share("c", ttl_s=365 * 100 * 86400, maximum_ttl_s=30 * 24 * 3600)

    assert (link.expires_at - link.shared_at).days == 30

    with pytest.raises(TypeError):
        new_share("c", ttl_s=3600)  # pyright: ignore[reportCallIssue] - the ceiling is required


async def test_the_store_refuses_a_link_built_past_the_ceiling(
    conversations: SqliteConversationStore,
) -> None:
    """``ShareLink`` is an ordinary public value object, so the clamp in ``new_share`` is not
    an enforcement point. The store is."""
    conversation_id = await conversations.create_conversation()
    forged = ShareLink(
        conversation_id=conversation_id,
        token="t",  # noqa: S106 - a fixture token, not a credential
        token_hash=hash_token("t"),
        shared_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(days=365 * 100),
    )

    with pytest.raises(ValueError, match="ceiling"):
        await conversations.create_share(forged, maximum_ttl_s=30 * 24 * 3600)


async def test_a_link_minted_for_one_conversation_cannot_be_installed_on_another(
    conversations: SqliteConversationStore,
) -> None:
    """``create_share`` took an id *and* a link and used only the id, so a link minted for A
    installed on B and served B's transcript to anyone holding A's token. There is now one id
    and it comes from the link."""
    signature = inspect.signature(conversations.create_share)

    assert "conversation_id" not in signature.parameters, (
        "two ids that must agree, with nothing making them"
    )


# --- F: the predicates the suite could not see ----------------------------------------------
#
# Six of the nine on the anonymous read were inert: removed one at a time, the suite stayed
# green. Three were genuinely dead; two — including the module's headline `deleted_at IS NULL`
# claim — passed only because `soft_delete_conversation` *also* clears the hash, so the test
# that appeared to prove the predicate proved the side effect instead. A dead predicate is not
# harmless: it is the one a later refactor deletes as redundant.
#
# Each test below drives the database into the state the predicate alone can refuse, which
# means writing that state directly. That is the point — no public method produces it, which
# is exactly why the suite could not reach it.


async def _force(store: SqliteConversationStore, conversation_id: str, **values: object) -> None:
    """Write a conversation row into a state no public method produces."""
    async with store.sessions.begin() as session:
        await session.execute(
            update(models.Conversation)
            .where(models.Conversation.id == conversation_id)
            .values(**values)
        )


async def test_a_soft_deleted_conversation_is_refused_even_with_its_hash_intact(
    conversations: SqliteConversationStore,
) -> None:
    """The module's headline claim, finally tested.

    ``soft_delete_conversation`` clears the hash *and* sets ``deleted_at``, so the existing
    test passed on either predicate alone. Setting only ``deleted_at`` is what isolates it —
    and it is the state a restore, a repair, or a second writer produces.
    """
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id)
    await _force(conversations, conversation_id, deleted_at=datetime.now(UTC))

    assert await read(conversations, link.token_hash) == []


async def test_a_conversation_marked_unshared_is_refused_even_with_its_hash_intact(
    conversations: SqliteConversationStore,
) -> None:
    """Revocation clears both, so ``shared IS TRUE`` never had to hold on its own."""
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id)
    await _force(conversations, conversation_id, shared=False)

    assert await read(conversations, link.token_hash) == []


async def test_a_link_with_no_expiry_is_refused_rather_than_treated_as_eternal(
    conversations: SqliteConversationStore,
) -> None:
    """Fails closed. A row with a hash and no expiry predates the feature or was written by
    something that skipped it, and "no expiry" reading as "never expires" is how the
    permanently-public link came about in the first place."""
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id)
    await _force(conversations, conversation_id, share_expires_at=None)

    assert await read(conversations, link.token_hash) == []


async def test_a_share_with_no_boundary_exposes_nothing(
    conversations: SqliteConversationStore,
) -> None:
    """``shared_at IS NOT NULL`` guards the snapshot. Without a boundary there is no answer to
    "which turns were shared", and the safe answer to that is none."""
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id)
    await _force(conversations, conversation_id, shared_at=None)

    assert await read(conversations, link.token_hash) == []


async def test_a_system_turn_is_never_served_anonymously(
    conversations: SqliteConversationStore,
) -> None:
    """``role IN ('user', 'assistant')`` is load-bearing **today**.

    Without it a ``system`` row is served — and served *relabeled* ``role='user'``, because
    the projection coerces the role before ``SharedTurn``'s pattern can reject it. A system
    prompt attributed to the person who asked the question is a worse disclosure than the row
    itself.
    """
    conversation_id, _ = await a_conversation(conversations)
    async with conversations.sessions.begin() as session:
        session.add(
            models.Message(
                id="msg-system",
                conversation_id=conversation_id,
                role="system",
                content="INTERNAL: never disclose pricing",
            )
        )
    link = await share(conversations, conversation_id)

    turns = await read(conversations, link.token_hash)

    assert all("INTERNAL" not in turn.content for turn in turns)
    assert {turn.role for turn in turns} <= {"user", "assistant"}


async def test_an_empty_token_matches_nothing_rather_than_a_null_hash(
    conversations: SqliteConversationStore,
) -> None:
    """``col == ""`` is a value comparison, but an *unset* token is ``NULL`` — and a
    conversation that was never shared has exactly that. The early return is what stops an
    empty token from being a key to every unshared conversation, if the comparison ever
    changed shape."""
    conversation_id, _ = await a_conversation(conversations)
    await _force(conversations, conversation_id, shared=True, share_token_hash=None)

    assert await read(conversations, "") == []
