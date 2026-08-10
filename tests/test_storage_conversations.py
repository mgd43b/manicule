"""Conversations, feedback and share links against the real schema.

Every read here carries a workspace predicate and ``deleted_at IS NULL``, and the one read
that cannot carry a workspace — the anonymous one — resolves the share token itself instead.
Both are corrections of the same shape of bug: a public read that is *almost* scoped.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import text as sql

from manicule.core.anchors import PageAnchor
from manicule.core.generation import FinishReason
from manicule.generation.answers import Citation, Verification
from manicule.generation.ports import Feedback, FeedbackReason, StoredMessage
from manicule.generation.sharing import ShareLink, new_share
from manicule.storage import models
from manicule.storage.conversations import SqliteConversationStore, UnknownConversationError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


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


async def share(
    store: SqliteConversationStore, conversation_id: str, *, now: datetime | None = None
) -> ShareLink:
    moment = now or datetime.now(UTC)
    link = new_share(conversation_id, ttl_s=3600, now=moment)
    await store.create_share(
        conversation_id,
        token_hash=link.token_hash,
        expires_at=link.expires_at,
        # The snapshot boundary is when the share happened, which is *now* — the turns above
        # were written a moment ago. `moment` may be a fixed instant chosen for the expiry
        # arithmetic, and using it here would put the boundary before the conversation.
        shared_at=datetime.now(UTC),
    )
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

    assert await conversations.find_shared(link.token_hash, now=NOW) == conversation_id
    assert await conversations.find_shared(link.token_hash, now=NOW + timedelta(hours=2)) is None
    assert await conversations.find_shared("some-other-hash", now=NOW) is None


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

    assert len(await conversations.shared_conversation(link.token_hash, now=NOW)) == 2
    assert await conversations.shared_conversation("not-a-token", now=NOW) == []


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

    assert await conversations.shared_conversation(link.token_hash, now=NOW) == []
    assert await conversations.find_shared(link.token_hash, now=NOW) is None


async def test_an_expired_link_stops_rendering(conversations: SqliteConversationStore) -> None:
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    assert (
        await conversations.shared_conversation(link.token_hash, now=NOW + timedelta(days=1)) == []
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

    assert (
        await conversations.shared_conversation(link.token_hash, now=NOW, sharing_enabled=False)
        == []
    )


async def test_an_anonymous_viewer_never_receives_passage_text_or_an_identifier(
    conversations: SqliteConversationStore,
) -> None:
    """The projection happens in the store, on the only path that serves an unauthenticated
    reader — not in a helper a route has to remember to call."""
    conversation_id, _ = await a_conversation(conversations)
    link = await share(conversations, conversation_id, now=NOW)

    turns = await conversations.shared_conversation(link.token_hash, now=NOW)

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

    exposed = await conversations.shared_conversation(link.token_hash, now=datetime.now(UTC))

    assert len(exposed) == 2
    assert all("after that" not in turn.content for turn in exposed)


async def test_re_sharing_replaces_the_previous_token(
    conversations: SqliteConversationStore,
) -> None:
    """There is never more than one live link per conversation to reason about."""
    conversation_id, _ = await a_conversation(conversations)
    first = await share(conversations, conversation_id, now=NOW)
    second = await share(conversations, conversation_id, now=NOW)

    assert await conversations.shared_conversation(first.token_hash, now=NOW) == []
    assert len(await conversations.shared_conversation(second.token_hash, now=NOW)) == 2


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
