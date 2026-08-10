"""Conversations, feedback and share links against the real schema.

Every read here carries ``deleted_at IS NULL``, including the unauthenticated one, and the
share link is resolved by hash with its expiry checked in the same statement. Both are
corrections of the same shape of bug: a public read that is *almost* scoped.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest_asyncio

from manicule.core.anchors import PageAnchor
from manicule.core.generation import FinishReason
from manicule.generation.answers import Citation, Verification
from manicule.generation.ports import Feedback, FeedbackReason, StoredMessage
from manicule.generation.sharing import new_share
from manicule.storage import models
from manicule.storage.conversations import SqliteConversationStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def citation(slot: int = 1) -> Citation:
    return Citation(
        slot=slot,
        document_id="doc-1",
        uri="https://example.invalid/doc-1",
        title="Deploy runbook",
        heading_path=("Operations", "Rollback"),
        anchor=PageAnchor(page=4),
        chunk_id="c1",
        quote="Roll back with `deploy --rollback`.",
        verification=Verification.RESOLVED,
    )


@pytest_asyncio.fixture
async def conversations(engine: AsyncEngine) -> SqliteConversationStore:
    built = SqliteConversationStore(engine)
    await built.ensure_workspace()
    return built


async def a_conversation(conversations: SqliteConversationStore) -> tuple[str, str]:
    """A conversation with one question and one answer. Returns both ids."""
    conversation_id = await conversations.create_conversation()
    await conversations.append(
        StoredMessage(conversation_id=conversation_id, role="user", content="how do we roll back?")
    )
    message_id = await conversations.append(
        StoredMessage(
            conversation_id=conversation_id,
            role="assistant",
            content="Roll back with the runbook.[[cite:1]]",
            citations=(citation(),),
            finish_reason=FinishReason.STOP,
        )
    )
    return conversation_id, message_id


async def test_history_returns_whole_turns_with_the_citations_they_were_stored_with(
    conversations: SqliteConversationStore,
) -> None:
    conversation_id, _ = await a_conversation(conversations)

    turns = await conversations.history(conversation_id)

    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[1].citations[0].chunk_id == "c1"
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


async def test_feedback_on_another_workspaces_message_is_not_accepted(
    engine: AsyncEngine,
) -> None:
    mine = SqliteConversationStore(engine)
    await mine.ensure_workspace()
    _, message_id = await a_conversation(mine)
    theirs = SqliteConversationStore(engine, workspace_id="someone-else")

    assert not await theirs.record_feedback(message_id, feedback=Feedback.POSITIVE)


async def test_a_share_link_resolves_only_while_it_is_live(
    conversations: SqliteConversationStore,
) -> None:
    conversation_id, _ = await a_conversation(conversations)
    link = new_share(conversation_id, ttl_s=3600, now=NOW)
    await conversations.create_share(
        conversation_id,
        token_hash=link.token_hash,
        expires_at=link.expires_at,
        shared_at=link.shared_at,
    )

    assert await conversations.find_shared(link.token_hash, now=NOW) == conversation_id
    assert await conversations.find_shared(link.token_hash, now=NOW + timedelta(hours=2)) is None
    assert await conversations.find_shared("some-other-hash", now=NOW) is None


async def test_revoking_clears_the_hash_rather_than_flipping_a_flag_beside_it(
    conversations: SqliteConversationStore,
) -> None:
    """A boolean turned off next to a still-valid token is a revocation that depends on every
    future reader remembering to check the boolean."""
    conversation_id, _ = await a_conversation(conversations)
    link = new_share(conversation_id, ttl_s=3600, now=NOW)
    await conversations.create_share(
        conversation_id,
        token_hash=link.token_hash,
        expires_at=link.expires_at,
        shared_at=link.shared_at,
    )

    assert await conversations.revoke_share(conversation_id)
    assert await conversations.find_shared(link.token_hash, now=NOW) is None


async def test_soft_deleting_a_conversation_revokes_its_link(
    conversations: SqliteConversationStore,
) -> None:
    """A public read that lacks ``deleted_at IS NULL`` — uniquely among every query in a file
    — is how deleting a conversation leaves its contents readable by anyone holding the URL.
    """
    conversation_id, _ = await a_conversation(conversations)
    link = new_share(conversation_id, ttl_s=3600, now=NOW)
    await conversations.create_share(
        conversation_id,
        token_hash=link.token_hash,
        expires_at=link.expires_at,
        shared_at=link.shared_at,
    )

    assert await conversations.soft_delete_conversation(conversation_id)

    assert await conversations.find_shared(link.token_hash, now=NOW) is None
    assert await conversations.history(conversation_id) == []


async def test_a_share_is_a_snapshot_so_later_turns_are_not_exposed(
    conversations: SqliteConversationStore,
) -> None:
    """Somebody shares after turn 2 and keeps using the conversation; turn 7 would otherwise
    be public the moment it is written, and nobody re-reads a link they already sent."""
    conversation_id, _ = await a_conversation(conversations)
    moment = datetime.now(UTC)
    link = new_share(conversation_id, ttl_s=3600, now=moment)
    await conversations.create_share(
        conversation_id,
        token_hash=link.token_hash,
        expires_at=link.expires_at,
        shared_at=moment,
    )
    await conversations.append(
        StoredMessage(conversation_id=conversation_id, role="user", content="and after that?")
    )

    exposed = await conversations.shared_messages(conversation_id)

    assert len(exposed) == 2
    assert all("after that" not in turn.content for turn in exposed)


async def test_re_sharing_replaces_the_previous_token(
    conversations: SqliteConversationStore,
) -> None:
    """There is never more than one live link per conversation to reason about."""
    conversation_id, _ = await a_conversation(conversations)
    first = new_share(conversation_id, ttl_s=3600, now=NOW)
    await conversations.create_share(
        conversation_id,
        token_hash=first.token_hash,
        expires_at=first.expires_at,
        shared_at=first.shared_at,
    )
    second = new_share(conversation_id, ttl_s=3600, now=NOW)
    await conversations.create_share(
        conversation_id,
        token_hash=second.token_hash,
        expires_at=second.expires_at,
        shared_at=second.shared_at,
    )

    assert await conversations.find_shared(first.token_hash, now=NOW) is None
    assert await conversations.find_shared(second.token_hash, now=NOW) == conversation_id


async def test_the_query_log_table_no_longer_carries_feedback() -> None:
    """Two homes for one fact is what this migration exists to remove."""
    assert not hasattr(models.QueryLog, "feedback")
    assert hasattr(models.Message, "feedback")
    assert hasattr(models.Message, "query_log_id")
    assert hasattr(models.Conversation, "share_token_hash")
    assert not hasattr(models.Conversation, "share_token")
