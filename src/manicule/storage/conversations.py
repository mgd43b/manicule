"""Conversations, messages, feedback and share links.

Separate from :class:`~manicule.storage.docstore.SqliteDocStore` because the protocols are
separate: a store may serve retrieval without ever holding a conversation, and folding these
into ``DocStore`` would make that impossible to express.

Two rules run through every read here and both are corrections of the same shape of bug.
**Every read applies ``deleted_at IS NULL``**, including the unauthenticated one — a soft
delete that does not revoke a public link is a delete that does not delete. And **a share
link is resolved by hash, with expiry checked in the same statement**, so there is no window
in which "we found the row" and "the link is still valid" are two different answers.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter
from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from manicule.core.generation import FinishReason
from manicule.generation.answers import Citation
from manicule.generation.history import Turn
from manicule.generation.ports import Feedback, FeedbackReason, StoredMessage
from manicule.storage import models
from manicule.storage.engine import session_factory
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

_CITATIONS: TypeAdapter[tuple[Citation, ...]] = TypeAdapter(tuple[Citation, ...])


def _new_id(prefix: str) -> str:
    """A random id for a row nobody derives from content.

    Deliberately not :func:`~manicule.core.ids.document_id`'s content digest: two identical
    questions asked twice are two conversations, and collapsing them would make a chat
    idempotent in a way nobody asked for.
    """
    return f"{prefix}_{secrets.token_hex(12)}"


DEFAULT_WORKSPACE = "default"


class SqliteConversationStore:
    """Conversation storage, bound to one workspace.

    Tenancy is a property of the handle rather than a parameter each call site has to
    remember, which is the same reason the document store is built that way: a scope you can
    forget to pass is not a scope.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        workspace_id: str = DEFAULT_WORKSPACE,
        sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._engine = engine
        self._workspace_id = workspace_id
        self._sessions = sessions or session_factory(engine)

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    # --- conversations --------------------------------------------------------------

    async def ensure_workspace(self) -> None:
        """Create this store's workspace row if it is absent. Idempotent.

        The same call the document store makes, and for the same reason: every table here
        cascades from ``workspaces``, so a handle bound to a workspace that has never been
        written is a foreign-key failure on the first conversation rather than on the first
        document.
        """
        async with self._sessions.begin() as session:
            if await session.get(models.Workspace, self._workspace_id) is None:
                session.add(
                    models.Workspace(id=self._workspace_id, name=self._workspace_id, settings={})
                )

    async def create_conversation(
        self,
        *,
        workspace_id: str | None = None,
        user_id: str | None = None,
        title: str | None = None,
    ) -> str:
        conversation_id = _new_id("conv")
        async with self._sessions.begin() as session:
            session.add(
                models.Conversation(
                    id=conversation_id,
                    workspace_id=workspace_id or self._workspace_id,
                    user_id=user_id,
                    title=title,
                    shared=False,
                )
            )
        return conversation_id

    async def soft_delete_conversation(self, conversation_id: str) -> bool:
        """Soft-delete a conversation, which also revokes any share link.

        The revocation is not a courtesy. A public read that lacks ``deleted_at IS NULL`` —
        uniquely among every query in a file — is how deleting a conversation leaves its
        contents readable by anyone holding the URL.
        """
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(models.Conversation)
                .where(
                    models.Conversation.id == conversation_id,
                    models.Conversation.workspace_id == self._workspace_id,
                    models.Conversation.deleted_at.is_(None),
                )
                .values(deleted_at=utcnow(), shared=False, share_token_hash=None)
            )
            return _touched(result)

    # --- messages -------------------------------------------------------------------

    async def append(self, message: StoredMessage) -> str:
        """Write one turn and return its id."""
        message_id = _new_id("msg")
        async with self._sessions.begin() as session:
            session.add(
                models.Message(
                    id=message_id,
                    conversation_id=message.conversation_id,
                    role=message.role,
                    content=message.content,
                    sources=cast("object", _CITATIONS.dump_python(message.citations, mode="json")),
                    profile_used=message.profile_used,
                    confidence_score=message.confidence_score,
                    response_time_ms=message.response_time_ms,
                    finish_reason=(message.finish_reason.value if message.finish_reason else None),
                    query_log_id=message.query_log_id,
                )
            )
        return message_id

    async def history(self, conversation_id: str, *, limit: int = 20) -> Sequence[Turn]:
        """Prior turns, oldest first, each carrying the citations it was stored with.

        ``limit`` bounds the *rows read*, and the newest are the ones kept: an old turn that
        is about to be dropped by the token budget anyway is not worth reading. The budget
        itself is applied afterwards, in whole paired turns, against the generation model's
        tokenizer.
        """
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Message)
                        .join(
                            models.Conversation,
                            models.Conversation.id == models.Message.conversation_id,
                        )
                        .where(
                            models.Message.conversation_id == conversation_id,
                            models.Conversation.workspace_id == self._workspace_id,
                            models.Conversation.deleted_at.is_(None),
                            models.Message.role.in_(("user", "assistant")),
                        )
                        .order_by(models.Message.created_at.desc(), models.Message.id.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [_to_turn(row) for row in reversed(rows)]

    async def record_feedback(
        self,
        message_id: str,
        *,
        feedback: Feedback,
        reason: FeedbackReason | None = None,
        comment: str = "",
    ) -> bool:
        """Rate an answer. **Returns whether a row actually matched.**

        Two corrections in one method. Reporting success without checking the row count means
        feedback on a mistyped or foreign id is silently accepted and never seen again; and
        the value is written from an enum rather than from whatever string arrived, so the
        column cannot fill up with vocabulary nobody defined.
        """
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(models.Message)
                .where(
                    models.Message.id == message_id,
                    models.Message.conversation_id.in_(
                        select(models.Conversation.id).where(
                            models.Conversation.workspace_id == self._workspace_id,
                            models.Conversation.deleted_at.is_(None),
                        )
                    ),
                )
                .values(
                    feedback=feedback.value,
                    feedback_reason=reason.value if reason else None,
                    feedback_comment=comment or None,
                    feedback_at=utcnow(),
                )
            )
            return _touched(result)

    # --- sharing --------------------------------------------------------------------

    async def create_share(
        self,
        conversation_id: str,
        *,
        token_hash: str,
        expires_at: datetime,
        shared_at: datetime,
    ) -> bool:
        """Record a minted link. Replaces any previous one for this conversation.

        Replacing rather than accumulating is what makes re-sharing an explicit new act that
        produces a **new snapshot**: the old token stops working the moment a new one is
        minted, so there is never more than one live link to reason about per conversation.
        """
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(models.Conversation)
                .where(
                    models.Conversation.id == conversation_id,
                    models.Conversation.workspace_id == self._workspace_id,
                    models.Conversation.deleted_at.is_(None),
                )
                .values(
                    shared=True,
                    share_token_hash=token_hash,
                    share_expires_at=expires_at,
                    shared_at=shared_at,
                )
            )
            return _touched(result)

    async def revoke_share(self, conversation_id: str) -> bool:
        """Revoke a link by **clearing the hash**, not by flipping a flag beside it.

        A boolean turned off next to a still-valid token is a revocation that depends on every
        future reader remembering to check the boolean.
        """
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(models.Conversation)
                .where(
                    models.Conversation.id == conversation_id,
                    models.Conversation.workspace_id == self._workspace_id,
                )
                .values(shared=False, share_token_hash=None, share_expires_at=None)
            )
            return _touched(result)

    async def find_shared(self, token_hash: str, *, now: datetime) -> str | None:
        """The conversation a live token names, or ``None``.

        **No workspace predicate, deliberately, and that is the whole hazard of this method.**
        An unauthenticated caller has no workspace, so the token *is* the authorization
        decision — which is why it is 256 bits, hashed at rest, expiring, revocable and rate
        limited by the route in front of it. What this must never do is return anything the
        caller could use to widen the decision: it hands back a conversation id and nothing
        else, so the workspace never travels to an anonymous reader.

        ``None`` covers unknown, expired, revoked and deleted alike. Distinguishing them for
        an anonymous caller tells them which of their guesses was closest.
        """
        async with self._sessions() as session:
            return (
                await session.execute(
                    select(models.Conversation.id).where(
                        models.Conversation.share_token_hash == token_hash,
                        models.Conversation.shared.is_(True),
                        models.Conversation.deleted_at.is_(None),
                        models.Conversation.share_expires_at.is_not(None),
                        models.Conversation.share_expires_at > now,
                    )
                )
            ).scalar_one_or_none()

    async def shared_messages(self, conversation_id: str) -> Sequence[Turn]:
        """The conversation as its share link exposes it: a **snapshot**, not a live view.

        Only turns written at or before ``shared_at`` are returned. Somebody shares after turn
        2, keeps using the conversation, and turn 7 would otherwise be public the moment it is
        written — and nobody re-reads a link they already sent. Re-sharing produces a new
        snapshot, which is an explicit act.
        """
        async with self._sessions() as session:
            conversation = (
                await session.execute(
                    select(models.Conversation).where(
                        models.Conversation.id == conversation_id,
                        models.Conversation.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if conversation is None or conversation.shared_at is None:
                return []
            rows = (
                (
                    await session.execute(
                        select(models.Message)
                        .where(
                            models.Message.conversation_id == conversation_id,
                            models.Message.role.in_(("user", "assistant")),
                            models.Message.created_at <= conversation.shared_at,
                        )
                        .order_by(models.Message.created_at, models.Message.id)
                    )
                )
                .scalars()
                .all()
            )
        return [_to_turn(row) for row in rows]


def _touched(result: object) -> bool:
    """Whether a statement actually changed a row.

    SQLAlchemy types ``execute`` as returning ``Result``, which has no ``rowcount``, while an
    UPDATE really returns a ``CursorResult``. Narrowing rather than ignoring the type keeps
    the important half honest: **the row count is the assertion**, because reporting success
    without checking it is how feedback on a mistyped id is accepted and never seen again.
    """
    return isinstance(result, CursorResult) and cast("CursorResult[Any]", result).rowcount > 0


def _to_turn(row: models.Message) -> Turn:
    role = "assistant" if row.role == "assistant" else "user"
    return Turn(role=role, content=row.content, citations=_citations(row.sources))


def _citations(stored: object) -> tuple[Citation, ...]:
    """Rehydrate stored citations, tolerating rows written before this column had a shape.

    A row that cannot be validated yields **no** citations rather than a partial set: a
    citation is a claim about a location, and half of one is not a weaker claim, it is a
    different one.
    """
    if not stored:
        return ()
    try:
        return _CITATIONS.validate_python(stored)
    except ValueError:
        return ()


def message_finish_reason(row: models.Message) -> FinishReason | None:
    """How a stored answer ended, or ``None`` for a turn that never had a generation."""
    if row.finish_reason is None:
        return None
    try:
        return FinishReason(row.finish_reason)
    except ValueError:
        return None


__all__ = ["DEFAULT_WORKSPACE", "SqliteConversationStore", "message_finish_reason"]
