"""What generation needs of conversation storage, stated as protocols rather than an import.

``manicule.generation`` pulls in core, config and the plugin machinery, and nothing else. The
SQLite store satisfies these structurally, and ``tests/test_import_boundary.py`` fails the
build if that stops being true — which is also what makes the negative tests affordable: a
store that breaks its part of the bargain is a twenty-line fake rather than a migrated
database.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.generation import FinishReason
from manicule.generation.answers import AnswerEnvelope, Citation
from manicule.generation.history import Turn
from manicule.generation.sharing import CitationLabel


class Feedback(StrEnum):
    """A rating on one answer."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class FeedbackReason(StrEnum):
    """The closed vocabulary a negative rating may carry.

    Closed rather than free text, so the reports are countable. Free text is still accepted
    alongside :attr:`OTHER`, because a vocabulary that cannot express the actual problem
    teaches people to pick the nearest wrong option.
    """

    WRONG = "wrong"
    INCOMPLETE = "incomplete"
    CITATION_WRONG = "citation-wrong"
    """**Why this vocabulary exists.**

    Verification cannot catch misattribution: a citation that resolves perfectly and supports
    nothing in the sentence it is attached to. No check in this system will ever fire on that,
    because firing on it means deciding entailment — which is the deferred hallucination
    guard, held behind a precision-and-recall measurement nobody has the labels for.

    A human reading the answer *can* see it. So this is the only detector this project has for
    its one uncaught citation failure, and the reports it produces are the labelled set that
    would let the guard be measured for recall as well as precision.
    """

    TOO_SLOW = "too-slow"
    OTHER = "other"


class StoredMessage(BaseModel):
    """One turn as it is written down.

    An assistant turn is persisted **whatever happened to it** — complete, truncated,
    content-filtered or failed — because a partial answer that exists nowhere on the server
    cannot be shared, cannot be joined to a retrieval run, and above all cannot be given
    feedback, which is exactly the answer that most needs it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_id: str = Field(min_length=1)
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str
    citations: tuple[Citation, ...] = ()
    envelope: AnswerEnvelope | None = None
    finish_reason: FinishReason | None = None
    profile_used: str | None = None
    confidence_score: float | None = None
    response_time_ms: int | None = None
    query_log_id: str | None = Field(
        default=None,
        description="The retrieval run behind this answer, when there was one. Feedback that "
        "cannot name what produced the answer is a mood, not a datum — and there are answers "
        "with no retrieval at all, which is why this is nullable and why feedback lives on "
        "the message rather than on the retrieval row.",
    )


@runtime_checkable
class ConversationStore(Protocol):
    """Reading history and writing turns. Sharing lives in :mod:`manicule.generation.sharing`."""

    async def create_conversation(
        self, *, user_id: str | None = None, title: str | None = None
    ) -> str:
        """Start a conversation and return its id.

        No workspace parameter: an implementation is bound to one, because a scope a caller
        can forget to pass — or pass wrongly — is not a scope.
        """
        ...

    async def history(self, conversation_id: str, *, limit: int = 20) -> Sequence[Turn]:
        """Prior turns, oldest first, with the citations each answer carried.

        A limit on rows read, not a budget: the token budget is applied afterwards, against
        the generation model's tokenizer, by whole paired turns.
        """
        ...

    async def append(self, message: StoredMessage) -> str:
        """Write a turn and return its id."""
        ...

    async def record_feedback(
        self,
        message_id: str,
        *,
        feedback: Feedback,
        reason: FeedbackReason | None = None,
        comment: str = "",
    ) -> bool:
        """Rate an answer. Returns whether a row actually matched.

        The return value is the correction: reporting success without checking that anything
        matched means feedback on a mistyped id is silently accepted and never seen again.
        """
        ...


class SharedTurn(BaseModel):
    """One turn of a conversation as an **anonymous** viewer of a share link receives it.

    A distinct type from :class:`~manicule.generation.history.Turn`, and the distinction is
    the enforcement. A shared transcript must carry citation *labels* and never passage text,
    document ids or anchors — and a rule kept by remembering to blank fields is one a route
    forgets. This type has nowhere to put them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(pattern="^(user|assistant)$")
    content: str
    citations: tuple[CitationLabel, ...] = ()


@runtime_checkable
class ShareStore(Protocol):
    """Minting, reading and revoking share links."""

    async def create_share(
        self, conversation_id: str, *, token_hash: str, expires_at: datetime, shared_at: datetime
    ) -> bool: ...

    async def revoke_share(self, conversation_id: str) -> bool: ...

    async def shared_conversation(
        self, token_hash: str, *, now: datetime, sharing_enabled: bool = True
    ) -> Sequence[SharedTurn]:
        """The conversation a live token names, projected for an anonymous reader.

        Deliberately **one** call rather than "resolve the token" followed by "read the
        messages". Two calls leave a window in which an owner's revocation lands between
        them, and they let a caller reach the second one holding only a conversation id.

        An empty result covers every reason at once — unknown token, expired link, revoked
        link, soft-deleted conversation, sharing switched off — because distinguishing them
        for an unauthenticated caller tells them which of their guesses was closest.
        """
        ...


__all__ = [
    "ConversationStore",
    "Feedback",
    "FeedbackReason",
    "ShareStore",
    "SharedTurn",
    "StoredMessage",
]
