"""Request bodies, which are also the published schema.

FastAPI derives the OpenAPI document from these, so what a client is told to send is what the
handler actually accepts. They are closed — ``extra="forbid"`` — for the same reason the
configuration models are: a field that is silently ignored looks exactly like one that worked.

Nothing here validates *policy*. A limit's upper bound is here, because a page size of a
million is a request shape rather than a decision; whether a profile exists, whether a role is
real, whether sharing is permitted are all the service's, and duplicating one of them here
would be a second opinion for the surfaces to disagree over.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.organization import CollectionRule


class Body(BaseModel):
    """Base for every request body: closed, so a typo is a 422 rather than a silence."""

    model_config = ConfigDict(extra="forbid")


class AskBody(Body):
    """A question for the corpus."""

    question: str = Field(min_length=1, description="A natural-language question.")
    profile: str | None = Field(
        default=None, description="``fast``, ``balanced`` or ``precise``. Omit for the configured."
    )
    limit: int | None = Field(
        default=None, ge=1, le=100, description="How many passages to retrieve before answering."
    )
    sources: tuple[str, ...] = Field(default=(), description="Restrict to these source names.")
    conversation_id: str | None = Field(
        default=None, description="Continue a conversation, and persist this turn to it."
    )


class ResearchBody(Body):
    """A question to research across several searches."""

    question: str = Field(min_length=1)
    profile: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    sources: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()


class FeedbackBody(Body):
    """A rating on one answer."""

    message_id: str = Field(min_length=1)
    feedback: str = Field(description="``positive`` or ``negative``.")
    reason: str | None = Field(
        default=None,
        description="For a negative rating: ``wrong``, ``incomplete``, ``citation-wrong``, "
        "``too-slow`` or ``other``.",
    )
    comment: str = Field(default="", max_length=2000)


class ConversationBody(Body):
    """A new conversation."""

    title: str | None = Field(default=None, max_length=200)


class ConversationPatch(Body):
    """A change to a conversation. Only the title is changeable."""

    title: str = Field(min_length=1, max_length=200)


class ShareBody(Body):
    """A request to mint a share link."""

    ttl_s: int | None = Field(
        default=None,
        gt=0,
        description="How long the link should live. Clamped to security.sharing.link_ttl_s, "
        "which is a ceiling rather than a default.",
    )


class CollectionBody(Body):
    """A new collection."""

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    rule: CollectionRule | None = None


class CollectionRuleBody(Body):
    """An explicit rule replacement; omission cannot be mistaken for clearing."""

    rule: CollectionRule


class CollectionUpdateBody(Body):
    """A change to what a collection is *for*, never to what is in it.

    Required rather than defaulted, the way :class:`ConversationPatch` treats a title. The
    write is a set, not a merge, so a defaulted field would make ``PATCH`` with an empty body
    erase the description instead of leaving it alone. An empty string clears it, deliberately.
    """

    description: str = Field(max_length=2000)


class CollectionNameBody(Body):
    """A new name for an existing collection."""

    name: str = Field(min_length=1, max_length=200)


class TagBody(Body):
    """A new tag, or the existing one of that name."""

    name: str = Field(min_length=1, max_length=200)
    color: str | None = Field(default=None, max_length=32)


class KeyBody(Body):
    """A request to mint an API key."""

    name: str = Field(min_length=1, max_length=200)
    role: str = Field(default="member", description="``admin``, ``member`` or ``viewer``.")
    expires_days: int | None = Field(default=None, ge=1, le=3650)


class SyncBody(Body):
    """A request to run one configured connector."""

    limit: int | None = Field(default=None, ge=1)
    acquire_only: bool = False


__all__ = [
    "AskBody",
    "Body",
    "CollectionBody",
    "ConversationBody",
    "ConversationPatch",
    "FeedbackBody",
    "KeyBody",
    "ShareBody",
    "SyncBody",
    "TagBody",
]
