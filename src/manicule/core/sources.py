"""Connector vocabulary: references, watermarks and discovery results.

Incremental sync tells you what changed. It cannot tell you what was deleted, because a
deleted page simply stops appearing — so reconciliation is a separate obligation, and the
types here keep it possible: a discovered document always carries an id that a later
reconciliation pass can compare against.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.content import Metadata

type SourceId = str
"""A document's stable identifier within its source. Unique per connector, not globally."""


class Watermark(BaseModel):
    """How far a connector got last time.

    :attr:`value` is opaque and connector-defined — a CQL timestamp, a Drive page token, a
    commit SHA. manicule stores it, hands it back, and never interprets it, which is what
    lets each source use its own native change signal instead of a lowest common
    denominator that works badly everywhere.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: str = Field(min_length=1)
    observed_at: datetime
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _observed_at_is_aware(self) -> Self:
        if self.observed_at.tzinfo is None:
            msg = "observed_at must be timezone-aware"
            raise ValueError(msg)
        return self


class DocRef(BaseModel):
    """Enough to fetch one document from its source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId = Field(min_length=1)
    uri: str = Field(min_length=1)
    metadata: Metadata = Field(
        default_factory=dict,
        description="Whatever the connector needs to fetch this and nothing more — a "
        "space key, a blob SHA, a bucket and key.",
    )


class DiscoveredDoc(BaseModel):
    """A document a connector found, and whether it is worth fetching.

    Discovery must be decidable without fetching: :attr:`version_token` is compared against
    the stored one, and an unchanged document is never downloaded. A connector that cannot
    supply one forces a full re-fetch of its entire source on every sync.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: DocRef
    version_token: str | None = Field(
        default=None,
        description="Opaque change token. ``None`` means the source offers none, and the "
        "document must be fetched and hashed to tell whether it changed.",
    )
    title: str = ""
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: Metadata = Field(default_factory=dict)
    force_fetch: bool = False
    """Bypass token equality for a source dependency whose expanded content may have changed.

    The source token remains the source's actual revision. Replacing it with ``None`` would
    force one fetch but also erase the stored token, making every later sync fetch the page
    again. This narrow flag carries the scheduling decision without changing source evidence.
    """

    @property
    def source_id(self) -> SourceId:
        return self.ref.source_id


class EnumerationProgress(BaseModel):
    """Aggregate-only facts about a connector's live enumeration, for durable diagnostics.

    Counts, sizes and closed booleans — never a source identity, space key, URL, credential
    or response fragment, because everything here is persisted on the acquisition run and
    served by every status surface. A connector that adapts its request shape mid-walk (the
    Confluence direct inventory shrinks its page size when a deep offset times out) is
    otherwise indistinguishable from one that is hung, and an operator who cannot tell those
    apart restarts a run that was working.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    offset: int = Field(
        default=0,
        ge=0,
        description="The current stream's next explicit offset — a count of rows already "
        "admitted, never a source identity.",
    )
    requested_page_size: int | None = Field(
        default=None,
        ge=1,
        description="The page size the next request will ask for, after any adaptation.",
    )
    timeout_retries: int = Field(
        default=0,
        ge=0,
        description="How many requests timed out and were re-asked with a smaller page.",
    )
    page_size_reduced: bool = Field(
        default=False,
        description="Whether any timeout shrank the requested page below its configured size.",
    )
    reached_empty_page: bool | None = Field(
        default=None,
        description="Whether the walk ended at a validated explicit empty page. ``None`` when "
        "the enumeration does not prove its end that way — an incremental query, or a "
        "connector without explicit offsets.",
    )


__all__ = [
    "DiscoveredDoc",
    "DocRef",
    "EnumerationProgress",
    "SourceId",
    "Watermark",
]
