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

    @property
    def source_id(self) -> SourceId:
        return self.ref.source_id


__all__ = [
    "DiscoveredDoc",
    "DocRef",
    "SourceId",
    "Watermark",
]
