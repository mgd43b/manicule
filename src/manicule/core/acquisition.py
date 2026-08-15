"""Durable source-acquisition vocabulary.

Discovery records are deliberately richer than queue messages: once committed they must be
enough to fetch and publish a source revision after the discovery cursor and process are gone.
The types here validate every JSON value at the storage boundary so callers never exchange
untyped dictionaries with the journal.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.content import Metadata
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark


class AcquisitionRunState(StrEnum):
    """Forward-only lifecycle of one connector enumeration and its local work."""

    ENUMERATING = "enumerating"
    ACQUIRING = "acquiring"
    INDEXING = "indexing"
    SETTLED = "settled"


class AcquisitionRecordState(StrEnum):
    """Durable state of one source identity within a run."""

    DISCOVERED = "discovered"
    ACQUIRING = "acquiring"
    ACQUIRED = "acquired"
    UNCHANGED = "unchanged"
    INDEXING = "indexing"
    SETTLED = "settled"
    RETRY = "retry"


class AcquisitionStage(StrEnum):
    """Stages allowed in persisted diagnostics."""

    ENUMERATION = "enumeration"
    ACQUISITION = "acquisition"
    INDEXING = "indexing"
    PUBLICATION = "publication"
    CAPACITY = "capacity"


class AcquisitionFailureCode(StrEnum):
    """Bounded diagnostic vocabulary; arbitrary exception text is intentionally absent."""

    AUTHENTICATION = "authentication"
    CAPACITY = "capacity"
    CURSOR_EXPIRED = "cursor_expired"
    FETCH_FAILED = "fetch_failed"
    MISSING_BODY = "missing_body"
    SOURCE_DELETED = "source_deleted"
    STALE_BODY = "stale_body"
    PARSE_FAILED = "parse_failed"
    EMBED_FAILED = "embed_failed"
    PUBLICATION_FAILED = "publication_failed"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class UnsetValue(Enum):
    """Sentinel type distinguishing an omitted update from an explicit ``None``."""

    UNSET = "unset"


UNSET = UnsetValue.UNSET


class AcquisitionDiagnostic(BaseModel):
    """A safe persisted failure envelope with no source text, URL or exception message."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    stage: AcquisitionStage
    code: AcquisitionFailureCode
    retryable: bool = True


class AcquisitionSource(BaseModel):
    """Everything learned at discovery that later acquisition/publication needs."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    ref: DocRef
    version_token: str | None = None
    title: str = ""
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    source_modified_at: datetime | None = None
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _source_modified_at_is_aware(self) -> Self:
        if self.source_modified_at is not None and self.source_modified_at.tzinfo is None:
            msg = "source_modified_at must be timezone-aware"
            raise ValueError(msg)
        return self

    @classmethod
    def from_discovered(
        cls,
        discovered: DiscoveredDoc,
        *,
        source_modified_at: datetime | None = None,
        provenance: Metadata | None = None,
    ) -> AcquisitionSource:
        """Build the durable form without asking call sites to reshape connector JSON."""
        return cls(
            ref=discovered.ref,
            version_token=discovered.version_token,
            title=discovered.title,
            media_type=discovered.media_type,
            size_bytes=discovered.size_bytes,
            source_modified_at=source_modified_at,
            metadata=discovered.metadata,
            provenance=provenance or {},
        )

    @property
    def source_id(self) -> str:
        return self.ref.source_id


class AcquisitionRun(BaseModel):
    """Read model for a durable run."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    id: str
    workspace_id: str
    connector_id: str
    connector: str
    state: AcquisitionRunState
    base_watermark: Watermark | None = None
    candidate_watermark: Watermark | None = None
    enumeration_completed_at: datetime | None = None
    watermark_committed_at: datetime | None = None
    lease_owner: str | None = None
    lease_generation: int = Field(ge=0)
    lease_expires_at: datetime | None = None
    discovered_count: int = Field(ge=0)
    acquired_count: int = Field(ge=0)
    indexed_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    metadata_bytes: int = Field(ge=0)
    acquired_blob_bytes: int = Field(ge=0)
    diagnostic: AcquisitionDiagnostic | None = None
    created_at: datetime
    updated_at: datetime


class AcquisitionRecord(BaseModel):
    """Read model for one journaled source identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    run_id: str
    sequence: int = Field(ge=0)
    source: AcquisitionSource
    state: AcquisitionRecordState
    blob_ref: str | None = None
    fetched_version_token: str | None = None
    attempts: int = Field(ge=0)
    diagnostic: AcquisitionDiagnostic | None = None
    created_at: datetime
    updated_at: datetime


__all__ = [
    "UNSET",
    "AcquisitionDiagnostic",
    "AcquisitionFailureCode",
    "AcquisitionRecord",
    "AcquisitionRecordState",
    "AcquisitionRun",
    "AcquisitionRunState",
    "AcquisitionSource",
    "AcquisitionStage",
    "UnsetValue",
]
