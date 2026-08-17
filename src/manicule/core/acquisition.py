"""Durable source-acquisition vocabulary.

Discovery records are deliberately richer than queue messages: once committed they must be
enough to fetch and publish a source revision after the discovery cursor and process are gone.
The types here validate every JSON value at the storage boundary so callers never exchange
untyped dictionaries with the journal.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.content import Metadata, RawDocument
from manicule.core.ids import content_hash
from manicule.core.provenance import PROVENANCE_KEY
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark


class AcquisitionRunState(StrEnum):
    """Forward-only lifecycle of one connector enumeration and its local work."""

    ENUMERATING = "enumerating"
    ACQUIRING = "acquiring"
    INDEXING = "indexing"
    SETTLED = "settled"


class AcquisitionInventoryState(StrEnum):
    """Whether this run's completed identity inventory can still prove source coverage."""

    CURRENT = "current"
    REENUMERATION_REQUIRED = "reenumeration_required"
    REENUMERATING = "reenumerating"
    RECONCILED = "reconciled"


class AcquisitionRecordState(StrEnum):
    """Durable state of one source identity within a run."""

    DISCOVERED = "discovered"
    ACQUIRING = "acquiring"
    ACQUIRED = "acquired"
    UNCHANGED = "unchanged"
    INDEXING = "indexing"
    SETTLED = "settled"
    RETRY = "retry"
    OMITTED = "omitted"


class SnapshotPromotionPolicy(StrEnum):
    """Completeness contract applied before a source snapshot becomes authoritative."""

    REQUIRE_COMPLETE = "require_complete"
    ALLOW_OMISSIONS = "allow_omissions"


class SnapshotCompleteness(StrEnum):
    """How much of the enumerated source a promoted snapshot can reproduce locally."""

    COMPLETE = "complete"
    PARTIAL = "partial"


class SnapshotItemOutcome(StrEnum):
    """Immutable byte-coverage result for one deterministic manifest member."""

    RETAINED = "retained"
    REUSED = "reused"
    OMITTED = "omitted"


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
    SNAPSHOT_MISSING = "snapshot_missing"
    SNAPSHOT_CORRUPT = "snapshot_corrupt"
    CORRUPT_BODY = "corrupt_body"
    SOURCE_DELETED = "source_deleted"
    STALE_BODY = "stale_body"
    PARSE_FAILED = "parse_failed"
    EMBED_FAILED = "embed_failed"
    PUBLICATION_FAILED = "publication_failed"
    INTERRUPTED = "interrupted"
    LEGACY_UNVERIFIED = "legacy_unverified"
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


class AcquisitionFence(BaseModel):
    """The persisted ownership generation authorizing one attempt-owned mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    run_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    generation: int = Field(ge=1)
    now: datetime

    @model_validator(mode="after")
    def _now_is_aware(self) -> Self:
        if self.now.tzinfo is None:
            msg = "acquisition fence time must be timezone-aware"
            raise ValueError(msg)
        return self


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
        if provenance is None:
            raw_provenance = discovered.metadata.get(PROVENANCE_KEY)
            if isinstance(raw_provenance, Mapping):
                provenance = dict(raw_provenance)
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


class AcquiredSource(BaseModel):
    """The complete, byte-validated source envelope retained for offline derivation."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    source_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    encoding: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    text_content: bool
    content_hash: str = Field(min_length=1)
    byte_length: int = Field(ge=0)

    @classmethod
    def from_raw(cls, raw: RawDocument) -> AcquiredSource:
        """Capture every non-body field needed to reconstruct ``raw`` from its blob."""
        data = raw.as_bytes()
        return cls(
            source_id=raw.source_id,
            uri=raw.uri,
            media_type=raw.media_type,
            encoding=raw.encoding,
            metadata=raw.metadata,
            text_content=isinstance(raw.content, str),
            content_hash=content_hash(data),
            byte_length=len(data),
        )

    def raw(self, data: bytes) -> RawDocument:
        """Rebuild the connector result, refusing corrupt or mismatched retained bytes."""
        if len(data) != self.byte_length or content_hash(data) != self.content_hash:
            msg = "retained source bytes do not match their acquisition snapshot"
            raise ValueError(msg)
        content: bytes | str = data.decode(self.encoding) if self.text_content else data
        return RawDocument(
            source_id=self.source_id,
            uri=self.uri,
            media_type=self.media_type,
            encoding=self.encoding,
            content=content,
            metadata=self.metadata,
        )


class AcquisitionRun(BaseModel):
    """Read model for a durable run."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    id: str
    workspace_id: str
    connector_id: str
    connector: str
    source_scope: str = ""
    scope_fingerprint: str = ""
    scope_inventory_complete: bool = True
    promotion_policy: SnapshotPromotionPolicy = SnapshotPromotionPolicy.REQUIRE_COMPLETE
    state: AcquisitionRunState
    base_watermark: Watermark | None = None
    base_watermark_scope_fingerprint: str | None = None
    candidate_watermark: Watermark | None = None
    enumeration_completed_at: datetime | None = None
    acquisition_completed_at: datetime | None = None
    promoted_at: datetime | None = None
    watermark_committed_at: datetime | None = None
    superseded_at: datetime | None = None
    superseded_by: str | None = None
    supersedes_run_id: str | None = None
    inventory_state: AcquisitionInventoryState = AcquisitionInventoryState.CURRENT
    reconciled_deleted_count: int = Field(default=0, ge=0)
    membership_hash: str | None = None
    completeness: SnapshotCompleteness | None = None
    omission_count: int = Field(default=0, ge=0)
    omission_reasons: dict[AcquisitionFailureCode, int] = Field(default_factory=lambda: {})
    lease_owner: str | None = None
    lease_generation: int = Field(ge=0)
    lease_expires_at: datetime | None = None
    discovered_count: int = Field(ge=0)
    acquired_count: int = Field(ge=0)
    indexed_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    reused_count: int = Field(default=0, ge=0)
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
    snapshot_outcome: SnapshotItemOutcome | None = None
    blob_ref: str | None = None
    acquired_source: AcquiredSource | None = None
    fetched_version_token: str | None = None
    attempts: int = Field(ge=0)
    snapshot_diagnostic: AcquisitionDiagnostic | None = None
    diagnostic: AcquisitionDiagnostic | None = None
    created_at: datetime
    updated_at: datetime


__all__ = [
    "UNSET",
    "AcquiredSource",
    "AcquisitionDiagnostic",
    "AcquisitionFailureCode",
    "AcquisitionFence",
    "AcquisitionInventoryState",
    "AcquisitionRecord",
    "AcquisitionRecordState",
    "AcquisitionRun",
    "AcquisitionRunState",
    "AcquisitionSource",
    "AcquisitionStage",
    "SnapshotCompleteness",
    "SnapshotItemOutcome",
    "SnapshotPromotionPolicy",
    "UnsetValue",
]
