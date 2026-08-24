"""The relational schema.

The tables carry the shape ``PLAN.md`` §2 and the durable acquisition boundary. The reasoning
for every column, index and constraint lives in the storage and ingest design documents; what
is here is the schema they describe.

Two conventions apply throughout and are not repeated per column:

* **Timestamps are :class:`~manicule.storage.types.UtcDateTime` set from Python**, never a SQL
  default. See that class for why a second writer is the problem.
* **Constraints are named by a convention on the metadata.** Batch migrations have to *name*
  the constraint they drop and SQLite generates anonymous ones, so a project without this
  discovers months in that it cannot migrate.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import Any

from pydantic import JsonValue
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from manicule.core.acquisition import (
    AcquisitionInventoryState,
    AcquisitionRecordState,
    AcquisitionRunState,
    SnapshotCompleteness,
    SnapshotItemOutcome,
    SnapshotPromotionPolicy,
)
from manicule.core.content import BlockKind, DocumentStatus, PipelineStage
from manicule.core.rebuild import RebuildState
from manicule.core.reconciliation import ReconciliationRunState
from manicule.storage.types import UtcDateTime, utcnow

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Deterministic constraint names.

Required, not cosmetic. SQLite cannot alter a constraint, so Alembic implements every such
change as create-copy-swap — and to drop a constraint it must name it. SQLite's
auto-generated names are not stable, so without this the first migration that changes a
``CHECK`` cannot be written.
"""

WITHOUT_ROWID: dict[str, Any] = {"sqlite_with_rowid": False}
"""For pure association tables: store in the primary-key B-tree, skip the rowid index."""


def _enum_values(enum_type: type[PyEnum]) -> list[str]:
    """Store an enum by its ``value``, not by its Python member name.

    ``StrEnum`` members are already strings, but SQLAlchemy defaults to persisting the member
    *name*. The two differ here — ``NO_EXTRACTABLE_TEXT`` against ``no_extractable_text`` —
    and the value is what every document, query and ``CHECK`` constraint in the design refers
    to.
    """
    return [str(member.value) for member in enum_type]


def _value_enum(enum_type: type[PyEnum], name: str) -> Enum:
    """A ``VARCHAR`` + ``CHECK`` column over an enum's values.

    ``native_enum=False`` because SQLite has no enum type, and ``create_constraint=True``
    because **it defaults to False** — without it this renders a bare ``VARCHAR`` and the
    constraint the design relies on does not exist. That is the failure this whole design
    guards against elsewhere: a check that cannot fail, reading as coverage.

    What it buys: a misspelled status is an error at the write that caused it, rather than a
    document that is silently unservable forever because retrieval filters on ``indexed``.
    """
    return Enum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=_enum_values,
    )


def _status_enum() -> Enum:
    return _value_enum(DocumentStatus, "document_status")


def _stage_enum() -> Enum:
    return _value_enum(PipelineStage, "pipeline_stage")


def _kind_enum() -> Enum:
    return _value_enum(BlockKind, "block_kind")


def _acquisition_run_state_enum() -> Enum:
    return _value_enum(AcquisitionRunState, "acquisition_run_state")


def _acquisition_inventory_state_enum() -> Enum:
    return _value_enum(AcquisitionInventoryState, "acquisition_inventory_state")


def _acquisition_record_state_enum() -> Enum:
    return _value_enum(AcquisitionRecordState, "acquisition_record_state")


def _reconciliation_run_state_enum() -> Enum:
    return _value_enum(ReconciliationRunState, "reconciliation_run_state")


def _rebuild_state_enum() -> Enum:
    return _value_enum(RebuildState, "rebuild_state")


class Base(DeclarativeBase):
    """Declarative base carrying the naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# --- identity and tenancy ----------------------------------------------------------------


class Workspace(Base):
    """A tenant. Every query is scoped to one; team mode makes that a security boundary."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    mode: Mapped[str] = mapped_column(Text, nullable=False, default="personal")
    settings: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=dict)
    derived_reset_epoch: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    """Monotonic fence invalidating derived writers assembled before a confirmed reset."""
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (CheckConstraint("mode IN ('personal', 'team')", name="mode_is_known"),)


class WorkspaceMember(Base):
    """Who may see a workspace, and in what role.

    Carries no API key column. The prior art kept a raw, unhashed key here, which is
    precisely what ``api_keys.key_hash`` exists to avoid; :class:`ApiKey` is the only key
    store.
    """

    __tablename__ = "workspace_members"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'member', 'viewer')", name="role_is_known"),
        WITHOUT_ROWID,
    )


# --- retained source bytes ---------------------------------------------------------------


class Blob(Base):
    """A retained original document, addressed by the hash of its bytes.

    Content-addressed, so the same attachment reachable from forty pages is stored once, and
    so a blob whose bytes do not hash to its own name is detectably corrupt without a
    reference copy.
    """

    __tablename__ = "blobs"

    hash: Mapped[str] = mapped_column(Text, primary_key=True)
    algo: Mapped[str] = mapped_column(Text, nullable=False, default="blake2b")
    media_type: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    stored_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    compression: Mapped[str] = mapped_column(Text, nullable=False, default="none")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint("size_bytes >= 0 AND stored_bytes >= 0", name="sizes_are_not_negative"),
    )


class AcquisitionMarker(Base):
    """Durable inventory for a filesystem acquisition-recovery marker."""

    __tablename__ = "acquisition_markers"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(Text)
    source_id: Mapped[str | None] = mapped_column(Text)
    blob_ref: Mapped[str | None] = mapped_column(Text)
    acquired_source: Mapped[JsonValue | None] = mapped_column(JSON)
    legacy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_acquisition_markers_run_id", "run_id"),
        Index("ix_acquisition_markers_blob_ref", "blob_ref"),
    )


# --- connectors --------------------------------------------------------------------------


class Connector(Base):
    """A configured source, and where its last sync got to."""

    __tablename__ = "connectors"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=dict)
    watermark: Mapped[JsonValue] = mapped_column(JSON)
    """Opaque, connector-defined. Stored and handed back, never interpreted.

    Separate from ``last_synced_at`` because a watermark is not always a timestamp — it may
    be a page token or a commit SHA.
    """
    watermark_scope_fingerprint: Mapped[str | None] = mapped_column(Text)

    sync_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    last_synced_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    error_message: Mapped[str | None] = mapped_column(Text)
    run_metadata: Mapped[JsonValue] = mapped_column("metadata", JSON, nullable=False, default=dict)
    """Last run's counters. Overwritten per run rather than accumulated, which is the right
    retention policy for a diagnostic (``docs/ingest.md`` §13.1)."""

    deleted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("uq_connectors_id_workspace_id", "id", "workspace_id", unique=True),
        Index(
            "uq_connectors_workspace_id_name",
            "workspace_id",
            "name",
            unique=True,
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


# --- durable acquisition ---------------------------------------------------------------


class AcquisitionRun(Base):
    """One immutable connector enumeration and the local work derived from it."""

    __tablename__ = "acquisition_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    connector_id: Mapped[str] = mapped_column(Text, nullable=False)
    connector_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_scope: Mapped[str] = mapped_column(Text, nullable=False, default="")
    scope_fingerprint: Mapped[str] = mapped_column(Text, nullable=False, default="")
    full_inventory_authority: Mapped[str] = mapped_column(Text, nullable=False, default="")
    scope_inventory_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    promotion_policy: Mapped[SnapshotPromotionPolicy] = mapped_column(
        Text,
        nullable=False,
        default=SnapshotPromotionPolicy.REQUIRE_COMPLETE,
    )
    state: Mapped[AcquisitionRunState] = mapped_column(
        _acquisition_run_state_enum(), nullable=False, default=AcquisitionRunState.ENUMERATING
    )
    base_watermark: Mapped[JsonValue | None] = mapped_column(JSON)
    base_watermark_scope_fingerprint: Mapped[str | None] = mapped_column(Text)
    candidate_watermark: Mapped[JsonValue | None] = mapped_column(JSON)
    enumeration_completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    acquisition_completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    promoted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    watermark_committed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    superseded_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    superseded_by: Mapped[str | None] = mapped_column(Text)
    supersedes_run_id: Mapped[str | None] = mapped_column(Text)
    inventory_state: Mapped[AcquisitionInventoryState] = mapped_column(
        _acquisition_inventory_state_enum(),
        nullable=False,
        default=AcquisitionInventoryState.CURRENT,
    )
    reconciled_deleted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    membership_hash: Mapped[str | None] = mapped_column(Text)
    completeness: Mapped[SnapshotCompleteness | None] = mapped_column(Text)
    omission_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    omission_reasons: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=dict)
    lease_owner: Mapped[str | None] = mapped_column(Text)
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    discovered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    acquired_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    indexed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unchanged_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reused_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    acquired_blob_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Aggregate-only adaptive enumeration facts — counts, a size, closed booleans; never a
    # source identity. NULLs mean "the connector made no such claim", which is a different
    # fact from zero or false and must survive the round trip.
    enumeration_offset: Mapped[int | None] = mapped_column(Integer)
    enumeration_page_size: Mapped[int | None] = mapped_column(Integer)
    enumeration_timeout_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enumeration_page_size_reduced: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    enumeration_reached_empty_page: Mapped[bool | None] = mapped_column(Boolean)
    diagnostic: Mapped[JsonValue | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["connector_id", "workspace_id"],
            ["connectors.id", "connectors.workspace_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("id", "workspace_id", "connector_id"),
        Index(
            "ix_acquisition_runs_workspace_connector_state", "workspace_id", "connector_id", "state"
        ),
        Index(
            "ix_acquisition_runs_workspace_connector_recovery",
            "workspace_id",
            "connector_id",
            "superseded_at",
            "state",
            "created_at",
        ),
        Index(
            "ix_acquisition_runs_inventory_recovery",
            "workspace_id",
            "connector_id",
            "inventory_state",
            "superseded_at",
            "created_at",
        ),
        Index(
            "ix_acquisition_runs_workspace_state_updated",
            "workspace_id",
            "state",
            "updated_at",
        ),
        Index(
            "ix_acquisition_runs_workspace_superseded_updated",
            "workspace_id",
            "superseded_at",
            "updated_at",
        ),
        Index(
            "ix_acquisition_runs_latest_promoted",
            "workspace_id",
            "connector_name",
            "scope_fingerprint",
            "promoted_at",
            "id",
        ),
        CheckConstraint("lease_generation >= 0", name="lease_generation_is_not_negative"),
        CheckConstraint(
            "discovered_count >= 0 AND acquired_count >= 0 AND indexed_count >= 0 "
            "AND unchanged_count >= 0 AND retry_count >= 0 AND metadata_bytes >= 0 "
            "AND acquired_blob_bytes >= 0",
            name="acquisition_run_counters_are_not_negative",
        ),
        CheckConstraint(
            "watermark_committed_at IS NULL OR enumeration_completed_at IS NOT NULL",
            name="committed_watermark_has_complete_enumeration",
        ),
        CheckConstraint(
            "watermark_committed_at IS NULL OR scope_inventory_complete = 1",
            name="committed_watermark_has_complete_scope_inventory",
        ),
        CheckConstraint("omission_count >= 0", name="snapshot_omissions_are_not_negative"),
        CheckConstraint(
            "reconciled_deleted_count >= 0", name="reconciled_deletions_are_not_negative"
        ),
        CheckConstraint("reused_count >= 0", name="reused_acquisition_count_is_not_negative"),
        CheckConstraint(
            "promotion_policy IN ('require_complete', 'allow_omissions')",
            name="snapshot_promotion_policy_is_known",
        ),
        CheckConstraint(
            "completeness IS NULL OR completeness IN ('complete', 'partial')",
            name="snapshot_completeness_is_known",
        ),
        CheckConstraint(
            "promoted_at IS NULL OR acquisition_completed_at IS NOT NULL",
            name="promoted_snapshot_has_complete_acquisition",
        ),
        CheckConstraint(
            "watermark_committed_at IS NULL OR promoted_at IS NOT NULL "
            "OR membership_hash = 'legacy-unverified'",
            name="committed_watermark_has_promoted_snapshot",
        ),
    )


class AcquisitionRecord(Base):
    """A source record acknowledged only after this row's transaction commits."""

    __tablename__ = "acquisition_records"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    connector_id: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    marker_name: Mapped[str | None] = mapped_column(Text)
    source_record: Mapped[JsonValue] = mapped_column(JSON, nullable=False)
    state: Mapped[AcquisitionRecordState] = mapped_column(
        _acquisition_record_state_enum(),
        nullable=False,
        default=AcquisitionRecordState.DISCOVERED,
    )
    snapshot_outcome: Mapped[SnapshotItemOutcome | None] = mapped_column(Text)
    blob_ref: Mapped[str | None] = mapped_column(ForeignKey("blobs.hash", ondelete="RESTRICT"))
    acquired_source: Mapped[JsonValue | None] = mapped_column(JSON)
    fetched_version_token: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    snapshot_diagnostic: Mapped[JsonValue | None] = mapped_column(JSON)
    diagnostic: Mapped[JsonValue | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "workspace_id", "connector_id"],
            [
                "acquisition_runs.id",
                "acquisition_runs.workspace_id",
                "acquisition_runs.connector_id",
            ],
            ondelete="CASCADE",
        ),
        UniqueConstraint("run_id", "source_id"),
        UniqueConstraint("run_id", "sequence"),
        Index("ix_acquisition_records_run_state_sequence", "run_id", "state", "sequence"),
        Index("ix_acquisition_records_run_blob_state", "run_id", "blob_ref", "state"),
        Index("ix_acquisition_records_marker_name", "marker_name", unique=True),
        Index(
            "ix_acquisition_records_run_source_version",
            "run_id",
            "source_id",
            "fetched_version_token",
        ),
        CheckConstraint(
            "sequence >= 0 AND attempts >= 0", name="acquisition_record_numbers_are_not_negative"
        ),
        CheckConstraint(
            "state NOT IN ('acquired', 'indexing') OR blob_ref IS NOT NULL",
            name="blob_backed_acquisition_states_have_a_blob",
        ),
        CheckConstraint(
            "snapshot_outcome IS NULL OR snapshot_outcome IN ('retained', 'reused', 'omitted')",
            name="snapshot_item_outcome_is_known",
        ),
        CheckConstraint(
            "snapshot_diagnostic IS NULL OR json_valid(snapshot_diagnostic)",
            name="snapshot_diagnostic_is_valid_json",
        ),
    )


class AcquisitionBlobBacklog(Base):
    """One content-addressed blob currently pinned by unfinished acquisition records."""

    __tablename__ = "acquisition_blob_backlog"

    blob_ref: Mapped[str] = mapped_column(
        ForeignKey("blobs.hash", ondelete="RESTRICT"), primary_key=True
    )
    reference_count: Mapped[int] = mapped_column(Integer, nullable=False)
    stored_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("reference_count > 0", name="acquisition_blob_backlog_has_references"),
        CheckConstraint(
            "stored_bytes >= 0", name="acquisition_blob_backlog_bytes_are_not_negative"
        ),
    )


class AcquisitionBacklogCapacity(Base):
    """The O(1) exact total for content-addressed acquisition backlog admission."""

    __tablename__ = "acquisition_backlog_capacity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    acquired_blob_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("id = 1", name="acquisition_backlog_capacity_is_singleton"),
        CheckConstraint(
            "acquired_blob_bytes >= 0", name="acquisition_backlog_capacity_is_not_negative"
        ),
    )


# --- durable reconciliation -------------------------------------------------------------


class ReconciliationRun(Base):
    """A full-source inventory whose completion is a durable safety boundary."""

    __tablename__ = "reconciliation_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    connector_id: Mapped[str] = mapped_column(Text, nullable=False)
    connector_name: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[ReconciliationRunState] = mapped_column(
        _reconciliation_run_state_enum(),
        nullable=False,
        default=ReconciliationRunState.ENUMERATING,
    )
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    live_count: Mapped[int | None] = mapped_column(Integer)
    missing_count: Mapped[int | None] = mapped_column(Integer)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["connector_id", "workspace_id"],
            ["connectors.id", "connectors.workspace_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("id", "workspace_id", "connector_id"),
        Index(
            "ix_reconciliation_runs_workspace_connector_scope_state",
            "workspace_id",
            "connector_id",
            "scope",
            "state",
            "created_at",
        ),
        CheckConstraint(
            "seen_count >= 0 AND (live_count IS NULL OR live_count >= 0) "
            "AND (missing_count IS NULL OR missing_count >= 0)",
            name="reconciliation_counts_are_not_negative",
        ),
        CheckConstraint(
            "state = 'enumerating' OR state = 'canceled' OR completed_at IS NOT NULL",
            name="completed_reconciliation_states_have_completion",
        ),
    )


class ReconciliationInventoryItem(Base):
    """One deduplicated source identity in a full enumeration."""

    __tablename__ = "reconciliation_inventory_items"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    connector_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "workspace_id", "connector_id"],
            [
                "reconciliation_runs.id",
                "reconciliation_runs.workspace_id",
                "reconciliation_runs.connector_id",
            ],
            ondelete="CASCADE",
        ),
        WITHOUT_ROWID,
    )


class ReconciliationCandidate(Base):
    """The immutable document revisions a refused proposal asked to delete."""

    __tablename__ = "reconciliation_candidates"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    connector_id: Mapped[str] = mapped_column(Text, nullable=False)
    publication_id: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    version_token: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "workspace_id", "connector_id"],
            [
                "reconciliation_runs.id",
                "reconciliation_runs.workspace_id",
                "reconciliation_runs.connector_id",
            ],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        WITHOUT_ROWID,
    )


# --- documents and chunks ----------------------------------------------------------------


class Document(Base):
    """The indexed record of one source document.

    Identity is ``(workspace_id, source, source_id)`` rather than the URI. A URI is display
    data chosen for a human to read, and nothing obliges a source to keep it fixed; identity
    has to rest on the handle the source itself uses.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    publication_id: Mapped[str] = mapped_column(
        Text, nullable=False, default="legacy", server_default="legacy"
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )

    source: Mapped[str] = mapped_column(Text, nullable=False)
    """Name of the connector instance that owns this document.

    A name rather than a foreign key to :class:`Connector`, because that is what
    :attr:`manicule.core.content.Document.source` carries. Documents therefore outlive the
    deletion of a connector row, which is the behavior reconciliation wants anyway.
    """

    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    media_type: Mapped[str] = mapped_column(Text, nullable=False)

    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    version_token: Mapped[str | None] = mapped_column(Text)
    original_ref: Mapped[str | None] = mapped_column(ForeignKey("blobs.hash", ondelete="RESTRICT"))
    original_omitted_reason: Mapped[str | None] = mapped_column(Text)

    status: Mapped[DocumentStatus] = mapped_column(
        _status_enum(), nullable=False, default=DocumentStatus.PENDING
    )
    status_detail: Mapped[str | None] = mapped_column(Text)
    failed_stage: Mapped[PipelineStage | None] = mapped_column(_stage_enum())

    parse_fp: Mapped[str | None] = mapped_column(Text)
    chunk_fp: Mapped[str | None] = mapped_column(Text)
    embed_fp: Mapped[str | None] = mapped_column(Text)
    glossary_fp: Mapped[str | None] = mapped_column(Text)
    """Which fingerprints this document was last built with.

    Per-document lineage is what makes invalidation set-valued: a grammar upgrade that
    changes only code documents is a query, not a corpus-wide rebuild.

    ``parse_fp`` is one of the two with no corpus-wide counterpart in ``index_state``,
    and deliberately so. One document has one parser, so there is no single parse identity a
    whole index can be compared against — a ``pypdfium2`` bump makes the PDFs stale and says
    nothing about the Markdown. ``NULL`` means no recorded lineage: every row predating the
    column, and every document produced by a parser manicule does not ship and therefore
    cannot version.

    ``glossary_fp`` is the other, for the opposite reason: there *is* one detector, but its
    output is repaired one document at a time from chunks already stored, so the comparison has
    to name documents rather than refuse a run. It is on this table rather than beside the
    entries so that the question "which documents disagree with the installed detector" is an
    indexed predicate over ``documents`` — answerable without reading one word of glossary text,
    and answerable at all for a document that correctly stores no entries. ``NULL`` means the
    entries were never computed; a fingerprint whose ``detector`` reads ``disabled`` means
    detection was switched off when this document was last ingested, and the two are different
    states on purpose.
    """

    doc_metadata: Mapped[JsonValue] = mapped_column("metadata", JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )
    indexed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_seen_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    """When reconciliation last saw this document at the source.

    Without it, a skip and a deletion are indistinguishable, and deletion detection has
    nothing to diff against.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        Index(
            "uq_documents_identity",
            "workspace_id",
            "source",
            "source_id",
            unique=True,
            sqlite_where=text("deleted_at IS NULL"),
        ),
        Index("ix_documents_workspace_id_status", "workspace_id", "status"),
        Index("ix_documents_workspace_id_uri", "workspace_id", "uri"),
        Index("ix_documents_content_hash", "content_hash"),
        Index("ix_documents_parse_fp", "parse_fp"),
        Index("ix_documents_chunk_fp", "chunk_fp"),
        Index("ix_documents_embed_fp", "embed_fp"),
        Index("ix_documents_glossary_fp", "glossary_fp"),
        Index(
            "ix_documents_workspace_live_id",
            "workspace_id",
            "id",
            sqlite_where=text("deleted_at IS NULL"),
        ),
        Index(
            "ix_documents_deleted_at",
            "workspace_id",
            "deleted_at",
            sqlite_where=text("deleted_at IS NOT NULL"),
        ),
        CheckConstraint(
            "(status = 'failed') = (failed_stage IS NOT NULL)",
            name="failed_stage_iff_failed",
        ),
    )


class SourceDependency(Base):
    """One document's expanded dependency on another source identity.

    The target need not have a live row: an include whose target was deleted is precisely the
    relationship that must survive long enough to re-fetch its parents. The parent document owns
    the edge, so a successful publication replaces its complete dependency set atomically.
    """

    __tablename__ = "source_dependencies"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    source: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    target_source_id: Mapped[str] = mapped_column(Text, primary_key=True)

    __table_args__ = (
        Index("ix_source_dependencies_target", "workspace_id", "source", "target_source_id"),
        WITHOUT_ROWID,
    )


class Chunk(Base):
    """A retrievable unit of text, with everything needed to cite it.

    ``seq`` is an explicit rowid alias because the FTS5 index is external-content over this
    table and needs a ``content_rowid``. ``id`` is the opaque, content-derived chunk id from
    :func:`manicule.core.ids.chunk_id`.
    """

    __tablename__ = "chunks"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    vector_id: Mapped[str] = mapped_column(Text, nullable=False)
    """Physical vector row for this publication; safe for every SQL cascade to tombstone."""
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    text: Mapped[str] = mapped_column(Text, nullable=False)
    """What gets shown and cited. Exactly the source text, with nothing added.

    Immutable after parse. ``docs/ingest.md`` §3.3.1 forbids middleware from altering it,
    because it is what ``Parser.resolve`` must reproduce.
    """

    embed_text: Mapped[str] = mapped_column(Text, nullable=False)
    """Exactly what the embedder saw. Stored rather than recomputed, so re-embedding needs no
    re-chunking and cannot silently apply a changed derivation rule."""

    heading_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    """The breadcrumb joined for FTS. Redundant with ``heading_path`` and written from one
    place, because FTS5 indexes a string and a JSON array is not one."""

    heading_path: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=list)
    kind: Mapped[BlockKind] = mapped_column(_kind_enum(), nullable=False, default=BlockKind.PROSE)
    lang: Mapped[str | None] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    anchor: Mapped[JsonValue] = mapped_column(JSON, nullable=False)
    """The tagged union from :mod:`manicule.core.anchors`, validated on the way in and out.

    The one JSON column whose shape is locked: changing it invalidates every stored citation.
    """

    chunk_metadata: Mapped[JsonValue] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("document_id", "position", name="uq_chunks_document_id_position"),
        Index("ix_chunks_kind", "kind"),
        CheckConstraint("position >= 0 AND token_count >= 0", name="counts_are_not_negative"),
    )


class ChunkRelation(Base):
    """A typed edge between two chunks.

    Real foreign keys, which is the direct payoff of chunks being a table: orphan cleanup is
    the cascade rather than a ``LIKE`` over string-formatted ids.
    """

    __tablename__ = "chunk_relations"

    source_chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    target_chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    relation_type: Mapped[str] = mapped_column(Text, primary_key=True)

    __table_args__ = (
        # Not redundant with the primary key: lookups are `source = ? OR target = ?`, and a
        # composite key leading with source cannot serve the second half of that predicate.
        Index("ix_chunk_relations_target_chunk_id", "target_chunk_id"),
        CheckConstraint("source_chunk_id <> target_chunk_id", name="no_self_relation"),
        WITHOUT_ROWID,
    )


class DocumentVersion(Base):
    """A prior state of a document, with the bytes that produced it."""

    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    original_ref: Mapped[str | None] = mapped_column(ForeignKey("blobs.hash", ondelete="RESTRICT"))
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    changes: Mapped[JsonValue] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    bytes_released_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_document_versions_document_id_version"),
    )


# --- organization ------------------------------------------------------------------------


class Collection(Base):
    """A named grouping of documents, manual or rule-driven."""

    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    auto_rules: Mapped[JsonValue] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_collections_ws_name"),)


class CollectionDocument(Base):
    """Membership of a document in a collection."""

    __tablename__ = "collection_documents"

    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (WITHOUT_ROWID,)


class GlossaryEntry(Base):
    """One acronym definition a document states, with where it says it.

    **Scoped through ``document_id`` and nothing else.** There is no ``workspace_id`` column
    here, and its absence is the design rather than an omission: a document id is derived from
    the workspace (:func:`~manicule.core.ids.document_id`), so a copy of the workspace on this
    row could only ever be a second answer to a question the foreign key already settles — and
    a second answer can disagree. Collection membership is likewise read through the join table
    at query time rather than copied here, because a collection's contents change without any
    glossary being re-ingested.

    ``chunk_id`` is a real foreign key with the same cascade the chunk table gets, so a
    re-parse that replaces a document's chunks takes its stale definitions with it. Without
    that, an edited glossary would keep answering with the line it used to have, citing a chunk
    that no longer exists.
    """

    __tablename__ = "glossary_entries"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )

    acronym: Mapped[str] = mapped_column(Text, nullable=False)
    """The normalized lookup key, upper case and stripped. Written by
    :func:`~manicule.core.glossary.normalize_acronym` and read back with keys produced by the
    same function; a store that normalized differently would miss silently."""

    display: Mapped[str] = mapped_column(Text, nullable=False)
    expansion: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str] = mapped_column(Text, nullable=False, default="")
    form: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_glossary_entries_acronym", "acronym"),
        Index("ix_glossary_entries_document_id", "document_id"),
        Index("ix_glossary_entries_chunk_id", "chunk_id"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="confidence_is_a_fraction"),
    )


class GlossaryAlias(Base):
    """Another key that resolves to an entry.

    A table rather than a JSON column on :class:`GlossaryEntry`, because this is the column a
    lookup filters on and SQLite cannot index inside JSON. A glossary with two hundred terms is
    queried on every search that names one of them; an unindexed ``LIKE`` over a JSON array
    would make the cheapest part of the feature the most expensive.
    """

    __tablename__ = "glossary_aliases"

    entry_id: Mapped[str] = mapped_column(
        ForeignKey("glossary_entries.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(Text, primary_key=True)

    __table_args__ = (Index("ix_glossary_aliases_key", "key"), WITHOUT_ROWID)


class Tag(Base):
    """A label applicable to documents."""

    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    color: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_tags_workspace_id_name"),)


class DocumentTag(Base):
    """Application of a tag to a document."""

    __tablename__ = "document_tags"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    __table_args__ = (WITHOUT_ROWID,)


# --- conversations -----------------------------------------------------------------------


class Conversation(Base):
    """A chat thread."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    shared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    share_token_hash: Mapped[str | None] = mapped_column(Text, unique=True)
    """The share token, **hashed**, exactly like ``api_keys.key_hash``.

    The argument is not that the hash protects this row from somebody holding the database —
    that person has the conversation anyway. It is that a share token is a live credential
    for an *unauthenticated* URL, and this database is backed up, exported and imported, so a
    plaintext token travels into artifacts that leave the access boundary that created it.
    The token is shown to its creator once and never stored.

    Revocation clears this column rather than flipping :attr:`shared` beside a still-valid
    token, so a revoked link stops resolving instead of merely looking revoked.
    """

    share_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    """When the link stops working. Enforced on every read.

    A capability with no expiry accumulates forever and the set of live ones becomes
    unknowable. A row with a hash and no expiry is treated as expired, which fails closed.
    """

    shared_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    """When it was shared — so an owner can see it, and so the audit record has a join.

    Also what makes a share a **snapshot**: only messages created at or before this moment are
    exposed. The alternative is a live view, where turn 7 becomes public the moment it is
    written and nobody re-reads a link they already sent.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class Message(Base):
    """One turn in a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[JsonValue] = mapped_column(JSON)
    """Citations, which embed anchors — so this column inherits the anchor lock: a stored
    conversation's citations must keep resolving."""

    profile_used: Mapped[str | None] = mapped_column(Text)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    response_time_ms: Mapped[int | None] = mapped_column(Integer)

    finish_reason: Mapped[str | None] = mapped_column(Text)
    """How generation ended. A partial or truncated answer is persisted like any other and
    must be distinguishable from a complete one; an answer that simply stops looks exactly
    like one that finished."""

    feedback: Mapped[str | None] = mapped_column(Text)
    """A rating on **this answer**.

    On the message rather than on ``query_logs``, and that is not bookkeeping preference.
    There are answers with no retrieval behind them, and answers whose retrieval succeeded
    and whose generation failed; both are ratable and neither has a usable query-log row.
    A message always exists, including for a partial answer.
    """

    feedback_reason: Mapped[str | None] = mapped_column(Text)
    """From a closed vocabulary. ``citation-wrong`` is why the vocabulary exists: it is the
    only detector this project has for citation misattribution, which verification cannot
    catch because catching it means deciding entailment."""

    feedback_comment: Mapped[str | None] = mapped_column(Text)
    feedback_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    query_log_id: Mapped[str | None] = mapped_column(
        ForeignKey("query_logs.id", ondelete="SET NULL")
    )
    """The retrieval run behind this answer, when there was one.

    ``SET NULL`` rather than ``CASCADE``: telemetry aging out must not delete the answer a
    person rated. Feedback that cannot name what produced the answer is a mood rather than a
    datum, so this is worth keeping — but the answer outlives the telemetry.
    """

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="role_is_known"),
        CheckConstraint(
            "feedback IS NULL OR feedback IN ('positive', 'negative')",
            name="feedback_is_known",
        ),
        CheckConstraint(
            "feedback_reason IS NULL OR feedback_reason IN "
            "('wrong', 'incomplete', 'citation-wrong', 'too-slow', 'other')",
            name="feedback_reason_is_known",
        ),
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
        Index(
            "ix_messages_feedback",
            "feedback",
            sqlite_where=text("feedback IS NOT NULL"),
        ),
    )


# --- security and telemetry --------------------------------------------------------------


class ApiKey(Base):
    """A hashed, scoped, expiring API key. The only key store there is."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    """Unique, which already creates the index. The prior art additionally declared an index
    on this column — a second copy of the same B-tree, maintained on every write."""

    key_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="member")
    scopes: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=list)
    allowed_ips: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=list)
    rate_limit: Mapped[int | None] = mapped_column(Integer)
    expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_used_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'member', 'viewer')", name="role_is_known"),
        Index("ix_api_keys_workspace_id", "workspace_id"),
    )


class AuditLog(Base):
    """Security-relevant events.

    Deliberately carries **no foreign keys**. An audit log that cascades away when the thing
    it audits is deleted is not an audit log.
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str | None] = mapped_column(Text)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[JsonValue] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_audit_logs_event_type_created_at", "event_type", "created_at"),
        Index("ix_audit_logs_workspace_id_created_at", "workspace_id", "created_at"),
    )


class QueryLog(Base):
    """Retrieval telemetry.

    ``workspace_id`` cascades, unlike :class:`AuditLog`: query text is user content scoped to
    a workspace, and retaining it past the workspace's deletion is a data-retention problem
    rather than a feature.
    """

    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    profile: Mapped[str | None] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[JsonValue] = mapped_column(JSON)
    reranked_chunk_ids: Mapped[JsonValue] = mapped_column(JSON)
    retrieval_score_avg: Mapped[float | None] = mapped_column(Float)
    rerank_score_avg: Mapped[float | None] = mapped_column(Float)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    response_time_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    # There is deliberately no `feedback` column here. A user rates an *answer*, an answer is
    # a message, and `messages.feedback` is where it lives. Keeping a second copy on the
    # retrieval row would be two homes for one fact, and the retrieval row is the one that
    # does not exist for a directly-routed answer or for a generation that failed.

    __table_args__ = (Index("ix_query_logs_workspace_id_created_at", "workspace_id", "created_at"),)


class Plugin(Base):
    """The installed-plugin registry.

    In the database rather than a JSON file beside it, so plugin state sits inside the same
    transactional and backup boundary as everything else. Carries no ``permissions`` column:
    an unenforced guarantee is worse than an absent one.
    """

    __tablename__ = "plugins"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[JsonValue] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    installed_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)


# --- index identity and sweeps -----------------------------------------------------------


class IndexState(Base):
    """One row per workspace describing what its derived indexes were built with.

    The fingerprints are canonical bytes in a ``TEXT`` column, not a JSON mapping. They are
    compared for byte equality, and a JSON column round-trips through a serializer that does
    not sort keys — so the stored bytes would depend on the insertion order of whatever
    mapping was passed in.
    """

    __tablename__ = "index_state"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    vector_namespace: Mapped[str] = mapped_column(
        Text, nullable=False, default="workspace", server_default="workspace"
    )
    """Physical layout selector.

    ``legacy`` keeps an upgraded workspace on the historical shared ``vectors/`` root until
    its first confirmed reset. ``workspace`` uses an opaque workspace-qualified child
    directory. The compatibility marker lets an upgrade avoid a multi-gigabyte eager copy
    while allowing reset and every fresh workspace to have independent fingerprints.
    """
    vector_table: Mapped[str | None] = mapped_column(Text)
    """A pointer, not a constant. It names a legacy table or a generation directory; re-embed
    moves it in one transaction, so a crash mid-rebuild leaves the old index live."""

    embed_fingerprint: Mapped[str | None] = mapped_column(Text)
    vector_inventory_digest: Mapped[str | None] = mapped_column(Text)
    chunk_fingerprint: Mapped[str | None] = mapped_column(Text)
    fts_tokenizer: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        CheckConstraint(
            "vector_namespace IN ('legacy', 'workspace')",
            name="vector_namespace_is_known",
        ),
    )


class CorpusRevision(Base):
    """Workspace-local revision moved by triggers on authoritative corpus mutations."""

    __tablename__ = "corpus_revision"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (CheckConstraint("revision >= 0", name="revision_is_not_negative"),)


class ReembedRunRecord(Base):
    """Durable re-embedding checkpoint and its current fenced lease."""

    __tablename__ = "reembed_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    commitment_json: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    checkpoint_json: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(Text)
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_expires_at: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        CheckConstraint("revision >= 0", name="revision_is_not_negative"),
        CheckConstraint("lease_generation >= 0", name="lease_generation_is_not_negative"),
    )


class ReembedCorpusSnapshot(Base):
    """A complete durable copy of the local corpus rows a rebuild is bound to."""

    __tablename__ = "reembed_corpus_snapshots"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    revision: Mapped[str] = mapped_column(Text, nullable=False)
    live_json: Mapped[str] = mapped_column(Text, nullable=False)
    complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inventory_digest: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_inventory_digest: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)


class ReembedSnapshotDocument(Base):
    __tablename__ = "reembed_snapshot_documents"

    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "snapshot_id"],
            ["reembed_corpus_snapshots.workspace_id", "reembed_corpus_snapshots.id"],
            ondelete="CASCADE",
        ),
        WITHOUT_ROWID,
    )


class ReembedSnapshotChunk(Base):
    __tablename__ = "reembed_snapshot_chunks"

    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "snapshot_id"],
            ["reembed_corpus_snapshots.workspace_id", "reembed_corpus_snapshots.id"],
            ondelete="CASCADE",
        ),
        WITHOUT_ROWID,
    )


class ReembedShadowGeneration(Base):
    """Immutable identity of one named Lance shadow generation."""

    __tablename__ = "reembed_shadow_generations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint_json: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    inventory_digest: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False, default="building")
    seal_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["reembed_runs.workspace_id", "reembed_runs.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("workspace_id", "run_id"),
    )


class ReembedPublicationReceipt(Base):
    """The immutable result of a run's single publication decision."""

    __tablename__ = "reembed_publication_receipts"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    receipt_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["reembed_runs.workspace_id", "reembed_runs.id"],
            ondelete="CASCADE",
        ),
    )


class DerivedGeneration(Base):
    """Durable control row for one unpublished or published corpus replacement."""

    __tablename__ = "derived_generations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_run_id: Mapped[str] = mapped_column(
        ForeignKey("acquisition_runs.id", ondelete="RESTRICT"), nullable=False
    )
    snapshot_membership_hash: Mapped[str] = mapped_column(Text, nullable=False)
    expected_item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    target_digest: Mapped[str] = mapped_column(Text, nullable=False)
    publication_identity_digest: Mapped[str] = mapped_column(Text, nullable=False)
    target: Mapped[JsonValue] = mapped_column(JSON, nullable=False)
    state: Mapped[RebuildState] = mapped_column(
        _rebuild_state_enum(), nullable=False, default=RebuildState.PLANNED
    )
    next_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    documents_built: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_built: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vectors_reused: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vectors_embedded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vector_publication_id: Mapped[str | None] = mapped_column(Text)
    expected_vector_table: Mapped[str | None] = mapped_column(Text)
    expected_vector_inventory_digest: Mapped[str | None] = mapped_column(Text)
    published_vector_inventory_digest: Mapped[str | None] = mapped_column(Text)
    evidence_inventory_digest: Mapped[str | None] = mapped_column(Text)
    evidence_verification_digest: Mapped[str | None] = mapped_column(Text)
    evidence_verification_lease_generation: Mapped[int | None] = mapped_column(Integer)
    evidence_verified_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    # Distinct from `lease_expires_at`/`updated_at`: only a durable replay or validation page
    # commit moves this, so a heartbeat renewing a stalled worker's lease cannot masquerade as
    # useful progress. See `RebuildCheckpoint.last_progress_at`.
    last_progress_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    replay_lease_generation: Mapped[int | None] = mapped_column(Integer)
    replay_checkpoint_sequence: Mapped[int | None] = mapped_column(Integer)
    replayed_vector_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_lease_generation: Mapped[int | None] = mapped_column(Integer)
    validation_checkpoint_sequence: Mapped[int | None] = mapped_column(Integer)
    validated_vector_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fence_generation: Mapped[int | None] = mapped_column(Integer)
    lease_owner: Mapped[str | None] = mapped_column(Text)
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    diagnostic_code: Mapped[str | None] = mapped_column(Text)
    diagnostic_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Only the current bounded storage-failure diagnostic is retained.  Raw driver details and
    # an unbounded event history belong in neither a generation row nor a public status read.
    storage_diagnostic: Mapped[JsonValue | None] = mapped_column(JSON)
    published_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "snapshot_run_id",
            "snapshot_membership_hash",
            "target_digest",
            "publication_identity_digest",
            name="uq_derived_generation_plan",
        ),
        Index("ix_derived_generations_workspace_state", "workspace_id", "state"),
        Index(
            "uq_derived_generations_workspace_fence",
            "workspace_id",
            "fence_generation",
            unique=True,
            sqlite_where=text("fence_generation IS NOT NULL"),
        ),
        CheckConstraint(
            "expected_item_count >= 0 AND next_sequence >= 0 AND documents_built >= 0 "
            "AND chunks_built >= 0 AND vectors_reused >= 0 AND vectors_embedded >= 0 "
            "AND lease_generation >= 0 AND diagnostic_count >= 0 "
            "AND (fence_generation IS NULL OR fence_generation >= 1) "
            "AND (replay_lease_generation IS NULL OR replay_lease_generation >= 1) "
            "AND (replay_checkpoint_sequence IS NULL OR replay_checkpoint_sequence >= 0) "
            "AND replayed_vector_count >= 0 "
            "AND (validation_lease_generation IS NULL OR validation_lease_generation >= 1) "
            "AND (validation_checkpoint_sequence IS NULL OR validation_checkpoint_sequence >= 0) "
            "AND validated_vector_count >= 0",
            name="derived_generation_counts_are_not_negative",
        ),
        CheckConstraint(
            "(state = 'published') = (published_at IS NOT NULL)",
            name="derived_generation_publication_timestamp_matches_state",
        ),
    )


class DerivedGenerationSnapshot(Base):
    """One promoted source scope bound into a workspace replacement generation."""

    __tablename__ = "derived_generation_snapshots"

    generation_id: Mapped[str] = mapped_column(
        ForeignKey("derived_generations.id", ondelete="CASCADE"), primary_key=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("acquisition_runs.id", ondelete="RESTRICT"), nullable=False
    )
    connector_name: Mapped[str] = mapped_column(Text, nullable=False)
    scope_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    membership_hash: Mapped[str] = mapped_column(Text, nullable=False)
    expected_item_count: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("generation_id", "run_id", name="uq_generation_snapshot_run"),
        CheckConstraint(
            "ordinal >= 0 AND expected_item_count >= 0",
            name="derived_generation_snapshot_counts_are_not_negative",
        ),
    )


class DerivedGenerationItem(Base):
    """One deterministic document replacement staged beside the live corpus."""

    __tablename__ = "derived_generation_items"

    generation_id: Mapped[str] = mapped_column(
        ForeignKey("derived_generations.id", ondelete="CASCADE"), primary_key=True
    )
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload_digest: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[JsonValue] = mapped_column(JSON, nullable=False)
    temporary_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index(
            "uq_derived_generation_items_document",
            "generation_id",
            "document_id",
            unique=True,
        ),
        CheckConstraint(
            "sequence >= 0 AND temporary_bytes >= 0",
            name="derived_generation_item_counts_are_not_negative",
        ),
        WITHOUT_ROWID,
    )


class VectorTombstone(Base):
    """A chunk id deleted from SQLite whose vector has not yet been swept.

    Not an optimization. Sweeping by comparing every id in the vector store against
    ``chunks`` races with concurrent ingest — an id written after the scan began looks like an
    orphan — and the sweep would delete live vectors. A tombstone list only ever names things
    that were deleted, so it cannot.
    """

    __tablename__ = "vector_tombstones"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(Text)
    """Owner of this exact physical row, or ``NULL`` for an unattributable legacy tombstone."""
    vector_namespace: Mapped[str | None] = mapped_column(Text)
    vector_table: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_vector_tombstones_workspace_deleted", "workspace_id", "deleted_at"),
        WITHOUT_ROWID,
    )


ALL_TABLES = tuple(Base.metadata.sorted_tables)
"""Every table, for migration and diagnostic code that needs to enumerate them."""

FTS_TOKENIZER = "porter unicode61 remove_diacritics 2"
"""Fixed at table creation, so it is recorded in :class:`IndexState` rather than hardcoded at
the call site. Porter is English-only; on another corpus ``unicode61`` alone is the better
choice, and that has to be a configuration decision rather than a silent assumption."""


__all__ = [
    "ALL_TABLES",
    "FTS_TOKENIZER",
    "NAMING_CONVENTION",
    "AcquisitionRecord",
    "AcquisitionRun",
    "ApiKey",
    "AuditLog",
    "Base",
    "Blob",
    "Chunk",
    "ChunkRelation",
    "Collection",
    "CollectionDocument",
    "Connector",
    "Conversation",
    "CorpusRevision",
    "DerivedGeneration",
    "DerivedGenerationItem",
    "DerivedGenerationSnapshot",
    "Document",
    "DocumentTag",
    "DocumentVersion",
    "GlossaryAlias",
    "GlossaryEntry",
    "IndexState",
    "Message",
    "Plugin",
    "QueryLog",
    "ReconciliationCandidate",
    "ReconciliationInventoryItem",
    "ReconciliationRun",
    "ReembedCorpusSnapshot",
    "ReembedPublicationReceipt",
    "ReembedRunRecord",
    "ReembedShadowGeneration",
    "ReembedSnapshotChunk",
    "ReembedSnapshotDocument",
    "Tag",
    "VectorTombstone",
    "Workspace",
    "WorkspaceMember",
]
