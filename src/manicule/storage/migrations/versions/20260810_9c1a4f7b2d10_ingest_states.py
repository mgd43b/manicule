"""ingest states: in-flight document statuses, and middleware as a failure stage

The pipeline needs three statuses the initial schema has no room for — ``fetching``,
``parsing`` and ``embedding`` — because the recovery sweep selects the in-flight states **by
name**. An allowlist fails closed; the alternative formulation, "everything that is not
``indexed``", sweeps ``container`` and ``no_extractable_text`` too, and both are terminal with
zero chunks by design.

``failed_stage`` gains ``middleware`` for the same kind of reason. A hook that raises is a
plugin problem, and filing it under the stage it happened to bound would send an operator to
read a parser that worked perfectly.

Both columns are ``VARCHAR`` + ``CHECK``, so widening the value set is a table rebuild on
SQLite. That is what ``batch_alter_table`` does, and it is why every constraint in this schema
is named: batch mode has to *name* the constraint it drops, and SQLite's own names are not
stable.

**The table is described here rather than reflected**, via ``copy_from``. Reflection recovers
the columns and constraints but not the order in which SQLite wrote them, so a rebuild emits
the two foreign keys in whichever order the reflector returned — and the downgrade/upgrade
round trip then produces DDL that differs from the DDL it started with, byte for byte, while
being identical in meaning. That difference is exactly what the round-trip test compares, and
it should stay strict: a batch rebuild that silently drops a ``CHECK`` reflects as a table
with all the right columns.

Revision ID: 9c1a4f7b2d10
Revises: 2674752edf52
Created: 2026-08-10 14:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "9c1a4f7b2d10"
down_revision: str | None = "2674752edf52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_STATUSES = (
    "pending",
    "parsed",
    "indexed",
    "container",
    "no_extractable_text",
    "unsupported_media_type",
    "failed",
    "skipped",
    "deleted",
)
_NEW_STATUSES = (
    "pending",
    "fetching",
    "parsing",
    "embedding",
    "parsed",
    "indexed",
    "container",
    "no_extractable_text",
    "unsupported_media_type",
    "failed",
    "skipped",
    "deleted",
)

_OLD_STAGES = ("discover", "fetch", "parse", "chunk", "embed", "store")
_NEW_STAGES = (*_OLD_STAGES, "middleware")


def _status(values: Sequence[str]) -> sa.Enum:
    return sa.Enum(*values, name="document_status", native_enum=False, create_constraint=True)


def _stage(values: Sequence[str]) -> sa.Enum:
    return sa.Enum(*values, name="pipeline_stage", native_enum=False, create_constraint=True)


def _documents(statuses: Sequence[str], stages: Sequence[str]) -> sa.Table:
    """The ``documents`` table as it stands on one side of this revision."""
    metadata = sa.MetaData()
    table = sa.Table(
        "documents",
        metadata,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("version_token", sa.Text(), nullable=True),
        sa.Column("original_ref", sa.Text(), nullable=True),
        sa.Column("original_omitted_reason", sa.Text(), nullable=True),
        sa.Column("status", _status(statuses), nullable=False),
        sa.Column("status_detail", sa.Text(), nullable=True),
        sa.Column("failed_stage", _stage(stages), nullable=True),
        sa.Column("chunk_fp", sa.Text(), nullable=True),
        sa.Column("embed_fp", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("indexed_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("last_seen_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("deleted_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.ForeignKeyConstraint(
            ["original_ref"],
            ["blobs.hash"],
            name="fk_documents_original_ref_blobs",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_documents_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (failed_stage IS NOT NULL)",
            name="ck_documents_failed_stage_iff_failed",
        ),
    )
    sa.Index("ix_documents_chunk_fp", table.c.chunk_fp)
    sa.Index("ix_documents_content_hash", table.c.content_hash)
    sa.Index(
        "ix_documents_deleted_at",
        table.c.workspace_id,
        table.c.deleted_at,
        sqlite_where=sa.text("deleted_at IS NOT NULL"),
    )
    sa.Index("ix_documents_embed_fp", table.c.embed_fp)
    sa.Index("ix_documents_workspace_id_status", table.c.workspace_id, table.c.status)
    sa.Index("ix_documents_workspace_id_uri", table.c.workspace_id, table.c.uri)
    sa.Index(
        "uq_documents_identity",
        table.c.workspace_id,
        table.c.source,
        table.c.source_id,
        unique=True,
        sqlite_where=sa.text("deleted_at IS NULL"),
    )
    return table


def upgrade() -> None:
    with op.batch_alter_table(
        "documents", schema=None, copy_from=_documents(_OLD_STATUSES, _OLD_STAGES)
    ) as batch_op:
        batch_op.alter_column(
            "status", existing_type=_status(_OLD_STATUSES), type_=_status(_NEW_STATUSES)
        )
        batch_op.alter_column(
            "failed_stage", existing_type=_stage(_OLD_STAGES), type_=_stage(_NEW_STAGES)
        )


def downgrade() -> None:
    # Anything still in flight when the schema narrows becomes ``pending``, which is the same
    # answer the recovery sweep gives: a document that is not ``indexed`` is not served, so
    # requeueing it is cheap and losing the distinction costs nothing. A middleware failure
    # becomes a ``parse`` failure, which is the closest surviving truth.
    op.execute(
        sa.text(
            "UPDATE documents SET status = 'pending' "
            "WHERE status IN ('fetching', 'parsing', 'embedding')"
        )
    )
    op.execute(
        sa.text("UPDATE documents SET failed_stage = 'parse' WHERE failed_stage = 'middleware'")
    )
    with op.batch_alter_table(
        "documents", schema=None, copy_from=_documents(_NEW_STATUSES, _NEW_STAGES)
    ) as batch_op:
        batch_op.alter_column(
            "failed_stage", existing_type=_stage(_NEW_STAGES), type_=_stage(_OLD_STAGES)
        )
        batch_op.alter_column(
            "status", existing_type=_status(_NEW_STATUSES), type_=_status(_OLD_STATUSES)
        )
