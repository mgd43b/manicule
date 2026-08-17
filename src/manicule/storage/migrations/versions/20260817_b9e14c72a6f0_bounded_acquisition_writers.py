"""bound acquisition writer work and persist exact blob backlog totals

Revision ID: b9e14c72a6f0
Revises: d31f8a6c20e7
Create Date: 2026-08-17
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "b9e14c72a6f0"
down_revision: str | Sequence[str] | None = "d31f8a6c20e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_acquisition_records_run_blob_state",
        "acquisition_records",
        ["run_id", "blob_ref", "state"],
        unique=False,
    )
    op.create_table(
        "acquisition_blob_backlog",
        sa.Column("blob_ref", sa.Text(), nullable=False),
        sa.Column("reference_count", sa.Integer(), nullable=False),
        sa.Column("stored_bytes", sa.Integer(), nullable=False),
        sa.CheckConstraint("reference_count > 0", name="acquisition_blob_backlog_has_references"),
        sa.CheckConstraint(
            "stored_bytes >= 0", name="acquisition_blob_backlog_bytes_are_not_negative"
        ),
        sa.ForeignKeyConstraint(["blob_ref"], ["blobs.hash"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("blob_ref"),
    )
    op.create_table(
        "acquisition_backlog_capacity",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("acquired_blob_bytes", sa.Integer(), nullable=False),
        sa.CheckConstraint("id = 1", name="acquisition_backlog_capacity_is_singleton"),
        sa.CheckConstraint(
            "acquired_blob_bytes >= 0", name="acquisition_backlog_capacity_is_not_negative"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        sa.text(
            "INSERT INTO acquisition_blob_backlog (blob_ref, reference_count, stored_bytes) "
            "SELECT ar.blob_ref, count(*), b.stored_bytes "
            "FROM acquisition_records ar "
            "JOIN acquisition_runs r ON r.id = ar.run_id "
            "JOIN blobs b ON b.hash = ar.blob_ref "
            "WHERE r.state != 'settled' AND ar.blob_ref IS NOT NULL "
            "AND ar.state IN ('discovered','acquiring','acquired','unchanged','indexing','retry') "
            "GROUP BY ar.blob_ref, b.stored_bytes"
        )
    )
    op.execute(
        sa.text(
            "INSERT INTO acquisition_backlog_capacity (id, acquired_blob_bytes) "
            "SELECT 1, coalesce(sum(stored_bytes), 0) FROM acquisition_blob_backlog"
        )
    )


def downgrade() -> None:
    op.drop_table("acquisition_backlog_capacity")
    op.drop_table("acquisition_blob_backlog")
    op.drop_index("ix_acquisition_records_run_blob_state", table_name="acquisition_records")
