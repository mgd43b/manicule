"""inventory durable acquisition staging markers

Revision ID: e83a21f96c40
Revises: b7e4d921ac60
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "e83a21f96c40"
down_revision: str | Sequence[str] | None = "b7e4d921ac60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "acquisition_markers",
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("blob_ref", sa.Text(), nullable=True),
        sa.Column("acquired_source", sa.JSON(), nullable=True),
        sa.Column("legacy", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", manicule.storage.types.UtcDateTime(), nullable=False
        ),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_index(
        "ix_acquisition_markers_run_id", "acquisition_markers", ["run_id"], unique=False
    )
    op.create_index(
        "ix_acquisition_markers_blob_ref", "acquisition_markers", ["blob_ref"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_acquisition_markers_blob_ref", table_name="acquisition_markers")
    op.drop_index("ix_acquisition_markers_run_id", table_name="acquisition_markers")
    op.drop_table("acquisition_markers")
