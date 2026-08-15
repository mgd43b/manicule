"""inventory durable acquisition staging markers

Revision ID: e83a21f96c40
Revises: b7e4d921ac60
Create Date: 2026-08-15
"""

from __future__ import annotations

import hashlib
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
    op.add_column("acquisition_records", sa.Column("marker_name", sa.Text(), nullable=True))
    connection = op.get_bind()
    cursor = ""
    while True:
        records = connection.execute(
            sa.text(
                "SELECT id, run_id, source_id FROM acquisition_records "
                "WHERE id > :cursor ORDER BY id LIMIT 1000"
            ),
            {"cursor": cursor},
        ).fetchall()
        if not records:
            break
        for record_id, run_id, source_id in records:
            marker_name = hashlib.blake2b(
                f"{run_id}\0{source_id}".encode(), digest_size=20
            ).hexdigest()
            connection.execute(
                sa.text("UPDATE acquisition_records SET marker_name = :marker WHERE id = :id"),
                {"marker": marker_name, "id": record_id},
            )
        cursor = records[-1][0]
    op.create_index(
        "ix_acquisition_records_marker_name",
        "acquisition_records",
        ["marker_name"],
        unique=True,
    )
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
    op.drop_index("ix_acquisition_records_marker_name", table_name="acquisition_records")
    op.drop_column("acquisition_records", "marker_name")
