"""index acquisition record marker names

Revision ID: d52f81a439bc
Revises: e83a21f96c40
Create Date: 2026-08-15
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "d52f81a439bc"
down_revision: str | Sequence[str] | None = "e83a21f96c40"
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


def downgrade() -> None:
    op.drop_index("ix_acquisition_records_marker_name", table_name="acquisition_records")
    op.drop_column("acquisition_records", "marker_name")
