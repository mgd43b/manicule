"""persist aggregate adaptive-enumeration progress on acquisition runs

Revision ID: a7d40c3e91b5
Revises: f3c18a9d72e1
Created: 2026-08-19 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d40c3e91b5"
down_revision: str | Sequence[str] | None = "f3c18a9d72e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.add_column(sa.Column("enumeration_offset", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("enumeration_page_size", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column(
                "enumeration_timeout_retries",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "enumeration_page_size_reduced",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        # Nullable on purpose: NULL is "the connector made no empty-page claim", which every
        # existing run and every search-backed run must keep reading as.
        batch.add_column(sa.Column("enumeration_reached_empty_page", sa.Boolean(), nullable=True))
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.alter_column(
            "enumeration_timeout_retries",
            existing_type=sa.Integer(),
            existing_nullable=False,
            server_default=None,
        )
        batch.alter_column(
            "enumeration_page_size_reduced",
            existing_type=sa.Boolean(),
            existing_nullable=False,
            server_default=None,
        )


def downgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.drop_column("enumeration_reached_empty_page")
        batch.drop_column("enumeration_page_size_reduced")
        batch.drop_column("enumeration_timeout_retries")
        batch.drop_column("enumeration_page_size")
        batch.drop_column("enumeration_offset")
