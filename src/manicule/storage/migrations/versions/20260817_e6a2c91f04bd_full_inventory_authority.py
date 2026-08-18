"""persist effective full-inventory authority

Revision ID: e6a2c91f04bd
Revises: c4b8d1e7a902
Created: 2026-08-17 20:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e6a2c91f04bd"
down_revision: str | Sequence[str] | None = "c4b8d1e7a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.add_column(
            sa.Column(
                "full_inventory_authority",
                sa.Text(),
                nullable=False,
                server_default="",
            )
        )
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.alter_column(
            "full_inventory_authority",
            existing_type=sa.Text(),
            existing_nullable=False,
            server_default=None,
        )


def downgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.drop_column("full_inventory_authority")
