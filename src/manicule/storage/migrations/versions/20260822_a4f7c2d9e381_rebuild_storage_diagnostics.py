"""persist bounded rebuild-storage diagnostics

Revision ID: a4f7c2d9e381
Revises: ee6f1a2c3d4b
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4f7c2d9e381"
down_revision: str | Sequence[str] | None = "ee6f1a2c3d4b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One versioned, aggregate-only record per generation.  Keeping this a JSON object makes
    # future vocabulary additions explicit at the model boundary while avoiding an unbounded
    # failure-event table that could accidentally retain sensitive backend messages.
    with op.batch_alter_table("derived_generations") as batch:
        batch.add_column(sa.Column("storage_diagnostic", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("derived_generations") as batch:
        batch.drop_column("storage_diagnostic")
