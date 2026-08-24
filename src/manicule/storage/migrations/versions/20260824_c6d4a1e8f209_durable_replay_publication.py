"""bind rebuild replay checkpoints to durable vector publications

Revision ID: c6d4a1e8f209
Revises: 7b81e43c0f6a
Create Date: 2026-08-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c6d4a1e8f209"
down_revision: str | Sequence[str] | None = "7b81e43c0f6a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("derived_generations") as batch:
        # Existing replay cursors intentionally remain unbound. They predate the physical
        # source/target proof and will perform one fresh replay before becoming resumable.
        batch.add_column(sa.Column("replay_source_publication_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("replay_target_publication_id", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("derived_generations") as batch:
        batch.drop_column("replay_target_publication_id")
        batch.drop_column("replay_source_publication_id")
