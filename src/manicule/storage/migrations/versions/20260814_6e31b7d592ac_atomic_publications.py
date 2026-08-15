"""active derived-state publication

Revision ID: 6e31b7d592ac
Revises: d4a90c7e15b3
Created: 2026-08-14 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6e31b7d592ac"
down_revision: str | None = "d4a90c7e15b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Give every existing relational/vector revision the shared legacy publication."""
    op.add_column(
        "documents",
        sa.Column("publication_id", sa.Text(), server_default="legacy", nullable=False),
    )


def downgrade() -> None:
    """Remove the publication pointer; vector columns are migrated lazily by LanceDB."""
    op.drop_column("documents", "publication_id")
