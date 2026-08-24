"""index live documents for re-embedding snapshots

Revision ID: 7b81e43c0f6a
Revises: a4f7c2d9e381
Create Date: 2026-08-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7b81e43c0f6a"
down_revision: str | Sequence[str] | None = "a4f7c2d9e381"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_workspace_live_id",
        "documents",
        ["workspace_id", "id"],
        unique=False,
        sqlite_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_documents_workspace_live_id", table_name="documents")
