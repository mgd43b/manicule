"""persist reverse source-content dependencies

Revision ID: ee6f1a2c3d4b
Revises: d4619bf4f012
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "ee6f1a2c3d4b"
down_revision: str | Sequence[str] | None = "d4619bf4f012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "source_dependencies",
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("parent_document_id", sa.Text(), nullable=False),
        sa.Column("target_source_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parent_document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id", "source", "parent_document_id", "target_source_id"),
        sqlite_with_rowid=False,
    )
    op.create_index(
        "ix_source_dependencies_target",
        "source_dependencies",
        ["workspace_id", "source", "target_source_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_source_dependencies_target", table_name="source_dependencies")
    op.drop_table("source_dependencies")
