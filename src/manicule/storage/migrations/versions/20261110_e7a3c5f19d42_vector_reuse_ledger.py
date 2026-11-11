"""a ledger of vectors copied between workspaces instead of embedded

``docs/storage.md`` §6.2.6: a workspace ingesting an embedding input another workspace already
holds a readable vector for copies that vector rather than calling the model. Callers scoped
to a workspace are never told it happened; the count of copied vectors is kept here, one row
per receiving workspace, for the operator.

A new table and nothing else, so no existing row is touched and the downgrade is a drop.

Revision ID: e7a3c5f19d42
Revises: d2f5b8e31a64
Created: 2026-11-10 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "e7a3c5f19d42"
down_revision: str | None = "d2f5b8e31a64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vector_reuse_ledger",
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("vectors", sa.Integer(), nullable=False),
        sa.Column("updated_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "vectors >= 0", name=op.f("ck_vector_reuse_ledger_vectors_is_not_negative")
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_vector_reuse_ledger_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("workspace_id", name=op.f("pk_vector_reuse_ledger")),
    )


def downgrade() -> None:
    op.drop_table("vector_reuse_ledger")
