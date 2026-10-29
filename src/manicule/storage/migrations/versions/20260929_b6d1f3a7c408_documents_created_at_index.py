"""an index for the repair selection's order

``select_documents`` orders by ``(created_at, id)`` within a workspace, and nothing indexed that,
so every page of a corpus-wide sweep sorted the whole filtered table to return one batch. A
keyset cursor (``after``) turns the page into a seek only if an index can be seeked in that
order; this is that index.

**Plain ``CREATE INDEX``, deliberately not a batch rebuild**, for the reason ``a3f7c1d8e520`` sets
out: the triggers on ``documents`` would be dropped by a table rebuild.

Revision ID: b6d1f3a7c408
Revises: a5c2e8f19b64
Created: 2026-09-29 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b6d1f3a7c408"
down_revision: str | None = "a5c2e8f19b64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_workspace_id_created_at_id",
        "documents",
        ["workspace_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_workspace_id_created_at_id", table_name="documents")
