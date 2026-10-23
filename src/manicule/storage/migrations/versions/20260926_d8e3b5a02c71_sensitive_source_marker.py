"""a durable record that a source has been synced by a sensitive connector

``connectors`` gains ``sensitive_synced_at``: when a connector whose registered type is
*sensitive* (mail, today) first began syncing this source. It is a column of its own, beside the
watermark it describes, and not a key in ``connectors.metadata``, because that column is a
diagnostic that every run overwrites in part and a security refusal must not rest on a field
whose neighbors are rewritten routinely.

**It is not backfilled.** No release before this one could register a sensitive connector, so no
existing row can be a source of one. ``NULL`` means "never synced by a sensitive connector".

``documents`` gains ``ix_documents_workspace_id_source``. The refusal asks whether *any* row, in
any state, still carries a marked source's name, and ``uq_documents_identity`` — the only index
on ``source`` — is partial on ``deleted_at IS NULL``, so a question about soft-deleted and purged
rows would otherwise scan the table at every server start and every ``doctor``.

**Plain ``ADD COLUMN`` and ``CREATE INDEX``, deliberately not a batch rebuild**, for the reason
``a3f7c1d8e520`` sets out: a SQLite batch alteration of ``documents`` drops the table, which
fires every ``ON DELETE CASCADE`` pointing at it. The downgrade drops the index before the
column for the reason that revision gives.

Revision ID: d8e3b5a02c71
Revises: f2b7c4e9a1d3
Created: 2026-09-26 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "d8e3b5a02c71"
down_revision: str | None = "f2b7c4e9a1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "connectors",
        sa.Column("sensitive_synced_at", manicule.storage.types.UtcDateTime(), nullable=True),
    )
    op.create_index(
        "ix_documents_workspace_id_source", "documents", ["workspace_id", "source"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_documents_workspace_id_source", table_name="documents")
    op.drop_column("connectors", "sensitive_synced_at")
