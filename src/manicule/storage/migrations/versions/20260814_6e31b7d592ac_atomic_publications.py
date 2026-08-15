"""active derived-state publication

Revision ID: 6e31b7d592ac
Revises: d4a90c7e15b3
Created: 2026-08-14 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from manicule.storage.fts import CREATE_TRIGGERS, DROP_TRIGGERS

revision: str = "6e31b7d592ac"
down_revision: str | None = "d4a90c7e15b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _require_legacy_safe_downgrade() -> None:
    """Refuse to erase the only pointer that distinguishes active vector generations."""
    connection = op.get_bind()
    generated_documents = connection.execute(
        sa.text("SELECT count(*) FROM documents WHERE publication_id <> 'legacy'")
    ).scalar_one()
    generated_chunks = connection.execute(
        sa.text("SELECT count(*) FROM chunks WHERE vector_id <> id")
    ).scalar_one()
    pending_cleanup = connection.execute(
        sa.text("SELECT count(*) FROM vector_tombstones")
    ).scalar_one()
    if generated_documents or generated_chunks or pending_cleanup:
        msg = (
            "refusing to downgrade atomic publications: this index contains generated "
            "publications or pending vector cleanup, and removing SQLite's publication "
            "pointers would let older code hydrate retired or staged Lance rows. Automatic "
            "projection is unsafe because Alembic owns only the SQLite transaction, not "
            "LanceDB. Rebuild into a clean data directory with the target release, or first "
            "purge the corpus and complete the vector tombstone sweep, then retry."
        )
        raise RuntimeError(msg)


def upgrade() -> None:
    """Give every existing relational/vector revision the shared legacy publication."""
    op.add_column(
        "documents",
        sa.Column("publication_id", sa.Text(), server_default="legacy", nullable=False),
    )
    op.add_column("chunks", sa.Column("vector_id", sa.Text(), nullable=True))
    op.execute("UPDATE chunks SET vector_id = id")
    for statement in DROP_TRIGGERS:
        op.execute(statement)
    with op.batch_alter_table("chunks") as batch:
        batch.alter_column("vector_id", existing_type=sa.Text(), nullable=False)
    for statement in CREATE_TRIGGERS:
        op.execute(statement)


def downgrade() -> None:
    """Remove the publication pointer; vector columns are migrated lazily by LanceDB."""
    _require_legacy_safe_downgrade()
    for statement in DROP_TRIGGERS:
        op.execute(statement)
    with op.batch_alter_table("chunks") as batch:
        batch.drop_column("vector_id")
    op.execute(CREATE_TRIGGERS[0])
    op.execute(
        """
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
            VALUES ('delete', old.seq, old.text, old.heading_text);
            INSERT OR IGNORE INTO vector_tombstones(chunk_id, deleted_at)
            VALUES (old.id, datetime('now'));
        END
        """
    )
    op.execute(CREATE_TRIGGERS[2])
    op.drop_column("documents", "publication_id")
