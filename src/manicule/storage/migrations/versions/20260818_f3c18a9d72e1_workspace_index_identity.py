"""workspace-scoped derived index identity

Revision ID: f3c18a9d72e1
Revises: e6a2c91f04bd
Created: 2026-08-18 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from manicule.storage.types import UtcDateTime

revision: str = "f3c18a9d72e1"
down_revision: str | Sequence[str] | None = "e6a2c91f04bd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _drop_revision_triggers() -> None:
    for table in ("documents", "chunks"):
        for event in ("insert", "update", "delete"):
            op.execute(f"DROP TRIGGER IF EXISTS {table}_reembed_revision_{event}")


def _create_workspace_revision_triggers() -> None:
    for event, reference in (("INSERT", "new"), ("UPDATE", "new"), ("DELETE", "old")):
        document_revision_scope = (
            "workspace_id IN (old.workspace_id, new.workspace_id)"
            if event == "UPDATE"
            else f"workspace_id = {reference}.workspace_id"
        )
        chunk_revision_scope = (
            "workspace_id IN (SELECT workspace_id FROM documents "
            "WHERE id IN (old.document_id, new.document_id))"
            if event == "UPDATE"
            else "workspace_id = (SELECT workspace_id FROM documents "  # noqa: S608
            f"WHERE id = {reference}.document_id)"
        )
        op.execute(
            f"CREATE TRIGGER documents_reembed_revision_{event.lower()} "  # noqa: S608
            f"AFTER {event} ON documents BEGIN "
            "UPDATE corpus_revision SET revision = revision + 1 WHERE "
            f"{document_revision_scope}; "
            f"UPDATE index_state SET vector_inventory_digest = NULL "
            f"WHERE {document_revision_scope}; END"
        )
        op.execute(
            f"CREATE TRIGGER chunks_reembed_revision_{event.lower()} "  # noqa: S608
            f"AFTER {event} ON chunks BEGIN "
            "UPDATE corpus_revision SET revision = revision + 1 WHERE "
            f"{chunk_revision_scope}; "
            "UPDATE index_state SET vector_inventory_digest = NULL WHERE "
            f"{chunk_revision_scope}; END"
        )


def _create_legacy_revision_triggers() -> None:
    for table in ("documents", "chunks"):
        for event in ("INSERT", "UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER {table}_reembed_revision_{event.lower()} "  # noqa: S608
                f"AFTER {event} ON {table} BEGIN "
                "UPDATE corpus_revision SET revision = revision + 1 WHERE id = 1; "
                "UPDATE index_state SET vector_inventory_digest = NULL WHERE id = 1; END"
            )


def upgrade() -> None:
    """Copy the historical singleton identity to each workspace without moving vectors."""
    _drop_revision_triggers()
    op.add_column(
        "workspaces",
        sa.Column("derived_reset_epoch", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_table(
        "workspace_corpus_revision",
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "revision >= 0", name=op.f("ck_corpus_revision_revision_is_not_negative")
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_corpus_revision_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("workspace_id", name=op.f("pk_corpus_revision")),
    )
    op.execute(
        "INSERT INTO workspace_corpus_revision (workspace_id, revision) "
        "SELECT id, COALESCE((SELECT revision FROM corpus_revision WHERE id = 1), 0) "
        "FROM workspaces"
    )
    op.drop_table("corpus_revision")
    op.rename_table("workspace_corpus_revision", "corpus_revision")
    op.execute(
        "CREATE TRIGGER workspaces_corpus_revision_insert AFTER INSERT ON workspaces BEGIN "
        "INSERT OR IGNORE INTO corpus_revision (workspace_id, revision) VALUES (new.id, 0); END"
    )
    op.create_table(
        "workspace_index_state",
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("vector_namespace", sa.Text(), server_default="workspace", nullable=False),
        sa.Column("vector_table", sa.Text(), nullable=True),
        sa.Column("embed_fingerprint", sa.Text(), nullable=True),
        sa.Column("vector_inventory_digest", sa.Text(), nullable=True),
        sa.Column("chunk_fingerprint", sa.Text(), nullable=True),
        sa.Column("fts_tokenizer", sa.Text(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.Column("updated_at", UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "vector_namespace IN ('legacy', 'workspace')",
            name=op.f("ck_index_state_vector_namespace_is_known"),
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id", name=op.f("pk_workspace_index_state")),
    )
    op.execute(
        """
        INSERT INTO workspace_index_state (
            workspace_id, vector_namespace, vector_table, embed_fingerprint,
            vector_inventory_digest, chunk_fingerprint, fts_tokenizer, created_at, updated_at
        )
        SELECT w.id, 'legacy', s.vector_table, s.embed_fingerprint,
               s.vector_inventory_digest, s.chunk_fingerprint, s.fts_tokenizer,
               s.created_at, s.updated_at
          FROM workspaces AS w
          CROSS JOIN index_state AS s
         WHERE s.id = 1
        """
    )
    op.drop_table("index_state")
    op.rename_table("workspace_index_state", "index_state")
    _create_workspace_revision_triggers()
    op.add_column("vector_tombstones", sa.Column("workspace_id", sa.Text(), nullable=True))
    op.add_column("vector_tombstones", sa.Column("vector_namespace", sa.Text(), nullable=True))
    op.add_column("vector_tombstones", sa.Column("vector_table", sa.Text(), nullable=True))
    op.create_index(
        "ix_vector_tombstones_workspace_deleted",
        "vector_tombstones",
        ["workspace_id", "deleted_at"],
    )
    # A pre-publication staged tombstone can still have a matching live chunk, in which case
    # its owner and immutable legacy binding are knowable.  Deleted rows are intentionally left
    # NULL: guessing would let one workspace sweep another's shared-root vector.  Those
    # unattributable rows remain protected until the last legacy consumer removes the root.
    op.execute(
        """
        UPDATE vector_tombstones
           SET workspace_id = (
                   SELECT d.workspace_id
                     FROM chunks AS c
                     JOIN documents AS d ON d.id = c.document_id
                    WHERE c.vector_id = vector_tombstones.chunk_id
                    LIMIT 1
               ),
               vector_namespace = 'legacy',
               vector_table = (
                   SELECT s.vector_table
                     FROM chunks AS c
                     JOIN documents AS d ON d.id = c.document_id
                     JOIN index_state AS s ON s.workspace_id = d.workspace_id
                    WHERE c.vector_id = vector_tombstones.chunk_id
                    LIMIT 1
               )
         WHERE EXISTS (
                   SELECT 1
                     FROM chunks AS c
                     JOIN documents AS d ON d.id = c.document_id
                    WHERE c.vector_id = vector_tombstones.chunk_id
               )
        """
    )
    op.execute("DROP TRIGGER IF EXISTS chunks_ad")
    op.execute(
        """
        CREATE TRIGGER documents_vector_tombstones_bd BEFORE DELETE ON documents BEGIN
            INSERT OR IGNORE INTO vector_tombstones(
                chunk_id, workspace_id, vector_namespace, vector_table, deleted_at
            )
            SELECT c.vector_id, old.workspace_id, s.vector_namespace, s.vector_table,
                   datetime('now')
              FROM chunks AS c
              LEFT JOIN index_state AS s ON s.workspace_id = old.workspace_id
             WHERE c.document_id = old.id;
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
            VALUES ('delete', old.seq, old.text, old.heading_text);
            INSERT OR IGNORE INTO vector_tombstones(
                chunk_id, workspace_id, vector_namespace, vector_table, deleted_at
            )
            SELECT old.vector_id, d.workspace_id, s.vector_namespace, s.vector_table,
                   datetime('now')
              FROM documents AS d
              LEFT JOIN index_state AS s ON s.workspace_id = d.workspace_id
             WHERE d.id = old.document_id;
        END
        """
    )


def downgrade() -> None:
    """Collapse only a still-compatible single identity back to the legacy singleton."""
    connection = op.get_bind()
    distinct = connection.execute(
        sa.text(
            "SELECT count(*) FROM (SELECT DISTINCT vector_table, embed_fingerprint, "
            "vector_inventory_digest, chunk_fingerprint, fts_tokenizer FROM index_state)"
        )
    ).scalar_one()
    nonlegacy = connection.execute(
        sa.text("SELECT count(*) FROM index_state WHERE vector_namespace <> 'legacy'")
    ).scalar_one()
    if distinct > 1 or nonlegacy:
        raise RuntimeError(
            "refusing to downgrade workspace index identity: workspaces have independent "
            "derived fingerprints or vector namespaces; reset them before retrying"
        )
    op.drop_column("workspaces", "derived_reset_epoch")
    _drop_revision_triggers()
    op.execute("DROP TRIGGER IF EXISTS workspaces_corpus_revision_insert")
    op.create_table(
        "legacy_corpus_revision",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.CheckConstraint("id = 1", name=op.f("ck_corpus_revision_is_a_singleton")),
        sa.CheckConstraint(
            "revision >= 0", name=op.f("ck_corpus_revision_revision_is_not_negative")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_corpus_revision")),
    )
    op.execute(
        "INSERT INTO legacy_corpus_revision (id, revision) "
        "SELECT 1, COALESCE(max(revision), 0) FROM corpus_revision"
    )
    op.drop_table("corpus_revision")
    op.rename_table("legacy_corpus_revision", "corpus_revision")
    op.execute("DROP TRIGGER IF EXISTS documents_vector_tombstones_bd")
    op.execute("DROP TRIGGER IF EXISTS chunks_ad")
    op.drop_index("ix_vector_tombstones_workspace_deleted", table_name="vector_tombstones")
    op.drop_column("vector_tombstones", "vector_table")
    op.drop_column("vector_tombstones", "vector_namespace")
    op.drop_column("vector_tombstones", "workspace_id")
    op.execute(
        """
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
            VALUES ('delete', old.seq, old.text, old.heading_text);
            INSERT OR IGNORE INTO vector_tombstones(chunk_id, deleted_at)
            VALUES (old.vector_id, datetime('now'));
        END
        """
    )
    op.create_table(
        "legacy_index_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vector_table", sa.Text(), nullable=True),
        sa.Column("embed_fingerprint", sa.Text(), nullable=True),
        sa.Column("vector_inventory_digest", sa.Text(), nullable=True),
        sa.Column("chunk_fingerprint", sa.Text(), nullable=True),
        sa.Column("fts_tokenizer", sa.Text(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.Column("updated_at", UtcDateTime(), nullable=False),
        sa.CheckConstraint("id = 1", name=op.f("ck_legacy_index_state_is_a_singleton")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_legacy_index_state")),
    )
    op.execute(
        """
        INSERT INTO legacy_index_state (
            id, vector_table, embed_fingerprint, vector_inventory_digest,
            chunk_fingerprint, fts_tokenizer, created_at, updated_at
        )
        SELECT 1, vector_table, embed_fingerprint, vector_inventory_digest,
               chunk_fingerprint, fts_tokenizer, created_at, updated_at
          FROM index_state
         ORDER BY workspace_id
         LIMIT 1
        """
    )
    op.drop_table("index_state")
    op.rename_table("legacy_index_state", "index_state")
    _create_legacy_revision_triggers()
