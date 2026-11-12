"""indexes on foreign-key child columns, and one for the rebuild capacity sum

SQLite enforces a foreign key on a parent ``DELETE`` by searching the child table for the key,
and it can only avoid a full scan with an index whose **leftmost** column is the child column.
``foreign_keys=ON`` is set on every connection, so these searches run on every deletion. Three
child columns had no such index, because each was the second or later column of every index on
its table:

- ``source_dependencies.parent_document_id`` — scanned once for every deleted document:
  retirement, purge, rebuild replacement, workspace reset.
- ``collection_documents.document_id`` — the same, and the column ``collections_for`` filters on.
- ``acquisition_records.blob_ref`` — scanned once for every blob garbage collection deletes,
  across the whole acquisition journal.
- ``reconciliation_candidates.document_id`` — scanned once for every deleted document.

Six smaller ones close the same gap where the parent is rarely deleted — a tag, an API key, a
pruned query log, a workspace's conversations, a rebuild's snapshot runs — so that the rule
holds for every foreign key and a test can say so rather than keeping a list of exceptions.

``derived_generation_items (generation_id, temporary_bytes)`` covers the capacity sum a rebuild
takes before and during each staging batch. The table is ``WITHOUT ROWID`` and stores
``temporary_bytes`` after the replacement payload, so without the index the sum walks every
staged byte on every batch, which makes a rebuild quadratic in what it stages.

Indexes only, so nothing is rebuilt, no row changes, and the downgrade drops them.

Revision ID: f4b1d8a27c63
Revises: e7a3c5f19d42
Created: 2026-11-12 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f4b1d8a27c63"
down_revision: str | None = "e7a3c5f19d42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_source_dependencies_parent_document_id",
        "source_dependencies",
        ["parent_document_id"],
    )
    op.create_index(
        "ix_collection_documents_document_id",
        "collection_documents",
        ["document_id"],
    )
    op.create_index(
        "ix_acquisition_records_blob_ref",
        "acquisition_records",
        ["blob_ref"],
    )
    op.create_index(
        "ix_derived_generation_items_generation_temporary_bytes",
        "derived_generation_items",
        ["generation_id", "temporary_bytes"],
    )
    op.create_index(
        "ix_reconciliation_candidates_document_id", "reconciliation_candidates", ["document_id"]
    )
    op.create_index("ix_document_tags_tag_id", "document_tags", ["tag_id"])
    op.create_index("ix_conversations_workspace_id", "conversations", ["workspace_id"])
    op.create_index("ix_messages_query_log_id", "messages", ["query_log_id"])
    op.create_index(
        "ix_derived_generations_snapshot_run_id", "derived_generations", ["snapshot_run_id"]
    )
    op.create_index(
        "ix_derived_generation_snapshots_run_id", "derived_generation_snapshots", ["run_id"]
    )
    op.create_index("ix_auth_sessions_key_id", "auth_sessions", ["key_id"])


def downgrade() -> None:
    op.drop_index("ix_auth_sessions_key_id", table_name="auth_sessions")
    op.drop_index(
        "ix_derived_generation_snapshots_run_id", table_name="derived_generation_snapshots"
    )
    op.drop_index("ix_derived_generations_snapshot_run_id", table_name="derived_generations")
    op.drop_index("ix_messages_query_log_id", table_name="messages")
    op.drop_index("ix_conversations_workspace_id", table_name="conversations")
    op.drop_index("ix_document_tags_tag_id", table_name="document_tags")
    op.drop_index(
        "ix_reconciliation_candidates_document_id", table_name="reconciliation_candidates"
    )
    op.drop_index(
        "ix_derived_generation_items_generation_temporary_bytes",
        table_name="derived_generation_items",
    )
    op.drop_index("ix_acquisition_records_blob_ref", table_name="acquisition_records")
    op.drop_index("ix_collection_documents_document_id", table_name="collection_documents")
    op.drop_index("ix_source_dependencies_parent_document_id", table_name="source_dependencies")
