"""scoped glossary entries

Two new tables, ``glossary_entries`` and ``glossary_aliases``, holding the acronym definitions
ingest detects so that a query naming a term can find it by lookup instead of by similarity.

**Two ``CREATE TABLE``s and nothing else, deliberately.** No existing table is altered, so
this revision cannot trigger the failure that makes SQLite migrations dangerous here: a batch
alteration is CREATE-temp, INSERT…SELECT, ``DROP TABLE``, RENAME, and with
``PRAGMA foreign_keys=ON`` — which this project sets on every connection — that ``DROP``
performs an implicit ``DELETE FROM`` and fires every cascade pointing at the table. Creating a
table touches none of that. ``tests/test_storage_migrations.py`` runs every revision over a
populated database precisely so that a future revision here cannot quietly become one that
does.

**Nothing is backfilled, and there is nothing to backfill from.** Definitions are produced by
parsing text, so an existing corpus arrives with an empty glossary and gains one on its next
ingest or re-parse. The alternative — deriving entries from stored chunks inside a migration —
would put the detector's rules in two places and freeze one copy of them into a revision that
can never be changed.

**Both foreign keys cascade, and the ``chunks`` one is the load-bearing half.** An entry cites
the chunk it was read out of. ``replace_chunks`` deletes and rewrites a document's chunks on
every re-parse, so without the cascade an edited glossary would keep answering with the line it
used to have, citing a chunk id that no longer resolves — a definition that is wrong, confident
and unfalsifiable from the outside.

Revision ID: 5f1c8a34b7d9
Revises: a71f3c9d0e55
Created: 2026-08-12 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5f1c8a34b7d9"
down_revision: str | None = "a71f3c9d0e55"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "glossary_entries",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("acronym", sa.Text(), nullable=False),
        sa.Column("display", sa.Text(), nullable=False),
        sa.Column("expansion", sa.Text(), nullable=False),
        sa.Column("location", sa.Text(), nullable=False),
        sa.Column("form", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name=op.f("ck_glossary_entries_confidence_is_a_fraction"),
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_glossary_entries_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_glossary_entries_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_glossary_entries")),
    )
    op.create_index("ix_glossary_entries_acronym", "glossary_entries", ["acronym"], unique=False)
    op.create_index("ix_glossary_entries_chunk_id", "glossary_entries", ["chunk_id"], unique=False)
    op.create_index(
        "ix_glossary_entries_document_id", "glossary_entries", ["document_id"], unique=False
    )
    op.create_table(
        "glossary_aliases",
        sa.Column("entry_id", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["entry_id"],
            ["glossary_entries.id"],
            name=op.f("fk_glossary_aliases_entry_id_glossary_entries"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("entry_id", "key", name=op.f("pk_glossary_aliases")),
        sqlite_with_rowid=False,
    )
    op.create_index("ix_glossary_aliases_key", "glossary_aliases", ["key"], unique=False)


def downgrade() -> None:
    # Children first. ``glossary_aliases`` points at ``glossary_entries``, and dropping the
    # parent while the child exists leaves a table whose foreign key refers to nothing — which
    # SQLite tolerates until the next write and then does not.
    op.drop_index("ix_glossary_aliases_key", table_name="glossary_aliases")
    op.drop_table("glossary_aliases")
    op.drop_index("ix_glossary_entries_document_id", table_name="glossary_entries")
    op.drop_index("ix_glossary_entries_chunk_id", table_name="glossary_entries")
    op.drop_index("ix_glossary_entries_acronym", table_name="glossary_entries")
    op.drop_table("glossary_entries")
