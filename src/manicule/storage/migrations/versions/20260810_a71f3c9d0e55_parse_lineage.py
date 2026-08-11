"""per-document parse lineage

``documents`` gains ``parse_fp``: the canonical
:class:`~manicule.core.fingerprints.ParseFingerprint` of the parser run that produced this
document's stored text and anchors. Until now the schema recorded which chunker and which
embedder built a document and nothing about which *parser* did, so a ``pypdfium2`` bump
rewrote what the same bytes reduce to with no column able to say which generation a row
belonged to.

**It is not backfilled, and that is the decision this revision is really making.** Every
existing row keeps ``NULL``, which reads as "no recorded lineage" everywhere it is consulted:
``reindex --re-parse`` selects those documents, and the next sync re-parses them. Writing
today's versions into them instead would be one statement and would assert something nobody
knows — that text extracted months ago came out of the libraries installed now — which is the
class of quiet, plausible falsehood the fingerprints exist to prevent. The price is a one-time
re-parse of the corpus, which reads retained bytes and touches no network.

**A plain ``ADD COLUMN``, deliberately not a batch rebuild.** SQLite implements a batch
alteration as CREATE-temp, INSERT…SELECT, ``DROP TABLE``, RENAME — and with
``PRAGMA foreign_keys=ON``, which this project sets on every connection, that ``DROP``
performs an implicit ``DELETE FROM`` and fires every ``ON DELETE CASCADE`` pointing at
``documents``: chunks, versions, tags and collection membership, emptied while the migration
reports success. Adding a nullable column needs none of that, so it does none of it. The
downgrade drops the index before the column because SQLite refuses ``DROP COLUMN`` on an
indexed column — the failure would be at the moment a downgrade is being run under pressure,
which is the worst possible time to discover it.

Revision ID: a71f3c9d0e55
Revises: c3f81a5b6e42
Created: 2026-08-10 22:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a71f3c9d0e55"
down_revision: str | None = "c3f81a5b6e42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("parse_fp", sa.Text(), nullable=True))
    op.create_index("ix_documents_parse_fp", "documents", ["parse_fp"], unique=False)


def downgrade() -> None:
    # Index first: SQLite's ALTER TABLE DROP COLUMN refuses a column an index refers to.
    op.drop_index("ix_documents_parse_fp", table_name="documents")
    op.drop_column("documents", "parse_fp")
