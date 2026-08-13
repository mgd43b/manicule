"""per-document glossary lineage

``documents`` gains ``glossary_fp``: the canonical
:class:`~manicule.core.fingerprints.GlossaryFingerprint` of the detector run that decided this
document's stored glossary entries. Until now the schema recorded which parser, chunker and
embedder built a document and nothing about which *detector* read its definitions — so five
corrections to detection rules landed against a corpus that kept the entries the old rules
produced and reported itself current, because ``parse_fp`` is what selection compares and no
detector change moves it.

**It is not backfilled, and that is the decision this revision is making.** Every existing row
keeps ``NULL``, which reads as "these entries were never computed by anything this index can
name" — so ``document reindex --stale-glossary`` selects them and recomputes from stored chunks.
Writing the installed fingerprint into them instead would be one statement and would assert
that entries detected before this column existed came out of the rules installed now, which is
false for every corpus indexed before today and is the class of quiet, plausible falsehood the
fingerprints exist to prevent. The price is one sweep, and that sweep reads chunks: no
connector, no retained bytes, no parser, no embedder.

**``NULL`` is also not "no entries".** A document that genuinely states no definitions records a
fingerprint and no rows once it has been recomputed, which is what makes "the current detector
finds nothing here" distinguishable from "nobody has looked". Before that recomputation the two
are the same row, which is exactly why the first release treats every unstamped document as
stale rather than trusting it.

**A plain ``ADD COLUMN``, deliberately not a batch rebuild**, for the reason ``a71f3c9d0e55``
sets out at length: SQLite implements a batch alteration as CREATE-temp, INSERT…SELECT,
``DROP TABLE``, RENAME, and with ``PRAGMA foreign_keys=ON`` — which this project sets on every
connection — that ``DROP`` fires every ``ON DELETE CASCADE`` pointing at ``documents``, emptying
chunks, versions, tags, collection membership and now glossary entries while reporting success.
Adding a nullable column needs none of that. The downgrade drops the index before the column,
because SQLite refuses ``DROP COLUMN`` on an indexed column and the discovery would be made at
the moment somebody is downgrading under pressure.

Revision ID: d4a90c7e15b3
Revises: b2e6d0c94a17
Created: 2026-08-13 16:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4a90c7e15b3"
down_revision: str | None = "b2e6d0c94a17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("glossary_fp", sa.Text(), nullable=True))
    op.create_index("ix_documents_glossary_fp", "documents", ["glossary_fp"], unique=False)


def downgrade() -> None:
    # Index first: SQLite's ALTER TABLE DROP COLUMN refuses a column an index refers to.
    op.drop_index("ix_documents_glossary_fp", table_name="documents")
    op.drop_column("documents", "glossary_fp")
