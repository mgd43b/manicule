"""per-document chunk-relation lineage

``documents`` gains ``relation_fp``: the canonical
:class:`~manicule.core.fingerprints.RelationFingerprint` of the extractor run that decided this
document's stored ``chunk_relations`` rows. Until now the schema recorded which parser, chunker,
embedder and detector built a document and nothing about which *extractor* derived its edges —
so a corrected link-normalization rule would land against a corpus that kept the edges the old
rule produced and reported itself current, because no other fingerprint moves when an extraction
rule does.

**It is not backfilled, and that is the decision this revision is making.** Every existing row
keeps ``NULL``, which reads as "these edges were never derived by anything this index can name".
On an installation with no relation middleware configured that is also the literal truth: there
are no edges, nothing has looked for any, and the first run that configures an extractor selects
the whole corpus. Writing the installed fingerprint into existing rows instead would assert that
their edges came out of rules that have never run over them, which is false for every corpus and
is the class of quiet, plausible falsehood the fingerprints exist to prevent.

**``NULL`` is also not "no links".** A document that genuinely contains no ``[[link]]`` records a
fingerprint and no rows once it has been scanned, which is what makes "the current extractor
found nothing here" distinguishable from "nobody has looked". Collapsing the two is the
expensive mistake rather than the untidy one: most of a corpus is link-free, so every repair
would re-select almost all of it and end with as much outstanding as it started with.

**A plain ``ADD COLUMN``, deliberately not a batch rebuild**, for the reason ``a71f3c9d0e55``
sets out and ``d4a90c7e15b3`` repeats: SQLite implements a batch alteration as CREATE-temp,
INSERT…SELECT, ``DROP TABLE``, RENAME, and with ``PRAGMA foreign_keys=ON`` — which this project
sets on every connection — that ``DROP`` fires every ``ON DELETE CASCADE`` pointing at
``documents``, emptying chunks, versions, tags, collection membership, glossary entries and now
the very relations this column describes, while reporting success. Adding a nullable column
needs none of that. The downgrade drops the index before the column, because SQLite refuses
``DROP COLUMN`` on an indexed column and the discovery would be made at the moment somebody is
downgrading under pressure.

Revision ID: a3f7c1d8e520
Revises: 2f8a6c1d9b47
Created: 2026-09-13 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3f7c1d8e520"
down_revision: str | None = "2f8a6c1d9b47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("relation_fp", sa.Text(), nullable=True))
    op.create_index("ix_documents_relation_fp", "documents", ["relation_fp"], unique=False)


def downgrade() -> None:
    # Index first: SQLite's ALTER TABLE DROP COLUMN refuses a column an index refers to.
    op.drop_index("ix_documents_relation_fp", table_name="documents")
    op.drop_column("documents", "relation_fp")
