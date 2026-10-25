"""an exact, trigger-kept total of the unpublished blobs

``BlobStore`` admits a new blob only if the bytes retained and not yet published, plus the new
one, fit ``max_acquired_blob_backlog_bytes``. It recomputed that sum on every admission by
scanning ``blobs`` against ``documents`` and ``document_versions``, which a first sync made
proportional to the number of messages it had already fetched. See ``manicule.storage
.blob_backlog`` for the design.

This revision adds ``blob_backlog_totals`` (one row), ``blob_backlog_members`` (the blobs it
counts, and what each added, so that counting one twice is impossible), the triggers that keep
them, and the two
indexes the triggers probe: ``documents.original_ref`` and ``document_versions.original_ref``,
which had none, so asking whether a blob is referenced was a scan.

**The total is computed here, from what is in the database, in Python.** An SQL ``SUM`` raises
on integer overflow where the scan this replaces did not, so rows are added one at a time, and a
total past what SQLite stores as an integer is stored as ``inexact``, which ``BlobStore`` reads as
"do not trust this" and answers by scanning, as before.

**Plain ``CREATE INDEX``, deliberately not a batch rebuild**, for the reason ``a3f7c1d8e520`` sets
out. The triggers are on ``documents`` and ``document_versions``, so a later batch alteration of
either would drop them as well as fire the cascades; ``tests/test_storage_blob_backlog.py`` fails
if they are missing at head.

Revision ID: a5c2e8f19b64
Revises: d8e3b5a02c71
Created: 2026-09-27 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from manicule.storage.blob_backlog import (
    CREATE_TRIGGERS,
    DROP_TRIGGERS,
    MEMBERS_FROM_BLOBS,
    initial_totals_sql,
)

revision: str = "a5c2e8f19b64"
down_revision: str | None = "d8e3b5a02c71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNPUBLISHED = (
    "FROM blobs WHERE hash NOT IN ("
    "SELECT original_ref FROM documents WHERE original_ref IS NOT NULL "
    "UNION SELECT original_ref FROM document_versions WHERE original_ref IS NOT NULL)"
)


def upgrade() -> None:
    op.create_table(
        "blob_backlog_totals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unpublished_bytes", sa.Integer(), nullable=False),
        sa.Column("unsized_rows", sa.Integer(), nullable=False),
        sa.Column("gc_pending_rows", sa.Integer(), nullable=False),
        sa.Column("inexact", sa.Integer(), nullable=False),
        sa.CheckConstraint("id = 1", name="blob_backlog_totals_is_singleton"),
    )
    op.create_table(
        "blob_backlog_members",
        sa.Column("hash", sa.Text(), primary_key=True),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("sized", sa.Integer(), nullable=False),
        sqlite_with_rowid=False,
    )
    op.create_index("ix_documents_original_ref", "documents", ["original_ref"], unique=False)
    op.create_index(
        "ix_document_versions_original_ref", "document_versions", ["original_ref"], unique=False
    )
    connection = op.get_bind()
    total: int | float = 0
    unsized = 0
    rows = connection.execute(
        sa.text(
            f"SELECT stored_bytes, typeof(stored_bytes) {_UNPUBLISHED} "
            "AND algo NOT LIKE 'gc_pending:%'"
        )
    )
    for stored_bytes, kind in rows:
        if kind == "integer":
            total += stored_bytes
        else:
            unsized += 1
    pending = connection.execute(
        sa.text("SELECT count(*) FROM blobs WHERE algo LIKE 'gc_pending:%'")
    ).scalar_one()
    # Past what SQLite holds as an integer the total cannot be kept, and says so.
    inexact = int(total > 2**63 - 1)
    op.execute(initial_totals_sql())
    op.execute(MEMBERS_FROM_BLOBS)
    connection.execute(
        sa.text(
            "UPDATE blob_backlog_totals SET unpublished_bytes = :total, unsized_rows = :unsized, "
            "gc_pending_rows = :pending, inexact = :inexact WHERE id = 1"
        ),
        {
            "total": 0 if inexact else total,
            "unsized": unsized,
            "pending": pending,
            "inexact": inexact,
        },
    )
    for statement in CREATE_TRIGGERS:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_TRIGGERS:
        op.execute(statement)
    op.drop_index("ix_document_versions_original_ref", table_name="document_versions")
    op.drop_index("ix_documents_original_ref", table_name="documents")
    op.drop_table("blob_backlog_members")
    op.drop_table("blob_backlog_totals")
