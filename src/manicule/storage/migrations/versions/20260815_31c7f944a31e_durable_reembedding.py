"""durable shadow re-embedding

Revision ID: 31c7f944a31e
Revises: c41d7ea923b8
Created: 2026-08-15 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "31c7f944a31e"
down_revision: str | None = "c41d7ea923b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CORPUS_TABLES = ("documents", "chunks")
_CORPUS_EVENTS = ("INSERT", "UPDATE", "DELETE")


def _trigger_name(table: str, event: str) -> str:
    return f"{table}_reembed_revision_{event.lower()}"


def _require_clean_downgrade() -> None:
    connection = op.get_bind()
    active = connection.execute(
        sa.text(
            "SELECT "
            "(SELECT count(*) FROM reembed_runs) + "
            "(SELECT count(*) FROM reembed_corpus_snapshots) + "
            "(SELECT count(*) FROM reembed_shadow_generations) + "
            "(SELECT count(*) FROM reembed_publication_receipts)"
        )
    ).scalar_one()
    published = connection.execute(
        sa.text("SELECT count(*) FROM index_state WHERE vector_table LIKE 'reembed-%'")
    ).scalar_one()
    if active or published:
        raise RuntimeError(
            "refusing to downgrade durable re-embedding while saved snapshots, runs, "
            "generations, receipts, or a published generation remain; clean them with the "
            "current release or rebuild into a target-version data directory"
        )


def upgrade() -> None:
    op.add_column("index_state", sa.Column("vector_inventory_digest", sa.Text(), nullable=True))
    op.create_table(
        "corpus_revision",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.CheckConstraint("id = 1", name=op.f("ck_corpus_revision_is_a_singleton")),
        sa.CheckConstraint(
            "revision >= 0", name=op.f("ck_corpus_revision_revision_is_not_negative")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_corpus_revision")),
    )
    op.execute("INSERT INTO corpus_revision (id, revision) VALUES (1, 0)")
    for table in _CORPUS_TABLES:
        for event in _CORPUS_EVENTS:
            op.execute(
                f"CREATE TRIGGER {_trigger_name(table, event)} AFTER {event} ON {table} "  # noqa: S608 - fixed identifiers
                "BEGIN "
                "UPDATE corpus_revision SET revision = revision + 1 WHERE id = 1; "
                "UPDATE index_state SET vector_inventory_digest = NULL WHERE id = 1; "
                "END"
            )
    op.create_table(
        "reembed_corpus_snapshots",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("live_json", sa.Text(), nullable=False),
        sa.Column("complete", sa.Boolean(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reembed_corpus_snapshots")),
    )
    op.create_table(
        "reembed_snapshot_documents",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["reembed_corpus_snapshots.id"],
            name=op.f("fk_reembed_snapshot_documents_snapshot_id_reembed_corpus_snapshots"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "snapshot_id", "document_id", name=op.f("pk_reembed_snapshot_documents")
        ),
        sqlite_with_rowid=False,
    )
    op.create_table(
        "reembed_snapshot_chunks",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["reembed_corpus_snapshots.id"],
            name=op.f("fk_reembed_snapshot_chunks_snapshot_id_reembed_corpus_snapshots"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "snapshot_id",
            "document_id",
            "position",
            "chunk_id",
            name=op.f("pk_reembed_snapshot_chunks"),
        ),
        sqlite_with_rowid=False,
    )
    op.create_table(
        "reembed_runs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("commitment_json", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("checkpoint_json", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.Text(), nullable=True),
        sa.Column("lease_generation", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "lease_generation >= 0",
            name=op.f("ck_reembed_runs_lease_generation_is_not_negative"),
        ),
        sa.CheckConstraint("revision >= 0", name=op.f("ck_reembed_runs_revision_is_not_negative")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reembed_runs")),
    )
    op.create_table(
        "reembed_shadow_generations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("fingerprint_json", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("inventory_digest", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("seal_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["reembed_runs.id"],
            name=op.f("fk_reembed_shadow_generations_run_id_reembed_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reembed_shadow_generations")),
        sa.UniqueConstraint("run_id", name=op.f("uq_reembed_shadow_generations_run_id")),
    )
    op.create_table(
        "reembed_publication_receipts",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("receipt_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["reembed_runs.id"],
            name=op.f("fk_reembed_publication_receipts_run_id_reembed_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_reembed_publication_receipts")),
    )


def downgrade() -> None:
    _require_clean_downgrade()
    op.drop_table("reembed_publication_receipts")
    op.drop_table("reembed_shadow_generations")
    op.drop_table("reembed_runs")
    op.drop_table("reembed_snapshot_chunks")
    op.drop_table("reembed_snapshot_documents")
    op.drop_table("reembed_corpus_snapshots")
    for table in reversed(_CORPUS_TABLES):
        for event in reversed(_CORPUS_EVENTS):
            op.execute(f"DROP TRIGGER {_trigger_name(table, event)}")
    op.drop_table("corpus_revision")
    op.drop_column("index_state", "vector_inventory_digest")
