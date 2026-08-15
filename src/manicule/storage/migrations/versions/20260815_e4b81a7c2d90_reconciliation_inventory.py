"""durable completed reconciliation inventories

Revision ID: e4b81a7c2d90
Revises: d52f81a439bc
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "e4b81a7c2d90"
down_revision: str | Sequence[str] | None = "d52f81a439bc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reconciliation_runs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("connector_name", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "enumerating",
                "completed",
                "proposed",
                "applied",
                "canceled",
                name="reconciliation_run_state",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("seen_count", sa.Integer(), nullable=False),
        sa.Column("live_count", sa.Integer(), nullable=True),
        sa.Column("missing_count", sa.Integer(), nullable=True),
        sa.Column("completed_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "state = 'enumerating' OR state = 'canceled' OR completed_at IS NOT NULL",
            name=op.f("ck_reconciliation_runs_completed_reconciliation_states_have_completion"),
        ),
        sa.CheckConstraint(
            "seen_count >= 0 AND (live_count IS NULL OR live_count >= 0) "
            "AND (missing_count IS NULL OR missing_count >= 0)",
            name=op.f("ck_reconciliation_runs_reconciliation_counts_are_not_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["connector_id", "workspace_id"],
            ["connectors.id", "connectors.workspace_id"],
            name=op.f("fk_reconciliation_runs_connector_id_connectors"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_reconciliation_runs_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reconciliation_runs")),
        sa.UniqueConstraint(
            "id",
            "workspace_id",
            "connector_id",
            name=op.f("uq_reconciliation_runs_id_workspace_id_connector_id"),
        ),
    )
    op.create_index(
        "ix_reconciliation_runs_workspace_connector_scope_state",
        "reconciliation_runs",
        ["workspace_id", "connector_id", "scope", "state", "created_at"],
        unique=False,
    )
    op.create_table(
        "reconciliation_inventory_items",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "workspace_id", "connector_id"],
            [
                "reconciliation_runs.id",
                "reconciliation_runs.workspace_id",
                "reconciliation_runs.connector_id",
            ],
            name=op.f("fk_reconciliation_inventory_items_run_id_reconciliation_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "source_id", name=op.f("pk_reconciliation_inventory_items")
        ),
        sqlite_with_rowid=False,
    )
    op.create_table(
        "reconciliation_candidates",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("publication_id", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_reconciliation_candidates_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "workspace_id", "connector_id"],
            [
                "reconciliation_runs.id",
                "reconciliation_runs.workspace_id",
                "reconciliation_runs.connector_id",
            ],
            name=op.f("fk_reconciliation_candidates_run_id_reconciliation_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "document_id", name=op.f("pk_reconciliation_candidates")
        ),
        sqlite_with_rowid=False,
    )


def downgrade() -> None:
    live = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM reconciliation_runs "
                "WHERE state IN ('enumerating', 'completed', 'proposed')"
            )
        )
        .scalar_one()
    )
    if live:
        msg = f"refusing to discard {live} unfinished reconciliation run(s)"
        raise RuntimeError(msg)
    op.drop_table("reconciliation_candidates")
    op.drop_table("reconciliation_inventory_items")
    op.drop_index(
        "ix_reconciliation_runs_workspace_connector_scope_state",
        table_name="reconciliation_runs",
    )
    op.drop_table("reconciliation_runs")
