"""durable derived-generation rebuilds

Revision ID: 71be92f03ad4
Revises: 4d8f12a6bc91
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "71be92f03ad4"
down_revision: str | Sequence[str] | None = "4d8f12a6bc91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "derived_generations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("snapshot_run_id", sa.Text(), nullable=False),
        sa.Column("snapshot_membership_hash", sa.Text(), nullable=False),
        sa.Column("expected_item_count", sa.Integer(), nullable=False),
        sa.Column("target_digest", sa.Text(), nullable=False),
        sa.Column("target", sa.JSON(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "planned",
                "building",
                "validating",
                "published",
                "failed",
                "canceled",
                name="rebuild_state",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("next_sequence", sa.Integer(), nullable=False),
        sa.Column("documents_built", sa.Integer(), nullable=False),
        sa.Column("chunks_built", sa.Integer(), nullable=False),
        sa.Column("vectors_reused", sa.Integer(), nullable=False),
        sa.Column("vectors_embedded", sa.Integer(), nullable=False),
        sa.Column("vector_publication_id", sa.Text(), nullable=True),
        sa.Column("expected_vector_table", sa.Text(), nullable=True),
        sa.Column("expected_vector_inventory_digest", sa.Text(), nullable=True),
        sa.Column("fence_generation", sa.Integer(), nullable=True),
        sa.Column("lease_owner", sa.Text(), nullable=True),
        sa.Column("lease_generation", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("diagnostic_code", sa.Text(), nullable=True),
        sa.Column("diagnostic_count", sa.Integer(), nullable=False),
        sa.Column("published_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "expected_item_count >= 0 AND next_sequence >= 0 AND documents_built >= 0 "
            "AND chunks_built >= 0 AND vectors_reused >= 0 AND vectors_embedded >= 0 "
            "AND lease_generation >= 0 AND diagnostic_count >= 0 "
            "AND (fence_generation IS NULL OR fence_generation >= 1)",
            name=op.f("ck_derived_generations_derived_generation_counts_are_not_negative"),
        ),
        sa.CheckConstraint(
            "(state = 'published') = (published_at IS NOT NULL)",
            name=op.f(
                "ck_derived_generations_derived_generation_publication_timestamp_matches_state"
            ),
        ),
        sa.ForeignKeyConstraint(["snapshot_run_id"], ["acquisition_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_derived_generations")),
        sa.UniqueConstraint(
            "workspace_id",
            "snapshot_run_id",
            "target_digest",
            name="uq_derived_generation_plan",
        ),
    )
    op.create_index(
        "ix_derived_generations_workspace_state",
        "derived_generations",
        ["workspace_id", "state"],
    )
    op.create_index(
        "uq_derived_generations_workspace_fence",
        "derived_generations",
        ["workspace_id", "fence_generation"],
        unique=True,
        sqlite_where=sa.text("fence_generation IS NOT NULL"),
    )
    op.create_table(
        "derived_generation_items",
        sa.Column("generation_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("payload_digest", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("temporary_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "sequence >= 0 AND temporary_bytes >= 0",
            name=op.f(
                "ck_derived_generation_items_derived_generation_item_counts_are_not_negative"
            ),
        ),
        sa.ForeignKeyConstraint(["generation_id"], ["derived_generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint(
            "generation_id", "sequence", name=op.f("pk_derived_generation_items")
        ),
        sqlite_with_rowid=False,
    )
    op.create_index(
        "uq_derived_generation_items_document",
        "derived_generation_items",
        ["generation_id", "document_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_derived_generation_items_document", table_name="derived_generation_items")
    op.drop_table("derived_generation_items")
    op.drop_index("uq_derived_generations_workspace_fence", table_name="derived_generations")
    op.drop_index("ix_derived_generations_workspace_state", table_name="derived_generations")
    op.drop_table("derived_generations")
