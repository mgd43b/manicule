"""workspace-wide derived-generation snapshots

Revision ID: c4b8d1e7a902
Revises: b9e14c72a6f0
Created: 2026-08-17 16:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4b8d1e7a902"
down_revision: str | Sequence[str] | None = "b9e14c72a6f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("derived_generations") as batch:
        batch.drop_constraint("uq_derived_generation_plan", type_="unique")
        batch.create_unique_constraint(
            "uq_derived_generation_plan",
            [
                "workspace_id",
                "snapshot_run_id",
                "snapshot_membership_hash",
                "target_digest",
                "publication_identity_digest",
            ],
        )
    op.create_table(
        "derived_generation_snapshots",
        sa.Column("generation_id", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("connector_name", sa.Text(), nullable=False),
        sa.Column("scope_fingerprint", sa.Text(), nullable=False),
        sa.Column("membership_hash", sa.Text(), nullable=False),
        sa.Column("expected_item_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "ordinal >= 0 AND expected_item_count >= 0",
            name=op.f("ck_derived_generation_snapshots_derived_generation_snapshot_counts_are_not_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["generation_id"], ["derived_generations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["acquisition_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("generation_id", "ordinal"),
        sa.UniqueConstraint(
            "generation_id", "run_id", name="uq_generation_snapshot_run"
        ),
    )
    op.execute(
        sa.text(
            "INSERT INTO derived_generation_snapshots "
            "(generation_id, ordinal, run_id, connector_name, scope_fingerprint, "
            "membership_hash, expected_item_count) "
            "SELECT dg.id, 0, dg.snapshot_run_id, ar.connector_name, ar.scope_fingerprint, "
            "dg.snapshot_membership_hash, dg.expected_item_count "
            "FROM derived_generations AS dg "
            "JOIN acquisition_runs AS ar ON ar.id = dg.snapshot_run_id"
        )
    )


def downgrade() -> None:
    multi = sa.text(
        "SELECT count(*) FROM (SELECT generation_id FROM derived_generation_snapshots "
        "GROUP BY generation_id HAVING count(*) > 1)"
    )
    if op.get_bind().execute(multi).scalar_one():
        raise RuntimeError(
            "refusing to downgrade workspace rebuild snapshots while multi-source "
            "derived generations exist"
        )
    new_format = sa.text(
        "SELECT count(*) FROM derived_generation_snapshots AS snapshots "
        "JOIN derived_generations AS generations ON generations.id = snapshots.generation_id "
        "WHERE snapshots.run_id != generations.snapshot_run_id "
        "OR snapshots.membership_hash != generations.snapshot_membership_hash"
    )
    if op.get_bind().execute(new_format).scalar_one():
        raise RuntimeError(
            "refusing to downgrade workspace rebuild snapshots while new-format "
            "derived generations exist"
        )
    op.drop_table("derived_generation_snapshots")
    with op.batch_alter_table("derived_generations") as batch:
        batch.drop_constraint("uq_derived_generation_plan", type_="unique")
        batch.create_unique_constraint(
            "uq_derived_generation_plan",
            [
                "workspace_id",
                "snapshot_run_id",
                "target_digest",
                "publication_identity_digest",
            ],
        )
