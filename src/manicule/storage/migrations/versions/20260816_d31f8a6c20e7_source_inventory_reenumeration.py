"""persist source-inventory invalidation and replacement lineage

Revision ID: d31f8a6c20e7
Revises: c97a3e2b10f4
Create Date: 2026-08-16
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "d31f8a6c20e7"
down_revision: str | Sequence[str] | None = "c97a3e2b10f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.add_column(sa.Column("supersedes_run_id", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "inventory_state",
                sa.String(length=22),
                nullable=False,
                server_default="current",
            ),
        )
        batch.add_column(
            sa.Column(
                "reconciled_deleted_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        batch.add_column(
            sa.Column("reused_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.create_check_constraint(
            "reconciled_deletions_are_not_negative", "reconciled_deleted_count >= 0"
        )
        batch.create_check_constraint(
            "reused_acquisition_count_is_not_negative", "reused_count >= 0"
        )
        batch.create_check_constraint(
            "acquisition_inventory_state",
            "inventory_state IN ('current', 'reenumeration_required', 'reenumerating', "
            "'reconciled')",
        )
    op.create_index(
        "ix_acquisition_runs_inventory_recovery",
        "acquisition_runs",
        ["workspace_id", "connector_id", "inventory_state", "superseded_at", "created_at"],
        unique=False,
    )
    # Preserve existing reuse accounting rather than making old snapshots appear to have fetched
    # every retained body after this aggregate becomes independently persisted.
    op.execute(
        sa.text(
            "UPDATE acquisition_runs SET reused_count = ("
            "SELECT count(*) FROM acquisition_records ar WHERE ar.run_id = acquisition_runs.id "
            "AND ar.snapshot_outcome = 'reused')"
        )
    )
    # Existing strict runs stuck on a typed post-enumeration deletion become recoverable on
    # their next ordinary sync. No candidate watermark or record evidence is rewritten.
    op.execute(
        sa.text(
            "UPDATE acquisition_runs SET inventory_state = 'reenumeration_required' "
            "WHERE enumeration_completed_at IS NOT NULL AND acquisition_completed_at IS NULL "
            "AND superseded_at IS NULL AND EXISTS ("
            "SELECT 1 FROM acquisition_records ar WHERE ar.run_id = acquisition_runs.id "
            "AND ar.state = 'retry' "
            "AND json_extract(ar.diagnostic, '$.stage') = 'acquisition' "
            "AND json_extract(ar.diagnostic, '$.code') = 'source_deleted')"
        )
    )
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.alter_column("inventory_state", server_default=None)
        batch.alter_column("reconciled_deleted_count", server_default=None)
        batch.alter_column("reused_count", server_default=None)


def downgrade() -> None:
    replacements = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM acquisition_runs "
                "WHERE inventory_state != 'current' OR supersedes_run_id IS NOT NULL"
            )
        )
        .scalar_one()
    )
    if replacements:
        raise RuntimeError(
            "cannot downgrade source inventory recovery while replacement history remains"
        )
    op.drop_index("ix_acquisition_runs_inventory_recovery", table_name="acquisition_runs")
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.drop_constraint("reconciled_deletions_are_not_negative", type_="check")
        batch.drop_constraint("reused_acquisition_count_is_not_negative", type_="check")
        batch.drop_constraint("acquisition_inventory_state", type_="check")
    op.drop_column("acquisition_runs", "reused_count")
    op.drop_column("acquisition_runs", "reconciled_deleted_count")
    op.drop_column("acquisition_runs", "inventory_state")
    op.drop_column("acquisition_runs", "supersedes_run_id")
