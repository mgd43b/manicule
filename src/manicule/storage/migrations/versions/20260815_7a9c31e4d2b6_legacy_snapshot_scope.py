"""record whether snapshot membership proves complete source scope

Revision ID: 7a9c31e4d2b6
Revises: 4d8f12a6bc91
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "7a9c31e4d2b6"
down_revision: str | Sequence[str] | None = "4d8f12a6bc91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.add_column(
            sa.Column(
                "scope_inventory_complete", sa.Boolean(), nullable=False, server_default=sa.true()
            )
        )
        batch.create_check_constraint(
            "committed_watermark_has_complete_scope_inventory",
            "watermark_committed_at IS NULL OR scope_inventory_complete = 1",
        )
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.alter_column("scope_inventory_complete", server_default=None)


def downgrade() -> None:
    incomplete = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM acquisition_runs WHERE scope_inventory_complete = 0"
            )
        )
        .scalar_one()
    )
    if incomplete:
        msg = (
            "cannot downgrade while "
            f"{incomplete} promoted or resumable legacy snapshot(s) require explicit "
            "incomplete-scope semantics"
        )
        raise RuntimeError(msg)
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.drop_constraint(
            "committed_watermark_has_complete_scope_inventory", type_="check"
        )
        batch.drop_column("scope_inventory_complete")
