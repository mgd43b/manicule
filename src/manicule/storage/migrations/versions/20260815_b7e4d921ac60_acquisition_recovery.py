"""record generation-fenced acquisition supersession

Revision ID: b7e4d921ac60
Revises: c41d7ea923b8
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "b7e4d921ac60"
down_revision: str | Sequence[str] | None = "c41d7ea923b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "acquisition_runs",
        sa.Column("superseded_at", manicule.storage.types.UtcDateTime(), nullable=True),
    )
    op.add_column("acquisition_runs", sa.Column("superseded_by", sa.Text(), nullable=True))
    op.create_index(
        "ix_acquisition_runs_workspace_connector_recovery",
        "acquisition_runs",
        ["workspace_id", "connector_id", "superseded_at", "state", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_acquisition_runs_workspace_state_updated",
        "acquisition_runs",
        ["workspace_id", "state", "updated_at"],
        unique=False,
    )
    op.create_index(
        "ix_acquisition_runs_workspace_superseded_updated",
        "acquisition_runs",
        ["workspace_id", "superseded_at", "updated_at"],
        unique=False,
    )


def downgrade() -> None:
    superseded = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM acquisition_runs WHERE superseded_at IS NOT NULL"))
        .scalar_one()
    )
    if superseded:
        msg = (
            "cannot downgrade acquisition recovery while "
            f"{superseded} explicitly superseded run(s) remain"
        )
        raise RuntimeError(msg)
    op.drop_index("ix_acquisition_runs_workspace_superseded_updated", table_name="acquisition_runs")
    op.drop_index("ix_acquisition_runs_workspace_state_updated", table_name="acquisition_runs")
    op.drop_index("ix_acquisition_runs_workspace_connector_recovery", table_name="acquisition_runs")
    op.drop_column("acquisition_runs", "superseded_by")
    op.drop_column("acquisition_runs", "superseded_at")
