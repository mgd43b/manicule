"""persist rebuild retained-evidence verification fences

Revision ID: c97a3e2b10f4
Revises: a83d4e106b29
Create Date: 2026-08-16
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "c97a3e2b10f4"
down_revision: str | Sequence[str] | None = "a83d4e106b29"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "derived_generations", sa.Column("evidence_inventory_digest", sa.Text(), nullable=True)
    )
    op.add_column(
        "derived_generations",
        sa.Column("evidence_verification_digest", sa.Text(), nullable=True),
    )
    op.add_column(
        "derived_generations",
        sa.Column("evidence_verification_lease_generation", sa.Integer(), nullable=True),
    )
    op.add_column(
        "derived_generations",
        sa.Column("evidence_verified_at", manicule.storage.types.UtcDateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("derived_generations", "evidence_verified_at")
    op.drop_column("derived_generations", "evidence_verification_lease_generation")
    op.drop_column("derived_generations", "evidence_verification_digest")
    op.drop_column("derived_generations", "evidence_inventory_digest")
