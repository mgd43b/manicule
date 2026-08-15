"""retain the complete acquired source envelope

Revision ID: c41d7ea923b8
Revises: f7c2a91d4e63
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "c41d7ea923b8"
down_revision: str | Sequence[str] | None = "f7c2a91d4e63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("acquisition_records", sa.Column("acquired_source", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("acquisition_records", "acquired_source")
