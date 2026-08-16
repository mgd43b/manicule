"""retain historical version metadata after releasing source bytes

Revision ID: a83d4e106b29
Revises: 71be92f03ad4
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "a83d4e106b29"
down_revision: str | Sequence[str] | None = "71be92f03ad4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("bytes_released_at", manicule.storage.types.UtcDateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_versions", "bytes_released_at")
