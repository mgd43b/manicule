"""normalize cleared connector watermarks to SQL NULL

Revision ID: 2f8a6c1d9b47
Revises: c6d4a1e8f209
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2f8a6c1d9b47"
down_revision: str | Sequence[str] | None = "c6d4a1e8f209"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLAlchemy JSON encoded a cleared Python None as the JSON literal `null`.  Watermark CAS
    # intentionally uses SQL `IS NULL`, so normalize only that exact encoded literal.  A valid
    # connector watermark whose value is the JSON string "null" is stored as '"null"' and is
    # therefore preserved.
    op.execute(sa.text("UPDATE connectors SET watermark = NULL WHERE watermark = 'null'"))


def downgrade() -> None:
    # A SQL NULL is the schema's original empty-checkpoint representation.  Reintroducing JSON
    # null would recreate the promotion conflict this data repair removes.
    pass
