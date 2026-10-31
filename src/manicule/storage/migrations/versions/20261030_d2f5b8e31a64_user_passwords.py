"""a password hash on ``users``

A person created here can now sign in with a handle and a password. The hash lives on the
person rather than on a membership, because a person is installation-wide and so is the
password that proves they are the same one. It is ``NULL`` for everyone who has none, which is
every existing row, and for every person an identity provider vouches for.

Added as the last column so the batch rebuild and the ORM agree on order, which the round-trip
test compares character for character. The downgrade drops it, and with it every password.

Revision ID: d2f5b8e31a64
Revises: c9a4e7d21b30
Created: 2026-10-30 19:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "d2f5b8e31a64"
down_revision: str | None = "c9a4e7d21b30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _users(*, password_hash: bool) -> sa.Table:
    """``users`` as it stands on one side of this revision."""
    metadata = sa.MetaData()
    columns = [
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("last_login_at", manicule.storage.types.UtcDateTime(), nullable=True),
    ]
    if password_hash:
        columns.append(sa.Column("password_hash", sa.Text(), nullable=True))
    return sa.Table(
        "users",
        metadata,
        *columns,
        sa.CheckConstraint(
            "provider IN ('google', 'github', 'local')", name="ck_users_provider_is_known"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("provider", "subject", name="uq_users_provider_subject"),
    )


def upgrade() -> None:
    with op.batch_alter_table(
        "users", schema=None, copy_from=_users(password_hash=False)
    ) as batch_op:
        batch_op.add_column(sa.Column("password_hash", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table(
        "users", schema=None, copy_from=_users(password_hash=True)
    ) as batch_op:
        batch_op.drop_column("password_hash")
