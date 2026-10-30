"""local users, and browser sessions bound to the key that started them

**``users.provider`` admits ``local``.** A person an administrator creates in this installation
has no identity provider behind them; their row says so with a provider of ``local`` and a
subject that is the handle they were given. Every existing row is a ``google`` or ``github``
account and stays as it is.

**``auth_sessions`` gains a nullable ``key_id``.** A browser can now exchange an API key for a
session, and the session has to remember which key it came from: the key's role caps what the
browser may do, and revoking or expiring the key ends the session. Every existing session was
started by an identity provider and records ``NULL``.

Both are batch rebuilds over tables described rather than reflected, for the reason
``c3f81a5b6e42`` records: reflection does not recover the order SQLite wrote the columns in, and
the round-trip test compares the DDL character for character. The migrator runs with foreign
keys off, so rebuilding ``users`` does not cascade into the tables that reference it.

Revision ID: c9a4e7d21b30
Revises: b6d1f3a7c408
Created: 2026-10-30 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "c9a4e7d21b30"
down_revision: str | None = "b6d1f3a7c408"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _users(*, providers: str) -> sa.Table:
    """``users`` as it stands on one side of this revision."""
    metadata = sa.MetaData()
    return sa.Table(
        "users",
        metadata,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("last_login_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.CheckConstraint(f"provider IN ({providers})", name="ck_users_provider_is_known"),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("provider", "subject", name="uq_users_provider_subject"),
    )


def _auth_sessions(*, key_id: bool) -> sa.Table:
    """``auth_sessions`` as it stands on one side of this revision."""
    metadata = sa.MetaData()
    columns: list[Any] = [
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("expires_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("revoked_at", manicule.storage.types.UtcDateTime(), nullable=True),
    ]
    if key_id:
        columns.append(sa.Column("key_id", sa.Text(), nullable=True))
    columns.extend(
        [
            sa.ForeignKeyConstraint(
                ["user_id"],
                ["users.id"],
                name="fk_auth_sessions_user_id_users",
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["workspace_id"],
                ["workspaces.id"],
                name="fk_auth_sessions_workspace_id_workspaces",
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id", name="pk_auth_sessions"),
            sa.UniqueConstraint("token_hash", name="uq_auth_sessions_token_hash"),
        ]
    )
    if key_id:
        columns.append(
            sa.ForeignKeyConstraint(
                ["key_id"],
                ["api_keys.id"],
                name="fk_auth_sessions_key_id_api_keys",
                ondelete="CASCADE",
            )
        )
    table = sa.Table("auth_sessions", metadata, *columns)
    sa.Index("ix_auth_sessions_user_id", table.c.user_id)
    sa.Index("ix_auth_sessions_workspace_id", table.c.workspace_id)
    return table


def upgrade() -> None:
    with op.batch_alter_table(
        "users", schema=None, copy_from=_users(providers="'google', 'github'")
    ) as batch_op:
        batch_op.drop_constraint(op.f("ck_users_provider_is_known"), type_="check")
        batch_op.create_check_constraint(
            "provider_is_known", "provider IN ('google', 'github', 'local')"
        )
    with op.batch_alter_table(
        "auth_sessions", schema=None, copy_from=_auth_sessions(key_id=False)
    ) as batch_op:
        batch_op.add_column(sa.Column("key_id", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_auth_sessions_key_id_api_keys",
            "api_keys",
            ["key_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    # A local person cannot exist where the constraint does not allow one, so they go, and
    # everything that names them goes with them. The migrator runs with foreign keys off, so
    # the cascades the schema declares are done here by hand.
    # Every session a key started goes, whoever owns it. Without its column nothing would
    # remember the key's role cap or revocation, and the cookie would become a full
    # membership session for a returning ``oauth`` installation.
    op.execute(sa.text("DELETE FROM auth_sessions WHERE key_id IS NOT NULL"))
    op.execute(
        sa.text(
            "DELETE FROM auth_sessions "
            "WHERE user_id IN (SELECT id FROM users WHERE provider = 'local')"
        )
    )
    op.execute(
        sa.text(
            "DELETE FROM api_keys WHERE user_id IN (SELECT id FROM users WHERE provider = 'local')"
        )
    )
    op.execute(
        sa.text(
            "DELETE FROM workspace_members "
            "WHERE user_id IN (SELECT id FROM users WHERE provider = 'local')"
        )
    )
    op.execute(sa.text("DELETE FROM users WHERE provider = 'local'"))
    with op.batch_alter_table(
        "auth_sessions", schema=None, copy_from=_auth_sessions(key_id=True)
    ) as batch_op:
        batch_op.drop_constraint(op.f("fk_auth_sessions_key_id_api_keys"), type_="foreignkey")
        batch_op.drop_column("key_id")
    with op.batch_alter_table(
        "users", schema=None, copy_from=_users(providers="'google', 'github', 'local'")
    ) as batch_op:
        batch_op.drop_constraint(op.f("ck_users_provider_is_known"), type_="check")
        batch_op.create_check_constraint("provider_is_known", "provider IN ('google', 'github')")
