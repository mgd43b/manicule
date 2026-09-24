"""team mode: people, sessions and security alerts

Three new tables and two rebuilt ones, which together turn the identity model from "whoever
holds a key" into "a person, in a role, in a workspace".

**``users``, ``auth_sessions`` and ``security_alerts`` are new and arrive empty.** A person is
created the first time they sign in through a configured provider; a session the moment they
do; an alert when a pattern of use crosses a threshold. None of them can be derived from
anything already stored, so nothing is backfilled.

**``workspace_members`` gains the foreign key it always implied, and a ``disabled_at``.** The
table was created with the initial schema and never written: nothing in manicule had a person
to record. Its ``user_id`` now refers to ``users``. Any row whose person does not exist is
removed rather than kept dangling — there can be none in practice, since no release ever
inserted one, and a membership naming nobody is not something the new foreign key could hold.

**``api_keys`` loses ``scopes`` and its ``user_id`` becomes a real reference.** ``scopes`` was
written empty on every key and read by nothing: roles are the authorization model, and a
second axis that no check consults is a setting that appears to be in force and is not.
``user_id`` held the *workspace name* — a placeholder from before there were people — so every
existing value is cleared to ``NULL``, which is what a key minted by the operator at the
command line now records. ``allowed_ips`` and ``rate_limit`` stay, and are now enforced.

The two rebuilds are batch operations over tables described rather than reflected, for the
reason ``c3f81a5b6e42`` records: reflection does not recover the order SQLite wrote the
columns in, and the round-trip test compares the DDL character for character.

Revision ID: e41b9c07d2a5
Revises: c7d1a4e83f26
Created: 2026-09-24 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "e41b9c07d2a5"
down_revision: str | None = "c7d1a4e83f26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ROLES = "role IN ('admin', 'member', 'viewer')"


def _api_keys(*, team_mode: bool) -> sa.Table:
    """``api_keys`` as it stands on one side of this revision."""
    metadata = sa.MetaData()
    columns: list[Any] = [
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=team_mode),
        sa.Column("role", sa.Text(), nullable=False),
    ]
    if not team_mode:
        columns.append(sa.Column("scopes", sa.JSON(), nullable=False))
    columns.extend(
        [
            sa.Column("allowed_ips", sa.JSON(), nullable=False),
            sa.Column("rate_limit", sa.Integer(), nullable=True),
            sa.Column("expires_at", manicule.storage.types.UtcDateTime(), nullable=True),
            sa.Column("last_used_at", manicule.storage.types.UtcDateTime(), nullable=True),
            sa.Column("revoked_at", manicule.storage.types.UtcDateTime(), nullable=True),
            sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
            sa.CheckConstraint(_ROLES, name="ck_api_keys_role_is_known"),
            sa.ForeignKeyConstraint(
                ["workspace_id"],
                ["workspaces.id"],
                name="fk_api_keys_workspace_id_workspaces",
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id", name="pk_api_keys"),
            sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        ]
    )
    if team_mode:
        columns.extend(
            [
                sa.CheckConstraint(
                    "rate_limit IS NULL OR rate_limit > 0",
                    name="ck_api_keys_rate_limit_is_positive",
                ),
                sa.ForeignKeyConstraint(
                    ["user_id"],
                    ["users.id"],
                    name="fk_api_keys_user_id_users",
                    ondelete="CASCADE",
                ),
            ]
        )
    table = sa.Table("api_keys", metadata, *columns)
    sa.Index("ix_api_keys_workspace_id", table.c.workspace_id)
    if team_mode:
        sa.Index("ix_api_keys_user_id", table.c.user_id)
    return table


def _workspace_members(*, team_mode: bool) -> sa.Table:
    """``workspace_members`` as it stands on one side of this revision."""
    metadata = sa.MetaData()
    columns: list[Any] = [
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
    ]
    if team_mode:
        columns.append(
            sa.Column("disabled_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
    columns.extend(
        [
            sa.CheckConstraint(_ROLES, name="ck_workspace_members_role_is_known"),
            sa.ForeignKeyConstraint(
                ["workspace_id"],
                ["workspaces.id"],
                name="fk_workspace_members_workspace_id_workspaces",
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("workspace_id", "user_id", name="pk_workspace_members"),
        ]
    )
    if team_mode:
        columns.append(
            sa.ForeignKeyConstraint(
                ["user_id"],
                ["users.id"],
                name="fk_workspace_members_user_id_users",
                ondelete="CASCADE",
            )
        )
    table = sa.Table("workspace_members", metadata, *columns, sqlite_with_rowid=False)
    if team_mode:
        sa.Index("ix_workspace_members_user_id", table.c.user_id)
    return table


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("last_login_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.CheckConstraint(
            "provider IN ('google', 'github')", name=op.f("ck_users_provider_is_known")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("provider", "subject", name=op.f("uq_users_provider_subject")),
    )

    # A membership naming nobody cannot satisfy the foreign key below. No release ever wrote
    # one, so this deletes nothing that exists; it is here so that the constraint is true of
    # every row the moment it arrives rather than of every row but the ones nobody checked.
    op.execute(sa.text("DELETE FROM workspace_members"))
    with op.batch_alter_table(
        "workspace_members", schema=None, copy_from=_workspace_members(team_mode=False)
    ) as batch_op:
        batch_op.add_column(
            sa.Column("disabled_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_workspace_members_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index("ix_workspace_members_user_id", ["user_id"], unique=False)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("expires_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("revoked_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_auth_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_auth_sessions_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_auth_sessions")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_auth_sessions_token_hash")),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"], unique=False)
    op.create_index(
        "ix_auth_sessions_workspace_id", "auth_sessions", ["workspace_id"], unique=False
    )

    with op.batch_alter_table(
        "api_keys", schema=None, copy_from=_api_keys(team_mode=False)
    ) as batch_op:
        batch_op.drop_column("scopes")
        batch_op.alter_column("user_id", existing_type=sa.Text(), nullable=True)
        batch_op.create_check_constraint(
            "rate_limit_is_positive", "rate_limit IS NULL OR rate_limit > 0"
        )
        batch_op.create_foreign_key(
            "fk_api_keys_user_id_users", "users", ["user_id"], ["id"], ondelete="CASCADE"
        )
        batch_op.create_index("ix_api_keys_user_id", ["user_id"], unique=False)
    # Every existing value is the workspace's own name, written as a placeholder before there
    # were people. A key minted without a person behind it now records NULL.
    op.execute(sa.text("UPDATE api_keys SET user_id = NULL"))

    op.create_table(
        "security_alerts",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("acknowledged_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("acknowledged_by", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('brute_force', 'key_abuse', 'export_volume')",
            name=op.f("ck_security_alerts_kind_is_known"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_security_alerts")),
    )
    op.create_index(
        "ix_security_alerts_workspace_id_created_at",
        "security_alerts",
        ["workspace_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_security_alerts_workspace_id_created_at", table_name="security_alerts")
    op.drop_table("security_alerts")

    # The column goes back to NOT NULL, and the placeholder it used to hold is the only value
    # there is to put back.
    op.execute(sa.text("UPDATE api_keys SET user_id = workspace_id WHERE user_id IS NULL"))
    with op.batch_alter_table(
        "api_keys", schema=None, copy_from=_api_keys(team_mode=True)
    ) as batch_op:
        batch_op.drop_index("ix_api_keys_user_id")
        batch_op.drop_constraint("fk_api_keys_user_id_users", type_="foreignkey")
        batch_op.drop_constraint("rate_limit_is_positive", type_="check")
        batch_op.alter_column("user_id", existing_type=sa.Text(), nullable=False)
        batch_op.add_column(
            sa.Column("scopes", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
        )

    op.drop_index("ix_auth_sessions_workspace_id", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")

    with op.batch_alter_table(
        "workspace_members", schema=None, copy_from=_workspace_members(team_mode=True)
    ) as batch_op:
        batch_op.drop_index("ix_workspace_members_user_id")
        batch_op.drop_constraint("fk_workspace_members_user_id_users", type_="foreignkey")
        batch_op.drop_column("disabled_at")

    op.drop_table("users")
