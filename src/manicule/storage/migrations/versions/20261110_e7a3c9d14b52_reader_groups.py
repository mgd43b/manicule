"""reader groups, and sources granted to them

Three new tables, all empty on arrival, which together let an administrator say that a
source of a workspace is readable only by some of the people the workspace admits.

**``reader_groups``** names a set of people, per workspace. **``reader_group_members``** puts
a person in a group, and goes with either. **``source_reader_grants``** grants a source to a
group; a source with no grant is readable by every reader of the workspace, exactly as every
source was before this revision, so an upgraded installation reads the same until somebody
grants something.

Nothing is added to ``documents``. A grant is evaluated when a query runs, through a join on
``documents.source``, so neither a column nor a rebuild of that table is needed — and a
rebuild of ``documents`` is the one operation this schema forbids, because dropping it fires
every cascade that points at it.

The grant's foreign key to its group carries no ``ON DELETE`` action, on purpose: a cascade
would delete a source's last grant with its group and so open the source to everyone.

Revision ID: e7a3c9d14b52
Revises: d2f5b8e31a64
Created: 2026-11-10 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "e7a3c9d14b52"
down_revision: str | None = "d2f5b8e31a64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reader_groups",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_reader_groups_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reader_groups")),
        sa.UniqueConstraint("workspace_id", "name", name="uq_reader_groups_workspace_id_name"),
    )
    op.create_table(
        "reader_group_members",
        sa.Column("group_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["reader_groups.id"],
            name=op.f("fk_reader_group_members_group_id_reader_groups"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_reader_group_members_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("group_id", "user_id", name=op.f("pk_reader_group_members")),
        sqlite_with_rowid=False,
    )
    op.create_index(
        "ix_reader_group_members_user_id", "reader_group_members", ["user_id"], unique=False
    )
    op.create_table(
        "source_reader_grants",
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("group_id", sa.Text(), nullable=False),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["reader_groups.id"],
            name=op.f("fk_source_reader_grants_group_id_reader_groups"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_source_reader_grants_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id", "source", "group_id", name=op.f("pk_source_reader_grants")
        ),
        sqlite_with_rowid=False,
    )
    op.create_index(
        "ix_source_reader_grants_group_id", "source_reader_grants", ["group_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_source_reader_grants_group_id", table_name="source_reader_grants")
    op.drop_table("source_reader_grants")
    op.drop_index("ix_reader_group_members_user_id", table_name="reader_group_members")
    op.drop_table("reader_group_members")
    op.drop_table("reader_groups")
