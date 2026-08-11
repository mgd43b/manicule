"""conversation sharing and message feedback

Two boundaries move here, which is why this is its own revision rather than a line in
somebody's feature branch.

**The share token stops being stored in plaintext.** It is a live bearer credential for an
unauthenticated URL, and this database is backed up, exported and imported — so a plaintext
token travels into artefacts that leave the access boundary that created it. Hashed, like
``api_keys.key_hash``, and shown to its creator exactly once. ``share_expires_at`` and
``shared_at`` arrive with it: a capability with no expiry accumulates forever and the set of
live ones becomes unknowable, and a share has to be a *snapshot* rather than a live view of a
conversation somebody kept using.

**Feedback moves onto the message.** A user rates an answer; an answer is a message. There
are answers with no retrieval behind them and answers whose retrieval succeeded and whose
generation failed, and both are ratable while neither has a usable ``query_logs`` row. So
``query_logs.feedback`` is dropped rather than left as a second home for one fact.

**Existing tokens are not migrated, they are revoked.** Hashing a stored plaintext token
would preserve every link that exists — which is the *opposite* of what this revision is for,
since those tokens have already travelled wherever the backups went. Rows carrying one are
un-shared, and re-sharing is one click. The downgrade cannot restore them either, and says
so, because a downgrade that invents plaintext tokens would be worse than one that loses
them.

Revision ID: c3f81a5b6e42
Revises: 9c1a4f7b2d10
Created: 2026-08-10 20:10:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "c3f81a5b6e42"
down_revision: str | None = "9c1a4f7b2d10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FEEDBACK = ("positive", "negative")
_REASONS = ("wrong", "incomplete", "citation-wrong", "too-slow", "other")


def _messages(*, with_feedback: bool) -> sa.Table:
    """``messages`` as it stands on one side of this revision.

    Described rather than reflected, for the reason the previous revision records: reflection
    recovers the columns and constraints but not the order SQLite wrote them in, so a rebuild
    emits DDL that differs byte for byte from what it started with while meaning the same
    thing — and that difference is exactly what the round-trip test compares.
    """
    metadata = sa.MetaData()
    columns: list[Any] = [
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("profile_used", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
    ]
    if with_feedback:
        columns.extend(
            [
                sa.Column("finish_reason", sa.Text(), nullable=True),
                sa.Column("feedback", sa.Text(), nullable=True),
                sa.Column("feedback_reason", sa.Text(), nullable=True),
                sa.Column("feedback_comment", sa.Text(), nullable=True),
                sa.Column("feedback_at", manicule.storage.types.UtcDateTime(), nullable=True),
                sa.Column("query_log_id", sa.Text(), nullable=True),
            ]
        )
    columns.append(sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False))
    columns.extend(
        [
            sa.PrimaryKeyConstraint("id", name="pk_messages"),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.id"],
                name="fk_messages_conversation_id_conversations",
                ondelete="CASCADE",
            ),
            sa.CheckConstraint(
                "role IN ('user', 'assistant', 'system')", name="ck_messages_role_is_known"
            ),
        ]
    )
    if with_feedback:
        columns.extend(
            [
                sa.ForeignKeyConstraint(
                    ["query_log_id"],
                    ["query_logs.id"],
                    name="fk_messages_query_log_id_query_logs",
                    ondelete="SET NULL",
                ),
                sa.CheckConstraint(
                    f"feedback IS NULL OR feedback IN "
                    f"({', '.join(repr(value) for value in _FEEDBACK)})",
                    name="ck_messages_feedback_is_known",
                ),
                sa.CheckConstraint(
                    f"feedback_reason IS NULL OR feedback_reason IN "
                    f"({', '.join(repr(value) for value in _REASONS)})",
                    name="ck_messages_feedback_reason_is_known",
                ),
            ]
        )
    table = sa.Table("messages", metadata, *columns)
    sa.Index("ix_messages_conversation_id_created_at", table.c.conversation_id, table.c.created_at)
    if with_feedback:
        sa.Index(
            "ix_messages_feedback", table.c.feedback, sqlite_where=sa.text("feedback IS NOT NULL")
        )
    return table


def upgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("share_expires_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("shared_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
        batch_op.drop_column("share_token")
        batch_op.add_column(sa.Column("share_token_hash", sa.Text(), nullable=True))
        batch_op.create_unique_constraint("uq_conversations_share_token_hash", ["share_token_hash"])

    # Revoked, not migrated. See the module docstring: every plaintext token that existed has
    # already travelled wherever this database's backups went.
    op.execute(sa.text("UPDATE conversations SET shared = 0"))

    with op.batch_alter_table(
        "messages", schema=None, copy_from=_messages(with_feedback=False)
    ) as batch_op:
        batch_op.add_column(sa.Column("finish_reason", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("feedback", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("feedback_reason", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("feedback_comment", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("feedback_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
        batch_op.add_column(sa.Column("query_log_id", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_messages_query_log_id_query_logs",
            "query_logs",
            ["query_log_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_check_constraint(
            "feedback_is_known",
            f"feedback IS NULL OR feedback IN ({', '.join(repr(v) for v in _FEEDBACK)})",
        )
        batch_op.create_check_constraint(
            "feedback_reason_is_known",
            f"feedback_reason IS NULL OR feedback_reason IN "
            f"({', '.join(repr(v) for v in _REASONS)})",
        )
        batch_op.create_index(
            "ix_messages_feedback", ["feedback"], sqlite_where=sa.text("feedback IS NOT NULL")
        )

    with op.batch_alter_table("query_logs", schema=None) as batch_op:
        batch_op.drop_column("feedback")


def downgrade() -> None:
    with op.batch_alter_table("query_logs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("feedback", sa.Text(), nullable=True))

    # Ratings are not copied back. `query_logs.feedback` is one nullable column per retrieval
    # run and `messages.feedback` is one per answer; the mapping is not one-to-one in either
    # direction, and inventing one would put ratings on runs nobody rated.
    with op.batch_alter_table(
        "messages", schema=None, copy_from=_messages(with_feedback=True)
    ) as batch_op:
        batch_op.drop_index("ix_messages_feedback")
        batch_op.drop_constraint("feedback_reason_is_known", type_="check")
        batch_op.drop_constraint("feedback_is_known", type_="check")
        batch_op.drop_constraint("fk_messages_query_log_id_query_logs", type_="foreignkey")
        batch_op.drop_column("query_log_id")
        batch_op.drop_column("feedback_at")
        batch_op.drop_column("feedback_comment")
        batch_op.drop_column("feedback_reason")
        batch_op.drop_column("feedback")
        batch_op.drop_column("finish_reason")

    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_constraint("uq_conversations_share_token_hash", type_="unique")
        batch_op.drop_column("share_token_hash")
        batch_op.drop_column("shared_at")
        batch_op.drop_column("share_expires_at")
        batch_op.add_column(sa.Column("share_token", sa.Text(), nullable=True))
        batch_op.create_unique_constraint("uq_conversations_share_token", ["share_token"])

    # A downgrade cannot restore plaintext tokens, because they were never stored. Leaving
    # `shared` set with a NULL token would describe links that resolve to nothing.
    op.execute(sa.text("UPDATE conversations SET shared = 0"))
