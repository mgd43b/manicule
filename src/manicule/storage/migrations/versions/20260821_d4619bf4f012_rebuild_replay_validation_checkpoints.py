"""persist durable resumable checkpoints for rebuild replay and validation

Revision ID: d4619bf4f012
Revises: a7d40c3e91b5
Created: 2026-08-21 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "d4619bf4f012"
down_revision: str | Sequence[str] | None = "a7d40c3e91b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("derived_generations") as batch:
        # Distinct from `lease_expires_at`: a heartbeat renewing a lease proves the worker is
        # alive, not that a page of replay or validation actually completed. NULL until the
        # first durable page commit.
        batch.add_column(
            sa.Column("last_progress_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
        # Replay checkpoint. `replay_lease_generation` is what makes the cursor trustworthy: it
        # names the lease generation whose physical vector namespace the cursor was verified
        # against. A takeover bumps `lease_generation` and copies into a fresh namespace, so a
        # cursor recorded under a stale lease generation is correctly ignored rather than
        # resumed — only a crash-and-retry under the *same* still-held lease reuses it.
        batch.add_column(sa.Column("replay_lease_generation", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("replay_checkpoint_sequence", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("replayed_vector_count", sa.Integer(), nullable=False, server_default="0")
        )
        # Validation checkpoint, same shape and the same reason: staged items are sealed once
        # `VALIDATING` begins, so a checkpoint recorded under the lease generation that is still
        # current is trustworthy evidence rather than a bare counter.
        batch.add_column(sa.Column("validation_lease_generation", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("validation_checkpoint_sequence", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("validated_vector_count", sa.Integer(), nullable=False, server_default="0")
        )
    with op.batch_alter_table("derived_generations") as batch:
        batch.alter_column(
            "replayed_vector_count",
            existing_type=sa.Integer(),
            existing_nullable=False,
            server_default=None,
        )
        batch.alter_column(
            "validated_vector_count",
            existing_type=sa.Integer(),
            existing_nullable=False,
            server_default=None,
        )
        batch.drop_constraint("derived_generation_counts_are_not_negative", type_="check")
        batch.create_check_constraint(
            "derived_generation_counts_are_not_negative",
            "expected_item_count >= 0 AND next_sequence >= 0 AND documents_built >= 0 "
            "AND chunks_built >= 0 AND vectors_reused >= 0 AND vectors_embedded >= 0 "
            "AND lease_generation >= 0 AND diagnostic_count >= 0 "
            "AND (fence_generation IS NULL OR fence_generation >= 1) "
            "AND (replay_lease_generation IS NULL OR replay_lease_generation >= 1) "
            "AND (replay_checkpoint_sequence IS NULL OR replay_checkpoint_sequence >= 0) "
            "AND replayed_vector_count >= 0 "
            "AND (validation_lease_generation IS NULL OR validation_lease_generation >= 1) "
            "AND (validation_checkpoint_sequence IS NULL OR validation_checkpoint_sequence >= 0) "
            "AND validated_vector_count >= 0",
        )


def downgrade() -> None:
    with op.batch_alter_table("derived_generations") as batch:
        batch.drop_constraint("derived_generation_counts_are_not_negative", type_="check")
        batch.create_check_constraint(
            "derived_generation_counts_are_not_negative",
            "expected_item_count >= 0 AND next_sequence >= 0 AND documents_built >= 0 "
            "AND chunks_built >= 0 AND vectors_reused >= 0 AND vectors_embedded >= 0 "
            "AND lease_generation >= 0 AND diagnostic_count >= 0 "
            "AND (fence_generation IS NULL OR fence_generation >= 1)",
        )
        batch.drop_column("validated_vector_count")
        batch.drop_column("validation_checkpoint_sequence")
        batch.drop_column("validation_lease_generation")
        batch.drop_column("replayed_vector_count")
        batch.drop_column("replay_checkpoint_sequence")
        batch.drop_column("replay_lease_generation")
        batch.drop_column("last_progress_at")
