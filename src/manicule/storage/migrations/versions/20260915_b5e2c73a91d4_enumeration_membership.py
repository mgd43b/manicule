"""record whether an enumeration walked a whole scope or resumed from a cursor

``acquisition_runs`` gains ``enumeration_membership``: ``full_inventory`` when the run
enumerated its scope from nothing, ``incremental`` when it resumed from a committed cursor and
therefore listed only what had changed. The schema already recorded ``completeness``, but that
answers a different question — whether the bytes of the members it *did* enumerate are held
locally. A one-document delta whose single body was retained is ``complete``, and the offline
rebuild planner read that as authority over the connector's whole membership, priced the one
document, called itself runnable, and handed publication a replacement that soft-deleted every
document the delta had no reason to mention.

**The backfill is ``base_watermark IS NOT NULL``**, which is exactly the fact the ingest
pipeline consults when it decides what to hand a connector's ``discover``: a run that inherited
a usable cursor asked for a delta. It is not a perfect reconstruction — a run that inherited a
cursor and then discarded it, because its scope fingerprint had moved or because a
re-enumeration was required, walked the whole scope and is recorded here as incremental. That
error is in the safe direction. An incremental run is refused deletion authority and must prove
its coverage from an earlier full inventory before a rebuild will plan it, so a full inventory
mislabelled as a delta costs one refusal an operator can resolve; a delta mislabelled as a full
inventory costs the corpus.

**A batch alteration, not a plain ``ADD COLUMN``**, because the column carries a CHECK
constraint, and SQLite can only acquire one by rebuilding the table. ``acquisition_runs`` is
safe to rebuild where ``documents`` is not: nothing declares ``ON DELETE CASCADE`` against it,
so the ``DROP TABLE`` inside the batch takes no other table's rows with it. The two-pass
``server_default`` then ``server_default=None`` is this project's standing idiom — the first
pass backfills every existing row, the second removes the default so the migrated schema
matches the model, which declares only a Python-side default and would otherwise fail
``compare_server_default``.

``derived_generation_snapshots`` gains ``contributed_item_count`` in the same revision, because
the two facts arrive together: once a connector can bind more than one run, a run's own retained
inventory and what it owes a particular generation stop being the same number. Existing rows are
backfilled from ``expected_item_count`` — every generation planned before this revision bound one
run per connector, so each binding did owe the whole of its own manifest, and a zero would tell
an in-flight rebuild it had nothing left to build. The table's counts constraint is widened to
order the two and renamed to match what it now says.

Revision ID: b5e2c73a91d4
Revises: a3f7c1d8e520
Created: 2026-09-15 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b5e2c73a91d4"
down_revision: str | None = "a3f7c1d8e520"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.add_column(
            sa.Column(
                # `String(length=14)`, matching the widest member, because the model renders
                # this enum as a VARCHAR with a CHECK rather than a native type. A `Text()`
                # column here passes every test that reads rows back and fails the one that
                # compares the migrated schema against the model.
                "enumeration_membership",
                sa.String(length=14),
                nullable=False,
                server_default="full_inventory",
            )
        )
    op.execute(
        sa.text(
            "UPDATE acquisition_runs SET enumeration_membership = 'incremental' "
            "WHERE base_watermark IS NOT NULL"
        )
    )
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.alter_column(
            "enumeration_membership",
            existing_type=sa.String(length=14),
            existing_nullable=False,
            server_default=None,
        )
        # Named for the enum, not the column: that is the name SQLAlchemy generates for an
        # `Enum(..., create_constraint=True)`, and a different one reads as schema drift.
        batch.create_check_constraint(
            "snapshot_membership",
            "enumeration_membership IN ('full_inventory', 'incremental')",
        )
    with op.batch_alter_table("derived_generation_snapshots") as batch:
        batch.add_column(
            sa.Column("contributed_item_count", sa.Integer(), nullable=False, server_default="0")
        )
    # Every generation planned before this revision bound one run per connector, so each
    # binding contributed the whole of its own retained inventory. Leaving the backfill at
    # zero would tell an in-flight rebuild that it has nothing left to build.
    op.execute(
        sa.text(
            "UPDATE derived_generation_snapshots SET contributed_item_count = expected_item_count"
        )
    )
    with op.batch_alter_table("derived_generation_snapshots") as batch:
        batch.alter_column(
            "contributed_item_count",
            existing_type=sa.Integer(),
            existing_nullable=False,
            server_default=None,
        )
        batch.drop_constraint("derived_generation_snapshot_counts_are_not_negative", type_="check")
        batch.create_check_constraint(
            "derived_generation_snapshot_counts_are_coherent",
            "ordinal >= 0 AND expected_item_count >= 0 "
            "AND contributed_item_count >= 0 "
            "AND contributed_item_count <= expected_item_count",
        )


def downgrade() -> None:
    with op.batch_alter_table("derived_generation_snapshots") as batch:
        batch.drop_constraint("derived_generation_snapshot_counts_are_coherent", type_="check")
        batch.drop_column("contributed_item_count")
        batch.create_check_constraint(
            "derived_generation_snapshot_counts_are_not_negative",
            "ordinal >= 0 AND expected_item_count >= 0",
        )
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.drop_constraint("snapshot_membership", type_="check")
        batch.drop_column("enumeration_membership")
