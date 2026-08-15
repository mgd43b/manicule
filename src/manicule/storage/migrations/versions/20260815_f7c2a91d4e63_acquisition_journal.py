"""durable acquisition runs and source journal

Revision ID: f7c2a91d4e63
Revises: 6e31b7d592ac
Created: 2026-08-15 15:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

revision: str = "f7c2a91d4e63"
down_revision: str | None = "6e31b7d592ac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _require_settled_downgrade() -> None:
    """Refuse to erase live work; settled diagnostic history may be discarded."""
    connection = op.get_bind()
    runs = connection.execute(
        sa.text("SELECT count(*) FROM acquisition_runs WHERE state != 'settled'")
    ).scalar_one()
    records = connection.execute(
        sa.text(
            "SELECT count(*) FROM acquisition_records WHERE state NOT IN ('settled', 'unchanged')"
        )
    ).scalar_one()
    if runs or records:
        msg = (
            "refusing to downgrade the acquisition journal while durable backlog exists: "
            f"{runs} unsettled run(s), {records} pending record(s). Settle this work with the "
            "current release before retrying."
        )
        raise RuntimeError(msg)


def upgrade() -> None:
    op.create_index(
        "uq_connectors_id_workspace_id",
        "connectors",
        ["id", "workspace_id"],
        unique=True,
    )
    op.create_table(
        "acquisition_runs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("connector_name", sa.Text(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "enumerating",
                "acquiring",
                "indexing",
                "settled",
                name="acquisition_run_state",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("base_watermark", sa.JSON(), nullable=True),
        sa.Column("candidate_watermark", sa.JSON(), nullable=True),
        sa.Column("enumeration_completed_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("watermark_committed_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("lease_owner", sa.Text(), nullable=True),
        sa.Column("lease_generation", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", manicule.storage.types.UtcDateTime(), nullable=True),
        sa.Column("discovered_count", sa.Integer(), nullable=False),
        sa.Column("acquired_count", sa.Integer(), nullable=False),
        sa.Column("indexed_count", sa.Integer(), nullable=False),
        sa.Column("unchanged_count", sa.Integer(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("metadata_bytes", sa.Integer(), nullable=False),
        sa.Column("acquired_blob_bytes", sa.Integer(), nullable=False),
        sa.Column("diagnostic", sa.JSON(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "discovered_count >= 0 AND acquired_count >= 0 AND indexed_count >= 0 "
            "AND unchanged_count >= 0 AND retry_count >= 0 AND metadata_bytes >= 0 "
            "AND acquired_blob_bytes >= 0",
            name=op.f("ck_acquisition_runs_acquisition_run_counters_are_not_negative"),
        ),
        sa.CheckConstraint(
            "lease_generation >= 0",
            name=op.f("ck_acquisition_runs_lease_generation_is_not_negative"),
        ),
        sa.CheckConstraint(
            "watermark_committed_at IS NULL OR enumeration_completed_at IS NOT NULL",
            name=op.f("ck_acquisition_runs_committed_watermark_has_complete_enumeration"),
        ),
        sa.ForeignKeyConstraint(
            ["connector_id", "workspace_id"],
            ["connectors.id", "connectors.workspace_id"],
            name=op.f("fk_acquisition_runs_connector_id_connectors"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_acquisition_runs_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_acquisition_runs")),
        sa.UniqueConstraint(
            "id",
            "workspace_id",
            "connector_id",
            name=op.f("uq_acquisition_runs_id_workspace_id_connector_id"),
        ),
    )
    op.create_index(
        "ix_acquisition_runs_workspace_connector_state",
        "acquisition_runs",
        ["workspace_id", "connector_id", "state"],
        unique=False,
    )
    op.create_table(
        "acquisition_records",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("source_record", sa.JSON(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "discovered",
                "acquiring",
                "acquired",
                "unchanged",
                "indexing",
                "settled",
                "retry",
                name="acquisition_record_state",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("blob_ref", sa.Text(), nullable=True),
        sa.Column("fetched_version_token", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("diagnostic", sa.JSON(), nullable=True),
        sa.Column("created_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", manicule.storage.types.UtcDateTime(), nullable=False),
        sa.CheckConstraint(
            "sequence >= 0 AND attempts >= 0",
            name=op.f("ck_acquisition_records_acquisition_record_numbers_are_not_negative"),
        ),
        sa.CheckConstraint(
            "state NOT IN ('acquired', 'indexing') OR blob_ref IS NOT NULL",
            name=op.f("ck_acquisition_records_blob_backed_acquisition_states_have_a_blob"),
        ),
        sa.ForeignKeyConstraint(
            ["blob_ref"],
            ["blobs.hash"],
            name=op.f("fk_acquisition_records_blob_ref_blobs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "workspace_id", "connector_id"],
            [
                "acquisition_runs.id",
                "acquisition_runs.workspace_id",
                "acquisition_runs.connector_id",
            ],
            name=op.f("fk_acquisition_records_run_id_acquisition_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_acquisition_records")),
        sa.UniqueConstraint(
            "run_id", "sequence", name=op.f("uq_acquisition_records_run_id_sequence")
        ),
        sa.UniqueConstraint(
            "run_id", "source_id", name=op.f("uq_acquisition_records_run_id_source_id")
        ),
    )
    op.create_index(
        "ix_acquisition_records_run_state_sequence",
        "acquisition_records",
        ["run_id", "state", "sequence"],
        unique=False,
    )


def downgrade() -> None:
    _require_settled_downgrade()
    op.drop_index("ix_acquisition_records_run_state_sequence", table_name="acquisition_records")
    op.drop_table("acquisition_records")
    op.drop_index("ix_acquisition_runs_workspace_connector_state", table_name="acquisition_runs")
    op.drop_table("acquisition_runs")
    op.drop_index("uq_connectors_id_workspace_id", table_name="connectors")
