"""authoritative source snapshot promotion

Revision ID: 4d8f12a6bc91
Revises: 8b7f91e42c0a
Create Date: 2026-08-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import manicule.storage.types

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "4d8f12a6bc91"
down_revision: str | Sequence[str] | None = "8b7f91e42c0a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("connectors") as batch:
        batch.add_column(sa.Column("watermark_scope_fingerprint", sa.Text(), nullable=True))
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.add_column(sa.Column("source_scope", sa.Text(), nullable=False, server_default=""))
        batch.add_column(
            sa.Column("scope_fingerprint", sa.Text(), nullable=False, server_default="")
        )
        batch.add_column(sa.Column("base_watermark_scope_fingerprint", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "promotion_policy",
                sa.Text(),
                nullable=False,
                server_default="require_complete",
            )
        )
        batch.add_column(
            sa.Column(
                "acquisition_completed_at", manicule.storage.types.UtcDateTime(), nullable=True
            )
        )
        batch.add_column(
            sa.Column("promoted_at", manicule.storage.types.UtcDateTime(), nullable=True)
        )
        batch.add_column(sa.Column("membership_hash", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "completeness",
                sa.Text(),
                nullable=True,
            )
        )
        batch.add_column(
            sa.Column("omission_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("omission_reasons", sa.JSON(), nullable=False, server_default="{}")
        )
    # A legacy committed watermark did not prove that UNCHANGED records retained bytes, and it
    # predates a canonical evidence digest. Preserve the cursor history but never forge an
    # authoritative snapshot marker from evidence the old schema did not require.
    op.execute(
        sa.text(
            "UPDATE acquisition_runs SET "
            "membership_hash = 'legacy-unverified', "
            "omission_count = (SELECT COUNT(*) FROM acquisition_records ar "
            "WHERE ar.run_id = acquisition_runs.id "
            "AND (ar.blob_ref IS NULL OR ar.acquired_source IS NULL)), "
            "omission_reasons = CASE WHEN (SELECT COUNT(*) FROM acquisition_records ar "
            "WHERE ar.run_id = acquisition_runs.id "
            "AND (ar.blob_ref IS NULL OR ar.acquired_source IS NULL)) > 0 "
            "THEN json_object('legacy_unverified', (SELECT COUNT(*) "
            "FROM acquisition_records ar WHERE ar.run_id = acquisition_runs.id "
            "AND (ar.blob_ref IS NULL OR ar.acquired_source IS NULL))) ELSE '{}' END "
            "WHERE watermark_committed_at IS NOT NULL"
        )
    )
    # The old mutable column was also free-form. Normalize it separately so ORM JSON loading
    # cannot be poisoned by malformed text before the defensive typed reader gets control.
    op.execute(
        sa.text(
            "UPDATE acquisition_records SET diagnostic = CASE "
            "WHEN diagnostic IS NULL THEN NULL "
            "WHEN NOT json_valid(diagnostic) THEN "
            "json_object('stage', 'acquisition', 'code', 'legacy_unverified', "
            "'retryable', json('true')) "
            "WHEN json_type(diagnostic) = 'object' THEN CASE WHEN "
            "json_type(diagnostic, '$.stage') = 'text' "
            "AND json_extract(diagnostic, '$.stage') IN "
            "('enumeration', 'acquisition', 'indexing', 'publication', 'capacity') "
            "AND json_type(diagnostic, '$.code') = 'text' "
            "AND json_extract(diagnostic, '$.code') IN "
            "('authentication', 'capacity', 'cursor_expired', 'fetch_failed', "
            "'missing_body', 'source_deleted', 'stale_body', 'parse_failed', "
            "'embed_failed', 'publication_failed', 'interrupted', "
            "'legacy_unverified', 'unknown') "
            "AND (json_type(diagnostic, '$.retryable') IS NULL "
            "OR json_type(diagnostic, '$.retryable') IN ('true', 'false')) "
            "AND NOT EXISTS (SELECT 1 FROM json_each(diagnostic) "
            "WHERE key NOT IN ('stage', 'code', 'retryable')) "
            "AND (SELECT COUNT(*) FROM json_each(diagnostic)) = "
            "(SELECT COUNT(DISTINCT key) FROM json_each(diagnostic)) "
            "THEN diagnostic ELSE json_object('stage', 'acquisition', "
            "'code', 'legacy_unverified', 'retryable', json('true')) END "
            "ELSE json_object('stage', 'acquisition', 'code', 'legacy_unverified', "
            "'retryable', json('true')) END"
        )
    )
    with op.batch_alter_table("acquisition_runs") as batch:
        for column in (
            "source_scope",
            "scope_fingerprint",
            "promotion_policy",
            "omission_count",
            "omission_reasons",
        ):
            batch.alter_column(column, server_default=None)
        batch.create_check_constraint(
            "snapshot_promotion_policy_is_known",
            "promotion_policy IN ('require_complete', 'allow_omissions')",
        )
        batch.create_check_constraint(
            "snapshot_completeness_is_known",
            "completeness IS NULL OR completeness IN ('complete', 'partial')",
        )
        batch.create_check_constraint("snapshot_omissions_are_not_negative", "omission_count >= 0")
        batch.create_check_constraint(
            "promoted_snapshot_has_complete_acquisition",
            "promoted_at IS NULL OR acquisition_completed_at IS NOT NULL",
        )
        batch.create_check_constraint(
            "committed_watermark_has_promoted_snapshot",
            "watermark_committed_at IS NULL OR promoted_at IS NOT NULL "
            "OR membership_hash = 'legacy-unverified'",
        )
    # SQLite stores enums as CHECK constraints, so adding a terminal omission outcome requires
    # a table rebuild. Alembic's batch operation preserves every row and foreign key.
    with op.batch_alter_table("acquisition_records") as batch:
        batch.add_column(sa.Column("snapshot_outcome", sa.Text(), nullable=True))
        batch.add_column(sa.Column("snapshot_diagnostic", sa.JSON(), nullable=True))
        batch.create_check_constraint(
            "snapshot_item_outcome_is_known",
            "snapshot_outcome IS NULL OR snapshot_outcome IN ('retained', 'reused', 'omitted')",
        )
        batch.create_check_constraint(
            "snapshot_diagnostic_is_valid_json",
            "snapshot_diagnostic IS NULL OR json_valid(snapshot_diagnostic)",
        )
        batch.alter_column(
            "state",
            existing_type=sa.Enum(
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
            type_=sa.Enum(
                "discovered",
                "acquiring",
                "acquired",
                "unchanged",
                "indexing",
                "settled",
                "retry",
                "omitted",
                name="acquisition_record_state",
                native_enum=False,
                create_constraint=True,
            ),
            existing_nullable=False,
        )
    op.create_index(
        "ix_acquisition_runs_latest_promoted",
        "acquisition_runs",
        ["workspace_id", "connector_name", "scope_fingerprint", "promoted_at", "id"],
    )
    op.create_index(
        "ix_acquisition_records_run_source_version",
        "acquisition_records",
        ["run_id", "source_id", "fetched_version_token"],
    )
    op.execute(
        sa.text(
            "UPDATE acquisition_records SET snapshot_diagnostic = CASE "
            "WHEN blob_ref IS NOT NULL AND acquired_source IS NOT NULL THEN NULL "
            "WHEN diagnostic IS NULL OR NOT json_valid(diagnostic) THEN "
            "json_object('stage', 'acquisition', 'code', 'legacy_unverified', "
            "'retryable', json('true')) "
            "WHEN json_type(diagnostic) = 'object' THEN CASE WHEN "
            "json_type(diagnostic, '$.stage') = 'text' "
            "AND json_extract(diagnostic, '$.stage') = 'acquisition' "
            "AND json_type(diagnostic, '$.code') = 'text' "
            "AND json_extract(diagnostic, '$.code') IN "
            "('authentication', 'capacity', 'cursor_expired', 'fetch_failed', "
            "'missing_body', 'source_deleted', 'stale_body', 'parse_failed', "
            "'embed_failed', 'publication_failed', 'interrupted', "
            "'legacy_unverified', 'unknown') "
            "AND (json_type(diagnostic, '$.retryable') IS NULL "
            "OR json_type(diagnostic, '$.retryable') IN ('true', 'false')) "
            "AND NOT EXISTS (SELECT 1 FROM json_each(diagnostic) "
            "WHERE key NOT IN ('stage', 'code', 'retryable')) "
            "AND (SELECT COUNT(*) FROM json_each(diagnostic)) = "
            "(SELECT COUNT(DISTINCT key) FROM json_each(diagnostic)) "
            "THEN diagnostic ELSE json_object('stage', 'acquisition', "
            "'code', 'legacy_unverified', 'retryable', json('true')) END "
            "ELSE json_object('stage', 'acquisition', 'code', 'legacy_unverified', "
            "'retryable', json('true')) END, "
            "snapshot_outcome = CASE "
            "WHEN state = 'unchanged' AND blob_ref IS NOT NULL AND acquired_source IS NOT NULL "
            "THEN 'reused' "
            "WHEN blob_ref IS NOT NULL AND acquired_source IS NOT NULL THEN 'retained' "
            "WHEN run_id IN (SELECT id FROM acquisition_runs WHERE promoted_at IS NOT NULL) "
            "THEN 'omitted' ELSE NULL END"
        )
    )


def downgrade() -> None:
    promoted = (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM acquisition_runs WHERE promoted_at IS NOT NULL"))
        .scalar_one()
    )
    omitted = (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM acquisition_records WHERE state = 'omitted'"))
        .scalar_one()
    )
    if promoted or omitted:
        msg = (
            "cannot downgrade authoritative snapshots while "
            f"{promoted} promoted manifest(s) and {omitted} omission record(s) remain"
        )
        raise RuntimeError(msg)
    op.drop_index("ix_acquisition_records_run_source_version", table_name="acquisition_records")
    op.drop_index("ix_acquisition_runs_latest_promoted", table_name="acquisition_runs")
    with op.batch_alter_table("acquisition_records") as batch:
        batch.drop_constraint("snapshot_diagnostic_is_valid_json", type_="check")
        batch.drop_constraint("snapshot_item_outcome_is_known", type_="check")
        batch.drop_column("snapshot_diagnostic")
        batch.drop_column("snapshot_outcome")
        batch.alter_column(
            "state",
            existing_type=sa.Enum(
                "discovered",
                "acquiring",
                "acquired",
                "unchanged",
                "indexing",
                "settled",
                "retry",
                "omitted",
                name="acquisition_record_state",
                native_enum=False,
                create_constraint=True,
            ),
            type_=sa.Enum(
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
            existing_nullable=False,
        )
    with op.batch_alter_table("acquisition_runs") as batch:
        batch.drop_constraint("committed_watermark_has_promoted_snapshot", type_="check")
        batch.drop_constraint("promoted_snapshot_has_complete_acquisition", type_="check")
        batch.drop_constraint("snapshot_omissions_are_not_negative", type_="check")
        batch.drop_constraint("snapshot_completeness_is_known", type_="check")
        batch.drop_constraint("snapshot_promotion_policy_is_known", type_="check")
    for column in (
        "omission_reasons",
        "omission_count",
        "completeness",
        "membership_hash",
        "promoted_at",
        "acquisition_completed_at",
        "promotion_policy",
        "base_watermark_scope_fingerprint",
        "scope_fingerprint",
        "source_scope",
    ):
        op.drop_column("acquisition_runs", column)
    op.drop_column("connectors", "watermark_scope_fingerprint")
