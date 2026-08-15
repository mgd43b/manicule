"""scope durable re-embedding artifacts to workspaces

Revision ID: 8b7f91e42c0a
Revises: 31c7f944a31e
Created: 2026-08-15 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8b7f91e42c0a"
down_revision: str | None = "31c7f944a31e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (
    "reembed_corpus_snapshots",
    "reembed_snapshot_documents",
    "reembed_snapshot_chunks",
    "reembed_runs",
    "reembed_shadow_generations",
    "reembed_publication_receipts",
)


def _backfill() -> None:
    connection = op.get_bind()
    workspace_ids = [
        str(row[0]) for row in connection.execute(sa.text("SELECT id FROM workspaces ORDER BY id"))
    ]
    sole_workspace = workspace_ids[0] if len(workspace_ids) == 1 else None
    snapshots = connection.execute(
        sa.text("SELECT id FROM reembed_corpus_snapshots ORDER BY id")
    ).all()
    for (snapshot_id,) in snapshots:
        owners = connection.execute(
            sa.text(
                "SELECT DISTINCT json_extract(payload_json, '$.workspace_id') "
                "FROM reembed_snapshot_documents WHERE snapshot_id = :snapshot_id"
            ),
            {"snapshot_id": snapshot_id},
        ).all()
        values = {str(row[0]) for row in owners if row[0] is not None}
        if len(values) == 1:
            workspace_id = values.pop()
        elif not values and sole_workspace is not None:
            workspace_id = sole_workspace
        else:
            raise RuntimeError(
                "cannot safely assign a legacy re-embedding snapshot to one workspace; "
                "remove unfinished legacy re-embedding state before upgrading"
            )
        if workspace_id not in workspace_ids:
            raise RuntimeError("legacy re-embedding snapshot names an unknown workspace")
        parameters = {"workspace_id": workspace_id, "snapshot_id": snapshot_id}
        connection.execute(
            sa.text(
                "UPDATE reembed_corpus_snapshots SET workspace_id = :workspace_id "
                "WHERE id = :snapshot_id"
            ),
            parameters,
        )
        connection.execute(
            sa.text(
                "UPDATE reembed_snapshot_documents SET workspace_id = :workspace_id "
                "WHERE snapshot_id = :snapshot_id"
            ),
            parameters,
        )
        connection.execute(
            sa.text(
                "UPDATE reembed_snapshot_chunks SET workspace_id = :workspace_id "
                "WHERE snapshot_id = :snapshot_id"
            ),
            parameters,
        )

    connection.execute(
        sa.text(
            "UPDATE reembed_runs SET workspace_id = ("
            "SELECT workspace_id FROM reembed_corpus_snapshots WHERE id = "
            "json_extract(reembed_runs.commitment_json, '$.snapshot.id'))"
        )
    )
    connection.execute(
        sa.text(
            "UPDATE reembed_shadow_generations SET workspace_id = ("
            "SELECT workspace_id FROM reembed_runs WHERE id = "
            "reembed_shadow_generations.run_id)"
        )
    )
    connection.execute(
        sa.text(
            "UPDATE reembed_publication_receipts SET workspace_id = ("
            "SELECT workspace_id FROM reembed_runs WHERE id = "
            "reembed_publication_receipts.run_id)"
        )
    )
    connection.execute(
        sa.text(
            "UPDATE reembed_runs SET commitment_json = json_set(commitment_json, "
            "'$.snapshot.workspace_id', workspace_id), checkpoint_json = json_set("
            "checkpoint_json, '$.workspace_id', workspace_id, "
            "'$.commitment.snapshot.workspace_id', workspace_id, "
            "'$.workspace_documents_completed', "
            "json_extract(checkpoint_json, '$.documents_completed'), "
            "'$.workspace_chunks_completed', "
            "json_extract(checkpoint_json, '$.chunks_completed')) "
            "WHERE workspace_id IS NOT NULL"
        )
    )
    connection.execute(
        sa.text(
            "UPDATE reembed_publication_receipts SET receipt_json = json_set("
            "receipt_json, '$.workspace_id', workspace_id) WHERE workspace_id IS NOT NULL"
        )
    )
    missing = sum(
        int(
            connection.execute(
                sa.text(f"SELECT count(*) FROM {table} WHERE workspace_id IS NULL")  # noqa: S608
            ).scalar_one()
        )
        for table in _TABLES
    )
    if missing:
        raise RuntimeError(
            "cannot safely assign legacy re-embedding state to a workspace; remove the "
            "unfinished legacy state before upgrading"
        )


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("workspace_id", sa.Text(), nullable=True))
    _backfill()
    with op.batch_alter_table("reembed_corpus_snapshots") as batch:
        batch.alter_column("workspace_id", existing_type=sa.Text(), nullable=False)
        batch.drop_constraint(op.f("pk_reembed_corpus_snapshots"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_corpus_snapshots"), ["workspace_id", "id"])
        batch.create_foreign_key(
            op.f("fk_reembed_corpus_snapshots_workspace_id_workspaces"),
            "workspaces",
            ["workspace_id"],
            ["id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("reembed_runs") as batch:
        batch.alter_column("workspace_id", existing_type=sa.Text(), nullable=False)
        batch.drop_constraint(op.f("pk_reembed_runs"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_runs"), ["workspace_id", "id"])
        batch.create_foreign_key(
            op.f("fk_reembed_runs_workspace_id_workspaces"),
            "workspaces",
            ["workspace_id"],
            ["id"],
            ondelete="CASCADE",
        )
    for table, primary in (
        ("reembed_snapshot_documents", ["workspace_id", "snapshot_id", "document_id"]),
        (
            "reembed_snapshot_chunks",
            ["workspace_id", "snapshot_id", "document_id", "position", "chunk_id"],
        ),
    ):
        with op.batch_alter_table(table) as batch:
            batch.alter_column("workspace_id", existing_type=sa.Text(), nullable=False)
            batch.drop_constraint(
                op.f(f"fk_{table}_snapshot_id_reembed_corpus_snapshots"),
                type_="foreignkey",
            )
            batch.drop_constraint(op.f(f"pk_{table}"), type_="primary")
            batch.create_primary_key(op.f(f"pk_{table}"), primary)
            batch.create_foreign_key(
                op.f(f"fk_{table}_workspace_id_reembed_corpus_snapshots"),
                "reembed_corpus_snapshots",
                ["workspace_id", "snapshot_id"],
                ["workspace_id", "id"],
                ondelete="CASCADE",
            )
    with op.batch_alter_table("reembed_shadow_generations") as batch:
        batch.alter_column("workspace_id", existing_type=sa.Text(), nullable=False)
        batch.drop_constraint(
            op.f("fk_reembed_shadow_generations_run_id_reembed_runs"), type_="foreignkey"
        )
        batch.drop_constraint(op.f("uq_reembed_shadow_generations_run_id"), type_="unique")
        batch.drop_constraint(op.f("pk_reembed_shadow_generations"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_shadow_generations"), ["workspace_id", "id"])
        batch.create_unique_constraint(
            op.f("uq_reembed_shadow_generations_workspace_id_run_id"),
            ["workspace_id", "run_id"],
        )
        batch.create_foreign_key(
            op.f("fk_reembed_shadow_generations_workspace_id_reembed_runs"),
            "reembed_runs",
            ["workspace_id", "run_id"],
            ["workspace_id", "id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("reembed_publication_receipts") as batch:
        batch.alter_column("workspace_id", existing_type=sa.Text(), nullable=False)
        batch.drop_constraint(
            op.f("fk_reembed_publication_receipts_run_id_reembed_runs"), type_="foreignkey"
        )
        batch.drop_constraint(op.f("pk_reembed_publication_receipts"), type_="primary")
        batch.create_primary_key(
            op.f("pk_reembed_publication_receipts"), ["workspace_id", "run_id"]
        )
        batch.create_foreign_key(
            op.f("fk_reembed_publication_receipts_workspace_id_reembed_runs"),
            "reembed_runs",
            ["workspace_id", "run_id"],
            ["workspace_id", "id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    connection = op.get_bind()
    active = sum(
        int(connection.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one())  # noqa: S608
        for table in _TABLES
    )
    if active:
        raise RuntimeError(
            "refusing to downgrade durable re-embedding workspace ownership; clean all "
            "re-embedding artifacts with the current release before downgrading"
        )
    # No rows exist, so recreating these private tables through batch mode cannot lose state.
    with op.batch_alter_table("reembed_publication_receipts") as batch:
        batch.drop_constraint(
            op.f("fk_reembed_publication_receipts_workspace_id_reembed_runs"),
            type_="foreignkey",
        )
        batch.drop_constraint(op.f("pk_reembed_publication_receipts"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_publication_receipts"), ["run_id"])
        batch.create_foreign_key(
            op.f("fk_reembed_publication_receipts_run_id_reembed_runs"),
            "reembed_runs",
            ["run_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.drop_column("workspace_id")
    with op.batch_alter_table("reembed_shadow_generations") as batch:
        batch.drop_constraint(
            op.f("fk_reembed_shadow_generations_workspace_id_reembed_runs"),
            type_="foreignkey",
        )
        batch.drop_constraint(
            op.f("uq_reembed_shadow_generations_workspace_id_run_id"), type_="unique"
        )
        batch.drop_constraint(op.f("pk_reembed_shadow_generations"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_shadow_generations"), ["id"])
        batch.create_unique_constraint(op.f("uq_reembed_shadow_generations_run_id"), ["run_id"])
        batch.create_foreign_key(
            op.f("fk_reembed_shadow_generations_run_id_reembed_runs"),
            "reembed_runs",
            ["run_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.drop_column("workspace_id")
    for table, primary in (
        ("reembed_snapshot_chunks", ["snapshot_id", "document_id", "position", "chunk_id"]),
        ("reembed_snapshot_documents", ["snapshot_id", "document_id"]),
    ):
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(
                op.f(f"fk_{table}_workspace_id_reembed_corpus_snapshots"),
                type_="foreignkey",
            )
            batch.drop_constraint(op.f(f"pk_{table}"), type_="primary")
            batch.create_primary_key(op.f(f"pk_{table}"), primary)
            batch.create_foreign_key(
                op.f(f"fk_{table}_snapshot_id_reembed_corpus_snapshots"),
                "reembed_corpus_snapshots",
                ["snapshot_id"],
                ["id"],
                ondelete="CASCADE",
            )
            batch.drop_column("workspace_id")
    with op.batch_alter_table("reembed_runs") as batch:
        batch.drop_constraint(op.f("fk_reembed_runs_workspace_id_workspaces"), type_="foreignkey")
        batch.drop_constraint(op.f("pk_reembed_runs"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_runs"), ["id"])
        batch.drop_column("workspace_id")
    with op.batch_alter_table("reembed_corpus_snapshots") as batch:
        batch.drop_constraint(
            op.f("fk_reembed_corpus_snapshots_workspace_id_workspaces"), type_="foreignkey"
        )
        batch.drop_constraint(op.f("pk_reembed_corpus_snapshots"), type_="primary")
        batch.create_primary_key(op.f("pk_reembed_corpus_snapshots"), ["id"])
        batch.drop_column("workspace_id")
