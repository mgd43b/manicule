"""The schema, and the connection settings that decide whether it means anything."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select, text

from manicule.core.errors import InsecureTargetError
from manicule.storage import models
from manicule.storage.engine import (
    PRAGMAS,
    create_engine,
    prepare_data_dir,
    secure_output_dir,
)
from manicule.storage.migrator import current, head_revision, upgrade
from manicule.storage.types import UtcDateTime, utcnow
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

EXPECTED_TABLES = frozenset(models.Base.metadata.tables)
DOCUMENTED_RELATIONAL_TABLES = 40


def test_the_documented_relational_table_count_matches_the_models() -> None:
    assert len(EXPECTED_TABLES) == DOCUMENTED_RELATIONAL_TABLES


async def test_the_migration_creates_every_table_the_models_declare(engine: AsyncEngine) -> None:
    """A model without a migration is a schema that only exists in one developer's head."""
    async with engine.connect() as connection:
        names = set(
            (await connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'")))
            .scalars()
            .all()
        )
    assert names >= EXPECTED_TABLES


async def test_a_migrated_database_is_at_head(engine: AsyncEngine) -> None:
    """A database behind head is one the code will query with columns that do not exist."""
    assert await current(engine) == head_revision()


async def test_snapshot_reuse_queries_use_bounded_composite_indexes(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        latest_plan = (
            await connection.execute(
                text(
                    "EXPLAIN QUERY PLAN SELECT id FROM acquisition_runs "
                    "WHERE workspace_id = 'default' AND connector_name = 'wiki' "
                    "AND scope_fingerprint = 'scope' AND promoted_at IS NOT NULL "
                    "ORDER BY promoted_at DESC, id DESC LIMIT 1"
                )
            )
        ).all()
        record_indexes = {
            str(row[1])
            for row in (
                await connection.execute(text("PRAGMA index_list('acquisition_records')"))
            ).all()
        }
        run_indexes = {
            str(row[1])
            for row in (
                await connection.execute(text("PRAGMA index_list('acquisition_runs')"))
            ).all()
        }

    latest_detail = " ".join(str(row[-1]) for row in latest_plan)
    assert "ix_acquisition_runs_latest_promoted" in latest_detail
    assert "ix_acquisition_runs_latest_promoted" in run_indexes
    assert "ix_acquisition_records_run_source_version" in record_indexes


async def test_foreign_keys_are_enforced_on_every_pooled_connection(engine: AsyncEngine) -> None:
    """The single most common way a schema full of REFERENCES enforces nothing.

    ``foreign_keys`` is per-connection and defaults to OFF, so a pragma applied once at
    startup leaves every later connection silently skipping referential integrity.
    """
    for _ in range(3):
        async with engine.connect() as connection:
            assert (await connection.execute(text("PRAGMA foreign_keys"))).scalar() == 1


async def test_every_declared_pragma_is_actually_applied(engine: AsyncEngine) -> None:
    """A pragma that silently failed to apply looks exactly like one that worked."""
    expected = {"journal_mode": "wal", "foreign_keys": 1, "busy_timeout": 5000}
    async with engine.connect() as connection:
        for name, want in expected.items():
            got = (await connection.execute(text(f"PRAGMA {name}"))).scalar()
            assert str(got).lower() == str(want).lower(), name
    assert {name for name, _ in PRAGMAS} >= set(expected)


async def test_a_foreign_key_violation_is_refused(engine: AsyncEngine) -> None:
    """Proof the pragma above is doing something, not just reporting a value."""
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    async with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO chunks (id, document_id, text, embed_text, heading_text, "
                    "heading_path, kind, position, token_count, anchor, metadata, created_at) "
                    "VALUES ('c', 'nonexistent', 't', 't', '', '[]', 'prose', 0, 1, '{}', '{}', "
                    "'2026-01-01 00:00:00')"
                )
            )


async def test_a_naive_timestamp_is_rejected_rather_than_assumed_to_be_utc() -> None:
    """A timestamp without a zone is missing information; guessing produces silent skew."""
    column = UtcDateTime()
    with pytest.raises(ValueError, match="naive datetime"):
        # A naive datetime is the whole point of this test.
        column.process_bind_param(datetime(2026, 1, 1, 12, 0), None)  # noqa: DTZ001  # pyright: ignore[reportArgumentType] - the decorator ignores the dialect


async def test_timestamps_round_trip_as_aware_utc(engine: AsyncEngine) -> None:
    """One writer and one format, so ORDER BY created_at means what it says."""
    from manicule.storage.engine import session_factory  # noqa: PLC0415

    before = utcnow()
    async with session_factory(engine).begin() as session:
        session.add(models.Workspace(id="w", name="w", settings={}))
    async with session_factory(engine)() as session:
        row = await session.get(models.Workspace, "w")
    assert row is not None
    assert row.created_at.tzinfo is UTC
    assert before <= row.created_at <= utcnow()


async def test_documents_are_unique_per_source_identity_while_live(engine: AsyncEngine) -> None:
    """Two live rows for one source silently splits a document and half its citations."""
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    document = make_document()
    await store.upsert_document(document)

    async with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, "
                    "media_type, content_hash, status, metadata, created_at, updated_at) "
                    "VALUES ('other', 'default', :source, :source_id, 'file:///b', 'B', "
                    "'text/plain', 'h', 'indexed', '{}', '2026-01-01 00:00:00', "
                    "'2026-01-01 00:00:00')"
                ),
                {"source": document.source, "source_id": document.source_id},
            )


async def test_a_soft_deleted_document_does_not_block_re_ingesting_the_same_source(
    engine: AsyncEngine,
) -> None:
    """The uniqueness index is partial for exactly this case."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    document = make_document()
    await store.upsert_document(document)
    await store.soft_delete_document(document.id)

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, "
                "media_type, content_hash, status, metadata, created_at, updated_at) "
                "VALUES ('fresh', 'default', :source, :source_id, 'file:///b', 'B', "
                "'text/plain', 'h', 'indexed', '{}', '2026-01-01 00:00:00', "
                "'2026-01-01 00:00:00')"
            ),
            {"source": document.source, "source_id": document.source_id},
        )


async def test_a_document_status_outside_the_enum_is_refused(engine: AsyncEngine) -> None:
    """A misspelled status makes a document invisible to retrieval, silently and forever."""
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO workspaces (id, name, mode, settings, created_at) "
                "VALUES ('w2','w2','personal','{}','2026-01-01 00:00:00')"
            )
        )
    async with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, "
                    "media_type, content_hash, status, metadata, created_at, updated_at) "
                    "VALUES ('bad', 'w2', 'fs', 'x', 'file:///c', 'C', 'text/plain', 'h', "
                    "'indexd', '{}', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
                )
            )


async def test_failed_stage_is_required_exactly_when_the_status_is_failed(
    engine: AsyncEngine,
) -> None:
    """'Re-run everything that died in parse' has to be a query, not a grep."""
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO workspaces (id, name, mode, settings, created_at) "
                "VALUES ('w3','w3','personal','{}','2026-01-01 00:00:00')"
            )
        )
    async with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, "
                    "media_type, content_hash, status, failed_stage, metadata, created_at, "
                    "updated_at) VALUES ('f', 'w3', 'fs', 'y', 'file:///d', 'D', 'text/plain', "
                    "'h', 'failed', NULL, '{}', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
                )
            )


async def test_index_state_holds_at_most_one_row_per_workspace(engine: AsyncEngine) -> None:
    """Two rows for one workspace would be two answers to one identity question."""
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO workspaces (id, name, mode, settings, created_at) "
                "VALUES ('index-workspace', 'index-workspace', 'personal', '{}', "
                "'2026-01-01 00:00:00')"
            )
        )
        await connection.execute(
            text(
                "INSERT INTO index_state (workspace_id, created_at, updated_at) "
                "VALUES ('index-workspace', '2026-01-01 00:00:00', "
                "'2026-01-01 00:00:00')"
            )
        )
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO index_state (workspace_id, created_at, updated_at) "
                    "VALUES ('index-workspace', '2026-01-01 00:00:00', "
                    "'2026-01-01 00:00:00')"
                )
            )


async def test_a_chunk_cannot_relate_to_itself(engine: AsyncEngine) -> None:
    """A self-edge is never meaningful and quietly breaks graph traversal."""
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    async with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "INSERT INTO chunk_relations (source_chunk_id, target_chunk_id, relation_type) "
                    "VALUES ('c1', 'c1', 'next')"
                )
            )


async def test_enum_columns_store_the_value_not_the_python_member_name(
    engine: AsyncEngine,
) -> None:
    """``NO_EXTRACTABLE_TEXT`` and ``no_extractable_text`` are not the same string.

    Every document, query and CHECK constraint in the design refers to the value.
    """
    from manicule.core.content import DocumentStatus  # noqa: PLC0415
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    document = make_document(status=DocumentStatus.NO_EXTRACTABLE_TEXT)
    document = document.model_copy(update={"status_detail": "no text layer"})
    await store.upsert_document(document)

    async with engine.connect() as connection:
        stored = (
            await connection.execute(
                text("SELECT status FROM documents WHERE id = :i"), {"i": document.id}
            )
        ).scalar_one()
    assert stored == "no_extractable_text"


async def test_the_data_directory_is_created_private(tmp_path: Path) -> None:
    """With original bytes retained this directory holds the corpus itself.

    A permission that depends on the invoking shell's umask is not a default.
    """
    root = prepare_data_dir(tmp_path / "private")
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "vectors").stat().st_mode & 0o777 == 0o700
    assert (root / "blobs").stat().st_mode & 0o777 == 0o700


async def test_an_output_directory_is_created_private(tmp_path: Path) -> None:
    """The ordinary path: a directory manicule makes for a copy of the corpus is ``0700``."""
    target = tmp_path / "archive"
    secure_output_dir(target, operation="export")
    assert target.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_parents_invented_along_the_way_are_private_too(tmp_path: Path) -> None:
    """``mkdir(parents=True)`` creates the ones above at the umask, which is not a decision.

    `upgrade` names a destination two levels down that nobody has made before, so this is the
    ordinary path rather than a corner: the leaf would be ``0700`` inside a ``0755`` directory
    manicule had just invented for itself.
    """
    target = tmp_path / "invented" / "monday"

    secure_output_dir(target, operation="export")

    assert target.stat().st_mode & 0o777 == 0o700
    assert target.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_a_parent_that_was_already_there_is_left_exactly_as_found(tmp_path: Path) -> None:
    """Only the target is judged. An existing ancestor is the operator's, not manicule's.

    Tightening one would be a mode change to a directory nobody asked about, which is how a
    tool ends up chmod-ing something that mattered to somebody else.
    """
    parent = tmp_path / "theirs"
    parent.mkdir()
    parent.chmod(0o755)

    secure_output_dir(parent / "ours", operation="backup")

    assert parent.stat().st_mode & 0o777 == 0o755
    assert (parent / "ours").stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_a_pre_existing_exposed_output_directory_is_refused(tmp_path: Path) -> None:
    """The case ``mkdir(mode=…, exist_ok=True)`` never reaches, and the reason #60 existed.

    The refusal names the operation, the path and the mode, because each of the three is a
    question an operator would otherwise have to answer by hand.
    """
    target = tmp_path / "shared"
    target.mkdir()
    target.chmod(0o755)

    with pytest.raises(InsecureTargetError) as refusal:
        secure_output_dir(target, operation="export")

    message = str(refusal.value)
    assert message.startswith("export target "), "the refusal says which command stopped"
    assert str(target) in message
    assert "055" in message
    assert "--allow-insecure-target" in message, "a refusal with no way past it is a wall"


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_consent_writes_into_an_exposed_directory_and_changes_nothing_else(
    tmp_path: Path,
) -> None:
    """The escape hatch permits; it does not silently tighten what the operator set."""
    target = tmp_path / "shared"
    target.mkdir()
    target.chmod(0o755)

    secure_output_dir(target, operation="backup", allow_insecure=True)

    assert target.stat().st_mode & 0o777 == 0o755


async def test_a_directory_that_came_back_wider_than_it_was_asked_for_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mkdir(mode=0o700)`` is a request, and a default POSIX ACL can answer it with more.

    Simulated rather than staged: a default ACL needs ``setfacl`` and a filesystem mounted to
    honor it, neither of which a suite can assume. What is exercised for real is the
    consequence — the mode is *checked after* creation rather than asserted before it, so even
    a directory manicule created itself can be refused, and is then removed again.
    """
    target = tmp_path / "widened"

    def widened(_: Path) -> int:
        return 0o055

    monkeypatch.setattr("manicule.storage.engine.exposure", widened)

    with pytest.raises(InsecureTargetError, match="group or other permissions"):
        secure_output_dir(target, operation="backup")

    assert not target.exists(), "created here and refused here: leave nothing behind"


async def test_a_directory_whose_mode_cannot_be_read_is_a_different_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that cannot be examined is not one that is exposed, and ``doctor`` agrees.

    Reporting the second for the first sends an operator to ``chmod`` a path whose real
    problem is that it is not there, or not theirs.
    """

    def unreadable(_: Path) -> int:
        raise PermissionError("no")

    monkeypatch.setattr("manicule.storage.engine.exposure", unreadable)

    with pytest.raises(InsecureTargetError, match="cannot be examined"):
        secure_output_dir(tmp_path / "opaque", operation="export")


async def test_every_model_is_reachable_from_the_metadata() -> None:
    """A table declared but absent from the metadata never reaches a migration."""
    declared = {table.name for table in models.Base.metadata.sorted_tables}
    assert declared >= EXPECTED_TABLES


async def test_a_second_engine_on_the_same_directory_sees_the_same_schema(
    data_dir: Path,
) -> None:
    """Migrating twice is a no-op, which is what makes startup safe to repeat."""
    first = create_engine(data_dir)
    await upgrade(first)
    await first.dispose()

    second = create_engine(data_dir)
    await upgrade(second)
    try:
        assert await current(second) == head_revision()
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        async with session_factory(second)() as session:
            assert (await session.execute(select(models.Workspace))).scalars().all() == []
    finally:
        await second.dispose()
