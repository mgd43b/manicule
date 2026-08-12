"""Backup and restore, exercised rather than documented.

An open storage format is not a restore procedure. The test that matters is the one that
destroys a data directory and asks whether the same query gives the same answer afterwards.
"""

from __future__ import annotations

import json
import os
import stat
from typing import TYPE_CHECKING

import pytest

from manicule.core.errors import InsecureTargetError
from manicule.storage.backup import (
    MANIFEST_NAME,
    BackupError,
    create_backup,
    read_manifest,
    restore_backup,
    verify_backup,
)
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import DATABASE_FILENAME, create_engine, database_path
from manicule.storage.migrator import current, head_revision, upgrade
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


async def _populate(engine: AsyncEngine, data_dir: Path) -> tuple[SqliteDocStore, str]:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)

    body = b"# Auth\n\nthe service handles authentication and token rotation"
    stored = await blobs.put(body, "text/markdown")
    assert isinstance(stored, StoredBlob)

    document = make_document(body=body)
    document = document.model_copy(update={"original_ref": stored.hash})
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id,
        [
            make_chunk(document, 0, "the service handles authentication"),
            make_chunk(document, 1, "and token rotation happens hourly"),
        ],
    )
    return store, document.id


async def test_a_restored_instance_answers_a_query_identically(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """The whole point. Same ids, same scores — not "similar".

    The bytes are the bytes and the ranking is deterministic, so anything less than identical
    means the restore lost something.
    """
    store, _ = await _populate(engine, data_dir)
    before = [(c.chunk.id, c.score) for c in await store.search_lexical("authentication", k=5)]
    assert before

    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)
    await engine.dispose()

    restored_dir = tmp_path / "restored"
    restore_backup(backup_dir, restored_dir)

    restored_engine = create_engine(restored_dir)
    try:
        restored_store = SqliteDocStore(restored_engine)
        after = [
            (c.chunk.id, c.score)
            for c in await restored_store.search_lexical("authentication", k=5)
        ]
        assert after == before
    finally:
        await restored_engine.dispose()


async def test_the_manifest_records_what_is_needed_to_refuse_a_bad_restore(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """A manifest that omits the schema revision cannot detect a backup from the future."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    manifest = await create_backup(engine, data_dir, backup_dir)

    assert manifest["alembic_revision"] == head_revision()
    assert manifest["counts"]["documents"] == 1
    assert manifest["counts"]["chunks"] == 2
    assert manifest["counts"]["blobs"] == 1
    assert manifest["files"], "an inventory is what makes tampering detectable"


async def test_retained_bytes_survive_the_round_trip(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """A restore that loses the blob store has demoted every document to rung 4."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)
    await engine.dispose()

    restored_dir = tmp_path / "restored"
    restore_backup(backup_dir, restored_dir)
    restored_engine = create_engine(restored_dir)
    try:
        blobs = BlobStore(restored_engine, restored_dir)
        store = SqliteDocStore(restored_engine)
        documents = await store.list_documents()
        assert documents
        reference = documents[0].original_ref
        assert reference is not None
        assert await blobs.verify(reference), "every restored blob must hash to its own name"
    finally:
        await restored_engine.dispose()


async def test_a_tampered_backup_is_refused(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """Restoring altered content silently is worse than not restoring."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)

    database_path(backup_dir).write_bytes(b"not a database")
    with pytest.raises(BackupError, match="integrity check failed"):
        verify_backup(backup_dir)


async def test_a_backup_with_an_unlisted_file_is_refused(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """An unlisted file is either corruption or something added after the fact."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)

    (backup_dir / "extra.txt").write_text("added later")
    with pytest.raises(BackupError, match="inventory does not mention"):
        verify_backup(backup_dir)


async def test_a_backup_from_a_newer_schema_is_refused(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """Running old code against a newer schema corrupts it quietly.

    Refusing to start is the better failure.
    """
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)

    manifest = read_manifest(backup_dir)
    manifest["alembic_revision"] = "ffffffffffff"
    (backup_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(BackupError, match="does not know"):
        restore_backup(backup_dir, tmp_path / "restored")


async def test_restoring_over_existing_data_requires_saying_so(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """Silently replacing a populated directory is a data-loss bug wearing a feature's hat."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)
    await engine.dispose()

    with pytest.raises(BackupError, match="force=True"):
        restore_backup(backup_dir, data_dir)

    restore_backup(backup_dir, data_dir, force=True)


async def test_backing_up_into_a_non_empty_directory_is_refused(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """Merging a new snapshot into an old one produces a backup that restores neither."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    (backup_dir / "leftover").write_text("from an earlier run")

    with pytest.raises(BackupError, match="not empty"):
        await create_backup(engine, data_dir, backup_dir)


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_backing_up_into_a_pre_existing_world_readable_directory_is_refused(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """The directory already exists, which is the case ``mkdir(mode=...)`` never reaches.

    A test that creates a fresh target proves nothing here: ``mode`` is applied on creation
    and the umask can only clear bits, so a directory manicule made is ``0700`` whatever the
    code says. The exposure arrives with a directory that was already there — an operator's
    ``~/backups``, a mounted volume, a second run into the same place.
    """
    await _populate(engine, data_dir)
    target = tmp_path / "shared"
    target.mkdir()
    target.chmod(0o755)

    with pytest.raises(InsecureTargetError, match="group or other permissions") as refusal:
        await create_backup(engine, data_dir, target)

    assert str(target) in str(refusal.value), "an unnamed path sends an operator hunting"
    assert "055" in str(refusal.value), "the mode objected to is half the diagnosis"
    assert not (target / DATABASE_FILENAME).exists(), "a refused backup writes nothing"


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_a_group_readable_target_is_refused_too(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """Group is not a safer kind of other. Both mean accounts that are not this one."""
    await _populate(engine, data_dir)
    target = tmp_path / "team"
    target.mkdir()
    target.chmod(0o750)

    with pytest.raises(InsecureTargetError, match="group or other permissions"):
        await create_backup(engine, data_dir, target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_an_exposed_target_is_written_when_it_is_asked_for_in_so_many_words(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """The escape hatch exists, is off by default, and does not quietly chmod the operator.

    Someone backing up onto a shared volume has a reason. What they do not get is the
    decision made for them, in either direction.
    """
    await _populate(engine, data_dir)
    target = tmp_path / "shared"
    target.mkdir()
    target.chmod(0o755)

    manifest = await create_backup(engine, data_dir, target, allow_insecure_target=True)

    assert manifest["files"], "the snapshot is written, not merely permitted"
    verify_backup(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755, (
        "the target is left as the operator set it; silently tightening it is a different "
        "surprise, not an absent one"
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_a_target_manicule_creates_for_itself_is_not_readable_by_anyone_else(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """The other half of the check: the ordinary path still produces ``0700``.

    Adding a refusal that also refused the case it was meant to allow would be a worse bug
    than the one it fixed, and a suite full of tests about refusals would not have noticed.
    """
    await _populate(engine, data_dir)
    target = tmp_path / "fresh"

    await create_backup(engine, data_dir, target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    verify_backup(target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_the_snapshot_database_is_not_left_at_the_umask(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """``sqlite3.connect`` creates at the umask, which is commonly ``0644``.

    The directory being ``0700`` is what gates it, but a backup copied onward file by file —
    to a share, into an archive — carries the mode of the file, not of the directory it used
    to sit in.
    """
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)

    assert stat.S_IMODE((backup_dir / DATABASE_FILENAME).stat().st_mode) == 0o600
    assert stat.S_IMODE((backup_dir / MANIFEST_NAME).stat().st_mode) == 0o600


async def test_backing_up_into_the_directory_being_backed_up_is_refused(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """The copy would include itself, and would not terminate usefully."""
    await _populate(engine, data_dir)
    with pytest.raises(BackupError, match="inside the data directory"):
        await create_backup(engine, data_dir, data_dir / "snapshots" / "one")


async def test_a_restored_database_is_at_the_same_migration_revision(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """A restore that lands at a different revision is a different database."""
    await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"
    await create_backup(engine, data_dir, backup_dir)
    await engine.dispose()

    restored_dir = tmp_path / "restored"
    restore_backup(backup_dir, restored_dir)
    restored_engine = create_engine(restored_dir)
    try:
        assert await current(restored_engine) == head_revision()
    finally:
        await restored_engine.dispose()


async def test_backing_up_a_directory_that_is_not_one_is_refused(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """A clear message beats a snapshot of nothing."""
    with pytest.raises(BackupError, match="not a manicule data directory"):
        await create_backup(engine, tmp_path / "empty", tmp_path / "backup")


async def test_a_hot_backup_stays_consistent_while_writes_continue(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    """SQLite is snapshotted first, so the derived stores can only ever be ahead of it.

    Extra vectors are inert because retrieval hydrates through a join. Missing ones would be
    a document marked indexed that silently returns nothing.
    """
    store, _ = await _populate(engine, data_dir)
    backup_dir = tmp_path / "backup"

    manifest = await create_backup(engine, data_dir, backup_dir)

    # A write landing after the snapshot must not appear in it.
    later = make_document(source_id="written-after")
    await store.upsert_document(later)
    await store.replace_chunks(later.id, [make_chunk(later, 0, "arrived after the snapshot")])

    verify_backup(backup_dir)
    assert manifest["counts"]["documents"] == 1

    restored_dir = tmp_path / "restored"
    restore_backup(backup_dir, restored_dir)
    restored_engine = create_engine(restored_dir)
    try:
        restored = SqliteDocStore(restored_engine)
        assert len(await restored.list_documents()) == 1
        assert await restored.search_lexical("arrived", k=5) == []
    finally:
        await restored_engine.dispose()


async def test_a_database_that_was_never_migrated_still_backs_up(tmp_path: Path) -> None:
    """A diagnostic backup of a broken instance is exactly when one is wanted."""
    fresh_dir = tmp_path / "fresh"
    fresh_engine = create_engine(fresh_dir)
    await upgrade(fresh_engine)
    try:
        manifest = await create_backup(fresh_engine, fresh_dir, tmp_path / "backup")
        assert manifest["counts"]["documents"] == 0
    finally:
        await fresh_engine.dispose()
