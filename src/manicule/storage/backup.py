"""Backup and restore, as a procedure rather than a claim.

An open storage format is not a restore procedure. Two stores with no shared transaction have
a consistency problem, and the ordering that solves it is the opposite of the intuitive one.

**SQLite first, vectors second.** SQLite is authoritative and the vector store may be a
superset of it, so the safe skew is the derived store being *ahead*: extra vectors are inert,
because retrieval hydrates through a join that cannot see them. Vectors that are *missing*
mean a document marked ``indexed`` that silently returns nothing.

**Never copy the database file.** ``manicule.db``, ``-wal`` and ``-shm`` copied one at a time
are three files captured at three instants, and a checkpoint landing between them produces a
result that will not open. :meth:`sqlite3.Connection.backup` is in the standard library and is
correct against a live database.

**A backup is a second copy of the corpus, and it is checked as one.** With retained source
bytes a snapshot is byte-identical to what the connectors fetched (``docs/storage.md`` §7.1),
so where it lands is a security decision rather than a filing one. :func:`create_backup`
refuses a group- or world-readable target — see
:func:`manicule.storage.engine.secure_output_dir`, which `export` calls too, for why asking
for a mode is not the same as having one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from manicule.core.errors import ManiculeError
from manicule.storage import models
from manicule.storage.engine import (
    BLOBS_DIRNAME,
    VECTORS_DIRNAME,
    database_path,
    prepare_data_dir,
    secure_output_dir,
)
from manicule.storage.migrator import current, head_revision
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

MANIFEST_NAME = "backup-manifest.json"
BACKUP_FORMAT = "manicule-backup"
BACKUP_VERSION = 1


class BackupError(ManiculeError):
    """A backup could not be created, or a restore refused to proceed.

    A :class:`~manicule.core.errors.ManiculeError` because every one of these is a refusal
    manicule decided on, not a defect: an unusable target, a tampered snapshot, a backup from
    a newer schema. Only errors in that hierarchy reach
    :func:`~manicule.app.dispatch.run_op`'s envelope — anything else propagates as a
    traceback, and a security refusal that reaches an operator as a stack trace is a refusal
    they will read as a crash.
    """


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One file in a snapshot, with what it should hash to."""

    path: str
    size: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for path in sorted(root.rglob("*")):
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            msg = f"refusing to inventory a symbolic link: {path}"
            raise BackupError(msg)
        if path.is_file():
            entries.append(
                FileEntry(
                    path=path.relative_to(root).as_posix(),
                    size=path.stat().st_size,
                    sha256=_sha256(path),
                )
            )
    return entries


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)


async def create_backup(
    engine: AsyncEngine,
    data_dir: Path,
    target: Path,
    *,
    allow_insecure_target: bool = False,
) -> dict[str, Any]:
    """Snapshot a data directory into ``target``, and return the manifest.

    The order is load-bearing and is argued in the module docstring. Callers must hold
    whatever lock stops the vector sweep and the blob GC — those are the only two operations
    that remove data, and they are the only two a hot backup has to exclude.

    Args:
        engine: The live engine, used to read counts and the migration revision.
        data_dir: What to back up.
        target: An empty or absent directory to write the snapshot into. Group- or
            world-readable targets are refused; see
            :func:`manicule.storage.engine.secure_output_dir`.
        allow_insecure_target: Write into a group- or world-readable target anyway. Off by
            default, and the default is the point.

    Returns:
        The manifest, as written.

    Raises:
        BackupError: The target is unusable, or the database is missing.
        InsecureTargetError: The target is group- or world-readable and
            ``allow_insecure_target`` was not given. A separate type because it describes the
            destination rather than the backup, and ``export`` raises the same one.
    """
    revision = await current(engine)
    counts = await _counts(engine)
    index_state = await _index_state(engine)
    # The filesystem work is blocking, and a backup copies the whole corpus. Running it on the
    # event loop would stall every concurrent query for the duration.
    return await asyncio.to_thread(
        _write_snapshot,
        data_dir,
        target,
        revision,
        counts,
        index_state,
        allow_insecure_target=allow_insecure_target,
    )


def _write_snapshot(
    data_dir: Path,
    target: Path,
    revision: str | None,
    counts: dict[str, int],
    index_state: dict[str, str | None],
    *,
    allow_insecure_target: bool,
) -> dict[str, Any]:
    """The blocking half of :func:`create_backup`, in snapshot order."""
    source_db = database_path(data_dir)
    if not source_db.exists():
        msg = f"no database at {source_db}; this is not a manicule data directory"
        raise BackupError(msg)
    if target.exists() and any(target.iterdir()):
        msg = f"backup target is not empty: {target}"
        raise BackupError(msg)
    resolved_target = target.resolve()
    resolved_source = data_dir.resolve()
    if resolved_target == resolved_source or resolved_source in resolved_target.parents:
        msg = (
            f"backup target {target} is inside the data directory being backed up; "
            f"the copy would include itself"
        )
        raise BackupError(msg)

    secure_output_dir(target, operation="backup", allow_insecure=allow_insecure_target)

    # 1. SQLite first. The online backup API produces one consistent file from a live
    #    database; copying the three files would not.
    snapshot_db = target / source_db.name
    source = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        snapshot = sqlite3.connect(snapshot_db)
        try:
            source.backup(snapshot)
        finally:
            snapshot.close()
    finally:
        source.close()
    # sqlite3 creates its file at the invoking shell's umask, which is commonly 0644 — the
    # whole index, every chunk of extracted text, readable by anyone who can reach it. The
    # copied trees keep their source modes and the manifest sets its own; this is the one
    # file in a snapshot that would otherwise be looser than what it came from.
    snapshot_db.chmod(0o600)

    # 2. Derived stores second, so they can only ever be ahead of the snapshot above.
    _copy_tree(data_dir / VECTORS_DIRNAME, target / VECTORS_DIRNAME)
    _copy_tree(data_dir / BLOBS_DIRNAME, target / BLOBS_DIRNAME)

    manifest: dict[str, Any] = {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "created_at": utcnow().isoformat(),
        "alembic_revision": revision,
        "embed_fingerprint": index_state.get("embed_fingerprint"),
        "chunk_fingerprint": index_state.get("chunk_fingerprint"),
        "fts_tokenizer": index_state.get("fts_tokenizer"),
        "vector_table": index_state.get("vector_table"),
        "counts": counts,
        "files": [asdict(entry) for entry in _inventory(target)],
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (target / MANIFEST_NAME).chmod(0o600)
    return manifest


def read_manifest(backup_dir: Path) -> dict[str, Any]:
    """Load and shape-check a manifest.

    Raises:
        BackupError: Absent, malformed, or not a manicule backup.
    """
    path = backup_dir / MANIFEST_NAME
    if not path.exists():
        msg = f"not a manicule backup: {MANIFEST_NAME} is missing from {backup_dir}"
        raise BackupError(msg)
    try:
        manifest: dict[str, Any] = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        msg = f"backup manifest is malformed: {error}"
        raise BackupError(msg) from error
    if manifest.get("format") != BACKUP_FORMAT or manifest.get("version") != BACKUP_VERSION:
        msg = (
            f"unsupported backup format {manifest.get('format')!r} "
            f"version {manifest.get('version')!r}"
        )
        raise BackupError(msg)
    return manifest


def verify_backup(backup_dir: Path) -> None:
    """Check every inventoried file against its recorded hash and size.

    Also refuses a snapshot containing files the inventory does not mention, because an
    unlisted file is either corruption or something added after the fact — and a restore that
    installs unknown content is not a restore.

    Raises:
        BackupError: Any file is missing, altered, or unlisted.
    """
    manifest = read_manifest(backup_dir)
    listed: dict[str, dict[str, Any]] = {
        str(entry["path"]): entry for entry in manifest.get("files", [])
    }
    if not listed:
        msg = "backup inventory is empty"
        raise BackupError(msg)

    for relative, entry in listed.items():
        path = backup_dir / relative
        if not path.is_file():
            msg = f"backup is missing an inventoried file: {relative}"
            raise BackupError(msg)
        if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
            msg = f"backup integrity check failed for {relative}"
            raise BackupError(msg)

    actual = {entry.path for entry in _inventory(backup_dir)}
    unlisted = sorted(actual - set(listed))
    if unlisted:
        msg = f"backup contains files the inventory does not mention: {unlisted}"
        raise BackupError(msg)


def restore_backup(backup_dir: Path, data_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Replace ``data_dir`` with a verified snapshot.

    Refuses when the snapshot was written by a newer schema than this build knows: running old
    code against a newer schema corrupts it quietly, which is worse than not starting.

    Args:
        backup_dir: A snapshot produced by :func:`create_backup`.
        data_dir: Where to restore to.
        force: Replace existing data. Without it, a populated directory is refused.

    Returns:
        The manifest that was restored.

    Raises:
        BackupError: Verification failed, the revision is unknown, or data exists and
            ``force`` was not given.
    """
    verify_backup(backup_dir)
    manifest = read_manifest(backup_dir)

    revision = manifest.get("alembic_revision")
    if revision is not None and revision != head_revision():
        known = _known_revisions()
        if revision not in known:
            msg = (
                f"backup was written at Alembic revision {revision}, which this build does "
                f"not know. It is from a newer manicule; upgrade before restoring."
            )
            raise BackupError(msg)

    existing = database_path(data_dir)
    if existing.exists() and not force:
        msg = f"{data_dir} already holds a database; pass force=True to replace it"
        raise BackupError(msg)

    prepare_data_dir(data_dir)
    for name in (existing.name, f"{existing.name}-wal", f"{existing.name}-shm"):
        (data_dir / name).unlink(missing_ok=True)
    for dirname in (VECTORS_DIRNAME, BLOBS_DIRNAME):
        shutil.rmtree(data_dir / dirname, ignore_errors=True)

    shutil.copy2(backup_dir / existing.name, existing)
    _copy_tree(backup_dir / VECTORS_DIRNAME, data_dir / VECTORS_DIRNAME)
    _copy_tree(backup_dir / BLOBS_DIRNAME, data_dir / BLOBS_DIRNAME)
    return manifest


def _known_revisions() -> set[str]:
    from alembic.script import ScriptDirectory  # noqa: PLC0415 - only needed on this path

    from manicule.storage.migrator import alembic_config  # noqa: PLC0415

    script = ScriptDirectory.from_config(alembic_config())
    return {revision.revision for revision in script.walk_revisions()}


async def _counts(engine: AsyncEngine) -> dict[str, int]:
    from manicule.storage.engine import session_factory  # noqa: PLC0415 - avoids a cycle

    async with session_factory(engine)() as session:
        return {
            "documents": (
                await session.execute(select(func.count()).select_from(models.Document))
            ).scalar_one(),
            "chunks": (
                await session.execute(select(func.count()).select_from(models.Chunk))
            ).scalar_one(),
            "blobs": (
                await session.execute(select(func.count()).select_from(models.Blob))
            ).scalar_one(),
        }


async def _index_state(engine: AsyncEngine) -> dict[str, str | None]:
    from manicule.storage.engine import session_factory  # noqa: PLC0415 - avoids a cycle

    async with session_factory(engine)() as session:
        row = await session.get(models.IndexState, 1)
    if row is None:
        return {}
    return {
        "embed_fingerprint": row.embed_fingerprint,
        "chunk_fingerprint": row.chunk_fingerprint,
        "fts_tokenizer": row.fts_tokenizer,
        "vector_table": row.vector_table,
    }


def manifest_created_at(manifest: dict[str, Any]) -> datetime:
    """Parse the manifest timestamp back into an aware datetime."""
    return datetime.fromisoformat(str(manifest["created_at"]))


__all__ = [
    "BACKUP_FORMAT",
    "BACKUP_VERSION",
    "MANIFEST_NAME",
    "BackupError",
    "FileEntry",
    "create_backup",
    "manifest_created_at",
    "read_manifest",
    "restore_backup",
    "verify_backup",
]
