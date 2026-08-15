"""Retained original bytes, so re-parsing never means re-fetching.

This is rung 3 of the blast-radius ladder, and the only thing standing between a parser bug
fix and a full re-crawl of a rate-limited API. Every other rung is a pure function of what is
already on disk; re-fetching is the one repair that can fail for reasons outside the machine,
and the one whose result is not reproducible.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from sqlalchemy import and_, delete, select, union
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.acquisition import AcquiredSource
from manicule.core.content import RawDocument, Retention
from manicule.core.ids import content_hash
from manicule.storage import models
from manicule.storage.engine import BLOBS_DIRNAME, session_factory

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

MAX_ORIGINAL_BYTES = 256 * 1024 * 1024
"""Above this, the bytes are not retained and the reason is recorded.

A four-gigabyte video attachment should not silently double the data directory. The document
still exists, with ``original_ref`` unset and ``original_omitted_reason`` saying why — absent
with a stated reason, visible in diagnostics, never a silent partial success.
"""

_COMPRESSIBLE_PREFIXES = ("text/", "application/json", "application/xml")
_COMPRESSIBLE_SUFFIXES = ("+json", "+xml")
STAGING_PARTIAL_STALE_SECONDS = 24 * 60 * 60
STAGING_PARTIAL_CLEANUP_LIMIT = 1_000
STAGING_MARKER_RECONCILE_LIMIT = 100
LEGACY_MARKER_RETENTION = timedelta(days=30)


@dataclass(frozen=True, slots=True)
class StoredBlob:
    """What was retained, and what it cost."""

    hash: str
    size_bytes: int
    stored_bytes: int
    compression: str


@dataclass(frozen=True, slots=True)
class OmittedBlob:
    """What was not retained, and why."""

    reason: str


@dataclass(frozen=True, slots=True)
class StagingCleanup:
    """Bounded, privacy-preserving result of abandoned staging-file cleanup."""

    scanned: int
    removed: int
    truncated: bool


def should_compress(media_type: str | None) -> bool:
    """Whether this media type is worth compressing.

    Text compresses; a PDF or a JPEG is already compressed and gzipping it spends CPU to grow
    the file slightly.
    """
    if not media_type:
        return False
    lowered = media_type.split(";")[0].strip().lower()
    return lowered.startswith(_COMPRESSIBLE_PREFIXES) or lowered.endswith(_COMPRESSIBLE_SUFFIXES)


class BlobStore:
    """Content-addressed storage for original source bytes.

    Immutable and sharded. Immutability is what makes the directory safe to copy at any moment
    with any tool, and content addressing is what makes verification free: a blob whose bytes
    do not hash to its own name is corrupt, and that is checkable without a reference copy.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        data_dir: Path,
        *,
        max_bytes: int = MAX_ORIGINAL_BYTES,
        sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._engine = engine
        self._root = data_dir / BLOBS_DIRNAME
        self._max_bytes = max_bytes
        self._sessions = sessions or session_factory(engine)
        self._staging_cleanup_lock = asyncio.Lock()
        self._staging_cleanup_complete = False
        self._legacy_scan: Iterator[os.DirEntry[str]] | None = None
        self._legacy_scan_complete = False
        self._marker_cursor = ""

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, digest: str) -> Path:
        """Where a blob lives. Sharded two levels, so no directory holds the whole corpus."""
        return self._root / "blake2b" / digest[:2] / digest[2:4] / digest

    def _stage_path(self, key: str) -> Path:
        return self._root / "acquisition-staging" / self._stage_name(key)

    @staticmethod
    def _stage_name(key: str) -> str:
        return hashlib.blake2b(key.encode(), digest_size=20).hexdigest()

    def _stage_partial_root(self) -> Path:
        return self._root / "acquisition-staging-partials"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _mkdir_durable(cls, path: Path) -> None:
        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            # A peer may win creation and die before syncing the new name. This contender
            # must still certify the ancestor before it builds beneath that directory.
            with contextlib.suppress(FileExistsError):
                directory.mkdir(mode=0o700)
            cls._fsync_directory(directory.parent)

    @classmethod
    def _write_durable(
        cls,
        destination: Path,
        payload: bytes,
        *,
        temporary_dir: Path | None = None,
    ) -> None:
        cls._mkdir_durable(destination.parent)
        temporary_parent = temporary_dir or destination.parent
        cls._mkdir_durable(temporary_parent)
        temporary = temporary_parent / f"{destination.name}.{uuid4().hex}.partial"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
            cls._fsync_directory(destination.parent)
            if temporary_parent != destination.parent:
                cls._fsync_directory(temporary_parent)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _publish_durable(cls, destination: Path, payload: bytes) -> bytes:
        """Publish once across processes and return the representation that won.

        A hard link is the POSIX no-clobber analog of an atomic rename: exactly one
        temporary inode can acquire ``destination``. Every contender then syncs the parent
        before it may create a database reference, including one that observed another
        process's newly linked name before that process reached its own directory sync.
        """
        cls._mkdir_durable(destination.parent)
        temporary = destination.with_name(f"{destination.name}.{uuid4().hex}.partial")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            with contextlib.suppress(FileExistsError):
                os.link(temporary, destination)
            cls._fsync_directory(destination.parent)
            return destination.read_bytes()
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    async def _durable_thread_call[T](call: Callable[[], T]) -> T:
        """Join an irreversible thread call before propagating task cancellation."""
        work = asyncio.create_task(asyncio.to_thread(call))
        current = asyncio.current_task()
        cancellation: asyncio.CancelledError | None = None
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError as error:
                # Cancellation cannot stop an OS thread. Repeated requests still wait for
                # this irreversible call to reach a known endpoint before returning. Clear the
                # request while joining so the next shield can suspend instead of spinning.
                cancellation = error
                if current is not None:
                    current.uncancel()
        # A durability failure is more informative than the cancellation that happened while
        # it was in flight. Only restore cancellation after a successful, known endpoint.
        result = work.result()
        if cancellation is not None:
            raise cancellation
        return result

    @classmethod
    def _published_blob(
        cls, destination: Path, data: bytes, digest: str, compression: str
    ) -> StoredBlob:
        proposed = gzip.compress(data, mtime=0) if compression == "gzip" else data
        published = cls._publish_durable(destination, proposed)
        if published == data:
            actual_compression = "none"
        else:
            try:
                decoded = gzip.decompress(published)
            except (OSError, EOFError) as error:
                msg = f"retained blob {digest} has an unrecognized representation"
                raise OSError(msg) from error
            if decoded != data:
                msg = f"retained blob {digest} does not match its content address"
                raise OSError(msg)
            actual_compression = "gzip"
        return StoredBlob(
            hash=digest,
            size_bytes=len(data),
            stored_bytes=len(published),
            compression=actual_compression,
        )

    async def _record_blob(self, stored: StoredBlob, media_type: str | None) -> None:
        """Make SQLite describe the immutable representation that is already on disk."""
        async with self._sessions.begin() as session:
            await session.execute(
                sqlite_insert(models.Blob)
                .values(
                    hash=stored.hash,
                    algo="blake2b",
                    media_type=media_type,
                    size_bytes=stored.size_bytes,
                    stored_bytes=stored.stored_bytes,
                    compression=stored.compression,
                )
                .on_conflict_do_update(
                    index_elements=[models.Blob.hash],
                    set_={
                        "size_bytes": stored.size_bytes,
                        "stored_bytes": stored.stored_bytes,
                        "compression": stored.compression,
                    },
                )
            )

    async def _store_blob(self, data: bytes, media_type: str | None) -> StoredBlob:
        digest = content_hash(data)
        compression = "gzip" if should_compress(media_type) else "none"
        stored = await self._durable_thread_call(
            lambda: self._published_blob(self.path_for(digest), data, digest, compression)
        )
        await self._record_blob(stored, media_type)
        return stored

    async def put(self, data: bytes, media_type: str | None = None) -> StoredBlob | OmittedBlob:
        """Retain bytes, or say why they were not retained.

        **Content first, reference second.** The file is written, fsynced and renamed into
        place, and only then does the row appear — so a ``documents.original_ref`` always
        resolves. Deletion runs the ordering in reverse for the same reason.

        Args:
            data: The bytes exactly as the connector delivered them.
            media_type: Used to decide whether compressing is worth the CPU.

        Returns:
            :class:`StoredBlob` when retained, :class:`OmittedBlob` when the size cap refused
            it.
        """
        if len(data) > self._max_bytes:
            return OmittedBlob(
                reason=(
                    f"original bytes not retained: {len(data)} exceeds the "
                    f"{self._max_bytes}-byte retention cap"
                )
            )

        return await self._store_blob(data, media_type)

    async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
        """:meth:`put`, expressed in the vocabulary the ingest pipeline speaks.

        The pipeline records a ``Retention`` on the document either way — a reference or the
        reason there is none — and it must not have to know which of two storage-side classes
        came back to do it. ``Retention`` lives in core precisely so neither side imports the
        other.
        """
        stored = await self.put(data, media_type)
        if isinstance(stored, OmittedBlob):
            return Retention(omitted_reason=stored.reason)
        return Retention(ref=stored.hash)

    async def retain_acquisition(
        self, key: str, raw: RawDocument
    ) -> tuple[Retention, AcquiredSource]:
        """Durably stage a complete source envelope so a pre-association crash can resume."""
        await self._cleanup_staging_once()
        acquired = AcquiredSource.from_raw(raw)
        data = raw.as_bytes()
        if len(data) > self._max_bytes:
            return await self.retain(data, raw.media_type), acquired
        compression = "gzip" if should_compress(raw.media_type) else "none"
        stored = await self._durable_thread_call(
            lambda: self._published_blob(
                self.path_for(acquired.content_hash),
                data,
                acquired.content_hash,
                compression,
            )
        )
        run_id, separator, source_id = key.partition("\0")
        marker: dict[str, object] = {
            "blob_ref": acquired.content_hash,
            "compression": stored.compression,
            "acquired_source": acquired.model_dump(mode="json"),
            "run_id": run_id if separator else None,
            "source_id": source_id if separator else None,
        }
        await self._record_marker(self._stage_name(key), marker, legacy=False)
        payload = json.dumps(
            marker,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        try:
            await self._durable_thread_call(
                lambda: self._write_durable(
                    self._stage_path(key), payload, temporary_dir=self._stage_partial_root()
                )
            )
        except BaseException:
            # Cancellation is propagated only after the joined thread reaches a known endpoint.
            # If it published the marker, its inventory must survive the canceled coroutine.
            if not self._stage_path(key).exists():
                await self._forget_markers([self._stage_name(key)])
            raise
        await self._record_blob(stored, raw.media_type)
        return Retention(ref=stored.hash), acquired

    async def _record_marker(
        self,
        name: str,
        payload: dict[str, object],
        *,
        legacy: bool,
        created_at: datetime | None = None,
    ) -> None:
        await self._record_markers([(name, payload, legacy, created_at or datetime.now(UTC))])

    async def _record_markers(
        self,
        markers: Sequence[tuple[str, dict[str, object], bool, datetime]],
    ) -> None:
        if not markers:
            return
        async with self._sessions.begin() as session:
            await session.execute(
                sqlite_insert(models.AcquisitionMarker)
                .values(
                    [
                        {
                            "name": name,
                            "run_id": payload.get("run_id"),
                            "source_id": payload.get("source_id"),
                            "blob_ref": payload.get("blob_ref"),
                            "acquired_source": payload.get("acquired_source"),
                            "legacy": legacy,
                            "created_at": created_at,
                        }
                        for name, payload, legacy, created_at in markers
                    ]
                )
                .on_conflict_do_nothing(index_elements=[models.AcquisitionMarker.name])
            )

    async def _forget_markers(self, names: Sequence[str]) -> None:
        if not names:
            return
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.AcquisitionMarker).where(models.AcquisitionMarker.name.in_(names))
            )

    async def cleanup_staging_partials(
        self,
        *,
        stale_after_seconds: float = STAGING_PARTIAL_STALE_SECONDS,
        limit: int = STAGING_PARTIAL_CLEANUP_LIMIT,
    ) -> StagingCleanup:
        """Remove only old abandoned staging writes, reporting aggregate counts."""
        root = self._stage_partial_root()
        if not root.exists() or limit <= 0:
            return StagingCleanup(scanned=0, removed=0, truncated=False)
        cutoff = time.time() - stale_after_seconds
        # This directory contains only temporary writes, separate from live recovery markers,
        # so ``limit + 1`` bounds directory traversal as well as deletions and report size.
        inspected = list(islice(root.iterdir(), limit + 1))
        truncated = len(inspected) > limit
        candidates = [path for path in inspected[:limit] if path.is_file()]
        removed = 0
        for path in candidates:
            try:
                if path.stat().st_mtime <= cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
        if removed:
            await self._durable_thread_call(lambda: self._fsync_directory(root))
        return StagingCleanup(
            scanned=min(len(inspected), limit), removed=removed, truncated=truncated
        )

    async def _cleanup_staging_once(self) -> None:
        """Run startup cleanup once rather than rescanning all markers for every document."""
        if self._staging_cleanup_complete:
            return
        async with self._staging_cleanup_lock:
            if self._staging_cleanup_complete:
                return
            await self.cleanup_staging_partials()
            await self.reconcile_acquisition_markers()
            self._staging_cleanup_complete = True

    async def reconcile_acquisition_markers(self) -> bool:
        """Advance one bounded legacy-scan and indexed-reconciliation page.

        ``False`` means legacy files remain unindexed, so callers must defer history deletion
        and blob collection. Indexed markers themselves block their run in the cleanup query.
        """
        inventory_complete = await self._index_legacy_marker_page()
        await self._reconcile_marker_page()
        return inventory_complete

    async def _index_legacy_marker_page(self) -> bool:  # noqa: PLR0912 - validates legacy data
        root = self._root / "acquisition-staging"
        if not root.exists():
            self._legacy_scan_complete = True
            return True
        if self._legacy_scan_complete:
            return True
        if self._legacy_scan is None:
            self._legacy_scan = os.scandir(root)
        entries = list(islice(self._legacy_scan, STAGING_MARKER_RECONCILE_LIMIT))
        if not entries:
            close = getattr(self._legacy_scan, "close", None)
            if close is not None:
                close()
            self._legacy_scan = None
            self._legacy_scan_complete = True
            return True
        names = [entry.name for entry in entries if entry.is_file()]
        async with self._sessions() as session:
            known = set(
                (
                    await session.execute(
                        select(models.AcquisitionMarker.name).where(
                            models.AcquisitionMarker.name.in_(names)
                        )
                    )
                )
                .scalars()
                .all()
            )
        parsed: list[tuple[os.DirEntry[str], dict[str, object]]] = []
        source_ids: set[str] = set()
        for entry in entries:
            if not entry.is_file() or entry.name in known:
                continue
            try:
                raw = json.loads(await asyncio.to_thread(Path(entry.path).read_text, "utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                raw = {}
            payload = cast("dict[str, object]", raw) if isinstance(raw, dict) else {}
            acquired = payload.get("acquired_source")
            acquired_mapping = (
                cast("dict[str, object]", acquired) if isinstance(acquired, dict) else {}
            )
            if isinstance(acquired_mapping.get("source_id"), str):
                source_ids.add(cast("str", acquired_mapping["source_id"]))
            parsed.append((entry, payload))
        inferred: dict[str, tuple[str, str]] = {}
        if source_ids:
            async with self._sessions() as session:
                records = (
                    (
                        await session.execute(
                            select(
                                models.AcquisitionRecord.run_id,
                                models.AcquisitionRecord.source_id,
                            ).where(models.AcquisitionRecord.source_id.in_(source_ids))
                        )
                    )
                    .tuples()
                    .all()
                )
            inferred = {
                self._stage_name(f"{run_id}\0{source_id}"): (run_id, source_id)
                for run_id, source_id in records
            }
        markers: list[tuple[str, dict[str, object], bool, datetime]] = []
        for entry, payload in parsed:
            explicit_run = payload.get("run_id")
            explicit_source = payload.get("source_id")
            identity = (
                (explicit_run, explicit_source)
                if isinstance(explicit_run, str) and isinstance(explicit_source, str)
                else inferred.get(entry.name)
            )
            if identity is not None:
                payload["run_id"], payload["source_id"] = identity
            try:
                created_at = datetime.fromtimestamp(entry.stat().st_mtime, UTC)
            except FileNotFoundError:
                continue
            markers.append((entry.name, payload, True, created_at))
        await self._record_markers(markers)
        return False

    async def _reconcile_marker_page(self) -> None:
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(
                        models.AcquisitionMarker,
                        models.AcquisitionRun.superseded_at,
                        models.AcquisitionRecord.blob_ref,
                        models.AcquisitionRecord.acquired_source,
                    )
                    .outerjoin(
                        models.AcquisitionRun,
                        models.AcquisitionRun.id == models.AcquisitionMarker.run_id,
                    )
                    .outerjoin(
                        models.AcquisitionRecord,
                        and_(
                            models.AcquisitionRecord.run_id == models.AcquisitionMarker.run_id,
                            models.AcquisitionRecord.source_id
                            == models.AcquisitionMarker.source_id,
                        ),
                    )
                    .where(models.AcquisitionMarker.name > self._marker_cursor)
                    .order_by(models.AcquisitionMarker.name)
                    .limit(STAGING_MARKER_RECONCILE_LIMIT)
                )
            ).all()
        if not rows:
            self._marker_cursor = ""
            return
        remove: list[str] = []
        cutoff = datetime.now(UTC) - LEGACY_MARKER_RETENTION
        for marker, superseded_at, record_blob_ref, record_acquired_source in rows:
            path = self._root / "acquisition-staging" / marker.name
            exact_association = (
                marker.blob_ref is not None
                and marker.blob_ref == record_blob_ref
                and cast("object", marker.acquired_source) == record_acquired_source
            )
            expired_orphan = marker.run_id is None and marker.created_at < cutoff
            removable = superseded_at is not None or exact_association or expired_orphan
            if path.exists() and not removable:
                continue
            await self._durable_thread_call(lambda path=path: self._unlink_durable(path))
            remove.append(marker.name)
        await self._forget_markers(remove)
        self._marker_cursor = rows[-1][0].name

    async def resume_acquisition(self, key: str) -> tuple[Retention, AcquiredSource] | None:
        """Recover a staged blob/envelope pair without contacting its source."""
        await self._cleanup_staging_once()
        path = self._stage_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
            acquired = AcquiredSource.model_validate(payload["acquired_source"])
            blob_ref = payload["blob_ref"]
            compression = payload["compression"]
            if not isinstance(blob_ref, str) or blob_ref != acquired.content_hash:
                return None
            if compression not in {"gzip", "none"}:
                return None
            raw = await asyncio.to_thread(self.path_for(blob_ref).read_bytes)
            data = gzip.decompress(raw) if compression == "gzip" else raw
            acquired.raw(data)
            retained = await self.retain(data, acquired.media_type)
            if retained.ref != blob_ref:
                return None
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return retained, acquired

    async def complete_acquisition(self, key: str) -> None:
        """Remove a staging marker only after the journal association commits."""
        path = self._stage_path(key)
        await self._durable_thread_call(lambda: self._unlink_durable(path))
        await self._forget_markers([path.name])

    @classmethod
    def _unlink_durable(cls, path: Path) -> None:
        """Unlink a name and durably record its absence as one joined operation."""
        try:
            path.unlink()
        except FileNotFoundError:
            if not path.parent.exists():
                return
        cls._fsync_directory(path.parent)

    async def get(self, digest: str) -> bytes | None:
        """Read retained bytes back, or ``None`` if they are not held."""
        async with self._sessions() as session:
            row = await session.get(models.Blob, digest)
        if row is None:
            return None
        path = self.path_for(digest)
        if not path.exists():
            return None
        raw = path.read_bytes()
        return gzip.decompress(raw) if row.compression == "gzip" else raw

    async def verify(self, digest: str) -> bool:
        """Whether the stored bytes still hash to their own name."""
        data = await self.get(digest)
        return data is not None and content_hash(data) == digest

    async def collect_garbage(self) -> Sequence[str]:
        """Delete blobs nothing references. Mark and sweep, never refcounts.

        A refcount is a number that has to survive every crash in every path that touches it,
        and when it is wrong it is wrong silently in both directions.

        The row goes first and the file second. Crashing between the two leaks a file, which a
        directory scan reclaims on the next pass; the reverse order would leave a row pointing
        at nothing. Every ordering decision here resolves the same way — prefer the failure
        that costs space over the failure that costs correctness.

        Returns:
            The hashes that were collected.
        """
        await self.cleanup_staging_partials()
        if not await self.reconcile_acquisition_markers():
            return []
        async with self._sessions() as session:
            referenced_documents = select(models.Document.original_ref).where(
                models.Document.original_ref.is_not(None)
            )
            referenced_versions = select(models.DocumentVersion.original_ref).where(
                models.DocumentVersion.original_ref.is_not(None)
            )
            referenced_acquisitions = select(models.AcquisitionRecord.blob_ref).where(
                models.AcquisitionRecord.blob_ref.is_not(None)
            )
            referenced_markers = select(models.AcquisitionMarker.blob_ref).where(
                models.AcquisitionMarker.blob_ref.is_not(None)
            )
            unreferenced = (
                (
                    await session.execute(
                        select(models.Blob.hash).where(
                            models.Blob.hash.not_in(
                                union(
                                    referenced_documents,
                                    referenced_versions,
                                    referenced_acquisitions,
                                    referenced_markers,
                                )
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

        collected: list[str] = []
        for digest in unreferenced:
            async with self._sessions.begin() as session:
                await session.execute(delete(models.Blob).where(models.Blob.hash == digest))
            self.path_for(digest).unlink(missing_ok=True)
            collected.append(digest)
        return collected

    async def orphaned_files(self) -> Sequence[Path]:
        """Files on disk that no ``blobs`` row claims.

        The other half of the sweep: a crash between deleting the row and unlinking the file
        leaks the file, and this is what finds it.
        """
        if not self._root.exists():
            return []
        async with self._sessions() as session:
            known = set((await session.execute(select(models.Blob.hash))).scalars().all())
        blob_root = self._root / "blake2b"
        if not blob_root.exists():
            return []
        return [path for path in blob_root.rglob("*") if path.is_file() and path.name not in known]


__all__ = [
    "MAX_ORIGINAL_BYTES",
    "BlobStore",
    "OmittedBlob",
    "StagingCleanup",
    "StoredBlob",
    "should_compress",
]
