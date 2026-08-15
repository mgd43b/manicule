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
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from sqlalchemy import delete, select, union
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.acquisition import AcquiredSource
from manicule.core.content import RawDocument, Retention
from manicule.core.ids import content_hash
from manicule.storage import models
from manicule.storage.engine import BLOBS_DIRNAME, session_factory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, digest: str) -> Path:
        """Where a blob lives. Sharded two levels, so no directory holds the whole corpus."""
        return self._root / "blake2b" / digest[:2] / digest[2:4] / digest

    def _stage_path(self, key: str) -> Path:
        name = hashlib.blake2b(key.encode(), digest_size=20).hexdigest()
        return self._root / "acquisition-staging" / name

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
    def _write_durable(cls, destination: Path, payload: bytes) -> None:
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
            temporary.replace(destination)
            cls._fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _publish_durable(cls, destination: Path, payload: bytes) -> bytes:
        """Publish once across processes and return the representation that won.

        A hard link is the POSIX no-clobber analogue of an atomic rename: exactly one
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
        await self.cleanup_staging_partials()
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
        payload = json.dumps(
            {
                "blob_ref": acquired.content_hash,
                "compression": stored.compression,
                "acquired_source": acquired.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        await self._durable_thread_call(lambda: self._write_durable(self._stage_path(key), payload))
        await self._record_blob(stored, raw.media_type)
        return Retention(ref=stored.hash), acquired

    async def cleanup_staging_partials(
        self,
        *,
        stale_after_seconds: float = STAGING_PARTIAL_STALE_SECONDS,
        limit: int = STAGING_PARTIAL_CLEANUP_LIMIT,
    ) -> StagingCleanup:
        """Remove only old abandoned staging writes, reporting aggregate counts."""
        root = self._root / "acquisition-staging"
        if not root.exists() or limit <= 0:
            return StagingCleanup(scanned=0, removed=0, truncated=False)
        cutoff = time.time() - stale_after_seconds
        candidates: list[Path] = []
        scanned = 0
        truncated = False
        for path in root.iterdir():
            if scanned == limit:
                truncated = True
                break
            scanned += 1
            if not path.is_file() or not path.name.endswith(".partial"):
                continue
            candidates.append(path)
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
        return StagingCleanup(scanned=scanned, removed=removed, truncated=truncated)

    async def resume_acquisition(self, key: str) -> tuple[Retention, AcquiredSource] | None:
        """Recover a staged blob/envelope pair without contacting its source."""
        await self.cleanup_staging_partials()
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

    @classmethod
    def _unlink_durable(cls, path: Path) -> None:
        """Unlink a name and durably record its absence as one joined operation."""
        try:
            path.unlink()
        except FileNotFoundError:
            if not path.parent.exists():
                return
        cls._fsync_directory(path.parent)

    async def _staged_blob_refs(self) -> set[str]:
        await self.cleanup_staging_partials()
        root = self._root / "acquisition-staging"
        if not root.exists():
            return set()
        refs: set[str] = set()
        for path in root.iterdir():
            if not path.is_file() or path.name.endswith(".partial"):
                continue
            try:
                payload = json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            ref = (
                cast("dict[str, object]", payload).get("blob_ref")
                if isinstance(payload, dict)
                else None
            )
            if isinstance(ref, str):
                refs.add(ref)
        return refs

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
            unreferenced = (
                (
                    await session.execute(
                        select(models.Blob.hash).where(
                            models.Blob.hash.not_in(
                                union(
                                    referenced_documents,
                                    referenced_versions,
                                    referenced_acquisitions,
                                )
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

        staged = await self._staged_blob_refs()
        collected: list[str] = []
        for digest in unreferenced:
            if digest in staged:
                continue
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
