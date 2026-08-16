"""Retained original bytes, so re-parsing never means re-fetching.

This is rung 3 of the blast-radius ladder, and the only thing standing between a parser bug
fix and a full re-crawl of a rate-limited API. Every other rung is a pure function of what is
already on disk; re-fetching is the one repair that can fail for reasons outside the machine,
and the one whose result is not reproducible.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast
from uuid import uuid4

from sqlalchemy import and_, delete, func, select, text, union, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.acquisition import AcquiredSource
from manicule.core.content import RawDocument, Retention
from manicule.core.ids import acquisition_marker_id, content_hash
from manicule.ingest.capacity import (
    CapacityDiagnostic,
    CapacityRefusedError,
    CapacityResource,
    require_blob_backlog_capacity,
    require_disk_headroom,
    translate_storage_capacity_errors,
)
from manicule.storage import models
from manicule.storage.engine import BLOBS_DIRNAME, session_factory

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterator, Sequence

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
DURABLE_LOCK_SHARDS = 256
GC_PENDING_PREFIX = "gc_pending:"
GC_IDENTITY_HEX_LENGTH = 32
GC_CAPACITY_SCAN_LIMIT = 1_000


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


class EvidencePinFence:
    """Canonical retained representations and a bounded managed-writer fence."""

    def __init__(self, store: BlobStore, root: Path) -> None:
        self._store = store
        self._root = root
        self._shard_bitmap = 0
        self._publication_locked = False

    def _remember_shard(self, digest: str) -> None:
        self._shard_bitmap |= 1 << self._store.evidence_lock_shard(digest)

    def _pin_path(self, digest: str) -> Path:
        return self._root / digest

    @staticmethod
    def _valid_digest(digest: str) -> bool:
        return len(digest) == GC_IDENTITY_HEX_LENGTH and all(
            character in "0123456789abcdef" for character in digest
        )

    async def pin(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        """Pin the currently named inode and return its cheap verified identity."""

        if not self._valid_digest(digest):
            return None

        self._remember_shard(digest)
        async with self._store.evidence_locks([digest]):
            return await asyncio.to_thread(
                self._store.pin_evidence_representation,
                digest,
                self._pin_path(digest),
                size_bytes,
                stored_bytes,
                compression,
            )

    async def verify(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        """Promote and hash one canonical representation under only its digest shard."""
        if not self._valid_digest(digest):
            return None
        self._remember_shard(digest)
        async with self._store.evidence_locks([digest]):
            pinned = await asyncio.to_thread(
                self._store.pin_evidence_representation,
                digest,
                self._pin_path(digest),
                size_bytes,
                stored_bytes,
                compression,
            )
            if pinned is None:
                return None
            verified = await self._store.evidence_identity(
                digest,
                size_bytes=size_bytes,
                stored_bytes=stored_bytes,
                compression=compression,
                verify_content=True,
            )
            return pinned if verified == pinned else None

    async def validate(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        """Recheck the pinned inode and its public name immediately before commit."""

        if not self._valid_digest(digest):
            return None

        self._remember_shard(digest)
        if self._publication_locked:
            return await asyncio.to_thread(
                self._store.validate_evidence_pin,
                digest,
                self._pin_path(digest),
                size_bytes,
                stored_bytes,
                compression,
            )
        async with self._store.evidence_locks([digest]):
            return await asyncio.to_thread(
                self._store.validate_evidence_pin,
                digest,
                self._pin_path(digest),
                size_bytes,
                stored_bytes,
                compression,
            )

    @contextlib.asynccontextmanager
    async def publication_fence(self) -> AsyncGenerator[None]:
        """Hold only represented lock shards across cheap final probes and DB commit."""
        shards = [
            shard for shard in range(DURABLE_LOCK_SHARDS) if self._shard_bitmap & (1 << shard)
        ]
        async with self._store.evidence_lock_shards(shards):
            self._publication_locked = True
            try:
                yield
            finally:
                self._publication_locked = False


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
        min_disk_headroom_bytes: int = 2 * 1024 * 1024 * 1024,
        max_acquired_blob_backlog_bytes: int = 20 * 1024 * 1024 * 1024,
        sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._engine = engine
        self._root = data_dir / BLOBS_DIRNAME
        self._max_bytes = max_bytes
        self._min_disk_headroom_bytes = min_disk_headroom_bytes
        self._max_acquired_blob_backlog_bytes = max_acquired_blob_backlog_bytes
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

    def evidence_path_for(self, digest: str) -> Path:
        """Canonical retained representation after offline verification promotes it."""
        return self._root / "evidence-pins" / "by-digest" / digest

    def _authoritative_path(self, digest: str) -> Path:
        canonical = self.evidence_path_for(digest)
        try:
            canonical.lstat()
        except FileNotFoundError:
            return self.path_for(digest)
        return canonical

    def _stage_path(self, key: str) -> Path:
        return self._root / "acquisition-staging" / self._stage_name(key)

    @contextlib.asynccontextmanager
    async def evidence_locks(self, digests: Sequence[str]) -> AsyncGenerator[None]:
        """Coordinate canonical evidence with managed writers using bounded lock shards."""
        async with self._durable_locks([f"blob:{digest}" for digest in sorted(digests)]):
            yield

    @staticmethod
    def evidence_lock_shard(digest: str) -> int:
        """Return the stable managed-writer lock shard for one content digest."""
        return BlobStore._durable_lock_shard(f"blob:{digest}")

    @contextlib.asynccontextmanager
    async def evidence_lock_shards(self, shards: Sequence[int]) -> AsyncGenerator[None]:
        """Hold an already-derived, cardinality-bounded evidence shard set."""
        async with self._durable_lock_shards(shards):
            yield

    @staticmethod
    def _stage_name(key: str) -> str:
        run_id, separator, source_id = key.partition("\0")
        if separator:
            return acquisition_marker_id(run_id, source_id)
        return hashlib.blake2b(key.encode(), digest_size=20).hexdigest()

    def _stage_partial_root(self) -> Path:
        return self._root / "acquisition-staging-partials"

    def _gc_root(self) -> Path:
        return self._root / "gc-intents"

    def _gc_paths(self, digest: str, token: str) -> tuple[Path, Path]:
        if not self._gc_identity_safe(digest, token):
            msg = "invalid garbage-collection identity"
            raise ValueError(msg)
        stem = f"{digest}.{token}"
        root = self._gc_root()
        return root / f"{stem}.blob", root / f"{stem}.json"

    @staticmethod
    def _gc_identity_safe(digest: str, token: str) -> bool:
        return (
            len(digest) == GC_IDENTITY_HEX_LENGTH
            and len(token) == GC_IDENTITY_HEX_LENGTH
            and all(character in "0123456789abcdef" for character in digest)
            and all(character in "0123456789abcdef" for character in token)
        )

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
    def _publish_durable(cls, destination: Path, payload: bytes) -> tuple[bytes, bool]:
        """Publish once and return the winning representation plus this call's ownership.

        A hard link is the POSIX no-clobber analog of an atomic rename: exactly one
        temporary inode can acquire ``destination``. Every contender then syncs the parent
        before it may create a database reference, including one that observed another
        process's newly linked name before that process reached its own directory sync.
        """
        cls._mkdir_durable(destination.parent)
        temporary = destination.with_name(f"{destination.name}.{uuid4().hex}.partial")
        created = False
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
            try:
                os.link(temporary, destination)
                created = True
            except FileExistsError:
                pass
            try:
                cls._fsync_directory(destination.parent)
                return destination.read_bytes(), created
            except BaseException as error:
                # The caller must still know whether it owns the winning link when fsync or
                # validation fails after publication. The attribute is process-private and
                # carries no source-shaped data.
                error.__dict__["_manicule_publication_created"] = created
                raise
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as error:
                error.__dict__["_manicule_publication_created"] = created
                error.__dict__["_manicule_publication_temporary"] = temporary
                raise

    @classmethod
    def _remove_published(cls, destination: Path) -> None:
        destination.unlink(missing_ok=True)
        cls._fsync_directory(destination.parent)

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
            cancellation.__dict__["_manicule_durable_result"] = result
            raise cancellation
        return result

    @staticmethod
    async def _joined_async_call[T](call: Coroutine[Any, Any, T]) -> T:
        """Finish an async durability sequence before honoring repeated cancellation."""
        work = asyncio.create_task(call)
        current = asyncio.current_task()
        cancellation: asyncio.CancelledError | None = None
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError as error:
                cancellation = error
                if current is not None:
                    current.uncancel()
        result = work.result()
        if cancellation is not None:
            raise cancellation
        return result

    @classmethod
    def _published_blob(
        cls, destination: Path, data: bytes, digest: str, compression: str
    ) -> tuple[StoredBlob, bool]:
        proposed = gzip.compress(data, mtime=0) if compression == "gzip" else data
        published, created = cls._publish_durable(destination, proposed)
        if published == data:
            actual_compression = "none"
        else:
            try:
                decoded = gzip.decompress(published)
            except (OSError, EOFError) as error:
                msg = f"retained blob {digest} has an unrecognized representation"
                replacement = OSError(msg)
                replacement.__dict__["_manicule_publication_created"] = created
                raise replacement from error
            if decoded != data:
                error = OSError(f"retained blob {digest} does not match its content address")
                error.__dict__["_manicule_publication_created"] = created
                raise error
            actual_compression = "gzip"
        return (
            StoredBlob(
                hash=digest,
                size_bytes=len(data),
                stored_bytes=len(published),
                compression=actual_compression,
            ),
            created,
        )

    @staticmethod
    async def _record_blob(
        session: AsyncSession, stored: StoredBlob, media_type: str | None
    ) -> None:
        """Make SQLite describe the immutable representation that is already on disk."""
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
                    "algo": "blake2b",
                    "size_bytes": stored.size_bytes,
                    "stored_bytes": stored.stored_bytes,
                    "compression": stored.compression,
                },
            )
        )

    async def _pending_blob_bytes(self, session: AsyncSession) -> int:
        published_documents = select(models.Document.original_ref).where(
            models.Document.original_ref.is_not(None)
        )
        published_versions = select(models.DocumentVersion.original_ref).where(
            models.DocumentVersion.original_ref.is_not(None)
        )
        valid_normal_size = and_(
            func.typeof(models.Blob.stored_bytes) == "integer",
            models.Blob.stored_bytes >= 0,
        )
        normal_filters = (
            models.Blob.algo.not_like(f"{GC_PENDING_PREFIX}%"),
            models.Blob.hash.not_in(union(published_documents, published_versions)),
        )
        invalid_normal = (
            await session.execute(
                select(models.Blob.hash).where(*normal_filters, ~valid_normal_size).limit(1)
            )
        ).scalar_one_or_none()
        if invalid_normal is not None:
            self._refuse_gc_pending()
        normal_described = 0
        normal_sizes = await session.stream_scalars(
            select(models.Blob.stored_bytes)
            .where(*normal_filters, valid_normal_size, models.Blob.stored_bytes > 0)
            .execution_options(yield_per=256)
        )
        async for stored_bytes in normal_sizes:
            if type(stored_bytes) is not int or stored_bytes < 0:
                await normal_sizes.close()
                self._refuse_gc_pending()
            if normal_described > self._max_acquired_blob_backlog_bytes - stored_bytes:
                await normal_sizes.close()
                normal_described = self._max_acquired_blob_backlog_bytes + 1
                break
            normal_described += stored_bytes
        pending_filter = models.Blob.algo.like(f"{GC_PENDING_PREFIX}%")
        pending_rows = (
            await session.execute(
                select(
                    models.Blob.hash,
                    models.Blob.algo,
                    models.Blob.stored_bytes,
                    func.typeof(models.Blob.hash),
                    func.typeof(models.Blob.algo),
                    func.typeof(models.Blob.stored_bytes),
                )
                .where(pending_filter)
                .limit(GC_CAPACITY_SCAN_LIMIT + 1)
            )
        ).all()
        if len(pending_rows) > GC_CAPACITY_SCAN_LIMIT:
            self._refuse_gc_pending()
        if any(
            type(digest) is not str
            or type(algo) is not str
            or type(stored_bytes) is not int
            or stored_bytes < 0
            or hash_type != "text"
            or algo_type != "text"
            or stored_type != "integer"
            for digest, algo, stored_bytes, hash_type, algo_type, stored_type in pending_rows
        ):
            self._refuse_gc_pending()
        pending_claims = {
            (digest, algo.removeprefix(GC_PENDING_PREFIX)): stored_bytes
            for digest, algo, stored_bytes, _, _, _ in pending_rows
        }
        pending_physical = await asyncio.to_thread(self._gc_artifact_bytes, pending_claims)
        return normal_described + pending_physical

    def _gc_artifact_bytes(self, pending_claims: dict[tuple[str, str], int]) -> int:
        """Conservatively count pending/orphan representations without an unbounded scan."""
        claims: dict[str, int] = {
            f"{digest}.{token}": stored_bytes
            for (digest, token), stored_bytes in pending_claims.items()
        }
        identity_inodes: dict[str, set[tuple[int, int]]] = {}
        inode_sizes: dict[tuple[int, int], int] = {}

        def charge(path: Path, identity: str) -> None:
            try:
                status = path.lstat()
            except FileNotFoundError:
                return
            except OSError:
                self._refuse_gc_pending()
            if not stat.S_ISREG(status.st_mode):
                self._refuse_gc_pending()
            inode = (status.st_dev, status.st_ino)
            identity_inodes.setdefault(identity, set()).add(inode)
            inode_sizes.setdefault(inode, max(0, status.st_size))

        for digest, token in pending_claims:
            if self._gc_identity_safe(digest, token):
                charge(self._authoritative_path(digest), f"{digest}.{token}")

        root = self._gc_root()
        try:
            root_status = root.lstat()
        except FileNotFoundError:
            root_status = None
        except OSError:
            self._refuse_gc_pending()
        if root_status is not None:
            if not stat.S_ISDIR(root_status.st_mode):
                self._refuse_gc_pending()
            try:
                inspected = list(islice(root.iterdir(), GC_CAPACITY_SCAN_LIMIT + 1))
            except OSError:
                self._refuse_gc_pending()
            if len(inspected) > GC_CAPACITY_SCAN_LIMIT:
                self._refuse_gc_pending()
            for path in inspected:
                identity = path.stem
                if path.suffix == ".blob":
                    charge(path, identity)
                elif path.suffix == ".json":
                    payload = self._read_gc_intent(path)
                    if payload is not None:
                        claims[identity] = max(claims.get(identity, 0), payload[2])
        return self._gc_inode_accounted_bytes(claims, identity_inodes, inode_sizes)

    @staticmethod
    def _gc_inode_accounted_bytes(
        claims: dict[str, int],
        identity_inodes: dict[str, set[tuple[int, int]]],
        inode_sizes: dict[tuple[int, int], int],
    ) -> int:
        """Charge each hard-linked representation once across all claiming identities."""
        inode_identities: dict[tuple[int, int], set[str]] = {}
        for identity, inodes in identity_inodes.items():
            for inode in inodes:
                inode_identities.setdefault(inode, set()).add(identity)
        remaining = set(identity_inodes)
        total = 0
        while remaining:
            frontier = [remaining.pop()]
            component_identities: set[str] = set()
            component_inodes: set[tuple[int, int]] = set()
            while frontier:
                identity = frontier.pop()
                if identity in component_identities:
                    continue
                component_identities.add(identity)
                for inode in identity_inodes[identity]:
                    component_inodes.add(inode)
                    for peer in inode_identities[inode]:
                        if peer not in component_identities:
                            remaining.discard(peer)
                            frontier.append(peer)
            physical_bytes = sum(inode_sizes[inode] for inode in component_inodes)
            claimed_bytes = max(claims.get(identity, 0) for identity in component_identities)
            total += max(physical_bytes, claimed_bytes)
        total += sum(
            claimed for identity, claimed in claims.items() if identity not in identity_inodes
        )
        return total

    @staticmethod
    async def _descriptor_is_durable_or_ambiguous(session: AsyncSession, digest: str) -> bool:
        """Preserve bytes when a committed descriptor exists or cannot be ruled out."""
        try:
            return await session.get(models.Blob, digest) is not None
        except Exception:  # noqa: BLE001 - ambiguity must preserve recoverable bytes
            return True

    @staticmethod
    def _marker_match_state(path: Path, digest: str) -> bool | None:
        """Whether a marker matches, with ``None`` for an ambiguous read failure."""
        try:
            decoded = cast("object", json.loads(path.read_text("utf-8")))
        except FileNotFoundError:
            return False
        except OSError:
            return None
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(decoded, dict):
            return False
        payload = cast("dict[str, object]", decoded)
        return payload.get("blob_ref") == digest and payload.get("compression") in {"gzip", "none"}

    async def _remove_owned_with_retry(self, destination: Path) -> bool:
        """Retry one transient unlink/fsync failure without looping on permanent refusal."""
        for _attempt in range(2):
            try:
                await self._durable_thread_call(lambda: self._remove_published(destination))
            except Exception:  # noqa: BLE001, S112 - fallback accounts the surviving file
                continue
            return True
        return False

    async def _cleanup_owned_blob(
        self,
        digest: str,
        destination: Path,
        stored: StoredBlob,
        media_type: str | None,
    ) -> None:
        """Delete an owned link or durably account it while excluding digest adopters."""
        try:
            async with self._sessions() as cleanup:
                # A waiter either commits before this reservation (and is observed below) or
                # starts after the unlink. It cannot adopt the file between probe and unlink.
                await cleanup.execute(text("BEGIN IMMEDIATE"))
                if not await self._descriptor_is_durable_or_ambiguous(cleanup, digest):
                    if await self._remove_owned_with_retry(destination):
                        await cleanup.rollback()
                    else:
                        # A permanent permission/filesystem refusal must not turn physical
                        # bytes into capacity-invisible state. This call owns the winning link,
                        # so its exact proposed representation is safe to describe.
                        await self._record_blob(cleanup, stored, media_type)
                        await cleanup.commit()
                else:
                    await cleanup.rollback()
        except BaseException:  # noqa: BLE001 - ambiguity preserves the recoverable file
            return

    async def _remove_failed_marker(self, path: Path) -> bool:
        """Remove an uncertified marker; ambiguity preserves its matching blob."""
        try:
            await self._durable_thread_call(lambda: self._unlink_durable(path))
        except BaseException:  # noqa: BLE001 - cleanup uncertainty preserves recovery bytes
            return False
        return True

    async def _cleanup_publication_temporary(
        self, failure: BaseException, destination: Path
    ) -> None:
        temporary = getattr(failure, "_manicule_publication_temporary", None)
        if not isinstance(temporary, type(destination)):
            return
        expected_prefix = f"{destination.name}."
        if (
            temporary.parent != destination.parent
            or not temporary.name.startswith(expected_prefix)
            or not temporary.name.endswith(".partial")
        ):
            return
        await self._durable_thread_call(lambda: self._unlink_durable(temporary))

    def _refuse_gc_pending(self) -> NoReturn:
        raise CapacityRefusedError(
            CapacityDiagnostic(
                resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
                limit=self._max_acquired_blob_backlog_bytes,
                used=self._max_acquired_blob_backlog_bytes,
                requested=1,
            )
        )

    @staticmethod
    def _publication_state(
        failure: BaseException, *, completed: bool, created: bool
    ) -> tuple[bool, bool]:
        """Recover publication ownership hidden by a joined call's exception boundary."""
        if completed:
            return True, created
        durable_result = getattr(failure, "_manicule_durable_result", None)
        match durable_result:
            case (StoredBlob(), result_created):
                return True, bool(result_created)
            case _:
                pass
        result_created = bool(getattr(failure, "_manicule_publication_created", False))
        return result_created, result_created

    async def _rollback_failed_store(
        self,
        session: AsyncSession,
        failure: BaseException,
        *,
        digest: str,
        destination: Path,
        publication_completed: bool,
        publication_created: bool,
        owned_representation: StoredBlob,
        media_type: str | None,
        marker_completed: bool,
        marker_path: Path | None,
        marker_existed: bool,
    ) -> None:
        await session.rollback()
        await self._cleanup_publication_temporary(failure, destination)
        publication_completed, publication_created = self._publication_state(
            failure,
            completed=publication_completed,
            created=publication_created,
        )
        marker_state = (
            False if marker_path is None else self._marker_match_state(marker_path, digest)
        )
        preserve = marker_state is True or (marker_state is None and marker_completed)
        if not preserve and marker_path is not None and not marker_existed:
            preserve = not await self._remove_failed_marker(marker_path)
        if publication_completed and publication_created and not preserve:
            await self._cleanup_owned_blob(digest, destination, owned_representation, media_type)

    @translate_storage_capacity_errors
    async def _store_blob(
        self,
        data: bytes,
        media_type: str | None,
        *,
        staging_key: str | None = None,
        acquired: AcquiredSource | None = None,
    ) -> StoredBlob:
        digest = content_hash(data)
        if staging_key is None:
            async with self._durable_locks([f"blob:{digest}"]):
                return await self._store_blob_locked(data, media_type)
        keys = [f"blob:{digest}", f"marker:{self._stage_name(staging_key)}"]
        async with self._durable_locks(keys):
            return await self._store_blob_locked(
                data,
                media_type,
                staging_key=staging_key,
                acquired=acquired,
            )

    async def _store_blob_locked(
        self,
        data: bytes,
        media_type: str | None,
        *,
        staging_key: str | None = None,
        acquired: AcquiredSource | None = None,
    ) -> StoredBlob:
        if (staging_key is None) != (acquired is None):
            msg = "an acquisition stage requires both its key and source envelope"
            raise ValueError(msg)
        digest = content_hash(data)
        compression = "gzip" if should_compress(media_type) else "none"
        proposed = gzip.compress(data, mtime=0) if compression == "gzip" else data
        owned_representation = StoredBlob(
            hash=digest,
            size_bytes=len(data),
            stored_bytes=len(proposed),
            compression=compression,
        )
        destination = self._authoritative_path(digest)
        publication_completed = False
        publication_created = False
        marker_completed = False
        marker_path = self._stage_path(staging_key) if staging_key is not None else None
        marker_existed = marker_path is not None and marker_path.exists()
        async with self._sessions() as session:
            # SQLite's write reservation spans the aggregate check, filesystem publication,
            # and descriptor commit. Separate BlobStore instances and processes therefore
            # cannot each spend the same observed capacity.
            await session.execute(text("BEGIN IMMEDIATE"))
            try:
                existing_row = await session.get(models.Blob, digest)
                if existing_row is not None and existing_row.algo.startswith(GC_PENDING_PREFIX):
                    await session.rollback()
                    self._refuse_gc_pending()
                destination_exists = destination.exists()
                described_bytes = 0 if existing_row is None else existing_row.stored_bytes
                if not destination_exists:
                    requested_growth = max(0, len(proposed) - described_bytes)
                else:
                    requested_growth = 0
                if requested_growth:
                    require_blob_backlog_capacity(
                        used=await self._pending_blob_bytes(session),
                        requested=requested_growth,
                        limit=self._max_acquired_blob_backlog_bytes,
                    )
                if not destination_exists:
                    require_disk_headroom(
                        free=shutil.disk_usage(self._root.parent).free,
                        requested=len(proposed),
                        minimum=self._min_disk_headroom_bytes,
                    )
                stored, publication_created = await self._durable_thread_call(
                    lambda: self._published_blob(destination, data, digest, compression)
                )
                publication_completed = True
                actual_growth = max(0, stored.stored_bytes - described_bytes)
                if destination_exists and actual_growth:
                    # A crash may leave a durable representation without its descriptor. Its
                    # actual bytes, not this caller's compression preference, consume backlog.
                    require_blob_backlog_capacity(
                        used=await self._pending_blob_bytes(session),
                        requested=actual_growth,
                        limit=self._max_acquired_blob_backlog_bytes,
                    )
                if staging_key is not None and acquired is not None:
                    run_id, separator, source_id = staging_key.partition("\0")
                    marker: dict[str, object] = {
                        "blob_ref": digest,
                        "compression": stored.compression,
                        "acquired_source": acquired.model_dump(mode="json"),
                        "run_id": run_id if separator else None,
                        "source_id": source_id if separator else None,
                    }
                    await self._record_marker_in_session(
                        session,
                        self._stage_name(staging_key),
                        marker,
                        legacy=False,
                        created_at=None,
                    )
                    payload = json.dumps(
                        marker,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    await self._durable_thread_call(
                        lambda: self._write_durable(
                            cast("Path", marker_path),
                            payload,
                            temporary_dir=self._stage_partial_root(),
                        )
                    )
                    marker_completed = True
                await self._record_blob(session, stored, media_type)
                await session.commit()
            except BaseException as failure:
                await self._joined_async_call(
                    self._rollback_failed_store(
                        session,
                        failure,
                        digest=digest,
                        destination=destination,
                        publication_completed=publication_completed,
                        publication_created=publication_created,
                        owned_representation=owned_representation,
                        media_type=media_type,
                        marker_completed=marker_completed,
                        marker_path=marker_path,
                        marker_existed=marker_existed,
                    )
                )
                raise
        return stored

    @translate_storage_capacity_errors
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
        stored = await self._store_blob(data, raw.media_type, staging_key=key, acquired=acquired)
        return Retention(ref=stored.hash), acquired

    @contextlib.asynccontextmanager
    async def _marker_locks(self, names: Sequence[str]) -> AsyncGenerator[None]:
        """Cross-process marker ordering backed by a fixed-size sharded lock pool."""
        async with self._durable_locks([f"marker:{name}" for name in names]):
            yield

    @contextlib.asynccontextmanager
    async def _durable_locks(self, keys: Sequence[str]) -> AsyncGenerator[None]:
        """Lock stable shards so cardinality cannot turn recovery metadata into an inode leak."""
        shards = sorted({self._durable_lock_shard(key) for key in keys})
        async with self._durable_lock_shards(shards):
            yield

    @staticmethod
    def _durable_lock_shard(key: str) -> int:
        return (
            int.from_bytes(hashlib.blake2b(key.encode(), digest_size=2).digest(), byteorder="big")
            % DURABLE_LOCK_SHARDS
        )

    @contextlib.asynccontextmanager
    async def _durable_lock_shards(self, shards: Sequence[int]) -> AsyncGenerator[None]:
        """Acquire validated stable shard ids in global order."""
        ordered = sorted(set(shards))
        if any(shard < 0 or shard >= DURABLE_LOCK_SHARDS for shard in ordered):
            raise ValueError("invalid durable lock shard")

        def acquire() -> list[int]:
            root = self._root / "acquisition-locks"
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptors: list[int] = []
            try:
                for shard in ordered:
                    descriptor = os.open(root / f"{shard:02x}.lock", os.O_CREAT | os.O_RDWR, 0o600)
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    descriptors.append(descriptor)
            except BaseException:
                self._release_marker_locks(descriptors)
                raise
            return descriptors

        work = asyncio.create_task(asyncio.to_thread(acquire))
        try:
            descriptors = await asyncio.shield(work)
        except asyncio.CancelledError:
            descriptors = await work
            await asyncio.to_thread(self._release_marker_locks, descriptors)
            raise
        try:
            yield
        finally:
            await asyncio.to_thread(self._release_marker_locks, descriptors)

    @staticmethod
    def _release_marker_locks(descriptors: Sequence[int]) -> None:
        for descriptor in reversed(descriptors):
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(descriptor)

    async def _record_marker(
        self,
        name: str,
        payload: dict[str, object],
        *,
        legacy: bool,
        created_at: datetime | None = None,
    ) -> None:
        async with self._sessions.begin() as session:
            await self._record_marker_in_session(
                session, name, payload, legacy=legacy, created_at=created_at
            )

    async def _record_marker_in_session(
        self,
        session: AsyncSession,
        name: str,
        payload: dict[str, object],
        *,
        legacy: bool,
        created_at: datetime | None,
    ) -> None:
        proposed = self._marker_identity(payload)
        path = self._root / "acquisition-staging" / name
        # A no-op conditional write is the serialization point for retries of one key.
        await session.execute(
            update(models.AcquisitionMarker)
            .where(models.AcquisitionMarker.name == name)
            .values(name=models.AcquisitionMarker.name)
        )
        existing = await session.get(models.AcquisitionMarker, name)
        if existing is None:
            session.add(
                models.AcquisitionMarker(
                    name=name,
                    run_id=cast("str | None", payload.get("run_id")),
                    source_id=cast("str | None", payload.get("source_id")),
                    blob_ref=cast("str | None", payload.get("blob_ref")),
                    acquired_source=cast("Any", payload.get("acquired_source")),
                    legacy=legacy,
                    created_at=created_at or datetime.now(UTC),
                )
            )
            return
        current = (
            existing.run_id,
            existing.source_id,
            existing.blob_ref,
            cast("object", existing.acquired_source),
        )
        if current == proposed:
            return
        disk = await self._physical_marker_identity(path)
        if disk is not None and disk != proposed:
            msg = f"acquisition marker {name!r} conflicts with durable recovery evidence"
            raise RuntimeError(msg)
        existing.run_id = cast("str | None", payload.get("run_id"))
        existing.source_id = cast("str | None", payload.get("source_id"))
        existing.blob_ref = cast("str | None", payload.get("blob_ref"))
        existing.acquired_source = cast("Any", payload.get("acquired_source"))
        existing.legacy = legacy
        existing.created_at = created_at or datetime.now(UTC)

    @staticmethod
    def _marker_identity(
        payload: dict[str, object],
    ) -> tuple[object, object, object, object]:
        return (
            payload.get("run_id"),
            payload.get("source_id"),
            payload.get("blob_ref"),
            payload.get("acquired_source"),
        )

    async def _physical_marker_identity(
        self, path: Path
    ) -> tuple[object, object, object, object] | None:
        try:
            raw = json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        return self._marker_identity(cast("dict[str, object]", raw))

    async def _record_markers(
        self,
        markers: Sequence[tuple[str, dict[str, object], bool, datetime]],
    ) -> None:
        if not markers:
            return
        statement = sqlite_insert(models.AcquisitionMarker).values(
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
        async with self._sessions.begin() as session:
            await session.execute(
                statement.on_conflict_do_nothing(index_elements=[models.AcquisitionMarker.name])
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

    async def _index_legacy_marker_page(self) -> bool:
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
        for entry in entries:
            if not entry.is_file() or entry.name in known:
                continue
            try:
                raw = json.loads(await asyncio.to_thread(Path(entry.path).read_text, "utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                raw = {}
            payload = cast("dict[str, object]", raw) if isinstance(raw, dict) else {}
            parsed.append((entry, payload))
        inferred: dict[str, tuple[str, str]] = {}
        if parsed:
            async with self._sessions() as session:
                records = (
                    (
                        await session.execute(
                            select(
                                models.AcquisitionRecord.run_id,
                                models.AcquisitionRecord.source_id,
                                models.AcquisitionRecord.marker_name,
                            ).where(
                                models.AcquisitionRecord.marker_name.in_(
                                    [entry.name for entry, _payload in parsed]
                                )
                            )
                        )
                    )
                    .tuples()
                    .all()
                )
            inferred = {
                marker_name: (run_id, source_id)
                for run_id, source_id, marker_name in records
                if marker_name is not None
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
            names = (
                (
                    await session.execute(
                        select(models.AcquisitionMarker.name)
                        .where(models.AcquisitionMarker.name > self._marker_cursor)
                        .order_by(models.AcquisitionMarker.name)
                        .limit(STAGING_MARKER_RECONCILE_LIMIT)
                    )
                )
                .scalars()
                .all()
            )
        if not names:
            self._marker_cursor = ""
            return
        async with self._marker_locks(names):
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(
                            models.AcquisitionMarker,
                            models.AcquisitionRun.id,
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
                        .where(models.AcquisitionMarker.name.in_(names))
                        .order_by(models.AcquisitionMarker.name)
                    )
                ).all()
            remove: list[str] = []
            cutoff = datetime.now(UTC) - LEGACY_MARKER_RETENTION
            for (
                marker,
                joined_run_id,
                superseded_at,
                record_blob_ref,
                record_acquired_source,
            ) in rows:
                path = self._root / "acquisition-staging" / marker.name
                exact_association = (
                    marker.blob_ref is not None
                    and marker.blob_ref == record_blob_ref
                    and cast("object", marker.acquired_source) == record_acquired_source
                )
                expired_orphan = marker.run_id is None and marker.created_at < cutoff
                missing_owner = marker.run_id is not None and joined_run_id is None
                removable = (
                    missing_owner
                    or superseded_at is not None
                    or exact_association
                    or expired_orphan
                )
                if not removable:
                    continue
                await self._durable_thread_call(lambda path=path: self._unlink_durable(path))
                remove.append(marker.name)
            await self._forget_markers(remove)
        self._marker_cursor = names[-1]

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
            data = await asyncio.to_thread(
                self._read_blob, self._authoritative_path(blob_ref), compression
            )
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
        if row is None or row.algo.startswith(GC_PENDING_PREFIX):
            return None
        path = self._authoritative_path(digest)
        try:
            return await asyncio.to_thread(self._read_blob, path, row.compression)
        except FileNotFoundError:
            return None

    @staticmethod
    def _read_blob(path: Path, compression: str) -> bytes:
        """Keep filesystem latency and decompression CPU off the acquisition event loop."""
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read()
        return gzip.decompress(raw) if compression == "gzip" else raw

    async def get_bounded(self, digest: str, *, max_bytes: int) -> bytes | None:
        """Read at most ``max_bytes`` of verified-size input into memory.

        The SQLite size is checked before opening or decompressing the representation. The
        streamed verifier is deliberately separate: callers can inventory a corpus without
        ever allocating one source body.
        """
        if max_bytes < 0:
            raise ValueError("max_bytes must not be negative")
        async with self._sessions() as session:
            row = await session.get(models.Blob, digest)
        if row is None or row.size_bytes > max_bytes:
            return None
        path = self._authoritative_path(digest)

        def bounded_read() -> bytes | None:
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
                observed = os.fstat(descriptor)
                if not stat.S_ISREG(observed.st_mode):
                    return None
                with os.fdopen(descriptor, "rb") as raw:
                    descriptor = None
                    if row.compression == "gzip":
                        with gzip.GzipFile(fileobj=raw) as handle:
                            data = handle.read(max_bytes + 1)
                    else:
                        if observed.st_size > max_bytes:
                            return None
                        data = raw.read()
            except (OSError, EOFError):
                return None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            return data if len(data) <= max_bytes else None

        return await asyncio.to_thread(bounded_read)

    async def verify(self, digest: str) -> bool:
        """Whether the stored bytes still hash to their own name."""
        data = await self.get(digest)
        return data is not None and content_hash(data) == digest

    async def contains(self, digest: str) -> bool:
        """Whether a rebuild may rely on this exact local input.

        Stronger than a path-existence check: a corrupt local file is missing as an offline
        rebuild input. Verification hashes a fixed-size stream and never materializes or
        fully decompresses a source body in memory.
        """
        async with self._sessions() as session:
            row = await session.get(models.Blob, digest)
        if row is None:
            return False
        return (
            await self.evidence_identity(
                digest,
                size_bytes=row.size_bytes,
                stored_bytes=row.stored_bytes,
                compression=row.compression,
                verify_content=True,
            )
            is not None
        )

    @contextlib.asynccontextmanager
    async def evidence_fence(self) -> AsyncGenerator[EvidencePinFence]:
        """Expose canonical pins; each operation locks only its represented digest shard."""
        root = self._root / "evidence-pins" / "by-digest"
        await asyncio.to_thread(self._mkdir_durable, root)
        yield EvidencePinFence(self, root)

    async def evidence_identity(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
        verify_content: bool,
    ) -> str | None:
        """Return a cheap stable representation identity, optionally after one full hash.

        The identity binds the descriptor and the exact named inode. A caller can therefore
        hash outside SQLite's writer transaction, persist the identity, and later reject a
        replaced, truncated, removed, or metadata-changed representation with only ``stat``.
        """
        path = self._authoritative_path(digest)

        def inspect() -> str | None:  # noqa: PLR0911 - every invalid representation fails closed
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError:
                return None
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_size != stored_bytes:
                    return None
                if verify_content:
                    hasher = hashlib.blake2b(digest_size=16)
                    hasher.update(size_bytes.to_bytes(8, "big"))
                    total = 0
                    with os.fdopen(os.dup(descriptor), "rb") as raw:
                        handle = gzip.GzipFile(fileobj=raw) if compression == "gzip" else raw
                        with contextlib.closing(handle):
                            while chunk := handle.read(1024 * 1024):
                                total += len(chunk)
                                if total > size_bytes:
                                    return None
                                hasher.update(chunk)
                    if total != size_bytes or hasher.hexdigest() != digest:
                        return None
                after = os.fstat(descriptor)
                named = path.stat(follow_symlinks=False)
            except (OSError, EOFError):
                return None
            finally:
                os.close(descriptor)
            observed = self._representation_stat(after)
            if (
                observed != self._representation_stat(before)
                or self._representation_stat(named) != observed
            ):
                return None
            return self._representation_identity(
                digest, size_bytes, stored_bytes, compression, after
            )

        return await asyncio.to_thread(inspect)

    @staticmethod
    def _representation_stat(
        observed: os.stat_result,
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        return (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_nlink,
            observed.st_uid,
            observed.st_gid,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )

    @classmethod
    def _representation_identity(
        cls,
        digest: str,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
        observed: os.stat_result,
    ) -> str:
        fields = (
            "v1",
            digest,
            size_bytes,
            stored_bytes,
            compression,
            *cls._representation_stat(observed),
        )
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()

    def pin_evidence_representation(
        self,
        digest: str,
        pin: Path,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        """Promote one legacy representation to its canonical no-follow pin name."""
        path = self._authoritative_path(digest)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError:
            return None
        created = False
        retained_pin = False
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != stored_bytes:
                return None
            try:
                os.link(path, pin, follow_symlinks=False)
                created = True
            except FileExistsError:
                pass
            if opened.st_mode & 0o222:
                # Retained blobs are immutable after their no-clobber publication.  Make that
                # contract effective for ordinary uncoordinated writers as well; a caller that
                # deliberately chmods it back still changes ctime and fails the final fence.
                os.fchmod(descriptor, stat.S_IRUSR)
                os.fsync(descriptor)
            if created:
                self._fsync_directory(pin.parent)
            linked = os.fstat(descriptor)
            named = path.stat(follow_symlinks=False)
            pinned = pin.stat(follow_symlinks=False)
            observed = self._representation_stat(linked)
            if (
                self._representation_stat(named) != observed
                or self._representation_stat(pinned) != observed
            ):
                return None
            if created and path != pin:
                # The pin is now the authoritative retained representation, not verification
                # metadata for a mutable alias.  Sync it before retiring the legacy name so a
                # crash always leaves at least one durable link to the exact verified inode.
                retained_pin = True
                self._unlink_durable(path)
            retained = os.fstat(descriptor)
            pinned = pin.stat(follow_symlinks=False)
            if self._representation_stat(pinned) != self._representation_stat(retained):
                return None
            retained_pin = True
            return self._representation_identity(
                digest, size_bytes, stored_bytes, compression, retained
            )
        except OSError:
            return None
        finally:
            os.close(descriptor)
            if created and not retained_pin:
                with contextlib.suppress(OSError):
                    pin.unlink()
                    self._fsync_directory(pin.parent)

    def validate_evidence_pin(
        self,
        digest: str,
        pin: Path,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        """Validate the authoritative retained inode with no content read."""
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(pin, flags)
            pinned = os.fstat(descriptor)
            named_pin = pin.stat(follow_symlinks=False)
            observed = self._representation_stat(pinned)
            if (
                not stat.S_ISREG(pinned.st_mode)
                or pinned.st_size != stored_bytes
                or self._representation_stat(named_pin) != observed
            ):
                return None
            return self._representation_identity(
                digest, size_bytes, stored_bytes, compression, pinned
            )
        except OSError:
            return None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    async def collect_garbage(self) -> Sequence[str]:
        """Delete blobs nothing references. Mark and sweep, never refcounts.

        A refcount is a number that has to survive every crash in every path that touches it,
        and when it is wrong it is wrong silently in both directions.

        The descriptor remains capacity-accounting state while the owned file is unlinked and
        its parent directory fsynced. Only then is the row removed. A failed deletion therefore
        leaves both file and descriptor for a later pass; cancellation cannot split the two
        durability steps because each candidate runs to a joined known endpoint.

        Returns:
            The hashes that were collected.
        """
        await self.cleanup_staging_partials()
        if not await self.reconcile_acquisition_markers():
            return []
        if not await asyncio.to_thread(self._gc_root_is_scannable):
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
            pending_rows = (
                (
                    await session.execute(
                        select(models.Blob.hash, models.Blob.algo).where(
                            models.Blob.algo.like(f"{GC_PENDING_PREFIX}%")
                        )
                    )
                )
                .tuples()
                .all()
            )

        collected: list[str] = []
        visited: set[str] = set()
        for digest, algo in pending_rows:
            token = algo.removeprefix(GC_PENDING_PREFIX)
            visited.add(digest)
            if self._gc_identity_safe(digest, token) and await self._joined_async_call(
                self._run_gc_intent(digest, token)
            ):
                collected.append(digest)
        for digest, token in self._gc_intents():
            if digest in visited:
                continue
            visited.add(digest)
            if await self._joined_async_call(self._run_gc_intent(digest, token)):
                collected.append(digest)
        for digest in unreferenced:
            if digest in visited:
                continue
            token = await self._mark_gc_candidate(digest)
            if token is not None and await self._joined_async_call(
                self._run_gc_intent(digest, token)
            ):
                collected.append(digest)
        return collected

    def _gc_intents(self) -> list[tuple[str, str]]:
        root = self._gc_root()
        try:
            paths = list(root.glob("*.json")) if root.is_dir() else []
        except OSError:
            return []
        intents: list[tuple[str, str]] = []
        for path in paths:
            digest, separator, token = path.stem.partition(".")
            if separator and self._gc_identity_safe(digest, token):
                intents.append((digest, token))
        return intents

    def _gc_root_is_scannable(self) -> bool:
        try:
            return stat.S_ISDIR(self._gc_root().lstat().st_mode)
        except FileNotFoundError:
            return True
        except OSError:
            return False

    @staticmethod
    async def _blob_is_referenced(session: AsyncSession, digest: str) -> bool:
        references = union(
            select(models.Document.original_ref.label("ref")).where(
                models.Document.original_ref == digest
            ),
            select(models.DocumentVersion.original_ref.label("ref")).where(
                models.DocumentVersion.original_ref == digest
            ),
            select(models.AcquisitionRecord.blob_ref.label("ref")).where(
                models.AcquisitionRecord.blob_ref == digest
            ),
            select(models.AcquisitionMarker.blob_ref.label("ref")).where(
                models.AcquisitionMarker.blob_ref == digest
            ),
        ).subquery()
        return (
            await session.execute(select(references.c.ref).limit(1))
        ).scalar_one_or_none() is not None

    async def _mark_gc_candidate(self, digest: str) -> str | None:
        """Persist a counted deletion intent in one short writer transaction."""
        token = uuid4().hex
        async with self._sessions() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            row = await session.get(models.Blob, digest)
            if row is None:
                await session.rollback()
                return None
            if row.algo.startswith(GC_PENDING_PREFIX):
                await session.rollback()
                return row.algo.removeprefix(GC_PENDING_PREFIX)
            if await self._blob_is_referenced(session, digest):
                await session.rollback()
                return None
            row.algo = f"{GC_PENDING_PREFIX}{token}"
            await session.commit()
        return token

    @classmethod
    def _quarantine_durable(cls, destination: Path, quarantine: Path) -> None:
        """Move a content-addressed name aside without overwriting another owner."""
        cls._mkdir_durable(quarantine.parent)
        if destination.exists():
            with contextlib.suppress(FileExistsError):
                os.link(destination, quarantine)
            cls._fsync_directory(quarantine.parent)
            destination.unlink(missing_ok=True)
            cls._fsync_directory(destination.parent)

    @classmethod
    def _restore_quarantine(cls, destination: Path, quarantine: Path) -> None:
        """Restore a pending blob when a durable reference appeared before finalization."""
        if not quarantine.exists():
            return
        cls._mkdir_durable(destination.parent)
        with contextlib.suppress(FileExistsError):
            os.link(quarantine, destination)
        cls._fsync_directory(destination.parent)
        quarantine.unlink(missing_ok=True)
        cls._fsync_directory(quarantine.parent)

    async def _normalize_gc_row(self, digest: str, token: str) -> None:
        async with self._sessions() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            row = await session.get(models.Blob, digest)
            if row is not None and row.algo == f"{GC_PENDING_PREFIX}{token}":
                row.algo = "blake2b"
                await session.commit()
            else:
                await session.rollback()

    async def _remove_gc_artifacts(self, quarantine: Path, intent: Path) -> bool:
        if not await self._remove_owned_with_retry(quarantine):
            return False
        return await self._remove_owned_with_retry(intent)

    async def _remove_evidence_pin(self, digest: str) -> bool:
        pin = self._root / "evidence-pins" / "by-digest" / digest
        if not pin.parent.exists():
            return True
        return await self._remove_owned_with_retry(pin)

    async def _remove_evidence_pin_if_unowned(self, digest: str) -> bool:
        async with self._sessions() as session:
            if await session.get(models.Blob, digest) is not None:
                return True
        return await self._remove_evidence_pin(digest)

    async def _remove_legacy_alias_if_unowned(self, digest: str) -> bool:
        async with self._sessions() as session:
            if await session.get(models.Blob, digest) is not None:
                return True
        legacy = self.path_for(digest)
        try:
            await asyncio.to_thread(legacy.lstat)
        except FileNotFoundError:
            return True
        return await self._remove_owned_with_retry(legacy)

    @staticmethod
    def _read_gc_intent(path: Path) -> tuple[str, str, int] | None:
        try:
            decoded = cast("object", json.loads(path.read_text("utf-8")))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        payload = cast("dict[object, object]", decoded)
        if set(payload) != {"digest", "token", "stored_bytes"}:
            return None
        digest = payload["digest"]
        token = payload["token"]
        stored_bytes = payload["stored_bytes"]
        if type(digest) is not str or type(token) is not str or type(stored_bytes) is not int:
            return None
        if stored_bytes < 0:
            return None
        return digest, token, stored_bytes

    @staticmethod
    def _gc_intent_matches(path: Path, *, digest: str, token: str, stored_bytes: int) -> bool:
        return BlobStore._read_gc_intent(path) == (digest, token, stored_bytes)

    @staticmethod
    def _gc_representation_matches(
        path: Path,
        *,
        digest: str,
        compression: str,
        size_bytes: int,
        stored_bytes: int,
    ) -> bool:
        if (
            type(digest) is not str
            or type(compression) is not str
            or compression not in {"none", "gzip"}
            or type(size_bytes) is not int
            or type(stored_bytes) is not int
            or size_bytes < 0
            or stored_bytes < 0
        ):
            return False
        try:
            raw = path.read_bytes()
            if len(raw) != stored_bytes:
                return False
            data = gzip.decompress(raw) if compression == "gzip" else raw
        except (OSError, EOFError):
            return False
        return len(data) == size_bytes and content_hash(data) == digest

    def _gc_recovery_is_valid(
        self,
        stored: StoredBlob,
        destination: Path,
        quarantine: Path,
        intent: Path,
        token: str,
    ) -> bool:
        if (
            type(stored.hash) is not str
            or type(stored.compression) is not str
            or stored.compression not in {"none", "gzip"}
            or type(stored.size_bytes) is not int
            or type(stored.stored_bytes) is not int
            or stored.size_bytes < 0
            or stored.stored_bytes < 0
        ):
            return False
        intent_exists = intent.exists()
        if intent_exists and not self._gc_intent_matches(
            intent,
            digest=stored.hash,
            token=token,
            stored_bytes=stored.stored_bytes,
        ):
            return False
        if quarantine.exists():
            if not intent_exists:
                return False
            if not self._gc_representation_matches(
                quarantine,
                digest=stored.hash,
                compression=stored.compression,
                size_bytes=stored.size_bytes,
                stored_bytes=stored.stored_bytes,
            ):
                return False
        if destination.exists() and not self._gc_representation_matches(
            destination,
            digest=stored.hash,
            compression=stored.compression,
            size_bytes=stored.size_bytes,
            stored_bytes=stored.stored_bytes,
        ):
            return False
        return quarantine.exists() or destination.exists()

    async def _run_gc_intent(self, digest: str, token: str) -> bool:
        """Serialize managed collection with verification/publication inode pins."""
        async with self._durable_locks([f"blob:{digest}"]):
            return await self._run_gc_intent_locked(digest, token)

    async def _run_gc_intent_locked(self, digest: str, token: str) -> bool:  # noqa: PLR0911
        """Resume one deletion intent without holding SQLite across filesystem durability."""
        if not self._gc_identity_safe(digest, token):
            return False
        destination = self._authoritative_path(digest)
        quarantine, intent = self._gc_paths(digest, token)
        async with self._sessions() as session:
            row = await session.get(models.Blob, digest)
            if row is None or row.algo != f"{GC_PENDING_PREFIX}{token}":
                pending_row = None
                pending_stored = None
                referenced = False
            else:
                pending_row = row
                pending_stored = StoredBlob(
                    hash=row.hash,
                    size_bytes=row.size_bytes,
                    stored_bytes=row.stored_bytes,
                    compression=row.compression,
                )
                referenced = await self._blob_is_referenced(session, digest)
        if pending_row is None or pending_stored is None:
            pin_removed = await self._remove_evidence_pin_if_unowned(digest)
            alias_removed = await self._remove_legacy_alias_if_unowned(digest)
            artifacts_removed = await self._remove_gc_artifacts(quarantine, intent)
            return pin_removed and alias_removed and artifacts_removed
        if referenced:
            if not await asyncio.to_thread(
                self._gc_recovery_is_valid,
                pending_stored,
                destination,
                quarantine,
                intent,
                token,
            ):
                return False
            await self._durable_thread_call(
                lambda: self._restore_quarantine(destination, quarantine)
            )
            await self._normalize_gc_row(digest, token)
            await self._remove_owned_with_retry(intent)
            return False
        payload = json.dumps(
            {"digest": digest, "stored_bytes": pending_stored.stored_bytes, "token": token},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        await self._durable_thread_call(lambda: self._write_durable(intent, payload))
        try:
            await self._durable_thread_call(
                lambda: self._quarantine_durable(destination, quarantine)
            )
        except Exception:  # noqa: BLE001 - the counted intent is restart-recoverable
            return False
        async with self._sessions() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            current = await session.get(models.Blob, digest)
            if current is None or current.algo != f"{GC_PENDING_PREFIX}{token}":
                await session.rollback()
            elif await self._blob_is_referenced(session, digest):
                current_stored = StoredBlob(
                    hash=current.hash,
                    size_bytes=current.size_bytes,
                    stored_bytes=current.stored_bytes,
                    compression=current.compression,
                )
                await session.rollback()
                if not await asyncio.to_thread(
                    self._gc_recovery_is_valid,
                    current_stored,
                    destination,
                    quarantine,
                    intent,
                    token,
                ):
                    return False
                await self._durable_thread_call(
                    lambda: self._restore_quarantine(destination, quarantine)
                )
                await self._normalize_gc_row(digest, token)
                await self._remove_owned_with_retry(intent)
                return False
            else:
                # The immediate transaction prevents a new durable reference between this
                # last reference probe and row deletion.  Remove the inode pin first so a
                # failed unlink leaves a retryable, capacity-accounted Blob row rather than a
                # stale pin that could conflict with later storage of the same content hash.
                if not await self._remove_evidence_pin(digest):
                    await session.rollback()
                    return False
                await session.execute(delete(models.Blob).where(models.Blob.hash == digest))
                await session.commit()
        pin_removed = await self._remove_evidence_pin_if_unowned(digest)
        alias_removed = await self._remove_legacy_alias_if_unowned(digest)
        artifacts_removed = await self._remove_gc_artifacts(quarantine, intent)
        return pin_removed and alias_removed and artifacts_removed

    async def orphaned_files(self) -> Sequence[Path]:
        """Files on disk that no ``blobs`` row claims.

        The other half of the sweep: a crash between deleting the row and unlinking the file
        leaks the file, and this is what finds it.
        """
        if not self._root.exists():
            return []
        async with self._sessions() as session:
            known = set((await session.execute(select(models.Blob.hash))).scalars().all())
        roots = (self._root / "blake2b", self._root / "evidence-pins" / "by-digest")
        return [
            path
            for root in roots
            if root.exists()
            for path in root.rglob("*")
            if path.is_file() and path.name not in known
        ]


__all__ = [
    "MAX_ORIGINAL_BYTES",
    "BlobStore",
    "OmittedBlob",
    "StagingCleanup",
    "StoredBlob",
    "should_compress",
]
