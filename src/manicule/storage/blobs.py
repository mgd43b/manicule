"""Retained original bytes, so re-parsing never means re-fetching.

This is rung 3 of the blast-radius ladder, and the only thing standing between a parser bug
fix and a full re-crawl of a rate-limited API. Every other rung is a pure function of what is
already on disk; re-fetching is the one repair that can fail for reasons outside the machine,
and the one whose result is not reproducible.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import delete, select, union

from manicule.core.content import Retention
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

        digest = content_hash(data)
        destination = self.path_for(digest)
        compression = "gzip" if should_compress(media_type) else "none"
        payload = gzip.compress(data, mtime=0) if compression == "gzip" else data

        if not destination.exists():
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination)

        stored = StoredBlob(
            hash=digest,
            size_bytes=len(data),
            stored_bytes=len(payload),
            compression=compression,
        )
        async with self._sessions.begin() as session:
            if await session.get(models.Blob, digest) is None:
                session.add(
                    models.Blob(
                        hash=digest,
                        algo="blake2b",
                        media_type=media_type,
                        size_bytes=stored.size_bytes,
                        stored_bytes=stored.stored_bytes,
                        compression=compression,
                    )
                )
        return stored

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
        return [path for path in self._root.rglob("*") if path.is_file() and path.name not in known]


__all__ = ["MAX_ORIGINAL_BYTES", "BlobStore", "OmittedBlob", "StoredBlob", "should_compress"]
