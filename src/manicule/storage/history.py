"""A document's timeline: what it used to say, and what happens when it is thrown away.

Versions and the trash live together because they answer the same question from two ends.
A version row says "this document used to hold different bytes"; a trash row says "this
document is on its way out". Both exist so that something which is no longer current is still
*explicable* — a citation that stops resolving says why, and a deletion says how much of it
can still be undone.

**Nothing outside this module writes a version.** History is recorded inside the transaction
that supersedes a document (:meth:`VersionsMixin._record_supersession`, called by
``upsert_document``), because that is the only place both states are visible at once and the
only place the write cannot be forgotten. A public "record a version" verb would be a second
author of a monotonic counter.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import case, delete, func, select, update

from manicule.core.content import Document, DocumentStatus
from manicule.core.organization import (
    CitationResolution,
    CitationState,
    DocumentVersion,
    Restoration,
    TrashEntry,
)
from manicule.storage import models
from manicule.storage.rows import to_chunk, to_document
from manicule.storage.scoped import WorkspaceScoped
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession

    from manicule.core.content import Metadata

DEFAULT_VERSION_BYTES_RETENTION_S = 30 * 24 * 3600.0
"""How long a superseded version's retained bytes are kept, by default.

``docs/storage.md`` §7 sets the policy: the live document's bytes are kept until the document
is purged from the trash, and a prior version's for a bounded window, because a chatty wiki
otherwise grows the blob store without bound. It is a default rather than a configuration
setting because nothing schedules the pass that would read one —
:meth:`VersionsMixin.release_expired_versions` is reached by an operator, not a timer (blob
collection, the pass that then deletes the files, *is* scheduled: ``docs/storage.md`` §8.2) —
and a setting nothing reads is a promise nothing keeps.
"""

RELEASED_AFTER_GRACE_REASON = "bytes released after the soft-delete grace period expired"
"""The ``documents.original_omitted_reason`` :meth:`TrashMixin.purge_document` records.

A constant, and compared exactly, because "the bytes were never kept" and "the bytes were kept
and have since been released" are different facts that a bare ``NULL`` ``original_ref`` cannot
tell apart, and the restore message and the citation resolution each say a different thing
depending on which it is.
"""

_VERSIONED_FIELDS = ("uri", "title", "media_type", "version_token", "content_hash", "status")
"""What a version's ``changes`` records the movement of.

Diagnostic, deliberately: it says what moved between one state and the next, never how to move
it back. A patch would be a second, weaker representation of the retained bytes, which are the
real answer to "what did it used to say".
"""


class VersionsMixin(WorkspaceScoped):
    """:class:`~manicule.core.protocols.VersionStore` over SQLite."""

    async def list_versions(self, document_id: str) -> Sequence[DocumentVersion]:
        async with self._sessions() as session:
            if await self._any_document(session, document_id) is None:
                return []
            rows = (
                (
                    await session.execute(
                        select(models.DocumentVersion)
                        .where(models.DocumentVersion.document_id == document_id)
                        .order_by(models.DocumentVersion.version)
                    )
                )
                .scalars()
                .all()
            )
            return [_to_version(row) for row in rows]

    async def get_version(self, document_id: str, version: int) -> DocumentVersion | None:
        async with self._sessions() as session:
            if await self._any_document(session, document_id) is None:
                return None
            row = (
                await session.execute(
                    select(models.DocumentVersion).where(
                        models.DocumentVersion.document_id == document_id,
                        models.DocumentVersion.version == version,
                    )
                )
            ).scalar_one_or_none()
            return None if row is None else _to_version(row)

    async def current_version(self, document_id: str) -> int:
        async with self._sessions() as session:
            if await self._any_document(session, document_id) is None:
                return 0
            return await self._next_version(session, document_id)

    async def resolve_citation(self, document_id: str, chunk_id: str) -> CitationResolution:
        async with self._sessions() as session:
            row = await self._any_document(session, document_id)
            if row is None:
                return CitationResolution(
                    state=CitationState.UNKNOWN,
                    reason=(
                        f"no document {document_id!r} in workspace {self._workspace_id!r}, so "
                        f"there is no history to explain chunk {chunk_id!r}"
                    ),
                )

            document = to_document(row)
            chunk_row = (
                await session.execute(
                    select(models.Chunk).where(
                        models.Chunk.id == chunk_id,
                        models.Chunk.document_id == document_id,
                    )
                )
            ).scalar_one_or_none()

            if row.deleted_at is not None:
                purged = row.status is DocumentStatus.DELETED
                return CitationResolution(
                    state=CitationState.DELETED,
                    document=document,
                    reason=(
                        f"document {document_id!r} is in the trash"
                        + (
                            f"; its content has been purged, and {_restore_cost(row)}"
                            if purged
                            else "; restoring it costs nothing while the grace period lasts"
                        )
                    ),
                )

            if chunk_row is not None:
                return CitationResolution(
                    state=CitationState.PRESENT,
                    chunk=to_chunk(chunk_row),
                    document=document,
                    reason="the chunk is stored and its document is not in the trash",
                )

            superseded = await self._latest_supersession(session, document_id)
            if superseded is None:
                return CitationResolution(
                    state=CitationState.UNKNOWN,
                    document=document,
                    reason=(
                        f"chunk {chunk_id!r} is not stored against document {document_id!r}, "
                        f"and the document has never been superseded — so this citation names "
                        f"text no recorded state of the document contained"
                    ),
                )
            return CitationResolution(
                state=CitationState.SUPERSEDED,
                document=document,
                reason=(
                    f"document {document_id!r} was re-ingested; it is now on version "
                    f"{superseded.version + 1} and the text chunk {chunk_id!r} named is not in "
                    f"it. A chunk id is derived from its own text and position, so a chunk that "
                    f"survived the re-parse kept its id and this one did not — the citation is "
                    f"absent rather than re-pointed at whatever replaced it"
                ),
            )

    async def release_expired_versions(self, cutoff: datetime, *, limit: int = 1000) -> int:
        """Let go of retained bytes for versions superseded before ``cutoff``.

        The rows stay. What is released is ``original_ref``, which pins a blob against
        :meth:`~manicule.storage.blobs.BlobStore.collect_garbage` for as long as it is set —
        so without this, recording versions would grow the blob store without bound and repeal
        the retention policy in ``docs/storage.md`` §7 by accident.

        The release is recorded on the row rather than left to be inferred, for the same reason
        ``documents.original_omitted_reason`` exists: "the bytes were never kept" and "the
        bytes were kept and have now been reclaimed" are different facts, and a bare ``NULL``
        is both.
        """
        released = 0
        async with self._sessions.begin() as session:
            rows = (
                (
                    await session.execute(
                        select(models.DocumentVersion)
                        .join(
                            models.Document,
                            models.Document.id == models.DocumentVersion.document_id,
                        )
                        .where(
                            models.Document.workspace_id == self._workspace_id,
                            models.DocumentVersion.created_at < cutoff,
                            models.DocumentVersion.original_ref.is_not(None),
                        )
                        .order_by(models.DocumentVersion.created_at)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                changes: dict[str, Any] = dict(cast("Any", row.changes) or {})
                changes["bytes_released_at"] = utcnow().isoformat()
                row.changes = cast("Any", changes)
                row.original_ref = None
                released += 1
        return released

    # --- written by the store, never by a caller ----------------------------------------------

    async def _record_supersession(
        self, session: AsyncSession, row: models.Document, incoming: Document
    ) -> None:
        """Record the state ``row`` is about to leave, if it is leaving one.

        Called from inside ``upsert_document``'s transaction, so the version and the write that
        superseded it either both happen or neither does. A caller-driven equivalent would be
        two transactions with a crash window between them, and the crash would lose exactly the
        history that exists to explain the change.

        **Keyed on ``content_hash``, not on "an upsert happened".** Ingest writes a document row
        far more often than a document changes — a status transition, a re-seen skip, a repair
        that only marks it indexed — and a version per write would fill the history with rows
        recording nothing, each pinning a blob. The hash is what ``docs/ingest.md`` §4.2 already
        uses to decide whether anything changed at all.
        """
        if row.content_hash == incoming.content_hash:
            return
        version = await self._next_version(session, row.id)
        chunk_count = (
            await session.execute(
                select(func.count())
                .select_from(models.Chunk)
                .where(models.Chunk.document_id == row.id)
            )
        ).scalar_one()
        session.add(
            models.DocumentVersion(
                id=str(uuid.uuid4()),
                document_id=row.id,
                version=version,
                content_hash=row.content_hash,
                original_ref=row.original_ref,
                chunk_count=chunk_count,
                changes=cast("Any", _changes(row, incoming)),
            )
        )

    async def _next_version(self, session: AsyncSession, document_id: str) -> int:
        highest = (
            await session.execute(
                select(func.max(models.DocumentVersion.version)).where(
                    models.DocumentVersion.document_id == document_id
                )
            )
        ).scalar_one_or_none()
        return (highest or 0) + 1

    async def _latest_supersession(
        self, session: AsyncSession, document_id: str
    ) -> models.DocumentVersion | None:
        return (
            await session.execute(
                select(models.DocumentVersion)
                .where(models.DocumentVersion.document_id == document_id)
                .order_by(models.DocumentVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()


def _restore_cost(row: models.Document) -> str:
    """What bringing a purged document back costs, from what its row says survived.

    Three answers, because three different things leave ``original_ref`` unset or set on a
    purged row. A reference still present is a purge from before the sweep released bytes (or
    one the sweep has not yet reached), and the cheap re-parse is real. The released reason is
    a purge that gave the bytes back, and only the source can supply them again. Anything else
    is a document whose bytes were never retained, with ``original_omitted_reason`` saying why.
    """
    if row.original_ref is not None:
        return "re-parse it from its retained bytes to serve it again"
    if row.original_omitted_reason == RELEASED_AFTER_GRACE_REASON:
        return (
            "its retained bytes were released with it when the grace period ended, so only a "
            "re-sync from the source can bring its content back"
        )
    return (
        "its bytes were not retained, so only a re-sync from the source can bring its content back"
    )


class TrashMixin(WorkspaceScoped):
    """:class:`~manicule.core.protocols.TrashStore` over SQLite."""

    async def soft_delete_document(self, document_id: str) -> None:
        """Move a document to the trash without touching the derived stores.

        Chunks, vectors and lexical rows all stay and become invisible at the hydrating join,
        which is what makes a restore inside the grace period free.

        **A second soft delete does not restart the clock.** Refreshing ``deleted_at`` would let
        a caller postpone the sweep for ever by deleting the same document repeatedly — and
        reconciliation soft-deletes on every pass over a source that no longer has the document,
        so "repeatedly" is the normal case rather than an abuse. A grace period that never
        expires is unbounded dilution of every vector search, which is the cost the period
        exists to bound.
        """
        async with self._sessions.begin() as session:
            row = await self._any_document(session, document_id)
            if row is not None and row.deleted_at is None:
                row.deleted_at = utcnow()

    async def purge_document(
        self, document_id: str, *, reason: str, deleted_before: datetime | None = None
    ) -> bool:
        """Finish a soft delete: drop the content, release the retained bytes, mark it purged.

        **One transaction, because the sweep cannot retry a half of it.** The selection of what
        to purge (:meth:`~manicule.storage.docstore.SqliteDocStore.soft_deleted_before`) skips
        ``deleted`` rows, so a crash between "flip the status" and "release the bytes" would
        leave a purged document that pins its bytes for ever and is never looked at again.
        Everything below commits together or not at all.

        **The check and the write are one statement.** The ``UPDATE`` carries the predicate —
        still in the trash, not already purged, and, when ``deleted_before`` is given, still
        deleted *before* that cutoff — and its row count is the answer. A restore that lands
        between the sweep's selection and this call has cleared ``deleted_at``, matches
        nothing, and wins: the document keeps its chunks, its status and its bytes. A restore
        followed by a second delete inside that window restarts the clock, and the cutoff is
        what keeps the sweep from purging it with no grace period at all.

        **What it does, when it matches:**

        - sets ``status = 'deleted'`` with ``reason`` as the detail, and clears ``failed_stage``
          as the schema's iff constraint requires;
        - deletes the document's chunks, which writes their vector tombstones in the same
          transaction (the trigger in ``docs/storage.md`` §8.2), so a crash after the commit
          leaves the vectors to the next sweep rather than orphaned;
        - sets ``documents.original_ref`` to ``NULL`` and, only if there was a reference, sets
          ``original_omitted_reason`` to :data:`RELEASED_AFTER_GRACE_REASON` — a document whose
          bytes were never kept keeps the reason it already had;
        - clears ``original_ref`` on every version of the document, stamping
          ``bytes_released_at`` on the row and in ``changes`` — the two places
          :meth:`VersionsMixin.release_expired_versions` and the source-history release each
          record it.

        The blobs themselves are not touched. Releasing the references is what makes them
        unreferenced, and :meth:`~manicule.storage.blobs.BlobStore.collect_garbage` deletes
        them — once the acquisition journal's own references to them have aged out as well.

        Not for ``reset_derived``'s documents, which are also ``deleted`` with ``deleted_at``
        set and whose retained bytes a rebuild needs: the predicate on ``status`` excludes them.

        Args:
            document_id: The document to purge.
            reason: What to record in ``status_detail``.
            deleted_before: Purge only if the document was soft-deleted before this instant.

        Returns:
            ``True`` if the document was purged. ``False`` if it was restored, already purged,
            or not found — in every case there is nothing more for the caller to do.
        """
        predicates = [
            models.Document.id == document_id,
            models.Document.workspace_id == self._workspace_id,
            models.Document.deleted_at.is_not(None),
            models.Document.status != DocumentStatus.DELETED,
        ]
        if deleted_before is not None:
            predicates.append(models.Document.deleted_at < deleted_before)
        async with self._sessions.begin() as session:
            marked = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.Document)
                    .where(*predicates)
                    .values(
                        status=DocumentStatus.DELETED,
                        status_detail=reason,
                        failed_stage=None,
                        # Evaluated against the row as it was, so the reference that is about to be
                        # cleared is what decides whether there is a release to record.
                        original_omitted_reason=case(
                            (
                                models.Document.original_ref.is_not(None),
                                RELEASED_AFTER_GRACE_REASON,
                            ),
                            else_=models.Document.original_omitted_reason,
                        ),
                        original_ref=None,
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            if marked.rowcount != 1:
                return False
            await session.execute(
                delete(models.Chunk).where(models.Chunk.document_id == document_id)
            )
            released_at = utcnow()
            versions = (
                (
                    await session.execute(
                        select(models.DocumentVersion).where(
                            models.DocumentVersion.document_id == document_id,
                            models.DocumentVersion.original_ref.is_not(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
            for version in versions:
                changes: dict[str, Any] = dict(cast("Any", version.changes) or {})
                changes["bytes_released_at"] = released_at.isoformat()
                version.changes = cast("Any", changes)
                version.original_ref = None
                version.bytes_released_at = released_at
        return True

    async def restore_document(self, document_id: str) -> Restoration:
        """Take a document out of the trash, and say what that achieved."""
        async with self._sessions.begin() as session:
            row = await self._any_document(session, document_id)
            if row is None:
                return Restoration(
                    document_id=document_id,
                    restored=False,
                    reason=f"no document {document_id!r} in workspace {self._workspace_id!r}",
                )
            if row.deleted_at is None:
                return Restoration(
                    document_id=document_id,
                    restored=False,
                    reason=f"document {document_id!r} is not in the trash",
                )

            rival = await self._live_identity_holder(session, row)
            if rival is not None:
                # Unreachable while ids come from `manicule.core.ids.document_id`, which derives
                # from the same three components the identity index is over — so a rival would
                # have to *be* this row. Checked anyway, because the alternative is the partial
                # unique index raising an IntegrityError that names a constraint instead of the
                # document standing in the way.
                return Restoration(
                    document_id=document_id,
                    restored=False,
                    reason=(
                        f"document {rival!r} already occupies "
                        f"({row.source!r}, {row.source_id!r}) in this workspace; restoring "
                        f"{document_id!r} would make two live documents claim one source"
                    ),
                )

            purged = row.status is DocumentStatus.DELETED
            row.deleted_at = None
            if not purged:
                return Restoration(
                    document_id=document_id,
                    restored=True,
                    reason=(
                        "restored inside the grace period; its chunks and vectors were never "
                        "removed"
                    ),
                )

            # The sweep has already taken the content. The row comes back empty, and `pending`
            # is exactly what that state means elsewhere: nothing has claimed it, retrieval does
            # not serve it, and a repair pass selects it.
            row.status = DocumentStatus.PENDING
            row.status_detail = "restored from the trash after the sweep purged its content"
            row.failed_stage = None
            return Restoration(
                document_id=document_id,
                restored=True,
                needs_reparse=True,
                reason=(
                    f"restored after the grace period, so it holds no chunks: {_restore_cost(row)}"
                ),
            )

    async def list_trash(
        self, *, grace_s: float, limit: int = 100, offset: int = 0
    ) -> Sequence[TrashEntry]:
        """What is in the trash, longest-deleted first — the order the sweep will take them in."""
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Document)
                        .where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_not(None),
                        )
                        .order_by(models.Document.deleted_at, models.Document.id)
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
            entries: list[TrashEntry] = []
            for row in rows:
                deleted_at = row.deleted_at
                if deleted_at is None:  # pragma: no cover - the predicate above excludes it
                    continue
                purged = row.status is DocumentStatus.DELETED
                entries.append(
                    TrashEntry(
                        document=to_document(row),
                        deleted_at=deleted_at,
                        purged=purged,
                        restorable_until=(
                            None if purged else deleted_at + timedelta(seconds=grace_s)
                        ),
                    )
                )
            return entries

    async def _live_identity_holder(
        self, session: AsyncSession, row: models.Document
    ) -> str | None:
        """Another live document already holding this one's ``(source, source_id)``."""
        return (
            await session.execute(
                select(models.Document.id).where(
                    models.Document.workspace_id == self._workspace_id,
                    models.Document.source == row.source,
                    models.Document.source_id == row.source_id,
                    models.Document.deleted_at.is_(None),
                    models.Document.id != row.id,
                )
            )
        ).scalar_one_or_none()


def _changes(row: models.Document, incoming: Document) -> Metadata:
    """What moved between the outgoing state and the one replacing it."""
    changed: dict[str, Any] = {}
    for field in _VERSIONED_FIELDS:
        before = getattr(row, field)
        after = getattr(incoming, field)
        before_value = before.value if isinstance(before, DocumentStatus) else before
        after_value = after.value if isinstance(after, DocumentStatus) else after
        if before_value != after_value:
            changed[field] = {"from": before_value, "to": after_value}
    return cast("Metadata", changed)


def _to_version(row: models.DocumentVersion) -> DocumentVersion:
    return DocumentVersion(
        id=row.id,
        document_id=row.document_id,
        version=row.version,
        content_hash=row.content_hash,
        original_ref=row.original_ref,
        chunk_count=row.chunk_count,
        changes=cast("Metadata", row.changes or {}),
        superseded_at=row.created_at,
    )


__all__ = [
    "DEFAULT_VERSION_BYTES_RETENTION_S",
    "TrashMixin",
    "VersionsMixin",
]
