"""The scheduled sweep that removes vectors SQLite has already forgotten.

Tombstones are written by a trigger inside the transaction that deletes a chunk, so the list
is never behind the truth. This is the pass that reads them.

**It is scheduled by :meth:`~manicule.app.served.Scheduler._run_sweep`**, on
``ingest.sweep_interval_s``, and reachable by hand through ``manicule sweep-vectors``. Said
here because this module spent a while describing a runner nothing ran: the trigger wrote
tombstones, the list only grew, and every deleted chunk kept its vector — competing for
top-``k`` slots ahead of the join that hides it — while three settings described a cadence
that did not exist.

**It reads tombstones. It never anti-joins.** Sweeping by comparing every id in the vector
table against ``chunks`` races concurrent ingest: an id written after the scan began looks
like an orphan, and the sweep deletes a live vector — a chunk that is served by the lexical
leg, missing from the dense one, and wrong in a way no error reports. A tombstone list only
ever names something that *was* deleted, so it cannot make that mistake. It is also cheap: a
small table instead of the whole index.

**Scheduled, not triggered by deletion.** Otherwise a large reconciliation produces a sweep
storm during a sync.

**What it yields to, stated as it is rather than as it should be.** The caller takes the
derived-mutation guard, which serializes a pass against a reset, a rebuild and a re-embed
publication — the operations that move the publication pointer underneath it. It is *not*
serialized against a running sync, and the tombstone design is what makes that safe rather
than merely tolerable: the list only ever names ids that were already deleted, so a vector
written after the pass began cannot be in it. A hot backup *is* excluded: the caller holds the
backup lock (:mod:`manicule.storage.backup_lock`) shared for the whole pass and hands it in as
the :class:`SweepGate`, so a backup waits for a running pass and a pass that finds a backup
running declines.

The soft-delete pass is the same sweep's second half. Within the grace period a restore is
free. After it, the document is *purged*
(:meth:`~manicule.ingest.ports.IngestStore.purge_document`): its chunks and vectors go, and so
does its claim on the raw bytes it was built from — the reference on the document and on each
of its versions is cleared in the same transaction as the status flip. A restore after that is
a re-sync from the source, not a re-parse. The bytes are given back because a deleted document
that pinned them forever would make "deleted" mean "no longer searchable" and nothing more.
The trade is real rather than free: unbounded free restore means unbounded dilution of every
vector search, because a soft-deleted chunk is still in the vector table competing for top-``k``
slots before the join that hides it can run. This applies to every document past the grace
period, whatever soft-deleted it.

**Blob collection is the pass's last step**, and it is what turns a released reference into a
freed file. The references are roots until something stops naming them, and the acquisition
journal that also names the bytes lapses on its own schedule (30 days after the last run that
listed the document settled, cleaned at the next durable sync). The file goes once both the
purge and the journal have let go; the clocks run side by side, so for a document the source
deleted that is about the grace period, not the sum of the two. Collection runs on every
unblocked pass rather than only after a purge, because roots lapse through paths this pass does
not see — the journal cleanup, an operator's history release — and a collection that waited for
a purge would leave those blobs on disk until some unrelated document happened to expire. One
with nothing to do is cheap: bounded staging and marker housekeeping and one anti-join over the
``blobs`` primary key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.ingest.ports import IngestStore

PURGE_DETAIL: Final = (
    "soft-delete grace period expired; chunks and vectors purged, retained bytes released"
)
"""The ``status_detail`` of a document the sweep has purged."""

UNREFERENCED_BLOB_MIN_AGE_S: Final = 24 * 3600.0
"""How old an unreferenced blob must be before the sweep's collection will delete it.

A blob is written before the document that references it commits, and a collection running
beside ingest must not delete one in that window. A day is far beyond any single document's
parse and embed, and matches the staging-partial staleness bound in
:mod:`manicule.storage.blobs`. The blobs this collection exists for — a purged document's, a
lapsed journal's — are weeks old by construction.
"""


@runtime_checkable
class VectorSweepTarget(Protocol):
    """The vector-store surface a sweep needs.

    Narrower than :class:`~manicule.core.protocols.VectorStore` on purpose: a sweep deletes
    and never searches, so nothing here can be the route by which a mismatched model gets its
    table opened.
    """

    async def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    async def delete_document(self, document_id: str) -> None: ...


@runtime_checkable
class BlobCollector(Protocol):
    """The blob-store surface the sweep's last step needs: delete what nothing references."""

    async def collect_garbage(self, *, min_age_s: float = 0.0) -> Sequence[str]: ...


class SweepGate(Protocol):
    """Whether the sweep may run right now.

    A protocol rather than a boolean argument because what blocks a sweep is held by a
    different component, and a sweep scheduled on a timer has to ask at the moment it fires
    rather than at the moment it was scheduled. The production gate is
    :class:`~manicule.storage.backup_lock.MaintenanceHold`: it answers "permitted" for exactly
    as long as it holds the backup lock, which the caller keeps for the whole pass.
    """

    def sweep_permitted(self) -> str:
        """Empty when the sweep may run, otherwise what is blocking it."""
        ...


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What one pass removed, and why it stopped if it did."""

    vectors_removed: int = 0
    documents_purged: int = 0
    blobs_collected: int = 0
    """Blobs deleted by the collection that ends the pass. A count: never the hashes."""
    blocked_by: str = ""

    @property
    def ran(self) -> bool:
        return not self.blocked_by


async def sweep_vectors(
    store: IngestStore,
    vectors: VectorSweepTarget,
    *,
    batch: int = 1000,
    soft_delete_grace_s: float = 30 * 24 * 3600.0,
    gate: SweepGate | None = None,
    blobs: BlobCollector | None = None,
    blob_min_age_s: float = UNREFERENCED_BLOB_MIN_AGE_S,
) -> SweepResult:
    """Retire tombstones, purge documents whose grace period has expired, collect blobs.

    Ordering matters and is the same ordering everything else in storage uses: the derived
    store is cleaned **before** the record of what to clean is dropped. A crash between the
    two leaves a tombstone naming a vector that is already gone, and the next pass deletes
    nothing and clears it — an idempotent no-op. The reverse order leaves a live vector with
    nothing left to say it should not be there, which is the failure that costs correctness
    rather than the one that costs a wasted pass.

    Args:
        store: Where the tombstones and the soft-deleted documents are.
        vectors: The store to remove from.
        batch: Tombstones retired per pass, so one sweep cannot monopolize the writer.
        soft_delete_grace_s: How long a soft-deleted document's chunks survive.
        gate: Asked, at the moment the sweep fires, whether it may run. A gate that is a lock
            must be held by the caller for the whole call; asking it again during the pass is
            a check that the hold is still there, not a way to acquire it.
        blobs: What to collect unreferenced blobs with, as the pass's last step. ``None`` skips
            it — a caller with no blob store, or one that runs collection itself.
        blob_min_age_s: Unreferenced blobs younger than this are left alone, because a
            publication may be about to reference them.

    Returns:
        What was removed, or what blocked the pass.
    """
    blocked = gate.sweep_permitted() if gate is not None else ""
    if blocked:
        return SweepResult(blocked_by=blocked)

    removed = 0
    purged = 0
    tombstoned = await store.take_tombstones(batch)
    if tombstoned:
        await vectors.delete_chunks(list(tombstoned))
        await store.clear_tombstones(tombstoned)
        removed = len(tombstoned)

    cutoff = datetime.now(UTC) - timedelta(seconds=soft_delete_grace_s)
    expired = await store.soft_deleted_before(cutoff, limit=batch)
    for document_id in expired:
        blocked = gate.sweep_permitted() if gate is not None else ""
        if blocked:
            # Asked again inside the loop, not only at the top. A purge of a thousand documents
            # is thousands of writes, and this is the check that the hold on the backup lock is
            # still there for each of them. The lock is held for the whole pass, so a backup
            # that arrives mid-purge waits instead of overlapping; this declines only if the
            # hold was lost.
            return SweepResult(vectors_removed=removed, documents_purged=purged, blocked_by=blocked)
        # **The store purges, and the vectors follow.** The purge is one transaction that
        # re-checks the document is still in the trash — a restore that landed after the
        # selection above wins and nothing is touched — then deletes its chunks, marks it
        # `deleted` and releases the retained bytes. Deleting the chunks writes tombstones for
        # their vectors in that same transaction, so the vector deletion below is the prompt
        # path rather than the only one: a crash between the commit and the call leaves the
        # vectors to the next pass, which reads the tombstones. The reverse order — vectors
        # first — would remove vectors from a document a racing restore then keeps.
        #
        # **And the status flip is what makes the sweep terminate.** `soft_deleted_before`
        # selects on `deleted_at`, which purging does not change, and excludes `deleted`
        # documents; without the flip the same first `batch` would be re-selected on every pass,
        # re-purged and re-counted forever, and anything past them never reached. The release
        # of the bytes is in the same transaction because of that: once the status says purged
        # the document is never selected again, so a crash between the two would pin the bytes
        # permanently.
        if not await store.purge_document(document_id, reason=PURGE_DETAIL, deleted_before=cutoff):
            continue
        await vectors.delete_document(document_id)
        purged += 1

    collected = 0
    if blobs is not None:
        blocked = gate.sweep_permitted() if gate is not None else ""
        if blocked:
            return SweepResult(vectors_removed=removed, documents_purged=purged, blocked_by=blocked)
        collected = len(await blobs.collect_garbage(min_age_s=blob_min_age_s))

    return SweepResult(vectors_removed=removed, documents_purged=purged, blobs_collected=collected)


__all__ = [
    "PURGE_DETAIL",
    "UNREFERENCED_BLOB_MIN_AGE_S",
    "BlobCollector",
    "SweepGate",
    "SweepResult",
    "VectorSweepTarget",
    "sweep_vectors",
]
