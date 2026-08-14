"""The scheduled sweep that removes vectors SQLite has already forgotten.

Tombstones are written by a trigger inside the transaction that deletes a chunk, so the list
is never behind the truth. Nothing had a *runner*; this is it.

**It reads tombstones. It never anti-joins.** Sweeping by comparing every id in the vector
table against ``chunks`` races concurrent ingest: an id written after the scan began looks
like an orphan, and the sweep deletes a live vector — a chunk that is served by the lexical
leg, missing from the dense one, and wrong in a way no error reports. A tombstone list only
ever names something that *was* deleted, so it cannot make that mistake. It is also cheap: a
small table instead of the whole index.

**Scheduled, not triggered by deletion.** Otherwise a large reconciliation produces a sweep
storm during a sync. And it yields, always, to two things:

* **a backup** — the backup lock blocks exactly this sweep and the blob GC, because they are
  the only two operations that remove data;
* **an active sync** — so an ingest run and a purge are never competing for the same vector
  table.

The soft-delete pass is the same sweep's second half. Within the grace period a restore is
free; after it, the document's chunks are purged and a restore costs a re-parse from retained
bytes — rung 3, still not a re-crawl. That is a real trade rather than a free lunch:
unbounded free restore means unbounded dilution of every vector search, because a soft-deleted
chunk is still in the vector table competing for top-``k`` slots before the join that hides it
can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from manicule.core.content import DocumentStatus

if TYPE_CHECKING:
    from manicule.ingest.ports import IngestStore


class VectorSweepTarget(Protocol):
    """The vector-store surface a sweep needs.

    Narrower than :class:`~manicule.core.protocols.VectorStore` on purpose: a sweep deletes
    and never searches, so nothing here can be the route by which a mismatched model gets its
    table opened.
    """

    async def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    async def delete_document(self, document_id: str) -> None: ...


class SweepGate(Protocol):
    """Whether the sweep may run right now.

    A protocol rather than a boolean argument because the two things that block a sweep —
    a running backup and an active sync — are held by different components, and a sweep
    scheduled on a timer has to ask at the moment it fires rather than at the moment it was
    scheduled.
    """

    def sweep_permitted(self) -> str:
        """Empty when the sweep may run, otherwise what is blocking it."""
        ...


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What one pass removed, and why it stopped if it did."""

    vectors_removed: int = 0
    documents_purged: int = 0
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
) -> SweepResult:
    """Retire tombstones, then purge documents whose grace period has expired.

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
        gate: Asked, at the moment the sweep fires, whether it may run.

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
            # is thousands of writes to the vector table, and a backup that starts halfway
            # through is exactly what the gate exists to get out of the way of.
            return SweepResult(vectors_removed=removed, documents_purged=purged, blocked_by=blocked)
        # Vectors first, chunks second, which is the same ordering as above and for the same
        # reason. Dropping the chunks writes tombstones for vectors that are already gone;
        # the next pass deletes nothing and clears them, which is an idempotent no-op. The
        # reverse order would leave live vectors with nothing recording that they should go.
        await vectors.delete_document(document_id)
        await store.replace_chunks(document_id, [])
        # **And then the document says it has been purged.** Without this the sweep never
        # terminates: `soft_deleted_before` selects on `deleted_at`, which purging does not
        # change, so the same documents come back on every pass — re-deleting vectors that are
        # already gone, re-emptying chunks that are already empty, and reporting the same
        # `documents_purged` forever. Worse, the ordered `LIMIT` means the same first `batch`
        # are re-selected every time, so anything past the first thousand is never purged at
        # all. `deleted` is the status that already means exactly this: the row is retained so
        # a citation can explain itself, and its content is gone.
        await store.set_status(
            document_id,
            DocumentStatus.DELETED,
            "soft-delete grace period expired; chunks and vectors purged",
        )
        purged += 1

    return SweepResult(vectors_removed=removed, documents_purged=purged)


__all__ = ["SweepGate", "SweepResult", "VectorSweepTarget", "sweep_vectors"]
