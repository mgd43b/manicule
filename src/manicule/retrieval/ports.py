"""What retrieval asks of a store beyond :class:`~manicule.core.protocols.DocStore`.

Both are structural and both are **optional**, which is the point. A store that implements
neither is a perfectly good document store; it simply changes how retrieval behaves around it,
and in one direction only:

- Without :class:`SupportsLiveChunkCount`, the dense leg cannot measure how dilute the vector
  table is, so it starts at its over-fetch floor and lets the retry in ``docs/retrieval.md``
  §4.4 make up the difference. **More round trips, identical candidates.**
- Without :class:`~manicule.core.retrieval.SupportsGeneration`, the L1 cache has no
  invalidation signal, so the retriever refuses to enable it and says why. **A cold cache,
  never a stale one.**

That asymmetry is deliberate. A capability a store lacks may cost throughput; it must never
change what a query returns, because then two deployments of manicule would answer the same
question differently and only one of them would be right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from manicule.core.embedding import IndexFingerprints


@runtime_checkable
class SupportsLiveChunkCount(Protocol):
    """A store that can say how many chunks a search in its workspace could return.

    The numerator of the derived over-fetch factor. Distinct from a plain chunk count, and the
    distinction is the whole value: the vector table holds a row per chunk with no column for
    tenancy, liveness or status, so what matters is the share of those rows that survive the
    hydrating join. Counting every chunk in the database instead would report a fifty-tenant
    index as a clean one and under-fetch on exactly the deployment that needs it most.
    """

    async def live_chunk_count(self) -> int:
        """Chunks of live, indexed documents in this store's workspace."""
        ...


@runtime_checkable
class SupportsVectorCount(Protocol):
    """A vector store that can report its row count.

    The *denominator* of the over-fetch factor, and it is the vector table's row count rather
    than the chunk count on purpose: unswept tombstones are still rows and still consume
    top-``k`` slots, so a fraction computed against SQLite's chunk count would call an index
    clean while it was full of pending deletions.
    """

    async def count(self) -> int:
        """How many vectors are stored."""
        ...


@runtime_checkable
class SupportsDocumentCount(Protocol):
    """A store that can count its documents without listing them."""

    async def count_documents(self) -> int:
        """Live documents in this store's workspace."""
        ...


@runtime_checkable
class SupportsIndexState(Protocol):
    """A store that can say what its index was built with and how much is in it."""

    async def index_fingerprints(self) -> IndexFingerprints: ...

    async def count_chunks(self, document_id: str | None = None) -> int: ...


__all__ = [
    "SupportsDocumentCount",
    "SupportsIndexState",
    "SupportsLiveChunkCount",
    "SupportsVectorCount",
]
