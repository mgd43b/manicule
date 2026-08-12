"""What retrieval asks of a store beyond :class:`~manicule.core.protocols.DocStore`.

Both are structural and both are **optional**, which is the point. A store that implements
neither is a perfectly good document store; it simply changes how retrieval behaves around it,
and in one direction only:

- Without :class:`SupportsLiveChunkCount`, the dense leg cannot measure how dilute the vector
  table is, so it starts at its over-fetch floor and lets the retry in ``docs/retrieval.md``
  §4.4 make up the difference. **More round trips, identical candidates.** (Only the numerator
  of that fraction is optional: every vector store reports its row count, because the protocol
  requires it — and it is the vector table's row count rather than the chunk count on purpose,
  since unswept tombstones are still rows and still consume top-``k`` slots.)
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
    from collections.abc import Sequence

    from manicule.core.embedding import IndexFingerprints
    from manicule.core.glossary import GlossaryEntry
    from manicule.core.retrieval import Filter


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


@runtime_checkable
class GlossarySource(Protocol):
    """A store that can look glossary terms up **within a filter's scope**.

    The filter is a parameter rather than an afterthought, and that is the whole security
    argument for this port. An entry names the document it was read out of, and a document id
    already carries the workspace (:func:`~manicule.core.ids.document_id` takes it as the first
    component of its digest) — but *collection* scope is a membership relation the entry cannot
    carry without holding a second copy that can go stale. So the store resolves both, in the
    same statement it selects the entries with, and there is no path that returns an entry and
    then filters it.

    A store that does not implement this is not defective: expansion is simply unavailable
    against it, and the retriever says so rather than silently searching one form.
    """

    async def entries_for(
        self,
        keys: Sequence[str],
        filter: Filter,  # noqa: A002 - mirrors the vocabulary every other scoped read uses
    ) -> Sequence[GlossaryEntry]:
        """Every entry in scope whose acronym or alias is one of ``keys``.

        Args:
            keys: Normalised lookup keys, as :func:`~manicule.core.glossary.normalise_acronym`
                produces them. A store must not normalise again: two normalisations that
                disagree produce a lookup that silently misses, which reads exactly like a
                corpus with no glossary in it.
            filter: The query's whole restriction. A store that cannot honour a field of it
                must refuse rather than drop it — an entry admitted by an ignored
                ``collection_ids`` is one collection's glossary leaking into another's search.
        """
        ...


__all__ = [
    "GlossarySource",
    "SupportsDocumentCount",
    "SupportsIndexState",
    "SupportsLiveChunkCount",
]
