"""The scheduled sweep #33 left without a runner, and the race it must not have."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import override

from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, DocumentStatus
from manicule.ingest.sweeps import UNREFERENCED_BLOB_MIN_AGE_S, sweep_vectors
from tests.fakes import make_chunks, make_document
from tests.ingest import fakes


class _Busy:
    """A gate that always says no, naming what is holding it."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def sweep_permitted(self) -> str:
        return self._reason


class _Free:
    def sweep_permitted(self) -> str:
        return ""


async def _stocked() -> tuple[fakes.MemoryIngestStore, fakes.MemoryVectors, list[Chunk]]:
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    document = make_document()
    store.documents[document.id] = document
    chunks = make_chunks(document, count=3)
    await store.replace_chunks(document.id, chunks)
    await vectors.upsert(chunks, [[0.1] * 5 for _ in chunks])
    return store, vectors, chunks


async def test_a_tombstoned_vector_is_removed() -> None:
    """The whole of the sweep, in the healthy case."""
    store, vectors, chunks = await _stocked()
    await store.replace_chunks(chunks[0].document_id, [])

    result = await sweep_vectors(store, vectors)

    assert result.vectors_removed == 3
    assert vectors.rows == {}
    assert store.tombstones == []


async def test_a_vector_written_during_the_sweep_survives() -> None:
    """Why tombstones exist at all, stated as the race they prevent.

    An anti-join between the vector table and ``chunks`` cannot distinguish a vector written
    after the scan began from an orphan — so it deletes a live vector, and the chunk is then
    served by the lexical leg, missing from the dense one, and wrong with nothing raised. The
    sweep reads a list of things that *were* deleted, so it cannot make that mistake.
    """
    store, vectors, chunks = await _stocked()
    await store.replace_chunks(chunks[0].document_id, [])

    concurrent = Chunk(
        id="written-mid-sweep",
        document_id="another-document",
        text="new",
        embed_text="new",
        anchor=Unlocated(reason="synthetic"),
        position=0,
        token_count=1,
    )
    await vectors.upsert([concurrent], [[0.2] * 5])

    await sweep_vectors(store, vectors)

    assert "written-mid-sweep" in vectors.rows, (
        "a vector nothing ever tombstoned must not be swept, whenever it arrived"
    )


async def test_a_sweep_yields_to_a_backup() -> None:
    """The backup lock blocks exactly this and the blob collector, because they remove data."""
    store, vectors, chunks = await _stocked()
    await store.replace_chunks(chunks[0].document_id, [])

    result = await sweep_vectors(store, vectors, gate=_Busy("a backup is running"))

    assert not result.ran
    assert result.blocked_by == "a backup is running"
    assert len(vectors.rows) == 3, "nothing was removed while the gate was closed"


async def test_a_sweep_yields_to_an_active_sync() -> None:
    """An ingest run and a purge must never compete for the same vector table."""
    store, vectors, _ = await _stocked()

    result = await sweep_vectors(store, vectors, gate=_Busy("a sync is running"))

    assert not result.ran


async def test_a_soft_deleted_document_survives_its_grace_period() -> None:
    """Inside the window a restore is free: no re-embed, no re-parse, no re-fetch."""
    store, vectors, chunks = await _stocked()
    document_id = chunks[0].document_id
    await store.soft_delete_document(document_id)
    store.deleted_at[document_id] = datetime.now(UTC)

    result = await sweep_vectors(store, vectors, soft_delete_grace_s=3600, gate=_Free())

    assert result.documents_purged == 0
    assert len(vectors.rows) == 3


async def test_a_soft_deleted_document_is_purged_after_its_grace_period() -> None:
    """Unbounded free restore would mean unbounded dilution of every vector search.

    A soft-deleted chunk is still in the vector table competing for top-``k`` slots before the
    join that hides it can run, so the trade is real rather than free.
    """
    store, vectors, chunks = await _stocked()
    document_id = chunks[0].document_id
    await store.soft_delete_document(document_id)
    store.deleted_at[document_id] = datetime.now(UTC) - timedelta(days=60)

    result = await sweep_vectors(store, vectors, soft_delete_grace_s=30 * 24 * 3600)

    assert result.documents_purged == 1
    assert vectors.deleted_documents == [document_id]
    assert store.chunks[document_id] == []


async def test_a_second_pass_over_the_same_tombstones_is_a_no_op() -> None:
    """A crash between removing the vectors and clearing the tombstones costs one wasted pass.

    Which is the right way round: prefer the failure that costs a pass over the one that costs
    correctness.
    """
    store, vectors, chunks = await _stocked()
    await store.replace_chunks(chunks[0].document_id, [])
    await sweep_vectors(store, vectors)

    again = await sweep_vectors(store, vectors)

    assert again.vectors_removed == 0


async def test_the_sweep_target_is_a_protocol_a_backend_can_fail_to_satisfy() -> None:
    """Checked at runtime, because ``VectorStore`` does not require chunk-level deletion.

    A plugin backend can legitimately implement the vector protocol without ``delete_chunks``.
    The wiring has to be able to *notice* that rather than casting past it, because a sweep
    that reported "0 removed" for a store it could not address would read exactly like a clean
    index — and the tombstone table would grow behind it forever.
    """
    from manicule.ingest.sweeps import VectorSweepTarget  # noqa: PLC0415

    class DocumentOnly:
        async def delete_document(self, document_id: str) -> None:
            del document_id

    class Both(DocumentOnly):
        async def delete_chunks(self, chunk_ids: list[str]) -> None:
            del chunk_ids

    assert not isinstance(DocumentOnly(), VectorSweepTarget)
    assert isinstance(Both(), VectorSweepTarget)


# --- the purge is the store's, and blob collection ends the pass -------------------------------


class _Collector:
    """A blob collector that records how it was asked and reports a fixed result."""

    def __init__(self, collected: tuple[str, ...] = ()) -> None:
        self.calls: list[float] = []
        self._collected = collected

    async def collect_garbage(self, *, min_age_s: float = 0.0) -> Sequence[str]:
        self.calls.append(min_age_s)
        return self._collected


async def _expired() -> tuple[fakes.MemoryIngestStore, fakes.MemoryVectors, str]:
    store, vectors, chunks = await _stocked()
    document_id = chunks[0].document_id
    await store.soft_delete_document(document_id)
    store.deleted_at[document_id] = datetime.now(UTC) - timedelta(days=60)
    return store, vectors, document_id


async def test_a_purged_document_is_marked_and_its_content_removed_through_the_store() -> None:
    store, vectors, document_id = await _expired()

    result = await sweep_vectors(store, vectors, soft_delete_grace_s=30 * 24 * 3600)

    assert result.documents_purged == 1
    assert store.documents[document_id].status is DocumentStatus.DELETED
    assert vectors.deleted_documents == [document_id]
    assert store.chunks[document_id] == []


async def test_a_purged_document_is_not_purged_again() -> None:
    """The termination argument, through the whole pass rather than the selection alone."""
    store, vectors, _ = await _expired()
    await sweep_vectors(store, vectors, soft_delete_grace_s=3600)

    again = await sweep_vectors(store, vectors, soft_delete_grace_s=3600)

    assert again.documents_purged == 0


async def test_a_document_restored_after_selection_is_neither_purged_nor_counted() -> None:
    """The race the purge re-check exists for, at the level of the sweep.

    The store hands back the id and a restore lands before the purge: the sweep must leave the
    document's vectors alone — a restored document served without its dense leg is wrong with no
    error reported.
    """

    class RestoredMidPass(fakes.MemoryIngestStore):
        @override
        async def soft_deleted_before(
            self, cutoff: datetime, *, limit: int = 1000
        ) -> Sequence[str]:
            selected = await super().soft_deleted_before(cutoff, limit=limit)
            self.deleted_at.clear()  # a restore, between selection and purge
            return selected

    store = RestoredMidPass()
    vectors = fakes.MemoryVectors()
    document = make_document()
    store.documents[document.id] = document
    chunks = make_chunks(document, count=3)
    await store.replace_chunks(document.id, chunks)
    await vectors.upsert(chunks, [[0.1] * 5 for _ in chunks])
    await store.soft_delete_document(document.id)
    store.deleted_at[document.id] = datetime.now(UTC) - timedelta(days=60)

    result = await sweep_vectors(store, vectors, soft_delete_grace_s=3600)

    assert result.documents_purged == 0
    assert vectors.deleted_documents == []
    assert len(vectors.rows) == 3
    assert len(store.chunks[document.id]) == 3
    assert store.documents[document.id].status is not DocumentStatus.DELETED


async def test_blob_collection_ends_the_pass_and_is_counted_never_named() -> None:
    store, vectors, _ = await _expired()
    collector = _Collector(("hash-one", "hash-two"))

    result = await sweep_vectors(
        store, vectors, soft_delete_grace_s=3600, blobs=collector, blob_min_age_s=7.0
    )

    assert collector.calls == [7.0], "collected once, with the age floor the caller chose"
    assert result.blobs_collected == 2
    assert "hash-one" not in repr(result)


async def test_blob_collection_runs_on_a_pass_that_purged_nothing() -> None:
    """Roots lapse through paths the sweep does not see, so it cannot wait for a purge."""
    store, vectors, _ = await _stocked()
    collector = _Collector(("lapsed",))

    result = await sweep_vectors(store, vectors, blobs=collector)

    assert result.documents_purged == 0
    assert collector.calls == [UNREFERENCED_BLOB_MIN_AGE_S]
    assert result.blobs_collected == 1


async def test_a_blocked_sweep_collects_nothing() -> None:
    store, vectors, _ = await _expired()
    collector = _Collector(("never",))

    result = await sweep_vectors(
        store, vectors, soft_delete_grace_s=3600, gate=_Busy("a backup is running"), blobs=collector
    )

    assert not result.ran
    assert collector.calls == []
    assert result.blobs_collected == 0


async def test_a_gate_that_closes_after_the_purges_stops_collection() -> None:
    """Asked again before the collection, because it is the step a backup most needs kept out."""

    class ClosesAfter:
        def __init__(self, permitted: int) -> None:
            self._left = permitted

        def sweep_permitted(self) -> str:
            self._left -= 1
            return "" if self._left >= 0 else "a backup is running"

    store, vectors, _ = await _expired()
    collector = _Collector(("never",))

    # top of the pass, one document, and then the check before collection
    result = await sweep_vectors(
        store, vectors, soft_delete_grace_s=3600, gate=ClosesAfter(2), blobs=collector
    )

    assert result.documents_purged == 1
    assert not result.ran
    assert collector.calls == []
