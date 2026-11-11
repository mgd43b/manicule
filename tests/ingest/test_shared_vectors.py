"""Reusing another workspace's stored vectors without sharing a row with it.

``docs/storage.md`` §6.2.6. Every count here comes from
:class:`~tests.ingest.fakes.CountingEmbedder`, which records one entry per batch it is handed:
the claim under test is about forward passes avoided, and only the embedder can see one.

Two workspaces are two pipelines over two stores, exactly as two tenants are in production:
nothing is shared between them except the lender, which can read the first workspace's vectors
and nothing else of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from manicule.core.embedding import StoredVector, VectorState, vector_checksum
from manicule.core.errors import ContextOverflowError
from manicule.core.ids import document_id
from manicule.ingest.embedding import EmbeddingWork, embed_or_reuse
from manicule.ingest.shared_vectors import Donor, SharedVectors
from tests.fakes import HashEmbedder
from tests.ingest import fakes
from tests.ingest.test_pipeline import build

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.content import Chunk
    from manicule.core.embedding import Vector
    from manicule.ingest.pipeline import IngestPipeline

SOURCE = "handbook"
"""One connector instance name, configured in both workspaces."""

PAGES = {
    f"page-{number}": "\n".join(f"Line {line} of invented page {number}." for line in range(4))
    for number in range(6)
}
"""Six invented pages of four lines each: twenty-four chunks under the line parser."""

CHUNKS = sum(len(text.splitlines()) for text in PAGES.values())


class StaticDonors:
    """The donor set, fixed for the length of a test."""

    def __init__(self, donors: Sequence[Donor]) -> None:
        self._donors = list(donors)

    async def donors(self, *, source: str, source_id: str) -> Sequence[Donor]:
        del source, source_id
        return self._donors


class RecordingLedger:
    """The operator's ledger, as a list of what it was told."""

    def __init__(self) -> None:
        self.recorded: list[int] = []

    async def record(self, vectors: int) -> None:
        self.recorded.append(vectors)


class RecordingLender:
    """A lender that answers from a fixed table and remembers every question."""

    def __init__(self, vectors: Mapping[str, Vector]) -> None:
        self.vectors = dict(vectors)
        self.asked: list[list[str]] = []

    async def lend(self, chunks: Sequence[Chunk]) -> Mapping[str, Vector]:
        self.asked.append([chunk.id for chunk in chunks])
        return {chunk.id: self.vectors[chunk.id] for chunk in chunks if chunk.id in self.vectors}


class OverCountingEmbedder(fakes.CountingEmbedder):
    """An embedder whose own tokenizer finds every text longer than it will read.

    The chunker's recorded counts pass; only the measurement the model call makes refuses. That
    is the check a borrowed vector must still be held to, and the one it would skip if borrowing
    came first.
    """

    def count_tokens(self, text: str) -> int:
        del text
        return self.fingerprint.max_sequence_length + 1


async def _workspace(
    name: str,
    *,
    donors: Sequence[Donor] | None = None,
    ledger: RecordingLedger | None = None,
    embedder: fakes.CountingEmbedder | None = None,
) -> tuple[IngestPipeline, fakes.MemoryIngestStore, fakes.MemoryVectors, fakes.CountingEmbedder]:
    """One workspace's pipeline, prepared the way the runtime prepares its store."""
    embedder = embedder or fakes.CountingEmbedder()
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(embedder.fingerprint)
    shared = (
        None
        if donors is None
        else SharedVectors(StaticDonors(donors), fingerprint=embedder.fingerprint, ledger=ledger)
    )
    pipeline, store, _ = build(
        vectors=vectors, embedder=embedder, workspace=name, shared_vectors=shared
    )
    return pipeline, store, vectors, embedder


async def _work_per_document(pipeline: IngestPipeline) -> list[EmbeddingWork]:
    """Each document's embedding report, in discovery order, as a sweep payload is built from."""
    connector = fakes.DictConnector(PAGES, name=SOURCE)
    reports: list[EmbeddingWork] = []
    async for discovered in connector.discover(None):
        reports.extend(
            outcome.embedding for outcome in await pipeline.ingest(connector, discovered)
        )
    return reports


async def test_a_second_workspace_syncing_a_held_source_makes_no_embedding_calls() -> None:
    first, _, held, first_model = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))
    assert sum(first_model.batches) == CHUNKS, "the first workspace embeds the corpus once"

    ledger = RecordingLedger()
    second, store, vectors, model = await _workspace(
        "beta", donors=[Donor(workspace="alpha", vectors=held)], ledger=ledger
    )
    report = await second.run(fakes.DictConnector(PAGES, name=SOURCE))

    assert report.by_status == {"indexed": len(PAGES)}
    assert model.batches == [], "every vector was already held by the other workspace"
    assert sum(ledger.recorded) == CHUNKS, "the operator's ledger counts every copied vector"

    # The copy is the stored vector exactly, filed under this workspace's own chunk and identity.
    by_input = {row.embed_text: row for row in held.rows.values()}
    assert len(vectors.rows) == CHUNKS
    for chunk_id, row in vectors.rows.items():
        donor = by_input[row.embed_text]
        assert row.vector == donor.vector
        assert row.checksum == vector_checksum(row.vector) == donor.checksum
        assert row.document_id != donor.document_id
        assert row.identity != donor.identity, "the identity stays scoped to the workspace"
        assert chunk_id not in held.rows
    owned = {document_id("beta", SOURCE, source_id) for source_id in PAGES}
    assert {row.document_id for row in vectors.rows.values()} == owned
    assert {document.id for document in store.documents.values()} == owned

    # And it reads back as this workspace's own vector, under this workspace's own rule.
    chunks = [chunk for chunks in store.chunks.values() for chunk in chunks]
    verdicts = await vectors.stored_vectors(chunks)
    assert {verdict.state for verdict in verdicts.values()} == {VectorState.READABLE}


async def test_what_a_borrowing_workspace_reports_is_what_it_would_report_alone() -> None:
    """The discriminating test for disclosure: the same sync, with and without a donor.

    Every field of every document's report is compared, because each is a way a caller of
    the second workspace could learn that another tenant already held its text.
    """
    first, _, held, _ = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))

    borrowing, _, _, borrowing_model = await _workspace(
        "beta", donors=[Donor(workspace="alpha", vectors=held)]
    )
    alone, _, _, alone_model = await _workspace("beta", donors=[])

    borrowed = await _work_per_document(borrowing)
    unaided = await _work_per_document(alone)

    assert borrowing_model.batches == []
    assert sum(alone_model.batches) == CHUNKS
    assert borrowed == unaided
    assert all(work.reused == 0 and work.embedded > 0 for work in borrowed)
    assert all(work.forward_calls == 1 for work in borrowed)


async def test_a_donor_vector_that_fails_its_checksum_is_not_lent() -> None:
    first, _, held, _ = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))
    damaged_id, damaged = next(iter(held.rows.items()))
    # One finite component moved, every piece of metadata intact: only the checksum can see it.
    held.rows[damaged_id] = damaged.__class__(
        document_id=damaged.document_id,
        vector=(damaged.vector[0] + 0.25, *damaged.vector[1:]),
        embed_text=damaged.embed_text,
        identity=damaged.identity,
        checksum=damaged.checksum,
    )

    second, _, vectors, model = await _workspace(
        "beta", donors=[Donor(workspace="alpha", vectors=held)]
    )
    await second.run(fakes.DictConnector(PAGES, name=SOURCE))

    assert sum(model.batches) == 1, "the damaged vector's chunk was embedded, nothing else"
    (copied,) = [row for row in vectors.rows.values() if row.embed_text == damaged.embed_text]
    (fresh,) = await HashEmbedder(dimension=5).embed([damaged.embed_text])
    assert copied.vector == tuple(fresh)
    assert copied.vector != held.rows[damaged_id].vector


async def test_a_donor_built_by_another_model_lends_nothing() -> None:
    """The fingerprint clause is checked by the lender, not trusted to whoever chose donors.

    A donor classifies its own rows against its own recorded model, so to itself every one of
    them is readable; only the borrower knows they are in the wrong space.
    """
    other = fakes.CountingEmbedder(dimension=7)
    first, _, held, _ = await _workspace("alpha", embedder=other)
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))

    second, _, _, model = await _workspace("beta", donors=[Donor(workspace="alpha", vectors=held)])
    await second.run(fakes.DictConnector(PAGES, name=SOURCE))

    assert sum(model.batches) == CHUNKS


async def test_a_differently_named_source_lends_nothing() -> None:
    """The match is the same document in the other workspace's terms, which needs its name."""
    first, _, held, _ = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name="runbooks"))

    second, _, _, model = await _workspace("beta", donors=[Donor(workspace="alpha", vectors=held)])
    await second.run(fakes.DictConnector(PAGES, name=SOURCE))

    assert sum(model.batches) == CHUNKS


async def test_a_resync_of_unchanged_vectors_never_asks_another_workspace() -> None:
    first, _, held, _ = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))
    pipeline, store, vectors, model = await _workspace("beta")
    await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))
    chunks = next(iter(store.chunks.values()))
    lender = RecordingLender({})

    _, work = await embed_or_reuse(model, chunks, vectors=vectors, lender=lender)

    assert lender.asked == [], "this workspace's own store answered for every chunk"
    assert (work.reused, work.embedded) == (len(chunks), 0)
    del held


async def test_a_refresh_never_borrows() -> None:
    """``--re-embed`` exists to replace vectors the reuse condition cannot see are wrong."""
    pipeline, store, vectors, model = await _workspace("beta")
    await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))
    chunks = next(iter(store.chunks.values()))
    lender = RecordingLender({chunk.id: (1.0, 0.0, 0.0, 0.0, 0.0) for chunk in chunks})
    model.batches.clear()

    _, work = await embed_or_reuse(model, chunks, vectors=vectors, lender=lender, reuse=False)

    assert lender.asked == []
    assert sum(model.batches) == len(chunks) == work.refreshed


async def test_the_oversize_refusal_runs_before_anything_is_borrowed() -> None:
    """A chunk the model would refuse is refused even when another workspace holds a vector."""
    embedder = OverCountingEmbedder()
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(embedder.fingerprint)
    pipeline, store, _ = build(vectors=fakes.MemoryVectors())
    await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))
    chunks = next(iter(store.chunks.values()))
    lender = RecordingLender({chunk.id: (1.0, 0.0, 0.0, 0.0, 0.0) for chunk in chunks})

    with pytest.raises(ContextOverflowError):
        await embed_or_reuse(embedder, chunks, vectors=vectors, lender=lender)

    assert lender.asked == []
    assert embedder.batches == []


async def test_borrowed_vectors_are_placed_where_their_chunks_are() -> None:
    """Half lent, half embedded: each vector lands at its own chunk's position."""
    pipeline, store, _, _ = await _workspace("beta")
    await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))
    chunks = [chunk for chunks in store.chunks.values() for chunk in chunks]
    lent = {
        chunk.id: tuple(1.0 if axis == index % 5 else 0.0 for axis in range(5))
        for index, chunk in enumerate(chunks)
        if index % 2 == 0
    }
    model = fakes.CountingEmbedder()
    empty = fakes.MemoryVectors()
    await empty.ensure_ready(model.fingerprint)

    ordered, work = await embed_or_reuse(model, chunks, vectors=empty, lender=RecordingLender(lent))

    expected = await HashEmbedder(dimension=5).embed([chunk.embed_text for chunk in chunks])
    for index, chunk in enumerate(chunks):
        assert tuple(ordered[index]) == (
            lent[chunk.id] if chunk.id in lent else tuple(expected[index])
        )
    assert sum(model.batches) == len(chunks) - len(lent)
    assert work.embedded == len(chunks)


class MisreportingStore(fakes.MemoryVectors):
    """A donor that hands back a vector beside every verdict, readable or not.

    No shipped store does: they return a vector only with a readable verdict. A plugin store is
    not bound by that except through its contract, so the lender reads the verdict, not the
    presence of numbers.
    """

    def __init__(self, state: VectorState) -> None:
        super().__init__()
        self.state = state

    @override
    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        return {
            chunk.id: StoredVector(state=self.state, vector=(1.0, 0.0, 0.0, 0.0, 0.0))
            for chunk in chunks
        }


@pytest.mark.parametrize("state", [VectorState.STALE, VectorState.CORRUPT])
async def test_only_a_readable_verdict_is_lent(state: VectorState) -> None:
    model = fakes.CountingEmbedder()
    donor = MisreportingStore(state)
    await donor.ensure_ready(model.fingerprint)
    shared = SharedVectors(
        StaticDonors([Donor(workspace="alpha", vectors=donor)]), fingerprint=model.fingerprint
    )
    pipeline, store, _, _ = await _workspace("beta")
    await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))
    chunks = next(iter(store.chunks.values()))

    assert await shared.lend(chunks, source=SOURCE, source_id="page-0") == {}


class BrokenDonors:
    """A donor set whose resolution fails, as another workspace's damaged state would."""

    async def donors(self, *, source: str, source_id: str) -> Sequence[Donor]:
        del source, source_id
        msg = "another workspace's index state could not be read"
        raise RuntimeError(msg)


async def test_a_failure_to_find_donors_never_fails_the_document() -> None:
    """Another workspace's state is never the reason this one's document is demoted."""
    model = fakes.CountingEmbedder()
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(model.fingerprint)
    shared = SharedVectors(BrokenDonors(), fingerprint=model.fingerprint)
    pipeline, store, _ = build(
        vectors=vectors, embedder=model, workspace="beta", shared_vectors=shared
    )

    report = await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))

    assert report.by_status == {"indexed": len(PAGES)}
    assert sum(model.batches) == CHUNKS
    assert all(document.status_detail is None for document in store.documents.values())


async def test_a_document_that_does_not_commit_is_not_counted() -> None:
    """The ledger counts passes saved, and a document that failed to store saved none."""
    first, _, held, _ = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))
    model = fakes.CountingEmbedder()
    refusing = fakes.RefusingVectors()
    await refusing.ensure_ready(model.fingerprint)
    ledger = RecordingLedger()
    shared = SharedVectors(
        StaticDonors([Donor(workspace="alpha", vectors=held)]),
        fingerprint=model.fingerprint,
        ledger=ledger,
    )
    pipeline, _, _ = build(
        vectors=refusing, embedder=model, workspace="beta", shared_vectors=shared
    )

    report = await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))

    assert "indexed" not in report.by_status
    assert model.batches == [], "the vectors were borrowed; only storing them failed"
    assert ledger.recorded == []


class SwitchableVectors(fakes.MemoryVectors):
    """A store whose writes can be made to fail after it already holds a document."""

    def __init__(self) -> None:
        super().__init__()
        self.refuse = False

    @override
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = "legacy",
    ) -> None:
        if self.refuse:
            msg = "the vector store is unavailable"
            raise RuntimeError(msg)
        await super().upsert(chunks, vectors, publication_id=publication_id)


async def test_a_failed_update_of_an_indexed_document_is_not_counted() -> None:
    """A failed update keeps the old revision served: still ``indexed``, and nothing borrowed used.

    The demotion of an already-indexed document leaves its status ``indexed`` and names the
    failure in ``failed_stage``, so status alone cannot tell the ledger whether the copy landed.
    """
    first, _, held, _ = await _workspace("alpha")
    await first.run(fakes.DictConnector(PAGES, name=SOURCE))
    model = fakes.CountingEmbedder()
    vectors = SwitchableVectors()
    await vectors.ensure_ready(model.fingerprint)
    ledger = RecordingLedger()
    shared = SharedVectors(
        StaticDonors([Donor(workspace="alpha", vectors=held)]),
        fingerprint=model.fingerprint,
        ledger=ledger,
    )
    second, store, _ = build(
        vectors=vectors, embedder=model, workspace="beta", shared_vectors=shared
    )
    await second.run(fakes.DictConnector(PAGES, name=SOURCE))
    assert sum(ledger.recorded) == CHUNKS

    edited = {**PAGES, "page-0": PAGES["page-0"] + "\nA line both workspaces now hold."}
    await first.run(fakes.DictConnector(edited, name=SOURCE))
    vectors.refuse = True
    connector = fakes.DictConnector(edited, name=SOURCE)
    outcomes = [
        outcome
        async for discovered in connector.discover(None)
        if discovered.ref.source_id == "page-0"
        for outcome in await second.ingest(connector, discovered)
    ]

    (updated,) = outcomes
    assert updated.status.value == "indexed", "the previous revision is still served"
    assert updated.failed_stage is not None, "and the update failed"
    assert model.batches == [], "the edited page's vectors were all borrowed"
    assert sum(ledger.recorded) == CHUNKS, "the failed update added nothing to the ledger"
    del store


async def test_forward_calls_match_across_several_batches() -> None:
    """The reported count is the unaided count even when borrowing removes whole batches."""
    pipeline, store, _, _ = await _workspace("beta")
    await pipeline.run(fakes.DictConnector(PAGES, name=SOURCE))
    chunks = [chunk for chunks in store.chunks.values() for chunk in chunks]
    lent = {chunk.id: (1.0, 0.0, 0.0, 0.0, 0.0) for chunk in chunks[: len(chunks) // 2 + 1]}

    async def cost(lender: RecordingLender | None) -> tuple[EmbeddingWork, list[int]]:
        model = fakes.CountingEmbedder()
        empty = fakes.MemoryVectors()
        await empty.ensure_ready(model.fingerprint)
        _, work = await embed_or_reuse(model, chunks, vectors=empty, lender=lender, maximum=3)
        return work, model.batches

    borrowing, borrowed_batches = await cost(RecordingLender(lent))
    alone, alone_batches = await cost(None)

    assert len(borrowed_batches) < len(alone_batches) == alone.forward_calls
    assert borrowing == alone
