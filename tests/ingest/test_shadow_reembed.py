"""Offline, resumable embedding rebuilds over synthetic stored chunk inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import override

import pytest

from manicule.core.content import Chunk, Document
from manicule.core.embedding import EmbedFingerprint, Vector
from manicule.ingest.reembed import (
    ReembedError,
    ReembedPlan,
    ReembedRun,
    ReembedState,
    ReembedValidationError,
    ShadowInspection,
    plan_reembed,
    resume_reembed,
    start_reembed,
)
from tests.fakes import HashEmbedder, make_chunks, make_document


class CountingEmbedder(HashEmbedder):
    def __init__(self, dimension: int = 5) -> None:
        super().__init__(dimension=dimension)
        self.calls: list[tuple[str, ...]] = []

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(tuple(texts))
        return await super().embed(texts)


class Corpus:
    def __init__(self, documents: list[tuple[Document, list[Chunk]]]) -> None:
        self.documents = documents

    async def select_documents(
        self, *, limit: int | None = None, offset: int = 0
    ) -> list[Document]:
        selected = [document for document, _ in self.documents]
        return selected[offset:] if limit is None else selected[offset : offset + limit]

    async def document_chunks(self, document_id: str) -> list[Chunk]:
        return next(chunks for document, chunks in self.documents if document.id == document_id)


class Journal:
    """A durable fake: new orchestrators read the same committed run object."""

    def __init__(self) -> None:
        self.runs: dict[str, ReembedRun] = {}
        self.fail_checkpoint_once = False

    async def create(self, run_id: str, plan: ReembedPlan) -> ReembedRun:
        run = ReembedRun(id=run_id, plan=plan)
        self.runs[run_id] = run
        return run

    async def get(self, run_id: str) -> ReembedRun | None:
        return self.runs.get(run_id)

    async def save(self, run: ReembedRun, *, expected_revision: int) -> ReembedRun:
        current = self.runs[run.id]
        if current.revision != expected_revision:
            raise RuntimeError("stale journal writer")
        if self.fail_checkpoint_once and run.next_document == 1:
            self.fail_checkpoint_once = False
            raise OSError("synthetic checkpoint interruption")
        saved = replace(run, revision=run.revision + 1)
        self.runs[run.id] = saved
        return saved


class Shadow:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
        self.fingerprint = ""
        self.inventory_digest = ""
        self.upsert_attempts = 0
        self.batch_sizes: list[int] = []
        self.override: ShadowInspection | None = None

    async def prepare(self, run_id: str, fingerprint: EmbedFingerprint) -> None:
        del run_id
        self.fingerprint = fingerprint.canonical()

    async def upsert(
        self,
        run_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
    ) -> None:
        self.upsert_attempts += len(chunks)
        self.batch_sizes.append(len(chunks))
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.rows[f"{run_id}:{chunk.id}"] = (chunk, tuple(vector))

    async def inspect(self, run_id: str) -> ShadowInspection:
        if self.override is not None:
            return self.override
        selected = [value for key, value in self.rows.items() if key.startswith(f"{run_id}:")]
        dimensions = {len(vector) for _, vector in selected}
        return ShadowInspection(
            rows=len(selected),
            unique_chunks=len({chunk.id for chunk, _ in selected}),
            dimension=dimensions.pop() if len(dimensions) == 1 else 0,
            finite=all(math.isfinite(value) for _, vector in selected for value in vector),
            fingerprint=self.fingerprint,
            inventory_digest=self.inventory_digest,
            lineage_valid=True,
            retrieval_ready=True,
        )


class Publisher:
    def __init__(self, shadow: Shadow) -> None:
        self.shadow = shadow
        self.live = {"old:chunk": (0.0, 1.0)}
        self.calls = 0

    async def publish(self, run: ReembedRun) -> None:
        self.calls += 1
        self.live = {
            key: vector for key, (_, vector) in self.shadow.rows.items() if key.startswith(run.id)
        }


def corpus_with_chunks(count: int = 3) -> Corpus:
    document = make_document().model_copy(
        update={"uri": "https://wiki.example.test/pages/synthetic-1"}
    )
    return Corpus([(document, make_chunks(document, count=count))])


async def test_dry_run_prices_local_work_without_calling_the_embedder() -> None:
    corpus = corpus_with_chunks(3)
    embedder = CountingEmbedder(dimension=7)

    plan = await plan_reembed(corpus, embedder.fingerprint, chunks_per_second=2.0)

    assert (plan.documents, plan.chunks) == (1, 3)
    assert plan.estimated_seconds == 1.5
    expected_inputs = sum(len(chunk.embed_text.encode("utf-8")) for chunk in corpus.documents[0][1])
    assert plan.peak_memory_bytes == expected_inputs + 3 * 7 * 4
    assert plan.temporary_disk_bytes == 3 * (7 * 4 + 512)
    assert plan.input_bytes > 0
    assert plan.runnable
    assert embedder.calls == []


async def test_an_interruption_after_shadow_write_resumes_without_duplicate_vectors() -> None:
    corpus = corpus_with_chunks(2)
    embedder = HashEmbedder(dimension=5)
    journal = Journal()
    shadow = Shadow()
    publisher = Publisher(shadow)
    run = await start_reembed(
        "synthetic-reembed-1", corpus=corpus, target=embedder.fingerprint, journal=journal
    )
    shadow.inventory_digest = run.plan.inventory_digest
    journal.fail_checkpoint_once = True

    with pytest.raises(OSError, match="synthetic checkpoint interruption"):
        await resume_reembed(
            run.id,
            corpus=corpus,
            embedder=embedder,
            journal=journal,
            shadow=shadow,
            publisher=publisher,
        )

    assert publisher.live == {"old:chunk": (0.0, 1.0)}
    completed = await resume_reembed(
        run.id,
        corpus=corpus,
        embedder=embedder,
        journal=journal,
        shadow=shadow,
        publisher=publisher,
    )

    assert completed.state is ReembedState.PUBLISHED
    assert completed.documents_completed == 1
    assert completed.chunks_completed == 2
    assert len(shadow.rows) == 2
    assert shadow.upsert_attempts == 4
    assert len(publisher.live) == 2


async def test_large_documents_are_written_to_the_shadow_in_bounded_batches() -> None:
    corpus = corpus_with_chunks(130)
    embedder = CountingEmbedder(dimension=5)
    journal = Journal()
    shadow = Shadow()
    publisher = Publisher(shadow)
    run = await start_reembed(
        "synthetic-reembed-bounded",
        corpus=corpus,
        target=embedder.fingerprint,
        journal=journal,
    )
    shadow.inventory_digest = run.plan.inventory_digest

    completed = await resume_reembed(
        run.id,
        corpus=corpus,
        embedder=embedder,
        journal=journal,
        shadow=shadow,
        publisher=publisher,
    )

    assert completed.state is ReembedState.PUBLISHED
    assert shadow.batch_sizes == [64, 64, 2]
    assert [len(call) for call in embedder.calls] == [64, 64, 2]


async def test_changed_inventory_is_rejected_before_publication() -> None:
    corpus = corpus_with_chunks(1)
    embedder = HashEmbedder(dimension=4)
    journal = Journal()
    shadow = Shadow()
    publisher = Publisher(shadow)
    run = await start_reembed(
        "synthetic-reembed-2", corpus=corpus, target=embedder.fingerprint, journal=journal
    )
    shadow.inventory_digest = run.plan.inventory_digest
    document, chunks = corpus.documents[0]
    corpus.documents[0] = (
        document.model_copy(update={"content_hash": "changed-after-plan"}),
        chunks,
    )

    with pytest.raises(ReembedValidationError, match="inventory changed"):
        await resume_reembed(
            run.id,
            corpus=corpus,
            embedder=embedder,
            journal=journal,
            shadow=shadow,
            publisher=publisher,
        )

    assert publisher.live == {"old:chunk": (0.0, 1.0)}
    assert journal.runs[run.id].state is ReembedState.FAILED


async def test_resume_refuses_an_operational_embedder_change_omitted_from_identity() -> None:
    corpus = corpus_with_chunks(1)
    planned = HashEmbedder(dimension=4)
    journal = Journal()
    shadow = Shadow()
    run = await start_reembed(
        "synthetic-reembed-config",
        corpus=corpus,
        target=planned.fingerprint,
        journal=journal,
    )
    changed = HashEmbedder(dimension=4)
    changed.fingerprint = changed.fingerprint.model_copy(update={"max_sequence_length": 64})

    assert changed.fingerprint.canonical() == planned.fingerprint.canonical()
    with pytest.raises(ReembedError, match="context limit"):
        await resume_reembed(
            run.id,
            corpus=corpus,
            embedder=changed,
            journal=journal,
            shadow=shadow,
            publisher=Publisher(shadow),
        )

    assert shadow.rows == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"finite": False}, "non-finite"),
        ({"dimension": 99}, "expected dimension"),
        ({"lineage_valid": False}, "invalid chunk or embedding lineage"),
        ({"retrieval_ready": False}, "retrieval-readiness"),
    ],
)
async def test_validation_faults_never_replace_the_live_generation(
    change: dict[str, object], message: str
) -> None:
    corpus = corpus_with_chunks(1)
    embedder = HashEmbedder(dimension=4)
    journal = Journal()
    shadow = Shadow()
    publisher = Publisher(shadow)
    run = await start_reembed(
        "synthetic-reembed-3", corpus=corpus, target=embedder.fingerprint, journal=journal
    )
    shadow.inventory_digest = run.plan.inventory_digest
    valid = ShadowInspection(
        rows=1,
        unique_chunks=1,
        dimension=4,
        finite=True,
        fingerprint=run.plan.target_fingerprint,
        inventory_digest=run.plan.inventory_digest,
        lineage_valid=True,
        retrieval_ready=True,
    )
    shadow.override = replace(valid, **change)

    with pytest.raises(ReembedValidationError, match=message):
        await resume_reembed(
            run.id,
            corpus=corpus,
            embedder=embedder,
            journal=journal,
            shadow=shadow,
            publisher=publisher,
        )

    assert publisher.calls == 0
    assert publisher.live == {"old:chunk": (0.0, 1.0)}


async def test_missing_local_chunks_are_itemized_without_exposing_document_text() -> None:
    document = make_document("confidential synthetic body").model_copy(
        update={"uri": "https://wiki.example.test/pages/synthetic-missing"}
    )
    embedder = CountingEmbedder()

    plan = await plan_reembed(Corpus([(document, [])]), embedder.fingerprint)

    assert not plan.runnable
    assert len(plan.unrepairable) == 1
    assert document.id in plan.unrepairable[0]
    assert "confidential synthetic body" not in plan.unrepairable[0]
    assert embedder.calls == []
