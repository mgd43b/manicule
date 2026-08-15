"""Adversarial contracts for offline shadow-generation re-embedding."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import override

import pytest

from manicule.core.content import Chunk, Document
from manicule.core.embedding import EmbedFingerprint, Vector
from manicule.ingest.reembed import (
    ChunkKey,
    CorpusSnapshot,
    LivePublication,
    PublicationReceipt,
    PublishOutcome,
    ReembedError,
    ReembedLease,
    ReembedPlan,
    ReembedRun,
    ReembedState,
    ReembedValidationError,
    ShadowGeneration,
    ShadowInspection,
    SnapshotChunk,
    SnapshotDocument,
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


class Authority:
    """Transactional fake for journal, shadow storage, lease fencing, and publication.

    All four share one authority just as a production relational adapter must. It is concrete
    enough to prove the atomic CAS and crash behavior instead of assuming them in a mock.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.corpus_revision = "revision-1"
        self.live = LivePublication("live-old", "old-fingerprint", "old-inventory")
        self.runs: dict[str, ReembedRun] = {}
        self.leases: dict[str, ReembedLease] = {}
        self.highest_fence: dict[str, int] = {}
        self.generations: dict[str, ShadowGeneration] = {}
        self.rows: dict[str, dict[str, tuple[Chunk, tuple[float, ...]]]] = {}
        self.receipts: dict[str, PublicationReceipt] = {}
        self.prepare_calls = 0
        self.upsert_attempts = 0
        self.max_upsert_batch = 0
        self.live_during_upsert: list[str] = []
        self.inspection_override: ShadowInspection | None = None
        self.fail_chunk_checkpoint_once = False
        self.fail_terminal_checkpoint_once = False

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def install_competing_winner(self, generation: str = "live-newer") -> None:
        self.live = LivePublication(generation, "newer-fingerprint", "newer-inventory")
        self.corpus_revision = f"{self.corpus_revision}-moved"

    def _assert_lease(self, run_id: str, lease: ReembedLease) -> None:
        current = self.leases.get(run_id)
        if current != lease or lease.expires_at <= self.now:
            raise ReembedError("stale or expired re-embedding lease")

    async def create(
        self,
        run_id: str,
        plan: ReembedPlan,
        *,
        owner_token: str,
        ttl_seconds: float,
    ) -> tuple[ReembedRun, ReembedLease]:
        run = ReembedRun(id=run_id, plan=plan)
        existing = self.runs.setdefault(run_id, run)
        if existing.plan != plan:
            raise ReembedError("run id already belongs to another immutable plan")
        lease = await self.acquire(run_id, owner_token, ttl_seconds=ttl_seconds)
        return existing, lease

    async def get(self, run_id: str) -> ReembedRun | None:
        return self.runs.get(run_id)

    async def acquire(self, run_id: str, owner_token: str, *, ttl_seconds: float) -> ReembedLease:
        current = self.leases.get(run_id)
        if current is not None and current.expires_at > self.now:
            if current.owner_token != owner_token:
                raise ReembedError("another owner holds the re-embedding lease")
            renewed = replace(current, expires_at=self.now + ttl_seconds)
            self.leases[run_id] = renewed
            return renewed
        generation = self.highest_fence.get(run_id, 0) + 1
        self.highest_fence[run_id] = generation
        lease = ReembedLease(owner_token, generation, self.now + ttl_seconds)
        self.leases[run_id] = lease
        return lease

    async def renew(self, run_id: str, lease: ReembedLease, *, ttl_seconds: float) -> ReembedLease:
        self._assert_lease(run_id, lease)
        renewed = replace(lease, expires_at=self.now + ttl_seconds)
        self.leases[run_id] = renewed
        return renewed

    async def save(
        self, run: ReembedRun, *, expected_revision: int, lease: ReembedLease
    ) -> ReembedRun:
        self._assert_lease(run.id, lease)
        current = self.runs[run.id]
        if current.revision != expected_revision:
            raise ReembedError("stale journal revision")
        if self.fail_chunk_checkpoint_once and run.chunk_after is not None:
            self.fail_chunk_checkpoint_once = False
            raise OSError("synthetic chunk checkpoint crash")
        if self.fail_terminal_checkpoint_once and run.state is ReembedState.PUBLISHED:
            self.fail_terminal_checkpoint_once = False
            raise OSError("synthetic publish checkpoint crash")
        saved = replace(run, revision=run.revision + 1)
        self.runs[run.id] = saved
        return saved

    async def open_or_create(
        self,
        run_id: str,
        *,
        fingerprint: EmbedFingerprint,
        inventory_digest: str,
        lease: ReembedLease,
    ) -> ShadowGeneration:
        self._assert_lease(run_id, lease)
        self.prepare_calls += 1
        offered = ShadowGeneration(
            id=f"shadow:{run_id}",
            run_id=run_id,
            fingerprint=fingerprint.canonical(),
            inventory_digest=inventory_digest,
        )
        existing = self.generations.setdefault(run_id, offered)
        if existing != offered:
            raise ReembedError("shadow identity is immutable")
        self.rows.setdefault(existing.id, {})
        return existing

    async def upsert(
        self,
        generation: ShadowGeneration,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        lease: ReembedLease,
    ) -> None:
        self._assert_lease(generation.run_id, lease)
        self.live_during_upsert.append(self.live.generation_id)
        self.upsert_attempts += len(chunks)
        self.max_upsert_batch = max(self.max_upsert_batch, len(chunks))
        target = self.rows[generation.id]
        for chunk, vector in zip(chunks, vectors, strict=True):
            target[chunk.id] = (chunk, tuple(vector))

    async def inspect(
        self, generation: ShadowGeneration, *, lease: ReembedLease
    ) -> ShadowInspection:
        self._assert_lease(generation.run_id, lease)
        if self.inspection_override is not None:
            return self.inspection_override
        rows = self.rows[generation.id]
        dimensions = {len(vector) for _, vector in rows.values()}
        return ShadowInspection(
            rows=len(rows),
            unique_chunks=len({chunk.id for chunk, _ in rows.values()}),
            dimension=dimensions.pop() if len(dimensions) == 1 else 0,
            finite=all(math.isfinite(value) for _, vector in rows.values() for value in vector),
            fingerprint=generation.fingerprint,
            inventory_digest=generation.inventory_digest,
            lineage_valid=True,
            retrieval_ready=True,
        )

    async def publish(
        self,
        run: ReembedRun,
        generation: ShadowGeneration,
        *,
        expected: LivePublication,
        expected_corpus_revision: str,
        lease: ReembedLease,
    ) -> PublicationReceipt:
        self._assert_lease(run.id, lease)
        existing = self.receipts.get(run.id)
        if existing is not None:
            return existing
        if self.live != expected or self.corpus_revision != expected_corpus_revision:
            outcome = PublishOutcome.SUPERSEDED
            published = None
        else:
            outcome = PublishOutcome.PUBLISHED
            published = generation.id
            self.live = LivePublication(
                generation.id, generation.fingerprint, generation.inventory_digest
            )
            self.corpus_revision = f"published:{run.id}"
        receipt = PublicationReceipt(
            id=f"receipt:{run.id}",
            run_id=run.id,
            outcome=outcome,
            expected=expected,
            observed_winner=self.live,
            published_generation_id=published,
        )
        self.receipts[run.id] = receipt
        return receipt


class Corpus:
    def __init__(
        self, authority: Authority, documents: Sequence[tuple[Document, Sequence[Chunk]]]
    ) -> None:
        self.authority = authority
        self.current = {document.id: (document, list(chunks)) for document, chunks in documents}
        self.snapshots: dict[str, dict[str, tuple[Document, list[Chunk]]]] = {}
        self.current_lineage = {
            document.id: ("chunk-fingerprint", "embed-fingerprint", "glossary-fingerprint")
            for document, _ in documents
        }
        self.snapshot_lineage: dict[str, dict[str, tuple[str, str, str]]] = {}
        self.snapshot_number = 0
        self.max_document_page = 0
        self.max_chunk_page = 0

    async def begin_snapshot(self) -> CorpusSnapshot:
        self.snapshot_number += 1
        snapshot_id = f"snapshot-{self.snapshot_number}"
        self.snapshots[snapshot_id] = copy.deepcopy(self.current)
        self.snapshot_lineage[snapshot_id] = copy.deepcopy(self.current_lineage)
        return CorpusSnapshot(
            snapshot_id,
            self.authority.corpus_revision,
            self.authority.live,
        )

    async def documents(
        self, snapshot: CorpusSnapshot, *, after: str | None, limit: int
    ) -> list[SnapshotDocument]:
        documents = [
            self._stored_document(snapshot, value[0])
            for key, value in sorted(self.snapshots[snapshot.id].items())
            if after is None or key > after
        ][:limit]
        self.max_document_page = max(self.max_document_page, len(documents))
        return documents

    async def document(self, snapshot: CorpusSnapshot, document_id: str) -> SnapshotDocument | None:
        found = self.snapshots[snapshot.id].get(document_id)
        return None if found is None else self._stored_document(snapshot, found[0])

    async def chunks(
        self,
        snapshot: CorpusSnapshot,
        document_id: str,
        *,
        after: ChunkKey | None,
        limit: int,
    ) -> list[SnapshotChunk]:
        chunks = sorted(
            self.snapshots[snapshot.id][document_id][1],
            key=lambda chunk: (chunk.position, chunk.id),
        )
        page = [
            chunk for chunk in chunks if after is None or ChunkKey(chunk.position, chunk.id) > after
        ][:limit]
        self.max_chunk_page = max(self.max_chunk_page, len(page))
        document = self.snapshots[snapshot.id][document_id][0]
        return [
            SnapshotChunk(
                chunk=chunk,
                vector_id=f"{document.publication_id}:{chunk.id}",
                sequence=chunk.position + 1,
            )
            for chunk in page
        ]

    def _stored_document(self, snapshot: CorpusSnapshot, document: Document) -> SnapshotDocument:
        chunk_fp, embed_fp, glossary_fp = self.snapshot_lineage[snapshot.id][document.id]
        return SnapshotDocument(
            workspace_id="default",
            document=document,
            chunk_fingerprint=chunk_fp,
            embed_fingerprint=embed_fp,
            glossary_fingerprint=glossary_fp,
        )


def synthetic_corpus(authority: Authority, count: int = 3, *, secret_uri: bool = False) -> Corpus:
    uri = (
        "https://wiki.example.test/private/synthetic-sensitive?id=hidden"
        if secret_uri
        else "https://wiki.example.test/pages/synthetic-1"
    )
    document = make_document().model_copy(
        update={"uri": uri, "metadata": {"synthetic_label": "internal-only"}}
    )
    return Corpus(authority, [(document, make_chunks(document, count=count))])


async def prepare_run(
    authority: Authority, corpus: Corpus, embedder: HashEmbedder, run_id: str = "run-1"
) -> ReembedRun:
    return await start_reembed(
        run_id,
        owner_token="owner-a",  # noqa: S106 - synthetic fenced lease identity, not a password
        corpus=corpus,
        target=embedder.fingerprint,
        journal=authority,
    )


async def execute(
    run: ReembedRun,
    authority: Authority,
    corpus: Corpus,
    embedder: HashEmbedder,
    *,
    owner: str = "owner-a",
    ttl: float = 30.0,
) -> ReembedRun:
    return await resume_reembed(
        run.id,
        owner_token=owner,
        corpus=corpus,
        embedder=embedder,
        journal=authority,
        shadow=authority,
        publisher=authority,
        lease_ttl_seconds=ttl,
    )


async def test_dry_run_is_aggregate_only_and_prices_combined_bounded_memory() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 130, secret_uri=True)
    embedder = CountingEmbedder(dimension=7)

    plan = await plan_reembed(corpus, embedder.fingerprint, chunks_per_second=2.0)
    repeated = await plan_reembed(corpus, embedder.fingerprint, chunks_per_second=2.0)

    assert (plan.documents, plan.chunks) == (1, 130)
    assert repeated.inventory_digest == plan.inventory_digest
    assert plan.estimated_seconds == 65.0
    assert plan.peak_memory_bytes > 64 * 7 * 4
    assert plan.temporary_disk_bytes == 130 * (7 * 4 + 512)
    assert corpus.max_chunk_page == 64
    assert corpus.max_document_page == 1
    assert embedder.calls == []
    rendered = repr(dict(plan.public_facts()))
    assert "synthetic-sensitive" not in rendered
    assert "wiki.example.test" not in rendered
    assert make_document().id not in rendered
    assert "wiki.example.test" not in repr(plan)


async def test_huge_document_build_is_keyset_paged_and_live_reads_stay_old_until_swap() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 130)
    embedder = CountingEmbedder(dimension=5)
    run = await prepare_run(authority, corpus, embedder)

    completed = await execute(run, authority, corpus, embedder)

    assert completed.state is ReembedState.PUBLISHED
    assert completed.chunks_completed == 130
    assert corpus.max_chunk_page == 64
    assert authority.max_upsert_batch == 64
    assert [len(call) for call in embedder.calls] == [64, 64, 2]
    assert set(authority.live_during_upsert) == {"live-old"}
    assert authority.live.generation_id == completed.shadow_generation_id


async def test_prepare_is_create_if_absent_and_never_clears_resumed_rows() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 2)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    authority.fail_chunk_checkpoint_once = True

    with pytest.raises(OSError, match="chunk checkpoint crash"):
        await execute(run, authority, corpus, embedder)
    generation_id = f"shadow:{run.id}"
    rows_after_crash = dict(authority.rows[generation_id])
    assert len(rows_after_crash) == 2

    completed = await execute(run, authority, corpus, embedder)

    assert completed.state is ReembedState.PUBLISHED
    assert authority.prepare_calls == 2
    assert authority.rows[generation_id].keys() == rows_after_crash.keys()
    assert authority.upsert_attempts == 4


async def test_prepare_refuses_to_rebind_an_existing_shadow_identity() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    lease = authority.leases[run.id]
    original = await authority.open_or_create(
        run.id,
        fingerprint=embedder.fingerprint,
        inventory_digest=run.plan.inventory_digest,
        lease=lease,
    )
    authority.rows[original.id]["sentinel"] = (
        corpus.snapshots[run.plan.snapshot.id][make_document().id][1][0],
        (1.0,),
    )

    with pytest.raises(ReembedError, match="identity is immutable"):
        await authority.open_or_create(
            run.id,
            fingerprint=embedder.fingerprint,
            inventory_digest="different-inventory",
            lease=lease,
        )
    assert "sentinel" in authority.rows[original.id]


async def test_publish_receipt_prevents_retry_from_rolling_newer_generation_back() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    authority.fail_terminal_checkpoint_once = True

    with pytest.raises(OSError, match="publish checkpoint crash"):
        await execute(run, authority, corpus, embedder)
    first_receipt = authority.receipts[run.id]
    assert first_receipt.outcome is PublishOutcome.PUBLISHED
    authority.install_competing_winner("generation-b")

    completed = await execute(run, authority, corpus, embedder)

    assert completed.receipt == first_receipt
    assert authority.live.generation_id == "generation-b"
    assert completed.state is ReembedState.PUBLISHED


async def test_atomic_publication_cas_reports_superseded_without_touching_winner() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    authority.install_competing_winner("generation-concurrent")

    completed = await execute(run, authority, corpus, embedder)

    assert completed.state is ReembedState.SUPERSEDED
    assert completed.receipt is not None
    assert completed.receipt.outcome is PublishOutcome.SUPERSEDED
    assert completed.receipt.observed_winner.generation_id == "generation-concurrent"
    assert authority.live.generation_id == "generation-concurrent"


async def test_expired_owner_is_fenced_from_shadow_journal_and_publication_after_takeover() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    lease_a = await authority.acquire(run.id, "owner-a", ttl_seconds=1.0)
    generation = await authority.open_or_create(
        run.id,
        fingerprint=embedder.fingerprint,
        inventory_digest=run.plan.inventory_digest,
        lease=lease_a,
    )
    authority.advance(2.0)
    lease_b = await authority.acquire(run.id, "owner-b", ttl_seconds=10.0)
    assert lease_b.generation > lease_a.generation
    chunk = corpus.snapshots[run.plan.snapshot.id][make_document().id][1][0]

    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.renew(run.id, lease_a, ttl_seconds=5.0)
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.save(run, expected_revision=run.revision, lease=lease_a)
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.open_or_create(
            run.id,
            fingerprint=embedder.fingerprint,
            inventory_digest=run.plan.inventory_digest,
            lease=lease_a,
        )
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.upsert(generation, [chunk], [[1.0] * 5], lease=lease_a)
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.publish(
            run,
            generation,
            expected=run.plan.snapshot.live,
            expected_corpus_revision=run.plan.snapshot.revision,
            lease=lease_a,
        )
    assert authority.live.generation_id == "live-old"


async def test_renewal_extends_ownership_and_takeover_waits_for_the_new_expiry() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    lease_a = authority.leases[run.id]
    authority.advance(20.0)
    renewed = await authority.renew(run.id, lease_a, ttl_seconds=30.0)
    assert renewed.generation == lease_a.generation
    authority.advance(20.0)

    with pytest.raises(ReembedError, match="another owner"):
        await authority.acquire(run.id, "owner-b", ttl_seconds=30.0)
    authority.advance(11.0)
    lease_b = await authority.acquire(run.id, "owner-b", ttl_seconds=30.0)
    assert lease_b.generation == renewed.generation + 1


async def test_complete_publication_identity_and_metadata_change_inventory_digest() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    first = await plan_reembed(corpus, embedder.fingerprint)
    document, chunks = corpus.current[make_document().id]
    corpus.current[document.id] = (
        document.model_copy(
            update={
                "publication_id": "publication-moved",
                "metadata": {"synthetic_label": "metadata-moved"},
            }
        ),
        chunks,
    )

    second = await plan_reembed(corpus, embedder.fingerprint)
    corpus.current_lineage[document.id] = (
        "chunk-fingerprint",
        "embed-fingerprint-moved",
        "glossary-fingerprint",
    )
    third = await plan_reembed(corpus, embedder.fingerprint)

    assert first.inventory_digest != second.inventory_digest
    assert second.inventory_digest != third.inventory_digest


async def test_operational_embedder_change_omitted_from_canonical_identity_is_refused() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    planned = CountingEmbedder(dimension=4)
    run = await prepare_run(authority, corpus, planned)
    changed = CountingEmbedder(dimension=4)
    changed.fingerprint = changed.fingerprint.model_copy(update={"max_sequence_length": 64})
    assert changed.fingerprint.canonical() == planned.fingerprint.canonical()

    with pytest.raises(ReembedError, match="context limit"):
        await execute(run, authority, corpus, changed)
    assert authority.generations == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"finite": False}, "non-finite"),
        ({"dimension": 99}, "expected dimension"),
        ({"lineage_valid": False}, "invalid chunk or embedding lineage"),
        ({"retrieval_ready": False}, "retrieval-readiness"),
    ],
)
async def test_validation_faults_never_replace_live_generation(
    change: dict[str, object], message: str
) -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder(dimension=4)
    run = await prepare_run(authority, corpus, embedder)
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
    authority.inspection_override = replace(valid, **change)

    with pytest.raises(ReembedValidationError, match=message):
        await execute(run, authority, corpus, embedder)
    assert authority.live.generation_id == "live-old"


async def test_missing_inputs_report_only_an_aggregate_count() -> None:
    authority = Authority()
    document = make_document("synthetic private body").model_copy(
        update={"uri": "https://wiki.example.test/private/synthetic-missing?token=hidden"}
    )
    corpus = Corpus(authority, [(document, [])])
    embedder = CountingEmbedder()

    plan = await plan_reembed(corpus, embedder.fingerprint)

    assert plan.unrepairable_documents == 1
    public = repr(dict(plan.public_facts()))
    assert document.id not in public
    assert document.uri not in public
    assert "synthetic private body" not in public
    assert embedder.calls == []
