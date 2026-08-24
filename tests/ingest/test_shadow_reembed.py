"""Adversarial contracts for offline shadow-generation re-embedding."""

from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import override

import pytest

from manicule.core.content import Chunk, Document
from manicule.core.embedding import (
    VECTOR_CHECKSUM_VERSION,
    EmbedFingerprint,
    Vector,
    VectorIntegrity,
    vector_checksum,
    verify_stored_checksum,
)
from manicule.core.errors import ContextOverflowError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.ingest.reembed import (
    ChunkKey,
    CorpusSnapshot,
    LivePublication,
    PublicationReceipt,
    PublishOutcome,
    ReembedCommitment,
    ReembedError,
    ReembedLease,
    ReembedRun,
    ReembedState,
    ReembedValidationError,
    ShadowGeneration,
    ShadowInspection,
    SnapshotChunk,
    SnapshotChunkDigester,
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


class ExactCountingEmbedder(CountingEmbedder):
    def __init__(self, dimension: int = 5) -> None:
        super().__init__(dimension=dimension)
        self.fingerprint = self.fingerprint.model_copy(
            update={
                "tokenizer_id": "synthetic/characters-v1",
                "max_sequence_length": 1024,
            }
        )

    def count_tokens(self, text: str) -> int:
        return len(text)


class Authority:
    """Single-event-loop protocol harness for journal, shadow, fencing, and publication.

    It demonstrates the orchestration's required adapter semantics and crash decisions. It is
    deliberately not evidence that SQLite and Lance implement those semantics; their separate
    on-disk adapter suite supplies that evidence. Durable corpus snapshots and operator surfaces
    remain issue #187 work.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.corpus_revision = "revision-1"
        self.live = LivePublication("live-old", "old-fingerprint", "old-inventory")
        self.runs: dict[str, ReembedRun] = {}
        self.leases: dict[str, ReembedLease] = {}
        self.highest_fence: dict[str, int] = {}
        self.generations: dict[str, ShadowGeneration] = {}
        self.seals: dict[str, ShadowInspection] = {}
        self.rows: dict[str, dict[str, tuple[SnapshotChunk, tuple[float, ...], str]]] = {}
        self.receipts: dict[str, PublicationReceipt] = {}
        self.prepare_calls = 0
        self.upsert_attempts = 0
        self.max_upsert_batch = 0
        self.live_during_upsert: list[str] = []
        self.inspection_override: ShadowInspection | None = None
        self.fail_chunk_checkpoint_once = False
        self.pause_first_upsert = False
        self.upsert_entered = asyncio.Event()
        self.release_upsert = asyncio.Event()
        self.pause_inspection = False
        self.inspection_entered = asyncio.Event()
        self.release_inspection = asyncio.Event()

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
        commitment: ReembedCommitment,
        *,
        owner_token: str,
        ttl_seconds: float,
    ) -> tuple[ReembedRun, ReembedLease]:
        run = ReembedRun(id=run_id, commitment=commitment)
        existing = self.runs.get(run_id)
        if existing is not None and existing.commitment != commitment:
            raise ReembedError("run id already belongs to another immutable plan")
        lease = await self.acquire(run_id, owner_token, ttl_seconds=ttl_seconds)
        if existing is None:
            self.runs[run_id] = run
            existing = run
        return existing, lease

    async def create_released(self, run_id: str, commitment: ReembedCommitment) -> ReembedRun:
        run = ReembedRun(id=run_id, commitment=commitment)
        existing = self.runs.get(run_id)
        if existing is not None and existing.commitment != commitment:
            raise ReembedError("run id already belongs to another immutable plan")
        if existing is None:
            self.runs[run_id] = run
            existing = run
        self.leases.pop(run_id, None)
        return existing

    async def get(self, run_id: str) -> ReembedRun | None:
        run = self.runs.get(run_id)
        receipt = self.receipts.get(run_id)
        if run is None or receipt is None:
            return run
        terminal = (
            ReembedState.PUBLISHED
            if receipt.outcome is PublishOutcome.PUBLISHED
            else ReembedState.SUPERSEDED
        )
        return replace(run, state=terminal, receipt=receipt, failure="")

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

    async def release(self, run_id: str, lease: ReembedLease) -> None:
        current = self.leases.get(run_id)
        if (
            current is None
            or current.owner_token != lease.owner_token
            or current.generation != lease.generation
        ):
            raise ReembedError("stale or expired re-embedding lease")
        self.leases[run_id] = replace(current, expires_at=self.now)

    async def save(
        self, run: ReembedRun, *, expected_revision: int, lease: ReembedLease
    ) -> ReembedRun:
        self._assert_lease(run.id, lease)
        current = self.runs[run.id]
        if current.revision != expected_revision:
            raise ReembedError("stale journal revision")
        if self.fail_chunk_checkpoint_once and run.chunks_completed > current.chunks_completed:
            self.fail_chunk_checkpoint_once = False
            raise OSError("synthetic chunk checkpoint crash")
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
        chunks: Sequence[SnapshotChunk],
        vectors: Sequence[Vector],
        *,
        lease: ReembedLease,
    ) -> None:
        self._assert_lease(generation.run_id, lease)
        if generation.id in self.seals:
            raise ReembedError("sealed shadow generation is immutable")
        if self.pause_first_upsert and not self.upsert_entered.is_set():
            self.upsert_entered.set()
            await self.release_upsert.wait()
        # The barrier stands in for embedding/backend-lock latency. Ownership must be tested
        # again at the physical mutation boundary, after every await that can admit takeover.
        self._assert_lease(generation.run_id, lease)
        self.live_during_upsert.append(self.live.generation_id)
        self.upsert_attempts += len(chunks)
        self.max_upsert_batch = max(self.max_upsert_batch, len(chunks))
        target = self.rows[generation.id]
        for stored, vector in zip(chunks, vectors, strict=True):
            physical_id = f"{generation.id}:{stored.chunk.id}"
            # A checksum over exactly what this store keeps, recorded on write and recomputed in
            # `inspect` — the same two-sided rule the Lance backend follows, so a validation
            # test run against this fake exercises the numerical fence rather than skipping it.
            target[physical_id] = (stored, tuple(vector), vector_checksum(vector))

    async def inspect(
        self, generation: ShadowGeneration, *, lease: ReembedLease
    ) -> ShadowInspection:
        self._assert_lease(generation.run_id, lease)
        if self.pause_inspection:
            self.inspection_entered.set()
            await self.release_inspection.wait()
        self._assert_lease(generation.run_id, lease)
        if self.inspection_override is not None:
            return self.inspection_override
        rows = self.rows[generation.id]
        dimensions = {len(vector) for _, vector, _ in rows.values()}
        stored_rows = sorted(
            (stored for stored, _, _ in rows.values()),
            key=lambda item: (
                item.chunk.document_id,
                item.chunk.position,
                item.chunk.id,
                item.vector_id,
                item.publication_id,
                item.sequence,
            ),
        )
        inventory = SnapshotChunkDigester()
        for stored in stored_rows:
            inventory.add(stored)
        verdicts = [
            verify_stored_checksum(
                vector, recorded=checksum, version=VECTOR_CHECKSUM_VERSION, required=True
            )
            for _, vector, checksum in rows.values()
        ]
        failures: dict[str, int] = {}
        for verdict in verdicts:
            if verdict is not VectorIntegrity.VERIFIED:
                failures[verdict.value] = failures.get(verdict.value, 0) + 1
        return ShadowInspection(
            rows=len(rows),
            unique_chunks=len({stored.chunk.id for stored in stored_rows}),
            dimension=dimensions.pop() if len(dimensions) == 1 else 0,
            finite=all(math.isfinite(value) for _, vector, _ in rows.values() for value in vector),
            fingerprint=generation.fingerprint,
            inventory_digest=inventory.hexdigest(),
            lineage_valid=all(
                physical_id == f"{generation.id}:{stored.chunk.id}"
                for physical_id, (stored, _, _) in rows.items()
            ),
            retrieval_ready=True,
            checksums_verified=sum(
                1 for verdict in verdicts if verdict is VectorIntegrity.VERIFIED
            ),
            checksum_failures=dict(sorted(failures.items())),
        )

    async def seal(
        self,
        generation: ShadowGeneration,
        inspection: ShadowInspection,
        *,
        lease: ReembedLease,
    ) -> None:
        self._assert_lease(generation.run_id, lease)
        actual = await self.inspect(generation, lease=lease)
        if actual != inspection:
            raise ReembedError("shadow changed between inspection and seal")
        existing = self.seals.setdefault(generation.id, inspection)
        if existing != inspection:
            raise ReembedError("shadow seal is immutable")

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
        if self.seals.get(generation.id) is None:
            raise ReembedError("shadow generation is not sealed")
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
        self.runs[run.id] = replace(
            run,
            state=(
                ReembedState.PUBLISHED
                if receipt.outcome is PublishOutcome.PUBLISHED
                else ReembedState.SUPERSEDED
            ),
            receipt=receipt,
            revision=run.revision + 1,
        )
        return receipt


class ReleaseFailAuthority(Authority):
    @override
    async def release(self, run_id: str, lease: ReembedLease) -> None:
        del run_id, lease
        raise OSError("synthetic release failure")


class Corpus:
    def __init__(
        self, authority: Authority, documents: Sequence[tuple[Document, Sequence[Chunk]]]
    ) -> None:
        self.authority = authority
        initial = {document.id: (document, tuple(chunks)) for document, chunks in documents}
        chunk_fingerprint = ChunkFingerprint(
            chunker="synthetic",
            version="1",
            max_tokens=512,
            overlap_tokens=0,
            tokenizer_id="whitespace",
        ).canonical()
        lineage = {
            document.id: (chunk_fingerprint, "embed-fingerprint", "glossary-fingerprint")
            for document, _ in documents
        }
        self.current_view = "view-1"
        self.versions: dict[str, dict[str, tuple[Document, tuple[Chunk, ...]]]] = {
            self.current_view: initial
        }
        self.lineage_versions: dict[str, dict[str, tuple[str, str, str]]] = {
            self.current_view: lineage
        }
        self.next_view = 2
        self.max_document_page = 0
        self.max_chunk_page = 0
        self.document_reads = 0
        self.chunk_reads = 0

    async def begin_snapshot(self) -> CorpusSnapshot:
        return CorpusSnapshot(
            self.current_view,
            self.authority.corpus_revision,
            self.authority.live,
        )

    async def discard_snapshot(self, snapshot_id: str) -> None:
        if snapshot_id != self.current_view:
            self.versions.pop(snapshot_id, None)
            self.lineage_versions.pop(snapshot_id, None)

    async def documents(
        self, snapshot: CorpusSnapshot, *, after: str | None, limit: int
    ) -> list[SnapshotDocument]:
        self.document_reads += 1
        documents = [
            self._stored_document(snapshot, value[0])
            for key, value in sorted(self.versions[snapshot.id].items())
            if after is None or key > after
        ][:limit]
        self.max_document_page = max(self.max_document_page, len(documents))
        return documents

    async def document(self, snapshot: CorpusSnapshot, document_id: str) -> SnapshotDocument | None:
        found = self.versions[snapshot.id].get(document_id)
        return None if found is None else self._stored_document(snapshot, found[0])

    async def chunks(
        self,
        snapshot: CorpusSnapshot,
        document_id: str,
        *,
        after: ChunkKey | None,
        limit: int,
    ) -> list[SnapshotChunk]:
        self.chunk_reads += 1
        chunks = sorted(
            self.versions[snapshot.id][document_id][1],
            key=lambda chunk: (chunk.position, chunk.id),
        )
        page = [
            chunk for chunk in chunks if after is None or ChunkKey(chunk.position, chunk.id) > after
        ][:limit]
        self.max_chunk_page = max(self.max_chunk_page, len(page))
        document = self.versions[snapshot.id][document_id][0]
        return [
            SnapshotChunk(
                chunk=chunk,
                vector_id=f"{document.publication_id}:{chunk.id}",
                publication_id=document.publication_id,
                sequence=chunk.position + 1,
            )
            for chunk in page
        ]

    def _stored_document(self, snapshot: CorpusSnapshot, document: Document) -> SnapshotDocument:
        chunk_fp, embed_fp, glossary_fp = self.lineage_versions[snapshot.id][document.id]
        return SnapshotDocument(
            workspace_id="default",
            document=document,
            chunk_fingerprint=chunk_fp,
            embed_fingerprint=embed_fp,
            glossary_fingerprint=glossary_fp,
        )

    def replace_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        rows = dict(self.versions[self.current_view])
        rows[document.id] = (document, tuple(chunks))
        self._advance(rows, dict(self.lineage_versions[self.current_view]))

    def replace_lineage(self, document_id: str, *, embed_fingerprint: str) -> None:
        rows = dict(self.versions[self.current_view])
        lineage = dict(self.lineage_versions[self.current_view])
        chunk_fp, _, glossary_fp = lineage[document_id]
        lineage[document_id] = (chunk_fp, embed_fingerprint, glossary_fp)
        self._advance(rows, lineage)

    def _advance(
        self,
        rows: dict[str, tuple[Document, tuple[Chunk, ...]]],
        lineage: dict[str, tuple[str, str, str]],
    ) -> None:
        view = f"view-{self.next_view}"
        self.next_view += 1
        self.versions[view] = rows
        self.lineage_versions[view] = lineage
        self.current_view = view
        self.authority.corpus_revision = f"corpus:{view}"

    def raw_chunks(self, snapshot: CorpusSnapshot, document_id: str) -> tuple[Chunk, ...]:
        return self.versions[snapshot.id][document_id][1]


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
    run = await start_reembed(
        run_id,
        owner_token="owner-a",  # noqa: S106 - synthetic fenced lease identity, not a password
        corpus=corpus,
        target=embedder.fingerprint,
        journal=authority,
    )
    # Most orchestration tests begin at the worker boundary. ``start_reembed`` deliberately
    # returns its lease before surfacing the recovery id, so acquire the worker's fresh fence.
    await authority.acquire(run.id, "owner-a", ttl_seconds=30.0)
    return run


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

    assert (plan.documents, plan.chunks) == (1, 130)
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
    assert set(asdict(plan)) < set(plan.public_facts())


async def test_start_is_atomic_and_never_depends_on_a_followup_release() -> None:
    authority = ReleaseFailAuthority()
    corpus = synthetic_corpus(authority, 1)

    run = await start_reembed(
        "known-before-call",
        owner_token="planning-owner",  # noqa: S106 - synthetic fence identity
        corpus=corpus,
        target=HashEmbedder().fingerprint,
        journal=authority,
    )

    assert run.id == "known-before-call"
    assert run.id not in authority.leases


async def test_protocol_keyset_pages_huge_document_and_keeps_old_winner_until_swap() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 130)
    embedder = CountingEmbedder(dimension=5)
    run = await prepare_run(authority, corpus, embedder)
    authority.pause_first_upsert = True

    rebuilding = asyncio.create_task(execute(run, authority, corpus, embedder))
    await asyncio.wait_for(authority.upsert_entered.wait(), timeout=1.0)
    # A concurrent reader sees the old tuple while a shadow batch is waiting to commit.
    assert authority.live == run.commitment.snapshot.live
    authority.release_upsert.set()
    completed = await asyncio.wait_for(rebuilding, timeout=1.0)

    assert completed.state is ReembedState.PUBLISHED
    assert completed.chunks_completed == 130
    assert corpus.max_chunk_page == 64
    assert authority.max_upsert_batch == 64
    assert [len(call) for call in embedder.calls] == [64, 64, 2]
    assert set(authority.live_during_upsert) == {"live-old"}
    assert authority.live.generation_id == completed.shadow_generation_id


async def test_small_documents_share_one_bounded_model_and_shadow_batch() -> None:
    authority = Authority()
    documents: list[tuple[Document, Sequence[Chunk]]] = []
    for index in range(4):
        document = make_document().model_copy(
            update={
                "id": f"document-{index}",
                "source_id": f"document-{index}",
                "uri": f"fake://document-{index}",
                "title": f"Document {index}",
            }
        )
        documents.append((document, make_chunks(document, count=8)))
    corpus = Corpus(authority, documents)
    embedder = CountingEmbedder(dimension=5)
    run = await prepare_run(authority, corpus, embedder)

    completed = await execute(run, authority, corpus, embedder)

    assert completed.state is ReembedState.PUBLISHED
    assert completed.documents_completed == 4
    assert completed.chunks_completed == 32
    assert [len(call) for call in embedder.calls] == [32]
    assert authority.max_upsert_batch == 32


async def test_long_context_model_batches_retained_short_chunks_to_the_chunk_budget() -> None:
    """A long model context must not reduce 1,024-token stored chunks to eight at a time.

    Each model call also causes a fenced shadow write, so deriving this from the 8K model
    context would turn these 64 chunks into eight forward-pass/write pairs instead of one.
    """
    authority = Authority()
    documents: list[tuple[Document, Sequence[Chunk]]] = []
    for index in range(8):
        document = make_document().model_copy(
            update={
                "id": f"long-context-document-{index}",
                "source_id": f"long-context-document-{index}",
                "uri": f"fake://long-context-document-{index}",
                "title": f"Long context document {index}",
            }
        )
        documents.append((document, make_chunks(document, count=8)))
    corpus = Corpus(authority, documents)
    chunk_fingerprint = ChunkFingerprint(
        chunker="synthetic",
        version="1",
        max_tokens=1024,
        overlap_tokens=0,
        tokenizer_id="whitespace",
    ).canonical()
    corpus.lineage_versions[corpus.current_view] = dict.fromkeys(
        corpus.lineage_versions[corpus.current_view],
        (chunk_fingerprint, "embed-fingerprint", "glossary-fingerprint"),
    )
    embedder = CountingEmbedder(dimension=5)
    embedder.fingerprint = embedder.fingerprint.model_copy(update={"max_sequence_length": 8192})
    run = await prepare_run(authority, corpus, embedder, "long-context-batches")

    completed = await execute(run, authority, corpus, embedder)

    assert completed.state is ReembedState.PUBLISHED
    assert [len(call) for call in embedder.calls] == [64]
    assert authority.max_upsert_batch == 64


async def test_private_commitment_hides_snapshot_model_path_and_digests_from_repr() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1, secret_uri=True)
    embedder = CountingEmbedder()
    private_path = "/synthetic/private/model/weights.bin"
    embedder.fingerprint = embedder.fingerprint.model_copy(update={"weights_ref": private_path})

    run = await prepare_run(authority, corpus, embedder, "private-commitment")

    rendered = f"{run!r} {run.commitment!r} {run.commitment.plan!r}"
    assert private_path not in rendered
    assert "wiki.example.test" not in rendered
    assert run.commitment.snapshot.id not in rendered
    assert run.commitment.inventory_digest not in rendered


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
        inventory_digest=run.commitment.chunk_inventory_digest,
        lease=lease,
    )
    raw = corpus.raw_chunks(run.commitment.snapshot, make_document().id)[0]
    authority.rows[original.id]["sentinel"] = (
        SnapshotChunk(raw, f"legacy:{raw.id}", "legacy", raw.position + 1),
        (1.0,),
        vector_checksum((1.0,)),
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
    completed = await execute(run, authority, corpus, embedder)
    first_receipt = authority.receipts[run.id]
    assert first_receipt.outcome is PublishOutcome.PUBLISHED
    authority.runs[run.id] = replace(completed, state=ReembedState.READY, receipt=None)
    authority.install_competing_winner("generation-b")

    completed = await execute(run, authority, corpus, embedder)

    assert completed.receipt == first_receipt
    assert authority.live.generation_id == "generation-b"
    assert completed.state is ReembedState.PUBLISHED


async def test_publication_protocol_cas_reports_superseded_without_touching_winner() -> None:
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
        inventory_digest=run.commitment.chunk_inventory_digest,
        lease=lease_a,
    )
    authority.advance(2.0)
    lease_b = await authority.acquire(run.id, "owner-b", ttl_seconds=10.0)
    assert lease_b.generation > lease_a.generation
    chunk = corpus.raw_chunks(run.commitment.snapshot, make_document().id)[0]
    stored_chunk = SnapshotChunk(chunk, f"legacy:{chunk.id}", "legacy", chunk.position + 1)

    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.renew(run.id, lease_a, ttl_seconds=5.0)
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.save(run, expected_revision=run.revision, lease=lease_a)
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.open_or_create(
            run.id,
            fingerprint=embedder.fingerprint,
            inventory_digest=run.commitment.chunk_inventory_digest,
            lease=lease_a,
        )
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.upsert(generation, [stored_chunk], [[1.0] * 5], lease=lease_a)
    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.publish(
            run,
            generation,
            expected=run.commitment.snapshot.live,
            expected_corpus_revision=run.commitment.snapshot.revision,
            lease=lease_a,
        )
    assert authority.live.generation_id == "live-old"


async def test_takeover_while_upsert_waits_refuses_stale_physical_write() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 1)
    embedder = CountingEmbedder()
    run = await prepare_run(authority, corpus, embedder)
    lease_a = authority.leases[run.id]
    generation = await authority.open_or_create(
        run.id,
        fingerprint=embedder.fingerprint,
        inventory_digest=run.commitment.chunk_inventory_digest,
        lease=lease_a,
    )
    chunk = corpus.raw_chunks(run.commitment.snapshot, make_document().id)[0]
    stored_chunk = SnapshotChunk(chunk, f"legacy:{chunk.id}", "legacy", chunk.position + 1)
    authority.pause_first_upsert = True

    stale_write = asyncio.create_task(
        authority.upsert(generation, [stored_chunk], [[1.0] * 5], lease=lease_a)
    )
    await asyncio.wait_for(authority.upsert_entered.wait(), timeout=1.0)
    authority.advance(31.0)
    lease_b = await authority.acquire(run.id, "owner-b", ttl_seconds=30.0)
    assert lease_b.generation > lease_a.generation
    authority.release_upsert.set()

    with pytest.raises(ReembedError, match="stale or expired"):
        await asyncio.wait_for(stale_write, timeout=1.0)
    assert authority.rows[generation.id] == {}
    assert authority.upsert_attempts == 0
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
    first = await prepare_run(authority, corpus, embedder, "identity-1")
    view = first.commitment.snapshot
    document = (await corpus.document(view, make_document().id)).document  # type: ignore[union-attr]
    chunks = corpus.raw_chunks(view, document.id)
    corpus.replace_document(
        document.model_copy(
            update={
                "publication_id": "publication-moved",
                "metadata": {"synthetic_label": "metadata-moved"},
            }
        ),
        chunks,
    )

    second = await prepare_run(authority, corpus, embedder, "identity-2")
    corpus.replace_lineage(document.id, embed_fingerprint="embed-fingerprint-moved")
    third = await prepare_run(authority, corpus, embedder, "identity-3")

    assert first.commitment.inventory_digest != second.commitment.inventory_digest
    assert second.commitment.inventory_digest != third.commitment.inventory_digest


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


async def test_shadow_reembed_refuses_actual_text_beyond_the_stored_chunk_policy() -> None:
    authority = Authority()
    document = make_document()
    chunk = make_chunks(document, count=1)[0].model_copy(
        update={"embed_text": "x" * 513, "token_count": 1}
    )
    corpus = Corpus(authority, [(document, (chunk,))])
    chunk_fingerprint = ChunkFingerprint(
        chunker="structural",
        version="3",
        max_tokens=512,
        overlap_tokens=64,
        tokenizer_id="synthetic/characters-v1",
    )
    corpus.lineage_versions[corpus.current_view][document.id] = (
        chunk_fingerprint.canonical(),
        "embed-fingerprint",
        "glossary-fingerprint",
    )
    embedder = ExactCountingEmbedder(dimension=4)
    run = await prepare_run(authority, corpus, embedder, "chunk-policy-refusal")

    with pytest.raises(ContextOverflowError, match="fingerprinted 512-token"):
        await execute(run, authority, corpus, embedder)

    assert embedder.calls == []
    assert authority.upsert_attempts == 0
    assert authority.live.generation_id == "live-old"


async def test_packed_shadow_reembed_keeps_the_conservative_stored_context_guard() -> None:
    authority = Authority()
    document = make_document()
    chunk = make_chunks(document, count=1)[0].model_copy(
        update={"embed_text": "x", "token_count": 1_025}
    )
    corpus = Corpus(authority, [(document, (chunk,))])
    embedder = ExactCountingEmbedder(dimension=4)
    run = await prepare_run(authority, corpus, embedder, "stored-context-refusal")

    with pytest.raises(ContextOverflowError, match="exceed the 1024-token limit"):
        await execute(run, authority, corpus, embedder)

    assert embedder.calls == []
    assert authority.upsert_attempts == 0


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("vector_id", "physical chunk inventory"),
        ("sequence", "physical chunk inventory"),
        ("physical_key", "invalid chunk or embedding lineage"),
        ("missing", "expected 2 rows"),
        # The one no other row in this table can catch. Every case above corrupts *metadata* —
        # a vector id, a sequence, a physical key, a whole row — and the inventory digest or the
        # lineage check disagrees with it. Move one finite component of the vector itself and
        # every one of those still agrees; only the checksum written beside it does not.
        ("vector", "numerical integrity"),
    ],
)
async def test_inspection_recomputes_actual_physical_rows_and_rejects_corruption(
    corruption: str, message: str
) -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 2)
    embedder = CountingEmbedder(dimension=4)
    run = await prepare_run(authority, corpus, embedder, f"corrupt-{corruption}")
    authority.pause_inspection = True
    rebuilding = asyncio.create_task(execute(run, authority, corpus, embedder))
    await asyncio.wait_for(authority.inspection_entered.wait(), timeout=1.0)
    generation_id = f"shadow:{run.id}"
    rows = authority.rows[generation_id]
    physical_id = next(iter(rows))
    stored, vector, checksum = rows[physical_id]
    if corruption == "vector_id":
        rows[physical_id] = (replace(stored, vector_id="corrupt-vector-id"), vector, checksum)
    elif corruption == "sequence":
        rows[physical_id] = (replace(stored, sequence=stored.sequence + 100), vector, checksum)
    elif corruption == "physical_key":
        rows["wrong-physical-key"] = rows.pop(physical_id)
    elif corruption == "vector":
        moved = struct.unpack(">I", struct.pack(">f", vector[0]))[0] + 1
        rows[physical_id] = (
            stored,
            (struct.unpack(">f", struct.pack(">I", moved))[0], *vector[1:]),
            checksum,
        )
    else:
        rows.pop(physical_id)
    authority.release_inspection.set()

    with pytest.raises(ReembedValidationError, match=message):
        await asyncio.wait_for(rebuilding, timeout=1.0)
    assert authority.live.generation_id == "live-old"


async def test_validation_uses_shadow_evidence_without_rereading_the_snapshot() -> None:
    authority = Authority()
    corpus = synthetic_corpus(authority, 2)
    embedder = CountingEmbedder(dimension=4)
    run = await prepare_run(authority, corpus, embedder, "no-validation-snapshot-rescan")
    corpus.document_reads = corpus.chunk_reads = 0
    authority.pause_inspection = True

    rebuilding = asyncio.create_task(execute(run, authority, corpus, embedder))
    await asyncio.wait_for(authority.inspection_entered.wait(), timeout=1.0)

    # One complete small document needs a document page, its chunk page, then the empty
    # successor page to move BUILDING to VALIDATING.  Validation must inspect the completed
    # shadow rather than planning that immutable input a second time.
    assert corpus.document_reads == 2
    assert corpus.chunk_reads == 1

    authority.release_inspection.set()
    assert (await rebuilding).state is ReembedState.PUBLISHED


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
        fingerprint=run.commitment.target_fingerprint,
        inventory_digest=run.commitment.chunk_inventory_digest,
        lineage_valid=True,
        retrieval_ready=True,
        checksums_verified=1,
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
