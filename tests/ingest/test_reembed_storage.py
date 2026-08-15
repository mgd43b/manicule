"""On-disk recovery and publication evidence for the concrete re-embed adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import insert, select, update

from manicule.core.ids import vector_id
from manicule.ingest.reembed import (
    ChunkKey,
    CorpusSnapshot,
    LivePublication,
    PublishOutcome,
    ReembedError,
    ReembedState,
    SnapshotChunk,
    SnapshotChunkDigester,
    SnapshotDocument,
    resume_reembed,
    start_reembed,
)
from manicule.storage import models
from manicule.storage.engine import VECTORS_DIRNAME
from manicule.storage.reembed import LanceShadowGenerations, SqliteReembedStore
from manicule.storage.types import utcnow
from manicule.storage.vectors import LanceVectorStore, PublishedLanceVectorStore, table_name
from tests.fakes import HashEmbedder
from tests.storage_helpers import fingerprint, make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.ingest.reembed import ReembedLease, ReembedRun, ShadowGeneration
    from manicule.storage.docstore import SqliteDocStore


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class LocalSnapshotCorpus:
    """A connector-free immutable local snapshot feeding the concrete storage evidence."""

    def __init__(
        self, snapshot: CorpusSnapshot, document: SnapshotDocument, chunk: SnapshotChunk
    ) -> None:
        self.snapshot = snapshot
        self.stored_document = document
        self.stored_chunk = chunk

    async def begin_snapshot(self) -> CorpusSnapshot:
        return self.snapshot

    async def documents(
        self, snapshot: CorpusSnapshot, *, after: str | None, limit: int
    ) -> Sequence[SnapshotDocument]:
        assert snapshot == self.snapshot
        if limit <= 0 or (after is not None and self.stored_document.document.id <= after):
            return []
        return [self.stored_document]

    async def document(self, snapshot: CorpusSnapshot, document_id: str) -> SnapshotDocument | None:
        assert snapshot == self.snapshot
        return self.stored_document if document_id == self.stored_document.document.id else None

    async def chunks(
        self,
        snapshot: CorpusSnapshot,
        document_id: str,
        *,
        after: ChunkKey | None,
        limit: int,
    ) -> Sequence[SnapshotChunk]:
        assert snapshot == self.snapshot
        key = ChunkKey(self.stored_chunk.chunk.position, self.stored_chunk.chunk.id)
        if document_id != self.stored_chunk.chunk.document_id or limit <= 0:
            return []
        return [self.stored_chunk] if after is None or key > after else []


async def seeded_run(
    engine: AsyncEngine,
    store: SqliteDocStore,
    data_dir: Path,
    clock: Clock,
    *,
    run_id: str,
) -> tuple[
    SqliteReembedStore,
    LanceShadowGenerations,
    ReembedRun,
    ReembedLease,
    ShadowGeneration,
    SnapshotChunk,
    list[float],
    LocalSnapshotCorpus,
]:
    document = make_document().model_copy(update={"publication_id": "old-publication"})
    chunk = make_chunk(document, 0, "durable local input")
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [chunk])
    source = SnapshotChunk(chunk, vector_id(document.publication_id, chunk.id), 1)
    digester = SnapshotChunkDigester()
    digester.add(source)
    inventory = digester.hexdigest()
    old_fingerprint = fingerprint(dimension=4, model_id="old/model")
    old_vectors = LanceVectorStore(data_dir / VECTORS_DIRNAME)
    await old_vectors.ensure_ready(old_fingerprint)
    await old_vectors.upsert(
        [chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=document.publication_id
    )
    await old_vectors.teardown()
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                id=1,
                vector_table=table_name(old_fingerprint),
                embed_fingerprint=old_fingerprint.model_dump_json(),
                vector_inventory_digest=inventory,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        revision = str(
            (
                await connection.execute(
                    select(models.CorpusRevision.revision).where(models.CorpusRevision.id == 1)
                )
            ).scalar_one()
        )
    target = HashEmbedder(dimension=4).fingerprint
    live = LivePublication(table_name(old_fingerprint), old_fingerprint.canonical(), inventory)
    snapshot = CorpusSnapshot("on-disk-snapshot", revision, live)
    corpus = LocalSnapshotCorpus(
        snapshot,
        SnapshotDocument(workspace_id="default", document=document),
        source,
    )
    authority = SqliteReembedStore(engine, clock=clock)
    run = await start_reembed(
        run_id,
        owner_token="owner-a",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        target=target,
        journal=authority,
        lease_ttl_seconds=30.0,
    )
    lease = await authority.acquire(run.id, "owner-a", ttl_seconds=30.0)
    shadows = LanceShadowGenerations(data_dir / VECTORS_DIRNAME, authority)
    generation = await shadows.open_or_create(
        run.id, fingerprint=target, inventory_digest=inventory, lease=lease
    )
    return authority, shadows, run, lease, generation, source, [0.0, 1.0, 0.0, 0.0], corpus


async def test_on_disk_shadow_validates_publishes_and_runtime_follows_pointer(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="durable-handoff"
    )

    await shadows.upsert(generation, [source], [vector], lease=lease)
    inspection = await shadows.inspect(generation, lease=lease)
    assert inspection.rows == inspection.unique_chunks == 1
    assert inspection.inventory_digest == run.commitment.chunk_inventory_digest
    assert inspection.lineage_valid
    assert inspection.retrieval_ready
    receipt = await authority.publish(
        run,
        generation,
        expected=run.commitment.snapshot.live,
        expected_corpus_revision=run.commitment.snapshot.revision,
        lease=lease,
    )

    assert receipt.outcome is PublishOutcome.PUBLISHED
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)
    await live.ensure_ready(HashEmbedder(dimension=4).fingerprint)
    rows = await live.search(vector, 1)
    assert len(rows) == 1
    assert rows[0].publication_id == generation.id
    assert (await store.get_document(source.chunk.document_id)).publication_id == generation.id  # type: ignore[union-attr]
    await live.teardown()


async def test_orchestration_resumes_idempotently_across_concrete_adapter_instances(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, _, _, _, _, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="durable-resume"
    )
    embedder = HashEmbedder(dimension=4)

    completed = await resume_reembed(
        run.id,
        owner_token="owner-a",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        embedder=embedder,
        journal=authority,
        shadow=shadows,
        publisher=authority,
        lease_ttl_seconds=30.0,
    )
    reopened = SqliteReembedStore(engine, clock=clock)
    resumed = await resume_reembed(
        run.id,
        owner_token="owner-a",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        embedder=embedder,
        journal=reopened,
        shadow=LanceShadowGenerations(data_dir / VECTORS_DIRNAME, reopened),
        publisher=reopened,
        lease_ttl_seconds=30.0,
    )

    assert completed.state is ReembedState.PUBLISHED
    assert resumed == completed
    assert resumed.receipt is not None


async def test_writer_lock_spans_lance_mutation_then_stale_owner_is_refused(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, _, run, lease_a, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="fenced-mutation"
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def barrier() -> None:
        entered.set()
        await release.wait()

    shadows = LanceShadowGenerations(data_dir / VECTORS_DIRNAME, authority, mutation_hook=barrier)
    writing = asyncio.create_task(shadows.upsert(generation, [source], [vector], lease=lease_a))
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    old = LanceVectorStore(data_dir / VECTORS_DIRNAME)
    await old.ensure_ready(fingerprint(dimension=4, model_id="old/model"))
    assert (await old.search([1.0, 0.0, 0.0, 0.0], 1))[0].publication_id == "old-publication"
    clock.advance(31.0)
    takeover = asyncio.create_task(authority.acquire(run.id, "owner-b", ttl_seconds=30.0))
    await asyncio.sleep(0.05)
    assert not takeover.done(), "SQLite's writer lock must prevent takeover during Lance I/O"
    release.set()
    await asyncio.wait_for(writing, timeout=2.0)
    lease_b = await asyncio.wait_for(takeover, timeout=2.0)
    assert lease_b.generation > lease_a.generation

    with pytest.raises(ReembedError, match="stale or expired"):
        await shadows.upsert(generation, [source], [vector], lease=lease_a)
    assert (await shadows.inspect(generation, lease=lease_b)).rows == 1
    await old.teardown()


async def test_inventory_cas_records_superseded_without_changing_live_rows(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="inventory-cas"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    async with engine.begin() as connection:
        await connection.execute(
            update(models.IndexState)
            .where(models.IndexState.id == 1)
            .values(vector_inventory_digest="competing-inventory")
        )

    receipt = await authority.publish(
        run,
        generation,
        expected=run.commitment.snapshot.live,
        expected_corpus_revision=run.commitment.snapshot.revision,
        lease=lease,
    )

    assert receipt.outcome is PublishOutcome.SUPERSEDED
    assert receipt.observed_winner.inventory_digest == "competing-inventory"
    document = await store.get_document(source.chunk.document_id)
    assert document is not None
    assert document.publication_id == "old-publication"
    assert await authority.live_generation_id() == run.commitment.snapshot.live.generation_id


async def test_publish_receipt_survives_checkpoint_crash_and_cleanup_is_terminal_only(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="receipt-crash"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    with pytest.raises(ReembedError, match="only failed or superseded"):
        await shadows.cleanup_terminal(run.id)
    first = await authority.publish(
        run,
        generation,
        expected=run.commitment.snapshot.live,
        expected_corpus_revision=run.commitment.snapshot.revision,
        lease=lease,
    )
    terminal = await authority.save(
        replace(run, state=ReembedState.SUPERSEDED, shadow_generation_id=generation.id),
        expected_revision=run.revision,
        lease=lease,
    )
    assert terminal.state is ReembedState.SUPERSEDED
    with pytest.raises(ReembedError, match="live shadow generation"):
        await shadows.cleanup_terminal(run.id)
    async with engine.begin() as connection:
        await connection.execute(
            update(models.IndexState)
            .where(models.IndexState.id == 1)
            .values(vector_table="reembed-competing-winner")
        )
    reopened = SqliteReembedStore(engine, clock=clock)
    retried = await reopened.publish(
        run,
        generation,
        expected=run.commitment.snapshot.live,
        expected_corpus_revision=run.commitment.snapshot.revision,
        lease=lease,
    )
    assert retried == first
    async with engine.connect() as connection:
        assert (
            await connection.execute(
                select(models.IndexState.vector_table).where(models.IndexState.id == 1)
            )
        ).scalar_one() == "reembed-competing-winner"

    assert await shadows.cleanup_terminal(run.id)
    assert not shadows.directory(generation.id).exists()


async def test_explicit_abandonment_makes_an_unfinished_generation_cleanup_eligible(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="abandoned-build"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)

    abandoned = await authority.abandon(run.id, lease=lease)

    assert abandoned.state is ReembedState.FAILED
    assert abandoned.failure == "abandoned by operator"
    assert await shadows.cleanup_terminal(run.id)
    assert not shadows.directory(generation.id).exists()
