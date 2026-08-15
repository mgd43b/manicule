"""On-disk recovery and publication evidence for the concrete re-embed adapters."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

import lancedb
import pytest
from sqlalchemy import insert, select, update

from manicule.core.ids import vector_id
from manicule.ingest.reembed import (
    PublishOutcome,
    ReembedError,
    ReembedState,
    SnapshotChunk,
    resume_reembed,
    start_reembed,
)
from manicule.storage import models
from manicule.storage.engine import VECTORS_DIRNAME
from manicule.storage.reembed import (
    LanceShadowGenerations,
    SqliteReembedCorpus,
    SqliteReembedStore,
)
from manicule.storage.types import utcnow
from manicule.storage.vectors import (
    LanceVectorStore,
    PublishedLanceVectorStore,
    VectorStoreStateError,
    quote,
    table_name,
)
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
    SqliteReembedCorpus,
]:
    document = make_document().model_copy(update={"publication_id": "old-publication"})
    chunk = make_chunk(document, 0, "durable local input")
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [chunk])
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
                vector_inventory_digest=None,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
    target = HashEmbedder(dimension=4).fingerprint
    corpus = SqliteReembedCorpus(engine)
    snapshot = await corpus.begin_snapshot()
    source = (await corpus.chunks(snapshot, document.id, after=None, limit=1))[0]
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
        run.id,
        fingerprint=target,
        inventory_digest=run.commitment.chunk_inventory_digest,
        lease=lease,
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
    with pytest.raises(ReembedError, match="exact validated shadow generation"):
        await shadows.seal(generation, replace(inspection, finite=False), lease=lease)
    await shadows.seal(generation, inspection, lease=lease)
    with pytest.raises(ReembedError, match="sealed shadow generation"):
        await shadows.upsert(generation, [source], [vector], lease=lease)
    receipt = await authority.publish(
        run,
        generation,
        expected=run.commitment.snapshot.live,
        expected_corpus_revision=run.commitment.snapshot.revision,
        lease=lease,
    )

    assert receipt.outcome is PublishOutcome.PUBLISHED
    async with engine.connect() as connection:
        revision = (
            await connection.execute(
                select(models.CorpusRevision.revision).where(models.CorpusRevision.id == 1)
            )
        ).scalar_one()
    assert str(revision) == run.commitment.snapshot.revision
    with pytest.raises(ReembedError, match="sealed shadow generation"):
        await shadows.upsert(generation, [source], [vector], lease=lease)
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)
    await live.ensure_ready(HashEmbedder(dimension=4).fingerprint)
    rows = await live.search(vector, 1)
    assert len(rows) == 1
    assert rows[0].publication_id == source.publication_id
    assert (
        await store.get_document(source.chunk.document_id)
    ).publication_id == source.publication_id  # type: ignore[union-attr]
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
    with pytest.raises(ReembedError, match="stale or expired"):
        await asyncio.wait_for(writing, timeout=2.0)
    lease_b = await asyncio.wait_for(takeover, timeout=2.0)
    assert lease_b.generation > lease_a.generation

    with pytest.raises(ReembedError, match="stale or expired"):
        await shadows.upsert(generation, [source], [vector], lease=lease_a)
    assert (await shadows.inspect(generation, lease=lease_b)).rows == 0
    await old.teardown()


@pytest.mark.parametrize("competing_inventory", ["competing-inventory", None])
async def test_inventory_cas_records_superseded_without_changing_live_rows(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path, competing_inventory: str | None
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="inventory-cas"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    inspection = await shadows.inspect(generation, lease=lease)
    await shadows.seal(generation, inspection, lease=lease)
    async with engine.begin() as connection:
        await connection.execute(
            update(models.IndexState)
            .where(models.IndexState.id == 1)
            .values(vector_inventory_digest=competing_inventory)
        )

    receipt = await authority.publish(
        run,
        generation,
        expected=run.commitment.snapshot.live,
        expected_corpus_revision=run.commitment.snapshot.revision,
        lease=lease,
    )

    assert receipt.outcome is PublishOutcome.SUPERSEDED
    assert receipt.observed_winner.inventory_digest == competing_inventory
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
    inspection = await shadows.inspect(generation, lease=lease)
    await shadows.seal(generation, inspection, lease=lease)
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


async def test_publication_requires_the_complete_durable_snapshot_record(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="missing-snapshot"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    inspection = await shadows.inspect(generation, lease=lease)
    await shadows.seal(generation, inspection, lease=lease)
    async with engine.begin() as connection:
        await connection.execute(
            update(models.ReembedCorpusSnapshot)
            .where(models.ReembedCorpusSnapshot.id == run.commitment.snapshot.id)
            .values(complete=False)
        )

    with pytest.raises(ReembedError, match="durable complete-corpus snapshot"):
        await authority.publish(
            run,
            generation,
            expected=run.commitment.snapshot.live,
            expected_corpus_revision=run.commitment.snapshot.revision,
            lease=lease,
        )


async def test_ordinary_corpus_mutation_invalidates_then_snapshot_bootstraps_inventory(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    _, _, run, _, _, source, _, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="inventory-bootstrap"
    )
    async with engine.connect() as connection:
        before = (
            await connection.execute(
                select(models.IndexState.vector_inventory_digest).where(models.IndexState.id == 1)
            )
        ).scalar_one()
    assert before == run.commitment.snapshot.live.inventory_digest

    await store.replace_chunks(source.chunk.document_id, [source.chunk])
    async with engine.connect() as connection:
        invalidated = (
            await connection.execute(
                select(models.IndexState.vector_inventory_digest).where(models.IndexState.id == 1)
            )
        ).scalar_one()
    assert invalidated is None

    rebuilt = await corpus.begin_snapshot()
    async with engine.connect() as connection:
        bootstrapped = (
            await connection.execute(
                select(models.IndexState.vector_inventory_digest).where(models.IndexState.id == 1)
            )
        ).scalar_one()
    assert bootstrapped == rebuilt.live.inventory_digest


async def test_missing_published_generation_is_fatal_and_is_not_recreated(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    del store
    missing = "reembed-missing-generation"
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                id=1,
                vector_table=missing,
                embed_fingerprint=HashEmbedder(dimension=4).fingerprint.model_dump_json(),
                vector_inventory_digest="synthetic-inventory",
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)

    with pytest.raises(VectorStoreStateError, match="does not exist"):
        await live.fingerprint()
    assert not (data_dir / VECTORS_DIRNAME / "generations" / missing).exists()


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    [
        ("id", "wrong-physical-key"),
        ("chunk_id", "wrong-chunk"),
        ("document_id", "wrong-document"),
        ("kind", "heading"),
        ("lang", "zz"),
        ("position", 99),
        ("embed_identity", "wrong-embedding-input"),
        ("source_vector_id", "wrong-source-vector"),
        ("source_publication_id", "wrong-source-publication"),
        ("source_sequence", 99),
    ],
)
async def test_inspection_recomputes_every_retrieval_and_source_identity_column(
    engine: AsyncEngine,
    store: SqliteDocStore,
    data_dir: Path,
    column: str,
    corrupt_value: str | int,
) -> None:
    clock = Clock()
    _, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id=f"corrupt-{column}"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    connection = await lancedb.connect_async(  # pyright: ignore[reportUnknownMemberType]
        str(shadows.directory(generation.id))
    )
    table = await connection.open_table(table_name(HashEmbedder(dimension=4).fingerprint))
    physical_id = vector_id(generation.id, source.chunk.id)
    await table.update(where=f"id = {quote(physical_id)}", updates={column: corrupt_value})

    inspection = await shadows.inspect(generation, lease=lease)

    assert (
        not inspection.lineage_valid
        or inspection.inventory_digest != run.commitment.chunk_inventory_digest
    )


async def test_inspection_pages_are_stably_keyset_bounded(data_dir: Path) -> None:
    document = make_document(source_id="large-shadow").model_copy(
        update={"publication_id": "source-publication"}
    )
    chunks = [make_chunk(document, index, f"chunk {index}") for index in range(513)]
    stored = [
        SnapshotChunk(
            chunk=chunk,
            vector_id=f"source-vector-{index}",
            publication_id=document.publication_id,
            sequence=index,
        )
        for index, chunk in enumerate(chunks)
    ]
    vectors = [[0.0, 1.0, 0.0, 0.0] for _ in stored]
    target = HashEmbedder(dimension=4).fingerprint
    shadow = LanceVectorStore(data_dir / "bounded-shadow")
    await shadow.ensure_ready(target)
    await shadow.upsert_snapshot(stored, vectors, publication_id="reembed-bounded")

    pages = [page async for page in shadow.inspection_pages(page_size=256)]

    assert [len(page) for page in pages] == [256, 256, 1]
    keys = [
        (str(row["document_id"]), int(str(row["position"])), str(row["chunk_id"]))
        for page in pages
        for row in page
    ]
    assert keys == sorted(keys)
    assert len(set(keys)) == 513
    await shadow.teardown()


async def test_new_publication_supersedes_a_prior_publish_before_checkpoint_winner(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, first, lease, generation, source, vector, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="first-publish-before-checkpoint"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    inspection = await shadows.inspect(generation, lease=lease)
    await shadows.seal(generation, inspection, lease=lease)
    first_receipt = await authority.publish(
        first,
        generation,
        expected=first.commitment.snapshot.live,
        expected_corpus_revision=first.commitment.snapshot.revision,
        lease=lease,
    )
    assert first_receipt.outcome is PublishOutcome.PUBLISHED

    second = await start_reembed(
        "second-winner",
        owner_token="owner-b",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        target=HashEmbedder(dimension=4).fingerprint,
        journal=authority,
        lease_ttl_seconds=30.0,
    )
    completed = await resume_reembed(
        second.id,
        owner_token="owner-b",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        embedder=HashEmbedder(dimension=4),
        journal=authority,
        shadow=shadows,
        publisher=authority,
        lease_ttl_seconds=30.0,
    )

    assert completed.state is ReembedState.PUBLISHED
    async with engine.connect() as connection:
        first_state = (
            await connection.execute(
                select(models.ReembedRunRecord.state).where(models.ReembedRunRecord.id == first.id)
            )
        ).scalar_one()
        first_shadow_state = (
            await connection.execute(
                select(models.ReembedShadowGeneration.state).where(
                    models.ReembedShadowGeneration.run_id == first.id
                )
            )
        ).scalar_one()
    assert first_state == ReembedState.SUPERSEDED.value
    assert first_shadow_state == "superseded"
