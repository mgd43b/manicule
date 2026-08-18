"""On-disk recovery and publication evidence for the concrete re-embed adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, override

import lancedb
import pytest
from pydantic import TypeAdapter
from sqlalchemy import delete, func, insert, select, update

from manicule.core.content import Chunk
from manicule.core.embedding import IndexFingerprints, Vector
from manicule.ingest.reembed import (
    CorpusSnapshot,
    PublishOutcome,
    ReembedError,
    ReembedRun,
    ReembedState,
    SnapshotChunk,
    SnapshotDocument,
    plan_reembed_commitment,
    resume_reembed,
    start_reembed,
)
from manicule.ingest.sweeps import sweep_vectors
from manicule.storage import models
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import VECTORS_DIRNAME
from manicule.storage.migrator import downgrade
from manicule.storage.reembed import (
    LanceShadowGenerations,
    SqliteReembedCorpus,
    SqliteReembedStore,
)
from manicule.storage.types import utcnow
from manicule.storage.vectors import (
    LanceVectorStore,
    PublishedLanceVectorStore,
    VectorStoreReprepareRequiredError,
    VectorStoreStateError,
    generation_pin,
    quote,
    table_name,
)
from tests.fakes import HashEmbedder
from tests.storage_helpers import fingerprint, make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.ingest.reembed import ReembedLease, ShadowGeneration
    from manicule.storage.docstore import SqliteDocStore


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FailingSnapshotCorpus(SqliteReembedCorpus):
    def __init__(self, engine: AsyncEngine, failure: BaseException) -> None:
        super().__init__(engine)
        self._failure = failure

    @override
    async def documents(
        self, snapshot: CorpusSnapshot, *, after: str | None, limit: int
    ) -> list[SnapshotDocument]:
        del snapshot, after, limit
        raise self._failure


class BlockingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__(dimension=4)
        self.entered = asyncio.Event()

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.entered.set()
        await asyncio.Event().wait()
        return await super().embed(texts)


class FailingEmbedder(HashEmbedder):
    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        del texts
        raise OSError("synthetic embedding failure")


async def test_bound_publication_cleanup_does_not_follow_a_later_pointer(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The recorded #187 table binding chooses the directory, not today's live pointer."""
    embed = fingerprint()
    root = LanceVectorStore(data_dir / VECTORS_DIRNAME)
    old_generation = LanceVectorStore(data_dir / VECTORS_DIRNAME / "generations" / "reembed-old")
    await root.ensure_ready(embed)
    await old_generation.ensure_ready(embed)
    root_chunk = make_chunk(make_document(source_id="root"), 0, "root")
    old_chunk = make_chunk(make_document(source_id="old"), 0, "old")
    await root.upsert([root_chunk], [[1.0] + [0.0] * 7], publication_id="same-name")
    await old_generation.upsert([old_chunk], [[0.0, 1.0] + [0.0] * 6], publication_id="same-name")
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)

    assert await live.delete_bound_publication(None, "same-name") == 1
    await root.teardown()
    root = LanceVectorStore(data_dir / VECTORS_DIRNAME)
    assert await root.publication_row_count("same-name") == 0
    assert await old_generation.publication_row_count("same-name") == 1

    assert await live.delete_bound_publication("reembed-old", "same-name") == 1
    await old_generation.teardown()
    old_generation = LanceVectorStore(data_dir / VECTORS_DIRNAME / "generations" / "reembed-old")
    assert await old_generation.publication_row_count("same-name") == 0


class RefusingCreateJournal:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure

    async def create_released(self, run_id: str, commitment: object) -> ReembedRun:
        del run_id, commitment
        raise self.failure


async def test_snapshots_runs_and_counts_are_workspace_scoped(  # noqa: PLR0915
    engine: AsyncEngine, data_dir: Path
) -> None:
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()
    stored_chunks: list[Chunk] = []
    for scoped, workspace, source_id in ((alpha, "alpha", "a"), (beta, "beta", "b")):
        document = make_document(source_id=source_id, workspace_id=workspace)
        chunk = make_chunk(document, 0, workspace)
        await scoped.upsert_document(document)
        await scoped.replace_chunks(document.id, [chunk])
        stored_chunks.append(chunk)
    old = fingerprint(dimension=4, model_id="old/model")
    old_vectors = LanceVectorStore(data_dir / VECTORS_DIRNAME)
    await old_vectors.ensure_ready(old)
    await old_vectors.upsert(
        stored_chunks,
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    await old_vectors.teardown()
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                [
                    {
                        "workspace_id": workspace,
                        "vector_namespace": "legacy",
                        "vector_table": table_name(old),
                        "embed_fingerprint": old.model_dump_json(),
                        "created_at": utcnow(),
                        "updated_at": utcnow(),
                    }
                    for workspace in ("alpha", "beta")
                ]
            )
        )

    target = HashEmbedder(dimension=4).fingerprint
    alpha_corpus = SqliteReembedCorpus(engine, "alpha")
    beta_corpus = SqliteReembedCorpus(engine, "beta")
    alpha_commitment = await plan_reembed_commitment(alpha_corpus, target)
    beta_commitment = await plan_reembed_commitment(beta_corpus, target)
    assert alpha_commitment.plan.documents == alpha_commitment.plan.chunks == 1
    assert beta_commitment.plan.documents == beta_commitment.plan.chunks == 1
    assert alpha_commitment.execution_plan.documents == 1
    assert alpha_commitment.execution_plan.chunks == 1
    assert beta_commitment.execution_plan.documents == 1
    assert beta_commitment.execution_plan.chunks == 1
    assert alpha_commitment.snapshot.workspace_id == "alpha"
    assert beta_commitment.snapshot.workspace_id == "beta"

    alpha_store = SqliteReembedStore(engine, "alpha")
    beta_store = SqliteReembedStore(engine, "beta")
    alpha_run = await alpha_store.create_released("same-run-id", alpha_commitment)
    beta_run = await beta_store.create_released("same-run-id", beta_commitment)
    assert alpha_run.workspace_id == "alpha"
    assert beta_run.workspace_id == "beta"
    assert await alpha_store.get("beta-only") is None
    beta_only = await beta_store.create_released("beta-only", beta_commitment)
    assert beta_only.workspace_id == "beta"
    assert await alpha_store.get("beta-only") is None

    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                select(
                    models.ReembedRunRecord.workspace_id,
                    func.count(models.ReembedRunRecord.id),
                ).group_by(models.ReembedRunRecord.workspace_id)
            )
        ).all()
        counts = {str(row[0]): int(row[1]) for row in rows}
    assert counts == {"alpha": 1, "beta": 2}

    beta_shadows = LanceShadowGenerations(data_dir / VECTORS_DIRNAME, beta_store)
    beta_published = await resume_reembed(
        beta_run.id,
        owner_token="beta-owner",  # noqa: S106
        corpus=beta_corpus,
        embedder=HashEmbedder(dimension=4),
        journal=beta_store,
        shadow=beta_shadows,
        publisher=beta_store,
    )
    assert beta_published.state is ReembedState.PUBLISHED
    beta_document = await beta.get_document(stored_chunks[1].document_id)
    assert beta_document is not None
    await beta.upsert_document(beta_document.model_copy(update={"title": "beta changed"}))
    async with engine.connect() as connection:
        revision_rows = (
            await connection.execute(
                select(
                    models.CorpusRevision.workspace_id,
                    models.CorpusRevision.revision,
                )
            )
        ).all()
        revisions = {str(row.workspace_id): int(row.revision) for row in revision_rows}
    assert str(revisions["alpha"]) == alpha_commitment.snapshot.revision
    assert str(revisions["beta"]) != beta_commitment.snapshot.revision

    alpha_shadows = LanceShadowGenerations(data_dir / VECTORS_DIRNAME, alpha_store)
    alpha_published = await resume_reembed(
        alpha_run.id,
        owner_token="alpha-owner",  # noqa: S106
        corpus=alpha_corpus,
        embedder=HashEmbedder(dimension=4),
        journal=alpha_store,
        shadow=alpha_shadows,
        publisher=alpha_store,
    )
    assert alpha_published.state is ReembedState.PUBLISHED
    assert (await beta_store.get(beta_run.id)).state is ReembedState.PUBLISHED  # type: ignore[union-attr]
    async with engine.connect() as connection:
        beta_shadow_state = (
            await connection.execute(
                select(models.ReembedShadowGeneration.state).where(
                    models.ReembedShadowGeneration.workspace_id == "beta",
                    models.ReembedShadowGeneration.run_id == beta_run.id,
                )
            )
        ).scalar_one()
    assert beta_shadow_state == "published"

    stale_alpha = await start_reembed(
        "alpha-mutated-after-snapshot",
        owner_token="alpha-stale-owner",  # noqa: S106
        corpus=alpha_corpus,
        target=target,
        journal=alpha_store,
    )
    alpha_document = await alpha.get_document(stored_chunks[0].document_id)
    assert alpha_document is not None
    await alpha.upsert_document(alpha_document.model_copy(update={"title": "alpha changed"}))
    stale_result = await resume_reembed(
        stale_alpha.id,
        owner_token="alpha-stale-owner",  # noqa: S106
        corpus=alpha_corpus,
        embedder=HashEmbedder(dimension=4),
        journal=alpha_store,
        shadow=alpha_shadows,
        publisher=alpha_store,
    )
    assert stale_result.state is ReembedState.SUPERSEDED

    live = PublishedLanceVectorStore(
        data_dir / VECTORS_DIRNAME,
        engine,
        workspace_id="alpha",
        identity_namespace="legacy",
    )
    await live.ensure_ready(HashEmbedder(dimension=4).fingerprint)
    assert await live.count() == 1
    assert [row.chunk.document_id for row in await live.search([1.0, 0.0, 0.0, 0.0], 10)] == [
        stored_chunks[0].document_id
    ]
    assert await alpha.get_document(stored_chunks[0].document_id) is not None
    assert await beta.get_document(stored_chunks[1].document_id) is not None
    await live.teardown()


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
                workspace_id="default",
                vector_namespace="legacy",
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


@pytest.mark.parametrize(
    "failure",
    [OSError("synthetic snapshot scan failure"), asyncio.CancelledError()],
    ids=["io-error", "cancellation"],
)
async def test_failed_or_canceled_plan_removes_every_private_snapshot_row(
    engine: AsyncEngine, store: SqliteDocStore, failure: BaseException
) -> None:
    await store.ensure_workspace()
    old = fingerprint(dimension=4, model_id="old/model")
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                workspace_id="default",
                vector_namespace="legacy",
                vector_table=table_name(old),
                embed_fingerprint=old.model_dump_json(),
                vector_inventory_digest=None,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
    corpus = FailingSnapshotCorpus(engine, failure)

    with pytest.raises(type(failure)):
        await plan_reembed_commitment(corpus, HashEmbedder(dimension=4).fingerprint)

    async with engine.connect() as connection:
        counts = [
            (await connection.execute(select(func.count()).select_from(model))).scalar_one()
            for model in (
                models.ReembedCorpusSnapshot,
                models.ReembedSnapshotDocument,
                models.ReembedSnapshotChunk,
            )
        ]
    assert counts == [0, 0, 0]
    await downgrade(engine, "6e31b7d592ac")


@pytest.mark.parametrize(
    "failure",
    [OSError("synthetic create refusal"), asyncio.CancelledError()],
    ids=["refusal", "cancellation"],
)
async def test_start_cleans_its_self_created_snapshot_when_run_creation_does_not_commit(
    engine: AsyncEngine, store: SqliteDocStore, failure: BaseException
) -> None:
    old = fingerprint(dimension=4, model_id="old/model")
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                workspace_id="default",
                vector_namespace="legacy",
                vector_table=table_name(old),
                embed_fingerprint=old.model_dump_json(),
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "local")])
    corpus = SqliteReembedCorpus(engine)

    with pytest.raises(type(failure)):
        await start_reembed(
            "not-created",
            owner_token="synthetic-owner",  # noqa: S106
            corpus=corpus,
            target=HashEmbedder(dimension=4).fingerprint,
            journal=RefusingCreateJournal(failure),  # type: ignore[arg-type]
        )

    async with engine.connect() as connection:
        assert (
            await connection.execute(select(func.count()).select_from(models.ReembedCorpusSnapshot))
        ).scalar_one() == 0


async def test_canceled_resume_releases_fence_for_immediate_takeover_and_abandon(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, _, _, _, _, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="cancel-release"
    )
    embedder = BlockingEmbedder()
    task = asyncio.create_task(
        resume_reembed(
            run.id,
            owner_token="owner-a",  # noqa: S106 - synthetic fence identity
            corpus=corpus,
            embedder=embedder,
            journal=authority,
            shadow=shadows,
            publisher=authority,
        )
    )
    await asyncio.wait_for(embedder.entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    takeover = await authority.acquire(run.id, "owner-b", ttl_seconds=30.0)
    abandoned = await authority.abandon(run.id, lease=takeover)
    assert abandoned.state is ReembedState.FAILED
    assert await authority.live_generation_id() == run.commitment.snapshot.live.generation_id


async def test_failed_resume_releases_fence_for_an_immediate_retry(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, _, _, _, _, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="failure-release"
    )

    with pytest.raises(OSError, match="synthetic embedding failure"):
        await resume_reembed(
            run.id,
            owner_token="owner-a",  # noqa: S106 - synthetic fence identity
            corpus=corpus,
            embedder=FailingEmbedder(dimension=4),
            journal=authority,
            shadow=shadows,
            publisher=authority,
        )

    takeover = await authority.acquire(run.id, "owner-b", ttl_seconds=30.0)
    await authority.assert_current(run.id, takeover)


async def test_on_disk_shadow_validates_publishes_and_runtime_follows_pointer(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="durable-handoff"
    )
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)
    await live.ensure_ready(fingerprint(dimension=4, model_id="old/model"))
    assert (await live.search([1.0, 0.0, 0.0, 0.0], 1))[0].publication_id == source.publication_id

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
                select(models.CorpusRevision.revision).where(
                    models.CorpusRevision.workspace_id == "default"
                )
            )
        ).scalar_one()
    assert str(revision) == run.commitment.snapshot.revision
    with pytest.raises(ReembedError, match="sealed shadow generation"):
        await shadows.upsert(generation, [source], [vector], lease=lease)
    with pytest.raises(VectorStoreReprepareRequiredError, match="prepare this handle"):
        await live.search(vector, 1)
    with pytest.raises(VectorStoreReprepareRequiredError, match="prepare this handle"):
        await live.upsert([source.chunk], [vector], publication_id=source.publication_id)
    await live.ensure_ready(HashEmbedder(dimension=4).fingerprint)
    await live.upsert([source.chunk], [vector], publication_id=source.publication_id)
    rows = await live.search(vector, 1)
    assert len(rows) == 1
    assert rows[0].publication_id == source.publication_id
    assert (
        await store.get_document(source.chunk.document_id)
    ).publication_id == source.publication_id  # type: ignore[union-attr]
    await store.replace_chunks(source.chunk.document_id, [])
    assert source.vector_id in await store.take_tombstones(10)
    swept = await sweep_vectors(store, live)
    assert swept.vectors_removed == 1
    assert await live.count() == 0
    await live.teardown()


async def test_lease_release_is_fenced_and_allows_immediate_worker_handoff(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, _, run, lease_a, _, _, _, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="release-fence"
    )
    clock.advance(31.0)
    lease_b = await authority.acquire(run.id, "owner-b", ttl_seconds=30.0)

    with pytest.raises(ReembedError, match="stale or expired"):
        await authority.release(run.id, lease_a)
    await authority.assert_current(run.id, lease_b)

    await authority.release(run.id, lease_b)
    lease_c = await authority.acquire(run.id, "owner-c", ttl_seconds=30.0)
    assert lease_c.generation > lease_b.generation


async def test_a_snapshot_bound_to_a_durable_run_cannot_be_discarded(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    _, _, run, _, _, _, _, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="bound-snapshot"
    )

    with pytest.raises(ReembedError, match="bound to a durable run"):
        await corpus.discard_snapshot(run.commitment.snapshot.id)


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
            .where(models.IndexState.workspace_id == "default")
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


async def test_publish_receipt_is_atomic_and_cannot_be_downgraded_or_replayed(
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
    terminal = await authority.get(run.id)
    assert terminal is not None
    assert terminal.state is ReembedState.PUBLISHED
    stale = replace(terminal, state=ReembedState.READY, receipt=None)
    async with engine.begin() as connection:
        await connection.execute(
            update(models.ReembedRunRecord)
            .where(models.ReembedRunRecord.id == run.id)
            .values(
                state=ReembedState.READY.value,
                checkpoint_json=TypeAdapter(ReembedRun).dump_json(stale).decode("utf-8"),
            )
        )
    reconciled = await authority.get(run.id)
    assert reconciled is not None
    assert reconciled.state is ReembedState.PUBLISHED
    assert reconciled.receipt == first
    resumed = await resume_reembed(
        run.id,
        owner_token="retry-owner",  # noqa: S106 - synthetic fence identity
        corpus=SqliteReembedCorpus(engine),
        embedder=HashEmbedder(dimension=4),
        journal=authority,
        shadow=shadows,
        publisher=authority,
    )
    assert resumed.state is ReembedState.PUBLISHED
    with pytest.raises(ReembedError, match="terminal publication decision"):
        await authority.abandon(run.id, lease=lease)
    with pytest.raises(ReembedError, match="receipt cannot be overwritten"):
        await authority.save(
            replace(run, state=ReembedState.FAILED, failure="stale downgrade"),
            expected_revision=run.revision,
            lease=lease,
        )
    async with engine.begin() as connection:
        await connection.execute(
            update(models.IndexState)
            .where(models.IndexState.workspace_id == "default")
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
                select(models.IndexState.vector_table).where(
                    models.IndexState.workspace_id == "default"
                )
            )
        ).scalar_one() == "reembed-competing-winner"

    with pytest.raises(ReembedError, match="only failed or superseded"):
        await shadows.cleanup_terminal(run.id)


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


async def test_shared_legacy_generation_cleanup_refuses_a_foreign_live_pointer(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, run, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="shared-legacy-cleanup"
    )
    await shadows.upsert(generation, [source], [vector], lease=lease)
    abandoned = await authority.abandon(run.id, lease=lease)
    assert abandoned.state is ReembedState.FAILED

    beta = SqliteDocStore(engine, workspace_id="beta")
    await beta.ensure_workspace()
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                workspace_id="beta",
                vector_namespace="legacy",
                vector_table=generation.id,
                embed_fingerprint=HashEmbedder(dimension=4).fingerprint.model_dump_json(),
                vector_inventory_digest=generation.inventory_digest,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )

    with pytest.raises(ReembedError, match=r"foreign workspace.*shared legacy generation"):
        await shadows.cleanup_terminal(run.id)
    assert shadows.directory(generation.id).exists()

    async with engine.begin() as connection:
        await connection.execute(
            update(models.IndexState)
            .where(models.IndexState.workspace_id == "beta")
            .values(vector_table=None, vector_inventory_digest=None)
        )
    assert await shadows.cleanup_terminal(run.id)
    assert not shadows.directory(generation.id).exists()


@pytest.mark.parametrize(
    "snapshot_change",
    [
        {"complete": False},
        {"inventory_digest": "same-count-wrong-document-inventory"},
        {"chunk_inventory_digest": "same-count-wrong-chunk-inventory"},
    ],
)
async def test_publication_requires_the_complete_digest_bound_snapshot_record(
    engine: AsyncEngine,
    store: SqliteDocStore,
    data_dir: Path,
    snapshot_change: dict[str, object],
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
            .values(**snapshot_change)
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
                select(models.IndexState.vector_inventory_digest).where(
                    models.IndexState.workspace_id == "default"
                )
            )
        ).scalar_one()
    assert before == run.commitment.snapshot.live.inventory_digest

    await store.replace_chunks(source.chunk.document_id, [source.chunk])
    async with engine.connect() as connection:
        invalidated = (
            await connection.execute(
                select(models.IndexState.vector_inventory_digest).where(
                    models.IndexState.workspace_id == "default"
                )
            )
        ).scalar_one()
    assert invalidated is None

    rebuilt = await corpus.begin_snapshot()
    async with engine.connect() as connection:
        bootstrapped = (
            await connection.execute(
                select(models.IndexState.vector_inventory_digest).where(
                    models.IndexState.workspace_id == "default"
                )
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
                workspace_id="default",
                vector_namespace="legacy",
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


async def test_published_handle_keeps_rebuild_namespaces_inside_one_pinned_pointer(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    del store
    embed = fingerprint(dimension=4)
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)
    await live.ensure_ready(embed)
    original = make_chunk(make_document(), 0, "rebuild vector")
    await live.upsert(
        [original],
        [[1.0, 0.0, 0.0, 0.0]],
        publication_id="rebuild-source",
    )

    assert await live.publication_is_complete(
        "rebuild-source", [original], embedding_fingerprint=embed.canonical()
    )
    await live.copy_publication("rebuild-source", "rebuild-takeover", [original])
    assert await live.publication_row_count("rebuild-takeover") == 1
    assert await live.publication_page_is_complete(
        "rebuild-takeover", [original], embedding_fingerprint=embed.canonical()
    )
    await live.teardown()


@pytest.mark.parametrize(
    "pointer",
    ["reembed-../outside", "reembed-private/generation", r"reembed-private\generation"],
)
async def test_invalid_published_generation_pointer_never_becomes_a_path(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path, pointer: str
) -> None:
    del store
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                workspace_id="default",
                vector_namespace="legacy",
                vector_table=pointer,
                embed_fingerprint=HashEmbedder(dimension=4).fingerprint.model_dump_json(),
                vector_inventory_digest="synthetic-inventory",
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
    live = PublishedLanceVectorStore(data_dir / VECTORS_DIRNAME, engine)

    with pytest.raises(VectorStoreStateError, match="generation pointer is invalid"):
        await live.fingerprint()

    assert not (data_dir / VECTORS_DIRNAME / "outside").exists()


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
    physical_id = source.vector_id
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

    connection = await lancedb.connect_async(  # pyright: ignore[reportUnknownMemberType]
        str(data_dir / "bounded-shadow")
    )
    table = await connection.open_table(table_name(target))
    await table.update(updates={"document_id": "duplicate", "position": 0, "chunk_id": "duplicate"})
    duplicate_pages = [page async for page in shadow.inspection_pages(page_size=256)]
    assert [len(page) for page in duplicate_pages] == [256, 256, 1]
    assert len({str(row["id"]) for page in duplicate_pages for row in page}) == 513
    await shadow.teardown()


async def test_inspection_does_not_skip_more_than_a_page_of_duplicate_logical_keys(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    _, shadows, _, lease, generation, source, vector, _ = await seeded_run(
        engine, store, data_dir, clock, run_id="duplicate-logical-rows"
    )
    corrupt = LanceVectorStore(shadows.directory(generation.id))
    await corrupt.open_existing(HashEmbedder(dimension=4).fingerprint)
    duplicates = [
        replace(source, vector_id=f"duplicate-physical-{index}", sequence=index)
        for index in range(300)
    ]
    await corrupt.upsert_snapshot(
        duplicates,
        [vector for _ in duplicates],
        publication_id=generation.id,
    )
    connection = await lancedb.connect_async(  # pyright: ignore[reportUnknownMemberType]
        str(shadows.directory(generation.id))
    )
    table = await connection.open_table(table_name(HashEmbedder(dimension=4).fingerprint))
    await table.update(
        updates={
            "id": "duplicate-physical",
            "document_id": "duplicate",
            "position": 0,
            "chunk_id": "duplicate",
        }
    )

    inspection = await shadows.inspect(generation, lease=lease)

    assert inspection.rows == 300
    assert inspection.unique_chunks == 0
    assert not inspection.lineage_valid
    await corrupt.teardown()


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


async def test_cleanup_waits_for_an_in_flight_search_pinned_to_the_old_generation(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    clock = Clock()
    authority, shadows, first, _, _, source, vector, corpus = await seeded_run(
        engine, store, data_dir, clock, run_id="pinned-old-reader"
    )
    embedder = HashEmbedder(dimension=4)
    first = await resume_reembed(
        first.id,
        owner_token="owner-a",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        embedder=embedder,
        journal=authority,
        shadow=shadows,
        publisher=authority,
        lease_ttl_seconds=30.0,
    )
    assert first.shadow_generation_id is not None

    entered = asyncio.Event()
    release = asyncio.Event()
    blocking = False

    async def operation_barrier() -> None:
        if blocking:
            entered.set()
            await release.wait()

    live = PublishedLanceVectorStore(
        data_dir / VECTORS_DIRNAME,
        engine,
        operation_hook=operation_barrier,
    )
    await live.ensure_ready(embedder.fingerprint)
    blocking = True
    searching = asyncio.create_task(live.search(vector, 1))
    await asyncio.wait_for(entered.wait(), timeout=2.0)

    second = await start_reembed(
        "new-winner-over-pinned-reader",
        owner_token="owner-b",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        target=embedder.fingerprint,
        journal=authority,
        lease_ttl_seconds=30.0,
    )
    second = await resume_reembed(
        second.id,
        owner_token="owner-b",  # noqa: S106 - synthetic lease identity, not a credential
        corpus=corpus,
        embedder=embedder,
        journal=authority,
        shadow=shadows,
        publisher=authority,
        lease_ttl_seconds=30.0,
    )
    assert second.state is ReembedState.PUBLISHED

    cleanup = asyncio.create_task(shadows.cleanup_terminal(first.id))
    await asyncio.sleep(0.05)
    assert not cleanup.done(), "cleanup must wait on the old reader's shared generation pin"
    release.set()
    rows = await asyncio.wait_for(searching, timeout=2.0)
    assert [row.chunk.id for row in rows] == [source.chunk.id]
    assert await asyncio.wait_for(cleanup, timeout=2.0)
    assert not shadows.directory(first.shadow_generation_id).exists()
    await live.teardown()


async def test_writer_queued_behind_reset_revalidates_durable_workspace_identity(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    """A pre-reset handle cannot recreate its old Lance store after reset releases the pin."""
    embed = fingerprint(dimension=4)
    await store.record_index_fingerprints(IndexFingerprints(embed=embed))
    directory = data_dir / VECTORS_DIRNAME
    live = PublishedLanceVectorStore(
        directory,
        engine,
        identity_namespace="workspace",
    )
    await live.ensure_ready(embed)
    source = make_document()
    stored = make_chunk(source, 0, "queued stale write")
    held = asyncio.Event()
    release = asyncio.Event()

    async def hold_reset_pin() -> None:
        async with generation_pin(directory, exclusive=True):
            held.set()
            await release.wait()

    holder = asyncio.create_task(hold_reset_pin())
    await asyncio.wait_for(held.wait(), timeout=2.0)
    writing = asyncio.create_task(live.upsert([stored], [[0.1, 0.2, 0.3, 0.4]]))
    await asyncio.sleep(0.05)
    assert not writing.done(), "the stale writer did not wait for reset's exclusive pin"
    async with engine.begin() as connection:
        await connection.execute(
            delete(models.IndexState).where(models.IndexState.workspace_id == "default")
        )
    release.set()
    await asyncio.wait_for(holder, timeout=2.0)

    with pytest.raises(VectorStoreReprepareRequiredError, match="identity was reset"):
        await asyncio.wait_for(writing, timeout=2.0)
    physical = LanceVectorStore(directory)
    await physical.open_existing(embed)
    assert await physical.count() == 0
    await physical.teardown()
    await live.teardown()
