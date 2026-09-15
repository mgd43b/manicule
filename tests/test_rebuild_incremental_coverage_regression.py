"""A replacement corpus may only be derived from evidence that names the whole corpus.

Acquisition answers two questions that had been collapsed into one. ``completeness`` asks
whether the bodies of the members a run enumerated are held locally; ``enumeration_membership``
asks whether that run enumerated the connector's whole membership or only what had changed
since a cursor. A one-document incremental manifest whose single body was retained answers
"complete" to the first and proves nothing at all about the second, and planning used to read
it as authority over everything the connector held.

Every fixture here is synthetic and every URI is under ``example.test``. Nothing in this module
contacts a source, downloads a model, or starts a browser.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event, select, update

from manicule.core.acquisition import SnapshotMembership
from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, DocumentStatus, RawDocument
from manicule.core.ids import chunk_id, document_id
from manicule.core.rebuild import (
    DerivedReplacement,
    RebuildPublicationConflictError,
    RebuildRefusalCode,
    RebuildState,
)
from manicule.storage import models
from manicule.storage.blobs import BlobStore
from manicule.storage.engine import session_factory
from manicule.storage.rebuild import SqliteRebuildStore
from manicule.storage.vectors import LanceVectorStore
from tests.storage_helpers import make_document
from tests.test_storage_rebuild import (
    NOW,
    promoted_snapshot,
    promoted_snapshot_many,
    rebuild_target,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.rebuild import RebuildCheckpoint, RebuildTarget
    from manicule.storage.docstore import SqliteDocStore

LETTERS = ("a", "b", "c")
"""Three documents: one a later delta changes, and two it has no reason to mention."""


def wiki_raws() -> tuple[RawDocument, ...]:
    return tuple(
        RawDocument(
            source_id=f"document-{letter}",
            uri=f"https://example.test/wiki/{letter}",
            media_type="text/plain",
            content=f"Synthetic document {letter}",
        )
        for letter in LETTERS
    )


async def live_full_inventory(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    *,
    connector: str = "wiki",
    run_id: str = "promoted-glossary-run",
    scope_fingerprint: str = "glossary-v1",
) -> tuple[str, tuple[RawDocument, ...]]:
    """A promoted full inventory of three documents, all three of them live."""
    raws = wiki_raws()
    run, _ = await promoted_snapshot_many(
        store,
        engine,
        data_dir,
        raws,
        run_id=run_id,
        connector=connector,
        scope_fingerprint=scope_fingerprint,
    )
    for raw in raws:
        await store.upsert_document(
            make_document(
                source=connector,
                source_id=raw.source_id,
                uri=raw.uri,
                media_type=raw.media_type,
                body=raw.as_bytes(),
            )
        )
    return run, raws


async def promoted_delta(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    *,
    source_id: str,
    connector: str = "wiki",
    run_id: str = "zz-incremental-example",
    scope_fingerprint: str = "glossary-v1",
) -> str:
    """One promoted run that inherits the committed cursor, so it enumerates a delta."""
    delta, _, _ = await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id=run_id,
        connector=connector,
        scope_fingerprint=scope_fingerprint,
        source_id=source_id,
    )
    return delta


def rebuild_store(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path, vectors: LanceVectorStore
) -> SqliteRebuildStore:
    return SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )


async def ready_vectors(data_dir: Path) -> tuple[LanceVectorStore, RebuildTarget]:
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    return vectors, target


async def live_source_ids(engine: AsyncEngine, connector: str = "wiki") -> set[str]:
    async with session_factory(engine)() as session:
        return set(
            (
                await session.execute(
                    select(models.Document.source_id).where(
                        models.Document.source == connector,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )


async def publish_plan(
    rebuilds: SqliteRebuildStore,
    vectors: LanceVectorStore,
    blobs: BlobStore,
    *,
    generation_id: str,
    workspace_id: str,
    owner: str,
) -> RebuildCheckpoint:
    """Derive and publish every input the plan feeds the builder, in one generation.

    Deliberately driven from :meth:`snapshot_inputs` rather than from the fixture's own list:
    what publication must cover is what planning bound, and a test that built its documents
    from the fixture would pass even if the two had diverged.
    """
    claimed = await rebuilds.claim_generation(
        generation_id,
        owner,
        now=NOW,
        # Validation checkpoints fence against a live clock rather than ``NOW``, so the lease
        # has to outlive real elapsed time and not just the fixed instant the fixture uses.
        expires_at=NOW + timedelta(days=36500),
    )
    inputs = await rebuilds.snapshot_inputs(generation_id, after_sequence=-1, limit=100)
    assert inputs, "a runnable plan must feed the builder something"
    staged: list[tuple[int, DerivedReplacement]] = []
    chunks: list[Chunk] = []
    for item in inputs:
        body = await blobs.get(item.blob_ref)
        assert body is not None, "planning proved these bytes are retained"
        text = body.decode()
        document = make_document(
            source=item.connector,
            source_id=item.source.source_id,
            uri=item.source.uri,
            media_type=item.source.media_type or "text/plain",
            body=body,
        ).model_copy(
            update={
                "id": document_id(workspace_id, item.connector, item.source.source_id),
                "publication_id": generation_id,
                "original_ref": item.blob_ref,
                "version_token": item.version_token,
                "status": DocumentStatus.INDEXED,
            }
        )
        chunk = Chunk(
            id=chunk_id(document.id, 0, text),
            document_id=document.id,
            text=text,
            embed_text=text,
            anchor=Unlocated(reason="plain text"),
            position=0,
            token_count=3,
        )
        chunks.append(chunk)
        staged.append(
            (
                item.sequence,
                DerivedReplacement(
                    document=document,
                    chunks=(chunk,),
                    parse_fingerprint="plain@2",
                    vector_embedded=1,
                ),
            )
        )
    await vectors.upsert(
        chunks,
        [[1.0, 0.0, 0.0, 0.0] for _ in chunks],
        publication_id=claimed.vector_publication_id,
    )
    await rebuilds.stage_replacements(
        generation_id,
        staged,
        expected_next_sequence=0,
        owner=owner,
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.begin_validation(
        generation_id, owner=owner, lease_generation=claimed.lease_generation, now=NOW
    )
    await rebuilds.validate_generation(
        generation_id, owner=owner, lease_generation=claimed.lease_generation, now=NOW
    )
    return await rebuilds.publish_generation(
        generation_id, owner=owner, lease_generation=claimed.lease_generation, now=NOW
    )


async def test_incremental_plan_must_cover_unchanged_live_documents_or_refuse(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    raws = tuple(
        RawDocument(
            source_id=f"document-{letter}",
            uri=f"https://example.test/wiki/{letter}",
            media_type="text/plain",
            content=f"Synthetic document {letter}",
        )
        for letter in ("a", "b", "c")
    )
    initial_run, _ = await promoted_snapshot_many(store, engine, data_dir, raws)
    for raw in raws:
        await store.upsert_document(
            make_document(
                source="wiki",
                source_id=raw.source_id,
                uri=raw.uri,
                media_type=raw.media_type,
                body=raw.as_bytes(),
            )
        )
    delta_run, _, _ = await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id="zz-incremental-example",
        scope_fingerprint="glossary-v1",
        source_id=raws[0].source_id,
    )
    async with session_factory(engine)() as session:
        initial = await session.get(models.AcquisitionRun, initial_run)
        delta = await session.get(models.AcquisitionRun, delta_run)
        assert initial is not None
        assert delta is not None
        assert delta.base_watermark is not None
        assert delta.base_watermark == initial.candidate_watermark

    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(delta_run, target, missing_limit=10, persist=False)
    assert not plan.runnable or plan.documents == 3, (
        f"unsafe replacement: runnable={plan.runnable}, covered={plan.documents}, expected=3"
    )


async def test_a_promoted_run_records_whether_it_walked_the_scope_or_resumed_a_cursor(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The fact the two completeness questions were collapsed into, made durable.

    Both runs are byte-complete. Only one of them enumerated the connector's membership, and
    nothing about either run's ``completeness`` says which.
    """
    initial, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)

    async with session_factory(engine)() as session:
        first = await session.get(models.AcquisitionRun, initial)
        second = await session.get(models.AcquisitionRun, delta)
    assert first is not None
    assert second is not None
    assert first.enumeration_membership is SnapshotMembership.FULL_INVENTORY
    assert second.enumeration_membership is SnapshotMembership.INCREMENTAL
    assert first.completeness == second.completeness, (
        "byte completeness cannot tell these two apart, which is the whole defect"
    )


async def test_an_incremental_plan_composes_the_full_inventory_behind_it(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The refusal is the safety fix; this is the migration actually completing.

    A grammar change makes every stored chunk stale, and the only supported repair is an
    offline rebuild from retained bytes. If the newest promoted run is a delta and planning
    could only ever refuse it, the operator's corpus would be permanently unmigratable without
    re-downloading every body. Reaching back through the committed cursor to the full
    inventory is what turns the refusal into a path.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10, persist=False)

    assert plan.runnable
    assert plan.documents == len(LETTERS)
    assert (plan.live_documents, plan.covered_documents, plan.uncovered_documents) == (3, 3, 0)
    inputs = await rebuilds.plan_rebuild(delta, target, missing_limit=10)
    fed = await rebuilds.snapshot_inputs(inputs.generation_id, after_sequence=-1, limit=100)
    assert {item.source.source_id for item in fed} == {raw.source_id for raw in raws}
    changed = next(item for item in fed if item.source.source_id == raws[0].source_id)
    assert changed.source.uri == "https://wiki.example.test/content/document-a", (
        "the newest run that named a document owns it, because it holds its current bytes"
    )


async def test_a_delta_with_no_reachable_full_inventory_refuses_and_changes_nothing(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Fail closed, and leave the corpus exactly as it was found."""
    initial, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    async with session_factory(engine).begin() as session:
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id == initial)
            .values(superseded_at=NOW)
        )
    before = await live_source_ids(engine)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10, persist=False)

    assert not plan.runnable
    assert plan.refusal is RebuildRefusalCode.INCOMPLETE_SOURCE_INVENTORY
    assert (plan.live_documents, plan.covered_documents, plan.uncovered_documents) == (3, 1, 2)
    assert plan.documents == 0, "a refusal prices nothing"
    assert await live_source_ids(engine) == before, "a refused plan is not a mutation"
    async with session_factory(engine)() as session:
        generations = (await session.execute(select(models.DerivedGeneration.id))).scalars().all()
    assert not generations, "a refusal must not leave a generation behind to be claimed"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("inventory_state", models.AcquisitionInventoryState.REENUMERATION_REQUIRED),
        ("candidate_watermark", None),
    ],
)
async def test_a_chain_link_that_does_not_connect_is_not_a_chain(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path, column: str, value: object
) -> None:
    """Continuity is the committed cursor, not adjacency in time.

    A predecessor whose inventory was invalidated by a source deletion, or whose watermark is
    not the one the delta resumed from, does not account for what the delta left out — and a
    chain that does not connect must read as no chain at all rather than as a near miss.
    """
    initial, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    async with session_factory(engine).begin() as session:
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id == initial)
            .values({column: value})
        )
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10, persist=False)

    assert plan.refusal is RebuildRefusalCode.INCOMPLETE_SOURCE_INVENTORY
    assert plan.uncovered_documents == 2


async def test_an_empty_delta_still_covers_the_inventory_behind_it(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """A sync that found nothing changed is the most common delta there is."""
    _, raws = await live_full_inventory(store, engine, data_dir)
    run = await store.create_acquisition_run(
        "zz-empty-delta",
        "wiki",
        source_scope="scope:glossary-v1",
        scope_fingerprint="glossary-v1",
    )
    claimed = await store.claim_acquisition_run(
        run.id, "worker", now=NOW, expires_at=NOW + timedelta(minutes=5)
    )
    assert claimed is not None
    from manicule.core.sources import Watermark  # noqa: PLC0415 - one local fixture needs it

    await store.complete_acquisition_enumeration(
        run.id,
        Watermark(value="v3", observed_at=NOW),
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await store.complete_snapshot_acquisition(
        run.id, lease_owner="worker", lease_generation=claimed.lease_generation, now=NOW
    )
    await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="glossary-v1",
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(run.id, target, missing_limit=10, persist=False)

    assert plan.runnable
    assert plan.documents == len(raws)
    assert (plan.live_documents, plan.covered_documents) == (3, 3)


async def test_a_rebuild_does_not_reinstate_a_document_the_corpus_has_retired(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Reaching back through history must not reach past a deletion.

    The full inventory still names the retired document — it was acquired before the removal,
    and immutable source history is not rewritten. Composing it with a later delta would
    otherwise republish exactly the document the corpus decided to stop serving.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    retired = raws[1]
    await store.soft_delete_document(document_id(store.workspace_id, "wiki", retired.source_id))
    await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(
        "zz-incremental-example", target, missing_limit=10, persist=True
    )

    assert plan.runnable
    assert plan.documents == 2
    assert (plan.live_documents, plan.uncovered_documents) == (2, 0)
    fed = await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=100)
    assert retired.source_id not in {item.source.source_id for item in fed}


async def test_two_connectors_with_different_histories_are_proven_separately(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """One connector's unbroken history is not evidence about another's."""
    _, raws = await live_full_inventory(store, engine, data_dir)
    await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    drive_raws = tuple(
        RawDocument(
            source_id=f"drive-{letter}",
            uri=f"https://example.test/drive/{letter}",
            media_type="text/plain",
            content=f"Synthetic drive document {letter}",
        )
        for letter in ("a", "b")
    )
    drive_run, _ = await promoted_snapshot_many(
        store,
        engine,
        data_dir,
        drive_raws,
        run_id="drive-full-run",
        connector="drive",
        scope_fingerprint="drive-v1",
    )
    for raw in drive_raws:
        await store.upsert_document(
            make_document(
                source="drive",
                source_id=raw.source_id,
                uri=raw.uri,
                media_type=raw.media_type,
                body=raw.as_bytes(),
            )
        )
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(drive_run, target, missing_limit=10, persist=False)

    assert plan.runnable
    assert plan.documents == len(raws) + len(drive_raws)
    assert (plan.live_documents, plan.uncovered_documents) == (5, 0)


async def test_a_missing_retained_body_in_the_composed_inventory_is_still_a_missing_input(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Coverage and retention are separate refusals, and reaching back does not merge them."""
    _, raws = await live_full_inventory(store, engine, data_dir)
    await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    async with session_factory(engine)() as session:
        blob = await session.scalar(
            select(models.AcquisitionRecord.blob_ref).where(
                models.AcquisitionRecord.source_id == raws[2].source_id
            )
        )
    assert blob is not None
    (data_dir / "blobs").mkdir(parents=True, exist_ok=True)
    for path in (data_dir / "blobs").rglob(f"{blob}*"):
        path.unlink()
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(
        "zz-incremental-example", target, missing_limit=10, persist=False
    )

    assert not plan.runnable
    assert plan.missing_count == 1
    assert plan.refusal is RebuildRefusalCode.MISSING_LOCAL_INPUT
    assert plan.covered_documents == 3, "the manifest names it; the bytes are what is gone"


async def test_publication_from_an_incremental_plan_keeps_every_covered_document(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The end-to-end shape of the defect: three documents in, three documents out.

    Before this change, publishing a generation planned from a one-document delta soft-deleted
    the two documents the delta had no reason to mention, because the run was byte-complete and
    byte completeness was read as deletion authority.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)
    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10)
    assert plan.runnable

    published = await publish_plan(
        rebuilds,
        vectors,
        BlobStore(engine, data_dir),
        generation_id=plan.generation_id,
        workspace_id=store.workspace_id,
        owner="incremental-publisher",
    )

    assert published.state is RebuildState.PUBLISHED
    assert await live_source_ids(engine) == {raw.source_id for raw in raws}
    async with session_factory(engine)() as session:
        chunked = (
            (
                await session.execute(
                    select(models.Document.source_id).where(
                        models.Document.chunk_fp == target.chunk_fingerprint,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
    assert set(chunked) == {raw.source_id for raw in raws}, (
        "every live document must carry the new chunk identity, or the migration is not done"
    )


async def test_an_incremental_publication_does_not_retire_what_it_cannot_speak_for(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Deletion authority is the second condition, and a delta never carries it.

    A document acquired outside the bound chain is not evidence that the source still has it —
    but it is not evidence that the source has removed it either, and only a full enumeration
    can tell those apart. So publication leaves it alone rather than tombstoning it.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)
    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10)
    await publish_plan(
        rebuilds,
        vectors,
        BlobStore(engine, data_dir),
        generation_id=plan.generation_id,
        workspace_id=store.workspace_id,
        owner="incremental-publisher",
    )

    async with session_factory(engine)() as session:
        tombstoned = await session.scalar(
            select(models.Document.source_id).where(models.Document.deleted_at.is_not(None))
        )
    assert tombstoned is None


async def test_a_new_promotion_between_planning_and_publication_refuses(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Planning and publication are held to the same membership proof.

    A run promoted after the plan was bound changes which runs a connector contributes, and
    therefore what the replacement would be authoritative over. Publishing the older proof
    against the newer corpus is exactly the class of mistake the plan identity exists to catch.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)
    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10)
    claimed = await rebuilds.claim_generation(
        plan.generation_id, "racing", now=NOW, expires_at=NOW + timedelta(days=36500)
    )
    await promoted_delta(
        store,
        engine,
        data_dir,
        source_id=raws[1].source_id,
        run_id="zzz-later-incremental",
    )

    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=100)
    assert caught.value.code is RebuildRefusalCode.SNAPSHOT_CHANGED
    with pytest.raises(RebuildPublicationConflictError) as raised:
        await rebuilds.publish_generation(
            plan.generation_id,
            owner="racing",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert raised.value.code is RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
    assert await live_source_ids(engine) == {raw.source_id for raw in raws}


async def test_republishing_a_composed_generation_is_idempotent(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """A retry after a lost response must not build a second corpus."""
    _, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)
    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10)
    first = await publish_plan(
        rebuilds,
        vectors,
        BlobStore(engine, data_dir),
        generation_id=plan.generation_id,
        workspace_id=store.workspace_id,
        owner="incremental-publisher",
    )

    replay = await rebuilds.plan_rebuild(delta, target, missing_limit=10)
    again = await rebuilds.publish_generation(
        first.generation_id,
        owner="incremental-publisher",
        lease_generation=first.lease_generation,
        now=NOW,
    )

    assert replay.generation_id == first.generation_id
    assert again.state is RebuildState.PUBLISHED
    assert await live_source_ids(engine) == {raw.source_id for raw in raws}


async def test_planning_a_composed_inventory_costs_a_constant_per_bound_run(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Reaching back through history must not turn planning into a per-document read.

    Composition adds queries per *run*, not per document: a manifest verification, a manifest
    integrity walk and a contribution cursor for each bound run, plus one coverage aggregate
    per connector. The number is asserted rather than described because the failure it guards
    against — a correlated subquery quietly evaluated once per row — looks identical from the
    outside until the corpus is large.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)
    record_selects = 0

    def count_record_selects(*args: object) -> None:
        nonlocal record_selects
        statement = args[2]
        if (
            isinstance(statement, str)
            and statement.lstrip().upper().startswith("SELECT")
            and "FROM acquisition_records" in statement
        ):
            record_selects += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_record_selects)
    try:
        plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10, persist=False)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_record_selects)

    assert plan.documents == 3, "three documents, so a per-document read would be visible"
    bound_runs, reads_per_run, coverage_aggregates = 2, 3, 1
    assert record_selects == bound_runs * reads_per_run + coverage_aggregates, (
        "manifest verification, integrity walk and contribution cursor for each bound run, "
        "plus one coverage aggregate for the connector — and nothing that scales with rows"
    )


async def test_a_live_document_no_bound_run_names_refuses_even_behind_a_full_inventory(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Reaching a full inventory is not the same as accounting for the corpus.

    The chain here connects perfectly — a full inventory, then a delta that resumed from its
    committed cursor — and still leaves one live document unnamed by either. A rebuild led by a
    delta can neither rebuild that document nor justify removing it, so proceeding would strand
    it at the old derived identity while reporting the migration complete.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    await store.upsert_document(
        make_document(
            source="wiki",
            source_id="document-d",
            uri="https://example.test/wiki/d",
            media_type="text/plain",
            body=b"a live document no manifest names",
        )
    )
    delta = await promoted_delta(store, engine, data_dir, source_id=raws[0].source_id)
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(delta, target, missing_limit=10, persist=False)

    assert plan.refusal is RebuildRefusalCode.INCOMPLETE_SOURCE_INVENTORY
    assert (plan.live_documents, plan.covered_documents, plan.uncovered_documents) == (4, 3, 1)


async def test_a_smaller_full_inventory_is_deletion_evidence_and_is_not_refused(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The opposite of the defect, and the reason the fix is not "refuse anything smaller".

    A source that genuinely lost a document produces a full enumeration that no longer names
    it. That enumeration covers less of the live corpus than the corpus contains, which is
    exactly the arithmetic an incremental manifest produces — and it means the opposite thing.
    A full inventory is deletion evidence, so the replacement runs and retires what the source
    no longer has.
    """
    _, raws = await live_full_inventory(store, engine, data_dir)
    await store.upsert_document(
        make_document(
            source="wiki",
            source_id="document-d",
            uri="https://example.test/wiki/d",
            media_type="text/plain",
            body=b"removed at the source before this enumeration",
        )
    )
    vectors, target = await ready_vectors(data_dir)
    rebuilds = rebuild_store(store, engine, data_dir, vectors)

    plan = await rebuilds.plan_rebuild(
        "promoted-glossary-run", target, missing_limit=10, persist=True
    )

    assert plan.runnable
    assert (plan.live_documents, plan.covered_documents, plan.uncovered_documents) == (4, 3, 1)
    await publish_plan(
        rebuilds,
        vectors,
        BlobStore(engine, data_dir),
        generation_id=plan.generation_id,
        workspace_id=store.workspace_id,
        owner="full-inventory-publisher",
    )
    assert await live_source_ids(engine) == {raw.source_id for raw in raws}, (
        "the document the source no longer has is retired, and only that one"
    )


async def test_a_connector_that_walks_its_whole_scope_keeps_its_deletion_authority(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The inherited cursor is not evidence about what discovery actually did.

    Several shipped connectors accept a watermark and discard it, so their second and hundredth
    runs enumerate exactly as much as their first. Reading membership off the cursor alone would
    quietly demote every one of them to incremental, costing them the deletion authority their
    manifests genuinely earn — and refusing their rebuilds the moment a source deletion left a
    live document the newest full enumeration no longer names.
    """
    await live_full_inventory(store, engine, data_dir)
    await store.upsert_document(
        make_document(
            source="wiki",
            source_id="document-d",
            uri="https://example.test/wiki/d",
            media_type="text/plain",
            body=b"removed at the source before this enumeration",
        )
    )
    second = await store.create_acquisition_run(
        "second-full-walk",
        "wiki",
        source_scope="scope:glossary-v1",
        scope_fingerprint="glossary-v1",
        enumerates_full_inventory=True,
    )

    assert second.enumeration_membership is SnapshotMembership.FULL_INVENTORY, (
        "the connector said it walks the whole scope, and it inherited a cursor anyway"
    )
    async with session_factory(engine)() as session:
        row = await session.get(models.AcquisitionRun, second.id)
    assert row is not None
    assert row.base_watermark is not None, "which is exactly why deriving it would be wrong"


def test_the_shipped_connectors_that_discard_the_cursor_say_so() -> None:
    """A declaration nothing reads back is a declaration that quietly stops being true.

    Each of these accepts a watermark and immediately discards it, and the only thing standing
    between that and an offline rebuild treating their manifests as deltas is this attribute.
    """
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415 - import cost
        ConfluenceSnapshotConnector,
    )
    from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415 - import cost
    from manicule.connectors.git_site import GitSiteConnector  # noqa: PLC0415 - import cost
    from manicule.ingest.pipeline import (  # noqa: PLC0415 - import cost
        snapshot_enumerates_full_inventory,
    )

    walkers = (FilesystemConnector, ConfluenceSnapshotConnector, GitSiteConnector)
    assert all(connector.enumerates_full_inventory for connector in walkers)
    assert not snapshot_enumerates_full_inventory(object()), (
        "and a connector that makes no claim is treated as incremental, not assumed complete"
    )
