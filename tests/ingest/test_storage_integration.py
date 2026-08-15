"""The pipeline against the real stores: a migrated SQLite database and LanceDB.

Everything else in this directory runs against in-memory doubles, which is the right default
and is not sufficient. The doubles agree with the protocol by construction; only the real store
can disagree with it — and the writes this ticket added are exactly the ones nothing else
exercises: lineage, tombstones, the recovery sweep, run counters, ``index_state``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, override

import pytest
from sqlalchemy import text

from manicule.connectors import CursorExpiredError, NotFoundError, SessionExpiredError
from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionRecord,
    AcquisitionRecordState,
    AcquisitionRunState,
)
from manicule.core.content import (
    IN_FLIGHT,
    LEGACY_PUBLICATION,
    Commit,
    DocumentStatus,
    PipelineStage,
    RawDocument,
    Retention,
)
from manicule.core.embedding import IndexFingerprints, Vector
from manicule.core.ids import content_hash, document_id, vector_id
from manicule.core.sources import Watermark
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.recovery import requeue_interrupted
from manicule.ingest.reindex import re_parse, reindex_document
from manicule.ingest.sweeps import sweep_vectors
from manicule.ingest.workers import InProcessRunner
from manicule.storage.blobs import BlobStore
from manicule.storage.vectors import LanceVectorStore
from tests.fakes import HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk, Document
    from manicule.core.sources import DiscoveredDoc, DocRef
    from manicule.storage.docstore import SqliteDocStore


def _pipeline(
    docs: SqliteDocStore,
    vectors: LanceVectorStore,
    blobs: BlobStore | None = None,
    embedder: HashEmbedder | None = None,
) -> IngestPipeline:
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=docs,
        chunker=chunker,
        embedder=embedder or HashEmbedder(),
        vectors=vectors,
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        blobs=blobs,
    )


def _confluence_snapshot(
    root: Path, *, page_id: str = "123456", version: int = 7, body: str | None = None
) -> Path:
    """One Confluence page snapshot: a manifest and a storage-format body beside it.

    Synthetic throughout — the page, the space, the host and the ids are invented here.
    """
    from manicule.connectors.confluence_snapshot import MANIFEST_NAME  # noqa: PLC0415

    directory = root / "ENG" / page_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "body.xhtml").write_text(
        body if body is not None else "<h2>Retry policy</h2>\n<p>The client retries twice.</p>\n",
        encoding="utf-8",
    )
    (directory / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "page_id": page_id,
                "title": "Retry policy",
                "space_key": "ENG",
                "canonical_url": (
                    f"https://docs.example.test/wiki/spaces/ENG/pages/{page_id}/Retry-policy"
                ),
                "version": version,
                "modified_at": "2026-03-04T05:06:07+00:00",
                "ancestors": ["Runbooks"],
                "retrieved_at": "2026-06-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return directory


async def test_annotated_state_survives_a_re_ingest_through_the_real_store(
    store: SqliteDocStore, data_dir: Path, tmp_path: Path
) -> None:
    """The claim that would have sunk the metadata-precedence reorder, against the real store.

    Fresh connector metadata now beats the stored copy, and the property that has to survive is the
    other one: per-document state written by :meth:`IngestStore.annotate` — which no connector
    supplies — must not be erased by a re-ingest. It survives because a key absent from
    ``raw.metadata`` overrides nothing, which needs no special case.

    **Asserted here rather than only against the in-memory double**, because the double got this
    wrong: its ``annotate`` wrote to a side dictionary nothing read, so an annotation vanished and
    the assertion passed for the wrong reason. Fixed there too, but the real store is the authority
    and this is where the claim belongs.
    """
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415
        ConfluenceSnapshotConnector,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _confluence_snapshot(corpus, version=7)

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    await pipeline.run(ConfluenceSnapshotConnector(corpus))
    document = (await store.list_documents())[0]
    await store.annotate(document.id, {"last_ingest_error": {"stage": "parse", "detail": "old"}})

    _confluence_snapshot(corpus, version=8, body="<h2>Retry policy</h2>\n<p>Three times.</p>\n")
    await pipeline.run(ConfluenceSnapshotConnector(corpus))

    after = (await store.list_documents())[0]
    assert after.metadata["last_ingest_error"] == {"stage": "parse", "detail": "old"}, (
        "state a connector never supplies was erased by the metadata reorder"
    )
    assert after.provenance is not None
    assert after.provenance.source is not None
    assert after.provenance.source.version == "8", "and the connector's own facts still refreshed"
    await vectors.teardown()


async def test_a_confluence_snapshot_cites_the_page_through_the_real_stores(
    store: SqliteDocStore, data_dir: Path, tmp_path: Path
) -> None:
    """A mirrored Confluence page, end to end, over a real directory and a migrated database.

    The connector's own suite stops at the bytes it hands over. This takes those bytes through the
    real pipeline into the real SQLite store, so it is what catches the record being lost in the
    JSON column's round trip, or the pipeline and the connector disagreeing about which field is
    the citation.
    """
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415
        ConfluenceSnapshotConnector,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _confluence_snapshot(corpus)

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    report = await _pipeline(store, vectors).run(ConfluenceSnapshotConnector(corpus))

    assert report.indexed == 1, "the manifest must not be indexed as a document of its own"
    stored = (await store.list_documents())[0]

    assert stored.source_id == "123456", "identity is the page id, not the directory"
    assert stored.title == "Retry policy"
    assert stored.uri == "https://docs.example.test/wiki/spaces/ENG/pages/123456/Retry-policy"

    record = stored.provenance
    assert record is not None, "the record must survive the JSON column round trip"
    assert record.source is not None
    assert record.source.version == "7"
    assert record.source.section_path == ("ENG", "Runbooks")
    assert record.snapshot is not None
    assert record.snapshot.path == "ENG/123456/body.xhtml"
    await vectors.teardown()


async def test_re_exporting_a_page_at_a_higher_version_replaces_it(
    store: SqliteDocStore, data_dir: Path, tmp_path: Path
) -> None:
    """One document, updated — not a second one. Over the real store, which is where it counts.

    Three claims in one run, because they are one claim from three sides: the page id keys the
    document so a re-export updates rather than duplicates; the record refreshes so the citation
    reports the version it was just told about; and the chunks are replaced rather than accumulated,
    which is what stops a stale half of the page staying retrievable.
    """
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415
        ConfluenceSnapshotConnector,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _confluence_snapshot(corpus, version=7)

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    await pipeline.run(ConfluenceSnapshotConnector(corpus))
    first = (await store.list_documents())[0]

    _confluence_snapshot(
        corpus, version=8, body="<h2>Retry policy</h2>\n<p>The client retries three times.</p>\n"
    )
    await pipeline.run(ConfluenceSnapshotConnector(corpus))

    documents = await store.list_documents()
    assert len(documents) == 1, "a re-export of one page is one document, not two"
    assert documents[0].id == first.id, "and the same document, so its citations still resolve"
    record = documents[0].provenance
    assert record is not None
    assert record.source is not None
    assert record.source.version == "8"
    chunks = await store.document_chunks(documents[0].id)
    texts = "\n".join(chunk.text for chunk in await store.get_chunks([c.id for c in chunks]))
    assert "three times" in texts
    assert "retries twice" not in texts, "the superseded text must not stay retrievable"
    await vectors.teardown()


async def test_re_ingesting_an_unchanged_snapshot_changes_nothing(
    store: SqliteDocStore, data_dir: Path, tmp_path: Path
) -> None:
    """Idempotence, and duplicate prevention, over the store that would actually duplicate.

    An export re-run nightly with nothing changed must cost nothing and must not grow the corpus.
    The chunk ids are compared rather than only counted: equal counts with different ids means every
    vector was rewritten, which is the expensive version of the same bug.
    """
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415
        ConfluenceSnapshotConnector,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _confluence_snapshot(corpus)

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    await pipeline.run(ConfluenceSnapshotConnector(corpus))
    document = (await store.list_documents())[0]
    before = [chunk.id for chunk in await store.document_chunks(document.id)]

    report = await pipeline.run(ConfluenceSnapshotConnector(corpus))

    assert len(await store.list_documents()) == 1
    after = [chunk.id for chunk in await store.document_chunks(document.id)]
    assert after == before, "an unchanged page must keep its chunk ids, and therefore its vectors"
    assert report.skipped_version or report.skipped_hash, (
        "an unchanged page should be skipped rather than re-parsed; neither counter moved"
    )
    await vectors.teardown()


async def test_a_mirrored_page_with_a_manifest_cites_the_page_through_the_real_stores(
    store: SqliteDocStore,
    data_dir: Path,
    tmp_path: Path,
) -> None:
    """The whole claim, end to end, over a real directory and a migrated database.

    Every other source-metadata test substitutes something: the interface tests build records
    directly, the sidecar tests stop at the connector, and the pipeline tests use an in-memory
    store. This one walks a real tree with a real manifest in it using the real
    :class:`~manicule.connectors.filesystem.FilesystemConnector`, and writes through the real
    SQLite store — so it is the test that catches the record being lost in the JSON column's
    round trip, or the connector and the readers disagreeing about the reserved key.

    It is also the assertion behind the sentence added to ``README.md``: a page stored as
    ``123456.html`` beside a ``123456.html.source.json`` cites as the retry policy, at its
    canonical address, with the local snapshot still on the record.

    The breadcrumb is deliberately not asserted here — this file's chunker is a fake that does
    not build one. ``tests/test_chunking.py`` covers that against the real chunker.
    """
    from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415
    from manicule.connectors.sidecar import manifest_path_for  # noqa: PLC0415

    # A corpus directory of its own: `data_dir` is also under `tmp_path`, so walking the latter
    # would index this test's own SQLite database and vector files as documents.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    page = corpus / "123456.html"
    page.write_text("Retry policy\nThe client retries twice.\n", encoding="utf-8")
    manifest_path_for(page).write_text(
        json.dumps(
            {
                "title": "Retry policy",
                "canonical_uri": "https://docs.example.test/pages/123456/retry-policy",
                "source_id": "123456",
                "version": "7",
                "modified_at": "2026-03-04T05:06:07+00:00",
                "retrieved_at": "2026-06-01T00:00:00+00:00",
                "section_path": ["Engineering", "Runbooks"],
            }
        ),
        encoding="utf-8",
    )

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    report = await _pipeline(store, vectors).run(FilesystemConnector(corpus, name="local"))

    assert report.indexed == 1, "the manifest must not be indexed as a document of its own"
    stored = (await store.list_documents())[0]

    # The citation, which is the point of the whole change.
    assert stored.title == "Retry policy"
    assert stored.uri == "https://docs.example.test/pages/123456/retry-policy"

    # The identity, which is now the page's own and no longer where the file happens to sit.
    assert stored.source_id == "123456", (
        "a manifest that declares a source_id declares the document's identity; keying on the "
        "path instead makes a reorganized mirror a corpus of new documents and orphans the old"
    )
    assert stored.id == document_id("default", "local", "123456")

    # The local snapshot, which must survive being superseded.
    record = stored.provenance
    assert record is not None, "the record must survive the JSON column round trip"
    assert record.snapshot is not None
    assert record.snapshot.path == "123456.html"
    assert record.source is not None
    assert record.source.version == "7"
    assert record.source.section_path == ("Engineering", "Runbooks")

    # Three distinct timestamps, none standing in for another.
    assert record.source.modified_at != record.snapshot.retrieved_at
    assert stored.indexed_at is not None
    assert stored.indexed_at not in {record.source.modified_at, record.snapshot.retrieved_at}


@pytest.mark.contract
async def test_the_sqlite_store_satisfies_the_ingest_surface(store: SqliteDocStore) -> None:
    """Structural conformance, checked rather than assumed.

    :class:`~manicule.ingest.ports.IngestStore` exists so that ingest imports no database. A
    protocol nothing is checked against is a protocol that drifts.
    """
    from manicule.ingest.ports import IngestStore  # noqa: PLC0415 - local to the assertion

    assert isinstance(store, IngestStore)


async def test_a_document_reaches_both_stores_and_is_marked_indexed_last(
    store: SqliteDocStore,
    data_dir: Path,
) -> None:
    """The whole path, through the components a real installation uses."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)

    report = await pipeline.run(fakes.DictConnector({"a": "alpha\nbeta"}))

    assert report.indexed == 1
    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert await store.count_chunks(document.id) == 2
    assert await vectors.count() == 2
    await vectors.teardown()


async def test_float32_overflow_never_stages_or_publishes_a_real_vector(
    store: SqliteDocStore,
    data_dir: Path,
) -> None:
    """A finite backend value outside Lance's range is a contextual document failure."""
    import warnings  # noqa: PLC0415

    class OverflowEmbedder(HashEmbedder):
        @override
        async def embed(self, texts: Sequence[str]) -> list[Vector]:
            return [[1e39, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    embedder = OverflowEmbedder(model_id="fake/overflowing")
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embedder.fingerprint)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = await _pipeline(store, vectors, embedder=embedder).run(
            fakes.DictConnector({"overflow": "finite in Python, not in float32"})
        )

    document = await store.find_document("memory", "overflow")
    assert report.by_status == {DocumentStatus.FAILED.value: 1}
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.publication_id == LEGACY_PUBLICATION
    assert document.status_detail is not None
    assert "non-finite" in document.status_detail
    assert "fake/overflowing" in document.status_detail
    assert await store.count_chunks(document.id) == 0
    assert await vectors.count() == 0
    assert await store.take_tombstones(10) == []
    await vectors.teardown()


async def test_deleting_chunks_tombstones_their_vectors_and_the_sweep_removes_them(
    store: SqliteDocStore,
    data_dir: Path,
) -> None:
    """The trigger is #33's; the runner is this ticket's, and it needs the real trigger."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    await pipeline.run(fakes.DictConnector({"a": "alpha\nbeta"}))
    document = await store.find_document("memory", "a")
    assert document is not None

    await store.replace_chunks(document.id, [])
    assert await store.take_tombstones(10), "the delete trigger must record the vectors to sweep"

    result = await sweep_vectors(store, vectors)

    assert result.vectors_removed == 2
    assert await vectors.count() == 0
    assert await store.take_tombstones(10) == []
    await vectors.teardown()


async def test_hard_delete_tombstones_the_active_physical_vectors(
    store: SqliteDocStore,
    data_dir: Path,
    engine: AsyncEngine,
) -> None:
    """Every document and workspace cascade carries the exact generation-keyed Lance id."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
    from tests.storage_helpers import make_chunk, make_document  # noqa: PLC0415

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    await _pipeline(store, vectors).run(fakes.DictConnector({"a": "alpha\nbeta"}))
    document = await store.find_document("memory", "a")
    assert document is not None
    chunks = list(await store.document_chunks(document.id))
    physical = {vector_id(document.publication_id, chunk.id) for chunk in chunks}
    other = SqliteDocStore(engine, workspace_id="other")
    await other.ensure_workspace()
    await other.delete_document(document.id)
    assert await store.get_document(document.id) is not None, "another workspace cannot delete it"

    doomed = SqliteDocStore(engine, workspace_id="doomed")
    await doomed.ensure_workspace()
    doomed_document = make_document(
        source="memory", source_id="workspace-member", workspace_id="doomed"
    ).model_copy(update={"publication_id": "workspace-publication"})
    doomed_document = await doomed.upsert_document(doomed_document)
    doomed_chunk = make_chunk(doomed_document, 0, "workspace body")
    await doomed.replace_chunks(doomed_document.id, [doomed_chunk])
    await vectors.upsert(
        [doomed_chunk],
        [[0.3] * HashEmbedder().fingerprint.dimension],
        publication_id=doomed_document.publication_id,
    )
    physical.add(vector_id(doomed_document.publication_id, doomed_chunk.id))
    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM workspaces WHERE id = 'doomed'"))

    await store.delete_document(document.id)
    tombstones = set(await store.take_tombstones(20))
    assert physical <= tombstones
    await sweep_vectors(store, vectors)
    assert await vectors.count() == 0
    await vectors.teardown()


async def test_a_legacy_tombstone_survives_a_publication_to_publication_flip(
    store: SqliteDocStore,
    data_dir: Path,
    engine: AsyncEngine,
) -> None:
    """P1 cleanup must not erase evidence for a still-present pre-publication vector."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    connector = fakes.DictConnector({"a": "unchanged body"})
    await pipeline.run(connector)
    first = await store.find_document("memory", "a")
    assert first is not None
    chunks = list(await store.document_chunks(first.id))
    assert len(chunks) == 1
    legacy_id = chunks[0].id
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO vector_tombstones (chunk_id, deleted_at) "
                "VALUES (:chunk_id, CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING"
            ),
            {"chunk_id": legacy_id},
        )

    await pipeline.ingest_raw(
        RawDocument(
            source_id="a",
            uri="memory://a",
            media_type=fakes.MEDIA_TYPE,
            content="unchanged body",
            metadata={"revision_note": "metadata-only publication change"},
        ),
        source="memory",
        force=True,
        expected=first.revision,
    )

    second = await store.find_document("memory", "a")
    assert second is not None
    assert second.publication_id != first.publication_id
    assert [chunk.id for chunk in await store.document_chunks(second.id)] == [legacy_id]
    assert legacy_id in await store.take_tombstones(20)
    await vectors.teardown()


async def test_real_lance_retry_keeps_the_publication_after_float32_round_trip(
    store: SqliteDocStore,
    data_dir: Path,
) -> None:
    """Reused normalized float32 values identify the same publication as raw model output."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    raw = RawDocument(source_id="a", uri="memory://a", media_type=fakes.MEDIA_TYPE, content="alpha")
    first_outcome = await pipeline.ingest_raw(raw, source="memory", force=True)
    first = await store.get_document(first_outcome[0].document_id)
    assert first is not None

    second_outcome = await pipeline.ingest_raw(
        raw,
        source="memory",
        force=True,
        expected=first.revision,
    )
    second = await store.get_document(second_outcome[0].document_id)
    assert second is not None
    assert second.publication_id == first.publication_id
    assert await vectors.count() == 1
    tombstones = set(await store.take_tombstones(20))
    chunks = await store.document_chunks(second.id)
    assert vector_id(second.publication_id, chunks[0].id) not in tombstones
    await vectors.teardown()


async def test_failure_inside_relational_publication_rolls_back_every_derived_row(
    store: SqliteDocStore,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after chunks are replaced still leaves the old publication wholly readable."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    connector = fakes.DictConnector({"a": "NOW - Network Operations Workspace\nold body"})
    await pipeline.run(connector)
    before = await store.find_document("memory", "a")
    assert before is not None
    old_chunks = list(await store.document_chunks(before.id))
    old_glossary = list(await store.glossary_entries(before.id))
    old_lineage = (before.parse_fp, await store.glossary_lineage(before.id))

    async def fail_between_writes(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("power lost between relational writes")

    monkeypatch.setattr(store, "_replace_entries", fail_between_writes)
    connector.documents["a"] = "SRE - Site Reliability Engineering\nnew body"
    report = await pipeline.run(connector)

    after = await store.find_document("memory", "a")
    assert after is not None
    assert report.indexed == 1, "the previous indexed revision remains the reported outcome"
    assert (after.content_hash, after.publication_id, after.status) == (
        before.content_hash,
        before.publication_id,
        DocumentStatus.INDEXED,
    )
    assert list(await store.document_chunks(after.id)) == old_chunks
    assert list(await store.glossary_entries(after.id)) == old_glossary
    assert (after.parse_fp, await store.glossary_lineage(after.id)) == old_lineage
    assert await store.take_tombstones(10), "staged vectors stay named for crash-safe cleanup"
    await sweep_vectors(store, vectors)
    assert await vectors.count() == len(old_chunks), "cleanup retained every active vector"
    await vectors.teardown()


async def test_the_recovery_sweep_requeues_only_the_in_flight_statuses(
    store: SqliteDocStore,
) -> None:
    """Against the real ``CHECK`` constraint, which is what makes the new statuses storable."""
    from tests.storage_helpers import make_document  # noqa: PLC0415 - the storage builder

    for status in (*sorted(IN_FLIGHT), DocumentStatus.CONTAINER):
        document = make_document(source_id=f"s-{status.value}", status=status)
        await store.upsert_document(document)

    requeued = await store.requeue_stale(
        IN_FLIGHT, datetime.now(UTC) + timedelta(seconds=1), detail="interrupted; requeued"
    )

    assert requeued == len(IN_FLIGHT)
    container = await store.find_document("fs", f"s-{DocumentStatus.CONTAINER.value}")
    assert container is not None
    assert container.status is DocumentStatus.CONTAINER


async def test_nothing_is_requeued_when_nothing_is_stale(store: SqliteDocStore) -> None:
    """Zero is the healthy answer, and a non-zero one at every startup means something."""
    from tests.storage_helpers import make_document  # noqa: PLC0415

    await store.upsert_document(make_document(status=DocumentStatus.PARSING))

    assert await requeue_interrupted(store, stale_after_s=3600) == 0


async def test_chunks_are_read_within_a_workspace_and_not_across_one(
    engine: AsyncEngine,
) -> None:
    """The repair verbs read by document id, and an id is not a scope.

    ``chunks`` has no workspace column, so a query on ``document_id`` alone answers about any
    tenant's document. Today's callers pass ids from scoped queries — but this is the read
    behind ``reindex --re-embed``, and a repair verb that can be pointed at an id is exactly
    where an unscoped read stops being theoretical.
    """
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415 - local to this test
    from tests.storage_helpers import make_chunk, make_document  # noqa: PLC0415

    theirs = SqliteDocStore(engine, workspace_id="them")
    await theirs.ensure_workspace()
    document = await theirs.upsert_document(make_document(workspace_id="them"))
    await theirs.replace_chunks(document.id, [make_chunk(document, 0, "theirs")])
    assert await theirs.document_chunks(document.id), "the seed must exist to be excluded"

    ours = SqliteDocStore(engine, workspace_id="us")
    await ours.ensure_workspace()

    assert await ours.document_chunks(document.id) == []


async def test_a_failure_cannot_be_recorded_without_the_stage_that_caused_it(
    store: SqliteDocStore,
) -> None:
    """The schema requires the pair, so the API must refuse the half of it that is not offered.

    Without the guard the call reaches SQLite and returns an ``IntegrityError`` naming a check
    constraint — the right outcome reached by luck, and a diagnosis that points at the schema
    rather than at the missing argument.
    """
    from tests.storage_helpers import make_document  # noqa: PLC0415

    document = await store.upsert_document(make_document())

    with pytest.raises(ValueError, match="failed_stage"):
        await store.set_status(document.id, DocumentStatus.FAILED, "boom")


async def test_the_index_state_round_trips_both_fingerprints(store: SqliteDocStore) -> None:
    """One row, because a data directory holds one index."""
    fingerprint = HashEmbedder().fingerprint
    chunk = fakes.BlockChunker.fingerprint

    assert (await store.index_fingerprints()).is_empty
    await store.record_index_fingerprints(
        IndexFingerprints(embed=fingerprint, chunk=chunk, vector_table="chunks__abcd1234")
    )

    read = await store.index_fingerprints()
    assert read.embed == fingerprint
    assert read.chunk == chunk
    assert read.vector_table == "chunks__abcd1234"


async def test_a_skip_records_liveness_and_the_new_token(store: SqliteDocStore) -> None:
    from tests.storage_helpers import make_document  # noqa: PLC0415

    document = await store.upsert_document(make_document())

    # S105/S106: a connector's change token, not a credential. It is opaque and compared for
    # equality, which is exactly why it reads like one to a scanner.
    token = "version-2"  # noqa: S105
    await store.record_seen(document.id, version_token=token)

    refreshed = await store.get_document(document.id)
    assert refreshed is not None
    assert refreshed.version_token == token


async def test_lineage_is_recorded_per_document_and_none_leaves_one_alone(
    store: SqliteDocStore,
) -> None:
    """``None`` means "unchanged", not "cleared" — the difference a re-embed depends on."""
    from tests.storage_helpers import make_document  # noqa: PLC0415

    document = await store.upsert_document(make_document())
    await store.set_lineage(document.id, chunk_fp="chunker-1", embed_fp="model-1")

    await store.set_lineage(document.id, chunk_fp=None, embed_fp="model-2")

    selected = await store.select_documents(chunk_fp_other_than="chunker-1")
    assert selected == [], "the chunk lineage must not have moved"


async def test_run_counters_land_on_the_connector_row(
    store: SqliteDocStore,
    data_dir: Path,
) -> None:
    """No new table: the column ``connectors.metadata`` already exists for this."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    recorded = await store.connector_metadata("memory")
    assert isinstance(recorded["last_run"], dict)
    assert recorded["last_run"]["discovered"] == 1
    await vectors.teardown()


async def test_cursor_expiry_preserves_sync_metadata_until_one_complete_retry(
    store: SqliteDocStore,
    data_dir: Path,
) -> None:
    """The real connector row distinguishes partial durable progress from its later checkpoint."""

    class Expiring(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__({"a": "alpha", "b": "beta"})
            self.expires = True

        @property
        @override
        def watermark(self) -> Watermark:
            return Watermark(value="position-2", observed_at=datetime.now(UTC))

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            async for found in super().discover(watermark):
                yield found
                if self.expires:
                    raise CursorExpiredError("the search cursor expired")

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(store, vectors)
    connector = Expiring()

    assert await store.connector_metadata("memory") == {}
    assert await store.get_watermark("memory") is None

    failed = await pipeline.run(connector)
    after_failure = await store.connector_metadata("memory")
    assert failed.error_type == "CursorExpiredError"
    assert failed.indexed == 1
    assert len(await store.list_documents()) == 1
    assert await store.get_watermark("memory") is None
    assert after_failure["last_synced_at"] is None
    assert after_failure["last_run"]["outcome"] == "incomplete"
    assert after_failure["last_run"]["watermark_advanced"] is False

    connector.expires = False
    completed = await pipeline.run(connector)
    after_retry = await store.connector_metadata("memory")
    watermark = await store.get_watermark("memory")
    assert completed.complete
    assert completed.watermark_advanced
    assert len(await store.list_documents()) == 2
    assert watermark is not None
    assert watermark.value == "position-2"
    assert isinstance(after_retry["last_synced_at"], str)
    assert after_retry["last_run"]["outcome"] == "complete"
    assert after_retry["last_run"]["watermark_advanced"] is True
    await vectors.teardown()


async def test_durable_enumeration_finishes_before_slow_indexing_can_age_a_cursor(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """A 100-record source reaches its true end while embedding is still parked.

    Ten synthetic page responses each take 19 ms on a manual clock, below the 500 ms cursor
    lifetime. Journal pages and both in-memory hand-offs are smaller than the corpus. The model
    gate proves indexing has not drained anything to make enumeration succeed.
    """
    clock = fakes.ManualClock()
    embedder = fakes.ClockedGatedEmbedder(clock, seconds_per_document=0.05)
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=embedder,
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
        detect_glossary=False,
    )
    connector = fakes.ExpiringCursorConnector(
        {f"synthetic-doc-{number:04d}": f"public synthetic line {number}" for number in range(100)},
        clock=clock,
        page_size=10,
        cursor_lifetime_seconds=0.5,
        response_seconds=0.019,
    )

    task = asyncio.create_task(pipeline.run(connector))
    try:
        await connector.enumeration_completed.wait()
        await embedder.gate.wait_for(1)

        durable = await store.latest_unsettled_acquisition_run(connector.name)
        assert durable is not None
        assert durable.discovered_count == 100
        assert durable.enumeration_completed_at is not None
        assert durable.candidate_watermark == connector.watermark
        assert connector.pages_requested == 10
        assert clock.now == pytest.approx(0.19)
        assert len(await store.list_acquisition_records(durable.id, limit=3)) == 3
        assert await store.get_watermark(connector.name) == connector.watermark, (
            "source coverage is checkpointed before the parked embedder completes"
        )
    finally:
        # A failed observation must not strand pipeline tasks or their SQLite connections.
        embedder.gate.open()
        report = await task

    indexed = {source_id async for source_id in store.known_source_ids(connector.name)}

    assert report.error_type == ""
    assert report.enumeration_completed
    assert report.indexed == 100
    assert len(indexed) == 100
    assert report.watermark_advanced
    assert await store.get_watermark(connector.name) == connector.watermark
    assert report.stages.fetch_queue.capacity == 2
    assert report.stages.fetch_queue.peak_depth <= 2
    assert report.stages.peak_discovery_records == 2
    assert report.stages.parse_queue.peak_depth <= report.stages.parse_queue.capacity
    assert await store.latest_unsettled_acquisition_run(connector.name) is None
    settled = await store.get_acquisition_run(durable.id)
    assert settled is not None
    assert settled.state is AcquisitionRunState.SETTLED
    records = await store.list_acquisition_records(durable.id)
    assert {record.state for record in records} == {AcquisitionRecordState.SETTLED}


async def test_crash_during_enumeration_keeps_only_the_committed_prefix(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    connector = fakes.PausedEnumerationConnector(
        {f"public-doc-{number}": f"line {number}" for number in range(10)},
        pause_after=3,
    )
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await connector.paused.wait()
    task.cancel()
    connector.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert durable.discovered_count == 3
    assert durable.enumeration_completed_at is None
    assert durable.candidate_watermark is None
    assert len(await store.list_acquisition_records(durable.id)) == 3
    assert await store.get_watermark(connector.name) is None


async def test_crash_after_enumeration_preserves_the_marker_records_and_candidate(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    class GatedJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.reader_arrived = asyncio.Event()
            self.release_reader = asyncio.Event()

        @override
        async def list_acquisition_records(
            self,
            run_id: str,
            *,
            states: Sequence[AcquisitionRecordState] | None = None,
            after_sequence: int | None = None,
            limit: int = 100,
        ) -> Sequence[AcquisitionRecord]:
            self.reader_arrived.set()
            await self.release_reader.wait()
            return await super().list_acquisition_records(
                run_id, states=states, after_sequence=after_sequence, limit=limit
            )

    store = GatedJournal()
    await store.ensure_workspace()
    clock = fakes.ManualClock()
    lease_clock = fakes.ManualLeaseClock()
    connector = fakes.ExpiringCursorConnector(
        {f"public-doc-{number}": f"line {number}" for number in range(10)},
        clock=clock,
        page_size=5,
        cursor_lifetime_seconds=0.5,
    )
    chunker = fakes.BlockChunker()
    blobs = BlobStore(engine, data_dir)

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=blobs,
            chunker=chunker,
            embedder=HashEmbedder(),
            vectors=fakes.MemoryVectors(),
            runner=InProcessRunner({"lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            shutdown_grace_s=30,
            detect_glossary=False,
            acquisition_clock=lease_clock,
        )

    task = asyncio.create_task(pipeline().run(connector))
    await store.reader_arrived.wait()
    task.cancel()
    # Deliver cancellation first so the graceful stop is visible before the blocked journal
    # read returns. No newly read record may be admitted during the drain window.
    await asyncio.sleep(0)
    store.release_reader.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert durable.discovered_count == 10
    assert durable.enumeration_completed_at is not None
    assert durable.candidate_watermark == connector.watermark
    records = await store.list_acquisition_records(durable.id)
    assert len(records) == 10
    assert {record.state for record in records} == {AcquisitionRecordState.DISCOVERED}
    assert await store.list_documents() == []
    assert await store.get_watermark(connector.name) is None

    pages_before_resume = connector.pages_requested
    with pytest.raises(RuntimeError, match="could not be claimed"):
        await pipeline().run(connector)
    assert connector.pages_requested == pages_before_resume

    lease_clock.advance(301)
    resumed = await pipeline().run(connector)

    assert connector.pages_requested == pages_before_resume, "the source was rediscovered"
    assert resumed.indexed == 10
    assert resumed.watermark_advanced
    assert await store.latest_unsettled_acquisition_run(connector.name) is None
    settled = await store.get_acquisition_run(durable.id)
    assert settled is not None
    assert settled.state is AcquisitionRunState.SETTLED
    assert await store.get_watermark(connector.name) == durable.candidate_watermark


async def test_duplicate_discovery_is_one_durable_record_and_one_publication(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class DuplicateSource(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__({"public-a": "alpha", "public-b": "beta"})
            self.enumerated = asyncio.Event()
            self.release = asyncio.Event()

        @property
        @override
        def watermark(self) -> Watermark:
            return Watermark(value="duplicate-position", observed_at=datetime.now(UTC))

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            async for discovered in super().discover(watermark):
                durable = discovered.model_copy(
                    update={
                        "ref": discovered.ref.model_copy(
                            update={
                                "uri": (
                                    f"https://source.example.test/documents/{discovered.source_id}"
                                )
                            }
                        )
                    }
                )
                yield durable
                if durable.source_id == "public-a":
                    yield durable
            self.enumerated.set()
            await self.release.wait()

    connector = DuplicateSource()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await connector.enumerated.wait()
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert durable.discovered_count == 2
    records = await store.list_acquisition_records(durable.id)
    assert [record.source.source_id for record in records] == ["public-a", "public-b"]

    connector.release.set()
    report = await task

    assert report.indexed == 2
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


async def test_expired_worker_is_fenced_before_publication_after_takeover(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """An embed begun by one generation cannot publish after a successor takes the run."""
    lease_clock = fakes.ManualLeaseClock()
    embedder = fakes.GatedEmbedder()
    chunker = fakes.BlockChunker()
    connector = fakes.DictConnector(
        {"public-fenced-document": "public synthetic line"}, name="fenced-synthetic-source"
    )
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=embedder,
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        acquisition_lease_s=1,
        acquisition_clock=lease_clock,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None

    lease_clock.advance(2)
    successor = await store.claim_acquisition_run(
        durable.id,
        "successor-attempt",
        now=lease_clock(),
        expires_at=lease_clock() + timedelta(seconds=1),
    )
    assert successor is not None
    assert successor.lease_generation > durable.lease_generation

    embedder.gate.open()
    report = await task
    stored = await store.find_document(connector.name, "public-fenced-document")

    assert report.indexed == 0
    assert report.error_type == "_LostAcquisitionLeaseError"
    assert stored is None, "the expired generation made no document revision servable"


@pytest.mark.parametrize(
    ("unchanged", "status"),
    [(True, DocumentStatus.INDEXED), (False, DocumentStatus.PARSING)],
)
async def test_takeover_fences_fetch_side_last_seen_and_status_writes(
    engine: AsyncEngine, *, unchanged: bool, status: DocumentStatus
) -> None:
    """A slow lookup cannot let an expired fetch worker mutate the document it found."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
    from tests.storage_helpers import make_document  # noqa: PLC0415

    class GatedFindStore(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.arm = False
            self.found = asyncio.Event()
            self.release = asyncio.Event()

        @override
        async def find_document(self, source: str, source_id: str) -> Document | None:
            document = await super().find_document(source, source_id)
            if self.arm:
                self.arm = False
                self.found.set()
                await self.release.wait()
            return document

    store = GatedFindStore()
    await store.ensure_workspace()
    source = "fenced-fetch-source"
    source_id = "public-fetch-document"
    stored = make_document(
        source=source,
        source_id=source_id,
        status=status,
        media_type=fakes.MEDIA_TYPE,
        body=b"public synthetic line",
    ).model_copy(update={"version_token": "same-token" if unchanged else "old-token"})
    before = await store.upsert_document(stored)
    connector = fakes.DictConnector({source_id: "public synthetic line"}, name=source)
    connector.tokens[source_id] = "same-token" if unchanged else "new-token"
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        acquisition_lease_s=300,
        acquisition_clock=lease_clock,
        detect_glossary=False,
    )

    store.arm = True
    task = asyncio.create_task(pipeline.run(connector))
    await store.found.wait()
    durable = await store.latest_unsettled_acquisition_run(source)
    assert durable is not None
    lease_clock.advance(301)
    successor = await store.claim_acquisition_run(
        durable.id,
        "successor-fetch-attempt",
        now=lease_clock(),
        expires_at=lease_clock() + timedelta(seconds=300),
    )
    assert successor is not None
    store.release.set()

    report = await task
    after = await store.find_document(source, source_id)

    assert report.error_type == "_LostAcquisitionLeaseError"
    assert after == before, "the stale worker changed last-seen/version metadata or status"


@pytest.mark.parametrize("fetch_fails", [False, True])
async def test_takeover_fences_hash_skip_and_failure_demotion(
    store: SqliteDocStore, *, fetch_fails: bool
) -> None:
    """Neither a hash skip nor a fetch failure may mutate after lease takeover."""
    from tests.storage_helpers import make_document  # noqa: PLC0415

    class GatedFailureConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__(
                {
                    "public-failure-document": (
                        "replacement line" if fetch_fails else "original line"
                    )
                },
                name="fenced-failure-source",
            )
            self.tokens["public-failure-document"] = "replacement-token"
            self.fetch_arrived = asyncio.Event()
            self.release_fetch = asyncio.Event()

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            self.fetch_arrived.set()
            await self.release_fetch.wait()
            if fetch_fails:
                raise RuntimeError("synthetic fetch failure")
            return await super().fetch(ref)

    connector = GatedFailureConnector()
    source_id = "public-failure-document"
    before = await store.upsert_document(
        make_document(
            source=connector.name,
            source_id=source_id,
            media_type=fakes.MEDIA_TYPE,
            body=b"original line",
        ).model_copy(update={"version_token": "original-token"})
    )
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        acquisition_lease_s=300,
        acquisition_clock=lease_clock,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await connector.fetch_arrived.wait()
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    lease_clock.advance(301)
    successor = await store.claim_acquisition_run(
        durable.id,
        "successor-failure-attempt",
        now=lease_clock(),
        expires_at=lease_clock() + timedelta(seconds=300),
    )
    assert successor is not None
    connector.release_fetch.set()

    report = await task
    after = await store.find_document(connector.name, source_id)

    assert report.error_type == "_LostAcquisitionLeaseError"
    assert after == before, "the stale worker changed version/last-seen data or failure state"


async def test_takeover_between_vector_staging_and_upsert_fences_the_vector_write(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """A valid tombstone stage is not authority for a later vector write after takeover."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    class GatedVectorStageStore(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.staged = asyncio.Event()
            self.release_stage = asyncio.Event()

        @override
        async def stage_vectors(self, publication_id: str, chunks: Sequence[Chunk]) -> None:
            await super().stage_vectors(publication_id, chunks)
            self.staged.set()
            await self.release_stage.wait()

    store = GatedVectorStageStore()
    await store.ensure_workspace()
    vectors = fakes.MemoryVectors()
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()
    connector = fakes.DictConnector(
        {"public-vector-document": "public synthetic line"}, name="fenced-vector-source"
    )
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=vectors,
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        acquisition_lease_s=300,
        acquisition_clock=lease_clock,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await store.staged.wait()
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    lease_clock.advance(301)
    successor = await store.claim_acquisition_run(
        durable.id,
        "successor-vector-attempt",
        now=lease_clock(),
        expires_at=lease_clock() + timedelta(seconds=300),
    )
    assert successor is not None
    store.release_stage.set()

    report = await task

    assert report.error_type == "_LostAcquisitionLeaseError"
    assert vectors.rows == {}, "the stale worker wrote vectors after its staged-store await"


async def test_acquisition_failure_does_not_mutate_the_prior_document(
    engine: AsyncEngine,
) -> None:
    """Source failure belongs to the journal; the published document is not a fetch ledger."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
    from tests.storage_helpers import make_document  # noqa: PLC0415

    class GatedAnnotationStore(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.annotated = asyncio.Event()

        @override
        async def annotate(self, document_id: str, updates: Mapping[str, Any]) -> None:
            await super().annotate(document_id, updates)
            self.annotated.set()

    store = GatedAnnotationStore()
    await store.ensure_workspace()
    connector = fakes.DictConnector(
        {"public-demotion-document": "replacement line"}, name="fenced-demotion-source"
    )
    connector.tokens["public-demotion-document"] = "replacement-token"
    connector.fail_fetch.add("public-demotion-document")
    existing = await store.upsert_document(
        make_document(
            source=connector.name,
            source_id="public-demotion-document",
            status=DocumentStatus.PARSING,
            media_type=fakes.MEDIA_TYPE,
            body=b"original line",
        ).model_copy(update={"version_token": "original-token"})
    )
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )

    report = await pipeline.run(connector)
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    after = await store.get_document(existing.id)
    record = (await store.list_acquisition_records(durable.id))[0]

    assert report.error_type == ""
    assert not store.annotated.is_set()
    assert record.state is AcquisitionRecordState.RETRY
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == "fetch_failed"
    assert after is not None
    assert after == existing


async def test_a_durable_limited_run_has_no_completion_marker_or_watermark(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    connector = fakes.PausedEnumerationConnector(
        {f"public-doc-{number}": f"line {number}" for number in range(10)},
        pause_after=9,
    )
    chunker = fakes.BlockChunker()
    lease_clock = fakes.ManualLeaseClock()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
        acquisition_clock=lease_clock,
    )

    report = await pipeline.run(connector, limit=3)
    durable = await store.latest_unsettled_acquisition_run(connector.name)

    assert report.limited
    assert not report.enumeration_completed
    assert report.indexed == 3
    assert durable is not None
    assert durable.discovered_count == 3
    assert durable.enumeration_completed_at is None
    assert durable.candidate_watermark is None
    assert await store.get_watermark(connector.name) is None

    lease_clock.advance(301)
    repeated = await pipeline.run(connector, limit=3)
    same_run = await store.latest_unsettled_acquisition_run(connector.name)

    assert repeated.limited
    assert repeated.indexed == 0
    assert connector.yields == 3, "a satisfied limit must not even open the source iterator"
    assert same_run is not None
    assert same_run.id == durable.id
    assert same_run.discovered_count == 3
    assert len(await store.list_acquisition_records(durable.id)) == 3


async def test_crash_after_snapshot_association_resumes_without_source_access(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """Once ACQUIRED is durable, even a crash during embedding cannot cause a re-download."""
    connector = fakes.DictConnector(
        {"public-resumable": "public retained source bytes"}, name="resumable-source"
    )
    lease_clock = fakes.ManualLeaseClock()
    blobs = BlobStore(engine, data_dir)
    gated = fakes.GatedEmbedder()
    chunker = fakes.BlockChunker()

    def pipeline(embedder: HashEmbedder | fakes.GatedEmbedder) -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=blobs,
            chunker=chunker,
            embedder=embedder,
            vectors=fakes.MemoryVectors(),
            runner=InProcessRunner({"lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            acquisition_clock=lease_clock,
            acquisition_lease_s=1,
            shutdown_grace_s=0,
            detect_glossary=False,
        )

    task = asyncio.create_task(pipeline(gated).run(connector))
    await gated.gate.wait_for(1)
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    records = await store.list_acquisition_records(durable.id)
    assert len(records) == 1
    assert records[0].blob_ref is not None
    assert records[0].acquired_source is not None
    fetches_before = list(connector.fetches)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    connector.fail_fetch.add("public-resumable")
    lease_clock.advance(2)

    resumed = await pipeline(HashEmbedder()).run(connector)

    assert connector.fetches == fetches_before
    assert resumed.indexed == 1
    stored = await store.find_document(connector.name, "public-resumable")
    assert stored is not None
    assert stored.original_ref == records[0].blob_ref


async def test_crash_after_file_retention_before_journal_association_uses_staging(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class GatedBlobStore(BlobStore):
        def __init__(self) -> None:
            super().__init__(engine, data_dir)
            self.retained = asyncio.Event()
            self.release = asyncio.Event()
            self.pause = True

        @override
        async def retain_acquisition(
            self, key: str, raw: RawDocument
        ) -> tuple[Retention, AcquiredSource]:
            result = await super().retain_acquisition(key, raw)
            if self.pause:
                self.retained.set()
                await self.release.wait()
            return result

    connector = fakes.DictConnector({"public-staged": "public staged bytes"}, name="staged-source")
    blobs = GatedBlobStore()
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=blobs,
            chunker=chunker,
            embedder=HashEmbedder(),
            vectors=fakes.MemoryVectors(),
            runner=InProcessRunner({"lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            acquisition_clock=lease_clock,
            acquisition_lease_s=1,
            shutdown_grace_s=0,
            detect_glossary=False,
        )

    task = asyncio.create_task(pipeline().run(connector))
    await blobs.retained.wait()
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    before = (await store.list_acquisition_records(durable.id))[0]
    assert before.state is AcquisitionRecordState.ACQUIRING
    assert before.blob_ref is None
    fetches = list(connector.fetches)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    connector.fail_fetch.add("public-staged")
    blobs.pause = False
    blobs.release.set()
    lease_clock.advance(2)

    resumed = await pipeline().run(connector)

    assert connector.fetches == fetches
    assert resumed.indexed == 1
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (SessionExpiredError("synthetic session expired"), "authentication"),
        (NotFoundError("synthetic source document disappeared"), "source_deleted"),
    ],
)
async def test_typed_acquisition_failures_block_source_coverage_without_publication(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    failure: Exception,
    code: str,
) -> None:
    class RefusingConnector(fakes.DictConnector):
        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            del ref
            raise failure

    connector = RefusingConnector({"public-refused": "never returned"}, name="refusing-source")
    chunker = fakes.BlockChunker()
    report = await IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    ).run(connector)

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.RETRY
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == code
    assert record.blob_ref is None
    assert not report.watermark_advanced
    assert await store.find_document(connector.name, "public-refused") is None


async def test_fetched_newer_revision_is_the_snapshot_and_publication_version(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class OrderedRevisionConnector(fakes.DictConnector):
        @staticmethod
        def fetched_revision_at_least(discovered: str, fetched: str) -> bool:
            try:
                return int(fetched.removeprefix("revision-")) >= int(
                    discovered.removeprefix("revision-")
                )
            except ValueError:
                return False

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            raw = await super().fetch(ref)
            return raw.model_copy(
                update={"metadata": {**raw.metadata, "version_token": "revision-5"}}
            )

    connector = OrderedRevisionConnector(
        {"public-moving": "newer public bytes"}, name="moving-source"
    )
    connector.tokens["public-moving"] = "revision-4"
    chunker = fakes.BlockChunker()
    report = await IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    ).run(connector)

    stored = await store.find_document(connector.name, "public-moving")
    assert report.indexed == 1
    assert stored is not None
    assert stored.version_token == "revision-5"  # noqa: S105 - synthetic source revision
    runs = await store.latest_unsettled_acquisition_run(connector.name)
    assert runs is None


@pytest.mark.parametrize("fetched", ["revision-3", "unrelated-token", None])
async def test_unproven_or_older_fetched_revision_is_retried_without_retention(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    fetched: str | None,
) -> None:
    class RefusingRevisionConnector(fakes.DictConnector):
        @staticmethod
        def fetched_revision_at_least(discovered: str, actual: str) -> bool:
            try:
                return int(actual.removeprefix("revision-")) >= int(
                    discovered.removeprefix("revision-")
                )
            except ValueError:
                return False

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            raw = await super().fetch(ref)
            metadata = dict(raw.metadata)
            if fetched is None:
                metadata.pop("version_token", None)
            else:
                metadata["version_token"] = fetched
            return raw.model_copy(update={"metadata": metadata})

    connector = RefusingRevisionConnector(
        {"public-stale": "untrusted public bytes"}, name="stale-revision-source"
    )
    connector.tokens["public-stale"] = "revision-4"
    chunker = fakes.BlockChunker()
    blobs = BlobStore(engine, data_dir)
    files_before = {path for path in blobs.root.rglob("blake2b/**/*") if path.is_file()}
    await IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=blobs,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    ).run(connector)

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.RETRY
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == "stale_body"
    assert record.blob_ref is None
    assert {path for path in blobs.root.rglob("blake2b/**/*") if path.is_file()} == files_before


async def test_a_re_parse_reads_retained_bytes_rather_than_the_network(
    store: SqliteDocStore,
    data_dir: Path,
    engine: AsyncEngine,
) -> None:
    """Rung 3 against the real blob store, including its content addressing."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    blobs = BlobStore(engine, data_dir)
    pipeline = _pipeline(store, vectors, blobs)
    connector = fakes.DictConnector({"a": "alpha\nbeta"})
    await pipeline.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.original_ref is not None
    connector.fetches.clear()

    report = await re_parse([document], pipeline=pipeline, blobs=blobs)

    assert report.documents == 1
    assert connector.fetches == [], "a re-parse is not a re-crawl"
    await vectors.teardown()


async def test_a_re_parse_refuses_a_snapshot_the_real_store_has_already_moved_past(
    store: SqliteDocStore,
    data_dir: Path,
    engine: AsyncEngine,
) -> None:
    """The compare-and-swap where it has to be a SQL statement rather than a dictionary.

    The in-memory double gets atomicity for nothing — there is no ``await`` between its
    comparison and its write, so no other task can be scheduled in between — which means it can
    satisfy the protocol while the real store does not. This drives the same refusal through a
    migrated database.

    No two tasks here, deliberately. The invariant is about *revisions*, not about timing: a
    re-parse holding a snapshot the store has moved past must decline whether it was overtaken a
    microsecond ago or an hour ago, and stating it serially is the same claim without a gate.
    """
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    blobs = BlobStore(engine, data_dir)
    pipeline = _pipeline(store, vectors, blobs)
    connector = fakes.DictConnector({"a": "alpha\nbeta"})
    await pipeline.run(connector)
    stale = await store.find_document("memory", "a")
    assert stale is not None

    connector.documents["a"] = "gamma\ndelta"
    assert (await pipeline.run(connector)).indexed == 1
    report = await re_parse([stale], pipeline=pipeline, blobs=blobs)

    assert len(report.superseded) == 1, (
        "the snapshot is two revisions behind and its commit has to be refused"
    )
    assert report.documents == 0, "and refusing it is not counting it as repaired"
    assert not report.failures, "nor as a failure of the document"
    current = await store.get_document(stale.id)
    assert current is not None
    assert current.content_hash != stale.content_hash
    stored = await store.document_chunks(stale.id)
    assert [chunk.text for chunk in stored] == ["gamma", "delta"], (
        "the newer sync's text is what the corpus holds, through the real store's own writes"
    )
    await vectors.teardown()


async def test_the_stores_guard_reads_an_absent_field_as_absent_rather_than_as_no_match(
    store: SqliteDocStore,
) -> None:
    """Three of the revision's five fields are nullable, and SQL will not compare them.

    ``column = NULL`` is never true in SQL — not even of a ``NULL`` — so a guard that reached
    the database spelled that way misses on every document with no version token, no retained
    reference and no recorded parse lineage. That is most of a corpus, and the symptom is not a
    crash: it is a sweep that repairs nothing and reports the whole index superseded by nobody.

    The store is written with ``==`` and is nonetheless correct, because SQLAlchemy renders
    ``column == None`` as ``column IS NULL``. That is a fact about the query builder rather than
    about this code, which is exactly why it is pinned here rather than trusted: the guard would
    stop holding the moment the clause became raw SQL, a parameter, or anything else that passes
    the value through instead of inspecting it.

    Checked in both directions on the same document, because the half that matters is the one
    that has to *succeed*: a guard that refused everything would satisfy any test that only ever
    looked for refusals.
    """
    from tests.storage_helpers import make_document  # noqa: PLC0415 - the storage builder

    document = make_document(source="memory", source_id="null-fields")
    stored = await store.upsert_document(document)
    assert (stored.version_token, stored.original_ref, stored.parse_fp) == (None, None, None), (
        "the fixture must leave all three unset, or this is a test about a different document"
    )

    hit = await store.commit_document(
        stored.model_copy(update={"title": "renamed by the guarded write"}),
        expected=stored.revision,
    )

    assert hit.committed is True
    assert hit.stored is not None
    assert hit.stored.title == "renamed by the guarded write"

    moved = await store.upsert_document(
        stored.model_copy(update={"content_hash": content_hash(b"something else")})
    )
    miss = await store.commit_document(
        stored.model_copy(update={"title": "written by the stale snapshot"}),
        expected=stored.revision,
    )

    assert miss.committed is False
    assert miss.stored is not None
    assert miss.stored.title == moved.title, "and the row is left exactly as the winner wrote it"
    assert miss.stored.content_hash == moved.content_hash


async def test_atomic_publication_cas_has_one_winner_across_two_real_sessions(
    store: SqliteDocStore,
    engine: AsyncEngine,
) -> None:
    """The conditional UPDATE is the first statement, so no stale read precedes the lock."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
    from tests.storage_helpers import make_document  # noqa: PLC0415

    original = await store.upsert_document(
        make_document(source="memory", source_id="publication-race")
    )
    contenders = [
        SqliteDocStore(engine),
        SqliteDocStore(engine),
    ]
    replacements = [
        original.model_copy(update={"title": title, "publication_id": publication})
        for title, publication in (("left", "publication-left"), ("right", "publication-right"))
    ]

    async def publish(index: int) -> Commit:
        return await contenders[index].publish_document(
            replacements[index],
            [],
            expected=original.revision,
            chunk_fp=None,
            embed_fp=None,
            parse_fp=None,
            glossary_entries=None,
            glossary_fp=None,
            original_omitted_reason=None,
        )

    results = await asyncio.gather(publish(0), publish(1))

    assert [result.committed for result in results].count(True) == 1
    miss = next(result for result in results if not result.committed)
    winner = next(result for result in results if result.committed)
    assert miss.stored == winner.stored
    assert await store.get_document(original.id) == winner.stored


async def test_stale_retained_failure_cannot_overwrite_an_indexed_winners_original(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure retention and its guarded row are one commit, not a trailing setter."""
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    from manicule.core.content import DocumentRevision  # noqa: PLC0415
    from manicule.storage import models  # noqa: PLC0415
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
    from tests.storage_helpers import make_document  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir)
    old_ref = (await blobs.retain(b"old source", "text/plain")).ref
    winner_ref = (await blobs.retain(b"winner source", "text/plain")).ref
    stale_failure_ref = (await blobs.retain(b"stale failed source", "text/plain")).ref
    assert old_ref
    assert winner_ref
    assert stale_failure_ref
    original = await store.upsert_document(
        make_document(
            source="memory",
            source_id="failure-race",
            status=DocumentStatus.NO_EXTRACTABLE_TEXT,
        ).model_copy(update={"original_ref": old_ref})
    )

    winner_store = SqliteDocStore(engine)
    loser_store = SqliteDocStore(engine)
    winner_entered = asyncio.Event()
    release_winner = asyncio.Event()
    loser_attempted = asyncio.Event()
    replace_entries = winner_store._replace_entries  # pyright: ignore[reportPrivateUsage]

    async def pause_winner(*args: object, **kwargs: object) -> None:
        winner_entered.set()
        await release_winner.wait()
        await replace_entries(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    target = loser_store._publication_target  # pyright: ignore[reportPrivateUsage]

    async def observe_loser(
        session: AsyncSession,
        document: Document,
        expected: DocumentRevision | None,
    ) -> tuple[models.Document | None, Commit | None]:
        loser_attempted.set()
        return await target(session, document, expected)

    monkeypatch.setattr(winner_store, "_replace_entries", pause_winner)
    monkeypatch.setattr(loser_store, "_publication_target", observe_loser)
    winner_document = original.model_copy(
        update={
            "content_hash": content_hash(b"winner source"),
            "original_ref": winner_ref,
            "publication_id": "winner-publication",
            "status": DocumentStatus.INDEXED,
            "status_detail": None,
            "failed_stage": None,
        }
    )
    failed_document = original.model_copy(
        update={
            "content_hash": content_hash(b"stale failed source"),
            "original_ref": stale_failure_ref,
            "status": DocumentStatus.FAILED,
            "status_detail": "parser crashed",
            "failed_stage": PipelineStage.PARSE,
        }
    )

    winner_task = asyncio.create_task(
        winner_store.publish_document(
            winner_document,
            [],
            expected=original.revision,
            chunk_fp=None,
            embed_fp=None,
            parse_fp=None,
            glossary_entries=[],
            glossary_fp="winner-glossary",
            original_omitted_reason=None,
        )
    )
    await winner_entered.wait()
    loser_task = asyncio.create_task(
        loser_store.publish_failure(
            failed_document,
            expected=original.revision,
            original_omitted_reason=None,
        )
    )
    await loser_attempted.wait()
    release_winner.set()
    winner, loser = await asyncio.gather(winner_task, loser_task)

    assert winner.committed
    assert not loser.committed
    assert loser.stored == winner.stored
    async with engine.connect() as connection:
        retained = (
            await connection.execute(
                text(
                    "SELECT original_ref, original_omitted_reason "
                    "FROM documents WHERE id = :document_id"
                ),
                {"document_id": original.id},
            )
        ).one()
    assert retained == (winner_ref, None)


async def test_initial_failure_rolls_back_retained_reference_with_its_row(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expected=None path cannot expose half of its retention decision."""
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    from tests.storage_helpers import make_document  # noqa: PLC0415

    retained_ref = (await BlobStore(engine, data_dir).retain(b"failed source", "text/plain")).ref
    assert retained_ref
    failed = make_document(
        source="memory",
        source_id="initial-failure-rollback",
        status=DocumentStatus.FAILED,
    ).model_copy(
        update={
            "original_ref": retained_ref,
            "status_detail": "parser crashed",
            "failed_stage": PipelineStage.PARSE,
        }
    )
    write_document = store._write_document  # pyright: ignore[reportPrivateUsage]

    async def interrupt_after_row_write(
        session: AsyncSession,
        document: Document,
    ) -> Document:
        await write_document(session, document)
        raise RuntimeError("power lost before retention outcome")

    monkeypatch.setattr(store, "_write_document", interrupt_after_row_write)

    with pytest.raises(RuntimeError, match="power lost before retention outcome"):
        await store.publish_failure(
            failed,
            expected=None,
            original_omitted_reason=None,
        )

    assert await store.get_document(failed.id) is None


async def test_the_stores_guard_refuses_on_a_corrected_record_over_bytes_that_never_moved(
    store: SqliteDocStore,
) -> None:
    """The half of the revision that cannot be a ``WHERE`` clause, on its own.

    Four of the five fields are columns and go into the conditional ``UPDATE``. The source
    record is not: it lives inside the metadata JSON, so it is compared in Python after that
    statement has taken the write lock. Nothing else here reaches that comparison — every other
    test moves the bytes, which moves three columns at once and refuses before it is consulted.

    A mirrored page whose manifest is corrected over an unchanged body moves nothing else at
    all: same bytes, same hash, same token, same retained reference, same parser. If the record
    is not part of the revision, a re-parse holding the old snapshot commits and writes the
    correction away — with a citation then naming a title the source has stopped using, and a
    successful repair reported for it.
    """
    from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata  # noqa: PLC0415
    from tests.storage_helpers import make_document  # noqa: PLC0415 - the storage builder

    stale = await store.upsert_document(make_document(source="memory", source_id="corrected"))
    corrected = await store.upsert_document(
        stale.model_copy(
            update={
                "title": "Retry policy",
                "metadata": {
                    PROVENANCE_KEY: Provenance(
                        source=SourceMetadata(title="Retry policy", version="8")
                    ).model_dump(mode="json")
                },
            }
        )
    )
    assert (corrected.content_hash, corrected.version_token, corrected.original_ref) == (
        stale.content_hash,
        stale.version_token,
        stale.original_ref,
    ), "no column moved, so every columnar half of the guard still matches the stale snapshot"
    assert corrected.provenance is not None

    miss = await store.commit_document(
        stale.model_copy(update={"title": "written back by the stale snapshot"}),
        expected=stale.revision,
    )

    assert miss.committed is False
    assert miss.stored is not None
    assert miss.stored.title == "Retry policy", "the correction is still what the row says"
    assert miss.stored.provenance == corrected.provenance


async def test_the_guarded_write_refuses_another_workspaces_id_rather_than_reporting_a_miss(
    engine: AsyncEngine,
) -> None:
    """A tenancy refusal, checked on the *guarded* write rather than only on the plain one.

    The conditional ``UPDATE`` is workspace-scoped, so an id belonging to another tenant matches
    nothing and comes back through the ordinary "somebody got there first" path — which is the
    trap: that path returns the row it found so that a caller can report what overtook it, and
    the row it found here belongs to somebody else. A caller in this workspace would be handed
    another workspace's title, URI and metadata, in a result that looks entirely routine.

    ``upsert_document`` has always refused this outright, and a guarded write that answered
    differently would make the refusal a property of which method was called.
    """
    from manicule.core.errors import ManiculeError  # noqa: PLC0415
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
    from manicule.storage.scoped import CrossWorkspaceCollisionError  # noqa: PLC0415
    from tests.storage_helpers import make_document  # noqa: PLC0415

    theirs = SqliteDocStore(engine, workspace_id="them")
    await theirs.ensure_workspace()
    document = await theirs.upsert_document(
        make_document(workspace_id="them", title="their private title")
    )
    ours = SqliteDocStore(engine, workspace_id="us")
    await ours.ensure_workspace()

    with pytest.raises(CrossWorkspaceCollisionError) as refused:
        await ours.commit_document(document, expected=document.revision)

    assert isinstance(refused.value, ManiculeError)
    assert "their private title" not in str(refused.value), (
        "the refusal names the workspaces and the id, not what the other tenant's row holds"
    )
    assert "them" in str(refused.value)


async def test_the_blob_store_speaks_the_retention_vocabulary(
    data_dir: Path,
    engine: AsyncEngine,
) -> None:
    """One type, so neither side has to import the other to say what happened."""
    blobs = BlobStore(engine, data_dir, max_bytes=4)

    kept = await blobs.retain(b"tiny", "text/plain")
    refused = await blobs.retain(b"far too large", "text/plain")

    assert isinstance(kept, Retention)
    assert kept.ref is not None
    assert refused.ref is None
    assert refused.omitted_reason is not None


async def test_a_purged_document_comes_back_with_its_citations_intact(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Delete, purge, restore, reindex — the whole loop, on the real stores.

    This is the claim the trash rests on. Inside the grace period a restore is free; outside
    it the sweep has taken the chunks and the vectors, and the way back is a single-document
    re-parse from retained bytes — rung 3, no network, and the source is never consulted.

    The assertion that matters is the last one. ``chunks.id`` is derived from
    ``(document_id, position, text)``, so re-parsing the same retained bytes reproduces the
    same ids: every citation into this document survives a deletion it outlived. If the ids
    moved, restoring a document would silently invalidate every answer that had ever quoted it,
    and nothing would report that either.
    """
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    blobs = BlobStore(engine, data_dir)
    pipeline = _pipeline(store, vectors, blobs)
    await pipeline.run(fakes.DictConnector({"a": "alpha\nbeta"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    before = [chunk.id for chunk in await store.document_chunks(document.id)]
    assert before, "the fixture must produce chunks, or this proves nothing about keeping them"

    await store.soft_delete_document(document.id)
    swept = await sweep_vectors(store, vectors, soft_delete_grace_s=-1.0)
    assert swept.documents_purged == 1
    assert await vectors.count() == 0

    restoration = await store.restore_document(document.id)
    assert restoration.restored
    assert restoration.needs_reparse, "the sweep took the content, so the row came back empty"

    report = await reindex_document(document.id, store=store, pipeline=pipeline, blobs=blobs)
    assert report.unrepairable == []
    assert report.failures == []
    assert report.documents == 1

    after = [chunk.id for chunk in await store.document_chunks(document.id)]
    assert after == before, (
        "a restored document must keep its chunk ids, or every citation into it now dangles"
    )
    served = await store.get_document(document.id)
    assert served is not None
    assert served.status is DocumentStatus.INDEXED


async def test_a_single_document_reindex_refuses_an_id_that_is_still_in_the_trash(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Restore first, then reindex. The other order finds nothing and must say why.

    The lookup is workspace-scoped and skips the trash, which is what makes a mistyped id, an
    id from another tenant and a deleted one all arrive as the same reported outcome rather
    than as a repair that quietly did nothing.
    """
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    blobs = BlobStore(engine, data_dir)
    pipeline = _pipeline(store, vectors, blobs)
    await pipeline.run(fakes.DictConnector({"a": "alpha"}))
    document = await store.find_document("memory", "a")
    assert document is not None
    await store.soft_delete_document(document.id)

    report = await reindex_document(document.id, store=store, pipeline=pipeline, blobs=blobs)

    assert report.documents == 0
    assert len(report.unrepairable) == 1
    assert "restored before it can be re-parsed" in report.unrepairable[0]


async def _enriched(root: Path, *, page_id: str = "1002", name: str = "1002.html") -> Path:
    """One enriched page with its manifest beside it, under ``root``. Synthetic throughout."""
    from manicule.connectors.enriched_html import write_sidecars  # noqa: PLC0415

    target = root / name
    target.write_text(
        f"<!doctype html><html><head><title>Retry Runbook</title></head><body>"
        f"<section data-source-metadata>"
        f"<p><strong>Page ID:</strong> {page_id}</p>"
        f"<p><strong>Version:</strong> 7</p>"
        f'<p><strong>Source:</strong> <a href="https://docs.example.test/pages/{page_id}">'
        f"canonical page</a></p>"
        f"</section>"
        f'<main data-document-representation="storage">'
        f"<h1>Retry Runbook</h1><p>The client retries twice.</p></main>"
        f"</body></html>",
        encoding="utf-8",
    )
    await asyncio.to_thread(write_sidecars, root, force=True)
    return target


async def test_a_collection_survives_the_page_being_moved(
    store: SqliteDocStore, data_dir: Path, tmp_path: Path
) -> None:
    """Curation is the thing a path-keyed identity loses, so it is the thing to prove survives.

    Collection membership and tags hang off ``documents.id`` with ``ON DELETE CASCADE``. Under
    path identity a reorganized mirror is a corpus of new documents beside a corpus of orphans,
    and every hand-curated collection quietly empties — a loss no re-sync repairs because the
    curation is not in the corpus. Under the page's own identity the row is the same row, so the
    membership is still there afterwards. Asserted against the real store because the cascade is
    the schema's and only the schema can disagree with it.
    """
    from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    await _enriched(corpus)

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    await _pipeline(store, vectors).run(FilesystemConnector(corpus, name="handbook"))
    document = (await store.list_documents())[0]
    assert document.source_id == "1002"

    collection = await store.create_collection("runbooks", description="on-call")
    assert await store.add_to_collection(collection.id, [document.id]) == 1

    moved = corpus / "reorganized"
    moved.mkdir()
    for name in ("1002.html", "1002.html.source.json"):
        (corpus / name).rename(moved / name)
    await _pipeline(store, vectors).run(FilesystemConnector(corpus, name="handbook"))

    assert len(await store.list_documents()) == 1, "the move produced a second document"
    after = (await store.list_documents())[0]
    assert after.id == document.id
    assert [held.id for held in await store.collection_documents(collection.id)] == [document.id], (
        "the collection lost its document when the page moved"
    )
    assert after.provenance is not None
    assert after.provenance.snapshot is not None
    assert after.provenance.snapshot.path == "reorganized/1002.html"
    await vectors.teardown()


async def test_migrated_chunks_are_replaced_by_the_next_sync_and_none_keeps_a_stale_id(
    store: SqliteDocStore, data_dir: Path, tmp_path: Path
) -> None:
    """The two consequences of leaving chunks alone, asserted end to end over the real store.

    ``chunk_id`` digests the document id, so re-keying a document leaves every one of its chunks
    with an id that no longer equals ``chunk_id(document_id, position, text)``. Nothing recomputes
    one and compares — ``chunk_id`` is called in exactly one place and ``glossary_entry_id`` in
    one, both to *mint* an id at write time — so the inconsistency is invisible to every read
    path. What matters is that it does not persist, and this is where that is checked rather than
    argued: migrate, sync, and every surviving chunk derives from the identity it now hangs off.

    It also pins the other half. Before the sync the document serves its old chunks — the
    generic-HTML parse, metadata banner and all — which is exactly what the change exists to keep
    out of the index and exactly why ``doctor``'s ``document-content`` check reports it.
    """
    from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415
    from manicule.core.ids import chunk_id, document_id  # noqa: PLC0415
    from manicule.storage.migrator import downgrade, upgrade  # noqa: PLC0415

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    await _enriched(corpus)
    engine = store.engine
    await downgrade(engine, "5f1c8a34b7d9")

    stale = "old-path-keyed-id"
    banner = "Page ID: 1002 — exported by the wiki mirror"
    async with engine.begin() as connection:
        # The store fixture has already created the default workspace; the downgrade left it.
        await connection.execute(
            text(
                "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, "
                "media_type, content_hash, status, metadata, created_at, updated_at) "
                "VALUES (:id, 'default', 'handbook', :path, 'file:///x', 'Retry Runbook', "
                "'text/html', 'stale-hash', 'indexed', :metadata, "
                "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
            ),
            {
                "id": stale,
                "path": str(corpus / "1002.html"),
                "metadata": json.dumps(
                    {
                        "source_provenance": {
                            "source": {"source_id": "1002", "title": "Retry Runbook"},
                            "snapshot": {"path": "1002.html"},
                            "unavailable_reason": "",
                        }
                    }
                ),
            },
        )
        await connection.execute(
            text(
                "INSERT INTO chunks (id, document_id, text, embed_text, heading_text, "
                "heading_path, kind, position, token_count, anchor, metadata, created_at) "
                "VALUES (:id, :doc, :text, :text, '', '[]', 'prose', 0, 5, :anchor, '{}', "
                "'2026-01-01T00:00:00+00:00')"
            ),
            {
                "id": chunk_id(stale, 0, banner),
                "doc": stale,
                "text": banner,
                "anchor": '{"kind": "unlocated", "reason": "seeded"}',
            },
        )

    await upgrade(engine)

    moved = document_id("default", "handbook", "1002")
    assert [row.id for row in await store.list_documents()] == [moved]
    before = await store.document_chunks(moved)
    assert [chunk.text for chunk in before] == [banner], (
        "the seed must actually seed the stale parse, or this test proves nothing"
    )
    assert before[0].id != chunk_id(moved, 0, banner), (
        "the chunk id must be stale after the move, or the assertion below is vacuous"
    )

    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    await _pipeline(store, vectors).run(FilesystemConnector(corpus, name="handbook"))

    after = await store.document_chunks(moved)
    assert after, "the sync left the document with no chunks at all"
    assert not any(chunk.text == banner for chunk in after), (
        "the metadata banner survived the sync, which is the defect this change exists to fix"
    )
    for chunk in after:
        assert chunk.id == chunk_id(moved, chunk.position, chunk.text), (
            "a chunk survived with an id that does not derive from the identity it hangs off"
        )
    await vectors.teardown()
