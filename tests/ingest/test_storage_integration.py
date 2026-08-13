"""The pipeline against the real stores: a migrated SQLite database and LanceDB.

Everything else in this directory runs against in-memory doubles, which is the right default
and is not sufficient. The doubles agree with the protocol by construction; only the real store
can disagree with it — and the writes this ticket added are exactly the ones nothing else
exercises: lineage, tombstones, the recovery sweep, run counters, ``index_state``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from manicule.core.content import IN_FLIGHT, DocumentStatus, Retention
from manicule.core.embedding import IndexFingerprints
from manicule.core.ids import content_hash, document_id
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
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.storage.docstore import SqliteDocStore


def _pipeline(
    docs: SqliteDocStore, vectors: LanceVectorStore, blobs: BlobStore | None = None
) -> IngestPipeline:
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=docs,
        chunker=chunker,
        embedder=HashEmbedder(),
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
        "path instead makes a reorganised mirror a corpus of new documents and orphans the old"
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
    path identity a reorganised mirror is a corpus of new documents beside a corpus of orphans,
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

    moved = corpus / "reorganised"
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
    assert after.provenance.snapshot.path == "reorganised/1002.html"
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
