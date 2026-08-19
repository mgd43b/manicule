"""The pipeline against the real stores: a migrated SQLite database and LanceDB.

Everything else in this directory runs against in-memory doubles, which is the right default
and is not sufficient. The doubles agree with the protocol by construction; only the real store
can disagree with it — and the writes this ticket added are exactly the ones nothing else
exercises: lineage, tombstones, the recovery sweep, run counters, ``index_state``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import resource
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from sqlalchemy import text

from manicule.app import control
from manicule.app.dispatch import run_op
from manicule.app.served import ControlHandler
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.connectors import CursorExpiredError, NotFoundError, SessionExpiredError
from manicule.connectors.config import FullInventoryAuthority
from manicule.connectors.confluence import ConfluenceConnector
from manicule.connectors.sessions import SessionVault
from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionFailureCode,
    AcquisitionFence,
    AcquisitionInventoryState,
    AcquisitionRecord,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
    AcquisitionStage,
    SnapshotCompleteness,
    SnapshotItemOutcome,
    SnapshotPromotionPolicy,
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
from manicule.core.rebuild import RebuildCheckpoint, RebuildState, RebuildTarget
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.ingest.capacity import CapacityDiagnostic, CapacityRefusedError, CapacityResource
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline, RunReport, Watching
from manicule.ingest.rebuild import build_offline_rebuilder
from manicule.ingest.recovery import requeue_interrupted
from manicule.ingest.reembed import ReembedState, resume_reembed, start_reembed
from manicule.ingest.reindex import re_parse, reindex_document
from manicule.ingest.sweeps import sweep_vectors
from manicule.ingest.workers import InProcessRunner
from manicule.parsers.chain import ParserChain
from manicule.parsers.config import PlaintextConfig
from manicule.parsers.plaintext import PlaintextParser
from manicule.parsers.versions import parse_fingerprint
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.models import FTS_TOKENIZER
from manicule.storage.rebuild import SqliteRebuildStore
from manicule.storage.vectors import LanceVectorStore
from tests.app.fakes import FakeBackend, FakeIngestion
from tests.connectors.fake_confluence import FakeConfluence, FakePage
from tests.connectors.support import Clock as ConfluenceClock
from tests.connectors.support import client_for, cloud_config, connected, server_config
from tests.fakes import HashEmbedder
from tests.ingest import fakes
from tests.ingest.test_shadow_reembed import Authority, Corpus, CountingEmbedder

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from manicule.core.content import Chunk, Document, ParsedBlock
    from manicule.core.sources import DiscoveredDoc, DocRef


class PublicationMemoryVectors(fakes.MemoryVectors):
    """Deterministic publication adapter for the bounded 1,000-item acceptance scenario."""

    def __init__(self) -> None:
        super().__init__()
        self.publications: dict[str, dict[str, fakes.VectorRow]] = {"legacy": self.rows}
        self.peak_upsert_items = 0
        self.retrieval_probes = 0

    @override
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = "legacy",
    ) -> None:
        self.peak_upsert_items = max(self.peak_upsert_items, len(chunks))
        rows = self.publications.setdefault(publication_id, {})
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows[chunk.id] = fakes.VectorRow(
                document_id=chunk.document_id,
                vector=tuple(vector),
                embed_text=chunk.embed_text,
                identity=self._identity_of(chunk),
            )

    @override
    async def count(self) -> int:
        return sum(len(rows) for rows in self.publications.values())

    async def publication_row_count(self, publication_id: str) -> int:
        return len(self.publications.get(publication_id, {}))

    async def publication_page_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        if self._fingerprint is None or self._fingerprint.canonical() != embedding_fingerprint:
            return False
        rows = self.publications.get(publication_id, {})
        return all(
            (row := rows.get(chunk.id)) is not None
            and row.document_id == chunk.document_id
            and row.embed_text == chunk.embed_text
            and row.identity == self._identity_of(chunk)
            for chunk in chunks
        )

    async def publication_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        return await self.publication_row_count(publication_id) == len(
            chunks
        ) and await self.publication_page_is_complete(
            publication_id,
            chunks,
            embedding_fingerprint=embedding_fingerprint,
        )

    async def copy_publication(
        self,
        source_publication_id: str,
        target_publication_id: str,
        chunks: Sequence[Chunk],
    ) -> None:
        source = self.publications[source_publication_id]
        target = self.publications.setdefault(target_publication_id, {})
        for chunk in chunks:
            target[chunk.id] = source[chunk.id]

    async def retrieve_publication(
        self, publication_id: str, chunk_ids: Sequence[str]
    ) -> tuple[fakes.VectorRow, ...]:
        """Exercise the fake's read boundary instead of inspecting its backing dictionary."""
        self.retrieval_probes += 1
        rows = self.publications.get(publication_id, {})
        return tuple(row for chunk_id in chunk_ids if (row := rows.get(chunk_id)) is not None)


class CountingPlaintextParser(PlaintextParser):
    """Count local parsing without retaining any document identity or text."""

    def __init__(self, config: PlaintextConfig) -> None:
        super().__init__(config)
        self.calls = 0
        self.active_calls = 0
        self.peak_active_calls = 0
        self.peak_input_bytes = 0

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        self.calls += 1
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        self.peak_input_bytes = max(self.peak_input_bytes, len(raw.as_bytes()))
        try:
            async for block in super().parse(raw):
                yield block
        finally:
            self.active_calls -= 1


class ObservedRebuildStore(SqliteRebuildStore):
    """Records externally visible publication pointer changes at the atomic store boundary."""

    def __init__(self, *args: Any, live_store: SqliteDocStore, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._live_store = live_store
        self.publication_transactions = 0
        self.publication_pointer_transitions = 0

    @override
    async def publish_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        before = {
            document.id: document.publication_id
            for document in await self._live_store.list_documents(limit=1_000)
        }
        published = await super().publish_generation(
            generation_id,
            owner=owner,
            lease_generation=lease_generation,
            now=now,
        )
        after = {
            document.id: document.publication_id
            for document in await self._live_store.list_documents(limit=1_000)
        }
        self.publication_transactions += 1
        self.publication_pointer_transitions += sum(
            before.get(document_id) != publication_id
            for document_id, publication_id in after.items()
        )
        return published


def _pipeline(
    docs: SqliteDocStore,
    vectors: LanceVectorStore,
    blobs: BlobStore | None = None,
    embedder: HashEmbedder | None = None,
    *,
    durable_acquisition: bool = False,
) -> IngestPipeline:
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=docs,
        acquisitions=docs if durable_acquisition else None,
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
    doomed_physical = vector_id(doomed_document.publication_id, doomed_chunk.id)
    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM workspaces WHERE id = 'doomed'"))

    await store.delete_document(document.id)
    tombstones = set(await store.take_tombstones(20))
    assert physical <= tombstones
    async with engine.connect() as connection:
        all_tombstones = set(
            (await connection.execute(text("SELECT chunk_id FROM vector_tombstones"))).scalars()
        )
    assert doomed_physical in all_tombstones
    assert doomed_physical not in tombstones, "one workspace cannot claim another's cleanup"
    await sweep_vectors(store, vectors)
    assert await vectors.count() == 1
    await vectors.delete_chunks([doomed_physical])
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
    assert legacy_id not in await store.take_tombstones(20)
    async with engine.connect() as connection:
        unowned = (
            await connection.execute(
                text(
                    "SELECT count(*) FROM vector_tombstones "
                    "WHERE chunk_id = :chunk_id AND workspace_id IS NULL"
                ),
                {"chunk_id": legacy_id},
            )
        ).scalar_one()
    assert unowned == 1, "an unattributable legacy cleanup record must remain protected"
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


async def test_journal_capacity_refusal_is_retryable_without_partial_acknowledgment(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    limited = SqliteDocStore(engine, max_journal_records=1)
    await limited.ensure_workspace()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=limited,
        acquisitions=limited,
        blobs=BlobStore(engine, data_dir, min_disk_headroom_bytes=1),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
    )

    report = await pipeline.run(
        fakes.DictConnector(
            {
                "private-source-one": "private body one",
                "private-source-two": "private body two",
            }
        )
    )

    assert report.error_type == "CapacityRefusedError"
    assert report.enumeration_completed is False
    assert report.watermark_advanced is False
    assert await limited.get_watermark("memory") is None
    runs = await limited.latest_unsettled_acquisition_run("memory")
    assert runs is not None
    records = await limited.list_acquisition_records(runs.id)
    assert len(records) == 1, "the refused discovery identity was never acknowledged"
    metadata = await limited.connector_metadata("memory")
    rendered = json.dumps(metadata, sort_keys=True)
    assert "journal_records" in rendered
    assert '"limit": 1' in rendered
    for private in (
        "private-source-one",
        "private-source-two",
        "private body one",
        "private body two",
    ):
        assert private not in rendered


async def test_blob_admission_refusal_persists_safe_retry_without_watermark(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(
            engine,
            data_dir,
            min_disk_headroom_bytes=1,
            max_acquired_blob_backlog_bytes=0,
        ),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
    )

    report = await pipeline.run(
        fakes.DictConnector({"private-capacity-source": "private capacity body"})
    )

    assert report.error_type == "CapacityRefusedError"
    assert report.enumeration_completed is True
    assert report.watermark_advanced is False
    assert await store.get_watermark("memory") is None
    run = await store.latest_unsettled_acquisition_run("memory")
    assert run is not None
    record = (await store.list_acquisition_records(run.id))[0]
    assert record.state is AcquisitionRecordState.RETRY
    assert record.blob_ref is None
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == "capacity"
    rendered = json.dumps(await store.connector_metadata("memory"), sort_keys=True)
    assert "acquired_blob_backlog_bytes" in rendered
    assert "private-capacity-source" not in rendered
    assert "private capacity body" not in rendered


async def test_capacity_only_exception_group_releases_lease_for_immediate_retry(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    refusal = CapacityRefusedError(
        CapacityDiagnostic(
            resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
            limit=10,
            used=10,
            requested=1,
        )
    )

    class GroupingPipeline(IngestPipeline):
        refusing = True

        @override
        async def _drive(self, run: Any) -> None:
            if self.refusing:
                raise ExceptionGroup("capacity workers", [refusal, refusal])
            await super()._drive(run)

    chunker = fakes.BlockChunker()
    pipeline = GroupingPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir, min_disk_headroom_bytes=1),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )
    connector = fakes.DictConnector({})

    refused = await pipeline.run(connector)
    assert refused.error_type == "CapacityRefusedError"
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert durable.lease_owner is None

    pipeline.refusing = False
    resumed = await pipeline.run(connector)
    assert resumed.error == ""


async def test_busy_terminal_release_is_fenced_for_immediate_pipeline_retry(
    engine: AsyncEngine,
    data_dir: Path,
    tmp_path: Path,
) -> None:
    class BusyTerminalStore(SqliteDocStore):
        fail_terminal = True
        terminal_call = False

        @override
        async def _begin_capacity_guard(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, session: AsyncSession
        ) -> None:
            if self.terminal_call:
                error = sqlite3.OperationalError("UPDATE private terminal metadata")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise error
            await super()._begin_capacity_guard(session)

        @override
        async def record_acquisition_run_metadata(
            self,
            run_id: str,
            owner: str,
            generation: int,
            *,
            now: datetime,
            updates: Mapping[str, Any],
            release: bool,
        ) -> bool:
            self.terminal_call = self.fail_terminal and release
            try:
                return await super().record_acquisition_run_metadata(
                    run_id,
                    owner,
                    generation,
                    now=now,
                    updates=updates,
                    release=release,
                )
            finally:
                if self.terminal_call:
                    self.fail_terminal = False
                self.terminal_call = False

    store = BusyTerminalStore(engine)
    await store.ensure_workspace()

    class RetryPipeline(IngestPipeline):
        seed_private_glossary = True

        @override
        async def _drive(self, run: Any) -> None:
            if self.seed_private_glossary:
                run.report.glossary_failures.append(
                    "private-document-id from private-source-id: glossary detection failed for "
                    "https://private.invalid/source?token=fake-secret-cinder"
                )
                return
            await super()._drive(run)

    chunker = fakes.BlockChunker()
    pipeline = RetryPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir, min_disk_headroom_bytes=1),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )
    connector = fakes.DictConnector({"public-retry": "public body"})

    first = await pipeline.run(connector)
    durable = await store.latest_unsettled_acquisition_run(connector.name)

    assert first.error_type == "StorageBusyError"
    assert first.glossary_failures == []
    assert durable is not None
    assert durable.lease_owner is not None, "the injected external writer prevented persistence"

    backend = FakeBackend()
    backend.ingestion_.report = first
    backend.settings.connectors[connector.name] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )
    service = ApplicationService(backend)
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        envelope = await control.connect(
            path,
            control.Invoke(op="connector_sync", arguments={"name": connector.name, "limit": None}),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    assert envelope["ok"] is False
    assert cast("dict[str, Any]", envelope["error"])["type"] == "StorageBusyError"
    assert cast("dict[str, Any]", envelope["data"])["retry_required"] is True
    public = json.dumps(
        {"report": first.as_metadata(), "envelope": envelope}, sort_keys=True
    ).lower()
    for private in (
        "private-document-id",
        "private-source-id",
        "private.invalid",
        "fake-secret-cinder",
        "token=",
    ):
        assert private not in public

    pipeline.seed_private_glossary = False
    resumed = await pipeline.run(connector)
    assert resumed.error == ""
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


async def test_mixed_exception_group_remains_a_crash_and_keeps_lease_fence(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    refusal = CapacityRefusedError(
        CapacityDiagnostic(
            resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
            limit=10,
            used=10,
            requested=1,
        )
    )

    class MixedFailurePipeline(IngestPipeline):
        @override
        async def _drive(self, run: Any) -> None:
            del run
            raise ExceptionGroup("mixed workers", [refusal, RuntimeError("database failed")])

    chunker = fakes.BlockChunker()
    pipeline = MixedFailurePipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir, min_disk_headroom_bytes=1),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )
    connector = fakes.DictConnector({})

    crashed = await pipeline.run(connector)
    assert crashed.error_type == "RuntimeError"
    assert crashed.error_message == "database failed"
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert durable.lease_owner is not None
    with pytest.raises(RuntimeError, match="could not be claimed"):
        await pipeline.run(connector)


async def test_member_capacity_refusal_resumes_indexing_by_reexpanding_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class RefusingMemberBlobs(BlobStore):
        refusing = True

        @override
        async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
            if self.refusing and data == b"private beta body":
                raise CapacityRefusedError(
                    CapacityDiagnostic(
                        resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
                        limit=100,
                        used=90,
                        requested=20,
                    )
                )
            return await super().retain(data, media_type)

    blobs = RefusingMemberBlobs(engine, data_dir, min_disk_headroom_bytes=1)
    chunker = fakes.BlockChunker()
    lease_clock = fakes.ManualLeaseClock()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=blobs,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"archive": fakes.FakeArchive(), "lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["archive", "lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
        acquisition_clock=lease_clock,
    )
    connector = fakes.DictConnector(
        {"private-bundle": "one=alpha\ntwo=private beta body\nthree=gamma"}
    )
    connector.media_types["private-bundle"] = fakes.CONTAINER_MEDIA_TYPE

    refused = await pipeline.run(connector)

    assert refused.error_type == "CapacityRefusedError"
    assert refused.watermark_advanced is False
    assert await store.get_watermark(connector.name) is None
    run = await store.latest_unsettled_acquisition_run(connector.name)
    assert run is not None
    record = (await store.list_acquisition_records(run.id))[0]
    assert record.state is AcquisitionRecordState.INDEXING
    assert await store.find_document("memory", "private-bundle") is not None
    assert await store.find_document("memory", "private-bundle!/one") is not None
    assert await store.find_document("memory", "private-bundle!/two") is None
    fetched = list(connector.fetches)
    rendered = repr(refused.as_metadata())
    assert "private-bundle" not in rendered
    assert "private beta body" not in rendered

    blobs.refusing = False
    lease_clock.advance(301)
    resumed = await pipeline.run(connector)

    assert resumed.error == ""
    assert connector.fetches == fetched, "retry must derive from the retained snapshot"
    assert await store.find_document("memory", "private-bundle!/two") is not None
    assert await store.find_document("memory", "private-bundle!/three") is not None
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


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
                    raise CursorExpiredError(
                        "the search cursor expired at https://private.invalid/?token=fake-secret"
                    )

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
    assert "fake-secret" not in str(after_failure)
    assert "private.invalid" not in failed.error

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


async def test_acquire_only_promotes_then_an_ordinary_sync_resumes_entirely_offline(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """Acquire-only is a durable boundary, not a mode that loses the local second half."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(
        store,
        vectors,
        blobs=BlobStore(engine, data_dir),
        durable_acquisition=True,
    )

    class SnapshotConnector(fakes.DictConnector):
        @property
        @override
        def watermark(self) -> Watermark:
            return Watermark(value="snapshot-3", observed_at=datetime(2026, 8, 15, tzinfo=UTC))

    connector = SnapshotConnector(
        {
            "page-0001": "synthetic alpha",
            "page-0002": "synthetic beta",
            "page-0003": "synthetic gamma",
        },
        name="synthetic-wiki",
    )

    acquired = await pipeline.run(connector, acquire_only=True)
    source_calls = tuple(connector.fetches)
    promoted = await store.latest_unsettled_acquisition_run(connector.name)

    assert acquired.enumeration_completed
    assert acquired.watermark_advanced
    assert acquired.snapshot_completeness == "complete"
    assert acquired.pending_derivation
    assert acquired.derivation_deferred
    assert not acquired.retry_required
    assert acquired.lifecycle_metadata()["outcome"] == "deferred"
    assert acquired.indexed == 0
    assert len(source_calls) == 3
    assert promoted is not None
    assert promoted.state is AcquisitionRunState.INDEXING

    resumed = await pipeline.run(connector)

    assert tuple(connector.fetches) == source_calls
    assert resumed.indexed == 3
    assert not resumed.pending_derivation
    assert len(await store.list_documents()) == 3
    assert await store.latest_unsettled_acquisition_run(connector.name) is None

    reused = await pipeline.run(connector)
    lifecycle = reused.lifecycle_metadata()
    assert tuple(connector.fetches) == source_calls
    assert lifecycle["acquired_items"] == 0
    assert lifecycle["reused_items"] == 3
    await vectors.teardown()


async def test_acquire_only_never_derives_a_strict_incomplete_snapshot_prefix(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """One retained member cannot escape before strict snapshot promotion succeeds."""
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(
        store,
        vectors,
        blobs=BlobStore(engine, data_dir),
        durable_acquisition=True,
    )
    connector = fakes.DictConnector(
        {"accepted": "retained body", "refused": "private body"}, name="strict-acquire-only"
    )
    connector.fail_fetch.add("refused")

    report = await pipeline.run(connector, acquire_only=True)
    unsettled = await store.latest_unsettled_acquisition_run(connector.name)

    assert report.snapshot_omissions == 1
    assert report.retry_required
    assert report.pending_derivation
    assert not report.derivation_deferred
    assert report.indexed == 0
    assert await store.list_documents() == []
    assert unsettled is not None
    assert unsettled.state is AcquisitionRunState.ACQUIRING
    await vectors.teardown()


async def test_bounded_acquire_only_prefix_is_not_reported_as_offline_derivation(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(HashEmbedder().fingerprint)
    pipeline = _pipeline(
        store,
        vectors,  # pyright: ignore[reportArgumentType]
        blobs=BlobStore(engine, data_dir),
        durable_acquisition=True,
    )

    report = await pipeline.run(
        fakes.DictConnector({"one": "alpha", "two": "beta"}),
        limit=1,
        acquire_only=True,
    )
    lifecycle = report.lifecycle_metadata()

    assert report.limited
    assert report.pending_derivation
    assert not report.derivation_deferred
    assert lifecycle["phase"] == "acquiring"
    assert lifecycle["outcome"] == "bounded"
    assert lifecycle["can_continue_offline"] is False
    assert await store.count_documents() == 0


async def test_real_confluence_pages_admit_10251_records_to_true_end(
    engine: AsyncEngine,
) -> None:
    """The real HTTP/page adapter and SQLite journal cross forty exact 250 boundaries."""

    class EnumerationObservedError(RuntimeError):
        pass

    clock = ConfluenceClock()

    class MeasuredJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine, max_journal_records=20_000)
            self.batch_sizes: list[int] = []
            self.scalar_appends = 0
            self.completions = 0

        @override
        async def append_acquisition_record(
            self,
            run_id: str,
            sequence: int,
            source: AcquisitionSource,
            *,
            lease_owner: str,
            lease_generation: int,
            now: datetime,
        ) -> AcquisitionRecord:
            self.scalar_appends += 1
            return await super().append_acquisition_record(
                run_id,
                sequence,
                source,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                now=now,
            )

        @override
        async def append_acquisition_records(
            self,
            run_id: str,
            sequence: int,
            sources: Sequence[AcquisitionSource],
            *,
            lease_owner: str,
            lease_generation: int,
            now: datetime,
        ) -> Sequence[AcquisitionRecord]:
            admitted = await super().append_acquisition_records(
                run_id,
                sequence,
                sources,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                now=now,
            )
            self.batch_sizes.append(len(sources))
            clock.advance(0.1)
            return admitted

        @override
        async def complete_acquisition_enumeration(
            self,
            run_id: str,
            candidate_watermark: Watermark | None,
            *,
            lease_owner: str,
            lease_generation: int,
            now: datetime,
        ) -> AcquisitionRun:
            self.completions += 1
            return await super().complete_acquisition_enumeration(
                run_id,
                candidate_watermark,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                now=now,
            )

    class EnumerationPipeline(IngestPipeline):
        async def _acquire_journal(self, run: object) -> bool:  # type: ignore[override]
            del run
            raise EnumerationObservedError

    total = 10_251
    instance = FakeConfluence(
        pages=[
            FakePage(
                id=f"synthetic-{number:05d}",
                title=f"Public synthetic {number}",
                space="ENG",
            )
            for number in range(total)
        ],
        page_size=250,
    )
    config = cloud_config(
        base_url=instance.base_url,
        spaces=["ENG"],
        include_attachments=False,
        page_size=250,
        cursor_lifetime_seconds=0.2,
    )
    _, client = client_for(instance, config, clock=clock)
    connector = ConfluenceConnector(config, client, name="synthetic-large-wiki")
    store = MeasuredJournal()
    await store.ensure_workspace()
    await connector.setup()
    try:
        report = await EnumerationPipeline(store=store, acquisitions=store).run(
            connector, acquire_only=True
        )
    finally:
        await connector.teardown()

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert report.error_type == "EnumerationObservedError"
    assert durable.discovered_count == total
    assert durable.enumeration_completed_at is not None
    connector_watermark = connector.watermark
    assert connector_watermark is not None
    assert durable.candidate_watermark is not None
    assert durable.candidate_watermark.value == connector_watermark.value
    assert durable.candidate_watermark.metadata == connector_watermark.metadata
    assert store.completions == 1
    assert store.scalar_appends == 0, "native Confluence pages must use atomic batch admission"
    assert store.batch_sizes == [250] * 41 + [1]
    assert max(store.batch_sizes) == 250
    search_requests = [
        request
        for request in instance.requests
        if request.url.path.endswith("/rest/api/content/search")
    ]
    assert len(search_requests) == 42
    assert clock.now == pytest.approx(4.2)


async def test_nonbatched_connector_uses_the_dedicated_scalar_admission(
    engine: AsyncEngine,
) -> None:
    """A singleton compatibility wrapper is not a native source-page capability."""

    class EnumerationObservedError(RuntimeError):
        pass

    class DispatchJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.scalar_appends = 0
            self.batch_appends = 0

        @override
        async def append_acquisition_record(
            self,
            run_id: str,
            sequence: int,
            source: AcquisitionSource,
            *,
            lease_owner: str,
            lease_generation: int,
            now: datetime,
        ) -> AcquisitionRecord:
            self.scalar_appends += 1
            return await super().append_acquisition_record(
                run_id,
                sequence,
                source,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                now=now,
            )

        @override
        async def append_acquisition_records(
            self,
            run_id: str,
            sequence: int,
            sources: Sequence[AcquisitionSource],
            *,
            lease_owner: str,
            lease_generation: int,
            now: datetime,
        ) -> Sequence[AcquisitionRecord]:
            self.batch_appends += 1
            return await super().append_acquisition_records(
                run_id,
                sequence,
                sources,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                now=now,
            )

    class EnumerationPipeline(IngestPipeline):
        async def _acquire_journal(self, run: object) -> bool:  # type: ignore[override]
            del run
            raise EnumerationObservedError

    journal = DispatchJournal()
    await journal.ensure_workspace()
    report = await EnumerationPipeline(store=journal, acquisitions=journal).run(
        fakes.DictConnector({"one": "public one", "two": "public two"}),
        acquire_only=True,
    )

    assert report.error_type == "EnumerationObservedError"
    assert journal.scalar_appends == 2
    assert journal.batch_appends == 0


async def test_cancellation_after_page_commit_does_not_request_the_next_cursor(
    engine: AsyncEngine,
) -> None:
    """The stop barrier is before ``anext`` rather than after a discarded response."""

    class PausedJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.committed = asyncio.Event()
            self.release = asyncio.Event()

        @override
        async def append_acquisition_records(
            self,
            run_id: str,
            sequence: int,
            sources: Sequence[AcquisitionSource],
            *,
            lease_owner: str,
            lease_generation: int,
            now: datetime,
        ) -> Sequence[AcquisitionRecord]:
            admitted = await super().append_acquisition_records(
                run_id,
                sequence,
                sources,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                now=now,
            )
            self.committed.set()
            await self.release.wait()
            return admitted

    class StopObservedPipeline(IngestPipeline):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.stopping = asyncio.Event()

        @override
        async def _stop_within_grace(self, run: Any, stages: asyncio.Task[None]) -> None:
            run.stop.set()
            self.stopping.set()
            await super()._stop_within_grace(run, stages)

    journal = PausedJournal()
    await journal.ensure_workspace()
    pipeline = StopObservedPipeline(store=journal, acquisitions=journal)
    connector = fakes.ExpiringCursorConnector(
        {f"synthetic-{number}": "public" for number in range(4)},
        clock=fakes.ManualClock(),
        page_size=2,
        cursor_lifetime_seconds=60,
    )

    task = asyncio.create_task(pipeline.run(connector, acquire_only=True))
    await journal.committed.wait()
    assert connector.pages_requested == 1
    task.cancel()
    await pipeline.stopping.wait()
    journal.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert connector.pages_requested == 1
    durable = await journal.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    assert durable.discovered_count == 2
    assert durable.enumeration_completed_at is None


async def test_durable_enumeration_finishes_before_slow_indexing_can_age_a_cursor(  # noqa: PLR0915 - one end-to-end evidence record
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The 1,000-document/500 ms acceptance cursor is drained before derivation.

    Ten synthetic page responses each take 19 ms on a manual clock, below the 500 ms cursor
    lifetime. Journal pages and both in-memory hand-offs are smaller than the corpus. The model
    gate proves indexing has not drained anything to make enumeration succeed. All addresses use
    the reserved ``wiki.example.test`` host and no wall clock or network participates.
    """
    clock = fakes.ManualClock()
    embedder = fakes.ClockedGatedEmbedder(clock, seconds_per_document=0.05)
    chunker = fakes.BlockChunker()
    vectors = PublicationMemoryVectors()
    await vectors.ensure_ready(embedder.fingerprint)
    blobs = BlobStore(engine, data_dir)
    parser = CountingPlaintextParser(PlaintextConfig())
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=blobs,
        chunker=chunker,
        embedder=embedder,
        vectors=vectors,
        runner=InProcessRunner({"plaintext": parser}),
        resolve_chain=lambda _: ["plaintext"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
        detect_glossary=False,
    )
    connector = fakes.ExpiringCursorConnector(
        {
            f"synthetic-doc-{number:04d}": f"public synthetic line {number}"
            for number in range(1_000)
        },
        clock=clock,
        page_size=100,
        cursor_lifetime_seconds=0.5,
        response_seconds=0.019,
    )

    task = asyncio.create_task(pipeline.run(connector))
    try:
        await connector.enumeration_completed.wait()
        # **Sixty seconds was a budget calibrated on the wrong machine.** This gate does not open
        # until a document has crossed the whole pipeline, and on the durable path every one of
        # the 1,000 documents is acquired before indexing starts — so the wait is behind 1,000
        # fetches and their retention, not behind one. Measured from `enumeration_completed` to
        # the first arrival: 14.2s on a 16-core developer machine. Against a 60s budget that is
        # a 4x margin, and CI's two-core runners are more than 4x slower at exactly this shape
        # of work. It failed there, reporting zero arrivals.
        #
        # `patience_s` is "how long the test is willing to be wrong for" rather than a timeout on
        # anything under test, and it only elapses when the test is *already* failing — so a
        # generous number costs nothing on a green run and buys back a false failure. What it
        # costs is that a genuine deadlock takes this long to report, which is the right trade
        # for a stage that is legitimately slow.
        await embedder.gate.wait_for(1, patience_s=300)

        durable = await store.latest_unsettled_acquisition_run(connector.name)
        assert durable is not None
        assert durable.discovered_count == 1_000
        assert durable.enumeration_completed_at is not None
        assert durable.candidate_watermark == connector.watermark
        assert connector.pages_requested == 10
        assert clock.now == pytest.approx(0.19)
        assert connector.maximum_cursor_age_seconds == 0
        enumeration_seconds = clock.now
        source_calls_before_derivation = len(connector.fetches)
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
    assert report.indexed == 1_000
    assert len(indexed) == 1_000
    assert await store.count_chunks() == 1_000
    assert await vectors.count() == 1_000
    assert len(connector.fetches) == source_calls_before_derivation == 1_000
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
    records = [record async for record in store.iter_acquisition_records(durable.id)]
    assert {record.state for record in records} == {AcquisitionRecordState.SETTLED}
    blob_refs = {record.blob_ref for record in records if record.blob_ref is not None}
    retained_bytes = sum(
        record.acquired_source.byte_length
        for record in records
        if record.acquired_source is not None
    )
    compressed_bytes = sum(blobs.path_for(blob_ref).stat().st_size for blob_ref in blob_refs)
    derivation_seconds = clock.now - enumeration_seconds
    initial_publications = {
        publication_id: rows for publication_id, rows in vectors.publications.items() if rows
    }
    assert sum(len(rows) for rows in initial_publications.values()) == 1_000

    live_documents = await store.list_documents(limit=1_000)
    corpus_rows = [
        (document, await store.document_chunks(document.id)) for document in live_documents
    ]
    authority = Authority()
    corpus = Corpus(authority, corpus_rows)
    second_embedder = CountingEmbedder(dimension=7)
    second_embedder.fingerprint = second_embedder.fingerprint.model_copy(
        update={"model_id": "synthetic/second-model"}
    )
    parser_calls_before_reembed = parser.calls
    source_calls_before_reembed = len(connector.fetches)
    reembed = await start_reembed(
        "synthetic-second-model",
        owner_token="planning-owner",  # noqa: S106 - synthetic lease identity
        corpus=corpus,
        target=second_embedder.fingerprint,
        journal=authority,
        document_page=37,
        target_batch_tokens=64,
        chunks_per_second=2_000,
    )
    assert reembed.commitment.plan.documents == 1_000
    assert reembed.commitment.plan.chunks == 1_000
    assert reembed.commitment.plan.estimated_seconds == pytest.approx(0.5)

    authority.fail_chunk_checkpoint_once = True
    with pytest.raises(OSError, match="chunk checkpoint crash"):
        await resume_reembed(
            reembed.id,
            owner_token="crashed-owner",  # noqa: S106 - synthetic lease identity
            corpus=corpus,
            embedder=second_embedder,
            journal=authority,
            shadow=authority,
            publisher=authority,
            document_page=37,
            target_batch_tokens=64,
        )
    crashed = await authority.get(reembed.id)
    assert crashed is not None
    assert crashed.state is ReembedState.BUILDING
    assert authority.live.generation_id == "live-old"
    assert len(authority.rows[f"shadow:{reembed.id}"]) == 1

    authority.pause_first_upsert = True
    resume_task = asyncio.create_task(
        resume_reembed(
            reembed.id,
            owner_token="resume-owner",  # noqa: S106 - synthetic lease identity
            corpus=corpus,
            embedder=second_embedder,
            journal=authority,
            shadow=authority,
            publisher=authority,
            document_page=37,
            target_batch_tokens=64,
        )
    )
    await authority.upsert_entered.wait()
    assert authority.live.generation_id == "live-old"
    old_index_rows: list[fakes.VectorRow] = []
    for publication_id, rows in initial_publications.items():
        old_index_rows.extend(await vectors.retrieve_publication(publication_id, tuple(rows)))
    assert len(old_index_rows) == 1_000
    authority.release_upsert.set()
    reembedded = await resume_task
    second_rows = authority.rows[reembedded.shadow_generation_id or ""]
    assert reembedded.state is ReembedState.PUBLISHED
    assert reembedded.chunks_completed == 1_000
    assert len(second_rows) == 1_000
    assert len({stored.chunk.id for stored, _ in second_rows.values()}) == 1_000
    assert set(authority.live_during_upsert) == {"live-old"}
    assert authority.live.generation_id == reembedded.shadow_generation_id
    assert parser.calls == parser_calls_before_reembed
    assert len(connector.fetches) == source_calls_before_reembed
    parser_calls_during_second_model = parser.calls - parser_calls_before_reembed
    assert vectors.retrieval_probes == len(initial_publications)

    reset = await store.reset_derived()
    assert reset.snapshot_items == 1_000
    assert await store.count_chunks() == 0

    installed_parse = parse_fingerprint("plaintext")
    assert installed_parse is not None
    target = RebuildTarget(
        parser_routing="synthetic-plaintext-routing-v1",
        parser_set=(installed_parse.canonical(),),
        chunk_fingerprint=chunker.fingerprint.canonical(),
        embedding_fingerprint=embedder.fingerprint.canonical(),
        embedding_config=embedder.fingerprint.model_dump_json(),
        glossary_fingerprint=glossary_fingerprint(enabled=False, middleware=()).canonical(),
        fts_tokenizer=FTS_TOKENIZER,
        batch_documents=7,
        max_memory_bytes=64 * 1024 * 1024,
        max_temporary_bytes=64 * 1024 * 1024,
    )
    rebuild_store = ObservedRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=blobs,
        vectors=vectors,
        live_store=store,
    )
    rebuild_clock = fakes.ManualLeaseClock()
    rebuilder = build_offline_rebuilder(
        store=rebuild_store,
        blobs=blobs,
        workspace_id=store.workspace_id,
        source=connector.name,
        parser_chain=ParserChain(
            parsers={"plaintext": parser}, chains={fakes.MEDIA_TYPE: ("plaintext",)}
        ),
        routing_identity=target.parser_routing,
        chunker=chunker,
        embedder=embedder,
        vectors=vectors,
        chunk_fingerprint=chunker.fingerprint,
        middleware=MiddlewareRunner(()),
        parse_runner=InProcessRunner(
            {"plaintext": parser}, middleware=MiddlewareRunner(()), chunker=chunker
        ),
        detect_glossary=False,
        clock=rebuild_clock,
    )
    plan = await rebuilder.dry_run(durable.id, target)
    source_calls_before_rebuild = len(connector.fetches)
    parser_calls_before_rebuild = parser.calls
    rebuilt = await rebuilder.run(durable.id, target, owner="synthetic-rebuild")

    assert plan.runnable
    assert plan.documents == 1_000
    assert rebuilt.state is RebuildState.PUBLISHED
    assert rebuilt.documents_built == 1_000
    assert rebuilt.chunks_built == 1_000
    assert len(connector.fetches) == source_calls_before_rebuild
    assert await store.count_documents() == 1_000
    assert await store.count_chunks() == 1_000
    live_vector_count = await vectors.publication_row_count(rebuilt.vector_publication_id)
    rebuilt_documents = await store.list_documents(limit=1_000)
    rebuilt_chunk_ids = {
        chunk.id
        for document in rebuilt_documents
        for chunk in await store.document_chunks(document.id)
    }
    assert live_vector_count == 1_000
    assert await vectors.count() == 2_000, "the old publication remains readable until cleanup"
    retained_old_rows: list[fakes.VectorRow] = []
    for publication_id, rows in initial_publications.items():
        retained_old_rows.extend(await vectors.retrieve_publication(publication_id, tuple(rows)))
    assert len(retained_old_rows) == 1_000
    observed_process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert observed_process_peak_rss > 0
    evidence = {
        "documents": 1_000,
        "cursor_lifetime_seconds": 0.5,
        "enumeration_seconds": enumeration_seconds,
        "maximum_cursor_age_seconds": connector.maximum_cursor_age_seconds,
        "journal_records_per_virtual_second": len(records) / enumeration_seconds,
        "journal_peak_backlog_items": durable.acquired_count - durable.indexed_count,
        "journal_records": len(records),
        "new_body_requests": len(connector.fetches),
        "changed_body_requests": 0,
        "reused_body_requests": 0,
        "failed_body_requests": report.unrecorded,
        "retained_source_bytes": retained_bytes,
        "compressed_blob_bytes": compressed_bytes,
        "parse_documents_per_virtual_second": len(records) / derivation_seconds,
        "chunks_per_virtual_second": report.indexed / derivation_seconds,
        "embeds_per_virtual_second": report.indexed / derivation_seconds,
        "fetch_queue_peak_items": report.stages.fetch_queue.peak_depth,
        "parse_queue_peak_items": report.stages.parse_queue.peak_depth,
        "peak_retained_bodies": report.stages.peak_bodies,
        "reembed_peak_memory_estimate_bytes": reembed.commitment.plan.peak_memory_bytes,
        "reembed_temp_disk_estimate_bytes": reembed.commitment.plan.temporary_disk_bytes,
        "observed_peak_parser_input_bytes": parser.peak_input_bytes,
        "observed_peak_parser_calls": parser.peak_active_calls,
        "observed_peak_vector_upsert_items": vectors.peak_upsert_items,
        "observed_process_peak_rss_native_units": observed_process_peak_rss,
        "rebuild_memory_bound_bytes": target.max_memory_bytes,
        "rebuild_temp_disk_bound_bytes": target.max_temporary_bytes,
        "source_calls_during_derivation": len(connector.fetches) - source_calls_before_derivation,
        "unique_source_documents": len(indexed),
        "unique_source_blobs": len(blob_refs),
        "unique_live_documents": len({document.id for document in rebuilt_documents}),
        "unique_live_chunks": len(rebuilt_chunk_ids),
        "unique_vectors": live_vector_count,
        "offline_rebuild_documents": rebuilt.documents_built,
        "source_calls_during_rebuild": len(connector.fetches) - source_calls_before_rebuild,
        "parser_calls_during_local_snapshot_rebuild": parser.calls - parser_calls_before_rebuild,
        "source_calls_during_second_model": len(connector.fetches) - source_calls_before_reembed,
        "parser_calls_during_second_model": parser_calls_during_second_model,
        "second_model_vectors": len(second_rows),
        "crash_checkpoint_rows": 1,
        "crash_checkpoint_state": crashed.state.value,
        "old_index_reads_before_flip": len(old_index_rows),
        "old_publication_reads_after_flip": len(retained_old_rows),
        "publication_transactions_observed": rebuild_store.publication_transactions,
        "publication_pointer_transitions": rebuild_store.publication_pointer_transitions,
        "second_model_published": reembedded.state is ReembedState.PUBLISHED,
        "published_generation": rebuilt.state is RebuildState.PUBLISHED,
        "watermark_committed": report.watermark_advanced,
        "snapshot_completeness": report.snapshot_completeness,
    }
    assert evidence == {
        "documents": 1_000,
        "cursor_lifetime_seconds": 0.5,
        "enumeration_seconds": pytest.approx(0.19),
        "maximum_cursor_age_seconds": 0,
        "journal_records_per_virtual_second": pytest.approx(1_000 / 0.19),
        "journal_peak_backlog_items": 1_000,
        "journal_records": 1_000,
        "new_body_requests": 1_000,
        "changed_body_requests": 0,
        "reused_body_requests": 0,
        "failed_body_requests": 0,
        "retained_source_bytes": retained_bytes,
        "compressed_blob_bytes": compressed_bytes,
        "parse_documents_per_virtual_second": pytest.approx(20),
        "chunks_per_virtual_second": pytest.approx(20),
        "embeds_per_virtual_second": pytest.approx(20),
        "fetch_queue_peak_items": report.stages.fetch_queue.peak_depth,
        "parse_queue_peak_items": report.stages.parse_queue.peak_depth,
        "peak_retained_bodies": report.stages.peak_bodies,
        "reembed_peak_memory_estimate_bytes": reembed.commitment.plan.peak_memory_bytes,
        "reembed_temp_disk_estimate_bytes": reembed.commitment.plan.temporary_disk_bytes,
        "observed_peak_parser_input_bytes": parser.peak_input_bytes,
        "observed_peak_parser_calls": 1,
        "observed_peak_vector_upsert_items": 1,
        "observed_process_peak_rss_native_units": observed_process_peak_rss,
        "rebuild_memory_bound_bytes": 64 * 1024 * 1024,
        "rebuild_temp_disk_bound_bytes": 64 * 1024 * 1024,
        "source_calls_during_derivation": 0,
        "unique_source_documents": 1_000,
        "unique_source_blobs": 1_000,
        "unique_live_documents": 1_000,
        "unique_live_chunks": 1_000,
        "unique_vectors": 1_000,
        "offline_rebuild_documents": 1_000,
        "source_calls_during_rebuild": 0,
        "parser_calls_during_local_snapshot_rebuild": 1_000,
        "source_calls_during_second_model": 0,
        "parser_calls_during_second_model": 0,
        "second_model_vectors": 1_000,
        "crash_checkpoint_rows": 1,
        "crash_checkpoint_state": "building",
        "old_index_reads_before_flip": 1_000,
        "old_publication_reads_after_flip": 1_000,
        "publication_transactions_observed": 1,
        "publication_pointer_transitions": 1_000,
        "second_model_published": True,
        "published_generation": True,
        "watermark_committed": True,
        "snapshot_completeness": "complete",
    }
    rendered_evidence = json.dumps(evidence, sort_keys=True)
    assert {"source_id", "uri", "title", "content", "secret"}.isdisjoint(evidence)
    for private_value in ("wiki.example.test", "public synthetic line", "token="):
        assert private_value not in rendered_evidence.lower()


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


@pytest.mark.parametrize("release_busy", [False, True])
async def test_crash_after_enumeration_preserves_the_marker_records_and_candidate(
    engine: AsyncEngine,
    data_dir: Path,
    release_busy: bool,
) -> None:
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    class GatedJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.reader_arrived = asyncio.Event()
            self.release_reader = asyncio.Event()
            self.releasing = False

        @override
        async def _begin_capacity_guard(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, session: AsyncSession
        ) -> None:
            if self.releasing:
                error = sqlite3.OperationalError("UPDATE private cancellation cleanup")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise error
            await super()._begin_capacity_guard(session)

        @override
        async def release_acquisition_lease(
            self,
            run_id: str,
            owner: str,
            generation: int,
            *,
            now: datetime,
        ) -> bool:
            self.releasing = release_busy
            try:
                return await super().release_acquisition_lease(run_id, owner, generation, now=now)
            finally:
                self.releasing = False

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
    assert (durable.lease_owner is not None) is release_busy
    records = await store.list_acquisition_records(durable.id)
    assert len(records) == 10
    assert {record.state for record in records} == {AcquisitionRecordState.DISCOVERED}
    assert await store.list_documents() == []
    assert await store.get_watermark(connector.name) is None

    pages_before_resume = connector.pages_requested
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
    vectors = fakes.MemoryVectors()
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
    documents = await store.list_documents()
    chunks = {
        document.id: [chunk.id for chunk in await store.document_chunks(document.id)]
        for document in documents
    }
    vector_ids = set(vectors.rows)
    fetches = list(connector.fetches)

    repeated = await pipeline.run(connector)

    assert repeated.skipped_version == 2
    assert connector.fetches == fetches
    assert [document.id for document in await store.list_documents()] == [
        document.id for document in documents
    ]
    assert {
        document.id: [chunk.id for chunk in await store.document_chunks(document.id)]
        for document in documents
    } == chunks
    assert set(vectors.rows) == vector_ids


async def test_enumeration_renews_its_lease_while_the_source_cursor_is_blocked(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The heartbeat is independent of source yields, so a slow page cannot age ownership."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    class ObservedJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.renewed = asyncio.Event()

        @override
        async def renew_acquisition_lease(
            self,
            run_id: str,
            owner: str,
            generation: int,
            *,
            now: datetime,
            expires_at: datetime,
        ) -> bool:
            result = await super().renew_acquisition_lease(
                run_id, owner, generation, now=now, expires_at=expires_at
            )
            self.renewed.set()
            return result

    class PausedDiscovery(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__({"public-page": "public text"}, name="slow-enumeration")
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.entered.set()
            await self.release.wait()
            async for discovered in super().discover(watermark):
                yield discovered

    store = ObservedJournal()
    await store.ensure_workspace()
    connector = PausedDiscovery()
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
        acquisition_lease_s=1,
        acquisition_clock=fakes.ManualLeaseClock(),
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await connector.entered.wait()
    await asyncio.wait_for(store.renewed.wait(), timeout=2)
    connector.release.set()
    report = await task

    assert report.indexed == 1
    assert report.error_type == ""


async def test_synchronous_document_preparation_does_not_block_lease_renewal(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The synchronous twin of the test above, and the one the async version could not catch.

    Enumeration blocks on an ``await``, so the heartbeat coroutine keeps its turn on the loop
    and that test passes whatever preparation does. Chunking, exact token counting and
    glossary detection are compute-bound Python that never awaits, and they used to run on the
    loop itself — so the heartbeat could not run *at all* while a document was being prepared,
    however long that took. Measured on one synthetic 2 MiB block with the production
    vocabulary: 513s of preparation against a 300s lease, zero of five renewals fired, and the
    first thing to reach the loop afterwards was a generation-fenced renewal that had already
    lost the run. The snapshot was complete and resumable; the sync was dead.

    Held on a ``threading.Event`` rather than a sleep, and released on the renewal *count*
    rather than a wall clock: the barrier stays shut until the heartbeat has fired twice, so a
    loop that cannot run the heartbeat does not make this flaky, it makes it hang and fail.
    """
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    renewals_wanted = 2

    class CountingJournal(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.renewals = 0
            self.renewed_enough = asyncio.Event()

        @override
        async def renew_acquisition_lease(
            self,
            run_id: str,
            owner: str,
            generation: int,
            *,
            now: datetime,
            expires_at: datetime,
        ) -> bool:
            result = await super().renew_acquisition_lease(
                run_id, owner, generation, now=now, expires_at=expires_at
            )
            self.renewals += 1
            if self.renewals >= renewals_wanted:
                self.renewed_enough.set()
            return result

    class HeldChunker(fakes.BlockChunker):
        """A chunker that blocks the way the real one does: synchronously, without awaiting."""

        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        @override
        def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
            self.entered.set()
            self.release.wait(timeout=30)
            return super().chunk(document, blocks)

    store = CountingJournal()
    await store.ensure_workspace()
    chunker = HeldChunker()
    connector = fakes.DictConnector(
        {"public-held-document": "public synthetic line"}, name="held-preparation-source"
    )
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
        acquisition_lease_s=1,
        acquisition_clock=fakes.ManualLeaseClock(),
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await asyncio.wait_for(asyncio.to_thread(chunker.entered.wait, 10), timeout=15)

    # The loop is expected to be fully alive while the synchronous stage is held: the
    # heartbeat renews, and a bounded durable status read answers.
    await asyncio.wait_for(store.renewed_enough.wait(), timeout=15)
    live = await asyncio.wait_for(store.latest_unsettled_acquisition_run(connector.name), timeout=5)
    assert live is not None, "a bounded status read has to answer while preparation is held"

    chunker.release.set()
    report = await asyncio.wait_for(task, timeout=30)

    assert store.renewals >= renewals_wanted
    assert report.indexed == 1
    assert report.error_type == ""


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
    assert report.error_type == "AcquisitionLeaseLostError"
    assert stored is None, "the expired generation made no document revision servable"


async def test_takeover_after_the_pipeline_check_is_refused_inside_publication_transaction(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The relational write validates ownership again after any awaited dispatch gap."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    class GatedPublicationStore(SqliteDocStore):
        def __init__(self) -> None:
            super().__init__(engine)
            self.arrived = asyncio.Event()
            self.release = asyncio.Event()

        @override
        async def _fence_acquisition_mutation(
            self, session: AsyncSession, fence: AcquisitionFence
        ) -> None:
            self.arrived.set()
            await self.release.wait()
            await super()._fence_acquisition_mutation(session, fence)

    class EmptyChunker(fakes.BlockChunker):
        @override
        def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
            del document, blocks
            return []

    store = GatedPublicationStore()
    await store.ensure_workspace()
    lease_clock = fakes.ManualLeaseClock()
    connector = fakes.DictConnector(
        {"public-race": "public synthetic line"}, name="transaction-fenced-source"
    )
    chunker = EmptyChunker()
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
        acquisition_lease_s=1,
        acquisition_clock=lease_clock,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await store.arrived.wait()
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    lease_clock.advance(2)
    successor = await store.claim_acquisition_run(
        durable.id,
        "successor-after-check",
        now=lease_clock(),
        expires_at=lease_clock() + timedelta(seconds=1),
    )
    assert successor is not None
    store.release.set()

    report = await task

    assert report.error_type == "AcquisitionLeaseLostError"
    assert await store.find_document(connector.name, "public-race") is None


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

    assert report.error_type == "AcquisitionLeaseLostError"
    assert after == before, "the stale worker changed last-seen/version metadata or status"
    if unchanged:
        records = await store.list_acquisition_records(durable.id)
        assert [record.state for record in records] == [AcquisitionRecordState.ACQUIRING]


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

    assert report.error_type == "AcquisitionLeaseLostError"
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
        async def fenced_stage_vectors(
            self,
            fence: AcquisitionFence,
            publication_id: str,
            chunks: Sequence[Chunk],
            *,
            expected_reset_epoch: int | None = None,
        ) -> None:
            await super().fenced_stage_vectors(
                fence,
                publication_id,
                chunks,
                expected_reset_epoch=expected_reset_epoch,
            )
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

    assert report.error_type == "AcquisitionLeaseLostError"
    assert vectors.rows == {}, "the stale worker wrote vectors after its staged-store await"

    fetches = list(connector.fetches)
    connector.fail_fetch.add("public-vector-document")
    lease_clock.advance(301)
    resumed = await pipeline.run(connector)

    assert connector.fetches == fetches
    assert resumed.indexed == 1
    assert vectors.rows, "resume did not rebuild the staged publication from its retained blob"


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
    assert durable.lease_owner is None
    assert await store.get_watermark(connector.name) is None

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


async def test_first_index_and_reparse_share_one_retained_snapshot_without_re_retention(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """Acquisition and reparse both derive from one blob instead of writing it again."""

    class CountingBlobs(BlobStore):
        def __init__(self) -> None:
            super().__init__(engine, data_dir)
            self.retain_calls = 0
            self.acquisition_calls = 0

        @override
        async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
            self.retain_calls += 1
            return await super().retain(data, media_type)

        @override
        async def retain_acquisition(
            self, key: str, raw: RawDocument
        ) -> tuple[Retention, AcquiredSource]:
            self.acquisition_calls += 1
            return await super().retain_acquisition(key, raw)

    blobs = CountingBlobs()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
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
    )
    connector = fakes.DictConnector(
        {"public-shared-snapshot": "public retained line"}, name="shared-snapshot-source"
    )

    first = await pipeline.run(connector)
    document = await store.find_document(connector.name, "public-shared-snapshot")

    assert first.indexed == 1
    assert document is not None
    assert document.original_ref is not None
    assert blobs.acquisition_calls == 1
    assert blobs.retain_calls == 0, "blob-fed first indexing retained its input a second time"

    reparsed = await re_parse([document], pipeline=pipeline, blobs=blobs)

    assert reparsed.documents == 1
    assert blobs.retain_calls == 0, "reparse retained an already-retained snapshot again"


async def test_failed_local_derivation_retries_from_blob_without_source_access(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """A failed embed remains a blob-backed RETRY and a later invocation needs no fetch."""

    class OnceFailingEmbedder(HashEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        @override
        async def embed(self, texts: Sequence[str]) -> list[Vector]:
            if self.fail:
                self.fail = False
                raise RuntimeError("synthetic local embed failure")
            return await super().embed(texts)

    connector = fakes.DictConnector(
        {"public-retry-document": "original public line"}, name="local-retry-source"
    )
    blobs = BlobStore(engine, data_dir)
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()

    def pipeline(embedder: HashEmbedder) -> IngestPipeline:
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
            acquisition_lease_s=1,
            acquisition_clock=lease_clock,
            detect_glossary=False,
        )

    seeded = await pipeline(HashEmbedder()).run(connector)
    original = await store.find_document(connector.name, "public-retry-document")
    assert seeded.indexed == 1
    assert original is not None

    connector.documents["public-retry-document"] = "replacement public line"
    embedder = OnceFailingEmbedder()
    first = await pipeline(embedder).run(connector)
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.RETRY, (
        first.error_type,
        first.error_message,
        first.error,
    )
    assert record.blob_ref is not None
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == "embed_failed"
    assert first.pending_derivation
    assert first.retry_required
    preserved = await store.find_document(connector.name, "public-retry-document")
    assert preserved is not None
    assert preserved.status is DocumentStatus.INDEXED
    assert preserved.publication_id == original.publication_id
    assert preserved.content_hash == original.content_hash
    fetches = list(connector.fetches)

    connector.fail_fetch.add("public-retry-document")
    resumed = await pipeline(embedder).run(connector)

    assert connector.fetches == fetches
    assert resumed.indexed == 1
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


async def test_offline_snapshot_finishes_from_blob_after_source_authorization_is_removed(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    tmp_path: Path,
) -> None:
    """The offline connector is only an acquisition source; derivation reopens its blob."""
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415
        ConfluenceSnapshotConnector,
    )

    class AuthorizedSnapshot(ConfluenceSnapshotConnector):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.authorized = True
            self.fetches = 0

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            self.fetches += 1
            if not self.authorized:
                raise SessionExpiredError("synthetic snapshot authorization removed")
            return await super().fetch(ref)

    class GatedBlobRead(BlobStore):
        def __init__(self) -> None:
            super().__init__(engine, data_dir)
            self.reading = asyncio.Event()
            self.release_read = asyncio.Event()

        @override
        async def get(self, digest: str) -> bytes | None:
            data = await super().get(digest)
            self.reading.set()
            await self.release_read.wait()
            return data

    corpus = tmp_path / "public-snapshot"
    corpus.mkdir()
    _confluence_snapshot(corpus, version=7)
    connector = AuthorizedSnapshot(corpus)
    blobs = GatedBlobRead()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=blobs,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"html": fakes.LineParser()}),
        resolve_chain=lambda _: ["html"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    )

    task = asyncio.create_task(pipeline.run(connector))
    await blobs.reading.wait()
    fetches = connector.fetches
    connector.authorized = False
    blobs.release_read.set()
    report = await task

    assert report.indexed == 1
    assert connector.fetches == fetches == 1
    stored = await store.find_document(connector.name, "123456")
    assert stored is not None
    assert stored.original_ref is not None
    assert stored.provenance is not None


async def test_container_expansion_runs_from_the_retained_parent_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The source fetches one container; its locally expanded members never touch the source."""
    connector = fakes.DictConnector(
        {"public-bundle": "one=alpha\ntwo=beta"}, name="retained-container-source"
    )
    connector.media_types["public-bundle"] = fakes.CONTAINER_MEDIA_TYPE
    chunker = fakes.BlockChunker()
    report = await IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"archive": fakes.FakeArchive(), "lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["archive", "lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    ).run(connector)

    container = await store.find_document(connector.name, "public-bundle")
    first = await store.find_document(connector.name, "public-bundle!/one")
    second = await store.find_document(connector.name, "public-bundle!/two")

    assert connector.fetches == ["public-bundle"]
    assert report.indexed == 2
    assert report.expanded == 2
    assert container is not None
    assert container.status is DocumentStatus.CONTAINER
    assert first is not None
    assert first.original_ref is not None
    assert second is not None
    assert second.original_ref is not None


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        (None, AcquisitionFailureCode.SNAPSHOT_MISSING),
        (b"corrupt local snapshot", AcquisitionFailureCode.SNAPSHOT_CORRUPT),
    ],
)
async def test_unavailable_local_snapshot_is_an_explicit_indexing_retry(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    replacement: bytes | None,
    code: AcquisitionFailureCode,
) -> None:
    class RefusingRead(BlobStore):
        @override
        async def get(self, digest: str) -> bytes | None:
            del digest
            return replacement

    connector = fakes.DictConnector(
        {"public-local-snapshot": "retained public text"}, name=f"local-{code.value}-source"
    )
    chunker = fakes.BlockChunker()
    report = await IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=RefusingRead(engine, data_dir),
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
    assert report.retry_required
    assert record.state is AcquisitionRecordState.RETRY
    assert record.diagnostic is not None
    assert record.diagnostic.stage is AcquisitionStage.INDEXING
    assert record.diagnostic.code is code


async def test_parser_chain_failure_is_classified_as_an_indexing_parse_retry(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    connector = fakes.DictConnector(
        {"public-parse-failure": "synthetic text"}, name="parse-failure-source"
    )
    chunker = fakes.BlockChunker()
    report = await IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=BlobStore(engine, data_dir),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"broken": fakes.ExplodingParser()}),
        resolve_chain=lambda _: ["broken"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        detect_glossary=False,
    ).run(connector)

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert report.retry_required
    assert record.diagnostic is not None
    assert record.diagnostic.stage is AcquisitionStage.INDEXING
    assert record.diagnostic.code is AcquisitionFailureCode.PARSE_FAILED


async def test_failed_container_member_retries_by_reexpanding_the_retained_parent(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """A member failure keeps the parent record retryable and needs no second source fetch."""

    class OnceFailingEmbedder(HashEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        @override
        async def embed(self, texts: Sequence[str]) -> list[Vector]:
            if self.fail:
                self.fail = False
                raise RuntimeError("synthetic member embed failure")
            return await super().embed(texts)

    class CountingBlobs(BlobStore):
        def __init__(self) -> None:
            super().__init__(engine, data_dir)
            self.member_retentions = 0

        @override
        async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
            self.member_retentions += 1
            return await super().retain(data, media_type)

    connector = fakes.DictConnector(
        {"public-retry-bundle": "one=alpha\ntwo=beta"}, name="retry-container-source"
    )
    connector.media_types["public-retry-bundle"] = fakes.CONTAINER_MEDIA_TYPE
    embedder = OnceFailingEmbedder()
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()
    blobs = CountingBlobs()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=blobs,
            chunker=chunker,
            embedder=embedder,
            vectors=fakes.MemoryVectors(),
            runner=InProcessRunner({"archive": fakes.FakeArchive(), "lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["archive", "lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            acquisition_lease_s=1,
            acquisition_clock=lease_clock,
            detect_glossary=False,
        )

    first_report = await pipeline().run(connector)
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.RETRY
    assert record.blob_ref is not None
    assert first_report.retry_required
    assert blobs.member_retentions == 2
    fetches = list(connector.fetches)

    connector.fail_fetch.add("public-retry-bundle")
    resumed = await pipeline().run(connector)

    assert connector.fetches == fetches
    assert resumed.by_status[DocumentStatus.INDEXED.value] == 1
    assert resumed.skipped_hash == 1
    assert blobs.member_retentions == 3, "retry retained the successful member a second time"
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


async def test_crashed_indexing_container_reexpands_its_retained_parent_on_resume(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """An INDEXING parent cannot hash-skip away members left behind by a process crash."""

    class GateSecondMember(HashEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.entered = asyncio.Event()

        @override
        async def embed(self, texts: Sequence[str]) -> list[Vector]:
            self.calls += 1
            if self.calls == 2:
                self.entered.set()
                await asyncio.Event().wait()
            return await super().embed(texts)

    connector = fakes.DictConnector(
        {"public-crash-bundle": "one=alpha\ntwo=beta"}, name="crash-container-source"
    )
    connector.media_types["public-crash-bundle"] = fakes.CONTAINER_MEDIA_TYPE
    gated = GateSecondMember()
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()
    blobs = BlobStore(engine, data_dir)
    vectors = fakes.MemoryVectors()

    def pipeline(embedder: HashEmbedder) -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=blobs,
            chunker=chunker,
            embedder=embedder,
            vectors=vectors,
            runner=InProcessRunner({"archive": fakes.FakeArchive(), "lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["archive", "lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            acquisition_clock=lease_clock,
            acquisition_lease_s=1,
            shutdown_grace_s=0,
            detect_glossary=False,
        )

    task = asyncio.create_task(pipeline(gated).run(connector))
    await gated.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.INDEXING
    assert await store.find_document(connector.name, "public-crash-bundle!/one") is not None
    assert await store.find_document(connector.name, "public-crash-bundle!/two") is None
    fetches = list(connector.fetches)

    connector.fail_fetch.add("public-crash-bundle")
    lease_clock.advance(2)
    resumed = await pipeline(HashEmbedder()).run(connector)

    assert connector.fetches == fetches
    assert resumed.indexed == 1
    assert resumed.skipped_hash == 1
    assert await store.find_document(connector.name, "public-crash-bundle!/two") is not None
    assert await store.latest_unsettled_acquisition_run(connector.name) is None


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


async def test_crash_after_acquired_association_reconciles_marker_before_history_cleanup(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """A committed association makes its marker redundant before superseded GC releases it."""

    class CrashAfterAssociationBlobStore(BlobStore):
        @override
        async def complete_acquisition(self, key: str) -> None:
            del key
            raise RuntimeError("synthetic crash after ACQUIRED commit")

    connector = fakes.DictConnector(
        {"public-associated": "public associated bytes"}, name="associated-marker-source"
    )
    lease_clock = fakes.ManualLeaseClock()
    chunker = fakes.BlockChunker()
    crashing_blobs = CrashAfterAssociationBlobStore(engine, data_dir)
    pipeline = IngestPipeline(
        store=store,
        acquisitions=store,
        blobs=crashing_blobs,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        acquisition_clock=lease_clock,
        acquisition_lease_s=1,
        detect_glossary=False,
    )

    report = await pipeline.run(connector)

    assert report.error_type == "RuntimeError"
    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.ACQUIRED
    assert record.blob_ref is not None
    staging = crashing_blobs.root / "acquisition-staging"
    assert len(list(staging.iterdir())) == 1

    await store.set_watermark(
        connector.name,
        Watermark(value="successor-position", observed_at=lease_clock()),
    )
    lease_clock.advance(2)
    successor = await store.claim_or_create_acquisition_run(
        connector.name,
        "successor-run",
        "successor-owner",
        now=lease_clock(),
        expires_at=lease_clock() + timedelta(seconds=1),
    )
    assert successor is not None
    superseded = await store.get_acquisition_run(durable.id)
    assert superseded is not None
    assert superseded.superseded_at is not None

    restarted_blobs = BlobStore(engine, data_dir)
    await restarted_blobs.reconcile_acquisition_markers()
    assert list(staging.iterdir()) == []
    assert await store.cleanup_acquisition_history(datetime(2100, 1, 1, tzinfo=UTC), limit=10) == 1

    assert await store.get_acquisition_run(durable.id) is None
    assert await restarted_blobs.collect_garbage() == [record.blob_ref]
    assert await restarted_blobs.get(record.blob_ref) is None


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
    ).run(connector, limit=1)

    durable = await store.latest_unsettled_acquisition_run(connector.name)
    assert durable is not None
    record = (await store.list_acquisition_records(durable.id))[0]
    assert record.state is AcquisitionRecordState.RETRY
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == code
    assert record.blob_ref is None
    assert not report.watermark_advanced
    assert report.snapshot_completeness == ""
    assert report.snapshot_omissions == 1
    assert report.snapshot_omission_reasons == {code: 1}
    assert not report.pending_derivation
    assert report.limited
    assert str(failure) not in str(report.as_metadata())
    assert await store.find_document(connector.name, "public-refused") is None

    backend = FakeBackend()
    backend.ingestion_.report = report
    backend.settings.connectors[connector.name] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )
    service = ApplicationService(backend)
    envelope = await run_op(
        "connector_sync",
        service.workspace,
        lambda: service.connector_sync(connector.name),
    )

    assert envelope.ok is False
    assert envelope.data is not None
    assert envelope.data["outcome"] == "incomplete"
    assert envelope.data["retry_required"] is True
    assert envelope.error is not None
    assert envelope.error.message == "1 source snapshot member(s) require retry"
    assert envelope.error.hint is not None
    assert "source acquisition failure" in envelope.error.hint
    assert "without contacting the source" not in envelope.error.hint


async def test_source_deleted_reenumerates_and_reuses_the_retained_prefix(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class VersionedVanishingConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__(
                {f"synthetic-{index}": f"public body {index}" for index in range(10)},
                name="versioned-vanishing-source",
            )
            self.phase = 0
            self.seen_watermarks: list[Watermark | None] = []
            self._watermark = Watermark(
                value="candidate-1", observed_at=datetime(2026, 8, 16, tzinfo=UTC)
            )

        @property
        @override
        def watermark(self) -> Watermark:
            return self._watermark

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.seen_watermarks.append(watermark)
            async for discovered in super().discover(watermark):
                yield discovered
            self.phase += 1
            if self.phase == 1:
                del self.documents["synthetic-9"]
                self._watermark = Watermark(
                    value="candidate-2", observed_at=datetime(2026, 8, 16, 0, 0, 1, tzinfo=UTC)
                )

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            if ref.source_id == "synthetic-9":
                raise NotFoundError("synthetic source item is no longer available")
            if self.phase > 1:
                raise AssertionError("unchanged retained bodies must not be fetched again")
            return await super().fetch(ref)

    connector = VersionedVanishingConnector()
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    first = await pipeline().run(connector, acquire_only=True)
    stale = await store.latest_unsettled_acquisition_run(connector.name)
    assert stale is not None
    stale_generation = stale.lease_generation
    assert stale.inventory_state is AcquisitionInventoryState.REENUMERATION_REQUIRED
    assert first.inventory_recovery == "reenumeration_required"
    assert first.snapshot_omission_reasons == {"source_deleted": 1}
    assert first.watermark_advanced is False

    second = await pipeline().run(connector, acquire_only=True)
    replacement = await store.latest_unsettled_acquisition_run(connector.name)
    assert replacement is not None
    assert replacement.id != stale.id
    assert replacement.supersedes_run_id == stale.id
    assert replacement.inventory_state is AcquisitionInventoryState.RECONCILED
    assert replacement.reconciled_deleted_count == 1
    assert replacement.discovered_count == 9
    assert replacement.unchanged_count == 0
    assert replacement.reused_count == 9
    assert replacement.acquired_count == 9
    assert replacement.watermark_committed_at is not None
    assert second.inventory_recovery == "reconciled"
    assert second.reconciled_deleted_items == 1
    assert second.durable_reused == 9
    assert second.snapshot_completeness == "complete"
    assert connector.seen_watermarks == [None, None]
    assert len(connector.fetches) == 9
    assert all(
        record.snapshot_outcome is SnapshotItemOutcome.REUSED
        for record in await store.list_acquisition_records(replacement.id)
    )

    fenced = await store.get_acquisition_run(stale.id)
    assert fenced is not None
    assert fenced.superseded_by == replacement.id
    assert fenced.superseded_at is not None
    assert fenced.lease_owner is None
    assert fenced.lease_generation > stale_generation


@pytest.mark.parametrize(
    ("documents", "remove_missing_identity", "poll_status"),
    [
        pytest.param(8_001, True, True, id="eight-thousand-reuses-with-polling"),
        pytest.param(8_001, False, False, id="eight-thousand-still-missing-without-polling"),
    ],
)
async def test_served_source_deletion_recovery_reuses_exact_evidence_and_keeps_server_healthy(  # noqa: PLR0915, PLR0917 - end-to-end acceptance matrix
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    documents: int,
    remove_missing_identity: bool,
    poll_status: bool,
) -> None:
    """The large recovery and its next connector cross the real control socket."""

    class RecoveringConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__(
                {
                    f"synthetic-{index:05d}": "public shared retained body"
                    for index in range(documents)
                },
                name=f"served-recovery-{documents}",
            )
            self.missing = f"synthetic-{documents - 1:05d}"
            self.discovery_pass = 0
            self.attempts: list[str] = []
            self._watermark = Watermark(
                value=f"served-candidate-{documents}",
                observed_at=datetime(2026, 8, 17, tzinfo=UTC),
            )

        @property
        @override
        def watermark(self) -> Watermark:
            return self._watermark

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.discovery_pass += 1
            async for discovered in super().discover(watermark):
                yield discovered

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            self.attempts.append(ref.source_id)
            if ref.source_id == self.missing:
                raise NotFoundError("synthetic source item is no longer available")
            if self.discovery_pass > 1:
                raise AssertionError("exact retained survivors must not be fetched again")
            return await super().fetch(ref)

    connector = RecoveringConnector()
    next_connector = fakes.DictConnector(
        {"public-next": "public next body"}, name=f"served-next-{documents}"
    )
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=BlobStore(engine, data_dir, min_disk_headroom_bytes=1),
            chunker=chunker,
            embedder=HashEmbedder(),
            vectors=fakes.MemoryVectors(),
            runner=InProcessRunner({"lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            fetch_concurrency=8,
            acquisition_lease_s=1_800,
            detect_glossary=False,
        )

    first = await pipeline().run(connector, acquire_only=True)

    # **Which half failed, when this run comes back with no deletion recorded.** This test has
    # failed intermittently in CI — roughly one run in four, across both parametrizations, both
    # Python versions and different shards — reporting `snapshot_omission_reasons == {}`. Empty,
    # not wrong: no omission of any kind was recorded.
    #
    # That is reachable two ways, and the report alone cannot tell them apart.
    # `_report_snapshot_omissions` runs only when `_acquire_journal` reports incomplete coverage,
    # and that reports complete when no record is left in DISCOVERED/ACQUIRING/RETRY without a
    # blob. A `NotFoundError` fetch parks the record in RETRY with no blob, so a run that reached
    # the deleted document cannot report complete. Either the deleted document was never reached
    # — enumeration or dispatch stopped short of the 8001st item — or it was reached and its
    # outcome was lost afterwards. The remedies are in different subsystems.
    #
    # `attempts` is appended in `fetch` before the raise, so it answers exactly that question and
    # costs nothing. It is asserted first so the failure names the half rather than leaving the
    # next reader with the same fork this comment describes.
    assert connector.missing in connector.attempts, (
        f"the deleted document {connector.missing!r} was never fetched, so this run could not "
        f"have observed the deletion at all — {len(connector.attempts)} of {documents} documents "
        f"were attempted, over {connector.discovery_pass} discovery pass(es). The run stopped "
        f"short rather than losing the outcome: look at enumeration and dispatch, not at how the "
        f"omission was recorded."
    )
    assert first.snapshot_omission_reasons == {"source_deleted": 1}, (
        "the first run must reach its typed deletion outcome before inventory is inspected. "
        f"The deleted document *was* fetched ({connector.missing!r} is in attempts), so the "
        f"deletion was observed and then lost between the fetch and the report — look at the "
        f"record's state transition and at _report_snapshot_omissions, not at enumeration."
    )
    predecessor = await store.latest_unsettled_acquisition_run(connector.name)
    assert predecessor is not None
    assert predecessor.inventory_state is AcquisitionInventoryState.REENUMERATION_REQUIRED
    if remove_missing_identity:
        del connector.documents[connector.missing]

    reuse_arrived = asyncio.Event()
    release_reuse = asyncio.Event()
    gate_reuse = True
    original_transition = store.transition_acquisition_record

    async def gated_transition(
        run_id: str,
        source_id: str,
        expected: AcquisitionRecordState,
        target: AcquisitionRecordState,
        **kwargs: Any,
    ) -> AcquisitionRecord:
        nonlocal gate_reuse
        if (
            gate_reuse
            and expected is AcquisitionRecordState.ACQUIRING
            and target is AcquisitionRecordState.ACQUIRED
            and kwargs.get("snapshot_outcome") is SnapshotItemOutcome.REUSED
        ):
            reuse_arrived.set()
            await release_reuse.wait()
        return await original_transition(run_id, source_id, expected, target, **kwargs)

    monkeypatch.setattr(store, "transition_acquisition_record", gated_transition)
    recovery_connector = connector

    class PipelineIngestion(FakeIngestion):
        @override
        async def sync(
            self,
            connector: str,
            *,
            limit: int | None = None,
            watching: Watching | None = None,
            acquire_only: bool = False,
        ) -> RunReport:
            del watching
            selected = (
                recovery_connector if connector == recovery_connector.name else next_connector
            )
            return await pipeline().run(selected, limit=limit, acquire_only=acquire_only)

        @override
        async def snapshot_status(self, connector: str) -> tuple[AcquisitionRun, bool] | None:
            active = await store.latest_unsettled_acquisition_run(connector)
            return None if active is None else (active, False)

    backend = FakeBackend()
    backend.store = cast("Any", store)
    backend.ingestion_ = PipelineIngestion()
    for name in (connector.name, next_connector.name):
        backend.settings.connectors[name] = ConnectorSettings.model_validate(
            {"type": "filesystem", "options": {"root": "."}}
        )
    service = ApplicationService(backend)
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        recovery = asyncio.create_task(
            control.connect(
                path,
                control.Invoke(
                    op="connector_sync",
                    arguments={"name": connector.name, "limit": None, "acquire_only": True},
                ),
                on_progress=lambda _: None,
            )
        )
        await asyncio.wait_for(reuse_arrived.wait(), timeout=120)
        if poll_status:
            snapshot_envelope, listing_envelope = await asyncio.gather(
                control.connect(
                    path,
                    control.Invoke(op="snapshot_status", arguments={"name": connector.name}),
                    on_progress=lambda _: None,
                ),
                control.connect(
                    path,
                    control.Invoke(op="connector_list"),
                    on_progress=lambda _: None,
                ),
            )
            assert snapshot_envelope["ok"] is True
            snapshot_data = cast("dict[str, Any]", snapshot_envelope["data"])
            lifecycle = cast("dict[str, Any]", snapshot_data["lifecycle"])
            assert cast("int", lifecycle["reused_items"]) < documents
            assert listing_envelope["ok"] is True
            listing_data = cast("dict[str, Any]", listing_envelope["data"])
            connectors = cast("list[dict[str, Any]]", listing_data["connectors"])
            assert {summary["name"] for summary in connectors} == {
                connector.name,
                next_connector.name,
            }
        gate_reuse = False
        release_reuse.set()
        envelope = await asyncio.wait_for(recovery, timeout=180)

        replacement = await store.latest_unsettled_acquisition_run(connector.name)
        assert replacement is not None
        survivors = documents - 1
        assert replacement.reused_count == survivors
        assert connector.attempts.count(connector.missing) == (2 - int(remove_missing_identity))
        assert len(connector.attempts) == documents + int(not remove_missing_identity)
        if remove_missing_identity:
            assert envelope["ok"] is True
            assert replacement.inventory_state is AcquisitionInventoryState.RECONCILED
            assert replacement.reconciled_deleted_count == 1
            assert replacement.promoted_at is not None
            assert replacement.watermark_committed_at is not None
        else:
            assert envelope["ok"] is False
            assert replacement.inventory_state is AcquisitionInventoryState.REENUMERATION_REQUIRED
            assert replacement.promoted_at is None
            assert replacement.watermark_committed_at is None

        following = await control.connect(
            path,
            control.Invoke(
                op="connector_sync",
                arguments={"name": next_connector.name, "limit": None, "acquire_only": True},
            ),
            on_progress=lambda _: None,
        )
        assert following["ok"] is True
    finally:
        release_reuse.set()
        await server.aclose()


@pytest.mark.parametrize(
    "changed_identity",
    ["tokenless", "uri", "media_type", "size", "metadata", "provenance", "scope"],
)
async def test_reenumeration_fetches_when_exact_reuse_identity_is_unproven(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    changed_identity: str,
) -> None:
    from manicule.core.provenance import PROVENANCE_KEY  # noqa: PLC0415

    class IdentityChangingConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__(
                {"synthetic-current": "public body v1", "synthetic-deleted": "gone"},
                name=f"exact-reuse-{changed_identity}",
            )
            self.tokens["synthetic-current"] = "v1"
            self.discovery_calls = 0
            self.scope = "synthetic-scope-v1"

        @property
        def source_scope(self) -> str:
            return self.scope

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.discovery_calls += 1
            async for discovered in super().discover(watermark):
                emitted = discovered
                if discovered.source_id == "synthetic-current":
                    if changed_identity == "tokenless":
                        emitted = emitted.model_copy(update={"version_token": None})
                    if self.discovery_calls == 2:
                        if changed_identity == "uri":
                            emitted = emitted.model_copy(
                                update={
                                    "ref": emitted.ref.model_copy(
                                        update={"uri": "memory://synthetic-current-moved"}
                                    )
                                }
                            )
                        elif changed_identity == "media_type":
                            emitted = emitted.model_copy(update={"media_type": "text/x-fake-moved"})
                        elif changed_identity == "size":
                            emitted = emitted.model_copy(
                                update={"size_bytes": len(self.documents[emitted.source_id])}
                            )
                        elif changed_identity == "metadata":
                            emitted = emitted.model_copy(
                                update={"metadata": {"synthetic-route": "moved"}}
                            )
                        elif changed_identity == "provenance":
                            emitted = emitted.model_copy(
                                update={"metadata": {PROVENANCE_KEY: {"synthetic-origin": "moved"}}}
                            )
                yield emitted
            if self.discovery_calls == 1:
                del self.documents["synthetic-deleted"]

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            if ref.source_id == "synthetic-deleted":
                raise NotFoundError("synthetic source item is no longer available")
            return await super().fetch(ref)

    connector = IdentityChangingConnector()
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    await pipeline().run(connector, acquire_only=True)
    if changed_identity in {"tokenless", "size"}:
        connector.documents["synthetic-current"] = "public body v2"
    if changed_identity == "media_type":
        connector.media_types["synthetic-current"] = "text/x-fake-moved"
    if changed_identity == "scope":
        connector.scope = "synthetic-scope-v2"

    await pipeline().run(connector, acquire_only=True)
    active = await store.latest_unsettled_acquisition_run(connector.name)
    assert active is not None
    assert connector.fetches.count("synthetic-current") == 2
    assert active.reused_count == 0
    current = (await store.list_acquisition_records(active.id))[0]
    assert current.source.source_id == "synthetic-current"
    assert current.snapshot_outcome is SnapshotItemOutcome.RETAINED
    assert current.acquired_source is not None
    if changed_identity in {"tokenless", "size"}:
        assert current.acquired_source.content_hash == content_hash(b"public body v2")
    if changed_identity == "scope":
        assert active.supersedes_run_id is None
        assert active.inventory_state is AcquisitionInventoryState.CURRENT
    else:
        assert active.supersedes_run_id is not None
        assert active.inventory_state is AcquisitionInventoryState.RECONCILED


@pytest.mark.parametrize("replacement", ["same", "newer", "unavailable"])
async def test_reenumerated_item_is_fetched_or_remains_an_honest_failure(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    replacement: str,
) -> None:
    class ReappearingConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__(
                {
                    "synthetic-a": "public body a",
                    "synthetic-b": "public body b",
                    "synthetic-target": "public target v1",
                },
                name=f"reappearing-{replacement}",
            )
            self.tokens["synthetic-target"] = "v1"
            self.phase = 0
            self.attempts: list[str] = []

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            async for discovered in super().discover(watermark):
                yield discovered
            self.phase += 1
            if self.phase == 1:
                del self.documents["synthetic-target"]

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            self.attempts.append(ref.source_id)
            if ref.source_id == "synthetic-target" and (
                self.phase == 1 or replacement == "unavailable"
            ):
                raise NotFoundError("synthetic source item is no longer available")
            if self.phase > 1 and ref.source_id != "synthetic-target":
                raise AssertionError("unchanged retained bodies must not be fetched again")
            return await super().fetch(ref)

    connector = ReappearingConnector()
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    await pipeline().run(connector, acquire_only=True)
    connector.documents["synthetic-target"] = (
        "public target v2" if replacement == "newer" else "public target v1"
    )
    connector.tokens["synthetic-target"] = "v2" if replacement == "newer" else "v1"
    second = await pipeline().run(connector, acquire_only=True)
    active = await store.latest_unsettled_acquisition_run(connector.name)
    assert active is not None
    assert connector.attempts.count("synthetic-a") == 1
    assert connector.attempts.count("synthetic-b") == 1
    assert connector.attempts.count("synthetic-target") == 2

    if replacement == "unavailable":
        assert active.inventory_state is AcquisitionInventoryState.REENUMERATION_REQUIRED
        assert active.promoted_at is None
        assert active.watermark_committed_at is None
        assert second.inventory_recovery == "reenumeration_required"
        assert second.snapshot_omission_reasons == {"source_deleted": 1}
        return

    assert active.inventory_state is AcquisitionInventoryState.RECONCILED
    assert active.promoted_at is not None
    assert active.reused_count == 2
    assert second.durable_reused == 2
    assert second.snapshot_completeness == "complete"
    records = await store.list_acquisition_records(active.id)
    assert sum(record.snapshot_outcome is SnapshotItemOutcome.REUSED for record in records) == 2
    assert sum(record.snapshot_outcome is SnapshotItemOutcome.RETAINED for record in records) == 1


@pytest.mark.parametrize("failure", ["authentication", "capacity"])
async def test_non_inventory_failures_resume_the_completed_manifest(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    failure: str,
) -> None:
    class RetrySameInventoryConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__({"synthetic-current": "public current body"}, name=f"resume-{failure}")
            self.discovery_calls = 0
            self.fetch_calls = 0

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.discovery_calls += 1
            async for discovered in super().discover(watermark):
                yield discovered

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            self.fetch_calls += 1
            if self.fetch_calls == 1:
                if failure == "authentication":
                    raise SessionExpiredError("synthetic session expired")
                raise CapacityRefusedError(
                    CapacityDiagnostic(
                        resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
                        limit=1,
                        used=1,
                        requested=1,
                    )
                )
            return await super().fetch(ref)

    connector = RetrySameInventoryConnector()
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    await pipeline().run(connector, acquire_only=True)
    first = await store.latest_unsettled_acquisition_run(connector.name)
    assert first is not None
    assert first.inventory_state is AcquisitionInventoryState.CURRENT
    second = await pipeline().run(connector, acquire_only=True)
    resumed = await store.latest_unsettled_acquisition_run(connector.name)
    assert resumed is not None
    assert resumed.id == first.id
    assert connector.discovery_calls == 1
    assert connector.fetch_calls == 2
    assert second.snapshot_completeness == "complete"
    assert second.inventory_recovery == ""


async def test_allow_omissions_cannot_publish_an_inventory_known_to_be_stale(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class DeletedConnector(fakes.DictConnector):
        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            del ref
            raise NotFoundError("synthetic source item is no longer available")

    connector = DeletedConnector({"synthetic-deleted": "public body"}, name="partial-stale")
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
        snapshot_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    ).run(connector, acquire_only=True)

    run = await store.latest_unsettled_acquisition_run(connector.name)
    assert run is not None
    assert run.inventory_state is AcquisitionInventoryState.REENUMERATION_REQUIRED
    assert run.promoted_at is None
    assert run.watermark_committed_at is None
    assert report.snapshot_completeness == ""
    assert report.inventory_recovery == "reenumeration_required"


@pytest.mark.parametrize("interruption", ["cursor", "limit"])
async def test_incomplete_replacement_inventory_never_reconciles_deletion(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    interruption: str,
) -> None:
    class InterruptedReplacementConnector(fakes.DictConnector):
        def __init__(self) -> None:
            super().__init__(
                {
                    "synthetic-a": "public body a",
                    "synthetic-b": "public body b",
                    "synthetic-deleted": "public deleted body",
                },
                name=f"interrupted-replacement-{interruption}",
            )
            self.discovery_calls = 0

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.discovery_calls += 1
            emitted = 0
            async for discovered in super().discover(watermark):
                if self.discovery_calls == 2 and interruption == "cursor" and emitted == 1:
                    raise CursorExpiredError("synthetic cursor expired")
                emitted += 1
                yield discovered
            if self.discovery_calls == 1:
                del self.documents["synthetic-deleted"]

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            if ref.source_id == "synthetic-deleted":
                raise NotFoundError("synthetic source item is no longer available")
            if self.discovery_calls > 1:
                raise AssertionError("replacement must reuse unchanged retained bodies")
            return await super().fetch(ref)

    connector = InterruptedReplacementConnector()
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    await pipeline().run(connector, acquire_only=True)
    interrupted = await pipeline().run(
        connector,
        acquire_only=True,
        limit=1 if interruption == "limit" else None,
    )
    replacement = await store.latest_unsettled_acquisition_run(connector.name)
    assert replacement is not None
    assert replacement.inventory_state is AcquisitionInventoryState.REENUMERATING
    assert replacement.enumeration_completed_at is None
    assert replacement.reconciled_deleted_count == 0
    assert replacement.reused_count == 1
    assert replacement.promoted_at is None
    assert interrupted.enumeration_completed is False
    assert interrupted.durable_reused == 1
    assert interrupted.reconciled_deleted_items == 0

    completed = await pipeline().run(connector, acquire_only=True)
    replacement = await store.latest_unsettled_acquisition_run(connector.name)
    assert replacement is not None
    assert replacement.inventory_state is AcquisitionInventoryState.RECONCILED
    assert replacement.reconciled_deleted_count == 1
    assert replacement.promoted_at is not None
    assert completed.snapshot_completeness == "complete"


async def test_strict_omission_with_an_existing_document_is_still_incomplete_and_retryable(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    from tests.storage_helpers import make_document  # noqa: PLC0415 - storage fixture

    class RefusingConnector(fakes.DictConnector):
        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            del ref
            raise SessionExpiredError("synthetic private credential detail")

    connector = RefusingConnector({"existing": "new body"}, name="strict-existing")
    connector.tokens["existing"] = "v2"
    await store.upsert_document(
        make_document(source=connector.name, source_id="existing").model_copy(
            update={"version_token": "v1"}
        )
    )
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

    assert report.unrecorded == 0
    assert report.snapshot_omissions == 1
    assert report.snapshot_omission_reasons == {"authentication": 1}
    assert report.complete is False
    last_run = cast("Mapping[str, object]", report.as_metadata()["last_run"])
    assert last_run["outcome"] == "incomplete"
    assert last_run["retry_required"] is True
    assert "private credential" not in str(report.as_metadata())


async def test_allow_omissions_reports_partial_snapshot_and_typed_aggregate(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class RefusingConnector(fakes.DictConnector):
        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            del ref
            raise SessionExpiredError("synthetic private credential detail")

    connector = RefusingConnector(
        {"public-omitted": "never returned"}, name="partial-snapshot-source"
    )
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
        snapshot_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    ).run(connector)

    assert report.snapshot_completeness == "partial"
    assert report.snapshot_omissions == 1
    assert report.snapshot_omission_reasons == {"authentication": 1}
    assert not report.watermark_advanced, "this source declares no native watermark"
    assert "private credential" not in str(report.as_metadata())
    assert await store.latest_unsettled_acquisition_run(connector.name) is None
    scope = hashlib.blake2b(f"instance:{connector.name}".encode(), digest_size=20).hexdigest()
    assert await store.latest_promoted_snapshot(connector.name, scope) is not None
    assert await store.find_document(connector.name, "public-omitted") is None


async def test_unchanged_revision_reuses_promoted_snapshot_without_body_download(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = fakes.DictConnector(
        {
            "public-reused-a": "retained public body a",
            "public-reused-b": "retained public body b",
        },
        name="reused-snapshot-source",
    )
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    first = await pipeline().run(connector)
    fetches = list(connector.fetches)
    latest_calls = 0
    latest_promoted_snapshot = store.latest_promoted_snapshot

    async def counted_latest(connector_name: str, scope_fingerprint: str):  # noqa: ANN202
        nonlocal latest_calls
        latest_calls += 1
        return await latest_promoted_snapshot(connector_name, scope_fingerprint)

    with monkeypatch.context() as patch:
        patch.setattr(store, "latest_promoted_snapshot", counted_latest)
        second = await pipeline().run(connector)

    assert first.snapshot_completeness == "complete"
    assert second.snapshot_completeness == "complete"
    assert second.skipped_version == 2
    assert latest_calls == 1, "one run verifies the reusable manifest only once"
    assert connector.fetches == fetches
    scope = hashlib.blake2b(f"instance:{connector.name}".encode(), digest_size=20).hexdigest()
    promoted = await store.latest_promoted_snapshot(connector.name, scope)
    assert promoted is not None
    assert all(
        record.snapshot_outcome is SnapshotItemOutcome.REUSED
        for record in await store.list_acquisition_records(promoted.id)
    )


WORKERS = 8
"""Acquisition workers the reusable-manifest test runs with, pinned rather than inherited.

Enough of them, against three times as many records, that several are inside the lookup together
on every run. Read from here rather than from ``IngestPipeline``'s default so that lowering that
default changes what the pipeline does and not what this test proves."""


async def test_the_reusable_manifest_is_verified_once_across_every_acquisition_worker(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim above, made to fail on demand rather than on a bad day.

    The test above states the same property with two documents, and two documents rarely put two
    workers in the lookup at once — so it held for months and then failed one CI shard, which is
    the worst way for a property to be checked. Three records per worker puts several of them in
    the lookup together every time, and the lookup is held open across a few event-loop turns so
    the window cannot close by luck on a fast machine.

    Instrumenting the unguarded version showed exactly this: ``acquire-1`` enters while
    ``acquire-0`` is still inside the query, reads the memo as ``False``, and issues it again.

    :data:`WORKERS` is passed to the pipeline rather than inherited from it. Eight is the default
    today, and a default lowered to one would leave this test green while it reproduced nothing —
    a test that cannot fail for the reason it exists, which is the shape #233 found in the
    lease-heartbeat test after a refactor moved the seam it patched.
    """
    connector = fakes.DictConnector(
        {f"public-shared-{index}": f"retained body {index}" for index in range(3 * WORKERS)},
        name="shared-manifest-source",
    )
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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
            fetch_concurrency=WORKERS,
        )

    first = await pipeline().run(connector)
    fetches = list(connector.fetches)
    calls = 0
    latest_promoted_snapshot = store.latest_promoted_snapshot

    async def held_open(connector_name: str, scope_fingerprint: str):  # noqa: ANN202
        nonlocal calls
        calls += 1
        # Yields rather than sleeps: every other worker runs to its own check while this one is
        # suspended, which is the window the memo has to survive, and it costs no wall clock.
        for _ in range(8):
            await asyncio.sleep(0)
        return await latest_promoted_snapshot(connector_name, scope_fingerprint)

    with monkeypatch.context() as patch:
        patch.setattr(store, "latest_promoted_snapshot", held_open)
        second = await pipeline().run(connector)

    assert first.snapshot_completeness == "complete"
    assert second.skipped_version == 3 * WORKERS
    assert calls == 1, "every worker after the first reads the memo instead of repeating the query"
    assert connector.fetches == fetches, "a verified manifest means no body is downloaded again"


async def test_direct_native_pages_restart_from_an_exact_durable_prefix(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = "https://wiki.example.test/confluence"
    pages = [
        FakePage(id=str(180100 + number), title=f"Current {number}", space="DOCS")
        for number in range(7)
    ]
    instance = FakeConfluence(base_url=base, pages=pages, page_size=2)
    config = server_config(
        base,
        spaces=("DOCS",),
        include_attachments=False,
        page_size=250,
        full_inventory_authority=FullInventoryAuthority.DIRECT_CURRENT_CONTENT,
    )
    connector = await connected(instance, config)
    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    original_append = store.append_acquisition_records
    committed_pages = 0

    async def interrupt_after_two_pages(*args: Any, **kwargs: Any):  # noqa: ANN202
        nonlocal committed_pages
        records = await original_append(*args, **kwargs)
        committed_pages += 1
        if committed_pages == 2:
            raise CursorExpiredError("synthetic interruption after a committed native page")
        return records

    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "append_acquisition_records", interrupt_after_two_pages)
            interrupted = await pipeline().run(connector, acquire_only=True)

        first_requests = [
            request
            for request in instance.requests
            if request.url.path.endswith("/rest/api/content")
        ]
        assert len(first_requests) == 2
        assert interrupted.enumeration_completed is False
        unfinished = await store.latest_unsettled_acquisition_run(connector.name)
        assert unfinished is not None
        assert unfinished.enumeration_completed_at is None
        assert unfinished.promoted_at is None
        assert unfinished.watermark_committed_at is None
        prefix = await store.list_acquisition_records(unfinished.id)
        assert [record.sequence for record in prefix] == [0, 1, 2, 3]
        first_body_requests = [
            request for request in instance.requests if "/rest/api/content/" in request.url.path
        ]
        assert len(first_body_requests) == 4

        instance.requests.clear()
        completed = await pipeline().run(connector, acquire_only=True)
    finally:
        await connector.teardown()

    assert completed.enumeration_completed
    assert completed.snapshot_completeness == "complete"
    promoted = await store.latest_promoted_snapshot(connector.name, connector.scope_fingerprint)
    assert promoted is not None
    records = await store.list_acquisition_records(promoted.id)
    assert [record.sequence for record in records] == list(range(7))
    restarted = [
        request for request in instance.requests if request.url.path.endswith("/rest/api/content")
    ]
    assert [request.url.params.get("start", "0") for request in restarted] == [
        "0",
        "2",
        "4",
        "6",
    ]
    body_requests = [
        request for request in instance.requests if "/rest/api/content/" in request.url.path
    ]
    assert len(body_requests) == 3, "the exact four-record durable prefix was fetched twice"


@pytest.mark.parametrize(
    "policy",
    [
        SnapshotPromotionPolicy.REQUIRE_COMPLETE,
        SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    ],
)
async def test_direct_member_deleted_after_true_end_never_promotes_stale_inventory(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    policy: SnapshotPromotionPolicy,
) -> None:
    page = FakePage(id="190100", title="Current", space="DOCS")
    instance = FakeConfluence(
        base_url="https://wiki.example.test/confluence",
        pages=[page],
    )
    config = server_config(
        instance.base_url,
        spaces=("DOCS",),
        include_attachments=False,
        full_inventory_authority=FullInventoryAuthority.DIRECT_CURRENT_CONTENT,
    )
    settings, client = client_for(instance, config)

    class DeleteAfterEnumeration(ConfluenceConnector):
        @override
        async def discover_batches(
            self, watermark: Watermark | None
        ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
            async for batch in super().discover_batches(watermark):
                yield batch
            instance.delete(page.id)

    connector = DeleteAfterEnumeration(settings, client, name=f"direct-{policy.value}")
    await connector.setup()
    chunker = fakes.BlockChunker()
    try:
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
            snapshot_policy=policy,
        ).run(connector, acquire_only=True)
    finally:
        await connector.teardown()

    assert report.inventory_recovery == "reenumeration_required"
    assert report.snapshot_completeness == ""
    run = await store.latest_unsettled_acquisition_run(connector.name)
    assert run is not None
    assert run.promoted_at is None
    assert run.watermark_committed_at is None


async def test_authority_change_fences_failed_inventory_and_reuses_exact_retained_bodies(  # noqa: PLR0915 - one end-to-end recovery/publication lifecycle
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class AuthorityConnector(fakes.DictConnector):
        source_scope = "synthetic-whole-space"

        def __init__(
            self,
            documents: Mapping[str, str],
            *,
            fingerprint: str,
            authority: str,
            name: str = "synthetic-wiki",
        ) -> None:
            super().__init__(documents, name=name)
            self.scope_fingerprint = fingerprint
            self.full_inventory_authority = authority
            self.deleted: set[str] = set()
            self.forbidden_fetches: set[str] = set()
            self.seen_watermarks: list[Watermark | None] = []
            self.tokens = {source_id: f"version-{source_id}" for source_id in documents}

        @property
        @override
        def watermark(self) -> Watermark:
            return Watermark(
                value=f"{self.full_inventory_authority}-authoritative-end",
                observed_at=datetime(2026, 8, 17, tzinfo=UTC),
            )

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.seen_watermarks.append(watermark)
            for source_id in sorted(self.documents):
                yield DiscoveredDoc(
                    ref=DocRef(
                        source_id=source_id,
                        uri=f"https://wiki.example.test/pages/{source_id}",
                    ),
                    version_token=self.tokens[source_id],
                    media_type="text/plain",
                )

        @override
        async def fetch(self, ref: DocRef) -> RawDocument:
            self.fetches.append(ref.source_id)
            if ref.source_id in self.deleted:
                raise NotFoundError("the synthetic current member is no longer available")
            if ref.source_id in self.forbidden_fetches:
                raise AssertionError("an unchanged retained body was fetched again")
            return RawDocument(
                source_id=ref.source_id,
                uri=ref.uri,
                media_type="text/plain",
                content=self.documents[ref.source_id],
                metadata={"version_token": self.tokens[ref.source_id]},
            )

    retained = {
        f"page-{number:03d}": f"retained synthetic body {number:03d}" for number in range(99)
    }
    search = AuthorityConnector(
        {**retained, "stale-page": "stale synthetic body"},
        fingerprint="search-inventory-fingerprint",
        authority="search",
    )
    search.deleted.add("stale-page")
    chunker = fakes.BlockChunker()
    blobs = BlobStore(engine, data_dir)
    embedder = HashEmbedder()
    vectors = LanceVectorStore(data_dir / "authority-vectors")
    await vectors.ensure_ready(embedder.fingerprint)
    parser = PlaintextParser(PlaintextConfig())

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
            store=store,
            acquisitions=store,
            blobs=blobs,
            chunker=chunker,
            embedder=embedder,
            vectors=vectors,
            runner=InProcessRunner({"plaintext": parser}),
            resolve_chain=lambda _: ["plaintext"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=chunker.fingerprint,
            detect_glossary=False,
        )

    incomplete = await pipeline().run(search)
    assert incomplete.inventory_recovery == "reenumeration_required"
    assert incomplete.snapshot_completeness == ""
    assert not incomplete.watermark_advanced

    direct = AuthorityConnector(
        {
            **retained,
            "page-102": "new synthetic body 102",
            "page-103": "new synthetic body 103",
            "page-104": "new synthetic body 104",
        },
        fingerprint="direct-current-inventory-fingerprint",
        authority="direct_current_content",
    )
    direct.forbidden_fetches.update(retained)
    complete = await pipeline().run(direct)

    assert complete.snapshot_completeness == "complete"
    assert complete.inventory_recovery == "reconciled"
    assert complete.discovered == 102
    assert complete.durable_reused == 99
    assert complete.reconciled_deleted_items == 1
    assert complete.watermark_advanced
    assert direct.seen_watermarks == [None]
    assert set(direct.fetches) == {"page-102", "page-103", "page-104"}
    promoted = await store.latest_promoted_snapshot(
        direct.name, "direct-current-inventory-fingerprint"
    )
    assert promoted is not None
    assert promoted.supersedes_run_id is not None
    assert promoted.full_inventory_authority == "direct_current_content"
    assert promoted.discovered_count == 102
    assert promoted.reconciled_deleted_count == 1
    assert promoted.watermark_committed_at is not None
    predecessor = await store.get_acquisition_run(promoted.supersedes_run_id)
    assert predecessor is not None
    assert predecessor.superseded_by == promoted.id
    assert predecessor.superseded_at is not None
    assert predecessor.lease_owner is None
    assert predecessor.lease_generation > 1

    direct.tokens["page-102"] = "version-page-102-replaced"
    direct.deleted.add("page-102")
    invalidated = await pipeline().run(direct)
    assert invalidated.inventory_recovery == "reenumeration_required"
    assert not invalidated.watermark_advanced
    assert direct.seen_watermarks[-1] == direct.watermark

    direct.deleted.remove("page-102")
    direct.documents.pop("page-102")
    direct.tokens.pop("page-102")
    reconciled = await pipeline().run(direct)

    assert reconciled.inventory_recovery == "reconciled"
    assert reconciled.snapshot_completeness == "complete"
    assert reconciled.discovered == 101
    assert reconciled.reconciled_deleted_items == 1
    assert reconciled.watermark_advanced
    assert direct.seen_watermarks[-1] is None, (
        "same-authority recovery must keep its durable base watermark for promotion CAS "
        "without turning the replacement discovery into an incremental walk"
    )

    final_snapshot = await store.latest_promoted_snapshot(
        direct.name, "direct-current-inventory-fingerprint"
    )
    assert final_snapshot is not None
    source_calls = len(direct.fetches)
    direct.forbidden_fetches.update(direct.documents)
    await store.reset_derived()
    assert await store.count_chunks() == 0

    installed_parse = parse_fingerprint("plaintext")
    assert installed_parse is not None
    target = RebuildTarget(
        parser_routing="synthetic-authority-routing-v1",
        parser_set=(installed_parse.canonical(),),
        chunk_fingerprint=chunker.fingerprint.canonical(),
        embedding_fingerprint=embedder.fingerprint.canonical(),
        embedding_config=embedder.fingerprint.model_dump_json(),
        glossary_fingerprint=glossary_fingerprint(enabled=False, middleware=()).canonical(),
        fts_tokenizer=FTS_TOKENIZER,
        batch_documents=7,
        max_memory_bytes=64 * 1024 * 1024,
        max_temporary_bytes=64 * 1024 * 1024,
    )
    rebuild_store = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=blobs,
        vectors=vectors,
    )
    rebuilder = build_offline_rebuilder(
        store=rebuild_store,
        blobs=blobs,
        workspace_id=store.workspace_id,
        source=direct.name,
        parser_chain=ParserChain(
            parsers={"plaintext": parser}, chains={"text/plain": ("plaintext",)}
        ),
        routing_identity=target.parser_routing,
        chunker=chunker,
        embedder=embedder,
        vectors=vectors,
        chunk_fingerprint=chunker.fingerprint,
        middleware=MiddlewareRunner(()),
        parse_runner=InProcessRunner(
            {"plaintext": parser}, middleware=MiddlewareRunner(()), chunker=chunker
        ),
        detect_glossary=False,
        clock=fakes.ManualLeaseClock(),
    )
    plan = await rebuilder.dry_run(final_snapshot.id, target)
    rebuilt = await rebuilder.run(final_snapshot.id, target, owner="synthetic-offline-rebuild")

    assert plan.runnable
    assert plan.documents == 101
    assert rebuilt.state is RebuildState.PUBLISHED
    assert rebuilt.documents_built == 101
    assert len(direct.fetches) == source_calls
    assert await store.count_documents() == 101
    settled = await store.get_acquisition_run(final_snapshot.id)
    assert settled is not None
    assert settled.state is AcquisitionRunState.SETTLED
    assert settled.acquired_count == settled.indexed_count == 101

    promoted_search = AuthorityConnector(
        {
            "fallback-a": "retained fallback a",
            "fallback-b": "retained fallback b",
            "fallback-c": "retained fallback c",
        },
        fingerprint="promoted-search-fingerprint",
        authority="search",
        name="promoted-search-wiki",
    )
    promoted_report = await pipeline().run(promoted_search)
    assert promoted_report.snapshot_completeness == "complete"
    older_promoted = await store.latest_promoted_snapshot(
        promoted_search.name, "promoted-search-fingerprint"
    )
    assert older_promoted is not None
    await pipeline().run(promoted_search)
    newest_promoted = await store.latest_promoted_snapshot(
        promoted_search.name, "promoted-search-fingerprint"
    )
    assert newest_promoted is not None
    assert newest_promoted.id != older_promoted.id
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE acquisition_runs SET membership_hash = 'corrupted' WHERE id = :run_id"),
            {"run_id": newest_promoted.id},
        )

    promoted_direct = AuthorityConnector(
        {
            "fallback-a": "retained fallback a",
            "fallback-b": "changed fallback b",
            "fallback-c": "retained fallback c",
        },
        fingerprint="promoted-direct-fingerprint",
        authority="direct_current_content",
        name="promoted-search-wiki",
    )
    promoted_direct.tokens["fallback-b"] = "version-fallback-b-changed"
    promoted_direct.forbidden_fetches.update({"fallback-a", "fallback-c"})
    fallback = await pipeline().run(promoted_direct)

    assert fallback.snapshot_completeness == "complete"
    assert fallback.durable_reused == 2
    assert promoted_direct.fetches == ["fallback-b"]


async def test_scope_change_resets_discovery_cursor_and_cas_binds_the_new_scope(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    class ScopedConnector(fakes.DictConnector):
        def __init__(self, documents: Mapping[str, str], *, scope: str) -> None:
            super().__init__(documents, name="scoped-cursor-source")
            self.source_scope = scope
            self.seen_watermarks: list[Watermark | None] = []
            self._watermark = Watermark(value=f"cursor:{scope}", observed_at=datetime.now(UTC))

        @property
        @override
        def watermark(self) -> Watermark:
            return self._watermark

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            self.seen_watermarks.append(watermark)
            async for discovered in super().discover(watermark):
                yield discovered

    chunker = fakes.BlockChunker()

    def pipeline() -> IngestPipeline:
        return IngestPipeline(
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

    first = ScopedConnector({"a": "scope a"}, scope="scope:A")
    await pipeline().run(first)
    first_fp = hashlib.blake2b(b"scope:A", digest_size=20).hexdigest()
    assert first.seen_watermarks == [None]
    assert await store.get_acquisition_watermark(first.name, first_fp) == first.watermark

    second = ScopedConnector({"b": "scope b"}, scope="scope:B")
    await pipeline().run(second)
    second_fp = hashlib.blake2b(b"scope:B", digest_size=20).hexdigest()
    assert second.seen_watermarks == [None]
    assert await store.get_acquisition_watermark(second.name, second_fp) == second.watermark
    assert await store.get_acquisition_watermark(second.name, first_fp) is None


async def test_resumed_promoted_indexing_reports_persisted_partial_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    connector = fakes.DictConnector({}, name="resumed-partial-source")
    scope = f"instance:{connector.name}"
    scope_fp = hashlib.blake2b(scope.encode(), digest_size=20).hexdigest()
    clock = fakes.ManualLeaseClock()
    created = await store.create_acquisition_run(
        "resumed-partial-run",
        connector.name,
        source_scope=scope,
        scope_fingerprint=scope_fp,
        promotion_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    )
    run = await store.claim_acquisition_run(
        created.id,
        "crashed-worker",
        now=clock(),
        expires_at=clock() + timedelta(seconds=1),
    )
    assert run is not None
    await store.append_acquisition_record(
        run.id,
        0,
        AcquisitionSource.from_discovered(
            DiscoveredDoc(ref=DocRef(source_id="omitted", uri="memory://omitted"))
        ),
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.complete_acquisition_enumeration(
        run.id,
        None,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.transition_acquisition_record(
        run.id,
        "omitted",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
        diagnostic=AcquisitionDiagnostic(
            stage=AcquisitionStage.ACQUISITION,
            code=AcquisitionFailureCode.AUTHENTICATION,
        ),
    )
    await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    promoted = await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint=scope_fp,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    assert promoted.completeness is SnapshotCompleteness.PARTIAL
    await store.transition_acquisition_run(
        run.id,
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.INDEXING,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    clock.advance(2)
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
        acquisition_clock=clock,
        acquisition_lease_s=1,
        detect_glossary=False,
    ).run(connector)

    assert report.snapshot_completeness == "partial"
    assert report.snapshot_omissions == 1
    assert report.snapshot_omission_reasons == {"authentication": 1}
    assert report.complete is True


async def test_post_promotion_blob_failure_does_not_invalidate_manifest_and_resumes_offline(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    connector = fakes.DictConnector(
        {"offline": "retained offline body"}, name="immutable-snapshot-diagnostic"
    )
    scope = f"instance:{connector.name}"
    scope_fp = hashlib.blake2b(scope.encode(), digest_size=20).hexdigest()
    clock = fakes.ManualLeaseClock()
    blobs = BlobStore(engine, data_dir)
    raw = RawDocument(
        source_id="offline",
        uri="memory://offline",
        media_type=fakes.MEDIA_TYPE,
        content="retained offline body",
        metadata={"version_token": "v1"},
    )
    stored = await blobs.put(raw.as_bytes(), raw.media_type)
    assert isinstance(stored, StoredBlob)
    created = await store.create_acquisition_run(
        "immutable-diagnostic-run",
        connector.name,
        source_scope=scope,
        scope_fingerprint=scope_fp,
    )
    run = await store.claim_acquisition_run(
        created.id,
        "crashed-worker",
        now=clock(),
        expires_at=clock() + timedelta(seconds=1),
    )
    assert run is not None
    await store.append_acquisition_record(
        run.id,
        0,
        AcquisitionSource.from_discovered(
            DiscoveredDoc(
                ref=DocRef(source_id="offline", uri="memory://offline"),
                version_token="v1",  # noqa: S106 - source revision, not a credential
                media_type=fakes.MEDIA_TYPE,
            )
        ),
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.complete_acquisition_enumeration(
        run.id,
        None,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.transition_acquisition_record(
        run.id,
        "offline",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.transition_acquisition_record(
        run.id,
        "offline",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
        blob_ref=stored.hash,
        acquired_source=AcquiredSource.from_raw(raw),
        fetched_version_token="v1",  # noqa: S106 - source revision, not a credential
    )
    await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint=scope_fp,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.transition_acquisition_run(
        run.id,
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.INDEXING,
        lease_owner="crashed-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    blobs.path_for(stored.hash).unlink()
    clock.advance(2)
    connector.fail_fetch.add("offline")
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
            acquisition_clock=clock,
            acquisition_lease_s=1,
            detect_glossary=False,
        )

    failed = await pipeline().run(connector)
    assert failed.retry_required
    assert failed.pending_derivation
    failed_record = (await store.list_acquisition_records(run.id))[0]
    assert failed_record.state is AcquisitionRecordState.RETRY
    assert failed_record.snapshot_diagnostic is None
    assert failed_record.diagnostic is not None
    assert failed_record.diagnostic.code is AcquisitionFailureCode.SNAPSHOT_MISSING
    assert await store.verify_snapshot_manifest(run.id)
    assert await store.latest_promoted_snapshot(connector.name, scope_fp) is not None
    assert await store.reusable_snapshot_record(connector.name, scope_fp, "offline", "v1")

    restored = await blobs.put(raw.as_bytes(), raw.media_type)
    assert isinstance(restored, StoredBlob)
    clock.advance(2)
    resumed = await pipeline().run(connector)

    assert resumed.indexed == 1
    assert connector.fetches == []
    assert await store.verify_snapshot_manifest(run.id)
    assert await store.latest_promoted_snapshot(connector.name, scope_fp) is not None
    settled = (await store.list_acquisition_records(run.id))[0]
    assert settled.state is AcquisitionRecordState.SETTLED
    assert settled.diagnostic is None


async def test_retained_prefix_index_retry_is_canonicalized_and_promoted_without_redownload(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    body = "retained prefix body"
    connector = fakes.DictConnector({"prefix": body}, name="retained-prefix-retry")
    connector.fail_fetch.add("prefix")
    version = content_hash(body)
    scope = f"instance:{connector.name}"
    scope_fp = hashlib.blake2b(scope.encode(), digest_size=20).hexdigest()
    clock = fakes.ManualLeaseClock()
    blobs = BlobStore(engine, data_dir)
    raw = RawDocument(
        source_id="prefix",
        uri="memory://prefix",
        media_type=fakes.MEDIA_TYPE,
        content=body,
        metadata={"version_token": version},
    )
    stored = await blobs.put(raw.as_bytes(), raw.media_type)
    assert isinstance(stored, StoredBlob)
    created = await store.create_acquisition_run(
        "retained-prefix-run",
        connector.name,
        source_scope=scope,
        scope_fingerprint=scope_fp,
    )
    run = await store.claim_acquisition_run(
        created.id,
        "prefix-worker",
        now=clock(),
        expires_at=clock() + timedelta(seconds=1),
    )
    assert run is not None
    await store.append_acquisition_record(
        run.id,
        0,
        AcquisitionSource.from_discovered(
            DiscoveredDoc(
                ref=DocRef(source_id="prefix", uri="memory://prefix"),
                version_token=version,
                media_type=fakes.MEDIA_TYPE,
            )
        ),
        lease_owner="prefix-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.transition_acquisition_record(
        run.id,
        "prefix",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="prefix-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    await store.transition_acquisition_record(
        run.id,
        "prefix",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="prefix-worker",
        lease_generation=run.lease_generation,
        now=clock(),
        blob_ref=stored.hash,
        acquired_source=AcquiredSource.from_raw(raw),
        fetched_version_token=version,
    )
    await store.transition_acquisition_record(
        run.id,
        "prefix",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.INDEXING,
        lease_owner="prefix-worker",
        lease_generation=run.lease_generation,
        now=clock(),
    )
    retried = await store.transition_acquisition_record(
        run.id,
        "prefix",
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.RETRY,
        lease_owner="prefix-worker",
        lease_generation=run.lease_generation,
        now=clock(),
        diagnostic=AcquisitionDiagnostic(
            stage=AcquisitionStage.INDEXING,
            code=AcquisitionFailureCode.PARSE_FAILED,
        ),
    )
    assert retried.blob_ref == stored.hash
    assert retried.snapshot_diagnostic is None
    # Simulate a row written by the pre-fix path: completion must repair all retained members.
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET snapshot_diagnostic = :diagnostic "
                "WHERE run_id = 'retained-prefix-run'"
            ),
            {
                "diagnostic": json.dumps(
                    {
                        "stage": "indexing",
                        "code": "parse_failed",
                        "retryable": True,
                    }
                )
            },
        )
    clock.advance(2)
    chunker = fakes.BlockChunker()
    report = await IngestPipeline(
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
        acquisition_clock=clock,
        acquisition_lease_s=1,
        detect_glossary=False,
    ).run(connector)

    assert report.indexed == 1
    assert report.snapshot_completeness == "complete"
    assert connector.fetches == []
    assert await store.verify_snapshot_manifest(run.id)
    assert await store.latest_promoted_snapshot(connector.name, scope_fp) is not None
    settled = (await store.list_acquisition_records(run.id))[0]
    assert settled.state is AcquisitionRecordState.SETTLED
    assert settled.snapshot_diagnostic is None


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
