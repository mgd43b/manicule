"""One document at a time, and none of them able to stop the rest.

The happy path is one test here. The rest are failures, because a pipeline whose purpose is
surviving failure is certified by nothing if only its happy path is exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, override

import pytest

from manicule.core.content import DocumentStatus, PipelineStage, RawDocument
from manicule.core.ids import content_hash
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import BlobSink, IngestPipeline
from manicule.ingest.workers import InProcessRunner
from tests.fakes import MEDIA_TYPE, HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from manicule.core.protocols import Embedder, Middleware


def build(
    *,
    store: fakes.MemoryIngestStore | None = None,
    vectors: fakes.MemoryVectors | None = None,
    parsers: Mapping[str, object] | None = None,
    chain: Sequence[str] = ("lines",),
    middleware: Sequence[Middleware] = (),
    embedder: Embedder | None = None,
    blobs: BlobSink | None = None,
    max_fetch_bytes: int = 256 * 1024 * 1024,
) -> tuple[IngestPipeline, fakes.MemoryIngestStore, fakes.MemoryVectors]:
    """A pipeline over in-memory everything, plus the store and vectors to assert against."""
    store = store or fakes.MemoryIngestStore()
    vectors = vectors or fakes.MemoryVectors()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=embedder or HashEmbedder(),
        vectors=vectors,
        runner=InProcessRunner(parsers or {"lines": fakes.LineParser()}),
        resolve_chain=lambda _: list(chain),
        middleware=MiddlewareRunner(middleware),
        chunk_fingerprint=chunker.fingerprint,
        blobs=blobs,
        max_fetch_bytes=max_fetch_bytes,
    )
    return pipeline, store, vectors


def discovered(source_id: str, text: str, media_type: str = MEDIA_TYPE) -> DiscoveredDoc:
    return DiscoveredDoc(
        ref=DocRef(source_id=source_id, uri=f"memory://{source_id}"),
        version_token=content_hash(text),
        media_type=media_type,
    )


# --- the whole path -------------------------------------------------------------------------


async def test_a_document_goes_from_bytes_to_indexed_with_chunks_and_vectors() -> None:
    """The end-to-end claim, in one test, so every failure test below has a baseline."""
    pipeline, store, vectors = build()
    connector = fakes.DictConnector({"a": "alpha\nbeta"})

    report = await pipeline.run(connector)

    assert report.indexed == 1
    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert len(store.chunks[document.id]) == 2
    assert len(vectors.rows) == 2
    assert store.lineage[document.id] == (
        fakes.BlockChunker.fingerprint.canonical(),
        HashEmbedder().fingerprint.canonical(),
    ), "per-document lineage is what makes a later invalidation a query rather than a rebuild"


async def test_the_write_order_marks_a_document_indexed_last() -> None:
    """A crash between vectors and the commit must leave nothing servable.

    Checked by breaking the vector store: chunks are written, the vector upsert fails, and the
    document must not be ``indexed``. If ``indexed`` were written first, this document would be
    served with no vectors at all and nobody would know.
    """
    pipeline, store, _ = build(vectors=fakes.RefusingVectors())
    connector = fakes.DictConnector({"a": "alpha"})

    await pipeline.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.failed_stage is PipelineStage.STORE
    assert store.chunks[document.id], "chunks are written first, and that is not a bug"


# --- one bad document never aborts a batch ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("a parser that raises", {"parsers": {"lines": fakes.ExplodingParser()}}),
        ("an embedder that fails", {"embedder": fakes.RefusingEmbedder()}),
        ("a store that refuses vectors", {"vectors": fakes.RefusingVectors()}),
        ("a hook that raises", {"middleware": (fakes.Exploder(),)}),
    ],
)
async def test_one_broken_document_leaves_the_rest_of_the_batch_indexed(
    name: str, kwargs: dict[str, object]
) -> None:
    """Four different ways to break one document, and none of them ends the run.

    The four are chosen because they fail at four different stages. A pipeline that caught
    only the parser's exception would pass a narrower version of this test and still lose a
    corpus to a slow model.
    """
    del name
    broken, _, _ = build(**kwargs)  # pyright: ignore[reportArgumentType]
    connector = fakes.DictConnector({"a": "alpha", "b": "beta", "c": "gamma"})

    report = await broken.run(connector)

    assert report.discovered == 3, "every document was attempted"
    assert report.clean, "a per-document failure is not a run failure"


async def test_a_fetch_failure_is_recorded_against_its_document_and_no_other() -> None:
    pipeline, store, _ = build()
    connector = fakes.DictConnector({"a": "alpha", "b": "beta"})
    connector.fail_fetch = {"a"}

    report = await pipeline.run(connector)

    assert report.indexed == 1
    assert report.by_status[DocumentStatus.FAILED.value] == 1
    assert await store.find_document("memory", "b") is not None


async def test_a_hook_that_raises_fails_one_document_and_is_not_disabled() -> None:
    """Auto-disabling a failing hook would make the corpus depend on ingest order."""
    hook = fakes.Exploder()
    pipeline, store, _ = build(middleware=(hook,))
    connector = fakes.DictConnector({"a": "alpha", "b": "beta"})

    await pipeline.run(connector)

    for source_id in ("a", "b"):
        document = await store.find_document("memory", source_id)
        assert document is not None
        assert document.failed_stage is PipelineStage.MIDDLEWARE, (
            "a hook failure is a plugin problem, not a problem with the stage it bounded"
        )


# --- parser chain outcomes ---------------------------------------------------------------------


async def test_a_chain_of_hard_failures_is_failed_and_never_unsupported() -> None:
    """The distinction the whole fallback design rests on.

    ``unsupported_media_type`` reads as "manicule does not handle this format" and sends
    whoever reads it to write a parser that already exists. A chain where every parser broke
    means something else entirely, and must say so.
    """
    pipeline, store, _ = build(
        parsers={"one": fakes.ExplodingParser(), "two": fakes.ExplodingParser()},
        chain=("one", "two"),
    )

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.failed_stage is PipelineStage.PARSE


async def test_a_chain_where_every_parser_declined_is_unsupported_media_type() -> None:
    """A parser that declined reported information; one that broke reported nothing."""
    pipeline, store, _ = build(
        parsers={"one": fakes.DecliningParser(), "two": fakes.DecliningParser()},
        chain=("one", "two"),
    )

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE


async def test_a_mixed_chain_of_a_failure_and_an_empty_parser_is_failed() -> None:
    """The case an implementation guesses at.

    A parser that broke leaves us genuinely not knowing whether text was there, so the
    scanned-corpus status must not absorb it — otherwise the 5% warning fires on library bugs.
    """
    pipeline, store, _ = build(
        parsers={"one": fakes.ExplodingParser(), "two": fakes.EmptyParser()},
        chain=("one", "two"),
    )

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.FAILED


async def test_a_document_with_no_extractable_text_is_stored_with_zero_chunks() -> None:
    """Storing the failure is what makes it countable, skippable and later repairable."""
    pipeline, store, _ = build(parsers={"empty": fakes.EmptyParser()}, chain=("empty",))

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.NO_EXTRACTABLE_TEXT
    assert document.status_detail
    assert store.chunks[document.id] == []


# --- change detection -----------------------------------------------------------------------


async def test_an_unchanged_version_token_skips_without_fetching() -> None:
    """Level 1 exists because level 2 requires a fetch, and over a rate-limited API that is
    the whole sync."""
    pipeline, _, _ = build()
    connector = fakes.DictConnector({"a": "alpha"})

    await pipeline.run(connector)
    connector.fetches.clear()
    report = await pipeline.run(connector)

    assert report.skipped_version == 1
    assert connector.fetches == [], "a level-1 skip must not touch the network"


async def test_a_touched_modification_date_with_unchanged_bytes_skips_at_level_two() -> None:
    """Level 1 can lie. Hashing the fetched bytes catches it before the expensive part."""
    pipeline, store, _ = build()
    connector = fakes.DictConnector({"a": "alpha"})
    await pipeline.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    store.documents[document.id] = document.model_copy(update={"version_token": "touched"})

    report = await pipeline.run(connector)

    assert report.skipped_hash == 1
    assert connector.fetches == ["a", "a"], "level 2 needs the bytes, so it does fetch"


async def test_a_skip_still_records_liveness_and_the_new_token() -> None:
    """Omitting either causes a specific bug: reconciliation cannot tell unchanged from gone,
    and a chatty source is re-fetched forever."""
    pipeline, store, _ = build()
    connector = fakes.DictConnector({"a": "alpha"})
    await pipeline.run(connector)
    document = await store.find_document("memory", "a")
    assert document is not None
    store.documents[document.id] = document.model_copy(update={"version_token": "touched"})

    await pipeline.run(connector)

    assert store.seen[document.id] >= 1
    refreshed = await store.find_document("memory", "a")
    assert refreshed is not None
    assert refreshed.version_token == content_hash("alpha")


async def test_a_document_requeued_after_a_crash_is_not_skipped() -> None:
    """The bug an allowlist of settled statuses exists to prevent.

    An interrupted run leaves a document ``pending`` with its token and hash already written.
    A skip rule that consulted only the token would skip it forever, and it would appear, to
    every count, to have been handled.
    """
    pipeline, store, _ = build()
    connector = fakes.DictConnector({"a": "alpha"})
    await pipeline.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    store.documents[document.id] = document.model_copy(
        update={"status": DocumentStatus.PENDING, "status_detail": None}
    )

    report = await pipeline.run(connector)

    assert report.skipped_version == 0
    assert report.indexed == 1


# --- a failed re-ingest never demotes a working document -----------------------------------------


async def test_a_failed_re_ingest_leaves_an_indexed_document_indexed() -> None:
    """A transient error during a routine re-sync must not remove a working document.

    The obvious shape — ``pending`` before parsing, ``failed`` after — makes a network blip
    unsearchable a document that was fine, while its chunks and vectors sit in both stores,
    intact and unreachable.
    """
    store = fakes.MemoryIngestStore()
    healthy, _, _ = build(store=store)
    connector = fakes.DictConnector({"a": "alpha"})
    await healthy.run(connector)
    document = await store.find_document("memory", "a")
    assert document is not None
    chunks_before = list(store.chunks[document.id])

    connector.documents["a"] = "alpha changed"
    broken, _, _ = build(store=store, parsers={"lines": fakes.ExplodingParser()})
    await broken.run(connector)

    after = await store.find_document("memory", "a")
    assert after is not None
    assert after.status is DocumentStatus.INDEXED
    assert store.chunks[document.id] == chunks_before
    assert after.content_hash == document.content_hash, (
        "the stored hash still describes the bytes the stored chunks came from"
    )
    assert "last_ingest_error" in after.metadata, (
        "the failure is still on the record; it simply does not cost a working document"
    )


async def test_a_re_ingest_that_finds_no_text_does_replace_an_indexed_document() -> None:
    """The exception, and it is not a softening.

    ``no_extractable_text`` is a conclusion about content that genuinely changed. Continuing
    to serve chunks derived from bytes the source no longer has would cite text the document
    does not contain, which is the one thing this project will not do.
    """
    store = fakes.MemoryIngestStore()
    healthy, _, _ = build(store=store)
    connector = fakes.DictConnector({"a": "alpha"})
    await healthy.run(connector)

    connector.documents["a"] = "replaced by a scan"
    empty, _, _ = build(store=store, parsers={"empty": fakes.EmptyParser()}, chain=("empty",))
    await empty.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.NO_EXTRACTABLE_TEXT
    assert store.chunks[document.id] == []


async def test_a_document_being_re_ingested_is_never_marked_in_flight() -> None:
    """In-flight statuses are not servable, so writing one over ``indexed`` unserves it."""
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store)
    connector = fakes.DictConnector({"a": "alpha"})
    await pipeline.run(connector)
    document = await store.find_document("memory", "a")
    assert document is not None

    seen: list[DocumentStatus] = []
    original = store.set_status

    async def record(document_id: str, status: DocumentStatus, detail: str = "") -> None:
        seen.append(status)
        await original(document_id, status, detail)

    store.set_status = record  # a spy over the fake, not a stub of the thing under test
    connector.documents["a"] = "alpha changed"
    await pipeline.run(connector)

    assert seen == [], "an indexed document keeps its status until there is a replacement"


# --- middleware in the pipeline ---------------------------------------------------------------


async def test_a_hook_returning_none_before_parse_skips_the_document() -> None:
    """The one short-circuit, and it is an ordinary outcome rather than a failure."""
    pipeline, store, _ = build(middleware=(fakes.Skipper(),))

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.SKIPPED


async def test_a_hook_rewriting_chunk_text_fails_the_document_rather_than_the_corpus() -> None:
    pipeline, store, _ = build(middleware=(fakes.TextRewriter(),))

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.failed_stage is PipelineStage.MIDDLEWARE


# --- containers -----------------------------------------------------------------------------


async def test_an_archive_becomes_a_container_and_its_members_become_documents() -> None:
    """A member of an archive is a document in its own right, not a chunk of the archive.

    Without this the archive parser's zero blocks read as an empty document and a zip full of
    reports is reported as though it were a scan.
    """
    pipeline, store, _ = build(
        parsers={"archive": fakes.FakeArchive(), "lines": fakes.LineParser()},
        chain=("archive", "lines"),
    )
    connector = fakes.DictConnector({"bundle": "one=alpha\ntwo=beta"})
    connector.media_types["bundle"] = fakes.CONTAINER_MEDIA_TYPE

    report = await pipeline.run(connector)

    container = await store.find_document("memory", "bundle")
    assert container is not None
    assert container.status is DocumentStatus.CONTAINER
    assert store.chunks[container.id] == [], "a container has zero chunks by design"
    assert report.by_status[DocumentStatus.INDEXED.value] == 2
    assert await store.find_document("memory", "bundle!/one") is not None


async def test_a_member_that_could_not_be_read_is_stored_with_its_reason() -> None:
    """ "The archive had 200 files and we indexed 197" is not a fact anybody would discover."""
    pipeline, store, _ = build(
        parsers={"archive": fakes.FakeArchive(), "lines": fakes.LineParser()},
        chain=("archive", "lines"),
    )
    connector = fakes.DictConnector({"bundle": "one=alpha\ntwo=!encrypted"})
    connector.media_types["bundle"] = fakes.CONTAINER_MEDIA_TYPE

    await pipeline.run(connector)

    failed = await store.find_document("memory", "bundle!/two")
    assert failed is not None
    assert failed.status is DocumentStatus.FAILED
    assert failed.status_detail == "member is encrypted"


# --- retained bytes ---------------------------------------------------------------------------


async def test_retained_bytes_are_recorded_against_the_document() -> None:
    blobs = fakes.MemoryBlobs()
    pipeline, store, _ = build(blobs=blobs)

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.original_ref == content_hash(b"alpha")
    assert store.originals[document.id] == (content_hash(b"alpha"), None)


async def test_bytes_that_were_not_retained_record_the_reason_rather_than_nothing() -> None:
    """Absent with a stated reason, visible in diagnostics, never a silent partial success."""
    pipeline, store, _ = build(blobs=fakes.MemoryBlobs(max_bytes=1))

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.original_ref is None
    _, reason = store.originals[document.id]
    assert reason is not None
    assert "exceeds the cap" in reason


async def test_a_document_larger_than_the_fetch_cap_fails_at_fetch() -> None:
    """The cap is enforced after the body arrives, because that is when its size is known."""
    pipeline, store, _ = build(max_fetch_bytes=2)

    await pipeline.run(fakes.DictConnector({"a": "alpha and more"}))

    assert await store.find_document("memory", "a") is None, (
        "nothing was fetched, so there is no content hash and no row to write"
    )


# --- run counters ------------------------------------------------------------------------------


async def test_the_watermark_advances_only_on_a_clean_run() -> None:
    """The whole of resumability: an interrupted sync re-enumerates from the last good point.

    Advancing after a run that failed part-way would skip every document the run never
    reached, on every future sync, silently — which is the opposite of resuming.
    """
    pipeline, store, _ = build()
    connector = _Positioned({"a": "alpha"})

    await pipeline.run(connector)
    assert store.watermarks["memory"].value == "position-1"

    connector.fail_discovery = True
    connector.position = "position-2"
    await pipeline.run(connector)

    assert store.watermarks["memory"].value == "position-1", (
        "a run that did not finish is not a run whose position may be believed"
    )


async def test_a_connector_that_cannot_report_a_position_writes_no_watermark() -> None:
    """A source with no change signal has nothing to report, and an invented one gets believed."""
    pipeline, store, _ = build()

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    assert store.watermarks == {}


async def test_the_bytes_that_are_retained_are_the_ones_the_connector_returned() -> None:
    """Retention happens before any hook, and both halves of that matter.

    ``content_hash`` is the hash of what the connector returned, so retaining post-hook bytes
    would leave the reference and the hash describing different content. And re-parse feeds
    retained bytes back through this same path — hooks included — so retaining transformed
    bytes would apply ``before_parse`` twice, compounding on every repair.
    """
    blobs = fakes.MemoryBlobs()
    pipeline, store, _ = build(blobs=blobs, middleware=(_Prefixing(),))

    await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.original_ref == document.content_hash, (
        "storage.md §4.2: original_ref is the same value as content_hash when retention worked"
    )
    assert blobs.data[document.content_hash] == b"alpha"


async def test_members_of_a_container_do_not_consume_a_discovery_limit() -> None:
    """One archive of five hundred members must not exhaust a limit of ten."""
    pipeline, _, _ = build(
        parsers={"archive": fakes.FakeArchive(), "lines": fakes.LineParser()},
        chain=("archive", "lines"),
    )
    connector = fakes.DictConnector({"bundle": "one=alpha\ntwo=beta\nthree=gamma"})
    connector.media_types["bundle"] = fakes.CONTAINER_MEDIA_TYPE

    report = await pipeline.run(connector)

    assert report.discovered == 1
    assert report.expanded == 3


class _Positioned(fakes.DictConnector):
    """A connector that can say how far it got, and can fail part-way through saying it."""

    def __init__(self, documents: Mapping[str, str]) -> None:
        super().__init__(documents)
        self.position = "position-1"
        self.fail_discovery = False

    @property
    def watermark(self) -> Watermark:
        return Watermark(value=self.position, observed_at=datetime.now(UTC))

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        async for found in super().discover(watermark):
            yield found
        if self.fail_discovery:
            msg = "the search cursor expired"
            raise RuntimeError(msg)


class _Prefixing(fakes.PassThrough):
    """Rewrites the fetched bytes before parsing, which a hook is allowed to do."""

    name = "prefixing"

    @override
    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        return raw.model_copy(update={"content": f"# header\n{raw.as_text()}"})


async def test_run_counters_land_on_the_connector_row() -> None:
    """Diagnostic, not relational: overwritten rather than accumulated, and no new table."""
    pipeline, store, _ = build()

    await pipeline.run(fakes.DictConnector({"a": "alpha", "b": "beta"}))

    recorded = store.connector_meta["memory"]["last_run"]
    assert isinstance(recorded, dict)
    assert recorded["discovered"] == 2
    assert recorded["by_status"] == {DocumentStatus.INDEXED.value: 2}


async def test_a_discovery_failure_marks_the_run_unclean_without_losing_what_was_done() -> None:
    """The watermark advances only on a clean run, so this is what stops it advancing."""
    pipeline, _, _ = build()

    class Failing(fakes.DictConnector):
        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            del watermark
            yield discovered("a", "alpha")
            msg = "the search cursor expired"
            raise RuntimeError(msg)

    report = await pipeline.run(Failing({"a": "alpha"}))

    assert report.indexed == 1
    assert not report.clean
    assert "cursor expired" in report.error


async def test_ingesting_the_same_bytes_twice_produces_the_same_chunk_ids() -> None:
    """Identity is derived, so a re-parse replaces rows rather than accumulating them."""
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store)
    raw = RawDocument(source_id="a", uri="memory://a", media_type=MEDIA_TYPE, content="alpha\nbeta")

    first = await pipeline.ingest_raw(raw, source="memory")
    before = [chunk.id for chunk in store.chunks[first[0].document_id]]
    await pipeline.ingest_raw(raw, source="memory", force=True)

    assert [chunk.id for chunk in store.chunks[first[0].document_id]] == before
