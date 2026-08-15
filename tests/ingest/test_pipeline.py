"""One document at a time, and none of them able to stop the rest.

The happy path is one test here. The rest are failures, because a pipeline whose purpose is
surviving failure is certified by nothing if only its happy path is exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING, override

import pytest

from manicule.core.content import Commit, DocumentStatus, PipelineStage, RawDocument, Retention
from manicule.core.errors import PolicyError
from manicule.core.fingerprints import ParseFingerprint
from manicule.core.ids import content_hash
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.ingest.capacity import CapacityDiagnostic, CapacityRefusedError, CapacityResource
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import BlobSink, IngestPipeline
from manicule.ingest.workers import InProcessRunner
from manicule.parsers.versions import parse_fingerprint
from tests.fakes import MEDIA_TYPE, HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence

    from manicule.core.content import Chunk, Document, DocumentRevision, ParsedBlock
    from manicule.core.glossary import GlossaryEntry
    from manicule.core.protocols import Chunker, Embedder, Middleware


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
    routes: Mapping[str, Sequence[str]] | None = None,
    parse_fingerprints: Callable[[str], ParseFingerprint | None] = parse_fingerprint,
    detect_glossary: bool = True,
    chunker: Chunker | None = None,
) -> tuple[IngestPipeline, fakes.MemoryIngestStore, fakes.MemoryVectors]:
    """A pipeline over in-memory everything, plus the store and vectors to assert against.

    ``routes`` resolves a chain per media type, for the tests that need two parsers to own two
    documents. ``chain`` is the single-chain shorthand every other test uses. ``chunker`` is for
    the tests that need the breadcrumb to move without the text moving, which no parser can do.
    """
    store = store or fakes.MemoryIngestStore()
    vectors = vectors or fakes.MemoryVectors()
    chunker = chunker or fakes.BlockChunker()

    def resolve(media_type: str) -> Sequence[str]:
        return list(chain if routes is None else routes[media_type])

    pipeline = IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=embedder or HashEmbedder(),
        vectors=vectors,
        runner=InProcessRunner(parsers or {"lines": fakes.LineParser()}),
        resolve_chain=resolve,
        middleware=MiddlewareRunner(middleware),
        chunk_fingerprint=chunker.fingerprint,
        blobs=blobs,
        max_fetch_bytes=max_fetch_bytes,
        parse_fingerprints=parse_fingerprints,
        detect_glossary=detect_glossary,
    )
    return pipeline, store, vectors


def parse_versions(**libraries: str) -> Callable[[str], ParseFingerprint | None]:
    """A parse-fingerprint source with a settable library version per parser name.

    A parser absent from ``libraries`` records nothing, which is how a third-party parser
    behaves: manicule cannot read its version, so it claims none.
    """

    def lookup(parser: str) -> ParseFingerprint | None:
        if parser not in libraries:
            return None
        return ParseFingerprint(
            parser=parser, version="1", libraries={f"lib-{parser}": libraries[parser]}
        )

    return lookup


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
    assert not store.chunks.get(document.id), "failed vectors must publish no relational chunks"


async def test_republishing_the_identical_generation_does_not_tombstone_active_vectors() -> None:
    """A content-addressed retry may reuse its physical rows, so it must not stage their cleanup."""
    pipeline, store, _ = build()
    connector = fakes.DictConnector({"a": "alpha"})
    await pipeline.run(connector)
    before = await store.find_document("memory", "a")
    assert before is not None
    staged = list(store.staged_publications)

    await pipeline.ingest_raw(
        RawDocument(source_id="a", uri="memory://a", media_type=MEDIA_TYPE, content="alpha"),
        source="memory",
        version_token=content_hash("alpha"),
        force=True,
    )

    after = await store.find_document("memory", "a")
    assert after is not None
    assert after.publication_id == before.publication_id
    assert store.staged_publications == staged


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


async def test_zero_chunk_publication_loses_its_cas_without_erasing_the_winner() -> None:
    """The empty branch uses the same guarded relational flip as a vector-bearing revision."""
    pipeline, store, vectors = build()
    await pipeline.run(fakes.DictConnector({"a": "old text"}))
    stale = await store.find_document("memory", "a")
    assert stale is not None
    old_chunks = list(store.chunks[stale.id])

    class OvertakingEmptyChunker(fakes.BlockChunker):
        @override
        def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
            del document, blocks
            winner = store.documents[stale.id].model_copy(
                update={"content_hash": content_hash("winner"), "version_token": "winner"}
            )
            store.documents[stale.id] = winner
            return []

    contender, _, _ = build(store=store, vectors=vectors, chunker=OvertakingEmptyChunker())
    outcomes = await contender.ingest_raw(
        RawDocument(
            source_id="a", uri="memory://a", media_type=MEDIA_TYPE, content="stale replacement"
        ),
        source="memory",
        force=True,
        expected=stale.revision,
    )

    assert outcomes[0].superseded
    assert store.documents[stale.id].content_hash == content_hash("winner")
    assert store.chunks[stale.id] == old_chunks


@pytest.mark.parametrize("conclusion", ["parser-empty", "unsupported", "container", "skipped"])
async def test_every_chunkless_conclusion_loses_its_cas_without_erasing_the_winner(
    conclusion: str,
) -> None:
    """Every successful zero-chunk exit reaches one guarded publication boundary."""

    class OvertakenStore(fakes.MemoryIngestStore):
        armed = False

        @override
        async def publish_document(
            self,
            document: Document,
            chunks: Sequence[Chunk],
            *,
            expected: DocumentRevision | None,
            chunk_fp: str | None,
            embed_fp: str | None,
            parse_fp: str | None,
            glossary_entries: Sequence[GlossaryEntry] | None,
            glossary_fp: str | None,
            original_omitted_reason: str | None,
        ) -> Commit:
            if self.armed:
                self.armed = False
                current = self.documents[document.id]
                self.documents[document.id] = current.model_copy(
                    update={"content_hash": content_hash("winner"), "version_token": "winner"}
                )
            return await super().publish_document(
                document,
                chunks,
                expected=expected,
                chunk_fp=chunk_fp,
                embed_fp=embed_fp,
                parse_fp=parse_fp,
                glossary_entries=glossary_entries,
                glossary_fp=glossary_fp,
                original_omitted_reason=original_omitted_reason,
            )

    store = OvertakenStore()
    healthy, _, vectors = build(store=store)
    await healthy.run(fakes.DictConnector({"a": "old text"}))
    stale = await store.find_document("memory", "a")
    assert stale is not None
    old_chunks = list(store.chunks[stale.id])
    store.armed = True

    raw = RawDocument(source_id="a", uri="memory://a", media_type=MEDIA_TYPE, content="replacement")
    if conclusion == "parser-empty":
        contender, _, _ = build(
            store=store,
            vectors=vectors,
            parsers={"empty": fakes.EmptyParser()},
            chain=("empty",),
        )
    elif conclusion == "unsupported":
        contender, _, _ = build(
            store=store,
            vectors=vectors,
            parsers={"decline": fakes.DecliningParser()},
            chain=("decline",),
        )
    elif conclusion == "container":
        raw = raw.model_copy(
            update={"media_type": fakes.CONTAINER_MEDIA_TYPE, "content": "member=child text"}
        )
        contender, _, _ = build(
            store=store,
            vectors=vectors,
            parsers={"archive": fakes.FakeArchive()},
            chain=("archive",),
        )
    else:
        contender, _, _ = build(store=store, vectors=vectors, middleware=(fakes.Skipper(),))

    outcomes = await contender.ingest_raw(
        raw,
        source="memory",
        force=True,
        expected=stale.revision,
    )

    assert outcomes[0].superseded
    assert store.documents[stale.id].content_hash == content_hash("winner")
    assert store.chunks[stale.id] == old_chunks


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
    indexed = await store.find_document("memory", "a")
    assert indexed is not None
    store.parse_lineage[indexed.id] = "old-parser"

    connector.documents["a"] = "replaced by a scan"
    empty, _, _ = build(store=store, parsers={"empty": fakes.EmptyParser()}, chain=("empty",))
    await empty.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.NO_EXTRACTABLE_TEXT
    assert store.chunks[document.id] == []
    assert store.lineage[document.id] == (None, None)
    assert document.id not in store.parse_lineage


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


async def test_capacity_refusal_is_not_downgraded_to_a_retention_omission() -> None:
    private_source = "private-source-cinder"
    private_body = "private body cinder"

    class RefusingBlobs(fakes.MemoryBlobs):
        @override
        async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
            del data, media_type
            raise CapacityRefusedError(
                CapacityDiagnostic(
                    resource=CapacityResource.DISK_HEADROOM_BYTES,
                    limit=100,
                    used=90,
                    requested=20,
                )
            )

    pipeline, store, _ = build(blobs=RefusingBlobs())
    report = await pipeline.run(fakes.DictConnector({private_source: private_body}))

    assert report.error_type == "CapacityRefusedError"
    assert report.watermark_advanced is False
    assert await store.find_document("memory", private_source) is None
    rendered = repr(report.as_metadata())
    assert "disk_headroom_bytes" in rendered
    assert private_source not in rendered
    assert private_body not in rendered


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
    @override
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


# --- parse lineage ----------------------------------------------------------------------------


async def test_a_document_records_the_parser_version_that_produced_its_text() -> None:
    """Without this column, two generations of extracted text look identical in the corpus.

    Change detection compares the connector's source bytes, which a library upgrade does not
    touch — so nothing already stored is ever re-read, while a newly ingested document with
    identical bytes parses differently.
    """
    pipeline, store, _ = build(parse_fingerprints=parse_versions(lines="1.0"))

    await pipeline.run(fakes.DictConnector({"a": "alpha\nbeta"}))

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.parse_fp is not None
    assert "1.0" in document.parse_fp


async def test_a_parser_bump_re_parses_its_documents_and_leaves_the_others_alone() -> None:
    """Selective invalidation, checked in both directions in one run.

    Two documents, two parsers, one library bump. The document that parser produced must be
    re-parsed; the other must not be re-fetched or re-parsed at all. Asserting only the first
    half would pass for an implementation that re-parses the entire corpus on any bump, which
    is the expensive mistake this field exists to avoid.
    """
    store = fakes.MemoryIngestStore()
    parsers = {"lines": fakes.LineParser(), "other": fakes.LineParser()}
    routes = {"memory/a": ("lines",), "memory/b": ("other",)}
    connector = fakes.DictConnector({"a": "alpha", "b": "beta"})
    connector.media_types = {"a": "memory/a", "b": "memory/b"}

    pipeline, _, _ = build(
        store=store,
        parsers=parsers,
        routes=routes,
        parse_fingerprints=parse_versions(lines="1.0", other="1.0"),
    )
    await pipeline.run(connector)

    first_a = await store.find_document("memory", "a")
    first_b = await store.find_document("memory", "b")
    assert first_a is not None
    assert first_b is not None
    before_b = [chunk.id for chunk in store.chunks[first_b.id]]

    bumped, _, _ = build(
        store=store,
        parsers=parsers,
        routes=routes,
        parse_fingerprints=parse_versions(lines="2.0", other="1.0"),
    )
    report = await bumped.run(connector)

    after_a = await store.find_document("memory", "a")
    after_b = await store.find_document("memory", "b")
    assert after_a is not None
    assert after_b is not None
    assert after_a.parse_fp != first_a.parse_fp, "the bumped parser's document must be re-parsed"
    assert "2.0" in (after_a.parse_fp or "")
    assert after_b.parse_fp == first_b.parse_fp, "the other parser's document must be left alone"
    assert report.skipped_version == 1, "and left alone means skipped before the fetch"
    assert [chunk.id for chunk in store.chunks[after_b.id]] == before_b


async def test_a_parser_bump_is_noticed_even_when_the_source_reports_no_change() -> None:
    """Level 1 needs the check as much as level 2, and for the sharper reason.

    A connector whose version token has not moved never reaches the byte comparison, so a
    check placed only at level 2 would leave every well-behaved source's corpus permanently
    stale — the documents would be skipped before anything looked at their bytes.
    """
    store = fakes.MemoryIngestStore()
    connector = fakes.DictConnector({"a": "alpha"})
    first, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="1.0"))
    await first.run(connector)
    assert connector.fetches == ["a"]

    unchanged, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="1.0"))
    await unchanged.run(connector)
    assert connector.fetches == ["a"], "an unchanged document must still skip before the fetch"

    bumped, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="2.0"))
    report = await bumped.run(connector)

    assert connector.fetches == ["a", "a"], "a moved parser must defeat the version-token skip"
    assert report.skipped_version == 0
    document = await store.find_document("memory", "a")
    assert document is not None
    assert "2.0" in (document.parse_fp or "")


async def test_a_parser_bump_is_noticed_by_a_source_that_reports_no_version_at_all() -> None:
    """Level 2 on its own, which is the only level a tokenless connector ever reaches.

    ``docs/ingest.md`` §4 notes that a connector supplying no ``version_token`` falls straight
    to the byte comparison. For those sources the level-1 check never runs, so the level-2 one
    is the whole of the guard — and the two are separate lines of code that a change could
    remove independently.
    """

    class Tokenless(fakes.DictConnector):
        """A source with no change signal, which is a real and supported shape."""

        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            del watermark
            for source_id in sorted(self.documents):
                yield DiscoveredDoc(
                    ref=DocRef(source_id=source_id, uri=f"memory://{source_id}"),
                    version_token=None,
                    media_type=MEDIA_TYPE,
                )

    store = fakes.MemoryIngestStore()
    connector = Tokenless({"a": "alpha"})
    first, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="1.0"))
    await first.run(connector)

    unchanged, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="1.0"))
    report = await unchanged.run(connector)
    assert report.skipped_hash == 1, "identical bytes still skip the parse, chunk and embed"

    bumped, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="2.0"))
    report = await bumped.run(connector)

    assert report.skipped_hash == 0, "a moved parser must defeat the content-hash skip"
    assert report.indexed == 1
    document = await store.find_document("memory", "a")
    assert document is not None
    assert "2.0" in (document.parse_fp or "")


async def test_a_parser_manicule_cannot_version_does_not_re_parse_on_every_sync() -> None:
    """``None`` on both sides is agreement, not ignorance.

    A third-party parser's version is not something this repository can read, so nothing is
    recorded and nothing is expected. Treating that as changed would re-parse a plugin corpus
    forever to learn nothing.
    """
    store = fakes.MemoryIngestStore()
    connector = fakes.DictConnector({"a": "alpha"})
    pipeline, _, _ = build(store=store, parse_fingerprints=parse_versions())

    await pipeline.run(connector)
    await pipeline.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.parse_fp is None
    assert connector.fetches == ["a"], "no recorded lineage must not mean re-fetch every sync"


async def test_a_document_no_parser_claimed_records_no_lineage_and_still_settles() -> None:
    """ "The chain found no text" names no parser, so there is no parser version to record.

    Every parser in the chain was tried and none produced a block, so ``parser_used`` is unset
    — and a fingerprint naming one of them would be a guess about which version decided the
    outcome. That is not a gap: ``no_extractable_text`` is already the selector for the
    re-parse pass that runs when the decision is revisited, which is what
    ``DocumentStatus.NO_EXTRACTABLE_TEXT`` documents. What must not happen is the document
    counting as stale on every sync and being re-fetched forever.
    """
    store = fakes.MemoryIngestStore()
    connector = fakes.DictConnector({"a": "   "})
    pipeline, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="1.0"))

    await pipeline.run(connector)
    await pipeline.run(connector)

    document = await store.find_document("memory", "a")
    assert document is not None
    assert document.status is DocumentStatus.NO_EXTRACTABLE_TEXT
    assert document.parse_fp is None
    assert connector.fetches == ["a"]


async def test_a_container_records_the_parser_version_that_expanded_it() -> None:
    """An archive's member set is a conclusion a parser version reached about these bytes.

    It is stored — the members are documents — so the lineage goes with it, and a bump to the
    expander re-expands rather than leaving a member list nothing can date.
    """
    store = fakes.MemoryIngestStore()
    connector = fakes.DictConnector({"bundle": "one=alpha"})
    connector.media_types["bundle"] = fakes.CONTAINER_MEDIA_TYPE
    pipeline, _, _ = build(
        store=store,
        parsers={"archive": fakes.FakeArchive(), "lines": fakes.LineParser()},
        chain=("archive", "lines"),
        parse_fingerprints=parse_versions(archive="1.0"),
    )

    await pipeline.run(connector)

    container = await store.find_document("memory", "bundle")
    assert container is not None
    assert container.status is DocumentStatus.CONTAINER
    assert "1.0" in (container.parse_fp or ""), (
        "the archive parser is what decided this member set, and it has a version"
    )


async def test_a_pipeline_cannot_be_built_on_estimated_chunk_boundaries() -> None:
    """The boundary in code, not only the one an operator meets.

    ``check_before_run`` is the once-per-run refusal, and a pipeline is constructible without
    going through it — while everything a pipeline writes is permanent. A chunker counting
    with a stand-in vocabulary must not be able to reach a store at all.
    """
    chunker = fakes.BlockChunker()
    provisional = chunker.fingerprint.model_copy(
        update={"tokenizer_id": "provisional:x1.5:tiktoken/cl100k_base@0.13.0"}
    )

    with pytest.raises(PolicyError, match="stand-in"):
        IngestPipeline(
            store=fakes.MemoryIngestStore(),
            chunker=chunker,
            embedder=HashEmbedder(),
            vectors=fakes.MemoryVectors(),
            runner=InProcessRunner({"lines": fakes.LineParser()}),
            resolve_chain=lambda _: ["lines"],
            middleware=MiddlewareRunner(()),
            chunk_fingerprint=provisional,
        )


async def test_an_uninstalled_parser_library_fails_one_document_and_not_the_run() -> None:
    """Reading a version can raise, and change detection is not the place to let it.

    ``parse_fingerprint`` raises for a distribution that is not installed, which is right
    where a repair is being planned — a partial set of current fingerprints is a repair that
    cannot succeed. Inside change detection it would escape into the discovery loop and end
    the enumeration, taking every other document in the batch with it. One document's problem
    must stay one document's.
    """

    def missing(parser: str) -> ParseFingerprint | None:
        msg = f"No package metadata was found for the library behind {parser!r}"
        raise PackageNotFoundError(msg)

    store = fakes.MemoryIngestStore()
    connector = fakes.DictConnector({"a": "alpha", "b": "beta"})
    seeded, _, _ = build(store=store, parse_fingerprints=parse_versions(lines="1.0"))
    await seeded.run(connector)

    broken, _, _ = build(store=store, parse_fingerprints=missing)
    report = await broken.run(connector)

    assert report.clean, "an unreadable version must not end the enumeration"
    assert report.discovered == 2, "and every document must still be attempted"
