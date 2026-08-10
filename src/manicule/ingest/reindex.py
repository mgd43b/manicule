"""Re-ingest from what is already on disk. Three verbs, three rungs of the ladder.

| Verb | Reads | Rung | Network |
|---|---|---|---|
| :func:`repair` | ``chunks`` | 1-2 | none |
| :func:`re_embed` | ``chunks.embed_text`` | 2 | none |
| :func:`re_parse` | ``blobs`` | 3 | none |
| a forced sync | the source | 4 | yes, rate-limited, **may fail** |

Only the last can fail for reasons outside the machine, and it is the only one that is not
reproducible. Everything above it is a pure function of what is already stored. That is the
whole return on retaining original bytes, and it is why ``--re-parse`` is a first-class verb
rather than a flag on sync.

**Selection is a query, never a scan**, because ``documents.chunk_fp`` and
``documents.embed_fp`` record per-document lineage: a tree-sitter grammar upgrade invalidates
code documents and nothing else, and saying so is one ``WHERE`` clause.

**A document with no retained bytes cannot be re-parsed**, and that is reported per document
rather than failing the run. Those are the only documents for which a re-crawl is the sole
repair, and they are exactly the ones an operator needs named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from manicule.core.content import DocumentStatus, RawDocument
from manicule.core.errors import ContextOverflowError
from manicule.ingest.embedding import embed_chunks

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from manicule.core.content import Document
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Embedder, VectorStore
    from manicule.ingest.pipeline import BlobSink, IngestPipeline
    from manicule.ingest.ports import IngestStore


@dataclass
class ReindexReport:
    """What one repair pass did, and what it could not do."""

    documents: int = 0
    chunks: int = 0
    unrepairable: list[str] = field(default_factory=list[str])
    """Documents naming the reason they could not be repaired, one line each."""

    failures: list[str] = field(default_factory=list[str])

    def note_unrepairable(self, document: Document, reason: str) -> None:
        self.unrepairable.append(f"{document.id} ({document.uri}): {reason}")


async def select(
    store: IngestStore,
    *,
    source: str | None = None,
    statuses: Collection[DocumentStatus] | None = None,
    media_types: Collection[str] | None = None,
    chunk_fingerprint: ChunkFingerprint | None = None,
    limit: int | None = None,
) -> Sequence[Document]:
    """The documents a repair verb should run over.

    ``chunk_fingerprint`` selects documents built by *something else* — the shape that makes a
    grammar upgrade a targeted repair rather than a corpus-wide rebuild.
    """
    return await store.select_documents(
        source=source,
        statuses=statuses,
        media_types=media_types,
        chunk_fp_other_than=chunk_fingerprint.canonical() if chunk_fingerprint else None,
        limit=limit,
    )


async def re_embed(
    documents: Sequence[Document],
    *,
    store: IngestStore,
    embedder: Embedder,
    vectors: VectorStore,
    chunk_fingerprint: ChunkFingerprint,
    target_batch_tokens: int = 16_384,
) -> ReindexReport:
    """Rebuild vectors from stored ``embed_text``. Rung 2: no parser, no network.

    **This is the path that most needs** :func:`~manicule.core.embedding.require_within_context`,
    and it gets it by going through :func:`~manicule.ingest.embedding.embed_chunks` like every
    other path. Re-embedding does not re-chunk, so the chunker's budget refusal never runs
    here; and a model reconfigured to a shorter sequence length leaves the embedding
    fingerprint identical, so no comparison fires either. Without the check, one command would
    truncate a whole corpus in silence.
    """
    report = ReindexReport()
    for document in documents:
        chunks = await store.document_chunks(document.id)
        if not chunks:
            if document.expects_chunks:
                report.note_unrepairable(document, "no stored chunks to re-embed")
            continue
        try:
            produced = await embed_chunks(
                embedder,
                chunks,
                chunk_fingerprint=chunk_fingerprint,
                target_batch_tokens=target_batch_tokens,
            )
        except ContextOverflowError as exc:
            # Fatal to this document and to nothing else. There is no partial-credit answer:
            # an embedding of the first half of a passage is not a worse embedding of the
            # passage, it is an embedding of something else.
            report.failures.append(f"{document.id}: {exc}")
            continue
        await vectors.upsert(chunks, produced)
        # Only the embedding lineage moves. Re-embedding does not re-chunk, so claiming a new
        # chunk fingerprint here would make a later "which documents need re-chunking" query
        # answer "none" about documents that do.
        await store.set_lineage(
            document.id, chunk_fp=None, embed_fp=embedder.fingerprint.canonical()
        )
        report.documents += 1
        report.chunks += len(chunks)
    return report


async def repair(
    documents: Sequence[Document],
    *,
    store: IngestStore,
    embedder: Embedder,
    vectors: VectorStore,
    chunk_fingerprint: ChunkFingerprint,
    target_batch_tokens: int = 16_384,
) -> ReindexReport:
    """Finish documents an interrupted run left half-written. Rungs 1-2.

    The crash windows in ``docs/storage.md`` §8.2 both land here. Chunks written and vectors
    not: re-embed those chunks. Vectors written and the document never marked ``indexed``:
    the upsert is idempotent by chunk id, so re-running it costs nothing and the document is
    then marked. A document with no chunks and a non-terminal status has nothing to finish and
    is named instead, because the repair it needs is a re-parse.
    """
    report = ReindexReport()
    for document in documents:
        chunks = await store.document_chunks(document.id)
        if not chunks:
            report.note_unrepairable(
                document, "no stored chunks; repair from retained bytes with re-parse"
            )
            continue
        try:
            produced = await embed_chunks(
                embedder,
                chunks,
                chunk_fingerprint=chunk_fingerprint,
                target_batch_tokens=target_batch_tokens,
            )
        except ContextOverflowError as exc:
            report.failures.append(f"{document.id}: {exc}")
            continue
        await vectors.upsert(chunks, produced)
        await store.upsert_document(
            document.model_copy(
                update={
                    "status": DocumentStatus.INDEXED,
                    "status_detail": None,
                    "failed_stage": None,
                }
            )
        )
        await store.set_lineage(
            document.id,
            chunk_fp=chunk_fingerprint.canonical(),
            embed_fp=embedder.fingerprint.canonical(),
        )
        report.documents += 1
        report.chunks += len(chunks)
    return report


async def re_parse(
    documents: Sequence[Document],
    *,
    pipeline: IngestPipeline,
    blobs: BlobSink,
) -> ReindexReport:
    """Run the current parser chain over retained bytes. Rung 3: no network.

    Identity rules are the same as first ingest, because it is the same code path: chunks are
    reconciled against the stored set by ``chunks.id``, and an id is derived from its content,
    so a chunk that survives a re-parse unchanged keeps its id and therefore its vector. A
    parser fix that changes one table in a hundred-page document re-embeds one table.

    A document whose bytes were never retained — the cap refused them, or it predates
    retention — is named with the reason rather than failing the run. Those are the only
    documents for which a re-crawl is the only repair.
    """
    report = ReindexReport()
    for document in documents:
        if document.original_ref is None:
            report.note_unrepairable(
                document,
                "no retained bytes (original_ref is unset); only a forced re-sync can repair it",
            )
            continue
        data = await blobs.get(document.original_ref)
        if data is None:
            report.note_unrepairable(
                document,
                f"retained bytes {document.original_ref} are missing from the blob store; "
                f"restore the data directory, or force a re-sync of this document",
            )
            continue
        raw = RawDocument(
            source_id=document.source_id,
            uri=document.uri,
            media_type=document.media_type,
            content=data,
            metadata=dict(document.metadata),
        )
        outcomes = await pipeline.ingest_raw(
            raw,
            source=document.source,
            version_token=document.version_token,
            title=document.title,
            existing=document,
            force=True,
        )
        for outcome in outcomes:
            if outcome.status is DocumentStatus.FAILED:
                report.failures.append(
                    f"{outcome.document_id or outcome.source_id}: {outcome.detail}"
                )
            else:
                report.documents += 1
                report.chunks += outcome.chunks
    return report


__all__ = ["ReindexReport", "re_embed", "re_parse", "repair", "select"]
