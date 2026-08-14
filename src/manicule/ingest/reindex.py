"""Re-ingest from what is already on disk. Four verbs, four rungs of the ladder.

| Verb | Reads | Rung | Network |
|---|---|---|---|
| :func:`redetect_glossary` | ``chunks.text`` | 0 | none |
| :func:`repair` | ``chunks`` | 1-2 | none |
| :func:`re_embed` | ``chunks.embed_text`` | 2 | none |
| :func:`re_parse` | ``blobs`` | 3 | none |
| a forced sync | the source | 4 | yes, rate-limited, **may fail** |

Only the last can fail for reasons outside the machine, and it is the only one that is not
reproducible. Everything above it is a pure function of what is already stored. That is the
whole return on retaining original bytes, and it is why ``--re-parse`` is a first-class verb
rather than a flag on sync.

**Rung 0 is cheaper than rung 1 and that is why it is a separate verb rather than a wider
sweep.** Re-detecting a glossary reads chunk text and writes rows: it runs no parser, opens no
blob, and never reaches the embedder — so it costs no GPU time and does not touch a vector.
Folding it into the re-parse sweep would work and would charge a corpus-sized parse and
re-embed for a change to a regular expression, which is the thing an operator most needs to be
able to avoid.

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
from manicule.core.errors import ContextOverflowError, PolicyError
from manicule.ingest.embedding import EmbeddingWork, embed_chunks, embed_or_reuse

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from manicule.core.content import Chunk, Document
    from manicule.core.fingerprints import ChunkFingerprint, GlossaryFingerprint, ParseFingerprint
    from manicule.core.glossary import GlossaryEntry
    from manicule.core.protocols import Embedder, VectorStore
    from manicule.ingest.pipeline import BlobSink, IngestPipeline
    from manicule.ingest.ports import GlossaryStore, IngestStore


NO_RETAINED_BYTES = "no retained bytes (original_ref is unset); only a forced re-sync can repair it"
"""Why a document cannot be re-parsed, worded once.

A constant rather than a literal at two raise sites, because the dry run reaches this
conclusion from the document alone and the real run reaches it from the blob store, and an
operator comparing a plan against the run it produced should not have to decide whether two
sentences describe the same document.
"""


@dataclass
class ReindexReport:
    """What one repair pass did, and what it could not do."""

    documents: int = 0
    chunks: int = 0
    unrepairable: list[str] = field(default_factory=list[str])
    """Documents naming the reason they could not be repaired, one line each."""

    failures: list[str] = field(default_factory=list[str])

    superseded: list[str] = field(default_factory=list[str])
    """Documents that moved on under the repair, one line each.

    Kept apart from ``failures`` because it is not one. The document is in a *better* state than
    the repair would have left it: something with newer bytes committed while this was reading
    older ones, and the guard at the commit is what stopped the older ones being written back.
    Counting it as a failure would put an expected outcome of ordinary concurrency into the list
    an operator is meant to investigate.
    """

    embedding: EmbeddingWork = field(default_factory=EmbeddingWork)
    """What the embedder was actually asked for across the pass.

    ``chunks`` above counts what was rebuilt; this counts what that cost. They are different
    numbers whenever a repair finds a vector it can keep, which after a narrow parser change
    is most of them.

    **A document that failed contributes what it spent.** A store-stage failure happens after
    the model has run, so the forward passes were paid for whether or not the vectors were
    written; a report that counted only the documents it managed to rebuild would say a sweep
    cost less precisely when things were going wrong.

    **A superseded document contributes nothing, and that is a fact about the guard rather than
    about this field.** The compare-and-swap fires on the first write a re-parse makes, which
    is before it is chunked or embedded, so a document a sync overtakes is dropped before it
    reaches the model. The window in which one could be superseded *after* embedding is the one
    ``_commit`` names as unreachable while the instance lock is held.
    """

    def note_unrepairable(self, document: Document, reason: str) -> None:
        self.unrepairable.append(f"{document.id} ({document.uri}): {reason}")

    def note_embedding(self, work: EmbeddingWork) -> None:
        """Fold one document's embed-stage accounting into the pass total."""
        self.embedding = _total(self.embedding, work)


def _total(left: EmbeddingWork, right: EmbeddingWork) -> EmbeddingWork:
    """Two embed-stage accountings added field by field.

    Written once rather than at each of the three places that aggregate, because a sum that
    forgets a field reports a smaller cost than was paid and nothing fails when it does.

    Reflective over the fields, which is what makes it maintenance-free and also what makes the
    refusal below necessary: ``+`` is defined on strings and on lists too, so a field that was
    not a count would be *concatenated* here rather than rejected, and the report would go
    quietly wrong instead of loudly.
    """
    totals: dict[str, int] = {}
    for name in EmbeddingWork.__dataclass_fields__:
        values = (getattr(left, name), getattr(right, name))
        if not all(isinstance(value, int) for value in values):
            msg = (
                f"EmbeddingWork.{name} is not a count, so two of them cannot be added. Every "
                f"field of an embed-stage accounting is summed across documents; a field that "
                f"is not a number needs its own rule rather than this one."
            )
            raise TypeError(msg)
        totals[name] = values[0] + values[1]
    return EmbeddingWork(**totals)


async def select(
    store: IngestStore,
    *,
    source: str | None = None,
    statuses: Collection[DocumentStatus] | None = None,
    media_types: Collection[str] | None = None,
    chunk_fingerprint: ChunkFingerprint | None = None,
    parse_fingerprints: Collection[ParseFingerprint] | None = None,
    glossary_fingerprint: GlossaryFingerprint | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[Document]:
    """The documents a repair verb should run over.

    ``chunk_fingerprint`` selects documents built by *something else* — the shape that makes a
    grammar upgrade a targeted repair rather than a corpus-wide rebuild.

    ``glossary_fingerprint`` does the same for the detector, and it is the selector without
    which a detector fix reaches nothing. Detection runs at ingest, downstream of parsing; a
    re-sync of unchanged bytes skips before it; and neither ``chunk_fp`` nor ``parse_fp`` moves
    when a detection rule changes. So a corpus can hold entries produced by rules corrected
    several times over and be *correct* about every fingerprint it records. Pass what the
    installed detector would produce now —
    :func:`~manicule.ingest.glossary_lineage.glossary_fingerprint` — and the selection is
    everything that disagrees with it, ``NULL`` included.

    ``parse_fingerprints`` does the same one stage earlier, and is what turns a library bump
    into a re-parse of the documents that library produced. Pass what every installed parser
    would produce now — :func:`~manicule.parsers.versions.current_parse_fingerprints` — and
    the selection is its complement: documents whose text came out of a version that is no
    longer installed, plus documents carrying no recorded lineage at all.

    **Both of these matter without waiting for a sync.** Change detection re-parses a stale
    document the next time its connector reports it, which is the right behavior and the
    wrong latency: between the upgrade and that sync, every anchor stored under the old
    version is being resolved by the new one. This is the selector that closes that window on
    demand, and it needs no network — re-parse reads retained bytes.

    ``offset`` is for the caller that repairs what it selects: see :func:`re_parse_stale`,
    which is the only one, and :meth:`~manicule.ingest.ports.IngestStore.select_documents` for
    why a page number would be the wrong cursor for a set that shrinks under it.
    """
    current = (
        {fingerprint.canonical() for fingerprint in parse_fingerprints}
        if parse_fingerprints is not None
        else None
    )
    return await store.select_documents(
        source=source,
        statuses=statuses,
        media_types=media_types,
        chunk_fp_other_than=chunk_fingerprint.canonical() if chunk_fingerprint else None,
        parse_fp_current=current,
        glossary_fp_other_than=(glossary_fingerprint.canonical() if glossary_fingerprint else None),
        limit=limit,
        offset=offset,
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

    **This verb deliberately does not reuse stored vectors**, and it is the only one that does
    not. :func:`~manicule.ingest.embedding.embed_or_reuse` would find every embedding input
    unchanged and skip every forward pass, which is precisely wrong here: this is what an
    operator runs when the vectors themselves are suspect — a reconfigured model, a restored
    directory, a half-finished rebuild — and a rebuild that reuses what it was asked to replace
    is not a rebuild. What it costs is stated up front by every refusal that recommends it.
    """
    report = ReindexReport()
    for document in documents:
        chunks = await store.document_chunks(document.id)
        if not chunks:
            if document.expects_chunks:
                report.note_unrepairable(document, "no stored chunks to re-embed")
            continue
        batches = 0

        def count_batch(batch: Sequence[Chunk], /) -> None:
            nonlocal batches
            del batch
            batches += 1

        try:
            produced = await embed_chunks(
                embedder,
                chunks,
                chunk_fingerprint=chunk_fingerprint,
                target_batch_tokens=target_batch_tokens,
                on_batch=count_batch,
            )
        except ContextOverflowError as exc:
            # Fatal to this document and to nothing else. There is no partial-credit answer:
            # an embedding of the first half of a passage is not a worse embedding of the
            # passage, it is an embedding of something else.
            report.failures.append(f"{document.id}: {exc}")
            continue
        # `vectors_new` and `vectors_replaced` are deliberately left at zero. This verb does
        # not read the rows it is about to overwrite, so it does not know which of them existed
        # — and a report that guessed "all replaced" would be asserting something it never
        # checked, in a module whose whole subject is not doing that.
        report.note_embedding(
            EmbeddingWork(
                chunks=len(chunks),
                embedded=len(chunks),
                input_changed=len(chunks),
                forward_calls=batches,
            )
        )
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

    **Only the chunks with no usable vector reach the model.** This is the verb for a run that
    stopped part-way, so the interesting case is a document most of whose vectors were written
    before the crash: embedding all of them again to finish the few that were not is the whole
    cost of the repair and none of its value. The chunks are the stored ones, so their
    embedding inputs are exactly what the index recorded — which is what makes the vectors that
    survived the crash reusable and the rest a repair.
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
            produced, work = await embed_or_reuse(
                embedder,
                chunks,
                vectors=vectors,
                chunk_fingerprint=chunk_fingerprint,
                # The stored chunks *are* the previous ones — this verb re-embeds what is in
                # the index rather than re-deriving it — so a chunk with no vector row is
                # always a missing vector rather than a new chunk.
                previous={chunk.id: chunk.embed_text for chunk in chunks},
                target_batch_tokens=target_batch_tokens,
            )
        except ContextOverflowError as exc:
            report.failures.append(f"{document.id}: {exc}")
            continue
        report.note_embedding(work)
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


async def reindex_document(
    document_id: str,
    *,
    store: IngestStore,
    pipeline: IngestPipeline,
    blobs: BlobSink,
) -> ReindexReport:
    """Re-parse one document from its retained bytes. Rung 3, for a single id.

    The narrow end of :func:`re_parse`, and the verb that finishes a restore. A document
    restored after the soft-delete grace period comes back holding no chunks — the sweep took
    them, which is the trade ``docs/storage.md`` §8.2 makes for not diluting every vector
    search — and :meth:`~manicule.core.protocols.TrashStore.restore_document` says so by
    returning ``needs_reparse``. This is what to run next, and it touches neither the network
    nor the source.

    It resolves the id through the store rather than taking a
    :class:`~manicule.core.content.Document`, and that is the whole reason it exists as a verb
    of its own: the lookup is workspace-scoped and skips the trash, so an id from another
    tenant, a mistyped one, and one still in the trash all come back named in the report rather
    than as a document nobody can account for. **Restore first, then reindex** — the other
    order finds nothing, because a soft-deleted document is not a document this lookup returns.
    """
    document = await store.get_document(document_id)
    if document is None:
        report = ReindexReport()
        report.unrepairable.append(
            f"{document_id}: no live document with that id in this workspace. A document in "
            f"the trash has to be restored before it can be re-parsed."
        )
        return report
    return await re_parse([document], pipeline=pipeline, blobs=blobs)


async def re_parse(
    documents: Sequence[Document],
    *,
    pipeline: IngestPipeline,
    blobs: BlobSink,
) -> ReindexReport:
    """Run the current parser chain over retained bytes. Rung 3: no network.

    Identity rules are the same as first ingest, because it is the same code path: chunks are
    reconciled against the stored set by ``chunks.id``, and an id is derived from its content,
    so a chunk that survives a re-parse unchanged keeps its id and therefore its vector.

    **What is embedded is a separate question from what keeps its id**, and the pipeline
    answers it separately: a chunk reaches the model only when its *embedding input* is new,
    changed, or has no readable vector stored against it
    (:func:`~manicule.ingest.embedding.embed_or_reuse`). So a parser fix that changes one table
    in a hundred-page document embeds that table and the chunks whose heading breadcrumbs moved
    with it — which is a larger set than the ids that changed and a far smaller one than the
    document.

    A document whose bytes were never retained — the cap refused them, or it predates
    retention — is named with the reason rather than failing the run. Those are the only
    documents for which a re-crawl is the only repair.

    **Every document here is a snapshot, and the pipeline is told so.** ``documents`` was read
    at some earlier moment — a page of a corpus-wide selection, a single row looked up by id —
    and between that read and the commit of the text this produces, a connector sync can fetch
    newer bytes for the same page and commit them. The re-parse would then write the *older*
    content back over them, successfully and silently. Passing each document's
    :attr:`~manicule.core.content.Document.revision` as the expected one makes that commit
    refuse instead; the document is reported ``superseded`` and nothing derived from the stale
    snapshot is written at all.
    """
    report = ReindexReport()
    for document in documents:
        if document.original_ref is None:
            report.note_unrepairable(document, NO_RETAINED_BYTES)
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
            expected=document.revision,
        )
        for outcome in outcomes:
            # Counted before the branching, and that is the point: the embed stage runs before
            # anything below decides which list this document lands in, so what it cost is the
            # same either way. The case that matters is a **store-stage failure** — the model
            # has already run by the time an upsert fails — and counting only the documents a
            # sweep managed to rebuild would report it as costing less the worse it went.
            #
            # Two things contribute zero here, for different reasons. A document superseded by a
            # concurrent sync never reached the model: the guard fires on the re-parse's first
            # write, which is before chunking. And a document that failed *inside* the embed
            # stage left no accounting behind — an unknown number of batches had already gone —
            # so zero is the honest floor rather than a guess at the figure.
            report.note_embedding(outcome.embedding)
            if outcome.superseded:
                # Ahead of the two failure shapes below, because a superseded document is
                # neither of them and would be read as one: it comes back carrying the status
                # the *winner* left, which for a completed sync is `indexed`.
                report.superseded.append(
                    f"{outcome.document_id or outcome.source_id}: a newer revision was committed "
                    f"while this was being re-parsed, so nothing from the older one was written"
                )
                continue
            # Two shapes of failure, because the pipeline has two. A document that was not
            # already working comes back ``failed``. A document that *was* ``indexed`` keeps its
            # status, its chunks and its vectors — the pipeline refuses to let a transient error
            # cost a working document — and the failure arrives as an ``indexed`` outcome
            # carrying a detail. Reading only the status counts that second case as a repair,
            # which is the worst available answer: the stored text is still the previous
            # parser's, the lineage was deliberately not advanced, and the report says the
            # document was rebuilt.
            if outcome.status is DocumentStatus.FAILED or (
                outcome.status is DocumentStatus.INDEXED and outcome.detail
            ):
                report.failures.append(
                    f"{outcome.document_id or outcome.source_id}: {outcome.detail}"
                )
            else:
                report.documents += 1
                report.chunks += outcome.chunks
    return report


DEFAULT_SWEEP_BATCH = 25
"""How many documents one page of the sweep holds.

Small, because the page is not where the work is. A document costs a parse, a chunking and an
embedding pass, and those are serialized anyway — so a larger page buys nothing but a longer
window in which an interrupted run has read rows it will not reach.
"""


@dataclass
class StaleSweep:
    """What one corpus-wide re-parse pass did, in counts an operator can act on.

    Every field is a count of *documents* except the two named for chunks, and the distinction
    is the whole reason both are here: a bump makes every document its parser produced stale,
    and only some of them come out different. ``reparsed`` is what was rebuilt; ``changed`` is
    what the rebuild actually moved.
    """

    dry_run: bool = False
    selected: int = 0
    """Documents whose recorded parse lineage is not one an installed parser would produce."""

    reparsed: int = 0
    """Documents rebuilt from retained bytes. Zero on a dry run, by construction."""

    unchanged: int = 0
    """Of ``reparsed``, those that came out with exactly the chunk ids they went in with."""

    changed: int = 0
    """Of ``reparsed``, those whose chunk set moved."""

    chunks_new: int = 0
    """Chunks the sweep produced that were not already stored, across every document.

    **Not a count of forward passes**, and the difference is worth stating because the obvious
    reading is wrong. A chunk's id is derived from its ``text``; what reaches the model is
    ``embed_text``, which carries the heading breadcrumb. An id that survived therefore does
    *not* prove the embedded string did — a document whose headings moved re-embeds chunks
    whose ids never changed. What this counts is how much of the corpus the re-parse produced
    anew; :attr:`embedding` is what it cost.

    **Nor is it the embedding cache's doing**, which this docstring claimed until somebody
    measured it. The cache de-duplicates identical ``embed_text`` within one run, which is not
    the shape of a corpus-wide sweep over distinct chunks; what avoids the forward passes is the
    persisted embedding-input identity, and ``docs/parsing.md`` §4.5 carries both sets of
    numbers.
    """

    chunks_kept: int = 0
    """Chunks that survived with their id, and therefore with the vector row already stored.

    **A statement about content identity, and about the row rather than the vector in it.** A
    kept chunk keeps the row every citation to it resolves through; the row's contents may
    still be written again, because what is embedded is ``embed_text`` and the breadcrumb in it
    can move under an id that did not. So this is not a count of vectors kept and not a count
    of embedding work avoided. What was avoided is :attr:`embedding`, measured at the model
    rather than inferred from row identity.
    """

    embedding: EmbeddingWork = field(default_factory=EmbeddingWork)
    """What the sweep cost at the embedder, summed over every document it rebuilt.

    The three-way partition — reused, re-embedded because the input changed, re-embedded
    because the vector was missing or corrupt — plus the number of batches the model was
    actually asked for. This is the field to read when pricing a parser bump; everything above
    it counts documents and chunks, which are not what an accelerator spends its time on.
    """

    unrepairable: int = 0
    failed: int = 0
    unrepairable_documents: list[str] = field(default_factory=list[str])
    """One line per document that cannot be repaired, naming the reason and the remedy."""

    failures: list[str] = field(default_factory=list[str])

    superseded: int = 0
    """Documents a newer revision overtook mid-repair, so the repair declined to commit.

    **Not a failure and not a repair**, which is why it is neither of the two counts either
    side of it. A connector sync committed newer bytes for the document while this sweep was
    parsing older ones, and the guard at the commit refused to write the older result back over
    them — so the corpus holds the *newer* text and this sweep did not touch it. Nothing needs
    doing about one; ``docs/ingest.md`` §8.5 says so at more length, and says what the two
    numbers mean together.
    """

    superseded_documents: list[str] = field(default_factory=list[str])
    """One line per superseded document: which it is and what overtook it.

    Named individually rather than only counted, on the same rule as the two lists above: a
    count nobody can attach to a document is a number nobody can check. An id and a sentence,
    never any of the retained text — this whole command's subject is content, and its report
    goes to a terminal and to whatever a shell pipeline points at.
    """


async def plan_stale(
    *,
    store: IngestStore,
    parse_fingerprints: Collection[ParseFingerprint],
    batch: int = DEFAULT_SWEEP_BATCH,
) -> StaleSweep:
    """What :func:`re_parse_stale` would do. The selection, and nothing else.

    **A function rather than a flag on the sweep, and the signature is the reason.** A dry run
    parses nothing, so it needs no parser chain, no chunker, no embedder, no vector store and
    no blob store — and taking them as arguments it would not use is how a plan ends up
    building a worker pool and loading a tokenizer to answer a question about rows. Worse, the
    checks that guard a real run can *refuse*: an index whose recorded fingerprints disagree
    with the configured components must not be written to, and must still be surveyable.

    The one thing this cannot report is a retained reference whose bytes have gone missing from
    the blob store, because finding that out is a blob read. Documents with no reference at all
    are named here; the rest is left to the run that is allowed to do work.

    Args:
        store: Where the selection is queried.
        parse_fingerprints: What every installed parser would produce now.
        batch: Documents per page.

    Returns:
        A sweep marked ``dry_run``, carrying ``selected`` and the unrepairable documents.
    """
    sweep = StaleSweep(dry_run=True)
    while True:
        # The offset is the count, exactly: a plan repairs nothing, so every document it has
        # seen is still in the selection and still in front of the next page.
        page = await select(
            store, parse_fingerprints=parse_fingerprints, limit=batch, offset=sweep.selected
        )
        if not page:
            return sweep
        for document in page:
            sweep.selected += 1
            if document.original_ref is None:
                sweep.unrepairable += 1
                sweep.unrepairable_documents.append(
                    f"{document.id} ({document.uri}): {NO_RETAINED_BYTES}"
                )


async def re_parse_stale(
    *,
    store: IngestStore,
    pipeline: IngestPipeline,
    blobs: BlobSink,
    parse_fingerprints: Collection[ParseFingerprint],
    batch: int = DEFAULT_SWEEP_BATCH,
) -> StaleSweep:
    """Re-parse every document a parser bump left behind. Rung 3, over the whole corpus.

    :func:`select` and :func:`re_parse` compose into exactly this and had no caller: the only
    shipped surface was the per-document verb, so closing the window an upgrade opens meant
    either writing Python against these functions or waiting for every document to come round
    on a connector sync. This is that sweep, and it touches no connector and no network —
    every byte it parses comes out of the blob store.

    **Paged, and the cursor is the count of documents left behind rather than a page number.**
    A repaired document leaves the selection, so the set shrinks under the iteration; an
    unrepairable one stays in it forever. Counting pages would skip the documents that shifted
    forward, and starting from zero every time would re-read the same unrepairable prefix until
    the end of the world. Advancing by exactly the documents this pass did *not* remove from
    the set is right in both directions, and it is what makes the sweep terminate on a corpus
    where nothing can be repaired at all.

    **A document is the transaction boundary.** Each one is committed by the pipeline before
    the next is read, so an interrupted run — ``Ctrl-C``, a killed process, a canceled task —
    leaves every document it finished internally consistent and every document it did not
    still selected. Resuming is running the command again; there is no state to clean up and
    no resume token to keep.

    **Idempotent for every document manicule can version.** A repaired document records the
    fingerprint an installed parser produces, so the next run does not select it and performs
    no embedding work at all. The exception is a document with *no* recorded lineage — a
    plugin parser produced it, and manicule has no version to compare — which
    :func:`~manicule.parsers.versions.parse_fingerprint` deliberately leaves eligible so that
    it is reachable at all. Those are re-parsed once per sweep and remain selectable.

    Args:
        store: Where the selection is queried and the chunk sets are compared.
        pipeline: The current chain. The **same** object a sync would use, so the sweep
            inherits its embedding lock rather than introducing a second consumer of the model.
        blobs: The retained bytes. The only source of content this reads.
        parse_fingerprints: What every installed parser would produce now, from
            :func:`~manicule.parsers.versions.current_parse_fingerprints`. Passing a partial
            set is a repair that cannot end, so it is the caller's job to get it whole.
        batch: Documents per page.

    Returns:
        The counts, and one line per document that could not be repaired.
    """
    sweep = StaleSweep()
    current = {fingerprint.canonical() for fingerprint in parse_fingerprints}
    left_behind = 0
    while True:
        page = await select(
            store, parse_fingerprints=parse_fingerprints, limit=batch, offset=left_behind
        )
        if not page:
            return sweep
        for document in page:
            sweep.selected += 1
            # Unconditional, and it is the loop's termination argument rather than a detail: a
            # document this pass repairs leaves the selection and a document it does not stays
            # in it, so the cursor has to move past everything in the second class. Counting
            # from the start of the page and correcting downwards for successes would be the
            # same arithmetic with one more way to get it wrong.
            left_behind += 1
            before = {chunk.id for chunk in await store.document_chunks(document.id)}
            report = await re_parse([document], pipeline=pipeline, blobs=blobs)
            # Folded before the branching, for the reason `re_parse` folds it before its own:
            # the embed stage runs before any of the outcomes below is decided. A document with
            # no retained bytes contributes zero because nothing embedded it, and a superseded
            # one contributes zero because the commit guard drops it before it is chunked — but
            # a document that failed at the store contributes the passes it had already spent.
            sweep.embedding = _total(sweep.embedding, report.embedding)
            if report.unrepairable:
                sweep.unrepairable += 1
                sweep.unrepairable_documents.extend(report.unrepairable)
                continue
            if report.failures:
                sweep.failed += 1
                sweep.failures.extend(report.failures)
                continue
            if report.superseded:
                sweep.superseded += 1
                sweep.superseded_documents.extend(report.superseded)
                # The cursor follows the *selection*, not the outcome, so a superseded document
                # gets the same read-back a repaired one gets. A sync that overtook this
                # document usually left it carrying a current fingerprint, so it is gone from
                # the selection and counting it as left behind would step the offset past a
                # document nobody has looked at.
                if await _left_the_selection(store, document.id, current):
                    left_behind -= 1
                continue
            sweep.reparsed += 1
            after = {chunk.id for chunk in await store.document_chunks(document.id)}
            sweep.chunks_new += len(after - before)
            sweep.chunks_kept += len(after & before)
            if after == before:
                sweep.unchanged += 1
            else:
                sweep.changed += 1
            if await _left_the_selection(store, document.id, current):
                left_behind -= 1


async def _left_the_selection(
    store: IngestStore, document_id: str, current: Collection[str]
) -> bool:
    """Whether a document the sweep has finished with is out of the selection now.

    Read back rather than assumed. A successful re-parse usually records a current fingerprint
    and leaves the selection — but not always, and the exception is the one that would hang the
    loop: a parser manicule does not ship records nothing, so the document is repaired, still
    selected, and has to keep its place in the cursor.
    """
    rebuilt = await store.get_document(document_id)
    return rebuilt is not None and rebuilt.parse_fp in current


DETECTION_IS_OFF = (
    "glossary detection is switched off (rag.glossary.detect_on_ingest = false), so there is "
    "no detector to bring documents up to date with. Recomputing would run rules the "
    "configuration says not to run; recording the disabled state instead would erase the "
    "record of which detector produced the entries that are still being served. Turn detection "
    "on and run this again."
)
"""Why glossary repair refuses rather than proceeding when detection is disabled.

The plan refuses too, and that is not an oversight. Under a disabled detector the installed
fingerprint *is* the disabled one, so the selection would be "every document whose entries a
detector did produce" — which on a working corpus is all of them, reported as work to do that
this command would not do. A number that large and that wrong is worse than a refusal that
names the setting.
"""


@dataclass
class GlossarySweep:
    """What one glossary-only repair pass did, in counts an operator can act on.

    Separate from :class:`StaleSweep` rather than more fields on it, because the two answer
    different questions and share no interesting number. A re-parse reports chunks kept and
    chunks new, because its cost is embedding and those are what it will be charged for. This
    reports entries, because its cost is a regular expression over text already in memory and
    the thing an operator wants to know is what the corrected rules did to the vocabulary.
    """

    dry_run: bool = False
    selected: int = 0
    """Documents whose recorded glossary lineage is not what the installed detector produces."""

    redetected: int = 0
    """Documents whose entries were recomputed. Zero on a dry run, by construction."""

    unchanged: int = 0
    """Of ``redetected``, those whose entry set came out exactly as it went in.

    **The expected majority, and the number that says a detector fix was narrow.** Lineage
    records the detector's identity rather than a document's content, so every document is stale
    after any change to it — and nothing can tell which ones come out different without running
    the rules. A document counted here still advances its fingerprint: the point of the sweep is
    that the corpus can say which detector read it, and "the answer did not move" is an answer.
    """

    changed: int = 0
    """Of ``redetected``, those whose entry set moved.

    Compared as a set of entries rather than as a count of them, and that is not fussiness: the
    change this feature was built for — #110's heading gate against #108's list markers —
    removes false entries and adds real ones on the same page, and a count would report a
    document whose whole vocabulary was replaced as untouched.
    """

    entries_before: int = 0
    entries_after: int = 0
    """Entries across every document redetected, going in and coming out.

    Two totals rather than a net, for the same reason: a pass that removes eleven false headings
    and adds eleven list definitions is not a pass that did nothing.
    """

    failed: int = 0
    failures: list[str] = field(default_factory=list[str])
    """One line per document the detector could not read, naming it and the error.

    A failure leaves that document's rows and its stale lineage exactly as they were, so it is
    still selected next time and still reported by ``doctor``. The sweep continues: one
    document's text tripping a rule is not a reason to leave the rest of a corpus on
    superseded rules.
    """

    unrepairable: int = 0
    unrepairable_documents: list[str] = field(default_factory=list[str])
    """Documents whose glossary cannot be computed because their chunks are gone.

    **A count of its own rather than more failures**, because the two send an operator to
    different places. A failure is a bug in this repository and clears when it is fixed; this is
    a document whose *inputs* are missing, and the remedy is a rung further up — a re-parse from
    retained bytes. Merging them would hide a corpus-sized "run ``--stale`` first" inside a
    number that reads as "the detector is broken".
    """

    superseded: int = 0
    superseded_documents: list[str] = field(default_factory=list[str])
    """Documents a newer sync overtook mid-recompute, so the write was declined.

    **Neither a failure nor a repair**, which is why it is neither of the counts above it, and
    it is the same reading :attr:`StaleSweep.superseded` applies one rung up. A connector sync
    committed newer chunks while this was reading older ones; the corpus holds the newer state
    and this sweep did not touch it. Nothing needs doing about one — the document is simply
    selected again next time, against chunks that are now current.
    """


def _entry_shape(
    entry: GlossaryEntry,
) -> tuple[str, str, str, str, str, str, str, tuple[str, ...]]:
    """One entry reduced to every stored field of it that can vary between two readings.

    **Every field, rather than the ones that decide a lookup**, and the difference was a real
    defect: an earlier version of this omitted ``display`` and ``location`` on the reasoning that
    nothing resolves through them. Both are *stored* and both are *shown* — ``display`` is the
    source's own spelling of the term, which a citation quotes instead of the normalized key, and
    ``location`` is where in the document it was found. A detector change that moved either would
    have rewritten what a reader is served while this reported the document unchanged, which is a
    quieter version of exactly the defect the fingerprint exists to remove.

    **Three stored columns are still absent, and the reason is that none of them can move**, which
    is a narrower claim than "every field" and the one this actually supports. ``document_id`` is
    the document being compared against itself. ``id`` is
    :func:`~manicule.core.ids.glossary_entry_id` of the chunk, the acronym and the expansion —
    three fields already here, so it cannot vary while they do not. ``created_at`` is stamped at
    write time and therefore differs on every comparison, so including it would report every
    document changed. A stored field that is none of those three belongs in this tuple.

    The three adjustments are all about comparing like with like. Aliases are sorted, because the
    store returns them sorted and the detector returns them in the order the source wrote them,
    and a comparison that disagreed about that would report a change nobody made. They stay
    nested rather than flattened, on the reasoning
    :func:`~manicule.ingest.middleware.text_digest` gives about NUL separation: flattened, a term
    whose location is ``A`` and whose alias is ``B`` would compare equal to one with no location
    and aliases ``A``, ``B``. And confidence is formatted to a string, because it round-trips
    through SQLite as a float and pinning the comparison to six places is cheaper to reason about
    than trusting two paths to produce bit-identical doubles.
    """
    return (
        entry.acronym,
        entry.display,
        entry.expansion,
        entry.chunk_id,
        entry.location,
        entry.form.value,
        f"{entry.confidence:.6f}",
        tuple(sorted(entry.aliases)),
    )


NO_STORED_CHUNKS = (
    "no stored chunks, and this document's status says it should have some. Its glossary "
    "cannot be recomputed from nothing: repair the chunks first with `document reindex "
    "--stale` or `document reindex <id>`, which reads the retained bytes, and run this again"
)
"""Why a document's glossary cannot be recomputed, worded once.

**The alternative is what this replaces, and it was worse than doing nothing.** Detecting over
an empty chunk list returns no entries, which is a perfectly well-formed derived result — so
the document was stamped with a current fingerprint on the strength of having read nothing, and
left the selection permanently. A missing-chunks problem thereby became an invisible
empty-glossary one, and the only signal that anything was wrong was gone.

Chunkless *by design* is a different state and is not this: a document that yielded no
extractable text really does state no definitions, and recording that is correct.
:attr:`~manicule.core.content.Document.expects_chunks` is the discriminator, and
:func:`re_embed` already uses it for exactly this distinction one rung up.
"""


@dataclass(frozen=True, slots=True)
class GlossaryOutcome:
    """What became of one document's glossary, in the four shapes this can take.

    A type rather than the ``tuple | str`` this used to be, because the union had two shapes and
    the sweep now has to tell four things apart — and "a string means something went wrong" is
    the kind of encoding that quietly acquires a fifth meaning.
    """

    entries_before: int = 0
    entries_after: int = 0
    moved: bool = False
    """Whether the entry set differs. Meaningless unless :attr:`repaired`."""

    failure: str = ""
    """Why the detector could not read this document. Nothing was written."""

    unrepairable: str = ""
    """Why this document's glossary cannot be computed at all, and what would fix it.

    Distinct from :attr:`failure` because the remedy is: a failure is a bug here and will clear
    when it is fixed, and this is a document whose *inputs* are missing and needs a repair at a
    higher rung. Reporting them as one number would put a corpus's worth of "run --stale" into
    the list an operator reads as "the detector is broken".
    """

    superseded: str = ""
    """What overtook this document while its glossary was being recomputed.

    **Neither a failure nor a repair**, on exactly the reading #119 established for the parse
    sweep: a sync committed newer chunks while this was reading older ones, the write was
    refused, and the corpus holds the *newer* state. Nothing needs doing about one. Counting it
    as a failure — which this did until the specification asked for the count separately — puts
    an expected outcome of ordinary concurrency in front of an operator as a defect.
    """

    @property
    def repaired(self) -> bool:
        """Whether entries were rewritten and the lineage advanced."""
        return not (self.failure or self.unrepairable or self.superseded)


async def redetect_glossary(
    document: Document,
    *,
    store: IngestStore,
    glossary: GlossaryStore,
    fingerprint: GlossaryFingerprint,
) -> GlossaryOutcome:
    """Recompute one document's entries from its stored chunks. Rung 0.

    **The cost boundary is the signature.** A store to read chunks through, a place to put
    entries, and the identity of what produced them. There is no ``pipeline``, no ``blobs``, no
    ``embedder`` and no ``vectors`` argument, so this function could not fetch a source, open a
    retained blob, run a parser or produce a vector if it wanted to — which is a stronger
    statement than a comment promising it does not, and
    ``tests/glossary/test_repair.py::test_the_repair_reaches_no_parser_no_blob_and_no_embedder``
    asserts it of this signature rather than of one run, because a run only shows what that run
    did.

    **A document whose chunks are missing is refused rather than stamped**, and the distinction
    it turns on is not "has no chunks". Detecting over an empty list returns no entries, which is
    a well-formed derived result — so an ``indexed`` document whose chunks have gone (a restore
    after the soft-delete sweep took them, ``storage.md`` §8.2) used to be recorded as current on
    the strength of having read nothing, and left the selection for ever. That converts a
    missing-chunks problem into an invisible empty-glossary one.
    :attr:`~manicule.core.content.Document.expects_chunks` separates the two, and
    :func:`re_embed` already uses it for the same distinction one rung up: a document that is
    chunkless *by design* really does state no definitions, and recording that is right.

    **The write is inside the failure handling as well as the detection**, and that is not
    defensive breadth. There is a real race: an entry's ``chunk_id`` is a foreign key, this reads
    the chunks and then writes rows citing them, and a sync re-ingesting the same document in
    between replaces exactly those chunks — so the insert fails with ``FOREIGN KEY constraint
    failed``. Unlike the parse sweep, this one takes no lock and shares none, because never
    reaching the model is the whole point of it.

    **That race is a supersession, not a failure, and it is told apart positively rather than
    from the exception's type.** Reading the error would mean matching a message or importing
    SQLAlchemy into ``manicule.ingest``, which the import boundary forbids and which would tie
    this to one store. So the chunks are read again: if the ids this was about to cite are no
    longer the document's, a sync committed newer ones underneath and the corpus now holds the
    *newer* state — nothing needs doing. If they are unchanged, the write failed for some other
    reason and that is a failure. One extra query, and only on the path that already went wrong.

    Returns:
        A :class:`GlossaryOutcome`. In every shape but the repaired one **nothing was written**,
        so the previous entries are still servable and the stale lineage is still stale — which
        is what keeps the document selected and retryable.
    """
    # Imported here rather than at module scope, which keeps `manicule.ingest.reindex` free of
    # the detector for every caller that only re-parses or re-embeds — this module is imported
    # by the app runtime to answer a plan, and a plan reads rows.
    from manicule.ingest.glossary import detect_entries  # noqa: PLC0415

    named = f"{document.id} ({document.uri})"
    # The refusal is decided before the previous entries are read, so a corpus of documents this
    # cannot repair costs one query each rather than two. It also reads in the order the
    # reasoning runs: whether there is anything to detect over, and only then what is there now.
    chunks = await store.document_chunks(document.id)
    if not chunks and document.expects_chunks:
        return GlossaryOutcome(unrepairable=f"{named}: {NO_STORED_CHUNKS}")
    before = await glossary.glossary_entries(document.id)
    try:
        entries = detect_entries(chunks, title=document.title, media_type=document.media_type)
    except Exception as exc:  # noqa: BLE001 - a detector bug costs this document and no other
        return GlossaryOutcome(failure=f"{named}: {type(exc).__name__}: {exc}")
    try:
        await glossary.replace_glossary_entries(
            document.id, entries, fingerprint=fingerprint.canonical()
        )
    except Exception as exc:  # noqa: BLE001 - one document's repair, never the sweep's
        if {chunk.id for chunk in await store.document_chunks(document.id)} != {
            chunk.id for chunk in chunks
        }:
            return GlossaryOutcome(
                superseded=(
                    f"{named}: a newer revision was committed while its glossary was being "
                    f"recomputed, so nothing from the older one was written"
                )
            )
        return GlossaryOutcome(failure=f"{named}: {type(exc).__name__}: {exc}")
    return GlossaryOutcome(
        entries_before=len(before),
        entries_after=len(entries),
        moved={_entry_shape(entry) for entry in before}
        != {_entry_shape(entry) for entry in entries},
    )


async def plan_stale_glossary(
    *,
    store: IngestStore,
    fingerprint: GlossaryFingerprint,
    batch: int = DEFAULT_SWEEP_BATCH,
) -> GlossarySweep:
    """What :func:`redetect_stale_glossary` would do. The selection, and nothing else.

    A function rather than a flag, for the reason :func:`plan_stale` gives: a plan reads rows,
    and taking a writer it would not use is how a survey acquires the ability to write.

    Args:
        store: Where the selection is queried.
        fingerprint: What the installed detector would produce now.
        batch: Documents per page.

    Raises:
        PolicyError: Detection is switched off. See :data:`DETECTION_IS_OFF`.
    """
    _require_detection(fingerprint)
    sweep = GlossarySweep(dry_run=True)
    while True:
        # The offset is the count, exactly: a plan writes nothing, so every document it has seen
        # is still in the selection and still in front of the next page.
        page = await select(
            store, glossary_fingerprint=fingerprint, limit=batch, offset=sweep.selected
        )
        if not page:
            return sweep
        sweep.selected += len(page)


async def redetect_stale_glossary(
    *,
    store: IngestStore,
    glossary: GlossaryStore,
    fingerprint: GlossaryFingerprint,
    batch: int = DEFAULT_SWEEP_BATCH,
) -> GlossarySweep:
    """Bring every document's glossary up to the installed detector. Rung 0, corpus-wide.

    **Paged, with the cursor counting what the pass left behind**, exactly as
    :func:`re_parse_stale` does and for the same reason: a repaired document leaves the
    selection, so the set shrinks under the iteration, and a page number would skip whatever
    shifted forward into the slots it vacated. Here the arithmetic is simpler because this
    function knows locally whether each document was repaired — there is no parser that might
    decline to record a fingerprint — so the cursor advances only for failures.

    **A document is the transaction boundary**, and cancellation is safe at one. Each document's
    rows and lineage are written together and committed before the next is read, so an
    interrupted run leaves every document it finished internally consistent and every document
    it did not still selected. Resuming is running the command again.

    **A second run selects nothing.** A repaired document records the installed fingerprint, so
    the predicate that found it no longer does — including for a document that produced no
    entries at all, which is the case a design recording lineage only against rows would loop on
    for ever.

    Args:
        store: Where the selection is queried and chunks are read.
        glossary: Where entries and their lineage are written.
        fingerprint: What the installed detector would produce now.
        batch: Documents per page.

    Raises:
        PolicyError: Detection is switched off. See :data:`DETECTION_IS_OFF`.
    """
    _require_detection(fingerprint)
    sweep = GlossarySweep()
    left_behind = 0
    while True:
        page = await select(
            store, glossary_fingerprint=fingerprint, limit=batch, offset=left_behind
        )
        if not page:
            return sweep
        for document in page:
            sweep.selected += 1
            outcome = await redetect_glossary(
                document, store=store, glossary=glossary, fingerprint=fingerprint
            )
            if not outcome.repaired:
                # Nothing was written in any of the three unrepaired shapes, so this document is
                # still in the selection and has to keep its place in the cursor. Advancing past
                # everything this pass did not remove from the set is what makes the loop
                # terminate on a corpus where nothing can be repaired at all — and it is right
                # for a supersession too, which will be selected again on the next run against
                # the chunks that overtook it.
                left_behind += 1
                if outcome.unrepairable:
                    sweep.unrepairable += 1
                    sweep.unrepairable_documents.append(outcome.unrepairable)
                elif outcome.superseded:
                    sweep.superseded += 1
                    sweep.superseded_documents.append(outcome.superseded)
                else:
                    sweep.failed += 1
                    sweep.failures.append(outcome.failure)
                continue
            sweep.redetected += 1
            sweep.entries_before += outcome.entries_before
            sweep.entries_after += outcome.entries_after
            if outcome.moved:
                sweep.changed += 1
            else:
                sweep.unchanged += 1


def _require_detection(fingerprint: GlossaryFingerprint) -> None:
    if fingerprint.detects:
        return
    raise PolicyError(DETECTION_IS_OFF)


__all__ = [
    "DEFAULT_SWEEP_BATCH",
    "DETECTION_IS_OFF",
    "NO_RETAINED_BYTES",
    "NO_STORED_CHUNKS",
    "GlossaryOutcome",
    "GlossarySweep",
    "ReindexReport",
    "StaleSweep",
    "plan_stale",
    "plan_stale_glossary",
    "re_embed",
    "re_parse",
    "re_parse_stale",
    "redetect_glossary",
    "redetect_stale_glossary",
    "reindex_document",
    "repair",
    "select",
]
