"""What the pipeline needs from storage, stated as protocols rather than as an import.

:class:`~manicule.core.protocols.DocStore` is the surface retrieval needs. Ingest needs
more — lineage, tombstones, run counters, the recovery sweep — and every one of those is a
write only the pipeline performs. Declaring them here keeps two properties that matter:

**Ingest imports no database.** ``manicule.ingest`` pulls in core, config and the plugin
machinery, and nothing else. The SQLite store satisfies these structurally, and
``tests/test_import_boundary.py`` fails the build if that stops being true.

**The negative tests can lie deliberately.** Every guard in this package is checked against
a store that breaks its part of the bargain, which is only affordable when the store is a
protocol rather than a migrated database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manicule.core.acquisition import UNSET, UnsetValue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Sequence
    from datetime import datetime

    from manicule.core.acquisition import (
        AcquiredSource,
        AcquisitionDiagnostic,
        AcquisitionRecord,
        AcquisitionRecordState,
        AcquisitionRun,
        AcquisitionRunState,
        AcquisitionSource,
    )
    from manicule.core.content import (
        Chunk,
        Commit,
        Document,
        DocumentRevision,
        DocumentStatus,
        Metadata,
    )
    from manicule.core.embedding import IndexFingerprints
    from manicule.core.glossary import GlossaryEntry
    from manicule.core.retrieval import Filter
    from manicule.core.sources import SourceId, Watermark


@runtime_checkable
class IngestStore(Protocol):
    """The relational operations one ingest run performs.

    Wider than :class:`~manicule.core.protocols.DocStore` and deliberately not a subclass of
    it: a store may serve retrieval without ever being written to by a pipeline, and a
    protocol that demanded both would make the read-only case impossible to express.
    """

    # --- documents ---------------------------------------------------------------------

    async def get_document(self, document_id: str) -> Document | None: ...

    async def find_document(self, source: str, source_id: SourceId) -> Document | None: ...

    async def upsert_document(self, document: Document) -> Document: ...

    async def commit_document(self, document: Document, *, expected: DocumentRevision) -> Commit:
        """Write a document, but only if the stored one is still ``expected``.

        :meth:`upsert_document` with a compare-and-swap, and the two halves have to be one
        operation. An operation that read the document, checked it, and then wrote would have a
        window between the check and the write that is the same width as the window it exists to
        close — so an implementation performs the comparison **inside the transaction that
        writes**, and takes the write lock before it compares rather than after.

        Nothing else about the write differs: the same row is replaced, the same supersession is
        recorded, and a caller that does not need the guard should keep using
        :meth:`upsert_document` rather than passing an expectation it invented.

        **A miss is not a failure.** It means another writer got there first with something
        newer, which is the outcome this exists to produce rather than an error to raise: the
        caller has a stale snapshot and its job is to stop, not to retry. ``docs/ingest.md``
        §8.5 states the invariant and what an operator does about a miss.

        Args:
            document: What to store if the guard holds.
            expected: The revision the caller derived its work from.

        Returns:
            A :class:`~manicule.core.content.Commit` saying whether the write happened, and
            carrying the row as it stands either way.
        """
        ...

    async def stage_vectors(self, publication_id: str, chunks: Sequence[Chunk]) -> None:
        """Record physical vector ids before they are written, for crash cleanup."""
        ...

    async def publish_failure(
        self,
        document: Document,
        *,
        expected: DocumentRevision | None,
        original_omitted_reason: str | None,
    ) -> Commit:
        """Atomically publish a failed row and its source-retention state."""
        ...

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
        """Atomically publish one document and all relational derived state."""
        ...

    async def set_status(
        self, document_id: str, status: DocumentStatus, detail: str = ""
    ) -> None: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def soft_delete_document(self, document_id: str) -> None: ...

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol it widens
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]: ...

    # --- chunks ------------------------------------------------------------------------

    async def replace_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> None: ...

    async def get_chunks(self, chunk_ids: Sequence[str]) -> Sequence[Chunk]: ...

    async def count_chunks(self, document_id: str | None = None) -> int: ...

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]:
        """Every chunk of one document, in position order.

        The read that ``reindex --re-embed`` and ``reindex --repair`` run over, so it returns
        whole chunks rather than ids: both need ``embed_text`` and ``token_count``, and
        fetching ids only to fetch the rows again is a round trip for data already found.
        """
        ...

    # --- ingest bookkeeping ------------------------------------------------------------

    async def record_seen(self, document_id: str, *, version_token: str | None = None) -> None:
        """Mark a document as still present at the source, optionally advancing its token.

        The write a skip must not omit. Without ``last_seen_at`` a skip and a deletion are
        indistinguishable to reconciliation; without the token, a source that touches its
        modification date on every save is re-fetched forever.
        """
        ...

    async def annotate(self, document_id: str, updates: Metadata) -> None:
        """Merge keys into a document's metadata, leaving the rest alone."""
        ...

    async def set_lineage(
        self,
        document_id: str,
        *,
        chunk_fp: str | None,
        embed_fp: str | None,
        parse_fp: str | None = None,
        glossary_fp: str | None = None,
    ) -> None:
        """Record which fingerprints *this document* was last built with.

        What makes invalidation set-valued rather than total: a grammar upgrade that changes
        code chunk boundaries and nothing else becomes a query, and so does a ``pypdfium2``
        bump that changes what a PDF's bytes reduce to.

        ``None`` means "leave this one alone", not "clear it". Re-embedding moves only the
        embedding lineage — it does not re-chunk or re-parse — and a store that read ``None``
        as a clear would make "which documents need re-chunking" answer "none" about documents
        that do.

        Args:
            document_id: Which document.
            chunk_fp: Canonical ``ChunkFingerprint``, or ``None`` to leave it.
            embed_fp: Canonical ``EmbedFingerprint``, or ``None`` to leave it.
            parse_fp: Canonical ``ParseFingerprint``, or ``None`` to leave it. Also ``None``
                for a document produced by a parser manicule does not ship and so cannot
                version — recording nothing is the honest answer, and repair selection reads
                it as "eligible" rather than as "current".
            glossary_fp: Canonical ``GlossaryFingerprint``, or ``None`` to leave it. **This is
                the path for a document whose entries were not rewritten**, and there is one
                such case: detection switched off, where the rows stay exactly as they are and
                what gets recorded is that no detector ran. When entries *are* rewritten the
                fingerprint travels with them through
                :meth:`GlossaryWriter.replace_glossary_entries`, so that the rows and the claim
                about which rules produced them are one transaction rather than two.
        """
        ...

    async def set_original(
        self, document_id: str, *, ref: str | None, omitted_reason: str | None
    ) -> None:
        """Point a document at its retained bytes, or say why there are none."""
        ...

    async def requeue_stale(
        self,
        statuses: Collection[DocumentStatus],
        older_than: datetime,
        *,
        detail: str,
    ) -> int:
        """Return in-flight documents to ``pending``, and say how many there were.

        ``statuses`` is an **allowlist**, always. The caller passes
        :data:`~manicule.core.content.IN_FLIGHT`; a store that reinterpreted it as "everything
        except these" would requeue terminal documents forever.
        """
        ...

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        glossary_fp_other_than: str | None = None,
        glossary_fp_unrecorded: bool = False,
    ) -> int:
        """How many documents match. A count, so a diagnostic need not page a corpus.

        ``glossary_fp_other_than`` is here rather than only on :meth:`select_documents`
        because ``doctor`` asks the question and wants the number: an operator is told how many
        documents disagree with the installed detector, and reading them out to count them
        would make a health check proportional to the corpus. ``glossary_fp_unrecorded`` is the
        ``NULL`` half of that, which is a one-time migration rather than ordinary staleness.
        """
        ...

    async def select_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        media_types: Collection[str] | None = None,
        chunk_fp_other_than: str | None = None,
        parse_fp_current: Collection[str] | None = None,
        glossary_fp_other_than: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Document]:
        """The selection ``reindex`` runs over. A query, never a scan.

        Args:
            source: Restrict to one connector.
            statuses: Restrict to these document statuses.
            media_types: Restrict to these media types.
            chunk_fp_other_than: Everything a *different* chunker built.
            glossary_fp_other_than: Everything a *different* detector read definitions out of,
                plus everything nothing has read definitions out of at all. A single value
                rather than the set ``parse_fp_current`` takes, because there is one detector
                where there are as many parsers as media types — and it is spelled as an
                exclusion rather than as "is current" so that the ``NULL`` rows, which are every
                document indexed before the column existed, fall inside the selection rather
                than outside it.
            parse_fp_current: Every parse fingerprint that is current, as canonical strings.
                Selects the complement — documents whose text was produced by a parser
                version no longer installed, plus documents with no recorded lineage at all.
                A set rather than a single value because there is no one current parse
                fingerprint: a corpus holds as many as it has parsers, and a ``pypdfium2``
                bump moves exactly one of them.
            limit: Cap the result.
            offset: How many of the selected documents to skip, in the store's own order.

                What lets a corpus-wide repair page through the selection **while repairing
                it**. A repaired document leaves the set, so paging by a fixed page number
                would skip the documents that shifted forward into the slots it vacated; a
                repair that cannot be repaired stays, and re-reading it every page is a loop
                that never ends. Advancing this by the number of documents a pass *left
                behind* is the one cursor that is right in both directions, and it is only
                sound because the order is stable across calls.
        """
        ...

    # --- sync state --------------------------------------------------------------------

    async def get_watermark(self, connector: str) -> Watermark | None: ...

    async def set_watermark(self, connector: str, watermark: Watermark) -> None: ...

    def known_source_ids(self, connector: str) -> AsyncIterator[SourceId]: ...

    async def connector_metadata(self, connector: str) -> Metadata:
        """Diagnostic state for a connector: last run's counters, last clean reconcile."""
        ...

    async def record_connector_metadata(self, connector: str, updates: Metadata) -> None:
        """Merge keys into a connector's metadata.

        Overwritten rather than accumulated, which is the correct retention policy for a
        diagnostic: a table that only ever grows needs a retention policy nobody asked for.
        """
        ...

    # --- index state -------------------------------------------------------------------

    async def index_fingerprints(self) -> IndexFingerprints: ...

    async def record_index_fingerprints(self, state: IndexFingerprints) -> None: ...

    # --- the vector sweep --------------------------------------------------------------

    async def take_tombstones(self, limit: int) -> Sequence[str]:
        """Chunk ids deleted from SQLite whose vectors have not yet been removed.

        A tombstone list, never an anti-join against ``chunks``. Comparing all vector ids
        against the chunk table races concurrent ingest: an id written after the scan began
        looks like an orphan, and the sweep deletes a live vector. A tombstone only ever names
        something that *was* deleted, so it cannot.
        """
        ...

    async def clear_tombstones(self, chunk_ids: Sequence[str]) -> None:
        """Retire tombstones whose vectors are gone. Idempotent."""
        ...

    async def soft_deleted_before(self, cutoff: datetime, *, limit: int) -> Sequence[str]:
        """Ids of documents soft-deleted before ``cutoff``, whose grace period has expired."""
        ...


@runtime_checkable
class AcquisitionStore(Protocol):
    """The durable source boundary, separate from the legacy publication store."""

    async def create_acquisition_run(self, run_id: str, connector: str) -> AcquisitionRun: ...

    async def get_acquisition_run(self, run_id: str) -> AcquisitionRun | None: ...

    async def latest_unsettled_acquisition_run(self, connector: str) -> AcquisitionRun | None: ...

    async def claim_or_create_acquisition_run(
        self,
        connector: str,
        run_id: str,
        owner: str,
        *,
        now: datetime,
        expires_at: datetime,
    ) -> AcquisitionRun | None:
        """Claim the newest unfinished run, or create its successor, atomically."""
        ...

    async def append_acquisition_record(
        self,
        run_id: str,
        sequence: int,
        source: AcquisitionSource,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRecord: ...

    async def complete_acquisition_enumeration(
        self,
        run_id: str,
        candidate_watermark: Watermark | None,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRun: ...

    async def claim_acquisition_run(
        self, run_id: str, owner: str, *, now: datetime, expires_at: datetime
    ) -> AcquisitionRun | None: ...

    async def renew_acquisition_lease(
        self,
        run_id: str,
        owner: str,
        generation: int,
        *,
        now: datetime,
        expires_at: datetime,
    ) -> bool: ...

    async def release_acquisition_lease(
        self,
        run_id: str,
        owner: str,
        generation: int,
        *,
        now: datetime,
    ) -> bool: ...

    async def transition_acquisition_run(
        self,
        run_id: str,
        expected: AcquisitionRunState,
        target: AcquisitionRunState,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
        diagnostic: AcquisitionDiagnostic | None = None,
    ) -> AcquisitionRun: ...

    async def list_acquisition_records(
        self,
        run_id: str,
        *,
        states: Sequence[AcquisitionRecordState] | None = None,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> Sequence[AcquisitionRecord]: ...

    async def transition_acquisition_record(
        self,
        run_id: str,
        source_id: str,
        expected: AcquisitionRecordState,
        target: AcquisitionRecordState,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
        blob_ref: str | None = None,
        acquired_source: AcquiredSource | None = None,
        fetched_version_token: str | UnsetValue | None = UNSET,
        diagnostic: AcquisitionDiagnostic | None = None,
    ) -> AcquisitionRecord: ...

    async def commit_acquisition_watermark(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> bool: ...

    async def cleanup_acquisition_history(self, cutoff: datetime, *, limit: int = 100) -> int:
        """Discard bounded terminal history; never unfinished or retryable work."""
        ...


@runtime_checkable
class GlossaryWriter(Protocol):
    """A store that can hold the definitions a document states.

    Separate from :class:`IngestStore` rather than another method on it, and the separation is
    the point: glossary detection is a feature that can be switched off, and a store that does
    not implement this must remain a perfectly good ingest target. Folding it in would make
    every conformant store owe an implementation of an optional feature.
    """

    async def replace_glossary_entries(
        self, document_id: str, entries: Sequence[GlossaryEntry], *, fingerprint: str
    ) -> None:
        """Make this document's entries exactly ``entries``, produced by ``fingerprint``.

        Replace rather than merge, on the same principle as ``replace_chunks``: a document is
        re-ingested whole, so merging would leave definitions from a version of the page that
        no longer exists — still queryable, still citing a passage nobody can read.

        **``fingerprint`` is required rather than optional, and that is the point of it.** It is
        the canonical :class:`~manicule.core.fingerprints.GlossaryFingerprint` of the detector
        that produced these entries, and making it a keyword nobody can omit is the loudest
        available answer to the question "what stops a future caller writing rows with no
        lineage" — a default would let one path forget, and a document with entries and no
        fingerprint is exactly the state this whole feature exists to make impossible.

        **Both writes are one transaction.** An implementation that wrote the rows and then the
        fingerprint would leave a crash window in which the entries are the new detector's and
        the column still names the old one — a document that reports itself stale while being
        current is merely wasteful, but the same window the other way round is a document
        reporting itself current while serving superseded rows, and that is the defect. Storing
        them together removes the question.

        ``entries`` may be empty, and an empty write is a real write: it records that the
        current detector read this document and found nothing, which is a different fact from
        nobody having read it.
        """
        ...


@runtime_checkable
class GlossaryStore(GlossaryWriter, Protocol):
    """A glossary store that can also be read back. What the rung-0 repair needs.

    Wider than :class:`GlossaryWriter` and deliberately a second protocol rather than two more
    methods on it: the pipeline only ever writes, and widening what it demands would make an
    optional feature more expensive to implement for the one caller that needs the least.

    The repair needs both reads for reasons it cannot avoid. It reports what a corrected
    detector *did* — entries gained, entries lost, documents whose vocabulary did not move —
    and a sweep that could not read what it replaced could only report that it ran.
    """

    async def glossary_entries(self, document_id: str) -> Sequence[GlossaryEntry]:
        """Every entry one document states, in a stable order."""
        ...

    async def glossary_lineage(self, document_id: str) -> str | None:
        """Which detector last decided this document's entries, or ``None`` if none has.

        The lineage read that touches no glossary text, which is what makes "is this corpus
        current" answerable at the cost of an indexed column rather than of the vocabulary.
        """
        ...


__all__ = ["AcquisitionStore", "GlossaryStore", "GlossaryWriter", "IngestStore"]
