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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Sequence
    from datetime import datetime

    from manicule.core.content import Chunk, Document, DocumentStatus, Metadata
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
    ) -> int: ...

    async def select_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        media_types: Collection[str] | None = None,
        chunk_fp_other_than: str | None = None,
        parse_fp_current: Collection[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Document]:
        """The selection ``reindex`` runs over. A query, never a scan.

        Args:
            source: Restrict to one connector.
            statuses: Restrict to these document statuses.
            media_types: Restrict to these media types.
            chunk_fp_other_than: Everything a *different* chunker built.
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
class GlossaryWriter(Protocol):
    """A store that can hold the definitions a document states.

    Separate from :class:`IngestStore` rather than another method on it, and the separation is
    the point: glossary detection is a feature that can be switched off, and a store that does
    not implement this must remain a perfectly good ingest target. Folding it in would make
    every conformant store owe an implementation of an optional feature.
    """

    async def replace_glossary_entries(
        self, document_id: str, entries: Sequence[GlossaryEntry]
    ) -> None:
        """Make this document's entries exactly ``entries``.

        Replace rather than merge, on the same principle as ``replace_chunks``: a document is
        re-ingested whole, so merging would leave definitions from a version of the page that
        no longer exists — still queryable, still citing a passage nobody can read.
        """
        ...


__all__ = ["GlossaryWriter", "IngestStore"]
