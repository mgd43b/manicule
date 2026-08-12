"""The relational half of storage: :class:`SqliteDocStore`.

SQLite is authoritative. The lexical index and the vector store are derived from it, which is
what gives the two-store consistency problem a single answer — rebuild the derived side —
rather than a reconciliation nobody can adjudicate.

**One class, six protocols.** ``DocStore`` is documents, chunks, lexical search and sync
state; the organisation surfaces — collections, tags, versions, the trash, chunk relations —
arrive as protocols of their own and are implemented by the mixins in the sibling modules.
They share :class:`~manicule.storage.scoped.WorkspaceScoped`, so there is one constructor, one
session factory and one workspace however many contracts this class satisfies. Splitting the
protocols and joining the implementation is deliberate in both directions: a caller declares
the narrow surface it needs, and a tenancy check exists once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import TypeAdapter
from sqlalchemy import bindparam, delete, func, select
from sqlalchemy import text as sql

from manicule.core.content import Document, DocumentStatus
from manicule.core.embedding import EmbedFingerprint, IndexFingerprints
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.retrieval import Candidate, Filter
from manicule.core.sources import SourceId, Watermark
from manicule.storage import models
from manicule.storage.fts import SEARCH_SQL, escape_match_query
from manicule.storage.glossary import GlossaryMixin
from manicule.storage.history import TrashMixin, VersionsMixin
from manicule.storage.organisation import CollectionsMixin, TagsMixin
from manicule.storage.relations import RelationsMixin
from manicule.storage.rows import apply_document, from_chunk, to_chunk, to_document
from manicule.storage.scoped import (
    DEFAULT_WORKSPACE,
    CrossWorkspaceCollisionError,
    WorkspaceScoped,
)
from manicule.storage.types import UtcDateTime, utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Mapping, Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from manicule.core.content import Chunk

_WATERMARK: TypeAdapter[Watermark] = TypeAdapter(Watermark)

_INDEX_STATE_ID = 1
"""``index_state`` holds one row, because a data directory holds one index."""

LISTABLE_FILTER_FIELDS: Final = frozenset(
    {
        "workspace_ids",
        "sources",
        "document_ids",
        "media_types",
        "updated_after",
        "updated_before",
    }
)
""":class:`~manicule.core.retrieval.Filter` fields :meth:`SqliteDocStore.list_documents` honours.

Document-level fields only. ``kinds`` and ``langs`` are properties of a chunk and have no
meaning in a list of documents; ``collection_ids`` and ``tag_ids`` need join tables this query
does not touch. All four are refused rather than ignored, for the same reason
:func:`~manicule.storage.vectors.predicate_for` refuses: a dropped restriction returns rows the
filter was written to exclude, and the listing still looks like it worked.
"""

SEARCHABLE_FILTER_FIELDS: Final = LISTABLE_FILTER_FIELDS | {"kinds", "langs"}
""":class:`~manicule.core.retrieval.Filter` fields :meth:`SqliteDocStore.search_lexical` honours.

The lexical leg is one statement against the authoritative store, so it applies the whole
filter inline, before ``LIMIT`` (``docs/retrieval.md`` §3.3). Only ``collection_ids`` and
``tag_ids`` remain, and those are resolved into ``document_ids`` before either store is
reached.
"""


class SqliteDocStore(
    CollectionsMixin,
    TagsMixin,
    VersionsMixin,
    TrashMixin,
    RelationsMixin,
    GlossaryMixin,
    WorkspaceScoped,
):
    """Every relational protocol manicule has, over SQLite.

    Bound to one workspace. Every read and write carries that scope, so tenancy is a property
    of the handle rather than a parameter each call site has to remember.

    Satisfies :class:`~manicule.core.protocols.DocStore`,
    :class:`~manicule.core.protocols.CollectionStore`,
    :class:`~manicule.core.protocols.TagStore`,
    :class:`~manicule.core.protocols.VersionStore`,
    :class:`~manicule.core.protocols.TrashStore`,
    :class:`~manicule.core.protocols.ChunkRelationStore` and
    :class:`~manicule.retrieval.ports.GlossarySource`.
    """

    # --- documents ------------------------------------------------------------------------

    async def get_document(self, document_id: str) -> Document | None:
        async with self._sessions() as session:
            row = await self._live_document(session, document_id)
            return None if row is None else to_document(row)

    async def find_document(self, source: str, source_id: SourceId) -> Document | None:
        """Look a document up by where it came from.

        Identity is ``(workspace, source, source_id)`` — never the URI, which is display data
        a source is free to change.
        """
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(models.Document).where(
                        models.Document.workspace_id == self._workspace_id,
                        models.Document.source == source,
                        models.Document.source_id == source_id,
                        models.Document.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            return None if row is None else to_document(row)

    async def upsert_document(self, document: Document) -> Document:
        async with self._sessions.begin() as session:
            row = await session.get(models.Document, document.id)
            if row is not None and row.workspace_id != self._workspace_id:
                msg = (
                    f"document id {document.id!r} belongs to workspace "
                    f"{row.workspace_id!r}, not {self._workspace_id!r}. Ids from "
                    f"manicule.core.ids.document_id include the workspace, so this id was "
                    f"built some other way."
                )
                raise CrossWorkspaceCollisionError(msg)
            if row is None:
                row = models.Document(id=document.id, workspace_id=self._workspace_id)
                session.add(row)
            else:
                # **History is written here or nowhere.** This is the one place both states of
                # a document are visible at once, and it is inside the transaction that
                # replaces one with the other — so the version and the change it explains
                # either both land or neither does. A caller-driven "record a version" would
                # be two transactions with a crash window between them, and the crash loses
                # exactly the record that exists to explain the change.
                await self._record_supersession(session, row, document)
            apply_document(row, document)
            row.last_seen_at = utcnow()
            await session.flush()
            return to_document(row)

    async def set_status(self, document_id: str, status: DocumentStatus, detail: str = "") -> None:
        """Record an outcome, keeping ``failed_stage`` consistent with ``status``.

        A batch records failures and keeps going, so this never raises for an unknown
        document — a document deleted underneath a run is not an error in the run.

        Raises:
            ValueError: For :attr:`~manicule.core.content.DocumentStatus.FAILED`, which this
                method cannot express: it takes no stage, and the schema's
                ``failed_stage_iff_failed`` constraint requires one. Without this the call
                reaches the database and comes back as an ``IntegrityError`` naming a
                constraint rather than the missing argument — the right outcome by luck and the
                wrong diagnosis. Write the whole document through :meth:`upsert_document`,
                which carries the stage.
        """
        if status is DocumentStatus.FAILED:
            msg = (
                "set_status cannot record 'failed': the schema requires a failed_stage with it "
                "and this method takes none. Use upsert_document with status, status_detail and "
                "failed_stage together."
            )
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            row = await session.get(models.Document, document_id)
            if row is None:
                return
            row.status = status
            row.status_detail = detail or None
            # Unconditional, because the guard above has already refused the one status that
            # may carry a stage. Leaving a stale `failed_stage` behind on a document that has
            # since moved on would break the schema's own iff constraint on the next write.
            row.failed_stage = None
            if status is DocumentStatus.INDEXED:
                row.indexed_at = utcnow()

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        statement = (
            select(models.Document)
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.deleted_at.is_(None),
            )
            .order_by(models.Document.created_at.desc(), models.Document.id)
            .limit(limit)
            .offset(offset)
        )
        self._require_honourable(filter, LISTABLE_FILTER_FIELDS, "list documents")
        if filter is not None:
            if filter.sources:
                statement = statement.where(models.Document.source.in_(filter.sources))
            if filter.document_ids:
                statement = statement.where(models.Document.id.in_(filter.document_ids))
            if filter.media_types:
                statement = statement.where(models.Document.media_type.in_(filter.media_types))
            if filter.updated_after is not None:
                statement = statement.where(models.Document.updated_at > filter.updated_after)
            if filter.updated_before is not None:
                statement = statement.where(models.Document.updated_at < filter.updated_before)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).scalars().all()
            return [to_document(row) for row in rows]

    async def delete_document(self, document_id: str) -> None:
        """Hard-delete a document and everything hanging off it. Idempotent.

        The cascade reaches ``chunks``, whose delete trigger removes the FTS rows and records
        vector tombstones — so the derived stores are cleaned by the database rather than by
        whichever caller remembered to.
        """
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.Document).where(
                    models.Document.id == document_id,
                    models.Document.workspace_id == self._workspace_id,
                )
            )

    # --- ingest bookkeeping -----------------------------------------------------------------

    async def record_seen(self, document_id: str, *, version_token: str | None = None) -> None:
        """Mark a document as still present at the source. The write a skip must not omit.

        Without ``last_seen_at`` a skip and a deletion are indistinguishable to reconciliation.
        Without advancing the token when the *bytes* were unchanged, a source that touches its
        modification date on every save is fetched again on every sync, forever.
        """
        async with self._sessions.begin() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                return
            row.last_seen_at = utcnow()
            if version_token is not None:
                row.version_token = version_token

    async def annotate(self, document_id: str, updates: Mapping[str, Any]) -> None:
        """Merge keys into a document's metadata, leaving everything else alone.

        A whole-object replace would lose whatever a middleware annotated a moment earlier, and
        lose it silently — the two writers are in different stages and neither can see the
        other.
        """
        async with self._sessions.begin() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                return
            merged: dict[str, Any] = dict(cast("Any", row.doc_metadata) or {})
            merged.update(updates)
            row.doc_metadata = cast("Any", merged)

    async def set_lineage(
        self,
        document_id: str,
        *,
        chunk_fp: str | None,
        embed_fp: str | None,
        parse_fp: str | None = None,
    ) -> None:
        """Record which fingerprints this document was last built with.

        ``None`` leaves a lineage unchanged rather than clearing it: re-embedding moves only the
        embedding lineage, and clearing the chunk one would make "which documents need
        re-chunking" answer "none" about documents that do. The same holds for ``parse_fp``,
        which no path other than a re-parse is entitled to move.
        """
        async with self._sessions.begin() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                return
            if parse_fp is not None:
                row.parse_fp = parse_fp
            if chunk_fp is not None:
                row.chunk_fp = chunk_fp
            if embed_fp is not None:
                row.embed_fp = embed_fp

    async def set_original(
        self, document_id: str, *, ref: str | None, omitted_reason: str | None
    ) -> None:
        """Point a document at its retained bytes, or record why it has none."""
        async with self._sessions.begin() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                return
            row.original_ref = ref
            row.original_omitted_reason = omitted_reason

    async def requeue_stale(
        self,
        statuses: Collection[DocumentStatus],
        older_than: datetime,
        *,
        detail: str = "",
    ) -> int:
        """Return documents stuck in ``statuses`` to ``pending``, and say how many.

        ``statuses`` is used exactly as given — an allowlist the caller chose. It is never
        inverted here, because a denylist would requeue terminal documents forever: a
        ``container`` has zero chunks by design, and so does a document that yielded no text.
        """
        wanted = list(statuses)
        if not wanted:
            return 0
        async with self._sessions.begin() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Document).where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_(None),
                            models.Document.status.in_(wanted),
                            models.Document.updated_at < older_than,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                row.status = DocumentStatus.PENDING
                row.status_detail = detail or None
                row.failed_stage = None
            return len(rows)

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(models.Document)
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.deleted_at.is_(None),
            )
        )
        if source is not None:
            statement = statement.where(models.Document.source == source)
        if statuses is not None:
            statement = statement.where(models.Document.status.in_(list(statuses)))
        async with self._sessions() as session:
            return (await session.execute(statement)).scalar_one()

    async def document_statistics(self) -> dict[str, dict[str, int]]:
        """Live document counts grouped by source, media type and status.

        Three ``GROUP BY`` queries rather than a listing the caller counts itself. A page of
        documents answers a different question, and summing one reports the page while looking
        exactly like a total — the same class of quiet wrongness as a filter a store dropped.

        Scoped like every other read here: the workspace is on the handle, and the trash is
        excluded, so a soft-deleted document is not counted as one that is in the index.
        """
        columns = {
            "by_source": models.Document.source,
            "by_media_type": models.Document.media_type,
            "by_status": models.Document.status,
        }
        grouped: dict[str, dict[str, int]] = {}
        async with self._sessions() as session:
            for label, column in columns.items():
                statement = (
                    select(column, func.count())
                    .where(
                        models.Document.workspace_id == self._workspace_id,
                        models.Document.deleted_at.is_(None),
                    )
                    .group_by(column)
                )
                rows = (await session.execute(statement)).all()
                grouped[label] = {
                    (value.value if isinstance(value, DocumentStatus) else str(value)): count
                    for value, count in rows
                }
        return grouped

    async def select_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        media_types: Collection[str] | None = None,
        chunk_fp_other_than: str | None = None,
        parse_fp_current: Collection[str] | None = None,
        limit: int | None = None,
    ) -> Sequence[Document]:
        """The selection a repair verb runs over. A query, never a scan.

        ``chunk_fp_other_than`` is what makes invalidation set-valued: "everything a different
        chunker built" is one indexed predicate, so a grammar upgrade repairs the documents in
        that language and leaves the corpus alone.

        ``parse_fp_current`` is the same idea one stage earlier, and it takes a *set* because
        parsing has no single corpus-wide identity: the complement of "every parse fingerprint
        that is current" is exactly the documents a library bump changed the text of. A
        ``NULL`` lineage is in that complement deliberately — no recorded fingerprint means no
        evidence the stored text is current, and a repair selector that assumed it was would
        skip precisely the documents predating the column.

        An empty collection is not the same as ``None``: ``None`` means "do not filter on
        parse lineage", while an empty set means "nothing is current", which selects every
        document. Both are reachable — the second from an installation with no parsers
        configured — so they are kept distinct rather than collapsed by a falsy test.
        """
        statement = (
            select(models.Document)
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.deleted_at.is_(None),
            )
            .order_by(models.Document.created_at, models.Document.id)
        )
        if source is not None:
            statement = statement.where(models.Document.source == source)
        if statuses is not None:
            statement = statement.where(models.Document.status.in_(list(statuses)))
        if media_types is not None:
            statement = statement.where(models.Document.media_type.in_(list(media_types)))
        if chunk_fp_other_than is not None:
            statement = statement.where(
                (models.Document.chunk_fp.is_(None))
                | (models.Document.chunk_fp != chunk_fp_other_than)
            )
        if parse_fp_current is not None:
            statement = statement.where(
                (models.Document.parse_fp.is_(None))
                | (models.Document.parse_fp.notin_(list(parse_fp_current)))
            )
        if limit is not None:
            statement = statement.limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).scalars().all()
            return [to_document(row) for row in rows]

    # --- index state ------------------------------------------------------------------------

    async def index_fingerprints(self) -> IndexFingerprints:
        """What this index says it was built with, or an empty answer if it has said nothing."""
        async with self._sessions() as session:
            row = await session.get(models.IndexState, _INDEX_STATE_ID)
            if row is None:
                return IndexFingerprints()
            return IndexFingerprints(
                embed=(
                    EmbedFingerprint.model_validate_json(row.embed_fingerprint)
                    if row.embed_fingerprint
                    else None
                ),
                chunk=(
                    ChunkFingerprint.model_validate_json(row.chunk_fingerprint)
                    if row.chunk_fingerprint
                    else None
                ),
                vector_table=row.vector_table,
            )

    async def record_index_fingerprints(self, state: IndexFingerprints) -> None:
        """Commit the index to a shape. One row, because there is one index."""
        async with self._sessions.begin() as session:
            row = await session.get(models.IndexState, _INDEX_STATE_ID)
            if row is None:
                row = models.IndexState(id=_INDEX_STATE_ID)
                session.add(row)
            row.embed_fingerprint = state.embed.model_dump_json() if state.embed else None
            row.chunk_fingerprint = state.chunk.model_dump_json() if state.chunk else None
            row.vector_table = state.vector_table

    # --- the vector sweep ---------------------------------------------------------------------

    async def take_tombstones(self, limit: int) -> Sequence[str]:
        """Chunk ids whose vectors have not yet been swept, oldest first.

        Read, never deleted here. The vectors go first and the tombstones are cleared second,
        so a crash between the two leaves a tombstone for a vector that is already gone — which
        the next pass handles as a no-op. The reverse order would leave a live vector with
        nothing left to record that it should go.
        """
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.VectorTombstone.chunk_id)
                        .order_by(
                            models.VectorTombstone.deleted_at, models.VectorTombstone.chunk_id
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    async def clear_tombstones(self, chunk_ids: Sequence[str]) -> None:
        """Retire tombstones whose vectors are gone. Idempotent."""
        if not chunk_ids:
            return
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.VectorTombstone).where(
                    models.VectorTombstone.chunk_id.in_(list(chunk_ids))
                )
            )

    async def soft_deleted_before(self, cutoff: datetime, *, limit: int = 1000) -> Sequence[str]:
        """Documents whose grace period has expired and whose content is still present.

        **Already-purged documents are excluded, and that is what makes the sweep terminate.**
        Purging removes chunks and vectors; it does not touch ``deleted_at``, because the row is
        retained so a citation can still explain itself. A selection keyed only on ``deleted_at``
        would therefore return the same documents on every pass — re-deleting vectors that are
        gone and re-emptying chunks that are already empty, forever — and, because the ``LIMIT``
        is over an ordered query, would return the *same first thousand* every time, so nothing
        past them would ever be purged at all. ``status = 'deleted'`` is what the sweep sets when
        it is done with a document, so it is what this excludes.
        """
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Document.id)
                        .where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_not(None),
                            models.Document.deleted_at < cutoff,
                            models.Document.status != DocumentStatus.DELETED,
                        )
                        .order_by(models.Document.deleted_at, models.Document.id)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    # --- chunks ---------------------------------------------------------------------------

    async def replace_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        """Replace a document's chunks wholesale. Re-parsing is not additive.

        Deleting first and inserting second means a chunk that survived a re-parse unchanged
        keeps its id — and therefore its vector — because the id is derived from its content.
        """
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.Chunk).where(models.Chunk.document_id == document_id)
            )
            for chunk in chunks:
                session.add(from_chunk(chunk, document_id))

    async def get_chunks(self, chunk_ids: Sequence[str]) -> Sequence[Chunk]:
        if not chunk_ids:
            return []
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Chunk).where(models.Chunk.id.in_(list(chunk_ids)))
                    )
                )
                .scalars()
                .all()
            )
            by_id = {row.id: to_chunk(row) for row in rows}
            return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]:
        """Every live chunk of one document in this workspace, in position order.

        Returned whole rather than as ids because both readers — repair and re-embed — need
        ``embed_text`` and ``token_count``, and fetching ids only to fetch the rows again is a
        round trip for data already found.

        **Joined to ``documents`` for the scope.** ``chunks`` carries no ``workspace_id`` of its
        own, so a query on ``document_id`` alone answers about any tenant's document and about
        soft-deleted ones. Today every caller passes an id that came from a scoped query, which
        makes this an unguarded boundary rather than a leak — but this is the read that feeds
        ``reindex --re-embed``, and a repair verb that can be pointed at an id is exactly where
        an unscoped read stops being theoretical.
        """
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Chunk)
                        .join(models.Document, models.Document.id == models.Chunk.document_id)
                        .where(
                            models.Chunk.document_id == document_id,
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_(None),
                        )
                        .order_by(models.Chunk.position, models.Chunk.seq)
                    )
                )
                .scalars()
                .all()
            )
            return [to_chunk(row) for row in rows]

    async def count_chunks(self, document_id: str | None = None) -> int:
        statement = select(func.count()).select_from(models.Chunk)
        if document_id is not None:
            statement = statement.where(models.Chunk.document_id == document_id)
        async with self._sessions() as session:
            return (await session.execute(statement)).scalar_one()

    async def live_chunk_count(self) -> int:
        """Chunks a search in this workspace could legitimately return.

        The numerator of the dense leg's over-fetch factor (``docs/retrieval.md`` §4.3): the
        vector table holds a row per chunk with no column for tenancy, liveness or status, so
        the fraction of its rows that survive the hydrating join is what says how far past
        ``k`` the leg has to reach. Deliberately narrower than :meth:`count_chunks`, which
        counts every chunk in the database including other tenants' and soft-deleted ones —
        using that as the numerator would report a diluted index as a clean one and under-fetch
        on exactly the deployments that need it most.

        One aggregate over an indexed join, computed once per generation rather than per query.
        """
        statement = (
            select(func.count())
            .select_from(models.Chunk)
            .join(models.Document, models.Document.id == models.Chunk.document_id)
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.deleted_at.is_(None),
                models.Document.status == DocumentStatus.INDEXED,
            )
        )
        async with self._sessions() as session:
            return (await session.execute(statement)).scalar_one()

    # --- lexical search -------------------------------------------------------------------

    async def search_lexical(
        self,
        text: str,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
    ) -> list[Candidate]:
        """BM25 over stored chunk text, best first.

        One joined statement, so ``LIMIT`` applies after the workspace, soft-delete and status
        filters rather than before them. Filtering afterwards silently returns fewer than
        ``k`` live rows, because deferred deletion leaves soft-deleted chunks in the index
        competing for the same slots.
        """
        self._require_honourable(filter, SEARCHABLE_FILTER_FIELDS, "search")
        match = escape_match_query(text)
        if not match:
            return []

        clauses = ["AND d.workspace_id = :workspace_id"]
        params: dict[str, Any] = {
            "match": match,
            "limit": k,
            "workspace_id": self._workspace_id,
        }
        timestamps: list[str] = []
        if filter is not None:
            for column, values in (
                ("d.source", sorted(filter.sources)),
                ("d.id", sorted(filter.document_ids)),
                ("d.media_type", sorted(filter.media_types)),
                ("c.kind", sorted(kind.value for kind in filter.kinds)),
                ("c.lang", sorted(filter.langs)),
            ):
                if not values:
                    continue
                names = _bind_names(column.replace(".", "_"), len(values))
                clauses.append(f"AND {column} IN ({', '.join(':' + n for n in names)})")
                params.update(dict(zip(names, values, strict=True)))
            for name, comparison, moment in (
                ("updated_after", ">", filter.updated_after),
                ("updated_before", "<", filter.updated_before),
            ):
                if moment is None:
                    continue
                clauses.append(f"AND d.updated_at {comparison} :{name}")
                params[name] = moment
                timestamps.append(name)

        statement = sql(SEARCH_SQL.format(extra="\n  ".join(clauses)))
        if timestamps:
            # Textual SQL binds nothing by type on its own, so a datetime would reach the
            # driver as whatever sqlite3 makes of it rather than in the one encoding
            # `UtcDateTime` writes. Same column, same conversion, one representation.
            statement = statement.bindparams(
                *(bindparam(name, type_=UtcDateTime()) for name in timestamps)
            )
        async with self._sessions() as session:
            rows = (await session.execute(statement, params)).all()
            chunk_ids = [str(row.chunk_id) for row in rows]
            chunks = {chunk.id: chunk for chunk in await self.get_chunks(chunk_ids)}

        candidates: list[Candidate] = []
        for row in rows:
            chunk = chunks.get(str(row.chunk_id))
            if chunk is None:  # pragma: no cover - the join guarantees the row exists
                continue
            # bm25() is negative and more negative is better; negate so higher is better.
            candidates.append(
                Candidate(
                    chunk=chunk,
                    score=-float(row.rank_score),
                    scores={"bm25": -float(row.rank_score)},
                )
            )
        return candidates

    # --- sync state -----------------------------------------------------------------------

    async def get_watermark(self, connector: str) -> Watermark | None:
        async with self._sessions() as session:
            row = await self._connector_row(session, connector)
            if row is None or row.watermark is None:
                return None
            return _WATERMARK.validate_python(row.watermark)

    async def set_watermark(self, connector: str, watermark: Watermark) -> None:
        """Record how far a sync got.

        Advanced only on a clean run, which is what makes an interrupted sync resume from the
        last good point rather than from the beginning.
        """
        async with self._sessions.begin() as session:
            row = await self._connector_row(session, connector)
            if row is None:
                row = models.Connector(
                    id=f"{self._workspace_id}:{connector}",
                    workspace_id=self._workspace_id,
                    name=connector,
                    type=connector,
                    config={},
                )
                session.add(row)
            row.watermark = watermark.model_dump(mode="json")
            row.last_synced_at = utcnow()

    async def connector_metadata(self, connector: str) -> dict[str, Any]:
        """A connector's diagnostic state: last run's counters, last clean reconcile."""
        async with self._sessions() as session:
            row = await self._connector_row(session, connector)
            if row is None:
                return {}
            return dict(cast("Any", row.run_metadata) or {})

    async def record_connector_metadata(self, connector: str, updates: Mapping[str, Any]) -> None:
        """Merge keys into a connector's metadata, dropping any set to ``None``.

        Overwritten rather than accumulated: run history is diagnostic, not relational, and a
        table that only ever grows needs a retention policy nobody has asked for. Dropping on
        ``None`` is what lets a confirmed proposal be *removed* rather than left as a null that
        every reader then has to interpret.
        """
        async with self._sessions.begin() as session:
            row = await self._connector_row(session, connector)
            if row is None:
                row = models.Connector(
                    id=f"{self._workspace_id}:{connector}",
                    workspace_id=self._workspace_id,
                    name=connector,
                    type=connector,
                    config={},
                )
                session.add(row)
            merged: dict[str, Any] = dict(cast("Any", row.run_metadata) or {})
            for key, value in updates.items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            row.run_metadata = cast("Any", merged)

    async def known_source_ids(self, connector: str) -> AsyncIterator[SourceId]:
        """Every source id currently indexed for a connector.

        Streamed rather than collected: reconciliation runs over whole corpora, and the diff
        does not need the list in memory at once.

        **Close it.** This is an async generator holding an open session while suspended, so a
        consumer that stops early leaves the session open until the generator is finalised —
        which happens at garbage-collection time, through the event loop's async-generator
        hook, possibly after the loop it belongs to has closed. Wrap consumption in
        :func:`contextlib.aclosing` rather than draining it bare, and the session closes when
        the block does.
        """
        async with self._sessions() as session:
            result = await session.stream(
                select(models.Document.source_id).where(
                    models.Document.workspace_id == self._workspace_id,
                    models.Document.source == connector,
                    models.Document.deleted_at.is_(None),
                )
            )
            async for (source_id,) in result:
                yield source_id

    # --- internals ------------------------------------------------------------------------

    async def _connector_row(
        self, session: AsyncSession, connector: str
    ) -> models.Connector | None:
        return (
            await session.execute(
                select(models.Connector).where(
                    models.Connector.workspace_id == self._workspace_id,
                    models.Connector.name == connector,
                    models.Connector.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()


def _bind_names(prefix: str, count: int) -> list[str]:
    """Named bind parameters, so a set never reaches SQL as interpolated text."""
    return [f"{prefix}_{index}" for index in range(count)]


__all__ = [
    "DEFAULT_WORKSPACE",
    "LISTABLE_FILTER_FIELDS",
    "SEARCHABLE_FILTER_FIELDS",
    "CrossWorkspaceCollisionError",
    "SqliteDocStore",
]
