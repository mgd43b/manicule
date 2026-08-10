"""The relational half of storage: :class:`SqliteDocStore`.

SQLite is authoritative. The lexical index and the vector store are derived from it, which is
what gives the two-store consistency problem a single answer — rebuild the derived side —
rather than a reconciliation nobody can adjudicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter
from sqlalchemy import delete, func, select
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from manicule.core.anchors import Anchor
from manicule.core.content import BlockKind, Document, DocumentStatus
from manicule.core.embedding import EmbedFingerprint, IndexFingerprints
from manicule.core.errors import ManiculeError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.retrieval import Candidate, Filter
from manicule.core.sources import SourceId, Watermark
from manicule.storage import models
from manicule.storage.engine import session_factory
from manicule.storage.fts import SEARCH_SQL, escape_match_query
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Mapping, Sequence
    from datetime import datetime

    from manicule.core.content import Chunk

_ANCHOR: TypeAdapter[Anchor] = TypeAdapter(Anchor)
_WATERMARK: TypeAdapter[Watermark] = TypeAdapter(Watermark)

_INDEX_STATE_ID = 1
"""``index_state`` holds one row, because a data directory holds one index."""

DEFAULT_WORKSPACE = "default"
"""Personal mode has one workspace and never says so out loud.

The workspace is bound to the store handle rather than passed per call, so a query cannot
forget it. Team mode supplies a real id; personal mode gets this one, and the isolation
predicate is identical either way — which means the path that enforces tenancy is exercised
by every test rather than only by the team-mode ones.
"""


class CrossWorkspaceCollisionError(ManiculeError):
    """A document id was offered to a workspace that does not own it.

    **This should be unreachable.** :func:`manicule.core.ids.document_id` takes the workspace
    as the first component of its digest, so two tenants indexing the same upstream source
    derive different ids by construction. What remains is a caller that built an id some other
    way, or a workspace mismatch between the handle and the document it was handed.

    Kept as a guard rather than deleted, because the failure it catches is silent: an id
    computed without the workspace lands on another tenant's row, overwriting content its
    author cannot read while its own document appears to vanish. An assertion that cannot fire
    costs one comparison per write; the same bug without it costs a tenant's data.
    """


class SqliteDocStore:
    """:class:`~manicule.core.protocols.DocStore` over SQLite.

    Bound to one workspace. Every read and write carries that scope, so tenancy is a property
    of the handle rather than a parameter each call site has to remember.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        workspace_id: str = DEFAULT_WORKSPACE,
        sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._engine = engine
        self._workspace_id = workspace_id
        self._sessions = sessions or session_factory(engine)

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    async def ensure_workspace(self) -> None:
        """Create this store's workspace row if it is absent. Idempotent."""
        async with self._sessions.begin() as session:
            existing = await session.get(models.Workspace, self._workspace_id)
            if existing is None:
                session.add(
                    models.Workspace(id=self._workspace_id, name=self._workspace_id, settings={})
                )

    # --- documents ------------------------------------------------------------------------

    async def get_document(self, document_id: str) -> Document | None:
        async with self._sessions() as session:
            row = await self._live_document(session, document_id)
            return None if row is None else _to_document(row)

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
            return None if row is None else _to_document(row)

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
            _apply_document(row, document)
            row.last_seen_at = utcnow()
            await session.flush()
            return _to_document(row)

    async def set_status(self, document_id: str, status: DocumentStatus, detail: str = "") -> None:
        """Record an outcome, keeping ``failed_stage`` consistent with ``status``.

        A batch records failures and keeps going, so this never raises for an unknown
        document — a document deleted underneath a run is not an error in the run.
        """
        async with self._sessions.begin() as session:
            row = await session.get(models.Document, document_id)
            if row is None:
                return
            row.status = status
            row.status_detail = detail or None
            if status is not DocumentStatus.FAILED:
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
        self._require_same_workspace(filter)
        if filter is not None:
            if filter.source is not None:
                statement = statement.where(models.Document.source == filter.source)
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
            return [_to_document(row) for row in rows]

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

    async def soft_delete_document(self, document_id: str) -> None:
        """Mark a document deleted without touching the derived stores.

        Chunks, vectors and FTS rows all stay. They become invisible at the join in
        :meth:`search_lexical`, so restoring is clearing a timestamp — no re-embed, no
        re-parse, no re-fetch.
        """
        async with self._sessions.begin() as session:
            row = await session.get(models.Document, document_id)
            if row is not None and row.workspace_id == self._workspace_id:
                row.deleted_at = utcnow()

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
        self, document_id: str, *, chunk_fp: str | None, embed_fp: str | None
    ) -> None:
        """Record which fingerprints this document was last built with.

        ``None`` leaves a lineage unchanged rather than clearing it: re-embedding moves only the
        embedding lineage, and clearing the chunk one would make "which documents need
        re-chunking" answer "none" about documents that do.
        """
        async with self._sessions.begin() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                return
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

    async def select_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        media_types: Collection[str] | None = None,
        chunk_fp_other_than: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Document]:
        """The selection a repair verb runs over. A query, never a scan.

        ``chunk_fp_other_than`` is what makes invalidation set-valued: "everything a different
        chunker built" is one indexed predicate, so a grammar upgrade repairs the documents in
        that language and leaves the corpus alone.
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
        if limit is not None:
            statement = statement.limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).scalars().all()
            return [_to_document(row) for row in rows]

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
        """Documents whose grace period has expired, so the sweep may purge their chunks."""
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Document.id)
                        .where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_not(None),
                            models.Document.deleted_at < cutoff,
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
                session.add(_from_chunk(chunk, document_id))

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
            by_id = {row.id: _to_chunk(row) for row in rows}
            return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]:
        """Every chunk of one document, in position order.

        Returned whole rather than as ids because both readers — repair and re-embed — need
        ``embed_text`` and ``token_count``, and fetching ids only to fetch the rows again is a
        round trip for data already found.
        """
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Chunk)
                        .where(models.Chunk.document_id == document_id)
                        .order_by(models.Chunk.position, models.Chunk.seq)
                    )
                )
                .scalars()
                .all()
            )
            return [_to_chunk(row) for row in rows]

    async def count_chunks(self, document_id: str | None = None) -> int:
        statement = select(func.count()).select_from(models.Chunk)
        if document_id is not None:
            statement = statement.where(models.Chunk.document_id == document_id)
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
        self._require_same_workspace(filter)
        match = escape_match_query(text)
        if not match:
            return []

        clauses = ["AND d.workspace_id = :workspace_id"]
        params: dict[str, Any] = {
            "match": match,
            "limit": k,
            "workspace_id": self._workspace_id,
        }
        if filter is not None and filter.source is not None:
            clauses.append("AND d.source = :source")
            params["source"] = filter.source
        if filter is not None and filter.document_ids:
            names = _bind_names("doc", len(filter.document_ids))
            clauses.append(f"AND d.id IN ({', '.join(':' + n for n in names)})")
            params.update(dict(zip(names, sorted(filter.document_ids), strict=True)))
        if filter is not None and filter.kinds:
            names = _bind_names("kind", len(filter.kinds))
            clauses.append(f"AND c.kind IN ({', '.join(':' + n for n in names)})")
            params.update(
                dict(zip(names, sorted(kind.value for kind in filter.kinds), strict=True))
            )

        statement = sql(SEARCH_SQL.format(extra="\n  ".join(clauses)))
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

    def _require_same_workspace(self, filter: Filter | None) -> None:  # noqa: A002
        """Refuse a filter that names a workspace this handle does not serve.

        The workspace comes from the handle, so a filter carrying a *different* one is a
        caller who believes they are querying somewhere else. Ignoring it silently would
        answer a question nobody asked, which is the shape a cross-tenant leak takes.
        """
        if filter is None or filter.workspace_id is None:
            return
        if filter.workspace_id != self._workspace_id:
            msg = (
                f"filter names workspace {filter.workspace_id!r} but this store serves "
                f"{self._workspace_id!r}. Open a store for that workspace instead."
            )
            raise CrossWorkspaceCollisionError(msg)

    async def _live_document(
        self, session: AsyncSession, document_id: str
    ) -> models.Document | None:
        row = await session.get(models.Document, document_id)
        if row is None or row.workspace_id != self._workspace_id or row.deleted_at is not None:
            return None
        return row

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


def _apply_document(row: models.Document, document: Document) -> None:
    # Writing a document is asserting that it exists, so an upsert clears a soft delete. A
    # document removed at the source and later restored there arrives through exactly this
    # path, and leaving the timestamp would index it into a row nothing can see.
    row.deleted_at = None
    row.source = document.source
    row.source_id = document.source_id
    row.uri = document.uri
    row.title = document.title
    row.media_type = document.media_type
    row.content_hash = document.content_hash
    row.version_token = document.version_token
    row.original_ref = document.original_ref
    row.status = document.status
    row.status_detail = document.status_detail
    row.failed_stage = document.failed_stage
    row.doc_metadata = cast("Any", dict(document.metadata))
    if document.status is DocumentStatus.INDEXED:
        row.indexed_at = utcnow()


def _to_document(row: models.Document) -> Document:
    return Document(
        id=row.id,
        source=row.source,
        source_id=row.source_id,
        uri=row.uri,
        title=row.title,
        content_hash=row.content_hash,
        version_token=row.version_token,
        original_ref=row.original_ref,
        media_type=row.media_type,
        status=row.status,
        status_detail=row.status_detail,
        failed_stage=row.failed_stage,
        metadata=cast("Any", row.doc_metadata or {}),
    )


def _from_chunk(chunk: Chunk, document_id: str) -> models.Chunk:
    return models.Chunk(
        id=chunk.id,
        document_id=document_id,
        text=chunk.text,
        embed_text=chunk.embed_text,
        heading_text=" > ".join(chunk.heading_path),
        heading_path=list(chunk.heading_path),
        kind=chunk.kind,
        lang=None,
        position=chunk.position,
        token_count=chunk.token_count,
        anchor=chunk.anchor.model_dump(mode="json"),
        chunk_metadata=cast("Any", dict(chunk.metadata)),
    )


def _to_chunk(row: models.Chunk) -> Chunk:
    from manicule.core.content import Chunk as ChunkModel  # noqa: PLC0415 - avoids a cycle

    heading_path = cast("list[str]", row.heading_path or [])
    return ChunkModel(
        id=row.id,
        document_id=row.document_id,
        text=row.text,
        embed_text=row.embed_text,
        anchor=_ANCHOR.validate_python(row.anchor),
        heading_path=tuple(heading_path),
        kind=BlockKind(row.kind),
        position=row.position,
        token_count=row.token_count,
        metadata=cast("Any", row.chunk_metadata or {}),
    )


__all__ = ["DEFAULT_WORKSPACE", "CrossWorkspaceCollisionError", "SqliteDocStore"]
