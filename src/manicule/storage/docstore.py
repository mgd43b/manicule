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
from manicule.core.errors import ManiculeError
from manicule.core.retrieval import Candidate, Filter
from manicule.core.sources import SourceId, Watermark
from manicule.storage import models
from manicule.storage.engine import session_factory
from manicule.storage.fts import SEARCH_SQL, escape_match_query
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from manicule.core.content import Chunk

_ANCHOR: TypeAdapter[Anchor] = TypeAdapter(Anchor)
_WATERMARK: TypeAdapter[Watermark] = TypeAdapter(Watermark)

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

    async def known_source_ids(self, connector: str) -> AsyncIterator[SourceId]:
        """Every source id currently indexed for a connector.

        Streamed rather than collected: reconciliation runs over whole corpora, and the diff
        does not need the list in memory at once.
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
