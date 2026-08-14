"""Typed edges between chunks: parent links and sibling links.

``chunk_relations`` is the table that only becomes possible once chunks are a table of their
own (``docs/storage.md`` §4.1). Both columns are real foreign keys with ``ON DELETE CASCADE``,
so an edge cannot outlive either end and orphan cleanup is the database's job rather than a
pattern match over formatted identifiers.

Two properties of the shipped schema drive everything here:

* **The composite primary key leads with ``source_chunk_id``, and there is a second index on
  ``target_chunk_id``.** That index exists because lookups are ``WHERE source = ? OR target =
  ?``, which the primary key alone cannot serve. So an edge is written **once** and read from
  both ends. A mirror row would double the table to answer a query the schema already answers,
  and would create a pair that can drift apart.
* **There is no workspace column**, because a chunk reaches its workspace through its document.
  Tenancy is therefore this module's to enforce on both ends of every edge, on the way in and
  on the way out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, or_, select

from manicule.core.errors import UnknownEntityError
from manicule.core.organization import ChunkEdge, ChunkRelationType
from manicule.storage import models
from manicule.storage.scoped import WorkspaceScoped

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from sqlalchemy.ext.asyncio import AsyncSession


class RelationsMixin(WorkspaceScoped):
    """:class:`~manicule.core.protocols.ChunkRelationStore` over SQLite."""

    async def relate(
        self, source_chunk_id: str, target_chunk_id: str, relation_type: ChunkRelationType
    ) -> None:
        """Record one edge. Idempotent.

        Both ends are checked against this workspace before anything is written. **That check
        is the whole tenancy boundary for this table.** ``chunk_relations`` has no workspace
        column and its foreign keys reach ``chunks``, which has none either — so nothing in the
        schema stops an edge from one tenant's chunk to another's, and a later lookup from the
        first would hand back the second's chunk id. It is a small leak and an entirely silent
        one: the write succeeds, the read succeeds, and the result is a valid identifier from a
        corpus the caller cannot otherwise see.
        """
        if source_chunk_id == target_chunk_id:
            msg = (
                f"a chunk cannot relate to itself; got {source_chunk_id!r} on both ends. The "
                f"schema refuses this too, with a CHECK constraint that would report the "
                f"constraint rather than the argument."
            )
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            await self._require_visible_chunks(session, (source_chunk_id, target_chunk_id))
            existing = await session.get(
                models.ChunkRelation, (source_chunk_id, target_chunk_id, relation_type.value)
            )
            if existing is None:
                session.add(
                    models.ChunkRelation(
                        source_chunk_id=source_chunk_id,
                        target_chunk_id=target_chunk_id,
                        relation_type=relation_type.value,
                    )
                )

    async def unrelate(
        self, source_chunk_id: str, target_chunk_id: str, relation_type: ChunkRelationType
    ) -> None:
        """Remove an edge in the direction it was written. Idempotent.

        Direction matters and is not symmetrized here. For
        :attr:`~manicule.core.organization.ChunkRelationType.PARENT` the two directions mean
        opposite things, and a delete that quietly removed both would take out the edge the
        caller meant to keep.
        """
        async with self._sessions.begin() as session:
            visible = await self._visible_chunk_ids(session, (source_chunk_id, target_chunk_id))
            if source_chunk_id not in visible:
                return
            await session.execute(
                delete(models.ChunkRelation).where(
                    models.ChunkRelation.source_chunk_id == source_chunk_id,
                    models.ChunkRelation.target_chunk_id == target_chunk_id,
                    models.ChunkRelation.relation_type == relation_type.value,
                )
            )

    async def related(
        self, chunk_id: str, *, types: AbstractSet[ChunkRelationType] = frozenset()
    ) -> Sequence[ChunkEdge]:
        """Every edge touching this chunk, from either end.

        ``WHERE source = ? OR target = ?`` is the predicate the schema's second index exists
        for, and it is written that way rather than as two queries unioned so the planner sees
        one statement.

        An empty ``types`` restricts nothing, following the same convention as
        :class:`~manicule.core.retrieval.Filter`: one spelling of "no restriction", so a caller
        that computed an empty set gets every edge rather than none of them.

        Edges whose far end is not visible to this handle are dropped — another workspace's
        chunk, or one belonging to a soft-deleted document. The first cannot be written through
        :meth:`relate` and would have to have arrived by hand; the second happens in the
        ordinary course of events, and surfacing it would let a deleted document reach a
        reader through a neighbor that is still live.
        """
        async with self._sessions() as session:
            if chunk_id not in await self._visible_chunk_ids(session, (chunk_id,)):
                return []
            statement = select(models.ChunkRelation).where(
                or_(
                    models.ChunkRelation.source_chunk_id == chunk_id,
                    models.ChunkRelation.target_chunk_id == chunk_id,
                )
            )
            if types:
                statement = statement.where(
                    models.ChunkRelation.relation_type.in_(
                        sorted(relation.value for relation in types)
                    )
                )
            rows = (
                (
                    await session.execute(
                        statement.order_by(
                            models.ChunkRelation.relation_type,
                            models.ChunkRelation.source_chunk_id,
                            models.ChunkRelation.target_chunk_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            far_ends = {
                row.target_chunk_id if row.source_chunk_id == chunk_id else row.source_chunk_id
                for row in rows
            }
            visible = await self._visible_chunk_ids(session, sorted(far_ends))
            return [
                ChunkEdge(
                    source_chunk_id=row.source_chunk_id,
                    target_chunk_id=row.target_chunk_id,
                    relation_type=ChunkRelationType(row.relation_type),
                )
                for row in rows
                if (row.target_chunk_id if row.source_chunk_id == chunk_id else row.source_chunk_id)
                in visible
            ]

    # --- internals --------------------------------------------------------------------------

    async def _visible_chunk_ids(self, session: AsyncSession, chunk_ids: Sequence[str]) -> set[str]:
        """Those of ``chunk_ids`` belonging to a live document in this workspace.

        Joined to ``documents`` because ``chunks`` carries no workspace of its own, which is
        the same reason :meth:`~manicule.storage.docstore.SqliteDocStore.document_chunks`
        joins.
        """
        wanted = list(dict.fromkeys(chunk_ids))
        if not wanted:
            return set()
        return set(
            (
                await session.execute(
                    select(models.Chunk.id)
                    .join(models.Document, models.Document.id == models.Chunk.document_id)
                    .where(
                        models.Chunk.id.in_(wanted),
                        models.Document.workspace_id == self._workspace_id,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

    async def _require_visible_chunks(
        self, session: AsyncSession, chunk_ids: Sequence[str]
    ) -> None:
        wanted = list(dict.fromkeys(chunk_ids))
        visible = await self._visible_chunk_ids(session, wanted)
        missing = [chunk_id for chunk_id in wanted if chunk_id not in visible]
        if missing:
            named = ", ".join(repr(chunk_id) for chunk_id in missing)
            msg = (
                f"no live chunk {named} in workspace {self._workspace_id!r}. An edge is only "
                f"written when both ends are visible to this handle: a relation reaching "
                f"outside it would return another workspace's chunk id to a lookup from "
                f"inside."
            )
            raise UnknownEntityError(msg)


__all__ = ["RelationsMixin"]
