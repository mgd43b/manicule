"""Answers about the index, for the queries the router keeps away from the corpus.

**A route nothing returns is not a route.** Each :class:`~manicule.retrieval.router.UtilityKind`
here names a handler that exists, and :func:`handlers_for` offers only the kinds the store in
front of it can actually answer — so a store without a document count does not advertise one,
and the phrase that would have asked for it falls through to retrieval instead of reaching a
handler that is not there.

A utility handler *does* read a store. That is the answer being computed, not the route being
chosen: the router itself stays a pure function over the query text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.content import Metadata
from manicule.retrieval.ports import SupportsDocumentCount, SupportsIndexState
from manicule.retrieval.router import UtilityKind

if TYPE_CHECKING:
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Query

LIST_LIMIT = 50
"""How many documents a listing shows. A bound, because "list documents" on a real corpus is
not a request for a hundred thousand rows."""


class UtilityAnswer(BaseModel):
    """A direct answer about the index, carrying no citations by construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: UtilityKind
    text: str = Field(description="A sentence a person can read.")
    data: Metadata = Field(
        default_factory=dict, description="The same answer, for a caller that will render it."
    )


class UtilityHandler(Protocol):
    """Computes one kind of direct answer."""

    async def __call__(self, query: Query) -> UtilityAnswer: ...


def handlers_for(docstore: DocStore) -> dict[UtilityKind, UtilityHandler]:
    """The utility kinds this store can answer.

    Structural checks rather than a fixed list, so that a store which grows the capability
    gains the route without anything here changing, and one that lacks it never advertises a
    route with nothing behind it.
    """
    handlers: dict[UtilityKind, UtilityHandler] = {
        UtilityKind.DOCUMENT_LIST: _DocumentList(docstore)
    }
    if isinstance(docstore, SupportsDocumentCount):
        handlers[UtilityKind.DOCUMENT_COUNT] = _DocumentCount(docstore)
    if isinstance(docstore, SupportsIndexState):
        handlers[UtilityKind.INDEX_STATUS] = _IndexStatus(docstore)
    return handlers


class _DocumentList:
    def __init__(self, docstore: DocStore) -> None:
        self._docstore = docstore

    async def __call__(self, query: Query) -> UtilityAnswer:
        documents = await self._docstore.list_documents(query.filter, limit=LIST_LIMIT)
        titles = [document.title or document.uri for document in documents]
        return UtilityAnswer(
            kind=UtilityKind.DOCUMENT_LIST,
            text=(
                f"{len(titles)} document(s): {', '.join(titles)}"
                if titles
                else "no documents are indexed in this workspace"
            ),
            data={"documents": [{"id": d.id, "title": d.title, "uri": d.uri} for d in documents]},
        )


class _DocumentCount:
    def __init__(self, docstore: SupportsDocumentCount) -> None:
        self._docstore = docstore

    async def __call__(self, query: Query) -> UtilityAnswer:
        del query  # the store is bound to one workspace, which is the whole scope here
        total = await self._docstore.count_documents()
        return UtilityAnswer(
            kind=UtilityKind.DOCUMENT_COUNT,
            text=f"{total} document(s) are indexed in this workspace",
            data={"documents": total},
        )


class _IndexStatus:
    def __init__(self, docstore: SupportsIndexState) -> None:
        self._docstore = docstore

    async def __call__(self, query: Query) -> UtilityAnswer:
        del query
        state = await self._docstore.index_fingerprints()
        chunks = await self._docstore.count_chunks()
        embedder = state.embed.describe() if state.embed else "nothing yet"
        return UtilityAnswer(
            kind=UtilityKind.INDEX_STATUS,
            text=f"{chunks} chunk(s) indexed, embedded by {embedder}",
            data={
                "chunks": chunks,
                "embedder": embedder,
                "vector_table": state.vector_table,
            },
        )


__all__ = ["LIST_LIMIT", "UtilityAnswer", "UtilityHandler", "handlers_for"]
