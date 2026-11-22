"""Promote a unique title match supported by the dense leader."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from manicule.core.content import DocumentStatus
from manicule.core.readers import admits
from manicule.retrieval.prefilter import join_filter
from manicule.retrieval.trace import record

if TYPE_CHECKING:
    from manicule.core.content import Document
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Candidate, Query


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _matches(text: str, document: Document) -> bool:
    title = _normalized(document.title)
    if text == title:
        return True
    if not document.uri.lower().startswith("file:"):
        return False
    try:
        filename = PurePosixPath(unquote(urlsplit(document.uri).path))
    except ValueError:
        return False
    return _normalized(filename.name) == title and _normalized(filename.stem) == text


async def promote(
    store: DocStore,
    query: Query,
    ordered: list[Candidate],
    *,
    require_dense_leader: bool,
) -> list[Candidate]:
    """Move one exact-title match, keeping the fused scores and the rest of the order."""
    text = _normalized(query.text)
    dense = [candidate for candidate in ordered if "dense" in candidate.scores]
    if not text or not dense:
        return ordered
    leader = None
    if require_dense_leader:
        leader = min(dense, key=lambda c: (-c.scores["dense"], c.chunk.id))
        if leader.chunk.document_id == ordered[0].chunk.document_id:
            return ordered

    scope = join_filter(query.filter)
    ids = frozenset(candidate.chunk.document_id for candidate in ordered)
    if scope.document_ids:
        ids &= scope.document_ids
    if not ids or (leader is not None and leader.chunk.document_id not in ids):
        return ordered
    scoped = scope.model_copy(update={"document_ids": ids})
    documents = await store.list_documents(scoped, limit=len(ids))
    publications = {(c.chunk.document_id, c.publication_id) for c in ordered}
    matches = {
        document.id: document.publication_id
        for document in documents
        if document.id in ids
        and document.status is DocumentStatus.INDEXED
        and admits(scope.readers, document.access)
        and (document.id, document.publication_id) in publications
        and _matches(text, document)
    }
    if (
        len(matches) != 1
        or ordered[0].chunk.document_id in matches
        or (leader is not None and matches.get(leader.chunk.document_id) != leader.publication_id)
    ):
        return ordered
    for i, candidate in enumerate(ordered):
        if (
            "dense" in candidate.scores
            and matches.get(candidate.chunk.document_id) == candidate.publication_id
        ):
            return [candidate, *ordered[:i], *ordered[i + 1 :]]
    return ordered


class TitleMatchStage:
    """Reorder one hit using its stored title and dense evidence."""

    def __init__(self, *, docstore: DocStore, name: str = "title_match") -> None:
        self.name = name
        self._docstore = docstore

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        ordered = await promote(self._docstore, query, candidates, require_dense_leader=True)
        record(promoted=ordered is not candidates)
        return ordered


__all__ = ["TitleMatchStage", "promote"]
