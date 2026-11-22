"""Round 7's promotion uses scoped document titles and leaves the RRF scores intact."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from manicule.core.content import DocumentStatus
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.retrieval.title_match import promote
from tests.storage_helpers import make_chunk, make_document


@pytest.mark.parametrize(
    ("dense_leader", "best_dense", "moved"),
    [
        (False, False, True),
        (True, False, False),
        (True, True, True),
    ],
)
async def test_promotes_one_dense_chunk_and_preserves_the_rest(
    dense_leader: bool, best_dense: bool, moved: bool
) -> None:
    target = make_document(source_id="target", title="My Notes.md", uri="file:///My%20Notes.md")
    other = make_document(source_id="other")
    store = AsyncMock()
    store.list_documents.return_value = [target, other]
    query = Query(text="  MY   notes ", filter=Filter(workspace_ids=frozenset({"w"})))
    ordered = [
        Candidate(chunk=make_chunk(other, 0, "first"), score=0.03, scores={"dense": 0.8}),
        Candidate(chunk=make_chunk(target, 0, "lexical"), score=0.02, scores={"lexical": 2.0}),
        Candidate(
            chunk=make_chunk(target, 1, "dense"),
            score=0.01,
            scores={"dense": 0.9 if best_dense else 0.7},
        ),
    ]
    result = await promote(store, query, ordered, require_dense_leader=dense_leader)
    if moved:
        assert result == [ordered[2], ordered[0], ordered[1]]
    else:
        assert result is ordered
    scope = query.filter.model_copy(update={"document_ids": frozenset({target.id, other.id})})
    if dense_leader and not best_dense:
        store.list_documents.assert_not_awaited()
    else:
        store.list_documents.assert_awaited_once_with(scope, limit=2)


async def test_ambiguous_titles_and_lexical_only_hits_do_not_promote() -> None:
    first = make_document(source_id="first", title="other")
    file = make_document(source_id="file", title="Notes.md", uri="file:///a/Notes.md")
    page = make_document(source_id="page", title="notes", uri="https://example.org/notes")
    store = AsyncMock()
    store.list_documents.return_value = [first, file, page]
    query = Query(text="notes", filter=Filter(workspace_ids=frozenset({"w"})))
    ordered = [
        Candidate(chunk=make_chunk(document, 0, "passage"), score=1, scores=scores)
        for document, scores in [
            (first, {"dense": 0.9}),
            (file, {"lexical": 10.0}),
            (page, {"dense": 0.8}),
        ]
    ]
    assert await promote(store, query, ordered, require_dense_leader=False) is ordered
    store.list_documents.return_value = [
        first,
        file,
        page.model_copy(update={"status": DocumentStatus.FAILED}),
    ]
    assert await promote(store, query, ordered, require_dense_leader=False) is ordered


async def test_a_title_from_a_new_publication_does_not_promote_a_stale_chunk() -> None:
    first = make_document(source_id="first", title="Other")
    page = make_document(source_id="page", title="Notes").model_copy(
        update={"publication_id": "new-publication"}
    )
    store = AsyncMock()
    store.list_documents.return_value = [first, page]
    query = Query(text="notes", filter=Filter(workspace_ids=frozenset({"w"})))
    ordered = [
        Candidate(chunk=make_chunk(first, 0, "other"), score=0.03, scores={"dense": 0.8}),
        Candidate(chunk=make_chunk(page, 0, "stale"), score=0.02, scores={"dense": 0.9}),
    ]

    assert await promote(store, query, ordered, require_dense_leader=True) is ordered
