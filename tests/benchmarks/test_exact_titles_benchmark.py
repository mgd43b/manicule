"""Round 7's promotion uses scoped document titles and leaves the RRF scores intact."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from manicule.core.content import DocumentStatus
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.title_match import TitleMatchStage, promote
from tests.benchmarks import exact_titles, query_shapes
from tests.benchmarks.exact_titles import CANDIDATES
from tests.benchmarks.query_shapes import ModelVerdicts, load_model_verdicts
from tests.benchmarks.thread_duplication import for_question
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path


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


async def test_expanded_query_promotion_is_counted_under_the_original_question() -> None:
    target = make_document(source_id="target", title="Access Tokens")
    other = make_document(source_id="other")
    store = AsyncMock()
    store.list_documents.return_value = [target, other]
    candidates = [
        Candidate(chunk=make_chunk(target, 0, "target"), score=0.9, scores={"dense": 0.9}),
        Candidate(
            chunk=make_chunk(other, 0, "other"),
            score=0.8,
            scores={"dense": 0.8, "lexical": 1.0},
        ),
    ]
    rewritten = Query(text="Access Tokens", filter=Filter(workspace_ids=frozenset({"w"})))
    selected: set[str] = set()
    stage = RRFStage()
    with CANDIDATES["dense-leader-title"](store, selected):
        ordered = await for_question("heading-original", stage.run(rewritten, candidates))
    assert ordered[0].chunk.document_id == target.id
    assert selected == {"heading-original"}


async def test_the_title_stage_returns_a_fresh_list_on_a_no_op() -> None:
    scope = Filter(workspace_ids=frozenset({"w"}))
    empty: list[Candidate] = []
    stage = TitleMatchStage(docstore=AsyncMock())
    output = await stage.run(Query(text="notes", filter=scope), empty)
    assert output == []
    assert output is not empty


def test_choosing_needs_two_distinct_required_models(tmp_path: Path) -> None:
    models = frozenset({"BAAI/bge-m3", "qwen3-embedding:0.6b"})
    files: list[Path] = []
    for i, model in enumerate(sorted(models)):
        path = tmp_path / f"verdict-{i}.json"
        payload = ModelVerdicts(model=model, verdicts={"dense-leader-title": (True, 0.0)})
        path.write_text(payload.model_dump_json(), encoding="utf-8")
        files.append(path)
    first, second = files
    assert len(load_model_verdicts([first, second], models)) == 2
    with pytest.raises(ValueError, match="one verdict file per model"):
        load_model_verdicts([first, first], models)


def test_round_seven_measures_mail_with_conversation_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    monkeypatch.setattr(query_shapes, "run_cli", run)
    exact_titles.main(["--docs", "docs"])
    assert run.call_args.kwargs.get("mail_pipeline") == "hybrid+collapse"
