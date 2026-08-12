"""What reaches a caller: the expansion, its source, and a conflict nobody resolved.

The rule these exist for is ``bugs/bug2.md`` §3's last bullet — *never present an expansion
without citation provenance*. It is the easiest rule in the whole feature to break, because
breaking it means writing one field fewer, and the result still looks like a working answer.

So the source is resolved through the **scoped** store rather than copied off the entry, and an
entry whose document this workspace cannot see is dropped rather than shown with a blank source.
The tests below were written after a mutation showed that neither behaviour was covered at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.app.service import ApplicationService
from manicule.core.glossary import (
    DefinitionForm,
    ExpansionConflict,
    GlossaryEntry,
    GlossaryMatch,
    MatchReason,
    QueryExpansion,
)
from tests.app.fakes import FakeBackend, make_document

if TYPE_CHECKING:
    from manicule.app import results as r

WORKSPACE = "default"
EXPANSION = "Network Operations Workspace"
OTHER = "Nightly Operations Watchdog"


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


def an_entry(
    document_id: str,
    *,
    expansion: str = EXPANSION,
    chunk_id: str = "chunk-1",
) -> GlossaryEntry:
    return GlossaryEntry(
        acronym="NOW",
        display="NOW",
        expansion=expansion,
        document_id=document_id,
        chunk_id=chunk_id,
        location="Glossary of terms",
        form=DefinitionForm.EM_DASH,
        confidence=0.95,
        aliases=(),
    )


def a_match(entry: GlossaryEntry) -> QueryExpansion:
    return QueryExpansion(
        original="What is NOW?",
        expanded=f"What is {EXPANSION}?",
        matches=(
            GlossaryMatch(surface="NOW", key="NOW", reason=MatchReason.EXACT_CASE, entry=entry),
        ),
    )


def _stock(backend: FakeBackend) -> str:
    """Put one visible document in the store and return its id."""
    document = make_document(WORKSPACE, source_id="glossary", title="Glossary of terms")
    backend.store.documents[document.id] = document
    backend.retriever_.candidates = []
    return document.id


async def _search(backend: FakeBackend) -> r.SearchResult:
    return await ApplicationService(backend).search("What is NOW?")


async def test_a_search_reports_the_expansion_with_the_document_it_came_from(
    backend: FakeBackend,
) -> None:
    document_id = _stock(backend)
    backend.retriever_.expansion = a_match(an_entry(document_id))

    found = await _search(backend)

    assert len(found.expansions) == 1
    reported = found.expansions[0]
    assert reported.expansion == EXPANSION
    assert reported.display == "NOW"
    assert reported.document_id == document_id
    assert reported.chunk_id == "chunk-1"
    assert reported.uri, "an expansion whose source has no uri is one nobody can open"
    assert reported.title == "Glossary of terms"
    assert reported.reason == MatchReason.EXACT_CASE.value
    assert found.expanded_query == f"What is {EXPANSION}?"
    assert found.query == "What is NOW?", "the original is what the caller asked"


async def test_an_expansion_whose_document_this_workspace_cannot_see_is_not_reported(
    backend: FakeBackend,
) -> None:
    """The guard the mutation exposed. Dropped, never rendered with an empty source.

    The entry came from a workspace-scoped lookup, so this can only fire on a store that
    leaked or on a document deleted between the lookup and the render. Either way the answer
    is the same, and it is not "show it with the source left blank".
    """
    _stock(backend)
    backend.retriever_.expansion = a_match(an_entry("a-document-nobody-can-see"))

    found = await _search(backend)

    assert found.expansions == ()


async def test_a_conflict_reaches_the_caller_with_every_candidate_and_no_winner(
    backend: FakeBackend,
) -> None:
    """A conflict a reader cannot go and look at is a warning they cannot act on."""
    first = make_document(WORKSPACE, source_id="left", title="Left glossary")
    second = make_document(WORKSPACE, source_id="right", title="Right glossary")
    backend.store.documents[first.id] = first
    backend.store.documents[second.id] = second
    backend.retriever_.expansion = QueryExpansion(
        original="What is NOW?",
        conflicts=(
            ExpansionConflict(
                key="NOW",
                surface="NOW",
                entries=(
                    an_entry(first.id, chunk_id="chunk-a"),
                    an_entry(second.id, expansion=OTHER, chunk_id="chunk-b"),
                ),
            ),
        ),
    )

    found = await _search(backend)

    assert found.expansions == (), "a conflicting term expands to nothing"
    assert found.expanded_query == ""
    assert len(found.conflicts) == 1
    assert {candidate.expansion for candidate in found.conflicts[0].candidates} == {
        EXPANSION,
        OTHER,
    }
    assert {candidate.title for candidate in found.conflicts[0].candidates} == {
        "Left glossary",
        "Right glossary",
    }


async def test_a_conflict_whose_candidates_no_longer_resolve_is_not_reported(
    backend: FakeBackend,
) -> None:
    """One readable definition is a definition, not a disagreement.

    Reporting it as a conflict would tell a reader two documents disagree while naming only
    one of them, which is worse than saying nothing.
    """
    only = make_document(WORKSPACE, source_id="left", title="Left glossary")
    backend.store.documents[only.id] = only
    backend.retriever_.expansion = QueryExpansion(
        original="What is NOW?",
        conflicts=(
            ExpansionConflict(
                key="NOW",
                surface="NOW",
                entries=(
                    an_entry(only.id, chunk_id="chunk-a"),
                    an_entry("gone", expansion=OTHER, chunk_id="chunk-b"),
                ),
            ),
        ),
    )

    found = await _search(backend)

    assert found.conflicts == ()


async def test_a_query_that_named_no_term_reports_nothing_at_all(
    backend: FakeBackend,
) -> None:
    _stock(backend)
    backend.retriever_.expansion = QueryExpansion(original="What is NOW?")

    found = await _search(backend)

    assert found.expansions == ()
    assert found.conflicts == ()
    assert found.expanded_query == ""


async def test_a_directly_routed_query_carries_no_expansion_object_at_all(
    backend: FakeBackend,
) -> None:
    """``None`` means no lookup happened, exactly as it does for ``confidence``."""
    _stock(backend)
    backend.retriever_.expansion = None

    found = await _search(backend)

    assert found.expansions == ()
    assert found.expanded_query == ""


async def test_an_answer_discloses_the_words_the_search_actually_ran_on(
    backend: FakeBackend,
) -> None:
    """The reader is entitled to know the search used words they did not type.

    Carried on ``ask`` as well as ``search``, because an answer built from a context reached
    through an expansion is the case where it matters most and is least visible.
    """
    document = make_document(WORKSPACE, source_id="glossary", title="Glossary of terms")
    backend.store.documents[document.id] = document
    backend.retriever_.candidates = []
    backend.retriever_.expansion = a_match(an_entry(document.id))

    answer = await ApplicationService(backend).ask("What is NOW?")

    assert [item.expansion for item in answer.expansions] == [EXPANSION]
    assert answer.expansions[0].document_id == document.id
    assert answer.expansions[0].title == "Glossary of terms"
