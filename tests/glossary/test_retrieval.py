"""The regression cases ``bugs/bug2.md`` lists, run end to end through the shipped retriever.

**The first test is the one that matters.** It demonstrates the baseline failing on this
fixture, and every other test here is only worth reading because that one passes: a corpus where
the definition already ranks well cannot show that anything was fixed, which is precisely how an
earlier attempt at this feature could not justify itself.

The embedder is :class:`~tests.evaluation.fakes.BagOfWordsEmbedder`, unchanged and not written
for this suite. It buries the definition for the same reason the real one does — the glossary
page is one chunk of twenty-five entries, so the words of any single definition are a fortieth
of its vector, while fifteen short passages *using* the acronym are almost entirely about it.
``test_measured.py`` runs the same fixture through BGE-M3 and reports the same failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.content import BlockKind
from manicule.core.glossary import MatchReason
from manicule.core.retrieval import ConfidenceBand, Query
from manicule.retrieval.cache import L1QueryCache
from manicule.retrieval.confidence import DEFINITION_CITED, NOTHING_RESEMBLES
from manicule.retrieval.expansion import GLOSSARY_SCORE_KEY, ExpansionPolicy
from tests.evaluation.fakes import BagOfWordsEmbedder
from tests.glossary import corpus, system

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk
    from manicule.retrieval.retriever import RetrievalResult, Retriever
    from manicule.storage.docstore import SqliteDocStore

LIMIT = 10
OTHER = "Nightly Operations Watchdog"


@pytest.fixture
async def indexed(store: SqliteDocStore) -> list[Chunk]:
    return await system.build_corpus(store)


@pytest.fixture
async def definition(store: SqliteDocStore, indexed: list[Chunk]) -> str:
    """The chunk id of the passage that defines the acronym under test."""
    entries = await store.glossary_entries(indexed[0].document_id)
    return next(entry.chunk_id for entry in entries if entry.acronym == corpus.ACRONYM)


async def _baseline(store: SqliteDocStore, chunks: Sequence[Chunk]) -> Retriever:
    """The identical pipeline with no glossary wired. One thing differs, and it is the feature."""
    return await system.retriever_over(store, BagOfWordsEmbedder(), chunks, glossary=False)


async def _with_glossary(
    store: SqliteDocStore,
    chunks: Sequence[Chunk],
    *,
    policy: ExpansionPolicy | None = None,
    cache: L1QueryCache | None = None,
) -> Retriever:
    return await system.retriever_over(
        store, BagOfWordsEmbedder(), chunks, policy=policy, cache=cache
    )


def _ask(text: str, **fields: object) -> Query:
    """One query at this suite's limit, with any extra filter fields a case needs.

    ``**fields`` is untyped for the same reason ``Filter`` itself has to be constructed that
    way here: the field names vary per test and pydantic validates the values, so a signature
    enumerating them would be a second copy of ``Filter`` kept in step by hand.
    """
    return Query(text=text, limit=LIMIT, filter=system.query_filter(**fields))  # pyright: ignore[reportArgumentType]


def _texts(result: RetrievalResult) -> list[str]:
    return [passage.chunk.id for passage in result.context.passages]


# --- the proof --------------------------------------------------------------------------------


async def test_the_baseline_does_not_return_the_definition_at_all(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """**The fixture reproduces the bug.** Without this, nothing below means anything.

    ``What is NOW?`` retrieves ten passages and the definition is not among them. Every passage
    that does reach the context merely *uses* the acronym, so an answer built from this context
    can describe how the term is used and cannot say what it stands for.
    """
    baseline = await _baseline(store, indexed)

    result = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert len(result.context.passages) == LIMIT, "the corpus is big enough to fill a context"
    assert system.rank_of(result.context.passages, definition) is None, (
        "the fixture does not reproduce the bug: the definition already ranks inside the "
        "context, so promoting it could not demonstrate an improvement"
    )


@pytest.mark.parametrize(
    "text",
    [corpus.QUERY_ACRONYM, corpus.QUERY_LOWER, corpus.QUERY_MIXED_CASE, corpus.QUERY_PUNCTUATED],
    ids=["upper", "lower", "mixed-case", "punctuated"],
)
async def test_the_definition_leads_the_results_once_the_glossary_is_wired(
    store: SqliteDocStore, indexed: list[Chunk], definition: str, text: str
) -> None:
    """``bugs/bug2.md`` acceptance criterion 1, over four spellings of one question.

    Rank 1 rather than "somewhere in the top ten", because an exact alias hit is a *lookup*.
    Its position should not depend on a cosine, which is the whole difference between this and
    a better ranking function.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(text))

    assert system.rank_of(result.context.passages, definition) == 1


async def test_an_expansion_is_never_presented_without_its_source(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """``bugs/bug2.md`` §3, last bullet, and acceptance criterion 3.

    Claim-level provenance: the expansion names the document, the chunk and the location it was
    read out of, and the chunk it names is the one that was promoted.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert result.expansion is not None
    match = result.expansion.matches[0]
    assert match.entry.expansion == corpus.EXPANSION
    assert match.entry.chunk_id == definition
    assert match.entry.document_id == indexed[0].document_id
    assert match.entry.location, "an expansion with no location is one nobody can look up"
    assert match.reason is MatchReason.EXACT_CASE


async def test_the_run_says_which_alias_fired_and_what_it_cost(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    report = result.trace.glossary
    assert report is not None
    assert report.consulted
    assert report.terms == (corpus.ACRONYM,)
    assert report.reasons == (MatchReason.EXACT_CASE.value,)
    assert report.expanded_query == f"What is {corpus.EXPANSION}?"
    assert report.promoted == 1
    assert report.second_pass, "the cost of the feature is on the record, not inferred"


async def test_a_promoted_passage_is_marked_as_authoritative_without_being_rescored(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """The mark goes beside the legs' scores, never into the slot fusion and confidence read.

    A detection confidence sitting where a cosine belongs would make a passage detected at 0.95
    look like a passage matched at 0.95, which is a different claim about a different thing.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    promoted = result.context.passages[0]
    assert promoted.chunk.id == definition
    assert promoted.scores[GLOSSARY_SCORE_KEY] == pytest.approx(0.95)
    assert promoted.score != promoted.scores[GLOSSARY_SCORE_KEY]


async def test_two_terms_defined_in_one_chunk_promote_one_passage(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """A glossary page states dozens of definitions in one chunk.

    So a query naming two of them resolves both matches to the same passage. Promoting it twice
    would report two promotions for one passage that moved, count one fetch as two, and hand
    the merge a duplicate it only has to collapse again. Found by re-reading the diff rather
    than by a failing test, which is why it now has one.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask("What are NOW and QUARTZ?"))

    assert result.expansion is not None
    assert len(result.expansion.matches) == 2, "both terms resolved"
    assert {match.entry.chunk_id for match in result.expansion.matches} == {definition}
    assert result.trace.glossary is not None
    assert result.trace.glossary.promoted == 1, "one passage moved, so one promotion"
    assert result.trace.glossary.promoted_from_store <= 1
    assert [passage.chunk.id for passage in result.context.passages].count(definition) == 1


# --- definitions carrying descriptions, and stylized spellings ---------------------------------


async def test_a_definition_with_a_description_is_retrieved_by_its_core_expansion(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The motivating example, end to end: ``What is NOVA?``.

    Before extraction learned where a description begins, this question had nothing to find.
    The line is thirteen words on the right of the dash, so no entry was written, so no alias
    could fire, so no passage was promoted — the failure is upstream of retrieval entirely and
    no amount of ranking work reaches it.

    What is asserted is the *expansion*, not merely that something was retrieved: an entry
    carrying the whole right-hand side would also promote the right passage, and would then
    splice nine words of prose into the second query and show the reader a description where a
    term belongs.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_DESCRIBED))

    assert result.expansion is not None
    assert result.expansion.matches, f"no glossary entry fired for {corpus.QUERY_DESCRIBED!r}"
    match = result.expansion.matches[0]
    assert match.entry.acronym == corpus.DESCRIBED_ACRONYM
    assert match.entry.expansion == corpus.DESCRIBED_EXPANSION
    assert system.rank_of(result.context.passages, match.entry.chunk_id) == 1


async def test_the_promoted_passage_still_carries_the_description_it_was_trimmed_of(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 1's second half: the entry is trimmed, the evidence is not.

    The reader is shown four words as the expansion and can follow the citation to a passage
    that states the whole thing. Trimming the *passage* to match the entry would be the version
    of this feature that quietly discards corpus text, and the citation would then resolve to
    something the document does not contain.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_DESCRIBED))

    assert result.expansion is not None
    match = result.expansion.matches[0]
    passage = next(
        candidate
        for candidate in result.context.passages
        if candidate.chunk.id == match.entry.chunk_id
    )
    assert corpus.DESCRIBED_ENTRY in passage.chunk.text
    assert "a service used to correlate operational signals" in passage.chunk.text
    assert match.entry.expansion not in {corpus.DESCRIBED_ENTRY, passage.chunk.text}


@pytest.mark.parametrize(
    "text",
    [corpus.QUERY_STYLIZED, corpus.QUERY_STYLIZED_LOWER],
    ids=["as-written", "lower-case"],
)
async def test_a_stylized_term_resolves_through_its_normalised_key(
    store: SqliteDocStore, indexed: list[Chunk], text: str
) -> None:
    """Requirement 4, end to end and through storage rather than in the detector alone.

    Both spellings reach the same entry: the key is normalised, so ``relay`` finds it, and the
    display is not, so what comes back is the document's own ``ReLAY``. The lower-case form also
    proves the key survived a database round trip — a store that normalised on write and read
    back differently would fail here and nowhere else.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(text))

    assert result.expansion is not None
    assert result.expansion.matches, f"no entry fired for {text!r}"
    entry = result.expansion.matches[0].entry
    assert entry.acronym == corpus.STYLIZED_ACRONYM
    assert entry.display == corpus.STYLIZED_DISPLAY
    assert entry.expansion == corpus.STYLIZED_EXPANSION
    assert system.rank_of(result.context.passages, entry.chunk_id) == 1


@pytest.mark.parametrize(
    "term",
    ["NOTE", "TODAY", "API"],
    ids=["note", "today", "api"],
)
async def test_prose_on_a_glossary_page_produces_no_entry_to_promote(
    store: SqliteDocStore, indexed: list[Chunk], term: str
) -> None:
    """The negatives through the real ingest path, on a real glossary page.

    ``system.index`` calls the shipped ``detect_entries``, so what this asserts is that nothing
    was ever written — not that something written was later filtered out. A term with no entry
    cannot be expanded, cannot be promoted and cannot be classified as an explicit definition,
    so one refusal at ingest closes all three doors at once.

    See ``corpus.PROSE_ON_THE_GLOSSARY_PAGE``: this is only a test because those lines sit under
    a title that supplies the evidence carrying them to exactly the threshold.
    """
    entries = await system.entries_by_acronym(store, indexed)

    assert term not in entries


async def test_the_glossary_pages_still_yield_the_definitions_they_state(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The other side of the refusal, so a rule that rejected everything could not pass.

    ``test_prose_on_a_glossary_page_produces_no_entry_to_promote`` is satisfied by a detector
    that writes nothing at all. This is the assertion that stops that from being a green suite:
    every line that *is* a definition is still one, across both pages.
    """
    entries = await system.entries_by_acronym(store, indexed)

    expected = len(corpus.GLOSSARY_ENTRIES) + len(corpus.SUPPLEMENT_ENTRIES)
    assert len(entries) == expected, "a definition was lost, or a line of prose was admitted"
    assert {corpus.ACRONYM, corpus.DESCRIBED_ACRONYM, corpus.STYLIZED_ACRONYM} <= set(entries)


# --- the queries that must not change ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [corpus.QUERY_ORDINARY_USE, corpus.QUERY_ABSENT, corpus.QUERY_FULL],
    ids=["ordinary-use", "nonexistent-acronym", "full-expansion"],
)
async def test_a_query_that_names_no_term_retrieves_exactly_what_it_did_before(
    store: SqliteDocStore, indexed: list[Chunk], text: str
) -> None:
    """``bugs/bug2.md`` acceptance criterion 4, and the reason this feature is safe to leave on.

    The ranking is compared **passage by passage**, not by a summary statistic. A test asserting
    only that the definition did not reach rank 1 would pass while the feature quietly reordered
    everything else.
    """
    baseline = await _baseline(store, indexed)
    retriever = await _with_glossary(store, indexed)

    before = await baseline.retrieve(_ask(text))
    after = await retriever.retrieve(_ask(text))

    assert _texts(after) == _texts(before)
    assert after.expansion is not None
    assert not after.expansion.fired
    assert after.trace.glossary is not None
    assert not after.trace.glossary.second_pass, "nothing fired, so nothing was searched twice"


async def test_switching_expansion_off_reproduces_the_baseline_exactly(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """``bugs/bug2.md`` §4: allow entity expansion to be disabled.

    Compared against the baseline rather than merely asserted to be "unchanged", because the
    claim being made is that the switch removes the feature rather than merely quietening it.
    """
    baseline = await _baseline(store, indexed)
    disabled = await _with_glossary(store, indexed, policy=ExpansionPolicy(enabled=False))

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await disabled.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert _texts(after) == _texts(before)
    assert after.expansion is not None
    assert not after.expansion.fired
    assert after.trace.glossary is not None
    assert not after.trace.glossary.consulted


async def test_the_ranking_is_the_same_on_every_run(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Stable ranking across repeated runs, which the spec lists as a regression case.

    Three runs rather than two: a merge whose order came from a set would often agree twice by
    chance, and the fold this replaced did exactly that.
    """
    retriever = await _with_glossary(store, indexed)
    query = _ask(corpus.QUERY_ACRONYM)

    runs = [_texts(await retriever.retrieve(query)) for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]


# --- conflicts, end to end ------------------------------------------------------------------------


async def test_a_conflicting_definition_is_surfaced_and_nothing_is_expanded(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """``bugs/bug2.md`` acceptance criterion 5, through the whole retriever.

    A second glossary arrives disagreeing with the first. The ranking falls back to what the
    baseline produced — no promotion, no second search — and the disagreement is reported with
    both sources.
    """
    await system.index(store, "second", "Team glossary", [f"NOW — {OTHER}"])
    chunks = await store.document_chunks(indexed[0].document_id)
    baseline = await _baseline(store, [*indexed, *await _second_chunks(store)])
    retriever = await _with_glossary(store, [*indexed, *await _second_chunks(store)])
    del chunks

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert after.expansion is not None
    assert not after.expansion.fired
    assert set(after.expansion.conflicts[0].expansions) == {corpus.EXPANSION, OTHER}
    assert after.trace.glossary is not None
    assert after.trace.glossary.conflicts == (corpus.ACRONYM,)
    assert after.trace.glossary.promoted == 0
    assert _texts(after) == _texts(before), "a conflict changes nothing about the ranking"


async def _second_chunks(store: SqliteDocStore) -> list[Chunk]:
    from manicule.core.ids import document_id  # noqa: PLC0415 - one helper needs it

    return list(await store.document_chunks(document_id(system.WORKSPACE, "fixture", "second")))


async def test_a_collection_scoped_query_consults_only_that_collections_glossary(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The spec's regression case for conflicting expansions across collections, end to end.

    Store-level isolation is asserted in ``test_storage.py``; this is the same claim through
    the whole retriever, which became testable when collection-scoped search landed. It also
    pins the *order* the retriever resolves things in: membership becomes document ids first,
    and the glossary lookup then sees the same ids the legs will. Resolving the glossary
    separately would be a second notion of what a collection contains, free to drift.
    """
    left, left_chunks = await system.index(
        store, "left", "Left glossary", [f"NOW — {corpus.EXPANSION}"]
    )
    right, _ = await system.index(store, "right", "Right glossary", [f"NOW — {OTHER}"])
    inner = await store.create_collection("inner")
    outer = await store.create_collection("outer")
    await store.add_to_collection(inner.id, [left.id])
    await store.add_to_collection(outer.id, [right.id])
    everything = [*indexed, *left_chunks, *await store.document_chunks(right.id)]
    retriever = await _with_glossary(store, everything)

    scoped_in = await retriever.retrieve(
        _ask(corpus.QUERY_ACRONYM, collection_ids=frozenset({inner.id}))
    )
    scoped_out = await retriever.retrieve(
        _ask(corpus.QUERY_ACRONYM, collection_ids=frozenset({outer.id}))
    )
    unscoped = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert scoped_in.expansion is not None
    assert scoped_in.expansion.matches[0].entry.expansion == corpus.EXPANSION
    assert scoped_in.expansion.conflicts == ()
    assert scoped_out.expansion is not None
    assert scoped_out.expansion.matches[0].entry.expansion == OTHER
    assert scoped_out.expansion.conflicts == ()
    assert unscoped.expansion is not None
    assert not unscoped.expansion.fired, "the workspace holds both, so neither is chosen"
    assert set(unscoped.expansion.conflicts[0].expansions) >= {corpus.EXPANSION, OTHER}


# --- the lookup path, which is what makes this more than a boost ------------------------------


async def test_a_definition_neither_search_returned_is_still_promoted(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """The case that separates a lookup from a re-ranking.

    With a candidate depth of one, neither the original nor the expanded search can possibly
    return the definition — so a feature that only reordered what search found would return
    nothing here. The entry names the chunk, so it is fetched by id.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(
        Query(text=corpus.QUERY_ACRONYM, limit=1, filter=system.query_filter())
    )

    assert result.trace.glossary is not None
    assert result.trace.glossary.promoted_from_store == 1
    assert system.rank_of(result.context.passages, definition) == 1


async def test_a_fetched_definition_still_obeys_a_chunk_level_restriction(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """The vocabulary lookup ignores ``kinds``; the promotion does not.

    A query restricted to code passages must not be handed a prose glossary page, however
    certain we are that it defines a term the query named. This is where that restriction is
    applied, and removing the check here is the way the feature would become a filter bypass.

    It is also the one query shape that separates "a definition exists" from "a definition is in
    front of the reader": the alias resolves, the entry is found, and the passage is then refused
    by the chunk-level filter. Written after a mutation showed that dropping the in-context
    condition from ``_cites_a_definition`` broke nothing — a result claiming an explicit definition
    while showing a context that does not contain one is the failure this half now catches.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM, kinds=frozenset({BlockKind.CODE})))

    assert result.expansion is not None
    assert result.expansion.fired, "the term still resolved — the restriction is not on vocabulary"
    assert system.rank_of(result.context.passages, definition) is None
    assert result.confidence is not None
    assert not result.confidence.explicit_definition, (
        "the definition was found and then refused, so the result may not claim it was cited"
    )
    assert result.confidence.reason != DEFINITION_CITED


async def test_a_definition_from_a_document_this_query_cannot_see_is_never_promoted(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The tenancy boundary on the fetch-by-id path, against a source that ignores its scope.

    Written after a mutation showed nothing covered it. The cold-fetch path takes a chunk id
    straight out of an entry, so it is the one place in this feature where a chunk can reach a
    caller without having come through a search — and the visibility join is the only thing
    standing there. Demonstrated with :class:`~tests.glossary.test_storage.LeakySource`,
    because a correctly-scoped source cannot produce the entry that tests it.
    """
    from tests.glossary.test_storage import LeakySource  # noqa: PLC0415

    hidden, _ = await system.index(store, "hidden", "Hidden glossary", [f"NOW — {OTHER}"])
    entries = await store.glossary_entries(hidden.id)
    await store.soft_delete_document(hidden.id)

    retriever = await system.retriever_over(
        store, BagOfWordsEmbedder(), indexed, glossary=LeakySource(entries[0])
    )

    result = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert result.expansion is not None
    assert result.expansion.fired, "the leaky source did hand over the entry"
    assert result.trace.glossary is not None
    assert result.trace.glossary.promoted == 0, (
        "a passage from a document this query cannot see reached the ranking"
    )
    assert all(passage.chunk.document_id != hidden.id for passage in result.context.passages)


async def test_the_expanded_ranking_is_interleaved_rather_than_appended(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Both query forms reach the context, which is the point of running the second search.

    Written after a mutation showed nothing covered it. Concatenating the two rankings puts
    every original candidate ahead of every expanded one, so at any realistic limit the second
    search contributes nothing at all and the whole second pass is spent for no effect.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))
    original_only = await (await _baseline(store, indexed)).retrieve(_ask(corpus.QUERY_ACRONYM))

    contributed = set(_texts(result)) - set(_texts(original_only))
    assert len(contributed) > 1, (
        "only the promoted definition is new, so the expanded search reached the context "
        "through nothing but promotion"
    )


async def test_a_cited_definition_is_not_also_reported_as_nothing_resembling(
    store: SqliteDocStore, indexed: list[Chunk], definition: str
) -> None:
    """The contradiction, gone — and **nothing numeric moved to remove it**.

    The baseline reports band ``none`` with "nothing in this corpus resembles your question"
    while the definition of the queried term is one of the passages the corpus holds. With the
    glossary wired the definition is at rank 1 and the sentence is replaced by one that is true.

    Every assertion about the *number* is an equality with the baseline, deliberately. A test
    that allowed the score to rise would pass just as happily for an unconditional boost, which
    is the fix this feature is forbidden to make: a detection confidence is not a cosine. What
    changed is the classification and the sentence, and the band stays ``none`` because the
    dense evidence really is at the corpus's noise level.
    """
    baseline = await _baseline(store, indexed)
    retriever = await _with_glossary(store, indexed)

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert before.confidence is not None
    assert after.confidence is not None
    assert before.confidence.reason == NOTHING_RESEMBLES, (
        "the baseline no longer reports the contradiction, so this fixture proves nothing"
    )
    assert not before.confidence.explicit_definition

    assert system.rank_of(after.context.passages, definition) == 1
    assert after.confidence.explicit_definition
    assert after.confidence.reason == DEFINITION_CITED
    assert after.confidence.reason != NOTHING_RESEMBLES

    assert after.confidence.score == pytest.approx(before.confidence.score)
    assert after.confidence.band is before.confidence.band
    assert after.confidence.components == before.confidence.components
    assert after.confidence.ceiling == before.confidence.ceiling


@pytest.mark.parametrize(
    "text",
    [corpus.QUERY_UNRELATED, corpus.QUERY_ABSENT, corpus.QUERY_ORDINARY_USE],
    ids=["unrelated", "nonexistent-acronym", "ordinary-use"],
)
async def test_a_question_with_no_definition_behind_it_still_reports_no_evidence(
    store: SqliteDocStore, indexed: list[Chunk], text: str
) -> None:
    """Requirement 7, and the half of the classification that has to stay shut.

    ``ordinary-use`` is the one that decides whether the rule is a rule. It names ``now``, which
    the glossary defines, and the corpus is full of passages containing the token — so a
    classification keyed on "a term was mentioned" or on lexical overlap would fire here. It is
    keyed on the query asking what the term *means*, and "should I restart the daemon now" does
    not ask that.
    """
    retriever = await _with_glossary(store, indexed)

    result = await retriever.retrieve(_ask(text))

    assert result.confidence is not None
    assert not result.confidence.explicit_definition
    assert result.confidence.band is ConfidenceBand.NONE
    assert result.confidence.reason == NOTHING_RESEMBLES


async def test_a_mention_of_the_term_is_not_enough_to_claim_a_definition(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The narrowest version of the rule, with every other condition satisfied.

    ``raise a ticket in NOVA today`` names a defined term, expansion fires, the defining passage
    is fetched, and it lands at **rank 1** — the reader is looking at a glossary entry for the
    token they typed. The classification still does not fire, because the remaining condition is
    the one that is false: the question is an instruction, not a request for a meaning.

    **It fires on ``exact_case``, which is the same ``MatchReason`` that ``What is NOVA?``
    records.** So the reason a match was admitted cannot distinguish these two queries, and a
    classification reading it would be true for both. That is why
    :func:`~manicule.retrieval.retriever._cites_a_definition` asks ``definitional_frame`` directly
    rather than inspecting the recorded reason — this assertion is what pins that down.

    This is requirement 7 at its sharpest. "We are showing you a definition" and "you asked what
    this means" are separate facts, and a classification keyed on the first alone would fire on
    every query that merely mentions a defined term.
    """
    retriever = await _with_glossary(store, indexed)
    defining = (await system.entries_by_acronym(store, indexed))[corpus.DESCRIBED_ACRONYM].chunk_id

    result = await retriever.retrieve(_ask(f"raise a ticket in {corpus.DESCRIBED_ACRONYM} today"))

    assert result.expansion is not None
    assert result.expansion.fired, "the fixture must actually promote, or it tests nothing"
    assert result.expansion.matches[0].reason is MatchReason.EXACT_CASE
    assert system.rank_of(result.context.passages, defining) == 1, (
        "the defining passage must be in front of the reader, or the rule is untested"
    )
    assert result.confidence is not None
    assert not result.confidence.explicit_definition


async def test_promotion_never_lowers_the_reported_confidence(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """A feature that can quietly downgrade the answers it does not help cannot be left on.

    A promoted passage carries no leg score, so it contributes nothing to the evidence in
    either direction; what it *can* do is displace a passage out of the context, and this is
    the assertion that says the displacement does not cost anything.
    """
    baseline = await _baseline(store, indexed)
    retriever = await _with_glossary(store, indexed)

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert before.confidence is not None
    assert after.confidence is not None
    assert after.confidence.score >= before.confidence.score


# --- the cache ----------------------------------------------------------------------------------


async def test_an_expanded_run_and_an_unexpanded_one_do_not_share_a_cache_entry(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Two rankings, one query text, one generation counter.

    The counter catches a definition being added, because that is a row. It catches neither
    expansion being switched off nor a second definition turning the term into a conflict —
    both change the ranking and neither changes the corpus.
    """
    cache = L1QueryCache(entries=16)
    expanding = await _with_glossary(store, indexed, cache=cache)
    disabled = await _with_glossary(
        store, indexed, policy=ExpansionPolicy(enabled=False), cache=cache
    )
    query = _ask(corpus.QUERY_ACRONYM)

    expanded = await expanding.retrieve(query)
    plain = await disabled.retrieve(query)

    assert _texts(expanded) != _texts(plain), (
        "one cache entry served both, so the key cannot tell an expanded run from a plain one"
    )


async def test_a_cache_hit_reports_the_same_expansion_a_miss_did(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """A hit is a different run, not a different answer about what the glossary said."""
    retriever = await _with_glossary(store, indexed, cache=L1QueryCache(entries=16))
    query = _ask(corpus.QUERY_ACRONYM)

    miss = await retriever.retrieve(query)
    hit = await retriever.retrieve(query)

    assert hit.trace.cached
    assert miss.expansion is not None
    assert hit.expansion is not None
    assert hit.expansion.matches[0].entry.chunk_id == miss.expansion.matches[0].entry.chunk_id
    assert _texts(hit) == _texts(miss)
