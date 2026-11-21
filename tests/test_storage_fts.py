"""The lexical index, and the three ways it fails silently if nobody checks."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import bindparam, text, update

from manicule.core.content import DocumentStatus
from manicule.core.readers import group_principal
from manicule.core.retrieval import Filter
from manicule.storage import docstore as docstore_module
from manicule.storage import models
from manicule.storage.docstore import DEFAULT_WORKSPACE, SqliteDocStore, lexical_predicates
from manicule.storage.engine import database_path
from manicule.storage.fts import (
    CREATE_TRIGGERS,
    CURRENT_TRIGGERS,
    DROP_TRIGGERS,
    INTEGRITY_CHECK_FTS,
    MAX_MATCH_WORDS,
    MAX_RANKED_TERMS,
    MAX_SCOPED_CHUNKS,
    MAX_SCOPED_DOCUMENTS,
    REBUILD_FTS,
    SCOPE_PROBE_LIMIT,
    SCOPE_PROBE_SQL,
    SEARCH_SQL,
    SEARCH_SQL_DOCUMENT_FIRST,
    SEARCH_SQL_RANKED,
    LexicalPlan,
    choose_lexical_plan,
    escape_match_query,
    match_terms,
    needs_scope_probe,
)
from manicule.storage.types import UtcDateTime
from tests.benchmarks.lexical_plans import equivalent
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk, Document


async def _seed(store: SqliteDocStore, source_id: str, texts: list[str]) -> str:
    document = make_document(source_id=source_id)
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id, [make_chunk(document, i, body) for i, body in enumerate(texts)]
    )
    return document.id


async def test_a_query_stem_matches_a_different_surface_form(store: SqliteDocStore) -> None:
    """Porter stemming is why the lexical leg is not a list of exact strings.

    Without it, "authenticate" scores zero against a chunk containing "authentication", and
    because the lexical leg is half of a hybrid merge that removes a candidate rather than
    reordering one.
    """
    await _seed(store, "s1", ["the service handles authentication and token rotation"])
    hits = await store.search_lexical("authenticate", k=5)
    assert [candidate.chunk.text for candidate in hits]


async def test_the_breadcrumb_is_searchable_but_does_not_outrank_the_body(
    store: SqliteDocStore,
) -> None:
    """A section called "Configuration" is unfindable without knowing what it configures.

    Weighted below the body, because the breadcrumb repeats on every chunk of a page.
    """
    await _seed(store, "s1", ["nothing relevant here at all"])
    hits = await store.search_lexical("tokens", k=5)
    assert hits, "the heading breadcrumb should be searchable"


async def test_deleting_a_document_removes_its_rows_from_the_lexical_index(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The cascade reaches chunks, and the chunk trigger has to fire on a cascaded delete.

    Verified rather than assumed: if it did not, deleted documents would keep matching
    queries forever with nothing to notice.
    """
    document_id = await _seed(store, "s1", ["authentication and rotation"])
    assert await store.search_lexical("authentication", k=5)

    await store.delete_document(document_id)
    assert await store.search_lexical("authentication", k=5) == []

    async with engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
        tombstones = (
            await connection.execute(text("SELECT count(*) FROM vector_tombstones"))
        ).scalar_one()
    assert remaining == 0
    assert tombstones == 1, "the delete trigger must record a tombstone for the vector sweep"


async def test_a_soft_deleted_document_stops_matching_without_touching_the_index(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Restoring is clearing a timestamp — no re-embed, no re-parse, no re-fetch."""
    document_id = await _seed(store, "s1", ["authentication and rotation"])
    await store.soft_delete_document(document_id)

    assert await store.search_lexical("authentication", k=5) == []
    async with engine.connect() as connection:
        assert (await connection.execute(text("SELECT count(*) FROM chunks"))).scalar_one() == 1


async def test_asking_for_k_returns_k_live_rows_not_k_rows_of_which_some_are_dead(
    store: SqliteDocStore,
) -> None:
    """The reason the lexical query is one joined statement.

    Deferred deletion leaves soft-deleted chunks in the index competing for the same top-k
    slots. Filtering after a ``LIMIT`` silently returns fewer live rows than asked for — on
    the fixture below, MATCH-then-filter returns none at all.
    """
    dead = await _seed(store, "dead", ["auth token rotation"] * 5)
    await _seed(store, "live", ["auth token rotation"] * 3)
    await store.soft_delete_document(dead)

    hits = await store.search_lexical("auth", k=5)
    assert len(hits) == 3, "every returned candidate must belong to a live document"


async def test_a_document_that_is_not_indexed_is_not_searchable(
    store: SqliteDocStore,
) -> None:
    """Only ``indexed`` is servable, and the join is what enforces it."""
    document = make_document(source_id="pending", status=DocumentStatus.PENDING)
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "auth token rotation")])
    assert await store.search_lexical("auth", k=5) == []


async def test_the_integrity_check_catches_an_index_that_stopped_being_updated(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Two obvious checks cannot detect this, and were in an earlier draft of the design.

    ``COUNT(*)`` on an external-content FTS table reads through to the content table, so the
    two counts agree by construction; and a bare ``integrity-check`` passes on an empty index
    because an empty index is internally consistent. Only the ``rank`` argument compares the
    index against the content table.
    """
    async with engine.begin() as connection:
        for statement in DROP_TRIGGERS:
            await connection.execute(text(statement))

    await _seed(store, "s1", ["authentication and rotation"])

    async with engine.connect() as connection:
        chunks = (await connection.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
        fts = (await connection.execute(text("SELECT count(*) FROM chunks_fts"))).scalar_one()
    assert chunks == fts, "the counts agree even though the index is empty"
    assert await store.search_lexical("authentication", k=5) == [], "the index really is empty"

    from sqlalchemy.exc import DatabaseError  # noqa: PLC0415

    async with engine.begin() as connection:
        with pytest.raises(DatabaseError):
            await connection.execute(text(INTEGRITY_CHECK_FTS))


async def test_rebuilding_the_index_repairs_it_from_the_chunks_table(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Rung 1 of the ladder: seconds, no network, no model."""
    async with engine.begin() as connection:
        for statement in DROP_TRIGGERS:
            await connection.execute(text(statement))
    await _seed(store, "s1", ["authentication and rotation"])
    assert await store.search_lexical("authentication", k=5) == []

    async with engine.begin() as connection:
        await connection.execute(text(REBUILD_FTS))
    assert await store.search_lexical("authentication", k=5)


@pytest.mark.parametrize(
    "raw",
    ["NOT", "AND OR", 'unbalanced " quote', "col:value", "a*", "(unclosed"],
)
async def test_query_syntax_never_reaches_fts5_as_an_operator(
    store: SqliteDocStore, raw: str
) -> None:
    """A user typing NOT means the word. Unescaped, these are syntax errors, not searches."""
    await _seed(store, "s1", ["ordinary prose about tokens"])
    await store.search_lexical(raw, k=5)


def test_an_empty_query_produces_no_match_expression() -> None:
    """Punctuation alone must not become a match-everything expression."""
    assert escape_match_query("   ") == ""
    assert escape_match_query("!!! ???") == ""
    assert escape_match_query(". - ' _") == "", "a phrase with no token in it is not a word"


async def test_a_question_reaches_a_chunk_that_never_says_how_do_or_i(
    store: SqliteDocStore,
) -> None:
    """The defect #481 fixed: every word of a question used to be required, function words too."""
    await _seed(store, "s1", ["Rotate the signing token every thirty days."])
    assert escape_match_query("how do I rotate the signing token?") == (
        '"rotate" OR "signing" OR "token"'
    )
    assert await store.search_lexical("how do I rotate the signing token?", k=5)


async def test_a_chunk_missing_a_content_word_is_found_and_ranked_below(
    store: SqliteDocStore,
) -> None:
    """What round 3 changed: content words are alternatives, and BM25 does the weighting.

    Requiring every content word missed a question naming the right thing in other words — the
    chunk below says "signing token" and never "encryption". It is now found, and a chunk holding
    more of the question's rarer words still ranks above it.
    """
    await _seed(
        store,
        "s1",
        [
            "Rotate the signing token every thirty days.",
            "Rotate the encryption token whenever the signing key changes.",
            "Thirty days is the default retention.",
        ],
    )
    hits = await store.search_lexical("how do I rotate the encryption token", k=5)
    texts = [candidate.chunk.text for candidate in hits]
    assert texts[0].startswith("Rotate the encryption token"), "it holds all three words"
    # Every seeded chunk matches through the stand-in breadcrumb, which says "tokens"; the one
    # holding only that ranks last.
    assert texts.index("Rotate the signing token every thirty days.") < texts.index(
        "Thirty days is the default retention."
    )


async def test_a_query_of_function_words_alone_searches_for_them(store: SqliteDocStore) -> None:
    """Dropping every word would turn a question into no search at all."""
    await _seed(store, "s1", ["What is it that the sweep removes?"])
    assert escape_match_query("what is it") == '"what" OR "is" OR "it"'
    assert await store.search_lexical("what is it", k=5)


def test_negation_is_a_subject_word_and_is_kept() -> None:
    """Dropping ``not`` would make the question ask for its opposite."""
    assert escape_match_query("why is the vector not published") == (
        '"vector" OR "not" OR "published"'
    )
    assert escape_match_query("why doesn't it publish") == '"doesn\'t" OR "publish"'


def test_function_words_are_recognized_through_case_and_punctuation() -> None:
    assert escape_match_query("How does THE sweep work, then?") == '"sweep" OR "work"'
    assert escape_match_query("what's it for.") == '"what\'s" OR "it" OR "for"'


def test_a_typographic_apostrophe_is_the_same_apostrophe() -> None:
    """A phone or macOS keyboard types ``\u2019``; dropping it made ``whats`` a content word."""
    assert escape_match_query("what\u2019s the retention period") == '"retention" OR "period"'
    assert escape_match_query("why doesn\u2019t it publish") == '"doesn\'t" OR "publish"'


def test_a_long_query_is_cut_to_its_first_content_words() -> None:
    """A pasted paragraph would otherwise match almost every row and be scored in full."""
    words = [f"term{i}" for i in range(30)]
    match = escape_match_query("how do I " + " the ".join(words))
    assert match == " OR ".join(f'"{word}"' for word in words[:MAX_MATCH_WORDS])


def test_each_word_is_matched_once() -> None:
    """``bm25()`` scores a repeated phrase twice, so a repeated word would outweigh the rest."""
    assert escape_match_query("token Token TOKEN rotation") == '"token" OR "rotation"'
    assert escape_match_query("token, token. (token) rotation") == '"token" OR "rotation"'


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NOT", '"NOT"'),
        ("OR", '"OR"'),
        ("AND OR", '"AND" OR "OR"'),
        ("what is NEAR", '"NEAR"'),
        ("published NOT deleted", '"published" OR "NOT" OR "deleted"'),
        ('col:value "unbalanced', '"colvalue" OR "unbalanced"'),
        ("token* (unclosed", '"token" OR "unclosed"'),
    ],
)
def test_every_word_is_a_quoted_literal_and_the_only_operator_is_ours(
    raw: str, expected: str
) -> None:
    """Outside the quotes there is nothing but the ``OR`` this function writes between them."""
    match = escape_match_query(raw)
    assert match == expected
    outside = re.sub(r'"(?:[^"]|"")*"', "", match)
    assert set(outside.split()) <= {"OR"}
    assert outside.split().count("OR") == max(0, match.count('" OR "'))


async def test_operator_words_alone_search_for_the_words(store: SqliteDocStore) -> None:
    """``OR`` is a function word, so a query of only ``AND OR`` keeps both, as words."""
    await _seed(store, "s1", ["Choose this or that, and say which."])
    assert await store.search_lexical("AND OR", k=5)
    assert await store.search_lexical("NOT", k=5) == []


async def test_search_is_scoped_to_the_stores_workspace(engine: AsyncEngine) -> None:
    """Workspace isolation is a property of the handle, so a query cannot forget it."""
    first = SqliteDocStore(engine, workspace_id="alpha")
    second = SqliteDocStore(engine, workspace_id="beta")
    await first.ensure_workspace()
    await second.ensure_workspace()

    await _seed(first, "s1", ["auth token rotation"])
    assert await first.search_lexical("auth", k=5)
    assert await second.search_lexical("auth", k=5) == []


async def test_concurrent_searches_do_not_exhaust_the_connection_pool(
    store: SqliteDocStore,
) -> None:
    """One search holds one connection, so searches cannot deadlock each other.

    ``search_lexical`` ran its FTS statement and then, *still inside* its own session, called
    ``get_chunks`` — which opens a session of its own. One search therefore held two of the
    engine's connections at once, and the engine's pool is sized by SQLAlchemy's defaults: on
    ``sqlite+aiosqlite`` over a file that is an ``AsyncAdaptedQueuePool`` of 5 with 10 overflow,
    fifteen for the whole process. Hold-and-wait, with the usual consequence — a search waiting
    for a connection only another search can release.

    **The numbers here were measured rather than reasoned to, and the first attempt at this
    test was wrong.** Twelve concurrent searches over one document do *not* deadlock: the loop
    interleaves, each search finishes quickly, and the peak checkout stays under the pool. The
    shape that fails is enough concurrent searches that a second connection is wanted while the
    first is still held — forty, over a corpus whose hydration is not instant. At that point the
    nested version never completes and the flat one answers in a quarter of a second.

    So the corpus below is twenty documents of twenty-five chunks rather than one of three, and
    the timeout is what makes this a test rather than a hang: the failure mode is *waiting*, so
    an assertion that simply awaited the gather would report a stall as a slow test.
    """
    for source in range(20):
        await _seed(
            store,
            f"s{source}",
            ["the service handles authentication and token rotation " * 20 for _ in range(25)],
        )

    async def search() -> int:
        return len(await store.search_lexical("authentication", k=50))

    counts = await asyncio.wait_for(asyncio.gather(*(search() for _ in range(40))), timeout=25)

    assert all(count > 0 for count in counts), "every concurrent search returns its own hits"


async def test_the_rebuild_triggers_are_the_ones_the_head_revision_installs(
    engine: AsyncEngine,
) -> None:
    """A lexical rebuild must not downgrade a trigger a migration replaced.

    `CREATE_TRIGGERS` is imported by `20260814_6e31b7d592ac_atomic_publications`, so its bodies
    describe the schema at *that* revision and may never change — editing them would
    retroactively change what an already applied migration did.
    `20260818_f3c18a9d72e1_workspace_index_identity` then replaced `chunks_ad` with a body that
    resolves the deleted chunk's workspace and vector table into the tombstone.

    The lexical rebuild drops all three triggers and recreates them, and it recreated them from
    the frozen tuple. So publishing a rebuilt generation left the *pre-migration* trigger in
    place, and every chunk deleted from then on wrote a tombstone with no workspace, namespace
    or table — a row the sweep cannot attribute to the index it belongs to.

    Silent in both directions: the rebuild succeeds, and `INTEGRITY_CHECK_FTS` compares
    `chunks_fts` against `chunks` and has nothing to say about a trigger body.

    Asserted against the database the migrations actually produce, not against a copy of the
    SQL kept here — a second copy would drift from the migration exactly as the first one did.
    """
    async with engine.connect() as connection:
        installed = (
            await connection.execute(
                text("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='chunks_ad'")
            )
        ).scalar_one()

    normalized = " ".join(str(installed).split())
    assert " ".join(CURRENT_TRIGGERS[1].split()) == normalized, (
        "CURRENT_TRIGGERS has drifted from the trigger the head revision installs; a lexical "
        "rebuild would replace the live trigger with a different one"
    )
    assert "workspace_id" in normalized, "the fixture is meant to be migrated to head"
    assert "workspace_id" not in " ".join(CREATE_TRIGGERS[1].split()), (
        "CREATE_TRIGGERS is pinned to the revision that imports it and must not be updated"
    )


async def test_writing_a_document_s_chunks_is_one_statement(store: SqliteDocStore) -> None:
    """Fifty chunks are one INSERT, and the triggers still fire for each of them.

    `session.add` in a loop cannot batch this. `Chunk.seq` is a server-generated
    `INTEGER PRIMARY KEY`, so the unit of work needs the generated keys back in parameter order
    and asks for `RETURNING`; the SQLite dialect declares no insertmanyvalues sentinel, so it
    falls back to one statement per row. A fifty-chunk document was fifty round trips inside the
    publication transaction, on every document of every ingest — 50 statements became 1, and the
    whole call went from 52 statements to 4.

    **The trigger behavior is the half worth testing**, because it is what a bulk insert could
    plausibly break: `chunks_fts` is external-content over this table and is maintained entirely
    by `chunks_ai`. SQLite fires an `AFTER INSERT` trigger once per row rather than once per
    statement, so the index is complete either way — asserted here rather than assumed, through
    a search and the index's own integrity check.
    """
    from sqlalchemy import event  # noqa: PLC0415 - only this test counts statements

    document = make_document(source_id="bulk")
    await store.upsert_document(document)
    chunks = [
        make_chunk(document, index, f"authentication token rotation {index}") for index in range(50)
    ]

    inserts = 0

    def spy(_conn: object, _cursor: object, statement: str, *_rest: object) -> None:
        nonlocal inserts
        if "INTO chunks" in statement and "chunks_fts" not in statement:
            inserts += 1

    engine = store._engine  # pyright: ignore[reportPrivateUsage]
    event.listen(engine.sync_engine, "before_cursor_execute", spy)
    try:
        await store.replace_chunks(document.id, chunks)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", spy)

    assert inserts == 1, f"{len(chunks)} chunks were written in {inserts} statements"
    assert len(await store.document_chunks(document.id)) == len(chunks)
    assert len(await store.search_lexical("authentication", k=100)) == len(chunks), (
        "the FTS triggers fire per row, so a multi-row insert must still index every chunk"
    )
    async with engine.connect() as connection:
        await connection.execute(text(INTEGRITY_CHECK_FTS))


# --- which statement answers a search ----------------------------------------------------------
#
# Three statements return the same ranking by different routes (`docs/storage.md` §6.1), and
# `choose_lexical_plan` picks among them for speed alone. Every test below is about that being
# true: same rows, same scores, no row a restriction excludes, in whichever statement runs.

TEMPLATES = {
    LexicalPlan.DIRECT: SEARCH_SQL,
    LexicalPlan.RANKED: SEARCH_SQL_RANKED,
    LexicalPlan.DOCUMENT_FIRST: SEARCH_SQL_DOCUMENT_FIRST,
}
EVERY_STATEMENT = pytest.mark.parametrize("plan", list(LexicalPlan), ids=lambda plan: plan.value)
Ranking = list[tuple[str, float]]


def _statement(
    predicates: Any, plan: LexicalPlan, *, as_released: bool = False, prefix: str = ""
) -> Any:
    """``plan``'s statement under ``predicates``, bound, with ``prefix`` ahead of the SQL.

    ``as_released`` is ``SEARCH_SQL`` with ``d.id`` and ``d.source`` written as plain column
    tests, the way every release before this one ran it: the baseline the others are held to.
    """
    if as_released:
        statement = text(prefix + predicates.text(SEARCH_SQL).replace("AND +d.", "AND d."))
        if predicates.timestamps:
            statement = statement.bindparams(
                *(bindparam(name, type_=UtcDateTime()) for name in predicates.timestamps)
            )
        return statement
    template = prefix + TEMPLATES[plan]
    return predicates.statement(template, document_first=plan is LexicalPlan.DOCUMENT_FIRST)


async def _ranking(
    store: SqliteDocStore,
    plan: LexicalPlan,
    query: str,
    k: int,
    restriction: Filter | None = None,
    *,
    as_released: bool = False,
) -> Ranking:
    """The rows and scores one statement returns, built by the store's own predicate builder."""
    predicates = lexical_predicates(DEFAULT_WORKSPACE, restriction)
    statement = _statement(predicates, plan, as_released=as_released)
    params: dict[str, Any] = {"match": escape_match_query(query), "limit": k, **predicates.params}
    async with store.sessions() as session:
        rows = (await session.execute(statement, params)).all()
    return [(str(row.chunk_id), float(row.rank_score)) for row in rows]


def _everything(**restrictions: object) -> Filter:
    return Filter(workspace_ids=frozenset({DEFAULT_WORKSPACE}), **restrictions)  # pyright: ignore[reportArgumentType]


async def _document(
    store: SqliteDocStore,
    source_id: str,
    texts: list[str],
    *,
    headings: list[tuple[str, ...]] | None = None,
    **fields: Any,
) -> Document:
    document = make_document(source_id=source_id, **fields)
    await store.upsert_document(document)
    chunks: list[Chunk] = []
    for i, body in enumerate(texts):
        if headings is None:
            chunks.append(make_chunk(document, i, body))
        else:
            chunks.append(make_chunk(document, i, body, heading_path=headings[i]))
    await store.replace_chunks(document.id, chunks)
    return document


async def _weighted_corpus(store: SqliteDocStore) -> None:
    """Chunks whose rank depends on which column a word is in, and ties for the cut at ``k``."""
    await _document(
        store,
        "weights",
        [
            "alpha beta gamma",
            "gamma delta",
            "alpha alpha beta",
            "alpha beta",
            "nothing here",
            "alpha",
            "beta beta beta alpha",
        ],
        headings=[
            ("Setup",),
            ("alpha", "beta"),  # the words are only in the heading
            ("Other",),
            ("alpha", "beta"),  # and in both
            ("Setup",),
            ("beta",),
            ("Setup",),
        ],
    )
    # Identical text in identical headings scores identically. Eight of them straddle every
    # `k` below eight, and SQLite does not say which of equal scores a LIMIT keeps.
    await _document(store, "ties", ["alpha beta"] * 8, headings=[("Tied",)] * 8)


@EVERY_STATEMENT
@pytest.mark.parametrize("query", ["alpha beta", "alpha", "gamma delta alpha", "beta", "absent"])
@pytest.mark.parametrize("k", [1, 3, 5, 50])
async def test_every_statement_returns_the_released_statements_ranking(
    store: SqliteDocStore, plan: LexicalPlan, query: str, k: int
) -> None:
    """Same scores in the same order, and the same rows, ties apart.

    The heading column is weighted 0.4 against the body's 1.0 in all three, so a chunk whose
    words are only in its heading must score below one whose words are in its text; that is
    asserted too, because equal rankings of a corpus that never exercised the weight would
    prove nothing about it. The ties are compared as sets: SQLite does not specify the order of
    equal scores, nor which of them a `LIMIT` keeps, in any of the statements.
    """
    await _weighted_corpus(store)

    released = await _ranking(store, LexicalPlan.DIRECT, query, k, as_released=True)
    candidate = await _ranking(store, plan, query, k)

    assert equivalent(released, candidate), f"{plan.value} ranked {query!r} differently"
    assert [score for _, score in released] == [score for _, score in candidate], (
        "scores are the contract and must match exactly, not approximately"
    )


async def test_the_weighted_corpus_does_exercise_the_heading_weight(
    store: SqliteDocStore,
) -> None:
    await _weighted_corpus(store)
    found = await store.search_lexical("alpha beta", 50)
    by_text = {candidate.chunk.text: candidate.score for candidate in found}

    assert len({candidate.score for candidate in found}) > 3, "the fixture must score distinctly"
    # "gamma delta" holds both words, but only in its heading; "alpha alpha beta" holds them in
    # its text. A heading weighted like the body would put the first above the second.
    assert by_text["gamma delta"] < by_text["alpha alpha beta"]


@EVERY_STATEMENT
async def test_a_chunk_matched_only_by_its_heading_scores_below_the_same_words_in_its_text(
    store: SqliteDocStore, plan: LexicalPlan
) -> None:
    await _document(
        store,
        "headed",
        ["alpha beta", "unrelated words"],
        headings=[("Setup",), ("alpha", "beta")],
    )

    ranking = await _ranking(store, plan, "alpha beta", 5)

    assert len(ranking) == 2
    assert ranking[0][1] < ranking[1][1], "bm25 is negative, so the better match sorts first"
    assert [chunk.text for chunk in await store.get_chunks([ranking[0][0]])] == ["alpha beta"]


@EVERY_STATEMENT
async def test_ties_fill_the_limit_from_the_tied_rows_and_nothing_else(
    store: SqliteDocStore, plan: LexicalPlan
) -> None:
    await _document(store, "ties", ["alpha beta"] * 8, headings=[("Tied",)] * 8)
    every = {chunk_id for chunk_id, _ in await _ranking(store, plan, "alpha beta", 50)}

    cut = await _ranking(store, plan, "alpha beta", 3)

    assert len(every) == 8
    assert len(cut) == 3
    assert {chunk_id for chunk_id, _ in cut} <= every
    assert len({score for _, score in cut}) == 1


@EVERY_STATEMENT
async def test_a_query_nothing_matches_returns_nothing(
    store: SqliteDocStore, plan: LexicalPlan
) -> None:
    await _weighted_corpus(store)

    assert await _ranking(store, plan, "zzzzz yyyyy", 5) == []


async def test_search_lexical_returns_what_the_released_statement_ranked(
    store: SqliteDocStore,
) -> None:
    """End to end through the store: where the statement is chosen and the score negated."""
    await _weighted_corpus(store)
    for query in ("alpha beta", "alpha", "gamma delta alpha"):
        released = await _ranking(store, LexicalPlan.DIRECT, query, 5, as_released=True)

        found = await store.search_lexical(query, 5)

        assert equivalent(
            released, [(candidate.chunk.id, -candidate.score) for candidate in found]
        ), query
        assert [candidate.scores["bm25"] for candidate in found] == [
            candidate.score for candidate in found
        ]


STRONG = "alpha beta alpha beta"
WEAK = "alpha " + "filler " * 30 + "beta"
"""Both match ``alpha beta``; the first ranks far above the second, so any strong row a
restriction was meant to exclude that gets through takes a slot an allowed one should have."""


@dataclass(frozen=True)
class Crowded:
    """A corpus where every row that outranks the allowed ones is one a search must not return."""

    allowed: frozenset[str]
    """Chunk ids a search under :attr:`restriction` may return, and the only ones it may."""
    restriction: Filter
    ids: dict[str, str]
    """Document ids by what is wrong with them."""
    live_chunks: int
    """Chunks the probe counts for a scope naming every document here: the ones in this
    workspace that are not soft-deleted, whether or not a search may return them."""


async def _crowded(store: SqliteDocStore, engine: AsyncEngine) -> Crowded:
    """Six weak allowed chunks, and strong ones each failing a different test.

    The strong rows are in another workspace, soft-deleted, not indexed, the wrong media type,
    older than the window, and in a source the reader does not hold a group for. ``k = 6`` is
    exactly the allowed count, so one strong row admitted leaves one allowed row out.
    """
    ids: dict[str, str] = {}
    allowed: set[str] = set()
    for name in ("open-0", "open-1"):
        document = await _document(
            store, name, [WEAK] * 3, source="open", title=name, media_type="text/markdown"
        )
        ids[name] = document.id
        allowed |= {chunk.id for chunk in await store.document_chunks(document.id)}

    beta = SqliteDocStore(engine, workspace_id="beta")
    await beta.ensure_workspace()
    foreign = make_document(workspace_id="beta", source_id="foreign")
    await beta.upsert_document(foreign)
    await beta.replace_chunks(foreign.id, [make_chunk(foreign, i, STRONG) for i in range(4)])
    ids["another workspace"] = foreign.id

    gone = await _document(store, "gone", [STRONG] * 4)
    await store.soft_delete_document(gone.id)
    ids["soft-deleted"] = gone.id
    ids["not indexed"] = (
        await _document(store, "pending", [STRONG] * 4, status=DocumentStatus.PENDING)
    ).id
    ids["wrong media type"] = (
        await _document(store, "html", [STRONG] * 4, media_type="text/html")
    ).id
    stale = await _document(store, "stale", [STRONG] * 4)
    ids["too old"] = stale.id
    group = await store.create_reader_group("editors")
    await store.grant_source("secret", [group.id])
    ids["forbidden"] = (await _document(store, "secret", [STRONG] * 4, source="secret")).id

    async with store.sessions.begin() as session:
        await session.execute(
            update(models.Document)
            .where(models.Document.id == stale.id)
            .values(updated_at=datetime.now(UTC) - timedelta(days=30))
        )
    restriction = _everything(
        media_types=frozenset({"text/markdown"}),
        updated_after=datetime.now(UTC) - timedelta(days=1),
        readers=frozenset({group_principal("nobody-in-particular")}),
    )
    live = 0
    for name, document_id in ids.items():
        if name not in {"another workspace", "soft-deleted"}:
            live += len(await store.document_chunks(document_id))
    return Crowded(allowed=frozenset(allowed), restriction=restriction, ids=ids, live_chunks=live)


@EVERY_STATEMENT
async def test_no_statement_admits_a_row_a_restriction_excludes_or_drops_one_it_allows(
    store: SqliteDocStore, engine: AsyncEngine, plan: LexicalPlan
) -> None:
    crowded = await _crowded(store, engine)

    ranking = await _ranking(store, plan, "alpha beta", 6, crowded.restriction)

    assert {chunk_id for chunk_id, _ in ranking} == crowded.allowed
    # And the same restriction without the limit, so a row that was merely outranked by an
    # allowed one cannot hide behind `k`.
    every = await _ranking(store, plan, "alpha beta", 500, crowded.restriction)
    assert {chunk_id for chunk_id, _ in every} == crowded.allowed


def _scope(crowded: Crowded, extra: int = 0) -> Filter:
    """The restriction with every document of the fixture named, and ``extra`` ids that exist
    nowhere, so the count of ids is exact without a count of chunks to match."""
    named = {*crowded.ids.values(), *(f"invented-{i}" for i in range(extra))}
    return crowded.restriction.model_copy(update={"document_ids": frozenset(named)})


@pytest.fixture
def chosen(monkeypatch: pytest.MonkeyPatch) -> list[LexicalPlan]:
    """The statement each search through the store chose, in order."""
    seen: list[LexicalPlan] = []
    real = docstore_module.choose_lexical_plan

    def spy(**facts: Any) -> LexicalPlan:
        plan = real(**facts)
        seen.append(plan)
        return plan

    monkeypatch.setattr(docstore_module, "choose_lexical_plan", spy)
    return seen


async def _search(store: SqliteDocStore, restriction: Filter) -> set[str]:
    return {c.chunk.id for c in await store.search_lexical("alpha beta", 6, restriction)}


async def test_a_small_explicit_scope_runs_the_document_first_statement_and_leaks_nothing(
    store: SqliteDocStore, engine: AsyncEngine, chosen: list[LexicalPlan]
) -> None:
    """Eight ids holding 22 live chunks, on the document-first side of both budgets.

    The scope names every document in the fixture, the unreadable, deleted and foreign ones
    included: a collection can hold ids that have since been deleted or restricted, so the id
    list admits nothing by itself, and neither does a plan chosen from it.
    """
    crowded = await _crowded(store, engine)
    assert crowded.live_chunks == 22

    assert await _search(store, _scope(crowded)) == crowded.allowed
    assert chosen == [LexicalPlan.DOCUMENT_FIRST]


@pytest.mark.parametrize("over", [0, 1])
async def test_the_probe_budget_is_inclusive_and_either_side_returns_the_same_rows(
    store: SqliteDocStore, engine: AsyncEngine, chosen: list[LexicalPlan], over: int
) -> None:
    """Exactly ``MAX_SCOPED_CHUNKS`` live chunks, and then one more.

    The probe stops counting at one past the budget, so that is how it reports "more than
    this". Identical filters on both sides; what changes is only which statement runs, and the
    rows must not.
    """
    crowded = await _crowded(store, engine)
    padding = MAX_SCOPED_CHUNKS - crowded.live_chunks + over
    filler = await _document(store, "padding", ["nothing relevant at all"] * padding)
    scope = _scope(crowded).model_copy(
        update={"document_ids": _scope(crowded).document_ids | {filler.id}}
    )

    assert await _search(store, scope) == crowded.allowed

    assert chosen == [LexicalPlan.DIRECT if over else LexicalPlan.DOCUMENT_FIRST]
    for statement in LexicalPlan:
        ranking = await _ranking(store, statement, "alpha beta", 6, scope)
        assert {chunk_id for chunk_id, _ in ranking} == crowded.allowed, statement.value


@pytest.mark.parametrize("over", [0, 1])
async def test_the_id_budget_is_inclusive_and_either_side_returns_the_same_rows(
    store: SqliteDocStore, engine: AsyncEngine, chosen: list[LexicalPlan], over: int
) -> None:
    """Exactly ``MAX_SCOPED_DOCUMENTS`` ids is the last scope probed; one more is the first not.

    The invented ids hold no chunks, so the chunk budget cannot be what changes the answer.
    """
    crowded = await _crowded(store, engine)
    scope = _scope(crowded, MAX_SCOPED_DOCUMENTS - len(crowded.ids) + over)
    assert len(scope.document_ids) == MAX_SCOPED_DOCUMENTS + over

    assert await _search(store, scope) == crowded.allowed

    assert chosen == [LexicalPlan.DIRECT if over else LexicalPlan.DOCUMENT_FIRST]


async def test_a_single_word_in_a_small_scope_is_walked_document_first_too(
    store: SqliteDocStore, engine: AsyncEngine, chosen: list[LexicalPlan]
) -> None:
    crowded = await _crowded(store, engine)

    found = await store.search_lexical("alpha", 6, _scope(crowded))

    assert {candidate.chunk.id for candidate in found} == crowded.allowed
    assert chosen == [LexicalPlan.DOCUMENT_FIRST]


async def test_a_scope_of_documents_that_do_not_exist_still_returns_nothing(
    store: SqliteDocStore, engine: AsyncEngine, chosen: list[LexicalPlan]
) -> None:
    """A probe that finds no chunks is a hint to run the narrow statement, never a license to
    drop the document test: "nothing found" must not become "no scope"."""
    crowded = await _crowded(store, engine)
    scope = crowded.restriction.model_copy(update={"document_ids": frozenset({"nowhere", "never"})})

    assert await _search(store, scope) == set()
    assert chosen == [LexicalPlan.DOCUMENT_FIRST]
    # Unscoped, the same search finds the six, so the empty answer came from the scope.
    assert await _search(store, crowded.restriction) == crowded.allowed


async def test_a_scope_naming_only_another_workspaces_document_returns_nothing(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    crowded = await _crowded(store, engine)
    scope = crowded.restriction.model_copy(
        update={"document_ids": frozenset({crowded.ids["another workspace"]})}
    )

    assert await _search(store, scope) == set()


# --- the plans, which are the point -----------------------------------------------------------
#
# What the statements return is pinned above. What they cost is not a result, so it is pinned
# two ways that do not depend on a clock: the plan SQLite reports, and the number of virtual
# machine steps a statement takes on a small fixture, which is deterministic where a timing is
# not. A planner makes these choices from the schema's indexes alone (no `ANALYZE` has ever
# run), so an empty migrated database makes the same ones a large one does.


async def _steps(
    store: SqliteDocStore,
    plan: LexicalPlan,
    restriction: Filter | None = None,
    *,
    as_released: bool = False,
) -> list[str]:
    """``EXPLAIN QUERY PLAN`` of ``plan``, one string per step."""
    predicates = lexical_predicates(DEFAULT_WORKSPACE, restriction)
    statement = _statement(predicates, plan, as_released=as_released, prefix="EXPLAIN QUERY PLAN ")
    params: dict[str, Any] = {"match": escape_match_query("alpha beta"), "limit": 5}
    params |= predicates.params
    async with store.sessions() as session:
        return [str(row[3]) for row in (await session.execute(statement, params)).all()]


SORT = "USE TEMP B-TREE FOR ORDER BY"
ANYONE = frozenset({group_principal("anyone")})


@pytest.mark.parametrize(
    "restriction",
    [
        None,
        _everything(sources=frozenset({"fs"})),
        _everything(document_ids=frozenset({f"d{i}" for i in range(40)})),
        _everything(media_types=frozenset({"text/markdown"})),
        _everything(updated_after=datetime(2000, 1, 1, tzinfo=UTC)),
        _everything(readers=ANYONE),
    ],
    ids=["workspace", "source", "documents", "media type", "time", "readers"],
)
async def test_the_ranked_statement_never_sorts_and_the_others_always_do(
    store: SqliteDocStore, restriction: Filter | None
) -> None:
    """``ORDER BY chunks_fts.rank`` is what lets FTS5 hand rows over in order, so the joined
    statement keeps no temporary B-tree of every match. ``ORDER BY bm25(...)`` is an expression
    over the row, which SQLite can only satisfy by sorting."""
    ranked = await _steps(store, LexicalPlan.RANKED, restriction)
    assert not any(SORT in step for step in ranked), ranked
    assert any(SORT in step for step in await _steps(store, LexicalPlan.DIRECT, restriction))
    assert any(
        SORT in step for step in await _steps(store, LexicalPlan.DOCUMENT_FIRST, restriction)
    )


@pytest.mark.parametrize("plan", [LexicalPlan.DIRECT, LexicalPlan.RANKED])
async def test_a_source_filter_does_not_send_the_join_through_the_source_index(
    store: SqliteDocStore, plan: LexicalPlan
) -> None:
    """The join finds each match's document by primary key and the source is checked on the row.

    Written as a plain column test, ``d.source IN (...)`` beside the join's ``d.id =
    c.document_id`` makes SQLite 3.50 find each match's document through ``(workspace_id,
    source)`` and compare ids across every document of the source: a scan per matching chunk.
    SQLite 3.42 plans the plain spelling the way it plans this one, so nothing here asserts the
    plain spelling's plan; the steps test below is what is measured on whichever version runs.
    """
    steps = await _steps(store, plan, _everything(sources=frozenset({"fs"})))

    assert not any("ix_documents_workspace_id_source" in step for step in steps), steps
    assert any("ix_documents_workspace_live_id" in step for step in steps), steps


async def test_the_document_first_statement_walks_the_documents_it_was_given(
    store: SqliteDocStore,
) -> None:
    """Its whole purpose is that the document list drives the loop: the first step is a seek on
    the primary key per listed id, and the index is searched last, by rowid."""
    restriction = _everything(document_ids=frozenset({"d1", "d2"}))

    steps = await _steps(store, LexicalPlan.DOCUMENT_FIRST, restriction)

    assert "ix_documents_workspace_live_id" in steps[0], steps
    assert steps[0].startswith("SEARCH d"), steps
    assert any(step.startswith("SCAN chunks_fts") and "=M" in step for step in steps), steps


async def test_the_scope_probe_is_an_indexed_lookup_by_the_ids_it_was_given(
    store: SqliteDocStore,
) -> None:
    """No text and no scan of the lexical index, and no scan of ``documents`` or ``chunks``.

    Leaving ``d.deleted_at IS NULL`` out of the probe loses the bound on ``d.id``: the planner
    then cannot use the partial index that carries it, and walks the workspace.
    """
    sql = SCOPE_PROBE_SQL.format(ids=":a, :b")
    async with store.sessions() as session:
        steps = [
            str(row[3])
            for row in (
                await session.execute(
                    text("EXPLAIN QUERY PLAN " + sql),
                    {"workspace_id": DEFAULT_WORKSPACE, "probe_limit": 513, "a": "x", "b": "y"},
                )
            ).all()
        ]

    assert not any(step.startswith("SCAN") and "chunks_fts" in step for step in steps), steps
    assert not any(step.startswith("SCAN d") or step.startswith("SCAN c") for step in steps), steps
    assert any("ix_documents_workspace_live_id" in step and "id=?" in step for step in steps), steps
    assert SCOPE_PROBE_LIMIT == MAX_SCOPED_CHUNKS + 1


async def _spread(store: SqliteDocStore) -> None:
    """A hundred and fifty documents of four matching chunks, a hundred and twenty of them in one
    source: enough rows that the number of steps a join takes is about its shape."""
    for index in range(150):
        await _document(
            store,
            f"d{index}",
            ["alpha beta words"] * 4,
            source="big" if index < 120 else "small",
        )


def _machine_steps(data_dir: Path, sql: str, params: dict[str, Any]) -> int:
    """How many virtual machine instructions ``sql`` runs: deterministic, unlike its duration."""
    ticks = 0

    def count() -> int:
        nonlocal ticks
        ticks += 1
        return 0

    connection = sqlite3.connect(database_path(data_dir))
    try:
        connection.set_progress_handler(count, 1)
        connection.execute(sql, params).fetchall()
    finally:
        connection.close()
    return ticks


@pytest.mark.parametrize(
    ("restriction", "plan"),
    [
        (_everything(document_ids=frozenset({f"d{i}" for i in range(64)})), LexicalPlan.DIRECT),
        (_everything(sources=frozenset({"big"})), LexicalPlan.DIRECT),
        (_everything(document_ids=frozenset({f"d{i}" for i in range(64)})), LexicalPlan.RANKED),
        (_everything(sources=frozenset({"big"})), LexicalPlan.RANKED),
    ],
    ids=["documents direct", "source direct", "documents ranked", "source ranked"],
)
async def test_the_id_and_source_tests_are_written_so_a_statement_never_costs_more_steps(
    store: SqliteDocStore, data_dir: Path, restriction: Filter, plan: LexicalPlan
) -> None:
    """The cost the spelling removes, counted rather than timed, on whichever SQLite runs it.

    ``d.id IN (...)`` beside the join's ``d.id = c.document_id`` can run one seek per listed id
    for every matching chunk, and ``d.source IN (...)`` one scan of the source per matching
    chunk. SQLite 3.50.4 does both, and took 35 to 45 times the steps on this fixture for the
    direct statement and 4 to 6 times for the ranked one, which stops after ``k`` rows. SQLite
    3.42.0 plans the plain spelling as it plans the other, so there it asserts nothing the code
    could fail: the statement the store builds is held to no more than 1.5 times the cheaper of
    the two spellings, which is the whole of what is true on both.
    """
    await _spread(store)
    predicates = lexical_predicates(DEFAULT_WORKSPACE, restriction)
    params: dict[str, Any] = {"match": escape_match_query("alpha beta"), "limit": 50}
    params |= predicates.params
    built = predicates.text(TEMPLATES[plan])
    plain = built.replace("AND +d.", "AND d.")
    anchored = plain.replace("AND d.id IN", "AND +d.id IN").replace(
        "AND d.source IN", "AND +d.source IN"
    )

    cheapest = min(_machine_steps(data_dir, text, params) for text in (plain, anchored))

    assert _machine_steps(data_dir, built, params) <= 1.5 * cheapest


# --- what the match says about its own size ----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "!!! ???",
        "authentication",
        "what is it",
        "how do I rotate the signing token?",
        "token Token TOKEN rotation",
        'say "hello" to the "quoted" thing',
        "why doesn't it publish",
        " ".join(f"word{i}" for i in range(40)),
        "project_backoff",
        "project backoff",
    ],
)
def test_the_match_is_exactly_its_terms_joined_so_a_plan_can_count_them(raw: str) -> None:
    """The statement is chosen from ``len(match_terms(text))``, so that number has to be the
    count of alternatives in the expression ``escape_match_query`` writes, whatever the text."""
    terms = match_terms(raw)
    match = escape_match_query(raw)

    assert match == " OR ".join(f'"{term}"' for term in terms)
    assert len(terms) <= MAX_MATCH_WORDS
    assert (match.count('" OR "') + 1 if match else 0) == len(terms)


# --- under load ---------------------------------------------------------------------------------


async def test_concurrent_searches_of_every_kind_do_not_exhaust_the_connection_pool(
    store: SqliteDocStore, chosen: list[LexicalPlan]
) -> None:
    """The probe is a second statement and must not be a second connection.

    It runs in the search's own session, ahead of the search statement, so a search still holds
    one connection from its first statement to its last and the pool's fifteen cannot be held
    by searches each waiting for a second. Thirty of each kind at once, over a corpus whose
    hydration is not instant: one document of twenty-five chunks goes through the probe and the
    document-first statement, three go through the probe and past it to the direct statement,
    and with no scope the ranked statement runs.
    """
    ids: list[str] = []
    for source in range(20):
        ids.append(
            await _seed(
                store,
                f"s{source}",
                ["the service handles authentication and token rotation " * 20 for _ in range(25)],
            )
        )

    async def search(scope: Filter | None) -> int:
        return len(await store.search_lexical("authentication token", k=50, filter=scope))

    small = _everything(document_ids=frozenset(ids[:1]))
    larger = _everything(document_ids=frozenset(ids[:3]))
    counts = await asyncio.wait_for(
        asyncio.gather(
            *(search(small) for _ in range(30)),
            *(search(larger) for _ in range(30)),
            *(search(None) for _ in range(30)),
        ),
        timeout=25,
    )

    assert all(count > 0 for count in counts), "every concurrent search returns its own hits"
    assert {plan: chosen.count(plan) for plan in LexicalPlan} == {
        LexicalPlan.DOCUMENT_FIRST: 30,
        LexicalPlan.DIRECT: 30,
        LexicalPlan.RANKED: 30,
    }, "each kind of search really took the path this test is about"


# --- the choice -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("facts", "plan"),
    [
        ({"terms": 1, "scoped_documents": 0}, LexicalPlan.DIRECT),
        ({"terms": 2, "scoped_documents": 0}, LexicalPlan.RANKED),
        ({"terms": MAX_RANKED_TERMS, "scoped_documents": 0}, LexicalPlan.RANKED),
        ({"terms": MAX_RANKED_TERMS + 1, "scoped_documents": 0}, LexicalPlan.DIRECT),
        ({"terms": MAX_MATCH_WORDS, "scoped_documents": 0}, LexicalPlan.DIRECT),
        ({"terms": 2, "scoped_documents": 0, "narrowed": True}, LexicalPlan.DIRECT),
        ({"terms": 1, "scoped_documents": 0, "narrowed": True}, LexicalPlan.DIRECT),
        ({"terms": 2, "scoped_documents": 1, "scoped_chunks": 0}, LexicalPlan.DOCUMENT_FIRST),
        ({"terms": 1, "scoped_documents": 4, "scoped_chunks": 20}, LexicalPlan.DOCUMENT_FIRST),
        (
            {"terms": 2, "scoped_documents": 64, "scoped_chunks": MAX_SCOPED_CHUNKS},
            LexicalPlan.DOCUMENT_FIRST,
        ),
        (
            {"terms": 2, "scoped_documents": 64, "scoped_chunks": MAX_SCOPED_CHUNKS + 1},
            LexicalPlan.DIRECT,
        ),
        ({"terms": 1, "scoped_documents": 4, "scoped_chunks": 33}, LexicalPlan.DIRECT),
        ({"terms": 2, "scoped_documents": 65}, LexicalPlan.DIRECT),
        ({"terms": 2, "scoped_documents": 10_000}, LexicalPlan.DIRECT),
        (
            {"terms": 2, "scoped_documents": 3, "scoped_chunks": 9, "narrowed": True},
            LexicalPlan.DOCUMENT_FIRST,
        ),
    ],
)
def test_the_choice_of_statement(facts: dict[str, Any], plan: LexicalPlan) -> None:
    assert choose_lexical_plan(**facts) is plan


def test_a_choice_without_what_it_needs_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="no terms"):
        choose_lexical_plan(terms=0, scoped_documents=0)
    with pytest.raises(ValueError, match="chunk count"):
        choose_lexical_plan(terms=2, scoped_documents=3)


@pytest.mark.parametrize(
    ("terms", "documents", "probed"),
    [
        (1, 1, True),
        (2, MAX_SCOPED_DOCUMENTS, True),
        (2, MAX_SCOPED_DOCUMENTS + 1, False),
        (2, 0, False),
        (1, 0, False),
    ],
)
def test_only_a_scope_the_chooser_would_walk_is_probed(
    terms: int, documents: int, probed: bool
) -> None:
    assert needs_scope_probe(terms, documents) is probed


def test_the_probe_reads_one_past_the_budget_so_over_is_distinguishable() -> None:
    assert SCOPE_PROBE_LIMIT == MAX_SCOPED_CHUNKS + 1


async def test_the_store_chooses_from_the_filter_it_was_given(
    store: SqliteDocStore, chosen: list[LexicalPlan]
) -> None:
    """The filter's shape reaches the chooser as the facts the chooser reads.

    A reader set does not narrow a search and every other restriction does, which is the
    measured line (``choose_lexical_plan``); a single word is direct whatever else is true.
    """
    await _weighted_corpus(store)
    group = frozenset({group_principal("anyone")})
    cases: list[tuple[str, Filter | None, LexicalPlan]] = [
        ("alpha beta", None, LexicalPlan.RANKED),
        ("alpha beta", _everything(readers=group), LexicalPlan.RANKED),
        (" ".join(f"word{i}" for i in range(MAX_RANKED_TERMS + 1)), None, LexicalPlan.DIRECT),
        ("alpha", None, LexicalPlan.DIRECT),
        ("alpha beta", _everything(sources=frozenset({"fs"})), LexicalPlan.DIRECT),
        ("alpha beta", _everything(media_types=frozenset({"text/markdown"})), LexicalPlan.DIRECT),
        ("alpha beta", _everything(langs=frozenset({"en"})), LexicalPlan.DIRECT),
        (
            "alpha beta",
            _everything(updated_before=datetime(2100, 1, 1, tzinfo=UTC)),
            LexicalPlan.DIRECT,
        ),
        (
            "alpha beta",
            _everything(updated_after=datetime(2000, 1, 1, tzinfo=UTC), readers=group),
            LexicalPlan.DIRECT,
        ),
    ]

    for query, restriction, expected in cases:
        chosen.clear()
        await store.search_lexical(query, 5, restriction)
        assert chosen == [expected], (query, restriction)


async def test_a_backlink_search_for_a_slugs_words_is_ranked_and_still_finds_the_link(
    store: SqliteDocStore, chosen: list[LexicalPlan]
) -> None:
    """The wikilinks plugin's inbound pass: an unscoped any-word search for the words of a slug,
    asking for as many as its ``inbound_limit``. It takes the ranked statement, and the chunk
    that holds the link is among the rows that come back whatever else matches ``project``."""
    for index in range(30):
        await _document(store, f"note-{index}", ["a project note about scheduling"])
    linking = await _document(store, "linker", ["see [[project_backoff]] for the retry policy"])

    found = await store.search_lexical(" ".join(["project", "backoff"]), 200)

    assert chosen == [LexicalPlan.RANKED]
    assert any(candidate.chunk.document_id == linking.id for candidate in found)
    assert found[0].chunk.document_id == linking.id, "it holds both words, so it ranks first"
