"""The lexical leg: a re-keyed score, a merge, and zero results as an event."""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.retrieval import Candidate
from manicule.retrieval.lexical import DEGRADED_REASON, LexicalStage
from manicule.retrieval.trace import LexicalReport, installed
from manicule.storage.docstore import SqliteDocStore
from manicule.testing import assert_pipeline_enforces_scope, assert_retrieval_stage_contract
from tests.retrieval.fakes import a_query, profiles
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def _stage(store: SqliteDocStore) -> LexicalStage:
    return LexicalStage(docstore=store, profiles=profiles(candidates=3, final_top_k=3))


async def _indexed(store: SqliteDocStore) -> list[str]:
    document = make_document(source_id="live")
    await store.upsert_document(document)
    chunks = [
        make_chunk(document, 0, "authentication tokens rotate"),
        make_chunk(document, 1, "unrelated prose about weather"),
    ]
    await store.replace_chunks(document.id, chunks)
    return [chunk.id for chunk in chunks]


async def test_the_stage_records_its_own_name_over_the_store_s_algorithm_key(
    store: SqliteDocStore,
) -> None:
    """The store writes ``bm25``, which names the algorithm; scores are keyed by *stage*.

    Both keys survive and the duplicate is harmless. What matters is that fusion reads the
    names it was configured with rather than a key some store happened to write — which is what
    lets this leg be swapped for a learned-sparse one without touching the fusion stage.
    """
    await _indexed(store)

    produced = await _stage(store).run(a_query("authentication"), [])

    assert produced
    assert produced[0].scores["lexical"] == produced[0].scores["bm25"]


async def test_a_better_match_scores_higher_after_the_store_negates_bm25(
    store: SqliteDocStore,
) -> None:
    """``bm25()`` is negative and more negative is better, so the ordering is ascending.

    Anything that treats it as a similarity, or takes its absolute value, inverts or flattens
    the ranking and still produces a plausible ranked list. The store negates it once; fusion
    then reads ranks and never magnitudes, which sidesteps the question entirely.
    """
    await _indexed(store)

    produced = await _stage(store).run(a_query("authentication"), [])

    assert all(candidate.scores["lexical"] > 0 for candidate in produced)
    assert produced == sorted(produced, key=lambda c: c.scores["lexical"], reverse=True)


async def test_it_merges_rather_than_replaces(store: SqliteDocStore) -> None:
    """A chunk both legs found carries both scores, which is what fusion and agreement read."""
    ids = await _indexed(store)
    chunks = await store.get_chunks(ids[:1])
    incoming = [Candidate(chunk=chunks[0], score=0.9, scores={"dense": 0.9})]

    produced = await _stage(store).run(a_query("authentication"), incoming)

    found = next(c for c in produced if c.chunk.id == ids[0])
    assert set(found.scores) >= {"dense", "lexical"}


async def test_zero_matches_is_recorded_and_makes_the_run_incomparable(
    store: SqliteDocStore,
) -> None:
    """An all-stopword query and a failed lexical index look identical from here.

    Either way the pipeline continues on one leg and the ranking it produces is well-formed —
    so the run has to carry that it was single-leg. A line on somebody's terminal is invisible
    to an evaluation harness, which would then compare a one-leg run against a two-leg one.
    """
    await _indexed(store)

    with installed() as frame:
        produced = await _stage(store).run(a_query("!!!"), [])
        report = LexicalReport.model_validate(frame.take_diagnostics())

    assert produced == []
    assert report.matched == 0
    assert report.degraded
    assert DEGRADED_REASON in frame.incomparable


async def test_the_leg_is_already_scoped(store: SqliteDocStore, engine: AsyncEngine) -> None:
    """Its filter is applied inside the statement, before ``LIMIT``.

    That is the fix for the same trap the dense leg absorbs by over-fetching, and it is why the
    scope invariant holds at *every* stage boundary rather than at one privileged point.
    """
    await _indexed(store)
    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    foreign = make_document(source_id="theirs", workspace_id="beta")
    await other.upsert_document(foreign)
    await other.replace_chunks(foreign.id, [make_chunk(foreign, 0, "authentication theirs")])

    await assert_pipeline_enforces_scope([_stage(store)], store, a_query("authentication"))


async def test_the_stage_contract_holds(store: SqliteDocStore) -> None:
    ids = await _indexed(store)
    chunks = await store.get_chunks(ids[:1])
    await assert_retrieval_stage_contract(
        _stage(store),
        a_query("authentication"),
        [Candidate(chunk=chunks[0], score=1.0, scores={"dense": 1.0})],
    )
