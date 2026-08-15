"""The dense leg: the top-``k`` trap on the side that cannot filter before ``LIMIT``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.content import DocumentStatus
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.retrieval.config import DenseConfig
from manicule.retrieval.dense import DenseStage, derive_over_fetch
from manicule.retrieval.trace import DenseReport, Regime, Shortfall, installed
from manicule.storage.docstore import SqliteDocStore
from manicule.testing import assert_pipeline_enforces_scope, assert_retrieval_stage_contract
from tests.fakes import HashEmbedder
from tests.retrieval.fakes import SCOPE, ListVectorStore, a_query, profiles
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk

pytestmark = pytest.mark.usefixtures("store")

SMALL = {"candidates": 3, "final_top_k": 3}
"""A profile small enough that the trap is reachable at all: with ``k`` at 20 an index has to
be twenty deep in invisible rows before anything is lost, and the failure is the same shape."""


async def _corpus(store: SqliteDocStore, engine: AsyncEngine) -> tuple[list[Chunk], list[Chunk]]:
    """Three live chunks, and ten a search must never return, ranked above them.

    Five belong to a soft-deleted document and five to another workspace — the two things the
    vector table cannot distinguish, because neither is a column it has.
    """
    excluded: list[Chunk] = []
    removed = make_document(source_id="removed")
    await store.upsert_document(removed)
    gone = [make_chunk(removed, position, f"authentication {position}") for position in range(5)]
    await store.replace_chunks(removed.id, gone)
    await store.soft_delete_document(removed.id)
    excluded.extend(gone)

    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    foreign_document = make_document(source_id="theirs", workspace_id="beta")
    await other.upsert_document(foreign_document)
    foreign = [
        make_chunk(foreign_document, position, f"authentication theirs {position}")
        for position in range(5)
    ]
    await other.replace_chunks(foreign_document.id, foreign)
    excluded.extend(foreign)

    live_document = make_document(source_id="live")
    await store.upsert_document(live_document)
    live = [
        make_chunk(live_document, position, f"authentication live {position}")
        for position in range(3)
    ]
    await store.replace_chunks(live_document.id, live)

    return excluded, live


def _stage(
    store: SqliteDocStore, vectors: ListVectorStore, *, config: DenseConfig | None = None
) -> DenseStage:
    return DenseStage(
        embedder=HashEmbedder(),
        vectors=vectors,
        docstore=store,
        profiles=profiles(**SMALL),
        config=config,
    )


async def test_excluded_rows_consume_the_top_k_slots_before_any_join(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The measured failure this leg exists to avoid, shown at the store level first.

    Asking the vector index for exactly ``k`` and joining afterwards returns **zero** live
    results here — not a marginal loss of recall but a total one, silently, with a well-formed
    empty result set. Nothing raises, and the answer looks like a corpus with nothing in it.
    """
    excluded, live = await _corpus(store, engine)
    vectors = ListVectorStore([*excluded, *live])

    naive = await vectors.search([0.0] * 5, 3)
    surviving = [
        candidate
        for candidate in naive
        if await store.get_document(candidate.chunk.document_id) is not None
    ]

    assert len(naive) == 3
    assert surviving == []
    assert len(live) == 3  # the corpus really did have three matching live chunks


async def test_the_dense_leg_over_fetches_past_them_and_returns_the_live_ones(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The same corpus, through the stage: all three live chunks, none of the ten."""
    excluded, live = await _corpus(store, engine)
    vectors = ListVectorStore([*excluded, *live])

    produced = await _stage(store, vectors).run(a_query(), [])

    assert [candidate.chunk.id for candidate in produced] == [chunk.id for chunk in live]
    assert vectors.requested[0] > 3, "the leg asked for exactly k and cannot have scoped anything"


async def test_the_leg_satisfies_the_pipeline_scope_assertion(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The structural guard that makes the vector store's ``workspace_ids`` exemption safe."""
    excluded, live = await _corpus(store, engine)
    stage = _stage(store, ListVectorStore([*excluded, *live]))

    kept = await assert_pipeline_enforces_scope([stage], store, a_query())

    assert {candidate.chunk.id for candidate in kept} == {chunk.id for chunk in live}


async def test_a_pending_document_is_invisible_to_the_leg(store: SqliteDocStore) -> None:
    """A document mid-ingest has chunks whose vectors and text need not agree yet."""
    waiting = make_document(source_id="waiting", status=DocumentStatus.PENDING)
    await store.upsert_document(waiting)
    chunk = make_chunk(waiting, 0, "authentication waiting")
    await store.replace_chunks(waiting.id, [chunk])

    produced = await _stage(store, ListVectorStore([chunk])).run(a_query(), [])

    assert produced == []


async def test_only_the_active_vector_publication_survives_hydration(
    store: SqliteDocStore,
) -> None:
    """Staged and retired rows may rank, but neither may cross the relational commit point."""
    document = make_document(source_id="published", status=DocumentStatus.INDEXED).model_copy(
        update={"publication_id": "current"}
    )
    await store.upsert_document(document)
    chunk = make_chunk(document, 0, "authentication publication")
    await store.replace_chunks(document.id, [chunk])
    vectors = ListVectorStore([chunk, chunk], scores=[1.0, 0.9], publications=["staged", "current"])

    produced = await _stage(store, vectors).run(a_query(limit=1), [])

    assert [(item.chunk.id, item.publication_id) for item in produced] == [(chunk.id, "current")]


async def test_the_chunk_that_comes_back_is_the_authoritative_one(
    store: SqliteDocStore,
) -> None:
    """SQLite is authoritative, and a divergence must resolve toward the truth.

    The vector row carries its own copy of the chunk so the store can satisfy the protocol
    with no relational database behind it. That copy is not what gets cited: if the two ever
    disagree, retrieval quoting the derived one would produce a citation the source does not
    contain.
    """
    document = make_document(source_id="live")
    await store.upsert_document(document)
    stored = make_chunk(document, 0, "authentication live")
    await store.replace_chunks(document.id, [stored])
    stale = stored.model_copy(update={"text": "text the index has not seen"})

    produced = await _stage(store, ListVectorStore([stale])).run(a_query(), [])

    assert [candidate.chunk.text for candidate in produced] == [stored.text]


async def test_the_similarity_floor_is_counted_apart_from_the_join(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """One means the index is dirty and the other means the query is hard.

    Collapsing them into "dropped" loses the only signal that says which of two very different
    problems you have.
    """
    excluded, live = await _corpus(store, engine)
    vectors = ListVectorStore([*excluded, *live], scores=[0.9] * len(excluded) + [0.9, 0.05, 0.05])
    stage = DenseStage(
        embedder=HashEmbedder(),
        vectors=vectors,
        docstore=store,
        profiles=profiles(candidates=3, final_top_k=3, min_score=0.5),
    )

    with installed() as frame:
        produced = await stage.run(a_query(), [])
        report = DenseReport.model_validate(frame.take_diagnostics())

    assert len(produced) == 1
    assert report.dropped_by_join == len(excluded)
    assert report.dropped_by_min_score == 2


@pytest.mark.parametrize(
    ("live_fraction", "expected"),
    [
        (1.0, 30),  # a clean personal index lands on the floor: 3 x k
        (0.5, 30),  # 2x is below the floor, so the floor wins
        (0.1, 100),  # a dilute index asks for 10x
        (0.02, 200),  # 50x is past the cap, which is the signal to invert the plan
        (0.0, 200),  # clamped, never divided by zero
    ],
)
def test_the_over_fetch_factor_is_derived_and_clamped(live_fraction: float, expected: int) -> None:
    """A constant multiplier is wrong in both directions; this one measures which it is in."""
    assert derive_over_fetch(10, live_fraction, DenseConfig()) == expected


def test_the_absolute_row_cap_bounds_the_work_independently_of_the_multiplier() -> None:
    """Every over-fetched row is a chunk decode, so the count is bounded on its own."""
    assert derive_over_fetch(500, 0.02, DenseConfig()) == 2000


def test_a_floor_above_the_cap_is_refused() -> None:
    """It would clamp the derived factor below its own floor and report the cap firing always."""
    with pytest.raises(ValueError, match="above overfetch_max"):
        DenseConfig(overfetch_min=30, overfetch_max=20)


async def test_a_store_with_no_live_count_changes_round_trips_and_not_results(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """A capability a store lacks may cost throughput. It must never change what comes back.

    Two deployments of manicule answering the same question differently is the failure this
    property exists to prevent, and it is the same shape as a platform difference that changes
    output rather than speed.
    """
    excluded, live = await _corpus(store, engine)
    vectors = ListVectorStore([*excluded, *live])

    class Countless:
        """A document store that answers everything except how much of it is live."""

        def __init__(self, inner: SqliteDocStore) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            if name == "live_chunk_count":
                raise AttributeError(name)
            return getattr(self._inner, name)

    measured = await _stage(store, ListVectorStore([*excluded, *live])).run(a_query(), [])
    unmeasured = await _stage(Countless(store), vectors).run(a_query(), [])  # pyright: ignore[reportArgumentType]

    assert [c.chunk.id for c in unmeasured] == [c.chunk.id for c in measured]


async def test_a_filter_that_resolves_to_nothing_returns_nothing(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """ "No join-requiring field set" and "resolved to nothing" are opposite instructions.

    Collapsing them is a filter bypass: a query restricted to a source with no documents would
    return the whole workspace, ranked and plausible, and every result would satisfy the query
    while violating the filter.
    """
    excluded, live = await _corpus(store, engine)
    stage = _stage(store, ListVectorStore([*excluded, *live]))
    query = Query(
        text="authentication",
        limit=3,
        filter=Filter(workspace_ids=SCOPE, sources=frozenset({"nowhere"})),
    )

    with installed() as frame:
        produced = await stage.run(query, [])
        report = DenseReport.model_validate(frame.take_diagnostics())

    assert produced == []
    assert report.regime is Regime.EMPTY


async def test_a_leg_stopped_by_its_own_caps_says_so(store: SqliteDocStore) -> None:
    """``exhausted_budget`` is a defect and the other two outcomes are not.

    It never quietly returns fewer than ``k`` and lets the caller assume the corpus had no
    more: the result is a floor, it caps confidence, and it makes the run non-comparable
    because it ran a different amount of search from one that finished.
    """
    document = make_document(source_id="live")
    await store.upsert_document(document)
    kept = make_chunk(document, 0, "authentication live")
    await store.replace_chunks(document.id, [kept])
    hidden = make_document(source_id="removed")
    await store.upsert_document(hidden)
    invisible = [
        make_chunk(hidden, position, f"authentication {position}") for position in range(60)
    ]
    await store.replace_chunks(hidden.id, invisible)
    await store.soft_delete_document(hidden.id)

    vectors = ListVectorStore([*invisible, kept])
    stage = _stage(store, vectors, config=DenseConfig(max_expansions=0, absolute_row_cap=2000))

    with installed() as frame:
        await stage.run(a_query(), [])
        report = DenseReport.model_validate(frame.take_diagnostics())

    assert report.outcome is Shortfall.EXHAUSTED_BUDGET
    assert report.survived < report.requested


async def test_a_corpus_with_nothing_more_is_not_a_defect(store: SqliteDocStore) -> None:
    """``exhausted_corpus`` means the store genuinely holds no more matching rows."""
    document = make_document(source_id="live")
    await store.upsert_document(document)
    only = make_chunk(document, 0, "authentication live")
    await store.replace_chunks(document.id, [only])

    with installed() as frame:
        await _stage(store, ListVectorStore([only])).run(a_query(), [])
        report = DenseReport.model_validate(frame.take_diagnostics())

    assert report.outcome is Shortfall.EXHAUSTED_CORPUS


async def test_expanding_recovers_what_the_first_fetch_missed(store: SqliteDocStore) -> None:
    """A factor that failed is usually wrong by more than a little, so a retry widens by four."""
    document = make_document(source_id="live")
    await store.upsert_document(document)
    live = [make_chunk(document, position, f"authentication {position}") for position in range(3)]
    await store.replace_chunks(document.id, live)
    hidden = make_document(source_id="removed")
    await store.upsert_document(hidden)
    invisible = [
        make_chunk(hidden, position, f"authentication x{position}") for position in range(30)
    ]
    await store.replace_chunks(hidden.id, invisible)
    await store.soft_delete_document(hidden.id)

    stage = DenseStage(
        embedder=HashEmbedder(),
        vectors=ListVectorStore([*invisible, *live]),
        docstore=store,
        profiles=profiles(**SMALL),
        # A store that will not report its live fraction, so the leg starts at the floor of 9
        # and has to expand to reach past the 30 invisible rows.
        config=DenseConfig(overfetch_min=3, overfetch_max=3),
    )

    with installed() as frame:
        produced = await stage.run(a_query(), [])
        report = DenseReport.model_validate(frame.take_diagnostics())

    assert len(produced) == 3
    assert report.expansions >= 1
    assert report.outcome is Shortfall.SATISFIED


async def test_the_leg_merges_rather_than_replaces(store: SqliteDocStore) -> None:
    """A chunk both legs found must carry both scores; that is what fusion reads."""
    document = make_document(source_id="live")
    await store.upsert_document(document)
    only = make_chunk(document, 0, "authentication live")
    await store.replace_chunks(document.id, [only])
    incoming = [Candidate(chunk=only, score=4.0, scores={"lexical": 4.0})]

    produced = await _stage(store, ListVectorStore([only])).run(a_query(), incoming)

    assert set(produced[0].scores) == {"lexical", "dense"}


async def test_the_stage_contract_holds(store: SqliteDocStore) -> None:
    """Candidates in, candidates out, input untouched, a new list returned."""
    document = make_document(source_id="live")
    await store.upsert_document(document)
    only = make_chunk(document, 0, "authentication live")
    await store.replace_chunks(document.id, [only])

    await assert_retrieval_stage_contract(
        _stage(store, ListVectorStore([only])),
        a_query(),
        [Candidate(chunk=only, score=1.0, scores={"lexical": 1.0})],
    )
