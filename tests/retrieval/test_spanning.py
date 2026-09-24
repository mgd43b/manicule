"""A search spanning workspaces: one scoped leg each, merged on cosine, and nothing more.

``docs/retrieval.md`` §3.2 settles the rule; these are the ways a merge could break it while
still returning a plausible ranking. Every fixture puts two workspaces in one database, and each
workspace's vector store returns rows at scores the test chose, so the order a merge produces is
decided by the numbers written here rather than by a hash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.errors import ConfigError
from manicule.core.retrieval import Candidate, Filter, PipelineIdentity, Query
from manicule.retrieval.assembly import ContextAssembler
from manicule.retrieval.cache import CachedRanking, L1QueryCache, cache_key
from manicule.retrieval.confidence import AGREEMENT, SIMILARITY
from manicule.retrieval.dense import DenseStage
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.lexical import LexicalStage
from manicule.retrieval.rerank import CrossEncoderReranker
from manicule.retrieval.retriever import Retriever
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.spanning import (
    WorkspaceLeg,
    leg_query,
    merge_on_similarity,
    rehydrate_across,
)
from manicule.retrieval.tokens import ContextTokenCounter
from manicule.storage.docstore import SqliteDocStore
from manicule.testing import assert_pipeline_enforces_scope
from tests.fakes import HashEmbedder
from tests.retrieval.fakes import FixedScorer, ListVectorStore, profiles
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk
    from manicule.core.protocols import RetrievalStage

ALPHA = "alpha"
BETA = "beta"

DEPTH = 2
"""Two candidates a leg, so that the top-k trap is two rows deep rather than twenty."""

SMALL = {"candidates": DEPTH, "final_top_k": DEPTH, "min_score": 0.0}


async def _workspace(engine: AsyncEngine, name: str) -> SqliteDocStore:
    store = SqliteDocStore(engine, workspace_id=name)
    await store.ensure_workspace()
    return store


async def _passages(
    store: SqliteDocStore, workspace: str, source_id: str, texts: Sequence[str]
) -> list[Chunk]:
    """One indexed document in ``workspace`` whose chunks are ``texts``."""
    document = make_document(source_id=source_id, workspace_id=workspace, title=source_id)
    await store.upsert_document(document)
    chunks = [make_chunk(document, position, text) for position, text in enumerate(texts)]
    await store.replace_chunks(document.id, chunks)
    return chunks


def _retriever(
    serving: SqliteDocStore,
    vectors: ListVectorStore,
    *,
    cache: L1QueryCache | None = None,
    rerank: FixedScorer | None = None,
    assert_scope: bool = False,
    dense: bool = True,
) -> Retriever:
    """The shipped retriever over ``serving``, with the single-workspace pipeline it would run."""
    resolved = profiles(**SMALL)
    stages: list[RetrievalStage] = []
    if dense:
        stages.append(
            DenseStage(
                embedder=HashEmbedder(), vectors=vectors, docstore=serving, profiles=resolved
            )
        )
    stages += [LexicalStage(docstore=serving, profiles=resolved)]
    if dense:
        stages.append(RRFStage())
    if rerank is not None:
        stages.append(CrossEncoderReranker(scorer=rerank, profiles=resolved))
    return Retriever(
        runner=PipelineRunner(stages, docstore=serving, assert_scope=assert_scope),
        docstore=serving,
        assembler=ContextAssembler(counter=ContextTokenCounter(), profiles=resolved),
        profiles=resolved,
        cache=cache,
        embed_fingerprint=HashEmbedder().fingerprint.canonical(),
        reranker_model_id=rerank.model_id if rerank is not None else None,
    )


def _spanning(*workspaces: str, text: str = "rotation", limit: int = 2) -> Query:
    return Query(text=text, limit=limit, filter=Filter(workspace_ids=frozenset(workspaces)))


def _order(candidates: Sequence[Candidate]) -> list[str]:
    return [candidate.chunk.text for candidate in candidates]


# --- the merge ---------------------------------------------------------------------------------


async def test_a_merged_ranking_is_ordered_by_cosine_across_workspaces(
    engine: AsyncEngine,
) -> None:
    """The settled rule: cosine, compared across workspaces, and never a rank per workspace.

    Alpha's second passage sits below beta's first, so a merge that interleaved by rank —
    which is what fusion does — would put it second rather than third.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_chunks = await _passages(alpha, ALPHA, "a", ["alpha best", "alpha second"])
    beta_chunks = await _passages(beta, BETA, "b", ["beta best", "beta second"])
    alpha_vectors = ListVectorStore(alpha_chunks, scores=[0.9, 0.6])
    beta_vectors = ListVectorStore(beta_chunks, scores=[0.8, 0.5])
    retriever = _retriever(alpha, alpha_vectors)

    result = await retriever.retrieve_across(
        _spanning(ALPHA, BETA, limit=4),
        [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)],
    )

    assert _order(result.candidates) == ["alpha best", "beta best", "alpha second", "beta second"]
    assert [result.origins[candidate.chunk.id] for candidate in result.candidates] == [
        (ALPHA,),
        (BETA,),
        (ALPHA,),
        (BETA,),
    ]
    # One dense span per workspace, attributed, and no lexical leg or fusion: BM25 cannot rank
    # one workspace against another and RRF over disjoint ladders ranks the workspaces.
    assert [(span.name, span.config.get("workspace")) for span in result.trace.stages] == [
        ("dense", ALPHA),
        ("dense", BETA),
    ]
    assert result.trace.pipeline.stages == ("dense",)
    assert result.trace.pipeline.rrf_k is None


async def test_a_workspace_whose_best_rows_are_deleted_still_contributes_its_next_best(
    engine: AsyncEngine,
) -> None:
    """The top-k trap in its cross-workspace form, and the per-leg over-fetch that avoids it.

    Beta's three highest-scoring rows belong to a soft-deleted document and its fourth beats
    everything alpha has. Asking each store for exactly ``k`` rows and filtering afterwards —
    the naive fan-out — would leave beta with nothing to offer and return alpha's two as though
    they were the best in either workspace.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_chunks = await _passages(alpha, ALPHA, "a", ["alpha one", "alpha two"])
    deleted = await _passages(beta, BETA, "gone", ["beta gone 1", "beta gone 2", "beta gone 3"])
    await beta.soft_delete_document(deleted[0].document_id)
    kept = await _passages(beta, BETA, "kept", ["beta survivor"])
    alpha_vectors = ListVectorStore(alpha_chunks, scores=[0.7, 0.6])
    beta_vectors = ListVectorStore([*deleted, *kept], scores=[0.99, 0.98, 0.97, 0.9])

    result = await _retriever(alpha, alpha_vectors).retrieve_across(
        _spanning(ALPHA, BETA),
        [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)],
    )

    assert _order(result.candidates) == ["beta survivor", "alpha one"]
    assert beta_vectors.requested[-1] > DEPTH, "beta's leg asked for exactly k"
    naive = await beta_vectors.search([0.0], DEPTH)
    assert all(candidate.chunk.document_id == deleted[0].document_id for candidate in naive), (
        "the fixture no longer reproduces the trap: beta's top k must all be deleted rows"
    )


async def test_the_merged_list_is_trimmed_to_the_pipeline_s_depth_only_after_merging(
    engine: AsyncEngine,
) -> None:
    """Each leg returns ``k``; the merge returns ``k`` overall, and they are the best ``k``.

    Trimming per leg and concatenating would return ``2k`` rows here; trimming the union before
    the legs had filtered is the trap above. Beta holds both of the best rows, so a merge that
    kept a quota per workspace would also be caught.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_chunks = await _passages(alpha, ALPHA, "a", ["alpha one", "alpha two", "alpha three"])
    beta_chunks = await _passages(beta, BETA, "b", ["beta one", "beta two", "beta three"])
    alpha_vectors = ListVectorStore(alpha_chunks, scores=[0.5, 0.4, 0.3])
    beta_vectors = ListVectorStore(beta_chunks, scores=[0.9, 0.8, 0.7])

    result = await _retriever(alpha, alpha_vectors).retrieve_across(
        _spanning(ALPHA, BETA),
        [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)],
    )

    assert _order(result.candidates) == ["beta one", "beta two"]
    assert set(result.origins) == {candidate.chunk.id for candidate in result.candidates}


async def test_a_reranker_scores_the_merged_pool_once_and_decides_the_order(
    engine: AsyncEngine,
) -> None:
    """One cross-encoder pass over passages from every workspace, not one pass per workspace.

    Reranking each leg separately would put each workspace's logits on a scale of its own
    candidates, and the reranker's order is the final one only if it saw them together.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_chunks = await _passages(alpha, ALPHA, "a", ["alpha passage"])
    beta_chunks = await _passages(beta, BETA, "b", ["beta passage"])
    alpha_vectors = ListVectorStore(alpha_chunks, scores=[0.9])
    beta_vectors = ListVectorStore(beta_chunks, scores=[0.5])
    scorer = FixedScorer({"alpha passage": -1.0, "beta passage": 3.0})

    result = await _retriever(alpha, alpha_vectors, rerank=scorer).retrieve_across(
        _spanning(ALPHA, BETA),
        [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)],
    )

    assert len(scorer.calls) == 1
    assert sorted(passage for _, passage in scorer.calls[0]) == ["alpha passage", "beta passage"]
    assert _order(result.candidates) == ["beta passage", "alpha passage"]
    assert result.trace.pipeline.stages == ("dense", "rerank")
    assert result.trace.pipeline.reranker_model_id == scorer.model_id


async def test_confidence_reads_the_merged_cosines_and_says_agreement_could_not_be_scored(
    engine: AsyncEngine,
) -> None:
    """One leg ran, so no passage can carry two legs' scores.

    Scoring agreement as zero would report weak corroboration for what is a property of the
    search; suppressing it lowers the ceiling and says why, which is the module's rule for a
    component that did not run.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_vectors = ListVectorStore(await _passages(alpha, ALPHA, "a", ["alpha"]), scores=[0.7])
    beta_vectors = ListVectorStore(await _passages(beta, BETA, "b", ["beta"]), scores=[0.6])

    result = await _retriever(alpha, alpha_vectors).retrieve_across(
        _spanning(ALPHA, BETA),
        [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)],
    )

    assert result.confidence is not None
    assert SIMILARITY in result.confidence.components
    assert AGREEMENT in result.confidence.suppressed
    assert result.expansion is None, "a spanning search consults no glossary"


# --- what the legs must be ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offered", "problem"),
    [
        ((ALPHA,), "named with no leg"),
        ((ALPHA, BETA, "gamma"), "offered and not named"),
        ((ALPHA, BETA, BETA), "offered twice"),
    ],
)
async def test_legs_must_cover_the_named_workspaces_exactly_once(
    engine: AsyncEngine, offered: tuple[str, ...], problem: str
) -> None:
    """A missing leg is a partial answer that reports itself whole; an extra one widens it."""
    alpha = await _workspace(engine, ALPHA)
    vectors = ListVectorStore([])
    legs = [WorkspaceLeg(name, alpha, vectors) for name in offered]

    with pytest.raises(ValueError, match=problem):
        await _retriever(alpha, vectors).retrieve_across(_spanning(ALPHA, BETA), legs)


async def test_a_pipeline_with_no_dense_leg_has_nothing_to_merge_on(engine: AsyncEngine) -> None:
    """Refused by name rather than merged on BM25, which ranks the workspaces, not the passages."""
    alpha = await _workspace(engine, ALPHA)
    vectors = ListVectorStore([])

    with pytest.raises(ConfigError, match="no dense stage"):
        await _retriever(alpha, vectors, dense=False).retrieve_across(
            _spanning(ALPHA, BETA),
            [WorkspaceLeg(ALPHA, alpha, vectors), WorkspaceLeg(BETA, alpha, vectors)],
        )


def test_a_candidate_with_no_cosine_is_refused_rather_than_sunk() -> None:
    """Sorting a missing score to the bottom would rank it by omission, on no scale at all."""
    document = make_document(workspace_id=ALPHA)
    scored = Candidate(chunk=make_chunk(document, 0, "scored"), score=0.5, scores={"dense": 0.5})
    unscored = Candidate(chunk=make_chunk(document, 1, "unscored"), score=0.9)

    assert merge_on_similarity([(ALPHA, [scored])], stage="dense").candidates == [scored]
    with pytest.raises(ValueError, match="no 'dense' score"):
        merge_on_similarity([(ALPHA, [scored, unscored])], stage="dense")


def test_a_chunk_two_legs_returned_keeps_both_origins() -> None:
    """A correct store returns a chunk from one workspace; the merge does not pick a winner.

    Keeping only the first leg's claim would hide a leak whenever the leaking leg ran second.
    Both claims survive, and the service refuses a hit claimed twice.
    """
    document = make_document(workspace_id=BETA)
    shared = Candidate(chunk=make_chunk(document, 0, "shared"), score=0.5, scores={"dense": 0.5})

    merged = merge_on_similarity([(ALPHA, [shared]), (BETA, [shared])], stage="dense")

    assert merged.candidates == [shared]
    assert merged.origins == {shared.chunk.id: (ALPHA, BETA)}


def test_a_leg_query_names_exactly_one_workspace_and_changes_nothing_else() -> None:
    """Each leg is the caller's request, narrowed to one workspace and widened in nothing."""
    query = Query(
        text="rotation",
        limit=3,
        filter=Filter(
            workspace_ids=frozenset({ALPHA, BETA}),
            sources=frozenset({"wiki"}),
            collection_ids=frozenset({"c1", "c2"}),
        ),
    )

    leg = leg_query(query, BETA)

    assert leg.filter.workspace_ids == frozenset({BETA})
    assert leg.filter.model_copy(update={"workspace_ids": query.filter.workspace_ids}) == (
        query.filter
    )
    assert leg.model_copy(update={"filter": query.filter}) == query


# --- scope -------------------------------------------------------------------------------------


async def test_every_leg_is_as_scoped_as_the_leg_it_was_bound_from(engine: AsyncEngine) -> None:
    """The conformance check the single-workspace pipeline passes, passed by a bound leg too.

    Beta's vector store holds alpha's rows and a deleted document's: everything a leg that
    skipped its join would return. Binding the configured leg to beta's handles must keep the
    join, because the join is inside the leg.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    foreign = await _passages(alpha, ALPHA, "a", ["alpha's own"])
    deleted = await _passages(beta, BETA, "gone", ["beta deleted"])
    await beta.soft_delete_document(deleted[0].document_id)
    live = await _passages(beta, BETA, "live", ["beta live"])
    beta_vectors = ListVectorStore([*foreign, *deleted, *live])
    configured = DenseStage(
        embedder=HashEmbedder(),
        vectors=ListVectorStore([]),
        docstore=alpha,
        profiles=profiles(**SMALL),
    )

    bound = configured.for_workspace(BETA, vectors=beta_vectors, docstore=beta)

    final = await assert_pipeline_enforces_scope(
        [bound], beta, leg_query(_spanning(ALPHA, BETA), BETA)
    )
    assert _order(final) == ["beta live"]


async def test_the_runtime_scope_assertion_holds_every_leg(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rag.assert_scope`` reaches a spanning search's legs, not only the ordinary pipeline.

    The dense leg's join is disabled for this test — the defect the assertion exists to catch —
    and beta's store then returns a deleted row. With the assertion on, the search fails rather
    than returning it; with it off, the row comes back, which is what proves the assertion and
    not something else is what stopped it.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_vectors = ListVectorStore(await _passages(alpha, ALPHA, "a", ["alpha"]), scores=[0.5])
    deleted = await _passages(beta, BETA, "gone", ["beta deleted"])
    await beta.soft_delete_document(deleted[0].document_id)
    beta_vectors = ListVectorStore(deleted, scores=[0.9])
    legs = [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)]

    async def unjoined(
        self: DenseStage, rows: Sequence[Candidate], join: Filter
    ) -> list[Candidate]:
        del self, join
        return list(rows)

    monkeypatch.setattr(DenseStage, "_hydrate", unjoined)

    with pytest.raises(AssertionError, match="cannot see"):
        await _retriever(alpha, alpha_vectors, assert_scope=True).retrieve_across(
            _spanning(ALPHA, BETA), legs
        )
    leaked = await _retriever(alpha, alpha_vectors).retrieve_across(_spanning(ALPHA, BETA), legs)
    assert "beta deleted" in _order(leaked.candidates)


async def test_collections_resolve_in_each_workspace_s_own_store(engine: AsyncEngine) -> None:
    """A collection is one workspace's, so each leg resolves the scope through its own store.

    Both workspaces have a collection of the same name and each id goes into the one filter;
    each leg must come back restricted to its own workspace's members — and a workspace whose
    collection holds nothing contributes nothing, never its whole corpus.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_in = await _passages(alpha, ALPHA, "in", ["alpha member"])
    alpha_out = await _passages(alpha, ALPHA, "out", ["alpha outsider"])
    beta_out = await _passages(beta, BETA, "out", ["beta outsider"])
    alpha_runbooks = await alpha.create_collection("runbooks")
    beta_runbooks = await beta.create_collection("runbooks")
    await alpha.add_to_collection(alpha_runbooks.id, [alpha_in[0].document_id])
    alpha_vectors = ListVectorStore([*alpha_out, *alpha_in], scores=[0.9, 0.5])
    beta_vectors = ListVectorStore(beta_out, scores=[0.95])
    query = Query(
        text="rotation",
        limit=4,
        filter=Filter(
            workspace_ids=frozenset({ALPHA, BETA}),
            collection_ids=frozenset({alpha_runbooks.id, beta_runbooks.id}),
        ),
    )

    result = await _retriever(alpha, alpha_vectors).retrieve_across(
        query,
        [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)],
    )

    assert _order(result.candidates) == ["alpha member"]


async def test_the_single_workspace_path_refuses_a_query_naming_several(
    engine: AsyncEngine,
) -> None:
    """``ask`` and ``research`` reach retrieval only through ``retrieve``, which spans nothing.

    Without the refusal the query would reach each stage's store, and whether it failed would
    depend on every one of them checking the workspace set — which a store a plugin supplies
    need not do.
    """
    alpha = await _workspace(engine, ALPHA)

    with pytest.raises(ValueError, match="retrieve_across"):
        await _retriever(alpha, ListVectorStore([])).retrieve(_spanning(ALPHA, BETA))


# --- the cache ---------------------------------------------------------------------------------


async def _cached_pair(engine: AsyncEngine) -> tuple[Retriever, list[WorkspaceLeg]]:
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_vectors = ListVectorStore(await _passages(alpha, ALPHA, "a", ["alpha"]), scores=[0.5])
    beta_vectors = ListVectorStore(await _passages(beta, BETA, "b", ["beta"]), scores=[0.9])
    retriever = _retriever(alpha, alpha_vectors, cache=L1QueryCache(entries=16))
    legs = [WorkspaceLeg(ALPHA, alpha, alpha_vectors), WorkspaceLeg(BETA, beta, beta_vectors)]
    return retriever, legs


async def test_a_spanning_search_is_served_from_the_cache_with_its_attribution(
    engine: AsyncEngine,
) -> None:
    """The second identical search is a hit, re-joined per workspace, and still attributed."""
    retriever, legs = await _cached_pair(engine)

    first = await retriever.retrieve_across(_spanning(ALPHA, BETA), legs)
    second = await retriever.retrieve_across(_spanning(ALPHA, BETA), legs)

    assert not first.trace.cached
    assert second.trace.cached
    assert _order(second.candidates) == _order(first.candidates) == ["beta", "alpha"]
    assert second.origins == first.origins


async def test_the_cache_never_serves_one_workspace_s_ranking_for_several_or_back(
    engine: AsyncEngine,
) -> None:
    """The same words, two scopes, two rankings — and the key is what keeps them apart.

    Served to a single-workspace query, a spanning ranking puts another tenant's passage in
    front of a caller who asked about one workspace; served the other way, a search that says
    it spanned two looked at one.
    """
    retriever, legs = await _cached_pair(engine)
    single = _spanning(ALPHA)

    alone = await retriever.retrieve(single)
    spanning = await retriever.retrieve_across(_spanning(ALPHA, BETA), legs)
    alone_again = await retriever.retrieve(single)

    assert not spanning.trace.cached, "a single-workspace ranking was served for two"
    assert "beta" in _order(spanning.candidates)
    assert alone_again.trace.cached
    assert "beta" not in _order(alone_again.candidates), "a spanning ranking was served for one"
    assert _order(alone_again.candidates) == _order(alone.candidates)


def test_the_cache_key_names_the_workspace_set() -> None:
    """The same words under the same pipeline, over different workspaces, are different keys.

    Pinned on the key itself, because two other things also keep the entries apart in practice —
    the spanning pipeline's identity differs from the ordinary one's, and re-hydration refuses an
    entry whose origins this search did not open — and a test through a search would pass on
    either of them with the workspace set dropped from the key. A dense-only pipeline has the
    spanning identity exactly, and there the key is all that separates them.
    """
    identity = PipelineIdentity(stages=("dense",))

    def key(*workspaces: str) -> str:
        return cache_key(_spanning(*workspaces), generation=1, identity=identity)

    assert len({key(ALPHA), key(ALPHA, BETA), key(ALPHA, "gamma"), key(BETA)}) == 4
    assert key(ALPHA, BETA) == key(BETA, ALPHA), "the set, not the order it was named in"


async def test_a_cached_spanning_ranking_re_joins_each_passage_in_its_own_workspace(
    engine: AsyncEngine,
) -> None:
    """A hit re-reads every chunk through the store of the workspace that returned it.

    Driven directly rather than through a second search, because a soft delete commits and the
    commit alone moves the cache key — so an end-to-end test would pass on the key without ever
    exercising the re-join. The re-join is what stands between a cached entry and a passage
    deleted, or never visible, in the workspace it was attributed to.
    """
    alpha, beta = await _workspace(engine, ALPHA), await _workspace(engine, BETA)
    alpha_chunk = (await _passages(alpha, ALPHA, "a", ["alpha"]))[0]
    beta_chunk = (await _passages(beta, BETA, "b", ["beta"]))[0]
    identity = PipelineIdentity(stages=("dense",))
    entry = CachedRanking(
        chunk_ids=(beta_chunk.id, alpha_chunk.id),
        scores=((("dense", 0.9),), (("dense", 0.5),)),
        identity=identity,
        origins=((BETA,), (ALPHA,)),
    )
    joins = {
        ALPHA: (alpha, Filter(workspace_ids=frozenset({ALPHA}))),
        BETA: (beta, Filter(workspace_ids=frozenset({BETA}))),
    }

    rebuilt = await rehydrate_across(entry, joins)
    assert rebuilt is not None
    candidates, origins = rebuilt
    assert _order(candidates) == ["beta", "alpha"]
    assert [candidate.score for candidate in candidates] == [0.9, 0.5]
    assert origins == {beta_chunk.id: (BETA,), alpha_chunk.id: (ALPHA,)}

    # Attributed to the wrong workspace: alpha's store cannot see beta's chunk, so it is stale.
    misattributed = CachedRanking(
        chunk_ids=entry.chunk_ids, scores=entry.scores, identity=identity, origins=((ALPHA,),) * 2
    )
    assert await rehydrate_across(misattributed, joins) is None

    # An origin this search did not open is not one it can vouch for.
    assert await rehydrate_across(entry, {ALPHA: joins[ALPHA]}) is None

    # A single-workspace entry carries no origins at all, and is never one of these.
    unattributed = CachedRanking(chunk_ids=entry.chunk_ids, scores=entry.scores, identity=identity)
    assert await rehydrate_across(unattributed, joins) is None

    await beta.soft_delete_document(beta_chunk.document_id)
    assert await rehydrate_across(entry, joins) is None
