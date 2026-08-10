"""The L1 cache: it holds decisions, so a hit cannot serve what a query may not see."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.retrieval import Candidate, Filter, PipelineIdentity, Query, SupportsGeneration
from manicule.retrieval.cache import L1QueryCache, cache_key, rehydrate
from manicule.storage.docstore import SqliteDocStore
from tests.retrieval.fakes import SCOPE, a_query
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk

IDENTITY = PipelineIdentity(stages=("dense", "lexical", "rrf"), rrf_k=60)


async def _live(store: SqliteDocStore) -> list[Chunk]:
    document = make_document(source_id="live")
    await store.upsert_document(document)
    chunks = [make_chunk(document, position, f"authentication {position}") for position in range(3)]
    await store.replace_chunks(document.id, chunks)
    return chunks


def test_the_document_store_reports_a_generation_counter(store: SqliteDocStore) -> None:
    """Without one, a cached ranking cannot be told apart from a stale one."""
    assert isinstance(store, SupportsGeneration)


async def test_every_committed_write_moves_the_counter(store: SqliteDocStore) -> None:
    """It counts *commits*, not calls to the write methods somebody remembered to instrument.

    The same reasoning that puts lexical-index synchronisation in triggers rather than in
    application code: a per-method bump covers only the paths its author enumerated, and the
    one nobody enumerated is the one that serves a stale ranking.
    """
    start = store.generation
    document = make_document(source_id="live")
    await store.upsert_document(document)
    after_upsert = store.generation
    await store.replace_chunks(document.id, [make_chunk(document, 0, "authentication")])
    after_chunks = store.generation
    await store.soft_delete_document(document.id)

    assert start < after_upsert < after_chunks < store.generation


async def test_reading_does_not_move_the_counter(store: SqliteDocStore) -> None:
    """A cache invalidated by its own reads is a cache that never hits."""
    await _live(store)
    settled = store.generation

    await store.list_documents()
    await store.search_lexical("authentication", 5)
    await store.get_document("missing")

    assert store.generation == settled


async def test_a_write_through_another_handle_still_invalidates(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The counter is impossible to bypass, not minimal.

    An ingest run, a repair verb or another workspace's handle writing to the same database
    moves this store's counter too. Over-invalidation costs a cold cache; under-invalidation
    serves a ranking computed over a corpus that no longer exists.
    """
    settled = store.generation
    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()

    assert store.generation > settled


def test_the_key_separates_two_pipelines(store: SqliteDocStore) -> None:
    """Comparing two pipelines is the harness's entire method.

    A cache that cannot tell them apart would serve pipeline A's ranking as pipeline B's
    result, and every difference measured between them would be zero.
    """
    query = a_query()
    one = cache_key(query, generation=1, identity=IDENTITY)
    two = cache_key(query, generation=1, identity=IDENTITY.model_copy(update={"rrf_k": 10}))
    assert one != two


def test_the_key_separates_two_depths() -> None:
    """``Query.limit`` looks like a presentation concern and is not.

    Retrieval depth is the larger of the limit and the profile's head, so a bigger limit is a
    deeper run — and serving a cached ten-result ranking to a request for fifty returns a short
    list that looks like a corpus with nothing more in it.
    """
    shallow = cache_key(a_query(limit=10), generation=1, identity=IDENTITY)
    deep = cache_key(a_query(limit=50), generation=1, identity=IDENTITY)
    assert shallow != deep


def test_the_key_separates_two_filters() -> None:
    """Two filters produce two rankings; a key omitting one answers a different question."""
    plain = Query(text="a", filter=Filter(workspace_ids=SCOPE))
    narrowed = Query(
        text="a", filter=Filter(workspace_ids=SCOPE, sources=frozenset({"confluence"}))
    )
    assert cache_key(plain, generation=1, identity=IDENTITY) != cache_key(
        narrowed, generation=1, identity=IDENTITY
    )


def test_the_key_is_stable_across_set_ordering() -> None:
    """A frozenset has no order, so a key derived from one must not depend on iteration."""
    left = Query(text="a", filter=Filter(workspace_ids=frozenset({"a", "b"})))
    right = Query(text="a", filter=Filter(workspace_ids=frozenset({"b", "a"})))
    assert cache_key(left, generation=1, identity=IDENTITY) == cache_key(
        right, generation=1, identity=IDENTITY
    )


def test_a_generation_bump_invalidates_everything_at_once() -> None:
    """No eviction pass and no per-entry bookkeeping: the counter is in the key."""
    query = a_query()
    assert cache_key(query, generation=1, identity=IDENTITY) != cache_key(
        query, generation=2, identity=IDENTITY
    )


async def test_a_hit_is_rehydrated_through_the_store(store: SqliteDocStore) -> None:
    """The entry holds ids and scores, never text, so the boundary is re-applied on every hit."""
    chunks = await _live(store)
    candidates = [Candidate(chunk=chunk, score=0.9, scores={"rrf": 0.03}) for chunk in chunks]
    entry = L1QueryCache.record(candidates, IDENTITY)

    rebuilt = await rehydrate(entry, store, Filter(workspace_ids=SCOPE))

    assert rebuilt is not None
    assert [candidate.chunk.id for candidate in rebuilt] == [chunk.id for chunk in chunks]
    assert rebuilt[0].scores == {"rrf": 0.03}


async def test_a_hit_cannot_serve_a_soft_deleted_chunk(store: SqliteDocStore) -> None:
    """The failure a content-caching version invites, made impossible rather than avoided."""
    chunks = await _live(store)
    entry = L1QueryCache.record(
        [Candidate(chunk=chunk, score=0.9, scores={"rrf": 0.03}) for chunk in chunks], IDENTITY
    )
    await store.soft_delete_document(chunks[0].document_id)

    assert await rehydrate(entry, store, Filter(workspace_ids=SCOPE)) is None


async def test_a_hit_cannot_serve_another_workspace_s_chunk(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    foreign_document = make_document(source_id="theirs", workspace_id="beta")
    await other.upsert_document(foreign_document)
    foreign = make_chunk(foreign_document, 0, "authentication theirs")
    await other.replace_chunks(foreign_document.id, [foreign])
    entry = L1QueryCache.record(
        [Candidate(chunk=foreign, score=0.9, scores={"rrf": 0.03})], IDENTITY
    )

    assert await rehydrate(entry, store, Filter(workspace_ids=SCOPE)) is None


async def test_a_partially_hydratable_entry_is_stale_rather_than_shortened(
    store: SqliteDocStore,
) -> None:
    """A shortened list would be correct and misleading.

    The ranking was computed over a candidate set that no longer exists, and the candidate that
    would have replaced the dropped one was never considered — so the honest answer is to run
    the pipeline again.
    """
    chunks = await _live(store)
    entry = L1QueryCache.record(
        [Candidate(chunk=chunk, score=0.9, scores={"rrf": 0.03}) for chunk in chunks], IDENTITY
    )
    await store.replace_chunks(chunks[0].document_id, chunks[:2])

    assert await rehydrate(entry, store, Filter(workspace_ids=SCOPE)) is None


def test_entries_are_bounded_and_least_recently_used() -> None:
    cache = L1QueryCache(entries=2)
    for index in range(3):
        cache.put(str(index), L1QueryCache.record([], IDENTITY))

    assert len(cache) == 2
    assert cache.get("0") is None
    assert cache.get("2") is not None


def test_a_zero_entry_cache_stores_nothing() -> None:
    """What an evaluation run sets: a hit is not a retrieval run, and its latency is the
    cache's."""
    cache = L1QueryCache(entries=0)
    cache.put("k", L1QueryCache.record([], IDENTITY))

    assert not cache.enabled
    assert cache.get("k") is None


def test_an_entry_past_its_ttl_is_a_miss() -> None:
    """A bound on staleness from anything the counter was not taught about."""
    cache = L1QueryCache(entries=4, ttl_s=0.0)
    cache.put("k", L1QueryCache.record([], IDENTITY))

    assert cache.get("k") is None


def test_history_is_not_in_the_key() -> None:
    """Retrieval runs on the query text; nothing in this pipeline reads history.

    Including it would guarantee a miss on every turn of a conversation — the one place a user
    actually repeats themselves.
    """
    payload = cache_key(a_query("what about tokens"), generation=1, identity=IDENTITY)
    assert payload == cache_key(a_query("what about tokens"), generation=1, identity=IDENTITY)


def test_a_cached_run_carries_the_defects_of_the_run_behind_it() -> None:
    """A hit reports the identity *and* the incomparability of the run that populated it."""
    entry = L1QueryCache.record([], IDENTITY, incomparable=["degraded"], exhausted_budget=True)
    assert entry.incomparable == ("degraded",)
    assert entry.exhausted_budget


@pytest.mark.parametrize("chunk_ids", [(), ("missing",)])
async def test_rehydration_of_an_absent_chunk(
    store: SqliteDocStore, chunk_ids: tuple[str, ...]
) -> None:
    entry = L1QueryCache.record([], IDENTITY).__class__(
        chunk_ids=chunk_ids, scores=tuple(() for _ in chunk_ids), identity=IDENTITY
    )
    rebuilt = await rehydrate(entry, store, Filter(workspace_ids=SCOPE))
    assert rebuilt == ([] if not chunk_ids else None)
