"""The L1 cache: it holds decisions, so a hit cannot serve what a query may not see."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import delete, false, insert, text, update

from manicule.core.content import BlockKind
from manicule.core.retrieval import Candidate, Filter, PipelineIdentity, Query, SupportsGeneration
from manicule.generation.ports import Feedback, StoredMessage
from manicule.retrieval.cache import L1QueryCache, cache_key, rehydrate
from manicule.retrieval.prefilter import join_filter
from manicule.storage import models
from manicule.storage.conversations import SqliteConversationStore
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import session_factory
from manicule.storage.scoped import NON_CORPUS_TABLES
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


async def test_every_committed_corpus_write_moves_the_counter(store: SqliteDocStore) -> None:
    """It counts *commits*, not calls to the write methods somebody remembered to instrument.

    The same reasoning that puts lexical-index synchronization in triggers rather than in
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


async def test_recording_who_asked_what_leaves_the_counter_alone(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """A search's query-log row, an audit entry, an alert, a conversation and its turns.

    Each is written *because* somebody asked something, after the ranking was computed, and no
    query reads any of them. When these moved the counter, every search through the service
    invalidated its own cached ranking and the next identical search always missed.
    """
    await _live(store)
    settled = store.generation

    async with session_factory(engine).begin() as session:
        session.add(models.QueryLog(id="q1", workspace_id=store.workspace_id, query="auth"))
        session.add(models.AuditLog(id="a1", event_type="search.cross_workspace", details={}))
        session.add(
            models.SecurityAlert(
                id="s1", workspace_id=store.workspace_id, kind="key_abuse", subject="key-1"
            )
        )
    conversations = SqliteConversationStore(engine, workspace_id=store.workspace_id)
    await conversations.ensure_workspace()
    conversation = await conversations.create_conversation(title="auth")
    turn = await conversations.append(
        StoredMessage(conversation_id=conversation, role="assistant", content="Weekly.")
    )
    await conversations.record_feedback(turn, feedback=Feedback.POSITIVE)
    await conversations.rename_conversation(conversation, "rotation")

    assert store.generation == settled


@pytest.mark.parametrize("table", sorted(models.Base.metadata.tables))
async def test_only_the_exempt_tables_can_be_written_without_moving_the_counter(
    store: SqliteDocStore, engine: AsyncEngine, table: str
) -> None:
    """Every table the schema declares, written the way the store writes: a structured statement.

    Both directions from one list, so a table added later is covered the day it is added —
    and it counts, because it is not on the exemption list, which is the direction that costs
    a cold cache rather than a stale ranking. The delete matches no row; the counter is not
    about what a statement did, it is about what it could have done.
    """
    settled = store.generation

    async with engine.begin() as connection:
        await connection.execute(delete(models.Base.metadata.tables[table]).where(false()))

    moved = store.generation > settled
    assert moved is (table not in NON_CORPUS_TABLES)


async def test_raw_sql_counts_even_when_it_names_an_exempt_table(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """A string that looks like an insert into the query log is not a proof that it is one.

    Only a statement SQLAlchemy built has a target the counter can read. A ``text()`` statement
    is classified by nothing but its own claim, and trusting that claim is the wrong direction
    to be wrong in — so it counts, whatever table it names.
    """
    settled = store.generation

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO query_logs (id, workspace_id, query, created_at) "
                "VALUES ('raw', :workspace, 'auth', datetime('now'))"
            ),
            {"workspace": store.workspace_id},
        )

    assert store.generation > settled


async def test_a_record_committed_beside_a_corpus_write_does_not_hide_it(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """One statement the counter cannot vouch for is enough to count the whole transaction."""
    chunks = await _live(store)
    settled = store.generation

    async with session_factory(engine).begin() as session:
        session.add(models.QueryLog(id="q1", workspace_id=store.workspace_id, query="auth"))
        await session.execute(
            update(models.Chunk).where(models.Chunk.id == chunks[0].id).values(text="rewritten")
        )

    assert store.generation > settled


async def test_a_rolled_back_write_neither_counts_nor_lingers(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The mark belongs to one transaction and is cleared when that transaction ends.

    Rolled back, the corpus is as it was, so nothing needs invalidating. And a mark left behind
    would make the *next* commit on the same pooled connection count — a query-log row
    invalidating the cache because of a write that never happened.
    """
    chunks = await _live(store)
    settled = store.generation

    async with engine.connect() as connection:
        await connection.execute(
            update(models.Chunk).where(models.Chunk.id == chunks[0].id).values(text="rewritten")
        )
        await connection.rollback()
        await connection.execute(
            insert(models.QueryLog).values(id="q1", workspace_id=store.workspace_id, query="a")
        )
        await connection.commit()

    assert store.generation == settled


async def test_a_ranking_computed_while_a_commit_is_landing_is_keyed_to_a_value_that_goes(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The counter moves again once the commit has returned, not only when it is announced.

    SQLAlchemy's ``commit`` event fires before the database commits. A search that read the
    counter in that window would read the new value and the old rows, and cache a ranking of a
    corpus about to stop existing under the key every later search uses. Observed here from a
    listener that runs inside the window: whatever it saw, the settled value is past it.
    """
    from sqlalchemy import event  # noqa: PLC0415 - only this test listens

    chunks = await _live(store)
    seen_inside: list[int] = []

    def inside_the_window(_connection: object) -> None:
        seen_inside.append(store.generation)

    event.listen(engine.sync_engine, "commit", inside_the_window)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                update(models.Chunk).where(models.Chunk.id == chunks[0].id).values(text="moved")
            )
    finally:
        event.remove(engine.sync_engine, "commit", inside_the_window)

    assert seen_inside, "the listener never ran, so this proves nothing"
    assert store.generation > seen_inside[-1], (
        "a ranking keyed to the value seen while the commit was landing would outlive it"
    )


async def test_changing_a_collection_s_membership_moves_the_counter(store: SqliteDocStore) -> None:
    """Membership decides what a collection-scoped query may return.

    The key already changes when it does, because it is computed from the membership resolved
    to document ids. The counter moves as well, which is what keeps that true for a reader of
    membership that is ever added somewhere other than the retriever's own resolution.
    """
    chunks = await _live(store)
    collection = await store.create_collection("handbook")
    settled = store.generation

    await store.add_to_collection(collection.id, [chunks[0].document_id])

    assert store.generation > settled


async def test_no_trigger_fires_on_a_table_whose_writes_do_not_count(engine: AsyncEngine) -> None:
    """A trigger is a write the statement that fired it does not name.

    The counter classifies an insert into the query log by its target. A trigger on the query
    log that also wrote to ``chunks`` would be a corpus write that never counted — so the
    exemption holds only while there are none, and this is where adding one fails.
    """
    async with engine.connect() as connection:
        rows = await connection.execute(
            text("SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger'")
        )
        triggers = {(name, table) for name, table in rows}

    assert {(name, table) for name, table in triggers if table in NON_CORPUS_TABLES} == set()
    assert triggers, "the schema has triggers elsewhere; a query returning none proves nothing"


async def test_a_cascade_starting_at_an_exempt_table_ends_at_one(engine: AsyncEngine) -> None:
    """A foreign-key action is the other write a statement does not name.

    Deleting a person cascades into their memberships, sessions and keys, and deleting a query
    log nulls the turns that pointed at it. Each is exempt, so each cascade is too. A cascade
    from an exempt table into one that is not would be a corpus write under an exempt statement.
    """
    async with engine.connect() as connection:
        tables = [
            name
            for (name,) in await connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        ]
        reached: set[tuple[str, str]] = set()
        for child in tables:
            keys = await connection.execute(text(f"PRAGMA foreign_key_list('{child}')"))
            for key in keys.mappings():
                acts = {key["on_delete"], key["on_update"]} - {"NO ACTION", "RESTRICT"}
                if key["table"] in NON_CORPUS_TABLES and acts:
                    reached.add((str(key["table"]), child))

    assert reached, "there are cascades out of exempt tables; finding none would prove nothing"
    assert {(parent, child) for parent, child in reached if child not in NON_CORPUS_TABLES} == set()


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


async def test_a_chunk_level_restriction_does_not_break_rehydration(
    store: SqliteDocStore,
) -> None:
    """A hit re-applies the *document-level* half of the filter, and only that half.

    ``kinds`` and ``langs`` are chunk properties; a query over ``documents`` has no column for
    either and the store refuses to pretend otherwise. Passing the whole filter here would make
    a query that works on a miss raise on a hit — the same query, the same corpus, a different
    outcome depending on cache state.

    Nothing is dropped by narrowing it. Those fields were applied when the ranking was computed,
    and a chunk id is derived from its content, so the chunk behind a cached id is the same
    chunk of the same kind. What can have changed is exactly the document-level half.
    """
    chunks = await _live(store)
    entry = L1QueryCache.record(
        [Candidate(chunk=chunk, score=0.9, scores={"rrf": 0.03}) for chunk in chunks], IDENTITY
    )
    narrowed = Filter(workspace_ids=SCOPE, kinds=frozenset({BlockKind.PROSE}))

    rebuilt = await rehydrate(entry, store, join_filter(narrowed))

    assert rebuilt is not None
    assert len(rebuilt) == len(chunks)


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


async def test_a_connection_invalidated_mid_transaction_rolls_back_cleanly(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The counter's rollback listener must not be what fails a rollback.

    An invalidated connection — a statement cancelled under it, or its driver gone — has no
    ``info`` to read, and asking for it raises. A listener that asked would turn the ordinary
    unwinding of a cancelled request into a ``PendingRollbackError`` raised from inside the
    rollback, and the task would not end cancelled. A mark it cannot clear only ever costs one
    extra bump.
    """
    chunks = await _live(store)
    async with engine.connect() as connection:
        await connection.execute(
            update(models.Chunk).where(models.Chunk.id == chunks[0].id).values(text="half")
        )
        await connection.invalidate()
        await connection.rollback()
    assert store.generation >= 0
