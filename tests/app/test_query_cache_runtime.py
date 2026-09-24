"""The L1 query cache, through the service and the shipped runtime rather than beside them.

The retriever's own tests call it directly, and a direct call writes nothing. Every question the
service answers writes afterwards — a ``query_logs`` row for each search, and a turn in its
conversation for each answer — on the same engine the generation counter listens to. So these
run the whole path: a real data directory, the real ingest pipeline, the real telemetry and
conversation writers, and :class:`~manicule.app.service.ApplicationService` in front. Only the
embedder, the chunker and the model are doubles, and none of them writes anything.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.ids import document_id
from manicule.core.retrieval import Filter, Query
from manicule.plugins.registry import discover
from manicule.storage.scoped import NON_CORPUS_TABLES
from tests.fakes import HashEmbedder
from tests.generation.fakes import ScriptedGenerator

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

WORKSPACE = "alpha"
SOURCE = "notes"
QUESTION = "orchard irrigation"
PASSAGES = {
    "irrigation": "The orchard irrigation schedule runs at dawn.",
    "pruning": "Orchard pruning happens in late winter.",
}


def _runtime(environment: Path) -> Runtime:
    """A runtime on ``environment`` with the cache on, as the shipped configuration has it."""
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415 - fakes, local to this harness

    found = discover()
    bound = found.registry.bind("test")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    bound.add(
        keys.GENERATOR.named("scripted"),
        lambda _: ScriptedGenerator(script=("The schedule runs at dawn.",)),
    )
    settings = Settings(
        data_dir=environment / "data",
        workspace=WORKSPACE,
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        llm={"generator": "scripted"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block"},  # pyright: ignore[reportArgumentType]
    )
    assert settings.rag.cache.enabled, "the shipped default is what is under test"
    return Runtime(settings, discovery=found)


async def _index(opened: Runtime, passages: Mapping[str, str]) -> None:
    from tests.ingest.fakes import DictConnector  # noqa: PLC0415 - fakes, local to this harness

    source = DictConnector(dict(passages), name=SOURCE)
    for source_id in passages:
        source.media_types[source_id] = "text/plain"
    report = await (await opened.pipeline()).run(source)
    assert report.indexed == len(passages)


@pytest.fixture
async def indexed(manicule_environment: Path) -> Path:
    async with _runtime(manicule_environment) as opened:
        await _index(opened, PASSAGES)
    return manicule_environment


async def test_a_repeated_search_is_served_from_the_cache(indexed: Path) -> None:
    """The second identical search is a hit, even though the first one wrote a query-log row.

    The regression this pins: the counter used to count *every* commit on the engine, and the
    service records each search in ``query_logs`` after retrieving it — so the search's own
    telemetry moved the key, and the next identical search always missed. The cache was
    reachable only by calling the retriever directly, which no surface does.
    """
    async with _runtime(indexed) as opened:
        service = ApplicationService(opened)
        first = await service.search(QUESTION, limit=5)
        second = await service.search(QUESTION, limit=5)
        logged = (await (await opened.telemetry()).query_logs())[1]

    assert logged == 2, "both searches must have written their query-log row"
    assert first.cached is False, "the first search was already a hit; the cache is dirty"
    assert second.cached is True, "the first search's own query-log row invalidated the cache"
    assert [hit.chunk_id for hit in second.hits] == [hit.chunk_id for hit in first.hits]


async def test_an_answer_leaves_its_ranking_in_the_cache_for_the_next_search(
    indexed: Path,
) -> None:
    """Answering writes a query-log row and the conversation's turns, and none of it ranks.

    Asked in a conversation and then searched with the same words and depth, so the search's
    key is the one the answer's retrieval stored under. A miss here is an answer's own record of
    itself invalidating the ranking it was built from.
    """
    async with _runtime(indexed) as opened:
        service = ApplicationService(opened)
        conversation = await service.conversation_create(title="irrigation")
        answered = await service.ask(QUESTION, limit=5, conversation_id=conversation.id)
        found = await service.search(QUESTION, limit=5)

    assert answered.message_id, "the answer must have been persisted as a turn"
    assert found.cached is True, "writing the answer's turn invalidated the cache"


async def test_a_document_indexed_between_two_searches_is_found_by_the_second(
    indexed: Path,
) -> None:
    """Exempting telemetry must not exempt the corpus.

    An addition is the case only the counter can catch. A deletion also makes a cached entry
    fail re-hydration, but a new document is simply absent from a cached ranking, and nothing
    about re-hydrating that ranking would notice.
    """
    async with _runtime(indexed) as opened:
        service = ApplicationService(opened)
        before = await service.search(QUESTION, limit=5)
        await _index(opened, {"drainage": "Orchard irrigation drains into the pond."})
        after = await service.search(QUESTION, limit=5)

    added = document_id(WORKSPACE, SOURCE, "drainage")
    assert added not in {hit.document_id for hit in before.hits}
    assert after.cached is False, "a ranking computed before the document existed was served"
    assert added in {hit.document_id for hit in after.hits}


async def test_retrieval_reads_none_of_the_tables_whose_writes_do_not_count(indexed: Path) -> None:
    """The exemption list claims no query reads these tables. This checks the claim.

    Every statement the shipped retriever sends is recorded — both legs, the glossary lookup,
    membership resolution for a collection-scoped query, and the join a hit is re-hydrated
    through — and none may name an exempt table. A retrieval that began reading one, a
    per-person view of a workspace say, would make that table's writes change what a query
    returns, and it would have to leave the list.
    """
    exempt = re.compile(rf"\b({'|'.join(sorted(NON_CORPUS_TABLES))})\b", re.IGNORECASE)
    sent: list[str] = []

    def record(_connection: object, _cursor: object, statement: str, *_rest: object) -> None:
        sent.append(statement)

    async with _runtime(indexed) as opened:
        organization = await opened.organization()
        handbook = await organization.create_collection("handbook")
        await organization.add_to_collection(
            handbook.id, [document_id(WORKSPACE, SOURCE, "irrigation")]
        )
        retriever = await opened.retriever()
        scope = frozenset({WORKSPACE})
        queries = [
            Query(text=QUESTION, limit=5, filter=Filter(workspace_ids=scope)),
            Query(
                text=QUESTION,
                limit=5,
                filter=Filter(workspace_ids=scope, collection_ids=frozenset({handbook.id})),
            ),
        ]
        engine = opened.require_engine().sync_engine
        event.listen(engine, "before_cursor_execute", record)
        try:
            results = [await retriever.retrieve(query) for query in queries for _ in range(2)]
        finally:
            event.remove(engine, "before_cursor_execute", record)

    assert [result.trace.cached for result in results] == [False, True, False, True], (
        "each query must run once and then be re-hydrated, or half the path went unrecorded"
    )
    assert any("collection_documents" in statement for statement in sent), (
        "membership resolution was not recorded, so the collection-scoped path went unchecked"
    )
    assert [statement for statement in sent if exempt.search(statement)] == []
