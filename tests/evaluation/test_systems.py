"""The adapter seam: what a side records about itself, and what it refuses to be.

The manicule adapter carries two refusals worth demonstrating rather than asserting — a
retriever whose cache can hit is not a system under comparison, and the configuration on a
record is the one the run reported rather than one supplied alongside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.config.settings import RagSettings
from manicule.evaluation.corpus import CorpusVersion, corpus_version_of
from manicule.evaluation.systems import (
    CACHE_MUST_BE_OFF,
    CallableSystem,
    ResultItem,
    RetrieverSystem,
)
from manicule.retrieval.cache import L1QueryCache
from manicule.retrieval.router import QueryRouter
from manicule.retrieval.utility import handlers_for
from tests.evaluation.fakes import BagOfWordsEmbedder
from tests.evaluation.pipeline import SCOPE, build_corpus, dense_only_retriever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.storage.docstore import SqliteDocStore


async def test_a_result_carries_the_configuration_the_run_reported(
    store: SqliteDocStore,
) -> None:
    """Observed, not declared. A declared configuration is a second copy that goes stale."""
    chunks = await build_corpus(store)
    embedder = BagOfWordsEmbedder()
    system = RetrieverSystem(
        await dense_only_retriever(store, embedder, chunks),
        config_label="dense-only",
        corpus_version=await corpus_version_of(store, label="fixture", workspace_ids=SCOPE),
        workspace_ids=SCOPE,
    )

    result = await system.search("aurora ledger configuration", limit=5)

    assert result.configuration["stages"] == ["dense"]
    assert result.configuration["embed_fingerprint"] == embedder.fingerprint.canonical()
    assert result.items
    assert [stage.name for stage in result.stages] == ["dense"]
    assert result.stages[0].candidates_out >= 1


async def test_a_retriever_whose_cache_can_hit_is_refused(store: SqliteDocStore) -> None:
    """A hit is one sample counted twice at the cache's latency.

    Refused at construction rather than filtered at reporting time, because a run that served
    half its queries from memory would report a latency improvement that is an artefact and a
    quality figure computed from fewer observations than it claims.
    """
    chunks = await build_corpus(store)
    cached = await dense_only_retriever(
        store, BagOfWordsEmbedder(), chunks, cache=L1QueryCache(entries=32)
    )
    assert cached.cache_available, "the fixture must actually have a usable cache"

    with pytest.raises(ValueError, match="cache"):
        RetrieverSystem(
            cached,
            config_label="cached",
            corpus_version=CorpusVersion(label="fixture", document_count=60),
            workspace_ids=SCOPE,
        )
    assert "rag.cache.enabled" in CACHE_MUST_BE_OFF


async def test_an_external_system_is_an_adapter_and_a_label() -> None:
    """The whole requirement on a system this repository did not build.

    No introspection, no shared types beyond the result shape, no assumptions about what it is
    — which is what makes comparing against something else possible at all.
    """
    calls: list[tuple[str, int]] = []

    async def search(text: str, limit: int) -> Sequence[ResultItem]:
        calls.append((text, limit))
        return [ResultItem(document_id="theirs-1", text="an answer")]

    system = CallableSystem(
        search,
        config_label="other-system",
        corpus_version=CorpusVersion(label="fixture", document_count=60),
        configuration={"endpoint": "http://localhost:9000/search"},
    )

    result = await system.search("a question", limit=4)

    assert calls == [("a question", 4)]
    assert result.config_label == "other-system"
    assert result.configuration == {"endpoint": "http://localhost:9000/search"}
    assert result.document_ids == ("theirs-1",)
    assert result.latency_ms >= 0.0


async def test_two_chunks_of_one_document_stay_two_results() -> None:
    """Collapsing them would be a judgement about diversity that this type does not make."""

    async def search(text: str, limit: int) -> Sequence[ResultItem]:
        del text, limit
        return [
            ResultItem(document_id="d1", chunk_id="c1", text="first"),
            ResultItem(document_id="d1", chunk_id="c2", text="second"),
        ]

    system = CallableSystem(
        search,
        config_label="other",
        corpus_version=CorpusVersion(label="fixture", document_count=60),
    )

    result = await system.search("q", limit=10)

    assert result.document_ids == ("d1", "d1")


async def test_a_query_the_router_answers_directly_is_marked_not_a_measurement(
    store: SqliteDocStore,
) -> None:
    """Two bugs live here, and the first hides the second.

    The route is a property of the query, not of the configuration. Recording it as
    configuration made a query set containing "hello" look like a pipeline that changed
    mid-run, and the harness refused the whole session with a message that misdiagnosed it
    completely.

    Underneath that: a routed-away query returns nothing because the corpus was never
    consulted. Left unmarked, the pairing is two empty lists that a judge scores as "neither",
    which reads as both systems failing a question neither was asked.
    """
    chunks = await build_corpus(store)
    retriever = await dense_only_retriever(
        store,
        BagOfWordsEmbedder(),
        chunks,
        router=QueryRouter(RagSettings().router, available=handlers_for(store)),
    )
    system = RetrieverSystem(
        retriever,
        config_label="routed",
        corpus_version=await corpus_version_of(store, label="fixture", workspace_ids=SCOPE),
        workspace_ids=SCOPE,
    )

    retrieved = await system.search("aurora ledger configuration", limit=5)
    greeted = await system.search("hello", limit=5)

    assert greeted.items == ()
    assert any("router answered this directly" in reason for reason in greeted.incomparable)
    assert retrieved.incomparable == ()
    assert greeted.configuration == retrieved.configuration, (
        "the route is not configuration; recording it as such turns an ordinary query set "
        "into a false report of configuration drift"
    )
