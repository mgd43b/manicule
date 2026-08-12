"""The whole pipeline, through the container, against real stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.config.settings import QueryCacheSettings, RagSettings, Settings
from manicule.container import keys
from manicule.container.container import Container, check_wiring
from manicule.core.content import BlockKind
from manicule.core.errors import ConfigError
from manicule.core.retrieval import ConfidenceBand, Filter, Query
from manicule.plugins.registry import ComponentRegistry
from manicule.retrieval.assembly import ContextAssembler
from manicule.retrieval.cache import L1QueryCache
from manicule.retrieval.dense import DenseStage
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.lexical import LexicalStage
from manicule.retrieval.plugin import PLUGIN
from manicule.retrieval.profile import Profiles
from manicule.retrieval.rerank import CrossEncoderReranker
from manicule.retrieval.retriever import Retriever, build_retriever
from manicule.retrieval.router import QueryRouter, UtilityKind
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.tokens import ContextTokenCounter
from manicule.retrieval.trace import Route
from manicule.retrieval.utility import handlers_for
from manicule.storage.docstore import SqliteDocStore
from tests.fakes import HashEmbedder
from tests.retrieval.fakes import SCOPE, FixedScorer, ListVectorStore, a_query
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk

SMALL = {"candidates": 3, "final_top_k": 3}


async def _corpus(store: SqliteDocStore) -> list[Chunk]:
    document = make_document(source_id="live")
    await store.upsert_document(document)
    chunks = [
        make_chunk(document, 0, "authentication tokens rotate weekly"),
        make_chunk(document, 1, "authentication is configured in the console"),
        make_chunk(document, 2, "unrelated prose about the weather"),
    ]
    await store.replace_chunks(document.id, chunks)
    return chunks


def _retriever(
    store: SqliteDocStore,
    chunks: list[Chunk],
    *,
    cache: L1QueryCache | None = None,
    rerank: bool = False,
    router: QueryRouter | None = None,
    assert_scope: bool = False,
    **overrides: object,
) -> Retriever:
    resolved = Profiles({**SMALL, **overrides})
    stages: list[object] = [
        DenseStage(
            embedder=HashEmbedder(),
            vectors=ListVectorStore(chunks),
            docstore=store,
            profiles=resolved,
        ),
        LexicalStage(docstore=store, profiles=resolved),
        RRFStage(),
    ]
    rerank_stage = None
    if rerank:
        reranker = CrossEncoderReranker(scorer=FixedScorer({}, default=2.0), profiles=resolved)
        stages.append(reranker)
        rerank_stage = reranker.name
    return Retriever(
        runner=PipelineRunner(stages, docstore=store, assert_scope=assert_scope),  # pyright: ignore[reportArgumentType]
        docstore=store,
        assembler=ContextAssembler(counter=ContextTokenCounter(), profiles=resolved),
        profiles=resolved,
        router=router,
        cache=cache,
        rerank_stage=rerank_stage,
        rrf_k=60,
        embed_fingerprint=HashEmbedder().fingerprint.canonical(),
        reranker_model_id="fake/cross-encoder" if rerank else None,
    )


async def test_a_query_runs_both_legs_fuses_them_and_assembles_a_context(
    store: SqliteDocStore,
) -> None:
    """The shape the whole design is for, end to end."""
    chunks = await _corpus(store)

    result = await _retriever(store, chunks).retrieve(a_query("authentication"))

    assert [span.name for span in result.trace.stages] == ["dense", "lexical", "rrf"]
    assert result.context.passages
    assert result.context.token_count > 0
    assert result.confidence is not None
    # Assembly is not a stage, and it runs after the last one — so its report reaches the trace
    # only because the retriever owns the frame rather than the runner. A trace that lost it
    # would still look complete, which is why this is asserted rather than assumed.
    assert result.trace.assembly is not None
    assert result.trace.assembly.tokenizer.startswith("tiktoken:")
    overlap = [c for c in result.candidates if {"dense", "lexical"} <= set(c.scores)]
    assert overlap, "a chunk both legs found should carry both scores"


async def test_every_stage_s_output_is_already_scoped(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The invariant the whole design rests on, checked at runtime on a live pipeline.

    Enforcing at the end is correct and useless: the excluded rows have already consumed the
    top-``k`` slots. Enforcing at every stage boundary makes it assertable anywhere.
    """
    chunks = await _corpus(store)
    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    foreign_document = make_document(source_id="theirs", workspace_id="beta")
    await other.upsert_document(foreign_document)
    foreign = make_chunk(foreign_document, 0, "authentication theirs")
    await other.replace_chunks(foreign_document.id, [foreign])

    result = await _retriever(store, [*chunks, foreign], assert_scope=True).retrieve(
        a_query("authentication")
    )

    assert all(candidate.chunk.id != foreign.id for candidate in result.candidates)


async def test_a_greeting_never_touches_the_corpus_and_carries_no_confidence(
    store: SqliteDocStore,
) -> None:
    """Absent, not 1.0 and not 0.0: "we did not look" is a different claim from "we looked"."""
    chunks = await _corpus(store)
    router = QueryRouter(RagSettings().router, available=handlers_for(store))

    result = await _retriever(store, chunks, router=router).retrieve(a_query("hello"))

    assert result.trace.route is Route.GREETING
    assert result.confidence is None
    assert result.context.passages == ()
    assert not result.cites_the_corpus


async def test_a_utility_question_is_answered_from_the_index(store: SqliteDocStore) -> None:
    """A utility handler reads a store; the router that chose it did not."""
    chunks = await _corpus(store)
    router = QueryRouter(RagSettings().router, available=handlers_for(store))

    result = await _retriever(store, chunks, router=router).retrieve(a_query("how many documents"))

    assert result.utility is not None
    assert result.utility.kind is UtilityKind.DOCUMENT_COUNT
    assert result.utility.data["documents"] == 1


async def test_a_direct_route_is_never_cached(store: SqliteDocStore) -> None:
    """They are already cheap, and the utility ones answer with live counts."""
    chunks = await _corpus(store)
    cache = L1QueryCache(entries=8)
    router = QueryRouter(RagSettings().router, available=handlers_for(store))

    await _retriever(store, chunks, cache=cache, router=router).retrieve(a_query("hi"))

    assert len(cache) == 0


async def test_a_second_identical_query_is_served_from_the_cache(
    store: SqliteDocStore,
) -> None:
    chunks = await _corpus(store)
    cache = L1QueryCache(entries=8)
    retriever = _retriever(store, chunks, cache=cache)

    first = await retriever.retrieve(a_query("authentication"))
    second = await retriever.retrieve(a_query("authentication"))

    assert not first.trace.cached
    assert second.trace.cached
    assert [c.chunk.id for c in second.candidates] == [c.chunk.id for c in first.candidates]


async def test_a_chunk_level_filter_behaves_the_same_on_a_hit_as_on_a_miss(
    store: SqliteDocStore,
) -> None:
    """The same query and the same corpus must not depend on cache state.

    ``kinds`` reaches the vector store, which has a column for it, and the lexical statement,
    which does too — but a document listing has neither, so re-applying the *whole* filter on a
    hit turns a working query into an error the second time it is asked.
    """
    chunks = await _corpus(store)
    cache = L1QueryCache(entries=8)
    retriever = _retriever(store, chunks, cache=cache)
    query = Query(
        text="authentication",
        limit=3,
        filter=Filter(workspace_ids=SCOPE, kinds=frozenset({BlockKind.PROSE})),
    )

    first = await retriever.retrieve(query)
    second = await retriever.retrieve(query)

    assert second.trace.cached
    assert [c.chunk.id for c in second.candidates] == [c.chunk.id for c in first.candidates]


async def test_a_cache_hit_is_never_counted_as_a_measurement(store: SqliteDocStore) -> None:
    """Its latency is the cache's, and a quality metric from one is one sample counted twice."""
    chunks = await _corpus(store)
    cache = L1QueryCache(entries=8)
    retriever = _retriever(store, chunks, cache=cache)

    await retriever.retrieve(a_query("authentication"))
    second = await retriever.retrieve(a_query("authentication"))

    assert not second.trace.comparable


async def test_a_write_makes_the_next_query_a_miss(store: SqliteDocStore) -> None:
    """The generation counter is in the key, so a commit invalidates everything at once."""
    chunks = await _corpus(store)
    cache = L1QueryCache(entries=8)
    retriever = _retriever(store, chunks, cache=cache)

    await retriever.retrieve(a_query("authentication"))
    await store.soft_delete_document(chunks[0].document_id)
    await store.upsert_document(make_document(source_id="another"))
    after = await retriever.retrieve(a_query("authentication"))

    assert not after.trace.cached


async def test_a_store_with_no_generation_counter_disables_the_cache(
    store: SqliteDocStore,
) -> None:
    """A cache with no invalidation signal is not a faster cache, it is a wrong one."""
    chunks = await _corpus(store)

    class Countless:
        def __init__(self, inner: SqliteDocStore) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            if name == "generation":
                raise AttributeError(name)
            return getattr(self._inner, name)

    retriever = _retriever(Countless(store), chunks, cache=L1QueryCache(entries=8))  # pyright: ignore[reportArgumentType]

    assert not retriever.cache_available
    first = await retriever.retrieve(a_query("authentication"))
    second = await retriever.retrieve(a_query("authentication"))
    assert not first.trace.cached
    assert not second.trace.cached


async def test_a_reranked_run_can_reach_high_confidence_and_an_unreranked_one_cannot(
    store: SqliteDocStore,
) -> None:
    """The most concrete difference between the profiles a user ever sees.

    ``fast`` is the profile that skips the verification step, so it must not be able to claim it
    verified. The ceiling is arithmetic, not a policy applied afterwards.
    """
    chunks = await _corpus(store)

    plain = await _retriever(store, chunks).retrieve(a_query("authentication"))
    reranked = await _retriever(store, chunks, rerank=True).retrieve(a_query("authentication"))

    assert plain.confidence is not None
    assert reranked.confidence is not None
    assert plain.confidence.ceiling == pytest.approx(0.70)
    assert reranked.confidence.ceiling > plain.confidence.ceiling
    assert plain.confidence.band is not ConfidenceBand.HIGH


async def test_a_query_that_matches_nothing_reports_none_with_a_reason(
    store: SqliteDocStore,
) -> None:
    """Distinct from absent: retrieval ran and the corpus had nothing."""
    await _corpus(store)
    query = Query(
        text="authentication",
        limit=3,
        filter=Filter(workspace_ids=SCOPE, sources=frozenset({"nowhere"})),
    )

    result = await _retriever(store, []).retrieve(query)

    assert result.confidence is not None
    assert result.confidence.band is ConfidenceBand.NONE
    assert result.confidence.reason


async def test_the_trace_names_the_pipeline_that_produced_it(store: SqliteDocStore) -> None:
    """Two runs whose identities differ are not two measurements of the same thing."""
    chunks = await _corpus(store)

    result = await _retriever(store, chunks, rerank=True).retrieve(a_query("authentication"))

    assert result.trace.pipeline.stages == ("dense", "lexical", "rrf", "rerank")
    assert result.trace.pipeline.rrf_k == 60
    assert result.trace.pipeline.reranker_model_id == "fake/cross-encoder"
    assert result.trace.pipeline.embed_fingerprint


# --- wiring ---------------------------------------------------------------------------------


def _registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("retrieval"))
    return registry


def _settings(**rag: object) -> Settings:
    return Settings(rag=RagSettings(**rag))  # pyright: ignore[reportArgumentType]


def test_the_plugin_registers_the_two_legs_the_fusion_and_a_reranker() -> None:
    """Built-in components take the same route a third-party plugin takes."""
    registry = _registry()
    assert set(registry.names(keys.RETRIEVAL_STAGE.kind)) == {"dense", "lexical", "rrf"}
    assert "cross_encoder" in registry.names(keys.RERANKER.kind)


def test_fusion_refuses_a_leg_the_pipeline_never_declares() -> None:
    """A typo would otherwise turn two-leg fusion into one-leg fusion silently.

    That is the same failure as a leg returning nothing, with a different cause and the same
    signature — and it would be recorded as a healthy run.
    """
    settings = _settings(pipeline=("dense", "rrf"))
    settings.plugins.config["retrieval_stage.rrf"] = {"legs": ["dense", "lexical"]}
    container = Container(settings, _registry())

    with pytest.raises(ConfigError, match=r"rag\.pipeline does not declare"):
        container.get(keys.RETRIEVAL_STAGE.named("rrf"))


def test_fusion_accepts_the_legs_the_pipeline_does_declare() -> None:
    settings = _settings(pipeline=("dense", "lexical", "rrf"))
    stage = Container(settings, _registry()).get(keys.RETRIEVAL_STAGE.named("rrf"))
    assert isinstance(stage, RRFStage)
    assert stage.legs == ("dense", "lexical")


def test_a_pipeline_naming_one_stage_twice_is_refused_before_construction() -> None:
    """Misconfiguration fails before construction, with the reason named."""
    problems = check_wiring(_settings(pipeline=("dense", "dense", "rrf")), _registry())
    assert any("more than once" in problem for problem in problems)


def test_an_unknown_stage_is_refused_with_the_alternatives_listed() -> None:
    problems = check_wiring(_settings(pipeline=("dense", "sparse", "rrf")), _registry())
    assert any("sparse" in problem and "Available" in problem for problem in problems)


async def test_the_composition_root_assembles_a_working_retriever(
    store: SqliteDocStore, tmp_path: Path
) -> None:
    """Everything wired from configuration and what plugins registered.

    The subsystem has to be assemblable by whoever serves a query without that caller knowing
    which stage does the fusing or which model reranks — both are read off the built pipeline
    rather than off configuration, so a recorded result names what actually ran.
    """
    chunks = await _corpus(store)
    registry = _registry()
    registry.add(keys.EMBEDDER.named("fake"), lambda _: HashEmbedder())
    registry.add(keys.VECTOR_STORE.named("memory"), lambda _: ListVectorStore(chunks))
    registry.add(keys.DOC_STORE.named("bound"), lambda _: store)
    settings = _settings(overrides={"candidates": 3, "final_top_k": 3})
    settings.embedding.provider = "fake"
    settings.storage.vector_db = "memory"  # pyright: ignore[reportAttributeAccessIssue]
    settings.storage.db = "bound"  # pyright: ignore[reportAttributeAccessIssue]
    settings.data_dir = tmp_path

    async with Container(settings, registry) as container:
        retriever = await build_retriever(container)
        result = await retriever.retrieve(a_query("authentication"))

    assert [span.name for span in result.trace.stages] == ["dense", "lexical", "rrf"]
    assert result.trace.pipeline.rrf_k == 60
    assert result.trace.pipeline.embed_fingerprint
    assert result.context.passages
    assert retriever.cache_available


async def test_the_composition_root_honours_the_cache_switch(
    store: SqliteDocStore, tmp_path: Path
) -> None:
    """An evaluation run turns the cache off by configuration, never by a code path."""
    chunks = await _corpus(store)
    registry = _registry()
    registry.add(keys.EMBEDDER.named("fake"), lambda _: HashEmbedder())
    registry.add(keys.VECTOR_STORE.named("memory"), lambda _: ListVectorStore(chunks))
    registry.add(keys.DOC_STORE.named("bound"), lambda _: store)
    settings = _settings(cache=QueryCacheSettings(enabled=False))
    settings.embedding.provider = "fake"
    settings.storage.vector_db = "memory"  # pyright: ignore[reportAttributeAccessIssue]
    settings.storage.db = "bound"  # pyright: ignore[reportAttributeAccessIssue]
    settings.data_dir = tmp_path

    async with Container(settings, registry) as container:
        retriever = await build_retriever(container)

    assert not retriever.cache_available
