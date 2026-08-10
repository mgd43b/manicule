"""The whole of retrieval, assembled: route, cache, pipeline, context, confidence.

    Query -> router -> L1 cache -> declared stages -> context assembly -> confidence

Three of those five are not stages, and each is outside the pipeline for a reason rather than
by omission. The router runs before everything and consults nothing. Context assembly emits a
different type, which is exactly what lets the stage list be reordered freely while this step
cannot be. Confidence produces neither candidates nor context — it is a report on the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from manicule.core.retrieval import (
    Candidate,
    Confidence,
    Context,
    PipelineIdentity,
    SupportsGeneration,
)
from manicule.retrieval import trace as tracing
from manicule.retrieval.cache import L1QueryCache, cache_key, rehydrate
from manicule.retrieval.confidence import score_confidence
from manicule.retrieval.prefilter import join_filter
from manicule.retrieval.router import QueryRouter, Routing, UtilityKind
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.trace import RetrievalTrace, Route
from manicule.retrieval.utility import UtilityAnswer, handlers_for

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.container.container import Container
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Query
    from manicule.retrieval.assembly import ContextAssembler
    from manicule.retrieval.profile import Profiles
    from manicule.retrieval.utility import UtilityHandler

CACHE_UNAVAILABLE = (
    "the document store reports no generation counter, so a cached ranking could not be told "
    "apart from a stale one"
)
"""Why the cache is off when configuration asked for it.

A cache without an invalidation signal is not a faster cache, it is a wrong one: it would
serve a ranking computed over a corpus that has since changed and nothing would say so. Off is
the only safe reading of "we cannot tell", and the reason is reported rather than inferred from
a hit rate of zero.
"""

CACHED_RUN = "this result came from the L1 cache, so it is the cache's latency and one sample"


@dataclass(slots=True)
class RetrievalResult:
    """Everything one query produced."""

    context: Context
    candidates: list[Candidate] = field(default_factory=list[Candidate])
    confidence: Confidence | None = None
    """``None`` when the router answered directly.

    Absent rather than 0.0 or 1.0, because "we did not look" and "we looked and there is
    nothing" are different claims and a single number conflates them. The second is the
    ``none`` band with a reason.
    """

    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    routing: Routing = field(default_factory=Routing)
    utility: UtilityAnswer | None = None

    @property
    def cites_the_corpus(self) -> bool:
        """Whether this answer may carry citations at all."""
        return self.routing.route is Route.RETRIEVE


class Retriever:
    """One workspace's retrieval, end to end."""

    def __init__(
        self,
        *,
        runner: PipelineRunner,
        docstore: DocStore,
        assembler: ContextAssembler,
        profiles: Profiles,
        router: QueryRouter | None = None,
        cache: L1QueryCache | None = None,
        legs: Sequence[str] = ("dense", "lexical"),
        rerank_stage: str | None = None,
        rrf_k: int | None = None,
        embed_fingerprint: str | None = None,
        reranker_model_id: str | None = None,
        utility_handlers: Mapping[UtilityKind, UtilityHandler] | None = None,
    ) -> None:
        self._runner = runner
        self._docstore = docstore
        self._assembler = assembler
        self._profiles = profiles
        self._router = router
        # `is None`, not `or`: an empty cache defines ``__len__`` and is therefore falsy, so
        # `cache or ...` silently replaces a perfectly good cold cache with a disabled one and
        # every query is a miss. Nothing raises, nothing is wrong with any answer, and the
        # feature simply does not exist.
        self._cache = L1QueryCache(entries=0) if cache is None else cache
        self._legs = tuple(legs)
        self._rerank_stage = rerank_stage
        self._rrf_k = rrf_k
        self._embed_fingerprint = embed_fingerprint
        self._reranker_model_id = reranker_model_id
        self._utility = dict(utility_handlers or handlers_for(docstore))

    @property
    def cache_available(self) -> bool:
        """Whether a ranking can be cached at all.

        Two conditions, and the second is not configuration: the cache must be switched on,
        *and* the store must be able to say when the index moved.
        """
        return self._cache.enabled and isinstance(self._docstore, SupportsGeneration)

    def identity(self, query: Query) -> PipelineIdentity:
        """What is about to produce this ranking."""
        return PipelineIdentity(
            stages=self._runner.names,
            profile=query.profile,
            overrides=dict(self._profiles.overrides),  # pyright: ignore[reportArgumentType]
            rrf_k=self._rrf_k,
            reranker_model_id=self._reranker_model_id,
            embed_fingerprint=self._embed_fingerprint,
        )

    async def retrieve(self, query: Query) -> RetrievalResult:
        """Route, retrieve, assemble and score one query."""
        started = time.perf_counter()
        routing = self._router.route(query.text) if self._router else Routing()
        if routing.bypasses_retrieval:
            return await self._direct(query, routing, started)

        identity = self.identity(query)
        key = self._key(query, identity)
        if key is not None:
            hit = await self._from_cache(key, query, identity, started)
            if hit is not None:
                return hit

        # One frame for the whole query, installed here rather than inside the runner: context
        # assembly is not a stage and runs after the last one, and its report belongs to the
        # same run. The runner reuses an ambient frame when it finds one.
        with tracing.installed() as frame:
            run = await self._runner.run(query)
            context = self._assembler.assemble(query, run.candidates)
            assembly = frame.assembly
            incomparable = list(frame.incomparable)

        exhausted = _exhausted_budget(run.spans)
        if exhausted:
            incomparable.append(tracing.Shortfall.EXHAUSTED_BUDGET.value)

        if key is not None:
            self._cache.put(
                key,
                L1QueryCache.record(
                    run.candidates,
                    identity,
                    incomparable=incomparable,
                    exhausted_budget=exhausted,
                ),
            )

        return RetrievalResult(
            context=context,
            candidates=run.candidates,
            confidence=self._confidence(context, identity, exhausted_budget=exhausted),
            trace=RetrievalTrace(
                route=routing.route,
                pipeline=identity,
                cached=False,
                total_ms=(time.perf_counter() - started) * 1000.0,
                stages=run.spans,
                assembly=assembly,
                incomparable=tuple(dict.fromkeys(incomparable)),
            ),
            routing=routing,
        )

    async def _direct(self, query: Query, routing: Routing, started: float) -> RetrievalResult:
        """Answer without consulting the corpus, and be visibly different about it.

        No citations, confidence absent, and no cache entry — a directly-routed query is
        already cheap, and the utility ones answer with live counts that a cache would
        staleness-bug for no gain.
        """
        answer: UtilityAnswer | None = None
        if routing.utility is not None:
            handler = self._utility.get(routing.utility)
            if handler is not None:
                answer = await handler(query)
        return RetrievalResult(
            context=Context(query=query),
            confidence=None,
            trace=RetrievalTrace(
                route=routing.route,
                pipeline=self.identity(query),
                total_ms=(time.perf_counter() - started) * 1000.0,
            ),
            routing=routing,
            utility=answer,
        )

    def _key(self, query: Query, identity: PipelineIdentity) -> str | None:
        if not self._cache.enabled:
            return None
        store = self._docstore
        if not isinstance(store, SupportsGeneration):
            return None
        return cache_key(query, generation=store.generation, identity=identity)

    async def _from_cache(
        self, key: str, query: Query, identity: PipelineIdentity, started: float
    ) -> RetrievalResult | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        candidates = await rehydrate(entry, self._docstore, join_filter(query.filter))
        if candidates is None:
            # Anything dropped means the ranking was computed over a candidate set that no
            # longer exists. A shortened list would be a correct answer to a different question.
            self._cache.evict(key)
            return None
        with tracing.installed() as frame:
            context = self._assembler.assemble(query, candidates)
            assembly = frame.assembly
        return RetrievalResult(
            context=context,
            candidates=candidates,
            confidence=self._confidence(
                context, entry.identity, exhausted_budget=entry.exhausted_budget
            ),
            trace=RetrievalTrace(
                route=Route.RETRIEVE,
                pipeline=entry.identity,
                cached=True,
                total_ms=(time.perf_counter() - started) * 1000.0,
                assembly=assembly,
                incomparable=(CACHED_RUN, *entry.incomparable),
            ),
            routing=Routing(),
        )

    def _confidence(
        self, context: Context, identity: PipelineIdentity, *, exhausted_budget: bool
    ) -> Confidence:
        degraded = [
            leg
            for leg in self._legs
            if not any(leg in candidate.scores for candidate in context.passages)
        ]
        return score_confidence(
            context.passages,
            identity=identity,
            legs=self._legs,
            degraded_legs=degraded,
            rerank_stage=self._rerank_stage,
            exhausted_budget=exhausted_budget,
        )


def _exhausted_budget(spans: Sequence[tracing.StageSpan]) -> bool:
    """Whether any stage stopped at its own caps rather than at the end of the corpus."""
    return any(
        span.diagnostics.get("outcome") == tracing.Shortfall.EXHAUSTED_BUDGET.value
        for span in spans
    )


async def build_retriever(container: Container) -> Retriever:
    """Assemble the whole of retrieval from configuration and what plugins registered.

    The composition root for this subsystem, and it lives here rather than as a branch in the
    container for the reason the container's own docstring gives: adding a component means
    writing a plugin, not editing a startup routine. What it does is read configuration and
    resolve components — nothing here decides anything a reader cannot find in ``rag``.

    Two values are read off the built pipeline rather than off configuration, because
    configuration names components and these are properties of the objects: the fusion
    constant and the legs it fuses come from whichever stage is doing the fusing, and the
    reranker's model id comes from the reranker. Both go into the run's identity, so a
    recorded result names what actually ran rather than what was asked for.
    """
    from manicule.container import keys  # noqa: PLC0415 - avoids a package-level import cycle
    from manicule.core.protocols import Reranker  # noqa: PLC0415
    from manicule.retrieval.assembly import ContextAssembler  # noqa: PLC0415
    from manicule.retrieval.fusion import RRFStage  # noqa: PLC0415
    from manicule.retrieval.profile import Profiles  # noqa: PLC0415
    from manicule.retrieval.tokens import ContextTokenCounter  # noqa: PLC0415

    settings = container.settings
    rag = settings.rag
    profiles = Profiles(rag.overrides)
    stages = await container.retrieval_pipeline()
    docstore = await container.aget(keys.DOC_STORE)
    embedder = await container.aget(keys.EMBEDDER)

    fusion = next((stage for stage in stages if isinstance(stage, RRFStage)), None)
    reranker = next((stage for stage in stages if isinstance(stage, Reranker)), None)
    handlers = handlers_for(docstore)

    return Retriever(
        runner=PipelineRunner(stages, docstore=docstore, assert_scope=rag.assert_scope),
        docstore=docstore,
        assembler=ContextAssembler(
            counter=ContextTokenCounter(
                encoding=rag.context.encoding,
                safety_factor=rag.context.safety_factor,
                drift_tolerance=rag.context.drift_tolerance,
            ),
            profiles=profiles,
        ),
        profiles=profiles,
        router=QueryRouter(rag.router, available=handlers) if rag.router.enabled else None,
        cache=L1QueryCache(
            entries=rag.cache.entries if rag.cache.enabled else 0, ttl_s=rag.cache.ttl_s
        ),
        legs=fusion.legs if fusion else (),
        rerank_stage=reranker.name if reranker else None,
        rrf_k=fusion.k if fusion else None,
        embed_fingerprint=embedder.fingerprint.canonical(),
        reranker_model_id=reranker.model_id if reranker else None,
        utility_handlers=handlers,
    )


__all__ = [
    "CACHED_RUN",
    "CACHE_UNAVAILABLE",
    "RetrievalResult",
    "Retriever",
    "build_retriever",
]
