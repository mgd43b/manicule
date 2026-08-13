"""The whole of retrieval, assembled: route, expand, cache, pipeline, context, confidence.

    Query -> router -> membership -> glossary -> L1 cache -> stages -> assembly -> confidence

Five of those seven are not stages, and each is outside the pipeline for a reason rather than by
omission. The router runs before everything and consults nothing. Context assembly emits a
different type, which is exactly what lets the stage list be reordered freely while this step
cannot be. Confidence produces neither candidates nor context — it is a report on the run.

**Glossary expansion is one of them, and it is here for the strongest of the reasons.**
:class:`~manicule.core.protocols.RetrievalStage` is locked, and widening it would invalidate
every recorded evaluation result. Expansion does not need it widened: what it produces is a
*second query*, so the declared pipeline runs over it unchanged, exactly as it ran over the
first. A stage that took one query and searched two would be a stage whose output could not be
replayed from its input.

The second run costs a second pass of the whole pipeline, and it is paid only when an alias
actually fires — which requires the query to name a glossary term as a whole token and to
satisfy the over-expansion rules in :mod:`manicule.retrieval.expansion`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from manicule.core.glossary import QueryExpansion, normalise_acronym
from manicule.core.protocols import CollectionStore, TagStore
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
from manicule.retrieval.expansion import (
    GLOSSARY_SCORE_KEY,
    ExpansionPolicy,
    definitional_frame,
    mark_authoritative,
    merge_rankings,
    resolve_expansion,
)
from manicule.retrieval.hydration import visible_documents
from manicule.retrieval.prefilter import join_filter
from manicule.retrieval.router import QueryRouter, Routing, UtilityKind
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.trace import GlossaryReport, RetrievalTrace, Route
from manicule.retrieval.utility import UtilityAnswer, handlers_for

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.container.container import Container
    from manicule.core.glossary import GlossaryMatch
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Query
    from manicule.retrieval.assembly import ContextAssembler
    from manicule.retrieval.ports import GlossarySource
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
    expansion: QueryExpansion | None = None
    """What the glossary said about this query: which alias fired, to what, from where, and
    what conflicted.

    ``None`` on exactly the runs where :attr:`confidence` is ``None`` — a query the router
    answered without consulting the corpus — and for the same reason: no lookup happened, which
    is a different claim from "a lookup happened and matched nothing". On every retrieval run it
    is present even when nothing fired, because *nothing named*, *expanded* and *conflicting*
    are three states of one object and collapsing two of them into a ``None`` would make a
    conflict indistinguishable from an ordinary query.

    Every match carries the entry it resolved through, and every entry carries the chunk it was
    read out of. That is what makes "never present an expansion without citation provenance" a
    property of the type rather than a rule each surface has to remember."""

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
        glossary: GlossarySource | None = None,
        expansion: ExpansionPolicy | None = None,
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
        self._glossary = glossary
        self._expansion = expansion or ExpansionPolicy()

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
        """Route, expand, retrieve, assemble and score one query."""
        started = time.perf_counter()
        routing = self._router.route(query.text) if self._router else Routing()
        if routing.bypasses_retrieval:
            return await self._direct(query, routing, started)

        # Membership becomes document ids here, before anything else looks at the filter.
        # Neither leg has a join to `collection_documents`, and both refuse the field rather
        # than drop it, so a query naming a collection fails in the store unless it is resolved
        # first. Here rather than in a leg because there are four readers of the filter and
        # only one of them is a leg: the lexical statement, the dense prefilter, the glossary
        # lookup, and the cache -- `_from_cache` rehydrates through `join_filter`, which
        # carries `collection_ids` too. Resolving once, above all four, is also what makes the
        # cache key correct: the key is computed from the resolved filter, so changing a
        # collection's membership changes the key and cannot serve a ranking computed over the
        # old set.
        resolved = await self._resolve_membership(query)
        if resolved is None:
            return self._matches_nothing(query, routing, started)
        query = resolved

        # After membership resolution and before the cache key. After, so the glossary lookup
        # sees the same resolved document ids the legs will -- one notion of what a collection
        # contains rather than a second one resolved separately. Before, because the expansion
        # changes what is retrieved, and a key that could not tell an expanded run from an
        # unexpanded one would serve one as the other.
        expansion = await resolve_expansion(query, self._glossary, self._expansion)

        identity = self.identity(query)
        key = self._key(query, identity, expansion)
        if key is not None:
            hit = await self._from_cache(key, query, identity, started, expansion)
            if hit is not None:
                return hit

        # One frame for the whole query, installed here rather than inside the runner: context
        # assembly is not a stage and runs after the last one, and its report belongs to the
        # same run. The runner reuses an ambient frame when it finds one.
        with tracing.installed() as frame:
            run = await self._runner.run(query)
            candidates = run.candidates
            second: list[Candidate] = []
            if expansion.fired:
                second = (await self._runner.run(_reworded(query, expansion.expanded))).candidates
            promoted, from_store = await self._definitions(query, expansion, candidates, second)
            if expansion.fired or promoted:
                # The merged list is no longer than a single run's would have been. Expansion
                # buys better passages, never more of them: a context that grew whenever a
                # glossary term was named would make every downstream budget a function of the
                # corpus's vocabulary.
                candidates = merge_rankings(
                    run.candidates,
                    second,
                    promoted=promoted,
                    limit=max(len(run.candidates), query.limit),
                )
            context = self._assembler.assemble(query, candidates)
            assembly = frame.assembly
            incomparable = list(frame.incomparable)

        exhausted = _exhausted_budget(run.spans)
        if exhausted:
            incomparable.append(tracing.Shortfall.EXHAUSTED_BUDGET.value)

        if key is not None:
            self._cache.put(
                key,
                L1QueryCache.record(
                    candidates,
                    identity,
                    incomparable=incomparable,
                    exhausted_budget=exhausted,
                ),
            )

        return RetrievalResult(
            context=context,
            candidates=candidates,
            confidence=self._confidence(
                context,
                identity,
                exhausted_budget=exhausted,
                explicit_definition=_cites_a_definition(query, expansion, context),
            ),
            trace=RetrievalTrace(
                route=routing.route,
                pipeline=identity,
                cached=False,
                total_ms=(time.perf_counter() - started) * 1000.0,
                stages=run.spans,
                assembly=assembly,
                glossary=self._report(expansion, promoted, from_store, second_pass=bool(second)),
                incomparable=tuple(dict.fromkeys(incomparable)),
            ),
            routing=routing,
            expansion=expansion,
        )

    async def _definitions(
        self,
        query: Query,
        expansion: QueryExpansion,
        *rankings: Sequence[Candidate],
    ) -> tuple[list[Candidate], int]:
        """The passages that define the terms that fired, in match order.

        Taken from whichever ranking already holds them and fetched by id when neither does —
        which is the case that makes this a *lookup* rather than a boost. On a corpus where an
        acronym is used far more often than it is defined, similarity ranks the usages above
        the definition and the definition is simply not in the candidate set; a feature that
        only reordered what search returned would do nothing at all there.

        A fetched passage is subjected to the same visibility join the dense leg applies, and
        then to the chunk-level restrictions the vocabulary lookup deliberately ignored. The
        entry is a document-level fact and cannot be used to return a chunk the query excluded.

        **One passage per chunk, however many terms it defines.** A glossary page states dozens
        of definitions in one chunk, so a query naming two of them resolves both matches to the
        same passage — and promoting it twice would report ``promoted=2`` for one passage that
        moved, count one fetch as two, and hand the merge a duplicate it only has to collapse
        again. The strongest detection confidence among the terms that landed on it is the one
        recorded, because that is the mark, not a ranking score.

        Returns:
            The promoted candidates, and how many of them neither search had found.
        """
        if not expansion.matches:
            return [], 0
        by_chunk = {candidate.chunk.id: candidate for ranking in rankings for candidate in ranking}
        missing = [match for match in expansion.matches if match.entry.chunk_id not in by_chunk]
        cold = await self._fetch_definitions(query, missing)

        promoted: dict[str, Candidate] = {}
        fetched = 0
        for match in expansion.matches:
            chunk_id = match.entry.chunk_id
            candidate = by_chunk.get(chunk_id) or cold.get(chunk_id)
            if candidate is None:
                continue
            seen = promoted.get(chunk_id)
            if seen is None and chunk_id in cold:
                fetched += 1
            confidence = max(
                match.entry.confidence, seen.scores.get(GLOSSARY_SCORE_KEY, 0.0) if seen else 0.0
            )
            promoted[chunk_id] = mark_authoritative(candidate, confidence)
        return list(promoted.values()), fetched

    async def _fetch_definitions(
        self, query: Query, matches: Sequence[GlossaryMatch]
    ) -> dict[str, Candidate]:
        """Defining chunks neither search returned, fetched by id and then scoped.

        Every restriction a retrieved candidate would have passed is applied here, in the same
        order and through the same helper: the document-level join first — workspace, soft
        delete, status and any post-filter — and then ``kinds`` and ``langs``, which the
        vocabulary lookup does not apply because it returns vocabulary rather than chunks.

        A candidate produced here carries **no leg score**, and that is deliberate rather than
        an omission. Confidence reads the dense leg's key and skips a passage that has none, so
        a promoted definition contributes nothing to the reported confidence in either
        direction — it cannot manufacture evidence, and it cannot be mistaken for a cosine
        nobody measured.
        """
        if not matches:
            return {}
        chunk_ids = list(dict.fromkeys(match.entry.chunk_id for match in matches))
        chunks = await self._docstore.get_chunks(chunk_ids)
        if not chunks:
            return {}
        visible = await visible_documents(
            self._docstore, join_filter(query.filter), [chunk.document_id for chunk in chunks]
        )
        allowed = query.filter
        return {
            chunk.id: Candidate(chunk=chunk, score=0.0)
            for chunk in chunks
            if chunk.document_id in visible
            and (not allowed.kinds or chunk.kind in allowed.kinds)
            and (not allowed.langs or chunk.lang in allowed.langs)
        }

    def _report(
        self,
        expansion: QueryExpansion,
        promoted: Sequence[Candidate],
        from_store: int,
        *,
        second_pass: bool,
    ) -> GlossaryReport | None:
        if self._glossary is None:
            return None
        return GlossaryReport(
            consulted=self._expansion.enabled,
            expanded_query=expansion.expanded,
            terms=tuple(match.key for match in expansion.matches),
            reasons=tuple(match.reason.value for match in expansion.matches),
            conflicts=tuple(conflict.key for conflict in expansion.conflicts),
            promoted=len(promoted),
            promoted_from_store=from_store,
            second_pass=second_pass,
        )

    async def _resolve_membership(self, query: Query) -> Query | None:
        """Turn ``collection_ids`` and ``tag_ids`` into ``document_ids``, or refuse.

        ``None`` means *no document can match*, and it is not the same as an empty result.
        A filter's set-valued fields default to empty and an empty field restricts nothing, so
        resolving an empty collection into ``document_ids=frozenset()`` would hand the legs a
        filter that searches the whole workspace — the narrowest request anyone can make,
        answered with the widest possible result, ranked and plausible. The caller returns no
        candidates instead.

        The import is deferred rather than module-level on purpose. ``manicule.retrieval``
        imports nothing from ``manicule.storage`` — the property ``prefilter``'s docstring
        already protects when it restates a constant rather than importing it — and this is
        the one place that needs a function living over there. Deferring keeps the package's
        module graph as it was, and the cost is paid only by a query that names a collection.

        Raises:
            ValueError: The store cannot resolve membership. Refused rather than dropped: a
                silently ignored restriction returns rows the filter was written to exclude,
                and the search still looks like it worked.
        """
        scope = query.filter
        if not (scope.collection_ids or scope.tag_ids):
            return query

        from manicule.storage.organisation import resolve_filter  # noqa: PLC0415 - only here

        store = self._docstore
        if not isinstance(store, CollectionStore) or not isinstance(store, TagStore):
            named = " and ".join(sorted(scope.restricting_fields & {"collection_ids", "tag_ids"}))
            msg = (
                f"this query restricts on {named}, and the document store behind it resolves "
                f"neither collections nor tags. Refused rather than dropped: applying the rest "
                f"of the filter would return documents the caller asked to exclude, and the "
                f"result would look like an ordinary search."
            )
            # `ValueError`, not the `TypeError` the isinstance test suggests. This is the same
            # refusal `_require_honourable` makes when a store is handed a field it has no
            # column for, and it reaches a caller as one kind of thing: a filter that cannot
            # be honoured here. Splitting it by which layer noticed would make the surfaces
            # report two different errors for one cause.
            raise ValueError(msg)  # noqa: TRY004

        resolved = await resolve_filter(scope, collections=store, tags=store)
        if resolved is None:
            return None
        return query.model_copy(update={"filter": resolved})

    def _matches_nothing(self, query: Query, routing: Routing, started: float) -> RetrievalResult:
        """The answer when membership resolved to no document at all.

        Confidence is *scored* rather than left ``None``. The two are different claims — the
        dataclass says so — and this is the ``none`` band with a reason: we looked, and the
        collection this query names holds nothing that could match.
        """
        context = Context(query=query)
        identity = self.identity(query)
        return RetrievalResult(
            context=context,
            candidates=[],
            confidence=self._confidence(context, identity, exhausted_budget=False),
            trace=RetrievalTrace(
                route=routing.route,
                pipeline=identity,
                cached=False,
                total_ms=(time.perf_counter() - started) * 1000.0,
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

    def _key(
        self, query: Query, identity: PipelineIdentity, expansion: QueryExpansion
    ) -> str | None:
        if not self._cache.enabled:
            return None
        store = self._docstore
        if not isinstance(store, SupportsGeneration):
            return None
        return cache_key(
            query, generation=store.generation, identity=identity, expanded=expansion.expanded
        )

    async def _from_cache(
        self,
        key: str,
        query: Query,
        identity: PipelineIdentity,
        started: float,
        expansion: QueryExpansion,
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
                context,
                entry.identity,
                exhausted_budget=entry.exhausted_budget,
                # The expansion is recomputed on a hit rather than cached, so the classification
                # is computed from the same three facts a miss would use and a cached answer
                # cannot report the contradiction a fresh one no longer can.
                explicit_definition=_cites_a_definition(query, expansion, context),
            ),
            trace=RetrievalTrace(
                route=Route.RETRIEVE,
                pipeline=entry.identity,
                cached=True,
                total_ms=(time.perf_counter() - started) * 1000.0,
                assembly=assembly,
                # The expansion is recomputed on every query rather than cached, so a hit
                # reports the same alias, the same provenance and the same conflicts a miss
                # would have. What it does not report is ``second_pass``: no pipeline ran.
                glossary=self._report(expansion, (), 0, second_pass=False),
                incomparable=(CACHED_RUN, *entry.incomparable),
            ),
            routing=Routing(),
            expansion=expansion,
        )

    def _confidence(
        self,
        context: Context,
        identity: PipelineIdentity,
        *,
        exhausted_budget: bool,
        explicit_definition: bool = False,
    ) -> Confidence:
        """Score the run, passing the legs this pipeline *declares*.

        It used to derive a ``degraded_legs`` list here — the legs no context passage carried a
        score for — and hand that over so their components would be suppressed. That was wrong
        twice. A leg is degraded when it *failed*, and no leg here can fail silently: neither
        catches exceptions, so a leg that returns has run and an empty return means it ran and
        matched nothing. The test also fires when a leg found plenty and none of its hits
        survived into the final few. Both readings turn a fact about the query into a fault of
        the system, and the suppression then *waived the penalty for the queries that earned
        it*: a nonsense query matches no keywords, so it paid no agreement penalty at all, while
        a real question that matched some paid one in full.
        """
        return score_confidence(
            context.passages,
            identity=identity,
            legs=self._legs,
            rerank_stage=self._rerank_stage,
            exhausted_budget=exhausted_budget,
            explicit_definition=explicit_definition,
        )


def _cites_a_definition(query: Query, expansion: QueryExpansion, context: Context) -> bool:
    """Whether this result answers a question about a term by showing that term's definition.

    Three conditions, and dropping any one of them turns a classification into a boost.

    1. **A glossary entry fired for the term**, which already means the entry cleared the
       policy's confidence floor and that no second definition of the term was in scope —
       :func:`~manicule.retrieval.expansion.resolve_expansion` reports disagreement as a conflict
       and produces no match for it. So a contested term never reaches this, and conflicts stay
       visible as conflicts instead of one of them being quietly promoted to "explicit".

    2. **The question asked what the term means.** Tested with
       :func:`~manicule.retrieval.expansion.definitional_frame` rather than by reading
       :class:`~manicule.core.glossary.MatchReason`, and that is a correction rather than a
       preference: the reasons are tried in order and ``exact_case`` wins first, so ``What is
       NOW?`` is recorded as ``exact_case`` and a check for ``definitional_frame`` in the reason
       would miss the central example of the feature. It also excludes the case that matters for
       the specification's requirement 7 — ``restart NOW before the window`` fires ``unambiguous``
       on a term that is not a common word, and a passage defining it is not an answer to that.

    3. **The defining passage is actually in the context.** Promotion can fail: the chunk may be
       outside the query's ``kinds`` or ``langs``, or assembly may have dropped it to fit the
       window. "We found a definition" and "we are showing you a definition" are different
       claims, and only the second may contradict "nothing here resembles your question", because
       only the second puts a counter-example in front of the reader.
    """
    if not expansion.matches:
        return False
    shown = {candidate.chunk.id for candidate in context.passages}
    return any(
        match.entry.chunk_id in shown and definitional_frame(query.text, match.surface)
        for match in expansion.matches
    )


def _reworded(query: Query, text: str) -> Query:
    """The same request, asked the way the glossary says it means.

    Everything but the text is copied, and the filter above all: the second search is subject to
    exactly the restrictions the first was. A rewritten query that widened its own scope would
    be a scope escape reachable by writing an acronym.
    """
    return query.model_copy(update={"text": text})


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
    from manicule.retrieval.ports import GlossarySource  # noqa: PLC0415
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
    # Structural, like every other optional capability retrieval reads off a store. A store
    # that cannot answer glossary lookups is not a defective store; expansion is simply
    # unavailable against it, and the trace says so by carrying no report at all.
    glossary = docstore if isinstance(docstore, GlossarySource) else None

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
        glossary=glossary,
        expansion=ExpansionPolicy(
            enabled=rag.glossary.enabled,
            min_entry_confidence=rag.glossary.min_entry_confidence,
            max_terms=rag.glossary.max_terms,
            homographs=frozenset(
                key for key in (normalise_acronym(word) for word in rag.glossary.homographs) if key
            ),
        ),
    )


__all__ = [
    "CACHED_RUN",
    "CACHE_UNAVAILABLE",
    "RetrievalResult",
    "Retriever",
    "build_retriever",
]
