"""A system under comparison: one adapter, one configuration label, one corpus version.

The seam this package is built on. Everything above it — the probe, the pairing, the report —
knows only that a system takes text and returns an ordered list of results, and says what
produced them. Two manicule configurations sit behind it, and so does anything else an
operator wants to compare against: what a system has to supply is an adapter and a label
handed over at runtime, and nothing in this repository names or knows about any particular one.

**Configuration is observed, not declared.** :attr:`SystemResult.configuration` is what
actually produced *this* result, and the harness refuses a run whose sides changed
configuration midway. A declared configuration is a second copy of the truth, and the copy is
the one that goes stale — a pipeline reconfigured between query 30 and query 31 would produce
a file where half the records name something that was not running when they were made, with
nothing in the file saying which half.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.content import Metadata
from manicule.core.retrieval import Filter, Query, RetrievalProfile
from manicule.evaluation.corpus import CorpusVersion

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from manicule.retrieval.retriever import RetrievalResult, Retriever
    from manicule.retrieval.trace import RetrievalTrace

CACHE_MUST_BE_OFF = (
    "the retriever under comparison has its L1 query cache available, and a cache hit is not a "
    "retrieval run: its latency is the cache's and its ranking is a second reading of one "
    "sample. Build the retriever for evaluation with rag.cache.enabled = false"
)
"""Why an otherwise-correct retriever is refused as a system under comparison."""


class ResultItem(BaseModel):
    """One retrieved passage, as a judge sees it and as the record keeps it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    title: str = ""
    text: str = Field(default="", description="What the judge reads. The passage, not a gloss.")
    score: float | None = Field(
        default=None,
        description="Whatever the system called a score. Recorded and never compared across "
        "sides: two systems' scores share a scale only by coincidence.",
    )
    location: str = Field(
        default="", description="Where this came from, in whatever form the system can state."
    )


class StageObservation(BaseModel):
    """One stage's turn, carried through from the run's trace.

    This is what makes per-stage attribution nearly free. A pipeline is a declared list of
    uniform stages, so two configurations that differ in one place produce records that differ
    in one stage — and the report can name it rather than leaving "something changed".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    wall_ms: float = Field(ge=0.0)
    candidates_in: int = Field(ge=0)
    candidates_out: int = Field(ge=0)
    config: Metadata = Field(default_factory=dict)


class SystemResult(BaseModel):
    """What one system returned for one query, with everything needed to read it later."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_label: str = Field(min_length=1)
    configuration: Metadata = Field(
        default_factory=dict,
        description="The configuration that produced this result. Observed from the run where "
        "the system can report it.",
    )
    corpus_version: CorpusVersion
    items: tuple[ResultItem, ...] = ()
    stages: tuple[StageObservation, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0.0)
    incomparable: tuple[str, ...] = Field(
        default=(),
        description="Why this run may not be counted as a measurement — a cache hit, a "
        "degraded leg, a search that stopped at its own budget. Non-empty means the pairing is "
        "recorded and excluded from the rate, with the reason kept.",
    )

    @property
    def document_ids(self) -> tuple[str, ...]:
        """The documents returned, in rank order, with repeats kept.

        Repeats are kept because two chunks of one document are two results: collapsing them
        would make a system that returned five chunks of the same page look like it returned
        one, which is a judgment about diversity and not this type's to make.
        """
        return tuple(item.document_id for item in self.items)


@runtime_checkable
class SystemUnderComparison(Protocol):
    """Something that answers a query, and can say what it is and what it holds.

    Three members and no more. A wider protocol would be a protocol only manicule can satisfy,
    and the comparison that matters most is against something manicule did not build.
    """

    @property
    def config_label(self) -> str:
        """A short human name for this side. Appears in every record and in the report."""
        ...

    @property
    def corpus_version(self) -> CorpusVersion:
        """What this system is searching. Checked against the other side before anything runs."""
        ...

    async def search(self, text: str, *, limit: int) -> SystemResult:
        """Retrieve for one query. Best first."""
        ...


class CallableSystem:
    """A system under comparison that is just an async callable and a label.

    The adapter for anything this repository does not own: a running service reached over HTTP,
    a command-line tool, a second machine. The operator writes the call, this times it and
    puts the label and the corpus version on the result.

    Deliberately the smallest possible surface. Anything more would be this package having an
    opinion about what the other system is, and the whole design here is that it does not need
    one.
    """

    def __init__(
        self,
        search: Callable[[str, int], Awaitable[Sequence[ResultItem]]],
        *,
        config_label: str,
        corpus_version: CorpusVersion,
        configuration: Metadata | None = None,
    ) -> None:
        self._search = search
        self._config_label = config_label
        self._corpus_version = corpus_version
        self._configuration: Metadata = dict(configuration or {})

    @property
    def config_label(self) -> str:
        return self._config_label

    @property
    def corpus_version(self) -> CorpusVersion:
        return self._corpus_version

    async def search(self, text: str, *, limit: int) -> SystemResult:
        started = time.perf_counter()
        items = await self._search(text, limit)
        elapsed = (time.perf_counter() - started) * 1000.0
        return SystemResult(
            config_label=self._config_label,
            configuration=dict(self._configuration),
            corpus_version=self._corpus_version,
            items=tuple(items),
            latency_ms=elapsed,
        )


class RetrieverSystem:
    """A manicule pipeline as a system under comparison.

    Two things it does beyond calling ``retrieve``, and both are refusals rather than features.

    **It will not run against a retriever whose cache can hit.** A cached ranking is the same
    sample counted twice at the cache's latency, so a run that quietly served half its queries
    from memory would report a latency improvement that is an artifact and a quality figure
    computed from half as many observations as it claims. The retriever already knows whether a
    hit is possible — configuration alone is not enough, since a store with no generation
    counter disables the cache regardless — so that is what is checked.

    **It records the run's own identity as the configuration.** Straight off the trace, so the
    stage list, the fusion constant, the reranker and the embedding fingerprint in the record
    are the ones that ran.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        config_label: str,
        corpus_version: CorpusVersion,
        workspace_ids: frozenset[str],
        profile: RetrievalProfile = RetrievalProfile.BALANCED,
    ) -> None:
        if retriever.cache_available:
            raise ValueError(CACHE_MUST_BE_OFF)
        self._retriever = retriever
        self._config_label = config_label
        self._corpus_version = corpus_version
        self._filter = Filter(workspace_ids=workspace_ids)
        self._profile = profile

    @property
    def config_label(self) -> str:
        return self._config_label

    @property
    def corpus_version(self) -> CorpusVersion:
        return self._corpus_version

    async def search(self, text: str, *, limit: int) -> SystemResult:
        query = Query(text=text, limit=limit, filter=self._filter, profile=self._profile)
        result = await self._retriever.retrieve(query)
        trace = result.trace
        items = tuple(
            ResultItem(
                document_id=candidate.chunk.document_id,
                chunk_id=candidate.chunk.id,
                text=candidate.chunk.text,
                score=candidate.score,
                location=" > ".join(candidate.chunk.heading_path),
            )
            for candidate in result.candidates[:limit]
        )
        # The route is deliberately *not* folded in here. It is a property of the query, not
        # of the configuration: a query set containing "hello" would otherwise make the
        # configuration differ between two queries of one run, and the harness would refuse the
        # run with a message about a pipeline that changed — a misdiagnosis of a query set doing
        # something entirely ordinary. Where the route matters, it matters as a reason this
        # pairing is not a measurement, which is the line below.
        configuration: Metadata = dict(trace.pipeline.model_dump(mode="json"))
        configuration["limit"] = limit
        return SystemResult(
            config_label=self._config_label,
            configuration=configuration,
            corpus_version=self._corpus_version,
            items=items,
            stages=tuple(
                StageObservation(
                    name=span.name,
                    wall_ms=span.wall_ms,
                    candidates_in=span.candidates_in,
                    candidates_out=span.candidates_out,
                    config=dict(span.config),
                )
                for span in trace.stages
            ),
            latency_ms=trace.total_ms,
            incomparable=_incomparable(result, trace),
        )


def _routed_away(route: str) -> str:
    """Why a query the router answered directly is not a retrieval measurement."""
    return (
        f"the router answered this directly ({route}), so the corpus was never consulted and "
        f"the empty result is not a retrieval failure"
    )


def _incomparable(result: RetrievalResult, trace: RetrievalTrace) -> tuple[str, ...]:
    """Every reason this run may not count towards a rate.

    The trace's own reasons — a degraded leg, a cache hit, a search stopped by its own budget —
    plus the one the trace records without calling incomparable: a query that never reached
    retrieval at all. Left uncounted, such a pairing is two empty lists a judge scores as
    "neither", which reads as both systems failing on a question neither was asked.
    """
    reasons = list(trace.incomparable)
    if not result.cites_the_corpus:
        reasons.append(_routed_away(trace.route.value))
    return tuple(reasons)


__all__ = [
    "CACHE_MUST_BE_OFF",
    "CallableSystem",
    "ResultItem",
    "RetrieverSystem",
    "StageObservation",
    "SystemResult",
    "SystemUnderComparison",
]
