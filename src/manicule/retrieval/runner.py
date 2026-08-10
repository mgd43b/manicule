"""Running a declared pipeline, and the three things no stage can do for itself.

A stage is a fold over a list. The runner holds the accumulator, calls each stage in declared
order, and:

1. **Times it**, with a monotonic clock around the ``await``.
2. **Counts** what went in and what came out.
3. **Records** the stage's declared configuration, so a recorded result names the thing it
   measured.

Timing from outside is not merely convenient. A stage wrapping its own body in a clock would
measure almost the same interval — but a self-reported number is **unverifiable and optional**.
Every stage gets timed identically, including third-party stages whose authors would never
think to instrument themselves; a stage cannot under-report by measuring only the part it
considers its own work; and there is nothing to check a self-reported figure against.

**Stages run sequentially.** The two legs touch different stores and could overlap, and the
saving is real but small: the lexical leg is one statement, the dense leg contains an embedding
forward pass that dominates it by an order of magnitude, and a reranker dominates both.
Concurrency would buy a few milliseconds and cost unambiguous per-stage attribution —
overlapping wall times do not sum to a pipeline latency, and the whole point of recording them
is that they can be subtracted. If leg latency ever turns out to matter, the shape to reach for
is a combinator stage that runs children concurrently and marks their spans as overlapping.
That is a stage, so it needs no widening.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manicule.core.errors import ConfigError
from manicule.core.retrieval import Candidate
from manicule.retrieval import trace as tracing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Metadata
    from manicule.core.protocols import DocStore, RetrievalStage
    from manicule.core.retrieval import Query


@runtime_checkable
class SupportsDescribe(Protocol):
    """A stage that can state its own configuration for the record.

    Optional, and read only by the runner. A stage that declines simply records nothing, which
    is honest; what it must never do is behave differently because someone is recording.
    """

    def describe(self) -> Metadata:
        """The stage's settings, as JSON-shaped values."""
        ...


@dataclass(slots=True)
class PipelineRun:
    """What one pass through the declared stages produced."""

    candidates: list[Candidate] = field(default_factory=list[Candidate])
    spans: tuple[tracing.StageSpan, ...] = ()
    incomparable: tuple[str, ...] = ()
    wall_ms: float = 0.0

    def diagnostics(self, stage: str) -> Metadata:
        """What one stage recorded, or an empty mapping if it recorded nothing."""
        span = next((span for span in self.spans if span.name == stage), None)
        return dict(span.diagnostics) if span else {}


def require_unique_names(stages: Sequence[RetrievalStage]) -> None:
    """Refuse a pipeline that names one stage twice.

    ``Candidate.scores`` is keyed by stage name, so two stages sharing one means the second
    silently overwrites the first's record — and the fused ranking is then computed from a
    ladder missing half its rungs, which produces a plausible ordering and no error.

    Raises:
        ConfigError: A name appears more than once.
    """
    seen: set[str] = set()
    duplicated: list[str] = []
    for stage in stages:
        if stage.name in seen and stage.name not in duplicated:
            duplicated.append(stage.name)
        seen.add(stage.name)
    if duplicated:
        msg = (
            f"retrieval pipeline names {', '.join(sorted(duplicated))} more than once. Scores "
            f"are keyed by stage name, so the second stage would overwrite the first's record "
            f"and fusion would read a ladder missing half its rungs — a plausible ranking, "
            f"computed from the wrong numbers."
        )
        raise ConfigError(msg)


class _Replay:
    """A stage that returns what the real one already produced.

    The runtime scope assertion re-uses ``assert_pipeline_enforces_scope`` rather than
    reimplementing its rule, and it must check *the candidates that were actually served*
    rather than a second run that may differ. Replaying each stage's recorded output gives
    both: one definition of the rule, applied to the run the caller is about to receive.
    """

    def __init__(self, name: str, output: Sequence[Candidate]) -> None:
        self.name = name
        self._output = list(output)

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query, candidates
        return list(self._output)


class PipelineRunner:
    """Calls a declared list of stages in order, and records what each one did."""

    def __init__(
        self,
        stages: Sequence[RetrievalStage],
        *,
        docstore: DocStore | None = None,
        assert_scope: bool = False,
    ) -> None:
        """Build a runner over ``stages``.

        Args:
            stages: The pipeline, in declared order.
            docstore: The store for the query's workspace. Required when ``assert_scope`` is
                set, because the assertion decides visibility by asking the store rather than
                by consulting a list handed in — the store is the thing retrieval actually
                consults.
            assert_scope: Check, on every query, that no stage emitted a chunk the scope
                excludes. Off by default: it costs a document lookup per candidate per stage.
                On, it holds a live pipeline to the property that makes the vector store's
                ``workspace_ids`` exemption safe, rather than leaving that to the suite.

        Raises:
            ConfigError: Two stages share a name, or the scope assertion was requested with no
                store to ask.
        """
        require_unique_names(stages)
        if assert_scope and docstore is None:
            msg = (
                "rag.assert_scope is on but the runner was given no document store. The "
                "assertion decides what is visible by asking the store, because that is what "
                "retrieval consults; there is nothing to ask."
            )
            raise ConfigError(msg)
        self._stages = list(stages)
        self._docstore = docstore
        self._assert_scope = assert_scope

    @property
    def stages(self) -> tuple[RetrievalStage, ...]:
        return tuple(self._stages)

    @property
    def names(self) -> tuple[str, ...]:
        """The declaration this runner executes. Part of every run's identity."""
        return tuple(stage.name for stage in self._stages)

    async def run(self, query: Query) -> PipelineRun:
        """Fold the query through every stage, recording each one.

        Raises:
            AssertionError: ``assert_scope`` is on and a stage emitted a chunk outside the
                query's scope. Deliberately fatal to the query: a scoped search that returned
                another tenant's chunk has already failed, and returning it with a warning
                attached is the failure this check exists to prevent.
        """
        # An ambient frame belongs to a caller recording more than the stages — context
        # assembly runs after the last stage and reports into the same run. Reusing it rather
        # than nesting is what keeps one query to one trace; a runner used on its own installs
        # its own, so a pipeline can still be run and measured without a retriever around it.
        ambient = tracing.current_frame()
        if ambient is not None:
            return await self._fold(query, ambient)
        with tracing.installed() as frame:
            return await self._fold(query, frame)

    async def _fold(self, query: Query, frame: tracing.TraceFrame) -> PipelineRun:
        started = time.perf_counter()
        first = len(frame.spans)
        candidates: list[Candidate] = []
        outputs: list[tuple[str, list[Candidate]]] = []

        for stage in self._stages:
            before = len(candidates)
            at = time.perf_counter()
            produced = await stage.run(query, list(candidates))
            elapsed = (time.perf_counter() - at) * 1000.0
            frame.spans.append(
                tracing.StageSpan(
                    name=stage.name,
                    wall_ms=elapsed,
                    candidates_in=before,
                    candidates_out=len(produced),
                    config=_describe(stage),
                    diagnostics=frame.take_diagnostics(),
                )
            )
            candidates = produced
            outputs.append((stage.name, produced))

        if self._assert_scope:
            await self._check_scope(query, outputs)
        return PipelineRun(
            candidates=candidates,
            spans=tuple(frame.spans[first:]),
            incomparable=tuple(frame.incomparable),
            wall_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _check_scope(
        self, query: Query, outputs: Sequence[tuple[str, list[Candidate]]]
    ) -> None:
        """Hold this run to the same rule the conformance suite applies.

        ``expect_results=False``, and that parameter is the whole difference between the two
        uses. In the suite, a pipeline that returned nothing has satisfied "returned nothing
        out of scope" without demonstrating anything, so an empty result is a fixture bug. At
        runtime an empty result is an ordinary outcome — the corpus genuinely had nothing — and
        failing on it would turn a normal query into an error.
        """
        # Deferred: manicule.testing is not on the query path, and importing it here would put
        # a test-support module in every serving process's import graph.
        from manicule.testing import assert_pipeline_enforces_scope  # noqa: PLC0415

        if self._docstore is None:  # pragma: no cover - the constructor refuses this
            return
        replay = [_Replay(name, produced) for name, produced in outputs]
        if not replay:
            return
        await assert_pipeline_enforces_scope(replay, self._docstore, query, expect_results=False)


def _describe(stage: object) -> Metadata:
    if isinstance(stage, SupportsDescribe):
        return stage.describe()
    return {}


__all__ = ["PipelineRun", "PipelineRunner", "SupportsDescribe", "require_unique_names"]
