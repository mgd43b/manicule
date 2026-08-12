"""What a retrieval run recorded about itself.

Two things live here, and the second is the reason the first is possible without widening
:class:`~manicule.core.protocols.RetrievalStage`.

**The trace.** One :class:`RetrievalTrace` per query, assembled by the runner: the run's
identity, a span per stage, and the assembly report. It exists so that two recorded results
can be compared *honestly*, which means it has to carry the things that make two runs **not**
comparable — a degraded leg, a dense leg stopped by its own budget, a cache hit, a different
pipeline. ``docs/retrieval.md`` §11.

**The frame.** A :class:`contextvars.ContextVar` holding the recorder the runner installed.
The dense stage knows things the runner cannot infer — how far it over-fetched, how many rows
the join removed, whether it exhausted the corpus — and those numbers have to reach the trace
without appearing in a stage's signature. The three ways to do that were: widen the return to
a tuple, which makes every recorded result unreplayable and is exactly the widening
``docs/contracts.md`` §3 warns against; a ``drain_report()`` on the stage, which is per-run
state on a container singleton and therefore a race that swaps two queries' diagnostics; and
this, which is per-task by construction.

The cost of a contextvar is implicit coupling, and it is paid down by one rule:

    **Nothing in the pipeline's behaviour may depend on the frame.**

A stage that returns different candidates when someone is recording is not a stage anybody can
measure. :func:`record` no-ops when no frame is installed, which is what makes the rule cheap
to keep; ``tests/retrieval/test_runner.py`` holds every shipped stage to it by running each
twice over identical input, once observed and once not, and requiring the same candidates —
and demonstrates first, on a stage written to misbehave, that the difference is detectable at
all.
"""

from __future__ import annotations

from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manicule.core.content import Metadata
from manicule.core.retrieval import PipelineIdentity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class Route(StrEnum):
    """Where the router sent a query. See :mod:`manicule.retrieval.router`."""

    RETRIEVE = "retrieve"
    GREETING = "greeting"
    UTILITY = "utility"


class Shortfall(StrEnum):
    """How the dense leg's search ended (``docs/retrieval.md`` §4.4).

    The three are not interchangeable, and collapsing them is the actual bug: two of them are
    ordinary outcomes and one is a defect, and a caller told only "fewer than ``k``" cannot
    tell which it got.
    """

    SATISFIED = "satisfied"
    """``survived >= k``. Nothing to report."""

    EXHAUSTED_CORPUS = "exhausted_corpus"
    """The store holds no more matching rows. The corpus genuinely has that much."""

    EXHAUSTED_BUDGET = "exhausted_budget"
    """The caps stopped the search before the store did. **The result is a floor, not an
    answer**: it counts against confidence and makes the run non-comparable, because a run
    that stopped early ran a different amount of search from one that did not."""


class Regime(StrEnum):
    """Which branch of the pre-filter rule the dense leg took (``docs/retrieval.md`` §3.3)."""

    UNRESOLVED = "unresolved"
    """No join-requiring field was set, so there was nothing to resolve. **Do not constrain.**"""

    PREFILTER = "prefilter"
    """The resolved document ids were small enough to push down. Selectivity is the store's."""

    POSTFILTER = "postfilter"
    """Too many ids to push down: over-fetch and post-filter."""

    EMPTY = "empty"
    """The join-requiring fields resolved to *nothing*, so the filter matches no document.

    Recorded as its own regime rather than folded into :attr:`UNRESOLVED`, because the two
    produce the same empty id set and mean opposite things. Collapsing them is a filter bypass:
    a query restricted to a collection that happens to be empty would return the whole
    workspace, ranked and plausible.
    """


class DenseReport(BaseModel):
    """What the dense leg did, in the numbers §3.3 and §4.3 said would settle their guesses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: int = Field(ge=0, description="``k``: candidates the leg was asked for.")
    fetched: int = Field(ge=0, description="Rows the vector store actually returned.")
    survived: int = Field(ge=0, description="Candidates left after the join and the floor.")
    over_fetch: int = Field(ge=0, description="``k-prime`` on the final attempt.")
    live_fraction: float = Field(
        ge=0.0, le=1.0, description="Live in-workspace chunks over rows in the vector table."
    )
    live_fraction_measured: bool = Field(
        default=True,
        description="Whether the fraction came from a measurement or from the floor. A store "
        "that cannot report its live chunk count changes how many round trips this leg takes "
        "and never what it returns — the retry loop makes up the difference.",
    )
    dropped_by_join: int = Field(
        default=0,
        ge=0,
        description="Rows removed because the document was foreign, deleted, unindexed or "
        "gone. A large number here means the index is dirty.",
    )
    dropped_by_min_score: int = Field(
        default=0,
        ge=0,
        description="Rows removed by the similarity floor. A large number here means the "
        "query is hard — a different diagnosis from the line above, which is why they are "
        "counted apart.",
    )
    expansions: int = Field(default=0, ge=0, description="Retries after an insufficient fetch.")
    outcome: Shortfall = Shortfall.SATISFIED
    regime: Regime = Regime.UNRESOLVED
    resolved_id_count: int = Field(
        default=0, ge=0, description="Documents the join-requiring fields resolved to."
    )
    resolved_id_count_exact: bool = Field(
        default=True,
        description="Whether the count above is the whole answer. Resolution stops one past "
        "``prefilter_id_limit``, so a query against a large corpus never materialises its "
        "document list to answer a question the first thousand rows already answered — and a "
        "count that is really a lower bound says so rather than skewing the distribution the "
        "threshold will eventually be set from.",
    )


class LexicalReport(BaseModel):
    """What the lexical leg matched.

    Zero is an event, not a warning: an all-stopword query and an FTS5 failure look the same
    from here, and either way the pipeline continues on one leg. What must not happen is the
    difference being a line on somebody's terminal — a run that silently became single-leg is
    a run the harness would otherwise compare against a two-leg one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_text: str = Field(description="The text the leg was given. The store owns escaping.")
    requested: int = Field(ge=0)
    matched: int = Field(ge=0)
    degraded: bool = Field(
        default=False, description="The leg contributed nothing, so fusion had one ladder."
    )


class FusionReport(BaseModel):
    """What fusion had to work with. Overlap is the interesting number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    legs: tuple[str, ...] = ()
    k: int = Field(ge=1, description="The RRF constant. Changing it changes every recorded run.")
    per_leg: dict[str, int] = Field(default_factory=dict)
    overlap: int = Field(default=0, ge=0, description="Candidates carrying every leg's score.")
    degraded: bool = Field(
        default=False, description="At least one configured leg contributed no candidates."
    )


class RerankReport(BaseModel):
    """Which model rescored how many pairs. A result that cannot name its reranker cannot be
    reproduced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    pairs: int = Field(ge=0)
    truncated_from: int = Field(
        default=0, ge=0, description="Candidates in, when the head was shorter than the list."
    )


class AssemblyReport(BaseModel):
    """What fitted into the context, what did not, and which tokenizer said so."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokenizer: str = Field(description="Encoding name and safety factor — never a model name.")
    tokens_used: int = Field(ge=0)
    tokens_available: int = Field(ge=0)
    passages: int = Field(ge=0)
    dropped: tuple[tuple[str, int], ...] = Field(
        default=(),
        description="``(chunk id, tokens)`` for each passage skipped for size, in rank order.",
    )


class GlossaryReport(BaseModel):
    """What glossary lookup did to one query.

    On the trace beside :class:`AssemblyReport` rather than in a stage's diagnostics, because
    expansion is not a stage: it runs before the pipeline and produces the *second query* the
    pipeline is then run over. A span would attribute it to whichever stage happened to be
    first.

    **A conflict is recorded even though nothing was expanded**, and that asymmetry is the
    point of the field. "Two definitions disagree" is the single most useful thing this feature
    can tell anybody, and it is the one outcome where the ranking looks exactly like a run in
    which no glossary existed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    consulted: bool = Field(
        default=False,
        description="Whether a glossary was consulted at all. ``False`` covers both 'expansion "
        "is off' and 'no store could answer', which are different from 'consulted and found "
        "nothing' — and a reader trying to work out why an obvious term did not expand needs "
        "to be able to tell them apart.",
    )
    expanded_query: str = Field(
        default="", description="The second query form. Empty when nothing fired."
    )
    terms: tuple[str, ...] = Field(
        default=(), description="The normalised keys that fired, in query order."
    )
    reasons: tuple[str, ...] = Field(
        default=(),
        description="Which rule admitted each fired term, positionally. Recorded because these "
        "are the rules that stop the feature rewriting every ordinary English word, and a rule "
        "nobody can see fire is a rule nobody can check.",
    )
    conflicts: tuple[str, ...] = Field(
        default=(), description="Keys with more than one definition in scope, so none expanded."
    )
    promoted: int = Field(
        default=0, ge=0, description="Definition passages lifted to the head of the ranking."
    )
    promoted_from_store: int = Field(
        default=0,
        ge=0,
        description="Of those, how many neither search returned and had to be fetched by id. "
        "This is the number that says whether the feature is doing anything a better ranking "
        "would have done anyway: zero means similarity already had them.",
    )
    second_pass: bool = Field(
        default=False,
        description="Whether the declared pipeline was run a second time. The cost of the "
        "feature, on the record rather than inferred from a latency that doubled.",
    )


class StageSpan(BaseModel):
    """One stage's turn: how long it took, what went in, what came out, what it was.

    Timed by the runner rather than by the stage, and not because a stage would measure a
    different interval — it would measure almost the same one. A self-reported number is
    **unverifiable and optional**: a third-party stage whose author never thought to
    instrument it gets timed identically, a stage cannot under-report by measuring only the
    part it considers its own work, and there is nothing to check a self-reported figure
    against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    wall_ms: float = Field(ge=0.0)
    candidates_in: int = Field(ge=0)
    candidates_out: int = Field(ge=0)
    config: Metadata = Field(
        default_factory=dict,
        description="The stage's declared configuration, so a recorded result names the thing "
        "it measured.",
    )
    diagnostics: Metadata = Field(
        default_factory=dict,
        description="Whatever the stage recorded through the trace frame. Open rather than a "
        "union of report types, so a plugin stage can record its own numbers without this "
        "module having heard of it.",
    )


class RetrievalTrace(BaseModel):
    """One run, in enough detail for the evaluation harness to refuse a bad comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: Route = Route.RETRIEVE
    pipeline: PipelineIdentity = Field(default_factory=PipelineIdentity)
    cached: bool = Field(
        default=False,
        description="A hit is not a retrieval run. Its latency is the cache's, and a quality "
        "metric computed from one is a metric computed twice from the same sample.",
    )
    total_ms: float = Field(default=0.0, ge=0.0)
    stages: tuple[StageSpan, ...] = ()
    assembly: AssemblyReport | None = None
    glossary: GlossaryReport | None = None
    """``None`` when the retriever has no glossary wired at all, which is a different statement
    from a report saying ``consulted=False``: the first means the capability is absent, the
    second means it was present and declined."""
    incomparable: tuple[str, ...] = Field(
        default=(),
        description="Why this run may not be compared with another. Empty means it may.",
    )

    @property
    def comparable(self) -> bool:
        """Whether this run is admissible as a measurement.

        The harness refuses a comparison when this is false rather than averaging across it.
        Without the refusal, "no retrieval feature without a measured improvement" is a slogan,
        because any two numbers can be subtracted.
        """
        return not self.incomparable

    def span(self, name: str) -> StageSpan | None:
        """The span for one stage, or ``None`` if that stage did not run."""
        return next((span for span in self.stages if span.name == name), None)


class TraceFrame:
    """The recorder a runner installs for the duration of one query.

    Mutable and deliberately not a pydantic model: it is filled in as the run proceeds and
    frozen into a :class:`RetrievalTrace` at the end.
    """

    def __init__(self) -> None:
        self.spans: list[StageSpan] = []
        self.assembly: AssemblyReport | None = None
        self.incomparable: list[str] = []
        self._current: dict[str, object] = {}

    def record(self, **fields: object) -> None:
        """Merge diagnostics into the stage currently running."""
        self._current.update(fields)

    def take_diagnostics(self) -> Metadata:
        """Everything recorded since the last stage ended, and reset for the next one."""
        taken = self._current
        self._current = {}
        return _as_json(taken)

    def not_comparable(self, reason: str) -> None:
        """Record why this run may not be compared with another."""
        if reason not in self.incomparable:
            self.incomparable.append(reason)


_FRAME: ContextVar[TraceFrame | None] = ContextVar("manicule_retrieval_trace", default=None)


def current_frame() -> TraceFrame | None:
    """The frame installed for this task, or ``None`` when nobody is recording."""
    return _FRAME.get()


def record(report: BaseModel | None = None, /, **fields: object) -> None:
    """Record diagnostics for the stage currently running, if anyone is listening.

    A no-op when no frame is installed, which is the whole point: a stage calls this
    unconditionally and behaves identically either way.
    """
    frame = _FRAME.get()
    if frame is None:
        return
    if report is not None:
        frame.record(**report.model_dump(mode="json"))
    if fields:
        frame.record(**fields)


def not_comparable(reason: str) -> None:
    """Mark the run in progress as inadmissible as a measurement. A no-op with no frame."""
    frame = _FRAME.get()
    if frame is not None:
        frame.not_comparable(reason)


def record_assembly(report: AssemblyReport) -> None:
    """Attach the assembly report to the run in progress. A no-op with no frame."""
    frame = _FRAME.get()
    if frame is not None:
        frame.assembly = report


class installed:  # noqa: N801 - a context manager used as a verb, like `contextlib.closing`
    """Install a fresh trace frame for the duration of a block.

    ``ContextVar`` rather than an attribute on anything shared: stages are container
    singletons serving concurrent queries, and per-run state on a shared object swaps two
    queries' diagnostics — plausible numbers attached to the wrong run, which is worse than no
    numbers at all.
    """

    def __init__(self) -> None:
        self.frame = TraceFrame()
        self._token: object = None

    def __enter__(self) -> TraceFrame:
        self._token = _FRAME.set(self.frame)
        return self.frame

    def __exit__(self, *exc: object) -> None:
        del exc
        token = self._token
        if token is not None:
            _FRAME.reset(token)  # pyright: ignore[reportArgumentType] - the token we just set
            self._token = None


def _as_json(values: Mapping[str, object]) -> Metadata:
    """Coerce recorded diagnostics into JSON-shaped values.

    Stages record through typed report models, which already dump to JSON. This exists for the
    loose ``record(key=value)`` form a plugin stage will reach for, so that one stage's stray
    object cannot make a whole trace unserialisable.
    """
    return {name: _json_value(value) for name, value in values.items()}


def _json_value(value: object) -> JsonValue:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, BaseModel):
        return cast("JsonValue", value.model_dump(mode="json"))
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        return {str(key): _json_value(item) for key, item in items.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_value(item) for item in cast("Iterable[object]", value)]
    # Anything else is a stage's stray object. Recorded as its repr rather than dropped or
    # raised: a diagnostic that cannot serialise must not be able to fail a query.
    return repr(value)


__all__ = [
    "AssemblyReport",
    "DenseReport",
    "FusionReport",
    "GlossaryReport",
    "LexicalReport",
    "Regime",
    "RerankReport",
    "RetrievalTrace",
    "Route",
    "Shortfall",
    "StageSpan",
    "TraceFrame",
    "current_frame",
    "installed",
    "not_comparable",
    "record",
    "record_assembly",
]
