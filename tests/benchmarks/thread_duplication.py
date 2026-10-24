"""What quoted reply chains cost retrieval over per-message mail documents, measured.

The figures in ``docs/connectors/gmail.md`` §8 come from here, and so does the decision recorded
there. Issue `#463 <https://github.com/mgd43b/manicule/issues/463>`_.

Run it::

    .venv/bin/python -m tests.benchmarks.thread_duplication
    .venv/bin/python -m tests.benchmarks.thread_duplication --model BAAI/bge-small-en-v1.5

**Through the shipped path.** The corpus from :mod:`tests.benchmarks.thread_corpus` is served by
the synthetic Gmail, read by the real Gmail connector, and indexed by
:meth:`~manicule.app.runtime.Runtime.pipeline` — the real mail parser, the shipped structural
chunker, a real ONNX embedding model and the real stores. Each pipeline is then a
:class:`~manicule.app.runtime.Runtime` over the same data directory with only ``rag.pipeline``
changed, queried through :class:`~manicule.evaluation.systems.RetrieverSystem`, which refuses a
retriever whose cache can hit and records the configuration the trace reports.

**Three pipelines, reported apart.** The shipped default (dense, BM25, reciprocal rank fusion),
and each leg alone. BM25's length normalization and a dense model's similarity respond to a
quoted copy differently, and one averaged figure would hide which leg the cost comes from.

**What is measured, per query.** Each query asks for one authored passage, and the generator
knows which message wrote it and which messages quote it. Of the top ``k`` hits:

- *found* — some chunk carrying the passage is there at all;
- *origin* — where the chunk from the message that wrote it ranks against the chunks quoting
  it: ``first``, ``tied`` with a quote ranked above it, ``below`` a quote, or ``quote only``
  when the answer is reachable solely through somebody quoting it;
- *repeats* — chunks carrying the passage beyond the first, which are slots spent showing the
  same answer again;
- *redundant* — chunks that carry nothing the chunks above them did not already carry;
- *companion* — whether the separate note restating the same decision made the page. It is
  the second relevant document, and the one a page full of quotes pushes out.

A chunk *carries* a passage when at least half of the passage's characters fall on lines inside
the chunk's anchor. Where each passage sits in each message is found by parsing the message with
the same parser the pipeline used, and every passage the generator put in a message must be
found there — the run refuses otherwise, rather than scoring against a ground truth it could not
place.

**The decision rule, fixed before the first run.** On the shipped pipeline at ``k = 10``, over
threads of ten or more messages, either of these makes the duplication a cost worth changing
something for:

- (A) a mean of **two or more redundant slots** per query, or companion recall **0.15 or more
  below** the two-message threads;
- (B) the origin **below a quote, or absent while a quote is found**, for **a quarter or more**
  of the queries whose answer was found.

If neither holds: no change. If either holds, the change is collapsing near-duplicate hits
within a thread at retrieval, provided that brings both under their thresholds without lowering
*found*; thread-level documents only if it cannot, because they change what a citation points
at.

Everything is synthetic: generated messages at ``example.org``, a fake Gmail, temporary
directories, no network. The only thing read from outside is the embedding model, from the
local model cache.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from manicule.app.runtime import Runtime
from manicule.cli.render import console
from manicule.config.settings import Settings
from manicule.connectors.gmail_client import PAGE_SIZE
from manicule.core.anchors import LineAnchor
from manicule.core.content import RawDocument
from manicule.core.retrieval import Filter, Query
from manicule.evaluation.corpus import corpus_version_of
from manicule.evaluation.queries import EvalQuery, Intent, Provenance, QuerySet
from manicule.evaluation.statistics import wilson_interval
from manicule.evaluation.systems import RetrieverSystem
from manicule.parsers.config import MailConfig
from manicule.parsers.mail import MailParser
from manicule.retrieval.fusion import DEGRADED_REASON as ONE_LADDER
from manicule.retrieval.lexical import DEGRADED_REASON as LEXICAL_EMPTY
from manicule.retrieval.retriever import Retriever
from tests.benchmarks.thread_corpus import STYLES, Corpus, Message, Passage, build_corpus
from tests.connectors.fake_gmail import FakeGmail, SimulatedTime
from tests.connectors.gmail_support import connector_over

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from manicule.plugins import Discovery
    from manicule.storage.docstore import SqliteDocStore

PIPELINES: dict[str, tuple[str, ...]] = {
    "hybrid": ("dense", "lexical", "rrf"),
    "dense": ("dense",),
    "lexical": ("lexical",),
    "hybrid+collapse": ("dense", "lexical", "rrf", "collapse"),
}
"""Report order. ``hybrid`` is the shipped ``rag.pipeline`` default."""

KS = (5, 10)
SOURCE = "mail"
CARRIES = 0.5
"""The share of a passage's characters a chunk must hold to count as carrying it."""

_QUOTE_MARKERS = re.compile(r"^\s*(?:>\s?)+")


# --- where each passage sits ---------------------------------------------------------------------


async def canonical_lines(raw: bytes) -> dict[int, str]:
    """The message's text by line number, as the mail parser numbers it."""
    document = RawDocument(
        source_id="probe", uri="mail://probe", media_type="message/rfc822", content=raw
    )
    lines: dict[int, str] = {}
    async for block in MailParser(MailConfig()).parse(document):
        anchor = block.anchor
        if not isinstance(anchor, LineAnchor):
            msg = f"a mail block anchored by {type(anchor).__name__}, not by line"
            raise TypeError(msg)
        for offset, text in enumerate(block.text.split("\n")):
            lines[anchor.start + offset] = text
    return lines


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def locate(lines: Mapping[int, str], passages: Iterable[Passage]) -> dict[str, dict[int, int]]:
    """For each passage, how many of its characters fall on each line.

    Quote markers are stripped from every line and whitespace is collapsed, so a passage wrapped
    and quoted three levels deep is found exactly as one written once.

    Raises:
        LookupError: A passage the generator put in this message is not in its parsed text.
    """
    text: list[str] = []
    owner: list[int] = []
    for number in sorted(lines):
        stripped = _normalized(_QUOTE_MARKERS.sub("", lines[number]))
        if not stripped:
            continue
        if text:
            text.append(" ")
            owner.append(number)
        text.append(stripped)
        owner.extend([number] * len(stripped))
    flat = "".join(text)
    found: dict[str, dict[int, int]] = {}
    for passage in passages:
        needle = _normalized(passage.text)
        start = flat.find(needle)
        if start < 0:
            msg = f"passage {passage.id} is not in the parsed message: {needle!r}"
            raise LookupError(msg)
        weights: dict[int, int] = defaultdict(int)
        for position in range(start, start + len(needle)):
            weights[owner[position]] += 1
        found[passage.id] = dict(weights)
    return found


def carried(anchor: LineAnchor, located: Mapping[str, Mapping[int, int]]) -> frozenset[str]:
    """The passages at least :data:`CARRIES` of whose characters lie inside ``anchor``."""
    held: set[str] = set()
    for passage, weights in located.items():
        inside = sum(count for line, count in weights.items() if anchor.start <= line <= anchor.end)
        if inside >= CARRIES * sum(weights.values()):
            held.add(passage)
    return frozenset(held)


# --- one indexed corpus --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkTruth:
    """What the generator knows about one indexed chunk."""

    message: Message
    passages: frozenset[str]


@dataclass
class Indexed:
    """A data directory holding the corpus, and the ground truth for every chunk in it."""

    data_dir: Path
    chunks: dict[str, ChunkTruth]
    documents: int
    chunk_count: int
    seconds: float


def settings_for(
    data_dir: Path, overrides: Mapping[str, Any], pipeline: Sequence[str] = PIPELINES["hybrid"]
) -> Settings:
    """One configuration, differing between pipelines in ``rag.pipeline`` alone.

    ``overrides`` are further ``Settings`` fields — the embedder, and a chunk budget for a
    model that attends to fewer tokens than the default budget.
    """
    return Settings(
        data_dir=data_dir,
        rag={"pipeline": tuple(pipeline), "cache": {"enabled": False}},  # pyright: ignore[reportArgumentType]
        **overrides,
    )


async def index(
    corpus: Corpus,
    data_dir: Path,
    *,
    overrides: Mapping[str, Any],
    discovery: Discovery | None = None,
) -> Indexed:
    """Serve the corpus from the synthetic Gmail and sync it through the shipped pipeline."""
    clock = SimulatedTime()
    server = FakeGmail(clock=clock, page_size=PAGE_SIZE)
    for thread in corpus.threads:
        first = ""
        for message in thread.messages:
            added = server.add_message(message.raw, thread_id=first or None)
            first = first or added.thread_id
            message.gmail_id = added.id
    for message in [*corpus.companions.values(), *corpus.distractors]:
        message.gmail_id = server.add_message(message.raw).id

    started = time.perf_counter()
    async with Runtime(settings_for(data_dir, overrides), discovery=discovery) as runtime:
        connector = connector_over(server, clock, name=SOURCE)
        await connector.setup()
        try:
            report = await (await runtime.pipeline()).run(connector)
        finally:
            await connector.teardown()
        elapsed = time.perf_counter() - started
        messages = corpus.messages
        if report.error or report.indexed != len(messages):
            msg = f"indexed {report.indexed} of {len(messages)} messages: {report.error or '-'}"
            raise RuntimeError(msg)
        store = cast("SqliteDocStore", await runtime.documents())
        truth: dict[str, ChunkTruth] = {}
        for message in messages:
            document = await store.find_document(SOURCE, message.gmail_id)
            if document is None:
                msg = f"message {message.key} was not indexed"
                raise RuntimeError(msg)
            located = locate(await canonical_lines(message.raw), message.carries)
            for chunk in await store.document_chunks(document.id):
                if not isinstance(chunk.anchor, LineAnchor):
                    msg = f"a mail chunk anchored by {type(chunk.anchor).__name__}"
                    raise TypeError(msg)
                truth[chunk.id] = ChunkTruth(message, carried(chunk.anchor, located))
    return Indexed(
        data_dir=data_dir,
        chunks=truth,
        documents=len(messages),
        chunk_count=len(truth),
        seconds=elapsed,
    )


# --- the queries ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Asked:
    """One query, the passage it asks for, and where that passage came from."""

    query: EvalQuery
    passage: Passage
    origin: Message
    companion: Message
    style: str
    length: int
    position: int
    form: str
    """``question`` or ``keywords``: the lexical leg sees only the second."""


FORMS = ("question", "keywords")


def queries(corpus: Corpus) -> tuple[QuerySet, list[Asked]]:
    """Two lookups per queried passage — the first, middle and last message of every thread.

    A natural question, and the same lookup as a few words the passage contains. The lexical leg
    matches a chunk only when it holds every query word that is not a function word, so the
    keywords always reach both legs and the question often reaches the dense leg alone;
    reporting them apart is what shows which leg a cost comes from. When #463 measured this, the
    function words were required too and no question reached the lexical leg at all (#481).
    """
    asked: list[Asked] = []
    for thread in corpus.threads:
        for position in thread.queried:
            origin = thread.messages[position]
            passage = origin.own
            for form, text in zip(FORMS, (passage.query, passage.keywords), strict=True):
                asked.append(
                    Asked(
                        query=EvalQuery(
                            id=f"{passage.id}.{form}",
                            text=text,
                            intent=Intent.LOOKUP,
                            note=f"{thread.style}, message {position + 1} of "
                            f"{len(thread.messages)}",
                        ),
                        passage=passage,
                        origin=origin,
                        companion=corpus.companions[passage.id],
                        style=thread.style,
                        length=len(thread.messages),
                        position=position,
                        form=form,
                    )
                )
    query_set = QuerySet(
        name="mail thread duplication",
        provenance=Provenance.AUTHORED,
        description="Generated by tests/benchmarks/thread_corpus.py: two lookups per passage.",
        queries=tuple(item.query for item in asked),
    )
    return query_set, asked


# --- scoring -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    truth: ChunkTruth


ORIGIN_STATES = ("first", "tied", "below", "quote only", "not found")


@dataclass(frozen=True)
class Outcome:
    """One query's top ``k`` under one pipeline, reduced to what the decision reads."""

    asked: Asked
    pipeline: str
    k: int
    origin: str
    repeats: int
    redundant: int
    companion: bool
    page: int
    """How many hits the page held. A shorter page has fewer repeats for no good reason, so
    this is reported beside them."""
    companion_pool: int | None = None
    """The companion's rank in everything the pipeline returned before the page was cut, or
    ``None`` when it was not among the candidates at all. A stage after fusion can only
    surface what fusion handed it."""
    single_leg: bool = False
    """The lexical leg matched nothing and fusion ran on one ladder. Kept and counted rather
    than refused: for a natural question it is how the shipped pipeline behaves, and the page
    is still the page a person would see."""
    first: int | None = None
    """The rank of the first chunk carrying the passage, quotation or original, or ``None`` when
    the page holds none. What a reciprocal rank is read from."""

    @property
    def found(self) -> bool:
        return self.origin != "not found"


def score(
    asked: Asked,
    hits: Sequence[Hit],
    *,
    pipeline: str,
    k: int,
    companion_pool: int | None = None,
    single_leg: bool = False,
) -> Outcome:
    """Reduce one ranked page to the measurements the module docstring defines."""
    page = list(hits[:k])
    target = asked.passage.id
    copies = [hit for hit in page if target in hit.truth.passages]
    origin = next((hit for hit in copies if hit.truth.message is asked.origin), None)
    if not copies:
        state = "not found"
    elif origin is None:
        state = "quote only"
    elif origin is copies[0]:
        state = "first"
    elif origin.score == copies[0].score:
        state = "tied"
    else:
        state = "below"
    seen: set[str] = set()
    redundant = 0
    for hit in page:
        if hit.truth.passages and hit.truth.passages <= seen:
            redundant += 1
        seen |= hit.truth.passages
    return Outcome(
        asked=asked,
        pipeline=pipeline,
        k=k,
        origin=state,
        repeats=max(0, len(copies) - 1),
        redundant=redundant,
        companion=any(hit.truth.message is asked.companion for hit in page),
        page=len(page),
        companion_pool=companion_pool,
        single_leg=single_leg,
        first=copies[0].rank if copies else None,
    )


@dataclass
class Run:
    """One pipeline's outcomes, the configuration it reported, and what each stage cost."""

    outcomes: list[Outcome]
    configuration: dict[str, Any]
    stage_ms: dict[str, list[float]]


async def run_pipeline(
    indexed: Indexed,
    asked: Sequence[Asked],
    *,
    name: str,
    overrides: Mapping[str, Any],
    discovery: Discovery | None = None,
    ks: Sequence[int] = KS,
    counted: frozenset[str] = frozenset(),
) -> Run:
    """Every query through one pipeline, scored at every ``k``.

    ``counted`` names further reasons a run may not be compared that are kept and scored rather
    than refused, as an empty lexical leg always is. This program passes none; a caller that
    measures how the shipped pipeline answers, rather than comparing two pipelines, passes the
    reasons it accepts as ordinary outcomes.

    The page comes from :class:`~manicule.evaluation.systems.RetrieverSystem`, which refuses a
    run whose configuration changes midway — and so does this. The companion's place in the
    whole candidate list comes from a second call to the same retriever, because the adapter
    returns only the page; the cache is off, so the second call is a second run of the same
    pipeline over the same index.
    """
    settings = settings_for(indexed.data_dir, overrides, PIPELINES[name])
    outcomes: list[Outcome] = []
    configuration: dict[str, Any] | None = None
    stage_ms: dict[str, list[float]] = defaultdict(list)
    async with Runtime(settings, discovery=discovery) as runtime:
        retriever = await runtime.retriever()
        if not isinstance(retriever, Retriever):
            msg = f"the runtime built a {type(retriever).__name__}, not the shipped Retriever"
            raise TypeError(msg)
        scope = frozenset({runtime.workspace})
        version = await corpus_version_of(
            cast("SqliteDocStore", await runtime.documents()),
            label="mail threads",
            workspace_ids=scope,
        )
        system = RetrieverSystem(
            retriever, config_label=name, corpus_version=version, workspace_ids=scope
        )
        for item in asked:
            result = await system.search(item.query.text, limit=max(ks))
            single_leg = any(r.startswith(LEXICAL_EMPTY) for r in result.incomparable)
            refused = [
                r
                for r in result.incomparable
                if not r.startswith(LEXICAL_EMPTY)
                and not (single_leg and r == ONE_LADDER)
                and r not in counted
            ]
            if refused:
                msg = f"{item.query.id} under {name}: {'; '.join(refused)}"
                raise RuntimeError(msg)
            if configuration is None:
                configuration = dict(result.configuration)
            elif dict(result.configuration) != configuration:
                msg = f"the {name} pipeline changed configuration midway, at {item.query.id}"
                raise RuntimeError(msg)
            for stage in result.stages:
                stage_ms[stage.name].append(stage.wall_ms)
            hits = [
                Hit(rank, entry.score or 0.0, indexed.chunks[entry.chunk_id or ""])
                for rank, entry in enumerate(result.items, start=1)
            ]
            pool = await retriever.retrieve(
                Query(text=item.query.text, limit=max(ks), filter=Filter(workspace_ids=scope))
            )
            companion_pool = next(
                (
                    rank
                    for rank, candidate in enumerate(pool.candidates, start=1)
                    if indexed.chunks[candidate.chunk.id].message is item.companion
                ),
                None,
            )
            outcomes.extend(
                score(
                    item,
                    hits,
                    pipeline=name,
                    k=k,
                    companion_pool=companion_pool,
                    single_leg=single_leg,
                )
                for k in ks
            )
    return Run(outcomes, configuration or {}, dict(stage_ms))


# --- the report ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    label: str
    queries: int
    found: float
    origin: dict[str, int]
    repeats: float
    redundant: float
    companions: int
    pooled: int
    """Queries whose companion was among the candidates at all, on the page or not."""
    page: float

    @property
    def companion(self) -> float:
        return self.companions / self.queries

    @property
    def companion_interval(self) -> tuple[float, float]:
        return wilson_interval(self.companions, self.queries)

    @property
    def misplaced_count(self) -> int:
        return self.origin["below"] + self.origin["quote only"]

    @property
    def found_count(self) -> int:
        return self.queries - self.origin["not found"]

    @property
    def misplaced(self) -> float:
        """Rule (B): origin below a quote or absent behind one, over queries that were found."""
        return self.misplaced_count / self.found_count if self.found_count else 0.0

    @property
    def misplaced_interval(self) -> tuple[float, float]:
        return wilson_interval(self.misplaced_count, self.found_count)


def summarize(label: str, outcomes: Sequence[Outcome]) -> Row:
    """One row over ``outcomes``.

    Raises:
        ValueError: There are none — a row of zeros would read as a measurement.
    """
    count = len(outcomes)
    if not count:
        msg = f"no queries for {label!r}: the corpus has no threads of that kind"
        raise ValueError(msg)
    states: dict[str, int] = dict.fromkeys(ORIGIN_STATES, 0)
    for outcome in outcomes:
        states[outcome.origin] += 1
    return Row(
        label=label,
        queries=count,
        found=sum(o.found for o in outcomes) / count,
        origin=states,
        repeats=statistics.fmean(o.repeats for o in outcomes),
        redundant=statistics.fmean(o.redundant for o in outcomes),
        companions=sum(o.companion for o in outcomes),
        pooled=sum(o.companion_pool is not None for o in outcomes),
        page=statistics.fmean(o.page for o in outcomes),
    )


def rows(outcomes: Sequence[Outcome], *, pipeline: str, k: int) -> list[Row]:
    """By thread length, then by style and by query form over the long threads, then overall."""
    chosen = [o for o in outcomes if o.pipeline == pipeline and o.k == k]
    lengths = sorted({o.asked.length for o in chosen})
    table = [
        summarize(f"{length} messages", [o for o in chosen if o.asked.length == length])
        for length in lengths
    ]
    long = [o for o in chosen if o.asked.length >= 10]
    for style in STYLES:
        picked = [o for o in long if o.asked.style == style]
        if picked:
            table.append(summarize(f"{style}, 10+", picked))
    for form in FORMS:
        picked = [o for o in long if o.asked.form == form]
        if picked:
            table.append(summarize(f"{form}, 10+", picked))
    first = [o for o in long if o.asked.position == 0]
    if first:
        table.append(summarize("first message, 10+", first))
    table.append(summarize("all", chosen))
    return table


def render(table: Sequence[Row]) -> str:
    """A Markdown table, ready for the design document."""
    header = (
        "| | queries | page | found | origin first | tied | below | quote only | repeats "
        "| redundant | companion | in pool |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in table:
        lines.append(
            f"| {row.label} | {row.queries} | {row.page:.1f} | {row.found:.2f} | "
            f"{row.origin['first']} | {row.origin['tied']} | {row.origin['below']} | "
            f"{row.origin['quote only']} | {row.repeats:.2f} | {row.redundant:.2f} | "
            f"{row.companion:.2f} | {row.pooled / row.queries:.2f} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class Verdict:
    """The pre-registered rule, applied to one pipeline at ``k = 10``."""

    long: Row
    short: Row

    @property
    def redundant(self) -> float:
        return self.long.redundant

    @property
    def companion_drop(self) -> float:
        return self.short.companion - self.long.companion

    @property
    def misplaced(self) -> float:
        return self.long.misplaced

    @property
    def rule_a(self) -> bool:
        return self.redundant >= 2.0 or self.companion_drop >= 0.15

    @property
    def rule_b(self) -> bool:
        return self.misplaced >= 0.25

    @property
    def change(self) -> bool:
        return self.rule_a or self.rule_b

    def describe(self) -> str:
        def interval(bounds: tuple[float, float]) -> str:
            return f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"

        return "\n".join(
            [
                f"  found {self.long.found:.2f} over threads of 10+, {self.short.found:.2f} "
                f"over 2-message threads",
                f"  rule A: redundant {self.redundant:.2f} (>= 2); companion "
                f"{self.short.companion:.2f} {interval(self.short.companion_interval)} at 2 "
                f"messages against {self.long.companion:.2f} "
                f"{interval(self.long.companion_interval)} at 10+, a drop of "
                f"{self.companion_drop:.2f} (>= 0.15) -> {self.rule_a}",
                f"  rule B: misplaced {self.misplaced:.2f} "
                f"{interval(self.long.misplaced_interval)} (>= 0.25) -> {self.rule_b}",
            ]
        )


def verdict(outcomes: Sequence[Outcome], *, pipeline: str = "hybrid", k: int = 10) -> Verdict:
    chosen = [o for o in outcomes if o.pipeline == pipeline and o.k == k]
    return Verdict(
        long=summarize("threads of 10+", [o for o in chosen if o.asked.length >= 10]),
        short=summarize("2-message threads", [o for o in chosen if o.asked.length == 2]),
    )


# --- the whole measurement -----------------------------------------------------------------------


@dataclass
class Measurement:
    indexed: Indexed
    query_set: QuerySet
    runs: dict[str, Run]

    @property
    def outcomes(self) -> list[Outcome]:
        return [outcome for run in self.runs.values() for outcome in run.outcomes]


async def measure(
    corpus: Corpus,
    data_dir: Path,
    *,
    overrides: Mapping[str, Any],
    discovery: Discovery | None = None,
    pipelines: Sequence[str] = tuple(PIPELINES),
    ks: Sequence[int] = KS,
) -> Measurement:
    indexed = await index(corpus, data_dir, overrides=overrides, discovery=discovery)
    query_set, asked = queries(corpus)
    runs = {
        name: await run_pipeline(
            indexed, asked, name=name, overrides=overrides, discovery=discovery, ks=ks
        )
        for name in pipelines
    }
    return Measurement(indexed, query_set, runs)


def _isolate(work: Path) -> None:
    """Keep an operator's own configuration out of the run, as the suite's fixture does."""
    for name in list(os.environ):
        if name.startswith("MANICULE_"):
            del os.environ[name]
    os.environ["XDG_CONFIG_HOME"] = str(work / "config")
    os.environ["XDG_DATA_HOME"] = str(work / "data-home")
    os.chdir(work)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--model", default="BAAI/bge-m3", help="embedding model (ONNX)")
    parser.add_argument("--seed", type=int, default=463)
    parser.add_argument("--copies", type=int, default=2, help="threads per length and style")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="the structural chunker's budget; needed for a model attending to under 512",
    )
    args = parser.parse_args(argv)

    corpus = build_corpus(seed=args.seed, copies=args.copies)
    overrides: dict[str, Any] = {"embedding": {"provider": "onnx", "model": args.model}}
    if args.chunk_tokens is not None:
        overrides["plugins"] = {"config": {"chunker.structural": {"max_tokens": args.chunk_tokens}}}
    with tempfile.TemporaryDirectory(prefix="thread-duplication-") as scratch:
        work = Path(scratch)
        _isolate(work)
        result = asyncio.run(measure(corpus, work / "index", overrides=overrides))

    out = console()

    def emit(text: str) -> None:
        # The report holds brackets — intervals, stage lists — that are not markup.
        out.print(text, markup=False)

    indexed = result.indexed
    emit(
        f"{indexed.documents} messages, {indexed.chunk_count} chunks, indexed in "
        f"{indexed.seconds:.0f}s with {args.model}; {len(result.query_set.queries)} queries; "
        f"seed {args.seed}"
    )
    outcomes = result.outcomes
    for name, run in result.runs.items():
        stages = run.configuration.get("stages")
        for k in KS:
            single = sum(o.single_leg for o in run.outcomes if o.k == k)
            emit(f"\n### {name} {stages}, k = {k}, {single} queries fused from one leg\n")
            emit(render(rows(outcomes, pipeline=name, k=k)))
        timings = ", ".join(
            f"{stage} {statistics.median(values):.1f}" for stage, values in run.stage_ms.items()
        )
        emit(f"\nmedian stage wall time, ms: {timings}")
    for name in ("hybrid", "hybrid+collapse"):
        if name in result.runs:
            decided = verdict(outcomes, pipeline=name)
            emit(f"\n{name}, k = 10 -> change: {decided.change}\n{decided.describe()}")


if __name__ == "__main__":
    main()
