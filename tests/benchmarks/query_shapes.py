"""Round 6 of :mod:`tests.benchmarks.lexical_questions`: fusing a short plain lookup differently.

The sixth round on one defect (#486, #489): on this repository's ``docs/``, hybrid ranks a
document's own title below where the dense leg alone ranks it. Round 4 measured fusion
constants, round 5 (:mod:`tests.benchmarks.title_lookups`) the lexical index and a reranker, and
neither found a repair that held under both embedders. Round 5's rule named this round next.
Run it::

    .venv/bin/python -m tests.benchmarks.query_shapes --docs <exported docs/ tree>
    .venv/bin/python -m tests.benchmarks.query_shapes --docs <tree> --provider ollama \\
        --model qwen3-embedding:0.6b --prefix-scheme qwen3 --tokenizer <dir>

**What was known when this rule was written.** Round 4's figures, including a lexical ladder
weighted 0.5 for every query: titles 0.853 to 0.896 under BGE-M3, identifier questions -0.212,
thread keyword lookups 0.915 to 0.878. Round 5's per-title diagnostic and its figures per family,
dense alone and hybrid, under both models, over ``docs/`` exported at ``9ce34e6e`` — the export
this round indexes too, so its incumbent is round 5's. The cause round 5 found: the lexical index
holds a chunk's headings and not its document's title, so for a one-word title the lexical ladder
is other documents' bodies, and fusion lets that ladder outvote the dense leg's first.

**The gate.** A query is a *plain short lookup* when :func:`~manicule.storage.fts.match_words`
finds at most :data:`SHORT` words in it, none of them a function word, and none spelled the way
only code spells things — an underscore, a dot, a digit, a lower-case letter followed by a
capital, or two capitals followed by a lower-case letter (:func:`plain_lookup`). The length was
chosen knowing every title is one or two words; the code-shape exclusion was added after
counting what a length gate alone selects, which was all 756 bare identifiers. What each gate
selects, counted on the export before any candidate ran:

======================  =========  ===================  ====================
group                   questions  plain, at most 2     plain, at most 16
======================  =========  ===================  ====================
titles                         19                   19                    19
headings                      518                   80                   112
thread keyword lookups        126                    0                   126
unanswerable                   11                    0                     3
bare identifiers              756                    0                     0
every other group               —                    0                     0
======================  =========  ===================  ====================

**The candidates**, in tie-break order, each against the shipped fusion:

- ``short-dense`` — a plain short lookup of at most 2 words is fused with no lexical ladder: its
  lexical reciprocal ranks are weighted 0, so the dense leg's order stands.
- ``short-half`` — the same queries with the lexical ladder weighted 0.5.
- ``plain-half`` — every plain lookup of at most 16 words, the match's own cap, weighted 0.5. The
  wider gate, which reaches the thread keyword lookups; round 4's 0.5 for every query cost those
  0.037, so this candidate is expected to fail, and is measured because the expectation is not a
  measurement.

Every other query is fused exactly as shipped. Measured by substituting the fusion stage's
arithmetic for the run, as round 4 did, with the query's text now read; a leg that returned
nothing still marks the run degraded and nothing else does, and every query the gate selected is
recorded by the substitute itself, so the gated subsets below come from the run, not from a
recomputation.

**The questions.** Round 5's, plus *bare identifiers*: every identifier of round 2's set asked as
itself, its own section the target — the lookup the lexical leg exists for, typed bare. The gate
must select none of them; the family is the check that the code-shape exclusion holds.

**The round 6 rule.** Hybrid, ``k = 10``, reciprocal rank paired by query, per model. Families:
*titles*; *headings*; *headings the gate selected*; *prose* (authored, labeled multi-hop,
held-out thread questions); *keyword* (thread keyword lookups); *keyword lookups the gate
selected*; *identifiers* (both question sets); *bare identifiers*. A family the gate selected
nothing of is not scored. A candidate **fixes the defect** under a model when:

1. On titles, it closes at least half the gap to the dense leg alone measured in the same run, and
   no title ranks worse — round 5's criterion, for round 5's reason.
2. In every other scored family, gated subsets included, the lower end of a 95% paired bootstrap
   interval on its mean difference is above -0.02. The gated subsets are scored on their own
   because a change confined to them is diluted in the whole family — 80 headings in 518 — which
   is how round 4 pooled a title loss out of sight.
3. No unanswerable question leaves the ``none`` band.

A qualifier fixes the defect under **both** models. The one with the highest mean difference over
the non-title families, averaged over the two models, is chosen; within 0.005, the earlier in
:data:`CANDIDATES`. No qualifier: no change.

**What a chosen candidate changes.** Two constants on ``FusionConfig`` — the gate's word count and
the gated lexical weight, both recorded in the stage's span config — the gate itself beside the
match builder, a field in ``FusionReport`` saying whether a query was gated, the fusion settings
in the L1 cache key (which today hashes ``rrf_k`` and no other), and ``docs/retrieval.md`` §5.1's
rule of no per-leg weighting, which this would end for one shape of query.

**Round 6 returned "no change".** Nothing fixed the defect under both models.
``short-dense`` brought titles to the dense leg's level or above — 0.844 to 0.934 under BGE-M3
(dense alone 0.934), 0.757 to 0.965 under Qwen3 (0.895) — and failed twice: the 80 headings it
gated moved +0.038 and +0.028 on average but split 16 better to 11 worse and 17 to 12, with lower
bounds of -0.033 and -0.031; and under BGE-M3 one title, ``web``, ranked worse, the one title the
dense leg alone ranks below hybrid. ``short-half`` held every family under BGE-M3 but ranked
``web`` worse too, and under Qwen3 closed under half the gap (0.794). ``plain-half`` cost the
thread keyword lookups -0.037 [-0.066, -0.011] and -0.071 [-0.107, -0.040], as round 4's global
0.5 had. The gate selected 99 distinct queries for the short candidates and 260 for the wide
one, under both models; no other family moved by any amount. Recorded from a checkout pinned at
the commit holding this rule, over the same export as round 5, both models side by side.

Run it from a checkout nobody is editing, over the same export (:func:`title_lookups.unchanged`
refuses an index that moves schema revision mid-run). Everything is local apart from the
embedding models.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import faulthandler
import re
import shutil
import signal
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from manicule.app.runtime import Runtime
from manicule.cli.render import console
from manicule.core.retrieval import ConfidenceBand
from manicule.retrieval import fusion as fusion_module
from manicule.retrieval import trace as trace_module
from manicule.storage.fts import FUNCTION_WORDS, match_words
from tests.benchmarks import lexical_questions as lq
from tests.benchmarks import thread_duplication
from tests.benchmarks import title_lookups as tl
from tests.benchmarks.thread_corpus import build_corpus
from tests.benchmarks.thread_duplication import _isolate  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from manicule.core.retrieval import Candidate, Query
    from manicule.storage.docstore import SqliteDocStore

SHORT = 2
PLAIN = 16
_CODE = re.compile(r"[_.0-9]|[a-z][A-Z]|[A-Z]{2}[a-z]")


def plain_lookup(text: str, most: int) -> bool:
    """At most ``most`` words, no function word, and no word spelled the way code is spelled."""
    words = match_words(text)
    return (
        0 < len(words) <= most
        and not any(word.lower() in FUNCTION_WORDS for word in words)
        and not any(_CODE.search(word) for word in words)
    )


@dataclass(frozen=True)
class Gate:
    """Which queries are gated, and the lexical ladder's weight for them."""

    most: int = 0
    lexical_weight: float = 1.0

    def gated(self, text: str) -> bool:
        return self.most > 0 and plain_lookup(text, self.most)


CANDIDATES: dict[str, Gate] = {
    "shipped": Gate(),
    "short-dense": Gate(most=SHORT, lexical_weight=0.0),
    "short-half": Gate(most=SHORT, lexical_weight=0.5),
    "plain-half": Gate(most=PLAIN, lexical_weight=0.5),
}
"""The incumbent first, then the candidates in tie-break order."""

FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...], bool]] = {
    "titles": (("titles",), (), False),
    "headings": (("headings",), (), False),
    "headings, gated": (("headings",), (), True),
    "prose": (("authored", "multi-hop"), ("question",), False),
    "keyword": ((), ("keywords",), False),
    "keyword, gated": ((), ("keywords",), True),
    "identifiers": (("identifiers", lq.TWO_WORDS), (), False),
    "bare identifiers": (("bare identifiers",), (), False),
}
"""Each family's ``docs/`` groups, mail-thread query forms, and whether only gated queries count."""

GROUPS = (*tl.GROUPS, "bare identifiers")


def bare_identifiers(root: Path) -> list[lq.Asked]:
    """Round 2's identifiers, each asked as itself."""
    return [
        lq.Asked(f"bare-{i}", term, "bare identifiers", (target,))
        for i, (term, target) in enumerate(lq.single_section_identifiers(root))
    ]


# --- the substitute fusion -----------------------------------------------------------------------


@contextlib.contextmanager
def gated_weighting(candidate: Gate, gated: set[str]) -> Generator[None]:
    """Fuse a gated query with its lexical ladder weighted; record every query it gated.

    The shipped stage's arithmetic, ladders, tie-break and report, with one change: the query's
    text decides the lexical leg's weight. A weight of 0 leaves the lexical candidates in the list,
    ordered after everything the dense leg ranked, exactly as fusion orders a zero score.
    """
    if candidate.most == 0:
        yield
        return
    shipped = fusion_module.RRFStage.run

    async def run(
        self: fusion_module.RRFStage, query: Query, candidates: list[Candidate]
    ) -> list[Candidate]:
        weight = 1.0
        if candidate.gated(query.text):
            gated.add(query.text)
            weight = candidate.lexical_weight
        fused: dict[str, float] = {}
        per_leg: dict[str, int] = {}
        seen: dict[str, set[str]] = {}
        for leg in self.legs:
            scale = weight if leg == "lexical" else 1.0
            ladder = sorted(
                (c for c in candidates if leg in c.scores),
                key=lambda c: (-c.scores[leg], c.chunk.id),
            )
            per_leg[leg] = len(ladder)
            for rank, item in enumerate(ladder, start=1):
                fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + scale / (self.k + rank)
                seen.setdefault(item.chunk.id, set()).add(leg)
        degraded = any(count == 0 for count in per_leg.values())
        trace_module.record(
            trace_module.FusionReport(
                legs=self.legs,
                k=self.k,
                per_leg=per_leg,
                overlap=sum(1 for legs in seen.values() if legs == set(self.legs)),
                degraded=degraded,
            )
        )
        if degraded:
            trace_module.not_comparable(fusion_module.DEGRADED_REASON)
        ordered = sorted(candidates, key=lambda c: (-fused.get(c.chunk.id, 0.0), c.chunk.id))
        return [c.scored_by(self.name, fused.get(c.chunk.id, 0.0)) for c in ordered]

    fusion_module.RRFStage.run = run
    try:
        yield
    finally:
        fusion_module.RRFStage.run = shipped


# --- the verdict ---------------------------------------------------------------------------------


def _reciprocals(
    docs: Sequence[lq.Outcome],
    threads: Sequence[thread_duplication.Outcome],
    family: str,
    gated: frozenset[str],
) -> dict[str, float]:
    groups, forms, only_gated = FAMILIES[family]
    found = {
        o.asked.id: lq.reciprocal(o.rank)
        for o in docs
        if o.asked.group in groups and (not only_gated or o.asked.text in gated)
    }
    found.update(
        {
            o.asked.query.id: lq.reciprocal(o.first)
            for o in threads
            if o.asked.form in forms and (not only_gated or o.asked.query.text in gated)
        }
    )
    return found


def verdicts(
    dense: Sequence[lq.Outcome],
    docs: Mapping[str, Sequence[lq.Outcome]],
    threads: Mapping[str, Sequence[thread_duplication.Outcome]],
    gated: Mapping[str, frozenset[str]],
) -> dict[str, tl.Verdict]:
    """Every candidate against the first, under round 6's rule, as round 5's :class:`Verdict`."""
    incumbent, *candidates = CANDIDATES
    nothing: frozenset[str] = frozenset()
    dense_titles = statistics.fmean(_reciprocals(dense, (), "titles", nothing).values())
    shipped_titles = _reciprocals(docs[incumbent], threads[incumbent], "titles", nothing)
    found: dict[str, tl.Verdict] = {}
    for name in candidates:
        families: dict[str, tl.Change] = {}
        for family in FAMILIES:
            before = _reciprocals(docs[incumbent], threads[incumbent], family, gated[name])
            if not before:
                continue
            after = _reciprocals(docs[name], threads[name], family, gated[name])
            differences = [after[key] - value for key, value in before.items()]
            families[family] = tl.Change(
                count=len(differences),
                change=statistics.fmean(differences),
                interval=lq.bootstrap_interval(differences),
                split=(sum(d > 0 for d in differences), sum(d < 0 for d in differences)),
            )
        titles = _reciprocals(docs[name], threads[name], "titles", nothing)
        found[name] = tl.Verdict(
            titles=(
                dense_titles,
                statistics.fmean(shipped_titles.values()),
                statistics.fmean(titles.values()),
            ),
            titles_worse=sum(titles[key] < value for key, value in shipped_titles.items()),
            families=families,
            unanswerable_out_of_none=sum(
                o.band is not ConfidenceBand.NONE
                for o in docs[name]
                if o.asked.group == "unanswerable"
            ),
        )
    return found


def choose(per_model: Sequence[Mapping[str, tuple[bool, float]]]) -> str | None:
    """The qualifier under every model with the best averaged non-title change; ties go early."""
    qualifying = [
        name for name in CANDIDATES if all(name in model and model[name][0] for model in per_model)
    ]
    if not qualifying:
        return None
    gains = {name: statistics.fmean(model[name][1] for model in per_model) for name in qualifying}
    best = max(gains.values())
    return next(name for name in qualifying if gains[name] >= best - tl.TIE)


# --- one model's run -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Measured:
    dense: list[lq.Outcome]
    docs: dict[str, list[lq.Outcome]]
    threads: dict[str, list[thread_duplication.Outcome]]
    gated: dict[str, frozenset[str]]


async def measure(args: argparse.Namespace, work: Path, emit: Callable[[str], None]) -> Measured:
    """Index each corpus once and ask every candidate everything."""
    overrides = lq.embedding_overrides(args)
    root = args.docs
    started = time.perf_counter()

    def progress(step: str) -> None:
        emit(f"[{time.perf_counter() - started:6.0f}s] {step}")

    index = await lq.index_docs(work / "docs", overrides=overrides, root=root)
    pinned = tl.revision(index.data_dir)
    indexes = [index.data_dir]
    progress(f"docs/ indexed at schema revision {pinned}")
    settings = lq.settings_for(index.data_dir, overrides, lq.PIPELINES["hybrid"])
    async with Runtime(settings) as runtime:
        store = cast("SqliteDocStore", await runtime.documents())
        asked = await lq.docs_questions(store, runtime.workspace, root)
    asked.extend(tl.heading_questions(root))
    asked.extend(bare_identifiers(root))
    tl.unchanged(indexes, pinned)
    emit(f"docs/ from {root}: {len(index.places)} chunks with {args.model}; {len(asked)} queries")

    corpus = build_corpus(seed=args.seed, copies=args.copies, distractors=args.distractors)
    shutil.rmtree(work / "threads", ignore_errors=True)
    indexed = await thread_duplication.index(corpus, work / "threads", overrides=overrides)
    indexes.append(indexed.data_dir)
    tl.unchanged(indexes, pinned)
    _, thread_asked = thread_duplication.queries(corpus)
    progress("mail threads indexed")

    dense = await lq.run_docs(
        index, asked, variant=lq.SHIPPED_MATCH, pipeline="dense", overrides=overrides
    )
    docs: dict[str, list[lq.Outcome]] = {}
    threads: dict[str, list[thread_duplication.Outcome]] = {}
    gated: dict[str, frozenset[str]] = {}
    for name, candidate in CANDIDATES.items():
        tl.unchanged(indexes, pinned)
        selected: set[str] = set()
        with gated_weighting(candidate, selected):
            docs[name] = await lq.run_docs(
                index, asked, variant=lq.SHIPPED_MATCH, pipeline="hybrid", overrides=overrides
            )
            with lq.matching(lq.VARIANTS[lq.SHIPPED_MATCH]):
                run = await thread_duplication.run_pipeline(
                    indexed,
                    thread_asked,
                    name="hybrid",
                    overrides=overrides,
                    ks=(10,),
                    counted=frozenset({lq.EXHAUSTED_BUDGET}),
                )
        threads[name] = run.outcomes
        gated[name] = frozenset(selected)
        progress(f"{name} asked, {len(selected)} distinct queries gated")
    tl.unchanged(indexes, pinned)
    return Measured(dense, docs, threads, gated)


def report(measured: Measured, emit: Callable[[str], None]) -> dict[str, tl.Verdict]:
    rows: list[tuple[str, str, lq.Row]] = []
    for name, outcomes in [("dense alone", measured.dense), *measured.docs.items()]:
        pipeline = "dense" if name == "dense alone" else "hybrid"
        rows.extend(
            (name, pipeline, lq.summarize(group, chosen))
            for group in GROUPS
            if (chosen := [o for o in outcomes if o.asked.group == group])
        )
    emit("\n### round 6: docs/\n")
    emit(lq.render(rows))
    emit("\n### round 6: mail threads, MRR@10\n")
    for name, outcomes in measured.threads.items():
        for form in thread_duplication.FORMS:
            chosen = [o for o in outcomes if o.asked.form == form]
            mrr = statistics.fmean(lq.reciprocal(o.first) for o in chosen)
            hits = sum(o.found for o in chosen)
            emit(f"  {name} {form}: found {hits}/{len(chosen)}, MRR {mrr:.3f}")
    found = verdicts(measured.dense, measured.docs, measured.threads, measured.gated)
    for name, verdict in found.items():
        emit(f"\n{name} against {next(iter(CANDIDATES))}:\n{verdict.describe()}")
    return found


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--docs", type=Path, help="a docs/ tree exported at one commit")
    parser.add_argument("--verdicts", type=Path, default=None, help="write verdicts here, as JSON")
    parser.add_argument(
        "--choose", type=Path, nargs="+", default=None, help="apply the rule to saved verdicts"
    )
    parser.add_argument("--model", default="BAAI/bge-m3", help="embedding model")
    parser.add_argument("--provider", default="onnx", help="embedding backend: onnx, mlx or ollama")
    parser.add_argument("--prefix-scheme", default=None, help="the model's prefix scheme")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--tokenizer", default=None, help="a served backend's tokenizer")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=lq.HELD_OUT_SEED)
    parser.add_argument("--copies", type=int, default=lq.HELD_OUT_COPIES)
    parser.add_argument("--distractors", type=int, default=lq.HELD_OUT_DISTRACTORS)
    parser.add_argument("--work", type=Path, default=None, help="keep the indexes here")
    args = parser.parse_args(argv)
    out = console()

    def emit(text: str) -> None:
        out.print(text, markup=False)

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.choose:
        chosen = choose([tl.loaded(path) for path in args.choose])
        emit(f"-> {'ship ' + chosen if chosen else 'no change'}")
        return
    if args.docs is None:
        parser.error("--docs is required to measure")
    args.docs = args.docs.resolve()
    if args.verdicts is not None:
        args.verdicts = args.verdicts.resolve()
    with contextlib.ExitStack() as stack:
        if args.work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="shapes-")))
        else:
            work = args.work.resolve()
            work.mkdir(parents=True, exist_ok=True)
        _isolate(work)
        found = report(asyncio.run(measure(args, work, emit)), emit)
        if args.verdicts is not None:
            tl.saved(found, args.verdicts)


if __name__ == "__main__":
    main()
