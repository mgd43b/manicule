"""Round 5 of :mod:`tests.benchmarks.lexical_questions`: why hybrid ranks a document's title low.

Issue `#489 <https://github.com/mgd43b/manicule/issues/489>`_. Round 4 (#486, #487) measured the
obvious repairs to fusion and kept ``K = 60`` with equal leg weights, leaving the defect standing:
on this repository's ``docs/``, hybrid ranks a document's own title below where the dense leg
alone ranks it — MRR@10 0.853 against 0.934 under BGE-M3, 0.764 against 0.895 under
Qwen3-Embedding-0.6B. Run it::

    .venv/bin/python -m tests.benchmarks.title_lookups --docs <exported docs/ tree>
    .venv/bin/python -m tests.benchmarks.title_lookups --docs <tree> --provider ollama \\
        --model qwen3-embedding:0.6b --prefix-scheme qwen3 --tokenizer <dir>

**What was known when this rule was written.** Round 4's figures, and one diagnostic run per
model over ``docs/`` at commit ``9ce34e6e``: for every title query, where the dense leg, the
lexical leg and hybrid each put the titled document. It showed the cause, which round 4's
arithmetic had only described. **The lexical index holds neither a document's title nor its
file name.** ``chunks.heading_text``, the column FTS5 indexes beside ``text``, is the chunk's
heading path alone, while the breadcrumb prefixed to ``embed_text`` also carries the source's
prefix, the document's place in its source and its title. So the dense leg reads "retrieval.md"
in front of every chunk of ``retrieval.md`` and the lexical leg never sees it: a one-word title
reaches the lexical leg only through the bodies of whichever documents use the word most. Under
BGE-M3 the lexical leg put the titled document 5th or lower, or nowhere, for 7 of the 19 titles,
and hybrid lost rank on 4 of them (``contracts``, ``memory-authoring``, ``parsing``,
``retrieval``); under Qwen3 it lost rank on 6. ``storage/fts.py``'s own comment on the 0.4
column weight says the breadcrumb is searchable "for the same reason it is prefixed to
``embed_text``", which describes a breadcrumb the column does not hold.

**The candidates**, in tie-break order, each against the shipped configuration:

- ``breadcrumb`` — ``heading_text`` holds the chunk's whole breadcrumb, the same elements
  :func:`manicule.chunking.breadcrumb.parts` gives the embedder (unelided, since FTS5 has no
  token budget), at the shipped column weight 0.4. Measured by rewriting ``heading_text`` in a
  copy of the shipped index, row by row, from each document's stored record — which is exactly
  what a migration would do to an existing installation, and what ingest would write afresh.
- ``breadcrumb-heading-1.0`` — the same, with the column weighted 1.0.
- ``heading-1.0`` — no index change; the shipped column at weight 1.0 (issue question 2).
- ``rerank`` — the shipped index with ``rag.reranker = "cross_encoder"``, which is
  ``BAAI/bge-reranker-v2-m3`` scoring the fused top 20 under the default ``balanced`` profile
  (issue question 4).

Query-shape-dependent fusion (issue question 3) is not a candidate in this round. It contradicts
``docs/retrieval.md`` §5.1's rank-only fusion and needs a field ``PipelineIdentity`` does not
have, so it is measured only if no candidate here fixes the defect.

**The questions.** Round 4's sets, from a ``docs/`` tree exported at one commit and passed as
``--docs`` so every candidate and both models index the same text, plus one new set:

- *headings* — every numbered heading in ``docs/`` whose text occurs once, asked as that text
  without its number, its target its own section. Derived mechanically, like the identifier
  sets. It answers issue question 5: 19 titles cannot carry a significance test (a fix that
  repaired every title hybrid now loses would be a sign test of 4 to 0 under BGE-M3,
  ``p = 0.125``), and they are also what the ``breadcrumb`` candidates index verbatim, so part
  of any title gain they show is by construction. The headings are keyword-shaped lookups
  those candidates add nothing to — a heading is in ``heading_text`` already — and a longer
  ``heading_text`` lowers what a heading match is worth, so it is the check that the gain on
  titles is not paid for by every other lookup of a document's own words.

**The round 5 rule.** Hybrid pipeline, ``k = 10``, reciprocal rank paired by query, applied per
model. Families: *titles*; *headings*; *prose* (authored, labeled multi-hop, held-out thread
questions); *keyword* (thread keyword lookups); *identifiers* (both identifier sets). A
candidate **fixes the defect** under a model when all of these hold:

1. On titles, it closes at least half the gap to the dense leg alone measured in the same run —
   ``MRR(candidate) >= MRR(shipped) + 0.5 * (MRR(dense) - MRR(shipped))`` — and no title query
   ranks worse than under the shipped configuration. A defect-fix criterion rather than a
   significance test, for the reason above.
2. In every other family, the lower end of a 95% paired bootstrap interval on its mean
   difference from the shipped configuration is above -0.02 — round 4's bound. The identifier
   family is where #485's gain lives, so this is the condition that keeps it.
3. No unanswerable question leaves the ``none`` band.

A candidate that fixes the defect under **both** models is a qualifier. Among the qualifiers,
the one with the highest mean difference over the four non-title families, averaged over the two
models, is chosen; within 0.005, the earlier in :data:`CANDIDATES` wins, which puts the change
that makes the existing documentation true before a column weight, and both before a reranker.
No qualifier: no change, and query-shape-dependent fusion is the next round. A candidate that
fixes the defect under one model alone is recorded, not shipped.

**What a chosen candidate changes.** ``breadcrumb``: ingest writes the whole breadcrumb into
``heading_text``, and a migration rewrites existing rows from each document's stored record —
FTS5's update trigger reindexes them, and nothing is re-embedded. A column weight: the two
``bm25()`` arguments in ``SEARCH_SQL``. ``rerank`` cannot change the default whatever it
measures, because the cross-encoder needs the ``rerank`` extra and torch, which a default
installation does not have; if it qualifies, ``docs/retrieval.md`` §6 records it as the setting
for an installation that has the extra, and the round's choice is the best qualifier among the
others. Its time per query is reported beside the verdict.

Everything is local apart from the embedding and reranking models, read from the local model
cache or the named ``ollama`` server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from manicule.app.runtime import Runtime
from manicule.chunking import breadcrumb
from manicule.cli.render import console
from manicule.core.retrieval import ConfidenceBand
from manicule.storage import docstore as docstore_module
from manicule.storage.fts import SEARCH_SQL
from tests.benchmarks import lexical_questions as lq
from tests.benchmarks import thread_duplication
from tests.benchmarks.thread_corpus import build_corpus
from tests.benchmarks.thread_duplication import _isolate  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from manicule.storage.docstore import SqliteDocStore

SHIPPED_WEIGHTS = "bm25(chunks_fts, 1.0, 0.4)"
DATABASE = "manicule.db"
HALF = 0.5
BOUND = -0.02
TIE = 0.005


@dataclass(frozen=True)
class Candidate:
    """What the lexical leg indexes, how it weights the heading column, and whether to rerank."""

    breadcrumb: bool = False
    heading_weight: float = 0.4
    rerank: bool = False


CANDIDATES: dict[str, Candidate] = {
    "shipped": Candidate(),
    "breadcrumb": Candidate(breadcrumb=True),
    "breadcrumb-heading-1.0": Candidate(breadcrumb=True, heading_weight=1.0),
    "heading-1.0": Candidate(heading_weight=1.0),
    "rerank": Candidate(rerank=True),
}
"""The incumbent first, then the candidates in tie-break order."""

FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "titles": (("titles",), ()),
    "headings": (("headings",), ()),
    "prose": (("authored", "multi-hop"), ("question",)),
    "keyword": ((), ("keywords",)),
    "identifiers": (("identifiers", lq.TWO_WORDS), ()),
}
"""Each family's ``docs/`` groups and mail-thread query forms."""

GROUPS = (*lq.GROUPS, "headings")

_HEADING = re.compile(r"^#+ ([0-9][0-9.]*) +(.+?)\s*$")


# --- the new question set ------------------------------------------------------------------------


def heading_questions(root: Path) -> list[lq.Asked]:
    """Every numbered heading whose text occurs once in ``root``, asked as that text."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(pattern.removesuffix("**")) for pattern in lq.EXCLUDED):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            heading = _HEADING.match(line)
            if heading:
                found.append((relative, heading.group(1).rstrip("."), heading.group(2)))
    counts = Counter(" ".join(text.casefold().split()) for _, _, text in found)
    return [
        lq.Asked(f"heading-{relative}-{number}", text, "headings", ((relative, number),))
        for relative, number, text in found
        if counts[" ".join(text.casefold().split())] == 1
    ]


# --- the candidates ------------------------------------------------------------------------------


async def rewrite_heading_text(data_dir: Path, overrides: Mapping[str, Any]) -> int:
    """Set every chunk's ``heading_text`` to its whole breadcrumb, from its document's record.

    What a migration does to an installation: the stored document is the input, the update
    trigger reindexes FTS5, and no vector is touched. Returns the rows rewritten.
    """
    settings = thread_duplication.settings_for(data_dir, overrides)
    values: list[tuple[str, str]] = []
    async with Runtime(settings) as runtime:
        store = cast("SqliteDocStore", await runtime.documents())
        for document in await store.list_documents(limit=100_000):
            hierarchy = breadcrumb.source_hierarchy(document)
            values.extend(
                (
                    breadcrumb.SEPARATOR.join(
                        breadcrumb.parts(document, chunk.heading_path, hierarchy=hierarchy)
                    ),
                    chunk.id,
                )
                for chunk in await store.document_chunks(document.id)
            )
    with contextlib.closing(sqlite3.connect(data_dir / DATABASE)) as db, db:
        db.executemany("UPDATE chunks SET heading_text = ? WHERE id = ?", values)
    return len(values)


def copied(source: Path, target: Path) -> Path:
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.lock"))
    return target


@contextlib.contextmanager
def heading_weight(weight: float) -> Generator[None]:
    """Score the heading column at ``weight`` for the duration, through the store's statement."""
    statement = SEARCH_SQL.replace(SHIPPED_WEIGHTS, f"bm25(chunks_fts, 1.0, {weight})")
    if SEARCH_SQL.count(SHIPPED_WEIGHTS) != 2:
        msg = f"SEARCH_SQL no longer scores with {SHIPPED_WEIGHTS} in both places"
        raise RuntimeError(msg)
    shipped = docstore_module.SEARCH_SQL
    docstore_module.SEARCH_SQL = statement
    try:
        yield
    finally:
        docstore_module.SEARCH_SQL = shipped


def candidate_overrides(overrides: Mapping[str, Any], candidate: Candidate) -> dict[str, Any]:
    merged = dict(overrides)
    if candidate.rerank:
        merged["rag"] = {**merged.get("rag", {}), "reranker": "cross_encoder"}
    return merged


# --- the verdict ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    count: int
    change: float
    interval: tuple[float, float]
    split: tuple[int, int]


@dataclass(frozen=True)
class Verdict:
    """One candidate against the shipped configuration, under one model."""

    titles: tuple[float, float, float]
    """MRR of titles: dense alone, shipped hybrid, this candidate."""
    titles_worse: int
    families: dict[str, Change]
    unanswerable_out_of_none: int

    @property
    def closes_half(self) -> bool:
        dense, shipped, candidate = self.titles
        return candidate >= shipped + HALF * (dense - shipped)

    @property
    def fixes(self) -> bool:
        return (
            self.closes_half
            and self.titles_worse == 0
            and all(f.interval[0] > BOUND for name, f in self.families.items() if name != "titles")
            and self.unanswerable_out_of_none == 0
        )

    @property
    def gain(self) -> float:
        return statistics.fmean(f.change for n, f in self.families.items() if n != "titles")

    def describe(self) -> str:
        dense, shipped, candidate = self.titles
        lines = [
            f"  titles MRR: dense {dense:.3f}, shipped {shipped:.3f}, candidate {candidate:.3f}; "
            f"half the gap closed {self.closes_half}, titles ranked worse {self.titles_worse}"
        ]
        lines.extend(
            f"  {name} ({f.count}): {f.change:+.3f} [{f.interval[0]:+.3f}, {f.interval[1]:+.3f}], "
            f"better/worse {f.split[0]}/{f.split[1]}"
            for name, f in self.families.items()
        )
        lines.append(
            f"  unanswerable out of none {self.unanswerable_out_of_none}; fixes the defect "
            f"{self.fixes}, mean non-title change {self.gain:+.3f}"
        )
        return "\n".join(lines)


def _reciprocals(
    docs: Sequence[lq.Outcome], threads: Sequence[thread_duplication.Outcome], family: str
) -> dict[str, float]:
    groups, forms = FAMILIES[family]
    found = {o.asked.id: lq.reciprocal(o.rank) for o in docs if o.asked.group in groups}
    found.update(
        {o.asked.query.id: lq.reciprocal(o.first) for o in threads if o.asked.form in forms}
    )
    return found


def verdicts(
    dense: Sequence[lq.Outcome],
    docs: Mapping[str, Sequence[lq.Outcome]],
    threads: Mapping[str, Sequence[thread_duplication.Outcome]],
) -> dict[str, Verdict]:
    """Every candidate in :data:`CANDIDATES` against the first, under round 5's rule."""
    incumbent, *candidates = CANDIDATES
    dense_titles = statistics.fmean(_reciprocals(dense, (), "titles").values())
    shipped_titles = _reciprocals(docs[incumbent], threads[incumbent], "titles")
    found: dict[str, Verdict] = {}
    for name in candidates:
        families: dict[str, Change] = {}
        for family in FAMILIES:
            before = _reciprocals(docs[incumbent], threads[incumbent], family)
            after = _reciprocals(docs[name], threads[name], family)
            differences = [after[key] - value for key, value in before.items()]
            families[family] = Change(
                count=len(differences),
                change=statistics.fmean(differences),
                interval=lq.bootstrap_interval(differences),
                split=(sum(d > 0 for d in differences), sum(d < 0 for d in differences)),
            )
        titles = _reciprocals(docs[name], threads[name], "titles")
        found[name] = Verdict(
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
    """The qualifier under every model with the best averaged non-title change; ties go early.

    ``per_model`` holds, per model, each candidate's ``(fixes, gain)`` as :func:`saved` wrote
    them. ``rerank`` is never chosen: it cannot be a default (see the module docstring).
    """
    qualifying = [
        name
        for name in CANDIDATES
        if name != "rerank" and all(name in model and model[name][0] for model in per_model)
    ]
    if not qualifying:
        return None
    gains = {name: statistics.fmean(model[name][1] for model in per_model) for name in qualifying}
    best = max(gains.values())
    return next(name for name in qualifying if gains[name] >= best - TIE)


def saved(found: Mapping[str, Verdict], path: Path) -> None:
    """One model's verdicts, as :func:`choose` reads them."""
    path.write_text(
        json.dumps({name: [v.fixes, v.gain] for name, v in found.items()}, indent=2) + "\n",
        encoding="utf-8",
    )


def loaded(path: Path) -> dict[str, tuple[bool, float]]:
    raw = cast("dict[str, list[Any]]", json.loads(path.read_text(encoding="utf-8")))
    return {name: (bool(fixes), float(gain)) for name, (fixes, gain) in raw.items()}


# --- one model's run -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Measured:
    dense: list[lq.Outcome]
    docs: dict[str, list[lq.Outcome]]
    threads: dict[str, list[thread_duplication.Outcome]]
    seconds: dict[str, float]
    """Wall time per ``docs/`` query, per candidate."""


async def measure(args: argparse.Namespace, work: Path, emit: Callable[[str], None]) -> Measured:
    """Index once per corpus, copy for the breadcrumb, and ask every candidate everything."""
    overrides = lq.embedding_overrides(args)
    root = args.docs.resolve()
    shipped_index = await lq.index_docs(work / "docs", overrides=overrides, root=root)
    settings = lq.settings_for(shipped_index.data_dir, overrides, lq.PIPELINES["hybrid"])
    async with Runtime(settings) as runtime:
        store = cast("SqliteDocStore", await runtime.documents())
        asked = await lq.docs_questions(store, runtime.workspace, root)
    asked.extend(heading_questions(root))
    rewritten_index = replace(
        shipped_index, data_dir=copied(shipped_index.data_dir, work / "docs-breadcrumb")
    )
    rows = await rewrite_heading_text(rewritten_index.data_dir, overrides)
    emit(
        f"docs/ from {root}: {shipped_index.documents} documents, {len(shipped_index.places)} "
        f"chunks ({rows} rewritten) with {args.model}; {len(asked)} queries"
    )

    corpus = build_corpus(seed=args.seed, copies=args.copies, distractors=args.distractors)
    shutil.rmtree(work / "threads", ignore_errors=True)
    shipped_threads = await thread_duplication.index(corpus, work / "threads", overrides=overrides)
    rewritten_threads = replace(
        shipped_threads, data_dir=copied(shipped_threads.data_dir, work / "threads-breadcrumb")
    )
    await rewrite_heading_text(rewritten_threads.data_dir, overrides)
    _, thread_asked = thread_duplication.queries(corpus)

    dense = await lq.run_docs(
        shipped_index, asked, variant=lq.SHIPPED_MATCH, pipeline="dense", overrides=overrides
    )
    docs: dict[str, list[lq.Outcome]] = {}
    threads: dict[str, list[thread_duplication.Outcome]] = {}
    seconds: dict[str, float] = {}
    for name, candidate in CANDIDATES.items():
        merged = candidate_overrides(overrides, candidate)
        index = rewritten_index if candidate.breadcrumb else shipped_index
        indexed = rewritten_threads if candidate.breadcrumb else shipped_threads
        with heading_weight(candidate.heading_weight):
            started = time.perf_counter()
            docs[name] = await lq.run_docs(
                index, asked, variant=lq.SHIPPED_MATCH, pipeline="hybrid", overrides=merged
            )
            seconds[name] = (time.perf_counter() - started) / len(asked)
            with lq.matching(lq.VARIANTS[lq.SHIPPED_MATCH]):
                run = await thread_duplication.run_pipeline(
                    indexed,
                    thread_asked,
                    name="hybrid",
                    overrides=merged,
                    ks=(10,),
                    counted=frozenset({lq.EXHAUSTED_BUDGET}),
                )
        threads[name] = run.outcomes
    return Measured(dense, docs, threads, seconds)


def report(measured: Measured, emit: Callable[[str], None]) -> dict[str, Verdict]:
    rows: list[tuple[str, str, lq.Row]] = []
    for name, outcomes in [("dense alone", measured.dense), *measured.docs.items()]:
        for group in GROUPS:
            chosen = [o for o in outcomes if o.asked.group == group]
            if chosen:
                rows.append(
                    (
                        name,
                        "dense" if name == "dense alone" else "hybrid",
                        lq.summarize(group, chosen),
                    )
                )
    emit("\n### round 5: docs/\n")
    emit(lq.render(rows))
    emit("\n### round 5: mail threads, MRR@10\n")
    for name, outcomes in measured.threads.items():
        for form in thread_duplication.FORMS:
            chosen = [o for o in outcomes if o.asked.form == form]
            mrr = statistics.fmean(lq.reciprocal(o.first) for o in chosen)
            hits = sum(o.found for o in chosen)
            emit(f"  {name} {form}: found {hits}/{len(chosen)}, MRR {mrr:.3f}")
    emit("\n### round 5: time per docs/ query, ms\n")
    for name, value in measured.seconds.items():
        emit(f"  {name}: {value * 1000:.0f}")
    found = verdicts(measured.dense, measured.docs, measured.threads)
    for name, verdict in found.items():
        emit(f"\n{name} against {next(iter(CANDIDATES))}:\n{verdict.describe()}")
    return found


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--docs",
        type=Path,
        help="a docs/ tree exported at one commit, so every run indexes the same text",
    )
    parser.add_argument(
        "--verdicts", type=Path, default=None, help="write this model's verdicts here, as JSON"
    )
    parser.add_argument(
        "--choose",
        type=Path,
        nargs="+",
        default=None,
        help="apply the rule to every model's saved verdicts instead of measuring",
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

    if args.choose:
        chosen = choose([loaded(path) for path in args.choose])
        emit(f"-> {'ship ' + chosen if chosen else 'no change'}")
        return
    if args.docs is None:
        parser.error("--docs is required to measure")
    with contextlib.ExitStack() as stack:
        if args.work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="titles-")))
        else:
            work = args.work.resolve()
            work.mkdir(parents=True, exist_ok=True)
        _isolate(work)
        measured = asyncio.run(measure(args, work, emit))
        found = report(measured, emit)
        if args.verdicts is not None:
            saved(found, args.verdicts)


if __name__ == "__main__":
    main()
