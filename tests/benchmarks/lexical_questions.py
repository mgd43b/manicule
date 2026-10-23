"""What the lexical leg contributes to a question, measured under each way of building its match.

The figures in ``docs/retrieval.md`` §4.1 come from here, and so does the decision recorded there.
Issue `#481 <https://github.com/mgd43b/manicule/issues/481>`_.

Run it::

    .venv/bin/python -m tests.benchmarks.lexical_questions
    .venv/bin/python -m tests.benchmarks.lexical_questions --model BAAI/bge-small-en-v1.5

**The defect it was built to measure.** The store quoted every token of the query and joined
them with spaces, which FTS5 reads as ``AND``: a chunk matched only if it held every word,
including "how", "do" and "I". A question therefore reached the dense leg alone while the trace,
the pipeline declaration and the documents all described a hybrid.

**The corpus is this repository's own ``docs/`` tree**, ``evaluation/`` excluded because it holds
the questions themselves. Anybody with the repository can rebuild it, which is what makes a
figure here checkable rather than reported. A second, synthetic corpus — the mail threads of
:mod:`tests.benchmarks.thread_corpus`, whose ground truth is known by construction — is measured
the same way, so no conclusion rests on questions written by the same hand that wrote the fix.

**The questions**, each kept apart in the report because they fail differently:

- *authored* — natural questions about one section of ``docs/``, each labeled with the section
  that answers it. Written from the section headings for this harness, so they are the
  supplementary set rather than the primary one, and their provenance says so.
- *multi-hop* — ``docs/evaluation/multi-hop-queries.json``, written before this harness existed.
  Its notes name the sections a correct answer joins; :data:`MULTI_HOP_TARGETS` transcribes
  them, and a query whose note names no section is scored for the lexical leg's emptiness only.
- *titles* — :func:`~manicule.evaluation.probe.probe_from_titles`, each document's own title as
  the query: the keyword-shaped lookups the old ``AND`` already served, and the regression guard.
- *example* — ``docs/evaluation/example-queries.json``, unlabeled, for emptiness only.
- *unanswerable* — questions about subjects ``docs/`` does not hold, plus gibberish, so the
  confidence separation ``docs/retrieval.md`` §8.4 measured can be checked under a leg that now
  matches something for nearly every question.
- *threads* — the synthetic mail corpus asked each passage twice, as a question and as words the
  passage contains (``tests/benchmarks/thread_duplication.py``).

**What is measured, per query and pipeline**: whether the lexical leg matched anything; the rank
of the first chunk inside a labeled section (``hit@k``, reciprocal rank); the confidence band;
and the lexical stage's wall time. The pipelines are the shipped hybrid and the lexical leg
alone, under each :data:`VARIANTS` entry, plus the dense leg alone once, since no variant
touches it. Separately, every variant's statement is timed against an FTS5 table of
:data:`SCALE` chunks, because a disjunction over "the" or "how" scores most of an index and the
cost of that grows with the corpus while ``docs/`` does not.

**The decision rule, fixed before the first run.** On the hybrid pipeline at ``k = 10``:

1. A variant is *admissible* only if, against ``and``, it loses no more than one query's hit
   on the titles and on the threads' keyword form; at :data:`SCALE` chunks its median statement
   time over the question sets stays under 50 ms; and no unanswerable query leaves the ``none``
   band.
2. Among admissible variants, the one with the most hits over the natural questions — authored,
   multi-hop and the threads' question form together — wins. Within two queries of each other,
   the simpler rule wins, in the order the variants are listed.
3. If no admissible variant beats ``and`` by more than two queries, nothing changes.

**Round 1 returned rule 3: no change.** The dense leg alone sat at the ``hit@10`` ceiling on
every labeled set, so hits could not separate the candidates, and the two disjunctions cost
multi-hop hits and MRR. Those sets have no question carrying a rare term the dense leg cannot
place, which is the shape the lexical leg exists for, so a second round was added. It is a
second round and not a revision: its rule was written after round 1 and before its own first run.

- *identifiers* — derived mechanically by :func:`identifier_questions`, not authored: every
  backticked, identifier-shaped term in ``docs/`` that occurs in exactly one numbered section,
  wrapped in one of :data:`TEMPLATES` by its sorted position, with that section as the target.

**The round 2 rule.** Candidates are ``and``, ``and-content`` and ``all-but-one``; the two
disjunctions are not reopened. On the hybrid pipeline at ``k = 10``, a candidate replaces ``and``
only if it gains more than two ``hit@10`` on the identifier questions, its MRR@10 there is not
lower, and its MRR@10 on each round 1 labeled set is no more than 0.02 below ``and``'s. Otherwise
nothing changes.

**Round 2 returned ``and-content``,** which is now the shipped rule: 603 identifier questions
found in the top ten against 506, MRR 0.643 against 0.476, and no round 1 set more than 0.005 of
MRR below ``and``. ``all-but-one`` tied it on identifiers and failed the MRR condition.

Everything is local: no network, temporary directories, and the only thing read from outside the
repository is the embedding model, from the local model cache.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from manicule.app.runtime import Runtime
from manicule.cli.render import console
from manicule.config.settings import Settings
from manicule.connectors.filesystem import FilesystemConnector
from manicule.core.retrieval import ConfidenceBand, Filter, Query
from manicule.evaluation.probe import probe_from_titles
from manicule.evaluation.queries import load_query_set
from manicule.evaluation.statistics import wilson_interval
from manicule.retrieval.lexical import DEGRADED_REASON as LEXICAL_EMPTY
from manicule.retrieval.retriever import Retriever
from manicule.retrieval.trace import Shortfall
from manicule.storage import docstore as docstore_module
from manicule.storage.fts import FUNCTION_WORDS, SEARCH_SQL, escape_match_query, match_words
from manicule.storage.models import FTS_TOKENIZER
from tests.benchmarks import thread_duplication
from tests.benchmarks.thread_corpus import Corpus, build_corpus
from tests.benchmarks.thread_duplication import _isolate  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from manicule.plugins import Discovery
    from manicule.storage.docstore import SqliteDocStore

REPOSITORY = Path(__file__).resolve().parents[2]
DOCS = REPOSITORY / "docs"
EXCLUDED = ("evaluation/**", "images/**")
SOURCE = "docs"
KS = (5, 10)
EXHAUSTED_BUDGET = Shortfall.EXHAUSTED_BUDGET.value
SCALE = 100_000
"""Chunks in the table the statements are timed against."""

PIPELINES: dict[str, tuple[str, ...]] = {
    "hybrid": ("dense", "lexical", "rrf"),
    "lexical": ("lexical",),
    "dense": ("dense",),
}


# --- the ways of building a match ----------------------------------------------------------------


def _literals(text: str) -> list[str]:
    """The store's own words for ``text``, each quoted."""
    return ['"' + word.replace('"', '""') + '"' for word in match_words(text)]


def _content(literals: list[str]) -> list[str]:
    """Without function words, unless that leaves nothing: then the words are the subject."""
    kept = [literal for literal in literals if literal.strip('"').lower() not in FUNCTION_WORDS]
    return kept or literals


def match_all(text: str) -> str:
    """The rule before #481, verbatim: every word, read by FTS5 from juxtaposition as ``AND``."""
    tokens: list[str] = []
    for raw in text.split():
        cleaned = "".join(char for char in raw if char.isalnum() or char in "-_.'")
        if not cleaned:
            continue
        tokens.append('"' + cleaned.replace('"', '""') + '"')
    return " ".join(tokens)


def match_any(text: str) -> str:
    """Any word, with BM25's IDF doing the weighting."""
    return " OR ".join(_literals(text))


def match_any_content(text: str) -> str:
    """Any word that is not a function word."""
    return " OR ".join(_content(_literals(text)))


def match_all_but_one_content(text: str) -> str:
    """Every content word but at most one: a minimum-should-match of ``n - 1``.

    FTS5 has no minimum-should-match, so it is spelled as a disjunction of the ``n`` conjunctions
    that each leave one word out. Below three words that is the conjunction itself.
    """
    kept = _content(_literals(text))
    if len(kept) < 3:
        return " ".join(kept)
    return " OR ".join("(" + " ".join(kept[:i] + kept[i + 1 :]) + ")" for i in range(len(kept)))


VARIANTS: dict[str, Callable[[str], str]] = {
    "and": match_all,
    "or": match_any,
    "or-content": match_any_content,
    "and-content": escape_match_query,
    "all-but-one": match_all_but_one_content,
}
"""Report order, simplest first, which is also rule 2's tie-break order.

``and`` is the rule this was built to replace. ``and-content`` — every word but the
:data:`~manicule.storage.fts.FUNCTION_WORDS` — won, and is the shipped
:func:`~manicule.storage.fts.escape_match_query` itself rather than a copy of it, so a change to
the shipped rule is measured by the next run without anybody remembering to edit this list."""


@contextlib.contextmanager
def matching(variant: Callable[[str], str]) -> Generator[None]:
    """Make the store build its match with ``variant`` for the duration."""
    shipped = docstore_module.escape_match_query
    docstore_module.escape_match_query = variant
    try:
        yield
    finally:
        docstore_module.escape_match_query = shipped


# --- the questions -------------------------------------------------------------------------------

Target = tuple[str, str]
"""A document under ``docs/`` and a section number in it: ``("retrieval.md", "4.5")``. A number
names its subsections too, so ``"6"`` is satisfied by ``6.2.1``."""

AUTHORED: tuple[tuple[str, str, Target], ...] = (
    ("why does min_score only apply to the dense leg", "dense-floor", ("retrieval.md", "4.5")),
    (
        "how does reciprocal rank fusion combine the two result lists",
        "rrf",
        ("retrieval.md", "5.1"),
    ),
    (
        "what happens to fusion when one leg returns fewer results than asked",
        "short-leg",
        ("retrieval.md", "5.3"),
    ),
    ("what does the K constant in RRF actually control", "rrf-k", ("retrieval.md", "5.4")),
    (
        "why is the reranker a cross-encoder instead of asking an LLM to score passages",
        "cross-encoder",
        ("retrieval.md", "6.1"),
    ),
    (
        "which tokenizer decides how much context the generator can be given",
        "context-tokens",
        ("retrieval.md", "7.2"),
    ),
    (
        "why are passages never cut in half to fit the context",
        "whole-passages",
        ("retrieval.md", "7.3"),
    ),
    (
        "how is cosine similarity turned into confidence when unrelated text still scores high",
        "noise-floor",
        ("retrieval.md", "8.4"),
    ),
    ("which questions does the router answer without searching", "router", ("retrieval.md", "9.1")),
    ("what goes into the cache key for a search", "cache-key", ("retrieval.md", "10.2")),
    (
        "when does the query result cache get invalidated",
        "cache-invalidation",
        ("retrieval.md", "10.3"),
    ),
    ("how many passages does each profile retrieve and rerank", "profiles", ("retrieval.md", "12")),
    ("how are acronyms from a glossary expanded in a search", "glossary", ("retrieval.md", "14")),
    (
        "why is the workspace filter treated as a security boundary",
        "tenancy",
        ("retrieval.md", "3.2"),
    ),
    ("how far past k does the dense leg over-fetch, and why", "overfetch", ("retrieval.md", "4.3")),
    (
        "how is the full text index kept in step with the chunks table",
        "fts-sync",
        ("storage.md", "6.1"),
    ),
    (
        "what does a search do while the vector index is still being built",
        "ann-build",
        ("storage.md", "6.2"),
    ),
    (
        "what happens if the process crashes between writing sqlite and writing the vectors",
        "crash",
        ("storage.md", "8"),
    ),
    ("how do I back up and restore the data directory consistently", "backup", ("storage.md", "9")),
    ("how is the membership of a collection worked out", "membership", ("storage.md", "11.2")),
    ("how do I get a deleted document back from the trash", "trash", ("storage.md", "11.6")),
    ("how do I move my index from lancedb to qdrant", "backends", ("storage.md", "6.8")),
    (
        "how does ingest decide a document has not changed and can be skipped",
        "change-detection",
        ("ingest.md", "4"),
    ),
    ("what stops one bad file from aborting the whole sync", "never-aborts", ("ingest.md", "6")),
    (
        "why can only one process use a data directory at a time",
        "one-instance",
        ("ingest.md", "6.5"),
    ),
    (
        "when is a document deleted because it disappeared from the source",
        "reconcile",
        ("ingest.md", "11"),
    ),
    (
        "how do I re-embed everything after changing the embedding model",
        "re-embed",
        ("ingest.md", "10"),
    ),
    ("what happens when I cancel a sync halfway through", "cancel", ("ingest.md", "13.3")),
    ("where is personal information redacted during ingest", "ingest-pii", ("ingest.md", "3.4")),
    (
        "how big is a chunk and how much do neighboring chunks overlap",
        "chunk-size",
        ("parsing.md", "1"),
    ),
    (
        "why must chunk size be counted with the embedding model's own tokenizer",
        "chunk-tokenizer",
        ("parsing.md", "1.2"),
    ),
    (
        "what text is actually embedded for a chunk compared with what is shown",
        "embed-text",
        ("parsing.md", "5"),
    ),
    (
        "what happens when one table is larger than the chunk budget",
        "oversize",
        ("parsing.md", "4.2"),
    ),
    ("which parser is used when the first one fails on a file", "fallback", ("parsing.md", "6")),
    ("how are source code files split into chunks", "code", ("parsing.md", "8")),
    (
        "how are tree-sitter grammars installed on a machine without internet",
        "grammar-bundle",
        ("parsing.md", "8.1.1"),
    ),
    (
        "how does the model cite a passage without writing the citation itself",
        "selects",
        ("generation.md", "3.1"),
    ),
    ("what happens when a citation cannot be verified", "verify-fail", ("generation.md", "3.4")),
    (
        "why does a request time out even though the model is still streaming",
        "timeouts",
        ("generation.md", "4.5"),
    ),
    (
        "is my conversation history used to change what gets retrieved",
        "history",
        ("generation.md", "8.4"),
    ),
    (
        "which kinds of personal data are removed before a prompt is sent to a provider",
        "egress-pii",
        ("generation.md", "7"),
    ),
    ("what does every tool response envelope contain", "envelope", ("surfaces.md", "2")),
    ("how are requests to the http api rate limited", "rate-limit", ("surfaces.md", "9.10")),
    ("how does the http api know who is making a request", "identity", ("surfaces.md", "9.2")),
    (
        "why does bge-m3 need CLS pooling rather than mean pooling",
        "pooling",
        ("embeddings.md", "4.1"),
    ),
    ("what is recorded in the embedding fingerprint", "embed-fingerprint", ("embeddings.md", "5")),
    ("is MLX faster than onnxruntime for embedding", "runtimes", ("embeddings.md", "3")),
    ("how do I run manicule as a background service on a mac", "launchd", ("deployment.md", "6.3")),
    (
        "can anyone who reads the data directory see everything that was indexed",
        "not-permission-aware",
        ("deployment.md", "1"),
    ),
    (
        "how does the gmail connector notice new and deleted messages",
        "gmail-discovery",
        ("connectors/gmail.md", "4"),
    ),
    ("where is the gmail refresh token kept", "gmail-token", ("connectors/gmail.md", "1.2")),
    (
        "how do I sign in to a confluence that sits behind single sign-on",
        "confluence-sso",
        ("connectors/confluence.md", "1.1"),
    ),
    (
        "what happens if I create a memory whose slug already exists",
        "overwrite",
        ("memory-authoring.md", "4.6"),
    ),
    (
        "how are wikilinks between memories turned into relations",
        "wikilinks",
        ("memory-authoring.md", "5"),
    ),
)
"""Question, id and the section that answers it."""

MULTI_HOP_TARGETS: dict[str, tuple[Target, ...]] = {
    "multihop-budget-units": (("parsing.md", "1.2"), ("retrieval.md", "7.2")),
    "multihop-citation-survives-redaction": (("generation.md", "7.4"), ("generation.md", "3.3")),
    "multihop-profile-refusal": (("retrieval.md", "7.4"), ("generation.md", "4.3")),
    "multihop-reranker-confidence": (("retrieval.md", "8.2"), ("retrieval.md", "8.3")),
    "multihop-ollama-endpoint": (("generation.md", "7.1"),),
    "multihop-fast-profile-high": (("retrieval.md", "8.3"),),
    "multihop-collection-vs-tag-scope": (("retrieval.md", "3.1"),),
    "multihop-cli-vs-mcp-absence": (("surfaces.md", "4"),),
    "multihop-ungrounded-vs-not-consulted": (("surfaces.md", "5"), ("retrieval.md", "9.3")),
    "multihop-anchor-lock-and-quote": (("generation.md", "3.6"), ("contracts.md", "1")),
    "multihop-glossary-conflict": (("retrieval.md", "14"),),
}
"""The sections each multi-hop note names, transcribed. A query whose note names a document but
no section, or a file outside ``docs/``, is left out rather than given a target its note does not
state."""

UNANSWERABLE: tuple[str, ...] = (
    "what is the boiling point of water at high altitude",
    "who won the football world cup in 1998",
    "how do I bake sourdough bread with a stiff starter",
    "what is the capital city of australia",
    "how many moons does jupiter have",
    "which vitamins are found in spinach",
    "how do I change the oil in a motorcycle",
    "what time does the sun set in winter in oslo",
    "zzzqqq unrelated nonsense xyzzy",
    "flibber wazzle quanto prink",
    "asdf qwer zxcv",
)
"""Subjects ``docs/`` does not hold, and gibberish. The answer to every one is ``none``."""


TEMPLATES = ("what is {} for", "where is {} used and why")
"""Question frames for :func:`identifier_questions`, alternated by sorted position."""

_IDENTIFIER = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")
_NUMBERED = re.compile(r"^#+ ([0-9][0-9.]*) ")


def _identifier_shaped(term: str) -> bool:
    """Long enough to be rare, and spelled the way only code spells things."""
    return len(term) >= 6 and ("_" in term.strip("_") or re.search(r"[a-z][A-Z]", term) is not None)


def numbered_sections(root: Path = DOCS) -> dict[Target, str]:
    """The text under each numbered heading, by document and section number.

    Text before the first numbered heading of a document is filed under an empty number, so it
    counts as an occurrence without ever being a target.
    """
    sections: dict[Target, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(pattern.removesuffix("**")) for pattern in EXCLUDED):
            continue
        number = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            heading = _NUMBERED.match(line)
            if heading:
                number = heading.group(1).rstrip(".")
            sections[relative, number].append(line)
    return {key: "\n".join(lines) for key, lines in sections.items()}


def identifier_questions(root: Path = DOCS) -> list[tuple[str, Target]]:
    """Each identifier-shaped backticked term found in exactly one numbered section, asked."""
    sections = numbered_sections(root)
    terms = sorted(
        {
            term
            for (_, number), text in sections.items()
            if number
            for term in _IDENTIFIER.findall(text)
            if _identifier_shaped(term)
        }
    )
    questions: list[tuple[str, Target]] = []
    for term in terms:
        bounded = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])")
        where = [key for key, text in sections.items() if bounded.search(text)]
        if len(where) == 1 and where[0][1]:
            template = TEMPLATES[len(questions) % len(TEMPLATES)]
            questions.append((template.format(term), where[0]))
    return questions


@dataclass(frozen=True)
class Asked:
    id: str
    text: str
    group: str
    targets: tuple[Target, ...] = ()
    """Empty for a query scored for emptiness and confidence only."""
    documents: frozenset[str] = frozenset()
    """For a title probe: the documents that are the answer."""


def _section_matches(heading: str, number: str) -> bool:
    first = heading.split(" ", 1)[0].rstrip(".")
    return first == number or first.startswith(number + ".")


@dataclass(frozen=True)
class Place:
    path: str
    heading: tuple[str, ...]
    document: str

    def within(self, target: Target) -> bool:
        path, number = target
        return self.path == path and any(_section_matches(h, number) for h in self.heading)


async def docs_questions(store: SqliteDocStore, workspace: str) -> list[Asked]:
    asked = [
        Asked(f"authored-{name}", text, "authored", (target,)) for text, name, target in AUTHORED
    ]
    multi_hop = load_query_set(DOCS / "evaluation" / "multi-hop-queries.json")
    for query in multi_hop.queries:
        targets = MULTI_HOP_TARGETS.get(query.id, ())
        asked.append(
            Asked(query.id, query.text, "multi-hop" if targets else "multi-hop, unlabeled", targets)
        )
    asked.extend(
        Asked(f"identifier-{i}", text, "identifiers", (target,))
        for i, (text, target) in enumerate(identifier_questions())
    )
    example = load_query_set(DOCS / "evaluation" / "example-queries.json")
    asked.extend(Asked(query.id, query.text, "example") for query in example.queries)
    asked.extend(
        Asked(f"unanswerable-{i}", text, "unanswerable") for i, text in enumerate(UNANSWERABLE)
    )
    for item in await probe_from_titles(store, workspace_ids=frozenset({workspace})):
        asked.append(Asked(f"title-{item.text}", item.text, "titles", documents=item.document_ids))
    return asked


# --- one indexed corpus --------------------------------------------------------------------------


def settings_for(data_dir: Path, overrides: Mapping[str, Any], pipeline: Sequence[str]) -> Settings:
    return Settings(
        data_dir=data_dir,
        rag={"pipeline": tuple(pipeline), "cache": {"enabled": False}},  # pyright: ignore[reportArgumentType]
        **overrides,
    )


@dataclass
class DocsIndex:
    data_dir: Path
    places: dict[str, Place]
    documents: int
    texts: list[tuple[str, str]]
    """Every chunk's text and breadcrumb, for the table the statements are timed against."""


async def index_docs(
    data_dir: Path,
    *,
    overrides: Mapping[str, Any],
    discovery: Discovery | None = None,
    root: Path = DOCS,
) -> DocsIndex:
    """Sync ``root`` through the shipped pipeline and note where every chunk sits."""
    settings = settings_for(data_dir, overrides, PIPELINES["hybrid"])
    async with Runtime(settings, discovery=discovery) as runtime:
        connector = FilesystemConnector(root, name=SOURCE, exclude=EXCLUDED)
        report = await (await runtime.pipeline()).run(connector)
        if report.error or report.by_status.get("failed"):
            msg = f"indexing docs/ failed: {report.error or report.by_status}"
            raise RuntimeError(msg)
        store = cast("SqliteDocStore", await runtime.documents())
        places: dict[str, Place] = {}
        texts: list[tuple[str, str]] = []
        documents = await store.list_documents(limit=10_000)
        for document in documents:
            path = document.uri.removeprefix(root.as_uri() + "/")
            for chunk in await store.document_chunks(document.id):
                places[chunk.id] = Place(path, chunk.heading_path, document.id)
                texts.append((chunk.text, " > ".join(chunk.heading_path)))
    return DocsIndex(data_dir, places, len(documents), texts)


# --- running a question set ----------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    asked: Asked
    variant: str
    pipeline: str
    rank: int | None
    """Of the first chunk inside a target, within the deepest ``k``; ``None`` when none is."""
    found: int
    """Distinct targets reached within the deepest ``k``."""
    lexical_matched: int | None
    """What the lexical leg matched, or ``None`` when the pipeline has no lexical leg."""
    band: ConfidenceBand | None
    lexical_ms: float | None
    exhausted: bool = False
    """A leg stopped at its own cap. Kept and counted: it is how the shipped pipeline answers an
    unanswerable question whose dense candidates all fall under the similarity floor."""


def _rank(asked: Asked, places: Sequence[Place]) -> tuple[int | None, int]:
    if asked.documents:
        rank = next((i for i, p in enumerate(places, 1) if p.document in asked.documents), None)
        return rank, int(rank is not None)
    rank = next(
        (i for i, p in enumerate(places, 1) if any(p.within(t) for t in asked.targets)), None
    )
    found = sum(any(p.within(t) for p in places) for t in asked.targets)
    return rank, found


async def run_docs(
    index: DocsIndex,
    asked: Sequence[Asked],
    *,
    variant: str,
    pipeline: str,
    overrides: Mapping[str, Any],
    discovery: Discovery | None = None,
) -> list[Outcome]:
    """Every query through one pipeline with the store building its match by ``variant``."""
    settings = settings_for(index.data_dir, overrides, PIPELINES[pipeline])
    outcomes: list[Outcome] = []
    with matching(VARIANTS[variant]):
        async with Runtime(settings, discovery=discovery) as runtime:
            retriever = await runtime.retriever()
            if not isinstance(retriever, Retriever) or retriever.cache_available:
                msg = "a measurement needs the shipped Retriever with no cache that can hit"
                raise TypeError(msg)
            scope = Filter(workspace_ids=frozenset({runtime.workspace}))
            for item in asked:
                result = await retriever.retrieve(
                    Query(text=item.text, limit=max(KS), filter=scope)
                )
                refused = [
                    reason
                    for reason in result.trace.incomparable
                    if not reason.startswith(LEXICAL_EMPTY)
                    and reason not in (thread_duplication.ONE_LADDER, EXHAUSTED_BUDGET)
                ]
                if refused:
                    msg = f"{item.id} under {variant}/{pipeline}: {'; '.join(refused)}"
                    raise RuntimeError(msg)
                lexical = next((s for s in result.trace.stages if s.name == "lexical"), None)
                places = [index.places[c.chunk.id] for c in result.candidates[: max(KS)]]
                rank, found = _rank(item, places)
                outcomes.append(
                    Outcome(
                        asked=item,
                        variant=variant,
                        pipeline=pipeline,
                        rank=rank,
                        found=found,
                        lexical_matched=(
                            None if lexical is None else cast("int", lexical.diagnostics["matched"])
                        ),
                        band=None if result.confidence is None else result.confidence.band,
                        lexical_ms=None if lexical is None else lexical.wall_ms,
                        exhausted=EXHAUSTED_BUDGET in result.trace.incomparable,
                    )
                )
    return outcomes


async def run_threads(
    work: Path,
    corpus: Corpus,
    *,
    variants: Sequence[str],
    overrides: Mapping[str, Any],
    discovery: Discovery | None = None,
) -> dict[tuple[str, str], list[thread_duplication.Outcome]]:
    """The mail corpus under each pipeline and variant, indexed afresh every time.

    Unlike ``docs/``, it is never reused from ``--work``: the synthetic mailbox is rebuilt per
    run, its ground truth is which message the generator gave each Gmail id, and a second sync
    into the same directory finds nothing new to index and so has no truth to read back.
    """
    data_dir = work / "threads"
    shutil.rmtree(data_dir, ignore_errors=True)
    indexed = await thread_duplication.index(
        corpus, data_dir, overrides=overrides, discovery=discovery
    )
    _, asked = thread_duplication.queries(corpus)
    runs: dict[tuple[str, str], list[thread_duplication.Outcome]] = {}
    for pipeline in ("dense", "hybrid", "lexical"):
        for variant in variants if pipeline != "dense" else variants[:1]:
            with matching(VARIANTS[variant]):
                run = await thread_duplication.run_pipeline(
                    indexed,
                    asked,
                    name=pipeline,
                    overrides=overrides,
                    ks=(10,),
                    discovery=discovery,
                    counted=frozenset({EXHAUSTED_BUDGET}),
                )
            runs[variant, pipeline] = run.outcomes
    return runs


# --- the statement at scale ----------------------------------------------------------------------


def time_at_scale(
    texts: Sequence[tuple[str, str]],
    questions: Sequence[str],
    *,
    size: int = SCALE,
    variants: Sequence[str] = tuple(VARIANTS),
) -> dict[str, tuple[float, float, float]]:
    """Each variant's median, p95 and mean matched rows for the shipped statement, ``size`` chunks.

    The ``docs/`` chunks are repeated until there are ``size`` of them. Repetition keeps every
    term's document frequency in proportion, and a posting list's length is what a disjunction
    pays for; it also keeps the joins and filters of the real statement, which run over every
    matching row before ``LIMIT``.
    """
    db = sqlite3.connect(":memory:")
    db.executescript(
        f"""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, workspace_id TEXT, deleted_at TEXT, status TEXT
        );
        CREATE TABLE chunks (
            seq INTEGER PRIMARY KEY, id TEXT, document_id TEXT, text TEXT, heading_text TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text, heading_text, content='chunks',
            content_rowid='seq', tokenize='{FTS_TOKENIZER}');
        """
    )
    per_document = 50
    db.executemany(
        "INSERT INTO documents VALUES (?, 'w', NULL, 'indexed')",
        ((f"d{i}",) for i in range(math.ceil(size / per_document))),
    )
    db.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
        ((i, f"c{i}", f"d{i // per_document}", *texts[i % len(texts)]) for i in range(size)),
    )
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    db.commit()
    statement = SEARCH_SQL.format(extra="AND d.workspace_id = :workspace_id")
    timings: dict[str, tuple[float, float, float]] = {}
    for name in variants:
        variant = VARIANTS[name]
        elapsed: list[float] = []
        matched: list[int] = []
        for text in questions:
            match = variant(text)
            started = time.perf_counter()
            rows = db.execute(
                statement, {"match": match, "limit": 50, "workspace_id": "w"}
            ).fetchall()
            elapsed.append((time.perf_counter() - started) * 1000)
            matched.append(
                db.execute(
                    "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (match,)
                ).fetchone()[0]
                if rows
                else 0
            )
        elapsed.sort()
        timings[name] = (
            statistics.median(elapsed),
            elapsed[min(len(elapsed) - 1, math.ceil(0.95 * len(elapsed)) - 1)],
            statistics.fmean(matched),
        )
    db.close()
    return timings


# --- the report ----------------------------------------------------------------------------------


@dataclass
class Row:
    label: str
    queries: int = 0
    empty: int = 0
    hits: dict[int, int] = field(default_factory=dict[int, int])
    reciprocal: float = 0.0
    found: int = 0
    targets: int = 0
    bands: dict[str, int] = field(default_factory=dict[str, int])
    exhausted: int = 0


def summarize(label: str, outcomes: Sequence[Outcome]) -> Row:
    row = Row(label, queries=len(outcomes))
    for outcome in outcomes:
        row.empty += outcome.lexical_matched == 0
        row.exhausted += outcome.exhausted
        for k in KS:
            row.hits[k] = row.hits.get(k, 0) + (outcome.rank is not None and outcome.rank <= k)
        row.reciprocal += 0.0 if outcome.rank is None else 1.0 / outcome.rank
        row.found += outcome.found
        row.targets += len(outcome.asked.targets) or int(bool(outcome.asked.documents))
        if outcome.band is not None:
            row.bands[outcome.band.value] = row.bands.get(outcome.band.value, 0) + 1
    return row


GROUPS = (
    "authored",
    "multi-hop",
    "titles",
    "identifiers",
    "multi-hop, unlabeled",
    "example",
    "unanswerable",
)


def _interval(successes: int, trials: int) -> str:
    low, high = wilson_interval(successes, trials)
    return f"[{low:.2f}, {high:.2f}]"


def render(rows: Sequence[tuple[str, str, Row]]) -> str:
    lines = [
        "| variant | pipeline | set | queries | lexical empty | hit@5 | hit@10 | MRR@10 "
        "| targets found | bands (none/low/medium/high) | exhausted |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for variant, pipeline, row in rows:
        labeled = row.targets > 0
        bands = "/".join(str(row.bands.get(b.value, 0)) for b in reversed(ConfidenceBand))
        lines.append(
            f"| {variant} | {pipeline} | {row.label} | {row.queries} | {row.empty} | "
            + (
                f"{row.hits[5]} | {row.hits[10]} {_interval(row.hits[10], row.queries)} | "
                f"{row.reciprocal / row.queries:.3f} | "
                f"{row.found}/{row.targets} | "
                if labeled
                else "- | - | - | - | "
            )
            + f"{bands} | {row.exhausted} |"
        )
    return "\n".join(lines)


async def report(args: argparse.Namespace, work: Path, emit: Callable[[str], None]) -> None:
    """Index, ask, time and print, in the order the module docstring reads."""
    overrides: dict[str, Any] = {"embedding": {"provider": "onnx", "model": args.model}}
    if args.chunk_tokens is not None:
        overrides["plugins"] = {"config": {"chunker.structural": {"max_tokens": args.chunk_tokens}}}
    started = time.perf_counter()
    index = await index_docs(work / "docs", overrides=overrides)
    settings = settings_for(index.data_dir, overrides, PIPELINES["hybrid"])
    async with Runtime(settings) as runtime:
        store = cast("SqliteDocStore", await runtime.documents())
        asked = await docs_questions(store, runtime.workspace)
    emit(
        f"docs/: {index.documents} documents, {len(index.places)} chunks, indexed in "
        f"{time.perf_counter() - started:.0f}s with {args.model}; {len(asked)} queries"
    )
    rows: list[tuple[str, str, Row]] = []
    for pipeline in ("dense", "hybrid", "lexical"):
        for variant in args.variants if pipeline != "dense" else args.variants[:1]:
            outcomes = await run_docs(
                index, asked, variant=variant, pipeline=pipeline, overrides=overrides
            )
            label = variant if pipeline != "dense" else "-"
            for group in GROUPS:
                chosen = [o for o in outcomes if o.asked.group == group]
                if chosen:
                    rows.append((label, pipeline, summarize(group, chosen)))
            lexical = [o.lexical_ms for o in outcomes if o.lexical_ms is not None]
            if lexical:
                median = statistics.median(lexical)
                emit(f"{variant}/{pipeline}: lexical stage median {median:.2f} ms")
    emit("\n### docs/\n")
    emit(render(rows))

    questions = [a.text for a in asked if a.group != "titles"]
    emit(f"\n### the statement at {args.scale} chunks, {len(questions)} queries\n")
    emit("| variant | median ms | p95 ms | mean rows matched |\n|---|---:|---:|---:|")
    timings = time_at_scale(index.texts, questions, size=args.scale, variants=args.variants)
    for name, (median, p95, matched) in timings.items():
        emit(f"| {name} | {median:.1f} | {p95:.1f} | {matched:.0f} |")

    if args.no_threads:
        return
    corpus = build_corpus(seed=args.seed, copies=args.copies)
    runs = await run_threads(work, corpus, variants=args.variants, overrides=overrides)
    emit("\n### synthetic mail threads, k = 10\n")
    emit("| variant | pipeline | form | queries | one leg | found |")
    emit("|---|---|---|---:|---:|---:|")
    for (variant, pipeline), outcomes in runs.items():
        for form in thread_duplication.FORMS:
            chosen = [o for o in outcomes if o.asked.form == form]
            emit(
                f"| {variant if pipeline != 'dense' else '-'} | {pipeline} | {form} | "
                f"{len(chosen)} | {sum(o.single_leg for o in chosen)} | "
                f"{sum(o.found for o in chosen)} |"
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--model", default="BAAI/bge-m3", help="embedding model (ONNX)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--no-threads", action="store_true", help="skip the synthetic mail corpus")
    parser.add_argument("--seed", type=int, default=463)
    parser.add_argument("--copies", type=int, default=2, help="threads per length and style")
    parser.add_argument("--scale", type=int, default=SCALE, help="chunks to time statements at")
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="keep the docs/ index in this directory, so a second run skips re-embedding it",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="the structural chunker's budget; needed for a model attending to under 512",
    )
    args = parser.parse_args(argv)
    out = console()

    def emit(text: str) -> None:
        # The report holds brackets — intervals, stage lists — that are not markup.
        out.print(text, markup=False)

    with contextlib.ExitStack() as stack:
        if args.work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="lexical-")))
        else:
            work = args.work.resolve()
            work.mkdir(parents=True, exist_ok=True)
        _isolate(work)
        asyncio.run(report(args, work, emit))


if __name__ == "__main__":
    main()
