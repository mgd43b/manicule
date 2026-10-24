"""What the lexical leg contributes to a question, measured under each way of building its match.

The figures in ``docs/retrieval.md`` §4.1 and §5.4 come from here, and so do the decisions
recorded there. Issues `#481 <https://github.com/mgd43b/manicule/issues/481>`_,
`#484 <https://github.com/mgd43b/manicule/issues/484>`_ and
`#486 <https://github.com/mgd43b/manicule/issues/486>`_.

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

**Round 3: ``or-content`` against ``and-content``, on its own.** Round 1 excluded the content-word
disjunction and round 2 never considered it; a later full run found it ahead on the identifier
questions (723 against 600 of 754, MRR 0.811 against 0.637) and behind on multi-hop (7 against 10
of 11) and authored MRR (0.696 against 0.751). Those figures, and round 1's latency for it on
prose questions (39.2 ms median, 70.2 ms p95 at 100,000 chunks), were known when this rule was
written. Two instruments were changed first, because each favored the disjunction:

- *Round 2's identifier set measures one word.* In "what is ``X`` for" every word but ``X`` is a
  function word, so the two rules build the same match; the whole difference lives in "where is
  ``X`` used and why", which is the word "used" asked 377 times. Round 3 asks every identifier
  once more with a second content word from :data:`SECOND_WORDS`, fixed before any run and chosen
  by sorted position (:func:`two_word_questions`).
- *Eleven multi-hop questions cannot show a loss.* The natural-question side adds the synthetic
  mail threads from a seed no earlier run used (:data:`HELD_OUT_SEED`), three copies of every
  length and style, scored by the reciprocal rank of the first chunk carrying the passage.

And latency is reported per question set, since the 11 ms quoted for the disjunction was an
average dominated by one- and two-word identifier questions.

**The round 3 rule.** Hybrid pipeline, ``k = 10``, reciprocal rank per query, paired by query.
``or-content`` replaces ``and-content`` only if every one of these holds:

1. *It gains*: on the two-word identifier questions, its mean reciprocal rank is at least 0.10
   higher, and a two-sided sign test over the queries whose rank differs gives ``p < 0.01``.
2. *It does not lose more than a third of that back*: on the natural questions — authored,
   labeled multi-hop and held-out thread questions, pooled — the lower end of a 95% paired
   bootstrap interval on the mean reciprocal-rank difference is above minus one third of the gain
   in 1, and never below -0.05. The exchange rate is stated so that it can be argued with: a
   question that names an identifier is worth three prose questions, no more.
3. No title and no thread keyword lookup is lost beyond one query each.
4. No unanswerable question leaves the ``none`` band.
5. At :data:`SCALE` chunks, its median statement time on every question set is under 50 ms and
   its p95 under 100 ms.

Otherwise ``and-content`` stays. As a diagnostic and not a condition, the natural-question
differences are split by whether ``and-content`` matched anything, which says whether the
disjunction's cost comes from adding weak matches where the conjunction had none.

**Round 3 returned "switch", and ``or-content`` is now the shipped rule.** Every condition held.
On the two-word identifier questions mean reciprocal rank rose by 0.276 (402 questions better, 63
worse, p = 1.7e-61); ``hit@10`` 723 against 461. On 191 natural questions the change was -0.002,
95% interval [-0.038, +0.034], against a margin of -0.05. No title or thread keyword lookup was
lost and no unanswerable question left ``none``. Latency was the closest condition: the slowest
set, multi-hop, took a median of 48.1 ms and a p95 of 73.4 ms at 100,000 chunks. The diagnostic
found the cost where expected: -0.021 on the 68 natural questions the conjunction had matched,
+0.008 on the 123 it had not.

**Round 3b: a cap on the disjunction, because a long query is not a question.** Review found
what round 3 never asked: a pasted paragraph becomes an ``OR`` over dozens of words, matches
nearly every row, and is scored in full before ``LIMIT`` — and nothing on any surface limits a
query's length. Round 3's largest question had nine content words, so a cap of
:data:`CAP` (16) leaves every figure above unchanged, which a test asserts. Two candidates:
``or-content`` uncapped, and ``or-first-16``, the disjunction over the first 16 content words.

- *pasted* — derived mechanically by :func:`paste_questions`: every ninth chunk of ``docs/``,
  its first 40 words and, where it has them, its first 120, each with the chunk's own numbered
  section as the target. Copied verbatim, so these favor the lexical leg; both candidates share
  that bias.

**The round 3b rule.** Hybrid pipeline, ``k = 10``, paired reciprocal rank over both windows
pooled. ``or-first-16`` ships if the lower end of a 95% paired bootstrap interval on its mean
difference from ``or-content`` is above -0.03, and at :data:`SCALE` chunks its median statement
time on each window is under 50 ms and its p95 under 100 ms — round 3's bound. If it meets the
time bound and fails the margin, the trade-off goes to the owner rather than being decided here;
so does a cap that fails the time bound.

**Round 3b failed the time bound, and the owner shipped the cap.** Quality could not separate
the two — every pasted query found its section at rank 1 under both, and under the dense leg
alone, so the paste set measures time and nothing else. The cap cut a 120-word paste from a
median of 254.9 ms (p95 339.7) to 82.2 ms (p95 100.4), and a 40-word one from 100.4 to 80.0 ms
(p95 124.4 to 97.7): about threefold where it matters, and still over the 50 ms bound, because
sixteen words still match about 90% of the rows. Disclosed as such in ``docs/retrieval.md`` §4.1.

**Round 4: fusion, under the shipped match.** Hybrid ranked document titles worse than the dense
leg alone under every match rule (MRR 0.853 against 0.934), and the cause is reciprocal rank
fusion's consensus: with ``K = 60`` a chunk both legs rank in the middle — dense 17th, lexical 15th
— scores 0.0263 and outranks the dense leg's first, which the lexical leg never returned, at
0.0164. For a one-word title the lexical ladder is mostly noise. Candidates, run with
``--fusion`` over the shipped match (:data:`FUSIONS`): ``K = 30``, ``K = 10``, and the lexical
ladder weighted 0.5 at ``K = 60``. Known when this rule was written: the title, authored and
identifier figures above, for ``K = 60``, and the arithmetic in this paragraph — which says a 0.5
weight cannot reverse that example and a smaller ``K`` can.

**The round 4 rule.** Hybrid pipeline, ``k = 10``, reciprocal rank paired by query, in three
families: *prose* (authored, labeled multi-hop, held-out thread questions), *keyword* (titles and
thread keyword lookups) and *identifiers* (both identifier sets). A candidate is admissible only
if, in every family, the lower end of a 95% paired bootstrap interval on its mean difference from
``K = 60`` is above -0.02, and no unanswerable question leaves ``none``. An admissible candidate
qualifies if some family gains at least 0.02 with a sign test giving ``p < 0.01``. The qualifier
with the highest mean of its three family gains replaces ``K = 60``; within 0.005, the earlier in
:data:`FUSIONS` wins, which puts the configuration-only changes before the one that needs code.
No qualifier: no change.

**Round 4 returned "no change".** No candidate qualified. ``K = 30`` moved no family's mean by more
than 0.003, though titles alone fell from 0.853 to 0.826: pooled with 126 thread keyword lookups, 19
titles barely move the keyword family, which is why round 5 scores them on their own
(:mod:`tests.benchmarks.title_lookups`). ``K = 10`` gained 0.014 on the identifier questions (94
better, none worse, p = 1e-28) — real, and under the 0.02 the rule asked for — and made titles
*worse*, MRR 0.853 to 0.809, the opposite of what the arithmetic above predicted: a sharper ``K``
sharpens the lexical ladder's top as well, and for a one-word title its top is often the wrong
document. The 0.5 lexical weight did lift titles to 0.896 and cost the identifier questions 0.212
(590 worse, 8 better), so it was not admissible. The title gap to the dense leg alone is real and
stays: every way of closing it measured here costs more elsewhere than it gains.

**Replication under a second embedder: Qwen3-Embedding-0.6B.** Every round above measured
BGE-M3. Rounds 3, 3b and 4 are rerun unchanged with ``qwen3-embedding:0.6b`` served by the
``ollama`` backend, ``prefix_scheme = "qwen3"`` and the model's own tokenizer
(``--provider ollama --model qwen3-embedding:0.6b --prefix-scheme qwen3 --tokenizer <dir>``).
What the result would change was written down before any of its output was read:

- *Round 3.* Its rule is applied as written. If ``or-content`` meets every condition, the shipped
  rule holds under both models. If it fails condition 1 or 2 under Qwen3, the match rule is not
  model-independent, and that goes to the owner as a decision about the shipped default — it is
  not reverted here. Conditions 3 to 5 concern the lexical statement and the confidence bands;
  the statement's time does not involve the embedder, so condition 5 can only repeat itself.
- *Round 3b.* The paste set's quality side could not separate the candidates under BGE-M3. If it
  can under Qwen3 and the cap loses more than the margin, that goes to the owner too.
- *Round 4.* ``K`` is one setting per installation (``plugins.config."retrieval_stage.rrf".k``),
  so the rule is applied per model. The shipped default changes only if both models qualify the
  same candidate. A candidate that qualifies under Qwen3 alone is recorded in
  ``docs/retrieval.md`` §5.4 as the setting for an installation embedding with it, not shipped.

**The replication.** Round 4 agreed: no candidate qualified, ``K = 10`` again gained a little on
identifiers (0.007) and the halved lexical ladder again cost them 0.160. Round 3b agreed: the cap
cost 0.005 of reciprocal rank on the pastes, inside the margin, and still missed the time bound.
Round 3 met conditions 1 to 4 — identifiers +0.257 (p = 4.8e-53), natural questions +0.012 with
a 95% interval of [-0.025, +0.049] — and missed condition 5: multi-hop took a median of 52.6 ms
at 100,000 chunks against 48.1 under BGE-M3. The criterion above was wrong to say the time could
only repeat itself. The statement does not involve the embedder, but the timed table is built
from the indexed chunks, and Qwen3's tokenizer cuts ``docs/`` into 1,198 longer chunks rather than
1,352, so a prose question matches more rows of each. The owner judged the miss small and
specific to this corpus and kept the shipped rule. Prose costs more under Qwen3 than under
BGE-M3 — authored MRR 0.635 against 0.716 for the conjunction — and the held-out thread questions
are what bring the pooled figure level.

Everything is local: no network, temporary directories, and the only thing read from outside the
repository is the embedding model, from the local model cache.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import math
import random
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
from manicule.evaluation.statistics import sign_test, wilson_interval
from manicule.retrieval import fusion as fusion_module
from manicule.retrieval import trace as trace_module
from manicule.retrieval.lexical import DEGRADED_REASON as LEXICAL_EMPTY
from manicule.retrieval.retriever import Retriever
from manicule.retrieval.trace import Shortfall
from manicule.storage import docstore as docstore_module
from manicule.storage.fts import (
    FUNCTION_WORDS,
    MAX_MATCH_WORDS,
    SEARCH_SQL,
    escape_match_query,
    match_words,
)
from manicule.storage.models import FTS_TOKENIZER
from tests.benchmarks import thread_duplication
from tests.benchmarks.thread_corpus import Corpus, build_corpus
from tests.benchmarks.thread_duplication import _isolate  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping, Sequence

    from manicule.core.retrieval import Candidate
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


def match_all_content(text: str) -> str:
    """Every word that is not a function word: the rule #482 shipped and round 3 replaced."""
    return " ".join(_content(_literals(text)))


CAP = MAX_MATCH_WORDS
"""Round 3b's cap on the disjunction, in content words. Above round 3's largest question (nine)."""


def match_any_content(text: str) -> str:
    """Any word that is not a function word, uncapped: the rule round 3 measured and chose."""
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
    "and-content": match_all_content,
    "all-but-one": match_all_but_one_content,
    "or-first-16": escape_match_query,
}
"""Report order, simplest first, which is also rule 2's tie-break order.

``and`` is the rule #481 replaced and ``and-content`` the one #482 shipped. ``or-content`` — any
word but the :data:`~manicule.storage.fts.FUNCTION_WORDS` — won round 3, and ``or-first-16``, the
same capped at :data:`~manicule.storage.fts.MAX_MATCH_WORDS`, shipped after round 3b. It is the
shipped :func:`~manicule.storage.fts.escape_match_query` itself rather than a copy of it, so a
change to the shipped rule is measured by the next run without anybody remembering to edit this
list."""


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


def indexed_markdown(root: Path = DOCS) -> Iterator[tuple[str, list[str]]]:
    """Every Markdown file the harness indexes, by path relative to ``root``, as its lines."""
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        if not any(relative.startswith(pattern.removesuffix("**")) for pattern in EXCLUDED):
            yield relative, path.read_text(encoding="utf-8").splitlines()


_FENCE = re.compile(r"^ {0,3}(```|~~~)")


def numbered_headings(root: Path = DOCS) -> list[tuple[str, str, str]]:
    """Every numbered heading as its document, its section number and its text after the number.

    A line inside a fenced code block is not a heading, however it starts: ``deployment.md``'s
    shell examples number their steps ``# 1.`` to ``# 3.``. :func:`numbered_sections` does not
    skip fences, and keeps not skipping them, because the identifier sets rounds 2 to 4 recorded
    were derived through it.
    """
    found: list[tuple[str, str, str]] = []
    for relative, lines in indexed_markdown(root):
        fence: str | None = None
        for line in lines:
            if opened := _FENCE.match(line):
                fence = None if fence == opened.group(1) else fence or opened.group(1)
            elif (
                fence is None
                and (heading := _NUMBERED.match(line))
                and (text := line[heading.end() :].strip())
            ):
                found.append((relative, heading.group(1).rstrip("."), text))
    return found


def numbered_sections(root: Path = DOCS) -> dict[Target, str]:
    """The text under each numbered heading, by document and section number.

    Text before the first numbered heading of a document is filed under an empty number, so it
    counts as an occurrence without ever being a target.
    """
    sections: dict[Target, list[str]] = defaultdict(list)
    for relative, lines in indexed_markdown(root):
        number = ""
        for line in lines:
            heading = _NUMBERED.match(line)
            if heading:
                number = heading.group(1).rstrip(".")
            sections[relative, number].append(line)
    return {key: "\n".join(lines) for key, lines in sections.items()}


def single_section_identifiers(root: Path = DOCS) -> list[tuple[str, Target]]:
    """Each identifier-shaped backticked term found in exactly one numbered section, sorted."""
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
    found: list[tuple[str, Target]] = []
    for term in terms:
        bounded = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])")
        where = [key for key, text in sections.items() if bounded.search(text)]
        if len(where) == 1 and where[0][1]:
            found.append((term, where[0]))
    return found


def identifier_questions(root: Path = DOCS) -> list[tuple[str, Target]]:
    """Round 2's set: each identifier in one of :data:`TEMPLATES`, alternated by position."""
    return [
        (TEMPLATES[i % len(TEMPLATES)].format(term), target)
        for i, (term, target) in enumerate(single_section_identifiers(root))
    ]


SECOND_WORDS = ("configured", "returned", "raised", "stored", "checked", "set", "read", "written")
"""Round 3's second content words, fixed before its first run. Verbs a person asks about a name
with, chosen without looking at which of them the sections happen to contain."""


def two_word_questions(root: Path = DOCS) -> list[tuple[str, Target]]:
    """Round 3's set: "when is ``X`` <word>", the word taken from :data:`SECOND_WORDS` in turn.

    Every question has exactly two content words, so the two rules build different matches for
    every one of them — unlike round 2's "what is ``X`` for", where they build the same one.
    """
    return [
        (f"when is {term} {SECOND_WORDS[i % len(SECOND_WORDS)]}", target)
        for i, (term, target) in enumerate(single_section_identifiers(root))
    ]


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


async def docs_questions(store: SqliteDocStore, workspace: str, root: Path = DOCS) -> list[Asked]:
    asked = [
        Asked(f"authored-{name}", text, "authored", (target,)) for text, name, target in AUTHORED
    ]
    multi_hop = load_query_set(root / "evaluation" / "multi-hop-queries.json")
    for query in multi_hop.queries:
        targets = MULTI_HOP_TARGETS.get(query.id, ())
        asked.append(
            Asked(query.id, query.text, "multi-hop" if targets else "multi-hop, unlabeled", targets)
        )
    asked.extend(
        Asked(f"identifier-{i}", text, "identifiers", (target,))
        for i, (text, target) in enumerate(identifier_questions(root))
    )
    asked.extend(
        Asked(f"two-words-{i}", text, "identifiers, two words", (target,))
        for i, (text, target) in enumerate(two_word_questions(root))
    )
    example = load_query_set(root / "evaluation" / "example-queries.json")
    asked.extend(Asked(query.id, query.text, "example") for query in example.queries)
    asked.extend(
        Asked(f"unanswerable-{i}", text, "unanswerable") for i, text in enumerate(UNANSWERABLE)
    )
    for item in await probe_from_titles(store, workspace_ids=frozenset({workspace})):
        asked.append(Asked(f"title-{item.text}", item.text, "titles", documents=item.document_ids))
    return asked


PASTE_STRIDE = 9
PASTE_WINDOWS = (40, 120)


def paste_questions(passages: Sequence[tuple[Place, str]]) -> list[Asked]:
    """Round 3b's long queries: the opening words of every :data:`PASTE_STRIDE`-th chunk.

    The target is the chunk's deepest numbered section; a chunk under no numbered heading is
    skipped rather than given a target its document does not state, and a window longer than the
    chunk is skipped rather than shortened, so every query is exactly its window's length.
    """
    asked: list[Asked] = []
    for position, (place, text) in enumerate(passages[::PASTE_STRIDE]):
        numbers = [
            heading.split(" ", 1)[0].rstrip(".")
            for heading in place.heading
            if heading[:1].isdigit()
        ]
        if not numbers:
            continue
        words = text.split()
        for window in PASTE_WINDOWS:
            if len(words) >= window:
                asked.append(
                    Asked(
                        f"pasted-{window}-{position}",
                        " ".join(words[:window]),
                        f"pasted, {window} words",
                        ((place.path, numbers[-1]),),
                    )
                )
    return asked


# --- one indexed corpus --------------------------------------------------------------------------


def settings_for(data_dir: Path, overrides: Mapping[str, Any], pipeline: Sequence[str]) -> Settings:
    return thread_duplication.settings_for(data_dir, overrides, pipeline)


@dataclass
class DocsIndex:
    data_dir: Path
    places: dict[str, Place]
    documents: int
    texts: list[tuple[str, str]]
    """Every chunk's text and breadcrumb, for the table the statements are timed against."""
    passages: list[tuple[Place, str]] = field(default_factory=list[tuple[Place, str]])
    """Every chunk's place and text, by document path and then position in the document."""


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
        passages: list[tuple[Place, str]] = []
        documents = await store.list_documents(limit=10_000)
        for document in documents:
            path = document.uri.removeprefix(root.as_uri() + "/")
            for chunk in await store.document_chunks(document.id):
                places[chunk.id] = Place(path, chunk.heading_path, document.id)
                texts.append((chunk.text, " > ".join(chunk.heading_path)))
                passages.append((places[chunk.id], chunk.text))
    passages.sort(key=lambda passage: passage[0].path)
    return DocsIndex(data_dir, places, len(documents), texts, passages)


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
    questions: Mapping[str, Sequence[str]],
    *,
    size: int = SCALE,
    variants: Sequence[str] = tuple(VARIANTS),
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Median, p95 and mean matched rows of the shipped statement, at ``size`` chunks.

    Keyed by variant and by question set, because a mean over every set is a mean over whichever
    set is largest: round 2's 754 identifier questions have one or two content words each, so
    they made the disjunction look cheap on the prose questions it costs most on.

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
    timings: dict[tuple[str, str], tuple[float, float, float]] = {}
    for name, (group, texts_asked) in itertools.product(variants, questions.items()):
        variant = VARIANTS[name]
        elapsed: list[float] = []
        matched: list[int] = []
        for text in texts_asked:
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
        timings[name, group] = (
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
    "identifiers, two words",
    "pasted, 40 words",
    "pasted, 120 words",
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


# --- round 3 ------------------------------------------------------------------------------------

INCUMBENT = "and-content"
"""The rule round 3 tested against: shipped by #482, replaced on round 3's result."""
CHALLENGER = "or-content"
HELD_OUT_SEED = 481
"""The thread corpus's seed for round 3. 463 built rounds 1 and 2, and 1729 was #463's own
held-out check; no lexical-variant run had seen this one before round 3's rule was written."""
HELD_OUT_COPIES = 3
HELD_OUT_DISTRACTORS = 5
"""Three copies of every length and style use 45 of the generator's 50 topics, which leaves five
for distractors rather than twenty."""
NATURAL = ("authored", "multi-hop")
"""The labeled ``docs/`` sets that join the held-out thread questions on round 3's loss side."""
TWO_WORDS = "identifiers, two words"


def reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


@dataclass(frozen=True)
class Pair:
    """One query under both rules: reciprocal ranks, and whether the shipped rule matched at all."""

    id: str
    shipped: float
    challenger: float
    shipped_empty: bool

    @property
    def difference(self) -> float:
        return self.challenger - self.shipped


def docs_pairs(
    outcomes: Mapping[tuple[str, str], Sequence[Outcome]], groups: Sequence[str]
) -> list[Pair]:
    """Hybrid outcomes of the two rules on ``docs/``, paired by query id."""
    challenger = {o.asked.id: o for o in outcomes[CHALLENGER, "hybrid"]}
    return [
        Pair(
            o.asked.id,
            reciprocal(o.rank),
            reciprocal(challenger[o.asked.id].rank),
            o.lexical_matched == 0,
        )
        for o in outcomes[INCUMBENT, "hybrid"]
        if o.asked.group in groups
    ]


def thread_pairs(
    runs: Mapping[tuple[str, str], Sequence[thread_duplication.Outcome]], form: str
) -> list[Pair]:
    """Hybrid outcomes of the two rules on the mail threads, paired by query id."""
    challenger = {o.asked.query.id: o for o in runs[CHALLENGER, "hybrid"]}
    return [
        Pair(
            o.asked.query.id,
            reciprocal(o.first),
            reciprocal(challenger[o.asked.query.id].first),
            o.single_leg,
        )
        for o in runs[INCUMBENT, "hybrid"]
        if o.asked.form == form
    ]


def bootstrap_interval(
    differences: Sequence[float], *, resamples: int = 10_000, seed: int = HELD_OUT_SEED
) -> tuple[float, float]:
    """A 95% percentile interval on the mean of paired differences, resampled with a fixed seed."""
    rng = random.Random(seed)  # noqa: S311 - a reproducible resample, not a key
    count = len(differences)
    means = sorted(statistics.fmean(rng.choices(differences, k=count)) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


@dataclass(frozen=True)
class RoundThree:
    """Round 3's rule applied: every condition, its figures, and the diagnostic."""

    gain: float
    gain_split: tuple[int, int]
    gain_p: float
    natural: float
    natural_interval: tuple[float, float]
    natural_count: int
    lost_titles: int
    lost_keywords: int
    unanswerable_out_of_none: int
    latency: dict[str, tuple[float, float]]
    """The challenger's median and p95 ms at :data:`SCALE` chunks, per question set. Both limits
    hold for every set: the set with the slowest median need not be the one with the worst p95."""
    by_match: dict[bool, tuple[int, float]]
    """Natural-question pairs by whether the shipped rule matched nothing: count, mean change."""

    @property
    def margin(self) -> float:
        return max(-0.05, -self.gain / 3)

    @property
    def conditions(self) -> dict[str, bool]:
        return {
            "1 gain": self.gain >= 0.10 and self.gain_p < 0.01,
            "2 not inferior": self.natural_interval[0] > self.margin,
            "3 titles and keywords": self.lost_titles <= 1 and self.lost_keywords <= 1,
            "4 unanswerable": self.unanswerable_out_of_none == 0,
            "5 latency": all(median < 50 and p95 < 100 for median, p95 in self.latency.values()),
        }

    @property
    def switch(self) -> bool:
        return all(self.conditions.values())

    def describe(self) -> str:
        low, high = self.natural_interval
        lines = [
            f"  1 two-word identifiers: mean reciprocal rank {self.gain:+.3f}, better/worse "
            f"{self.gain_split[0]}/{self.gain_split[1]}, sign test p = {self.gain_p:.2g}",
            f"  2 natural questions ({self.natural_count}): mean reciprocal rank "
            f"{self.natural:+.3f}, 95% interval [{low:+.3f}, {high:+.3f}], margin "
            f"{self.margin:+.3f}",
            f"  3 titles lost {self.lost_titles}, thread keywords lost {self.lost_keywords}",
            f"  4 unanswerable questions out of none: {self.unanswerable_out_of_none}",
            f"  5 {CHALLENGER}: slowest median {max(self.latency.items(), key=lambda t: t[1][0])}"
            f", worst p95 {max(self.latency.items(), key=lambda t: t[1][1])}",
        ]
        lines.extend(f"  {name}: {held}" for name, held in self.conditions.items())
        for empty, (count, change) in sorted(self.by_match.items()):
            lines.append(
                f"  diagnostic: {count} natural questions where {INCUMBENT} matched "
                f"{'nothing' if empty else 'something'}, mean change {change:+.3f}"
            )
        lines.append(f"  -> {'switch to ' + CHALLENGER if self.switch else 'keep ' + INCUMBENT}")
        return "\n".join(lines)


def _lost(incumbent: Sequence[bool], challenger: Sequence[bool]) -> int:
    """Lookups the incumbent found and the challenger did not, paired by query: a gain elsewhere
    does not pay for one."""
    return sum(was and not now for was, now in zip(incumbent, challenger, strict=True))


def round_three_latency(
    timings: Mapping[tuple[str, str], tuple[float, float, float]],
) -> dict[str, tuple[float, float]]:
    """The challenger's median and p95 per question set that round 3's condition 5 reads.

    Round 3's sets only: the pastes are round 3b's subject and were added after round 3's rule
    was written, so reading them here would apply a bound round 3 never set to them.
    """
    return {
        group: (median, p95)
        for (name, group), (median, p95, _) in timings.items()
        if name == CHALLENGER and group not in PASTED
    }


def round_three(
    outcomes: Mapping[tuple[str, str], Sequence[Outcome]],
    runs: Mapping[tuple[str, str], Sequence[thread_duplication.Outcome]],
    timings: Mapping[tuple[str, str], tuple[float, float, float]],
) -> RoundThree:
    """Apply the round 3 rule the module docstring states."""
    gained = docs_pairs(outcomes, (TWO_WORDS,))
    better = sum(p.difference > 0 for p in gained)
    worse = sum(p.difference < 0 for p in gained)
    natural = docs_pairs(outcomes, NATURAL) + thread_pairs(runs, "question")
    differences = [p.difference for p in natural]

    titles = {
        variant: [
            o.rank is not None and o.rank <= max(KS)
            for o in outcomes[variant, "hybrid"]
            if o.asked.group == "titles"
        ]
        for variant in (INCUMBENT, CHALLENGER)
    }
    keywords = {
        variant: [o.found for o in runs[variant, "hybrid"] if o.asked.form == "keywords"]
        for variant in (INCUMBENT, CHALLENGER)
    }
    by_match: dict[bool, tuple[int, float]] = {}
    for empty in (True, False):
        chosen = [p.difference for p in natural if p.shipped_empty is empty]
        if chosen:
            by_match[empty] = (len(chosen), statistics.fmean(chosen))
    latency = round_three_latency(timings)
    return RoundThree(
        gain=statistics.fmean(p.difference for p in gained),
        gain_split=(better, worse),
        gain_p=sign_test(better, worse),
        natural=statistics.fmean(differences),
        natural_interval=bootstrap_interval(differences),
        natural_count=len(natural),
        lost_titles=_lost(titles[INCUMBENT], titles[CHALLENGER]),
        lost_keywords=_lost(keywords[INCUMBENT], keywords[CHALLENGER]),
        unanswerable_out_of_none=sum(
            o.band is not ConfidenceBand.NONE
            for o in outcomes[CHALLENGER, "hybrid"]
            if o.asked.group == "unanswerable"
        ),
        latency=latency,
        by_match=by_match,
    )


CAPPED = "or-first-16"
PASTED: tuple[str, ...] = ("pasted, 40 words", "pasted, 120 words")


@dataclass(frozen=True)
class RoundThreeB:
    """Round 3b's rule applied: the cap against the uncapped disjunction on long queries."""

    change: float
    interval: tuple[float, float]
    count: int
    timings: dict[str, tuple[float, float]]
    """Per paste window: the cap's median and p95 ms at :data:`SCALE` chunks."""

    @property
    def within_margin(self) -> bool:
        return self.interval[0] > -0.03

    @property
    def fast_enough(self) -> bool:
        return all(median < 50 and p95 < 100 for median, p95 in self.timings.values())

    @property
    def verdict(self) -> str:
        if self.within_margin and self.fast_enough:
            return f"ship {CAPPED}"
        return "a trade-off for the owner to decide"

    def describe(self) -> str:
        low, high = self.interval
        lines = [
            f"  pasted ({self.count}): mean reciprocal rank {self.change:+.3f}, 95% interval "
            f"[{low:+.3f}, {high:+.3f}], margin -0.030 -> {self.within_margin}",
            *(
                f"  {group}: {CAPPED} median {median:.1f} ms, p95 {p95:.1f} ms"
                for group, (median, p95) in self.timings.items()
            ),
            f"  time bound -> {self.fast_enough}",
            f"  -> {self.verdict}",
        ]
        return "\n".join(lines)


def round_three_b(
    outcomes: Mapping[tuple[str, str], Sequence[Outcome]],
    timings: Mapping[tuple[str, str], tuple[float, float, float]],
) -> RoundThreeB:
    """Apply the round 3b rule the module docstring states."""
    capped = {o.asked.id: o for o in outcomes[CAPPED, "hybrid"]}
    differences = [
        reciprocal(capped[o.asked.id].rank) - reciprocal(o.rank)
        for o in outcomes[CHALLENGER, "hybrid"]
        if o.asked.group in PASTED
    ]
    return RoundThreeB(
        change=statistics.fmean(differences),
        interval=bootstrap_interval(differences),
        count=len(differences),
        timings={group: timings[CAPPED, group][:2] for group in PASTED},
    )


# --- round 4 ------------------------------------------------------------------------------------

SHIPPED_MATCH = "or-first-16"


@dataclass(frozen=True)
class Fusion:
    """One fusion configuration: the RRF constant and the lexical ladder's weight."""

    k: int = 60
    lexical_weight: float = 1.0


FUSIONS: dict[str, Fusion] = {
    "k60": Fusion(),
    "k30": Fusion(k=30),
    "k10": Fusion(k=10),
    "k60-lexical-0.5": Fusion(lexical_weight=0.5),
}
"""Round 4's incumbent first, then its candidates in tie-break order."""

FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "prose": (("authored", "multi-hop"), ("question",)),
    "keyword": (("titles",), ("keywords",)),
    "identifiers": (("identifiers", TWO_WORDS), ()),
}
"""Each family's ``docs/`` groups and mail-thread query forms."""


def fusion_overrides(overrides: Mapping[str, Any], fusion: Fusion) -> dict[str, Any]:
    """``overrides`` with the RRF constant set through the fusion stage's own configuration."""
    merged = dict(overrides)
    plugins = dict(merged.get("plugins", {}))
    config = dict(plugins.get("config", {}))
    config["retrieval_stage.rrf"] = {"k": fusion.k}
    plugins["config"] = config
    merged["plugins"] = plugins
    return merged


@contextlib.contextmanager
def weighting(lexical_weight: float) -> Generator[None]:
    """Fuse with the lexical ladder's reciprocal ranks scaled by ``lexical_weight``.

    The shipped stage has no leg weights (``docs/retrieval.md`` §5.1), so the candidate is
    measured by substituting its arithmetic for the duration: the same ladders, recovered the
    same way and tie-broken the same way, with one leg's contribution multiplied.
    """
    if lexical_weight == 1.0:
        yield
        return
    with weighted_fusion(lambda _: lexical_weight):
        yield


@contextlib.contextmanager
def weighted_fusion(lexical_weight: Callable[[Query], float]) -> Generator[None]:
    """Fuse each query with its lexical ladder scaled by ``lexical_weight(query)``.

    The one substitute for the fusion stage's arithmetic every round measuring a leg weight uses
    — round 4's constant through :func:`weighting`, round 6's per query shape — so they measure
    the same fusion apart from the weight. A weight of 0 leaves the lexical candidates in the
    list, ordered after everything the dense leg ranked, as fusion orders a zero score; only a
    leg that returned nothing marks the run degraded.
    """
    shipped = fusion_module.RRFStage.run

    async def run(
        self: fusion_module.RRFStage, query: Query, candidates: list[Candidate]
    ) -> list[Candidate]:
        weight = lexical_weight(query)
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
            for rank, candidate in enumerate(ladder, start=1):
                fused[candidate.chunk.id] = fused.get(candidate.chunk.id, 0.0) + scale / (
                    self.k + rank
                )
                seen.setdefault(candidate.chunk.id, set()).add(leg)
        # Recorded exactly as the shipped stage records them, so a weighted run's trace carries
        # the same leg counts, overlap and degraded flag as every other candidate's.
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


@dataclass(frozen=True)
class FamilyChange:
    count: int
    change: float
    interval: tuple[float, float]
    split: tuple[int, int]

    @property
    def p(self) -> float:
        return sign_test(*self.split)


@dataclass(frozen=True)
class RoundFour:
    """One candidate against ``K = 60``, per family."""

    families: dict[str, FamilyChange]
    unanswerable_out_of_none: int

    @property
    def admissible(self) -> bool:
        return self.unanswerable_out_of_none == 0 and all(
            family.interval[0] > -0.02 for family in self.families.values()
        )

    @property
    def qualifies(self) -> bool:
        return self.admissible and any(
            family.change >= 0.02 and family.p < 0.01 for family in self.families.values()
        )

    @property
    def gain(self) -> float:
        return statistics.fmean(family.change for family in self.families.values())

    def describe(self) -> str:
        lines = [
            f"  {name} ({f.count}): {f.change:+.3f} [{f.interval[0]:+.3f}, {f.interval[1]:+.3f}], "
            f"better/worse {f.split[0]}/{f.split[1]}, p = {f.p:.2g}"
            for name, f in self.families.items()
        ]
        lines.append(
            f"  unanswerable out of none {self.unanswerable_out_of_none}; admissible "
            f"{self.admissible}, qualifies {self.qualifies}, mean gain {self.gain:+.3f}"
        )
        return "\n".join(lines)


def _reciprocals(
    docs: Sequence[Outcome], threads: Sequence[thread_duplication.Outcome], family: str
) -> dict[str, float]:
    groups, forms = FAMILIES[family]
    found = {o.asked.id: reciprocal(o.rank) for o in docs if o.asked.group in groups}
    found.update({o.asked.query.id: reciprocal(o.first) for o in threads if o.asked.form in forms})
    return found


def round_four(
    docs: Mapping[str, Sequence[Outcome]],
    threads: Mapping[str, Sequence[thread_duplication.Outcome]],
) -> dict[str, RoundFour]:
    """Every candidate in :data:`FUSIONS` against the first, under round 4's rule."""
    incumbent, *candidates = FUSIONS
    verdicts: dict[str, RoundFour] = {}
    for name in candidates:
        families: dict[str, FamilyChange] = {}
        for family in FAMILIES:
            before = _reciprocals(docs[incumbent], threads[incumbent], family)
            after = _reciprocals(docs[name], threads[name], family)
            differences = [after[key] - value for key, value in before.items()]
            families[family] = FamilyChange(
                count=len(differences),
                change=statistics.fmean(differences),
                interval=bootstrap_interval(differences),
                split=(sum(d > 0 for d in differences), sum(d < 0 for d in differences)),
            )
        verdicts[name] = RoundFour(
            families,
            sum(
                o.band is not ConfidenceBand.NONE
                for o in docs[name]
                if o.asked.group == "unanswerable"
            ),
        )
    return verdicts


def round_four_winner(verdicts: Mapping[str, RoundFour]) -> str | None:
    """The qualifier with the highest mean gain; within 0.005, the earlier in :data:`FUSIONS`."""
    qualifying = [name for name in FUSIONS if name in verdicts and verdicts[name].qualifies]
    if not qualifying:
        return None
    best = max(verdicts[name].gain for name in qualifying)
    return next(name for name in qualifying if verdicts[name].gain >= best - 0.005)


def embedding_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The embedder the command line names, and the chunk budget it needs, as settings."""
    embedding: dict[str, Any] = {"provider": args.provider, "model": args.model}
    if args.prefix_scheme is not None:
        embedding["prefix_scheme"] = args.prefix_scheme
    config: dict[str, Any] = {}
    if args.chunk_tokens is not None:
        config["chunker.structural"] = {"max_tokens": args.chunk_tokens}
    if args.provider == "ollama":
        config["embedder.ollama"] = {"base_url": args.ollama_url}
    if args.tokenizer is not None:
        config.setdefault(f"embedder.{args.provider}", {})["tokenizer"] = args.tokenizer
    overrides: dict[str, Any] = {"embedding": embedding}
    if config:
        overrides["plugins"] = {"config": config}
    return overrides


async def report_fusion(args: argparse.Namespace, work: Path, emit: Callable[[str], None]) -> None:
    """Round 4: every fusion in :data:`FUSIONS` over both corpora, then the verdict."""
    overrides = embedding_overrides(args)
    index = await index_docs(work / "docs", overrides=overrides)
    settings = settings_for(index.data_dir, overrides, PIPELINES["hybrid"])
    async with Runtime(settings) as runtime:
        store = cast("SqliteDocStore", await runtime.documents())
        asked = await docs_questions(store, runtime.workspace)
    corpus = build_corpus(seed=args.seed, copies=args.copies, distractors=args.distractors)
    data_dir = work / "threads"
    shutil.rmtree(data_dir, ignore_errors=True)
    indexed = await thread_duplication.index(corpus, data_dir, overrides=overrides)
    _, thread_asked = thread_duplication.queries(corpus)
    docs: dict[str, list[Outcome]] = {}
    threads: dict[str, list[thread_duplication.Outcome]] = {}
    rows: list[tuple[str, str, Row]] = []
    for name, fusion in FUSIONS.items():
        merged = fusion_overrides(overrides, fusion)
        with weighting(fusion.lexical_weight):
            docs[name] = await run_docs(
                index, asked, variant=SHIPPED_MATCH, pipeline="hybrid", overrides=merged
            )
            with matching(VARIANTS[SHIPPED_MATCH]):
                run = await thread_duplication.run_pipeline(
                    indexed,
                    thread_asked,
                    name="hybrid",
                    overrides=merged,
                    ks=(10,),
                    counted=frozenset({EXHAUSTED_BUDGET}),
                )
        threads[name] = run.outcomes
        for group in GROUPS:
            chosen = [o for o in docs[name] if o.asked.group == group]
            if chosen:
                rows.append((name, "hybrid", summarize(group, chosen)))
    emit("\n### round 4: fusion, docs/\n")
    emit(render(rows))
    emit("\n### round 4: fusion, mail threads\n")
    for name, outcomes in threads.items():
        for form in thread_duplication.FORMS:
            chosen = [o for o in outcomes if o.asked.form == form]
            mrr = statistics.fmean(reciprocal(o.first) for o in chosen)
            found = sum(o.found for o in chosen)
            emit(f"  {name} {form}: found {found}/{len(chosen)}, MRR {mrr:.3f}")
    verdicts = round_four(docs, threads)
    for name, verdict in verdicts.items():
        emit(f"\n{name} against {next(iter(FUSIONS))}:\n{verdict.describe()}")
    winner = round_four_winner(verdicts)
    emit(f"\n-> {'replace K = 60 with ' + winner if winner else 'no change'}")


async def _report_docs(
    index: DocsIndex,
    asked: Sequence[Asked],
    variants: Sequence[str],
    overrides: Mapping[str, Any],
    emit: Callable[[str], None],
) -> dict[tuple[str, str], list[Outcome]]:
    """Every pipeline and variant over ``docs/``, printed, and kept for round 3."""
    rows: list[tuple[str, str, Row]] = []
    measured: dict[tuple[str, str], list[Outcome]] = {}
    for pipeline in ("dense", "hybrid", "lexical"):
        for variant in variants if pipeline != "dense" else variants[:1]:
            outcomes = await run_docs(
                index, asked, variant=variant, pipeline=pipeline, overrides=overrides
            )
            measured[variant, pipeline] = outcomes
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
    return measured


async def report(args: argparse.Namespace, work: Path, emit: Callable[[str], None]) -> None:
    """Index, ask, time and print, in the order the module docstring reads."""
    overrides = embedding_overrides(args)
    started = time.perf_counter()
    index = await index_docs(work / "docs", overrides=overrides)
    settings = settings_for(index.data_dir, overrides, PIPELINES["hybrid"])
    async with Runtime(settings) as runtime:
        store = cast("SqliteDocStore", await runtime.documents())
        asked = await docs_questions(store, runtime.workspace)
    asked.extend(paste_questions(index.passages))
    emit(
        f"docs/: {index.documents} documents, {len(index.places)} chunks, indexed in "
        f"{time.perf_counter() - started:.0f}s with {args.model}; {len(asked)} queries"
    )
    measured = await _report_docs(index, asked, args.variants, overrides, emit)

    questions: dict[str, list[str]] = defaultdict(list)
    for item in asked:
        if item.group != "titles":
            questions[item.group].append(item.text)
    emit(f"\n### the statement at {args.scale} chunks, per question set\n")
    emit("| variant | set | queries | median ms | p95 ms | mean rows matched |")
    emit("|---|---|---:|---:|---:|---:|")
    timings = time_at_scale(index.texts, questions, size=args.scale, variants=args.variants)
    for (name, group), (median, p95, matched) in timings.items():
        emit(
            f"| {name} | {group} | {len(questions[group])} | {median:.1f} | {p95:.1f} | "
            f"{matched:.0f} |"
        )

    if {CHALLENGER, CAPPED} <= set(args.variants):
        emit(f"\n### round 3b: {CAPPED} against {CHALLENGER}\n")
        emit(round_three_b(measured, timings).describe())

    if not args.no_threads:
        await _report_threads(
            args, work, overrides=overrides, emit=emit, measured=measured, timings=timings
        )


async def _report_threads(
    args: argparse.Namespace,
    work: Path,
    *,
    overrides: Mapping[str, Any],
    emit: Callable[[str], None],
    measured: Mapping[tuple[str, str], Sequence[Outcome]],
    timings: Mapping[tuple[str, str], tuple[float, float, float]],
) -> None:
    """The mail half, and round 3's verdict, which needs both halves."""
    corpus = build_corpus(seed=args.seed, copies=args.copies, distractors=args.distractors)
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
    if {INCUMBENT, CHALLENGER} <= set(args.variants):
        emit(f"\n### round 3: {CHALLENGER} against {INCUMBENT}\n")
        emit(round_three(measured, runs, timings).describe())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--model", default="BAAI/bge-m3", help="embedding model")
    parser.add_argument("--provider", default="onnx", help="embedding backend: onnx, mlx or ollama")
    parser.add_argument(
        "--prefix-scheme", default=None, help="the model's prefix scheme, e.g. qwen3"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="the ollama backend's server; the default is this machine",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="a served backend's tokenizer: a directory holding tokenizer.json, or a repo id",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--no-threads", action="store_true", help="skip the synthetic mail corpus")
    parser.add_argument(
        "--fusion", action="store_true", help="run round 4, the fusion candidates, instead"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=HELD_OUT_SEED,
        help="the mail corpus's seed; rounds 1 and 2 used 463",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=HELD_OUT_COPIES,
        help="mail threads per length and style; rounds 1 and 2 used 2",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=HELD_OUT_DISTRACTORS,
        help="unrelated mail messages; rounds 1 and 2 used 20",
    )
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
        asyncio.run((report_fusion if args.fusion else report)(args, work, emit))


if __name__ == "__main__":
    main()
