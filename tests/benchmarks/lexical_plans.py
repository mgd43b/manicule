"""Which statement answers a lexical search fastest, on a corpus built to make them differ.

``python -m tests.benchmarks.lexical_plans --chunks 100000``

Four statements are timed on the same questions and the same restrictions:

- ``today`` is ``SEARCH_SQL`` as released, with a list of document ids or sources written as
  plain column tests. SQLite then runs one seek per listed id, or one scan of the source, for
  every chunk the match finds.
- ``direct`` is that statement with those two tests written ``+d.id`` and ``+d.source``, which
  is how :func:`~manicule.storage.docstore.lexical_predicates` writes them now.
- ``ranked`` is ``SEARCH_SQL_RANKED``: ``ORDER BY chunks_fts.rank``, which FTS5 can hand back in
  order without the temporary B-tree ``ORDER BY bm25()`` needs.
- ``doc-first`` is ``SEARCH_SQL_DOCUMENT_FIRST``, which walks an explicit document list and asks
  the index about each of its chunks. It is only timed where a list is given.

:func:`~manicule.storage.fts.choose_lexical_plan` picks among the last three, and this measures
where each wins and what the pick costs where it is wrong. Every row also checks that the
statements returned the same ranking, ties compared as sets.

**The corpus is synthetic and has the real schema.** It is a migrated database, so the indexes
the planner chooses between are the shipped ones, and no ``ANALYZE`` has run, exactly as in
production. Text is drawn from a Zipf vocabulary (``w0`` the commonest) at 120 words a chunk, and
each chunk is stored twice, as ``text`` and as the breadcrumbed ``embed_text``, so a row is about
as wide as a real one. Three planted words (``zr0001``, ``zr001``, ``zr01``) sit in 0.01%, 0.1%
and 1% of chunks. Documents carry a source, a media type and a timestamp drawn so a filter can
pass about half, 5% or 0.1% of them, and the reader grants are laid out the same way. Five percent
of documents are soft-deleted and five percent are not indexed, which is the dilution the joined
statement exists to survive. A second workspace holds a tenth of the index and a third holds 500
chunks, which is a small workspace in a large installation.

**Three kinds of time are reported, and they are not interchangeable.** ``warm`` is a long-lived
connection running a statement again, the way a server's pool does, in a shuffled order each
round, with the first round untimed. ``cold`` is a new process with a new connection running it
once. Neither purges the operating system's page cache (that takes root, and this does not ask
for it), so ``cold`` is cold to SQLite and never to the disk: a corpus that does not fit in
memory would pay more for every statement that visits many rows, ``direct`` most of all, and this
cannot show that. ``store`` runs :meth:`~manicule.storage.docstore.SqliteDocStore.search_lexical`
itself, probe and hydration included, as released and as chosen.

Every number is one machine's, at one corpus size, with one SQLite, and the first line of the
output names the version because the plan, not the SQL, is what moves between them.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from manicule.core.readers import group_principal
from manicule.core.retrieval import Filter
from manicule.storage import docstore as docstore_module
from manicule.storage.docstore import SqliteDocStore, lexical_predicates
from manicule.storage.engine import PRAGMAS, create_engine, database_path, prepare_data_dir
from manicule.storage.fts import (
    SCOPE_PROBE_LIMIT,
    SCOPE_PROBE_SQL,
    SEARCH_SQL,
    SEARCH_SQL_DOCUMENT_FIRST,
    SEARCH_SQL_RANKED,
    LexicalPlan,
    choose_lexical_plan,
    escape_match_query,
    match_terms,
    needs_scope_probe,
    needs_workspace_share,
)
from manicule.storage.migrator import upgrade

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

WORKSPACE = "w"
SMALL_WORKSPACE = "ws_small"
OTHER_WORKSPACE = "ws_other"

VOCABULARY = 5000
WORDS_PER_CHUNK = 120
BULK_CHUNKS = 50
SMALL_WORKSPACE_CHUNKS = 500
SECTION = 5
BASE_DAY = datetime(2026, 6, 1, tzinfo=UTC)
SPAN_DAYS = 1000


@dataclass(frozen=True)
class Variant:
    """A statement and the way its predicates are written."""

    name: str
    plan: LexicalPlan
    spelling: str


TODAY = Variant("today", LexicalPlan.DIRECT, "as-was")
DIRECT = Variant("direct", LexicalPlan.DIRECT, "anchored")
RANKED = Variant("ranked", LexicalPlan.RANKED, "anchored")
FIRST = Variant("doc-first", LexicalPlan.DOCUMENT_FIRST, "driving")
VARIANTS = (TODAY, DIRECT, RANKED, FIRST)
"""``today`` is the statement as released: ``SEARCH_SQL`` with ``d.id`` and ``d.source`` written as
plain column tests. ``direct`` is the same statement with those two written so SQLite cannot seek
on them (``+d.id``), which is how the store now runs it; ``ranked`` and ``doc-first`` are the other
two statements."""
BY_PLAN = {variant.plan: variant for variant in VARIANTS[1:]}

TEMPLATES = {
    LexicalPlan.DIRECT: SEARCH_SQL,
    LexicalPlan.RANKED: SEARCH_SQL_RANKED,
    LexicalPlan.DOCUMENT_FIRST: SEARCH_SQL_DOCUMENT_FIRST,
}

SOURCES = (("s50", 0.50), ("s30", 0.30), ("s15", 0.15), ("s5", 0.049), ("s01", 0.001))
MEDIA = (
    ("text/markdown", 0.55),
    ("text/html", 0.40),
    ("application/pdf", 0.049),
    ("text/x-rare", 0.001),
)
KINDS = (("prose", 0.50), ("list", 0.45), ("code", 0.049), ("table", 0.001))
GRANTS = {"s50": "gA", "s30": "gB", "s15": "gC", "s5": "gD"}
"""Which reader group each gated source is granted to. ``s01`` carries no gate."""

READERS: dict[str, frozenset[str]] = {
    "readers 95%": frozenset({"gA", "gB", "gC"}),
    "readers 50%": frozenset({"gA"}),
    "readers 15%": frozenset({"gC"}),
    "readers 5%": frozenset({"gD"}),
    "readers 0.1%": frozenset(),
}
"""Reader sets and about what fraction of documents each can read, given :data:`GRANTS`."""

ANCHOR = '{"kind":"unlocated","reason":"synthetic"}'
"""What every chunk says about where it came from, so the store can hydrate one."""

PLANTED = (("zr0001", 0.0001), ("zr001", 0.001), ("zr01", 0.01))

FAMILIES = (("fam5", 16, 5), ("fam8", 200, 8), ("fam9", 64, 9))
"""Documents of a known chunk count, named ``fam<chunks>-<n>``, so an explicit scope of an exact
size can be asked for on either side of the probe's budget."""


# --- the corpus --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """A built database and the sizes a run needs to describe it."""

    directory: Path
    chunks: int
    bulk_documents: int

    @property
    def path(self) -> Path:
        return database_path(self.directory)


def _pick(rng: np.random.Generator, choices: Sequence[tuple[str, float]], count: int) -> list[str]:
    names = [name for name, _ in choices]
    weights = np.array([weight for _, weight in choices])
    picked = rng.choice(len(names), size=count, p=weights / weights.sum())
    # A source or type held by 0.1% of documents must exist at every size, or a shape that
    # asks for it measures an empty set and calls it selective.
    for index in range(len(names)):
        if count > index:
            picked[index] = index
    return [names[i] for i in picked]


def _timestamp(moment: datetime) -> str:
    """``UtcDateTime``'s encoding: naive UTC, ``YYYY-MM-DD HH:MM:SS.ffffff``."""
    return moment.astimezone(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")


async def _template(directory: Path) -> None:
    built = create_engine(directory)
    try:
        await upgrade(built)
    finally:
        await built.dispose()


def _plan_documents(
    rng: np.random.Generator, chunks: int
) -> tuple[list[tuple[str, str, int]], int]:
    """Every document as ``(id, workspace, chunk count)``, shuffled, and how many are bulk."""
    family_chunks = sum(count * size for _, count, size in FAMILIES)
    other = chunks // 10
    bulk = max(chunks - family_chunks - SMALL_WORKSPACE_CHUNKS - other, BULK_CHUNKS * 130)
    bulk_documents = bulk // BULK_CHUNKS
    other_documents = max(other // BULK_CHUNKS, 1)
    documents = [(f"bulk-{i}", WORKSPACE, BULK_CHUNKS) for i in range(bulk_documents)]
    documents += [(f"ot-{i}", OTHER_WORKSPACE, BULK_CHUNKS) for i in range(other_documents)]
    documents += [
        (f"{name}-{i}", WORKSPACE, size) for name, count, size in FAMILIES for i in range(count)
    ]
    documents += [(f"sm-{i}", SMALL_WORKSPACE, SMALL_WORKSPACE_CHUNKS // 20) for i in range(20)]
    return [documents[i] for i in rng.permutation(len(documents))], bulk_documents


def _insert_documents(
    db: sqlite3.Connection, rng: np.random.Generator, documents: list[tuple[str, str, int]]
) -> None:
    """The workspaces, the documents and the reader grants, with the attributes a filter reads."""
    count = len(documents)
    sources = _pick(rng, SOURCES, count)
    media = _pick(rng, MEDIA, count)
    age = rng.integers(0, SPAN_DAYS, size=count)
    fate = rng.random(count)
    stamps = [
        _timestamp(BASE_DAY - timedelta(days=int(days), seconds=i % 3600))
        for i, days in enumerate(age)
    ]
    now = _timestamp(BASE_DAY)
    for name in (WORKSPACE, SMALL_WORKSPACE, OTHER_WORKSPACE):
        db.execute(
            "INSERT INTO workspaces (id, name, mode, settings, created_at) "
            "VALUES (?, ?, 'team', '{}', ?)",
            (name, name, now),
        )
    rows: list[tuple[Any, ...]] = []
    for i, (doc, workspace, _) in enumerate(documents):
        # The scope families and the small workspace are always live and indexed, so a shape that
        # names them measures the size it says it does.
        protected = workspace == SMALL_WORKSPACE or not doc.startswith(("bulk-", "ot-"))
        deleted = None if protected or fate[i] >= 0.05 else stamps[i]
        status = "pending" if not protected and 0.05 <= fate[i] < 0.10 else "indexed"
        uri = f"synthetic://{doc}"
        rows.append(
            (
                *(doc, workspace, sources[i], doc, uri, doc, media[i], f"hash-{doc}", status),
                *(stamps[i], stamps[i], deleted),
            )
        )
    db.executemany(
        "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, media_type, "
        "content_hash, status, metadata, created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)",
        rows,
    )
    db.executemany(
        "INSERT INTO source_reader_grants (workspace_id, source, group_id, created_at) "
        "VALUES (?, ?, ?, ?)",
        ((WORKSPACE, source, group, now) for source, group in GRANTS.items()),
    )


def _insert_chunks(
    db: sqlite3.Connection, rng: np.random.Generator, documents: list[tuple[str, str, int]]
) -> int:
    """Every document's chunks, drawn from the Zipf vocabulary. Returns how many were written."""
    weights = 1.0 / np.arange(1, VOCABULARY + 1)
    weights /= weights.sum()
    vocabulary = [f"w{i}" for i in range(VOCABULARY)]
    total = sum(size for _, _, size in documents)
    plant: dict[int, list[str]] = {}
    for word, fraction in PLANTED:
        for seq in rng.choice(total, size=max(int(total * fraction), 2), replace=False):
            plant.setdefault(int(seq), []).append(word)
    kinds = _pick(rng, KINDS, total)
    now = _timestamp(BASE_DAY)

    seq = 0
    batch: list[tuple[Any, ...]] = []
    for doc, _, size in documents:
        draws: list[list[int]] = rng.choice(
            VOCABULARY, size=(size, WORDS_PER_CHUNK), p=weights
        ).tolist()
        headings: list[str] = []
        for _ in range(-(-size // SECTION)):
            picks: list[int] = rng.integers(50, 3000, size=3).tolist()
            headings.append(" ".join(vocabulary[j] for j in picks))
        for position in range(size):
            seq += 1
            text = " ".join([vocabulary[j] for j in draws[position]] + plant.get(seq - 1, []))
            heading = headings[position // SECTION]
            # Stored twice, as the product stores it (``text`` and the breadcrumbed
            # ``embed_text``), so a row is as wide as a real one and a join that reads it
            # touches as many pages.
            batch.append(
                (
                    *(seq, f"c{seq}", doc, text, f"{heading}\n\n{text}", heading),
                    *(kinds[seq - 1], position, ANCHOR, now),
                )
            )
        if len(batch) >= 20_000:
            _flush(db, batch)
            batch = []
    _flush(db, batch)
    return seq


OWNER_MARKER = ".lexical-plans-corpus"
"""Written into a directory this program built a corpus in, and the only thing that lets it
delete one: a ``--work`` that names somebody's directory must never be emptied."""


class DirectoryNotOwnedError(Exception):
    """``--work`` names a non-empty directory this program may not delete."""


def _claim(directory: Path, *, overwrite: bool) -> None:
    """Make ``directory`` empty and ours, or refuse and touch nothing."""
    if directory.exists() and any(directory.iterdir()):
        owned = (directory / OWNER_MARKER).is_file()
        if not owned:
            msg = f"{directory} is not empty and was not built by this benchmark; not deleting it"
            raise DirectoryNotOwnedError(msg)
        if not overwrite:
            msg = f"{directory} holds a corpus from an earlier run; pass --overwrite to rebuild it"
            raise DirectoryNotOwnedError(msg)
        shutil.rmtree(directory)
    prepare_data_dir(directory)
    (directory / OWNER_MARKER).write_text("built by tests.benchmarks.lexical_plans\n")


def build_corpus(
    directory: Path, chunks: int, *, seed: int = 20260101, overwrite: bool = False
) -> Corpus:
    """A migrated database holding about ``chunks`` synthetic chunks, loaded in one transaction.

    Text is generated, and nothing here reads a real document. The triggers that keep the
    lexical index current are left in place so ``chunks_fts`` is built the way the product
    builds it. A directory that is not empty is refused unless it carries :data:`OWNER_MARKER`
    and ``overwrite`` is set.
    """
    _claim(directory, overwrite=overwrite)
    asyncio.run(_template(directory))
    rng = np.random.default_rng(seed)
    documents, bulk_documents = _plan_documents(rng, chunks)
    db = sqlite3.connect(database_path(directory))
    db.execute("PRAGMA synchronous = OFF")
    db.execute("PRAGMA foreign_keys = OFF")
    _insert_documents(db, rng, documents)
    written = _insert_chunks(db, rng, documents)
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    return Corpus(directory=directory, chunks=written, bulk_documents=bulk_documents)


def _flush(db: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    db.executemany(
        "INSERT INTO chunks (seq, id, document_id, text, embed_text, heading_text, "
        "heading_path, kind, position, token_count, anchor, metadata, created_at, vector_id) "
        "VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, 0, ?, '{}', ?, '')",
        batch,
    )


# --- the shapes --------------------------------------------------------------------------------

QUERIES = {
    "common-3": "w19 w49 w99",
    "common-2": "w9 w19",
    "mid-3": "w199 w399 w799",
    "prose-6": " ".join(f"w{i}" for i in (19, 39, 79, 159, 319, 639)),
    "prose-9": " ".join(f"w{i}" for i in (9, 19, 39, 59, 99, 159, 259, 459, 799)),
    "common+rare": "w19 zr001",
    "rare-2": "zr0001 zr001",
    "no-match": "qqqa qqqb",
    "single-common": "w19",
    "single-rare": "zr001",
    "paste-16": " ".join(
        f"w{i}"
        for i in (9, 19, 29, 49, 79, 99, 149, 199, 299, 399, 499, 799, 999, 1499, 1999, 2999)
    ),
}
"""Questions by the shape of their match. ``w<n>`` is the ``n+1``-th commonest word, held by
about ``1 - (1 - 1/((n+1) * 9.1)) ** 120`` of chunks: 73% for ``w9``, 48% for ``w19``, 12% for
``w99``, 1.6% for ``w799``. ``common-3`` therefore matches about two chunks in three."""


@dataclass(frozen=True)
class Shape:
    """One search: a question, a restriction and the workspace it runs in."""

    name: str
    text: str
    filter: Filter | None = None
    workspace: str = WORKSPACE
    limit: int = 50

    def documents(self) -> tuple[str, ...]:
        return tuple(sorted(self.filter.document_ids)) if self.filter else ()


def _after(fraction: float) -> datetime:
    return BASE_DAY - timedelta(days=SPAN_DAYS * fraction)


def _scope(name: str, count: int) -> Filter:
    return Filter(
        workspace_ids=frozenset({WORKSPACE}),
        document_ids=frozenset(f"{name}-{i}" for i in range(count)),
    )


def scopes(bulk_documents: int) -> dict[str, tuple[str, Filter | None]]:
    """Every restriction measured, by name, with the workspace each runs in."""
    every = frozenset({WORKSPACE})
    found: dict[str, tuple[str, Filter | None]] = {
        "no filter": (WORKSPACE, None),
        "source 50%": (WORKSPACE, Filter(workspace_ids=every, sources=frozenset({"s50"}))),
        "source 5%": (WORKSPACE, Filter(workspace_ids=every, sources=frozenset({"s5"}))),
        "source 0.1%": (WORKSPACE, Filter(workspace_ids=every, sources=frozenset({"s01"}))),
        "media 55%": (
            WORKSPACE,
            Filter(workspace_ids=every, media_types=frozenset({"text/markdown"})),
        ),
        "media 5%": (
            WORKSPACE,
            Filter(workspace_ids=every, media_types=frozenset({"application/pdf"})),
        ),
        "media 0.1%": (
            WORKSPACE,
            Filter(workspace_ids=every, media_types=frozenset({"text/x-rare"})),
        ),
        "time 50%": (WORKSPACE, Filter(workspace_ids=every, updated_after=_after(0.5))),
        "time 5%": (WORKSPACE, Filter(workspace_ids=every, updated_after=_after(0.05))),
        "time 0.1%": (WORKSPACE, Filter(workspace_ids=every, updated_after=_after(0.001))),
        "kind 5%": (WORKSPACE, Filter(workspace_ids=every, kinds=frozenset({_kind("code")}))),
        "kind 0.1%": (WORKSPACE, Filter(workspace_ids=every, kinds=frozenset({_kind("table")}))),
        "small workspace": (SMALL_WORKSPACE, Filter(workspace_ids=frozenset({SMALL_WORKSPACE}))),
    }
    for label, groups in READERS.items():
        held = frozenset(group_principal(group) for group in groups)
        found[label] = (WORKSPACE, Filter(workspace_ids=every, readers=held))
    found["ids 2x8=16"] = (WORKSPACE, _scope("fam8", 2))
    found["ids 4x5=20"] = (WORKSPACE, _scope("fam5", 4))
    found["ids 4x8=32"] = (WORKSPACE, _scope("fam8", 4))
    found["ids 8x5=40"] = (WORKSPACE, _scope("fam5", 8))
    found["ids 8x8=64"] = (WORKSPACE, _scope("fam8", 8))
    found["ids 12x8=96"] = (WORKSPACE, _scope("fam8", 12))
    found["ids 16x8=128"] = (WORKSPACE, _scope("fam8", 16))
    found["ids 64x8=512"] = (WORKSPACE, _scope("fam8", 64))
    found["ids 64x9=576"] = (WORKSPACE, _scope("fam9", 64))
    found["ids 65x8=520"] = (WORKSPACE, _scope("fam8", 65))
    found["ids 200x8"] = (WORKSPACE, _scope("fam8", 200))
    found["ids 4 bulk=200"] = (WORKSPACE, _scope("bulk", 4))
    found["ids 10 bulk=500"] = (WORKSPACE, _scope("bulk", 10))
    found["ids 11 bulk=550"] = (WORKSPACE, _scope("bulk", 11))
    found["ids 64 bulk=3200"] = (WORKSPACE, _scope("bulk", min(64, bulk_documents)))
    # What a collection resolves to is bounded by MAX_RESOLVED_DOCUMENTS (10,000), not by the
    # probe's budget, so the statements that take over past it are measured at that scale too.
    for count in (2000, 10_000):
        if bulk_documents >= count:
            found[f"ids {count} bulk"] = (WORKSPACE, _scope("bulk", count))
    return found


def _kind(name: str) -> Any:
    from manicule.core.content import BlockKind  # noqa: PLC0415 - only these two shapes need it

    return BlockKind(name)


def shapes(bulk_documents: int, queries: Sequence[str] | None = None) -> list[Shape]:
    """Every question against every restriction it is worth asking of, then the odd ones."""
    wanted = list(queries or QUERIES)
    made: list[Shape] = []
    for query, (scope, (workspace, restriction)) in itertools.product(
        wanted, scopes(bulk_documents).items()
    ):
        made.append(Shape(f"{query} @ {scope}", QUERIES[query], restriction, workspace))
    # A memory slug's words, unfiltered, 200 results: the wikilinks plugin's backlink search.
    made.append(Shape("wikilinks @ no filter", "w9 zr001", None, WORKSPACE, 200))
    return made


# --- running one statement ---------------------------------------------------------------------


def connect(path: Path) -> sqlite3.Connection:
    """A reader's connection, configured as :data:`~manicule.storage.engine.PRAGMAS` configures
    the product's, and with no statistics: the corpus has never been analyzed."""
    db = sqlite3.connect(path)
    for name, value in PRAGMAS:
        db.execute(f"PRAGMA {name} = {value}")
    return db


def statement_for(variant: Variant, shape: Shape) -> tuple[str, dict[str, Any]]:
    """The SQL and bindings ``variant`` runs for ``shape``, built by the store's own builder."""
    predicates = lexical_predicates(shape.workspace, shape.filter)
    text = predicates.text(TEMPLATES[variant.plan], document_first=variant.spelling == "driving")
    if variant.spelling == "as-was":
        text = text.replace("AND +d.", "AND d.")
    params: dict[str, Any] = {
        "match": escape_match_query(shape.text),
        "limit": shape.limit,
        **predicates.params,
    }
    for name in predicates.timestamps:
        params[name] = _timestamp(params[name])
    return text, params


def scoped_chunks(db: sqlite3.Connection, shape: Shape) -> int | None:
    """What the store's probe returns for ``shape``, or ``None`` when it would not run one."""
    ids = shape.documents()
    if not needs_scope_probe(len(match_terms(shape.text)), len(ids)):
        return None
    names = [f"scope_document_{i}" for i in range(len(ids))]
    sql = SCOPE_PROBE_SQL.format(ids=", ".join(":" + name for name in names))
    params = {
        "workspace_id": shape.workspace,
        "probe_limit": SCOPE_PROBE_LIMIT,
        **dict(zip(names, ids, strict=True)),
    }
    return int(db.execute(sql, params).fetchone()[0])


def chosen(db: sqlite3.Connection, shape: Shape) -> Variant:
    """The statement the shipped chooser picks for ``shape``."""
    terms = len(match_terms(shape.text))
    narrowed = lexical_predicates(shape.workspace, shape.filter).narrowed
    share = None
    if needs_workspace_share(terms, len(shape.documents()), narrowed=narrowed):
        counts = dict(
            db.execute(
                "SELECT workspace_id, count(*) FROM documents WHERE deleted_at IS NULL "
                "GROUP BY workspace_id"
            ).fetchall()
        )
        share = counts.get(shape.workspace, 0) / sum(counts.values())
    plan = choose_lexical_plan(
        terms=terms,
        scoped_documents=len(shape.documents()),
        scoped_chunks=scoped_chunks(db, shape),
        narrowed=narrowed,
        workspace_share=share,
    )
    return BY_PLAN[plan]


class TooSlowError(Exception):
    """A statement outran the cap."""


@contextlib.contextmanager
def capped(db: sqlite3.Connection, seconds: float) -> Generator[None]:
    """Interrupt whatever ``db`` runs after ``seconds``, so one pathological plan cannot hang a
    sweep. A statement that hits the cap is reported as slower than the cap, not as the cap."""
    deadline = time.perf_counter() + seconds
    db.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 50_000)
    try:
        yield
    except sqlite3.OperationalError as error:
        if "interrupted" in str(error):
            raise TooSlowError from error
        raise
    finally:
        db.set_progress_handler(None, 0)


def run(
    db: sqlite3.Connection, variant: Variant, shape: Shape, cap: float
) -> tuple[float, list[tuple[str, float]]]:
    """Milliseconds and rows of ``variant`` for ``shape``; :class:`TooSlowError` past ``cap``."""
    sql, params = statement_for(variant, shape)
    started = time.perf_counter()
    with capped(db, cap):
        rows = db.execute(sql, params).fetchall()
    return (time.perf_counter() - started) * 1000, [(str(r[0]), float(r[1])) for r in rows]


def equivalent(
    first: list[tuple[str, float]],
    second: list[tuple[str, float]],
    *,
    limit: int,
    full: list[tuple[str, float]] | None = None,
) -> bool:
    """Whether two results of a search with ``LIMIT limit`` are the same ranking.

    SQLite does not specify the order of equal scores, nor which of them a ``LIMIT`` keeps, so
    ordered equality would be a stricter claim than the statements make. Scores must match
    position by position, and the chunks of each score must match as a set, with one exception.
    When a result fills the limit its last score group may have been cut, and the two results
    may then hold different members of it: each must hold only members of that group in
    ``full``, the same search without the limit. Without ``full`` that one group is unchecked.
    A result shorter than ``limit`` was not cut, so every group of it is checked.
    """
    if [score for _, score in first] != [score for _, score in second]:
        return False
    if not first:
        return True
    cut = len(first) >= limit
    last = first[-1][1]
    for score in {score for _, score in first}:
        ours = {c for c, s in first if s == score}
        theirs = {c for c, s in second if s == score}
        if cut and score == last:
            if full is None:
                continue
            group = {c for c, s in full if s == score}
            if not (ours <= group and theirs <= group):
                return False
        elif ours != theirs:
            return False
    return True


def plan_of(db: sqlite3.Connection, variant: Variant, shape: Shape) -> list[str]:
    """``EXPLAIN QUERY PLAN`` of ``variant`` for ``shape``, one string per step."""
    sql, params = statement_for(variant, shape)
    return [str(row[3]) for row in db.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]


def sorts(steps: Sequence[str]) -> bool:
    """Whether the plan sorts the matches itself, which is the cost ``rank`` ordering removes."""
    return any("USE TEMP B-TREE FOR ORDER BY" in step for step in steps)


# --- the measurement ---------------------------------------------------------------------------


@dataclass
class Measured:
    shape: Shape
    times: dict[str, float | None]
    """Median milliseconds by variant name, ``None`` for one that outran the cap."""
    chosen: Variant
    agree: bool
    matched: int


def warm(
    db: sqlite3.Connection,
    shape: Shape,
    variants: Sequence[Variant],
    *,
    reps: int,
    cap: float,
    seed: int = 0,
) -> Measured:
    """Median time of each variant for ``shape`` on one connection, run in a shuffled order each
    round so no variant always follows the same one's cache. The first round is not timed."""
    rng = random.Random(seed)  # noqa: S311 - a shuffle, not a secret
    times: dict[str, list[float]] = {variant.name: [] for variant in variants}
    results: dict[str, list[tuple[str, float]]] = {}
    slow: set[str] = set()
    for round_ in range(reps + 1):
        order = list(variants)
        rng.shuffle(order)
        for variant in order:
            if variant.name in slow:
                continue
            try:
                elapsed, rows = run(db, variant, shape, cap)
            except TooSlowError:
                slow.add(variant.name)
                continue
            if round_ == 0:
                results[variant.name] = rows
            else:
                times[variant.name].append(elapsed)
    reference = next(iter(results.values()), None)
    return Measured(
        shape=shape,
        times={
            name: (None if name in slow or not t else statistics.median(t))
            for name, t in times.items()
        },
        chosen=chosen(db, shape),
        agree=all(equivalent(reference, rows, limit=shape.limit) for rows in results.values())
        if reference is not None
        else True,
        matched=len(reference or []),
    )


def cold(corpus: Corpus, shape: Shape, variant: Variant, *, reps: int, cap: float) -> float | None:
    """Median ms of ``variant`` on a fresh process and connection, or ``None`` past the cap."""
    times: list[float] = []
    for _ in range(reps):
        spec = json.dumps(
            {"path": str(corpus.path), "shape": encode_shape(shape), "variant": variant.name}
        )
        done = subprocess.run(  # noqa: S603 - this interpreter and arguments built above
            [
                sys.executable,
                "-m",
                "tests.benchmarks.lexical_plans",
                "--child",
                spec,
                "--cap",
                str(cap),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        text = done.stdout.strip().splitlines()[-1]
        if text == "slow":
            return None
        times.append(float(text))
    return statistics.median(times)


def encode_shape(shape: Shape) -> dict[str, Any]:
    return {
        "name": shape.name,
        "text": shape.text,
        "workspace": shape.workspace,
        "limit": shape.limit,
        "filter": shape.filter.model_dump(mode="json") if shape.filter else None,
    }


def _decode(raw: dict[str, Any]) -> Shape:
    restriction = Filter.model_validate(raw["filter"]) if raw["filter"] else None
    return Shape(raw["name"], raw["text"], restriction, raw["workspace"], raw["limit"])


def child(spec: str, cap: float) -> None:
    """One cold run: a new connection, the statement once, and the milliseconds on stdout."""
    raw = json.loads(spec)
    started = time.perf_counter()
    db = connect(Path(raw["path"]))
    try:
        shape = _decode(raw["shape"])
        variant = next(v for v in VARIANTS if v.name == raw["variant"])
        run(db, variant, shape, cap)
    except TooSlowError:
        sys.stdout.write("slow\n")
        return
    finally:
        db.close()
    sys.stdout.write(f"{(time.perf_counter() - started) * 1000:.3f}\n")


# --- through the store -------------------------------------------------------------------------


@contextlib.contextmanager
def released() -> Generator[None]:
    """Make the store run as it did before the chooser: ``SEARCH_SQL``, its id and source tests
    written as plain columns, and no probe."""
    chooser = docstore_module.choose_lexical_plan
    probe = docstore_module.needs_scope_probe
    plain = docstore_module._FILTERS_NOT_TO_SEEK_ON  # pyright: ignore[reportPrivateUsage]

    def always_direct(**_facts: Any) -> LexicalPlan:
        return LexicalPlan.DIRECT

    def never_probe(_terms: int, _documents: int) -> bool:
        return False

    docstore_module.choose_lexical_plan = always_direct
    docstore_module.needs_scope_probe = never_probe
    docstore_module._FILTERS_NOT_TO_SEEK_ON = frozenset()  # pyright: ignore[reportPrivateUsage]
    try:
        yield
    finally:
        docstore_module.choose_lexical_plan = chooser
        docstore_module.needs_scope_probe = probe
        docstore_module._FILTERS_NOT_TO_SEEK_ON = plain  # pyright: ignore[reportPrivateUsage]


async def through_store(
    corpus: Corpus, shapes_: Sequence[Shape], *, reps: int
) -> dict[str, tuple[float, float]]:
    """Median milliseconds of ``search_lexical`` per shape: as released, then as chosen now.

    Includes the probe, the hydration and SQLAlchemy, which is what a caller pays.
    """
    engine = create_engine(corpus.directory)
    out: dict[str, tuple[float, float]] = {}
    try:
        stores = {
            name: SqliteDocStore(engine, workspace_id=name)
            for name in {s.workspace for s in shapes_}
        }
        for shape in shapes_:
            store = stores[shape.workspace]
            timings: dict[str, list[float]] = {"released": [], "chosen": []}
            for round_ in range(reps + 1):
                for label in ("released", "chosen"):
                    with released() if label == "released" else contextlib.nullcontext():
                        started = time.perf_counter()
                        await store.search_lexical(shape.text, shape.limit, shape.filter)
                        elapsed = (time.perf_counter() - started) * 1000
                    if round_:
                        timings[label].append(elapsed)
            out[shape.name] = (
                statistics.median(timings["released"]),
                statistics.median(timings["chosen"]),
            )
    finally:
        await engine.dispose()
    return out


# --- the report --------------------------------------------------------------------------------


def worth_forcing(shape: Shape) -> bool:
    """Whether to time the document-first statement on ``shape``.

    Only for an explicit scope, which is what it is for. On an unscoped search it walks every
    document and costs seconds, and a table that waits on that for every shape measures
    patience.
    """
    return 0 < len(shape.documents()) <= 200


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def render(rows: Sequence[Measured], *, cold_times: dict[str, dict[str, float | None]]) -> str:
    """A table of milliseconds per variant, the statement the chooser picks, and agreement.

    ``>cap`` is a variant that outran the cap; ``-`` is one that was not run.
    """
    names = [variant.name for variant in VARIANTS]
    lines = [
        "| shape | rows | " + " | ".join(names) + " | chooser picks | agree |",
        "|---|---:|" + "---:|" * len(names) + "---|---|",
    ]
    for row in rows:
        cells = [
            ">cap" if name in row.times and row.times[name] is None else _cell(row.times.get(name))
            for name in names
        ]
        lines.append(
            f"| {row.shape.name} | {row.matched} | {' | '.join(cells)} "
            f"| {row.chosen.name} | {'yes' if row.agree else 'NO'} |"
        )
    if cold_times:
        lines += ["", "| cold shape | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for name, times in cold_times.items():
            cells = [
                ">cap" if key in times and times[key] is None else _cell(times.get(key))
                for key in names
            ]
            lines.append(f"| {name} | {' | '.join(cells)} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--chunks", type=int, default=100_000)
    parser.add_argument("--work", type=Path, default=None, help="keep the corpus here")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild a corpus this benchmark built earlier in --work (it deletes that directory)",
    )
    parser.add_argument("--reuse", action="store_true", help="measure an existing corpus in --work")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--cap", type=float, default=20.0, help="seconds before a plan is abandoned"
    )
    parser.add_argument(
        "--document-first-cap",
        type=float,
        default=5.0,
        help="seconds before document-first is abandoned on a shape with no explicit scope",
    )
    parser.add_argument(
        "--skip-today",
        action="store_true",
        help="do not time the statement as released, which can take longer than the cap",
    )
    parser.add_argument("--only", default="", help="regular expression over shape names")
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--plans", action="store_true", help="print EXPLAIN QUERY PLAN per shape")
    parser.add_argument("--build-only", action="store_true", help="build the corpus and stop")
    parser.add_argument("--cold", action="store_true", help="also time fresh processes")
    parser.add_argument("--store", action="store_true", help="also time SqliteDocStore")
    parser.add_argument(
        "--store-only", action="store_true", help="time SqliteDocStore and nothing else"
    )
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child is not None:
        child(args.child, args.cap)
        return

    if args.reuse and args.work is None:
        parser.error("--reuse needs the --work it was built in")
    with contextlib.ExitStack() as stack:
        work = args.work or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="plans-")))
        report(args, work)


def report(args: argparse.Namespace, work: Path) -> None:  # noqa: PLR0912 - one table per flag
    """Build or open the corpus in ``work`` and print every table the arguments ask for."""
    started = time.perf_counter()
    if args.reuse:
        loaded = sqlite3.connect(database_path(work))
        bulk = loaded.execute("SELECT count(*) FROM documents WHERE id LIKE 'bulk-%'").fetchone()[0]
        total = loaded.execute("SELECT count(*) FROM chunks").fetchone()[0]
        loaded.close()
        corpus = Corpus(work, total, bulk)
    else:
        corpus = build_corpus(work, args.chunks, overwrite=args.overwrite)
    sys.stdout.write(
        f"sqlite {sqlite3.sqlite_version}; {corpus.chunks} chunks; "
        f"built in {time.perf_counter() - started:.0f}s\n"
    )
    sys.stdout.flush()
    if args.build_only:
        return
    pattern = re.compile(args.only)
    chosen_shapes = [
        s for s in shapes(corpus.bulk_documents, args.queries) if pattern.search(s.name)
    ]
    if args.store_only:
        timings = asyncio.run(through_store(corpus, chosen_shapes, reps=args.reps))
        sys.stdout.write("| store shape | released | chosen |\n|---|---:|---:|\n")
        for name, (before, after) in timings.items():
            sys.stdout.write(f"| {name} | {before:.1f} | {after:.1f} |\n")
        return
    db = connect(corpus.path)
    if chosen_shapes:
        # The first statement on a corpus reads pages nothing has touched since it was built, and
        # would charge that to whichever variant happened to run first.
        warm(db, chosen_shapes[0], VARIANTS[1:3], reps=0, cap=args.cap)
    rows: list[Measured] = []
    for shape in chosen_shapes:
        timed = VARIANTS if worth_forcing(shape) else VARIANTS[:3]
        rows.append(
            warm(
                db,
                shape,
                tuple(v for v in timed if not (args.skip_today and v is TODAY)),
                reps=args.reps,
                cap=args.cap if shape.documents() else min(args.cap, args.document_first_cap),
            )
        )
        sys.stdout.write(render(rows[-1:], cold_times={}).splitlines()[-1] + "\n")
        sys.stdout.flush()
    cold_times: dict[str, dict[str, float | None]] = {}
    if args.cold:
        for shape in chosen_shapes:
            cold_times[shape.name] = {
                v.name: cold(corpus, shape, v, reps=3, cap=args.cap)
                for v in (VARIANTS if worth_forcing(shape) else VARIANTS[:3])
                if not (args.skip_today and v is TODAY)
            }
    sys.stdout.write("\n" + render(rows, cold_times=cold_times) + "\n")
    if args.plans:
        for shape in chosen_shapes:
            for variant in VARIANTS:
                steps = plan_of(db, variant, shape)
                flag = " [SORTS]" if sorts(steps) else ""
                sys.stdout.write(f"{shape.name} / {variant.name}{flag}: {' | '.join(steps)}\n")
    if args.store:
        timings = asyncio.run(through_store(corpus, chosen_shapes, reps=args.reps))
        sys.stdout.write("\n| store shape | released | chosen |\n|---|---:|---:|\n")
        for name, (before, after) in timings.items():
            sys.stdout.write(f"| {name} | {before:.1f} | {after:.1f} |\n")
    db.close()


if __name__ == "__main__":
    main()
