"""The lexical index: an FTS5 table over ``chunks``, kept in step by triggers.

Two decisions carry the whole design.

**External content.** The virtual table stores only the inverted index and reads the text
back from ``chunks``. One copy of the corpus, in the authoritative store, and a whole class
of "the two copies disagree" bug that cannot occur.

**Triggers, not application code.** A trigger runs inside the same transaction as the row
change and cannot be bypassed by a migration, a repair path, or a write someone forgot about.
Application-level synchronization covers only the write paths its author remembered.
"""

from __future__ import annotations

import re
from enum import StrEnum

from manicule.storage.models import FTS_TOKENIZER

FTS_TABLE = "chunks_fts"

FTS_SHADOW_TABLES = (
    "chunks_fts",
    "chunks_fts_data",
    "chunks_fts_idx",
    "chunks_fts_content",
    "chunks_fts_docsize",
    "chunks_fts_config",
)
"""What FTS5 creates behind the virtual table.

Alembic's autogenerate does not model virtual tables, so it sees every one of these as a
table present in the database and absent from the models — and helpfully emits
``drop_table``. ``env.py`` filters on this tuple.
"""

CREATE_FTS = f"""
CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(
    text,
    heading_text,
    content='chunks',
    content_rowid='seq',
    tokenize='{FTS_TOKENIZER}'
)
"""

_SAFE_TOKENIZER = re.compile(r"[A-Za-z0-9_ ]+\Z")


def create_fts(tokenizer: str) -> str:
    """Render the virtual-table DDL for one validated FTS5 tokenizer declaration.

    The declaration is persisted rebuild configuration, not a SQL fragment. FTS5 accepts its
    tokenizer as a string mini-language; limiting it to names, integers, underscores and
    spaces covers the built-ins while making quotes, punctuation and statement separators
    unrepresentable.
    """
    if not tokenizer or _SAFE_TOKENIZER.fullmatch(tokenizer) is None:
        raise ValueError("FTS tokenizer contains unsupported characters")
    return f"""
CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(
    text,
    heading_text,
    content='chunks',
    content_rowid='seq',
    tokenize='{tokenizer}'
)
"""


CREATE_TRIGGERS = (
    """
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text, heading_text)
        VALUES (new.seq, new.text, new.heading_text);
    END
    """,
    """
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
        VALUES ('delete', old.seq, old.text, old.heading_text);
        INSERT OR IGNORE INTO vector_tombstones(chunk_id, deleted_at)
        VALUES (old.vector_id, datetime('now'));
    END
    """,
    """
    CREATE TRIGGER chunks_au AFTER UPDATE OF text, heading_text ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
        VALUES ('delete', old.seq, old.text, old.heading_text);
        INSERT INTO chunks_fts(rowid, text, heading_text)
        VALUES (new.seq, new.text, new.heading_text);
    END
    """,
)
"""Insert, delete and update, all inside the caller's transaction.

The delete trigger also records a vector tombstone. It is the one place a SQL-side timestamp
is used, because nothing orders ``vector_tombstones`` against another table's timestamps —
the sweep reads the whole table.

Hard-deleting a document reaches ``chunks`` through ``ON DELETE CASCADE``, and these fire on
cascaded deletes: verified directly on SQLite 3.51 at two cascade levels, with and without
``PRAGMA recursive_triggers``, which does not govern this case. ``tests/test_storage_fts.py``
holds it.
"""

_CHUNKS_AD_CURRENT = """
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
        VALUES ('delete', old.seq, old.text, old.heading_text);
        INSERT OR IGNORE INTO vector_tombstones(
            chunk_id, workspace_id, vector_namespace, vector_table, deleted_at
        )
        SELECT old.vector_id, d.workspace_id, s.vector_namespace, s.vector_table,
               datetime('now')
          FROM documents AS d
          LEFT JOIN index_state AS s ON s.workspace_id = d.workspace_id
         WHERE d.id = old.document_id;
    END
    """
"""The delete trigger as the **head** revision installs it, byte for byte."""

CURRENT_TRIGGERS = (CREATE_TRIGGERS[0], _CHUNKS_AD_CURRENT, CREATE_TRIGGERS[2])
"""What a live database's triggers must be, as opposed to what a migration once created.

**Two tuples because a migration is frozen and a rebuild is not.** ``CREATE_TRIGGERS`` is
imported by ``20260814_6e31b7d592ac_atomic_publications``, so its bodies describe the schema at
*that* revision and may never change — editing them would retroactively change what an already
applied migration did.

``20260818_f3c18a9d72e1_workspace_index_identity`` then replaced ``chunks_ad`` with the body
above, which resolves the deleted chunk's workspace and vector table and writes them into the
tombstone. Anything recreating triggers from the frozen tuple therefore *downgrades* a
migrated database: the lexical rebuild in ``SqliteRebuildStore`` drops all three and recreates
them, so publishing a rebuilt generation left the pre-migration trigger in place, and every
chunk deleted afterwards wrote a tombstone with no workspace, namespace or table in it — a row
the sweep cannot attribute to the index it belongs to.

Silent in both directions: the rebuild succeeds, the integrity check passes because it compares
``chunks_fts`` against ``chunks`` and has nothing to say about a trigger body, and the
tombstones only stop being attributable from that point on.
"""

DROP_TRIGGERS = (
    "DROP TRIGGER IF EXISTS chunks_au",
    "DROP TRIGGER IF EXISTS chunks_ad",
    "DROP TRIGGER IF EXISTS chunks_ai",
)

REBUILD_FTS = f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES('rebuild')"
"""Rung 1 of the blast-radius ladder: rebuild the lexical index from ``chunks``, free."""

INTEGRITY_CHECK_FTS = f"INSERT INTO {FTS_TABLE}({FTS_TABLE}, rank) VALUES('integrity-check', 1)"
"""The ``rank`` argument is the whole check.

Two obvious alternatives do not work, and both were in an earlier draft of the design:

* ``COUNT(*)`` on each table can never disagree. ``chunks_fts`` is external-content, so
  counting it reads through to ``chunks``; the two are the same number by construction.
* A bare ``integrity-check`` passes on a completely empty index, because it verifies only
  that the index is internally consistent — and an empty index is consistent with itself.

Only the ``1`` argument compares the index against the content table. Checked directly: with
the triggers dropped and rows inserted, both counts agree, ``MATCH`` returns nothing, and a
bare ``integrity-check`` passes; this form raises. A silently empty lexical index halves
hybrid retrieval while every health check reports green.
"""

SEARCH_SQL = """
SELECT c.id AS chunk_id, bm25(chunks_fts, 1.0, 0.4) AS rank_score
FROM chunks_fts
JOIN chunks c ON c.seq = chunks_fts.rowid
JOIN documents d ON d.id = c.document_id
WHERE chunks_fts MATCH :match
  AND d.deleted_at IS NULL
  AND d.status = 'indexed'
  {extra}
ORDER BY bm25(chunks_fts, 1.0, 0.4)
LIMIT :limit
"""
"""One statement, so ``LIMIT`` applies *after* the joins and filters.

Running ``MATCH … LIMIT k`` first and filtering afterwards silently returns fewer than ``k``
live rows: deletion is deferred, so chunks of soft-deleted documents are still in the index
and still compete for those slots, as are chunks from other workspaces — which ``chunks_fts``
cannot distinguish, because the workspace lives on ``documents``. Measured on a fixture with
three matching live chunks against five soft-deleted and five cross-workspace, ``k = 5``
returned **zero** live results this way and all three the joined way.

Column weights ``1.0`` and ``0.4``: the heading path has to be searchable, because a section
called "Configuration" is unfindable without knowing what it configures, but it repeats on every
chunk of a section, so at full weight it floods term frequencies and depresses the IDF of the
terms that identify the page. It is the headings alone — not the breadcrumb ``embed_text``
carries, which adds the source's hierarchy and the document's title. Round 5 of
``docs/retrieval.md`` §5.4 measured indexing the whole breadcrumb, and weighting this column
1.0, and neither qualified under both embedders.

``bm25()`` returns a *negative* number and a better match is more negative, so the ordering is
ascending. Its first argument must be the table's real name — it is parsed as a column
reference, so a query that aliases the FTS table fails with ``no such column``.
"""

SEARCH_SQL_RANKED = """
SELECT c.id AS chunk_id, chunks_fts.rank AS rank_score
FROM chunks_fts
JOIN chunks c ON c.seq = chunks_fts.rowid
JOIN documents d ON d.id = c.document_id
WHERE chunks_fts MATCH :match
  AND chunks_fts.rank MATCH 'bm25(1.0, 0.4)'
  AND d.deleted_at IS NULL
  AND d.status = 'indexed'
  {extra}
ORDER BY chunks_fts.rank
LIMIT :limit
"""
"""``SEARCH_SQL`` ordered by FTS5's hidden ``rank`` column instead of by an expression.

Same joins, same predicates, same weights, same ``LIMIT`` after them. What differs is who sorts.
``ORDER BY bm25(...)`` is an expression over each joined row, so SQLite scores and sorts every
match: the plan shows ``USE TEMP B-TREE FOR ORDER BY``. ``ORDER BY chunks_fts.rank`` is a
column of the virtual table, which FTS5 can hand back already in order
(https://www.sqlite.org/fts5.html#sorting_by_auxiliary_function_results), so the joins run in
rank order and stop once ``LIMIT`` rows have passed every predicate. That is a win when most
matches are never looked at, and the whole of its cost when few rows pass: with a selective
filter it walks, one join lookup apiece, nearly every match in rank order. Hence
:func:`choose_lexical_plan` sends it only the searches with nothing narrow to find.

**Three things in the text are load-bearing.** The weights are set per statement, by
``chunks_fts.rank MATCH 'bm25(1.0, 0.4)'``, and not by the table-wide ``rank`` option, which
every other reader of the table would inherit and which the frozen migrations never set. The
column must be written ``chunks_fts.rank``, for the same reason ``bm25()``'s first argument must
be the table's real name. And there is no second sort key: ``ORDER BY chunks_fts.rank, c.seq``
is an expression again and brings the temporary B-tree back. ``rank`` holds exactly the numbers
``bm25()`` returns, so ``rank_score`` means what it does in the other two.

Not available to a search over fewer than two words, nor to one with an explicit document scope,
and why is measured: ``docs/retrieval.md`` §4.1.
"""

SEARCH_SQL_DOCUMENT_FIRST = """
SELECT c.id AS chunk_id, bm25(chunks_fts, 1.0, 0.4) AS rank_score
FROM documents d
CROSS JOIN chunks c ON c.document_id = d.id
CROSS JOIN chunks_fts ON chunks_fts.rowid = c.seq
WHERE chunks_fts MATCH :match
  AND d.deleted_at IS NULL
  AND d.status = 'indexed'
  {extra}
ORDER BY bm25(chunks_fts, 1.0, 0.4), c.seq
LIMIT :limit
"""
"""``SEARCH_SQL`` with the join order forced: documents, then their chunks, then the index.

For an explicit document list small enough to walk. ``CROSS JOIN`` is how SQLite is told not to
reorder: the listed documents are the outer loop, each one's chunks are found by
``(document_id, position)``, and ``chunks_fts`` is asked about one rowid at a time, so the work
is the size of the list rather than the size of the match. It is the opposite of what is wanted
for anything bigger, because the index is probed once per chunk and each probe pays for the
terms' whole posting lists — which is why this is never used without a bound on the chunks
(:data:`MAX_SCOPED_CHUNKS`) and never as a blanket hint on the others.

``c.seq`` is a second sort key here and only here. This statement sorts in a temporary B-tree
anyway, so the key costs nothing, and it makes the order of equal scores the order of the
index, which is what the other two give in practice. SQLite specifies the order of equal scores
in none of them, nor which of a tied group a ``LIMIT`` keeps, so the tests compare ties as sets.
"""


_FUNCTION_WORDS = (
    # interrogatives
    "how what why when where which who whom whose"
    # auxiliaries and modals
    " do does did is are was were be been being am can could should would will shall may"
    " might must has have had"
    # pronouns and determiners
    " i me my we us our you your it its they them their he him his she her this that these"
    " those a an the some any"
    # prepositions and conjunctions
    " of to in on at by for from with about into as than then there and or if so but"
    # contractions that are only a pronoun and an auxiliary
    " i'm i've it's what's that's there's let's"
)
FUNCTION_WORDS = frozenset(_FUNCTION_WORDS.split())
"""Words that carry the shape of a question rather than its subject, dropped from the match.

**Deliberately small, and deliberately missing** ``not``, ``no`` and every negated contraction:
dropping those would make "why is the vector not published" ask for its opposite. This is not
:mod:`manicule.retrieval.homographs`, which keeps its words and only asks for a second reason
before a glossary term expands; these are removed, and only from the lexical leg's match — the
dense leg embeds the whole question.
"""


_APOSTROPHES = str.maketrans(dict.fromkeys("\u2018\u2019\u02bc", "'"))
"""Typographic apostrophes, which macOS and phone keyboards type by default, as the ASCII one.

Without this ``what\u2019s`` loses its apostrophe to the cleaning below, becomes ``whats``, is
no function word, and is required in the match — so a question typed with a smart quote would
match nothing, exactly as every question did before #481.
"""

_INNER = "-_.'"
"""Kept inside a word and trimmed from its ends. FTS5's ``unicode61`` splits on all four, so
neither choice changes what a word matches; keeping them inside keeps ``re-embed`` and
``don't`` recognizable, and trimming them makes ``token.`` the same word as ``token``."""


def match_words(text: str) -> list[str]:
    """Each word of ``text`` as the lexical leg reads it: cleaned, in order, each once.

    Exposed so ``tests/benchmarks/lexical_questions.py`` builds every candidate rule from the
    same words this one does; a measurement whose variants split a query differently would
    credit the match rule with a difference the splitting made.
    """
    words: list[str] = []
    seen: set[str] = set()
    for raw in text.translate(_APOSTROPHES).split():
        word = "".join(char for char in raw if char.isalnum() or char in _INNER).strip(_INNER)
        if word and word.lower() not in seen:
            seen.add(word.lower())
            words.append(word)
    return words


MAX_MATCH_WORDS = 16
"""The most content words a match carries; the rest of a longer query is not searched for.

A disjunction matches every row holding any of its words and ``SEARCH_SQL`` scores all of them
before ``LIMIT``, so a pasted paragraph — and nothing on any surface limits a query's length —
became an ``OR`` over dozens of words that matched almost the whole index. Measured at 100,000
chunks in round 3b of ``tests/benchmarks/lexical_questions.py``: a 120-word paste took a median
of 255 ms uncapped and 82 ms with this cap, with no change in where its section ranked. That is
still over the 50 ms the questions were held to, and the owner chose to ship it so (#485).

Sixteen because it changes nothing a question does: the longest question any round measured
has nine content words. The first sixteen are kept rather than the rarest, because rarity needs
the index's term statistics, which this function does not read.
"""


def match_terms(text: str) -> list[str]:
    """The words :func:`escape_match_query` searches for, in order: what is left of ``text``.

    The one place a query is cut down to its content words and capped, so the number of
    alternatives a match carries is read from the same list that writes it. A plan chosen from
    a count taken some other way could disagree with the expression it was chosen for.
    """
    words = match_words(text)
    content = [word for word in words if word.lower() not in FUNCTION_WORDS]
    return (content or words)[:MAX_MATCH_WORDS]


def escape_match_query(text: str) -> str:
    """Turn user input into an FTS5 ``MATCH`` expression: any content word, each a literal.

    **Every word is quoted, so none can be an operator.** FTS5's query language has operators
    (``AND``, ``OR``, ``NOT``, ``NEAR``, ``*``, ``:``, parentheses). A user typing ``NOT``
    means the word, and a user typing an unbalanced quote means nothing at all — unquoted, both
    are a syntax error rather than a search. The one operator in the expression is the ``OR``
    written here, between quoted phrases, and it never comes from the user's text.

    **Function words are dropped, and the rest are alternatives, not requirements.** BM25's IDF
    does the weighting: a chunk holding the rare words of the question ranks above one holding
    its common ones, and a chunk holding only some of them can still be found. Requiring every
    word — what this did before #481 — made a question match nothing, and requiring every
    content word (#482) still missed a question that named an identifier along with one more
    word the section never writes, such as "when is ``X`` checked". Chosen by three
    pre-registered rounds in ``tests/benchmarks/lexical_questions.py`` (``docs/retrieval.md``
    §4.1); in the third, on 754 such questions over this repository's ``docs/``, mean reciprocal
    rank rose by 0.276, with no measurable loss on 191 natural questions.

    **What that costs is time, and it grows with the corpus.** ``SEARCH_SQL`` scores every row
    the expression matches before ``LIMIT``, and a disjunction matches far more rows than a
    conjunction: at 100,000 chunks the median prose question took 38 to 48 ms against about 1 ms
    for the conjunction. Dropping the function words roughly halves it — with them, the same
    questions matched nearly every row and took 77 to 98 ms. Those figures are the released
    statement's; :func:`choose_lexical_plan` now picks among three by what the search is, and
    ``docs/retrieval.md`` §4.1 measures each.

    **At most :data:`MAX_MATCH_WORDS` content words, the first ones.** A long paste is cut, not
    refused: what the cap bounds is the time a long query costs, which it reduces about threefold
    and does not bring under the bound a question meets.

    **A query made only of function words keeps them.** "What is it" is then a search for those
    three words rather than no search, and ``NOT`` or ``OR`` on its own is still the word. Each
    word is kept once (:func:`match_words`): a repeated phrase is scored twice by ``bm25()``.
    Two spellings the stemmer folds together, ``token`` and ``tokens``, are still two words.

    **The list is English**, like the Porter stemmer in
    :data:`~manicule.storage.models.FTS_TOKENIZER`. A question in another language keeps its
    function words as alternatives, which BM25 weights down where they are common.

    Args:
        text: Whatever the user typed.

    Returns:
        A ``MATCH`` expression, or the empty string when nothing searchable remains.
    """
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in match_terms(text))


class LexicalPlan(StrEnum):
    """Which statement runs a lexical search: the same rows and scores, a different route."""

    DIRECT = "direct"
    RANKED = "ranked"
    DOCUMENT_FIRST = "document-first"


MAX_SCOPED_DOCUMENTS = 64
"""The most listed documents :func:`choose_lexical_plan` will probe the chunks of."""

MAX_SCOPED_CHUNKS = 32
"""The most chunks a listed scope may hold and still be walked document-first.

Measured, not derived: the per-chunk cost of :data:`SEARCH_SQL_DOCUMENT_FIRST` grows with how
common the query's words are, so the size at which the direct statement overtakes it varies by
query. On a synthetic million-chunk index it was about 64 chunks for three common words, over
500 for three middling ones, and about 20 for sixteen common words. 32 is inside the break-even
for every one of those but the last, and where it applies the walk was 2 to 15 times faster
(``docs/retrieval.md`` §4.1)."""

SCOPE_PROBE_LIMIT = MAX_SCOPED_CHUNKS + 1
"""How many chunk positions the probe fetches: one more than the budget, so "over" is visible."""

SCOPE_PROBE_SQL = """
SELECT count(*) FROM (
    SELECT c.seq
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.workspace_id = :workspace_id
      AND d.deleted_at IS NULL
      AND d.id IN ({ids})
    LIMIT :probe_limit
)
"""
"""How many chunks, up to :data:`SCOPE_PROBE_LIMIT`, an explicit document list holds here.

A hint for choosing a statement and nothing else. It reads no text and does not touch the
lexical index, and it never limits or widens a result: the search that follows carries every
predicate whatever this returns, so a count that is zero, stale or wrong costs time and cannot
admit or hide a row. Both ``d.workspace_id`` and ``d.deleted_at IS NULL`` stay in it because
together they are what lets SQLite use ``ix_documents_workspace_live_id`` for the ``id IN``
bound; without the second, it walks the workspace."""


MAX_RANKED_TERMS = 6
"""The most alternatives for which :func:`choose_lexical_plan` uses the ranked statement.

Ranked has to score every match to order them, where the direct statement scores only the rows
that pass the joins. With few words that is cheap beside the join lookups it saves; with many it
is not. On a synthetic million-chunk index the ranked statement was 1.2 to 2.3 times faster than
the direct one at two to six words, a little slower at nine (0.94 times as fast) and slower
again at sixteen (0.87 times)."""


def needs_scope_probe(terms: int, scoped_documents: int) -> bool:
    """Whether :func:`choose_lexical_plan` needs the chunk count of an explicit document scope."""
    return terms >= 1 and 0 < scoped_documents <= MAX_SCOPED_DOCUMENTS


def choose_lexical_plan(
    *,
    terms: int,
    scoped_documents: int,
    scoped_chunks: int | None = None,
    narrowed: bool = False,
) -> LexicalPlan:
    """Which statement answers a search, from facts about it that cost nothing to know.

    The three statements return the same rows with the same scores, so this is a choice of
    speed and nothing rides on it being right. It is a pure function of its arguments so that
    the rule can be tested without a database and the benchmark can report what it would pick.

    Args:
        terms: Alternatives in the match, from :func:`match_terms`.
        scoped_documents: Documents the caller listed. ``0`` means no list; an empty list is
            "no restriction" inside the store, and collection scopes are turned into lists
            (or into "match nothing") before a store is reached.
        scoped_chunks: Chunks those documents hold, up to :data:`SCOPE_PROBE_LIMIT`, from
            :data:`SCOPE_PROBE_SQL`. Required exactly when :func:`needs_scope_probe` says so.
        narrowed: Whether a source, media type, chunk kind, language or time bound restricts
            the search. A reader set does not count; see below.

    Returns:
        :attr:`LexicalPlan.DOCUMENT_FIRST` for a listed scope of at most
        :data:`MAX_SCOPED_DOCUMENTS` documents holding at most :data:`MAX_SCOPED_CHUNKS`
        chunks, because the walk is then the size of the scope. Otherwise
        :attr:`LexicalPlan.RANKED` for a disjunction of two to :data:`MAX_RANKED_TERMS` words
        that is not narrowed and not listed, and :attr:`LexicalPlan.DIRECT` for the rest.

    Raises:
        ValueError: ``terms`` is below one, or a probe-sized scope arrives without its count.

    **Why the rest is direct.** Measured on a synthetic million-chunk index
    (``tests/benchmarks/lexical_plans.py``, ``docs/retrieval.md`` §4.1): the direct statement is
    the released one, so choosing it never regresses a search, and the ranked statement walks the
    matches in rank order until enough pass. That is why it was 1.6 to 2.3 times *slower* when
    the workspace is a small part of the index, and 1.2 to 5 times slower under a narrow filter
    or a listed scope. One word keeps the direct statement, which is the released behavior: the
    synthetic index says ranked was 1.7 times faster for a common word, but a single term is the
    shape the released statement was tuned on, so the rule stays cautious there until a
    measurement on real data says otherwise.

    **A reader set is the one restriction that does not narrow, and that is a call.** Ranked
    was as fast or faster than direct at every share of the index a reader could see from 95%
    down to 5% (0.98 to 2.5 times as fast), and slower when 0.1% was readable (0.75 to 0.97
    times as fast for three of four queries, 1.13 for the fourth). The rule for a restriction
    nothing measured is to keep the direct statement; this one was measured, a gated
    installation's readers usually see most of what exists, and the loss is at the extreme.
    """
    if terms < 1:
        msg = "a search with no terms has no statement"
        raise ValueError(msg)
    if scoped_documents > 0:
        if scoped_documents > MAX_SCOPED_DOCUMENTS:
            return LexicalPlan.DIRECT
        if scoped_chunks is None:
            msg = "a scope of up to MAX_SCOPED_DOCUMENTS documents needs its chunk count"
            raise ValueError(msg)
        return (
            LexicalPlan.DOCUMENT_FIRST if scoped_chunks <= MAX_SCOPED_CHUNKS else LexicalPlan.DIRECT
        )
    if terms == 1 or terms > MAX_RANKED_TERMS or narrowed:
        return LexicalPlan.DIRECT
    return LexicalPlan.RANKED


__all__ = [
    "CREATE_FTS",
    "CREATE_TRIGGERS",
    "CURRENT_TRIGGERS",
    "DROP_TRIGGERS",
    "FTS_SHADOW_TABLES",
    "FTS_TABLE",
    "FUNCTION_WORDS",
    "INTEGRITY_CHECK_FTS",
    "MAX_MATCH_WORDS",
    "MAX_SCOPED_CHUNKS",
    "MAX_SCOPED_DOCUMENTS",
    "REBUILD_FTS",
    "SCOPE_PROBE_LIMIT",
    "SCOPE_PROBE_SQL",
    "SEARCH_SQL",
    "SEARCH_SQL_DOCUMENT_FIRST",
    "SEARCH_SQL_RANKED",
    "LexicalPlan",
    "choose_lexical_plan",
    "create_fts",
    "escape_match_query",
    "match_terms",
    "match_words",
    "needs_scope_probe",
]
