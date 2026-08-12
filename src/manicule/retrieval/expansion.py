"""Query-time glossary lookup: which alias fired, what it expanded to, and what conflicted.

**Outside the pipeline, beside the router.** ``RetrievalStage`` is locked
(:class:`~manicule.core.protocols.RetrievalStage`), and widening it would invalidate every
recorded evaluation result. Nothing here needs it widened: expansion produces a *second query*,
not a different kind of stage, and the retriever runs the declared pipeline over it exactly as
it runs the pipeline over the first. That is the same shape the router, the cache, context
assembly and confidence already have — the retriever's docstring lists three things that are
not stages and says each is outside the pipeline for a reason rather than by omission. This is
a fourth, for the same reason.

**Deterministic, with no model anywhere.** ``bugs/bug2.md`` §5 requires it, and the whole
module is a dictionary lookup, three regular expressions and a word list.

**Two things it will not do.**

*It will not choose between conflicting definitions.* Two definitions of one term in scope is
reported as a conflict and expands nothing. Highest confidence, most recent and
first-alphabetically are all defensible and all silent, and a silently chosen definition
produces an answer that is fluent, cited, and about the wrong thing — the one failure mode a
glossary feature has that plain search does not.

*It will not rewrite every occurrence of an ordinary English word.* A term that is also a
common word needs corroborating evidence before it expands: either the query wrote it the way
the glossary writes it, or the query is asking about it rather than using it. Neither test
needs a corpus statistic, a model, or a threshold anybody tuned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from manicule.core.glossary import (
    ExpansionConflict,
    GlossaryMatch,
    MatchReason,
    QueryExpansion,
    normalise_acronym,
    normalise_expansion,
)
from manicule.retrieval.homographs import is_common_word

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from manicule.core.glossary import GlossaryEntry
    from manicule.core.retrieval import Candidate, Query
    from manicule.retrieval.ports import GlossarySource

GLOSSARY_SCORE_KEY: Final = "glossary"
"""Where a candidate records that it is an authoritative definition of a term that fired.

Written into :attr:`~manicule.core.retrieval.Candidate.scores` beside the legs' entries so a
surface can mark the passage, and deliberately **never** made the effective score: confidence
reads the dense leg's key and would otherwise find a detection confidence sitting in a slot
that means cosine.
"""

# Both apostrophes, because a query typed on a phone carries the typographic one and a query
# typed in a terminal carries the plain one, and ``what's NOW`` must work either way.
_TOKEN_RE: Final = re.compile(r"[\w&/.'’-]+", re.UNICODE)  # noqa: RUF001

_FRAMES: Final[tuple[str, ...]] = (
    r"^\s*{term}\s*\??\s*$",
    r"\bwhat\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?{term}",
    r"\bwhat(?:'s|’s)\s+(?:an?\s+|the\s+)?{term}",  # noqa: RUF001 - both apostrophes
    r"\bwhat\s+(?:does|do|did)\s+{term}\s+(?:mean|stand\s+for|refer\s+to)\b",
    r"{term}\s+(?:stands?\s+for|means?|is\s+short\s+for|is\s+an?\s+acronym)\b",
    r"\b(?:define|explain|expand)\s+(?:the\s+)?{term}",
    r"\b(?:definition|meaning|expansion|abbreviation)\s+of\s+(?:the\s+)?{term}",
    r"\bwho\s+(?:is|are)\s+{term}",
)
"""Query shapes that ask *about* a token rather than using it.

The rule that lets ``what is now?`` reach a glossary while ``restart the daemon now`` does
not, without either one needing a capital letter. A question is not a use, and the difference
is visible in the sentence rather than inferable only from a model.

**It is a rule about queries, never about the corpus.** Nothing here is applied to a document:
manicule does not rewrite what it indexed, so the worst an over-broad frame can do is run one
extra search whose provenance is on screen.

Every ``{term}`` is bounded by :func:`definitional_frame` rather than by a ``\\b`` written into
the pattern, and that is a fix rather than a style. ``\\b`` needs a word character on one side,
so a surface ending in a dot — ``N.O.W.``, which is the punctuation variant this feature is
required to handle — could never match one at the end of a question. The bug was found by the
regression case for punctuation variants, which is what that case is for.
"""


def definitional_frame(text: str, surface: str) -> bool:
    """Whether ``text`` asks about ``surface`` rather than using it.

    The surface is bounded by *character class* lookarounds rather than by ``\\b``. A word
    boundary is defined between a word character and a non-word character, so a term that ends
    in punctuation — ``N.O.W.`` — has no boundary after it to match, and a question ending in
    ``?`` gives it nothing on the other side either. ``(?!\\w)`` asks the question actually
    intended: is this the end of the token, whatever the token ends with.
    """
    term = rf"(?<!\w){re.escape(surface)}(?!\w)"
    return any(
        re.search(frame.format(term=term), text, re.IGNORECASE | re.UNICODE) for frame in _FRAMES
    )


@dataclass(frozen=True, slots=True)
class ExpansionPolicy:
    """What query-time expansion is allowed to do.

    A frozen value rather than a settings object, so that the retriever's behaviour is a
    function of what it was handed and a test can state a policy in one line.
    """

    enabled: bool = True
    """Off means no lookup happens at all — not a lookup whose result is discarded. A disabled
    feature that still queries a store is a feature that can still be slow and can still fail."""

    min_entry_confidence: float = 0.6
    """Detection confidence an entry needs before a *query* will act on it. Separate from the
    ingest threshold on purpose: raising this rejects weak entries already in the index, which
    is the only remedy available to an operator who cannot re-ingest a corpus."""

    homographs: frozenset[str] = field(default_factory=frozenset[str])
    """Extra terms to treat as ordinary English words, normalised. The words that collide with
    a corpus's terms are a property of that corpus, so this extends the shipped list rather
    than replacing it."""

    max_terms: int = 3
    """How many distinct terms one query may expand. A query naming four glossary terms is
    asking something a definition lookup cannot answer, and expanding all of them produces a
    second query that is a list of noun phrases."""


def candidate_surfaces(text: str) -> list[tuple[str, str]]:
    """``(surface, normalised key)`` for every token of ``text`` that could be a term.

    Order is first appearance and duplicates are dropped by *key*, so ``NOW`` and ``now`` in one
    query resolve once — through the first occurrence, whose case is then what the rules read.
    """
    seen: dict[str, tuple[str, str]] = {}
    for token in _TOKEN_RE.findall(text):
        key = normalise_acronym(token)
        if key and key not in seen:
            seen[key] = (token, key)
    return list(seen.values())


def _why_it_fires(
    text: str, surface: str, entry: GlossaryEntry, policy: ExpansionPolicy
) -> MatchReason | None:
    """Which rule admits this occurrence, or ``None`` if none does."""
    if surface in {entry.display, entry.acronym}:
        return MatchReason.EXACT_CASE
    if definitional_frame(text, surface):
        return MatchReason.DEFINITIONAL_FRAME
    key = normalise_acronym(surface)
    if not is_common_word(key) and key not in policy.homographs:
        return MatchReason.UNAMBIGUOUS
    return None


def group_by_key(
    entries: Iterable[GlossaryEntry], policy: ExpansionPolicy
) -> dict[str, list[GlossaryEntry]]:
    """Entries indexed by every key that should find them, weakest ones dropped.

    A store may return an entry through any of its keys, so grouping happens here rather than
    in storage: the alias set is part of the entry, and a store that indexed only the primary
    acronym would make aliases silently unfindable.
    """
    grouped: dict[str, list[GlossaryEntry]] = {}
    for entry in entries:
        if entry.confidence < policy.min_entry_confidence:
            continue
        for key in entry.keys:
            grouped.setdefault(key, []).append(entry)
    return grouped


def _distinct_expansions(entries: Sequence[GlossaryEntry]) -> dict[str, list[GlossaryEntry]]:
    distinct: dict[str, list[GlossaryEntry]] = {}
    for entry in entries:
        distinct.setdefault(normalise_expansion(entry.expansion), []).append(entry)
    return distinct


def expanded_text(original: str, matches: Sequence[GlossaryMatch]) -> str:
    """``original`` with each fired surface replaced by its expansion.

    Substitution rather than concatenation, and it was measured rather than assumed. Against a
    synthetic corpus where a term's expansion is written out only where it is defined, the
    question form carrying the expansion ranked the defining passage far higher than the
    expansion alone did — a query is a sentence, and a sentence with a noun phrase spliced into
    it stays one, while a bare noun phrase is a different kind of text from the one the index
    holds.

    The original is *not* discarded: the caller searches both, which is what ``bugs/bug2.md``
    §3 requires. This produces the second form only.
    """
    text = original
    for match in matches:
        text = re.sub(
            rf"(?<!\w){re.escape(match.surface)}(?!\w)", match.entry.expansion, text, count=1
        )
    return text


async def resolve_expansion(
    query: Query, source: GlossarySource | None, policy: ExpansionPolicy
) -> QueryExpansion:
    """What the glossary has to say about one query.

    Returns a :class:`~manicule.core.glossary.QueryExpansion` in every case, including the case
    where nothing fired — the caller needs to be able to tell "expansion is off", "no term was
    named" and "a term was named and conflicts" apart, and a ``None`` return would collapse all
    three into one.
    """
    nothing = QueryExpansion(original=query.text)
    if not policy.enabled or source is None:
        return nothing

    surfaces = candidate_surfaces(query.text)
    if not surfaces:
        return nothing

    entries = await source.entries_for([key for _, key in surfaces], query.filter)
    grouped = group_by_key(entries, policy)
    if not grouped:
        return nothing

    matches: list[GlossaryMatch] = []
    conflicts: list[ExpansionConflict] = []
    for surface, key in surfaces:
        found = grouped.get(key)
        if not found:
            continue
        distinct = _distinct_expansions(found)
        if len(distinct) > 1:
            # Reported whether or not the occurrence would have fired. A conflict is a fact
            # about the corpus, and hiding it behind a case rule would mean the one query most
            # likely to surface a contradiction — the one that named the term properly — is the
            # only one that ever does.
            conflicts.append(
                ExpansionConflict(
                    key=key,
                    surface=surface,
                    entries=tuple(best for group in distinct.values() for best in group[:1]),
                )
            )
            continue
        entry = max(next(iter(distinct.values())), key=lambda item: item.confidence)
        if len(matches) >= policy.max_terms:
            continue
        reason = _why_it_fires(query.text, surface, entry, policy)
        if reason is None:
            continue
        matches.append(GlossaryMatch(surface=surface, key=key, reason=reason, entry=entry))

    if not matches:
        return QueryExpansion(original=query.text, conflicts=tuple(conflicts))

    second = expanded_text(query.text, matches)
    return QueryExpansion(
        original=query.text,
        expanded="" if second == query.text else second,
        matches=tuple(matches),
        conflicts=tuple(conflicts),
    )


# --- merging the two rankings ---------------------------------------------------------------


def _combined_scores(first: Mapping[str, float], second: Mapping[str, float]) -> dict[str, float]:
    """Per key, the better of two runs' opinions of one passage.

    The two runs are two readings of one request, so a passage's support is the best either
    found. Taking the maximum is also the only combination that cannot make expansion *lower* a
    reported confidence, which matters more than the arithmetic: a feature that can quietly
    downgrade the answers it does not help is a feature nobody can leave switched on.
    """
    merged = dict(first)
    for name, value in second.items():
        merged[name] = max(merged.get(name, value), value)
    return merged


def mark_authoritative(candidate: Candidate, confidence: float) -> Candidate:
    """Record that this passage is a definition, without disturbing its effective score.

    Not :meth:`~manicule.core.retrieval.Candidate.scored_by`, and the difference is the bug it
    avoids: that method makes the value the candidate's *effective* score, so a detection
    confidence would land in the slot fusion and the surface read as a ranking score, and a
    passage detected at 0.95 would outrank everything for a reason that is not about the query.
    """
    scores = {**candidate.scores, GLOSSARY_SCORE_KEY: confidence}
    return candidate.model_copy(update={"scores": scores})


def merge_rankings(
    original: Sequence[Candidate],
    expanded: Sequence[Candidate],
    *,
    promoted: Sequence[Candidate] = (),
    limit: int,
) -> list[Candidate]:
    """One ranking from two, with the authoritative definitions first.

    Three rules, in order:

    1. **Definitions lead.** A passage known to define a term the query named is not a search
       result that happened to score well; it is the answer to the question that was asked, and
       its position should not depend on a cosine. This is the whole of the feature: an exact
       alias hit is a *lookup*, and a lookup that then had to win a similarity contest against
       forty passages using the same word would not be one.
    2. **Then the two rankings, alternating.** Neither query form is privileged — the original
       is what the user typed and the expanded one is what the glossary says they meant — so
       they interleave rather than concatenate. Concatenating would bury the second list below
       the first and make the second search pointless at any realistic limit.
    3. **A passage appears once.** Found by both runs, it keeps the better of the two opinions
       for every leg and the earlier of the two positions.

    Deterministic in full: same inputs, same output, every run.
    """
    merged: dict[str, Candidate] = {}
    order: list[str] = []

    def offer(candidate: Candidate) -> None:
        chunk_id = candidate.chunk.id
        seen = merged.get(chunk_id)
        if seen is None:
            merged[chunk_id] = candidate
            order.append(chunk_id)
            return
        scores = _combined_scores(seen.scores, candidate.scores)
        merged[chunk_id] = seen.model_copy(
            update={"scores": scores, "score": max(seen.score, candidate.score)}
        )

    for candidate in promoted:
        offer(candidate)
    for first, second in _interleaved(original, expanded):
        if first is not None:
            offer(first)
        if second is not None:
            offer(second)

    return [merged[chunk_id] for chunk_id in order[:limit]]


def _interleaved(
    first: Sequence[Candidate], second: Sequence[Candidate]
) -> Iterable[tuple[Candidate | None, Candidate | None]]:
    for index in range(max(len(first), len(second))):
        yield (
            first[index] if index < len(first) else None,
            second[index] if index < len(second) else None,
        )


__all__ = [
    "GLOSSARY_SCORE_KEY",
    "ExpansionPolicy",
    "candidate_surfaces",
    "definitional_frame",
    "expanded_text",
    "group_by_key",
    "mark_authoritative",
    "merge_rankings",
    "resolve_expansion",
]
