"""Glossary vocabulary: what a definition is, and what matching one produced.

The types two sides speak. Ingest detects definitions and writes :class:`GlossaryEntry`;
retrieval reads them back and produces a :class:`QueryExpansion`. Neither imports the other,
and nothing here imports a database, a model or a tokenizer — the whole feature is
deterministic by construction, and a module that could reach a generative model would make
that claim unverifiable.

**Normalization is defined once, here.** ``NOW``, ``now``, ``N.O.W.`` and ``(NOW)`` are the
same key; a store that normalized on write and a query path that normalized differently on
read would produce a lookup that silently missed, which is indistinguishable from a corpus
with no glossary in it.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MIN_ACRONYM_LENGTH: Final = 2
"""Below this a "term" is a letter, and every corpus is full of those."""

MAX_ACRONYM_LENGTH: Final = 12
"""Above this the token is a word, not an abbreviation. Twelve is generous on purpose: the
cost of admitting a long one is a lookup that misses, and the cost of refusing a real one is a
definition nobody can find."""

_STRIPPABLE: Final = "()[]{}<>\"'“”‘’.,;:!?…"  # noqa: RUF001 - the curly quotes are the point
"""Punctuation removed from either end of a surface form before it becomes a key.

Kept as one constant because query-time and ingest-time stripping must agree exactly. A
trailing question mark is the specific case that matters: ``What is NOW?`` tokenizes to
``NOW?`` and would otherwise never match anything.
"""

_INTERNAL_DOTS: Final = re.compile(r"(?<=\w)\.(?=\w)")
"""``N.O.W.`` and ``NOW`` are the same term written two ways."""


def normalize_acronym(surface: str) -> str:
    """The lookup key for a surface form.

    Case-folded to upper, stripped of surrounding punctuation, internal dots removed, and
    NFKC-normalized so that a full-width or ligatured variant does not become a second term.

    Returns the empty string for anything that cannot be a key, which callers test rather than
    guessing from length: an empty key that reached a store would match every row whose column
    happened to be empty.
    """
    folded = unicodedata.normalize("NFKC", surface).strip()
    folded = _INTERNAL_DOTS.sub("", folded)
    folded = folded.strip(_STRIPPABLE)
    if not folded:
        return ""
    upper = folded.upper()
    if not (MIN_ACRONYM_LENGTH <= len(upper) <= MAX_ACRONYM_LENGTH):
        return ""
    if not any(character.isalpha() for character in upper):
        return ""
    if not all(character.isalnum() or character in "-&/" for character in upper):
        return ""
    return upper


def normalize_expansion(expansion: str) -> str:
    """A comparison key for two expansions, so trivial differences are not conflicts.

    Case and internal whitespace only. Word order, wording and punctuation are **not**
    normalized away: "Network Operations Workspace" and "Workspace, Network Operations" are two
    claims about what a term means, and deciding they are one is exactly the silent choice this
    feature is forbidden to make.
    """
    return " ".join(unicodedata.normalize("NFKC", expansion).casefold().split()).strip(" .")


class DefinitionForm(StrEnum):
    """The written shape a definition was recognized in.

    Recorded rather than discarded because it is half of the confidence: a two-column table row
    inside a page titled "Glossary" is a definition, and the same two strings either side of a
    bracket in running prose might be an aside.
    """

    EM_DASH = "em_dash"
    """``NOW — Network Operations Workspace``. Also en dash and a spaced hyphen."""

    COLON = "colon"
    """``NOW: Network Operations Workspace``."""

    PARENTHETICAL = "parenthetical"
    """``Network Operations Workspace (NOW)``. The weakest form, because a bracket in prose is
    not always a definition."""

    DEFINITION_LIST = "definition_list"
    """A Markdown definition list: the term on one line, ``: expansion`` on the next."""

    TABLE_ROW = "table_row"
    """A two-column table row, ``| NOW | Network Operations Workspace |``."""

    HEADING = "heading"
    """A heading naming the term, with the expansion in the text beneath it."""


class GlossaryEntry(BaseModel):
    """One definition, with everything needed to cite it and to decide whether to trust it.

    Scoped by :attr:`document_id`, which already carries the workspace —
    :func:`~manicule.core.ids.document_id` takes it as the first component of its digest — so
    an entry cannot be attributed to a tenant that does not own the document it came from.
    Collection scope is a property of the document rather than of the entry, and is applied by
    whoever reads these back; copying it here would create a second copy that can disagree with
    the membership table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    acronym: str = Field(min_length=MIN_ACRONYM_LENGTH, description="The normalized lookup key.")
    display: str = Field(
        min_length=1,
        description="The canonical form as the source wrote it — ``NOW``, not ``now``. What a "
        "reader is shown, so a citation quotes the document rather than our normalization.",
    )
    expansion: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(
        min_length=1,
        description="The chunk the definition was read out of. This is the citation: an "
        "expansion presented without one is an assertion manicule cannot support, which "
        "``bugs/bug2.md`` §3 forbids outright.",
    )
    location: str = Field(
        default="",
        description="Where in the document, in the source's own terms — a heading breadcrumb "
        "or a line reference. Display only; the chunk id is what resolves.",
    )
    form: DefinitionForm
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the text reads as a definition rather than as prose that "
        "happens to contain two adjacent strings. Never a probability of correctness.",
    )
    aliases: tuple[str, ...] = Field(
        default=(),
        description="Other normalized keys that resolve to this entry — a dotted spelling, a "
        "second short form the source gave. The acronym itself is not repeated here.",
    )

    @property
    def keys(self) -> tuple[str, ...]:
        """Every normalized key that should find this entry."""
        return (self.acronym, *self.aliases)


class MatchReason(StrEnum):
    """Why an occurrence in a query was allowed to expand.

    Recorded per match because these are the rules that stop the feature rewriting every
    ordinary use of an English word, and a rule nobody can see fire is a rule nobody can check.
    """

    EXACT_CASE = "exact_case"
    """The query wrote the term the way the glossary writes it — ``NOW``, not ``now``."""

    DEFINITIONAL_FRAME = "definitional_frame"
    """The query asks *about* the token rather than using it: "what is now", "define now",
    "what does now stand for". A question is not a use."""

    UNAMBIGUOUS = "unambiguous"
    """The token is not an ordinary English word, so no case or framing evidence is needed."""


class GlossaryMatch(BaseModel):
    """One alias that fired, and the entry it resolved to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    surface: str = Field(min_length=1, description="The token as the query wrote it.")
    key: str = Field(min_length=1, description="The normalized key it resolved through.")
    reason: MatchReason
    entry: GlossaryEntry


class ExpansionConflict(BaseModel):
    """One term with more than one definition in scope, reported rather than resolved.

    **Nothing picks a winner.** Highest detection confidence, most recently indexed and
    first-alphabetically are all defensible and all silent, and a silently chosen definition is
    the failure mode a glossary feature has that plain search does not: the answer is fluent,
    cited, and about the wrong thing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    entries: tuple[GlossaryEntry, ...] = Field(min_length=2)

    @property
    def expansions(self) -> tuple[str, ...]:
        """The distinct expansions in play, in the order they were found."""
        seen: dict[str, str] = {}
        for entry in self.entries:
            seen.setdefault(normalize_expansion(entry.expansion), entry.expansion)
        return tuple(seen.values())


class QueryExpansion(BaseModel):
    """What glossary lookup did to one query, whole.

    Carries the original text as well as the expanded one, because retaining the original is a
    requirement rather than an implementation detail: an expanded query is a second reading of
    the question, never a replacement for it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    original: str = Field(min_length=1)
    expanded: str = Field(
        default="",
        description="The second query form, or empty when nothing fired. Empty and equal to "
        "the original are different outcomes and neither is inferred from the other.",
    )
    matches: tuple[GlossaryMatch, ...] = ()
    conflicts: tuple[ExpansionConflict, ...] = ()

    @property
    def fired(self) -> bool:
        """Whether a second query form was produced."""
        return bool(self.expanded) and bool(self.matches)

    @property
    def definition_chunk_ids(self) -> tuple[str, ...]:
        """The chunks that carry the definitions that fired, in match order, deduplicated."""
        return tuple(dict.fromkeys(match.entry.chunk_id for match in self.matches))


__all__ = [
    "MAX_ACRONYM_LENGTH",
    "MIN_ACRONYM_LENGTH",
    "DefinitionForm",
    "ExpansionConflict",
    "GlossaryEntry",
    "GlossaryMatch",
    "MatchReason",
    "QueryExpansion",
    "normalize_acronym",
    "normalize_expansion",
]
