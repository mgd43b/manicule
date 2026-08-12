"""Finding explicit glossary definitions in text that has already been chunked.

Deterministic, and that is a requirement rather than a preference: ``bugs/bug2.md`` §5 fixes
that the core lookup must work without a generative model, so detection is regular expressions
over lines and nothing else. There is no model to be unavailable, no call to be rate limited,
and re-ingesting a document produces the same entries on every machine.

**Detection runs over chunks, not over blocks.** A definition has to be citable, and the chunk
is what a citation resolves to — recording a definition against a block would leave the
expansion with a source it could not point at, which §3 forbids outright. It costs nothing:
the chunker preserves source text verbatim, so the lines a block would have offered are the
lines a chunk offers.

**The hard part is not finding definitions, it is refusing prose.** ``Note: the scheduler
restarts nightly`` is exactly the shape of ``NOW: Network Operations Workspace``, and a
detector that admits the first fills a glossary with sentences. Two independent gates apply,
and they are independent on purpose:

1. **Shape.** The term must be written like an abbreviation — predominantly upper case. This
   is what rejects ``Note:`` without consulting anything else, and it cannot be bought off by
   a confident-looking context.
2. **Confidence.** Everything else is evidence, combined and compared against a threshold: the
   written form, whether the expansion's initials actually spell the term, and whether the
   surrounding document says it is a glossary.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from manicule.core.glossary import (
    MAX_ACRONYM_LENGTH,
    DefinitionForm,
    GlossaryEntry,
    normalise_acronym,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from manicule.core.content import Chunk

MIN_DEFINITION_CONFIDENCE: Final = 0.6
"""Below this, the text is prose that resembles a definition.

Set from the combinations below rather than from a sweep, and the arithmetic is the argument:
a dash form whose initials spell the term clears it on its own (0.80); a dash form on a page
that calls itself a glossary clears it on that alone (0.60); a colon form with neither piece of
evidence does not (0.40), because a colon in prose is a colon in prose.
"""

_FORM_WEIGHT: Final[dict[DefinitionForm, float]] = {
    DefinitionForm.DEFINITION_LIST: 0.55,
    DefinitionForm.TABLE_ROW: 0.55,
    DefinitionForm.EM_DASH: 0.45,
    DefinitionForm.HEADING: 0.45,
    DefinitionForm.COLON: 0.40,
    DefinitionForm.PARENTHETICAL: 0.35,
}
"""How much the written form alone is worth.

A definition list and a table row are structures a writer builds *to* define things, so they
carry most of the way on their own. A parenthetical is worth least because ``the workspace
(now retired)`` is the same shape as ``Network Operations Workspace (NOW)``.
"""

INITIALS_EVIDENCE: Final = 0.35
"""Worth for an expansion whose initials spell the term.

The strongest single signal available without a model, and the cheapest: it is a property of
the two strings rather than of anything around them, so it survives a document being retitled,
re-chunked or quoted somewhere else.
"""

GLOSSARY_CONTEXT_EVIDENCE: Final = 0.15
"""Worth for a document or heading that says it is a glossary.

Deliberately the smallest of the three. It is evidence about the *page*, and a page can be a
glossary and still contain a sentence with a colon in it — so it must never be enough on its
own to admit a form that carries nothing else.
"""

GLOSSARY_WORDS: Final[frozenset[str]] = frozenset(
    {
        "abbreviation",
        "abbreviations",
        "acronym",
        "acronyms",
        "definition",
        "definitions",
        "glossary",
        "initialism",
        "initialisms",
        "nomenclature",
        "terminology",
        "terms",
        "vocabulary",
    }
)

MAX_EXPANSION_WORDS: Final = 10
"""Longer than this and it is a sentence about the term, not the term written out."""

_UPPERCASE_SHARE: Final = 0.6
"""How much of a term's alphabetic content must be upper case.

Not 1.0, because ``IPv6`` and ``mDNS`` are abbreviations and a rule that refused them would be
wrong about real corpora. Not lower, because at 0.5 a two-letter capitalised word — ``It``,
``Of`` — becomes an abbreviation and every sentence beginning becomes a candidate.
"""

_FUNCTION_WORDS: Final[frozenset[str]] = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
)
"""Words an acronym is free to skip. ``ATLAS`` spells out with ``And``; ``RAM`` skips ``of``.
Both spellings are checked, because which one a writer used is not knowable from the string."""

_TERM = rf"[A-Za-z][\w&/.-]{{0,{MAX_ACRONYM_LENGTH - 1}}}"

# RUF001 on the next line is the whole point of it: an em dash and an en dash are the two
# characters a writer actually reaches for here, and replacing them with a hyphen would leave
# the commonest glossary form in existence undetectable.
_DASH_RE: Final = re.compile(rf"^\s*(?P<term>{_TERM})\s+[—–]\s+(?P<expansion>\S.*?)\s*$")  # noqa: RUF001
_HYPHEN_RE: Final = re.compile(rf"^\s*(?P<term>{_TERM})\s+-\s+(?P<expansion>\S.*?)\s*$")
_COLON_RE: Final = re.compile(rf"^\s*(?P<term>{_TERM})\s*:\s+(?P<expansion>\S.*?)\s*$")
_TABLE_RE: Final = re.compile(r"^\s*\|\s*(?P<term>[^|]+?)\s*\|\s*(?P<expansion>[^|]+?)\s*\|\s*$")
_DEFINITION_MARKER_RE: Final = re.compile(r"^\s*:\s+(?P<expansion>\S.*?)\s*$")
_HEADING_RE: Final = re.compile(rf"^\s*#{{1,6}}\s+(?P<term>{_TERM})\s*$")
# Likewise the typographic apostrophe: a word processor writes one and a plain editor writes
# the other, and an expansion is not allowed to depend on which produced the page.
_PARENTHETICAL_RE: Final = re.compile(
    rf"(?P<expansion>[A-Za-z][\w'’-]*(?:[ \t]+[\w'’&/-]+){{1,{MAX_EXPANSION_WORDS - 1}}})"  # noqa: RUF001
    r"\s*\(\s*(?P<terms>[^()]{2,64}?)\s*\)"
)
_TABLE_RULE_RE: Final = re.compile(r"^[\s|:-]+$")


def acronym_shaped(surface: str) -> bool:
    """Whether a term is *written* like an abbreviation.

    The gate that does not negotiate. Every other signal in this module is evidence to be
    weighed; this one is a property of the token itself, and it is what keeps ``Note:``,
    ``Warning:`` and ``Example -`` out of a glossary however glossary-like the page around them
    looks.
    """
    letters = [character for character in surface if character.isalpha()]
    if len(letters) < 2:  # noqa: PLR2004 - restating MIN_ACRONYM_LENGTH would not be clearer
        return False
    upper = sum(1 for character in letters if character.isupper())
    return surface[:1].isupper() and upper / len(letters) >= _UPPERCASE_SHARE


def initials_of(expansion: str, *, skip_function_words: bool) -> str:
    """The letters an expansion's words begin with, upper case."""
    words = [word for word in re.split(r"[\s/-]+", expansion) if word]
    if skip_function_words:
        words = [word for word in words if word.casefold() not in _FUNCTION_WORDS]
    return "".join(word[0] for word in words if word[:1].isalpha()).upper()


def initials_match(acronym: str, expansion: str) -> bool:
    """Whether ``expansion``'s initials spell ``acronym``.

    Both spellings are accepted — every word, and every word that is not a function word —
    because ``ATLAS`` includes its ``And`` and ``RAM`` skips its ``of``, and nothing in the
    strings says which convention the writer followed.
    """
    letters = "".join(character for character in acronym if character.isalnum()).upper()
    if not letters:
        return False
    return letters in {
        initials_of(expansion, skip_function_words=False),
        initials_of(expansion, skip_function_words=True),
    }


def _mentions_glossary(*texts: Iterable[str]) -> bool:
    for group in texts:
        for text in group:
            if any(word in GLOSSARY_WORDS for word in re.findall(r"[a-z]+", text.casefold())):
                return True
    return False


def _usable_expansion(expansion: str) -> str:
    """``expansion`` cleaned up, or the empty string if it is not one.

    Rejects the two shapes that look like definitions and are not: something with more than one
    sentence in it, and something long enough to be a description. Both are prose that happens
    to sit after a separator.
    """
    trimmed = expansion.strip().strip("|").strip()
    trimmed = re.sub(r"\s+", " ", trimmed).rstrip(".").strip()
    if not trimmed or ". " in trimmed:
        return ""
    words = trimmed.split()
    if not (1 <= len(words) <= MAX_EXPANSION_WORDS):
        return ""
    if not any(character.isalpha() for character in trimmed):
        return ""
    return trimmed


def score_definition(
    acronym: str, expansion: str, form: DefinitionForm, *, glossary_context: bool
) -> float:
    """How strongly this reads as a definition rather than as prose.

    Additive and capped, with each term's justification on its own constant. Not a probability
    of anything: it answers "is this text a definition", never "is this definition right".
    """
    score = _FORM_WEIGHT[form]
    if initials_match(acronym, expansion):
        score += INITIALS_EVIDENCE
    if glossary_context:
        score += GLOSSARY_CONTEXT_EVIDENCE
    return min(1.0, round(score, 4))


class _Candidate:
    __slots__ = ("display", "expansion", "extra", "form")

    def __init__(
        self, display: str, expansion: str, form: DefinitionForm, extra: tuple[str, ...] = ()
    ) -> None:
        self.display = display
        self.expansion = expansion
        self.form = form
        self.extra = extra


_LEADING_ARTICLES: Final[frozenset[str]] = frozenset({"a", "an", "the", "our", "its", "this"})
"""Words that begin the sentence rather than the term.

``The Network Operations Workspace (NOW)`` defines three words, not four, and a detector that
kept the article would store an expansion no document ever writes and no query ever matches.
"""


def _phrase_before(captured: str, acronym: str) -> str:
    """Which words of ``captured`` are the term, rather than the sentence around it.

    A parenthetical has no left-hand delimiter — the regular expression has to guess where the
    phrase starts, and it guesses greedily, so ``The Network Operations Workspace (NOW)`` yields
    a four-word expansion for a three-letter acronym.

    Resolved by asking the acronym: the **shortest** suffix whose initials spell it is the
    phrase the writer meant. Shortest rather than longest, and that is the whole of the fix —
    ``initials_match`` skips function words, so ``The Network Operations Workspace`` spells
    ``NOW`` just as well as ``Network Operations Workspace`` does, and taking the first match
    from the left keeps the article every time.

    When nothing spells it, the leading article is dropped and the rest is kept, because an
    article is never part of a term and everything else might be.
    """
    words = captured.split()
    for length in range(1, len(words) + 1):
        candidate = " ".join(words[len(words) - length :])
        if initials_match(acronym, candidate):
            return candidate
    while words and words[0].casefold() in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words)


def _from_line(line: str, previous: str) -> list[_Candidate]:
    """Every definition one line offers, in the order the forms are tried.

    ``previous`` is the line above, which the definition-list form needs and nothing else uses.
    """
    found: list[_Candidate] = []

    for pattern, form in (
        (_DASH_RE, DefinitionForm.EM_DASH),
        (_HYPHEN_RE, DefinitionForm.EM_DASH),
        (_COLON_RE, DefinitionForm.COLON),
    ):
        matched = pattern.match(line)
        if matched:
            found.append(
                _Candidate(matched["term"].strip(), matched["expansion"], form),
            )
            break

    table = _TABLE_RE.match(line)
    if table and not _TABLE_RULE_RE.match(line):
        found.append(
            _Candidate(table["term"].strip(), table["expansion"], DefinitionForm.TABLE_ROW)
        )

    marker = _DEFINITION_MARKER_RE.match(line)
    if marker and previous.strip():
        found.append(
            _Candidate(previous.strip(), marker["expansion"], DefinitionForm.DEFINITION_LIST)
        )

    for parenthetical in _PARENTHETICAL_RE.finditer(line):
        terms = [part.strip() for part in parenthetical["terms"].split(",") if part.strip()]
        if terms:
            found.append(
                _Candidate(
                    terms[0],
                    _phrase_before(parenthetical["expansion"], terms[0]),
                    DefinitionForm.PARENTHETICAL,
                    tuple(terms[1:]),
                )
            )

    return found


def _heading_definitions(lines: Sequence[str], breadcrumb: Sequence[str]) -> list[_Candidate]:
    """Headings that name a term, with the expansion in the text beneath them.

    **Two routes, because a heading reaches a chunk two ways.** Plain text and Markdown that a
    parser left alone keep the ``### NOW`` line, so the first route reads it out of the text.
    But the structural chunker lifts headings into
    :attr:`~manicule.core.content.Chunk.heading_path` and the ``#`` never appears in
    :attr:`~manicule.core.content.Chunk.text` at all — so on the corpus this feature is actually
    for, a detector with only the first route finds nothing and looks like it works.
    """
    found: list[_Candidate] = []
    for index, line in enumerate(lines):
        heading = _HEADING_RE.match(line)
        if heading is None:
            continue
        body = next((later for later in lines[index + 1 :] if later.strip()), "")
        if body:
            found.append(
                _Candidate(heading["term"].strip(), body, DefinitionForm.HEADING),
            )

    innermost = breadcrumb[-1].strip() if breadcrumb else ""
    body = next((line for line in lines if line.strip()), "")
    if innermost and body and not _HEADING_RE.match(body):
        found.append(_Candidate(innermost, body, DefinitionForm.HEADING))
    return found


def detect_in_chunk(chunk: Chunk, *, glossary_context: bool = False) -> list[GlossaryEntry]:
    """Every definition one chunk states, deduplicated by term and expansion.

    Args:
        chunk: The chunk to read. Its own text only — a definition is recorded against the
            passage that states it, so the citation resolves to something a reader can check.
        glossary_context: Whether the *document* around this chunk says it is a glossary.
            Passed in rather than inferred here, because the evidence lives in the title, and a
            chunk does not carry one.
    """
    lines = chunk.text.splitlines()
    context = glossary_context or _mentions_glossary(chunk.heading_path, lines[:1])
    location = " > ".join(chunk.heading_path)

    candidates = _heading_definitions(lines, chunk.heading_path)
    previous = ""
    for line in lines:
        candidates.extend(_from_line(line, previous))
        previous = line

    entries: dict[tuple[str, str], GlossaryEntry] = {}
    for candidate in candidates:
        if not acronym_shaped(candidate.display):
            continue
        acronym = normalise_acronym(candidate.display)
        expansion = _usable_expansion(candidate.expansion)
        if not acronym or not expansion or normalise_acronym(expansion) == acronym:
            continue
        confidence = score_definition(acronym, expansion, candidate.form, glossary_context=context)
        if confidence < MIN_DEFINITION_CONFIDENCE:
            continue
        aliases = tuple(
            dict.fromkeys(
                key
                for key in (normalise_acronym(extra) for extra in candidate.extra)
                if key and key != acronym
            )
        )
        entry = GlossaryEntry(
            acronym=acronym,
            display=candidate.display.strip(),
            expansion=expansion,
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            location=location,
            form=candidate.form,
            confidence=confidence,
            aliases=aliases,
        )
        key = (acronym, expansion.casefold())
        previous_entry = entries.get(key)
        # One definition stated twice in a chunk is one definition. The more confident reading
        # wins, which is a choice *within* an agreeing pair rather than between disagreeing
        # ones — the second is the one this feature is forbidden to make.
        if previous_entry is None or previous_entry.confidence < confidence:
            entries[key] = entry
    return list(entries.values())


def detect_entries(
    chunks: Sequence[Chunk], *, title: str = "", media_type: str = ""
) -> list[GlossaryEntry]:
    """Every definition a document's chunks state.

    Args:
        chunks: The document's chunks, as they will be stored.
        title: The document's title. Read only for the word "glossary" and its relatives.
        media_type: The document's media type, unused today and named so that a caller does not
            have to be changed when a format-specific rule arrives.
    """
    del media_type
    context = _mentions_glossary([title])
    found: list[GlossaryEntry] = []
    for chunk in chunks:
        found.extend(detect_in_chunk(chunk, glossary_context=context))
    return found


__all__ = [
    "GLOSSARY_CONTEXT_EVIDENCE",
    "GLOSSARY_WORDS",
    "INITIALS_EVIDENCE",
    "MAX_EXPANSION_WORDS",
    "MIN_DEFINITION_CONFIDENCE",
    "acronym_shaped",
    "detect_entries",
    "detect_in_chunk",
    "initials_match",
    "initials_of",
    "score_definition",
]
