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
"""Longer than this and it is a sentence about the term, not the term written out.

Applied to whichever reading of the right-hand side :func:`core_expansion` settles on, so a
definition trailed by a description is measured on its expansion rather than on the description
— which is what stopped ``NOVA — Network Operations Visibility Assistant, a service used to
correlate operational signals across systems`` from being recorded at all.
"""

_NEVER_OPENS_AN_EXPANSION: Final[frozenset[str]] = frozenset(
    {
        # Subordinators. What follows one is the circumstance in which something is true, and a
        # circumstance is not a thing that can be named: ``when enabled, the process starts
        # automatically`` says *when*, and no term is called ``when``.
        "after",
        "although",
        "because",
        "before",
        "if",
        "once",
        "since",
        "though",
        "unless",
        "until",
        "when",
        "whenever",
        "whether",
        "while",
        # Words that point instead of naming. ``this paragraph describes an operational
        # consideration`` refers to something already on the page; an expansion is self-contained
        # and refers to nothing, because it *is* the thing being introduced. ``_LEADING_ARTICLES``
        # already makes this judgement about ``this`` for the parenthetical form — same call,
        # applied to the other forms.
        "he",
        "it",
        "she",
        "that",
        "these",
        "they",
        "this",
        "those",
        "we",
        "you",
        # Existential ``there``. ``there is no supported route`` asserts; it does not name.
        "there",
    }
)
"""Words no expansion begins with, tested at the first word and nowhere else.

**The rule that refuses prose on a page that admits everything else.** A spaced hyphen on a page
titled "Glossary" scores exactly ``MIN_DEFINITION_CONFIDENCE`` — ``_FORM_WEIGHT[EM_DASH]`` plus
:data:`GLOSSARY_CONTEXT_EVIDENCE`, 0.45 + 0.15 — so *every* upper-case token followed by a dash
is admitted there on the strength of the page alone. Measured on ``origin/main`` before this
rule existed, both ``NOTE - this paragraph describes an operational consideration, not a term``
and ``API - when enabled, the process starts automatically`` were recorded as definitions: nine
words and six, so :data:`MAX_EXPANSION_WORDS` never fired on either and nothing else looked.

**What this does not test, stated because an earlier draft claimed it did.** It is not a test for
a noun phrase, and the fixtures above are not counter-examples to one: ``this paragraph`` *is* a
noun phrase. What actually disqualifies both of them is the finite verb — ``describes``,
``starts`` — and nothing here looks for a verb. Finding one is a parser's job, this module is
deterministic by construction and has no parser to call, and a word list that pretended to do it
would be a rule whose justification is wider than its mechanism. So the claim is narrowed to what
the list can actually support: these particular words do not begin expansions, one word at a
time, for the two reasons given beside them. Prose opening with an ordinary noun or an imperative
verb — ``WARNING — do not edit these records by hand`` — is admitted, and that is a known gap.

**Do not close it by adding ``note``, ``warning`` and ``caution`` as refused *terms*.** The
objection is not that such a list would overclaim — it would not; admonition labels are document
furniture and a list of them tests exactly what it names. The objection is the trade. ``WARNING —
a log level indicating a recoverable condition`` is an ordinary definition in operations and
logging documentation, which is the corpus this project exists for: the gate buys one false
positive and sells a false negative in the domain most likely to be indexed. That stays a bad
trade however carefully the list is drawn, so drawing it more carefully is not the fix.

The fix, if a real corpus ever needs one, is **structural and lives in the parser rather than
here**. An admonition in Confluence storage format is ``<ac:structured-macro ac:name="warning">``
and not a ``TERM — text`` line at all; :mod:`manicule.connectors.macros` already reads those
elements by name. A parser knows what a word list can only guess at, and by the time text reaches
this module the distinction has been flattened away.

**A first word written like an abbreviation is the abbreviation, not the pronoun**, so
:func:`acronym_shaped` is consulted before the list. This is not a nicety: it was measured.
``ITSM — IT service management`` and ``ITIL — IT infrastructure library`` are ordinary
definitions whose initials do not spell their terms, so the list is the only judge of them, and
without the shape test ``IT`` collides with the pronoun ``it`` and both are silently lost. The
same gate that tells ``NOW`` from ``Note`` on the left of the dash tells ``IT`` from ``it`` on
the right of it.

The list never overrides initials evidence. If a phrase's initials spell the term then the two
strings agree about what the term is, which is stronger than anything one word can say — which
is why ``ONCE — Operational Node Configuration Engine`` and ``WHEN — Workload Health Event
Notifier`` survive being their own counter-examples.
"""

_DESCRIPTION_BOUNDARY_RE: Final = re.compile(r"[,;]|(?<=[.!?])\s")
"""Where a description may begin: a comma, a semicolon, or the end of a sentence.

Each match yields a *candidate* prefix and nothing more. Truncating at the first comma
unconditionally is the obvious version of this and it is wrong in both directions — it would cut
``Retention And Vault Index Node Export, Cold`` in half, and it would hand
``when enabled, the process starts automatically`` a two-word prefix that looks far more like an
expansion than the sentence it came from. :func:`core_expansion` still has to be convinced.
"""

_UPPERCASE_SHARE: Final = 0.6
"""How much of a term's alphabetic content must be upper case.

Not 1.0, because ``IPv6`` is an abbreviation — two of its three letters are upper case, so it
clears 0.6 and would fail a stricter rule. Not lower, because at 0.5 a two-letter capitalised
word — ``It``, ``Of`` — becomes an abbreviation and every sentence beginning becomes a candidate.

**This share is only half of the gate, and the other half is not negotiable by tuning it.**
:func:`acronym_shaped` also requires an initial capital, so a term beginning lower case is refused
at the first character whatever its share. ``mDNS`` is 0.75 upper and still refused; so are
``eSIM`` and ``gRPC``. This docstring used to cite ``mDNS`` as a reason for the value of this
constant, which was wrong twice over: the constant does not admit it, and no value of the constant
could. Lower-case-initial abbreviations are outside what this module detects. Widening it is a
behaviour change across all six written forms — every sentence that begins ``it``, ``the`` or
``and`` becomes a candidate term — and it wants its own measurement, not an edit here.
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


def has_a_refused_opening(expansion: str) -> bool:
    """Whether ``expansion`` begins with a word listed in :data:`_NEVER_OPENS_AN_EXPANSION`.

    Named for what it tests rather than for what a phrase beginning that way tends to be. It
    reads one word and compares it against a list; it does not decide whether the phrase is a
    clause, a noun phrase or a sentence, and calling it something that implied otherwise would be
    the wider-than-the-mechanism claim the constant's docstring warns about.

    An abbreviation is exempt, because ``IT`` and ``it`` are different words that casefold to one
    — see the constant.
    """
    first = next((word for word in re.split(r"[^\w'’]+", expansion) if word), "")  # noqa: RUF001
    if acronym_shaped(first):
        return False
    return first.casefold() in _NEVER_OPENS_AN_EXPANSION


def _phrase_after(captured: str, acronym: str) -> str:
    """Which words of ``captured`` are the term, rather than the description following it.

    **The mirror image of :func:`_phrase_before`, and deliberately the same idea rather than a
    second mechanism.** That function has no left-hand delimiter to work from and asks the
    acronym which *suffix* spells it; this one has no right-hand delimiter and asks the acronym
    which *prefix* does. Read them together — one rule, applied from each end.

    Shortest wins here as it does there, for the same reason: the first prefix that spells the
    term is where the term stops, and a longer one that also spells it has swallowed prose.
    Candidate prefixes are the description boundaries only, so this can never cut mid-phrase.

    Returns the empty string when no prefix spells the acronym, which the caller reads as "no
    evidence about where this ends" rather than as "no expansion here".
    """
    for boundary in _DESCRIPTION_BOUNDARY_RE.finditer(captured):
        prefix = _usable_expansion(captured[: boundary.start()])
        if prefix and initials_match(acronym, prefix):
            return prefix
    return ""


def core_expansion(acronym: str, expansion: str) -> str:
    """The term written out, with any trailing description removed. Empty if there is none.

    **The boundary decision is conditional on evidence, never applied first and scored
    afterwards**, and that ordering is the whole of this function. Cutting at the first comma and
    then scoring the prefix inverts the two rules that were doing the work: measured on
    ``origin/main``, ``API - when enabled, the process starts automatically`` truncates to the
    two-word ``when enabled``, which is *shorter* and therefore *more* expansion-shaped by every
    length rule in this module. Truncation removes the very thing that was refusing it.

    So a prefix has to earn the cut, and only one signal is strong enough to award it:

    1. **Initials, whole then trimmed.** If the entire right-hand side spells the term it is kept
       entire — ``ATLAS — Automated Transfer Ledger And Scheduler`` has no description to strip.
       Otherwise :func:`_phrase_after` asks the acronym where the term ends, exactly as
       :func:`_phrase_before` asks it where a parenthetical's term begins. ``NOVA — Network
       Operations Visibility Assistant, a service used to correlate operational signals across
       systems`` is thirteen words and refused outright today; its four-word prefix spells
       ``NOVA``, and that agreement between the two strings is what says where the term ends and
       the prose begins. Nothing else in this module knows.

    2. **No initials, no cut.** The right-hand side is then taken whole or not at all, and it is
       refused if it opens with a word no expansion opens with. Keeping it whole is the
       conservative reading: without initials there is no evidence about *where* an expansion
       ends, and a guess would store a phrase the source never wrote. ``HTTP — HyperText Transfer
       Protocol, the protocol of the web`` therefore keeps its description, which is what it did
       before this function existed; :data:`MAX_EXPANSION_WORDS` still bounds it.

    The opening test applies only in the second case, which is also the only case where it can do
    harm — measured at 24 of 26 real definitions kept before the abbreviation exemption and 26 of
    26 after. Initials agreement is a property of two strings and outranks a judgement made from
    one word of one of them.
    """
    whole = _usable_expansion(expansion)
    if whole and initials_match(acronym, whole):
        return whole
    trimmed = _phrase_after(expansion, acronym)
    if trimmed:
        return trimmed
    if whole and not has_a_refused_opening(whole):
        return whole
    return ""


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

    :func:`_phrase_after` is this same question asked from the other end — where a term *stops*
    when a description follows it — and answers it the same way, by asking the acronym. Change
    one and read the other.
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
        expansion = core_expansion(acronym, candidate.expansion)
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
    "core_expansion",
    "detect_entries",
    "detect_in_chunk",
    "has_a_refused_opening",
    "initials_match",
    "initials_of",
    "score_definition",
]
