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

**A term is three strings, and keeping them apart is a correctness property.** The *display* is
what the source wrote — ``SaFeR`` — and it is stored verbatim, because a citation should quote
the document rather than our normalisation. The *lookup key* is
:func:`~manicule.core.glossary.normalise_acronym` of it — ``SAFER`` — and it is the only one of
the three that anything resolves through, at ingest and at query time alike. The *initial
skeleton* is :func:`initial_skeleton` of it — ``SFR`` — and it is a **comparison form**: it exists
so that ``Service Failure Reporter`` can be recognised as the expansion, and it is stored nowhere,
resolves nothing, and breaks no tie. Two definitions of ``SAFER`` disagree whatever their
capitalisation says. The skeleton is computed where it is compared for exactly that reason — a
stored copy is how a comparison form becomes a second key by accident.
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
    from collections.abc import Iterable, Mapping, Sequence

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

**Read from the page and from nothing nearer**, which used not to be true and is the whole of why
this is stated twice. :func:`detect_in_chunk` also passed the chunk's own first line to
:func:`_mentions_glossary`, so a candidate could supply the evidence that carried it: on a
document titled "Operations handbook", the chunk

    TODO - add the storage terms next quarter.
    XXX - some other note here.

recorded **two** entries, because ``terms`` is in :data:`GLOSSARY_WORDS` and the first line is
where it was looked for. Moving that same line to second position recorded none. Evidence a
candidate can write for itself is not evidence, and 0.45 + 0.15 is exactly
:data:`MIN_DEFINITION_CONFIDENCE`, so it was decisive every time it fired.

The title and the heading path are the two things a chunk cannot edit about itself. Dropping the
first line lost nothing measurable: over every fixture in ``tests/glossary`` — both glossary
pages, the forty-five ordinary passages, ``PROSE_ON_THE_GLOSSARY_PAGE``, and the whole labelled
skeleton corpus — the detected set is identical before and after.
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

**Matched positions are filtered by :func:`_description_boundaries` and this pattern is never
scanned directly**, because punctuation inside a bracket belongs to the bracket. See there.
"""

_BRACKETS: Final[Mapping[str, str]] = {"(": ")", "[": "]", "{": "}"}
"""Bracket pairs tracked when reading a right-hand side, opener to closer."""

_CLOSERS: Final[frozenset[str]] = frozenset(_BRACKETS.values())


_PAIRED_QUOTES: Final[frozenset[str]] = frozenset({'"', "“", "”"})
"""Double quotes an expansion must not be cut in half of.

**Counted for evenness rather than matched as a pair**, because the straight ``"`` is the same
character at both ends and the typographic pair is not reliably present on the same line. Evenness
is the property that actually matters here: an odd count means the phrase opens a quotation it
never closes, which is the same signature :func:`brackets_balance` looks for.

**The apostrophe is deliberately absent, and that is the whole reason this is a separate set.**
``'`` is a quote in ``the 'ops desk' rota`` and a letter in ``it's`` and ``don't``, and nothing
here can tell those apart — a rule including it would refuse ordinary English. That is a real gap
rather than an oversight: a single-quoted example marker is an intentionally unsupported form,
recorded in :func:`_description_boundaries`.
"""


def brackets_balance(text: str) -> bool:
    """Whether ``text`` closes every bracket and quotation it opens.

    **A stored expansion with an unmatched bracket is not untidy, it is a signature.** It is what
    truncation inside a parenthetical leaves behind, and it cannot arise from a source phrase that
    was cut at a place its author would recognise. Checking it costs one pass and catches the
    whole family without anybody having to understand which boundary rule went wrong — which is
    the argument for keeping it even though :func:`_description_boundaries` now prevents the case
    that motivated it. Measured: with the boundary model correct and this check removed,
    ``XYZ - Alpha [Beta, Gamma and several further clauses go here`` is still stored as
    ``'Alpha [Beta'``, so the two rules are independent rather than one being the other's
    leftovers.

    **Double quotes are counted here and are not boundary-aware**, which is a deliberate
    asymmetry. ``RNE - Regional Network Edge "e.g., a gateway": …`` is the same defect with a
    different delimiter, and :func:`_description_boundaries` does not read quotes — so the cut is
    still offered at the comma inside them, and what stops ``'Regional Network Edge "e.g'`` being
    stored is this check refusing it. The line yields no definition rather than a wrong one. That
    is the conservative half of the trade and it is named as a limitation rather than presented as
    support; see :data:`_PAIRED_QUOTES` for why the apostrophe cannot join it.
    """
    expected: list[str] = []
    for character in text:
        if character in _BRACKETS:
            expected.append(_BRACKETS[character])
        elif character in _CLOSERS and (not expected or expected.pop() != character):
            return False
    if expected:
        return False
    return sum(text.count(quote) for quote in _PAIRED_QUOTES) % 2 == 0


def _description_boundaries(text: str) -> list[int]:
    """Every position at which a description may begin, brackets respected.

    Two corrections to reading :data:`_DESCRIPTION_BOUNDARY_RE` directly, and the first is what
    this exists for.

    **Punctuation inside a bracket belongs to the bracket.** ``RNE - Regional Network Edge
    (e.g., a gateway): Connects a private network to an upstream network`` offered exactly one
    boundary before this function existed, and it was the comma inside ``e.g.,``. The prefix
    ``Regional Network Edge (e.g`` then *passed* :func:`initials_match`, because
    :func:`initials_of` keeps only words whose first character is alphabetic — ``(e.g`` begins
    with a bracket, contributed nothing, and the three surviving words spelled ``RNE`` exactly.
    So the cut was awarded by an artefact of that filter rather than by evidence, and ``RNE`` was
    stored as meaning ``Regional Network Edge (e.g``.

    **A top-level opening bracket is itself a candidate.** Skipping the nested punctuation alone
    is not enough: it leaves that line with no boundary at all — ``:`` is not one and the final
    ``.`` has no whitespace after it — so the fourteen-word right-hand side stays refused by
    :data:`MAX_EXPANSION_WORDS` and the definition disappears. Failing closed is better than
    storing a fragment, but the phrase before the bracket is the expansion and the source says so.

    **Offering a position is not awarding a cut**, which is the whole of requirement 5 and the
    reason this is not "cut at the first parenthesis". :func:`_phrase_after` still asks the
    acronym whether the prefix spells it, exactly as it does for a comma. ``Regional Network
    Edge`` spells ``RNE`` and earns the cut; a parenthetical that is genuinely part of an
    expansion is reached only when the whole right-hand side already failed, and is kept whenever
    no prefix before it agrees.

    **A qualifier is dropped from a long line and kept from a short one, and that is a decision
    rather than an accident.** It is the sharpest objection to this rule, so it is written down
    rather than left to be discovered:

    - ``Regional Network Edge (Type 2)``
      is stored as ``Regional Network Edge (Type 2)`` — the bracket is kept.
    - ``Regional Network Edge (Type 2), the branch hardware profile``
      is stored as ``Regional Network Edge`` — the bracket is dropped.

    Identical brackets, opposite outcomes, and the discriminator is length. The reason is the rule
    this module already runs everywhere else: **the shortest prefix that spells the term is where
    the term stops.** :func:`_phrase_before` says it in those words and :func:`_phrase_after`
    inherits it. When the whole right-hand side can be taken, it is taken and no boundary is
    consulted — the conservative case, and the one that keeps requirement 4. When it cannot, the
    question "where does the term end" has to be answered, and among prefixes that spell it the
    shortest is the answer. ``Regional Network Edge`` is shorter than ``Regional Network Edge
    (Type 2)`` and both spell ``RNE``, so the shorter wins, exactly as it would if the brackets
    were a comma.

    **The cost, stated so nobody has to rediscover it:** a bracketed qualifier that really is part
    of the term is dropped from a line long enough to need cutting. The alternative is to prefer
    the *longest* agreeing prefix at a bracket, which would invert the shortest-wins rule for this
    one position and would keep ``Regional Network Edge (e.g., a gateway)`` on a short line for the
    same reason. Nothing available here tells a qualifier from an example — that is another verb
    test — so the choice is which way to be wrong, and it is made in favour of the rule that
    already governs every other boundary.

    **No list of example markers.** ``e.g.`` and ``for example`` are what the specification names,
    and a rule keyed on them would be narrower than its own justification: what disqualifies the
    parenthetical here is not the phrase inside it but that the words before it already spell the
    term. Evidence does the work, so the marker list would be decoration that fails on the first
    parenthetical nobody enumerated.

    **Known limitation: whether a trailing parenthetical is kept depends on the first character
    inside it.** Two lines that mean the same thing are treated differently, and the mechanism
    deciding it is the ``word[:1].isalpha()`` filter in :func:`initials_of` — the same filter
    whose misfiring is the defect this function was written to fix. Measured:

    - ``RNE — Regional Network Edge (Type 2)`` keeps its bracket. ``(Type`` starts with ``(`` and
      ``2)`` with a digit, so both are dropped by the filter, the surviving three words spell
      ``RNE``, the *whole* right-hand side matches and no boundary is ever consulted.
    - ``RNE — Regional Network Edge (Type Two)`` loses it. ``Two)`` starts with ``T``, which is
      alphabetic, so the initials read ``RNET``, the whole no longer spells the term,
      :func:`_phrase_after` runs, and the prefix before the bracket earns the cut.

    Removing the bad consequence at the boundary did not remove the filter's influence on
    retention, and this is recorded rather than fixed because the two cases want opposite
    outcomes and nothing available here distinguishes them. Making the filter keep bracketed
    tokens would change what every expansion's initials are, which is a far wider change than
    this one and would want its own measurement.

    """
    depth = 0
    depths: list[int] = []
    openings: list[int] = []
    for index, character in enumerate(text):
        depths.append(depth)
        if character in _BRACKETS:
            if depth == 0:
                openings.append(index)
            depth += 1
        elif character in _CLOSERS:
            depth = max(0, depth - 1)
    nested_free = {
        match.start()
        for match in _DESCRIPTION_BOUNDARY_RE.finditer(text)
        if depths[match.start()] == 0
    }
    return sorted(nested_free | set(openings))


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

_CAMEL_COMPONENT_RE: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
"""Where a compound word's components meet: a lower-case character followed by an upper-case one.

``SORT — SecOps Reliability Toolkit`` is a definition whose ordinary word initials spell ``SRT``
and whose *component* initials spell the term. The boundary is a property of the writer's own
capitalisation, so this is one decomposition rather than a search — ``SecOps`` yields ``Sec`` and
``Ops`` and nothing else can be read out of it.

**Only this boundary, and the omission is the limit of the rule.** ``HTTPServer`` is not split;
that would need a second boundary — upper case followed by upper-then-lower — and every extra
boundary is extra authority to cut a right-hand side in half. One boundary, spelled by the writer
in the plainest way available, is the narrowest rule that reads ``SecOps``.

**Splitting can only lengthen a phrase's initials, which makes it the safer of the two widenings
in this module.** A longer initials string is satisfied by fewer terms, so a phrase that spells a
term by its components spells it on *more* agreement than one that spells it by its words.
:func:`initial_skeleton` runs the other way and needs a bound; this does not.
"""

MIN_SKELETON_LENGTH: Final = 3
"""How many significant characters an :func:`initial_skeleton` needs before it may award a cut.

**The bound on the one widening in this module that runs in the dangerous direction.** A skeleton
is shorter than the key it stands beside — that is what it is for — and a shorter comparison form
is a weaker constraint, because fewer words have to agree before a prefix may call itself the
expansion. Swept over the labelled corpus in ``tests/glossary/skeleton_corpus.py``, 18 positives
and 17 negatives:

=====  ==========  =======  ===================  ==========================================
Bound  Precision   Recall   Boundary precision   What moved
=====  ==========  =======  ===================  ==========================================
2      0.947       1.000    1.000                ``WEB = 'when enabled'``, a false positive
3      1.000       1.000    1.000                —
4      1.000       0.944    0.941                ``AuDiT`` lost, skeleton ``ADT``
=====  ==========  =======  ===================  ==========================================

Three is the only value that scores perfectly, and it is pinned on both sides by one case each.
Below it, ``WEb - when enabled, the process starts automatically`` — the ticket's own ``API``
negative under a stylized term whose skeleton is ``WE`` — has ``when enabled`` cut out of it and
stored as the meaning of ``WEB``. Above it, ``AuDiT — Automated Data Trail`` is lost. The
motivating example sits exactly on the bound as well: ``SaFeR`` skeletons to ``SFR``, three
characters, **zero margin** — the same margin :data:`_UPPERCASE_SHARE` has on the same term's
shape.

**What the corpus does *not* say, recorded because the obvious argument for this constant turns
out to be unsupported.** The intuition is that short comparison forms cut prose more often, so the
prose should show it. Measured over the forty-five ordinary passages in
``tests/glossary/corpus.py`` — every description-boundary prefix, under all four readings of its
initials, counted by length:

===  ====  ====  ====  ====  ====  ====  ====  =====  =====  =====
k    1     3     4     5     6     7     8     9      10     11
===  ====  ====  ====  ====  ====  ====  ====  =====  =====  =====
n    1     1     7     13    18    12    4     5      3      2
===  ====  ====  ====  ====  ====  ====  ====  =====  =====  =====

There is **no k=2 column**: not one of those passages offers a two-word boundary prefix, and the
distribution peaks at six. So ordinary prose is not what condemns a bound of two, and a docstring
claiming it did — an earlier draft of this one did, with numbers nobody had run — would have been
citing a measurement that says the opposite. What condemns it is the constructed line above, which
is in the corpus precisely because the natural population does not contain its own worst case.

**A two-character comparison form is dangerous, and the key can already be one.** ``core_expansion
('WE', 'when enabled, the process starts automatically')`` returns ``when enabled`` on
``origin/main``, before any of this: :data:`~manicule.core.glossary.MIN_ACRONYM_LENGTH` is 2, so a
two-letter *key* has always had this authority. That is not fixed here — it is the merged
design's, it is out of this change's scope, and narrowing it would refuse ``IO``, ``ID`` and
``DB``. It is recorded because it is the reason the bound belongs on the skeleton specifically:
the skeleton is authority a term is granted **in addition** to its key, and additional authority
is the kind worth being strict about.
"""

_TERM = rf"[A-Za-z][\w&/.-]{{0,{MAX_ACRONYM_LENGTH - 1}}}"

# RUF001 on the next line is the whole point of it: an em dash and an en dash are the two
# characters a writer actually reaches for here, and replacing them with a hyphen would leave
# the commonest glossary form in existence undetectable.
_DASH_RE: Final = re.compile(rf"^\s*(?P<term>{_TERM})\s+[—–]\s+(?P<expansion>\S.*?)\s*$")  # noqa: RUF001
_HYPHEN_RE: Final = re.compile(rf"^\s*(?P<term>{_TERM})\s+-\s+(?P<expansion>\S.*?)\s*$")
_COLON_RE: Final = re.compile(rf"^\s*(?P<term>{_TERM})\s*:\s+(?P<expansion>\S.*?)\s*$")
_TABLE_RE: Final = re.compile(
    r"^\s*\|?\s*(?P<term>[^|]+?)\s*\|\s*(?P<expansion>[^|]+?)\s*(?:\|.*)?$"
)
"""The first two cells of a table row, however the parser spelled the row.

**Outer pipes are optional, and requiring them meant this form had never fired outside
Markdown.** A Markdown pipe table reaches a chunk as the author wrote it —
``| NOW | Network Operations Workspace |`` — but every parser that *renders* a table writes
``" | ".join(cells)`` with no outer pipes: :mod:`~manicule.parsers.confluence`,
:mod:`~manicule.parsers.web` and :mod:`~manicule.parsers.adf` alike, and
:mod:`~manicule.parsers.spreadsheet` uses a tab and so is outside this rule entirely. Measured
on ``origin/main``, ``_TABLE_RE.match('NOW | Network Operations Workspace')`` returned ``None``
while the hand-written ``| NOW | ... |`` in the one test covering this form matched — a rule
written to a fixture no renderer produces.

**Only the first two cells, and the rest are ignored rather than refused.** ``Term | Meaning |
Owner`` is the commonest real glossary layout and the two-group rule had no reading of it at
all. The extra columns are metadata about the term — an owner, a status, a review date — not
part of what it means, so the trailing ``(?:\\|.*)?`` consumes them without capturing. The
expansion group is non-greedy, so it stops at the first cell boundary rather than swallowing
the row.

**This rule may only be widened together with row-boundary metadata from the parsers**, and
that ordering is the whole risk of the change rather than a nicety. ``_split_table`` splits a
table at row boundaries only when the block carries ``rows``; without it the chunker splits the
table as prose and cuts mid-row. Measured on a 300-row Confluence table with this regex widened
and ``rows`` still absent: 299 entries detected, of which one was
``WGX = 'Wlpha'`` against a source row reading ``Wlpha Geta Exchange`` — a truncated expansion
at confidence 0.70, above the threshold, carrying a correct citation. Detecting nothing was
strictly better than that, so the parsers emit ``rows`` in the same change.
"""
_DEFINITION_MARKER_RE: Final = re.compile(r"^\s*:\s+(?P<expansion>\S.*?)\s*$")
_HEADING_RE: Final = re.compile(rf"^\s*#{{1,6}}\s+(?P<term>{_TERM})\s*$")
# Likewise the typographic apostrophe: a word processor writes one and a plain editor writes
# the other, and an expansion is not allowed to depend on which produced the page.
_PARENTHETICAL_RE: Final = re.compile(
    rf"(?P<expansion>[A-Za-z][\w'’-]*(?:[ \t]+[\w'’&/-]+){{1,{MAX_EXPANSION_WORDS - 1}}})"  # noqa: RUF001
    r"\s*\(\s*(?P<terms>[^()]{2,64}?)\s*\)"
)
_TABLE_RULE_RE: Final = re.compile(r"^[\s|:-]+$")

_LIST_MARKER_RE: Final = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
"""A list item's bullet or number: structure a parser wrote, not text an author did.

Every parser that renders a list supplies the marker itself — ``- `` for a bullet, ``1. `` for an
ordered item, two spaces of indent per level in :mod:`~manicule.parsers.confluence`,
:mod:`~manicule.parsers.web` and :mod:`~manicule.parsers.adf` alike. The author typed
``NOW - Network Operations Workspace``; the ``- `` in front of it is the parser saying "this was
a list".

**Removing it is what makes a bulleted glossary detectable at all**, and the gap was total rather
than partial. Every form anchors its term at the first non-space character — ``_HYPHEN_RE`` and
its two neighbours all begin ``^\\s*(?P<term>...)`` — and a marker occupies exactly that position,
so a bullet defeated all three at once. Measured on ``origin/main``:

- ``detect_in_chunk('- HDR - Hot Draining Router')`` → ``[]``
- ``detect_in_chunk('HDR - Hot Draining Router')`` → one entry, confidence 0.95

The same held for the colon and em-dash spellings, for ordered lists, and for nested items, which
is why this is one rule applied to the line rather than an alternation added to three patterns.

**Applied to every line, including a code chunk's.** A unified diff's ``- import os`` therefore
reads as ``import os`` here. That is deliberate and bounded: nothing downstream is loosened, so
such a line still has to be written like an abbreviation, still has to survive
:func:`core_expansion`, and still has to clear :data:`MIN_DEFINITION_CONFIDENCE`. Keying this off
:attr:`~manicule.core.content.Chunk.kind` was the alternative and is worse — a chunk holding a
list and the prose around it is not kind ``LIST``, so the fix would silently not apply to the
commonest real layout while looking like it did.
"""


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


def initial_skeleton(display: str) -> str:
    """The significant skeleton of a deliberately mixed-case spelling. Empty when there is none.

    ``SaFeR`` is written with lower-case letters that belong to its *spelling* and not to its
    initials: the writer capitalised ``S``, ``F`` and ``R`` because those are the letters the
    expansion's words begin with, and left ``a`` and ``e`` in lower case because they are
    connective. The skeleton is the upper-case and numeric characters in order — ``SFR``.

    **A comparison form, never a key.** :func:`~manicule.core.glossary.normalise_acronym` owns
    the lookup and keeps it: ``SaFeR`` stores under ``SAFER``, a reader who types ``safer`` finds
    it, and a reader who types ``SFR`` does not, because ``SFR`` is never written anywhere. It is
    also never a tie-breaker — two definitions of ``SAFER`` are two definitions of ``SAFER``
    whatever their internal capitalisation says, which is the ticket's requirement 7 and the
    reason this is a function rather than a field. A stored skeleton is a second copy that can
    disagree with the spelling it came from, and a second copy is how a comparison form becomes a
    key by accident.

    Three conditions, and each refuses a case the others do not:

    1. **An initial capital.** ``mDNS`` skeletons to nothing here, as it is refused by
       :func:`acronym_shaped` there. The rule is the same one twice on purpose: a term whose own
       first letter is lower case is outside what this module detects, and a skeleton that
       silently began at the second character would be reading a term the source did not write.
    2. **At least one lower-case letter.** Without one there is no deliberate mixing to read, and
       the skeleton would simply restate the key — ``NOW`` skeletons to ``NOW``. That is not
       wrong, it is nothing, and returning it would put a second copy of the key into every
       comparison for no gain.
    3. **At least :data:`MIN_SKELETON_LENGTH` characters.** The whole of the bound; see there.

    Numeric characters are kept because the ticket names them, and the consequence is stated
    rather than papered over: :func:`initials_of` takes a word's first character only when it is
    alphabetic, so ``IPv6`` skeletons to ``IP6`` and no expansion's initials can ever spell it.
    Retaining the digit makes such a skeleton unmatchable, which is the conservative outcome —
    dropping it would leave ``IP``, two characters, and the bound would refuse that anyway.
    """
    if not display[:1].isupper():
        return ""
    if not any(character.islower() for character in display):
        return ""
    skeleton = "".join(
        character for character in display if character.isupper() or character.isdigit()
    )
    if len(skeleton) < MIN_SKELETON_LENGTH:
        return ""
    return skeleton


def term_forms(acronym: str, display: str = "") -> frozenset[str]:
    """Every spelling of a term that an expansion's initials are allowed to spell.

    At most two: the normalised key, and the :func:`initial_skeleton` of the source's own
    spelling when there is one. Both are computed from the term alone.

    **Nothing here reads the expansion**, and that is half of why requirement 3 holds. The set of
    strings a phrase is permitted to spell is fixed before any phrase is looked at, so no phrase
    can widen it by containing the right letters.
    """
    forms = {"".join(character for character in acronym if character.isalnum()).upper()}
    if display:
        forms.add(initial_skeleton(display))
    return frozenset(forms) - {""}


def initials_of(
    expansion: str, *, skip_function_words: bool, split_components: bool = False
) -> str:
    """The letters an expansion's words begin with, upper case.

    ``split_components`` reads a compound word's parts instead of the whole word, at the
    :data:`_CAMEL_COMPONENT_RE` boundary — ``SecOps Reliability Toolkit`` gives ``SORT`` rather
    than ``SRT``. Function words are dropped before the split rather than after it: a writer
    skips a small word, not half of a compound, and dropping ``And`` from ``AndOps`` would be a
    rule about letters rather than about words.
    """
    words = [word for word in re.split(r"[\s/-]+", expansion) if word]
    if skip_function_words:
        words = [word for word in words if word.casefold() not in _FUNCTION_WORDS]
    if split_components:
        words = [part for word in words for part in _CAMEL_COMPONENT_RE.split(word) if part]
    return "".join(word[0] for word in words if word[:1].isalpha()).upper()


def initials_forms(expansion: str) -> frozenset[str]:
    """Every reading of an expansion's significant initials. Four at most.

    Two independent choices, neither knowable from the string: whether the writer spelled the
    function words out — ``ATLAS`` includes its ``And``, ``RAM`` skips its ``of`` — and whether a
    compound word contributes one initial or one per component.

    **Every reading is produced from token boundaries, before any term is consulted.** The
    boundaries are whitespace, ``/``, ``-`` and the camel-case boundary, all of them written into
    the text by whoever wrote it. This function cannot be steered by the term it is about to be
    compared against, because it is never told what that term is.
    """
    return frozenset(
        initials_of(expansion, skip_function_words=skip, split_components=split)
        for skip in (False, True)
        for split in (False, True)
    ) - {""}


def initials_match(acronym: str, expansion: str, *, display: str = "") -> bool:
    """Whether ``expansion``'s initials spell ``acronym``.

    ``display`` is the source's own spelling of the term, which admits its
    :func:`initial_skeleton` as a second form the initials may spell. Omitting it is the narrow
    reading — the key alone — and every caller that has a display spelling to hand passes it.

    **Set intersection over whole strings, and that is requirement 3's guarantee in one line.**
    Two closed sets are built independently — :func:`term_forms` from the term,
    :func:`initials_forms` from the expansion's token boundaries — and the test is whether they
    share a member. There is no containment test, no scan, and no position at which the
    comparison could stop early having found the letters it wanted: a phrase either spells the
    term exactly, under one of four readings of its own boundaries, or it does not. An
    implementation that walked the expansion looking for the term's letters would pass every
    positive fixture in this suite, which is why
    ``test_a_free_subsequence_scan_would_match_and_this_matcher_refuses`` builds the counterpart
    that separates them rather than asserting that some unrelated string fails to match.
    """
    return bool(term_forms(acronym, display) & initials_forms(expansion))


def _mentions_glossary(*texts: Iterable[str]) -> bool:
    for group in texts:
        for text in group:
            if any(word in GLOSSARY_WORDS for word in re.findall(r"[a-z]+", text.casefold())):
                return True
    return False


def _usable_expansion(expansion: str) -> str:
    """``expansion`` cleaned up, or the empty string if it is not one.

    Rejects the three shapes that look like definitions and are not: something with more than one
    sentence in it, something long enough to be a description, and something holding a bracket it
    never closes. The first two are prose that happens to sit after a separator; the third is a
    phrase no source wrote, assembled by cutting inside a parenthetical — see
    :func:`brackets_balance`.
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
    if not brackets_balance(trimmed):
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


def _phrase_after(captured: str, acronym: str, *, display: str = "") -> str:
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
    for boundary in _description_boundaries(captured):
        prefix = _usable_expansion(captured[:boundary])
        if prefix and initials_match(acronym, prefix, display=display):
            return prefix
    return ""


def core_expansion(acronym: str, expansion: str, *, display: str = "") -> str:
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

    **``display`` widens what counts as initials agreement, and therefore widens the authority to
    cut.** That is the whole risk of passing it, and it is why the two forms it admits are bounded
    where they are: :data:`_CAMEL_COMPONENT_RE` reads one boundary and can only lengthen a
    phrase's initials, and :func:`initial_skeleton` is refused below
    :data:`MIN_SKELETON_LENGTH`. Neither adds a way to *find* letters in a phrase; both add a way
    for a phrase's existing token boundaries to spell a term the source wrote down.
    """
    whole = _usable_expansion(expansion)
    if whole and initials_match(acronym, whole, display=display):
        return whole
    trimmed = _phrase_after(expansion, acronym, display=display)
    if trimmed:
        return trimmed
    if whole and not has_a_refused_opening(whole):
        return whole
    return ""


def a_heading_may_define(acronym: str, expansion: str, *, display: str = "") -> bool:
    """Whether a heading and the text beneath it may be read as a definition at all.

    **The one form whose separator nobody typed.** Every other written form carries a mark the
    author chose to mean "is defined as" — a dash, a colon, a definition list, a table cell
    boundary, a pair of parentheses. A heading followed by a paragraph carries none: it is the
    shape of *every structured document ever written*, so the relationship is inferred here and
    asserted nowhere in the source. Pairing the weakest possible form with a page-level title
    check meant that on any page whose title said "glossary", **every section became a
    definition**: ``_FORM_WEIGHT[HEADING]`` 0.45 plus :data:`GLOSSARY_CONTEXT_EVIDENCE` 0.15 is
    exactly :data:`MIN_DEFINITION_CONFIDENCE`, cleared by being equal. Measured on ten heading
    sections whose bodies were ordinary sentences, ten of ten were stored, including ``NOW`` as
    ``'The Network Operations Workspace holds the runbooks'``.

    So the two strings have to agree. Initials agreement is a property of the term and the
    phrase rather than of the page around them, which is what the page title could never supply.

    **The harm this prevents is not the false statement, it is the feature disabling itself.**
    A wrong entry is not merely a bad answer sitting in an index: two disagreeing expansions of
    one term are an
    :class:`~manicule.core.glossary.ExpansionConflict`, and
    :func:`~manicule.retrieval.expansion.resolve_expansion` correctly refuses to choose between
    them. So one heading section naming an acronym takes a **correct** 0.95 definition on another
    page from rank 1 to absent from the context, with ``explicit_definition`` false and no match
    at all — and reports a conflict, which is a legitimate state, so nothing looks broken. A
    corpus holding a real glossary and ordinary heading-structured documentation degrades term by
    term as those pages accumulate.

    **What this costs, both halves, because only the pair is honest.** Against
    ``REAL_DEFINITIONS_WITHOUT_INITIALS`` — the curated population of ordinary definitions whose
    initials do not spell them, ``CPU``/``central processor``, ``ITSM``/``IT service
    management`` — the answer depends entirely on the form the content is authored in:

    - **As the repository actually writes them**, ``TERM — expansion``, this costs **nothing**:
      10 of 10 detected and 10 of 10 kept, because a dash is a separator its author typed. The
      six near-miss backronyms on the canonical glossary page are likewise untouched, and so is
      every other entry in that corpus.
    - **Authored as heading sections**, the same ten are all refused: 10 of 10 lost.

    Read only the second and this rule looks expensive; read only the first and it looks free.
    Before relaxing it, note that the population it refuses and the population it exists to stop
    are the same shape — a heading, a body, and no agreement between them — and nothing available
    here tells them apart. ``tests/glossary/test_detection.py`` records the search for a cheaper
    discriminator: word count is refuted (20 of 49 real expansions are exactly six words), and
    "contains a determiner" scores perfectly on the corpus while missing 4 of 10 of the real
    failure class and refusing the ``National Association for **the** Advancement…`` class the
    corpus happens to contain none of.
    """
    return initials_match(acronym, expansion, display=display)


def score_definition(
    acronym: str,
    expansion: str,
    form: DefinitionForm,
    *,
    glossary_context: bool,
    display: str = "",
) -> float:
    """How strongly this reads as a definition rather than as prose.

    Additive and capped, with each term's justification on its own constant. Not a probability
    of anything: it answers "is this text a definition", never "is this definition right".

    ``display`` is passed for the same reason :func:`core_expansion` takes it, and passing it in
    one place and not the other would be the incoherent version: the boundary decision would cut
    ``SecOps Reliability Toolkit`` out of a longer line on the strength of initials agreement and
    the score would then report that no initials agreement was found. One notion of agreement,
    read once, used by both.
    """
    score = _FORM_WEIGHT[form]
    if initials_match(acronym, expansion, display=display):
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


def _phrase_before(captured: str, term: str) -> str:
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

    ``term`` is the surface the parentheses held rather than a normalised key, and it is used as
    both spellings :func:`initials_match` accepts. Renamed from ``acronym`` when the second
    spelling arrived: it had always been the display form, and calling it the key made the call
    below look like a mistake.
    """
    words = captured.split()
    for length in range(1, len(words) + 1):
        candidate = " ".join(words[len(words) - length :])
        if initials_match(term, candidate, display=term):
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
    lines = [_LIST_MARKER_RE.sub("", line) for line in chunk.text.splitlines()]
    context = glossary_context or _mentions_glossary(chunk.heading_path)
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
        expansion = core_expansion(acronym, candidate.expansion, display=candidate.display)
        if not acronym or not expansion or normalise_acronym(expansion) == acronym:
            continue
        if candidate.form is DefinitionForm.HEADING and not a_heading_may_define(
            acronym, expansion, display=candidate.display
        ):
            continue
        confidence = score_definition(
            acronym,
            expansion,
            candidate.form,
            glossary_context=context,
            display=candidate.display,
        )
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
    "MIN_SKELETON_LENGTH",
    "a_heading_may_define",
    "acronym_shaped",
    "brackets_balance",
    "core_expansion",
    "detect_entries",
    "detect_in_chunk",
    "has_a_refused_opening",
    "initial_skeleton",
    "initials_forms",
    "initials_match",
    "initials_of",
    "score_definition",
    "term_forms",
]
