"""Reading definitions out of text, and refusing prose that looks like one.

Two halves, and the second is the one that decides whether this feature is worth having. Any
regular expression finds ``NOW — Network Operations Workspace``; the question is whether the
same expression fills a glossary with every sentence containing a colon.
"""

from __future__ import annotations

import re

import pytest

from manicule.core.content import Chunk, Document
from manicule.core.glossary import DefinitionForm, normalise_acronym, normalise_expansion
from manicule.ingest import glossary as ingest_glossary
from manicule.ingest.glossary import (
    GLOSSARY_CONTEXT_EVIDENCE,
    INITIALS_EVIDENCE,
    MIN_DEFINITION_CONFIDENCE,
    acronym_shaped,
    core_expansion,
    detect_entries,
    detect_in_chunk,
    has_a_refused_opening,
    initial_skeleton,
    initials_match,
    score_definition,
    term_forms,
)
from manicule.parsers.config import (
    CONFLUENCE_MEDIA_TYPE,
    MARKDOWN_MEDIA_TYPES,
    ConfluenceConfig,
    MarkdownConfig,
)
from manicule.parsers.confluence import ConfluenceStorageParser
from manicule.parsers.markdown import MarkdownParser
from manicule.parsers.web import WebConfig, WebParser
from tests.glossary.corpus import PROSE_ON_THE_GLOSSARY_PAGE
from tests.parsers.support import document_for, make_chunker, raw_of
from tests.storage_helpers import make_chunk, make_document

MARKDOWN_MEDIA_TYPE = min(MARKDOWN_MEDIA_TYPES)
WEB_MEDIA_TYPE = "text/html"
"""One of the Markdown types, chosen deterministically so routing is not a coin toss."""

FULL_WIDTH_SAFER = "\uff33a\uff26e\uff32"
"""``SaFeR`` with its three capitals written as full-width forms.

The shape a CJK-locale exporter emits, and the input ``normalise_acronym`` has always collapsed
on purpose. Written as escapes rather than pasted so the codepoints are legible in a diff and the
linter's ambiguous-character rule can stay on for every other string in this file — a ``noqa``
here would suppress it for the one line where a confusable character is the entire point.
"""

DOCUMENT = make_document(source_id="glossary", title="Glossary of terms")
PLAIN = make_document(source_id="handbook", title="Operations handbook")
EXPANSION = "Network Operations Workspace"

LONG_TABLE: list[tuple[str, str]] = [
    (
        f"{chr(ord('A') + index % 26)}{chr(ord('A') + index // 26 % 26)}X",
        f"{chr(ord('A') + index % 26)}lpha {chr(ord('A') + index // 26 % 26)}eta Exchange",
    )
    for index in range(300)
]
"""Three hundred rows, which is enough to split under any tokenizer this suite uses.

Generated rather than written out because the content is irrelevant and the *length* is the
whole fixture: what matters is that the table cannot fit in one chunk, so the chunker has to
choose a boundary. Every expansion is three words, so a truncation is visible as a shorter one
rather than as a plausible alternative.
"""


def chunk(
    text: str, *, document: Document = DOCUMENT, heading_path: tuple[str, ...] = ("Glossary",)
) -> Chunk:
    return make_chunk(document, 0, text, heading_path=heading_path)


def _rendered_rows(media_type: str) -> list[str]:
    """Every line :data:`LONG_TABLE` legitimately produces, header and delimiter included.

    A chunk may only contain lines from this set. Anything else is a fragment the chunker made
    by cutting somewhere that was not a row boundary.
    """
    if media_type == CONFLUENCE_MEDIA_TYPE:
        return ["Term | Meaning", *(f"{term} | {text}" for term, text in LONG_TABLE)]
    return ["| Term | Meaning |", "|---|---|", *(f"| {t} | {x} |" for t, x in LONG_TABLE)]


# --- the forms the spec lists ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "form"),
    [
        (f"NOW — {EXPANSION}", DefinitionForm.EM_DASH),
        (f"NOW – {EXPANSION}", DefinitionForm.EM_DASH),  # noqa: RUF001 - an en dash, on purpose
        (f"NOW - {EXPANSION}", DefinitionForm.EM_DASH),
        (f"NOW: {EXPANSION}", DefinitionForm.COLON),
        (f"The {EXPANSION} (NOW) holds the runbooks.", DefinitionForm.PARENTHETICAL),
        (f"NOW\n: {EXPANSION}", DefinitionForm.DEFINITION_LIST),
        (f"| NOW | {EXPANSION} |", DefinitionForm.TABLE_ROW),
        (f"### NOW\n\n{EXPANSION}", DefinitionForm.HEADING),
    ],
    ids=[
        "em-dash",
        "en-dash",
        "spaced-hyphen",
        "colon",
        "parenthetical",
        "definition-list",
        "table-row",
        "heading",
    ],
)
def test_every_written_form_the_spec_names_is_detected(text: str, form: DefinitionForm) -> None:
    """``bugs/bug2.md`` §1, one case per bullet.

    Parametrised rather than written as one test with eight assertions, so a form that stops
    working is named in the failure rather than hidden behind whichever assertion ran first.
    """
    entries = detect_in_chunk(chunk(text))

    found = [entry for entry in entries if entry.acronym == "NOW"]
    assert found, f"{form.value} produced no entry from {text!r}"
    assert found[0].expansion == EXPANSION
    assert found[0].form is form
    assert found[0].chunk_id == chunk(text).id, "the entry must cite the chunk it came from"


def test_a_heading_lifted_into_the_breadcrumb_is_still_a_heading_definition() -> None:
    """The route the structural chunker actually takes.

    ``### NOW`` never reaches ``chunk.text`` once headings become the breadcrumb, so a detector
    reading only the text finds nothing on the corpus this feature exists for — while passing
    every test written against raw Markdown.
    """
    entries = detect_in_chunk(make_chunk(DOCUMENT, 0, EXPANSION, heading_path=("Glossary", "NOW")))

    assert [(entry.acronym, entry.form) for entry in entries] == [("NOW", DefinitionForm.HEADING)]


# --- list markers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"- NOW - {EXPANSION}",
        f"* NOW - {EXPANSION}",
        f"+ NOW - {EXPANSION}",
        f"1. NOW - {EXPANSION}",
        f"  - NOW - {EXPANSION}",
        f"- NOW — {EXPANSION}",
        f"- NOW: {EXPANSION}",
    ],
    ids=["bullet", "asterisk", "plus", "ordered", "nested", "em-dash", "colon"],
)
def test_a_definition_written_as_a_list_item_is_detected(text: str) -> None:
    """A glossary written as a bulleted list produced nothing at all.

    Every form anchors its term at the first non-space character, so the marker defeated all
    three line rules at once rather than one of them: measured on ``origin/main``,
    ``'- HDR - Hot Draining Router'`` produced ``[]`` while the identical line without its
    marker produced an entry at 0.95.

    Parametrised over the spellings the parsers actually emit — :mod:`~manicule.parsers.
    confluence`, :mod:`~manicule.parsers.web` and :mod:`~manicule.parsers.adf` write ``- `` and
    ``1. `` with two spaces of indent per level, and Markdown passes through whichever of
    ``-``, ``*`` and ``+`` the author typed.
    """
    entries = detect_in_chunk(chunk(text))

    assert [entry.acronym for entry in entries] == ["NOW"]
    assert entries[0].expansion == EXPANSION, "the marker is not part of what the term means"


def test_a_list_marker_is_not_part_of_a_heading_definitions_expansion() -> None:
    """The other half of the marker rule, and it is a *stored value* rather than a miss.

    A heading section whose body is a bulleted list took the body verbatim, so ``PFC`` was
    recorded as meaning ``'- Partition Failover Coordinator'`` — dash included, in the string a
    reader is shown and a query matches against. Detecting nothing would have been the safer
    failure; this one published a malformed expansion under a correct citation.
    """
    entries = detect_in_chunk(
        make_chunk(
            DOCUMENT, 0, "- Partition Failover Coordinator", heading_path=("Glossary", "PFC")
        )
    )

    assert [entry.expansion for entry in entries] == ["Partition Failover Coordinator"]


def test_a_bare_list_item_is_not_a_definition() -> None:
    """Removing the marker must not turn every list into a glossary.

    The marker rule deletes structure and loosens nothing else, so what refused a bulleted line
    before still refuses it: an item with no separator offers no expansion to find, and
    ``- Note - ...`` is turned away by the two gates the module docstring calls independent.

    **They really are independent here, which is why this asserts absence rather than naming one
    of them.** Measured by removing each in turn: with only :data:`_UPPERCASE_SHARE` dropped to
    0.2 this still passes, because ``this`` is in ``_NEVER_OPENS_AN_EXPANSION``; with only the
    word list emptied it still passes, because ``Note`` is 0.25 upper case. Remove both and the
    line is stored as ``NOTE = 'this is an ordinary remark'`` at exactly 0.60 — which is the
    failure this case exists to produce, and the reason it is worth keeping despite passing
    whether or not the marker rule is present.
    """
    assert detect_in_chunk(chunk(f"- {EXPANSION}")) == []
    assert detect_in_chunk(chunk("- Note - this is an ordinary remark")) == []


# --- table rows --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"NOW | {EXPANSION}",
        f"| NOW | {EXPANSION} |",
        f"NOW | {EXPANSION} | Platform",
        f"| NOW | {EXPANSION} | Platform |",
        f"NOW | {EXPANSION} | Platform | 2026-01-01",
    ],
    ids=["rendered", "markdown", "three-columns", "markdown-three", "four-columns"],
)
def test_a_table_row_is_detected_however_the_parser_spelled_it(text: str) -> None:
    """Outer pipes are optional and the columns after the second are ignored.

    Requiring outer pipes meant this form fired for Markdown and for nothing else: every parser
    that *renders* a table writes ``" | ".join(cells)`` without them, so
    ``_TABLE_RE.match('NOW | Network Operations Workspace')`` was ``None`` on ``origin/main``
    while the hand-written fixture in the one test covering the form matched.

    A third column is metadata about the term — an owner, a status, a review date — and
    ``Term | Meaning | Owner`` is the commonest real glossary layout, which the two-group rule
    had no reading of at all.
    """
    entries = detect_in_chunk(chunk(text))

    assert [entry.acronym for entry in entries] == ["NOW"]
    assert entries[0].expansion == EXPANSION, "the expansion is the second cell, not the row"
    assert entries[0].form is DefinitionForm.TABLE_ROW


@pytest.mark.parametrize(
    ("media_type", "body"),
    [
        (
            CONFLUENCE_MEDIA_TYPE,
            "<table><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{t}</td><td>{x}</td></tr>" for t, x in LONG_TABLE)
            + "</tbody></table>",
        ),
        (
            MARKDOWN_MEDIA_TYPE,
            "| Term | Meaning |\n|---|---|\n" + "".join(f"| {t} | {x} |\n" for t, x in LONG_TABLE),
        ),
    ],
    ids=["confluence", "markdown"],
)
async def test_a_table_too_long_for_one_chunk_stores_no_truncated_expansion(
    media_type: str, body: str
) -> None:
    """**The coupling, and the reason the table rule may not ship on its own.**

    ``_split_table`` divides a table at row boundaries only when the parser describes them in
    ``rows``; without that it splits the table as prose and cuts wherever the token budget
    lands — mid-row, and sometimes mid-cell. A severed row is worse than a lost one, because
    ``WGX | Wlpha `` is still a well-formed ``TERM | expansion`` line whose expansion is a
    fragment, so it is stored as a confident definition carrying a correct citation.

    Measured with the regex widened and ``rows`` absent: 299 entries from 300 rows, one of them
    ``WGX = 'Wlpha'`` against a source row reading ``Wlpha Geta Exchange``, at 0.70. On Markdown
    — where this form already fired before any of this — the same defect was live on
    ``origin/main``: eight line fragments and four truncated expansions.

    Asserted against the source rows rather than by counting, because the count was never the
    problem. Every stored expansion has to be what its row actually said.

    **Row integrity is asserted separately from expansion correctness, and that is not
    belt-and-braces.** Measured while writing this: with ``rows`` removed the Markdown case still
    produced perfect expansions, because Markdown severs at the leading ``|`` and this rule now
    treats outer pipes as optional — so the regex silently repaired the damage and the expansion
    assertion alone guarded nothing on that parser. What is actually broken there is the chunk,
    which no longer holds whole rows, so that is what gets asserted.
    """
    parser = (
        ConfluenceStorageParser(ConfluenceConfig())
        if media_type == CONFLUENCE_MEDIA_TYPE
        else MarkdownParser(MarkdownConfig())
    )
    raw = raw_of(body, media_type, uri="glossary", title="Platform glossary")
    blocks = [block async for block in parser.parse(raw)]
    chunker = make_chunker()
    await chunker.setup()
    chunks = chunker.chunk(document_for(raw, title="Platform glossary"), blocks)

    assert len(chunks) > 1, "the fixture must be long enough to split, or it tests nothing"
    entries = detect_entries(chunks, title="Platform glossary")

    whole = {row for chunk_ in chunks for row in chunk_.text.splitlines() if row.strip()}
    severed = [row for row in whole if row not in set(_rendered_rows(media_type))]
    assert severed == [], "the chunker split the table somewhere other than a row boundary"

    source = dict(LONG_TABLE)
    stored = {entry.acronym: entry.expansion for entry in entries}
    assert {term: stored.get(term) for term in source} == source


def test_a_tables_header_and_rule_rows_are_not_definitions() -> None:
    """What stops a table's own furniture becoming entries.

    The header is refused by the shape gate — ``Term`` is one letter in four upper case — and
    the delimiter by ``_TABLE_RULE_RE``. Both matter now that outer pipes are optional, because
    a rendered header row is exactly the shape of a rendered data row.
    """
    text = f"Term | Meaning\n| --- | --- |\nNOW | {EXPANSION}"

    assert [entry.acronym for entry in detect_in_chunk(chunk(text))] == ["NOW"]


def test_a_term_written_with_dots_normalises_to_the_same_key() -> None:
    entries = detect_in_chunk(chunk(f"N.O.W. — {EXPANSION}"))

    assert [entry.acronym for entry in entries] == ["NOW"]
    assert entries[0].display == "N.O.W.", "the document's own spelling is what gets shown"


def test_a_parenthetical_listing_two_short_forms_records_the_second_as_an_alias() -> None:
    entries = detect_in_chunk(chunk(f"The {EXPANSION} (NOW, NETOPS) is where tickets live."))

    assert entries[0].acronym == "NOW"
    assert entries[0].aliases == ("NETOPS",)
    assert entries[0].keys == ("NOW", "NETOPS")


# --- a definition followed by a description ---------------------------------------------------

DESCRIBED = (
    "NOVA - Network Operations Visibility Assistant, a service used to correlate operational "
    "signals across systems."
)
"""Thirteen words on the right of the dash, of which four are the term.

Refused outright before ``core_expansion`` existed: :data:`MAX_EXPANSION_WORDS` is 10, so the
whole line produced no entry, and query expansion and promotion had nothing to work with.
"""


@pytest.mark.parametrize(
    ("boundary", "text"),
    [
        ("comma", DESCRIBED),
        (
            "semicolon",
            "NOVA — Network Operations Visibility Assistant; it correlates operational signals.",
        ),
        (
            "sentence",
            "NOVA: Network Operations Visibility Assistant. It correlates operational signals "
            "across every connected system.",
        ),
    ],
)
def test_a_description_after_a_boundary_is_dropped(boundary: str, text: str) -> None:
    """The three boundaries the specification names, one case each.

    Each right-hand side is a different length and a different form; what they share is that the
    words before the boundary spell ``NOVA`` and the words after them do not.
    """
    entries = detect_in_chunk(chunk(text))

    assert [entry.expansion for entry in entries] == ["Network Operations Visibility Assistant"], (
        f"the {boundary} boundary did not separate the expansion from the description"
    )


def test_a_description_after_a_sentence_boundary_is_dropped() -> None:
    """The case that used to be asserted as a refusal, now asserted as an extraction.

    Named separately from the parametrised sweep because it is the fixture whose expectation
    this change *inverts*, and a reviewer looking for that should find it by name.
    """
    entries = detect_in_chunk(
        chunk(f"NOW — {EXPANSION}. It replaced three spreadsheets and a mailbox")
    )

    assert [entry.expansion for entry in entries] == [EXPANSION]


def test_the_trimmed_entry_still_cites_the_chunk_that_states_the_whole_line() -> None:
    """Trimming the expansion must not cost the citation.

    The stored expansion is four words and the chunk it cites states all thirteen. What is
    asserted is the *link* — that the entry points at the passage the description is in — because
    that is the part a change here could break. Asserting that the chunk still contains its own
    text would be asserting the fixture; the claim that the reader is shown the whole line is
    made where it can fail, against a passage that has been through storage and retrieval, by
    ``test_the_promoted_passage_still_carries_the_description_it_was_trimmed_of``.
    """
    passage = chunk(DESCRIBED)

    entry = detect_in_chunk(passage)[0]

    assert entry.expansion == "Network Operations Visibility Assistant"
    assert entry.chunk_id == passage.id
    assert entry.expansion in passage.text
    assert entry.expansion != passage.text, "the entry is the term, not the line it came from"


def test_a_stylized_spelling_is_displayed_and_a_normalised_key_is_stored() -> None:
    """Requirement 4, at the point of detection.

    ``ReLAY`` is written with deliberate internal case. What is *shown* is the source's own
    spelling; what is *looked up* is the normalised key, and a reader who types either finds it.
    The same line also carries a description, so the two features are proved to compose rather
    than each being proved on a fixture built for it alone.
    """
    entries = detect_in_chunk(
        chunk("ReLAY — Retention Export Ledger And Yield, the nightly export path.")
    )

    assert [entry.acronym for entry in entries] == ["RELAY"]
    assert entries[0].display == "ReLAY"
    assert entries[0].expansion == "Retention Export Ledger And Yield"


def test_a_right_hand_side_with_no_initials_evidence_is_kept_whole() -> None:
    """The conservative half of the boundary rule, stated as a test rather than left implicit.

    ``central processor`` does not spell ``CPU`` under any reading this module has, so nothing
    here knows where the expansion ends, and guessing at the comma would store a phrase on no
    evidence at all. The whole right-hand side is kept and the length rule still bounds it. This
    is a limitation being pinned down, not a behaviour being praised.

    **This case used to be ``HTTP — HyperText Transfer Protocol, used by every browser``, and it
    was moved rather than deleted.** That line is now cut correctly, because ``HyperText`` is a
    compound whose components spell the two ``T``s — see
    ``test_a_compound_word_supplies_initials_its_whole_word_does_not``, which asserts the new
    behaviour by name. The limitation this test is about is real and still has cases; it just no
    longer has that one.
    """
    entries = detect_in_chunk(chunk("CPU — central processor, the part that executes instructions"))

    assert [entry.expansion for entry in entries] == [
        "central processor, the part that executes instructions"
    ]


# --- definition lists -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "media_type",
    [CONFLUENCE_MEDIA_TYPE, WEB_MEDIA_TYPE],
    ids=["confluence", "web"],
)
async def test_a_definition_list_is_detected_through_the_real_parser(media_type: str) -> None:
    """**A detector rule that was correct, tested, and unreachable for want of an input.**

    ``DefinitionForm.DEFINITION_LIST`` and ``_DEFINITION_MARKER_RE`` have been here since the
    feature shipped, with a unit test feeding them ``'NOW\n: Network Operations Workspace'`` by
    hand. No parser produced that shape: ``<dt>`` and ``<dd>`` both rendered as ``- ``, so the
    two lines were indistinguishable and the form fired on nothing a corpus contains. Measured
    at 0 of 4 on a ``<dl>`` glossary.

    So this asserts through the parser rather than on a hand-written string, which is the whole
    point — the unit test passed throughout and said nothing about whether the form worked.
    Parametrised over both parsers that render ``<dl>``; ADF has no definition-list node type.
    """
    body = (
        "<dl><dt>NOW</dt><dd>Network Operations Workspace</dd>"
        "<dt>QRS</dt><dd>Queue Replay Service</dd></dl>"
    )
    parser = (
        ConfluenceStorageParser(ConfluenceConfig())
        if media_type == CONFLUENCE_MEDIA_TYPE
        else WebParser(WebConfig())
    )
    raw = raw_of(
        body if media_type == CONFLUENCE_MEDIA_TYPE else f"<html><body>{body}</body></html>",
        media_type,
        uri="glossary",
        title="Platform glossary",
    )
    blocks = [block async for block in parser.parse(raw)]
    chunker = make_chunker()
    await chunker.setup()
    chunks = chunker.chunk(document_for(raw, title="Platform glossary"), blocks)

    entries = detect_entries(chunks, title="Platform glossary")

    assert {entry.acronym: entry.expansion for entry in entries} == {
        "NOW": "Network Operations Workspace",
        "QRS": "Queue Replay Service",
    }
    assert {entry.form for entry in entries} == {DefinitionForm.DEFINITION_LIST}


async def test_a_second_definition_under_one_term_is_not_silently_chosen_between() -> None:
    """Two ``<dd>`` under one ``<dt>`` record the first and refuse the second.

    Through the parser rather than from a hand-written string, for the reason this whole change
    exists: a unit test fed the shape it wants proves the detector reads that shape, and says
    nothing about whether a ``<dl>`` produces it. That gap is what let ``DEFINITION_LIST`` pass
    its tests for as long as it did while firing on nothing.

    The line above the second ``: `` is itself a ``: `` line, which is not a term, so the shape
    gate refuses it. That is the conservative outcome and it is asserted rather than left to
    chance: choosing which of two definitions a term has is the judgement this feature is
    forbidden to make.
    """
    raw = raw_of(
        "<dl><dt>NOW</dt><dd>Network Operations Workspace</dd>"
        "<dd>Nightly Operations Watch</dd></dl>",
        CONFLUENCE_MEDIA_TYPE,
        uri="glossary",
        title="Platform glossary",
    )
    parser = ConfluenceStorageParser(ConfluenceConfig())
    blocks = [block async for block in parser.parse(raw)]
    chunker = make_chunker()
    await chunker.setup()
    chunks = chunker.chunk(document_for(raw, title="Platform glossary"), blocks)

    assert blocks[0].text.splitlines() == [
        "NOW",
        ": Network Operations Workspace",
        ": Nightly Operations Watch",
    ]
    entries = detect_entries(chunks, title="Platform glossary")
    assert [(entry.acronym, entry.expansion) for entry in entries] == [
        ("NOW", "Network Operations Workspace")
    ]


# --- brackets ------------------------------------------------------------------------------------

PARENTHETICAL_DESCRIPTION = (
    "Regional Network Edge (e.g., a gateway): Connects a private network to an upstream network."
)
"""Fourteen words, of which three are the term, and a comma buried inside a parenthetical.

The comma is the whole case. It is the *only* boundary the line offered before brackets were
read: ``:`` is not one, and the final ``.`` has no whitespace after it so the sentence
alternative never fires either.
"""


@pytest.mark.parametrize(
    "line",
    [
        f"RNE - {PARENTHETICAL_DESCRIPTION}",
        f"RNE — {PARENTHETICAL_DESCRIPTION}",
        f"RNE: {PARENTHETICAL_DESCRIPTION}",
    ],
    ids=["spaced-hyphen", "em-dash", "colon"],
)
def test_a_comma_inside_a_parenthetical_does_not_end_the_expansion(line: str) -> None:
    """**The regression, and it stored a phrase no source ever wrote.**

    Measured on ``origin/main``: ``RNE`` was recorded as meaning ``'Regional Network Edge (e.g'``.
    The path is worth restating because the last step is the surprising one.

    1. The whole right-hand side is fourteen words, so :data:`MAX_EXPANSION_WORDS` refuses it and
       ``core_expansion`` falls through to the boundary search.
    2. The only boundary found is the comma **inside** ``(e.g., a gateway)``.
    3. The prefix is ``Regional Network Edge (e.g``, which is four words and survives.
    4. ``initials_match('RNE', 'Regional Network Edge (e.g')`` returns **True** — because
       ``initials_of`` keeps only words whose first character is alphabetic, ``(e.g`` begins with
       a bracket and contributes nothing, and the three surviving words spell ``RNE`` exactly.

    So the cut was awarded by an artefact of the initials filter rather than by evidence about
    where the expansion ends. Parametrised over three separators because the fault is in
    ``core_expansion``, which every written form shares — a fix proved on one of them would say
    nothing about the other two.
    """
    entries = detect_in_chunk(chunk(line))

    assert [entry.expansion for entry in entries] == ["Regional Network Edge"]


def test_a_parenthetical_that_is_part_of_the_expansion_is_kept() -> None:
    """The conservative direction: a bracket only *offers* a cut, and offering is not awarding.

    Nothing here is a rule about parentheses. ``core_expansion`` tries the whole right-hand side
    first, and this one is short enough and spells its term, so it is returned entire and no
    boundary is ever consulted. That ordering is what keeps requirement 4 without a second rule
    to balance against requirement 2.
    """
    entries = detect_in_chunk(chunk("RNE - Regional Network Edge (Gateway)"))

    assert [entry.expansion for entry in entries] == ["Regional Network Edge (Gateway)"]


def test_an_unclosed_parenthetical_cannot_produce_an_invented_expansion() -> None:
    """A phrase assembled by cutting inside a bracket is not a phrase the source contains.

    Two independent things stop it, which is why this asserts on both lines. The prefix
    ``Alpha (Beta`` is refused by :func:`brackets_balance` however plausible it reads, and no
    other prefix spells ``XYZ`` — so nothing is stored at all rather than something tidy-looking
    being stored on no evidence.

    The second line is the counterpart that must still work: an unclosed bracket does not
    poison a prefix that was already earned before it, because ``Regional Network Edge`` is
    contiguous in the source and spells the term.
    """
    assert detect_in_chunk(chunk("XYZ - Alpha (Beta, Gamma and several further clauses here")) == []

    kept = detect_in_chunk(chunk("RNE - Regional Network Edge (e.g. a gateway of some kind"))
    assert [entry.expansion for entry in kept] == ["Regional Network Edge"]


def test_a_stored_expansion_never_holds_an_unmatched_bracket() -> None:
    """The invariant, asserted directly rather than through the case that motivated it.

    Worth having on its own terms: an unmatched bracket is the *signature* of truncation inside a
    parenthetical, so this would have caught the defect above without anybody having to work out
    which boundary rule went wrong. It is cheap, it is a property of the stored string, and it
    does not depend on the boundary model staying as it is.
    """
    lines = [
        f"RNE - {PARENTHETICAL_DESCRIPTION}",
        "RNE - Regional Network Edge (Gateway)",
        "RNE - Regional Network Edge (e.g. a gateway of some kind",
        "XYZ - Alpha [Beta, Gamma and several further clauses go here",
        "ATLAS — Automated Transfer Ledger And Scheduler (v2), the nightly export runner",
    ]
    stored = [entry.expansion for line in lines for entry in detect_in_chunk(chunk(line))]

    assert stored, "the fixture must produce entries, or it asserts nothing"
    for expansion in stored:
        assert ingest_glossary.brackets_balance(expansion), (
            f"{expansion!r} holds a bracket it never closes"
        )


def test_a_parenthetical_survives_when_no_prefix_before_it_spells_the_term() -> None:
    """**Requirement 4, and the fixture is chosen so that it can fail.**

    A naive "cut at the first parenthesis" rule stores ``central processor`` here. This one
    stores the whole thing, because nothing authorises the cut: ``central processor`` spells
    ``CP`` and not ``CPU``, so the bracket is offered and refused, and with no boundary earned
    the right-hand side is taken whole exactly as it was before brackets were read at all.

    That is the difference between offering a position and awarding a cut, tested on a line where
    the two rules disagree rather than on one where the candidate is rejected outright and any
    implementation would look correct.

    **Retained for the right reason, which was checked rather than assumed.** There are two ways
    a bracket survives and only one of them is evidence. This is the evidence one: the boundary
    *was* offered at position 18 and the prefix was refused. The other route — the whole
    right-hand side matching because :func:`~manicule.ingest.glossary.initials_of` dropped the
    bracketed tokens, so no boundary was ever consulted — is the accidental one, and it is pinned
    separately by ``test_a_trailing_bracket_survives_only_when_the_whole_side_is_kept``. Measured
    here: ``initials_forms('central processor (the execution unit)')`` is ``{'CPEU'}`` against a
    term of ``CPU``, so no reading spells it and the filter is not what saved the bracket.
    """
    entries = detect_in_chunk(chunk("CPU — central processor (the execution unit)"))

    assert [entry.expansion for entry in entries] == ["central processor (the execution unit)"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("RNE — Regional Network Edge (Type 2)", "Regional Network Edge (Type 2)"),
        ("RNE — Regional Network Edge (Type Two)", "Regional Network Edge"),
        (
            "RNE — Regional Network Edge (Type 2), the branch hardware profile",
            "Regional Network Edge",
        ),
    ],
    ids=["whole-spells-it", "whole-stops-spelling-it", "whole-too-long"],
)
def test_a_trailing_bracket_survives_only_when_the_whole_side_is_kept(
    line: str, expected: str
) -> None:
    """**Three outcomes, one rule, and the ids name the clause rather than the symptom.**

    An earlier version of this test called these cases ``short-keeps-it`` and ``long-drops-it``,
    which was true of the two fixtures it had and false in general — the middle case here is
    short and is cut. Length is a *consequence* of the rule, not the rule:

    1. ``core_expansion`` keeps the **whole** right-hand side when it is usable and its initials
       spell the term, consulting no boundary at all.
    2. Otherwise the shortest prefix that spells the term wins, and a top-level bracket is one
       more position at which that is asked.

    ``(Type 2)`` is kept by clause 1: ``(Type`` and ``2)`` are dropped by
    :func:`~manicule.ingest.glossary.initials_of`'s first-character filter, so the whole still
    spells ``RNE``. ``(Type Two)`` falls to clause 2 because ``Two)`` begins with a letter,
    survives the filter, and makes the whole spell ``RNET``. The third case falls to clause 2 by
    length instead. Same rule, three routes through it.

    **The middle case is also a known limitation**, and it is asserted rather than hidden: two
    lines that mean the same thing are treated differently, decided by the same
    ``word[:1].isalpha()`` filter whose misfiring is the defect this change exists to fix.
    Removing its bad consequence at the boundary did not remove its influence on retention. If
    somebody later makes the two agree, this fails and says so.
    """
    entries = detect_in_chunk(chunk(line))

    assert [entry.expansion for entry in entries] == [expected]


def test_a_quoted_example_marker_yields_no_definition_rather_than_a_truncated_one() -> None:
    """The limitation, pinned as a test so it is a known edge rather than a surprise.

    The same defect with a different delimiter: ``_description_boundaries`` reads brackets and
    not quotes, so the cut is still *offered* at the comma inside ``"e.g., a gateway"``. What
    stops ``'Regional Network Edge "e.g'`` reaching the index is
    :func:`~manicule.ingest.glossary.brackets_balance` refusing an odd number of double quotes,
    and the line then yields nothing at all.

    Nothing is stored, which is the safe direction, and the definition that could have been read
    is lost, which is the cost. Asserted rather than left implicit because a silent limitation is
    the kind that gets rediscovered as a bug.

    The apostrophe cannot join that rule — ``it's`` and ``don't`` would make ordinary English
    unbalanced — so a single-quoted example marker is unsupported for a reason no amount of care
    removes.
    """
    quoted = 'RNE - Regional Network Edge "e.g., a gateway": Connects a private network upstream'

    assert detect_in_chunk(chunk(quoted)) == []


def test_an_apostrophe_does_not_make_an_ordinary_expansion_unbalanced() -> None:
    """The counterpart that keeps the quote rule from costing more than it buys."""
    assert ingest_glossary.brackets_balance("the ops desk's rota")
    assert ingest_glossary.brackets_balance('a "quoted" aside')
    assert not ingest_glossary.brackets_balance('a "quoted aside')


# --- compound words and stylized spellings ------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "line", "expansion"),
    [
        (
            "SORT",
            "SORT — SecOps Reliability Toolkit, a package that operations teams install on a host.",
            "SecOps Reliability Toolkit",
        ),
        (
            "HTTP",
            "HTTP — HyperText Transfer Protocol, used by every browser",
            "HyperText Transfer Protocol",
        ),
    ],
    ids=["secops", "hypertext"],
)
def test_a_compound_word_supplies_initials_its_whole_word_does_not(
    term: str, line: str, expansion: str
) -> None:
    """A camel-cased word contributes one initial per component, which is what awards the cut.

    ``SecOps Reliability Toolkit`` spells ``SRT`` by its words and ``SORT`` by its components, so
    the whole right-hand side is thirteen words with no agreement — no entry at all — until the
    split is read. The boundary is the writer's own capital letter and nothing else: see
    ``test_a_compound_written_without_the_internal_capital_earns_no_cut`` for the same letters
    written flat.

    ``HyperText Transfer Protocol`` is here because it **moved**. It used to be this module's
    example of the conservative fallback — no initials evidence, so keep the whole right-hand side
    including ``used by every browser`` — and under component initials it spells ``HTTP`` exactly
    and the description is trimmed. That is the correct expansion of the term, so this is a
    documented limitation being closed rather than a behaviour changing by accident, and the test
    that pinned the old reading now says so and uses a different term.
    """
    entries = detect_in_chunk(chunk(line))

    assert [(entry.acronym, entry.expansion) for entry in entries] == [(term, expansion)]


def test_a_compound_written_without_the_internal_capital_earns_no_cut() -> None:
    """The negative case that limits the compound rule, and the whole reason it is about capitals.

    ``Secops`` is ``SecOps`` with one letter changed. It is one component, its initials spell
    ``SRT``, and no prefix of this line agrees with ``SORT`` — so nothing awards a cut, the whole
    right-hand side is over the length bound, and there is no entry. A rule that split compounds
    by consulting a list of prefixes would take both spellings and could not tell a reader which
    property of the text it was reading.
    """
    line = "SORT — Secops Reliability Toolkit, a package that operations teams install on a host."
    right = line.partition(" — ")[2]

    assert not initials_match("SORT", "Secops Reliability Toolkit", display="SORT")
    assert core_expansion("SORT", right, display="SORT") == "", "a cut was awarded"
    assert detect_in_chunk(chunk(line)) == []


def test_a_stylized_spelling_supplies_a_skeleton_its_key_does_not() -> None:
    """The second convention, on the ticket's own example.

    ``SaFeR`` looks up as ``SAFER`` — five characters, which ``Service Failure Reporter`` does not
    spell. Its deliberate capitals spell ``SFR``, which the expansion does. Both representations
    survive: the entry is keyed by ``SAFER``, displayed as ``SaFeR``, and the skeleton appears
    nowhere on it.
    """
    line = "SaFeR — Service Failure Reporter, a component that groups related failures together."

    assert initial_skeleton("SaFeR") == "SFR"
    assert not initials_match("SAFER", "Service Failure Reporter")
    assert initials_match("SAFER", "Service Failure Reporter", display="SaFeR")

    entries = detect_in_chunk(chunk(line))

    assert [(entry.acronym, entry.display, entry.expansion) for entry in entries] == [
        ("SAFER", "SaFeR", "Service Failure Reporter")
    ]
    assert entries[0].keys == ("SAFER",), "the skeleton is a comparison form, never a second key"


@pytest.mark.parametrize(
    ("text", "acronym", "display", "expansion"),
    [
        (
            "The Service Failure Reporter (SaFeR) groups related failures.",
            "SAFER",
            "SaFeR",
            "Service Failure Reporter",
        ),
        (
            "The SecOps Reliability Toolkit (SORT) is installed on every host.",
            "SORT",
            "SORT",
            "SecOps Reliability Toolkit",
        ),
    ],
    ids=["skeleton", "components"],
)
def test_a_parenthetical_resolves_its_left_boundary_by_the_same_two_forms(
    text: str, acronym: str, display: str, expansion: str
) -> None:
    """``_phrase_before`` is the same rule from the other end, so it gets the same two forms.

    Written because the change threads the display spelling through *both* boundary functions and
    only one of them had a fixture. The parenthetical form is where it matters most, and not for
    the boundary: ``_phrase_before`` drops the leading article anyway, so the phrase is the same
    either way. What the agreement buys is :data:`INITIALS_EVIDENCE` — a parenthetical is worth
    0.35 and a glossary page 0.15, which is 0.50 against a 0.60 threshold, so **without the
    skeleton this line produces no entry at all**. Measured by removing it: the stylized case goes
    from one entry to zero.
    """
    entries = detect_in_chunk(chunk(text))

    assert [(entry.acronym, entry.display, entry.expansion) for entry in entries] == [
        (acronym, display, expansion)
    ]


@pytest.mark.parametrize(
    ("display", "skeleton"),
    [
        ("SaFeR", "SFR"),
        ("ReCAP", "RCAP"),
        ("AuDiT", "ADT"),
        # All upper case: there is no deliberate mixing to read, and a skeleton equal to the key
        # is not evidence, it is the key written twice.
        ("NOW", ""),
        ("N.O.W.", ""),
        # Below the bound. `MoJo` skeletons to `MJ`, which any two-word phrase satisfies.
        ("MoJo", ""),
        ("WEb", ""),
        # A lower-case initial, refused here as well as by `acronym_shaped`, so that a caller
        # reaching this function directly is not told that `mDNS` is a term spelled `DNS`.
        ("mDNS", ""),
        ("gRPC", ""),
        ("", ""),
    ],
)
def test_a_skeleton_needs_deliberate_mixed_case_above_the_bound(
    display: str, skeleton: str
) -> None:
    """Each condition on :func:`initial_skeleton`, with a case that only that condition refuses."""
    assert initial_skeleton(display) == skeleton


def test_the_bound_is_what_stops_a_short_skeleton_cutting_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:data:`MIN_SKELETON_LENGTH` watched failing, the only way to know it is load-bearing.

    ``WEb`` skeletons to ``WE`` and the ticket's own ``API`` negative has the boundary prefix
    ``when enabled``, whose initials are ``WE``. Lower the bound to two and the cut is awarded and
    ``when enabled`` becomes the recorded meaning of ``WEB`` — the exact failure ``core_expansion``
    warns about, arriving through the new evidence rather than through truncation. At three there
    is no skeleton, no cut, and the opening-word rule refuses the whole right-hand side.

    The mutation is applied to the constant rather than to the function so that what is
    demonstrated is the *bound* doing the work, not an implementation detail of how it is applied.
    """
    line = "WEb - when enabled, the process starts automatically."

    assert detect_in_chunk(chunk(line)) == []

    monkeypatch.setattr(ingest_glossary, "MIN_SKELETON_LENGTH", 2)

    admitted = detect_in_chunk(chunk(line))

    assert [(entry.acronym, entry.expansion) for entry in admitted] == [("WEB", "when enabled")], (
        "the bound is not what refuses this line, so it is not the guard it is named for"
    )


def _free_subsequence_scan(form: str, phrase: str) -> bool:
    """A **deliberately wrong** matcher: does ``form`` appear in ``phrase`` as any subsequence.

    Written out here because requirement 3 cannot be tested the obvious way round. "Arbitrary
    subsequence matching is impossible" asserted as *some unrelated string does not match* proves
    nothing at all — it passes on an implementation that scans freely, because a freely scanning
    matcher also refuses strings whose letters are not there. The assertion has to be built the
    other way: a phrase a free scan **accepts** and the shipped matcher must refuse. This function
    is the free scan, so the fixtures below can be shown to be adversarial rather than asserted to
    be.
    """
    wanted = iter(form.upper())
    letter = next(wanted, None)
    for character in phrase.upper():
        if character == letter:
            letter = next(wanted, None)
    return letter is None


@pytest.mark.parametrize(
    ("term", "form", "phrase", "line"),
    [
        (
            "SORT",
            "SORT",
            "Storage Operations Roster",
            "SORT — Storage Operations Roster, the rota that names who is on call each week.",
        ),
        (
            "SaFeR",
            "SFR",
            "Service for Escalation Routing",
            "SaFeR — Service for Escalation Routing, a queue that pages the duty manager.",
        ),
    ],
    ids=["key-scan", "skeleton-scan"],
)
def test_a_free_subsequence_scan_would_match_and_this_matcher_refuses(
    term: str, form: str, phrase: str, line: str
) -> None:
    """Requirement 3, as a case that separates the two implementations rather than as a claim.

    One fixture per comparison form, because the two would fail differently. ``SORT``'s letters
    appear in order inside ``Storage Operations Roster`` — ``S``, then ``O`` and ``t`` inside
    ``Operations``, then ``R`` — so a scan for the *key* takes it. ``SFR`` appears in order inside
    ``Service for Escalation Routing``, so a scan for the *skeleton* takes that one; its key
    ``SAFER`` does not appear at all, which is why the skeleton needed its own fixture.

    What the shipped matcher does instead is compare complete strings: the phrase's initials under
    four readings of its own token boundaries, against the two spellings of the term, by set
    intersection. There is no position at which it could stop early having found the letters it
    wanted. Both lines run past the length bound, so refusing the cut refuses the entry outright
    and the two outcomes are told apart by more than a difference in stored text.
    """
    assert _free_subsequence_scan(form, phrase), (
        f"{phrase!r} does not contain {form} as a subsequence, so this fixture cannot tell a "
        f"scanning matcher from a boundary-respecting one"
    )

    assert not initials_match(normalise_acronym(term), phrase, display=term)
    assert core_expansion(normalise_acronym(term), phrase, display=term) == phrase, (
        "the phrase alone is short enough to keep whole; the refusal under test is of the cut"
    )
    assert detect_in_chunk(chunk(line)) == []


@pytest.mark.parametrize("text", PROSE_ON_THE_GLOSSARY_PAGE, ids=["note", "today", "api"])
def test_the_widened_matcher_awards_no_cut_inside_the_protected_prose(text: str) -> None:
    """The negatives measured against the **wider** net, which is the claim that needed proving.

    ``test_prose_on_a_glossary_page_is_still_refused`` asserts that no entry is written. This
    asserts the step before it and the one the widening could actually have broken: that no
    *prefix* of these lines is ever selected. Truncation is the dangerous direction — ``when
    enabled`` is shorter than the sentence it came from and therefore more expansion-shaped by
    every length rule in the module — so a widening that admitted a cut here would produce a
    better-looking false positive than the one the module started with.

    Whichever way each line is refused, ``core_expansion`` returns the whole right-hand side or
    nothing. Never a proper prefix of it.
    """
    term, _, right = text.partition(" - ")
    whole = re.sub(r"\s+", " ", right.strip()).rstrip(".").strip()

    core = core_expansion(normalise_acronym(term), right, display=term)

    assert core in {"", whole}, f"a prefix of {right!r} was selected as the expansion of {term}"


# --- refusing prose -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Note: the scheduler restarts nightly and the report lags by one cycle.",
        "Warning - do not edit this file by hand.",
        "Example: a document that has never been indexed.",
        "Tip — rotate the key before it expires.",
    ],
    ids=["note", "warning", "example", "tip"],
)
def test_prose_shaped_like_a_definition_is_refused(text: str) -> None:
    """The gate that does not negotiate.

    Every one of these has exactly the shape of a real definition — a capitalised word, a
    separator, a phrase. What refuses them is that the word is not *written* like an
    abbreviation, and no amount of glossary-looking context can buy that off.
    """
    assert detect_in_chunk(chunk(text)) == []


def test_a_glossary_page_does_not_admit_the_prose_inside_it() -> None:
    """The interaction that makes the two gates worth having separately.

    A page titled "Glossary" gives every line on it the context evidence, so a rule that
    weighed context alone would admit the sentence with a colon in it. The shape gate is what
    stops that, and this is the case where only the shape gate can.
    """
    text = f"NOW — {EXPANSION}\nNote: this list is reviewed each quarter."

    entries = detect_in_chunk(chunk(text), glossary_context=True)

    assert [entry.acronym for entry in entries] == ["NOW"]


@pytest.mark.parametrize(
    "text",
    [
        "NOTE - this paragraph describes an operational consideration, not a term.",
        "Today - the system is operating normally.",
        "API - when enabled, the process starts automatically.",
    ],
    ids=["note", "today", "api"],
)
def test_prose_on_a_glossary_page_is_still_refused(text: str) -> None:
    """The negatives, **on a page that says it is a glossary**, where the margin is exactly zero.

    *Do not move these off a glossary page.* ``chunk()`` puts them on one, and that placement is
    the entire test. Off a glossary page a spaced hyphen scores ``_FORM_WEIGHT[EM_DASH]`` = 0.45
    against a 0.60 threshold, so every line here would be refused by arithmetic that has nothing
    to do with this change and the test would pass whether the change was right or wrong. On a
    glossary page :data:`GLOSSARY_CONTEXT_EVIDENCE` adds the missing 0.15, the total is exactly
    0.60, and the scoring gate admits all three. ``test_the_negatives_are_admitted_by_score``
    below asserts that margin directly, so this docstring cannot quietly go out of date.

    Two of the three were **live false positives** before this change rather than hypothetical
    ones. Measured on ``origin/main``: ``NOTE`` produced the entry ``this paragraph describes an
    operational consideration, not a term`` (nine words) and ``API`` produced ``when enabled, the
    process starts automatically`` (six). Both sit under :data:`MAX_EXPANSION_WORDS`, so no length
    rule ever looked at them. What refuses them now is their first word — see
    ``test_a_real_definition_is_not_refused_for_the_word_it_opens_with`` for the other direction,
    which is the one that constrains how long the word list may get.

    ``Today`` is the weakest of the three and is refused by :func:`acronym_shaped` — one of five
    letters upper — as it was before. It is kept because the specification names it, and it must
    not be read as evidence for the rule the other two exercise.
    """
    assert detect_in_chunk(chunk(text)) == []


REAL_DEFINITIONS_WITHOUT_INITIALS: list[tuple[str, str]] = [
    ("K8S", "Kubernetes"),
    ("CPU", "central processor"),
    ("ID", "identifier"),
    ("DB", "database"),
    ("FAQ", "frequently asked questions list"),
    ("IOU", "I owe you"),
    ("NIC", "network card"),
    ("PSU", "power supply"),
    # The two that were measured being refused. `IT` casefolds to the pronoun `it`, and without
    # the abbreviation exemption in `has_a_refused_opening` both of these were silently lost.
    ("ITSM", "IT service management"),
    ("ITIL", "IT infrastructure library"),
]
"""Ordinary definitions whose initials do **not** spell their terms.

This is the population the word list can hurt, and it is the only one that can constrain how
long the list is allowed to get. Every other fixture in this module is a false positive; a rule
measured only against those can be made perfect by refusing everything.

``HTTP — HyperText Transfer Protocol`` was a member and is not one any more, and the reason is
worth recording rather than editing away. Component initials read ``HyperText`` as two words, so
that pair now *does* spell its term and the case no longer reaches the rule this population
exists to constrain — the parametrised assertion below caught it, which is what that assertion is
for. It moved to ``test_a_compound_word_supplies_initials_its_whole_word_does_not``. ``NIC`` and
``PSU`` replace it so the population does not quietly shrink by one every time a case graduates.
"""


@pytest.mark.parametrize(
    ("first_word", "refused"),
    [
        ("when", True),
        ("this", True),
        ("These", True),
        ("there", True),
        # An abbreviation that casefolds onto a listed pronoun. The shape gate is what tells them
        # apart, and it is the same gate that tells ``NOW`` from ``Note`` on the other side of
        # the dash.
        ("IT", False),
        ("it", True),
        ("It", True),
        ("Network", False),
        ("", False),
    ],
)
def test_an_abbreviation_is_never_mistaken_for_the_pronoun_it_casefolds_onto(
    first_word: str, refused: bool
) -> None:
    """``IT`` and ``it`` are different words that ``casefold`` makes one.

    Asserted at the level of the predicate as well as through ``core_expansion`` because this is
    where the distinction is drawn, and because ``It`` — sentence-initial, one of two letters
    upper — has to stay refused: a capitalised pronoun at the start of prose is still a pronoun.
    """
    assert has_a_refused_opening(f"{first_word} something else".strip()) is refused


@pytest.mark.parametrize(("term", "expansion"), REAL_DEFINITIONS_WITHOUT_INITIALS)
def test_a_real_definition_is_not_refused_for_the_word_it_opens_with(
    term: str, expansion: str
) -> None:
    """Recall, measured as loudly as precision, exactly where the word list is the only judge.

    None of these spells its term, so :func:`core_expansion` cannot fall back on initials and the
    opening rule decides alone. That is the whole population at risk, and it is where a list that
    grew by one plausible-looking word would start quietly deleting definitions.

    ``ITSM`` and ``ITIL`` are here because they were **measured being refused**: ``IT`` casefolds
    to the pronoun ``it``. They are the reason :func:`has_a_refused_opening` asks
    :func:`acronym_shaped` before it consults the list, and they fail if that exemption is
    removed.
    """
    assert not initials_match(term, expansion), (
        f"{expansion!r} spells {term}, so this case never reaches the rule it is meant to test"
    )
    assert core_expansion(term, expansion) == expansion


@pytest.mark.parametrize("ending", [".", "!", "?"])
def test_an_expansion_cut_at_a_sentence_end_keeps_none_of_its_punctuation(ending: str) -> None:
    """All three sentence endings, because the module used to know two lists of them.

    ``_DESCRIPTION_BOUNDARY_RE`` offers a boundary after ``.``, ``!`` and ``?`` alike, so a cut
    can land after any of the three; ``_usable_expansion`` stripped ``.`` alone. The surviving
    ``!`` or ``?`` was then stored as part of the term's meaning and went into query rewrites.

    Parametrised rather than written once with a period, because a period passed before this
    change and would have gone on passing: the case that fails is the ending nobody stripped.
    """
    right = f"Network Operations Visibility Assistant{ending} It correlates operational signals"

    assert core_expansion("NOVA", right) == "Network Operations Visibility Assistant"


@pytest.mark.parametrize("ending", [".", "!", "?"])
def test_two_sentences_are_never_stored_as_one_expansion(ending: str) -> None:
    """Refusing "more than one sentence in it" was true of ``.`` and false of the other two.

    **The term is chosen so that its initials spell the entire right-hand side**, because that is
    the only branch on which a two-sentence string can survive whole. ``core_expansion`` keeps the
    whole side when its initials agree, and ``ABCD`` is what ``Alpha Bravo! Charlie Delta``
    spells — so on ``origin/main`` that string was kept entire and stored as what ``ABCD`` means.
    ``_phrase_after`` could never have produced it: it cuts at the boundary, so a prefix is at
    most the first sentence.

    Asserted here rather than against the private cleaner that does the refusing. The value that
    differs is the *stored expansion*, which is the thing that was wrong; a test at this edge says
    so directly and goes on working if that helper is renamed. The trailing-punctuation strip does
    not cover this case — nothing trails, the sentence ends in the middle — so this is a genuinely
    separate rule from the one pinned above.
    """
    assert core_expansion("ABCD", f"Alpha Bravo{ending} Charlie Delta") == ""


def test_a_stylized_terms_skeleton_is_reachable_from_a_normalised_key() -> None:
    """The skeleton and the key have to be in one normal form, or the pair compares against none.

    ``normalise_acronym`` NFKC-normalises deliberately, so :data:`FULL_WIDTH_SAFER` keys under an
    ASCII ``SAFER``. Reading the display raw skeletoned it to a full-width ``SFR``, which no
    expansion's initials can ever spell — the second comparison form was not wrong so much as
    unreachable, and the detection was lost silently.

    The ASCII spelling is asserted alongside as the control: it passed before this change, and a
    test that only used it would pin nothing.
    """
    assert initial_skeleton("SaFeR") == "SFR", "the ASCII control moved"
    assert initial_skeleton(FULL_WIDTH_SAFER) == "SFR"
    assert initials_match(
        normalise_acronym(FULL_WIDTH_SAFER),
        "Secure Automated Framework Estimating Risk",
        display=FULL_WIDTH_SAFER,
    )


def test_both_comparison_forms_are_normalised_even_when_the_caller_normalised_neither() -> None:
    """``_phrase_before`` passes the raw surface as the key, so ``term_forms`` cannot assume one.

    Two callers disagree about what they hand this: ``detect`` passes a ``normalise_acronym``
    key, ``_phrase_before`` passes the parenthesised surface unchanged. Normalising only inside
    ``initial_skeleton`` would leave the second building a set with one full-width member and one
    ASCII one — the same defect, one call site along. Asserted on the set itself because that is
    where two normal forms can coexist.
    """
    assert set(term_forms(FULL_WIDTH_SAFER, FULL_WIDTH_SAFER)) == {"SAFER", "SFR"}


@pytest.mark.parametrize(
    ("term", "expansion"),
    [
        ("WEN", "When Event Notifier"),
        ("TIS", "These Index Services"),
        ("TPD", "this platform directory"),
    ],
)
def test_initials_evidence_outranks_the_word_the_expansion_opens_with(
    term: str, expansion: str
) -> None:
    """The ordering inside ``core_expansion``, pinned where it can actually fail.

    Each expansion **opens with a listed word** and its initials **spell the term**, which is the
    only combination where the ordering is observable. Reversing it — consulting the list before
    trusting the initials — refuses all three.

    These are constructed rather than found, and deliberately so. An earlier version of this test
    used terms like ``ONCE — Operational Node Configuration Engine`` on the theory that a term
    named after a listed word was at risk. It is not: the list is checked against the
    *expansion's* first word and never against the term, so those cases were not exercising the
    rule at all and passed with the ordering reversed. A mutation is what showed it.
    """
    assert has_a_refused_opening(expansion), "this case does not reach the rule it is pinning"
    assert initials_match(term, expansion)
    assert core_expansion(term, expansion) == expansion


def test_the_negatives_are_admitted_by_score_and_refused_by_shape() -> None:
    """Why the fixtures above have to sit on a glossary page, asserted rather than described.

    If this ever fails, the negatives above have stopped being load-bearing: their refusal would
    have moved into the confidence arithmetic, where it would hold for every dash form on the
    page rather than for prose specifically.
    """
    score = score_definition(
        "API",
        "when enabled, the process starts automatically",
        DefinitionForm.EM_DASH,
        glossary_context=True,
    )

    assert score == pytest.approx(0.45 + GLOSSARY_CONTEXT_EVIDENCE)
    assert score >= MIN_DEFINITION_CONFIDENCE, (
        "the scoring gate admits this line; only the expansion rules refuse it"
    )


@pytest.mark.parametrize(
    "expansion",
    [
        "a workspace where the network operations team keeps every runbook it owns today",
        "a shared place for runbooks. It replaced three spreadsheets",
    ],
    ids=["too-long", "more-than-one-sentence"],
)
def test_an_expansion_that_is_really_a_sentence_is_refused(expansion: str) -> None:
    """Neither reading of the right-hand side spells the term, so neither is taken.

    The second case used to read ``Network Operations Workspace. It replaced three
    spreadsheets`` and asserted a refusal. It is now *admitted*, trimmed to the expansion, and
    that is the required behaviour rather than a regression — a sentence boundary is one of the
    description boundaries a definition is allowed to be followed by, and the initials of the
    first sentence spell ``NOW``. ``test_a_description_after_a_sentence_boundary_is_dropped``
    covers it. What survives here is the case the guard was always really about: prose with
    **nothing** saying where a term ends is refused whole, at every boundary in it.
    """
    assert detect_in_chunk(chunk(f"NOW — {expansion}")) == []


def test_a_term_defined_as_itself_is_not_a_definition() -> None:
    assert detect_in_chunk(chunk("NOW — NOW")) == []


def test_a_table_rule_row_is_not_a_definition() -> None:
    text = "| Term | Meaning |\n|------|---------|\n| NOW | " + EXPANSION + " |"

    assert [entry.acronym for entry in detect_in_chunk(chunk(text))] == ["NOW"]


# --- the confidence, and what it is made of ---------------------------------------------------


def test_initials_are_matched_with_and_without_the_small_words() -> None:
    """``ATLAS`` spells out its ``And``; ``RAM`` skips its ``of``. Nothing says which."""
    assert initials_match("ATLAS", "Automated Transfer Ledger And Scheduler")
    assert initials_match("RAM", "Random Access of Memory")
    assert not initials_match("NOW", "Nightly Operations")


@pytest.mark.parametrize(
    ("surface", "shaped"),
    [
        ("NOW", True),
        ("N.O.W.", True),
        ("IPv6", True),
        ("Note", False),
        ("Warning", False),
        ("A", False),
        # Lower-case-initial abbreviations, which this gate refuses and no value of
        # `_UPPERCASE_SHARE` could admit: all three are above the share and fail on the first
        # character. Asserted because the constant's docstring cited `mDNS` as a *reason* for its
        # value while the function rejected it, and a justification citing a case it refuses is
        # the same defect as a test named wider than its assertion.
        ("mDNS", False),
        ("eSIM", False),
        ("gRPC", False),
    ],
)
def test_only_abbreviation_shaped_terms_pass_the_shape_gate(surface: str, shaped: bool) -> None:
    assert acronym_shaped(surface) is shaped


def test_a_lower_case_initial_abbreviation_is_refused_on_its_first_character() -> None:
    """The half of the shape gate that tuning ``_UPPERCASE_SHARE`` cannot reach.

    ``mDNS`` is 0.75 upper case — comfortably over the 0.6 share — and still refused, because the
    gate also requires an initial capital. Stated as its own case so the *reason* is pinned and
    not just the outcome: a future reader lowering the share to admit these would find this test
    still red, which is the correct answer, since the initial-capital rule is what refuses them.
    """
    letters = [character for character in "mDNS" if character.isalpha()]
    share = sum(1 for character in letters if character.isupper()) / len(letters)

    assert share > 0.6, "the share is not what refuses it"
    assert not acronym_shaped("mDNS")
    assert acronym_shaped("MDNS"), "the same letters with an initial capital are admitted"


def test_the_dash_form_clears_the_threshold_on_its_initials_alone() -> None:
    """Stated as arithmetic, because the threshold was set from these combinations.

    A test asserting only "it is detected" would pass if the threshold moved to zero.
    """
    score = score_definition("NOW", EXPANSION, DefinitionForm.EM_DASH, glossary_context=False)

    assert score >= MIN_DEFINITION_CONFIDENCE
    assert score == pytest.approx(0.45 + INITIALS_EVIDENCE)


def test_a_colon_form_with_neither_piece_of_evidence_does_not_clear_it() -> None:
    score = score_definition(
        "ABC", "Nightly Operations Report", DefinitionForm.COLON, glossary_context=False
    )

    assert score < MIN_DEFINITION_CONFIDENCE


def test_glossary_context_alone_cannot_admit_a_colon_form() -> None:
    """The smallest of the three pieces of evidence, and it must stay too small to decide.

    A page can be a glossary and still contain a sentence with a colon in it, so context has to
    be worth less than the gap between the colon form and the threshold.
    """
    score = score_definition(
        "ABC", "Nightly Operations Report", DefinitionForm.COLON, glossary_context=True
    )

    assert GLOSSARY_CONTEXT_EVIDENCE > 0.0
    assert score < MIN_DEFINITION_CONFIDENCE


def test_a_document_titled_glossary_lifts_a_form_that_carries_nothing_else() -> None:
    """Detection reads the *document's* title, which a chunk does not carry."""
    text = "FATHOM — Failure Analysis Tooling Home"
    on_a_glossary = detect_entries([chunk(text, document=DOCUMENT)], title="Glossary of terms")
    elsewhere = detect_entries(
        [chunk(text, document=PLAIN, heading_path=("Runbook",))], title="Operations handbook"
    )

    assert [entry.acronym for entry in on_a_glossary] == ["FATHOM"]
    assert elsewhere == [], "the same line in a runbook has neither piece of evidence"


def test_a_candidate_line_cannot_supply_the_glossary_evidence_that_admits_it() -> None:
    """Context is read from the page, never from the text being judged.

    ``detect_in_chunk`` used to pass the chunk's own first line to ``_mentions_glossary``, so a
    line containing any of :data:`GLOSSARY_WORDS` certified itself. ``terms`` is one of them, and
    0.45 + 0.15 is exactly :data:`MIN_DEFINITION_CONFIDENCE`, so it decided every case it touched:
    on a document titled "Operations handbook" this chunk recorded **two** entries, and moving the
    same line to second position recorded none.

    The position dependence is what makes it a bug rather than a loose threshold — the two lines
    below are the same two lines in both orders, and nothing about the page differs.
    """
    lines = ["TODO - add the storage terms next quarter.", "XXX - some other note here."]

    first = detect_in_chunk(chunk("\n".join(lines), document=PLAIN, heading_path=("Handbook",)))
    reversed_order = detect_in_chunk(
        chunk("\n".join(reversed(lines)), document=PLAIN, heading_path=("Handbook",))
    )

    assert first == [], "a line naming 'terms' is not evidence that its own page is a glossary"
    assert reversed_order == []


def test_a_heading_path_naming_a_glossary_is_still_evidence() -> None:
    """What the page *may* say about itself, kept: the breadcrumb is not the candidate's text.

    Narrowing context to the title alone would lose a glossary that is one section of a longer
    page, which is an ordinary way to write one.
    """
    entries = detect_in_chunk(
        chunk("FATHOM — Failure Analysis Tooling Home", document=PLAIN, heading_path=("Glossary",))
    )

    assert [entry.acronym for entry in entries] == ["FATHOM"]


# --- normalisation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("surface", "key"),
    [
        ("NOW", "NOW"),
        ("now", "NOW"),
        ("NOW?", "NOW"),
        ("(NOW)", "NOW"),
        ("N.O.W.", "NOW"),
        ("“NOW”", "NOW"),
        ("N", ""),
        ("", ""),
        ("supercalifragilistic", ""),
    ],
)
def test_a_surface_form_normalises_to_one_key(surface: str, key: str) -> None:
    """Query time and ingest time must agree exactly, so there is one function and one test."""
    assert normalise_acronym(surface) == key


def test_two_expansions_are_the_same_only_when_they_say_the_same_thing() -> None:
    """Case and spacing are noise; word order is a different claim.

    Normalising word order away would silently merge two definitions that disagree, which is
    the one thing this feature must never do quietly.
    """
    assert normalise_expansion("Network Operations Workspace") == normalise_expansion(
        "network   operations workspace."
    )
    assert normalise_expansion("Network Operations Workspace") != normalise_expansion(
        "Workspace, Network Operations"
    )
