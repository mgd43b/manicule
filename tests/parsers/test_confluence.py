"""Confluence storage format: what it keeps, and what it must refuse to say.

Two obligations carry the weight here, and neither is about markup fidelity.

**Configuration is not content.** A macro's parameters — a code block's language, a diagram's
engine, a Jira macro's JQL query — are settings, not sentences. Read as generic HTML they became
prose blocks, went into the vector, and were quotable in a citation as words the page had said.
The tests below assert their absence at the top level, inside a table cell and inside a list
item, because the leak came from a flattening helper and a fix applied only at the top level
would look complete while a cell still leaked.

**Content that looks like an instruction stays inert.** Every string in a storage-format body is
untrusted: the body is a wiki page anybody with write access can edit, and an export is a file
anybody can edit before it is ingested. Macro bodies, parameter values and DOT source are all
checked for having stayed *text* — asserted on the rendered document as well as on the blocks,
because "no element was created" is a property of the tree rather than of the extracted string.

The corpus round-trip and the location budget are the last test in the file, as everywhere.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

import pytest
from selectolax.lexbor import LexborHTMLParser

from manicule.chunking import StructuralChunker
from manicule.chunking.sentences import paragraphs
from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import aclose, parsing, read_blocks
from manicule.parsers.config import (
    CONFLUENCE_MEDIA_TYPE,
    CONFLUENCE_MEDIA_TYPES,
    WEB_MEDIA_TYPES,
    ConfluenceConfig,
    WebConfig,
)
from manicule.parsers.confluence import (
    INTERPRETED_MACROS,
    ConfluenceStorageParser,
    dot_parse_warning,
)
from manicule.parsers.plugin import PARSERS
from manicule.parsers.web import WebParser, recover_cdata
from manicule.testing import assert_round_trip, normalise
from tests.corpus.confluence import (
    ACCOUNT_ID,
    CONTAINER_PRESET,
    JQL_QUERY,
    MACRO_DEFINITIONS,
    SCRIPT_PAYLOAD,
    TOPOLOGY_DOT,
)
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

MEDIA_TYPE = CONFLUENCE_MEDIA_TYPE

CORPUS_FIXTURES = (
    "typical.storage",
    "topology.storage",
    "unsupported.storage",
    "hostile.storage",
    "malformed.storage",
    "degenerate.storage",
    "astral.storage",
    "macro-body.storage",
    "empty.storage",
    "handbook-large.storage",
)
"""Every fixture the six assertions run over. Listed rather than globbed, so a generator that
stops writing one fails here instead of quietly shrinking the corpus.

``structure.storage`` is absent, and only from *this* list. It repeats a heading path on
purpose, so the discrimination assertion cannot tell its anchored section from its unanchored
one — the reason it is named in ``test_every_parser.py``'s ``ambiguous``. The shipped contract
still runs over it there, and what the fixture is actually for is covered by
``test_a_repeated_heading_path_is_unlocated_unless_an_anchor_macro_addresses_it``."""

DECLINED_FIXTURES = ("mojibake.storage",)
"""Fixtures this parser must refuse, so that the next parser in the chain gets a turn."""

MIN_CORPUS_BLOCKS = 140
"""A floor under the corpus, because every ratio passes trivially on an empty one."""


def _parser() -> ConfluenceStorageParser:
    return ConfluenceStorageParser(ConfluenceConfig())


async def _blocks(source: str, *, title: str = "") -> list[ParsedBlock]:
    return await read_blocks(_parser(), raw_of(source, MEDIA_TYPE, title=title))


async def _corpus_blocks(corpus: Path, name: str) -> list[ParsedBlock]:
    return await read_blocks(_parser(), raw_from(corpus / "confluence" / name, MEDIA_TYPE))


def _texts(blocks: list[ParsedBlock]) -> str:
    """Everything this document would put in the index, as one string."""
    return "\n".join(block.text for block in blocks)


# --- configuration is not content ----------------------------------------------------------


async def test_a_macro_parameter_is_never_indexed_as_document_text() -> None:
    """The defect this parser exists for, in its simplest form.

    ``<ac:parameter ac:name="language">python</ac:parameter>`` is a setting. Walked as generic
    HTML it is an unknown element wrapping a text node, so it became a one-word prose block
    reading "python" — in the vector, and quotable in a citation as though the page said it.
    """
    blocks = await _blocks(
        '<ac:structured-macro ac:name="code">'
        '<ac:parameter ac:name="language">python</ac:parameter>'
        "<ac:plain-text-body><![CDATA[import os]]></ac:plain-text-body>"
        "</ac:structured-macro>"
    )

    assert [block.kind for block in blocks] == [BlockKind.CODE]
    assert blocks[0].text == "import os"
    assert blocks[0].lang == "python", "the language is kept — as a property, not as prose"
    assert "python" not in _texts(blocks)


async def test_a_jql_query_is_configuration_rather_than_something_the_page_says() -> None:
    """The parameter whose leak was worst, because of what it is.

    An engine name and a language name are noise in the index. A JQL query names a project, a
    status and sometimes a person, and it was being indexed as a sentence the page contained.
    """
    blocks = await _blocks(
        f'<ac:structured-macro ac:name="jira">'
        f'<ac:parameter ac:name="jqlQuery">{JQL_QUERY}</ac:parameter>'
        f"</ac:structured-macro>"
    )

    assert JQL_QUERY not in _texts(blocks)
    assert "ORDERS" not in _texts(blocks)
    placeholder = blocks[0]
    assert placeholder.metadata["parameters"] == ["jqlQuery"], (
        "the name is kept so the omission is auditable"
    )
    assert JQL_QUERY not in str(placeholder.metadata), "the value is not"


async def test_a_parameter_inside_a_table_cell_does_not_reach_the_index(corpus: Path) -> None:
    """Where a fix applied only at the top level would still leak.

    The HTML parser flattens a cell with ``text(deep=True)``, which walks straight through a
    nested macro and picks up its parameters. A table is the most natural place in Confluence to
    put a macro, so this is the case a top-level-only fix would leave open while looking closed.

    **Two independent routes hold this, and disabling either one leaves this test green.** Found
    by trying: ``_inline_parts`` skips ``_CONFIGURATION_ONLY`` elements, *and* it hands a macro to
    ``_inline_macro_text`` rather than recursing into it — so a parameter in a cell is unreachable
    by two separate mechanisms, and only removing both turns this red (on ``'sql' not in
    table.text.split()``). Written down because the consequence is a trap: somebody deleting one
    route would see this pass and reasonably conclude the route they deleted was dead. It was not;
    it was redundant. The property is what is asserted, and the redundancy is why it holds.
    """
    blocks = await _corpus_blocks(corpus, "structure.storage")
    table = next(block for block in blocks if block.kind is BlockKind.TABLE)

    assert "SELECT lag FROM replicas" in table.text, "the cell's real content survives"
    assert "sql" not in table.text.split(), "its language does not"
    assert "jqlQuery" not in table.text
    assert "project = CAPACITY" not in table.text


async def test_a_parameter_inside_a_list_item_does_not_reach_the_index(corpus: Path) -> None:
    """The same leak, one container along, because the flattening is shared."""
    blocks = await _corpus_blocks(corpus, "structure.storage")
    lists = [block for block in blocks if block.kind is BlockKind.LIST]
    joined = "\n".join(block.text for block in lists)

    assert "df -h" in joined, "the command survives"
    assert "bash" not in joined.split(), "the language it was written in does not"


async def test_a_task_state_is_a_property_of_its_task_rather_than_a_block(corpus: Path) -> None:
    """ "complete" was arriving as its own one-word block.

    A word the page never says, in the index, citable — and useless, because a citation reading
    "complete" tells a reader nothing and cannot be acted on.
    """
    blocks = await _corpus_blocks(corpus, "typical.storage")
    tasks = next(block for block in blocks if block.kind is BlockKind.LIST)

    assert tasks.text.splitlines() == [
        "- [x] Announce the rotation in the platform channel",
        "- [ ] Delete the retired key material from the escrow bucket",
        "- [ ] Record the new fingerprint against the quarterly audit",
    ]
    assert tasks.metadata == {"tasks": 3, "complete": 1}
    assert not any(block.text.strip() == "complete" for block in blocks)
    assert not any(block.text.strip() in {"1", "2", "3"} for block in blocks), (
        "nor may a task's database id become a block"
    )


async def test_a_rendered_title_survives_beside_a_body_that_must_stay_verbatim() -> None:
    """A code macro's title is the caption Confluence draws above the block, so it is content.

    It cannot be merged into the block it captions: a code body has to come back character for
    character, and a caption spliced into the first line is no longer the source the page holds.
    So it is a block above the block, which is what Confluence draws.
    """
    blocks = await _blocks(
        '<ac:structured-macro ac:name="code">'
        '<ac:parameter ac:name="language">python</ac:parameter>'
        '<ac:parameter ac:name="title">rotate.py</ac:parameter>'
        "<ac:plain-text-body><![CDATA[import os]]></ac:plain-text-body>"
        "</ac:structured-macro>"
    )

    assert [(block.kind, block.text) for block in blocks] == [
        (BlockKind.PROSE, "rotate.py"),
        (BlockKind.CODE, "import os"),
    ]
    assert blocks[0].metadata["caption"] is True
    assert "python" not in _texts(blocks), "the language is still configuration"


async def test_a_macro_nobody_enumerated_keeps_its_title_out_of_the_index() -> None:
    """The enumeration fails safe, which is the whole reason it is a table.

    ``jira`` renders a title too, but until somebody adds it to :data:`RENDERED_PARAMETERS` the
    parser does not guess. Being wrong this way costs a caption; being wrong the other way puts a
    JQL query in a citation, so the default has to be silence.
    """
    blocks = await _blocks(
        '<ac:structured-macro ac:name="jira">'
        '<ac:parameter ac:name="title">Open orders</ac:parameter>'
        f'<ac:parameter ac:name="jqlQuery">{JQL_QUERY}</ac:parameter>'
        "</ac:structured-macro>"
    )

    assert _texts(blocks) == "[unsupported macro: jira]"
    assert blocks[0].metadata["parameters"] == ["title", "jqlQuery"], (
        "both names recorded, so the omission is auditable"
    )
    assert "Open orders" not in str(blocks[0].metadata), "and neither value is"


async def test_a_panel_title_is_the_one_parameter_that_is_content() -> None:
    """The exception to the rule, and why it is not a slippery slope.

    A panel's title is drawn on the page in the panel's own header. A reader sees it, quotes it
    and searches for it, so it is content that happens to be carried as a parameter. The test is
    whether Confluence shows the value to a reader, not where it sits in the markup.

    **The header is a paragraph of its own, which it was not before.** It used to be joined to
    the body with a single newline, which made "Do not skip the dry run It has taken checkout
    down twice." one paragraph to everything downstream — the header read as the opening words
    of the body. A rendered parameter is a rendered element beside the body, not part of its
    first sentence, so it is separated by the same blank line every other part of a macro's
    text is.
    """
    blocks = await _blocks(
        '<ac:structured-macro ac:name="warning">'
        '<ac:parameter ac:name="title">Do not skip the dry run</ac:parameter>'
        "<ac:rich-text-body><p>It has taken checkout down twice.</p></ac:rich-text-body>"
        "</ac:structured-macro>"
    )

    assert blocks[0].text == "Do not skip the dry run\n\nIt has taken checkout down twice."


# --- content that looks like an instruction --------------------------------------------------


async def test_a_script_in_a_macro_body_stays_text_and_never_becomes_an_element() -> None:
    """The recovery must not become a promotion.

    CDATA exists so a body can hold ``<`` and ``&`` without being markup. Recovering it by
    pasting it back unescaped would turn inert text into a live element — a content-loss bug
    traded for an execution one. Asserted on the rendered tree as well as on the block, because
    "no element was created" is a fact about the document rather than about the string.
    """
    source = (
        f'<ac:structured-macro ac:name="code">'
        f"<ac:plain-text-body><![CDATA[{SCRIPT_PAYLOAD}]]></ac:plain-text-body>"
        f"</ac:structured-macro>"
    )

    blocks = await _blocks(source)
    assert blocks[0].text == SCRIPT_PAYLOAD, "kept exactly, as text"

    rendered = LexborHTMLParser(recover_cdata(source))
    assert not rendered.css("script"), "and no script element was ever created"


async def test_a_script_in_a_parameter_value_stays_inert(corpus: Path) -> None:
    """Parameters come from the same page and the same author as the bodies do.

    The CDATA fix would have promoted ``<![CDATA[<script>…]]>`` to a live element if the
    recovered text had been pasted back unescaped, and a parameter value is the same untrusted
    string arriving through a different door.
    """
    raw = raw_from(corpus / "confluence" / "hostile.storage", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)

    rendered = LexborHTMLParser(recover_cdata(raw.as_text()))
    created = [node.text(deep=True) for node in rendered.css("script")]
    assert created == ["alert('page')"], (
        "exactly one script element exists — the one the fixture writes as a real element. "
        "Every escaped payload and every CDATA body stayed text, so nothing was promoted"
    )

    panel = next(block for block in blocks if block.kind is BlockKind.PANEL)
    assert panel.text.startswith(SCRIPT_PAYLOAD), (
        "a panel title is content, so it is indexed — as the literal characters it is"
    )


async def test_a_raw_script_element_on_a_page_is_dropped_with_its_contents(
    corpus: Path,
) -> None:
    """The other half: a value that is text stays text, and markup that is markup goes.

    A space administrator can put raw HTML on a page, and an export is a file anybody can edit
    before it is ingested. A script body in the index matches queries by accident and cites a
    line no reader ever saw, so it is removed rather than flattened into prose.
    """
    blocks = await _corpus_blocks(corpus, "hostile.storage")

    assert "alert('page')" not in _texts(blocks)


async def test_hostile_dot_source_is_preserved_without_being_evaluated(corpus: Path) -> None:
    """DOT is kept exactly and nothing lays it out.

    A diagram definition is a program in a small language. Preserving it is right; running a
    layout engine over untrusted input to find out whether it is valid is exactly what a parser
    must not do, so validity is judged structurally and recorded rather than executed.
    """
    blocks = await _corpus_blocks(corpus, "hostile.storage")
    dot = next(block for block in blocks if block.lang == "dot")

    assert SCRIPT_PAYLOAD in dot.text, "the source is kept whole, payload and all"
    assert dot.metadata["rendered"] is False
    assert dot.metadata["engine"] == SCRIPT_PAYLOAD, (
        "an engine name is recorded as declared and never dispatched on"
    )


async def test_a_user_mention_is_a_display_reference_and_never_an_account_id(
    corpus: Path,
) -> None:
    """An account id is a directory identifier, not what the page shows.

    Indexing one takes a personal identifier out of the system that governs it and puts it
    somewhere with different rules — searchable, quotable, and exported with the corpus. Where
    the page supplies no display name there is nothing to show, so the mention says only that
    somebody was mentioned.
    """
    blocks = await _corpus_blocks(corpus, "hostile.storage")

    assert ACCOUNT_ID not in _texts(blocks)
    assert "557058" not in _texts(blocks)
    assert not any(ACCOUNT_ID in str(block.metadata) for block in blocks)
    assert any("@user" in block.text for block in blocks), (
        "the mention is still visible as a mention"
    )


async def test_a_named_mention_shows_the_name_a_reader_would_see() -> None:
    blocks = await _blocks(
        '<p>Ask <ac:link><ri:user ri:username="asha.patel"/></ac:link> first.</p>'
    )
    assert blocks[0].text == "Ask @asha.patel first."


async def test_an_attachment_is_referenced_and_never_fetched(corpus: Path) -> None:
    """Attachment ingestion is a separate feature, and parsing must not reach the network."""
    blocks = await _corpus_blocks(corpus, "hostile.storage")
    image = next(
        block
        for block in blocks
        if block.kind is BlockKind.MEDIA and block.metadata.get("attachment")
    )

    assert image.metadata["attachment"] == "throughput.png"
    assert image.text == "Throughput over the window", "the alt text is what it contributes"


async def test_an_event_handler_attribute_contributes_nothing(corpus: Path) -> None:
    """An attribute is never text. Recorded because the fixture carries one deliberately."""
    blocks = await _corpus_blocks(corpus, "hostile.storage")

    assert "onerror" not in _texts(blocks)
    assert "alert('attribute')" not in _texts(blocks)


# --- what storage format means ---------------------------------------------------------------


async def test_a_code_macro_keeps_its_body_and_declares_its_language(corpus: Path) -> None:
    blocks = await _corpus_blocks(corpus, "typical.storage")
    code = next(block for block in blocks if block.kind is BlockKind.CODE)

    assert code.lang == "python"
    assert code.text.startswith("from signing import Keyring")
    assert "keyring.retire(issued.predecessor, grace_hours=24)" in code.text
    assert code.text.splitlines()[1] == "", "blank lines inside a body are part of it"


async def test_a_graphviz_macro_is_preserved_inert_with_its_engine(corpus: Path) -> None:
    blocks = await _corpus_blocks(corpus, "topology.storage")
    diagrams = [block for block in blocks if block.lang == "dot"]

    first = diagrams[0]
    assert first.metadata["engine"] == "neato", "the declared engine, not the default"
    assert first.metadata["rendered"] is False
    assert "parse_warning" not in first.metadata
    assert first.text == TOPOLOGY_DOT, (
        "character for character, not merely containing the arrows. A substring assertion would "
        "pass a parser that reflowed the source onto one line, and in DOT the line structure is "
        "what a person reads"
    )


async def test_a_graphviz_macro_that_declares_no_engine_records_the_default(
    corpus: Path,
) -> None:
    """ "The author did not choose" and "the author chose dot" produce the same diagram."""
    blocks = await _corpus_blocks(corpus, "topology.storage")
    second = [block for block in blocks if block.lang == "dot"][1]

    assert second.metadata["engine"] == "dot"


async def test_invalid_dot_is_retained_with_a_warning_rather_than_dropped(corpus: Path) -> None:
    """A diagram that does not compile is still what somebody wrote.

    It is also the version a person debugging the page most needs to find, so dropping it
    removes exactly the content the search existed to surface.
    """
    blocks = await _corpus_blocks(corpus, "topology.storage")
    broken = [block for block in blocks if block.lang == "dot"][1]

    assert "gamma -> delta;" in broken.text, "retained"
    warning = broken.metadata["parse_warning"]
    assert isinstance(warning, str)
    assert "unclosed" in warning


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("digraph g { a -> b; }", None),
        ("  strict digraph g { a -> b }  ", None),
        ("graph g { a -- b }", None),
        # A record label. Braces inside a string are ordinary Graphviz, not structure, and
        # counting them would put "the diagram body never ends" beside a diagram that compiles.
        ('digraph g { a [shape=record, label="{left|right}"]; }', None),
        ('digraph g { a [label="}"]; b -> c; }', None),
        ('digraph g { a [label="\\"quoted\\""]; }', None),
        ("digraph g { a -> b;", "unclosed"),
        ("digraph g } {", "closing brace"),
        ("subgraph cluster { a }", "does not begin with"),
        ("   ", "empty"),
    ],
)
def test_the_dot_check_reads_structure_without_running_anything(
    source: str, expected: str | None
) -> None:
    """Deliberately shallow, and the shallowness is the design.

    A false warning on valid DOT is noise attached to content that was kept anyway. Invoking a
    real layout engine over untrusted input to be certain is the one thing this must not do.
    """
    warning = dot_parse_warning(source)
    if expected is None:
        assert warning is None
    else:
        assert warning is not None
        assert expected in warning


async def test_a_panel_keeps_the_severity_its_macro_declares(corpus: Path) -> None:
    """Flattened to prose, "Do not skip the dry run" reads as an ordinary sentence."""
    blocks = await _corpus_blocks(corpus, "typical.storage")
    panel = next(block for block in blocks if block.kind is BlockKind.PANEL)

    assert panel.metadata == {"macro": "warning", "severity": "warning"}


async def test_a_panel_nested_in_a_panel_keeps_its_own_severity(corpus: Path) -> None:
    blocks = await _corpus_blocks(corpus, "structure.storage")
    panels = [block for block in blocks if block.kind is BlockKind.PANEL]
    severities = [block.metadata["severity"] for block in panels]

    assert "note" in severities
    assert "warning" in severities, "the inner warning is not absorbed into the outer note"


async def test_a_table_keeps_its_cell_boundaries_and_counts_its_header(corpus: Path) -> None:
    """A table flattened to a run of words loses which value belongs to which column.

    ``rows`` carries the same lines as the text, and it is what lets the chunker split a long
    table at a row boundary instead of as prose — a prose split cuts mid-row, and a severed row
    still looks like ``TERM | expansion`` while its expansion is a fragment.
    """
    blocks = await _corpus_blocks(corpus, "typical.storage")
    table = next(block for block in blocks if block.kind is BlockKind.TABLE)

    rows = [
        "Environment | Key age limit | Owner",
        "Staging | 30 days | Platform",
        "Production | 90 days | Security",
    ]
    assert table.metadata == {"header_rows": 1, "rows": rows}
    assert table.text.splitlines() == rows


async def test_a_page_link_reads_as_its_title_and_an_external_link_as_its_words(
    corpus: Path,
) -> None:
    blocks = await _corpus_blocks(corpus, "typical.storage")
    joined = _texts(blocks)

    assert "open Maintenance calendar and claim the window" in joined
    assert "the published runbook" in joined


async def test_a_link_keeps_its_target_beside_the_text_rather_than_inside_it() -> None:
    """A title is what the sentence says; it is not an address.

    Two spaces can hold pages with the same name, so a reader following a citation needs to know
    which was meant — and a URL spliced into the prose is noise in the vector and nonsense read
    aloud. The reference is kept as data next to the text that mentions it.
    """
    blocks = await _blocks(
        '<p>See <ac:link><ri:page ri:content-title="Maintenance calendar" '
        'ri:space-key="ENG"/></ac:link>, '
        '<a href="https://example.test/runbooks/signing">the runbook</a> and '
        '<ac:link><ri:attachment ri:filename="plan.pdf"/></ac:link>.</p>'
    )

    assert blocks[0].text == "See Maintenance calendar, the runbook and plan.pdf."
    assert blocks[0].metadata["links"] == [
        {"kind": "page", "title": "Maintenance calendar", "space": "ENG"},
        {"kind": "external", "href": "https://example.test/runbooks/signing"},
        {"kind": "attachment", "filename": "plan.pdf"},
    ]


async def test_a_mention_records_that_a_person_was_linked_and_nothing_more() -> None:
    """Redaction that moves an identifier from text to metadata is not redaction.

    The account id is absent from both, which is the only version of this claim worth making.
    """
    blocks = await _blocks(
        f'<p>Asked by <ac:link><ri:user ri:account-id="{ACCOUNT_ID}"/></ac:link>.</p>'
    )

    assert blocks[0].metadata["links"] == [{"kind": "user"}]
    assert ACCOUNT_ID not in str(blocks[0].metadata)
    assert ACCOUNT_ID not in blocks[0].text


async def test_a_link_inside_a_table_cell_is_recorded_too() -> None:
    """The same plumbing, so a reference in a cell is not quietly a lesser reference."""
    blocks = await _blocks(
        "<table><tr><td>Owner</td>"
        '<td><ac:link><ri:page ri:content-title="Platform team"/></ac:link></td></tr></table>'
    )

    assert blocks[0].kind is BlockKind.TABLE
    assert blocks[0].metadata["links"] == [{"kind": "page", "title": "Platform team"}]


async def test_a_link_body_wins_over_the_title_of_what_it_points_at() -> None:
    """The body is what the sentence reads as, so it is what the sentence keeps."""
    blocks = await _blocks(
        "<p>See <ac:link><ri:page ri:content-title="
        '"Quarterly capacity review"/>'
        "<ac:plain-text-link-body><![CDATA[last quarter's review]]></ac:plain-text-link-body>"
        "</ac:link> for the numbers.</p>"
    )

    assert blocks[0].text == "See last quarter's review for the numbers."


# --- unsupported macros ------------------------------------------------------------------------


async def test_an_unsupported_macro_leaves_a_named_placeholder(corpus: Path) -> None:
    """Silence would be indistinguishable from the page having been empty there."""
    blocks = await _corpus_blocks(corpus, "unsupported.storage")
    placeholders = [block for block in blocks if block.metadata.get("unsupported")]
    named = sorted(str(block.metadata["macro"]) for block in placeholders)

    assert named == ["expand", "jira", "roadmap-planner", "toc"]
    assert all(
        block.text == f"[unsupported macro: {block.metadata['macro']}]" for block in placeholders
    )


async def test_an_unsupported_macro_keeps_the_body_it_carries(corpus: Path) -> None:
    """A placeholder must not become an excuse to discard content needing no interpretation.

    An ``expand`` macro's rich text is ordinary prose that happens to sit behind a disclosure
    triangle. The container is unsupported; the paragraph inside it is not.
    """
    blocks = await _corpus_blocks(corpus, "unsupported.storage")
    joined = _texts(blocks)

    assert "the upstream load balancer gives up at one hundred and twenty" in joined
    assert '{"lanes": ["platform", "billing"]}' in joined, (
        "a plain-text body is kept too, as inert code"
    )
    assert "Why the threshold is ninety seconds" in joined, (
        "an expand macro's title is the label Confluence draws on the page, so it is content by "
        "the same test a panel's title is. Being unsupported describes the macro's behaviour and "
        "is never a licence to drop the words it renders"
    )


async def test_a_generated_navigation_macro_has_no_body_to_keep(corpus: Path) -> None:
    """A table of contents is generated at render time; the page does not contain its text."""
    blocks = await _corpus_blocks(corpus, "unsupported.storage")
    toc = next(block for block in blocks if block.metadata.get("macro") == "toc")

    assert toc.text == "[unsupported macro: toc]"
    assert "maxLevel" not in _texts(blocks)
    assert toc.metadata["parameters"] == ["maxLevel"]


async def test_placeholders_can_be_turned_off_and_the_content_still_is_not_invented() -> None:
    """Off is a real choice and a lossy one: only the notice is configurable."""
    source = (
        '<ac:structured-macro ac:name="roadmap-planner">'
        '<ac:parameter ac:name="timeline">quarters</ac:parameter>'
        "</ac:structured-macro>"
    )
    parser = ConfluenceStorageParser(ConfluenceConfig(keep_unsupported_macros=False))
    blocks = await read_blocks(parser, raw_of(source, MEDIA_TYPE))

    assert blocks == []
    assert "quarters" not in _texts(blocks), "and the parameter is still not content"


# --- paragraphs inside a macro body ------------------------------------------------------------


def _macro_body(blocks: list[ParsedBlock]) -> ParsedBlock:
    """The prose block ``macro-body.storage``'s container contributes."""
    return next(
        block
        for block in blocks
        if block.kind is BlockKind.PROSE and block.metadata.get("macro") == "custom-container"
    )


async def test_consecutive_paragraphs_in_a_macro_body_stay_separate_paragraphs(
    corpus: Path,
) -> None:
    """The defect: three ``<p>`` elements became one paragraph on the way out of the macro.

    A single newline is not a paragraph boundary to anything downstream.
    :func:`~manicule.chunking.sentences.paragraphs` splits on blank lines, so a body joined
    with ``\\n`` is one paragraph — and once it is past the chunk budget it is split into
    sentences and repacked with spaces, at which point the boundaries the page had are
    unrecoverable.

    Asserted on ``paragraphs`` rather than on a substring of the text, because the substring
    version passes on a ``\\n`` join too: the point is not that the characters are present but
    that the same splitter the chunker uses agrees there are three of them.
    """
    blocks = await _corpus_blocks(corpus, "macro-body.storage")
    body = _macro_body(blocks)

    assert paragraphs(body.text)[: len(MACRO_DEFINITIONS)] == list(MACRO_DEFINITIONS), (
        "three source paragraphs, in source order, still three after assembly"
    )


async def test_an_inline_br_is_not_a_paragraph_boundary(corpus: Path) -> None:
    """The distinction the rule must not flatten: a line break inside a paragraph is one.

    **``<br>`` contributes nothing at all today, and that is a separate defect left alone
    here.** ``_inline_parts`` yields no text for it and ``_collapse`` would fold a newline into
    a space regardless, so ``a<br/>b`` reads ``ab`` — in this parser and, identically, in the
    web parser it shares the convention with. Making it a genuine line break means changing
    what ``_collapse`` does to every table cell, heading and list item, which is a wider change
    than a paragraph-boundary fix and belongs in its own.

    What is asserted is the property this change is responsible for: the ``<br>`` did not
    *become* a paragraph. The exact joined text is asserted with it so the day ``<br>`` starts
    contributing something, this test says so rather than passing quietly.
    """
    blocks = await _corpus_blocks(corpus, "macro-body.storage")
    written = next(
        part for part in paragraphs(_macro_body(blocks).text) if "line break inside" in part
    )

    assert written == (
        "A line break inside a paragraph is not a paragraph break.The clause after the br "
        "element belongs to the paragraph that opened above it."
    ), "one paragraph, and `<br>` currently contributes no character of its own"


async def test_a_structured_block_inside_a_macro_body_is_still_its_own_block(
    corpus: Path,
) -> None:
    """The paragraph rule applies to prose and must not widen to swallow structure.

    A code macro nested in the container keeps its kind and its language. Merged into the
    prose it would be four lines of YAML in the middle of a paragraph, indexed as sentences
    and quotable as though the page had written it that way.
    """
    blocks = await _corpus_blocks(corpus, "macro-body.storage")
    code = next(block for block in blocks if block.kind is BlockKind.CODE)

    assert code.text == "gateways:\n  - name: primary\n    retries: 3", "verbatim, and its own"
    assert code.lang == "yaml"
    assert "gateways:" not in _macro_body(blocks).text, "and not also inside the prose"


async def test_a_parameter_on_the_container_stays_out_of_the_reassembled_body(
    corpus: Path,
) -> None:
    """The rendered-parameter table is keyed by macro, and the new join must not bypass it.

    ``title`` is content on a ``panel`` and configuration on a macro nobody enumerated. Joining
    the parts differently is no reason for a value to change sides.
    """
    blocks = await _corpus_blocks(corpus, "macro-body.storage")

    assert CONTAINER_PRESET not in _texts(blocks)
    assert "yaml" not in _texts(blocks).split(), "nor the nested macro's language"
    placeholder = next(block for block in blocks if block.metadata.get("unsupported"))
    assert placeholder.metadata["parameters"] == ["title"], "the name, so the omission is audible"
    assert CONTAINER_PRESET not in str(placeholder.metadata)


async def test_a_panel_body_gets_the_same_paragraph_rule_as_an_unsupported_one() -> None:
    """Both assembly paths, because a fix applied to one would look complete.

    A panel is the macro most likely to hold several paragraphs, and it reaches a different
    function from the unsupported case. The severity survives the change, which is the thing a
    panel is kept as a panel for.
    """
    blocks = await _blocks(
        '<ac:structured-macro ac:name="note">'
        "<ac:rich-text-body>"
        "<p>The first observation.</p><p>The second observation.</p>"
        "</ac:rich-text-body></ac:structured-macro>"
    )

    assert blocks[0].kind is BlockKind.PANEL
    assert blocks[0].metadata["severity"] == "note"
    assert paragraphs(blocks[0].text) == ["The first observation.", "The second observation."]


async def test_a_heading_inside_a_macro_is_still_merged_rather_than_emitted() -> None:
    """Which parts are merged is unchanged; only how they are joined.

    Emitting the heading would open a section whose scope is the inside of the macro, and every
    block after the macro would be filed under a heading the page does not have at that level.
    It is still merged — now as a paragraph of its own rather than as the first line of the
    paragraph beneath it.
    """
    blocks = await _blocks(
        "<h1>Page</h1>"
        '<ac:structured-macro ac:name="expand">'
        "<ac:rich-text-body><h3>Inside the macro</h3><p>Under that heading.</p>"
        "</ac:rich-text-body></ac:structured-macro>"
        "<p>After the macro.</p>"
    )
    body = next(block for block in blocks if block.kind is BlockKind.PROSE)
    after = blocks[-1]

    assert paragraphs(body.text) == ["Inside the macro", "Under that heading."]
    assert not any(block.text == "Inside the macro" for block in blocks), "merged, not emitted"
    assert after.heading_path == ("Page",), "the page's hierarchy is untouched by it"


async def test_reassembling_a_macro_body_is_deterministic(corpus: Path) -> None:
    """Two parses of the same bytes are the same blocks.

    Not a formality on this path: the body is assembled from a walk that appends into a list,
    and an assembly that depended on set or dictionary ordering would produce a different
    ``content_hash`` on the second machine and re-embed a corpus for nothing.
    """
    raw = raw_from(corpus / "confluence" / "macro-body.storage", MEDIA_TYPE)

    first = await read_blocks(_parser(), raw)
    second = await read_blocks(_parser(), raw)

    assert first == second


async def test_an_anchor_still_resolves_to_the_text_the_reassembled_body_claims(
    corpus: Path,
) -> None:
    """Requirement three's last clause, asserted directly rather than through the harness.

    The blank lines are added by the *assembly*, and the anchor addresses the source. Resolution
    normalises whitespace, so the added boundaries do not move what the anchor covers — but that
    is the kind of claim that should be run rather than reasoned about.
    """
    raw = raw_from(corpus / "confluence" / "macro-body.storage", MEDIA_TYPE)
    parser = _parser()
    blocks = await read_blocks(parser, raw)
    body = _macro_body(blocks)

    resolved = await parser.resolve(body.anchor, raw)

    assert resolved is not None
    for paragraph in paragraphs(body.text):
        assert normalise(paragraph) in normalise(resolved)


# --- anchors -------------------------------------------------------------------------------


async def test_a_heading_is_addressed_by_the_id_its_page_publishes(corpus: Path) -> None:
    blocks = await _corpus_blocks(corpus, "typical.storage")
    headings = [block.anchor for block in blocks if block.kind is BlockKind.HEADING]

    assert headings[:2] == [
        HeadingAnchor(path=("Rotating a signing key",), fragment="rotating-a-signing-key"),
        HeadingAnchor(
            path=("Rotating a signing key", "Before you begin"), fragment="before-you-begin"
        ),
    ]


async def test_an_anchor_macro_publishes_an_address_without_being_content() -> None:
    """Confluence's own way of naming a section, and it is not a sentence on the page."""
    blocks = await _blocks(
        '<ac:structured-macro ac:name="anchor">'
        '<ac:parameter ac:name="">retries</ac:parameter>'
        "</ac:structured-macro><h2>Retry policy</h2><p>Twice, then give up.</p>"
    )

    assert [block.kind for block in blocks] == [BlockKind.HEADING, BlockKind.PROSE]
    assert blocks[0].anchor == HeadingAnchor(path=("Retry policy",), fragment="retries")
    assert "retries" not in blocks[0].text


async def test_a_repeated_heading_path_is_unlocated_unless_an_anchor_macro_addresses_it(
    corpus: Path,
) -> None:
    """The fixture ``structure.storage`` is excused assertion 3 for exactly this shape.

    Two sections share a heading path. The one an ``anchor`` macro addresses is resolvable; the
    one nothing addresses is honestly unlocatable, because a citation that lands on the wrong
    section of the right page is worse than an admission that there is no address.
    """
    blocks = await _corpus_blocks(corpus, "structure.storage")
    configurations = [
        block
        for block in blocks
        if block.kind is BlockKind.HEADING and block.text == "Configuration"
    ]

    assert len(configurations) == 2
    first, second = configurations
    assert isinstance(first.anchor, Unlocated)
    assert "ambiguous heading path" in first.anchor.reason
    assert second.anchor == HeadingAnchor(
        path=("Capacity review", "Region", "Configuration"),
        fragment="retention-configuration",
    )

    parser = _parser()
    raw = raw_from(corpus / "confluence" / "structure.storage", MEDIA_TYPE)
    resolved = await parser.resolve(second.anchor, raw)
    assert resolved is not None
    assert "retention windows and cold tiering" in resolved
    assert "shard allocator" not in resolved, "and it resolves to its own section, not both"
    assert await parser.resolve(first.anchor, raw) is None


# --- degenerate and hostile input --------------------------------------------------------------


@pytest.mark.parametrize("name", DECLINED_FIXTURES)
async def test_input_that_does_not_decode_is_declined(corpus: Path, name: str) -> None:
    """Declining lets the next parser in the chain try; mojibake in the index cites nothing."""
    raw = raw_from(corpus / "confluence" / name, MEDIA_TYPE)
    with pytest.raises(ParseError, match="not decodable"):
        await read_blocks(_parser(), raw)


async def test_an_empty_body_produces_no_blocks_rather_than_an_empty_one(corpus: Path) -> None:
    assert await _corpus_blocks(corpus, "empty.storage") == []


async def test_macros_with_nothing_in_them_produce_nothing(corpus: Path) -> None:
    """A code macro with no body is not a code block, and an empty task is not a task."""
    blocks = await _corpus_blocks(corpus, "degenerate.storage")

    assert not any(block.kind is BlockKind.CODE for block in blocks)
    assert not any(block.kind is BlockKind.TABLE for block in blocks)
    assert "rust" not in _texts(blocks)
    assert "A macro with no name attribute." not in _texts(blocks)


async def test_broken_storage_xhtml_still_produces_blocks(corpus: Path) -> None:
    """An export truncated mid-write must still be read for what it does contain."""
    report = await check_fixture(_parser(), _raw_of(corpus, "malformed.storage"))

    assert report.blocks >= 3


async def test_a_parameter_outside_any_macro_is_still_not_content(corpus: Path) -> None:
    """The rule holds however malformed the document is, which is why it is checked twice."""
    blocks = await _corpus_blocks(corpus, "malformed.storage")

    assert "A parameter outside any macro at all." not in _texts(blocks)


def _raw_of(corpus: Path, name: str) -> RawDocument:
    return raw_from(corpus / "confluence" / name, MEDIA_TYPE)


# --- calibration: the harness has to bite ------------------------------------------------------


class _NextSectionParser(ConfluenceStorageParser):
    """Anchors every block to the section after the one it came from.

    The shape a real off-by-one takes: every citation resolves, every one lands one section away,
    and nothing raises. If the round-trip harness does not catch this it is not checking anything.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        blocks = [block async for block in super().parse(raw)]
        anchors = [block.anchor for block in blocks]
        for index, block in enumerate(blocks):
            yield block.model_copy(update={"anchor": anchors[(index + 1) % len(anchors)]})


async def test_anchoring_each_block_to_the_next_section_fails_the_round_trip(
    corpus: Path,
) -> None:
    raw = _raw_of(corpus, "typical.storage")
    with pytest.raises(AssertionError):
        await assert_round_trip(_NextSectionParser(ConfluenceConfig()), raw, fixture="shifted")


# --- the stream ----------------------------------------------------------------------------


async def test_a_block_stream_stopped_after_one_block_is_not_left_suspended(
    corpus: Path,
) -> None:
    """A consumer that stops early must not strand the generator."""
    stream = _parser().parse(_raw_of(corpus, "typical.storage"))
    async for _ in stream:
        break
    await aclose(stream)
    assert await _is_closed(stream)


async def test_an_exception_between_blocks_still_closes_the_stream(corpus: Path) -> None:
    """A failing assertion mid-iteration is an early exit like any other, and the common one."""
    seen: list[AsyncIterator[ParsedBlock]] = []
    with pytest.raises(_StopEarlyError):
        await _fail_between_blocks(_raw_of(corpus, "typical.storage"), seen)
    assert await _is_closed(seen[0])


class _StopEarlyError(RuntimeError):
    """Stands in for an assertion failing between two blocks."""


async def _fail_between_blocks(raw: RawDocument, seen: list[AsyncIterator[ParsedBlock]]) -> None:
    async with parsing(_parser(), raw) as blocks:
        seen.append(blocks)
        async for _ in blocks:
            raise _StopEarlyError


async def _is_closed(stream: AsyncIterator[object]) -> bool:
    """Whether a stream is finished with, rather than merely paused."""
    try:
        await anext(stream)
    except StopAsyncIteration:
        return True
    return False


# --- the declaration the connector reads -------------------------------------------------------


@pytest.mark.parametrize("macro", sorted(INTERPRETED_MACROS))
async def test_every_macro_declared_interpreted_is_actually_read(macro: str) -> None:
    """The set the snapshot connector filters its diagnostic by, checked against behaviour.

    Derived from the dispatch tables, so reading it back would prove only that a set equals
    itself. What matters is the claim it makes to another package — "this one is understood" — and
    the only honest check is to hand the parser one of each and see that none of them comes back
    as a placeholder. A macro named here but not really read would make the connector under-report
    a genuine loss, which is the direction that matters.
    """
    blocks = await _blocks(
        f'<ac:structured-macro ac:name="{macro}">'
        '<ac:parameter ac:name="">named</ac:parameter>'
        "<ac:plain-text-body><![CDATA[digraph g { a -> b }]]></ac:plain-text-body>"
        "<ac:rich-text-body><p>Body prose.</p></ac:rich-text-body>"
        "</ac:structured-macro>"
    )

    assert not any(block.metadata.get("unsupported") for block in blocks), (
        f"{macro} is declared interpreted but came back as a placeholder"
    )


async def test_a_macro_outside_the_declaration_is_a_placeholder() -> None:
    """The negative control. A declaration everything satisfies distinguishes nothing."""
    blocks = await _blocks('<ac:structured-macro ac:name="roadmap-planner"/>')

    assert [block.metadata.get("unsupported") for block in blocks] == [True]
    assert "roadmap-planner" not in INTERPRETED_MACROS


# --- the generic HTML parser must not regress --------------------------------------------------


async def test_the_html_parser_still_owns_html_and_is_untouched_by_this_one(corpus: Path) -> None:
    """A new parser for a dialect must not change what the general one does.

    Two ways this could go wrong and neither would raise. The profiled media type could be claimed
    by both parsers, making routing depend on registration order; or storage-format handling could
    have been added to ``web.py`` and changed what an ordinary web page produces. The first is
    asserted against the registry, the second by reading a plain HTML fixture through the HTML
    parser and requiring the blocks it always produced.

    ``web.py`` is deliberately unmodified by this change. That is easy to assert today and easy to
    stop being true, which is why it is a test rather than a sentence in a commit message.
    """
    claiming = {
        registration.name
        for registration in PARSERS
        if CONFLUENCE_MEDIA_TYPE in registration.media_types
    }
    assert claiming == {"confluence"}, "the profiled type routes to exactly one parser"
    assert CONFLUENCE_MEDIA_TYPE not in WEB_MEDIA_TYPES
    assert "text/html" not in CONFLUENCE_MEDIA_TYPES, "and the general type is not claimed here"

    web = WebParser(WebConfig())
    blocks = await read_blocks(web, raw_from(corpus / "web" / "typical.html", "text/html"))
    assert blocks, "the HTML parser still reads ordinary HTML"
    assert all(block.text.strip() for block in blocks)


async def test_the_html_parser_still_recovers_cdata_for_documents_that_are_not_confluence(
    corpus: Path,
) -> None:
    """#90 is not superseded by this parser, and must not be quietly reverted by it.

    A CDATA section in any HTML document is content its author intended. Storage format is no
    longer routed through the HTML parser, so the obvious tidy-up is to decide the recovery was a
    Confluence special case and remove it — it never was, and removing it would silently delete
    content from every other document carrying one.
    """
    recovered = recover_cdata("<p>before</p><![CDATA[kept & <b>escaped</b>]]><p>after</p>")
    blocks = await read_blocks(WebParser(WebConfig()), raw_of(recovered, "text/html"))

    assert any("kept & <b>escaped</b>" in block.text for block in blocks)


# --- the corpus ------------------------------------------------------------------------------


async def test_the_whole_corpus_round_trips_within_its_location_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, every assertion, and the budget that stops the easy way out."""
    raws = [raw_from(corpus / "confluence" / name, MEDIA_TYPE) for name in CORPUS_FIXTURES]
    await check_corpus(_parser(), raws, chunker=chunker, min_blocks=MIN_CORPUS_BLOCKS)
