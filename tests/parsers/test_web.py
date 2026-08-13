"""HTML parsing, and the deep link that only exists where the author made one.

Everything here turns on one asymmetry. A heading's *path* comes from the markup and is
always available; a heading's *address* comes from an ``id=`` the page publishes, and often
does not exist. Synthesising the missing one would be easy and would produce a citation that
looks precise and lands at the top of a page manicule does not serve — so the fragment is
``None``, and where that leaves two sections indistinguishable the blocks are ``Unlocated``.

:func:`test_synthesising_a_fragment_the_page_never_published_fails_the_round_trip` and
:func:`test_inventing_an_anchor_for_an_ambiguous_path_fails_the_round_trip` are what make
those two rules load-bearing rather than stated: each runs the implementation that takes the
easy option and requires the harness to reject it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, read_blocks
from manicule.parsers.base import slugify
from manicule.parsers.web import WebConfig, WebParser, recover_cdata
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

MEDIA_TYPE = "text/html"

CORPUS_FIXTURES = (
    "typical.html",
    "structure.html",
    "heading-only.html",
    "no-trailing-newline.htm",
    "empty.html",
    "astral.html",
    "unclosed.html",
    "manual-large.html",
)
"""Every fixture this parser is expected to parse. Listed rather than globbed, so a
generator that stops writing one fails here instead of quietly shrinking the corpus."""

MIN_CORPUS_BLOCKS = 140
"""A floor under the corpus, because every ratio passes trivially on an empty one."""

AMBIGUOUS = """<html><head><title>Repeats</title></head><body>
<h2>Overview</h2>
<p>The first section, which the page never gave an address.</p>
<h2>Overview</h2>
<p>The second, which the path alone cannot tell from the first.</p>
</body></html>"""


def _parser() -> WebParser:
    return WebParser(WebConfig())


async def _blocks(html: str, *, title: str = "") -> list[ParsedBlock]:
    return await read_blocks(_parser(), raw_of(html, MEDIA_TYPE, title=title))


class _NextSectionParser(WebParser):
    """Anchors every block to the section after the one it came from.

    The defect the round-trip contract exists to catch: nothing raises, every anchor is
    well-formed, and each citation quotes the section below the one it names.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        blocks = [block async for block in super().parse(raw)]
        anchors = [block.anchor for block in blocks]
        for index, block in enumerate(blocks):
            yield block.model_copy(update={"anchor": anchors[(index + 1) % len(anchors)]})


class _SluggingParser(WebParser):
    """Invents a fragment for every heading, whether or not the page publishes one.

    The tempting shortcut: it makes each anchor look complete, and it deep-links to nothing,
    because the slug exists only inside manicule.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            anchor = block.anchor
            if isinstance(anchor, HeadingAnchor) and anchor.fragment is None:
                invented = HeadingAnchor(path=anchor.path, fragment=slugify(anchor.path[-1]))
                yield block.model_copy(update={"anchor": invented})
            else:
                yield block


class _GuessingParser(WebParser):
    """Answers an ambiguous heading path with an anchor instead of an admission."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            if isinstance(block.anchor, Unlocated):
                guess = HeadingAnchor(path=block.heading_path, fragment=None)
                yield block.model_copy(update={"anchor": guess})
            else:
                yield block


# --- the corpus --------------------------------------------------------------------------


async def test_the_whole_corpus_round_trips_within_its_location_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, every assertion, and the budget that stops the easy way out.

    HTML declares 0.05 rather than 0.00 because a page that publishes no ``id=`` and repeats
    a heading path genuinely has no address to offer. The ceiling is what keeps that an
    exception: a parser that answered ``Unlocated`` for everything would pass the other five
    assertions and fail this one.
    """
    raws = [raw_from(corpus / "web" / name, MEDIA_TYPE) for name in CORPUS_FIXTURES]
    await check_corpus(_parser(), raws, chunker=chunker, min_blocks=MIN_CORPUS_BLOCKS)


async def test_undecodable_bytes_are_declined_rather_than_indexed_as_replacement_characters(
    corpus: Path,
) -> None:
    """A page that is not text is not this parser's, and saying so lets the chain continue.

    Indexing the replacement characters instead would produce chunks that match queries by
    accident and cite nothing.
    """
    raw = raw_from(corpus / "web" / "mojibake.html", MEDIA_TYPE)
    with pytest.raises(ParseError, match="not decodable"):
        await read_blocks(_parser(), raw)


# --- fragments ---------------------------------------------------------------------------


async def test_a_heading_is_addressed_by_the_id_its_page_publishes(corpus: Path) -> None:
    """The author's ``id`` is the address a citation must use, because it is the one that works.

    Ours would be a different string for the same section, so the link would open at the top
    of the page while claiming to open at the heading.
    """
    raw = raw_from(corpus / "web" / "typical.html", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    headings = [block.anchor for block in blocks if block.kind is BlockKind.HEADING]
    assert headings[:2] == [
        HeadingAnchor(path=("Running the service",), fragment="running-the-service"),
        HeadingAnchor(path=("Running the service", "Starting up"), fragment="starting-up"),
    ]


async def test_an_empty_anchor_element_before_a_heading_is_that_heading_s_address(
    corpus: Path,
) -> None:
    """The pattern that predates ``id`` on arbitrary elements, still emitted by generators.

    It is the author's address in exactly the sense the heading's own ``id`` is, so ignoring
    it would throw away a working deep link and report the section as unaddressable.
    """
    raw = raw_from(corpus / "web" / "structure.html", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    power = next(block for block in blocks if block.heading_path[-1:] == ("Power",))
    assert power.anchor == HeadingAnchor(path=("Fabric", "Power"), fragment="legacy-power")


async def test_a_heading_with_no_published_address_carries_no_fragment() -> None:
    """``fragment=None`` and a document-level link is the honest output.

    The path still resolves the section within the document; what is missing is a URL
    fragment, and inventing one would not resolve on a page manicule does not serve.
    """
    blocks = await _blocks("<h2>Unaddressed</h2><p>Body under it.</p>", title="Page")
    assert blocks[0].anchor == HeadingAnchor(path=("Unaddressed",), fragment=None)


async def test_a_heading_holding_only_an_anchor_does_not_become_an_empty_path_element() -> None:
    """A heading that names nothing is not a heading path element, and not a block either.

    ``<h2><a name="…"></a></h2>`` is a real pattern and it has no text. Emitting it would put
    a block with no text into the index and an empty element into every breadcrumb below it,
    so the content stays in the section that encloses it.
    """
    blocks = await _blocks('<h2><a name="x"></a></h2><p>Body under it.</p>', title="Page")
    assert [block.text for block in blocks] == ["Body under it."]
    assert blocks[0].anchor == HeadingAnchor(path=("Page",), fragment=None)


async def test_synthesising_a_fragment_the_page_never_published_fails_the_round_trip() -> None:
    """The rule above is load-bearing: with it removed, the harness goes red.

    An invented slug addresses nothing, so resolving it returns no text — which is what an
    anchor nobody can resolve looks like from the outside, and exactly what a reader gets
    when they follow the citation.
    """
    raw = raw_of("<h2>Unaddressed</h2><p>Body under it.</p>", MEDIA_TYPE, title="Page")
    with pytest.raises(AssertionError):
        await assert_round_trip(_SluggingParser(WebConfig()), raw, fixture="invented")


async def test_two_sections_with_one_path_and_no_address_are_unlocated() -> None:
    """Two sections one address cannot tell apart are reported, not guessed at.

    Resolution is fragment first and path second; with no fragment on either, the path names
    both, so an anchor built from it would resolve to one of them or to neither.
    """
    blocks = await _blocks(AMBIGUOUS)
    assert all(isinstance(block.anchor, Unlocated) for block in blocks)
    reason = blocks[0].anchor.reason if isinstance(blocks[0].anchor, Unlocated) else ""
    assert "ambiguous heading path" in reason
    assert "id=" in reason


async def test_inventing_an_anchor_for_an_ambiguous_path_fails_the_round_trip() -> None:
    """The guard above is load-bearing: with it removed, the harness goes red."""
    with pytest.raises(AssertionError):
        await assert_round_trip(
            _GuessingParser(WebConfig()), raw_of(AMBIGUOUS, MEDIA_TYPE), fixture="ambiguous"
        )


async def test_a_repeated_id_addresses_the_first_heading_that_carries_it() -> None:
    """A duplicate ``id`` is invalid HTML, and browsers resolve it to the first element.

    So it is not the second section's address, whatever the markup says. Taking it anyway
    would produce two citations that open the same place while naming different sections.
    """
    blocks = await _blocks(
        '<h2 id="dup">Earlier</h2><p>First body.</p><h2 id="dup">Later</h2><p>Second body.</p>',
        title="Page",
    )
    headings = [block.anchor for block in blocks if block.kind is BlockKind.HEADING]
    assert headings == [
        HeadingAnchor(path=("Earlier",), fragment="dup"),
        HeadingAnchor(path=("Later",), fragment=None),
    ]


# --- content before the first heading ----------------------------------------------------


async def test_text_above_the_first_heading_is_addressed_by_the_document_title(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The title is a real heading path element, and page-level is where that text lives.

    There is no fragment because there is no section to deep-link to, which is a coarser
    citation rather than a wrong one.
    """
    raw = raw_from(corpus / "web" / "structure.html", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    assert blocks[0].anchor == HeadingAnchor(path=("structure",), fragment=None)
    assert blocks[0].heading_path == ()
    await check_fixture(_parser(), raw, chunker=chunker)


async def test_a_page_with_no_title_cannot_address_what_precedes_its_first_heading() -> None:
    """With no title there is no path to build, so the blocks say so.

    The alternative — a made-up root element — would put a heading nobody wrote into the
    breadcrumb, and the breadcrumb reaches the embedder.
    """
    blocks = await _blocks("<body><p>Loose text.</p><h1>Later</h1><p>Body.</p></body>")
    assert isinstance(blocks[0].anchor, Unlocated)
    assert "no title" in blocks[0].anchor.reason


async def test_a_title_a_heading_also_claims_cannot_address_the_text_above_it() -> None:
    """One path naming both the page and a section inside it resolves to neither.

    This is the same ambiguity as two identically-titled sections, arriving from the other
    direction, and it gets the same answer.
    """
    html = "<body><p>Loose text above.</p><h1>Overview</h1><p>Body under it.</p></body>"
    blocks = await _blocks(html, title="Overview")
    assert isinstance(blocks[0].anchor, Unlocated)
    assert "is itself called" in blocks[0].anchor.reason


async def test_the_title_element_names_a_page_fetched_without_one() -> None:
    """A page fetched on its own still has a title: the one it carries in its head."""
    blocks = await _blocks(
        "<html><head><title>Its own name</title></head>"
        "<body><p>Loose text.</p><h1>Later</h1><p>Body.</p></body></html>"
    )
    assert blocks[0].anchor == HeadingAnchor(path=("Its own name",), fragment=None)


# --- block kinds -------------------------------------------------------------------------


async def test_a_script_body_never_reaches_the_index(corpus: Path) -> None:
    """Script and style text is code and presentation, not prose.

    Indexed, it matches queries by accident and cites a line no reader ever saw.
    """
    raw = raw_from(corpus / "web" / "unclosed.html", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    assert not any("tracking" in block.text for block in blocks)
    assert not any("rebeccapurple" in block.text for block in blocks)
    assert any("must still be indexed" in block.text for block in blocks)


async def test_a_table_keeps_its_rows_and_counts_its_header(corpus: Path) -> None:
    """A table flattened to a run of words loses which value belongs to which column.

    ``header_rows`` comes from the ``<thead>`` rather than from the first row looking bold,
    because a split table repeats those rows into every part.
    """
    raw = raw_from(corpus / "web" / "typical.html", MEDIA_TYPE)
    table = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.TABLE
    )
    rows = ["Signal | Alarm above", "lease age | 90 seconds", "journal depth | 4096 records"]
    assert table.metadata == {"header_rows": 1, "rows": rows}
    assert table.text.splitlines() == rows, (
        "``rows`` is what the chunker splits on; lines that disagree with the text would "
        "reassemble a table the parser never produced"
    )
    assert table.text.splitlines()[0] == "Signal | Alarm above"


async def test_code_keeps_its_indentation_and_names_its_language(corpus: Path) -> None:
    """Collapsing whitespace inside a ``<pre>`` changes what the code says."""
    raw = raw_from(corpus / "web" / "structure.html", MEDIA_TYPE)
    code = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.CODE
    )
    assert code.lang == "python"
    assert code.text.splitlines()[1].startswith("    return")


async def test_a_nested_list_keeps_its_nesting(corpus: Path) -> None:
    """Depth is meaning in a list: a fifth-level item read as a first-level one is a lie."""
    raw = raw_from(corpus / "web" / "structure.html", MEDIA_TYPE)
    listing = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.LIST
    )
    assert listing.text.splitlines()[-1].startswith("        1. fifth level")


async def test_loose_text_outside_a_block_element_is_still_indexed() -> None:
    """A paragraph written without a ``<p>`` is still a paragraph.

    Dropping it would index the page as shorter than it is, and nothing would report the
    loss.
    """
    blocks = await _blocks("<div>Loose <b>inline</b> text.<p>And a real paragraph.</p></div>")
    assert [block.text for block in blocks] == ["Loose inline text.", "And a real paragraph."]


async def test_an_image_contributes_its_alt_text_and_nothing_else(corpus: Path) -> None:
    """With no OCR, an image with no alt text contributes nothing, and that is the honest
    outcome rather than an empty block that would be retrievable and cite nothing."""
    raw = raw_from(corpus / "web" / "typical.html", MEDIA_TYPE)
    media = [block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.MEDIA]
    assert [block.text for block in media] == ["Three machines around one shared volume"]
    assert media[0].metadata == {"src": "topology.png"}

    assert await _blocks('<img src="undescribed.png">') == []


# --- degenerate and hostile input --------------------------------------------------------


async def test_an_empty_document_yields_no_blocks_rather_than_raising() -> None:
    """Nothing to parse is an outcome, not a failure."""
    assert await _blocks("") == []


async def test_tags_that_were_never_closed_still_produce_blocks(corpus: Path) -> None:
    """Half the web is like this, and a parser that refused it would index none of that half."""
    raw = raw_from(corpus / "web" / "unclosed.html", MEDIA_TYPE)
    report = await check_fixture(_parser(), raw)
    assert report.blocks >= 4


async def test_astral_plane_text_survives_a_heading(corpus: Path) -> None:
    """Emoji and CJK-extension characters are two UTF-16 units and one character.

    Anything counting the wrong one truncates the heading, and the truncated string is what
    every breadcrumb and every citation into that section would then carry.
    """
    raw = raw_from(corpus / "web" / "astral.html", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    assert blocks[0].heading_path == ("🚀 Checklist for 𠀋 builds",)


# --- resolution --------------------------------------------------------------------------


async def test_an_anchor_resolves_against_a_parser_that_has_never_seen_the_document(
    corpus: Path,
) -> None:
    """Resolution re-derives the location from the bytes, never from what parsing remembered.

    A parser that resolved out of its own memory would verify nothing: the round trip would
    be comparing the parser against itself.
    """
    raw = raw_from(corpus / "web" / "typical.html", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    fresh: Parser = _parser()
    resolved = await fresh.resolve(blocks[-1].anchor, raw)
    assert resolved is not None
    assert blocks[-1].text in resolved


async def test_an_anchor_from_another_document_does_not_resolve() -> None:
    """An anchor that no longer fits its page is reported as unresolvable rather than
    approximated, because an approximate citation cannot be told from a correct one."""
    parser = _parser()
    raw = raw_of("<h1 id='here'>Here</h1><p>Body.</p>", MEDIA_TYPE, title="Page")
    assert await parser.resolve(HeadingAnchor(path=("Absent",), fragment="absent"), raw) is None
    assert await parser.resolve(HeadingAnchor(path=("Absent",), fragment=None), raw) is None


async def test_anchoring_each_block_to_the_next_section_fails_the_round_trip(
    corpus: Path,
) -> None:
    """The harness catches the mistake it exists to catch.

    Shifting every anchor by one section leaves output that is well-formed in every respect a
    type checker or a schema can see. Only resolving the anchors finds it.
    """
    raw = raw_from(corpus / "web" / "typical.html", MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_NextSectionParser(WebConfig()), raw, fixture="shifted")


# --- CDATA sections, which an HTML parser turns into comments --------------------------------

CODE_MACRO = (
    "<h2>Retry policy</h2>"
    '<ac:structured-macro ac:name="code">'
    '<ac:parameter ac:name="language">python</ac:parameter>'
    "<ac:plain-text-body><![CDATA[def retry(attempts):\n"
    "    return attempts > 0]]></ac:plain-text-body>"
    "</ac:structured-macro>"
)
"""A Confluence code macro, in the shape Server and Data Center actually serve.

Invented here — no real page, no real host. What matters is the ``<![CDATA[…]]>`` wrapper, which
Confluence puts around the body of every ``code``, ``noformat`` and ``graphviz`` macro.
"""

GRAPHVIZ_MACRO = (
    "<h2>Topology</h2>"
    '<ac:structured-macro ac:name="graphviz">'
    '<ac:plain-text-body><![CDATA[digraph G { client -> server [label="retry"]; }]]>'
    "</ac:plain-text-body></ac:structured-macro>"
)
"""The same wrapper around DOT source, which fails differently and worse.

``->`` inside the body terminates the bogus comment early, so part of the DOT leaks into the
document as prose and the rest is eaten — a corruption rather than a clean loss.
"""


async def test_a_cdata_section_reaches_the_document_as_text() -> None:
    """CDATA content is content, and an HTML parser silently deletes it.

    ``<![CDATA[…]]>`` is not a construct HTML has outside foreign content, so a conforming HTML
    parser reparses it as a **bogus comment** — lexbor produces ``<!--[CDATA[…]]-->`` — and the
    text inside is gone. Not degraded: absent.

    This is the whole of a live defect. Confluence Server and Data Center wrap the body of every
    ``code``, ``noformat`` and ``graphviz`` macro in CDATA, storage-format bodies are routed as
    ``text/html``, and so every code block on every one of those pages has been missing from the
    index while a fragment of it sits there quotable as prose.

    The second assertion is the sharper one. A test that only checked the body was present would
    pass for a parser that emitted the raw ``]]>`` delimiter along with it, which is what the
    broken behaviour already does.
    """
    blocks = await read_blocks(WebParser(WebConfig()), raw_of(CODE_MACRO, MEDIA_TYPE))
    text = "\n".join(block.text for block in blocks)

    assert "def retry(attempts):" in text, (
        "the macro body is absent from the document: an HTML parser reparsed its CDATA wrapper as "
        "a comment, so every Confluence code block is missing from the index"
    )
    assert "]]>" not in text, "a CDATA delimiter leaked into the document as content"
    assert "<![CDATA[" not in text


async def test_dot_source_in_a_cdata_section_is_not_corrupted() -> None:
    """The Graphviz case, which fails differently: partly eaten, partly leaked.

    ``->`` inside a bogus comment terminates it early, so the DOT is split — the opening is
    swallowed with the comment and the tail escapes into the document as prose. A reader gets a
    fragment of a diagram definition presented as the page's words.

    Asserted separately from the code macro because the mechanism is different and a fix for one
    does not obviously cover the other.
    """
    blocks = await read_blocks(WebParser(WebConfig()), raw_of(GRAPHVIZ_MACRO, MEDIA_TYPE))
    text = "\n".join(block.text for block in blocks)

    assert "digraph G {" in text, "the DOT source was eaten by the bogus-comment reparse"
    assert "]]>" not in text, "the tail of the DOT leaked out as prose with its delimiter"


async def test_markup_inside_a_cdata_section_stays_inert() -> None:
    """Recovering CDATA must not *promote* text into markup, which is the risk the fix creates.

    CDATA exists so a body can contain ``<`` and ``&`` without being parsed as markup, so a
    recovery that pasted the text back unescaped would turn ``<![CDATA[<script>…]]>`` from inert
    content into a live element — trading a content-loss bug for an execution one, on a corpus
    whose pages anybody with write access to a wiki can edit.

    Both halves are asserted: the text is *present* as text, and it did not become an element. The
    presence half is what stops this passing for a parser that went back to deleting the section.
    """
    hostile = (
        "<h2>Retry policy</h2>"
        "<ac:plain-text-body><![CDATA[<script>window.__owned=1</script>"
        "if a < b && c > d: pass]]></ac:plain-text-body>"
    )
    parser = WebParser(WebConfig())
    raw = raw_of(hostile, MEDIA_TYPE)
    blocks = await read_blocks(parser, raw)
    text = "\n".join(block.text for block in blocks)

    assert "window.__owned=1" in text, "the recovered text is missing, so nothing is asserted here"
    assert "if a < b && c > d: pass" in text, "the escaped characters did not survive as characters"
    assert not any(block.kind is BlockKind.MEDIA for block in blocks), (
        "the recovered markup became an element rather than text"
    )
    # The document the parser actually saw: a text node, not a script element.
    assert "<script>" not in _rendered(hostile), (
        "the recovery pasted markup back unescaped, promoting inert content to a live element"
    )


def _rendered(document: str) -> str:
    """What the CDATA recovery hands to the HTML engine, for asserting on directly."""
    return recover_cdata(document)


async def test_an_unterminated_cdata_section_is_left_alone() -> None:
    """A malformed document is not an invitation to invent where its content ended.

    Guessing a terminator would fabricate a block boundary out of nothing, so an unclosed section
    is handed to the HTML engine exactly as it arrived and its own error recovery decides.

    **The assertion is on the rendered string, not on the blocks.** An earlier version checked only
    the first two blocks, which left the unterminated section free to be escaped and appended as a
    third — so the test passed with the "leave it alone" branch removed entirely. Found by removing
    it. What pins the behaviour is that the marker is still there, untransformed.
    """
    document = "<h2>Retry policy</h2><p>x</p><ac:body><![CDATA[never closed"

    assert _rendered(document) == document, (
        "an unterminated section was transformed; the recovery guessed where it ended"
    )
    blocks = await read_blocks(WebParser(WebConfig()), raw_of(document, MEDIA_TYPE))
    assert [block.text for block in blocks][:2] == ["Retry policy", "x"]


async def test_ordinary_html_passes_through_the_recovery_unchanged() -> None:
    """Every document without a CDATA section — which is almost all of them — is untouched.

    The property that matters is that the recovery cannot corrupt ordinary HTML: no escaping
    applied outside a section, nothing reordered, nothing dropped. A document containing ``&`` and
    ``<`` in text is the case that would break if the escape were applied to the whole input rather
    than to recovered bodies.

    **Deliberately not asserted on identity.** An earlier version checked
    ``_rendered(document) is document`` to prove the early return was taken — and that assertion
    can never fail, because CPython's ``"".join([x])`` returns ``x`` itself. The early return is a
    performance detail with no observable behaviour, so there is nothing honest to assert about it;
    this asserts the observable property instead.
    """
    plain = "<h1>Retry policy &amp; backoff</h1><p>Use a &lt; b for the bound.</p>"
    assert _rendered(plain) == plain

    # And the same claim for text *beside* a section, which is the case the fast path above does
    # not exercise: an earlier version of this test used only a CDATA-free document, so escaping
    # the surrounding markup as well would have left it green. Found by trying exactly that.
    mixed = f"{plain}<ac:body><![CDATA[a < b]]></ac:body>{plain}"
    rendered = _rendered(mixed)
    assert rendered.startswith(plain), "markup before a section was escaped"
    assert rendered.endswith(plain), "markup after a section was escaped"
    assert "a &lt; b" in rendered, "the recovered body was not escaped"


async def test_a_cdata_section_that_is_not_a_macro_body_is_recovered_too() -> None:
    """A page *about* CDATA gets the same treatment, and that is deliberate rather than incidental.

    The recovery knows nothing about Confluence — it is a rule about HTML — so a document showing
    CDATA syntax as an example has that example recovered as text. That is the correct outcome for
    the same reason it is correct for a macro body: the author put content there and an HTML parser
    would have deleted it.

    Pinned so the next person changing this knows the behaviour was chosen. The alternative — only
    recovering sections inside `ac:` elements — would make the rule Confluence-specific and would
    silently keep losing content from every other document.
    """
    document = (
        "<h2>Writing XML</h2>"
        "<p>Wrap literal markup in a section:</p>"
        '<pre><![CDATA[<config enabled="true"/>]]></pre>'
    )
    blocks = await read_blocks(WebParser(WebConfig()), raw_of(document, MEDIA_TYPE))
    text = "\n".join(block.text for block in blocks)

    assert '<config enabled="true"/>' in text, "an ordinary document's CDATA example was deleted"
    assert "]]>" not in text
    assert any(block.kind is BlockKind.CODE for block in blocks), (
        "the example is inside <pre>, so it should still be a code block"
    )


async def test_script_and_style_are_decomposed_before_the_recovery_matters() -> None:
    """The legacy ``//<![CDATA[`` idiom never reaches a reader, and that is someone else's default.

    ``WebConfig.drop_tags`` decomposes ``script`` and ``style``, so the wrapper old documents put
    around inline JavaScript is gone before anything cites it. **Asserted rather than assumed**: it
    is a dependency on another module's default, and if that default ever changed, this recovery
    would start surfacing script bodies as document text.
    """
    document = (
        "<h2>Retry policy</h2>"
        "<script>//<![CDATA[\nwindow.__owned=1\n//]]></script>"
        "<style>/*<![CDATA[*/ .x{} /*]]>*/</style>"
        "<p>The client retries twice.</p>"
    )
    blocks = await read_blocks(WebParser(WebConfig()), raw_of(document, MEDIA_TYPE))
    text = "\n".join(block.text for block in blocks)

    assert {"script", "style"} <= WebConfig().drop_tags, (
        "drop_tags no longer decomposes script/style, so the CDATA recovery would surface "
        "inline JavaScript as document text"
    )
    assert "window.__owned=1" not in text
    assert ".x{}" not in text
    assert "The client retries twice." in text
