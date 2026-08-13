"""Confluence ADF parsing: typed nodes in, located blocks out.

Two obligations carry most of the weight here.

**The fragment has to be Confluence's.** A citation into Confluence deep-links with
``#{anchor}``, and Confluence derives that anchor from the heading text, case included. An
anchor derived any other way is a link that opens the top of the page while claiming to open
the section — the same defect as a wrong page number, arriving through a URL.

**A node type nobody has seen yet is not a broken document.** ADF gains node types; a page
containing one still has all its other text, and refusing the page would lose it.

One thing this suite asserts directly rather than through the corpus: two sections with the
same heading title. ADF heading text is exactly the title, so two such sections produce two
blocks that read identically, which the discrimination assertion rejects on principle — it
cannot tell two identical strings apart, and neither can a reader. The fragments *do* tell
them apart, so :func:`test_two_sections_called_overview_resolve_to_different_text` checks
that where it can be checked.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, read_blocks
from manicule.parsers.adf import ADF_MEDIA_TYPE, ADFConfig, ADFParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

CORPUS_FIXTURES = (
    "typical.json",
    "structure.json",
    "preamble.json",
    "heading-only.json",
    "empty-document.json",
    "astral.json",
    "unknown-nodes.json",
    "page-large.json",
)
"""Every fixture this parser is expected to parse. Listed rather than globbed, so a
generator that stops writing one fails here instead of quietly shrinking the corpus."""

DECLINED_FIXTURES = ("empty.json", "not-adf.json", "mojibake.json")
"""Fixtures this parser must refuse, so that the next parser in the chain gets a turn."""

MIN_CORPUS_BLOCKS = 140
"""A floor under the corpus, because every ratio passes trivially on an empty one."""


def _parser(**overrides: bool) -> ADFParser:
    return ADFParser(ADFConfig(**overrides))


def _text(value: str) -> dict[str, object]:
    return {"type": "text", "text": value}


def _paragraph(value: str) -> dict[str, object]:
    return {"type": "paragraph", "content": [_text(value)]}


def _heading(level: int, value: str) -> dict[str, object]:
    return {"type": "heading", "attrs": {"level": level}, "content": [_text(value)]}


def _doc(*content: dict[str, object]) -> str:
    return json.dumps({"type": "doc", "version": 1, "content": list(content)})


async def _blocks(document: str, *, title: str = "Page", **overrides: bool) -> list[ParsedBlock]:
    return await read_blocks(_parser(**overrides), raw_of(document, ADF_MEDIA_TYPE, title=title))


AMBIGUOUS = _doc(
    _heading(2, "!!!"),
    _paragraph("The first section, whose heading yields no Confluence anchor."),
    _heading(2, "!!!"),
    _paragraph("The second, which the path alone cannot tell from the first."),
)


class _NextSectionParser(ADFParser):
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


class _GuessingParser(ADFParser):
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

    ADF declares 0.00 unlocated: Confluence derives an anchor for every heading, so a section
    without one is a section whose heading is punctuation. The ceiling is what stops a parser
    satisfying the other five assertions by never producing a location at all.
    """
    raws = [raw_from(corpus / "adf" / name, ADF_MEDIA_TYPE) for name in CORPUS_FIXTURES]
    await check_corpus(_parser(), raws, chunker=chunker, min_blocks=MIN_CORPUS_BLOCKS)


@pytest.mark.parametrize("name", DECLINED_FIXTURES)
async def test_input_that_is_not_an_adf_body_is_declined(corpus: Path, name: str) -> None:
    """Declining is information: it says the document is not this parser's kind.

    This media type's prefix is ``application/json``, so plain JSON reaches here by
    misrouting more often than by accident. Raising rather than producing blocks is what lets
    the structured-data parser have it, and what keeps ``unsupported_media_type`` reachable.
    """
    raw = raw_from(corpus / "adf" / name, ADF_MEDIA_TYPE)
    with pytest.raises(ParseError):
        await read_blocks(_parser(), raw)


async def test_a_document_with_no_content_yields_no_blocks_rather_than_raising() -> None:
    """An empty page is an outcome, not a failure.

    Raising would advance the fallback chain and end as ``failed``; yielding nothing ends as
    ``no_extractable_text``, which is the truthful record and the one that is re-indexable.
    """
    assert await _blocks(_doc()) == []


# --- fragments ---------------------------------------------------------------------------


async def test_a_heading_anchor_keeps_the_case_confluence_publishes() -> None:
    """A URL fragment is case-sensitive, so a lowercased anchor deep-links to nothing.

    This is the whole reason the fragment is derived here rather than slugified: a slug is
    the right answer where the source publishes no address, and the wrong one where it does.
    """
    blocks = await _blocks(_doc(_heading(2, "Token Refresh"), _paragraph("Body text.")))
    assert blocks[0].anchor == HeadingAnchor(path=("Token Refresh",), fragment="Token-Refresh")


async def test_two_sections_called_overview_resolve_to_different_text() -> None:
    """Confluence numbers a repeated heading from the *second* occurrence, and so does this.

    Counting from the first would rename the section that was there before the duplicate
    arrived, breaking every citation already written against it. The check that matters is
    the last one: two anchors that resolved to the same span would be two citations nobody
    can tell apart.
    """
    document = _doc(
        _heading(2, "Overview"),
        _paragraph("The first section with this title."),
        _heading(2, "Overview"),
        _paragraph("The second, sharing its path exactly."),
    )
    raw = raw_of(document, ADF_MEDIA_TYPE, title="Page")
    blocks = await read_blocks(_parser(), raw)
    headings = [block.anchor for block in blocks if block.kind is BlockKind.HEADING]
    assert headings == [
        HeadingAnchor(path=("Overview",), fragment="Overview"),
        HeadingAnchor(path=("Overview",), fragment="Overview-1"),
    ]

    parser = _parser()
    first = await parser.resolve(headings[0], raw)
    second = await parser.resolve(headings[1], raw)
    assert first is not None
    assert second is not None
    assert "first section" in first
    assert "sharing its path" in second


async def test_a_heading_that_yields_no_anchor_and_repeats_a_path_is_unlocated() -> None:
    """Two sections one address cannot tell apart are reported, not guessed at.

    A heading of punctuation derives no Confluence anchor, so neither section has a fragment
    and the path names both.
    """
    blocks = await _blocks(AMBIGUOUS)
    assert all(isinstance(block.anchor, Unlocated) for block in blocks)
    reason = blocks[0].anchor.reason if isinstance(blocks[0].anchor, Unlocated) else ""
    assert "ambiguous heading path" in reason


async def test_inventing_an_anchor_for_an_ambiguous_path_fails_the_round_trip() -> None:
    """The guard above is load-bearing: with it removed, the harness goes red.

    A check nobody has watched fail is a check nobody knows works.
    """
    raw = raw_of(AMBIGUOUS, ADF_MEDIA_TYPE, title="Page")
    with pytest.raises(AssertionError):
        await assert_round_trip(_GuessingParser(ADFConfig()), raw, fixture="ambiguous")


# --- content before the first heading ----------------------------------------------------


async def test_text_above_the_first_heading_is_addressed_by_the_page_title(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The page title is a real heading path element and page-level is where that text lives.

    No fragment, because there is no section to deep-link to — a coarser citation rather than
    a wrong one.
    """
    raw = raw_from(corpus / "adf" / "preamble.json", ADF_MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    assert blocks[0].anchor == HeadingAnchor(path=("preamble",), fragment=None)
    assert blocks[0].heading_path == ()
    await check_fixture(_parser(), raw, chunker=chunker)


async def test_a_page_fetched_without_a_title_cannot_address_what_precedes_its_headings() -> None:
    """With no title there is no path to build, so the blocks say so.

    ADF carries no title node — the page title is Confluence metadata — so a body fetched
    without it genuinely has nothing to name the text above its first heading.
    """
    blocks = await _blocks(_doc(_paragraph("Loose text."), _heading(1, "Later")), title="")
    assert isinstance(blocks[0].anchor, Unlocated)
    assert "without a title" in blocks[0].anchor.reason


async def test_a_title_a_heading_also_claims_cannot_address_the_text_above_it() -> None:
    """One path naming both the page and a section inside it resolves to neither."""
    document = _doc(_paragraph("Loose text above."), _heading(1, "Overview"), _paragraph("Body."))
    blocks = await _blocks(document, title="Overview")
    assert isinstance(blocks[0].anchor, Unlocated)
    assert "is itself called" in blocks[0].anchor.reason


# --- node handling -----------------------------------------------------------------------


async def test_a_panel_keeps_its_severity(corpus: Path) -> None:
    """A warning panel is not ordinary prose, and a warning split in half is still a warning.

    The severity is metadata rather than a sentence prepended to the text, so the citation
    quotes the panel and nothing else while the chunker can still keep the parts labelled.
    """
    raw = raw_from(corpus / "adf" / "typical.json", ADF_MEDIA_TYPE)
    panel = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.PANEL
    )
    assert panel.metadata == {"panel_type": "warning"}


async def test_a_code_block_names_its_language_and_keeps_its_indentation(corpus: Path) -> None:
    """Collapsing whitespace inside a code block changes what the code says."""
    raw = raw_from(corpus / "adf" / "structure.json", ADF_MEDIA_TYPE)
    code = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.CODE
    )
    assert code.lang == "python"
    assert code.text.splitlines()[1].startswith("    return")


async def test_a_table_keeps_its_rows_and_counts_its_header(corpus: Path) -> None:
    """A table flattened to a run of words loses which value belongs to which column.

    ``header_rows`` comes from the ``tableHeader`` cells, because a table too large for one
    chunk repeats those rows into every part and the wrong ones mislabel every column.

    ``rows`` is what makes that split happen. The chunker splits at row boundaries only when
    the block describes them and otherwise falls back to prose splitting, which cuts mid-row —
    so a docstring promising the header-repeating split while emitting only ``header_rows``
    described something that never ran. The two are asserted together for that reason.
    """
    raw = raw_from(corpus / "adf" / "typical.json", ADF_MEDIA_TYPE)
    table = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.TABLE
    )
    assert table.metadata == {
        "header_rows": 1,
        "rows": [
            "Signal | Alarm above",
            "lease age | 90 seconds",
            "journal depth | 4096 records",
        ],
    }
    assert table.text.splitlines() == table.metadata["rows"], (
        "the rendered text and the declared rows must be the same lines, or the chunker "
        "reassembles a table the parser never produced"
    )


async def test_a_nested_list_keeps_its_nesting(corpus: Path) -> None:
    """Depth is meaning in a list: a fifth-level item read as a first-level one is a lie."""
    raw = raw_from(corpus / "adf" / "structure.json", ADF_MEDIA_TYPE)
    listing = next(
        block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.LIST
    )
    assert listing.text.splitlines()[-1].startswith("        - fifth level")


async def test_collapsed_content_is_still_content(corpus: Path) -> None:
    """An ``expand`` hides its body in the browser, not in the document.

    Skipping it would index a page as missing text that is plainly there when a reader opens
    the section, and nothing would report the gap.
    """
    raw = raw_from(corpus / "adf" / "structure.json", ADF_MEDIA_TYPE)
    texts = [block.text for block in await read_blocks(_parser(), raw)]
    assert "Historic feeds" in texts
    assert "Two feeds per rack, from separate boards on separate floors." in texts


async def test_a_macro_body_is_walked_rather_than_swallowed(corpus: Path) -> None:
    """A ``bodiedExtension`` wraps real content, and the wrapper is not the interesting part."""
    raw = raw_from(corpus / "adf" / "structure.json", ADF_MEDIA_TYPE)
    texts = [block.text for block in await read_blocks(_parser(), raw)]
    assert "A macro body, which must not swallow the text inside it." in texts


async def test_an_unknown_node_type_keeps_the_text_inside_it(corpus: Path) -> None:
    """ADF gains node types, and a page containing one is not a broken page.

    Its children are walked, so nothing inside is lost; a node whose children are all inline
    becomes prose, because that is what a node with inline content is. Refusing the document
    would lose everything else on the page over one unfamiliar wrapper.
    """
    raw = raw_from(corpus / "adf" / "unknown-nodes.json", ADF_MEDIA_TYPE)
    texts = [block.text for block in await read_blocks(_parser(), raw)]
    assert "Inline content inside an unfamiliar node." in texts
    assert "Block content nested inside another unfamiliar node." in texts


async def test_an_attachment_contributes_its_alt_text_and_nothing_else(corpus: Path) -> None:
    """With no OCR, the alt text or filename is all an attachment reference can honestly say.

    The bytes are a document of their own, parsed by the chain like any other file.
    """
    raw = raw_from(corpus / "adf" / "structure.json", ADF_MEDIA_TYPE)
    media = [block for block in await read_blocks(_parser(), raw) if block.kind is BlockKind.MEDIA]
    assert media[0].text == "Rack elevation drawing"
    assert media[0].metadata == {"media_type": "file"}


async def test_a_card_is_a_cross_reference_and_can_be_turned_off(corpus: Path) -> None:
    """Cards link pages together, which is a graph worth keeping — and prose it is not.

    Configurable rather than fixed because a corpus where every page links to a dozen others
    embeds a great many URLs and very little meaning.
    """
    raw = raw_from(corpus / "adf" / "structure.json", ADF_MEDIA_TYPE)
    kept = [block.text for block in await read_blocks(_parser(), raw)]
    assert "https://example.invalid/neighbouring-page" in kept

    dropped = [block.text for block in await read_blocks(_parser(keep_card_links=False), raw)]
    assert "https://example.invalid/neighbouring-page" not in dropped


async def test_astral_plane_text_survives_a_heading_and_its_anchor(corpus: Path) -> None:
    """Emoji and CJK-extension characters are two UTF-16 units and one character.

    Anything counting the wrong one truncates the heading, which changes the anchor every
    citation into that section was written against.
    """
    raw = raw_from(corpus / "adf" / "astral.json", ADF_MEDIA_TYPE)
    heading = (await read_blocks(_parser(), raw))[0]
    assert heading.heading_path == ("🚀 Checklist for 𠀋 builds",)
    assert isinstance(heading.anchor, HeadingAnchor)
    assert heading.anchor.fragment == "Checklist-for-𠀋-builds"


# --- resolution --------------------------------------------------------------------------


async def test_an_anchor_resolves_against_a_parser_that_has_never_seen_the_document(
    corpus: Path,
) -> None:
    """Resolution re-derives the location from the bytes, never from what parsing remembered.

    A parser that resolved out of its own memory would verify nothing: the round trip would
    be comparing the parser against itself.
    """
    raw = raw_from(corpus / "adf" / "typical.json", ADF_MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    fresh: Parser = _parser()
    resolved = await fresh.resolve(blocks[-1].anchor, raw)
    assert resolved is not None
    assert blocks[-1].text in resolved


async def test_an_anchor_from_another_page_does_not_resolve() -> None:
    """An anchor that no longer fits its page is reported as unresolvable rather than
    approximated, because an approximate citation cannot be told from a correct one."""
    parser = _parser()
    raw = raw_of(_doc(_heading(1, "Here"), _paragraph("Body.")), ADF_MEDIA_TYPE, title="Page")
    assert await parser.resolve(HeadingAnchor(path=("Absent",), fragment="Absent"), raw) is None
    assert await parser.resolve(HeadingAnchor(path=("Absent",), fragment=None), raw) is None


async def test_anchoring_each_block_to_the_next_section_fails_the_round_trip(
    corpus: Path,
) -> None:
    """The harness catches the mistake it exists to catch.

    Shifting every anchor by one section leaves output that is well-formed in every respect a
    type checker or a schema can see. Only resolving the anchors finds it, which is why
    resolution is part of the protocol rather than a convention.
    """
    raw = raw_from(corpus / "adf" / "typical.json", ADF_MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_NextSectionParser(ADFConfig()), raw, fixture="shifted")
