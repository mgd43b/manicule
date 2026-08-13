"""Markdown and MDX parsing, and the two things about it that fail silently.

The first is arithmetic. ``markdown-it-py`` reports a 0-based, half-open line span and every
anchor manicule stores is 1-based and inclusive, so an off-by-one here produces a citation
that resolves to the paragraph next door — which reads perfectly and is wrong.
:func:`test_a_line_anchor_counts_from_one_and_includes_its_last_line` pins the conversion,
and :func:`test_anchoring_each_block_to_the_next_section_fails_the_round_trip` shows that the
round-trip harness notices when it moves.

The second is a fragment that addresses two places. Two sections can share a heading path,
and the fragment is the only thing that tells them apart; where there is no fragment either,
the honest answer is ``Unlocated``.
:func:`test_inventing_an_anchor_for_an_ambiguous_path_fails_the_round_trip` runs the
implementation that guesses instead, and requires it to fail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import HeadingAnchor, LineAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, read_blocks
from manicule.parsers.markdown import MarkdownConfig, MarkdownParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

MEDIA_TYPE = "text/markdown"
MDX_MEDIA_TYPE = "text/mdx"

CORPUS_FIXTURES: tuple[tuple[str, str], ...] = (
    ("typical.md", MEDIA_TYPE),
    ("structure.md", MEDIA_TYPE),
    ("preamble.md", MEDIA_TYPE),
    ("heading-only.md", MEDIA_TYPE),
    ("no-trailing-newline.md", MEDIA_TYPE),
    ("empty.md", MEDIA_TYPE),
    ("astral.md", MEDIA_TYPE),
    ("components.mdx", MDX_MEDIA_TYPE),
    ("handbook-large.md", MEDIA_TYPE),
)
"""Every fixture this parser is expected to parse. Listed rather than globbed, so a
generator that stops writing one fails here instead of quietly shrinking the corpus."""

MIN_CORPUS_BLOCKS = 150
"""A floor under the corpus, because every ratio passes trivially on an empty one."""


def _parser() -> MarkdownParser:
    return MarkdownParser(MarkdownConfig())


async def _blocks(text: str, media_type: str = MEDIA_TYPE) -> list[ParsedBlock]:
    return await read_blocks(_parser(), raw_of(text, media_type))


class _NextSectionParser(MarkdownParser):
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


class _GuessingParser(MarkdownParser):
    """Answers an ambiguous heading path with an anchor instead of an admission.

    This is what "resolution order is fragment first, path second" looks like when the
    fallback is allowed to guess: the anchor is well-formed and addresses two sections, so
    resolving it can only return one of them or neither.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            if isinstance(block.anchor, Unlocated):
                guess = HeadingAnchor(path=block.heading_path, fragment=None)
                yield block.model_copy(update={"anchor": guess})
            else:
                yield block


AMBIGUOUS = """# !!!

The first section, whose heading slugifies to nothing at all.

# !!!

The second, which the path alone cannot tell from the first.
"""


# --- the corpus --------------------------------------------------------------------------


async def test_the_whole_corpus_round_trips_within_its_location_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, every assertion, and the budget that stops the easy way out.

    Assertions one to four and six hold per fixture; the fifth is the corpus-wide ceiling on
    ``Unlocated``, which is what stops "never invent a location" being satisfied by never
    producing one. Markdown declares 0.00, because a source line number is always available.
    """
    raws = [raw_from(corpus / "markdown" / name, media) for name, media in CORPUS_FIXTURES]
    await check_corpus(_parser(), raws, chunker=chunker, min_blocks=MIN_CORPUS_BLOCKS)


async def test_undecodable_bytes_are_declined_rather_than_indexed_as_replacement_characters(
    corpus: Path,
) -> None:
    """A file that is not text is not this parser's, and saying so lets the chain continue.

    Indexing the replacement characters instead would produce chunks that match queries by
    accident and cite nothing, and would make ``unsupported_media_type`` unreachable because
    some parser always claimed every document.
    """
    raw = raw_from(corpus / "markdown" / "mojibake.md", MEDIA_TYPE)
    with pytest.raises(ParseError, match="not decodable"):
        await read_blocks(_parser(), raw)


# --- line arithmetic ---------------------------------------------------------------------


async def test_a_line_anchor_counts_from_one_and_includes_its_last_line() -> None:
    """The single conversion this parser makes, pinned to numbers a person can read.

    ``token.map`` is 0-based and half-open. Every anchor here is 1-based and inclusive,
    because that is what ``notes.md:3`` means to a reader and what every editor shows. Off by
    one in either direction produces a citation that resolves to adjacent text.
    """
    blocks = await _blocks("alpha\nbeta\n\ngamma\n")
    assert [block.anchor for block in blocks] == [
        LineAnchor(start=1, end=2),
        LineAnchor(start=4, end=4),
    ]


async def test_a_list_is_not_anchored_to_the_blank_line_that_ended_it() -> None:
    """``markdown-it-py`` ends a list's span on the blank line after it.

    Anchoring that line would claim a line the block does not contain, which is the same
    defect as an off-by-one and shows up the same way: the citation quotes one line too many.
    """
    blocks = await _blocks("- one\n- two\n\nA paragraph after the list.\n")
    assert blocks[0].kind is BlockKind.LIST
    assert blocks[0].anchor == LineAnchor(start=1, end=2)


async def test_content_above_the_first_heading_is_addressed_by_line_number(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Markdown never needs an invented root heading for its preamble.

    Other formats have to fall back to the document title for text that precedes every
    heading. Markdown does not: source line numbers are exact, already in hand, and resolve
    to precisely the lines the block came from.
    """
    raw = raw_from(corpus / "markdown" / "preamble.md", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    above = [block for block in blocks if isinstance(block.anchor, LineAnchor)]
    assert len(above) == 2
    assert [block.heading_path for block in above] == [(), ()]
    await check_fixture(_parser(), raw, chunker=chunker)


# --- headings, paths and fragments -------------------------------------------------------


async def test_two_sections_called_overview_get_different_fragments(corpus: Path) -> None:
    """A heading path repeats constantly, so the fragment is what resolves a citation.

    Both sections here are second-level and both are called "Overview", so their paths are
    identical. The ``-1`` suffix is counted from the second occurrence, as every Markdown
    host does it, so the section that was there first keeps the address it already had.
    """
    blocks = await read_blocks(
        _parser(), raw_from(corpus / "markdown" / "structure.md", MEDIA_TYPE)
    )
    overviews = [
        block.anchor
        for block in blocks
        if block.kind is BlockKind.HEADING and block.heading_path[-1:] == ("Overview",)
    ]
    assert overviews == [
        HeadingAnchor(path=("Alpha", "Overview"), fragment="overview"),
        HeadingAnchor(path=("Alpha", "Overview"), fragment="overview-1"),
    ]


async def test_each_repeated_heading_resolves_to_its_own_section(corpus: Path) -> None:
    """The fragments are only worth having if they address different text.

    Distinct anchors that resolve to the same span are two citations nobody can tell apart,
    which is the failure the discrimination assertion exists to catch — asserted directly
    here so that the reason is legible without reading the harness.
    """
    raw = raw_from(corpus / "markdown" / "structure.md", MEDIA_TYPE)
    parser = _parser()
    first = await parser.resolve(
        HeadingAnchor(path=("Alpha", "Overview"), fragment="overview"), raw
    )
    second = await parser.resolve(
        HeadingAnchor(path=("Alpha", "Overview"), fragment="overview-1"), raw
    )
    assert first is not None
    assert second is not None
    assert "ATX heading" in first
    assert "setext way" in second
    assert first != second


async def test_a_section_stops_at_the_next_heading_of_any_level() -> None:
    """A section that swallowed its subsections would be far larger than the text it names.

    The tightness bound allows a resolved section to exceed its blocks by a fifth, for its
    own heading line and the whitespace between its blocks. Including subsections exceeds it
    by however deep the nesting goes, so every nested document would fail.
    """
    source = "# Outer\n\nOuter prose.\n\n## Inner\n\nInner prose.\n"
    resolved = await _parser().resolve(
        HeadingAnchor(path=("Outer",), fragment="outer"), raw_of(source, MEDIA_TYPE)
    )
    assert resolved == "# Outer\n\nOuter prose."


async def test_a_heading_path_that_skips_a_level_is_not_padded_with_an_invented_one() -> None:
    """A document that jumps from ``#`` to ``###`` produces a two-element path.

    Padding with an empty element would reach the embedder through the breadcrumb as a
    heading nobody wrote, and a wrong breadcrumb is worse than a short one.
    """
    blocks = await _blocks("# Top\n\n### Deep\n\nProse under the deep heading.\n")
    assert blocks[-1].heading_path == ("Top", "Deep")


async def test_a_heading_with_no_text_does_not_become_an_empty_path_element() -> None:
    """A heading that names nothing cannot be part of a breadcrumb.

    An empty element would reach the embedder as a heading nobody wrote, and the block under
    it would be addressed by a path with a hole in it. The content belongs to the enclosing
    section instead, which here is the document above the first real heading.
    """
    blocks = await _blocks("#\n\nBody under a heading with no text.\n")
    assert [block.heading_path for block in blocks] == [()]
    assert blocks[0].anchor == LineAnchor(start=3, end=3)


async def test_an_ambiguous_heading_path_with_no_fragment_is_unlocated() -> None:
    """Two sections one address cannot tell apart are reported, not guessed at.

    A heading of punctuation slugifies to nothing, so neither section has a fragment, and the
    path names both. An anchor here would resolve to one of them or to neither — so there is
    no anchor, and the reason says why.
    """
    blocks = await _blocks(AMBIGUOUS)
    assert all(isinstance(block.anchor, Unlocated) for block in blocks)
    reason = blocks[0].anchor.reason if isinstance(blocks[0].anchor, Unlocated) else ""
    assert "ambiguous heading path" in reason


async def test_inventing_an_anchor_for_an_ambiguous_path_fails_the_round_trip() -> None:
    """The guard above is load-bearing: with it removed, the harness goes red.

    A check nobody has watched fail is a check nobody knows works. This is the same parser
    with the admission replaced by a guess, and the round trip must reject it.
    """
    raw = raw_of(AMBIGUOUS, MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_GuessingParser(MarkdownConfig()), raw, fixture="ambiguous")


# --- block kinds -------------------------------------------------------------------------


async def test_a_fence_carries_the_language_from_its_info_string() -> None:
    """``lang`` is what lets a code chunk be split on its own terms rather than as prose."""
    blocks = await _blocks("```python\nvalue = 1\n```\n")
    assert blocks[0].kind is BlockKind.CODE
    assert blocks[0].lang == "python"


async def test_a_table_records_how_many_header_rows_it_has() -> None:
    """A table too large for one chunk repeats its header into every part.

    Recorded by the parser, which can see the delimiter row, rather than guessed downstream
    from the first row looking like a header — a part carrying the wrong rows mislabels every
    column in it.

    **``header_rows`` counts source lines, so an ordinary one-header table reports 2.** The
    delimiter is not a data row and a part beginning after it would not be a pipe table at all,
    so whatever the header-repeating split carries has to include it. ``rows`` is the source
    lines themselves, and without it the split never happened: a 300-row table was cut wherever
    the token budget landed, producing eight line fragments and four truncated glossary
    expansions stored against correct citations.
    """
    blocks = await _blocks("| Setting | Default |\n|---|---|\n| retries | 3 |\n")
    assert blocks[0].kind is BlockKind.TABLE
    assert blocks[0].metadata == {
        "header_rows": 2,
        "rows": ["| Setting | Default |", "|---|---|", "| retries | 3 |"],
    }
    assert blocks[0].text.splitlines() == blocks[0].metadata["rows"]


async def test_front_matter_does_not_become_a_heading_nobody_wrote() -> None:
    """Left in place, front matter is read as a setext heading by CommonMark.

    The first key becomes an ``h2`` and every heading path below it hangs off ``title: …``.
    The second half of this test is the evidence: with the option off, that is exactly what
    happens.
    """
    source = "---\ntitle: A page\n---\n\n# Real heading\n\nProse.\n"
    kept = await _blocks(source)
    assert [block.heading_path for block in kept][-1] == ("Real heading",)

    unstripped = MarkdownParser(MarkdownConfig(front_matter=False))
    paths = [
        block.heading_path for block in await read_blocks(unstripped, raw_of(source, MEDIA_TYPE))
    ]
    assert ("title: A page",) in paths


# --- MDX ---------------------------------------------------------------------------------


async def test_a_jsx_component_is_media_and_its_children_stay_markdown(corpus: Path) -> None:
    """A component invocation is markup: noise in a vector and nonsense in a citation.

    So the tag is a ``media`` block naming the component, and the Markdown between an opening
    and closing tag is parsed as the prose it is — including when the author wrote the tags
    tight against their children, which is the form that would otherwise be swallowed whole.
    """
    raw = raw_from(corpus / "markdown" / "components.mdx", MDX_MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    components = [
        block.metadata.get("component") for block in blocks if block.kind is BlockKind.MEDIA
    ]
    assert components == ["Callout", "Callout", "Note", "Note", "Chart"]
    prose = [block.text for block in blocks if block.kind is BlockKind.PROSE]
    assert "Tight components leave their children as Markdown even without blank lines." in prose


async def test_a_component_tag_inside_a_code_fence_stays_code(corpus: Path) -> None:
    """An MDX page documenting a component shows its tags, and showing is not invoking.

    Treating those lines as invocations would blank them out of the fence, so the page would
    lose the example it exists to give and gain ``media`` blocks for components nobody used.
    """
    raw = raw_from(corpus / "markdown" / "components.mdx", MDX_MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    fence = next(block for block in blocks if block.kind is BlockKind.CODE)
    assert fence.lang == "jsx"
    assert '<Banner tone="note">' in fence.text
    assert "Banner" not in [block.metadata.get("component") for block in blocks]


async def test_a_component_tag_in_a_plain_markdown_file_is_kept_as_text() -> None:
    """``.md`` is not ``.mdx``, and a line that looks like a tag there is text.

    Treating it as a component would drop the line from the index, on a guess about a file
    the author wrote as Markdown.
    """
    blocks = await _blocks('<Callout kind="warning">\n', MEDIA_TYPE)
    assert [block.kind for block in blocks] == [BlockKind.PROSE]


# --- degenerate input --------------------------------------------------------------------


async def test_an_empty_document_yields_no_blocks_rather_than_raising() -> None:
    """Nothing to parse is an outcome, not a failure.

    Raising would advance the fallback chain and end as ``failed``; yielding nothing ends as
    ``no_extractable_text``, which is the truthful record and the one that is re-indexable.
    """
    assert await _blocks("") == []


async def test_a_document_of_one_heading_still_produces_a_block() -> None:
    """A stub page is content, and a parser that returned nothing would report it as empty."""
    blocks = await _blocks("# A stub page\n")
    assert [block.text for block in blocks] == ["# A stub page"]


async def test_a_file_with_no_trailing_newline_keeps_its_last_line(corpus: Path) -> None:
    """The last line of a file without a final newline is still a line.

    Off-by-one arithmetic that assumes a trailing newline drops it, and the loss is invisible
    — the citation simply stops a sentence early.
    """
    raw = raw_from(corpus / "markdown" / "no-trailing-newline.md", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    assert blocks[-1].text.endswith("where a file usually ends.")


async def test_astral_plane_text_survives_a_heading_and_its_slug(corpus: Path) -> None:
    """Emoji and CJK-extension characters are two UTF-16 units and one character.

    Anything counting the wrong one truncates the heading, which changes the slug, which
    changes the fragment every citation into that section was written against.
    """
    raw = raw_from(corpus / "markdown" / "astral.md", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    heading = blocks[0]
    assert heading.heading_path == ("🚀 Launch checklist for 𠀋 builds",)
    assert isinstance(heading.anchor, HeadingAnchor)
    assert heading.anchor.fragment == "launch-checklist-for-𠀋-builds"


# --- resolution --------------------------------------------------------------------------


async def test_an_anchor_resolves_against_a_parser_that_has_never_seen_the_document(
    corpus: Path,
) -> None:
    """Resolution re-derives the location from the bytes, never from what parsing remembered.

    A parser that resolved out of its own memory would verify nothing: the round trip would
    be comparing the parser against itself, and would keep passing after the document changed
    underneath it.
    """
    raw = raw_from(corpus / "markdown" / "typical.md", MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    fresh: Parser = _parser()
    resolved = await fresh.resolve(blocks[-1].anchor, raw)
    assert resolved is not None
    assert blocks[-1].text in resolved


async def test_an_anchor_from_another_document_does_not_resolve() -> None:
    """An anchor that no longer fits its document is reported as unresolvable.

    Clamping to the nearest lines instead would hide a diverged citation behind a plausible
    quotation, which is precisely the failure that cannot be seen from the outside.
    """
    parser = _parser()
    raw = raw_of("# Short\n", MEDIA_TYPE)
    assert await parser.resolve(LineAnchor(start=40, end=41), raw) is None
    assert await parser.resolve(HeadingAnchor(path=("Absent",), fragment="absent"), raw) is None


async def test_anchoring_each_block_to_the_next_section_fails_the_round_trip(
    corpus: Path,
) -> None:
    """The harness catches the mistake it exists to catch.

    Shifting every anchor by one section leaves a parser whose output is well-formed in every
    respect a type checker or a schema can see. Only resolving the anchors finds it, which is
    why resolution is part of the protocol rather than a convention.
    """
    raw = raw_from(corpus / "markdown" / "structure.md", MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_NextSectionParser(MarkdownConfig()), raw, fixture="shifted")
