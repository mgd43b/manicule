"""DOCX parsing: sections are the location, and there are no pages.

The two assertions worth reading first are
:func:`test_a_docx_never_reports_a_page_number`, which is the user-visible limitation
``docs/parsing.md`` §2.5 insists on stating rather than papering over, and
:func:`test_shifting_every_anchor_one_section_along_fails_the_round_trip`, which shows the
round-trip harness catching the defect it exists for: anchors that are well-formed, plausible
and one section out.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import Anchor, HeadingAnchor, PageAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, read_blocks
from manicule.parsers.word import WORD_MEDIA_TYPE, WORD_MEDIA_TYPES, WordConfig, WordParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

HARNESS_FIXTURES = (
    "word_typical.docx",
    "word_structurally_hard.docx",
    "word_untitled_preamble.docx",
    "word_degenerate_heading_only.docx",
    "word_degenerate_empty.docx",
    "word_hostile_astral.docx",
    "word-large.docx",
)
"""Every fixture the six assertions run over.

``word_repeated_heading_path.docx`` is deliberately absent. Its sections share their heading
*text*, so one section's resolved span contains the other's whole heading block, and assertion
3 of ``docs/parsing.md`` §3.3 cannot tell that apart from an anchor pointing at the wrong
section. What that fixture is for is asserted directly in
:func:`test_two_sections_with_one_heading_path_and_no_fragment_are_unlocated`.
"""

DECLINED_FIXTURES = (
    "word_degenerate_zero_bytes.docx",
    "word_hostile_truncated.docx",
    "word_hostile_plain_zip.docx",
)


def _parser(config: WordConfig | None = None) -> WordParser:
    return WordParser(config or WordConfig())


async def _blocks(path: Path, config: WordConfig | None = None) -> list[ParsedBlock]:
    return await read_blocks(_parser(config), raw_from(path, WORD_MEDIA_TYPE))


def _fragment_of(block: ParsedBlock) -> str | None:
    assert isinstance(block.anchor, HeadingAnchor), f"expected a heading anchor, got {block.anchor}"
    return block.anchor.fragment


async def _resolve(path: Path, anchor: Anchor) -> str | None:
    return await _parser().resolve(anchor, raw_from(path, WORD_MEDIA_TYPE))


# --- no pages, ever ----------------------------------------------------------------------


async def test_a_docx_never_reports_a_page_number(corpus: Path) -> None:
    """A `.docx` stores a flow of paragraphs; pages are produced by a layout engine.

    Pagination depends on the fonts, the printer metrics and the Word version, none of which
    are in the file — so any page number reported here would be invented, and an invented
    number is exactly the citation this project exists not to produce. The structurally hard
    fixture even contains an explicit page break, which is a lower bound on the page count and
    still not a pagination.
    """
    for name in HARNESS_FIXTURES:
        for block in await _blocks(corpus / "word" / name):
            assert not isinstance(block.anchor, PageAnchor), (
                f"{name}: a DOCX block carries a page anchor, which cannot have been measured"
            )
            assert "page" not in block.metadata, f"{name}: block metadata claims a page"


# --- the six assertions ------------------------------------------------------------------


async def test_the_corpus_round_trips_and_stays_inside_its_location_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, blocks and chunks, plus the corpus-wide unlocated ceiling.

    Chunked as well as parsed, because a chunk's anchor is a merge of its blocks' anchors and a
    merge is exactly where a location widens without anything raising.
    """
    raws = [raw_from(corpus / "word" / name, WORD_MEDIA_TYPE) for name in HARNESS_FIXTURES]
    reports = await check_corpus(_parser(), raws, chunker=chunker, min_blocks=200)
    assert sum(report.chunks for report in reports) > 0


async def test_shifting_every_anchor_one_section_along_fails_the_round_trip(corpus: Path) -> None:
    """The guard is load-bearing: a plausible, well-formed, wrong anchor must be caught.

    Nothing raises in the parser that does this. Every anchor is a real section of the real
    document, every fragment resolves, and every citation quotes text from the wrong place —
    which is the failure mode the six assertions exist for, and the reason this suite has a
    parser that commits it on purpose.
    """
    raw = raw_from(corpus / "word" / "word_typical.docx", WORD_MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_ShiftedSectionParser(), raw, fixture="shifted")


# --- sections resolve to themselves ------------------------------------------------------


async def test_a_section_resolves_up_to_the_next_heading_and_not_into_its_subsections(
    corpus: Path,
) -> None:
    """A section is its own content, not its content plus every section beneath it.

    Resolving into subsections would make a top-level section resolve to most of the document:
    the tightness bound in ``docs/parsing.md`` §3.3 fails on any nested document, and a
    citation of an introduction quotes the whole chapter.
    """
    path = corpus / "word" / "word_typical.docx"
    blocks = await _blocks(path)
    deployment = next(block for block in blocks if block.text == "Deployment")
    resolved = await _resolve(path, deployment.anchor)

    assert resolved is not None
    assert "Cut a release branch" in resolved
    assert "Demote the previous artifact" not in resolved, (
        "the Deployment section resolved into its Rollback subsection"
    )


async def test_headings_whose_slugs_collide_get_distinct_fragments(corpus: Path) -> None:
    """ "Roll-back" and "Roll back" slugify to the same string, and must not share a fragment.

    A fragment is the resolution key (``docs/parsing.md`` §2.3). Two sections sharing one
    address means following it lands on whichever the parser happened to see first, so the
    second occurrence gets ``-1`` — counted from the second, so the section that was there
    first keeps the name it already had.
    """
    path = corpus / "word" / "word_structurally_hard.docx"
    blocks = await _blocks(path)
    hyphenated = next(block for block in blocks if block.text == "Roll-back")
    spaced = next(block for block in blocks if block.text == "Roll back")

    assert _fragment_of(hyphenated) == "roll-back"
    assert _fragment_of(spaced) == "roll-back-1"

    first = await _resolve(path, hyphenated.anchor)
    second = await _resolve(path, spaced.anchor)
    assert first is not None
    assert second is not None
    assert "network policy generation" in first
    assert "routing table generation" in second


async def test_two_sections_with_one_heading_path_and_no_fragment_are_unlocated(
    corpus: Path,
) -> None:
    """An address that names two places names neither, and the reason has to be actionable.

    Two headings of ``🌏`` slugify to nothing, so no fragment can be synthesized and the path
    is the only address — and it repeats. The blocks are ``Unlocated`` with a reason naming the
    path, the number of places it hits, and what to change. The two ``Configuration`` sections
    in the same fixture are the contrast: identical paths, but sluggable text, so they are
    told apart by their fragments and stay citable.
    """
    path = corpus / "word" / "word_repeated_heading_path.docx"
    blocks = await _blocks(path)

    unlocated = [block for block in blocks if isinstance(block.anchor, Unlocated)]
    assert [block.text for block in unlocated] == [
        "🌏",
        "First unsluggable section.",
        "🌏",
        "Second unsluggable section.",
    ]
    reason = unlocated[0].anchor.reason if isinstance(unlocated[0].anchor, Unlocated) else ""
    assert "addresses 2 places" in reason
    assert "rename" in reason, f"the reason is not actionable: {reason!r}"

    configuration = [block for block in blocks if block.text == "Configuration"]
    assert [_fragment_of(block) for block in configuration] == ["configuration", "configuration-1"]
    first = await _resolve(path, configuration[0].anchor)
    assert first is not None
    assert "First configuration section." in first
    assert "Second configuration section." not in first


# --- content before the first heading ----------------------------------------------------


async def test_content_before_the_first_heading_is_addressed_by_the_document_title(
    corpus: Path,
) -> None:
    """A preamble has no heading, so the only honest address is the document itself.

    Its ``heading_path`` stays empty: the title is not a heading, and the breadcrumb already
    carries the document title, so putting it in the path would either stutter or claim a
    section nobody wrote.
    """
    path = corpus / "word" / "word_typical.docx"
    raw = raw_of(path.read_bytes(), WORD_MEDIA_TYPE, uri=path.name)
    blocks = await read_blocks(_parser(), raw)
    preamble = blocks[0]

    assert preamble.text.startswith("This runbook covers")
    assert preamble.anchor == HeadingAnchor(path=("Deployment Runbook",), fragment=None), (
        "the preamble is addressed by the title the file declares in its core properties"
    )
    assert preamble.heading_path == ()

    resolved = await _parser().resolve(preamble.anchor, raw)
    assert resolved == "This runbook covers the release train and who to wake."


async def test_content_before_the_first_heading_is_unlocated_when_nothing_names_the_document(
    corpus: Path,
) -> None:
    """No heading and no title means no address, and ``Unlocated`` says so with a remedy.

    This is what the 0.05 unlocated budget in ``docs/parsing.md`` §3.4 is spent on. It is a
    property of the document rather than a shortcoming of the parser, which is why the reason
    names both things the author could change.
    """
    path = corpus / "word" / "word_untitled_preamble.docx"
    raw = raw_of(path.read_bytes(), WORD_MEDIA_TYPE, uri=path.name)
    blocks = await read_blocks(_parser(), raw)

    assert isinstance(blocks[0].anchor, Unlocated)
    reason = blocks[0].anchor.reason
    assert "no title" in reason
    assert "heading above this content" in reason
    assert await _parser().resolve(blocks[0].anchor, raw) is None


# --- tables and lists --------------------------------------------------------------------


async def test_a_table_is_one_block_that_describes_its_own_rows(corpus: Path) -> None:
    """A table is atomic, and the chunker needs its rows to split it without severing one.

    ``metadata["rows"]`` is the rendered lines rather than a count, because that is the
    boundary a row split cuts at; ``header_rows`` is what gets repeated into every part
    (``docs/parsing.md`` §4.2). A count would leave the chunker guessing where a row ends,
    and a guess there produces half a row with a header on top of it.
    """
    blocks = await _blocks(corpus / "word" / "word_typical.docx")
    table = next(block for block in blocks if block.kind is BlockKind.TABLE)

    rows = table.metadata["rows"]
    assert isinstance(rows, list)
    assert "\n".join(str(row) for row in rows) == table.text
    assert table.metadata["header_rows"] == 1
    assert table.metadata["column_count"] == 3
    assert table.text.startswith("Step\tOwner\tTimeout")


async def test_a_merged_header_cell_is_reported_for_every_column_it_spans(corpus: Path) -> None:
    """A merged cell stores its text once and covers the grid positions it spans.

    Repeating it across the span is what the reader sees, and it is what lets a part split off
    the bottom of the table carry the header that applies to *its* columns rather than a blank.
    """
    blocks = await _blocks(corpus / "word" / "word_structurally_hard.docx")
    merged = next(block for block in blocks if block.text.startswith("Quarter"))
    assert merged.text.splitlines()[0] == "Quarter\tQuarter\tRegion\tOwner"


async def test_a_run_of_list_paragraphs_is_one_block_carrying_its_nesting_levels(
    corpus: Path,
) -> None:
    """A bulleted list is one structure, and its depth comes from the style, not indentation.

    One block per bullet would hand the chunker a boundary at every item, where §4.2 asks it
    to split at top-level items. The levels are read off the style name — ``List Bullet 3`` —
    because indentation is a rendering property a template can set to anything.
    """
    blocks = await _blocks(corpus / "word" / "word_structurally_hard.docx")
    nested = next(
        block
        for block in blocks
        if block.kind is BlockKind.LIST and block.text.startswith("Depth 1 item")
    )
    assert nested.metadata["list_levels"] == [1, 2, 3, 4, 5]
    assert len(nested.text.splitlines()) == 5


async def test_a_style_declared_as_a_heading_becomes_one(corpus: Path) -> None:
    """A house template naming its own heading styles has heading structure to recover.

    Without the declaration the parser sees ``List Bullet 4`` and treats it as a list item,
    which is the honest reading of an undeclared style. With it, the paragraph opens a section
    — and that is a configuration decision rather than a guess about the template.
    """
    path = corpus / "word" / "word_structurally_hard.docx"
    plain = await _blocks(path)
    declared = await _blocks(path, WordConfig(extra_heading_styles={"List Bullet 4": 2}))

    assert not any(
        block.kind is BlockKind.HEADING and block.text.startswith("Depth 4") for block in plain
    )
    assert any(
        block.kind is BlockKind.HEADING and block.text.startswith("Depth 4") for block in declared
    )


# --- astral text -------------------------------------------------------------------------


async def test_astral_text_survives_into_the_block_and_into_its_fragment(corpus: Path) -> None:
    """Codepoints above the basic multilingual plane are content, including in a heading.

    A slug keeps the word characters, which under Unicode matching includes CJK and
    mathematical letters, and drops the symbols — so an emoji-only heading has no fragment at
    all and falls back to its path.
    """
    path = corpus / "word" / "word_hostile_astral.docx"
    blocks = await _blocks(path)

    heading = blocks[0]
    assert heading.text == "配置 𝔘nicode"  # noqa: RUF001 - the astral letter is the assertion
    assert _fragment_of(heading) == "配置-𝔘nicode"  # noqa: RUF001 - same
    section = await _resolve(path, heading.anchor) or ""
    assert "🜁" in section

    emoji_heading = next(block for block in blocks if block.text == "🌏🌍🌎")
    assert _fragment_of(emoji_heading) is None, (
        "an emoji heading has nothing sluggable, so inventing a fragment would invent an address"
    )
    assert emoji_heading.anchor == HeadingAnchor(path=("🌏🌍🌎",), fragment=None)


# --- declines and empties ----------------------------------------------------------------


@pytest.mark.parametrize("name", DECLINED_FIXTURES)
async def test_an_unreadable_package_is_declined_rather_than_indexed(
    corpus: Path, name: str
) -> None:
    """Declining lets the next parser in the chain try; indexing wreckage cites nothing.

    Zero bytes, a truncated ``word/document.xml`` and a plain zip under a `.docx` name are
    three different ways for the input not to be ours, and all three have to arrive as
    ``ParseError`` naming what was expected rather than as a stack trace or an empty parse.
    """
    with pytest.raises(ParseError, match=r"not a readable \.docx package"):
        await _blocks(corpus / "word" / name)


async def test_an_empty_document_yields_no_blocks_and_does_not_raise(corpus: Path) -> None:
    """ "Nothing to extract" and "the parser broke" must stay different outcomes.

    They lead to different statuses and different remedies — ``no_extractable_text`` is a
    question about the document, ``failed`` is a bug report — so an empty but well-formed
    package parses to nothing at all, quietly.
    """
    assert await _blocks(corpus / "word" / "word_degenerate_empty.docx") == []


async def test_a_single_heading_is_a_section_of_one_block(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """A stub page is a real document, and it still has to round-trip.

    Its only block is its heading, which means the resolved section and the block text are the
    same string — the tightest case there is, and the one where an off-by-one in the section
    boundary would show up as an empty resolve.
    """
    raw = raw_from(corpus / "word" / "word_degenerate_heading_only.docx", WORD_MEDIA_TYPE)
    report = await check_fixture(_parser(), raw, chunker=chunker)
    assert report.blocks == 1


class _ShiftedSectionParser:
    """Anchors every block to the section after the one it came from.

    The house pattern from ``tests/fakes.py``: a parser that breaks the rule, so the guard can
    be watched failing. This one is not obviously wrong from the outside — every anchor names a
    real section of a real document and resolves cleanly — which is precisely why the
    discrimination assertion has to be the thing that catches it.
    """

    media_types = WORD_MEDIA_TYPES
    profile = WordParser.profile

    def __init__(self) -> None:
        self._real = WordParser(WordConfig())

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        blocks = await read_blocks(self._real, raw)
        anchors = list(dict.fromkeys(block.anchor for block in blocks))
        shifted = {
            anchor: anchors[(index + 1) % len(anchors)] for index, anchor in enumerate(anchors)
        }
        for block in blocks:
            yield block.model_copy(update={"anchor": shifted[block.anchor]})

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        return await self._real.resolve(anchor, raw)


_PARSERS: Sequence[Parser] = (WordParser(WordConfig()), _ShiftedSectionParser())
"""Type-checked conformance: both satisfy the protocol, so the fake is a parser and not a
mock of one."""
