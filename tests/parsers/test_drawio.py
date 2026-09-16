"""The draw.io parser: two encodings, one PNG wrapper, and two payloads that must be refused.

A ``.drawio`` attachment is the one diagram source that arrives as a file rather than as a code
block, and that changes what can go wrong with it. A code block is text the page already holds;
an attachment is a compressed, attacker-controlled payload, and the two things this suite is
most concerned with are that both of draw.io's encodings are read and that neither of them can
be used to make the parser do unbounded work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.content import BlockKind
from manicule.core.errors import ParseError
from manicule.parsers import grammars
from manicule.parsers.config import DRAWIO_MEDIA_TYPE, DrawioConfig
from manicule.parsers.diagrams import reading
from manicule.parsers.drawio import DrawioParser
from manicule.parsers.expansion import media_type_for
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

pytestmark = pytest.mark.anyio


@pytest.fixture
def parser() -> DrawioParser:
    return DrawioParser(DrawioConfig())


@pytest.fixture
def diagrams(corpus: Path) -> Path:
    return corpus / "drawio"


async def blocks(parser: DrawioParser, path: Path):  # noqa: ANN201 - list[ParsedBlock], inferred
    return [block async for block in parser.parse(raw_from(path, DRAWIO_MEDIA_TYPE))]


# --- the two encodings ---------------------------------------------------------------------


async def test_a_compressed_diagram_is_read_into_its_model(
    parser: DrawioParser, diagrams: Path
) -> None:
    """The ordinary export: URL-encoded, raw-deflated, base64. Anything less is unreadable."""
    found = await blocks(parser, diagrams / "typical.drawio")

    assert len(found) == 1
    assert found[0].lang == "mxfile"
    assert found[0].kind is BlockKind.CODE
    assert "Auth Service" in found[0].text
    assert found[0].text.lstrip().startswith("<mxGraphModel")


async def test_an_uncompressed_diagram_is_read_without_guessing(
    parser: DrawioParser, diagrams: Path
) -> None:
    """draw.io writes plain XML when compression is off, and that is half of real exports."""
    found = await blocks(parser, diagrams / "uncompressed.drawio")

    assert len(found) == 1
    assert "<mxGraphModel" in found[0].text
    assert "Gateway" in found[0].text


async def test_a_diagram_body_in_neither_spelling_is_refused_rather_than_guessed_at(
    parser: DrawioParser,
) -> None:
    """A third encoding read as one of these two yields plausible mojibake, not a failure.

    Declining hands the document to the next parser in the chain and, if none takes it, leaves
    it visibly ``unsupported_media_type`` — which is a state an operator can act on.
    """
    raw = raw_of(
        '<mxfile><diagram id="a" name="A">not base64 and not xml !!!</diagram></mxfile>',
        DRAWIO_MEDIA_TYPE,
    )

    with pytest.raises(ParseError, match="neither XML nor base64"):
        [block async for block in DrawioParser(DrawioConfig()).parse(raw)]


# --- pages and anchors ---------------------------------------------------------------------


async def test_each_page_becomes_its_own_block(parser: DrawioParser, diagrams: Path) -> None:
    """A draw.io file is tabbed, and two tabs are two diagrams rather than one long one."""
    found = await blocks(parser, diagrams / "multi-page.drawio")

    assert [block.heading_path for block in found] == [("Architecture",), ("Pipeline",)]
    assert found[0].text != found[1].text


async def test_two_pages_with_one_name_stay_separately_addressable(
    parser: DrawioParser, diagrams: Path
) -> None:
    """Naming two tabs the same thing is ordinary, and it must not collapse two citations.

    The tie breaks in the fragment rather than in the path, so the first tab keeps the plain
    name a reader would cite and the second is still reachable.
    """
    found = await blocks(parser, diagrams / "repeated-names.drawio")

    assert [block.anchor for block in found] == [
        HeadingAnchor(path=("Overview",)),
        HeadingAnchor(path=("Overview",), fragment="2"),
    ]


async def test_a_page_with_no_name_and_no_id_is_unlocated_rather_than_numbered(
    parser: DrawioParser,
) -> None:
    """A tab's position is not a name the file gave it, and inventing one invents a citation.

    Built here rather than shipped in the corpus: draw.io writes an ``id`` on every diagram it
    saves, so a file with neither is hand-written, and putting one in a corpus whose ratios are
    meant to describe real exports would make the declared budget describe a bug instead.
    """
    raw = raw_of(
        "<mxfile><diagram><mxGraphModel><root>"
        '<mxCell id="0"/><mxCell id="a" value="Lonely" vertex="1" parent="0"/>'
        "</root></mxGraphModel></diagram></mxfile>",
        DRAWIO_MEDIA_TYPE,
    )

    found = [block async for block in parser.parse(raw)]

    assert len(found) == 1
    assert isinstance(found[0].anchor, Unlocated)
    assert found[0].heading_path == ()
    assert await parser.resolve(found[0].anchor, raw) is None


async def test_a_page_with_an_empty_body_contributes_nothing(
    parser: DrawioParser, diagrams: Path
) -> None:
    """An empty tab is a real thing to save, and an empty block would be a citable nothing."""
    assert await blocks(parser, diagrams / "empty.drawio") == []


# --- the PNG wrapper -----------------------------------------------------------------------


async def test_a_drawio_png_is_read_from_its_embedded_source(
    parser: DrawioParser, diagrams: Path
) -> None:
    """``.drawio.png`` renders anywhere *and* round-trips, because the XML rides along in it."""
    found = await blocks(parser, diagrams / "typical.drawio.png")

    assert len(found) == 1
    assert "Auth Service" in found[0].text


async def test_a_png_carrying_no_diagram_says_so(parser: DrawioParser) -> None:
    """A picture of a diagram is not a diagram, and must not be indexed as an empty one."""
    picture = b"\x89PNG\r\n\x1a\n" + (0).to_bytes(4, "big") + b"IEND" + (0).to_bytes(4, "big")
    raw = raw_of(picture, DRAWIO_MEDIA_TYPE)

    with pytest.raises(ParseError, match="no embedded mxfile"):
        [block async for block in parser.parse(raw)]


def test_a_compound_suffix_routes_to_the_diagram_rather_than_to_the_image() -> None:
    """Resolving ``.drawio.png`` on its last suffix throws away the diagram inside it."""
    assert media_type_for("architecture.drawio.png") == DRAWIO_MEDIA_TYPE
    assert media_type_for("architecture.drawio") == DRAWIO_MEDIA_TYPE
    assert media_type_for("screenshot.png") != DRAWIO_MEDIA_TYPE


# --- the two refusals ----------------------------------------------------------------------


async def test_a_payload_that_expands_past_the_ceiling_is_refused_while_it_expands(
    parser: DrawioParser, diagrams: Path
) -> None:
    """The bound is on what the payload expands to, enforced as it expands.

    ``docs/parsing.md`` §9.3 settles this for zip members and it is the same threat here: a
    declared size is a field the attacker wrote, and by the time ``len()`` can be taken the
    memory is already spent. Refused under the shipped default, not a test-only ceiling.
    """
    with pytest.raises(ParseError, match="expands past the"):
        await blocks(parser, diagrams / "expansion-bomb.drawio")


async def test_the_ceiling_is_configuration_and_moving_it_moves_the_refusal(
    diagrams: Path,
) -> None:
    """An operator with larger diagrams can raise it; the refusal is not a constant."""
    tight = DrawioParser(DrawioConfig(max_decompressed_bytes=32))

    with pytest.raises(ParseError, match="expands past the 32-byte ceiling"):
        [
            block
            async for block in tight.parse(raw_from(diagrams / "typical.drawio", DRAWIO_MEDIA_TYPE))
        ]


async def test_a_doctype_is_refused_before_entities_can_be_expanded(
    parser: DrawioParser, diagrams: Path
) -> None:
    """Entity expansion has no ceiling available to stop it, so the declaration is refused.

    An mxfile has no legitimate use for a document type declaration, which is what makes
    refusing the declaration a cheaper and more certain answer than bounding its consequences.
    """
    with pytest.raises(ParseError, match="DOCTYPE"):
        await blocks(parser, diagrams / "doctype.drawio")


# --- the reading -----------------------------------------------------------------------------


async def test_the_relationships_reach_the_embedder_through_the_diagram_reader(
    parser: DrawioParser, diagrams: Path
) -> None:
    """Same reading as Graphviz and mermaid get, which is why it is one reader table.

    The edge resolves through to both labels, and the box nothing connects to is still
    reported — a diagram of boxes with no edges states something, and its labels are all of it.
    """
    found = await blocks(parser, diagrams / "typical.drawio")

    said = reading("mxfile", found[0].text, budget=4000, max_statements=64)

    assert said is not None
    assert "Auth Service → Token Store: validates against" in said
    assert "Audit Log" in said


async def test_a_wrapped_cell_keeps_its_label_and_is_not_read_twice(
    parser: DrawioParser, diagrams: Path
) -> None:
    """``<object>`` owns the label when a shape has custom fields, and owns the cell inside it.

    Both halves are defects waiting to happen: reading only ``mxCell`` loses every label on a
    shape with custom properties, and reading the wrapper *and* its cell draws every such edge
    twice.
    """
    found = await blocks(parser, diagrams / "uncompressed.drawio")

    said = reading("mxfile", found[0].text, budget=4000, max_statements=64)

    assert said is not None
    assert said.count("routes to") == 1
    assert "Gateway edge → Core & friends: routes to" in said


async def test_a_reading_needs_no_grammar_and_so_no_grammar_bundle(
    parser: DrawioParser, diagrams: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """draw.io is XML this parser already decoded, so tree-sitter is never consulted for it.

    Worth pinning: a missing grammar bundle is a real state on a machine, and letting it
    silence a reading that never wanted a grammar would make an unrelated install step decide
    whether draw.io diagrams embed as relationships.
    """
    found = await blocks(parser, diagrams / "typical.drawio")
    asked: list[str] = []

    def refuse(language: str) -> object:
        asked.append(language)
        raise AssertionError(language)

    monkeypatch.setattr(grammars, "load_parser", refuse)
    said = reading("mxfile", found[0].text, budget=4000, max_statements=64)

    assert said is not None
    assert asked == []


# --- the shipped obligations -------------------------------------------------------------


async def test_every_fixture_round_trips(parser: DrawioParser, diagrams: Path) -> None:
    """Anchors resolve to the text their blocks claim, over the whole corpus at once."""
    readable = (
        "typical.drawio",
        "multi-page.drawio",
        "uncompressed.drawio",
        "repeated-names.drawio",
    )

    await check_corpus(
        parser,
        [raw_from(diagrams / name, DRAWIO_MEDIA_TYPE) for name in readable],
        min_blocks=4,
    )


async def test_the_parser_contract_holds_for_a_repeated_page_name(
    parser: DrawioParser, diagrams: Path
) -> None:
    """The fragment tie-break has to survive the contract, not merely look distinct."""
    await check_fixture(parser, raw_from(diagrams / "repeated-names.drawio", DRAWIO_MEDIA_TYPE))
