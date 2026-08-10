"""Jupyter parsing: the heading tree, the cell id, and the version that decides which.

Cell ids arrived in nbformat 4.5, so this format has two addressing regimes in one file type
and the tests are organised around that: with ids, a block is addressed by ``(path, cell id)``;
without them the path is all there is, and where the path repeats there is no address at all.
:func:`test_a_repeated_path_without_cell_ids_is_unlocated_and_says_what_fixes_it` is the one to
read — the reason has to tell the reader that re-saving the notebook restores exact addressing,
because that is a thing they can do.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, parsing, read_blocks
from manicule.parsers.notebook import NOTEBOOK_MEDIA_TYPE, NotebookConfig, NotebookParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, raw_from

HARNESS_FIXTURES = (
    "notebook_typical.ipynb",
    "notebook_structurally_hard.ipynb",
    "notebook_degenerate_heading_only.ipynb",
    "notebook_degenerate_no_cells.ipynb",
    "notebook_hostile_astral.ipynb",
    "notebook_hostile_no_cell_ids.ipynb",
    "notebook-large.ipynb",
)
"""Every fixture the six assertions run over.

``notebook_hostile_repeated_path.ipynb`` is deliberately absent: its two sections share their
heading *text*, so one section's resolved span contains the other's whole heading block, and
assertion 3 of ``docs/parsing.md`` §3.3 cannot tell that apart from a misplaced anchor. Every
block in it is ``Unlocated`` in any case, which is asserted directly instead.
"""


def _parser(config: NotebookConfig | None = None) -> NotebookParser:
    return NotebookParser(config or NotebookConfig())


async def _blocks(path: Path, config: NotebookConfig | None = None) -> list[ParsedBlock]:
    return await read_blocks(_parser(config), raw_from(path, NOTEBOOK_MEDIA_TYPE))


def _heading_anchor(block: ParsedBlock) -> HeadingAnchor:
    assert isinstance(block.anchor, HeadingAnchor), f"expected a heading anchor: {block.anchor}"
    return block.anchor


async def _resolve(path: Path, anchor: Anchor) -> str | None:
    return await _parser().resolve(anchor, raw_from(path, NOTEBOOK_MEDIA_TYPE))


# --- addressing ---------------------------------------------------------------------------


async def test_a_cell_id_is_the_fragment_when_the_notebook_has_them(corpus: Path) -> None:
    """``cell-<id>`` addresses one cell, where a heading path addresses a section.

    The id is the address the file itself defines, which is what ``docs/parsing.md`` §2.3 asks
    for: where the source publishes an address, a citation uses that one rather than one we
    invented, because ours would not survive the document being edited around it.
    """
    path = corpus / "notebook" / "notebook_typical.ipynb"
    blocks = await _blocks(path)

    assert _heading_anchor(blocks[0]).fragment == "cell-intro"
    assert _heading_anchor(blocks[0]).path == ("Retry budget analysis",)

    code = next(block for block in blocks if block.kind is BlockKind.CODE)
    assert _heading_anchor(code).fragment == "cell-load"
    assert _heading_anchor(code).path == ("Retry budget analysis",)

    resolved = await _resolve(path, code.anchor)
    assert resolved is not None
    assert "read_metrics" in resolved
    assert "Establishing where the retries go." not in resolved, (
        "a cell anchor resolves to its cell, not to the section around it"
    )


async def test_a_notebook_below_45_has_no_fragment_and_is_addressed_by_its_path(
    corpus: Path,
) -> None:
    """Cell ids arrived in nbformat 4.5. Below it, inventing one would invent an address.

    Converting the notebook to 4.5 would generate ids — so every citation into it would point
    at an address created on whichever machine ran the conversion, and two machines would
    disagree. The parser reads the file as it is and addresses the section.
    """
    path = corpus / "notebook" / "notebook_hostile_no_cell_ids.ipynb"
    blocks = await _blocks(path)

    assert all(_heading_anchor(block).fragment is None for block in blocks)
    assert blocks[0].metadata["nbformat"] == "4.4"

    section = _heading_anchor(blocks[0])
    resolved = await _resolve(path, section)
    assert resolved is not None
    assert "legacy_total = 7" in resolved, (
        "with no cell ids the section is the address, so it covers the cells inside it"
    )
    assert "Legacy appendix" not in resolved


async def test_a_repeated_path_without_cell_ids_is_unlocated_and_says_what_fixes_it(
    corpus: Path,
) -> None:
    """No fragment and a path that names two places is no address at all.

    ``docs/parsing.md`` §2.5 asks for the reason to say that upgrading the notebook fixes it,
    because it does: re-saving from Jupyter assigns cell ids and every block becomes citable.
    A reason of "unknown" would leave a reader with a document that cannot be cited and no idea
    why.
    """
    path = corpus / "notebook" / "notebook_hostile_repeated_path.ipynb"
    blocks = await _blocks(path)

    setup = [
        block
        for block in blocks
        if "installation walkthrough" in block.text or block.text == "Setup"
    ]
    teardown = [block for block in blocks if block.text.startswith("How to remove")]
    assert len(setup) == 4, "both Setup sections and their prose take part"
    assert all(isinstance(block.anchor, Unlocated) for block in setup)
    # The section between them is addressed perfectly well: only the repeated path is ambiguous,
    # and the refusal has to be that narrow or it would refuse whole notebooks at a time.
    assert all(isinstance(block.anchor, HeadingAnchor) for block in teardown)

    anchor = setup[0].anchor
    assert isinstance(anchor, Unlocated)
    assert "predates cell ids" in anchor.reason
    assert "nbformat 4.5 or later" in anchor.reason
    assert "names 2 places" in anchor.reason
    assert await _resolve(path, anchor) is None


async def test_two_sections_in_one_markdown_cell_resolve_separately(corpus: Path) -> None:
    """A cell id alone is not a unique address, because one cell can open two sections.

    Both sections carry the same fragment, so resolution matches the path as well — which makes
    the anchor tighter than the cell rather than looser. Resolving by fragment alone would make
    a citation of the first section quote the second, and it would fail the tightness bound at
    ``k=1.2`` on any cell like this one.
    """
    path = corpus / "notebook" / "notebook_structurally_hard.ipynb"
    blocks = await _blocks(path)

    model = next(block for block in blocks if block.text == "Capacity model")
    inputs = next(block for block in blocks if block.text == "Inputs")
    assert (
        _heading_anchor(model).fragment == _heading_anchor(inputs).fragment == "cell-two-sections"
    )
    assert _heading_anchor(model).path == ("Capacity model",)
    assert _heading_anchor(inputs).path == ("Capacity model", "Inputs")

    first = await _resolve(path, model.anchor)
    second = await _resolve(path, inputs.anchor)
    assert first is not None
    assert second is not None
    assert "The model runs weekly." in first
    assert "Headroom, growth rate" not in first
    assert "Headroom, growth rate" in second


async def test_a_hash_inside_a_fenced_block_is_not_a_heading(corpus: Path) -> None:
    """``# not a heading`` inside a fence is a comment in whatever the fence holds.

    Reading it as a heading would put it into the path of every cell below it, and the path
    reaches the embedder through the breadcrumb — so the mistake does not merely mislabel one
    block, it moves the vectors of everything after it.
    """
    blocks = await _blocks(corpus / "notebook" / "notebook_structurally_hard.ipynb")

    assert not any(
        block.kind is BlockKind.HEADING and "not a heading" in block.text for block in blocks
    )
    fenced = next(block for block in blocks if "not a heading, a comment" in block.text)
    assert fenced.kind is BlockKind.PROSE
    assert all("not a heading" not in element for block in blocks for element in block.heading_path)


# --- code and outputs ---------------------------------------------------------------------


async def test_a_code_cell_is_a_code_block_tagged_with_the_kernel_language(corpus: Path) -> None:
    """The language comes from the notebook's own metadata, never from the code.

    ``None`` rather than "python" when the file does not say: a wrong language tag sends a block
    to the wrong syntax highlighter and, later, the wrong grammar.
    """
    blocks = await _blocks(corpus / "notebook" / "notebook_typical.ipynb")
    code = [block for block in blocks if block.kind is BlockKind.CODE]

    assert code
    assert all(block.lang == "python" for block in code)
    assert all(block.metadata["cell_type"] == "code" for block in code)


async def test_text_outputs_are_content_and_image_only_outputs_are_counted(corpus: Path) -> None:
    """What a reader sees under the code is frequently the only statement of the result.

    Stream output and ``text/plain`` results become a prose block on the same cell. An output
    that is only an image contributes no text — optical character recognition is out of scope —
    and is counted in metadata rather than becoming a block with no text, because a block with
    no text is not a block and a chunk of it would cite nothing.
    """
    typical = await _blocks(corpus / "notebook" / "notebook_typical.ipynb")
    output = next(block for block in typical if block.metadata.get("cell_output") is True)
    assert output.kind is BlockKind.PROSE
    assert "Loaded 41 820 rows" in output.text
    assert "(41820, 7)" in output.text

    hard = await _blocks(corpus / "notebook" / "notebook_structurally_hard.ipynb")
    plot = next(block for block in hard if "plot_headroom" in block.text)
    assert plot.metadata["image_outputs"] == 1
    assert not any(block.text.strip() == "" for block in hard)

    error = next(block for block in hard if "OverflowError" in block.text)
    assert "horizon beyond the fitted range" in error.text


async def test_outputs_can_be_left_out_by_configuration(corpus: Path) -> None:
    """A notebook whose outputs are megabytes of logging is a real notebook, and a real setting.

    Turning them off changes what is indexed, not where anything is: the code blocks keep the
    anchors they had.
    """
    path = corpus / "notebook" / "notebook_typical.ipynb"
    without = await _blocks(path, NotebookConfig(include_outputs=False))

    assert not any(block.metadata.get("cell_output") for block in without)
    assert not any("Loaded 41 820 rows" in block.text for block in without)
    assert any(block.kind is BlockKind.CODE for block in without)


async def test_a_raw_cell_is_prose_unless_it_is_excluded(corpus: Path) -> None:
    """A raw cell holds content a conversion step consumes, which is prose to a query."""
    path = corpus / "notebook" / "notebook_structurally_hard.ipynb"
    included = await _blocks(path)
    raw_cell = next(block for block in included if block.metadata.get("cell_type") == "raw")
    assert raw_cell.kind is BlockKind.PROSE
    assert "author: platform team" in raw_cell.text

    excluded = await _blocks(path, NotebookConfig(include_raw_cells=False))
    assert not any(block.metadata.get("cell_type") == "raw" for block in excluded)


async def test_a_markdown_heading_builds_the_heading_path_and_nothing_else_does(
    corpus: Path,
) -> None:
    """Only markdown headings set the path. A code comment is not a heading.

    The deep list is the check that a level-2 heading replaces its sibling rather than nesting
    under it: ``## Assumptions`` after ``## Inputs`` is a sibling, and a stack that pushed
    instead of replacing would bury half the notebook under "Inputs".
    """
    blocks = await _blocks(corpus / "notebook" / "notebook_structurally_hard.ipynb")
    assumptions = next(block for block in blocks if block.text.startswith("- Level one"))

    assert assumptions.heading_path == ("Capacity model", "Assumptions")
    assert "Level five assumption" in assumptions.text


# --- the six assertions -------------------------------------------------------------------


async def test_the_corpus_round_trips_and_stays_inside_its_location_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, blocks and chunks, plus the 0.05 unlocated ceiling.

    What the ceiling pays for is a property of the file rather than of the parser: a notebook
    below 4.5 whose heading path repeats, and cells before the first heading in a notebook with
    no title.
    """
    raws = [raw_from(corpus / "notebook" / name, NOTEBOOK_MEDIA_TYPE) for name in HARNESS_FIXTURES]
    reports = await check_corpus(_parser(), raws, chunker=chunker, min_blocks=150)
    assert sum(report.chunks for report in reports) > 0


async def test_shifting_every_anchor_one_cell_along_fails_the_round_trip(corpus: Path) -> None:
    """The guard is load-bearing: every anchor is real, resolvable, and one cell out.

    Nothing raises. Every fragment names a cell the notebook has and every path names a section
    it has; only the correspondence between the text and the address is broken, which is why it
    is tested rather than reviewed.
    """
    raw = raw_from(corpus / "notebook" / "notebook_typical.ipynb", NOTEBOOK_MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_ShiftedCellParser(), raw, fixture="shifted")


# --- declines and empties -----------------------------------------------------------------


async def test_nbformat_three_is_declined_rather_than_converted(corpus: Path) -> None:
    """Converting would generate the cell ids the file does not have.

    That is the whole argument: a conversion invents addresses, and a citation into an invented
    address is worse than a document that says out loud it needs upgrading first. So the message
    names the command to run.
    """
    with pytest.raises(ParseError, match=r"nbformat 3\.0 is not readable here"):
        await _blocks(corpus / "notebook" / "notebook_hostile_version_three.ipynb")

    with pytest.raises(ParseError, match="jupyter nbconvert"):
        await _blocks(corpus / "notebook" / "notebook_hostile_version_three.ipynb")


@pytest.mark.parametrize(
    "name", ["notebook_hostile_not_json.ipynb", "notebook_degenerate_zero_bytes.ipynb"]
)
async def test_something_that_is_not_a_notebook_is_declined(corpus: Path, name: str) -> None:
    """Declining lets the next parser in the chain try; indexing wreckage cites nothing."""
    with pytest.raises(ParseError, match="not a readable notebook"):
        await _blocks(corpus / "notebook" / name)


async def test_a_notebook_with_no_cells_yields_no_blocks_and_does_not_raise(corpus: Path) -> None:
    """An empty notebook is empty, not broken, and the two lead to different remedies."""
    assert await _blocks(corpus / "notebook" / "notebook_degenerate_no_cells.ipynb") == []


async def test_a_single_heading_cell_round_trips(corpus: Path, chunker: StructuralChunker) -> None:
    """A stub notebook is a real document. Its resolved section is exactly its one block."""
    raw = raw_from(
        corpus / "notebook" / "notebook_degenerate_heading_only.ipynb", NOTEBOOK_MEDIA_TYPE
    )
    report = await check_fixture(_parser(), raw, chunker=chunker)
    assert report.blocks == 1


async def test_astral_text_survives_into_the_heading_the_fragment_and_the_output(
    corpus: Path,
) -> None:
    """Codepoints above the basic multilingual plane are content everywhere they appear."""
    path = corpus / "notebook" / "notebook_hostile_astral.ipynb"
    blocks = await _blocks(path)

    assert blocks[0].text == "解析 𝔄nalysis"  # noqa: RUF001 - the astral letter is the assertion
    assert "🜄" in blocks[1].text
    assert any("🌐" in block.text for block in blocks)


# --- stream lifecycle --------------------------------------------------------------------


async def test_stopping_after_one_block_leaves_nothing_held_open(corpus: Path) -> None:
    """An abandoned generator must not be left suspended holding the notebook it opened.

    CPython finalises a live async generator through the event loop that created it, so one
    still suspended when that loop has closed is finalised against a torn-down runtime — a
    crash inside the interpreter's allocator rather than a warning. This parser reads the whole
    notebook before it yields anything, so nothing is live at a suspension point, and re-parsing
    immediately afterwards checks that nothing was left in a state the next read trips over.
    The pattern is shown catching a release placed after the loop in
    ``tests/parsers/test_word.py`` and in ``tests/test_parser_streams.py``.
    """
    raw = raw_from(corpus / "notebook" / "notebook_typical.ipynb", NOTEBOOK_MEDIA_TYPE)
    parser = _parser()
    async with parsing(parser, raw) as blocks:
        async for _ in blocks:
            break
    assert len(await read_blocks(parser, raw)) > 1


class _ShiftedCellParser:
    """Anchors every block to the address of the group after the one it came from.

    The house pattern from ``tests/fakes.py``: a parser whose citations are well-formed and one
    cell wrong, so the assertion that catches it can be watched working.
    """

    media_types = frozenset({NOTEBOOK_MEDIA_TYPE})
    profile = NotebookParser.profile

    def __init__(self) -> None:
        self._real = NotebookParser(NotebookConfig())

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


_PARSERS: Sequence[Parser] = (NotebookParser(NotebookConfig()), _ShiftedCellParser())
"""Type-checked conformance: the fake is a parser, not a mock of one."""
