"""The round-trip harness, held to its own standard.

Every assertion here is checked twice: once with a parser that behaves, and once with a fake
that breaks exactly the rule under test. A suite that only asserts the happy path certifies
nothing — it would pass just as well against a harness whose checks had been deleted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from manicule.core.anchors import Anchor, HeadingAnchor, LineAnchor, PageAnchor, Rect, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.testing import (
    ParserProfile,
    RoundTripReport,
    assert_location_budget,
    assert_round_trip,
    normalise,
)

MEDIA_TYPE = "text/plain"

SECTIONS: tuple[tuple[str, str], ...] = (
    ("Overview", "The service issues short-lived tokens to callers."),
    ("Configuration", "Rotation runs hourly and is not configurable per tenant."),
    ("Rollback", "Restore the previous release and re-run the migration."),
)

DOCUMENT = "\n\n".join(f"# {title}\n{body}" for title, body in SECTIONS)

_SUBSECTION = "Rollback restores the previous release and re-runs the migration."
_WHOLE_SECTION = f"Deployment covers rollout and rollback. {_SUBSECTION}"


def _raw(text: str = DOCUMENT) -> RawDocument:
    return RawDocument(source_id="s", uri="doc.txt", media_type=MEDIA_TYPE, content=text)


class SectionParser:
    """A well-behaved parser: one block per section, resolvable by heading path."""

    media_types = frozenset({MEDIA_TYPE})
    profile = ParserProfile(name="section", max_unlocated_ratio=0.0, max_pagelevel_ratio=None)

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for title, body in _sections(raw.as_text()):
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text=body,
                anchor=HeadingAnchor(path=(title,), fragment=title.lower()),
                heading_path=(title,),
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        if not isinstance(anchor, HeadingAnchor):
            return None
        for title, body in _sections(raw.as_text()):
            if (title,) == anchor.path:
                return f"# {title}\n{body}"
        return None


def _sections(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for part in text.split("\n\n"):
        lines = part.splitlines()
        if lines and lines[0].startswith("# "):
            found.append((lines[0][2:], "\n".join(lines[1:]).strip()))
    return found


async def test_a_parser_whose_anchors_resolve_passes_every_assertion() -> None:
    """The control. Without it, a harness that failed everything would look strict."""
    report = await assert_round_trip(SectionParser(), _raw(), fixture="doc.txt")
    assert report.blocks == len(SECTIONS)
    assert report.unlocated == 0


# --- assertion 1: containment ------------------------------------------------------------


class InventingParser(SectionParser):
    """Claims text the source does not contain."""

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            yield block.model_copy(update={"text": block.text + " Approved by legal."})


async def test_a_block_claiming_text_the_source_lacks_is_caught() -> None:
    """A chunk whose text is not at the location it names is a fabricated quotation."""
    with pytest.raises(AssertionError, match="does not claim"):
        await assert_round_trip(InventingParser(), _raw())


# --- assertion 2: tightness --------------------------------------------------------------


class WholeDocumentParser(SectionParser):
    """Every anchor addresses the entire document — which passes containment."""

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        return raw.as_text() if isinstance(anchor, HeadingAnchor) else None


async def test_an_anchor_covering_the_whole_document_is_caught_by_tightness() -> None:
    """Containment alone is satisfied by pointing at everything.

    Tightness is what does the real work: an anchor much larger than the text it addresses
    cites its neighbours too, and the citation still resolves.
    """
    with pytest.raises(AssertionError, match="resolves to"):
        await assert_round_trip(WholeDocumentParser(), _raw())


async def test_several_chunks_sharing_one_anchor_are_measured_as_a_group() -> None:
    """A long section split into parts must not fail for being split.

    Grouping by anchor is what makes tightness correct rather than merely strict. Comparing a
    whole section against one of its four parts would fail a parser behaving perfectly, and
    the usual repair — loosening the bound until the suite passes — dissolves the assertion.
    """

    class SplitParser(SectionParser):
        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            async for block in super().parse(raw):
                words = block.text.split()
                middle = len(words) // 2
                for half in (" ".join(words[:middle]), " ".join(words[middle:])):
                    yield block.model_copy(update={"text": half})

    report = await assert_round_trip(SplitParser(), _raw())
    assert report.blocks == len(SECTIONS) * 2


# --- assertion 3: discrimination ---------------------------------------------------------


class ShiftedSectionParser(SectionParser):
    """Every block carries the *next* section's anchor — the classic off-by-one."""

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        titles = [title for title, _ in _sections(raw.as_text())]
        index = 0
        async for block in super().parse(raw):
            wrong = titles[(index + 1) % len(titles)]
            index += 1
            yield block.model_copy(
                update={"anchor": HeadingAnchor(path=(wrong,), fragment=wrong.lower())}
            )


async def test_an_anchor_shifted_onto_the_next_section_is_caught() -> None:
    """A fragment assigned to the wrong section resolves cleanly and quotes the wrong text.

    This is the failure with no symptom: the citation opens, the section exists, and the
    words on the screen are not the words the chunk claimed.
    """
    with pytest.raises(AssertionError):
        await assert_round_trip(ShiftedSectionParser(), _raw())


async def test_a_section_that_genuinely_contains_its_subsection_is_not_a_failure() -> None:
    """Real nesting is excluded, and the exclusion is derived rather than declared.

    A declaration would also excuse the mistakes. Deriving it from the heading paths means a
    parser cannot opt out of discrimination by claiming its anchors nest.
    """

    class NestingParser:
        media_types = frozenset({MEDIA_TYPE})
        profile = ParserProfile(name="nesting", max_unlocated_ratio=0.0)

        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            del raw
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text=_WHOLE_SECTION,
                anchor=HeadingAnchor(path=("Deployment",)),
            )
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text=_SUBSECTION,
                anchor=HeadingAnchor(path=("Deployment", "Rollback")),
            )

        async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
            del raw
            if not isinstance(anchor, HeadingAnchor):
                return None
            return _WHOLE_SECTION if anchor.path == ("Deployment",) else _SUBSECTION

    await assert_round_trip(NestingParser(), _raw())


async def test_the_nesting_exclusion_does_not_cover_two_unrelated_sections() -> None:
    """The exclusion has to be narrow, or it is a way out of discrimination entirely.

    Same containment, same texts — but the two heading paths are siblings rather than parent
    and child, so one resolving to the other's text is exactly the confusion the assertion
    exists to catch.
    """

    class SiblingParser:
        media_types = frozenset({MEDIA_TYPE})
        profile = ParserProfile(name="siblings", max_unlocated_ratio=0.0)

        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            del raw
            yield ParsedBlock(
                kind=BlockKind.PROSE, text=_WHOLE_SECTION, anchor=HeadingAnchor(path=("Deploy",))
            )
            yield ParsedBlock(
                kind=BlockKind.PROSE, text=_SUBSECTION, anchor=HeadingAnchor(path=("Roll",))
            )

        async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
            del raw
            if not isinstance(anchor, HeadingAnchor):
                return None
            return _WHOLE_SECTION if anchor.path == ("Deploy",) else _SUBSECTION

    with pytest.raises(AssertionError):
        await assert_round_trip(SiblingParser(), _raw())


# --- assertion 4: determinism ------------------------------------------------------------


class DriftingParser(SectionParser):
    """Produces a different block list on the second call."""

    def __init__(self) -> None:
        self._calls = 0

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        self._calls += 1
        blocks = [block async for block in super().parse(raw)]
        for block in blocks if self._calls == 1 else reversed(blocks):
            yield block


async def test_a_parser_that_varies_between_runs_is_caught() -> None:
    """Chunk ids are derived from content and position, so a varying parser churns the corpus
    on every re-ingest even when nothing in the document changed."""
    with pytest.raises(AssertionError, match="different blocks"):
        await assert_round_trip(DriftingParser(), _raw())


# --- assertion 5: the location budget ----------------------------------------------------


def test_a_parser_that_gives_up_on_every_block_blows_its_unlocated_budget() -> None:
    """Rule 1 — never invent a location — is satisfiable by never producing one."""
    profile = ParserProfile(name="evasive", max_unlocated_ratio=0.05)
    reports = [RoundTripReport(fixture="f", blocks=10, unlocated=10)]
    with pytest.raises(AssertionError, match="Unlocated"):
        assert_location_budget(profile, reports)


def test_a_parser_whose_boxes_stopped_working_blows_its_page_level_budget() -> None:
    """Losing the rectangles is silent: the citation still names the right page.

    The ratio is the only signal that box extraction broke, which is why it is a declared
    ceiling rather than a thing somebody notices.
    """
    profile = ParserProfile(name="pdf", max_unlocated_ratio=0.0, max_pagelevel_ratio=0.10)
    reports = [RoundTripReport(fixture="f", blocks=10, page_level=4)]
    with pytest.raises(AssertionError, match="resolve only to a page"):
        assert_location_budget(profile, reports)


def test_a_parser_that_declared_no_page_budget_may_not_emit_page_anchors() -> None:
    """A parser that starts producing page-level anchors has to say so, rather than
    inheriting an unstated allowance."""
    profile = ParserProfile(name="markdown", max_unlocated_ratio=0.0, max_pagelevel_ratio=None)
    with pytest.raises(AssertionError, match="declared no page-level budget"):
        assert_location_budget(profile, [RoundTripReport(fixture="f", blocks=4, page_level=1)])


def test_a_shrinking_corpus_cannot_pass_a_ratio_by_having_nothing_in_it() -> None:
    """Every ratio is satisfied by an empty corpus, so the corpus size is asserted too."""
    profile = ParserProfile(name="pdf", max_unlocated_ratio=0.0)
    with pytest.raises(AssertionError, match="fewer than"):
        assert_location_budget(profile, [RoundTripReport(fixture="f", blocks=2)], blocks=20)


def test_budgets_pass_when_the_parser_stays_inside_them() -> None:
    profile = ParserProfile(name="pdf", max_unlocated_ratio=0.0, max_pagelevel_ratio=0.10)
    assert_location_budget(profile, [RoundTripReport(fixture="f", blocks=20, page_level=2)])


# --- page-level anchors are exempt from tightness, not from everything --------------------


async def test_a_page_level_anchor_is_exempt_from_tightness_and_capped_by_the_budget() -> None:
    """Resolving a page-level anchor returns a whole page, which no bound could constrain.

    So it is budgeted instead — a parser cannot quietly make every anchor page-level to get
    through, because assertion 5 counts them.
    """

    class PageParser:
        media_types = frozenset({"application/pdf"})
        profile = ParserProfile(name="pages", max_unlocated_ratio=0.0, max_pagelevel_ratio=1.0)

        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            del raw
            yield ParsedBlock(kind=BlockKind.PROSE, text="alpha", anchor=PageAnchor(page=1))
            yield ParsedBlock(kind=BlockKind.PROSE, text="beta", anchor=PageAnchor(page=2))

        async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
            del raw
            if not isinstance(anchor, PageAnchor):
                return None
            return "alpha and a great deal of other text" if anchor.page == 1 else "beta likewise"

    report = await assert_round_trip(PageParser(), _raw())
    assert report.page_level == 2


async def test_a_rect_bearing_page_anchor_is_held_to_the_tighter_bound() -> None:
    """Boxes bound the quote, so a page anchor carrying them gets 5% of slack, not a page."""

    class BoxedParser:
        media_types = frozenset({"application/pdf"})
        profile = ParserProfile(name="boxed", max_unlocated_ratio=0.0, max_pagelevel_ratio=0.0)

        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            del raw
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text="alpha",
                anchor=PageAnchor(page=1, rects=(Rect(x0=0.1, y0=0.1, x1=0.2, y1=0.2),)),
            )

        async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
            del raw
            if not isinstance(anchor, PageAnchor):
                return None
            return "alpha and a great deal of other text"

    with pytest.raises(AssertionError, match="resolves to"):
        await assert_round_trip(BoxedParser(), _raw())


# --- Unlocated -----------------------------------------------------------------------------


async def test_an_unlocated_anchor_must_not_resolve_to_anything() -> None:
    """ "We could not determine a location" and "here it is" cannot both be true."""

    class ConfusedParser:
        media_types = frozenset({MEDIA_TYPE})
        profile = ParserProfile(name="confused", max_unlocated_ratio=1.0)

        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            del raw
            yield ParsedBlock(
                kind=BlockKind.PROSE, text="alpha", anchor=Unlocated(reason="no positions")
            )

        async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
            del raw, anchor
            return "alpha"

    with pytest.raises(AssertionError, match="Unlocated"):
        await assert_round_trip(ConfusedParser(), _raw())


# --- normalisation -------------------------------------------------------------------------


def test_de_hyphenation_runs_before_whitespace_is_collapsed() -> None:
    """Order is load-bearing, and running it the other way round fails silently.

    Collapsing whitespace first destroys the line breaks de-hyphenation needs, after which
    hyphenated words simply stop being joined and every affected comparison quietly starts
    relying on the substring check being lenient.
    """
    assert normalise("config-\nuration options") == "configuration options"


def test_ligatures_and_invisible_characters_are_folded_but_content_is_not_rewritten() -> None:
    """NFC rather than NFKC: NFKC would fold the ligatures for free and also rewrite ``½``,
    superscripts and full-width forms, which are content a citation must reproduce."""
    assert normalise("ﬁle sepa­rately\u200b now") == "file separately now"
    assert normalise("½ cup") == "½ cup"


async def test_a_line_anchor_that_is_one_line_out_is_caught() -> None:
    """The house fake, applied to the harness that is supposed to catch it."""

    class OffByOneParser:
        media_types = frozenset({MEDIA_TYPE})
        profile = ParserProfile(name="offbyone", max_unlocated_ratio=0.0)

        async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
            for number, line in enumerate(raw.as_text().splitlines(), start=1):
                if line.strip():
                    yield ParsedBlock(
                        kind=BlockKind.PROSE,
                        text=line,
                        anchor=LineAnchor(start=number + 1, end=number + 1),
                    )

        async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
            if not isinstance(anchor, LineAnchor):
                return None
            lines = raw.as_text().splitlines()
            if anchor.end > len(lines):
                return None
            return "\n".join(lines[anchor.start - 1 : anchor.end])

    with pytest.raises(AssertionError):
        await assert_round_trip(OffByOneParser(), _raw())
