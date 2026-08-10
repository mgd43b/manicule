"""Anchors are the type it is most expensive to get wrong. These tests say why."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from manicule.core.anchors import (
    Anchor,
    CellAnchor,
    HeadingAnchor,
    LineAnchor,
    PageAnchor,
    Rect,
    Unlocated,
    is_located,
)


class Holder(BaseModel):
    """A stand-in for anything that stores an anchor."""

    anchor: Anchor


ANCHORS: list[Anchor] = [
    PageAnchor(page=4, rects=(Rect(x0=0.1, y0=0.2, x1=0.9, y1=0.3),)),
    HeadingAnchor(path=("Guide", "Configuration"), fragment="configuration"),
    LineAnchor(start=10, end=24, symbol="build_container"),
    CellAnchor(sheet="Sheet1", ref="B4:D12"),
    Unlocated(reason="the source reports no page structure"),
]


@pytest.mark.parametrize("anchor", ANCHORS, ids=lambda a: a.kind)
def test_every_anchor_round_trips_through_json(anchor: Anchor) -> None:
    """Anchors are written to the index and read back, so this is the load-bearing test."""
    stored = Holder(anchor=anchor).model_dump_json()
    assert Holder.model_validate_json(stored).anchor == anchor


def test_unlocated_is_a_member_carrying_a_reason() -> None:
    """ "We could not determine a location" must be distinguishable from "nobody asked"."""
    anchor = Unlocated(reason="scanned page, no text layer")
    assert not is_located(anchor)
    assert anchor.reason

    with pytest.raises(ValidationError):
        Unlocated(reason="")


def test_page_anchor_keeps_rectangles_separate() -> None:
    """A quote across a column break has two boxes.

    Merging them into one envelope highlights text that was never quoted, which is worse
    than highlighting nothing.
    """
    left = Rect(x0=0.05, y0=0.10, x1=0.45, y1=0.90)
    right = Rect(x0=0.55, y0=0.10, x1=0.95, y1=0.30)
    anchor = PageAnchor(page=2, rects=(left, right))

    assert len(anchor.rects) == 2
    assert anchor.rects == (left, right)


def test_page_anchor_with_no_rectangles_is_allowed() -> None:
    """ "This page" is a correct if coarse location, and better than an invented box."""
    assert PageAnchor(page=1).rects == ()


@pytest.mark.parametrize("page", [0, -1])
def test_pages_are_one_based(page: int) -> None:
    with pytest.raises(ValidationError):
        PageAnchor(page=page)


def test_rectangles_are_normalised_and_ordered() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Rect(x0=0.0, y0=0.0, x1=1.5, y1=1.0)
    with pytest.raises(ValidationError, match="out of order"):
        Rect(x0=0.8, y0=0.0, x1=0.2, y1=1.0)


def test_a_quote_at_the_very_edge_of_the_page_is_not_rejected() -> None:
    """Composing a rotation with a crop offset lands the last bits of a float outside 1.0.

    Refusing that would fail a legitimate citation for being exactly where the reader can
    see it is.
    """
    rect = Rect(x0=-1e-9, y0=0.5, x1=1.0 + 1e-9, y1=0.6)
    assert rect.x0 == 0.0
    assert rect.x1 == 1.0


def test_a_transform_that_is_actually_wrong_still_fails() -> None:
    """The tolerance absorbs float noise, not a mistake. A wrong box is wrong by percentages."""
    with pytest.raises(ValidationError):
        Rect(x0=0.0, y0=0.0, x1=1.01, y1=1.0)
    with pytest.raises(ValidationError):
        Rect(x0=0.0, y0=0.0, x1=612.0, y1=792.0)


def test_a_zero_width_run_of_glyphs_does_not_fail_on_reversed_corners() -> None:
    rect = Rect(x0=0.25, y0=0.4, x1=0.25 - 1e-9, y1=0.45)
    assert rect.x1 == rect.x0


def test_the_tolerance_does_not_hide_a_genuinely_reversed_rectangle() -> None:
    with pytest.raises(ValidationError, match="out of order"):
        Rect(x0=0.6, y0=0.1, x1=0.5, y1=0.2)


def test_line_ranges_are_inclusive_and_ordered() -> None:
    assert LineAnchor(start=7, end=7).end == 7
    with pytest.raises(ValidationError, match="ends before it starts"):
        LineAnchor(start=9, end=4)


def test_heading_path_must_locate_something() -> None:
    with pytest.raises(ValidationError):
        HeadingAnchor(path=())


def test_cell_anchor_recombines_into_a1_notation() -> None:
    assert CellAnchor(sheet="Sheet1", ref="B4:D12").a1 == "Sheet1!B4:D12"
    assert CellAnchor(sheet="Q3", ref="A1").a1 == "Q3!A1"


@pytest.mark.parametrize("ref", ["", "B", "4", "Sheet1!B4", "B4:", "b4"])
def test_cell_references_are_validated(ref: str) -> None:
    with pytest.raises(ValidationError):
        CellAnchor(sheet="Sheet1", ref=ref)


def test_anchors_are_frozen_and_hashable() -> None:
    """Anchors are identity, not state. Hashability is what lets citations be deduplicated."""
    anchor = LineAnchor(start=1, end=2)
    with pytest.raises(ValidationError):
        anchor.start = 5

    same: list[object] = [anchor, LineAnchor(start=1, end=2)]
    assert len({hash(one) for one in same}) == 1


def test_the_discriminator_rejects_a_mixed_up_anchor() -> None:
    """A page number smuggled into a heading anchor must not validate."""
    with pytest.raises(ValidationError):
        Holder.model_validate({"anchor": {"kind": "heading", "page": 3}})
