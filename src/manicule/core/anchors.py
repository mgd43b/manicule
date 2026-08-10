"""Anchors — where a citation points.

The rule this module exists to enforce: **a location is correct, or it is absent.** There
is no best-guess member. A parser that cannot determine where a piece of text came from
returns :class:`Unlocated` with a reason, which is visible in diagnostics, rather than a
plausible-looking number that points at nothing.

.. warning::

   This shape is locked once a corpus has been ingested. Every stored citation is written
   against it, and changing a field invalidates all of them. Additive changes to a single
   member are survivable; renaming, removing, or changing the meaning of a field is not.

Design decisions that are not obvious from the field list:

``Unlocated`` is a member, not ``None``
    "We could not determine a location" and "nobody asked for one" are different facts.
    Encoding the first as ``None`` makes them indistinguishable and hides parser failures.

``PageAnchor.rects`` is a list, never a merged envelope
    A quote that spans a column break occupies two boxes. Merging them yields a rectangle
    covering text that was not quoted, which is a wrong highlight rather than a missing one.

Rectangles are in normalised page coordinates
    ``0.0``-``1.0`` on both axes, origin top-left. Storing points would require storing the
    page dimensions alongside every rectangle to render it; normalising once at parse time,
    where the page box is known, removes that coupling and makes the values checkable.

``kind`` is a discriminator
    Anchors are serialised into the index and read back. A tagged union round-trips
    unambiguously; an untagged one does not.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Sheet-relative A1 reference: a single cell (``B4``) or a range (``B4:D12``).
_A1_REFERENCE = re.compile(r"^[A-Z]{1,3}[1-9][0-9]{0,6}(:[A-Z]{1,3}[1-9][0-9]{0,6})?$")


class _AnchorBase(BaseModel):
    """Shared configuration. Anchors are frozen because they are identity, not state."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Rect(BaseModel):
    """An axis-aligned rectangle in normalised page coordinates.

    Origin is the top-left of the page box; ``x1``/``y1`` are the bottom-right corner.
    Both axes run ``0.0`` to ``1.0``, so a rectangle is meaningful without knowing the page
    size, at any zoom, in any renderer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Rect:
        if self.x1 < self.x0 or self.y1 < self.y0:
            msg = (
                f"rectangle corners are out of order: "
                f"({self.x0}, {self.y0}) to ({self.x1}, {self.y1})"
            )
            raise ValueError(msg)
        return self


class PageAnchor(_AnchorBase):
    """A location on a numbered page. PDF pages, PPTX slides.

    ``rects`` may be empty, which means "this page" with no finer location — a correct if
    coarse answer. It must never contain a rectangle that was inferred rather than measured.
    """

    kind: Literal["page"] = "page"
    page: int = Field(ge=1, description="1-based page or slide number, as the reader sees it.")
    rects: tuple[Rect, ...] = ()


class HeadingAnchor(_AnchorBase):
    """A location in a heading hierarchy. Confluence, Markdown, DOCX, HTML.

    ``path`` is the breadcrumb from the document root to the containing section, outermost
    first. ``fragment`` is the URL fragment that deep-links to it where the source defines
    one, so a citation opens at the section rather than at the top of the page.
    """

    kind: Literal["heading"] = "heading"
    path: tuple[str, ...] = Field(min_length=1)
    fragment: str | None = None


class LineAnchor(_AnchorBase):
    """A line range in a text file. Source code.

    Lines are 1-based and the range is inclusive on both ends, matching every editor and
    every code host. ``symbol`` is the enclosing definition where one can be identified.
    """

    kind: Literal["line"] = "line"
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    symbol: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> LineAnchor:
        if self.end < self.start:
            msg = f"line range ends before it starts: {self.start}-{self.end}"
            raise ValueError(msg)
        return self


class CellAnchor(_AnchorBase):
    """A cell or cell range in a spreadsheet. XLSX, CSV.

    ``sheet`` and ``ref`` are stored apart so a consumer can filter by sheet without
    parsing. :attr:`a1` recombines them into the form a spreadsheet application accepts.
    """

    kind: Literal["cell"] = "cell"
    sheet: str = Field(min_length=1)
    ref: str = Field(pattern=_A1_REFERENCE.pattern)

    @property
    def a1(self) -> str:
        """The combined reference, e.g. ``Sheet1!B4:D12``."""
        return f"{self.sheet}!{self.ref}"


class Unlocated(_AnchorBase):
    """No location could be determined, and why.

    The reason is required and is shown in diagnostics. A parser emitting many of these is
    a parser to fix, which is only possible because the reason is recorded here rather than
    discarded.
    """

    kind: Literal["unlocated"] = "unlocated"
    reason: str = Field(min_length=1)


type Anchor = Annotated[
    PageAnchor | HeadingAnchor | LineAnchor | CellAnchor | Unlocated,
    Field(discriminator="kind"),
]
"""Where a citation points. See the module docstring before changing this."""


def is_located(anchor: Anchor) -> bool:
    """True when the anchor identifies a place in the source document."""
    return not isinstance(anchor, Unlocated)


__all__ = [
    "Anchor",
    "CellAnchor",
    "HeadingAnchor",
    "LineAnchor",
    "PageAnchor",
    "Rect",
    "Unlocated",
    "is_located",
]
