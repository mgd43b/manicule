"""The heading breadcrumb: what tells the embedder where a chunk sits.

``text`` is what a user is shown as the quotation. ``embed_text`` is what the model reads.
Storing both means retrieval scaffolding never leaks into a quotation, and a section titled
"Configuration" is still findable by someone searching for what it configures.

```
embed_text = breadcrumb + "\\n\\n" + text     (when a breadcrumb exists)
embed_text = text                            (otherwise)
```
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from manicule.core.content import string_list

if TYPE_CHECKING:
    from manicule.core.content import Document

SEPARATOR = " > "
ELLIPSIS = "…"

ENDS_KEPT = 2
"""How many elements survive elision: the outermost and the innermost.

The outermost says which corpus and product area; the innermost says what this particular
section is. Everything between them is the part a reader can infer."""


def elements(*groups: Iterable[str]) -> tuple[str, ...]:
    """Flatten breadcrumb components coarsest to finest, dropping blanks and stutters.

    Adjacent repeats are collapsed: a page titled "Auth Service" under a parent titled "Auth
    Service" yields one element, and a Markdown file whose only ``h1`` equals its title drops
    the ``h1``. Without this roughly a third of real breadcrumbs stutter, and the repetition
    reaches the embedder as emphasis nobody intended.
    """
    collected: list[str] = []
    for group in groups:
        for value in group:
            cleaned = " ".join(value.split())
            if not cleaned:
                continue
            if collected and collected[-1].casefold() == cleaned.casefold():
                continue
            collected.append(cleaned)
    return tuple(collected)


def parts(
    document: Document,
    heading_path: Sequence[str],
    *,
    hierarchy: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """One chunk's breadcrumb elements: the source's prefix, its hierarchy, the title, the headings.

    The one definition of what a breadcrumb *says*. :func:`render` decides how much of it fits
    in front of ``embed_text``. ``hierarchy`` is :func:`source_hierarchy` of ``document``,
    passed in by a caller that has already resolved it once for the whole document.
    """
    return elements(
        string_list(document.metadata.get("breadcrumb_prefix")) or (),
        source_hierarchy(document) if hierarchy is None else hierarchy,
        (document.title,),
        heading_path,
    )


def source_hierarchy(document: Document) -> tuple[str, ...]:
    """Where a document sits in its source, coarsest first.

    Two spellings of one fact, in precedence order, because they arrived in that order. A
    validated :class:`~manicule.core.provenance.SourceMetadata` is preferred when the document
    carries one; ``metadata["ancestors"]`` is the older untyped convention that connectors
    without a record still fill in, and it keeps working exactly as it did.

    **The record wins where both are present, and that is the safe way round.** Its
    ``section_path`` has been through depth, length and control-character validation;
    ``ancestors`` has been through none. Preferring the unvalidated spelling would mean a
    connector that supplied both got the weaker of its own two answers embedded into every
    vector — and a breadcrumb is not something anybody reads afterwards to check.

    This is also the whole of what propagates from a document's source record into a chunk, and
    it propagates as *text the embedder reads* rather than as a copy on the chunk row. Nothing
    document-level is duplicated per chunk; ``docs/contracts.md`` §2 fixes what a chunk is, and
    a citation resolves the rest through ``document_id``.
    """
    record = document.provenance
    if record is not None and record.source is not None and record.source.section_path:
        return record.source.section_path
    return tuple(string_list(document.metadata.get("ancestors")) or ())


def render(parts: Sequence[str], count_tokens: Callable[[str], int], budget: int) -> str:
    """Join ``parts`` into a breadcrumb that fits ``budget`` tokens.

    Over budget, elements are dropped **from the middle outward**, never from the tail. The
    two ends carry the most information: the outermost says which corpus and product area,
    the innermost says what this particular section is. Truncating the tail throws away the
    element that disambiguates "Configuration", which is the one the breadcrumb existed for.

    Returns:
        The breadcrumb, or ``""`` when there are no parts. An empty breadcrumb is a
        legitimate answer — a ``.txt`` at the root of a filesystem source has no hierarchy,
        and inventing one would put a fabricated signal into the vector.
    """
    kept = list(parts)
    if not kept or budget <= 0:
        return ""

    joined = SEPARATOR.join(kept)
    if count_tokens(joined) <= budget:
        return joined

    while len(kept) > ENDS_KEPT:
        kept.pop(len(kept) // 2)
        joined = _with_gap(kept)
        if count_tokens(joined) <= budget:
            return joined

    joined = _with_gap(kept)
    if count_tokens(joined) <= budget:
        return joined
    return _truncate_last(kept, count_tokens, budget)


def _with_gap(kept: Sequence[str]) -> str:
    """Render an elided breadcrumb, marking the gap so nobody reads it as complete.

    The marker survives down to two elements. Without it a six-level path elided to its two
    ends reads as a two-level path, which is a different claim about the document.
    """
    return SEPARATOR.join((kept[0], ELLIPSIS, *kept[1:]))


def _truncate_last(kept: Sequence[str], count_tokens: Callable[[str], int], budget: int) -> str:
    """Last resort: the first and last elements alone still do not fit.

    The final element is cut on a word boundary and marked. Cutting mid-word would produce a
    breadcrumb that reads as a different heading.
    """
    first = kept[0]
    last = kept[-1] if len(kept) > 1 else ""
    words = last.split()
    while words:
        candidate = SEPARATOR.join(filter(None, (first, " ".join(words) + ELLIPSIS)))
        if count_tokens(candidate) <= budget:
            return candidate
        words.pop()
    # Even the outermost element alone is over budget; cut it the same way rather than
    # returning nothing, because "which corpus" is the last thing worth keeping.
    outer = first.split()
    while len(outer) > 1:
        outer.pop()
        candidate = " ".join(outer) + ELLIPSIS
        if count_tokens(candidate) <= budget:
            return candidate
    # A single identifier-heavy element may have no word boundary at all. Returning it whole
    # would make ``budget`` advisory precisely on the fallback path. Cut by character against
    # the exact counter; the ellipsis makes the loss explicit, and an empty result is safer
    # than scaffolding that leaves no room for citable content.
    low, high = 1, len(first)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = first[:middle] + ELLIPSIS
        if count_tokens(candidate) <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


__all__ = ["ELLIPSIS", "SEPARATOR", "elements", "render"]
