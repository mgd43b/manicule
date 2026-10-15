"""Front matter: where it ends, and what it says.

Three callers need the same answer to "is this front matter", and they used to share it by one
of them living inside the Markdown parser: the parser masks a closed block so CommonMark does not
read it as a heading, authoring refuses a block that opens and never closes, and the knowledge
reader (:mod:`manicule.ingest.knowledge`) parses what a closed block says. A second spelling of
``---`` would be a second answer, and the three would agree until one of them was changed.

**This module imports no Markdown library**, and that is why the rule is here rather than beside
the parser. Authoring and ingest ask the question on paths that never parse a document, and
pulling in ``markdown-it`` to learn where a fence is would put a parser on a path whose whole
point is that it does not run one.

**Reading is bounded, and the bound is a refusal rather than a truncation.** Front matter is
text inside a corpus file, which makes it input from whoever wrote the file. A block past
:data:`MAX_FRONT_MATTER_CHARS` is reported as unreadable, because a truncated block is a
*different* block: the half that survived would be read as though it were all of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FRONT_MATTER_FENCE: Final = "---"
"""What opens and closes a front-matter block."""

MAX_FRONT_MATTER_CHARS: Final = 64 * 1024
"""Longest block this module will hand to a YAML reader."""


class FrontMatterError(ValueError):
    """The block is there and cannot be read, with a sentence saying why."""


def front_matter_end(lines: Sequence[str]) -> int:
    """How many leading lines are front matter. Zero when there is none.

    **Zero is also the answer for a fence that never closes**, which is the case the parser
    cannot do anything about and authoring therefore refuses at the door: an unterminated
    block is read as content, and CommonMark turns its last line into a setext heading the
    document does not have. Two callers, one rule.
    """
    if not lines or lines[0].strip() != FRONT_MATTER_FENCE:
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONT_MATTER_FENCE:
            return index + 1
    return 0


def front_matter_lines(lines: Sequence[str]) -> Sequence[str]:
    """The lines *between* the fences, or none when the document has no closed block."""
    end = front_matter_end(lines)
    return lines[1 : end - 1] if end else ()


def read_front_matter(lines: Sequence[str]) -> Mapping[str, object]:
    """What the front matter says, as the mapping it declares.

    An empty mapping for a document with no block, and for a block that is not a mapping at
    all — a bare list is front matter nobody here reads, not an error in the document.

    YAML 1.2 through ``ruamel.yaml``'s safe loader, for the reason
    :mod:`manicule.parsers.structured` gives: PyYAML implements 1.1, where ``no`` is a boolean
    and ``2026-03-02`` is a date nobody asked for. Safe, so a tag in the block cannot construct
    an arbitrary object, and duplicate keys are refused rather than resolved to the last one —
    a repeated key is a field that silently lost a value.

    Raises:
        FrontMatterError: The block is over the size bound or is not valid YAML. The message is
            the reader's own first line, because the usual cause is a quoted message with a
            ``: `` in it and the line number is what finds it.
    """
    inner = front_matter_lines(lines)
    if not inner:
        return {}
    block = "\n".join(inner)
    if len(block) > MAX_FRONT_MATTER_CHARS:
        msg = (
            f"the front matter is {len(block)} characters, over the {MAX_FRONT_MATTER_CHARS} "
            f"limit, so it was not read"
        )
        raise FrontMatterError(msg)
    # Imported here: ruamel is a parser dependency, and a process that never reads front
    # matter should not pay for it at start-up.
    from ruamel.yaml import YAML  # noqa: PLC0415
    from ruamel.yaml.error import YAMLError  # noqa: PLC0415

    # Three exception families, because the library raises three. `YAMLError` is a syntax or
    # structure problem. `ValueError` is what its *constructor* raises for a scalar that parses
    # and cannot exist — `2026-13-45T00:00:00Z` is a well-formed timestamp with no month 13, and
    # reaches here as a bare `ValueError` from `datetime`. `RecursionError` is an alias that
    # contains itself. A reader that let any of them out would fail the whole document's ingest
    # over a field nobody asked it to read.
    try:
        loaded = YAML(typ="safe").load(block)
    except (YAMLError, ValueError, RecursionError) as exc:
        text = str(exc).strip()
        reason = text.splitlines()[0] if text else type(exc).__name__
        msg = (
            f"the front matter is not valid YAML ({reason}). A quoted message or a title that "
            f"contains ': ' has to be wrapped in quotes"
        )
        raise FrontMatterError(msg) from exc
    if not isinstance(loaded, dict):
        return {}
    return cast("Mapping[str, object]", loaded)


__all__ = [
    "FRONT_MATTER_FENCE",
    "MAX_FRONT_MATTER_CHARS",
    "FrontMatterError",
    "front_matter_end",
    "front_matter_lines",
    "read_front_matter",
]
