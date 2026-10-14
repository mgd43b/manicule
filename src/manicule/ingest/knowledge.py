"""Reading a structured memory out of a Markdown file, and refusing a malformed one at the door.

Two callers, one rule, and they differ only in what they do with a bad document:

:func:`read`, at ingest, **never raises**. A file arrives by a sync, a ``git pull`` or another
tool's write, and a document that declares a record it cannot have is still a document. It is
stored with the *reason* (:attr:`~manicule.core.knowledge.Knowledge.unavailable_reason`), because
the alternative — storing nothing — presents as a memory nobody cited, which is indistinguishable
from one that never tried.

:func:`check`, at authoring, **raises**. Refusing at the door is the only point at which the
author is still holding the text and the error can be acted on; a refusal six weeks later in a
log is a diagnostic nobody reads. It also checks what ingest cannot: the *body*, whose sections
are the part of a memory that is indexed and retrieved.

**What makes a document a structured memory is one key.** ``knowledge_type`` in the front matter,
at column 0. A document without it is ordinary Markdown whatever else its front matter says —
``status:`` and ``source:`` mean something in half the documentation sites in the world, and
reading them here would put a refusal on pages nobody authored as memories.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from manicule.core.knowledge import (
    FIELDS,
    MARKER,
    Knowledge,
    KnowledgeError,
    KnowledgeRecord,
    bounded_reason,
    normalize_slug,
)
from manicule.parsers.front_matter import (
    MAX_FRONT_MATTER_CHARS,
    FrontMatterError,
    front_matter_end,
    front_matter_lines,
    read_front_matter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SECTIONS: Final = ("Problem", "Answer", "Applicability")
"""The sections a structured memory's body carries, under its ``# Title``.

The body rather than the front matter, because the body is what is indexed: a reader finds a
memory by the words of its problem and its answer, and a heading is what lets a hit say *which*
of the three matched.
"""

MISPLACED: Final = frozenset({"sources", "citations", "citation"})
"""Spellings that mean *evidence* and are read by nothing.

Refused when they sit beside ``knowledge_type``, because the failure they cause is silent: a
memory whose citations were written under the wrong name looks, in every hit, like one that has
none.
"""

_READ_BEYOND_BLOCK = 4096
"""How far past the largest readable block the scan goes looking for its closing fence."""

_TOP_KEY = re.compile(r"""^["']?([A-Za-z_][A-Za-z0-9_-]*)["']?\s*:""")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_CODE_FENCE = re.compile(r"^ {0,3}(```|~~~)")


def _top_level_keys(inner: Sequence[str]) -> set[str]:
    """The keys a front-matter block declares at column 0, found without parsing it.

    Textual on purpose: whether a document *declares* a record has to be answerable when the
    YAML is broken, because a broken block that mentioned ``knowledge_type`` is exactly the
    document whose refusal somebody needs to see.
    """
    return {match[1] for line in inner if (match := _TOP_KEY.match(line))}


def _lines_of(text: str) -> list[str]:
    return text[: MAX_FRONT_MATTER_CHARS + _READ_BEYOND_BLOCK].splitlines()


def read(text: str) -> Knowledge | None:
    """The structured memory ``text`` declares, or ``None`` when it does not declare one.

    Never raises. A declared record that cannot be stored comes back as a
    :class:`~manicule.core.knowledge.Knowledge` holding the reason.
    """
    lines = _lines_of(text)
    if MARKER not in _top_level_keys(front_matter_lines(lines)):
        return None
    try:
        declared = read_front_matter(lines)
        record = KnowledgeRecord.from_declared({k: declared[k] for k in FIELDS if k in declared})
    except (FrontMatterError, KnowledgeError) as exc:
        return Knowledge(unavailable_reason=bounded_reason(str(exc)))
    return Knowledge(record=record)


def check(text: str, *, slug: str) -> None:
    """Refuse ``text`` if it is a structured memory that would not be read as one.

    Silent on an ordinary document, including one with no front matter at all: nothing here
    makes authoring stricter for a collection that holds anything else.

    Raises:
        KnowledgeError: The front matter declares ``knowledge_type`` and is broken, spells
            evidence under a name nothing reads, or the body lacks a title or a section; or the
            front matter declares evidence and no ``knowledge_type``, which would be ignored.
    """
    # The whole text, not the bounded slice `read` takes: this is the one place the body is
    # checked, and a long memory whose last section sat past the bound would be refused for
    # lacking a section it has. Authoring already holds the document in memory and the
    # connector's own `max_bytes` bounds it, so there is nothing here to protect.
    lines = text.splitlines()
    keys = _top_level_keys(front_matter_lines(lines))
    if MARKER not in keys:
        stray = sorted(keys & ({"evidence_status", "evidence", "relationships"} | MISPLACED))
        if stray:
            msg = (
                f"the front matter has {', '.join(stray)} and no {MARKER}, so none of it is "
                f"read. Add {MARKER} (a structured memory) or remove them"
            )
            raise KnowledgeError(msg)
        return
    misplaced = sorted(keys & MISPLACED)
    if misplaced:
        msg = (
            f"the front matter has {', '.join(misplaced)}, which nothing reads. Citations go "
            f"under evidence, each with kind, locator, said_at and quote"
        )
        raise KnowledgeError(msg)
    parsed = read(text)
    if parsed is None or parsed.record is None:
        raise KnowledgeError(
            parsed.unavailable_reason
            if parsed
            else f"the front matter is over {MAX_FRONT_MATTER_CHARS} characters, so it was not read"
        )
    if any(
        normalize_slug(link.target) == normalize_slug(slug) for link in parsed.record.relationships
    ):
        msg = f"relationships names this memory ({slug!r}) as its own target"
        raise KnowledgeError(msg)
    _require_body(lines[front_matter_end(lines) :])


def _require_body(body: Sequence[str]) -> None:
    """Refuse a body without a title and the three sections, each with something in it."""
    headings: list[tuple[int, str, int]] = []
    fence = ""
    for index, line in enumerate(body):
        marker = _CODE_FENCE.match(line)
        if marker:
            fence = "" if fence == marker[1] else (fence or marker[1])
            continue
        if fence:
            continue
        found = _HEADING.match(line)
        if found:
            headings.append((len(found[1]), found[2].strip().rstrip(":").strip(), index))
    problems: list[str] = []
    first = headings[0] if headings else None
    if first is None or first[0] != 1 or not first[1]:
        problems.append("a first line `# Title` naming the memory")
    for section in SECTIONS:
        if not _has_content(body, headings, section):
            problems.append(f"`## {section}` with text under it")
    if problems:
        msg = (
            f"the body is missing {'; '.join(problems)}. A structured memory is a `# Title` "
            f"and {', '.join(f'`## {name}`' for name in SECTIONS)}, because the body is what is "
            f"indexed and a reader finds it by its problem and its answer"
        )
        raise KnowledgeError(msg)


def _has_content(
    body: Sequence[str], headings: Sequence[tuple[int, str, int]], section: str
) -> bool:
    """Whether a level-2 heading called ``section`` has any text before the next heading."""
    for position, (level, title, index) in enumerate(headings):
        if level != 2 or title.lower() != section.lower():  # noqa: PLR2004 - section headings are level 2
            continue
        end = headings[position + 1][2] if position + 1 < len(headings) else len(body)
        if any(line.strip() for line in body[index + 1 : end]):
            return True
    return False


__all__ = ["MISPLACED", "SECTIONS", "check", "read"]
