"""Fixtures for the Confluence ADF parser, generated rather than committed.

ADF is JSON, so generating it from Python is the only way to keep it readable: the node tree
is built out of small helpers here, where a reviewer can see what a ``panel`` or a
``tableHeader`` is, instead of being a thousand-line blob nobody diffs.

The four kinds ``docs/parsing.md`` §3.5 requires, and what each is here to catch:

**Typical** — a page of the sort Confluence mostly holds: headings, paragraphs, a fenced code
block, a table with a header row, a nested list and a warning panel.

**Structurally hard** — a list nested five deep; a table; a code block; an ``expand`` whose
collapsed body is still content; and a ``bodiedExtension``, because a macro that wraps content
must not swallow it.

**Degenerate** — a document with no content at all, a heading with nothing under it, and a
page with no title, which is the case that costs the text above the first heading its address.

**Hostile** — bytes that are not JSON, JSON that is not ADF, malformed UTF-8, an unknown node
type, and astral-plane text in a heading. The first three must be declined so the next parser
in the chain gets a turn; the last two must be parsed, because a document is not broken for
containing a node type that postdates this code.

Every block's text in every generated file is distinct, and no block's text contains
another's. That is a requirement rather than tidiness: the discrimination assertion
(``docs/parsing.md`` §3.3) compares each block's text against every other anchor's resolved
text, and two blocks that read identically cannot be told apart by anything, a person reading
the citation included. In ADF a heading's text is exactly its title, so **no heading title is
repeated in this corpus and none of them recurs in a neighbouring sentence** — the duplicate
titles that exercise Confluence's ``-1`` anchor suffix are asserted directly in
``tests/parsers/test_adf.py`` instead, where the two sections can be told apart by the
fragments rather than by their text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = ["LARGE_SECTIONS", "build"]

LARGE_SECTIONS = 12
"""Sections in the over-cap fixture. Few and long rather than many and short: the streaming
path is exercised by the byte count, while the assertions that compare every block against
every other are quadratic in the block count."""

Node = dict[str, object]


def _text(value: str) -> Node:
    return {"type": "text", "text": value}


def _paragraph(value: str) -> Node:
    return {"type": "paragraph", "content": [_text(value)]}


def _heading(level: int, value: str) -> Node:
    return {"type": "heading", "attrs": {"level": level}, "content": [_text(value)]}


def _item(value: str, nested: Node | None = None) -> Node:
    content: list[Node] = [_paragraph(value)]
    if nested is not None:
        content.append(nested)
    return {"type": "listItem", "content": content}


def _bullets(*items: Node) -> Node:
    return {"type": "bulletList", "content": list(items)}


def _cell(kind: str, value: str) -> Node:
    return {"type": kind, "content": [_paragraph(value)]}


def _row(kind: str, *values: str) -> Node:
    return {"type": "tableRow", "content": [_cell(kind, value) for value in values]}


def _document(*content: Node) -> Node:
    return {"type": "doc", "version": 1, "content": list(content)}


_TYPICAL = _document(
    _heading(1, "Running the ledger"),
    _paragraph("Three machines share one journal volume and take turns holding the lease."),
    _heading(2, "Starting up"),
    _paragraph("Each process reads its settings once, then announces itself on the bus."),
    {
        "type": "codeBlock",
        "attrs": {"language": "bash"},
        "content": [_text("ledgerctl start --wait\nledgerctl status")],
    },
    _heading(2, "Watching it"),
    _paragraph("The dashboard shows one row per machine and one column per measurement."),
    {
        "type": "table",
        "content": [
            _row("tableHeader", "Signal", "Alarm above"),
            _row("tableCell", "lease age", "90 seconds"),
            _row("tableCell", "journal depth", "4096 records"),
        ],
    },
    _bullets(
        _item("green means the lease is fresh"),
        _item("amber means it was renewed late", _bullets(_item("which a rolling restart does"))),
    ),
    {
        "type": "panel",
        "attrs": {"panelType": "warning"},
        "content": [_paragraph("Never take two machines out of service at the same moment.")],
    },
)

_STRUCTURE = _document(
    _heading(1, "Fabric"),
    _paragraph("Every rack holds eight machines, one spare, and a pair of switches."),
    _heading(2, "Cabling"),
    _paragraph("Each machine takes two uplinks, and no two uplinks share a switch."),
    {
        "type": "codeBlock",
        "attrs": {"language": "python"},
        "content": [_text('def uplinks(machine):\n    return (machine + "-a", machine + "-b")')],
    },
    {
        "type": "table",
        "content": [
            _row("tableHeader", "Port", "Speed"),
            _row("tableCell", "1", "25G"),
            _row("tableCell", "2", "25G"),
        ],
    },
    _bullets(
        _item(
            "outermost item",
            _bullets(
                _item(
                    "second level",
                    _bullets(
                        _item(
                            "third level",
                            _bullets(
                                _item(
                                    "fourth level",
                                    _bullets(_item("fifth level, as deep as this corpus goes")),
                                )
                            ),
                        )
                    ),
                )
            ),
        )
    ),
    _heading(2, "Power"),
    {
        "type": "expand",
        "attrs": {"title": "Historic feeds"},
        "content": [_paragraph("Two feeds per rack, from separate boards on separate floors.")],
    },
    {
        "type": "bodiedExtension",
        "attrs": {"extensionKey": "excerpt"},
        "content": [_paragraph("A macro body, which must not swallow the text inside it.")],
    },
    {
        "type": "mediaSingle",
        "content": [
            {
                "type": "media",
                "attrs": {"id": "abc-123", "type": "file", "alt": "Rack elevation drawing"},
            }
        ],
    },
    {"type": "blockCard", "attrs": {"url": "https://example.invalid/neighbouring-page"}},
)

_PREAMBLE = _document(
    _paragraph("A paragraph written above every heading on this page."),
    _heading(1, "The only heading here"),
    _paragraph("And a single paragraph beneath it, addressed by that heading instead."),
)

_HEADING_ONLY = _document(_heading(1, "A stub page that never got written"))

_EMPTY_DOC = _document()

_ASTRAL = _document(
    _heading(1, "🚀 Checklist for 𠀋 builds"),
    _paragraph("The identifier 𡃁 appears in the body as well as in the heading above it."),
    _bullets(_item("🛰 confirm the relay answers"), _item("𠮟 confirm the reviewer signed off")),
)

_UNKNOWN = _document(
    _heading(1, "Nodes from a later schema"),
    {"type": "futureBanner", "content": [_text("Inline content inside an unfamiliar node.")]},
    {
        "type": "futureSection",
        "content": [_paragraph("Block content nested inside another unfamiliar node.")],
    },
    _paragraph("Ordinary prose, written after both of them."),
)

_NOT_ADF = {"kind": "page", "body": "JSON that is valid and is not a document tree"}

_MOJIBAKE = b'{"type": "doc", "version": 1, "content": [{"type": "text", "text": "\xff\xfe("}]}'

_LARGE_TOPICS = (
    "quorum loss",
    "disk pressure",
    "clock drift",
    "certificate expiry",
    "network partition",
    "memory ballooning",
    "queue backlog",
    "replica lag",
    "cache stampede",
    "log rotation",
    "index rebuild",
    "credential rotation",
)


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    _write(dest / "typical.json", _TYPICAL)
    _write(dest / "structure.json", _STRUCTURE)
    _write(dest / "preamble.json", _PREAMBLE)
    _write(dest / "heading-only.json", _HEADING_ONLY)
    _write(dest / "empty-document.json", _EMPTY_DOC)
    _write(dest / "astral.json", _ASTRAL)
    _write(dest / "unknown-nodes.json", _UNKNOWN)
    _write(dest / "not-adf.json", _NOT_ADF)
    _write(dest / "page-large.json", _large())
    (dest / "empty.json").write_bytes(b"")
    (dest / "mojibake.json").write_bytes(_MOJIBAKE)


def _write(path: Path, document: Mapping[str, object]) -> None:
    # sort_keys so the bytes are the same on every machine, which is what makes the
    # determinism assertion a statement about the parser rather than about dict ordering.
    path.write_text(json.dumps(document, indent=1, sort_keys=True), encoding="utf-8")


def _large() -> Node:
    """A page past the size cap, every paragraph of it distinct.

    Distinct by construction rather than by inspection: the section number is zero-padded so
    that section one's heading is not a prefix of section eleven's, and every paragraph names
    both its section and its own position within it.
    """
    content: list[Node] = [_heading(1, "Incident handbook")]
    for index, topic in enumerate(_LARGE_TOPICS, start=1):
        content.append(_heading(2, f"Runbook {index:02d}: {topic}"))
        content.extend(_paragraph(body) for body in _steps(index, topic))
    return _document(*content)


def _steps(index: int, topic: str) -> Sequence[str]:
    return [
        " ".join(
            f"Step {step:02d} of runbook {index:02d} for {topic} continues with observation "
            f"{number:03d}, which records what the operator saw and what they did about it."
            for number in range(1, 21)
        )
        for step in range(1, 10)
    ]
