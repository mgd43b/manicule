"""``CAPABILITIES.md`` counts what it lists, and its MCP section names what is registered.

**What this checks, and what it deliberately does not.** The document is a hand-maintained
ledger, and every number in it was previously a claim nobody verified — which is the shape this
project keeps failing on. Two things here are cheap to derive and are therefore derived:

* **Each heading's number equals the entries under it**, and the summary table agrees with the
  headings, and the total is their sum. This is self-consistency rather than truth, but it is
  the error that actually happens: adding an entry and updating one number of the three. It
  caught exactly that in the change that introduced it — a section grew by four entries and the
  heading was raised by three.
* **The MCP section names the registered tools**, compared as a set of names rather than a
  count, because a count agrees with itself while naming the wrong tool.

The CLI and HTTP sections are *not* checked against the code, and pretending otherwise would be
worse than leaving them alone. Neither is a one-to-one list of anything the code has: the CLI
section maps another tool's verb vocabulary onto manicule's commands — 35 ticked entries
against 20 commands — and the HTTP section is a curated list of endpoints, 46 ticked against 76
walked route objects, because a websocket, a redirect and an unversioned liveness probe are all
routes and none is an item in that ledger. A comparison between those pairs would need a
mapping maintained by hand, and a hand-maintained mapping asserting a hand-maintained list is a
second thing to get wrong wearing the costume of a check.
"""

from __future__ import annotations

import re
from pathlib import Path

from manicule.mcp.server import TOOL_NAMES

CAPABILITIES = Path(__file__).resolve().parents[1] / "CAPABILITIES.md"

BUILT = re.compile(r"^- \[x\] ", re.MULTILINE)
ABSENT = re.compile(r"^- \[ \] ", re.MULTILINE)
HEADING = re.compile(r"^## (?P<area>[^\n—]+?) — (?P<count>\d+)$", re.MULTILINE)
TABLE_ROW = re.compile(r"^\| (?P<area>[A-Za-z ]+?) \| (?P<count>\d+) \|", re.MULTILINE)
TOTAL_ROW = re.compile(r"^\| \*\*Total\*\* \| \*\*(?P<count>\d+)\*\* \|", re.MULTILINE)


def _document() -> str:
    return CAPABILITIES.read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    """Each ``## Area — N`` heading mapped to the body under it."""
    found: dict[str, str] = {}
    for match in HEADING.finditer(text):
        body = text[match.end() :]
        nextish = body.find("\n## ")
        found[match.group("area").strip()] = body if nextish < 0 else body[:nextish]
    return found


def test_every_heading_counts_the_entries_beneath_it() -> None:
    """The error that actually happens: four entries added, the number raised by three."""
    text = _document()
    sections = _sections(text)
    assert sections, "no '## Area — N' headings were found; the parse collapsed"

    for match in HEADING.finditer(text):
        area = match.group("area").strip()
        claimed = int(match.group("count"))
        body = sections[area]
        entries = len(BUILT.findall(body)) + len(ABSENT.findall(body))
        assert claimed == entries, (
            f"'## {area} — {claimed}' lists {entries} entries. Struck-through entries count "
            f"too: they are capabilities deliberately declined, which is a decision the "
            f"ledger records rather than an item it omits."
        )


def test_the_summary_table_agrees_with_the_headings_and_its_own_total() -> None:
    """Three numbers per area, written in two places, and nothing reconciled them."""
    text = _document()
    headings = {
        match.group("area").strip(): int(match.group("count")) for match in HEADING.finditer(text)
    }
    rows = {
        match.group("area").strip(): int(match.group("count")) for match in TABLE_ROW.finditer(text)
    }
    assert rows, "the summary table did not parse"

    for area, claimed in rows.items():
        if area in headings:
            assert claimed == headings[area], (
                f"the summary table says {area} has {claimed}; its heading says {headings[area]}"
            )

    total = TOTAL_ROW.search(text)
    assert total is not None, "the summary table has no total row"
    assert int(total.group("count")) == sum(rows.values()), (
        f"the total says {total.group('count')}; the rows sum to {sum(rows.values())}"
    )


def test_the_mcp_section_names_the_tools_the_server_registers() -> None:
    """Names, not a count.

    A count agrees with itself while naming a tool that does not exist, and the ledger is read
    by people deciding whether a capability is there. ``TOOL_NAMES`` is itself checked against
    the registered decorators at server build time, so this closes the last gap between what
    is registered, what is listed in code, and what is claimed in prose.
    """
    body = _sections(_document())["MCP tools"]
    listed = set(re.findall(r"^- \[x\] `([a-z_]+)`$", body, re.MULTILINE))
    assert listed == set(TOOL_NAMES), (
        f"the MCP section and the registered tools disagree. Listed and not registered: "
        f"{sorted(listed - set(TOOL_NAMES)) or 'none'}. Registered and not listed: "
        f"{sorted(set(TOOL_NAMES) - listed) or 'none'}."
    )
