"""Fixtures for the Markdown and MDX parser, generated rather than committed.

Generation keeps the repository small, makes every fixture's structure reviewable as code
rather than as an opaque blob, and lets the awkward cases — malformed UTF-8, a file past the
size cap — be three lines here instead of a permanent nuisance in git.

The four kinds ``docs/parsing.md`` §3.5 requires, and what each is here to catch:

**Typical** — a release runbook with front matter, sections, a fenced block and a list: the
shape of nine documents in ten.

**Structurally hard** — a heading path that repeats, written once as an ATX heading and once
as a setext one; a second "Configuration" under a different parent; a table; a fence; and a
list nested five deep. The repeat is the point: two sections with the same path must still be
resolvable, which is what the fragment is for.

**Degenerate** — zero bytes, a heading with nothing under it, and a file with no trailing
newline. Each breaks a different assumption about line arithmetic.

**Hostile** — malformed UTF-8, which must be declined rather than indexed as replacement
characters, and astral-plane text in a heading, which is where anything counting bytes where
it means characters comes apart.

Two rules constrain every line written here, and both come from the round-trip contract
rather than from taste:

- **No block's text may repeat, or contain another block's text.** The discrimination
  assertion (``docs/parsing.md`` §3.3) compares each block's text against every other
  anchor's resolved text, and two blocks that read identically cannot be told apart by
  anything — including a person reading the citation.
- **A repeated heading needs distinguishable source lines.** Markdown block text is the
  source slice, so ``## Overview`` and its setext twin differ even though their paths do not,
  which is how this corpus can carry a repeating heading path at all.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["LARGE_SECTIONS", "build"]

LARGE_SECTIONS = 12
"""Sections in the over-cap fixture. Few and long rather than many and short: the streaming
path is exercised by the byte count, while the assertions that compare every block against
every other are quadratic in the block count."""

_TYPICAL = """---
title: Deploying the ledger service
reviewed: 2026-08-01
---

# Deploying the ledger service

The ledger runs on three machines and keeps its journal on a shared volume.

## Before you start

Check that the volume is mounted and that nobody else holds the deploy lock.

- Confirm the mount point answers
- Confirm the lock is free
- Announce the window in the operations channel

## Running it

Copy the artefact into place, then restart each machine one at a time.

```bash
ledgerctl roll --wait
```

Wait for the health endpoint to answer before moving on to the next one.

## If it goes wrong

Put the previous artefact back; it stays on disk for seven days.

> Going backwards does not undo a schema migration. Those apply forward only.
"""

_STRUCTURE = """# Alpha

Prose that belongs to the top-level section and to nothing under it.

## Configuration

Alpha keeps its settings in a single file, read once at boot.

```python
def load(path: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in open(path))
```

| Setting | Default |
|---|---|
| retries | 3 |
| timeout | 30s |

## Overview

The first sibling with this title, written as an ATX heading.

Overview
--------

The second sibling, written the setext way and sharing the path above.

# Beta

Configuration
-------------

Beta reads the same keys from the environment instead of from disk.

- outermost item
  - second level
    - third level
      - fourth level
        - fifth level, as deep as this corpus goes
"""

_PREAMBLE = """Two paragraphs sit above the first heading of this file.

Both are addressed by line number, which Markdown reports exactly and for free.

# The only heading here

And a single paragraph beneath it, addressed by that heading instead.
"""

_HEADING_ONLY = "# A stub page that never got written\n"

_NO_TRAILING_NEWLINE = "# Terse\n\nOne paragraph, and no newline where a file usually ends."

_ASTRAL = """# 🚀 Launch checklist for 𠀋 builds

The identifier 𡃁 appears in the body as well as in the heading above it.

- 🛰 confirm the relay is up
- 𠮟 confirm the reviewer signed off
"""

_MOJIBAKE = b"# A heading that decodes\n\nAnd a paragraph that does not: \xff\xfe\x00\xc3\x28\n"

_COMPONENTS = """import { Callout } from "../components/callout"

# Ledger widgets

<Callout kind="warning">

Widgets are cached for an hour, so a change takes that long to show up.

</Callout>

Ordinary prose, written after the component and before the next one.

<Note>
Tight components leave their children as Markdown even without blank lines.
</Note>

<Chart data={series} />

The figure above is generated when the site is built.
"""

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
    _write(dest / "typical.md", _TYPICAL)
    _write(dest / "structure.md", _STRUCTURE)
    _write(dest / "preamble.md", _PREAMBLE)
    _write(dest / "heading-only.md", _HEADING_ONLY)
    _write(dest / "no-trailing-newline.md", _NO_TRAILING_NEWLINE)
    _write(dest / "empty.md", "")
    _write(dest / "astral.md", _ASTRAL)
    _write(dest / "components.mdx", _COMPONENTS)
    (dest / "mojibake.md").write_bytes(_MOJIBAKE)
    _write(dest / "handbook-large.md", _large())


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _large() -> str:
    """A document past the size cap, every paragraph of it distinct.

    Distinct by construction rather than by inspection: the section number is zero-padded so
    that section one's heading is not a prefix of section eleven's, and every paragraph names
    both its section and its own position within it.
    """
    parts: list[str] = ["# Incident handbook\n"]
    for index, topic in enumerate(_LARGE_TOPICS, start=1):
        parts.append(f"\n## Runbook {index:02d}: {topic}\n")
        for paragraph in range(1, 10):
            body = " ".join(
                f"Step {paragraph:02d} of runbook {index:02d} for {topic} continues with "
                f"observation {number:03d}, which records what the operator saw and what "
                f"they did about it."
                for number in range(1, 21)
            )
            parts.append(f"\n{body}\n")
    return "".join(parts)
