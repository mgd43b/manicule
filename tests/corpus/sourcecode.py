"""Fixtures for the code parser, generated rather than committed.

Generation keeps the repository small, makes every fixture's structure reviewable as code
instead of as an opaque blob, and lets the hostile cases exist without being committed at
all. It also makes the awkward ones cheap: a file larger than the size cap, or one holding
deliberately malformed UTF-8, is three lines here and a permanent nuisance in git.

The four kinds ``docs/parsing.md`` §3.5 requires, and what each is here to catch:

**Typical** — an ordinary module in Python, Rust and TypeScript. Three languages because the
symbol path is three different paths: Python is named entirely by the pack's tags query,
Rust needs this repository's table for ``impl`` scoping, and TypeScript's exposed query names
none of its classes or functions at all.

**Structurally hard** — nested classes and methods; a single function far past the block
budget, which is the only thing that exercises the descent in ``_fit``; a decorator-heavy
module, where the block covers a node that is not itself a definition; Rust modules, which
are where a ``::`` separator stops being cosmetic; and one file over the size cap, for the
streaming path.

**Degenerate** — zero bytes, comments and nothing else, and no trailing newline. Each one
breaks a different arithmetic assumption about lines.

**Hostile** — malformed UTF-8, which must be declined rather than indexed as replacement
characters; a syntactically broken file, which must still produce blocks with real line
anchors rather than raising, because a repository always contains one; and astral-plane text
in both an identifier and a string literal, which is where a parser that counts bytes where
it means characters comes apart.

Every line of every generated file is distinct once leading whitespace is collapsed. That is
not tidiness: the round-trip contract's discrimination assertion compares each block's text
against every other block's resolved text, and two blocks that are textually identical cannot
be told apart by anything, including a person reading the citation.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["build"]


_TYPICAL_PYTHON = '''"""Token issuing for the sourcecode fixture corpus."""

import json
import logging

LOGGER = logging.getLogger("corpus.tokens")
DEFAULT_TTL_SECONDS = 3600


class Token:
    """One issued token and the moment it stops being valid."""

    def __init__(self, value: str, expires_at: int) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_live(self, now: int) -> bool:
        """Whether this token has any life left in it."""
        return now < self.expires_at


class TokenStore:
    """Issues tokens and replaces them before they lapse."""

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._issued: dict[str, Token] = {}

    def issue(self, subject: str, at: int) -> Token:
        """Mint a token for one subject."""
        minted = Token(value=subject + "-" + str(at), expires_at=at + self._ttl)
        self._issued[subject] = minted
        return minted

    def refresh(self, subject: str, at: int) -> Token:
        """Replace a subject's token with a later one."""
        previous = self._issued[subject]
        LOGGER.debug("replacing token lapsing at %s", previous.expires_at)
        return self.issue(subject, at)


def load_policy(path: str) -> dict[str, object]:
    """Read an issuing policy from a JSON document."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
'''


_TYPICAL_RUST = """//! Counters keyed by name, for the sourcecode fixture corpus.

use std::collections::HashMap;

pub const DEFAULT_CAPACITY: usize = 16;

pub struct Store {
    entries: HashMap<String, u64>,
}

impl Store {
    pub fn new() -> Self {
        Store {
            entries: HashMap::with_capacity(DEFAULT_CAPACITY),
        }
    }

    pub fn insert(&mut self, key: &str, amount: u64) -> Option<u64> {
        self.entries.insert(key.to_string(), amount)
    }

    pub fn total(&self) -> u64 {
        self.entries.values().copied().sum()
    }
}

pub fn describe(store: &Store) -> String {
    format!("store holding {} in total", store.total())
}
"""


_TYPICAL_TYPESCRIPT = """// Session bookkeeping for the sourcecode fixture corpus.

export interface Session {
    readonly id: string;
    readonly issuedAt: number;
}

export class SessionCache {
    private readonly held = new Map<string, Session>();

    remember(session: Session): void {
        this.held.set(session.id, session);
    }

    lookup(id: string): Session | undefined {
        return this.held.get(id);
    }
}

export function openSession(id: string, clock: () => number): Session {
    return { id, issuedAt: clock() };
}
"""


_NESTED_CLASSES = '''"""Deliberately nested definitions, to check the symbol chain reaches the bottom."""


class Outer:
    """The outermost container."""

    label = "outer"

    class Middle:
        """One level in."""

        label = "middle"

        class Inner:
            """Two levels in, where a naive symbol walk stops being right."""

            def deepest(self, seed: int) -> int:
                """Return something derived from the seed, at maximum depth."""
                return seed * 3 + 1

        def middling(self, seed: int) -> int:
            """A method one level up from the deepest one."""
            return seed * 2

    def shallow(self, seed: int) -> int:
        """A method on the outermost class."""
        return seed + 7
'''


def _oversized_function() -> str:
    """One function whose body alone far exceeds the block budget.

    Nothing else in the corpus reaches the descent in ``_fit``: a file of ordinary
    definitions is split at the top level and never needs a boundary inside one.
    """
    header = [
        '"""A single function too large for one block."""',
        "",
        "",
        "def accumulate_readings(readings: list[int]) -> dict[str, int]:",
        '    """Fold a long list of readings into a summary, at length."""',
        "    summary: dict[str, int] = {}",
    ]
    body = [
        f"    summary[\"reading_{index:03d}\"] = readings[{index}] * {index + 2} "
        f"if len(readings) > {index} else {index}"
        for index in range(48)
    ]
    return "\n".join([*header, *body, "    return summary", ""])


_DECORATED = '''"""Decorator-heavy definitions: the block covers a node that is not itself a definition."""

import functools

REGISTRY: dict[str, object] = {}


def register(name: str):
    """Record a handler under a name."""

    def decorate(target):
        REGISTRY[name] = target
        return target

    return decorate


@register("archive")
@functools.lru_cache(maxsize=32)
def archive_handler(payload: str) -> str:
    """Handle one archive request, memoised."""
    return payload.upper()


@register("restore")
class RestoreHandler:
    """Handle restore requests, with the decorator above the class keyword."""

    @staticmethod
    def handle(payload: str) -> str:
        """Turn a payload back into its original casing."""
        return payload.lower()
'''


_SCOPED_RUST = """//! Nested modules and impls, where `::` stops being cosmetic.

pub mod ledger {
    pub struct Anchor {
        pub page: u32,
    }

    impl Anchor {
        pub fn render(&self) -> String {
            format!("p. {}", self.page)
        }
    }

    pub mod audit {
        pub fn checksum(seed: u32) -> u32 {
            seed.wrapping_mul(2_654_435_761)
        }
    }
}

pub fn describe_anchor(anchor: &ledger::Anchor) -> String {
    anchor.render()
}
"""


_COMMENTS_ONLY = """# This file holds nothing but commentary.
# It exists because a file of comments still has content worth citing,
# and a parser that emits no blocks for it has silently lost a page of prose.
# Two blank comment lines follow, which are the awkward part.
#
#
# The last line carries no newline problem of its own.
"""


_NO_TRAILING_NEWLINE = '''"""No newline closes this file, which is where line counting usually breaks."""

FIRST_CONSTANT = 1


def only_function() -> int:
    """Return the one constant this module defines."""
    return FIRST_CONSTANT'''


_BROKEN_SYNTAX = '''"""Syntactically broken on purpose; the grammar produces ERROR nodes for it."""

import sys


def unterminated(argument:
    """This signature never closes, so the tree below here is a guess."""
    return argument


class MissingColon
    def orphan(self):
        return sys.maxsize
'''


_ASTRAL = '''"""Astral-plane text in an identifier and in a string literal."""

BANNER = "\U0001f30d spans the whole world \U0001f680"


def \U00020022_measure(sample: str) -> int:
    """Named with a CJK Extension B ideograph, which is four bytes and one character."""
    return len(sample)


class \U00020022Registry:
    """A class whose name also begins outside the basic multilingual plane."""

    def describe(self) -> str:
        """Return the banner, emoji and all."""
        return BANNER
'''


_BASH = """#!/usr/bin/env bash
# Environment preparation, for the language whose symbols come from the node-type table.
set -euo pipefail

WORKSPACE_ROOT="/srv/manicule"

function prepare_workspace() {
  mkdir -p "${WORKSPACE_ROOT}/cache"
  chmod 0750 "${WORKSPACE_ROOT}"
}

function report_disk_usage() {
  du -sh "${WORKSPACE_ROOT}/cache" || echo "cache directory is absent"
}

prepare_workspace
report_disk_usage
"""


_SQL = """-- Totals by region, for the language with no symbol source at all.
CREATE TABLE regional_totals (
    region TEXT NOT NULL,
    amount INTEGER NOT NULL DEFAULT 0
);

INSERT INTO regional_totals (region, amount) VALUES ('emea', 41);

SELECT region, SUM(amount) AS total
FROM regional_totals
GROUP BY region
ORDER BY total DESC;
"""


def _large_module() -> str:
    """The one file over the 256 KiB cap, for the streaming path.

    Every generated line carries its own index so that no two blocks are textually equal —
    two identical blocks would fail discrimination for a reason that has nothing to do with
    the parser.
    """
    parts = ['"""A generated module large enough to exceed the fixture size cap."""', ""]
    for index in range(110):
        parts.append("")
        parts.append(f"def stage_{index:03d}(payload: dict[str, int]) -> dict[str, int]:")
        parts.append(f'    """Stage {index:03d} of a long generated pipeline."""')
        parts.append(f"    result_{index:03d} = dict(payload)")
        for step in range(28):
            parts.append(
                f'    result_{index:03d}["step_{index:03d}_{step:02d}"] = '
                f"payload.get(\"seed\", {step}) * {index * 31 + step + 1}"
            )
        parts.append(f"    return result_{index:03d}")
        parts.append("")
    return "\n".join(parts) + "\n"


_TEXT_FIXTURES: dict[str, str] = {
    "typical-tokens.py": _TYPICAL_PYTHON,
    "typical-store.rs": _TYPICAL_RUST,
    "typical-sessions.ts": _TYPICAL_TYPESCRIPT,
    "hard-nested-classes.py": _NESTED_CLASSES,
    "hard-decorated-handlers.py": _DECORATED,
    "hard-scoped-modules.rs": _SCOPED_RUST,
    "table-symbols.sh": _BASH,
    "absent-symbols.sql": _SQL,
    "degenerate-comments-only.py": _COMMENTS_ONLY,
    "degenerate-no-trailing-newline.py": _NO_TRAILING_NEWLINE,
    "hostile-broken-syntax.py": _BROKEN_SYNTAX,
    "hostile-astral.py": _ASTRAL,
}

MALFORMED_UTF8 = b'"""Latin-1 in a file declared as UTF-8."""\n\n\ndef caf\xe9() -> int:\n    return 1\n'
"""A byte sequence no UTF-8 decoder accepts. ``0xE9`` opens a three-byte sequence that never
arrives, so this cannot be mistaken for an encoding the parser should have guessed."""


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)

    for name, content in _TEXT_FIXTURES.items():
        (dest / name).write_text(content, encoding="utf-8")

    (dest / "hard-oversized-function.py").write_text(_oversized_function(), encoding="utf-8")
    # The one deliberate exception to the corpus size cap, named so the intent is visible in
    # the directory listing rather than only in the generator.
    (dest / "module-large.py").write_text(_large_module(), encoding="utf-8")

    # Zero bytes, written explicitly rather than by touching the path, so the fixture is
    # obviously deliberate to anyone reading the directory.
    (dest / "degenerate-empty.py").write_bytes(b"")
    (dest / "hostile-malformed-utf8.py").write_bytes(MALFORMED_UTF8)
