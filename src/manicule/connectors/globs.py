"""Path globs, as every connector that admits or refuses a path agrees to read them.

One implementation rather than one per connector, because two would drift and the drift is
invisible: a pattern that admitted a page from a Git site and skipped the same relative path on
disk is wrong in neither file on its own. The rules are stated here once — what a pattern may
say, and what it matches — and a connector chooses only which paths to ask about.

**Patterns are relative and POSIX, and that is checked rather than trusted.** A pattern is
matched against a path relative to whatever the connector calls its boundary — a repository's
content root, a walked directory root — so an absolute pattern, a backslash, or a ``..`` segment
is either a pattern that can never match or one reaching for something outside the boundary.
Both are refused when configuration is loaded, where an operator is still looking at the file,
rather than silently matching nothing three stages later.

**No filesystem is consulted.** Matching is text against text, which is what lets the same
function decide a Git blob path, a walked directory and a file that does not exist yet.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "MAX_PATTERN_LENGTH",
    "admitted",
    "excluded",
    "matches",
    "path_glob",
]

MAX_PATTERN_LENGTH: Final = 1_024
"""The longest a single pattern may be.

A bound rather than no bound, because patterns arrive from configuration and every one of them
is tested against every path. The number is far above any pattern anybody writes on purpose.
"""


def path_glob(value: str) -> str:
    """One configured pattern, validated. Returns it unchanged, or raises.

    Written to be used from a pydantic ``field_validator``, so it raises :class:`ValueError` and
    lets the model report which field and which entry.

    Raises:
        ValueError: The pattern is empty, padded, over :data:`MAX_PATTERN_LENGTH`, absolute,
            Windows-separated, holds a NUL byte, or contains a ``.`` or ``..`` segment.
    """
    if not value or len(value) > MAX_PATTERN_LENGTH or value != value.strip():
        raise ValueError("path patterns must contain 1 to 1024 unpadded characters")
    if "\\" in value or value.startswith("/") or "\0" in value:
        raise ValueError("path patterns must be relative POSIX globs")
    if any(segment in {".", ".."} for segment in value.split("/")):
        raise ValueError("path patterns must not contain dot or traversal segments")
    return value


def matches(path: str, pattern: str) -> bool:
    """Match POSIX globs, including the useful zero-directory meaning of ``**/``.

    ``fnmatch`` alone reads ``**/drafts/**`` as requiring at least one directory before
    ``drafts``, so a top-level ``drafts/`` would survive a pattern written to catch drafts
    anywhere. The second test is that missing case, and nothing else.
    """
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:])
    )


def excluded(path: str, patterns: Sequence[str]) -> bool:
    """Whether any pattern names ``path``. Empty patterns exclude nothing."""
    return any(matches(path, pattern) for pattern in patterns)


def admitted(path: str, *, include: Sequence[str], exclude: Sequence[str]) -> bool:
    """Whether ``path`` is admitted by ``include`` and not then refused by ``exclude``.

    An exclusion always wins, which is the only ordering that lets a broad ``include`` be
    written once and corrected in place rather than being restated as a list of exceptions.
    """
    return excluded(path, include) and not excluded(path, exclude)
