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

import glob
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "MAX_PATTERN_LENGTH",
    "admitted",
    "matches",
    "matches_any",
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


@lru_cache(maxsize=1_024)
def _matcher(pattern: str) -> re.Pattern[str]:
    """One pattern compiled, kept because every pattern is tested against every path.

    Bounded rather than unbounded: patterns come from configuration, so the live set is a
    handful, and a cap means a caller that ever passes arbitrary text cannot grow this without
    limit. :func:`glob.translate` sanitizes a malformed bracket expression rather than emitting
    a regex that will not compile, so a pattern that passed :func:`path_glob` cannot raise here.
    """
    return re.compile(glob.translate(pattern, recursive=True, include_hidden=True, seps="/"))


def matches(path: str, pattern: str) -> bool:
    """Whether ``pattern`` names ``path``, reading the glob the way a POSIX shell does.

    **``*`` stops at ``/`` and ``**`` crosses it.** So ``archive/*`` is the one level and
    ``archive/**`` is the subtree, and ``*.md`` is the Markdown at the top rather than every
    ``.md`` in the corpus. That is what anybody writing the pattern means, and it is the
    difference between an exclusion that names a directory and one that quietly reaches into
    every directory under it.

    ``**/`` matches zero directories as well as more, so ``**/drafts/**`` catches a top-level
    ``drafts/`` too. That used to need a special case here, spelled as a second ``fnmatch``
    against the pattern with its ``**/`` removed; the recursive translation has the meaning
    built in, so the special case is gone rather than merely working.

    ``include_hidden`` is on because whether a dot-file is a document is the connector's
    question, not the matcher's — the filesystem connector has a setting for it, and a glob
    that silently refused to match ``.github/**`` would be answering it from here.
    """
    return _matcher(pattern).match(path) is not None


def matches_any(path: str, patterns: Sequence[str]) -> bool:
    """Whether any of ``patterns`` names ``path``. No patterns match nothing.

    One primitive named for what it does rather than for what either caller wants it to mean:
    the same test decides admission and refusal, and a helper called ``excluded`` would read as
    a lie in :func:`admitted`, where it is asked about the *include* list.
    """
    return any(matches(path, pattern) for pattern in patterns)


def admitted(path: str, *, include: Sequence[str], exclude: Sequence[str]) -> bool:
    """Whether ``path`` is admitted by ``include`` and not then refused by ``exclude``.

    An exclusion always wins, which is the only ordering that lets a broad ``include`` be
    written once and corrected in place rather than being restated as a list of exceptions.
    """
    return matches_any(path, include) and not matches_any(path, exclude)
