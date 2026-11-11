"""Where a memory's evidence may point: a web address, or one of a small closed set of local ones.

A structured memory's evidence used to have to be an ``http`` or ``https`` address, held to the
rule a document's own canonical address is (:func:`manicule.core.provenance.require_citable_uri`).
That left no way to cite what was said in a Claude Code session or written in a file under version
control, because neither has a web permalink, and the workarounds were all worse: a fabricated
``https://`` address, no evidence, or a reason kept as prose that loses its date and its quote.

**Evidence only.** A document's canonical address keeps the web-only rule, and so does every
scheme this module does not name: ``file:``, ``javascript:``, ``data:`` and a raw local path stay
refused, with the same sentence as before. A path on one machine is not a citation.

**A closed set, each with its own grammar.** Naming a scheme is not enough to accept it; its
locator must be well-formed, because a locator that cannot be followed is the failure this exists
to avoid. The set is deliberately small and is not an extension point: a scheme is added here, with
its grammar and its tests, or it is refused.

**Two layers, and the split is the point.**

*Whether a locator is well-formed* is a property of the string and is checked by the model
(:class:`manicule.core.knowledge.EvidenceSource`), which has no configuration and is validated on
every read.

*Whether this installation accepts the scheme* is the operator's choice
(:data:`SETTING`), and is checked only where that choice is in hand: at the authoring door and when
a document is ingested. It is **never** checked when a stored record is read back. A stored record
that stopped validating the day an operator switched a scheme off would silently lose its evidence
from every listing, which is the data loss that refusing at the door exists to avoid.

Locators are attested, never verified, exactly like a web one. Nothing here opens a transcript or
a repository.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Final

from manicule.core.provenance import require_citable_uri, require_printable

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

__all__ = [
    "LOCAL_SCHEMES",
    "SETTING",
    "evidence_identity",
    "local_scheme",
    "refuse_disabled_scheme",
    "require_evidence_locator",
]

SETTING: Final = "authoring.local_evidence"
"""The configuration key that turns local schemes on, named in every refusal about one."""

CLAUDE_SESSION: Final = "claude-session"
"""One message in a Claude Code transcript: ``claude-session:<session-uuid>#<message-uuid>``."""

GIT: Final = "git"
"""A file in a Git repository at a commit: ``git:<repo>@<commit>:<path>[#L<start>[-L<end>]]``."""

LOCAL_SCHEMES: Final[tuple[str, ...]] = (CLAUDE_SESSION, GIT)
"""Every scheme an operator can turn on, and the only ones evidence may use besides the web."""

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_UUID_RE = re.compile(_UUID)
_SESSION = re.compile(rf"(?P<session>{_UUID})#(?P<message>{_UUID})")
_INVISIBLE = frozenset({"Cf", "Zl", "Zp"})
"""Format and line-separator characters, refused in a local locator.

Bidirectional overrides and zero-width characters are not control characters, so
:func:`~manicule.core.provenance.require_printable` lets them through, and in a path — the one
locator part that is not constrained to ASCII by its grammar — they reorder or hide what a terminal
prints."""
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_LINES = re.compile(r"L(?P<start>[0-9]{1,9})(?:-L(?P<end>[0-9]{1,9}))?")


def _claim(locator: str) -> tuple[str, str] | None:
    """The local scheme ``locator`` claims and what follows its colon, or ``None``.

    A scheme is recognized **case-insensitively** so that ``Claude-Session:`` is refused as a
    malformed local locator rather than falling through to the web rule and being reported as an
    unknown scheme, which would send its author looking for the wrong fix.

    ``git://host/repo`` is a network address, not a git locator, so it claims nothing and gets the
    web rule's refusal ("cite an http or https address") rather than a grammar lesson.
    """
    scheme, separator, body = locator.partition(":")
    lowered = scheme.lower()
    if not separator or lowered not in LOCAL_SCHEMES or body.startswith("//"):
        return None
    return lowered, body


def local_scheme(locator: str) -> str | None:
    """The local scheme ``locator`` claims, or ``None`` when it claims none."""
    claim = _claim(locator)
    return claim[0] if claim else None


def evidence_identity(locator: str) -> str:
    """What makes two locators the same place, for counting independent sources.

    The locator itself, except that a ``git:`` locator drops its line range: two ranges of one file
    at one commit are one source said twice, not two places. A web address is compared as written,
    as it always was.
    """
    if local_scheme(locator) == GIT:
        return locator.partition("#")[0]
    return locator


def require_evidence_locator(value: str, *, field: str, web_hint: str = "") -> None:
    """Refuse ``value`` unless it is a web address or a well-formed local locator.

    Anything that does not claim one of :data:`LOCAL_SCHEMES` is handed to the web rule unchanged,
    so every refusal that existed before this module does — ``file:``, ``javascript:``, an absent
    scheme, a host-less address — says what it said.

    Whether the scheme is *enabled* is not decided here: see the module note.

    Raises:
        ValueError: Naming ``field`` and what was wrong with the locator.
    """
    claim = _claim(value)
    if claim is None:
        require_citable_uri(value, field=field, hint=web_hint)
        return
    scheme, body = claim
    require_printable(value, field=field)
    for character in value:
        if unicodedata.category(character) in _INVISIBLE:
            msg = (
                f"{field} contains the invisible formatting character U+{ord(character):04X}, "
                f"which could make a locator print as something it is not"
            )
            raise ValueError(msg)
    if value[: len(scheme)] != scheme:
        msg = f"{field} {value!r} must spell its scheme in lowercase: '{scheme}:'"
        raise ValueError(msg)
    if scheme == CLAUDE_SESSION:
        _require_session(body, value=value, field=field)
    else:
        _require_git(body, value=value, field=field)


def refuse_disabled_scheme(locator: str, *, field: str, enabled: Collection[str]) -> None:
    """Refuse ``locator`` if its local scheme is one this installation has not turned on.

    A web address, and a locator that claims no local scheme, are never refused here.

    Raises:
        ValueError: Naming the scheme and :data:`SETTING`, which is what turns it on.
    """
    scheme = local_scheme(locator)
    if scheme is None or scheme in enabled:
        return
    msg = (
        f"{field} uses the '{scheme}' scheme, which this installation has not enabled. "
        f"Add '{scheme}' to {SETTING} in the configuration, or cite an http or https address"
    )
    raise ValueError(msg)


def _require_session(body: str, *, value: str, field: str) -> None:
    if _SESSION.fullmatch(body):
        return
    session, hash_mark, message = body.partition("#")
    if not _is_uuid(session):
        problem = "its session id is not a lowercase UUID"
    elif not hash_mark or not message:
        problem = "it names no message: a locator points at one message, after a '#'"
    else:
        problem = "its message id is not a lowercase UUID"
    msg = (
        f"{field} {value!r} is not a claude-session locator: {problem}. "
        f"The form is claude-session:<session-uuid>#<message-uuid>"
    )
    raise ValueError(msg)


def _require_git(body: str, *, value: str, field: str) -> None:
    form = "The form is git:<repo>@<commit>:<path>#L<first>-L<last>, with the lines optional"

    def refuse(problem: str) -> ValueError:
        return ValueError(f"{field} {value!r} is not a git locator: {problem}. {form}")

    repository, at, rest = body.partition("@")
    if (
        not at
        or not _REPOSITORY.fullmatch(repository)
        or any(segment in {"", ".", ".."} for segment in repository.split("/"))
    ):
        raise refuse(
            "the repository must be a name of letters, digits, '.', '_', '-' or '/', then '@'"
        )
    commit, colon, located = rest.partition(":")
    if not colon or not _COMMIT.fullmatch(commit):
        raise refuse("the commit must be a full 40- or 64-character lowercase hex object id")
    path, hash_mark, lines = located.partition("#")
    _require_repository_path(path, refuse)
    if not hash_mark:
        return
    found = _LINES.fullmatch(lines)
    if found is None:
        raise refuse(f"{lines!r} is not a line range like L10 or L10-L24")
    start = int(found["start"])
    end = int(found["end"]) if found["end"] else start
    if start < 1 or end < start:
        raise refuse("a line range starts at line 1 or later and does not end before it starts")


def _require_repository_path(path: str, refuse: Callable[[str], ValueError]) -> None:
    """A path inside a repository: relative, normalized, and never a way out of it."""
    if not path:
        raise refuse("the path is empty")
    segments = path.split("/")
    if path.startswith("/") or "\\" in path or any(s in {"", ".", ".."} for s in segments):
        raise refuse(
            "the path must be relative to the repository root, with no empty, '.' or '..' "
            "segment and no backslash"
        )


def _is_uuid(text: str) -> bool:
    return _UUID_RE.fullmatch(text) is not None
