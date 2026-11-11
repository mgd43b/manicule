"""Remote Git addresses and revisions, validated without importing anything that runs Git.

Configuration validates a source's ``repository`` and ``revision`` when it is loaded, and
configuration is read in every process that starts — including ``manicule doctor`` on a machine
that will never sync. So the rules live here, importing only the standard library, and the module
that actually fetches (:mod:`manicule.connectors.git_mirror`) is loaded when somebody syncs.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "is_remote",
    "mirror_directory",
    "normalize_remote_url",
    "require_fetchable_revision",
]

_UNSAFE_REVISION: Final = re.compile(r"[\s:\\\0\x7f~^?*\[]|\.\.|@\{|^-|^/|/$|//")
"""What a revision may not contain to be spliced into a refspec.

``:`` would let a configured value name a destination ref of its own, a leading ``-`` is an
option, and the rest are the characters Git itself forbids in a ref name — refused here so the
operator reads a sentence about their configuration rather than a failed fetch.
"""


def is_remote(repository: str) -> bool:
    """Whether ``repository`` is a network address rather than a path on this machine."""
    return "://" in repository


def normalize_remote_url(value: str) -> str:
    """One remote address, validated. Returns it without a trailing slash, or raises.

    Written to be called from a pydantic ``field_validator``, so it raises :class:`ValueError`.
    The value is never echoed into the message: an address is exactly the field somebody pastes
    a token into by mistake.
    """
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("repository must be a valid https URL") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(
            "a remote repository must be an https URL with a hostname. Use a path on this "
            "machine for a local checkout; ssh and plain http are not supported"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "repository must not contain credentials. Name the environment variable that holds "
            "the token in token_env instead"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("repository must not contain a query string or a fragment")
    return f"https://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def mirror_directory(root: Path, url: str) -> Path:
    """Where the mirror of ``url`` lives under ``root``.

    Named by a digest of the address, so two sources that point at one remote share one copy and
    a path never carries anything an operator typed.
    """
    return root / hashlib.blake2b(url.encode(), digest_size=10).hexdigest()


def require_fetchable_revision(revision: str) -> None:
    """Refuse a revision that cannot be spliced into a refspec. Raises :class:`ValueError`."""
    if not revision or _UNSAFE_REVISION.search(revision):
        raise ValueError(
            "revision must be a branch, a tag or a commit id: no spaces, ':', '..', and no "
            "leading '-' or '/'"
        )
