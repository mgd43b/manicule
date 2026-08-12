"""Deterministic identifiers.

Ids are derived, not generated. Re-ingesting an unchanged document produces the same ids,
so a re-parse replaces rows rather than accumulating duplicates, and two machines indexing
the same corpus agree on what they called things.
"""

from __future__ import annotations

import hashlib

_HASH = hashlib.blake2b
_DIGEST_BYTES = 16


def _digest(*parts: str | bytes) -> str:
    hasher = _HASH(digest_size=_DIGEST_BYTES)
    for part in parts:
        chunk = part.encode("utf-8") if isinstance(part, str) else part
        # Length-prefixed so that ("ab", "c") and ("a", "bc") cannot collide.
        hasher.update(len(chunk).to_bytes(8, "big"))
        hasher.update(chunk)
    return hasher.hexdigest()


def content_hash(data: bytes | str) -> str:
    """Hash source bytes for change detection and deduplication."""
    return _digest(data)


def document_id(workspace_id: str, source: str, source_id: str) -> str:
    """A document's id: stable across re-ingest, unique across workspaces and connectors.

    **``workspace_id`` is part of the identity, not a filter applied afterwards.** Two
    workspaces indexing the same upstream source are two documents. Deriving the id from
    ``(source, source_id)`` alone makes them one, and the consequence is not a clash anybody
    notices: the second workspace's write lands on the first workspace's row, overwriting
    content its author cannot read, while its own document appears to vanish because the row
    belongs to somebody else. Workspace isolation is enforced on every query, and an identity
    that ignores the workspace defeats it before any query runs.

    The cost is honest and small. The same source synced into two workspaces produces two
    documents, two chunk sets and two sets of vectors — which is what isolation *means*. It
    does not duplicate the corpus itself: retained bytes are content-addressed, so both
    workspaces reference one blob.
    """
    return _digest("document", workspace_id, source, source_id)


def chunk_id(document_id_: str, position: int, text: str) -> str:
    """A chunk's id.

    Includes the text as well as the position, so that editing a document's third paragraph
    changes the third chunk's id and leaves the others alone — the vector rows for unchanged
    chunks survive a re-parse.
    """
    return _digest("chunk", document_id_, str(position), text)


def glossary_entry_id(chunk_id_: str, acronym: str, expansion: str) -> str:
    """A glossary entry's id.

    Derived from the chunk that states the definition and from the definition itself, so
    re-ingesting an unchanged glossary replaces its rows instead of accumulating a second copy
    of every term. Including the expansion means editing a definition produces a new id, which
    is what makes "this document now says something different" a visible change rather than an
    in-place overwrite of a row somebody may already have cited.

    The chunk id already carries the document, which already carries the workspace, so an entry
    minted for one tenant cannot collide with another's.
    """
    return _digest("glossary", chunk_id_, acronym, expansion)


__all__ = ["chunk_id", "content_hash", "document_id", "glossary_entry_id"]
