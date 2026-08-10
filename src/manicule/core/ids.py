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


def document_id(source: str, source_id: str) -> str:
    """A document's id: stable across re-ingest, unique across connectors."""
    return _digest("document", source, source_id)


def chunk_id(document_id_: str, position: int, text: str) -> str:
    """A chunk's id.

    Includes the text as well as the position, so that editing a document's third paragraph
    changes the third chunk's id and leaves the others alone — the vector rows for unchanged
    chunks survive a re-parse.
    """
    return _digest("chunk", document_id_, str(position), text)


__all__ = ["chunk_id", "content_hash", "document_id"]
