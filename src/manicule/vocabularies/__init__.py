"""BPE vocabularies: where they come from, and why never from a query.

``tiktoken`` ships no vocabularies in its wheel. Every encoding it knows is a file on a blob
store, fetched on first use — which put a network call on manicule's query path, on a host
that had already indexed a corpus offline and looked like a working install.

Two modules close it, and they are the vocabulary counterparts of
:mod:`manicule.parsers.grammars` and :mod:`manicule.parsers.grammar_bundle`:

:mod:`manicule.vocabularies.store`
    Where ``tiktoken``'s cache is, what is in it, the pre-seed that fills it, and
    :func:`~manicule.vocabularies.store.load_encoding` — the query path's only door, and one
    that cannot open onto the network.
:mod:`manicule.vocabularies.bundle`
    A directory of vocabularies plus a manifest, built on a machine with network access and
    carried to one without, so that a pre-seed succeeds on a host with nothing to fetch from.

Importing this package costs no ``tiktoken`` import: everything that touches the library does
so inside the function that needs it.
"""

from __future__ import annotations

from manicule.vocabularies.store import (
    CACHE_DIR_ENV,
    Blob,
    VocabularyFetchError,
    VocabularyUnavailableError,
    blobs_for,
    bundle_status,
    cache_directory,
    cache_path,
    load_encoding,
    missing_vocabularies,
    prefetch,
    required_encodings,
)

__all__ = [
    "CACHE_DIR_ENV",
    "Blob",
    "VocabularyFetchError",
    "VocabularyUnavailableError",
    "blobs_for",
    "bundle_status",
    "cache_directory",
    "cache_path",
    "load_encoding",
    "missing_vocabularies",
    "prefetch",
    "required_encodings",
]
