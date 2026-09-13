"""What every vector backend stores, and what it is allowed to be asked.

Two backends now hold vectors — an embedded LanceDB directory and a network-backed Qdrant
collection — and the parts of a stored vector that are *not* a property of either engine live
here: the names of the fields a row carries, the digest that names the space they belong to,
the :class:`~manicule.core.retrieval.Filter` fields a store may answer by itself, and the
read-side verdict on a row's numbers.

**One rule stated once, rather than one rule per backend.** A field name, a filter exemption
or an integrity verdict duplicated into a second module is a rule that can be changed in one
place and not the other, and the two only disagree on the rows a person is looking at during
an incident. ``docs/storage.md`` §6.2 is the design these names implement; §6.3 is the Qdrant
store that reads them back out of a payload rather than a column.

Nothing here imports a database, and that is load-bearing rather than tidy: an installation
that configures Qdrant must not need LanceDB and PyArrow on disk to find out what a row is
called. ``tests/test_import_boundary.py`` holds the line.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Final, cast

from manicule.core.embedding import (
    FLOAT32_EPSILON,
    UNRECORDED_CHECKSUM,
    VectorIntegrity,
    is_finite_vector,
    verify_stored_checksum,
)

if TYPE_CHECKING:
    from manicule.core.embedding import EmbedFingerprint, Vector
    from manicule.core.retrieval import Filter

ID_COLUMN: Final = "id"
"""The physical row key: ``vector_id(publication_id, chunk_id)``."""

CHUNK_ID_COLUMN: Final = "chunk_id"
PUBLICATION_COLUMN: Final = "publication_id"
VECTOR_COLUMN: Final = "vector"
CHUNK_COLUMN: Final = "chunk_json"
IDENTITY_COLUMN: Final = "embed_identity"
CHECKSUM_COLUMN: Final = "vector_checksum"
CHECKSUM_VERSION_COLUMN: Final = "vector_checksum_version"
SOURCE_VECTOR_ID_COLUMN: Final = "source_vector_id"
SOURCE_PUBLICATION_COLUMN: Final = "source_publication_id"
SOURCE_SEQUENCE_COLUMN: Final = "source_sequence"
SOURCE_CREATED_AT_COLUMN: Final = "source_created_at"

DOCUMENT_ID_COLUMN: Final = "document_id"
KIND_COLUMN: Final = "kind"
LANG_COLUMN: Final = "lang"
POSITION_COLUMN: Final = "position"

FINGERPRINT_HASH_LENGTH: Final = 8

TABLE_PREFIX: Final = "chunks__"
"""Vector spaces are named ``chunks__<fp8>`` — a Lance table, or a Qdrant collection."""

FILTERABLE_COLUMNS: Final = frozenset(
    {DOCUMENT_ID_COLUMN, KIND_COLUMN, LANG_COLUMN, POSITION_COLUMN}
)
"""The promoted fields a predicate may name.

An allowlist rather than a convention: every field name that reaches a predicate is checked
against this set, so a future edit that threads a name in from somewhere less trustworthy
fails loudly instead of composing a query out of it.
"""

PUSHED_DOWN_FILTER_FIELDS: Final = frozenset({"document_ids", "kinds", "langs"})
""":class:`~manicule.core.retrieval.Filter` fields a vector store can answer by itself.

One entry per promoted field. Every other field needs a join no vector backend has the columns
for; those are resolved in the document store first and arrive here as ``document_ids``
(``docs/retrieval.md`` §3.3).

Shared by both backends deliberately, and not widened for the one whose engine could answer
more. Qdrant can filter a payload on anything written into it, so a Qdrant-only exception here
would be easy and would make dense retrieval mean something different depending on which store
an installation configured — the same query returning different rows on two backends is the
outcome ``docs/retrieval.md`` §3.3 exists to prevent.
"""

EXEMPT_FILTER_FIELDS: Final = frozenset({"workspace_ids"})
""":class:`~manicule.core.retrieval.Filter` fields a vector store neither honors nor refuses.

**A named exemption rather than an omission, because the two look identical in a loop and only
one of them is deliberate.** ``workspace_ids`` is a security boundary (``PLAN.md`` §14), and no
vector backend carries a field for it — not by oversight but by design: tenancy and liveness
live on ``documents`` in the authoritative store, and copying them into a derived one creates a
value that can disagree (``docs/storage.md`` §6.2).

The boundary therefore moved rather than disappeared. It is enforced by the hydrating join
inside the dense stage (``docs/retrieval.md`` §4.2), which is also what stops soft-deleted and
cross-workspace rows consuming top-``k`` slots, and
:func:`manicule.testing.assert_pipeline_enforces_scope` is what holds a pipeline to it.
"""


def fingerprint_hash(fingerprint: EmbedFingerprint) -> str:
    """The short hash that names a vector space (``docs/storage.md`` §6.5).

    Taken over the canonical identity bytes, so two fingerprints share a name only if they
    share a vector space.
    """
    digest = hashlib.sha256(fingerprint.canonical().encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_HASH_LENGTH]


def space_name(fingerprint: EmbedFingerprint) -> str:
    """The Lance table, or Qdrant collection, these vectors belong in."""
    return f"{TABLE_PREFIX}{fingerprint_hash(fingerprint)}"


def unhonored_filter_fields(filter: Filter | None) -> list[str]:  # noqa: A002 - the domain's word
    """The fields ``filter`` restricts on that no vector store can honor, sorted.

    Empty for the two cases that are the same instruction to a store — no filter at all, and a
    filter restricting only fields resolved elsewhere. A caller turns a non-empty result into a
    refusal; quietly dropping a restriction returns results the filter was written to exclude,
    and the search still looks like it worked.
    """
    if filter is None:
        return []
    return sorted(filter.restricting_fields - PUSHED_DOWN_FILTER_FIELDS - EXEMPT_FILTER_FIELDS)


def refuse_unhonored_fields(filter: Filter | None) -> None:  # noqa: A002 - the domain's word
    """Raise unless every field ``filter`` restricts on is one a vector store can answer.

    One message for both backends, because the instruction it gives the caller — resolve those
    fields in the document store and pass the result as ``document_ids`` — is a property of the
    retrieval design rather than of an engine.

    Raises:
        ValueError: When ``filter`` sets a field no vector store can honor and that has not
            been granted an exemption.
    """
    unhonored = unhonored_filter_fields(filter)
    if not unhonored:
        return
    msg = (
        f"the vector store has no field for {', '.join(unhonored)}, so it cannot honor "
        f"{'them' if len(unhonored) > 1 else 'it'}. Resolve those fields in the document "
        f"store and pass the result as document_ids; ignoring them here would return results "
        f"the filter was written to exclude."
    )
    raise ValueError(msg)


def embed_text_of(record: dict[str, Any]) -> str | None:
    """The ``embed_text`` of the chunk a row carries, or ``None`` if the row cannot say.

    Reads the one field rather than validating the whole :class:`~manicule.core.content.Chunk`,
    because this is a hot read on every document a sweep touches and the rest of the model is
    not being asked about. ``None`` covers every way the field can fail to answer — not JSON,
    not an object, no ``embed_text``, an ``embed_text`` that is not a string — since they all
    mean the same thing to the caller: this row cannot be checked against itself.
    """
    try:
        decoded = json.loads(str(record[CHUNK_COLUMN]))
    except (ValueError, TypeError, KeyError):
        return None
    if not isinstance(decoded, dict):
        return None
    value = cast("dict[str, object]", decoded).get("embed_text")
    return value if isinstance(value, str) else None


def checksum_of(record: dict[str, Any]) -> tuple[str, str]:
    """The numerical-integrity pair a row carries, as two strings.

    ``get`` rather than indexing, because a space that predates the fields does not have them
    and a row read before a migration has run therefore answers nothing at all. Absent, ``NULL``
    and empty all mean the same thing here — no checksum was recorded — and
    :func:`~manicule.core.embedding.verify_stored_checksum` decides whether that is a refusal.
    """
    return (
        str(record.get(CHECKSUM_COLUMN) or UNRECORDED_CHECKSUM),
        str(record.get(CHECKSUM_VERSION_COLUMN) or UNRECORDED_CHECKSUM),
    )


def row_integrity(record: dict[str, Any]) -> VectorIntegrity:
    """The numerical verdict on one row read straight from a vector store.

    The read-path half of the reuse classification, for the queries that have a vector and its
    checksum but no chunk to classify against — search results and the coverage scan.
    Provenance is not its business and it does not pretend otherwise: a row can be
    :attr:`~manicule.core.embedding.VectorIntegrity.VERIFIED` here and still be stale, which is
    what makes the two checks two checks.

    A record with *neither* checksum field came from a space that predates them, and reads as
    unverified. That is a different absence from a row whose *vector* is null, which is a row
    nothing can be established about, and conflating the two would drop every result a
    pre-upgrade store returns.

    **Both fields, not one.** A record carrying the version and not the digest is not a row
    that predates the pair — it is a row whose two halves were not written together, which
    :func:`~manicule.core.embedding.verify_stored_checksum` calls malformed and a search must
    drop. Testing only the digest would let such a row through as merely unverified, and the
    coverage count that reports it as malformed would then disagree with the search that ranked
    it. A Lance table cannot produce one, because both columns arrive in one migration; a
    payload is per-point and can.
    """
    if CHECKSUM_COLUMN not in record and CHECKSUM_VERSION_COLUMN not in record:
        return VectorIntegrity.UNVERIFIED
    stored = record.get(VECTOR_COLUMN)
    if stored is None:
        return VectorIntegrity.UNREADABLE
    values = [float(value) for value in stored]
    if not is_finite_vector(values):
        return VectorIntegrity.NON_FINITE
    checksum, version = checksum_of(record)
    return verify_stored_checksum(values, recorded=checksum, version=version, required=False)


def unit(vector: Vector) -> list[float]:
    """``vector`` scaled to length one, so that cosine distance is ``1 - similarity``.

    A vector of all zeros has no direction and is returned unchanged. Nothing here can invent
    one for it, and cosine similarity against it is undefined rather than small — see each
    store's ``search`` for what it does about that.

    **A vector already of unit length within float32 precision is also returned unchanged**,
    and that is what makes a reused vector a reused vector rather than a very slightly
    different one. Read a stored vector back and the ``float32`` rounding leaves its length a
    few parts in 10^8 from one; dividing by that length and rounding to ``float32`` again lands
    on a different value in roughly one row in five hundred, measured. So without this,
    re-writing a row with the vector it already holds would perturb the odd row by one ulp, and
    "the vector was not recomputed" would be a claim no test could make exactly. The correction
    being skipped is smaller than :data:`~manicule.core.embedding.FLOAT32_EPSILON`, which is
    smaller than the stored representation can express: it moves bits and cannot move meaning.
    """
    values = [float(value) for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0 or abs(norm - 1.0) < FLOAT32_EPSILON:
        return values
    return [value / norm for value in values]


__all__ = [
    "CHECKSUM_COLUMN",
    "CHECKSUM_VERSION_COLUMN",
    "CHUNK_COLUMN",
    "CHUNK_ID_COLUMN",
    "DOCUMENT_ID_COLUMN",
    "EXEMPT_FILTER_FIELDS",
    "FILTERABLE_COLUMNS",
    "FINGERPRINT_HASH_LENGTH",
    "IDENTITY_COLUMN",
    "ID_COLUMN",
    "KIND_COLUMN",
    "LANG_COLUMN",
    "POSITION_COLUMN",
    "PUBLICATION_COLUMN",
    "PUSHED_DOWN_FILTER_FIELDS",
    "SOURCE_CREATED_AT_COLUMN",
    "SOURCE_PUBLICATION_COLUMN",
    "SOURCE_SEQUENCE_COLUMN",
    "SOURCE_VECTOR_ID_COLUMN",
    "TABLE_PREFIX",
    "VECTOR_COLUMN",
    "checksum_of",
    "embed_text_of",
    "fingerprint_hash",
    "refuse_unhonored_fields",
    "row_integrity",
    "space_name",
    "unhonored_filter_fields",
    "unit",
]
