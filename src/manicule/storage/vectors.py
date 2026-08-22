"""LanceDB behind :class:`~manicule.core.protocols.VectorStore`.

``docs/storage.md`` §6.2, §6.3 and §6.5 are the design this implements. The three things it
exists to get right, in the order they bite:

**The dimension is never a literal.** The vector table is created on first
:meth:`LanceVectorStore.ensure_ready` from :attr:`EmbedFingerprint.dimension`, and the row
schema is built at run time from that number. There is no constant to fall out of date, no
buffer sized ahead of the model, and no assertion that knows better than the embedder.

**The directory says what it holds.** A one-row ``_manicule_meta`` table beside the vectors
carries the fingerprint they were produced with, so swapping one ``vectors/`` directory for
another instance's — or restoring half a backup — is detectable rather than merely wrong.
The vector table is named ``chunks__<fp8>`` after the first eight hex characters of the
fingerprint hash, so two spaces cannot occupy one name even for the length of a rebuild.

**A different model is refused, not accepted at the same size.** Comparison goes through
:meth:`~manicule.core.fingerprints.Fingerprint.require_match`, which compares the canonical
serialization byte for byte. Two unrelated 1024-dimension models pass any size check and
turn every stored vector into noise relative to every new query, with nothing downstream
able to notice.

Two deliberate departures from ``docs/storage.md`` §6.2, both recorded here because the
document is the thing that has to change if they are wrong:

``chunk_json``
    §6.2 says the Lance row holds no text: a search returns ``(id, distance)`` and SQLite
    hydrates. But :meth:`~manicule.core.protocols.VectorStore.search` returns
    :class:`~manicule.core.retrieval.Candidate`, which carries a whole
    :class:`~manicule.core.content.Chunk`, and
    :func:`~manicule.testing.assert_vector_store_is_dimension_agnostic` builds the store
    with nothing to hydrate from. So the chunk travels with the vector. It travels as one
    canonical JSON column rather than as a spread of columns because
    :meth:`~pydantic.BaseModel.model_dump_json` and its inverse round-trip whatever ``Chunk``
    has *now*: a hand-written column mapping is a second place to remember when the type
    gains a field, and the field nobody remembers is the one that goes missing in silence.

``position`` is ``int64``, not ``int32``
    The schema is declared through :class:`lancedb.pydantic.LanceModel` so that nothing here
    imports ``pyarrow``, which ships no type information and would put an untyped library in
    the middle of every schema expression. Python ``int`` maps to ``int64``. The column is a
    filter target, not a storage cost worth an untyped dependency.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
#
# lancedb ships `py.typed`, but annotates its public surface in terms of `pyarrow`, which
# ships neither `py.typed` nor stubs — so every lancedb call site is "partially unknown"
# through no fault of the code here, and a per-call `# pyright: ignore` would mean roughly
# thirty of them, each brittle under `reportUnnecessaryTypeIgnoreComment`. The suppression
# is scoped to this file and to the three diagnostics the untyped dependency causes;
# everything else pyright checks in strict mode still applies. Values crossing back out of
# lancedb are converted explicitly, so nothing untyped escapes this module.

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

import lancedb
from lancedb.index import IvfPq
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector as FixedSizeVector
from lancedb.query import ColumnOrdering
from pydantic import create_model

from manicule.core.ann import (
    PQ_CODE_BITS,
    AnnIndex,
    AnnIndexBuild,
    AnnIndexState,
    ann_index_name,
    classify,
    parse_ann_index_name,
    partitions_for,
    sub_vectors_for,
)
from manicule.core.content import LEGACY_PUBLICATION, Chunk
from manicule.core.embedding import (
    FLOAT32_EPSILON,
    UNRECORDED_CHECKSUM,
    UNRECORDED_IDENTITY,
    VECTOR_CHECKSUM_VERSION,
    EmbedFingerprint,
    StoredVector,
    VectorChecksumBackfill,
    VectorChecksumCoverage,
    VectorIntegrity,
    VectorState,
    canonical_stored_vector,
    choose_stored_vector,
    classify_stored_vector,
    embedding_input_identity,
    is_finite_vector,
    vector_checksum,
    verify_stored_checksum,
)
from manicule.core.errors import ManiculeError
from manicule.core.ids import vector_id
from manicule.core.retrieval import Candidate, Filter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from lancedb.db import AsyncConnection
    from lancedb.table import AsyncTable
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.embedding import Vector
    from manicule.ingest.reembed import SnapshotChunk

META_TABLE: Final = "_manicule_meta"
"""Where the fingerprint lives, beside the vectors it describes."""

TABLE_PREFIX: Final = "chunks__"
"""Vector tables are ``chunks__<fp8>``; the suffix is the fingerprint hash (§6.5)."""

FINGERPRINT_HASH_LENGTH: Final = 8

ID_COLUMN: Final = "id"
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
DISTANCE_COLUMN: Final = "_distance"

DISTANCE_METRIC: Final = "cosine"
"""The metric every query and every index is built with.

One constant rather than two string literals, because the query's metric and the index's
metric agreeing is not a detail: an IVF-PQ index trained under L2 and queried under cosine
partitions the space one way and probes it another, and the result is a ranked list that is
wrong without being empty. ``docs/storage.md`` §6.2 is why it is cosine at all — vectors are
L2-normalized on the way in, so ``1 - distance`` is a real cosine similarity.
"""

IDENTITY_QUERY_PAGE: Final = 512
"""Chunk ids per ``IN`` predicate when reading rows back. See :meth:`LanceVectorStore._rows_for`."""

INTEGRITY_SCAN_PAGE: Final = 512
"""Rows per page for the checksum scan and the checksum backfill.

The bound that keeps both operations constant-memory over a corpus of any size. A page holds
``page_size`` vectors — at 1024 float32 components that is about two megabytes — and nothing
accumulates across pages except integers.
"""

_VECTOR_STATE_PRIORITY: Final = {
    VectorState.ABSENT: 0,
    VectorState.STALE: 1,
    VectorState.CORRUPT: 2,
    VectorState.READABLE: 3,
}
"""Best evidence across several physical publications of one logical chunk."""

FILTERABLE_COLUMNS: Final = frozenset({"document_id", "kind", "lang", "position"})
"""The promoted columns a predicate may name.

An allowlist rather than a convention: every column name that reaches a predicate is checked
against this set, so a future edit that threads a name in from somewhere less trustworthy
fails loudly instead of composing a query out of it.
"""

PUSHED_DOWN_FILTER_FIELDS: Final = frozenset({"document_ids", "kinds", "langs"})
""":class:`~manicule.core.retrieval.Filter` fields this store can answer by itself.

One entry per promoted column. Every other field needs a join the vector table has no columns
for; those are resolved in the document store first and arrive here as ``document_ids``
(``docs/retrieval.md`` §3.3).
"""

EXEMPT_FILTER_FIELDS: Final = frozenset({"workspace_ids"})
""":class:`~manicule.core.retrieval.Filter` fields this store neither honors nor refuses.

**A named exemption rather than an omission, because the two look identical in a loop and
only one of them is deliberate.** ``workspace_ids`` is a security boundary (``PLAN.md`` §14),
and the vector table has no column for it — not by oversight but by design: tenancy and
liveness live on ``documents`` in the authoritative store, and copying them into a derived one
creates a value that can disagree (``docs/storage.md`` §6.2).

The boundary therefore moved rather than disappeared. It is enforced by the hydrating join
inside the dense stage (``docs/retrieval.md`` §4.2), which is also what stops soft-deleted and
cross-workspace rows consuming top-``k`` slots, and
:func:`manicule.testing.assert_pipeline_enforces_scope` is what holds a pipeline to it. That
check is the reason this exemption is acceptable at all: without it, "the boundary is enforced
somewhere else" is a claim nothing verifies.
"""


_FOREIGN_INDEX_DETAIL: Final = (
    "an index this installation did not build carries the vector column; its partition count "
    "and build generation are unknown, and the maintenance boundary will not replace it"
)


class VectorStoreStateError(ManiculeError):
    """The store was used outside the state the operation needs.

    Either before :meth:`LanceVectorStore.ensure_ready` established which fingerprint the
    vectors belong to, or against a ``_manicule_meta`` table that no longer says one thing.
    Both are wiring or tampering, not user error, and both are fatal to the operation:
    guessing the fingerprint is exactly the mistake the meta table exists to prevent.
    """


class VectorStoreReprepareRequiredError(VectorStoreStateError):
    """The live publication moved to a vector space this handle has not prepared."""


_EXCLUSIVE_PINS: ContextVar[frozenset[Path]] = ContextVar(
    "manicule_exclusive_vector_pins", default=frozenset()
)


@asynccontextmanager
async def generation_pin(directory: Path, *, exclusive: bool = False) -> AsyncGenerator[None]:
    """Cross-process pin preventing cleanup from deleting a generation during an operation."""
    resolved = await asyncio.to_thread(directory.resolve)
    if resolved in _EXCLUSIVE_PINS.get():
        yield
        return
    pins = directory.parent / ".pins"
    pins.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(pins / f"{directory.name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        while True:
            try:
                fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        token = None
        if exclusive:
            token = _EXCLUSIVE_PINS.set(_EXCLUSIVE_PINS.get() | {resolved})
        try:
            yield
        finally:
            if token is not None:
                _EXCLUSIVE_PINS.reset(token)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def fingerprint_hash(fingerprint: EmbedFingerprint) -> str:
    """The short hash that names a vector table (``docs/storage.md`` §6.5).

    Taken over the canonical identity bytes, so two fingerprints share a table name only if
    they share a vector space.
    """
    digest = hashlib.sha256(fingerprint.canonical().encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_HASH_LENGTH]


def table_name(fingerprint: EmbedFingerprint) -> str:
    """The vector table these vectors belong in."""
    return f"{TABLE_PREFIX}{fingerprint_hash(fingerprint)}"


def quote(value: str) -> str:
    """Render ``value`` as a SQL string literal, doubling any quote inside it.

    Doubling is the whole escape: Lance's SQL dialect does not read a backslash as an escape
    character, so ``''`` is what closes the hole. A chunk id or document id is arbitrary text
    from a connector — a Confluence title, a file path, a git ref — and interpolating one
    unescaped is a predicate the caller gets to write.
    """
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def membership(column: str, values: Iterable[str]) -> str:
    """An ``IN`` term over ``column``, with every literal quoted.

    Raises:
        ValueError: If ``column`` is not a promoted column. The caller passes a constant
            today; this is what keeps that true.
    """
    if column not in FILTERABLE_COLUMNS:
        msg = (
            f"{column!r} is not a filterable column of the vector table. "
            f"Promoted columns are {', '.join(sorted(FILTERABLE_COLUMNS))}."
        )
        raise ValueError(msg)
    listed = ", ".join(quote(value) for value in sorted(values))
    return f"{column} IN ({listed})"


def predicate_for(filter: Filter | None) -> str | None:  # noqa: A002 - the domain's word
    """The Lance predicate for ``filter``, or ``None`` when nothing pushes down.

    ``None`` covers two cases that are the same instruction to this store: no filter at all,
    and a filter restricting only fields resolved elsewhere — which
    :data:`EXEMPT_FILTER_FIELDS` names, one field, with the reason attached.

    Raises:
        ValueError: When ``filter`` sets a field this store can neither honor nor has been
            granted an exemption for. Refusing is the point: quietly dropping a restriction
            returns results the filter was written to exclude, and the search still looks
            like it worked.
    """
    if filter is None:
        return None

    unhonored = sorted(filter.restricting_fields - PUSHED_DOWN_FILTER_FIELDS - EXEMPT_FILTER_FIELDS)
    if unhonored:
        msg = (
            f"the vector table has no column for {', '.join(unhonored)}, so this store "
            f"cannot honor {'them' if len(unhonored) > 1 else 'it'}. Resolve those fields "
            f"in the document store and pass the result as document_ids; ignoring them here "
            f"would return results the filter was written to exclude."
        )
        raise ValueError(msg)

    terms: list[str] = []
    if filter.document_ids:
        terms.append(membership("document_id", filter.document_ids))
    if filter.kinds:
        terms.append(membership("kind", [kind.value for kind in filter.kinds]))
    if filter.langs:
        terms.append(membership("lang", filter.langs))
    return " AND ".join(terms) if terms else None


def _embed_text_of(record: dict[str, Any]) -> str | None:
    """The ``embed_text`` of the chunk a row carries, or ``None`` if the row cannot say.

    Reads the one field rather than validating the whole :class:`~manicule.core.content.Chunk`,
    because this is a hot read on every document a sweep touches and the rest of the model is
    not being asked about. ``None`` covers every way the column can fail to answer — not JSON,
    not an object, no ``embed_text``, an ``embed_text`` that is not a string — since they all
    mean the same thing to the caller: this row cannot be checked against itself.
    """
    try:
        decoded = json.loads(str(record[CHUNK_COLUMN]))
    except (ValueError, TypeError, KeyError):
        return None
    if not isinstance(decoded, dict):
        return None
    value = decoded.get("embed_text")
    return value if isinstance(value, str) else None


def _unrecorded_checksum_predicate() -> str:
    """Rows that record *neither* half of the numerical-integrity pair.

    Pair-aware rather than checksum-only, and the difference is a row that carries one column
    and not the other. Such a row is not an upgrade backlog item — it is a row whose two halves
    were not written together, which
    :func:`~manicule.core.embedding.verify_stored_checksum` calls ``malformed``. Counting it as
    unrecorded would inflate the backfill's number with damage, and — the part that actually
    bites — it would put the row inside the backfill's page, where a freshly computed digest
    over whatever the vector is *now* would erase the contradiction the row was announcing.
    Half a record is evidence; overwriting it is the one thing the backfill must not do.

    ``NULL`` is spelled out beside ``''`` because a column added by a migration and a column
    written by a row are not guaranteed to agree on which absence they use, and
    :func:`_checksum_of` already treats the two the same on the read side.
    """
    empty = quote("")
    return (
        f"(({CHECKSUM_COLUMN} = {empty} OR {CHECKSUM_COLUMN} IS NULL) "
        f"AND ({CHECKSUM_VERSION_COLUMN} = {empty} OR {CHECKSUM_VERSION_COLUMN} IS NULL))"
    )


def _half_written_checksum_predicate() -> str:
    """Rows carrying exactly one half of the pair.

    Malformed by inspection of the columns alone — no vector is read and nothing is hashed —
    which is what lets the counting mode of :meth:`LanceVectorStore.checksum_coverage` report
    them. Without it a half-written row is invisible to every surface that cannot afford a
    scan, and "complete" would be true over a table holding one.
    """
    empty = quote("")
    recorded = f"({CHECKSUM_COLUMN} <> {empty} AND {CHECKSUM_COLUMN} IS NOT NULL)"
    versioned = f"({CHECKSUM_VERSION_COLUMN} <> {empty} AND {CHECKSUM_VERSION_COLUMN} IS NOT NULL)"
    return f"(({recorded} AND NOT {versioned}) OR (NOT {recorded} AND {versioned}))"


def _checksum_of(record: dict[str, Any]) -> tuple[str, str]:
    """The numerical-integrity pair a row carries, as two strings.

    ``get`` rather than indexing, because a table that predates the columns does not have them
    and a row read before :meth:`LanceVectorStore._ensure_generation_columns` has run therefore
    answers nothing at all. Absent, ``NULL`` and empty all mean the same thing here — no
    checksum was recorded — and :func:`~manicule.core.embedding.verify_stored_checksum` decides
    whether that is a refusal.
    """
    return (
        str(record.get(CHECKSUM_COLUMN) or UNRECORDED_CHECKSUM),
        str(record.get(CHECKSUM_VERSION_COLUMN) or UNRECORDED_CHECKSUM),
    )


def _row_integrity(record: dict[str, Any]) -> VectorIntegrity:
    """The numerical verdict on one row read straight from the table.

    The read-path half of :meth:`LanceVectorStore._verdict`, for the queries that have a vector
    and its checksum but no chunk to classify against — search results and the coverage scan.
    Provenance is not its business and it does not pretend otherwise: a row can be
    :attr:`~manicule.core.embedding.VectorIntegrity.VERIFIED` here and still be stale, which is
    what makes the two checks two checks.

    A record with no checksum column at all came from a table that predates them, and reads as
    unverified. That is a different absence from a row whose *vector* is null, which is a row
    nothing can be established about, and conflating the two would drop every result a
    pre-upgrade directory returns.
    """
    if CHECKSUM_COLUMN not in record:
        return VectorIntegrity.UNVERIFIED
    stored = record.get(VECTOR_COLUMN)
    if stored is None:
        return VectorIntegrity.UNREADABLE
    values = [float(value) for value in stored]
    if not is_finite_vector(values):
        return VectorIntegrity.NON_FINITE
    checksum, version = _checksum_of(record)
    return verify_stored_checksum(values, recorded=checksum, version=version, required=False)


def unit(vector: Vector) -> list[float]:
    """``vector`` scaled to length one, so that cosine distance is ``1 - similarity``.

    A vector of all zeros has no direction and is returned unchanged. Nothing here can invent
    one for it, and cosine similarity against it is undefined rather than small — see
    :meth:`LanceVectorStore.search` for what the store does about that.

    **A vector already of unit length within the column's precision is also returned
    unchanged**, and that is what makes a reused vector a reused vector rather than a
    very slightly different one. Read a stored vector back and the ``float32`` rounding leaves
    its length a few parts in 10^8 from one; dividing by that length and rounding to
    ``float32`` again lands on a different value in roughly one row in five hundred, measured.
    So without this, re-writing a row with the vector it already holds would perturb the odd
    row by one ulp, and "the vector was not recomputed" would be a claim no test could make
    exactly. The correction being skipped is smaller than :data:`FLOAT32_EPSILON`, which is
    smaller than the column can represent: it moves bits and cannot move meaning.
    """
    values = [float(value) for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0 or abs(norm - 1.0) < FLOAT32_EPSILON:
        return values
    return [value / norm for value in values]


class _MetaRow(LanceModel):
    """The one row of ``_manicule_meta``.

    Two columns because they answer different questions. ``embed_fingerprint`` is the whole
    model, which is what :meth:`LanceVectorStore.fingerprint` has to return — the canonical
    form holds identity fields only and cannot rebuild the type. ``canonical`` is the
    identity serialization the table name hashes, comparable byte for byte by anything that
    would rather not parse it.
    """

    embed_fingerprint: str
    canonical: str


def _row_model(dimension: int) -> type[LanceModel]:
    """The Lance row schema for vectors of ``dimension``, built when the dimension is known.

    ``vector`` becomes ``fixed_size_list<float32, dimension>``. The promoted columns are
    duplicated out of the chunk so a predicate can be pushed down without decoding
    ``chunk_json``; they are derived at write time from the same object, so they cannot drift.

    ``vector_checksum`` and ``vector_checksum_version`` are the numerical-integrity pair, and
    they are two columns rather than one string because the version decides how the checksum is
    read: a build that meets a format it does not implement has to refuse the row rather than
    parse a prefix out of it. ``float32`` above is why the checksum is defined over ``binary32``
    — the preimage is the persisted representation, not the ``float64`` a caller handed in.
    """
    fields: dict[str, Any] = {
        ID_COLUMN: (str, ...),
        CHUNK_ID_COLUMN: (str, ...),
        PUBLICATION_COLUMN: (str, ...),
        VECTOR_COLUMN: (FixedSizeVector(dimension), ...),
        "document_id": (str, ...),
        "kind": (str, ...),
        "lang": (str | None, None),
        "position": (int, ...),
        CHUNK_COLUMN: (str, ...),
        IDENTITY_COLUMN: (str, ...),
        CHECKSUM_COLUMN: (str, ...),
        CHECKSUM_VERSION_COLUMN: (str, ...),
        SOURCE_VECTOR_ID_COLUMN: (str | None, None),
        SOURCE_PUBLICATION_COLUMN: (str | None, None),
        SOURCE_SEQUENCE_COLUMN: (int | None, None),
        SOURCE_CREATED_AT_COLUMN: (str | None, None),
    }
    return create_model("ManiculeVectorRow", __base__=LanceModel, **fields)


class LanceVectorStore:
    """:class:`~manicule.core.protocols.VectorStore` on a local LanceDB directory.

    One instance owns one ``vectors/`` directory. Construction opens nothing; the connection
    and the table appear on first use, so building the component costs no I/O and a store
    pointed at a directory that does not exist yet is a valid thing to hold.

    What each method does before :meth:`ensure_ready` has run is deliberate rather than
    incidental. :meth:`fingerprint`, :meth:`count` and :meth:`delete_document` read the
    directory and answer honestly — a fresh instance pointed at a populated directory reports
    what is there, because answering ``0`` would be a lie the caller cannot detect.
    :meth:`upsert` and :meth:`search` refuse: both need to know which vector space they are
    in, and ``docs/storage.md`` §6.3 makes the refusal total rather than ingest-only, because
    querying an old index with a new model returns plausible, ranked, meaningless results.
    """

    def __init__(self, directory: Path) -> None:
        """Point the store at the directory that holds — or will hold — its vectors."""
        self._directory = directory
        self._lock = asyncio.Lock()
        self._connection: AsyncConnection | None = None
        self._fingerprint: EmbedFingerprint | None = None
        self._table: AsyncTable | None = None
        self._columns: frozenset[str] | None = None
        self._middleware: tuple[str, ...] = ()

    # --- lifecycle -----------------------------------------------------------------------

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        """Prepare the store for vectors from ``fingerprint``.

        First call writes ``_manicule_meta`` and creates ``chunks__<fp8>`` at the dimension
        the embedder reports. Later calls compare what is stored.

        A table created before :data:`IDENTITY_COLUMN` existed gains it here, filled with
        :data:`UNRECORDED_IDENTITY`. That is the whole of the migration: it is one Lance
        ``add_columns``, it needs no re-embedding, and it costs no forward pass — see
        :meth:`stored_vectors` for what an unrecorded identity is allowed to be derived from.

        Raises:
            FingerprintMismatchError: When the directory already holds vectors from a
                different model, including a different model of the same size.
        """
        async with self._lock:
            connection = await self._connect()
            stored = await self._stored_fingerprint(connection)
            if stored is None:
                await connection.create_table(
                    META_TABLE,
                    data=[
                        {
                            "embed_fingerprint": fingerprint.model_dump_json(),
                            "canonical": fingerprint.canonical(),
                        }
                    ],
                    schema=_MetaRow,
                )
            else:
                stored.require_match(fingerprint)
            self._table = await self._ensure_table(connection, fingerprint)
            self._columns = None
            self._fingerprint = fingerprint
            self._middleware = tuple(embed_text_middleware)

    async def teardown(self) -> None:
        """Close the LanceDB connection. Safe to call when none was ever opened."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._table = None
        self._columns = None

    async def open_existing(self, expected: EmbedFingerprint | None = None) -> EmbedFingerprint:
        """Open an already published directory; never create metadata or an empty table."""
        async with self._lock:
            if not self._directory.exists():
                raise VectorStoreStateError(
                    f"published vector generation {self._directory} does not exist"
                )
            connection = await self._connect()
            stored = await self._stored_fingerprint(connection)
            if stored is None:
                raise VectorStoreStateError(
                    f"published vector generation {self._directory} has no fingerprint metadata"
                )
            if expected is not None:
                stored.require_match(expected)
            name = table_name(stored)
            if name not in await self._table_names(connection):
                raise VectorStoreStateError(
                    f"published vector generation {self._directory} is missing {name}"
                )
            self._table = await connection.open_table(name)
            self._columns = None
            self._fingerprint = stored
            return stored

    async def fingerprint(self) -> EmbedFingerprint | None:
        """The fingerprint these vectors were built with, or ``None`` if there are none."""
        if self._fingerprint is not None:
            return self._fingerprint
        async with self._lock:
            return await self._stored_fingerprint(await self._connect())

    # --- writing -------------------------------------------------------------------------

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = LEGACY_PUBLICATION,
    ) -> None:
        """Store ``vectors`` against ``chunks`` within one publication generation.

        Raises:
            ValueError: If the two sequences differ in length, or if a vector is not the
                dimension the index was built for.
            VectorStoreStateError: If :meth:`ensure_ready` has not run.
        """
        table, fingerprint = self._ready()
        if len(chunks) != len(vectors):
            msg = (
                f"{len(chunks)} chunk(s) were offered with {len(vectors)} vector(s). They are "
                f"positional, so a mismatch means some chunk would be stored against another "
                f"chunk's vector."
            )
            raise ValueError(msg)
        if not chunks:
            return
        rows = [
            self._row(chunk, vector, fingerprint, publication_id)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        merge = table.merge_insert(ID_COLUMN)
        if publication_id == LEGACY_PUBLICATION:
            # Compatibility callers reuse logical ids and therefore still require replacement.
            merge = merge.when_matched_update_all()
        # Content-addressed publication ids include the vectors themselves. Their physical rows
        # are immutable: a stale acquisition generation may finish an external write after
        # takeover, but it can neither overwrite the successor's identical generation nor make
        # the row servable without the transaction-fenced relational pointer flip.
        await merge.when_not_matched_insert_all().execute(rows)

    async def upsert_snapshot(
        self,
        chunks: Sequence[SnapshotChunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str,
    ) -> None:
        """Write a shadow page with its complete source-row identity."""
        table, fingerprint = self._ready()
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} snapshot chunk(s) were offered with {len(vectors)} vector(s)"
            )
        if not chunks:
            return
        rows: list[dict[str, object]] = []
        for stored, vector in zip(chunks, vectors, strict=True):
            row = self._row(stored.chunk, vector, fingerprint, publication_id)
            row[ID_COLUMN] = stored.vector_id
            row[SOURCE_VECTOR_ID_COLUMN] = stored.vector_id
            row[SOURCE_PUBLICATION_COLUMN] = stored.publication_id
            row[PUBLICATION_COLUMN] = stored.publication_id
            row[SOURCE_SEQUENCE_COLUMN] = stored.sequence
            row[SOURCE_CREATED_AT_COLUMN] = (
                None if stored.created_at is None else stored.created_at.isoformat()
            )
            rows.append(row)
        await (
            table.merge_insert(ID_COLUMN)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    async def inspection_rows(self) -> list[dict[str, Any]]:
        """Physical rows needed to validate a named shadow generation."""
        table, _ = self._ready()
        return (
            await table.query()
            .select(
                [
                    ID_COLUMN,
                    CHUNK_ID_COLUMN,
                    PUBLICATION_COLUMN,
                    VECTOR_COLUMN,
                    CHUNK_COLUMN,
                    "document_id",
                    "kind",
                    "lang",
                    "position",
                    IDENTITY_COLUMN,
                    CHECKSUM_COLUMN,
                    CHECKSUM_VERSION_COLUMN,
                    SOURCE_VECTOR_ID_COLUMN,
                    SOURCE_PUBLICATION_COLUMN,
                    SOURCE_SEQUENCE_COLUMN,
                    SOURCE_CREATED_AT_COLUMN,
                ]
            )
            .to_list()
        )

    async def inspection_pages(
        self, *, page_size: int = 256
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Stream physical validation rows in bounded, stable ordered batches.

        This is one ordered Lance query rather than a keyset query per page.  Shadow
        inspection is deliberately a full read, so resubmitting an increasingly selective sort
        for every batch adds latency without adding an isolation property; the before/after
        storage revision fence in the caller detects a concurrent shadow mutation.
        """
        if page_size < 1:
            raise ValueError("inspection page size must be positive")
        table, _ = self._ready()
        columns = [
            ID_COLUMN,
            CHUNK_ID_COLUMN,
            PUBLICATION_COLUMN,
            VECTOR_COLUMN,
            CHUNK_COLUMN,
            "document_id",
            "kind",
            "lang",
            "position",
            IDENTITY_COLUMN,
            CHECKSUM_COLUMN,
            CHECKSUM_VERSION_COLUMN,
            SOURCE_VECTOR_ID_COLUMN,
            SOURCE_PUBLICATION_COLUMN,
            SOURCE_SEQUENCE_COLUMN,
            SOURCE_CREATED_AT_COLUMN,
        ]
        ordering = [
            ColumnOrdering(column_name="document_id"),
            ColumnOrdering(column_name="position"),
            ColumnOrdering(column_name=CHUNK_ID_COLUMN),
            ColumnOrdering(column_name=ID_COLUMN),
        ]
        reader = (
            await table.query()
            .select(columns)
            .order_by(ordering)
            .to_batches(max_batch_length=page_size)
        )
        async for batch in reader:
            page = batch.to_pylist()
            if page:
                yield page

    async def storage_revision(self) -> str:
        """The immutable Lance commit version currently visible through this handle."""
        table, _ = self._ready()
        return str(await table.version())

    async def delete_document(self, document_id: str) -> None:
        """Remove every vector belonging to a document. Idempotent."""
        table = await self._existing_table()
        if table is None:
            return
        await table.delete(f"document_id = {quote(document_id)}")

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Remove named vectors. Idempotent, and what the tombstone sweep calls.

        The historical parameter name is retained by the narrow sweep protocol; its values are
        physical vector ids, which equal logical chunk ids only for the legacy publication. By
        id, from a list the sweep was handed — never by anti-joining the whole table against
        ``chunks``. That comparison races concurrent ingest: an id written after the scan began
        looks like an orphan, and the sweep deletes a live vector.
        """
        if not chunk_ids:
            return
        table = await self._existing_table()
        if table is None:
            return
        listed = ", ".join(quote(chunk_id) for chunk_id in sorted(set(chunk_ids)))
        await table.delete(f"{ID_COLUMN} IN ({listed})")

    async def delete_chunks_counted(self, chunk_ids: Sequence[str]) -> int:
        """Delete exact physical ids and report rows that actually existed."""
        if not chunk_ids:
            return 0
        table = await self._existing_table()
        if table is None:
            return 0
        listed = ", ".join(quote(chunk_id) for chunk_id in sorted(set(chunk_ids)))
        rows = await table.query().where(f"{ID_COLUMN} IN ({listed})").select([ID_COLUMN]).to_list()
        await table.delete(f"{ID_COLUMN} IN ({listed})")
        return len({str(row[ID_COLUMN]) for row in rows})

    # --- reading -------------------------------------------------------------------------

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol and the domain
    ) -> list[Candidate]:
        """Return up to ``k`` nearest chunks, best first.

        Stored vectors are L2-normalized and the metric is cosine, so ``score`` is
        ``1 - distance``: a real cosine similarity in ``[-1, 1]``, not a monotone transform of
        a distance that happens to rank the same way. It is clamped to that interval, which
        float error can otherwise exceed by an ulp or two.

        **A row whose numbers do not match its checksum is dropped rather than ranked.** A
        candidate is a chunk plus a score, and the score is a distance to the stored vector — so
        returning one computed against corrupted numbers would put a result in a ranked list on
        the strength of bytes nothing vouches for. The cost is one SHA-256 over the ``k`` rows a
        query actually returns, not over the corpus, and rows written before the contract are
        unverified rather than dropped. A search can therefore return fewer than ``k``
        candidates over a damaged directory, which is the honest outcome: the alternative is
        backfilling the shortfall with rows that were ranked behind the ones being refused.

        **A query with no direction.** The zero vector is not near anything; cosine
        similarity against it is undefined for every row, and Lance drops the resulting
        undefined distances, so a plain vector search would return an empty list — which
        reads as "the corpus has nothing", a different and false claim. Instead the store
        returns the first ``k`` rows the filter admits, each scored ``0.0``: the one value
        that asserts neither similarity nor difference. A stored zero vector is subject to
        the same undefinedness from the other side and is never ranked into a result, which
        is why :meth:`count` and a search over the whole table can legitimately disagree.

        Raises:
            ValueError: If ``vector`` is not the dimension the index was built for, or if
                ``filter`` sets a field this store cannot honor.
            VectorStoreStateError: If :meth:`ensure_ready` has not run.
        """
        table, fingerprint = self._ready()
        query = unit(vector)
        if len(query) != fingerprint.dimension:
            msg = (
                f"a {len(query)}-dimension query was offered to an index built for "
                f"{fingerprint.dimension} by {fingerprint.describe()}."
            )
            raise ValueError(msg)
        predicate = predicate_for(filter)
        if k <= 0:
            return []
        if not any(query):
            return await self._unranked(table, k, predicate)

        search = table.vector_search(query).distance_type(DISTANCE_METRIC)
        if predicate is not None:
            search = search.where(predicate)
        columns = [ID_COLUMN, PUBLICATION_COLUMN, CHUNK_COLUMN, DISTANCE_COLUMN]
        columns.extend(await self._integrity_columns(table))
        records = await search.select(columns).limit(k).to_list()
        return [
            Candidate(
                chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
                publication_id=str(record[PUBLICATION_COLUMN]),
                score=min(1.0, max(-1.0, 1.0 - float(record[DISTANCE_COLUMN]))),
            )
            for record in records
            if _row_integrity(record).accepts
        ]

    async def count(self) -> int:
        """How many vectors are stored."""
        table = await self._existing_table()
        if table is None:
            return 0
        return await table.count_rows()

    # --- approximate search --------------------------------------------------------------

    async def ann_index_state(self, *, threshold: int) -> AnnIndexState:
        """Whether search here is exhaustive, indexed, or overdue for a rebuild.

        Reads only what already exists: the row count, and whatever LanceDB says about the
        indexes on the vector column. Nothing is recorded on the side, so this cannot disagree
        with the store it describes and a crash cannot leave it stale — there is no second copy
        of the answer to go out of date.

        A directory with no vector table at all reports zero rows rather than raising. Asking a
        fresh installation whether its index is current is a reasonable question with a
        reasonable answer, and :meth:`ensure_ready` has not necessarily run.
        """
        table = await self._existing_table()
        if table is None:
            return AnnIndexState(
                lifecycle=classify(threshold=threshold, rows=0, index=None),
                threshold=threshold,
                rows=0,
                detail="this directory holds no vectors yet",
            )
        rows = await table.count_rows()
        index = await self._ann_index(table)
        return AnnIndexState(
            lifecycle=classify(threshold=threshold, rows=rows, index=index),
            threshold=threshold,
            rows=rows,
            index=index,
            detail="" if index is None or index.recognized else _FOREIGN_INDEX_DETAIL,
        )

    async def build_ann_index(
        self, *, threshold: int, force: bool = False, dry_run: bool = False
    ) -> AnnIndexBuild:
        """Bring the ANN index up to what :meth:`ann_index_state` says is wanted.

        **The new index is created before the old one is dropped**, which is the whole of the
        promise that a failed build never costs the search path. A build that raises leaves the
        previous index in place and serving; a build that succeeds and then dies before the drop
        leaves two indexes, which costs disk and one wasted pass and is repaired by running this
        again. Between those two failures the cheap one is the one that survives, which is the
        ordering ``docs/storage.md`` §8.2 already argues for on the sweep.

        Nothing here stops a search. LanceDB scans fragments the index does not cover and merges
        them into the ranked result, so a corpus mid-build answers from the old index plus a
        flat scan of the tail — slower, never wrong, never empty.

        Args:
            threshold: The row count at which an index becomes wanted, and — applied to the
                rows an existing index does not cover — at which it becomes stale.
            force: Build even when nothing is due, at the current row count. For an operator
                who has changed the partition rule or wants the tail folded in early.
            dry_run: Report what a build would do and write nothing.

        Raises:
            VectorStoreStateError: If the vector column already carries an index this project
                did not create. Replacing it is somebody's deliberate act to undo, not this
                boundary's to guess at.
        """
        before = await self.ann_index_state(threshold=threshold)
        unchanged = AnnIndexBuild(before=before, after=before, dry_run=dry_run)
        if before.index is not None and not before.index.recognized:
            raise VectorStoreStateError(
                f"the vector column carries an index named {before.index.name!r}, which this "
                f"installation did not build. Drop it before asking for a managed index: two "
                f"indexes on one column is not a state this boundary will create."
            )
        if not force and not before.due:
            return replace(unchanged, detail=f"nothing is due: the index is {before.lifecycle}")
        if not before.buildable:
            return replace(
                unchanged,
                detail=(
                    f"{before.rows} vectors is below the {before.minimum_rows} an 8-bit "
                    f"product quantizer needs to train"
                ),
            )
        table = await self._existing_table()
        if table is None:  # defensive: `buildable` already required rows, which requires a table
            return replace(unchanged, detail="this directory holds no vectors yet")
        fingerprint = await self.fingerprint()
        if fingerprint is None:  # defensive: a table cannot exist without its meta row
            raise VectorStoreStateError(
                f"{self._directory} holds vectors with no recorded fingerprint; repair the "
                f"directory before building an index over them"
            )
        generation = 1 if before.index is None else (before.index.build_generation or 0) + 1
        partitions = partitions_for(before.rows)
        sub_vectors = sub_vectors_for(fingerprint.dimension)
        name = ann_index_name(build_generation=generation, num_partitions=partitions)
        if dry_run:
            return replace(
                unchanged,
                detail=(
                    f"would build {name}: IVF_PQ over {before.rows} vectors, {partitions} "
                    f"partitions, {sub_vectors} sub-vectors, {DISTANCE_METRIC} distance"
                ),
            )
        await table.create_index(
            VECTOR_COLUMN,
            config=IvfPq(
                distance_type=DISTANCE_METRIC,
                num_partitions=partitions,
                num_sub_vectors=sub_vectors,
                # Passed rather than defaulted: the row floor this method refuses below is
                # ``2 ** num_bits``, and a library default that moved would move one of those
                # two numbers and not the other.
                num_bits=PQ_CODE_BITS,
            ),
            name=name,
            replace=True,
        )
        await self._drop_superseded_indexes(table, keeping=name)
        after = await self.ann_index_state(threshold=threshold)
        return AnnIndexBuild(before=before, after=after, built=True, detail=f"built {name}")

    async def _ann_index(self, table: AsyncTable) -> AnnIndex | None:
        """The index on the vector column, preferring the newest one this project built.

        More than one can exist for exactly as long as it takes a repeat build to clear it —
        see the ordering :meth:`build_ann_index` explains — so this picks rather than refuses.

        **A foreign index wins the report whenever one is present, even beside a managed one.**
        Preferring ours would report ``recognized`` over a column that also carries an index
        this project cannot account for — and :meth:`build_ann_index` refuses beside one, so
        the status would have described a healthy index while every attempt to maintain it was
        declined for a reason nothing had reported. The unaccountable index is the fact that
        decides what the boundary may do, so it is the fact that gets reported.
        """
        listed = [
            config
            for config in await table.list_indices()
            if VECTOR_COLUMN in [str(column) for column in config.columns]
        ]
        if not listed:
            return None
        parsed = [(parse_ann_index_name(str(config.name)), config) for config in listed]
        foreign = [config for read, config in parsed if read is None]
        ours = [(read, config) for read, config in parsed if read is not None]
        read, config = (None, foreign[0]) if foreign else max(ours, key=lambda pair: pair[0][0])
        name = str(config.name)
        statistics = await table.index_stats(name)
        details = getattr(config, "index_details", None)
        compression = details.get("compression") if isinstance(details, dict) else None
        sub_vectors = compression.get("num_sub_vectors") if isinstance(compression, dict) else None
        return AnnIndex(
            name=name,
            index_type=str(statistics.index_type) if statistics else str(config.index_type),
            distance_type=(
                str(statistics.distance_type)
                if statistics and statistics.distance_type is not None
                else None
            ),
            indexed_rows=int(statistics.num_indexed_rows) if statistics else 0,
            unindexed_rows=int(statistics.num_unindexed_rows) if statistics else 0,
            num_sub_vectors=int(sub_vectors) if isinstance(sub_vectors, int) else None,
            build_generation=None if read is None else read[0],
            num_partitions=None if read is None else read[1],
        )

    async def _drop_superseded_indexes(self, table: AsyncTable, *, keeping: str) -> None:
        """Remove earlier builds of ours, and only ours.

        An index nobody here named is left where it is. :meth:`build_ann_index` refuses to run
        beside one at all, so reaching this with a foreign index present means it appeared
        during the build — and deleting something an operator made, during an operation that
        never said it would, is not a repair.
        """
        for config in await table.list_indices():
            name = str(config.name)
            if name == keeping or parse_ann_index_name(name) is None:
                continue
            if VECTOR_COLUMN in [str(column) for column in config.columns]:
                await table.drop_index(name)

    # --- numerical integrity -------------------------------------------------------------

    async def checksum_coverage(
        self, *, recompute: bool = False, page_size: int = INTEGRITY_SCAN_PAGE
    ) -> VectorChecksumCoverage:
        """How many stored vectors carry a checksum, and — on request — how many still match.

        Two modes, because two different questions get called "coverage" and only one of them
        is affordable on a status page.

        **Counting** is two ``count_rows`` predicates over an indexed-by-nothing string column
        and touches no vector. It answers "has the backfill finished", which is the question an
        upgrade is actually asking, and it is what ``status`` and ``doctor`` call.

        **Recomputing** reads every row's vector in bounded pages and hashes it. It answers
        "are the numbers still what they were", costs a scan of the corpus, and is what the
        checksum command performs when an operator asks for it. It never calls the embedder,
        never touches a source system, and never holds more than one page.

        Args:
            recompute: Verify each recorded checksum rather than only counting it.
            page_size: Rows per page while recomputing. The scan is bounded by this and by
                nothing else, so a corpus of any size costs one page of memory.

        Returns:
            Aggregate counts and a typed failure split. Never a checksum value, a vector
            component or a chunk identifier.
        """
        if page_size < 1:
            raise ValueError("integrity scan page size must be positive")
        table = await self._existing_table()
        if table is None:
            return VectorChecksumCoverage(scanned=False)
        available = frozenset(str(field.name) for field in await table.schema())
        rows = await table.count_rows()
        if not {CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN} <= available:
            # A table that predates the columns records no checksums, which is exactly what
            # `recorded = 0` says. Reporting it as unscanned would hide a corpus that is owed a
            # backfill behind the same value a store without the capability returns.
            return VectorChecksumCoverage(rows=rows, recomputed=recompute)
        recorded = rows - await table.count_rows(_unrecorded_checksum_predicate())
        if not recompute:
            # A half-written pair is malformed by looking at the two columns, which costs a
            # third predicate and no vector read. Reported here rather than left to the scan,
            # because a surface that cannot afford a scan is exactly the one that would
            # otherwise call a table holding such a row complete.
            half_written = await table.count_rows(_half_written_checksum_predicate())
            return VectorChecksumCoverage(
                rows=rows,
                recorded=recorded,
                failed=half_written,
                failures=({VectorIntegrity.MALFORMED.value: half_written} if half_written else {}),
            )
        verified = 0
        failures: dict[str, int] = {}
        async for page in self._integrity_pages(table, page_size=page_size):
            for record in page:
                integrity = _row_integrity(record)
                if integrity is VectorIntegrity.VERIFIED:
                    verified += 1
                elif integrity is not VectorIntegrity.UNVERIFIED:
                    failures[integrity.value] = failures.get(integrity.value, 0) + 1
        return VectorChecksumCoverage(
            rows=rows,
            recorded=recorded,
            verified=verified,
            failed=sum(failures.values()),
            failures=dict(sorted(failures.items())),
            recomputed=True,
        )

    async def backfill_checksums(
        self, *, limit: int = INTEGRITY_SCAN_PAGE, dry_run: bool = False
    ) -> VectorChecksumBackfill:
        """Give a bounded page of pre-checksum rows the checksum their stored vector implies.

        **It hashes what is on disk and nothing else.** No embedder, no parser, no connector,
        no source system, no retained snapshot. That is what makes it affordable, and it is
        also the limit of what it can claim: a row damaged *before* this ran gets a checksum
        over the damaged bytes, so the backfill establishes integrity from here forward rather
        than retroactively. ``docs/storage.md`` §6.2.5 states that in the operator's terms, and
        it is why a legacy table is reported as unverified rather than as verified-on-upgrade.

        **Resumable and idempotent by construction.** The page is selected by "records no
        checksum", so a row this pass finished is not a row the next pass can see. There is no
        cursor to persist and none to lose: an interruption costs at most the page in flight,
        and running it again after it finishes reads zero rows and writes nothing.

        **Rows whose vector cannot be hashed are left alone**, counted as
        :attr:`~manicule.core.embedding.VectorChecksumBackfill.unhashable`. Writing a checksum
        over a non-finite vector would certify a row that can never be ranked.

        **A row carrying one half of the pair is never selected at all.** It is malformed
        rather than unrecorded, and a pass that hashed it would replace a row announcing that
        its two halves disagree with one that verifies — the same laundering the unhashable
        branch above refuses, arriving through the column rather than the vector. See
        :func:`_unrecorded_checksum_predicate`.

        Args:
            limit: Rows to consider in this pass. The bound on both the query and the memory.
            dry_run: Report what a pass would do and write nothing.

        Returns:
            What the pass did, and how many rows still record no checksum.

        Raises:
            VectorStoreStateError: :meth:`ensure_ready` has not run, so the schema this writes
                into may not have the columns yet.
        """
        if limit < 1:
            raise ValueError("checksum backfill limit must be positive")
        table, _ = self._ready()
        # This command *is* the migration boundary for a published generation, which
        # `open_existing` deliberately is not: a read must never evolve the schema of a
        # generation somebody is searching. Idempotent, so a resumed pass costs nothing.
        await self._ensure_generation_columns(table)
        # The same predicate coverage counts, so the two can never disagree about which rows
        # are owed a checksum — and so a half-written pair stays outside this page.
        predicate = _unrecorded_checksum_predicate()
        outstanding = await table.count_rows(predicate)
        if outstanding == 0:
            return VectorChecksumBackfill(dry_run=dry_run)
        rows = await table.query().where(predicate).limit(limit).to_list()
        written: list[dict[str, Any]] = []
        unhashable = 0
        for record in rows:
            stored = record.get(VECTOR_COLUMN)
            values = None if stored is None else [float(value) for value in stored]
            if values is None or not is_finite_vector(values):
                unhashable += 1
                continue
            record[CHECKSUM_COLUMN] = vector_checksum(values)
            record[CHECKSUM_VERSION_COLUMN] = VECTOR_CHECKSUM_VERSION
            written.append(record)
        if written and not dry_run:
            # A whole-row merge on the physical id rather than a per-row `UPDATE`: one Lance
            # commit for the page, so a crash lands either before or after it and never inside.
            #
            # **`when_matched_update_all` and nothing else.** Without that restriction a row
            # deleted between the read above and the merge would be *re-inserted* by it — a
            # maintenance pass resurrecting a vector the tombstone sweep had just removed, which
            # is a worse failure than the one it exists to prevent. Unmatched rows are dropped,
            # so a concurrent delete wins and the next pass simply finds nothing there.
            await table.merge_insert(ID_COLUMN).when_matched_update_all().execute(written)
        return VectorChecksumBackfill(
            scanned=len(rows),
            written=len(written),
            unhashable=unhashable,
            remaining=outstanding if dry_run else await table.count_rows(predicate),
            dry_run=dry_run,
        )

    async def _integrity_pages(
        self, table: AsyncTable, *, page_size: int
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Every row's vector and checksum pair, in bounded pages ordered by physical id.

        Ordered by :data:`ID_COLUMN` because it is the merge key and therefore unique, so the
        keyset cannot collapse two rows onto one cursor and skip a suffix. Four columns are
        selected and no more: this is a numerical check and it has no business reading chunk
        text, a document id or a publication.
        """
        after: str | None = None
        while True:
            query = (
                table.query()
                .select([ID_COLUMN, VECTOR_COLUMN, CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN])
                .order_by([ColumnOrdering(column_name=ID_COLUMN)])
                .limit(page_size)
            )
            if after is not None:
                query = query.where(f"{ID_COLUMN} > {quote(after)}")
            page = await query.to_list()
            if not page:
                return
            yield page
            after = str(page[-1][ID_COLUMN])

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> Mapping[str, StoredVector]:
        """What this store holds for each of ``chunks``, and whether it can still be used.

        **What a verdict means is decided by
        :func:`~manicule.core.embedding.classify_stored_vector`**, which every backend shares
        and which is the only place the rule is written down. This method's job is to produce
        the three things a Lance row knows — its recorded identity, the ``embed_text`` of the
        chunk stored beside it, and the vector — and to look in the two places a row can be.

        **Two lookups, because a chunk id is not the only way a row can belong to a chunk.**
        The first is by id. The second is by embedding-input identity, for the chunks the first
        did not answer: a chunk id carries its position, so inserting one paragraph renames
        every chunk below it while moving no embedding input at all, and keyed on the id alone
        that edit re-embeds the whole document.
        :func:`~manicule.core.embedding.choose_stored_vector` decides which verdicts the second
        lookup is allowed to override.

        **A row written before the identity column is reconstructed, not distrusted.** Its
        embedding input is read from the chunk in :data:`CHUNK_COLUMN`, which
        :meth:`upsert` wrote from the same object, in the same call, as the vector beside it —
        so it is the exact prior embedding input rather than a guess at one. That is what makes
        the migration free: an existing corpus keeps every vector it has. The reconstruction is
        reported through
        :attr:`~manicule.core.embedding.StoredVector.identity_recorded` rather than hidden,
        and writing the row again records the identity for good.

        **Answers without requiring :meth:`ensure_ready`**, like :meth:`count` and
        :meth:`delete_document` and unlike :meth:`upsert` and :meth:`search`. The fingerprint
        it compares against is the one the directory records, which is the one the stored
        vectors were made with — the only fingerprint the question is about. A directory that
        holds nothing answers that it holds nothing for every chunk, which is true. The one
        thing an unprepared instance does not know is the middleware declaration, which it
        takes as empty; that can only make an identity fail to match, so the cost is a
        re-embed rather than a wrong vector.
        """
        verdicts = {chunk.id: StoredVector(state=VectorState.ABSENT) for chunk in chunks}
        if not chunks:
            return verdicts
        table = await self._existing_table()
        fingerprint = await self.fingerprint()
        if table is None or fingerprint is None:
            return verdicts
        available = frozenset(str(field.name) for field in await table.schema())
        has_identity = IDENTITY_COLUMN in available
        logical_id_column = CHUNK_ID_COLUMN if CHUNK_ID_COLUMN in available else ID_COLUMN

        by_id = {chunk.id: chunk for chunk in chunks}
        for record in await self._rows_for(
            table,
            sorted(by_id),
            logical_id_column=logical_id_column,
            available=available,
        ):
            chunk = by_id.get(str(record[logical_id_column]))
            if chunk is None:  # pragma: no cover - the predicate asked for these ids only
                continue
            found = self._verdict(chunk, record, fingerprint)
            current = verdicts[chunk.id]
            if _VECTOR_STATE_PRIORITY[found.state] > _VECTOR_STATE_PRIORITY[current.state]:
                verdicts[chunk.id] = found

        wanted = {
            chunk.id: self._identity_of(chunk, fingerprint)
            for chunk in chunks
            if verdicts[chunk.id].state in {VectorState.ABSENT, VectorState.STALE}
        }
        if not (wanted and has_identity):
            return verdicts
        found = await self._rows_by_identity(
            table, sorted(set(wanted.values())), available=available
        )
        for chunk_id, identity in wanted.items():
            record = found.get(identity)
            if record is None:
                continue
            verdicts[chunk_id] = choose_stored_vector(
                verdicts[chunk_id], self._verdict(by_id[chunk_id], record, fingerprint)
            )
        return verdicts

    async def publication_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        """Whether one unpublished generation holds an exact readable vector per chunk.

        Unlike :meth:`stored_vectors`, this never searches another publication or falls back
        by embedding-input identity. It is the validation fence immediately before an atomic
        relational generation flip, so an old readable vector is not evidence that the new
        publication is complete.
        """
        fingerprint = await self.fingerprint()
        table = await self._existing_table()
        if fingerprint is None or table is None:
            return not chunks and await self.publication_row_count(publication_id) == 0
        if fingerprint.canonical() != embedding_fingerprint:
            return False
        if await self.publication_row_count(publication_id) != len(chunks):
            return False
        for start in range(0, len(chunks), IDENTITY_QUERY_PAGE):
            page = chunks[start : start + IDENTITY_QUERY_PAGE]
            if not await self.publication_page_is_complete(
                publication_id, page, embedding_fingerprint=embedding_fingerprint
            ):
                return False
        return True

    async def publication_row_count(self, publication_id: str) -> int:
        """Count every physical row in a publication, including unexpected extras."""
        table = await self._existing_table()
        if table is None:
            return 0
        return await table.count_rows(f"{PUBLICATION_COLUMN} = {quote(publication_id)}")

    async def delete_publication(self, publication_id: str) -> int:
        """Delete one non-live physical publication and return its prior row count."""
        table = await self._existing_table()
        if table is None:
            return 0
        predicate = f"{PUBLICATION_COLUMN} = {quote(publication_id)}"
        rows = await table.count_rows(predicate)
        if rows:
            await table.delete(predicate)
        return rows

    async def publication_page_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        """Validate one bounded chunk page without accepting rows from another publication.

        **Checksums are required here and nowhere else on the read path.** This is the fence
        immediately before a generation becomes live, and every row in a generation this build
        staged was written by :meth:`_row`, which records one. So a row that carries none is
        either from a generation staged by an older build or a row this one did not write, and
        publishing either as verified would make the coverage number a lie at the exact moment
        it starts being relied on. A pre-checksum generation left half-built across an upgrade
        is therefore refused rather than adopted; ``docs/storage.md`` §6.2.5 says what to run.
        """
        if len(chunks) > IDENTITY_QUERY_PAGE:
            raise ValueError("vector validation page exceeds the fixed bound")
        fingerprint = await self.fingerprint()
        table = await self._existing_table()
        if fingerprint is None or table is None:
            return not chunks
        if fingerprint.canonical() != embedding_fingerprint:
            return False
        if not chunks:
            return True
        available = frozenset(str(field.name) for field in await table.schema())
        if not {CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN} <= available:
            # The table predates the contract, so no row in it can carry a checksum and no
            # publication out of it can be checksum-required. Refused rather than exempted.
            return False
        chunk_ids = ", ".join(quote(chunk.id) for chunk in chunks)
        records = await (
            table.query()
            .where(
                f"{PUBLICATION_COLUMN} = {quote(publication_id)} "
                f"AND {CHUNK_ID_COLUMN} IN ({chunk_ids})"
            )
            .select(
                [
                    CHUNK_ID_COLUMN,
                    IDENTITY_COLUMN,
                    CHUNK_COLUMN,
                    VECTOR_COLUMN,
                    CHECKSUM_COLUMN,
                    CHECKSUM_VERSION_COLUMN,
                ]
            )
            .limit(len(chunks) + 1)
            .to_list()
        )
        by_id = {str(record[CHUNK_ID_COLUMN]): record for record in records}
        return len(records) == len(chunks) == len(by_id) and all(
            chunk.id in by_id
            and self._verdict(chunk, by_id[chunk.id], fingerprint, require_checksum=True).state
            is VectorState.READABLE
            for chunk in chunks
        )

    async def copy_publication(
        self,
        source_publication_id: str,
        target_publication_id: str,
        chunks: Sequence[Chunk],
    ) -> None:
        """Copy one bounded checkpoint page for lease takeover, identity and integrity verified.

        Three separate refusals, in order: the page must be complete and current under the
        source publication, every row's checksum must still describe its vector, and the target
        rows are then written with checksums re-derived from the values actually stored.
        """
        if not chunks:
            return
        if len(chunks) > IDENTITY_QUERY_PAGE:
            raise ValueError("vector takeover page exceeds the fixed bound")
        fingerprint = await self.fingerprint()
        table = await self._existing_table()
        if fingerprint is None or table is None:
            raise VectorStoreStateError("checkpoint vectors are unavailable for takeover")
        if not await self.publication_page_is_complete(
            source_publication_id,
            chunks,
            embedding_fingerprint=fingerprint.canonical(),
        ):
            raise VectorStoreStateError("checkpoint vector page is incomplete or stale")
        listed = ", ".join(quote(chunk.id) for chunk in chunks)
        records = await (
            table.query()
            .where(
                f"{PUBLICATION_COLUMN} = {quote(source_publication_id)} "
                f"AND {CHUNK_ID_COLUMN} IN ({listed})"
            )
            .select([CHUNK_ID_COLUMN, VECTOR_COLUMN, CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN])
            .limit(len(chunks) + 1)
            .to_list()
        )
        vectors_by_id: dict[str, tuple[float, ...]] = {}
        for record in records:
            values = tuple(float(value) for value in record[VECTOR_COLUMN])
            checksum, version = _checksum_of(record)
            integrity = verify_stored_checksum(
                values, recorded=checksum, version=version, required=True
            )
            if integrity is not VectorIntegrity.VERIFIED:
                # The page passed `publication_page_is_complete` a moment ago, so reaching here
                # means the source changed underneath the copy. Refusing is the only answer that
                # does not propagate whatever it changed into a second publication.
                msg = (
                    f"a checkpoint vector failed numerical integrity ({integrity.value}) while "
                    f"being copied for takeover; the source publication is not fit to replay"
                )
                raise VectorStoreStateError(msg)
            vectors_by_id[str(record[CHUNK_ID_COLUMN])] = values
        # Copied through `upsert`, which re-derives the checksum from the canonical values it is
        # about to write rather than carrying the source row's string across. Carrying it would
        # let one digest end up attached to bytes it was never taken from — the one failure a
        # copy is in a position to introduce.
        await self.upsert(
            chunks,
            [vectors_by_id[chunk.id] for chunk in chunks],
            publication_id=target_publication_id,
        )

    async def _rows_for(
        self,
        table: AsyncTable,
        chunk_ids: Sequence[str],
        *,
        logical_id_column: str,
        available: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Every stored row among ``chunk_ids``, read in bounded pages.

        Paged because the predicate is an ``IN`` list and the caller's set is not bounded by
        anything this method controls: one query per document is small, one query per corpus
        is a SQL string megabytes long. The page size is a property of the query, not of the
        work, so it needs no tuning knob.

        A logical chunk id may match several publication rows, so this query has no row limit.
        Every match is classified and the strongest usable evidence wins.

        ``available`` is the table's actual column set, so this asks a table that predates a
        column for what it has rather than for what the current schema declares. A row from a
        table without the identity column reads as
        :data:`~manicule.core.embedding.UNRECORDED_IDENTITY`, and one from a table without the
        checksum columns as :data:`~manicule.core.embedding.UNRECORDED_CHECKSUM` — which is in
        both cases what it is.
        """
        columns = [logical_id_column, VECTOR_COLUMN, CHUNK_COLUMN]
        columns.extend(
            column
            for column in (IDENTITY_COLUMN, CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN)
            if column in available
        )

        rows: list[dict[str, Any]] = []
        for start in range(0, len(chunk_ids), IDENTITY_QUERY_PAGE):
            page = chunk_ids[start : start + IDENTITY_QUERY_PAGE]
            listed = ", ".join(quote(chunk_id) for chunk_id in page)
            rows.extend(
                await table.query()
                .where(f"{logical_id_column} IN ({listed})")
                .select(columns)
                .to_list()
            )
        return rows

    async def _rows_by_identity(
        self, table: AsyncTable, identities: Sequence[str], *, available: frozenset[str]
    ) -> dict[str, dict[str, Any]]:
        """One row per embedding-input identity, for the identities that have one.

        Which row, when several chunks share an embedding input, is not a choice worth making:
        a vector is a pure function of that input under a fixed fingerprint, so every row
        recorded against it holds the same vector and any of them answers the question.

        **No ``limit``, deliberately, and this is where one would be a bug rather than a
        safeguard.** An identity is not a key: two chunks with the same text under the same
        breadcrumb record the same one, so a page of *n* identities can match many more than
        *n* rows. A limit of *n* would then return every row of one popular identity and none
        of the others — losing reuse for the rest, silently, in an order decided by however the
        rows happen to be laid out. The ``IN`` predicate is the bound, and what it bounds is
        the duplicate density of the corpus.
        """
        columns = [VECTOR_COLUMN, CHUNK_COLUMN, IDENTITY_COLUMN]
        columns.extend(
            column for column in (CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN) if column in available
        )
        rows: dict[str, dict[str, Any]] = {}
        for start in range(0, len(identities), IDENTITY_QUERY_PAGE):
            page = identities[start : start + IDENTITY_QUERY_PAGE]
            listed = ", ".join(quote(identity) for identity in page)
            found = (
                await table.query()
                .where(f"{IDENTITY_COLUMN} IN ({listed})")
                .select(columns)
                .to_list()
            )
            for record in found:
                rows.setdefault(str(record[IDENTITY_COLUMN]), record)
        return rows

    def _verdict(
        self,
        chunk: Chunk,
        record: dict[str, Any],
        fingerprint: EmbedFingerprint,
        *,
        require_checksum: bool = False,
    ) -> StoredVector:
        """Classify one stored row against the chunk it is being offered for.

        The five things a Lance row knows are read here — its recorded identity, the chunk
        beside it, the vector, and the checksum pair — and what they *mean* is decided by
        :func:`~manicule.core.embedding.classify_stored_vector`, which every backend shares so
        that two of them cannot answer one question two ways.

        ``require_checksum`` is the caller's policy rather than the row's property: the same
        row is unverified-but-readable to reuse and a refusal to a publication fence. See
        :meth:`publication_page_is_complete`.
        """
        stored = record.get(VECTOR_COLUMN)
        checksum, version = _checksum_of(record)
        return classify_stored_vector(
            chunk,
            recorded_identity=str(record.get(IDENTITY_COLUMN) or UNRECORDED_IDENTITY),
            stored_embed_text=_embed_text_of(record),
            stored_vector=None if stored is None else [float(value) for value in stored],
            embed=fingerprint,
            middleware=self._middleware,
            recorded_checksum=checksum,
            recorded_checksum_version=version,
            require_checksum=require_checksum,
        )

    async def _unranked(self, table: AsyncTable, k: int, predicate: str | None) -> list[Candidate]:
        """Candidates for a query the store cannot rank. See :meth:`search`.

        Checksums are verified here too. These rows are not ranked against anything, but they
        are still returned as the corpus's answer, and a read path that let a mismatched row
        through when the query happened to have no direction would be a hole in the rule shaped
        exactly like a rarely exercised branch.
        """
        query = table.query()
        if predicate is not None:
            query = query.where(predicate)
        columns = [ID_COLUMN, PUBLICATION_COLUMN, CHUNK_COLUMN]
        columns.extend(await self._integrity_columns(table))
        records = await query.select(columns).limit(k).to_list()
        return [
            Candidate(
                chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
                publication_id=str(record[PUBLICATION_COLUMN]),
                score=0.0,
            )
            for record in records
            if _row_integrity(record).accepts
        ]

    def _row(
        self,
        chunk: Chunk,
        vector: Vector,
        fingerprint: EmbedFingerprint,
        publication_id: str,
    ) -> dict[str, object]:
        """One Lance row: the normalized vector, the promoted columns, the chunk, its identity.

        The checksum is taken from ``values`` — the output of
        :func:`~manicule.core.embedding.canonical_stored_vector`, which is the exact tuple this
        row stores — rather than from ``vector``. Hashing the argument would hash a
        representation that never reaches disk, and every readback would then disagree with it.
        """
        backend = fingerprint.backend or "an unspecified backend"
        try:
            values = canonical_stored_vector(vector)
        except ValueError as exc:
            msg = (
                f"chunk {chunk.id!r} was offered a vector with non-finite values for "
                f"{fingerprint.describe()} from {backend}. NaN and infinity cannot participate "
                "in cosine distance, so the vector was refused before storage."
            )
            raise ValueError(msg) from exc
        if len(values) != fingerprint.dimension:
            msg = (
                f"chunk {chunk.id!r} was offered a {len(values)}-dimension vector but the "
                f"index was built for {fingerprint.dimension}. The dimension comes from the "
                f"embedder's fingerprint, so a disagreement here means two embedders are in "
                f"play."
            )
            raise ValueError(msg)
        return {
            ID_COLUMN: vector_id(publication_id, chunk.id),
            CHUNK_ID_COLUMN: chunk.id,
            PUBLICATION_COLUMN: publication_id,
            VECTOR_COLUMN: values,
            "document_id": chunk.document_id,
            "kind": chunk.kind.value,
            "lang": chunk.lang,
            "position": chunk.position,
            CHUNK_COLUMN: chunk.model_dump_json(),
            IDENTITY_COLUMN: self._identity_of(chunk, fingerprint),
            CHECKSUM_COLUMN: vector_checksum(values),
            CHECKSUM_VERSION_COLUMN: VECTOR_CHECKSUM_VERSION,
        }

    def _identity_of(self, chunk: Chunk, fingerprint: EmbedFingerprint) -> str:
        """This store's one rule for what a stored vector's embedding input was.

        Takes the chunk rather than its ``embed_text``, because the identity is scoped by the
        document the chunk belongs to and a bare string cannot say which that is.
        """
        return embedding_input_identity(
            chunk.embed_text,
            document_id=chunk.document_id,
            embed=fingerprint,
            middleware=self._middleware,
        )

    def _ready(self) -> tuple[AsyncTable, EmbedFingerprint]:
        """The open table and its fingerprint, or a refusal naming what was skipped."""
        if self._table is None or self._fingerprint is None:
            msg = (
                "the vector store has not been prepared: call ensure_ready(fingerprint) "
                "first. Until it has run the store does not know which vector space it is "
                "holding, and vectors from two models are not comparable."
            )
            raise VectorStoreStateError(msg)
        return self._table, self._fingerprint

    async def _connect(self) -> AsyncConnection:
        if self._connection is None:
            self._connection = await lancedb.connect_async(self._directory)
        return self._connection

    async def _table_names(self, connection: AsyncConnection) -> frozenset[str]:
        listed = await connection.list_tables()
        return frozenset(str(name) for name in listed.tables)

    async def _stored_fingerprint(self, connection: AsyncConnection) -> EmbedFingerprint | None:
        """What the directory says it holds, or ``None`` if it has never held anything.

        Raises:
            VectorStoreStateError: If ``_manicule_meta`` does not hold exactly one row, or
                holds one whose canonical form disagrees with the fingerprint beside it. The
                table exists to make the directory self-describing; one that describes two
                things, or contradicts itself, is not something to pick a winner from.
        """
        if META_TABLE not in await self._table_names(connection):
            return None
        table = await connection.open_table(META_TABLE)
        rows = await table.query().limit(2).to_list()
        if len(rows) != 1:
            msg = (
                f"{META_TABLE} in {self._directory} holds {len(rows)} rows; it must hold "
                f"exactly one. This directory does not describe a single index — restore it "
                f"from a backup, or rebuild the vectors from the stored chunk text."
            )
            raise VectorStoreStateError(msg)
        stored = EmbedFingerprint.model_validate_json(str(rows[0]["embed_fingerprint"]))
        canonical = str(rows[0]["canonical"])
        if stored.canonical() != canonical:
            msg = (
                f"{META_TABLE} in {self._directory} contradicts itself: the recorded "
                f"identity is {canonical}, but the fingerprint stored beside it canonicalizes "
                f"to {stored.canonical()}. The row has been edited or half-written; restore "
                f"the directory rather than trusting either half."
            )
            raise VectorStoreStateError(msg)
        return stored

    async def _ensure_table(
        self, connection: AsyncConnection, fingerprint: EmbedFingerprint
    ) -> AsyncTable:
        name = table_name(fingerprint)
        if name in await self._table_names(connection):
            table = await connection.open_table(name)
            await self._ensure_generation_columns(table)
            return table
        return await connection.create_table(name, schema=_row_model(fingerprint.dimension))

    async def _ensure_generation_columns(self, table: AsyncTable) -> None:
        """Add the columns a table created before them does not have.

        The whole schema migration for an existing ``vectors/`` directory, and it is
        deliberately the cheapest one available: columns of empty strings, no row rewritten, no
        vector read and no forward pass. Every existing row is then
        :data:`UNRECORDED_IDENTITY`, which :meth:`stored_vectors` reconstructs from the chunk
        the row already carries, and :data:`UNRECORDED_CHECKSUM`, which reads as
        :attr:`~manicule.core.embedding.VectorIntegrity.UNVERIFIED` rather than as damage — so
        the upgrade costs an ``add_columns`` and nothing else, and an existing corpus keeps
        every vector it has. Idempotent, because :meth:`ensure_ready` runs on every process
        start.

        **The identity column and the checksum columns migrate on opposite terms**, and the
        difference is the whole compatibility policy. An unrecorded identity can be
        *reconstructed*, exactly, from the chunk stored beside the vector, so nothing is owed.
        An unrecorded checksum cannot be reconstructed from anything — hashing the stored
        vector now records what the bytes are today, not what they were when written — so the
        backfill in :meth:`backfill_checksums` establishes coverage going forward and says so,
        and until it has run those rows are reported as unverified rather than verified.
        """
        schema = await table.schema()
        names = {str(field.name) for field in schema}
        additions: dict[str, str] = {}
        if IDENTITY_COLUMN not in names:
            additions[IDENTITY_COLUMN] = quote(UNRECORDED_IDENTITY)
        if CHUNK_ID_COLUMN not in names:
            additions[CHUNK_ID_COLUMN] = ID_COLUMN
        if PUBLICATION_COLUMN not in names:
            additions[PUBLICATION_COLUMN] = quote(LEGACY_PUBLICATION)
        if CHECKSUM_COLUMN not in names:
            additions[CHECKSUM_COLUMN] = quote(UNRECORDED_CHECKSUM)
        if CHECKSUM_VERSION_COLUMN not in names:
            additions[CHECKSUM_VERSION_COLUMN] = quote(UNRECORDED_CHECKSUM)
        if additions:
            await table.add_columns(additions)
            self._columns = None

    async def _integrity_columns(self, table: AsyncTable) -> list[str]:
        """The columns a read needs to verify a row's numbers, for the tables that have them.

        Empty for a table written before the contract: it records no checksums, so selecting
        the columns would be a query error and verifying is not a thing that can be done. Those
        rows read as :attr:`~manicule.core.embedding.VectorIntegrity.UNVERIFIED` — see
        :meth:`backfill_checksums` for what clears that.
        """
        available = await self._available_columns(table)
        if not {CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN} <= available:
            return []
        return [VECTOR_COLUMN, CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN]

    async def _available_columns(self, table: AsyncTable) -> frozenset[str]:
        """Which columns this table actually has, read once per open handle.

        Every read path that touches a column added by a migration has to ask, because a
        directory written by an older build does not have it and selecting it is a query error
        rather than a null. Cached because ``search`` asks on every query and the answer can
        only change where this instance opens the table or adds a column — the four places that
        clear it.
        """
        if self._columns is None:
            self._columns = frozenset(str(field.name) for field in await table.schema())
        return self._columns

    async def _existing_table(self) -> AsyncTable | None:
        """The vector table if the directory has one, without requiring :meth:`ensure_ready`.

        Reads the fingerprint the directory records rather than one supplied by a caller, so
        this cannot be the path by which a mismatched model gets its table opened.

        Takes the lock, because its callers do not hold it and two concurrent ones would
        otherwise each open a connection, leaving one that :meth:`teardown` never closes.
        """
        if self._table is not None:
            return self._table
        async with self._lock:
            connection = await self._connect()
            stored = await self._stored_fingerprint(connection)
            if stored is None:
                return None
            name = table_name(stored)
            if name not in await self._table_names(connection):
                return None
            return await connection.open_table(name)


class PublishedLanceVectorStore:
    """A live vector handle that follows SQLite's publication pointer between operations.

    Legacy ``chunks__…`` pointers resolve to the root directory. Re-embedding generation
    pointers resolve to their named sibling directory. Each operation pins one generation for
    its duration, so cleanup cannot remove it in flight. A pointer flip to a different
    fingerprint requires :meth:`ensure_ready` with that fingerprint before any operation;
    same dimensions are not evidence that two embedding spaces are compatible.
    """

    def __init__(
        self,
        directory: Path,
        engine: AsyncEngine,
        *,
        workspace_id: str = "default",
        identity_namespace: str | None = None,
        expected_reset_epoch: int | None = None,
        operation_hook: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._directory = directory
        self._engine = engine
        self._workspace_id = workspace_id
        # A handle built against an existing identity must never silently become the first
        # writer of a new one.  Reset removes that identity only after fencing leases and
        # cleaning its physical store; this durable expectation is the post-flock CAS that
        # prevents a writer queued behind reset from recreating the stale store afterwards.
        self._identity_namespace = identity_namespace
        self._expected_reset_epoch = expected_reset_epoch
        self._stores: dict[str, LanceVectorStore] = {}
        self._middleware: tuple[str, ...] = ()
        self._configured: EmbedFingerprint | None = None
        self._publication_pointer: str | None = None
        self._operation_hook = operation_hook

    @property
    def publication_pointer(self) -> str | None:
        """The SQLite pointer resolved by the most recent operation."""
        return self._publication_pointer

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        self._middleware = tuple(embed_text_middleware)
        self._configured = fingerprint
        async with self._operation() as store:
            await store.ensure_ready(fingerprint, embed_text_middleware=embed_text_middleware)

    async def teardown(self) -> None:
        for store in self._stores.values():
            await store.teardown()
        self._stores.clear()

    async def fingerprint(self) -> EmbedFingerprint | None:
        async with self._operation() as store:
            return await store.fingerprint()

    async def physical_fingerprint(self) -> EmbedFingerprint | None:
        """Read existing Lance metadata without preparing or creating a physical store."""
        while True:
            binding = await self._binding()
            pointer, _namespace, _epoch = binding
            key = _published_generation_key(pointer)
            directory = (
                self._directory / "generations" / key if key != "legacy" else self._directory
            )
            async with generation_pin(self._directory):
                if await self._binding() != binding:
                    continue
                if not await asyncio.to_thread(directory.exists):
                    return None
                if directory == self._directory:
                    store = self._stores.setdefault(key, LanceVectorStore(directory))
                    return await store.fingerprint()
                async with generation_pin(directory):
                    if await self._binding() != binding:
                        continue
                    if not await asyncio.to_thread(directory.exists):
                        return None
                    store = self._stores.setdefault(key, LanceVectorStore(directory))
                    return await store.fingerprint()

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = LEGACY_PUBLICATION,
    ) -> None:
        async with self._operation() as store:
            await store.upsert(chunks, vectors, publication_id=publication_id)

    async def delete_document(self, document_id: str) -> None:
        async with self._operation() as store:
            await store.delete_document(document_id)

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        async with self._operation() as store:
            await store.delete_chunks(chunk_ids)

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - protocol spelling
    ) -> list[Candidate]:
        async with self._operation() as store:
            return await store.search(vector, k, filter)

    async def count(self) -> int:
        async with self._operation() as store:
            return await store.count()

    async def checksum_coverage(
        self, *, recompute: bool = False, page_size: int = INTEGRITY_SCAN_PAGE
    ) -> VectorChecksumCoverage:
        """The live generation's checksum coverage.

        The vector root is checked before an operation is opened, on the same terms as
        :meth:`ann_index_state`: opening one takes a generation pin and creates directories,
        and this is a read behind ``status``. No root means nothing was looked at, which is
        :attr:`~manicule.core.embedding.VectorChecksumCoverage.scanned` rather than a zero.
        """
        if not await asyncio.to_thread(self._directory.exists):
            return VectorChecksumCoverage(scanned=False)
        async with self._operation() as store:
            return await store.checksum_coverage(recompute=recompute, page_size=page_size)

    async def backfill_checksums(
        self, *, limit: int = INTEGRITY_SCAN_PAGE, dry_run: bool = False
    ) -> VectorChecksumBackfill:
        """Run one bounded backfill pass against the live generation.

        Scoped to the published generation this handle follows and to nothing else. A workspace
        has its own vector root, and a retired generation is a different directory, so a pass
        can neither reach another tenant's rows nor rewrite a generation that is no longer
        live — the two boundaries this is the first operation to rewrite rows across.
        """
        if not await asyncio.to_thread(self._directory.exists):
            return VectorChecksumBackfill(dry_run=dry_run)
        async with self._operation() as store:
            return await store.backfill_checksums(limit=limit, dry_run=dry_run)

    async def ann_index_state(self, *, threshold: int) -> AnnIndexState:
        """The live generation's index state, named with the generation it describes.

        An index belongs to the physical generation it was built in, so a re-embed that
        publishes a new one starts from no index and reports :attr:`AnnLifecycle.PENDING`
        honestly. Carrying the pointer on the state is what stops that reading as a regression:
        the coverage did not drop, the generation changed underneath it.

        **The vector root is checked before an operation is opened**, because opening one takes
        a generation pin and a LanceDB connection, and both create directories. This is the read
        behind ``index_status``; a read that has to build part of the store in order to report
        that the store is empty is not one an operator can run freely. No root means no
        generation under it either, so the absence is the whole answer.
        """
        if not await asyncio.to_thread(self._directory.exists):
            return AnnIndexState(
                lifecycle=classify(threshold=threshold, rows=0, index=None),
                threshold=threshold,
                rows=0,
                detail="this workspace holds no vectors yet",
            )
        async with self._operation() as store:
            state = await store.ann_index_state(threshold=threshold)
        return replace(state, generation=self._publication_pointer or LEGACY_PUBLICATION)

    async def build_ann_index(
        self, *, threshold: int, force: bool = False, dry_run: bool = False
    ) -> AnnIndexBuild:
        """Build into whichever generation is live, pinned for the length of the build.

        The pin is what makes a long build safe beside generation cleanup: the directory the
        index is being written into cannot be removed while it is being written.
        """
        async with self._operation() as store:
            build = await store.build_ann_index(threshold=threshold, force=force, dry_run=dry_run)
        pointer = self._publication_pointer or LEGACY_PUBLICATION
        return replace(
            build,
            before=replace(build.before, generation=pointer),
            after=replace(build.after, generation=pointer),
        )

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> Mapping[str, StoredVector]:
        async with self._operation() as store:
            return await store.stored_vectors(chunks)

    async def publication_row_count(self, publication_id: str) -> int:
        """Count a rebuild namespace in the currently pinned live generation."""
        async with self._operation() as store:
            return await store.publication_row_count(publication_id)

    async def delete_publication(self, publication_id: str) -> int:
        """Delete a terminal non-live publication through the pinned live store."""
        async with self._operation() as store:
            return await store.delete_publication(publication_id)

    async def delete_bound_publication(self, vector_table: str | None, publication_id: str) -> int:
        """Delete from the generation recorded at planning, independent of today's pointer.

        Cleanup may retry after #187 has swapped the live pointer. The immutable generation row
        is the authority for which physical directory received the namespace; following the
        current pointer here would delete a same-named namespace from the wrong generation.
        """
        key = _published_generation_key(vector_table)
        directory = self._directory / "generations" / key if key != "legacy" else self._directory
        store = self._stores.setdefault(key, LanceVectorStore(directory))
        async with self._existing_operation_pin(directory) as exists:
            if not exists:
                return 0
            if key != "legacy":
                await store.open_existing()
            return await store.delete_publication(publication_id)

    async def delete_bound_chunks(self, vector_table: str | None, vector_ids: Sequence[str]) -> int:
        """Delete exact staged/live rows from the physical generation recorded before reset."""
        if not vector_ids:
            return 0
        key = _published_generation_key(vector_table)
        directory = self._directory / "generations" / key if key != "legacy" else self._directory
        store = self._stores.setdefault(key, LanceVectorStore(directory))
        async with self._existing_operation_pin(directory) as exists:
            if not exists:
                return 0
            if key != "legacy":
                await store.open_existing()
            return await store.delete_chunks_counted(vector_ids)

    async def publication_page_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        """Validate one rebuild page without crossing the live pointer boundary."""
        async with self._operation() as store:
            return await store.publication_page_is_complete(
                publication_id,
                chunks,
                embedding_fingerprint=embedding_fingerprint,
            )

    async def publication_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        """Validate an exact rebuild namespace in one pinned live generation."""
        async with self._operation() as store:
            return await store.publication_is_complete(
                publication_id,
                chunks,
                embedding_fingerprint=embedding_fingerprint,
            )

    async def copy_publication(
        self,
        source_publication_id: str,
        target_publication_id: str,
        chunks: Sequence[Chunk],
    ) -> None:
        """Replay takeover vectors without accidentally crossing a #187 pointer swap."""
        async with self._operation() as store:
            await store.copy_publication(
                source_publication_id,
                target_publication_id,
                chunks,
            )

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[LanceVectorStore]:
        """Pin and revalidate one pointer before exposing its store to an operation."""
        while True:
            binding = await self._binding()
            pointer, namespace, epoch = binding
            key = _published_generation_key(pointer)
            directory = (
                self._directory / "generations" / key if key != "legacy" else self._directory
            )
            async with generation_pin(self._directory):
                current_binding = await self._binding()
                if current_binding != binding:
                    continue
                self._require_current_handle(namespace, epoch)
                async with self._selected_generation_pin(directory) as exists:
                    current_binding = await self._binding()
                    if current_binding != binding:
                        continue
                    if not exists:
                        raise VectorStoreStateError(
                            f"published vector generation {directory} does not exist"
                        )
                    self._require_current_handle(namespace, epoch)
                    self._publication_pointer = pointer
                    store = await self._prepared_store(key, directory)
                    if self._operation_hook is not None:
                        await self._operation_hook()
                    yield store
                    return

    @asynccontextmanager
    async def _selected_generation_pin(self, directory: Path) -> AsyncGenerator[bool]:
        """Pin a child after root validation and report whether it still exists.

        The caller re-reads the moving SQLite binding before interpreting ``False`` as
        corruption.  A publisher may legitimately flip the pointer and remove the old child
        while this operation waits for its pin; that case retries the new publication.
        """
        if directory == self._directory:
            yield True
            return
        if not await asyncio.to_thread(directory.exists):
            yield False
            return
        async with generation_pin(directory):
            yield await asyncio.to_thread(directory.exists)

    @asynccontextmanager
    async def _existing_operation_pin(self, directory: Path) -> AsyncGenerator[bool]:
        """Pin an existing bound directory without recreating it after reset.

        The existence checks deliberately happen after the root pin and, for a child, after
        its pin.  A cleanup queued behind reset must return zero rather than recreate the
        deleted ``generations/.pins`` tree while trying to lock a path that no longer exists.
        """
        async with generation_pin(self._directory):
            if not await asyncio.to_thread(directory.exists):
                yield False
                return
            if directory == self._directory:
                yield True
                return
            async with generation_pin(directory):
                yield await asyncio.to_thread(directory.exists)

    def _require_current_handle(self, namespace: str | None, epoch: int) -> None:
        if self._expected_reset_epoch is not None and epoch != self._expected_reset_epoch:
            raise VectorStoreReprepareRequiredError(
                "the workspace derived-reset epoch changed while this vector handle was "
                "waiting; rebuild the runtime handle before writing"
            )
        if self._identity_namespace is not None and (
            namespace is None or namespace != self._identity_namespace
        ):
            raise VectorStoreReprepareRequiredError(
                "the workspace index identity was reset while this vector handle was waiting; "
                "rebuild the runtime handle before writing"
            )

    async def _binding(self) -> tuple[str | None, str | None, int]:
        from sqlalchemy import select  # noqa: PLC0415 - storage remains lazy

        from manicule.storage import models  # noqa: PLC0415 - avoids import cycle at startup

        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        models.IndexState.vector_table,
                        models.IndexState.vector_namespace,
                        models.Workspace.derived_reset_epoch,
                    )
                    .select_from(models.Workspace)
                    .outerjoin(
                        models.IndexState,
                        models.IndexState.workspace_id == models.Workspace.id,
                    )
                    .where(models.Workspace.id == self._workspace_id)
                )
            ).one_or_none()
        if row is None:
            return None, None, 0
        return (
            None if row.vector_table is None else str(row.vector_table),
            None if row.vector_namespace is None else str(row.vector_namespace),
            int(row.derived_reset_epoch),
        )

    async def _pointer(self) -> str | None:
        pointer, _namespace, _epoch = await self._binding()
        return pointer

    async def _prepared_store(self, key: str, directory: Path) -> LanceVectorStore:
        store = self._stores.setdefault(key, LanceVectorStore(directory))
        if key != "legacy":
            stored = await store.open_existing()
            if self._configured is not None and stored.canonical() != self._configured.canonical():
                raise VectorStoreReprepareRequiredError(
                    "the live vector publication changed embedding fingerprints; prepare this "
                    "handle with the live embedder before searching or writing"
                )
        else:
            stored = await store.fingerprint()
            fingerprint = self._configured or stored
            if fingerprint is not None:
                await store.ensure_ready(fingerprint, embed_text_middleware=self._middleware)
        return store


def _published_generation_key(pointer: str | None) -> str:
    """Map a database pointer to one safe local directory component."""
    if pointer is None or not pointer.startswith("reembed-"):
        return "legacy"
    if re.fullmatch(r"reembed-[A-Za-z0-9._-]+", pointer) is None:
        raise VectorStoreStateError(
            "the published vector generation pointer is invalid; repair index state before "
            "searching or writing vectors"
        )
    return pointer


def workspace_vector_directory(root: Path, workspace_id: str) -> Path:
    """Opaque, stable physical namespace for one workspace's independent vector identity."""
    digest = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()
    return root / "workspaces" / digest


async def reset_vector_directory(directory: Path, *, legacy_root: bool) -> bool:
    """Remove one workspace's physical identity after all exact row cleanup has settled.

    The historical root may contain the new ``workspaces/`` children, so it is cleared through
    Lance's table API and its legacy generation tree rather than recursively deleting the root.
    """
    if not await asyncio.to_thread(directory.exists):
        return False
    if not legacy_root:
        await asyncio.to_thread(shutil.rmtree, directory)
        return True
    connection = await lancedb.connect_async(directory)
    try:
        listed = await connection.list_tables()
        for name in listed.tables:
            await connection.drop_table(str(name))
    finally:
        connection.close()
    generations = directory / "generations"
    if generations.exists():
        await asyncio.to_thread(shutil.rmtree, generations)
    workspace_root = directory / "workspaces"
    if not workspace_root.exists():
        await asyncio.to_thread(shutil.rmtree, directory)
    return True


__all__ = [
    "CHECKSUM_COLUMN",
    "CHECKSUM_VERSION_COLUMN",
    "EXEMPT_FILTER_FIELDS",
    "FILTERABLE_COLUMNS",
    "FLOAT32_EPSILON",
    "IDENTITY_COLUMN",
    "INTEGRITY_SCAN_PAGE",
    "META_TABLE",
    "PUSHED_DOWN_FILTER_FIELDS",
    "SOURCE_CREATED_AT_COLUMN",
    "SOURCE_PUBLICATION_COLUMN",
    "SOURCE_SEQUENCE_COLUMN",
    "SOURCE_VECTOR_ID_COLUMN",
    "TABLE_PREFIX",
    "LanceVectorStore",
    "PublishedLanceVectorStore",
    "VectorStoreReprepareRequiredError",
    "VectorStoreStateError",
    "fingerprint_hash",
    "generation_pin",
    "membership",
    "predicate_for",
    "quote",
    "reset_vector_directory",
    "table_name",
    "unit",
    "workspace_vector_directory",
]
