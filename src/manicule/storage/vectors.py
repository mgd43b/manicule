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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final

import lancedb
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector as FixedSizeVector
from lancedb.query import ColumnOrdering
from pydantic import create_model

from manicule.core.content import LEGACY_PUBLICATION, Chunk
from manicule.core.embedding import (
    FLOAT32_EPSILON,
    UNRECORDED_IDENTITY,
    EmbedFingerprint,
    StoredVector,
    VectorState,
    canonical_stored_vector,
    choose_stored_vector,
    classify_stored_vector,
    embedding_input_identity,
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
SOURCE_VECTOR_ID_COLUMN: Final = "source_vector_id"
SOURCE_PUBLICATION_COLUMN: Final = "source_publication_id"
SOURCE_SEQUENCE_COLUMN: Final = "source_sequence"
SOURCE_CREATED_AT_COLUMN: Final = "source_created_at"
DISTANCE_COLUMN: Final = "_distance"

IDENTITY_QUERY_PAGE: Final = 512
"""Chunk ids per ``IN`` predicate when reading rows back. See :meth:`LanceVectorStore._rows_for`."""

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


class VectorStoreStateError(ManiculeError):
    """The store was used outside the state the operation needs.

    Either before :meth:`LanceVectorStore.ensure_ready` established which fingerprint the
    vectors belong to, or against a ``_manicule_meta`` table that no longer says one thing.
    Both are wiring or tampering, not user error, and both are fatal to the operation:
    guessing the fingerprint is exactly the mistake the meta table exists to prevent.
    """


class VectorStoreReprepareRequiredError(VectorStoreStateError):
    """The live publication moved to a vector space this handle has not prepared."""


@asynccontextmanager
async def generation_pin(directory: Path, *, exclusive: bool = False) -> AsyncGenerator[None]:
    """Cross-process pin preventing cleanup from deleting a generation during an operation."""
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
        yield
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
            self._fingerprint = fingerprint
            self._middleware = tuple(embed_text_middleware)

    async def teardown(self) -> None:
        """Close the LanceDB connection. Safe to call when none was ever opened."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._table = None

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
        """Read physical validation rows in bounded, stable keyset pages."""
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
        after: tuple[str, int, str, str] | None = None
        while True:
            query = table.query().select(columns).order_by(ordering).limit(page_size)
            if after is not None:
                document_id, position, chunk_id, physical_id = after
                query = query.where(
                    f"document_id > {quote(document_id)} OR "
                    f"(document_id = {quote(document_id)} AND "
                    f"(position > {position} OR "
                    f"(position = {position} AND "
                    f"({CHUNK_ID_COLUMN} > {quote(chunk_id)} OR "
                    f"({CHUNK_ID_COLUMN} = {quote(chunk_id)} AND "
                    f"{ID_COLUMN} > {quote(physical_id)})))))"
                )
            page = await query.to_list()
            if not page:
                return
            yield page
            last = page[-1]
            after = (
                str(last["document_id"]),
                int(str(last["position"])),
                str(last[CHUNK_ID_COLUMN]),
                str(last[ID_COLUMN]),
            )

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

        search = table.vector_search(query).distance_type("cosine")
        if predicate is not None:
            search = search.where(predicate)
        records = (
            await search.select([ID_COLUMN, PUBLICATION_COLUMN, CHUNK_COLUMN, DISTANCE_COLUMN])
            .limit(k)
            .to_list()
        )
        return [
            Candidate(
                chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
                publication_id=str(record[PUBLICATION_COLUMN]),
                score=min(1.0, max(-1.0, 1.0 - float(record[DISTANCE_COLUMN]))),
            )
            for record in records
        ]

    async def count(self) -> int:
        """How many vectors are stored."""
        table = await self._existing_table()
        if table is None:
            return 0
        return await table.count_rows()

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
        schema = await table.schema()
        columns = {str(field.name) for field in schema}
        has_identity = IDENTITY_COLUMN in columns
        logical_id_column = CHUNK_ID_COLUMN if CHUNK_ID_COLUMN in columns else ID_COLUMN

        by_id = {chunk.id: chunk for chunk in chunks}
        for record in await self._rows_for(
            table,
            sorted(by_id),
            logical_id_column=logical_id_column,
            has_identity=has_identity,
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
        found = await self._rows_by_identity(table, sorted(set(wanted.values())))
        for chunk_id, identity in wanted.items():
            record = found.get(identity)
            if record is None:
                continue
            verdicts[chunk_id] = choose_stored_vector(
                verdicts[chunk_id], self._verdict(by_id[chunk_id], record, fingerprint)
            )
        return verdicts

    async def _rows_for(
        self,
        table: AsyncTable,
        chunk_ids: Sequence[str],
        *,
        logical_id_column: str,
        has_identity: bool,
    ) -> list[dict[str, Any]]:
        """Every stored row among ``chunk_ids``, read in bounded pages.

        Paged because the predicate is an ``IN`` list and the caller's set is not bounded by
        anything this method controls: one query per document is small, one query per corpus
        is a SQL string megabytes long. The page size is a property of the query, not of the
        work, so it needs no tuning knob.

        A logical chunk id may match several publication rows, so this query has no row limit.
        Every match is classified and the strongest usable evidence wins.

        A row from a table without the identity column reads as
        :data:`~manicule.core.embedding.UNRECORDED_IDENTITY`, which is what it is.
        """
        columns = [logical_id_column, VECTOR_COLUMN, CHUNK_COLUMN]
        if has_identity:
            columns.append(IDENTITY_COLUMN)

        rows: list[dict[str, Any]] = []
        for start in range(0, len(chunk_ids), IDENTITY_QUERY_PAGE):
            page = chunk_ids[start : start + IDENTITY_QUERY_PAGE]
            listed = ", ".join(quote(chunk_id) for chunk_id in page)
            found = (
                await table.query()
                .where(f"{logical_id_column} IN ({listed})")
                .select(columns)
                .to_list()
            )
            for record in found:
                record.setdefault(IDENTITY_COLUMN, UNRECORDED_IDENTITY)
            rows.extend(found)
        return rows

    async def _rows_by_identity(
        self, table: AsyncTable, identities: Sequence[str]
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
        rows: dict[str, dict[str, Any]] = {}
        for start in range(0, len(identities), IDENTITY_QUERY_PAGE):
            page = identities[start : start + IDENTITY_QUERY_PAGE]
            listed = ", ".join(quote(identity) for identity in page)
            found = (
                await table.query()
                .where(f"{IDENTITY_COLUMN} IN ({listed})")
                .select([VECTOR_COLUMN, CHUNK_COLUMN, IDENTITY_COLUMN])
                .to_list()
            )
            for record in found:
                rows.setdefault(str(record[IDENTITY_COLUMN]), record)
        return rows

    def _verdict(
        self, chunk: Chunk, record: dict[str, Any], fingerprint: EmbedFingerprint
    ) -> StoredVector:
        """Classify one stored row against the chunk it is being offered for.

        The three things a Lance row knows are read here; what they *mean* is decided by
        :func:`~manicule.core.embedding.classify_stored_vector`, which every backend shares so
        that two of them cannot answer one question two ways.
        """
        stored = record.get(VECTOR_COLUMN)
        return classify_stored_vector(
            chunk,
            recorded_identity=str(record[IDENTITY_COLUMN] or UNRECORDED_IDENTITY),
            stored_embed_text=_embed_text_of(record),
            stored_vector=None if stored is None else [float(value) for value in stored],
            embed=fingerprint,
            middleware=self._middleware,
        )

    async def _unranked(self, table: AsyncTable, k: int, predicate: str | None) -> list[Candidate]:
        """Candidates for a query the store cannot rank. See :meth:`search`."""
        query = table.query()
        if predicate is not None:
            query = query.where(predicate)
        records = (
            await query.select([ID_COLUMN, PUBLICATION_COLUMN, CHUNK_COLUMN]).limit(k).to_list()
        )
        return [
            Candidate(
                chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
                publication_id=str(record[PUBLICATION_COLUMN]),
                score=0.0,
            )
            for record in records
        ]

    def _row(
        self,
        chunk: Chunk,
        vector: Vector,
        fingerprint: EmbedFingerprint,
        publication_id: str,
    ) -> dict[str, object]:
        """One Lance row: the normalized vector, the promoted columns, the chunk, its identity."""
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
        """Add :data:`IDENTITY_COLUMN` to a table created before it existed.

        The whole migration for an existing ``vectors/`` directory, and it is deliberately the
        cheapest one available: a column of empty strings, no row rewritten, no vector read and
        no forward pass. Every existing row is then :data:`UNRECORDED_IDENTITY`, which
        :meth:`stored_vectors` reconstructs from the chunk the row already carries — so the
        upgrade costs an ``add_columns`` and nothing else, and an existing corpus keeps every
        vector it has. Idempotent, because :meth:`ensure_ready` runs on every process start.
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
        if additions:
            await table.add_columns(additions)

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
        operation_hook: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._directory = directory
        self._engine = engine
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

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> Mapping[str, StoredVector]:
        async with self._operation() as store:
            return await store.stored_vectors(chunks)

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[LanceVectorStore]:
        """Pin and revalidate one pointer before exposing its store to an operation."""
        while True:
            pointer = await self._pointer()
            key = pointer if pointer and pointer.startswith("reembed-") else "legacy"
            directory = (
                self._directory / "generations" / key if key != "legacy" else self._directory
            )
            if key == "legacy":
                self._publication_pointer = pointer
                store = await self._prepared_store(key, directory)
                if self._operation_hook is not None:
                    await self._operation_hook()
                yield store
                return
            async with generation_pin(directory):
                if await self._pointer() != pointer:
                    continue
                self._publication_pointer = pointer
                store = await self._prepared_store(key, directory)
                if self._operation_hook is not None:
                    await self._operation_hook()
                yield store
                return

    async def _pointer(self) -> str | None:
        from sqlalchemy import select  # noqa: PLC0415 - storage remains lazy

        from manicule.storage import models  # noqa: PLC0415 - avoids import cycle at startup

        async with self._engine.connect() as connection:
            value = (
                await connection.execute(
                    select(models.IndexState.vector_table).where(models.IndexState.id == 1)
                )
            ).scalar_one_or_none()
        return None if value is None else str(value)

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


__all__ = [
    "EXEMPT_FILTER_FIELDS",
    "FILTERABLE_COLUMNS",
    "FLOAT32_EPSILON",
    "IDENTITY_COLUMN",
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
    "table_name",
    "unit",
]
