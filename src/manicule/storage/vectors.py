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
serialisation byte for byte. Two unrelated 1024-dimension models pass any size check and
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
import hashlib
import math
from typing import TYPE_CHECKING, Any, Final

import lancedb
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector as FixedSizeVector
from pydantic import create_model

from manicule.core.content import Chunk
from manicule.core.embedding import EmbedFingerprint
from manicule.core.errors import ManiculeError
from manicule.core.retrieval import Candidate, Filter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from lancedb.db import AsyncConnection
    from lancedb.table import AsyncTable

    from manicule.core.embedding import Vector

META_TABLE: Final = "_manicule_meta"
"""Where the fingerprint lives, beside the vectors it describes."""

TABLE_PREFIX: Final = "chunks__"
"""Vector tables are ``chunks__<fp8>``; the suffix is the fingerprint hash (§6.5)."""

FINGERPRINT_HASH_LENGTH: Final = 8

ID_COLUMN: Final = "id"
VECTOR_COLUMN: Final = "vector"
CHUNK_COLUMN: Final = "chunk_json"
DISTANCE_COLUMN: Final = "_distance"

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
""":class:`~manicule.core.retrieval.Filter` fields this store neither honours nor refuses.

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
        ValueError: When ``filter`` sets a field this store can neither honour nor has been
            granted an exemption for. Refusing is the point: quietly dropping a restriction
            returns results the filter was written to exclude, and the search still looks
            like it worked.
    """
    if filter is None:
        return None

    unhonoured = sorted(
        filter.restricting_fields - PUSHED_DOWN_FILTER_FIELDS - EXEMPT_FILTER_FIELDS
    )
    if unhonoured:
        msg = (
            f"the vector table has no column for {', '.join(unhonoured)}, so this store "
            f"cannot honour {'them' if len(unhonoured) > 1 else 'it'}. Resolve those fields "
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


def unit(vector: Vector) -> list[float]:
    """``vector`` scaled to length one, so that cosine distance is ``1 - similarity``.

    A vector of all zeros has no direction and is returned unchanged. Nothing here can invent
    one for it, and cosine similarity against it is undefined rather than small — see
    :meth:`LanceVectorStore.search` for what the store does about that.
    """
    values = [float(value) for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


class _MetaRow(LanceModel):
    """The one row of ``_manicule_meta``.

    Two columns because they answer different questions. ``embed_fingerprint`` is the whole
    model, which is what :meth:`LanceVectorStore.fingerprint` has to return — the canonical
    form holds identity fields only and cannot rebuild the type. ``canonical`` is the
    identity serialisation the table name hashes, comparable byte for byte by anything that
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
        VECTOR_COLUMN: (FixedSizeVector(dimension), ...),
        "document_id": (str, ...),
        "kind": (str, ...),
        "lang": (str | None, None),
        "position": (int, ...),
        CHUNK_COLUMN: (str, ...),
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

    # --- lifecycle -----------------------------------------------------------------------

    async def ensure_ready(self, fingerprint: EmbedFingerprint) -> None:
        """Prepare the store for vectors from ``fingerprint``.

        First call writes ``_manicule_meta`` and creates ``chunks__<fp8>`` at the dimension
        the embedder reports. Later calls compare what is stored.

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

    async def teardown(self) -> None:
        """Close the LanceDB connection. Safe to call when none was ever opened."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._table = None

    async def fingerprint(self) -> EmbedFingerprint | None:
        """The fingerprint these vectors were built with, or ``None`` if there are none."""
        if self._fingerprint is not None:
            return self._fingerprint
        async with self._lock:
            return await self._stored_fingerprint(await self._connect())

    # --- writing -------------------------------------------------------------------------

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Store ``vectors`` against ``chunks``, replacing any rows for those chunk ids.

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
            self._row(chunk, vector, fingerprint.dimension)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await (
            table.merge_insert(ID_COLUMN)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    async def delete_document(self, document_id: str) -> None:
        """Remove every vector belonging to a document. Idempotent."""
        table = await self._existing_table()
        if table is None:
            return
        await table.delete(f"document_id = {quote(document_id)}")

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Remove named vectors. Idempotent, and what the tombstone sweep calls.

        By id, from a list the sweep was handed — never by anti-joining the whole table against
        ``chunks``. That comparison races concurrent ingest: an id written after the scan began
        looks like an orphan, and the sweep deletes a live vector. Tombstones exist so this
        method can only ever name something that was deleted.
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

        Stored vectors are L2-normalised and the metric is cosine, so ``score`` is
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
                ``filter`` sets a field this store cannot honour.
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
        records = await search.select([ID_COLUMN, CHUNK_COLUMN, DISTANCE_COLUMN]).limit(k).to_list()
        return [
            Candidate(
                chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
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

    # --- internals -----------------------------------------------------------------------

    async def _unranked(self, table: AsyncTable, k: int, predicate: str | None) -> list[Candidate]:
        """Candidates for a query the store cannot rank. See :meth:`search`."""
        query = table.query()
        if predicate is not None:
            query = query.where(predicate)
        records = await query.select([ID_COLUMN, CHUNK_COLUMN]).limit(k).to_list()
        return [
            Candidate(chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])), score=0.0)
            for record in records
        ]

    @staticmethod
    def _row(chunk: Chunk, vector: Vector, dimension: int) -> dict[str, object]:
        """One Lance row: the normalised vector, the promoted columns, and the chunk."""
        values = unit(vector)
        if len(values) != dimension:
            msg = (
                f"chunk {chunk.id!r} was offered a {len(values)}-dimension vector but the "
                f"index was built for {dimension}. The dimension comes from the embedder's "
                f"fingerprint, so a disagreement here means two embedders are in play."
            )
            raise ValueError(msg)
        return {
            ID_COLUMN: chunk.id,
            VECTOR_COLUMN: values,
            "document_id": chunk.document_id,
            "kind": chunk.kind.value,
            "lang": chunk.lang,
            "position": chunk.position,
            CHUNK_COLUMN: chunk.model_dump_json(),
        }

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
                f"identity is {canonical}, but the fingerprint stored beside it canonicalises "
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
            return await connection.open_table(name)
        return await connection.create_table(name, schema=_row_model(fingerprint.dimension))

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


__all__ = [
    "EXEMPT_FILTER_FIELDS",
    "FILTERABLE_COLUMNS",
    "META_TABLE",
    "PUSHED_DOWN_FILTER_FIELDS",
    "TABLE_PREFIX",
    "LanceVectorStore",
    "VectorStoreStateError",
    "fingerprint_hash",
    "membership",
    "predicate_for",
    "quote",
    "table_name",
    "unit",
]
