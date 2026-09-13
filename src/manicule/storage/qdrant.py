"""Qdrant behind :class:`~manicule.core.protocols.VectorStore`.

``docs/storage.md`` §6.3 is the design this implements, and §6.2 is the one it mirrors: the
row a Lance table holds in columns, a Qdrant collection holds in a point's payload, under the
same names, written from the same object by the same rules. Everything that is a property of
*manicule* rather than of an engine — the field names, the filter fields a store may answer,
the integrity verdict — comes from :mod:`manicule.storage.vector_schema`, so the two backends
cannot drift into answering one question two ways.

Four things here are not simply the Lance store with a different client under it.

**The index lives on a machine this process does not own.** That is the whole point — it is
what lets several manicule processes, or a container with no persistent volume, share one
index — and it is also the cost: the vector store becomes a network dependency, backups become
Qdrant's problem rather than a directory copy, and a local-only data policy has something new
to say about the corpus leaving this machine. ``docs/deployment.md`` §1.1 is the operational
consequence; :meth:`~manicule.config.settings.Settings.policy_problems` is where the policy
one is refused rather than documented.

**One collection per workspace and per embedding space.** The Lance store puts a workspace in
its own directory and a fingerprint in its own table inside it. Qdrant has one flat namespace,
so both live in the name: ``<prefix>_<workspace digest>_chunks__<fp8>``. Two workspaces
therefore cannot meet, and neither can two embedding spaces — which matters more here than it
does on local disk, because a shared server is the configuration this store exists for.

**Two *installations* are kept apart by the prefix, and that is the operator's to set.** Named
rather than derived, because nothing identifies an installation that survives the things an
installation survives: a data directory moves, a container is rebuilt, a workspace is called
``default`` on both machines. So two installations that share a server and both leave
``storage.qdrant.collection_prefix`` at its default share collections — which is safe for
retrieval, since the hydrating join admits only document ids the local authority knows, and is
wrong for everything else: ``count`` reports the union, and a second installation whose
embedder differs is refused by the fingerprint record rather than served. Give each
installation its own prefix on a shared server; ``docs/deployment.md`` §6.5 says so where an
operator will read it.

**A readback is narrowed to float32 before it is believed.** The numerical-integrity checksum
is defined over the ``binary32`` values that were persisted. Qdrant stores exactly those, but
the REST transport serializes them as JSON decimal text, and parsing that text yields a
``float64`` a few parts in 10^9 away — enough for every recomputed digest to disagree and for
a healthy corpus to read as entirely corrupt. Narrowing the readback undoes precisely the
transport's rounding and nothing else: it is *not* :func:`canonical_stored_vector`, which
would also re-normalize and so would quietly repair a stored vector that had genuinely drifted
off unit length, which is the one thing the checksum exists to notice. Over gRPC the
narrowing is an identity operation, which is the point — the store's answers do not depend on
which transport an operator configured.

**A non-legacy publication's rows are immutable, enforced by a read.** ``docs/storage.md`` §6.4
keys a physical row by publication plus chunk so that staging a replacement cannot overwrite
the generation retrieval is still serving. Lance gets this from ``merge_insert`` with only
``when_not_matched_insert_all``. Qdrant's ``upsert`` is an unconditional write, so the same
rule is kept by asking which of the batch's ids already exist and writing only the rest.

This store deliberately does **not** implement
:class:`~manicule.core.protocols.AnnIndexMaintenance`. Qdrant builds and maintains its own HNSW
index on its own schedule, so there is no build for an operator to trigger and no partition
count this project chose; reporting an IVF-PQ lifecycle for an index manicule neither built nor
can replace would be describing a mechanism that is not there. Nor does it implement the
shadow-generation surface re-embedding needs: ``manicule.app.runtime`` refuses a durable
re-embed on any backend that does not, by name, and that refusal is the honest answer rather
than a half-built one.
"""

from __future__ import annotations

import hashlib
import struct
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlsplit

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.conversions.common_types import PointId
from qdrant_client.http.exceptions import ApiException

from manicule.core.content import LEGACY_PUBLICATION, Chunk
from manicule.core.embedding import (
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
)
from manicule.core.errors import VectorStoreStateError
from manicule.core.ids import vector_id
from manicule.core.lifecycle import HealthReport, HealthState
from manicule.core.retrieval import Candidate
from manicule.storage.vector_schema import (
    CHECKSUM_COLUMN,
    CHECKSUM_VERSION_COLUMN,
    CHUNK_COLUMN,
    CHUNK_ID_COLUMN,
    DOCUMENT_ID_COLUMN,
    ID_COLUMN,
    IDENTITY_COLUMN,
    KIND_COLUMN,
    LANG_COLUMN,
    POSITION_COLUMN,
    PUBLICATION_COLUMN,
    VECTOR_COLUMN,
    checksum_of,
    embed_text_of,
    fingerprint_hash,
    refuse_unhonored_fields,
    row_integrity,
    space_name,
    unit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.embedding import Vector
    from manicule.core.retrieval import Filter

POINT_NAMESPACE: Final = uuid.UUID("bf695e23-5f0a-50ac-8b49-8ab96140066a")
"""The UUIDv5 namespace every point id is derived in.

Written as the literal it evaluates to rather than as a ``uuid5`` call over a string, because
this value is part of the on-disk format: change the string the namespace was derived from and
every existing point acquires a new id, which reads as an empty collection rather than as an
error. A constant cannot drift; an expression can.
"""

WORKSPACE_DIGEST_LENGTH: Final = 16
"""Hex characters of the workspace SHA-256 that reach a collection name."""

META_COLLECTION_SUFFIX: Final = "_meta"
"""Where the fingerprint lives — one point per workspace, beside the vectors it describes."""

META_DIMENSION: Final = 1
"""The meta collection's vector width.

Qdrant has no collection-level metadata and no table that is not a vector collection, so the
fingerprint is a point, and a point needs a vector. One dimension, never searched, retrieved
only by id. The alternative — a reserved point inside the vector collection — would put a row
that is not a vector inside every count, every scroll and every unranked search, and each of
those would then need to remember to exclude it.
"""

UPSERT_BATCH: Final = 256
"""Points per write. The bound on one request's size and on what a failed request costs."""

SCROLL_PAGE: Final = 512
"""Points per page for the integrity scan and the backfill.

The bound that keeps both operations constant-memory over a corpus of any size. A page holds
``SCROLL_PAGE`` vectors — at 1024 float32 components that is about two megabytes — and nothing
accumulates across pages except integers.
"""

RETRIEVE_PAGE: Final = 512
"""Ids per ``retrieve``. One request per document is small; one per corpus is not a request."""

INDEXED_PAYLOAD_FIELDS: Final = (
    (DOCUMENT_ID_COLUMN, models.PayloadSchemaType.KEYWORD),
    (KIND_COLUMN, models.PayloadSchemaType.KEYWORD),
    (LANG_COLUMN, models.PayloadSchemaType.KEYWORD),
    (CHUNK_ID_COLUMN, models.PayloadSchemaType.KEYWORD),
    (IDENTITY_COLUMN, models.PayloadSchemaType.KEYWORD),
    (CHECKSUM_COLUMN, models.PayloadSchemaType.KEYWORD),
)
"""Every payload field a query filters on, indexed when the collection is created.

Created up front rather than when a query first needs one, because Qdrant's filterable-HNSW
traversal can only use an index the graph was built against: adding one to a populated
collection leaves the existing graph without the filter-aware edges, so the index is present
and the query is still a scan. The list is exactly the fields this module filters on — one
entry per filter this store can build, checked by ``tests/test_storage_qdrant.py``.
"""


def workspace_digest(workspace_id: str) -> str:
    """The opaque, stable namespace one workspace occupies in a collection name.

    The **same SHA-256** the embedded store names a workspace's directory with, truncated to
    :data:`WORKSPACE_DIGEST_LENGTH`. Two properties are wanted and they pull against each
    other: a directory name nobody reads can carry all 64 hex characters, while a collection
    name is what an operator reads off a Qdrant dashboard looking for their corpus. Sixteen is
    the compromise, and 64 bits over the handful of workspaces one installation has is not a
    collision anybody will meet.

    A workspace id is arbitrary text and a collection name is not, which is why it is hashed at
    all rather than spelled out.
    """
    return hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:WORKSPACE_DIGEST_LENGTH]


def collection_for(prefix: str, workspace_id: str, fingerprint: EmbedFingerprint) -> str:
    """The collection these vectors belong in, on a server that may hold several corpora.

    Both scopes are in the name because Qdrant has one flat namespace: the workspace, so two
    corpora cannot share rows, and the fingerprint hash, so two embedding spaces cannot.
    """
    return f"{prefix}_{workspace_digest(workspace_id)}_{space_name(fingerprint)}"


def meta_collection_for(prefix: str) -> str:
    """Where every workspace's fingerprint is recorded on this server."""
    return f"{prefix}{META_COLLECTION_SUFFIX}"


def point_id_for(row_id: str) -> str:
    """The Qdrant point id for a physical row id.

    Qdrant accepts an unsigned integer or a UUID and nothing else, while a physical row id is
    :func:`~manicule.core.ids.vector_id`'s output — a digest for a published row, and the chunk
    id itself for a legacy one, which is arbitrary text. A UUIDv5 over the row id is a total
    function onto what Qdrant accepts, and a deterministic one: the same row written twice
    lands on the same point, which is what makes a retry an overwrite rather than a duplicate.

    The row id is kept in the payload as well, because this direction does not invert and a
    caller that has the point needs to be able to say which row it is.
    """
    return str(uuid.uuid5(POINT_NAMESPACE, row_id))


def as_float32(values: Sequence[float]) -> list[float]:
    """``values`` rounded to the ``binary32`` each one already was when it was stored.

    The REST transport writes a stored ``float32`` as JSON decimal text with just enough digits
    to identify it, and parsing that text produces the nearest ``float64`` — a different number,
    by a few parts in 10^9, from the one that was persisted. The checksum is defined over the
    persisted representation, so without this every recomputed digest disagrees and a healthy
    corpus reads as entirely corrupt.

    Deliberately narrower than :func:`~manicule.core.embedding.canonical_stored_vector`, which
    also re-normalizes: re-normalizing a readback would silently repair a stored vector that had
    drifted off unit length, and noticing exactly that is what the checksum is for. Over gRPC
    the stored values arrive as ``float32`` already and this returns them unchanged.
    """
    return [struct.unpack("!f", struct.pack("!f", value))[0] for value in values]


def is_local(client: AsyncQdrantClient) -> bool:
    """Whether ``client`` runs Qdrant in this process rather than talking to a server.

    Read from the client's own public ``init_options`` and using the library's own rule —
    ``:memory:`` or a ``path`` selects local mode — rather than by reaching for a private
    attribute. Local mode is a test convenience with a different engine underneath: it searches
    exhaustively and ignores payload indexes, warning when one is created. Asking it to create
    them anyway would make every test that builds a store emit a warning, and this project turns
    warnings into errors.
    """
    options = client.init_options
    return bool(options.get("path")) or options.get("location") == ":memory:"


class QdrantVectorStore:
    """:class:`~manicule.core.protocols.VectorStore` on a Qdrant server.

    Construction opens nothing: the client is handed in, already configured, and the first
    request is whatever the first operation needs. That is deliberate and matches the Lance
    store — ``manicule doctor``, a plugin listing and a completion script all build the
    container, and none of them should dial a database to do it.

    The client's lifetime belongs to whoever passed it in unless ``owns_client`` says
    otherwise. The plugin factory constructs one per store and says so, which is what makes
    :meth:`teardown` — called by the container on shutdown — close the sockets.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        workspace_id: str,
        collection_prefix: str,
        owns_client: bool = False,
    ) -> None:
        self._client = client
        self._workspace = workspace_id
        self._prefix = collection_prefix
        self._owns_client = owns_client
        self._fingerprint: EmbedFingerprint | None = None
        self._middleware: tuple[str, ...] = ()

    @property
    def workspace_id(self) -> str:
        """The workspace this handle serves, which is half of every collection name it uses.

        Public because a person looking at a Qdrant dashboard needs it to find their corpus:
        the digest in the collection name is deliberately opaque, and the only way back to the
        workspace it stands for is to hash the candidate and compare.
        """
        return self._workspace

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        """Prepare the store for vectors from ``fingerprint``.

        The first call creates the collection at the dimension the embedder reports, indexes
        every payload field a query filters on, and records the fingerprint. Later calls
        compare what is recorded.

        The collection is created before the fingerprint is recorded, and both steps check for
        what they are about to make. An interruption between them therefore leaves a collection
        with no fingerprint beside it, which the next call completes; the other order would
        leave a fingerprint claiming a collection that does not exist, and every read would then
        fail against a store that reported itself ready.

        **The residual limit, stated rather than hidden.** Reading the record and writing it are
        two requests, so two processes preparing one workspace for the *first* time can both see
        nothing and both record. The last write wins. Vectors do not mix — a collection's name
        carries the fingerprint hash, so each process writes to its own — and the cost is that
        one of the two is later refused by a record naming the other's model. Reaching it needs
        two processes, one workspace, two embedders and no prior ingest; a single data directory
        admits one writer at a time (``docs/ingest.md`` §6.5), so in practice it needs two
        installations sharing a prefix, which is the configuration above says not to have.

        Raises:
            FingerprintMismatchError: When this workspace already holds vectors from a
                different model, including a different model of the same size.
        """
        recorded = await self._recorded_fingerprint()
        if recorded is not None:
            recorded.require_match(fingerprint)
        await self._ensure_collection(fingerprint)
        if recorded is None:
            await self._record_fingerprint(fingerprint)
        self._fingerprint = fingerprint
        self._middleware = tuple(embed_text_middleware)

    async def fingerprint(self) -> EmbedFingerprint | None:
        """The fingerprint this store was built with, or ``None`` if it holds nothing yet."""
        return await self._recorded_fingerprint()

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = LEGACY_PUBLICATION,
    ) -> None:
        """Store vectors against chunks in one publication generation.

        A physical point is keyed by ``publication_id`` plus logical chunk id, so staging a
        replacement cannot overwrite the generation retrieval still serves. For any publication
        but the legacy one that keying is a promise of immutability, and here it is kept by a
        read: the ids already present are asked for and excluded, so a replayed checkpoint
        writes only what is missing rather than rewriting points a validation pass has already
        accepted.

        **The residual limit, stated rather than hidden.** That read and this write are two
        requests, not one transaction. Qdrant offers no insert-if-absent and no compare-and-set,
        so where the embedded store gets immutability from a single ``merge_insert`` commit that
        no concurrent writer can interleave with, this gets it from a check followed by an act.
        A second writer that starts after the check and finishes after the write replaces the
        point rather than being refused. What that costs is bounded by what the two writers can
        disagree about: the id is derived from the publication and the chunk, the payload is
        derived from the chunk, and the vector is a pure function of the chunk's embedding input
        under the fingerprint both writers had to match to get here — so a race between two
        writers of the *same* publication rewrites a point with the same content, and the case
        it does not cover is a writer holding a vector from an embedder that is not
        reproducible. The replay this guard is actually for — one worker resuming its own
        checkpoint — is sequential and is fully covered.

        Raises:
            ValueError: If ``chunks`` and ``vectors`` are different lengths, or if a vector is
                not the dimension the collection was created for, or is not finite.
            VectorStoreStateError: If :meth:`ensure_ready` has not run.
        """
        fingerprint = self._ready()
        if len(chunks) != len(vectors):
            msg = (
                f"upsert was given {len(chunks)} chunks and {len(vectors)} vectors. They are "
                f"parallel sequences; a mismatch means the caller has lost track of which "
                f"vector belongs to which chunk, and storing the overlap would key some of "
                f"them to the wrong text."
            )
            raise ValueError(msg)
        if not chunks:
            return

        points = [
            self._point(chunk, vector, fingerprint, publication_id)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if publication_id != LEGACY_PUBLICATION:
            points = await self._without_existing(points, fingerprint)
        collection = self._collection(fingerprint)
        for start in range(0, len(points), UPSERT_BATCH):
            await self._client.upsert(
                collection_name=collection,
                points=points[start : start + UPSERT_BATCH],
                wait=True,
            )

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> Mapping[str, StoredVector]:
        """What this store holds for each of ``chunks``, and whether it can still be used.

        **What a verdict means is decided by
        :func:`~manicule.core.embedding.classify_stored_vector`**, which every backend shares
        and which is the only place the rule is written down. This method's job is to produce
        the three things a stored point knows — its recorded identity, the ``embed_text`` of the
        chunk stored beside it, and the vector — and to look in the two places a point can be.

        **Two lookups, because a chunk id is not the only way a point can belong to a chunk.**
        The first is by id. The second is by embedding-input identity, for the chunks the first
        did not answer: a chunk id carries its position, so inserting one paragraph renames
        every chunk below it while moving no embedding input at all, and keyed on the id alone
        that edit re-embeds the whole document.

        **Answers without requiring :meth:`ensure_ready`**, like :meth:`count` and
        :meth:`delete_document`. The fingerprint it compares against is the one the server
        records, which is the one the stored vectors were made with. A workspace that holds
        nothing answers that it holds nothing for every chunk, which is true.
        """
        verdicts = {chunk.id: StoredVector(state=VectorState.ABSENT) for chunk in chunks}
        if not chunks:
            return verdicts
        fingerprint = await self._recorded_fingerprint()
        if fingerprint is None or not await self._collection_exists(fingerprint):
            return verdicts

        by_id = {chunk.id: chunk for chunk in chunks}
        for record in await self._records_matching(
            fingerprint, CHUNK_ID_COLUMN, sorted(by_id), with_vectors=True
        ):
            chunk = by_id.get(str(record.get(CHUNK_ID_COLUMN, "")))
            if chunk is None:  # pragma: no cover - the filter asked for these ids only
                continue
            found = self._verdict(chunk, record, fingerprint)
            if _STATE_PRIORITY[found.state] > _STATE_PRIORITY[verdicts[chunk.id].state]:
                verdicts[chunk.id] = found

        wanted = {
            chunk.id: self._identity_of(chunk, fingerprint)
            for chunk in chunks
            if verdicts[chunk.id].state in {VectorState.ABSENT, VectorState.STALE}
        }
        if not wanted:
            return verdicts
        by_identity: dict[str, dict[str, Any]] = {}
        for record in await self._records_matching(
            fingerprint, IDENTITY_COLUMN, sorted(set(wanted.values())), with_vectors=True
        ):
            by_identity.setdefault(str(record.get(IDENTITY_COLUMN, "")), record)
        for chunk_id, identity in wanted.items():
            record = by_identity.get(identity)
            if record is None:
                continue
            verdicts[chunk_id] = choose_stored_vector(
                verdicts[chunk_id], self._verdict(by_id[chunk_id], record, fingerprint)
            )
        return verdicts

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol and the domain
    ) -> list[Candidate]:
        """Return up to ``k`` nearest chunks, best first.

        Stored vectors are L2-normalized and the metric is cosine, so Qdrant's score *is* a
        cosine similarity in ``[-1, 1]`` rather than a distance to be subtracted from one. It
        is clamped to that interval, which float error can otherwise exceed by an ulp or two.

        **A point whose numbers do not match its checksum is dropped rather than ranked.** A
        candidate is a chunk plus a score, and the score is computed against the stored vector —
        so returning one computed against corrupted numbers would put a result in a ranked list
        on the strength of bytes nothing vouches for. A search can therefore return fewer than
        ``k`` candidates over a damaged collection, which is the honest outcome.

        **A query with no direction.** The zero vector is not near anything and cosine
        similarity against it is undefined for every row, so ranking it would be inventing an
        order. Instead the store returns the first ``k`` points the filter admits, each scored
        ``0.0``: the one value that asserts neither similarity nor difference.

        Raises:
            ValueError: If ``vector`` is not the dimension the collection was created for, or
                if ``filter`` sets a field this store cannot honor.
            VectorStoreStateError: If :meth:`ensure_ready` has not run.
        """
        fingerprint = self._ready()
        query = unit(vector)
        if len(query) != fingerprint.dimension:
            msg = (
                f"a {len(query)}-dimension query was offered to an index built for "
                f"{fingerprint.dimension} by {fingerprint.describe()}."
            )
            raise ValueError(msg)
        condition = self._filter(filter)
        if k <= 0:
            return []
        if not any(query):
            return await self._unranked(fingerprint, k, condition)

        response = await self._client.query_points(
            collection_name=self._collection(fingerprint),
            query=query,
            query_filter=condition,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )
        candidates: list[Candidate] = []
        for point in response.points:
            record = _record_of(point.payload, point.vector)
            if not row_integrity(record).accepts:
                continue
            candidates.append(
                Candidate(
                    chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
                    publication_id=str(record.get(PUBLICATION_COLUMN) or LEGACY_PUBLICATION),
                    score=min(1.0, max(-1.0, float(point.score))),
                )
            )
        return candidates

    async def delete_document(self, document_id: str) -> None:
        """Remove every vector belonging to a document. Idempotent."""
        fingerprint = await self._recorded_fingerprint()
        if fingerprint is None or not await self._collection_exists(fingerprint):
            return
        await self._client.delete(
            collection_name=self._collection(fingerprint),
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=DOCUMENT_ID_COLUMN,
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Remove the named physical rows. Idempotent, and takes row ids rather than chunk ids.

        The tombstone sweep takes its ids from ``chunks.vector_id`` in the relational store —
        physical row ids — because a logical chunk id can name several publications' rows and
        the sweep means exactly the one it recorded. Qdrant deletes by point id, which is that
        row id under :func:`point_id_for`, so the sweep needs no second lookup to say what it
        meant, and the historical parameter name is the narrow protocol's
        (:class:`~manicule.ingest.sweeps.VectorSweepTarget`).
        """
        fingerprint = await self._recorded_fingerprint()
        if fingerprint is None or not chunk_ids or not await self._collection_exists(fingerprint):
            return
        ids = [point_id_for(row_id) for row_id in chunk_ids]
        collection = self._collection(fingerprint)
        for start in range(0, len(ids), UPSERT_BATCH):
            await self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=list(ids[start : start + UPSERT_BATCH])),
                wait=True,
            )

    async def count(self) -> int:
        """How many vectors are stored."""
        fingerprint = await self._recorded_fingerprint()
        if fingerprint is None or not await self._collection_exists(fingerprint):
            return 0
        result = await self._client.count(collection_name=self._collection(fingerprint), exact=True)
        return int(result.count)

    async def checksum_coverage(
        self, *, recompute: bool = False, page_size: int = SCROLL_PAGE
    ) -> VectorChecksumCoverage:
        """How many stored vectors carry a checksum, and — on request — how many still match.

        Two modes, because two different questions get called "coverage" and only one of them
        is affordable on a status page.

        **Counting** is three ``count`` calls with a payload filter and reads no vector. It
        answers "has the backfill finished", and it is what ``status`` and ``doctor`` call —
        which matters more here than on local disk, where the cheap mode merely saves I/O and
        here it saves pulling the whole corpus back over a socket.

        **Recomputing** reads every point's vector in bounded pages and hashes it. It answers
        "are the numbers still what they were", costs a scan of the corpus, and is what the
        checksum command performs when an operator asks for it.

        A collection reached over a network is shared infrastructure — anything holding the API
        key can write a point into it — so "every row carries a checksum" is measured here
        rather than assumed from the fact that this store always writes one.

        Args:
            recompute: Verify each recorded checksum rather than only counting it.
            page_size: Points per page while recomputing.

        Returns:
            Aggregate counts and a typed failure split. Never a checksum value, a vector
            component or a chunk identifier.
        """
        if page_size < 1:
            raise ValueError("integrity scan page size must be positive")
        fingerprint = await self._recorded_fingerprint()
        if fingerprint is None or not await self._collection_exists(fingerprint):
            return VectorChecksumCoverage(scanned=False)
        collection = self._collection(fingerprint)
        rows = await self._count_matching(collection, None)
        recorded = rows - await self._count_matching(collection, _unrecorded_checksum())
        if not recompute:
            # A half-written pair is malformed by looking at the two fields, which costs a
            # third count and no vector read. Reported here rather than left to the scan,
            # because a surface that cannot afford a scan is exactly the one that would
            # otherwise call a collection holding such a point complete.
            half_written = await self._count_matching(collection, _half_written_checksum())
            return VectorChecksumCoverage(
                rows=rows,
                recorded=recorded,
                failed=half_written,
                failures=({VectorIntegrity.MALFORMED.value: half_written} if half_written else {}),
            )
        verified = 0
        failures: dict[str, int] = {}
        async for record in self._scan(fingerprint, page_size=page_size, with_vectors=True):
            integrity = row_integrity(record)
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
        self, *, limit: int = SCROLL_PAGE, dry_run: bool = False
    ) -> VectorChecksumBackfill:
        """Give a bounded page of pre-checksum points the checksum their stored vector implies.

        **It hashes what is stored and nothing else.** No embedder, no parser, no connector, no
        source system. That is what makes it affordable, and it is also the limit of what it can
        claim: a point damaged *before* this ran gets a checksum over the damaged numbers, so
        the backfill establishes integrity from here forward rather than retroactively.

        **Resumable and idempotent by construction.** The page is selected by "records no
        checksum", so a point this pass finished is not a point the next pass can see. There is
        no cursor to persist and none to lose.

        **Points whose vector cannot be hashed are left alone**, counted as
        :attr:`~manicule.core.embedding.VectorChecksumBackfill.unhashable`. Writing a checksum
        over a non-finite vector would certify a point that can never be ranked.

        **A point carrying one half of the pair is never selected at all.** It is malformed
        rather than unrecorded, and a pass that hashed it would replace a point announcing that
        its two halves disagree with one that verifies — the same laundering the unhashable
        branch refuses, arriving through the payload rather than through the vector.

        Args:
            limit: Points to consider in this pass. The bound on both the request and the
                memory.
            dry_run: Report what a pass would do and write nothing.

        Returns:
            What the pass did, and how many points still record no checksum.
        """
        if limit < 1:
            raise ValueError("checksum backfill limit must be positive")
        fingerprint = await self._recorded_fingerprint()
        if fingerprint is None or not await self._collection_exists(fingerprint):
            return VectorChecksumBackfill(dry_run=dry_run)
        collection = self._collection(fingerprint)
        unrecorded = _unrecorded_checksum()
        outstanding = await self._count_matching(collection, unrecorded)
        if outstanding == 0:
            return VectorChecksumBackfill(dry_run=dry_run)

        found, _ = await self._client.scroll(
            collection_name=collection,
            scroll_filter=unrecorded,
            limit=limit,
            with_payload=False,
            with_vectors=True,
        )
        written = unhashable = 0
        for point in found:
            values = _vector_of(point.vector)
            if values is None or not is_finite_vector(values):
                unhashable += 1
                continue
            written += 1
            if dry_run:
                continue
            # One point at a time rather than one call for the page: Qdrant sets a payload for
            # a set of ids or for a filter, and neither writes a *different* digest per point.
            # A point deleted between the read and the write is simply not there, so a
            # concurrent sweep wins and the next pass finds nothing — which is the outcome the
            # embedded store gets from restricting its merge to rows that already match.
            await self._client.set_payload(
                collection_name=collection,
                payload={
                    CHECKSUM_COLUMN: vector_checksum(tuple(values)),
                    CHECKSUM_VERSION_COLUMN: VECTOR_CHECKSUM_VERSION,
                },
                points=[point.id],
                wait=True,
            )
        return VectorChecksumBackfill(
            scanned=len(found),
            written=written,
            unhashable=unhashable,
            remaining=(
                outstanding if dry_run else await self._count_matching(collection, unrecorded)
            ),
            dry_run=dry_run,
        )

    async def health(self) -> HealthReport:
        """Whether the server is reachable, and what to do when it is not.

        The one check a local directory never needs and a network index always does. It asks
        for the collection list rather than for ``/healthz``: that exercises the transport,
        the TLS and the credential this installation actually configured, where an unauthorized
        ping would report a healthy server this process still cannot use.
        """
        where = self._endpoint()
        try:
            await self._client.get_collections()
        except Exception as exc:  # noqa: BLE001 - every failure here is a report, not a raise
            return HealthReport(
                state=HealthState.FAILING,
                detail=f"the Qdrant server at {where} did not answer: {type(exc).__name__}: {exc}",
                remedy=(
                    "check that storage.vector_db_url points at a running Qdrant, that this "
                    "host can reach it, and that storage.qdrant.api_key matches the server's."
                ),
            )
        if not is_local(self._client) and is_cleartext_remote(where):
            return HealthReport(
                state=HealthState.DEGRADED,
                detail=(
                    f"Qdrant at {where} answered, over http rather than https. The chunk's "
                    f"text is stored beside its vector and the API key is sent with every "
                    f"request, so both cross the network in the clear."
                ),
                remedy=(
                    "Use an https endpoint, or accept it knowingly on a network you control — "
                    "a cluster-internal service is the case this is reported rather than "
                    "refused for."
                ),
            )
        return HealthReport(state=HealthState.OK, detail=f"Qdrant at {where} answered")

    async def teardown(self) -> None:
        """Close the client, when this store is the one that opened it.

        Safe to call twice and after a failed start, which is when it is most needed.
        """
        if self._owns_client:
            await self._client.close()
            self._owns_client = False

    # --- internals ------------------------------------------------------------------------

    def _collection(self, fingerprint: EmbedFingerprint) -> str:
        return collection_for(self._prefix, self._workspace, fingerprint)

    def _meta_collection(self) -> str:
        return meta_collection_for(self._prefix)

    def _meta_point(self) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, f"meta:{self._workspace}"))

    def _endpoint(self) -> str:
        options = self._client.init_options
        url = options.get("url") or options.get("location")
        if isinstance(url, str) and url:
            return url
        host = options.get("host") or "localhost"
        return f"{host}:{options.get('port')}"

    def _ready(self) -> EmbedFingerprint:
        """The prepared fingerprint, or a refusal naming what was skipped."""
        if self._fingerprint is None:
            msg = (
                "the vector store has not been prepared: call ensure_ready(fingerprint) "
                "first. Until it has run the store does not know which vector space it is "
                "holding, and vectors from two models are not comparable."
            )
            raise VectorStoreStateError(msg)
        return self._fingerprint

    async def _collection_exists(self, fingerprint: EmbedFingerprint) -> bool:
        return await self._client.collection_exists(self._collection(fingerprint))

    async def _ensure_collection(self, fingerprint: EmbedFingerprint) -> None:
        """Create the collection and its payload indexes, or finish creating them.

        The indexes are (re)declared on **every** call rather than only when the collection is
        made, because creating one that exists is a no-op on the server and skipping them is
        not recoverable: a crash between the create and the index loop would otherwise leave a
        collection whose filters are permanently scans, with nothing to notice it. An index is
        invisible to correctness — Qdrant answers a filter with or without one — so the failure
        mode this closes is a corpus that silently gets slower as it grows.
        """
        collection = self._collection(fingerprint)
        if not await self._client.collection_exists(collection):
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=fingerprint.dimension, distance=models.Distance.COSINE
                ),
            )
        if is_local(self._client):
            return
        for field, schema in INDEXED_PAYLOAD_FIELDS:
            await self._client.create_payload_index(
                collection_name=collection, field_name=field, field_schema=schema
            )

    async def _recorded_fingerprint(self) -> EmbedFingerprint | None:
        """What the server says this workspace holds, or ``None`` if it has never held anything.

        Raises:
            VectorStoreStateError: If the recorded point contradicts itself — a fingerprint
                whose canonical form is not the one stored beside it. The record exists to make
                the collection self-describing; one that contradicts itself is not something to
                pick a winner from.
        """
        meta = self._meta_collection()
        if not await self._client.collection_exists(meta):
            return None
        found = await self._client.retrieve(
            collection_name=meta, ids=[self._meta_point()], with_payload=True
        )
        if not found:
            return None
        payload = found[0].payload or {}
        recorded = payload.get("embed_fingerprint")
        if not isinstance(recorded, str) or not recorded:
            msg = (
                f"{meta} holds a record for workspace {self._workspace!r} that names no "
                f"fingerprint. A point written by something other than this store, or one "
                f"half-written, describes an index nothing can identify — and guessing the "
                f"fingerprint is the mistake the record exists to prevent. Remove the point "
                f"and let the next ingest write it, or restore it."
            )
            raise VectorStoreStateError(msg)
        stored = EmbedFingerprint.model_validate_json(recorded)
        canonical = str(payload.get("canonical", ""))
        if stored.canonical() != canonical:
            msg = (
                f"{meta} contradicts itself for workspace {self._workspace!r}: the recorded "
                f"identity is {canonical}, but the fingerprint stored beside it canonicalizes "
                f"to {stored.canonical()}. The point has been edited or half-written; restore "
                f"it rather than trusting either half."
            )
            raise VectorStoreStateError(msg)
        return stored

    async def _record_fingerprint(self, fingerprint: EmbedFingerprint) -> None:
        meta = self._meta_collection()
        if not await self._client.collection_exists(meta):
            await self._client.create_collection(
                collection_name=meta,
                vectors_config=models.VectorParams(
                    size=META_DIMENSION, distance=models.Distance.DOT
                ),
            )
        await self._client.upsert(
            collection_name=meta,
            points=[
                models.PointStruct(
                    id=self._meta_point(),
                    vector=[1.0],
                    payload={
                        "embed_fingerprint": fingerprint.model_dump_json(),
                        "canonical": fingerprint.canonical(),
                        "workspace_id": self._workspace,
                        "collection": self._collection(fingerprint),
                        "fingerprint_hash": fingerprint_hash(fingerprint),
                    },
                )
            ],
            wait=True,
        )

    def _point(
        self,
        chunk: Chunk,
        vector: Vector,
        fingerprint: EmbedFingerprint,
        publication_id: str,
    ) -> models.PointStruct:
        """One stored point: the normalized vector, the promoted fields, the chunk, its identity.

        The checksum is taken from the output of
        :func:`~manicule.core.embedding.canonical_stored_vector`, which is the exact tuple this
        point stores, rather than from ``vector``. Hashing the argument would hash a
        representation that never reaches the server, and every readback would then disagree
        with it.
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
        row_id = vector_id(publication_id, chunk.id)
        return models.PointStruct(
            id=point_id_for(row_id),
            vector=list(values),
            payload={
                ID_COLUMN: row_id,
                CHUNK_ID_COLUMN: chunk.id,
                PUBLICATION_COLUMN: publication_id,
                DOCUMENT_ID_COLUMN: chunk.document_id,
                KIND_COLUMN: chunk.kind.value,
                LANG_COLUMN: chunk.lang,
                POSITION_COLUMN: chunk.position,
                CHUNK_COLUMN: chunk.model_dump_json(),
                IDENTITY_COLUMN: self._identity_of(chunk, fingerprint),
                CHECKSUM_COLUMN: vector_checksum(values),
                CHECKSUM_VERSION_COLUMN: VECTOR_CHECKSUM_VERSION,
            },
        )

    async def _without_existing(
        self, points: Sequence[models.PointStruct], fingerprint: EmbedFingerprint
    ) -> list[models.PointStruct]:
        """``points`` minus the ones already stored. See :meth:`upsert`."""
        collection = self._collection(fingerprint)
        ids = [point.id for point in points]
        present: set[PointId] = set()
        for start in range(0, len(ids), RETRIEVE_PAGE):
            found = await self._client.retrieve(
                collection_name=collection,
                ids=list(ids[start : start + RETRIEVE_PAGE]),
                with_payload=False,
                with_vectors=False,
            )
            present.update(point.id for point in found)
        return [point for point in points if point.id not in present]

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

    def _verdict(
        self,
        chunk: Chunk,
        record: dict[str, Any],
        fingerprint: EmbedFingerprint,
    ) -> StoredVector:
        """Classify one stored point against the chunk it is being offered for.

        What the point knows is read here; what it *means* is decided by
        :func:`~manicule.core.embedding.classify_stored_vector`, which every backend shares so
        that two of them cannot answer one question two ways.
        """
        stored = record.get(VECTOR_COLUMN)
        checksum, version = checksum_of(record)
        return classify_stored_vector(
            chunk,
            recorded_identity=str(record.get(IDENTITY_COLUMN) or UNRECORDED_IDENTITY),
            stored_embed_text=embed_text_of(record),
            stored_vector=None if stored is None else [float(value) for value in stored],
            embed=fingerprint,
            middleware=self._middleware,
            recorded_checksum=checksum,
            recorded_checksum_version=version,
        )

    def _filter(self, filter: Filter | None) -> models.Filter | None:  # noqa: A002 - the domain's
        """The Qdrant filter for ``filter``, or ``None`` when nothing pushes down.

        Raises:
            ValueError: When ``filter`` sets a field this store can neither honor nor has been
                granted an exemption for.
        """
        refuse_unhonored_fields(filter)
        if filter is None:
            return None
        conditions: list[models.Condition] = []
        if filter.document_ids:
            conditions.append(_any_of(DOCUMENT_ID_COLUMN, sorted(filter.document_ids)))
        if filter.kinds:
            conditions.append(_any_of(KIND_COLUMN, sorted(kind.value for kind in filter.kinds)))
        if filter.langs:
            conditions.append(_any_of(LANG_COLUMN, sorted(filter.langs)))
        return models.Filter(must=conditions) if conditions else None

    async def _unranked(
        self, fingerprint: EmbedFingerprint, k: int, condition: models.Filter | None
    ) -> list[Candidate]:
        """Candidates for a query the store cannot rank. See :meth:`search`."""
        records, _ = await self._client.scroll(
            collection_name=self._collection(fingerprint),
            scroll_filter=condition,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )
        candidates: list[Candidate] = []
        for point in records:
            record = _record_of(point.payload, point.vector)
            if not row_integrity(record).accepts:
                continue
            candidates.append(
                Candidate(
                    chunk=Chunk.model_validate_json(str(record[CHUNK_COLUMN])),
                    publication_id=str(record.get(PUBLICATION_COLUMN) or LEGACY_PUBLICATION),
                    score=0.0,
                )
            )
        return candidates

    async def _records_matching(
        self,
        fingerprint: EmbedFingerprint,
        field: str,
        values: Sequence[str],
        *,
        with_vectors: bool,
    ) -> list[dict[str, Any]]:
        """Every stored point whose ``field`` is one of ``values``, read in bounded pages.

        Paged because the caller's set is not bounded by anything this method controls: one
        query per document is small, one query per corpus is a filter with a hundred thousand
        terms in it. A value may match several publications' points, so no page carries a limit
        on how many it may return — every match is classified and the strongest wins.
        """
        collection = self._collection(fingerprint)
        records: list[dict[str, Any]] = []
        for start in range(0, len(values), RETRIEVE_PAGE):
            page = models.Filter(must=[_any_of(field, values[start : start + RETRIEVE_PAGE])])
            offset: PointId | None = None
            while True:
                found, offset = await self._client.scroll(
                    collection_name=collection,
                    scroll_filter=page,
                    limit=SCROLL_PAGE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                records.extend(_record_of(point.payload, point.vector) for point in found)
                if offset is None:
                    break
        return records

    async def _scan(
        self, fingerprint: EmbedFingerprint, *, page_size: int, with_vectors: bool
    ) -> AsyncGenerator[dict[str, Any]]:
        """Every stored point, one bounded page at a time."""
        collection = self._collection(fingerprint)
        offset: PointId | None = None
        while True:
            found, offset = await self._client.scroll(
                collection_name=collection,
                limit=max(1, page_size),
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for point in found:
                yield _record_of(point.payload, point.vector)
            if offset is None:
                return

    async def _count_matching(self, collection: str, condition: models.Filter | None) -> int:
        result = await self._client.count(
            collection_name=collection, count_filter=condition, exact=True
        )
        return int(result.count)


_STATE_PRIORITY: Final = {
    VectorState.ABSENT: 0,
    VectorState.STALE: 1,
    VectorState.CORRUPT: 2,
    VectorState.READABLE: 3,
}
"""Best evidence across several physical publications of one logical chunk."""


def _unrecorded(field: str) -> models.Filter:
    """Points whose ``field`` records nothing.

    ``IsEmptyCondition`` covers a missing key and a null; the equality covers the empty string,
    which Qdrant does not call empty and which a writer that cleared a field rather than
    removing it leaves behind. Both are spelled out for the same reason the embedded store
    spells out ``NULL`` beside ``''``: two absences that mean one thing must be asked about
    together, or a count is right about the rows it happened to see.
    """
    return models.Filter(
        should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key=field)),
            models.FieldCondition(key=field, match=models.MatchValue(value="")),
        ]
    )


def _unrecorded_checksum() -> models.Filter:
    """Points that record *neither* half of the numerical-integrity pair.

    Pair-aware rather than checksum-only, and the difference is a point that carries one field
    and not the other. Such a point is not a backfill backlog item — it is a point whose two
    halves were not written together, which
    :func:`~manicule.core.embedding.verify_stored_checksum` calls malformed. Counting it as
    unrecorded would inflate the backfill's number with damage, and — the part that actually
    bites — it would put the point inside the backfill's page, where a freshly computed digest
    over whatever the vector is *now* would erase the contradiction the point was announcing.
    """
    return models.Filter(must=[_unrecorded(CHECKSUM_COLUMN), _unrecorded(CHECKSUM_VERSION_COLUMN)])


def _half_written_checksum() -> models.Filter:
    """Points carrying exactly one half of the pair.

    Malformed by inspection of the payload alone — no vector is read and nothing is hashed —
    which is what lets the counting mode of :meth:`QdrantVectorStore.checksum_coverage` report
    them. Without it a half-written point is invisible to every surface that cannot afford a
    scan, and "complete" would be true over a collection holding one.
    """
    checksum = _unrecorded(CHECKSUM_COLUMN)
    version = _unrecorded(CHECKSUM_VERSION_COLUMN)
    return models.Filter(
        should=[
            models.Filter(must=[checksum], must_not=[version]),
            models.Filter(must=[version], must_not=[checksum]),
        ]
    )


def is_cleartext_remote(endpoint: str) -> bool:
    """Whether ``endpoint`` reaches another machine without TLS.

    Asked only of a client that talks to a server — an in-process Qdrant has no transport to
    secure, and its ``:memory:`` location parses as neither a scheme nor a host, so
    ``endpoint_egress`` would call it remote.

    Reported by :meth:`QdrantVectorStore.health` rather than refused by configuration, and the
    distinction is deliberate. Refusing would break the deployment this backend mostly exists
    for — a Qdrant reached inside a cluster, where the endpoint is routinely plain http on a
    network the operator already controls — and no other endpoint in manicule is scheme-checked
    at all: a connector takes http or https, and a model provider's ``base_url`` is classified
    for *egress* rather than for transport. What is true regardless is that the chunk's text and
    the API key cross in the clear, which an operator should be told rather than left to infer.
    :attr:`~manicule.core.lifecycle.HealthState.DEGRADED` is the state for exactly this: working,
    and not as it should be.
    """
    from manicule.config.providers import endpoint_egress  # noqa: PLC0415 - config is heavier

    if not endpoint_egress(endpoint).leaves_machine:
        return False
    return urlsplit(endpoint).scheme.lower() != "https"


def _any_of(field: str, values: Sequence[str]) -> models.FieldCondition:
    """A membership condition over a payload field."""
    return models.FieldCondition(key=field, match=models.MatchAny(any=list(values)))


def _vector_of(stored: object) -> list[float] | None:
    """The stored vector a point carries, narrowed to float32, or ``None`` if it carries none.

    Named vectors would arrive as a mapping. This store creates collections with a single
    unnamed vector, so a mapping here means the collection was not created by manicule, and
    guessing which of several vectors was meant is not something to do to a corpus.
    """
    if stored is None or isinstance(stored, dict):
        return None
    if not isinstance(stored, (list, tuple)):  # pragma: no cover - defensive
        return None
    values = cast("Sequence[float]", stored)
    try:
        return as_float32([float(value) for value in values])
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _record_of(payload: Mapping[str, Any] | None, stored: object) -> dict[str, Any]:
    """One point, as the flat record every backend-agnostic helper reads.

    The payload already uses the shared field names, so this is the payload plus the vector
    under :data:`~manicule.storage.vector_schema.VECTOR_COLUMN` — which is where a Lance row
    carries it, and therefore where :func:`~manicule.storage.vector_schema.row_integrity` and
    the reuse classification look.
    """
    record = dict(payload or {})
    record[VECTOR_COLUMN] = _vector_of(stored)
    return record


__all__ = [
    "INDEXED_PAYLOAD_FIELDS",
    "META_COLLECTION_SUFFIX",
    "POINT_NAMESPACE",
    "SCROLL_PAGE",
    "UPSERT_BATCH",
    "ApiException",
    "QdrantVectorStore",
    "as_float32",
    "collection_for",
    "is_cleartext_remote",
    "is_local",
    "meta_collection_for",
    "point_id_for",
    "workspace_digest",
]
