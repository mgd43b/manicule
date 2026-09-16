"""What the Qdrant vector store promises, and the guards that keep it honest.

Three failures drive this file, and none of them raises on its own.

Vectors from a second model of the same dimension mix in silently and every later answer is
drawn from a space the query does not live in. A stored vector read back over the REST
transport arrives as a ``float64`` a few parts in 10^9 from the ``float32`` that was persisted,
so a store that believes the readback verbatim reports a healthy corpus as entirely corrupt.
And two installations sharing one server collide on a collection name unless something puts
their scopes in it.

The conformance suites cover what every backend owes at the protocol level; the tests here
cover what this one does with a network, a flat namespace and a payload where a column used to
be. They run against Qdrant's in-process mode, which needs no server — the guarantees under
test are the store's. The suite at the bottom is the one that needs a real one, and says so.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import struct
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest
from qdrant_client import AsyncQdrantClient, models

from manicule.config.settings import Settings
from manicule.container.container import check_wiring
from manicule.container.keys import ComponentKey
from manicule.core.content import LEGACY_PUBLICATION, BlockKind, Chunk
from manicule.core.embedding import (
    VECTOR_CHECKSUM_VERSION,
    VectorIntegrity,
    canonical_stored_vector,
    vector_checksum,
)
from manicule.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    VectorStoreStateError,
)
from manicule.core.ids import vector_id
from manicule.core.lifecycle import HealthState
from manicule.core.protocols import (
    AdoptingVectorStore,
    AnnIndexMaintenance,
    PublicationAwareVectorStore,
    PublicationBoundVectorStore,
    ResettableVectorStore,
    VectorIntegrityMaintenance,
    VectorStore,
)
from manicule.core.retrieval import Filter
from manicule.plugins import BuildContext, ComponentRegistry
from manicule.storage.config import QdrantVectorStoreConfig
from manicule.storage.plugin import PLUGIN, build_qdrant_vector_store
from manicule.storage.qdrant import (
    INDEXED_PAYLOAD_FIELDS,
    CollectionShape,
    QdrantVectorStore,
    as_float32,
    collection_for,
    dials_in_force,
    is_cleartext_remote,
    is_local,
    meta_collection_for,
    owns_collection,
    point_id_for,
    workspace_digest,
)
from manicule.storage.vector_paths import workspace_vector_directory
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
    space_name,
)
from manicule.testing.contracts import (
    assert_protocol_signatures,
    assert_vector_store_adopts_rows_verbatim,
    assert_vector_store_is_dimension_agnostic,
    assert_vector_store_records_vector_checksums,
    assert_vector_store_rejects_foreign_vectors,
    assert_vector_store_reuses_by_embedding_input,
)
from tests.qdrant_support import (
    TEST_COLLECTION_PREFIX,
    local_client,
    remote_client,
    require_server,
)
from tests.storage_helpers import fingerprint, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Callable

WORKSPACE: Final = "workspace-under-test"


class _NoComponents:
    """A resolver that provides nothing, for a factory that asks it for nothing."""

    def get[T](self, key: ComponentKey[T]) -> T:
        raise AssertionError(f"the vector store asked for {key}, which it should not need")


def chunk(chunk_id: str, *, position: int = 0, lang: str | None = "en") -> Chunk:
    """A chunk whose every field the store promotes is set to something distinguishable."""
    document = make_document(source="fs", source_id=f"doc-of-{chunk_id}")
    made = make_chunk(document, position, f"the text of {chunk_id}", lang=lang)
    return made.model_copy(update={"id": chunk_id})


def spread(dimension: int, index: int) -> list[float]:
    """A one-hot vector, so similarity between two of them is decided by ``index``."""
    return [1.0 if position == index % dimension else 0.0 for position in range(dimension)]


def scope(*ids: str) -> Filter:
    """A filter carrying only the workspace scope, which restricts nothing in this store."""
    return Filter(workspace_ids=frozenset({WORKSPACE}), document_ids=frozenset(ids))


@pytest.fixture
def client() -> AsyncQdrantClient:
    """A Qdrant inside this process, holding nothing."""
    return local_client()


@pytest.fixture
def make_store(client: AsyncQdrantClient) -> Callable[[], QdrantVectorStore]:
    """A factory whose stores share a server and share nothing else.

    Each store gets its own workspace, which is how a real server keeps two corpora apart, so
    the isolation the conformance suites rely on is the isolation the product ships rather than
    a fixture arranging one.
    """
    made: list[QdrantVectorStore] = []

    def factory() -> QdrantVectorStore:
        store = QdrantVectorStore(
            client,
            workspace_id=f"{WORKSPACE}-{len(made)}",
            collection_prefix=TEST_COLLECTION_PREFIX,
        )
        made.append(store)
        return store

    return factory


@pytest.fixture
def store(client: AsyncQdrantClient) -> QdrantVectorStore:
    """A store on an empty server that has never been prepared."""
    return QdrantVectorStore(
        client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX
    )


async def prepared(store: QdrantVectorStore, dimension: int = 4) -> QdrantVectorStore:
    """A store that has been through ``ensure_ready`` at ``dimension``."""
    await store.ensure_ready(fingerprint(dimension))
    return store


# --- conformance ---------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_store_satisfies_the_vector_store_protocol(store: QdrantVectorStore) -> None:
    """Structural conformance, including the part ``isinstance`` deliberately does not check.

    ``@runtime_checkable`` checks the attributes exist and nothing about what they accept, so a
    store whose ``stored_vectors`` took ``chunk_ids`` where the protocol says ``chunks`` would
    pass every ``isinstance`` in the codebase and fail at the first keyword call — which, for a
    method reached once per document, is somewhere in the middle of a corpus sweep.
    """
    assert isinstance(store, VectorStore)
    assert_protocol_signatures(store, VectorStore)


@pytest.mark.contract
async def test_the_store_works_at_whatever_dimension_the_embedder_reports(
    make_store: Callable[[], QdrantVectorStore],
) -> None:
    """A hardcoded dimension anywhere — collection config, buffer, assertion — fails one."""
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await assert_vector_store_is_dimension_agnostic(make_store, chunks)


@pytest.mark.contract
async def test_the_store_refuses_vectors_from_a_second_model_of_the_same_size(
    make_store: Callable[[], QdrantVectorStore],
) -> None:
    """A size check passes this case, and every answer afterwards is quietly meaningless."""
    await assert_vector_store_rejects_foreign_vectors(make_store)


@pytest.mark.contract
async def test_the_store_records_a_checksum_over_what_it_persists(
    make_store: Callable[[], QdrantVectorStore],
) -> None:
    """Hashing the caller's argument passes the one-hot case and fails a real embedder's."""
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await assert_vector_store_records_vector_checksums(make_store, chunks)


@pytest.mark.contract
async def test_the_store_carries_an_adopted_row_rather_than_rebuilding_it(
    make_store: Callable[[], QdrantVectorStore],
) -> None:
    """Rebuilding the row from its chunk certifies a damaged vector as verified."""
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await assert_vector_store_adopts_rows_verbatim(make_store, chunks)


async def test_adoption_refuses_a_row_whose_chunk_did_not_come_with_it(
    store: QdrantVectorStore,
) -> None:
    """A point without its chunk is a point every search that ranks it fails on.

    The chunk travels with the vector because ``search`` returns one with no database behind
    it, so storing the point anyway does not lose a little context — it plants a row that
    raises a validation error in whichever process ranks it next, long after the migration that
    wrote it reported success. Refused where the id and the vector beside it are refused.
    """
    await store.ensure_ready(fingerprint(4))
    row = {
        "id": "row-1",
        "chunk_id": "chunk-1",
        "publication_id": "legacy",
        "document_id": "doc-1",
        "kind": "paragraph",
        "lang": "en",
        "position": 0,
        "chunk_json": "",
        "embed_identity": "identity",
        "vector_checksum": "unrecorded",
        "vector_checksum_version": "unrecorded",
        "vector": [1.0, 0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match="no chunk beside its vector"):
        await store.adopt_rows([row])

    assert await store.count() == 0, "a refused row must not be half-written"


@pytest.mark.contract
async def test_the_store_satisfies_the_adopting_protocol(store: QdrantVectorStore) -> None:
    """The capability a migration asks of the object rather than of the module.

    Signatures as well as attributes, for the reason the ``VectorStore`` check above gives: a
    migration resolves this with ``isinstance`` and then calls it, so a store whose
    ``adopt_rows`` took something other than a sequence of rows would pass the resolution and
    fail partway through a corpus.
    """
    assert isinstance(store, AdoptingVectorStore)
    assert_protocol_signatures(store, AdoptingVectorStore)


@pytest.mark.contract
async def test_the_store_answers_reuse_on_the_embedding_input(
    make_store: Callable[[], QdrantVectorStore],
) -> None:
    """Reuse keyed on the chunk id re-embeds a document that moved nothing the model sees."""
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await assert_vector_store_reuses_by_embedding_input(make_store, chunks)


def test_the_store_declines_the_capabilities_it_does_not_have(store: QdrantVectorStore) -> None:
    """Qdrant maintains its own index, and says so by not claiming the maintenance protocol.

    An ``ann_index_state`` that reported an IVF-PQ lifecycle for an HNSW index this project
    neither built nor can replace would be describing a mechanism that is not there — and every
    surface reads that report as measured. Absence is the honest answer, and the one the
    protocol was written to allow.
    """
    assert not isinstance(store, AnnIndexMaintenance)
    assert isinstance(store, VectorIntegrityMaintenance)


def test_the_store_does_not_claim_the_publication_capabilities(store: QdrantVectorStore) -> None:
    """The negative half of the check the runtime makes on every start, asserted here.

    ``manicule.app.runtime`` decides whether a store wants the publication-following wrapper,
    whether a durable re-embed can run, and whether a derived reset may proceed, by asking
    whether the store satisfies these two protocols. It asks the object rather than importing
    LanceDB's classes to ``isinstance`` against, because that import is ``SIGILL`` on a CPU
    without AVX2 — which made this backend unusable on the hardware it exists to serve.

    So a false here is load-bearing. ``teardown`` is the trap: this store has one, and a
    protocol shaped around it would match, and the runtime would then close a client the
    container had already torn down. What actually separates the two is the publication surface,
    which this store has none of — ``docs/storage.md`` §6.7 is why it has none.
    """
    assert not isinstance(store, PublicationAwareVectorStore)
    assert not isinstance(store, PublicationBoundVectorStore)


@pytest.mark.contract
def test_the_store_claims_the_capability_a_derived_reset_needs(store: QdrantVectorStore) -> None:
    """The one capability that has to be claimed, because nothing else can supply it.

    A derived reset deletes the rows the relational store tombstoned, which needs no capability
    at all, and then has to discard what is around them. On a directory backend the runtime
    removes the directory itself; here it holds a client, so the only thing that can drop a
    collection — or the fingerprint record that would otherwise refuse the next model — is the
    store. Not claiming this is how ``reset-index`` came to have no working form on this backend
    (#377).
    """
    assert isinstance(store, ResettableVectorStore)
    assert_protocol_signatures(store, ResettableVectorStore)


# --- naming and identity -------------------------------------------------------------------


def test_two_workspaces_do_not_share_a_collection() -> None:
    """The isolation a directory gives the embedded store has to be in the name here."""
    embed = fingerprint(4)
    one = collection_for(TEST_COLLECTION_PREFIX, "alpha", embed)
    two = collection_for(TEST_COLLECTION_PREFIX, "beta", embed)

    assert one != two
    assert one.startswith(f"{TEST_COLLECTION_PREFIX}_")
    assert one.endswith(space_name(embed))


def test_the_workspace_digest_is_the_one_the_embedded_store_uses() -> None:
    """Two backends naming one workspace two ways is a relationship nothing else would catch.

    The embedded store names a workspace's directory with the full SHA-256; a collection name is
    read by operators, so this takes a prefix of the same digest rather than a different hash of
    the same string. Pinned here because the two are derived in different modules and only this
    asserts they agree.
    """
    expected = hashlib.sha256(WORKSPACE.encode("utf-8")).hexdigest()

    assert workspace_digest(WORKSPACE) == expected[:16]
    assert workspace_vector_directory(Path("/root"), WORKSPACE).name == expected


def test_two_models_do_not_share_a_collection() -> None:
    """Two 8-dimension models pass every size check and must still not meet."""
    one = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(8, model_id="a"))
    two = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(8, model_id="b"))

    assert one != two


def test_a_point_id_is_a_uuid_derived_from_the_row_id() -> None:
    """Qdrant takes an integer or a UUID, and a row id is neither.

    Deterministic is the whole requirement: the same row written twice has to land on the same
    point, or a retry duplicates rather than overwrites.
    """
    first = point_id_for("chunks-and-such/legacy")
    assert first == point_id_for("chunks-and-such/legacy")
    assert uuid.UUID(first).version == 5
    assert first != point_id_for("chunks-and-such/other")


def test_every_field_a_filter_can_name_is_indexed() -> None:
    """A filter on an unindexed payload field is a full scan that still returns the right rows.

    The failure this catches is silent by construction: adding a filter and forgetting its
    index costs latency and nothing else, so nothing goes red and the corpus simply gets slower
    as it grows.
    """
    indexed = {field for field, _ in INDEXED_PAYLOAD_FIELDS}

    assert {DOCUMENT_ID_COLUMN, KIND_COLUMN, LANG_COLUMN} <= indexed, "search filters on these"
    assert {CHUNK_ID_COLUMN, IDENTITY_COLUMN} <= indexed, "stored_vectors looks these up"
    assert CHECKSUM_COLUMN in indexed, "the backfill selects on this"


def test_local_mode_is_recognized_without_reaching_for_a_private_attribute(
    client: AsyncQdrantClient,
) -> None:
    """Payload indexes warn in local mode, and this project turns warnings into errors."""
    assert is_local(client)
    served = AsyncQdrantClient(url="http://qdrant.invalid:6333", check_compatibility=False)
    assert not is_local(served)


# --- what the store stores -------------------------------------------------------------------


async def test_the_payload_carries_every_field_the_shared_schema_names(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """A Lance row's columns and a Qdrant point's payload are one schema, not two."""
    await prepared(store)
    stored = chunk("chunk-one", lang="en")
    await store.upsert([stored], [spread(4, 0)])

    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    found = await client.retrieve(collection, ids=[point_id_for(stored.id)], with_payload=True)
    payload = found[0].payload or {}

    assert payload[ID_COLUMN] == stored.id
    assert payload[CHUNK_ID_COLUMN] == stored.id
    assert payload[DOCUMENT_ID_COLUMN] == stored.document_id
    assert payload[KIND_COLUMN] == stored.kind.value
    assert payload[LANG_COLUMN] == "en"
    assert payload[CHECKSUM_VERSION_COLUMN] == VECTOR_CHECKSUM_VERSION
    assert Chunk.model_validate_json(str(payload[CHUNK_COLUMN])).id == stored.id


async def test_the_fingerprint_is_recorded_once_per_workspace(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """A collection that cannot say which model made it is a collection nothing can verify."""
    await prepared(store)

    assert await client.collection_exists(meta_collection_for(TEST_COLLECTION_PREFIX))
    recorded = await store.fingerprint()
    assert recorded is not None
    assert recorded.matches(fingerprint(4))


async def test_a_meta_point_that_contradicts_itself_is_refused(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """Half a record is evidence; picking a winner from it is how a corpus acquires a lie."""
    await prepared(store)
    meta = meta_collection_for(TEST_COLLECTION_PREFIX)
    point = (await client.scroll(meta, limit=1, with_payload=True))[0][0]
    await client.set_payload(meta, payload={"canonical": "not-what-it-says"}, points=[point.id])

    with pytest.raises(VectorStoreStateError, match="contradicts itself"):
        await store.fingerprint()


async def test_a_meta_point_that_names_no_fingerprint_is_refused(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """A collection reached over a network can be written to by anything holding the key.

    A point with no fingerprint at all describes an index nothing can identify. Without a typed
    refusal the JSON parse raises instead, so `fingerprint`, `count`, `delete_document` and
    `stored_vectors` would each fail with a pydantic error about a field nobody in the caller's
    stack has heard of.
    """
    await prepared(store)
    meta = meta_collection_for(TEST_COLLECTION_PREFIX)
    point = (await client.scroll(meta, limit=1, with_payload=True))[0][0]
    await client.delete_payload(meta, keys=["embed_fingerprint"], points=[point.id], wait=True)

    with pytest.raises(VectorStoreStateError, match="names no fingerprint"):
        await store.fingerprint()


async def test_a_point_carrying_half_the_checksum_pair_is_not_ranked(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """Coverage calls it malformed, so search must not rank it.

    A payload is per-point, so unlike a table's columns it can carry the version and not the
    digest. Reading only the digest would let such a point through as merely unverified — and
    the count that reports it as malformed would then disagree with the search that ranked it.
    """
    await prepared(store)
    good, half = chunk("good"), chunk("half", position=1)
    await store.upsert([good, half], [spread(4, 0), spread(4, 1)])
    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.delete_payload(
        collection, keys=[CHECKSUM_COLUMN], points=[point_id_for(half.id)], wait=True
    )

    ranked = await store.search(spread(4, 1), k=5)
    coverage = await store.checksum_coverage()

    assert [result.chunk.id for result in ranked] == ["good"]
    assert coverage.failures == {VectorIntegrity.MALFORMED.value: 1}


async def test_preparing_twice_is_not_a_second_collection(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """``ensure_ready`` runs on every ingest, so it has to be idempotent rather than careful."""
    await prepared(store)
    await store.upsert([chunk("chunk-one")], [spread(4, 0)])
    await prepared(store)

    assert await store.count() == 1
    collections = {entry.name for entry in (await client.get_collections()).collections}
    assert len(collections) == 2, collections


async def test_a_collection_left_without_its_fingerprint_is_completed(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """The interruption the two-step creation is ordered for.

    The collection is made first, so a crash between the steps leaves one with no fingerprint
    beside it — which the next call finishes. The other order would leave a fingerprint naming
    a collection that does not exist, and every read would then fail against a store that
    reported itself ready.
    """
    await prepared(store)
    await client.delete_collection(meta_collection_for(TEST_COLLECTION_PREFIX))
    assert await store.fingerprint() is None

    await store.ensure_ready(fingerprint(4))

    recorded = await store.fingerprint()
    assert recorded is not None
    assert recorded.matches(fingerprint(4))


async def test_a_second_model_is_refused_before_anything_is_written(
    store: QdrantVectorStore,
) -> None:
    """Comparing before creating is what stops a refusal leaving a collection behind."""
    await prepared(store, dimension=8)

    with pytest.raises(FingerprintMismatchError):
        await store.ensure_ready(fingerprint(8, model_id="someone/else"))


# --- writing -----------------------------------------------------------------------------


async def test_an_unprepared_store_refuses_to_write_or_search(store: QdrantVectorStore) -> None:
    """Guessing the fingerprint is exactly the mistake the recorded one exists to prevent."""
    with pytest.raises(VectorStoreStateError, match="ensure_ready"):
        await store.upsert([chunk("chunk-one")], [spread(4, 0)])
    with pytest.raises(VectorStoreStateError, match="ensure_ready"):
        await store.search(spread(4, 0), k=1)


async def test_parallel_sequences_of_different_lengths_are_refused(
    store: QdrantVectorStore,
) -> None:
    """Storing the overlap would key some vectors to the wrong text, silently."""
    await prepared(store)

    with pytest.raises(ValueError, match="parallel"):
        await store.upsert([chunk("one"), chunk("two", position=1)], [spread(4, 0)])


async def test_a_vector_of_the_wrong_dimension_is_refused(store: QdrantVectorStore) -> None:
    """A disagreement here means two embedders are in play."""
    await prepared(store)

    with pytest.raises(ValueError, match="dimension"):
        await store.upsert([chunk("one")], [spread(8, 0)])


async def test_a_non_finite_vector_is_refused_before_storage(store: QdrantVectorStore) -> None:
    """NaN cannot participate in cosine distance, and a stored one poisons every ranking."""
    await prepared(store)

    with pytest.raises(ValueError, match="non-finite"):
        await store.upsert([chunk("one")], [[float("nan"), 0.0, 0.0, 0.0]])


async def test_a_legacy_row_is_overwritten_and_a_published_one_is_not(
    store: QdrantVectorStore,
) -> None:
    """The immutability a content-addressed publication promises, kept by a read.

    Lance gets this from ``merge_insert`` with only ``when_not_matched_insert_all``. Qdrant's
    upsert is an unconditional write, so a replayed checkpoint would otherwise rewrite rows a
    validation pass has already accepted.
    """
    await prepared(store)
    stored = chunk("chunk-one")

    await store.upsert([stored], [spread(4, 0)])
    await store.upsert([stored], [spread(4, 1)])
    reused = (await store.stored_vectors([stored]))[stored.id]
    assert list(reused.vector) == spread(4, 1), "the legacy generation is the mutable one"

    await store.upsert([stored], [spread(4, 2)], publication_id="publication-a")
    await store.upsert([stored], [spread(4, 3)], publication_id="publication-a")

    published = [
        candidate
        for candidate in await store.search(spread(4, 2), k=5)
        if candidate.publication_id == "publication-a"
    ]
    assert len(published) == 1
    assert published[0].score == pytest.approx(1.0), "the replay did not overwrite the first"


async def test_an_empty_upsert_writes_nothing(store: QdrantVectorStore) -> None:
    """A batch with nothing in it is a request not to make one."""
    await prepared(store)
    await store.upsert([], [])

    assert await store.count() == 0


# --- reading -----------------------------------------------------------------------------


async def test_search_scores_are_cosine_similarities(store: QdrantVectorStore) -> None:
    """Stored vectors are unit length and the metric is cosine, so the score is the similarity.

    A backend returning a *distance* here would rank correctly and report every number as its
    own complement, which no test of ordering alone would catch.
    """
    await prepared(store)
    near, far = chunk("near"), chunk("far", position=1)
    await store.upsert([near, far], [spread(4, 0), spread(4, 1)])

    results = await store.search(spread(4, 0), k=2)

    assert [result.chunk.id for result in results] == ["near", "far"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)


async def test_a_query_with_no_direction_returns_rows_rather_than_nothing(
    store: QdrantVectorStore,
) -> None:
    """Cosine similarity against the zero vector is undefined for every row.

    Ranking it would be inventing an order, and returning nothing would read as "the corpus is
    empty" — a different and false claim.
    """
    await prepared(store)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(4, index) for index in range(3)])

    results = await store.search([0.0, 0.0, 0.0, 0.0], k=2)

    assert len(results) == 2
    assert all(result.score == 0.0 for result in results)


async def test_a_search_for_nothing_returns_nothing(store: QdrantVectorStore) -> None:
    """``k`` of zero is a question with no answer, not a question about everything."""
    await prepared(store)
    await store.upsert([chunk("one")], [spread(4, 0)])

    assert await store.search(spread(4, 0), k=0) == []


async def test_a_query_of_the_wrong_dimension_is_refused(store: QdrantVectorStore) -> None:
    """A query in another space returns a confident ranking of unrelated rows."""
    await prepared(store)

    with pytest.raises(ValueError, match="dimension"):
        await store.search(spread(8, 0), k=1)


async def test_the_filter_pushes_down_what_it_can_and_refuses_the_rest(
    store: QdrantVectorStore,
) -> None:
    """Quietly dropping a restriction returns results the filter was written to exclude."""
    await prepared(store)
    one, two = chunk("one"), chunk("two", position=1)
    await store.upsert([one, two], [spread(4, 0), spread(4, 1)])

    restricted = await store.search(spread(4, 0), k=5, filter=scope(one.document_id))
    assert [result.chunk.id for result in restricted] == ["one"]

    with pytest.raises(ValueError, match="collection_ids"):
        await store.search(
            spread(4, 0),
            k=5,
            filter=Filter(
                workspace_ids=frozenset({WORKSPACE}), collection_ids=frozenset({"anything"})
            ),
        )


async def test_a_workspace_scope_alone_restricts_nothing_here(store: QdrantVectorStore) -> None:
    """The boundary moved to the hydrating join rather than disappearing; this store is exempt."""
    await prepared(store)
    await store.upsert([chunk("one")], [spread(4, 0)])

    results = await store.search(
        spread(4, 0), k=5, filter=Filter(workspace_ids=frozenset({"someone-else"}))
    )

    assert [result.chunk.id for result in results] == ["one"]


async def test_a_filter_on_kind_and_lang_pushes_down(store: QdrantVectorStore) -> None:
    """Both are promoted fields, so neither needs a round trip through the document store."""
    await prepared(store)
    prose = chunk("prose", lang="en")
    table = chunk("table", position=1, lang="fr").model_copy(update={"kind": BlockKind.TABLE})
    await store.upsert([prose, table], [spread(4, 0), spread(4, 1)])

    by_kind = await store.search(
        spread(4, 0),
        k=5,
        filter=Filter(workspace_ids=frozenset({WORKSPACE}), kinds=frozenset({BlockKind.TABLE})),
    )
    by_lang = await store.search(
        spread(4, 0),
        k=5,
        filter=Filter(workspace_ids=frozenset({WORKSPACE}), langs=frozenset({"en"})),
    )

    assert [result.chunk.id for result in by_kind] == ["table"]
    assert [result.chunk.id for result in by_lang] == ["prose"]


async def test_a_corrupted_row_is_dropped_from_a_ranking(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """A score computed against corrupted numbers is a result nothing vouches for."""
    await prepared(store)
    good, bad = chunk("good"), chunk("bad", position=1)
    await store.upsert([good, bad], [spread(4, 0), spread(4, 1)])

    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.set_payload(
        collection, payload={CHECKSUM_COLUMN: "0" * 64}, points=[point_id_for(bad.id)]
    )

    results = await store.search(spread(4, 1), k=5)

    assert [result.chunk.id for result in results] == ["good"]


# --- deleting ----------------------------------------------------------------------------


async def test_deleting_a_document_removes_every_vector_it_owns(
    store: QdrantVectorStore,
) -> None:
    """Idempotent, and scoped to one document: a sweep runs this against a live corpus."""
    await prepared(store)
    one, two = chunk("one"), chunk("two", position=1)
    await store.upsert([one, two], [spread(4, 0), spread(4, 1)])

    await store.delete_document(one.document_id)
    await store.delete_document(one.document_id)

    assert await store.count() == 1


async def test_deleting_chunks_takes_row_ids(store: QdrantVectorStore) -> None:
    """The sweep holds ``StoredVector.vector_id`` and means exactly the row it looked at."""
    await prepared(store)
    one, two = chunk("one"), chunk("two", position=1)
    await store.upsert([one, two], [spread(4, 0), spread(4, 1)])
    await store.delete_chunks([vector_id(LEGACY_PUBLICATION, one.id)])
    await store.delete_chunks([vector_id(LEGACY_PUBLICATION, one.id)])

    assert await store.count() == 1


async def test_deleting_from_a_store_that_holds_nothing_is_not_an_error(
    store: QdrantVectorStore,
) -> None:
    """A sweep runs before the first ingest as readily as after it."""
    await store.delete_document("no-such-document")
    await store.delete_chunks(["no-such-row"])

    assert await store.count() == 0


# --- resetting -----------------------------------------------------------------------------


async def test_a_reset_drops_this_workspace_s_collection_and_forgets_its_model(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """Both halves, because either one alone leaves a workspace that cannot be re-indexed.

    Dropping the collection and keeping the record leaves the next ``ensure_ready`` comparing
    against a model whose vectors are gone; keeping the collection and dropping the record
    leaves rows from a model nothing will admit again.
    """
    await prepared(store)
    await store.upsert([chunk("one")], [spread(4, 0)])
    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))

    assert await store.reset_storage() is True

    assert not await client.collection_exists(collection)
    assert await store.fingerprint() is None
    assert await store.count() == 0


async def test_a_reset_lets_the_next_ingest_arrive_from_a_different_model(
    store: QdrantVectorStore,
) -> None:
    """What a reset is *for*, stated as the thing an operator does the day after.

    The recorded fingerprint is the part a row sweep cannot reach, and until #377 nothing else
    reached it either: every derived reset on this backend refused, so the record survived and
    re-indexing under a new embedder was refused by an index that no longer held anything. That
    is the state this asserts an installation can now get out of without a throwaway pod.
    """
    await prepared(store)
    await store.upsert([chunk("one")], [spread(4, 0)])
    await store.reset_storage()

    await store.ensure_ready(fingerprint(8))
    await store.upsert([chunk("two")], [spread(8, 1)])

    assert await store.count() == 1


async def test_a_reset_leaves_every_other_workspace_on_the_server_alone(
    make_store: Callable[[], QdrantVectorStore],
) -> None:
    """One flat namespace, several corpora, and a reset asked about exactly one of them.

    An installation shares a server the way it never shares a directory, so the blast radius of
    a reset is a property of the name it matches on rather than of the storage it is deleting.
    """
    reset, kept = make_store(), make_store()
    for held in (reset, kept):
        await prepared(held)
        await held.upsert([chunk(f"row-of-{held.workspace_id}")], [spread(4, 0)])

    assert await reset.reset_storage() is True

    assert await kept.count() == 1
    assert await kept.fingerprint() is not None, "another workspace lost its recorded model"


async def test_a_reset_leaves_a_second_installation_whose_prefix_contains_this_one(
    client: AsyncQdrantClient,
) -> None:
    """``collection_prefix`` is free text, and one installation's can contain another's.

    ``docs/deployment.md`` §6.5 asks operators to give each installation its own prefix on a
    shared server, and the cost of ignoring it has to stay what that section says it is — two
    corpora that cannot see each other's rows — rather than one installation deleting the
    other's. A prefix of ``<ours>_<our workspace digest>`` puts every collection the second
    installation owns behind this one's ownership prefix, so a reset matching on the opening of
    a name would take a stranger's corpus with it.

    The second installation is built to be the worst case rather than a plausible one: nobody
    types a workspace digest into their configuration on purpose, and a rule that only holds
    for names nobody would choose is not a rule.
    """
    ours = QdrantVectorStore(
        client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX
    )
    theirs = QdrantVectorStore(
        client,
        workspace_id="a-different-installation",
        collection_prefix=f"{TEST_COLLECTION_PREFIX}_{workspace_digest(WORKSPACE)}",
    )
    for store in (ours, theirs):
        await prepared(store)
        await store.upsert([chunk(f"row-of-{store.workspace_id}")], [spread(4, 0)])
    stranger = collection_for(
        f"{TEST_COLLECTION_PREFIX}_{workspace_digest(WORKSPACE)}",
        "a-different-installation",
        fingerprint(4),
    )
    assert stranger.startswith(f"{TEST_COLLECTION_PREFIX}_{workspace_digest(WORKSPACE)}_"), (
        "this name must open with our ownership prefix, or the test proves nothing"
    )
    assert not owns_collection(TEST_COLLECTION_PREFIX, WORKSPACE, stranger)

    assert await ours.reset_storage() is True

    assert await client.collection_exists(stranger)
    assert await theirs.count() == 1
    assert await theirs.fingerprint() is not None


async def test_a_reset_reaches_a_collection_the_record_does_not_name(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """Collections are matched by name, not resolved through the meta record, and this is why.

    A workspace that has held two embedding spaces has two collections and a record naming one
    of them. Resolving through the record would drop the collection the reset could already see
    and leave the one nothing would go looking for again — a corpus's worth of vectors from a
    model this installation no longer runs, invisible to every count it reports.
    """
    await prepared(store)
    orphan = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(8))
    await client.create_collection(
        collection_name=orphan,
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )

    assert await store.reset_storage() is True

    assert not await client.collection_exists(orphan)


async def test_resetting_a_workspace_that_never_held_a_vector_removes_nothing(
    store: QdrantVectorStore,
) -> None:
    """A reset runs on an installation whose state nobody is sure of, and says what it found.

    ``False`` is the honest answer rather than a failure: there was nothing to remove. Running
    it twice has to give the same answer for the same reason.
    """
    assert await store.reset_storage() is False

    await prepared(store)
    assert await store.reset_storage() is True
    assert await store.reset_storage() is False


# --- numerical integrity -------------------------------------------------------------------


async def test_coverage_counts_what_it_says_it_counted(store: QdrantVectorStore) -> None:
    """A cheap count and a recomputed digest establish different things and say which."""
    await prepared(store)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(4, index) for index in range(3)])

    counted = await store.checksum_coverage()
    assert counted.scanned
    assert counted.rows == 3
    assert counted.recorded == 3
    assert not counted.recomputed
    assert counted.verified == 0

    recomputed = await store.checksum_coverage(recompute=True)
    assert recomputed.recomputed
    assert recomputed.verified == 3
    assert recomputed.failed == 0
    assert recomputed.complete


async def test_coverage_over_a_store_that_holds_nothing_says_it_looked_at_nothing(
    store: QdrantVectorStore,
) -> None:
    """ "Nothing is wrong" and "nothing was looked at" are the two answers worth separating."""
    coverage = await store.checksum_coverage()

    assert not coverage.scanned
    assert coverage.rows == 0


async def test_a_damaged_vector_is_counted_as_a_refusal(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """The bit flip a provenance check cannot see: finite, in range, and not what was written."""
    await prepared(store)
    stored = chunk("chunk-one")
    await store.upsert([stored], [spread(4, 0)])
    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.upsert(
        collection,
        points=[
            models.PointStruct(
                id=point_id_for(stored.id),
                vector=[0.0, 1.0, 0.0, 0.0],
                payload=(
                    await client.retrieve(
                        collection, ids=[point_id_for(stored.id)], with_payload=True
                    )
                )[0].payload,
            )
        ],
        wait=True,
    )

    coverage = await store.checksum_coverage(recompute=True)

    assert coverage.failed == 1
    assert coverage.failures == {VectorIntegrity.MISMATCHED.value: 1}


async def test_the_backfill_gives_an_unrecorded_row_the_checksum_its_vector_implies(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """A network collection is shared infrastructure; a point can arrive without a checksum."""
    await prepared(store)
    stored = chunk("chunk-one")
    await store.upsert([stored], [spread(4, 0)])
    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.delete_payload(
        collection,
        keys=[CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN],
        points=[point_id_for(stored.id)],
        wait=True,
    )
    assert (await store.checksum_coverage()).recorded == 0

    planned = await store.backfill_checksums(dry_run=True)
    assert planned.dry_run
    assert planned.scanned == 1
    assert planned.written == 1, "a dry run reports what a pass would write"
    assert planned.remaining == 1
    assert (await store.checksum_coverage()).recorded == 0, "and writes none of it"

    done = await store.backfill_checksums()

    assert done.written == 1
    assert done.remaining == 0
    assert done.done
    assert (await store.checksum_coverage(recompute=True)).verified == 1


async def test_the_backfill_never_selects_a_half_written_pair(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """Half a record is evidence, and hashing it would replace the contradiction with a pass.

    A point carrying a checksum and no version, or the reverse, is malformed rather than
    unrecorded. The backfill must leave it announcing that, and the cheap coverage count must
    be able to see it without reading a single vector.
    """
    await prepared(store)
    stored = chunk("chunk-one")
    await store.upsert([stored], [spread(4, 0)])
    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.delete_payload(
        collection,
        keys=[CHECKSUM_VERSION_COLUMN],
        points=[point_id_for(stored.id)],
        wait=True,
    )

    counted = await store.checksum_coverage()
    done = await store.backfill_checksums()

    assert counted.failed == 1
    assert counted.failures == {VectorIntegrity.MALFORMED.value: 1}
    assert not counted.complete
    assert done.scanned == 0
    assert done.written == 0


async def test_a_checksum_cleared_to_an_empty_string_counts_as_unrecorded(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    """Qdrant does not call an empty string empty, and a writer that cleared a field leaves one.

    Two absences that mean one thing have to be asked about together, or the backfill finishes
    while a point it should have written still records nothing — and coverage calls the
    collection complete on the strength of a field holding no digest at all.
    """
    await prepared(store)
    stored = chunk("chunk-one")
    await store.upsert([stored], [spread(4, 0)])
    collection = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.set_payload(
        collection,
        payload={CHECKSUM_COLUMN: "", CHECKSUM_VERSION_COLUMN: ""},
        points=[point_id_for(stored.id)],
        wait=True,
    )

    counted = await store.checksum_coverage()
    done = await store.backfill_checksums()

    assert counted.recorded == 0
    assert counted.unverified == 1
    assert done.written == 1
    assert (await store.checksum_coverage(recompute=True)).verified == 1


# --- the network ---------------------------------------------------------------------------


def test_a_readback_is_narrowed_to_the_float32_that_was_stored() -> None:
    """The REST transport's rounding, undone exactly and no further.

    A ``float32`` serialized as JSON decimal text and parsed back is a ``float64`` a few parts
    in 10^9 away. Believing it verbatim makes every recomputed digest disagree, which reads as a
    corpus that is entirely corrupt rather than as a transport that rounds.
    """
    persisted = struct.unpack("!f", struct.pack("!f", 0.1234567890123))[0]
    as_json_would_parse = float(f"{persisted:.9g}")

    assert as_json_would_parse != persisted
    assert as_float32([as_json_would_parse]) == [persisted]
    assert as_float32([persisted]) == [persisted], "gRPC already delivers float32"


def test_narrowing_is_not_canonicalization() -> None:
    """Re-normalizing a readback would repair the drift the checksum exists to notice."""
    drifted = [2.0, 0.0, 0.0, 0.0]

    assert as_float32(drifted) == drifted
    assert list(canonical_stored_vector(drifted)) == [1.0, 0.0, 0.0, 0.0]


async def test_health_reports_the_server_it_could_not_reach(client: AsyncQdrantClient) -> None:
    """The check a local directory never needs and a network index always does."""
    reachable = QdrantVectorStore(
        client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX
    )
    assert (await reachable.health()).state is HealthState.OK

    unreachable = QdrantVectorStore(
        AsyncQdrantClient(url="http://127.0.0.1:1", timeout=1, check_compatibility=False),
        workspace_id=WORKSPACE,
        collection_prefix=TEST_COLLECTION_PREFIX,
    )

    report = await unreachable.health()

    assert report.state is HealthState.FAILING
    assert "127.0.0.1:1" in report.detail
    assert "storage.vector_db_url" in report.remedy


async def test_health_says_so_when_a_remote_endpoint_is_cleartext() -> None:
    """Reported rather than refused: an in-cluster http endpoint is the normal case.

    Refusing would break the deployment this backend mostly exists for, and no other endpoint
    in manicule is scheme-checked. What is true either way is that the chunk's text and the API
    key cross in the clear, which an operator should be told rather than left to infer.
    """
    assert is_cleartext_remote("http://qdrant.internal:6333")
    assert not is_cleartext_remote("https://qdrant.internal:6333")
    assert not is_cleartext_remote("http://127.0.0.1:6333"), "loopback leaves no machine"
    assert not is_cleartext_remote("http://localhost:6333")


async def test_teardown_closes_only_a_client_the_store_owns(
    client: AsyncQdrantClient,
) -> None:
    """The container tears every component down; a borrowed client is not this store's to close."""
    borrowed = QdrantVectorStore(
        client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX
    )
    await borrowed.teardown()
    await borrowed.ensure_ready(fingerprint(4))  # the borrowed client is still open

    owned = QdrantVectorStore(
        local_client(),
        workspace_id=WORKSPACE,
        collection_prefix=TEST_COLLECTION_PREFIX,
        owns_client=True,
    )
    await owned.teardown()
    await owned.teardown()


# --- wiring ------------------------------------------------------------------------------


def test_configuration_can_select_this_store_and_the_registry_provides_it() -> None:
    """Selecting a component nothing installed provides is the failure `check_wiring` reports.

    Both halves matter and only the second used to hold: the settings type has to accept the
    name, and the registry has to have something under it. A `Literal` that refused the value
    would fail before the registry was ever consulted.
    """
    settings = Settings.model_validate(
        {"storage": {"vector_db": "qdrant", "vector_db_url": "http://127.0.0.1:6333"}}
    )
    registry = ComponentRegistry()
    PLUGIN.register(registry)

    assert not [problem for problem in check_wiring(settings, registry) if "vector_db" in problem]


async def test_the_factory_builds_a_store_without_dialing_anything() -> None:
    """`manicule doctor` builds every component, and must not stall on the one it diagnoses.

    The client's own version check runs a blocking request from inside its constructor and
    warns out of a background thread when nothing answers, so this asserts construction against
    an address that refuses connections and expects neither a delay nor a warning.
    """
    settings = Settings.model_validate(
        {"storage": {"vector_db": "qdrant", "vector_db_url": "http://127.0.0.1:1"}}
    )
    built = build_qdrant_vector_store(
        BuildContext(
            settings=settings,
            config=QdrantVectorStoreConfig(),
            data_dir=settings.data_dir,
            cache_dir=settings.cache_dir,
            components=_NoComponents(),
        )
    )

    assert isinstance(built, QdrantVectorStore)
    await built.teardown()


def _built(settings: Settings) -> QdrantVectorStore:
    built = build_qdrant_vector_store(
        BuildContext(
            settings=settings,
            config=QdrantVectorStoreConfig(),
            data_dir=settings.data_dir,
            cache_dir=settings.cache_dir,
            components=_NoComponents(),
        )
    )
    assert isinstance(built, QdrantVectorStore)
    return built


async def test_the_factory_hands_every_configured_dial_to_the_store() -> None:
    """A dial the factory forgets to copy is a setting that reads as in force and is not.

    Every dial is set away from its default, so a missing copy leaves a default where the
    configured value should be; and the names are matched rather than listed, so a dial added
    to the shape without a setting of the same name fails here instead of in a review.
    """
    tuned = {
        "quantization": "scalar",
        "quantization_always_ram": False,
        "on_disk_vectors": True,
        "on_disk_payload": False,
        "hnsw_m": 32,
        "hnsw_ef_construct": 256,
        "indexing_threshold_kb": 0,
    }
    assert set(tuned) == {dial.name for dial in dataclasses.fields(CollectionShape)}
    settings = Settings.model_validate(
        {"storage": {"vector_db": "qdrant", "vector_db_url": "http://127.0.0.1:1", "qdrant": tuned}}
    )

    built = _built(settings)

    assert dataclasses.asdict(built.shape) == tuned
    await built.teardown()


async def test_the_configured_defaults_are_the_shape_a_store_has_without_one() -> None:
    """Two statements of Qdrant's defaults, held to one another.

    The settings are what an installation reads and the shape is what a store built outside the
    factory uses, and both claim to be the collection every installation already has. If they
    disagreed, the first start after an upgrade would rewrite every collection to whichever was
    wrong.
    """
    settings = Settings.model_validate(
        {"storage": {"vector_db": "qdrant", "vector_db_url": "http://127.0.0.1:1"}}
    )

    built = _built(settings)

    assert built.shape == CollectionShape()
    await built.teardown()


def test_the_factory_refuses_to_build_without_an_endpoint() -> None:
    """Normally refused earlier by policy; refused here too, for a caller outside that path."""
    settings = Settings.model_validate({"storage": {"vector_db": "qdrant"}})

    with pytest.raises(ConfigError, match=r"storage\.vector_db_url"):
        build_qdrant_vector_store(
            BuildContext(
                settings=settings,
                config=QdrantVectorStoreConfig(),
                data_dir=settings.data_dir,
                cache_dir=settings.cache_dir,
                components=_NoComponents(),
            )
        )


# --- the collection's shape ------------------------------------------------------------------

STOCK_COLLECTION: Final[dict[str, dict[str, Any]]] = {
    "params": {
        "vectors": {"size": 4, "distance": "Cosine"},
        "shard_number": 1,
        "replication_factor": 1,
        "write_consistency_factor": 1,
        "on_disk_payload": True,
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000,
        "max_indexing_threads": 0,
        "on_disk": False,
    },
    "optimizer_config": {
        "deleted_threshold": 0.2,
        "vacuum_min_vector_number": 1000,
        "default_segment_number": 0,
        "indexing_threshold": 10000,
        "flush_interval_sec": 5,
    },
    "wal_config": {"wal_capacity_mb": 32, "wal_segments_ahead": 0, "wal_retain_closed": 1},
}
"""What ``qdrant/qdrant`` reports for a collection created with only a size and a distance.

Recorded from v1.19.1, the server CI runs, and identical from v1.17.0. It is the collection every
installation made before its shape was configurable, which is why it is a recording rather than
something built here: the property under test is that a store leaves *that* collection alone.
"""


def _config(**changes: dict[str, Any]) -> models.CollectionConfig:
    """The stock collection with each section of ``changes`` merged into its own."""
    merged: dict[str, dict[str, Any]] = {
        name: dict(section) for name, section in STOCK_COLLECTION.items()
    }
    for name, section in changes.items():
        merged[name] = {**merged.get(name, {}), **section}
    return models.CollectionConfig.model_validate(merged)


def _drift(shape: CollectionShape, config: models.CollectionConfig) -> dict[str, object]:
    vector = config.params.vectors
    assert isinstance(vector, models.VectorParams)
    return {item.setting: item.in_force for item in shape.drift(config, vector)}


def test_the_default_shape_is_the_collection_a_stock_server_already_has() -> None:
    """An upgrade must not write to a collection nobody tuned.

    Every default is compared against what the server reports rather than against a value
    somebody believed was Qdrant's, so a default that is wrong — the client's own docstring
    gives the indexing threshold as 20,000 — reshapes nothing here and fails this instead.
    """
    assert _drift(CollectionShape(), _config()) == {}


def test_a_vector_s_own_value_is_the_one_in_force() -> None:
    """A vector's HNSW and quantization take precedence over the collection's.

    Comparing the collection's value alone would report a collection this store had tuned as
    untuned, and the reshape would then repeat on every start without ever converging.
    """
    config = _config(
        params={
            "vectors": {
                "size": 4,
                "distance": "Cosine",
                "hnsw_config": {"m": 32},
                "quantization_config": {"scalar": {"type": "int8", "always_ram": True}},
            }
        },
        quantization_config={"product": {"compression": "x16"}},
    )

    in_force = _drift(CollectionShape(hnsw_m=32, quantization="scalar"), config)

    assert in_force == {}


def test_a_collection_s_value_is_in_force_where_the_vector_has_none() -> None:
    """Quantization set on the collection by hand is quantization, whatever the vector says.

    Read as "none", it would leave a product-quantized collection standing under a
    configuration that asks for no quantization at all.
    """
    config = _config(quantization_config={"product": {"compression": "x16"}})

    assert _drift(CollectionShape(), config) == {"quantization": "product"}
    assert _drift(CollectionShape(quantization="scalar"), config) == {"quantization": "product"}


def test_the_newer_memory_placement_is_read_as_the_flag_it_replaced() -> None:
    """Qdrant is deprecating the placement flags for a ``memory`` enum, and may report either.

    A server that answered in the new terms would otherwise read as having every placement at
    its default, and a store would ask it for the same change on every start.
    """
    config = _config(
        params={
            "vectors": {
                "size": 4,
                "distance": "Cosine",
                "memory": "cold",
                "quantization_config": {"scalar": {"type": "int8", "memory": "pinned"}},
            },
            "payload": {"memory": "cached"},
        }
    )

    shape = CollectionShape(quantization="scalar", on_disk_vectors=True, on_disk_payload=False)

    assert _drift(shape, config) == {}


def test_placement_of_a_quantized_copy_is_not_compared_when_there_is_none() -> None:
    """``quantization_always_ram`` means nothing without quantization, on either side."""
    assert _drift(CollectionShape(quantization_always_ram=False), _config()) == {}


def test_every_dial_that_differs_is_named_with_what_the_collection_has() -> None:
    """The reshape sends exactly the dials that differ, and the refusal names them."""
    config = _config(
        params={"on_disk_payload": False},
        hnsw_config={"ef_construct": 64},
        optimizer_config={"indexing_threshold": 20000},
    )

    assert _drift(CollectionShape(), config) == {
        "on_disk_payload": False,
        "hnsw_ef_construct": 64,
        "indexing_threshold_kb": 20000,
    }


def test_the_dials_read_from_a_collection_are_exactly_the_shape_s() -> None:
    """A dial read and never compared, or compared and never read, is a silent half-feature."""
    vector = _config().params.vectors
    assert isinstance(vector, models.VectorParams)

    assert set(dials_in_force(_config(), vector)) == {
        dial.name for dial in dataclasses.fields(CollectionShape)
    }


def test_an_update_carries_only_the_dials_that_differ() -> None:
    """A request that restated every dial would re-send placement and quantization with it.

    HNSW goes as a pair even when one half differs, so a server that merges the diff and one
    that replaces it both end at the configured graph.
    """
    config = _config(hnsw_config={"ef_construct": 64})
    vector = config.params.vectors
    assert isinstance(vector, models.VectorParams)
    shape = CollectionShape()

    update = shape.update_for(shape.drift(config, vector), config, vector)

    assert update.vectors == {
        "": models.VectorParamsDiff(hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100))
    }
    assert (update.quantization, update.optimizers, update.params) == (None, None, None)


def test_turning_quantization_off_clears_both_places_it_lives() -> None:
    """Clearing only the vector's would leave the collection's in force behind it."""
    config = _config(
        params={
            "vectors": {
                "size": 4,
                "distance": "Cosine",
                "quantization_config": {"scalar": {"type": "int8"}},
            }
        },
        quantization_config={"product": {"compression": "x16"}},
    )
    vector = config.params.vectors
    assert isinstance(vector, models.VectorParams)
    shape = CollectionShape()

    update = shape.update_for(shape.drift(config, vector), config, vector)

    assert update.vectors == {
        "": models.VectorParamsDiff(quantization_config=models.Disabled.DISABLED)
    }
    assert update.quantization == models.Disabled.DISABLED


@pytest.mark.parametrize("newer", [False, True], ids=["flags", "memory"])
def test_placement_is_written_in_the_vocabulary_the_collection_reports(newer: bool) -> None:
    """``memory`` overrides the flag it replaces, and a server that predates it drops it.

    So a flag written to a component reporting ``memory`` is accepted and changes nothing, and
    ``memory`` written to one that does not is accepted and changes nothing. Either would be
    refused on every start as a server that did not apply a change.
    """
    params: dict[str, Any] = {"vectors": {"size": 4, "distance": "Cosine"}}
    if newer:
        params = {
            "vectors": {"size": 4, "distance": "Cosine", "memory": "cached"},
            "payload": {"memory": "cold"},
        }
    config = _config(params=params)
    vector = config.params.vectors
    assert isinstance(vector, models.VectorParams)
    shape = CollectionShape(on_disk_vectors=True, on_disk_payload=False)

    update = shape.update_for(shape.drift(config, vector), config, vector)

    if newer:
        assert update.vectors == {"": models.VectorParamsDiff(memory=models.Memory.COLD)}
        assert update.params == models.CollectionParamsDiff(
            payload=models.PayloadStorageParams(memory=models.Memory.CACHED)
        )
    else:
        assert update.vectors == {"": models.VectorParamsDiff(on_disk=True)}
        assert update.params == models.CollectionParamsDiff(on_disk_payload=False)


async def test_a_new_collection_is_created_in_the_configured_shape(
    client: AsyncQdrantClient,
) -> None:
    """The half of a shape the in-process engine keeps, which is the half written on the vector."""
    shape = CollectionShape(
        quantization="scalar",
        quantization_always_ram=False,
        on_disk_vectors=True,
        hnsw_m=32,
        hnsw_ef_construct=256,
    )
    store = QdrantVectorStore(
        client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX, shape=shape
    )

    await prepared(store)

    info = await client.get_collection(
        collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    )
    vector = info.config.params.vectors
    assert isinstance(vector, models.VectorParams)
    assert vector.on_disk is True
    assert vector.hnsw_config == models.HnswConfigDiff(m=32, ef_construct=256)
    assert vector.quantization_config == models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=False)
    )


async def test_local_mode_is_never_asked_to_reshape(
    client: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-process engine ignores every update and says nothing, so it is not asked.

    Asked anyway, the comparison afterwards would find nothing applied and refuse — every test
    in this file that prepares a store twice under two shapes would fail for a property of the
    test double rather than of the store.
    """

    async def refuse(*_: object, **__: object) -> bool:
        raise AssertionError("local mode was asked to update a collection")

    monkeypatch.setattr(client, "update_collection", refuse)
    await prepared(
        QdrantVectorStore(client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX)
    )
    tuned = QdrantVectorStore(
        client,
        workspace_id=WORKSPACE,
        collection_prefix=TEST_COLLECTION_PREFIX,
        shape=CollectionShape(quantization="scalar", hnsw_m=64),
    )

    await prepared(tuned)


@pytest.mark.parametrize(
    ("vectors", "refusal"),
    [
        (
            models.VectorParams(
                size=4, distance=models.Distance.COSINE, datatype=models.Datatype.FLOAT16
            ),
            "float16 vectors",
        ),
        (
            models.VectorParams(
                size=4, distance=models.Distance.COSINE, datatype=models.Datatype.UINT8
            ),
            "uint8 vectors",
        ),
        (models.VectorParams(size=8, distance=models.Distance.COSINE), "8-dimension vectors"),
        (models.VectorParams(size=4, distance=models.Distance.DOT), "ranked by Dot"),
        (
            {"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
            "named vectors",
        ),
    ],
    ids=["float16", "uint8", "wrong-size", "dot", "named"],
)
async def test_a_collection_this_store_cannot_use_is_refused_before_it_is_written(
    store: QdrantVectorStore,
    client: AsyncQdrantClient,
    vectors: models.VectorParams | dict[str, models.VectorParams],
    refusal: str,
) -> None:
    """A collection bearing this store's name and shaped by something else.

    The float16 case is the one that motivates the check. Every write succeeds, every readback
    returns numbers other than the float32 the checksum was taken over, and the whole corpus
    reads as corrupt and drops silently out of search. Refused when the store is prepared, it
    is one error naming the collection instead.
    """
    name = collection_for(TEST_COLLECTION_PREFIX, WORKSPACE, fingerprint(4))
    await client.create_collection(collection_name=name, vectors_config=vectors)

    with pytest.raises(VectorStoreStateError, match=refusal):
        await prepared(store)


# --- against a real server -------------------------------------------------------------------


@pytest.fixture
def server_client() -> AsyncQdrantClient:
    """A client on a real Qdrant, or a skipped test saying how to get one."""
    return remote_client(require_server())


@pytest.fixture
def server_store(server_client: AsyncQdrantClient) -> QdrantVectorStore:
    """A store on a real server, in a workspace of its own so reruns do not collide."""
    return QdrantVectorStore(
        server_client,
        workspace_id=f"{WORKSPACE}-{uuid.uuid4()}",
        collection_prefix=TEST_COLLECTION_PREFIX,
    )


async def _drop(store: QdrantVectorStore, client: AsyncQdrantClient, embed_dimension: int) -> None:
    """Leave the server as it was found, including the meta collection workspaces share.

    A suite pointed at a real server has to clean up after itself or it accretes a collection
    per run, and the one that is easy to forget is the shared one: each store drops its own
    vectors and the fingerprint record outlives them. The workspace's record is found by the
    ``workspace_id`` the store itself writes into the payload rather than by recomputing a point
    id, and the collection goes only once it is empty — another test's workspace may still be in
    it.
    """
    collection = collection_for(
        TEST_COLLECTION_PREFIX,
        store.workspace_id,
        fingerprint(embed_dimension),
    )
    if await client.collection_exists(collection):
        await client.delete_collection(collection)

    meta = meta_collection_for(TEST_COLLECTION_PREFIX)
    if not await client.collection_exists(meta):
        return
    await client.delete(
        collection_name=meta,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id", match=models.MatchValue(value=store.workspace_id)
                    )
                ]
            )
        ),
        wait=True,
    )
    if (await client.count(meta, exact=True)).count == 0:
        await client.delete_collection(meta)


@pytest.mark.contract
async def test_a_real_server_stores_the_float32_that_was_written(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """The property local mode cannot show, and the one the checksum contract rests on.

    Local mode is a Python reimplementation that hands back the list it was given, so a store
    that mishandled the transport's rounding would pass every other test in this file.
    """
    try:
        await server_store.ensure_ready(fingerprint(8))
        stored = chunk("chunk-one")
        offered = [0.3, -0.4, 0.5, 0.1, -0.2, 0.05, 0.7, -0.15]
        await server_store.upsert([stored], [offered])

        verdict = (await server_store.stored_vectors([stored]))[stored.id]

        assert verdict.integrity is VectorIntegrity.VERIFIED
        assert tuple(verdict.vector) == canonical_stored_vector(offered)
        assert vector_checksum(tuple(verdict.vector)) == vector_checksum(
            canonical_stored_vector(offered)
        )
    finally:
        await _drop(server_store, server_client, 8)


@pytest.mark.contract
async def test_a_real_server_indexes_the_payload_fields_queries_filter_on(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """Local mode ignores payload indexes, so only a server can show they were created."""
    try:
        await server_store.ensure_ready(fingerprint(8))
        collection = collection_for(
            TEST_COLLECTION_PREFIX,
            server_store.workspace_id,
            fingerprint(8),
        )

        info = await server_client.get_collection(collection)
        schema = info.payload_schema or {}

        assert {field for field, _ in INDEXED_PAYLOAD_FIELDS} <= set(schema)
    finally:
        await _drop(server_store, server_client, 8)


async def test_a_real_server_regains_an_index_that_was_never_created(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """A crash between creating the collection and indexing it must not be permanent.

    An index is invisible to correctness — Qdrant answers a filter with or without one — so
    nothing goes red when this is skipped and the corpus simply gets slower as it grows. The
    collection is what ``ensure_ready`` checks for, so the indexes have to be declared on every
    call rather than only on the one that made it.
    """
    try:
        await server_store.ensure_ready(fingerprint(8))
        collection = collection_for(
            TEST_COLLECTION_PREFIX,
            server_store.workspace_id,
            fingerprint(8),
        )
        await server_client.delete_payload_index(collection, field_name=DOCUMENT_ID_COLUMN)
        dropped = await server_client.get_collection(collection)
        assert DOCUMENT_ID_COLUMN not in (dropped.payload_schema or {})

        await server_store.ensure_ready(fingerprint(8))

        healed = await server_client.get_collection(collection)
        assert DOCUMENT_ID_COLUMN in (healed.payload_schema or {})
    finally:
        await _drop(server_store, server_client, 8)


@pytest.mark.contract
async def test_a_real_server_answers_the_whole_reuse_contract(
    server_client: AsyncQdrantClient,
) -> None:
    """The conformance suite again, over a socket, against the engine that ships."""
    made: list[QdrantVectorStore] = []

    def factory() -> QdrantVectorStore:
        store = QdrantVectorStore(
            server_client,
            workspace_id=f"{WORKSPACE}-{uuid.uuid4()}",
            collection_prefix=TEST_COLLECTION_PREFIX,
        )
        made.append(store)
        return store

    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    try:
        await assert_vector_store_reuses_by_embedding_input(factory, chunks)
        await assert_vector_store_records_vector_checksums(factory, chunks)
    finally:
        for store in made:
            await _drop(store, server_client, 8)


def _handle(
    client: AsyncQdrantClient, workspace_id: str, shape: CollectionShape
) -> QdrantVectorStore:
    """Another handle on a workspace, as a restart with edited settings would build one."""
    return QdrantVectorStore(
        client, workspace_id=workspace_id, collection_prefix=TEST_COLLECTION_PREFIX, shape=shape
    )


def _record_updates(
    client: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    """Every ``update_collection`` the client is asked for, passed through to the server."""
    calls: list[dict[str, object]] = []
    original = client.update_collection

    async def recording(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(client, "update_collection", recording)
    return calls


async def _in_force(client: AsyncQdrantClient, store: QdrantVectorStore) -> dict[str, object]:
    config = (await client.get_collection(store.storage_name(fingerprint(8)))).config
    vector = config.params.vectors
    assert isinstance(vector, models.VectorParams)
    return dials_in_force(config, vector)


def _expected(shape: CollectionShape) -> dict[str, object]:
    """What a collection in ``shape`` reads as: no placement for a quantized copy it lacks."""
    expected: dict[str, object] = dataclasses.asdict(shape)
    if shape.quantization == "none":
        expected["quantization_always_ram"] = None
    return expected


TUNED: Final = CollectionShape(
    quantization="scalar",
    quantization_always_ram=True,
    on_disk_vectors=True,
    on_disk_payload=False,
    hnsw_m=32,
    hnsw_ef_construct=200,
    indexing_threshold_kb=5000,
)
"""Every dial away from its default, so a dial that is not applied cannot pass as one that is."""


async def test_a_real_server_leaves_a_collection_nobody_tuned_alone(
    server_store: QdrantVectorStore,
    server_client: AsyncQdrantClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upgrade: a collection the previous release made, prepared by this one.

    It is made here exactly as that release made it — a size and a distance, nothing else — and
    it must not be written to. An update that changed nothing would still be an update: a
    request every installation sends on its first start, and a re-optimization on any server
    that does not diff what it is sent.
    """
    name = server_store.storage_name(fingerprint(8))
    try:
        await server_client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
        )
        before = (await server_client.get_collection(name)).config
        updates = _record_updates(server_client, monkeypatch)

        await server_store.ensure_ready(fingerprint(8))

        assert updates == []
        assert (await server_client.get_collection(name)).config == before
    finally:
        await _drop(server_store, server_client, 8)


async def test_a_real_server_creates_a_collection_in_the_whole_configured_shape(
    server_store: QdrantVectorStore,
    server_client: AsyncQdrantClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Including the dials local mode drops at creation: payload placement and the threshold.

    And with no update behind it, because a collection created in the wrong shape and then
    corrected would pass every assertion about the result.
    """
    tuned = _handle(server_client, server_store.workspace_id, TUNED)
    updates = _record_updates(server_client, monkeypatch)
    try:
        await tuned.ensure_ready(fingerprint(8))

        assert updates == []
        assert await _in_force(server_client, tuned) == _expected(TUNED)
    finally:
        await _drop(server_store, server_client, 8)


async def test_a_real_server_brings_an_existing_collection_to_an_edited_shape(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """A setting read only at creation does nothing to the collection that already exists.

    Every dial is changed on a collection with a vector in it and then changed back, so each is
    shown to move in both directions — as configuration in force. Whether a quantized copy is
    then actually built and read around is the next test's question, and it needs a corpus
    large enough for Qdrant to build one.
    """
    try:
        await server_store.ensure_ready(fingerprint(8))
        await server_store.upsert([chunk("chunk-one")], [spread(8, 0)])

        await _handle(server_client, server_store.workspace_id, TUNED).ensure_ready(fingerprint(8))
        assert await _in_force(server_client, server_store) == _expected(TUNED)

        await _handle(server_client, server_store.workspace_id, CollectionShape()).ensure_ready(
            fingerprint(8)
        )
        assert await _in_force(server_client, server_store) == _expected(CollectionShape())
        assert await server_store.count() == 1
    finally:
        await _drop(server_store, server_client, 8)


async def test_a_real_server_drops_quantization_the_collection_itself_carries(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """Quantization set on the collection by hand, which the vector's own value falls back to.

    Clearing only the vector's value would report success, change nothing, and — because the
    comparison afterwards reads the collection's value — be refused on every start.
    """
    name = server_store.storage_name(fingerprint(8))
    try:
        await server_store.ensure_ready(fingerprint(8))
        await server_client.update_collection(
            collection_name=name,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
            ),
        )

        await server_store.ensure_ready(fingerprint(8))

        assert (await server_client.get_collection(name)).config.quantization_config is None
    finally:
        await _drop(server_store, server_client, 8)


async def test_a_real_server_moves_a_placement_somebody_set_in_the_newer_vocabulary(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """A component that reports ``memory`` is moved by ``memory``, not by the flag it overrides.

    Set by hand here, as an operator reading current Qdrant documentation would set it. A
    server that predates the field drops it, reports the flags, and is moved by those, so this
    passes against both — which is the point of writing in the reported vocabulary.
    """
    name = server_store.storage_name(fingerprint(8))
    moved = CollectionShape(on_disk_vectors=True, on_disk_payload=False)
    try:
        await server_store.ensure_ready(fingerprint(8))
        await server_client.update_collection(
            collection_name=name,
            vectors_config={"": models.VectorParamsDiff(memory=models.Memory.CACHED)},
            collection_params=models.CollectionParamsDiff(
                payload=models.PayloadStorageParams(memory=models.Memory.COLD)
            ),
        )

        await _handle(server_client, server_store.workspace_id, moved).ensure_ready(fingerprint(8))

        assert await _in_force(server_client, server_store) == _expected(moved)
    finally:
        await _drop(server_store, server_client, 8)


async def test_a_real_server_that_does_not_apply_a_change_is_refused(
    server_store: QdrantVectorStore,
    server_client: AsyncQdrantClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that accepts a dial and ignores it — an older Qdrant — leaves a setting inert.

    That is the failure the whole reshape exists to close, so it is refused with the dial and
    both values named rather than logged and served past.
    """

    async def accept_and_ignore(**_: object) -> bool:
        return True

    try:
        await server_store.ensure_ready(fingerprint(8))
        monkeypatch.setattr(server_client, "update_collection", accept_and_ignore)
        tuned = _handle(server_client, server_store.workspace_id, CollectionShape(hnsw_m=32))

        with pytest.raises(
            VectorStoreStateError, match=r"storage\.qdrant\.hnsw_m is 32, the collection has 16"
        ):
            await tuned.ensure_ready(fingerprint(8))
    finally:
        monkeypatch.undo()
        await _drop(server_store, server_client, 8)


QUANTIZED_POINTS: Final = 300
"""Enough eight-dimension vectors to fill several segments past a one-kilobyte threshold."""


async def _wait_until_indexed(client: AsyncQdrantClient, collection: str, points: int) -> None:
    """Return once every point is in an indexed segment, which is where a quantized copy lives.

    Qdrant builds the graph and the quantized vectors when its optimizer turns an appendable
    segment into an indexed one, in the background and after the write returns. Reading before
    that reads a plain segment with no copy in it, and passes whatever quantization does.
    """
    info = await client.get_collection(collection)
    for _ in range(300):
        info = await client.get_collection(collection)
        if info.status is models.CollectionStatus.GREEN and info.indexed_vectors_count == points:
            return
        await asyncio.sleep(0.1)
    pytest.fail(
        f"{collection} indexed {info.indexed_vectors_count} of {points} points in 30 seconds, so "
        f"no quantized segment exists and this test would prove nothing about one."
    )


@pytest.mark.contract
async def test_a_real_server_keeps_a_quantized_corpus_verifiable(
    server_store: QdrantVectorStore, server_client: AsyncQdrantClient
) -> None:
    """Quantization is offered because it keeps the originals, and this is that claim, checked.

    A quantized copy that replaced what ``stored_vectors`` reads would fail every checksum, and
    the corpus would drop out of search while every request succeeded — the reason ``datatype``
    is not offered at all. So the corpus is made large enough, and the threshold low enough,
    that every point is in an indexed segment carrying the int8 copy before anything is read:
    three points under the default threshold never leave a plain segment, and a test over those
    passes with quantization off.
    """
    shape = dataclasses.replace(TUNED, indexing_threshold_kb=1)
    tuned = _handle(server_client, server_store.workspace_id, shape)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(QUANTIZED_POINTS)]
    offered = [
        [byte / 255 - 0.5 for byte in hashlib.sha256(f"point-{index}".encode()).digest()[:8]]
        for index in range(QUANTIZED_POINTS)
    ]
    try:
        await tuned.ensure_ready(fingerprint(8))
        await tuned.upsert(chunks, offered)
        await _wait_until_indexed(
            server_client, tuned.storage_name(fingerprint(8)), QUANTIZED_POINTS
        )

        verdicts = await tuned.stored_vectors(chunks)
        coverage = await tuned.checksum_coverage(recompute=True)
        probes = (0, 137, QUANTIZED_POINTS - 1)
        ranked = [await tuned.search(offered[index], k=1) for index in probes]

        assert {verdict.integrity for verdict in verdicts.values()} == {VectorIntegrity.VERIFIED}
        assert [tuple(verdicts[item.id].vector) for item in chunks] == [
            canonical_stored_vector(vector) for vector in offered
        ]
        assert (coverage.verified, coverage.failed) == (QUANTIZED_POINTS, 0)
        assert [[result.chunk.id for result in found] for found in ranked] == [
            [f"chunk-{index}"] for index in probes
        ]
    finally:
        await _drop(server_store, server_client, 8)
