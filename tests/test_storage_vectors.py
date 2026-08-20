"""What the LanceDB vector store promises, and the guards that keep it honest.

Two failures drive most of this file, and neither raises on its own. Vectors from a second
model of the same dimension mix in silently and every later answer is drawn from a space the
query does not live in; and a document id interpolated into a predicate unescaped is a
predicate the connector's data gets to write. The conformance suites cover the first at the
protocol level, and the tests here cover both at the level where the mistake is actually
made.
"""

# pyright: reportUnknownMemberType=false
#
# Two tests below tamper with `_manicule_meta` through lancedb directly, which is the only
# way to produce a directory that describes itself twice. lancedb annotates its surface in
# terms of `pyarrow`, which ships no type information, so those call sites are "partially
# unknown" — see the same note in `manicule.storage.vectors`. Nothing else in this file is
# affected; strict checking otherwise applies.

from __future__ import annotations

import random
import shutil
from typing import TYPE_CHECKING, Final

import lancedb
import pytest
from lancedb.index import IvfPq

from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.ann import MINIMUM_ANN_INDEX_THRESHOLD, AnnLifecycle, partitions_for
from manicule.core.content import BlockKind, Chunk
from manicule.core.embedding import (
    EmbedFingerprint,
    Pooling,
    VectorState,
    classify_stored_vector,
    embedding_input_identity,
)
from manicule.core.errors import FingerprintMismatchError
from manicule.core.protocols import VectorStore
from manicule.core.retrieval import Filter
from manicule.storage.engine import VECTORS_DIRNAME
from manicule.storage.vectors import (
    CHUNK_ID_COLUMN,
    DISTANCE_METRIC,
    EXEMPT_FILTER_FIELDS,
    IDENTITY_COLUMN,
    META_TABLE,
    PUBLICATION_COLUMN,
    VECTOR_COLUMN,
    LanceVectorStore,
    PublishedLanceVectorStore,
    VectorStoreStateError,
    predicate_for,
    quote,
    table_name,
    unit,
)
from manicule.testing import (
    assert_protocol_signatures,
    assert_vector_store_is_dimension_agnostic,
    assert_vector_store_rejects_foreign_vectors,
    assert_vector_store_reuses_by_embedding_input,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.embedding import Vector


SCOPE = frozenset({"default"})
"""The workspace every filter here is built for.

``Filter.workspace_ids`` is required and this store is not the thing that enforces it — see
``EXEMPT_FILTER_FIELDS`` — so every filter below carries a scope and none of them expects a
predicate to come out of it.
"""


def fingerprint(dimension: int = 4, model_id: str = "test/model") -> EmbedFingerprint:
    """An embedder identity at ``dimension``. Nothing in the tests assumes the number."""
    return EmbedFingerprint(
        model_id=model_id,
        dimension=dimension,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=512,
    )


def chunk(
    chunk_id: str,
    document_id: str = "doc-1",
    *,
    kind: BlockKind = BlockKind.PROSE,
    position: int = 0,
    lang: str = "en",
) -> Chunk:
    """A chunk carrying a located anchor, so the round trip has something to lose."""
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=f"the text of {chunk_id}",
        embed_text=f"Section > the text of {chunk_id}",
        anchor=HeadingAnchor(path=("Section",), fragment="section"),
        heading_path=("Section",),
        kind=kind,
        position=position,
        token_count=7,
        metadata={"lang": lang},
    )


@pytest.fixture
def store(tmp_path: Path) -> LanceVectorStore:
    """A store on an empty directory that does not exist yet."""
    return LanceVectorStore(tmp_path / "vectors")


async def prepared(directory: Path, dimension: int = 4) -> LanceVectorStore:
    """A store that has been through ``ensure_ready`` at ``dimension``."""
    store = LanceVectorStore(directory)
    await store.ensure_ready(fingerprint(dimension))
    return store


def spread(dimension: int, index: int) -> list[float]:
    """A one-hot vector, so similarity between two of them is decided by ``index``."""
    return [1.0 if position == index % dimension else 0.0 for position in range(dimension)]


# --- conformance -------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_store_satisfies_the_vector_store_protocol(store: LanceVectorStore) -> None:
    """Structural conformance for the backend that ships, which nothing checked until now.

    ``MemoryVectorStore`` has been signature-checked against ``VectorStore`` since the protocol
    existed and ``LanceVectorStore`` never has — so the one implementation an installation
    actually runs was the one nothing held to the protocol's shape.
    ``SqliteDocStore`` has had this test all along; its opposite number did not.

    Both halves are needed, and the second is the one that bites. ``isinstance`` checks the
    attributes exist; ``@runtime_checkable`` deliberately checks nothing about what they
    accept, so a store whose ``stored_vectors`` took ``chunk_ids`` where the protocol says
    ``chunks`` would pass every ``isinstance`` in the codebase and fail at the first keyword
    call — which, for a method reached once per document, is somewhere in the middle of a
    corpus sweep.
    """
    assert isinstance(store, VectorStore)
    assert_protocol_signatures(store, VectorStore)


@pytest.mark.contract
async def test_the_store_works_at_whatever_dimension_the_embedder_reports(
    tmp_path: Path,
) -> None:
    """A hardcoded dimension anywhere — schema, buffer, assertion — fails one of these."""
    made: list[LanceVectorStore] = []

    def make_store() -> VectorStore:
        store = LanceVectorStore(tmp_path / f"vectors-{len(made)}")
        made.append(store)
        return store

    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await assert_vector_store_is_dimension_agnostic(make_store, chunks)


@pytest.mark.contract
async def test_the_store_refuses_vectors_from_a_second_model_of_the_same_size(
    tmp_path: Path,
) -> None:
    """A size check passes this case, and every answer afterwards is quietly meaningless."""
    made: list[LanceVectorStore] = []

    def make_store() -> VectorStore:
        store = LanceVectorStore(tmp_path / f"vectors-{len(made)}")
        made.append(store)
        return store

    await assert_vector_store_rejects_foreign_vectors(make_store)


# --- round trip --------------------------------------------------------------------------


async def test_a_chunk_comes_back_from_search_with_everything_it_went_in_with(
    store: LanceVectorStore,
) -> None:
    """A stage after the first scores the text and cites the anchor; an id is not enough."""
    await store.ensure_ready(fingerprint())
    original = chunk("chunk-a")
    await store.upsert([original], [spread(4, 0)])

    found = await store.search(spread(4, 0), k=1)

    assert [candidate.chunk for candidate in found] == [original]


async def test_the_score_of_an_identical_vector_is_a_cosine_similarity_of_one(
    store: LanceVectorStore,
) -> None:
    """``score`` is compared against a threshold downstream, so its scale has to be real."""
    await store.ensure_ready(fingerprint())
    await store.upsert([chunk("chunk-a")], [[3.0, 0.0, 0.0, 0.0]])

    found = await store.search([9.0, 0.0, 0.0, 0.0], k=1)

    assert found[0].score == pytest.approx(1.0)


async def test_an_orthogonal_vector_scores_zero_and_ranks_below_a_parallel_one(
    store: LanceVectorStore,
) -> None:
    """Ordering and scale come from the same number; a transform that keeps one loses the other."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("parallel"), chunk("orthogonal", position=1)],
        [spread(4, 0), spread(4, 1)],
    )

    found = await store.search(spread(4, 0), k=2)

    assert [candidate.chunk.id for candidate in found] == ["parallel", "orthogonal"]
    assert found[1].score == pytest.approx(0.0)


async def test_upserting_a_chunk_again_replaces_it_rather_than_storing_it_twice(
    store: LanceVectorStore,
) -> None:
    """Re-ingest is not additive: a chunk indexed twice is a duplicate citation."""
    await store.ensure_ready(fingerprint())
    await store.upsert([chunk("chunk-a")], [spread(4, 0)])
    await store.upsert([chunk("chunk-a", document_id="doc-2")], [spread(4, 1)])

    assert await store.count() == 1
    found = await store.search(spread(4, 1), k=1)
    assert found[0].chunk.document_id == "doc-2"


async def test_a_stored_vector_is_normalized_whatever_length_it_arrived_at(
    store: LanceVectorStore,
) -> None:
    """Cosine as ``1 - distance`` holds only for unit vectors, so normalizing is not optional."""
    await store.ensure_ready(fingerprint())
    await store.upsert([chunk("chunk-a")], [[0.0, 40.0, 0.0, 0.0]])

    found = await store.search([0.0, 0.25, 0.0, 0.0], k=1)

    assert found[0].score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-inf", "negative-inf"],
)
async def test_a_non_finite_vector_is_refused_before_storage(
    store: LanceVectorStore, value: float
) -> None:
    """Shape alone cannot make a value usable in cosine-distance arithmetic."""
    embed = fingerprint().model_copy(update={"backend": "test-backend"})
    await store.ensure_ready(embed)
    vector = spread(4, 0)
    vector[1] = value

    with pytest.raises(ValueError, match="non-finite") as raised:
        await store.upsert([chunk("broken")], [vector])

    message = str(raised.value)
    assert "non-finite" in message
    assert "test/model" in message
    assert "test-backend" in message
    assert await store.count() == 0


async def test_float32_overflow_is_refused_without_a_warning_or_partial_row(
    store: LanceVectorStore,
) -> None:
    """A finite Python float can still be non-finite in Lance's physical type."""
    import warnings  # noqa: PLC0415

    embed = fingerprint().model_copy(update={"backend": "overflowing-backend"})
    await store.ensure_ready(embed)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="non-finite") as raised:
            await store.upsert(
                [chunk("overflow")],
                [[1e39, 0.0, 0.0, 0.0]],
                publication_id="never-published",
            )

    assert "overflowing-backend" in str(raised.value)
    assert await store.count() == 0


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-inf", "negative-inf"],
)
def test_an_existing_non_finite_vector_is_not_reused(value: float) -> None:
    """A damaged or legacy row must cause repair, not poison every future retrieval."""
    item = chunk("broken")
    embed = fingerprint()
    identity = embedding_input_identity(item.embed_text, document_id=item.document_id, embed=embed)
    vector = spread(4, 0)
    vector[1] = value

    verdict = classify_stored_vector(
        item,
        recorded_identity=identity,
        stored_embed_text=item.embed_text,
        stored_vector=vector,
        embed=embed,
    )

    assert verdict.state is VectorState.CORRUPT
    assert not verdict.is_reusable
    assert verdict.vector == ()


# --- filters -----------------------------------------------------------------------------


async def test_a_document_filter_excludes_the_chunks_of_every_other_document(
    store: LanceVectorStore,
) -> None:
    """The filter is the tenancy and scoping mechanism; one that does not exclude is decoration."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("wanted", document_id="doc-1"), chunk("unwanted", document_id="doc-2")],
        [spread(4, 0), spread(4, 0)],
    )

    found = await store.search(
        spread(4, 0), k=10, filter=Filter(workspace_ids=SCOPE, document_ids=frozenset({"doc-1"}))
    )

    assert [candidate.chunk.id for candidate in found] == ["wanted"]


async def test_a_kind_filter_excludes_the_kinds_it_does_not_name(
    store: LanceVectorStore,
) -> None:
    """``kind`` is promoted to its own column precisely so this does not need a join."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("prose"), chunk("code", kind=BlockKind.CODE, position=1)],
        [spread(4, 0), spread(4, 0)],
    )

    found = await store.search(
        spread(4, 0), k=10, filter=Filter(workspace_ids=SCOPE, kinds=frozenset({BlockKind.CODE}))
    )

    assert [candidate.chunk.id for candidate in found] == ["code"]


async def test_a_filter_narrows_the_search_before_k_is_applied(
    store: LanceVectorStore,
) -> None:
    """Filtering the top ``k`` afterwards returns fewer than ``k``, and the caller reads that
    as a corpus with nothing more in it (``docs/storage.md`` §6.6)."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [
            chunk("nearest", document_id="doc-1"),
            chunk("near", document_id="doc-1", position=1),
            chunk("furthest", document_id="doc-2", position=2),
        ],
        [spread(4, 0), [0.9, 0.1, 0.0, 0.0], spread(4, 3)],
    )

    found = await store.search(
        spread(4, 0), k=1, filter=Filter(workspace_ids=SCOPE, document_ids=frozenset({"doc-2"}))
    )

    assert [candidate.chunk.id for candidate in found] == ["furthest"]


async def test_a_filter_field_the_vector_table_cannot_answer_is_refused_not_ignored() -> None:
    """Applying part of a filter and dropping the rest returns rows it was written to exclude."""
    with pytest.raises(ValueError, match="collection_ids"):
        predicate_for(Filter(workspace_ids=SCOPE, collection_ids=frozenset({"c1"})))


async def test_the_workspace_scope_is_exempt_by_name_rather_than_by_omission() -> None:
    """The one field this store drops on purpose, because its enforcement moved.

    Tenancy has no Lance column and will not get one; the hydrating join in the dense stage
    applies it instead (``docs/retrieval.md`` §4.2). What makes that safe rather than merely
    stated is ``assert_pipeline_enforces_scope``, and what stops it looking like an oversight
    is that the field is named in ``EXEMPT_FILTER_FIELDS`` rather than missing from a loop.
    """
    assert "workspace_ids" in EXEMPT_FILTER_FIELDS
    assert predicate_for(Filter(workspace_ids=SCOPE)) is None
    assert predicate_for(None) is None


async def test_a_language_restriction_reaches_the_promoted_column(
    store: LanceVectorStore,
) -> None:
    """The Lance table promotes ``lang``; before #36 no filter field could name it."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [
            chunk("english", document_id="doc-1"),
            chunk("french", document_id="doc-2", position=1, lang="fr"),
        ],
        [spread(4, 0), spread(4, 1)],
    )

    found = await store.search(
        spread(4, 0), k=5, filter=Filter(workspace_ids=SCOPE, langs=frozenset({"fr"}))
    )

    assert [candidate.chunk.id for candidate in found] == ["french"]

    # A language tag is connector data like any other identifier, and reaches the predicate
    # through the same escape. A filter that widened to the whole table would look like a
    # working search.
    hostile = await store.search(
        spread(4, 0), k=5, filter=Filter(workspace_ids=SCOPE, langs=frozenset({"fr' OR '1'='1"}))
    )
    assert hostile == []


# --- predicate safety --------------------------------------------------------------------


async def test_a_quote_in_an_identifier_cannot_close_the_literal_it_sits_in() -> None:
    """Document ids are connector data — a page title, a path — and go straight into SQL."""
    assert quote("d1' OR '1'='1") == "'d1'' OR ''1''=''1'"


async def test_an_identifier_shaped_like_a_predicate_deletes_only_itself(
    store: LanceVectorStore,
) -> None:
    """The injection that matters here is a delete that widens to the whole table."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("kept", document_id="doc-1"), chunk("also-kept", document_id="doc-2", position=1)],
        [spread(4, 0), spread(4, 1)],
    )

    await store.delete_document("doc-1' OR '1'='1")

    assert await store.count() == 2


async def test_a_chunk_id_containing_a_quote_survives_storage_and_retrieval(
    store: LanceVectorStore,
) -> None:
    """An id that breaks the query is an id that silently never comes back."""
    await store.ensure_ready(fingerprint())
    awkward = chunk("it's a chunk", document_id="it's a doc")
    await store.upsert([awkward], [spread(4, 0)])

    found = await store.search(
        spread(4, 0),
        k=1,
        filter=Filter(workspace_ids=SCOPE, document_ids=frozenset({"it's a doc"})),
    )

    assert [candidate.chunk.id for candidate in found] == ["it's a chunk"]


# --- deletion ----------------------------------------------------------------------------


async def test_deleting_a_document_twice_leaves_the_store_as_the_first_delete_did(
    store: LanceVectorStore,
) -> None:
    """Reconciliation re-deletes what a crashed run already deleted, and must not raise."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("gone", document_id="doc-1"), chunk("stays", document_id="doc-2", position=1)],
        [spread(4, 0), spread(4, 1)],
    )

    await store.delete_document("doc-1")
    await store.delete_document("doc-1")

    assert await store.count() == 1
    assert [candidate.chunk.id for candidate in await store.search(spread(4, 1), k=5)] == ["stays"]


async def test_deleting_from_a_directory_that_holds_nothing_is_not_an_error(
    store: LanceVectorStore,
) -> None:
    """Deletion runs during recovery, when the store may never have been written to."""
    await store.delete_document("doc-1")

    assert await store.count() == 0


# --- fingerprints and persistence --------------------------------------------------------


async def test_a_store_that_holds_nothing_reports_no_fingerprint(
    store: LanceVectorStore,
) -> None:
    """``None`` is how a caller tells a fresh index from one built by another model."""
    assert await store.fingerprint() is None


async def test_a_second_store_on_the_same_directory_reads_what_is_there(
    tmp_path: Path,
) -> None:
    """The meta table exists so a directory describes itself to a process that did not write it."""
    directory = tmp_path / "vectors"
    written = await prepared(directory)
    await written.upsert([chunk("chunk-a")], [spread(4, 0)])
    await written.teardown()

    reopened = LanceVectorStore(directory)

    stored = await reopened.fingerprint()
    assert stored is not None
    assert stored.matches(fingerprint())
    assert await reopened.count() == 1


async def test_the_vector_table_is_named_after_the_fingerprint_it_holds(
    tmp_path: Path,
) -> None:
    """Two spaces cannot share a table name, so a rebuild can run beside the live one (§6.5)."""
    directory = tmp_path / "vectors"
    await prepared(directory)

    written = {path.stem for path in directory.iterdir()}

    assert written == {META_TABLE, table_name(fingerprint())}


async def test_the_same_fingerprint_offered_twice_is_accepted(tmp_path: Path) -> None:
    """Every run calls ``ensure_ready``; refusing the unchanged case would refuse every restart."""
    directory = tmp_path / "vectors"
    first = await prepared(directory)
    await first.upsert([chunk("chunk-a")], [spread(4, 0)])
    await first.teardown()

    second = LanceVectorStore(directory)
    await second.ensure_ready(fingerprint())

    assert await second.count() == 1


async def test_a_model_that_differs_only_in_pooling_is_refused(tmp_path: Path) -> None:
    """Pooling is chosen by manicule and changes the space; the dimension does not move."""
    store = await prepared(tmp_path / "vectors")
    other = fingerprint().model_copy(update={"pooling": Pooling.CLS})

    with pytest.raises(FingerprintMismatchError, match="pooling"):
        await store.ensure_ready(other)


async def test_a_meta_table_that_describes_two_indexes_is_refused(tmp_path: Path) -> None:
    """Picking a winner from a contradictory directory is how half a backup gets trusted."""
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    await store.teardown()

    connection = await lancedb.connect_async(directory)
    meta = await connection.open_table(META_TABLE)
    duplicate = fingerprint(model_id="other/model")
    await meta.add(
        [
            {
                "embed_fingerprint": duplicate.model_dump_json(),
                "canonical": duplicate.canonical(),
            }
        ]
    )

    with pytest.raises(VectorStoreStateError, match="2 rows"):
        await LanceVectorStore(directory).fingerprint()


async def test_a_meta_row_that_contradicts_itself_is_refused(tmp_path: Path) -> None:
    """The canonical column is what a repair tool compares; a row where they disagree is edited."""
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    await store.teardown()

    connection = await lancedb.connect_async(directory)
    await connection.drop_table(META_TABLE)
    tampered = await connection.create_table(
        META_TABLE,
        data=[
            {
                "embed_fingerprint": fingerprint().model_dump_json(),
                "canonical": fingerprint(model_id="other/model").canonical(),
            }
        ],
    )
    assert await tampered.count_rows() == 1

    with pytest.raises(VectorStoreStateError, match="contradicts itself"):
        await LanceVectorStore(directory).fingerprint()


# --- refusals ----------------------------------------------------------------------------


async def test_searching_before_the_store_is_prepared_is_refused(
    store: LanceVectorStore,
) -> None:
    """Retrieval refuses too, not only ingest: an unopened index cannot name its space (§6.3)."""
    with pytest.raises(VectorStoreStateError, match="ensure_ready"):
        await store.search([0.0, 1.0, 0.0, 0.0], k=1)


async def test_writing_before_the_store_is_prepared_is_refused(
    store: LanceVectorStore,
) -> None:
    """Without a fingerprint there is no dimension to build a table at, and no space to claim."""
    with pytest.raises(VectorStoreStateError, match="ensure_ready"):
        await store.upsert([chunk("chunk-a")], [[0.0, 1.0, 0.0, 0.0]])


async def test_a_vector_of_the_wrong_dimension_is_refused_on_the_way_in(
    store: LanceVectorStore,
) -> None:
    """Two embedders in one pipeline; the store is the last place it can be seen."""
    await store.ensure_ready(fingerprint())

    with pytest.raises(ValueError, match="index was built for 4"):
        await store.upsert([chunk("chunk-a")], [[1.0, 0.0]])


async def test_a_query_of_the_wrong_dimension_is_refused(store: LanceVectorStore) -> None:
    """A query from another model does not merely rank badly; it ranks meaninglessly."""
    await store.ensure_ready(fingerprint())

    with pytest.raises(ValueError, match="2-dimension query"):
        await store.search([1.0, 0.0], k=1)


async def test_chunks_and_vectors_of_different_lengths_are_refused(
    store: LanceVectorStore,
) -> None:
    """They are positional: a mismatch stores some chunk against another chunk's vector."""
    await store.ensure_ready(fingerprint())

    with pytest.raises(ValueError, match="2 chunk"):
        await store.upsert([chunk("a"), chunk("b", position=1)], [spread(4, 0)])


async def test_upserting_nothing_stores_nothing(store: LanceVectorStore) -> None:
    """A document that produced no chunks reaches here; it must not become an empty write."""
    await store.ensure_ready(fingerprint())
    empty_chunks: list[Chunk] = []
    empty_vectors: list[Vector] = []

    await store.upsert(empty_chunks, empty_vectors)

    assert await store.count() == 0


# --- degenerate vectors ------------------------------------------------------------------


async def test_a_vector_with_no_direction_is_left_alone_by_normalization() -> None:
    """There is no unit vector to scale it to, and inventing one would fabricate a direction."""
    assert unit([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


async def test_a_query_with_no_direction_returns_candidates_it_does_not_claim_to_rank(
    store: LanceVectorStore,
) -> None:
    """Cosine against the zero vector is undefined.

    Returning an empty list would claim the corpus is empty, which is a different answer.
    """
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("chunk-a"), chunk("chunk-b", position=1)],
        [spread(4, 0), spread(4, 1)],
    )

    found = await store.search([0.0, 0.0, 0.0, 0.0], k=2)

    assert len(found) == 2
    assert {candidate.score for candidate in found} == {0.0}


async def test_a_query_with_no_direction_still_honors_the_filter(
    store: LanceVectorStore,
) -> None:
    """The unrankable path is a second route to the rows, and skips no scoping on the way."""
    await store.ensure_ready(fingerprint())
    await store.upsert(
        [chunk("wanted", document_id="doc-1"), chunk("unwanted", document_id="doc-2", position=1)],
        [spread(4, 0), spread(4, 1)],
    )

    found = await store.search(
        [0.0, 0.0, 0.0, 0.0],
        k=5,
        filter=Filter(workspace_ids=SCOPE, document_ids=frozenset({"doc-1"})),
    )

    assert [candidate.chunk.id for candidate in found] == ["wanted"]


async def test_asking_for_no_candidates_returns_none_of_them(store: LanceVectorStore) -> None:
    """Lance rejects a limit of zero, and a caller that wants nothing is not an error."""
    await store.ensure_ready(fingerprint())
    await store.upsert([chunk("chunk-a")], [spread(4, 0)])

    assert await store.search(spread(4, 0), k=0) == []


async def test_a_chunk_without_a_location_round_trips_as_unlocated(
    store: LanceVectorStore,
) -> None:
    """``Unlocated`` is a member with a reason; flattening it to a guess is the whole failure."""
    await store.ensure_ready(fingerprint())
    unplaced = Chunk(
        id="unplaced",
        document_id="doc-1",
        text="text with nowhere to point",
        embed_text="text with nowhere to point",
        anchor=Unlocated(reason="the parser could not place it"),
        position=0,
        token_count=5,
    )
    await store.upsert([unplaced], [spread(4, 0)])

    found = await store.search(spread(4, 0), k=1)

    assert found[0].chunk.anchor == unplaced.anchor


# --- embedding-input identity ------------------------------------------------------------


@pytest.mark.contract
async def test_the_store_answers_reuse_on_the_embedding_input(tmp_path: Path) -> None:
    """The reuse contract, against the backend that ships rather than against a double.

    A store answering on chunk id alone passes every other check in this file: its writes
    succeed, its searches return, and its answers are drawn from the right space. What it gets
    wrong is a chunk whose ``embed_text`` moved while its ``text`` did not — silently, for as
    long as the corpus lives.
    """
    made: list[LanceVectorStore] = []

    def make_store() -> VectorStore:
        store = LanceVectorStore(tmp_path / f"vectors-{len(made)}")
        made.append(store)
        return store

    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await assert_vector_store_reuses_by_embedding_input(make_store, chunks)


async def test_publications_keep_two_vectors_for_one_stable_chunk_id(tmp_path: Path) -> None:
    """Changing only embed_text must not overwrite the vector the active revision still uses."""
    store = LanceVectorStore(tmp_path / "vectors")
    await store.ensure_ready(fingerprint())
    original = chunk("stable")
    rewritten = original.model_copy(update={"embed_text": f"new heading > {original.text}"})

    await store.upsert([original], [spread(4, 0)], publication_id="old")
    await store.upsert([rewritten], [spread(4, 1)], publication_id="new")

    assert await store.count() == 2
    candidates = await store.search([0.0] * 4, 10)
    assert {(item.chunk.id, item.publication_id) for item in candidates} == {
        (original.id, "old"),
        (original.id, "new"),
    }


async def test_a_content_addressed_publication_is_physically_immutable(tmp_path: Path) -> None:
    """A stale generation cannot overwrite a successor's row after losing its DB lease."""
    store = LanceVectorStore(tmp_path / "vectors")
    await store.ensure_ready(fingerprint())
    original = chunk("stable")
    conflicting = original.model_copy(update={"document_id": "stale-writer"})

    await store.upsert([original], [spread(4, 0)], publication_id="content-addressed")
    await store.upsert([conflicting], [spread(4, 1)], publication_id="content-addressed")

    assert await store.count() == 1
    found = await store.search(spread(4, 0), k=1)
    assert found[0].chunk.document_id == original.document_id
    assert found[0].score == pytest.approx(1.0)


async def test_terminal_publication_cleanup_is_exact_and_idempotent(tmp_path: Path) -> None:
    store = await prepared(tmp_path / "publication-cleanup")
    obsolete = chunk("obsolete")
    published = chunk("published")
    await store.upsert([obsolete], [spread(4, 0)], publication_id="obsolete-generation")
    await store.upsert([published], [spread(4, 1)], publication_id="published-generation")

    assert await store.delete_publication("obsolete-generation") == 1
    assert await store.delete_publication("obsolete-generation") == 0
    assert await store.publication_row_count("published-generation") == 1


async def test_generation_validation_scopes_all_documents_to_the_exact_publication(
    tmp_path: Path,
) -> None:
    store = await prepared(tmp_path / "multi-document-publication")
    first = chunk("first", document_id="document-1")
    second = chunk("second", document_id="document-2")
    unrelated = chunk("unrelated", document_id="document-old")
    await store.upsert([unrelated], [[1.0, 0.0, 0.0, 0.0]], publication_id="generation")
    assert not await store.publication_is_complete(
        "generation", [], embedding_fingerprint=fingerprint().canonical()
    )
    assert await store.publication_is_complete(
        "empty-generation", [], embedding_fingerprint=fingerprint().canonical()
    )
    await store.upsert([first], [[0.0, 1.0, 0.0, 0.0]], publication_id="generation")

    assert not await store.publication_is_complete(
        "generation", [first, second], embedding_fingerprint=fingerprint().canonical()
    )
    await store.upsert([second], [[0.0, 0.0, 1.0, 0.0]], publication_id="generation")
    assert not await store.publication_is_complete(
        "generation", [first, second], embedding_fingerprint=fingerprint().canonical()
    )
    await store.upsert([first], [[0.0, 1.0, 0.0, 0.0]], publication_id="generation-clean")
    await store.upsert([second], [[0.0, 0.0, 1.0, 0.0]], publication_id="generation-clean")
    assert await store.publication_is_complete(
        "generation-clean", [first, second], embedding_fingerprint=fingerprint().canonical()
    )


async def test_a_reused_vector_is_the_stored_vector_to_the_last_bit(
    store: LanceVectorStore,
) -> None:
    """Reading a row back and writing it again leaves the row exactly as it was.

    The claim "an unchanged chunk's vector was not recomputed" is only checkable if this is
    exactly true, and it is not free: a stored ``float32`` vector reads back with a length a
    few parts in 10^8 from one, and re-normalizing it perturbs the odd row by an ulp.
    :data:`~manicule.storage.vectors.FLOAT32_EPSILON` is what stops that.

    **A fixed seed and 256 rows**, because the guard must fail deterministically when it is
    removed and the defect it catches is rare. This exact fixture holds four rows that drift
    without it — measured against real Lance by removing the guard and counting. Two earlier
    fixtures, one of 200 rows built from tidy arithmetic and one of 64 from this seed, both
    passed with the guard gone and would have certified nothing.
    """
    await store.ensure_ready(fingerprint(8))
    rng = random.Random(0)  # noqa: S311 - a fixture, seeded so the guard fails deterministically
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(256)]
    # Deliberately neither one-hot nor normalized, so the store does the scaling and the round
    # trip has something to lose.
    vectors: list[Vector] = [[rng.uniform(-1.0, 1.0) for _ in range(8)] for _ in chunks]
    await store.upsert(chunks, vectors)

    first = await store.stored_vectors(chunks)
    assert all(verdict.state is VectorState.READABLE for verdict in first.values())
    await store.upsert(chunks, [first[chunk_.id].vector for chunk_ in chunks])
    again = await store.stored_vectors(chunks)

    assert {key: verdict.vector for key, verdict in again.items()} == {
        key: verdict.vector for key, verdict in first.items()
    }, "a row written back with the vector it already held is the row it already was"


async def test_a_table_written_before_the_identity_column_keeps_every_vector_it_has(
    tmp_path: Path,
) -> None:
    """The migration of an existing ``vectors/`` directory, against real Lance.

    The column is added in ``ensure_ready`` and every existing row reads as unrecorded. The
    conservative reading of an unrecorded identity — distrust it — would re-embed a whole
    corpus to learn what each row already says; the embedding input is instead reconstructed
    from the chunk the row was written with, which is what makes the upgrade free.
    """
    directory = tmp_path / "vectors"
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    vectors: list[Vector] = [spread(4, index) for index in range(3)]

    legacy = LanceVectorStore(directory)
    await legacy.ensure_ready(fingerprint())
    await legacy.upsert(chunks, vectors)
    await legacy.teardown()
    await _drop_identity_column(directory, fingerprint())

    reopened = LanceVectorStore(directory)
    verdicts = await reopened.stored_vectors(chunks)
    assert all(verdict.state is VectorState.READABLE for verdict in verdicts.values()), (
        "every row from before the column is still this chunk's vector, and reconstructing "
        "that from the chunk beside it costs no forward pass"
    )
    assert not any(verdict.identity_recorded for verdict in verdicts.values()), (
        "and each is reported as reconstructed, so the one-time backfill is a number rather "
        "than something that happened quietly"
    )

    await reopened.ensure_ready(fingerprint())
    await reopened.upsert(chunks, [verdicts[chunk_.id].vector for chunk_ in chunks])
    settled = await reopened.stored_vectors(chunks)
    assert all(verdict.identity_recorded for verdict in settled.values()), (
        "writing the rows records their identities, so the backfill happens once"
    )


async def test_a_legacy_table_is_readable_before_generation_columns_are_migrated(
    tmp_path: Path,
) -> None:
    """stored_vectors is a preflight read, so it cannot depend on ensure_ready migration."""
    directory = tmp_path / "vectors"
    original = chunk("legacy-chunk")
    vector = spread(4, 2)
    writer = LanceVectorStore(directory)
    await writer.ensure_ready(fingerprint())
    await writer.upsert([original], [vector])
    await writer.teardown()
    await _drop_generation_columns(directory, fingerprint())

    reopened = LanceVectorStore(directory)
    verdict = (await reopened.stored_vectors([original]))[original.id]

    assert verdict.state is VectorState.READABLE
    assert verdict.vector
    await reopened.teardown()


async def test_a_chunk_refiled_under_a_new_id_still_finds_its_vector(
    store: LanceVectorStore,
) -> None:
    """A chunk id carries its position, so an insertion renames everything below it.

    Nothing here changes an embedding input. Without the identity-keyed lookup, inserting one
    paragraph at the top of a document would re-embed every chunk under it — the same waste
    this path exists to remove, arriving through the other door.
    """
    await store.ensure_ready(fingerprint())
    original = chunk("chunk-a", position=0)
    await store.upsert([original], [spread(4, 1)])

    shifted = original.model_copy(update={"id": "chunk-a-moved", "position": 5})
    verdicts = await store.stored_vectors([shifted])

    assert verdicts[shifted.id].state is VectorState.READABLE
    assert list(verdicts[shifted.id].vector) == spread(4, 1)


async def test_a_row_whose_metadata_contradicts_the_chunk_beside_it_is_repaired(
    tmp_path: Path,
) -> None:
    """Identity metadata claiming a vector exists is not the same as a usable vector existing.

    Reached by writing the row through lancedb directly, because nothing in manicule can
    produce it — which is the point: what this guards against is a half-written directory, a
    restored backup, or an edited table, none of which asks permission. Two rows, one damaged
    each way, and the undamaged row beside them stays readable so the check cannot pass by
    condemning everything.

    **A row of the wrong dimension is deliberately not a case here.** The vector column is
    ``fixed_size_list<float32, dimension>``, so Lance cannot hold one; the classification
    exists for a backend whose column is not fixed-width, and
    :func:`~manicule.core.embedding.classify_stored_vector` is where it is tested.
    """
    directory = tmp_path / "vectors"
    store = LanceVectorStore(directory)
    await store.ensure_ready(fingerprint())
    intact, lying, unreadable = (chunk(f"chunk-{name}") for name in ("a", "b", "c"))
    await store.upsert([intact, lying, unreadable], [spread(4, index) for index in range(3)])
    await store.teardown()

    # `lying` keeps the identity that says it embeds its own current text, beside a chunk that
    # says it embeds something else. `unreadable` keeps everything except a decodable chunk.
    await _rewrite_chunk_json(directory, lying.id, lying.model_copy(update={"embed_text": "?"}))
    await _rewrite_chunk_json(directory, unreadable.id, None)

    reopened = LanceVectorStore(directory)
    verdicts = await reopened.stored_vectors([intact, lying, unreadable])

    assert verdicts[intact.id].state is VectorState.READABLE
    assert verdicts[lying.id].state is VectorState.CORRUPT, (
        "a row that says two different things about what it embedded is not evidence that a "
        "vector is current, and it is the one thing it must not be taken as"
    )
    assert verdicts[unreadable.id].state is VectorState.CORRUPT
    assert verdicts[lying.id].vector == ()
    assert verdicts[unreadable.id].vector == ()


async def _rewrite_chunk_json(directory: Path, chunk_id: str, replacement: Chunk | None) -> None:
    """Overwrite one row's chunk column, leaving its vector and identity alone.

    ``None`` writes something that is not a chunk at all.
    """
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(fingerprint()))
    encoded = "not json at all" if replacement is None else replacement.model_dump_json()
    await table.update(where=f"id = {quote(chunk_id)}", updates={"chunk_json": encoded})
    connection.close()


async def _drop_identity_column(directory: Path, embed: EmbedFingerprint) -> None:
    """Make a table look like one written before the identity column existed."""
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(embed))
    await table.drop_columns([IDENTITY_COLUMN])
    connection.close()


async def _drop_generation_columns(directory: Path, embed: EmbedFingerprint) -> None:
    """Recreate the schema shape written before publication generations existed."""
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(embed))
    await table.drop_columns([CHUNK_ID_COLUMN, PUBLICATION_COLUMN])
    connection.close()


# --- the ANN index lifecycle -----------------------------------------------------------------
#
# `docs/storage.md` §6.2 promises a transition — exhaustive below `ann_index_threshold`, an
# IVF-PQ index above it — and #261 found nothing in the product executing it. These are the
# other half of `tests/test_core_ann.py`: that a real LanceDB table reaches each state the rule
# describes, and answers correctly in every one of them.
#
# The threshold used here is 256 rather than the shipped 100 000, because 256 is the floor an
# 8-bit product quantizer can train at and therefore the smallest corpus that can exercise a
# real build. A test that stubbed the build would be a test of the classification, which
# `tests/test_core_ann.py` already is.

ANN_THRESHOLD: Final = MINIMUM_ANN_INDEX_THRESHOLD


def scattered(dimension: int, count: int, *, seed: int = 3) -> list[list[float]]:
    """``count`` deterministic vectors spread through the positive orthant.

    Not `spread`: one-hot vectors take only `dimension` distinct values, and a product
    quantizer trained on four distinct points is not a quantizer. These give the codebook
    something to actually partition.
    """
    rng = random.Random(seed)  # noqa: S311 - a fixture, seeded so a build is reproducible
    return [[rng.random() + 0.1 for _ in range(dimension)] for _ in range(count)]


async def _stocked(directory: Path, count: int, *, dimension: int = 4) -> LanceVectorStore:
    """A store holding ``count`` scattered vectors plus one exactly at the first axis.

    The last chunk is `"target"` and its vector is the axis a query can be aimed down, so
    "did search still find the right row" is a question with an unambiguous answer.
    """
    store = await prepared(directory, dimension)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(count)]
    vectors = scattered(dimension, count)
    await store.upsert([*chunks, chunk("target", position=count)], [*vectors, spread(dimension, 0)])
    return store


async def test_a_corpus_below_the_threshold_keeps_exact_search_and_is_offered_no_index(
    tmp_path: Path,
) -> None:
    """The state #261 says is correct, and the one the document already describes.

    Exhaustive search over a few tens of thousands of vectors is fast and *exact*. The
    maintenance boundary is asked here and declines, which is the behavior that keeps an
    early index — and the recall it would permanently cost — from being built by a cron job.
    """
    store = await _stocked(tmp_path / "vectors", 10)

    state = await store.ann_index_state(threshold=ANN_THRESHOLD)
    build = await store.build_ann_index(threshold=ANN_THRESHOLD)

    assert state.lifecycle is AnnLifecycle.EXHAUSTIVE
    assert state.exact
    assert not state.due
    assert state.index is None
    assert not build.built
    assert (await store.ann_index_state(threshold=ANN_THRESHOLD)).index is None
    await store.teardown()


async def test_crossing_the_threshold_makes_a_build_due_and_the_boundary_performs_it(
    tmp_path: Path,
) -> None:
    """#261's reproduction, end to end, at the pinned LanceDB version.

    The corpus crosses the documented transition point through the ordinary write path, the
    supported maintenance boundary is invoked, and the index metadata is then read back from
    LanceDB itself. Before this ticket every one of these assertions failed at the last step,
    because nothing scheduled a build at all.
    """
    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)
    due = await store.ann_index_state(threshold=ANN_THRESHOLD)

    build = await store.build_ann_index(threshold=ANN_THRESHOLD)

    assert due.lifecycle is AnnLifecycle.PENDING, "the corpus is past the documented threshold"
    assert due.exact, "pending is exact and slow, never approximate"
    assert build.built
    after = build.after
    assert after.lifecycle is AnnLifecycle.READY
    assert after.index is not None
    assert after.index.index_type == "IVF_PQ"
    assert after.index.distance_type == DISTANCE_METRIC
    assert after.index.num_partitions == partitions_for(after.rows)
    assert after.index.build_generation == 1
    assert after.index.coverage == 1.0
    assert not after.due
    await store.teardown()


async def test_search_answers_correctly_on_both_sides_of_the_build(tmp_path: Path) -> None:
    """The index changes what a query costs, not which row is nearest to it.

    Asserted on the one row whose answer is not a matter of degree: a vector lying exactly on
    the query's axis is the nearest neighbor at cosine 1.0 whether the search scanned every
    row or probed a partition. The *other* neighbors do move — that is what an approximate
    index is — so nothing here asserts their order, which would be asserting that IVF-PQ is
    not IVF-PQ.
    """
    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)
    query = spread(4, 0)
    before = await store.search(query, k=3)

    await store.build_ann_index(threshold=ANN_THRESHOLD)
    after = await store.search(query, k=3)

    assert before[0].chunk.id == "target"
    assert before[0].score == pytest.approx(1.0)
    assert after[0].chunk.id == "target", "an index must not lose the row it is nearest to"
    assert after[0].score == pytest.approx(1.0)
    assert len(after) == 3, "search stays available and still fills the requested k"
    await store.teardown()


async def test_vectors_published_after_a_build_are_searched_before_they_are_indexed(
    tmp_path: Path,
) -> None:
    """The refresh policy's premise: an uncovered row is latency owed, never a result missing.

    LanceDB scans the fragments the index does not cover and merges them into the ranked
    result, so a publication that lands the instant after a build is findable immediately.
    Were that not true the whole staleness policy would be a correctness bug rather than a
    performance one, and the index could not be refreshed on a schedule at all.
    """
    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)
    await store.build_ann_index(threshold=ANN_THRESHOLD)

    await store.upsert([chunk("published-after-the-build", position=1)], [spread(4, 1)])
    state = await store.ann_index_state(threshold=ANN_THRESHOLD)
    found = await store.search(spread(4, 1), k=1)

    assert state.index is not None
    assert state.index.unindexed_rows == 1
    assert state.index.coverage < 1.0
    assert state.lifecycle is AnnLifecycle.READY, "one uncovered row is not a stale index"
    assert found[0].chunk.id == "published-after-the-build"
    assert found[0].score == pytest.approx(1.0)
    await store.teardown()


async def test_an_uncovered_tail_past_the_threshold_goes_stale_and_a_rebuild_clears_it(
    tmp_path: Path,
) -> None:
    """The stated refresh policy, and the bounded action that answers it.

    The rebuild is one build: it trains at the current row count, takes the new partition
    count that implies, and increments the build generation so two status reads at identical
    coverage can still be told apart. The superseded index is dropped rather than accumulated —
    two indexes on one column is disk spent twice to answer one question.
    """
    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)
    first = await store.build_ann_index(threshold=ANN_THRESHOLD)
    await store.upsert(
        [chunk(f"late-{index}", position=index) for index in range(ANN_THRESHOLD)],
        scattered(4, ANN_THRESHOLD, seed=9),
    )

    stale = await store.ann_index_state(threshold=ANN_THRESHOLD)
    rebuilt = await store.build_ann_index(threshold=ANN_THRESHOLD)

    assert stale.lifecycle is AnnLifecycle.STALE
    assert stale.due
    assert rebuilt.built
    assert rebuilt.after.lifecycle is AnnLifecycle.READY
    assert rebuilt.after.index is not None
    assert rebuilt.after.index.build_generation == 2
    assert rebuilt.after.index.coverage == 1.0
    assert first.after.index is not None
    assert rebuilt.replaced == first.after.index.name
    await store.teardown()


async def test_a_dry_run_reports_the_build_it_would_do_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Every other boundary here plans by default, and an index build is the most expensive."""
    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)

    planned = await store.build_ann_index(threshold=ANN_THRESHOLD, dry_run=True)

    assert planned.dry_run
    assert not planned.built
    assert str(partitions_for(planned.before.rows)) in planned.detail
    assert (await store.ann_index_state(threshold=ANN_THRESHOLD)).index is None
    await store.teardown()


async def test_a_corpus_too_small_to_train_is_declined_rather_than_attempted(
    tmp_path: Path,
) -> None:
    """``--force`` over a handful of vectors would otherwise reach LanceDB and raise.

    Configuration cannot produce this — a threshold below the codebook size is refused at
    startup — but ``--force`` names a row count of its own, so the floor is checked where the
    build happens as well as where the setting is read.
    """
    store = await _stocked(tmp_path / "vectors", 10)

    declined = await store.build_ann_index(threshold=ANN_THRESHOLD, force=True)

    assert not declined.built
    assert str(MINIMUM_ANN_INDEX_THRESHOLD) in declined.detail
    assert (await store.ann_index_state(threshold=ANN_THRESHOLD)).index is None
    await store.teardown()


async def test_switching_the_threshold_off_stops_new_builds_without_dropping_an_old_one(
    tmp_path: Path,
) -> None:
    """A configuration change that destroyed a built index on its way past would be a surprise.

    ``disabled`` says new builds will not happen. What is already built is still the search
    path, so it is still what status describes.
    """
    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)
    await store.build_ann_index(threshold=ANN_THRESHOLD)

    off = await store.ann_index_state(threshold=0)
    declined = await store.build_ann_index(threshold=0)

    assert off.lifecycle is AnnLifecycle.READY
    assert off.index is not None
    assert not off.due
    assert not declined.built
    assert (await store.ann_index_state(threshold=0)).index is not None
    await store.teardown()


async def test_an_index_this_installation_did_not_build_is_refused_rather_than_replaced(
    tmp_path: Path,
) -> None:
    """Somebody made it deliberately, and the boundary does not know what for.

    Its partition count and build generation are unrecoverable — LanceDB records neither, and
    the name does not carry them — so it is reported with those fields empty rather than with
    a plausible guess, and every attempt to manage it refuses by name.
    """
    directory = tmp_path / "vectors"
    stocked = await _stocked(directory, ANN_THRESHOLD)
    await stocked.teardown()
    await _create_foreign_index(directory)
    # Reopened rather than reused, because that is the shape this actually arrives in: somebody
    # built the index against the directory, and the next process to start finds it there.
    store = await prepared(directory)

    state = await store.ann_index_state(threshold=ANN_THRESHOLD)

    assert state.index is not None
    assert not state.index.recognized
    assert state.index.num_partitions is None
    assert state.index.build_generation is None
    assert "did not build" in state.detail
    with pytest.raises(VectorStoreStateError, match="somebodys_own_index"):
        await store.build_ann_index(threshold=ANN_THRESHOLD, force=True)
    await store.teardown()


async def test_the_index_state_of_a_directory_with_no_vectors_is_an_answer_not_an_error(
    store: LanceVectorStore,
) -> None:
    """A fresh installation asking whether its index is current has a reasonable question."""
    state = await store.ann_index_state(threshold=ANN_THRESHOLD)

    assert state.rows == 0
    assert state.index is None
    assert state.exact
    assert not state.due


async def _create_foreign_index(directory: Path) -> None:
    """An index on the vector column under a name this project would never choose."""
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(fingerprint()))
    await table.create_index(
        VECTOR_COLUMN,
        config=IvfPq(distance_type=DISTANCE_METRIC, num_partitions=4, num_sub_vectors=1),
        name="somebodys_own_index",
    )
    connection.close()


async def test_a_published_handle_answers_about_an_index_it_has_not_created_yet(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """A fresh installation's status must not raise on the way to saying "nothing yet".

    The live handle follows SQLite's publication pointer and pins a generation for every
    operation, and on a first run there is no vectors directory to pin. Asking whether search
    is still exhaustive is a reasonable question before anything has been ingested — it is on
    the same payload as the document count, which is zero rather than an error.
    """
    root = data_dir / VECTORS_DIRNAME
    shutil.rmtree(root, ignore_errors=True)
    live = PublishedLanceVectorStore(root, engine)

    state = await live.ann_index_state(threshold=ANN_THRESHOLD)
    untouched = not root.exists()
    build = await live.build_ann_index(threshold=ANN_THRESHOLD)

    assert untouched, (
        "`index_status` is classified read-only, and opening a LanceDB connection creates the "
        "directory it points at: reporting that a store is empty must not build part of it"
    )
    assert state.rows == 0
    assert state.lifecycle is AnnLifecycle.EXHAUSTIVE
    assert state.generation is None, (
        "there is no published generation yet, and naming one would be inventing the very "
        "thing the field exists to report"
    )
    assert not build.built


async def test_a_build_that_fails_leaves_the_previous_index_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering promise, checked by breaking the build rather than by reading the code.

    The new index is created before the old one is dropped, so a build that raises has taken
    nothing away: the previous index is still there, still covers what it covered, and still
    answers. The opposite ordering would spend the failure on the search path — an installation
    that dropped back to a linear scan because a rebuild it never asked for went wrong.
    """
    from lancedb.table import AsyncTable  # noqa: PLC0415 - only this test replaces a method

    store = await _stocked(tmp_path / "vectors", ANN_THRESHOLD)
    first = await store.build_ann_index(threshold=ANN_THRESHOLD)
    await store.upsert(
        [chunk(f"late-{index}", position=index) for index in range(ANN_THRESHOLD)],
        scattered(4, ANN_THRESHOLD, seed=9),
    )

    async def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "synthetic build failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(AsyncTable, "create_index", refuse)
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        await store.build_ann_index(threshold=ANN_THRESHOLD)
    monkeypatch.undo()

    survived = await store.ann_index_state(threshold=ANN_THRESHOLD)
    found = await store.search(spread(4, 0), k=1)

    assert first.after.index is not None
    assert survived.index is not None
    assert survived.index.name == first.after.index.name, (
        "the failed build must not have taken the last known-good index with it"
    )
    assert survived.index.build_generation == 1
    assert survived.lifecycle is AnnLifecycle.STALE, (
        "still stale, because the refresh that would have cleared it did not happen"
    )
    assert found[0].chunk.id == "target", "and the corpus still answers from it"
    await store.teardown()
