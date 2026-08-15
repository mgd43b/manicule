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
from typing import TYPE_CHECKING

import lancedb
import pytest

from manicule.core.anchors import HeadingAnchor, Unlocated
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
from manicule.storage.vectors import (
    EXEMPT_FILTER_FIELDS,
    IDENTITY_COLUMN,
    META_TABLE,
    LanceVectorStore,
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
