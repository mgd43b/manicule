"""Moving a corpus between vector backends, and the five refusals that keep it honest.

The property under test throughout is that a migration is a *copy*: what the destination holds
afterwards is what the source held, component for component, with the identity and the checksum
the source recorded rather than ones derived on the way past. Everything else here is a refusal,
and each one exists because the alternative is a destination that looks fine and is not.

The pair is exercised for real — a LanceDB directory on disk and a Qdrant in this process —
rather than through fakes of either, because what is being checked is precisely that two
independently written stores agree about what a row is. A fake of either side would be a third
implementation of the row shape and would agree with itself.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

import pytest
from qdrant_client import AsyncQdrantClient, models

from manicule.core.embedding import VectorIntegrity, VectorState
from manicule.core.errors import FingerprintMismatchError, VectorMigrationError
from manicule.core.ids import vector_id
from manicule.storage.qdrant import (
    QdrantVectorStore,
    as_float32,
    collection_for,
    meta_collection_for,
    point_id_for,
)
from manicule.storage.vector_migration import migrate_vectors
from manicule.storage.vector_schema import (
    CHECKSUM_COLUMN,
    ID_COLUMN,
    IDENTITY_COLUMN,
    VECTOR_COLUMN,
)
from manicule.storage.vectors import LanceVectorStore
from tests.qdrant_support import TEST_COLLECTION_PREFIX, local_client, require_server
from tests.storage_helpers import fingerprint, make_chunk, make_document
from tests.vector_helpers import nudged, read_column, rewrite_row

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from manicule.core.content import Chunk
    from manicule.core.embedding import EmbedFingerprint

WORKSPACE = "migration-workspace"
DIMENSION = 8


def chunk(chunk_id: str, *, position: int = 0) -> Chunk:
    document = make_document(source="fs", source_id=f"doc-of-{chunk_id}")
    made = make_chunk(document, position, f"the text of {chunk_id}", lang="en")
    return made.model_copy(update={"id": chunk_id})


def realistic(index: int) -> list[float]:
    """A vector a real embedder might return: off unit length until storage normalizes it.

    Deliberately not one-hot. A one-hot vector is already exactly unit, so every rounding
    question a copy can get wrong — the source's normalization, the destination's, the float32
    narrowing between them — has the same answer for it, and a store that got all three wrong
    would still round-trip it perfectly.
    """
    return [((index + position) % 7) / 3.0 + 0.017 for position in range(DIMENSION)]


def exact(index: int) -> list[float]:
    """A one-hot vector, whose norm is exactly 1 and which therefore cannot be perturbed.

    For the assertions that must run in local mode and must not depend on its arithmetic: a
    backend re-normalizing this divides by exactly 1.0 and writes back what it was given, on
    every platform. :func:`realistic` is the right input everywhere the numbers are the subject
    and a server is doing the storing.
    """
    return [1.0 if position == index % DIMENSION else 0.0 for position in range(DIMENSION)]


@pytest.fixture
def client() -> AsyncQdrantClient:
    return local_client()


@pytest.fixture
def target(client: AsyncQdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client, workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX
    )


@pytest.fixture
async def opened() -> AsyncGenerator[list[LanceVectorStore]]:
    """Every embedded store a test opened, closed when it ends.

    A registry rather than a per-test ``teardown`` call, because the connection has to be
    released on the failing paths too — and most of the tests here fail on purpose. An
    unclosed LanceDB connection raises a ``ResourceWarning`` whenever the collector next runs,
    and ``filterwarnings = ["error"]`` turns that into a failure of whatever unrelated test
    happened to be executing at the time.
    """
    stores: list[LanceVectorStore] = []
    try:
        yield stores
    finally:
        for store in stores:
            await store.teardown()


async def seeded(
    directory: Path,
    chunks: Sequence[Chunk],
    opened: list[LanceVectorStore],
    *,
    vectors: Callable[[int], list[float]] = realistic,
) -> LanceVectorStore:
    """An embedded store holding one vector per chunk, as an ordinary ingest would leave it."""
    source = LanceVectorStore(directory)
    opened.append(source)
    await source.ensure_ready(fingerprint(DIMENSION))
    await source.upsert(list(chunks), [vectors(index) for index in range(len(chunks))])
    return source


async def rows_of(store: LanceVectorStore) -> dict[str, dict[str, Any]]:
    """Every row the source holds, keyed by physical id."""
    found: dict[str, dict[str, Any]] = {}
    async for page in store.inspection_pages(page_size=2):
        for row in page:
            found[str(row["id"])] = row
    return found


async def test_every_row_the_source_held_reaches_the_destination_with_its_own_fields(
    tmp_path: Path,
    target: QdrantVectorStore,
    client: AsyncQdrantClient,
    opened: list[LanceVectorStore],
) -> None:
    """Every row arrives, at the id it had, carrying the fields the source recorded.

    **The vector itself is checked against a server instead, and that split is the point.**
    Local mode re-normalizes on write in float64, which moves roughly one vector in eleven by a
    single ulp — measured at 47 of 500 random unit vectors here, against 0 of 500 on
    qdrant/qdrant:v1.19.1. So numeric fidelity is a fourth property only a real server can
    answer, alongside the three ``.github/workflows/ci.yml`` already names, and asserting it
    here would be asserting the arithmetic of a reimplementation — which fails by platform
    rather than by defect.

    What local mode *can* answer honestly is everything that is not the vector: the row count,
    the derived point id, and the payload. Those are carried as strings and integers and no
    backend rewrites them.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(5)]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    before = await rows_of(source)

    outcome = await migrate_vectors(source, target, generation="legacy", dry_run=False)

    assert outcome.copied == 5
    assert outcome.source_rows == 5
    assert await target.count() == 5
    for row_id, row in before.items():
        stored = (
            await client.retrieve(
                collection_name=target.storage_name(fingerprint(DIMENSION)),
                ids=[point_id_for(row_id)],
                with_payload=True,
            )
        )[0]
        payload = stored.payload or {}
        assert payload[ID_COLUMN] == row_id, (
            "the physical row id did not survive, so the relational authority's "
            "`chunks.vector_id` no longer names anything in the destination"
        )
        assert payload[IDENTITY_COLUMN] == row[IDENTITY_COLUMN]
        assert payload[CHECKSUM_COLUMN] == row[CHECKSUM_COLUMN], (
            "the checksum was rewritten rather than carried, which is what turns a drifted "
            "vector into a verified one"
        )


async def test_the_destination_reports_every_migrated_vector_as_numerically_intact(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """The check an operator is told to run afterwards, run here so the advice is known good.

    A carried checksum is only worth carrying if the destination can still recompute it from
    what it stored. Qdrant narrows a readback to float32 before believing it; a migration that
    wrote a vector the narrowing could not recover would leave a corpus reading as entirely
    corrupt, which is indistinguishable from real damage at the moment somebody looks.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(4)]
    source = await seeded(tmp_path / "vectors", chunks, opened)

    await migrate_vectors(source, target, generation="legacy", dry_run=False)

    # Presence, not verification. Recomputing a digest here would be recomputing it over local
    # mode's arithmetic, which re-normalizes and moves about one vector in eleven — so the
    # assertion would fail by platform rather than by defect. That every carried checksum still
    # *verifies* is asserted against a real server, in the transport test below.
    coverage = await target.checksum_coverage()
    assert coverage.rows == 4
    assert coverage.recorded == 4, (
        "a migrated row reached the destination without the checksum the source recorded, so "
        "nothing downstream can tell its numbers from any other numbers"
    )
    assert coverage.unverified == 0


async def test_a_migrated_vector_is_reused_rather_than_embedded_again(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """The reason anybody runs this, stated as the thing that must be true afterwards.

    Carrying ``embed_identity`` is what makes the destination answer ``READABLE`` for a chunk it
    has never embedded. A migration that moved every vector and dropped the identity would look
    complete and cost a corpus-sized forward pass on the next ingest, which is the entire
    expense it was run to avoid.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    # `exact` rather than `realistic`, because the verdict under test is decided partly by the
    # numerical check: a vector local mode had perturbed would read back CORRUPT rather than
    # READABLE, and this would fail for a reason that has nothing to do with identity.
    source = await seeded(tmp_path / "vectors", chunks, opened, vectors=exact)
    expected = await source.stored_vectors(chunks)

    await migrate_vectors(source, target, generation="legacy", dry_run=False)

    after = await target.stored_vectors(chunks)
    for chunk_id, verdict in after.items():
        assert verdict.state == expected[chunk_id].state, (
            f"the destination's reuse verdict for {chunk_id} differs from the source's, so the "
            f"next ingest embeds a chunk whose vector was just carried across"
        )


async def test_a_plan_copies_nothing_and_creates_nothing(
    tmp_path: Path,
    target: QdrantVectorStore,
    client: AsyncQdrantClient,
    opened: list[LanceVectorStore],
) -> None:
    """A dry run against a destination nobody has decided to use must leave no trace of itself.

    Creating the collection in order to count it would mean an operator who ran the plan, read
    it and chose not to migrate had a collection on a shared server anyway — with this
    installation's workspace digest in its name, which is exactly the residue
    ``reset-index`` then has to be told about.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    source = await seeded(tmp_path / "vectors", chunks, opened)

    outcome = await migrate_vectors(source, target, generation="legacy", dry_run=True)

    assert outcome.dry_run
    assert outcome.source_rows == 3
    assert outcome.copied == 0
    assert outcome.storage_name
    existing = {collection.name for collection in (await client.get_collections()).collections}
    assert existing == set(), (
        f"the plan left {sorted(existing)} behind on the server. A plan reports; it does not "
        f"provision."
    )


async def test_a_destination_that_already_holds_rows_is_refused(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """Merging would mix this corpus with rows whose provenance nothing here can establish."""
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    await target.ensure_ready(fingerprint(DIMENSION))
    await target.upsert([chunks[0]], [realistic(0)])

    with pytest.raises(VectorMigrationError, match="already holds 1 vector"):
        await migrate_vectors(source, target, generation="legacy", dry_run=False)


async def test_a_plan_refuses_a_populated_destination_before_reporting_a_copy(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """The refusal belongs to the plan too, or the plan describes a copy that cannot happen."""
    chunks = [chunk("chunk-0")]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    await target.ensure_ready(fingerprint(DIMENSION))
    await target.upsert([chunks[0]], [realistic(0)])

    with pytest.raises(VectorMigrationError, match="already holds"):
        await migrate_vectors(source, target, generation="legacy", dry_run=True)


async def test_a_destination_built_for_another_model_is_refused_by_the_plan(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """A size check passes two unrelated models of one width, and every answer after is noise.

    Checked by the plan and not only by the copy, so the refusal arrives while an operator is
    still deciding rather than halfway through the operation they decided on.
    """
    chunks = [chunk("chunk-0")]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    await target.ensure_ready(fingerprint(DIMENSION, model_id="a/different-model"))

    with pytest.raises(FingerprintMismatchError):
        await migrate_vectors(source, target, generation="legacy", dry_run=True)


async def test_a_source_that_records_no_fingerprint_is_refused(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """Nothing is established about what such vectors mean, and nothing here guesses."""
    empty = LanceVectorStore(tmp_path / "vectors")
    opened.append(empty)

    with pytest.raises(VectorMigrationError, match="records no embedding fingerprint"):
        await migrate_vectors(empty, target, generation="legacy", dry_run=True)


async def test_a_source_row_whose_checksum_no_longer_describes_it_stops_the_migration(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """The refusal that matters most, because the alternative is silent and permanent.

    The damage is one component moved to the adjacent float32 — finite, the same length, the
    same magnitude, and still ranking roughly where it did. Nothing but the checksum notices,
    which is the point: every other check a migration could make passes this row.

    Carried across, it arrives somewhere the original is no longer there to be compared
    against. Skipped instead, it leaves the destination quietly short of the corpus, and no
    vector store is ever asked whether it is complete. So the migration stops, names the chunk,
    and says what to run to see the whole picture.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    directory = tmp_path / "vectors"
    source = await seeded(directory, chunks, opened)
    embed = fingerprint(DIMENSION)
    held = await read_column(directory, embed, chunks[0].id, VECTOR_COLUMN)
    await source.teardown()
    # The vector moves and the checksum does not, which is the state a write cannot produce:
    # `upsert` rehashes what it stores, so a row damaged through the store is self-consistent
    # again. `rewrite_row` is how every other integrity suite here reaches the same state.
    await rewrite_row(directory, embed, chunks[0].id, {VECTOR_COLUMN: nudged(held)})

    reopened = LanceVectorStore(directory)
    opened.append(reopened)
    await reopened.open_existing()
    with pytest.raises(VectorMigrationError, match="stopped rather than carrying it"):
        await migrate_vectors(reopened, target, generation="legacy", dry_run=False)


async def test_progress_is_reported_while_a_copy_runs(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """A copy large enough to be worth doing is long enough that silence reads as a hang."""
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    said: list[str] = []

    await migrate_vectors(source, target, generation="legacy", report=said.append, dry_run=False)

    assert any("copying 3 vector(s)" in line for line in said), said
    assert any("copied 3 vector(s)" in line for line in said), said


async def test_a_plan_says_nothing_while_it_counts(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """Progress belongs to work. A plan that narrated itself would be noise in a pipeline."""
    chunks = [chunk("chunk-0")]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    said: list[str] = []

    await migrate_vectors(source, target, generation="legacy", report=said.append, dry_run=True)

    assert said == []


async def test_the_physical_row_id_survives_the_move(
    tmp_path: Path,
    target: QdrantVectorStore,
    client: AsyncQdrantClient,
    opened: list[LanceVectorStore],
) -> None:
    """The id is what a tombstone sweep deletes by, so a copy that renamed rows would orphan them.

    The destination derives its point id from the physical row id and keeps the row id in the
    payload, because the derivation does not invert. Both halves are checked: a store that kept
    only the UUID would leave nothing for the relational authority's ``chunks.vector_id`` to
    match against.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    source = await seeded(tmp_path / "vectors", chunks, opened)

    await migrate_vectors(source, target, generation="legacy", dry_run=False)

    for stored in chunks:
        expected = vector_id("legacy", stored.id)
        found = await client.retrieve(
            collection_name=target.storage_name(fingerprint(DIMENSION)),
            ids=[point_id_for(expected)],
            with_payload=True,
        )
        assert found, f"no point at the id derived from {expected}"
        assert (found[0].payload or {})["id"] == expected


# --- against a real server ------------------------------------------------------------------
#
# Local mode is a Python reimplementation that hands back the list it was given, so every
# assertion above about a vector surviving the move is made against a store that could not have
# damaged it. The transport is where a migration's one property actually gets decided: Qdrant's
# REST serialization writes a stored `float32` as JSON decimal text, and parsing that yields a
# `float64` a few parts in 10^9 away. A copy that did not survive that would leave every
# migrated row failing its carried checksum — a healthy corpus reading as entirely corrupt,
# discovered at the moment somebody runs the verification they were told to run.


@pytest.fixture
async def server_target() -> AsyncGenerator[QdrantVectorStore]:
    """A store on a real Qdrant, in a workspace of its own, cleaned up afterwards."""
    client = AsyncQdrantClient(url=require_server(), timeout=30)
    store = QdrantVectorStore(
        client,
        workspace_id=f"{WORKSPACE}-{uuid.uuid4()}",
        collection_prefix=TEST_COLLECTION_PREFIX,
    )
    try:
        yield store
    finally:
        collection = collection_for(
            TEST_COLLECTION_PREFIX, store.workspace_id, fingerprint(DIMENSION)
        )
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
        meta = meta_collection_for(TEST_COLLECTION_PREFIX)
        if await client.collection_exists(meta):
            await client.delete(
                collection_name=meta,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="workspace_id",
                                match=models.MatchValue(value=store.workspace_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            if (await client.count(meta, exact=True)).count == 0:
                await client.delete_collection(meta)
        await client.close()


@pytest.mark.contract
async def test_a_migrated_vector_survives_the_transport_intact(
    tmp_path: Path, server_target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """Every carried checksum still verifies over a socket, which is what the advice rests on.

    The vectors are deliberately not unit-length one-hots: those round-trip through any
    serialization, so a store that mishandled the rounding would pass. These are what an
    embedder returns, normalized on the way into Lance and narrowed on the way back out of
    Qdrant, and the digest is the same at both ends or the migration did not preserve anything.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(6)]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    before = await rows_of(source)

    outcome = await migrate_vectors(source, server_target, generation="legacy", dry_run=False)

    assert outcome.copied == 6
    coverage = await server_target.checksum_coverage(recompute=True)
    assert coverage.rows == 6
    assert coverage.verified == 6, (
        f"{coverage.failed} migrated row(s) failed their carried checksum over the transport, "
        f"so the copy is not the bit-exact move the whole operation claims to be"
    )
    verdicts = await server_target.stored_vectors(chunks)
    assert all(v.integrity is VectorIntegrity.VERIFIED for v in verdicts.values())
    assert all(v.state is VectorState.READABLE for v in verdicts.values()), (
        "a migrated row read back as not reusable over a socket, so the next ingest embeds a "
        "corpus that was just carried across to avoid exactly that"
    )

    # Component for component, which only a server can be asked. The readback is narrowed to
    # float32 first, because that is the representation the checksum is defined over and the
    # transport serializes it as decimal text; comparing the parsed float64 would fail on a
    # store that had done nothing wrong.
    assert len(before) == 6
    verifier = AsyncQdrantClient(url=require_server(), timeout=30)
    try:
        for row_id, row in before.items():
            stored = (
                await verifier.retrieve(
                    collection_name=server_target.storage_name(fingerprint(DIMENSION)),
                    ids=[point_id_for(row_id)],
                    with_vectors=True,
                )
            )[0]
            # Narrowed the way the store narrows it: a point's `vector` is typed as any of
            # several shapes because a collection may hold named vectors, and this one does not.
            returned = cast("list[float]", stored.vector or [])
            assert as_float32(returned) == [float(value) for value in row[VECTOR_COLUMN]], (
                f"the vector for {row_id} changed on the way across; a migration that perturbs "
                f"the numbers is a re-embed with extra steps"
            )
    finally:
        await verifier.close()


async def test_a_destination_whose_identity_record_is_gone_is_still_refused(
    tmp_path: Path,
    target: QdrantVectorStore,
    client: AsyncQdrantClient,
    opened: list[LanceVectorStore],
) -> None:
    """The populated-destination refusal has to survive the destination forgetting itself.

    ``count`` resolves the collection through the fingerprint record, so a destination whose
    record has gone — a reset that got halfway, a point removed by hand — answers zero while
    its storage is still full. Asking that question would satisfy the refusal with exactly the
    case it exists to catch, and rows of unknown provenance would be merged into the corpus
    being moved. ``rows_in_space`` names the collection from the fingerprint instead.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    source = await seeded(tmp_path / "vectors", chunks, opened)
    embed = fingerprint(DIMENSION)
    await target.ensure_ready(embed)
    await target.upsert([chunks[0]], [realistic(0)])
    # Remove the workspace's fingerprint record, leaving its collection populated.
    await client.delete(
        collection_name=f"{TEST_COLLECTION_PREFIX}_meta",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id", match=models.MatchValue(value=WORKSPACE)
                    )
                ]
            )
        ),
        wait=True,
    )
    assert await target.count() == 0, "the premise: the store now under-reports itself"

    with pytest.raises(VectorMigrationError, match="already holds 1 vector"):
        await migrate_vectors(source, target, generation="legacy", dry_run=True)


async def test_a_copy_that_lands_short_is_refused_rather_than_reported(
    tmp_path: Path, target: QdrantVectorStore, opened: list[LanceVectorStore]
) -> None:
    """The destination is counted at the end, not trusted from the writes.

    ``copied`` says what was asked for and the count says what is there. A backend that
    coalesced two rows, or a source that changed under the read, shows up in the difference and
    nowhere else — and a migration reporting success over a short destination is discovered by
    a search that quietly returns less.
    """
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    source = await seeded(tmp_path / "vectors", chunks, opened)

    async def swallow_one(rows: Sequence[Mapping[str, Any]]) -> int:
        await QdrantVectorStore.adopt_rows(target, list(rows)[1:])
        return len(rows)

    with pytest.raises(VectorMigrationError, match="did not reproduce the source"):
        await migrate_vectors(
            source,
            _SwallowingTarget(target, swallow_one),  # pyright: ignore[reportArgumentType]
            generation="legacy",
            dry_run=False,
        )


class _SwallowingTarget:
    """A destination that reports writing everything and keeps one row less."""

    def __init__(
        self,
        inner: QdrantVectorStore,
        adopt: Callable[[Sequence[Mapping[str, Any]]], Awaitable[int]],
    ) -> None:
        self._inner = inner
        self._adopt = adopt

    def storage_name(self, embed: EmbedFingerprint) -> str:
        return self._inner.storage_name(embed)

    async def fingerprint(self) -> EmbedFingerprint | None:
        return await self._inner.fingerprint()

    async def ensure_ready(self, embed: EmbedFingerprint, **kwargs: object) -> None:
        del kwargs
        await self._inner.ensure_ready(embed)

    async def rows_in_space(self, embed: EmbedFingerprint) -> int:
        return await self._inner.rows_in_space(embed)

    async def count(self) -> int:
        return await self._inner.count()

    async def adopt_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return await self._adopt(rows)
