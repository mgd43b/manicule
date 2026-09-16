"""The four things a vector migration refuses before it reads a row, and the pointer it repairs.

Each refusal is here because the alternative is not an error but a plausible-looking success.
A migration that ran against the wrong configuration, an unpublished generation, or a missing
directory would report a faithful copy of something nobody asked it to copy — and a migration
that finished and left the relational pointer naming a directory the new backend has never
heard of would leave every report about the index quietly wrong until the next ingest.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import pytest
from sqlalchemy import insert, select, update

from manicule.app.runtime import (
    _Maintenance,  # pyright: ignore[reportPrivateUsage] - the unit under test
)
from manicule.config.settings import Settings
from manicule.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    VectorMigrationError,
    VectorStoreStateError,
)
from manicule.core.rebuild import RebuildState
from manicule.ingest.reembed import ReembedState
from manicule.storage import models
from manicule.storage import vectors as vectors_module
from manicule.storage.config import QDRANT_VECTOR_STORE_NAME
from manicule.storage.qdrant import QdrantVectorStore
from manicule.storage.vector_paths import workspace_vector_directory
from manicule.storage.vector_schema import space_name
from manicule.storage.vectors import LanceVectorStore
from tests.qdrant_support import TEST_COLLECTION_PREFIX, local_client
from tests.storage_helpers import fingerprint, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.app.runtime import Runtime
    from manicule.core.content import Chunk

WORKSPACE = "default"
DIMENSION = 8


class _MigrationRuntime:
    """Exactly what ``migrate_vectors`` resolves from a runtime, and nothing else.

    A fake of the four members rather than a built container, because the cases under test are
    configurations the container would refuse to assemble — a Qdrant name over a store that
    cannot adopt, most of all, which is what a third-party plugin registered under the
    configured name would produce.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        vectors: object,
        directory: Path,
        *,
        vector_db: str = QDRANT_VECTOR_STORE_NAME,
    ) -> None:
        self.settings = Settings.model_validate(
            {"storage": {"vector_db": vector_db, "vector_db_url": "http://127.0.0.1:6333"}}
            if vector_db == QDRANT_VECTOR_STORE_NAME
            else {"storage": {"vector_db": vector_db}}
        )
        self.workspace = WORKSPACE
        self._engine = engine
        self._vectors = vectors
        self._directory = directory
        self.guard_depth = 0
        """How many derived-mutation guards this runtime is currently inside.

        Recorded so a test can ask *when* the guard was held rather than whether it was taken
        at all — which is the whole of the difference between serializing a migration and
        serializing only the half of it that writes.
        """

    def require_engine(self) -> AsyncEngine:
        return self._engine

    async def vectors(self) -> object:
        return self._vectors

    async def vector_directory(self) -> Path:
        return self._directory

    @asynccontextmanager
    async def derived_mutation_guard(self) -> AsyncGenerator[None]:
        self.guard_depth += 1
        try:
            yield
        finally:
            self.guard_depth -= 1


class _NotAnAdoptingStore:
    """A vector store with no adoption path, as a third-party backend may legitimately be."""

    async def ensure_ready(self, *_args: object, **_kwargs: object) -> None:
        return

    async def count(self) -> int:
        return 0


def chunk(chunk_id: str, *, position: int = 0) -> Chunk:
    document = make_document(source="fs", source_id=f"doc-of-{chunk_id}")
    made = make_chunk(document, position, f"the text of {chunk_id}", lang="en")
    return made.model_copy(update={"id": chunk_id})


async def seed_lance(directory: Path, chunks: Sequence[Chunk]) -> None:
    """Leave a published embedded index at ``directory``, as an ingest would.

    Closed before returning, deliberately: the store under test opens the directory itself, and
    an unclosed connection here would raise a ``ResourceWarning`` on the next collection — which
    ``filterwarnings = ["error"]`` turns into a failure of whichever test is running then.
    """
    source = LanceVectorStore(directory)
    await source.ensure_ready(fingerprint(DIMENSION))
    await source.upsert(
        list(chunks),
        [
            [((index + column) % 5) / 2.0 + 0.11 for column in range(DIMENSION)]
            for index in range(len(chunks))
        ],
    )
    await source.teardown()


async def set_index_state(engine: AsyncEngine, **values: Any) -> None:
    """Create or move this workspace's index-state row."""
    async with engine.begin() as connection:
        existing = (
            await connection.execute(
                select(models.IndexState.workspace_id).where(
                    models.IndexState.workspace_id == WORKSPACE
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            await connection.execute(
                insert(models.IndexState).values(workspace_id=WORKSPACE, **values)
            )
            return
        await connection.execute(
            update(models.IndexState)
            .where(models.IndexState.workspace_id == WORKSPACE)
            .values(**values)
        )


@pytest.fixture
def target() -> QdrantVectorStore:
    return QdrantVectorStore(
        local_client(), workspace_id=WORKSPACE, collection_prefix=TEST_COLLECTION_PREFIX
    )


async def test_migrating_while_configured_for_the_embedded_store_is_refused(
    store: object, engine: AsyncEngine, tmp_path: Path, target: QdrantVectorStore
) -> None:
    """There is no destination: the source and the configured store would be the same thing."""
    del store
    runtime = _MigrationRuntime(engine, target, tmp_path, vector_db="lancedb")
    with pytest.raises(ConfigError, match="the store this reads"):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()


async def test_a_backend_that_cannot_adopt_rows_is_refused_by_capability_not_by_name(
    store: object, engine: AsyncEngine, tmp_path: Path
) -> None:
    """The refusal a backend added later inherits without anybody editing a list.

    Asked of the object through ``AdoptingVectorStore``, so a store registered under a
    configured name that cannot take another backend's rows is told so — rather than reaching
    the copy and failing partway with an ``AttributeError`` naming a method nobody called.
    """
    del store
    runtime = _MigrationRuntime(engine, _NotAnAdoptingStore(), tmp_path)
    with pytest.raises(ConfigError, match="cannot adopt rows"):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()


async def test_an_unpublished_reembed_run_refuses_the_migration(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """An unpublished generation is a moment, not a thing to copy.

    Its rows carry replay lineage in columns a destination is under no obligation to have a
    home for, so moving them can drop the provenance a resumed replay reads — and it would be
    discovered by the replay, long after the directory had been removed.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    await set_index_state(engine, vector_table=space_name(fingerprint(DIMENSION)))
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.ReembedRunRecord).values(
                id="run-1",
                workspace_id=WORKSPACE,
                commitment_json="{}",
                state=ReembedState.BUILDING.value,
                checkpoint_json="{}",
            )
        )
    runtime = _MigrationRuntime(engine, target, root)

    with pytest.raises(VectorMigrationError, match="1 re-embed"):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()


async def test_a_published_reembed_run_does_not_refuse_the_migration(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """The other half, so the guard is a fence rather than a wall.

    A settled run left its generation published and its lineage inert. Refusing here too would
    mean an installation that had ever re-embedded could never migrate.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    await set_index_state(engine, vector_table=space_name(fingerprint(DIMENSION)))
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.ReembedRunRecord).values(
                id="run-1",
                workspace_id=WORKSPACE,
                commitment_json="{}",
                state=ReembedState.PUBLISHED.value,
                checkpoint_json="{}",
            )
        )
    runtime = _MigrationRuntime(engine, target, root)

    outcome = await _Maintenance(cast("Runtime", runtime)).migrate_vectors()
    assert outcome.source_rows == 1


async def test_an_absent_embedded_index_is_refused_rather_than_reported_as_empty(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """A workspace that never indexed has nothing to move, and saying "copied 0" implies it did."""
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await set_index_state(engine, vector_table=None)
    runtime = _MigrationRuntime(engine, target, root)

    with pytest.raises(ConfigError, match="no embedded vector index"):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()


async def test_a_completed_migration_retargets_a_generation_pointer(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """The cross-backend hazard, fixed by the operation that creates it.

    A ``reembed-…`` pointer names a directory under the embedded root. The destination has no
    generations at all, so left alone the pointer survives the move and goes on naming
    something the configured backend has never heard of — which ``doctor`` reports, the backup
    manifest records, and the reset path already refuses with "the corpus was moved between
    backends with a rebuild still pending".
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    generation = root / "generations" / "reembed-run-1"
    await seed_lance(generation, [chunk("chunk-0"), chunk("chunk-1", position=1)])
    await set_index_state(engine, vector_table="reembed-run-1")
    runtime = _MigrationRuntime(engine, target, root)

    outcome = await _Maintenance(cast("Runtime", runtime)).migrate_vectors(dry_run=False)

    assert outcome.generation == "reembed-run-1"
    assert outcome.copied == 2
    async with engine.connect() as connection:
        pointer = (
            await connection.execute(
                select(models.IndexState.vector_table).where(
                    models.IndexState.workspace_id == WORKSPACE
                )
            )
        ).scalar_one()
    assert pointer == space_name(fingerprint(DIMENSION)), (
        "the generation pointer still names an embedded directory after the corpus moved off it"
    )


async def test_a_plan_leaves_the_generation_pointer_alone(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """A plan that repaired the pointer would be a plan that changed the installation."""
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    generation = root / "generations" / "reembed-run-1"
    await seed_lance(generation, [chunk("chunk-0")])
    await set_index_state(engine, vector_table="reembed-run-1")
    runtime = _MigrationRuntime(engine, target, root)

    await _Maintenance(cast("Runtime", runtime)).migrate_vectors()

    async with engine.connect() as connection:
        pointer = (
            await connection.execute(
                select(models.IndexState.vector_table).where(
                    models.IndexState.workspace_id == WORKSPACE
                )
            )
        ).scalar_one()
    assert pointer == "reembed-run-1"


async def test_an_unpublished_rebuild_refuses_the_migration(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """The other in-flight generation, and the branch whose query shape differs.

    A re-embed run records its state in a text column and a rebuild records its in a typed enum
    one, so the two halves of this guard are compared against different things — strings on one
    side, enum members on the other. Exercising only the re-embed half would leave a query that
    silently matched nothing looking exactly like a workspace with no rebuilds in flight.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    await set_index_state(engine, vector_table=space_name(fingerprint(DIMENSION)))
    async with engine.begin() as connection:
        # `acquisition_runs` carries a composite foreign key to `(connectors.id,
        # connectors.workspace_id)`, so the connector has to exist before the run does.
        await connection.execute(
            insert(models.Connector).values(
                id="fs", workspace_id=WORKSPACE, name="fs", type="filesystem", config={}
            )
        )
        await connection.execute(
            insert(models.AcquisitionRun).values(
                id="run-a", workspace_id=WORKSPACE, connector_id="fs", connector_name="fs"
            )
        )
        await connection.execute(
            insert(models.DerivedGeneration).values(
                id="gen-1",
                workspace_id=WORKSPACE,
                snapshot_run_id="run-a",
                snapshot_membership_hash="hash",
                expected_item_count=1,
                target_digest="digest",
                publication_identity_digest="identity",
                target={},
                state=RebuildState.BUILDING,
            )
        )
    runtime = _MigrationRuntime(engine, target, root)

    with pytest.raises(VectorMigrationError, match="1 rebuild"):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()


async def test_a_directory_holding_another_models_vectors_is_refused(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """The physical fingerprint and the recorded one have to agree before anything moves.

    They do on any installation nothing has been done to by hand. The case this refuses is a
    vectors directory swapped in from elsewhere, or restored from a backup taken against a
    different model: copying then moves model A's vectors into the destination while the
    relational authority goes on naming model B, and nothing notices until the next ingest is
    refused against a corpus that has already moved.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    # The directory holds `test/model`; the authority was told it holds another.
    other = fingerprint(DIMENSION, model_id="some/other-model")
    await set_index_state(engine, embed_fingerprint=other.model_dump_json())
    runtime = _MigrationRuntime(engine, target, root)

    with pytest.raises(FingerprintMismatchError):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()


async def test_a_workspace_that_recorded_no_fingerprint_is_not_refused(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """Nothing recorded is nothing to disagree with, so the check is a fence and not a wall."""
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    await set_index_state(engine, embed_fingerprint=None)
    runtime = _MigrationRuntime(engine, target, root)

    outcome = await _Maintenance(cast("Runtime", runtime)).migrate_vectors()
    assert outcome.source_rows == 1


async def test_a_source_that_cannot_be_opened_does_not_leak_its_connection(
    store: object,
    engine: AsyncEngine,
    data_dir: Path,
    target: QdrantVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal while opening the source closes the handle it had already taken.

    ``open_existing`` holds the connection before it validates what is behind it, and this
    refusal happens before the caller's ``finally`` — so the leak is on the path that *fails*,
    which is the path least likely to be exercised by hand. An unclosed LanceDB connection
    becomes a ``ResourceWarning`` whenever the collector next runs, and Python attributes an
    unhandled one to whatever is executing then rather than to this: under
    ``filterwarnings = ["error"]`` that fails an unrelated test, intermittently and under load.
    This branch has already been bitten twice by that shape, in another subsystem.

    Asserted on the store's own handle rather than on a warning, because a warning is raised
    when the collector decides to and a test that waited for one would be the flake it is
    checking for.
    """
    del store
    opened: list[LanceVectorStore] = []
    real = vectors_module.LanceVectorStore

    def recording(directory: Path) -> LanceVectorStore:
        made = real(directory)
        opened.append(made)
        return made

    monkeypatch.setattr(vectors_module, "LanceVectorStore", recording)
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    # A directory that exists and holds no fingerprint metadata: `open_existing` connects, and
    # then refuses what it finds.
    root.mkdir(parents=True)
    await set_index_state(engine, vector_table=space_name(fingerprint(DIMENSION)))
    runtime = _MigrationRuntime(engine, target, root)

    with pytest.raises(VectorStoreStateError):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors()

    assert opened, "the refusal happened before a source handle was built; test premise is gone"
    for handle in opened:
        # Reached past the store deliberately: whether a connection is held has no public
        # surface, and the alternative — waiting for the ResourceWarning it eventually raises —
        # is the flake this is checking for.
        held = handle._connection  # pyright: ignore[reportPrivateUsage]
        assert held is None, (
            "the source store was left holding a LanceDB connection after refusing to open. "
            "Nothing closes it, and the ResourceWarning it eventually raises is attributed to "
            "whatever unrelated work is running at the time"
        )


async def test_the_preflight_reads_happen_inside_the_derived_mutation_guard(
    store: object,
    engine: AsyncEngine,
    data_dir: Path,
    target: QdrantVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both reads that decide whether a migration may proceed are only worth what they are
    worth while nothing else can move them.

    Guarding the copy alone leaves two windows, and the second is the expensive one: a
    generation published between the pointer read and the guard leaves the migration holding
    the superseded source, and ``_retarget_index_state`` then overwrites the newly published
    pointer with the old space. The data directory's writer lock closes neither, because it
    excludes other *processes* and both tasks are inside one served one.

    Asserted on where the guard is held rather than by racing two tasks, because a test that
    tried to lose the race would pass whenever it happened to win it.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    await set_index_state(engine, vector_table=space_name(fingerprint(DIMENSION)))
    runtime = _MigrationRuntime(engine, target, root)
    depths: dict[str, int] = {}
    # Spied rather than reimplemented, so the wrapper cannot drift from the real preflight.
    real_preflight = _Maintenance._refuse_migration_in_flight  # pyright: ignore[reportPrivateUsage]

    async def recording(self: _Maintenance) -> None:
        depths["preflight"] = runtime.guard_depth
        await real_preflight(self)

    monkeypatch.setattr(_Maintenance, "_refuse_migration_in_flight", recording)

    await _Maintenance(cast("Runtime", runtime)).migrate_vectors(dry_run=False)
    assert depths["preflight"] == 1, (
        "the in-flight check and the pointer read ran outside the guard, so a re-embed starting "
        "or publishing between them and the copy would be invisible to both"
    )

    depths.clear()
    await target.reset_storage()
    await _Maintenance(cast("Runtime", runtime)).migrate_vectors(dry_run=True)
    assert depths["preflight"] == 0, (
        "a plan took the guard, so a diagnostic somebody runs to decide whether to migrate now "
        "waits on the work they are deciding about"
    )


async def test_a_workspace_with_no_index_state_row_is_refused_before_anything_is_copied(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """Nothing to retarget means nothing to report, so the refusal comes before the copy.

    A row recording no fingerprint and no row at all were indistinguishable through
    ``scalar_one_or_none``, and only the first is safe to proceed on. Without the row, the
    retarget's ``UPDATE`` matches nothing — which is not an error to SQL — so the migration
    would copy the whole corpus, report success, and leave the destination with no
    ``vector_table`` pointer for ``doctor`` and the backup manifest to name.

    Asserted on the destination staying empty, because "refused" and "refused before doing the
    expensive half" are different claims and only the second is worth making.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0"), chunk("chunk-1", position=1)])
    # Deliberately no index-state row for this workspace.
    runtime = _MigrationRuntime(engine, target, root)

    with pytest.raises(VectorMigrationError, match="no index-state row"):
        await _Maintenance(cast("Runtime", runtime)).migrate_vectors(dry_run=False)

    assert await target.count() == 0, "the corpus was copied before the refusal"


async def test_a_row_recording_no_fingerprint_still_migrates(
    store: object, engine: AsyncEngine, data_dir: Path, target: QdrantVectorStore
) -> None:
    """The distinction the refusal above rests on, asserted from the other side.

    A workspace that has an index-state row but has never recorded a fingerprint has nothing
    for the physical one to disagree with, and something to retarget when the copy finishes.
    Refusing it too would turn a fence into a wall.
    """
    del store
    root = workspace_vector_directory(data_dir / "vectors", WORKSPACE)
    await seed_lance(root, [chunk("chunk-0")])
    await set_index_state(engine, embed_fingerprint=None)
    runtime = _MigrationRuntime(engine, target, root)

    outcome = await _Maintenance(cast("Runtime", runtime)).migrate_vectors(dry_run=False)

    assert outcome.copied == 1
    async with engine.connect() as connection:
        pointer = (
            await connection.execute(
                select(models.IndexState.vector_table).where(
                    models.IndexState.workspace_id == WORKSPACE
                )
            )
        ).scalar_one()
    assert pointer == space_name(fingerprint(DIMENSION))
