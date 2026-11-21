"""Authoring while a sync is running: the shipped runtime, the real pipeline, SQLite and Lance.

``document_create`` promises that on success the file, the indexed document and its collection
membership are durable and immediately searchable. These tests hold that promise against a
background sync of another source that is parked **inside its fetch**, which is the shape of a
real sync waiting on a server and the one that used to stall every write in the process.

Only the embedder and the chunker are doubles. Every pause is an event, so each assertion is
about arrival order and none can pass by being slow or fail on a loaded machine; the gate of
every test is opened in a ``finally`` so a failing assertion cannot leave a run hanging.
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, override

import pytest

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import AuthoringSettings, ConnectorSettings, Settings
from manicule.container import keys
from manicule.core.errors import PolicyError
from manicule.core.ids import document_id
from manicule.plugins.registry import discover
from tests.fakes import HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from pathlib import Path

WORKSPACE = "alpha"
AUTHORED = "authored"
MAIL = "mail"
WIKI = "wiki"
PATIENCE_S = 20.0
"""How long a test waits for something that must happen. A failure to finish inside it is the
stall itself, so it is reported as a failure naming what was waiting rather than as a hang."""


class _GatedConnector(fakes.DictConnector):
    """A connector whose fetch parks until the test lets it go: a source waiting on a server."""

    def __init__(self, documents: dict[str, str], *, name: str, gate: fakes.Gate) -> None:
        super().__init__(documents, name=name)
        self.gate = gate
        for source_id in documents:
            self.media_types[source_id] = "text/plain"

    @override
    async def fetch(self, ref: Any) -> Any:
        raw = await super().fetch(ref)
        await self.gate.pass_through()
        return raw


def _runtime(environment: Path, *, collections: tuple[str, ...] = ("memories",)) -> Runtime:
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415 - fakes, local to this harness

    found = discover()
    bound = found.registry.bind("test")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    authored = environment / AUTHORED
    authored.mkdir(exist_ok=True)
    settings = Settings(
        data_dir=environment / "data",
        workspace=WORKSPACE,
        connectors={
            AUTHORED: ConnectorSettings(type="filesystem", options={"root": str(authored)})
        },
        authoring=AuthoringSettings(source=AUTHORED, collections=collections),
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block"},  # pyright: ignore[reportArgumentType]
        # A run parked in its fetch is canceled by waiting out this grace, so a test that
        # cancels one should not wait out the shipped default.
        ingest={"shutdown_grace_s": 0.5},  # pyright: ignore[reportArgumentType]
    )
    return Runtime(settings, discovery=found)


@pytest.fixture
async def opened(manicule_environment: Path) -> AsyncIterator[Runtime]:
    async with _runtime(manicule_environment, collections=("memories", "archive")) as running:
        yield running


@pytest.fixture
async def service(opened: Runtime) -> ApplicationService:
    service = ApplicationService(opened)
    await service.collection_create("memories")
    await service.collection_create("archive")
    # Warm, so every test starts from the steady state of a served process: the pipeline built
    # once by the first write, which is the case the stall was measured in.
    await service.document_create(collection="memories", slug="warm", body="warm up the index")
    return service


@asynccontextmanager
async def _backgrounds(
    runtime: Runtime, *sources: tuple[str, fakes.Gate]
) -> AsyncGenerator[list[asyncio.Task[Any]]]:
    """Syncs of the named sources, in order, each parked in its fetch on its own gate.

    Every gate is opened before any task is awaited: a later source queues behind an earlier
    one's slot, so waiting for it while the earlier gate is still shut would hang the teardown.
    """
    tasks: list[asyncio.Task[Any]] = []
    for name, gate in sources:
        connector = _GatedConnector({"page": f"{name} body"}, name=name, gate=gate)

        async def sync(connector: _GatedConnector = connector) -> Any:
            async with runtime.leased_pipeline(connector.name) as pipeline:
                return await pipeline.run(connector)

        tasks.append(asyncio.create_task(sync(), name=f"sync:{name}"))
    try:
        yield tasks
    finally:
        for _, gate in sources:
            gate.open()
        await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def _background(
    runtime: Runtime, name: str, gate: fakes.Gate
) -> AsyncGenerator[asyncio.Task[Any]]:
    """A sync of ``name`` running the way the scheduler runs one, parked in its fetch."""
    async with _backgrounds(runtime, (name, gate)) as tasks:
        yield tasks[0]


async def _within(awaitable: Any, doing: str) -> Any:
    try:
        async with asyncio.timeout(PATIENCE_S):
            return await awaitable
    except TimeoutError:
        pytest.fail(f"{doing} did not finish while the other source was parked")


class _GatedRun:
    """A run that is not parked, to prove a slot is free."""

    def __init__(self, runtime: Runtime, name: str) -> None:
        self._runtime = runtime
        self._connector = _GatedConnector(
            {"page": f"{name} body"}, name=name, gate=fakes.Gate(opened=True)
        )

    async def run(self) -> Any:
        async with self._runtime.leased_pipeline(self._connector.name) as pipeline:
            return await pipeline.run(self._connector)


# --- the contract, held against a parked sync -----------------------------------------------


async def test_a_created_document_is_searchable_while_another_source_is_parked_in_its_fetch(
    opened: Runtime, service: ApplicationService
) -> None:
    gate = fakes.Gate()
    async with _background(opened, MAIL, gate):
        await gate.wait_for(1)

        created = await _within(
            service.document_create(
                collection="memories", slug="dawn", body="the lighthouse keeper rises at dawn"
            ),
            "the write",
        )
        found = await _within(
            service.search("lighthouse keeper", collections=["memories"], limit=5), "the search"
        )

        assert gate.inside == 1, "the other source really was parked in its fetch throughout"

    assert created.indexed
    assert created.member
    assert created.chunks >= 1
    assert created.document_id in {hit.document_id for hit in found.hits}, (
        "success means immediately searchable, with the other source's fetch still unreleased"
    )


async def test_a_second_background_source_cannot_take_the_neighbor_slot(
    opened: Runtime, service: ApplicationService
) -> None:
    """Two parked background runs would have filled the barrier and put the stall back."""
    mail_gate, wiki_gate = fakes.Gate(), fakes.Gate()
    async with _backgrounds(opened, (MAIL, mail_gate), (WIKI, wiki_gate)):
        await mail_gate.wait_for(1)
        for _ in range(20):
            await asyncio.sleep(0)
        assert wiki_gate.entries == 0, "the second background source waits for the first"

        created = await _within(
            service.document_create(collection="memories", slug="tide", body="tide tables"),
            "the write",
        )
        mail_gate.open()
        await _within(wiki_gate.wait_for(1), "the second background source after the first")

    assert created.indexed


async def test_a_sync_of_the_authoring_source_itself_still_queues_the_write(
    opened: Runtime, service: ApplicationService
) -> None:
    """The residual: one source never races itself, so a write waits for its own source's run."""
    gate = fakes.Gate()
    async with _background(opened, AUTHORED, gate):
        await gate.wait_for(1)
        write = asyncio.create_task(
            service.document_create(collection="memories", slug="queued", body="queued body")
        )
        for _ in range(30):
            await asyncio.sleep(0)
        assert not write.done()
        gate.open()
        created = await _within(write, "the write after its own source finished")

    assert created.indexed


# --- cancellation ----------------------------------------------------------------------------


async def test_a_canceled_sync_releases_its_slot(
    opened: Runtime, service: ApplicationService
) -> None:
    gate = fakes.Gate()
    async with _background(opened, MAIL, gate) as sync:
        await gate.wait_for(1)
        sync.cancel()
        await asyncio.gather(sync, return_exceptions=True)

        queued = asyncio.create_task(
            _GatedRun(opened, WIKI).run(), name="sync:wiki-after-the-cancel"
        )
        report = await _within(queued, "a source that queued behind the canceled one")

    assert report.indexed == 1
    assert not opened.derived_barrier.maintenance_pending


async def test_a_canceled_write_does_not_wedge_the_authoring_slot(
    opened: Runtime, service: ApplicationService
) -> None:
    gate = fakes.Gate()
    async with _background(opened, AUTHORED, gate):
        await gate.wait_for(1)
        write = asyncio.create_task(
            service.document_create(collection="memories", slug="abandoned", body="abandoned")
        )
        for _ in range(30):
            await asyncio.sleep(0)
        write.cancel()
        await asyncio.gather(write, return_exceptions=True)
        gate.open()

    created = await _within(
        service.document_create(collection="memories", slug="after", body="after the cancel"),
        "a write after a canceled one",
    )
    assert created.indexed


# --- maintenance -----------------------------------------------------------------------------


async def test_pending_maintenance_queues_a_write_and_the_write_then_succeeds(
    opened: Runtime, service: ApplicationService
) -> None:
    """The documented cost: a write queues behind maintenance that is itself waiting on a sync."""
    gate = fakes.Gate()
    entered, leave = asyncio.Event(), asyncio.Event()

    async def maintenance() -> None:
        async with opened.derived_mutation_guard():
            entered.set()
            await leave.wait()

    async with _background(opened, MAIL, gate):
        await gate.wait_for(1)
        reset = asyncio.create_task(maintenance())
        for _ in range(30):
            await asyncio.sleep(0)
        assert not entered.is_set(), "maintenance waits for the running source"
        write = asyncio.create_task(
            service.document_create(collection="memories", slug="behind", body="behind reset")
        )
        for _ in range(30):
            await asyncio.sleep(0)
        assert not write.done(), "admission is closed while maintenance waits"

        gate.open()
        await _within(entered.wait(), "maintenance after the source drained")
        assert not write.done(), "and stays closed while maintenance runs"
        leave.set()
        await reset
        created = await _within(write, "the write after maintenance")

    assert created.indexed


async def test_a_pipeline_fetched_across_a_reset_is_the_rebuilt_one(opened: Runtime) -> None:
    """A run must not be handed a pipeline built against the identity a reset just retired."""
    before = await opened.pipeline()
    leased: list[object] = []

    async def lease() -> None:
        async with opened.leased_pipeline(AUTHORED) as pipeline:
            leased.append(pipeline)

    async with opened.derived_mutation_guard():
        # A fresh context: a task created here would inherit the maintenance hold, as a worker
        # of the maintenance task should, and would not be the independent caller under test.
        waiting = asyncio.create_task(lease(), context=contextvars.Context())
        for _ in range(30):
            await asyncio.sleep(0)
        assert not leased, "the lease queues behind the maintenance that is running"
        await opened.invalidate_derived_runtime()
    await _within(waiting, "the lease after maintenance")

    assert leased
    assert leased[0] is not before
    assert leased[0] is await opened.pipeline()


async def test_a_built_pipeline_is_returned_without_waiting_for_a_running_source(
    opened: Runtime, service: ApplicationService
) -> None:
    del service
    gate = fakes.Gate()
    async with _background(opened, MAIL, gate):
        await gate.wait_for(1)
        pipeline = await _within(opened.pipeline(), "pipeline()")
        vectors = await _within(opened.prepared_vectors(), "prepared_vectors()")

    assert pipeline is await opened.pipeline()
    assert vectors is await opened.prepared_vectors()


# --- authoring contract ----------------------------------------------------------------------


async def test_overlapping_creates_of_one_slug_never_overwrite_silently(
    opened: Runtime, service: ApplicationService, manicule_environment: Path
) -> None:
    gate = fakes.Gate()
    async with _background(opened, MAIL, gate):
        await gate.wait_for(1)
        outcomes = await _within(
            asyncio.gather(
                service.document_create(collection="memories", slug="same", body="first writer"),
                service.document_create(collection="memories", slug="same", body="second writer"),
                return_exceptions=True,
            ),
            "the overlapping writes",
        )

    created = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    refused = [outcome for outcome in outcomes if isinstance(outcome, PolicyError)]
    assert len(created) == 1
    assert len(refused) == 1, "the loser is told the slug is taken, not given a silent overwrite"
    written = (manicule_environment / AUTHORED / "memories" / "same.md").read_text()
    assert ("first writer" in written) != ("second writer" in written), (
        "the file holds exactly one writer's body, never a blend"
    )


async def test_unrelated_creates_overlap_without_interference(
    opened: Runtime, service: ApplicationService
) -> None:
    results = await _within(
        asyncio.gather(
            *(
                service.document_create(
                    collection="memories", slug=f"note-{number}", body=f"note number {number}"
                )
                for number in range(4)
            )
        ),
        "four unrelated writes",
    )

    assert all(result.indexed and result.member for result in results)
    assert len({result.document_id for result in results}) == 4


async def test_a_create_does_not_count_placement_and_an_explicit_index_still_does(
    opened: Runtime, service: ApplicationService, manicule_environment: Path
) -> None:
    placements: list[str] = []
    original = service._placement  # pyright: ignore[reportPrivateUsage]

    async def counting(source: str) -> Any:
        placements.append(source)
        return await original(source)

    service._placement = counting  # type: ignore[method-assign]

    created = await service.document_create(
        collection="memories", slug="quiet", body="quiet document"
    )
    assert created.indexed
    assert placements == [], "authoring reads only the run's error and its own readback"

    directory = manicule_environment / "elsewhere"
    directory.mkdir()
    (directory / "page.md").write_text("an explicit path\n")
    report = await service.index_path(directory, source="elsewhere")

    assert placements == ["elsewhere"], "index_path still reports source-wide placement"
    assert report.collected is not None
    assert report.uncollected is not None


async def test_a_failed_index_still_names_the_kept_file(
    opened: Runtime, service: ApplicationService, manicule_environment: Path
) -> None:
    ingestion = await service._backend.ingestion()  # pyright: ignore[reportPrivateUsage]

    async def explode(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        msg = "the index is unavailable"
        raise PolicyError(msg)

    ingestion.index_path = explode

    created = await service.document_create(
        collection="memories", slug="kept", body="a file that is kept"
    )

    assert not created.indexed
    assert created.path == str(manicule_environment / AUTHORED / "memories" / "kept.md")
    assert (manicule_environment / AUTHORED / "memories" / "kept.md").exists()
    assert document_id(WORKSPACE, AUTHORED, created.path) == created.document_id


async def test_a_move_re_indexes_outside_the_authoring_lock(
    opened: Runtime, service: ApplicationService
) -> None:
    """The forced re-index is exclusive and waits for every sync; it must not hold the lock."""
    await service.document_create(collection="memories", slug="travels", body="a traveling fact")
    held_during: list[bool] = []

    async def reclaim(path: Any, source: str) -> None:
        del path, source
        held_during.append(service._authoring.locked())  # pyright: ignore[reportPrivateUsage]

    service._reclaim = reclaim  # type: ignore[method-assign]

    moved = await service.document_move(
        slug="travels", from_collection="memories", to_collection="archive"
    )

    assert moved.moved.indexed
    assert held_during == [False]
