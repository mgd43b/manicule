"""Operator re-embedding through the production runtime and on-disk adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, override

import pytest
from sqlalchemy import func, insert, select

from manicule.app.runtime import AssemblyError, Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.embedding import Vector
from manicule.ingest.reembed import ReembedCapacityError, ReembedError, ReembedState
from manicule.plugins.registry import discover
from manicule.storage import models
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import VECTORS_DIRNAME
from manicule.storage.types import utcnow
from manicule.storage.vectors import LanceVectorStore, table_name
from tests.fakes import HashEmbedder, MemoryVectorStore
from tests.storage_helpers import fingerprint, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class CountingEmbedder(HashEmbedder):
    """A local target whose forward passes are observable and cannot reach a network."""

    def __init__(self) -> None:
        super().__init__(dimension=4, model_id="target/model")
        self.texts = 0

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.texts += len(texts)
        return await super().embed(texts)


def _runtime(data_dir: Path, embedder: CountingEmbedder) -> Runtime:
    found = discover()
    found.registry.bind("reembed-operator-test").add(
        keys.EMBEDDER.named("local"), lambda _: embedder
    )
    return Runtime(
        Settings(
            data_dir=data_dir,
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )


async def _seed_live_generation(runtime: Runtime, data_dir: Path) -> None:
    await runtime.documents()
    engine = runtime.require_engine()
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    document = make_document().model_copy(update={"publication_id": "old-publication"})
    chunk = make_chunk(document, 0, "durable local input")
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [chunk])

    old = fingerprint(dimension=4, model_id="old/model")
    vectors = LanceVectorStore(data_dir / VECTORS_DIRNAME)
    await vectors.ensure_ready(old)
    await vectors.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=document.publication_id)
    await vectors.teardown()
    async with engine.begin() as connection:
        await connection.execute(
            insert(models.IndexState).values(
                id=1,
                vector_table=table_name(old),
                embed_fingerprint=old.model_dump_json(),
                vector_inventory_digest=None,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )


async def test_plan_is_zero_embed_private_safe_and_discards_its_snapshot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    embedder = CountingEmbedder()
    async with _runtime(data_dir, embedder) as runtime:
        await _seed_live_generation(runtime, data_dir)
        service = ApplicationService(runtime)
        report = await service.reembed_plan()

        assert embedder.texts == 0
        assert report.documents == report.chunks == 1
        public = report.model_dump_json()
        assert "private" not in public
        assert "old-publication" not in public
        assert str(data_dir) not in public
        async with runtime.require_engine().connect() as connection:
            snapshots = (
                await connection.execute(
                    select(func.count()).select_from(models.ReembedCorpusSnapshot)
                )
            ).scalar_one()
        assert snapshots == 0, "a transient dry-run snapshot became durable retained state"


async def test_start_returns_recovery_id_and_resume_survives_process_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    embedder = CountingEmbedder()
    first = _runtime(data_dir, embedder)
    async with first:
        await _seed_live_generation(first, data_dir)
        first_service = ApplicationService(first)
        started = await first_service.reembed_start("restart-run")
        assert started.state == ReembedState.PLANNED
        assert started.retry_required
        assert embedder.texts == 0, "start must return the recovery id before the forward pass"
        run_id = started.run_id
        replayed = await first_service.reembed_start("restart-run")
        assert replayed == started, "a lost control reply must be recoverable by replaying start"
        async with first.require_engine().connect() as connection:
            row = (
                await connection.execute(
                    select(
                        models.ReembedRunRecord.lease_owner,
                        models.ReembedRunRecord.lease_expires_at,
                    ).where(models.ReembedRunRecord.id == run_id)
                )
            ).one()
            snapshots = (
                await connection.execute(
                    select(func.count()).select_from(models.ReembedCorpusSnapshot)
                )
            ).scalar_one()
        assert row == (None, None), "start returned while a planning owner still held the run"
        assert snapshots == 1, "replaying a lost start reply created a private orphan snapshot"

    second = _runtime(data_dir, embedder)
    async with second:
        service = ApplicationService(second)
        before = await service.reembed_status(run_id)
        assert before.state == ReembedState.PLANNED

        async def refuse_model() -> HashEmbedder:
            pytest.fail("a capacity refusal constructed the embedding model")

        def no_capacity(_: Path) -> SimpleNamespace:
            return SimpleNamespace(free=0)

        with monkeypatch.context() as refused:
            refused.setattr("manicule.app.runtime.shutil.disk_usage", no_capacity)
            refused.setattr(second, "embedder", refuse_model)
            with pytest.raises(ReembedCapacityError):
                await service.reembed_resume(run_id)

        finished = await service.reembed_resume(run_id)
        assert finished.state == ReembedState.PUBLISHED
        assert finished.published
        assert not finished.retry_required
        assert embedder.texts == 1
        assert (await service.reembed_status(run_id)).state == ReembedState.PUBLISHED

        # The runtime prepares its long-lived pointer-following handle for the winner before
        # reporting success. A same-dimension model switch therefore neither queries with the
        # old model nor returns a false-positive success that requires a hidden restart.
        vectors = await second.vectors()
        assert await vectors.fingerprint() == embedder.fingerprint


async def test_capacity_refusal_is_typed_and_leaves_no_unreachable_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    embedder = CountingEmbedder()
    async with _runtime(data_dir, embedder) as runtime:
        await _seed_live_generation(runtime, data_dir)

        def no_capacity(_: Path) -> SimpleNamespace:
            return SimpleNamespace(free=0)

        monkeypatch.setattr("manicule.app.runtime.shutil.disk_usage", no_capacity)

        with pytest.raises(ReembedCapacityError, match="free local disk space"):
            await ApplicationService(runtime).reembed_start("capacity-run")

        assert embedder.texts == 0
        async with runtime.require_engine().connect() as connection:
            snapshots = (
                await connection.execute(
                    select(func.count()).select_from(models.ReembedCorpusSnapshot)
                )
            ).scalar_one()
            runs = (
                await connection.execute(select(func.count()).select_from(models.ReembedRunRecord))
            ).scalar_one()
        assert (snapshots, runs) == (0, 0)


@pytest.mark.parametrize("operation", ["plan", "start"])
async def test_custom_vector_backend_is_refused_before_snapshot_model_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    data_dir = tmp_path / "data"
    embedder = CountingEmbedder()
    async with _runtime(data_dir, embedder) as runtime:

        async def custom_vectors() -> MemoryVectorStore:
            return MemoryVectorStore()

        async def refuse_model() -> HashEmbedder:
            pytest.fail("an unsupported vector backend constructed the embedding model")

        monkeypatch.setattr(runtime, "vectors", custom_vectors)
        monkeypatch.setattr(runtime, "embedder", refuse_model)
        service = ApplicationService(runtime)
        call = service.reembed_plan() if operation == "plan" else service.reembed_start("known-run")

        with pytest.raises(ReembedError, match="SQLite/Lance vector backend"):
            await call

        with pytest.raises(AssemblyError, match="engine has not been opened"):
            runtime.require_engine()
