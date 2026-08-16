"""Production assembly of connector-free offline replacement generations."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, cast

import pytest

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.rebuild import RebuildState
from manicule.core.source_lifecycle import LifecycleRefusalError
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.reembed import ReembedState
from manicule.plugins.registry import discover
from manicule.storage.docstore import SqliteDocStore
from tests.app.test_reembed_runtime import CountingEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from pathlib import Path


def _runtime(
    data_dir: Path, embedder: CountingEmbedder, *, detect_glossary: bool = True
) -> Runtime:
    found = discover()
    found.registry.bind("offline-rebuild-runtime-test").add(
        keys.EMBEDDER.named("local"), lambda _: embedder
    )
    return Runtime(
        Settings(
            data_dir=data_dir,
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            ingest={"parse_workers": 1},  # pyright: ignore[reportArgumentType]
            rag={"glossary": {"detect_on_ingest": detect_glossary}},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )


async def test_acquire_only_runtime_never_constructs_the_derived_stack(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-source"
    source.mkdir()
    for number in range(10):
        (source / f"document-{number}.txt").write_text(f"synthetic document {number}")
    calls = 0

    def forbidden_embedder(_context: object) -> NoReturn:
        nonlocal calls
        calls += 1
        raise AssertionError("the embedding device must not be constructed for acquire-only")

    found = discover()
    found.registry.bind("acquire-only-runtime-test").add(
        keys.EMBEDDER.named("local"), forbidden_embedder
    )
    runtime = Runtime(
        Settings(
            data_dir=tmp_path / "data",
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            storage={"retain_source_bytes": True},  # pyright: ignore[reportArgumentType]
            connectors={
                "synthetic-files": {
                    "type": "filesystem",
                    "options": {"root": str(source)},
                }
            },  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )

    async with runtime:
        result = await ApplicationService(runtime).connector_sync(
            "synthetic-files", acquire_only=True
        )
        status = await ApplicationService(runtime).snapshot_status("synthetic-files")

        assert result.discovered == 10
        assert result.lifecycle.enumerated_items == 10
        assert result.lifecycle.acquired_items == 10
        assert result.lifecycle.pending_items == status.lifecycle.pending_items
        assert status.lifecycle.enumerated_items == result.lifecycle.enumerated_items
        assert status.lifecycle.acquired_items == result.lifecycle.acquired_items
        assert status.lifecycle.omitted_items == result.lifecycle.omitted_items
        assert status.lifecycle.failed_items == result.lifecycle.failed_items
        assert calls == 0
        documents = cast("SqliteDocStore", await runtime.documents())
        assert await documents.count_documents() == 0
        assert "pipeline" not in runtime._slots  # pyright: ignore[reportPrivateUsage]
        assert "vectors" not in runtime._slots  # pyright: ignore[reportPrivateUsage]
        assert "prepared_vectors" not in runtime._slots  # pyright: ignore[reportPrivateUsage]


async def test_snapshot_status_after_restart_uses_no_connector_factory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    connector = fakes.DictConnector({"one": "alpha", "two": "beta"}, name="synthetic-wiki")
    async with _runtime(data_dir, CountingEmbedder()) as runtime:
        report = await (await runtime.acquisition_pipeline()).run(connector, acquire_only=True)
        expected = report.lifecycle_metadata()

    restarted = Runtime(
        Settings(
            data_dir=data_dir,
            connectors={
                "synthetic-wiki": {
                    "type": "filesystem",
                    "options": {"root": str(tmp_path / "unreachable-source")},
                }
            },  # pyright: ignore[reportArgumentType]
        )
    )

    async def forbidden_connector(_name: str) -> NoReturn:
        raise AssertionError("snapshot status must not construct or authenticate a connector")

    restarted.connector = forbidden_connector  # type: ignore[method-assign]
    async with restarted:
        service = ApplicationService(restarted)
        status = await service.snapshot_status("synthetic-wiki")
        verified = await service.snapshot_verify(status.snapshot_id)
        assert status.lifecycle.enumerated_items == expected["enumerated_items"]
        assert status.lifecycle.acquired_items == expected["acquired_items"]
        assert status.lifecycle.omitted_items == expected["omitted_items"]
        assert status.lifecycle.failed_items == expected["failed_items"]
        assert verified.lifecycle == status.lifecycle
        assert verified.verified


async def test_runtime_rebuilds_a_promoted_snapshot_without_a_connector_capability(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    embedder = CountingEmbedder()
    embedder.fingerprint = embedder.fingerprint.model_copy(update={"max_sequence_length": 1024})
    connector = fakes.DictConnector(
        {
            "one": "The Network Operations Workspace (NOW, NETOPS) coordinates tickets. "
            + " ".join(["alpha"] * 100),
            "two": " ".join(["beta"] * 100),
            "three": " ".join(["gamma"] * 100),
        },
        name="runtime-snapshot",
    )
    connector.media_types.update({"one": "text/plain", "two": "text/plain", "three": "text/plain"})
    async with _runtime(data_dir, embedder) as runtime:
        acquired = await (await runtime.pipeline()).run(connector, acquire_only=True)
        source_calls = tuple(connector.fetches)
        documents = cast("SqliteDocStore", await runtime.documents())
        snapshot = await documents.latest_unsettled_acquisition_run(connector.name)

        assert acquired.derivation_deferred
        assert snapshot is not None
        snapshot_id = snapshot.id
        plan = await (await runtime.ingestion()).rebuild_plan(snapshot.id)
        assert plan.runnable
        assert plan.documents == 3
        assert await (await runtime.ingestion()).rebuild_status(plan.generation_id) is None, (
            "read-only planning must not create a durable generation"
        )

        checkpoint = await (await runtime.ingestion()).rebuild_run(snapshot.id, "runtime-owner")

        assert checkpoint.state is RebuildState.PUBLISHED
        assert checkpoint.documents_built == 3
        assert checkpoint.chunks_built == 3
        assert tuple(connector.fetches) == source_calls
        assert await documents.count_documents() == 3
        first = next(
            document for document in await documents.list_documents() if document.source_id == "one"
        )
        assert (await documents.glossary_entries(first.id))[0].aliases == ("NETOPS",)

    second = CountingEmbedder()
    second.fingerprint = second.fingerprint.model_copy(
        update={"model_id": "second/model", "max_sequence_length": 1024}
    )
    async with _runtime(data_dir, second) as restarted:
        planned = await (await restarted.ingestion()).reembed_start(
            "second-model", "planning-owner"
        )
        assert planned.chunks_completed == 0
        assert second.texts == 0

    resumed_embedder = CountingEmbedder()
    resumed_embedder.fingerprint = resumed_embedder.fingerprint.model_copy(
        update={"model_id": "second/model", "max_sequence_length": 1024}
    )
    async with _runtime(data_dir, resumed_embedder) as resumed:
        published = await (await resumed.ingestion()).reembed_resume("second-model", "resume-owner")

        assert published.state is ReembedState.PUBLISHED
        assert published.chunks_completed == 3
        assert resumed_embedder.texts == 3
        assert tuple(connector.fetches) == source_calls

        reset = await (await resumed.maintenance()).reset_derived()
        rebuilt_again = await (await resumed.ingestion()).rebuild_run(
            snapshot_id, "post-reset-owner"
        )
        with pytest.raises(LifecycleRefusalError, match="resumable work"):
            await (await resumed.maintenance()).plan_snapshot_deletion(snapshot_id)

        assert reset.snapshot_items == 3
        assert rebuilt_again.state is RebuildState.PUBLISHED
        assert rebuilt_again.documents_built == 3
        assert tuple(connector.fetches) == source_calls


async def test_runtime_rebuild_honors_disabled_glossary_identity_without_source_calls(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "disabled-data"
    embedder = CountingEmbedder()
    embedder.fingerprint = embedder.fingerprint.model_copy(update={"max_sequence_length": 1024})
    connector = fakes.DictConnector(
        {"glossary": "The Network Operations Workspace (NOW, NETOPS) coordinates tickets."},
        name="disabled-runtime-snapshot",
    )
    connector.media_types["glossary"] = "text/plain"

    async with _runtime(data_dir, embedder, detect_glossary=False) as runtime:
        acquired = await (await runtime.pipeline()).run(connector, acquire_only=True)
        source_calls = tuple(connector.fetches)
        documents = cast("SqliteDocStore", await runtime.documents())
        snapshot = await documents.latest_unsettled_acquisition_run(connector.name)
        assert acquired.derivation_deferred
        assert snapshot is not None

        ingestion = await runtime.ingestion()
        disabled = glossary_fingerprint(enabled=False, middleware=()).canonical()
        published = await ingestion.rebuild_run(snapshot.id, owner="disabled-runtime-owner")

        assert published.state is RebuildState.PUBLISHED
        assert tuple(connector.fetches) == source_calls
        document = (await documents.list_documents())[0]
        assert await documents.glossary_entries(document.id) == []
        assert await documents.glossary_lineage(document.id) == disabled
