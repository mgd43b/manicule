"""Production assembly of connector-free offline replacement generations."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import httpx
import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from manicule.api.app import build_app
from manicule.app import control
from manicule.app.runtime import Runtime
from manicule.app.served import ControlHandler
from manicule.app.service import ApplicationService
from manicule.cli import main as cli
from manicule.config.settings import Settings
from manicule.connectors.sessions import SessionVault
from manicule.container import keys
from manicule.core.acquisition import AcquisitionRecordState, AcquisitionRunState
from manicule.core.errors import ManiculeError
from manicule.core.rebuild import RebuildState
from manicule.core.source_lifecycle import LifecycleRefusalError
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.reembed import ReembedState
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import discover
from manicule.storage import models
from manicule.storage.docstore import SqliteDocStore
from tests.app.test_reembed_runtime import CountingEmbedder
from tests.embedding_support import write_model
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import Mapping


def _runtime(
    data_dir: Path,
    embedder: CountingEmbedder,
    *,
    detect_glossary: bool = True,
    connector_name: str | None = None,
) -> Runtime:
    found = discover()
    found.registry.bind("offline-rebuild-runtime-test").add(
        keys.EMBEDDER.named("local"),
        lambda _: embedder,
        metadata_factory=lambda _: embedder.fingerprint,
    )
    return Runtime(
        Settings(
            data_dir=data_dir,
            connectors=(
                {}
                if connector_name is None
                else {
                    connector_name: {
                        "type": "filesystem",
                        "options": {"root": str(data_dir / "disabled-source")},
                    }
                }
            ),  # pyright: ignore[reportArgumentType]
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            ingest={"parse_workers": 1},  # pyright: ignore[reportArgumentType]
            rag={"glossary": {"detect_on_ingest": detect_glossary}},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )


@dataclass
class _ConstructionTraps:
    embedder: int = 0
    chunker: int = 0
    vectors: int = 0
    workers: int = 0
    rebuilder: int = 0
    connector: int = 0
    metadata: int = 0


async def _acquired_snapshot(data_dir: Path) -> str:
    async with Runtime(Settings(data_dir=data_dir)) as runtime:
        connector = fakes.DictConnector({"one": "durable local bytes"}, name="plan-source")
        await (await runtime.acquisition_pipeline()).run(connector, acquire_only=True)
        documents = cast("SqliteDocStore", await runtime.documents())
        snapshot = await documents.latest_unsettled_acquisition_run(connector.name)
        assert snapshot is not None
        return snapshot.id


def _planning_runtime(
    data_dir: Path, traps: _ConstructionTraps, monkeypatch: pytest.MonkeyPatch
) -> Runtime:
    def forbidden_embedder(_context: object) -> NoReturn:
        traps.embedder += 1
        raise AssertionError("planning constructed the embedder/model device")

    def forbidden_chunker(_context: object) -> NoReturn:
        traps.chunker += 1
        raise AssertionError("planning constructed the chunker")

    def forbidden_vectors(_context: object) -> NoReturn:
        traps.vectors += 1
        raise AssertionError("planning initialized a vector store")

    def forbidden_connector(_context: object) -> NoReturn:
        traps.connector += 1
        raise AssertionError("planning initialized a connector/network capability")

    found = discover()
    registry = found.registry.bind("metadata-only-plan-test")
    embedder = found.registry.record(keys.EMBEDDER.named("onnx"))
    found.registry._records[(ComponentKind.EMBEDDER, "onnx")] = replace(  # pyright: ignore[reportPrivateUsage]
        embedder, factory=forbidden_embedder
    )
    chunker = found.registry.record(keys.CHUNKER.named("structural"))
    found.registry._records[(ComponentKind.CHUNKER, "structural")] = replace(  # pyright: ignore[reportPrivateUsage]
        chunker, factory=forbidden_chunker
    )
    registry.add(keys.CONNECTOR.named("network-trap"), forbidden_connector)
    vector = found.registry.record(keys.VECTOR_STORE.named("lancedb"))
    found.registry._records[(ComponentKind.VECTOR_STORE, "lancedb")] = replace(  # pyright: ignore[reportPrivateUsage]
        vector, factory=forbidden_vectors
    )
    cached_card = write_model(
        data_dir / "cached-card", max_seq_length=1024, max_position_embeddings=2048
    )

    def cached_snapshot_download(
        _repo: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        local_files_only: bool = False,
    ) -> str:
        del revision, allow_patterns
        assert local_files_only, "metadata resolution allowed a model-repository network call"
        traps.metadata += 1
        return str(cached_card)

    monkeypatch.setattr("huggingface_hub.snapshot_download", cached_snapshot_download)
    return Runtime(
        Settings(
            data_dir=data_dir,
            embedding={
                "provider": "onnx",
                "model": "acme/metadata-model",
                "revision": "1" * 40,
            },  # pyright: ignore[reportArgumentType]
            connectors={"plan-source": {"type": "network-trap"}},  # pyright: ignore[reportArgumentType]
            plugins={
                "config": {
                    "embedder.onnx": {
                        "weights": "acme/metadata-export",
                        "weights_revision": "2" * 40,
                    }
                }
            },  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )


def _install_planning_traps(
    monkeypatch: pytest.MonkeyPatch, runtime: Runtime, traps: _ConstructionTraps
) -> None:
    def forbidden_workers(*_args: object, **_kwargs: object) -> NoReturn:
        traps.workers += 1
        raise AssertionError("planning initialized a worker pool")

    def forbidden_rebuilder(*_args: object, **_kwargs: object) -> NoReturn:
        traps.rebuilder += 1
        raise AssertionError("planning initialized an offline rebuilder")

    monkeypatch.setattr("manicule.ingest.workers.WorkerPool", forbidden_workers)
    monkeypatch.setattr("manicule.ingest.rebuild.build_offline_rebuilder", forbidden_rebuilder)


@pytest.mark.parametrize("surface", ["http", "control", "cli"])
async def test_fresh_served_rebuild_plan_is_metadata_only_on_every_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    data_dir = tmp_path / surface
    snapshot_id = await _acquired_snapshot(data_dir)
    traps = _ConstructionTraps()
    runtime = _planning_runtime(data_dir, traps, monkeypatch)
    _install_planning_traps(monkeypatch, runtime, traps)

    async with runtime:
        service = ApplicationService(runtime)
        payload: dict[str, Any]
        if surface == "http":
            app = build_app(service)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                response = await client.get(f"/api/v1/admin/rebuild/snapshots/{snapshot_id}")
            assert response.status_code == 200
            payload = cast("dict[str, Any]", response.json())
        else:
            socket = control.socket_path(data_dir)
            server = control.ControlServer(socket, ControlHandler(service, SessionVault()))
            await server.start()
            try:
                if surface == "control":
                    payload = cast(
                        "dict[str, Any]",
                        await control.connect(
                            socket,
                            control.Invoke(
                                op="rebuild_plan", arguments={"snapshot_id": snapshot_id}
                            ),
                            on_progress=lambda _message: None,
                        ),
                    )
                else:

                    def listening(_overrides: Mapping[str, Any]) -> Path:
                        return socket

                    monkeypatch.setattr(cli.proxy, "listening", listening)
                    cli.STATE.overrides = {}
                    cli.STATE.workspace = None
                    result = await asyncio.to_thread(
                        CliRunner().invoke,
                        cli.app,
                        ["--json", "rebuild", "plan", snapshot_id],
                    )
                    assert result.exit_code == 0, result.output
                    payload = json.loads(result.stdout)
            finally:
                await server.aclose()

        assert payload["ok"] is True
        assert cast("dict[str, Any]", payload["data"])["runnable"] is True
        async with runtime.require_engine().connect() as connection:
            generations = await connection.scalar(select(func.count(models.DerivedGeneration.id)))
        assert generations == 0, "planning persisted a shadow generation"
        assert traps == _ConstructionTraps(metadata=1)
        assert "vectors" not in runtime._slots  # pyright: ignore[reportPrivateUsage]
        assert "prepared_vectors" not in runtime._slots  # pyright: ignore[reportPrivateUsage]


async def test_rebuild_run_refuses_stale_component_metadata_before_persisting(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "metadata-mismatch"
    snapshot_id = await _acquired_snapshot(data_dir)
    embedder = CountingEmbedder()
    embedder.fingerprint = embedder.fingerprint.model_copy(update={"max_sequence_length": 1024})
    declared = embedder.fingerprint.model_copy(update={"max_sequence_length": 2048})
    found = discover()
    found.registry.bind("stale-rebuild-metadata-test").add(
        keys.EMBEDDER.named("local"),
        lambda _: embedder,
        metadata_factory=lambda _: declared,
    )
    runtime = Runtime(
        Settings(
            data_dir=data_dir,
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            ingest={"parse_workers": 1},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )

    async with runtime:
        ingestion = await runtime.ingestion()
        assert (await ingestion.rebuild_plan(snapshot_id)).runnable
        with pytest.raises(
            ManiculeError, match="metadata disagrees with the executable rebuild stack: embedder"
        ):
            await ingestion.rebuild_run(snapshot_id, "mismatch-owner")
        async with runtime.require_engine().connect() as connection:
            generations = await connection.scalar(select(func.count(models.DerivedGeneration.id)))
        assert generations == 0


async def test_metadata_only_plan_failure_does_not_expose_a_configured_model_path(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "private-safe"
    snapshot_id = await _acquired_snapshot(data_dir)
    private_model = "/private/operator/models/secret-client-model"
    runtime = Runtime(
        Settings(
            data_dir=data_dir,
            embedding={
                "provider": "onnx",
                "model": private_model,
                "revision": "1" * 40,
            },  # pyright: ignore[reportArgumentType]
        )
    )

    async with runtime:
        app = build_app(ApplicationService(runtime))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.get(f"/api/v1/admin/rebuild/snapshots/{snapshot_id}")

    assert response.status_code != 200
    rendered = response.text
    assert private_model not in rendered
    assert "configured embedding identity is unavailable" in rendered


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


async def test_runtime_rebuilds_a_promoted_snapshot_without_a_connector_capability(  # noqa: PLR0915
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
    async with _runtime(data_dir, embedder, connector_name=connector.name) as runtime:
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

        async def forbidden_connector(_name: str) -> NoReturn:
            raise AssertionError("offline publication must not construct a connector")

        runtime.connector = forbidden_connector  # type: ignore[method-assign]

        checkpoint = await (await runtime.ingestion()).rebuild_run(snapshot.id, "runtime-owner")
        status = await ApplicationService(runtime).snapshot_status(connector.name)
        settled = await documents.get_acquisition_run(snapshot.id)
        records = await documents.list_acquisition_records(snapshot.id)
        metadata = await documents.connector_metadata(connector.name)

        assert checkpoint.state is RebuildState.PUBLISHED
        assert checkpoint.documents_built == 3
        assert checkpoint.chunks_built == 3
        assert tuple(connector.fetches) == source_calls
        assert settled is not None
        assert settled.state is AcquisitionRunState.SETTLED
        assert {record.state for record in records} == {AcquisitionRecordState.SETTLED}
        assert status.state == "settled"
        assert status.lifecycle.phase == "complete"
        assert status.lifecycle.outcome == "complete"
        assert status.lifecycle.pending_items == 0
        assert status.lifecycle.backlog_items == 0
        assert status.lifecycle.can_continue_offline
        assert await documents.verify_snapshot_manifest(snapshot.id)
        last_run = cast("Mapping[str, Any]", metadata["last_run"])
        metadata_lifecycle = cast("Mapping[str, Any]", last_run["lifecycle"])
        assert last_run["outcome"] == "complete"
        assert last_run["retry_required"] is False
        assert metadata_lifecycle["phase"] == "complete"
        assert metadata_lifecycle["outcome"] == "complete"
        assert metadata_lifecycle["pending_items"] == 0
        assert metadata_lifecycle["backlog_items"] == 0
        assert metadata_lifecycle["snapshot_completeness"] == "complete"
        assert metadata_lifecycle["reproducibility_policy"] == "require_complete"
        public_metadata = json.dumps(metadata, sort_keys=True)
        assert "Network Operations Workspace" not in public_metadata
        assert "https://wiki.example.test/content/" not in public_metadata
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
        with pytest.raises(LifecycleRefusalError, match="anchors a derived generation"):
            await (await resumed.maintenance()).plan_snapshot_deletion(snapshot_id)

        assert reset.snapshot_items == 3
        assert rebuilt_again.state is RebuildState.PUBLISHED
        assert rebuilt_again.documents_built == 3
        assert tuple(connector.fetches) == source_calls

    sync_embedder = CountingEmbedder()
    sync_embedder.fingerprint = sync_embedder.fingerprint.model_copy(
        update={"model_id": "second/model", "max_sequence_length": 1024}
    )
    async with _runtime(data_dir, sync_embedder) as sync_runtime:
        unchanged = await (await sync_runtime.pipeline()).run(connector)

        assert tuple(connector.fetches) == source_calls
        assert sync_embedder.texts == 0
        assert unchanged.indexed == 0
        assert unchanged.lifecycle_metadata()["reused_items"] == 3

        connector.documents["three"] = " ".join(["changed"] * 100)
        changed = await (await sync_runtime.pipeline()).run(connector)

        assert tuple(connector.fetches) == (*source_calls, "three")
        assert sync_embedder.texts == 1
        assert changed.indexed == 1
        assert changed.lifecycle_metadata()["reused_items"] == 2


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


def test_process_smoke_restarts_and_settles_without_duplicate_work(tmp_path: Path) -> None:
    tool = Path(__file__).parents[2] / "tools" / "smoke_offline_rebuild_settlement.py"
    completed = subprocess.run(  # noqa: S603 - current interpreter and repository-owned tool
        [
            sys.executable,
            str(tool),
            "--data-dir",
            str(tmp_path / "data"),
            "--source-dir",
            str(tmp_path / "source"),
            "--max-rss-bytes",
            str(512 * 1024 * 1024),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["ok"] is True
    assert payload["peak_rss_bytes"] <= payload["rss_bound_bytes"]
    assert payload["phases"]["acquire"]["embed_texts"] == 0
    assert payload["phases"]["publish"]["vectors_embedded"] == 3
    assert payload["phases"]["restart"]["verified"] is True
    assert payload["phases"]["restart"]["pending_items"] == 0
    assert payload["phases"]["restart"]["foreign_key_violations"] == 0
    assert payload["phases"]["second"]["vectors_reused"] == 3
    assert (
        payload["phases"]["second"]["generation_id"]
        != payload["phases"]["publish"]["generation_id"]
    )
    assert payload["phases"]["unchanged"] == {
        "counts": {"blobs": 3, "chunks": 3, "documents": 3, "generations": 2},
        "counts_unchanged": True,
        "embed_texts": 0,
        "ingested": 0,
        "max_rss_bytes": payload["phases"]["unchanged"]["max_rss_bytes"],
        "reused_items": 3,
    }
