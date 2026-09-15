"""Incomplete ingest is one outcome at every automation boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING, Any, Never, cast, override

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from manicule.api.envelopes import SERVICE_UNAVAILABLE, status_for
from manicule.app import commands, control
from manicule.app.commands import Command
from manicule.app.dispatch import run_op
from manicule.app.results import Envelope
from manicule.app.served import ControlHandler, Scheduler
from manicule.app.service import ApplicationService
from manicule.cli import main as cli
from manicule.cli import proxy
from manicule.config.settings import ConnectorSettings
from manicule.connectors.sessions import SessionVault
from manicule.core.errors import StorageBusyError
from manicule.ingest.capacity import CapacityDiagnostic, CapacityRefusedError, CapacityResource
from manicule.ingest.pipeline import RunReport
from manicule.mcp.server import build_server
from tests.api.support import client_for
from tests.app.fakes import FakeBackend, FakeStore, make_document

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _service(report: RunReport) -> tuple[ApplicationService, FakeBackend]:
    backend = FakeBackend()
    backend.ingestion_.report = report
    backend.settings.connectors["synthetic-wiki"] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )
    return ApplicationService(backend), backend


def _incomplete() -> RunReport:
    return RunReport(
        connector="synthetic-wiki",
        discovered=200,
        by_status={"indexed": 180},
        error="CursorExpiredError: the search cursor expired",
        error_type="CursorExpiredError",
        error_message="the search cursor expired",
        enumeration_completed=False,
    )


def _capacity_incomplete() -> RunReport:
    report = RunReport(
        connector="synthetic-wiki",
        enumeration_completed=False,
        glossary_failures=[
            "private-source-id: https://private.invalid/doc?token=fake-secret-cinder"
        ],
    )
    report.refuse_capacity(
        CapacityRefusedError(
            CapacityDiagnostic(
                resource=CapacityResource.JOURNAL_METADATA_BYTES,
                limit=100,
                used=90,
                requested=20,
            )
        )
    )
    return report


def _storage_busy_incomplete() -> RunReport:
    report = RunReport(
        connector="synthetic-wiki",
        discovered=2,
        by_status={"indexed": 1},
        enumeration_completed=False,
        glossary_failures=[
            "private-document-id from private-source-id: glossary detection failed for "
            "https://private.invalid/source?token=fake-secret-cinder"
        ],
    )
    report.refuse_storage_busy(StorageBusyError())
    return report


async def _envelope(service: ApplicationService) -> Envelope:
    return await run_op(
        "connector_sync",
        service.workspace,
        lambda: service.connector_sync("synthetic-wiki"),
    )


async def test_incomplete_sync_keeps_partial_data_in_a_failure_envelope() -> None:
    service, _ = _service(_incomplete())
    envelope = await _envelope(service)

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "CursorExpiredError"
    assert envelope.data is not None
    assert envelope.data["outcome"] == "incomplete"
    assert envelope.data["discovered"] == 200
    assert envelope.data["ingested"] == 180
    assert envelope.data["retry_required"] is True
    assert envelope.data["watermark_advanced"] is False
    lifecycle = cast("dict[str, Any]", envelope.data["lifecycle"])
    assert lifecycle["phase"] == "acquiring"
    assert lifecycle["outcome"] == "incomplete"
    assert lifecycle["enumerated_items"] == 200
    assert lifecycle["pending_items"] == 0
    assert lifecycle["committed_watermark_present"] is False


async def test_partial_snapshot_omissions_reach_the_shared_operator_envelope() -> None:
    service, _ = _service(
        RunReport(
            connector="synthetic-wiki",
            discovered=3,
            snapshot_completeness="partial",
            snapshot_omissions=2,
            snapshot_omission_reasons={"authentication": 1, "missing_body": 1},
        )
    )

    envelope = await _envelope(service)

    assert envelope.ok is True
    assert envelope.data is not None
    assert envelope.data["snapshot_completeness"] == "partial"
    assert envelope.data["snapshot_omissions"] == 2
    assert envelope.data["snapshot_omission_reasons"] == {
        "authentication": 1,
        "missing_body": 1,
    }


async def test_strict_snapshot_omission_is_an_incomplete_retryable_failure_envelope() -> None:
    service, _ = _service(
        RunReport(
            connector="synthetic-wiki",
            discovered=1,
            by_status={"indexed": 1},
            snapshot_omissions=1,
            snapshot_omission_reasons={"authentication": 1},
        )
    )

    envelope = await _envelope(service)

    assert envelope.ok is False
    assert envelope.data is not None
    assert envelope.data["outcome"] == "incomplete"
    assert envelope.data["retry_required"] is True
    assert envelope.data["unrecorded"] == 0
    assert envelope.error is not None
    assert envelope.error.type == "IncompleteIngestError"


async def test_inventory_recovery_reaches_the_shared_private_safe_envelope() -> None:
    service, _ = _service(
        RunReport(
            connector="synthetic-wiki",
            discovered=9,
            snapshot_omissions=1,
            snapshot_omission_reasons={"source_deleted": 1},
            inventory_recovery="reenumeration_required",
        )
    )

    envelope = await _envelope(service)

    assert envelope.ok is False
    assert envelope.data is not None
    assert envelope.data["inventory_recovery"] == "reenumeration_required"
    assert envelope.data["reconciled_deleted_items"] == 0
    lifecycle = cast("dict[str, Any]", envelope.data["lifecycle"])
    assert lifecycle["inventory_recovery"] == "reenumeration_required"
    assert lifecycle["reconciled_deleted_items"] == 0
    assert envelope.error is not None
    assert envelope.error.message == (
        "a fresh source inventory is required before this snapshot can be promoted"
    )
    assert "fence the stale inventory" in envelope.error.hint
    rendered = str(envelope.as_json())
    assert "source-id" not in rendered
    assert "https://wiki.example.test/private" not in rendered


async def test_pending_durable_derivation_is_a_retry_required_failure_envelope() -> None:
    service, _ = _service(
        RunReport(
            connector="synthetic-wiki",
            discovered=1,
            pending_derivation=True,
            enumeration_completed=True,
            watermark_advanced=True,
        )
    )

    envelope = await _envelope(service)

    assert envelope.ok is False
    assert envelope.data is not None
    assert envelope.data["outcome"] == "incomplete"
    assert envelope.data["retry_required"] is True
    assert envelope.error is not None
    assert envelope.error.type == "IncompleteIngestError"


async def test_requested_acquire_only_is_a_successful_deferred_offline_outcome() -> None:
    service, backend = _service(
        RunReport(
            connector="synthetic-wiki",
            discovered=3,
            pending_derivation=True,
            derivation_deferred=True,
            enumeration_completed=True,
            watermark_advanced=True,
            snapshot_completeness="complete",
        )
    )

    envelope = await run_op(
        "connector_sync",
        service.workspace,
        lambda: service.connector_sync("synthetic-wiki", acquire_only=True),
    )

    assert envelope.ok is True
    assert backend.ingestion_.sync_acquire_only == [True]
    assert envelope.data is not None
    assert envelope.data["retry_required"] is False
    assert envelope.data["derivation_deferred"] is True
    lifecycle = cast("dict[str, Any]", envelope.data["lifecycle"])
    assert lifecycle["phase"] == "rebuilding"
    assert lifecycle["outcome"] == "deferred"
    assert lifecycle["can_continue_offline"] is True


async def test_capacity_refusal_is_typed_retryable_and_aggregate_only() -> None:
    service, _ = _service(_capacity_incomplete())
    envelope = await _envelope(service)

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "CapacityRefusedError"
    assert envelope.data is not None
    assert envelope.data["outcome"] == "incomplete"
    assert envelope.data["retry_required"] is True
    assert envelope.data["watermark_advanced"] is False
    lifecycle = cast("dict[str, Any]", envelope.data["lifecycle"])
    assert lifecycle["outcome"] == "refused"
    assert lifecycle["refusal"] == {
        "code": "capacity",
        "count": 1,
        "resource": "journal_metadata_bytes",
        "limit": 100,
        "used": 90,
        "requested": 20,
    }
    rendered = json.dumps(envelope.as_json(), sort_keys=True)
    assert "journal_metadata_bytes" in rendered
    for private in (
        "source_id",
        "private-source-id",
        "private.invalid",
        "uri",
        "title",
        "body",
        "secret",
        "token=",
    ):
        assert private not in rendered.lower()

    metadata = _capacity_incomplete().as_metadata()
    persisted = json.dumps(metadata, sort_keys=True)
    assert '"limit": 100' in persisted
    assert '"used": 90' in persisted
    assert '"requested": 20' in persisted
    for private in (
        "source_id",
        "private-source-id",
        "private.invalid",
        "uri",
        "title",
        "body",
        "secret",
        "token=",
    ):
        assert private not in persisted.lower()


async def test_storage_busy_report_and_control_envelope_are_aggregate_only(
    tmp_path: Path,
) -> None:
    report = _storage_busy_incomplete()
    assert report.glossary_failures == []
    assert report.error_type == "StorageBusyError"
    assert report.discovered == 2
    assert report.indexed == 1

    service, _ = _service(report)
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        envelope = await control.connect(
            path,
            control.Invoke(
                op="connector_sync", arguments={"name": "synthetic-wiki", "limit": None}
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    assert envelope["ok"] is False
    error = cast("dict[str, Any]", envelope["error"])
    assert error["type"] == "StorageBusyError"
    assert error["hint"] == ("Run the same ingest operation again; its watermark was not advanced.")
    data = cast("dict[str, Any]", envelope["data"])
    assert data["discovered"] == 2
    assert data["ingested"] == 1
    assert data["retry_required"] is True

    rendered = json.dumps(
        {"report": report.as_metadata(), "envelope": envelope}, sort_keys=True
    ).lower()
    for private in (
        "document_id",
        "source_id",
        "private-document-id",
        "private-source-id",
        "private.invalid",
        "uri",
        "title",
        "body",
        "secret",
        "token=",
    ):
        assert private not in rendered


async def test_watch_batch_preserves_a_child_capacity_refusal(tmp_path: Path) -> None:
    service, backend = _service(_capacity_incomplete())
    envelope = await run_op(
        "index_changes",
        service.workspace,
        lambda: service.index_changes([tmp_path], source="synthetic-wiki"),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "CapacityRefusedError"
    assert envelope.data is not None
    assert envelope.data["outcome"] == "incomplete"
    assert envelope.data["retry_required"] is True
    assert backend.ingestion_.paths == [tmp_path]


async def test_archive_capacity_refusal_requires_a_forced_recovery_import(tmp_path: Path) -> None:
    service, backend = _service(_capacity_incomplete())
    archive = tmp_path / "private-archive-cinder"
    archive.mkdir()

    report = await service.import_corpus(archive)

    assert report.retry_required is True
    assert report.incomplete_reason is not None
    assert report.incomplete_reason.type == "CapacityRefusedError"
    assert "force enabled" in report.incomplete_reason.hint
    assert backend.ingestion_.imported == [archive]
    rendered = repr(report.model_dump(mode="json"))
    assert "private-archive-cinder" not in rendered


async def test_raw_repair_capacity_refusal_is_typed_nonzero_and_http_503() -> None:
    refusal = CapacityRefusedError(
        CapacityDiagnostic(
            resource=CapacityResource.DISK_HEADROOM_BYTES,
            limit=100,
            used=90,
            requested=20,
        )
    )

    async def repair() -> Never:
        raise refusal

    envelope = await run_op("document_reindex", "default", repair)

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "CapacityRefusedError"
    assert "Free durable ingest capacity" in envelope.error.hint
    assert status_for(envelope) == SERVICE_UNAVAILABLE


def test_json_and_human_cli_fail_for_the_same_incomplete_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = _service(_incomplete())

    async def dispatch(command: Command) -> Envelope:
        return await run_op(
            command.op,
            service.workspace,
            lambda: commands.run(service, command, commands.silent),
        )

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    runner = CliRunner()
    machine = runner.invoke(cli.app, ["--json", "connector", "sync", "synthetic-wiki"])
    human = runner.invoke(cli.app, ["connector", "sync", "synthetic-wiki"])

    assert machine.exit_code == 1
    payload = cast("dict[str, Any]", json.loads(machine.stdout))
    assert payload["ok"] is False
    assert cast("dict[str, Any]", payload["data"])["outcome"] == "incomplete"
    assert human.exit_code == 1
    assert "outcome" in human.stdout
    assert "incomplete" in human.stdout
    assert "retry required" in human.stdout


def test_capacity_refusal_makes_cli_nonzero_without_private_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = _service(_capacity_incomplete())

    async def dispatch(command: Command) -> Envelope:
        return await run_op(
            command.op,
            service.workspace,
            lambda: commands.run(service, command, commands.silent),
        )

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    runner = CliRunner()
    results = (
        runner.invoke(cli.app, ["--json", "connector", "sync", "synthetic-wiki"]),
        runner.invoke(cli.app, ["--json", "index", ".", "--reindex"]),
    )

    for result in results:
        assert result.exit_code == 1
        assert "CapacityRefusedError" in result.stdout
        assert "journal_metadata_bytes" in result.stdout
        for private in ("source_id", "uri", "title", "body", "secret", "token="):
            assert private not in result.stdout.lower()


def test_multi_connector_shell_orchestration_does_not_log_incomplete_as_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, backend = _service(_incomplete())
    backend.settings.connectors["synthetic-files"] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )

    async def dispatch(command: Command) -> Envelope:
        return await run_op(
            command.op,
            service.workspace,
            lambda: commands.run(service, command, commands.silent),
        )

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    runner = CliRunner()
    completed: list[str] = []

    backend.ingestion_.report = _incomplete()
    first = runner.invoke(cli.app, ["--json", "connector", "sync", "synthetic-wiki"])
    if first.exit_code == 0:
        completed.append("synthetic-wiki")

    backend.ingestion_.report = RunReport(connector="synthetic-files", discovered=3)
    second = runner.invoke(cli.app, ["--json", "connector", "sync", "synthetic-files"])
    if second.exit_code == 0:
        completed.append("synthetic-files")

    assert completed == ["synthetic-files"]


async def test_control_socket_preserves_partial_failure_data(tmp_path: Path) -> None:
    service, _ = _service(_incomplete())
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        envelope = await control.connect(
            path,
            control.Invoke(
                op="connector_sync", arguments={"name": "synthetic-wiki", "limit": None}
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    data = cast("dict[str, Any]", envelope["data"])
    assert envelope["ok"] is False
    assert data["outcome"] == "incomplete"
    assert data["discovered"] == 200


def test_json_cli_exits_nonzero_for_an_incomplete_result_from_the_running_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = _service(_incomplete())
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))

    def listening(overrides: Mapping[str, Any]) -> Path:
        del overrides
        return path

    monkeypatch.setattr(proxy, "listening", listening)
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def serve() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        ready.set()
        loop.run_forever()
        loop.run_until_complete(server.aclose())
        loop.close()

    thread = threading.Thread(target=serve, name="incomplete-control-server")
    thread.start()
    try:
        assert ready.wait(timeout=5), "the control server did not start"
        result = CliRunner().invoke(cli.app, ["--json", "connector", "sync", "synthetic-wiki"])
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive(), "the control server did not stop"

    assert result.exit_code == 1
    envelope = cast("dict[str, Any]", json.loads(result.stdout))
    assert envelope["ok"] is False
    assert cast("dict[str, Any]", envelope["data"])["outcome"] == "incomplete"
    assert cast("dict[str, Any]", envelope["error"])["type"] == "CursorExpiredError"


def test_http_and_mcp_report_the_same_incomplete_outcome() -> None:
    service, backend = _service(_incomplete())
    mcp = build_server(service)
    tool = asyncio.run(mcp.call_tool("connector_sync", {"name": "synthetic-wiki"}))

    with client_for(backend) as client:
        response = client.post("/api/v1/admin/connectors/synthetic-wiki/sync", json={})
    http = cast("dict[str, Any]", response.json())
    mcp_result = cast("dict[str, Any]", tool.structured_content)

    assert response.status_code == SERVICE_UNAVAILABLE
    assert mcp_result["ok"] is False
    assert http["ok"] is False
    assert cast("dict[str, Any]", mcp_result["data"])["outcome"] == "incomplete"
    assert cast("dict[str, Any]", http["data"])["outcome"] == "incomplete"
    assert (
        cast("dict[str, Any]", mcp_result["data"])["lifecycle"]
        == cast("dict[str, Any]", http["data"])["lifecycle"]
    )
    assert (
        cast("dict[str, Any]", mcp_result["error"])["type"]
        == cast("dict[str, Any]", http["error"])["type"]
    )


def test_http_and_mcp_share_effective_full_inventory_authority() -> None:
    report = RunReport(
        connector="synthetic-wiki",
        discovered=102,
        full_inventory_authority="direct_current_content",
        snapshot_completeness="complete",
        watermark_advanced=True,
        durable_acquired=3,
        durable_reused=99,
        reconciled_deleted_items=1,
    )
    service, backend = _service(report)
    backend.settings.connectors["synthetic-wiki"] = ConnectorSettings.model_validate(
        {
            "type": "confluence",
            "options": {
                "base_url": "https://wiki.example.test/confluence",
                "deployment": "server",
                "personal_access_token": "synthetic-token",
                "spaces": ["DOCS"],
                "full_inventory_authority": "direct_current_content",
            },
        }
    )
    mcp = build_server(service)
    tool = asyncio.run(mcp.call_tool("connector_sync", {"name": "synthetic-wiki"}))
    with client_for(backend) as client:
        http = cast(
            "dict[str, Any]",
            client.post("/api/v1/admin/connectors/synthetic-wiki/sync", json={}).json(),
        )
        web = client.get("/ui/connectors")
    mcp_result = cast("dict[str, Any]", tool.structured_content)
    for envelope in (http, mcp_result):
        data = cast("dict[str, Any]", envelope["data"])
        lifecycle = cast("dict[str, Any]", data["lifecycle"])
        assert data["full_inventory_authority"] == "direct_current_content"
        assert lifecycle["full_inventory_authority"] == "direct_current_content"
        rendered = json.dumps(envelope).lower()
        for private in ("source_id", "page title", "blob_hash", "cookie", "username"):
            assert private not in rendered
    listed = asyncio.run(service.connector_list())
    assert listed.connectors[0].full_inventory_authority == "direct_current_content"
    assert web.status_code == 200
    assert "direct_current_content" in web.text
    assert "DOCS" not in web.text


async def test_control_socket_preserves_aggregate_full_inventory_authority(
    tmp_path: Path,
) -> None:
    report = RunReport(
        connector="synthetic-wiki",
        discovered=102,
        full_inventory_authority="direct_current_content",
        snapshot_completeness="complete",
        watermark_advanced=True,
    )
    service, _ = _service(report)
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        envelope = await control.connect(
            path,
            control.Invoke(
                op="connector_sync", arguments={"name": "synthetic-wiki", "limit": None}
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    data = cast("dict[str, Any]", envelope["data"])
    lifecycle = cast("dict[str, Any]", data["lifecycle"])
    assert data["full_inventory_authority"] == "direct_current_content"
    assert lifecycle["full_inventory_authority"] == "direct_current_content"
    rendered = json.dumps(envelope).lower()
    for private in ("source_id", "page title", "blob_hash", "cookie", "username"):
        assert private not in rendered


def test_unknown_connector_authority_cannot_become_an_aggregate_data_channel() -> None:
    report = RunReport(connector="synthetic-wiki")
    cast("Any", report).full_inventory_authority = "private-space-DOCS"
    service, _ = _service(report)
    result = asyncio.run(service.connector_sync("synthetic-wiki"))

    assert result.full_inventory_authority == ""
    assert result.lifecycle.full_inventory_authority == ""
    assert "private-space" not in result.model_dump_json()


async def test_scheduler_counts_a_returned_incomplete_report_as_a_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    service, _ = _service(_incomplete())

    class Clock:
        def __init__(self) -> None:
            self.arrived = asyncio.Event()
            self.release = asyncio.Event()

        async def sleep(self, seconds: float) -> None:
            del seconds
            self.arrived.set()
            await self.release.wait()
            self.release.clear()

    clock = Clock()
    scheduler = Scheduler(service, {"synthetic-wiki": 60}, sleep=clock.sleep)
    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        clock.arrived.clear()
        clock.release.set()
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
    finally:
        await scheduler.aclose()

    recorded = scheduler.scheduled["synthetic-wiki"]
    assert recorded.runs == 0
    assert recorded.failures == 1
    assert recorded.last_outcome == "incomplete"
    assert recorded.retry_required is True
    assert recorded.last_error_type == "CursorExpiredError"
    assert recorded.last_lifecycle is not None
    assert recorded.last_lifecycle.outcome == "incomplete"
    assert recorded.last_lifecycle.enumerated_items == 200
    assert "will be retried" in capsys.readouterr().err


async def test_bounded_and_durable_document_failure_outcomes_are_not_reclassified() -> None:
    bounded, _ = _service(RunReport(connector="synthetic-wiki", discovered=10, limited=True))
    bounded_envelope = await _envelope(bounded)
    assert bounded_envelope.ok is True
    assert bounded_envelope.data is not None
    assert bounded_envelope.data["outcome"] == "bounded"
    assert bounded_envelope.data["retry_required"] is False

    durable, _ = _service(
        RunReport(connector="synthetic-wiki", discovered=1, by_status={"failed": 1})
    )
    durable_envelope = await _envelope(durable)
    assert durable_envelope.ok is True
    assert durable_envelope.data is not None
    assert durable_envelope.data["outcome"] == "complete"


async def test_connector_list_exposes_the_last_machine_readable_outcome() -> None:
    class DiagnosticStore(FakeStore):
        @override
        async def connector_metadata(self, connector: str) -> dict[str, object]:
            assert connector == "synthetic-wiki"
            return {
                "last_run": {
                    "outcome": "incomplete",
                    "retry_required": True,
                    "error_type": "CursorExpiredError",
                    "enumeration_completed": False,
                    "watermark_advanced": False,
                }
            }

    service, backend = _service(_incomplete())
    backend.store = DiagnosticStore()
    listed = await service.connector_list()
    summary = listed.connectors[0]
    assert summary.last_outcome == "incomplete"
    assert summary.retry_required is True
    assert summary.last_error_type == "CursorExpiredError"
    assert summary.last_enumeration_completed is False
    assert summary.last_watermark_advanced is False


async def test_connector_metadata_and_list_share_the_closed_lifecycle_status() -> None:
    report = _capacity_incomplete()

    class DiagnosticStore(FakeStore):
        @override
        async def connector_metadata(self, connector: str) -> dict[str, object]:
            assert connector == "synthetic-wiki"
            return cast("dict[str, object]", report.as_metadata())

    service, backend = _service(report)
    backend.store = DiagnosticStore()
    listed = await service.connector_list()
    lifecycle = listed.connectors[0].last_lifecycle

    assert lifecycle is not None
    assert lifecycle.outcome == "refused"
    assert lifecycle.refusal is not None
    assert lifecycle.refusal.resource == "journal_metadata_bytes"
    serialized = lifecycle.model_dump_json()
    for private in ("source_id", "private-source-id", "private.invalid", "secret", "token="):
        assert private not in serialized.lower()


# --- collection placement -----------------------------------------------------------------


def _clean() -> RunReport:
    """A run that went perfectly: everything discovered, everything indexed, nothing amiss."""
    return RunReport(connector="synthetic-wiki", discovered=3, by_status={"indexed": 3})


async def test_a_sync_reports_how_much_of_its_source_no_collection_holds() -> None:
    """The number that makes "503 indexed, 0 failed, outcome complete" readable.

    Every counter a run keeps is about what it did, and all of them can be perfect while the
    corpus the run contributed to answers nothing a collection-scoped search asks. This is the
    one fact in the report that is measured afterwards rather than counted during, and a run
    that placed nothing anywhere says so in its own output instead of in a later refusal.
    """
    service, backend = _service(_clean())
    for index in range(3):
        document = backend.store.add(
            make_document(backend.workspace, source="synthetic-wiki", source_id=f"page-{index}.md")
        )
        backend.organization_.documents[document.id] = document

    payload = await service.connector_sync("synthetic-wiki")

    assert payload.outcome == "complete"
    assert payload.ingested == 3
    assert payload.collected == 0
    assert payload.uncollected == 3


async def test_a_sync_counts_the_source_it_ran_over_and_not_the_whole_workspace() -> None:
    """Scoped to the connector, because the report is about this source's corpus.

    A workspace-wide figure would move when a second connector synced, which makes a number
    printed under one source's name a fact about another one. The documents a run skipped as
    unchanged stay in, though: membership is evaluated rather than stored, so they are in
    exactly the collections the freshly indexed ones are.
    """
    service, backend = _service(_clean())
    mine = backend.store.add(
        make_document(backend.workspace, source="synthetic-wiki", source_id="mine.md")
    )
    theirs = backend.store.add(
        make_document(backend.workspace, source="other-source", source_id="theirs.md")
    )
    for document in (mine, theirs):
        backend.organization_.documents[document.id] = document
    collection = await backend.organization_.create_collection("alpha")
    await backend.organization_.add_to_collection(collection.id, [mine.id])

    payload = await service.connector_sync("synthetic-wiki")

    assert payload.collected == 1, "the collection holding this source's document was not seen"
    assert payload.uncollected == 0, "another source's uncollected document was counted here"


async def test_a_run_that_could_not_count_membership_says_so_rather_than_reporting_none_held(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``None`` rather than zero, and the sync still succeeds.

    Zero documents in no collection is the *healthy* answer, so a path that failed to ask must
    not be able to report it. And the sync itself ran, indexed a corpus and advanced its
    watermark: failing it over a diagnostic count would report the opposite of what happened.
    """
    service, backend = _service(_clean())

    async def refuse() -> Never:
        msg = "the database is locked"
        raise StorageBusyError(msg)

    backend.organization = refuse

    payload = await service.connector_sync("synthetic-wiki")

    assert payload.outcome == "complete", "a diagnostic count failed the run it was reporting on"
    assert payload.ingested == 3
    assert payload.collected is None
    assert payload.uncollected is None
    assert "collection membership could not be counted" in caplog.text


async def test_a_delete_racing_the_two_counts_does_not_fail_the_run_it_describes() -> None:
    """The counts are two statements, and nothing holds a lock across them.

    A document removed between the uncollected count and the total leaves the first above the
    second, so a bare `total - uncollected` is negative. `collected` is declared `ge=0`, and
    which way that breaks depends on how the field is set: constructed, it raises and a
    completed sync reports a validation error instead of its result; attached with
    `model_copy`, which does not re-validate, it serializes a negative document count into the
    envelope. Reconciling the pair is what makes neither reachable.
    """
    service, backend = _service(_clean())
    for index in range(3):
        document = backend.store.add(
            make_document(backend.workspace, source="synthetic-wiki", source_id=f"page-{index}.md")
        )
        backend.organization_.documents[document.id] = document
    # The organization store still sees three; the document store has already lost one, which is
    # what the two statements see either side of a delete.
    backend.store.documents.popitem()

    payload = await service.connector_sync("synthetic-wiki")

    assert payload.outcome == "complete"
    assert payload.collected == 0
    assert payload.uncollected == 2, "the pair was not reconciled against the total it came with"


async def test_an_import_leaves_collection_placement_unmeasured(tmp_path: Path) -> None:
    """`connector` on an import is a run label, not a source, and zero would be a lie.

    Every entry is ingested under the source the archive recorded for it, and an archive may
    carry several; nothing is ever filed under `"import"`. Counting that name matches no
    document and reports `0` in no collection — the *healthy* answer to a question never asked,
    which is the confident zero `None` exists to keep out of this field.
    """
    service, backend = _service(
        RunReport(connector="import", discovered=2, by_status={"indexed": 2})
    )
    for index in range(2):
        document = backend.store.add(
            make_document(backend.workspace, source="confluence", source_id=f"page-{index}.md")
        )
        backend.organization_.documents[document.id] = document
    archive = tmp_path / "corpus.tar.gz"
    archive.write_bytes(b"not read by the fake")

    payload = await service.import_corpus(archive)

    assert payload.ingested == 2
    assert payload.collected is None, (
        "an import reported a placement count for a source nothing uses"
    )
    assert payload.uncollected is None


async def test_a_rule_this_build_cannot_read_does_not_fail_the_sync_it_describes() -> None:
    """`CollectionRule` forbids unknown fields, so a newer manicule's rule raises on an older one.

    Counting reads every stored rule through `model_validate`, deliberately — a hand-edited
    `auto_rules` row should fail where it is read rather than quietly select a different set.
    Raising out of a read is right; raising out of a completed sync, and taking the result of
    an ingest that already happened with it, is not.
    """
    service, backend = _service(_clean())
    document = backend.store.add(
        make_document(backend.workspace, source="synthetic-wiki", source_id="page.md")
    )
    backend.organization_.documents[document.id] = document

    async def refuse(*, source: str | None = None) -> Never:
        del source
        raise ValidationError.from_exception_data("CollectionRule", [])

    backend.organization_.count_uncollected = refuse

    payload = await service.connector_sync("synthetic-wiki")

    assert payload.outcome == "complete", "a rule this build cannot read failed the run"
    assert payload.ingested == 3
    assert payload.collected is None
    assert payload.uncollected is None
