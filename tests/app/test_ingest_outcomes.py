"""Incomplete ingest is one outcome at every automation boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING, Any, Never, cast, override

import pytest
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
from manicule.ingest.capacity import CapacityDiagnostic, CapacityRefusedError, CapacityResource
from manicule.ingest.pipeline import RunReport
from manicule.mcp.server import build_server
from tests.api.support import client_for
from tests.app.fakes import FakeBackend, FakeStore

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
        cast("dict[str, Any]", mcp_result["error"])["type"]
        == cast("dict[str, Any]", http["error"])["type"]
    )


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
