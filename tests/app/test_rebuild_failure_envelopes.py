"""Expected rebuild failures have one bounded meaning on every unattended surface."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy.exc import IntegrityError
from typer.testing import CliRunner

from manicule.api.envelopes import CONFLICT, SERVICE_UNAVAILABLE, UNPROCESSABLE, status_for
from manicule.app import control
from manicule.app.dispatch import error_info, run_op
from manicule.app.results import failed
from manicule.app.served import ControlHandler
from manicule.app.service import (
    ApplicationService,
    _rebuild_run_report,  # pyright: ignore[reportPrivateUsage]
)
from manicule.cli import main as cli
from manicule.connectors.sessions import SessionVault
from manicule.core.rebuild import (
    RebuildCheckpoint,
    RebuildDerivationError,
    RebuildLeaseError,
    RebuildPublicationConflictError,
    RebuildPublicationValidationError,
    RebuildRefusalCode,
    RebuildState,
    RebuildStorageCause,
    RebuildStorageDiagnostic,
    RebuildStorageError,
    RebuildStorageStage,
    RebuildTerminalError,
    RebuildTerminalGenerationError,
    RebuildValidationError,
)
from manicule.mcp.server import build_server
from manicule.web.rendering import panel
from tests.api.support import client_for
from tests.app.fakes import FakeBackend

PRIVATE_MARKERS = (
    "private body",
    "wiki.example.test",
    "cookie=secret",
    "insert into",
    "/private/machine",
    "traceback",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (RebuildStorageError(), SERVICE_UNAVAILABLE),
        (RebuildDerivationError(), UNPROCESSABLE),
        (RebuildValidationError(), UNPROCESSABLE),
        (RebuildLeaseError(), CONFLICT),
        (RebuildTerminalError(), CONFLICT),
    ],
)
async def test_expected_rebuild_failures_have_stable_status_and_recovery_guidance(
    failure: Exception, expected_status: int
) -> None:
    async def failing():  # noqa: ANN202 - failure-only operation
        raise failure

    envelope = await run_op("rebuild_run", "default", failing)

    assert envelope.ok is False
    assert envelope.error == error_info(failure)
    assert envelope.error is not None
    assert envelope.error.hint
    assert status_for(envelope) == expected_status
    rendered = str(envelope.as_json()).lower()
    assert not any(marker in rendered for marker in PRIVATE_MARKERS)


def test_rebuild_status_exposes_only_the_safe_storage_diagnostic() -> None:
    diagnostic = RebuildStorageDiagnostic(
        stage=RebuildStorageStage.VALIDATION_CHECKPOINT,
        cause=RebuildStorageCause.CAPACITY,
        retryable=False,
        namespace_usable=False,
        occurred_at=datetime.now(UTC),
        correlation_id="4f" * 16,
        operator_hint="Free durable storage, then resume the same generation.",
    )
    report = _rebuild_run_report(
        RebuildCheckpoint(
            generation_id="generation-public",
            state=RebuildState.VALIDATING,
            next_sequence=3,
            documents_built=3,
            chunks_built=6,
            vectors_reused=2,
            vectors_embedded=4,
            diagnostic_code=RebuildRefusalCode.STORAGE_FAILED,
            storage_diagnostic=diagnostic,
        )
    )

    assert report.lifecycle.phase == "operator_required"
    assert report.lifecycle.outcome == "operator_required"
    assert report.lifecycle.can_continue_offline is False
    assert report.storage_diagnostic is not None
    assert report.storage_diagnostic.stage == "validation_checkpoint"
    assert report.storage_diagnostic.cause == "capacity"
    assert report.storage_diagnostic.operator_hint == diagnostic.operator_hint
    rendered = str(report.model_dump(mode="json")).lower()
    assert not any(marker in rendered for marker in PRIVATE_MARKERS)


def test_rebuild_status_labels_an_active_takeover_replay() -> None:
    report = _rebuild_run_report(
        RebuildCheckpoint(
            generation_id="generation-public",
            state=RebuildState.BUILDING,
            next_sequence=3,
            documents_built=3,
            chunks_built=6,
            vectors_reused=2,
            vectors_embedded=4,
            lease_owner="private-owner-token",
            lease_generation=2,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            takeover_replay=True,
        )
    )

    assert report.lifecycle.phase == "takeover_replay"
    assert report.lifecycle.outcome == "running"


@pytest.mark.parametrize(
    ("internal", "expected_type", "expected_status"),
    [
        (
            IntegrityError(
                "INSERT INTO private_table VALUES (?)",
                ("private body cookie=secret",),
                RuntimeError("/private/machine/workspace.sqlite"),
            ),
            "RebuildStorageError",
            SERVICE_UNAVAILABLE,
        ),
        (
            RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED),
            "RebuildLeaseError",
            CONFLICT,
        ),
        (
            RebuildPublicationValidationError(),
            "RebuildValidationError",
            UNPROCESSABLE,
        ),
    ],
)
def test_http_mcp_and_web_share_the_same_private_safe_rebuild_failure(
    internal: Exception, expected_type: str, expected_status: int
) -> None:
    backend = FakeBackend()
    backend.ingestion_.rebuild_failure = internal
    service = ApplicationService(backend)

    with client_for(backend) as client:
        response = client.get("/api/v1/admin/rebuild/generations/generation-public")
    http = cast("dict[str, Any]", response.json())
    mcp = asyncio.run(
        build_server(service).call_tool("rebuild_status", {"generation_id": "generation-public"})
    )
    mcp_result = cast("dict[str, Any]", mcp.structured_content)
    web = asyncio.run(
        panel(
            "rebuild_status",
            service,
            lambda: service.rebuild_status("generation-public"),
        )
    ).envelope.as_json()

    assert response.status_code == expected_status
    assert http == mcp_result == web
    assert cast("dict[str, Any]", http["error"])["type"] == expected_type
    rendered = str(http).lower()
    assert not any(marker in rendered for marker in PRIVATE_MARKERS)


@pytest.mark.asyncio
async def test_served_control_returns_terminal_rebuild_failure_as_an_envelope(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    backend.ingestion_.rebuild_failure = RebuildTerminalGenerationError("private terminal state")
    service = ApplicationService(backend)
    path = control.socket_path(tmp_path)
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        envelope = await control.connect(
            path,
            control.Invoke(op="rebuild_run", arguments={"snapshot_id": "snapshot-public"}),
            on_progress=lambda _message: None,
        )
    finally:
        await server.aclose()

    assert envelope["ok"] is False
    error = cast("dict[str, Any]", envelope["error"])
    assert error["type"] == "RebuildTerminalError"
    assert error["hint"]
    assert not any(marker in str(envelope).lower() for marker in PRIVATE_MARKERS)


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_rebuild_failure_is_structured_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, json_output: bool
) -> None:
    expected = failed("rebuild_run", "default", error_info(RebuildStorageError()))

    async def dispatch(_command: object):  # noqa: ANN202 - exact private CLI seam
        return expected

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    arguments = ["--json"] if json_output else []
    result = CliRunner().invoke(
        cli.app,
        [*arguments, "rebuild", "execute", "snapshot-public"],
    )

    assert result.exit_code == 1
    if json_output:
        assert json.loads(result.stdout) == expected.as_json()
    else:
        assert "RebuildStorageError" in result.stderr
        assert "cleanup plan" in result.stderr
    rendered = (result.stdout + result.stderr).lower()
    assert not any(marker in rendered for marker in PRIVATE_MARKERS)
