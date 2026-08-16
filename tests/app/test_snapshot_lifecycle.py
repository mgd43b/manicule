"""Durable snapshot status is one private-safe result on every read surface."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, cast

from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.core.acquisition import (
    AcquisitionRun,
    AcquisitionRunState,
    SnapshotCompleteness,
    SnapshotPromotionPolicy,
)
from manicule.core.rebuild import (
    RebuildEstimate,
    RebuildRefusalCode,
    RebuildRefusedError,
)
from manicule.core.sources import Watermark
from manicule.mcp.server import build_server
from tests.api.support import client_for
from tests.app.fakes import FakeBackend

_NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def _backend() -> FakeBackend:
    backend = FakeBackend()
    backend.settings.connectors["synthetic-wiki"] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )
    backend.ingestion_.snapshot = AcquisitionRun(
        id="snapshot-aggregate-1",
        workspace_id=backend.workspace,
        connector_id="connector-aggregate-1",
        connector="synthetic-wiki",
        source_scope="scope",
        scope_fingerprint="scope-fingerprint",
        promotion_policy=SnapshotPromotionPolicy.REQUIRE_COMPLETE,
        state=AcquisitionRunState.INDEXING,
        candidate_watermark=Watermark(value="cursor-1000", observed_at=_NOW),
        enumeration_completed_at=_NOW,
        acquisition_completed_at=_NOW,
        promoted_at=_NOW,
        watermark_committed_at=_NOW,
        membership_hash="manifest-digest",
        completeness=SnapshotCompleteness.COMPLETE,
        lease_generation=1,
        discovered_count=1_000,
        acquired_count=1_000,
        indexed_count=250,
        unchanged_count=0,
        retry_count=0,
        metadata_bytes=50_000,
        acquired_blob_bytes=200_000,
        created_at=_NOW,
        updated_at=_NOW,
    )
    return backend


def test_http_and_mcp_return_the_same_aggregate_snapshot_status() -> None:
    backend = _backend()
    service = ApplicationService(backend)
    mcp = build_server(service)
    tool = asyncio.run(mcp.call_tool("snapshot_status", {"name": "synthetic-wiki"}))

    with client_for(backend) as client:
        response = client.get("/api/v1/admin/connectors/synthetic-wiki/snapshot")

    http = cast("dict[str, Any]", response.json())
    mcp_result = cast("dict[str, Any]", tool.structured_content)
    assert response.status_code == 200
    http_lifecycle = cast("dict[str, Any]", cast("dict[str, Any]", http["data"])["lifecycle"])
    mcp_lifecycle = cast("dict[str, Any]", cast("dict[str, Any]", mcp_result["data"])["lifecycle"])
    assert {
        key: value for key, value in http_lifecycle.items() if key != "oldest_backlog_age_seconds"
    } == {key: value for key, value in mcp_lifecycle.items() if key != "oldest_backlog_age_seconds"}
    data = cast("dict[str, Any]", http["data"])
    lifecycle = cast("dict[str, Any]", data["lifecycle"])
    assert lifecycle["enumerated_items"] == 1_000
    assert lifecycle["acquired_items"] == 1_000
    assert lifecycle["pending_items"] == 750
    assert lifecycle["backlog_bytes"] == 200_000
    assert lifecycle["can_continue_offline"] is True


def test_verify_is_read_only_aggregate_and_contains_no_manifest_member_detail() -> None:
    backend = _backend()
    service = ApplicationService(backend)
    mcp = build_server(service)
    tool = asyncio.run(mcp.call_tool("snapshot_verify", {"snapshot_id": "snapshot-aggregate-1"}))
    result = cast("dict[str, Any]", tool.structured_content)

    assert result["ok"] is True
    data = cast("dict[str, Any]", result["data"])
    assert data["verified"] is True
    assert data["verification_performed"] is True
    rendered = str(result).lower()
    for private in ("source_id", "uri", "title", "body", "secret", "token="):
        assert private not in rendered


def test_corrupt_verify_is_a_failed_envelope_on_http_and_mcp() -> None:
    backend = _backend()
    backend.ingestion_.snapshot_verified = False
    service = ApplicationService(backend)
    mcp = build_server(service)
    tool = asyncio.run(mcp.call_tool("snapshot_verify", {"snapshot_id": "snapshot-aggregate-1"}))

    with client_for(backend) as client:
        response = client.get("/api/v1/admin/snapshots/snapshot-aggregate-1/verify")

    http = cast("dict[str, Any]", response.json())
    result = cast("dict[str, Any]", tool.structured_content)
    assert response.status_code >= 400
    assert http["ok"] is result["ok"] is False
    assert cast("dict[str, Any]", http["error"])["type"] == "SnapshotVerificationError"
    assert cast("dict[str, Any]", result["error"])["type"] == "SnapshotVerificationError"


async def test_rebuild_plan_is_deferred_and_live_status_reports_remaining_items() -> None:
    service = ApplicationService(_backend())

    plan = await service.rebuild_plan("snapshot-aggregate-1")
    status = await service.rebuild_status(plan.generation_id)

    assert plan.lifecycle.dry_run
    assert plan.lifecycle.outcome == "deferred"
    assert status.expected_items == 2
    assert status.lifecycle.pending_items == 1
    assert status.lifecycle.estimated_remaining_items == 1


async def test_rebuild_refusal_is_a_typed_failure_envelope() -> None:
    estimate = RebuildEstimate(
        generation_id="aggregate-generation",
        snapshot_run_id="aggregate-snapshot",
        documents=2,
        expected_items=2,
        known_source_bytes=10,
        estimated_chunks=2,
        estimated_seconds=1,
        estimated_peak_memory_bytes=100,
        estimated_temporary_bytes=200,
        missing_count=1,
        refusal=RebuildRefusalCode.MISSING_LOCAL_INPUT,
    )

    async def refused():  # noqa: ANN202 - inferred failure-only coroutine
        raise RebuildRefusedError(RebuildRefusalCode.MISSING_LOCAL_INPUT, estimate)

    envelope = await run_op("rebuild_run", "default", refused)

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "RebuildRefusedError"
    assert envelope.error.message == "missing_local_input"
