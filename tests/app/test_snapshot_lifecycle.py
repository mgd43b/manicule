"""Durable snapshot status is one private-safe result on every read surface."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
        full_inventory_authority="direct_current_content",
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


async def test_an_unfinished_run_whose_lease_has_lapsed_is_not_reported_as_running() -> None:
    """The distinction an operator needs and could not previously make.

    ``running`` used to mean "unfinished and no diagnostic recorded", which a worker that lost
    its acquisition lease cannot correct on its way out — losing the lease is precisely the
    loss of the right to write the row. So a run being actively derived and a run nobody has
    touched since a worker died projected the same word, and the difference between them is
    whether waiting is the correct thing to do.

    The failure that produced this: one synthetic multi-megabyte block held the event loop
    inside synchronous preparation for longer than the lease, the heartbeat coroutine could
    not run, and the run was lost while its snapshot stayed complete and resumable. Status
    still said the sync was running. Nothing was.
    """
    backend = _backend()
    run = backend.ingestion_.snapshot
    assert run is not None

    live = run.model_copy(
        update={
            "lease_owner": "pipeline:synthetic-owner",
            "lease_expires_at": datetime.now(UTC) + timedelta(minutes=5),
        }
    )
    backend.ingestion_.snapshot = live
    working = await ApplicationService(backend).snapshot_status("synthetic-wiki")

    lapsed = live.model_copy(update={"lease_expires_at": datetime.now(UTC) - timedelta(minutes=5)})
    backend.ingestion_.snapshot = lapsed
    idle = await ApplicationService(backend).snapshot_status("synthetic-wiki")

    released = live.model_copy(update={"lease_owner": None, "lease_expires_at": None})
    backend.ingestion_.snapshot = released
    after_release = await ApplicationService(backend).snapshot_status("synthetic-wiki")

    assert working.lifecycle.outcome == "running", (
        "a live lease is the one thing that makes 'running' true, so it has to still say it"
    )
    assert idle.lifecycle.outcome == "incomplete", (
        "an expired lease is a run with no worker: unfinished, inactive, and resumable from "
        "what is already committed"
    )
    assert after_release.lifecycle.outcome == "incomplete", (
        "a lease released by a canceled run is the same situation reached politely"
    )
    assert idle.lifecycle.can_continue_offline == working.lifecycle.can_continue_offline, (
        "losing a lease costs the run its worker, not its retained bytes"
    )


def test_a_lapsed_lease_reads_the_same_on_http_and_mcp() -> None:
    """The lease-derived value is a surface fact, so both surfaces have to carry it.

    The test above asserts parity on a run whose fixture leaves the lease fields unset, which
    now projects ``incomplete`` — so it would keep passing if ``running`` were computed on only
    one of the two surfaces, or on neither. This pins the distinction itself: the same run, once
    with a live lease and once expired, has to read the same way through both.
    """
    backend = _backend()
    run = backend.ingestion_.snapshot
    assert run is not None
    owned = {
        "lease_owner": "pipeline:synthetic-owner",
        "lease_expires_at": datetime.now(UTC) + timedelta(minutes=5),
    }

    seen: dict[str, set[str]] = {}
    for label, update in (
        ("live", owned),
        ("lapsed", {**owned, "lease_expires_at": datetime.now(UTC) - timedelta(minutes=5)}),
    ):
        backend.ingestion_.snapshot = run.model_copy(update=update)
        service = ApplicationService(backend)
        tool = asyncio.run(
            build_server(service).call_tool("snapshot_status", {"name": "synthetic-wiki"})
        )
        with client_for(backend) as client:
            response = client.get("/api/v1/admin/connectors/synthetic-wiki/snapshot")
        assert response.status_code == 200
        http = cast("dict[str, Any]", cast("dict[str, Any]", response.json())["data"])
        mcp = cast("dict[str, Any]", cast("dict[str, Any]", tool.structured_content)["data"])
        http_outcome = cast("dict[str, Any]", http["lifecycle"])["outcome"]
        mcp_outcome = cast("dict[str, Any]", mcp["lifecycle"])["outcome"]
        assert http_outcome == mcp_outcome, (
            f"the {label} lease reads {http_outcome!r} over HTTP and {mcp_outcome!r} over MCP"
        )
        seen[label] = {cast("str", http_outcome)}

    assert seen["live"] == {"running"}
    assert seen["lapsed"] == {"incomplete"}, (
        "an expired lease has to reach both surfaces as an unowned run, or an operator reading "
        "one of them still cannot tell active work from resumable work nobody is doing"
    )


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
    assert data["full_inventory_authority"] == "direct_current_content"
    assert lifecycle["full_inventory_authority"] == "direct_current_content"


async def test_unchanged_members_do_not_hide_pending_acquired_derivation() -> None:
    backend = _backend()
    snapshot = backend.ingestion_.snapshot
    assert snapshot is not None
    backend.ingestion_.snapshot = snapshot.model_copy(
        update={"discovered_count": 1_200, "unchanged_count": 200, "reused_count": 200}
    )

    status = await ApplicationService(backend).snapshot_status("synthetic-wiki")

    assert status.lifecycle.reused_items == 200
    assert status.lifecycle.pending_items == 750


async def test_settled_partial_snapshot_keeps_source_retry_pending() -> None:
    backend = _backend()
    snapshot = backend.ingestion_.snapshot
    assert snapshot is not None
    backend.ingestion_.snapshot = snapshot.model_copy(
        update={
            "state": AcquisitionRunState.SETTLED,
            "completeness": SnapshotCompleteness.PARTIAL,
            "promotion_policy": SnapshotPromotionPolicy.ALLOW_OMISSIONS,
            "discovered_count": 3,
            "acquired_count": 2,
            "indexed_count": 2,
            "omission_count": 1,
            "acquired_blob_bytes": 0,
        }
    )

    status = await ApplicationService(backend).snapshot_status("synthetic-wiki")

    assert status.lifecycle.phase == "acquiring"
    assert status.lifecycle.outcome == "incomplete"
    assert status.lifecycle.pending_items == status.lifecycle.backlog_items == 1
    assert status.lifecycle.can_continue_offline is False


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


async def test_rebuild_plan_missing_inputs_is_a_typed_refused_surface_result() -> None:
    backend = _backend()
    backend.ingestion_.rebuild_missing_count = 1

    plan = await ApplicationService(backend).rebuild_plan("snapshot-aggregate-1")

    assert not plan.runnable
    assert plan.refusal_code == "missing_local_input"
    assert plan.lifecycle.outcome == "refused"
    assert plan.lifecycle.refusal is not None
    assert plan.lifecycle.refusal.code == "missing_local_input"


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
