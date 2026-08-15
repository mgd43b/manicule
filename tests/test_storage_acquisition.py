"""The durable source-acquisition boundary against migrated SQLite."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionFailureCode,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
    AcquisitionStage,
    SnapshotCompleteness,
    SnapshotItemOutcome,
    SnapshotPromotionPolicy,
)
from manicule.core.content import Metadata, RawDocument
from manicule.core.errors import UnknownEntityError
from manicule.core.provenance import PROVENANCE_KEY
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.ingest.capacity import CapacityRefusedError, CapacityResource
from manicule.storage.acquisition import (
    AcquisitionConflictError,
    AcquisitionCoverageError,
    AcquisitionWatermarkConflictError,
    InvalidAcquisitionTransitionError,
)
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.migrator import current, downgrade, upgrade
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


def _watermark(value: str) -> Watermark:
    return Watermark(value=value, observed_at=datetime(2026, 8, 15, tzinfo=UTC))


def _source(
    source_id: str = "page-1",
    *,
    uri: str | None = None,
    version_token: str = "v1",  # noqa: S107 - source revision, not a credential
) -> AcquisitionSource:
    discovered = DiscoveredDoc(
        ref=DocRef(
            source_id=source_id,
            uri=uri or f"https://example.test/pages/{source_id}",
            metadata={"opaque_id": source_id},
        ),
        version_token=version_token,
        title="A durable title",
        media_type="text/html",
        size_bytes=12,
        metadata={"space": "docs"},
    )
    return AcquisitionSource.from_discovered(
        discovered,
        source_modified_at=datetime(2026, 8, 15, 12, tzinfo=UTC),
        provenance={"source_kind": "synthetic"},
    )


def test_discovered_citation_provenance_is_copied_into_the_durable_manifest() -> None:
    provenance: Metadata = {
        "source": {"title": "Canonical title", "uri": "https://source.test/p/1"}
    }
    discovered = DiscoveredDoc(
        ref=DocRef(source_id="page-1", uri="https://cache.test/page-1"),
        version_token="v1",  # noqa: S106 - source revision, not a credential
        title="Local title",
        media_type="text/html",
        metadata={PROVENANCE_KEY: provenance},
    )

    source = AcquisitionSource.from_discovered(discovered)

    assert source.provenance == provenance
    assert source.metadata[PROVENANCE_KEY] == provenance


def _acquired(data: bytes, source_id: str = "page-1") -> AcquiredSource:
    return AcquiredSource.from_raw(
        RawDocument(
            source_id=source_id,
            uri=f"https://example.test/pages/{source_id}",
            media_type="text/html",
            content=data,
        )
    )


_NOW = datetime(2026, 8, 15, 13, tzinfo=UTC)


async def _retain_record(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    run: AcquisitionRun,
    *,
    source_id: str = "page-1",
    owner: str = "worker",
    version_token: str = "v1",  # noqa: S107 - source revision, not a credential
) -> str:
    body = f"retained {source_id}".encode()
    blob = await BlobStore(engine, data_dir).put(body, "text/html")
    assert isinstance(blob, StoredBlob)
    await store.transition_acquisition_record(
        run.id,
        source_id,
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner=owner,
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        run.id,
        source_id,
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner=owner,
        lease_generation=run.lease_generation,
        now=_NOW,
        blob_ref=blob.hash,
        acquired_source=_acquired(body, source_id),
        fetched_version_token=version_token,
    )
    return blob.hash


async def _claimed_run(
    store: SqliteDocStore,
    run_id: str = "run",
    connector: str = "wiki",
    owner: str = "worker",
) -> AcquisitionRun:
    await store.create_acquisition_run(run_id, connector)
    claimed = await store.claim_acquisition_run(
        run_id, owner, now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert claimed is not None
    return claimed


async def test_append_is_durable_idempotent_and_run_scoped(store: SqliteDocStore) -> None:
    run = await _claimed_run(store, "run-1")
    first = await store.append_acquisition_record(
        run.id,
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    repeated = await store.append_acquisition_record(
        run.id,
        99,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )

    assert first == repeated
    persisted = await store.get_acquisition_run(run.id)
    assert persisted is not None
    assert persisted.discovered_count == 1
    assert [record.source.source_id for record in await store.list_acquisition_records(run.id)] == [
        "page-1"
    ]

    with pytest.raises(AcquisitionConflictError, match="different data"):
        await store.append_acquisition_record(
            run.id,
            1,
            _source(uri="https://example.test/moved"),
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )


async def test_records_page_forward_by_sequence_without_loading_the_run(
    store: SqliteDocStore,
) -> None:
    run = await _claimed_run(store)
    for sequence in range(5):
        await store.append_acquisition_record(
            run.id,
            sequence,
            _source(f"page-{sequence}"),
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )

    first = await store.list_acquisition_records(run.id, limit=2)
    second = await store.list_acquisition_records(
        run.id, after_sequence=first[-1].sequence, limit=2
    )
    third = await store.list_acquisition_records(
        run.id, after_sequence=second[-1].sequence, limit=2
    )

    assert [[record.sequence for record in page] for page in (first, second, third)] == [
        [0, 1],
        [2, 3],
        [4],
    ]


async def test_concurrent_journal_reservations_cannot_exceed_the_record_limit(
    engine: AsyncEngine,
) -> None:
    limited = SqliteDocStore(engine, max_journal_records=1)
    await limited.ensure_workspace()
    run = await _claimed_run(limited)

    outcomes = await asyncio.gather(
        *(
            limited.append_acquisition_record(
                run.id,
                sequence,
                _source(f"page-{sequence}"),
                lease_owner="worker",
                lease_generation=run.lease_generation,
                now=_NOW,
            )
            for sequence in range(2)
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
    refused = [outcome for outcome in outcomes if isinstance(outcome, CapacityRefusedError)]
    assert len(refused) == 1
    assert refused[0].diagnostic.resource is CapacityResource.JOURNAL_RECORDS
    persisted = await limited.get_acquisition_run(run.id)
    assert persisted is not None
    assert persisted.discovered_count == 1
    assert len(await limited.list_acquisition_records(run.id)) == 1


async def test_metadata_refusal_acknowledges_nothing_and_exposes_only_aggregates(
    engine: AsyncEngine,
) -> None:
    limited = SqliteDocStore(engine, max_journal_metadata_bytes=1)
    await limited.ensure_workspace()
    run = await _claimed_run(limited)
    private_title = "private roadmap cinder"
    private_url = "https://example.test/page?token=fake-secret-cinder"
    source = _source(uri=private_url).model_copy(update={"title": private_title})

    with pytest.raises(CapacityRefusedError) as caught:
        await limited.append_acquisition_record(
            run.id,
            0,
            source,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
    rendered = f"{caught.value!s}\n{caught.value!r}\n{caught.value.diagnostic.as_metadata()!r}"
    assert private_title not in rendered
    assert private_url not in rendered
    assert "fake-secret-cinder" not in rendered
    assert caught.value.diagnostic.resource is CapacityResource.JOURNAL_METADATA_BYTES
    persisted = await limited.get_acquisition_run(run.id)
    assert persisted is not None
    assert persisted.discovered_count == 0
    assert persisted.metadata_bytes == 0
    assert await limited.list_acquisition_records(run.id) == []


async def test_journal_disk_headroom_refuses_before_acknowledgment(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited = SqliteDocStore(engine, min_disk_headroom_bytes=8)
    await limited.ensure_workspace()
    run = await _claimed_run(limited)

    def nearly_full(path: object) -> SimpleNamespace:
        del path
        return SimpleNamespace(total=100, used=90, free=10)

    monkeypatch.setattr("manicule.storage.acquisition.shutil.disk_usage", nearly_full)

    with pytest.raises(CapacityRefusedError) as caught:
        await limited.append_acquisition_record(
            run.id,
            0,
            _source(),
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )

    assert caught.value.diagnostic.resource is CapacityResource.DISK_HEADROOM_BYTES
    persisted = await limited.get_acquisition_run(run.id)
    assert persisted is not None
    assert persisted.discovered_count == 0
    assert await limited.list_acquisition_records(run.id) == []


async def test_below_floor_recovery_and_idempotency_can_shrink_or_settle_backlog(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited = SqliteDocStore(engine, min_disk_headroom_bytes=8)
    await limited.ensure_workspace()
    lease = await _claimed_run(limited)
    source = _source()
    record = await limited.append_acquisition_record(
        lease.id,
        0,
        source,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    def below_floor(path: object) -> SimpleNamespace:
        del path
        return SimpleNamespace(total=100, used=99, free=1)

    monkeypatch.setattr("manicule.storage.acquisition.shutil.disk_usage", below_floor)

    assert await limited.create_acquisition_run(lease.id, "wiki")
    assert (
        await limited.append_acquisition_record(
            lease.id,
            0,
            source,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
    ) == record
    assert await limited.renew_acquisition_lease(
        lease.id,
        "worker",
        lease.lease_generation,
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=2),
    )
    await limited.complete_acquisition_enumeration(
        lease.id,
        _watermark("settled-below-floor"),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await limited.transition_acquisition_record(
        lease.id,
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.UNCHANGED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert await limited.commit_acquisition_watermark(
        lease.id,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    settled = await limited.transition_acquisition_run(
        lease.id,
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert settled.state is AcquisitionRunState.SETTLED


async def test_concurrent_idempotent_writers_recheck_before_the_disk_floor(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = SqliteDocStore(engine, min_disk_headroom_bytes=8)
    second = SqliteDocStore(engine, min_disk_headroom_bytes=8)
    await first.ensure_workspace()
    original_guard = SqliteDocStore._begin_capacity_guard  # pyright: ignore[reportPrivateUsage]

    async def race_once(*operations: Awaitable[object]) -> list[object]:
        barrier = asyncio.Barrier(2)
        disk_checks = 0

        async def gated_guard(session: AsyncSession) -> None:
            await barrier.wait()
            await original_guard(session)

        def falling_headroom(path: object) -> SimpleNamespace:
            nonlocal disk_checks
            del path
            disk_checks += 1
            return SimpleNamespace(
                total=1_000,
                used=500 if disk_checks == 1 else 999,
                free=500 if disk_checks == 1 else 1,
            )

        monkeypatch.setattr(SqliteDocStore, "_begin_capacity_guard", staticmethod(gated_guard))
        monkeypatch.setattr("manicule.storage.acquisition.shutil.disk_usage", falling_headroom)
        outcomes = await asyncio.gather(*operations)
        assert disk_checks == 1, "the idempotent loser performed a growth-only floor check"
        return list(outcomes)

    created = await race_once(
        first.create_acquisition_run("raced-run", "wiki"),
        second.create_acquisition_run("raced-run", "wiki"),
    )
    assert created[0] == created[1]
    lease = await first.claim_acquisition_run(
        "raced-run", "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert lease is not None

    appended = await race_once(
        first.append_acquisition_record(
            lease.id,
            0,
            _source(),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        ),
        second.append_acquisition_record(
            lease.id,
            99,
            _source(),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        ),
    )
    assert appended[0] == appended[1]


async def test_run_identity_and_records_do_not_cross_workspaces(
    engine: AsyncEngine,
) -> None:
    first = SqliteDocStore(engine, workspace_id="one")
    second = SqliteDocStore(engine, workspace_id="two")
    await first.ensure_workspace()
    await second.ensure_workspace()

    run = await _claimed_run(first, "same-run")
    private_title = "private cross-workspace title cinder"
    private_url = "https://example.test/private?token=fake-secret-cinder"
    private_source = _source(uri=private_url).model_copy(update={"title": private_title})
    await first.append_acquisition_record(
        "same-run",
        0,
        private_source,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )

    assert await second.get_acquisition_run("same-run") is None
    with pytest.raises(AcquisitionConflictError, match="requested run identity"):
        await second.create_acquisition_run("same-run", "wiki")
    with pytest.raises(UnknownEntityError) as cross_workspace:
        await second.append_acquisition_record(
            "same-run",
            0,
            private_source,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
    with pytest.raises(AcquisitionConflictError) as wrong_lease:
        await first.append_acquisition_record(
            "same-run",
            0,
            private_source,
            lease_owner="other-worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
    rendered = f"{cross_workspace.value!s}\n{wrong_lease.value!s}"
    assert private_title not in rendered
    assert private_url not in rendered
    assert "fake-secret-cinder" not in rendered


async def test_candidate_watermark_is_not_committed_until_coverage(
    store: SqliteDocStore,
) -> None:
    await store.set_watermark("wiki", _watermark("base"))
    lease = await _claimed_run(store)
    assert lease.base_watermark == _watermark("base")
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    completed = await store.complete_acquisition_enumeration(
        "run",
        _watermark("candidate"),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    assert completed.candidate_watermark == _watermark("candidate")
    assert completed.enumeration_completed_at is not None
    assert completed.watermark_committed_at is None
    assert await store.get_watermark("wiki") == _watermark("base")

    with pytest.raises(AcquisitionCoverageError, match="durable source coverage"):
        await store.commit_acquisition_watermark(
            "run",
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )

    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.UNCHANGED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    committed = await store.commit_acquisition_watermark(
        "run",
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert committed
    persisted = await store.get_acquisition_run("run")
    assert persisted is not None
    assert persisted.watermark_committed_at is not None
    assert await store.get_watermark("wiki") == _watermark("candidate")


async def test_unchanged_coverage_cannot_lose_its_provenance(
    store: SqliteDocStore,
) -> None:
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    unchanged = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.UNCHANGED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    assert unchanged.state is AcquisitionRecordState.UNCHANGED
    persisted = await store.get_acquisition_run("run")
    assert persisted is not None
    assert persisted.unchanged_count == 1
    assert persisted.indexed_count == 0
    with pytest.raises(InvalidAcquisitionTransitionError, match=r"invalid.*transition"):
        await store.transition_acquisition_record(
            "run",
            "page-1",
            AcquisitionRecordState.UNCHANGED,
            AcquisitionRecordState.SETTLED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )


async def test_completion_without_a_candidate_watermark_can_settle(
    store: SqliteDocStore,
) -> None:
    lease = await _claimed_run(store, connector="filesystem")
    completed = await store.complete_acquisition_enumeration(
        "run",
        None,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    assert completed.enumeration_completed_at is not None
    assert completed.candidate_watermark is None
    assert not await store.commit_acquisition_watermark(
        "run",
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    settled = await store.transition_acquisition_run(
        "run",
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert settled.state is AcquisitionRunState.SETTLED
    assert settled.watermark_committed_at is None
    assert await store.get_watermark("filesystem") is None
    assert not await store.renew_acquisition_lease(
        "run",
        "worker",
        lease.lease_generation,
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=2),
    )
    with pytest.raises(AcquisitionConflictError, match="settled"):
        await store.transition_acquisition_record(
            "run",
            "not-present",
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )


async def test_only_enumeration_completion_can_enter_acquiring(store: SqliteDocStore) -> None:
    lease = await _claimed_run(store)
    with pytest.raises(InvalidAcquisitionTransitionError):
        await store.transition_acquisition_run(
            "run",
            AcquisitionRunState.ENUMERATING,
            AcquisitionRunState.ACQUIRING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )


async def test_lifecycle_edges_and_lease_generation_are_compare_and_swap_guarded(
    store: SqliteDocStore,
) -> None:
    now = datetime(2026, 8, 15, 13, tzinfo=UTC)
    await store.create_acquisition_run("run", "wiki")
    first = await store.claim_acquisition_run(
        "run", "first", now=now, expires_at=now + timedelta(seconds=10)
    )
    assert first is not None
    assert first.lease_generation == 1
    assert (
        await store.claim_acquisition_run(
            "run", "first", now=now, expires_at=now + timedelta(seconds=20)
        )
        is None
    )
    unchanged_generation = await store.get_acquisition_run("run")
    assert unchanged_generation is not None
    assert unchanged_generation.lease_generation == first.lease_generation
    await store.complete_acquisition_enumeration(
        "run",
        _watermark("end"),
        lease_owner="first",
        lease_generation=first.lease_generation,
        now=now,
    )
    assert (
        await store.claim_acquisition_run(
            "run", "second", now=now, expires_at=now + timedelta(seconds=20)
        )
        is None
    )
    second = await store.claim_acquisition_run(
        "run",
        "second",
        now=now + timedelta(seconds=11),
        expires_at=now + timedelta(seconds=30),
    )
    assert second is not None
    assert second.lease_generation == 2
    assert not await store.renew_acquisition_lease(
        "run",
        "first",
        first.lease_generation,
        now=now + timedelta(seconds=11),
        expires_at=now + timedelta(seconds=40),
    )

    with pytest.raises(InvalidAcquisitionTransitionError):
        await store.transition_acquisition_run(
            "run",
            AcquisitionRunState.ACQUIRING,
            AcquisitionRunState.ENUMERATING,
            lease_owner="second",
            lease_generation=second.lease_generation,
            now=now + timedelta(seconds=11),
        )
    moved = await store.transition_acquisition_run(
        "run",
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.INDEXING,
        lease_owner="second",
        lease_generation=second.lease_generation,
        now=now + timedelta(seconds=11),
    )
    assert moved.state is AcquisitionRunState.INDEXING
    settled = await store.transition_acquisition_run(
        "run",
        AcquisitionRunState.INDEXING,
        AcquisitionRunState.SETTLED,
        lease_owner="second",
        lease_generation=second.lease_generation,
        now=now + timedelta(seconds=11),
    )
    assert settled.state is AcquisitionRunState.SETTLED


async def test_claim_or_create_serializes_repeated_sync_requests(
    store: SqliteDocStore,
) -> None:
    """Two callers cannot both turn a missing run into independent enumerations."""
    expiry = _NOW + timedelta(minutes=1)
    first, second = await asyncio.gather(
        store.claim_or_create_acquisition_run(
            "wiki", "first", "owner-a", now=_NOW, expires_at=expiry
        ),
        store.claim_or_create_acquisition_run(
            "wiki", "second", "owner-b", now=_NOW, expires_at=expiry
        ),
    )

    winners = [run for run in (first, second) if run is not None]
    assert len(winners) == 1
    assert (await store.latest_unsettled_acquisition_run("wiki")) == winners[0]
    persisted = [
        await store.get_acquisition_run("first"),
        await store.get_acquisition_run("second"),
    ]
    assert sum(run is not None for run in persisted) == 1


async def test_claim_reconciles_every_legacy_overlap_to_one_authoritative_safe_run(
    store: SqliteDocStore,
) -> None:
    older = await _claimed_run(store, "older-safe", "wiki", "older-owner")
    newer = await _claimed_run(store, "newer-safe", "wiki", "newer-owner")

    blocked = await store.claim_or_create_acquisition_run(
        "wiki",
        "unused",
        "third-owner",
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=1),
    )

    assert blocked is None, "the authoritative newer run still has a live owner"
    authoritative = await store.get_acquisition_run(newer.id)
    superseded = await store.get_acquisition_run(older.id)
    assert authoritative is not None
    assert authoritative.superseded_at is None
    assert superseded is not None
    assert superseded.superseded_by == authoritative.id
    assert superseded.lease_generation == older.lease_generation + 1
    assert (
        await store.renew_acquisition_lease(
            older.id,
            "older-owner",
            older.lease_generation,
            now=_NOW,
            expires_at=_NOW + timedelta(minutes=2),
        )
        is False
    )


async def test_an_active_connector_does_not_block_an_independent_connector(
    store: SqliteDocStore,
) -> None:
    expiry = _NOW + timedelta(minutes=1)
    first = await store.claim_or_create_acquisition_run(
        "wiki", "wiki-run", "wiki-owner", now=_NOW, expires_at=expiry
    )
    independent = await store.claim_or_create_acquisition_run(
        "drive", "drive-run", "drive-owner", now=_NOW, expires_at=expiry
    )
    blocked = await store.claim_or_create_acquisition_run(
        "wiki", "duplicate", "other-owner", now=_NOW, expires_at=expiry
    )

    assert first is not None
    assert independent is not None
    assert independent.connector == "drive"
    assert blocked is None


async def test_takeover_normalizes_in_flight_records_to_their_durable_inputs(
    store: SqliteDocStore,
) -> None:
    run = await _claimed_run(store)
    await store.append_acquisition_record(
        run.id,
        0,
        _source("fetching"),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        run.id,
        "fetching",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )

    takeover = await store.claim_or_create_acquisition_run(
        "wiki",
        "must-not-be-created",
        "successor",
        now=_NOW + timedelta(minutes=2),
        expires_at=_NOW + timedelta(minutes=3),
    )

    assert takeover is not None
    assert takeover.id == run.id
    assert takeover.lease_generation == run.lease_generation + 1
    record = (await store.list_acquisition_records(run.id))[0]
    assert record.state is AcquisitionRecordState.RETRY
    assert record.diagnostic is not None
    assert record.diagnostic.code.value == "interrupted"


async def test_recovery_supersedes_a_run_from_an_obsolete_connector_position(
    store: SqliteDocStore,
) -> None:
    obsolete = await _claimed_run(store, "obsolete", "wiki", "old-owner")
    winner = await _claimed_run(store, "winner", "wiki", "winning-owner")
    await store.complete_acquisition_enumeration(
        winner.id,
        _watermark("new-position"),
        lease_owner="winning-owner",
        lease_generation=winner.lease_generation,
        now=_NOW,
    )
    await store.commit_acquisition_watermark(
        winner.id,
        lease_owner="winning-owner",
        lease_generation=winner.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_run(
        winner.id,
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.SETTLED,
        lease_owner="winning-owner",
        lease_generation=winner.lease_generation,
        now=_NOW,
    )

    recovered = await store.claim_or_create_acquisition_run(
        "wiki",
        "successor",
        "new-owner",
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=1),
    )

    assert recovered is not None
    assert recovered.id == "successor"
    assert recovered.base_watermark == _watermark("new-position")
    superseded = await store.get_acquisition_run(obsolete.id)
    assert superseded is not None
    assert superseded.superseded_at is not None
    assert superseded.superseded_by == "successor"
    assert superseded.lease_generation == obsolete.lease_generation + 1
    with pytest.raises(AcquisitionConflictError, match="lease changed or expired"):
        await store.append_acquisition_record(
            obsolete.id,
            0,
            _source("stale-mutation"),
            lease_owner="old-owner",
            lease_generation=obsolete.lease_generation,
            now=_NOW,
        )
    assert (
        await store.claim_acquisition_run(
            obsolete.id,
            "reviver",
            now=_NOW + timedelta(minutes=2),
            expires_at=_NOW + timedelta(minutes=3),
        )
        is None
    )


async def test_cleanup_is_bounded_and_discards_only_superseded_retry_work(
    store: SqliteDocStore,
) -> None:
    settled = await _claimed_run(store, "settled", "finished")
    await store.complete_acquisition_enumeration(
        settled.id,
        None,
        lease_owner="worker",
        lease_generation=settled.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_run(
        settled.id,
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.SETTLED,
        lease_owner="worker",
        lease_generation=settled.lease_generation,
        now=_NOW,
    )
    retry = await _claimed_run(store, "retry", "unfinished", "retry-owner")
    await store.append_acquisition_record(
        retry.id,
        0,
        _source("durable-retry"),
        lease_owner="retry-owner",
        lease_generation=retry.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        retry.id,
        "durable-retry",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="retry-owner",
        lease_generation=retry.lease_generation,
        now=_NOW,
    )
    winner = await _claimed_run(store, "winner", "unfinished", "winner-owner")
    await store.complete_acquisition_enumeration(
        winner.id,
        _watermark("winner"),
        lease_owner="winner-owner",
        lease_generation=winner.lease_generation,
        now=_NOW,
    )
    await store.commit_acquisition_watermark(
        winner.id,
        lease_owner="winner-owner",
        lease_generation=winner.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_run(
        winner.id,
        AcquisitionRunState.ACQUIRING,
        AcquisitionRunState.SETTLED,
        lease_owner="winner-owner",
        lease_generation=winner.lease_generation,
        now=_NOW,
    )
    replacement = await store.claim_or_create_acquisition_run(
        "unfinished",
        "replacement",
        "replacement-owner",
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=1),
    )
    assert replacement is not None
    await store.append_acquisition_record(
        replacement.id,
        0,
        _source("authoritative-retry"),
        lease_owner="replacement-owner",
        lease_generation=replacement.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        replacement.id,
        "authoritative-retry",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="replacement-owner",
        lease_generation=replacement.lease_generation,
        now=_NOW,
    )
    retry = await store.get_acquisition_run(retry.id)
    assert retry is not None
    assert retry.superseded_at is not None

    removed = await store.cleanup_acquisition_history(datetime(2100, 1, 1, tzinfo=UTC), limit=1)

    assert removed == 1
    assert await store.get_acquisition_run(settled.id) is None
    assert await store.get_acquisition_run(retry.id) is not None
    records = await store.list_acquisition_records(retry.id)
    assert len(records) == 1
    assert records[0].state is AcquisitionRecordState.RETRY

    removed = await store.cleanup_acquisition_history(datetime(2100, 1, 1, tzinfo=UTC), limit=10)

    assert removed == 2
    assert await store.get_acquisition_run(retry.id) is None
    authoritative = await store.get_acquisition_run(replacement.id)
    assert authoritative is not None
    records = await store.list_acquisition_records(replacement.id)
    assert [record.state for record in records] == [AcquisitionRecordState.RETRY]


async def test_stale_generation_cannot_publish_connector_run_metadata(
    store: SqliteDocStore,
) -> None:
    stale = await _claimed_run(store, "diagnostic-run", "wiki", "stale-owner")
    successor = await store.claim_acquisition_run(
        stale.id,
        "successor-owner",
        now=_NOW + timedelta(minutes=2),
        expires_at=_NOW + timedelta(minutes=3),
    )
    assert successor is not None
    assert await store.record_acquisition_run_metadata(
        successor.id,
        "successor-owner",
        successor.lease_generation,
        now=_NOW + timedelta(minutes=2),
        updates={"last_run": {"outcome": "successor"}},
        release=False,
    )

    assert not await store.record_acquisition_run_metadata(
        stale.id,
        "stale-owner",
        stale.lease_generation,
        now=_NOW,
        updates={"last_run": {"outcome": "stale"}},
        release=True,
    )
    assert (await store.connector_metadata("wiki"))["last_run"] == {"outcome": "successor"}


async def test_orderly_release_is_generation_fenced_and_immediately_claimable(
    store: SqliteDocStore,
) -> None:
    run = await _claimed_run(store)

    assert not await store.release_acquisition_lease(
        run.id,
        "stale-worker",
        run.lease_generation,
        now=_NOW,
    )
    assert await store.release_acquisition_lease(
        run.id,
        "worker",
        run.lease_generation,
        now=_NOW,
    )
    released = await store.get_acquisition_run(run.id)
    assert released is not None
    assert released.lease_owner is None
    assert released.lease_expires_at is None

    replacement = await store.claim_acquisition_run(
        run.id,
        "replacement",
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=1),
    )
    assert replacement is not None
    assert replacement.lease_generation == run.lease_generation + 1


async def test_expired_owner_cannot_append_or_complete(store: SqliteDocStore) -> None:
    lease = await _claimed_run(store)
    expired = _NOW + timedelta(minutes=1)

    with pytest.raises(AcquisitionConflictError, match="expired"):
        await store.append_acquisition_record(
            "run",
            0,
            _source(),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=expired,
        )
    with pytest.raises(AcquisitionConflictError, match="expired"):
        await store.complete_acquisition_enumeration(
            "run",
            _watermark("end"),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=expired,
        )
    assert await store.list_acquisition_records("run") == []


def test_invalid_source_validation_does_not_echo_source_shaped_input() -> None:
    sensitive_url = "https://example.test/private?token=do-not-log"
    with pytest.raises(ValidationError) as raised:
        AcquisitionSource.model_validate(
            {
                "ref": {
                    "source_id": "page",
                    "uri": "",
                    "metadata": {"sensitive": sensitive_url},
                },
                "metadata": {"title": sensitive_url},
            }
        )

    assert sensitive_url not in str(raised.value)


async def test_retry_blocks_settlement_and_attempts_increment_without_losing_version(
    store: SqliteDocStore,
) -> None:
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        "run",
        None,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    first = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert first.attempts == 1
    retry = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        fetched_version_token="source-v1",  # noqa: S106 - revision, not a credential
    )
    second = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert retry.fetched_version_token == "source-v1"  # noqa: S105
    assert second.fetched_version_token == "source-v1"  # noqa: S105
    assert second.attempts == 2
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    cleared = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        fetched_version_token=None,
    )
    assert cleared.fetched_version_token is None
    assert cleared.attempts == 3
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    with pytest.raises(AcquisitionConflictError, match="active records"):
        await store.transition_acquisition_run(
            "run",
            AcquisitionRunState.ACQUIRING,
            AcquisitionRunState.SETTLED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
    latest = await store.latest_unsettled_acquisition_run("wiki")
    assert latest is not None
    assert latest.id == "run"
    assert latest.retry_count == 1


async def test_watermark_commit_is_ordered_cas_and_replay_safe(store: SqliteDocStore) -> None:
    older = await _claimed_run(store, "older")
    newer = await _claimed_run(store, "newer", owner="new-worker")
    assert older.base_watermark is None
    assert newer.base_watermark is None
    await store.complete_acquisition_enumeration(
        "newer",
        _watermark("new"),
        lease_owner="new-worker",
        lease_generation=newer.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        "older",
        _watermark("old"),
        lease_owner="worker",
        lease_generation=older.lease_generation,
        now=_NOW,
    )

    assert await store.commit_acquisition_watermark(
        "newer",
        lease_owner="new-worker",
        lease_generation=newer.lease_generation,
        now=_NOW,
    )
    first_run = await store.get_acquisition_run("newer")
    first_metadata = await store.connector_metadata("wiki")
    assert first_run is not None
    with pytest.raises(AcquisitionWatermarkConflictError, match="changed"):
        await store.commit_acquisition_watermark(
            "older",
            lease_owner="worker",
            lease_generation=older.lease_generation,
            now=_NOW,
        )
    assert await store.get_watermark("wiki") == _watermark("new")

    assert await store.commit_acquisition_watermark(
        "newer",
        lease_owner="new-worker",
        lease_generation=newer.lease_generation,
        now=_NOW + timedelta(seconds=30),
    )
    replayed = await store.get_acquisition_run("newer")
    assert replayed is not None
    assert replayed.watermark_committed_at == first_run.watermark_committed_at
    assert await store.connector_metadata("wiki") == first_metadata


async def test_snapshot_promotion_rolls_back_its_marker_when_watermark_cas_loses(
    store: SqliteDocStore,
) -> None:
    older = await _claimed_run(store, "older-snapshot")
    newer = await _claimed_run(store, "newer-snapshot", owner="new-worker")
    for run, owner, candidate in (
        (older, "worker", "old"),
        (newer, "new-worker", "new"),
    ):
        await store.complete_acquisition_enumeration(
            run.id,
            _watermark(candidate),
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.complete_snapshot_acquisition(
            run.id,
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )

    await store.promote_snapshot_and_commit_watermark(
        newer.id,
        expected_scope_fingerprint="",
        lease_owner="new-worker",
        lease_generation=newer.lease_generation,
        now=_NOW,
    )
    with pytest.raises(AcquisitionWatermarkConflictError, match="changed"):
        await store.promote_snapshot_and_commit_watermark(
            older.id,
            expected_scope_fingerprint="",
            lease_owner="worker",
            lease_generation=older.lease_generation,
            now=_NOW,
        )

    unpromoted = await store.get_acquisition_run(older.id)
    assert unpromoted is not None
    assert unpromoted.acquisition_completed_at is not None
    assert unpromoted.promoted_at is None
    assert unpromoted.watermark_committed_at is None
    assert await store.get_watermark("wiki") == _watermark("new")


async def test_failed_append_transaction_acknowledges_nothing(store: SqliteDocStore) -> None:
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source("one"),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    with pytest.raises(IntegrityError):
        await store.append_acquisition_record(
            "run",
            0,
            _source("two"),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )

    records = await store.list_acquisition_records("run")
    assert [record.source.source_id for record in records] == ["one"]
    run = await store.get_acquisition_run("run")
    assert run is not None
    assert run.discovered_count == 1


async def test_acquired_blob_is_not_garbage_and_does_not_publish_a_document(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(engine, data_dir)
    stored = await blobs.put(b"new unpublished revision", "text/plain")
    assert isinstance(stored, StoredBlob)
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        "run",
        _watermark("end"),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    acquired = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        blob_ref=stored.hash,
        acquired_source=_acquired(b"new unpublished revision"),
        fetched_version_token="v1",  # noqa: S106 - source revision, not a credential
    )

    assert acquired.blob_ref == stored.hash
    assert await store.find_document("wiki", "page-1") is None
    assert await blobs.collect_garbage() == []
    assert await blobs.get(stored.hash) == b"new unpublished revision"

    retry = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert retry.blob_ref == stored.hash
    indexing = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.INDEXING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    assert indexing.blob_ref == stored.hash
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    persisted = await store.get_acquisition_run("run")
    assert persisted is not None
    assert persisted.indexed_count == 1


async def test_acquired_state_requires_a_retained_blob_reference(
    store: SqliteDocStore,
) -> None:
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    with pytest.raises(
        InvalidAcquisitionTransitionError,
        match="requires a retained blob reference",
    ):
        await store.transition_acquisition_record(
            "run",
            "page-1",
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )

    records = await store.list_acquisition_records("run")
    assert len(records) == 1
    assert records[0].state is AcquisitionRecordState.ACQUIRING
    assert records[0].blob_ref is None


async def test_retry_requires_an_existing_or_new_blob_before_indexing(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(engine, data_dir)
    stored = await blobs.put(b"retried revision", "text/plain")
    assert isinstance(stored, StoredBlob)
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    with pytest.raises(
        InvalidAcquisitionTransitionError,
        match="indexing record requires a retained blob reference",
    ):
        await store.transition_acquisition_record(
            "run",
            "page-1",
            AcquisitionRecordState.RETRY,
            AcquisitionRecordState.INDEXING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )

    retry = (await store.list_acquisition_records("run"))[0]
    assert retry.state is AcquisitionRecordState.RETRY
    assert retry.blob_ref is None
    indexing = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.INDEXING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        blob_ref=stored.hash,
    )
    assert indexing.state is AcquisitionRecordState.INDEXING
    assert indexing.blob_ref == stored.hash


async def test_settled_retry_does_not_require_a_blob(store: SqliteDocStore) -> None:
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    settled = await store.transition_acquisition_record(
        "run",
        "page-1",
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    assert settled.state is AcquisitionRecordState.SETTLED
    assert settled.blob_ref is None
    persisted = await store.get_acquisition_run("run")
    assert persisted is not None
    assert persisted.indexed_count == 0


@pytest.mark.parametrize(
    "state",
    [AcquisitionRecordState.ACQUIRED, AcquisitionRecordState.INDEXING],
)
async def test_database_requires_blobs_for_blob_backed_states(
    store: SqliteDocStore,
    engine: AsyncEngine,
    state: AcquisitionRecordState,
) -> None:
    lease = await _claimed_run(store)
    await store.append_acquisition_record(
        "run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    with pytest.raises(IntegrityError, match="blob_backed_acquisition_states_have_a_blob"):
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE acquisition_records SET state = :state "
                    "WHERE run_id = 'run' AND source_id = 'page-1'"
                ),
                {"state": state.value},
            )

    record = (await store.list_acquisition_records("run"))[0]
    assert record.state is AcquisitionRecordState.DISCOVERED
    assert record.blob_ref is None


async def test_snapshot_markers_are_distinct_and_promotion_commits_watermark_atomically(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run = await store.create_acquisition_run(
        "promote-run",
        "wiki",
        source_scope="space:ENG",
        scope_fingerprint="scope-eng-v1",
    )
    claimed = await store.claim_acquisition_run(
        run.id, "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert claimed is not None
    run = claimed
    await store.append_acquisition_record(
        run.id,
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    enumerated = await store.complete_acquisition_enumeration(
        run.id,
        _watermark("candidate"),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert enumerated.enumeration_completed_at is not None
    assert enumerated.acquisition_completed_at is None
    assert enumerated.promoted_at is None
    retained = await _retain_record(store, engine, data_dir, run)

    acquired = await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert acquired.acquisition_completed_at is not None
    assert acquired.membership_hash
    assert acquired.promoted_at is None
    assert await store.get_watermark("wiki") is None
    resumed_acquisition = await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert resumed_acquisition.acquisition_completed_at == acquired.acquisition_completed_at
    assert resumed_acquisition.membership_hash == acquired.membership_hash

    promoted = await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="scope-eng-v1",
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert promoted.promoted_at == promoted.watermark_committed_at
    assert promoted.completeness is SnapshotCompleteness.COMPLETE
    resumed_promotion = await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="scope-eng-v1",
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert resumed_promotion.promoted_at == promoted.promoted_at
    assert resumed_promotion.watermark_committed_at == promoted.watermark_committed_at
    assert (await store.list_acquisition_records(run.id))[0].snapshot_outcome is (
        SnapshotItemOutcome.RETAINED
    )
    assert await store.get_watermark("wiki") == _watermark("candidate")
    assert await BlobStore(engine, data_dir).collect_garbage() == []
    assert await BlobStore(engine, data_dir).get(retained) is not None


async def test_strict_snapshot_omission_is_typed_resumable_and_unpromoted(
    store: SqliteDocStore,
) -> None:
    run = await _claimed_run(store, "strict-run")
    await store.append_acquisition_record(
        run.id,
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        run.id,
        _watermark("candidate"),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        run.id,
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
        diagnostic=AcquisitionDiagnostic(
            stage=AcquisitionStage.ACQUISITION,
            code=AcquisitionFailureCode.AUTHENTICATION,
        ),
    )

    with pytest.raises(AcquisitionCoverageError, match="1 required source records"):
        await store.complete_snapshot_acquisition(
            run.id,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
    persisted = await store.get_acquisition_run(run.id)
    assert persisted is not None
    assert persisted.acquisition_completed_at is None
    assert persisted.promoted_at is None
    assert persisted.watermark_committed_at is None
    assert persisted.omission_count == 1
    assert persisted.omission_reasons == {AcquisitionFailureCode.AUTHENTICATION: 1}
    record = (await store.list_acquisition_records(run.id))[0]
    assert record.diagnostic is not None
    assert record.diagnostic.code is AcquisitionFailureCode.AUTHENTICATION
    assert record.snapshot_diagnostic == record.diagnostic
    assert await store.get_watermark("wiki") is None


async def test_manifest_verifier_rejects_revision_and_envelope_substitution(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    created = await store.create_acquisition_run(
        "verified-run", "wiki", source_scope="space:ENG", scope_fingerprint="eng"
    )
    run = await store.claim_acquisition_run(
        created.id, "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert run is not None
    await store.append_acquisition_record(
        run.id,
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        run.id,
        None,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await _retain_record(store, engine, data_dir, run)
    await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="eng",
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert await store.verify_snapshot_manifest(run.id)

    secret = "private-corrupt-diagnostic-do-not-print"  # noqa: S105 - redaction fixture
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET snapshot_diagnostic = :diagnostic "
                "WHERE run_id = 'verified-run'"
            ),
            {"diagnostic": json.dumps(secret)},
        )
    corrupted = (await store.list_acquisition_records(run.id))[0]
    assert corrupted.snapshot_diagnostic is not None
    assert corrupted.snapshot_diagnostic.code is AcquisitionFailureCode.LEGACY_UNVERIFIED
    assert secret not in str(corrupted)
    assert not await store.verify_snapshot_manifest(run.id)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET snapshot_diagnostic = NULL "
                "WHERE run_id = 'verified-run'"
            )
        )
    assert await store.verify_snapshot_manifest(run.id)

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET fetched_version_token = 'substituted' "
                "WHERE run_id = 'verified-run'"
            )
        )
    assert not await store.verify_snapshot_manifest(run.id)
    assert await store.latest_promoted_snapshot("wiki", "eng") is None
    assert await store.reusable_snapshot_record("wiki", "eng", "page-1", "substituted") is None
    with pytest.raises(AcquisitionConflictError, match="evidence"):
        await store.promote_snapshot_and_commit_watermark(
            run.id,
            expected_scope_fingerprint="eng",
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET fetched_version_token = 'v1', "
                "acquired_source = json_set(acquired_source, '$.byte_length', 999) "
                "WHERE run_id = 'verified-run'"
            )
        )
    assert not await store.verify_snapshot_manifest(run.id)


async def test_allow_omissions_promotes_partial_snapshot_with_aggregate_reasons(
    store: SqliteDocStore,
) -> None:
    run = await store.create_acquisition_run(
        "partial-run",
        "wiki",
        source_scope="space:ENG",
        scope_fingerprint="scope-eng-v1",
        promotion_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    )
    claimed = await store.claim_acquisition_run(
        run.id, "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert claimed is not None
    run = claimed
    for sequence, source_id in enumerate(("missing-1", "missing-2")):
        await store.append_acquisition_record(
            run.id,
            sequence,
            _source(source_id),
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_record(
            run.id,
            source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.RETRY,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
            diagnostic=AcquisitionDiagnostic(
                stage=AcquisitionStage.ACQUISITION,
                code=AcquisitionFailureCode.AUTHENTICATION,
            ),
        )
    await store.complete_acquisition_enumeration(
        run.id,
        _watermark("partial"),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    acquired = await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert acquired.omission_count == 2
    assert acquired.omission_reasons == {AcquisitionFailureCode.AUTHENTICATION: 2}

    promoted = await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="scope-eng-v1",
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert promoted.completeness is SnapshotCompleteness.PARTIAL
    assert {record.state for record in await store.list_acquisition_records(run.id)} == {
        AcquisitionRecordState.OMITTED
    }
    assert {record.snapshot_outcome for record in await store.list_acquisition_records(run.id)} == {
        SnapshotItemOutcome.OMITTED
    }
    assert {
        record.snapshot_diagnostic.code
        for record in await store.list_acquisition_records(run.id)
        if record.snapshot_diagnostic is not None
    } == {AcquisitionFailureCode.AUTHENTICATION}
    assert await store.get_watermark("wiki") == _watermark("partial")


@pytest.mark.parametrize(
    "raw_diagnostic",
    [
        '"private-current-scalar-do-not-print"',
        '["private-current-list-do-not-print"]',
        "null",
        None,
        '{"stage":"acquisition","code":"unknown","retryable":true,"private":"secret"}',
        '{"source_id":"private-source-shaped","uri":"https://private.invalid"}',
        '{"stage":"acquisition","code":"not-real","retryable":true}',
        '{"stage":"indexing","code":"unknown","retryable":true}',
    ],
    ids=(
        "scalar",
        "list",
        "json-null",
        "sql-null",
        "extra-private",
        "source-shaped",
        "unknown-code",
        "wrong-stage",
    ),
)
async def test_completion_canonicalizes_current_omission_diagnostics_before_promotion(
    store: SqliteDocStore,
    engine: AsyncEngine,
    raw_diagnostic: str | None,
) -> None:
    created = await store.create_acquisition_run(
        "canonical-current-run",
        "wiki",
        source_scope="space:canonical",
        scope_fingerprint="canonical",
        promotion_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    )
    run = await store.claim_acquisition_run(
        created.id, "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert run is not None
    await store.append_acquisition_record(
        run.id,
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        run.id,
        None,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        run.id,
        "page-1",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
        diagnostic=AcquisitionDiagnostic(
            stage=AcquisitionStage.ACQUISITION,
            code=AcquisitionFailureCode.UNKNOWN,
        ),
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET snapshot_diagnostic = :diagnostic "
                "WHERE run_id = 'canonical-current-run'"
            ),
            {"diagnostic": raw_diagnostic},
        )

    completed = await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert completed.omission_reasons == {AcquisitionFailureCode.LEGACY_UNVERIFIED: 1}
    record = (await store.list_acquisition_records(run.id))[0]
    assert record.snapshot_diagnostic is not None
    assert record.snapshot_diagnostic.code is AcquisitionFailureCode.LEGACY_UNVERIFIED
    canonical = record.snapshot_diagnostic.model_dump(mode="json")
    async with engine.connect() as connection:
        stored_raw = (
            await connection.execute(
                text(
                    "SELECT snapshot_diagnostic FROM acquisition_records "
                    "WHERE run_id = 'canonical-current-run'"
                )
            )
        ).scalar_one()
    assert json.loads(stored_raw) == canonical
    assert "private" not in stored_raw

    await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="canonical",
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert await store.verify_snapshot_manifest(run.id)
    assert await store.latest_promoted_snapshot("wiki", "canonical") is not None

    substituted = '{"source_id":"private-post-freeze","uri":"https://private.invalid"}'
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_records SET snapshot_diagnostic = :diagnostic "
                "WHERE run_id = 'canonical-current-run'"
            ),
            {"diagnostic": substituted},
        )
    assert not await store.verify_snapshot_manifest(run.id)
    assert await store.latest_promoted_snapshot("wiki", "canonical") is None


async def test_promotion_and_reuse_are_isolated_by_workspace_connector_and_scope(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run = await store.create_acquisition_run(
        "scope-run", "wiki", source_scope="space:ENG", scope_fingerprint="eng"
    )
    claimed = await store.claim_acquisition_run(
        run.id, "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert claimed is not None
    run = claimed
    await store.append_acquisition_record(
        run.id,
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await store.complete_acquisition_enumeration(
        run.id,
        None,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    await _retain_record(store, engine, data_dir, run)
    await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    with pytest.raises(AcquisitionConflictError, match="scope fingerprint"):
        await store.promote_snapshot_and_commit_watermark(
            run.id,
            expected_scope_fingerprint="other",
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
    await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="eng",
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )
    assert await store.reusable_snapshot_record("wiki", "eng", "page-1", "v1") is not None
    assert await store.reusable_snapshot_record("wiki", "other", "page-1", "v1") is None
    assert await store.reusable_snapshot_record("other", "eng", "page-1", "v1") is None

    other = SqliteDocStore(engine, workspace_id="other-workspace")
    await other.ensure_workspace()
    assert await other.latest_promoted_snapshot("wiki", "eng") is None
    assert await other.reusable_snapshot_record("wiki", "eng", "page-1", "v1") is None


async def test_reuse_uses_the_same_run_id_tiebreaker_as_latest_promoted_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    for run_id, owner in (("same-time-a", "worker-a"), ("same-time-z", "worker-z")):
        created = await store.create_acquisition_run(
            run_id,
            "wiki",
            source_scope="space:ENG",
            scope_fingerprint="same-scope",
        )
        run = await store.claim_acquisition_run(
            created.id,
            owner,
            now=_NOW,
            expires_at=_NOW + timedelta(minutes=1),
        )
        assert run is not None
        await store.append_acquisition_record(
            run.id,
            0,
            _source(),
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.complete_acquisition_enumeration(
            run.id,
            None,
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await _retain_record(store, engine, data_dir, run, owner=owner)
        await store.complete_snapshot_acquisition(
            run.id,
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.promote_snapshot_and_commit_watermark(
            run.id,
            expected_scope_fingerprint="same-scope",
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )

    latest = await store.latest_promoted_snapshot("wiki", "same-scope")
    reusable = await store.reusable_snapshot_record("wiki", "same-scope", "page-1", "v1")

    assert latest is not None
    assert reusable is not None
    assert latest.id == "same-time-z"
    assert reusable.run_id == latest.id


async def test_reuse_never_falls_back_behind_the_authoritative_latest_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    for run_id, owner, version_token in (
        ("a-older-v1", "worker-old", "v1"),
        ("z-newer-v2", "worker-new", "v2"),
    ):
        created = await store.create_acquisition_run(
            run_id,
            "wiki",
            source_scope="space:ENG",
            scope_fingerprint="versioned-scope",
        )
        run = await store.claim_acquisition_run(
            created.id,
            owner,
            now=_NOW,
            expires_at=_NOW + timedelta(minutes=1),
        )
        assert run is not None
        await store.append_acquisition_record(
            run.id,
            0,
            _source(version_token=version_token),
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.complete_acquisition_enumeration(
            run.id,
            None,
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await _retain_record(
            store,
            engine,
            data_dir,
            run,
            owner=owner,
            version_token=version_token,
        )
        await store.complete_snapshot_acquisition(
            run.id,
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.promote_snapshot_and_commit_watermark(
            run.id,
            expected_scope_fingerprint="versioned-scope",
            lease_owner=owner,
            lease_generation=run.lease_generation,
            now=_NOW,
        )

    latest = await store.latest_promoted_snapshot("wiki", "versioned-scope")

    assert latest is not None
    assert latest.id == "z-newer-v2"
    assert await store.reusable_snapshot_record("wiki", "versioned-scope", "page-1", "v1") is None
    reusable = await store.reusable_snapshot_record("wiki", "versioned-scope", "page-1", "v2")
    assert reusable is not None
    assert reusable.run_id == latest.id


async def test_concurrent_blob_reservations_preserve_backlog_and_publications(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    limited = SqliteDocStore(engine, max_acquired_blob_backlog_bytes=10)
    await limited.ensure_workspace()
    published = make_document(source="wiki", source_id="published-before-refusal")
    await limited.upsert_document(published)
    before = await limited.find_document("wiki", "published-before-refusal")
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    stored = [
        await blobs.put(payload, "application/octet-stream") for payload in (b"a" * 10, b"b" * 10)
    ]
    assert all(isinstance(blob, StoredBlob) for blob in stored)

    lease = await _claimed_run(limited)
    for sequence in range(2):
        await limited.append_acquisition_record(
            lease.id,
            sequence,
            _source(f"page-{sequence}"),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
    await limited.complete_acquisition_enumeration(
        lease.id,
        _watermark("uncommitted-after-refusal"),
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    for sequence in range(2):
        await limited.transition_acquisition_record(
            lease.id,
            f"page-{sequence}",
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )

    outcomes = await asyncio.gather(
        *(
            limited.transition_acquisition_record(
                lease.id,
                f"page-{sequence}",
                AcquisitionRecordState.ACQUIRING,
                AcquisitionRecordState.ACQUIRED,
                lease_owner="worker",
                lease_generation=lease.lease_generation,
                now=_NOW,
                blob_ref=blob.hash,
                acquired_source=_acquired((b"a" * 10, b"b" * 10)[sequence], f"page-{sequence}"),
                fetched_version_token="v1",  # noqa: S106 - synthetic revision
            )
            for sequence, blob in enumerate(stored)
            if isinstance(blob, StoredBlob)
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
    refused = [outcome for outcome in outcomes if isinstance(outcome, CapacityRefusedError)]
    assert len(refused) == 1
    assert refused[0].diagnostic.resource is CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES
    recovered = SqliteDocStore(engine, max_acquired_blob_backlog_bytes=10)
    run = await recovered.get_acquisition_run(lease.id)
    assert run is not None
    assert run.acquired_blob_bytes == 10
    assert run.watermark_committed_at is None
    records = await recovered.list_acquisition_records(lease.id)
    assert sum(record.blob_ref is not None for record in records) == 1
    assert await recovered.find_document("wiki", "published-before-refusal") == before

    winner = next(
        index for index, outcome in enumerate(outcomes) if not isinstance(outcome, BaseException)
    )
    loser = 1 - winner
    await recovered.transition_acquisition_record(
        lease.id,
        f"page-{winner}",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.INDEXING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await recovered.transition_acquisition_record(
        lease.id,
        f"page-{winner}",
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    loser_blob = stored[loser]
    assert isinstance(loser_blob, StoredBlob)
    await recovered.transition_acquisition_record(
        lease.id,
        f"page-{loser}",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        blob_ref=loser_blob.hash,
        acquired_source=_acquired((b"a" * 10, b"b" * 10)[loser], f"page-{loser}"),
        fetched_version_token="v1",  # noqa: S106 - synthetic revision
    )
    resumed = await recovered.get_acquisition_run(lease.id)
    assert resumed is not None
    assert resumed.acquired_blob_bytes == 10, "settled bytes no longer consume backlog capacity"


async def test_shared_blob_is_charged_once_and_remains_pinned_until_every_record_settles(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    limited = SqliteDocStore(engine, max_acquired_blob_backlog_bytes=10)
    await limited.ensure_workspace()
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    shared = await blobs.put(b"same bytes", "application/octet-stream")
    replacement = await blobs.put(b"new-bytes!", "application/octet-stream")
    assert isinstance(shared, StoredBlob)
    assert isinstance(replacement, StoredBlob)
    lease = await _claimed_run(limited)

    for sequence in range(3):
        source_id = f"page-{sequence}"
        await limited.append_acquisition_record(
            lease.id,
            sequence,
            _source(source_id),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
        await limited.transition_acquisition_record(
            lease.id,
            source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )

    for source_id in ("page-0", "page-1"):
        await limited.transition_acquisition_record(
            lease.id,
            source_id,
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
            blob_ref=shared.hash,
            acquired_source=_acquired(b"same bytes", source_id),
        )

    run = await limited.get_acquisition_run(lease.id)
    assert run is not None
    assert run.acquired_blob_bytes == shared.stored_bytes

    await limited.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.INDEXING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await limited.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    still_pinned = await limited.get_acquisition_run(lease.id)
    assert still_pinned is not None
    assert still_pinned.acquired_blob_bytes == shared.stored_bytes
    with pytest.raises(CapacityRefusedError):
        await limited.transition_acquisition_record(
            lease.id,
            "page-2",
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
            blob_ref=replacement.hash,
            acquired_source=_acquired(b"new-bytes!", "page-2"),
        )

    await limited.transition_acquisition_record(
        lease.id,
        "page-1",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.INDEXING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await limited.transition_acquisition_record(
        lease.id,
        "page-1",
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    released = await limited.get_acquisition_run(lease.id)
    assert released is not None
    assert released.acquired_blob_bytes == 0
    await limited.transition_acquisition_record(
        lease.id,
        "page-2",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        blob_ref=replacement.hash,
        acquired_source=_acquired(b"new-bytes!", "page-2"),
    )


async def test_lowered_backlog_limit_never_blocks_capacity_releasing_transitions(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    initial = SqliteDocStore(engine, max_acquired_blob_backlog_bytes=100)
    await initial.ensure_workspace()
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    stored = [
        await blobs.put(bytes([value]) * 10, "application/octet-stream") for value in range(3)
    ]
    assert all(isinstance(blob, StoredBlob) for blob in stored)
    lease = await _claimed_run(initial)
    for sequence, blob in enumerate(stored):
        assert isinstance(blob, StoredBlob)
        source_id = f"page-{sequence}"
        await initial.append_acquisition_record(
            lease.id,
            sequence,
            _source(source_id),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
        await initial.transition_acquisition_record(
            lease.id,
            source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
        await initial.transition_acquisition_record(
            lease.id,
            source_id,
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
            blob_ref=blob.hash,
            acquired_source=_acquired(bytes([sequence]) * 10, source_id),
            fetched_version_token="v1",  # noqa: S106 - synthetic revision
        )

    reduced = SqliteDocStore(engine, max_acquired_blob_backlog_bytes=10)
    await reduced.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.INDEXING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await reduced.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.SETTLED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    run = await reduced.get_acquisition_run(lease.id)
    assert run is not None
    assert run.acquired_blob_bytes == 20


async def test_retry_and_reacquire_keep_retained_bytes_in_backlog_accounting(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    limited = SqliteDocStore(engine, max_acquired_blob_backlog_bytes=10)
    await limited.ensure_workspace()
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    first = await blobs.put(b"a" * 10, "application/octet-stream")
    second = await blobs.put(b"b" * 10, "application/octet-stream")
    assert isinstance(first, StoredBlob)
    assert isinstance(second, StoredBlob)
    lease = await _claimed_run(limited)
    for sequence in range(2):
        await limited.append_acquisition_record(
            lease.id,
            sequence,
            _source(f"page-{sequence}"),
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
    await limited.complete_acquisition_enumeration(
        lease.id,
        None,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    for sequence in range(2):
        await limited.transition_acquisition_record(
            lease.id,
            f"page-{sequence}",
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
        )
    await limited.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
        blob_ref=first.hash,
        acquired_source=_acquired(b"a" * 10, "page-0"),
    )
    await limited.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.RETRY,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await limited.transition_acquisition_record(
        lease.id,
        "page-0",
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )

    retrying = await limited.get_acquisition_run(lease.id)
    assert retrying is not None
    assert retrying.acquired_blob_bytes == 10
    with pytest.raises(CapacityRefusedError):
        await limited.transition_acquisition_record(
            lease.id,
            "page-1",
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="worker",
            lease_generation=lease.lease_generation,
            now=_NOW,
            blob_ref=second.hash,
            acquired_source=_acquired(b"b" * 10, "page-1"),
        )


async def test_migration_preserves_publications_and_creates_no_backlog(data_dir: Path) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="6e31b7d592ac")
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        published = make_document(source="wiki", source_id="already-indexed")
        await store.upsert_document(published)
        before = await store.find_document("wiki", "already-indexed")

        await upgrade(engine)

        assert before is not None
        assert await store.find_document("wiki", "already-indexed") == before
        assert await store.latest_unsettled_acquisition_run("wiki") is None
    finally:
        await engine.dispose()


async def test_migration_does_not_forge_legacy_unchanged_coverage_into_a_complete_snapshot(
    data_dir: Path,
) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="c41d7ea923b8")
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        timestamp = "2026-08-15T13:00:00+00:00"
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO connectors "
                    "(id, workspace_id, name, type, config, watermark, sync_interval_seconds, "
                    "status, metadata, created_at) VALUES "
                    "('default:wiki', 'default', 'wiki', 'wiki', '{}', NULL, 300, "
                    "'active', '{}', :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            await connection.execute(
                text(
                    "INSERT INTO acquisition_runs "
                    "(id, workspace_id, connector_id, connector_name, state, base_watermark, "
                    "candidate_watermark, enumeration_completed_at, watermark_committed_at, "
                    "lease_generation, discovered_count, acquired_count, indexed_count, "
                    "unchanged_count, retry_count, metadata_bytes, acquired_blob_bytes, "
                    "created_at, updated_at) VALUES "
                    "('legacy-run', 'default', 'default:wiki', 'wiki', 'settled', NULL, NULL, "
                    ":timestamp, :timestamp, 0, 1, 0, 0, 1, 0, 1, 0, :timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            await connection.execute(
                text(
                    "INSERT INTO acquisition_records "
                    "(id, run_id, workspace_id, connector_id, sequence, source_id, "
                    "source_record, state, blob_ref, acquired_source, attempts, created_at, "
                    "updated_at) VALUES "
                    "('legacy-record', 'legacy-run', 'default', 'default:wiki', 0, "
                    "'private-source-do-not-print', '{}', 'unchanged', NULL, NULL, 0, "
                    ":timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )

        await upgrade(engine)

        migrated = await store.get_acquisition_run("legacy-run")
        assert migrated is not None
        assert migrated.acquisition_completed_at is None
        assert migrated.promoted_at is None
        assert migrated.completeness is None
        assert migrated.membership_hash == "legacy-unverified"
        assert migrated.omission_count == 1
        assert migrated.omission_reasons == {AcquisitionFailureCode.LEGACY_UNVERIFIED: 1}
        assert await store.latest_promoted_snapshot("wiki", "") is None
    finally:
        await engine.dispose()


async def test_migration_redacts_untyped_legacy_record_diagnostics_and_remains_resumable(
    data_dir: Path,
) -> None:
    engine = create_engine(data_dir)
    secret = "private-diagnostic-sentinel-do-not-print"  # noqa: S105 - redaction fixture
    try:
        await upgrade(engine, revision="c41d7ea923b8")
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        timestamp = "2026-08-15T13:00:00+00:00"
        diagnostics: list[str | None] = [
            json.dumps(secret),
            json.dumps([secret]),
            None,
            json.dumps(
                {
                    "stage": "acquisition",
                    "code": "not-a-real-code",
                    "retryable": True,
                    "private": secret,
                }
            ),
            json.dumps(
                {
                    "source_id": "public-source-shaped-diagnostic",
                    "uri": f"https://example.test/?private={secret}",
                    "media_type": "text/plain",
                }
            ),
            f"{{malformed-{secret}",
            json.dumps({"stage": "acquisition", "code": "authentication", "retryable": True}),
        ]
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO connectors "
                    "(id, workspace_id, name, type, config, watermark, sync_interval_seconds, "
                    "status, metadata, created_at) VALUES "
                    "('default:legacy-diagnostics', 'default', 'legacy-diagnostics', 'wiki', "
                    "'{}', NULL, 300, 'active', '{}', :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            await connection.execute(
                text(
                    "INSERT INTO acquisition_runs "
                    "(id, workspace_id, connector_id, connector_name, state, base_watermark, "
                    "candidate_watermark, enumeration_completed_at, lease_generation, "
                    "discovered_count, acquired_count, indexed_count, unchanged_count, "
                    "retry_count, metadata_bytes, acquired_blob_bytes, created_at, updated_at) "
                    "VALUES ('legacy-diagnostics-run', 'default', "
                    "'default:legacy-diagnostics', 'legacy-diagnostics', 'acquiring', NULL, NULL, "
                    ":timestamp, 0, 7, 0, 0, 7, 0, 7, 0, :timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            for sequence, diagnostic in enumerate(diagnostics):
                await connection.execute(
                    text(
                        "INSERT INTO acquisition_records "
                        "(id, run_id, workspace_id, connector_id, sequence, source_id, "
                        "source_record, state, blob_ref, acquired_source, attempts, diagnostic, "
                        "created_at, updated_at) VALUES "
                        "(:id, 'legacy-diagnostics-run', 'default', "
                        "'default:legacy-diagnostics', :sequence, :source_id, :source_record, "
                        "'unchanged', NULL, NULL, 0, :diagnostic, :timestamp, :timestamp)"
                    ),
                    {
                        "id": f"legacy-diagnostic-{sequence}",
                        "sequence": sequence,
                        "source_id": f"public-source-{sequence}",
                        "source_record": json.dumps(
                            _source(f"public-source-{sequence}").model_dump(mode="json")
                        ),
                        "diagnostic": diagnostic,
                        "timestamp": timestamp,
                    },
                )

        await upgrade(engine)

        records = await store.list_acquisition_records("legacy-diagnostics-run")
        assert len(records) == 7
        snapshot_diagnostics = [record.snapshot_diagnostic for record in records]
        assert all(diagnostic is not None for diagnostic in snapshot_diagnostics)
        assert [
            diagnostic.code for diagnostic in snapshot_diagnostics if diagnostic is not None
        ] == [
            AcquisitionFailureCode.LEGACY_UNVERIFIED,
            AcquisitionFailureCode.LEGACY_UNVERIFIED,
            AcquisitionFailureCode.LEGACY_UNVERIFIED,
            AcquisitionFailureCode.LEGACY_UNVERIFIED,
            AcquisitionFailureCode.LEGACY_UNVERIFIED,
            AcquisitionFailureCode.LEGACY_UNVERIFIED,
            AcquisitionFailureCode.AUTHENTICATION,
        ]
        assert secret not in str(records)
        claimed = await store.claim_acquisition_run(
            "legacy-diagnostics-run",
            "worker",
            now=_NOW,
            expires_at=_NOW + timedelta(minutes=1),
        )
        assert claimed is not None
        with pytest.raises(AcquisitionCoverageError, match="7 required source records") as refused:
            await store.complete_snapshot_acquisition(
                claimed.id,
                lease_owner="worker",
                lease_generation=claimed.lease_generation,
                now=_NOW,
            )
        assert secret not in str(refused.value)
        persisted = await store.get_acquisition_run(claimed.id)
        assert persisted is not None
        assert persisted.acquisition_completed_at is None
        assert persisted.promoted_at is None
        assert persisted.omission_count == 7
        assert persisted.omission_reasons == {
            AcquisitionFailureCode.LEGACY_UNVERIFIED: 6,
            AcquisitionFailureCode.AUTHENTICATION: 1,
        }
        assert not await store.verify_snapshot_manifest(claimed.id)
        assert await store.latest_promoted_snapshot("legacy-diagnostics", "") is None

        await downgrade(engine, "c41d7ea923b8")
        assert await current(engine) == "c41d7ea923b8"
    finally:
        await engine.dispose()


async def test_snapshot_migration_downgrade_refuses_promoted_manifests_with_redacted_count(
    data_dir: Path,
) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        secret_scope = "private-scope-token-do-not-print"  # noqa: S105 - redaction fixture
        created = await store.create_acquisition_run(
            "promoted-for-downgrade",
            "wiki",
            source_scope=secret_scope,
            scope_fingerprint="private-fingerprint-do-not-print",
        )
        run = await store.claim_acquisition_run(
            created.id,
            "worker",
            now=_NOW,
            expires_at=_NOW + timedelta(minutes=1),
        )
        assert run is not None
        await store.complete_acquisition_enumeration(
            run.id,
            None,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.complete_snapshot_acquisition(
            run.id,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.promote_snapshot_and_commit_watermark(
            run.id,
            expected_scope_fingerprint=run.scope_fingerprint,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )

        with pytest.raises(RuntimeError, match=r"1 promoted manifest\(s\)") as refused:
            await downgrade(engine, "c41d7ea923b8")
        assert secret_scope not in str(refused.value)
        assert run.scope_fingerprint not in str(refused.value)
        assert await current(engine) == "4d8f12a6bc91"
    finally:
        await engine.dispose()


async def test_downgrade_refuses_to_discard_acquisition_backlog(data_dir: Path) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        await _claimed_run(store, "private-enumerating-run", "enumerating-source")

        acquiring = await _claimed_run(
            store, "private-acquiring-run", "acquiring-source", "acquiring-worker"
        )
        await store.complete_acquisition_enumeration(
            acquiring.id,
            None,
            lease_owner="acquiring-worker",
            lease_generation=acquiring.lease_generation,
            now=_NOW,
        )

        indexing = await _claimed_run(
            store, "private-indexing-run", "indexing-source", "indexing-worker"
        )
        await store.complete_acquisition_enumeration(
            indexing.id,
            None,
            lease_owner="indexing-worker",
            lease_generation=indexing.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_run(
            indexing.id,
            AcquisitionRunState.ACQUIRING,
            AcquisitionRunState.INDEXING,
            lease_owner="indexing-worker",
            lease_generation=indexing.lease_generation,
            now=_NOW,
        )

        retry = await _claimed_run(store, "private-retry-run", "retry-source", "retry-worker")
        await store.append_acquisition_record(
            retry.id,
            0,
            _source("private-source-id"),
            lease_owner="retry-worker",
            lease_generation=retry.lease_generation,
            now=_NOW,
        )
        await store.complete_acquisition_enumeration(
            retry.id,
            None,
            lease_owner="retry-worker",
            lease_generation=retry.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_record(
            retry.id,
            "private-source-id",
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="retry-worker",
            lease_generation=retry.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_record(
            retry.id,
            "private-source-id",
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.RETRY,
            lease_owner="retry-worker",
            lease_generation=retry.lease_generation,
            now=_NOW,
        )

        with pytest.raises(
            RuntimeError, match=r"4 unsettled run\(s\), 1 pending record\(s\)"
        ) as raised:
            await downgrade(engine, "6e31b7d592ac")
        assert "private-retry-run" not in str(raised.value)
        assert "private-source-id" not in str(raised.value)
    finally:
        await engine.dispose()


async def test_downgrade_discards_only_fully_settled_journal_history(data_dir: Path) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        run = await _claimed_run(store)
        await store.append_acquisition_record(
            run.id,
            0,
            _source(),
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.complete_acquisition_enumeration(
            run.id,
            None,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_record(
            run.id,
            "page-1",
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.UNCHANGED,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_run(
            run.id,
            AcquisitionRunState.ACQUIRING,
            AcquisitionRunState.SETTLED,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )

        await downgrade(engine, "6e31b7d592ac")

        assert await current(engine) == "6e31b7d592ac"
    finally:
        await engine.dispose()


async def test_acquired_envelope_downgrade_refuses_with_aggregate_redacted_count(
    data_dir: Path,
) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        secret_id = "private-source-token-do-not-print"  # noqa: S105 - redaction fixture
        secret_uri = "https://example.test/private?credential=do-not-print"  # noqa: S105 - fixture
        body = b"private body text must not appear"
        blob = await BlobStore(engine, data_dir).put(body, "text/plain")
        assert isinstance(blob, StoredBlob)
        run = await _claimed_run(store)
        await store.append_acquisition_record(
            run.id,
            0,
            _source(secret_id, uri=secret_uri),
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.complete_acquisition_enumeration(
            run.id,
            None,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_record(
            run.id,
            secret_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
        )
        await store.transition_acquisition_record(
            run.id,
            secret_id,
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="worker",
            lease_generation=run.lease_generation,
            now=_NOW,
            blob_ref=blob.hash,
            acquired_source=_acquired(body, secret_id),
        )

        # Downstream migrations are clean here; isolate this migration's own refusal boundary.
        await downgrade(engine, "c41d7ea923b8")
        with pytest.raises(RuntimeError) as refused:
            await downgrade(engine, "f7c2a91d4e63")

        message = str(refused.value)
        assert "1 recoverable snapshot records" in message
        assert secret_id not in message
        assert secret_uri not in message
        assert body.decode() not in message
        # The envelope migration was isolated before exercising its own downgrade refusal.
        assert await current(engine) == "c41d7ea923b8"
    finally:
        await engine.dispose()
