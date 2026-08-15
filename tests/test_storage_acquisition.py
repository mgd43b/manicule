"""The durable source-acquisition boundary against migrated SQLite."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
)
from manicule.core.content import RawDocument
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
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
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


def _watermark(value: str) -> Watermark:
    return Watermark(value=value, observed_at=datetime(2026, 8, 15, tzinfo=UTC))


def _source(source_id: str = "page-1", *, uri: str | None = None) -> AcquisitionSource:
    discovered = DiscoveredDoc(
        ref=DocRef(
            source_id=source_id,
            uri=uri or f"https://example.test/pages/{source_id}",
            metadata={"opaque_id": source_id},
        ),
        version_token="v1",  # noqa: S106 - source revision, not a credential
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


async def test_run_identity_and_records_do_not_cross_workspaces(
    engine: AsyncEngine,
) -> None:
    first = SqliteDocStore(engine, workspace_id="one")
    second = SqliteDocStore(engine, workspace_id="two")
    await first.ensure_workspace()
    await second.ensure_workspace()

    run = await _claimed_run(first, "same-run")
    await first.append_acquisition_record(
        "same-run",
        0,
        _source(),
        lease_owner="worker",
        lease_generation=run.lease_generation,
        now=_NOW,
    )

    assert await second.get_acquisition_run("same-run") is None
    with pytest.raises(AcquisitionConflictError, match="another connector or workspace"):
        await second.create_acquisition_run("same-run", "wiki")


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

        with pytest.raises(RuntimeError) as refused:
            await downgrade(engine, "f7c2a91d4e63")

        message = str(refused.value)
        assert "1 recoverable snapshot records" in message
        assert secret_id not in message
        assert secret_uri not in message
        assert body.decode() not in message
        assert await current(engine) == "d52f81a439bc"
    finally:
        await engine.dispose()
