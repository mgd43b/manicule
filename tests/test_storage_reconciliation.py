"""Durable full inventories are the only path from enumeration to deletion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from manicule.core.acquisition import AcquisitionRecordState, AcquisitionSource
from manicule.core.reconciliation import CompletedInventory, ReconciliationAssessment
from manicule.core.sources import DiscoveredDoc, DocRef
from manicule.ingest.reconcile import (
    PROPOSED_DELETION_KEY,
    Reconciliation,
    confirm_proposed_deletion,
    reconcile,
)
from manicule.storage import models
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import session_factory
from manicule.storage.reconciliation import (
    INVENTORY_PAGE_LIMIT,
    ReconciliationInventoryError,
    ReconciliationJournalMixin,
)
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

_NOW = datetime(2026, 8, 15, 18, tzinfo=UTC)
_CONNECTOR = "wiki"
_SCOPE = "spaces=ENG roots=all"


async def _documents(store: SqliteDocStore, count: int) -> None:
    for number in range(count):
        await store.upsert_document(make_document(_CONNECTOR, f"page-{number}"))


async def _complete(
    store: SqliteDocStore,
    run_id: str,
    source_ids: Sequence[str],
    *,
    connector: str = _CONNECTOR,
    scope: str = _SCOPE,
) -> CompletedInventory:
    await store.begin_reconciliation_inventory(run_id, connector, scope)
    for offset in range(0, len(source_ids), INVENTORY_PAGE_LIMIT):
        await store.append_reconciliation_inventory_page(
            run_id,
            connector,
            scope,
            source_ids[offset : offset + INVENTORY_PAGE_LIMIT],
        )
    return await store.complete_reconciliation_inventory(run_id, connector, scope, now=_NOW)


class _MustNotEnumerate:
    name = _CONNECTOR

    async def reconcile(self) -> AsyncIterator[str]:
        raise AssertionError("the completed durable inventory should be resumed")
        yield "unreachable"  # pragma: no cover


class _FailsAfterOne:
    name = _CONNECTOR

    def __init__(self, failure: Exception) -> None:
        self._failure = failure

    async def reconcile(self) -> AsyncIterator[str]:
        yield "page-0"
        raise self._failure


async def test_a_crash_after_completion_resumes_without_reenumerating(
    store: SqliteDocStore,
) -> None:
    await _documents(store, 20)
    await _complete(store, "completed-before-crash", [f"page-{n}" for n in range(19)])

    result = await reconcile(
        _MustNotEnumerate(),  # type: ignore[arg-type] - deliberately minimal connector spy
        store,
        scope=_SCOPE,
        now=_NOW,
    )

    assert result.applied_count == 1
    assert result.missing_count == 1
    assert await store.find_document(_CONNECTOR, "page-19") is None


async def test_duplicate_and_large_inventories_are_page_bounded_and_deduplicated(
    store: SqliteDocStore,
) -> None:
    source_ids = [f"page-{number}" for number in range(2_005)]
    await store.begin_reconciliation_inventory("large", _CONNECTOR, _SCOPE)

    with pytest.raises(ValueError, match="limited"):
        await store.append_reconciliation_inventory_page(
            "large",
            _CONNECTOR,
            _SCOPE,
            [*source_ids[:INVENTORY_PAGE_LIMIT], "one-too-many"],
        )

    inserted = 0
    for offset in range(0, len(source_ids), INVENTORY_PAGE_LIMIT):
        page = source_ids[offset : offset + INVENTORY_PAGE_LIMIT]
        inserted += await store.append_reconciliation_inventory_page(
            "large", _CONNECTOR, _SCOPE, page
        )
    inserted += await store.append_reconciliation_inventory_page(
        "large", _CONNECTOR, _SCOPE, source_ids[:100]
    )
    completed = await store.complete_reconciliation_inventory("large", _CONNECTOR, _SCOPE, now=_NOW)

    assert inserted == 2_005
    assert completed.seen_count == 2_005


async def test_completed_handles_are_workspace_and_scope_bound(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    completed = await _complete(store, "scoped", [])
    other = SqliteDocStore(engine, workspace_id="other")
    await other.ensure_workspace()

    with pytest.raises(ReconciliationInventoryError, match="workspace/connector/scope"):
        await other.assess_reconciliation_inventory(completed, max_delete_fraction=0.1, now=_NOW)
    wrong_scope = completed.model_copy(update={"scope": "spaces=SECRET"})
    with pytest.raises(ReconciliationInventoryError, match="workspace/connector/scope"):
        await store.assess_reconciliation_inventory(wrong_scope, max_delete_fraction=0.1, now=_NOW)
    wrong_connector = completed.model_copy(update={"connector": "private-wiki"})
    with pytest.raises(ReconciliationInventoryError, match="workspace/connector/scope"):
        await store.assess_reconciliation_inventory(
            wrong_connector, max_delete_fraction=0.1, now=_NOW
        )


async def test_incomplete_and_canceled_inventories_cannot_propose_or_delete(
    store: SqliteDocStore,
) -> None:
    await _documents(store, 3)
    await store.begin_reconciliation_inventory("partial", _CONNECTOR, _SCOPE)
    await store.append_reconciliation_inventory_page("partial", _CONNECTOR, _SCOPE, ["page-0"])
    assert await store.cancel_reconciliation_inventory("partial", _CONNECTOR, _SCOPE)

    assert await store.latest_completed_reconciliation_inventory(_CONNECTOR, _SCOPE) is None
    with pytest.raises(ReconciliationInventoryError, match="active inventory"):
        await store.complete_reconciliation_inventory("partial", _CONNECTOR, _SCOPE, now=_NOW)
    for number in range(3):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)


@pytest.mark.parametrize(
    "failure", [PermissionError("expired auth"), OSError("capacity exhausted")]
)
async def test_auth_and_capacity_failures_never_reach_deletion_logic(
    store: SqliteDocStore, failure: Exception
) -> None:
    await _documents(store, 3)

    result = await reconcile(
        _FailsAfterOne(failure),  # type: ignore[arg-type] - deliberately minimal connector spy
        store,
        scope=_SCOPE,
        now=_NOW,
    )

    assert result.error
    assert result.applied_count == 0
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)
    for number in range(3):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None


async def test_ceiling_records_a_bounded_proposal_and_confirmation_applies_reviewed_revisions(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _documents(store, 10)
    completed = await _complete(store, "proposal", [f"page-{n}" for n in range(5)])

    proposed = await store.assess_reconciliation_inventory(
        completed, max_delete_fraction=0.1, now=_NOW
    )

    assert proposed.refused
    assert proposed.missing_count == 5
    metadata = await store.connector_metadata(_CONNECTOR)
    assert metadata[PROPOSED_DELETION_KEY] == {
        "run_id": "proposal",
        "scope": _SCOPE,
        "live": 10,
        "missing": 5,
        "recorded_at": _NOW.isoformat(),
    }
    assert "source_ids" not in metadata[PROPOSED_DELETION_KEY]

    assert (
        await store.confirm_reconciliation_proposal(_CONNECTOR, scope="a-different-scope", now=_NOW)
        is None
    )

    # Even an unchanged document observed after review is no longer the absence the human
    # confirmed. Publication and content identity deliberately remain unchanged here.
    observed = await store.find_document(_CONNECTOR, "page-7")
    assert observed is not None
    publication = observed.publication_id
    content = observed.content_hash
    sessions = session_factory(store.engine)
    async with sessions() as session:
        previous_seen = (
            await session.execute(
                select(models.Document.last_seen_at).where(models.Document.id == observed.id)
            )
        ).scalar_one()
    assert previous_seen is not None
    run = await store.create_acquisition_run("unchanged-observation", _CONNECTOR)
    lease = await store.claim_acquisition_run(
        run.id, "worker", now=_NOW, expires_at=_NOW + timedelta(minutes=1)
    )
    assert lease is not None
    source = AcquisitionSource.from_discovered(
        DiscoveredDoc(ref=DocRef(source_id="page-7", uri="https://example.test/pages/page-7"))
    )
    await store.append_acquisition_record(
        run.id,
        0,
        source,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    await store.transition_acquisition_record(
        run.id,
        "page-7",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    import manicule.storage.acquisition as acquisition_module  # noqa: PLC0415

    monkeypatch.setattr(acquisition_module, "utcnow", lambda: previous_seen)
    await store.settle_unchanged_acquisition_record(
        run.id,
        "page-7",
        observed.id,
        lease_owner="worker",
        lease_generation=lease.lease_generation,
        now=_NOW,
    )
    async with sessions() as session:
        current_seen = (
            await session.execute(
                select(models.Document.last_seen_at).where(models.Document.id == observed.id)
            )
        ).scalar_one()
    assert current_seen == previous_seen + timedelta(microseconds=1)
    observed_again = await store.find_document(_CONNECTOR, "page-7")
    assert observed_again is not None
    assert observed_again.publication_id == publication
    assert observed_again.content_hash == content
    confirmed = await store.confirm_reconciliation_proposal(_CONNECTOR, scope=_SCOPE, now=_NOW)

    assert confirmed is not None
    assert confirmed.applied_count == 4
    assert await store.find_document(_CONNECTOR, "page-7") is not None
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)


async def test_dry_run_never_proposes_or_deletes(store: SqliteDocStore) -> None:
    await _documents(store, 4)
    completed = await _complete(store, "dry", ["page-0"])

    result = await store.assess_reconciliation_inventory(
        completed, max_delete_fraction=0.1, dry_run=True, now=_NOW
    )

    assert result.dry_run
    assert result.missing_count == 3
    assert await store.latest_completed_reconciliation_inventory(_CONNECTOR, _SCOPE) is None
    with pytest.raises(ReconciliationInventoryError, match="completed inventory handle"):
        await store.assess_reconciliation_inventory(completed, max_delete_fraction=1.0, now=_NOW)
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)
    for number in range(4):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None


async def test_a_crash_during_dry_run_cannot_restore_deletion_authority(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _documents(store, 4)
    completed = await _complete(store, "crashing-dry-run", ["page-0"])

    async def crash_after_consumption(
        self: ReconciliationJournalMixin, inventory: CompletedInventory
    ) -> ReconciliationAssessment:
        del self, inventory
        raise RuntimeError("process died while computing preview")

    monkeypatch.setattr(
        ReconciliationJournalMixin,
        "_assess_consumed_dry_run",
        crash_after_consumption,
    )
    with pytest.raises(RuntimeError, match="process died"):
        await store.assess_reconciliation_inventory(
            completed, max_delete_fraction=0.1, dry_run=True, now=_NOW
        )

    assert await store.latest_completed_reconciliation_inventory(_CONNECTOR, _SCOPE) is None
    with pytest.raises(ReconciliationInventoryError, match="completed inventory handle"):
        await store.assess_reconciliation_inventory(completed, max_delete_fraction=1.0, now=_NOW)
    for number in range(4):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None


async def test_a_new_full_inventory_invalidates_an_older_confirmation_question(
    store: SqliteDocStore,
) -> None:
    await _documents(store, 4)
    completed = await _complete(store, "old-proposal", ["page-0"])
    proposed = await store.assess_reconciliation_inventory(
        completed, max_delete_fraction=0.1, now=_NOW
    )
    assert proposed.refused

    await store.begin_reconciliation_inventory("newer-run", _CONNECTOR, "new-scope")

    assert await store.confirm_reconciliation_proposal(_CONNECTOR, scope=_SCOPE, now=_NOW) is None
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)
    for number in range(4):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None


async def test_a_new_inventory_preserves_completed_evidence_from_another_scope(
    store: SqliteDocStore,
) -> None:
    completed = await _complete(store, "completed-old-scope", ["page-0"])

    await store.begin_reconciliation_inventory("new-scope-run", _CONNECTOR, "spaces=OPS")

    assert await store.latest_completed_reconciliation_inventory(_CONNECTOR, _SCOPE) == completed


async def test_durable_confirmation_rejects_an_omitted_scope(store: SqliteDocStore) -> None:
    await _documents(store, 4)
    completed = await _complete(store, "scope-required", ["page-0"])
    proposed = await store.assess_reconciliation_inventory(
        completed, max_delete_fraction=0.1, now=_NOW
    )
    assert proposed.refused

    with pytest.raises(ValueError, match="requires the current reconciliation scope"):
        await confirm_proposed_deletion(_CONNECTOR, store, now=_NOW)

    for number in range(4):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None


async def test_durable_confirmation_without_a_proposal_returns_a_typed_refusal(
    store: SqliteDocStore,
) -> None:
    result = await confirm_proposed_deletion(_CONNECTOR, store, now=_NOW, scope=_SCOPE)

    assert isinstance(result, Reconciliation)
    assert result.connector == _CONNECTOR
    assert result.refused == "no durable deletion proposal exists for the current scope"
