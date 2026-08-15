"""Durable full inventories are the only path from enumeration to deletion."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from manicule.core.reconciliation import CompletedInventory
from manicule.ingest.reconcile import PROPOSED_DELETION_KEY, reconcile
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.reconciliation import INVENTORY_PAGE_LIMIT, ReconciliationInventoryError
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
    return await store.complete_reconciliation_inventory(
        run_id, connector, scope, now=_NOW
    )


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
    completed = await store.complete_reconciliation_inventory(
        "large", _CONNECTOR, _SCOPE, now=_NOW
    )

    assert inserted == 2_005
    assert completed.seen_count == 2_005


async def test_completed_handles_are_workspace_and_scope_bound(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    completed = await _complete(store, "scoped", [])
    other = SqliteDocStore(engine, workspace_id="other")
    await other.ensure_workspace()

    with pytest.raises(ReconciliationInventoryError, match="workspace/connector/scope"):
        await other.assess_reconciliation_inventory(
            completed, max_delete_fraction=0.1, now=_NOW
        )
    wrong_scope = completed.model_copy(update={"scope": "spaces=SECRET"})
    with pytest.raises(ReconciliationInventoryError, match="workspace/connector/scope"):
        await store.assess_reconciliation_inventory(
            wrong_scope, max_delete_fraction=0.1, now=_NOW
        )
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
    await store.append_reconciliation_inventory_page(
        "partial", _CONNECTOR, _SCOPE, ["page-0"]
    )
    assert await store.cancel_reconciliation_inventory("partial", _CONNECTOR, _SCOPE)

    assert await store.latest_completed_reconciliation_inventory(_CONNECTOR, _SCOPE) is None
    with pytest.raises(ReconciliationInventoryError, match="active inventory"):
        await store.complete_reconciliation_inventory(
            "partial", _CONNECTOR, _SCOPE, now=_NOW
        )
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
    store: SqliteDocStore,
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
        await store.confirm_reconciliation_proposal(
            _CONNECTOR, scope="a-different-scope", now=_NOW
        )
        is None
    )

    # A document republished after review is no longer the revision the human confirmed.
    await store.upsert_document(make_document(_CONNECTOR, "page-7", body=b"new revision"))
    confirmed = await store.confirm_reconciliation_proposal(
        _CONNECTOR, scope=_SCOPE, now=_NOW
    )

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
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)
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

    assert await store.confirm_reconciliation_proposal(_CONNECTOR, now=_NOW) is None
    assert PROPOSED_DELETION_KEY not in await store.connector_metadata(_CONNECTOR)
    for number in range(4):
        assert await store.find_document(_CONNECTOR, f"page-{number}") is not None
