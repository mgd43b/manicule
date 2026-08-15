"""Deletion detection, and the three guards that keep it from deleting a corpus."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from manicule.core.content import DocumentStatus
from manicule.core.ids import document_id
from manicule.ingest.reconcile import (
    LAST_RECONCILE_KEY,
    PROPOSED_DELETION_KEY,
    confirm_proposed_deletion,
    due,
    reconcile,
)
from tests.fakes import make_document
from tests.ingest import fakes


def _indexed(store: fakes.MemoryIngestStore, source: str, source_ids: list[str]) -> None:
    for source_id in source_ids:
        document = make_document().model_copy(
            update={
                "id": document_id("default", source, source_id),
                "source": source,
                "source_id": source_id,
                "status": DocumentStatus.INDEXED,
            }
        )
        store.documents[document.id] = document


async def test_a_document_that_stopped_appearing_is_soft_deleted() -> None:
    """Incremental sync cannot see this: a deleted page simply stops being mentioned."""
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(20)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(20)})
    connector.hidden = {"doc-7"}

    result = await reconcile(connector, store)

    assert result.deleted == ("doc-7",)
    assert result.clean
    assert store.deleted_at, "soft, always: restore is clearing a timestamp"


async def test_a_partial_enumeration_deletes_nothing_at_all() -> None:
    """Guard 1, and the failure that makes reconciliation dangerous.

    The ids seen before a 429 are a *prefix*, not the truth. Diffing a prefix against the
    stored set marks everything not yet enumerated as deleted — one transient error and the
    corpus is gone. Partial results are discarded, never salvaged.
    """
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(20)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(20)})
    connector.reconcile_fails_after = 3

    result = await reconcile(connector, store)

    assert not result.clean
    assert result.deleted == ()
    assert store.deleted_at == {}
    assert "429" in result.error


async def test_a_partial_enumeration_does_not_advance_the_reconcile_clock() -> None:
    """Otherwise a source that fails at the same point every week looks perfectly healthy."""
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", ["a", "b"])
    connector = fakes.DictConnector({"a": "text", "b": "text"})
    connector.reconcile_fails_after = 1

    await reconcile(connector, store)

    assert LAST_RECONCILE_KEY not in store.connector_meta.get("memory", {})


async def test_a_deletion_above_the_ceiling_is_refused_and_recorded() -> None:
    """Guard 2. A genuine bulk deletion is rare and worth a human; a bug is not rare at all."""
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(10)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(10)})
    connector.hidden = {f"doc-{n}" for n in range(5)}

    result = await reconcile(connector, store, max_delete_fraction=0.1)

    assert result.refused
    assert not result.clean
    assert store.deleted_at == {}, "nothing is deleted while a proposal is awaiting confirmation"
    assert PROPOSED_DELETION_KEY in store.connector_meta["memory"]


async def test_a_refused_proposal_can_be_confirmed_without_re_enumerating() -> None:
    """Confirming is a decision about the set somebody looked at.

    Re-enumerating first would confirm a different set than the one that was reviewed, which
    is the failure mode of every "are you sure?" that recomputes its own question.
    """
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(10)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(10)})
    connector.hidden = {f"doc-{n}" for n in range(5)}
    await reconcile(connector, store, max_delete_fraction=0.1)

    applied = await confirm_proposed_deletion("memory", store)

    assert len(applied) == 5
    assert len(store.deleted_at) == 5
    assert PROPOSED_DELETION_KEY not in store.connector_meta["memory"]


async def test_reconciliation_never_hard_deletes() -> None:
    """Guard 3 is what makes guard 2 tunable rather than terrifying."""
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(20)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(20)})
    connector.hidden = {"doc-3"}

    await reconcile(connector, store)

    assert len(store.documents) == 20, "the rows are all still there; only a timestamp moved"


async def test_a_dry_run_never_records_a_proposal_or_soft_deletes() -> None:
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", ["a", "b", "c"])
    connector = fakes.DictConnector({"a": "text", "b": "text", "c": "text"})
    connector.hidden = {"b", "c"}

    result = await reconcile(connector, store, max_delete_fraction=0.1, dry_run=True)

    assert result.dry_run
    assert result.missing_count == 2
    assert store.deleted_at == {}
    assert PROPOSED_DELETION_KEY not in store.connector_meta.get("memory", {})


async def test_a_clean_pass_records_when_it_happened() -> None:
    """Deletion detection that runs when somebody remembers is deletion detection that does not."""
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", ["a"])
    connector = fakes.DictConnector({"a": "text"})

    await reconcile(connector, store)

    assert LAST_RECONCILE_KEY in store.connector_meta["memory"]


def test_a_connector_that_has_never_reconciled_is_due() -> None:
    """The answer that matters most, and the one an ``is None`` check keeps getting wrong."""
    assert due({}, interval_s=3600)


def test_a_connector_reconciled_within_the_interval_is_not_due() -> None:
    recent = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    assert not due({LAST_RECONCILE_KEY: recent}, interval_s=3600)


def test_a_connector_reconciled_before_the_interval_is_due() -> None:
    stale = (datetime.now(UTC) - timedelta(days=14)).isoformat()
    assert due({LAST_RECONCILE_KEY: stale}, interval_s=7 * 24 * 3600)


def test_an_unreadable_timestamp_is_treated_as_overdue() -> None:
    """Failing closed: an unreadable record must not silently disable deletion detection."""
    assert due({LAST_RECONCILE_KEY: "not a timestamp"}, interval_s=1)
