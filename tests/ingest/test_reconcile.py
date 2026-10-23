"""Deletion detection, and the three guards that keep it from deleting a corpus."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from manicule.core.content import DocumentStatus
from manicule.core.ids import document_id
from manicule.ingest.reconcile import (
    LAST_DISCARDED_KEY,
    LAST_RECONCILE_KEY,
    PROPOSED_DELETION_KEY,
    confirm_proposed_deletion,
    discard_proposed_deletion,
    due,
    last_attempt,
    reconcile,
    standing,
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


async def test_confirming_a_proposal_retires_what_was_derived_from_it() -> None:
    """The confirmation path removes the same things the unproposed one does.

    Two paths reach a deletion — straight through when the ceiling is not crossed, and through
    a reviewed proposal when it is — and only one of them had been taught that a soft delete
    does not cascade. A member left behind here is unreachable for good: its source id was never
    in any inventory, so no later pass can name it.
    """
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(10)])
    container = store.documents[document_id("default", "memory", "doc-0")]
    member = make_document().model_copy(
        update={
            "id": document_id("default", "memory", "zip:doc-0!/inside.txt"),
            "source": "memory",
            "source_id": "zip:doc-0!/inside.txt",
            "status": DocumentStatus.INDEXED,
            "container_id": container.id,
            "container_depth": 1,
        }
    )
    store.documents[member.id] = member
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(10)})
    connector.hidden = {f"doc-{n}" for n in range(5)}
    await reconcile(connector, store, max_delete_fraction=0.1)

    await confirm_proposed_deletion("memory", store)

    assert member.id in store.deleted_at, (
        "the member of a confirmed-deleted container has to go with it, or it is live and "
        "unreachable — connector reconciliation excludes derived documents by design"
    )


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


async def test_a_partial_enumeration_names_the_failure_class_for_the_surfaces_to_report() -> None:
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(20)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(20)})
    connector.reconcile_fails_after = 3

    result = await reconcile(connector, store)

    assert (result.error_type, result.deleted) == ("RuntimeError", ())


async def test_a_refused_proposal_can_be_discarded_and_deletes_nothing() -> None:
    store = fakes.MemoryIngestStore()
    _indexed(store, "memory", [f"doc-{n}" for n in range(10)])
    connector = fakes.DictConnector({f"doc-{n}": "text" for n in range(10)})
    connector.hidden = {f"doc-{n}" for n in range(5)}
    await reconcile(connector, store, max_delete_fraction=0.1)

    result = await discard_proposed_deletion("memory", store, now=NOW)

    assert (result.missing_count, result.live_count, result.applied_count) == (5, 10, 0)
    assert store.deleted_at == {}
    metadata = store.connector_meta["memory"]
    assert PROPOSED_DELETION_KEY not in metadata
    assert metadata[LAST_DISCARDED_KEY] == NOW.isoformat()
    assert LAST_RECONCILE_KEY not in metadata, "a discard is not a clean pass"
    assert (await discard_proposed_deletion("memory", store, now=NOW)).refused


NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)


def test_a_connector_with_a_proposal_waiting_is_not_due_however_old_it_is() -> None:
    """A newer inventory would take the proposal from the person answering it."""
    metadata = {
        LAST_RECONCILE_KEY: (NOW - timedelta(days=90)).isoformat(),
        PROPOSED_DELETION_KEY: {"live": 10, "missing": 6},
    }

    assert not due(metadata, interval_s=3600, now=NOW)


def test_a_discard_restarts_the_interval_without_counting_as_a_clean_pass() -> None:
    metadata = {
        LAST_RECONCILE_KEY: (NOW - timedelta(days=30)).isoformat(),
        LAST_DISCARDED_KEY: (NOW - timedelta(minutes=5)).isoformat(),
    }

    assert not due(metadata, interval_s=3600, now=NOW)
    assert last_attempt(metadata) == NOW - timedelta(minutes=5)
    assert standing(metadata, live_documents=1).last_clean_at == NOW - timedelta(days=30)


def test_overdue_allows_a_pass_in_progress_some_slack() -> None:
    metadata = {LAST_RECONCILE_KEY: (NOW - timedelta(seconds=5000)).isoformat()}
    state = standing(metadata, live_documents=4)

    assert state.due(interval_s=3600, now=NOW)
    assert not state.overdue(interval_s=3600, now=NOW), "late is not yet overdue"
    assert state.overdue(interval_s=2000, now=NOW)


def test_a_connector_with_no_documents_is_neither_due_nor_overdue() -> None:
    state = standing({}, live_documents=0)

    assert not state.due(interval_s=60, now=NOW)
    assert not state.overdue(interval_s=60, now=NOW)


def test_standing_reads_counts_from_either_kind_of_proposal() -> None:
    durable = standing(
        {PROPOSED_DELETION_KEY: {"live": 10, "missing": 6, "recorded_at": NOW.isoformat()}},
        live_documents=10,
    )
    in_memory = standing(
        {PROPOSED_DELETION_KEY: {"live": 10, "source_ids": ["a", "b", "c"]}}, live_documents=10
    )

    assert (durable.proposal_pending, durable.proposal_missing, durable.proposal_live) == (
        True,
        6,
        10,
    )
    assert durable.proposal_recorded_at == NOW
    assert (in_memory.proposal_missing, in_memory.proposal_live) == (3, 10)


def test_standing_survives_a_malformed_record() -> None:
    state = standing(
        {LAST_RECONCILE_KEY: "not a time", PROPOSED_DELETION_KEY: {"live": "x"}}, live_documents=2
    )

    assert state.last_clean_at is None
    assert state.proposal_pending is True
    assert (state.proposal_missing, state.proposal_live) == (0, 0)
