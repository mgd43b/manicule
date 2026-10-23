"""Deletion detection as the product offers it: operation, schedule, status and diagnosis.

``manicule.ingest.reconcile`` and the storage under it are proven in their own suites. What none
of them could prove is that anything *calls* them, which is the gap #466 closed, so everything
here goes through :class:`~manicule.app.service.ApplicationService` or the scheduler that drives
it. The service sits over the fakes in ``tests/app/fakes.py`` — the ingest port records what it
was asked, and the store holds the connector metadata the schedule and the checks read — with the
end-to-end run against a real database in ``tests/app/test_reconcile_runtime.py``.

**Every payload is serialized and searched for the ids the fake hands back.** The ingest port is
allowed to carry source ids; nothing past the service is, and a status surface that listed what
was deleted from a mailbox would keep the thing a deletion exists to get rid of.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from manicule.api.envelopes import CONFLICT, NOT_FOUND, SERVICE_UNAVAILABLE, status_for
from manicule.app import results as r
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.connectors.errors import SessionMissingError
from manicule.core.errors import PolicyError, UnknownEntityError
from manicule.ingest.pipeline import RunReport
from manicule.ingest.reconcile import (
    LAST_DISCARDED_KEY,
    LAST_RECONCILE_KEY,
    PROPOSED_DELETION_KEY,
    Reconciliation,
)
from tests.app.fakes import FakeBackend, make_document

if TYPE_CHECKING:
    from collections.abc import Mapping

SECRET_IDS = ("secret-id-1", "secret-id-2")
"""What the fake ingest port reports it deleted. None of it may appear in any payload."""

NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)
DAY = 24 * 3600.0


def service_with(
    backend: FakeBackend | None = None, **connectors: Mapping[str, Any]
) -> tuple[ApplicationService, FakeBackend]:
    """A service whose settings carry the given sources, each holding one document."""
    backend = backend or FakeBackend()
    service = ApplicationService(backend)
    service.settings.connectors.clear()
    for name, spec in connectors.items():
        service.settings.connectors[name] = ConnectorSettings.model_validate(
            {"type": "filesystem", "options": {"root": "."}, **spec}
        )
        backend.store.add(make_document(backend.workspace, source=name, source_id="page.md"))
    return service, backend


def seen_serialized(payload: r.Payload) -> str:
    return json.dumps(payload.model_dump(mode="json"))


def assert_no_source_id(payload: r.Payload) -> None:
    text = seen_serialized(payload)
    for secret in SECRET_IDS:
        assert secret not in text


# --- the operation ------------------------------------------------------------------------------


async def test_a_pass_reports_counts_and_never_a_source_id() -> None:
    service, backend = service_with(wiki={})

    result = await service.connector_reconcile("wiki")

    assert (result.action, result.outcome) == ("reconcile", "applied")
    assert (result.seen, result.live, result.missing, result.deleted) == (11, 13, 2, 2)
    assert backend.ingestion_.reconcile_calls == [("wiki", False, 0.1)]
    assert_no_source_id(result)


async def test_a_dry_run_is_asked_for_and_reports_what_would_happen() -> None:
    service, backend = service_with(wiki={})

    result = await service.connector_reconcile("wiki", dry_run=True)

    assert (result.action, result.outcome, result.deleted) == ("dry_run", "previewed", 0)
    assert result.missing == 2
    assert backend.ingestion_.reconcile_calls == [("wiki", True, 0.1)]
    assert_no_source_id(result)


async def test_the_deletion_ceiling_that_reaches_the_pass_is_the_configured_one() -> None:
    service, backend = service_with(wiki={})
    service.settings.ingest.reconcile_max_delete_fraction = 0.25

    await service.connector_reconcile("wiki")

    assert backend.ingestion_.reconcile_calls == [("wiki", False, 0.25)]


async def test_an_unknown_connector_is_refused_before_anything_runs() -> None:
    service, backend = service_with(wiki={})

    with pytest.raises(UnknownEntityError, match="no connector named 'nope'"):
        await service.connector_reconcile("nope")

    assert backend.ingestion_.reconcile_calls == []


async def test_a_disabled_connector_is_refused_rather_than_skipped() -> None:
    """The switch an operator checked must mean the same thing for reconciliation as for sync."""
    service, backend = service_with(wiki={"enabled": False})

    with pytest.raises(PolicyError, match=r"enabled = false"):
        await service.connector_reconcile("wiki")

    assert backend.ingestion_.reconcile_calls == []


# --- how an unsuccessful pass reaches a caller ------------------------------------------------


async def test_a_pass_the_ceiling_refused_is_a_conflict_that_keeps_its_counts() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.reconcile_result = Reconciliation(
        connector="wiki",
        seen=4,
        refused="reconciling 'wiki' would soft-delete 6 of 10 live document(s)",
        proposed=SECRET_IDS,
        live_count=10,
        missing_count=6,
    )

    envelope = await run_op(
        "connector_reconcile", "default", lambda: service.connector_reconcile("wiki")
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "ReconciliationRefusedError"
    assert "6 of 10 live document(s)" in envelope.error.message
    assert "--confirm" in envelope.error.hint
    assert "--discard" in envelope.error.hint
    assert status_for(envelope) == CONFLICT
    assert envelope.data is not None
    assert (envelope.data["outcome"], envelope.data["missing"]) == ("proposed", 6)
    assert envelope.data["proposal_pending"] is True
    for secret in SECRET_IDS:
        assert secret not in json.dumps(envelope.as_json())


async def test_a_backup_holding_the_data_is_reported_as_unavailable_not_as_a_clean_pass() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.reconcile_result = Reconciliation(
        connector="wiki", blocked="a backup holds the data directory"
    )

    envelope = await run_op(
        "connector_reconcile", "default", lambda: service.connector_reconcile("wiki")
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "ReconciliationBlockedError"
    assert status_for(envelope) == SERVICE_UNAVAILABLE


async def test_a_source_that_could_not_be_enumerated_names_the_class_and_not_the_message() -> None:
    """A source client's message can carry the URL it was asking for."""
    service, backend = service_with(wiki={})
    backend.ingestion_.reconcile_result = Reconciliation(
        connector="wiki",
        seen=3,
        error="RemoteError: 502 fetching https://wiki.example.org/rest/api/secret-page-id",
        error_type="RemoteError",
    )

    envelope = await run_op(
        "connector_reconcile", "default", lambda: service.connector_reconcile("wiki")
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "RemoteError"
    assert "nothing was deleted" in envelope.error.message
    assert status_for(envelope) == SERVICE_UNAVAILABLE
    assert "wiki.example.org" not in json.dumps(envelope.as_json())
    assert "secret-page-id" not in json.dumps(envelope.as_json())


# --- confirming and discarding ----------------------------------------------------------------


async def test_confirming_a_proposal_applies_the_reviewed_set_and_reports_counts() -> None:
    service, backend = service_with(wiki={})

    result = await service.connector_reconcile_resolve("wiki", confirm=True)

    assert (result.action, result.outcome, result.deleted) == ("confirm", "confirmed", 5)
    assert backend.ingestion_.resolve_calls == [("wiki", True)]
    assert backend.ingestion_.reconcile_calls == [], "confirming must not enumerate the source"
    assert_no_source_id(result)


async def test_discarding_a_proposal_deletes_nothing() -> None:
    service, backend = service_with(wiki={})

    result = await service.connector_reconcile_resolve("wiki", confirm=False)

    assert (result.action, result.outcome, result.deleted) == ("discard", "discarded", 0)
    assert backend.ingestion_.resolve_calls == [("wiki", False)]


async def test_answering_a_proposal_that_is_not_there_says_so_as_a_not_found() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.resolve_result = Reconciliation(
        connector="wiki", refused="no durable deletion proposal exists for the current scope"
    )

    envelope = await run_op(
        "connector_reconcile_resolve",
        "default",
        lambda: service.connector_reconcile_resolve("wiki", confirm=True),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "UnknownEntityError"
    assert status_for(envelope) == NOT_FOUND


async def test_a_proposal_waiting_when_a_connector_was_switched_off_can_still_be_answered() -> None:
    service, backend = service_with(wiki={"enabled": False})

    result = await service.connector_reconcile_resolve("wiki", confirm=False)

    assert result.outcome == "discarded"
    assert backend.ingestion_.resolve_calls == [("wiki", False)]


async def test_answering_a_proposal_for_an_unknown_connector_is_refused() -> None:
    service, _ = service_with(wiki={})

    with pytest.raises(UnknownEntityError):
        await service.connector_reconcile_resolve("nope", confirm=True)


# --- the interval ---------------------------------------------------------------------------


def test_a_connectors_own_interval_overrides_the_installation_default() -> None:
    service, _ = service_with(mail={"reconcile_interval_s": 3600}, wiki={})
    service.settings.ingest.reconcile_interval_s = 7 * DAY

    assert service.reconcile_interval_s("mail") == 3600
    assert service.reconcile_interval_s("wiki") == 7 * DAY


def test_a_reconcile_interval_of_zero_or_less_is_refused() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        ConnectorSettings.model_validate({"type": "filesystem", "reconcile_interval_s": 0})


# --- when a pass is due -----------------------------------------------------------------------


def recorded(backend: FakeBackend, name: str, **metadata: object) -> None:
    backend.store.connector_state[name] = dict(metadata)


async def test_a_connector_that_has_never_reconciled_and_holds_documents_is_due() -> None:
    service, backend = service_with(wiki={})

    result = await service.reconcile_if_due("wiki")

    assert result is not None
    assert result.outcome == "applied"
    assert backend.ingestion_.reconcile_calls == [("wiki", False, 0.1)]


async def test_a_pass_inside_its_interval_is_not_due() -> None:
    service, backend = service_with(wiki={})
    recorded(backend, "wiki", **{LAST_RECONCILE_KEY: datetime.now(UTC).isoformat()})

    assert await service.reconcile_if_due("wiki") is None
    assert backend.ingestion_.reconcile_calls == []


async def test_a_pass_past_its_interval_is_due_again() -> None:
    service, backend = service_with(wiki={})
    service.settings.ingest.reconcile_interval_s = DAY
    long_ago = datetime.now(UTC) - timedelta(days=2)
    recorded(backend, "wiki", **{LAST_RECONCILE_KEY: long_ago.isoformat()})

    assert await service.reconcile_if_due("wiki") is not None


async def test_the_per_connector_interval_decides_whether_a_pass_is_due() -> None:
    service, backend = service_with(mail={"reconcile_interval_s": 600}, wiki={})
    service.settings.ingest.reconcile_interval_s = 7 * DAY
    twenty_minutes_ago = (datetime.now(UTC) - timedelta(minutes=20)).isoformat()
    recorded(backend, "mail", **{LAST_RECONCILE_KEY: twenty_minutes_ago})
    recorded(backend, "wiki", **{LAST_RECONCILE_KEY: twenty_minutes_ago})

    assert await service.reconcile_if_due("mail") is not None
    assert await service.reconcile_if_due("wiki") is None


async def test_a_connector_with_no_documents_is_not_enumerated() -> None:
    """A source nothing has been synced from has nothing to lose; asking it costs requests."""
    service, backend = service_with(wiki={})
    backend.store.documents.clear()

    assert await service.reconcile_if_due("wiki") is None
    assert backend.ingestion_.reconcile_calls == []


async def test_a_connector_with_a_proposal_waiting_is_not_asked_again() -> None:
    """A newer inventory supersedes the proposal an operator is reviewing."""
    service, backend = service_with(wiki={})
    recorded(backend, "wiki", **{PROPOSED_DELETION_KEY: {"live": 10, "missing": 6}})

    assert await service.reconcile_if_due("wiki") is None
    assert backend.ingestion_.reconcile_calls == []


async def test_a_discarded_proposal_restarts_the_interval_instead_of_being_proposed_again() -> None:
    service, backend = service_with(wiki={})
    recorded(backend, "wiki", **{LAST_DISCARDED_KEY: datetime.now(UTC).isoformat()})

    assert await service.reconcile_if_due("wiki") is None


async def test_a_disabled_connector_is_never_due() -> None:
    service, backend = service_with(wiki={"enabled": False})

    assert await service.reconcile_if_due("wiki") is None
    assert backend.ingestion_.reconcile_calls == []


# --- after a full sync ------------------------------------------------------------------------


def full_walk(**changes: object) -> RunReport:
    report = RunReport(connector="wiki", discovered=3)
    report.full_enumeration = True
    report.enumeration_completed = True
    for name, value in changes.items():
        setattr(report, name, value)
    return report


async def test_a_completed_full_sync_is_followed_by_a_pass() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.report = full_walk()

    payload = await service.connector_sync("wiki")

    assert backend.ingestion_.reconcile_calls == [("wiki", False, 0.1)]
    assert payload.reconciliation is not None
    assert payload.reconciliation.outcome == "applied"
    assert_no_source_id(payload)


@pytest.mark.parametrize(
    "report",
    [
        pytest.param(RunReport(connector="wiki"), id="an-incremental-run"),
        pytest.param(full_walk(enumeration_completed=False), id="an-enumeration-that-stopped"),
        pytest.param(full_walk(limited=True), id="a-run-stopped-by-a-limit"),
        pytest.param(full_walk(unrecorded=1), id="an-incomplete-run"),
        pytest.param(full_walk(error="boom"), id="a-run-that-errored"),
    ],
)
async def test_only_a_run_that_walked_the_whole_source_is_followed_by_a_pass(
    report: RunReport,
) -> None:
    """A prefix is exactly what reconciliation must never be handed."""
    service, backend = service_with(wiki={})
    backend.ingestion_.report = report

    payload = await service.connector_sync("wiki")

    assert backend.ingestion_.reconcile_calls == []
    assert payload.reconciliation is None


async def test_a_sync_bounded_by_a_limit_or_acquiring_only_is_not_followed_by_a_pass() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.report = full_walk()

    await service.connector_sync("wiki", limit=5)
    await service.connector_sync("wiki", acquire_only=True)

    assert backend.ingestion_.reconcile_calls == []


async def test_a_sync_is_not_followed_by_a_pass_while_a_proposal_waits() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.report = full_walk()
    recorded(backend, "wiki", **{PROPOSED_DELETION_KEY: {"live": 10, "missing": 6}})

    payload = await service.connector_sync("wiki")

    assert backend.ingestion_.reconcile_calls == []
    assert payload.reconciliation is None


async def test_a_pass_that_fails_after_a_sync_does_not_fail_the_sync() -> None:
    service, backend = service_with(wiki={})
    backend.ingestion_.report = full_walk()
    backend.ingestion_.reconcile_failure = SessionMissingError("no session held")

    payload = await service.connector_sync("wiki")

    assert payload.outcome == "complete"
    assert payload.reconciliation is not None
    assert payload.reconciliation.outcome == "failed"
    assert payload.reconciliation.error_type == "SessionMissingError"
    assert "no session held" not in seen_serialized(payload)


# --- what the status surfaces say -------------------------------------------------------------


async def test_the_listing_reports_the_interval_the_last_pass_and_whether_it_is_late() -> None:
    service, backend = service_with(mail={"reconcile_interval_s": 3600}, wiki={})
    service.settings.ingest.reconcile_interval_s = 7 * DAY
    clean = datetime.now(UTC) - timedelta(hours=5)
    recorded(backend, "mail", **{LAST_RECONCILE_KEY: clean.isoformat()})

    listing = await service.connector_list()

    mail, wiki = listing.connectors
    assert mail.name == "mail"
    assert mail.reconcile_interval_s == 3600
    assert mail.last_reconciled_at == clean.isoformat()
    assert mail.reconcile_overdue is True
    assert wiki.reconcile_interval_s == 7 * DAY
    assert wiki.last_reconciled_at is None
    assert wiki.reconcile_overdue is True, "it has documents and has never been checked"


async def test_the_listing_reports_a_waiting_proposal_as_counts() -> None:
    service, backend = service_with(wiki={})
    recorded(
        backend,
        "wiki",
        **{
            PROPOSED_DELETION_KEY: {
                "run_id": "run-1",
                "scope": "whole-connector:wiki",
                "live": 10,
                "missing": 6,
                "source_ids": list(SECRET_IDS),
            }
        },
    )

    listing = await service.connector_list()

    summary = listing.connectors[0]
    assert summary.reconcile_proposal_pending is True
    assert (summary.reconcile_proposal_missing, summary.reconcile_proposal_live) == (6, 10)
    assert summary.reconcile_overdue is False, "it is waiting on a person, not late"
    assert_no_source_id(listing)


async def test_a_disabled_connector_has_no_interval_and_is_never_late() -> None:
    service, _ = service_with(wiki={"enabled": False})

    summary = (await service.connector_list()).connectors[0]

    assert summary.reconcile_interval_s is None
    assert summary.reconcile_overdue is False


def check(diagnosis: r.Diagnosis, name: str) -> r.Check:
    found = {candidate.name: candidate for candidate in diagnosis.checks}
    assert name in found, f"no check named {name!r}; there were {sorted(found)}"
    return found[name]


async def test_doctor_reports_a_connector_overdue_for_reconciliation() -> None:
    service, backend = service_with(wiki={})
    recorded(backend, "wiki", **{LAST_RECONCILE_KEY: (NOW - timedelta(days=30)).isoformat()})

    overdue = check(await service.doctor(), "reconciliation-overdue")

    assert overdue.state == "degraded"
    assert overdue.facts == {"overdue": ["wiki"], "count": 1}
    assert overdue.remedy == "manicule connector reconcile wiki"
    assert "'wiki'" in overdue.detail


async def test_doctor_is_quiet_when_deletion_detection_is_keeping_up() -> None:
    service, backend = service_with(wiki={})
    recorded(backend, "wiki", **{LAST_RECONCILE_KEY: datetime.now(UTC).isoformat()})

    diagnosis = await service.doctor()

    assert check(diagnosis, "reconciliation-overdue").state == "ok"
    assert check(diagnosis, "reconciliation-proposals").state == "ok"


async def test_doctor_gives_a_late_pass_some_slack() -> None:
    """A pass in progress, a restart or one failed attempt must not turn doctor amber."""
    service, backend = service_with(wiki={})
    service.settings.ingest.reconcile_interval_s = DAY
    recorded(
        backend,
        "wiki",
        **{LAST_RECONCILE_KEY: (datetime.now(UTC) - timedelta(hours=40)).isoformat()},
    )

    assert check(await service.doctor(), "reconciliation-overdue").state == "ok"


async def test_doctor_does_not_flag_a_connector_that_holds_no_documents() -> None:
    service, backend = service_with(wiki={})
    backend.store.documents.clear()

    assert check(await service.doctor(), "reconciliation-overdue").state == "ok"


async def test_doctor_does_not_flag_a_disabled_connector() -> None:
    service, _ = service_with(wiki={"enabled": False})

    assert check(await service.doctor(), "reconciliation-overdue").state == "ok"


async def test_doctor_reports_an_unconfirmed_proposal_with_counts_and_no_ids() -> None:
    service, backend = service_with(wiki={})
    recorded(
        backend,
        "wiki",
        **{PROPOSED_DELETION_KEY: {"live": 10, "missing": 6, "source_ids": list(SECRET_IDS)}},
    )

    diagnosis = await service.doctor()
    proposals = check(diagnosis, "reconciliation-proposals")

    assert proposals.state == "degraded"
    assert proposals.facts == {"waiting": ["wiki"], "missing": 6}
    assert "6 of 10 live document(s)" in proposals.detail
    assert "--confirm" in proposals.detail
    assert "--discard" in proposals.detail
    assert proposals.remedy == "manicule connector list"
    for secret in SECRET_IDS:
        assert secret not in json.dumps(diagnosis.model_dump(mode="json"))
    assert check(diagnosis, "reconciliation-overdue").state == "ok", (
        "a connector waiting on a person is not late"
    )


async def test_the_new_checks_sit_beside_the_connectors_check_in_the_published_order() -> None:
    service, _ = service_with(wiki={})

    names = [candidate.name for candidate in (await service.doctor()).checks]

    position = names.index("connectors")
    assert names[position : position + 3] == [
        "connectors",
        "reconciliation-overdue",
        "reconciliation-proposals",
    ]


async def test_an_unreadable_store_makes_both_checks_unknown_rather_than_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, backend = service_with(wiki={})

    async def broken(_: str) -> Mapping[str, object]:
        raise OSError("disk went away")

    monkeypatch.setattr(backend.store, "connector_metadata", broken)

    diagnosis = await service.doctor()

    assert check(diagnosis, "reconciliation-overdue").state == "unknown"
    assert check(diagnosis, "reconciliation-proposals").state == "unknown"
