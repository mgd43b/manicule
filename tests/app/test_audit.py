"""Every audited operation names who did it and where from, and covers the events #13 asks for.

Before this slice, ``_audit`` called ``telemetry.record_audit`` with neither ``actor`` nor
``ip_address`` — the two keyword arguments the protocol has always accepted — so **every** row
in the audit trail read as if nobody had done it, from nowhere. The first test below is the
regression: it fails against the unfixed method, because the unfixed method never reads
:func:`~manicule.app.caller.current` at all.
"""

from __future__ import annotations

import pytest

from manicule.app.caller import Caller, acting_as
from manicule.app.service import ApplicationService
from manicule.config.settings import Role, Settings
from tests.app.fakes import FakeBackend


def _audited_backend() -> FakeBackend:
    return FakeBackend(settings=Settings(security={"audit": {"enabled": True}}))  # pyright: ignore[reportArgumentType]


@pytest.fixture
def backend() -> FakeBackend:
    return _audited_backend()


@pytest.fixture
def service(backend: FakeBackend) -> ApplicationService:
    return ApplicationService(backend)


# --- the regression: actor and address on every row ---------------------------------------------


async def test_an_audited_operation_records_the_acting_callers_key_and_address(
    backend: FakeBackend, service: ApplicationService
) -> None:
    """The defect this slice fixes, pinned down: a networked caller's key id and address must
    reach the row, not just a record that *something* happened.

    Run this against ``_audit`` with ``actor=`` and ``ip_address=`` removed from its call to
    ``telemetry.record_audit`` and it fails — both fields come back ``None`` — which is what
    made the whole audit trail unusable for "who did this and from where" before the fix.
    """
    caller = Caller(role=Role.ADMIN, key_id="key-77", address="203.0.113.9")
    with acting_as(caller):
        await service.api_key_create("widget", role="member")
    [row] = backend.telemetry_.audits
    assert row["actor"] == "key-77"
    assert row["ip_address"] == "203.0.113.9"


async def test_the_local_operator_is_recorded_as_local(
    backend: FakeBackend, service: ApplicationService
) -> None:
    """The CLI and stdio-MCP path: nothing calls ``acting_as`` there, so the context variable's
    own default — the local operator — is what ends up on the row."""
    await service.api_key_create("widget", role="member")
    [row] = backend.telemetry_.audits
    assert row["actor"] == "local"
    assert row["ip_address"] is None


async def test_a_caller_with_no_key_id_is_recorded_by_user_id_then_as_anonymous(
    backend: FakeBackend, service: ApplicationService
) -> None:
    with acting_as(Caller(role=Role.MEMBER, user_id="alice", address="203.0.113.1")):
        await service.api_key_create("alices-key", role="viewer")
    [row] = backend.telemetry_.audits
    assert row["actor"] == "alice"


async def test_an_empty_address_is_recorded_as_no_address_rather_than_an_empty_string(
    backend: FakeBackend, service: ApplicationService
) -> None:
    """``current().address or None`` — an empty string is not a value worth storing as one."""
    with acting_as(Caller(role=Role.ADMIN, key_id="key-1", address="")):
        await service.api_key_create("widget", role="member")
    [row] = backend.telemetry_.audits
    assert row["ip_address"] is None


# --- coverage: the events #13 asks for that this slice adds -------------------------------------


async def test_a_failed_authentication_is_audited_with_the_credential_kind_and_never_the_value(
    backend: FakeBackend, service: ApplicationService
) -> None:
    with acting_as(Caller(address="203.0.113.2")):
        await service.record_failed_authentication(credential_kind="bearer")
    [row] = backend.telemetry_.audits
    assert row["event_type"] == "auth.failed"
    assert row["details"] == {"credential_kind": "bearer"}


async def test_a_config_change_is_audited_by_key_never_by_value(
    backend: FakeBackend, service: ApplicationService
) -> None:
    await service.config_set("rag.profile", "fast")
    [row] = backend.telemetry_.audits
    assert row["event_type"] == "config.changed"
    assert row["details"] == {"key": "rag.profile"}
    assert "fast" not in str(row["details"])


async def test_a_workspace_switch_is_audited(
    backend: FakeBackend, service: ApplicationService
) -> None:
    await service.workspace_switch("other", create=True)
    events = [row["event_type"] for row in backend.telemetry_.audits]
    assert "workspace.switched" in events


async def test_a_collection_create_and_delete_are_audited(
    backend: FakeBackend, service: ApplicationService
) -> None:
    created = await service.collection_create("runbooks")
    await service.collection_delete(created.id)
    events = [row["event_type"] for row in backend.telemetry_.audits]
    assert events == ["collection.created", "collection.deleted"]


async def test_an_api_key_creation_is_audited_with_owner_allowed_ips_and_rate_limit(
    backend: FakeBackend, service: ApplicationService
) -> None:
    with acting_as(Caller(role=Role.ADMIN)):
        await service.api_key_create(
            "widget", role="member", allowed_ips=["203.0.113.0/24"], rate_limit=30
        )
    [row] = backend.telemetry_.audits
    details = row["details"]
    assert isinstance(details, dict)
    assert details["allowed_ips"] == ["203.0.113.0/24"]
    assert details["rate_limit"] == 30
    assert details["owner"] is None


async def test_a_security_alert_is_audited_and_persisted_even_though_it_is_a_second_write(
    backend: FakeBackend, service: ApplicationService
) -> None:
    from manicule.app.alerts import AlertEvent  # noqa: PLC0415 - only this test needs it

    event = AlertEvent(kind="brute_force", subject="203.0.113.9", details={"count": 20})
    await service.record_security_alert(event)
    [row] = backend.security_alerts_.alerts
    assert row["kind"] == "brute_force"
    events = [entry["event_type"] for entry in backend.telemetry_.audits]
    assert events == ["security.alert"]


async def test_a_security_alert_is_recorded_even_when_auditing_is_off() -> None:
    """The model's own docstring: an alert nobody could see is not one, so the row and the
    warning log happen regardless of ``security.audit.enabled`` — only the *audit-log* entry
    follows that switch, like every other event ``_audit`` writes."""
    backend = FakeBackend()  # auditing off by default
    service = ApplicationService(backend)
    from manicule.app.alerts import AlertEvent  # noqa: PLC0415

    event = AlertEvent(kind="brute_force", subject="203.0.113.9", details={"count": 20})
    await service.record_security_alert(event)
    assert len(backend.security_alerts_.alerts) == 1
    assert backend.telemetry_.audits == []


async def test_acknowledging_an_alert_is_audited_and_records_who(
    backend: FakeBackend, service: ApplicationService
) -> None:
    alert_id = await backend.security_alerts_.record_alert("brute_force", "203.0.113.9", details={})
    with acting_as(Caller(role=Role.ADMIN, key_id="admin-key")):
        acknowledged = await service.security_alert_acknowledge(alert_id)
    assert acknowledged.acknowledged_by == "admin-key"
    events = [row["event_type"] for row in backend.telemetry_.audits]
    assert events == ["security.alert_acknowledged"]


# --- the doctor check --------------------------------------------------------------------------


async def test_doctor_reports_ok_with_no_unacknowledged_alerts(
    service: ApplicationService,
) -> None:
    diagnosis = await service.doctor()
    check = next(item for item in diagnosis.checks if item.name == "security_alerts")
    assert check.state == "ok"


async def test_doctor_is_degraded_while_an_alert_is_unacknowledged_and_names_the_count(
    backend: FakeBackend, service: ApplicationService
) -> None:
    await backend.security_alerts_.record_alert("brute_force", "203.0.113.9", details={})
    await backend.security_alerts_.record_alert("key_abuse", "key-1", details={})
    diagnosis = await service.doctor()
    check = next(item for item in diagnosis.checks if item.name == "security_alerts")
    assert check.state == "degraded"
    assert check.facts["unacknowledged"] == 2
    assert "auth alerts" in check.remedy
    assert "auth ack-alert" in check.remedy


async def test_doctor_returns_to_ok_once_every_alert_is_acknowledged(
    backend: FakeBackend, service: ApplicationService
) -> None:
    alert_id = await backend.security_alerts_.record_alert("brute_force", "203.0.113.9", details={})
    with acting_as(Caller(role=Role.ADMIN)):
        await service.security_alert_acknowledge(alert_id)
    diagnosis = await service.doctor()
    check = next(item for item in diagnosis.checks if item.name == "security_alerts")
    assert check.state == "ok"
