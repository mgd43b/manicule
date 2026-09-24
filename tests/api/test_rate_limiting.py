"""In-process rate limiting, over the real HTTP surface.

Boundaries only: exactly ``burst`` requests are admitted and the next one is refused, at 429,
with a ``Retry-After`` header — and the probes that must never be refused, whatever the load,
stay exempt. ``tests/app/test_throttle.py`` covers the bucket arithmetic itself with a fake
clock; this file is about the surface wiring it into ``identify``.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from manicule.app.service import ApplicationService
from tests.api.support import app_for, backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

TOO_MANY_REQUESTS = 429
OK = 200
UNAUTHORIZED = 401


def _throttled(burst: int = 2, per_minute: int = 120, **extra: object) -> FakeBackend:
    security: dict[str, object] = {"rate_limit": {"burst": burst, "per_minute": per_minute}}
    security.update(extra)
    backend, _ = backend_with_a_document(security=security)
    return backend


def test_exactly_burst_requests_succeed_then_the_next_is_429() -> None:
    backend = _throttled(burst=2)
    with client_for(backend) as client:
        assert client.get("/api/v1/documents").status_code == OK
        assert client.get("/api/v1/documents").status_code == OK
        third = client.get("/api/v1/documents")
    assert third.status_code == TOO_MANY_REQUESTS
    body = envelope(third)
    assert body["ok"] is False
    assert body["error"]["type"] == "RateLimitedError"
    assert "Retry-After" in third.headers
    assert int(third.headers["Retry-After"]) >= 1


def test_disabling_rate_limiting_makes_every_request_unlimited() -> None:
    backend = _throttled(burst=1)
    backend.settings.security.rate_limit.enabled = False
    with client_for(backend) as client:
        for _ in range(10):
            assert client.get("/api/v1/documents").status_code == OK


def test_health_and_ready_probes_are_exempt_from_the_caller_bucket() -> None:
    """A supervisor polling on a fixed schedule must never be refused by load."""
    backend = _throttled(burst=1)
    with client_for(backend) as client:
        for _ in range(10):
            assert client.get("/healthz").status_code == OK
            assert client.get("/readyz").status_code == OK
        # The caller bucket is untouched by the probes: an ordinary route still gets its full
        # burst afterward.
        assert client.get("/api/v1/documents").status_code == OK


def test_distinct_addresses_get_independent_buckets() -> None:
    backend = _throttled(burst=1)
    with client_for(backend, peer="203.0.113.1") as first:
        assert first.get("/api/v1/documents").status_code == OK
        assert first.get("/api/v1/documents").status_code == TOO_MANY_REQUESTS
    with client_for(backend, peer="203.0.113.2") as second:
        assert second.get("/api/v1/documents").status_code == OK


def test_a_page_request_is_refused_with_html_not_json() -> None:
    """Browser routes get the HTML refusal page, on the same content type its other refusals use."""
    backend = _throttled(burst=1)
    with client_for(backend) as client:
        client.get("/ui/", headers={"Accept": "text/html"})
        refused = client.get("/ui/", headers={"Accept": "text/html"})
    assert refused.status_code == TOO_MANY_REQUESTS
    assert "text/html" in refused.headers["content-type"]
    assert "Retry-After" in refused.headers


def test_a_per_key_rate_limit_overrides_the_installation_default() -> None:
    backend = _throttled(burst=100, per_minute=6000, auth={"mode": "api_key"})
    service = ApplicationService(backend)
    issued = asyncio.run(service.api_key_create("capped", role="viewer", rate_limit=1))
    headers = {"X-API-Key": issued.secret}
    with client_for(backend) as client:
        assert client.get("/api/v1/documents", headers=headers).status_code == OK
        capped = client.get("/api/v1/documents", headers=headers)
    assert capped.status_code == TOO_MANY_REQUESTS


def test_guessing_past_the_budget_is_refused_and_a_working_key_from_that_address_is_not() -> None:
    """Brute force from an address is slowed; nobody sharing that address is signed out by it.

    Behind a proxy nobody configured as trusted, or one office's NAT, every caller has the same
    address. A bucket that refused *correct* credentials once somebody had spent it would let
    one client with a stale key lock every member out — and would buy nothing, because a key is
    256 bits and a session is signed. So failures past the budget are refused with 429 and a
    working key from the very same address goes through.
    """
    backend = _throttled(burst=100, per_minute=6000, auth={"mode": "api_key"})
    backend.settings.security.rate_limit.failed_auth_per_minute = 2
    service = ApplicationService(backend)
    issued = asyncio.run(service.api_key_create("real", role="viewer"))
    with client_for(backend) as client:
        bad = {"X-API-Key": "mnk_wrong"}
        assert client.get("/api/v1/documents", headers=bad).status_code == UNAUTHORIZED
        assert client.get("/api/v1/documents", headers=bad).status_code == UNAUTHORIZED
        guessing = client.get("/api/v1/documents", headers=bad)
        good = {"X-API-Key": issued.secret}
        working = client.get("/api/v1/documents", headers=good)
        anonymous = client.get("/healthz")
    assert guessing.status_code == TOO_MANY_REQUESTS
    assert "Retry-After" in guessing.headers
    assert working.status_code == OK
    assert anonymous.status_code == OK


def test_one_key_arriving_from_many_addresses_raises_a_key_abuse_alert() -> None:
    """The same secret from more places than one person's devices explain is a leaked key.

    ``security.alerts.key_address_threshold`` distinct addresses within the window fire one
    ``key_abuse`` alert naming the key — and one fewer fires nothing.
    """
    backend = _throttled(burst=100, per_minute=6000, auth={"mode": "api_key"})
    backend.settings.security.alerts.key_address_threshold = 3
    service = ApplicationService(backend)
    issued = asyncio.run(service.api_key_create("shared", role="viewer"))
    headers = {"X-API-Key": issued.secret}
    # One application — one monitor — reached from three addresses.
    app = app_for(backend)
    for last in (1, 2):
        with TestClient(app, client=(f"203.0.113.{last}", 41234)) as client:
            assert client.get("/api/v1/documents", headers=headers).status_code == OK
    assert backend.security_alerts_.alerts == []
    with TestClient(app, client=("203.0.113.3", 41234)) as client:
        assert client.get("/api/v1/documents", headers=headers).status_code == OK
    assert [(alert["kind"], alert["subject"]) for alert in backend.security_alerts_.alerts] == [
        ("key_abuse", issued.key.id)
    ]


def test_a_successful_authentication_never_spends_the_failed_auth_budget() -> None:
    backend = _throttled(burst=100, per_minute=6000, auth={"mode": "api_key"})
    backend.settings.security.rate_limit.failed_auth_per_minute = 1
    service = ApplicationService(backend)
    issued = asyncio.run(service.api_key_create("real", role="viewer"))
    headers = {"X-API-Key": issued.secret}
    with client_for(backend) as client:
        for _ in range(5):
            assert client.get("/api/v1/documents", headers=headers).status_code == OK


def test_the_mcp_mount_is_metered_by_the_same_bucket_as_ordinary_routes() -> None:
    """One caller, one bucket: a request to ``/mcp`` spends the same budget as one to a route.

    ``identify`` is app-wide middleware that runs ahead of routing, so it already charges every
    request bound for the mount — this proves that rather than asserting it structurally.
    """
    backend = _throttled(burst=1)
    with client_for(backend) as client:
        assert client.get("/api/v1/documents").status_code == OK
        mcp_response = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert mcp_response.status_code == TOO_MANY_REQUESTS


def test_a_header_sent_to_an_installation_that_checks_none_is_not_a_guess() -> None:
    """Under ``auth.mode = 'none'`` nothing is verified, so a header is not a failed attempt.

    A client configured with a key keeps sending it when the server it talks to is served with
    authentication off — the ``--no-authentication`` deployment behind an ingress is exactly
    that. Counting each of those requests as a failed authentication would throttle an ordinary
    client out of an installation that asks it for nothing.
    """
    backend = _throttled(burst=100, per_minute=6000)
    backend.settings.security.rate_limit.failed_auth_per_minute = 1
    headers = {"Authorization": "Bearer mnk_configured_for_somewhere_else"}
    with client_for(backend) as client:
        for _ in range(5):
            assert client.get("/api/v1/documents", headers=headers).status_code == OK
    assert backend.security_alerts_.alerts == []


def test_a_failed_authentication_is_audited_as_nobody_rather_than_as_the_operator() -> None:
    """The row names an anonymous network caller and its address, never ``local``.

    ``local`` is the operator at this machine. A failed guess from the network recorded under
    that name would make the one row an investigation starts from point at the wrong party.
    """
    backend = _throttled(
        burst=100, per_minute=6000, auth={"mode": "api_key"}, audit={"enabled": True}
    )
    with client_for(backend, peer="203.0.113.9") as client:
        refused = client.get("/api/v1/documents", headers={"X-API-Key": "mnk_wrong"})
    assert refused.status_code == UNAUTHORIZED
    failed = [row for row in backend.telemetry_.audits if row["event_type"] == "auth.failed"]
    assert [(row["actor"], row["ip_address"]) for row in failed] == [("anonymous", "203.0.113.9")]


def test_a_key_limited_to_an_address_authenticates_from_it_and_nowhere_else() -> None:
    """``allowed_ips`` is checked against the address the proxy policy resolved for the request.

    Without the address reaching verification, a key with any ``allowed_ips`` is refused from
    everywhere — including the address it was minted for — which reads as a key that does not
    work rather than a restriction that does.
    """
    backend = _throttled(burst=100, per_minute=6000, auth={"mode": "api_key"})
    backend.keys_.memberships.clear()
    service = ApplicationService(backend)
    issued = asyncio.run(
        service.api_key_create("pinned", role="viewer", allowed_ips=["203.0.113.0/24"])
    )
    headers = {"X-API-Key": issued.secret}
    with client_for(backend, peer="203.0.113.7") as inside:
        assert inside.get("/api/v1/documents", headers=headers).status_code == OK
    with client_for(backend, peer="198.51.100.7") as outside:
        assert outside.get("/api/v1/documents", headers=headers).status_code == UNAUTHORIZED
