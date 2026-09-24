"""In-process rate limiting, over the real HTTP surface.

Boundaries only: exactly ``burst`` requests are admitted and the next one is refused, at 429,
with a ``Retry-After`` header — and the probes that must never be refused, whatever the load,
stay exempt. ``tests/app/test_throttle.py`` covers the bucket arithmetic itself with a fake
clock; this file is about the surface wiring it into ``identify``.
"""

from __future__ import annotations

import asyncio

from manicule.app.service import ApplicationService
from tests.api.support import backend_with_a_document, client_for, envelope
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


def test_the_failed_auth_bucket_refuses_even_a_correct_key_from_the_same_address() -> None:
    """Brute force from an address stops *that address* — not only the guesses it made.

    Once an address has exhausted its failed-authentication budget, every further request from
    it is refused before a credential is even checked, including one that would have worked.
    That is deliberate: the bucket metering *guesses* has to hold even when the very next guess
    would have been the real key, or an attacker who eventually stumbles onto — or steals — a
    valid key defeats the whole protection on the first correct attempt.
    """
    backend = _throttled(burst=100, per_minute=6000, auth={"mode": "api_key"})
    backend.settings.security.rate_limit.failed_auth_per_minute = 2
    service = ApplicationService(backend)
    issued = asyncio.run(service.api_key_create("real", role="viewer"))
    with client_for(backend) as client:
        bad = {"X-API-Key": "mnk_wrong"}
        assert client.get("/api/v1/documents", headers=bad).status_code == UNAUTHORIZED
        assert client.get("/api/v1/documents", headers=bad).status_code == UNAUTHORIZED
        good = {"X-API-Key": issued.secret}
        refused = client.get("/api/v1/documents", headers=good)
    assert refused.status_code == TOO_MANY_REQUESTS


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
