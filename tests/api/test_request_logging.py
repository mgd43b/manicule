"""The HTTP access record is useful without becoming a second credential store.

These tests drive the production application so the request record covers the middleware,
FastAPI's exception handlers and the mounted route table together. The record deliberately names
the operation and outcome rather than copying a URL: a shared conversation token is a bearer
credential, and an access log must not become another place that credential lives.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from tests.api.support import backend_with_a_document, client_for, envelope

REQUEST_LOGGER = "manicule.requests"
EVIL_ORIGIN = "https://evil.example"
WIDGET_ORIGIN = "https://widget.example"
HOST = "127.0.0.1:8765"


def _event(caplog: Any) -> dict[str, Any]:
    """Return the one structured request record emitted by the test's request."""
    records = [record for record in caplog.records if record.name == REQUEST_LOGGER]
    assert len(records) == 1, [record.getMessage() for record in records]
    payload = json.loads(records[0].getMessage())
    assert set(payload) == {
        "event",
        "surface",
        "operation",
        "outcome",
        "duration_ms",
        "method",
        "status",
        "timestamp",
    }
    stamp = datetime.fromisoformat(payload["timestamp"])
    assert stamp.tzinfo is not None
    assert stamp.utcoffset() == UTC.utcoffset(stamp)
    return payload


def _logging(caplog: Any) -> None:
    """Make the request logger visible without changing the application's logger policy."""
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)


def test_a_successful_request_emits_one_sanitized_event(caplog: Any) -> None:
    _logging(caplog)
    backend, _ = backend_with_a_document()

    with client_for(backend) as client:
        response = client.get("/api/v1/documents")

    assert response.status_code == 200
    event = _event(caplog)
    assert event["event"] == "request"
    assert event["surface"] == "http"
    assert event["operation"] == "document_list"
    assert event["outcome"] == "ok"
    assert event["method"] == "GET"
    assert event["status"] == 200
    assert isinstance(event["duration_ms"], float)
    assert event["duration_ms"] >= 0


def test_authentication_refusal_is_logged_as_an_error(caplog: Any) -> None:
    _logging(caplog)
    backend, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})

    with client_for(backend) as client:
        response = client.get("/api/v1/documents")

    assert response.status_code == 401
    assert envelope(response)["ok"] is False
    event = _event(caplog)
    assert event["operation"] == "document_list"
    assert event["outcome"] == "error"
    assert event["status"] == 401


def test_cross_origin_refusal_is_logged_once(caplog: Any) -> None:
    _logging(caplog)
    backend, document = backend_with_a_document()

    with client_for(backend) as client:
        response = client.post(
            f"/api/v1/documents/{document.id}/restore",
            headers={"Origin": EVIL_ORIGIN, "Sec-Fetch-Site": "cross-site", "Host": HOST},
        )

    assert response.status_code == 403
    event = _event(caplog)
    # The guard runs in the outer middleware before Starlette has matched a route.
    assert event["operation"] == "unmatched"
    assert event["outcome"] == "error"
    assert event["status"] == 403


def test_not_found_and_validation_failures_have_their_route_outcomes(caplog: Any) -> None:
    _logging(caplog)
    backend, _ = backend_with_a_document()

    with client_for(backend) as client:
        missing = client.get("/there-is-no-such-route")
        assert missing.status_code == 404
        missing_event = _event(caplog)
        assert missing_event["operation"] == "unmatched"
        assert missing_event["outcome"] == "error"
        assert missing_event["status"] == 404

        caplog.clear()
        invalid = client.post("/api/v1/documents")

    assert invalid.status_code == 422
    invalid_event = _event(caplog)
    assert invalid_event["operation"] == "document_create"
    assert invalid_event["outcome"] == "error"
    assert invalid_event["status"] == 422


def test_a_share_token_never_reaches_the_request_log(caplog: Any) -> None:
    _logging(caplog)
    share_value = "share-token-that-must-stay-out-of-logs"
    backend, _ = backend_with_a_document()

    with client_for(backend) as client:
        response = client.get(f"/shared/{share_value}")

    assert response.status_code == 200
    event = _event(caplog)
    assert event["operation"] == "shared_conversation"
    assert share_value not in json.dumps(event)


def test_a_cors_preflight_is_logged_without_a_path_or_query(caplog: Any) -> None:
    _logging(caplog)
    backend, _ = backend_with_a_document(
        security={"transport": {"allowed_origins": [WIDGET_ORIGIN]}}
    )

    with client_for(backend) as client:
        response = client.options(
            "/api/v1/documents?api_key=never-log-this",
            headers={
                "Origin": WIDGET_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )

    assert response.status_code == 200
    event = _event(caplog)
    assert event["operation"] == "unmatched"
    assert event["outcome"] == "ok"
    assert event["method"] == "OPTIONS"
    assert event["status"] == 200


def test_request_logging_can_be_disabled(caplog: Any) -> None:
    _logging(caplog)
    backend, _ = backend_with_a_document(logging={"requests": False})

    with client_for(backend) as client:
        response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert [record for record in caplog.records if record.name == REQUEST_LOGGER] == []
