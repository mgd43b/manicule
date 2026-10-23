"""Deletion detection over HTTP and MCP: statuses, bodies, and what never appears in them.

The service decides what a pass means (``tests/app/test_reconcile.py``); these tests hold the two
network surfaces to the same account, with the ingest port behind the service a fake that
deliberately hands back source ids so that "counts only" has something to fail on.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.ingest.reconcile import Reconciliation
from manicule.mcp.server import build_server
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

CONFLICT = 409
NOT_FOUND = 404
UNAVAILABLE = 503
SECRET_IDS = ("secret-id-1", "secret-id-2")


def configured() -> FakeBackend:
    backend, _ = backend_with_a_document()
    backend.settings.connectors["wiki"] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )
    return backend


def test_a_pass_over_http_answers_with_counts_and_no_source_id() -> None:
    backend = configured()
    with client_for(backend) as client:
        response = client.post("/api/v1/admin/connectors/wiki/reconcile", json={})

    body = envelope(response)
    assert response.status_code == 200
    assert body["ok"] is True
    assert body["op"] == "connector_reconcile"
    assert (body["data"]["outcome"], body["data"]["deleted"]) == ("applied", 2)
    for secret in SECRET_IDS:
        assert secret not in json.dumps(body)


def test_a_dry_run_over_http_reaches_the_port_as_a_dry_run() -> None:
    backend = configured()
    with client_for(backend) as client:
        response = client.post("/api/v1/admin/connectors/wiki/reconcile", json={"dry_run": True})

    assert envelope(response)["data"]["outcome"] == "previewed"
    assert backend.ingestion_.reconcile_calls == [("wiki", True, 0.1)]


def test_a_refused_pass_is_a_409_that_keeps_its_counts() -> None:
    backend = configured()
    backend.ingestion_.reconcile_result = Reconciliation(
        connector="wiki", refused="over", proposed=SECRET_IDS, live_count=10, missing_count=6
    )
    with client_for(backend) as client:
        response = client.post("/api/v1/admin/connectors/wiki/reconcile", json={})

    body = envelope(response)
    assert response.status_code == CONFLICT
    assert body["error"]["type"] == "ReconciliationRefusedError"
    assert body["data"]["missing"] == 6
    for secret in SECRET_IDS:
        assert secret not in json.dumps(body)


def test_a_pass_held_off_by_a_backup_is_a_503() -> None:
    backend = configured()
    backend.ingestion_.reconcile_result = Reconciliation(connector="wiki", blocked="a backup")
    with client_for(backend) as client:
        response = client.post("/api/v1/admin/connectors/wiki/reconcile", json={})

    assert response.status_code == UNAVAILABLE
    assert envelope(response)["error"]["type"] == "ReconciliationBlockedError"


def test_an_unknown_connector_is_a_404() -> None:
    backend = configured()
    with client_for(backend) as client:
        response = client.post("/api/v1/admin/connectors/nope/reconcile", json={})

    assert response.status_code == NOT_FOUND
    assert envelope(response)["error"]["type"] == "UnknownEntityError"


@pytest.mark.parametrize(("action", "confirm"), [("confirm", True), ("discard", False)])
def test_a_proposal_is_answered_over_http(action: str, *, confirm: bool) -> None:
    backend = configured()
    with client_for(backend) as client:
        response = client.post(
            "/api/v1/admin/connectors/wiki/reconcile/resolve", json={"action": action}
        )

    body = envelope(response)
    assert response.status_code == 200
    assert body["data"]["action"] == action
    assert backend.ingestion_.resolve_calls == [("wiki", confirm)]


def test_an_answer_that_is_neither_confirm_nor_discard_is_refused_before_anything_runs() -> None:
    backend = configured()
    with client_for(backend) as client:
        response = client.post(
            "/api/v1/admin/connectors/wiki/reconcile/resolve", json={"action": "delete"}
        )

    assert response.status_code == 422
    assert backend.ingestion_.resolve_calls == []


def test_the_request_bodies_are_closed() -> None:
    backend = configured()
    with client_for(backend) as client:
        response = client.post(
            "/api/v1/admin/connectors/wiki/reconcile", json={"dry_run": True, "force": True}
        )

    assert response.status_code == 422
    assert backend.ingestion_.reconcile_calls == []


async def call(backend: FakeBackend, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    server = build_server(ApplicationService(backend))
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
    assert isinstance(result.structured_content, dict)
    return result.structured_content


async def test_the_tool_reports_counts_and_no_source_id() -> None:
    backend = configured()

    body = await call(backend, "connector_reconcile", {"name": "wiki", "dry_run": True})

    assert (body["op"], body["ok"], body["data"]["outcome"]) == (
        "connector_reconcile",
        True,
        "previewed",
    )
    for secret in SECRET_IDS:
        assert secret not in json.dumps(body)


async def test_the_resolve_tool_answers_a_proposal() -> None:
    backend = configured()

    body = await call(backend, "connector_reconcile_resolve", {"name": "wiki", "action": "confirm"})

    assert body["data"]["outcome"] == "confirmed"
    assert backend.ingestion_.resolve_calls == [("wiki", True)]


async def test_the_resolve_tool_refuses_an_answer_it_does_not_know() -> None:
    backend = configured()

    with pytest.raises(ToolError):
        await call(backend, "connector_reconcile_resolve", {"name": "wiki", "action": "delete"})

    assert backend.ingestion_.resolve_calls == []


async def test_the_tools_say_what_they_do_to_a_client_that_asks() -> None:
    """Hints, not authorization: but a client deciding whether to prompt reads them."""
    server = build_server(ApplicationService(configured()))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    reconcile = tools["connector_reconcile"].annotations
    resolve = tools["connector_reconcile_resolve"].annotations
    assert reconcile is not None
    assert resolve is not None
    assert (reconcile.read_only_hint, reconcile.destructive_hint, reconcile.open_world_hint) == (
        False,
        True,
        True,
    )
    assert (resolve.read_only_hint, resolve.destructive_hint, resolve.open_world_hint) == (
        False,
        True,
        False,
    )
