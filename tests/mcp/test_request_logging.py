"""MCP tool-call records cover the whole dispatch boundary and contain no call content."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.mcp.server import build_server
from tests.app.fakes import FakeBackend

REQUEST_LOGGER = "manicule.requests"


def _service(*, requests: bool = True) -> ApplicationService:
    settings = Settings()
    logging_settings = settings.logging.model_copy(update={"requests": requests})
    return ApplicationService(
        FakeBackend(settings=settings.model_copy(update={"logging": logging_settings}))
    )


def _listen(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)


def _events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == REQUEST_LOGGER
    ]


def _event(caplog: pytest.LogCaptureFixture) -> dict[str, Any]:
    events = _events(caplog)
    assert len(events) == 1, events
    event = events[0]
    assert set(event) == {
        "event",
        "timestamp",
        "surface",
        "operation",
        "outcome",
        "duration_ms",
    }
    assert event["event"] == "request"
    assert isinstance(event["timestamp"], str)
    assert event["surface"] == "mcp"
    assert isinstance(event["duration_ms"], float)
    assert event["duration_ms"] >= 0
    return event


async def test_a_successful_tool_call_emits_one_event(caplog: pytest.LogCaptureFixture) -> None:
    _listen(caplog)

    async with Client(build_server(_service())) as client:
        result = await client.call_tool("collection_list", {})

    assert result.structured_content is not None
    assert result.structured_content["ok"] is True
    event = _event(caplog)
    assert event["operation"] == "collection_list"
    assert event["outcome"] == "ok"


async def test_an_application_failure_envelope_is_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MCP completed normally, but manicule refused the requested operation."""
    _listen(caplog)

    async with Client(build_server(_service())) as client:
        result = await client.call_tool(
            "collection_counts", {"collection_id": "there-is-no-such-collection"}
        )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is False
    event = _event(caplog)
    assert event["operation"] == "collection_counts"
    assert event["outcome"] == "error"


@pytest.mark.parametrize(
    ("name", "arguments", "expected_operation"),
    [
        ("secret-unknown-tool", {}, "unknown"),
        ("collection_counts", {"unexpected": "private-argument"}, "collection_counts"),
    ],
)
async def test_dispatch_failures_are_logged_before_a_tool_can_run(
    caplog: pytest.LogCaptureFixture,
    name: str,
    arguments: dict[str, str],
    expected_operation: str,
) -> None:
    """Unknown names and invalid arguments both pass through ``on_call_tool``."""
    _listen(caplog)

    async with Client(build_server(_service())) as client:
        with pytest.raises(ToolError):
            await client.call_tool(name, arguments)

    event = _event(caplog)
    assert event["operation"] == expected_operation
    assert event["outcome"] == "error"
    assert "secret-unknown-tool" not in json.dumps(event)
    assert "private-argument" not in json.dumps(event)


async def test_an_unexpected_exception_is_logged_without_its_message(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _listen(caplog)
    service = _service()
    private_message = "exception-message-that-must-not-be-recorded"

    async def explode() -> Any:
        raise RuntimeError(private_message)

    monkeypatch.setattr(service, "collection_list", explode)
    async with Client(build_server(service)) as client:
        with pytest.raises(ToolError):
            await client.call_tool("collection_list", {})

    event = _event(caplog)
    assert event["operation"] == "collection_list"
    assert event["outcome"] == "error"
    assert private_message not in json.dumps(event)


async def test_cancellation_is_recorded_and_still_propagates(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _listen(caplog)
    service = _service()
    entered = asyncio.Event()
    never = asyncio.Event()

    async def blocked() -> Any:
        entered.set()
        await never.wait()

    monkeypatch.setattr(service, "collection_list", blocked)
    async with Client(build_server(service)) as client:
        task = asyncio.create_task(client.call_tool("collection_list", {}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    event = _event(caplog)
    assert event["operation"] == "collection_list"
    assert event["outcome"] == "cancelled"


async def test_request_logging_can_be_disabled(caplog: pytest.LogCaptureFixture) -> None:
    _listen(caplog)

    async with Client(build_server(_service(requests=False))) as client:
        result = await client.call_tool("collection_list", {})

    assert result.structured_content is not None
    assert result.structured_content["ok"] is True
    assert _events(caplog) == []
