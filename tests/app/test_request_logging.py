"""Request lifetime and logging setup, without a socket or a buffered response."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP
from starlette.types import Message, Receive, Scope, Send

from manicule.app import request_logging
from manicule.app.request_logging_http import RequestLoggingMiddleware
from manicule.app.service import ApplicationService
from manicule.config.settings import LoggingSettings, Settings
from manicule.mcp.serve import serve
from tests.app.fakes import FakeBackend


def scope() -> Scope:
    return {"type": "http", "method": "GET", "path": "/private", "query_string": b"secret"}


async def receive() -> Message:
    return {"type": "http.request", "body": b""}


async def discard(message: Message) -> None:
    del message


def events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "manicule.requests"
    ]


async def test_a_stream_is_not_buffered_and_is_logged_only_after_its_last_body(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")
    now = 100.0
    monkeypatch.setattr("manicule.app.request_logging_http.perf_counter", lambda: now)
    monkeypatch.setattr(request_logging, "perf_counter", lambda: now)
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def app(request_scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal now
        del request_scope, receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"private", "more_body": True})
        assert len(sent) == 2
        assert events(caplog) == []
        now += 2
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    await RequestLoggingMiddleware(app)(scope(), receive, send)
    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
        "http.response.body",
    ]
    (event,) = events(caplog)
    assert event["duration_ms"] == 2000
    assert event["outcome"] == "ok"
    assert "private" not in json.dumps(event)


@pytest.mark.parametrize("started", [False, True])
async def test_exceptions_propagate_and_log_once_without_their_message(
    caplog: pytest.LogCaptureFixture, started: bool
) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        if started:
            await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("private exception detail")

    with pytest.raises(RuntimeError, match="private exception detail"):
        await RequestLoggingMiddleware(app)(scope(), receive, discard)
    (event,) = events(caplog)
    assert event["outcome"] == "error"
    assert event["status"] == (200 if started else 500)
    assert "private" not in json.dumps(event)


async def test_cancellation_propagates_with_no_invented_http_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive, send
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await RequestLoggingMiddleware(app)(scope(), receive, discard)
    (event,) = events(caplog)
    assert event["outcome"] == "cancelled"
    assert event["status"] is None


async def test_a_returned_but_unfinished_stream_is_incomplete(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": True})

    await RequestLoggingMiddleware(app)(scope(), receive, discard)
    assert events(caplog)[0]["outcome"] == "incomplete"


async def test_concurrent_requests_do_not_share_status_or_timing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")
    both_started = asyncio.Event()
    count = 0

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal count
        del receive
        await send({"type": "http.response.start", "status": scope["test_status"], "headers": []})
        count += 1
        if count == 2:
            both_started.set()
        await both_started.wait()
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestLoggingMiddleware(app)
    await asyncio.gather(
        middleware({**scope(), "test_status": 200}, receive, discard),
        middleware({**scope(), "test_status": 403}, receive, discard),
    )
    assert {(event["status"], event["outcome"]) for event in events(caplog)} == {
        (200, "ok"),
        (403, "error"),
    }


@pytest.mark.parametrize("kind", ["websocket", "lifespan"])
async def test_non_http_scopes_pass_through_without_logging(
    caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")
    received: list[Scope] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del receive, send
        received.append(scope)

    request_scope = {"type": kind}
    await RequestLoggingMiddleware(app)(request_scope, receive, discard)
    assert received == [request_scope]
    assert events(caplog) == []


async def test_unknown_methods_are_not_echoed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="manicule.requests")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send({"type": "http.response.start", "status": 405, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    await RequestLoggingMiddleware(app)({**scope(), "method": "SECRET"}, receive, discard)
    assert events(caplog)[0]["method"] == "OTHER"
    assert "SECRET" not in caplog.text


@pytest.fixture
def isolated_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(request_logging.logger, "handlers", [])
    monkeypatch.setattr(request_logging.logger, "level", logging.NOTSET)
    monkeypatch.setattr(request_logging.logger, "propagate", True)
    yield
    for handler in request_logging.logger.handlers:
        handler.close()


@pytest.mark.usefixtures("isolated_logger")
def test_startup_installs_one_json_stderr_handler_and_leaves_stdout_clean(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path)
    request_logging.configure_request_logging(settings)
    request_logging.configure_request_logging(settings)
    request_logging.record_request(
        surface="mcp", operation="search", outcome="ok", started=perf_counter()
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert len(output.err.splitlines()) == 1
    event = json.loads(output.err)
    assert event["operation"] == "search"
    assert datetime.fromisoformat(event["timestamp"]).tzinfo == UTC
    assert len(request_logging.logger.handlers) == 2
    assert (tmp_path / "logs" / "requests.jsonl").read_text() == output.err


@pytest.mark.usefixtures("isolated_logger")
def test_an_embedders_explicit_logger_configuration_is_preserved(tmp_path: Path) -> None:
    handler = logging.NullHandler()
    request_logging.logger.addHandler(handler)
    request_logging.logger.setLevel(logging.WARNING)
    request_logging.configure_request_logging(Settings(data_dir=tmp_path))
    assert request_logging.logger.handlers == [handler]
    assert request_logging.logger.level == logging.WARNING
    assert request_logging.logger.propagate is True
    assert not (tmp_path / "logs").exists()


def test_requests_default_on_and_the_environment_can_disable_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANICULE_LOGGING__REQUESTS", raising=False)
    assert Settings().logging.requests is True
    monkeypatch.setenv("MANICULE_LOGGING__REQUESTS", "false")
    assert Settings().logging.requests is False


@pytest.mark.usefixtures("isolated_logger")
@pytest.mark.parametrize("enabled", [False, True])
async def test_mcp_only_startup_keeps_raw_access_logs_off_and_honors_the_switch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, tmp_path: Path
) -> None:
    run = AsyncMock()
    monkeypatch.setattr(FastMCP, "run_http_async", run)
    service = ApplicationService(
        FakeBackend(settings=Settings(data_dir=tmp_path, logging=LoggingSettings(requests=enabled)))
    )
    await serve(service, transport="http")
    options = run.call_args.kwargs
    assert options["uvicorn_config"]["access_log"] is False
    assert bool(options["middleware"]) is enabled
    if enabled:
        assert options["middleware"][0].cls is RequestLoggingMiddleware
    assert bool(request_logging.logger.handlers) is enabled


@pytest.mark.usefixtures("isolated_logger")
async def test_stdio_startup_installs_the_same_stderr_sink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = AsyncMock()
    monkeypatch.setattr(FastMCP, "run_stdio_async", run)
    await serve(
        ApplicationService(FakeBackend(settings=Settings(data_dir=tmp_path))), transport="stdio"
    )
    run.assert_awaited_once_with(show_banner=False)
    assert len(request_logging.logger.handlers) == 2
    assert (tmp_path / "logs" / "requests.jsonl").exists()
