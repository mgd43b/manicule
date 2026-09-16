"""ASGI access summaries that never inspect a URL or buffer a streaming response."""

from __future__ import annotations

from asyncio import CancelledError
from time import perf_counter
from typing import Literal

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from manicule.app.request_logging import record_request

METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)
ERROR_STATUS = 400


class RequestLoggingMiddleware:
    """Observe the entire response lifetime, including streams and early middleware refusals.

    All state belongs to the invocation: concurrent requests cannot inherit each other's
    status. WebSocket frames and lifespan messages pass through unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        status: int | None = None
        complete = False
        outcome: Literal["ok", "error", "cancelled", "incomplete"] = "error"

        async def observe(message: Message) -> None:
            nonlocal status, complete
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True

        try:
            await self.app(scope, receive, observe)
            outcome = "incomplete" if not complete else "ok"
            if status is not None and status >= ERROR_STATUS:
                outcome = "error"
        except CancelledError:
            outcome = "cancelled"
            raise
        finally:
            # Routing adds only server-defined objects. An unmatched URL never becomes a
            # fallback name: paths can carry live share credentials, including on a 404.
            route = scope.get("route")
            operation = getattr(route, "name", None)
            if not isinstance(operation, str) or not operation:
                operation = "unmatched"
            method = scope.get("method", "")
            record_request(
                surface="http",
                operation=operation,
                outcome=outcome,
                started=started,
                method=method if method in METHODS else "OTHER",
                status=500 if status is None and outcome == "error" else status,
            )
