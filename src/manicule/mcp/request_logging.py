"""Content-free request summaries around MCP tool execution.

The hook is deliberately ``on_call_tool`` rather than general MCP request middleware: tool
calls are the operations an operator needs to account for, while discovery, initialization and
keepalive traffic would turn the log into protocol noise.  It wraps name lookup and argument
validation as well as the tool itself, so a rejected call still produces one request record.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Literal, override

from fastmcp.server.middleware import Middleware

from manicule.app.request_logging import record_request

if TYPE_CHECKING:
    from fastmcp.server.middleware import CallNext, MiddlewareContext
    from fastmcp.tools import ToolResult
    from mcp.types import CallToolRequestParams


type Outcome = Literal["ok", "error", "cancelled", "incomplete"]


class RequestLoggingMiddleware(Middleware):
    """Record one sanitized event for every attempted tool call.

    ``operations`` comes from the server's own registration set.  A name supplied by a caller
    is only emitted when it is in that set; every unknown name becomes the fixed literal
    ``unknown``.  Arguments, results and exception text never cross this module's boundary.
    """

    def __init__(self, operations: frozenset[str]) -> None:
        self._operations = operations

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Wrap lookup, validation and execution without changing their behavior."""
        started = perf_counter()
        operation = context.message.name if context.message.name in self._operations else "unknown"
        try:
            result = await call_next(context)
        except asyncio.CancelledError:
            record_request(surface="mcp", operation=operation, outcome="cancelled", started=started)
            raise
        except Exception:
            record_request(surface="mcp", operation=operation, outcome="error", started=started)
            raise

        record_request(
            surface="mcp", operation=operation, outcome=_result_outcome(result), started=started
        )
        return result


def _result_outcome(result: ToolResult) -> Outcome:
    """Classify both MCP errors and manicule's successful MCP error envelopes."""
    if result.is_error:
        return "error"
    envelope = result.structured_content
    if isinstance(envelope, dict):
        if envelope.get("ok") is False:
            return "error"
        if envelope.get("ok") is True:
            return "ok"
    return "incomplete"


__all__ = ["RequestLoggingMiddleware"]
