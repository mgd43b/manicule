"""Local request summaries with an explicit, content-free field set.

Transport startup installs the stderr handler; constructing a service or importing this
module never changes process logging. Embedders can supply their own handler on this logger.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

logger = logging.getLogger("manicule.requests")


def configure_request_logging() -> None:
    """Install one JSON Lines stderr sink, leaving an explicitly configured logger alone."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # A root handler must not duplicate these lines or add a prefix that breaks JSON Lines.
    logger.propagate = False


def record_request(
    *,
    surface: Literal["http", "mcp"],
    operation: str,
    outcome: Literal["ok", "error", "cancelled", "incomplete"],
    started: float,
    method: str | None = None,
    status: int | None = None,
) -> None:
    """Emit aggregate metadata only; callers supply names from the registered surface.

    No free-form exception, URL, argument or payload parameter exists here. JSON escaping
    also keeps each record on one line, even if a registered name contains a newline.
    """
    record: dict[str, object] = {
        "event": "request",
        "timestamp": datetime.now(UTC).isoformat(),
        "surface": surface,
        "operation": operation,
        "outcome": outcome,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
    }
    if surface == "http":
        record.update(method=method, status=status)
    logger.info(json.dumps(record, separators=(",", ":")))
