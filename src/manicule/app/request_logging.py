"""Local request summaries with an explicit, content-free field set.

Transport startup installs file and stderr handlers; constructing a service or importing this
module never changes process logging. Embedders can supply their own handler on this logger.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import suppress
from datetime import UTC, datetime
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from time import perf_counter
from typing import TYPE_CHECKING, Literal, override

from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from manicule.config.settings import Settings

logger = logging.getLogger("manicule.requests")


def _private_opener(path: str, flags: int) -> int:
    """Create private files and tighten an existing log when it is reopened."""
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


class _RequestFileHandler(RotatingFileHandler):
    """Keep every new file private after rollover as well as on first startup."""

    @override
    def _open(self) -> TextIOWrapper:
        # Path.open has no opener argument. Opening with 0600 avoids a window in which a
        # newly created file has the process umask's potentially broader permissions.
        return open(
            self.baseFilename,
            "a",
            encoding=self.encoding,
            errors=self.errors,
            opener=_private_opener,
        )


def configure_request_logging(settings: Settings) -> None:
    """Install bounded JSON Lines file and stderr sinks, preserving explicit handlers.

    A server owns its data directory's writer lock. Independent processes must not point
    their rotating handlers at the same custom file.
    """
    if not settings.logging.requests or logger.handlers:
        return
    path = settings.data_dir / settings.logging.file.expanduser()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Rotation can inherit backups from a previous setup. Tighten the files this
        # handler manages before it starts moving them into later retention slots.
        for number in range(1, settings.logging.backup_count + 1):
            with suppress(FileNotFoundError):
                path.with_name(f"{path.name}.{number}").chmod(0o600)
        file_handler = _RequestFileHandler(
            path,
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        msg = (
            f"Cannot open request log {path}. Check directory permissions or set logging.file "
            "to a writable location. Set logging.requests=false to disable request logging."
        )
        raise ConfigError(msg) from exc
    formatter = logging.Formatter("%(message)s")
    for handler in (file_handler, logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # A root handler must not duplicate these lines or add a prefix that breaks JSON Lines.
    logger.propagate = False


def record_request(
    *,
    surface: Literal["http", "mcp"],
    operation: str,
    outcome: Literal["ok", "error", "canceled", "incomplete"],
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
