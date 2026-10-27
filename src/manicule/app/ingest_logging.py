"""Where a server's ingest progress lines go.

Python drops a record below ``WARNING`` when no handler has been configured for its logger and
no ancestor sets a level, which is the state every ``manicule.*`` logger is in until something
installs one. Only ``manicule.requests`` ever had a handler, so an ``INFO`` line from the ingest
pipeline, such as the per-minute stage timings, reached nobody. A test that raises the level for
itself cannot see that, which is how this shipped.

Like request logging, this runs at transport startup and never at import or at construction of a
service, so embedding manicule does not change a host process's logging. A host that configured
the root logger, or attached a handler of its own to ``manicule.ingest``, keeps its handlers
and receives the lines through them. It is not a setting: these lines carry
only counts, stage names and a connector's name, and a server that cannot say where its time went
is the failure it exists to prevent.
"""

from __future__ import annotations

import logging

INGEST_LOGGER = "manicule.ingest"
"""The logger the ingest package's loggers are all children of."""


def configure_ingest_logging() -> None:
    """Make ingest ``INFO`` lines emit, and give them a standard-error sink if nothing else has one.

    The level is the part that matters: a record is filtered by its own logger's effective level
    and then handed to handlers without the root's level being asked, so ``INFO`` on
    ``manicule.ingest`` is enough for a host's own root handlers to receive the lines. The
    standard-error handler is added only when neither this logger nor the root has a handler, so
    a configured host gets each line once, in its own format, and keeps the ingest warnings it
    already received. Standard error rather than standard output, so a stdio transport's protocol
    stream is untouched. Idempotent, and an explicit level on the logger is left as it is.
    """
    ingest = logging.getLogger(INGEST_LOGGER)
    if ingest.level == logging.NOTSET:
        ingest.setLevel(logging.INFO)
    if ingest.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ingest.addHandler(handler)


__all__ = ["INGEST_LOGGER", "configure_ingest_logging"]
