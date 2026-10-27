"""Where a server's ingest progress lines go.

Python drops a record below ``WARNING`` when no handler has been configured for its logger and
no ancestor sets a level, which is the state every ``manicule.*`` logger is in until something
installs one. Only ``manicule.requests`` ever had a handler, so an ``INFO`` line from the ingest
pipeline, such as the per-minute stage timings, reached nobody. A test that raises the level for
itself cannot see that, which is how this shipped.

Like request logging, this runs at transport startup and never at import or at construction of a
service, so embedding manicule does not change a host process's logging. An embedder who attached
a handler of their own to ``manicule.ingest`` keeps it. It is not a setting: these lines carry
only counts, stage names and a connector's name, and a server that cannot say where its time went
is the failure it exists to prevent.
"""

from __future__ import annotations

import logging

INGEST_LOGGER = "manicule.ingest"
"""The logger the ingest package's loggers are all children of."""


def configure_ingest_logging() -> None:
    """Send ingest ``INFO`` and above to standard error, once, preserving explicit handlers.

    Standard error rather than standard output, so a stdio transport's protocol stream is
    untouched. Idempotent: a second call finds the handler and returns.
    """
    ingest = logging.getLogger(INGEST_LOGGER)
    if ingest.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ingest.addHandler(handler)
    ingest.setLevel(logging.INFO)
    # A root handler must not print every line twice with a second prefix.
    ingest.propagate = False


__all__ = ["INGEST_LOGGER", "configure_ingest_logging"]
