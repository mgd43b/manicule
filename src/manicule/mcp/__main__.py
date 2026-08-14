"""``python -m manicule.mcp`` — the MCP server over stdio.

Here as well as ``manicule start --mcp-only`` because a client that spawns a server usually
wants to name an interpreter and a module rather than trust a console script to be on the
PATH it happens to have. Same server, same service, same tools.
"""

from __future__ import annotations

import asyncio
import sys

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.core.errors import ManiculeError
from manicule.mcp.serve import serve


async def _run() -> None:
    # The MCP surface can index, sync and reindex, so this process is a writer and holds the
    # data directory for as long as it serves. `acquire` before the runtime is entered so that
    # a second server refuses on the way in — a stdio server that had already begun speaking
    # the protocol would report the refusal as a tool failure much later, or not at all.
    runtime = Runtime.open()
    runtime.acquire()
    async with runtime:
        await serve(ApplicationService(runtime), transport="stdio")


def main() -> None:
    """Start the server over stdio, which opens no socket.

    A refusal is written to stderr and exits non-zero rather than reaching a client as a
    traceback: stdout is the protocol channel here, and anything printed to it that is not a
    message is a parse error at the other end.
    """
    try:
        asyncio.run(_run())
    except ManiculeError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - the module's whole purpose
    main()
