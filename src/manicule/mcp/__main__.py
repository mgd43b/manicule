"""``python -m manicule.mcp`` — the MCP server over stdio.

Here as well as ``manicule start --mcp-only`` because a client that spawns a server usually
wants to name an interpreter and a module rather than trust a console script to be on the
PATH it happens to have. Same server, same service, same tools.
"""

from __future__ import annotations

import asyncio

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.mcp.serve import serve


async def _run() -> None:
    async with Runtime.open() as runtime:
        await serve(ApplicationService(runtime), transport="stdio")


def main() -> None:
    """Start the server over stdio, which opens no socket."""
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover - the module's whole purpose
    main()
