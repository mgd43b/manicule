"""The process ``tests/mcp/test_stdio.py`` spawns, so that stdio is a pipe and not a metaphor.

Runs :func:`manicule.mcp.serve.serve` — the production entry point, over the production
``build_server`` — against the ``Engineering Architecture`` fixture instead of a real data
directory. What that swaps out is storage; what it keeps is everything the smoke test is about:
a separate process, real JSON-RPC framing over real pipes, the real tool registry, and the real
envelope on the way back.

Storage is the fake because the alternative is worse rather than because it is easier. A child
holding a real data directory would need an embedder, a migrated database and a corpus to index
before it could answer a scoped search — which is a test of installation, running under a
`slow` marker, that fails for a dozen reasons having nothing to do with the protocol.

**Nothing may be written to stdout but the protocol.** A stray ``print`` here is a parse error
at the other end, which is the whole reason ``serve`` passes ``show_banner=False``.
"""

from __future__ import annotations

import asyncio

from manicule.mcp.serve import serve
from tests.mcp.qualification import build_fixture


async def _run() -> None:
    service, _ = await build_fixture()
    await serve(service, transport="stdio")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover - measured in the child, not the parent
    main()
