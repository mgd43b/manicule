"""The MCP surface: nineteen tools over the application service.

Importing this package imports FastMCP. That is fine here and would not be in
:mod:`manicule.core` — the boundary is that *core* carries no implementation dependencies, and
a surface is an implementation.
"""

from __future__ import annotations

from manicule.mcp.server import INSTRUCTIONS, SERVER_NAME, TOOL_NAMES, build_server

__all__ = ["INSTRUCTIONS", "SERVER_NAME", "TOOL_NAMES", "build_server"]
