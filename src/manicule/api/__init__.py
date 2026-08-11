"""The HTTP API: the third surface over :class:`~manicule.app.service.ApplicationService`.

Like the command line and the MCP server, it decides nothing. A route parses a request, calls
one service method, and renders the envelope that comes back — the **same** envelope the other
two emit, built by the same function over the same payload models, so a consumer that can read
one surface has read all three.

What is genuinely this surface's own is here rather than in a route: who is calling
(:mod:`manicule.api.security`), whose address that is when a proxy is in front
(:mod:`manicule.api.proxy`), how an outcome becomes a status code
(:mod:`manicule.api.envelopes`), and how an answer stream becomes frames
(:mod:`manicule.api.streaming`). Each of those is a rule that has to hold for every route, and
a rule applied per route is a rule missing from the route somebody added last.

Importing this package pulls in FastAPI. Nothing in :mod:`manicule.app` imports it, which is
what keeps ``import manicule`` free of a web framework — ``tests/test_import_boundary.py``
fails the build otherwise.
"""

from __future__ import annotations

from manicule.api.app import ROUTE_GROUPS, build_app
from manicule.api.proxy import ProxyPolicy
from manicule.api.security import Principal
from manicule.api.serve import address_for, application, serve

__all__ = [
    "ROUTE_GROUPS",
    "Principal",
    "ProxyPolicy",
    "address_for",
    "application",
    "build_app",
    "serve",
]
