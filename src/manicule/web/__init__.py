"""The browser surface: twelve areas of server-rendered HTML over the one application service.

A fourth adapter, under exactly the rules the other three are written to. It parses a request,
calls one service method through :func:`~manicule.app.dispatch.run_op`, and renders the
**envelope** every other surface returns. Nothing here decides anything: no filter is computed,
no policy is interpreted, no second opinion about what a document listing is exists in a
template.

Three decisions shape the whole package, and each is stated where it is made:

**It renders server-side, against the service, rather than consuming its own HTTP API.**
:mod:`manicule.web.rendering` says why at length. The short form: a process that called itself
over HTTP would need a credential to talk to itself, would serialize and re-parse every payload
for nothing, and would put a second network hop inside a page load.

**Escaping is the load-bearing property, so it is unconditional.** The Jinja environment is
built with ``autoescape=True`` for every template, whatever its name, because what this surface
renders is model output over documents somebody else wrote — a document title, a heading path,
an answer body and a citation label are all attacker-influenced text arriving at HTML.

**It adds no operation.** Every page reads through a service method that already has a route on
the HTTP API. There is no upload, no configuration write, no connector creation and no
plugin install here, because there is none there — and this surface is reachable by the same
browser that would be pointed at a hostile page a moment earlier.
"""

from __future__ import annotations

from manicule.web.areas import AREAS, NAVIGATION
from manicule.web.pages import router
from manicule.web.rendering import UI_POLICY
from manicule.web.security import PageRefusedError, refused_page

__all__ = ["AREAS", "NAVIGATION", "UI_POLICY", "PageRefusedError", "refused_page", "router"]
