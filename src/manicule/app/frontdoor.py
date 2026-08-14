"""What answers at ``/``, in each of the three ways this project can be served.

``GET /`` was a 404 on all three. The browser surface is at ``/ui``, the JSON API under
``/api/v1`` and MCP at ``/mcp/``, so an operator who ran ``manicule serve --transport http`` and
opened the address the process had just printed was told there was nothing there — by that
process, at that address. Finding the front door required already knowing the layout, which is
the one thing a front door exists to make unnecessary.

**The wording lives here rather than in either surface, because there are two of them and they
may not import each other.** :mod:`manicule.api.app` builds a FastAPI application;
:mod:`manicule.mcp.serve` must not load FastAPI at all, because the default transport is stdio
and an editor spawning it pays for every import. So the strings are assembled here, out of
nothing but the path constants, and each surface renders them with a response class it already
has.

**Three modes, three answers, and the two that are not a redirect are the point.**

*The whole server* — ``--transport http``. ``/`` redirects to :data:`UI`. The address bar ends
up somewhere real, so the page can be bookmarked, linked and reloaded, which is what a redirect
buys over serving the dashboard at two paths.

*No browser surface* — ``--no-web``. ``/`` says what **is** served and where.
:func:`without_the_browser_surface`. Redirecting here would be worse than the 404 it replaced:
it would send somebody to a second 404 having first told them the thing was somewhere.

*MCP alone* — ``--mcp-only``. Same reasoning, less surface: :func:`mcp_alone` names the one
endpoint this process has. It matters most here, because ``--mcp-only`` is the mode whose
operator is about to paste an address into a client's configuration, and the bare address is
not the one that works.

Both texts end by naming the flag that is switching the browser surface off and what to do
instead, because "there is nothing here" is only half of what somebody at that address needs.
"""

from __future__ import annotations

import textwrap

UI = "/ui"
"""The browser surface's dashboard, which is what ``/`` is a door to."""

API = "/api/v1"
"""The JSON API's prefix. Every versioned route hangs off it."""

DOCS = "/api/docs"
"""The interactive API documentation, served from the same origin off the generated schema."""

MCP = "/mcp"
"""Where MCP answers on a socket — the mount prefix, without the trailing slash.

Stated here rather than in either surface because **both** of them need it and neither may
import the other: :data:`manicule.api.app.MCP_PATH` is this constant, and
:func:`manicule.mcp.serve.serve` passes it to the library rather than accepting a default that
can change in a release nobody read.
"""

MCP_ENDPOINT = f"{MCP}/"
"""The MCP address a client is configured with — trailing slash included.

The mount answers at ``/mcp/`` and Starlette redirects ``/mcp`` to it, so both work in a browser
and only one works in a client that will not follow a redirect on a ``POST``. What gets printed
and what gets served at ``/`` is therefore the one that always works.
"""

TEMPORARY_REDIRECT = 307
"""The status ``/`` redirects with, and **temporary** is the whole of the decision.

A browser caches a permanent redirect and keeps honoring it long after the server stopped
sending it — there is no way to reach out and clear it, so a 301 or a 308 here would be a
layout decision that outlives the layout on every machine that ever visited. This project has
already moved MCP onto this port once; ``/`` pointing at ``/ui`` is a default, not a promise.

307 rather than 302 because 307 is the one that is defined to preserve the method. ``/`` is a
``GET``-only route today, so nothing observes the difference — which is exactly why it costs
nothing to state the unambiguous one now instead of relying on a rewrite that the older statuses
permit.
"""


def to_the_browser_surface(query: str = "") -> str:
    """Where ``/`` sends a browser, carrying whatever was on the way in.

    The query string is forwarded rather than dropped. Nothing the dashboard reads arrives that
    way today, so this is not fixing a broken link — it is refusing to build a door that
    silently discards what somebody typed to get through it.
    """
    return f"{UI}?{query}" if query else UI


def without_the_browser_surface(base: str) -> str:
    """What ``/`` says under ``--no-web``: the API and MCP are here, the pages are not.

    Args:
        base: The origin this request arrived on, with no trailing slash — so the paths below
            come out as addresses a person can copy rather than as fragments they have to
            assemble.
    """
    return _page(
        "manicule is running here. The browser surface is switched off (--no-web), so this "
        "address has no page to show you. These are the surfaces this process is serving:",
        _listing(
            base,
            (("JSON API", API), ("API documentation", DOCS), ("MCP", MCP_ENDPOINT)),
        ),
        "Start the server without --no-web and this address opens the browser surface.",
    )


def mcp_alone(base: str) -> str:
    """What ``/`` says under ``--mcp-only``: one endpoint, and it is not this one.

    Args:
        base: The origin this request arrived on, with no trailing slash.
    """
    return _page(
        "manicule is running here, serving MCP and nothing else (--mcp-only). There is no "
        "browser surface and no JSON API on this port.",
        _listing(base, (("MCP", MCP_ENDPOINT),)),
        "Start the server without --mcp-only and this address opens the browser surface.",
    )


def _listing(base: str, entries: tuple[tuple[str, str], ...]) -> str:
    """The surfaces, one per line, labels padded so the addresses line up.

    Padded from the labels rather than to a number written here, so renaming one or adding
    another keeps the column and nobody has to notice.
    """
    width = max(len(label) for label, _ in entries)
    return "\n".join(f"  {label.ljust(width)}  {base}{path}" for label, path in entries)


def _page(opening: str, listing: str, closing: str) -> str:
    """The three parts, wrapped to something a terminal and a browser both read.

    Plain text rather than HTML, and that is a decision rather than laziness: this is served on
    the two modes that have **no** browser surface, so a page needing a stylesheet would be
    asking for one from a surface that was switched off. Text renders identically in a browser,
    in ``curl`` and in a log.
    """
    return f"{_wrap(opening)}\n\n{listing}\n\n{_wrap(closing)}\n"


def _wrap(text: str, width: int = 78) -> str:
    """Hard-wrap a paragraph, because ``text/plain`` is not reflowed by anything reading it."""
    return textwrap.fill(text, width=width)


__all__ = [
    "API",
    "DOCS",
    "MCP",
    "MCP_ENDPOINT",
    "TEMPORARY_REDIRECT",
    "UI",
    "mcp_alone",
    "to_the_browser_surface",
    "without_the_browser_surface",
]
