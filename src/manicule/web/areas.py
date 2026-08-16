"""The application areas, named as data.

In a module of their own because both the pages and the frame that renders them need the list,
and a constant defined in one and imported by the other is how a circular import starts. It is
the same reason ``manicule.api.app.ROUTE_GROUPS`` exists: "this surface offers exactly these" is
a test rather than a count somebody keeps in their head.
"""

from __future__ import annotations

AREAS: tuple[str, ...] = (
    "dashboard",
    "chat",
    "documents",
    "collections",
    "connectors",
    "health",
    "plugins",
    "settings",
    "workspaces",
    "admin",
    "reembed",
    "lifecycle",
    "auth",
    "layout",
)
"""All areas. ``layout`` is the one that is not a page.

It is the frame every other area is rendered inside, and ``tests/web/test_pages.py`` asserts it
by checking that every template extends it — not by looking for a route that does not exist.
"""

NAVIGATION: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "/ui", "Dashboard"),
    ("chat", "/ui/chat", "Chat"),
    ("documents", "/ui/documents", "Documents"),
    ("collections", "/ui/collections", "Collections"),
    ("connectors", "/ui/connectors", "Connectors"),
    ("plugins", "/ui/plugins", "Plugins"),
    ("workspaces", "/ui/workspaces", "Workspaces"),
    ("health", "/ui/health", "Health"),
    ("admin", "/ui/admin", "Admin"),
    ("reembed", "/ui/reembed", "Re-embed"),
    ("lifecycle", "/ui/lifecycle", "Lifecycle"),
    ("auth", "/ui/auth", "API keys"),
    ("settings", "/ui/settings", "Settings"),
)
"""Area, path and label, for the navigation and for the command palette.

One list. The palette reads the links the frame rendered from this, out of the DOM, so the
keyboard route to a page and the clicked route to it cannot come apart — and the palette needs
no list of its own to fall out of date.
"""


__all__ = ["AREAS", "NAVIGATION"]
