"""Plugins: listing them, browsing the registry, and the one verb that is not here.

**manicule never installs a plugin, and this surface is why the rule is written down.** A
plugin is imported into this process and runs with everything it has — the database, the
credentials, the network. Installing one means fetching and executing code. A route that did
that would be a remote-code-execution endpoint reachable by anything holding an admin key,
which is a category of thing an unattended caller should not be able to reach at all.

So ``POST /api/v1/plugins/install`` is deliberately absent, and ``POST /api/v1/plugins/{name}``
*enables* one that is already installed. A plugin that is not installed comes back with the
command that would install it and nothing is run.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.security import AdminPrincipal, ViewerPrincipal

router = APIRouter(prefix="/api/v1/plugins", tags=["plugins"])


@router.get("", name="plugin_list", summary="Installed plugins and the components each registers.")
async def list_plugins(service: Service, caller: ViewerPrincipal) -> Response:
    """What is installed, and what each plugin contributed to the container."""
    del caller
    return await respond("plugin_list", service, lambda: service.plugin_list(registry=False))


@router.get("/search", name="plugin_list", summary="Browse the community registry.")
async def search_registry(
    service: Service,
    caller: AdminPrincipal,
    q: Annotated[str | None, Query(description="Filter the listing by name or summary.")] = None,
) -> Response:
    """The community listing, when configuration permits consulting it.

    Only fetched while ``plugins.allow_install`` is on. A plugin runs with this process's full
    authority, so browsing a catalog of them is opt-in — and when it is off, the payload says
    so rather than returning an empty list that reads like "none are available".

    The filter is the service's, not this route's. It is one substring match, and that is
    exactly the size of rule that ends up implemented twice and differently.
    """
    del caller
    return await respond(
        "plugin_list", service, lambda: service.plugin_list(registry=True, query=q)
    )


@router.post("/{name}", name="plugin_add", summary="Enable an installed plugin.")
async def enable_plugin(service: Service, caller: AdminPrincipal, name: str) -> Response:
    """Enable a plugin that is **already installed**. This never fetches or runs code.

    It takes effect at the next start, because a plugin registers components into a container
    that is already built.
    """
    del caller
    return await respond("plugin_add", service, lambda: service.plugin_add(name))


@router.delete("/{name}", name="plugin_remove", summary="Disable a plugin.")
async def disable_plugin(service: Service, caller: AdminPrincipal, name: str) -> Response:
    """Disable a plugin. The distribution stays installed and is not touched."""
    del caller
    return await respond("plugin_remove", service, lambda: service.plugin_remove(name))


__all__ = ["router"]
