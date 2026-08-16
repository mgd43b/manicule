"""Twelve areas, each a page that runs operations and renders their envelopes.

Every handler here has the same three lines: admit the reader, run one or more operations
through :func:`~manicule.web.rendering.panel`, render. There is no fourth line, and a page that
grew one would be behavior living in a surface — the thing ``docs/surfaces.md`` §1 says this
layer may not contain.

## What this surface does not offer, and why the checklist has two holes in it

**There is no upload.** [#12](https://github.com/mgd43b/manicule/issues/12) asks for
drag-and-drop; [#11](https://github.com/mgd43b/manicule/issues/11) decided, and tested by name,
that ``POST /api/v1/documents/upload`` does not exist — accepting bytes over HTTP and writing
them into the corpus is an ingest path with no filesystem permission check and no path the
operator chose. Adding one here to satisfy a checklist would undo that decision quietly, from
a different package, and adding a "browse the server's filesystem and index this path" route
instead would be worse: it turns a browser into a reader of every file the process can open.
So the documents area is complete without an ingest verb, and says on the page that documents
arrive through ``manicule index <path>`` or a configured connector.

**Settings are read-only.** ``config_get`` and ``config_set`` have no HTTP route for the same
reason, and this package calls neither — ``tests/web/test_boundaries.py`` reads the source tree
and fails if it ever does. What the settings area shows instead is the installation's *posture*
as the diagnostics already report it: what ``doctor`` checked, what the index committed to, and
where the data directory is. Changing any of it is a command at a terminal, which is where a
change that can repoint an installation at a different data directory belongs.

## Roles

Each page asks for the floor the routes behind it ask for. Where an area spans both — plugins,
whose listing is a viewer's and whose *health* is an admin's — the page takes the higher floor
rather than rendering a different page per role. A template with a role branch in it is a
policy decision in a template.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

# Imported at run time rather than under `TYPE_CHECKING`: FastAPI resolves a handler's return
# annotation to build the schema, and a forward reference it cannot resolve is a failure at
# `openapi()` rather than at import — which is to say, in the browser rather than in the build.
from fastapi.responses import HTMLResponse, PlainTextResponse

from manicule.api.context import Service
from manicule.web.areas import AREAS, NAVIGATION
from manicule.web.rendering import SCRIPT, STYLESHEET, panel, render
from manicule.web.security import Guest, Operator, Reader

router = APIRouter(prefix="/ui", tags=["web"])

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


# --- the frame's own assets -------------------------------------------------------------------


@router.get("/static/manicule.css", name="ui_stylesheet", summary="The stylesheet.")
async def stylesheet() -> PlainTextResponse:
    """A constant, served with no path parameter anywhere near it.

    Two files are served rather than one directory. A static mount takes a path from the
    request and resolves it against a directory, which is a traversal question somebody has to
    keep getting right; two constants are not that question at all.
    """
    return PlainTextResponse(
        content=STYLESHEET,
        media_type="text/css; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )


@router.get("/static/manicule.js", name="ui_script", summary="The progressive-enhancement script.")
async def script() -> PlainTextResponse:
    """The script, as a constant, for the reason the widget's is one.

    No request value is interpolated into what a browser executes, so there is no reflected
    injection path into it — not one that is escaped correctly, one that does not exist. What
    it does is streaming, the command palette, the theme toggle and the buttons that call the
    JSON API; it renders every piece of text through ``textContent``.
    """
    return PlainTextResponse(
        content=SCRIPT,
        media_type="text/javascript; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )


# --- dashboard --------------------------------------------------------------------------------


@router.get("", name="ui_dashboard", summary="What this installation holds, at a glance.")
async def dashboard(service: Service, caller: Reader) -> HTMLResponse:
    """Counts, the diagnosis, and the workspaces — the three a viewer may read.

    Deliberately not ``index_status``: the fingerprints and the data directory are an admin's,
    on the API and therefore here. The settings area shows them.
    """
    return render(
        "dashboard.html",
        area="dashboard",
        title="Dashboard",
        service=service,
        caller=caller,
        panels={
            "stats": await panel("stats", service, service.stats),
            "doctor": await panel("doctor", service, service.doctor),
            "workspaces": await panel("workspace_list", service, service.workspace_list),
        },
    )


# --- chat -------------------------------------------------------------------------------------


@router.get("/chat", name="ui_chat", summary="Ask the corpus, and the conversations so far.")
async def chat(service: Service, caller: Reader, *, limit: Limit = 50) -> HTMLResponse:
    """The asking surface. The answer itself arrives over SSE from the chat route.

    A member is what ``POST /api/v1/chat`` requires, and this page requires a viewer, because
    the page is a read: it lists conversations. Asking from it is the API's refusal to make,
    and a page that pre-empted it would be a second authorization decision.
    """
    return render(
        "chat.html",
        area="chat",
        title="Chat",
        service=service,
        caller=caller,
        panels={
            "conversations": await panel(
                "conversation_list", service, lambda: service.conversation_list(limit=limit)
            )
        },
        extra={"conversation_id": "", "turns": ()},
    )


@router.get(
    "/chat/{conversation_id}", name="ui_conversation", summary="One conversation, with its turns."
)
async def conversation(
    service: Service, caller: Reader, conversation_id: str, *, limit: Limit = 50
) -> HTMLResponse:
    """The owner's view: full citations, because the reader could have retrieved them.

    The anonymous view of the same conversation is a different page over a different operation
    returning a different type — see :func:`shared`.
    """
    messages = await panel(
        "conversation_messages",
        service,
        lambda: service.conversation_messages(conversation_id, limit=limit),
    )
    return render(
        "chat.html",
        area="chat",
        title="Chat",
        service=service,
        caller=caller,
        panels={
            "conversations": await panel(
                "conversation_list", service, lambda: service.conversation_list(limit=limit)
            ),
            "messages": messages,
        },
        primary="messages",
        extra={"conversation_id": conversation_id, "turns": messages.data.get("turns") or ()},
    )


# --- documents --------------------------------------------------------------------------------


@router.get("/documents", name="ui_documents", summary="A page of this workspace's documents.")
async def documents(
    service: Service,
    caller: Reader,
    *,
    limit: Limit = 50,
    offset: Offset = 0,
    source: Annotated[str | None, Query()] = None,
    media_type: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    """Newest first, scoped to this workspace and checked again on the way out."""
    listing = await panel(
        "document_list",
        service,
        lambda: service.document_list(
            limit=limit, offset=offset, source=source, media_type=media_type
        ),
    )
    return render(
        "documents.html",
        area="documents",
        title="Documents",
        service=service,
        caller=caller,
        panels={"documents": listing},
        primary="documents",
        extra={
            "source": source or "",
            "media_type": media_type or "",
            "offset": offset,
            "limit": limit,
        },
    )


@router.get("/documents/trash", name="ui_trash", summary="What is in the trash, and restoring it.")
async def trash(
    service: Service, caller: Reader, *, limit: Limit = 50, offset: Offset = 0
) -> HTMLResponse:
    """Longest-deleted first — the order the sweep will take them in.

    Declared **above** ``/documents/{document_id}``: Starlette matches in declaration order, so
    the parameterized route would otherwise swallow ``trash`` as an id.
    """
    entries = await panel(
        "document_trash", service, lambda: service.document_trash(limit=limit, offset=offset)
    )
    return render(
        "trash.html",
        area="documents",
        title="Trash",
        service=service,
        caller=caller,
        panels={"trash": entries},
        primary="trash",
    )


@router.get(
    "/documents/{document_id}", name="ui_document", summary="One document, as it was chunked."
)
async def document(service: Service, caller: Reader, document_id: str) -> HTMLResponse:
    """The workbench view: the document and the blocks retrieval actually sees.

    One operation rather than two, because ``workbench`` already returns the document summary
    alongside its blocks — and a page that called ``document_get`` *and* ``workbench`` would
    read the corpus twice to render one thing.
    """
    return render(
        "document.html",
        area="documents",
        title="Document",
        service=service,
        caller=caller,
        panels={
            "workbench": await panel("workbench", service, lambda: service.workbench(document_id))
        },
        primary="workbench",
        extra={"document_id": document_id},
    )


@router.get("/search", name="ui_search", summary="Rank passages without asking a model anything.")
async def search(
    service: Service,
    caller: Reader,
    *,
    q: Annotated[str, Query()] = "",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    profile: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    """The cheap half of ``ask``: ranked passages, each with the score every stage gave it.

    **An absent or blank query is a page, not an error.** The frame's search box carries no
    ``required`` attribute and submits on Enter, so ``GET /ui/search?q=`` is one keystroke away
    from every page on this surface; and ``/ui/search`` with no query at all is what a bookmark
    or a typed URL produces. Rejecting either at validation returned the raw JSON envelope —
    with a traceback frame naming this file's path on the server — into a browser window, which
    is both a dead end for the reader and an unnecessary disclosure. Blank renders the form with
    nothing under it instead, which is the state the page is *for* before a query is typed.
    """
    if not q.strip():
        return render(
            "search.html",
            area="documents",
            title="Search",
            service=service,
            caller=caller,
            panels={},
            extra={"query": "", "profile": profile or ""},
        )
    return render(
        "search.html",
        area="documents",
        title="Search",
        service=service,
        caller=caller,
        panels={
            "search": await panel(
                "search", service, lambda: service.search(q, limit=limit, profile=profile)
            )
        },
        primary="search",
        extra={"query": q, "profile": profile or ""},
    )


# --- collections ------------------------------------------------------------------------------


@router.get("/collections", name="ui_collections", summary="Collections and tags.")
async def collections(service: Service, caller: Reader) -> HTMLResponse:
    """The two ways a person groups a corpus, side by side because they are one decision."""
    return render(
        "collections.html",
        area="collections",
        title="Collections",
        service=service,
        caller=caller,
        panels={
            "collections": await panel("collection_list", service, service.collection_list),
            "tags": await panel("tag_list", service, service.tag_list),
        },
    )


# --- connectors -------------------------------------------------------------------------------


@router.get("/connectors", name="ui_connectors", summary="Configured sources and their sync state.")
async def connectors(service: Service, caller: Operator) -> HTMLResponse:
    """An admin's page, because ``GET /api/v1/admin/connectors`` is an admin's route.

    A connector names a system this installation reaches into and holds credentials for it.
    There is no page that creates one: sources are declared in configuration, where the whole
    set is reviewable in one place.
    """
    return render(
        "connectors.html",
        area="connectors",
        title="Connectors",
        service=service,
        caller=caller,
        panels={"connectors": await panel("connector_list", service, service.connector_list)},
    )


# --- health -----------------------------------------------------------------------------------


@router.get("/health", name="ui_health", summary="Everything doctor checks.")
async def health(service: Service, caller: Reader) -> HTMLResponse:
    """``doctor``, rendered. It builds nothing expensive, which is why it is safe here."""
    return render(
        "health.html",
        area="health",
        title="Health",
        service=service,
        caller=caller,
        panels={"doctor": await panel("doctor", service, service.doctor)},
        primary="doctor",
    )


# --- plugins ----------------------------------------------------------------------------------


@router.get("/plugins", name="ui_plugins", summary="Installed plugins and their health.")
async def plugins(service: Service, caller: Operator) -> HTMLResponse:
    """Both panels, so the page takes the higher of the two floors.

    ``plugin_list`` is a viewer's on the API and ``plugin_health`` is an admin's. One page
    showing both asks for an admin rather than rendering a different page per role — a
    template with a role branch in it is a policy decision in a template.

    Nothing here installs anything. ``manicule`` never installs a plugin: one is imported into
    this process and runs with everything it has.
    """
    return render(
        "plugins.html",
        area="plugins",
        title="Plugins",
        service=service,
        caller=caller,
        panels={
            "plugins": await panel(
                "plugin_list", service, lambda: service.plugin_list(registry=False)
            ),
            "health": await panel("plugin_health", service, service.plugin_health),
        },
    )


# --- workspaces -------------------------------------------------------------------------------


@router.get("/workspaces", name="ui_workspaces", summary="Workspaces, with the active one marked.")
async def workspaces(service: Service, caller: Reader) -> HTMLResponse:
    """One person with several corpora, not several people.

    Counts are the active workspace's only: this process is scoped to one tenant and cannot
    read another's rows, including to count them. Switching is a command-line operation and
    has no route on any network surface, so there is no button here that pretends otherwise.
    """
    return render(
        "workspaces.html",
        area="workspaces",
        title="Workspaces",
        service=service,
        caller=caller,
        panels={"workspaces": await panel("workspace_list", service, service.workspace_list)},
        primary="workspaces",
    )


# --- admin ------------------------------------------------------------------------------------


@router.get("/admin", name="ui_admin", summary="Index state, search quality, telemetry and audit.")
async def admin(
    service: Service, caller: Operator, *, limit: Limit = 25, offset: Offset = 0
) -> HTMLResponse:
    """Everything the admin group already computes, rendered once.

    No number on this page is calculated here. Search quality in particular is the evaluation
    harness's own report, verbatim, caveats included: an endpoint that computed a second score
    would produce a number nobody could reconcile with the first.
    """
    return render(
        "admin.html",
        area="admin",
        title="Admin",
        service=service,
        caller=caller,
        panels={
            "index": await panel("index_status", service, service.index_status),
            "quality": await panel("search_quality", service, service.search_quality),
            "queries": await panel(
                "query_logs", service, lambda: service.query_logs(limit=limit, offset=offset)
            ),
            "audit": await panel(
                "audit_log", service, lambda: service.audit_log(limit=limit, offset=offset)
            ),
            "plugins": await panel("plugin_health", service, service.plugin_health),
            "connectors": await panel("connector_list", service, service.connector_list),
        },
        extra={"limit": limit, "offset": offset},
    )


# --- re-embedding -----------------------------------------------------------------------------


async def _reembed_page(
    service: Service,
    caller: Operator,
    *,
    run_id: str = "",
) -> HTMLResponse:
    panels = {"plan": await panel("reembed_plan", service, service.reembed_plan)}
    if run_id:
        panels["status"] = await panel(
            "reembed_status", service, lambda: service.reembed_status(run_id)
        )
    return render(
        "reembed.html",
        area="reembed",
        title="Re-embed",
        service=service,
        caller=caller,
        panels=panels,
        extra={"run_id": run_id},
    )


@router.get("/reembed", name="ui_reembed", summary="Plan and inspect durable re-embedding.")
async def reembed_page(
    service: Service,
    caller: Operator,
    *,
    run_id: Annotated[str, Query(max_length=200)] = "",
) -> HTMLResponse:
    return await _reembed_page(service, caller, run_id=run_id)


# --- source and derived lifecycle --------------------------------------------------------------


@router.get(
    "/lifecycle",
    name="ui_lifecycle",
    summary="Dry-run source and derived lifecycle boundaries.",
)
async def lifecycle_page(
    service: Service,
    caller: Operator,
    *,
    before: Annotated[datetime | None, Query()] = None,
    run_id: Annotated[str, Query(max_length=200)] = "",
) -> HTMLResponse:
    """Read-only lifecycle plans; destructive confirmation stays outside the browser."""
    panels = {
        "reset": await panel(
            "lifecycle_reset_derived",
            service,
            lambda: service.lifecycle_reset_derived(dry_run=True),
        ),
        "cleanup": await panel(
            "lifecycle_cleanup_generations",
            service,
            lambda: service.lifecycle_cleanup_generations(dry_run=True),
        ),
    }
    if before is not None:
        panels["history"] = await panel(
            "lifecycle_release_history",
            service,
            lambda: service.lifecycle_release_history(before, dry_run=True),
        )
    if run_id:
        panels["snapshot"] = await panel(
            "lifecycle_delete_snapshot",
            service,
            lambda: service.lifecycle_delete_snapshot(run_id, dry_run=True),
        )
    return render(
        "lifecycle.html",
        area="lifecycle",
        title="Lifecycle",
        service=service,
        caller=caller,
        panels=panels,
        extra={"before": before.isoformat() if before is not None else "", "run_id": run_id},
    )


# --- settings ---------------------------------------------------------------------------------


@router.get("/settings", name="ui_settings", summary="This installation's posture. Read-only.")
async def settings(service: Service, caller: Operator) -> HTMLResponse:
    """What the installation *is*, from the diagnostics — never from ``config_get``.

    ``config_get`` and ``config_set`` are absent from every network surface on purpose, and
    this page does not reach around that. What it renders is the same two operations an
    operator would run to answer "is this installation set up the way I think": ``doctor``,
    which names the configuration, the transport, the plugins, the storage and the index; and
    ``index_status``, which names the data directory, the schema revision and the fingerprints
    the index has committed to.

    Everything on it is a fact about the running system rather than the contents of a file, and
    changing any of it is ``manicule config set`` at a terminal.
    """
    return render(
        "settings.html",
        area="settings",
        title="Settings",
        service=service,
        caller=caller,
        panels={
            "doctor": await panel("doctor", service, service.doctor),
            "index": await panel("index_status", service, service.index_status),
        },
    )


# --- auth -------------------------------------------------------------------------------------


@router.get("/auth", name="ui_auth", summary="This person's API keys, and how they authenticate.")
async def auth(service: Service, caller: Operator) -> HTMLResponse:
    """One person managing their own keys. There is no user administration here.

    manicule is single-user until it is feature complete; roles, invitations and identity
    providers belong to team mode ([#13](https://github.com/mgd43b/manicule/issues/13)). What
    this page is for is the two things a single operator needs: the keys that exist, and what
    this installation currently demands of a caller.

    Records, never secrets. Only digests are stored, so there is no secret to render — a key's
    one copy is in the response that minted it.
    """
    return render(
        "auth.html",
        area="auth",
        title="API keys",
        service=service,
        caller=caller,
        panels={
            "api_keys": await panel("api_key_list", service, service.api_key_list),
            "providers": await panel("auth_providers", service, service.auth_providers),
        },
        extra={"key_id": caller.identity.key_id, "key_name": caller.identity.key_name},
    )


# --- the one page with no credential ------------------------------------------------------------


@router.get(
    "/shared/{token}", name="ui_shared", summary="Read a shared conversation. No credential."
)
async def shared(service: Service, caller: Guest, token: str) -> HTMLResponse:
    """The anonymous view, through the redaction that already exists.

    ``ApplicationService.shared_conversation`` hashes the token, resolves it in one statement
    with expiry, revocation, soft-delete and the snapshot boundary as predicates of that
    statement, and returns citation **labels** — a title, a breadcrumb the block kind permits,
    a page number where there is one, and whether the claim verified. This page renders that
    and nothing else. There is no second path to conversation data here, which is the whole
    reason the redaction is worth anything.

    A smaller frame, too: an anonymous reader gets no navigation into areas they cannot reach.
    """
    return render(
        "shared.html",
        area="",
        title="Shared conversation",
        service=service,
        caller=caller,
        panels={
            "shared": await panel(
                "shared_conversation", service, lambda: service.shared_conversation(token)
            )
        },
        layout="bare.html",
    )


__all__ = ["AREAS", "NAVIGATION", "router"]
