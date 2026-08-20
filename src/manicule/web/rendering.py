"""Turning an envelope into a page, and the escaping that makes that safe.

## Why this renders against the service rather than against its own HTTP API

The alternative — a browser surface that fetches ``/api/v1/...`` from the same process that is
serving it — was rejected for four reasons, in order of weight:

1. **It would need a credential to talk to itself.** With ``security.auth.mode = api_key``
   every request carries a key, including one this process made to its own socket. There is no
   key it could legitimately hold: minting one at startup is a second credential nobody
   revoked, and skipping authentication for "internal" calls is a bypass with a friendly name.
2. **A second hop inside a page load.** Serializing a payload to JSON, opening a socket to
   ourselves, parsing it back, and then rendering it is work that buys nothing — the same
   objects are already in this process.
3. **Two failure vocabularies.** A connection refused to our own port is not an operation that
   failed, and a page would have to distinguish them.
4. **Parity is easier to *prove* this way.** Every page runs
   :func:`~manicule.app.dispatch.run_op` over one service method, so the page's data is
   literally the envelope the MCP tool and the ``--json`` command produce. ``tests/web/`` asserts
   that equality rather than asserting a resemblance.

What that costs is one thing, and it is the thing this module has to keep paying: rendering
in-process means the HTML is built here, so **HTML escaping is this package's responsibility**
rather than something a JSON boundary did for it.

## Escaping

``autoescape=True``, for every template, unconditionally. Not
:func:`~jinja2.select_autoescape`, which decides by file extension and therefore stops
protecting anything the moment somebody renders a fragment from a string or names a template
``.txt``.

What arrives here is not this project's text. A document title is whatever was in the file that
was indexed; a heading path is whatever the parser found in it; an answer body is a model's
output over those documents, and a citation label is the document's own words traveling back
out under manicule's name. Every one of those is a place where ``<script>`` reaches HTML, and
the corpus is exactly where an attacker who can get a file indexed would put one.

``tests/web/test_escaping.py`` puts markup in all four and proves it renders inert — and then
switches autoescaping off and proves the same test fails, because a guard nobody has watched
fail is a guard nobody has tested.

## Undefined names are errors

``StrictUndefined``: a template naming a field the payload does not have raises rather than
rendering an empty string. A page that silently renders nothing where a number should be is the
exact failure this project keeps finding — green, wrong, and invisible.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from manicule.api.envelopes import OK, status_for
from manicule.app.dispatch import run_op
from manicule.core.errors import ManiculeError
from manicule.core.version import CORE_VERSION
from manicule.web.areas import NAVIGATION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from manicule.api.security import Principal
    from manicule.app.results import Envelope, ErrorInfo, JsonValue, Payload
    from manicule.app.service import ApplicationService

HERE = Path(__file__).parent

TEMPLATE_DIR = HERE / "templates"
"""Where the templates live. Inside the package, so a wheel carries them."""

STATIC_DIR = HERE / "static"
"""The stylesheet and the script, read once at import and served as constants."""

UI_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)
"""The ``Content-Security-Policy`` every page in this surface serves.

The application-wide policy is ``default-src 'none'``, which is right for JSON and wrong for a
document: a browser applies it to the page and then refuses the page's own stylesheet and
script. So this surface states its own, and states it **narrowly**.

Every clause is doing something:

- ``script-src 'self'`` with **no** ``'unsafe-inline'``: the page cannot execute anything this
  installation did not serve. That is what makes the escaping property above worth having —
  with inline script permitted, a successful injection would run.
- ``style-src 'self'``, same reasoning, which is why the stylesheet is a file rather than a
  ``<style>`` block. Inline styles are a real exfiltration channel through selectors.
- ``connect-src 'self'``: the page streams an answer from this origin and may talk to nothing
  else. A page that could ``fetch()`` anywhere is a page that can post the corpus somewhere.
- ``form-action 'self'``: the search box is a GET form; nothing may retarget it.
- ``base-uri 'none'``: an injected ``<base>`` would silently repoint every relative URL on the
  page, including the script's.
- ``frame-ancestors 'none'``: a chat box in an invisible frame is a clickjacking target. The
  widget is the part of manicule that is *meant* to be embedded; this surface is not.
"""

STYLESHEET = (STATIC_DIR / "manicule.css").read_text(encoding="utf-8")
"""The stylesheet, as a constant. No request value can reach what a browser is sent."""

SCRIPT = (STATIC_DIR / "manicule.js").read_text(encoding="utf-8")
"""The script, as a constant, for the same reason the widget's is one."""

STYLESHEET_PATH = "/ui/static/manicule.css"
SCRIPT_PATH = "/ui/static/manicule.js"
"""Named once here and used by the layout, so the routes and the markup cannot drift apart."""


class PayloadEnvironment(Environment):
    """Jinja, with one rule changed: on a mapping, a **field wins over a method**.

    Jinja's ``foo.bar`` tries ``getattr`` first and falls back to ``foo['bar']``. Payloads are
    rendered as the plain dictionaries the envelope carries, so any field sharing a name with a
    ``dict`` method resolves to the method — and the page renders a bound method, or worse,
    treats one as truthy and shows the wrong branch. ``ApiKeyList.keys`` is exactly that field,
    and it is not a name anybody would think to avoid.

    So for a mapping the item is tried first and there is no fallback to the attribute: a
    payload has no methods worth reaching from a template, and reaching one is always the bug
    rather than the intent. Absent names still become ``StrictUndefined`` and still raise.

    The consequence is that ``{{ table.items() }}`` no longer works, which is why the templates
    use Jinja's ``| items`` filter. That is the correct direction: a template asking a *payload*
    for its methods is the thing this override exists to prevent.
    """

    @override
    def getattr(self, obj: object, attribute: str) -> object:
        if isinstance(obj, dict):
            values = cast("dict[str, object]", obj)
            if attribute in values:
                return values[attribute]
            return self.undefined(obj=obj, name=attribute)
        return super().getattr(obj, attribute)


def build_environment() -> Environment:
    """The template environment: autoescaping, strict, and with no globals of its own.

    No global functions and no filters are installed. A template that could call into the
    application would be a template that can decide something, and the whole claim of this
    package is that it decides nothing.
    """
    return PayloadEnvironment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        # Unconditional, and not by file extension. See this module's docstring.
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        auto_reload=False,
    )


ENVIRONMENT = build_environment()
"""One environment for the process. Templates are packaged data and never change at run time."""


@dataclass(frozen=True, slots=True)
class Panel:
    """One operation's result, as a page reads it.

    A page is usually several operations — a dashboard is counts *and* an index state *and* a
    diagnosis — and they fail independently. Wrapping each one keeps that honest: a panel whose
    operation failed renders its own error where it sits, and the rest of the page still says
    what it knows. A page that collapsed to a single error because one of five reads failed
    would be hiding four answers it had.
    """

    envelope: Envelope

    @property
    def op(self) -> str:
        return self.envelope.op

    @property
    def ok(self) -> bool:
        return self.envelope.ok

    @property
    def error(self) -> ErrorInfo | None:
        return self.envelope.error

    @property
    def data(self) -> Mapping[str, JsonValue]:
        """The payload, or an empty mapping on a failure.

        Empty rather than ``None`` so a template can iterate without a null check, and never a
        stand-in value: a panel that failed says so through :attr:`ok`, and a page that read
        :attr:`data` without checking gets nothing rather than something invented.
        """
        return self.envelope.data or {}


async def panel(
    op: str, service: ApplicationService, call: Callable[[], Awaitable[Payload]]
) -> Panel:
    """Run one service operation and wrap its envelope.

    The same :func:`~manicule.app.dispatch.run_op` the command line, the MCP server and the
    HTTP routes use. That is the whole of the parity claim: the bytes a page renders come from
    the object the other three surfaces serialize.
    """
    return Panel(envelope=await run_op(op, service.workspace, call))


type PanelCall = Callable[[], Awaitable[Payload]]
"""One page panel's operation, ready to run: the service call with its arguments already bound."""


async def panels(
    service: ApplicationService, requested: Mapping[str, tuple[str, PanelCall]]
) -> dict[str, Panel]:
    """Run every panel a page is made of at once, and return them by name.

    A page is several independent operations, and awaiting them in a dict literal ran them in
    the order somebody happened to write them: the counts, then a diagnosis that leaves the
    machine, then the workspaces — each waiting on the one above it for no reason other than
    the shape of the source. One slow panel therefore delayed panels that did not depend on
    it, which is the part a reader notices, because the page arrives at the speed of its worst
    operation plus all the others.

    Nothing here decides what a page shows. A failed panel is still that panel's failure, in
    the same envelope :func:`panel` builds, and the returned order is the requested one so a
    template does not reorder itself when the network does.
    """
    ordered = list(requested)
    started = [panel(op, service, call) for op, call in (requested[name] for name in ordered)]
    return dict(zip(ordered, await asyncio.gather(*started), strict=True))


def render(
    template: str,
    *,
    area: str,
    title: str,
    service: ApplicationService,
    caller: Principal,
    panels: Mapping[str, Panel],
    primary: str | None = None,
    extra: Mapping[str, Any] | None = None,
    layout: str = "layout.html",
) -> HTMLResponse:
    """Render one page, or the failure of the operation the page is *about*.

    Args:
        template: The template file, relative to :data:`TEMPLATE_DIR`.
        area: Which of :data:`~manicule.web.pages.AREAS` this page is, for the navigation.
        title: The page's own title.
        service: The service, for the workspace name on the frame.
        caller: Who is reading, so the frame can say so and hide what they cannot reach.
        panels: The operations this page ran, by the name its template knows them as.
        primary: The panel this page *is*. When that one failed there is no page to render —
            a document detail with no document is not a page with an empty section — so the
            response becomes the problem page, with the status the error's type implies.
        extra: Page-specific context. It may not name a key the frame supplies, and a
            collision is **refused** rather than resolved: silently winning would let a page
            put its own value where the workspace name or the reader's role goes, and silently
            losing would make a page render without the value it passed. Both are the kind of
            wrong that looks right.
        layout: The frame to extend. The shared-conversation page passes a smaller one,
            because an anonymous reader has no business being shown a navigation full of
            links they cannot follow.

    Raises:
        ManiculeError: ``extra`` names a key the frame already supplies.
    """
    failed = panels.get(primary) if primary is not None else None
    frame: dict[str, Any] = {
        "area": area,
        "title": title,
        "navigation": NAVIGATION,
        "panels": panels,
        "workspace": service.workspace,
        "version": CORE_VERSION,
        "role": caller.role.value,
        "auth_mode": caller.identity.mode,
        "authenticated": caller.identity.authenticated,
        "stylesheet": STYLESHEET_PATH,
        "script": SCRIPT_PATH,
        "layout": layout,
    }
    clash = sorted(set(extra or {}) & set(frame))
    if clash:
        msg = (
            f"the page context {clash} would overwrite the frame's own. Rename the page's key: "
            f"a template reading 'workspace' or 'role' must get the one this request resolved."
        )
        raise ManiculeError(msg)
    context: dict[str, Any] = {**frame, **(extra or {})}
    if failed is not None and not failed.ok:
        return html_response(
            ENVIRONMENT.get_template("problem.html").render(
                {**context, "problem": failed, "title": "Refused"}
            ),
            status=status_for(failed.envelope),
        )
    return html_response(ENVIRONMENT.get_template(template).render(context), status=OK)


def html_response(body: str, *, status: int) -> HTMLResponse:
    """One HTML response, with this surface's own policy and no caching.

    ``no-store`` matches the application-wide header and is repeated here for a reason a page
    has and JSON does not: a rendered page holds answer text and document titles in the
    browser's back-forward cache, and a shared machine is exactly where somebody presses Back.
    """
    return HTMLResponse(
        content=body,
        status_code=status,
        headers={"Content-Security-Policy": UI_POLICY, "Cache-Control": "no-store"},
    )


__all__ = [
    "ENVIRONMENT",
    "SCRIPT",
    "SCRIPT_PATH",
    "STATIC_DIR",
    "STYLESHEET",
    "STYLESHEET_PATH",
    "TEMPLATE_DIR",
    "UI_POLICY",
    "Panel",
    "build_environment",
    "html_response",
    "panel",
    "render",
]
