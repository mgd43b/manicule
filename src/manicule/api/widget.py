"""The embeddable chat widget: a script another site includes.

This is the only thing manicule serves that is intended to run **inside somebody else's page**,
so the rules it is written under are different from every other route's.

**Nothing from a request is ever interpolated into what is served.** The script below is a
module-level constant. There is no template, no format string and no request value anywhere in
the response, so there is no reflected-injection path into it — not one that is escaped
correctly, one that does not exist.

**The widget builds DOM, never markup.** Every piece of answer text, every citation label and
every error message reaches the page through ``textContent``. ``innerHTML`` does not appear in
the script, and the one test that matters asserts that. An answer is model output over indexed
documents; treating it as markup would make any document in the corpus a script-injection
vector into every page that embeds the widget.

**It renders in a shadow root.** The host page's CSS cannot reach in and the widget's cannot
leak out, so embedding it cannot silently restyle somebody's site into something it is not.

**The key it carries is as public as the page.** A widget on a public page has to authenticate
from the browser, and anything the browser holds, a reader holds. That is stated here rather
than hidden: mint a **dedicated** key for a widget, give it the least role that works, and
revoke it on its own when the page changes. It is not a secret and manicule does not pretend
otherwise — the widget keeps it in a closure and never writes it to storage, which stops it
leaking further than the page it is already in, and nothing more.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse

WIDGET_SCRIPT = r"""/* manicule chat widget. Builds DOM; never markup. */
(function () {
  "use strict";
  var self = document.currentScript;
  if (!self) { return; }
  var endpoint = self.getAttribute("data-endpoint") || "";
  var key = self.getAttribute("data-key") || "";
  var title = self.getAttribute("data-title") || "Ask the corpus";
  var mount = document.createElement("div");
  mount.className = "manicule-widget";
  var root = mount.attachShadow({ mode: "closed" });

  var style = document.createElement("style");
  style.textContent = [
    ":host{all:initial;font-family:system-ui,-apple-system,sans-serif;}",
    ".box{border:1px solid #d0d0d0;border-radius:8px;padding:12px;max-width:36rem;}",
    ".title{font-weight:600;margin-bottom:8px;}",
    ".log{max-height:20rem;overflow-y:auto;margin-bottom:8px;}",
    ".turn{margin-bottom:10px;white-space:pre-wrap;line-height:1.45;}",
    ".role{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;opacity:.6;}",
    ".cite{font-size:.8rem;opacity:.75;margin-top:4px;}",
    ".row{display:flex;gap:6px;}",
    "input{flex:1;padding:6px 8px;border:1px solid #c0c0c0;border-radius:6px;font:inherit;}",
    "button{padding:6px 12px;border-radius:6px;border:1px solid #c0c0c0;font:inherit;}",
    ".error{color:#a12; font-size:.85rem;}"
  ].join("");
  root.appendChild(style);

  var box = document.createElement("div");
  box.className = "box";
  var heading = document.createElement("div");
  heading.className = "title";
  /* textContent, so a title attribute cannot introduce markup into the host page. */
  heading.textContent = title;
  var log = document.createElement("div");
  log.className = "log";
  var row = document.createElement("div");
  row.className = "row";
  var input = document.createElement("input");
  input.type = "text";
  input.setAttribute("aria-label", title);
  var send = document.createElement("button");
  send.type = "button";
  send.textContent = "Ask";
  row.appendChild(input);
  row.appendChild(send);
  box.appendChild(heading);
  box.appendChild(log);
  box.appendChild(row);
  root.appendChild(box);
  self.parentNode.insertBefore(mount, self.nextSibling);

  function turn(role) {
    var wrap = document.createElement("div");
    wrap.className = "turn";
    var label = document.createElement("div");
    label.className = "role";
    label.textContent = role;
    var body = document.createElement("div");
    wrap.appendChild(label);
    wrap.appendChild(body);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return body;
  }

  function cite(target, citation) {
    var line = document.createElement("div");
    line.className = "cite";
    var trail = (citation.heading_path || []).join(" > ");
    line.textContent =
      "[" + citation.slot + "] " + (citation.title || "") + (trail ? " - " + trail : "");
    target.parentNode.appendChild(line);
  }

  function fail(message) {
    var line = document.createElement("div");
    line.className = "error";
    line.textContent = message;
    log.appendChild(line);
  }

  function ask() {
    var question = input.value.trim();
    if (!question) { return; }
    input.value = "";
    turn("you").textContent = question;
    var body = turn("manicule");
    var headers = { "Content-Type": "application/json" };
    if (key) { headers["X-API-Key"] = key; }
    fetch(endpoint + "/api/v1/chat/stream", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ question: question })
    }).then(function (response) {
      if (!response.ok || !response.body) {
        fail("The request was refused (" + response.status + ").");
        return;
      }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) { return; }
          buffer += decoder.decode(chunk.value, { stream: true });
          var frames = buffer.split("\n\n");
          buffer = frames.pop();
          frames.forEach(function (raw) {
            var name = "";
            var data = "";
            raw.split("\n").forEach(function (line) {
              if (line.indexOf("event: ") === 0) { name = line.slice(7); }
              if (line.indexOf("data: ") === 0) { data = line.slice(6); }
            });
            if (!data) { return; }
            var parsed;
            try { parsed = JSON.parse(data); } catch (error) { return; }
            if (name === "delta") {
              body.textContent += parsed.text || "";
            } else if (name === "citation" && parsed.citation) {
              cite(body, parsed.citation);
            } else if (name === "final" && parsed.ok === false) {
              fail((parsed.error && parsed.error.message) || "Something went wrong.");
            }
            log.scrollTop = log.scrollHeight;
          });
          return pump();
        });
      }
      return pump();
    }).catch(function () {
      fail("The corpus could not be reached.");
    });
  }

  send.addEventListener("click", ask);
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter") { ask(); }
  });
})();
"""
"""The widget, as a constant.

A constant rather than a template, and that is the security property: there is no request
value anywhere in this string, so no request can put anything into what a third-party page
executes.
"""

DEMO_PAGE = """<main style="font-family: system-ui, sans-serif; margin: 2rem; max-width: 40rem">
<h1>manicule chat widget</h1>
<p>This page embeds the widget the same way another site would. It is served from this
installation's own origin, so it works without any cross-origin configuration; a real
embedding needs the embedding site's origin listed in
<code>security.transport.allowed_origins</code>, and the page's own origin listed in
<code>security.transport.widget_allowed_domains</code> if it is to be framed.</p>
<script src="/widget/widget.js" data-endpoint="" data-title="Ask the corpus"></script>
</main>
"""
"""A page to look at the widget on. Static, and reflects nothing.

Served from this installation's own origin so it needs no CORS entry, which also makes it a
usable check that the widget works before an operator starts configuring origins.
"""

DEMO_POLICY = (
    "default-src 'none'; script-src 'self'; connect-src 'self'; "
    "style-src 'unsafe-inline'; frame-ancestors 'none'"
)
"""The one page on this surface that is a document rather than data, so the one exception.

Every other response carries ``default-src 'none'``, which is correct for JSON and correct for
the script itself — and **wrong for the page that loads the script**, because a browser applies
it to the document and refuses the ``<script src>``. The demo page therefore states its own
policy: its own origin for the script and for the call it makes, inline styles because the
widget writes a ``<style>`` element into its shadow root, and still no framing.

Narrow on purpose. `script-src 'self'` and no `'unsafe-inline'` for script means this page
cannot execute anything that was not served by this installation, which is the property that
matters on the one route that returns HTML.
"""

router = APIRouter(prefix="/widget", tags=["widget"])


@router.get("/widget.js", name="widget_script", summary="The embeddable chat widget.")
async def widget_script() -> PlainTextResponse:
    """The widget script.

    ``nosniff`` because the response is executed by a browser: without it, a client that
    disagrees about the media type decides for itself what this content is.
    """
    return PlainTextResponse(
        content=WIDGET_SCRIPT,
        media_type="text/javascript; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )


@router.get("", name="widget_demo", summary="A page that embeds the widget, for looking at it.")
async def widget_demo() -> HTMLResponse:
    """A static page. No request value reaches it.

    It sets its **own** ``Content-Security-Policy``, because the application-wide one is
    ``default-src 'none'`` — right for JSON and for the script, and wrong for the document that
    loads the script, which a browser would refuse. The middleware uses ``setdefault``, so a
    route that states a policy keeps it.
    """
    return HTMLResponse(
        content=DEMO_PAGE,
        headers={"Cache-Control": "no-cache", "Content-Security-Policy": DEMO_POLICY},
    )


__all__ = ["DEMO_PAGE", "DEMO_POLICY", "WIDGET_SCRIPT", "router"]
