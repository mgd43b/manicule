# The browser surface: twelve areas, no build step

`manicule.web`. Server-rendered HTML over the same `ApplicationService` the command line, the
MCP server and the HTTP API are adapters over. It is mounted on the same application as the API
— `manicule start --transport http` serves both — at `/ui`.

---

## 1. The stack, and why

**Server-rendered Jinja2 templates, one hand-written stylesheet, one hand-written script, and
nothing else.** No Node, no bundler, no CSS framework, no client-side framework. The only new
dependency is `jinja2`, which is pure Python and arrives with `uv sync` like everything else.

The choice was between that and a single-page application. One constraint decided it, and it is
a requirement rather than a preference:

> **The install path must not gain a build toolchain.** `uv sync` is the whole install, and the
> container image builds and runs with no network at all — its final stage runs `doctor`,
> `init`, `index` and `search` under `--network=none`.

A single-page application fails that in a way that cannot be worked around honestly:

- **Compiled assets have to come from somewhere.** Either the build runs at install time — a
  toolchain in the install path, which is the thing forbidden — or the compiled bundle is
  committed to the repository. A committed bundle is machine-generated code nobody reviews, in
  the one part of this project that renders model output over documents somebody else wrote. It
  is also the part where a reviewer most needs to be able to read what runs.
- **The image would need it too.** Building the assets inside the container adds a Node stage to
  an image that is already 3.4 GB, and reintroduces a network dependency to a build whose whole
  claim is that it has none.
- **A CDN is not an option.** The page's `Content-Security-Policy` is `default-src 'none'` with
  `script-src 'self'`. Loading a framework from someone else's origin means loosening exactly
  the directive that makes the escaping below worth having, and it means the browser surface
  stops working on the air-gapped installation the container exists to support.

That rules out Tailwind as well, and for the same reason rather than a different one. The
standalone binary avoids Node but is still a platform-specific executable run at build time to
produce a file — a build toolchain that happens not to be Node's. The hand-written stylesheet is
245 lines, is reviewable, and is a file this repository owns.

It also rules out fetching a hypermedia library — htmx or anything like it — as a vendored
minified blob. The interactions this surface actually needs are a streaming answer, a command
palette, a theme toggle and a dozen buttons that call the JSON API. That is a few hundred lines
of ordinary JavaScript this project can read, and the embeddable widget was already written
exactly that way for exactly that reason.

**What is given up.** No optimistic UI, no client-side routing, and a full page load after a
mutation. Those are real, and they are the right trade for a surface whose ordinary user is one
person on loopback: a page load against a local process is a few milliseconds, and the streaming
answer — the one interaction where latency is visible — is streamed, over the SSE endpoint the
API already publishes.

**If this is revisited**, the question to ask is not "is a SPA nicer" but "has the no-toolchain
requirement changed". It is the whole of the argument, and it is the one thing a future
implementation has to answer.

---

## 2. Where the behaviour is

Nowhere in this package. Every page does the same three things:

1. admit the reader through `manicule.api.security.require` — the *same* function the JSON
   routes use, not a second implementation;
2. run one or more operations through `manicule.app.dispatch.run_op`, which is what produces the
   envelope every other surface serialises;
3. render that envelope.

**It renders against the service rather than consuming its own HTTP API.** The alternative — a
page that fetches `/api/v1/...` from the process serving it — was rejected because it would need
a credential to talk to itself. With `security.auth.mode = api_key` every request carries a key,
and there is none this process could legitimately hold: minting one at startup is a credential
nobody revoked, and exempting "internal" calls is a bypass with a friendly name. It would also
put a socket round trip inside a page load for no gain, and give a page two failure vocabularies
to distinguish.

`tests/app/test_surface_parity.py` grew a **fourth column** for this rather than a parallel file.
It cannot compare HTML byte for byte with an envelope, so it asserts the claim in the form HTML
can carry: a value the MCP tool reported is found in the page, and a failure the tool reports is
the failure the page shows — same type, same message, same hint.

---

## 3. The twelve areas

| Area | Page | Reads | Floor |
|---|---|---|---|
| dashboard | `/ui` | `stats`, `doctor`, `workspace_list` | viewer |
| chat | `/ui/chat`, `/ui/chat/{id}` | `conversation_list`, `conversation_messages` | viewer |
| documents | `/ui/documents`, `/ui/documents/{id}`, `/ui/documents/trash`, `/ui/search` | `document_list`, `workbench`, `document_trash`, `search` | viewer |
| collections | `/ui/collections` | `collection_list`, `tag_list` | viewer |
| connectors | `/ui/connectors` | `connector_list` | admin |
| health | `/ui/health` | `doctor` | viewer |
| plugins | `/ui/plugins` | `plugin_list`, `plugin_health` | admin |
| settings | `/ui/settings` | `doctor`, `index_status` | admin |
| workspaces | `/ui/workspaces` | `workspace_list` | viewer |
| admin | `/ui/admin` | `index_status`, `search_quality`, `query_logs`, `audit_log`, `plugin_health`, `connector_list` | admin |
| auth | `/ui/auth` | `api_key_list`, `auth_providers` | admin |
| layout | — | — | — |

Plus `GET /ui/shared/{token}`, which takes no credential at all, and two constants:
`/ui/static/manicule.css` and `/ui/static/manicule.js`.

`layout` is the one area that is not a page: it is the frame the other eleven are rendered
inside, and `tests/web/test_pages.py` asserts that by checking every page template extends it.

**Each page asks for the floor its routes ask for.** Where an area spans two — plugins, whose
listing is a viewer's and whose *health* is an admin's — the page takes the higher one rather
than rendering a different page per role. A template with a role branch in it is a policy
decision in a template.

**Every page is a `GET`.** Mutations go from the browser to the JSON API routes that already
exist, with a JSON content type. The browser surface introduces no write path of its own, which
is what makes §6 the whole boundary rather than half of it.

---

## 4. Escaping

**This is the security property the surface introduces**, and it is not hygiene.

The Jinja environment is built with `autoescape=True` for every template unconditionally — not
`select_autoescape`, which decides by file extension and stops protecting anything the moment a
template is named something else. Four kinds of text arrive here and none of them is this
project's:

| Field | Where it comes from |
|---|---|
| a document title | the file that was indexed |
| a heading path | what the parser found inside it |
| an answer body | a model writing about that document |
| a citation label and quote | the document's own words, under manicule's name |

`tests/web/test_escaping.py` plants hostile markup in all four, checks every page that renders
them **against what the route returned**, and then switches autoescaping off and asserts the same
page *does* carry the raw script tag. Without that last test the others would pass for a fixture
that never had any markup in it.

Two supporting decisions:

- **`StrictUndefined`.** A template naming a field the payload does not have raises rather than
  rendering an empty string. A page that silently shows nothing where a number should be is this
  project's recurring failure: green, wrong, invisible.
- **A field wins over a method.** Jinja's `foo.bar` tries the attribute before the item, so a
  payload field sharing a name with a `dict` method resolves to the method — `ApiKeyList.keys`
  is exactly that, and the page renders a bound method or takes the wrong branch on a truthy
  one. `PayloadEnvironment` reverses the order for mappings. The templates use the `| items`
  filter as a result, which is the right direction: a template asking a payload for its methods
  is always the bug.

The script builds DOM and never markup — every piece of streamed answer text and every citation
label reaches the page through `textContent` — which is the same rule the embeddable widget is
written under, asserted the same way, against what the route served.

---

## 5. What a browser cannot present, said plainly

There is no session cookie in this build, deliberately: a key is presented on every request, and
a signed cookie would be a second credential type with its own expiry, revocation and CSRF story.
A browser cannot attach a header to a top-level navigation.

So on an installation with `security.auth.mode = api_key`, **a page load carries no credential
and this surface refuses it** — with an HTML page that says so and says what to use instead,
rather than a JSON envelope in a browser window.

That is a real limitation rather than a bug, and the configuration this surface is *for* is the
one manicule ships as: one person, on loopback, with `auth.mode = none`, where the caller is the
operator at this machine and holds the authority the command line already gives them. An
interactive login belongs to team mode ([#13](https://github.com/mgd43b/manicule/issues/13)),
where the session, its revocation and its CSRF story can be designed together instead of one of
the three arriving on its own.

manicule is **single-user oriented** until it is feature complete, and this surface reflects
that: no user management, no roles UI, no invitations, no login screens. The auth area is one
person looking at their own API keys and at what this installation currently demands of a caller.

Workspaces are **not** multi-user — one person with several corpora. Workspace scoping is a
correctness property and is enforced on every read: as a predicate in the store, and as identity
arithmetic at the surface. `tests/web/test_tenancy.py` drives the pages against the same
deliberately broken stores `tests/api/test_tenancy.py` uses — ones that ignore the workspace
filter *and* the limit — and asserts the refusal against the **rendered HTML**, because a page is
where a leak would actually be read.

---

## 6. What this surface will not do

It adds no operation. Every page reads through a service method that already has a route on the
HTTP API, so [`surfaces.md` §9.6](surfaces.md#96-what-the-http-surface-will-not-do) is inherited
whole: no hard delete, no reset, no backup or restore, no import or export, no upgrade, no
plugin install, no connector creation, no benchmark.

Two of those collide with the ticket that asked for this surface, and both are resolved in favour
of the boundary:

**There is no drag-and-drop upload.** `POST /api/v1/documents/upload` does not exist, by a
decision that is tested by name: accepting bytes over HTTP and writing them into the corpus is an
ingest path with no filesystem permission check and no path an operator chose, and this is the
surface an unattended caller reaches. Adding one from a different package would have undone that
quietly. The obvious substitute — "type a path and let the server index it" — is worse: it turns
a browser into a reader of every file the process can open. So the documents area is complete
without an ingest verb and says on the page that documents arrive through `manicule index <path>`
or a configured connector.

**Settings are read-only.** `config get` and `config set` have no route on any network surface,
for the same class of reason: reading and writing configuration over the network is how an
installation gets repointed at a different data directory by something holding a key. The
settings area does not reach around that — it shows the installation's *posture* as `doctor` and
`index_status` already report it: what was checked, what the index committed to, where the data
directory is. Everything on it is a fact about the running process rather than the contents of a
file, and changing any of it is `manicule config set` at a terminal.

`tests/web/test_boundaries.py` asserts both: the paths are absent under `/ui` as well as under
`/api`, this package's source contains no call to `config_get` or `config_set`, and every path
the served script fetches is matched against the routes the application actually mounted — so an
operation cannot arrive by way of JavaScript either.

---

## 7. What a browser is told

**Its own `Content-Security-Policy`, narrower than the default rather than looser.** The
application-wide policy is `default-src 'none'`, which is right for JSON and wrong for a
document: a browser applies it to the page and then refuses the page's own stylesheet. So each
page states:

```
default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:;
connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'
```

No `'unsafe-inline'` for script or style, which is why the stylesheet and the script are files
rather than blocks — and which is what makes §4 worth having, because a successful injection
would otherwise run. `base-uri 'none'` because an injected `<base>` silently repoints every
relative URL on the page. `frame-ancestors 'none'` because a chat box in an invisible frame is a
clickjacking target: the widget is the part of manicule meant to be embedded, and this is not.

**No page is cached.** A rendered page holds answer text and document titles, and a shared
machine is exactly where somebody presses Back.

**The shared-conversation page uses a smaller frame.** A guest holds a bearer URL and nothing
else, so a navigation listing every area would disclose what exists here to somebody who can
reach none of it.

---

## 8. Cross-site requests

`manicule.api.origins`. **An unsafe method that a browser says came from another site is refused
unless configuration named that site.**

The question only arises with a browser surface, because cross-site request forgery needs a
browser that will attach *ambient* authority to a request some other page caused — and manicule's
ambient case is the one it ships as: loopback with no credential at all. A page on the internet
cannot read the response, but with a "simple" request it does not need to. A form `POST` is sent
and its effect happens; CORS hides the reply and nothing else.

Two signals, and the first cannot be forged:

- **`Sec-Fetch-Site`** is a forbidden header name — page script cannot set it — and every current
  browser sends it. `same-origin` and `none` (a typed URL or a bookmark) are admitted;
  `same-site` is not, because a sibling subdomain is a different origin.
- **`Origin`** is the fallback for anything older, compared against the `Host` the request was
  addressed to. Scheme is deliberately not compared: behind a TLS-terminating proxy the request
  arrives as `http` while the browser's `Origin` says `https`, and a check that failed there is a
  policy operators switch off.

**A request with neither header is admitted**, and that is deliberate. `curl`, a script, an
assistant holding a key — none of them sends either, and none has ambient authority to abuse.
Refusing them would break every non-browser client to defend against a threat only browsers
create.

An origin listed in `security.transport.allowed_origins` may still write, because the widget is
the one part of manicule that is meant to be cross-origin and a widget asks questions, which is a
`POST`.

**The websocket is checked separately, and it is the worse case.** An HTTP middleware never sees
a websocket scope, and a browser applies *no* cross-origin policy to a `WebSocket`: no preflight,
no CORS, and the page reads every frame that comes back. So a cross-origin socket to an
installation with no credential is not a write whose answer is hidden — it is the corpus,
answering questions, to a page the operator merely visited. `manicule.api.routes.sockets` checks
the handshake's `Origin` through the same decision, before `accept` and before the credential is
looked at, and closes with a policy-violation code. A handshake with no `Origin` — a script, an
assistant — is admitted, exactly as over HTTP.

---

## 9. Keyboard and appearance

- **Command palette**: `Ctrl`/`Cmd`+`K`, or `?`. It reads the navigation out of the DOM the
  frame rendered, so the keyboard route to a page and the clicked route to it cannot come apart
  and the palette has no list of its own to fall out of date.
- **`/`** focuses the search box; `Esc` closes the palette; arrow keys and `Enter` move and open.
- **A skip link and real landmarks** — `header`, `nav`, `main`, `footer` — so the surface is
  navigable without a pointer.
- **Dark mode** follows `prefers-color-scheme` and can be overridden by a toggle that wins in
  both directions. The preference is the one thing this page stores. The widget stores nothing at
  all, precisely because what it holds is a credential; this page holds none.
