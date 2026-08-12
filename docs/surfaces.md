# Surfaces: the CLI, the MCP server, the HTTP API, the browser, and the shape of what they return

Four surfaces, one service, one output contract. This document says what that contract is,
because `--json` is something scripts and assistants parse, and a shape nobody wrote down is
whatever the code happened to do last.

- **The application service** (`manicule.app.service.ApplicationService`) has all the
  behaviour.
- **The command line** (`manicule.cli`), **the MCP server** (`manicule.mcp`), **the HTTP API**
  (`manicule.api`) and **the browser surface** (`manicule.web`) are adapters over it. None of
  them decides anything.

The browser surface renders the same envelope as HTML rather than serialising it, and has its
own document: [`web.md`](web.md). Everything below applies to it too.

---

## 1. Why the layer exists

A rule implemented in the command line is a rule the MCP tool does not have — and the MCP tool
is the one an assistant calls unattended. The HTTP API is worse again: it is reachable by a
script, a widget in somebody else's page, or anything holding a key. Workspace scoping,
credential masking, refusing to install a plugin, refusing a wide bind: every one of those is a
rule that has to hold on all three surfaces or it does not hold.

So the surfaces are thin by construction and the property is checked rather than intended.
`tests/app/test_surface_parity.py` runs the same operation through all of them and compares the
results. It fails the moment they stop being the same call. The browser column cannot be a byte
comparison — a page is HTML — so it asserts the same claim in the form HTML can carry: a value
the tool reported is on the page, and a failure the tool reports is the failure the page shows,
with the same type, message and hint.

### What each layer may contain

| Layer | May | May not |
|---|---|---|
| `manicule.cli` | Parse arguments, read stdin, render, set the exit status | Query a store, compute a filter, decide a policy |
| `manicule.mcp` | Declare tools, describe them, pass arguments through | Anything the CLI may not |
| `manicule.api` | Route, authenticate, decide a status code, frame a stream | Anything the CLI may not |
| `manicule.web` | Render an envelope as HTML, escape it, choose a template | Anything the CLI may not — and it adds no operation of its own |
| `manicule.app.service` | Everything else | Import a database, a model runtime or a web framework |
| `manicule.app.runtime` | Build components, own the lifecycle | Decide anything a surface could ask about |

The service is written against the protocols in `manicule.app.ports`, so the suites drive it
against components that break their half of the bargain — a store that ignores its workspace,
a retriever that returns another tenant's chunk. That is the only way the guards can be
watched firing.

---

## 2. The envelope

Every `--json` emission, every MCP tool result and every HTTP response body is **one JSON
object** in this shape:

```json
{
  "version": "0.1.0",
  "op": "search",
  "ok": true,
  "workspace": "default",
  "data": { "...": "operation-specific" },
  "error": null
}
```

| Field | Always present | Meaning |
|---|---|---|
| `version` | yes | The contract version, which is manicule's own. Something to branch on that is not a guess about which fields exist |
| `op` | yes | The operation's name. **Identical to the MCP tool's name**, so a log line, a shell pipeline and a tool call all name one operation the same way |
| `ok` | yes | Whether this succeeded. Read it first |
| `workspace` | yes | The tenant the operation ran in — **including on failures**, because identity here is workspace-scoped and an answer whose scope is invisible cannot be audited |
| `data` | yes, `null` when `ok` is false | The payload |
| `error` | yes, `null` when `ok` is true | What went wrong |

Both keys are always present, one of them `null`. `exclude_none` is deliberately off: absent
and null are the same thing to a shell pipeline and different things to a typed client.

### Failures

```json
{
  "version": "0.1.0",
  "op": "document_get",
  "ok": false,
  "workspace": "default",
  "data": null,
  "error": {
    "type": "UnknownEntityError",
    "message": "no live document 'abc' in workspace 'default'. …",
    "hint": "Run the matching `list` command to see what this workspace holds."
  }
}
```

`type` is the exception class name. `hint` is what to do about it, and is empty when there is
nothing specific to say.

**A failure is a result, not an exception.** The MCP tool returns this rather than raising, so
an assistant gets something it can act on instead of a transport error with no shape. The CLI
prints it and exits **1**. The HTTP API sends it as the body, with a status code derived from
`error.type` — **the body is the envelope even when the status is not 200**, so a client that
reads `ok` first never has two shapes to parse:

| `error.type` | Status |
|---|---|
| `UnauthenticatedError` | 401 |
| `ForbiddenError`, `PolicyError` | 403 |
| `UnknownEntityError`, `UnknownComponentError`, `UnknownConversationError` | 404 |
| `NameInUseError`, `FingerprintMismatchError` | 409 |
| `RequestValidationError` | 422 |
| `CrossWorkspaceError`, `OSError` | 500 |
| anything else | 400 |

`CrossWorkspaceError` is **5xx** deliberately. Nothing the caller sent could have produced it:
a store returned another tenant's row and the surface refused it. Reporting that as a client
error would file a defect against the caller.

A request FastAPI rejects before a handler runs — a missing parameter, an unknown body field —
is re-dressed as the same envelope rather than left in FastAPI's own `{"detail": [...]}` shape.
That is the most common failure a client meets, and it is the one place a second response shape
would otherwise appear.

**Defects still propagate.** Anything that is not a `ManiculeError`, a `ValueError` or an
`OSError` is a bug in manicule, and dressing one up as a well-formed envelope is how a broken
installation reports success at being broken.

### Streams

`--json` is not on the command line's exit status alone:

- **stdout carries the envelope and nothing else.** No banner, no progress, no prose.
- **Everything human goes to stderr.** `manicule --json search x | jq` on a failed run reads
  an empty stream rather than an error message `jq` cannot parse. `--json` is an option of
  `manicule`, not of each command, so it goes **before** the command name; after it, Typer
  rejects it as an unknown option with exit status 2.
- **Exit status is 0 on success, 1 on a failed operation, 2 on a usage error** that Typer
  rejected before the service was reached.

`start` is the one exception, and it is not a hedge: under the default stdio transport
**stdout is the MCP protocol channel**, so an envelope written there would be a corrupt
message rather than a result. Its address envelope goes to stderr under `--json` as well as
without it.

Without `--json`, `manicule ask` streams tokens to the terminal as they arrive — but only when
stdout is a terminal. Into a pipe it does not, because interleaving tokens with whatever the
consumer is doing helps nobody. Streaming is a *view*: the payload is identical either way,
and `tests/app/test_service.py` asserts it.

---

## 3. Stability

**Adding a field is not a breaking change.** Payloads are closed models, so a field appears
here before it appears in output; a consumer that ignores unknown keys keeps working.

**Removing or renaming one is**, and moves `version`.

**`op` values are stable.** They are the MCP tool names, which are the ticket's names.

Three fields are identities and are never reformatted for display: `embed_fingerprint`,
`chunk_fingerprint` and every `anchor`. A prettied identity is one nobody can compare, which
is the only thing it is for. `IndexStatus` carries `embedding` and `chunking` alongside them
for reading.

---

## 4. The operations

Nineteen MCP tools and nineteen CLI commands. They are not a one-to-one mapping: some
commands group several operations, and some operations have no tool at all.

| Operation (`op`) | MCP tool | Command | Payload |
|---|---|---|---|
| `ask` | ✓ | `ask` | answer, citations, confidence |
| `search` | ✓ | `search` | ranked passages |
| `index_path` | ✓ | `index <path>` | run counters |
| `index_changes` | — | `index --watch` | run counters |
| `index_status` | ✓ | `index` | counts and fingerprints |
| `stats` | ✓ | `index --stats` | counts, grouped three ways |
| `document_list` | ✓ | `document list` | a page of documents |
| `document_get` | ✓ | `document get` | one document, optionally its chunks |
| `document_delete` | ✓ | `document delete` | what was removed, and how |
| `document_reindex` | ✓ | `document reindex` | what was repaired |
| `doctor` | ✓ | `doctor` | diagnostics |
| `connector_list` | ✓ | `connector list` | configured sources |
| `connector_sync` | ✓ | `connector sync` | run counters |
| `connector_login` | — | `connector login` | who a captured browser session belongs to |
| `config_get` | ✓ | `config get` / `config show` | configuration, redacted |
| `config_set` | ✓ | `config set` | the key, before and after |
| `workspace_list` | ✓ | `workspace list` | workspaces, active marked |
| `workspace_switch` | ✓ | `workspace switch` | previous and current |
| `plugin_list` | ✓ | `plugin list` | installed plugins and components |
| `plugin_add` | ✓ | `plugin add` | what was enabled |
| `plugin_remove` | ✓ | `plugin remove` | what was disabled |
| `backup` / `restore` | — | `backup` | where it went, what it holds |
| `export` | — | `export` | a portable archive |
| `import` | — | `import` | run counters |
| `reset_index` | — | `reset-index --yes` | what was removed |
| `init` | — | `init` | what was written and decided |
| `upgrade` | — | `upgrade` | current, target, and the command to run |
| `start` / `stop` | — | `start` / `stop` | the address, and whether it is loopback |
| `completion` | — | `completion` | a shell script |
| `auth_create_key` / `auth_list_keys` / `auth_revoke_key` | — | `auth …` | API keys |

### Operations with no MCP tool, and why

`reset_index`, `backup`, `restore`, `import`, `upgrade`, `start`, `stop`, `connector_login` and
the `auth` verbs are command-line only. Each of them either destroys data, mints a credential,
or changes what the installation *is* — and a tool an assistant can call unattended should not
be able to do any of that. The nineteen tools read the corpus, write documents into it, and
adjust configuration. That is the whole surface.

`connector_login` is in that list for the credential reason and for one more: it reads a secret
from a terminal without echoing it. A surface that cannot do that would have to accept the
secret as a parameter, and a session cookie in a tool call is a session cookie in a transcript.

---

## 5. Payloads

Defined in `manicule.app.results` — that module is the definition, this section is the tour.

### `ask` → `AnswerResultPayload`

`text`, `citations[]`, `dropped`, `confidence`, `confidence_band`, `corpus_consulted`,
`ungrounded`, `context_truncated`, `redacted`, `finish_reason`, `error`, `conversation_id`,
`message_id`, `model`, `elapsed_ms`.

Three of those are separate claims and are never combined:

- `confidence` is **absent** — not `0.0` — when the corpus was not consulted. "We did not
  look" and "we looked and there is nothing" are different answers.
- `ungrounded` means the context was non-empty and **nothing survived verification**. An
  answer with no citations because the corpus was not consulted is not ungrounded.
- `dropped` counts citations the model emitted that could not be verified. They were deleted
  rather than shown.

Each citation carries `slot`, `document_id`, `chunk_id`, `uri`, `title`, `heading_path`,
`kind`, `anchor`, `quote` and `verification`.

### `search` → `SearchResult`

`query`, `profile`, `count`, `hits[]`, `confidence`, `confidence_band`, `confidence_reason`,
`route`, `cached`, `truncated`, `elapsed_ms`.

Each hit carries the passage, its document, its anchor, its effective `score` **and** `scores`
— the score every pipeline stage gave it. The per-stage history is kept because "reranking
helped" is only checkable while the pre-rerank score survives.

### `index_path` / `connector_sync` / `import` → `IngestReport`

`connector`, `discovered`, `ingested`, `skipped`, `failed`, `expanded`, `by_status`, `error`,
`elapsed_ms`.

`by_status` is the run's own counter table rather than a summary of it. A document that ended
`no_extractable_text` is neither an ingest nor a failure, and collapsing the two would hide
exactly the outcome that needs looking at.

`error` non-empty means the run did not finish. **The watermark was not advanced**, so running
it again resumes.

### `doctor` → `Diagnosis`

`state` and `checks[]`, each `{name, state, detail}`. States are `ok`, `degraded`, `failing`
and `unknown` — the last is a check that could not run, which is deliberately not `ok`.

Checks: `configuration`, `transport`, `plugins`, `storage`, `permissions`, `index`, `grammars`,
and `component:<kind>:<name>` for anything already constructed.

**`doctor` builds nothing expensive.** No model runtime is loaded and no document is read, so
it is safe on an installation that is not working — which is the only time anybody runs it.

**`manicule doctor --fix` is the one exception, and it is a flag for that reason.** It performs
the repairs `doctor` knows how to perform — today exactly one, seeding the declared tree-sitter
grammars from an offline bundle if one is installed and from the grammar release otherwise
([`parsing.md`](parsing.md#81-grammar-packaging-is-the-real-problem) §8.1) — and then reports
the state that resulted. It is the only part of this command that writes to the machine or uses
the network, and it is passed by **the command line alone**: the MCP tool and `GET
/api/v1/health` call the report, because a diagnostic an assistant can reach should not be able
to start a download. `manicule init` runs the same repair, which is how a fresh install ends up
with grammars rather than discovering at first index that it has none.

`stats`, `index_status` and `doctor` are deliberately thin. Trends, history and alerting
belong to Operations ([#14](https://github.com/mgd43b/manicule/issues/14)); a surface that
invented them here would be a second, weaker copy of that subsystem.

---

## 6. Where a server listens

`manicule start` serves MCP over **stdio** by default, which opens no socket at all. There is
no address to get wrong on the path everybody uses.

`--transport http` serves the **HTTP API**; add `--mcp-only` to serve the MCP protocol over
that socket instead. Either way the address goes through `manicule.app.bind.resolve_bind`, and
a non-loopback bind needs **all three** of:

1. a host that is not loopback — and the configured default is `127.0.0.1`, so this is always
   something a person wrote down;
2. `--allow-public-bind`, which no configuration file can set and no default supplies;
3. `security.auth.mode` set to something other than `none`.

Any one missing is a refusal naming which. None of the three can be reached by omission: the
absent value in each case is the safe one.

`tests/app/test_bind.py` asserts each condition separately, asserts the positive case so the
policy is not merely "refuse everything", and reads the source tree to check that **no module
but the bind policy names an all-interfaces address**. That last one is what survives a future
server that binds a literal instead of asking.

There is a **second** refusal, and it is deliberately not the same code. `resolve_bind` decides
an address; `manicule.api.app.build_app` decides whether an *application* may exist at all, and
refuses to build an unauthenticated one whose address is not loopback. That one fires even when
something other than `manicule start` is doing the listening — a container entry point, a
production ASGI server, a hand-written uvicorn call.

---

## 7. Tenancy

Everything is scoped to one workspace, and the scope is enforced twice by two mechanisms that
cannot fail the same way.

**In the store**, as a predicate: the handle carries the workspace and no method takes one
(`manicule.storage.scoped`). This is where isolation is enforced.

**At the surface**, as arithmetic: `document_id` is a digest of
`(workspace_id, source, source_id)`, so recomputing it from a document's own fields proves the
id was minted for this workspace (`manicule.app.tenancy`). It consults nothing and would still
fire if every `WHERE` clause in storage were deleted.

A foreign document is **refused whole**, never filtered out — a shortened list is a success
report for a partial answer. The refusal never quotes the offending title, URI or text: a leak
reported by quoting what leaked is still a leak.

`ask` performs the check **before the model is called**, so nothing from another tenant
reaches a provider. `tests/app/test_tenancy.py` asserts that through the answerer's own call
log rather than through the exception, because an implementation that refused after streaming
would raise exactly the same error having already sent the passages.

`tests/api/test_tenancy.py` does the same through the routes, against the same deliberately
broken stores — one that ignores its workspace filter *and* its limit, so the surface's own
identity check is what has to fire rather than a truncation catching a foreign row by accident.
Every case there has a control beside it: the same route against a correct store returns the
tenant's own document, because a surface that refused everything would satisfy the negatives
and be useless.

`tests/web/test_tenancy.py` does it a third time, through the pages and against the same broken
stores, asserting on the **rendered HTML** — because a page is where a leak would actually be
read, and a title that never reached a payload could still reach a heading or a link.

---

## 8. Two things the surfaces refuse to do

**manicule does not install plugins.** A plugin is imported into this process and runs with
everything it has (`CONTRIBUTING.md`). `plugin add` enables one that is already installed; for
one that is not, it reports the command that would install it and runs nothing. A tool an
assistant can call unattended must not be able to fetch and execute a package.

**manicule does not upgrade itself.** `upgrade` takes a backup — the part that is dangerous to
skip — and reports the exact command. A failure part-way through an install leaves the
installation holding your index broken, and that is not a state to reach from a command that
reads like a version bump.

Both are design decisions rather than omissions, and both are stated in the output rather than
left to be discovered.

---

## 9. The HTTP API

`manicule.api`. Eleven route groups, `manicule start --transport http`, OpenAPI at
`/api/docs`. Every route parses a request, calls one service method, and renders the envelope
above.

### 9.1 The groups

| Group | Routes |
|---|---|
| health | `GET /healthz`, `GET /readyz`, `GET /api/v1/health`, `GET /api/v1/stats`, `GET /api/v1/workspaces` |
| documents | `GET /api/v1/documents`, `GET /api/v1/documents/{id}`, `DELETE /api/v1/documents/{id}`, `GET /api/v1/documents/trash`, `POST /api/v1/documents/{id}/restore`, `POST /api/v1/documents/{id}/reindex`, `GET /api/v1/search` |
| chat | `POST /api/v1/chat`, `POST /api/v1/chat/stream`, `POST /api/v1/chat/feedback` |
| conversations | `GET`/`POST /api/v1/conversations`, `GET /api/v1/conversations/{id}/messages`, `PATCH`/`DELETE /api/v1/conversations/{id}`, `POST`/`DELETE /api/v1/conversations/{id}/share`, `GET /shared/{token}` |
| collections | `GET`/`POST /api/v1/collections`, `DELETE /api/v1/collections/{id}`, `GET /api/v1/collections/{id}/documents`, `POST`/`DELETE /api/v1/collections/{id}/documents/{docId}` |
| tags | `GET`/`POST /api/v1/tags`, `DELETE /api/v1/tags/{id}`, `POST`/`DELETE /api/v1/documents/{docId}/tags/{tagId}` |
| admin | `GET /api/v1/admin/stats`, `/query-logs`, `/audit-logs`, `/search-quality`, `/plugins`, `/connectors`, `POST /api/v1/admin/connectors/{name}/sync` |
| plugins | `GET /api/v1/plugins`, `GET /api/v1/plugins/search`, `POST`/`DELETE /api/v1/plugins/{name}` |
| auth | `GET /auth/providers`, `GET /auth/session`, `GET`/`POST /api/v1/auth/keys`, `DELETE /api/v1/auth/keys/{nameOrId}` |
| workbench | `GET /api/v1/workbench?document_id=…` |
| websocket chat | `WS /api/v1/chat/ws` |

Plus the embeddable widget: `GET /widget/widget.js` and a static page at `GET /widget`, and the
browser surface at `/ui` — twelve areas of server-rendered HTML over the same service, mounted on
the same application. It is not a twelfth route group: it publishes no operation of its own, and
[`web.md`](web.md) is its document.

`/healthz` and `/readyz` are the only routes that answer without an envelope, because they
answer a probe rather than a person. They also answer different questions: `/healthz` opens
nothing, `/readyz` asks the store whether the index is usable. Collapsing them gives you a
liveness probe that restarts a healthy process because a database file is briefly locked.

Both are unauthenticated, so both carry `{"status": …}` and nothing else — no counts, no
configuration and no reason for an unready answer. "412 documents, chunker structural" is a
corpus fingerprint, and so, more quietly, is the text of a database error. What "ready" *means*
is the service's decision (`ApplicationService.ready`), not the route's.

### 9.2 Identity

`security.auth.mode` decides. With `api_key`, a key is presented on **every** request as
`Authorization: Bearer <key>` or `X-API-Key: <key>` — never a query parameter, because a
credential in a URL is a credential in the access log, the browser history and every `Referer`
the page sends. On the websocket, where a browser cannot set headers, it travels in the
`Sec-WebSocket-Protocol` header as `manicule.api-key.<key>` and the server echoes the
subprotocol back.

Roles are a floor: `viewer` reads, `member` writes, `admin` administers. A route asks for the
least it needs.

**A refusal names the same operation a success would.** A refused request never reaches its
service call, so `op` comes from the matched route — and every route therefore carries an
explicit `name=` equal to the service method it calls. Without that it would be the handler's
Python function name (`list_documents` rather than `document_list`), and an access log of
refusals would be unjoinable to one of successes. `tests/api/test_contract.py` enumerates the
mounted routes and fails on a name that is not an operation.

With `security.auth.mode = none` there is no credential and the caller is treated as the
operator — which is only tolerable because that configuration cannot be reached from anywhere
but loopback, refused twice (§6).

### 9.3 Whose address a request has

`X-Forwarded-For` is a header, and clients set headers. Read without qualification it turns
every IP-based decision into a value the caller chose, silently.

So: **a forwarding header is believed only from a peer inside
`security.transport.trusted_proxies`, and that list is empty by default.** With no proxies
configured the header is not read at all — the socket peer is the answer, and a socket peer is
not something a caller can set. When a proxy *is* trusted, the client is the **right-most**
entry that is not itself a trusted proxy, because a caller controls the left-hand end of the
header: everything it fabricates is prepended, and what the real proxy appended is at the
right. An entry that does not parse as an address is skipped rather than passed along.

A malformed CIDR in `trusted_proxies` is refused at startup. It would otherwise fail closed and
silently, with the operator believing a policy is in force that is not.

### 9.4 Cross-origin, framing and the widget

The widget runs inside somebody else's page, which is the only part of manicule that is
deliberately cross-origin.

- **CORS is explicit or absent.** With no `security.transport.allowed_origins` the middleware
  is not installed, which means same-origin. Configuration refuses `*` outright: a wildcard
  over a document index means any page a user visits can read it.
- **Credentials are never permitted cross-origin.** A key is presented per request; there is no
  cookie to attach, and `allow-credentials` is the ingredient a CSRF needs.
- **Framing is refused unless somebody named the frames.** Every response carries
  `frame-ancestors`, naming `security.transport.widget_allowed_domains` and `'none'` when that
  is empty.
- **The widget builds DOM, never markup.** Every piece of answer text and every citation label
  reaches the page through `textContent`. An answer is model output over indexed documents;
  treating it as markup would make any document in the corpus a script into every embedding
  page. The served script is a module-level constant with no request value in it.
- **A widget key is as public as the page it is on.** manicule does not pretend otherwise: mint
  a dedicated key, give it the least role that works, and revoke it on its own.

The application-wide policy is `default-src 'none'`, which is right for JSON and for the script
and **wrong for the one page that loads the script** — a browser applies it to the document and
refuses the `<script src>`. So `GET /widget` states its own, narrower policy: `script-src
'self'`, `connect-src 'self'`, inline styles for the shadow root, and still `frame-ancestors
'none'`. Nothing else on the surface is loosened.

### 9.5 Streaming

`POST /api/v1/chat/stream` emits `delta`, `citation`, `drop` and one `final` event. The `final`
frame is the **identical envelope** `POST /api/v1/chat` would have returned, because the service
builds the settled payload once and both paths carry it. A client that reads only that frame
has made the non-streaming call.

Every frame is `json.dumps` of a model dump. An SSE event ends at a blank line, so answer text
containing one would terminate a frame early on a hand-built `data:` line — a frame-injection
primitive built out of a model writing a paragraph break.

The websocket carries the same sequence, wrapped as `{"event": …, "data": …}`, over a
connection that answers several questions.

### 9.6 What the HTTP surface will not do

Seven operations exist on the command line and have **no route**. This surface is the one an
unattended caller reaches, so each of them is absent rather than merely guarded:

| Absent | Why |
|---|---|
| `document delete --hard` | Unrecoverable. The route soft-deletes, and `POST /documents/{id}/restore` undoes it |
| `reset-index` | Empties the workspace with no restore path |
| `backup` / `restore` | One writes wherever the caller names; the other overwrites the live data directory |
| `import` / `export` | The same, over a corpus archive |
| `upgrade` | Fetching and executing code |
| plugin *install* | The same. `POST /plugins/{name}` enables one already installed |
| document *upload* | An ingest path with no filesystem permission check and no path the operator chose |
| creating a connector | A connector holds credentials and reaches a remote system. Sources are declared in configuration, where the whole set is reviewable in one place |
| a benchmark endpoint | A benchmark on request is one HTTP call away from an unusable installation |
| `config get` / `config set` | Reading and writing configuration over the network is how an installation gets repointed at a different data directory |

`tests/api/test_routes.py` asserts each absence by name. An absence with no test is an absence
that comes back.

The browser surface adds **no operation**, so it inherits all of it — and
`tests/web/test_boundaries.py` asserts the same absences again under `/ui`, because an absence
protected by a test that only knows about `/api` is an absence a second package can undo. Two of
these were asked for by [#12](https://github.com/mgd43b/manicule/issues/12)'s checklist and are
deliberately not built; [`web.md` §6](web.md#6-what-this-surface-will-not-do) says why.

### 9.6.1 Cross-site writes

An unsafe method a browser says came from another origin is refused unless that origin is in
`security.transport.allowed_origins`. `manicule.api.origins` decides it, checked in middleware
before routing.

The threat needs a browser holding *ambient* authority, which is the posture manicule ships as:
loopback with `security.auth.mode = none`, where there is no credential and the caller is whoever
can reach the port. CORS hides the **response** to a cross-origin request and does not stop a
"simple" one being sent, so a form `POST` from a page the operator merely visited would take
effect. `Sec-Fetch-Site` is the primary signal because page script cannot set it; `Origin`
compared against `Host` is the fallback. A request with neither header — every non-browser
client — is unaffected.

**The websocket gets the same check, at the handshake**, because middleware never sees a
websocket scope — and it is the more serious case: a browser applies no cross-origin policy to a
`WebSocket` at all, so the page reads every frame rather than only causing an effect it cannot
see. It is refused before `accept` and before the credential is read.

### 9.7 Telemetry, and what a failed write costs

`search` and `ask` record a row in `query_logs`, and that recording is the **service's** rather
than a surface's: telemetry written only by whichever surface remembered to write it describes
that surface's traffic instead of the installation's.

The two writes this surface makes are treated **differently on purpose**:

- **A failed query-log write does not fail the query.** Retrieval is a read; recording it is a
  write, and on SQLite a write can lose to a lock. Letting it propagate would make a search that
  worked yesterday return 500 today because an observability insert could not get the writer. It
  is logged at warning — without the query text, which is user content — rather than swallowed.
- **A failed audit write fails the operation it was auditing.** A trail with holes in it is
  worse than none, because the holes are invisible and the operation reported success.

### 9.8 Search quality

`GET /api/v1/admin/search-quality` **reports**; it does not measure. `manicule.evaluation` is
the only thing in this project that decides whether one retrieval configuration beats another,
and it refuses to report at all for a system it cannot distinguish from guessing. The route
reads that harness's own store and renders its own report.

`available: false` means nobody has judged any pairs — which is the truth, and is not a score
of zero. `is_evidence: false` means the query set behind the numbers is an **example** one, and
the harness's caveat travels with it. See [`evaluation.md`](evaluation.md).

---

## 10. Export and import

`export` writes **retained source bytes and document metadata**. No chunks and no vectors:
`ArchiveManifest` has nowhere to put them. `import` feeds the archive through the ordinary
ingest pipeline, so the importing installation derives chunks and vectors with **its own**
fingerprints.

Copying chunks between machines would move an index built by another chunker and another
embedder into a store whose fingerprints say otherwise — the silent mismatch that
`EmbedFingerprint` and `ChunkFingerprint` exist to prevent. The archive version is checked
before a single byte is read, and a newer one is refused rather than parsed optimistically.

For a byte-identical copy of an installation, use `backup` and `restore`. They are different
operations answering different questions.
