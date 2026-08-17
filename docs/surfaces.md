# Surfaces: the CLI, the MCP server, the HTTP API, the browser, and the shape of what they return

Four surfaces, one service, one output contract. This document says what that contract is,
because `--json` is something scripts and assistants parse, and a shape nobody wrote down is
whatever the code happened to do last.

- **The application service** (`manicule.app.service.ApplicationService`) has all the
  behavior.
- **The command line** (`manicule.cli`), **the MCP server** (`manicule.mcp`), **the HTTP API**
  (`manicule.api`) and **the browser surface** (`manicule.web`) are adapters over it. None of
  them decides anything.

The browser surface renders the same envelope as HTML rather than serializing it, and has its
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
| `data` | yes; usually `null` when `ok` is false | The payload, including partial ingest counters on an incomplete run |
| `error` | yes, `null` when `ok` is true | What went wrong |

Both keys are always present. Ordinarily one is `null`; an incomplete ingest is the one
deliberate exception and carries both a typed `error` and its partial `IngestReport` in `data`.
That lets automation branch on `ok` without losing the durable counters it needs for diagnosis.
`exclude_none` is deliberately off: absent and null are the same thing to a shell pipeline and
different things to a typed client.

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
| an incomplete ingest carrying partial `data` | 503 |
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

- **stdout carries the envelope and nothing else.** No banner, no progress, no prose, and no
  ANSI escape sequence even when the terminal has asked for color.
- **Everything human goes to stderr.** `manicule --json search x | jq` on a failed run reads
  an empty stream rather than an error message `jq` cannot parse.
- **`--json` goes on either side of the command name.** `manicule --json doctor` and
  `manicule doctor --json` are the same invocation, and naming it in both positions at once is
  not an error. It was once accepted only *before* the command name, and after it Typer
  rejected it as an unknown option with exit status 2 — which is worth stating because the
  restriction was written up twice as though somebody had chosen it.
- **So does `--workspace`/`-w`**, and it is the same option in both places rather than a
  general one and a specific one. That distinction decides what happens when the two positions
  disagree — see below.
- **Exit status is 0 on success, 1 on a failed operation, 2 on a usage error** that Typer
  rejected before the service was reached.

### Color, and the three variables that decide it

Color applies to the **human** output only. Under `--json` the envelope is written straight to
stdout rather than through Rich, so no color variable can put a byte in it — that is a
property of how it is written, not a setting, and it holds in every environment below.

manicule imposes no color policy of its own: `manicule.cli.render.console` passes no color
arguments, so both conventions are honored by Rich and the behavior is Rich's. What that
delegation means, as of Rich 14:

| Environment | Human output |
|---|---|
| `FORCE_COLOR` set | colored |
| `FORCE_COLOR` and `NO_COLOR` set | **not** colored |
| `FORCE_COLOR` set, `TERM=dumb` | **not** colored |
| `NO_COLOR` alone | not colored |
| nothing set | colored only when stdout is a terminal |

**`NO_COLOR` wins over `FORCE_COLOR`, but not by overriding it** — and the distinction is worth
stating because it is what makes the pair predictable. They are separate mechanisms:
`FORCE_COLOR` declares that the stream *is a terminal*, and `NO_COLOR` strips the color back
out of what gets written to it. With both set the stream is treated as a terminal and the
output has no color in it. Escape sequences that are not color — bold, dim — still appear,
which is what [no-color.org](https://no-color.org/) asks for: it governs color, not styling.

**`TERM=dumb` is not a color switch and is the one that surprises.** It declares what the
terminal can *render*, and Rich believes a capability over a request: `TERM=dumb` with
`FORCE_COLOR` set produces no escape sequences at all. Any environment whose terminal cannot
render ANSI reports it, so the same command is colored in one shell and plain in another with
nothing about manicule having changed.

`TTY_COMPATIBLE` is the same shape and is checked **before** `FORCE_COLOR`: `TTY_COMPATIBLE=0`
means "not a terminal" whatever else is set.

The table is pinned by `tests/app/test_cli.py`, which asserts each row against captured stdout
rather than trusting this document — so a Rich upgrade that changed what an operator's
`NO_COLOR` does would fail the suite rather than the operator. The suite sets these variables
explicitly for the same reason: color is decided entirely by the environment, so a test that
inherited the caller's shell would report on that shell rather than on manicule.

### Options `manicule` and its commands share

Two options are declared on `manicule` itself and accepted by every command as well:
`--json` and `--workspace`/`-w`. `--version` is deliberately **not** among them: it replaces
the invocation rather than modifying it — eager, prints one line, exits — so
`manicule doctor --version` would be asking for the version *of doctor*, which does not exist.

That list is not maintained by hand. `tests/app/test_cli.py` reads the root callback's real
parameters out of the built command tree and requires each one to be either shared with the
commands or carrying a written reason for not being, so an option added later fails until
somebody classifies it rather than quietly becoming the next thing that cannot be typed where
people type it.

**Naming a shared option in both positions at once is not an error — except when the two
disagree.** `--json` is a flag, so saying it twice says the same thing twice and the positions
cannot contradict each other. `--workspace` carries a value, and that value is a tenancy
boundary:

```
$ manicule --workspace a doctor --workspace a     # fine: one workspace, said twice
$ manicule --workspace a doctor --workspace b     # refused, exit 2, naming both
```

The refusal is deliberate and last-wins was rejected. Last-wins is defensible when two
positions mean "general" then "specific", but by construction this is the *same* option in two
places, so there is no specificity to appeal to. Choosing silently would run the operation in a
workspace the operator also named, and the envelope would report the winner as though it were
the whole request — a wrong-tenant run that reads exactly like a correct one, in a system where
scope is an auditability property (§7) and cross-workspace access is a 5xx. Two contradictory
instructions in one invocation is a typo, not a plan, and it is refused for the same reason
`manicule backup --output X --restore Y` is.

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

Thirty-seven MCP tools and twenty-seven CLI commands. They are not a one-to-one mapping: some
commands group several operations, and some operations have no tool at all. Both counts are
asserted rather than written down — `tests/app/test_surface_parity.py` reads them off the built
server and the built command tree.

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
| `document_reindex` | ✓ | `document reindex <id>` | what was repaired |
| `document_reindex_stale` | — | `document reindex --stale` | counts for a corpus-wide re-parse |
| `reembed_plan` | — | `reembed plan` | aggregate cost and capacity estimates |
| `reembed_start` | — | `reembed start <run-id>` | ownerless durable run under an id chosen before the call |
| `reembed_resume` | — | `reembed execute` / `reembed resume` | aggregate durable progress/publication result |
| `reembed_status` | ✓ | `reembed status` / `reembed inspect` | private-safe aggregate durable progress |
| `reembed_abandon` | — | `reembed abandon` | terminal state without a live-pointer change |
| `reembed_cleanup` | — | `reembed cleanup` | whether terminal non-live storage was removed |
| `rebuild_plan` | ✓ | `rebuild plan` | connector-free snapshot cost and capacity estimate |
| `rebuild_run` | — | `rebuild execute` / `rebuild resume` | durable derived generation publication |
| `rebuild_status` | ✓ | `rebuild status` | private-safe aggregate rebuild checkpoint |
| `lifecycle_reset_derived` | ✓ dry-run only | `reset-derived --dry-run/--yes` | aggregate derived rows removed; source roots retained |
| `lifecycle_cleanup_generations` | ✓ dry-run only | `cleanup-derived-generations [--yes]` | eligible/protected generations and temporary bytes |
| `lifecycle_release_history` | ✓ dry-run only | `release-source-history BEFORE [--yes]` | policy-eligible history and uniquely released bytes |
| `lifecycle_delete_snapshot` | ✓ dry-run only | `snapshot-delete RUN_ID [--confirm TOKEN]` | aggregate unrecoverable item/byte impact and confirmation token |
| `doctor` | ✓ | `doctor` | diagnostics |
| `connector_list` | ✓ | `connector list` | configured sources |
| `snapshot_status` | ✓ | `connector snapshot` | aggregate durable snapshot status |
| `snapshot_verify` | ✓ | `connector verify` | aggregate manifest integrity result |
| `connector_sync` | ✓ | `connector sync [--acquire-only]` | run counters and shared lifecycle status |
| `connector_login` | — | `connector login` | who a captured browser session belongs to |
| `config_get` | ✓ | `config get` / `config show` | configuration, redacted |
| `config_set` | ✓ | `config set` | the key, before and after |
| `workspace_list` | ✓ | `workspace list` | workspaces, active marked |
| `workspace_switch` | ✓ | `workspace switch` | previous and current |
| `plugin_list` | ✓ | `plugin list` | installed plugins and components |
| `plugin_add` | ✓ | `plugin add` | what was enabled |
| `plugin_remove` | ✓ | `plugin remove` | what was disabled |
| `collection_create` | ✓ | `collection create` | the collection that was made |
| `collection_list` | ✓ | `collection list` | every collection, and the rule each carries |
| `collection_rename` | ✓ | `collection rename` | the collection, renamed |
| `collection_update` | ✓ | `collection update` | the collection, described |
| `collection_delete` | ✓ | `collection delete` | what was deleted; the documents survive |
| `collection_add` | ✓ | `collection add` | how many memberships changed |
| `collection_remove` | ✓ | `collection remove` | the same, the other way |
| `collection_documents` | ✓ | `collection documents` | a page of a collection's documents |
| `collection_counts` | ✓ | `collection counts` | documents and chunks, counted now |
| `collection_orphans` | — | `collection orphans` | documents in no collection, and what was done |
| `connector_sidecar` | — | `connector sidecar` | a manifest written beside each mirrored page |
| `backup` / `restore` | — | `backup` | where it went, what it holds |
| `export` | — | `export` | a portable archive |
| `import` | — | `import` | run counters |
| `reset_index` | — | `reset-index --yes` | derived state removed and durable snapshot items retained |
| `init` | — | `init` | what was written and decided |
| `upgrade` | — | `upgrade` | current, target, and the command to run |
| `start` / `stop` | — | `start` / `stop` | the address, and whether it is loopback |
| `completion` | — | `completion` | a shell script |
| `auth_create_key` / `auth_list_keys` / `auth_revoke_key` | — | `auth …` | API keys |

### 4.0.1 Shared lifecycle status

Snapshot acquisition and derived-generation work carry one closed `lifecycle` object through the
same envelope on CLI JSON, authenticated admin HTTP, stdio/control MCP, connector metadata and
scheduler records. Human CLI renders the enclosing operation while JSON consumers receive the
same object unchanged. It contains aggregate counts, snapshot completeness and promotion facts,
watermark presence, backlog and offline-continuation facts, phase/outcome/rate/remaining work,
producer identities where they are safe, inventory recovery state and reconciled deletion count,
and a typed aggregate capacity or missing-input refusal.

An unavailable fact is null or empty, never an invented zero. The object cannot contain source
ids, paths, URIs, titles, content or exception context. Re-embedding exposes fingerprints and a
one-way identity of the live generation, not its private corpus-snapshot handle. Read-only MCP
registration is unchanged: adding status fields does not add a write tool to the network surface.

`inventory_recovery` has four public-safe values: empty means ordinary same-manifest work;
`reenumeration_required` means a completed inventory was invalidated and cannot promote;
`reenumerating` means its fenced replacement is discovering from the committed position; and
`reconciled` means that replacement reached authoritative exhaustion. `reused_items` counts
validated retained bodies, and `reconciled_deleted_items` counts predecessor identities absent
from that complete replacement. Neither count identifies a member. Incomplete enumeration keeps
the recovery active and the deletion count at zero.

### Operations with no MCP tool, and why

`reset_index`, `backup`, `restore`, `import`, `upgrade`, `start`, `stop`, `connector_login`,
`connector_sidecar`, `collection_orphans`, `document_reindex_stale`, the five mutating or
corpus-scanning `reembed` operations, `rebuild_run`, and the `auth` verbs are
command-line only. Each of them either destroys data, mints a credential, writes into the
operator's own corpus directory, or changes what the installation *is* — and a tool an
assistant can call unattended should not be able to do any of that. The thirty-seven tools read
the corpus, write documents into it, group them, and adjust configuration. That is the whole
surface. Four of these absences are asserted by name in `tests/app/test_surface_parity.py` —
`collection_orphans`, `connector_sidecar`, `connector_login` and `document_reindex_stale`,
each of which was argued about rather than obvious. The rest are held by the tool count and by
this list.

`document_reindex_stale` is there for a fourth reason, and it is the one that also keeps it off
the HTTP surface (`tests/api/test_routes.py`). It re-parses, re-chunks and re-embeds every
document a parser bump touched, and it takes as long as the corpus is long — so an unattended
caller able to start one has the machine's accelerator for an hour, which is the argument
already made for refusing a benchmark endpoint. `document_reindex` stays on every surface,
because one document is a bound.

MCP retains only `reembed_status`: an assistant cannot spend corpus-sized accelerator, disk and
time unattended. Authenticated admin HTTP has plan/start/resume/abandon/cleanup parity. The Web
page is deliberately read-only: it displays a workspace-safe plan and accepts an opaque id for
status, but has no mutation controls or JavaScript action handlers and never lists or discovers
runs. Workspace ownership is checked before a supplied id resolves, so another tenant's id is the
same sanitized not-found as an unknown id.

`connector_login` is in that list for the credential reason and for one more: it reads a secret
from a terminal without echoing it. A surface that cannot do that would have to accept the
secret as a parameter, and a session cookie in a tool call is a session cookie in a transcript.

### 4.1 What each tool says it does, and why that is not permission

Every tool publishes the four hints MCP defines — `readOnlyHint`, `destructiveHint`,
`idempotentHint`, `openWorldHint` — in `tools/list`. Twenty-two of the thirty-seven say they only
read.

**They are a description, and nothing in manicule reads them back.** No tool is gated on its own
annotation, there is no server-wide `approve` or `trusted` default, and a client that ignores the
whole block reaches exactly what it reached before. That is the specification's own position and
it is also the only honest one here: the same operations are on the HTTP surface and the command
line, neither of which has annotations at all, so a server enforcing its own hints would be
enforcing them on one caller in three.

**What they buy is an operator being able to approve one tool.** Before this, a non-interactive
client had two options for `search`: prompt a person on every call, or auto-approve the server —
and the same server deletes documents, rewrites configuration, enables plugins, synchronizes
connectors and re-indexes. "Trust the whole server" is not a narrower permission than "trust
`document_delete`", it is the same permission. One pass over `tools/list` now separates them, so
the right operator policy is to approve the read-only tools by name and leave the rest asking.

The four questions are asked in English at each registration and translated in
`manicule.mcp.server.hints`, once. None of them has a default, so a tool added later answers all
four or does not build:

| Question | Hint | True when |
|---|---|---|
| does it read? | `readOnlyHint` | the index, the configuration and the installation are as it found them |
| does it remove? | `destructiveHint` | it can remove or overwrite something the call cannot put back from its own arguments |
| is it repeatable? | `idempotentHint` | calling it again with the same arguments changes nothing further |
| does it reach out? | `openWorldHint` | it can reach a remote system, or a part of this machine manicule does not own |

Four of the answers are worth stating because a name would have got them wrong:

- **`ask` is not read-only.** It reads the corpus exactly as `search` does, and with a
  `conversation_id` it persists the turn — and the model it calls may be a provider on somebody
  else's machine. Two independent reasons, either one sufficient.
- **`plugin_list` is read-only *and* open-world.** `registry=True` fetches the community
  listing over the network. It still writes nothing, so both hints are true at once.
- **`plugin_add` reaches out too**, which reads oddly for a tool that installs nothing: a name
  that is not installed sends it to the registry to find out whether it exists, so the refusal
  can name the command that would install it.
- **`collection_remove` is not destructive.** It names the documents it removes, so
  `collection_add` with the same ones restores the membership. `collection_delete` is the one
  that cannot be undone from its own arguments.

**`search` carries `readOnlyHint: true` and does perform one write**, and it is stated here
rather than hidden: retrieval appends a row to the local query log. That is the exception this
project already makes for it — `manicule.app.dispatch.READ_ONLY_OPS` has carried it since before
the annotations existed, and §9.7 says why a failed write there does not fail the query. It
records that a read happened; it changes nothing any search, answer or listing reports, and it is
readable only through the admin query-log surface. Nothing else on the read-only side writes at
all, which is asserted rather than claimed: `tests/mcp/test_annotations.py` calls every tool that
says it reads and compares the whole backend across the call.

**`READ_ONLY_OPS` is not this list and must not be wired to it.** It answers a different
question — does this operation need the data directory's exclusive writer lock — and it answers
"read-only" for `ask`, `backup`, `export` and `init`, none of which is read-only in the sense a
client cares about. Two questions, two answers, and the overlap is a coincidence of the English.

### 4.2 Qualifying a question against a collection, without spending the context on it

The server's `instructions` tell a fresh client to do four things in order, and
`tests/mcp/qualification.py` is that procedure written as something that runs, over the protocol,
against a synthetic corpus. The order is not interchangeable.

1. **`collection_list`** — resolve the name. Keep both identities: `id` is stable and is what
   `collection_counts` takes, `name` is what `search` and `ask` take as a scope.
2. **`collection_counts(collection_id)`** — the document and chunk totals.
3. **`search(query, collections=[name], limit=5)`** — three of these at most, each aimed at a
   different part of the question, and a fourth only to close a gap that can be named.
4. **One more search, scoped the same way, for the thing you expect to be absent.** A control.

**Why counts are a separate operation.** They are computed from membership when asked. Nothing in
`collection_list` carries a total, and copying one there to save a round trip would make every
number it reports as old as the last write — while a rule-driven collection has no materialized
membership to have remembered in the first place. `manicule.app.service.collection_counts` pages
the same clause `collection_documents` pages, rather than issuing a second counting query, for
the same reason: a number that disagrees with the list it claims to count is worse than a slower
number.

**Why every query repeats its scope.** There is no session, and scope is read from the call's
own `collections` argument and nowhere else — so a `search` that omits it searches the **whole
workspace**, no matter how the previous call was scoped. The scope therefore travels on each
call, and comes back in `data.collections`, which is what lets a caller check that the argument
arrived rather than assuming it. A name that is not a collection here is refused with
`UnknownEntityError` and **no search runs**: a restriction that silently vanished would return the
whole workspace, ranked and plausible, which is the one failure mode worse than an error.

**Why top-`limit` absence is not corpus absence.** `search` returns the top passages of one
ranking. Finding nothing means the top of that ranking held nothing — not that the corpus does
not hold it — so the refusal a client writes has to carry the scope, the size of the collection
and the sample it took. "Nothing in *Engineering Architecture* supports this; the collection holds
3 documents and 15 chunks and the search returned 1 passage, none of which mentions the topic" is
a true sentence. "There is no such thing" is not, and nothing about the result distinguishes them.

**The recipe is bounded, and the bound is measured rather than assumed.** Each run records the
number of searches, the passages asked for, the passages returned, the passages left after
deduplication by document and heading, the serialized bytes of every MCP result, and an estimated
generator-token contribution where this machine has a BPE vocabulary — `None` where it does not,
because an unmeasured run reported as zero is an unmeasured run inside every budget. The ceilings
are declared before the run, not read off it.

**There is no summarizer between the search and the client.** The recipe assembles evidence and
hands it on; deciding what the evidence means happens outside it, where it can be seen. A
compaction step would be a component choosing what the client is allowed to read, and it would
have to be proposed explicitly, keep provenance, be replaceable, and be evaluated on its own.

Nothing in that suite depends on a hosted model, and that is deliberate: the transcript, the
accounting, the evidence and the refusal's inputs are all deterministic. Running a real model
over a `Qualification` is a useful thing to do by hand, and the record of one belongs with the
provider, the model, the reasoning setting, the date and the exact prompt — not in a test that
would go red when somebody else's sampler changed.

### 4.3 Interpreting retrieval as a client

The MCP initialization and the `search` and `ask` tool descriptions carry the following rules
because MCP is the primary unattended surface. The fields are already part of the shared payload;
the guidance adds no MCP-only result shape, and `tests/app/test_surface_parity.py` continues to
hold search envelopes byte-identical across MCP, HTTP and `--json`.

**Choose the cheapest operation that preserves the evidence.** `search` is the default when a
client needs passages, needs to test whether the corpus supports a claim, or can do its own
synthesis. `ask` spends a model call to synthesize prose and bind citations. Omitting `profile`
uses the configured profile. `fast` fetches fewer candidates and skips the reranker, so its
confidence ceiling cannot reach `high` **in the shipped definition**; `balanced` reranks a larger
set; `precise` searches the widest set and costs the most. A profile or
installation-configuration change means a different pipeline identity, and confidence from
different pipeline identities is not compared as if it were one measurement.

**Confidence is evidence strength, not answer correctness.** `confidence` and
`confidence_band` describe how strongly the retrieved passages support the query. They are not a
calibrated probability that a generated answer is correct. `none` and `low` are insufficient
support even when `hits[]` contains fluent-looking text; `medium` and `high` still require reading
the passages. `confidence_reason` is the human explanation and may be reworded, so clients branch
on the band and the fields below rather than parsing it.

For `search`, a client verifies `collections[]` before using the result and treats
`truncated: true` as a partial result: ranked candidates were dropped to fit the context budget.
For `ask`, four fields distinguish states that an empty citation list cannot:

- `corpus_consulted: false` — retrieval did not run, which is not evidence of absence.
- `ungrounded: true` — passages were found but no citation survived verification.
- `context_truncated: true` — the answer saw only the passages that fit the context budget.
- `dropped > 0` — the named number of model-emitted citations failed verification and were
  removed.

**Recovery preserves scope.** For `none` or `low`, verify the echoed scope and
`collection_counts`, then ask a more specific or meaningfully different query under that same
scope. One `precise` retry is reasonable when missing evidence matters; cycling profiles until a
number rises is not. For truncation, narrow the question or scope, or use `precise` when its cost
is acceptable. An unknown collection is resolved again with `collection_list`; it is never
retried unscoped. An `ok: false` envelope is handled from `error.hint` when present rather than by
blind query changes.

---

## 5. Payloads

Defined in `manicule.app.results` — that module is the definition, this section is the tour.

### `ask` → `AnswerResultPayload`

`expansions[]`, `conflicts[]`, `explicit_definition`, `question`, `text`, `citations[]`,
`dropped`, `confidence`, `confidence_band`, `confidence_reason`, `corpus_consulted`,
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

`expansions[]`, `conflicts[]`, `explicit_definition`, `query`, `profile`, `count`, `hits[]`,
`confidence`, `confidence_band`, `confidence_reason`, `expanded_query`, `route`, `cached`,
`truncated`, `elapsed_ms`, `collections[]`.

Each hit carries the passage, its document, its anchor, its effective `score` **and** `scores`
— the score every pipeline stage gave it. The per-stage history is kept because "reranking
helped" is only checkable while the pre-rerank score survives.

### `explicit_definition` → the `Glossed` contract

The first three fields of both payloads above are one contract, declared once on `Glossed` and
inherited by both — which is why they lead. `expansions[]` and `conflicts[]` are what the
glossary had to say about the query; `explicit_definition` is whether the result *answers* it
by showing a definition.

Three things about it are contract rather than implementation detail:

- **It is a classification, never a quantity.** It is copied from
  `Confidence.explicit_definition` ([`retrieval.md`](retrieval.md) §14.6.1) and enters no
  arithmetic, so `confidence`, `confidence_band` and the components behind them are exactly
  what they were before this field existed. `true` beside `0.00 (none)` is a normal result and
  not a contradiction: the similarity really is at the corpus's noise level, and somebody
  really did write down what the term means. Presenting the number as improved because this is
  set is the one misreading it exists to prevent.
- **It defaults to `false`, and `false` means "no claim".** A payload stored before the field
  existed parses and reports `false`; a client written before it existed parses one that
  carries it. `false` covers an ordinary use of a defined token, a term two documents disagree
  about, a defining passage that did not reach the delivered context, a directly-routed query,
  and a corpus that was never consulted — all of which are "we are not showing you a
  definition" rather than "there is none".
- **`true` always resolves to a passage.** The model refuses `explicit_definition: true`
  alongside an empty `expansions[]`, so a client can read the first expansion's `document_id`,
  `chunk_id`, `title` and `uri` without a presence check. Each expansion also carries its own
  `provenance` — the same `SourceReference` a hit or a citation carries, `null` for a document
  with no authoritative record — so the source identity of a definition is on the expansion
  rather than joined from a hit that may not be there.

`confidence_reason` says the same thing in English and remains the reader-facing explanation.
It is prose written for a person: parse the boolean, not the sentence.

### `provenance` → `SourceReference`

On every search hit, every answer citation, every glossary expansion and every document
summary — and `null` on all four unless the document carries authoritative source metadata
([`storage.md`](storage.md) §4.2.1): a locally mirrored page with a sidecar manifest, or any
connector supplying the same record.

`title`, `canonical_uri`, `source_id`, `version`, `content_type`, `modified_at` and
`section_path` describe the **publication**. `snapshot_path`, `snapshot_checksum` and
`retrieved_at` describe **this installation's copy**. `indexed_at` is neither: it is when
manicule indexed that copy.

`content_type` is the media type the **source** published, which is not always the one this
installation stored — a page served as one thing and mirrored to a file whose suffix says
another has two answers, and `DocumentSummary.media_type` is the local one. A reader
being pointed at the document needs the first group; an audit of what was actually read needs the
second; reproducing a result months later needs both — so the citation carries both rather than
choosing. `unavailable_reason` is present when a record was attempted and refused, because a
silently ignored manifest presents as a citation naming a filename, which is indistinguishable from
having written no manifest at all.

Three things about it are contract rather than implementation detail:

- **`null`, not an empty object**, for a document with no record. "There is no canonical address"
  and "the canonical address is the empty string" are different claims, and a consumer branching on
  presence should have one thing to look at.
- **The field is `provenance`, not `source`.** `source` is already on `DocumentSummary` and means
  the name of the connector that owns the document. Two senses of one word on one model is a field
  somebody reads wrong once and then builds on.
- **`title` and `uri` on the citation itself are already canonical** wherever a record exists,
  because the pipeline writes them into the columns every surface reads. This block is the
  structured form, for a consumer that needs the version it cited or the snapshot it was read from
  rather than a line to display.

`conversation_messages` replays stored citations and reports `provenance: null` on them. A stored
citation records **what was shown**, its title and URI frozen as they were at answer time; this
block reports **what is true now**. Attaching a live one to a historical record would let a
replayed conversation claim it had cited a version that did not exist yet.

### `index_path` / `connector_sync` / `import` → `IngestReport`

`connector`, `discovered`, `ingested`, `skipped`, `failed`, `expanded`, `by_status`, `error`,
`outcome`, `enumeration_completed`, `watermark_advanced`, `retry_required`,
`intentionally_bounded`, `unrecorded`, `incomplete_reason`, `elapsed_ms`.

`by_status` is the run's own counter table rather than a summary of it. A document that ended
`no_extractable_text` is neither an ingest nor a failure, and collapsing the two would hide
exactly the outcome that needs looking at.

`outcome` is the automation contract:

| Outcome | Envelope / CLI | Meaning |
|---|---|---|
| `complete` | `ok: true`, exit 0 | Enumeration finished and every accepted item has a durable outcome. `failed > 0` may still be present when those failures are recorded and repairable. |
| `bounded` | `ok: true`, exit 0 | `--limit` intentionally stopped a prefix. `intentionally_bounded` is true and no watermark advances. |
| `incomplete` | `ok: false`, exit 1; HTTP 503 | An unexpected enumeration or pipeline failure occurred, or an accepted document left no durable record. Retry is required. |

An incomplete envelope retains this payload in `data` and puts the same typed reason in
`error` and `data.incomplete_reason`; counters therefore survive the failure signal. The old
free-form `data.error` remains for compatibility, but new callers should branch on `ok` and
`data.outcome`, never parse it. `enumeration_completed` answers only whether the source walk
finished: it can be true while `outcome` is `incomplete` when `unrecorded > 0`.

The control socket proxies the same envelope unchanged. MCP returns it as a tool result. The
scheduler counts `incomplete` as a failure and retries on its next interval. `connector_list`
exposes `last_outcome`, `retry_required`, `last_error_type`,
`last_enumeration_completed`, and `last_watermark_advanced` from the connector's last-run
metadata, so an operator need not infer failure from a missing `last_synced_at`.

Before this contract, a cursor expiry could return `ok: true` with `data.error` populated. It
now retains the counters while failing explicitly:

```json
{
  "op": "connector_sync",
  "ok": false,
  "data": {
    "outcome": "incomplete",
    "enumeration_completed": false,
    "watermark_advanced": false,
    "retry_required": true,
    "discovered": 200,
    "ingested": 180
  },
  "error": {
    "type": "CursorExpiredError",
    "message": "the search cursor expired"
  }
}
```

### `doctor` → `Diagnosis`

`state`, `schema_version`, `manicule_version`, `checked_at` and `checks[]`, each
`{name, state, detail, facts, remedy}`. States are `ok`, `degraded`, `failing` and `unknown` —
the last is a check that could not run, which is deliberately not `ok`.

`state` is the **worst** state among the checks. `checked_at` is ISO 8601 in UTC: a health
record with no time on it cannot be told from a stale one somebody pasted. `schema_version` is
the shape of this payload and moves only when the shape does, which is what a consumer pinning
behavior actually wants — `manicule_version` and the envelope's `version` both move with every
release whether or not anything changed.

Checks: `configuration`, `transport`, `plugins`, `storage`, `permissions`, `index`, `grammars`,
`vocabularies`, `models`, and `component:<kind>:<name>` for anything already constructed.

**`name` is the stable identifier.** It is what a monitor selects on, so it is chosen once and
does not move with the wording. `detail` is the sentence a person reads and is free to be
reworded; `facts` is the same finding as data, so that nobody has to recover a number by
parsing English; `remedy` is what to do about it — a command where there is one, otherwise the
shortest actionable instruction. `remedy` is empty on a healthy check, and on one whose repair
depends on how the state was reached rather than on a step that can be named: an empty `remedy`
means manicule has nothing specific to suggest, never that the check is fine.

**These four states are the only status vocabulary manicule has**, and `--json` reports exactly
the words the terminal prints. A contract spelling its statuses differently from the human
output is a trap: somebody reads the screen, writes `error`, and matches nothing forever. A
consumer that needs the conventional triple maps `ok`→`ok`, `degraded`→`warning`,
`failing`→`error`, and `unknown`→`warning` — `unknown` has no equivalent in that triple, which
is the reason the triple is not what is emitted: "could not be measured" and "measured, fine"
are different facts and collapsing them loses the one worth acting on.

**Nothing here carries a secret, a credential, a token or an environment variable's value.**
Paths through `$HOME` are reported as `~/…`: the home directory's name is the account name and
was never the part anybody needed, while everything below it is kept so the reader can still
`cd` to what it names and paste the `chmod` back. A path outside the home directory —
`/srv/manicule` — is reported whole, because it names no account. Where a check is caused by an
environment variable, it names the **variable** and that it is set, never its contents.

### `doctor`'s exit status

`manicule doctor` exits **0 whenever it produced a diagnosis, whatever the diagnosis says**,
and this is deliberate rather than an omission. The exit status tracks the envelope's `ok`
across every operation uniformly — 0 for a result, 1 for an operation that failed, 2 for a
usage error Typer rejected. Producing a diagnosis of a broken machine *is* the operation
succeeding. Making this one command exit non-zero on a `failing` check would leave `"ok": true`
and a non-zero status disagreeing, so a script reading the envelope and a script reading `$?`
would reach opposite conclusions about the same run.

So a health gate reads the payload rather than the status, and `docs/deployment.md` §2 carries
the recipe. `doctor` still exits **1** when it could not produce a diagnosis at all — a
configuration that will not load — and **2** on a usage error.

**`doctor` builds nothing expensive.** No model runtime is loaded and no document is read, so
it is safe on an installation that is not working — which is the only time anybody runs it.

**`manicule doctor --fix` is the one exception, and it is a flag for that reason.** It performs
the repairs `doctor` knows how to perform — today two, and they are one repair against two
libraries with the same gap: seeding the declared tree-sitter grammars
([`parsing.md`](parsing.md#81-grammar-packaging-is-the-real-problem) §8.1) and the BPE
vocabularies every search measures a context with
([`retrieval.md`](retrieval.md#72-two-token-counters-and-using-the-wrong-one-is-a-category-error) §7.2), each from an offline
bundle if one is installed and from its upstream otherwise — and then reports the state that
resulted. They are the only parts of this command that write to the machine or use the network,
and the flag is passed by **the command line alone**: the MCP tool and `GET /api/v1/health`
call the report, because a diagnostic an assistant can reach should not be able to start a
download. `manicule init` runs the same repairs, which is how a fresh install ends up able to
parse code and answer a question rather than discovering at first index, or at first search,
that it cannot.

The two absences are reported at different severities on purpose. **Missing grammars are
`degraded`**: a corpus of Markdown and PDFs works perfectly without one, and a red check on a
healthy machine teaches an operator to ignore `doctor`. **A missing vocabulary is `failing`**:
every context is measured with it whatever the corpus holds, so a machine without one cannot
answer a question at all.

`stats`, `index_status` and `doctor` are deliberately thin. Trends, history and alerting
belong to Operations ([#14](https://github.com/mgd43b/manicule/issues/14)); a surface that
invented them here would be a second, weaker copy of that subsystem.

---

## 6. Where a server listens

`manicule start` serves MCP over **stdio** by default, which opens no socket at all. There is
no address to get wrong on the path everybody uses, and stdio carries the **whole** tool
surface — stdin and stdout are a pipe between one client and one process, so a write tool on it
is unreachable from a network by construction.

`--transport http` serves the **HTTP API, the browser surface and MCP together**, on one port.
`--mcp-only` serves MCP alone over that socket. Either way the address goes through
`manicule.app.bind.resolve_bind`, and a non-loopback bind needs **all three** of:

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

### 6.1 MCP over a socket carries the read-only tools only

The endpoint is `/mcp` on the same port, and a client is configured with the trailing slash:
`http://127.0.0.1:8765/mcp/`. A **path rather than a second port**, because one port is one bind
decision, one address in a plist, one firewall rule and one thing to remember; a second port
would need its own answer to every question §6 answers here, and the way that goes wrong is that
it gets a *different* answer.

**`--mcp-only` answers at that same address, directly**, which is a thing this had to be told
rather than a thing it did. The two ways of serving MCP go through different code — one is a
mount on the HTTP application, the other hands the server to the library — and left to the
library's default the second put the endpoint at `/mcp` and redirected `/mcp/` to it, the
opposite way round from the first. Both answered *something* at both addresses, so nothing
failed and a browser saw nothing; what paid for it was a client configured from this page,
sending a `POST` to `/mcp/` and getting a 307 it may not re-send a body for.
`manicule.mcp.serve.serve` now passes the path rather than accepting one, and
`tests/app/test_front_door.py` drives both ways of serving it with redirects switched off.

**Every mutating tool is absent from it.** Not refused — absent. `manicule.mcp.server` is asked
for the read-only surface, and it never calls `@mcp.tool` for a tool whose `readOnlyHint` is not
true, so there is no handler behind `document_delete`, `connector_sync`, `config_set`,
`plugin_add`, `index_path`, `ask` or the nine collection verbs on that server object. A call to
one is an unknown tool.

That is the same guarantee `tests/api/test_routes.py` keeps for the HTTP route table, kept the
same way and asserted in the same file: `ABSENT` names the operations with no route, and
`ABSENT_TOOLS` names the tools with no registration, each with the reason it is absent. Both
lists are held to being complete rather than merely true, so a tool added tomorrow lands in one
of the two or fails a test.

**The classification is the one already at the registrations** — the four hints of §4.1, decided
from behavior and checked against it by `tests/mcp/test_annotations.py`. There is no second
table of "tools a socket may carry", and no setting that grants an exception: a structural
guarantee traded for a configuration one is a guarantee that fails silently. The server also
*says* so — the read-only surface's instructions tell a client the write tools are not there and
where they are — so "I cannot do that" is available before a turn is spent discovering it.

**Nothing about a call outlives it.** The mount is stateless and answers with JSON rather than an
event stream, so there is no session identifier, no server-side session table, and no connection
a client can hold open. Two clients cannot see each other's state because there is no state to
see; what they share is the process — one `Runtime`, one pipeline, one session vault, one
schedule — which is right, because each of those is a fact about the process rather than about a
caller. `tests/api/test_both_surfaces.py` drives two clients at once over a real socket and
proves each is answered with what it asked for.

**Write operations are reachable where a person is present**: at the command line, over stdio,
and over the control socket of `docs/deployment.md` §6.1. A write over the network is out of
scope rather than unimplemented — it is its own decision with its own threat model.

### 6.2 Stopping it

Four things stop, in this order, because each step's work is what the next must not interrupt:

1. **the scheduler**, so no new sync starts — canceling a loop cancels the sync it was inside;
2. **the ingest stages** of whatever was running, which drain within `ingest.shutdown_grace_s`;
3. **the control socket**, which waits for the write commands already in flight;
4. **the MCP sessions and the HTTP server**, together and last, because one lifespan owns both.

A second interrupt stops waiting. Each step announces itself on stderr, because a stop can take
as long as the grace window plus whatever a proxied command is still doing — which is exactly
the interval in which somebody reaches for `kill -9` and gets the half-written index the grace
window exists to prevent.

**manicule handles the signal itself, and that is a change rather than a detail.** uvicorn
captures `SIGINT` and `SIGTERM` for the length of `serve()`, and on the way out it restores the
previous handler and re-raises — so the transport shuts down *first* and the three steps above
run afterwards, in whatever order is left. `manicule.api.serve.Server` overrides one method to
take the signals back. `tests/app/test_shutdown.py` asserts the order from outside a real
process that was sent a real `SIGTERM`; reverting that override turns it red.

### 6.3 What answers at `/`

The address the server prints is the address somebody opens, and it used to be a 404 — told by
the process they had just started, at the address it had just named. Everything is at a path:
the browser surface at `/ui`, the JSON API under `/api/v1`, MCP at `/mcp/`. Finding the front
door meant already knowing the layout, which is the one thing a front door exists to remove.

**On a whole server, `/` redirects to `/ui`.** A redirect rather than the dashboard served at
two paths, so what ends up in the address bar is somewhere real — bookmarkable, linkable,
reloadable. It is **307**, and temporary is the decision rather than a detail: a browser caches
a permanent redirect and goes on honoring it long after the server stopped sending one, with no
way to reach out and clear it. MCP moved onto this port in #143; `/` pointing at `/ui` is a
default, not a promise. The `Location` is relative, so it names this server's own path rather
than a value read out of the `Host` header, and the query string is carried across rather than
dropped.

**With no browser surface, `/` says so and names what is served.** `--no-web` and `--mcp-only`
each remove the thing a redirect would point at, and **redirecting to a surface an operator
switched off would be worse than the 404 it replaces** — it spends a second request to reach the
same answer, having first claimed the thing was somewhere. So each answers 200 with plain text:
what this process is serving, at absolute addresses built from the address the request arrived
on, and the flag that is suppressing the pages. `--mcp-only` is the one that matters most,
because its operator's next move is to paste an address into a client's configuration and the
bare address is not the one that works.

The wording is `manicule.app.frontdoor`, which is neither surface's, because `manicule.api.app`
builds a FastAPI application and `manicule.mcp.serve` may not load FastAPI at all — stdio is the
default transport and an editor spawning it pays for every import. `tests/app/test_front_door.py`
covers each of the three modes by name.

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
skip — into `<data_dir>-backups/pre-upgrade-<unix-seconds>`, names that path in its payload,
and reports the exact command. A failure part-way through an install leaves the installation
holding your index broken, and that is not a state to reach from a command that reads like a
version bump.

Both are design decisions rather than omissions, and both are stated in the output rather than
left to be discovered.

---

## 9. The HTTP API

`manicule.api`. Twelve route groups, `manicule start --transport http`, OpenAPI at
`/api/docs`. Every route parses a request, calls one service method, and renders the envelope
above — except the twelfth, which is the MCP endpoint of §6.1 and speaks its own protocol.

### 9.1 The groups

| Group | Routes |
|---|---|
| health | `GET /healthz`, `GET /readyz`, `GET /api/v1/health`, `GET /api/v1/stats`, `GET /api/v1/workspaces` |
| documents | `GET /api/v1/documents`, `GET /api/v1/documents/{id}`, `DELETE /api/v1/documents/{id}`, `GET /api/v1/documents/trash`, `POST /api/v1/documents/{id}/restore`, `POST /api/v1/documents/{id}/reindex`, `GET /api/v1/search` |
| chat | `POST /api/v1/chat`, `POST /api/v1/chat/stream`, `POST /api/v1/chat/feedback` |
| conversations | `GET`/`POST /api/v1/conversations`, `GET /api/v1/conversations/{id}/messages`, `PATCH`/`DELETE /api/v1/conversations/{id}`, `POST`/`DELETE /api/v1/conversations/{id}/share`, `GET /shared/{token}` |
| collections | `GET`/`POST /api/v1/collections`, `PATCH`/`DELETE /api/v1/collections/{id}`, `POST /api/v1/collections/{id}/name`, `GET /api/v1/collections/{id}/counts`, `GET /api/v1/collections/{id}/documents`, `POST`/`DELETE /api/v1/collections/{id}/documents/{docId}` |
| tags | `GET`/`POST /api/v1/tags`, `DELETE /api/v1/tags/{id}`, `POST`/`DELETE /api/v1/documents/{docId}/tags/{tagId}` |
| admin | `GET /api/v1/admin/stats`, `/reembed/{run_id}`, `/query-logs`, `/audit-logs`, `/search-quality`, `/plugins`, `/connectors`, `POST /api/v1/admin/connectors/{name}/sync` |
| plugins | `GET /api/v1/plugins`, `GET /api/v1/plugins/search`, `POST`/`DELETE /api/v1/plugins/{name}` |
| auth | `GET /auth/providers`, `GET /auth/session`, `GET`/`POST /api/v1/auth/keys`, `DELETE /api/v1/auth/keys/{nameOrId}` |
| workbench | `GET /api/v1/workbench?document_id=…` |
| websocket chat | `WS /api/v1/chat/ws` |
| mcp | `POST /mcp/` — the read-only tool surface of §6.1 |

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
| `reset-index` | Resets derived search state while retaining durable source and history roots |
| `backup` / `restore` | One writes wherever the caller names; the other overwrites the live data directory |
| `import` / `export` | The same, over a corpus archive |
| `upgrade` | Fetching and executing code |
| plugin *install* | The same. `POST /plugins/{name}` enables one already installed |
| document *upload* | An ingest path with no filesystem permission check and no path the operator chose |
| creating a connector | A connector holds credentials and reaches a remote system. Sources are declared in configuration, where the whole set is reviewable in one place |
| a benchmark endpoint | A benchmark on request is one HTTP call away from an unusable installation |
| `config get` / `config set` | Reading and writing configuration over the network is how an installation gets repointed at a different data directory |
| `reembed plan/start/execute/abandon/cleanup` | A full-corpus snapshot or accelerator/disk migration is an unbounded local operator action; only aggregate status is remotely readable |

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
