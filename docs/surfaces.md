# Surfaces: the CLI, the MCP server, and the shape of what they return

Two surfaces, one service, one output contract. This document says what that contract is,
because `--json` is something scripts and assistants parse, and a shape nobody wrote down is
whatever the code happened to do last.

- **The application service** (`manicule.app.service.ApplicationService`) has all the
  behaviour.
- **The command line** (`manicule.cli`) and **the MCP server** (`manicule.mcp`) are adapters
  over it. Neither decides anything.
- **The HTTP API** ([#11](https://github.com/mgd43b/manicule/issues/11)) will be a third one.

---

## 1. Why the layer exists

A rule implemented in the command line is a rule the MCP tool does not have — and the MCP tool
is the one an assistant calls unattended. Workspace scoping, credential masking, refusing to
install a plugin, refusing a wide bind: every one of those is a rule that has to hold on both
surfaces or it does not hold.

So the surfaces are thin by construction and the property is checked rather than intended.
`tests/app/test_surface_parity.py` runs the same operation through both and compares the
results. It fails the moment they stop being the same call.

### What each layer may contain

| Layer | May | May not |
|---|---|---|
| `manicule.cli` | Parse arguments, read stdin, render, set the exit status | Query a store, compute a filter, decide a policy |
| `manicule.mcp` | Declare tools, describe them, pass arguments through | Anything the CLI may not |
| `manicule.app.service` | Everything else | Import a database, a model runtime or a web framework |
| `manicule.app.runtime` | Build components, own the lifecycle | Decide anything a surface could ask about |

The service is written against the protocols in `manicule.app.ports`, so the suites drive it
against components that break their half of the bargain — a store that ignores its workspace,
a retriever that returns another tenant's chunk. That is the only way the guards can be
watched firing.

---

## 2. The envelope

Every `--json` emission and every MCP tool result is **one JSON object** in this shape:

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
prints it and exits **1**.

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

`reset_index`, `backup`, `restore`, `import`, `upgrade`, `start`, `stop` and the `auth` verbs
are command-line only. Each of them either destroys data, mints a credential, or changes what
the installation *is* — and a tool an assistant can call unattended should not be able to do
any of that. The nineteen tools read the corpus, write documents into it, and adjust
configuration. That is the whole surface.

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

Checks: `configuration`, `transport`, `plugins`, `storage`, `index`, and `component:<kind>:<name>`
for anything already constructed.

**`doctor` builds nothing expensive.** No model runtime is loaded and no document is read, so
it is safe on an installation that is not working — which is the only time anybody runs it.

`stats`, `index_status` and `doctor` are deliberately thin. Trends, history and alerting
belong to Operations ([#14](https://github.com/mgd43b/manicule/issues/14)); a surface that
invented them here would be a second, weaker copy of that subsystem.

---

## 6. Where a server listens

`manicule start` serves MCP over **stdio** by default, which opens no socket at all. There is
no address to get wrong on the path everybody uses.

`--transport http` goes through `manicule.app.bind.resolve_bind`, and a non-loopback bind
needs **all three** of:

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

## 9. Export and import

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
