# Capabilities

What a user can do, extracted from the OpenDocuments source rather than summarized. This is the capability floor manicule should meet or deliberately decline.

Tick when built. Strike through when dropped, with a reason. These are user-visible capabilities — *how* they are implemented is open, and in several cases should differ.

Regenerate:

```bash
uv run tools/extract_surface.py ../OpenDocuments > CAPABILITIES.md
```

| Area | Items | Ticket |
|---|---:|---|
| CLI capability items | 48 | #8 — **built** |
| MCP capability items | 44 | #8 — **built** |
| HTTP endpoints | 62 | #11 — **built** |
| File types | 18 | #4 |
| Settings | 40 | #1 |
| **Total** | **212** | |

## CLI — 48

Ticket: #8 — **built.** The 48 rows below are upstream capability items, not the current command
count. Manicule exposes 31 top-level commands; `manicule --help` and the asserted inventory in
[`docs/surfaces.md`](docs/surfaces.md) are authoritative. Several bare verbs below are subcommands,
so the mapping is noted where it is not obvious. The output shape is also a contract there.

- [x] `add <name>` → `plugin add`
- [ ] `add <type>` — adding a connector. Sources are declared in configuration, where the
  whole set is reviewable in one place; `connector sync` runs one. A command that wrote a
  source into the config file would be a second way to edit it
- [x] `auth <name>` → the `auth` group
- [x] `create <name>` → `collection create`
- [x] `create-key` → `auth create-key`
- [x] `delete <id>` → `document delete`
- [x] `delete <name>` → `collection delete`
- [ ] ~~`dev`~~ — a development server. There is one way to start manicule and it is `start`
- [ ] ~~`edit`~~ — opening the config file in `$EDITOR`. `config set` validates the whole tree
  before writing, and an editor that saves an invalid file has already saved it
- [x] `get <id>` → `document get`
- [ ] ~~`install`~~ — installing a plugin. A plugin runs with this process's full authority,
  so manicule reports the command that would install one and runs nothing. `docs/surfaces.md` §8
- [x] `list-keys` → `auth list-keys`
- [x] `list` → `document list`, `connector list`, `workspace list`, `plugin list`
- [ ] `login` — OAuth. #13
- [ ] ~~`publish`~~ — publishing a plugin. That is a package index's job
- [x] `remove <name>` → `plugin remove`
- [x] `reset` → `reset-index`
- [ ] `restore <id>` — restoring a document from the trash. #10
- [x] `revoke-key <nameOrId>` → `auth revoke-key`
- [x] `search <query>`
- [x] `show` → `config show`
- [x] `status` → `index` with no path, and `doctor`
- [x] `switch <name>` → `workspace switch`
- [x] `sync [name]` → `connector sync`
- [ ] `test` — dialing a connector without ingesting. #9
- [ ] `trash` — listing the trash. #10
- [ ] `update [name]` — editing a connector. See `add <type>`
- [x] ask — option `--json` — a **global** option, so every command has it
- [x] ask — option `--profile <profile>`
- [ ] ~~ask — option `--stdin`~~ — no flag needed. An argument that was not given and a pipe
  that is attached are unambiguous, so `echo "…" | manicule ask` just works
- [x] auth — option `--role <role>`
- [x] backup — option `--force`
- [x] backup — option `-o, --output <path>`
- [x] completion — option `--shell <shell>`
- [x] export-cmd — option `--output <path>`
- [x] import-cmd — option `--force`
- [x] index-cmd — option `--reindex`
- [x] index-cmd — option `--watch`
- [ ] plugin — option `--type <type>` — filtering the registry by component kind. The registry
  itself lands with the fetcher; filtering an empty list is not a feature
- [x] reset-index — option `--yes`
- [x] search — option `--top <n>`
- [x] search — option `--type <type>`
- [x] start — option `--mcp-only`
- [x] start — option `--no-web`
- [x] start — option `-p, --port <port>`
- [x] upgrade — option `--skip-backup`
- [x] upgrade — option `--version <version>`
- [ ] workspace — option `--mode <mode>` — personal or team. #13 owns team mode

### Added, because the commands above could not do their job without them

- `--workspace/-w` — run one command in another workspace, without editing configuration.
- `index --stats` — counts grouped by source, media type and status.
- `index --source` — the source name documents are recorded under. It is part of their
  identity, so it is a decision rather than a constant.
- `document reindex` — re-parse from retained bytes, with no network and no re-crawl. With
  `--stale` instead of an id, every document an installed parser has moved past. With
  `--stale-glossary`, every document whose definitions came out of a detector this build has
  since changed — read back from the chunks already stored, so it runs no parser, fetches
  nothing and produces no vector.
- `plugin list --registry` — browse the community listing, when configuration allows it.
- `start --transport` and `start --allow-public-bind` — the second is the only way to bind a
  non-loopback address, and no configuration file can supply it.
- `ask --repl` — the interactive prompt, which is also what `ask` with no question does at a
  terminal.
- `reset-derived`, `cleanup-derived-generations`, `release-source-history`, and
  `snapshot-delete` — distinct lifecycle boundaries; the last requires a token from its
  aggregate dry run before it releases authoritative snapshot ownership.
- `connector sync --acquire-only`, `connector snapshot`, `connector verify`, and `rebuild
  plan/execute/resume/status` — a durable retained-source hand-off, aggregate verification, and
  connector-free derived publication with no separate settlement command.

## MCP tools — 44

Ticket: #8 — **built.** Forty-three implemented tools over the same application service the command
line calls, registered with FastMCP decorators. Names are unprefixed: an MCP client namespaces by
server, so a prefix would be the server's name written twice.

The checklist below has forty-four capability items because it also records the deliberately
declined catch-all `run-command` surface; that item is not a registered MCP tool.

**All forty-three over stdio; the twenty-six read-only ones over a socket.** MCP is also served
at `/mcp/` on the HTTP port, and every mutating tool is *absent* from that surface rather than
refused on it — see [`docs/surfaces.md`](docs/surfaces.md) §6.1. Over stdio the write tools are
unreachable from a network by construction, and a socket has to replace that property rather
than assume it.

- [x] `ask`
- [x] `collection_add`
- [x] `collection_counts`
- [x] `collection_create`
- [x] `collection_delete`
- [x] `collection_documents`
- [x] `collection_list`
- [x] `collection_remove`
- [x] `collection_rename`
- [x] `collection_update`
- [x] `collection_rule_show`
- [x] `collection_rule_set`
- [x] `collection_rule_clear`
- [x] `config_get`
- [x] `config_set`
- [x] `connector_list`
- [x] `connector_sync`
- [x] `doctor`
- [x] `document_delete`
- [x] `document_get`
- [x] `document_resolve`
- [x] `document_list`
- [x] `document_reindex`
- [x] `index_path`
- [x] `index_status`
- [x] `lifecycle_cleanup_generations`
- [x] `lifecycle_delete_snapshot`
- [x] `lifecycle_release_history`
- [x] `lifecycle_reset_derived`
- [x] `reembed_status`
- [x] `rebuild_plan`
- [x] `rebuild_status`
- [x] `plugin_add`
- [x] `plugin_list`
- [x] `plugin_remove`
- [x] `search`
- [x] `snapshot_status`
- [x] `snapshot_verify`
- [x] `stats`
- [x] `vector_checksum`
- [x] `vector_index_build`
- [x] `workspace_list`
- [x] `workspace_switch`
- [ ] ~~a catch-all tool~~ — not built. A tool whose arguments decide which operation runs is
  a tool no client can describe, and the description is how an assistant knows when to call it.

**Deliberately not tools.** `reset-index`, `backup`, `restore`, `import`, `upgrade`, `start`,
`stop`, `collection orphans`, `sweep-vectors`, rebuild execution/resume and the `auth` verbs are
command-line only. Each destroys data, mints a credential, changes what the installation is or
starts wide derived publication work, and a surface called unattended should not be able to do
any of that. `sweep-vectors` purges soft-deleted documents whose grace period has expired, after
which restoring one costs a re-parse rather than being free — and it needs no assistant, because
a served installation already runs it on a schedule.
`collection orphans` moves every document outside every collection into the trash, which in a
corpus where collections are optional is most of it.

## HTTP endpoints — 62

Ticket: #11 — **built.** Twelve route groups over the same application service the CLI and the
MCP server use: health · documents · chat · conversations · collections · tags · admin ·
plugins · auth · workbench · websocket chat · mcp, plus an embeddable widget. Every response is
the envelope `--json` prints; the reference is [`docs/surfaces.md`](docs/surfaces.md) §9. The
twelfth is the mounted MCP endpoint, which speaks its own protocol rather than returning
envelopes over routes.

Struck-through entries are **deliberately absent**, each for a stated reason. This surface is
the one an unattended caller reaches, so an operation that destroys data, opens an ingest path
or executes code stays on the command line. `tests/api/test_routes.py` asserts each absence by
name — an absence with no test is an absence that comes back.

- [x] `DELETE /api/v1/collections/:id/documents/:docId`
- [x] `DELETE /api/v1/collections/:id`
- [x] `DELETE /api/v1/conversations/:id`
- [x] `DELETE /api/v1/documents/:docId/tags/:tagId`
- [x] `DELETE /api/v1/documents/:id`
- [x] `DELETE /api/v1/plugins/:name`
- [x] `DELETE /api/v1/tags/:id`
- [x] `GET    /api/v1/admin/audit-logs`
- [ ] ~~`GET    /api/v1/admin/benchmark`~~ — a benchmark run on request, from a surface an unattended caller reaches, is one HTTP call away from an unusable installation
- [x] `GET    /api/v1/admin/connectors`
- [x] `GET    /api/v1/admin/lifecycle/derived-generations` — aggregate dry run only
- [x] `GET    /api/v1/admin/lifecycle/reset-derived` — aggregate dry run only
- [x] `GET    /api/v1/admin/lifecycle/snapshots/:runId` — aggregate impact and token; no delete
- [x] `GET    /api/v1/admin/lifecycle/source-history` — aggregate dry run only
- [x] `GET    /api/v1/admin/plugins`
- [x] `GET    /api/v1/admin/query-logs`
- [x] `GET    /api/v1/admin/search-quality`
- [x] `GET    /api/v1/admin/stats`
- [x] `GET    /api/v1/admin/vector-checksum` — aggregate vector numerical-integrity coverage
- [x] `GET    /api/v1/admin/vector-index` — aggregate ANN index dry run only
- [x] `GET    /api/v1/collections/:id/counts`
- [x] `GET    /api/v1/collections/:id/documents`
- [x] `GET    /api/v1/collections`
- [x] `GET    /api/v1/conversations/:id/messages`
- [x] `GET    /api/v1/conversations`
- [x] `GET    /api/v1/documents/:id`
- [x] `GET    /api/v1/documents/trash`
- [x] `GET    /api/v1/documents`
- [x] `GET    /api/v1/health`
- [x] `GET    /api/v1/healthz` — served at `/healthz`, outside the versioned prefix. A liveness
  probe answers a process supervisor rather than a person, and it must not move when the API
  version does
- [x] `GET    /api/v1/plugins/search`
- [x] `GET    /api/v1/plugins`
- [x] `GET    /api/v1/readyz` — served at `/readyz`, for the same reason. It asks the store
  whether the index is usable, which `/healthz` deliberately does not
- [x] `GET    /api/v1/stats`
- [x] `GET    /api/v1/tags`
- [x] `GET    /api/v1/workbench`
- [x] `GET    /api/v1/workspaces`
- [ ] ~~`GET    /auth/callback/:provider`~~ — OAuth. #13
- [ ] ~~`GET    /auth/login/:provider`~~ — OAuth. #13
- [x] `GET    /auth/providers`
- [x] `PATCH  /api/v1/collections/:id` — the description. Renaming is its own route: it can
  fail with a 409 and describing cannot, and one route returning either status depending on
  which field was present is a route a caller cannot reason about
- [x] `PATCH  /api/v1/conversations/:id`
- [x] `POST   /api/v1/admin/connectors/:name/sync`
- [ ] ~~`POST   /api/v1/admin/connectors/:type`~~ — a connector holds credentials and reaches a remote system. Sources are declared in configuration, where the whole set is reviewable in one place; `POST /admin/connectors/:name/sync` runs one that is already declared
- [ ] ~~`POST   /api/v1/admin/connectors/github/sync`~~ — there is no per-connector route. `POST /admin/connectors/:name/sync` runs whichever connector configuration names
- [ ] ~~`POST   /api/v1/admin/connectors/github`~~ — see `POST /api/v1/admin/connectors/:type`
- [x] `POST   /api/v1/chat/feedback`
- [x] `POST   /api/v1/chat/stream`
- [x] `POST   /api/v1/chat`
- [x] `POST   /api/v1/collections/:id/documents/:docId`
- [x] `POST   /api/v1/collections/:id/name` — rename. No document moves and nothing is
  re-embedded
- [x] `POST   /api/v1/collections`
- [ ] ~~`POST   /api/v1/collections/orphans`~~ — deleting every document outside every
  collection. It destroys data, so it stays on the command line as `collection orphans
  --confirm`, where a person is present. `tests/api/test_routes.py` asserts the absence
- [x] `POST   /api/v1/conversations/:id/share`
- [x] `POST   /api/v1/conversations`
- [x] `POST   /api/v1/documents/:docId/tags/:tagId`
- [x] `POST   /api/v1/documents/:id/restore`
- [ ] ~~`POST   /api/v1/documents/upload`~~ — accepting bytes over HTTP is an ingest path with no filesystem permission check and no path the operator chose. `manicule index <path>` is the ingest this build offers
- [ ] ~~`POST   /api/v1/plugins/install`~~ — installing a plugin fetches and executes code with this process's full authority. `POST /api/v1/plugins/:name` enables one that is already installed
- [x] `POST   /api/v1/tags`
- [ ] ~~`POST   /auth/logout`~~ — there is no session cookie in this build — a key is presented on every request — so there is nothing to log out of. `GET /auth/session` reports who a request is
- [x] `POST   /auth/session` — as `GET /auth/session`, which *reports* an identity rather than
  creating one. There is no session cookie in this build: a key is presented on every request,
  and a signed cookie would be a second credential type with its own expiry, revocation and
  CSRF story

## File types — 18

Ticket: #4

- [ ] `.csv` (parser-xlsx)
- [ ] `.docx` (parser-docx)
- [ ] `.eml` (parser-email)
- [ ] `.htm` (parser-html)
- [ ] `.html` (parser-html)
- [ ] `.ipynb` (parser-jupyter)
- [ ] `.json` (core/structured)
- [ ] `.md` (core/markdown)
- [ ] `.mdx` (core/markdown)
- [ ] `.msg` (parser-email)
- [ ] `.pdf` (parser-pdf)
- [ ] `.pptx` (parser-pptx)
- [ ] `.toml` (core/structured)
- [ ] `.txt` (core/plaintext)
- [ ] `.xlsx` (parser-xlsx)
- [ ] `.yaml` (core/structured)
- [ ] `.yml` (core/structured)
- [ ] `.zip` (core/archive)

## Settings — 40

Ticket: #1 — **built.** `manicule.config.settings`, Pydantic v2 + pydantic-settings.

Every setting below has a home in the tree, with validation and a default. The flat names
the extractor produced are ambiguous in three places — `provider`, `model` and `storage`
each name two different things — so the column on the right is where each one actually
lives. Names are `snake_case`, being Python.

Sources layer rather than compete, highest priority first: values passed in code,
environment variables (`MANICULE_` prefix, `__` for nesting), `.env` files, the config file
(`manicule.toml` or `$XDG_CONFIG_HOME/manicule/config.toml`), then defaults. Setting one
field in the environment overrides that field and leaves the rest of the file in force.

Provider credentials additionally follow the conventional `<PROVIDER>_API_KEY` names, so
`LLM_PROVIDER` plus `OPENAI_API_KEY` is a complete setup with no config file at all.

- [x] `allowedEndpoints` → `security.transport.allowed_endpoints`
- [x] `allowedOrigins` → `security.transport.allowed_origins`
- [x] `audit` → `security.audit`
- [x] `auth` → `security.auth`
- [x] `autoRedact` → `security.data_policy.auto_redact`
- [x] `cloudAllowed` → `security.data_policy.source_restrictions.cloud_allowed`
- [x] `connectors` → `connectors` — by instance name, each declaring a `type`
- [x] `dataDir` → `data_dir` — top level; XDG by default, `MANICULE_DATA_DIR` overrides
- [x] `dataPolicy` → `security.data_policy`
- [x] `db` → `storage.db`
- [x] `destination` → `security.audit.destination`
- [x] `embedding` → `embedding` — its own section; embeddings and generation are separate runtimes
- [x] `events` → `events`, and `security.audit.events`
- [x] `llm` → `llm`
- [x] `localOnly` → `security.data_policy.source_restrictions.local_only`
- [x] `locale` → `ui.locale`, and top-level `locale`
- [x] `method` → `security.data_policy.auto_redact.method`
- [x] `mode` → `mode` — `personal` or `team`
- [x] `model` → `embedding.model` and `llm.model`
- [x] `parserFallbacks` → `parser_fallbacks` — keyed by media type, `*` supplies a global tail
- [x] `patterns` → `security.data_policy.auto_redact.patterns`
- [x] `plugins` → `plugins` — an allow/deny filter over what discovery found, plus per-component config
- [x] `profile` → `rag.profile` — `fast` / `balanced` / `precise`
- [x] `provider` → `embedding.provider` and `llm.provider`
- [x] `providers` → `providers` (credentials, by provider), and `security.auth.providers` (OAuth)
- [x] `rag` → `rag`
- [x] `replacement` → `security.data_policy.auto_redact.replacement`
- [x] `security` → `security`
- [x] `sourceRestrictions` → `security.data_policy.source_restrictions`
- [x] `storage` → `storage` (database and vectors), and `security.storage` (at rest)
- [x] `telemetry` → `telemetry` — off unless switched on
- [x] `theme` → `ui.theme`
- [x] `transport` → `security.transport` — binds loopback by default
- [x] `ui` → `ui`
- [x] `vectorDb` → `storage.vector_db`
- [x] `webhooks` → `events.webhooks`
- [x] `widgetAllowedDomains` → `security.transport.widget_allowed_domains`
- [x] `workspaceOverrides` → `security.data_policy.workspace_overrides`
- [x] `workspace` → `workspace`
- [x] ~~`5`~~ — not a setting. An extraction artifact: the regex matched `5:` inside the
  model name `qwen2.5:14b`.

### Deliberately absent

**No embedding dimension setting.** The dimension is a property of the model, read from the
embedder's fingerprint at run time. A configurable dimension is a value that can disagree
with the model, and when it does, the index is silently wrong.

**No alternative database or vector store in the enum.** `storage.db` is `sqlite` and
`storage.vector_db` is `lancedb`, and those are the only accepted values. Naming
alternatives in configuration advertises support that does not exist.

**No encryption-at-rest switch.** A setting that raises "not implemented" when you turn it on
is a feature list entry, not a feature.

### Added, because the settings above could not do their job without them

- `security.transport.bind_host` / `port` / `trusted_proxies` — loopback by default, and
  binding wider requires authentication to be enabled. Enforced at startup, not documented
  and hoped for.
- `storage.retain_source_bytes` — keeps fetched bytes so re-parsing never means re-fetching,
  and so an anchor can be verified against the document it was made from.
- `storage.checksum_backfill_batch` — rows one pass of the vector-checksum backfill rewrites.
  It is the bound on both the memory the pass holds and the work a crash can lose, and it
  exists as a setting because the right value depends on corpus size rather than on manicule.
- `rag.chunker`, `rag.pipeline`, `rag.reranker` — a retrieval pipeline is a declared list of
  stages, so two pipelines can be compared by configuration rather than by editing code.
- `rag.glossary.*` — glossary-aware acronym retrieval: whether ingest reads definitions out of
  documents, whether a query naming one is expanded, how confident an entry has to be before a
  query acts on it, and which extra words to treat as ordinary English. There is deliberately no
  setting that resolves a conflict between two definitions; conflicts are always reported.
- `plugins.config` — per-component settings, validated against each component's own model.
- `embedding.cache_entries` — keyed by model identity, so a model change can never serve
  vectors made by the previous one.
- `llm.generator` — which registered generator **component** to build, separate from
  `llm.provider`, which names the **vendor**. The two answer different questions and
  conflating them made the default configuration unrunnable: one implementation reaches every
  vendor through a `base_url`, so the component is not a function of the vendor.
