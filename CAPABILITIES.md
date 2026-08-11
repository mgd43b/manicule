# Capabilities

What a user can do, extracted from the OpenDocuments source rather than summarised. This is the capability floor manicule should meet or deliberately decline.

Tick when built. Strike through when dropped, with a reason. These are user-visible capabilities — *how* they are implemented is open, and in several cases should differ.

Regenerate:

```bash
uv run tools/extract_surface.py ../OpenDocuments > CAPABILITIES.md
```

| Area | Items | Ticket |
|---|---:|---|
| CLI | 48 | #8 — **built** |
| MCP tools | 20 | #8 — **built** |
| HTTP endpoints | 52 | #11 |
| File types | 18 | #4 |
| Settings | 40 | #1 |
| **Total** | **178** | |

## CLI — 48

Ticket: #8 — **built.** Nineteen commands: `ask` `search` `index` `document` `connector`
`workspace` `auth` `plugin` `config` `backup` `export` `import` `reset-index` `doctor` `init`
`start` `stop` `upgrade` `completion`. Several of the bare verbs below are subcommands of one
of those, so the mapping is noted where it is not obvious. The output shape is a contract:
[`docs/surfaces.md`](docs/surfaces.md).

- [x] `add <name>` → `plugin add`
- [ ] `add <type>` — adding a connector. Sources are declared in configuration, where the
  whole set is reviewable in one place; `connector sync` runs one. A command that wrote a
  source into the config file would be a second way to edit it
- [x] `auth <name>` → the `auth` group
- [ ] `create <name>` — collections. #10
- [x] `create-key` → `auth create-key`
- [x] `delete <id>` → `document delete`
- [ ] `delete <name>` — collections. #10
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
- [ ] `test` — dialling a connector without ingesting. #9
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
- `document reindex` — re-parse from retained bytes, with no network and no re-crawl.
- `plugin list --registry` — browse the community listing, when configuration allows it.
- `start --transport` and `start --allow-public-bind` — the second is the only way to bind a
  non-loopback address, and no configuration file can supply it.
- `ask --repl` — the interactive prompt, which is also what `ask` with no question does at a
  terminal.

## MCP tools — 20

Ticket: #8 — **built.** Nineteen tools over the same application service the command line
calls, registered with FastMCP decorators. Names are unprefixed: an MCP client namespaces by
server, so a prefix would be the server's name written twice.

- [x] `ask`
- [x] `config_get`
- [x] `config_set`
- [x] `connector_list`
- [x] `connector_sync`
- [x] `doctor`
- [x] `document_delete`
- [x] `document_get`
- [x] `document_list`
- [x] `document_reindex`
- [x] `index_path`
- [x] `index_status`
- [x] `plugin_add`
- [x] `plugin_list`
- [x] `plugin_remove`
- [x] `search`
- [x] `stats`
- [x] `workspace_list`
- [x] `workspace_switch`
- [ ] ~~a catch-all tool~~ — not built. A tool whose arguments decide which operation runs is
  a tool no client can describe, and the description is how an assistant knows when to call it.

**Deliberately not tools.** `reset-index`, `backup`, `restore`, `import`, `upgrade`, `start`,
`stop` and the `auth` verbs are command-line only. Each destroys data, mints a credential or
changes what the installation is, and a surface called unattended should not be able to do any
of that.

## HTTP endpoints — 52

Ticket: #11

- [ ] `DELETE /api/v1/collections/:id/documents/:docId`
- [ ] `DELETE /api/v1/collections/:id`
- [ ] `DELETE /api/v1/conversations/:id`
- [ ] `DELETE /api/v1/documents/:docId/tags/:tagId`
- [ ] `DELETE /api/v1/documents/:id`
- [ ] `DELETE /api/v1/plugins/:name`
- [ ] `DELETE /api/v1/tags/:id`
- [ ] `GET    /api/v1/admin/audit-logs`
- [ ] `GET    /api/v1/admin/benchmark`
- [ ] `GET    /api/v1/admin/connectors`
- [ ] `GET    /api/v1/admin/plugins`
- [ ] `GET    /api/v1/admin/query-logs`
- [ ] `GET    /api/v1/admin/search-quality`
- [ ] `GET    /api/v1/admin/stats`
- [ ] `GET    /api/v1/collections/:id/documents`
- [ ] `GET    /api/v1/collections`
- [ ] `GET    /api/v1/conversations/:id/messages`
- [ ] `GET    /api/v1/conversations`
- [ ] `GET    /api/v1/documents/:id`
- [ ] `GET    /api/v1/documents/trash`
- [ ] `GET    /api/v1/documents`
- [ ] `GET    /api/v1/health`
- [ ] `GET    /api/v1/healthz`
- [ ] `GET    /api/v1/plugins/search`
- [ ] `GET    /api/v1/plugins`
- [ ] `GET    /api/v1/readyz`
- [ ] `GET    /api/v1/stats`
- [ ] `GET    /api/v1/tags`
- [ ] `GET    /api/v1/workbench`
- [ ] `GET    /api/v1/workspaces`
- [ ] `GET    /auth/callback/:provider`
- [ ] `GET    /auth/login/:provider`
- [ ] `GET    /auth/providers`
- [ ] `PATCH  /api/v1/conversations/:id`
- [ ] `POST   /api/v1/admin/connectors/:name/sync`
- [ ] `POST   /api/v1/admin/connectors/:type`
- [ ] `POST   /api/v1/admin/connectors/github/sync`
- [ ] `POST   /api/v1/admin/connectors/github`
- [ ] `POST   /api/v1/chat/feedback`
- [ ] `POST   /api/v1/chat/stream`
- [ ] `POST   /api/v1/chat`
- [ ] `POST   /api/v1/collections/:id/documents/:docId`
- [ ] `POST   /api/v1/collections`
- [ ] `POST   /api/v1/conversations/:id/share`
- [ ] `POST   /api/v1/conversations`
- [ ] `POST   /api/v1/documents/:docId/tags/:tagId`
- [ ] `POST   /api/v1/documents/:id/restore`
- [ ] `POST   /api/v1/documents/upload`
- [ ] `POST   /api/v1/plugins/install`
- [ ] `POST   /api/v1/tags`
- [ ] `POST   /auth/logout`
- [ ] `POST   /auth/session`

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
- [x] ~~`5`~~ — not a setting. An extraction artefact: the regex matched `5:` inside the
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
- `rag.chunker`, `rag.pipeline`, `rag.reranker` — a retrieval pipeline is a declared list of
  stages, so two pipelines can be compared by configuration rather than by editing code.
- `plugins.config` — per-component settings, validated against each component's own model.
- `embedding.cache_entries` — keyed by model identity, so a model change can never serve
  vectors made by the previous one.
- `llm.generator` — which registered generator **component** to build, separate from
  `llm.provider`, which names the **vendor**. The two answer different questions and
  conflating them made the default configuration unrunnable: one implementation reaches every
  vendor through a `base_url`, so the component is not a function of the vendor.
