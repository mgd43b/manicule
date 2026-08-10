# Capabilities

What a user can do, extracted from the OpenDocuments source rather than summarised. This is the capability floor manicule should meet or deliberately decline.

Tick when built. Strike through when dropped, with a reason. These are user-visible capabilities — *how* they are implemented is open, and in several cases should differ.

Regenerate:

```bash
uv run tools/extract_surface.py ../OpenDocuments > CAPABILITIES.md
```

| Area | Items | Ticket |
|---|---:|---|
| CLI | 48 | #8 |
| MCP tools | 20 | #8 |
| HTTP endpoints | 52 | #11 |
| File types | 18 | #4 |
| Settings | 40 | #1 |
| **Total** | **178** | |

## CLI — 48

Ticket: #8

- [ ] `add <name>`
- [ ] `add <type>`
- [ ] `auth <name>`
- [ ] `create <name>`
- [ ] `create-key`
- [ ] `delete <id>`
- [ ] `delete <name>`
- [ ] `dev`
- [ ] `edit`
- [ ] `get <id>`
- [ ] `install`
- [ ] `list-keys`
- [ ] `list`
- [ ] `login`
- [ ] `publish`
- [ ] `remove <name>`
- [ ] `reset`
- [ ] `restore <id>`
- [ ] `revoke-key <nameOrId>`
- [ ] `search <query>`
- [ ] `show`
- [ ] `status`
- [ ] `switch <name>`
- [ ] `sync [name]`
- [ ] `test`
- [ ] `trash`
- [ ] `update [name]`
- [ ] ask — option `--json`
- [ ] ask — option `--profile <profile>`
- [ ] ask — option `--stdin`
- [ ] auth — option `--role <role>`
- [ ] backup — option `--force`
- [ ] backup — option `-o, --output <path>`
- [ ] completion — option `--shell <shell>`
- [ ] export-cmd — option `--output <path>`
- [ ] import-cmd — option `--force`
- [ ] index-cmd — option `--reindex`
- [ ] index-cmd — option `--watch`
- [ ] plugin — option `--type <type>`
- [ ] reset-index — option `--yes`
- [ ] search — option `--top <n>`
- [ ] search — option `--type <type>`
- [ ] start — option `--mcp-only`
- [ ] start — option `--no-web`
- [ ] start — option `-p, --port <port>`
- [ ] upgrade — option `--skip-backup`
- [ ] upgrade — option `--version <version>`
- [ ] workspace — option `--mode <mode>`

## MCP tools — 20

Ticket: #8

- [ ] `opendocuments_ask`
- [ ] `opendocuments_config_get`
- [ ] `opendocuments_config_set`
- [ ] `opendocuments_connector_list`
- [ ] `opendocuments_connector_sync`
- [ ] `opendocuments_doctor`
- [ ] `opendocuments_document_delete`
- [ ] `opendocuments_document_get`
- [ ] `opendocuments_document_list`
- [ ] `opendocuments_document_reindex`
- [ ] `opendocuments_index_path`
- [ ] `opendocuments_index_status`
- [ ] `opendocuments_plugin_add`
- [ ] `opendocuments_plugin_list`
- [ ] `opendocuments_plugin_remove`
- [ ] `opendocuments_search`
- [ ] `opendocuments_stats`
- [ ] `opendocuments_workspace_list`
- [ ] `opendocuments_workspace_switch`
- [ ] `opendocuments`

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
