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

Ticket: #1

- [ ] `5`
- [ ] `allowedEndpoints`
- [ ] `allowedOrigins`
- [ ] `audit`
- [ ] `auth`
- [ ] `autoRedact`
- [ ] `cloudAllowed`
- [ ] `connectors`
- [ ] `dataDir`
- [ ] `dataPolicy`
- [ ] `db`
- [ ] `destination`
- [ ] `embedding`
- [ ] `events`
- [ ] `llm`
- [ ] `localOnly`
- [ ] `locale`
- [ ] `method`
- [ ] `mode`
- [ ] `model`
- [ ] `parserFallbacks`
- [ ] `patterns`
- [ ] `plugins`
- [ ] `profile`
- [ ] `provider`
- [ ] `providers`
- [ ] `rag`
- [ ] `replacement`
- [ ] `security`
- [ ] `sourceRestrictions`
- [ ] `storage`
- [ ] `telemetry`
- [ ] `theme`
- [ ] `transport`
- [ ] `ui`
- [ ] `vectorDb`
- [ ] `webhooks`
- [ ] `widgetAllowedDomains`
- [ ] `workspaceOverrides`
- [ ] `workspace`
