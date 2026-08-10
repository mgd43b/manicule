# manicule

A port of [OpenDocuments](https://github.com/joungminsung/OpenDocuments) to Python.

OpenDocuments works and runs daily on real documents. The architecture is proven, so it
is not being redesigned — manicule copies the feature set and changes the stack.

**Target:** ~33,000 lines of TypeScript across 5 packages and 21 plugins.

---

## What it does

Enumerated from the source, not the README, which overstates in places.

### Connectors — 8

GitHub (PAT, polling/webhook) · Notion (integration token) · Google Drive (OAuth or
service account) · **Confluence** (API token + email) · S3/GCS (cloud credentials) ·
Swagger/OpenAPI (public specs) · Web crawler (optional cookies/headers) · Web search
(Tavily, at query time)

Plus local filesystem with watch mode, and web upload.

### Parsers — 12

PDF · DOCX · XLSX/CSV · HTML · Jupyter · email · source code · PPTX · Markdown ·
plain text · structured (JSON/YAML/TOML) · archive

With fallback chains — if one parser fails the next tries.

### Model providers — 5

Ollama · OpenAI · Anthropic · Google · xAI, plus generic OpenAI-compatible endpoints.

### Ingest

Discover → fetch → parse → chunk → embed → store. Middleware hooks at every stage,
content-hash dedup, per-document error status that never aborts a batch.

### Document management

Collections · tags · versioning · chunk relations · soft delete with trash and restore ·
single-document reindex.

### Retrieval

Hybrid dense + BM25 full-text, merged with reciprocal rank fusion, then reranked.
Confidence scoring, context-window fitting, three profiles (fast/balanced/precise).

OpenDocuments also has HyDE, multi-query expansion, query decomposition, intent
classification, cross-lingual expansion, parent-document retrieval, propositions,
cross-encoder rerank, prompt compression and a hallucination guard. Its evaluation
harness scores at chance, so none of these are known to help. Port the proven path;
treat the rest as optional.

### Chat

Streaming answers with citations and confidence · multi-turn memory · shared
conversation links · feedback capture.

### CLI — 19 commands

`ask` `search` `index` `document` `connector` `workspace` `auth` `plugin` `config`
`backup` `export` `import` `reset-index` `doctor` `init` `start` `stop` `upgrade`
`completion` — with interactive REPL, stdin piping and `--json`.

### MCP server — 19 tools

ask · search · index_path · document_list · document_get · document_delete ·
document_reindex · index_status · stats · doctor · connector_list · connector_sync ·
config_get · config_set · workspace_list · workspace_switch · plugin_list · plugin_add ·
plugin_remove

### HTTP API — 11 route groups

health · documents · chat · conversations · collections · tags · admin · plugins ·
auth · workbench · websocket chat. Plus an embeddable chat widget.

### Web UI — 12 areas

chat · documents · collections · connectors · dashboard · health · plugins · settings ·
workspaces · admin · auth · layout. Command palette, keyboard nav, dark mode.

### Team mode

API keys (hashed, scoped, expiring) · roles (admin/member/viewer) · rate limiting ·
OAuth SSO via Google and GitHub · workspace isolation enforced on every query ·
audit log · security alerts · PII redaction.

### Operations

Backup and restore · export/import · `doctor` diagnostics · telemetry and query logs ·
event bus with webhook dispatch · degraded-mode warnings · update checks.

### Plugin system

Four kinds — parser, connector, model, middleware — with lifecycle hooks, dependency
and conflict declarations, config schemas, and a scaffolding command.

### Data model — 16 tables

`documents` `chunk_relations` `collections` `collection_documents` `tags`
`document_tags` `document_versions` `connectors` `conversations` `messages`
`workspaces` `workspace_members` `api_keys` `audit_logs` `query_logs` `plugins`

---

## What changes

| | OpenDocuments | manicule | Why |
|---|---|---|---|
| Language | TypeScript | Python | The retrieval and document ecosystem lives here |
| PDF parsing | `pdf-parse` | layout-aware parser | `pdf-parse` gives a text blob with no real page boundaries — the cause of the citation bug below |
| Embeddings | Ollama / cloud | MLX on Apple Silicon | Native, fast, local |
| Vectors | LanceDB | LanceDB | Already the right choice |
| Metadata | SQLite | SQLite, or fold into LanceDB | Open — decide when building storage |
| Plugins | npm packages | Python entry points | Same model, native packaging |

---

## Defects to fix on the way

Found while reviewing. Real, and worth not reproducing.

1. **PDF citations point at pages that do not exist.** Extracted text is split on
   blank-line runs and the fragments are numbered as pages. Citations are the product,
   so this is the one that matters most. *A citation carries a correct location, or
   none.*
2. **No auth by default, on every interface.** Personal mode has no authentication and
   binds all interfaces; the documented Docker deployment publishes it. Bind loopback
   by default.
3. **Plugin permissions are enforced nowhere.** A typed `permissions` field with passing
   tests and no runtime check. Enforce it or drop the field — an unenforced guarantee is
   worse than none.
4. **PII redaction does something other than advertised.** It runs at ingest,
   permanently destroying data in the index, while the docs claim it protects data sent
   to cloud models. Pick one behaviour, build that, document that.

---

## Build order

| | | Notes |
|---|---|---|
| 1 | Protocols and wiring | Four plugin kinds. Everything plugs into this |
| 2 | Storage and data model | 16 tables, migrations, vectors |
| 3 | Embedding | Fixes vector dimensionality — settle before the schema exists |
| 4 | Parsers and chunking | The one real upgrade |
| 5 | Ingest pipeline | Middleware, fallbacks, dedup |
| 6 | Retrieval | Dense + BM25 + fusion + rerank |
| 7 | Generation and chat | Streaming, citations, multi-turn |
| 8 | **MCP and CLI** | Ship early — daily use surfaces problems no test will |
| 9 | Connectors | All eight |
| 10 | Document management | Collections, tags, versions |
| 11 | Model providers | All five |
| 12 | HTTP API and web UI | |
| 13 | Team mode | Auth, workspaces, audit, redaction |
| 14 | Operations | Backup, export, doctor, telemetry |
| 15 | Parity check | Does the port answer as well as the original, on the same corpus? |

Steps 3 and 4 come before 2 finishes: the storage schema fixes vector dimensionality and
assumes a chunk size, and changing either later means re-embedding everything.
