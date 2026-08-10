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

## The stack

Python 3.12+. Where a choice is a like-for-like swap it is marked *equivalent*; where the
Python option is genuinely better than what OpenDocuments has, the gain is stated.

### Foundation

| | Choice | Why |
|---|---|---|
| Packaging | **uv** | One tool for env, lockfile, and `uv tool install` for distribution — a single command install with no container. Also gives PEP 723 inline metadata for standalone scripts |
| Lint / format | **ruff** | Replaces black, isort and flake8 with one fast tool |
| Types | **pyright**, strict | Closest thing to the TypeScript strictness being left behind |
| Config & validation | **Pydantic v2** + pydantic-settings | **Gain.** OpenDocuments hand-rolls config validation. One declarative layer covers config, API models and plugin manifests |
| Tests | **pytest** + pytest-asyncio | *equivalent* |

### Storage

| | Choice | Why |
|---|---|---|
| Vectors | **LanceDB** | Keep it. Embedded, no server, ACID, versioned, same Rust core as the TypeScript binding. It was already the right call |
| Metadata | **SQLite** via SQLAlchemy 2.0 | *equivalent* — but see the coupling below |
| Migrations | **Alembic** | **Gain.** OpenDocuments hand-rolls a migration runner over eight numbered `.sql` files |
| Lexical search | **SQLite FTS5** | What OpenDocuments already uses for the BM25 leg |

**The coupling worth knowing:** the BM25 half of hybrid retrieval comes from SQLite's
FTS5. Folding metadata into LanceDB to run one store means moving BM25 to LanceDB's
Tantivy index instead. Both work; they are one decision, not two.

### Document parsing — the largest upgrade

This is where TypeScript was weakest and where the citation bug comes from.

| Format | Choice | Why |
|---|---|---|
| PDF | **pypdfium2** fast path, **docling** or **marker** optional | Real page and bounding-box provenance instead of a text blob. **Avoid PyMuPDF** — it is AGPL, which is a problem for an MIT project |
| Code | **tree-sitter** | **Biggest single gain.** Real ASTs for 40+ languages, replacing OpenDocuments' pattern-matched function and class splitting |
| DOCX | python-docx | *equivalent* |
| XLSX | **python-calamine** | Rust-backed, much faster than openpyxl |
| PPTX | python-pptx | *equivalent* |
| HTML | **trafilatura** | Built for content extraction — strips navigation and boilerplate properly |
| Jupyter | nbformat | *equivalent* |
| Email | stdlib `email` | *equivalent* |
| Markdown | markdown-it-py | *equivalent* |
| Structured | stdlib json / tomllib, PyYAML | *equivalent* |

### Models

| | Choice | Why |
|---|---|---|
| Embeddings | **MLX** (`mlx-embeddings`) | Native Apple Silicon. Read token states from the inner encoder and pool in our own numpy — the library's `last_hidden_state` field returns the *pooled* vector, and getting that wrong costs 0.856 cosine silently |
| Embedding fallback | onnxruntime | For anywhere that is not Apple Silicon |
| Generation | **litellm** | **Gain.** One dependency replaces five hand-written provider clients. Ollama, OpenAI, Anthropic, Google, xAI and OpenAI-compatible endpoints behind one call |
| Reranking | sentence-transformers CrossEncoder | *equivalent* |

### Surfaces

| | Choice | Why |
|---|---|---|
| HTTP | **FastAPI** | Async, OpenAPI generated from types, native SSE streaming, Pydantic-native |
| MCP | **FastMCP** / official `mcp` SDK | **Gain.** OpenDocuments hand-rolls a 500-line MCP server; this is decorators over functions |
| CLI | **Typer** + **Rich** | **Gain.** Typer derives the CLI from type hints. Rich replaces hand-rolled progress output and log formatting |
| Web UI | **HTMX + Jinja2 + Tailwind standalone** | **Gain.** Deletes an entire Node toolchain. The surface is search, streaming chat, and lists — Tailwind ships standalone binaries needing no npm. Alternative: keep a React SPA if the chat UX needs it |
| Auth | **authlib** + itsdangerous | *equivalent* — OAuth for Google/GitHub, signed session cookies |
| Scheduling | **apscheduler** | Connector sync polling. No broker, unlike Celery |

### What this deletes

Eight subsystems OpenDocuments wrote by hand that become dependencies or decorators:

- five model provider clients → **litellm**
- a migration runner → **Alembic**
- config schema validation → **Pydantic**
- an MCP protocol server → **FastMCP**
- CLI argument parsing and terminal formatting → **Typer + Rich**
- pattern-based code chunking → **tree-sitter**
- a React build toolchain → **HTMX**
- PDF text reconstruction → **pypdfium2** page provenance

That is most of the difference between 33,000 lines and something considerably smaller.

### Open

**Metadata store.** SQLite + LanceDB, or LanceDB alone. Decide when building storage —
it also decides where BM25 comes from.

**PDF parser.** pypdfium2 alone may be enough; docling and marker score better on layout
benchmarks but pull in torch and downloaded weights. Decide against real documents from
your corpus, not benchmarks.

**Web UI.** HTMX is the smaller path. A React SPA is the known-good one, since
OpenDocuments already has it working.

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
