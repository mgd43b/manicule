# manicule

Self-hosted document search and answers. Index documents from wherever they live — disk,
git, Notion, Confluence, Drive, S3, the web — ask questions in natural language, and get
answers with citations that resolve to a real location in a real document. Usable from the
CLI, from a browser, and by AI assistants over MCP.

## Prior art

[OpenDocuments](https://github.com/joungminsung/OpenDocuments) is the closest existing
thing, and it works — it runs daily on real documents. That matters: it establishes the
feature set is achievable and worth having, and its source is a detailed map of both what
to build and what to do differently.

This document references it constantly for that reason. It is a reference point, not a
template. Roughly 33,000 lines of TypeScript, and several of its subsystems are worth
improving on rather than reproducing — its PDF citations point at pages that do not exist,
its Confluence extraction destroys every table and code block, its connectors re-enumerate
entire sources on each sync, and its retrieval quality has never been measured because its
evaluation harness scores at random chance.

Where the design below says a subsystem carries over, that is a judgement that
OpenDocuments got it right. Where it does not, the reason is stated.

---

## Stack at a glance

| Layer | Choice |
|---|---|
| Language / tooling | Python 3.12+ · uv · ruff · pyright strict · pytest |
| Config & models | Pydantic v2 · pydantic-settings |
| Vectors | LanceDB |
| Metadata | SQLite · SQLAlchemy 2.0 async · Alembic |
| Lexical search | SQLite FTS5 |
| Embeddings | MLX on Apple Silicon · onnxruntime fallback |
| Generation | litellm |
| Reranking | sentence-transformers CrossEncoder |
| HTTP | FastAPI · uvicorn |
| MCP | FastMCP |
| CLI | Typer · Rich |
| Web UI | HTMX · Jinja2 · Tailwind standalone |
| HTTP client | httpx (async everywhere) |
| Scheduling | apscheduler |
| Auth | authlib · itsdangerous |

---

## 1. Foundation

| | OpenDocuments | manicule | |
|---|---|---|---|
| Runtime | Node 20, TypeScript, ESM | Python 3.12+ | |
| Packaging | npm + Turborepo monorepo | **uv** workspace | `uv tool install manicule` — one command, no container |
| Lint/format | none configured | **ruff** | |
| Types | tsc strict | **pyright** strict | |
| Config validation | hand-rolled schema | **Pydantic v2** | **Gain.** One declarative layer for config, API models, plugin manifests |
| Tests | vitest, 51 tasks | **pytest** + pytest-asyncio | |

## 2. Storage & data model

Sixteen tables carry over unchanged in shape: `documents` `chunk_relations` `collections`
`collection_documents` `tags` `document_tags` `document_versions` `connectors`
`conversations` `messages` `workspaces` `workspace_members` `api_keys` `audit_logs`
`query_logs` `plugins`.

| | OpenDocuments | manicule | |
|---|---|---|---|
| Vectors | LanceDB | **LanceDB** | Already correct. Embedded, no server, ACID, versioned |
| Metadata | SQLite (better-sqlite3) | **SQLite** via SQLAlchemy 2.0 async | |
| Migrations | hand-rolled runner over 8 `.sql` files | **Alembic** | **Gain.** Autogenerate, downgrade, branching |
| Lexical | SQLite FTS5 | **SQLite FTS5** | |

**Coupling:** BM25 comes from FTS5. Folding metadata into LanceDB to run a single store
means moving BM25 to LanceDB's Tantivy index. One decision, not two. Start with SQLite —
it is what works today.

## 3. Plugin system

Four kinds — parser, connector, model, middleware — with `setup` / `teardown` /
`healthCheck` / `metrics` hooks, dependency and conflict declarations, config schemas.

| | OpenDocuments | manicule | |
|---|---|---|---|
| Contract | TypeScript interfaces | **`typing.Protocol`** | Structural, no inheritance required |
| Discovery | npm package names, dynamic `import()` | **entry points** via `importlib.metadata` | |
| Config schema | ad-hoc objects | **Pydantic models** | Validation and docs for free |
| Wiring | 925-line `bootstrap.ts` | **container assembled at startup**, injected | **Gain.** No god function |
| Isolation | `permissions` field, enforced nowhere | **no such field** | See defects |

Built-in implementations use the same public entry-point path as third-party ones.
OpenDocuments kept built-in parsers inside core *and* a plugin interface; the plugin path
rotted because nothing depended on it.

## 4. Ingest pipeline

Discover → fetch → parse → chunk → embed → store, with middleware hooks at each stage,
parser fallback chains, content-hash dedup, and per-document error status that never
aborts a batch. **Carries over as-is.**

Additions: retain original source bytes so re-parsing never requires re-fetching, and make
re-ingest against a pinned corpus a first-class operation rather than a re-crawl.

## 5. Parsers — 12

| Format | OpenDocuments | manicule | |
|---|---|---|---|
| **PDF** | `pdf-parse` | **pypdfium2** fast path; **docling** or **marker** optional | **Gain.** Real page and bbox provenance. Avoid PyMuPDF — AGPL, incompatible with MIT |
| **Code** | pattern-matched functions/classes | **tree-sitter** | **Biggest gain.** Real ASTs, 40+ languages |
| HTML | `node-html-parser` | **selectolax** structural; **trafilatura** for crawled pages only | Two different jobs — fidelity vs boilerplate removal |
| DOCX | `mammoth` | **python-docx** | |
| XLSX/CSV | `xlsx` | **python-calamine** | Rust-backed, much faster |
| PPTX | custom XML | **python-pptx** | |
| Jupyter | custom JSON | **nbformat** | |
| Email | custom | stdlib **`email`** | |
| Markdown | custom | **markdown-it-py** | |
| Plain text | custom | stdlib | |
| Structured | custom | stdlib `json`/`tomllib`, **PyYAML** | |
| Archive | placeholder | **zipfile** + recurse into the parser chain | OpenDocuments never implemented this |

## 6. Connectors — 8

OpenDocuments re-enumerates nearly everything on every sync and dedups by content hash.
Only GitHub has a real change token. Each connector below gets proper incremental sync.

| Source | OpenDocuments | manicule sync | Client |
|---|---|---|---|
| **GitHub** | `api.github.com`, blob `sha` | Trees API `recursive=1`, blob SHA as token, **ETag conditional requests**; webhooks optional | `githubkit` (async, typed) or httpx |
| **Notion** | `/v1` cursor paging, no incremental filter | `/v1/search` sorted by `last_edited_time` against a **watermark**; blocks fetched recursively | httpx |
| **Google Drive** | `files.list` + `pageToken` — pagination, not change tracking | **Changes API** — `getStartPageToken` then `changes.list`. Export native Docs/Sheets/Slides via `files.export` | `google-auth` + httpx |
| **Confluence** | full space walk, `body.storage`, **regex tag strip** | **CQL watermark** (`lastmodified > …`); **ADF** (`atlas_doc_format`) on Cloud, `body.storage` on Server/DC; ingest attachments through the parser chain | httpx against REST v2 |
| **S3 / GCS** | listing + pageToken | `list_objects_v2` with prefix + continuation; **ETag + LastModified** as change token | `aioboto3`; `obstore`/`fsspec` for one interface over both |
| **Swagger** | fetch, no change signal | Fetch and hash the spec; chunk **per endpoint** with parameters and schemas | httpx + `openapi-spec-validator` |
| **Web crawler** | content hash | **Conditional GET** (ETag / If-Modified-Since) plus content hash; honour `robots.txt` | httpx + **protego** + selectolax |
| **Web search** | Tavily, query time | unchanged — no sync | httpx |

Plus local filesystem with watch mode (**watchfiles**, Rust-backed) and web upload.

**Confluence is where the gap is widest.** OpenDocuments requests
`body.storage` — Confluence's XHTML dialect with `<ac:structured-macro>` elements — then
runs `html.replace(/<[^>]+>/g, ' ')`. Tables, code blocks, headings and macros collapse
into a run of words. ADF gives a typed JSON document tree instead, mapping directly onto
the chunk model with no markup parsing at all.

## 7. Embeddings

| | OpenDocuments | manicule | |
|---|---|---|---|
| Runtime | Ollama / cloud HTTP | **MLX** (`mlx-embeddings`) | Native Apple Silicon |
| Fallback | — | **onnxruntime** | Anywhere not Apple Silicon |
| Pooling | whatever the provider does | **ours, in numpy** | **Gain.** `mlx-embeddings`' `last_hidden_state` returns the *pooled* vector; token states are one attribute down. CLS vs mean pooling on the target model differs by **0.856 cosine** — silently |
| Caching | in-memory L2 | same, keyed by model identity | Model identity recorded in the index; a mismatch is a loud error with a re-embed path |

## 8. Retrieval

Hybrid dense + BM25, reciprocal rank fusion, rerank, confidence scoring, context-window
fitting, three profiles (fast / balanced / precise). **Carries over.**

| | Choice |
|---|---|
| Dense | LanceDB vector search |
| Lexical | SQLite FTS5 (BM25) |
| Fusion | RRF |
| Rerank | sentence-transformers CrossEncoder |

OpenDocuments additionally has HyDE, multi-query expansion, query decomposition, intent
classification, cross-lingual expansion, parent-document retrieval, propositions,
cross-encoder rerank, prompt compression and a hallucination guard. **Its evaluation
harness scores at random chance**, so none are known to help. Port the proven path; add
the rest only with a measured improvement.

## 9. Generation & chat

Streaming answers with citations and confidence, multi-turn memory, shared conversation
links, feedback capture. **Carries over.**

| | OpenDocuments | manicule | |
|---|---|---|---|
| Providers | 5 hand-written clients (Ollama, OpenAI, Anthropic, Google, xAI) | **litellm** | **Gain.** One dependency, plus every OpenAI-compatible endpoint |
| Streaming | custom SSE | FastAPI SSE | |

## 10. CLI — 19 commands

`ask` `search` `index` `document` `connector` `workspace` `auth` `plugin` `config`
`backup` `export` `import` `reset-index` `doctor` `init` `start` `stop` `upgrade`
`completion` — with interactive REPL, stdin piping, `--json`.

| | OpenDocuments | manicule | |
|---|---|---|---|
| Framework | Commander.js | **Typer** | Derives the CLI from type hints |
| Output | hand-rolled log formatting | **Rich** | **Gain.** Tables, progress, spinners, syntax highlighting |
| Completion | custom script | Typer built-in | |

## 11. MCP server — 19 tools

ask · search · index_path · document_list · document_get · document_delete ·
document_reindex · index_status · stats · doctor · connector_list · connector_sync ·
config_get · config_set · workspace_list · workspace_switch · plugin_list · plugin_add ·
plugin_remove

**FastMCP** replaces a hand-rolled 500-line protocol server with decorators over
functions. **Gain.**

## 12. HTTP API — 11 route groups

health · documents · chat · conversations · collections · tags · admin · plugins · auth ·
workbench · websocket chat. Plus an embeddable chat widget.

**FastAPI** — async, OpenAPI generated from types, native SSE, Pydantic-native.
Binds loopback by default (see defects).

## 13. Web UI — 12 areas

chat · documents · collections · connectors · dashboard · health · plugins · settings ·
workspaces · admin · auth · layout. Command palette, keyboard navigation, dark mode.

**HTMX + Jinja2 + Tailwind standalone** replaces React + Vite + Tailwind, deleting the
Node toolchain — Tailwind ships standalone binaries needing no npm, and streaming chat
works over SSE. This is the least certain choice in the plan: the React SPA is
known-good and already written. Decide when you get there.

## 14. Team mode & security

API keys (prefixed, SHA-256 hashed, scoped, expiring) · roles admin/member/viewer ·
rate limiting · OAuth SSO (Google, GitHub) · workspace isolation enforced on every query ·
audit log · security alerts · PII redaction. **Carries over.**

| | Choice |
|---|---|
| OAuth | **authlib** |
| Sessions | **itsdangerous** signed cookies, HttpOnly, SameSite=Strict |
| API keys | stdlib `secrets` + `hashlib` |
| Rate limiting | in-process token bucket |

Worth keeping from OpenDocuments: its trusted-proxy handling is genuinely good — proper
CIDR allowlists, no naive `X-Forwarded-For` trust. Port that logic, not the defaults.

## 15. Operations

Backup and restore · export/import · `doctor` diagnostics · telemetry and query logs ·
event bus with webhook dispatch · degraded-mode warnings · update checks.
**Carries over.**

| | Choice |
|---|---|
| Scheduling | **apscheduler** — connector polling, no broker |
| Webhooks | httpx with retry and backoff |
| Diagnostics | Rich-formatted `doctor` |

---

## Defects to fix on the way

Real, found in the source, worth not reproducing.

1. **PDF citations point at pages that do not exist.** Extracted text is split on
   blank-line runs and the fragments numbered as pages. Citations are the product.
   *A citation carries a correct location, or none.*
2. **Confluence extraction destroys structure.** A regex strips every tag from Confluence
   storage format. Use ADF.
3. **No auth by default, on every interface.** Personal mode has no authentication and
   binds all interfaces; the documented Docker deployment publishes it. Bind loopback.
4. **Plugin permissions are enforced nowhere.** Typed, tested, never checked at runtime.
   Do not ship the field.
5. **PII redaction does something other than advertised.** It runs at ingest, permanently
   destroying data in the index, while the docs claim it protects data sent to cloud
   models. Pick one behaviour, build it, document it.
6. **The evaluation harness scores at chance.** Its test embedder is
   `sin(sum of character codes)`. Twelve retrieval features rest on it.

---

## Build order

| | | |
|---|---|---|
| 1 | Protocols, container, config | Everything plugs in here |
| 2 | Storage and data model | 16 tables, Alembic, LanceDB |
| 3 | Embeddings | Fixes vector dimensionality — settle before the schema exists |
| 4 | Parsers and chunking | The largest upgrade |
| 5 | Ingest pipeline | Middleware, fallbacks, dedup, retention |
| 6 | Retrieval | Dense + BM25 + RRF + rerank |
| 7 | Generation and chat | litellm, streaming, citations, multi-turn |
| 8 | **MCP and CLI** | Ship early — daily use finds what tests do not |
| 9 | Connectors | All eight, with real incremental sync |
| 10 | Document management | Collections, tags, versions, chunk relations |
| 11 | HTTP API | |
| 12 | Web UI | |
| 13 | Team mode | Auth, workspaces, audit, redaction |
| 14 | Operations | Backup, export, doctor, telemetry, webhooks |
| 15 | Quality baseline | Measure retrieval quality on a real corpus. OpenDocuments is one baseline to beat; the point is knowing the number at all |

Steps 3 and 4 land before 2 finishes: the schema fixes vector dimensionality and assumes a
chunk size, and changing either later means re-embedding everything.
