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

Where the design below says a subsystem carries over, that is a judgment that
OpenDocuments got it right. Where it does not, the reason is stated.

---

## Stack at a glance

| Layer | Choice |
|---|---|
| Language / tooling | Python 3.14+ · uv · ruff · pyright strict · pytest |
| Config & models | Pydantic v2 · pydantic-settings |
| Vectors | LanceDB |
| Metadata | SQLite · SQLAlchemy 2.0 async · Alembic |
| Lexical search | SQLite FTS5 |
| Embeddings | **BGE-M3** on **MLX**, in-process · **onnxruntime** for parity and portability |
| Generation | **Ollama** locally, via **litellm** · any hosted provider through the same call |
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
| Runtime | Node 20, TypeScript, ESM | Python 3.14+ | |
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

**Decided: SQLite plus LanceDB, and therefore FTS5 for BM25.**

Sixteen tables of collections, tags, versions, workspaces, API keys and audit logs is
relational work — joins, foreign keys, transactional updates. Expressing that in a
columnar vector store is fighting the tool. FTS5 is mature and gives BM25 for free, where
LanceDB's Tantivy index is newer and has had incremental-indexing limitations. And it is
what the working reference implementation does.

This closes the BM25 question with it — the two were one decision.

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
| **PDF** | `pdf-parse` | **pypdfium2** fast path; **docling** or **marker** optional | **Gain.** Real page and bbox provenance. `pypdfium2` is BSD/Apache. PyMuPDF stays rejected — see the license note below. **No OCR in v1** — see below |
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

### What the licensing has settled on, and what it never changed

The project is MIT (`LICENSE`). It was GPL-3.0-or-later for a period, to accommodate the MLX
embedding runtime; that runtime now ships as its own GPL-3.0-or-later distribution
(`packages/manicule-mlx`), and core is permissive again — so a copyleft dependency in
`manicule` itself is refused. **Neither parsing decision changes**, and the two are refused for
different reasons:

- **`extract-msg` (GPL-3.0) is refused because manicule is MIT.** It was briefly admissible
  during the copyleft period and is not now.
  [#21](https://github.com/mgd43b/manicule/issues/21) is therefore a permissively licensed
  reader or a hand-written MAPI property reader, not a one-line dependency.
- **PyMuPDF (AGPL-3.0) is refused under any license this project might carry**, which is the
  stricter rule and the more durable one. GPLv3 §13 permits combining with AGPLv3 code, but
  the combination then carries AGPL §13: anyone who *runs* it as a network service owes source
  to its users. manicule ships an HTTP API and a web UI, so that is a live obligation in team
  mode rather than a formality — one falling on operators rather than on us. `pypdfium2` is
  permissively licensed and already delivers page and bbox provenance, so taking on a
  network-copyleft obligation to swap it out buys nothing.

### OCR — out of scope for v1

v1 ingests Confluence, where content arrives as ADF text. OCR adds real weight, and the
engine choice is worse than it looks: Apple Vision (via `ocrmac`) is Mac-only while
RapidOCR runs anywhere, so the same document would chunk differently depending on where it
was ingested — a corpus-consistency hazard, not just a dependency.

**Out of scope must not mean silent.** A PDF yielding no extractable text gets a visible
`no_extractable_text` document status, not an empty document that looks successfully
indexed. Revisit when a real corpus contains scanned documents worth having.

## 6. Connectors — 8

**v1 ships Confluence only** ([#9](https://github.com/mgd43b/manicule/issues/9)); the other seven follow once the
protocol is proven against a real implementation. Full Confluence design in
[`docs/connectors/confluence.md`](docs/connectors/confluence.md).

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
| **Web crawler** | content hash | **[Git-backed site first, then HTTP](docs/connectors/web-crawler.md)**; Conditional GET (ETag / If-Modified-Since) plus content hash; honor `robots.txt` | httpx + **protego** + selectolax |
| **Web search** | Tavily, query time | unchanged — no sync | httpx |

Plus local filesystem with watch mode (**watchfiles**, Rust-backed) and web upload.

**Confluence is where the gap is widest.** OpenDocuments requests
`body.storage` — Confluence's XHTML dialect with `<ac:structured-macro>` elements — then
runs `html.replace(/<[^>]+>/g, ' ')`. Tables, code blocks, headings and macros collapse
into a run of words. ADF gives a typed JSON document tree instead, mapping directly onto
the chunk model with no markup parsing at all.

## 7. Model runtimes — MLX and Ollama

Two runtimes, split by job. This is deliberate: the two have different requirements and
neither is good at both.

### Embeddings — `BAAI/bge-m3` on MLX, in-process

| | OpenDocuments | manicule | |
|---|---|---|---|
| Model | provider default | **`BAAI/bge-m3`** — 1024d, 8192 tokens, MIT, multilingual | Configuration, not a constant. See [`docs/embeddings.md`](docs/embeddings.md) |
| Runtime | Ollama / cloud HTTP | **MLX** (`mlx-embeddings`) | **No server process.** What keeps `uv tool install manicule` a single command with nothing to operate alongside it |
| Second backend | — | **onnxruntime** | Not a fallback — the parity check. Also the path off Apple Silicon |
| Pooling | whatever the provider does | **ours, in numpy** | **Gain.** BGE-M3 pools with CLS, and the MLX convenience field mean-pools. See below |
| Caching | in-memory L2 | keyed by the full `EmbedFingerprint` | A model change is a loud error with a re-embed path, never quietly worse results |

**Why not speed.** An earlier draft justified MLX with "~50% faster than llama.cpp on
embeddings." That figure has no traceable primary measurement and should not be repeated.
The argument for MLX is that it runs in-process; benchmark it during #3 if the number
matters.

**onnxruntime is the enforcement mechanism, not the fallback.** `mlx-embeddings` is version
0.1.0 and assigns `last_hidden_state` different meanings on different architectures. So the
same text, through the same model, under both backends, must produce vectors within a stated
tolerance — asserted in tests, not assumed. That is the Apple-hardware principle in its
operative form: **optimize execution for Apple hardware freely; never let the platform change
what ends up in the index.** Throughput may vary by machine. Vectors may not.

**The model is switchable, and switching is a priced operation.** There is a known-good set,
one is active at a time, and changing it means a full re-embed — bounded, resumable, and
loud rather than silent, because `EmbedFingerprint` refuses a mismatched index. Citations
survive it: `chunks.id` derives from content, not from the model.

**Why pooling is ours.** `mlx-embeddings` offers a convenience field, and on the chosen
model it is the wrong pooling. `BAAI/bge-m3` pools with **CLS**; the library's XLM-RoBERTa
implementation computes `text_embeds` with **mean pooling, unconditionally**. Anyone
trusting the obviously-named field gets correctly-shaped, normalized vectors from the wrong
reduction, with no error raised.

The same library also gives `last_hidden_state` different meanings on different
architectures — genuine token states on `xlm_roberta` and `bert`, the *pooled* vector on
`modernbert` — so neither field can be trusted by name. manicule reads token states and
pools them in its own numpy, driven by the model's declared configuration. CLS and mean
diverge more the longer the chunk, so at the 512-token budget the gap is at its widest.
Full detail, with what was measured and what still must be, in
[`docs/embeddings.md`](docs/embeddings.md) §4.

### Ollama is not an embedding backend

Settled, not a preference. `/api/embed` returns pooled, already-normalized vectors
regardless of input, and **no configuration changes that** — setting `LLAMA_ARG_POOLING`
reaches the engine and then breaks the endpoint for every input. Without token states
there is no pooling control, and the failure above becomes unavoidable.

### Generation — Ollama, through litellm

| | OpenDocuments | manicule | |
|---|---|---|---|
| Local | one of five hand-written clients | **Ollama** | Model management MLX lacks — pull, serve, swap |
| Hosted | four more hand-written clients | **litellm** | **Gain.** One dependency covers Ollama, OpenAI, Anthropic, Google, xAI and any OpenAI-compatible endpoint |
| Switching | provider plugin per vendor | a `base_url` | Local and hosted are the same code path |

Ollama is the local default and is **optional** — litellm will point at a hosted model
instead, so the install does not require it.

### The alternative, recorded

`llama-server` can do both: embeddings with `--pooling none` and OpenAI-compatible
generation, from one process and one model format, portable off Apple Silicon. Rejected
because it makes embeddings a server dependency rather than an in-process call, and gives
up Metal-native execution on the machine this is built for. Defensible if operating one
runtime ever matters more than those two things.

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

And one operation above it that OpenDocuments has no equivalent for: `research` answers a
question from several searches rather than one, and reports from a deliberately wider context.
It reuses the answer path whole rather than building a second citation chain — see
[`docs/research.md`](docs/research.md).

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

## 16. Cross-cutting subsystems

Found in a source audit against the plan — each is real behavior with no obvious home in
the sections above.

| | What it does | manicule |
|---|---|---|
| **Caching** | `RAGCache` with TTL — L1 query results, L2 embeddings, L3 web-search | Keep. Materially affects perceived latency. Key the embedding cache by model identity so a model change cannot serve stale vectors |
| **Query routing** | Deterministic classifier so greetings and utility queries never reach the model | Keep. Cheap, and it stops trivial input consuming an LLM call |
| **Token counting** | tiktoken, lazily initialized | **tiktoken** — same library, Python-native. Drives context-window fitting |
| **Config loading** | 196 lines resolving provider API keys from `.env` by convention | Keep the behavior, express it in **pydantic-settings** |
| **Hardware detection** | CPU and RAM probing to recommend a model during `init` | Keep, and extend — detect Apple Silicon and unified memory to pick the embedding backend |
| **Plugin compatibility** | `checkCompatibility` against a declared `coreVersion` | Keep. Version mismatch is a loud error, not a runtime surprise |
| **Community registry** | GitHub-hosted list of community plugins, browsable and installable | Keep. Decide whether install stays admin-only — in OpenDocuments it shells out to a package manager |
| **Cross-workspace search** | Admin-only search spanning workspaces | Keep, gated on team mode |

### HTTP surface is larger than "11 route groups"

**52 endpoints**, not 11: admin 12 · collections 6 · conversations 6 · documents 6 ·
auth 5 · tags 5 · health 4 · plugins 4 · chat 3 · workbench 1, plus a websocket channel.
The workbench is a single read-only endpoint behind a `document:read` scope.


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
   models. Pick one behavior, build it, document it.
6. **The evaluation harness scores at chance.** Its test embedder is
   `sin(sum of character codes)`. Twelve retrieval features rest on it.

---

## Build order

Issue numbers match these steps exactly — step 4 is [#4](https://github.com/mgd43b/manicule/issues/4).

| | Ticket | Phase | |
|---|---|---|---|
| 1 | [Protocols, container & config](https://github.com/mgd43b/manicule/issues/1) | Core | Everything plugs in here |
| 2 | [Storage & data model](https://github.com/mgd43b/manicule/issues/2) | Core | 16 tables, Alembic, LanceDB |
| 3 | [Embeddings](https://github.com/mgd43b/manicule/issues/3) | Core | ⚠️ fixes vector dimensionality |
| 4 | [Parsers & chunking](https://github.com/mgd43b/manicule/issues/4) | Core | ⚠️ fixes chunk size. The largest upgrade |
| 5 | [Ingest pipeline](https://github.com/mgd43b/manicule/issues/5) | Core | |
| 6 | [Retrieval](https://github.com/mgd43b/manicule/issues/6) | Core | |
| 7 | [Generation & chat](https://github.com/mgd43b/manicule/issues/7) | Usable | |
| 8 | [MCP server & CLI](https://github.com/mgd43b/manicule/issues/8) | Usable | **Ship early** — daily use finds what tests do not |
| 15 | [Retrieval quality baseline](https://github.com/mgd43b/manicule/issues/15) | Usable | Starts here, continues forever |
| 9 | [Connectors](https://github.com/mgd43b/manicule/issues/9) | Reach | All eight, with real incremental sync |
| 10 | [Document management](https://github.com/mgd43b/manicule/issues/10) | Reach | |
| 11 | [HTTP API](https://github.com/mgd43b/manicule/issues/11) | Interfaces | Eleven route groups over the same service the CLI and MCP use |
| 12 | [Web UI](https://github.com/mgd43b/manicule/issues/12) | Interfaces | |
| 13 | [Team mode & security](https://github.com/mgd43b/manicule/issues/13) | Production | |
| 14 | [Operations](https://github.com/mgd43b/manicule/issues/14) | Production | |

**Not an ordering constraint — a runtime guardrail.** Vector dimensionality comes from the
embedder's fingerprint at run time, and the vector table is created at first ingest, not
when #2 is written. Chunk size does not touch the schema at all. So #2 can be built before
#3 and #4 land, provided it takes `D` as a parameter and **refuses to run with a hardcoded
value**.

What is expensive is *indexing a real corpus* before both are settled, because changing
either means re-embedding everything. That is a guard in the code, not a position in this
table.

Everything else can slip without penalty. #15 is deliberately early and never finishes —
it is the gate on every deferred retrieval feature in #6.
