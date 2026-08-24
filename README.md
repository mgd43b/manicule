<div align="center">

# manicule

**Self-hosted retrieval infrastructure for AI assistants.**

An agent searches a private corpus or asks a grounded question, and gets evidence that resolves
to a real location in a real document — a page, a heading, a line, a cell.

[![CI](https://github.com/mgd43b/manicule/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mgd43b/manicule/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/manicule.svg)](https://pypi.org/project/manicule/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](pyproject.toml)
[![packaged with uv](https://img.shields.io/badge/packaged%20with-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

[Install](#install) · [First run](#first-run) · [The four surfaces](#the-four-surfaces) ·
[As a server](#running-it-as-a-server) · [In a container](#in-a-container) ·
[Design](#the-idea-it-is-organized-around) · [Layout](#layout) · [Extending](#extending-it)

</div>

```bash
manicule init                     # pick a backend, write a config, seed what no wheel ships
manicule index ~/Documents        # walk it, parse it, chunk it, embed it
manicule search "retry policy"    # ranked passages, no model involved
manicule ask "how do retries work?"
manicule doctor                   # what is wrong, and what to do about it
```

> [!NOTE]
> **Early, and runnable.** All four surfaces work today: point it at a directory, search it, ask
> it questions, read it in a browser, hand the same operations to an assistant over MCP, or serve
> them over HTTP. Install it with `uv tool install "manicule[all]"` — [below](#install). It is
> alpha and the version is `0.x`, which means what it says: interfaces may change between minor
> versions, and the envelope contract in [`docs/surfaces.md`](docs/surfaces.md) is the part to
> depend on. See [`PLAN.md`](PLAN.md) for the shape of the whole and the order it is being
> built in.

## What it is

**MCP is the primary interface.** The core value is the accuracy of the embedding and retrieval
pipeline, plus evidence a caller can inspect at its source. The command line and HTTP expose the
same service; the browser is a functional operator and retrieval-inspection console, not the
primary knowledge-work interface.

**Re-indexing is copy-on-write at the document boundary.** Vectors are staged under a publication
id, then the document, chunks, glossary and lineage become active in one relational transaction.
A failed or interrupted refresh keeps the previous indexed revision searchable; unpublished
vectors are filtered during hydration and reclaimed by the normal sweep. The same atomic flip
applies when parsing, middleware or chunking concludes that a document has no chunks.

> [!IMPORTANT]
> **Two sources exist today**: a local directory tree, and Confluence. Seven more — GitHub,
> Notion, Drive, S3/GCS, Swagger, a crawler and web search — are designed in
> [`PLAN.md`](PLAN.md) and tracked in [#16](https://github.com/mgd43b/manicule/issues/16). None
> of them is built, and nothing below describes them.

Every command that emits data takes `--json`, on either side of the command name — `manicule
--json search …` and `manicule search … --json` are the same invocation, and `--workspace`/`-w`
works the same way.

Most of them are also MCP tools, so an assistant reaches the same operations through `manicule
start --mcp-only`, and `manicule start --transport http` serves them over HTTP with an OpenAPI
document at `/api/docs`. All three emit the same envelope, and the shape is a contract written
down in [`docs/surfaces.md`](docs/surfaces.md).

## Install

> Requires Python 3.14 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "manicule[all]"
manicule --version
```

That is the whole install — an isolated environment, `manicule` on `PATH`, and `uv tool upgrade
manicule` later. With [pipx](https://pipx.pypa.io) it is `pipx install "manicule[all]"`, and
`uvx --from "manicule[all]" manicule --version` runs it once without installing anything at all.

**On Apple silicon, add the Metal backend:**

```bash
uv tool install "manicule[all]" --with manicule-mlx
```

[`manicule-mlx`](packages/manicule-mlx) is four to five times faster on the indexing path, and it
is a separate distribution because it is GPL-3.0-or-later while manicule is MIT — see
[License](#license) for what that means for you. Without it manicule runs `onnx` everywhere, and
the vectors are the same either way: `backend` is excluded from the embedding fingerprint, so
adding or removing the package is an operation rather than a migration.

> [!IMPORTANT]
> **`[all]` is not decoration — leaving it off gives you a program that cannot start.** manicule
> itself carries no implementation dependencies, deliberately: that boundary is what lets a
> plugin author depend on its contracts without installing a vector database and a model
> runtime, and [`tests/test_import_boundary.py`](tests/test_import_boundary.py) fails the build
> if it erodes. `[all]` is the extra that turns the library into the program — storage,
> embeddings, the parsers, ingest, retrieval, generation, the connectors and the serving stack,
> about 240 MB. A bare `pip install manicule` succeeds and then tells you this rather than
> raising.

Two extras are **not** in `[all]`, and the reason is size rather than taste:

| | Cost | Add it with |
|:---|:---|:---|
| `rerank` | torch, and on Linux 2.7 GB of CUDA wheels behind it | `uv tool install "manicule[all,rerank]"` |
| `browser-auth` | playwright, then a browser download | `uv tool install "manicule[all,browser-auth]"` |

Neither is needed to index, search or ask. `rerank` buys the cross-encoder that the `balanced`
and `precise` retrieval profiles rescore with — `fast` sets `rerank=False` and never constructs
one — and `browser-auth` buys interactive sign-in for a Confluence Server behind an identity
provider that has disabled tokens — manicule opens a browser and you sign in there.

**The Chrome extension needs neither**, and is the other way round: it reads the session already
in your own browser and hands it over, so nothing is installed, nothing is downloaded and no
window opens. `manicule browser-auth install`, then load the directory it prints. See
[Confluence auth](docs/connectors/confluence.md#11a1-reusing-the-session-in-your-own-browser).

Working on manicule rather than with it is [Development](#development); running it without a
Python at all is [In a container](#in-a-container).


> [!TIP]
> **Two embedding backends, and one of them is chosen for you.** `manicule init` probes the
> machine — it prints `embedding backend 'mlx' chosen for arm64 on Darwin` — and picks `mlx` on
> Apple silicon, which runs the embedder on Metal in-process, or `onnx` everywhere else. Which
> one you get changes how long an ingest takes and **never what comes out of it**: the `backend
> parity (macOS)` job in CI exists to hold that line, comparing vectors from both backends on a
> runner that can build both. If it ever goes red the fix is to the code, not to the tolerance.

## First run

The shortest path from an install to an answer about your own documents.

```bash
manicule init                             # config, grammars, vocabularies — seconds
manicule index ~/Documents                # walk it, parse it, chunk it, embed it
manicule search "how are citations verified"
```

### `init` first — it does more than write a file

It picks the embedding backend this machine can run, then pre-seeds the two things no Python
wheel ships: the 26 tree-sitter grammars the code and diagram readers need, and the two BPE
vocabularies every search measures a context with. Both are small, and both are the kind of
thing manicule refuses to download in the middle of a question. Skip `init` and the first
`search` refuses rather than fetching — `doctor` calls a missing vocabulary `failing`, in as many words, because
no corpus can be searched without one.

### The model weights are the one thing `init` does not fetch

The first `index` does. They are the big artifact: about **1.1 GB** for `BAAI/bge-m3` on Apple
silicon (the MLX conversion, fp16) and about **2.3 GB** elsewhere (the ONNX export). `init` says
so on its way past, and `manicule doctor` says so too, because the download itself is quiet — a
Hugging Face progress bar, and a stretch with no manicule output at all.

To take that wait deliberately, before you have a corpus to be impatient about:

```bash
uv run tools/prefetch_embedding_models.py --backend mlx    # or --backend onnx
```

<details>
<summary><b>How weights are pinned to vector identity</b></summary>

Weights are pinned as part of vector identity, not merely downloaded by model name. The exact
executed hub commit (or a digest for local weights) is recorded with the index. Only the
built-in ONNX/MLX artifact pairs covered by the parity suite are portable across backends;
custom remote weights must set an immutable `weights_revision`, and changing any artifact
requires `reindex --re-embed` rather than reusing incomparable vectors.

Local model card/tokenizer inputs are content-addressed too, including when weights are
separate; local `embedding.revision` claims are rejected. The MCP `index_status` result exposes
the exact `weights_ref` and its compatibility identity.

</details>

### `ask` additionally needs a generator

`search` needs only the embedder. The default configuration expects
[Ollama](https://ollama.com) on `localhost:11434` serving `qwen2.5:14b`:

```bash
ollama pull qwen2.5:14b
manicule ask "what does an anchor carry when the location is unknown?"
```

An installed, already-authenticated Codex or Claude CLI can answer instead. Select the built-in
`cli` generator once; both `manicule ask` and browser chat then use it through the same retrieval,
redaction and citation-verification path:

```bash
manicule config set llm.generator cli
manicule config set llm.provider codex       # or: claude
manicule config set llm.model default        # keep the CLI's configured model
manicule config set llm.context_window 32768 # use a conservative value the model supports
manicule ask "what does an anchor carry when the location is unknown?"
```

The executable must be on the `PATH` of the process running manicule. Restart a running server
after changing these settings. The CLI owns its own login, but its destination is treated as
remote for data-policy purposes because manicule cannot inspect where the command sends a prompt.

### When something does not work

`manicule doctor` reports what is wrong and what to do about it, and it is the first thing to run.
`manicule doctor --fix` repairs what it can: the grammars and the vocabularies, from an offline
bundle when one is installed and from upstream otherwise. It is the only part of that command
that writes to the machine or uses the network, which is why it is a flag — and it does **not**
fetch the model weights, which is why the line above exists.

> [!WARNING]
> **A query never fetches a vocabulary or a grammar.** Those are seeded by a step you can watch
> fail, and a query that finds one missing refuses rather than reaching for the network: `search`
> exits non-zero with `VocabularyUnavailableError` and names the cache it looked in. The model
> weights are the one artifact fetched on demand rather than refused — which is why `init` and
> `doctor` both announce them while they are still to come, and why the prefetch line above
> exists.

If `doctor` reports an obsolete fingerprint on an otherwise empty workspace, the explicit repair
is `manicule reset-index --yes`. It removes only that workspace's derived chunks, memberships,
FTS/vector visibility, publication checkpoints and cached runtime handles. Retained source
snapshots, document-version history, connector configuration, credentials and other workspaces
survive. The command is idempotent and its JSON result separates relational rows, vector rows,
publications, terminalized generations, retained snapshots, fingerprint cleanup and runtime-cache
invalidation. A non-empty fingerprint mismatch remains a refusal: use the rebuild or re-embed
path instead of discarding a searchable corpus.

### What it costs to wait

| Step | Measured here |
|:---|---:|
| `manicule init`, with every cache cold | 9 s |
| first `index` of `docs/`, model still to download | 1 m 21 s and 2 m 04 s, on two runs |
| the same `index` with the model already present | 38 s |
| the same `index` inside the container, on ONNX | 5 m 04 s |

One machine — an Apple M4 Max — against this repository's `docs/`: 13 documents, about 677 kB.
They are here to set expectations about orders of magnitude. They are not benchmarks, and a
busier machine moves them: the 38-second run above took 54 seconds with a container build
alongside it.

## Durable connector hand-off and offline rebuild

A configured connector can stop after it has promoted a complete, locally retained source
snapshot. The snapshot can then be verified and rebuilt without constructing the connector or
contacting the source:

```bash
manicule connector sync handbook --acquire-only
manicule connector snapshot handbook --json       # copy data.snapshot_id
manicule connector verify SNAPSHOT_ID
manicule rebuild plan SNAPSHOT_ID
manicule rebuild execute SNAPSHOT_ID
manicule rebuild status GENERATION_ID
```

There is no separate settlement command: successful publication atomically settles the exact
acquisition manifests it consumed, and rebuild has no connector or source fallback. Planning
binds the newest promoted snapshot for every connector scope in the workspace into one ordered
shadow generation. An interrupted worker resumes its durable sequence checkpoint, and the old
corpus remains queryable until the complete multi-source replacement validates and publishes in
one transaction.

The structural chunk policy is component configuration. Defaults remain 512 final `embed_text`
tokens with up to 64 tokens of prose/list overlap:

```toml
[plugins.config."chunker.structural"]
max_tokens = 768
overlap_tokens = 96
```

These values are fingerprinted and must fit the configured embedding model's effective context
window. Changing either value requires `rebuild plan` followed by `rebuild execute`; it rechunks
and re-embeds retained originals alongside the live corpus. A larger budget is a
retrieval-quality, index-size and embedding-cost tradeoff, not an automatic improvement.

<details>
<summary><b>Recovery state, deletion reconciliation and Confluence pagination internals</b></summary>

`connector snapshot` and `connector list` expose aggregate recovery state. Authentication,
transport, capacity and temporary body failures retry the same valid manifest. A confirmed
post-enumeration `source_deleted` reports `reenumeration_required`; the next ordinary sync fences
that run, starts one replacement from the last committed watermark and reports `reenumerating`.
Matching retained bodies are validated and reused. Only a replacement discovery that reaches its
real end may report `reconciled` and remove an absent identity from required membership. A limit,
expired cursor, cancellation or failed discovery never proves deletion.

Confluence discovery and deletion enumeration preserve the source's own response-page boundary.
Each page (at most 250 records) is admitted atomically before `_links.next` is followed, so a
large space pays one capacity-guarded SQLite writer transaction per source page rather than one
per record. Cursor expiry, repetition, corruption and cross-origin guards remain fail-closed; a
10,251-record synthetic run crosses the 10,000 boundary and records true completion only after
the final one-record page. Cursor-cycle history and subtree membership use temporary,
fixed-cache SQLite indexes, keeping pagination and subtree scope bookkeeping independent of
corpus size in process memory.

Server and Data Center whole-space connectors may set
`full_inventory_authority = "direct_current_content"`. Complete discovery and reconciliation
then use the direct current-content inventory, while incremental discovery remains CQL-backed.
The compatibility default is `search`; Cloud and page-tree scopes keep their existing CQL
behavior even when the direct option is present. The effective authority is included in durable
cursor identity and aggregate connector/snapshot status, so switching authority performs one
complete replacement without exposing configured spaces. Exact retained bodies are reused after
their source revision and acquisition evidence are revalidated. Direct native pagination is
scope-pinned on every request and only true exhaustion can authorize deletion reconciliation.

Strict policy never promotes while a current member lacks validated evidence. An
`allow_omissions` snapshot remains honestly partial, and it cannot turn a known-stale inventory
green. Once a promoted snapshot publishes, source acquisition and derived publication settle in
the same transaction; a complete snapshot reaches zero backlog while its retained source bytes
remain available for later connector-free rebuilds. Status exposes counts and typed states, never
source ids, paths, URLs, bodies, blob hashes, credentials or copied source exceptions.

</details>

The full contract and recovery details are in
[`docs/ingest.md`](docs/ingest.md#831-durable-discovery-then-bounded-hand-offs) and the
shared result shape is in [`docs/surfaces.md`](docs/surfaces.md#401-shared-lifecycle-status).

## The four surfaces

| Surface | Started by | Shape |
|:---|:---|:---|
| **MCP** | `manicule start --mcp-only` | 41 tools over stdio, which opens no socket; 24 read-only tools at `/mcp/` when served over a port |
| **Command line** | `manicule <command>` | 27 commands; `--json` anywhere data is emitted |
| **HTTP API** | `manicule start --transport http` | 12 route groups on `127.0.0.1:8765`, OpenAPI at `/api/docs` |
| **Browser** | the same process, at `/ui` | Functional operator and retrieval-inspection console; 12 areas of server-rendered HTML, 11 in the navigation |

They are adapters over one application service, and `tests/app/test_surface_parity.py` holds
them to it: for the same operation and the same arguments the CLI under `--json`, the MCP tool
and the HTTP route return **byte-identical** envelopes, and the browser page is asserted to
show what that envelope said rather than anything it worked out for itself.

### The command line

Twenty-eight commands; `manicule --help` lists them. Under `--json` the result envelope is the
whole of stdout — no prose, no progress, nothing else — and a failure is that same envelope with
`"ok": false`, a typed `error` and a non-zero exit status. So `jq` reads well-formed JSON whether
the command succeeded or not.

Rule-driven collections select current and future documents without enumerating ids or
rebuilding the index:

```console
manicule collection create "Team A" --source wiki-team-a --source wiki-team-a-archive
manicule collection rule show COLLECTION_ID
manicule collection rule set COLLECTION_ID --source wiki-team-a
manicule collection rule clear COLLECTION_ID
```

Sources within a rule are alternatives; source, media-type, tag, and update-bound fields are
combined. Manual members remain unioned with the rule. Creating, replacing, or clearing a rule
changes only collection metadata: existing indexes adopt it immediately, with no source fetch,
re-ingestion, chunking, or re-embedding.

<details>
<summary><b>How a connector sync reports its outcome</b></summary>

Connector syncs additionally report `data.outcome` as `complete`, `bounded`, or `incomplete`.
An incomplete sync exits non-zero and keeps its partial counters in `data`; a requested
`--limit` is `bounded`, exits zero, and never advances the watermark. A document-level failure
that left a durable repairable row does not by itself make the source enumeration incomplete.
If an external SQLite writer remains present beyond the bounded retry policy, the failure type is
`StorageBusyError`; retrying resumes the committed acquisition prefix without advancing the
watermark or exposing SQL and local paths in the envelope.

</details>

### The HTTP API

Twelve route groups over the same service — health, documents, chat with SSE streaming,
conversations and shareable links, collections, tags, admin, plugins, auth, a workbench, a
websocket channel and an MCP endpoint — plus an embeddable chat widget at `/widget`. `manicule
start --transport http` serves them on `127.0.0.1:8765`, and only there unless three separate
things say otherwise. It prints where it is listening, and every path on it that you might want
next:

```console
HTTP API on http://127.0.0.1:8765 (this machine only)
browser surface    http://127.0.0.1:8765/ui
MCP endpoint       http://127.0.0.1:8765/mcp/
API documentation  http://127.0.0.1:8765/api/docs
```

**The address itself works.** `/` redirects to the browser surface — temporarily, so no browser
caches it — and when there is no browser surface to redirect to, it says so and names what this
process *is* serving. `docs/surfaces.md` §6.3.

**MCP is served from that same process and port**, at `/mcp/`, and it carries the **read-only
tools only** — the write tools are not registered on it rather than refused, so there is no
handler behind `document_delete` or `connector_sync` there at all. Over stdio, where one client
talks to one process down a pipe, the whole surface is offered. `docs/surfaces.md` §6.1 says why.

`/api/docs` is Swagger over the OpenAPI document at `/api/openapi.json`. Every response is the
same envelope the CLI prints under `--json`.

### The browser surface

The functional operator and inspection console, not the primary knowledge-work interface. It is
server-rendered HTML at `/ui`, on the same socket, with eleven areas in its navigation: a
dashboard; chat with streaming citations, confidence and feedback; documents, their chunks, the
trash and restore; collections and tags; connectors; plugins; workspaces; health; an admin
dashboard; your own API keys; and settings. Command palette on `Ctrl`/`Cmd`+`K`, keyboard
navigation, dark mode. `manicule start --no-web` prints `browser surface    off (--no-web)`,
keeps the API, and answers 404 for every `/ui` path — and `/` lists the surfaces that are still
there rather than redirecting to one that is not.

![The manicule browser surface: a search for "how are citations verified" over this repository's own docs, showing ten ranked passages, the confidence band with the sentence explaining it, and each hit labeled with the document and the heading path the passage came from](docs/images/browser-search.png)

It adds **no build toolchain**: Jinja2 templates, one hand-written stylesheet and one
hand-written script, so `uv sync` is still the whole install and the container image stays free
of Node. It also adds **no operation** — every page reads through a service method that already
has a route, so there is no upload and no configuration write here either.
[`docs/web.md`](docs/web.md) has the reasoning, including what that costs.

### The MCP server

The primary interface: forty-one tools over the same service, speaking stdio by default,
which opens no socket at all. To let Claude Code use your index:

```bash
claude mcp add manicule -s project -- "$(pwd)/.venv/bin/manicule" start --mcp-only
```

`-s project` is what puts it in `.mcp.json` beside the project, where it is checked in and
everyone working on the repository gets it; the default scope is `local`, which records the
server for you alone. What it writes:

```json
{
  "mcpServers": {
    "manicule": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/manicule",
      "args": ["start", "--mcp-only"],
      "env": {}
    }
  }
}
```

The server is also reachable as `python -m manicule.mcp`, for a client that would rather name an
interpreter and a module than trust a console script to be on the PATH it happens to have.

## Running it as a server

```console
$ manicule serve                    # holds the data directory, runs the schedule, keeps sessions
```

One process owns the data directory at a time. `manicule serve` is that process for as long as
it runs, and three things follow:

- **Write commands go to it.** `connector sync`, `index`, `document reindex`, `rebuild execute`
  and the repair verbs detect the running server and forward to it over a `0600` Unix domain
  socket, streaming progress back. You get the same result, the same output and the same exit
  status you would have got locally, because it is the same operation. With no server they refuse
  and say to start one — they never start one for you.
- **Reads never need it.** `search`, `ask`, `doctor` and `document list` take no lock and work
  whether or not a server is running.
- **Confluence sessions live in its memory and nowhere else** — no keychain, no file, no
  environment variable, so nothing prompts for a password and nothing is left on disk. They are
  gone when it stops, which is the right lifetime now that
  `manicule connector login --browser` makes signing in again a few seconds of clicking — or
  nothing at all, with the Chrome extension, which notices you signing in to the wiki normally
  and hands the new session over by itself. **No server, no sync.**

Per-source schedules are configuration:

```toml
[connectors.handbook]
type = "confluence"
schedule_s = 3600      # a served manicule syncs this hourly, with nothing typed
```

A website whose source pages already live in a local Git repository can be indexed without
copying those source bytes into Manicule's blob store:

```toml
[connectors.product-docs]
type = "git-site"
schedule_s = 300
retain_source_bytes = false

[connectors.product-docs.options]
repository = "/srv/product-docs"
revision = "HEAD"
content_root = "docs"
base_url = "https://docs.example.com/"
```

Each sync resolves one commit, inventories only committed page blobs, and keeps using that commit
even if `HEAD` moves before fetching finishes. Markdown, MDX and HTML routes are inferred from
their paths by default; sites with custom permalinks can commit an authoritative route manifest.
See [`docs/connectors/web-crawler.md`](docs/connectors/web-crawler.md) for the manifest format,
include/exclude rules and the operational tradeoff of disabled source-byte retention.

An admin may also start one already-configured connector through
`POST /api/v1/admin/connectors/{name}/sync`. The route cannot declare or reconfigure a source:
connectors hold credentials and reach remote systems, so the complete set remains in
configuration where it can be reviewed. Per-source schedules remain the unattended path.
`docs/deployment.md` §6 has the whole of it.

## In a container

A [`Dockerfile`](Dockerfile) and a [`compose.yaml`](compose.yaml) are here, and the image is
self-contained: the grammars, the model weights and the Python environment all arrive with it.
The last thing the build does is run `doctor`, index a corpus and search it **with the network
switched off**, so an image that would have fetched something on first use fails to build
instead.

```bash
docker compose build                                   # ~2.3 GB of model weights, once
docker compose run --rm manicule doctor
docker compose run --rm manicule index /corpus/docs    # this repository, mounted read-only
docker compose run --rm manicule search "how are citations verified"
```

The build downloads the weights and the grammars, and the resulting image is about **3.4 GB**.
That cost is paid at `build`, where a long step is legible, rather than inside the first `index`,
where it looks like a hang.

The image runs as an unprivileged user with a `0700` data directory and publishes no port; the
compose file additionally drops every capability. It runs the ONNX backend, because MLX is Apple
silicon and no Linux container can use it: **the same vectors, at a lower rate** — indexing this
repository's `docs/` took 5 minutes 4 seconds in the container against 38 seconds natively on
MLX. `manicule ask` needs a generator; the compose file points at an Ollama on the host, which is
one line to change.

> [!WARNING]
> **MCP is better run natively.** Handing a container's stdio to an assistant means the client
> spawning `docker compose run`, and the failure modes of that — a stale container, a build that
> has not happened, a volume that is not there — surface to the assistant as a tool that will not
> start. The container is for the CLI and for batch ingest; the two are the same index if they
> share a data directory.

[`docs/deployment.md`](docs/deployment.md) covers what the data directory holds, the permissions
it needs, and what publishing a port will require when there is one.

## The idea it is organized around

> **A citation carries a correct location, or none at all.**

That sounds like a small thing. It decides most of the architecture:

- Parsers return *located blocks*, not text, because structure is visible exactly once — while
  the markup is still in hand — and no downstream component can recover it afterwards.
- `Anchor` has an `Unlocated` member carrying a reason, rather than using `None`. "We could not
  determine a location" and "nobody asked" are different facts, and only one of them is a bug to
  fix.
- Every parser must pass a round-trip check: resolving an anchor returns the text the chunk
  claims. It is a test, not a convention, because the failure is silent — a citation pointing at
  a page that does not exist looks exactly like one that does.
- A locally mirrored page can say what it is a copy of. A file named `123456.html` cites as
  `123456.html` unless something supplies the document's real title and address, so an adjacent
  `123456.html.source.json` may — and the canonical identity and the local snapshot are then
  *both* kept, in two types neither of which is able to hold the other's fields. A citation that
  is precise about a file nobody else has is a correct location for the wrong thing.

The same instinct runs through the rest. Wherever something can be wrong quietly, there is a
guard that makes it loud: a mismatched embedding model, a chunk budget past what the model
attends to, a scanned PDF that yielded nothing, a plugin built for another version.

## Layout

| Package | What is in it |
|:---|:---|
| `src/manicule/core` | The types and protocols everything is written against. No implementation dependencies |
| `src/manicule/config` | One declarative layer over the config file, the environment and plugin manifests |
| `src/manicule/plugins` | Manifests, compatibility checking, entry-point discovery |
| `src/manicule/container` | Typed resolution and lifecycle. Assembled at startup, injected |
| `src/manicule/testing` | Conformance suites every implementation must pass |
| `src/manicule/app` | The application service. All the behavior, once, for every surface |
| `src/manicule/cli` | Twenty-eight commands over that service, and nothing else |
| `src/manicule/mcp` | Forty-one MCP tools over that service, and nothing else |
| `src/manicule/api` | Twelve HTTP route groups over that service, and nothing else |
| `src/manicule/extension` | A Chrome extension that hands this browser's Confluence session to a local manicule. No build step |
| `src/manicule/web` | Twelve areas of HTML — eleven pages and the frame they render inside. No build step, no new operation |
| `packages/manicule-plugin-example` | The smallest complete plugin. Copy it to start one |

The four surfaces are adapters: they parse arguments, call one method, and render what comes
back. A rule that lived in one of them would be a rule the others did not have — and two of them
are called unattended, so that is not a distinction worth risking.

**Nothing binds a network socket unless it is asked to.** The MCP server speaks stdio by default,
which opens no socket at all; every HTTP bind goes through one policy that starts at loopback,
and widening it takes an address somebody wrote down, an explicit flag no config file can supply,
and authentication switched on. Any one missing is a refusal.

**Nothing believes a forwarded address unless it was told to.** `X-Forwarded-For` is read only
from a peer inside `security.transport.trusted_proxies`, which is empty by default — so on an
ordinary install the header is not consulted at all, and every IP-based decision rests on a
socket peer rather than on a value the caller chose.

## Extending it

Everything pluggable is a `typing.Protocol`, and the ten registerable kinds are `Parser`,
`Chunker`, `Embedder`, `VectorStore`, `DocStore`, `RetrievalStage`, `Reranker`, `Generator`,
`Connector` and `Middleware`. Implementations are found through the `manicule.plugins`
entry-point group.

Built-in components use that same path — there is no shorter internal route — so the extension
mechanism is exercised by every installation rather than only by the people using it. A plugin
interface that nothing depends on rots without anyone noticing.

Importing manicule gives you the contracts and nothing heavier: no vector database, no model
runtime, no web framework. That boundary is enforced by a test, not by good intentions.

> [!CAUTION]
> **Plugins run in-process with full privileges** — the network, the filesystem, the environment.
> There is no sandbox and no `permissions` declaration, because manicule cannot enforce one and an
> unenforced guarantee is worse than an absent one. Install plugins you would run as yourself.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the plugin-authoring rules and the definition of
done.

## Development

> Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync --all-groups
uv run pytest
uv run ruff check . && uv run pyright
```

## License

**MIT.** See [`LICENSE`](LICENSE).

Nothing in manicule's dependency closure is copyleft. A plugin you write is yours to license as
you choose, and so is anything you build on top.

**One optional package is not MIT, and it is packaged separately for exactly that reason.**
[`manicule-mlx`](packages/manicule-mlx) is the Metal-native embedding backend for Apple silicon,
roughly four to five times faster than the default on the indexing path. It links
`mlx-embeddings`, which is GPL-3.0, so that package is **GPL-3.0-or-later**. manicule was
GPL-3.0-or-later itself until the backend moved out of it.

What that means:

| You install | You get | License of the combination |
|:---|:---|:---|
| `uv pip install manicule` | An MIT program, with `onnx` as the embedding backend. Runs everywhere | MIT |
| `uv pip install manicule manicule-mlx` | Faster on Apple silicon | GPL-3.0 on your machine |

Running it obliges you to nothing; the GPL's obligations attach to *distribution*. A plugin that
imports `manicule_mlx` is very likely a derivative work of it. A plugin that does not, is not.

**Switching backends never re-embeds.** `backend` is excluded from the embedding fingerprint's
identity, and the two agree to cosine 0.99999998 with identical retrieval ranking — asserted in
`packages/manicule-mlx/tests/test_parity.py` rather than assumed. Installing or removing the
package is an operation, not a migration.
