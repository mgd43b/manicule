# manicule

Self-hosted document search and answers. Index documents from wherever they live — disk,
git, Notion, Confluence, Drive, S3, the web — ask questions in natural language, and get
answers with citations that resolve to a real location in a real document. Usable from the
command line, over HTTP, and by AI assistants over MCP.

> **Early, and runnable.** All three surfaces work today: point it at a directory, search it,
> ask it questions, hand the same operations to an assistant over MCP, or serve them over HTTP.
> There is no release on PyPI yet, so it is installed from a checkout — [below](#install). The
> web UI is not built; see [`PLAN.md`](PLAN.md) for the shape of the whole and the order it is
> being built in.

```bash
manicule init                     # choose a backend this machine can run, write a config
manicule index ~/Documents        # walk it, parse it, chunk it, embed it
manicule search "retry policy"    # ranked passages, no model involved
manicule ask "how do retries work?"
manicule doctor                   # what is wrong, and what to do about it
```

Every command that emits data takes `--json` — before the command name, `manicule --json
search …`, because it is an option of `manicule` rather than of each command — and most of
them are also MCP tools, so an assistant reaches the same operations through `manicule start
--mcp-only`, and `manicule start --transport http` serves them over HTTP with an OpenAPI
document at `/api/docs`. All three emit the same envelope, and the shape is a contract written
down in [`docs/surfaces.md`](docs/surfaces.md).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
git clone https://github.com/mgd43b/manicule && cd manicule
uv sync --all-extras
uv run manicule --version
```

`--all-extras` is the whole system: the parser stack, the storage stack, the embedding backend
this machine can run, the connectors, and the optional cross-encoder reranker that comes with
torch.

Everything below writes `manicule`; from a checkout that is `uv run manicule`, or
`.venv/bin/manicule` if you would rather not type `uv run` each time.

**Two embedding backends, and one of them is chosen for you.** `manicule init` probes the
machine: `mlx` on Apple silicon, which runs the embedder on Metal in-process, and `onnx`
everywhere else. Which one you get changes how long an ingest takes and **never what comes out
of it** — the `backend parity (macOS)` job in CI exists to hold that line, comparing vectors
from both backends on a runner that can build both. If it ever goes red the fix is to the
code, not to the tolerance.

## First run

The shortest path from a clone to an answer about your own documents. `manicule init` writes a
config file; the first `index` downloads the embedding model and then does not again. For the
default `BAAI/bge-m3` that is about 1.1 GB on Apple silicon — the MLX conversion, in fp16 —
and about 2.3 GB elsewhere, where the ONNX export is what runs.

```bash
manicule init
manicule index docs                       # this repository's own design documents
manicule search "how are citations verified"
manicule ask "what does an anchor carry when the location is unknown?"
```

`search` needs nothing but the embedder. **`ask` additionally needs a generator**, and the
default configuration expects [Ollama](https://ollama.com) on `localhost:11434` serving
`qwen2.5:14b`:

```bash
ollama pull qwen2.5:14b
manicule config set llm.model qwen2.5:14b   # or any model that Ollama is serving
```

`manicule doctor` reports what is wrong and what to do about it, and it is the first thing to
run when something does not work. `manicule doctor --fix` repairs what it can — today that is
seeding the tree-sitter grammars, which `manicule init` already does and which is the one thing
here that may use the network.

## The four surfaces

**The command line** is nineteen commands; `manicule --help` lists them. Under `--json` the
result envelope is the whole of stdout, so a failed run piped into `jq` reads an empty stream
rather than a prose error.

**The HTTP API** is eleven route groups over the same service — documents, chat with SSE
streaming, conversations and shareable links, collections, tags, admin, plugins, auth, a
workbench, a websocket channel — plus an embeddable chat widget. `manicule start --transport
http` serves it, on loopback unless three separate things say otherwise.

**The browser surface** is twelve areas of server-rendered HTML at `/ui`, on the same socket:
chat with streaming citations, confidence and feedback; documents, their chunks, the trash and
restore; collections and tags; connectors, plugins, workspaces, health, an admin dashboard and
your own API keys. Command palette on `Ctrl`/`Cmd`+`K`, keyboard navigation, dark mode.

It adds **no build toolchain**: Jinja2 templates, one hand-written stylesheet and one
hand-written script, so `uv sync` is still the whole install and the container image stays free
of Node. It also adds **no operation** — every page reads through a service method that already
has a route, so there is no upload and no configuration write here either.
[`docs/web.md`](docs/web.md) has the reasoning, including what that costs.

**The MCP server** is nineteen tools over the same service, and it speaks stdio by default —
which opens no socket at all. To let Claude Code use your index:

```bash
claude mcp add manicule -- "$(pwd)/.venv/bin/manicule" start --mcp-only
```

which writes `.mcp.json` beside the project:

```json
{
  "mcpServers": {
    "manicule": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/manicule",
      "args": ["start", "--mcp-only"]
    }
  }
}
```

The server is also reachable as `python -m manicule.mcp`, for a client that would rather name
an interpreter and a module than trust a console script to be on the PATH it happens to have.

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

The build downloads the weights and the grammars, and the resulting image is about 3.4 GB.
That cost is paid at `build`, where a long step is legible, rather than inside the first
`index`, where it looks like a hang. Indexing this repository's `docs/` — about 600 kB of
markdown — takes something under four minutes on an M-series Mac running Docker Desktop.

The image runs as an unprivileged user with a `0700` data directory and publishes no port; the
compose file additionally drops every capability. It runs the ONNX backend, because MLX is
Apple silicon and no Linux container can use it: **the same vectors, at a lower rate.**
`manicule ask` needs a generator; the compose file points at an Ollama on the host, which is
one line to change.

**MCP is better run natively.** Handing a container's stdio to an assistant means the client
spawning `docker compose run`, and the failure modes of that — a stale container, a build that
has not happened, a volume that is not there — surface to the assistant as a tool that will
not start. The container is for the CLI and for batch ingest; the two are the same index if
they share a data directory.

[`docs/deployment.md`](docs/deployment.md) covers what the data directory holds, the
permissions it needs, and what publishing a port will require when there is one.

## The idea it is organised around

**A citation carries a correct location, or none at all.**

That sounds like a small thing. It decides most of the architecture:

- Parsers return *located blocks*, not text, because structure is visible exactly once —
  while the markup is still in hand — and no downstream component can recover it afterwards.
- `Anchor` has an `Unlocated` member carrying a reason, rather than using `None`. "We could
  not determine a location" and "nobody asked" are different facts, and only one of them is
  a bug to fix.
- Every parser must pass a round-trip check: resolving an anchor returns the text the chunk
  claims. It is a test, not a convention, because the failure is silent — a citation
  pointing at a page that does not exist looks exactly like one that does.

The same instinct runs through the rest. Wherever something can be wrong quietly, there is a
guard that makes it loud: a mismatched embedding model, a chunk budget past what the model
attends to, a scanned PDF that yielded nothing, a plugin built for another version.

## Layout

| | |
|---|---|
| `src/manicule/core` | The types and protocols everything is written against. No implementation dependencies |
| `src/manicule/config` | One declarative layer over the config file, the environment and plugin manifests |
| `src/manicule/plugins` | Manifests, compatibility checking, entry-point discovery |
| `src/manicule/container` | Typed resolution and lifecycle. Assembled at startup, injected |
| `src/manicule/testing` | Conformance suites every implementation must pass |
| `src/manicule/app` | The application service. All the behaviour, once, for every surface |
| `src/manicule/cli` | Nineteen commands over that service, and nothing else |
| `src/manicule/mcp` | Nineteen MCP tools over that service, and nothing else |
| `src/manicule/api` | Eleven HTTP route groups over that service, and nothing else |
| `src/manicule/web` | Twelve areas of HTML over that service. No build step, no new operation |
| `packages/manicule-plugin-example` | The smallest complete plugin. Copy it to start one |

The four surfaces are adapters: they parse arguments, call one method, and render what comes
back. A rule that lived in one of them would be a rule the others did not have — and two of them
are called unattended, so that is not a distinction worth risking. A test runs the same operation
through all of them and compares the results.

**Nothing binds a network socket unless it is asked to.** The MCP server speaks stdio by
default, which opens no socket at all; every HTTP bind goes through one policy that starts at
loopback, and widening it takes an address somebody wrote down, an explicit flag no config file
can supply, and authentication switched on. Any one missing is a refusal.

**Nothing believes a forwarded address unless it was told to.** `X-Forwarded-For` is read only
from a peer inside `security.transport.trusted_proxies`, which is empty by default — so on an
ordinary install the header is not consulted at all, and every IP-based decision rests on a
socket peer rather than on a value the caller chose.

## Extending it

Everything pluggable is a `typing.Protocol`: `Parser`, `Chunker`, `Embedder`, `VectorStore`,
`DocStore`, `RetrievalStage`, `Reranker`, `Generator`, `Connector`, `Middleware`.
Implementations are found through the `manicule.plugins` entry-point group.

Built-in components use that same path — there is no shorter internal route — so the
extension mechanism is exercised by every installation rather than only by the people using
it. A plugin interface that nothing depends on rots without anyone noticing.

Importing manicule gives you the contracts and nothing heavier: no vector database, no model
runtime, no web framework. That boundary is enforced by a test, not by good intentions.

**Plugins run in-process with full privileges** — the network, the filesystem, the
environment. There is no sandbox and no `permissions` declaration, because manicule cannot
enforce one and an unenforced guarantee is worse than an absent one. Install plugins you
would run as yourself.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the plugin-authoring rules and the definition
of done.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --all-groups
uv run pytest
uv run ruff check . && uv run pyright
```

## Licence

**GPL-3.0-or-later.** See [`LICENSE`](LICENSE).

The embedding runtime decided this. `mlx-embeddings` is GPL-3.0, and running embeddings
in-process on Apple Silicon is what keeps `uv tool install manicule` a single command with no
server to operate alongside it. Changing the licence was chosen over changing the dependency.

**This reaches plugins.** They load in-process, in the same address space, through
`importlib.metadata` entry points — not over a socket or a subprocess boundary. A plugin
distributed to others is very likely a derivative work under the GPL, which was not true when
this project was MIT. That is a real consequence for the community registry
([#8](https://github.com/mgd43b/manicule/issues/8)) and it is stated here rather than
discovered by whoever publishes the first one. Nothing in this repository decides it for you:
take advice if you intend to distribute a plugin under other terms.
