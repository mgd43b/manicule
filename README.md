# manicule

Self-hosted document search and answers. Index documents from wherever they live — disk,
git, Notion, Confluence, Drive, S3, the web — ask questions in natural language, and get
answers with citations that resolve to a real location in a real document. Usable from the
command line, from a browser, and by AI assistants over MCP.

> **Early, and runnable.** `uv tool install manicule` gives you a working index: point it at
> a directory, search it, ask it questions. The HTTP API and the web UI are not built yet; see
> [`PLAN.md`](PLAN.md) for the shape of the whole and the order it is being built in.

```bash
manicule init                     # choose a backend this machine can run, write a config
manicule index ~/Documents        # walk it, parse it, chunk it, embed it
manicule search "retry policy"    # ranked passages, no model involved
manicule ask "how do retries work?"
manicule doctor                   # what is wrong, and what to do about it
```

Every command takes `--json`, and every one of them is also an MCP tool — so an assistant
reaches the same operations through `manicule start --mcp-only`. The output shape is a
contract, written down in [`docs/surfaces.md`](docs/surfaces.md).

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
| `packages/manicule-plugin-example` | The smallest complete plugin. Copy it to start one |

The two surfaces are adapters: they parse arguments, call one method, and render what comes
back. A rule that lived in one of them would be a rule the other did not have — and the MCP
tool is the one called unattended, so that is not a distinction worth risking. A test runs the
same operation through both and compares the results.

**Nothing binds a network socket unless it is asked to.** The MCP server speaks stdio by
default, which opens no socket at all; the HTTP transport binds loopback, and widening it
takes an address somebody wrote down, an explicit flag no config file can supply, and
authentication switched on. Any one missing is a refusal.

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
