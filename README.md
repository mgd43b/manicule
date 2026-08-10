# manicule

Self-hosted document search and answers. Index documents from wherever they live — disk,
git, Notion, Confluence, Drive, S3, the web — ask questions in natural language, and get
answers with citations that resolve to a real location in a real document. Usable from the
command line, from a browser, and by AI assistants over MCP.

> **Early.** This repository currently contains the core contracts and the wiring that
> assembles them. The parsers, the storage, the models and the interfaces are being built
> against these seams; see [`PLAN.md`](PLAN.md) for the shape of the whole and the order it
> is being built in.

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
| `packages/manicule-plugin-example` | The smallest complete plugin. Copy it to start one |

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
