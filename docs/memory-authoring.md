# Memory corpus authoring

Two things let manicule serve as durable memory for an assistant rather than only as search over
documents something else wrote: a caller can author a document, and `[[wikilinks]]` in markdown
become graph edges. This page describes both as they are built.

---

## 1. Why

Every document enters manicule through a connector pulling from a source, and until `document_create`
nothing let a caller supply a body and have it become part of the corpus. That single gap was what
stood between manicule and being an assistant's memory store: an assistant has to be able to *write*
a memory, not only search one.

The corpus that motivates it is real: ~500 markdown facts, one self-contained fact per file, in a git
repository. They were previously scattered across per-project memory directories keyed on a lossy
encoding of the working directory. See `project_manicule_memory_corpus_direction` in this project's
memory collection for the decision and the alternatives weighed.

Corpus shape, which the design assumes:

```markdown
---
name: project_metallb_cluster_vlan_vip     # equals the filename stem
description: One line. This is what retrieval ranks on.
metadata:
  node_type: memory
  type: project | feedback | reference | user | hardware
---

Prose. Links to related facts as [[project_home_apiserver_vip]].
```

## 2. Non-goal: manicule is not a note database

`document_create` **writes a markdown file into the `filesystem` connector's configured root and
then ingests that path.** The file remains the record. The connector remains how content enters.
Manicule stays retrieval infrastructure over sources.

This is load-bearing rather than stylistic:

- it keeps data-ownership, migration and backup obligations out of this project;
- the corpus stays a directory of markdown that git versions and any other tool can read, so there
  is always an exit;
- it introduces no second notion of what a document is, and no storage path that bypasses the
  pipeline.

If an implementation detail starts requiring manicule to own document state that is not a file, the
design has drifted — stop and raise it.

## 3. What this is built on

| Capability | Where |
|---|---|
| Collections, manual members + `auto_rules` `CollectionRule`, membership evaluated at read time | `core/organization.py`, `api/routes/organization.py` |
| Collection-scoped search; an unknown collection name is **refused**, never dropped | `app/service.py::_collection_scope` |
| Workspaces, tenancy enforced in the store handle and again at the surface | `docs/storage.md` §4.2 |
| `chunk_relations` — directional, idempotent, with a conformance suite | `testing/contracts.py::assert_chunk_relation_store_contract` |
| `document_versions`, trash / soft delete | `docs/storage.md` §11 |
| Filesystem connector, with root containment already enforced | `connectors/filesystem.py::contains` |
| Markdown front matter skipped as content rather than read as a heading | `parsers/config.py::MarkdownConfig.front_matter` |
| Four surfaces over one service, parity enforced | `docs/surfaces.md`, `tests/app/test_surface_parity.py` |
| Plugin extension via the `manicule.plugins` entry-point group | `packages/manicule-plugin-example` |

---

## 4. `document_create`

### 4.1 Shape

On `ApplicationService`:

```python
async def document_create(
    self,
    *,
    collection: str,      # name, not id — a name is what a person and an assistant have
    slug: str,            # identity; becomes the filename stem
    body: str,            # complete markdown, front matter included
    overwrite: bool = False,
) -> DocumentCreated
```

### 4.2 Identity is the slug, not the title

`document_id` derives from `(workspace, source, source_id)`, and for the filesystem connector
`source_id` is the path. Whatever this operation picks becomes the document's identity permanently,
so it cannot be derived from anything mutable.

**The caller supplies a slug; manicule derives `<collection>/<slug>.md` beneath the connector
root.** A title can change without breaking identity, citations or inbound links. It also matches
the corpus convention already in use, where the filename stem equals the `name:` in front matter.

The caller never supplies a path. That is what keeps traversal impossible by construction rather
than by validation.

### 4.3 Collection membership is part of the operation

Projects *are* collections here. A create that did not take one would drop every document into an
unscoped pile, and a caller would have to follow every write with a `collection_add` that might
fail separately.

The name resolves through the same `find_collection` path `_collection_scope` uses, so an unknown
name is refused rather than silently widening scope, and membership is written in the same
operation as the document. `DocumentCreated.member` reports whether that second write happened, so
a caller that found the document in search but not in its collection has something in the result
that says so.

**This operation is not the only way in, and for a long time it behaved as though it were.** The
membership it writes is a manual one, per document. A file that reached the same directory by a
sync — a corpus pulled from git, a file another tool wrote, or §4.5's kept file that a later sync
picked up — joined no collection at all, and the only symptom was a collection-scoped search
returning less. Give the collection a `uri_prefixes` rule naming its directory beneath the root
(`docs/storage.md` §11.2) and the directory *is* the membership, so a document joins by arriving
however it arrived. `manicule doctor`'s `authoring` check reports a configured collection that
has no such rule, because nothing else about that state is visible.

The manual write stays regardless, and is not made conditional on a rule existing. A collection
here may legitimately have no rule, and a `document_create` that assigned membership one way
when a rule was present and another way when it was not would be a second notion of what a
collection contains — which is the thing §11.2 keeps down to one.

### 4.4 Synchronous through publication

**It returns only once the document is published and searchable.** An assistant that writes a
memory and immediately searches for it must find it; anything else makes the tool feel broken in
exactly the situation it exists for.

The copy-on-write publication flip gives the atomic boundary. Embedding a single small markdown
file is not a reason to go asynchronous — and a filesystem watcher firing after the write is
*worse* here, which is part of the point.

### 4.5 Partial failure: keep the file

The file is written, then ingestion fails. The contract is:

**Keep the file. Return a failure result naming the path.**

A later sync indexes it. Deleting a caller's content because embedding hiccuped is the worse
failure of the two, and it contradicts §2 — the file is the record, so the record survives even
when the index does not.

`DocumentCreated.indexed` is false, and `manicule.app.dispatch.run_op` turns that into `ok: false`
with the payload still attached, the same way it reports an ingest run that must be retried.
`Envelope.one_outcome_shape` holds the pair together: a failed `document_create` envelope may carry
data only for a document that reports itself unindexed, and only when the error beside it names the
same path.

### 4.6 Overwrite is refused by default

The corpus convention is to update the fact that already exists rather than add a near-duplicate,
so callers *will* write to an existing slug deliberately. They will also do it by accident.

The default refuses, and `overwrite` proceeds. The refusal names the document that already holds
the slug, so the caller can read it and decide. A file on disk that nothing has indexed counts as
holding the slug too — a write whose ingest failed, or a file another tool put there — because
silently overwriting it would lose content manicule has never seen.

**`overwrite` is the whole of update. There is no patch or append.** The corpus is one
self-contained fact per file, not a log, and the convention is that a caller reads the existing
fact, revises it, and writes the whole thing back. A blind append that never read the file is also
the fastest way to corrupt front matter, since it cannot know where the `---` fence ends. If
append-without-read turns out to be needed later it is additive.

### 4.7 Front matter belongs to the caller

The caller supplies complete markdown including front matter. Manicule does not synthesize `name`,
`description` or `type`. The corpus has its own schema and its own conventions; a retrieval system
inventing metadata for someone else's documents is how two sources of truth start.

What is validated is that what arrives parses as intended. An empty body is refused — there would
be nothing to index or cite. A front-matter fence that opens and never closes is refused, because
left alone CommonMark reads its last line as a setext heading and hangs every heading path in the
document beneath a heading nobody wrote. One byte is added and no other: a trailing newline where
the body has none, which is a property of the file rather than of the content.

### 4.8 Path safety

The slug is a single path segment. Separators, `.`, `..`, absolute paths and anything that does not
survive normalization unchanged are refused by name. The collection is held to the same rule,
because it is a directory beneath the root as well as a scope. The final path is then resolved and
checked for containment with `FilesystemConnector.contains` — the same check that decides whether a
stored document's path may be *read*. §4.2 already means the caller never names a path; the
containment check is a fact where the derivation is an argument, and this is the one boundary where
being wrong means writing outside the corpus.

### 4.9 Surfaces

Behavior lives in `ApplicationService` and nowhere else. The MCP tool, the CLI command
(`manicule document create <collection> <slug>`, body from `--file` or standard input) and the HTTP
route (`POST /api/v1/documents`) produce byte-identical envelopes;
`tests/app/test_document_create.py` compares all three.

**The browser surface has no authoring form, deliberately.** `POST /ui/documents` is asserted
*absent* by `tests/web/test_boundaries.py`: that surface renders envelopes and does not author, and
adding a form here would undo that decision from a different package — which is the exact failure
that file exists to prevent. The browser surface's part in parity is rendering what the tool
reports, which it already does for the documents listing.

### 4.10 Both transports

**`document_create` works over stdio *and* over the network.** Both:

- **stdio** — local development, the CLI, working offline, and the transition period before
  anything is deployed.
- **network** — the target deployment is manicule in k8s with **no manicule process on the laptop
  at all**. If authoring were stdio-only, every write would require a local daemon, which is
  precisely the thing being eliminated. A read-only network surface makes the deployment pointless.

**The transport decides the surface, exactly as it does for everything else.** No new mode, no
configuration flag:

| Transport | Surface |
|---|---|
| stdio | everything, including `document_create` |
| socket | the read-only set **plus `document_create`**, and nothing else that mutates |

The property that mattered is that the absence of every *other* write tool stays **mechanical
rather than reasoned**, and it does. `_Registrar.tool` registers a tool only when its
`readOnlyHint` is true or its name is in `manicule.mcp.server.NETWORK_AUTHORING` — a frozenset
holding one name — so there is no handler behind any other write tool on a socket. An absence,
not a refusal. `tests/api/test_routes.py` asserts that row as a set operation: the published set
*is* the read-only set plus that constant, so a second write tool cannot drift in behind the
first.

**That set does not depend on authentication, and the attempt to make it depend on
authentication is worth recording.** When `--no-authentication` was added
([`surfaces.md` §6](surfaces.md#6-where-a-server-listens)) the set was briefly emptied for
`security.auth.mode = none`, on the reasoning that without a credential
`manicule.api.security.Principal` resolves an anonymous caller to `admin`, `require_network_member`'s
member floor is cleared by everybody, and anything routing to the port could then write into a
corpus read back as standing instructions.

The reasoning about the exposure was right. The conclusion was wrong, because it made *this
deployment* impossible: manicule serving this corpus to Claude and Codex on machines running no
manicule of their own, where authoring is the entire point of the socket. They could search the
corpus and not write to it. A read-only network surface makes the deployment pointless — the
sentence §4.10 already used, applied to its own mitigation.

So `security.auth.mode` decides **who may call** `document_create` and never whether the socket
carries it. On an unauthenticated bind the answer to "who" is everyone who can route to the
port, over the MCP mount and over `POST /api/v1/documents` alike, and that is accepted rather
than mitigated: it takes an argument no configuration file can supply, the startup banner names
this corpus at the moment it becomes writable, and `manicule doctor` reports it as failing for
as long as it holds.

Two constraints ride along and both are enforced:

- **Auth is non-negotiable on a socket unless an operator says otherwise at a terminal.**
  `manicule.app.bind.require_authoring_authentication`
  refuses when authoring is configured and `security.auth.mode` is `none` — loopback included,
  where `resolve_bind` asks for nothing. Two callers, because there are two ways a socket carries
  the tool: `manicule.mcp.serve.address_for` for `--mcp-only`, and `manicule.api.app.build_app`
  for the application everything else is served from — the second so the refusal fires when a
  container entry point or a production ASGI server is doing the listening. The condition is
  authoring being *configured* rather than the tool existing, so an installation that never wanted
  it is not asked to turn authentication on for a feature it does not use.

  **`--no-authentication` waives it**, and the waiver is the point rather than a hole in it.
  The refusal exists so that nobody serves authoring unauthenticated *by omission*; an operator
  who typed the argument is asserting the network in front of the process is one they own — a
  private LAN, or a cluster behind an ingress that authenticates for us. That is the target
  deployment: MCP over HTTPS behind a PKI, with no manicule on the laptop at all. It opens both
  doors at once, `document_create` and `POST /api/v1/documents`, because they are the same write
  and an anonymous administrator clears the member floor on either.
- **Scope is a configured writable collection**, not any collection the workspace holds. Creating a
  collection is therefore not also the act of granting write access to it.

`tests/api/test_routes.py::ABSENT_TOOLS` does not name `document_create`, and says why: `index_path`
walks any directory the process can read and `config_set` rewrites the running configuration,
whereas this writes one document, to one workspace, in one configured collection, beneath the
connector root, at a path the caller never supplies. Bounded authority, not unbounded. That list's
value is that every entry explains itself, so the absence explains itself too.

**Why the extra care, given it is only one tool:** this corpus is read as *instructions*, not just
data. `feedback_` memories are treated as standing instructions by every assistant that recalls
them. So write access here is the ability to inject behavior into future sessions across every
project — a materially higher-stakes write than indexing a directory, and the reason the default
stays off and the scope stays narrow.

### 4.11 Configuration

```toml
[authoring]
source = "memories"        # a configured filesystem connector instance
collections = ["memory"]   # the collections that may be authored into
```

Both are empty by default, and empty means authoring is off: the tool is published on every surface
and refuses every call naming these settings. Both are required together, because either alone
describes an operation that cannot run — a source with no collection has nowhere to put a document
that is not an unscoped pile, and collections with no source have no root to be written beneath.

Each name is a collection **and** a single path segment beneath the root, so the collection
wants a rule saying the same thing. Once per collection:

```bash
manicule collection create memory --uri-prefix /corpus/memory
```

Configuration deliberately does not create these or set their rules. Creating a collection is
already a separate act from granting write access to it — that separation is the whole reason
`collections` is a list of names rather than "every collection" — and a setting that silently
created and re-ruled workspace objects on startup would undo it. `manicule doctor` reports the
gap instead: `failing` for a configured collection the workspace does not have, `degraded` for
one whose rule does not select its own directory, each naming the command that fixes it.

---

## 5. The wikilink middleware

A plugin: `packages/manicule-plugin-wikilinks`, registered under `keys.MIDDLEWARE`.

### 5.1 What it does

During ingest it scans a stored document's chunk text for `[[slug]]`, resolves each to a document,
and writes a `chunk_relations` edge — then looks the other way, for documents already stored that
link to *this* one, and writes those too.

The graph is already written in the corpus — the homelab memories alone reference
`[[project_metallb_cluster_vlan_vip]]` nine times — it was simply never extracted. This makes the
existing 500 documents a graph without anyone editing them.

### 5.2 Relation types

Two shapes appear in the corpus and they do not mean the same thing, so there are two types:

- `ChunkRelationType.MENTIONS` — a wikilink in prose. A soft reference: evidence that two documents
  are about related things.
- `ChunkRelationType.LINKS_TO` — a wikilink that is the content of its line: a list item, with or
  without a leading verb (`relates_to [[x]]`, `- [[x]]`), or a short label followed by links and
  nothing else (`Related: [[a]], [[b]]`). The author asserting a relationship.

They are not collapsed. Ranking by mention count and ranking by declared links give different
answers on the same corpus, and that difference is unrecoverable once the two are one type.

### 5.3 Forward references survive

A `[[slug]]` may name a memory that does not exist yet — the writing convention explicitly allows
it, as a marker of something worth writing later. Unresolved links are therefore **normal, not
errors**.

Of the two available designs — persist unresolved edges and resolve lazily, or re-resolve on each
target publish — this is the **second**, and the first is not available: `chunk_relations` has real
foreign keys to `chunks`, so an edge to a document that does not exist has nothing to point at. So
when a document is stored, the middleware searches the lexical index for stored chunks naming its
slug, re-scans each with the real rules, and writes the edges that now resolve. The search is
bounded by `inbound_limit` (200 by default); a linking document outside that bound acquires its
edge the next time it is itself scanned or repaired.

### 5.4 Slug normalization, and what a slug can reach

The corpus is inconsistent between hyphens and underscores — `feedback-commit-everything.md` and
`feedback_btctrader_hands_off.md` both exist, and links are written both ways. Resolution
normalizes rather than matching literally: every separator run folds to a single hyphen, case
folds, a `.md` suffix is dropped, and `[[target|alias]]` and `[[target#heading]]` address `target`.
Without that, roughly half the graph would silently fail to connect.

A slug is a filename stem, and stems are not unique across sources — so `sources` bounds which
documents a link may resolve into. Empty means every source, which is right for an installation
whose corpus is one directory of memories and wrong for one that also indexes a wiki: an unscoped
link that lands on a same-named page elsewhere is a *wrong* edge, which is the worse direction.

### 5.5 The existing corpus is the point — the stage has its own fingerprint

~500 documents already carry wikilinks. A middleware that only ran on new ingests would leave the
graph empty until something happened to re-ingest everything, so backfill is not optional.

There is **no backfill command**, and this is **not folded into `ChunkFingerprint`**.
`GlossaryFingerprint` is the precedent, and its docstring is written about exactly this situation:
glossary detection is "a separate stage with rules of its own that change independently" of
parsing, chunking and embedding, and inferring its freshness from `parse_fp` would "leave the media
types nobody bumped stale for ever". Relation extraction is the same shape — its rules (slug
normalization, relation typing, §5.2–5.4) move on their own schedule, and none of the existing
fingerprints shift when they do.

So `RelationFingerprint` is modeled on `GlossaryFingerprint`, including `middleware` among its
`IDENTITY_FIELDS`: an edge names a *chunk*, and which chunk holds a link follows from boundaries any
hook may move in `after_parse`, carrying no declaration at all. It is recorded in
`documents.relation_fp` and backfill is the repair selector that already exists:

```bash
manicule document reindex --stale-relations [--dry-run] [--batch N]
```

`reindex.rescan_stale_relations` is the verb behind it, and it keeps the glossary sweep's cost
boundary exactly: it builds no pipeline, so there is no chunker, no embedder, no vector store and
no blob store in it — the middleware chain and the document store are all it needs. On first
enable the selection is the entire corpus, because every row records `NULL` until something has
scanned it.

On first enable that selection is the entire corpus, because every row records `NULL` until
something has scanned it.

**Who writes what.** The plugin writes the edges and states its own rules through
`ChunkRelationExtractor.relation_rules`; the pipeline writes the fingerprint, in `_observe`, **only
on the path where every `after_store` hook returned**. An extractor that raises leaves the column
exactly as it was, so the document stays selected by the next repair — where stamping regardless
would claim an extractor had run over a document it failed on. An installation with no relation
middleware configured records `RelationFingerprint.disabled()`, which is a value an operator can
read and is what makes installing one select the whole corpus.

**An empty result is a derived result.** A document scanned under the current rules that contains no
wikilinks records the fingerprint with zero edges. That is a different fact from a document never
scanned at all, which records `NULL`. Collapse the two and every link-free document — most of a
corpus — is re-selected on every repair, for ever.

### 5.6 Why a plugin

`[[wikilinks]]` are a memory/Obsidian idiom, not a universal markdown one. Keeping the convention
out of the core parser is correct on its own, and it exercises the public extension path on a real
component rather than only on the example.

## 6. Testing

- `tests/app/test_document_create.py` — surface parity across the three envelope adapters, identity
  stability under a retitle, the overwrite refusal naming the existing document, partial failure
  leaving the file and reporting `ok: false`, and containment for every shape of unsafe slug.
- `tests/api/test_routes.py` — the network surface asserted as a set operation, and `ABSENT_TOOLS`
  explaining why authoring is not in it.
- `tests/mcp/test_transports.py` — a socket serving configured authoring without authentication
  refuses to start, and the same configuration over stdio does not.
- `tests/mcp/test_stdio.py` — `document_create` is in `WRITE_TOOLS`, so the read-only session must
  not call it and the whole surface is still carried over a pipe.
- `tests/ingest/test_relation_lineage.py` — the fingerprint a chain produces, where the pipeline
  stamps it, the three states that are easy to conflate, and what the repair selects.
- `packages/manicule-plugin-wikilinks/tests` — the rules, both directions, idempotent re-ingest
  producing one edge, a forward reference resolving when its target is published, and hyphen and
  underscore variants reaching the same target. Against a real store, so `relate`'s idempotence and
  the lexical index's tokenization are the product's rather than a fake's.

## 7. Out of scope

- Deployment to k8s, the HTTPS/PKI surface, and the commit-and-push sidecar.
- Migrating the corpus off qmd. It keeps running over the same files throughout.
- Bidirectional sync between the cluster corpus and a local clone. The accepted trade is a
  read-only clone pulled on demand.
