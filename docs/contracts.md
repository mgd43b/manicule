# Core contracts

The types and protocols everything else is written against. Tickets
[#1](https://github.com/mgd43b/manicule/issues/1) and
[#4](https://github.com/mgd43b/manicule/issues/4).

Signatures below are design, not implementation — they fix the shape of the seams, not how
anything behind them works.

Two of these are **expensive to change later** and are called out where they appear:
`Anchor`, because changing it invalidates every stored citation, and `RetrievalStage`,
because changing it invalidates every recorded evaluation result.

---

## 1. Anchor — where a citation points

The most important type in the system, and the one OpenDocuments gets wrong: it splits
extracted PDF text on blank-line runs and numbers the fragments as pages, producing
citations that point at pages which do not exist.

**The rule this type exists to enforce: a location is correct, or it is absent.** There is
no "best guess" member.

```
Anchor = PageAnchor | HeadingAnchor | LineAnchor | CellAnchor | Unlocated

PageAnchor      page: int, rects: list[Rect]        # PDF, PPTX slides
HeadingAnchor   path: list[str], fragment: str|None # Confluence, Markdown, DOCX, HTML
LineAnchor      start: int, end: int, symbol: str|None  # source code
CellAnchor      sheet: str, ref: str                # XLSX — "Sheet1!B4:D12"
Unlocated       reason: str                         # parser could not determine one
```

**Design decisions worth stating:**

- **`Unlocated` is a real member, not `None`.** It carries a reason, so "we do not know"
  is distinguishable from "nobody asked", and it shows up in diagnostics rather than
  silently degrading.
- **`rects` is a list, never a merged envelope.** A quote spanning a column break has two
  boxes; merging them produces a rectangle covering text that was not quoted.
- **`fragment` is the anchor a URL can link to** — Confluence derives heading anchors from
  heading text, so a citation deep-links to the exact section rather than the page.
- Every anchor must satisfy a **round-trip check**: resolving it returns the text the
  chunk claims. This is a test obligation on every parser, not a convention.

⚠️ **Locked once ingest runs.** Changing the shape invalidates every stored citation.

## 2. Content types

```
RawDocument     source_id, uri, media_type, bytes|str, metadata
ParsedBlock     kind, text, anchor, heading_path, lang, metadata
Chunk           id, document_id, text, embed_text, anchor,
                heading_path, kind, position, token_count
Document        id, source, uri, title, content_hash, version_token,
                original_ref, status, metadata
```

**`ParsedBlock.kind`** — `prose | heading | table | code | list | panel | media`. This is
what lets the chunker keep tables and code blocks whole instead of splitting them at a
character count.

**`text` versus `embed_text`.** `text` is what gets cited and shown. `embed_text` is what
the embedder sees, and carries the heading breadcrumb prefixed to it — a section called
"Configuration" is unretrievable without knowing what it configures. Storing both means
the citation is not polluted by retrieval scaffolding.

**`version_token`** is opaque and connector-defined: a git blob SHA, a Confluence
`version.number`, an S3 ETag. The ingest pipeline compares it and never interprets it.

**`original_ref`** points at the retained source bytes, so re-parsing never means
re-fetching.

## 3. Protocols

```
Parser
    media_types: set[str]
    parse(raw: RawDocument) -> AsyncIterator[ParsedBlock]

Chunker
    chunk(blocks: Iterable[ParsedBlock]) -> list[Chunk]

Embedder
    fingerprint: EmbedFingerprint
    encode(texts: list[str]) -> TokenStates      # tier A: pre-pooled
    embed(texts: list[str]) -> list[Vector]      # tier B: pooled only

VectorStore
    upsert(chunks: list[Chunk], vectors: list[Vector]) -> None
    search(vector: Vector, k: int, filter: Filter|None) -> list[Candidate]

DocStore
    # metadata, collections, tags, versions, conversations

RetrievalStage
    name: str
    run(query: Query, candidates: list[Candidate]) -> list[Candidate]

Generator
    generate(query: Query, context: Context) -> AsyncIterator[Token]

Connector
    discover(watermark: Watermark|None) -> AsyncIterator[DiscoveredDoc]
    fetch(ref: DocRef) -> RawDocument
    reconcile() -> AsyncIterator[SourceId]       # for deletion detection
```

**`Parser` returns blocks, not text.** Structure is discovered once, by the component that
can actually see it, and never re-derived downstream from prose.

**`Embedder` has two tiers.** Tier A returns pre-pooled token states and manicule does the
pooling; tier B returns finished vectors. The distinction exists because a backend's
convenience output cannot be trusted to be the model's own pooling: `mlx-embeddings`
computes its XLM-RoBERTa `text_embeds` with mean pooling unconditionally, while the chosen
model pools with CLS, and it binds `last_hidden_state` to the *pooled* vector on some
architectures and to genuine token states on others. Both produce well-shaped, normalised
vectors and raise nothing. Tier B backends are therefore admitted only by measurement, since
they cannot be verified by inspection. See [`embeddings.md`](embeddings.md) §3.2 and §4.1.

**`Connector.discover` takes a watermark and `reconcile` exists separately.** Incremental
sync tells you what changed; it cannot tell you what was deleted, because a deleted page
simply stops appearing. Without a reconciliation pass the index serves removed documents
forever. Making it part of the protocol means no connector can quietly omit it.

**`RetrievalStage` is uniform — `candidates in, candidates out`.** A pipeline is a declared
list of stages, so the evaluation harness can compare whole pipelines by configuration
rather than by editing code. This is what makes "no retrieval feature without a measured
improvement" mechanically enforceable rather than a discipline to remember.

⚠️ **`RetrievalStage` is locked after the evaluation harness exists.** Widening it later
invalidates every recorded result. Widen it now if at all.

## 4. Settled

**Metadata store: SQLite plus LanceDB**, and therefore SQLite FTS5 for BM25. Sixteen
relational tables belong in a relational store; a columnar vector store is the wrong tool
for joins and transactional updates. See `PLAN.md` §2.

**Vector dimensionality is a runtime parameter**, read from `Embedder.fingerprint` — never
a constant. The vector table is created at first ingest, and ingest must refuse to start
if the fingerprint does not match what the index was built with.

## 5. Deliberately absent

**No `permissions` field on plugins.** OpenDocuments ships one that is typed, tested, and
enforced nowhere. Plugins run in-process with full privileges; the documentation says so.
An unenforced guarantee is worse than an absent one.

**No `score` on `Chunk`.** Scores belong to a retrieval run, not to stored content.

**No provider-specific types.** Generation goes through one interface with a `base_url`;
there is no Anthropic type and no OpenAI type.

---

## 6. Open, and why they are open

**`Filter` shape on `VectorStore.search`.** Now that the store is settled — SQLite plus
LanceDB — this is a LanceDB predicate plus a metadata pre-filter, but the exact split
between them wants contact with real data volumes.

**Whether `Context` assembly is a `RetrievalStage`** or a distinct step. It behaves like
one, but it emits a different type.
