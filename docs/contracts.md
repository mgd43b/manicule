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
    # documents, chunks, lexical search, sync state

CollectionStore                                  # named sets, manual or rule-driven
TagStore                                         # arbitrary labels
VersionStore                                     # history across re-ingests
    resolve_citation(document_id, chunk_id) -> CitationResolution
TrashStore                                       # soft delete, and the two ways back
ChunkRelationStore                               # parent and sibling links
    # conversations belong to the ticket that builds chat

RetrievalStage
    name: str
    run(query: Query, candidates: list[Candidate]) -> list[Candidate]

Generator
    model_id: str
    context_window: int                          # served, not advertised
    generate(query: Query, context: Context) -> AsyncIterator[Token]

Connector
    watermark: Watermark|None                    # read-only; last *completed* enumeration
    discover(watermark: Watermark|None) -> AsyncIterator[DiscoveredDoc]
    fetch(ref: DocRef) -> RawDocument
    reconcile() -> AsyncIterator[SourceId]       # for deletion detection
```

**`Parser` returns blocks, not text.** Structure is discovered once, by the component that
can actually see it, and never re-derived downstream from prose.

**Storage is six protocols, not one, and one class implements all six.** `DocStore` was
deliberately partial in [#1](https://github.com/mgd43b/manicule/issues/1) — documents, chunks,
lexical search and sync watermarks, the ingest and retrieval critical path — on the stated
promise that organization would arrive as protocols of its own.
[#10](https://github.com/mgd43b/manicule/issues/10) is that arrival. Splitting them is what
lets a component declare the narrow surface it needs: a stage that resolves a collection into
document ids asks for a `CollectionStore` and cannot reach a document's chunks with the handle
it was given. Joining the *implementation* is what keeps the workspace boundary in one place —
`SqliteDocStore` has one constructor, one session factory and one tenancy check however many
contracts it satisfies. See [`storage.md`](storage.md) §11.

**`VersionStore.resolve_citation` takes the document as well as the chunk, and the reason is
the anchor rule.** `chunks.id` is derived from `(document_id, position, text)`, so a chunk that
survives a re-parse unchanged keeps its id and one whose text moved does not — the old id
*dangles* rather than silently re-pointing at whatever replaced it. A dangling id is opaque and
there is nothing left to look it up against, so the document comes too. What comes back is the
absence, labeled: `present`, `superseded`, `deleted`, or `unknown`. Nothing resolves a citation
into a superseded version to the text that replaced it. That would be §1's forbidden case
exactly — a location that is plausible and wrong — arriving through the one path built to
explain why a citation stopped working.

**`Embedder` has two tiers.** Tier A returns pre-pooled token states and manicule does the
pooling; tier B returns finished vectors. The distinction exists because a backend's
convenience output cannot be trusted to be the model's own pooling: `mlx-embeddings`
computes its XLM-RoBERTa `text_embeds` with mean pooling unconditionally, while the chosen
model pools with CLS, and it binds `last_hidden_state` to the *pooled* vector on some
architectures and to genuine token states on others. Both produce well-shaped, normalized
vectors and raise nothing. Tier B backends are therefore admitted only by measurement, since
they cannot be verified by inspection. See [`embeddings.md`](embeddings.md) §3.2 and §4.1.

**`Generator.generate` is iterated through `generating()`**, the sibling of `parsing()` in
`manicule.core.protocols`, for the reason [#35](https://github.com/mgd43b/manicule/pull/35)
established and one that is worse here: an abandoned generation stream holds an open HTTP
response to a model that is still working — billed tokens nobody will read on a hosted provider,
and on a local one the only model on the machine, occupied until it finishes. Its close carries a
hard deadline, because a shutdown that can block indefinitely on a misbehaving remote server is a
worse failure than a leaked socket.

**`Generator.context_window` is the window that will be *served*, not the model's trained
maximum.** Ollama applies a runtime `num_ctx` that defaults far below what modern models are
trained for, and a prompt over it is truncated from the front rather than refused — discarding
the system prompt and the citation protocol, and presenting as a model that ignores
instructions. The attribute exists so that the startup cross-check in
[`retrieval.md`](retrieval.md) §7.4 has something to read: an assembled context that cannot fit
is a refusal naming both numbers, not a runtime truncation.

**`Connector.discover` takes a watermark and `reconcile` exists separately.** Incremental
sync tells you what changed; it cannot tell you what was deleted, because a deleted page
simply stops appearing. Without a reconciliation pass the index serves removed documents
forever. Making it part of the protocol means no connector can quietly omit it.

**`Connector.watermark` produces the one `discover` consumes.** `storage.md` §4.7 ships a
`connectors.watermark` column and `discover` takes a watermark, so the system had somewhere to
persist one and no way, through this protocol, to obtain it. A read-only member closes that;
the concrete connector may compute it however it likes.

Its meaning is the part worth writing down: **safe to persist if and only if every document
`discover` yielded has been durably committed.** The connector's promise is narrower — that it
reflects a *complete* enumeration and never a partial one — and the rest is the caller's
obligation ([`ingest.md`](ingest.md) §13.2 already makes advancing it conditional on a clean
run). Storing a watermark for work that was not committed does not delay those documents, it
makes them **permanently invisible**: the next sync starts past them, nothing raises, and there
is nothing to notice. Same class as a citation pointing at a page that does not exist.

The race between "yielded" and "committed" is not eliminated, it is made survivable —
connectors overlap their queries slightly rather than resuming exactly, and content-hash dedup
absorbs the repeat (`connectors/confluence.md` §2). Widened in
[#9](https://github.com/mgd43b/manicule/issues/9), before anything had synced and therefore
before any stored watermark existed to invalidate.

**`RetrievalStage` is uniform — `candidates in, candidates out`.** A pipeline is a declared
list of stages, so the evaluation harness can compare whole pipelines by configuration
rather than by editing code. This is what makes "no retrieval feature without a measured
improvement" mechanically enforceable rather than a discipline to remember.

⚠️ **`RetrievalStage` is locked.** The evaluation harness exists
([`evaluation.md`](evaluation.md), [#15](https://github.com/mgd43b/manicule/issues/15)) and
writes versioned records to disk, so widening this now invalidates every recorded result
rather than merely threatening to.

## 4. Settled

**Metadata store: SQLite plus LanceDB**, and therefore SQLite FTS5 for BM25. Sixteen
relational tables belong in a relational store; a columnar vector store is the wrong tool
for joins and transactional updates. See `PLAN.md` §2.

**Vector dimensionality is a runtime parameter**, read from `Embedder.fingerprint` — never
a constant. The vector table is created at first ingest, and ingest must refuse to start
if the fingerprint does not match what the index was built with.

**There are three fingerprints, and they are compared at three scopes.** `EmbedFingerprint`
and `ChunkFingerprint` describe one process applied to a whole corpus, so both are compared
once per run and a mismatch refuses the run. `ParseFingerprint` describes one parser applied
to one document, so it is recorded in `documents.parse_fp` and compared per document — a
`pypdfium2` bump invalidates the PDFs and says nothing about the Markdown.
[`parsing.md`](parsing.md) §3.0 has the reasoning, [`storage.md`](storage.md) §6.4 the
storage. All three refuse rather than warn: there is nothing downstream that can detect mixed
output, and a corpus whose chunk boundaries were measured with a stand-in vocabulary rather
than the embedder's own is refused before any comparison at all.

## 5. Deliberately absent

**No `permissions` field on plugins.** OpenDocuments ships one that is typed, tested, and
enforced nowhere. Plugins run in-process with full privileges; the documentation says so.
An unenforced guarantee is worse than an absent one.

**No `score` on `Chunk`.** Scores belong to a retrieval run, not to stored content.

**No provider-specific types.** Generation goes through one interface with a `base_url`;
there is no Anthropic type and no OpenAI type.

---

## 6. Formerly open, now settled

Both remaining questions were retrieval questions, and
[`retrieval.md`](retrieval.md) settles them.

**`Filter` shape on `VectorStore.search` — settled in [`retrieval.md`](retrieval.md) §3.**
The field list is fixed, and `workspace_ids` is **required, non-empty and set-valued**: it is
a security boundary rather than a performance question, and a boundary you can forget to pass
is not a boundary. The split that was waiting on data volumes is settled as a *rule* rather
than a constant — fields with a promoted Lance column push down, fields needing a join resolve
in SQLite into a document-id set first, and which of the two plans runs is decided per query by
a derived over-fetch factor and a configurable id-list threshold. Both inputs to that decision
are recorded on every query, so the threshold gets set from measurement instead of argument.
`workspace_ids` alone pushes down to neither store: it is enforced by the hydrating join inside
the dense stage, which is also what stops soft-deleted and cross-workspace rows consuming
top-`k` slots. The shape that shipped in `manicule.core.retrieval` predates this and was
reconciled by [#36](https://github.com/mgd43b/manicule/issues/36); the pipeline that makes the
`workspace_ids` exemption safe is built in
[#6](https://github.com/mgd43b/manicule/issues/6), and
`assert_pipeline_enforces_scope` runs against it both in the suite and, opt-in, on every query.

**`Context` assembly is not a `RetrievalStage` — settled in [`retrieval.md`](retrieval.md)
§7.1.** It emits `Context` rather than `list[Candidate]`, and keeping the two types distinct is
precisely what lets the stage list be reordered freely while this step is not. A stage that
emitted a different type would make every stage's signature a union.

**`RetrievalStage` was reconsidered and deliberately not widened** — [`retrieval.md`](retrieval.md)
§2.3 re-argues the three widenings #1 rejected against a working design, and answers the one new
pressure that building it surfaced (per-stage diagnostics) without touching the signature. The
⚠️ in §3 has since taken effect: the harness in [`evaluation.md`](evaluation.md) reads the
stage list off every run's trace and stores it in each preference record, so the signature is
locked rather than pending.
