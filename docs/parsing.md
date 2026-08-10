# Parsing and chunking

Design for the twelve parsers, the anchor strategy behind every citation, and the chunker.
Ticket [#4](https://github.com/mgd43b/manicule/issues/4).

`PLAN.md` §5 picks the libraries and settles OCR. [`docs/contracts.md`](contracts.md) §1
fixes the `Anchor` type and §2 the content types. This document owns the mechanism: what
each parser emits, where each anchor comes from, how a chunk gets its size, and what
happens on every path that does not go well.

**Two things here are expensive to change after a corpus is indexed**, and both are guarded
by a fingerprint rather than a note:

- **Chunk size and overlap** (§1) — changing either means re-chunking and re-embedding
  every document.
- **Anchor construction** (§2) — changing it invalidates every stored citation.

---

## 1. Chunk size — 512 tokens, 64 overlap

**Decided: a 512-token budget on `embed_text`, 64 tokens of overlap, one budget for every
block kind.**

This is a runtime guardrail, not a tuning knob. The reasoning below is the whole of the
argument; if it stops holding, §1.8 says so explicitly.

### 1.1 The binding constraint is the embedder's context window

An embedding model has a maximum sequence length. Past it, every library in the stack
**truncates silently** — `transformers` and `sentence-transformers` both do, and neither
raises. A 900-token chunk handed to a 512-token model produces a vector describing the
first 512 tokens, while the stored chunk still claims all 900. Retrieval then misses text
the citation says is there.

That is the same defect class as a citation pointing at a page that does not exist: the
stored artifact claims more than the system actually indexed. It is worse in one respect —
there is no visible symptom at all.

The candidate embedders do not agree on the limit:

| Model family | Max sequence length |
|---|---:|
| BERT-family retrieval models (`bge-*`, `e5-*`, `gte-*`) | 512 |
| `all-MiniLM-L6-v2` | 512 positional, **256 as shipped** |
| EmbeddingGemma (`embeddinggemma-300m`) | 2048 |
| `nomic-embed-text-v1.5` | 8192 |
| Qwen3-Embedding | 32768 |

512 is the floor of everything except one, and that exception is instructive: MiniLM's
`sentence_bert_config.json` sets `max_seq_length: 256` even though the architecture allows
512. **The model's positional limit and the library's configured limit are different
numbers, and the smaller one wins without saying so.** A design that reasons about
architecture limits gets this wrong.

So the mechanism is not "512 is safe everywhere". It is:

> The chunker reads the **effective** sequence length from `Embedder.fingerprint` and
> **refuses to start** when `budget_tokens` exceeds it.

512 is the default because it clears the entire realistic candidate set bar one, and the
one it does not clear produces a loud startup error naming both numbers, not a silently
degraded index.

**Why not 256**, which would clear MiniLM too. 256 tokens is roughly a long paragraph. An
answer that spans a paragraph and its follow-up no longer fits in one chunk, so retrieval
returns half of it and the generator has to be lucky enough to pull the neighbour as well.
It also doubles the chunk count, which doubles embedding time, index size, and the number
of near-duplicate results competing for slots in the fused ranking. The published work on
retrieval chunking converges on the 256–512 band for question-answering corpora; 512 is the
top of that band, and the top is where the model limit also happens to sit.

**Why not 2048**, on a model that permits it. Two reasons, and the second is the one that
matters here. A single vector summarising 2048 tokens is a weaker signal — the embedding
averages over more topics, so a chunk covering four subjects retrieves poorly for all four.
And a budget set by whichever model is installed today makes the **corpus non-portable**: a
2048-token corpus cannot be re-embedded with a 512-token model without re-chunking, which
changes every `text` a user has ever been shown. At 512, swapping the embedder is a
re-embed — expensive, but the chunk boundaries and therefore the citations are unchanged.
That property is worth more than the marginal recall a larger window might buy, and it is
the argument that survives the model landscape moving.

**This requires something of [#3](https://github.com/mgd43b/manicule/issues/3):**
`EmbedFingerprint` must carry `max_sequence_length` — the effective one, already net of
any instruction prefix the backend prepends (EmbeddingGemma's `search_document: ` and the
E5 family's `passage: ` both consume real tokens) — and must expose the model's own
tokenizer for counting. Both are stated in §1.2 and §1.7.

### 1.2 Count with the embedder's tokenizer, never an estimator

A budget enforced against a hard model limit must be measured with the tokenizer that
enforces that limit. Estimating is not good enough, for three compounding reasons:

- **Vocabularies disagree.** `tiktoken`'s `cl100k_base`/`o200k_base` are ~100k–200k BPE
  vocabularies; the retrieval models above use 30k–256k WordPiece or SentencePiece.
  WordPiece splits the same English prose into *more* tokens, and the gap widens sharply
  for code, identifiers, and any non-Latin script.
- **Estimation error is one-directional in the dangerous direction.** Undercounting
  produces a chunk that overflows the model and is silently truncated. Overcounting only
  wastes budget.
- **Sampling makes it worse.** The tempting optimisation — encode the first few thousand
  characters and extrapolate — turns a bounded error into an unbounded one on exactly the
  documents that matter, the long ones.

So:

> `Chunker` takes a `count_tokens: Callable[[str], int]` from the bound `Embedder`. It is
> the model's own tokenizer, called on the exact string the model will see.

`tiktoken` stays in the stack for context-window fitting at generation time
(`PLAN.md` §16), which is a different job against a different model. It is used in the
chunker only when no embedder is bound — parsing without embedding, as in a dry-run parse
or a fixture test — and then a **1.5× safety factor** applies and the resulting chunks are
marked `provisional` and refused by ingest. Provisional chunks never reach the index.

### 1.3 The budget is on `embed_text`, not `text`

`embed_text` is `text` with the heading breadcrumb prefixed (contracts.md §2, and §5
below). The embedder sees `embed_text`. Therefore the model limit applies to `embed_text`,
and a budget measured on `text` overflows by exactly the breadcrumb length.

```
budget_tokens (512)  =  breadcrumb_budget_tokens (64)  +  text budget (448)
```

The breadcrumb budget is reserved unconditionally, whether or not a given chunk has a
breadcrumb. Reserving it conditionally would make chunk boundaries depend on heading depth,
so the same paragraph would chunk differently under a deep heading than a shallow one —
and a document reorganisation would silently re-chunk sections it did not touch.

### 1.4 One budget for every block kind

The obvious refinement is per-kind budgets: tables get more room because they are dense,
code gets less because it is repetitive. **Rejected.**

Any per-kind budget above the model floor is a silent-truncation generator, and it fires
precisely on the content that motivated the exception. A table budget of 800 tokens against
a 512-token model means every large table is indexed by its first two-thirds, with no error
and no symptom — the exact failure §1.1 exists to prevent. A per-kind budget *below* the
floor is just a smaller budget with extra configuration surface and no measurement behind
it.

One number, checked once, against the model. Kind-specific behaviour lives in *how* an
oversized block is split (§4.2), which is where it belongs — that is a structural question,
not a size question.

### 1.5 Overlap — 64 tokens, prose and lists only

Overlap exists for one reason: the sentence that answers the question can straddle a
boundary, and a chunk holding half of it retrieves poorly and reads worse when cited.

**64 tokens (12.5%)** is the default because overlap is pure duplication — it costs index
size, embedding time, and result-list quality, since two overlapping chunks covering the
same sentence both match the same query and consume two slots in the fused ranking. With
structure-aware splitting, boundaries already land at paragraph and heading edges where
straddling is rare, so the marginal value of a wider window is low and the cost is linear.

Two rules make it behave:

- **Overlap applies to `prose` and `list` only.** Never `code`, `table`, `panel`, `heading`
  or `media`. Overlapping a table means half a table appears twice with a repeated header
  and no way to tell the copies apart; overlapping code emits a fragment whose
  `LineAnchor` duplicates another chunk's lines, which breaks the tightness assertion in
  §3.3.
- **Overlap is taken in whole units, never mid-sentence.** Fill backwards with complete
  sentences (falling back to complete paragraphs when a single sentence exceeds the
  window) until the next one would exceed 64 tokens. A window that cuts mid-sentence
  produces a chunk starting on a fragment, which is what the overlap was meant to avoid.

### 1.6 Minimum chunk size — 64 tokens, merged backwards

A trailing 8-token chunk is retrieval noise: short texts produce vectors dominated by their
few tokens and they win queries they should lose.

A chunk below `min_tokens` (64) is merged into the preceding chunk **when that chunk has
the same `kind` and the merge stays within budget**. Otherwise it stands alone. It is never
dropped — dropping is data loss, and silent data loss is the thing this document is mostly
about.

### 1.7 `ChunkFingerprint` — the guardrail, in code

`PLAN.md`'s build-order note is right that chunk size does not touch the schema. It does not
follow that it needs no persisted state. Changing it means re-chunk *and* re-embed, so it
gets the same mechanical treatment `Embedder.fingerprint` gets for dimensionality:

```
ChunkFingerprint  budget_tokens, overlap_tokens, breadcrumb_budget_tokens,
                  min_tokens, chunker_version, tokenizer_id, grammar_pack_version
```

Recorded at first ingest alongside the embedder fingerprint. **Ingest refuses to start when
the stored fingerprint differs from the running one**, and names the differing field and
the re-index command. This is the same guard, in the same place, for the same reason.

Two of the fields are less obvious than they look:

- **`tokenizer_id`** — the same budget measured with a different tokenizer produces
  different boundaries. A model swap that keeps the dimension but changes the vocabulary
  would otherwise pass the embedder check and quietly re-chunk.
- **`grammar_pack_version`** — a tree-sitter grammar upgrade changes parse trees, which
  changes code chunk boundaries (§8.3). It invalidates *only* code documents, so this field
  supports a partial re-parse rather than a full one.

### 1.8 What would have to be true to change it

Both of these, not either:

1. **The floor rises.** manicule drops BERT-family embedders from the supported set, or
   `Embedder.fingerprint` starts carrying a limit that makes 512 provably slack for every
   supported model. The check in §1.1 makes this observable rather than assumed.
2. **A measured improvement on the [#15](https://github.com/mgd43b/manicule/issues/15)
   baseline.** Chunk size is a retrieval parameter, so it falls under the same rule as every
   other retrieval feature: no change without a measured gain on a fixed query set. #15
   exists partly for this.

Absent both, 512/64 stands. And the cost of changing it is a full re-embed of the corpus,
which the fingerprint makes an explicit, priced operation rather than an accident.

---

## 2. Anchors

### 2.1 Rules every parser obeys

1. **An anchor is constructed from a location the source or the library reports, never from
   a heuristic over extracted text.** A `PageAnchor` is built from a page index the PDF
   library returns. Text-position heuristics — blank-line runs, form feeds, page-number
   regexes — never synthesise one. A document whose pagination cannot be recovered yields
   `Unlocated` with a reason.
2. **`Unlocated` carries a reason a human can act on.** `"no text layer"`,
   `"page rotation unsupported"`, `"source positions unavailable for TOML"` — not
   `"unknown"`. The reason surfaces in `doctor` (§6.6).
3. **`rects` is a list of the boxes the quoted text actually occupies.** Never a merged
   envelope: a quote spanning a column break has two boxes, and their union covers text
   that was not quoted. A parser that cannot produce boxes emits `rects=[]` — which is a
   page-level anchor, honest about being page-level — rather than one box covering the
   page.
4. **Every anchor round-trips (§3).** A parser that cannot resolve an anchor it emits may
   not emit it.
5. **`Unlocated` is bounded, not free.** Every parser declares a maximum `Unlocated` ratio
   over its fixture corpus and the test suite enforces it (§3.4). Otherwise rule 1 is
   satisfiable by returning `Unlocated` for everything.

### 2.2 `Rect` — one coordinate convention, normalised at parse time

`Rect` is `(x0, y0, x1, y1)` in **points (1/72 inch), origin top-left, y increasing
downward, relative to the page as displayed.**

Normalising at parse time means a consumer needs the page number and nothing else. Two
transforms happen inside the PDF parser and never leak out:

- **Origin flip.** PDF user space is origin-bottom-left. Viewers, canvases and CSS are
  top-left. Storing user space would make every consumer re-derive the flip from the page
  height, and one of them would forget.
- **Rotation.** A page with `/Rotate 90` displays rotated, but text coordinates are reported
  in the unrotated space. A rect stored unrotated draws in the wrong place — usually off the
  page entirely, which at least fails visibly, but on a square page fails invisibly. The
  parser applies the rotation and stores display coordinates. A page whose rotation is not
  a multiple of 90 (malformed, but it occurs) yields `rects=[]` with the page number still
  correct.

PPTX uses the same convention. `python-pptx` reports shape geometry in EMU, origin
top-left; the parser converts (`1 pt = 12700 EMU`) and stores points. Same `Rect`, same
consumer code, different source unit.

### 2.3 `HeadingAnchor.fragment` is the resolution key

`path` is a list of heading strings. It is for display and for the breadcrumb. It is **not**
sufficient to resolve against, because heading paths repeat — two sections called
"Configuration" under two different parents collide the moment the parents match, and
"Overview" collides constantly.

`fragment` carries the unique, resolvable address:

- **When the source defines fragments, use the source's.** Confluence derives heading
  anchors from heading text and appends `-1`, `-2`… for duplicates
  ([`confluence.md`](connectors/confluence.md) §8); HTML authors write `id=`. Using ours
  instead would produce citations that do not deep-link.
- **When the source defines none, synthesise one** with GitHub-style slugification
  (lowercase, non-alphanumerics to hyphens, collapse runs, trim) plus the same `-N`
  de-duplication suffix, counted in document order. Markdown, DOCX and Jupyter go this way.
  The slug is stored in the parser's block index so resolution is an exact lookup rather
  than a re-derivation.
- **When the source defines none and we cannot usefully invent one, `fragment` is `None`.**
  This is the HTML-without-`id` case: a slug we invent will not work as a URL fragment on a
  page we do not control, so inventing one produces a citation that silently fails to
  deep-link. `path` is still populated; the citation resolves to the document.

Resolution order is `fragment` first, `path` second, and a `HeadingAnchor` whose `path` is
ambiguous and whose `fragment` is `None` **does not round-trip** and therefore may not be
emitted — the parser emits `Unlocated(reason="ambiguous heading path")` instead.

### 2.4 Per-format anchor strategy

Twelve parsers, eighteen extensions, plus the v1 source. Every row names the `Anchor`
variant, where the location physically comes from, and whether provenance is real.

| Parser | Extensions | Anchor | Location source | Provenance |
|---|---|---|---|---|
| **PDF** | `.pdf` | `PageAnchor(page, rects)` | pdfium page index; char boxes for the quoted range, merged into per-line rects | **Exact.** Page and box both real |
| **Code** | 40+ | `LineAnchor(start, end, symbol)` | tree-sitter node byte range → line numbers; `symbol` from the AST (§8.2) | **Exact** |
| **Confluence ADF** | — (v1 source) | `HeadingAnchor(path, fragment)` | ADF `heading` nodes; fragment is Confluence's own anchor | **Exact**, deep-links to the section |
| **Markdown** | `.md` `.mdx` | `HeadingAnchor(path, fragment)` | `markdown-it-py` token `map` (source line span) → heading tree; slug synthesised | **Exact** |
| **HTML** | `.html` `.htm` | `HeadingAnchor(path, fragment\|None)` | `selectolax` heading elements; fragment from the nearest preceding `id=` | **Partial** — fragment only where the author supplied an `id` |
| **DOCX** | `.docx` | `HeadingAnchor(path, fragment)` | paragraph style (`Heading N`); slug synthesised | **Sections only.** No page numbers, ever — see §2.5 |
| **XLSX / CSV** | `.xlsx` `.csv` | `CellAnchor(sheet, ref)` | sheet name + the row/column range the block covers, as `Sheet1!B4:D12`. A CSV has no sheet, so `sheet` is the file stem | **Exact** |
| **PPTX** | `.pptx` | `PageAnchor(page, rects)` | slide index (1-based, presentation order); shape geometry → `Rect` | **Exact** |
| **Jupyter** | `.ipynb` | `HeadingAnchor(path, "cell-<id>")` | markdown-cell heading tree; fragment is the nbformat cell `id` | **Exact** for nbformat ≥ 4.5; see §2.5 below |
| **Email** | `.eml` `.msg` | `LineAnchor(start, end, None)` | line span within the canonical body part (§10) | **Exact**, given the part-selection rule |
| **Plain text** | `.txt` | `LineAnchor(start, end, None)` | source line numbers | **Exact** |
| **Structured** | `.json` `.yaml` `.yml` `.toml` | `LineAnchor(start, end, symbol)` | source line span; `symbol` is the JSON Pointer / dotted key | **Exact** where positions exist (§11) |
| **Archive** | `.zip` | *(none — emits no chunks)* | members become their own documents with their own anchors (§9) | N/A |

Twelve parsers over eighteen extensions, matching `PLAN.md` §5 and the `CAPABILITIES.md`
file-type list: XLSX and CSV share one parser because the anchor and the block model are
identical once a CSV is given a sheet name. The code parser is the exception to the
eighteen — its extension set is the grammar pack's language list, which is deliberately
wider than the capability floor, since real ASTs across many languages is one of the two
upgrades this ticket exists for. Confluence ADF is a thirteenth row and not a
file type — it registers under `application/json;profile=atlas-doc-format` and its node
handling is specified in [`confluence.md`](connectors/confluence.md) §5, listed here because
it is the v1 source and its anchors have to obey the same rules as everything else.

`.mdx` goes through the Markdown parser. JSX component tags are not Markdown; they are
emitted as `media` blocks carrying their tag name, never as prose, because a component
invocation embedded as text is noise in the vector and nonsense in a citation. Any Markdown
inside a component's children is parsed normally.

`heading_path` on `ParsedBlock` is populated for every parser that can recover one,
including the ones whose anchor is not a `HeadingAnchor` — a PPTX slide title and an XLSX
sheet name are both heading path elements, and they feed the breadcrumb in §5. A parser
that cannot recover a heading path emits an empty one rather than a guess, because the
breadcrumb goes into the embedding and a wrong breadcrumb is worse than none.

### 2.5 Where honest provenance is impossible

Named plainly, because each is a user-visible limitation rather than an implementation gap.

**DOCX page numbers do not exist.** A `.docx` stores a flow of paragraphs; pages are
produced by a layout engine at render time and depend on the fonts, the printer metrics and
the Word version. `python-docx` cannot know them, and neither can anything else short of
running a layout engine. Explicit page breaks are recorded in the file, but they are a lower
bound on page count, not a pagination. **A DOCX citation is "§ Deployment > Rollback", never
"p. 7".** Any implementation that reports a DOCX page number has invented it.

**PDF heading structure is not recoverable in general.** A PDF has no heading semantics —
only glyphs with font sizes. Inferring a hierarchy from font size is a heuristic, and
`heading_path` feeds the breadcrumb, which goes into the embedding; a wrong heading actively
degrades retrieval rather than merely failing to help. So:

> PDF `heading_path` comes from the document outline (`/Outlines` bookmarks) when the PDF
> has one, and is **empty** otherwise. Never from font-size clustering.

The consequence is real: an outline-less PDF — most PDFs — produces chunks whose breadcrumb
is the document title alone. The page and box provenance is still exact. Improving this is
what the optional `docling` / `marker` parsers are for; they are layout models and they
belong behind the fallback chain, off by default, not woven into the fast path.

**HTML deep links exist only where the author made them.** No `id=`, no fragment. We could
synthesise a slug, but it would not resolve on a page we do not serve, producing a citation
that looks precise and lands at the top of the page. `fragment=None` and a document-level
link is the honest output.

**Scanned PDFs yield nothing, by decision.** OCR is out of scope for v1 (`PLAN.md` §5). The
mechanism — how "nothing" is distinguished from "crashed" — is §6.5.

**Jupyter notebooks below nbformat 4.5 have no cell ids.** Cell ids arrived in 4.5. Below
that, the fragment is `None` and the heading path is the only address; where the heading
path is ambiguous the block is `Unlocated(reason="notebook predates cell ids")`. The
notebook is still indexed and still cited at document level. Upgrading the notebook file
fixes it, which is worth saying in the diagnostic.

**`.msg` has no permissively-licensed parser.** §10.

---

## 3. The round-trip contract

contracts.md §1 calls this "a test obligation on every parser, not a convention". This
section is the obligation, written so a parser either satisfies it or does not compile past
review.

### 3.1 Resolution has to be part of the protocol

The obligation is unenforceable as `Parser` currently stands, because nothing in the
protocol can resolve an anchor. It needs a third member:

```
Parser
    media_types: set[str]
    parse(raw: RawDocument) -> AsyncIterator[ParsedBlock]
    resolve(anchor: Anchor, raw: RawDocument) -> str | None
```

`resolve` returns the source text the anchor addresses, or `None` for `Unlocated`. It takes
`RawDocument` because `Document.original_ref` retains the source bytes (`PLAN.md` §4), so
resolution never re-fetches. It is used by the test harness below and by the citation path
that shows a user what was quoted.

A separate `Resolver` protocol registered under the same entry point would work equally
well. What does not work is leaving it out; that is what turns the round-trip rule into
aspiration. [#1](https://github.com/mgd43b/manicule/issues/1) owns the protocol code and
this ask has been raised there.

### 3.2 Normalisation, stated once

Exact string equality is the wrong assertion. PDF extraction reintroduces ligatures and
hyphenation, HTML collapses whitespace differently from its source, and DOCX splits a
sentence across runs. So both sides of every comparison pass through one normaliser, and it
is defined here rather than per-parser:

1. Unicode NFC.
2. Replace all whitespace runs (including ` `, `​`, `\f`) with a single space.
3. Remove soft hyphens (`­`); join words split by `-` immediately before a line break.
4. Map the ligatures `ﬁ ﬂ ﬀ ﬃ ﬄ` to their letter sequences.
5. Strip leading and trailing whitespace.

The normaliser is versioned and its version is part of `chunker_version` (§1.7). It is used
by the tests and by nothing else — stored `text` is never normalised, because `text` is what
gets shown.

### 3.3 The six assertions

`assert_round_trip(parser, fixture)` runs against every chunk a fixture produces.

1. **Containment.** `normalize(chunk.text)` is a substring of
   `normalize(parser.resolve(chunk.anchor, raw))`. The chunk says where it came from, and
   it is there.

2. **Tightness.** Chunks are first **grouped by identical anchor** — several chunks can
   legitimately share one, as when a long section under one `HeadingAnchor` splits into
   four. For each group:

   ```
   len(normalize(resolved))  <=  k * sum(len(normalize(c.text)) for c in group)
   ```

   | Anchor | `k` | Why |
   |---|---:|---|
   | `LineAnchor` | 1.0 | the line span is the text |
   | `CellAnchor` | 1.0 | the cell range is the text |
   | `PageAnchor` **with rects** | 1.05 | the boxes bound the quote; slack for glyphs clipped by a box edge |
   | `PageAnchor` **with `rects=[]`** | — | **exempt — capped by assertion 5 instead** |
   | `HeadingAnchor` | 1.2 | the resolved section includes its own heading line and inter-block whitespace |

   Grouping is what makes this assertion correct rather than merely strict. Comparing a
   whole section against one of its four chunks would fail a parser that is behaving
   perfectly, and the usual repair — loosening `k` until the suite passes — dissolves the
   assertion entirely.

   Tightness is what does the real work. Containment alone is satisfied by an anchor
   pointing at the whole document, and by an anchor pointing at the wrong page of a document
   that repeats a sentence. A page-level `PageAnchor` cannot pass tightness and is not asked
   to; it is *budgeted* instead, so a parser cannot quietly make every anchor page-level to
   get through.

3. **Discrimination.** For each ordered pair of **distinct** anchors `(a, b)` in a fixture,
   `normalize(text_of(b))` is **not** contained in `normalize(resolve(a))`.

   Three exclusions, each for a reason rather than for convenience:
   - **Page-level `PageAnchor`s** (`rects=[]`) are excluded as `a`, since resolving one
     returns a whole page and every other chunk on that page is legitimately inside it.
   - **Adjacent prose chunks sharing an overlap window** (§1.5) are compared on their
     non-overlapping remainder only. The shared sentences are duplicated by design.
   - **Anchors whose locations genuinely nest** — a `HeadingAnchor` for a section and one
     for its subsection — are excluded, and the parser declares the nesting.

   Everything else must discriminate. This is the assertion that fires when every anchor
   points at page 1, or when a page index is off by one, or when a heading fragment is
   assigned to the wrong section — all of which pass containment, and all of which can pass
   tightness on a short document.

4. **Determinism.** Parsing the same bytes twice yields byte-identical chunk lists: same
   ids, same text, same anchors, same order. Set and dict iteration order, `os.walk` order
   inside archives, and floating-point rect arithmetic are all places this breaks. It
   matters because chunk ids are derived from content and position, and a non-deterministic
   parser produces a corpus that churns on every re-ingest.

5. **Location budget.** Over the parser's whole fixture corpus:
   - `Unlocated` chunks ≤ the parser's declared `max_unlocated_ratio`;
   - `PageAnchor` with `rects=[]` ≤ the parser's declared `max_pagelevel_ratio`.

   Both ratios are declared by the parser and asserted by the harness. Without them,
   "a citation carries a correct location, or none" is trivially satisfiable by never
   carrying one.

6. **Idempotence across a re-ingest.** Re-parsing an unchanged document produces chunk ids
   identical to the stored ones, so an unchanged document costs zero re-embedding. Chunk
   ids are therefore derived deterministically from `(document_id, position, hash(text))`
   and never from a counter, a UUID, or a timestamp.

Assertions 1–4 and 6 are per-fixture; assertion 5 is per-corpus and runs once at the end of
each parser's suite.

### 3.4 Declared budgets, per parser

Starting values. Each is a ceiling the suite enforces, and lowering one is a normal
improvement; raising one requires a note in the PR saying which fixture forced it.

| Parser | `max_unlocated_ratio` | `max_pagelevel_ratio` |
|---|---:|---:|
| PDF | 0.00 | 0.10 |
| Code, Plain text, XLSX, CSV, Markdown, Confluence ADF | 0.00 | — |
| PPTX | 0.00 | 0.20 |
| DOCX, HTML, Jupyter | 0.05 | — |
| Email | 0.05 | — |
| Structured | 0.10 | — |

PDF is `0.00` unlocated and `0.10` page-level because pdfium reports a page index for every
page — a PDF chunk is never *unlocated*, at worst it is *page-level* when box extraction
fails on a given run of glyphs. Structured data gets the widest unlocated budget because
TOML is the one format in the set with no line-position API (§11).

### 3.5 The fixture corpus

**Generated wherever generation is possible, committed only where it is not.**

Generating fixtures from a script (`tests/fixtures/build.py`) keeps the repository small,
makes every fixture's structure inspectable as code rather than as a binary, and lets the
hostile cases — a zip bomb, a 400-page PDF — exist without being committed. `reportlab`
builds the PDFs, `python-docx` / `python-pptx` / `openpyxl` build the Office files,
`zipfile` builds the archives.

Committed fixtures are limited to what cannot be generated faithfully: a real-world `.msg`,
a PDF exported by a real word processor (generated PDFs have suspiciously clean text
layers), a scanned page. Every committed fixture is public-domain, CC0, or authored for the
repository, recorded in `tests/fixtures/PROVENANCE.md`. **No customer documents and no
scraped files of uncertain licence** — a fixture corpus is published with the project.

Per parser, four kinds, all four required:

| Kind | What it is |
|---|---|
| **Typical** | a well-formed document of the sort the parser will mostly see |
| **Structurally hard** | multi-column PDF; a table spanning a page break; merged cells; nested lists five deep; a heading path that repeats; a code file with nested classes |
| **Degenerate** | zero bytes; headers with no body; a single heading and nothing else; one cell; a `.txt` with no trailing newline |
| **Hostile** | malformed UTF-8; a PDF with no text layer; a zip bomb; a `.docx` whose internal XML is truncated; a CSV with 200 columns and an unterminated quote |

Fixture size cap: **256 KiB**, with one deliberate exception per parser for a generated
large file that exercises the streaming path.

**Expected-anchor files.** Every fixture has a sibling `<name>.expected.json` listing, per
chunk: `position`, `kind`, the anchor, `token_count`, and the first 48 characters of `text`.
Regenerated by `tests/fixtures/regen.py` and reviewed in the diff.

This is what makes the chunk-size guardrail visible rather than trusted: a change that
alters chunk boundaries — a tokenizer swap, a grammar upgrade, an accidental budget edit —
shows up as a large diff in these files during review, instead of as a quietly re-chunked
corpus after merge.

---

## 4. Structure-aware chunking

### 4.1 Boundaries come from blocks, never from prose

The chunker consumes `Iterable[ParsedBlock]` and treats `kind` and `heading_path` as facts.
It does not look at the text to decide whether something is a heading, a table, or code.
Structure is discovered once, by the parser that can see it (contracts.md §3), and
re-deriving it downstream both duplicates the work and does it worse — the parser had the
`<h2>` element or the ADF `heading` node; the chunker would have a line that starts with a
capital letter.

The algorithm:

1. **Start a new chunk at every `heading` block.** Headings are boundaries. A section is
   the natural retrieval unit and it is the unit the breadcrumb describes.
2. **Accumulate consecutive blocks** while `embed_text` stays within budget. Blocks of
   different `kind` may share a chunk — a paragraph introducing a table belongs with it —
   but an atomic block (`table`, `code`) is never *partially* included.
3. **Close the chunk** when the next block would exceed budget, or at the next `heading`,
   or at the end of the document.
4. **Carry overlap** into the next chunk per §1.5, when both chunks are `prose` or `list`.
5. **Merge a sub-minimum trailing chunk** backwards per §1.6.
6. **A chunk's `kind`** is the kind of the majority of its tokens, ties going to the first
   block. `position` is its ordinal within the document.

A `heading` block never becomes a chunk on its own — it is a boundary and a breadcrumb
component, not content. The exception is a document that is *only* headings (a stub page, a
table of contents), which would otherwise produce zero chunks and look like
`no_extractable_text`. There, headings are emitted as `prose`, and the diagnostic says why.

`media` blocks are never chunked. Alt text and captions are content and are appended to the
adjacent prose chunk; the binary is not. With OCR out of scope, an image with neither alt
text nor caption contributes nothing to the index, and that is the honest outcome rather
than a gap to paper over.

### 4.2 When an atomic block exceeds the budget

Tables and code blocks stay whole. Sometimes they cannot. This is the interesting case, and
"truncate" is never the answer.

**Tables — split by rows, repeat the header.**

Split at row boundaries into as few parts as possible, and **prepend the header row(s) to
every part.** A table part without its header is a grid of numbers; with it, each part is
independently meaningful and independently retrievable. Header rows are known from the
parser (`ParsedBlock.metadata.header_rows`), not guessed from the first row being bold.

Every part carries:
- the same `heading_path` and the same breadcrumb — the parts are the same table;
- `metadata.table_part = (i, n)` and `metadata.rows = (first, last)` in source row numbers;
- for XLSX and CSV, its **own** `CellAnchor` narrowed to that part's rows. Because the
  header is repeated into the part's `text`, the ref must cover the header too, so
  `CellAnchor.ref` is **comma-separated A1 notation** — `Sheet1!A1:D1,A25:D48` — which is
  Excel's own syntax for a multi-area range and is what `resolve` parses. Splitting a
  spreadsheet table *improves* provenance: each part addresses exactly its own rows, and
  tightness (§3.3) holds where a whole-table anchor would fail it.
- for a table inside a page or ADF document, the block's `HeadingAnchor`, shared across
  parts and disambiguated by `position`.

If a **single row** exceeds the budget, split it by cells, each cell carrying its column
header. If a **single cell** exceeds the budget, it is prose and splits as prose, with
`metadata.cell` naming its coordinates. There is no depth at which this gives up.

**Code — split at the highest AST boundary that fits.**

Descend the tree-sitter parse tree: try top-level definitions first, then nested ones, then
statement boundaries, then blank-line runs, then lines. Never split mid-token, mid-string
or mid-comment.

Splitting code costs nothing in provenance and gains something: each part gets its own real
`LineAnchor`, with `symbol` set to the definition that part covers. A 900-line file becomes
chunks that each cite the function they contain. The enclosing symbol chain reaches the
embedder through the ordinary breadcrumb mechanism (§5) — `src/auth/token.py > TokenStore >
refresh` — so no comment is injected into `text` and the cited code stays byte-honest.

When the grammar for a language is unavailable, the fallback is blank-line runs then lines,
`symbol=None`, and the line numbers are still exact. §8.1 explains why that path should be
rare.

**Panels — split as prose, keep the semantic.**

A `panel` that overflows splits like prose, and **every part keeps `kind="panel"` and its
severity in metadata.** A warning panel is not ordinary prose
([`confluence.md`](connectors/confluence.md) §5); a warning panel split in half is still two
pieces of warning.

**Lists — split at top-level items, never inside one.**

Preserve the nesting prefix: a part beginning at a third-level item repeats its ancestor
item text as context, in `embed_text` only.

**Prose — paragraph, then sentence, then token.**

Sentence segmentation uses a tokenizer-free rule (terminator plus whitespace plus an
uppercase or digit start, with an abbreviation exception list), because pulling in a
sentence-segmentation model to draw chunk boundaries would put a second model's version
into `chunker_version`. Only a single sentence longer than the budget — a minified line, a
base64 blob pasted into a page — falls through to a hard token split, and that fact is
recorded in `metadata.hard_split = true` so `doctor` can count it.

---

## 5. `text` versus `embed_text`

`text` is cited and shown. `embed_text` is what the embedder sees. Storing both means the
citation is not polluted by retrieval scaffolding (contracts.md §2).

```
embed_text = breadcrumb + "\n\n" + text     (when a breadcrumb exists)
embed_text = text                            (otherwise)
```

### 5.1 Breadcrumb construction

Coarsest to finest, joined with `" > "`:

```
<source or space> > <ancestor titles…> > <document title> > <heading path…>
```

```
ENG > Platform > Auth Service > Token Refresh > Configuration
src/auth/token.py > TokenStore > refresh
Q3 Forecast.xlsx > Regional > EMEA
```

Four rules, each fixing something that otherwise happens constantly:

- **Deduplicate adjacent repeats.** A page titled "Auth Service" under a parent titled
  "Auth Service" yields one, not two. A Markdown file whose single H1 equals its title drops
  the H1. Without this, roughly a third of real breadcrumbs stutter.
- **Never include the part index.** A table split into three parts gets one breadcrumb, not
  three. Position is not semantic, and putting it in `embed_text` makes near-identical
  chunks embed differently for no reason.
- **An empty breadcrumb is legitimate.** A `.txt` at the root of a filesystem source has no
  hierarchy. `embed_text == text`, and nothing is invented. A fabricated breadcrumb is a
  fabricated signal in the vector.
- **The breadcrumb never appears in `text`.** `text` is what a user is shown as the quote,
  and a quote prefixed with navigation is not a quote.

### 5.2 Budget and middle elision

**64 tokens**, counted with the embedder's tokenizer (§1.2), reserved unconditionally
(§1.3).

Over budget, **elide from the middle**, not the tail:

```
ENG > Platform > Auth Service > Token Refresh > Rotation > Configuration
ENG > Platform > … > Rotation > Configuration
```

The two ends carry the most information — the outermost element says which corpus and which
product area, the innermost says what this specific section is. A tail truncation throws
away the innermost, which is the one that disambiguates "Configuration". Drop whole
elements from the middle outward until it fits; if the first and last elements together
still exceed 64 tokens, truncate the last element on a word boundary and append `…`.

### 5.3 Stored, not derived

`embed_text` is recomputable in principle and stored anyway, for two reasons: recomputing it
requires the whole document's heading tree, which is not available when reading a single
chunk back; and a change to the breadcrumb rules must show up as an explicit re-embed with a
`chunker_version` bump, not as silent drift between chunks written before and after.

---

## 6. Parser selection and the fallback chain

### 6.1 Resolving a media type

In order, first hit wins:

1. **Explicit per-source override** in connector config.
2. **The media type the connector reports.** Authoritative when present — Confluence, S3
   and HTTP all state one, and they are right more often than a filename is.
3. **Filename extension.**
4. **Content sniffing** (magic bytes).

Sniffing is a **tiebreaker, never an override** of a declared media type, with one
exception: `application/octet-stream` is not a claim, it is the absence of one, and sniffing
replaces it. Letting sniffing override a real declaration is how a `.docx` gets parsed as a
zip (§9.4).

### 6.2 `parserFallbacks`

One of the 40 settings. **Keyed by media type**, value an ordered list of parser names:

```toml
[parserFallbacks]
"application/pdf" = ["pdf", "docling"]
"text/html"       = ["html", "trafilatura"]
"*"               = ["plaintext"]
```

- The **first entry is the primary**; there is no separate "primary parser" concept, which
  removes a whole class of question about how the two interact.
- **`"*"` supplies a global tail**, appended to every chain. Shipping `["plaintext"]` there
  means an unknown text-ish file is indexed with real `LineAnchor`s rather than skipped.
- User configuration **replaces** the chain for a media type rather than merging into it.
  Merging produces chains nobody can predict from reading the config.
- **A named parser that is not installed is a startup error**, not a silent skip. A chain
  whose behaviour depends on what happens to be installed reproduces exactly the hazard the
  OCR decision exists to avoid: the same document ingesting differently on different
  machines.
- The **resolved chain is deterministic and recorded** on each document
  (`metadata.parsers_attempted`), so a result is explicable months later.

### 6.3 What counts as failure

Chains rot when "failure" is fuzzy. Three categories, and only two of them advance the
chain:

**Hard failure — advance.** The parser raises; declares the input unsupported; produces
structurally invalid blocks; exceeds its per-parser time limit; or exceeds its per-parser
memory limit. The reason is recorded per attempt.

**Empty output — advance, and remember that it happened.** The parser returned zero
text-bearing blocks without raising. This *must* advance the chain, because that is the
entire purpose of putting `docling` behind `pypdfium2`. It is tracked separately from hard
failure because if *every* parser in the chain comes back empty, the document is
`no_extractable_text`, not `parse_failed` — see §6.5.

**Degraded output — does not advance.** A parser that produced text but only `Unlocated`
anchors, or only page-level ones, **has succeeded.** Falling back on quality grounds makes
the chain non-deterministic, makes results depend on thresholds nobody tuned, and doubles
parse cost on exactly the documents that are already slow. If a parser's quality is
unacceptable, reorder the chain; do not make the runtime guess.

### 6.4 `Document.status` after parsing

[#1](https://github.com/mgd43b/manicule/issues/1) owns the enum; these are the values the
parse stage produces, and each is load-bearing.

| Status | Condition | Chunks |
|---|---|---:|
| `parsed` | a parser returned ≥ 1 chunk | ≥ 1 |
| `no_extractable_text` | chain completed, nothing hard-failed, every parser returned zero text | 0 |
| `parse_failed` | every parser in the chain hard-failed | 0 |
| `unsupported_media_type` | no chain matched and no `"*"` tail applied | 0 |
| `container` | an archive whose members were expanded into their own documents (§9) | 0 |

**The mixed case, stated because it is the one an implementation guesses at.** A chain where
one parser hard-failed and another returned empty, with no parser producing text, is
`parse_failed` — **not** `no_extractable_text`. A parser that broke leaves us genuinely not
knowing whether text was there, and `no_extractable_text` means something specific: the
tooling worked and there was nothing to find. Widening it to cover "something broke and the
rest found nothing" would make the 5% warning in §6.5 fire on library bugs and stop meaning
"you have a scanned corpus".

Falling back is **not** a status. It is `metadata.parser_used` plus
`metadata.parsers_attempted = [(name, outcome, reason)]`, surfaced through `doctor` (§6.6).
Status stays coarse enough to filter on; metadata carries the detail.

**The four chunk-less statuses still store the document** — id, uri, title, `content_hash`,
`version_token`, `original_ref`, metadata — with **zero chunks and zero rows in the vector
store.**

Storing the failures is the point. An unstored failure is re-fetched on every sync, is
absent from `document list`, and is invisible to any future re-parse. Stored, it has a
`version_token` so sync skips it, it appears in listings with a reason, and
`manicule document reindex --status no_extractable_text` becomes a one-line command the day
OCR arrives.

**A chunk-less document has no anchor, and gets no placeholder chunk.** The tempting
alternative — emit one `Unlocated` chunk with empty text so the document "has" one — puts a
vector of nothing into the index, where it matches queries by accident and consumes a slot
in every result list. Zero chunks means zero vector rows. The document is still citable at
document level via its `uri`, which is not an `Anchor` and is not pretending to be.

### 6.5 `no_extractable_text`, precisely

The settled OCR decision (`PLAN.md` §5) is only as good as this path, so it is specified in
full.

**Definition.** After the chain has run to completion: no parser raised a hard failure, and
the union of all blocks returned by all parsers contains zero characters after
normalisation (§3.2). Whitespace-only, form-feed-only and empty-string blocks all count as
zero.

**The fallback chain runs first, in full.** This is the answer to "does a fallback get to
try?" — yes, always, and empty output is precisely one of the two conditions that advances
the chain (§6.3). `no_extractable_text` is a statement about the *whole chain*, never about
one parser. A PDF whose default chain is `["pdf", "docling"]` gets both before the status is
assigned.

**Applies to any format, not just PDF.** An empty `.docx`, a `.pptx` of images with no
speaker notes, a `.txt` of whitespace, a `.zip` containing only unparseable members. The
status carries `metadata.reason` distinguishing them:
`"no text layer"`, `"document contains no text runs"`, `"all slides are images and no
speaker notes are present"`.

**Distinct from `parse_failed`, and the distinction is the deliverable.** `parse_failed`
means the tooling broke and the document may well contain text. `no_extractable_text` means
the tooling worked and there is nothing there to extract. They call for different actions —
one is a bug report, the other is a scanning-and-OCR question — and collapsing them into
"error" is what makes a corpus of scanned PDFs look like an indexing outage.

**Not an error for batch purposes.** It does not abort a batch (`PLAN.md` §4) and it does
not fail an ingest run. It is a normal outcome, counted and reported.

**Surfacing, three places:**

- `manicule index` prints a per-run summary line:
  `4 documents yielded no extractable text (likely scanned) — manicule document list
  --status no_extractable_text`.
- `manicule document list --status no_extractable_text` lists them with their reasons.
- `doctor` reports the count, and **warns when the ratio exceeds 5% of a source**, because
  at that point the likely explanation is a scanned corpus and the OCR decision deserves
  revisiting — which is exactly the trigger `PLAN.md` §5 names ("revisit when a real corpus
  contains scanned documents worth having"). The warning is how anyone finds out that day
  has arrived.

### 6.6 `doctor` and ingest reporting

`doctor` reports, per source and per media type:

- document counts by status, with `no_extractable_text`, `parse_failed` and
  `unsupported_media_type` broken out by reason;
- **fallback rate** — the share of documents where the primary parser did not produce the
  result. A rising fallback rate is the early signal of a library regression or a change at
  the source;
- **`Unlocated` and page-level ratios per parser, against the §3.4 budgets.** The test-time
  budget and the production signal are the same number, so a parser degrading on real
  documents in a way the fixtures do not cover is visible rather than inferred;
- **hard-split count** (§4.2) — chunks that fell through to a token split, which usually
  means a document full of minified or encoded content;
- the active `ChunkFingerprint` and `EmbedFingerprint`.

---

## 7. PDF

**`pypdfium2` is the fast path.** Licensing is the reason the obvious choice is wrong:
PyMuPDF is AGPL, which is incompatible with an MIT project regardless of how good it is.
pypdfium2 wraps Google's pdfium under a permissive licence and is not slower in any way that
shows up here.

`docling` and `marker` are optional entries in the fallback chain, not default dependencies.
They are layout models, they are heavy, and they earn their place on documents where
outline-free structure actually matters (§2.5) — configured per source, never on by default.

**Extraction and boxes.** Per page, take a text page, extract the text, and for each chunk's
character range ask the text page for the rectangles covering it. Multiple rects per range
is the normal case, not an edge case — a quote spanning a line break has one rect per line,
and a quote spanning a column break has rects on both sides of the gutter. They are stored
as they come (rule 3, §2.1).

**Traps, each of which produces a wrong citation rather than an error:**

- **Page rotation.** §2.2. Applied at parse time; stored coordinates are display
  coordinates.
- **Page index origin.** pdfium is 0-based; `PageAnchor.page` is **1-based**, matching what a
  viewer shows and what a person says. Converted once, at the boundary, and asserted by the
  discrimination test (§3.3, assertion 3), which is the test that catches an off-by-one.
- **Encrypted PDFs.** A PDF with a user password cannot be read; status `parse_failed` with
  `reason="encrypted"`. A PDF with only an owner password *can* be read by pdfium and is
  parsed normally — the distinction matters because owner-password PDFs are common and
  refusing them would drop real content.
- **Reading order.** pdfium returns text in content-stream order, which for a multi-column
  layout can interleave columns. Blocks are emitted in the order pdfium reports; no
  reordering heuristic is applied, because a reordering heuristic that is wrong produces
  chunks whose text never appeared contiguously in the document, which fails containment
  (§3.3, assertion 1) — and a multi-column PDF is a required structurally-hard fixture
  (§3.5) precisely so this is measured rather than assumed.
- **No text layer.** Zero characters across every page → the chain advances, and if nothing
  in it produces text, `no_extractable_text` (§6.5). Never an empty `parsed` document.

---

## 8. Code — tree-sitter

### 8.1 Grammar packaging is the real problem

40+ languages means 40+ grammars, each a C library. The historical options were all bad: one
PyPI package per grammar (40 dependencies, a compiler on the user's machine), or building a
shared library at install time (the API for which was removed from `py-tree-sitter` at
0.22).

**Decision: `tree-sitter-language-pack`, as a required dependency.**

It ships pre-built grammars for well over a hundred languages as binary wheels, tracks
current `py-tree-sitter`, and is maintained — it is the successor to the older bundled-
grammar package, which stopped tracking Python releases.

The obvious objection is size: the wheel is tens of megabytes, and `uv tool install
manicule` being one command is a stated goal (`PLAN.md` §1). The objection does not survive
arithmetic. The dependency set already includes MLX, LanceDB, and `sentence-transformers`
for the reranker — and `sentence-transformers` pulls PyTorch, which is on the order of a
gigabyte. Grammar wheels are a rounding error against that, and paying it buys the removal
of an entire failure mode.

That failure mode is the reason it is **required rather than an optional extra.** An
optional grammar pack means code files chunk one way on a machine that has it and another
way on a machine that does not — different boundaries, different embeddings, one corpus.
That is the same corpus-consistency hazard that put OCR out of scope, arriving through a
different door. A guardrail applied to OCR and waived for grammars is not a guardrail.

**Two obligations on whoever implements this:**

- **Record the real wheel size** in the PR that adds the dependency. "Tens of megabytes" is
  the estimate this decision rests on; if it turns out to be hundreds, the decision is worth
  re-examining and the arithmetic above is the thing to redo.
- **Audit the bundled grammar licences.** The pack itself is permissive, but it aggregates
  grammars from many upstream repositories with individually-chosen licences. Almost all are
  MIT or Apache-2.0, but "almost all" is not a licence audit, and this project has already
  rejected one excellent library on exactly these grounds. Dump the licence list at
  packaging time and fail the build on any copyleft entry, with an allowlist for any that
  are deliberately excluded from the shipped set.

### 8.2 Deriving `LineAnchor.symbol`

`symbol` is the qualified name of the smallest definition enclosing the chunk:
`TokenStore.refresh`, `parse_config`, `Anchor::render`.

Derived from a small declarative table mapping, per language, the node types that introduce
a symbol and the field holding its name:

```
python:     function_definition → name, class_definition → name
typescript: function_declaration → name, class_declaration → name,
            method_definition → name, interface_declaration → name
rust:       function_item → name, struct_item → name, impl_item → type
go:         function_declaration → name, method_declaration → name, type_declaration → …
```

Walk from the chunk's first node to the root, collecting names from matching ancestors, and
join innermost-last with **the language's own scope separator** — `.` for Python, JavaScript
and Java, `::` for Rust, C++ and PHP — because the symbol is read by people who know the
language, and `Anchor.render` for Rust reads as a mistake.

A language absent from the table yields `symbol=None` — a `LineAnchor` with exact line
numbers and no symbol, which is honest and still cites correctly. Because the pack version
is pinned (§8.3) and required (§8.1), which languages fall into that case is identical on
every machine, so this degradation does not reintroduce the corpus-consistency hazard.

The table is maintained in-repo rather than read from upstream `tags.scm` query files,
because the pack does not reliably ship those and a query file that is present for some
languages and absent for others produces exactly the machine-dependent variation §8.1
rejects. Consuming upstream tag queries where they can be vendored deterministically is a
worthwhile later improvement and is filed (§13).

`symbol` also feeds the breadcrumb (§5.1), so it reaches the embedder: a chunk of a
`refresh` method embeds with `src/auth/token.py > TokenStore > refresh` in front of it,
which is what makes a query for "token refresh" find the method rather than the README.

### 8.3 Grammar versions are part of the fingerprint

A grammar upgrade changes parse trees. Changed trees mean changed split points (§4.2), which
mean changed chunk boundaries, which mean stored embeddings that no longer correspond to the
chunks that would be produced today.

`grammar_pack_version` is therefore in `ChunkFingerprint` (§1.7). Because it can only affect
code documents, a mismatch on this field alone permits a **partial** re-parse — re-chunk and
re-embed the code documents, leave everything else — rather than the full re-index a
`budget_tokens` change demands. That distinction is worth having in the guard rather than
discovering it during an upgrade.

---

## 9. Archives

`.zip` recurses into the parser chain. This is new work; it needs the security thinking that
recursive extraction always needs.

### 9.1 Members are documents, not chunks

A member of an archive is a document in its own right. A PDF inside a zip is a PDF: it gets
its own `Document`, its own parser, its own chunks and its own anchors, exactly as if it had
been fetched directly.

```
uri              zip:<container-uri>!/reports/2026-q1.pdf
source_id        derived from container source_id + inner path
metadata         container_document_id, container_depth, member_compressed_size
```

The `!/` separator is the long-standing convention for addressing inside a container and
survives being pasted into a bug report.

**The container itself emits no chunks** and gets status `container` (§6.4). It is not
`no_extractable_text` — nothing failed, and conflating the two would put every archive into
the bucket that triggers the OCR warning in §6.5. The container is not indexed as a manifest
either; a chunk listing filenames is retrieval noise that competes with the real content
inside it. The member list is metadata, visible through `document list --container <id>`.

**Deletion cascades.** Removing a container removes its members; a container whose bytes
changed has its members re-derived rather than merged, since matching old members to new
ones is guesswork.

**Breadcrumbs nest.** A member's breadcrumb begins with the container's, so a chunk from
`archive.zip!/reports/2026-q1.pdf` embeds under `archive.zip > reports > 2026-q1.pdf > …`.
The nesting is what makes the member findable when its own filename is generic.

### 9.2 Depth and cycles

- **`maxDepth = 3`**, counted from the top-level document. A zip in a zip in a zip is
  already unusual; deeper is either a mistake or an attack. A member beyond the limit is
  stored as a document with status `unsupported_media_type` and
  `reason="archive nesting depth exceeded"` — visible, not silently dropped.
- **Cycle detection by content hash along the recursion path.** A zip cannot literally
  contain itself, but self-referential nesting via identical content is trivial to
  construct, and it costs nothing to defend: keep the set of member content hashes on the
  current path and stop when one repeats, with a reason.
- **Recursion is breadth-first with a global member budget**, so a wide archive cannot
  starve the rest of an ingest batch by depth-first descent into one branch.

### 9.3 Zip-bomb defence — four limits, because any one is bypassable

| Limit | Default | What it catches |
|---|---:|---|
| Total uncompressed bytes, whole tree | 1 GiB | the general case |
| Per-member compression ratio | 100:1 | the classic single-file bomb |
| Member count, whole tree | 10 000 | the many-tiny-files variant |
| Per-member uncompressed bytes | 64 MiB | one member exhausting the tree budget alone |

**The total-bytes limit is enforced while streaming, never from the header.**
`ZipInfo.file_size` is a field in the archive, which is to say it is attacker-controlled. A
bomb declares a small uncompressed size and expands anyway. Members are read through a
counting wrapper that aborts when the budget is exhausted, and the declared size is used
only as a cheap pre-filter.

Two more, which are about names rather than sizes:

- **Path traversal.** Member names may contain `../`, absolute paths, drive letters, or
  backslashes. Members are parsed in memory and never written to disk, which removes most of
  the risk — but the name still becomes part of a `uri` shown to users and stored in the
  index, so names are normalised and any name escaping the archive root is rejected with a
  reason rather than sanitised into something plausible.
- **Symlink members.** Zip can store symlinks in the Unix external attributes. They are
  skipped, with a reason. `zipfile` does not follow them today; the defence is against a
  future extraction path being added without remembering.

Exceeding a limit **fails that member, or that archive, and never the batch.** The container
gets `parse_failed` with the limit that tripped; members already extracted keep their
documents. An archive limit is a bound on damage, not a reason to lose the run.

**Encrypted members** yield a member document with status `parse_failed` and
`reason="encrypted archive member"`. The archive keeps going.

### 9.4 The trap: OOXML files are zips

`.docx`, `.xlsx` and `.pptx` are zip containers. So is `.epub`, so are many others. A
content sniffer looking for `PK\x03\x04` will happily identify every Office document as an
archive, and the archive parser will happily recurse into one and index `word/document.xml`
as a member.

Two defences, both required, because either alone fails:

- **Media type resolution runs before parser dispatch** and sniffing never overrides a
  declared type (§6.1). A `.docx` with a correct extension or a correct declared media type
  never reaches the archive parser.
- **The archive parser declares `media_types` narrowly** — `application/zip` and
  `application/x-zip-compressed` — and additionally refuses any input whose zip directory
  contains `[Content_Types].xml` (OOXML) or a `mimetype` member at offset 0 (ODF/EPUB),
  with `reason="OOXML container, not an archive"`. That covers the case where the extension
  is wrong *and* the declared type is `application/octet-stream`, which is the realistic
  version of this bug.

---

## 10. Email

**`.eml` — stdlib `email`.** The parser needs one rule to be deterministic, because
`LineAnchor` addresses a line span and a multipart message has several candidate bodies:

> The canonical body is the **first `text/plain` part in depth-first order**; failing that,
> the **first `text/html` part**, run through the HTML parser. Line numbers address the
> decoded, transfer-decoding-removed text of that part.

Headers (`From`, `To`, `Cc`, `Date`, `Subject`) become a single `prose` block preceding the
body, with its own line span. `Subject` also becomes the document title and the first
element of the heading path, since it is the only structure an email has.

Quoted reply chains are kept. Trimming them is a retrieval optimisation with a real downside
— the quoted text is frequently the only statement of the thing being replied to — and it is
not a parsing decision.

**Attachments recurse into the parser chain**, as archive members do (§9.1): each becomes a
document with `container_document_id` pointing at the message. This matches what the
Confluence connector does with page attachments
([`confluence.md`](connectors/confluence.md) §6), so a PDF is a PDF wherever it arrived
from.

**`.msg` is the one format whose obvious library is licence-incompatible.** The maintained
Python `.msg` parser is GPL-3.0, which is the same class of problem as PyMuPDF and has the
same answer: do not take the dependency.

The route that works is that a `.msg` file is a compound-file (OLE/CFBF) container, readable
with `olefile` (BSD, already permissive), holding MAPI properties as named streams. The one
that matters:

```
__substg1.0_007D001F    PR_TRANSPORT_MESSAGE_HEADERS   the original RFC 5322 headers
__substg1.0_1000001F    PR_BODY                        plain-text body
__substg1.0_0037001F    PR_SUBJECT
__attach_version1.0_#XXXXXXXX/…                        attachment storages
```

Property-name suffixes are the MAPI type tag: `001F` UTF-16 string, `001E` 8-bit string,
`0102` binary.

So `.msg` is a **shim, not a second parser**: read the transport headers and the body,
reconstitute an RFC 5322 message, and hand it to the `.eml` code path. Anchors, chunking and
round-trip behaviour are then identical between the two formats by construction, which is
worth more than it costs.

**Its trap:** `PR_TRANSPORT_MESSAGE_HEADERS` is absent on messages that never traversed a
transport — drafts and items in Sent Items, which is a large share of what people export.
There the shim synthesises headers from `PR_SUBJECT`, the sender properties and the
recipient table. The synthesised path is a required fixture (§3.5).

Because this is a MAPI property reader rather than a library call, it is sized as its own
unit of work and filed separately (§13). `.eml` ships in v1 regardless; `.msg` is not
blocking.

---

## 11. Structured data

`.json`, `.yaml`, `.yml`, `.toml`. The design goal is that **block text is an exact slice of
source lines**, which makes `LineAnchor` correct by construction rather than by
reconstruction — the alternative, pretty-printing parsed values and then hunting for them in
the source, produces anchors that drift the moment formatting differs.

- **YAML** — `ruamel.yaml` in round-trip mode carries line and column marks on mappings and
  sequences. Blocks split at top-level keys; `symbol` is the dotted key path.
- **JSON** — YAML 1.2 is very nearly a superset of JSON, so the same mark-bearing parser
  gives positions for JSON files. It is *nearly*, not exactly: duplicate keys, tab
  indentation and a few escape sequences differ. So JSON is validated with stdlib `json`
  first — that decides whether the document is valid and what its values are — and the
  mark-bearing parse is used only for positions. If the two disagree or the position parse
  fails, the file is still indexed and its blocks carry
  `Unlocated(reason="JSON source positions unavailable")`.
- **TOML** — `tomllib` gives values and no positions. TOML has a usable structural signal
  instead: table headers are `[dotted.name]` at line start. Blocks split at table headers,
  and the line span is found by a single forward scan for the literal header. Arrays of
  tables (`[[name]]`) are disambiguated by occurrence index. A document with no table
  headers at all — a flat key-value file — is one block with an exact whole-file span.
- **Very large structured files** split at the top-level key level and then, if a single
  key's value still exceeds budget, at the next level down, with `symbol` deepening
  accordingly. A leaf scalar longer than the budget is prose and splits as prose.

`Structured` carries the widest `Unlocated` budget in §3.4 (0.10) for exactly one reason:
the JSON position path can legitimately decline.

---

## 12. Where this differs from the prior art

> Confined to this section deliberately. It records design information — these are defects
> found in a working system, and each one motivates a rule above — but nothing here is
> load-bearing for the specification, and the rules stand on their own terms without it.

| | OpenDocuments | manicule |
|---|---|---|
| PDF pages | extracted text split on blank-line runs, fragments numbered as pages | page index from the PDF library; rects from character boxes |
| PDF boxes | none | per-line rects, rotation applied, top-left points |
| Token counting | `tiktoken` for `gpt-4o`, sampled and extrapolated past 10 000 characters | the embedder's own tokenizer, on the exact string it will see |
| Chunk budget | seven per-kind budgets, up to 800 tokens | one budget, 512, checked against the model's limit at startup |
| Budget applies to | the chunk text | `embed_text`, breadcrumb included |
| Empty parse | indistinguishable from a crash — both end as `error` | `no_extractable_text`, distinct, stored, reported, re-indexable |
| Fallback config | keyed by extension, ships empty | keyed by media type, ships real chains, missing parser is a startup error |
| Archive | entries decoded as UTF-8 text; no recursion | members are documents, parsed by the full chain |
| Archive limits | declared sizes trusted from the header | counted while streaming |
| Code structure | pattern-matched function and class boundaries | tree-sitter ASTs, real line anchors and symbols |
| Chunk size change | silent re-chunk | `ChunkFingerprint` mismatch refuses to run |

---

## 13. Filed, not deferred

Real tickets, because "later" is how these disappear.

| What | Why it is not in v1 |
|---|---|
| **`.msg` MAPI property reader** (§10) | a hand-written property reader, not a library call, and `.eml` covers the common case. The route is specified; only the work is out of scope |
| **OCR** (`PLAN.md` §5) | settled. `no_extractable_text` and the 5% `doctor` warning (§6.5) are the trigger for revisiting |
| **`docling` / `marker` in the default PDF chain** (§7) | layout models are heavy and unmeasured; they belong behind #15 like every other quality change |
| **Upstream `tags.scm` symbol queries** (§8.2) | better symbol coverage than the in-repo table, but only once grammar-query files can be vendored deterministically |
| **PDF reading-order recovery for multi-column layouts** (§7) | any heuristic here risks emitting text that never appeared contiguously; needs the measured baseline first |

---

## 14. Checklist against ticket #4

- **Each parser behind the protocol from #1** — §2.4, plus the `resolve` addition in §3.1.
- **Round-trip check** — §3, six assertions and two declared budgets per parser.
- **Structure-aware chunking consuming parser output** — §4. `kind` and `heading_path` are
  facts from the parser; the chunker never inspects prose to find structure.
- **OCR decided explicitly** — settled in `PLAN.md` §5; the mechanism is §6.5.
- **Chunk size settled** — §1. 512 tokens on `embed_text`, 64 overlap, guarded by
  `ChunkFingerprint`.
