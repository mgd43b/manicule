# Generation and chat

Design for the step that turns an assembled context into a streamed, cited answer, and for
the conversation state around it. Ticket [#7](https://github.com/mgd43b/manicule/issues/7).

Everything upstream is settled. The stores are built ([`storage.md`](storage.md)), the
embedder is settled ([`embeddings.md`](embeddings.md)), the ingest pipeline is designed
([`ingest.md`](ingest.md)), and the retrieval pipeline — including context assembly, the
token budgets and the confidence score — is settled in [`retrieval.md`](retrieval.md).
**`Context` is this document's input, and it is not a `RetrievalStage`'s output type by
design** ([`retrieval.md`](retrieval.md) §2.4, §7.1). That boundary is not reopened here.

This document starts where the model call begins: what is put in front of a model, what
comes back, what of that is allowed to reach a user, and what is written down afterwards.

> **Prior art.** OpenDocuments is referenced in clearly-marked callouts like this one, where
> the comparison carries design information. Everything outside these callouts stands on its
> own.

---

## 1. The whole design in five sentences

**A model never writes a citation, it selects one. Every citation is verified against the
retained source bytes before it reaches a reader, and an unverified one is deleted rather
than shown. Deleting a marker is the only edit anything downstream of the model may make to
the answer. Redaction is a projection applied on egress and never touches the artefact a
citation is verified against. Confidence and citation accounting are two different numbers
and are never combined into one.**

The first two are the ticket. The third is the one that is easy to lose: once you accept
that some citation must sometimes be removed, every convenient repair — trimming a sentence,
rewriting a clause, re-generating the tail — becomes available, and each of them changes what
the user was told for a reason the user cannot see. `contracts.md` §1 already made this call
for locations ("a location is correct, or it is absent") and PR
[#32](https://github.com/mgd43b/manicule/pull/32) made it for cited text. This document makes
it for the answer.

---

## 2. The shape

### 2.1 What runs, in order

```
      Query + Context (+ Conversation)
                │
        ┌───────▼────────┐
        │ policy         │  egress class; drops local-only passages, never adds
        ├────────────────┤
        │ history        │  whole turns, newest first, into history_tokens
        ├────────────────┤
        │ prompt         │  system + turns + numbered passages + question
        ├────────────────┤    ┌────────────────────────┐
        │ redaction      │    │ citation verification  │  starts here: it needs
        ├────────────────┤    │ (concurrent)           │  no model output
        │ Generator      │    └───────────┬────────────┘
        │   └─ litellm   │                │
        ├────────────────┤                │
        │ citation binder│ ◄──────────────┘  marker → slot → verified Citation
        └───────┬────────┘
                ▼
      AnswerEvent stream → persisted Message → feedback, sharing
```

Two things about that diagram are load-bearing.

**Verification starts before the first token.** It depends only on `Context`, which is
already known. By the time a marker appears in the stream — hundreds of milliseconds of
model latency later, at the earliest — the answer for that passage is usually already in
hand. This is what makes per-citation verification affordable on the answer path rather than
a thing that gets skipped for latency.

**The binder sits between the `Generator` and the caller, and it is not pluggable.** §2.2.

### 2.2 What is a protocol and what is deliberately not

`Generator` stays exactly the seam `contracts.md` §3 declares: `model_id`, and
`generate(query, context) -> AsyncIterator[Token]`. It is the provider adapter and nothing
else. Everything that makes a citation trustworthy — the marker binder, verification, egress
policy, redaction, persistence — lives above it, in `manicule.generation`, and **is not
behind a protocol.**

That placement is the same argument `retrieval.md` §2.4 makes for the hydrating join, applied
in the opposite direction, and for the identical reason. The join goes *inside* the dense
stage because a boundary configuration can omit is not a boundary. Verification goes
*outside* the `Generator` protocol because a boundary a **plugin** can omit is not a boundary
either — and a third-party generator is exactly the thing that would implement `generate` and
forget to verify anything. A plugin supplies a provider. It does not get to supply the part
of the system the ticket is about.

Concretely, an installed generator plugin can change which model answers, how it is reached,
what it costs and how fast it streams. It cannot change which citations survive.

### 2.3 One thing `Generator` needs and does not have

The startup cross-check `retrieval.md` §7.4 requires — profile budgets against the model's
context window, refused before the first query rather than discovered by exceeding it —
needs the window. Nothing on the protocol exposes it:

```
Generator
    model_id: str
    context_window: int        # ← does not exist today
    generate(query, context) -> AsyncIterator[Token]
```

This document does not edit `contracts.md`; three implementation tickets are building against
it. The change is filed as [#41](https://github.com/mgd43b/manicule/issues/41), and §4.3 says
what the number must mean, because the obvious reading of it is wrong on the default runtime.

It is additive and it is not `RetrievalStage` or `Anchor`, so it carries neither lock.

---

## 3. Citations

This is the ticket. Everything else in this document is in service of it.

### 3.1 The model never writes a citation — it selects one

The single design decision from which the rest follows.

Context passages are numbered when the prompt is built: slot 1 to slot *N*, where *N* is
`final_top_k` — 3, 5 or 10 by profile. The model is asked to cite **slots**, not paths, not
titles, not page numbers. When a marker naming slot 3 survives verification, the `Citation`
that reaches the caller is constructed entirely from `Context.passages[2]`:

```
Citation
    slot: int                  # ordinal within this answer
    document_id: str
    uri: str
    title: str
    heading_path: tuple[str, ...]
    anchor: Anchor
    chunk_id: str
    quote: str                 # chunk.text, whole and unmodified
    verification: Verification
```

**Not one field of that comes from the model.** The model contributes a small integer.

This is not a stylistic preference about prompt formats; it deletes an entire class of
failure. A model cannot invent a page number, because it never writes one. It cannot mangle a
file path, because it never sees one it could copy wrong. It cannot cite a document that was
not retrieved, because there is no slot for one. The failures that remain are enumerable, and
§3.3 and §3.5 enumerate them.

> **Prior art.** `generator.ts` asks the model to "Cite every claim using
> `[Source: filename#section]` format", builds each context block as
> `[Source: ${r.sourcePath}${section}]`, and then never parses the answer. There is no
> citation parser in the repository — `fullAnswer` is never matched, split, or validated —
> and the `sources` array attached to the response is the whole post-retrieval list,
> independent of what the model actually cited. A hallucinated
> `[Source: /docs/nonexistent.md#Foo]` is streamed verbatim, stored verbatim, and rendered as
> a citation. The label is also lossy by construction: it carries only
> `headingHierarchy.at(-1)`, so two chunks from the same file under the same leaf heading
> produce identical citations, and nothing maps the string back to a `chunkId`. Even if
> somebody wrote a parser, there is nothing for it to resolve to.

### 3.2 The marker, and why it is not `[1]`

```
[[cite:3]]            one slot
[[cite:3,5]]          several
```

`[1]` was the obvious choice and it is unusable. It occurs constantly in ordinary technical
prose, in bibliographies the corpus may contain, and — fatally — in code, where `argv[1]` and
`items[0]` are everywhere and the corpus is full of code blocks by design
([`parsing.md`](parsing.md) keeps them whole for exactly that reason). A binder scanning for
`[1]` eats the answer. `[[3]]` collides with wiki-link syntax; `[^3]` is a Markdown footnote
reference that a quoted passage can legitimately contain.

`[[cite:N]]` has no plausible collision in prose or code, is cheap to detect on a character
stream, and — the reason for the `cite:` prefix rather than a bare `[[3]]` — a malformed
attempt is still *recognisable as an attempt*, so it can be counted rather than mistaken for
prose. Four or five tokens per citation against a 1024-token answer budget is not a cost worth
optimising.

Three rules on the binder, all narrow on purpose:

- **It operates on the raw character stream and is not Markdown-aware.** Being Markdown-aware
  means parsing a partially-received document, which is guesswork, and the prior art shows
  where that leads (§3.6).
- **A marker that does not close within 64 characters is not a marker.** The buffered text is
  released verbatim. Without this bound, one unterminated `[[cite:` stalls the stream forever.
- **The binder only ever deletes, plus one normalisation of syntax it defined itself.**
  `[[cite: 3]]` and `[[cite:3 ]]` normalise to `[[cite:3]]`; a marker whose slots all fail
  verification is deleted; a marker with slots `3,5` where only 3 verifies becomes
  `[[cite:3]]`. No other character of the answer is touched by anything, ever. Surviving
  markers stay in the stored answer text, so a stored answer still says where its citations
  were, and `messages.sources` holds the citation records positionally.

**Marker syntax occurring inside a context passage is escaped when the passage is rendered.**
This is not hypothetical: manicule's own documentation describes the syntax, and manicule's
own documentation is exactly the sort of thing someone indexes. A passage containing a literal
`[[cite:3]]` that the model then quotes would otherwise bind — to a real passage, so it would
even verify — producing a citation nobody asked for.

### 3.3 Verification is a three-level ladder, and the level reached is reported

| Level | Check | Cost | Catches |
|---|---|---|---|
| **0 — bound** | The slot is an integer in `1..len(Context.passages)` | free | Invention. A model naming slot 9 of 5 |
| **1 — located** | The passage's anchor is not `Unlocated` | free | A citation pointing nowhere by the parser's own admission |
| **2 — resolvable** | `Parser.resolve(anchor, raw)` over the retained bytes returns text containing what the chunk claims | a blob read and a parse, cached | Anchors that have drifted from the document, missing or corrupt source bytes, and anchors written by a parser version that is no longer running |

Level 2 is the one that needs justifying, because `CONTRIBUTING.md` already imposes the
round-trip obligation on every parser and `assert_parser_contract` already enforces it. It is
still not sufficient, for four reasons that have nothing to do with parser quality:

1. **The parser suite runs against fixtures, not against this corpus.** A real document with a
   structure no fixture had produces anchors nothing has ever resolved.
2. **The bytes can be gone.** `original_ref` points into the blob store; a blob store can lose
   a file, and `retain_source_bytes` can be off (§3.7). A citation into a document whose bytes
   are missing cannot be shown with a highlight, and the reader should not be told otherwise.
3. **A stored conversation replays its citations later.** [`storage.md`](storage.md) §4.7 puts
   the ⚠️ anchor lock on `messages.sources` precisely because "a stored conversation's citations
   must keep resolving". At replay the document may have been re-ingested. This is the case where
   verification is not belt-and-braces — it is the only check there is.
4. **Restores and repairs.** A backup restored from before a parser change, or a `doctor`
   repair that rebuilt a derived index, can leave anchors written by code that no longer runs.

The level reached is recorded per citation and travels with the answer. A citation verified to
level 1 is a weaker claim than one verified to level 2, and the difference is visible rather
than averaged away.

### 3.4 What happens when verification fails

**The citation is dropped. The answer is not.**

Its marker is deleted from the answer text — the only edit anything is permitted to make — the
sentence stands exactly as the model wrote it, and the drop is reported in band on the response
and persisted alongside the message, carrying the slot it named and the reason it failed.

The three alternatives, and why each is worse:

**Refuse the answer.** Disproportionate: one bad marker in six destroys five good citations and
a correct answer. Worse, it is not implementable honestly on a stream — by the time a marker at
80% fails, 80% of the answer is on the reader's screen, and "refusing" it means retracting text
already delivered. That is the prior art's half-streamed-answer failure with the sign flipped.
It also hands the model a denial of service against itself: emit one bad marker, kill the
answer.

**Drop the sentence containing it.** The most tempting and the most damaging. Removing a
sentence changes the meaning of the ones around it — a following sentence beginning "This is
because…" now refers to nothing, and a dropped negation inverts a paragraph. And it cannot be
done reliably on a stream: sentence boundaries in text containing code blocks, lists and
abbreviations are a guess, and acting on that guess means the answer the reader gets is a
rewrite nobody reviewed.

> **Prior art, and what sentence-level surgery costs.** `grounding.ts` splits the answer on
> `/(?<=[.!?])\s+/`, annotates sentences, and reassembles with `.join(' ')`. Every newline,
> blank line, list structure and fenced code block in the answer is flattened into
> single-spaced prose — on the `precise` profile, which is the one that asks for code. The
> `code` intent prompt explicitly requests code blocks; the strict-mode guard silently
> destroys them.

**Surface a warning and keep the citation.** This is the defect `PLAN.md` #1 names, restated:
a citation that points at a page which does not exist, now with a badge. `contracts.md` §1
already refuses the "best guess with a caveat" member at the anchor level. A warning attached
to a wrong location is still a wrong location, and the badge is what gets ignored.

So: dropped, in band, with a reason. **Not a log line** — `retrieval.md` §4.1 already names
`console.warn('[retriever] FTS5 search failed, using dense-only')` as the shape of a defect
that leaves the product looking normal, and this is the same shape one layer up.

One escalation, because it is a different event: **if every citation in an answer is dropped
while the context was non-empty, the answer is flagged `ungrounded`.** A single failed marker
is a model slip. All of them failing is a model that ignored its context wholesale, or a
corpus whose bytes are gone, and it must not render as an ordinary confident answer. It is a
flag, not a refusal — the answer may still be useful and the reader gets to decide with the
fact in front of them.

**Zero citations offered is recorded, not judged.** An answer with no markers at all may be
the correct answer ("the sources do not cover this"), and no mechanism can distinguish that
from a model that forgot. So it is counted, surfaced, and left alone — and §12.3 explains
which mechanism *does* eventually distinguish them.

### 3.5 What verification does not catch, stated plainly

**Misattribution.** The model cites slot 3 for a claim that slot 1 supports. Every level of
the ladder passes: slot 3 is in range, its anchor is located, and it resolves to exactly the
text the chunk claims. The citation is *resolvable* and *wrong*.

This is not a gap to be closed later by a cleverer check; it is a different problem. Detecting
it means deciding whether a passage entails a sentence, which is the hallucination guard that
`retrieval.md` §13 defers behind a precision-and-recall measurement it does not yet have.

So the guarantee this system makes is exact and narrower than "the citations are correct":

> **Every citation resolves to a real location in a real document, and the text shown at that
> location is the text the model was given.** It is not a claim that the passage supports the
> sentence.

Saying the narrower thing is the point. A system that claimed the wider one would be believed,
and `contracts.md` §5 already records what this project thinks of guarantees nothing enforces.
The narrow claim is worth having on its own: it is the difference between a citation a reader
can open and check, and one that leads nowhere.

Two smaller uncaught cases, for completeness. A model can emit a marker in a place that makes
grammatical nonsense of the sentence; nothing repairs that, by §3.4. And a model can cite the
same slot for every sentence in an answer that drew on three; that is a quality problem, not a
correctness one, and it is visible in the citation accounting.

### 3.6 The quote is the chunk's text, whole

A `Citation.quote` is `Chunk.text`, byte for byte. Not `embed_text`, which carries the heading
breadcrumb (`contracts.md` §2) and is retrieval scaffolding rather than something anyone
quoted. Not a normalised form — `manicule.testing.normalise` says it itself: "Stored text is
never normalised… showing a whitespace-flattened, ligature-substituted rendering of a
quotation is a change to the quotation." Not a trimmed form.

`retrieval.md` §7.3 already forbids assembly from trimming a passage to fit, on the grounds
that an `Anchor` describes the whole chunk and trimming makes the anchor claim more than the
text says. That rule does not weaken at the display layer. An interface that wants to show
less may show a **contiguous substring with an explicit indication that it is one**; it may
never store that substring, send it back as the source, or let it reach a place where it could
be mistaken for the passage.

> **Prior art.** `fitToContextWindow` truncates the last chunk and pushes
> `{ ...chunk, content: truncatedContent.trim() + '...' }` into the results, and that mutated
> object is what is persisted into `messages.sources` and rendered in the source-preview
> modal. `attachParentContext` goes further and swaps `content` for the enclosing
> `parentSection`, so on `balanced` and `precise` the "chunk" a user inspects is a different
> span from the one `chunkId` names. `expandWithSiblings` injects neighbouring chunks that
> were never retrieved on relevance, at a fabricated `score * 0.6`, into the same list. Three
> independent ways for the displayed source to be something other than the source.

### 3.7 Where verification runs, and what it actually costs

**Per distinct document, not per citation.** One parse of a document serves every anchor in
it. With `final_top_k` between 3 and 10 and passages frequently sharing a document, the usual
answer needs one to three parses.

**Cached on `(chunk_id, document.version_token)`.** Chunk ids are content-derived and
`version_token` changes whenever the document does, so the key is exact and can never go
stale — the same property `retrieval.md` §7.2 relies on for token counts. A verified anchor
stays verified until its document changes. Across a corpus's life this is close to a
once-per-chunk cost, not a per-query one.

**Started concurrently with the generation call**, since it needs no model output. On a warm
cache it has finished before the request reaches the provider.

**Bounded, and the bound is a failure rather than a bypass.** `citation_verify_timeout_s`
defaults to 5.0, measured from the start of the answer, which is generous because the work
began before the first token. A marker whose verification has not completed when the marker
needs to be emitted waits for the remainder of the budget and is then **dropped, with reason
`verification_timeout`** — distinct from `unresolvable`, because one means the disk is slow
and the other means the citation is wrong. Dropping a possibly-good citation is the
uncomfortable half of this; sending an unverified one under a design whose entire claim is
verification is the unacceptable half. A nonzero timeout rate is an operational defect that
`doctor` reports, not a property of the corpus — the same distinction `retrieval.md` §4.4
draws between `exhausted_budget` and `exhausted_corpus`.

**When the bytes are not retained.** `StorageSettings.retain_source_bytes` may be off, which
is legitimate and documented as making every re-index a re-crawl. Level 2 is then impossible.
The response says so: verification degrades to the strongest level available and **names the
level it reached**, rather than silently reporting the same word for two different amounts of
checking. Because the setting is startup configuration, the degradation is knowable at
startup, so it is also a `degraded` health check with a remedy naming the setting — not a
per-query surprise.

### 3.8 One predicate, shared with the parser suite

The runtime check and the test-suite check must be the same check. Two notions of "this anchor
resolves" would drift, and the drift would show up as citations that pass CI and fail in
production, or the reverse.

`manicule.testing.roundtrip._assert_containment` is that predicate today:

```python
normalise(item.text) in normalise(text or "")
```

`#7`'s implementation extracts it — `normalise` is already a public module with no test
dependencies — into one shared function that both `_assert_containment` and the runtime
verifier call. Not a copy. `manicule.testing.normalise` exists precisely because "the usual
repair — loosening the comparison per parser until the suite passes — leaves no assertion at
all", and a second, runtime-only comparison would be that repair by another route.

Tightness, discrimination, determinism, the location budget and idempotence stay where they
are: they are properties of a *parser*, checkable only over a corpus of fixtures, and none of
them is answerable about a single anchor at answer time.

---

## 4. The provider interface

### 4.1 One call

```python
await litellm.acompletion(
    model=...,           # "<provider>/<model>"
    messages=[...],      # §6
    stream=True,
    stream_options={"include_usage": True},   # §4.10 — not optional
    base_url=...,        # None for hosted providers with a default
    api_key=...,
    temperature=...,
    max_tokens=...,
    timeout=...,
)
```

One dependency in the `generation` extra, one call, and local versus hosted differs by
`base_url` and the model prefix. `contracts.md` §5 forbids provider-specific types and this is
what makes that cheap rather than aspirational: there is no per-provider branch to keep a type
alive.

> **Prior art.** Five hand-written clients, one per vendor, with no shared base. The OpenAI
> and Grok SSE loops are byte-identical apart from an error string; Anthropic's is the same
> loop minus the `[DONE]` sentinel. All five duplicate the health check, the environment-
> variable fallback, and the reader/decoder/buffer/`lines.pop()` framing. **Google's does not
> stream at all** — it calls `:generateContent` rather than `:streamGenerateContent` and
> yields the entire answer as one chunk, so a Gemini deployment produces "streaming" that is
> a single blob after a stall of up to 120 seconds, invisible to every layer above it. The
> shared `GenerateOpts` type is the OpenAI chat-completions shape, so Google's
> `thinkingBudget` had to be smuggled in through a global environment variable and Ollama's
> `think: false` is hardcoded in the request body. Provider identity leaks into core as
> `const cloudProviders = new Set(['openai', 'anthropic', 'google', 'grok'])`.

**litellm's types stop at the boundary.** `ModelResponse`, `Delta`, `StreamingChoices` and
litellm's exception classes are converted inside the adapter into `Token`, `Usage`,
`FinishReason` and `manicule.core.errors` types. Nothing above the adapter imports litellm,
and `tests/test_import_boundary.py` keeps it out of `import manicule`.

### 4.2 Model selection, and the `ollama_chat/` trap

`LlmSettings` carries `provider` and `model` separately; litellm wants one string. The
composition is `f"{provider}/{model}"` with one correction that is not cosmetic:

> **The Ollama provider prefix is `ollama_chat`, not `ollama`.** They are different endpoints.
> `ollama/` routes to `/api/generate`, the raw completion endpoint, which does **not** apply
> the model's chat template. `ollama_chat/` routes to `/api/chat`, which does.

A chat-tuned model driven through the completion endpoint gets a prompt in a shape it was
never trained on. It still answers — plausibly, in fluent prose, with no error anywhere — and
it follows instructions measurably worse, which for this design means it follows the citation
protocol worse. That is a silent quality regression with no symptom other than more dropped
markers.

So `ollama` in configuration maps to `ollama_chat/` in the call, the resolved model string is
recorded in the trace, and `doctor` prints it. Configuration keeps the name an operator
expects; the wire gets the endpoint that works.

Everything else follows litellm's conventions unchanged: `openai/gpt-…`, `anthropic/claude-…`,
`gemini/…`, `xai/…`, and `openai/<model>` with a `base_url` for any OpenAI-compatible endpoint.

**One generation model, not two.** The ticket says "bulk work local, interactive work wherever
is best", and the split is real but it is not a generation-time switch. The bulk path in this
system is *embedding*, which is local and in-process by construction (`PLAN.md` §7) and never
negotiable. Generation is the interactive path. Where a cheaper secondary generation call
appears — a conversation title — it uses the same generator with a smaller `max_tokens`, not
a second configured model, because a second model means a second context window to cross-check
(§4.3), a second tokenizer to calibrate (§9), a second egress class to police (§7.1), and a
second thing to get wrong. The deployment choice the ticket describes is made once, in
configuration, and the invariant across it is §15's: **which model answers may change
throughput and wording; it may not change which citations survive.**

### 4.3 The context window is the runtime window, not the advertised one

`retrieval.md` §7.4 requires a startup refusal:

```
context_tokens + history_tokens + system_prompt_tokens + generation_reserve
        must fit the configured generator's context window
```

and assigns the enforcement point here. The trap is in the last three words.

**Ollama's served window is `num_ctx`, a runtime option that defaults far below what modern
models are trained for**, and a prompt exceeding it is truncated to fit rather than refused.
So `qwen2.5:14b`, trained at 32768, is served at the default `num_ctx` unless something says
otherwise — and manicule's `balanced` profile assembles 16384 context tokens plus 1024 of
history against it. The overflow does not error. It is discarded, and it is discarded from the
**front** of the prompt, which is where the system prompt and the citation protocol live.

The failure mode is therefore: the model answers, fluently, having never seen the instruction
telling it how to cite, and every marker is missing or malformed. Nothing raises. It reads as a
weak model.

Three consequences:

- **`Generator.context_window` means the window that will actually be served**, not the
  model's trained maximum. For Ollama that is read from `/api/show` combined with what
  manicule itself sets.
- **manicule sets `num_ctx` explicitly on every Ollama call**, derived from the profile
  arithmetic above. Derived rather than configured, so it cannot disagree with the budget it
  exists to satisfy.
- **The first response's true prompt count is checked against the estimate** (§9.2). If a
  server truncated, the true count comes back pinned at the window and the discrepancy is
  visible immediately rather than as a mysterious quality drop.

**The shipped defaults do not all fit, and this document says so rather than leaving it to be
discovered.** With `LlmSettings.model = "qwen2.5:14b"` (32768 trained) and a system prompt of
roughly 400 tokens:

| Profile | context | history | system | reserve (`max_tokens`) | total | fits 32768? |
|---|---|---|---|---|---|---|
| `fast` | 8192 | 512 | ~400 | 1024 | ~10128 | yes |
| `balanced` | 16384 | 1024 | ~400 | 1024 | ~18832 | yes |
| `precise` | 32768 | 2048 | ~400 | 1024 | ~36240 | **no** |

`precise` does not fit the default model's own window, before `num_ctx` is even considered.
That is a real finding, not a rounding issue, and the honest response is the one the rule
already prescribes: **it is refused at startup, with both numbers named and the three fixes
listed** — a model with a longer window, a lower `context_tokens` override, or a different
profile. The profile numbers belong to
[#6](https://github.com/mgd43b/manicule/issues/6)/[#1](https://github.com/mgd43b/manicule/issues/1)
and are not changed here. What is not acceptable is shipping a profile that silently truncates.

### 4.4 Startup refusals

`CONTRIBUTING.md`: "Anything configuration names must be checked against what is installed, at
startup, with the alternatives listed." Applied to a generator, in `setup()`:

| Check | On failure |
|---|---|
| litellm recognises the composed model string | Refuse, naming the string and the provider prefixes available |
| Credentials for a hosted provider resolve | Refuse, naming the environment variable it looked for |
| For Ollama: the endpoint answers, and the named model is present | Refuse, listing the models that *are* pulled and printing `ollama pull <model>` |
| The window cross-check of §4.3 | Refuse, naming both totals and the three fixes |
| Custom redaction patterns compile | Refuse, naming the pattern and the regex error (§7.3) |
| `cloud_allowed = false` with a remote endpoint | Refuse. Not a silent fallback to local (§7.1) |

A live probe of a hosted provider is not one of these. It costs money and latency on every
start, and the two things that can be checked without it — the credential resolves, litellm
knows the model — catch the mistakes people actually make. `doctor` performs the live probe on
request, which is where a check with a cost belongs.

### 4.5 Three timeouts, because one covers the wrong interval

```
first_token_timeout_s   60      connect, queue, prompt eval, model load
stream_idle_timeout_s   30      the gap between two tokens
timeout_s              120      total wall clock for one generation
```

> **Prior art.** `fetchWithTimeout` builds an `AbortController`, sets a 120-second timer, and
> clears it in a `finally` around `await fetch(...)` — which resolves **when the response
> headers arrive**. The entire budget covers time-to-first-byte. Once streaming begins there
> is no timeout at all, so a provider that opens a stream and then stops sending blocks
> forever, and the request is indistinguishable from a slow answer.

`first_token_timeout_s` is generous because a cold local model must be loaded into memory
first, and on a Mac that is a real multi-second cost the first time (§15). `stream_idle_timeout_s`
is the one the prior art lacks entirely: it is what turns a hung provider into an error. The
total is the backstop for a model that streams slowly forever, and it is what
`LlmSettings.timeout_s` becomes.

All three produce the same in-band outcome as any other mid-stream failure (§4.7), with a
distinct error string, because "the provider never started" and "the provider stopped
half-way" call for different remedies.

### 4.6 Retries stop at the first token

**Before the first token, a failure is retryable: nothing has been delivered, so a retry is
invisible and correct.** Bounded exponential backoff with jitter, `max_retries` defaulting to
2, on connection errors, timeouts, 429 and 5xx. Not on authentication failures, not on
context-window errors, not on content filters — none of those get better by being asked again,
and retrying them turns one clear error into three and a longer wait.

**After the first token, a failure is terminal.** There is no correct retry: restarting means
the reader watches the answer rewind and rewrite itself, and continuing means splicing two
independently-sampled answers into one text that neither model produced. Both are worse than
stopping, and both would put text on the screen that no single generation ever emitted, which
this design refuses in the same way it refuses rewriting a sentence (§3.4).

**A specific consequence: litellm's own `num_retries` is not used for streaming.** litellm
retries the whole `acompletion` call, which under `stream=True` restarts the stream from the
beginning — exactly the rewind above, done by a library that cannot know a token was already
delivered. The retry loop is manicule's, and its boundary is the first token, because that is
the only place that knows.

### 4.7 What a provider failure does to a half-streamed answer

Three things, and the second is the one prior art gets wrong in both directions.

**It is reported in band.** `Token(finish_reason=ERROR, error=...)`. The type was built for
this — "A stream that dies mid-answer says so in band; it does not just stop." A truncated
answer that simply ends is indistinguishable from a complete one, and the reader has no way to
know which they got.

**The partial answer is persisted.** Whatever text was produced is written as a `Message`
carrying its finish reason, so it exists on the server, has an id, can be shared, and — the
part that matters — **can be given feedback.** §5.3 covers how that survives a cancelled task.

> **Prior art, both halves.** On the streaming path, persistence is gated behind
> `if (!streamError && fullAnswer)`. A stream dying at 80% leaves the reader looking at 80% of
> an answer that exists nowhere on the server: no `messages` row, no `query_logs` row, no
> `queryId`, so no feedback is even possible on the answers that most need it. The browser
> keeps the partial text visible, so a reload silently deletes it. On the **non**-streaming
> path the same failure does the opposite: the exception is swallowed and `answer` is set to
> `'An error occurred while generating the answer. Please try again.'`, returned HTTP 200 with
> the real `sources` array and a real confidence score attached, persisted as an assistant
> message, **and cached for five minutes** — so every identical query for the next five minutes
> receives a canned error string presented as a confident, sourced answer. One failure, two
> endpoints, two opposite bugs.

**Citations already emitted stand.** They were verified when they were emitted; the provider
failing afterwards says nothing about them. The answer is marked truncated, not uncited.

### 4.8 Errors are mapped, not re-exported

litellm raises a mapped exception hierarchy — `AuthenticationError`, `RateLimitError`,
`APIConnectionError`, `ContextWindowExceededError`, `ContentPolicyViolationError` and others.
Those are litellm types, so they stop at the adapter and become `manicule.core.errors` types,
for the reason `contracts.md` §5 gives: a provider-shaped type in core forces every consumer
to learn every provider.

The mapping preserves what an operator needs to act, per `CONTRIBUTING.md`'s rule that errors
name what was wrong, what was expected, and what to do:

- `ContextWindowExceededError` is a **defect in the §4.3 cross-check**, not a runtime condition
  to absorb. It means the estimate and the server disagreed by more than the safety factor. It
  surfaces with both counts and the model name.
- `AuthenticationError` names the environment variable that was read.
- `RateLimitError` retries only under §4.6 and then reports the provider's own retry hint if
  there is one.
- Everything unmapped keeps the provider's message text. Discarding it is how "OpenAI error:
  429" becomes an operator's whole afternoon.

> **Prior art.** `throw new Error(\`OpenAI error: ${res.status}\`)` — the response body is
> discarded, so quota, rate-limit and content-filter details never reach anyone. Worse, all
> four streaming clients wrap frame parsing in `catch {}`, so a provider emitting an error
> object mid-stream (OpenAI's `{"error": {...}}` frame) is dropped as "malformed" and the
> stream simply ends early, indistinguishable from a complete answer. Anthropic's client
> handles only `content_block_delta` and ignores `error` and `message_stop` entirely.

### 4.9 Ollama is optional, and the install proves it

`generation = ["litellm>=1.55"]`. No Ollama client library — litellm speaks HTTP — and no
import of anything Ollama-shaped anywhere. Ollama is a *runtime* dependency of one
*configuration*, not an install dependency of manicule, which is what the ticket's comment
requires and what keeps `uv tool install manicule` a single command.

Two behaviours follow. A hosted configuration never probes an Ollama endpoint, never mentions
it in an error, and never reports it in health. And an Ollama configuration whose endpoint is
absent fails at startup with `ollama serve` and `ollama pull <model>` in the remedy — not at
the first question a user asks.

### 4.10 `stream_options` is not optional

OpenAI-compatible streaming **omits usage entirely unless `stream_options: {include_usage:
true}` is sent.** Without it, `usage` is `None` on every chunk including the last.

That is not a missing statistic. `retrieval.md` §7.2 builds the whole token-budget calibration
on the true prompt count — "Measuring once beats a safety factor forever, and #7 owns the call
that makes it available" — and without this flag the calibration never runs, never fails, and
never says why. The estimate stays a guess forever while everything reports normally.

So it is sent on every call, and a provider that returns no usage after being asked is recorded
as such (§9.3) rather than treated as agreeing with the estimate.

---

## 5. Streaming

### 5.1 An async generator, with `finally` around the yield

`Generator.generate` returns an `AsyncIterator[Token]` and in practice every implementation is
an async generator — the same situation `Parser.parse` is in, and
[#35](https://github.com/mgd43b/manicule/pull/35) already established what that costs and what
it requires. The `parsing()` context manager and `aclose()` helper in
`manicule.core.protocols` exist because a generator abandoned part-way stays suspended holding
whatever it had open at the `yield`, and CPython finalises a live async generator through the
loop that created it — so one still suspended when that loop closes is finalised late, from the
wrong loop, and the observable result is a crash inside the interpreter's allocator on a stack
naming no library anyone here wrote.

Generation gets the identical treatment, and the resource being held is worse than a file
handle:

```python
async with generating(generator, query, context) as tokens:
    async for token in tokens:
        ...
```

`generating()` is `parsing()`'s sibling, in the same module, closing the stream in a `finally`
on every exit path. Implementations hold the provider connection in `try`/`finally` around
their `yield`, because `aclose()` throws `GeneratorExit` in at the suspension point and only a
`finally` runs after that.

**What an abandoned generation stream holds is an open HTTP response to a model that is still
working.** On a hosted provider that is tokens being billed for an answer nobody will read. On
Ollama it is worse in a way that matters on a single machine: the model keeps generating until
the client goes away, so one abandoned stream occupies the only local model until it finishes.
A user who closes a browser tab and asks again immediately is queued behind their own
abandoned answer.

### 5.2 Two ways a stream dies, one cleanup

They arrive differently and both must release:

- **`aclose()`** — the consumer stopped early, or `generating()`'s `finally` ran. Raises
  `GeneratorExit` at the `yield`.
- **`asyncio.CancelledError`** — the client disconnected and the task was cancelled. Arrives at
  whatever `await` the generator is suspended on, which is usually inside the provider read.

One `try`/`finally` covers both. Two rules on what may go in that `finally`:

**Cleanup awaits nothing unbounded.** After `GeneratorExit`, an `await` that never completes
hangs `aclose()` — and `aclose()` is being awaited by a `finally` somewhere else, so the hang
propagates into whatever was trying to shut down. Closing an httpx response is bounded; a
tidy-up that talks to a database is not, and does not belong here.

**Cleanup never yields.** Yielding after `GeneratorExit` raises `RuntimeError: async generator
ignored GeneratorExit`, which replaces a clean shutdown with a confusing traceback. The final
`Token` carrying `finish_reason` is emitted on the normal path only; a stream nobody is reading
has nobody to tell.

**The provider connection close has its own hard deadline.** Past it the connection is
abandoned to the pool's own teardown, because a shutdown path that can block indefinitely on a
misbehaving remote server is a worse failure than a leaked socket.

> **Prior art.** The server's SSE handler has no `AbortController`, no disconnect detection and
> no `stream.onAbort`; `RAGEngine.queryStream` is an async generator with no `try`/`finally` at
> all. If the client disconnects, the loop keeps pulling from the provider to completion and
> the upstream `fetch` is never aborted — the call is paid for in full. The web client does
> pass an `AbortSignal`, but it only aborts the browser's own fetch; nothing propagates to the
> server.

### 5.3 The partial answer is persisted by the wrapper, and the write is shielded

Persistence cannot live in the consumer, because on a disconnect there is no consumer left to
run it. It lives in the answer wrapper — the thing that owns the accumulated text — in its own
`finally`, which runs on completion, on error, and on cancellation alike.

That `finally` runs in a task that has just been cancelled, so **the write is shielded and
bounded**: `asyncio.shield` over a short database write with a deadline, and no possibility of
the shielded work restarting the request or issuing another provider call. Without the shield,
the first `await` inside the `finally` re-raises `CancelledError` and the partial answer is
lost — which is precisely the outcome §4.7 exists to prevent, arriving by a subtler route than
the prior art's `if (!streamError)`. With an unbounded shield, a cancelled request can outlive
its own shutdown.

The persisted message carries its finish reason (`stop`, `length`, `content_filter`, `error`)
and its citation accounting, so a truncated or abandoned answer is a first-class stored object
that can be read, shared and rated rather than a gap in the record.

### 5.4 One use, and honest backpressure

**The iterator `generate` returns is single-use.** Restarting it means a second provider call
with a second bill; it is not a replayable sequence, and nothing pretends otherwise.

**A slow consumer slows the provider, and that is correct.** The alternative is buffering an
unbounded amount of generated text in memory to keep the model busy on behalf of a client that
cannot keep up. Backpressure through to the provider is the behaviour that makes an abandoned
stream cheap and a slow reader harmless.

---

## 6. The prompt

### 6.1 Messages, not a string

A `messages` array with a real system message, real prior turns, and the question last.

> **Prior art.** `generateAnswer` never passes `systemPrompt` to the provider, so the field is
> always `undefined` on the answer path: Anthropic's native `system` parameter is never set,
> Google's `systemInstruction` is never set, OpenAI and Grok never send a `role: "system"`
> message, and Ollama never uses `/api/chat`. Every instruction — including the citation rule —
> is concatenated into a single user message built by string interpolation. `GenerateInput.systemPrompt`
> is unreachable from the engine entirely; there is no configuration path to a custom system
> prompt at all. `maxTokens: 1024` is hardcoded in the same function, overriding the
> `generationBuffer` the context-window fitter carefully reserved.

Order: **system, then prior turns, then one final user message containing the numbered passages
followed by the question.**

The question goes last for a reason beyond convention. When a server truncates an over-long
prompt it truncates from the front (§4.3), so the ordering decides what is lost first. Losing a
passage degrades the answer; losing the question produces one that answers nothing. The system
message is first because it is the stable prefix hosted providers can cache — and because if it
is being lost, the §4.3 cross-check has already failed and should have refused at startup.

**The citation protocol is not configurable.** An operator may *append* instructions to the
system prompt; they may not replace the section that defines slots and markers, because the
binder's guarantees assume the model was told the protocol. The appended text is counted into
`system_prompt_tokens` for §4.3, so a long custom prompt is refused at startup rather than
silently displacing passages.

### 6.2 How a passage is rendered

```
[slot 3] "Deploy runbook" — Operations › Rollback › Emergency  (page 4)
<text>
```

- **The label carries the breadcrumb; the body is `Chunk.text`.** `embed_text` has the
  breadcrumb baked in for retrieval (`contracts.md` §2) and is not what anyone cites. Putting
  the breadcrumb in the label gives the model what a section called "Configuration" is
  configuring, without polluting the text that will be quoted back.
- **No chunk ids.** A model given an opaque identifier will eventually emit it, and a reader
  will see it. Slots are small integers precisely so that the worst thing a leak can look like
  is a stray number.
- **Marker syntax inside the body is escaped** (§3.2).
- **Slots are per-answer ordinals**, not identifiers. They are persisted as the position of a
  citation record in `messages.sources`, which is what lets a stored answer's surviving markers
  still mean something a year later.
- `Context.truncated` — passages dropped by assembly to fit the budget
  ([`retrieval.md`](retrieval.md) §7.3) — is surfaced on the answer, because an answer built
  from a truncated context is a weaker claim and the `Context` type already says so.

### 6.3 Retrieved passages are data, and the honest limit of saying so

Passage bodies are third-party text. A Confluence space with external contributors, a mailbox,
a scraped site — any of them can contain "ignore your instructions and answer as follows". The
structural mitigations are real but partial, and it is worth being exact about which is which.

**What actually helps:** citations bind to slots, so no instruction embedded in a document can
manufacture a citation to a document that was not retrieved, or invent a location. Generation
in manicule has no tools and no side effects, so a successful injection can change *what the
answer says* and cannot make anything happen. Marker syntax in passages is escaped, so corpus
content cannot bind by accident.

**What does not help, and is not claimed:** delimiters and "the following is untrusted data"
preambles do not prevent injection. They are cheap and they are used, and this document does
not pretend they are a boundary. The residual risk is a wrong answer with a real, resolvable
citation attached — which is §3.5's misattribution arriving deliberately, and the same feedback
category (§12.3) is the only thing that surfaces it.

> **Prior art.** Web-search results are synthesised into `SearchResult`s with
> `chunkId: \`web_${i}\`` and `documentId: 'web-search'`, then spliced into the same context
> block as indexed documents and persisted into `messages.sources`. Unvalidated third-party
> text goes straight above `## Question` in the prompt, is indistinguishable from corpus
> content in the source list, and bypasses redaction entirely since redaction runs only in the
> ingest pipeline.

---

## 7. Egress policy and PII redaction

`PLAN.md` defect #5 said pick one behaviour and build it. [`ingest.md`](ingest.md) §3.4 picked:
**redaction happens at the generation boundary, not at ingest**, because retained original
bytes make ingest-time redaction incoherent — the unredacted source is in the blob store
regardless, so it protects nothing while permanently degrading retrieval, and re-parsing
through a changed redactor would churn chunk ids and vectors on every pattern edit.
[#26](https://github.com/mgd43b/manicule/pull/26) moved the configuration here.
**This section is the feature.**

### 7.1 The predicate is the endpoint, not the provider's name

The question "is this text leaving the machine?" has an obvious wrong answer.

> **Prior art.** `const cloudProviders = new Set(['openai', 'anthropic', 'google', 'grok'])`,
> hardcoded in `bootstrap.ts`. It is wrong in both directions the moment `base_url` exists:
> `provider: 'ollama'` pointed at `http://gpu-box.lan:11434` is classified local and crosses
> the network, while an OpenAI-compatible endpoint on `127.0.0.1` is classified cloud. The set
> also has to be edited every time a provider is added, and nothing fails when someone forgets.

The classification is derived from the **resolved endpoint**:

```
egress = LOCAL   if the resolved base_url host is loopback
         REMOTE  otherwise, including every hosted provider default
```

Loopback is the only host that is local by construction. A LAN address is remote — it is
another machine, on a network, and "the office GPU box" is exactly the case where an operator
believes otherwise.

Two honest limits, stated because a security predicate that is quietly approximate is worse
than one that is openly so. A local proxy on `127.0.0.1` that forwards to a hosted provider
classifies as `LOCAL` and manicule cannot see through it; and a hosted provider is trusted to
be what its hostname says. Neither is detectable from inside the process. Both are covered the
same way: the classification is **recorded on every answer and in the trace**, so what manicule
believed is auditable, and `RedactionSettings` gains a `scope` (§7.2) so an operator who knows
about the proxy can force redaction on regardless.

**`cloud_allowed = false` plus a remote endpoint is a startup refusal**, not a silent fallback
to a local model. A policy that quietly downgrades the model instead of failing is a policy
nobody knows is in force.

### 7.2 What is redacted, and what deliberately is not

Redaction applies to **everything on the egress path and nothing else**:

| Redacted | Not redacted | Why |
|---|---|---|
| Passage bodies in the prompt | `Chunk.text` in the index | The index is local and is what a citation is verified against (§7.4) |
| The user's query, as sent | `query_logs.query` | Storage is local; `AtRestSettings.redact_logs_content` is the separate, existing knob for logs |
| Conversation history, as sent | `messages.content` | Same reason. A stored conversation is local |
| — | The model's answer | It has already left. Redacting the reply protects nothing and would rewrite the answer, which §3.4 refuses |
| — | The system prompt and slot labels | manicule's own text, not user content |

**Redacting the query is the correction the prior art most needs** — a user pasting a
customer's email into the chat box currently ships it to the provider verbatim and stores it in
plaintext — and it has a cost worth naming: retrieval already ran on the *unredacted* query, so
the model may be asked about `[REDACTED]` for a term the retrieval matched exactly. That is the
right trade (retrieval is local; the model is not) and it is a real source of confused answers
when aggressive patterns are enabled.

`RedactionSettings` gains one field:

```
scope: remote | always      default: remote
```

`remote` is the whole point of the feature: what leaves the machine is redacted, what stays
does not, so a fully local install pays nothing. `always` exists for the proxy case in §7.1 and
for operators who want the model's *input* uniform regardless of where it runs — which is also
the only way to make a local and a hosted deployment produce comparable answers.

### 7.3 The detectors

Built-in, named, versioned detectors — `email`, `phone`, `credit-card`, `ip-address` — selected
by name in `RedactionSettings.patterns`. Named rather than expressed as raw regexes in
configuration, so they can be tested, improved and reasoned about, and so a config file is a
policy rather than a program.

Custom patterns are accepted and **compiled at startup; a pattern that does not compile is a
refusal.**

> **Prior art.** Custom patterns are compiled inside a bare `catch {}`. A bad regex is silently
> dropped, no warning, and redaction is quietly weaker than the configuration says it is —
> which is the exact failure `CONTRIBUTING.md`'s "a setting that appears to be in force and
> silently is not" rule exists to prevent.

On the three methods:

- **`replace`** (default) substitutes `replacement`. Simple and total.
- **`hash`** exists for one reason and it should be stated: it preserves co-reference, so the
  same address is the same token in every passage and the model can still tell that two
  mentions are one person. **It must be salted with a per-installation secret that never
  leaves the machine.** An unsalted digest of an email address is reversible by anyone with a
  word list, and a truncated one collides; sending a hash instead of the value would then be a
  privacy theatre that costs answer quality and buys nothing.
- **`remove`** deletes the span. Cheapest to reason about, worst for the model, since it
  removes the evidence that anything was there.

**Detectors are recall-oriented and will fire on things that are not personal data** — a
version string that looks like a phone number, an internal identifier that looks like a card.
That is why this is off by default, and why enabling it is a decision with a stated cost rather
than a free precaution.

**A regex over 32k tokens of context on every query, with operator-supplied patterns, is a
denial-of-service surface.** Python's `re` cannot be interrupted, so redaction runs in a worker
thread with a wall-clock deadline (`redaction_timeout_s`, default 5.0). **Exceeding it fails
the query.** The fail-safe direction is refuse-to-send; there is no path in this design where a
timeout, an exception or a mistake results in unredacted text going to a remote model. Custom
patterns are the operator's risk, catastrophic backtracking is the named hazard, and the
built-in detectors are the supported surface.

### 7.4 Redaction never touches what a citation is verified against

The interaction the ticket flags, resolved by construction rather than by care.

**Redaction is a projection applied to a copy on its way out.** The verification chain —
`Chunk.text` ↔ `Anchor` ↔ retained original bytes — never sees a redacted string. Verification
compares the resolved source text against `Chunk.text` (§3.8), both unredacted, both
authoritative. If verification ran against what was *sent*, every redacted passage would fail
its round trip and the feature would delete every citation it touched.

Three consequences worth writing down:

**A citation can point at a span containing text the model never saw.** The prompt said
`[REDACTED]`; the citation resolves to the original address. This is correct and it is the
feature working: `ingest.md` §3.4 chose redaction to control *what leaves the machine*, not to
hide data from the person who indexed it and already has read access to the corpus. Anyone
expecting the source preview to be redacted is expecting the ingest-time feature that was
rejected — and would be getting a false sense of it, since the unredacted bytes are in the blob
store regardless.

**A shared conversation is a different audience, and it is handled separately.** §11.3.

**Offsets computed against redacted text are meaningless against the original.** `replace` and
`remove` both change lengths, so a character span the model produced over the redacted passage
does not address the same characters of the source. Nothing today is exposed to this, because
citations bind to whole passages and `Anchor` addresses a whole chunk — but it is the concrete
reason quote-level highlighting is not a free extension, and the reason it is listed in §16
with a condition rather than as a nice-to-have.

### 7.5 `local_only` sources drop passages; they do not refuse the query

`SourceRestrictions.local_only` and `cloud_allowed`, and `WorkspaceOverride.cloud_allowed`, are
declared in `manicule.config.settings` today and nothing reads them. This document gives them
their meaning.

> **Prior art.** `sourceRestrictions: { localOnly: [], cloudAllowed: [] }` is declared in the
> config schema and in the defaults, and I could find no code that reads either array. A
> policy field nothing enforces is `contracts.md` §5's plugin `permissions` field wearing a
> different name.

**A passage from a `local_only` source may not be sent to a remote model.** When the egress
class is `REMOTE`, such passages are removed from the context before the prompt is built, and
the removal is surfaced in band exactly like a dropped citation.

The alternative — refusing the whole query — is worse, and the reason is proportionality: it
makes the mere *existence* of one restricted document break unrelated questions that happened
to retrieve it at rank 7. Dropping the passage answers the question from what policy permits
and says what it could not use. Search still shows the document, because search is local and
only generation crosses the boundary, so a user learns the document exists and that its content
did not leave. They already had read access; nothing is disclosed that was not.

Three rules to keep it coherent:

- **Policy filtering only removes.** It never reorders, never adds, and never re-runs assembly.
  Re-assembling to backfill the freed budget would make the context a function of which model
  you asked, and two runs that saw different passages are not comparable.
- **A source restriction is a floor, not a default.** `WorkspaceOverride.cloud_allowed = true`
  does not release a `local_only` source. The narrower rule wins, because the broader one is
  the one somebody sets for convenience.
- **If every passage is dropped by policy, the answer is generated from an empty context and
  says so.** It is not silently answered from the model's own knowledge dressed in the shape of
  a corpus answer.

### 7.6 What redaction costs, stated once

It degrades answers. A model that cannot see the address cannot answer a question about the
address, and detectors that fire on non-personal data remove information the answer needed.
That cost is why the feature is off by default and why `scope: remote` is the default when it
is on — a local deployment should not pay a privacy cost for a threat it does not have.

This is the same shape of statement `ingest.md` §3.4 made about the ingest-time version. The
difference is that here the cost is paid per request, is reversible by changing a setting, and
leaves nothing on disk to regret.

---

## 8. Conversation memory

### 8.1 Whole turns, in pairs, newest first

History is a list of `Message` rows read from the store — role, content, timestamp — not a
flattened blob.

> **Prior art.** `getConversationHistory` takes `messages.slice(-6)`, applies
> `.content.substring(0, 500)` to each, and joins them into one string with `User:`/`Assistant:`
> prefixes. Two hard truncations before any budgeting happens, and `substring(0, 500)` cuts
> mid-word and mid-Markdown — an assistant turn's citations are usually severed. The result is
> then re-trimmed line-by-line against a token budget, with `estimateTokens` re-run per word
> for the last line.

The rules:

- **Whole turns only.** A half message misrepresents what was said, in the same way a trimmed
  passage misrepresents a source (`retrieval.md` §7.3). If a turn does not fit, it is dropped.
- **Turns are dropped in user/assistant pairs.** Keeping an assistant turn whose question is
  gone leaves the model an answer to something it cannot see, which is worse than having
  neither — it invites the model to infer the missing question.
- **Newest first; oldest dropped.** No summarisation. A rolling summary is a generated artefact
  that then gets treated as a record of what was said, and it costs a model call per turn.
- **The current user turn is never dropped.** It is the question. If it alone does not fit
  `history_tokens` plus the question's own room, that is a refusal with the numbers named, not a
  truncation.
- **Measured with the generator's tokenizer** (§9), never the embedder's.

### 8.2 History does not borrow from context, and context does not borrow from history

`context_tokens` and `history_tokens` are separate budgets and neither lends to the other.

A shared pool sounds strictly better and is not: a long conversation would silently starve
retrieval, so the tenth turn of a chat gets fewer passages than the first for the same question,
and the answer gets worse for a reason invisible to the user and to
[#15](https://github.com/mgd43b/manicule/issues/15). Fixed budgets mean a long conversation
loses old turns — which the user can see, and which is the thing they would have chosen to lose.

Both budgets, plus the system prompt and the generation reserve, are the §4.3 startup
cross-check. That check is what makes fixed budgets safe: the arithmetic is verified once
against the real window rather than hoped about per query.

### 8.3 Markers in history are neutralised, never re-bound

**Slot numbers are per-answer.** Turn 1's `[[cite:3]]` referred to turn 1's third passage; turn
4 has an entirely different context and its slot 3 is a different document. Feeding turn 1's
answer back verbatim gives the model marker syntax that binds to something else, and the model
copies the pattern.

So markers in history are rewritten to a **neutral, non-bindable textual reference** before the
history is sent — `[cited: "Deploy runbook" § Rollback]` — carrying enough for the model to
follow the conversation and nothing the binder will act on. They are never re-verified and never
re-bound; the citation records for prior turns already exist in `messages.sources` and are what a
reader sees.

This is the one place the answer text is transformed, and it is worth being precise that it does
not contradict §3.4: the transformation is applied to a **copy on its way into a prompt**, exactly
like redaction, and the stored answer is untouched. Nothing a reader sees changes.

### 8.4 History does not condition retrieval, and that is visible

`retrieval.md` §10.2 settles that conversation history is not in the L1 cache key, because
nothing in the retrieval pipeline reads history. The consequence belongs here: **a follow-up
retrieves on its own text alone.** "And what about the second one?" retrieves for that sentence,
which matches nothing useful.

This is a real limitation and it is not papered over. The fix is history-conditioned query
rewriting, deferred in `retrieval.md` §13 with its measurement, and `retrieval.md` §10.2 already
specifies that history joins the cache key in the same commit that ships it. Until then the
answer's trace records that the retrieval saw no history, so a bad follow-up is diagnosable
rather than mysterious.

---

## 9. Token counting

### 9.1 Three tokenizers, and only one of them counts here

`retrieval.md` §7.2 names two budgets measured with different tokenizers and calls using one for
the other a category error. With a reranker there are three vocabularies in play:

| Quantity | Tokenizer | Owner |
|---|---|---|
| Chunk size, 512 | The embedder's — XLM-RoBERTa SentencePiece | `parsing.md` §1.2 |
| Reranker input | The cross-encoder's | `retrieval.md` §6.2 |
| Context window, history budget, generation reserve | **The generation model's** | here |

**`Chunk.token_count` must never be used for a generation budget.** It is sitting on every
candidate, it is a plausible number, and it is measured in the wrong units for a model that is
not generating anything. It is wrong by an unknown factor that varies by language and by
content type.

### 9.2 The estimate, and the one measurement that replaces it

The estimate follows `retrieval.md` §7.2 exactly and this document adds nothing to it:
`tiktoken.get_encoding("o200k_base")` by encoding name rather than
`encoding_for_model("gpt-4o")` — naming a model that is not being used makes the estimate look
authoritative — with a per-model safety factor biased toward overcounting, no sampling, and
counts cached by content-derived `chunk.id`.

> **Prior art.** A module-level singleton hardcoded to `encodingForModel('gpt-4o')`, with no
> parameter and no configuration. The defaults it measures for are `qwen2.5:14b`,
> `claude-sonnet-4`, `gemini-2.5-flash` and `grok-3` — four different tokenizers, none of them
> `o200k_base`. On failure it degrades silently to a character ratio the docstring self-reports
> as "~85% accuracy for English, ~70% for Korean", for a project that ships a Korean README.
> Above 10 000 characters it samples 4 000 from the start and 4 000 from the middle and
> extrapolates — **the end is never sampled**, so a document with a large table or code block
> at the tail is systematically undercounted, in the direction that overruns the window. A
> third heuristic (`charsPerToken` of 4 or 1.5 by CJK ratio) then decides truncation length, so
> the truncation and the count disagree by construction.

**What #7 owns is turning the estimate into a measurement.** The true prompt token count comes
back from the provider as `Usage.prompt_tokens` on the final `Token` — Ollama's
`prompt_eval_count`, and the equivalent from every hosted provider — which is why §4.10 makes
`stream_options` non-optional. The estimate is compared against it after every call and the
drift is recorded per model.

The comparison also catches the §4.3 truncation: a server that trimmed the prompt reports a true
count pinned at its window, which is a different signature from ordinary tokenizer drift and is
reported as such.

### 9.3 Drift is reported, never auto-tuned

The tempting move is to feed measured drift back into the safety factor. It is refused, twice
over.

**An estimator that adapts silently makes two runs non-comparable.** The same query with the
same profile fits a different number of passages depending on what the process has seen since it
started. That is the property `retrieval.md` §11.2 protects, arriving through a side door.

**It can adapt in the unsafe direction.** A run of short English answers lowers the factor; the
first long CJK or code-heavy prompt then overflows a window that a fixed factor would have
respected.

So drift beyond tolerance is an **error-level event with both numbers and the model named**, and
`doctor` reports the observed distribution and recommends a factor. The human changes the
setting. `CONTRIBUTING.md`'s "configuration is declarative" applies to the values a system uses
to protect itself, not only to the ones an operator types.

**A provider that reports no usage after being asked** is recorded as `usage_unavailable`, which
is distinct from "the estimate was correct". Treating silence as agreement is how a calibration
loop reports success forever without ever running.

### 9.4 `max_tokens` is the generation reserve — one number, not two

`LlmSettings.max_tokens` is what the model is allowed to produce **and** the `generation_reserve`
term in §4.3's cross-check. Deliberately the same number.

> **Prior art.** `context-window.ts` reserves 10–15% of the window as a `generationBuffer`, and
> `generateAnswer` then requests a hardcoded `maxTokens: 1024` with no reference to it. On
> `fast`, 8192 × 0.1 = 819 tokens are reserved and 1024 are requested. A test pins the
> hardcoded value in place. Two numbers for one quantity, and they disagree by default.

A `FinishReason.LENGTH` therefore means exactly one thing — the answer hit the budget that was
reserved for it — and the answer is labelled truncated, as the enum's docstring already
requires.

---

## 10. Confidence, and citation accounting

### 10.1 Two numbers, never blended

The ticket asks for "confidence surfaced to the caller".
[#6](https://github.com/mgd43b/manicule/issues/6) already defines `Confidence`, and
`retrieval.md` §8.1 is emphatic about what it is: a statement about **retrieval support**,
computed before generation, uncalibrated, not a probability, not about the answer, and not
comparable across pipeline configurations.

#7 surfaces that number **unchanged**, alongside a second and entirely separate one:

| | `Confidence` (#6) | Citation accounting (#7) |
|---|---|---|
| Question | How well-supported is this by the corpus? | Did the citations on this answer verify? |
| Computed | Before generation, from the retrieval | During and after generation |
| Scale | A weighted score with bands, comparable only within one pipeline identity | Counts and reasons. No score |
| Calibrated | No, and says so | Not a probability at all |

**They are never combined into one figure.** Blending a retrieval-support score with a citation
count produces a number that answers neither question, and it would destroy #6's rule that
confidence is comparable only within a pipeline identity — because the blend would also move
with the generation model.

> **Prior art.** `calculateConfidence` is called with `rerankScores: []` always, so the
> `avgRerank` term falls back to `avgRetrieval` and 70% of the number is one variable counted
> twice. The scores it averages have been overwritten by RRF (~0.016), multiplied by a metadata
> boost of up to 1.25, and dragged down by sibling chunks injected at `score * 0.6`. Then the
> UI discards it: `ChatMessage.tsx` displays `max(...)` of the top-4 source scores relabelled
> "Evidence match" whenever any source exists, rendered as `Math.round(metricScore * 100)%`
> **unclamped** — so it can exceed 100%, and under RRF it is normally about 2%. The number
> persisted to `query_logs` is the engine's; the number the user saw is neither.

### 10.2 What the caller receives

```
AnswerEnvelope
    confidence:        Confidence | None      # #6's, verbatim, or absent
    citations:         tuple[Citation, ...]   # verified, in slot order
    dropped:           tuple[CitationDrop, ...]  # slot, reason, level reached
    verification_level: Verification          # the strongest level available this run
    ungrounded:        bool                   # context was non-empty, zero citations survived
    context_truncated: bool                   # Context.truncated, forwarded
    policy_dropped:    tuple[PolicyDrop, ...] # §7.5
    finish_reason:     FinishReason
    usage:             Usage | None
```

**An interface must not render a single blended percentage from these.** That is not a
suggestion about visual design; it is the requirement that makes both numbers mean anything,
and it is the specific failure recorded above.

### 10.3 Generation never modifies confidence

Not to lower it when citations are dropped, not to raise it when they all verify. `Confidence`
is a statement about retrieval, and retrieval did not change because the model behaved badly.
Making generation write to it would mean two answers from the same retrieval carry different
confidences, which makes the number unusable for the thing it exists for.

The relationship is the other way round: a high confidence with an `ungrounded` answer is a
meaningful and diagnosable combination — the evidence was there and the model did not use it —
and it is only visible because the two numbers stayed separate.

### 10.4 The direct route keeps its promises

`retrieval.md` §9.3 settles the router bypass: an answer that never consulted the corpus carries
**no citations**, states that the corpus was not consulted, and has confidence **absent** —
not 1.0 and not 0.0.

#7 honours all three, and adds the one that only exists at this layer: **a directly-routed
answer cannot acquire citations.** With an empty `Context` there are no slots, so every marker
the model emits fails at level 0 and is deleted. `ungrounded` is *not* set, because it means
"the context was non-empty and nothing survived", and this context was empty by design.

> **Prior art.** `handleDirect` hardcodes `{ score: 1, level: 'high', reason: 'Direct
> response' }` for a canned greeting — the highest confidence the system can express, for an
> answer that consulted nothing.

---

## 11. Shared conversation links

The schema already carries `conversations.shared` and `conversations.share_token UNIQUE`
([`storage.md`](storage.md) §4.7). What the feature *does* is settled here, and the starting
point is that the obvious implementation is an exfiltration primitive.

> **Prior art, in full, because every part of it matters.** `GET /shared/:token` is registered
> **before** the auth middleware and before both rate limiters, so it is unauthenticated and
> un-rate-limited — a test asserts the former as intended behaviour. It resolves the workspace
> **from the row**, so the token alone is the entire authorization decision. It does
> `SELECT *`, returning `workspace_id` and the token itself to an anonymous caller, and it has
> no `deleted_at IS NULL` predicate — uniquely among every query in that file — so soft-deleting
> a conversation does not revoke its link. There is no unshare endpoint, no `shared = 0` write,
> no expiry column and no expiry check anywhere in the repository: once shared, permanently
> public. Messages added after sharing are included automatically, so the link is a live view.
> And `getMessages` returns `sources` as stored, which is the full serialised `SearchResult[]`
> **including `content`** — verbatim chunk text from private indexed documents, rendered in the
> viewer's source-preview modal. Creating one requires only the `ask` scope, so anyone who can
> chat can publish. It is not a transcript link; it is a public read endpoint over the corpus.

### 11.1 The link is a bearer capability, and is treated like one

- `secrets.token_urlsafe(32)` — 256 bits. Compared in constant time.
- **Stored hashed**, like `api_keys.key_hash`, and shown to its creator exactly once. The
  argument is not that the token protects the row from someone holding the database — that
  person has the conversation anyway. It is that a share token is a **live credential for an
  unauthenticated URL**, and the database is backed up, exported and imported
  ([`storage.md`](storage.md) §9, `PLAN.md` §15). Plaintext tokens travel into those artefacts
  and out of the access boundary that created them.
- The public route is rate-limited like every other route, and served with `X-Robots-Tag:
  noindex`. An unauthenticated URL that search engines crawl is a different feature from the one
  anyone asked for.
- Creating a share is an audited event (`audit_logs`), because publishing internal content is
  exactly the class of action an audit log exists for.

### 11.2 A snapshot, not a live view

Sharing captures the conversation **as it stands at that moment**. Later turns are not exposed.

The live version has a failure that is obvious once stated and invisible while building it:
somebody shares a conversation after turn 2 and continues using it, and turn 7 is public the
moment it is written. Nobody re-reads a link they already sent. A snapshot means the thing that
was shared is the thing that was reviewed.

Re-sharing after further turns is an explicit new act, producing a new snapshot.

### 11.3 What an anonymous viewer sees, and the tension this resolves

This is the hard part, because two of this document's own commitments pull against each other:
citations are the product and should be checkable, and passage text is corpus content that a
person with no workspace membership must not receive.

**Resolution: a shared conversation shows the questions, the answers, and the citation
*labels* — document title, heading path, page — together with the verification state recorded
when the answer was generated. It does not show passage text, and its citations do not link into
the corpus.** An authenticated viewer with access to the workspace opens the same conversation
and sees the passages, because they could have retrieved them anyway.

So the *same message renders differently by audience, and the difference is content only* —
never the existence of a citation, never its label, never whether it verified. The anonymous
viewer is told "this claim was verified against 'Deploy runbook' § Rollback at generation time"
and cannot read the runbook. That is a weaker guarantee than checking it themselves, and it is
honestly labelled as an attestation rather than dressed up as a link they could follow.

Sharing is an explicit act whose confirmation states exactly what becomes public, in those
terms. And because a document *title* can itself be sensitive, team mode can disable sharing
entirely — one switch, in `security`, rather than a per-field disclosure policy nobody will
configure correctly.

### 11.4 Revocation and expiry are not optional

- **Revocation exists**, as a real endpoint, and clears the hash rather than flipping a boolean
  beside a still-valid token.
- **Soft-deleting the conversation revokes the link.** The public read applies
  `deleted_at IS NULL`, like every other read in the system.
- **Links expire.** `share_expires_at`, defaulting to 30 days, set at creation and enforced on
  every read. A capability with no expiry accumulates forever, and the set of live ones becomes
  unknowable.
- **Both are visible to the owner**: which conversations are shared, when each link expires,
  and one action to revoke.

These are additive columns on a merged table and are filed as
[#42](https://github.com/mgd43b/manicule/issues/42) rather than done here.

---

## 12. Feedback

### 12.1 It attaches to a message

A user rates an **answer**. An answer is a `Message`. So feedback attaches to a message id, not
to a `query_logs` row.

That is not bookkeeping preference. `query_logs` is retrieval telemetry — one row per retrieval
run — and there are answers with no retrieval behind them (the direct route, §10.4) and answers
whose retrieval succeeded and whose generation failed (§4.7). Both are ratable and neither has a
usable query-log row. Meanwhile a message always exists, including for a partial answer, because
§5.3 guarantees it.

`messages` gains `feedback`, `feedback_reason` and `query_log_id`, and the existing
`query_logs.feedback` column loses its job — additive migration, filed as
[#42](https://github.com/mgd43b/manicule/issues/42) with the sharing columns.

Two small behaviours, both corrections:

- **An unknown or foreign message id is a 404.** Prior art returns `{ saved: true }` without
  checking whether any row matched, so feedback on a mistyped id silently succeeds and is
  reported as saved.
- **The value is validated against the enum.** Prior art types it `'positive' | 'negative'` in
  TypeScript and writes whatever string arrives to the database.

### 12.2 What it is for, and what it must never become

**For:** finding queries that fail, so they can be promoted into
[#15](https://github.com/mgd43b/manicule/issues/15)'s fixed query set as labelled cases; and
operational alerting on a rising negative rate.

**Never:** an automatic input to retrieval, ranking, caching or prompting. A pipeline whose
behaviour depends on accumulated user feedback cannot be compared across runs — the same
configuration produces different results in week 2 than in week 1 — and that comparison is the
entire method `retrieval.md` §2.4 and §11.2 exist to protect. Feedback informs a human who
changes a configuration, or a fixture that a harness measures against. It does not close a loop
on its own.

For the first purpose the record has to be sufficient to reconstruct the run, so it carries the
message id, the retrieval trace identity (pipeline, profile, reranker `model_id`, embedding
fingerprint), the generation identity (model, egress class, redaction scope), and the citation
accounting. Feedback that cannot name what produced the answer is a mood, not a datum.

> **Prior art.** One nullable `query_logs.feedback` column, binary, no comment field, no user
> attribution, destructively upserted with no history. The dashboard counts positives; nothing
> else reads it. The `'query:feedback'` event is declared on the event bus and never emitted
> anywhere. And the `intent` column those rows carry is hardcoded to `'general'` at the insert,
> though `classifyIntent` runs and its result is even streamed to the client — so the admin
> dashboard's intent distribution is always a single bar.

### 12.3 The one category that matters

Feedback is `positive | negative` plus an optional reason from a small closed vocabulary:

```
wrong · incomplete · citation-wrong · too-slow · other (+ free text)
```

**`citation-wrong` is why the vocabulary exists.** §3.5 establishes that verification cannot
catch misattribution: a citation that resolves perfectly and supports nothing in the sentence
it is attached to. No check in this system will ever fire on that, because firing on it means
deciding entailment.

A human reading the answer *can* see it. So `citation-wrong` is the only detector this project
has for its one uncaught citation failure, and the reports it produces are the labelled set that
would let the deferred hallucination guard (`retrieval.md` §13) be measured for precision *and*
recall rather than shipped on faith.

It is also, in the meantime, the first answer-side quality signal the project has at all —
`retrieval.md` §13 notes that intent classification cannot be evaluated because "#15 does not
have an answer-quality metric". Citation accounting (§10.2) is objective and needs no labels;
`citation-wrong` is the labelled complement to it.

---

## 13. No answer cache

`retrieval.md` §10.2 records that caching a generated answer "is a different feature with
different invalidation, and it belongs to #7 if it belongs anywhere". **It does not belong.**

- **Generation is sampled.** `temperature` defaults to 0.2, so a cache turns one sample into
  the canonical answer for everyone who asks that question next. Setting it to 0 does not fix
  this — it narrows the distribution and does not make batched GPU inference deterministic.
- **Invalidation would have to cover things the generation counter knows nothing about**:
  conversation history, egress class, redaction scope and patterns, the generator model and its
  `max_tokens`, and the system prompt. A key omitting any of them serves an answer computed
  under different rules.
- **The expensive half is already cached.** L1 caches the retrieval decision
  (`retrieval.md` §10), which is the embedding forward pass and possibly a cross-encoder. What
  an answer cache would additionally save is one model call — the part a user is watching stream
  and the part most likely to be worth re-sampling.
- **The failure mode is documented above**: prior art caches the generation-failure placeholder
  for five minutes with a real confidence score and a real source list attached.

The one configuration that would justify revisiting: a public or demo deployment answering the
same small set of questions repeatedly at temperature 0, where the saving is real and the
sampling objection is weakest. That is a deployment shape this project does not have, and it
would need its own key covering the list above.

---

## 14. The generation trace

One `GenerationTrace` per answer, alongside `retrieval.md` §11's `RetrievalTrace`, and a return
value for the same reasons.

| Scope | Fields |
|---|---|
| Model | resolved model string, resolved endpoint, egress class, `context_window`, `num_ctx` sent |
| Budget | estimated prompt tokens, true prompt tokens, drift, encoding name, safety factor, completion tokens |
| Timing | first-token latency, total latency, retries and why, finish reason |
| Citations | slots offered, markers seen, verified, dropped with reasons, verification level, cache hits |
| Policy | redaction scope, which detectors fired and how many times, passages dropped by `local_only` |
| History | turns offered, turns sent, turns dropped, history tokens |

**The trace never contains document text, query text, or matched values.** Recording which
detector fired and how often is diagnostic; recording what it matched turns the trace into the
leak the detector existed to prevent. This is the same instinct as
`AtRestSettings.redact_logs_content`, applied to a structure that did not exist when that
setting was written.

It sits beside the retrieval trace and does not go into `query_logs`, for the reason
`retrieval.md` §11.2 gives about the retrieval trace: that table is whole-query product
telemetry, and per-call detail there is a schema change in service of a consumer that keeps its
results elsewhere.

---

## 15. Apple hardware

The standing rule is `PLAN.md` §7's: **optimise execution for Apple hardware freely; never let
the platform change what ends up in the index.** Generation needs that restated, because the
naive reading of it is impossible here.

**The same prompt to the same model on different hardware produces different text.** That is
inherent to sampling, and no platform discipline changes it. So the invariant for generation is
not the tokens:

> Platform may change throughput, latency and wording. It may not change **which citations
> survive**, **what is redacted**, or **what is persisted**.

Every one of those three is computed by manicule, in Python, from data that does not depend on
the accelerator: verification is a parse and a string comparison, redaction is a regex pass, and
persistence is a database write. A Mac and a Linux box answering the same question give
different prose and identical citation guarantees. That is the property worth having, and it is
testable — unlike text equality, which is not.

Three concrete Apple-specific notes, all throughput:

- **`keep_alive`.** Ollama unloads an idle model after five minutes, and the next question then
  pays a multi-second load from disk. For interactive use that is the single largest avoidable
  latency, so `keep_alive` is set (default `10m`) and is a pure throughput knob: it changes
  nothing about any answer.
- **`num_ctx` costs unified memory.** The KV cache scales with the window, and §4.3 requires
  manicule to *demand* a window large enough for the profile. On a 16 GB Mac, a 14B model at
  4-bit plus a 36k-token KV cache is not a configuration that runs well, and `precise` on the
  default model does not pass the cross-check anyway (§4.3). The honest position: the startup
  refusal is the right place to find that out, and `doctor` reports the memory implication of
  the window it is about to demand.
- **Local generation and local embedding compete.** The embedder is in-process MLX
  (`PLAN.md` §7) and Ollama is a separate process; a large ingest running during interactive
  chat contends for the same GPU and unified memory. This is a scheduling observation rather
  than a design change — but it is why "bulk work local, interactive work wherever is best"
  (§4.2) is a real deployment consideration and not just a cost one.

---

## 16. Deferred, with the condition that would un-defer each

| Feature | What would have to be true |
|---|---|
| **Quote-level citation** — an anchor into a span within a passage | Requires `Anchor` to address a sub-chunk span, which is a ⚠️ locked type, *and* an answer to §7.4's offset problem, since redaction changes lengths. Both, or neither |
| **Answer cache** | A deployment shape with repeated identical questions at temperature 0, plus a key covering history, egress class, redaction settings, model and prompt (§13) |
| **History-conditioned query rewriting** | Owned by `retrieval.md` §13. Ships with history joining the L1 cache key in the same commit |
| **Hallucination guard** | `retrieval.md` §13's measurement: precision *and* recall against a labelled set. §12.3's `citation-wrong` reports are how that set gets built |
| **A second generation model for bulk work** | A bulk generation workload that actually exists. Today the bulk path is embedding, which is local by construction (§4.2) |
| **Streaming answer edits** (retracting or annotating already-delivered text) | Nothing. It is refused, not deferred (§3.4) |

---

## Appendix A: decisions this document made

| Decision | Where |
|---|---|
| The model selects a slot; every citation field is built from `Context`, none from the model | §3.1 |
| `[[cite:N]]`, chosen against `[1]`, `[[N]]` and `[^N]` for collision with code and prose | §3.2 |
| The binder only ever deletes, plus normalising syntax it defined itself | §3.2 |
| Marker syntax inside a passage is escaped before rendering | §3.2 |
| Verification is a three-level ladder, and the level reached is reported per citation | §3.3 |
| **A failed citation is dropped; the answer is never refused, rewritten or trimmed** | §3.4 |
| All citations dropped with a non-empty context flags the answer `ungrounded` | §3.4 |
| Misattribution is out of reach and the guarantee is stated narrowly rather than widened | §3.5 |
| Verification is per document, cached on `(chunk_id, version_token)`, started before the first token | §3.7 |
| A verification timeout is a drop with its own reason, not a bypass | §3.7 |
| The runtime check reuses the parser suite's containment predicate rather than a second one | §3.8 |
| The binder and verification sit above `Generator` and are not pluggable | §2.2 |
| `Generator` needs `context_window`; filed rather than edited into `contracts.md` | §2.3 |
| Ollama is reached through `ollama_chat/`, because `ollama/` skips the chat template | §4.2 |
| One generation model; the bulk/interactive split is satisfied by the embedder being local | §4.2 |
| `context_window` means the served window; `num_ctx` is derived and sent explicitly | §4.3 |
| The shipped `precise` profile does not fit the default model and is refused at startup | §4.3 |
| Three timeouts — first token, inter-token idle, total — because one covers the wrong interval | §4.5 |
| Retries stop at the first token; litellm's `num_retries` is not used for streaming | §4.6 |
| A mid-stream failure is in band, and the partial answer is persisted and ratable | §4.7 |
| litellm exceptions are mapped at the adapter; provider message text is preserved | §4.8 |
| `stream_options={"include_usage": True}` is mandatory, or the calibration never runs | §4.10 |
| `generating()` mirrors `parsing()`; cleanup is bounded, never yields, and covers both exits | §5.1, §5.2 |
| Persistence lives in the wrapper's `finally` and is shielded against the cancellation | §5.3 |
| System / turns / passages-then-question, with the question last so truncation loses a passage | §6.1 |
| The citation protocol section of the system prompt is not configurable | §6.1 |
| The label carries the breadcrumb; the body is `Chunk.text`; no chunk ids in the prompt | §6.2 |
| Egress is classified by resolved endpoint, never by provider name, with the limits stated | §7.1 |
| Query and history are redacted on egress; the answer and the index are not | §7.2 |
| `RedactionSettings.scope` (`remote` default, `always` available) | §7.2 |
| Detectors are named and versioned; a custom pattern that does not compile is a startup refusal | §7.3 |
| `hash` must be salted per installation, and exists only to preserve co-reference | §7.3 |
| Redaction runs under a deadline, and exceeding it fails the query rather than sending plaintext | §7.3 |
| Verification runs against unredacted `Chunk.text`; redaction never touches the citation chain | §7.4 |
| A citation may point at a span the model never saw, and that is the feature working | §7.4 |
| `local_only` drops passages and surfaces the drop; it does not refuse the query | §7.5 |
| A source restriction is a floor that a workspace override cannot release | §7.5 |
| History is whole turns, in pairs, newest first, current turn never dropped | §8.1 |
| History and context budgets do not lend to each other | §8.2 |
| Markers in replayed history are neutralised to a non-bindable form, never re-bound | §8.3 |
| Drift between estimate and true count is reported, never auto-tuned | §9.3 |
| `max_tokens` is both the output cap and the reserve — one number | §9.4 |
| `Confidence` and citation accounting are surfaced side by side and never blended | §10.1 |
| Generation never writes to `Confidence` | §10.3 |
| A directly-routed answer cannot acquire citations, and is not `ungrounded` | §10.4 |
| Share tokens are hashed, expiring, revocable, audited, rate-limited and `noindex` | §11.1, §11.4 |
| A share is a snapshot, not a live view | §11.2 |
| An anonymous viewer gets citation labels and verification state, never passage text | §11.3 |
| Feedback attaches to a message, not to a query-log row | §12.1 |
| Feedback never closes a loop automatically; it feeds #15 and an operator | §12.2 |
| `citation-wrong` exists because it is the only detector for misattribution | §12.3 |
| No answer cache, with the one configuration that would revisit it named | §13 |
| The generation trace records which detectors fired, never what they matched | §14 |
| The Apple-hardware invariant for generation is the citation guarantee, not the text | §15 |

## Appendix B: what the merged documents did not cover

Places this design had to decide something no merged document had a position on.

- **How a citation is produced at all.** Every merged document specifies what a citation must
  satisfy — `contracts.md` §1's round trip, `parsing.md`'s anchors, `retrieval.md` §7.3's
  whole-passage rule, PR #32's ban on rewriting cited text — and none of them says how the text
  of an answer comes to carry one. §3.1 settles it as slot selection, which is what makes the
  rest of the guarantees reachable from generation.
- **`Generator` cannot answer the question `retrieval.md` §7.4 asks it.** The startup
  cross-check needs a context window and the protocol exposes none (§2.3,
  [#41](https://github.com/mgd43b/manicule/issues/41)).
- **The shipped `precise` profile does not fit the shipped default model** (§4.3). `retrieval.md`
  §7.4 predicted that a profile could exceed a window and required a refusal; nobody had done the
  arithmetic against the defaults that are actually in `manicule.config`.
- **`SourceRestrictions.local_only`, `SourceRestrictions.cloud_allowed` and
  `WorkspaceOverride.cloud_allowed` had no defined behaviour.** They are declared in merged
  configuration and nothing reads them; §7.5 gives them one.
- **`RedactionSettings` had no scope, no detector registry and no failure semantics.** The
  section's docstring settles *where* redaction happens; §7.2 and §7.3 settle what it does, what
  it costs, and which direction it fails in.
- **What a shared link exposes.** The schema carries `shared` and `share_token`; no document said
  whether a link is authenticated, whether it expires, whether it can be revoked, or whether an
  anonymous viewer receives passage text (§11).
- **Where feedback lives.** `query_logs.feedback` is the merged column and it is the wrong home
  for the reasons in §12.1.
- **Whether generated answers are cached.** `retrieval.md` §10.2 explicitly passed the question
  here; §13 answers it as no, with the condition that would reopen it.
- **What "citations verified resolvable" excludes.** The ticket's phrase reads as a guarantee
  about correctness; §3.5 states the narrower thing the system can actually enforce, because a
  guarantee that overreaches gets believed.

## Appendix C: filed, not deferred

| Ticket | What | Why not here |
|---|---|---|
| [#41](https://github.com/mgd43b/manicule/issues/41) | **Add `context_window` to the `Generator` protocol** (§2.3, §4.3) | It changes `docs/contracts.md` and `manicule.core.protocols`, which three implementation tickets are building against right now. Additive, and neither `Anchor` nor `RetrievalStage`, so it carries no lock — but it is a seam change and belongs in its own reviewable commit |
| [#42](https://github.com/mgd43b/manicule/issues/42) | **Conversation schema for sharing and feedback** (§11.1, §11.4, §12.1) — hash `share_token`, add `share_expires_at` and `shared_at`, add `messages.feedback`, `messages.feedback_reason`, `messages.query_log_id`, `messages.finish_reason` | It is an additive Alembic migration against tables merged under [#2](https://github.com/mgd43b/manicule/issues/2), and [#11](https://github.com/mgd43b/manicule/issues/11) and [#13](https://github.com/mgd43b/manicule/issues/13) both depend on the result. Worth landing as its own change for the same reason [#36](https://github.com/mgd43b/manicule/issues/36) was: it moves a security boundary |

[#28](https://github.com/mgd43b/manicule/issues/28) — refuse-to-ingest rules — stays where
`ingest.md` §3.4 filed it. It is the coherent version of "this content must never be stored",
which is a different requirement from "this content must not leave the machine", and this
document implements only the second.

## Appendix D: checklist against ticket #7

- **Generation behind the protocol from #1** — §2.2. `Generator` is unchanged apart from the
  filed `context_window` addition, and everything that guarantees a citation sits deliberately
  outside it so no plugin can omit it.
- **Streaming** — §5, with the #35 async-generator lifecycle applied: `generating()` closes on
  every path, cleanup is bounded and never yields, and both `GeneratorExit` and
  `CancelledError` release the provider connection.
- **Every citation verified resolvable — correct, or absent** — §3, with the three-level ladder
  (§3.3), the drop rule (§3.4), the honest exclusion (§3.5), and one predicate shared with the
  parser conformance suite (§3.8).
- **Confidence surfaced to the caller** — §10. #6's number verbatim, beside citation accounting,
  never blended, never written to by generation.
- **Multi-turn conversation memory** — §8. Whole turns in pairs, a budget that does not lend,
  and markers neutralised so a replayed answer cannot bind to a new context.
- **Shared conversation links** — §11. A hashed, expiring, revocable, audited capability over a
  snapshot, showing labels and verification state rather than passage text.
- **Feedback capture** — §12. Attached to a message, sufficient to reconstruct the run, never an
  automatic input to the pipeline, and carrying the one category that detects what verification
  cannot.
- **Local default on the Mac; hosted APIs available; bulk local, interactive wherever is best** —
  §4.1, §4.2, §4.9, §15. One litellm call, `base_url` and a model prefix as the only difference,
  Ollama optional at install, and the invariant across the choice stated as the citation
  guarantee rather than the text.
- **litellm as the single provider interface** — §4, including the `ollama_chat/` endpoint, the
  `num_ctx` window trap, three timeouts, the first-token retry boundary, mapped exceptions and
  mandatory `stream_options`.
- **PII redaction at the generation boundary** — §7, which is where `PLAN.md` defect #5 and
  `ingest.md` §3.4 left it and where it is now specified.
