# Ingest pipeline

Design for `discover → fetch → parse → chunk → embed → store`. Ticket
[#5](https://github.com/mgd43b/manicule/issues/5).

**This document owns enforcement, not rules.** What a parser does when it fails is settled in
[`parsing.md`](parsing.md) §6. What the stores guarantee under a crash is settled in
[`storage.md`](storage.md) §8. Which fingerprints exist and what they contain is settled in
[`parsing.md`](parsing.md) §1.7 and [`storage.md`](storage.md) §6.3. The pipeline is the thing
that has to make all of that actually happen on a real corpus, on a real machine, while a
document is hanging and another is trying to allocate 8 GiB.

So the interesting content here is the part nobody else can specify: **what happens when the
failure is not an exception.**

> **Prior art.** OpenDocuments appears in clearly-marked callouts like this where the
> comparison carries design information. Everything outside them stands on its own.

---

## 1. Division of labor

| Concern | Owned by | This document |
|---|---|---|
| Parser semantics, fallback outcomes, `Document.status` | `parsing.md` §6 | Runs the chain; enforces the limits it names |
| Chunk budget, `ChunkFingerprint` | `parsing.md` §1 | Checks it once, before any work |
| Write order, crash windows, invariants | `storage.md` §8 | Honors it; adds lease recovery |
| `EmbedFingerprint`, the refusal | `storage.md` §6.3 | Runs it once, before any embedding |
| Retained bytes, blob ordering | `storage.md` §7 | Writes them; re-parses from them |
| Deletion sweep, soft-delete grace | `storage.md` §8.2 | Drives `reconcile`; schedules the sweep |
| Connector protocol | `contracts.md` §3 | Calls it; never interprets its tokens |

Everything above is cited rather than restated. Where this document appears to disagree with
one of them, that is a bug in this document.

---

## 2. The shape

**The unit of work is one document. A batch is a scheduling artifact with no semantics of its
own.** This is what makes "one bad document never aborts a batch" a structural property rather
than a promise: there is no batch-level transaction to abort, and no batch-level state a
document can corrupt. It is also what makes the staged run in §8.3 possible without weakening
anything: several documents may be in flight because no two of them share any state.

| Stage | Input | Output | Where it runs | Bounded by |
|---|---|---|---|---|
| discover | `Watermark \| None` | `DiscoveredDoc` stream | async, in-process | connector page size |
| fetch | `DocRef` | `RawDocument` | async, in-process | HTTP timeout, size cap |
| parse | `RawDocument` | `ParsedBlock` stream | **worker subprocess** | wall clock + memory |
| chunk | `ParsedBlock` stream | `Chunk` list | in-process | token budget |
| embed | `Chunk` list | vectors | **in-process** | batch size |
| store | chunks + vectors | committed document | in-process | SQLite transaction |

One thing in that table is load-bearing and is argued in §6: **parse runs in a subprocess, and
embed does not.**

> **Corrected during implementation.** This table originally put *chunk* in the worker too.
> It cannot be there, and the reason is the middleware contract rather than anything about
> chunking: `after:parse` operates on blocks, middleware is a plugin the container builds, and
> a `spawn`ed worker would have to re-discover plugins and construct middleware that may hold
> network clients and per-run state. So blocks come back across the boundary and are chunked
> in the parent, which is also what §6.3's "workers receive bytes and return blocks" already
> said. The worker still owns the whole of what needs a deadline: the parser chain, and
> container expansion, which is a parser's work under the same limits.

### 2.1 Discover is not fetch

`Connector.discover` yields `DiscoveredDoc` — identity, `version_token`, and enough metadata to
decide whether to fetch at all. Fetching the body during discovery would make the change check
(§4) pointless, because the expensive part has already happened.

This is why `discover` takes a watermark and `fetch` takes a `DocRef`
(`contracts.md` §3): they are separately schedulable, and the gap between them is where the
skip decision lives.

### 2.2 `Document.status` is this pipeline's state machine

The enum belongs to [#1](https://github.com/mgd43b/manicule/issues/1), and both
[`storage.md`](storage.md) §4.2 and [`parsing.md`](parsing.md) §6.4 name the members they
depend on. Neither enumerates the whole set, because neither owns the transitions — **the
pipeline does**, and it is the only component that sees every one. Collected here so #1 has a
single place to read it from, since the value set is `CHECK`-constrained and extending it later
means an Alembic batch rebuild (`storage.md` §3.4).

```
        ┌──────────────── §6.4 sweep, from any in-flight state ───────────────┐
        │                                                                     │
        v                                                                     │
     pending ──> fetching ──> parsing ──> parsed ──> embedding ──> indexed    │
                     │            │                      │           │        │
                     └────────────┴──────────────────────┘           │        │
                                  │                                  │        │
                                  v                    (a failed re-ingest    │
                            failed + failed_stage       leaves this alone,    │
                            no_extractable_text         §9)                   │
                            unsupported_media_type                            │
                            container ────────────────────────────────────────┘
                                                        (never swept, §2.2)
```

| Member | Set by | Terminal | Servable |
|---|---|---|---|
| `pending` | discovery, or the §6.4 sweep | no | no |
| `fetching` `parsing` `embedding` | the pipeline, in flight — and only for a document with nothing servable to lose (§9) | no | no |
| `parsed` | parse stage, ≥ 1 chunk (`parsing.md` §6.4) | no | no |
| `indexed` | store stage, last write (`storage.md` §8.2 I3) | yes | **yes** |
| `failed` + `failed_stage` | **any** stage (`parsing.md` §6.4) | yes | no |
| `no_extractable_text` `unsupported_media_type` `container` | parse stage (`parsing.md` §6.4) | yes | no |

**Failure is one member plus a stage, not a member per stage.** `parsing.md` §6.4 carries
`failed` with a `failed_stage: PipelineStage` discriminator, which is the right shape and
resolves the concern that prompted this section: fetch, parse and middleware failures do not
each mint an enum member, so the `CHECK`-constrained set does not grow every time a stage is
added. This document contributes only `fetching` and `embedding` — two in-flight states — and
`failed_stage` values for the stages it owns.

> **Implemented as three, not two.** #1 shipped `DocumentStatus` without any of the in-flight
> states, so `fetching`, `parsing` **and** `embedding` all arrive here, along with
> `middleware` as a `failed_stage` value. Both columns are `VARCHAR` + `CHECK`, so widening
> them is the Alembic batch rebuild §3.4 said it would be — revision `9c1a4f7b2d10`, with a
> downgrade that maps in-flight back to `pending` and a middleware failure back to `parse`.
> `middleware` is placed after the six stages and documented as a hook boundary rather than a
> stage, so "the six, in order" stays true: a hook that raises is a plugin problem, and filing
> it under the stage it happened to bound would send an operator to read a parser that worked.

**Two terminal states are easy to sweep by mistake, and must not be.** `container` has zero
chunks *by design* — an archive whose members became their own documents has nothing of its own
to embed — and `no_extractable_text` has zero chunks because there was nothing to find. Both
look like "stopped before embedding" to a careless `WHERE` clause. The sweep in §6.4 therefore
selects the in-flight states **by name** rather than selecting everything that is not `indexed`,
which is the formulation that would swallow both:

```sql
WHERE status IN ('fetching', 'parsing', 'embedding')   -- never NOT IN (...)
```

An allowlist fails closed: a status added later is not swept until someone adds it. A denylist
fails open, and the failure is a terminal document being requeued forever.

**Change detection needs the mirror-image allowlist, and this document missed it.** A skip is
also a `WHERE`, and getting it wrong is just as quiet. A run interrupted mid-ingest leaves a
document requeued to `pending` — with its `version_token` and `content_hash` already written,
because both are recorded before the parse that failed. A skip rule consulting only the token
would then skip that document on every subsequent sync, forever, while every counter reported
it as handled. So `SETTLED` names the statuses in which a stored document is a finished answer
about the bytes it holds, and only those may skip. Both sets are in
`manicule.core.content`, and a test asserts they do not overlap and that no status is missing
from both — a status in neither is never requeued *and* never skipped.

**Only `indexed` is servable**, and that is enforced by the join in `storage.md` §6.2 rather
than by anything here. Every other state is invisible to retrieval whatever exists in the
derived stores, which is what makes an interrupted document harmless rather than wrong.

---

## 3. Middleware

Hooks at stage boundaries, from the plugin system in
[#1](https://github.com/mgd43b/manicule/issues/1):

```
before:fetch   after:fetch
before:parse   after:parse
before:chunk   after:chunk
before:embed   after:embed
before:store   after:store
```

### 3.1 A hook transforms, and the return value is the contract

```python
class Middleware(Protocol):
    stage: PipelineStage
    order: int
    mutates_embedded_text: bool          # §3.3

    async def run(self, value: T) -> T: ...
```

The pipeline **assigns the return value**. Not a suggestion — the single most important line in
the middleware implementation is `value = await hook.run(value)`.

> **Prior art, and the bug to not reproduce.** `MiddlewareRunner.run()` is written as a
> transform — it folds handlers over the value and returns the result — and every one of the
> four call sites discards what it returns: `await middleware.run('before:parse', rawDoc)`,
> then carries on using `rawDoc`. Middleware that mutates its argument in place works, because
> objects are passed by reference. Middleware that returns a new value is silently ignored.
> The system therefore behaves correctly often enough to look correct, which is the worst
> available outcome — the failure surfaces only for the subset of plugins written in the
> idiomatic style the signature advertises.

**A returned value is validated, not trusted.** A hook that returns the wrong type, or `None`,
fails the document with `failed` / `failed_stage=middleware` and names the hook. Silently substituting the
original input would resurrect exactly the bug above.

### 3.2 What a middleware may and may not do

| May | May not |
|---|---|
| Transform the value it is given | Hold a `DocStore` or `VectorStore` handle |
| Read configuration | Write to either store directly |
| Emit events and metrics | Mutate another document |
| Annotate `Document.metadata` / `Chunk.metadata` | Set or alter an `Anchor` |
| Skip a document by returning `None` from `before_parse` | **Alter `Chunk.text` — ever (§3.3)** |
| Rewrite `embed_text`, having declared it (§3.3) | Perform network I/O without a declared timeout |

> **Corrected during implementation.** This row said "raising `SkipDocument(reason)`". #1
> settled it the other way, in `manicule.core.protocols.Middleware`: `before_parse` returns
> `None` to drop a document. That is the better shape and the doc is what was wrong — a
> document excluded by configuration is an ordinary outcome, not an exception, and it records
> as `skipped` rather than as a failure. There is one short-circuit, not two.

**No store handle** is the important one. A middleware that writes to the stores bypasses the
write ordering in `storage.md` §8.2, and the ordering is what makes crash recovery possible.
Middleware operates on values in flight; the pipeline alone commits.

**No anchor mutation.** Anchors are locked once ingest runs (`contracts.md` §1) and a
middleware that adjusts them invalidates stored citations with no record of having done so.

**Failure semantics.** A middleware that raises fails *that document*, with status
`failed`, `failed_stage=middleware`, and the hook name in `metadata`. It does not abort the batch and it does not
disable the hook — a hook that fails on one document is usually a document problem, and
auto-disabling would make the corpus depend on ingest order.

**Ordering is declared and total.** Hooks sort by `(order, name)`. Registration order is not a
contract, because it depends on entry-point enumeration.

### 3.3 A middleware that changes embedded text is a fingerprint input

This is the one non-obvious rule, and it falls out of two decisions made elsewhere.

`chunks.id` is derived from `embed_text` (`storage.md` §3.2), and the `ChunkFingerprint` /
`EmbedFingerprint` refusals exist to guarantee that everything in the index came from the same
pipeline (`storage.md` §6.3). A middleware that rewrites text between parse and embed defeats
both: two instances with identical configuration and different middleware produce different
vectors from identical source bytes, and **neither refusal notices**, because neither
fingerprint knows the middleware exists.

So: a middleware that alters text which reaches `embed_text` **must declare
`mutates_embedded_text = True`**, and the pipeline folds the sorted set of
`(name, version)` for all such middleware into the `ChunkFingerprint` it compares at startup.
Adding, removing or upgrading one is then exactly as loud as changing the chunk budget, which
is what it is.

Undeclared mutation is not detectable at runtime, so this is a contract rather than a check —
but a declared-but-inert middleware costs only a spurious re-index, while an undeclared active
one silently corrupts the space. The default is therefore `True` for every hook positioned
where text can still reach the embedder — `after:parse`, `before:chunk`, `after:chunk` and
**`before:embed`** — and a plugin must opt *out*.

`before:embed` is the most direct case of the four and the easiest to overlook: it operates on
chunks that are about to be embedded, so a rewrite there reaches `embed_text` with nothing
downstream to normalize it away.

### 3.3.1 `text` is immutable after parse — forbidden, not fingerprinted

`embed_text` is mutable and fingerprinted. **`text` is neither.** It cannot be changed by any
middleware at any hook, and no declaration makes it permissible.

The asymmetry is not stylistic. `text` is what a citation displays and what `Parser.resolve`
must reproduce: [`parsing.md`](parsing.md) §3.3 makes round-tripping a per-parser test
obligation — resolving a chunk's anchor returns the text the chunk claims. A middleware
rewriting `text` breaks that invariant *after every parser test has passed*, producing a corpus
that is internally consistent and whose citations quote text the source document does not
contain. No fingerprint repairs that, because a fingerprint records which pipeline produced the
corpus and this corpus is wrong in a way no version of the pipeline makes right. It is also the
precise defect class this project exists not to reproduce (`PLAN.md` defect #1).

So the capability boundary is expressed in the type rather than in prose. `Chunk.text` carries
`Field(frozen=True)`, so it can be set at construction and assignment afterwards raises:

```python
class Chunk(BaseModel):
    text:       str = Field(frozen=True)   # citable; immutable after parse
    embed_text: str                        # mutable, and a fingerprint input
```

That stops in-place mutation. It does not stop a middleware constructing replacement `Chunk`
objects, so the pipeline adds a check across the hook chain — one digest over the ordered
`text` values before the first hook and after the last:

```python
before = sha256(b"\0".join(c.text.encode() for c in chunks)).digest()
...
if sha256(...) != before:
    raise MiddlewareViolation(hook, "Chunk.text was modified")
```

One hash over data already in memory, once per document per hook chain. Cheap enough that it
runs always rather than under a debug flag — a check that runs only when someone suspects a
problem does not catch the problem nobody suspected.

**Where this belongs.** Stated here because the pipeline is what enforces it, and asserted from
the parsing side as an invariant of the round-trip contract. If [#1](https://github.com/mgd43b/manicule/issues/1)
can express it in the middleware protocol — a hook receiving a `text`-readonly view — that is
strictly better than either, and both statements become descriptions of a thing the type system
already guarantees.

### 3.4 PII redaction, decided

`PLAN.md` defect #5 says pick one behavior. **Redaction happens at the generation boundary,
not at ingest**, and the reason is a consequence of retained original bytes rather than a
preference.

Redacting at ingest permanently destroys data in the index. With `original_ref`
(`storage.md` §7), the unredacted bytes are still on disk in the blob store — so ingest-time
redaction **protects nothing while destroying retrieval quality**. Worse, it is not even
stable: re-parsing from retained bytes (§10) re-derives text through whatever redactor version
is current, so chunk ids and vectors change whenever the redactor's patterns change, and by
§3.3 that is a full re-index.

At the generation boundary the same feature is coherent: text is redacted on its way to a
hosted model, the index keeps what it found, and turning the feature on or off changes nothing
on disk.

**The alternative, recorded.** If the requirement is genuinely "this content must never be
stored", redaction is the wrong tool — that is a *refusal to ingest*, filed as
[#28](https://github.com/mgd43b/manicule/issues/28). Redacting on the way in and keeping the
original beside it satisfies nobody.

---

## 4. Change detection

Two levels, cheapest first, and the conditions they carry.

```
1. version_token differs?  no, parse lineage current                      -> skip, record liveness, done
2. content_hash differs?   no, parse lineage current, source record same  -> skip, record liveness, update token
3. otherwise                                                              -> full ingest
```

**"Unchanged" is a claim about the stored text, not only about the bytes.** Both levels
compare what the *source* has, and neither can see that the parser reading those bytes has
moved underneath them — which is why a `pypdfium2` bump used to be silent: nothing already
stored was ever re-read, while a newly ingested document with identical bytes parsed
differently, and the corpus quietly held two generations of extracted text. So a document
skips only when `documents.parse_fp` matches what its parser would produce today
([`parsing.md`](parsing.md) §3.0). The condition sits on **both** levels, and level 1 is the
one that matters: a source whose version token has not moved never reaches the byte
comparison at all, so a check placed only at level 2 would leave every well-behaved
connector's corpus permanently stale.

**And it is a claim about the metadata, not only about the text.** The same trap, one field
along. A document may carry an authoritative source record ([`storage.md`](storage.md) §4.2.1) —
a mirrored page's real title, canonical URL, source identity and version, out of an adjacent
sidecar manifest. Correcting that manifest changes what every citation of the document *says*
while leaving the page's own bytes byte-for-byte identical, so `content_hash` agrees and level 2
skips. Worse than merely skipping: the skip path then records the **new** `version_token`, so the
corrected record is never read again on any later sync either, and the corpus cites a version it
was told about and then declined to look at. Level 2 therefore also compares the record the fetch
just brought back against the stored one.

The condition sits on level 2 alone, and that asymmetry is deliberate rather than an oversight:
level 1 runs *before* the fetch, so there is no record in hand to compare. What covers level 1 is
the connector's own change token — the filesystem connector folds the manifest's size and
modification time into it, so an edit to either half of a document-plus-manifest pair moves the
token. A connector that supplies records must move its token when its metadata moves, for the same
reason any connector must move it when its bytes do.

Both comparisons go through the validating accessor, so an unusable record on either side reads as
absent and compares equal to another absent one — two documents about which nothing authoritative
is known are not different documents. And a fetch that brings *no* record leaves a stored one
alone rather than counting as a change, so a connector that supplies metadata on only some of its
paths does not re-ingest its corpus on every run.

Parse lineage is selective by construction — the comparison is against the parser *this document*
used,
so a PDF library bump re-parses the PDFs and leaves the Markdown untouched — and it costs a
re-parse rather than a re-fetch, because level 2 has already fetched. A parser manicule does
not ship records no lineage and expects none, so a plugin corpus does not re-parse forever to
learn nothing.

**The pipeline never interprets `version_token`.** It is opaque and connector-defined
(`contracts.md` §2) — a git blob SHA, a Confluence `version.number`, an S3 ETag. The pipeline
compares it for equality with the stored one and does nothing else with it. No ordering, no
parsing, no "is this newer". A connector that wants ordering semantics implements them in
`discover` and its watermark.

**Level 2 exists because level 1 can lie.** A source that touches `lastmodified` on every save
reports a new `version_token` for an unchanged body. Hashing the fetched bytes catches it and
avoids a parse, chunk and embed cycle — which is the expensive part, not the fetch.

**Level 1 exists because level 2 requires a fetch.** Over a rate-limited API across ten
thousand pages, that difference is the whole sync.

**A connector that supplies no `version_token` falls straight to level 2**, which means every
sync fetches every document and the saving comes only from skipping parse, chunk and embed.
That is correct rather than degraded — it is the best available behavior when the source
offers no change signal — but it is a materially different cost profile, so `doctor` reports
which connectors are running without a token rather than leaving a slow sync unexplained.

### 4.1 A skip is not a no-op

The skip path still writes. Three things, and omitting any of them causes a specific bug:

| Write | Omitting it causes |
|---|---|
| `last_seen_at` on the document | `reconcile` (§11) cannot distinguish "unchanged" from "gone" |
| `version_token`, when level 2 skipped | Every future sync re-fetches this document forever |
| Connector watermark, after the run | The next sync re-enumerates from the beginning |

> **Prior art.** The skip path returns early after updating the source version, and there is no
> `last_seen_at` at all — which is consistent, because there is no reconciliation pass to use
> it. Deletion is simply never detected (`contracts.md` §3).

**The trap: a skip must not skip the refusals.** The fingerprint checks in §7 run once per
*run*, before any document is examined, precisely so that a corpus which happens to be entirely
unchanged cannot sail past them and leave the operator believing the configuration is
compatible.

### 4.2 Content hash is over fetched bytes, before parsing

`documents.content_hash` is the sha256 of what the connector returned, and it is the same value
as `original_ref` when retention succeeded (`storage.md` §4.2). Hashing parsed output instead
would make the hash depend on the parser version, so a parser upgrade would look like every
document changing.

That is right, and it is exactly why the parser version needs a column of its own. The hash
answers "did the source change"; `documents.parse_fp` answers "did what we made of it change";
and conflating them into one value would make an upgrade indistinguishable from a corpus-wide
edit — every document reported as modified, its version history gaining a revision nobody made.
Two questions, two columns, and only the second moves on a bump.

---

## 5. Running the fallback chain

`parsing.md` §6 owns which parser runs, what counts as failure, and what status results. The
pipeline owns three things that document names but deliberately does not implement.

**1. The chain is resolved once, per document, before the first attempt** — and recorded in
`metadata.parsers_attempted` as it proceeds. Resolving lazily would let a config reload
mid-chain produce a chain that never existed.

**2. Each attempt gets its own limits.** `parsing.md` §6.3 makes "exceeds its per-parser time
limit" and "exceeds its per-parser memory limit" hard failures that advance the chain. Those
limits are per *attempt*, not per document: a chain of three parsers on a 30-second limit can
legitimately take 90 seconds before the document fails. A per-document limit would make the
last parser in a chain fail for reasons belonging to the first.

**3. A parser that is killed is a hard failure, not a lost document.** This is §6. It is
specifically *not* a decline in the sense of `parsing.md` §6.3 — a parser that declined
inspected the input and reported that it is not its kind, which is information, whereas a
parser the pipeline killed reported nothing at all. Collapsing the two would let a chain of
timeouts end at `unsupported_media_type`, which reads as "manicule does not handle this format"
when the truth is "every parser that handles this format ran out of time".

> **Prior art.** `parseWithFallback` wraps each attempt in `try/catch` and advances on a thrown
> exception or empty output. There is no time limit and no memory limit, so the two failure
> modes that actually take down an ingest run — a parser that hangs and a parser that
> allocates without bound — are not failures at all: they are the process stopping. And when
> every parser has been tried it throws `No parser found`, which the outer handler turns into
> status `error`, so `no_extractable_text`, `unsupported_media_type` and a genuine parse failure all
> collapse into one bucket. `parsing.md` §6.4 separates them; this is where that separation
> has to survive contact with a real chain.

---

## 6. What "never aborts a batch" actually means

The requirement is in `PLAN.md` §4 and issue #5. Taken seriously it is the hardest thing in
this document, because **`try/except` covers only one of the three ways a document takes down a
run.**

| Failure | Caught by `try/except`? | Why |
|---|---|---|
| Parser raises | Yes | Ordinary exception |
| Parser hangs | **No** | The call never returns to be caught |
| Parser exhausts memory | **No** | The OS kills the process, or Python dies allocating |

### 6.1 A timeout that does not work

`asyncio.wait_for` cancels the *await*, not the work. A parser sitting inside a C extension —
`pypdfium2`, `lxml`, `tree-sitter`, `python-calamine`, all of which manicule uses — holds the
GIL or blocks in native code and observes no cancellation until it returns on its own. Running
it in a thread does not help, because Python cannot kill a thread.

**Timeouts are only enforceable across a process boundary.** That is the entire reason parse
runs in a subprocess: not isolation for its own sake, but because it is the only place a
deadline can be enforced against native code.

### 6.2 Memory limits do not exist on the primary platform

The obvious mechanism is `RLIMIT_AS` in the child before it does any work. On Linux that works.
**On macOS it does not**, which matters because macOS on Apple Silicon is the platform manicule
is built for (`PLAN.md` §7).

Measured on Darwin, both the system Python 3.9 and Homebrew Python 3.13:

```
RLIMIT_AS:   (9223372036854775807, 9223372036854775807)
RLIMIT_DATA: (9223372036854775807, 9223372036854775807)
RLIMIT_RSS:  (9223372036854775807, 9223372036854775807)
  set RLIMIT_AS soft=256MiB -> ValueError: current limit exceeds maximum limit
  set RLIMIT_DATA soft=256MiB -> ValueError: current limit exceeds maximum limit
  allocated 512 MiB anyway -> caps NOT enforced
```

All three limits report unlimited, all three refuse to be set, and a child allocates half a
gigabyte under a nominal 256 MiB cap without complaint. A design that says "the worker sets
`RLIMIT_AS`" is correct on CI and inert on the developer's machine — which is the worst place
for a resource limit to be missing, because that is where the malformed PDF gets opened first.

**So detection is platform-specific and the action is not:**

| | Detection | Action |
|---|---|---|
| Linux | `RLIMIT_AS` in the child, pre-`exec` | Child dies allocating; parent sees the exit |
| macOS | Parent polls child RSS (`psutil`) on a timer | Parent sends `SIGKILL` |

`SIGKILL` works identically on both — verified, child exit code `-9`. `psutil` is a permissive
(BSD) dependency and a natural fit alongside the hardware detection in `PLAN.md` §16, though
nothing there requires it today; if that dependency is unwelcome, the macOS path can shell out
to `ps`, at the cost of a fork per poll.

> **Corrected during implementation, and the correction is the interesting one.** The table
> above makes the *mechanism* platform-specific, which is right, and the *enforced quantity*
> platform-specific too, which is not. `RLIMIT_AS` bounds address space; RSS polling bounds
> resident memory. They are different numbers, so a parser that reserves a large arena without
> touching it dies on Linux and succeeds on macOS — the same document, two outcomes, decided by
> the machine. That is precisely the split `PLAN.md` §7 forbids.
>
> **So resident memory is the enforced quantity everywhere**, sampled by the parent, and
> `SIGKILL` carries it on both platforms. `RLIMIT_AS` is still applied in the child wherever
> the kernel accepts it, at four times the resident bound: loose enough that it cannot fire
> before the uniform check in any realistic case, tight enough to catch an allocation so sudden
> that a 250 ms sample misses it. A backstop, not the policy. The child reports whether it was
> able to take that limit, so the difference is observable rather than assumed.

`psutil` is an optional extra rather than a requirement, and the `ps` fallback is the tested
path when it is absent — because a limit that silently stops applying because a package was not
installed is worse than one that costs a fork per poll. Neither choice changes what gets
indexed, which is the test for whether a dependency may be optional at all.

The parent-side poll is a sampling check, so it can overshoot between ticks. That is accepted
and stated: the goal is to stop a runaway before it takes the machine down, not to enforce a
byte-exact quota. Poll interval defaults to 250 ms.

### 6.3 The worker pool

```
parse_workers        default: min(4, cpu_count - 1), never 0
parse_timeout        default: 30 s per attempt
parse_memory_limit   default: 1 GiB per worker
```

- **`spawn`, not `fork`.** `fork` in a process that has loaded MLX and opened SQLite copies
  both into a child that must not touch either. `spawn` costs interpreter startup once per
  worker, amortized over the run.
- **Workers are recycled after `max_documents_per_worker` (default 500)** to bound the effect
  of leaks in native parser libraries, which is a category of bug no amount of care in
  manicule prevents.
- **A killed worker is replaced immediately**; the pool size is what the run depends on, not
  the identity of any worker.
- **Workers hold no store handles.** They receive bytes and return blocks. Everything
  transactional happens in the parent, which is what keeps `storage.md` §8.2's ordering true
  regardless of how many workers died.

### 6.4 Attributing a death to a document

When a worker dies, the parent knows which document it dispatched, marks it `failed` with
`failed_stage=parse` and `status_detail="worker killed: timeout"` or `"worker killed: memory
limit"`, and continues. The batch is unaffected because the batch was never a unit.

**When the parent dies**, in-flight documents are left in a non-terminal status. Recovery is a
startup sweep, and it needs no new schema — `status` and `updated_at` already exist:

```sql
UPDATE documents
   SET status = 'pending', status_detail = 'interrupted; requeued'
 WHERE status IN ('fetching', 'parsing', 'embedding')
   AND updated_at < :now - :stale_after
```

`stale_after` defaults to 1 hour, comfortably above any per-document limit. This is the same
principle as `storage.md` §8.2: a document that is not `indexed` is not served, so an
interrupted document is invisible rather than wrong, and requeueing it is cheap.

### 6.5 One instance per data directory

The sweep above, the tombstone sweep (`storage.md` §8.2) and the blob GC (`storage.md` §7) all
assume a single writer. WAL permits multiple processes, so the assumption has to be enforced
rather than hoped for: **an exclusive lock on `<data_dir>/manicule.lock`, held for the process
lifetime.** A second instance fails to start with the holder's PID, rather than starting and
requeueing the first instance's in-flight documents out from under it.

**It is acquired.** `ingest.recovery.InstanceLock` implements it — an exclusive `flock`, the
holder's PID in the refusal, released on exit — and `Runtime.__aenter__` takes it for every
process that writes. It said "designed, and nothing acquires it" here until recently, which is
why §8.6 exists: the classification of who takes it is the part that has to be got right and
kept right, not the lock.

**`flock`, and the file is a note rather than a lock.** The kernel releases it when the holding
process goes, however it went — so a `SIGKILL`, a power cut or an `OOM` kill all leave the
directory immediately available to the next writer, with the file still sitting there holding a
dead process's number. That number exists so the refusal can name a holder. **Deleting the file
is not the remedy for anything**: it does not release a lock, because the lock is not in the
file, and doing it while a process is running deletes the note without touching the exclusion —
after which two writers see an empty directory and both proceed. If a writer is refused and you
believe nothing is running, the answer is to find the process the message names.

The document-revision guard in §8.5 is deliberately not built on this and holds whether it is
taken or not. §8.4 is why both exist.

### 6.6 Embed is in-process, and that is a deliberate asymmetry

The embedder is in-process by design — that is what keeps `uv tool install manicule` a single
command with no server to operate (`PLAN.md` §7). It cannot be moved behind the same subprocess
boundary without either reloading the model per worker or reintroducing exactly the server
process the design rejects.

So the embed stage does not get the protection the parse stage gets, and the honest response is
to remove the failure modes instead of catching them:

- **A too-long input cannot reach the embedder**, because the `ChunkFingerprint` check (§7)
  refuses at startup when the chunk budget exceeds `max_sequence_length`. This is the single
  reason that check is a startup refusal rather than a per-document guard.
- **The live working set is bounded by batch size**, derived from the fingerprint rather than
  hardcoded (§8.2). This is worth stating precisely, because an earlier version of this line
  said "memory is bounded by batch size" and that turned out to be false in the way that
  matters. Batch size bounds what a forward pass holds *at once*; it says nothing about what
  the runtime *retains afterwards*. The defect in
  [`embeddings.md`](embeddings.md) §3.5 reached 36 GiB at `batch_size = 1`, because MLX keeps
  every buffer it has finished with. Retention is bounded separately, by `cache_limit_mb`.
- **Input is not attacker-shaped by the time it arrives.** Parse and chunk have already run,
  in a sandbox, and produced bounded token counts.

Residual risk, stated rather than hidden: an OOM inside MLX takes the process down, and §6.4's
sweep is what recovers it. That is a worse outcome than a killed parse worker, and it is the
price of an in-process embedder.

**§6.2's RSS polling does not see the embedder at all**, and not only because embed is
in-process rather than in a child. Metal allocations are not ordinary resident pages, so
resident memory is the wrong instrument here even when it is pointed at the right process — it
fell while the process grew twenty-two gigabytes. The embedder publishes `mlx_cache_bytes`
and friends as metrics for exactly this reason.

> **Prior art.** Everything runs in one process, `BATCH_SIZE` is the constant `32` regardless of
> model or chunk size, and contextual-retrieval and chunk-augmentation features issue two LLM
> calls *per chunk* inside the ingest loop with no concurrency bound and no isolation — so an
> LLM timeout on chunk 400 of a document fails the whole document, and a slow endpoint stalls
> the run with no signal distinguishing it from a large corpus.

---

## 7. The refusals

All are specified elsewhere; the pipeline is what runs them. All run **once per run, before
the first document is discovered**, and all are hard refusals.

```
0. the boundaries were measured, not estimated                   (parsing.md §1.2)
1. EmbedFingerprint   config vs index_state vs _manicule_meta   (storage.md §6.3)
2. ChunkFingerprint   config vs index_state                      (parsing.md §1.7)
3. budget_tokens <= max_sequence_length                          (parsing.md §1.1)
```

**Check 0 is first because nothing the others discover can resolve it.** A chunker with no
bound embedder counts with a stand-in vocabulary and inflates the result by a fixed safety
factor, so its boundaries are neither the model's numbers nor reproducible from them.
Provisional chunks are for a dry-run parse or a fixture build; there is no configuration that
makes them fit to serve. It reads nothing — the answer is in the running `ChunkFingerprint`'s
own `tokenizer_id` — so it costs nothing to put first. The same refusal runs again in
`IngestPipeline`'s constructor, which is not redundant: a pipeline is constructible without
going through this function, and everything a pipeline writes is permanent.

**There is no parse-fingerprint refusal here, and §4 is why.** Parsing is per document, so a
stale parser version is not a fact about the run — it is a fact about some of the documents in
it, and the response is to re-parse those and no others. A refusal would have to be
corpus-wide, and a corpus-wide answer to a per-document question is wrong in both directions:
it would either refuse a Markdown corpus over a PDF library, or grow mid-run and refuse a
corpus at the moment it gained its first PDF.

**Check 3 is a cross-check between the two fingerprints**, not a property of either. The chunk
budget lives in `ChunkFingerprint`; `max_sequence_length` lives in `EmbedFingerprint`. Each is
individually valid while the pair is incoherent — a 512-token budget against a 256-token model
silently truncates every chunk, producing vectors for the first half of the text and citations
that point at all of it. Nothing but the pipeline is positioned to compare them, because
nothing else holds both.

**Why once per run and not per document.** A per-document check would run tens of thousands of
times to answer a question whose inputs cannot change mid-run, and — worse — a corpus that is
entirely unchanged would skip every one of them (§4.1), leaving a mismatched configuration
undiscovered until the first new document arrives.

**Why before discovery and not after.** Discovery is the rate-limited part. Refusing after a
forty-minute enumeration is a worse version of refusing immediately.

The failure message and the two exits are specified in `storage.md` §6.3. The pipeline adds one
thing: when it refuses, it prints the count of documents that *would* have been re-embedded, so
`--re-embed` is a priced decision rather than an open-ended one.

---

## 8. Concurrency, batching and backpressure

### 8.1 Three different resources, three different limits

| Stage | Limited by | Default |
|---|---|---|
| discover / fetch | Remote rate limits | `fetch_concurrency = 8`, per connector |
| parse / chunk | CPU cores | `parse_workers = min(4, cpu_count - 1)` |
| embed | **Memory and one accelerator** | serialized; one batch at a time |
| store | SQLite single writer, per-document exclusion | serialized per document |

§8.3 is how those four are actually arranged, and §8.3.3 is which of them to change when a sync
is slow.

**Embedding is serialized, and this is the difference an in-process embedder makes.** With a
model server, concurrency is a connection-pool question and more requests mean more throughput
up to the server's limit. With MLX in-process there is one model, one unified-memory pool, and
one GPU; issuing two batches concurrently produces contention rather than throughput. So embed
is a single consumer, and the parallelism upstream of it exists to keep that consumer fed.

### 8.2 Batch size is derived, not constant

```
embed_batch_size = clamp(target_batch_tokens // budget_tokens, 1, 64)
```

Both inputs come from fingerprints already on hand. A constant batch size is wrong in both
directions: 32 chunks of 512 tokens is a very different allocation from 32 chunks of 8 000, and
the second is where an in-process embedder runs the machine out of memory.

`target_batch_tokens` is the one tunable, and it is the honest place to put the knob because it
is the quantity that actually maps to memory.

### 8.3 The stage topology, as built

> **This section described intent for a long time and the loop did none of it.** `run` awaited
> one document's complete fetch, parse, chunk, embed and commit path before pulling the next, so
> `fetch_concurrency` guarded a semaphore that ordinarily never saw a second caller. Measured
> before the change, with `fetch_concurrency = 4` over eight documents, the most fetches that
> ever overlapped was **one**. `queue_depth_factor` and `shutdown_grace_s` were configurable and
> read by nothing. What follows is what the code now does.

**Durable enumeration, bounded acquisition, then local derivation.** Every bound
below comes from configuration; none is derived from how many tasks happen to exist.

When source-byte retention is enabled, the runtime explicitly wires its document store through
the pipeline's `acquisitions` handle; retention-disabled and protocol-only stores use the legacy
bounded live-derivation path. A durable journal without durable bytes is deliberately never
constructed. The durable path's source
checkpoint is independent of parsing and publication: every current revision is either proved
unchanged or retained and associated with its complete fetched source envelope before the
candidate watermark can become the connector watermark.

```
discover → acquisition journal             one task; commit before advancing
   │  bounded sequence-paged reader
   │  fetch hand-off      depth = queue_depth_factor × fetch_concurrency
   ▼
fetch × fetch_concurrency                  change detection (level 1), then the network
   │  fsync + hash + journal association
   ▼
acquired source snapshot                  committed source watermark after full coverage
   │  bounded local blob reader
   │  parse hand-off      depth = queue_depth_factor × ingest workers
   ▼
ingest × (parse_workers + 1)               parse in the pool, chunk, embed under one
                                           lock, commit under the document's own
```

| Stage | Width | Where the bound comes from |
|---|---|---|
| discover → journal | 1 | One cursor; each identity commits before the next source record is requested |
| journal reader | 1 | Waits for hand-off room, then reads one durable record |
| fetch | `fetch_concurrency` (8) | One task per permitted in-flight fetch |
| parse | at most `parse_workers + 1` attempts | The ingest workers are the only callers of the pool |
| embed | 1 | `asyncio.Lock` around the model call, and nothing else |
| store | per document | The keyed mutation lock, plus SQLite's own single writer |

**Why the ingest stage is `parse_workers + 1` and not `parse_workers`.** One document is
normally inside the embedder, holding the accelerator; without the extra worker a parse slot
sits idle behind it for the length of every forward pass. Not `+ 2` or more: past that, the
extra workers only queue for the same accelerator, and each one is holding a fetched body in
memory while it waits.

**Level-1 change detection sits in acquisition, after durable discovery.** It is what avoids the
fetch. Changed bytes are content-addressed, fsynced and bound to the journal record before the
record becomes acquired. The fetched URI, media type, encoding, metadata, byte length and hash
are stored with that record, so later derivation reconstructs the connector result without the
connector. A fetched revision newer than discovery is recorded as the acquired revision.

**One document is never in two stages.** The concurrency is between documents. A document's
record, chunks, glossary and vectors are published under the keyed lock in §8.4 and guarded by
the compare-and-swap in §8.5, exactly as when one document went through at a time — concurrency
makes those guards more reachable, not less needed.

**Members of a container are expanded by the worker that holds the container**, breadth-first,
rather than being fed back to the front of the pipeline. A wide archive is therefore one worker's
work while the others carry on with other documents, which is simpler than re-entering the
stages and costs only that one archive's serialization.

### 8.3.1 Durable discovery, then bounded hand-offs

Discovery does not feed the fetch queue. It appends one validated source record to the
acquisition journal, waits for that transaction to commit, and only then asks the connector for
the next record. On true iterator exhaustion it atomically records the completion marker and
the connector's candidate watermark. A limit, cancellation, cursor/source failure or journal
admission failure leaves that marker absent.

Only after enumeration stops does a sequence-paged journal reader feed bounded acquisition
workers. Once every item has coverage, `commit_acquisition_watermark` atomically checks the
complete inventory and publishes the candidate watermark. A second bounded reader then loads
and verifies journal-owned blobs for parsing. When embed falls behind, only this local reader is
blocked; the source cursor is already gone and the source checkpoint is already safe.

A crash after the acquired record commits resumes from its blob without another body request.
Blob retention also writes a run-and-record-keyed staging envelope before the journal transition;
if the process stops in that narrow window, takeover recovers the staged hash and fetched source
envelope instead of downloading the body again. The marker is removed only after the fenced
`ACQUIRED` transition commits and pins the same blob.
Every new marker is registered in an indexed relational inventory before either its blob or
marker file can exist. Blob sweep conditionally rechecks every relational root in the same
transaction that deletes a candidate, closing the candidate-selection race. If
the process dies after association but before unlink, a bounded reconciliation page compares
run/source identity, blob hash and acquired envelope with the journal; an exact match makes the
marker redundant. A superseded run's marker is removed even when the association never
committed, because its incremented generation can never resume. History cleanup excludes every
run still named by the inventory, so it cannot erase the evidence before either decision.
Authoritative pre-association markers remain resumable; explicit markers whose owning run has
been cascade-deleted are removed rather than becoming permanent GC roots.

Older marker files are admitted to the inventory in bounded directory pages and matched to
journal records in one batched query, including legacy payloads without run/source fields by
checking their hashed run-and-source key. Until a complete legacy pass finishes, history cleanup
and blob collection fail closed. Unmatched legacy files receive a 30-day safe-harbor entry and
then expire; indexed marker blob references participate directly in GC's SQL mark set. No sync
or inventory call walks the directory or performs one query per marker. Legacy ownership is
looked up by the marker-name index, so one page cannot expand into unbounded source history.
Durable staging writes are joined before task cancellation returns. A process death can still
leave a temporary staging file; acquisition startup and blob inventory remove only day-old
partials from a partial-only directory, in genuinely bounded batches whose report contains
aggregate counts rather than source names, URIs or envelope contents. Live recovery markers cannot
consume that cleanup budget. Marker deletion and its parent-directory sync are likewise one
joined operation, so repeated cancellation cannot return while the deletion is still non-durable.
Authentication loss before that point leaves a typed retry and withholds the watermark;
authentication loss afterwards cannot block parsing, embedding or publication because those
phases make no connector calls. Missing, stale and deleted source bodies likewise remain typed
acquisition retries and cannot publish older bytes.

First indexing, offline-snapshot indexing and re-parse all enter the same derivation function
with a retained reference. That function parses the bytes it was given and records that existing
reference; it does not retain the top-level body again. Container members are different documents
created by local expansion, so they retain their own member bytes once. A local parse, embed or
publication failure leaves the journal record retryable with its blob reference intact. A later
invocation forces that retained snapshot through derivation again without discovery or fetch.
An orderly invocation that leaves retained derivation pending reports an incomplete,
retry-required outcome and releases its fenced lease so the retry can begin immediately. Process
cancellation and crashes do not release ownership; takeover still waits for lease expiry.

Fetched revision evidence is source-owned. A versioned discovery record must receive a fetched
token; opaque token sources require exact equality, while a connector may expose an ordering
proof (Confluence uses integer page revisions) to accept a body that advanced after discovery.
Missing, older, malformed or unrelated evidence is a typed stale-body retry and is refused before
the bytes enter retention.

**The most a run holds in memory**, with the defaults on a four-core machine — where
`default_worker_count()` is `min(4, cpu_count - 1)` = 3, so there are four ingest workers:

```
  fetched bodies  =  parse hand-off  +  ingest workers  +  fetch workers blocked on put
                  =  2 × 4           +  4               +  8                            =  20
```

`tests/ingest/test_stage_widths.py` derives those numbers from the pipeline and fails if either
this arithmetic or the formula behind it moves, because a documented limit that has quietly
stopped matching the code is the defect this whole section was.

The fetch hand-off holds durable source records directly. The journal reader waits for a free
slot before reading one record, so the read page plus queued records together never exceed the
hand-off capacity. **The complete discovery result is durable but never loaded into memory**,
and neither is the complete fetched corpus; `peak_discovery_records` and the queue peaks make
both claims checkable.

**The end-of-stream is not subject to the bound**, and the reason is a deadlock that this
otherwise has. A stage closes the hand-off in front of it from a `finally`, which is where it
lands when the run is being torn down — and at that moment its consumers are being canceled too.
A bounded queue would make delivering the end-of-stream wait for a drain that is never coming,
inside a `finally`, which is a teardown that never completes. So the items are bounded by a
semaphore and the sentinels go into an unbounded queue: closing a stage cannot wait.

### 8.3.2 What a run reports about its own stages

`connectors.metadata.last_run.stages` carries counts and never content: accepted documents, each
hand-off's capacity, peak depth and how many times a producer had to wait, peak occupancy of the
fetch, parse and embed stages, and peak retained bodies. It is what §14's "queue depth and embed
throughput during a run" is made of, and what makes each bound above checkable from outside.

**Two of them are process-wide and are labeled as such.** The parse pool and the embedder are
one pool and one accelerator for the whole process, so a re-parse sweep running beside a sync is
counted in `peak_parses` and `peak_embeds` too. Retained bodies belong to the run.

`blocked_puts` on the fetch hand-off is direct evidence that the bounded journal reader waited
for local fetch capacity. It no longer describes source discovery, which has already completed.

### 8.3.3 Tuning it on a local Apple Silicon machine

The in-process MLX embedder is the bottleneck by construction, and everything upstream exists to
keep it fed. So the tuning order is: leave `target_batch_tokens` alone unless memory is the
problem, and change the widths only when a measurement says which stage is idle.

| Symptom | Read | Change |
|---|---|---|
| Sync is slow, `peak_fetches` well below `fetch_concurrency` | The source is not the limit | Nothing; look downstream |
| `blocked_puts` is zero and the run is slow | Nothing is waiting on anything; the source is the limit | Raise `fetch_concurrency`, within the remote's rate limit |
| `blocked_puts` is large and `peak_parses` is at its ceiling | Parse is the limit | Raise `parse_workers`, up to one below the core count |
| `blocked_puts` is large and `peak_parses` is low | The model is the limit, which is the expected steady state | Nothing here helps; this is what the machine does |
| Memory pressure during a sync | Too much held at once | Lower `queue_depth_factor` to 1, then `fetch_concurrency` |
| Memory pressure inside the model | The batch is too large for this budget | Lower `target_batch_tokens` (§8.2) |

`parse_workers` above `cpu_count - 1` costs more than it buys: the parent is doing the embedding
and every write, and a machine with no core left for it makes the accelerator wait on the
scheduler. `fetch_concurrency` is bounded by the remote rather than by the machine — a source
that rate-limits at four concurrent requests is not improved by eight.

`tools/benchmark_ingest_stages.py` measures the shape of all of this against the sequential loop
it replaced, over a synthetic delayed source. Its timings are synthetic and are not a throughput
claim; what they are good for is showing that the stages overlap and that the bounds hold while
they do.

> **Prior art.** The sync loop bounds concurrency with a `Set` of in-flight promises and
> `Promise.race`, which is a queue with no backpressure onto discovery: `discover()` is pulled
> as fast as tasks retire. It also decrements the in-flight set in a `.finally` that is not
> ordered against the `race` resolution, so the effective concurrency can drift above the
> configured limit. A semaphore acquired before dispatch and released after has neither
> property.

---

### 8.4 Four exclusions, and none of them is any of the others

They are easy to confuse because they are all "the lock", and confusing them is how a
lost-update bug ships: for a while this document said a concurrent sync and a corpus-wide
re-parse were serialized *because* they shared the embed stage's lock, which is a statement
about the model being read as a statement about the database.

| What | Mechanism | Scope | Excludes |
|---|---|---|---|
| Model / accelerator | `asyncio.Lock` on the pipeline | one process, **all** documents | two batches inside the embedder at once |
| Per-document mutation | keyed `asyncio.Lock`, one entry per document id | one process, **one** document | two operations writing one document's record, chunks and vectors at once |
| Data directory | `flock` on `manicule.lock` (§6.5, §8.6) | one machine, all processes | a second *writer* starting at all |
| Document revision | compare-and-swap at the commit (§8.5) | durable; no process, no lock | a commit derived from a document that has since moved |

**The model lock says nothing about the database.** Two operations can queue politely for the
embedder, one after the other, and then write over each other's documents — which is exactly
what happened. It is one lock, held around one stage, and the stages either side of it are
where a document is read and where it is written.

**All four are now reached by a single sync**, which they were not while the loop was sequential.
Before §8.3, one connector sync had one document in flight, so the second row could only be
contended by a *second* operation — a sweep beside a sync — and the first row had one caller by
construction. A staged run contends both from inside one command. Nothing about the four
changed; what changed is that a run now exercises them, and the tests in
`tests/ingest/test_concurrency.py` assert each of the first two separately rather than inferring
one from the other.

**The mutation lock is keyed, and that is the whole reason it is affordable.** A document's
vectors are staged and its relational publication is flipped under this lock; what must not
interleave is that one document's sequence, not all work everywhere. A
pipeline-wide lock would make a sweep over ten thousand pages block every sync in the
installation for the length of it, to fix a problem no two unrelated pages have. The entry is
dropped when the last holder leaves, so a sweep does not accumulate one lock per row it has
finished with.

**The mutation lock is not durable and is not the guard.** It is an `asyncio.Lock` on a
pipeline object: it holds inside one event loop in one process, and a second process opened on
the same data directory takes no part in it. That is §6.5's job, and §6.5 is now taken by every
writer (§8.6). The invariant in the last row still needs neither, and that is the point of
defense in depth: the process lock is the one that can be absent because somebody ran a writer
against a directory a *third-party* tool was also writing, and the compare-and-swap is what
holds when it is.

### 8.5 Optimistic commit: an operation may not commit on a document that moved

**The invariant.** An operation derived from document revision *R* does not commit if the
stored document moved past *R* while it was running.

The revision is not a duplicate counter. It is derived from what is already stored — active
publication, content hash, version token, retained-source reference, parse lineage, and the
source record. Source fields detect a connector winner; the publication detects another repair
that committed different derived state over the same bytes. `status` is deliberately not part
of it: a re-parse may retain `indexed` throughout, and transient state is not source identity.

**Who is guarded.** Every operation that derives its content from something already stored and
has a command to reach it: `document reindex <id>` and `document reindex --stale`, both of which
run through `reindex.re_parse`. **A connector sync is never guarded**, and that is not an
omission: it is holding the newest bytes the source has, so there is nothing it could be losing
to. A sync always wins.

`reindex.repair` and `reindex.re_embed` — rungs 1 and 2 of §10's ladder — are **not** guarded.
Both would need it: each reads a document, derives from it and writes back, which is the shape
this section is about. Neither is reachable, because neither has a caller anywhere in `src/`;
they are library verbs with tests and no command. Stated rather than left implicit, because the
day one of them gets a surface is the day it needs an expected revision, and the omission would
otherwise have to be rediscovered.

**Where the check is.** In the relational publication transaction, not before it. On an
expected-revision path the conditional `UPDATE` is the transaction's first SQL statement, so it
takes SQLite's write lock before any snapshot read can go stale; then
the same transaction replaces chunks and glossary rows, advances lineage and flips the active
publication. A `SELECT` followed by those writes would have a gap exactly as wide as the one
being closed. Vectors are deliberately staged before this check: a losing run may leave inert
rows, but cannot replace any row in the active publication, and tombstones make those rows
reclaimable. Parser-empty, container, unsupported, middleware-skipped and chunker-empty branches
all call the same guarded transaction.

**What an operator does about a `superseded` result: nothing.** It is not a failure and it is
not work left undone. It says a connector sync committed newer bytes for that document while
the sweep was re-parsing older ones, and the sweep declined to write the older result over the
newer one. The corpus holds the newer text. The document is either already current, in which
case there is nothing to repair, or still stale, in which case the next run of the same command
picks it up — re-running the sweep converges with nothing done by hand in between. A sweep
reporting a *lot* of them is reporting that it was run during a large sync, not that anything
is wrong.

**The mode that is intentionally not supported** remains two processes writing one data
directory concurrently. The commit guard nevertheless fails safely: a loser can stage an
unpublished vector generation, but it cannot alter the winner's relational publication or its
active vectors. The directory exclusion in §6.5 still prevents the wasted work and is taken by
every writer (§8.6).

### 8.6 Who takes the data directory, and who does not

**The guarantee.** One writer process per resolved data directory, at a time. A second writer
does not wait, queue or retry: it fails before it has opened the database, naming the lock
path, the holder's process id where the file can supply one, and what to do instead.

**Writers take it for their whole life**, from before the schema migration and before the
recovery sweep to after the engine is disposed. `Runtime.__aenter__` is where it happens, which
is the earliest moment a runtime exists — a lock taken after recovery has requeued another
process's in-flight documents has protected nothing that mattered. Servers, watch mode and the
REPL call `Runtime.acquire()` on the way in so the refusal arrives before a port is bound
rather than out of the middle of a serving loop.

**Readers take nothing.** `search`, `ask`, `doctor`, every listing and every `--json` read run
normally while a sweep or a sync holds the directory, which is the point: a lock that stopped
`doctor` running during the operation somebody wanted diagnosed would be worse than no lock.
`manicule.app.dispatch.READ_ONLY_OPS` is the list and `writes()` applies it.

**The default is "writer".** An operation not in that list takes the lock. That is the safe
direction — the unsafe one is a command added in a hurry that indexes beside a running sweep —
and `tests/app/test_process_exclusion.py` enumerates the command line's own operations and
fails if any is classified as neither, so the decision is made rather than inherited.

**Three boundaries where a reader is not purely a reader**, audited rather than assumed:

| What | Which way it falls | Why |
|---|---|---|
| The schema migration | Reader takes the lock **for the migration only** | It rebuilds tables. A reader meeting a database behind head has to migrate it; it does so under the lock and gives it straight back. On a directory already at head nothing is taken, which is the ordinary case. |
| The workspace row | Reader, unlocked | One guarded `INSERT` of a row nobody else is inserting. Taking an exclusive lock for it would refuse readers for no gain. |
| `query_logs` | Reader, unlocked | `ask` and `search` record one row per retrieval. It is observability, it is already written so that losing the SQLite writer cannot fail the query, and making a read conditional on a write is the failure that code exists to avoid. |

`backup` and `export` are readers, and the cost is worth stating: they copy a directory that
another process may be writing, so a copy taken during active indexing is a copy of something
moving. That was true before this lock existed and is not fixed by it. Making them writers
would mean no backup could be taken while a server was up, which is when somebody most wants
one; a consistent copy under load needs a snapshot, not a lock.

**Shutdown.** The lock is released after the container and after the engine, so it outlives
every storage operation it was taken to exclude. `Ctrl-C` and `SIGTERM` reach the runtime's
`__aexit__` and release it there. Anything more abrupt is the kernel's job and needs nothing
from manicule — see §6.5 on why deleting the file is not a remedy.

**How this relates to the other two guards.** The process lock excludes a second *process*.
Storage transactions exclude a second *statement*. The document compare-and-swap (§8.5)
excludes a commit derived from a document that has since moved. They are three different
scopes; none replaces either of the others, and this one is the outermost.

## 9. Storing

The write order and its crash windows are `storage.md` §8.2 and are not restated. The pipeline
adds two rules.

**A failed re-ingest must not demote a working document.** If a document is `indexed` and a
re-ingest fails at any stage, it stays `indexed` with its existing chunks and vectors, and the
failure is recorded in `status_detail` or `metadata.last_ingest_error`, according to whether the
document was already indexed. A new status is applied only when there is something to replace the
old content with. A parse failure preserves
the indexed source revision wholesale — token, content hash, retained-byte reference, routing and
provenance — because adopting the failed fetch's token would make the next healthy sync skip bytes
the document does not hold. The error is the only new fact that attempt established.
For a document without a working publication, the failed row and its byte-retention outcome are
one transaction: an initial (`expected=None`) failure cannot expose only one of them, and a reparse
failure uses the same revision compare-and-swap as a successful publication. A stale failure can
therefore neither demote a concurrent winner nor replace the winner's `original_ref` afterward.

**A terminal *determination* does replace it, and that is not a softening of the rule.**
Implementation forced the distinction: `failed` means we do not know whether the new bytes hold
text, so destroying a working answer over it is indefensible. `no_extractable_text`,
`unsupported_media_type` and `container` are conclusions *about the new bytes* — the source
document changed, and continuing to serve chunks derived from bytes it no longer has would cite
text the document does not contain. Between "keep serving something stale" and "keep serving
something the source never said", this project has only ever had one answer. So a failure never
demotes; a conclusion does.

The same asymmetry governs the in-flight statuses: they are written only for a document with no
servable content. An `indexed` document is never marked `fetching` or `parsing`, because those
states are not servable and writing one would unserve a working document for the duration of
every re-sync — which is the very bug this section exists to prevent, arriving by the back door.

> **Prior art, and the reason this rule exists.** `updateDocumentForReindex` sets
> `status = 'pending'` *before* parsing begins, and the failure handler sets `status = 'error'`.
> Retrieval filters on `status = 'indexed'`. So a document that was indexed and searchable
> becomes unsearchable the moment a re-ingest starts, and stays unsearchable if the parse fails
> — while its chunks and vectors are still sitting in both stores, intact and now unreachable.
> A transient network error during a routine re-sync silently removes a working document from
> the index. If the process is interrupted mid-re-ingest it stays `pending` forever, because
> nothing sweeps it.

**Terminal no-content statuses still store the document.** `parsing.md` §6.4 requires it, and
the pipeline is what has to resist the temptation to treat "zero chunks" as "nothing to do".
Storing the failure is what makes it re-queryable, skippable on the next sync, and reachable by
`document reindex --status no_extractable_text` the day OCR lands. Every non-failed terminal
conclusion publishes the document, empty chunks, empty glossary and lineage in one transaction;
none first exposes a new row and clears its derivatives afterwards.

---

## 10. Re-ingest from retained originals

`PLAN.md` §4 asks for re-ingest against a pinned corpus rather than a re-crawl. With
`storage.md` §7's retained bytes that becomes three distinct operations, each landing on a rung
of the blast-radius ladder (`storage.md` §1):

| Operation | Reads | Rung | Network | Shipped as |
|---|---|---|---|---|
| **re-detect glossary** | `chunks.text` | 0 | none | `document reindex --stale-glossary` |
| `repair` | `chunks` | 1–2 | none | `ingest.reindex.repair`, no command |
| **re-embed** | `chunks.embed_text` | 2 | none | `reembed plan/start/execute/status/abandon/cleanup` |
| **re-parse** | `blobs` | 3 | none | `document reindex <id>`, `document reindex --stale` |
| a forced sync | the source | 4 | yes, rate-limited, **may fail** | `index <path> --reindex`, for a path |

**Rung 0 is below repair, and it is a rung of its own rather than a wider `--stale`.** Glossary
detection reads chunk text and writes rows: no connector, no retained bytes, no parser, no
chunker, no embedder, no vector. Folding it into the re-parse sweep would work and would charge
a corpus-sized parse and re-embed for a change to a regular expression, which is the cost an
operator most needs to be able to avoid. §10.2 is the whole of it.

**`re-embed` is the one verb that deliberately does not reuse a stored vector**, and it is
worth stating here rather than only in its docstring, because a specification for durable reuse
would naturally list it among the paths that should. It is the verb an operator reaches for when
the *vectors themselves* are suspect — a reconfigured model, a restored `vectors/` directory,
a rebuild that stopped half way. Reuse would find every embedding input unchanged and skip every
forward pass, so the one command that means "I do not trust these vectors" would succeed, report
success, and change nothing. Every other path reuses; this one is the escape hatch, and an
escape hatch that quietly does nothing is worse than none.

**Only the last one can fail for reasons outside the machine**, and it is the only one that is
not reproducible. Everything above it is a pure function of what is already on disk. That is
the whole return on retaining bytes, and it is why re-parse is a first-class verb rather
than a flag on sync.

The right-hand column is there because not every rung is an operator-facing command. Repair runs
on the recovery path. Re-embed and re-parse have explicit verbs. And rung 4 ships for a *path* — `index <path>
--reindex` skips change detection — while a configured connector has no `--force`, so the only
way to make one re-fetch today is to change what the source reports.

**Selection is a query, not a scan**, because of the per-document lineage in `storage.md` §6.4:

```sql
-- a tree-sitter grammar upgrade invalidates code documents and nothing else
SELECT id FROM documents WHERE chunk_fp <> :current AND media_type IN (:code_types)
```

`select()` accepts the same selectors as `document list` — status, connector, media type,
chunk fingerprint — so "re-parse everything that came out of the old PDF parser" is expressible
without a bespoke flag. The shipped commands expose the one selector an upgrade needs, and a
single document id, which is the narrow end of the same verb and the one a restore reaches for
(§11.2).

**Re-parse is subject to the same identity rules as first ingest.** It runs the current parser
chain over the retained bytes, produces chunks, and reconciles them against the stored set by
`chunks.id`. Unchanged chunks keep their ids and their vectors (`storage.md` §3.2), so a parser
fix that changes one table in a hundred-page document leaves every other chunk's vector alone.

**A document with no retained bytes cannot be re-parsed**, and the command says so per document
rather than failing the run: `original_ref IS NULL` because retention was capped or the document
predates retention. Those are listed with their `original_omitted_reason` and are the only
documents for which a forced re-sync is the only option.

### 10.1 `document reindex --stale`, the corpus-wide end

```
manicule document reindex --stale [--dry-run] [--batch N]
manicule document reindex <id>
```

One verb, two ends. An id repairs one document; `--stale` repairs every document an installed
parser has moved past. Neither touches the network.

**What makes a document stale.** `documents.parse_fp` records the fingerprint of the parser
that produced its text — manicule's own rules version for that parser plus the versions of the
libraries it reads with. A document is stale when that string is not one
`parsers.versions.current_parse_fingerprints()` would produce now: a `pypdfium2` release, a
`selectolax` release, or a hand-bumped `PARSERS[...].rules` after a change to what a parser
emits. A `NULL` lineage is also selected — no recorded fingerprint is no evidence the stored
text is current — which is how documents predating the column stay reachable.

**Why every document is re-parsed when only some change.** A `parse_fp` records the parser's
version, not the document's content, so after a bump it matches for none of that parser's
documents. Nothing can tell in advance which pages the change actually moves without parsing
them, and a fingerprint that could would be a hash of the output rather than of the rules. The
report says so afterwards: `reparsed` is what was rebuilt, `changed` is what came out
different, and on a narrow bump the second is a small fraction of the first.

**Why unchanged vector rows are kept.** A chunk's id is derived from its content and position,
so a chunk the re-parse did not move comes back with the id it already had, the vector row
stored against that id is still its row, and every citation that resolved to it still resolves.
`chunks_kept` counts those; `chunks_new` counts the chunks the sweep produced that were not
already stored.

**Neither is a count of forward passes**, and that is not a caveat about precision — they are
answers to a different question. A row surviving is not the vector inside it surviving: a chunk
is embedded from `embed_text`, which carries the heading breadcrumb, so a document whose
headings moved re-embeds chunks whose ids never changed, and their rows are rewritten in place.
The mismatch runs the other way too — a document with a paragraph inserted at the top renumbers
every chunk below it without moving one embedded string, so those chunks are *not* in
`chunks_kept` and their vectors are reused anyway.

**What is reused, and under what condition**, is `storage.md` §6.2's numbered reuse invariant —
three clauses, of which the third — *a readable stored vector actually exists for that
identity* — is manicule's own and is absent from both written specifications for this problem.

**What is a count of forward passes is `embedding`**, and it is measured at the model rather
than inferred from row identity. The sweep partitions every prepared chunk three ways and
reports the parts separately, because the remedy differs:

| Field | What it is | What it means |
|---|---|---|
| `embedding.reused` | vectors taken from the store | the embedding input was unchanged and a readable vector was found |
| `embedding.input_changed` | chunks sent to the model | the input is new or has moved — including a chunk whose id survived |
| `embedding.repaired` | chunks sent to the model | the input was unchanged and the stored vector was missing or unusable |
| `embedding.forward_calls` | batches the model was asked for | the number accelerator time is proportional to |
| `embedding.first_seen` | chunks sent to the model | the document held nothing to reuse — growth, not change |
| `embedding.cache_hits` | chunks the embedder's warm cache served | the process-local layer, reported apart from durable reuse |
| `embedding.vectors_backfilled` | rows whose identity was reconstructed | a `vectors/` directory predating the identity column, converting itself as it is swept |

A rising `repaired` on a corpus nobody edited is the one line here that asks for attention: it
says vector rows are going missing or arriving damaged, which is a question about the data
directory rather than about a parser.

**There is deliberately no counter for reuse missed because the embedding fingerprint
changed.** A changed fingerprint does not produce misses. It refuses the run before discovery
(§7) and prints what a rebuild would cost, and the vectors live in a table named after the
fingerprint (`storage.md` §6.5) so a new one never meets the old rows at all. A counter for it
could only ever read zero, and a number that cannot move is a worse artifact than a sentence
saying why — this repository has shipped a byte cap that bounded nothing and an interval that
scheduled nothing, and both read as working features.

**Reproducing the numbers.** `tests/benchmarks/embedding_reuse.py` is the program the figures
in this section and in `parsing.md` §4.5 come from. It is in the repository rather than in
somebody's notes because a figure cited as an argument, reproducible only by whoever took it,
is a claim whose check is not where its reader is:

```bash
.venv/bin/python -m tests.benchmarks.embedding_reuse                      # the current cost
.venv/bin/python -m tests.benchmarks.embedding_reuse --no-reuse           # the cost before
.venv/bin/python -m tests.benchmarks.embedding_reuse --changed-fraction 0.1
```

`--changed-fraction` is what prices a *partial* migration: a bump whose parser moves that
fraction of documents embeds that fraction of chunks. It counts forward passes, not seconds at
a real model — the embedder is a counting stand-in, and the constant factor against a real one
is unmeasured.

**What a dry run guarantees.** `--dry-run` performs the selection and nothing else: no parse,
no chunking, no embedding, no blob read, no write to the database, the vector store or the
lineage columns. It reports the same `selected` count the real run would, and it names the
documents whose `original_ref` is unset. The one thing it cannot tell you is whether a retained
reference still resolves to bytes on disk — that is a blob read, and it is left to the run that
is allowed to do work.

**Documents with no retained bytes.** Reported per document with the reason and the remedy,
never as a reason to stop. There are two ways to get there: retention refused the bytes at
ingest (`original_ref IS NULL`, with `original_omitted_reason` saying why) or the blob is gone
from the data directory. The first needs a forced re-sync, which is rung 4 and the one rung
that can fail; the second may only need the data directory restored. The sweep will not fetch
on its own — leaving rung 3 is a decision an operator makes.

**Stopping and resuming.** A document is the transaction boundary. Its vectors are staged, then
one relational transaction publishes the document, chunks, glossary and lineage together.
Interrupting a sweep leaves the prior indexed publication servable and selected; staged vector
rows are inert and the normal tombstone sweep reclaims them. Run the command again. Documents
already repaired are not selected a second time, so a restart picks up the remainder rather
than starting the corpus over.

**Batching and the cursor.** The selection is paged, `--batch` documents at a time, and the
cursor is the number of documents a pass *left behind* rather than a page number. A repaired
document leaves the selection, so the set shrinks under the iteration: counting pages would
skip the documents that shift forward into the vacated slots, and restarting at zero each time
would re-read an unrepairable prefix for ever. Embedding batches are the pipeline's own
(`embeddings.md`), unchanged — the sweep introduces no second consumer of the model, so a
concurrent sync and a sweep queue for the accelerator rather than contending for it. That is a
statement about the model and not about the database; §8.4 is the difference.

**Idempotence, and its one exception.** A second run immediately afterwards selects nothing and
performs no embedding work. The exception is a document produced by a parser manicule does not
ship: `parse_fingerprint` has no version to read for it, so it records `NULL` and is selected by
every sweep. It is re-parsed exactly once per run and remains selectable afterwards. That is
the deliberate trade in `parsers/versions.py` — a plugin corpus that no repair can reach would
be worse — and it is why the sweep tracks what it left behind rather than trusting the selection
to empty itself.

**Concurrency.** Running the sweep while a connector syncs is expected, and §8.4 is the whole
picture. The short version: the sweep runs through the pipeline the runtime already built, so
it shares the embed stage's lock (§6.6) with any sync beside it and the two never reach the
model at once — **which serializes the model and nothing else**. What keeps the two from
writing over each other on the *same* page is the per-document mutation lock within a process,
and the compare-and-swap at the commit (§8.5) everywhere, including where no lock is held.

**`superseded` is a third outcome, beside `reparsed` and `failed`.** A document a connector
sync overtook mid-repair is reported under it, named, and left alone: the sweep declined to
write an older parse over newer bytes, so the corpus has the newer text and the sweep did no
work on it. Nothing needs doing about one — §8.5 has the longer answer, and re-running the
command converges without any manual step.

### 10.2 `document reindex --stale-glossary`, and why detection needs a rung of its own

```
manicule document reindex --stale-glossary [--dry-run] [--batch N]
```

**The problem this closes.** Glossary entries are derived from stored chunks by a detector whose
grammar and evidence rules live in this repository and change independently of every other
stage. Detection runs at ingest, downstream of parsing; a re-sync of unchanged bytes skips
before it; and `documents.parse_fp` does not move when a detection rule does. So a corpus
indexed before a detector fix keeps the entries the old rules produced, reports a current
parser, chunker and embedder fingerprint, and is **correct about all three**. Five detector
corrections landed in one day and not one of them could reach an existing index.

Inferring glossary freshness from parse freshness is not the fix. It would migrate the media
types that happened to get a parser bump, leave every other one stale, and hide the problem
behind a coupling nobody could see — so parsing and detection are independently accountable, and
`documents.glossary_fp` is the column that makes them so (`storage.md` §6.4).

**What the fingerprint is made of.** `GlossaryFingerprint` carries three identity fields:

| Field | What it covers | Why |
|---|---|---|
| `detector` | which detection strategy ran, or `disabled` | two strategies would otherwise be indistinguishable from one typo fix |
| `rules` | a SHA-256 over `ingest/glossary.py` and `core/glossary.py` | the grammar, the persistence threshold, the evidence weights, the boundary model **and** the normalization that turns a surface into a key |
| `libraries` | `name@version` for everything outside this repository that decides a stored entry | a digest catches a rule *we* change and cannot catch one changing underneath an unchanged file |
| `middleware` | `name@version` for every configured hook | detection reads `Chunk.text` boundaries and `heading_path`, neither of which any middleware declaration covers |

**`libraries` is derived from the sources' own imports, not from a list.** `pydantic` validates
`GlossaryEntry`'s field constraints, so it decides which rows may be persisted at all.
`unicodedata` is the sharper one: `normalize_acronym` NFKC-folds a surface into the stored
*lookup key*, and #121 put NFKC into `initial_skeleton` as well — so the character database
version decides what a term is filed under.

**The interpreter is recorded beside it, at its feature version**, because `unicodedata` is not
the only standard-library module that decides an entry and recording it alone would catch one of
two failures with the same shape. `re` compiles every written form, `str.isupper` is the whole of
the shape gate, and `str.casefold` decides whether two aliases are one. None has a distribution.

`python@3.13` rather than `python@3.13.11` is deliberate. CPython's policy is that a patch
release fixes bugs without changing documented behavior, so re-staling a corpus on one would
invalidate for a change that is not supposed to exist — and two machines a patch apart would
disagree about a restored index for a reason nobody could act on. **The accepted risk is a
behavior change that ships in a patch release anyway.**

**What the derivation does not see** is written down in `glossary_lineage.DERIVED_FROM`, in the
style `NOT_DIGESTED` uses, because a derived mechanism fails silently when its derivation has a
hole. Imports at any nesting are covered — inside a function, under `if TYPE_CHECKING`, in a
`try/except ImportError` — since the walk reads the whole syntax tree. What it cannot follow is a
*dynamic* import, and the suite refuses one by a single rule: **a call is refused when the name
at the call site is `import_module` or `__import__`, plain name or attribute.**

That rule is worth stating rather than illustrating, because both of its edges follow from it.
It refuses `__import__(...)`, `builtins.__import__(...)`, `importlib.import_module(...)` and a
bare `import_module(...)` bound by a `from` import — *and* an unrelated
`whatever.import_module(...)`, which is the over-breadth it buys its simplicity with. A false
positive there is one loud failure telling somebody what to decide; a false negative is a
dependency deciding stored entries with no version recorded anywhere, in silence.

**It is not a prohibition, and `DERIVED_FROM` says so rather than sounding like one.** Anything
whose call site names something else escapes: `f = importlib.import_module` then `f('x')`, a
`getattr`, a call built in `eval`. Closing those needs value-flow analysis, which an AST walk is
not. What the rule buys is raising the cost of the mistake from accidental to deliberate, and the
three open routes are asserted open by a test — so the claim cannot go stale in either
direction.

**It is derived, not maintained.** `ParserVersions.rules` is a number somebody has to remember
to move, and its own table records two parsers bumped for changes they did not make, by
somebody who noticed. Detection has no dependencies at all — it is regular expressions over
lines, by requirement (`retrieval.md` §14) — so every input to what gets stored is a byte in one
of two files, and the honest identity is a digest of those bytes. Each of the five corrections
would have moved it with nobody to remember anything.

The trade is stated rather than hidden: a digest over bytes cannot tell a rule from the
paragraph explaining it, so a comment-only edit makes the corpus stale. That is the direction
worth being wrong in. Over-invalidation costs one pass over stored text with no GPU in it;
under-invalidation costs a definition served from rules that no longer exist, silently. A
normalized digest that skipped comments would fail the other way, and its failure would be the
kind nobody notices.

**That has now happened once, deliberately, and it is worth naming rather than leaving to be
discovered from a count.** The American-English sweep renamed `normalise_acronym` and
`normalise_expansion` to `normalize_acronym` and `normalize_expansion` and respelled prose in
both digested files. No detection rule changed — the same input yields the same entries — but
the bytes moved, so `rules` went from `sha256:35aaf33f…` to `sha256:e6867497…` and every
document indexed before it is now selectable by `document reindex --stale-glossary`. **That is
the mechanism working, not a regression**: a digest that could tell a rename from a rule change
would be the normalized digest this section just refused. The repair is the cheapest one in the
system — a pass over stored text, no connector, no parser, no embedder, no GPU, no network —
and at the time it landed nothing real was indexed, so it cost nothing at all. An operator with
a live corpus pays one sweep they can watch finish.

The one thing still maintained by hand is the *list* of digested files, and forgetting it is
made as loud as it can be made: `detector_imports()` reads what those files actually import out
of their own syntax trees — `TYPE_CHECKING` imports included, because those never execute and so
are invisible to any runtime inspection — and the suite fails, by name, unless every `manicule`
module they reach is either digested or listed with the reason it cannot change what gets
stored.

**`min_entry_confidence` is deliberately not in it.** That setting is applied at query time, so
it changes which stored entries a query will act on and not which entries are stored. The
threshold that decides persistence is `MIN_DEFINITION_CONFIDENCE`, which is a constant in the
digested source and therefore already covered.

**What the command costs.** It reads `documents` to select, reads `chunks` per document, and
writes `glossary_entries`, `glossary_aliases` and one column. It constructs no pipeline on
either path, so it never builds a chunker, an embedder, a vector store or a pool of parse
workers, and it never opens the blob store. The cost boundary is the signature of
`ingest.reindex.redetect_glossary`: there is no `pipeline`, `blobs`, `embedder` or `vectors`
parameter for it to reach a parser or the model through, and the suite asserts that rather than
trusting it.

A consequence worth stating: the fingerprint refusals that guard a writing run are **not** run
here. They exist because a corpus must not be written to by a different chunker or embedder, and
this writes neither chunks nor vectors — so an index whose embedder disagrees with configuration
can still be brought on to current detection rules.

**Empty is a result.** A document the current detector reads and finds nothing in records the
fingerprint and no entries. Without that, "no definitions on this page" and "nobody has looked"
would be one state, every sweep would select every prose page for ever, and the feature's own
report would be meaningless. `unchanged` counts documents whose entry set came out as it went
in — the expected majority after a narrow rule change — and they still advance their
fingerprint.

`changed` compares entry **sets**, not counts. The change this exists for removes false
heading-derived entries and adds newly detectable list definitions on the same page, and a count
would report a document whose whole vocabulary was replaced as untouched.

**Detection switched off has explicit semantics, and this is the choice.** With
`rag.glossary.detect_on_ingest = false`, an ingest leaves a document's existing entries exactly
as they are — that is what the setting is for, and clearing them would be a silent erasure
arrived at through configuration — and records a fingerprint whose `detector` reads `disabled`.
That is a value in the column rather than an absence: `NULL` already means "never recomputed",
and if a disabled run wrote nothing the two states would be one. Turning detection back on
changes the installed fingerprint, so every document stamped this way is selected by the next
survey without anybody having to know to ask.

The command itself refuses while detection is off, naming the setting. Recomputing would run
rules the configuration says not to run; stamping the disabled state instead would erase the
record of which detector produced the entries still being served. The plan refuses too — under
a disabled detector the installed fingerprint *is* the disabled one, so the selection would be
every document a detector ever read, reported as outstanding work this command would not do.

**A detector failure fails closed.** `detect_entries` has no model to be unavailable, so it
raising means a bug here. Nothing is written: the entries a working detector produced stay
exactly where they are and stay servable, and the fingerprint is **not** advanced, so the
document remains selected by this command and reported by `doctor` until the fix ships. The rest
of the ingest is untouched — a glossary bug does not cost a working index — and the run names
the document, because a detector that has stopped working behind a screen of green counters is
the other half of failing silently.

### 10.3 Four lineages, four migrations, and how to tell which one you need

They are not interchangeable, and the price of each is the reason:

| Lineage | Moves when | Repaired by | Costs |
|---|---|---|---|
| `documents.parse_fp` | a parser's rules or one of its libraries changes | `document reindex --stale` | a parse from retained bytes, then an embed of whatever moved |
| `index_state.chunk_fingerprint` | the chunker, its budget, its tokenizer or a grammar changes | a re-index; the corpus-wide refusal is what stops mixing | a re-chunk and a re-embed of everything |
| `index_state.embed_fingerprint` | the model, its dimension or its normalization changes | `ingest.reindex.re_embed` | an embedding pass, no parsing |
| `documents.glossary_fp` | any detection or normalization rule changes, or a dependency of one does | `document reindex --stale-glossary` | a pass over stored text; **no GPU at all** |

Two properties follow from the table that are easy to get wrong in either direction.

**A parser bump is not a substitute for a detector bump.** It would migrate the media types that
happened to get one and leave every other stale, and it charges a corpus-sized parse and re-embed
for a change to a regular expression. It also cannot say *why* glossary rows changed, because the
fingerprint it moved describes a parser.

**A detector bump is not a reason to re-parse.** `glossary_fp` is a column on the row and
deliberately not a field on the domain `Document`, so ingest's change detection has no way to
consult it even by accident — which is what stops a corrected regular expression from making
every document in a corpus look like it needs its bytes read again.

**What requires a detector-version bump: nothing does, and that is the design.** The fingerprint
is a digest of `ingest/glossary.py` and `core/glossary.py` plus the versions of what they import,
so any edit to a rule, a threshold, an evidence weight, a written form or the normalization moves
it without anybody deciding to. #121 is the demonstration: it changed sentence-final punctuation
handling and NFKC-normalized `initial_skeleton`, needed no manual bump, and every document it
affects became selectable on the next survey.

**Migration policy for the first release.** Every existing row arrives with `glossary_fp IS
NULL`, which is selected. It is not backfilled: writing the installed fingerprint into rows
detected before anything recorded a detector would assert that they came out of the rules
installed now, which is false for every corpus indexed before the column and is exactly the
plausible falsehood the fingerprints exist to prevent. So the first run of this command after
upgrading sweeps the whole corpus — reading chunks, touching no network and no model — and every
run after it selects nothing.

**What the sweep reports, and why six numbers rather than three.** `selected` is the size of the
disagreement; `redetected` splits into `unchanged` and `changed`; and the three that are not
repairs are kept apart because they send an operator to three different places:

| Count | What happened | What to do |
|---|---|---|
| `failed` | the detector raised on this document's text | nothing — it is a defect here, and the document stays selected until it is fixed |
| `unrepairable` | the document's chunks are gone, so there is nothing to detect over | `document reindex --stale`, a rung up, then run this again |
| `superseded` | a sync committed newer chunks mid-recompute, so the write was declined | nothing — the corpus holds the newer state, and the document is selected again next run |

**A missing-chunks document is refused rather than stamped**, and that is the one of the three
worth stating twice. Detecting over no chunks returns no entries, which is a *well-formed*
derived result — so recording it would convert a missing-chunks problem into an invisible
empty-glossary one and take the document out of the selection permanently. Chunkless *by design*
is a different state: a document that yielded no extractable text really does state no
definitions, and recording that is correct. `Document.expects_chunks` is the discriminator, and
`re_embed` already uses it for the same distinction one rung up.

**A supersession is neither a failure nor a repair**, on exactly the reading §8.5 gives for the
re-parse sweep. This sweep takes no lock and shares none, because never reaching the model is the
point of it — so an entry's `chunk_id` foreign key is what notices: the rows cite chunks a sync
has just replaced, and the write is refused. It is told apart from a genuine failure *positively*
— the chunk ids are read again, and ids that have moved are what a sync leaves behind — rather
than by matching an error message, which would tie this to one store.

**Adding an input to the fingerprint re-stales the corpus, and that is the mechanism working.**
`libraries` was added after the column shipped, so a document stamped by the first release
disagrees with the installed detector and is selected again. It is not a regression and not a
second migration: the earlier fingerprint described fewer inputs than actually decide a stored
entry, so it was over-claiming, and a corpus that had been swept under it was current only by
the narrower definition. The cost is one more pass over stored text.

**Ordering, when both sweeps are wanted.** Run `--stale` first. A re-parse re-runs detection on
every document it rebuilds, so doing it second would redo work the glossary sweep had just
finished. Passing both flags in one invocation is refused rather than sequenced: they differ by
whether the machine spends an afternoon embedding, and finding that out from the elapsed time is
the one way nobody should have to find it out.

---

### 10.3 Durable whole-index re-embedding

```
manicule reembed plan
manicule reembed start RUN_ID       # choose and record this before invoking the command
manicule reembed execute RUN_ID     # `resume` is an alias
manicule reembed status RUN_ID      # `inspect` is an alias
manicule reembed abandon RUN_ID
manicule reembed cleanup RUN_ID
```

This workflow reads the authoritative SQLite document and chunk rows into an immutable,
keyset-readable snapshot. It never calls a connector, blob fallback, parser or chunker. `plan`
also makes **zero embedding forward passes**: it reports aggregate document/chunk/input counts,
bounded peak-memory and temporary-disk estimates, elapsed-time estimate, target dimension and a
one-way target identity. The transient plan snapshot is deleted before the command returns.
Source ids, URIs, snapshot/revision handles, weights paths, complete configuration and inventory
digests never cross an operator or network surface.

`start` performs the same exact-target plan, checks local temporary capacity, and atomically
persists the complete snapshot and an immediately acquirable, ownerless journal row under the
id the operator supplied **before embedding starts**. There is no create-then-release gap. If a
control reply is lost, replaying `start` with the same id returns the same run without taking a
second snapshot. A capacity refusal is typed, nonzero, and leaves neither an
unreachable run nor a retained transient snapshot. `execute`/`resume` acquires a new
monotonically fenced lease, builds a named shadow generation in bounded pages, checkpoints after
each page, independently inspects every retrieval-critical stored field, seals the exact digest,
then asks SQLite for one atomic pointer/fingerprint/inventory compare-and-swap. Live reads stay on
the old generation throughout the build. A process exit is recovered by running `resume` with
the same id; publication receipts make a publish-before-checkpoint retry return the original
winner rather than roll a newer one back.

Every snapshot scan failure and cancellation removes its unbound durable snapshot under a
cancellation-shielded cleanup. Every orderly execute failure or cancellation likewise releases
its owner/generation fence so a new process can retry or abandon immediately; a process that
actually dies runs no cleanup, so its lease still requires TTL expiry and a higher-generation
takeover. Publication writes its receipt and terminal checkpoint in the same SQLite transaction.
For installations upgraded from the earlier split checkpoint, the receipt remains authoritative:
status and resume report `published`, and abandon cannot turn the live run into `failed`.

The workflow currently requires the built-in SQLite/Lance backend. A custom vector store is
refused before a snapshot, model, or run is constructed unless it supplies the complete named
shadow-generation, inspection, atomic-publication and cleanup contract; ordinary `VectorStore`
methods are not enough to emulate that safely.

After a successful publication the runtime explicitly prepares its long-lived pointer-following
vector handle for the configured target. A handle still prepared for the old fingerprint refuses
searches and writes with `VectorStoreReprepareRequiredError`, including same-dimension changes;
it never silently applies an old embedder to the new generation. Cleanup obtains an exclusive
generation pin and only removes failed or superseded, non-live storage, so an in-flight reader
cannot lose its directory. `abandon` makes an unfinished run terminal without moving the live
pointer.

Snapshots, runs, generations and receipts are keyed by workspace as well as their opaque id, and
every journal, lease, publication and cleanup lookup repeats that ownership predicate. The vector
pointer is installation-wide, however, so its immutable snapshot and replacement generation cover
all workspaces; otherwise publishing Alpha would make Beta's vectors disappear. The public plan
and progress projection counts only the owning workspace, while private build accounting and
capacity checks cover the full installation. Legacy state is backfilled only when its stored
document payloads identify exactly one owning workspace (or an empty legacy database has exactly
one workspace); ambiguous state refuses migration rather than guessing ownership.

The serving scheduler runs one restart-recovery job for ownerless nonterminal runs in its active
workspace. It records aggregate recovered/failure counts, while the durable run remains the
authoritative status. Each run is isolated: one corrupt or transient refusal is counted by error
class and recovery continues with later runs, without logging ids, paths or messages. This is
process-crash recovery, not connector scheduling: no connector or parser is called, and a live
fenced owner is never taken over before lease expiry. The local CLI and authenticated admin HTTP
routes expose plan, start, resume, abandon and cleanup. `/ui/reembed` is read-only plan/status and
contains no mutation controls or JavaScript handlers. MCP retains the aggregate read-only
`reembed_status` tool. No network or browser surface lists opaque run ids.

## 11. `reconcile()` and deletion

Incremental sync cannot detect deletion, because a deleted document simply stops appearing —
which is why `reconcile` is a separate protocol method rather than an implementation detail
(`contracts.md` §3).

**Cadence.** After every full sync, and on a schedule (`reconcile_interval`, default weekly)
for connectors that only ever sync incrementally. Deletion detection that runs only when
someone remembers is deletion detection that does not run.

**Mechanics.** `reconcile()` yields the `SourceId`s that currently exist. Storage journals them
in pages under an immutable workspace, connector and configured-scope identity. Only iterator
exhaustion atomically marks that run complete and returns the typed completed-inventory handle
accepted by the diff. The diff is a server-side anti-join and its result is recorded as counts;
neither the source corpus nor the stored corpus is copied into a process-sized set. IDs only —
no bodies, no versions — which is what makes a weekly full enumeration affordable.

A crash after the completion transaction does not contact the source again: the next attempt
finds and applies that completed inventory. A partial or canceled run has no such handle and is
permanently ineligible. Incremental acquisition uses a different journal, so authentication,
capacity and cursor failures there cannot accidentally become reconciliation evidence.

### 11.1 The safety rule: never mass-delete on a partial enumeration

This is the failure that makes reconciliation dangerous, and it needs a mechanical guard rather
than care.

If `reconcile()` raises partway through — an expired cursor, a 429, a network drop — the IDs
seen so far are a *prefix*, not the truth. Diffing a prefix against the full stored set marks
everything not yet enumerated as deleted. One transient error, and the corpus is soft-deleted.

Three guards, all required:

1. **The diff is applied only on a storage-proved clean completion.** The apply transaction
   revalidates the handle's workspace, connector, scope, completion time and deduplicated count
   while holding SQLite's writer lock. A `reconcile` that raises produces no deletions at all.
   Partial results remain diagnostic rows, not deletion authority.
2. **A deletion ceiling.** If the diff would delete more than `reconcile_max_delete_fraction`
   (default 10%) of a connector's live documents, the pipeline refuses, records the proposed
   deletion, and surfaces it for confirmation. Candidate document revisions live in a relational
   proposal rather than an unbounded metadata list. Confirmation applies only those reviewed
   revisions; a document republished meanwhile is skipped. A newer full inventory invalidates
   the older question. A genuine bulk deletion is rare and worth a human; a bug that looks like
   one is not rare at all.
3. **Soft delete only.** Reconciliation never hard-deletes, so a mistaken reconciliation is
   recoverable by clearing `deleted_at` — free, within the grace period (`storage.md` §8.2).

Guard 3 is what makes guard 2 tunable rather than terrifying.

A dry-run consumes its completed inventory in the same transaction that computes the preview.
It cannot later be replayed as deletion authority; an applying run must obtain a fresh full
inventory. Proposal confirmation likewise requires the caller's current scope token and rejects
an omitted or mismatched token instead of trusting the proposal's copy. Each proposal candidate
captures the document's source-observation revision (version token plus a strictly advancing
`last_seen_at`). A later unchanged `record_seen` therefore invalidates the candidate just as a
new publication does: evidence that the source exists after the proposal wins over confirmation.

### 11.2 Interaction with the grace period and the sweep

`storage.md` §8.2 gives soft-deleted documents a grace period (default 30 days) before the
sweep purges their chunks, trading free restore against vector top-`k` dilution. The pipeline
owns when that sweep runs:

- **The sweep is scheduled, not triggered by deletion** — otherwise a large reconciliation
  produces a sweep storm during a sync.
- **It does not run during a backup.** The backup lock blocks exactly two things — this sweep
  and the blob GC — because they are the only two that remove data (`storage.md` §9.1).
- **It does not run during an active sync**, so an ingest run and a purge are never competing
  for the same vector table.
- **`doctor` reports the soft-deleted fraction**, which is the input to tuning the vector leg's
  over-fetch factor (`storage.md` §8.2).

A document that is restored inside the grace period needs no re-embed. Outside it, restore is a
re-parse from retained bytes — rung 3, still not a re-crawl.

`manicule.ingest.reindex.reindex_document` is that re-parse for one id, and it is what
`TrashStore.restore_document` points at when it reports `needs_reparse`. It resolves the id
through the store rather than taking a `Document`, and the lookup is workspace-scoped and skips
the trash — so **restore first, then reindex**. The other order finds nothing and says so,
rather than reporting a repair that did nothing. A document whose bytes were never retained
cannot take this path at all, and the restore says that too: for it, `sync --force` is the only
option, which is the one rung that can fail.

---

## 12. Watch mode

Local filesystem watching with **watchfiles** (`PLAN.md` §6). A watch event is not a sync, and
conflating them causes specific bugs.

| | Connector sync | Watch event |
|---|---|---|
| Trigger | Schedule or command | Filesystem notification |
| Scope | Everything since a watermark | One path |
| Watermark | Advanced on success | None |
| Reconcile | Yes, on the cadence in §11 | **No** |
| Deletion | Detected by reconcile | Reported directly by the event |

**Watch never drives reconciliation.** A watcher sees a subtree, and a subtree is exactly the
partial enumeration §11.1 refuses to diff on. Deletions arrive as explicit delete events, which
are trustworthy in a way that "did not appear in this walk" is not.

**Debounce, because editors do not write files the way the naive model assumes.** A single
logical save commonly produces several events: many editors write to a temporary file and
rename over the target; some truncate and rewrite in place, briefly presenting a zero-byte
file. Ingesting on the first event indexes a partial or empty document.

- Coalesce events per path over `watch_debounce` (default 500 ms).
- Ignore editor scratch patterns (`.swp`, `~`, `.tmp`, `.#*`, `4913`).
- **Re-`stat` after the debounce and skip if the file is gone** — the temp file from a
  rename-based save is created and removed inside the window.
- Treat a rename as delete-then-create by path, then let content-hash dedup (§4) collapse it
  back into a single unchanged document, because that is what it is.

**The initial walk is a sync, not a watch.** Starting a watcher on a directory that has never
been indexed performs a full enumeration first, with a watermark, and only then attaches the
watcher. Events arriving during the walk are queued rather than dropped.

**Backpressure applies unchanged.** A `git checkout` across a large repository produces
thousands of events in a second. They enter the same bounded queue as everything else (§8.3),
and the debounce collapses most of them before they get there.

---

## 13. Progress, resumability and cancellation

### 13.1 A run is a record, not a log line

A first sync of a large Confluence space is hours. It must survive being interrupted.

Each run records counters by terminal status and the machine-readable run outcome:
`complete`, `bounded`, or `incomplete`. The last-run metadata also records whether enumeration
completed, whether a watermark advanced, whether retry is required, and the typed failure
reason. This makes an incomplete source walk observable without inferring it from a missing
watermark or parsing a sentence.

Two records serve different purposes. `connectors.metadata.last_run` is the overwritten public
diagnostic summary. `acquisition_runs` and `acquisition_records` are relational correctness
state: committed source coverage, completion/candidate markers, leases and pending local work.
They are not inferred from the summary and may outlive the process that created them. Their
retention is governed by settlement rather than by diagnostic-history policy (`storage.md`
§4.1).

### 13.2 Enumeration durability and recovery are separate

An interrupted enumeration now leaves an explicit, committed prefix rather than only whatever
documents happened to publish before the process stopped:

1. Each source identity is committed before discovery advances past it.
2. The completion marker and candidate watermark exist only after true iterator exhaustion.
3. A crash before that point leaves the run visibly incomplete; a crash after it preserves the
   complete inventory and candidate.

Recovery is automatic at the production pipeline boundary. Selection and claim are one SQLite
write transaction: the newest safe unfinished run is authoritative, every other overlapping
legacy run is superseded and generation-fenced in that same transaction, or a new run is created
and claimed. Two simultaneous sync requests therefore cannot both observe absence and open
independent source cursors, and an upgraded database cannot retain an older live owner beside
the selected run. A live lease on the one authoritative run makes the second caller a clean
loser. An expired or orderly-released lease increments the generation; every later journal
mutation and every publication boundary is fenced by that generation. A source call left in
`acquiring` becomes retry work at takeover, while `indexing` remains the exact checkpoint needed
to replay partially expanded containers from retained bytes without source access.

An unchanged-token result is one fenced transaction too: it moves the journal record to
`unchanged` and refreshes the indexed document's `last_seen_at` under the same writer lock. A
takeover cannot land between durable coverage and presence bookkeeping. The public `last_run`
diagnostic is likewise written under the exact run generation; on an orderly return that write
and lease release are one transaction. A stale worker may return its own in-memory report, but
it cannot overwrite the successor's connector diagnostic.

The pipeline's renewal immediately before publication is only a fast failure and lease refresh;
it is not the publication lock. Every attempt-owned relational mutation starts its own SQLite
transaction with a conditional write that validates workspace, run, owner, generation, expiry,
unsettled state and non-supersession. That first statement takes the writer lock through the
document, status, annotation, lineage, glossary or tombstone mutation. A takeover therefore
commits before the guard and the stale write is refused, or waits until the guarded transaction
commits—there is no awaited check-then-write gap.

Enumeration, acquisition and indexing each run with an independent lease heartbeat. Their task
groups are children of the sync call, cancellation stops admission and joins every task within
the configured grace period, and no worker survives return. Different connectors hold different
run leases and can progress independently; every claim and record predicate remains workspace
scoped.

The retention policy is positive-state and deliberately asymmetric. Incomplete enumeration,
`retry`, `acquired`, and `indexing` records on the authoritative run are retained without an age
cutoff because they are recovery input. When a newer successful run has already advanced beyond
an older unfinished base watermark, recovery records `superseded_at` and the replacement run
id, then increments the older generation before starting from the current position. That older
backlog can never progress after fencing, so superseded history becomes eligible for bounded
30-day cleanup even when its records have active-state names. Settled journal history has the
same retention period, but a malformed settled run with active records is preserved. Deleting
eligible history releases only its acquisition references. Blob GC still
uses its full publication/version/acquisition mark set, so a blob is removed only after no live
publication, history row, acquisition record, or staging marker names it.

**Where the new watermark comes from, which this document did not say.** `Connector.discover`
*took* a position and nothing returned one, so a pipeline written to the protocol alone could
never advance it — and "the watermark advances only on a clean run" was a rule about a write
that never happened. Two workers found the same hole from opposite ends, and it is now
`Connector.watermark` in `contracts.md` §3: a read-only property answering how far the last
*completed* enumeration got, `None` when the source has no change signal or the enumeration did
not finish.

**Two checks guard that write and neither is redundant.** `assert_connector_contract` catches
a connector that advances its watermark as it yields — by abandoning a stream after one document
and requiring the position has not moved — and it catches it on an *uninterrupted* run, which is
the only kind anyone looks at. The pipeline's own gate catches a caller that persists a
watermark for work that was not committed, on a run that has already gone wrong. A connector can
pass the first and still lose documents through a caller that ignores the second.

Deleting either restores a failure whose symptom is documents that exist in the source, were
enumerated once, and are in no index — permanently, with nothing raised, and no later sync
fixing it. This is written down in both places on purpose: two tests that look like they overlap
are what this guarantee has to look like, and a reader who finds them without the reason will
delete one.

**The exception, stated:** a connector whose `discover` is not restartable from a watermark
re-enumerates fully. That is a connector property, not a pipeline one, and it is visible in
`doctor` rather than hidden.

### 13.2.1 What "clean" means once completion order is not discovery order

A watermark is **one value describing a whole enumeration**, so there is no partial version of it
to get wrong. That is exactly why it must not be written unless the whole enumeration landed:
under a staged run several documents are in flight at any moment, and "the last one finished"
says nothing about the first.

So the gate is three conditions, not one. Each is a different way for the connector's reported
position to describe more than what actually landed:

| Condition | Fails when | Why the position would lie |
|---|---|---|
| `clean` | `discover` raised | The enumeration is a prefix, and the position describes the whole |
| not `limited` | `--limit` stopped discovery early | Same, for a reason nothing went wrong |
| `unrecorded == 0` | A document reached no durable record at all | It exists at the source, was seen once, and is nowhere |

**`unrecorded` counts exactly one shape, and the narrowness is the point.** A fetch that failed
for a document the index had never seen writes nothing — there is no row to record the failure
against, so it is in no listing and no repair can select it. Advance the position past that and
no later sync enumerates it either.

A document that failed and **did** leave a row is a different case and does not hold the
watermark back. It is stored, countable, selectable by `document reindex`, and compared again by
the next sync. Holding the position for it would make one permanently broken document
re-enumerate a whole corpus for ever, which is a cost with no corresponding loss.

These same three conditions define the public outcome without changing the gate. A discovery
error or `unrecorded > 0` is `incomplete` and requires retry. A limit is `bounded` and is not an
unexpected failure. A full walk whose document failures all left durable rows is `complete`.
The application surfaces report `watermark_advanced` separately because a complete connector
may legitimately offer no watermark at all.

### 13.2.2 `--limit N` bounds acceptance

A sequential loop could count completions and stop; a staged run has several documents in flight
by the time the *N*th is accepted, so the bound is applied where discovery hands work downstream.

- At most *N* **top-level** documents are accepted. Members found inside a container are counted
  separately and never consume the limit — one archive of five hundred files must not exhaust a
  limit of ten.
- Everything accepted is carried to a terminal outcome before the run returns. Nothing is left
  abandoned in a hand-off.
- The run is reported as **bounded**: `clean` (nothing went wrong), `limited` (discovery stopped
  early), and therefore not `complete`. **No watermark is written.** `--limit 10` against ten
  thousand documents that advanced a position would skip the other nine thousand nine hundred
  and ninety on every subsequent sync, silently.

### 13.3 Cancellation

`Ctrl-C`, `manicule sync --stop`, or shutdown:

1. Stop pulling from `discover`. Cursors are abandoned, not persisted — they expire anyway
   (`confluence.md` §2).
2. Let accepted documents finish, up to `shutdown_grace_s` (default 30 s). A document mid-embed
   is close to done and finishing it is cheaper than redoing it.
3. Cancel whatever is still running at the deadline. Those documents stay non-terminal and are
   swept by §6.4.
4. **Do not advance the watermark.** A canceled run is an incomplete run.
5. **Do not run reconciliation.** §11.1's first guard, in its most likely form.

A second `Ctrl-C` skips step 2. The recovery path is the same either way, which is what makes
the impatient case safe.

**How step 1 brings the rest down.** Stopping discovery is the whole protocol rather than the
first of several: the fetch hand-off closes, its workers finish what they hold and exit on the
end-of-stream, closing the parse hand-off behind them, and the ingest workers drain what is
queued. The grace window is a bound on *that*, not a separate drain. So step 2 finishes what was
accepted rather than only what was mid-flight, which is the stronger of the two and costs
nothing extra.

**Nothing outlives the command.** The stages are children of the call that started them, in one
`asyncio.TaskGroup`, and every path — clean, failed, canceled, or canceled twice — joins them
before returning. A canceled run re-raises rather than returning a report, so there is no path on
which a watermark or a set of run counters is written for work that was interrupted.

**The source is not read during the drain.** A canceled sync that kept enumerating for the length
of the grace window would still be hammering a rate-limited API after somebody pressed `Ctrl-C`,
which is the opposite of what they asked for. At most one further document is produced by the
source and dropped — discovery reads the stop at the top of its loop, so an item already
in hand is abandoned rather than never fetched. Nothing is lost by that: no watermark was
written, so the next sync sees it again.

---

## 14. `doctor` — ingest checks

Complementing the storage checks (`storage.md` §10) and the parse checks (`parsing.md` §6.6).

There is no `doctor` command yet — it belongs to the CLI work, alongside the storage checks
(`storage.md` §10) and the parse checks (`parsing.md` §6.6), none of which have one either.
What the pipeline owes it is the *data*, and each row below names something already recorded
rather than something to be derived later: statuses and `updated_at` on `documents`, kill counts
by reason on the worker pool, `last_run` counters and `last_clean_reconcile_at` on
`connectors.metadata`, `proposed_deletion` where guard 2 fired, `original_omitted_reason` on
every document that has no retained bytes, and the lock file's holder.

| Check | Detects |
|---|---|
| Documents in a non-terminal status older than `stale_after` | A parent that died without a sweep (§6.4) |
| Worker kill counts, by reason, per media type | A parser hanging or leaking on a particular format |
| Fetch error rate per connector | Credential expiry, rate limiting, an endpoint change |
| Documents whose `last_seen_at` predates the last clean reconcile | Deletion detection not running |
| Time since last clean `reconcile` per connector | §11's cadence silently not happening |
| Proposed-deletion refusals awaiting confirmation | §11.1 guard 2 having fired |
| Documents with `original_ref IS NULL`, by reason | The set for which rung 4 is the only repair |
| Queue depth and stage occupancy during a run (`last_run.stages`, §8.3.2) | Where the bottleneck actually is; §8.3.3 reads it |
| `manicule.lock` holder | A second writer having been attempted (§6.5) |

---

## Appendix A: decisions this document made

Calls made in the absence of a stated position.

| Decision | Where |
|---|---|
| Middleware transforms; the return value is assigned and validated | §3.1 |
| An explicit may/may-not list, with no store handle for middleware | §3.2 |
| Text-mutating middleware is a `ChunkFingerprint` input, defaulting to opt-out | §3.3 |
| The complete `Document.status` set and its transitions, collected for #1 | §2.2 |
| PII redaction moves to the generation boundary; ingest-time redaction rejected | §3.4 |
| Two-level change detection, both levels conditioned on current parse lineage | §4, §4.1 |
| Parse in `spawn`ed worker subprocesses; embed deliberately not (chunk moved to the parent, §2) | §6 |
| Memory bounding is platform-split: `RLIMIT_AS` on Linux, RSS polling on macOS | §6.2 |
| Crash recovery is a startup sweep on `status` + `updated_at`, no new schema | §6.4 |
| One writer per data directory, by a lock file every writer acquires; readers take nothing | §6.5, §8.6 |
| Every refusal runs once per run, before discovery, plus a budget/context cross-check | §7 |
| Embed batch size derived from both fingerprints, not a constant | §8.2 |
| Commit discovery to a journal before bounded local processing, so local backpressure cannot expire cursors | §8.3 |
| A failed re-ingest never demotes a working document | §9 |
| Three re-ingest verbs mapped to ladder rungs; re-parse is first-class | §10 |
| The corpus-wide re-parse is a flag on the document verb, and command line only | §10.1 |
| Reconcile: clean-completion-only, a deletion ceiling, soft delete only | §11.1 |
| The sweep is scheduled and yields to backup and sync | §11.2 |
| Watch never reconciles; debounce with a post-debounce re-`stat` | §12 |
| Enumeration coverage is journaled; recovery/takeover remains separate orchestration | §13 |

Decisions the implementation added, each argued where it appears:

| Decision | Where |
|---|---|
| Resident memory is the enforced quantity on every platform; `RLIMIT_AS` is a looser backstop | §6.2 |
| Chunking runs in the parent, because `after:parse` is a plugin hook | §2 |
| Change detection needs its own allowlist (`SETTLED`), for the mirror of §6.4's reason | §2.2 |
| A terminal *conclusion* replaces an indexed document; a *failure* never does | §9 |
| In-flight statuses are written only for a document with nothing servable to lose | §9 |
| Container expansion is a parse attempt, decided in the worker that holds the parser | §2 |
| `middleware` is a `failed_stage` value, positioned after the six stages | §2.2 |
| A connector reports its position through `Connector.watermark`, asked only on a clean run | §13.2 |
| Retention happens before any hook, so retained bytes are the connector's and `content_hash` describes them | §4.2 |
| Members of a container are counted apart from discovered documents, so one archive cannot exhaust a `--limit` | §13.1 |
| A run is three stages and two bounded hand-offs; the ingest stage is `parse_workers + 1` wide | §8.3 |
| Level-1 change detection belongs to the fetch stage, so a skip never costs a fetch or blocks paging | §8.3 |
| An end-of-stream is not subject to the queue bound, because closing a stage happens during teardown | §8.3.1 |
| A watermark needs a *complete* run, not merely a clean one: not limited, and nothing left unrecorded | §13.2.1 |
| `--limit` bounds acceptance rather than completion, and reports the run as bounded | §13.2.2 |
| Cancellation stops discovery and lets that close the stages in order; the grace window bounds it | §13.3 |

## Appendix B: filed, not deferred

| Ticket | What | Why not here |
|---|---|---|
| [#28](https://github.com/mgd43b/manicule/issues/28) | **Refuse-to-ingest content rules** (§3.4) | The coherent version of the requirement PII redaction was misfilling. Needs a policy language and a decision about what happens to already-ingested matches; neither is an ingest-pipeline question |

---

## Appendix C: checklist against ticket #5

- **Middleware hooks at each stage** — §3, with the transform contract, the capability list,
  and the fingerprint consequence.
- **Parser fallback chains** — §5 for enforcement; `parsing.md` §6 for semantics.
- **Content-hash dedup; re-ingest only what changed** — §4, two levels, plus what a skip must
  still write.
- **Per-document error status; one bad document never aborts a batch** — §6, specified for
  raise, hang, OOM, and parent death.
- **Local directory watch** — §12, including why a watch event is not a sync.
- **Re-ingest from retained originals as a first-class operation** — §10, three verbs against
  the blast-radius ladder.
