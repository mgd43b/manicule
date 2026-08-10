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

## 1. Division of labour

| Concern | Owned by | This document |
|---|---|---|
| Parser semantics, fallback outcomes, `Document.status` | `parsing.md` §6 | Runs the chain; enforces the limits it names |
| Chunk budget, `ChunkFingerprint` | `parsing.md` §1 | Checks it once, before any work |
| Write order, crash windows, invariants | `storage.md` §8 | Honours it; adds lease recovery |
| `EmbedFingerprint`, the refusal | `storage.md` §6.3 | Runs it once, before any embedding |
| Retained bytes, blob ordering | `storage.md` §7 | Writes them; re-parses from them |
| Deletion sweep, soft-delete grace | `storage.md` §8.2 | Drives `reconcile`; schedules the sweep |
| Connector protocol | `contracts.md` §3 | Calls it; never interprets its tokens |

Everything above is cited rather than restated. Where this document appears to disagree with
one of them, that is a bug in this document.

---

## 2. The shape

**The unit of work is one document. A batch is a scheduling artefact with no semantics of its
own.** This is what makes "one bad document never aborts a batch" a structural property rather
than a promise: there is no batch-level transaction to abort, and no batch-level state a
document can corrupt.

| Stage | Input | Output | Where it runs | Bounded by |
|---|---|---|---|---|
| discover | `Watermark \| None` | `DiscoveredDoc` stream | async, in-process | connector page size |
| fetch | `DocRef` | `RawDocument` | async, in-process | HTTP timeout, size cap |
| parse | `RawDocument` | `ParsedBlock` stream | **worker subprocess** | wall clock + memory |
| chunk | `ParsedBlock` stream | `Chunk` list | worker subprocess | token budget |
| embed | `Chunk` list | vectors | **in-process** | batch size |
| store | chunks + vectors | committed document | in-process | SQLite transaction |

Two things in that table are load-bearing and are argued in §6: **parse and chunk run in a
subprocess, and embed does not.**

### 2.1 Discover is not fetch

`Connector.discover` yields `DiscoveredDoc` — identity, `version_token`, and enough metadata to
decide whether to fetch at all. Fetching the body during discovery would make the change check
(§4) pointless, because the expensive part has already happened.

This is why `discover` takes a watermark and `fetch` takes a `DocRef`
(`contracts.md` §3): they are separately schedulable, and the gap between them is where the
skip decision lives.

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
fails the document with `middleware_failed` and names the hook. Silently substituting the
original input would resurrect exactly the bug above.

### 3.2 What a middleware may and may not do

| May | May not |
|---|---|
| Transform the value it is given | Hold a `DocStore` or `VectorStore` handle |
| Read configuration | Write to either store directly |
| Emit events and metrics | Mutate another document |
| Annotate `Document.metadata` / `Chunk.metadata` | Set or alter an `Anchor` |
| Skip a document by raising `SkipDocument(reason)` | Alter `chunks.id`, or anything it derives from |
| Declare itself a text mutator (§3.3) | Perform network I/O without a declared timeout |

**No store handle** is the important one. A middleware that writes to the stores bypasses the
write ordering in `storage.md` §8.2, and the ordering is what makes crash recovery possible.
Middleware operates on values in flight; the pipeline alone commits.

**No anchor mutation.** Anchors are locked once ingest runs (`contracts.md` §1) and a
middleware that adjusts them invalidates stored citations with no record of having done so.

**Failure semantics.** A middleware that raises fails *that document*, with status
`middleware_failed` and the hook name in `metadata`. It does not abort the batch and it does not
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
one silently corrupts the space. The default is therefore `True` for any hook at
`after:parse`, `before:chunk` or `after:chunk`, and a plugin must opt *out*.

### 3.4 PII redaction, decided

`PLAN.md` defect #5 says pick one behaviour. **Redaction happens at the generation boundary,
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

Two levels, cheapest first.

```
1. version_token differs?   no  -> skip, record liveness, done
2. content_hash differs?    no  -> skip, record liveness, update version_token
3. otherwise                    -> full ingest
```

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

**3. A parser that is killed is a hard failure, not a lost document.** This is §6.

> **Prior art.** `parseWithFallback` wraps each attempt in `try/catch` and advances on a thrown
> exception or empty output. There is no time limit and no memory limit, so the two failure
> modes that actually take down an ingest run — a parser that hangs and a parser that
> allocates without bound — are not failures at all: they are the process stopping. And when
> every parser has been tried it throws `No parser found`, which the outer handler turns into
> status `error`, so `no_extractable_text`, `unsupported_media_type` and `parse_failed` all
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
(BSD) dependency already implied by the hardware detection in `PLAN.md` §16.

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
  worker, amortised over the run.
- **Workers are recycled after `max_documents_per_worker` (default 500)** to bound the effect
  of leaks in native parser libraries, which is a category of bug no amount of care in
  manicule prevents.
- **A killed worker is replaced immediately**; the pool size is what the run depends on, not
  the identity of any worker.
- **Workers hold no store handles.** They receive bytes and return blocks. Everything
  transactional happens in the parent, which is what keeps `storage.md` §8.2's ordering true
  regardless of how many workers died.

### 6.4 Attributing a death to a document

When a worker dies, the parent knows which document it dispatched, marks it `parse_failed` with
`reason="worker killed: timeout"` or `"worker killed: memory limit"`, and continues. The batch
is unaffected because the batch was never a unit.

**When the parent dies**, in-flight documents are left in a non-terminal status. Recovery is a
startup sweep, and it needs no new schema — `status` and `updated_at` already exist:

```sql
UPDATE documents
   SET status = 'pending', error_message = 'interrupted; requeued'
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
- **Memory is bounded by batch size**, derived from the fingerprint rather than hardcoded
  (§8.2).
- **Input is not attacker-shaped by the time it arrives.** Parse and chunk have already run,
  in a sandbox, and produced bounded token counts.

Residual risk, stated rather than hidden: an OOM inside MLX takes the process down, and §6.4's
sweep is what recovers it. That is a worse outcome than a killed parse worker, and it is the
price of an in-process embedder.

> **Prior art.** Everything runs in one process, `BATCH_SIZE` is the constant `32` regardless of
> model or chunk size, and contextual-retrieval and chunk-augmentation features issue two LLM
> calls *per chunk* inside the ingest loop with no concurrency bound and no isolation — so an
> LLM timeout on chunk 400 of a document fails the whole document, and a slow endpoint stalls
> the run with no signal distinguishing it from a large corpus.

---

## 7. The two refusals

Both are specified elsewhere; the pipeline is what runs them. Both run **once per run, before
the first document is discovered**, and both are hard refusals.

```
1. EmbedFingerprint   config vs index_state vs _manicule_meta   (storage.md §6.3)
2. ChunkFingerprint   config vs index_state                      (parsing.md §1.7)
3. budget_tokens <= max_sequence_length                          (parsing.md §1.1)
```

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
| embed | **Memory and one accelerator** | serialised; one batch at a time |
| store | SQLite single writer | serialised |

**Embedding is serialised, and this is the difference an in-process embedder makes.** With a
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

### 8.3 Backpressure is a bounded queue

Stages are connected by bounded queues (`asyncio.Queue`, `maxsize = 2 × the consumer's
parallelism`). When embed falls behind, the chunk queue fills, parse workers block on `put`,
fetch blocks behind them, and discovery stops pulling pages.

That last consequence is the point. Unbounded queues turn a slow embedder into unbounded
memory growth *and* cause a subtler failure: a connector that keeps enumerating will exhaust
its cursors. Confluence search cursors expire (`confluence.md` §2), so a sync that races ahead
during a slow embed can fail pagination partway through — a failure that looks like a connector
bug and is actually missing backpressure.

> **Prior art.** The sync loop bounds concurrency with a `Set` of in-flight promises and
> `Promise.race`, which is a queue with no backpressure onto discovery: `discover()` is pulled
> as fast as tasks retire. It also decrements the in-flight set in a `.finally` that is not
> ordered against the `race` resolution, so the effective concurrency can drift above the
> configured limit. A semaphore acquired before dispatch and released after has neither
> property.

---

## 9. Storing

The write order and its crash windows are `storage.md` §8.2 and are not restated. The pipeline
adds two rules.

**A failed re-ingest must not demote a working document.** If a document is `indexed` and a
re-ingest fails at any stage, it stays `indexed` with its existing chunks and vectors, and the
failure is recorded in `error_message` and `metadata.last_ingest_error`. The new status is
applied only when there is something to replace the old content with.

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
`document reindex --status no_extractable_text` the day OCR lands.

---

## 10. Re-ingest from retained originals

`PLAN.md` §4 asks for re-ingest against a pinned corpus rather than a re-crawl. With
`storage.md` §7's retained bytes that becomes three distinct operations, each landing on a rung
of the blast-radius ladder (`storage.md` §1):

| Command | Reads | Rung | Network |
|---|---|---|---|
| `reindex --repair` | `chunks` | 1–2 | none |
| `reindex --re-embed` | `chunks.embed_text` | 2 | none |
| `reindex --re-parse` | `blobs` | 3 | none |
| `sync --force` | the source | 4 | yes, rate-limited, **may fail** |

**Only the last one can fail for reasons outside the machine**, and it is the only one that is
not reproducible. Everything above it is a pure function of what is already on disk. That is
the whole return on retaining bytes, and it is why `--re-parse` is a first-class verb rather
than a flag on sync.

**Selection is a query, not a scan**, because of the per-document lineage in `storage.md` §6.4:

```sql
-- a tree-sitter grammar upgrade invalidates code documents and nothing else
SELECT id FROM documents WHERE chunk_fp <> :current AND media_type IN (:code_types)
```

`--re-parse` accepts the same selectors as `document list` — `--status`, `--connector`,
`--media-type`, `--container`, `--chunk-fp` — so "re-parse everything that came out of the old
PDF parser" is expressible without a bespoke flag.

**Re-parse is subject to the same identity rules as first ingest.** It runs the current parser
chain over the retained bytes, produces chunks, and reconciles them against the stored set by
`chunks.id`. Unchanged chunks keep their ids and their vectors (`storage.md` §3.2), so a parser
fix that changes one table in a hundred-page document re-embeds one table.

**A document with no retained bytes cannot be re-parsed**, and the command says so per document
rather than failing the run: `original_ref IS NULL` because retention was capped or the document
predates retention. Those are listed with their `original_omitted_reason` and are the only
documents for which `sync --force` is the only option.

---

## 11. `reconcile()` and deletion

Incremental sync cannot detect deletion, because a deleted document simply stops appearing —
which is why `reconcile` is a separate protocol method rather than an implementation detail
(`contracts.md` §3).

**Cadence.** After every full sync, and on a schedule (`reconcile_interval`, default weekly)
for connectors that only ever sync incrementally. Deletion detection that runs only when
someone remembers is deletion detection that does not run.

**Mechanics.** `reconcile()` yields the `SourceId`s that currently exist. The pipeline diffs
against stored documents for that connector and soft-deletes the difference. IDs only — no
bodies, no versions — which is what makes a weekly full enumeration affordable.

### 11.1 The safety rule: never mass-delete on a partial enumeration

This is the failure that makes reconciliation dangerous, and it needs a mechanical guard rather
than care.

If `reconcile()` raises partway through — an expired cursor, a 429, a network drop — the IDs
seen so far are a *prefix*, not the truth. Diffing a prefix against the full stored set marks
everything not yet enumerated as deleted. One transient error, and the corpus is soft-deleted.

Three guards, all required:

1. **The diff is applied only on a clean completion.** A `reconcile` that raises produces no
   deletions at all. Partial results are discarded, not salvaged.
2. **A deletion ceiling.** If the diff would delete more than `reconcile_max_delete_fraction`
   (default 10%) of a connector's live documents, the pipeline refuses, records the proposed
   deletion, and surfaces it for confirmation. A genuine bulk deletion is rare and worth a
   human; a bug that looks like one is not rare at all.
3. **Soft delete only.** Reconciliation never hard-deletes, so a mistaken reconciliation is
   recoverable by clearing `deleted_at` — free, within the grace period (`storage.md` §8.2).

Guard 3 is what makes guard 2 tunable rather than terrifying.

### 11.2 Interaction with the grace period and the sweep

`storage.md` §8.2 gives soft-deleted documents a grace period (default 30 days) before the
sweep purges their chunks, trading free restore against vector top-`k` dilution. The pipeline
owns when that sweep runs:

- **The sweep is scheduled, not triggered by deletion** — otherwise a large reconciliation
  produces a sweep storm during a sync.
- **It does not run during a backup**, which is the one thing the backup lock blocks
  (`storage.md` §9.1).
- **It does not run during an active sync**, so an ingest run and a purge are never competing
  for the same vector table.
- **`doctor` reports the soft-deleted fraction**, which is the input to tuning the vector leg's
  over-fetch factor (`storage.md` §8.2).

A document that is restored inside the grace period needs no re-embed. Outside it, restore is a
`--re-parse` from retained bytes — rung 3, still not a re-crawl.

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

Each run records id, connector, start/end, watermark before and after, and counters by outcome
(`discovered`, `skipped_version`, `skipped_hash`, `indexed`, plus one per terminal status). The
counters are what the summary line in `parsing.md` §6.5 is built from, and what makes the
fallback-rate signal in §6.6 computable.

**This needs no new table.** `connectors.status`, `error_message`, `last_synced_at` and
`watermark` already exist (`storage.md` §4.7); per-run counters are `metadata` on that row.
Resisting a `runs` table is deliberate — run history is diagnostic, not relational, and a table
that only ever grows needs a retention policy nobody has asked for.

### 13.2 Resume is a consequence of the design, not a feature

Resumability is already paid for by three decisions made for other reasons:

1. **The watermark advances only on a clean run**, so an interrupted sync re-enumerates from
   the last good point rather than from the beginning.
2. **Change detection (§4) makes re-enumeration cheap** — already-ingested documents skip at
   level 1 without a fetch.
3. **`documents.status` is the per-document progress marker**, and §6.4's sweep requeues
   anything caught in flight.

So resume is: run it again. There is no checkpoint file, no resume token, and nothing to
corrupt. The only cost is re-enumeration, which is exactly the cost the watermark exists to
bound.

**The exception, stated:** a connector whose `discover` is not restartable from a watermark
re-enumerates fully. That is a connector property, not a pipeline one, and it is visible in
`doctor` rather than hidden.

### 13.3 Cancellation

`Ctrl-C`, `manicule sync --stop`, or shutdown:

1. Stop pulling from `discover`. Cursors are abandoned, not persisted — they expire anyway
   (`confluence.md` §2).
2. Let in-flight documents finish, up to `shutdown_grace` (default 30 s). A document mid-embed
   is close to done and finishing it is cheaper than redoing it.
3. Kill parse workers still running at the deadline. Their documents stay non-terminal and are
   swept by §6.4.
4. **Do not advance the watermark.** A cancelled run is an incomplete run.
5. **Do not run reconciliation.** §11.1's first guard, in its most likely form.

A second `Ctrl-C` skips step 2. The recovery path is the same either way, which is what makes
the impatient case safe.

---

## 14. `doctor` — ingest checks

Complementing the storage checks (`storage.md` §10) and the parse checks (`parsing.md` §6.6).

| Check | Detects |
|---|---|
| Documents in a non-terminal status older than `stale_after` | A parent that died without a sweep (§6.4) |
| Worker kill counts, by reason, per media type | A parser hanging or leaking on a particular format |
| Fetch error rate per connector | Credential expiry, rate limiting, an endpoint change |
| Documents whose `last_seen_at` predates the last clean reconcile | Deletion detection not running |
| Time since last clean `reconcile` per connector | §11's cadence silently not happening |
| Proposed-deletion refusals awaiting confirmation | §11.1 guard 2 having fired |
| Documents with `original_ref IS NULL`, by reason | The set for which rung 4 is the only repair |
| Queue depth and embed throughput during a run | Where the bottleneck actually is |
| `manicule.lock` holder | A second instance having been attempted (§6.5) |

---

## Appendix A: decisions this document made

Calls made in the absence of a stated position.

| Decision | Where |
|---|---|
| Middleware transforms; the return value is assigned and validated | §3.1 |
| An explicit may/may-not list, with no store handle for middleware | §3.2 |
| Text-mutating middleware is a `ChunkFingerprint` input, defaulting to opt-out | §3.3 |
| PII redaction moves to the generation boundary; ingest-time redaction rejected | §3.4 |
| Two-level change detection, and a skip that still writes three things | §4, §4.1 |
| Parse and chunk in `spawn`ed worker subprocesses; embed deliberately not | §6 |
| Memory bounding is platform-split: `RLIMIT_AS` on Linux, RSS polling on macOS | §6.2 |
| Crash recovery is a startup sweep on `status` + `updated_at`, no new schema | §6.4 |
| One instance per data directory, enforced by a lock file | §6.5 |
| Both refusals run once per run, before discovery, plus a budget/context cross-check | §7 |
| Embed batch size derived from both fingerprints, not a constant | §8.2 |
| Bounded queues, so backpressure reaches discovery and cursors do not expire | §8.3 |
| A failed re-ingest never demotes a working document | §9 |
| Three re-ingest verbs mapped to ladder rungs; `--re-parse` is first-class | §10 |
| Reconcile: clean-completion-only, a deletion ceiling, soft delete only | §11.1 |
| The sweep is scheduled and yields to backup and sync | §11.2 |
| Watch never reconciles; debounce with a post-debounce re-`stat` | §12 |
| Resume needs no checkpoint; run counters live in `connectors.metadata` | §13 |

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
