# Retrieval

Design for the retrieval pipeline: two legs, a fusion, a rerank, and a context assembled
inside a token budget. Ticket [#6](https://github.com/mgd43b/manicule/issues/6).

The stores are settled and built ([`storage.md`](storage.md)), the embedder is settled
([`embeddings.md`](embeddings.md)), and the seam every stage is written against is settled
([`contracts.md`](contracts.md) §3). This document starts above all of them: given a query
and those stores, what runs, in what order, what each step is allowed to see, and what the
result is allowed to claim.

> **Prior art.** Clearly-marked callouts like this one record a design manicule considered and
> rejected, and say what the rejection buys. They are here because a decision is only legible
> beside the alternative it ruled out. Every claim in this document stands on manicule's own
> behavior and is checkable against this repository.

---

## 1. The whole design in five sentences

**A pipeline is a declared list of uniform stages. Every stage's output is already live,
in-workspace and visible. Fusion sees ranks and never scores. Confidence describes the
retrieval, not the answer. Nothing new ships without a measured improvement on
[#15](https://github.com/mgd43b/manicule/issues/15).**

Everything below follows from those five, and the reason they are five rather than four is
the second one. It would have been cheaper to enforce the workspace and soft-delete boundary
once, at the end, just before results reach a caller. That design has a failure mode that
this project has already hit once in the lexical leg and measured: the filtering happens
*after* the ranking, the excluded rows have already consumed the top-`k` slots, and the query
returns a well-formed empty list ([`storage.md`](storage.md) §6.1). Enforcing at the end is
correct and useless. Enforcing at every stage boundary makes it an invariant that can be
asserted at any point in the pipeline, which is what turns "we filtered" into something a test
can fail on.

---

## 2. The pipeline

### 2.1 The shape

```
              Query
                │
        ┌───────▼────────┐
        │ router         │   deterministic; greetings and utility queries stop here
        └───────┬────────┘
                │ RETRIEVE
        ┌───────▼────────┐
        │ L1 cache       │   ranked chunk ids, keyed by generation; hit → hydrate → done
        └───────┬────────┘
                │ miss
   ─────────────┼─────────────────────────────  declared stages, in configuration order
        ┌───────▼────────┐
        │ dense          │   embed → LanceDB k′ → hydrating join → k live candidates
        ├────────────────┤
        │ lexical        │   FTS5 BM25, one joined statement, merged into the list
        ├────────────────┤
        │ rrf            │   per-leg rank ladders → reciprocal rank fusion
        ├────────────────┤
        │ rerank         │   cross-encoder over the head; profile-gated
        └───────┬────────┘
   ─────────────┼─────────────────────────────
        ┌───────▼────────┐
        │ assembly       │   token budget, whole passages only → Context
        ├────────────────┤
        │ confidence     │   a statement about the retrieval
        └───────┬────────┘
                ▼
        Context + Confidence + RetrievalTrace
```

The declared part is `rag.pipeline`, which ships as `("dense", "lexical", "rrf")` with the
reranker appended when the profile asks for one and configuration names one
(`manicule.container.Container.retrieval_pipeline`). Everything outside the fenced region is
not a stage, and §2.4 says why that is not a loophole.

### 2.2 A stage is a fold, and the runner is the only thing that times it

`RetrievalStage.run(query, candidates) -> list[Candidate]` is a fold over a list. The runner
holds the accumulator, calls each stage in declared order, and does three things no stage can
do for itself:

1. **Times it**, with `time.perf_counter()` around the `await`.
2. **Counts** what went in and what came out.
3. **Records** the stage's declared configuration, so a recorded result names the thing it
   measured.

Timing from outside is not merely convenient, and the reason is not that a stage would measure a
different interval — wrapping its own body in `perf_counter` would measure almost the same one.
It is that a self-reported number is **unverifiable and optional**. Every stage gets timed
identically, including third-party plugin stages whose authors would never think to instrument
themselves; a stage cannot under-report by forgetting or by measuring only the part it considers
its own work; and there is nothing to compare a self-reported figure against. Timing is the
runner's concern rather than the stage's, which is the same reasoning that keeps it out of the
signature (§2.3).

**Stages run sequentially.** The dense and lexical legs touch different stores and could run
concurrently, and the saving is real but small: the lexical leg is one SQLite statement, and
the dense leg contains an embedding forward pass that dominates it by an order of magnitude,
against a reranker that dominates both. Concurrency would buy a few milliseconds and cost
unambiguous per-stage attribution — overlapping wall times do not sum to a pipeline latency,
and the whole point of recording them is that #15 can subtract. If #15 ever shows leg latency
matters, the shape to reach for is a combinator stage that runs children concurrently and
marks their spans as overlapping in the trace. That is a stage, so it needs no widening.

**Stage names are unique within a pipeline, and the container refuses a duplicate.**
`Candidate.scores` is keyed by stage name; two stages sharing one means the second silently
overwrites the first's record, and the fused ranking is then computed from a ladder that is
missing half its rungs.

### 2.3 `RetrievalStage` is not widened

Ticket #1 widened it once, from sync to async, and rejected three further widenings. This
document re-argues all three against a working design and adds the one new pressure that
building the design surfaced. **The conclusion is that it stays exactly as it is:**

```
RetrievalStage
    name: str
    async def run(query: Query, candidates: list[Candidate]) -> list[Candidate]
```

**Shared state between stages, for the query embedding.** Still rejected. Both the dense stage
and any future stage needing the query vector call the embedder; the embedding cache is keyed
by the canonical fingerprint and the exact text ([`embeddings.md`](embeddings.md) §8), so the
second call is a dictionary lookup. The cost of the alternative is that stage *n* depends on
stage *n−1* having populated something, and a pipeline whose stages cannot be reordered or
removed independently is not a pipeline #15 can attribute anything to.

**Timing and diagnostics on the return value.** Still rejected for timing — §2.2 — but the
*diagnostics* half is where a real pressure appeared, and it deserves a straight answer rather
than a restatement. The dense stage genuinely knows things the runner cannot infer: how many
rows it over-fetched, how many survived the join, how many expansions it needed, whether it
exhausted the table. Those are exactly the numbers §4.4 needs recorded. Three ways to get
them out:

| Option | Why not |
|---|---|
| Widen the return to `(candidates, StageReport)` | Every stage now returns a tuple, including the ones with nothing to report, and every recorded result predating the change is unreplayable. This is the widening the warning in `contracts.md` §3 exists to prevent |
| An optional `drain_report()` on the stage | Stages are container singletons shared across concurrent queries. Per-run state on a shared object is a race, and the failure is two queries swapping diagnostics — plausible-looking numbers attached to the wrong run |
| A `contextvars.ContextVar` trace frame, installed by the runner | **Chosen.** Per-task by construction, so concurrent queries cannot cross; invisible to stages that ignore it; no signature changes |

The contextvar is implicit coupling and that is a real cost, paid down by one rule and one
test: **nothing in the pipeline's behavior may depend on the trace frame**. `assert_retrieval_stage_contract`
installs no frame, so a stage that only works while someone is recording fails there — but the
sharper failure is a stage that behaves *differently* under one, and that needs both runs. So
each shipped stage is run twice over identical input, once observed and once not, and must
produce the same candidates; a stage written to misbehave demonstrates first that the
difference is detectable at all. A check that has never seen a failure is not evidence.

**A stage context object.** Still rejected, and the reasoning is unchanged: this is the
widening that never gets narrowed again. Everything proposed for it in this design has landed
somewhere better — the filter is on `Query`, the profile is on `Query`, per-leg scores are on
`Candidate.scores`, and diagnostics go to the trace frame.

**One thing the narrow signature forced, and it turned out to be an improvement.** Fusion needs
each leg's *rank ladder*, and a flat `list[Candidate]` does not obviously carry one. §5.2 shows
it does. Working that out produced a fusion stage that is configured with the names of the legs
it fuses rather than hardcoding `"dense"` and `"bm25"` — which is precisely what lets the
learned-sparse leg (§13) be measured against BM25 by editing configuration.

### 2.4 What "independently switchable" means, and the three things that are not switchable

#15's whole method is comparing two pipelines that differ in exactly one place. That requires
switching to be *declarative*: `rag.pipeline` is a tuple of names in configuration, the
reranker is a name or `None`, and every numeric knob lives in `ProfileConfig` with per-field
overrides. No comparison in #15 should ever require editing code, and if one does, that is a
defect in this design rather than in the harness.

Three things are deliberately **not** switchable, and calling them stages would have made them
look switchable:

- **The hydrating join is inside the dense stage, not beside it.** It is not a quality feature;
  it is the workspace, soft-delete and status boundary (§4.2). A pipeline that could be
  configured without it is a pipeline that can be configured into a cross-tenant leak, and
  configuration is not where a security boundary should live. Folding it into the leg also buys
  the §1 invariant: *every* stage's output is already scoped, so the assertion holds everywhere
  rather than at one privileged point.
- **Context assembly is not a stage.** It emits `Context`, not `list[Candidate]`. This closes
  the second question in `contracts.md` §6 the same way the merged `Context` docstring already
  leans: keeping the types distinct is exactly why a stage list is freely reorderable and this
  step is not. A stage that emitted a different type would make every stage's signature a union.
- **Confidence is not a stage.** It reads the assembled context and the trace and produces
  neither candidates nor context. It is a report on the run.

---

## 3. `Filter` — settled

`contracts.md` §6 has carried this as open since #1: *"a LanceDB predicate plus a metadata
pre-filter, but the exact split between them wants contact with real data volumes."*
[`storage.md`](storage.md) §6.6 proposed a shape and explicitly declined to close it. Storage
is now built and merged, and the shape that shipped in `manicule.core.retrieval` differs from
both. **This section closes it.**

### 3.1 The settled shape

```python
class Filter(BaseModel):
    """A restriction on which chunks a search may return."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_ids: frozenset[str]  # required, non-empty
    document_ids: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    collection_ids: frozenset[str] = frozenset()
    tag_ids: frozenset[str] = frozenset()
    media_types: frozenset[str] = frozenset()
    kinds: frozenset[BlockKind] = frozenset()
    langs: frozenset[str] = frozenset()
    updated_after: datetime | None = None
    updated_before: datetime | None = None
```

Every field is a conjunct; within a field, membership is a disjunction; an unset field
restricts nothing. That part is unchanged from what shipped. Four things change.

**`workspace_id: str | None` becomes `workspace_ids: frozenset[str]`, required and non-empty.**
§3.2.

**`source: str | None` becomes `sources: frozenset[str]`.** A scalar among seven set-valued
fields is a drafting accident, and "these two connectors" is an ordinary query. The name follows
the merged vocabulary — `Document.source`, `DocStore.find_document(source, source_id)` — rather
than `storage.md` §6.6's `connector_ids`, because a filter field that does not match the column
it filters is a field people get wrong.

**`langs` is added.** The Lance table already promotes a `lang` column
([`storage.md`](storage.md) §6.2) that no filter field can currently reach. A promoted column
nothing can name is dead weight in every row of the index, and language is a real query
restriction on a corpus the embedder was chosen to handle in 100+ languages.

**`extra: dict[str, JsonValue]` is removed.** It shipped so that "a store can accept a predicate
this type cannot yet express without anyone widening it prematurely." With the shape settled it
has no remaining job, and it should go rather than linger, for three reasons that compound:

- It is an untyped predicate channel on the type that carries a security boundary. A field whose
  meaning is whatever the store decides cannot be reviewed, and this is the one type where a
  reviewer must be able to see the whole restriction.
- It is unusable today anyway. `predicate_for` in `manicule.storage.vectors` treats any
  non-default field outside its pushdown set as unhonorable and raises, so a populated `extra`
  is not a flexible escape hatch — it is an exception.
- It is the same shape as the stage context object rejected in §2.3, and for the same reason:
  an escape hatch on a type this central never narrows again.

**`is_empty` goes with it.** With `workspace_ids` required, a filter is never empty, and the
property currently exists to let `predicate_for` short-circuit.

### 3.2 `workspace_ids` is a security boundary, not a performance question

This is the part of the shape that is not negotiable, and the reason is not about vector stores
at all.

**Required, because a boundary you can forget to pass is not a boundary.** Optionality here is
not a convenience; it is a default that silently means "every workspace". `PLAN.md` §14 makes
workspace isolation an invariant of every query, and an invariant expressed as an optional
parameter is an invariant enforced by everyone remembering.

**Non-empty, because an empty set is worse than a missing one.** `frozenset()` reads at a
glance as "no restriction" and means "match nothing". Both readings are catastrophic in
opposite directions, so the type refuses it and the caller has to say which they meant.

**Set-valued, because the single-value version is the one that gets made optional.**
`PLAN.md` §16 has admin cross-workspace search. With a scalar field, that feature arrives and
the only way to express it is `None`, which turns the required field straight back into an
optional one and undoes the paragraph above. Making it a set means the cross-workspace case is
`{"a", "b"}` — the same field, more members, nothing weakened.

**Where it comes from, and how it survives contact with the merged store.** `SqliteDocStore` is
constructed for one workspace and refuses a filter naming a different one. That is a good
property and this design does not change it. Instead:

> **Cross-workspace search is N scoped queries merged, never one unscoped query.**

The retrieval layer fans `workspace_ids` out to one store handle per workspace and merges. Three
consequences worth stating:

- The store keeps its "one query, one workspace" property, so the leak stays impossible by
  construction rather than by a predicate somebody wrote correctly.
- Results stay attributable per workspace, which is what an admin running a cross-workspace
  search actually wants to see.
- **The merge across workspaces is not RRF.** A chunk lives in exactly one workspace, so no
  chunk appears in two ladders and reciprocal rank fusion degenerates into "whichever workspace
  had the shorter result list wins". Merge on cosine similarity instead, which *is* comparable
  across workspaces because every vector came from one model in one space
  ([`storage.md`](storage.md) §6.2) — and, when a reranker ran, on the reranker score, which is
  comparable for the same reason. **BM25 is not comparable across workspaces**, because IDF is
  computed over the whole `chunks_fts` index while relevance is being judged per workspace;
  merging on it would rank the workspaces against each other rather than the chunks.

N is bounded by configuration and the feature is gated on team mode.

**What #6 built, and what waits for team mode.** The rule above is a rule about *merging*, and
merging needs N store handles. Obtaining them is a workspace registry — team-mode plumbing in
the storage layer, not a retrieval question — and it does not exist. What ships with the
pipeline is the property that makes waiting safe rather than risky: `SqliteDocStore` refuses a
filter naming a workspace it does not serve, with a message pointing at the fan-out, so a
cross-workspace query today is an error naming its own remedy rather than a query that quietly
answers about one workspace. The merge itself lands in the same change as the registry, and the
rule it must follow is settled above.

### 3.3 The split, settled as a rule rather than as a constant

`storage.md` §6.6 named the honest difficulty: pushing a resolved id list down works while the
list is small, and above some size the better plan inverts to over-fetch-and-post-filter. It
called the threshold "genuinely unknown" and it still is. **What this section settles is not the
number — it is the decision procedure, and the fact that every query records the two inputs that
set it.**

| Filter field | Resolved by | Why |
|---|---|---|
| `document_ids`, `kinds`, `langs` | Lance predicate | A promoted column exists ([`storage.md`](storage.md) §6.2) |
| `workspace_ids` | **Neither.** The hydrating join (§4.2) | No Lance column, deliberately: promoting it creates a value that can disagree with SQLite |
| `sources`, `media_types`, `collection_ids`, `tag_ids`, `updated_*` | SQLite, into `document_ids`, then pushed down | Each needs a join the vector table has no columns for |

The lexical leg needs none of this: it is one SQL statement against the authoritative store and
applies the whole filter inline, before `LIMIT` ([`storage.md`](storage.md) §6.1).

The dense leg has two regimes — push the ids down, or over-fetch and post-filter — plus one
early exit. The rule that picks between them:

```
1.  If no join-requiring field is set, there is nothing to resolve:
        push down what has a column, over-fetch (§4.3), and let the
        hydrating join do the rest.
2.  Otherwise resolve those fields in SQLite into a document-id set.
3.  If the set is EMPTY, the filter matches no document:
        return no candidates. Do not fall through.
4.  If |set| <= prefilter_id_limit:
        push it down as document_ids. Selectivity is now the store's problem.
5.  Otherwise:
        push down only what has a column, over-fetch, and post-filter.
```

**Step 3 is not a special case, it is the one that has to be written down.** "No join-requiring
field was set" and "the join-requiring fields resolved to nothing" both produce an empty
document-id set, and they are opposite instructions: the first means *do not constrain*, the
second means *constrain to nothing*. Collapsing them is a filter bypass — a query filtered to a
collection that happens to be empty would return the whole workspace, ranked and plausible. It
is the same shape as the empty `workspace_ids` hazard in §3.2, one layer down, and it is why
step 1 tests whether the *fields* are set rather than whether the *result* is empty.

**Resolution stops one row past the limit**, which the rule above did not say and the
implementation forced. Step 2 as written resolves the join-requiring fields into a document-id
set; on a corpus with a million documents in one source, "resolve" would mean listing a million
rows to answer a question the first thousand already answered. So the resolving query asks for
`prefilter_id_limit + 1`: at or below the limit the count is exact and the ids push down, and
above it the only thing anyone needed to know is that there were more. The trace records
`resolved_id_count_exact` alongside the count, because a lower bound recorded as a measurement
would skew the very distribution the threshold is meant to be set from.

`prefilter_id_limit` starts at **1000** and is configuration, not a constant. It is a starting
value and this document says so plainly rather than dressing it up: an `IN` list of a thousand
string literals is a predicate LanceDB can still plan usefully, and beyond that the predicate
starts costing more than the over-fetch it saves.

**What makes it settleable rather than a guess forever:** every query's trace records
`resolved_id_count` and the derived over-fetch factor (§4.3). #15's runs therefore carry the
distribution of both across a real corpus, and the threshold gets set from that distribution
instead of from an argument. That is what "contact with real data volumes" was waiting for, and
this design produces the contact as a side effect of running.

**The escape hatch, with its trigger stated.** One case defeats both plans: a workspace that is a
small slice of a large corpus *and* has more documents than `prefilter_id_limit`. The pre-filter
list is then too big to push down and the derived over-fetch exceeds its cap, so neither branch of
the rule is good. The condition is precise —

```
derived_overfetch > overfetch_max · k   AND   resolved_id_count > prefilter_id_limit
```

— and the answer if it is ever observed is a `workspace_id` column promoted into the Lance table,
which [`storage.md`](storage.md) §6.2 rejects today for a good reason (a value that can disagree
with SQLite) that would then have to be traded against a worse one. It is a storage change and it
becomes a storage ticket when the trace shows the condition occurring, not before. Nothing about
correctness depends on it: the post-filter plan is always correct, only slower.

### 3.4 What this costs in already-merged code

The shape above is not what shipped, so closing the question has a bill, and it is a small one.
Filed as [#36](https://github.com/mgd43b/manicule/issues/36) rather than done here, because this
document owns no code and two implementation tickets are in flight, and **paid there** — the
list below is what that ticket did, kept as the record of why each item was necessary:

- `Filter()` no longer constructs. `predicate_for` uses `default = Filter()` as a comparison
  sentinel and `filter.is_empty` as a short-circuit; both need replacing with a comparison
  against declared field defaults.
- `PUSHED_DOWN_FILTER_FIELDS` gains `langs`, and needs a companion set naming `workspace_ids` as
  **deliberately not pushed down**. This is the delicate one: the current code raises on any
  field it cannot honor, and that refusal is doing real work. It must keep raising for
  everything except the one field whose enforcement moved somewhere stronger, and the exemption
  has to be a named constant with the reason attached, not an omission.
- `SqliteDocStore._require_same_workspace` compares a scalar; it becomes a subset check, and the
  fan-out in §3.2 means it will only ever see its own workspace.
- The exemption above is the one place a security field is knowingly dropped by a store, so it
  gets a structural guard rather than a comment: `manicule.testing` grows
  `assert_pipeline_enforces_scope(pipeline, docstore, query)`, which runs a pipeline against a
  fixture holding soft-deleted, `pending` and foreign-workspace chunks and asserts that **no
  stage's output** contains one. A dense stage that skipped the join fails it. The same check is
  available as an opt-in runtime assertion in the pipeline runner, off by default.

---

## 4. The two legs

### 4.1 Lexical

The statement is settled and built ([`storage.md`](storage.md) §6.1): one joined query over
`chunks_fts`, `chunks` and `documents`, filters inline, `LIMIT` last, `ORDER BY bm25(...)`
ascending. Retrieval adds three things and changes none of it.

**It re-keys the score.** `DocStore.search_lexical` returns candidates carrying `scores["bm25"]`
— a key describing the *algorithm*, not the stage. The stage is named `lexical`, and
`Candidate.scores` is keyed by *stage* name, so the stage records its own name over the top.
`scored_by` merges rather than replaces, so both keys survive; the duplicate is harmless and
nothing needs to strip it. What matters is that fusion reads the names it was configured with
(§5.2) and never a key some store happened to write — which is what lets the lexical leg be
swapped for a learned-sparse one without touching the fusion stage.

**It merges rather than replaces.** The lexical stage receives whatever the dense stage produced
and returns the union: a chunk both legs found carries both scores, via `Candidate.scored_by`,
which returns a copy. This is why the conformance suite forbids returning the input list — a
merge is the natural place to accidentally mutate in place.

**Zero results is an event, not a warning.** An empty match is legitimate: an all-stopword query,
a query that tokenizes to nothing after `escape_match_query`. It is also what an FTS5 failure
looks like. Either way the pipeline continues with one leg and the ranking it produces is
well-formed, so the trace records `matched: 0` and the run is marked single-leg. §5.3 says what
#15 must do with that.

> **Corrected while building it.** An earlier draft of §11.1 had this leg record *the escaped
> match string*. It cannot, and should not: escaping belongs to `DocStore.search_lexical`, and a
> stage that reached into `manicule.storage.fts` to reproduce it would both import SQLAlchemy
> into a package that needs none and hardcode one store's query language into a leg that is
> meant to be swappable for a learned-sparse one. The trace records the text the leg was
> **given**, and the store owns what it makes of it.

> **Prior art.** `retriever.ts` wraps the FTS5 call in `try/catch` and, on failure, logs
> `console.warn('[retriever] FTS5 search failed, using dense-only')` and continues. The query
> succeeds, the answer looks normal, and the pipeline that produced it is not the pipeline the
> configuration describes. An evaluation harness cannot see the difference, which is the specific
> reason a degraded leg has to be part of a run's recorded identity rather than a line on
> someone's terminal.

### 4.2 Dense, and the top-`k` trap on the other side

The trap was found in the lexical leg and measured there: with `k = 5`, three matching live
chunks, five in a soft-deleted document and five in another workspace, `MATCH`-then-hydrate
returned **zero** live in-workspace results — a total loss of recall, silently, with a
well-formed empty result set. The lexical leg fixed it by filtering inside the statement, before
`LIMIT`.

**The dense leg cannot do that, and the reason is structural rather than an oversight.** The
Lance table holds a physical `id`, logical `chunk_id`, `publication_id`, `vector`,
`document_id`, `kind`, `lang`, `position` and `chunk_json`. It holds no `deleted_at`, no
`status`, and no `workspace_id`, and it holds none of them
deliberately: liveness and tenancy live on `documents`, in the authoritative store, and copying
them into a derived one creates a value that can disagree. So `VectorStore.search(v, k)` returns
`k` rows of which an unknown number are invisible, and the join that removes them necessarily
runs afterwards.

The dense stage is therefore three operations that are one stage:

```
embed(query)                                  # cached by fingerprint + text
  → VectorStore.search(v, k′, pushdown)       # k′ > k, see §4.3
  → hydrating join through documents          # workspace, deleted_at, status, publication
  → k live candidates, scored by cosine
```

The join is the one from `storage.md` §6.2 — `WHERE c.id IN (…) AND d.deleted_at IS NULL AND
d.status = 'indexed'`, plus `d.workspace_id`, followed by equality between the vector candidate's
publication and `d.publication_id`. It is inside the stage rather than beside it for
the reason in §2.4: a boundary that configuration can omit is not a boundary.

**The join also re-reads the chunk.** Lance returns `chunk_json`, which is a second copy of the
chunk text carried so the store can satisfy the protocol on its own
([`storage.md`](storage.md) §6.2). Retrieval uses the SQLite row, because SQLite is
authoritative and a divergence between the two copies must resolve toward the truth rather than
toward whichever one the query happened to read. The Lance copy is what makes
`assert_vector_store_is_dimension_agnostic` pass with no relational store behind it; it is not
what gets cited.

### 4.3 The over-fetch factor is derived, not constant

A constant multiplier is wrong in both directions: 2× is too little for a fifty-workspace
deployment and 20× is wasteful for a personal one, and neither knows which it is in. The factor
is computed from a quantity the system can measure about itself, on the same principle as the
embed batch size in [`ingest.md`](ingest.md) §8.2.

```
live_fraction = chunks of live, indexed documents IN THIS WORKSPACE
                ───────────────────────────────────────────────────
                          rows in the vector table

k′ = ceil(k / clamp(live_fraction, 0.05, 1.0))
k′ = max(k′, overfetch_min · k)
k′ = min(k′, overfetch_max · k, absolute_row_cap)
```

**The numerator is workspace-scoped and stops there — it does not model the rest of the filter.**
That is deliberate, and it is what keeps the fraction cacheable. Workspace and liveness are the
two exclusions the dense leg *cannot* push down and must therefore absorb by over-fetching;
everything else in the filter either has a Lance column or takes the pre-filter path in §3.3, so
it is already the store's problem rather than the over-fetch's. Folding the whole filter into the
fraction would make it a per-query aggregate — two `COUNT`s on the hot path of every search, to
refine a number that then gets clamped and rounded to a multiple anyway.

| Knob | Default | Why that value |
|---|---|---|
| `overfetch_min` | 3 | A healthy single-workspace index still loses rows to the soft-delete grace window, in-flight documents and unswept tombstones. 3× removes the retry from the common path, and over an exhaustive search below the ANN threshold (`docs/storage.md` §6.2) it is not measurable. That threshold now has a lifecycle behind it rather than only a number, so the claim is checkable: `manicule index` with no path says whether this index is still exhaustive. Past the threshold the over-fetch stops being free — it costs probes against an IVF_PQ index rather than a longer linear scan, which is the same interaction §3.3 flags for filters |
| `overfetch_max` | 20 | Past this the plan should have inverted to the pre-filter regime (§3.3); the cap is what makes that visible in the trace rather than absorbed as latency |
| `absolute_row_cap` | 2000 | Every over-fetched row is a `chunk_json` decode. The cap bounds the work independently of the multiplier |

**The denominator is the vector table's row count, not the chunk count.** Unswept tombstones are
still rows in Lance and still consume top-`k` slots; a fraction computed against SQLite's chunk
count would call an index clean while it was full of pending deletions.

**It is computed once per `(generation, workspace)`, not per query.** Two counts — one SQLite
aggregate, one `VectorStore.count()` — cached against the same generation counter that invalidates
the L1 cache (§10.3), so ingest, re-embed, delete and restore all refresh it for free. Keying on
the workspace as well as the generation is what the paragraph above buys: the fraction depends on
nothing else that varies between queries.

**What the numbers mean in practice.** A personal deployment with one workspace and a few
percent of soft-deleted content lands at the floor: `k′ = 3k`, and for the balanced profile that
is 60 rows fetched to keep 20. A fifty-workspace team deployment where each workspace holds 2%
of the corpus computes `live_fraction ≈ 0.02` and asks for 50×, hits `overfetch_max`, and the
cap firing *is the signal* that this deployment should be running the pre-filter plan. The
derived factor is not only a number; it is the detector for which regime you are in.

### 4.4 When the over-fetch is insufficient

Over-fetching does not guarantee `k` survivors and cannot be made to. What happens next is
specified rather than left to whoever implements it:

1. **Expand and retry**, at most twice, each time `k′ ← min(4·k′, absolute_row_cap)`. Four
   because a factor that failed is usually wrong by more than a little, and two expansions
   because a third would cost more than the query is worth.
2. **Stop when the leg has seen everything.** If `k′` reaches the number of rows the pushed-down
   predicate admits — the whole table when there is no predicate — every candidate the store can
   offer has been examined and there is nothing left to expand into.
3. **Report, in the trace**: `requested`, `fetched`, `survived`, `expansions`, and one of
   `satisfied | exhausted_corpus | exhausted_budget`.

The three outcomes are not interchangeable and collapsing them is the actual bug:

| Outcome | Means | Is it a defect? |
|---|---|---|
| `satisfied` | `survived >= k` | No |
| `exhausted_corpus` | The store holds no more matching rows | No. The corpus genuinely has that much |
| `exhausted_budget` | The caps stopped the search before the store did | **Yes.** The result is a floor, not an answer |

**It never quietly returns fewer than `k` and lets the caller assume the corpus had no more.**
That is the requirement `storage.md` §6.6 stated and this is the shape that satisfies it. An
`exhausted_budget` leg is surfaced to the caller alongside the results, counts against
confidence (§8.2), and — the part that matters for #15 — makes the run non-comparable to one that
was satisfied, because the two ran different amounts of search.

### 4.5 `min_score` belongs to the dense leg and to nothing else

`ProfileConfig.min_score` ships as 0.35 for every profile with a `ge=0.0, le=1.0` bound and the
description "floor below which a candidate drops". Where it is applied is the whole question, and
two of the three plausible places are unusable — the first arithmetically, the second because the
quantity has no absolute scale to compare a constant against.

**Not on the fused score.** With two legs and `K = 60`, the maximum reachable RRF score is
`2/61 ≈ 0.033`. A floor of 0.3 applied there discards every candidate, in every profile, and
returns an empty result set that looks exactly like a corpus with nothing in it. This is not a
tuning question; a fused score and a `[0,1]` floor are not on the same scale and never will be.

**Not on BM25.** `bm25()` is corpus-relative and unbounded; it has no absolute scale for a floor
to mean anything against. The best lexical match in a small corpus and a mediocre one in a large
corpus can produce the same magnitude.

**On the dense leg's cosine similarity, inside the dense stage, before fusion.** Vectors are
L2-normalized and the metric is cosine, so the score really is a cosine similarity in `[-1, 1]`
([`storage.md`](storage.md) §6.2) — the one number in the pipeline with an absolute meaning that
survives leaving the run it was computed in. Negative similarities clamp to 0 before comparison.

**The shipped values have been measured, and all three were wrong.** They were 0.5 / 0.3 / 0.15,
inherited rather than calibrated. Swept against the query set in §8.4:

| Floor | Relevant passages kept | Answerable queries still answered | Nonsense passages kept |
|---|---|---|---|
| 0.15, 0.30 | 160/160 | 16/16 | **210/210** |
| 0.40 | 160/160 | 16/16 | 64/210 |
| 0.45 | 160/160 | 16/16 | 9/210 |
| 0.50 | **153/160** | 16/16 | 0/210 |
| 0.60 | 42/160 | **15/16** | 0/210 |

So `balanced`'s 0.3 and `precise`'s 0.15 sat *below the noise* and could not fire — every passage
of every unanswerable query survived them. `fast`'s 0.5 sat *inside* the relevant range and was
discarding 7 of 160 passages a real query wanted. One floor was eating answers and two were
decorative.

**All three are now 0.35, and they no longer vary by profile**, because the quantity does not:
junk is junk at any cost setting, and what a profile actually trades is `candidates`,
`final_top_k` and `rerank`.

**0.35 sits deliberately below the point that separates the two populations, not on it.** That is
the whole design decision, and the reason is [`embeddings.md`](embeddings.md)'s runtime parity
guarantee. MLX and ONNX agree to cosine 0.9999 per vector, which permits roughly 0.01 of movement
in a query-passage cosine. A floor placed at the separation — around 0.45-0.48, where the sweep's
knee is — would put the decision boundary within backend drift of real passages, so the same
query could return a passage on Apple Silicon and drop it on x86. **Platform may change
throughput; it must never change output.** A threshold is a cliff, and a cliff is the wrong
instrument for a judgment this close.

So the floor is a junk filter and **not** the relevance decision. The relevance decision lives in
confidence (§8.4), which is continuous: backend drift nudges a reported number by a thousandth
and never changes which passages come back.

**Interaction with over-fetch:** the floor is applied *after* the hydrating join, so a discarded
low-similarity candidate does not count as a survivor, and the retry in §4.4 can be triggered by
the floor rather than by exclusions. That is correct — the leg's job is to produce `k` candidates
worth fusing — and the trace distinguishes `dropped_by_join` from `dropped_by_min_score`, because
one of those means the index is dirty and the other means the query is hard.

---

## 5. Fusion

### 5.1 RRF takes ranks, and only ranks

```
rrf_score(d) = Σ over legs where d appears:  1 / (K + rank_leg(d))       K = 60, ranks 1-based
```

**The entire reason to use RRF is that it does not need the legs' scores to be comparable**, and
they are not: cosine similarity is a bounded, absolute, model-defined quantity, and BM25 is an
unbounded, corpus-relative one whose sign is negative and whose better values are more negative.
There is no scaling that makes them commensurable across corpora. Discarding the magnitudes and
keeping the order is not an approximation — it is the point.

So: **no score weighting, no per-leg weighting, no normalization step.** A leg contributes rank
positions and nothing else.

> **Prior art, and what happens when the magnitudes come back in.** `reciprocalRankFusion` takes
> a `scoreWeighted` flag that multiplies `1/(k + rank + 1)` by the item's own score, and the
> retriever passes `true`. What the lexical leg's score is at that point:
> `searchFTS` returns `1 / (1 + Math.abs(row.rank))`, where `row.rank` is FTS5's `bm25()`. That
> value is negative, and a *better* match is more negative — so taking its absolute value and
> inverting maps the best lexical hit to the **smallest** score. The rows arrive in the right
> order, `ORDER BY rank`, and are then multiplied by a weight that is largest for the worst of
> them. The rank signal and the score weighting actively fight, the output is still a plausible
> ranked list, and nothing raises. `storage.md` §6.1 already names taking `bm25()`'s absolute
> value as a way to flatten or invert a ranking; this is that mistake compounded by
> reintroducing the very magnitudes RRF exists to discard.

### 5.2 Recovering each leg's rank ladder from a flat list

Fusion receives `list[Candidate]`, not a list per leg. The ladders are still there:

```
for each configured leg name L:
    ladder = [c for c in candidates if L in c.scores]
    ladder.sort(key=lambda c: c.scores[L], reverse=True)
    rank_L(c) = index in ladder, 1-based
```

This is exact, not approximate, because each leg's score is monotone within that leg: cosine
descending is the dense order, and `search_lexical` already negates `bm25()` so higher is better
there too. A candidate absent from a leg has no key for it and contributes nothing from it.

**The legs are configured, not hardcoded.** `RRFStage(name="rrf", legs=("dense", "lexical"),
k=60)`. Three consequences, and the second is the one that matters:

- Replacing the lexical leg with a learned-sparse one (§13) is a configuration edit.
- **A named leg that is not present earlier in the declared pipeline is a startup refusal.** A
  typo would otherwise turn two-leg fusion into one-leg fusion silently, which is the §4.1
  failure with a different cause and the same signature. Checked when the container assembles the
  pipeline, not on the first query.
- Three legs is expressible without a code change, which is what makes "is three legs better
  than two" a #15 question rather than a rewrite.

### 5.3 What RRF does when a leg comes back short

**A missing candidate contributes zero from that leg. It is not imputed a worst rank and not
penalized.** The consequences are worth being explicit about, because they look like bugs and are
not:

- A candidate ranked 1st in one leg and absent from the other scores `1/61 ≈ 0.0164`.
- A candidate ranked 3rd and 4th in *both* scores `1/63 + 1/64 ≈ 0.0315`, and outranks it.

That is RRF working as designed: **cross-leg agreement beats a single strong opinion.** It is
also why a floor on the fused score is meaningless (§4.5) and why a reranker after fusion earns
its place (§6).

**A short leg shortens its ladder, and that needs no correction.** If lexical returns 6
candidates and dense returns 20, the lexical ladder has 6 rungs. Nothing should pad it to 20 with
imputed ranks; the fused ordering for the tail is then determined by the dense leg alone, which is
the correct answer to "only one leg had an opinion about these".

**A leg returning zero silently becomes a single-leg pipeline.** The output is well-formed and
correctly ordered — by one leg. The rule:

> A degraded run is part of that run's recorded identity. #15 must refuse to compare a run whose
> trace shows a leg returned zero against one where both legs ran, and the harness fails the
> comparison rather than averaging over it.

This is the difference between a metric that can move because the corpus is hard and a metric that
moved because FTS5 threw. Both are legitimate outcomes of a query and only one is a legitimate
input to a measurement.

### 5.4 `K = 60`, and what `K` is actually doing

60 is the value from the original RRF paper and the near-universal default, and it stays — but
it is configuration and it is recorded in the trace, because changing it changes every number
#15 has ever written down.

What is worth understanding before anyone tunes it: **RRF is a consensus operator, not a ranking
operator.** With `K = 60` and 20 candidates per leg, a rank-20 candidate scores `61/80 ≈ 76%` of
what a rank-1 candidate scores from the same leg. Within-leg ordering is compressed almost flat
on purpose, so that appearing in both legs dominates appearing high in one. At `candidates = 50`
(`precise`) the bottom of the ladder is still at `61/110 ≈ 55%` of the top — flatter still,
because the ladder is longer.

Two things follow. Lowering `K` sharpens within-leg ranking and weakens the consensus effect,
which is the *opposite* of why RRF was chosen. And the fused list is a good candidate set and a
mediocre final ordering, which is exactly the job description for a cross-encoder.

---

## 6. Rerank

### 6.1 A cross-encoder, not a language model asked for a number

`PLAN.md` §8 says `sentence-transformers CrossEncoder`, and the distinction is not pedantic. A
cross-encoder encodes `(query, passage)` jointly and emits one relevance logit from a model
trained for exactly that. It is deterministic, it is cheap relative to generation, and its output
is a scalar on a fixed scale for a fixed model.

> **Prior art.** `cross-encoder.ts` is not one. It prompts the *generation* model with "Rate how
> well the passage answers the query. Respond with a single integer from 0 to 10", one call per
> candidate, `maxTokens: 8`, and parses the first integer out of the reply with a regex —
> returning **0** when nothing parses, which is indistinguishable from a genuine "irrelevant".
> The reranked head is then concatenated with an unreranked tail still carrying whatever the
> fusion left on it — RRF sums on the order of 0.016 — while the head's are 0.0–1.0, so one list
> carries two scales that differ by nearly two orders of magnitude. And on any
> exception the function returns the input unchanged, so a failed rerank is a silent no-op that
> the profile still reports as reranked. Four separate ways for the ranking to be wrong without
> anything raising, in 62 lines.

Three rules follow directly:

- **The reranker's failure is the query's failure.** It raises; it does not return the input. A
  profile that says `rerank: true` and produced an unreranked list has lied to #15 about which
  pipeline ran.
- **The reranker truncates to what it scored.** It rescores the head — `profile.candidates`
  entries — and returns only those. No mixed-scale tail. The container refuses a profile where
  `final_top_k > candidates`, since that is the only configuration in which the tail would have
  been needed.
- **`Reranker.model_id` is recorded on every run.** The protocol already requires it, for exactly
  this: a recorded result that cannot name its reranker cannot be reproduced.

### 6.2 The model

**`BAAI/bge-reranker-v2-m3`**, configurable, lazily loaded, never loaded under `fast`.

It is the family pair for the embedder settled in `embeddings.md` §1, and the reason to pick a
matched pair here is not brand tidiness: bge-m3 was chosen substantially because it is
multilingual in one space, which closed [#31](https://github.com/mgd43b/manicule/issues/31). A
monolingual English reranker placed after it would take a correctly-retrieved non-English passage
and rank it down — undoing the property the embedder was chosen for, at the last stage before the
answer, where it is least visible.

Honest costs:

- It is an XLM-RoBERTa-large-sized model, meaningfully larger than the small English rerankers it
  would be compared against. On a corpus that is entirely English, a smaller monolingual reranker
  is very likely a better trade, and the configuration knob exists so an operator can make that
  choice — but the *default* has to be the one that does not silently break multilingual
  retrieval.
- `sentence-transformers` brings torch, which is why it belongs behind its own extra rather than
  in the storage or embeddings stack. `fast` never loads it.
- Its scores are unbounded logits, model-specific, and comparable only within one model's
  rankings. Never mix them with cosine similarities, and never compare a confidence computed
  under one reranker to one computed under another (§8.2).

### 6.3 Cost is linear in `candidates`, which is what the profiles are actually buying

The reranker scores `profile.candidates` pairs: 20 on `balanced`, 50 on `precise`, none on
`fast`. Cost is linear in that count and dominates the rest of the pipeline by a wide margin —
the two legs are one forward pass over a short query plus two indexed lookups, while this is
`candidates` forward passes over full-length passages. So the three profiles are not three
settings; they are "no second model", "a second model over 20 passages", and "a second model over
50". #15 records the constant; this document records which term dominates.

---

## 7. Context assembly

### 7.1 Not a stage, and what it emits

Assembly consumes the pipeline's output and produces `Context`: the passages, their token count,
and whether anything was dropped. `contracts.md` §6 lists "whether `Context` assembly is a
`RetrievalStage`" as open; this document closes it as **no**, on the grounds the merged `Context`
docstring already gives — it emits a different type, and keeping the types distinct is exactly
what makes the stage list freely reorderable while this step is not.

### 7.2 Two token counters, and using the wrong one is a category error

There are two token budgets in manicule and they are measured with different tokenizers, for
different models, to protect against different failures.

| Budget | Tokenizer | Protects against |
|---|---|---|
| Chunk size, 512 | The **embedder's**, on the exact string it will see ([`parsing.md`](parsing.md) §1.2) | Silent truncation inside the embedder, producing a vector claiming text it never saw |
| Context window | The **generator's** | An assembled context larger than the window, truncated by the server |

**`Chunk.token_count` is the first of those, and using it for the second is the category error
this section exists to prevent.** It is measured in XLM-RoBERTa SentencePiece units for a model
that is not generating anything. It is sitting right there on every candidate, it is a plausible
number, and it is wrong for this purpose by an unknown factor.

**`tiktoken` is a stand-in for the second, and it is named as one.** `PLAN.md` §16 and #6's
comment both settle it, and the reason to keep it is that it is a real BPE implementation rather
than a character heuristic. But the generator is Ollama-hosted (`PLAN.md` §7), running Llama or
Qwen or Mistral vocabularies, none of which are tiktoken's. So:

- Use `tiktoken.get_encoding("o200k_base")`, **not** `encoding_for_model("gpt-4o")`. Naming a
  model that is not being used makes the estimate look authoritative. The encoding name goes in
  the trace.
- Apply a per-model safety factor, in the direction that matters. Undercounting overflows the
  window and gets the context truncated by the server, which is the silent failure; overcounting
  wastes budget, which costs a passage. Same reasoning and the same direction as
  `PROVISIONAL_SAFETY_FACTOR` in `manicule.chunking.tokens`.
- **Never sample and extrapolate.** [`parsing.md`](parsing.md) §1.2 already rejects it for the
  chunk budget; it is worse here, because the fitter's entire job is not to overflow and
  extrapolation turns a bounded error into an unbounded one on precisely the longest inputs.
- **Then stop guessing.** Ollama's generate response carries `prompt_eval_count` — the true token
  count, from the model that counts. The fitter's estimate is compared against it after the first
  call and the drift is recorded per model; drift beyond tolerance is an error worth surfacing,
  not a rounding difference. Measuring once beats a safety factor forever, and #7 owns the call
  that makes it available.
- Counts are cached by `chunk.id`. Chunk ids are content-derived, so the cache is exact and can
  never go stale.

**And the vocabulary comes off the machine, never off the query.** `tiktoken` ships no
vocabularies in its wheel; `get_encoding` downloads `o200k_base` from a Microsoft-hosted blob
store on first use, which put an HTTPS call on the path that assembles a context. An air-gapped
host therefore indexed a corpus perfectly and then failed at the first *question*, with an
error naming a blob storage host — the failure split across two moments, the second one
unexplained (#61).

So the fitter resolves its encoding through `manicule.vocabularies`, which is the vocabulary
counterpart of what [`parsing.md`](parsing.md) §8.1.1 does for grammars and follows it move for
move: a pre-seed an operator runs and can watch fail, an offline bundle consulted before the
network so a host with no route to anything can still pre-seed, and a query path that **cannot
fetch at all** — `load_encoding` shuts `tiktoken`'s single door to the network for the duration
of the call, so a vocabulary that was never seeded is a refusal naming the encoding, the cache
that was read and where a bundle was looked for. The counter resolves at construction, which
is while `build_retriever` is assembling retrieval, so the refusal arrives at startup rather
than inside a query.

**Something runs that pre-seed, and it is the same something that seeds the grammars.**
`manicule init` seeds both, and `manicule doctor --fix` repairs both — one call each, so an
install and a repair cannot drift into meaning different things. A pre-seed that fails is a
note rather than an exception, exactly as [`parsing.md`](parsing.md) §8.1 has it: the
configuration is already written and an air-gapped host must still be able to finish
installing. `doctor` reports a missing vocabulary as **failing** where it reports a missing
grammar as **degraded**, and the asymmetry is deliberate. A corpus of Markdown and PDFs works
perfectly with no grammars, so a red check there would be a red check on a healthy machine —
which teaches an operator to ignore `doctor`. There is no corpus that works without a
vocabulary: every context is measured with it, so a machine without one cannot answer at all.
`--fix` is offered by the command line and by nothing else — an assistant holding a tool call
and a health endpoint must not be able to start a download.

The recorded identity is unchanged and that is the point: `provisional:x1.5:tiktoken/…@0.13.0`
names a release, and the release was only ever a *claim* about the bytes while those bytes
arrived from a URL nobody checked. Every byte the pre-seed writes is now verified against the
digest the installed `tiktoken` declares for it, so the claim is checked. Adding a digest to
the identifier would rewrite every chunk fingerprint in every index to record what the version
already implies.

### 7.3 Whole passages only — a truncated citation is a broken citation

**Assembly includes a candidate entirely or not at all.** It never trims a passage to fit.

This is not a style preference. `contracts.md` §1 puts a round-trip obligation on every anchor:
resolving it returns the text the chunk claims, and it is a test obligation on every parser. A
`PageAnchor`'s rects, a `CellAnchor`'s range and a `LineAnchor`'s span all describe the *whole*
chunk. Trim the text and the anchor now points at more than the passage says, which is the exact
class of defect the `Anchor` type was designed to make impossible — and PR #32 tightened the
middleware contract to forbid rewriting cited text for the same reason. Assembly is downstream of
both and does not get an exemption.

> **Prior art.** `fitToContextWindow` truncates the first non-fitting chunk to the remaining
> space, walks back to a sentence boundary, appends `'...'`, and stops. The cited text then
> differs from the source, and the citation points at a span the reader will not find. The
> character budget is estimated from tokens via a CJK ratio and a `charsPerToken` of 4 or 1.5,
> so how much gets cut is itself an approximation.

**Skip and continue, do not stop.** A non-fitting candidate is skipped and the next one is tried.
Chunking keeps tables and code blocks whole ([`contracts.md`](contracts.md) §2), so one large
passage in the middle of the ranking is normal, and stopping there would discard every good
passage behind it. Order is never changed — only membership. `Context.truncated` is set, and the
trace lists which candidates were dropped and how large they were.

**The budget is a rail, not the selection mechanism, and saying so is more useful than implying
otherwise.** [`parsing.md`](parsing.md) §1.3 splits the 512-token chunk budget as `64` for the
breadcrumb and `448` for the text, and the breadcrumb never appears in `text` at all
([`parsing.md`](parsing.md) §5.1) — so a passage reaching the context is at most **448** embedder
tokens, not 512. Selection is done by `final_top_k` in every shipped profile and the fitter never
binds. It exists for the configurations that do not ship — a raised chunk budget, a raised
`final_top_k`, a long history — and it asserts rather than handles the case where even the
top-ranked passage does not fit, because the shipped budgets make that arithmetically impossible.

That last sentence is a claim about numbers, so §12 sets the numbers from it rather than the
other way round. It is deliberately **not** a validator on `ProfileConfig`: raising
`final_top_k` without raising the budget is an ordinary thing to want, typical passages are a
fraction of the chunk budget, and refusing a configuration that works in practice on the
strength of a worst case that will not occur is the kind of over-strictness that teaches people
to route around a check. What happens instead is visible rather than silent — the fitter skips
what does not fit, `Context.truncated` is set, and the trace names each dropped passage and its
size.

One caveat that keeps the rail honest: those bounds are in the *embedder's* tokenizer and the
budget is in the *generator's* (§7.2). The margin is wide enough that no plausible ratio between
two vocabularies closes it, which is why this is a rail and not a calculation.

### 7.4 The window cross-check is a startup refusal

`context_tokens` and `history_tokens` are budgets for *manicule's* content. The generator's
context window is a property of a model configured somewhere else. Nothing currently compares
them, and the `balanced` profile's 16384 + 1024 against an 8k-window local model is an assembled
context twice the size of the window on every query.

**So it is checked once, at startup, when the generator is bound:**

```
context_tokens + history_tokens + system_prompt_tokens + generation_reserve
        must fit the configured generator's context window
```

A profile that does not fit is a refusal with both numbers named, not a runtime truncation. This
is the same discipline as [`ingest.md`](ingest.md) §7's budget/context cross-check, run in the
same place and for the same reason: a limit that can only be discovered by exceeding it gets
discovered in production. #7 owns the generator and therefore owns the enforcement point; this
document owns the requirement, and `manicule.retrieval.assembly.window_problem` is the predicate
both sides share so that neither has to re-derive the arithmetic.

**And the profile numbers had to move, because `precise` did not pass its own check.** #43 made
startup refuse and name the fix rather than change the numbers, on the grounds that the profile
numbers belonged to this document. They are settled in §12.

---

## 8. Confidence

### 8.1 What it is not

**It is not a probability that the answer is correct.** Nothing in this pipeline is calibrated
against anything, and presenting an uncalibrated score as a probability is the kind of claim that
gets believed.

**It is not about the answer at all.** It is computed before generation, from the retrieval. It
says how much supporting evidence was found and how strongly the two independent methods of
finding it agreed. An answer can be wrong with high confidence — the evidence was there and the
model misread it — and that is not a bug in this number.

**It is not comparable across configurations.** A confidence computed under `precise` with one
reranker and one under `fast` with none are different measurements. The value therefore travels
with the pipeline identity that produced it, and #15 never compares two of them across
configurations.

The honest one-line description, and the one the UI should use: **how well-supported this answer
is by the corpus.**

### 8.2 What goes in, and what cannot

Only quantities with a defined scale are admissible.

| Component | Weight | Source |
|---|---|---|
| Similarity | 0.55 | Strongest `scores["dense"]` **per document**, negatives clamped to 0, each rescaled against the corpus noise level, combined by noisy-OR (§8.4) |
| Cross-leg agreement | 0.15 | Fraction of the **evidence-bearing** passages carrying both leg scores, scaled by the evidence (§8.4) |
| Reranker | 0.30 | Mean `sigmoid(logit)` over the context passages — **present only when a reranker ran** |

Two details in the similarity row are load-bearing, and both were defects fixed by measurement:

- **"that carry one."** A passage the lexical leg found and the dense leg never ranked has *no*
  similarity. Reading the absent score as `0.0` asserts the dense leg looked at it and found it
  orthogonal to the query — it never looked. On the query that exposed this, BM25's top hit was
  averaged in as a zero and cost the best answer in the corpus a twentieth of a point.
- **"rescaled."** Raw cosine is not centered on zero, so unrelated text scores ~0.45 rather than
  ~0.0 and lands in the same band as a real match. §8.4 is the measurement and the constants.

**There is no support-breadth term, and its removal was a measured correction.** `min(distinct
documents / 3, 1.0)` was meant to read as corroboration and read as *diffusion* instead: noise
scatters across a corpus and a good answer concentrates in one place, so the term systematically
paid the worse result. On the pair that exposed the defect, a nonsense query reached four
documents and took the full 0.15 while a correctly-focused answer reached one and took 0.05.
Telling corroboration from diffusion means knowing whether the documents *agree*, which is an
entailment check and not a count. Its weight went to similarity, which keeps the total at 1.0 and
therefore keeps `fast`'s 0.70 ceiling exactly where §8.3 puts it.

What is excluded, and why:

- **The fused RRF score.** A rank artifact bounded by `2/61`; it has no absolute meaning (§4.5).
- **BM25.** Corpus-relative and unbounded.
- **Keyword coverage.** Replaced by cross-leg agreement, which is the same idea done properly.
  Substring matching of query keywords against passage text does no stemming and no IDF
  weighting, so a query for `authenticate` scores zero coverage against a passage containing
  `authentication` — the precise failure `storage.md` §6.1 records as the reason the FTS5
  tokenizer is `porter`. The lexical leg already solved this; the confidence score should ask the
  lexical leg rather than re-implement a worse version of it.

**Two states that are not "low".** No retrieval attempted (the router answered directly, §9)
yields confidence **absent** — not 0.0. Retrieval attempted and nothing found yields the `none`
band with a reason. The distinction is the one `contracts.md` §1 makes for `Unlocated`: "we did
not look" and "we looked and there is nothing" are different claims, and a single zero conflates
them.

**An `exhausted_budget` leg (§4.4) caps confidence at `medium`,** because the retrieval is known
to be a floor rather than a result.

**Suppression tracks the shape of the pipeline, never what one query happened to match.** A
component is suppressed when the pipeline could not produce it — no reranker ran, or fewer than
two legs are *declared*, or no context passage carries a dense score at all. It contributes
nothing, its weight is not redistributed, the ceiling drops accordingly, and the trace says which
component went and why.

What suppression is **not** for is a leg that ran and found nothing, and getting that backwards
was most of the reported defect. The rule used to key on "no context passage carries this leg's
score", which reads as a broken leg and is nothing of the kind: neither leg catches exceptions,
so a leg that returns has run, and an empty return means *it ran and the query matched nothing* —
a fact about the query, and evidence. The same test also fired when a leg found plenty and none
of its hits survived into the final few. The consequence ran exactly the wrong way: a nonsense
query matches no keywords, so it had its agreement component **waived**, while a real question
that matched some paid the penalty in full. Confidence must not blame the corpus for a fault in
the pipeline — and it must not excuse a query for being too poor to match anything either.

### 8.3 No fallback term, and why `fast` cannot report high confidence

When no reranker ran, the reranker term **contributes zero and the remaining weights are not
renormalized.** The arithmetic maximum under `fast` is therefore 0.70.

> **The rule, stated as the property it protects.** A component that did not run must never be
> filled in from one that did. Substituting the retrieval average into the reranker's empty slot
> is the tempting shape, because it keeps the scale full — and it would make retrieval count for
> `0.55 + 0.30 = 0.85` instead of 0.55, so **turning the reranker off would raise the reported
> confidence** for identical retrieval. The pipeline that skipped the verification step would
> claim more than the one that ran it. Renormalizing the remaining weights reaches the same place
> by a more respectable route and is refused for the same reason: manicule reports what it
> measured, and a measurement not taken lowers the ceiling rather than borrowing a number.

Bands, applied to that single absolute scale:

| Band | Score | Reachable under |
|---|---|---|
| `high` | ≥ 0.75 | `balanced`, `precise` |
| `medium` | ≥ 0.45 | all |
| `low` | ≥ 0.10 | all |
| `none` | < 0.10, or nothing retrieved | all |

**The `none` boundary is 0.10 because that is where the measurement put the gap** (§8.4). Once
similarity is rescaled the two populations separate with nothing between them — unanswerable
questions reached at most 0.032 and real ones at least 0.162, across all three profiles. A
boundary at 0.20 sat *inside* the real population, so `precise`, the profile that looks hardest,
reported "nothing here resembles your question" for questions the corpus answers. That is the
original defect wearing the other mask.

**`fast` topping out at `medium` is the intended behavior, not an artifact.** `fast` is the
profile that skips the verification step; it should not be able to claim it verified. This is
also the most concrete difference between the profiles that a user ever sees, which makes the
cost of choosing `fast` visible at the moment it matters.

**The scalar is never reported alone.** `Confidence` carries the band, the components that
produced it, and the pipeline identity. A number that cannot say why it is 0.62 is a number
nobody can act on, and one that cannot say what produced it is one #15 cannot compare.

### 8.4 Calibrating similarity against the corpus's own noise

**Cosine is not centered on zero, and pretending otherwise is what let nonsense outrank a real
question.** Dense retrieval always returns its nearest neighbors; on a corpus with nothing
relevant in it, those neighbors are still returned and are *not* far away in absolute terms.

The measurement, over manicule's own documentation — 13 documents, 604 chunks, BGE-M3, all
three profiles — asked 16 questions the corpus answers and 22 it demonstrably cannot (subjects
absent from it, plus gibberish). Read as the mean cosine over the passages that reached a
context, which is the quantity confidence actually scores:

| Query set | min | median | max |
|---|---|---|---|
| Questions the corpus answers (16) | 0.531 | 0.597 | 0.641 |
| Questions it cannot (16) | 0.353 | 0.391 | 0.457 |
| Gibberish (6) | 0.356 | 0.372 | 0.451 |

The two populations are cleanly separated and **neither is near zero**. Fed in raw, the worst
real question and the best nonsense one differ by 0.07 on a scale whose bands are cut at 0.45 —
so they land in the same band, which is exactly what was observed. So similarity is rescaled:

```
evidence(p) = clamp01((cosine(p) - NOISE_SIMILARITY) / (STRONG_SIMILARITY - NOISE_SIMILARITY))
component   = 1 - Π over documents d of (1 - max evidence(p) for p in d)

              NOISE_SIMILARITY = 0.54      STRONG_SIMILARITY = 0.65
```

Below the noise level the answer is 0.0, because "further from the query than unrelated text" is
not a finer grade of relevance.

**The statistic is per passage, combined afterwards — not a mean over the context — and that
correction came from a second report.** A short question retrieved the exactly-correct passage at
rank 1 and reported confidence 0.0, band `none`. Measured against a synthetic glossary defining an
acronym that collides with an ordinary English word:

| Query | Correct passage | Its cosine | Mean over the context | Reported |
|---|---|---|---|---|
| "What is NOW?" | rank 1 | 0.623 | 0.387 | **0.0 `none`** |
| "What is the Network Operations Workspace?" | rank 1 | 0.702 | 0.408 | **0.0 `none`** |
| a deliberately lexical phrasing | rank 1 | 0.747 | 0.457 | 0.035 `none` |

**A mean answers "how on-topic is the typical passage shown", and nobody asked that.** The
pipeline fills the context to `final_top_k` whether or not the corpus holds that many relevant
passages, so a narrow question is *guaranteed* filler — and averaging made that filler count as
evidence against the answer in front of it. It also explains the asymmetry in the report: the
lexical phrasing scored non-zero only because its mean happened to land a hundredth above the
floor, not because lexical evidence was being counted.

Noisy-OR replaces it because four properties are needed at once, and no single summary statistic
has them:

- **Filler costs nothing.** A passage at zero evidence multiplies by one.
- **Independent support compounds.** Two documents that each answer the question are better
  support than one, which is what the retired breadth term was reaching for and getting backwards
  — and the difference is that a document only counts once it independently clears the floor, so
  scattered *noise* still contributes nothing.
- **Duplicates do not multiply.** The strongest passage *per document* is taken first, so ten
  chunks of one page are one observation. Otherwise a finely-chunked document manufactures
  certainty, which is a property of the ingest configuration reported as a property of evidence.
- **It saturates rather than overflowing**, so no clamp has to hide an out-of-range number.

#### Corroboration is scaled by the evidence

Cross-leg agreement is measured over the **evidence-bearing** passages, for the same reason: a
query with one strong, doubly-confirmed answer used to score 1/5 for perfect corroboration
because the denominator counted four passages nobody claimed were relevant.

It is then multiplied by the evidence level, because **you cannot corroborate more than you
have.** Counting alone handed an unrelated query the full agreement weight: exactly one passage
cleared the floor, barely, both legs happened to touch it, and 1/1 paid out in full — 0.15 of a
number whose whole job in that case is to say the corpus holds nothing.

#### Where the floor sits, and why not at either edge

Per passage the two populations sit far closer than their context means did, which is why moving
to a per-passage statistic required re-measuring the constant rather than carrying it over:

| Measured, per passage | Value |
|---|---|
| Strongest passage any unanswerable query reached | **0.5194** |
| Weakest top passage any answerable query produced | **0.5598** |

`NOISE_SIMILARITY = 0.54` is the middle of that gap rather than either edge, and mid-gap is
deliberate — the same argument §4.5 makes for the retrieval floor. MLX and ONNX agree to cosine
0.9999, worth about 0.01 of movement in a query-passage cosine, so a constant placed against
either edge could be crossed by a backend change. Platform may change throughput; it must never
change output.

#### What the rescale did

Same index, same queries, before and after:

| Query | Originally | After the rescale | After per-passage evidence |
|---|---|---|---|
| `zzzqqq unrelated nonsense xyzzy` | 0.330 `low` | 0.002 `none` | **0.000 `none`** |
| `how are citations verified` | 0.246 `low` | 0.449 `low` | **0.470 `medium`** |
| "What is NOW?", correct at rank 1 | — | **0.000 `none`** | **0.414 `low`** |

Across the whole query set, on **every** profile: every one of the 16 answerable questions scores
`low` or better, every one of the 22 unanswerable ones scores `none`, and **all 16 answerable
questions still return passages** — none of this costs recall. Worst answerable score against
best unanswerable score: 0.332 vs 0.000 (`fast`), 0.398 vs 0.000 (`balanced`), 0.408 vs 0.000
(`precise`). The margin is wider than it was under the mean, and no unanswerable query reaches a
non-zero score at all.

#### Confidence is not comparable across profiles

`PipelineIdentity` travels with every score and §8.1 says the number is not comparable across
configurations, of which a profile is one. Compare a profile against itself over time, never
against another profile. Combining per document rather than averaging removed the largest source
of cross-profile drift — a deeper profile no longer scores lower merely for showing more of the
weak tail — but a reranked pipeline still reaches a ceiling an unreranked one cannot, and that
difference is real rather than an artifact.

### 8.5 The diagnostic

`explain_confidence` returns every input to a score: the components and what each weighed, every
suppressed component and why, the normalization constants, the band thresholds, the weights, and
per passage its raw cosine, its rescaled evidence, which legs scored it, and whether it was
counted or displaced by a stronger passage from the same document.

**It carries no passage text.** Diagnostics travel further than results — into logs, bug reports
and screenshots — and one that carried corpus text would turn every one of those into a
disclosure. The chunk id is enough for anyone entitled to read the passage and useless to anyone
who is not.

It runs `score_confidence` rather than reimplementing it, so the explanation cannot drift from the
number it explains. A diagnostic that computes its own answer is a second implementation, and the
one that disagrees is always the one nobody reads.

---

## 9. The query router

### 9.1 Deterministic, and only the routes that exist

A pure function over the query text: no model call, no store access, and nothing but the text
decides the route. It runs before the L1 cache. `PLAN.md` §16 keeps it because trivial input
should not consume an LLM call, and that is reason enough. (A `UTILITY` *handler* will go on to
read a store — that is the answer being computed, not the route being chosen.)

```
Route = RETRIEVE | GREETING | UTILITY(kind)
```

**A route nothing returns is not a route.** The prior art declares four (`rag`, `direct`,
`web_only`, `rag_web`) and `routeQuery` can only ever return two of them; the other two are a
type that documents a feature the function does not have. Each `UTILITY` kind here names a
handler that exists — document count, document list, index status — and the container fails to
start if a declared kind has none.

### 9.2 Full match, never prefix, and tuned for precision

**A greeting route requires the entire input to be a greeting**, modulo surrounding punctuation
and whitespace, under a short length bound. Not a prefix match.

The prefix version is not a small imprecision. `/^(hi|hello|hey|howdy|yo|sup|greetings)\b/i`
routes **"yo-yo manufacturing tolerances"** away from the corpus, because `-` is a non-word
character and the boundary matches. `/^(thanks|thank\s+you)\b/i` routes **"thanks for the memory
dump — what does it say?"** to a canned reply. Both are ordinary queries against a technical
corpus, and both get an answer that never touched the index. Anchoring at both ends deletes the
entire class.

The governing rule, which also settles how much effort the pattern list deserves:

> **The router is tuned for precision, not recall.** A missed greeting costs one retrieval, which
> is harmless. A false greeting costs a wrong answer to a real question, which is not. When in
> doubt, retrieve.

That rule is also the answer to multilingualism. The corpus is multilingual by design
([`embeddings.md`](embeddings.md) §1.2) and any greeting list will be incomplete; the pattern list
is configuration, ships small, and being incomplete costs only latency.

### 9.3 A direct answer has no citations, and says so

The router is a retrieval bypass, which makes it the one path where an answer legitimately has no
sources. It must therefore be visibly different rather than quietly identical:

- The response carries **no citations**, and states that the corpus was not consulted.
- Confidence is **absent**, not 1.0 and not 0.0 (§8.2).
- Directly-routed queries are **not cached** (§10). They are already cheap, and the utility ones
  answer with live counts that a cache would staleness-bug for no gain.
- The route taken is in the trace, so #15 can see that a query in its set never reached retrieval
  — which would otherwise show up as a mysterious zero.

---

## 10. The L1 query-result cache

`PLAN.md` §16 has three caches. **This document owns L1 only.** L2, the embedding cache, is
settled in [`embeddings.md`](embeddings.md) §8, keyed on the canonical fingerprint and the
post-middleware `embed_text`. L3 is a web-search cache and belongs to whichever ticket adds a
web-search connector; it is not a retrieval-pipeline cache.

### 10.1 It caches decisions, not content

The cached value is the ranked list of **chunk ids with their per-stage scores** — the decision
the pipeline reached — and never the chunk text. On a hit, the ids are re-hydrated through the
same join the dense leg uses (§4.2).

This is not a memory optimization. It is what makes the cache incapable of the failure a
content-caching version invites:

- **A cache hit cannot serve a soft-deleted, unindexed or foreign-workspace chunk**, because it
  holds no chunks. The boundary is re-enforced on every hit rather than snapshotted at the moment
  of the miss.
- Re-hydration costs one indexed `IN` query against SQLite, against a full pipeline that includes
  an embedding forward pass and possibly a cross-encoder.

**If hydration drops anything, the entry is stale: evict it and run the pipeline.** Returning a
shortened list would be correct but misleading — the ranking was computed over a candidate set
that no longer exists, and the replacement for the dropped candidate was never considered.

### 10.2 The key

A hash over the canonical form of, in order:

```
generation counter
the whole Filter, canonicalized (workspace_ids sorted, then every other field)
profile name + effective overrides
Query.limit
pipeline declaration (stage names, in order)
reranker model_id, or null
RRF K
query text (trimmed, otherwise exact)
```

Notes on four of those:

- **`Query.limit` is in the key even though it looks like a presentation concern.** Retrieval
  depth is `max(limit, final_top_k)` (Appendix B), so a larger `limit` is a deeper run, and
  serving a cached 10-result ranking to a request for 50 would return a short list that looks
  like a corpus with nothing more in it — the §4.4 failure arriving through the cache.

- **The whole `Filter`, not just the workspace.** Two filters produce two different rankings; a
  key that omits one is a cache that answers a different question.
- **The pipeline declaration and the reranker id.** Comparing two pipelines is #15's entire
  method, and a cache that cannot tell them apart would serve pipeline A's ranking as pipeline
  B's result. #15 also runs with the cache **disabled**, which is a configuration flag rather
  than a code path.
- **Conversation history is *not* in the key.** Retrieval runs on the query text; nothing in this
  pipeline reads history. Including it would guarantee a miss on every turn of a conversation —
  the one place a user actually repeats themselves. If a history-conditioned query rewrite ever
  ships (§13), history joins the key in the same commit.

> **Prior art.** `buildCacheKey(query, profile, conversationHistory)` includes the history and
> omits the workspace, the filter and the index generation. It caches the full `QueryResult`,
> answer text and source content included, which is why the history is in there: it is an *answer*
> cache wearing a retrieval cache's name. manicule's L1 caches retrieval only. Caching a generated
> answer is a different feature with different invalidation, and it belongs to
> [#7](https://github.com/mgd43b/manicule/issues/7) if it belongs anywhere.

### 10.3 Invalidation is a generation counter, and re-embed is why

An in-memory counter, bumped by any commit that changes what a query could return: document
upsert, `replace_chunks`, soft delete and undelete, hard delete, reconcile-driven deletion, a
`doctor` repair that rewrites a derived index, a restore, and the `vector_table` swap at the end
of `reindex --re-embed`. The counter is in the key, so a bump invalidates everything at once with
no eviction pass and no per-entry bookkeeping.

The list is a liability, and the right way to hold it is to bump on the **write paths in the
document store** rather than at each of these call sites — the same reasoning that puts FTS5
synchronization in triggers rather than in application code ([`storage.md`](storage.md) §6.1).
Application-level bookkeeping covers only the write paths someone remembered.

**And "the write paths in the document store" is still a list, so it is not what shipped.** A
bump at the top of each write method is a per-method list with the same weakness one layer
down: the method nobody annotates is the one that serves a stale ranking. The counter instead
counts **committed transactions on the store's engine**, which is the closest thing SQLAlchemy
has to a trigger — a write path cannot avoid committing, and a read cannot reach it, because a
session closed without a commit rolls back. Verified in both directions: `upsert_document`,
`replace_chunks` and `soft_delete_document` each move it, and `get_document`, `list_documents`
and `search_lexical` do not. It over-counts, deliberately — a watermark write bumps it too, and
so does a write through any other handle on the same database — and over-counting costs a cold
cache while under-counting serves a ranking computed over a corpus that no longer exists.

**An in-process counter is sufficient, and the reason is a property this project already
enforces:** exactly one instance per data directory, held by an exclusive lock for the process
lifetime ([`ingest.md`](ingest.md) §6.5). The writer and the reader are the same process, so
there is no cross-process invalidation problem to solve. A design that assumed otherwise would be
solving a problem the lock file already removed.

**Why the counter and not the embedding fingerprint.** The fingerprint looks like the natural key
and is not sufficient. A mismatch causes the store to refuse to open for retrieval as well as
ingest ([`storage.md`](storage.md) §6.3), so an ordinary model change cannot happen under a
running process — but `reindex --re-embed` builds the new table alongside the old one and moves
`index_state.vector_table` in a single transaction *without restarting*
([`storage.md`](storage.md) §6.5). That is a fingerprint change under a live cache, and the
generation counter covers it because the swap bumps it. The same counter also refreshes the
cached `live_fraction` behind the over-fetch factor (§4.3).

A TTL sits underneath as a bound on staleness from anything the counter was not taught about —
five minutes, configuration, and belt-and-braces rather than the mechanism.

### 10.4 What a hit may not do

A cache hit is not a retrieval run. Its trace records `cached: true` and carries the identity of
the run that populated it, and **#15 never counts a hit as a measurement**. Latency measured on a
hit is the cache's latency, and a quality metric computed from one is a metric computed twice
from the same sample.

---

## 11. Per-stage latency and the retrieval trace

### 11.1 What is recorded

One `RetrievalTrace` per query, assembled by the runner:

| Scope | Fields |
|---|---|
| Run | route, profile + effective overrides, pipeline declaration, RRF `K`, reranker `model_id`, embed fingerprint, cached, total wall time |
| Every stage | name, wall time, candidates in, candidates out |
| `dense` | `k`, derived `k′`, `live_fraction` and whether it was measured, fetched, dropped by join, dropped by `min_score`, survived, expansions, outcome (§4.4), `resolved_id_count` and whether that count is exact, regime (§3.3) |
| `lexical` | query text as the leg received it, rows requested, rows matched, degraded flag |
| `rrf` | legs fused, per-leg candidate counts, overlap count, degraded flag |
| `rerank` | pairs scored, model id |
| Assembly | tokens used, tokens available, tokenizer identity, passages dropped and their sizes |

Per-stage latency is what makes #15's attribution possible at all: a quality difference between
two runs is attributable to a stage only if you can see which stage's cost and output changed.
The `dense` row is longer than the others because §3.3 and §4.3 both said the same thing — the
thresholds that are currently guesses get set from these fields once they have run against a real
corpus.

### 11.2 A trace is part of a run's identity

The trace exists so that two recorded results can be compared honestly, which means it has to
carry the things that make two runs *not* comparable:

- A degraded leg (§5.3).
- An `exhausted_budget` dense leg (§4.4).
- A cache hit (§10.4).
- A different pipeline declaration, `K`, reranker, profile, or embedding fingerprint.

#15 refuses the comparison when any of these differ, rather than averaging across them. That
refusal is the mechanism behind "no retrieval feature without a measured improvement": without it,
the rule is a slogan, because any two numbers can be subtracted.

**The consumer now exists.** [`evaluation.md`](evaluation.md) builds it: a pairing where either
side's trace carries an `incomparable` reason is recorded and excluded from the rate, with the
reason kept and the exclusion count printed. It reads `pipeline` off the trace as the
configuration a record names, and the per-stage spans as the attribution — which is why
`RetrievalStage` is now locked rather than merely due to be
([`contracts.md`](contracts.md) §3).

**Where it lives.** The trace is a return value, surfaced through `--json` and the API, and
consumed by #15's harness, which writes its own versioned result artifacts. It does **not** go
into `query_logs` — that table's `response_time_ms` is whole-query product telemetry, and a
per-stage trace there would be a schema change in service of a consumer that keeps its results in
the repository anyway. If operations later wants per-stage latency persisted
([#14](https://github.com/mgd43b/manicule/issues/14)), it is one nullable JSON column and an
additive migration — it invalidates nothing and rebuilds nothing.

---

## 12. The three profiles, concretely

Named settings are only useful if the names mean something specific. What actually differs:

| | `fast` | `balanced` | `precise` |
|---|---|---|---|
| Candidates per leg | 10 | 20 | 50 |
| Dense rows fetched (floor, §4.3) | 30 | 60 | 150 |
| `min_score` on the dense leg | 0.35 | 0.35 | 0.35 |
| Fused set, before rerank | ≤ 20 | ≤ 40 | ≤ 100 |
| Cross-encoder | **not loaded** | 20 pairs | 50 pairs |
| Passages into context | 3 | 5 | 10 |
| Context / history tokens | 4096 / 512 | 5632 / 1024 | 12288 / 2048 |
| Smallest generator window that fits | 8k | 8k | 16k |
| Model loads on the query path | 1 (embedder) | 2 | 2 |
| Confidence ceiling (§8.3) | **0.70 — cannot reach `high`** | 1.0 | 1.0 |

The differences that matter are the last three rows. `fast` is the profile where the second model
is never loaded — that is the latency difference, and everything else is a rounding error beside
it. `precise` is `balanced` with 2.5× the reranker cost and a much lower similarity floor, which
is a bet that the cross-encoder can rescue passages the dense leg nearly discarded. Whether that
bet pays is a #15 measurement and one of the first worth running, because it is the cheapest
change to make if it does not.

### 12.1 Where the token budgets come from

The three budgets this document first carried — 8192 / 16384 / 32768 — were inherited rather
than derived, and they were wrong in two ways that only showed up when something finally
compared them against anything.

**They were unreachable.** A passage reaching a context is at most 448 embedder tokens (§7.3).
Ten of those, at a vocabulary ratio of 2.0 — far past any plausible ratio between a
SentencePiece embedder vocabulary and a BPE estimate for the generator — plus per-passage
framing, is about **9 300** generator tokens. `precise` therefore reserved 32768 for something
that could not exceed a third of it. A budget nothing can reach is not a rail; it is a number
that tells a reader nothing.

**And `precise` did not fit its own default model.** `32768 + 2048 + ~400 + 1024 = 36 240`
against `qwen2.5:14b`'s 32768-token window, which §7.4 turns into a startup refusal. `balanced`
had the same shape one size down: 16384 + 1024 against an 8k local model is an assembled context
twice the size of the window, on every query.

So each budget is now set from what its own `final_top_k` can actually hold, with 1.2–1.5×
headroom:

| | `fast` | `balanced` | `precise` |
|---|---|---|---|
| Largest possible context, generator tokens | 2 784 | 4 640 | 9 280 |
| `context_tokens` | 4 096 | 5 632 | 12 288 |
| Headroom | 1.47× | 1.21× | 1.32× |
| Total with history, prompt and reserve | 6 032 | 8 080 | 15 760 |

Two consequences worth remembering, and the first is the one an operator feels: **`fast` and
`balanced` both fit an 8k window**, prompt and generation reserve included, and **`precise`
needs 16k** — the default generator's 32768 fits all three with room to spare. The second is
that the rail is now a real one: the fitter still cannot bind on a shipped profile, but it is
within a factor of 1.5 of doing so rather than a factor of five, so an override that pushes past
it is an override a reader can see coming.

The similarity floor no longer varies by profile and is no longer inherited: §4.5 carries the
sweep that set it to 0.35 everywhere, and §8.4 the reason the relevance decision lives in
confidence rather than in a threshold.

Every knob is overridable per field (`ProfileConfig` + `rag.overrides`), and overrides start from
the named profile so changing one cannot silently move another.

---

## 13. Deliberately deferred, and the measurement that would un-defer each

Ticket #6 lists nine features carried by the prior art, plus the learned-sparse leg recorded
against #6 separately. **None of them is known to help**, because the harness that would know
scores at random chance — its test embedder is `sin(sum of character codes)`. That is a statement
about the evidence, not about the features.

manicule's own harness is [`evaluation.md`](evaluation.md), and the first property it was built
to have is that this cannot happen to it: a configuration is put through a known-answer probe and
measured against what guessing would do before any preference is recorded, and one that cannot be
distinguished from guessing gets no report at all. The measurements in the table below are run
through it.

The rule is `PLAN.md` §8's: each ships with a measured improvement on #15's fixed query set, or
does not ship. A rule is only enforceable if it says what would count, so:

| Feature | What would have to be measured, on #15's fixed query set |
|---|---|
| **HyDE** | nDCG@10 of the fused list, on and off, restricted to the regime it claims — queries where the dense leg's top-1 cosine is below a floor. Report added p50 latency in the same table: it costs a generation call per query, so a small win that doubles latency does not ship |
| **Multi-query expansion** | recall@50 of the union of legs, N=3 against N=1 — **at equal total candidate budget**. Compared against simply raising `candidates` by 3×, or it is buying recall with fetch rather than with expansion |
| **Query decomposition** | Only meaningful on multi-hop questions, and #15's query set has no labeled multi-hop subset. Building that subset is the first deliverable; then nDCG@10 on it alone, since averaging it into the full set will hide the effect either way |
| **Intent classification** | Its only proposed consumer is per-intent context allocation, which does not change retrieval at all — the same passages are retrieved and the window is divided differently. Needs an *answer*-quality metric, which #15 does not have yet. Nothing to measure it with today, and that is the finding |
| **Cross-lingual expansion** | The null hypothesis is that it adds nothing, because bge-m3 is already one multilingual space ([`embeddings.md`](embeddings.md) §1.2). recall@20 on query-in-A / gold-passage-in-B pairs. A likely outcome is that this measurement retires the feature rather than admitting it |
| **Parent-document retrieval** | nDCG@`final_top_k` with parents substituted for chunks, **plus a citation check**: the anchor must still resolve to the quoted span. A parent that widens the citation is a regression at any nDCG, because `contracts.md` §1 does not trade accuracy of location for relevance |
| **Propositions** | Changes what is indexed and therefore the `ChunkFingerprint`, so adopting it costs a re-chunk *and* a re-embed — rungs 3 and 2 of the blast-radius ladder, and no re-fetch, because retained original bytes keep rung 4 out of it. recall@10 has to improve by enough to justify that, and it changes what a citation points at — same anchor obligation as above |
| **Prompt compression** | Not a retrieval feature: it changes what the generator sees. Answer quality at fixed context tokens, and a hard check that no cited span was rewritten — PR #32 forbids middleware rewriting cited text and compression is the same act at the other end of the pipeline |
| **Hallucination guard** | Precision *and* recall of the guard itself against a labeled set of grounded and ungrounded answers. Recall alone is the trap: a guard that suppresses correct answers is worse than no guard, and only precision shows it |
| **BGE-M3 learned-sparse leg** | Recorded against #6 already. nDCG@10 with learned-sparse replacing the FTS5 BM25 leg — **on a multilingual corpus first**, where Porter stemming is English-only and the current lexical leg is at its weakest. Also a runtime change: neither installed backend exposes the head ([`embeddings.md`](embeddings.md) §1.4) |

Two structural notes. Every one of these is a stage or a leg, so measuring it is a configuration
change and a run — which is the property §2.4 exists to protect. And the fusion stage taking its
leg names from configuration (§5.2) is what makes the last row a two-line config edit rather than
a rewrite of the fusion code.

## 14. Glossary-aware entity and acronym retrieval

The problem `bugs/bug2.md` states: a glossary defines an acronym that is also an ordinary English
word, and a question naming it retrieves passages that *use* the term rather than the one that
defines it.

### 14.1 What the failure actually is, measured

It is not a threshold that needs adjusting, and it is not visible on a small fixture. Measured
with BGE-M3 over a synthetic corpus (`tests/glossary/corpus.py`), the same question against three
progressively more realistic versions of it:

| Fixture | Rank of the definition |
|---|---|
| The glossary line as its own short passage, thirty ordinary uses of "now" around it | **1 of 33** — no failure at all |
| The glossary as one chunk holding 25 entries | **1 of 31**, cosine **0.4655** — ranked fine, but below the §8.4 noise floor, so confidence says `none` |
| The above, plus fifteen passages that *use* the acronym in running text | **15 of 61** — absent from a ten-passage context |

Two ingredients are needed and a fixture missing either proves nothing. **The definition is
diluted inside a chunk**: chunking is 512/64, so a glossary page arrives as one chunk and any one
definition is a fortieth of its vector. **The term is used far more often than it is defined**,
which is what every corpus looks like once a term exists — and those usage passages are short, on
topic, and contain the acronym.

The consequence for the design is the important part. The expanded *embedding* does not rescue
this: searching for `Network Operations Workspace` alone ranks the glossary 8 of 61, which would
not reach a ten-passage context either. What rescues it is that **an exact alias hit is a lookup,
not a search** — ingest recorded which chunk defines the term, so the definition is fetched by id
and promoted with its provenance rather than made to win a similarity contest.

### 14.2 Where it lives, and why it is not a stage

`RetrievalStage` is locked (§2.3). Expansion does not need it widened, because what it produces
is a **second query**: the declared pipeline runs over it unchanged. So the retriever's list of
things that are not stages grows from three to four —

    Query → router → glossary → L1 cache → declared stages → context assembly → confidence

— for the same reason as the other three. A stage that took one query and searched two would be a
stage whose output could not be replayed from its input, which is exactly what §2.3 protects.

The second pass costs a full second run of the pipeline, and it is paid only when an alias fires.
`RetrievalTrace.glossary` records whether it did, so the cost is on the record rather than
inferred from a latency that doubled.

### 14.3 Detection, at ingest

`manicule.ingest.glossary` reads definitions out of chunks — chunks rather than blocks, because a
definition has to be citable and a chunk is what a citation resolves to. Six written forms: em
dash, colon, parenthetical, Markdown definition list, two-column table row, and a heading followed
by its definition (read both from the text and from the breadcrumb, because the structural chunker
lifts headings out of the text entirely).

The hard part is refusing prose. `Note: the scheduler restarts nightly` has exactly the shape of a
real definition, and two independent gates apply:

1. **Shape.** The term must be *written* like an abbreviation — predominantly upper case. This is
   what rejects `Note:` without consulting anything else, and no amount of glossary-looking
   context can buy it off.
2. **Confidence.** Everything else is evidence: the written form, whether the expansion's initials
   spell the term, and whether the document says it is a glossary. The threshold is 0.6, set from
   the combinations rather than from a sweep — a dash form whose initials match clears it alone
   (0.80), a dash form on a glossary page clears it alone (0.60), and a colon form with neither
   does not (0.40).

**Detection is versioned, and the version is not the parser's.** Every rule in this section is
manicule's own and changes independently of parsing, chunking and embedding — so an index can
report a current fingerprint for all three while serving entries produced by rules corrected
several times since. `documents.glossary_fp` records which detector decided a document's
entries, including when it decided there were none, and `manicule document reindex
--stale-glossary` recomputes the documents that disagree with the installed one from chunks
already stored: no parser, no connector, no embedder. [`ingest.md`](ingest.md) §10.2 has the
fingerprint's inputs, the disabled-detection semantics and the migration policy.

**`rag.glossary.min_entry_confidence` is a query-time floor and not part of that lineage.** It
decides which stored entries a query will act on, which is why raising it takes effect against a
corpus already indexed — the remedy available to somebody who cannot re-ingest. The threshold
that decides what is *stored* is the 0.6 above, and moving that is a detector change like any
other.

### 14.3.1 Where the term ends and the description begins

A glossary line is often a definition followed by prose about it:

```
NOVA — Network Operations Visibility Assistant, a service used to correlate operational signals.
```

Thirteen words on the right of the dash. `MAX_EXPANSION_WORDS` is 10, so this was refused whole
and **no entry was written at all** — which is upstream of retrieval entirely: nothing to expand
with, nothing to promote, nothing to cite. No amount of ranking work reaches it.

The obvious fix is to cut at the first comma and score what is left. That is wrong in the one
direction that matters, and the arithmetic says so. `API - when enabled, the process starts
automatically.` truncates to `when enabled` — two words, which every length rule in the module
likes *better* than the sentence it came from. **Truncation removes the very thing that was
refusing the line.** So the cut has to be earned before it is made, never scored afterwards.

Only one signal is strong enough to award it. `core_expansion` tries the whole right-hand side
first and keeps it if its initials spell the term; otherwise `_phrase_after` walks the description
boundaries — comma, semicolon, end of sentence — and takes the first prefix whose initials spell
it. `Network Operations Visibility Assistant` spells NOVA and the description does not, and that
agreement between two strings is the only thing here that knows where a term ends.

**This is one rule applied from both ends, not two mechanisms.** `_phrase_before` already resolves
a parenthetical's *left* boundary the same way — `The Network Operations Workspace (NOW)` has no
left delimiter, so it asks the acronym which suffix spells it. `_phrase_after` has no right
delimiter and asks which prefix does. Shortest wins in both, for the same reason: the first span
that spells the term is where the term stops, and a longer one that also spells it has swallowed
the sentence around it. Read them together; change one and read the other.

Where there is no initials evidence there is no cut. `CPU — central processor, the part that
executes instructions` keeps its description, because nothing in the text says where the expansion
stops and guessing would store a phrase the source never wrote. This is a known limit, not a claim
of completeness.

The **passage is never trimmed**. Only the stored expansion is; the chunk the entry cites still
contains the whole line, so a citation resolves to text the document actually holds.

### 14.3.1.1 What counts as initials, and the bound on each widening

Ordinary word initials miss two conventions technical writing is full of, and both were failing
the same way — the whole right-hand side kept and then refused as too long, so no entry at all:

```
SORT  — SecOps Reliability Toolkit, a package that operations teams install on a host.
SaFeR — Service Failure Reporter, a component that groups related failures together.
```

`SecOps` is one word and two of the term's four letters; `SaFeR` looks up as `SAFER` and its
expansion supplies three initials, not five. Two comparison forms are added, and **because initials
agreement is the sole authority to cut a prefix off a right-hand side, each one widens the
authority to truncate** — which is the dangerous direction, since `when enabled` is shorter than
the sentence it came from and therefore looks *more* like an expansion. So each is bounded:

| Form | Read from | Bound on it |
|---|---|---|
| Word initials | Whitespace, `/`, `-` | Unchanged |
| Component initials | One camel boundary: lower-case or digit followed by upper case | That boundary only. `HTTPServer` is **not** split — a second rule is more authority to cut |
| Initial skeleton | The upper-case and numeric characters of a deliberately mixed-case display | Initial capital, at least one lower-case letter, and `MIN_SKELETON_LENGTH` = 3 characters |

The two run in opposite directions and only one needed a floor. Splitting a compound can only
*lengthen* a phrase's initials, and a longer string is satisfied by fewer terms — so it demands
more agreement, not less. A skeleton is *shorter* than the key it stands beside, which is a weaker
constraint: fewer words have to agree before a prefix may call itself the expansion. Swept over
the labeled corpus in `tests/glossary/skeleton_corpus.py`, 18 positives and 17 negatives:

| Bound | Precision | Recall | Boundary precision | What moved |
|---|---|---|---|---|
| 2 | 0.947 | 1.000 | 1.000 | `WEB = 'when enabled'`, a false positive |
| **3** | **1.000** | **1.000** | **1.000** | — |
| 4 | 1.000 | 0.944 | 0.941 | `AuDiT` lost |

Three is pinned on both sides by one case each, and `SaFeR` skeletons to exactly three characters
— zero margin, the same margin `_UPPERCASE_SHARE` has on the same term's shape.

**The intuitive argument for the floor is not the one that holds.** Short forms ought to cut prose
more often, so the prose ought to show it — and it does not: across the forty-five ordinary
passages in `tests/glossary/corpus.py` there is not one two-word description-boundary prefix, and
the distribution of prefix-initial lengths peaks at six. What condemns a bound of two is a
constructed line, `WEb - when enabled, the process starts automatically`, which is the §14.3.2
`API` negative under a term whose skeleton is `WE`. Related and **not** fixed here: a two-letter
*key* has always had this authority — `core_expansion('WE', 'when enabled, …')` returns `when
enabled` on `origin/main` — because `MIN_ACRONYM_LENGTH` is 2. Narrowing that would refuse `IO`,
`ID` and `DB`, and it is not what this change is about; it is recorded because it is the reason
the floor belongs on the skeleton specifically, which is authority granted *in addition* to a key.

**Arbitrary subsequences are refused structurally rather than by a rule.** Two closed sets are
built independently — the term's spellings from the term, the expansion's initials from its own
token boundaries — and matching is set intersection over whole strings. Nothing scans an expansion
looking for a term's letters, and nothing can, because the function that reads the expansion is
never told what term it is about to be compared against. A test asserting that some unrelated
string fails to match would prove none of this, because a scanning matcher refuses those too. So
`test_a_free_subsequence_scan_would_match_and_this_matcher_refuses` writes the scanning matcher
out, shows it accepting `Storage Operations Roster` for `SORT` and `Service for Escalation Routing`
for `SFR`, and shows this one refusing both.

`HTTP — HyperText Transfer Protocol, used by every browser` was this section's example of the
conservative fallback and is now cut correctly, which is why the paragraph above uses `CPU`. That
is a documented limitation closing, not a behavior drifting: `HyperText` is a compound and its
components spell the term exactly.

**A term is three strings and the third resolves nothing.** The *display* is what the source
wrote, stored verbatim; the *lookup key* is `normalize_acronym` of it and is the only one anything
resolves through, at ingest and at query time alike; the *initial skeleton* is a comparison form,
computed where it is compared and stored nowhere. `SaFeR` is found by `safer` and not by `SFR`,
and two definitions of `SAFER` remain a conflict whatever their capitalization says — §14.5's rule
that nothing picks a winner is not weakened by there being a new way to spell the loser.

Measured end to end over that corpus: detection precision and recall **1.000/1.000** against
`origin/main`'s 1.000/0.833, expansion-boundary precision **1.000** against 0.933, zero false
positive entries, hit rate **13/14** at k=1 and k=3 and **14/14** at k=10, unsupported-query
rejection **6/6**, and **no confidence band or score moved on any of the 24 queries**. The one
question that does not reach rank 1 is `What does SecOps Reliability Toolkit stand for?`: it names
the expansion and never writes the term, so no alias fires and the glossary does nothing for it.
That is a real limit of a lookup keyed by term, and the query is kept in the corpus rather than
dropped for a rounder number.

### 14.3.2 Why exact lexical matching is not enough to call something a definition

A line's shape is not evidence that it defines anything. On a page titled "Glossary" a spaced
hyphen scores 0.45 from its form plus 0.15 from the page — exactly the 0.60 threshold — so
*every* upper-case token followed by a dash is admitted there on the strength of the page alone.
Measured before this rule existed, both of these were recorded as real glossary entries:

```
NOTE - this paragraph describes an operational consideration, not a term.
API  - when enabled, the process starts automatically.
```

Nine words and six, so `MAX_EXPANSION_WORDS` never fired on either and nothing else looked.

What refuses them is their **first word**. `has_a_refused_opening` compares it against
`_NEVER_OPENS_AN_EXPANSION`, a short list of subordinators and of words that point rather than
name. It never overrides initials evidence: if a phrase's initials spell the term, the two
strings agree about what the term is, which is stronger than anything one word can say — which is
why `ONCE — Operational Node Configuration Engine` and `WHEN — Workload Health Event Notifier`
survive being their own counter-examples.

**What the rule does not test, stated because an earlier draft of this section claimed it did.**
It is not a test for a noun phrase, and the two lines above are not counter-examples to one:
`this paragraph` *is* a noun phrase. What actually disqualifies both is the finite verb —
`describes`, `starts` — and nothing here looks for a verb. Finding one is a parser's job, this
module is deterministic by construction and has no parser to call, and a word list that pretended
otherwise would be a rule whose justification is wider than its mechanism. The claim is therefore
narrowed to what the list can support: these particular words do not begin expansions, one word
at a time. Prose opening with an ordinary noun or an imperative verb — `WARNING — do not edit
these records by hand.` — is admitted, and that is a known gap.

**Do not close it by adding `note`, `warning` and `caution` as refused terms.** The objection is
not that such a list would overclaim — it would not; admonition labels are document furniture and
a list of them tests exactly what it names. The objection is the trade. `WARNING — a log level
indicating a recoverable condition` is an ordinary definition in operations and logging
documentation, which is the corpus this project exists for: the gate buys one false positive and
sells a false negative in the domain most likely to be indexed. That stays a bad trade however
carefully the list is drawn.

The fix, if a real corpus ever needs one, is **structural and belongs in the parser**. An
admonition in Confluence storage format is `<ac:structured-macro ac:name="warning">` and not a
`TERM — text` line at all, and `manicule.connectors.macros` already reads those elements by name.
By the time text reaches the detector the distinction has been flattened away, which is why no
word list can recover it.

**Recall is measured as loudly as precision, because this rule fires exactly where the safety net
is absent.** It only runs when initials evidence is missing, which is also where a large share of
legitimate definitions live: `K8S — Kubernetes` spells nothing, nor does `CPU — central
processor`. The measurement found the list refusing real definitions — `ITSM — IT service
management` and `ITIL — IT infrastructure library`, because `IT` casefolds onto the pronoun `it`.
That is why the first word is put through `acronym_shaped` before the list is consulted: the same
gate that tells `NOW` from `Note` on the left of the dash tells `IT` from `it` on the right of it.
Real definitions kept went from **24 of 26 to 26 of 26**, with prose refusal unchanged at 5 of 5.

`Today - the system is operating normally.` never reaches any of this: one of five letters is
upper case, so the shape gate refuses it.

**Entries already written by the defect do not clear on their own.** Detection runs at ingest, and
a re-sync of a document whose bytes have not changed is skipped before it reaches the detector —
measured at both levels, including a source that issues a new version token for an unchanged body,
which level 2 catches by content hash:

```
initial ingest: status=indexed entries=['NOW']
after planting:  ['NOTE', 'NOW']
re-sync same token:  skipped='hash' entries=['NOTE', 'NOW']
re-sync new token:   skipped='hash' entries=['NOTE', 'NOW']
forced (re_parse):   skipped=''     entries=['NOW']
```

A forced pass clears it, because `_store_definitions` replaces unconditionally including with an
empty list. That is `manicule index <path> --reindex`, or `reindex.re_parse`, both of which pass
`force=True` — an existing flag on an existing command, so there is no remediation path to build
and none was built.

### 14.4 Scope is a correctness property

Entries are stored in `glossary_entries` and `glossary_aliases`, scoped through `document_id` and
nothing else: a document id is derived from the workspace, so a copy of the workspace on the entry
row could only ever be a second answer to a question the foreign key already settles.

Collection membership is **not** copied onto the entry, because a collection's contents change
without any glossary being re-ingested. A lookup resolves it through `resolve_filter` — the same
function collection-scoped search resolves through, so there is one notion of what a collection
contains rather than a second one that drifts.

`kinds` and `langs` are projected out before the lookup, exactly as `join_filter` projects them out
before a query over `documents` (§3.3). They restrict which *chunks* a search may return, and this
lookup returns vocabulary; a query for table passages is not asking to be kept ignorant of what an
acronym in its own text means. They are applied where they belong — to the promoted passage, which
is a chunk — so a chunk-level restriction still decides what comes back.

### 14.5 Over-expansion, which is what stops this making retrieval worse

An occurrence expands only when one of three rules admits it, and the rule that fired is recorded:

| Rule | Fires when |
|---|---|
| `exact_case` | The query wrote the term the way the glossary writes it — `NOW`, not `now` |
| `definitional_frame` | The query asks *about* the token rather than using it: "what is now", "define now", "what does now stand for". A question is not a use |
| `unambiguous` | The term is not an ordinary English word, so no other evidence is needed |

The third consults a short list of common English words (`manicule.retrieval.homographs`),
extensible per deployment through `rag.glossary.homographs`. It is deliberately short and short in
one direction: over-expansion is only expensive for words a corpus contains dozens of innocent uses
of, and a longer list would start refusing legitimate terms.

Nothing here is applied to a document. manicule does not rewrite what it indexed, so the worst an
over-broad rule can do is run one extra search whose provenance is on screen.

**Conflicts are never resolved.** Two definitions of one term in scope expand nothing and are
reported as a conflict with both sources. Highest detection confidence, most recent and
first-alphabetically are all defensible and all silent, and a silently chosen definition produces
an answer that is fluent, cited, and about the wrong thing — the one failure mode a glossary
feature has that plain search does not. Narrowing the scope is not a tie-break either: a
collection holding both definitions holds a disagreement.

The **firing rules run first**, and the order was corrected after watching the command line render
the other way round. A term used as an ordinary English word does not expand whether or not the
corpus disagrees about it, so reporting a conflict there put a glossary banner above the results
for *should I restart the daemon now* — a question that was never about the term. A banner that
appears on questions it does not concern is one readers learn to skip, which costs exactly the case
it exists for.

### 14.6 What it does to confidence: nothing numeric, by construction

A promoted definition carries **no leg score**. `evidence_per_passage` (§8.2) skips a passage the
dense leg never ranked, so promotion contributes nothing in either direction — it cannot
manufacture evidence, and it cannot be mistaken for a cosine nobody measured. Where a passage was
found by both the original and the expanded search, the better of the two opinions is kept per leg,
which is the only combination that cannot let the feature quietly *lower* a reported confidence.

Measured on the fixture above, `What is NOW?` reports **0.2117 `low`** before and after. That is
non-zero, which is `bugs/bug2.md`'s second acceptance criterion, and it is low because a chunk that
is mostly about twenty-four other terms genuinely is weak evidence. Making the number larger would
mean letting a detection confidence stand in for a cosine, which §8 refuses everywhere else.

### 14.6.1 An explicit definition is a classification, not a quantity

The number staying still left a sentence that was simply false. A run could report band `none`
with "*nothing in this corpus resembles your question*" while showing, at rank 1, the glossary
entry defining the exact term the question named. That is not a threshold set too high; it is two
paths that never spoke, one reading a cosine and the other reading a lookup.

`Confidence.explicit_definition` is what they say to each other. It is a **boolean
classification**: it enters no weighted sum, moves no band, and leaves `components` and `ceiling`
untouched. Its whole effect is that `NOTHING_RESEMBLES` is replaced by `DEFINITION_CITED`, which
says what was found instead of asserting the corpus is empty of it.

It is set only when all three of these hold, and dropping any one turns it into a boost:

| Condition | Why it is necessary |
|---|---|
| A glossary entry fired for the term | Already means the entry cleared the confidence floor and was not contested — a disagreeing pair is reported as a conflict and fires nothing, so a contested term never reaches this |
| The query asked what the term **means** | Tested with `definitional_frame`, not by reading the recorded `MatchReason`. `raise a ticket in NOVA today` records `exact_case`, the same reason `What is NOVA?` records, so the reason cannot tell them apart |
| The defining passage is **in the context** | "We found a definition" and "we are showing you one" are different claims, and only the second may contradict "nothing here resembles your question" |

The second condition is what keeps exact lexical overlap from establishing support: a query that
merely *mentions* a defined term gets the definition promoted and still reports no evidence,
because it did not ask.

**No numeric glossary component ships**, and that is a measured result rather than caution. On the
evaluation corpus the classification separates the two populations perfectly — seven definitional
queries true, five non-definitional false, zero false positives — but that is twelve queries over
three defined terms on one synthetic corpus, which is nowhere near enough to set a weight. Setting
one from it would be adjusting a constant to fit the motivating example, which §8.4 forbids by
name. So there is no zero-weight component and no unused configuration key to find later and
wonder about; when there is a corpus big enough to calibrate against, the classification is
already there to weigh.

This is also why an explicit definition is not the same thing as a high similarity. Similarity
asks how much the passages *resemble* the question, which is a property of two embeddings.
"Someone wrote down what this term means, and here is the line" is a property of the corpus's
structure, and no cosine expresses it — which is exactly why the two had to be reported side by
side rather than added together.

#### 14.6.2 It leaves retrieval as a boolean

`explicit_definition` is on the public `search` and `ask` payloads, on every surface that
publishes them, and on the streamed answer's `final` frame. It is **copied** from the
`Confidence` above and never recomputed at the boundary: the three conditions need the query
text, the glossary lookup and the assembled context, and a second opinion assembled from
`confidence_reason` would be a machine contract parsing English prose written for a person.

Two things can make the payload's `false` where this module's is `true`, and both are absences
rather than disagreements. A query the router answered directly carries no `Confidence` at all,
so there is nothing to copy. And a definition whose document became unreadable between the
lookup and the render loses its provenance — the payload model refuses to claim a citation it
cannot name, so the claim goes with it. [`surfaces.md`](surfaces.md) §5 is the contract.

### 14.7 The cache key

`cache_key` carries the expanded query form. The generation counter catches a definition being
added, because that is a row; it catches neither expansion being switched off nor a second
definition turning a term into a conflict, and both change the ranking without changing the corpus.

### 14.8 Multi-word terms: a design note, not a plan

A glossary is full of terms that are not abbreviations — `Change Freeze Window`, `Golden Path`,
`Blast Radius`. None of them can be detected today, and this records **why the obvious fix is not
the fix**, so that whoever takes it starts from the right question.

**The gate that refuses them is not the one it looks like.** `acronym_shaped` is the visible
rule and it does refuse them — `Golden Path` is 0.20 upper case against a 0.6 share — but
deleting it would change nothing, because `normalize_acronym` in `core/glossary.py` refuses them
independently and it is the one that decides the *key*:

| Surface | Upper | Length | Charset | `normalize_acronym` |
|---|---|---|---|---|
| `Golden Path` | `GOLDEN PATH` | 11 ≤ 12 | space not in `isalnum() or -&/` | `''` |
| `Blast Radius` | `BLAST RADIUS` | 12 ≤ 12 | space refused | `''` |
| `Change Freeze Window` | `CHANGE FREEZE WINDOW` | **20 > `MAX_ACRONYM_LENGTH`** | space refused | `''` |

`detect_in_chunk` then skips on `if not acronym`. So the question is not "how upper-case must a
term look" but **what a glossary lookup key is allowed to be**, and that is a different question
with a wider blast radius.

**What widening the key would touch, all at once.** The key is not a detection detail; it is the
string everything resolves through, at ingest and at query time alike (§14.3):

- **`core/glossary.py`** — `MAX_ACRONYM_LENGTH` is 12 because "above this the token is a word,
  not an abbreviation", which is the assumption being discarded rather than tuned. The charset
  rule would have to admit the separator, and `GlossaryEntry.acronym` carries a `min_length`
  constraint that decides what may be persisted at all.
- **The query normalizer** — a query is tokenized and each token normalized before lookup
  (§14.5). A multi-word key cannot be found by a single-token lookup however it is stored, so
  matching would need an n-gram pass over the query, and the homograph rules that stop `now`
  expanding every sentence would need an equivalent for phrases.
- **Storage** — keys and aliases are indexed columns, and `glossary_entry_id` derives a row's
  primary key from its content.
- **Detection** — every written form's left-hand side is `_TERM`, bounded by
  `MAX_ACRONYM_LENGTH` and admitting no spaces.

**And the evidence model has nothing to offer a phrase.** `INITIALS_EVIDENCE` is the strongest
signal detection has, and it is meaningless here: `Golden Path` has no initials to spell and
nothing to agree with. What remains is the form weight and the page-level context — precisely
the pairing §14.3 refuses for headings, because it admits anything shaped like the form. So
multi-word terms do not merely need a wider key; they need **a source of evidence this feature
does not currently have**, and choosing one is the real work.

**Not started, deliberately.** This is a note about a change nobody has been asked to make. The
measurements behind it are in the nine-shape sweep: 0 of 3 multi-word terms detected, refused by
`normalize_acronym` rather than by `acronym_shaped`.

---

## Appendix A: decisions this document made

Calls made in the absence of a stated position.

| Decision | Where |
|---|---|
| `RetrievalStage` is **not** widened; the three #1 rejections re-argued against a working design | §2.3 |
| Per-stage diagnostics travel by `contextvars` trace frame, with a no-frame conformance run | §2.3 |
| Stages run sequentially; concurrency deliberately declined for measurement clarity | §2.2 |
| Stage names unique within a pipeline; the container refuses duplicates | §2.2 |
| `Filter` settled: `workspace_ids` required/non-empty, `sources` and `langs` set-valued, `extra` removed | §3.1 |
| Cross-workspace search is N scoped queries merged on cosine, never one unscoped query, never RRF | §3.2 |
| The pre-filter/post-filter split is a rule with recorded inputs, not a constant | §3.3 |
| "No join-requiring field set" and "resolved to the empty set" are opposite instructions, not one case | §3.3 |
| The hydrating join lives *inside* the dense stage, so scope is a per-stage invariant | §2.4, §4.2 |
| Over-fetch is derived from a measured `live_fraction`, with floor, cap and row cap | §4.3 |
| `live_fraction` is workspace-scoped only, so it stays cacheable per `(generation, workspace)` | §4.3 |
| Three distinct shortfall outcomes; `exhausted_budget` is a defect and the others are not | §4.4 |
| `min_score` applies to dense cosine only, is 0.35 everywhere, and is a junk filter rather than the relevance decision | §4.5, §8.4 |
| RRF is rank-only: no score weighting, no leg weighting, no normalization | §5.1 |
| Per-leg ranks are recovered from `Candidate.scores`; the fusion stage is configured with leg names | §5.2 |
| A missing leg is recorded and makes the run non-comparable rather than merely logged | §5.3 |
| The reranker raises rather than passing through, and truncates to what it scored | §6.1 |
| `bge-reranker-v2-m3` by default, because a monolingual reranker would undo the embedder's choice | §6.2 |
| Context assembly is not a `RetrievalStage` — closes the second open item in `contracts.md` §6 | §2.4, §7.1 |
| Two token counters, named; `Chunk.token_count` must never be used for context fitting | §7.2 |
| `tiktoken` by encoding name, no sampling, safety factor, then calibrated against `prompt_eval_count` | §7.2 |
| Assembly includes whole passages or none; skip-and-continue, never truncate | §7.3 |
| Profile-versus-generator window is a startup refusal | §7.4 |
| Confidence is a retrieval-support score with named components, no fallback term | §8.2, §8.3 |
| `fast` cannot reach `high` confidence, by arithmetic and on purpose | §8.3 |
| A degraded leg suppresses the agreement component rather than scoring it zero — confidence never blames the corpus for a pipeline fault | §8.2 |
| A description is separated from an expansion only where initials evidence says where it begins; no evidence, no cut | §14.3.1 |
| Prose is refused on its first word against a closed list, with an abbreviation exemption; a verb list is declined as a thing a word list cannot do, and the gap it leaves is recorded | §14.3.2 |
| Entries written by the old defect persist until a forced pass; no remediation path was built, because `--reindex` already is one | §14.3.2 |
| An explicit definition is a named classification with no weight, not a confidence component; no weight is set until a corpus can calibrate one | §14.6.1 |
| Router: full-match only, tuned for precision, no citations and absent confidence on a direct route | §9 |
| L1 caches ranked ids, not content, so a hit cannot leak a deleted or foreign chunk | §10.1 |
| L1 key excludes conversation history and includes the generation counter, `Query.limit` and the pipeline identity | §10.2 |
| The generation counter is bumped on the document store's write paths, not at each caller | §10.3 |
| Invalidation by in-process generation counter, justified by the one-instance lock; re-embed is the case that forces it | §10.3 |
| The trace is a return value, not a `query_logs` column | §11.2 |
| `Query.limit` bounds what is returned to a caller; `final_top_k` bounds what enters the context | Appendix B |

## Appendix B: what the merged documents did not cover

Places this design had to decide something no merged document had a position on. Each is called
out here because a reader of the other documents will not have seen it coming.

- **`Query.limit` versus `ProfileConfig.final_top_k`.** Both shipped, both mean "how many
  candidates come out", and neither document reconciles them. Settled as two different consumers:
  `limit` is what a `search` call returns to a human, `final_top_k` is what an `ask` call puts in
  the model's context. Retrieval depth is `max(limit, final_top_k)`. Neither type changes.
- **Where `min_score` is applied.** `ProfileConfig` defines the value and nothing said what it is
  a floor *on*. Two of the three candidate answers return an empty result set for every query
  (§4.5).
- **The profile/context-window cross-check.** Nothing compared manicule's token budgets against
  the generator's window (§7.4).
- **`contracts.md` §6's second open item**, whether `Context` assembly is a stage. Settled here
  as "not a stage" (§7.1). Both of §6's remaining questions were retrieval questions, so §6 is
  now a record of what settled them rather than a list of what is open.
- **`storage.md` §6.6 is superseded, not contradicted.** It proposed a `Filter` shape and said
  plainly that it was not closing the question; §3 closes it with a different shape and gives the
  reasoning for each difference. The section keeps its value as the record of what was considered
  before storage was built, which is the same relationship `embeddings.md` §3.1 has to its own
  earlier draft.

## Appendix C: filed, not deferred

| Ticket | What | Why not here |
|---|---|---|
| [#36](https://github.com/mgd43b/manicule/issues/36) | **Reshape `Filter` to the settled form** (§3.1, §3.4), and add `assert_pipeline_enforces_scope` | It changes `manicule.core.retrieval` and both stores — merged code owned by #1 and #2 — while two implementation tickets are in flight. It is also worth landing as its own reviewable change, because it moves a security boundary |

## Appendix E: what #6 changed in this document

Implementation is the only review a design gets that can disagree with it. Six places where it
did, each fixed above rather than noted:

| Where | What building it showed |
|---|---|
| §3.2 | The cross-workspace **merge rule** is a retrieval decision and is settled; the **fan-out** needs a workspace registry that is team-mode storage plumbing. The store's refusal is what holds the line meanwhile, and it names its own remedy |
| §3.3 | Resolution has to stop one row past `prefilter_id_limit`, and the count it records is then a lower bound. A figure recorded as exact when it is not would skew the distribution the threshold is to be set from |
| §4.1, §11.1 | The lexical trace records the query text the leg was **given**, not the escaped match string. Escaping belongs to the store; reproducing it in the stage would import a database driver into a package that needs none and hardcode one store's query language into a swappable leg |
| §7.3, §12.1 | The token budgets were inherited, unreachable by a factor of three to five, and `precise` failed its own startup cross-check against the model this project ships with. They are now derived from what each profile can hold |
| §8.2 | The suppression rule is about the **cause**, not about one named term: a degraded dense leg has to suppress the similarity component for exactly the reason a degraded lexical leg suppresses agreement |
| §10.3 | "Bump on the write paths in the document store" is still a list. The counter counts **committed transactions**, which a write cannot avoid and a read cannot reach |
| §10.1 | A cache hit re-applies the **document-level** half of the filter, not the whole of it. `kinds` and `langs` are chunk properties with no column in a query over `documents`, so passing them made a query that worked on a miss raise on a hit. Nothing is dropped by narrowing it: those fields were applied when the ranking was computed, and a chunk id is content-derived, so the chunk behind a cached id is the same chunk of the same kind — what can have changed is exactly the document-level half |

Two more that are additions rather than corrections. `manicule.retrieval.assembly.window_problem`
is the shared predicate for §7.4's cross-check, so #7's enforcement point and this document's
requirement cannot drift apart; and `build_retriever` is the subsystem's composition root, which
reads the fusion constant and the reranker's model id **off the built pipeline** rather than off
configuration, so a recorded result names what actually ran.

## Appendix D: checklist against ticket #6

- **Dense + BM25 → RRF → cross-encoder → context assembly** — §4, §5, §6, §7, with the top-`k`
  trap closed on the dense side (§4.2–§4.4).
- **Each stage independently switchable, so #15 can attribute differences** — §2.4: pipelines are
  declared in configuration, the fusion stage takes its legs by name, and the three non-switchable
  steps are named with the reason each is not a quality feature.
- **Per-stage latency recorded** — §2.2 for who measures it, §11 for what is recorded and what
  makes two runs non-comparable.
- **Confidence scoring** — §8, defined as a statement about retrieval, with the components that
  are admissible and the ones that are not.
- **Three profiles** — §12, differing concretely rather than by name.
- **Caching, routing and token counting** (the audit addition on #6) — §10, §9, §7.2. L1 only; L2
  is settled in `embeddings.md` §8 and L3 belongs to a web-search connector.
- **The deferred features, each with its measurement** — §13, including the learned-sparse leg
  recorded against this ticket.
- **`RetrievalStage` locked without widening** — §2.3.
- **`Filter` closed** — §3, and `contracts.md` §6 updated.
