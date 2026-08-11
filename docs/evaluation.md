# Evaluation

How manicule finds out whether its retrieval is any good, and — the part that carries the
weight — how it refuses to report a number that is not evidence. Ticket
[#15](https://github.com/mgd43b/manicule/issues/15).

The retrieval pipeline is settled and built ([`retrieval.md`](retrieval.md)) and the seam
every stage is written against is settled ([`contracts.md`](contracts.md) §3). This document
starts above them: given two configurations and a corpus, what is measured, what is refused,
and what a recorded result is allowed to claim.

---

## 1. The whole design in five sentences

**A system must demonstrate it retrieves before its preferences are recorded. Judgement is
pairwise and blinded, never an absolute relevance label. A result names the corpus it was
measured against and the configuration that produced it. Categories are reported as rows, and
the average is one row among them. A report says what it is not.**

The first of those is the reason the rest exist. An evaluation harness is a measuring
instrument, and the characteristic failure of a measuring instrument is not reading wrong — it
is reading *plausibly* while measuring nothing. A harness built on a component with no
semantic content produces well-formed reports, sensible-looking win rates, and confident
conclusions about features that were never distinguishable from noise. Nothing in the output
says so, and the damage compounds: every feature added on top is now justified by a number,
and all the numbers came from the same place.

So the load-bearing part of this subsystem is not the comparison. It is the refusal.

---

## 2. The query set

A versioned JSON document rather than a list of strings. The format is
`manicule.evaluation.queries`, schema version 1:

```json
{
  "schema_version": 1,
  "name": "knowledge base, august",
  "provenance": "exported",
  "description": "…",
  "exported_at": "2026-08-10T09:00:00+00:00",
  "queries": [
    {
      "id": "ql-4417",
      "text": "how does incremental sync decide a document has changed",
      "intent": "how_does_x_work",
      "thumbs": "up",
      "note": "…"
    }
  ]
}
```

Four properties, each corresponding to a way a query set silently stops being what it claims.

**`provenance` is required**, and it is one of `exported`, `authored` or `example`. A set of
questions someone invented to exercise the code and a set taken from what people actually
asked are not the same evidence, and by the time a figure is being quoted nobody remembers
which produced it. The declaration travels into every recorded preference and into the report,
and a report built from an `example` set leads with a line saying the numbers are illustrative.
`QuerySet.is_evidence` is a field-derived property, not a convention about presentation.

**`intent` is a field, not a comment.** Four categories plus `uncategorised`, chosen because
they fail *differently* rather than because they are a taxonomy: a lexical leg carries exact
identifiers and a dense leg carries paraphrase, and a reranker that earns its cost on
explanatory questions often costs precision on identifier lookups. One averaged number over
all four hides every one of those effects. `uncategorised` is reported as its own row rather
than folded into another — a bucket that quietly absorbs the awkward ones makes every other
row look cleaner than it is.

**`schema_version` is checked against a set of readable versions**, not a maximum. A newer
format loaded by an older build would otherwise arrive with fields missing.

**Unknown fields are refused, not dropped.** An export gaining a column has to fail loudly.
The thumbs signal is exactly the kind of field that would vanish this way: present in the
export, absent from the model, and nothing anywhere reporting that the set being measured is
thinner than the file it came from.

### 2.1 Exporting is the operator's step

Reading query logs out of a running system needs credentials and a schema this project does
not own, so it is not part of this package and does not gate it. What ships is the target:
build `EvalQuery` values from whatever the export produced, declare the provenance, and call
`dump_query_set`.

`docs/evaluation/example-queries.json` ships as a worked example of the format. It declares
itself `example`, which means every report built from it says so on its first line. It is not
a measurement of anything and there is no configuration under which it becomes one.

---

## 3. The corpus is pinned, on both sides and across time

The property being protected: **same content on both sides, so a difference is retrieval and
not what was indexed.** It stops being true in two different ways, and they need different
instruments.

**Across the two sides of one comparison**, the `label` is an operator's assertion that both
systems are pointed at the same documents, checked before any preference is recorded. It is an
assertion rather than a proof because one side may be a system manicule cannot introspect at
all — see §4.

**Across runs of the same side, weeks apart**, the label is the *weakest* instrument, because
it is precisely the thing that does not change when the corpus does: documents get added, the
label still says `knowledge-base`, and last month's win rate is compared against this month's
over different content. So a system that can compute one records a `digest` — sha256 over
sorted `(document id, content hash)` pairs — and a report over records whose digests disagree
refuses rather than averaging.

Absence of a digest is recorded as absence, never as agreement. A report where either side
could not produce one prints that the corpus identity was *asserted* rather than verified.

Chunk and document counts are recorded and never refused on. Two systems chunk differently, and
a refusal there would block exactly the cross-system comparison this is built for, for a reason
that is not about the corpus.

---

## 4. A system under comparison is an adapter and a label

```
SystemUnderComparison
    config_label: str
    corpus_version: CorpusVersion
    async def search(text: str, *, limit: int) -> SystemResult
```

Three members, and the narrowness is the point: a wider protocol would be one only manicule
can satisfy, and the comparison that matters most is against something manicule did not build.
`CallableSystem` wraps any async callable, so an external system is an adapter and a
configuration label supplied at runtime — a service over HTTP, a command-line tool, another
machine.

`RetrieverSystem` is the manicule side, and it carries two refusals.

**It will not run against a retriever whose cache can hit.** A cached ranking is one sample
counted twice at the cache's latency. A run that quietly served half its queries from memory
would report a latency improvement that is an artefact and a quality figure computed from half
as many observations as it claims. The retriever already knows whether a hit is *possible* —
configuration alone is not enough, since a store with no generation counter disables the cache
regardless ([`retrieval.md`](retrieval.md) §10) — so that is what is checked.

**The configuration on a record is the one the run reported**, read straight off
`RetrievalTrace.pipeline`: the stage list, the fusion constant, the reranker id and the
embedding fingerprint that actually ran. A configuration supplied alongside the adapter would
be a second copy of the truth, and the copy is the one that goes stale. The harness refuses a
run whose sides changed configuration midway, because half the records would then name a
pipeline that was not running when they were made and nothing in the file would say which half.

The **route is deliberately not part of that configuration.** It is a property of the query:
folding it in makes a query set containing "hello" look like a pipeline that changed between
query 3 and query 4, and the run is then refused with a message that misdiagnoses it entirely.
Where the route matters it matters as a reason a pairing is not a measurement, so a query the
router answered directly ([`retrieval.md`](retrieval.md) §9) is marked incomparable — the
corpus was never consulted, and two empty lists a judge scores as "neither" would otherwise
read as both systems failing a question neither was asked.

### 4.1 Per-stage attribution comes nearly free

Every result carries a `StageObservation` per stage — name, wall time, candidates in and out,
and the stage's declared configuration — copied from the trace. The report then computes which
stages the two sides did not share, as a set operation.

That is free because of a decision made upstream: a pipeline is a declared list of uniform
stages ([`retrieval.md`](retrieval.md) §2.4), so two configurations differing in one place
produce records differing in one stage. `RetrievalStage` is **not** widened to make this work,
and nothing here asks it to be.

---

## 5. The discrimination probe

**A system that retrieves at chance cannot be reported as anything but useless.**

Before any preference is recorded, each side is put through a set of questions whose answers
are known by construction — no relevance judgements and no labelling session.
`probe_from_titles` derives them from the corpus: a document's own title, used as the query,
with that document as the known answer. That is the least a retrieval system can be asked to
do, which is exactly what a liveness check wants; a probe only a good system passes would be a
quality benchmark, and this is not one.

**The verdict is a hypothesis test, not a threshold.** "It got 6 of 20" says nothing until it
is compared against what guessing would do. With `k` results drawn from a corpus of `N`
documents, chance is `k / N` per item, hits are binomial, and the probe reports the probability
that chance alone would have done at least this well. At or above `alpha` — 0.01 by default —
the system is at chance and no preferences are collected for it.

**And the probe refuses when it could not have told the difference.** Three refusals, each
guarding a verdict that was never in doubt:

| Refusal | Why it is not a caveat |
|---|---|
| The corpus is small enough that `k` results cover most of it | Chance approaches certainty and every system passes |
| The system cannot say how many documents it holds | Chance is unknown, so any p-value is invented |
| Too few items for a *perfect* run to reach `alpha` | A check whose failing verdict is unconditional is not a check. It would report a flawless system as being at chance — the mirror image of the failure this exists to prevent, and just as useless |

The third refusal names how many items the probe would need, computed from the chance rate.

`k / N` treats the `k` results as `k` distinct documents. A system returning several chunks of
one document examines fewer than that, so the real chance of a hit is lower and this null is an
over-estimate — an error that runs in the safe direction and only that direction, since an
over-stated null makes the test harder to pass and can never admit a system that is guessing.

**A `ProbeOutcome` re-does its own arithmetic** whenever one is constructed: `hit_rate`,
`chance_rate` and `p_value` are all recomputed from `hits`, `trials`, `k` and `pool_size`, and
a disagreement beyond floating-point slack refuses the record. Recording the inputs beside the
verdict is not enough on its own, because outcomes are read back off disk and a file is exactly
where a hand-edited or foreign record enters. Without it, a record claiming a decisive
`p_value` beside one hit in twenty-four passes every other check in this package.

### 5.1 The rule is enforced in three places, and none is redundant

| Where | What it stops |
|---|---|
| `PreferenceHarness.certify`, before the first query | A session that records first and filters later, leaving judgements about noise on disk |
| `PreferenceRecord`'s validator | A record built by any other path — the model itself refuses to construct one |
| `build_report`, over records read back | A file written by some other tool, or by an older build. Files outlive the process that wrote them, so the rule has to hold where the number is produced |

### 5.2 What proves the probe works

Two tests, and the second is not a formality. `tests/evaluation/test_probe.py` builds the
shipped `Retriever` over the shipped `DenseStage` against a migrated database, twice, changing
exactly one thing: the embedder.

- An embedder whose vector is a hash of the whole string — correctly shaped, correctly
  normalised, deterministic, and unrelated to meaning — is **refused**, at 1 hit in 24 against
  a 5% chance rate (p = 0.71).
- An embedder with real semantic content is **admitted**, at 24 hits in 24 (p = 6e-32).

Without the second, a probe that refused everything would pass the first while being exactly as
uninformative as one that refused nothing.

The pipeline under test is dense-only, deliberately. A hybrid pipeline retrieves through BM25
as well, so it would find a document by its title however meaningless its vectors were — and
would be right to pass, because the *system* retrieves. What has to be catchable is a pipeline
whose only retrieval mechanism has no semantic content.

---

## 6. Judgement is pairwise, and blinded

Two ranked lists side by side and one keypress: `a`, `b`, tie, neither, skip, quit. Seconds per
query is the design constraint.

**Pairwise rather than absolute relevance labels.** Absolute judgements need a scale, a
definition of relevant and a minute of attention per passage. A preference answers the question
actually being asked — *is this configuration better than the one I have* — and it answers it
at a cost that gets a whole query set finished. Absolute labels and nDCG come later, and **only
if pairwise preference stops discriminating between candidate configurations.** Building them
first is how an evaluation set never gets finished.

**Tie and `neither` are different outcomes.** A tie says both lists answered about equally
well; `neither` says both failed. Collapsing them would let a query set on which both systems
are useless read as a dead heat, which is the reading that stops anybody investigating.

**Sides are blinded.** Which system appears as A is a keyed hash of the query id: deterministic,
reproducible from the recorded seed, stable when queries are added or reordered, and
uncorrelated with anything about the systems. Position bias is large and free to remove. What
that buys is demonstrated rather than asserted — a judge that always picks the first list
produces a split whose interval contains 0.5, where without blinding it would report 100% for
whichever side was passed first: a clean, significant, entirely false result.

**Records are append-only JSON Lines.** They are evidence. A file that can be edited in place
has whatever history the last writer decided it had.

---

## 7. The report

Per intent category, as rows, with the overall figure as one row among them rather than the
headline. Three things every row carries:

- **A Wilson interval.** Seven wins from ten is 0.70, and its 95% interval runs from 0.40 to
  0.89 — visibly containing the point where neither system is better. Wilson rather than the
  normal approximation because at 10/10 the latter gives an interval of *zero width*: an exact
  claim from ten samples.
- **A two-sided sign test.** Ties and `neither` are dropped rather than split between the
  sides; splitting them would manufacture evidence from judgements that deliberately expressed
  none.
- **The counts**, including how many pairings were excluded and why.

A pairing where either run was degraded, cached or stopped by its own budget
([`retrieval.md`](retrieval.md) §11.2) is **recorded and not counted**. It is evidence about
the run; it is not a comparison of two pipelines. The exclusion count is printed, because a run
where most pairings were excluded is a finding about the run and a rate computed from the
remainder without saying so is not.

`build_report` refuses more than it reports. Records spanning more than one pair of systems,
more than one corpus, or more than one query-set provenance are not summarised together.

---

## 8. What this gates

Every retrieval feature deferred in [#6](https://github.com/mgd43b/manicule/issues/6) —
HyDE, multi-query expansion, query decomposition, intent classification, cross-lingual
expansion, parent-document retrieval, propositions, prompt compression, the hallucination
guard, and the learned-sparse leg. Each ships with a measured improvement or does not ship.
[`retrieval.md`](retrieval.md) §13 states what would have to be measured for each; this
document is what makes the rule enforceable rather than a discipline to remember.

Two consequences follow immediately.

**`RetrievalStage` is now locked.** [`contracts.md`](contracts.md) §3 carried the warning that
it is locked once the evaluation harness exists. It exists. Widening it invalidates every
recorded result, and a recorded result is a file on disk with a schema version in it.

**Absolute relevance labels and nDCG are deliberately absent.** Not an oversight and not a
gap to be filled at the first opportunity: they are a different and far more expensive
instrument, and the trigger for building them is pairwise preference failing to separate two
candidate configurations. Until that happens, building them first is the expensive half of an
evaluation set spent on the question that was already answerable.

---

## Appendix A: decisions this document made

| Decision | Where |
|---|---|
| Provenance is a required field on a query set, and drives whether a report calls itself evidence | §2 |
| The corpus label is a refusal, the digest is a stronger refusal, counts are neither | §3 |
| A system under comparison is three members, so an external system needs only an adapter | §4 |
| A retriever whose cache can hit is refused as a system under comparison | §4 |
| Configuration is observed from the trace, never declared alongside the adapter | §4 |
| Chance is a hypothesis test against `k / N`, at `alpha = 0.01` | §5 |
| A probe that could not have detected a perfect system refuses to run | §5 |
| The chance-level rule is enforced at run time, at record construction, and at report time | §5.1 |
| The route is a property of the query, not the configuration, and a routed-away query is not a measurement | §4 |
| A probe outcome recomputes its own derived figures and refuses a record whose arithmetic does not hold | §5 |
| Sides are blinded by a keyed hash of the query id | §6 |
| Wilson intervals rather than the normal approximation | §7 |
| Ties and `neither` are counted apart and both dropped from the rate | §7 |
| Statistics are exact and in pure Python, so the harness carries no numerical dependency | §7 |
