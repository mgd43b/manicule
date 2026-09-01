# Multi-step research

Design for the operation that answers one question from several searches: plan the question
into queries, run them, decide whether another round is worth it, and report with citations
that resolve.

Everything upstream is settled. The retrieval pipeline, context assembly and the confidence
score are settled in [`retrieval.md`](retrieval.md); generation, the citation guarantee, egress
policy and redaction are settled in [`generation.md`](generation.md); the envelope every surface
returns is settled in [`surfaces.md`](surfaces.md). **The boundary this document does not
reopen is the citation guarantee of [`generation.md`](generation.md) §3.** It is inherited whole
rather than restated, and §5 is about the one mechanism that makes inheriting it possible.

This document starts where the first planning call begins: what is decided before anything is
searched, what a cycle may carry forward, and what is written down afterwards.

---

## 1. The whole design in five sentences

**A sub-question is a Query and nothing else. Every cycle's retrieval is an ordinary pipeline
run over it. Nothing a model writes between cycles is ever evidence — it only decides what to
search next. The report is an ordinary answer over one assembled context, so the citation
guarantee is inherited rather than re-implemented. Every bound is declared before the run
rather than discovered during it.**

The third is the one that is easy to lose, and losing it is the failure this whole design is
arranged against. A loop that summarizes each cycle and hands the summaries to the report has
replaced the corpus with the model's notes about the corpus. The report then writes about
passages it never read, cites them through a paraphrase nobody verified, and every citation
still passes all three levels of the ladder — because the ladder checks that a slot resolves to
real bytes, not that the sentence in front of it is what those bytes say. So notes exist, they
are recorded, and they reach exactly one place: the prompt that picks the next query.

---

## 2. The shape

### 2.1 What runs, in order

```
      Question + the Query an `ask` would have run
                │
        ┌───────▼────────┐
        │ plan           │  one model call → sub-questions, or the question as asked
        ├────────────────┤
        │ search         │  one ordinary Retriever.retrieve per sub-question, bounded
        ├────────────────┤
        │ ledger         │  de-duplicated by chunk id, scores merged by maximum
        ├────────────────┤
        │ gaps           │  one model call → more sub-questions, or stop
        │   └── loop, up to research.max_cycles or research.timeout_s
        ├────────────────┤
        │ assemble       │  the ordinary fitter, against a widened profile
        ├────────────────┤
        │ Answerer       │  ← the whole of generation.md, unchanged
        └───────┬────────┘
                ▼
        ResearchReportPayload
```

Two things about that diagram are load-bearing.

**The loop stops at `assemble`.** It returns passages, and the application service runs the
answer path over them. `manicule.research` imports no binder, no verifier and no redactor, and
that is not tidiness — it is why there is no second implementation of the citation chain that
could omit one of them.

**The arrow into `Answerer` carries a `Context` and nothing else.** Not findings, not the plan,
not a summary. Whatever the loop learned between cycles has already been spent by the time the
report is written.

### 2.2 What is a protocol and what is deliberately not

The loop takes a `Generator` and a `Retrieving` — the same two seams the application service
already holds — and is otherwise a concrete class. It is **not** a plugin component, for the
reason [`generation.md`](generation.md) §2.2 gives about the binder: a boundary a plugin can
omit is not a boundary. A third-party research loop is exactly the thing that would gather
passages and hand them to a model without the egress filter, and the operation's name would
still be `research`.

What *is* replaceable is the part that should be: which model plans, how it is reached, and what
it costs. That is `llm.generator`, and it is the same choice an ordinary answer makes.

### 2.3 What a cycle may carry forward

Exactly two things, and both are counts rather than content: the list of sub-questions already
searched, and how many distinct passages the ledger holds. §3.2 is why.

---

## 3. Planning

### 3.1 A plan is a request, not an instruction

The model proposes sub-questions; the loop decides how many to run. `research.max_sub_questions`
truncates a plan that is too long, `MAX_SUB_QUESTION_LEN` discards an entry that is a paragraph
rather than a query, and a repeat of a search already run is dropped rather than spent.

The parser is deliberately tolerant — fenced JSON, a bare list, a leading "Here is the plan:"
are all the ordinary case, and a parser that raised on them would make the feature fail most of
the time. What it will not do is guess. A reply with no recognizable JSON yields nothing, the
run searches the question as asked, and **`model_planned` records that it did**. A run that
silently degraded to a single search is otherwise indistinguishable in its output from a
question that only ever had one facet, and a persistently broken planner would present as a
corpus of simple questions.

### 3.2 There is no summarizer, and that is a decision

[`surfaces.md`](surfaces.md) §4.2 already refused one:

> There is no summarizer between the search and the client. The recipe assembles evidence and
> hands it on; deciding what the evidence means happens outside it, where it can be seen. A
> compaction step would be a component choosing what the client is allowed to read, and it
> would have to be proposed explicitly, keep provenance, be replaceable, and be evaluated on
> its own.

This design does not propose one. The obvious way to build a multi-step researcher is to
summarize each cycle and write the report from the summaries, and it is rejected here for the
reason §1 gives: the report would cite passages it never read. What replaces it is that the
report reads passages directly, out of a context deliberately wider than one answer's.

That has a real cost and it is stated rather than hidden. **The evidence a report can read is
bounded by `research.report_tokens`, not by what the searches found.** A run that finds sixty
relevant passages reports from the twenty or so that fit. `passages_found` and `passages_cited`
are both on the payload precisely so the gap is visible; a single number would hide it.

### 3.3 The gap call sees counts, never passages

It is given the question, the searches already run, and how many distinct passages came back.
Not the passages. This step chooses the next query, and a step that read the evidence would be
the summarizer above, arriving through a side door — it would form a view of what the corpus
says, and that view would shape the report through the queries it proposed without ever being
visible in it.

It is also the cheap step. A prompt of counts and one-line queries costs a fraction of a prompt
carrying passages, which is what makes another cycle a decision worth making rather than an
expense to avoid.

---

## 4. Scope, and the escape it would otherwise open

**Everything but the text is copied from the original query, the filter above all.** This is the
same rule `Retriever._reworded` applies to a glossary rewrite, and here it matters more: a
glossary rewrite is deterministic and derived from an entry in the corpus, while a sub-question
is a string a model wrote after reading a question a user typed. A sub-question that could carry
its own `workspace_ids` would be a tenancy escape reachable by wording a question a particular
way, and the model would not have to be adversarial to produce one — only wrong.

The tenancy check then runs **once, over the whole accumulated ledger**, before the report's
model call. Checking each retrieval separately would leave the union unchecked, and the union is
what reaches the model.

**A planning call names no workspace at all.** It retrieves nothing, so its `Query` carries a
placeholder scope rather than the caller's: naming a real tenant on a request that reads none of
its documents would put that tenant into whatever the provider records.

---

## 5. Citations in a report

### 5.1 The report is an `ask`

There is no research-specific citation path. The service assembles the ledger into a `Context`,
builds an `AnswerRequest`, and runs `Answerer` — so the egress filter, the redaction projection,
the slot numbering, the marker binder and the three-level verification ladder are the same
objects doing the same work.

This is the whole reason a research report's citations are worth what an `ask`'s citations are
worth. Everything [`generation.md`](generation.md) §3 guarantees holds here by construction
rather than by a promise this document makes.

### 5.2 Slots are positional, so ledger order is citation order

A slot is an index into `Context.passages`. The ledger's order — best score first, ties broken
by when a passage was first seen — is therefore the numbering the report cites against, and
nothing may reorder it between assembly and the prompt. A reordering after rendering produces
citations that pass every level of verification and name the wrong passage, which is the exact
class of defect the numbering exists to prevent.

De-duplication by `chunk.id` is part of the same rule. Nothing downstream enforces uniqueness,
and the binder de-duplicates by *slot* rather than by chunk — so one passage found by two
sub-questions would be numbered twice, cited twice, and counted twice in every figure built on
`CitationAccounting`.

### 5.3 A widened profile, and the check it needs

The report is assembled against the configured profile with three fields raised:
`final_top_k` to `research.report_passages`, `context_tokens` to `research.report_tokens`, and
`candidates` with them — because `final_top_k <= candidates` is a validator, and a profile asked
for more passages than its pipeline was told to fetch cannot be satisfied. The overrides are
rebuilt through `profile_config` rather than copied onto a `ProfileConfig` with `model_copy`,
which skips exactly that validator. An operator's own overrides are kept.

`research.report_tokens` is wider than any profile's `context_tokens` **by design**, which means
[`retrieval.md`](retrieval.md) §7.4's startup cross-check says nothing about it. So it gets its
own, in `manicule.research.loop.plan_problem`, checked before the first search rather than
discovered when a server truncates a prompt from the front and the model appears to stop
following instructions.

---

## 6. Confidence, and the number this design does not invent

**There is no run-level confidence.** Each sub-question's retrieval carries its own, verbatim,
on its own step. A mean across several describes none of them: [`retrieval.md`](retrieval.md)
§8 is explicit that a `Confidence` is not comparable across pipeline identities, and two
retrievals for two different queries are two measurements of two different things.

`corroborated` — cited passages that more than one sub-question retrieved — is the one signal a
multi-step run has that a single retrieval does not. It is reported beside the citations and
**not folded into any score**, for the reason [`generation.md`](generation.md) §10.1 gives about
confidence and citation accounting: two numbers that answer two questions stay two numbers.

---

## 7. Cost, and the three bounds on it

| Bound | Setting | What it prevents |
|---|---|---|
| Rounds | `research.max_cycles` | a model that always wants one more |
| Searches per round | `research.max_sub_questions` | a plan that fans out without limit |
| Wall clock | `research.timeout_s` | a corpus slow enough that the other two do not bind |

`research.concurrency` bounds retrievals in flight. Deliberately small: the embedder serializes
every forward pass through one worker thread, so a wider fan-out queues there while each task
still holds a database connection, and the connection pool is what runs out — in another
request, which is where the failure would be diagnosed.

The deadline stops the *next* cycle rather than abandoning the run. A report that arrives late
is better than a run that returns nothing, and `stopped_early` says which happened.

**Bounds are what make this callable from an unattended surface at all.**
[`surfaces.md`](surfaces.md) §4 keeps `document_reindex_stale` off MCP and HTTP because "an
unattended caller able to start one has the machine's accelerator for an hour", and keeps
`document_reindex` on every surface "because one document is a bound". A research run is bounded
before it starts, which is the same argument.

---

## 8. What is deliberately not here

**No streaming.** The report is written in one pass at the end of a run that is mostly
retrieval, so a stream would be a long silence and then everything at once. The CLI streams the
report's own tokens when a person is watching, exactly as `ask` does; HTTP does not offer an SSE
route for it, and `POST /api/v1/chat/stream` remains the streaming surface for a single answer.

**No conversation.** `research` takes no `conversation_id` and persists no turn. A report is a
document rather than a turn in a dialogue, and threading one into a conversation would put
twenty passages of context into the history budget of every following question.

**No answer cache.** [`generation.md`](generation.md) §13's refusal applies unchanged, and more
strongly: a research run is expensive, which makes caching it tempting and makes a stale one
more misleading.

**No second model for planning.** [`generation.md`](generation.md) §4.2's "one generation model,
not two" holds. The planning calls are cheap — counts and one-line queries — so the argument for
a cheaper second model is weak, and the cost of a second context window to cross-check and a
second egress class to police is not.

---

## 9. Deferred, with the condition that would un-defer each

| Deferred | What would have to be true |
|---|---|
| A measured win on the multi-hop query set | [`retrieval.md`](retrieval.md) §13 defers query decomposition until a labeled multi-hop subset exists and names building it as the first deliverable. That half is done — [`evaluation/multi-hop-queries.json`](evaluation/multi-hop-queries.json). This feature is decomposition one level up, above the pipeline rather than inside it, so it widens no locked contract, but it owes the same measurement: nDCG@10 on that set alone, a decomposing configuration against a single-query one. **That has not been run**, and until it has, this feature is unmeasured rather than measured-and-good. |
| Streaming the report | A surface that shows the searches as they complete, so the silence is filled by progress rather than nothing |
| Re-ranking the ledger across cycles | Evidence that a cross-encoder pass over the union beats each search's own ranking, measured on the query set above |
| Carrying a run's evidence into a follow-up question | A history budget that can hold it, which today it cannot |

---

## Appendix A: decisions this document made

| Decision | Where |
|---|---|
| Notes never become evidence; the report reads passages | §1, §3.2 |
| The loop returns evidence and does not answer | §2.1 |
| Not a plugin component | §2.2 |
| A plan is truncated, not obeyed | §3.1 |
| A failed plan is recorded, not hidden | §3.1 |
| The gap call sees counts, never passages | §3.3 |
| The filter is copied verbatim onto every sub-question | §4 |
| Tenancy is checked once over the union | §4 |
| A planning call names no workspace | §4 |
| The report is an ordinary `ask` | §5.1 |
| Ledger order is citation order | §5.2 |
| The widened profile is rebuilt through validation | §5.3 |
| `report_tokens` gets its own startup check | §5.3 |
| No run-level confidence | §6 |
| Corroboration is reported, never folded into a score | §6 |
| No streaming, no conversation, no cache, no second model | §8 |

## Appendix B: checklist against the feature

- [x] Several retrievals for one question, each an ordinary pipeline run
- [x] Citations that resolve, through the existing verification ladder
- [x] Bounded before the run starts, on rounds, searches and wall clock
- [x] The same envelope, payload and error contract as every other operation
- [x] CLI, MCP and HTTP, with the MCP tool marked as reaching out and not read-only
- [x] A labeled multi-hop query set to measure against — `docs/evaluation/multi-hop-queries.json`
- [ ] A measured improvement on it — §9, and it has not been run
