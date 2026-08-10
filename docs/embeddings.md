# Embeddings

Design for the embedding backend, the pooling path, and the model. Ticket
[#3](https://github.com/mgd43b/manicule/issues/3).

`PLAN.md` §7 settles the runtime split. [`docs/contracts.md`](contracts.md) §3 fixes the
two-tier `Embedder` protocol. [`docs/storage.md`](storage.md) §6.3 defines the fingerprint
refusal. This document makes the decision none of them make — **which model** — and specifies
the pooling path that the two-tier protocol exists to protect.

**One thing here is expensive to change after a corpus is indexed:** the model, because it
fixes `D` and every stored vector lives in its space. It is guarded by the same fingerprint
refusal `storage.md` §6.3 already specifies.

Everything empirical below was measured on this hardware against real checkpoints. Where a
figure is taken from a model card rather than measured, it says so.

---

## 1. The model — `Alibaba-NLP/gte-modernbert-base`, D = 768

**Decided.** Apache-2.0, 149M parameters, 768 dimensions, 8192 max sequence length, CLS
pooling, no instruction prefixes.

### 1.1 What eliminated everything else

Four gates, applied in order. Most candidates die on one of the first two, which is why the
quality argument is short.

**Gate 1 — `max_sequence_length` ≥ the 512-token chunk budget**
([`parsing.md`](parsing.md) §1). A model that truncates below the budget indexes a chunk's
opening and silently discards its tail.

| Eliminated | Ships at |
|---|---:|
| `all-MiniLM-L12-v2` | 128 |
| `all-mpnet-base-v2` | 384 |
| `all-MiniLM-L6-v2` | 256 |

All three have backbones that permit 512; all three ship configured lower, and the shipped
number is the one that truncates. This is the trap `parsing.md` §1.1 names — the positional
limit and the configured limit are different numbers and the smaller one wins silently.

**Gate 2 — a permissive licence.** This project has rejected PyMuPDF (AGPL) and
`extract-msg` (GPL-3.0) at selection time rather than working around them later
([`parsing.md`](parsing.md) §12), and a model is not exempt.

| Model | Licence | Verdict |
|---|---|---|
| `jina-embeddings-v3` | **CC-BY-NC-4.0** | **Eliminated.** Non-commercial only |
| `google/embeddinggemma-300m` | **Gemma Terms, gated** | **Eliminated** — see §1.4 |
| everything else below | MIT or Apache-2.0 | passes |

**Gate 3 — runnable in-process on the target machine.** Under onnxruntime (§3.3) this means
*an ONNX export exists*, and **every surviving candidate has one**. So this gate eliminates
nothing, and saying so is the point.

**The gate it is deliberately *not*.** `mlx-embeddings` implements seven text architectures,
which would have disqualified `nomic-embed-text-v1.5` (`nomic_bert`),
`gte-Qwen2-1.5B-instruct` (`qwen2`) and `Alibaba-NLP/gte-base-en-v1.5` / `gte-large-en-v1.5`
(`new`). That is a constraint on **a backend**, and letting it silently become a constraint
on **the model** would mean picking a worse embedder because a library happened to support
it. An earlier draft of this section did exactly that. It is corrected here rather than
quietly fixed, because the elimination looked principled and was not.

`onnxruntime` runs on Apple Silicon; it is not a non-Apple fallback. A model outside MLX's
seven is still fully runnable locally on the target hardware. What is given up is
Metal-native execution — throughput — **not** the platform, and not correctness. That is the
trade named in §1.3 and the principle in `PLAN.md` §7.

One naming trap, since it eliminated the wrong repository once already: `thenlper/gte-base`
and `gte-large` are plain `BertModel`, while `Alibaba-NLP/gte-base-en-v1.5` is architecture
`new`. Two families share a name and only one has the architecture issue — which, per the
above, is no longer disqualifying anyway.

**Gate 4 — retrieval quality per parameter.** What survives, with MTEB v1 English retrieval
(mean nDCG@10 over the 15-dataset set, from each card's `model-index`):

| Model | D | max_seq | Params | Licence | Retrieval |
|---|---:|---:|---:|---|---:|
| `gte-Qwen2-1.5B-instruct` | 1536 | 32768 | **1.78B** | Apache-2.0 | **58.29** |
| **`gte-modernbert-base`** | **768** | **8192** | **149M** | **Apache-2.0** | **~54.8** ¹ |
| `mxbai-embed-large-v1` | 1024 | 512 | 335M | Apache-2.0 | 54.39 |
| `bge-large-en-v1.5` | 1024 | 512 | 335M | MIT | 54.29 |
| `bge-base-en-v1.5` | 768 | 512 | 110M | MIT | 53.25 |
| `gte-large` (thenlper) | 1024 | 512 | 335M | MIT | 52.22 |
| `multilingual-e5-large` | 1024 | 512 | 560M | MIT | 51.43 |
| `gte-base` (thenlper) | 768 | 512 | 110M | MIT | 51.14 |
| `e5-large-v2` | 1024 | 512 | 335M | MIT | 50.56 |
| `e5-base-v2` | 768 | 512 | 110M | MIT | 50.29 |
| `multilingual-e5-base` | 768 | 512 | 278M | MIT | 48.88 |
| `nomic-embed-text-v1.5` | 768 | 8192 | 137M | Apache-2.0 | 53.01 |

¹ The card reports 55.33, but its BEIR table substitutes `CQADupstackAndroidRetrieval` — one
easy subtask — for MTEB's 12-subforum `CQADupstack` aggregate. Correcting for that is worth
about −0.5, so the MTEB-comparable figure is ~54.8. **Adjusted down here rather than quoted
at face value**, because comparing a corrected number against uncorrected ones is how
benchmark tables mislead.

### 1.2 Why this one

It is the best retrieval score available **at a size that fits a self-hosted tool** — a third
to a half the parameters of everything within a point of it, and a twelfth of the one model
that scores higher (§1.3). Three of its properties matter more here than the score:

- **8192 max sequence length against a 512-token budget** removes an entire class of
  arithmetic. No special-token accounting, no prefix headroom, no truncation risk under any
  configuration. Compare `bge-base-en-v1.5`, which is *exactly* 512: BERT's `[CLS]` and
  `[SEP]` consume two, so its usable content budget is 510, and a 512-token chunk overflows
  by two tokens. That is survivable but it means the chunker and the embedder have to agree
  on an arithmetic nobody will remember (§4.3). Here there is no arithmetic.
- **No instruction prefixes at all** (`prompts: {}`). Every prefix-bearing model introduces
  an indexing/query asymmetry that must be applied consistently forever, and prefixes are
  string-concatenated *before* tokenization, so they consume the budget from the tail of the
  chunk. E5 is the sharp case — `passage: ` is mandatory on the **document** side, so a
  512-token chunk becomes 515 and loses its end. Nothing to get wrong is better than a rule
  to remember.
- **Its traps are the ones this design already defends against.** ModernBERT is the single
  architecture where the backend rebinds `last_hidden_state` to the pooled vector (§3.1), and
  the one where pooling choice matters most (§4.1). Choosing it makes the pooling discipline
  load-bearing rather than theoretical — and a defence that is exercised is a defence that
  works. The alternative, choosing BERT because its traps are milder, means shipping
  safeguards nothing tests.

### 1.3 The one model that scores higher, and what rejecting it costs

`gte-Qwen2-1.5B-instruct` leads the table at **58.29**, about **+3.5** over the chosen model —
a far larger gap than anything else discussed here. It is Apache-2.0 and it passes gates 1
and 2. It is rejected on **cost, stated explicitly rather than buried in an architecture
gate**:

| | `gte-modernbert-base` | `gte-Qwen2-1.5B-instruct` |
|---|---:|---:|
| Parameters | 149M | **1.78B** (12×) |
| Dimensions | 768 | **1536** (2× storage) |
| 1M chunks, fp32 | ~2.9 GiB | **~5.7 GiB** |
| Retrieval | ~54.8 | 58.29 |

Twelve times the parameters is twelve times the ingest compute for every chunk, in-process,
on a laptop — and doubling `D` doubles vector storage and distance cost permanently, against
§1.5's finding that dimension buys nothing on its own. A 1.78B-parameter model is also a
different operational proposition: memory pressure during ingest becomes a real constraint
rather than a rounding error.

**This is a judgement, not a derivation, and it is the weakest link in this document.** +3.5
nDCG is a real difference and someone optimising purely for retrieval quality would take it.
It is declined because manicule is a self-hosted tool that must stay installable and
ingestable on one machine, and because the +3.5 is an MTEB figure that has never been
measured on the corpus this will actually index.

**It is the first thing to re-examine under [#15](https://github.com/mgd43b/manicule/issues/15).**
If measured retrieval on a real corpus shows the gap holding, the trade is worth revisiting
with real ingest timings beside it. What must not happen is this model disappearing from
consideration because a backend did not implement `qwen2` — which is what an earlier draft of
§1.1 did.

**Recorded alternative: `bge-base-en-v1.5`.** MIT rather than Apache-2.0, plain BERT, 1.5
points lower. It is the right answer if ModernBERT's maturity becomes a problem or if a
hand-written MLX encoder is ever needed (§3.4) — a BERT encoder is a few hundred well-
understood lines; a ModernBERT encoder is rotary embeddings, alternating local/global
attention, and GeGLU. Switching is a re-embed, priced by the fingerprint, not a redesign.

### 1.4 Why not EmbeddingGemma, which is otherwise the strongest

It scores highest of anything considered, it is genuinely Matryoshka-trained, and it is
eliminated on licence — so the reasoning is recorded rather than left implicit.

**The Gemma Terms are not an open-source licence.** Redistribution is permitted, including
commercially, but the Prohibited Use Policy is incorporated by reference, binds downstream
recipients, and **Google may update it after the fact**. Every recipient of anything built on
it inherits obligations that MIT and Apache-2.0 do not impose.

Two practical consequences on top of the legal one. The repository is **gated** — an
unauthenticated download returns 401, which breaks cold-start container builds and CI unless
a token is provisioned or an ungated mirror is used. And its pipeline is
`Transformer → mean pool → Dense(768→3072) → Dense(3072→768) → Normalize`: **the two Dense
layers are part of the model**, so anything reimplementing from the backbone alone produces
confident garbage.

Also worth recording, since it looks like a footgun and is not: the model card page on
`ai.google.dev` displays "Apache 2.0" in its footer. That is the licence of the page's code
samples, not the weights. An automated reader misparsed exactly this during research.

For a self-hosted tool this is defensible and many projects would take it. It is refused here
for consistency: a use-restricted licence with a mutable policy is a heavier obligation than
either library this project has already declined, and taking it for a model while refusing it
for a PDF library would make the rule arbitrary.

### 1.5 Why 768 and not 1024

The owner floated 1024 early. **The evidence says dimension is not a quality knob** — it is
inherited from backbone width, and paying for it directly buys nothing.

**Within a single model, dimension is nearly free.** Matryoshka models isolate the variable
perfectly: same weights, same training, only `D` changes.

| Model | 768 → 512 | Cost |
|---|---|---:|
| EmbeddingGemma (MTEB v2 English) | 69.67 → 69.18 | **−0.49** |
| nomic-embed-text-v1.5 (MTEB overall) | 62.28 → 61.96 | **−0.32** |

Shedding a third of the dimensions costs under half a point. If dimension itself carried
retrieval signal, it could not be that cheap.

**Across base/large pairs, the gain comes with 3.1× the parameters** — and it is small and
inconsistent, which is the tell:

| Family | 768-dim | 1024-dim | Δ |
|---|---:|---:|---:|
| bge-en-v1.5 | 53.25 | 54.29 | +1.04 |
| gte (thenlper) | 51.14 | 52.22 | +1.08 |
| **e5-v2** | 50.29 | 50.56 | **+0.27** |

E5 settles it: tripling parameters and adding 256 dimensions bought **0.27 nDCG**. If
dimension were the driver that number would resemble the others.

**And 768 models routinely beat 1024 models outright.** The chosen model, at 768 dimensions
and 149M parameters, outscores `mxbai` (1024, 335M), `bge-large` (1024, 335M) and
`e5-large-v2` (1024, 335M). OpenAI published the same result more starkly: `text-embedding-3-large`
truncated to **256** outperforms `ada-002` at its full **1536**.

**The cost of choosing 1024 anyway** is +33% on vector storage, on distance computation, and
on every index rebuild, permanently. At 768 with fp32, a vector is 3 KiB; a million chunks is
~2.9 GiB before overhead. At 1024 the same corpus is ~3.8 GiB for, on this evidence, no
retrieval gain.

**One caveat that must travel with this reasoning:** the "dimension is nearly free" evidence
comes entirely from models *genuinely trained with Matryoshka loss*. Truncating a model that
was not falls off a cliff — mixedbread's own published curve for `mxbai-embed-large-v1` shows
retrieval retention of 95% at 512 but **67% at 128**, against ~95% at 128 for EmbeddingGemma.
`gte-modernbert-base` is **not** MRL-trained, so **768 is fixed, not a starting point**. There
is no shrink-later option here, and pretending otherwise would be the sort of latent
assumption that surfaces two years in.

### 1.6 What would have to be true to change it

Changing the model means re-embedding the corpus, so both of these, not either:

1. **A measured improvement on the [#15](https://github.com/mgd43b/manicule/issues/15)
   baseline**, on manicule's own corpus. Every figure in §1.1 is MTEB, which is open-domain
   and English-heavy; none of it was measured on an enterprise wiki. The ranking above is the
   best available prior, not evidence about the corpus this will actually index.
2. **A licence and architecture that still pass gates 2 and 3.**

The one foreseeable trigger is **multilingual content**. This model is English-only. A
Confluence instance with substantial non-English pages should be re-evaluated against
`multilingual-e5-base` (768, MIT) or `bge-m3`, and that is a corpus fact nobody can settle
in advance.

---

## 2. Storage and latency consequences of `D`

Recorded because they are the whole cost of the dimension decision, and because
`storage.md` §6.5 sizes the vector table from them.

| | 768 (chosen) | 1024 (rejected) |
|---|---:|---:|
| Bytes per vector, fp32 | 3,072 | 4,096 |
| 100k chunks | ~293 MiB | ~391 MiB |
| 1M chunks | ~2.9 GiB | ~3.8 GiB |
| Distance cost | baseline | +33% |

Vectors are stored **fp32**. Storing fp16 halves this and is tempting, but the fingerprint
records the space and a later change is a re-embed, so it is not a knob to flip casually —
and quantisation belongs to the vector index, where LanceDB can apply it without changing
what manicule computed.

---

## 3. The runtime

### 3.1 The finding that reshapes this section

`PLAN.md` §7 names `mlx-embeddings` as the embedding runtime. **It is GPL-3.0.** Verified
from the installed distribution metadata, not from the repository page:

```
Name: mlx-embeddings
Version: 0.1.0
License: GNU General Public License v3
Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)
```

This is the same class of problem as PyMuPDF and `extract-msg`, and it is **worse in
kind**, because those are parsers behind a fallback chain while this would be a required,
in-process dependency of the core product. manicule is MIT. A required runtime import of a
GPL-3.0 library is not something an MIT licence file survives contact with.

It is also, on the evidence of this ticket, a **0.1.0 package that gives the same field
different meanings on different architectures** (§3.2) — the maturity profile that produces
exactly this class of surprise.

### 3.2 What the backend actually does per architecture, measured

Loading real checkpoints and inspecting the returned object:

| Architecture | `last_hidden_state` is | Verified on |
|---|---|---|
| `bert` | **3-D token states** `(3, 398, 384)` | `bge-small-en-v1.5` |
| `xlm_roberta` | 3-D token states | `bge-m3` |
| `gemma3_text` | 3-D token states | `embeddinggemma-300m` |
| `qwen3` | 3-D token states | `Qwen3-Embedding-0.6B` |
| **`modernbert`** | **2-D pooled vector** `(3, 768)` | `nomicai-modernbert-embed-base` |
| `modernbert` with `ForMaskedLM` | 3-D token states | `answerdotai/ModernBERT-base` |

The library rebinds the local variable before returning it, so the *same module* returns
different ranks depending on the checkpoint's `architectures` field.

**This inconsistency is a worse hazard than a uniform lie.** A uniform one is caught the
first time anyone checks. This one is correct on BERT, wrong on ModernBERT, and wrong only
for some ModernBERT checkpoints — so it passes review on whichever model the reviewer
happened to try. It is the direct reason for the 3-D assertion in §5.

### 3.3 Decision: onnxruntime is the primary backend

**This inverts the relationship in `PLAN.md` §7, and the justification for MLX does not
survive inspection.**

`PLAN.md` §7's stated argument for MLX is that it *runs in-process*, so there is no server to
operate and `uv tool install manicule` stays one command. That argument is correct and it
**does not distinguish MLX from onnxruntime** — onnxruntime is equally in-process. The same
section explicitly withdraws the only MLX-specific claim: *"An earlier draft justified MLX
with '~50% faster than llama.cpp on embeddings.' That figure has no traceable primary
measurement and should not be repeated."*

So the settled decision rests on a property both options have, plus a speed claim the plan
itself disowns — and the only maintained implementation of the MLX path is GPL-3.0.

| | onnxruntime | `mlx-embeddings` |
|---|---|---|
| Licence | **MIT** | **GPL-3.0** |
| In-process | yes | yes |
| Runs off Apple Silicon | yes | no |
| Architecture support | any exported graph | 7 implemented, allowlisted |
| Maturity | mature, widely deployed | 0.1.0 |
| Chosen model available | in-repo ONNX, 8 quantisations | requires conversion |

**So: onnxruntime is the embedding runtime.** It satisfies the actual requirement, on every
platform, under a compatible licence, with no architecture allowlist and no rebinding trap —
because we run the graph and read its output tensor directly.

### 3.4 MLX is not abandoned — it is unblocked from a different direction

Metal-native execution is still worth having on the machine this is built for. What is
refused is *the GPL dependency*, not MLX.

Apple's `mlx` itself is **MIT**. A future MLX path implements the encoder against `mlx`
directly, without `mlx-embeddings` in the dependency set. That is less work than it sounds,
because this design already bypasses nearly everything the library provides: we do our own
pooling (§4), our own normalisation, our own tokenization to obtain attention masks (§4.2),
and we cannot use its `generate()` helper anyway (§7). What remains is the encoder forward
pass and weight loading.

It is **not in v1**, and it is a filed ticket rather than an intention (§9). Two conditions
gate it:

1. **A measured speedup** over onnxruntime with the CoreML execution provider, on this
   hardware, on real chunk batches. `PLAN.md` §7 already refuses to repeat unmeasured speed
   claims; this is that rule applied to its own conclusion.
2. **Parity with onnxruntime** (§6), which is what makes a second implementation admissible
   at all.

Recorded honestly: implementing a ModernBERT encoder is materially harder than a BERT one.
If the MLX path is ever built and that cost dominates, `bge-base-en-v1.5` (§1.2) is the
recorded fallback — but that is a re-embed, and it should be driven by measurement rather
than by implementation convenience.

---

## 4. Pooling — the thing this ticket exists to get right

### 4.1 Why pooling is ours, with the real numbers

`contracts.md` §3 puts pooling in manicule's hands. The measurement behind that decision was
quoted as a single figure, and the single figure understates it.

**CLS versus mean pooling, measured on `gte-modernbert-base`**, both L2-normalised, computed
in numpy from the true 3-D token states:

| tokens | 7 | 21 | 42 | 79 | 156 | 301 | **452** |
|---|---:|---:|---:|---:|---:|---:|---:|
| cosine | 0.900 | 0.873 | 0.843 | 0.819 | 0.765 | 0.746 | **0.693** |

**It is a curve, not a constant, and it decays with length.** The often-quoted **0.856** sits
in the short-chunk band around 42–79 tokens.

**This is the headline, and it connects two independent pieces of work.** The chunk budget is
512 tokens ([`parsing.md`](parsing.md) §1). That places the system at the **far right of this
curve** — the operating point where the two poolings disagree *most*. The honest figure at the
size manicule actually embeds is nearer **0.70** than 0.856. The old number understated the
risk at exactly the place it is largest.

**And the same comparison on BERT stays mild:**

| `bge-small-en-v1.5` | 6 tok | 125 tok | 480 tok |
|---|---:|---:|---:|
| cosine | 0.960 | 0.919 | 0.873 |

So pooling choice is **nearly harmless on BERT (0.87–0.96) and severe on ModernBERT (0.69 at
452)**. A backend verified on BERT and shipped on ModernBERT would conceal the entire problem
— which is precisely the situation this project would have been in.

### 4.2 The pooling path

```
tokenize(texts) -> input_ids, attention_mask     # ours; masks are needed and not surfaced
run graph       -> token_states (batch, seq, D)  # 3-D, asserted (§5)
pool            -> vector (batch, D)             # per the model's declared pooling
L2 normalise    -> unit vector
```

Four rules:

- **Pooling is read from the model's declared configuration**, never assumed. For the chosen
  model it is **CLS**. It is recorded in the fingerprint (§5), so a model whose pooling
  differs cannot share an index with one whose pooling does not.
- **Mean pooling is attention-mask-weighted.** An unweighted mean over a padded batch
  averages in the padding and produces a different vector for the same text depending on what
  else was in its batch — a batch-order-dependent index.
- **We tokenize.** The backend does not surface attention masks from its embedding call, and
  mean pooling is impossible without them. Tokenizing ourselves also fixes truncation
  explicitly rather than inheriting a library default (§7).
- **L2 normalisation is always ours and always applied.** The chosen model has **no
  `Normalize` module** in its sentence-transformers pipeline despite its card recommending
  normalisation, so anything calling it through a raw path gets unnormalised vectors and
  cosine scores that silently disagree with every published number. `mxbai-embed-large-v1`
  has the same gap.

### 4.3 The chunk budget interaction, resolved

`parsing.md` §1.1 specifies that the chunker reads the effective sequence length from
`EmbedFingerprint` and **refuses to start** if the budget exceeds it. This section supplies
the number, and it must be the *usable* one:

> **`EmbedFingerprint.max_sequence_length` is usable content tokens** — the model's limit
> minus special tokens minus any document-side instruction prefix. It is not the raw
> `max_position_embeddings`.

For the chosen model: 8192 limit, no prefix, ModernBERT adds `[CLS]`/`[SEP]`, so **usable
≈ 8190**. The 512-token budget clears it with three orders of magnitude to spare.

The definition matters even though this model makes it moot, because it is what the rejected
candidates would have needed. `bge-base-en-v1.5` at 512 has **510** usable, so a 512-token
budget would overflow by two tokens and lose the tail of every full chunk — and the refusal
would fire on the shipped defaults. E5 would have **507** on the document side. The
effective budget is therefore `min(configured_budget, usable_tokens)`, computed at startup and
recorded in `ChunkFingerprint.max_tokens`, so the number that built the corpus is the number
stored rather than the number configured.

**`max_sequence_length` is a required field, not an optional one.** "Unknown" is precisely
the state that produces silent truncation, so the type does not permit expressing it — an
embedder that cannot report its own limit cannot be bound to a chunker. The refusal itself is
the chunker's obligation: it resolves the `Embedder` as a construction dependency and refuses
to start when `max_tokens > embedder.fingerprint.max_sequence_length`.

---

## 5. `EmbedFingerprint`

The fingerprint is what `storage.md` §6.3 compares to refuse a mismatched ingest, and — per
§8 — the embedding cache key. It must contain **everything that moves the vector for
identical text**.

| Field | Why it moves the vector |
|---|---|
| `model_id` | the weights |
| `revision` | a repository can be updated in place; a digest or commit pins it |
| `architecture` | determines which tensor the extraction path even reads (§3.2) |
| `dtype` | fp32 / bf16 / quantised weights give different outputs |
| `pooling` | CLS vs mean → 0.69 cosine at chunk length (§4.1) |
| `normalize` | unnormalised vs unit vectors are not comparable |
| `max_sequence_length` | usable content tokens; the chunker's refusal reads this (§4.3) |
| `query_prefix`, `document_prefix` | same weights, different prefix, different space |
| `dimension` | `D` |
| `backend` | onnxruntime vs a future MLX path; parity is asserted (§6), not assumed |

Every one of these was verified to change the output for identical input. The list is a
**minimum, not a closed set** — which is why comparison is by bytes and not field-by-field.
A field added later is caught for free by byte equality and is silently ignored by a
comparison written before it existed.

### 5.1 Canonical serialisation

One function produces fingerprint bytes, and nothing else may:

```python
def canonical(fp: Mapping[str, object]) -> bytes:
    return json.dumps(fp, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

fp8 = hashlib.sha256(canonical(fp)).hexdigest()[:8]
```

Each flag earns its place: `sort_keys` makes the bytes independent of construction order;
`separators` pins whitespace rather than describing it; `ensure_ascii` keeps encoding and
Unicode normalisation out of the comparison; `allow_nan` rejects values that are not JSON and
would round-trip inconsistently.

**Fingerprint values are restricted to `str | int | bool | None`** and nested containers of
those, enforced at construction. No floats — the failure is concrete rather than theoretical:

```
0.1 + 0.2  ->  {"x":0.30000000000000004}
0.3        ->  {"x":0.3}
```

Same intended value, two arrival paths, two different `fp8`s, two different vector table
names, and a spurious refusal. Anything fractional is carried as a string.

This function lives here because the fields are an embeddings concern; `storage.md` §4.6
persists the bytes it returns, and its §6.3 compares them. One definition, two consumers.

---

## 6. Tests that can actually fail

### 6.1 The 3-D assertion

Issue #3 is explicit that *"a test that only checks 'a vector came back' passes on the pooled
value and certifies the exact bug it exists to catch."* §3.2 shows why that is not
hypothetical.

```
assert token_states.ndim == 3
assert token_states.shape[0] == len(texts)
assert token_states.shape[1] == input_ids.shape[1]     # sequence, not 1
assert token_states.shape[2] == fingerprint.dimension
```

Checking `ndim == 3` alone is not enough — a `(batch, 1, D)` array is 3-D and is a pooled
vector wearing a sequence axis. Asserting the sequence length **against the tokenizer's own
output** is what makes it a real check, and it must run on **inputs of different lengths**,
because equal-length inputs make a padded batch and a pooled batch indistinguishable in
shape.

### 6.2 Parity between backends

The primary backend is a third-party runtime and a future MLX path would be our own code.
Neither is verifiable by reading it. **Parity is what makes a second implementation
admissible** — and it is the standing check on the first, since two independent
implementations agreeing is evidence and one implementation running is not.

**Parity is also the enforcement mechanism for the Apple-hardware principle**
(`PLAN.md` §7): optimise execution for the platform freely, but never let the platform change
what ends up in the index. A second backend is admissible only if it agrees with the first,
which is what keeps "runs faster on a Mac" from quietly becoming "indexes differently on a
Mac". If two backends cannot be brought within tolerance, the corpus is not portable across
machines — and that is a finding for the architect, not a tolerance to widen.

For every backend pair, over a fixture corpus spanning short, medium and full-budget inputs,
single and batched, ASCII and astral-plane:

| Assertion | Tolerance |
|---|---|
| Cosine similarity between backends, per input | **≥ 0.9999** |
| Max absolute element difference | ≤ 1e-3 |
| Vector norm, each backend | 1.0 ± 1e-5 |
| Fingerprints identical except `backend` | exact |

Cosine rather than element equality because fp arithmetic differs legitimately between
runtimes; 0.9999 rather than 1.0 for the same reason. A backend that cannot meet this is not
a backend — it is a different model, and it needs its own fingerprint and its own corpus.

### 6.3 Determinism and batch invariance

- **Same text, same vector, across runs and processes.** Non-determinism here churns the
  cache and makes the corpus unreproducible.
- **Batch invariance:** a text embedded alone and the same text embedded in a batch of eight
  with padding must agree to the parity tolerance. This is the test that catches unweighted
  mean pooling (§4.2) — which is correct on unpadded input and wrong the moment a batch has
  mixed lengths, so it passes any test written with same-length fixtures.

---

## 7. Traps, each verified

**`pooler_output` is not an embedding.** On BERT it is `tanh(dense(CLS))` — a head trained
for next-sentence prediction, not retrieval. Measured cosine to the raw CLS vector: **−0.05**.
It is the most naturally-named field in the output object and using it produces something
worse than random. Nothing in manicule reads it.

**fp16 ModernBERT NaNs on padded batches.** `gte-modernbert-base` ships `torch_dtype:
float16`, and in fp16 the additive attention mask saturates to `-inf`, NaN-ing the softmax for
padded rows. Measured: every padded row returned NaN token states; only unpadded full-length
rows were finite. **Use a bf16 or fp32 conversion.** `dtype` is in the fingerprint (§5), so
this is recorded rather than incidental, and a NaN check on output belongs in the same
assertion block as §6.1.

**The library's tokenizer default truncates long-context models.** The helper defaults to
`max_length=512` while the chosen model's tokenizer reports 8192, so a caller relying on the
default silently truncates. Immaterial at a 512-token budget and dangerous the moment anyone
raises it — which is another reason tokenization is ours (§4.2).

**Prefixes consume the token budget.** In sentence-transformers the prompt is
string-concatenated before tokenization, and `include_prompt: false` only excludes prompt
tokens from the *pooling average* — it does not restore truncation budget. There is no
configuration that makes a prefix free. Moot for the chosen model, which has none, and
recorded because it eliminated E5 (§1.2).

**Truncating a non-MRL model falls off a cliff.** §1.5. `D` is fixed at 768 here.

---

## 8. The embedding cache

An L2 cache of computed vectors, keyed by **the canonical `EmbedFingerprint` bytes plus the
exact string the embedder is asked to encode** — `(canonical(fp), embed_text) → vector`.

**"Exact" is load-bearing, and it is the `before:embed` hook that makes it so.** A middleware
running at that hook rewrites `embed_text` *after* the chunker produced it and *before* the
encoder sees it. So the cache must key on the **post-middleware** string. Keying on what the
chunker emitted would return a vector computed from different text — a cache that is wrong
only when a middleware is installed, which is the worst possible distribution of a bug.

The backend's own instruction prefix is **not** part of the key, because `document_prefix` is
already a fingerprint field (§5): the same text under a changed prefix gets a different
fingerprint and therefore a different key. Including it as well would be redundant, and
redundant key material is how two call sites end up disagreeing about what the key is.

**Not keyed by model name.** A name admits two models with different pooling, prefix, dtype or
revision, and would serve a confidently wrong vector for the right text with no error — the
same failure class as a dimension-only check, one layer down.

Two properties follow from using the same bytes `storage.md` §4.6 persists, and they are how
you know it is the right key:

- **Cache and index agree by construction.** A vector served from cache is admissible in the
  live table because both are keyed on the same object.
- **A fingerprint change cold-misses automatically.** There is no flush step to forget. This
  closes a real hazard: `reindex --re-embed` against a name-keyed cache would repopulate the
  *new* vector table with *old-space* vectors **and report success**, defeating the refusal
  mechanism from the inside — the cache is the one component that can launder a stale vector
  past the check designed to catch it.

**Not keyed by workspace or document, deliberately.** The cache is a memo over a pure
function; it has no result set to filter and no `k` to dilute. If two workspaces embed
identical text the correct vector is the same vector, and one computing it independently
would get identical bytes. Keying by workspace would destroy dedup exactly where hit rates
are highest — repeated boilerplate, a standard header, the same legal paragraph on 200 pages,
one attachment reachable from 40 Confluence pages — and multiply every one of those misses by
the number of workspaces. Soft-delete has no bearing either: the key is *text*, not chunk, and
a vector for text that also appeared in a deleted document is still the correct vector for
that text.

**Documented property, not a defect:** a cache shared across workspaces is a weak timing
oracle for "has anyone on this instance embedded exactly this text". It cannot reveal
content — probing requires already holding the text — and for a self-hosted single-tenant
tool that is a property worth stating so nobody rediscovers it as a finding and redesigns
around it.

---

## 9. Filed, not deferred

| Ticket | What | Why not v1 |
|---|---|---|
| [#30](https://github.com/mgd43b/manicule/issues/30) | **MLX backend implemented against `mlx` directly** (§3.4) | Metal-native execution is worth having, but only without the GPL dependency, only with a measured speedup over onnxruntime with CoreML, and only under §6.2 parity |
| [#31](https://github.com/mgd43b/manicule/issues/31) | **Multilingual model evaluation** (§1.6) | the chosen model is English-only; a Confluence instance with substantial non-English content needs `multilingual-e5-base` or `bge-m3`, and that is a corpus fact |
| [#15](https://github.com/mgd43b/manicule/issues/15) | **Model quality measured on manicule's own corpus** (§1.6) | every figure here is MTEB, which is open-domain and English-heavy. Belongs to #15 |

---

## 10. Checklist against ticket #3

- **MLX backend behind the `Embedder` protocol** — **changed, with reasons.** onnxruntime is
  primary (§3.3); the MLX path is filed (§9) and blocked on licence, measurement and parity.
  The two-tier protocol in `contracts.md` §3 is unchanged.
- **onnxruntime fallback** — promoted from fallback to primary, and to the reference
  implementation parity is asserted against (§6.2).
- **Pooling and L2 normalisation in our code, driven by model config** — §4.2.
- **A test asserting the token-state array is 3-D** — §6.1, including why `ndim == 3` alone
  is insufficient.
- **Model identity recorded alongside the index; a mismatch is a loud error** — §5, compared
  by `storage.md` §6.3.
- **Embedding cache keyed by model identity** — §8, tightened to the full fingerprint.
- **Fixes vector dimensionality** — **`D = 768`** (§1.5).
