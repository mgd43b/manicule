# Embeddings

Design for the embedding model, the runtime that executes it, and the pooling in between.
Ticket [#3](https://github.com/mgd43b/manicule/issues/3).

Everything here exists to prevent one failure: **a vector that is plausible and wrong.** An
embedder that crashes is a morning's work. An embedder that returns well-shaped vectors from
the wrong pooling, or the wrong half of a truncated chunk, degrades every answer the system
will ever give and raises nothing. Each section below is a guard against a specific instance
of that, and each guard is a test rather than a convention.

---

## 1. The model — `BAAI/bge-m3`

| | |
|---|---|
| Architecture | `XLMRobertaModel` (`model_type: xlm-roberta`) |
| Dimension | **1024** |
| `max_seq_length` | **8192** |
| Pooling | **CLS** |
| License | **MIT** |
| Languages | multilingual, 100+ |

Verified from the repository rather than the card prose: `config.json` gives
`architectures: ["XLMRobertaModel"]`, `hidden_size: 1024`, `max_position_embeddings: 8194`;
`sentence_bert_config.json` gives `max_seq_length: 8192`; `1_Pooling/config.json` gives
`pooling_mode_cls_token: true` with `pooling_mode_mean_tokens: false`.

That last line is the most operationally important fact in this document, and §4.1 is about
why. **None of it is hardcoded**: `manicule.embedding.cards` reads those four files at startup
and refuses to start when a model declares nothing, rather than filling in a default. A default
is a value that looks like a measurement and is not.

### 1.0 The weights are not in that repository

Found during implementation, and it changes a little of what follows. **`BAAI/bge-m3` publishes
`pytorch_model.bin` and no safetensors**, which is the only weight format MLX reads. So on
Apple hardware the bytes that actually execute come from a community conversion —
`mlx-community/bge-m3-mlx-fp16` — while the *identity* (pooling, dimension, tokenizer,
revision) still comes from `BAAI/bge-m3`. Reading identity from the artifact would fail
outright: the conversion ships no `1_Pooling/config.json` at all, so a backend reading its
pooling from what it loaded would find nothing and fall back to mean, on a CLS model.

Which conversion is not a detail, because the conversions are not equivalent:

| `mlx-community/bge-m3-mlx-…` | cosine to fp16 |
|---|---:|
| `fp16` | 1.0000 |
| `8bit` | 0.9996 – 0.9998 |
| `4bit` | 0.9249 – 0.9694 |

Measured, on CLS-pooled normalized vectors, on the texts §6.2 uses. **A quantized conversion is
a different vector space wearing the same name**, and since `backend` is excluded from
fingerprint identity nothing downstream would notice one mixed into an index built by another.
So quantization is **refused at load**, by the one component positioned to see it. That is the
Apple-hardware principle drawn precisely: use Metal, use fp16 storage, use whatever is fastest,
and stop at anything that moves the vectors.

The artifact that ran is recorded in `EmbedFingerprint.weights_ref` (§5) so a vector can be
traced to its bytes.

### 1.1 The license gate moved, and it moved the whole section

An earlier draft of this design eliminated candidates on a **permissive-license** gate, and
chose the runtime on the same grounds. manicule is now **GPL-3.0-or-later** (`LICENSE`), and
that gate is gone in one direction and intact in the other:

- **GPL-3.0 dependencies are ordinary.** `mlx-embeddings` is GPL-3.0 — verified from the
  installed distribution metadata, `Classifier: License :: OSI Approved :: GNU General Public
  License v3 (GPLv3)` — and was the reason the relicense happened at all. It is usable.
- **AGPL-3.0 dependencies are still refused**, because AGPL §13 puts a source obligation on
  anyone *running* the result as a network service, and manicule ships an HTTP API and a web
  UI. That obligation would fall on operators rather than on us. See
  [`parsing.md`](parsing.md) §12.
- **Non-commercial and gated licenses remain eliminated outright**, which is what removed
  `jina-embeddings-v3` (CC-BY-NC-4.0) and `google/embeddinggemma-300m` (gated Gemma Terms).

BGE-M3's weights are **MIT**, so none of this constrains the model itself. It constrains what
may execute it, and the answer is now "either backend" (§3).

### 1.2 Why this one

Four properties, in the order they matter.

**It is multilingual, and that absorbs a ticket.** BGE-M3 is trained across 100+ languages in
one vector space, so a query in one language retrieves passages in another.
[#31](https://github.com/mgd43b/manicule/issues/31) existed to add a multilingual model
later; it is closed by this choice rather than left open.

**Its architecture is the safe side of a live hazard.** `mlx-embeddings` returns genuine 3-D
token states for `xlm_roberta` and rebinds `last_hidden_state` to the *pooled* vector for
`modernbert` (§3.2). The outgoing candidate was a ModernBERT model, so this choice walks away
from the one architecture where that trap is armed. That is a real reduction in risk, not a
tie-breaker.

**8192 tokens leaves the chunk budget no room to be wrong.** The budget is 512
([`parsing.md`](parsing.md) §1), so the refusal in §4.3 has sixteen times the headroom it
needs. Several otherwise-strong candidates ship configured at 512 or below, where a 512-token
budget overflows by the special tokens alone.

**One model, one space, dense retrieval.** BGE-M3 also has learned-sparse and ColBERT
multi-vector heads. **They are out of scope** — §1.4 says what that costs and what it does
not.

### 1.3 What `D = 1024` costs, honestly

The outgoing candidate was 768. Larger vectors cost storage and distance time, both linear in
`D`, and neither is a reason to prefer a worse model.

| | D = 768 | **D = 1024** |
|---|---:|---:|
| Bytes per vector (float32) | 3 072 | **4 096** |
| 100 000 chunks | 293 MiB | **391 MiB** |
| 1 000 000 chunks | 2.9 GiB | **3.9 GiB** |

A gigabyte per million chunks, and roughly a third more arithmetic per distance computation.
For a self-hosted corpus that is not a constraint worth trading retrieval quality or
multilingual coverage for. Two things already in the design absorb most of it: exhaustive
search stays exact below the ANN threshold ([`storage.md`](storage.md) §6.2), and vectors are
the *derived* store — rebuildable, and not what a backup is protecting
([`storage.md`](storage.md) §1).

Recorded rather than argued: an earlier draft made a 768-over-1024 case specific to the models
then under consideration. It does not transfer, and it is not re-derived here.

### 1.4 Dense only — what the other two heads are, and what skipping them costs

BGE-M3's distinguishing feature is three representations from one model: **dense**,
**learned sparse** (a lexical weight per vocabulary token), and **ColBERT** (a vector per
token, scored by late interaction). **manicule uses the dense leg and nothing else.**

**Neither backend exposes the other two anyway.** Verified by reading the installed package:
`mlx_embeddings/models/xlm_roberta.py` returns `last_hidden_state`, `text_embeds` and
`pooler_output`, and there is **no `sparse_linear` or `colbert_linear` anywhere in the
distribution**. The heads exist as `sparse_linear.pt` and `colbert_linear.pt` in the model
repository — separate weight files, not part of `XLMRobertaModel` — so using them means the
reference implementation (`FlagEmbedding`) and a different runtime, not a flag.

**What that costs is bounded and known.** The lexical leg of hybrid retrieval is SQLite FTS5
BM25, built and tested ([`storage.md`](storage.md) §6.1), and it is unaffected by this. A
learned-sparse leg would be an *alternative* to BM25, not a missing piece — and swapping it
in is a retrieval decision that must earn its place with a measured improvement on the
[#15](https://github.com/mgd43b/manicule/issues/15) baseline, exactly like every other
retrieval feature. Filed against [#6](https://github.com/mgd43b/manicule/issues/6) with
`needs-evidence` rather than left as an aspiration here.

### 1.5 What would have to be true to change it

Both, not either:

1. **A measured improvement on the [#15](https://github.com/mgd43b/manicule/issues/15)
   baseline**, on a fixed query set, on a corpus resembling the real one.
2. **The switch is affordable when it happens** — which §2 is about, and which is now a
   supported operation rather than a migration.

Absent both, `bge-m3` stands.

---

## 2. The model is configuration, not a constant

**One model is active at a time, and which one is a setting.** Not concurrent models, not
per-workspace models, and no routing between them — those were considered and are not being
built.

This is mostly already true, because the storage design was built for it:

| Mechanism | Where | What it gives |
|---|---|---|
| `index_state.vector_table` is a **pointer** | [`storage.md`](storage.md) §4.6 | The new table is built alongside the old one and the pointer moves in one transaction |
| `EmbedFingerprint` refusal | [`storage.md`](storage.md) §6.3 | A mismatched index refuses to open — for retrieval as well as ingest |
| Fingerprint persisted in three places | [`storage.md`](storage.md) §6.3 | Config, SQLite and the vector directory must agree |
| `chunks.embed_text` is stored | [`storage.md`](storage.md) §4.3 | Re-embedding needs no re-chunk, no re-parse and no re-fetch |

So switching models is **rung 2 of the blast-radius ladder** — a full re-embed, from data
already on disk. No schema change is required to support plurality, and none is made here.

**The known-good set** is configuration with a validated default:

```toml
[embedding]
model = "BAAI/bge-m3"          # the default
provider = "mlx"               # or "onnx"

[plugins.config."embedder.mlx"]
weights = ""                   # the artifact to execute, when it is not the model's own repo
pooling = ""                   # only for a model that declares none; contradicting one is refused
max_sequence_length = 0        # only for a model that declares none, in usable content tokens
```

A model outside the known-good set is accepted, and everything that decides vector-space
compatibility is read from **its own repository** rather than from a table — which is stronger
than the earlier wording here, that such a model "must supply its own fingerprint fields". It
must supply them only where its repository declares nothing, and a setting that *contradicts* a
declaration is refused rather than obeyed: a setting that overrules how the weights were
trained is worse than one that is ignored, because it succeeds.

There is no `normalize` setting. Normalization is always applied (§4.2), so there is nothing to
configure and no way to configure it wrongly.

**What switching costs, stated up front rather than discovered:**

- A full re-embed of every chunk. Bounded by corpus size, resumable, and priced by
  `reindex --re-embed`, which prints the chunk count and an estimate before starting
  ([`storage.md`](storage.md) §6.3).
- **Citations survive.** `chunks.id` derives from `(document_id, position, text)` and not from
  the model, so anchors, quotes and stored citations are untouched. This is the property that
  makes a model switch an operation rather than a migration.
- The old vector table is dropped only after the pointer moves, so an interrupted switch
  leaves the old index live and serving.

---

## 3. The runtime

### 3.1 MLX is the primary backend, and the relicense is why

An earlier draft of this design made **onnxruntime** primary, on the grounds that
`mlx-embeddings` is GPL-3.0 and manicule was MIT. That premise is gone: the project is
GPL-3.0-or-later, chosen deliberately so that this dependency is usable
([`parsing.md`](parsing.md) §12).

With the license objection removed, the Apple-hardware principle decides it:

> Optimize execution for Apple hardware freely; **never let the platform change what ends up
> in the index.**

MLX gives Metal-native execution in-process on the machine this is built for. onnxruntime
gives the same in-process property everywhere else. Both satisfy "no server to operate", so
the platform argument is what remains, and it favors MLX on Apple Silicon.

**The maturity objection is not withdrawn, it is converted into a test.** `mlx-embeddings` is
version **0.1.0** and gives the same attribute different meanings on different architectures
(§3.2). That is exactly the profile that produces silent wrongness — so §3.4 makes the second
backend the mechanism that catches it, rather than a fallback nobody runs.

### 3.2 What the backend does per architecture, measured

Loading real checkpoints and inspecting the returned object:

| Architecture | `last_hidden_state` is | Verified on |
|---|---|---|
| `bert` | **3-D token states** | `bge-small-en-v1.5` |
| **`xlm_roberta`** | **3-D token states** | **`bge-m3`** |
| `gemma3_text` | 3-D token states | `embeddinggemma-300m` |
| `qwen3` | 3-D token states | `Qwen3-Embedding-0.6B` |
| **`modernbert`** | **2-D pooled vector** | `nomicai-modernbert-embed-base` |
| `modernbert` with `ForMaskedLM` | 3-D token states | `answerdotai/ModernBERT-base` |

Corroborated by reading the source. In `models/xlm_roberta.py` the returned
`last_hidden_state=sequence_output` is the encoder output. In `models/modernbert.py` the local
variable is **reassigned** — `last_hidden_state = last_hidden_state[:, 0]` for CLS, or
`mean_pooling(...)` — and then returned under the same name. The same module therefore returns
different ranks depending on the checkpoint's `architectures` field.

**An inconsistent lie is worse than a uniform one.** A uniform one is caught the first time
anyone checks. This one is correct on BERT, correct on XLM-RoBERTa, and wrong on ModernBERT —
so it passes review on whichever model the reviewer happened to try. It is the direct reason
for the 3-D assertion in §6.1, and it is the reason §1.2 counts XLM-RoBERTa as a risk
reduction rather than a coincidence.

**The ONNX side has the same problem in a different shape, found during implementation.**
Nothing standardizes an export's output names. `BAAI/bge-m3` calls its two outputs
`token_embeddings` and `sentence_embedding` — *neither* of which is `last_hidden_state`, so a
name-based lookup finds no token states at all. `BAAI/bge-small-en-v1.5` publishes only the
3-D output and no pooled one. And reading `outputs[0]` is a guess about export order.

So the ONNX backend **selects its output by rank, never by name or position**: exactly one
output is `(batch, sequence, dimension)`, and an export where that is not true cannot serve as
a tier A backend and says so. Rank is the property that is actually stable across exporters.

### 3.3 onnxruntime is the parity check, not the fallback

Both backends are supported, and the second one has a job beyond portability.

| | MLX | onnxruntime |
|---|---|---|
| License | GPL-3.0 | MIT |
| In-process | yes | yes |
| Apple Silicon | Metal-native | yes, CPU |
| Off Apple Silicon | no | yes |
| Maturity | 0.1.0 | mature |
| `bge-m3` availability | `xlm_roberta` implemented | ONNX export in-repo (`onnx/model.onnx`) |

**The requirement: same text, same model, same pooling, both backends, vectors within a
stated tolerance.** Asserted in tests (§6.2), not assumed. `backend` is recorded in
`EmbedFingerprint` and **excluded from its identity**, and that exclusion is precisely the bet
this parity test exists to verify — it is what lets a corpus move between machines without a
re-embed. If parity cannot be brought within tolerance, the correction is to move `backend`
into the identity set, which makes a runtime change a loud error with a re-embed path.

**Measured, and it holds.** MLX on fp16 weights against the fp32 ONNX export, CLS-pooled and
L2-normalized, on short English, non-English, code and a 402-token passage:

| | cosine, worst of four | largest component difference |
|---|---:|---:|
| `bge-m3` | 1.000000 | 1.8 × 10⁻⁵ |
| `bge-small-en-v1.5` | 1.000000 | 2.4 × 10⁻⁵ |

So `backend` stays out of `IDENTITY_FIELDS`, and the two runtimes write byte-identical
canonical fingerprints — which is what an index actually compares. The gate in the test sits
about a hundred times looser than the measurement, and still tight enough to catch a
substituted model: 8-bit quantized `bge-m3` scores 0.9998 and fails it.

**One thing the ONNX backend gives up for this.** It runs on `CPUExecutionProvider`, including
on Apple Silicon where CoreML would be faster. A reference that varies by accelerator measures
nothing, and the fast path on Apple hardware already exists — it is called MLX.

That is the Apple-hardware principle in operative form. Throughput may differ by machine.
Vectors may not.

---

## 4. Pooling — the thing this ticket exists to get right

### 4.1 BGE-M3 pools with CLS, and the convenience field mean-pools

This is the trap for this model, and it is armed by default.

`1_Pooling/config.json` in the model repository declares `pooling_mode_cls_token: true` and
`pooling_mode_mean_tokens: false`. **BGE-M3 is a CLS-pooled model.**

`mlx_embeddings/models/xlm_roberta.py` computes its convenience output as:

```python
text_embeds = mean_pooling(sequence_output, attention_mask)
text_embeds = normalize_embeddings(text_embeds)
```

**Unconditionally, for every XLM-RoBERTa checkpoint.** So a caller that reaches for
`text_embeds` — the obviously-named field, the one an example would use — gets **mean pooling
on a model trained for CLS**, silently, with correctly-shaped normalized vectors and no error.

This is not a hypothetical version of the pooling hazard. It is the specific, current,
default-path instance of it for the model this project has chosen, and it is why the pooling
path below reads token states and pools them here rather than trusting any field the backend
offers.

**How wrong is it?** CLS and mean pooling of the same token states diverge with sequence
length. Measured on `gte-modernbert-base`, both L2-normalized, from true 3-D token states:

| tokens | 7 | 42 | 156 | **452** |
|---|---:|---:|---:|---:|
| cosine | 0.900 | 0.843 | 0.765 | **0.693** |

The same comparison stays mild on BERT (0.87–0.96). So the magnitude is architecture-dependent
and the *direction* is not: the two poolings disagree more the longer the chunk, and the chunk
budget is 512 — the far right of that curve.

**Now measured on BGE-M3 itself**, which is what this section previously deferred to §6 rather
than quoting from another model. Real token states, both reductions L2-normalized:

| tokens | 15 | 20 | 22 | **402** |
|---|---:|---:|---:|---:|
| cosine(CLS, mean) | 0.788 | 0.801 | 0.761 | **0.659** |

Worse than the ModernBERT curve above at comparable lengths, and the reduction the obvious
field hands you. Confirmed in the same run: `text_embeds` reproduces our *mean* pool to cosine
1.000, so the disagreement is the reduction and not arithmetic. `pooler_output` measures
**−0.04 to +0.02** against raw CLS — not a worse version of it, unrelated to it.

### 4.2 The pooling path

```
tokenize(texts) -> input_ids, attention_mask     # ours; masks are needed and not surfaced
run model       -> token_states (batch, seq, D)  # 3-D, asserted (§6.1)
pool            -> vector (batch, D)             # CLS, per the model's declared config
L2 normalize    -> unit vector
```

Four rules:

- **Pooling is read from the model's declared configuration, never assumed**, and recorded in
  the fingerprint. For BGE-M3 it is CLS. A model whose pooling differs cannot share an index
  with one whose pooling does not, because the fingerprint refuses.
- **Mean pooling, where a model calls for it, is attention-mask-weighted.** An unweighted mean
  over a padded batch averages in the padding, so the same text produces a different vector
  depending on what else was in its batch — a batch-order-dependent index.
- **We tokenize.** The backend does not surface attention masks from its embedding call, and
  tokenizing ourselves also makes truncation explicit rather than inherited from a library
  default (§7).
- **L2 normalization is ours and is always applied**, so a model repository that omits a
  `Normalize` step cannot produce vectors whose cosine scores silently disagree with every
  published number.

### 4.3 The chunk budget interaction, resolved

[`parsing.md`](parsing.md) §1.1 has the chunker read the sequence length from
`EmbedFingerprint` and **refuse to start** if the budget exceeds it. This section supplies the
number, and it must be the *usable* one:

> **`EmbedFingerprint.max_sequence_length` is usable content tokens** — the model's limit
> minus special tokens minus any document-side instruction prefix. Not raw
> `max_position_embeddings`.

For BGE-M3: an 8192 limit, no instruction prefix, XLM-RoBERTa wraps with `<s>` and `</s>`, so
**usable ≈ 8190**. The 512-token budget clears it by a factor of sixteen.

The definition still matters, because it is what a future model will need: `bge-base-en-v1.5`
at 512 has **510** usable, so a 512-token budget would overflow by exactly the special tokens
and lose the tail of every full chunk — with the refusal firing on shipped defaults. The
effective budget is `min(configured_budget, usable_tokens)`, computed at startup and recorded
in `ChunkFingerprint.max_tokens`, so the number that built the corpus is the number stored.

`max_sequence_length` is required, not optional. "Unknown" is exactly the state that produces
silent truncation, so the type does not permit expressing it.

---

## 5. `EmbedFingerprint`

The type ships in `manicule.core.embedding`. Its identity set is `model_id`, `revision`,
`dimension`, `pooling`, `normalized`, `tokenizer_id`, compared as canonical bytes
([`storage.md`](storage.md) §4.6). For BGE-M3:

```
model_id            BAAI/bge-m3
revision            <commit sha, pinned>
dimension           1024
pooling             cls
normalized          true
tokenizer_id        BAAI/bge-m3
max_sequence_length 8190      # recorded, excluded from identity
backend             mlx|onnx  # recorded, excluded from identity — §3.3
weights_ref         mlx-community/bge-m3-mlx-fp16   # recorded, excluded from identity — §1.0
```

**`revision` is pinned rather than left unset.** The type permits `None` and records it
faithfully, but an unpinned model is one whose weights can change under a corpus without the
fingerprint noticing — the exact silent-degradation this whole document is about.

Three exclusions carry reasoning that must not be lost, and all three are stated in the
shipped docstring: `max_sequence_length` is excluded because including it would force a
re-embed whenever the limit *rises*, and the property that matters is checked directly by
`require_within_context`; `backend` is excluded on the parity bet §3.3 now measures;
`weights_ref` is excluded for the same reason and rests on the same measurement, with the one
re-encoding that *does* move the vectors — quantization — refused at load rather than absorbed
here (§1.0).

**`weights_ref` is added by this ticket**, additively and outside identity, so no stored
fingerprint changes meaning. It exists because §1.0 turned out to be true: the artifact that
runs is frequently not the repository named in `model_id`, and an index that cannot say which
bytes made its vectors is not diagnosable.

**`architecture` is still not an identity field, and implementation settled the case against
adding it.** The concern was that architecture decides which tensor the extraction path reads,
upstream of pooling. Two things answer it. `revision` is pinned, and a pinned commit fixes the
architecture along with everything else, so the field is functionally determined wherever it
would matter. And the failure it guards against is now caught directly rather than inferred:
reading the wrong tensor produces the wrong *rank*, which `require_token_states` refuses on
every batch (§6.1). A recorded-but-unenforced field would have been a third thing to keep in
step with no failure it could prevent. `ModelCard.architecture` carries it for diagnostics.

---

## 6. Tests that can actually fail

Where they live: `tests/test_embedding_pooling.py` and `tests/test_embedding_models.py` need no
weights and run everywhere; `tests/test_embedding_embedder.py` runs the shared path against
stub backends; `tests/test_embedding_backends.py` needs real weights, finds them in the local
model cache, and skips what is absent — except under `MANICULE_REQUIRE_EMBEDDING_MODELS`, which
CI sets after pre-seeding, because a conformance suite that skips reports green while checking
nothing.

Every negative check has a fake that shows it firing, in `tests/embedding_fakes.py`:
`PrePooledEmbedder` returns the pooled vector where token states belong, `WrongWidthEmbedder`
returns the wrong dimension, `UnmaskedMeanEmbedder` pools without the mask, and `NameKeyedCache`
keys on the model name. Each was confirmed load-bearing by disabling the guard and watching the
suite go red.

### 6.1 The 3-D assertion

Before anything is pooled, assert the token states are rank 3 with shape
`(batch, sequence, dimension)`. On a backend that returns a 2-D pooled vector under the same
name, pooling silently becomes a no-op over the batch axis and every vector is wrong in the
same plausible way.

This is not defensive programming — §3.2 documents a shipped library that does exactly this on
a neighboring architecture.

### 6.2 Parity between backends

Same texts, same model, same pooling, MLX and onnxruntime, cosine within a stated tolerance
per vector and on the mean. This is the enforcement mechanism for §3.3's exclusion of
`backend` from fingerprint identity, and it is the test that decides whether a corpus is
portable between machines.

The tolerance is stated in the test rather than left implicit, and a failure is not a flake:
it means one backend is not computing what the other is, and the response is to move
`backend` into the identity set rather than to widen the tolerance.

### 6.3 Pooling is the model's, not the backend's

Embed a text through the full path and through the backend's convenience field, and assert
they **differ** for BGE-M3 — because the convenience field mean-pools and the model is CLS
(§4.1). A test asserting they *match* would be asserting the bug. The same test also asserts
that the convenience field equals our *mean* pool, which is what makes the disagreement a
difference of reduction rather than of arithmetic.

**And the mirror image, on ONNX.** `bge-m3`'s export publishes a pooled output that genuinely
*is* the declared CLS pooling — cosine 1.000 to our own path — while `bge-small-en-v1.5`'s
publishes none at all. So the shortcut is wrong on MLX, right on one ONNX export and absent
from another, with nothing in any of the names to tell them apart. That is why the rule is
"never trust it" rather than "prefer ours where it matters", and it is asserted rather than
described.

### 6.4 Determinism and batch invariance

The same text embeds to the same vector across runs, and to the same vector regardless of
what else shares its batch. Batch-dependence is what an unmasked mean pool produces, and it is
invisible without this test because every individual vector looks fine.

### 6.5 The shipped conformance suites

`assert_embedder_contract` and `assert_refuses_oversized_chunks` from `manicule.testing`, plus
`assert_protocol_signatures` against `Embedder` and `TokenStateEmbedder` —
`@runtime_checkable` checks that an attribute exists and never what it accepts.

---

## 7. Traps

- **Truncation is silent.** Past `max_sequence_length` the input is dropped with no error, and
  the stored vector describes an opening fragment while the chunk claims all of its text.
  `require_within_context` is called on every path that embeds stored chunks, and re-embed is
  the path that needs it most because it does not re-chunk.
- **The convenience field is the wrong pooling** for this model. §4.1.
- **`max_position_embeddings` is not the usable length.** BGE-M3's config says 8194 and its
  usable content length is ~8190; other models ship configured far below their architectural
  limit, and the shipped number is the one that truncates.
- **An unpinned `revision` lets weights change under a corpus** without the fingerprint
  changing.
- **Normalization may be absent** from a model's declared pipeline even when its card
  recommends it, so it is always applied here rather than assumed.
- **A conversion is not the model.** `BAAI/bge-m3` ships no safetensors, so MLX runs community
  weights; a quantized one is a different vector space under the same name and is refused at
  load. §1.0.
- **An ONNX export's output names mean nothing.** `bge-m3`'s 3-D output is called
  `token_embeddings`, so a name-based lookup finds no token states at all. Select by rank. §3.2.
- **MLX is lazy, and its streams are thread-local.** A forward pass returns an unevaluated
  graph; materializing it on a different thread aborts the process — `libc++abi: terminating …
  There is no Stream(gpu, N) in current thread` — which is not an exception and cannot be
  caught. Found the first time the backend ran under `asyncio.to_thread`, which hands out
  whichever pool thread is free. Each embedder therefore owns **one** worker thread, loads on
  it, runs on it, and converts to numpy on it before returning.
- **The Hugging Face cache follows `XDG_CACHE_HOME`.** The suite redirects that per test, so a
  model sitting on disk becomes invisible and every model suite skips — green, having checked
  nothing. `tests/conftest.py` pins `HF_HUB_CACHE` for the session, the same hazard the
  tree-sitter grammar cache had and the same fix.
- **A switch inside `MANICULE_` is deleted before it is read.** The test environment clears that
  namespace per test so a developer's configuration cannot leak in. The switch that turns a
  missing model from a skip into a failure was first named `MANICULE_REQUIRE_EMBEDDING_MODELS`
  and was therefore scrubbed: CI set it, every case skipped, and the job reported success — the
  failure the switch exists to prevent, occurring inside it. It is now
  `REQUIRE_EMBEDDING_MODELS`, read at import time, and a test asserts both. Found by reading a
  green CI log, not by a test.

---

## 8. The embedding cache

A pure-function memo: `(canonical(EmbedFingerprint), embed_text) -> vector`.

**Keyed by the full canonical fingerprint, not a model name.** `PLAN.md` §16's "key by model
identity" is too loose: the same weights with a different pooling, prefix, dtype or revision
produce a different vector for the same text, and a name-keyed cache would serve a confidently
wrong one. Keying on the same bytes `index_state` stores gives two properties for free — a
cached vector is admissible in the live index by construction, and a fingerprint change
invalidates the cache automatically, with no flush step to forget.

That second property closes a real hazard: `reindex --re-embed` against a name-keyed cache
would repopulate the new table with old-space vectors **and report success**.

**The key is the post-middleware `embed_text`** — the exact string handed to the model — not
the pre-middleware text. A middleware installation would otherwise make the cache return
vectors computed from different text.

**Not keyed by workspace or document**, which would destroy the deduplication that makes the
cache worth having, exactly where hit rates are highest: repeated boilerplate, the same
attachment reachable from forty pages. A cache hit reveals nothing — the caller already holds
the text, and computing it themselves returns the identical vector. One honest caveat: a
shared cache is a weak timing oracle for "has anyone here embedded this exact text", which is
a documented property of a self-hosted tool rather than a defect.

---

## 9. Filed, not deferred

| Ticket | What | Why not here |
|---|---|---|
| [#6](https://github.com/mgd43b/manicule/issues/6) | **BGE-M3 learned-sparse leg** as an alternative to FTS5 BM25 (§1.4) | A retrieval feature, and retrieval features earn their place with a measured improvement on #15. Labeled `needs-evidence`. Neither backend exposes the head today, so it is a runtime change as well as a retrieval one |

## 10. Checklist against ticket #3

- **Model settled** — §1, `BAAI/bge-m3`, 1024d, 8192 tokens, MIT, verified from the repository.
- **Dimensionality is a runtime parameter** — read from `EmbedFingerprint`, never a literal;
  the vector table is created at first ingest ([`storage.md`](storage.md) §6.3).
- **Pooling is ours** — §4, CLS per the model's declared config, from asserted 3-D token
  states, never from the backend's convenience field.
- **Two backends, with parity as the enforcement** — §3.3, §6.2. Measured: cosine 1.000000,
  largest component difference 1.8 × 10⁻⁵. `backend` stays out of identity.
- **The model is switchable configuration** — §2, with the cost of switching stated.
- **Multilingual** — absorbs [#31](https://github.com/mgd43b/manicule/issues/31).
- **The cache is keyed on the canonical fingerprint** — §8, and `NameKeyedCache` in the test
  fakes shows what the alternative serves.
- **`require_within_context` on every path that embeds chunks** — `PooledEmbedder.embed_chunks`,
  which is the path re-embed uses, held to it by `assert_refuses_oversized_chunks`.

### 10.1 What implementation changed in this document

Recorded rather than quietly folded in, because each was a claim that turned out to be wrong or
unfinished:

| Section | Was | Is |
|---|---|---|
| §1.0 | absent | The MLX weights are a community conversion, and quantized ones are refused |
| §2 | "a model outside the known-good set must supply its own fingerprint fields" | It supplies only what its repository fails to declare, and may not contradict it |
| §3.2 | the hazard was MLX's | ONNX exports name their outputs freely too; selection is by rank |
| §3.3 | parity was a bet | parity is a measurement, and it holds |
| §4.1 | the BGE-M3 cosine was deferred to §6 | 0.66–0.80, measured |
| §5 | `architecture` "raised, not added" | settled against, with the reason |
| §5 | two non-identity fields | three: `weights_ref` added |
| §7 | five traps | nine; four of the new ones only appear when the code runs, and one only in CI |
