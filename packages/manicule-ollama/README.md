# manicule-ollama

An embedding backend for [manicule](https://github.com/mgd43b/manicule) that runs the model on
an [Ollama](https://ollama.com) server instead of in this process.

It exists for a specific shape of deployment: a host that is the wrong place to embed on — the
case it was written for is a pod on Ivy Bridge Xeons, with no AVX2 for onnxruntime's fast
kernels to use — next to a GPU node already running Ollama for generation. Moving the forward
pass across the network is what makes that installation practical.

```toml
[embedding]
provider = "ollama"
model = "qwen3-embedding:0.6b"

[plugins.config."embedder.ollama"]
base_url = "http://ollama.internal:11434"
tokenizer = "Qwen/Qwen3-Embedding-0.6B"
tokenizer_revision = "<the exact 40-character commit>"
```

`prefix_scheme` belongs with that: `qwen3-embedding:0.6b` wants `prefix_scheme = "qwen3"` and
`nomic-embed-text` wants `"nomic"`. It is `[embedding]` rather than a setting of this backend's
because it is not this backend's to decide — see below.

**`tokenizer` takes a directory as readily as a repository id**, which is how this backend is
configured inside the container image. That image sets `HF_HUB_OFFLINE=1` and points `HF_HOME`
at a baked, read-only path, so a repository id that was not pre-seeded at build time cannot be
fetched — while a local path is answered without `huggingface_hub` being imported at all:

```toml
tokenizer = "/models/qwen3-embedding"   # a directory holding tokenizer.json; no revision
```

Mount one `tokenizer.json` there, or bake a cache on a machine that has a network with
`uv run tools/prefetch_embedding_models.py --backend ollama`, which reads `tokenizer` and
`tokenizer_revision` straight out of this configuration. A tokenizer that cannot be resolved
is a refusal naming both routes, at construction, before a corpus exists.

## What it measures, and why there is so much of it

manicule calls this a **tier B** backend: the server pools the token states and hands back a
finished vector, so the reduction actually applied cannot be verified by inspection. The
`Embedder` protocol calls that "the floor rather than the norm", and backends are admitted on
that basis. What follows is the price of admission.

**The width is measured, not declared.** The fingerprint is the sole source of the vector
dimension and the vector table is created from it, so this backend embeds a probe string at
construction and counts the vector that comes back. The GGUF's `embedding_length` is read too,
and a disagreement between the two is refused rather than resolved.

**The identity carries the server's own digest.** `EmbedFingerprint` leaves `backend` out of
identity only because `weights_identity` carries the runtime boundary, and portability between
backends is an allowlisted measurement rather than an inference. Nothing licenses this backend
to share an identity with any other, so it does not: `weights_identity` is
`artifact:ollama:<model>@sha256:<digest>`. An `ollama pull` that changes the bytes
behind an unchanged tag changes the digest, which invalidates the vectors the previous pull
made — and `health()` fails loudly if it happens while manicule is running.

**The limit is the served one, not the declared one.** Measured against a server holding
`qwen3-embedding:0.6b`, whose GGUF declares a 32768-token context: an `/api/embed` carrying no
options was served at **4096**, and a longer input came back as a well-formed vector built from
its first 4095 tokens. So this backend sends `num_ctx` on every request and derives
`max_sequence_length` from the number it sent — then checks at setup that the server honors it.

**Truncation is turned off at the server.** Ollama's `truncate` defaults to true, which is the
silent failure `require_within_context` exists to catch. Every request sets it false, so an
over-long input is a refusal rather than a vector describing an opening fragment. Setup proves
the flag is in force by sending something too long and requiring the 400.

**The tokenizer is configuration, and it is checked.** Ollama serves GGUF and exposes no
tokenizer, while manicule counts tokens to place chunk boundaries and to refuse text the model
would truncate. So `tokenizer` is required and has no default — deriving one from the model's
name would be a guess recorded in an identity field. What makes it admissible is that setup
embeds probe strings one at a time and compares this vocabulary's counts against the server's
own `prompt_eval_count`. A single token of disagreement on a single probe is a refusal.

## What it deliberately does not do

**It applies no query/document prefix — core does.** `nomic-embed-text` is trained with
asymmetric `search_query:`/`search_document:` prefixes and Qwen3-Embedding with a query-side
instruction, so applying one, or failing to, changes the vector for the same text. A backend
cannot do it: `Embedder.embed` is the only entry point, and ingest and dense retrieval call it
identically, so nothing in here can tell which side it is serving. So the scheme is
`[embedding] prefix_scheme`, it is recorded in `EmbedFingerprint.prefix_scheme`, and it is
applied at the two call sites that know — `manicule.ingest.embedding` and
`manicule.retrieval.dense`. `docs/embeddings.md` §9.1 has the argument, including why
`ChunkFingerprint.embed_text_middleware` looks like the home for it and is not.

This backend's `weights_identity` once ended `:prefix=none`, a marker standing in for that
field while core had none. It is gone, and its removal is the point: `weights_identity` is
written by whichever backend built the fingerprint, so the term protected *this* backend while
`onnx` and `mlx` recorded nothing — and now that core carries the question for all three,
keeping it would record the same fact twice in the field least able to be compared across
them.

**It cannot be told how to pool.** The server pools. Configuration's `pooling` is consulted
only when the GGUF declares none, and a setting that contradicts the GGUF is refused — because
it would succeed, and record a fingerprint claiming the vectors came from a reduction they did
not.

## Installation

It comes with `manicule[all]`, and with the container image, because the image is the one place
an operator cannot add a backend afterwards:

```bash
uv tool install "manicule[all]"     # or just the backend: manicule[ollama]
```

MIT, like manicule itself. Unlike `manicule-mlx` — whose separate distribution is a licensing
consequence — this one is separate because its model is a remote service, and the HTTP client
and deployment topology that come with that should not be imposed on an installation embedding
in process.

The same asymmetry decides what the container carries. `manicule-mlx` is excluded from it
because there is no Linux image in which Metal is a valid answer; this one is included because
an HTTP client is valid *precisely* in a container — a pod that cannot embed well beside a GPU
node that can is the deployment it was written for.
