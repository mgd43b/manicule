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
`artifact:ollama:<model>@sha256:<digest>:prefix=none`. An `ollama pull` that changes the bytes
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

**It applies no query/document prefix.** `nomic-embed-text` is trained with asymmetric
`search_query:`/`search_document:` prefixes and Qwen3-Embedding with a query-side instruction,
so applying one — or failing to — changes the vector for the same text. manicule has no
query/document distinction to hang that on: `Embedder.embed` is the only entry point, and
ingest and dense retrieval call it identically, so a backend cannot tell which side it is
serving. Inventing a rule here would put a retrieval decision inside a plugin and leave it out
of the fingerprint entirely. So this backend applies nothing and *says so in the identity*: the
`prefix=none` term is what makes a future prefix mechanism a fingerprint change and a re-embed
rather than a silent quality regression.

**It cannot be told how to pool.** The server pools. Configuration's `pooling` is consulted
only when the GGUF declares none, and a setting that contradicts the GGUF is refused — because
it would succeed, and record a fingerprint claiming the vectors came from a reduction they did
not.

## Installation

Not on PyPI yet. Install from the repository:

```bash
uv pip install "manicule[all]" ./packages/manicule-ollama
```

MIT, like manicule itself. Unlike `manicule-mlx` — whose separate distribution is a licensing
consequence — this one is separate because its model is a remote service, and the HTTP client
and deployment topology that come with that should not be imposed on an installation embedding
in process.
