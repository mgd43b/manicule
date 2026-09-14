"""The Ollama backend: a tier B embedder whose model runs on another host.

Tier B is "the floor rather than the norm" (:class:`~manicule.core.protocols.Embedder`), and
this is what the floor buys and what it costs. What it buys is a manicule that embeds on a
machine it is not running on — the case it was written for is a pod on Ivy Bridge Xeons with no
AVX2, beside a GPU node already running Ollama. What it costs is the guarantee
:mod:`manicule.embedding.base` exists to provide: the server pools, so the reduction actually
applied cannot be verified by inspection.

So the measurement story is the whole of this file's defense, and it is deliberately larger
than an in-process backend's would be. Six properties are checked against the live server
before it is usable, and each corresponds to a failure that is otherwise silent:

* **the vocabulary** — this backend's token counts are compared with the server's own
  ``prompt_eval_count``, one text at a time. A configured tokenizer is a claim, and an
  unchecked claim in ``tokenizer_id`` would move every chunk boundary in the corpus.
* **the limit** — a probe of exactly ``max_sequence_length`` content tokens must be accepted.
  A server serving less than the fingerprint claims truncates without saying so.
* **the refusal** — a probe past the served context must be *refused*. That is what proves
  ``truncate: false`` is in force, and it is the one check that fails loudly on a server that
  ignores the flag.
* **the width** — every returned vector is the width the fingerprint declares, which is what
  the vector table was created from.
* **normalization** — the server's vectors are already unit length, verified rather than
  assumed, and then normalized exactly so that ``normalized=True`` is true rather than nearly.
* **finiteness** — a ``NaN`` has the right shape and serializes cleanly, and cosine distance
  against it is undefined.

What is *not* checked, and cannot be, is that the pooling is the one the GGUF declares. That is
the admission tier B makes, and the honest place to record it is here rather than in a comment
saying it is fine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final, override

from manicule.core.content import Chunk
from manicule.core.embedding import (
    EmbedFingerprint,
    Vector,
    is_finite_vector,
    require_within_context,
)
from manicule.core.errors import ConfigError, ContextOverflowError, ManiculeError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.lifecycle import HealthReport, Lifecycle, Metric
from manicule.embedding.cache import EmbeddingCache
from manicule.embedding.cards import ModelCard
from manicule.embedding.runtimes.tokenization import FastTokenizer
from manicule_ollama.client import (
    BACKEND,
    EmbedResult,
    OllamaClient,
    OllamaContextOverflowError,
    OllamaUnavailableError,
)
from manicule_ollama.served import ServedModel, ceiling_verified, mark_ceiling_verified

NORM_TOLERANCE: Final = 1e-4
"""How far a returned vector may sit from unit length before the server is refused.

Measured: ``qwen3-embedding:0.6b`` answered 0.9999999 and ``nomic-embed-text`` 1.00000008, which
is float32 noise around exactly one. The gate sits about a thousand times looser than the
measurement, which leaves room for a different build without leaving room for a model that does
not normalize at all — and an unnormalized model matters, because every cosine score manicule
computes assumes it.
"""

PROBE_UNITS: Final[tuple[str, ...]] = ("word", "alpha", "the", "a", "x")
"""Candidate fillers for a length probe; the first that is one token in this vocabulary wins.

Several, because "one common word is one token" is true of most vocabularies and not all of
them, and a probe built from a two-token filler measures half the boundary it claims to.
``alpha`` is in the list for manicule's own synthetic test vocabulary, which knows ten words.
"""

_PROBE_CORRECTIONS: Final = 8
"""How many times a probe's length may be adjusted before the vocabulary is called irregular."""

TOKENIZER_PROBES: Final[tuple[str, ...]] = (
    "",
    "a",
    "the retention window is ninety days, after which archived pages are purged",
    "El gato se sienta en la alfombra y mira por la ventana durante horas.",
    "def embed(texts): return [pool(model(text)) for text in texts]",
    "「こんにちは世界」 — mixed script, an em dash, and an emoji 🎉",
)
"""What the configured vocabulary is checked on.

Not decoration. The empty string measures the special tokens the model wraps every input in,
which is the term that turns a context length into a content budget. The rest are where two
vocabularies of the same family diverge: a non-Latin script, source code, and text that mixes
CJK with astral-plane codepoints. A check on English prose alone passes for tokenizers that
disagree by 30% on exactly the documents a corpus is hardest on.
"""


class OllamaEmbedder(Lifecycle):
    """Embeds by asking an Ollama server, and holds itself to what it can measure of it."""

    def __init__(
        self,
        served: ServedModel,
        client: OllamaClient,
        *,
        cache_dir: Path | None = None,
        keep_alive: str = "5m",
        batch_size: int = 32,
        cache_entries: int = 10_000,
    ) -> None:
        """Args:
        served: The measured model, whose fingerprint already exists — see
            :func:`manicule_ollama.served.resolve` for why that happens at construction.
        client: The connection to the server that measured it.
        cache_dir: Where :func:`manicule_ollama.served.record` wrote this model's declaration.
            ``None`` means the full-context probe runs on every setup rather than once per
            configuration — correct, and forty seconds slower on a 32768-token model.
        keep_alive: How long the server keeps the model resident between requests.
        batch_size: Texts per ``/api/embed`` request.
        cache_entries: Vectors memoized, keyed on the canonical fingerprint.
        """
        self.card: ModelCard = served.card
        """Read by :func:`manicule.ingest.workers.worker_config`, which hands ``card.path``'s
        ``tokenizer.json`` to isolated parse workers. A worker without it chunks with the
        provisional counter, and ingest refuses provisional chunks — so this is not an
        incidental attribute but the reason a served backend needs a local vocabulary at all."""

        self.fingerprint: EmbedFingerprint = served.fingerprint
        self.backend: Final = BACKEND
        self._served = served
        self._client = client
        self._cache_dir = cache_dir
        self._keep_alive = keep_alive
        self._batch_size = batch_size
        self._cache = EmbeddingCache(cache_entries)
        self._tokenizer = FastTokenizer(served.card.path / "tokenizer.json")
        self._embedded = 0
        self._requests = 0
        self._server_tokens = 0
        self._verified = False
        self._closed = False
        self._superseded: str | None = None
        """The digest the server answered with, once it stopped being the one in the
        fingerprint. Set by :meth:`health`, and refused by :meth:`embed` from then on."""

    # --- the embedder protocol -----------------------------------------------------------

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """One vector per text, in order.

        Callers embedding *stored chunks* use :meth:`embed_chunks` instead, which adds the
        context check that re-embed has no other guard for.
        """
        if not texts:
            return []
        self._require_the_model_has_not_moved()

        slots, pending = self._cache.lookup(self.fingerprint, texts)
        resolved: dict[str, Vector] = {}
        if pending:
            computed = await self._embed_uncached(pending)
            resolved = dict(zip(pending, computed, strict=True))
            for text, vector in resolved.items():
                self._cache.put(self.fingerprint, text, vector)

        return [
            slot if slot is not None else resolved[text]
            for slot, text in zip(slots, texts, strict=True)
        ]

    async def embed_chunks(
        self, chunks: Sequence[Chunk], chunk_fingerprint: ChunkFingerprint | None = None
    ) -> list[Vector]:
        """Embed stored chunks, refusing any the model cannot read in full.

        **The budget guard, not the ingest path.** Every route that embeds stored chunks goes
        through :func:`manicule.ingest.embedding.embed_chunks`, which makes the same checks and
        then applies the document half of
        :attr:`~manicule.core.embedding.EmbedFingerprint.prefix_scheme` — which this method
        cannot, because a backend has no way to tell a document from a query. So this is what
        :func:`manicule.testing.assert_refuses_oversized_chunks` holds this backend to, and not
        a shortcut into an index.

        The check exists because re-embedding reads stored ``embed_text`` without re-chunking,
        so the chunker's budget refusal never runs; and a sequence limit that *fell* — which on
        this backend is one ``num_ctx`` away, or one ``ollama pull`` of a model with a shorter
        context — leaves the embedding fingerprint identical, so no comparison fires either.
        """
        measured = [
            chunk.model_copy(update={"token_count": self.count_tokens(chunk.embed_text)})
            for chunk in chunks
        ]
        require_within_context(chunks, self.fingerprint)
        require_within_context(measured, self.fingerprint, chunk_fingerprint)
        return await self.embed([chunk.embed_text for chunk in chunks])

    def count_tokens(self, text: str) -> int:
        """Content tokens, the way the served model will count them.

        Special tokens are excluded, because
        :attr:`~manicule.core.embedding.EmbedFingerprint.max_sequence_length` is usable content
        tokens: the two numbers are compared to each other, so they have to measure the same
        thing. That this local count really is the server's is not assumed — see
        :meth:`setup`.
        """
        return len(self._tokenizer.content_ids(text))

    # --- lifecycle ------------------------------------------------------------------------

    @override
    async def setup(self) -> None:
        """Check every claim this backend makes against the server, before anything uses it.

        Runs in the order the failures matter. The vocabulary first, because a wrong one makes
        every later number meaningless; then the limit, because it is what the chunker has
        already been bound to; then the refusal, which is what stands between an over-long
        chunk and a vector describing its opening.
        """
        await self._verify_tokenizer()
        await self._verify_context_limit()
        self._verified = True

    @override
    async def teardown(self) -> None:
        """Release the connection pool. Safe after a failed setup, and safe twice."""
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()

    @override
    async def health(self) -> HealthReport:
        """Whether the server is reachable **and still serving the bytes this index was built on**.

        The second half is what an in-process backend does not need. ``ollama pull`` on the
        server replaces a model's blob under an unchanged name, and every vector written
        afterwards comes from a different space — with nothing in manicule's own state changed
        to notice it. The digest is in the fingerprint, so comparing it here turns that into a
        failing health check rather than a corpus that quietly stops agreeing with itself.
        """
        if self._closed:
            return HealthReport.failing(
                f"{self.fingerprint.describe()} has been torn down",
                remedy="Build the embedder through the container, which manages its lifecycle.",
            )
        try:
            digest = await self._client.digest(self._served.info.model)
        except ManiculeError as exc:
            # Every error this package raises, rather than the two it usually raises here.
            # `SupportsHealth` says a health check reports instead of raising, and the caller
            # is a diagnostic asking every component at once — so one that escapes takes down
            # the surface that was about to say which component is unwell.
            return HealthReport.failing(
                f"{self.fingerprint.describe()} on {self._client.base_url}: {exc}",
                remedy=f"Check that the server answers `curl {self._client.base_url}/api/tags` "
                f"and holds {self._served.info.model!r}.",
            )
        if digest != self._served.info.digest:
            # **Recorded, not merely reported.** Ingest and retrieval call `embed` directly and
            # never consult a health report, so a check that only said so would watch vectors
            # from the new model being appended to the old model's index between sweeps. Once
            # this is known, embedding stops.
            self._superseded = digest
            return HealthReport.failing(
                f"{self._served.info.model!r} on {self._client.base_url} now resolves to "
                f"sha256:{digest}, but this index's vectors were made by "
                f"sha256:{self._served.info.digest}. The model was re-pulled under the same "
                f"name; anything embedded from here on would be a different vector space in "
                f"the same table.",
                remedy="Restart manicule to pick up the new model — which will refuse to write "
                "into the existing index and name the re-embed — or restore the previous "
                "model on the server.",
            )
        if not self._verified:
            return HealthReport.failing(
                f"{self.fingerprint.describe()} has not been verified against "
                f"{self._client.base_url}",
                remedy="Start the container, which runs setup().",
            )
        return HealthReport.healthy(
            f"{self.fingerprint.describe()} on {self._client.base_url}, "
            f"served at num_ctx={self._served.num_ctx}"
        )

    @override
    def metrics(self) -> Sequence[Metric]:
        """The shared embedder metrics, plus what the *server* says the work cost.

        ``ollama_prompt_tokens`` is the one to watch, and it is the server's own count rather
        than this backend's. Divided by ``embedding_texts_embedded`` it is the mean tokens per
        text actually evaluated, which is the number that moves when a chunk budget changes —
        and the number that would quietly stop moving if inputs began being truncated.
        """
        labels = {"backend": self.backend, "model": self.fingerprint.model_id}
        return (
            Metric(name="embedding_cache_hits", value=float(self._cache.hits), labels=labels),
            Metric(name="embedding_cache_misses", value=float(self._cache.misses), labels=labels),
            Metric(name="embedding_cache_entries", value=float(len(self._cache)), labels=labels),
            Metric(name="embedding_texts_embedded", value=float(self._embedded), labels=labels),
            Metric(name="ollama_requests", value=float(self._requests), labels=labels),
            Metric(
                name="ollama_prompt_tokens",
                value=float(self._server_tokens),
                unit="tokens",
                labels=labels,
            ),
            Metric(
                name="ollama_num_ctx",
                value=float(self._served.num_ctx),
                unit="tokens",
                labels=labels,
            ),
        )

    def _require_the_model_has_not_moved(self) -> None:
        """Refuse to embed once the server is known to be serving different bytes.

        The digest is in the fingerprint, so vectors made after an ``ollama pull`` belong to a
        different space — and the index has no way to tell them apart, because the fingerprint
        it compares against has not changed. A health check that reported the mismatch and let
        embedding continue would leave the two mixed in one table for as long as the process
        ran.

        Deliberately **not** a per-request check. Asking ``/api/tags`` before every batch would
        double the requests an ingest makes to catch something that happens when a person runs
        a command on the server; what this does is make the first observation final.
        """
        if self._superseded is None:
            return
        msg = (
            f"{self._served.info.model!r} on {self._client.base_url} is now sha256:"
            f"{self._superseded}, and this embedder's vectors were made by sha256:"
            f"{self._served.info.digest}. Embedding stops here rather than appending vectors "
            f"from a different model to an index whose fingerprint says the model did not "
            f"change. Restart manicule to pick up the new one — which will refuse the existing "
            f"index and name the re-embed — or restore the previous model on the server."
        )
        raise ConfigError(msg)

    # --- internals ------------------------------------------------------------------------

    async def _embed_uncached(self, texts: Sequence[str]) -> list[Vector]:
        """Refuse what the model cannot read, then embed the rest one batch at a time."""
        self._require_within_limit(texts)
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            result = await self._request(batch)
            if len(result.vectors) != len(batch):
                msg = (
                    f"{self._client.base_url} returned {len(result.vectors)} vectors for "
                    f"{len(batch)} inputs. They are matched positionally, so a short answer "
                    f"cannot be realigned — it would attach each vector to the wrong text."
                )
                raise OllamaUnavailableError(msg)
            vectors.extend(
                self._finish(vector, index) for index, vector in enumerate(result.vectors)
            )
        self._embedded += len(texts)
        return vectors

    async def _request(self, batch: Sequence[str]) -> EmbedResult:
        """One request, with the server's context refusal translated into manicule's."""
        try:
            result = await self._client.embed(
                self._served.info.model,
                batch,
                num_ctx=self._served.num_ctx,
                keep_alive=self._keep_alive,
            )
        except OllamaContextOverflowError as exc:
            # Reachable only if this backend's own tokenizer disagreed with the server about a
            # text `_require_within_limit` had just cleared. That is the failure the setup
            # check exists to prevent, so arriving here means it has started being wrong since.
            counts = [self.count_tokens(text) for text in batch]
            msg = (
                f"{self._client.base_url} refused a batch as longer than the "
                f"{self._served.num_ctx}-token context it serves, although this backend "
                f"measured every text in it at or under "
                f"{self.fingerprint.max_sequence_length} content tokens (largest: "
                f"{max(counts, default=0)}). The configured tokenizer and the server no longer "
                f"agree, so "
                f"chunk boundaries measured with it are not measurements of anything this "
                f"model reads. Server said: {exc}"
            )
            raise ContextOverflowError(msg) from exc
        self._requests += 1
        if result.prompt_eval_count > 0:
            self._server_tokens += result.prompt_eval_count
        return result

    def _require_within_limit(self, texts: Sequence[str]) -> None:
        """Refuse over-long text here, before the server is asked.

        **Belt and braces, and both are load-bearing.** This check is the one that can name the
        offending text, because it is the only place that still has the batch. The server-side
        ``truncate: false`` is what catches this check being wrong — a tokenizer that has
        drifted from the model — and it is the difference between a refusal and a corpus of
        vectors describing opening fragments. Neither replaces the other.
        """
        # The card's `input_capacity` rather than the fingerprint's `max_sequence_length`: the
        # strings here already carry whichever half of the prefix scheme applies, and
        # `max_sequence_length` is the budget left for a chunk once the document half has been
        # charged. Comparing the prefixed text against it would charge the prefix twice.
        limit = self.card.input_capacity
        lengths = [self.count_tokens(text) for text in texts]
        oversized = [(index, length) for index, length in enumerate(lengths) if length > limit]
        if not oversized:
            return
        worst = sorted(oversized, key=lambda pair: pair[1], reverse=True)[:3]
        listed = ", ".join(f"text {index} ({count} tokens)" for index, count in worst)
        msg = (
            f"{len(oversized)} of {len(lengths)} texts exceed the {limit}-token limit of "
            f"{self.fingerprint.describe()}: {listed}. The request is sent with "
            f"truncate=false so the server would refuse it too, but it is refused here so the "
            f"message can say which text to shorten. Shorten it, or chunk it first."
        )
        raise ContextOverflowError(msg)

    def _finish(self, vector: Vector, index: int) -> Vector:
        """Check a returned vector, and normalize it exactly.

        The check is the tier B substitute for having produced the vector ourselves: width,
        because the vector table was created from the fingerprint's number; and finiteness,
        because a ``NaN`` has the right shape, serializes cleanly, and makes cosine distance
        undefined wherever it lands.

        The normalization is not a correction. :meth:`setup` has already refused a server whose
        vectors are not unit length, so what happens here is the last bit of float32 noise being
        removed from something already normalized — which is what makes
        ``EmbedFingerprint.normalized`` exactly true rather than nearly true. Silently
        normalizing an *unnormalized* model would be the other thing entirely, and the setup
        check is what stops this line from being that.
        """
        expected = self.fingerprint.dimension
        if len(vector) != expected:
            msg = (
                f"vector {index} came back with {len(vector)} dimensions where "
                f"{self.fingerprint.describe()} declares {expected}. The fingerprint is what "
                f"the index was built against, so a disagreement here corrupts every later "
                f"search."
            )
            raise OllamaUnavailableError(msg)
        if not is_finite_vector(vector):
            msg = (
                f"vector {index} from {self._client.base_url} contains a non-finite component. "
                f"It would serialize cleanly and make every cosine distance against it "
                f"undefined, so it is refused rather than stored."
            )
            raise OllamaUnavailableError(msg)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            msg = (
                f"vector {index} from {self._client.base_url} is all zeros, which has no "
                f"direction and therefore no cosine similarity to anything."
            )
            raise OllamaUnavailableError(msg)
        return [value / norm for value in vector]

    # --- the setup-time measurements ------------------------------------------------------

    async def _verify_tokenizer(self) -> None:
        """Check the configured vocabulary against the server's own token count.

        One probe per request, deliberately: ``prompt_eval_count`` is a total across the whole
        request, so a batch would pass while two inputs were wrong in opposite directions.

        This is what turns ``tokenizer`` from an inference into a measurement, and it is the
        reason this backend is allowed to name a Hugging Face repository for a model it is
        running out of a GGUF at all. A single token of disagreement on a single probe is a
        refusal, because the number it moves is the chunk budget, and a budget measured with
        the wrong vocabulary undercounts — which is the direction that truncates.
        """
        specials = self.card.special_token_count
        for probe in TOKENIZER_PROBES:
            result = await self._request([probe])
            # **Here rather than only in the context check**, because that one returns early
            # when the ceiling is already recorded — so on every start after the first, nothing
            # would have inspected a vector before ingest did, and `_finish` normalizes rather
            # than refuses. These probes are being embedded anyway; the check is free.
            self._require_unit_norm(result.vectors)
            expected = self.count_tokens(probe) + specials
            if result.prompt_eval_count < 0:
                msg = (
                    f"{self._client.base_url} returned no prompt_eval_count, so the configured "
                    f"tokenizer cannot be checked against the model it claims to describe. "
                    f"This backend will not count tokens with an unverified vocabulary: the "
                    f"count is what decides chunk boundaries and what refuses text the model "
                    f"would truncate."
                )
                raise ConfigError(msg)
            if result.prompt_eval_count != expected:
                msg = (
                    f"the configured tokenizer ({self.fingerprint.tokenizer_id}) makes "
                    f"{expected} tokens of {probe!r} where {self._served.info.model!r} on "
                    f"{self._client.base_url} read {result.prompt_eval_count}. They are not "
                    f"the same vocabulary. Chunk boundaries are placed with this tokenizer and "
                    f"checked against that model's limit, so the two have to agree exactly — "
                    f"set `tokenizer` to the repository the served GGUF was converted from."
                )
                raise ConfigError(msg)

    async def _verify_context_limit(self) -> None:
        """Check the derived limit against the server, in both directions.

        Two probes, and they answer different questions.

        The **accepted** one sends exactly ``max_sequence_length`` content tokens. It fails when
        the server reads less than the fingerprint claims — a ``num_ctx`` that did not take, a
        model whose GGUF overstates its context — which is the direction that truncates, and it
        fails here rather than in the middle of an ingest run that has already written vectors.

        The **refused** one sends a text past the served context. It fails when the server
        answers anyway, which means ``truncate: false`` is not in force — an older Ollama, a
        proxy rewriting the body — and that is the single assumption this whole backend rests
        on. Without it every over-long input becomes a well-formed vector describing an opening
        fragment, and nothing anywhere raises.
        """
        await self._verify_the_limit_is_reachable()

        over = self._text_of_length(self._served.num_ctx + 1)
        try:
            await self._client.embed(
                self._served.info.model,
                [over],
                num_ctx=self._served.num_ctx,
                keep_alive=self._keep_alive,
            )
        except OllamaContextOverflowError:
            return
        msg = (
            f"{self._client.base_url} accepted a probe of {self._served.num_ctx + 1} content "
            f"tokens against a {self._served.num_ctx}-token context instead of refusing it, so "
            f"`truncate: false` is not being honored. Everything past the context is then "
            f"dropped with no error and the vector describes only the opening — which is the "
            f"one failure this backend has no second guard for. Upgrade the Ollama server, or "
            f"check for a proxy between manicule and it that rewrites request bodies."
        )
        raise ConfigError(msg)

    async def _verify_the_limit_is_reachable(self) -> None:
        """Embed exactly ``max_sequence_length`` content tokens, once per configuration.

        **The expensive half, and the one that is remembered.** It is a full-context forward
        pass — 40 seconds for ``qwen3-embedding:0.6b`` at 32768, measured, against 0.04 for the
        refusal below — and what it establishes does not change while the digest, the served
        context and the derived limit all stay the same. So the result is recorded beside the
        declaration it belongs to, and any change to any of those three rewrites that record
        and brings the probe back.

        What it establishes is *when* a mismatch is noticed rather than whether. A server
        reading less than the fingerprint claims fails loudly either way, because every request
        carries ``truncate: false`` — but failing here is failing before a corpus is built,
        and failing later is failing partway through building one.
        """
        model = self._served.info.model
        # The *configured* name is the cache key — see `ServedModel.configured_name`. Reading
        # it under the canonical one would miss every record written for a bare name, and the
        # expensive probe would run on every start while looking as though it were cached.
        key = self._served.configured_name
        if self._cache_dir is not None and ceiling_verified(
            self._cache_dir, self._client.base_url, key
        ):
            return

        limit = self.card.input_capacity
        try:
            result = await self._request([self._text_of_length(limit)])
        except ContextOverflowError as exc:
            msg = (
                f"{model!r} on {self._client.base_url} refused a probe of exactly {limit} "
                f"content tokens, which is the limit {self.fingerprint.describe()} advertises "
                f"and the number the chunker has already been bound to. The server is reading "
                f"less than this backend derived from num_ctx={self._served.num_ctx}. Lower "
                f"`num_ctx`, or `max_sequence_length`, so the advertised limit is one the "
                f"server will honor. ({exc})"
            )
            raise ConfigError(msg) from exc
        self._require_unit_norm(result.vectors)
        if self._cache_dir is not None:
            mark_ceiling_verified(self._cache_dir, self._client.base_url, key)

    def _text_of_length(self, tokens: int) -> str:
        """A string this tokenizer measures at exactly ``tokens`` content tokens.

        Exactly, because both probes measure a boundary: one token out and the check is about
        the position next to the one it means to test. And built from a unit this vocabulary is
        confirmed to make one token of, rather than from a word assumed to be one — ``"word "``
        with its trailing space is two tokens to Qwen's vocabulary and one to a WordPiece, which
        is the sort of thing that makes a probe quietly wrong at the far end of a 32768-token
        context.

        Corrected in **whole units**, not characters. Trimming one character at a time
        re-tokenizes the entire probe on each pass, which on a 32k-token string is a measurable
        fraction of a second repeated thousands of times; adjusting by the exact shortfall
        converges in one or two passes on any vocabulary whose count is near-linear in its
        input, and refuses rather than spins on one that is not.
        """
        unit = self._probe_unit()
        count = tokens
        words = tokens
        for _ in range(_PROBE_CORRECTIONS):
            text = " ".join([unit] * words)
            count = self.count_tokens(text)
            if count == tokens:
                return text
            adjusted = words + (tokens - count)
            if adjusted < 1:
                break
            words = adjusted
        msg = (
            f"could not build a probe of exactly {tokens} tokens with "
            f"{self.fingerprint.tokenizer_id}: {_PROBE_CORRECTIONS} adjustments left it at "
            f"{count}. The context boundary cannot be measured with a probe of unknown length, "
            f"so it is refused rather than approximated."
        )
        raise ConfigError(msg)

    def _probe_unit(self) -> str:
        """A word this vocabulary makes exactly one token of, found rather than assumed."""
        for candidate in PROBE_UNITS:
            if self.count_tokens(candidate) == 1:
                return candidate
        msg = (
            f"none of {PROBE_UNITS} is a single token in "
            f"{self.fingerprint.tokenizer_id}, so this backend cannot build a probe of a known "
            f"length to measure the served context with."
        )
        raise ConfigError(msg)

    def _require_unit_norm(self, vectors: Sequence[Vector]) -> None:
        """Refuse a server whose vectors are not already normalized.

        ``EmbedFingerprint.normalized`` is recorded as ``True`` for this backend, and every
        cosine score manicule computes assumes it. On a tier A backend that is made true by
        :mod:`manicule.embedding.pooling`; here it is the server's doing, so it is checked
        instead of claimed. Measured: 0.9999999 and 1.00000008 on the two models this was
        written against, which is float32 noise either side of one.
        """
        for index, vector in enumerate(vectors):
            # Finiteness first, and not merely for tidiness: `abs(nan - 1.0) > tolerance` is
            # **false**, so a probe full of NaN would pass the comparison below and this
            # backend would record `normalized=True` about a vector that has no length at all.
            # `_finish` catches it on a later real embedding; the point of this check is to
            # catch it before the fingerprint is trusted.
            if not is_finite_vector(vector):
                msg = (
                    f"{self._served.info.model!r} on {self._client.base_url} returned a "
                    f"non-finite component in probe {index}. Nothing about normalization can "
                    f"be established from it, and a NaN compares false against every "
                    f"tolerance rather than failing one."
                )
                raise ConfigError(msg)
            norm = math.sqrt(sum(value * value for value in vector))
            if abs(norm - 1.0) > NORM_TOLERANCE:
                msg = (
                    f"{self._served.info.model!r} on {self._client.base_url} returned a vector "
                    f"of length {norm:.6f} (probe {index}), not 1. This backend records "
                    f"normalized=True and every cosine score manicule computes assumes it. "
                    f"Normalizing an unnormalized model here would hide the disagreement "
                    f"rather than resolve it, so it is refused."
                )
                raise ConfigError(msg)


__all__ = ["NORM_TOLERANCE", "PROBE_UNITS", "TOKENIZER_PROBES", "OllamaEmbedder"]
