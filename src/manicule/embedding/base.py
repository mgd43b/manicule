"""The embedding path both backends share: tokenize, read token states, pool here.

The division of labour is deliberate. A backend contributes exactly one thing — a forward
pass returning per-token hidden states — and everything that decides what a vector *means*
happens in this module and in :mod:`manicule.embedding.pooling`, once, for every runtime:

* tokenizing, because the backends' own embedding calls do not surface attention masks, and a
  mask-free mean pool averages in the padding;
* refusing over-long input, because past the sequence limit a model drops the remainder and
  raises nothing;
* the reduction, because the field a backend calls its embedding is not reliably the
  reduction the model was trained with;
* L2 normalisation, because a repository can omit its ``Normalize`` step while still
  publishing cosine scores that assume it.

What is left to a backend is throughput. That is the Apple-hardware principle expressed as a
class boundary: the platform gets the forward pass, and cannot reach the output.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, override

import numpy as np

from manicule.core.content import Chunk
from manicule.core.embedding import (
    EmbedFingerprint,
    TokenStates,
    Vector,
    require_within_context,
)
from manicule.core.errors import ContextOverflowError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.lifecycle import HealthReport, Lifecycle, Metric
from manicule.embedding.cache import EmbeddingCache
from manicule.embedding.cards import ModelCard, load_tokenizer
from manicule.embedding.pooling import pool_token_states, require_token_states

if TYPE_CHECKING:
    from manicule.embedding.runtimes.tokenization import FastTokenizer


class PooledEmbedder(Lifecycle, ABC):
    """A tier A embedder: exposes token states, and pools them in manicule's own numpy.

    Subclasses implement :meth:`_forward` and nothing else about what a vector is.
    """

    def __init__(
        self,
        card: ModelCard,
        *,
        backend: str,
        weights_ref: str,
        batch_size: int = 32,
        cache_entries: int = 10_000,
    ) -> None:
        """Build the identity and the tokenizer; leave the weights to :meth:`setup`.

        The fingerprint has to exist before setup runs: the chunker resolves the embedder as a
        construction dependency and refuses to start when its budget exceeds this model's
        sequence limit, and that refusal has to happen before ingest rather than after. So
        construction reads a few kilobytes of declaration and a tokenizer, and the gigabytes
        wait.
        """
        self.card = card
        self.fingerprint: EmbedFingerprint = card.fingerprint(
            backend=backend, weights_ref=weights_ref
        )
        self.backend = backend
        self._batch_size = batch_size
        self._cache = EmbeddingCache(cache_entries)
        self._tokenizer: FastTokenizer = load_tokenizer(card)
        self._embedded = 0
        # One thread, and always the same one. MLX's streams are thread-local: weights loaded
        # on one thread and evaluated on another abort the process outright — `libc++abi:
        # terminating ... There is no Stream(gpu, N) in current thread`, not an exception
        # anything can catch. `asyncio.to_thread` hands out whichever pool thread is free, so
        # it is exactly the wrong tool here. A single worker also serialises forward passes,
        # which is what one accelerator wants anyway.
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"manicule-embed-{backend}"
        )
        self._closed = False

    # --- what a backend supplies ---------------------------------------------------------

    @abstractmethod
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Run the encoder and return per-token hidden states, unpooled and unnormalised.

        **Materialised into numpy here, on the worker thread, before returning.** MLX is lazy:
        a forward pass hands back an unevaluated graph bound to a stream that belongs to the
        thread that built it, and MLX's streams are thread-local. Converting it anywhere else —
        including in the pooling code one call up — evaluates it on a thread where its stream
        does not exist and aborts the process: ``libc++abi: terminating ... There is no
        Stream(gpu, N) in current thread``. That is a hard abort, not an exception, so nothing
        catches it and nothing reports it. Returning a materialised array makes the boundary
        the type.

        Rank is checked by the caller: a backend handing back an already-pooled vector is an
        error rather than something to detect by shape and adapt to.
        """

    @abstractmethod
    def _load(self) -> None:
        """Download, vet and load the weights. **Runs on the worker thread**, always.

        Synchronous by design: it is called through :meth:`_on_worker`, so that the thread
        holding the model is the thread that built it.
        """

    @abstractmethod
    def _unload(self) -> None:
        """Release the weights. Also on the worker thread, and safe to call twice."""

    @abstractmethod
    def _loaded(self) -> bool:
        """Whether the weights are in memory. Read by :meth:`health`."""

    # --- the embedder protocol -----------------------------------------------------------

    async def encode(self, texts: Sequence[str]) -> TokenStates:
        """Per-token hidden states and the attention mask for ``texts``.

        One padded batch, so every row shares a sequence axis and the result can be pooled in
        one operation. Large inputs are correspondingly large in memory — ``(n, length,
        dimension)`` floats — which is inherent to what tier A returns; :meth:`embed` pools
        batch by batch and never holds more than one.
        """
        if not texts:
            return TokenStates(
                states=np.zeros((0, 0, self.fingerprint.dimension), dtype=np.float32),
                attention_mask=np.zeros((0, 0), dtype=np.int64),
                dimension=self.fingerprint.dimension,
            )
        ids, mask = self._tokenize(texts)
        blocks: list[np.ndarray] = []
        for start in range(0, len(ids), self._batch_size):
            stop = start + self._batch_size
            states = await self._run(ids[start:stop], mask[start:stop])
            require_token_states(states, backend=self.backend, model_id=self.fingerprint.model_id)
            blocks.append(states)
        return TokenStates(
            states=np.concatenate(blocks, axis=0),
            attention_mask=mask,
            dimension=self.fingerprint.dimension,
        )

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """One vector per text, in order, from the model's own pooling.

        Callers embedding *stored chunks* use :meth:`embed_chunks` instead, which adds the
        context check that re-embed has no other guard for.
        """
        if not texts:
            return []

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

        **This is the path re-embed uses**, and the reason the check lives here rather than in
        the chunker alone. Re-embedding reads stored ``embed_text`` without re-chunking, so the
        chunker's budget refusal never runs; and a sequence limit that *fell* leaves the
        embedding fingerprint identical, so no comparison fires either. Everything past the
        limit would be dropped in silence, and each stored vector would describe an opening
        fragment while its chunk still claimed all of its text.
        """
        require_within_context(chunks, self.fingerprint, chunk_fingerprint)
        return await self.embed([chunk.embed_text for chunk in chunks])

    def count_tokens(self, text: str) -> int:
        """Content tokens, the way this model will count them.

        Special tokens are excluded, because
        :attr:`~manicule.core.embedding.EmbedFingerprint.max_sequence_length` is usable content
        tokens: the two numbers are compared to each other, so they have to measure the same
        thing.
        """
        return len(self._tokenizer.content_ids(text))

    # --- lifecycle ------------------------------------------------------------------------

    @override
    async def setup(self) -> None:
        """Load the weights, on the worker thread that will use them."""
        await self._on_worker(self._load)

    @override
    async def teardown(self) -> None:
        """Release the weights and stop the worker.

        Safe after a failed setup, which is when it is most needed, and safe twice: the second
        call has no worker left to submit to and says so by doing nothing.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await self._on_worker(self._unload)
        finally:
            self._worker.shutdown(wait=True)

    @override
    async def health(self) -> HealthReport:
        if not self._loaded():
            return HealthReport.failing(
                f"{self.fingerprint.describe()} is not loaded on the {self.backend} backend",
                remedy="Start the container, which runs setup(), or check the model download.",
            )
        return HealthReport.healthy(f"{self.fingerprint.describe()} on {self.backend}")

    @override
    def metrics(self) -> Sequence[Metric]:
        labels = {"backend": self.backend, "model": self.fingerprint.model_id}
        return (
            Metric(name="embedding_cache_hits", value=float(self._cache.hits), labels=labels),
            Metric(name="embedding_cache_misses", value=float(self._cache.misses), labels=labels),
            Metric(name="embedding_cache_entries", value=float(len(self._cache)), labels=labels),
            Metric(name="embedding_texts_embedded", value=float(self._embedded), labels=labels),
        )

    # --- internals ------------------------------------------------------------------------

    async def _embed_uncached(self, texts: Sequence[str]) -> list[Vector]:
        """Tokenize, forward and pool, one batch at a time.

        Batches are pooled as they are produced rather than accumulated, so peak memory is one
        batch of token states however long the input list is.
        """
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            ids, mask = self._tokenize(batch)
            states = await self._run(ids, mask)
            vectors.extend(
                pool_token_states(
                    TokenStates(
                        states=states, attention_mask=mask, dimension=self.fingerprint.dimension
                    ),
                    self.fingerprint.pooling,
                    backend=self.backend,
                    model_id=self.fingerprint.model_id,
                )
            )
        self._embedded += len(texts)
        return vectors

    async def _on_worker[T](self, work: Callable[[], T]) -> T:
        """Run ``work`` on this embedder's one thread, off the event loop.

        The runtimes are synchronous and compute-bound, and an ingest run embeds while an HTTP
        request is in flight; a thread is the difference between a slow index and a frozen one.
        That it is always the *same* thread is not a detail — see the constructor.
        """
        return await asyncio.get_running_loop().run_in_executor(self._worker, work)

    async def _run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """A forward pass, on the worker thread."""
        return await self._on_worker(lambda: self._forward(input_ids, attention_mask))

    def _tokenize(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Token ids and attention mask, refusing anything the model would truncate.

        Truncation is not enabled on the tokenizer, so an over-long input arrives here at full
        length and is refused by name. A tokenizer configured to truncate would instead hand
        back a well-formed shortened sequence, and the vector built from it would describe an
        opening fragment while its caller believed it described the whole text.
        """
        encoded = self._tokenizer.encode_batch(texts)
        limit = self.fingerprint.max_sequence_length
        specials = self.card.special_token_count
        # From the mask, never from ``len(ids)``. The batch is padded to its longest member, so
        # every row's id list is that length — measuring it would report the longest text's size
        # for every text in the batch, and name innocent ones as the offenders. The error would
        # still fire, on the right batch, saying the wrong thing about which text to shorten.
        lengths = [sum(row) - specials for row in encoded.attention_mask]
        oversized = [(index, length) for index, length in enumerate(lengths) if length > limit]
        if oversized:
            worst = sorted(oversized, key=lambda pair: pair[1], reverse=True)[:3]
            listed = ", ".join(f"text {index} ({count} tokens)" for index, count in worst)
            msg = (
                f"{len(oversized)} of {len(lengths)} texts exceed the {limit}-token limit of "
                f"{self.fingerprint.describe()}: {listed}. Truncation is deliberately not "
                f"enabled: the model would drop the remainder without an error and the vector "
                f"would describe only the opening. Shorten the text, or chunk it first."
            )
            raise ContextOverflowError(msg)

        ids = np.array(encoded.ids, dtype=np.int64)
        mask = np.array(encoded.attention_mask, dtype=np.int64)
        return ids, mask


__all__ = ["PooledEmbedder"]
