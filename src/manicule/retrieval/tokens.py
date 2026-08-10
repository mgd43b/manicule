"""Counting the tokens the *generator* will charge for.

There are two token budgets in manicule and they are measured with different tokenizers, for
different models, to protect against different failures:

===================  ===================  ==================================================
Budget               Tokenizer            Protects against
===================  ===================  ==================================================
Chunk size, 512      the **embedder's**   silent truncation inside the embedder, producing a
                                          vector claiming text it never saw
Context window       the **generator's**  an assembled context larger than the window, which
                                          the server truncates from the front
===================  ===================  ==================================================

``Chunk.token_count`` is the first of those, and using it for the second is the category error
this module exists to prevent. It is measured in the embedder's SentencePiece units, for a
model that is not generating anything; it is sitting on every candidate, it is a plausible
number, and it is wrong for this purpose by an unknown factor.

``tiktoken`` is a stand-in for the second and is named as one. The generator is Ollama-hosted
and runs a Llama, Qwen or Mistral vocabulary — none of them tiktoken's — so this is an
estimate. Four rules keep an estimate honest:

* **By encoding name, never by model name.** ``encoding_for_model("gpt-4o")`` would make an
  estimate look authoritative about a model that is not being used.
* **A safety factor, in the direction that matters.** Undercounting overflows the window and
  the prompt is truncated silently; overcounting costs a passage.
* **Never sample and extrapolate.** The fitter's entire job is not to overflow, and
  extrapolation turns a bounded error into an unbounded one on precisely the longest inputs.
* **Then stop guessing.** Ollama's response carries ``prompt_eval_count`` — the true count,
  from the model that counts — and :meth:`ContextTokenCounter.observe` compares the estimate
  against it. Measuring once beats a safety factor forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from manicule.core.content import Chunk

DEFAULT_ENCODING: Final = "o200k_base"
DEFAULT_SAFETY_FACTOR: Final = 1.2
DEFAULT_DRIFT_TOLERANCE: Final = 0.15

MAX_CACHED_COUNTS: Final = 50_000
"""How many per-chunk counts to remember.

The counts themselves can never go stale — a chunk id is derived from its content, so a
changed chunk is a different key — but a process serving a large corpus for weeks would
otherwise accumulate one entry per chunk it has ever cited. The cap is on memory, not on
correctness: an evicted entry is recomputed and gives the same answer.
"""


class ContextTokenDriftError(ManiculeError):
    """The estimate was materially below what the generator actually charged.

    Fatal, and deliberately asymmetric. Overcounting costs a passage and is visible in the
    trace; **undercounting overflows the window**, and a server that truncates a prompt from
    the front discards the system prompt and the citation protocol, then presents as a model
    that does not follow instructions. The whole point of comparing against the generator's own
    count is to catch that before it is diagnosed as a prompting problem.
    """


class ContextTokenCounter:
    """An estimate of what a generator will charge for a piece of text, biased upward.

    Not a tokenizer for any particular model, and it does not claim to be one. What it is:
    a real BPE implementation rather than a character heuristic, inflated by a stated factor,
    identified by encoding name in every trace, and checkable against the generator's own count
    the moment one is available.
    """

    def __init__(
        self,
        *,
        encoding: str = DEFAULT_ENCODING,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        drift_tolerance: float = DEFAULT_DRIFT_TOLERANCE,
    ) -> None:
        if safety_factor < 1.0:
            msg = (
                f"safety_factor {safety_factor} is below 1.0, which biases the estimate "
                f"downward — the one direction that overflows the generator's window and gets "
                f"the prompt truncated with nothing raised."
            )
            raise ValueError(msg)
        self.encoding_name = encoding
        self.safety_factor = safety_factor
        self.drift_tolerance = drift_tolerance
        self.drift: dict[str, float] = {}
        self._encoding: Any = None
        self._counts: dict[str, int] = {}

    @property
    def identity(self) -> str:
        """What the trace records. Names the encoding and the factor, never a model."""
        return f"tiktoken:{self.encoding_name}x{self.safety_factor:g}"

    def count(self, text: str) -> int:
        """Estimated generator tokens for ``text``, rounded up after the safety factor."""
        if not text:
            return 0
        encoded = self._encode(text)
        return int(encoded * self.safety_factor) + 1

    def count_chunk(self, chunk: Chunk) -> int:
        """Estimated generator tokens for a chunk's citable text, memoised by chunk id.

        ``chunk.text`` rather than ``chunk.embed_text``: the breadcrumb exists to make a
        passage findable and never appears in what is quoted, so counting it would reserve
        budget for text the model will not see.
        """
        cached = self._counts.get(chunk.id)
        if cached is not None:
            return cached
        counted = self.count(chunk.text)
        if len(self._counts) >= MAX_CACHED_COUNTS:
            self._counts.clear()
        self._counts[chunk.id] = counted
        return counted

    def observe(self, model_id: str, estimated: int, actual: int) -> float:
        """Compare an estimate against the generator's own ``prompt_eval_count``.

        Args:
            model_id: The generator that produced ``actual``. Drift is per model, because the
                vocabularies differ.
            estimated: What this counter said before the call.
            actual: What the generator charged.

        Returns:
            The signed relative drift, ``(estimated - actual) / actual``. Positive means this
            counter over-estimated, which is the safe direction.

        Raises:
            ContextTokenDriftError: The estimate was below the true count by more than
                :attr:`drift_tolerance`. Raised in that direction only: an over-estimate wastes
                budget and is recorded, while an under-estimate is the one that truncates.
        """
        if actual <= 0:
            return 0.0
        relative = (estimated - actual) / actual
        self.drift[model_id] = relative
        if relative < -self.drift_tolerance:
            suggested = actual / max(estimated, 1) * self.safety_factor
            msg = (
                f"the context fitter estimated {estimated} tokens for {model_id!r} and the "
                f"model charged {actual} — {abs(relative):.0%} more than budgeted, past a "
                f"tolerance of {self.drift_tolerance:.0%}. An under-estimate overflows the "
                f"window and the prompt is truncated from the front, which discards the system "
                f"prompt and presents as a model that ignores instructions. Raise "
                f"rag.context.safety_factor to at least {suggested:.2f}."
            )
            raise ContextTokenDriftError(msg)
        return relative

    def _encode(self, text: str) -> int:
        if self._encoding is None:
            # Deferred: tiktoken is a retrieval extra, and importing manicule must not load it.
            import tiktoken  # noqa: PLC0415

            self._encoding = tiktoken.get_encoding(self.encoding_name)
        tokens: Any = self._encoding.encode(text)
        return len(tokens)


__all__ = [
    "DEFAULT_DRIFT_TOLERANCE",
    "DEFAULT_ENCODING",
    "DEFAULT_SAFETY_FACTOR",
    "MAX_CACHED_COUNTS",
    "ContextTokenCounter",
    "ContextTokenDriftError",
]
