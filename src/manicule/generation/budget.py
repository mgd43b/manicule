"""Counting tokens for the *generation* model, and the one measurement that replaces it.

There are three tokenizers in play in manicule and only one of them counts here.
:attr:`~manicule.core.content.Chunk.token_count` is measured in the **embedder's**
SentencePiece units for a model that is not generating anything; it is sitting on every
candidate, it is a plausible number, and using it for a generation budget is a category
error that is wrong by an unknown factor varying with language and content type.

``tiktoken`` is a stand-in and is named as one. It is used by *encoding name* rather than
``encoding_for_model("gpt-4o")``, because naming a model that is not being used makes the
estimate look authoritative. Nothing is sampled and extrapolated: the fitter's whole job is
not to overflow, and extrapolation turns a bounded error into an unbounded one on precisely
the longest inputs.

**Then it stops guessing.** The provider returns a true prompt count, and the estimate is
compared against it after every call. That comparison is only worth anything if the "true"
number is true — see :func:`usable_prompt_tokens`.

**The window cross-check is not here.** It is
:func:`manicule.retrieval.assembly.window_problem`, because retrieval owns the requirement and
one predicate stated twice is two predicates that will disagree. Generation owns the
*enforcement point* — :meth:`~manicule.generation.provider.LitellmGenerator.setup` — because
that is where the served window becomes known.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from manicule.core.content import Chunk
from manicule.core.generation import Usage

GENERATION_ENCODING = "o200k_base"
"""The stand-in vocabulary, by encoding name.

The generator is usually Ollama-hosted, running a Llama, Qwen or Mistral vocabulary — none of
which is this one. Recording the encoding name in the trace is what keeps the estimate
honest about being an estimate.
"""


@dataclass(slots=True)
class TokenEstimator:
    """Estimates prompt tokens, inflated by a safety factor, cached by chunk id.

    The cache is keyed by :attr:`~manicule.core.content.Chunk.id`, which is content-derived,
    so it is exact and can never go stale.
    """

    safety_factor: float = 1.15
    encoding_name: str = GENERATION_ENCODING
    _encode: Callable[[str], int] | None = field(default=None, init=False)
    _chunks: dict[str, int] = field(default_factory=dict[str, int], init=False)

    def count(self, text: str) -> int:
        """Estimated tokens for ``text``, rounded up after the safety factor."""
        if not text:
            return 0
        return int(self.raw_count(text) * self.safety_factor) + 1

    def raw_count(self, text: str) -> int:
        """The uninflated count, for reporting drift against a true measurement."""
        if not text:
            return 0
        if self._encode is None:
            self._encode = _tiktoken_counter(self.encoding_name)
        return self._encode(text)

    def count_chunk(self, chunk: Chunk) -> int:
        """Estimated tokens for a chunk's citable text, cached by its content-derived id."""
        cached = self._chunks.get(chunk.id)
        if cached is None:
            cached = self.count(chunk.text)
            self._chunks[chunk.id] = cached
        return cached

    def count_all(self, texts: Iterable[str]) -> int:
        return sum(self.count(text) for text in texts)


def _tiktoken_counter(encoding_name: str) -> Callable[[str], int]:
    """A ``tiktoken`` counter, imported here so nothing pays for it until it counts.

    Through :func:`manicule.vocabularies.load_encoding` rather than ``tiktoken.get_encoding``,
    which fetches the vocabulary from a blob store when the cache cannot answer — on the path
    that answers a question, where a download is the least useful thing that can happen. The
    pre-seed is :func:`manicule.vocabularies.prefetch`.
    """
    from manicule import vocabularies  # noqa: PLC0415 - deliberately not a module-level import

    encoding = vocabularies.load_encoding(encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def usable_prompt_tokens(usage: Usage | None, estimate: int) -> int | None:
    """The provider's prompt count, or ``None`` when it cannot be trusted as a measurement.

    **The hazard is not a missing number, it is a plausible one in the field reserved for a
    measurement.** When a provider reports no usage, the generation library substitutes its
    own estimator on the non-streaming path and ``0`` on the streaming path, and neither
    announces itself. A calibration loop fed an estimate on both sides agrees with itself
    forever and reports excellent health.

    So two readings degrade to "unknown" rather than being recorded as measurements: a count
    of ``0``, and a count that matches manicule's own estimate exactly. Neither is proof of a
    fallback on its own — a genuinely empty completion and a lucky exact match both exist —
    which is why this degrades rather than raising. A number that came from an estimator
    agreeing with an estimator is not evidence.
    """
    if usage is None or usage.prompt_tokens <= 0:
        return None
    if usage.prompt_tokens == estimate:
        return None
    return usage.prompt_tokens


def drift_problem(*, estimate: int, measured: int | None, tolerance: float, model: str) -> str:
    """An error-level description of tokenizer drift, or ``""``.

    Reported, **never** auto-tuned. Feeding measured drift back into the safety factor makes
    two runs non-comparable — the same query fits a different number of passages depending on
    what the process has seen since it started — and it adapts in the unsafe direction, since
    a run of short English answers lowers the factor and the first long CJK or code-heavy
    prompt then overflows a window a fixed factor would have respected.
    """
    if measured is None or measured <= 0:
        return ""
    relative = abs(estimate - measured) / measured
    if relative <= tolerance:
        return ""
    return (
        f"prompt token estimate {estimate} against {model}'s true count {measured} "
        f"({relative:.0%} drift, tolerance {tolerance:.0%}). The estimate uses the "
        f"{GENERATION_ENCODING!r} encoding, which is not this model's vocabulary. Raise "
        f"llm.token_safety_factor if the estimate is low; a count pinned at the context "
        f"window instead of tracking the estimate means the server trimmed the prompt."
    )


__all__ = [
    "GENERATION_ENCODING",
    "TokenEstimator",
    "drift_problem",
    "usable_prompt_tokens",
]
