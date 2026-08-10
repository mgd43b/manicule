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
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from manicule.config.profiles import ProfileConfig
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
    """A ``tiktoken`` counter, imported here so nothing pays for it until it counts."""
    import tiktoken  # noqa: PLC0415 - deliberately not a module-level import

    encoding = tiktoken.get_encoding(encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True, slots=True)
class WindowBudget:
    """The four terms of the startup cross-check, and their total."""

    context_tokens: int
    history_tokens: int
    system_prompt_tokens: int
    generation_reserve: int

    @property
    def total(self) -> int:
        return (
            self.context_tokens
            + self.history_tokens
            + self.system_prompt_tokens
            + self.generation_reserve
        )

    def describe(self) -> str:
        return (
            f"{self.context_tokens} context + {self.history_tokens} history + "
            f"{self.system_prompt_tokens} system prompt + {self.generation_reserve} reserved "
            f"for the answer = {self.total}"
        )


def window_budget(
    profile: ProfileConfig, *, system_prompt_tokens: int, max_tokens: int
) -> WindowBudget:
    """The budget a profile demands of a generator's window.

    ``max_tokens`` appears once, as both the output cap and the reserve. Two numbers for one
    quantity disagree by default — and then a ``length`` finish reason stops meaning "the
    answer hit the budget reserved for it".
    """
    return WindowBudget(
        context_tokens=profile.context_tokens,
        history_tokens=profile.history_tokens,
        system_prompt_tokens=system_prompt_tokens,
        generation_reserve=max_tokens,
    )


def window_problem(budget: WindowBudget, *, context_window: int, model: str, profile: str) -> str:
    """The refusal when a profile does not fit, or ``""`` when it does.

    A refusal at startup rather than a truncation at run time, because on the default local
    runtime neither overflow behaviour is acceptable to rely on: older builds truncate the
    prompt **from the front**, silently discarding the system prompt and the citation
    protocol and presenting as a model that ignores instructions, while current builds *grow*
    the context to hold the prompt, which on unified memory is a spill to CPU or an outright
    failure to allocate.

    Both alternatives to naming it are worse than the arithmetic being wrong, which is why
    this reports both totals and the three things that fix it.
    """
    if budget.total <= context_window:
        return ""
    return (
        f"the {profile!r} profile needs {budget.total} tokens ({budget.describe()}) but "
        f"{model} will serve a window of {context_window}. A prompt over the window is "
        f"truncated or grown by the runtime rather than refused, so this is checked here "
        f"instead of being discovered in production. Choose a model with a longer window, "
        f"lower rag.overrides.context_tokens, or select a smaller profile."
    )


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
    "WindowBudget",
    "drift_problem",
    "usable_prompt_tokens",
    "window_budget",
    "window_problem",
]
