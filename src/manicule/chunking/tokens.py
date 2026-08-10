"""Counting tokens the way the model that enforces the limit counts them.

A budget checked against a hard model limit has to be measured with the tokenizer that
enforces that limit. Estimating is not good enough, and the error runs one way:

- **Vocabularies disagree.** ``tiktoken``'s BPE vocabularies are 100k-200k entries; the
  retrieval models manicule targets use 30k-256k WordPiece or SentencePiece. WordPiece
  splits the same English prose into *more* tokens, and the gap widens sharply for code,
  identifiers and any non-Latin script.
- **Undercounting is the dangerous direction.** A chunk that overflows the model is
  truncated with no error raised, and is then indexed as its opening tokens while still
  claiming all of its text. Overcounting only wastes budget.
- **Sampling makes it worse.** Encoding the first few thousand characters and extrapolating
  turns a bounded error into an unbounded one on exactly the documents that matter.

So the counter comes from the bound embedder. When no embedder is bound — a dry-run parse, a
fixture test — a provisional counter stands in, applies a safety factor, and marks its output
so ingest refuses it. Provisional chunks never reach an index.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from manicule.core.embedding import EmbedFingerprint

PROVISIONAL_SAFETY_FACTOR = 1.5
"""How much a provisional count is inflated.

A stand-in vocabulary can only undercount by an unknown margin, and undercounting is the
direction that truncates. Inflating trades wasted budget — which costs nothing but a few
extra chunks — for the one error that is silent.
"""


@runtime_checkable
class SupportsTokenCount(Protocol):
    """An embedder that can count tokens the way it will tokenize them.

    This is the shape ``docs/parsing.md`` §1.1 asks of the embedding work: a fingerprint is
    not enough, because the chunker has to measure the exact string the model will see. An
    embedder without it can still be used — the chunker refuses to start rather than guessing
    on its behalf, and says which method is missing.
    """

    fingerprint: EmbedFingerprint

    def count_tokens(self, text: str) -> int:
        """How many tokens this model will make of ``text``."""
        ...


class TokenCounter:
    """A token count, and the identity of whatever produced it.

    ``tokenizer_id`` goes into :class:`~manicule.core.fingerprints.ChunkFingerprint`, because
    the same budget measured with a different vocabulary produces different chunk boundaries.
    A model swap that keeps the dimension but changes the vocabulary would otherwise pass the
    embedder check and quietly re-chunk the corpus.
    """

    def __init__(
        self, tokenizer_id: str, count: Callable[[str], int], *, provisional: bool
    ) -> None:
        self.tokenizer_id = tokenizer_id
        self.provisional = provisional
        self._count = count

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        raw = self._count(text)
        if not self.provisional:
            return raw
        return int(raw * PROVISIONAL_SAFETY_FACTOR) + 1

    @classmethod
    def bound_to(cls, embedder: SupportsTokenCount) -> TokenCounter:
        """Count with the model's own tokenizer, on the exact string it will see."""
        return cls(
            embedder.fingerprint.tokenizer_id or embedder.fingerprint.model_id,
            embedder.count_tokens,
            provisional=False,
        )

    @classmethod
    def provisionally(cls, count: Callable[[str], int] | None = None) -> TokenCounter:
        """Count without a model, inflating the result and marking it.

        Args:
            count: The stand-in counter. Defaults to ``tiktoken``'s ``cl100k_base``, loaded
                on first use so that a chunker bound to a real embedder never imports it.
        """
        if count is not None:
            return cls("provisional", count, provisional=True)
        return cls("provisional", _tiktoken_counter(), provisional=True)


def _tiktoken_counter() -> Callable[[str], int]:
    """A ``tiktoken`` counter, imported here so the fast path never pays for it."""
    import tiktoken  # noqa: PLC0415 - deliberately not a module-level import

    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))


__all__ = ["PROVISIONAL_SAFETY_FACTOR", "SupportsTokenCount", "TokenCounter"]
