"""A typed seam over ``tokenizers``.

manicule tokenizes rather than letting a backend do it, for two reasons that both end in a
wrong vector nobody sees. The backends' embedding calls do not surface attention masks, and a
mean pool without a mask averages in the padding — so the same text embeds differently
depending on the longest text that shared its batch. And a tokenizer configured to truncate
returns a well-formed shortened sequence, which becomes a vector describing the opening of a
chunk that still claims all of its text.

So the wrapper here does two things the raw library will not do for you: it pads with the
model's own pad token, and it **never truncates**, leaving an over-long sequence intact so that
the caller can refuse it by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Encoded:
    """One batch, as plain Python. Converted to arrays by whoever needs arrays."""

    ids: list[list[int]]
    attention_mask: list[list[int]]


class FastTokenizer:
    """The model's own vocabulary, padded and never truncating."""

    def __init__(self, tokenizer_file: Path) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415 - kept out of import time

        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.no_truncation()

    def enable_padding(self, pad_id: int, pad_token: str) -> None:
        """Pad to the longest sequence in each batch, with the model's own pad token."""
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

    def token_to_id(self, token: str) -> int | None:
        found: int | None = self._tokenizer.token_to_id(token)
        return found

    def id_to_token(self, token_id: int) -> str | None:
        found: str | None = self._tokenizer.id_to_token(token_id)
        return found

    def special_token_count(self) -> int:
        """How many tokens this model wraps every input in.

        Measured by encoding the empty string rather than counted off a list of names: what
        matters is what the tokenizer actually adds, because that is what eats into the
        sequence budget. XLM-RoBERTa answers 2, for ``<s>`` and ``</s>``.
        """
        wrapped: list[int] = self._tokenizer.encode("", add_special_tokens=True).ids
        return len(wrapped)

    def content_ids(self, text: str) -> list[int]:
        """Token ids for ``text`` with no special tokens, for counting content."""
        ids: list[int] = self._tokenizer.encode(text, add_special_tokens=False).ids
        return ids

    def encode_batch(self, texts: Sequence[str]) -> Encoded:
        """Ids and attention masks for a batch, padded to its longest member."""
        encoded = self._tokenizer.encode_batch(list(texts))
        ids: list[list[int]] = [item.ids for item in encoded]
        mask: list[list[int]] = [item.attention_mask for item in encoded]
        return Encoded(ids=ids, attention_mask=mask)


__all__ = ["Encoded", "FastTokenizer"]
