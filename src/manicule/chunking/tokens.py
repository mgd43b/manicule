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

**Marking it is the whole of the guard, so the mark says everything.** ``tokenizer_id`` is an
identity field of :class:`~manicule.core.fingerprints.ChunkFingerprint`, and a constant there
would stand in for three separate things that each move every boundary: the stand-in
vocabulary's own version, whatever counter a caller injected, and
:data:`PROVISIONAL_SAFETY_FACTOR`. So the id is assembled from all three, in one place, at
construction — not chosen by a caller and not defaulted. A counter that cannot say what it
is cannot be built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol, runtime_checkable

from manicule.core.embedding import EmbedFingerprint
from manicule.core.errors import ConfigError
from manicule.core.fingerprints import PROVISIONAL_TOKENIZER_PREFIX

PROVISIONAL_SAFETY_FACTOR = 1.5
"""How much a provisional count is inflated.

A stand-in vocabulary can only undercount by an unknown margin, and undercounting is the
direction that truncates. Inflating trades wasted budget — which costs nothing but a few
extra chunks — for the one error that is silent.

It is in the recorded identity rather than only here, because it multiplies every count the
chunker takes: changing it to 1.6 rewrites every boundary in the corpus, and a fingerprint
that did not move would call the two chunkings interchangeable.
"""

TIKTOKEN_ENCODING: Final = "cl100k_base"
"""The stand-in vocabulary, pinned. Recorded with its distribution version, not on its own —
the encoding name is stable across releases whose token boundaries are not."""


def provisional_tokenizer_id(base: str) -> str:
    """The recorded identity of a count taken without the model that will embed it.

    Three things decide a provisional boundary and all three are here: the prefix that makes
    the count refusable at all, the safety factor that inflates it, and whatever vocabulary
    stood in. Assembled by one function so that no caller can supply two of the three.

    Args:
        base: What did the counting — ``tiktoken/cl100k_base@0.13.0`` for the default, or a
            caller's own name for an injected counter.

    Returns:
        ``provisional:x1.5:tiktoken/cl100k_base@0.13.0`` and the like. The factor is
        formatted with ``repr`` precision so that 1.5 and 1.50000001 are different strings.
    """
    return f"{PROVISIONAL_TOKENIZER_PREFIX}x{PROVISIONAL_SAFETY_FACTOR!r}:{base}"


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
        """Bind a counter to the identity of whatever does the counting.

        Args:
            tokenizer_id: What counted. For a bound embedder this is the model's own
                tokenizer; for a stand-in it is the vocabulary that stood in, and the
                recorded id is then :func:`provisional_tokenizer_id` of it.
            count: The counting itself.
            provisional: Whether this is a stand-in. Stamped into the recorded id **here**,
                rather than left to the classmethod below, because this constructor is public
                and a caller reaching it directly would otherwise inflate every count by
                :data:`PROVISIONAL_SAFETY_FACTOR` while recording an id that says it did not.

        Raises:
            ConfigError: ``tokenizer_id`` is empty. An unnamed counter produces boundaries no
                fingerprint can describe, which is the one state this class exists to make
                unrepresentable.
        """
        if not tokenizer_id:
            msg = (
                "a token counter must name what counted: 'tokenizer_id' is empty. The same "
                "text is a different number of tokens under a different vocabulary, so a "
                "chunk fingerprint built from an unnamed counter would call two chunkings "
                "interchangeable."
            )
            raise ConfigError(msg)
        self.tokenizer_id = provisional_tokenizer_id(tokenizer_id) if provisional else tokenizer_id
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
    def provisionally(
        cls, count: Callable[[str], int] | None = None, *, tokenizer_id: str | None = None
    ) -> TokenCounter:
        """Count without a model, inflating the result and marking it.

        Args:
            count: The stand-in counter. Omitted, it is ``tiktoken``'s
                :data:`TIKTOKEN_ENCODING`, imported on first use so that a chunker bound to a
                real embedder never pays for it.
            tokenizer_id: What ``count`` is, when one is supplied. **Required with it**, and
                rejected without it. A callable has no version anyone can read: its
                ``__qualname__`` is ``<lambda>`` for every lambda in a module, and its
                ``id()`` differs between two runs of the same program, so neither derives an
                identity that is both distinguishing and reproducible. The caller is the only
                party that knows, so the caller says.

        Raises:
            ConfigError: A counter was supplied without a name, or a name without a counter.
                The first is the defect this argument exists for — two stand-in counters that
                disagree about every boundary, sharing one fingerprint. The second would put
                a caller's label on ``tiktoken``'s work.
        """
        if count is None:
            if tokenizer_id is not None:
                msg = (
                    f"'tokenizer_id' was given without a counter, so it would name "
                    f"tiktoken's {TIKTOKEN_ENCODING} vocabulary something it is not. Pass "
                    f"the counter it describes, or omit both and let the default name itself."
                )
                raise ConfigError(msg)
            return cls(tiktoken_tokenizer_id(), _tiktoken_counter(), provisional=True)
        if tokenizer_id is None:
            msg = (
                "a stand-in token counter must name itself: pass 'tokenizer_id' alongside "
                "'count'. Two counters that disagree about every boundary would otherwise "
                "record the same chunk fingerprint, and an index built with one would be "
                "accepted as an index built with the other."
            )
            raise ConfigError(msg)
        return cls(tokenizer_id, count, provisional=True)


def tiktoken_tokenizer_id() -> str:
    """The stand-in vocabulary's identity: encoding *and* installed distribution version.

    The encoding name alone is not enough and that gap is the reason this function exists.
    ``cl100k_base`` has meant subtly different token boundaries across ``tiktoken`` releases,
    so a bump would move every provisional chunk boundary under an identifier that never
    moved. Read from distribution metadata rather than from the module, so asking costs no
    import of the Rust extension.

    Raises:
        PackageNotFoundError: ``tiktoken`` is not installed, so there is no stand-in counter
            to name. Raised where it can be acted on rather than absorbed into a default:
            a version that falls back to a constant is the constant this function replaced.
    """
    from importlib.metadata import version  # noqa: PLC0415 - see docstring

    return f"tiktoken/{TIKTOKEN_ENCODING}@{version('tiktoken')}"


def _tiktoken_counter() -> Callable[[str], int]:
    """A ``tiktoken`` counter, imported here so the fast path never pays for it."""
    import tiktoken  # noqa: PLC0415 - deliberately not a module-level import

    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


__all__ = [
    "PROVISIONAL_SAFETY_FACTOR",
    "TIKTOKEN_ENCODING",
    "SupportsTokenCount",
    "TokenCounter",
    "provisional_tokenizer_id",
    "tiktoken_tokenizer_id",
]
