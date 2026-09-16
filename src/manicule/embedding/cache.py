"""The embedding cache: a memo over a pure function.

``(canonical(EmbedFingerprint), embed_text) -> vector``, and the first half of that key is
the whole design.

**Keyed by the full canonical fingerprint, never a model name.** The same weights with a
different pooling, revision or tokenizer produce a different vector for the same text, and
those are exactly the changes a name does not carry. Keying on the identical bytes
``index_state`` stores buys two properties without a mechanism: a cached vector is admissible
in the live index by construction, and a fingerprint change invalidates the cache with no
flush step for anyone to forget.

That second property closes a real hole. ``reindex --re-embed`` against a name-keyed cache
repopulates the *new* vector table with vectors from the *old* space and reports success —
laundering stale vectors past the refusal that exists to stop exactly that, at the one moment
the refusal is being deliberately stood down.

**The key is the post-middleware ``embed_text``**, the exact string handed to the model.
Keying the pre-middleware text would make an installed middleware return vectors computed
from different text than the caller believes.

**Not keyed by workspace or document.** That would destroy deduplication precisely where hit
rates are highest — repeated boilerplate, one attachment reachable from forty pages — and a
hit reveals nothing the caller does not already hold, since it holds the text and could
compute the same vector itself. One honest caveat: a shared cache is a weak timing oracle for
"has anyone here embedded this exact string", which is a property of a self-hosted tool rather
than a defect.

**The cache owns its values, so it stores them frozen.** A memo that hands back the list it is
holding is not a memo: one caller editing a returned vector in place changes what every later
caller is told, and nothing anywhere would report it — the fingerprint still matches, the
counts still add up, and the index fills with vectors the model never produced. So a vector is
converted to a tuple on the way in and that tuple is what every reader gets, which also settles
the subtler half: two copies of one text in a single batch share an object by design, and
sharing an immutable one is free where sharing a mutable one is a defect.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from manicule.core.embedding import EmbedFingerprint, Vector

type CacheKey = tuple[str, str]
"""``(fingerprint.canonical(), embed_text)``."""

type FrozenVector = tuple[float, ...]
"""What the cache stores and returns: a :data:`~manicule.core.embedding.Vector` a caller cannot
edit. ``Sequence[float]`` already admits it, so nothing downstream changes shape."""


class EmbeddingCache:
    """A bounded least-recently-used memo, keyed on identity rather than on a name.

    Not thread-safe and deliberately not locked: it is reached from one embedder, whose
    forward passes are serialized anyway, and a lock here would cost more than the lookup.
    """

    def __init__(self, capacity: int) -> None:
        """Args:
        capacity: How many vectors to keep. ``0`` disables the cache without needing a
            second code path — every lookup misses and every store is dropped, so the
            embedder that holds it behaves identically either way.
        """
        if capacity < 0:
            msg = f"cache capacity cannot be negative, got {capacity}"
            raise ValueError(msg)
        self._capacity = capacity
        self._entries: OrderedDict[CacheKey, FrozenVector] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def key(fingerprint: EmbedFingerprint, text: str) -> CacheKey:
        """The cache key for one text under one embedder.

        :meth:`~manicule.core.fingerprints.Fingerprint.canonical` and not ``model_id``: it is
        the same serialization the index is written against, so anything that would make a
        vector inadmissible also makes it unreachable.
        """
        return (fingerprint.canonical(), text)

    def get(self, fingerprint: EmbedFingerprint, text: str) -> FrozenVector | None:
        """The stored vector for ``text`` under ``fingerprint``, or ``None``.

        The returned tuple *is* the stored one. Handing out the cache's own object is safe
        precisely because it cannot be edited.
        """
        key = self.key(fingerprint, text)
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry

    def put(self, fingerprint: EmbedFingerprint, text: str, vector: Vector) -> FrozenVector:
        """Store ``vector`` frozen, evicting the least recently used entry if full.

        Returns the frozen value, which is the point of returning anything: the caller is
        about to hand this vector to whoever asked for it, and giving back the cache's own
        copy is what stops the two diverging. A disabled cache still freezes and returns, so
        ``capacity = 0`` changes how much is remembered and nothing about what callers hold.
        """
        frozen = tuple(float(value) for value in vector)
        if self._capacity == 0:
            return frozen
        key = self.key(fingerprint, text)
        self._entries[key] = frozen
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return frozen

    def lookup(
        self, fingerprint: EmbedFingerprint, texts: Sequence[str]
    ) -> tuple[list[FrozenVector | None], list[str]]:
        """Resolve a whole batch at once.

        Returns:
            A slot per input, holding the cached vector or ``None``, and the distinct texts
            still needing the model, in first-seen order. Distinct, so a batch of forty copies
            of the same boilerplate costs one forward pass rather than forty — which is where
            a corpus's hit rate actually comes from.
        """
        slots: list[FrozenVector | None] = []
        pending: list[str] = []
        seen: set[str] = set()
        for text in texts:
            cached = self.get(fingerprint, text)
            slots.append(cached)
            if cached is None and text not in seen:
                seen.add(text)
                pending.append(text)
        return slots, pending

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["CacheKey", "EmbeddingCache", "FrozenVector"]
