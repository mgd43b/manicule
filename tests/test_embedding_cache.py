"""The embedding cache, and the one mistake it exists to make impossible.

A cache keyed by anything less than the full fingerprint serves vectors from one space to a
caller working in another. The failure mode is not a wrong answer — it is ``reindex
--re-embed`` filling a fresh vector table with the previous model's vectors and printing
success, at the one moment the refusal that would have caught it is deliberately stood down.
"""

from __future__ import annotations

from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.embedding.cache import EmbeddingCache
from tests.embedding_fakes import NameKeyedCache


def fingerprint(
    *, pooling: Pooling = Pooling.CLS, revision: str | None = "abc123", dimension: int = 4
) -> EmbedFingerprint:
    return EmbedFingerprint(
        model_id="BAAI/bge-m3",
        revision=revision,
        dimension=dimension,
        pooling=pooling,
        normalized=True,
        tokenizer_id="BAAI/bge-m3",
        max_sequence_length=8190,
        backend="mlx",
    )


def test_the_key_is_the_canonical_fingerprint() -> None:
    """The same bytes ``index_state`` stores, so a cached vector is admissible by construction."""
    print_ = fingerprint()

    assert EmbeddingCache.key(print_, "hello") == (print_.canonical(), "hello")


def test_a_changed_pooling_invalidates_the_cache_with_no_flush_step() -> None:
    """The property that closes the ``--re-embed`` hole.

    Same model, same weights, same dimension, different reduction — which is a different vector
    space. Keying on the canonical fingerprint makes the old entries unreachable rather than
    stale, so there is no invalidation step for anyone to forget to run.
    """
    cache = EmbeddingCache(capacity=10)
    cache.put(fingerprint(pooling=Pooling.CLS), "shared boilerplate", [1.0, 0.0, 0.0, 0.0])

    assert cache.get(fingerprint(pooling=Pooling.MEAN), "shared boilerplate") is None
    assert cache.get(fingerprint(pooling=Pooling.CLS), "shared boilerplate") is not None


def test_a_changed_revision_invalidates_the_cache() -> None:
    """Unpinned weights can change under a corpus; a pinned revision is how that is noticed."""
    cache = EmbeddingCache(capacity=10)
    cache.put(fingerprint(revision="abc123"), "text", [1.0, 0.0, 0.0, 0.0])

    assert cache.get(fingerprint(revision="def456"), "text") is None


def test_a_name_keyed_cache_serves_vectors_from_the_wrong_space() -> None:
    """The fake that shows the keying is load-bearing rather than decorative.

    ``NameKeyedCache`` is what "key by model identity" means read loosely. It returns the
    CLS-pooled vector to a caller that has switched the model to mean pooling — a vector from a
    space cosine 0.66-0.80 away, handed over as a hit, with nothing to notice it.
    """
    cache = NameKeyedCache(capacity=10)
    cls_vector = [1.0, 0.0, 0.0, 0.0]
    cache.put(fingerprint(pooling=Pooling.CLS), "text", cls_vector)

    served = cache.get(fingerprint(pooling=Pooling.MEAN), "text")

    assert served == cls_vector, "the fake is meant to be wrong; if it stopped being, fix it"
    assert EmbeddingCache(capacity=10).get(fingerprint(pooling=Pooling.MEAN), "text") is None


def test_the_cache_is_bounded_and_evicts_the_least_recently_used() -> None:
    """An unbounded memo over a corpus is a memory leak with a hit rate."""
    cache = EmbeddingCache(capacity=2)
    print_ = fingerprint()
    for name in ("a", "b"):
        cache.put(print_, name, [1.0])

    cache.get(print_, "a")  # makes "b" the least recently used
    cache.put(print_, "c", [1.0])

    assert len(cache) == 2
    assert cache.get(print_, "b") is None
    assert cache.get(print_, "a") is not None


def test_a_capacity_of_zero_disables_the_cache_without_a_second_code_path() -> None:
    """Configuration turns it off; the embedder holding it behaves identically either way."""
    cache = EmbeddingCache(capacity=0)
    print_ = fingerprint()
    cache.put(print_, "text", [1.0])

    assert len(cache) == 0
    assert cache.get(print_, "text") is None


def test_lookup_asks_for_each_distinct_text_once() -> None:
    """Where a corpus's hit rate actually comes from: the same boilerplate on forty pages."""
    cache = EmbeddingCache(capacity=10)
    print_ = fingerprint()
    cache.put(print_, "cached", [1.0])

    slots, pending = cache.lookup(print_, ["cached", "new", "new", "other"])

    assert slots[0] == [1.0]
    assert slots[1] is None
    assert pending == ["new", "other"]


def test_hits_and_misses_are_counted_for_diagnostics() -> None:
    """A cache whose hit rate is unobservable cannot be sized."""
    cache = EmbeddingCache(capacity=10)
    print_ = fingerprint()
    cache.put(print_, "text", [1.0])
    cache.get(print_, "text")
    cache.get(print_, "absent")

    assert (cache.hits, cache.misses) == (1, 1)
