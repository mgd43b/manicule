"""The benchmark is executed by the suite, at a size that costs nothing.

A benchmark nobody runs is a script that stops working and says nothing. So the suite runs
the stand-in matrix small — free — and asserts the *shape* of the result: that the largest
tokenizer call does not grow with the block, and that doubling the block does not quadruple
the work. The magnitudes quoted in the fix's report come from the default size with the real
vocabulary, which is the run below that skips unless the assets are already on the machine.

The real-vocabulary check asserts the same structural facts rather than a number of seconds.
Its value is that it takes them through the encode path that actually cost the time, so a
regression that only a real tokenizer exposes cannot hide behind a stand-in.
"""

from __future__ import annotations

import pytest

from manicule.chunking import MAX_TOKENS
from tests.benchmarks.bounded_tokenization import (
    BGE_M3_REPO,
    KIB,
    SHAPES,
    bge_m3_counter,
    measure,
    render,
    run,
    stand_in_counter,
)

SMALL = (64 * KIB, 128 * KIB)


def test_the_program_still_runs_and_renders() -> None:
    measurements = run(SMALL)

    assert len(measurements) == len(SHAPES) * len(SMALL)
    assert "chunk budget" in render(measurements)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_largest_call_does_not_grow_with_the_block(shape: str) -> None:
    """The invariant, as the benchmark reports it.

    Duplicated from the unit suite on purpose: this asserts against the numbers a reader of
    the benchmark output is being asked to trust, so the two cannot drift apart silently.
    """
    small = measure(shape, SMALL[0], stand_in_counter, tokenizer="stand-in")
    large = measure(shape, SMALL[1], stand_in_counter, tokenizer="stand-in")

    assert small.largest_call_chars == large.largest_call_chars
    assert large.amplification <= small.amplification * 1.2, (
        f"characters tokenized per character of input rose from {small.amplification} to "
        f"{large.amplification} when the block doubled. A bounded call size with a rising "
        f"amplification is the quadratic search returning through the enclosing loop"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_chunk_fits_the_budget_the_fingerprint_claims(shape: str) -> None:
    item = measure(shape, SMALL[1], stand_in_counter, tokenizer="stand-in")

    assert 0 < item.max_final_tokens <= MAX_TOKENS


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_bound_holds_under_the_production_vocabulary(shape: str) -> None:
    """Opt-in: the same assertions through BGE-M3's own encode path.

    Skipped rather than failed when the model is absent, and never downloaded — a suite that
    fetched a model to run a benchmark would make every fresh checkout pay for it.
    """
    try:
        count = bge_m3_counter()
    except RuntimeError as error:  # pragma: no cover - depends on the machine's model cache
        pytest.skip(f"{BGE_M3_REPO} not cached: {error}")

    small = measure(shape, SMALL[0], count, tokenizer="bge-m3")
    large = measure(shape, SMALL[1], count, tokenizer="bge-m3")

    assert small.largest_call_chars == large.largest_call_chars
    assert large.amplification <= small.amplification * 1.2
    assert 0 < large.max_final_tokens <= MAX_TOKENS, (
        f"the largest final embedding input measured {large.max_final_tokens} tokens against "
        f"a {MAX_TOKENS}-token budget, under the vocabulary that will actually embed it"
    )
