"""The benchmark is executed by the suite, at a size that costs nothing.

A benchmark nobody runs is a script that stops working and says nothing, and this one is cited
in two design documents as an argument. So the suite runs it — small, so it is free — and
asserts the shape of its result rather than its magnitude: that reuse suppressed embeds
everything, that reuse on embeds nothing, and that the fraction of documents whose text moves
is the fraction of chunks that reach the model.

The magnitudes in the documents come from the default size, which is deliberately larger than
the embedding cache and correspondingly slower. This is the guard that the program still works,
not a re-taking of the published numbers.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.embedding_reuse import CACHE_CAPACITY, Measurement, measure, render

DOCUMENTS = 4
CHUNKS_EACH = 5
TOTAL = DOCUMENTS * CHUNKS_EACH


async def test_suppressing_reuse_embeds_the_corpus_twice() -> None:
    """The "before" row is the path ingest took before durable reuse, and must still be it."""
    first, second = await measure(
        documents=DOCUMENTS,
        chunks_per_document=CHUNKS_EACH,
        changed_fraction=0.0,
        reuse=False,
    )

    assert first.embedded == TOTAL
    assert second.embedded == TOTAL, (
        "with reuse suppressed a re-parse embeds every chunk again — if this ever reads zero "
        "the comparison in the documents has quietly become a comparison of nothing"
    )
    assert second.forward_calls > 0


async def test_reuse_removes_the_re_parse_entirely_when_no_text_moves() -> None:
    """The claim the documents make, at a size the suite can afford."""
    first, second = await measure(
        documents=DOCUMENTS,
        chunks_per_document=CHUNKS_EACH,
        changed_fraction=0.0,
        reuse=True,
    )

    assert first.embedded == TOTAL
    assert (second.embedded, second.forward_calls) == (0, 0)
    assert second.documents == DOCUMENTS, "every document was still rebuilt from retained bytes"


@pytest.mark.parametrize("fraction", [0.25, 0.5])
async def test_the_fraction_embedded_is_the_fraction_that_moved(fraction: float) -> None:
    """What a structural rebuild costs is proportional to what it actually changes.

    The property another caller needs from this program: not that reuse works, but that the
    number it reports tracks the change. A benchmark that only ever measured the all-or-nothing
    cases could not be used to price a partial migration.
    """
    _, second = await measure(
        documents=DOCUMENTS,
        chunks_per_document=CHUNKS_EACH,
        changed_fraction=fraction,
        reuse=True,
    )

    assert second.embedded == round(DOCUMENTS * fraction) * CHUNKS_EACH


def test_the_rendering_lines_up_and_names_both_passes() -> None:
    """It is pasted into pull requests, so it has to survive being pasted."""
    passes = [
        Measurement("first ingest", 2, 10, 0, 10, 1, 0.5, 1.0),
        Measurement("re-parse", 2, 10, 3, 0, 0, 0.25, 1.0),
    ]

    rendered = render(passes).splitlines()

    assert "first ingest" in rendered[0]
    assert "re-parse" in rendered[0]
    assert len({len(line) for line in rendered}) == 1, (
        "every row ends in the same column, which is what makes the two passes comparable by "
        "eye — values are right-aligned in fixed columns, so a ragged edge means one overflowed"
    )
    assert all(len(line) <= 80 for line in rendered), (
        "and the table fits a default terminal, so a pipe does not reflow it into nonsense"
    )
    assert "served by the warm cache" in rendered[3]


def test_the_default_corpus_is_larger_than_the_cache_it_is_measured_against() -> None:
    """The one property the published numbers depend on and the suite does not otherwise run.

    At or below the cache's capacity a warm process could account for the whole result, so the
    default size is what makes the figures in the documents about durable reuse rather than
    about an LRU. The suite runs the benchmark small; nothing else checks that the size the
    documents quote is still the size that means something.
    """
    default_documents, default_chunks_each = 20, 1_000

    assert default_documents * default_chunks_each > CACHE_CAPACITY
