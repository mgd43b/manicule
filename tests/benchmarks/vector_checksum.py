"""What numerical-integrity validation costs, next to the thing it replaces.

The design argument for a checksum is a price comparison: re-running the embedding model would
detect a corrupted vector too, and it costs a forward pass per row. This program measures the
three numbers that argument depends on, and keeps them apart so none of them can hide inside
another:

``hash``
    Checksum creation alone, over vectors already in memory. No storage, no Arrow, no I/O.

``verify``
    Recomputing a recorded checksum and comparing it. The read-path cost per row.

``scan``
    The same verification through :meth:`LanceVectorStore.checksum_coverage`, which also pays
    LanceDB's scan and deserialization. Subtracting ``verify`` from it is what separates "the
    checksum is expensive" from "reading a million vectors off disk is expensive", and those
    call for entirely different responses.

It also reports ``embedder_calls``, which is always zero and is measured rather than asserted in
prose: the whole design rests on validation never reaching a model, and a program that merely
*said* so would not notice the day somebody added a fallback. The embedder here is a counting
stand-in that raises if it is ever asked for a vector.

Everything is synthetic — generated vectors, temporary directories, no network, no corpus.

Run it::

    .venv/bin/python -m tests.benchmarks.vector_checksum
    .venv/bin/python -m tests.benchmarks.vector_checksum --rows 20000 --dimension 1024
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from manicule.core.anchors import Unlocated
from manicule.core.content import BlockKind, Chunk
from manicule.core.embedding import (
    VECTOR_CHECKSUM_VERSION,
    EmbedFingerprint,
    Pooling,
    canonical_stored_vector,
    vector_checksum,
    verify_stored_checksum,
)
from manicule.storage.vectors import LanceVectorStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.embedding import Vector

FLOAT32_BYTES = 4
"""The persisted width of one component, which is what ``bytes_per_second`` is counted in."""

WRITE_PAGE = 512
"""Rows per ``upsert``. Setup cost, deliberately outside every measurement below."""


@dataclass(frozen=True, slots=True)
class Measurement:
    """One phase, priced per row and per persisted byte."""

    phase: str
    rows: int
    dimension: int
    seconds: float
    rows_per_second: float
    bytes_per_second: float
    peak_rss_bytes: int
    checksum_coverage: float
    embedder_calls: int


class RefusingEmbedder:
    """An embedder that fails rather than embedding, so a call is a crash and not a slowdown.

    The claim being measured is "validation does not invoke the embedder". A counter alone would
    record the violation and carry on producing a number somebody might publish; this makes the
    violation the result.
    """

    def __init__(self, dimension: int) -> None:
        self.calls = 0
        self.fingerprint = EmbedFingerprint(
            model_id="benchmark/refusing",
            dimension=dimension,
            pooling=Pooling.MEAN,
            normalized=True,
            tokenizer_id="benchmark/tokenizer",
            max_sequence_length=512,
        )

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls += len(texts)
        msg = (
            "numerical-integrity validation asked an embedder for a vector. Checking a checksum "
            "must never cost a forward pass — that is the entire argument for having one."
        )
        raise AssertionError(msg)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"benchmark-chunk-{index:07d}",
        document_id="benchmark-document",
        text=f"synthetic chunk {index}",
        embed_text=f"Benchmark > synthetic chunk {index}",
        anchor=Unlocated(reason="synthetic benchmark chunk"),
        kind=BlockKind.PROSE,
        position=index,
        token_count=8,
        metadata={"lang": "en"},
    )


def _vectors(rows: int, dimension: int, seed: int = 20260821) -> list[tuple[float, ...]]:
    generator = random.Random(seed)  # noqa: S311 - synthetic vectors, not keys
    return [
        canonical_stored_vector([generator.gauss(0.0, 1.0) for _ in range(dimension)])
        for _ in range(rows)
    ]


def _measurement(
    phase: str,
    *,
    rows: int,
    dimension: int,
    seconds: float,
    coverage: float,
    embedder: RefusingEmbedder,
) -> Measurement:
    return Measurement(
        phase=phase,
        rows=rows,
        dimension=dimension,
        seconds=seconds,
        rows_per_second=rows / seconds if seconds else float("inf"),
        bytes_per_second=(rows * dimension * FLOAT32_BYTES) / seconds if seconds else float("inf"),
        peak_rss_bytes=_peak_rss_bytes(),
        checksum_coverage=coverage,
        embedder_calls=embedder.calls,
    )


async def measure(*, rows: int, dimension: int, page_size: int = 512) -> list[Measurement]:
    """Time the three phases over ``rows`` synthetic vectors of ``dimension``.

    Setup — generating vectors and writing them through LanceDB — is outside every timer. What
    is being priced is the check, not the fixture.
    """
    embedder = RefusingEmbedder(dimension)
    vectors = _vectors(rows, dimension)
    checksums = [vector_checksum(vector) for vector in vectors]

    started = time.perf_counter()
    for vector in vectors:
        vector_checksum(vector)
    hashing = time.perf_counter() - started

    started = time.perf_counter()
    for vector, checksum in zip(vectors, checksums, strict=True):
        verify_stored_checksum(
            vector, recorded=checksum, version=VECTOR_CHECKSUM_VERSION, required=True
        )
    verifying = time.perf_counter() - started

    with tempfile.TemporaryDirectory() as temporary:
        store = LanceVectorStore(Path(temporary) / "vectors")
        await store.ensure_ready(embedder.fingerprint)
        for start in range(0, rows, WRITE_PAGE):
            page = range(start, min(rows, start + WRITE_PAGE))
            await store.upsert(
                [_chunk(index) for index in page], [vectors[index] for index in page]
            )

        started = time.perf_counter()
        coverage = await store.checksum_coverage(recompute=True, page_size=page_size)
        scanning = time.perf_counter() - started
        await store.teardown()

    return [
        _measurement(
            "hash", rows=rows, dimension=dimension, seconds=hashing, coverage=1.0, embedder=embedder
        ),
        _measurement(
            "verify",
            rows=rows,
            dimension=dimension,
            seconds=verifying,
            coverage=1.0,
            embedder=embedder,
        ),
        _measurement(
            "scan",
            rows=rows,
            dimension=dimension,
            seconds=scanning,
            coverage=coverage.fraction,
            embedder=embedder,
        ),
    ]


def render(measurements: Sequence[Measurement]) -> str:
    """A fixed-column table, because this gets pasted into pull requests."""
    header = (
        f"{'phase':>8} {'rows/s':>12} {'MB/s':>10} {'seconds':>9} {'coverage':>9} {'embeds':>7}"
    )
    lines = [header]
    for measurement in measurements:
        lines.append(
            f"{measurement.phase:>8} "
            f"{measurement.rows_per_second:>12,.0f} "
            f"{measurement.bytes_per_second / 1e6:>10,.1f} "
            f"{measurement.seconds:>9.3f} "
            f"{measurement.checksum_coverage:>9.3f} "
            f"{measurement.embedder_calls:>7}"
        )
    return "\n".join(lines)


async def _main(rows: int, dimension: int, *, as_json: bool) -> None:
    measurements = await measure(rows=rows, dimension=dimension)
    if as_json:
        for measurement in measurements:
            print(json.dumps(asdict(measurement), sort_keys=True))  # noqa: T201 - benchmark output
        return
    print(render(measurements))  # noqa: T201 - benchmark output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(_main(arguments.rows, arguments.dimension, as_json=arguments.json))


if __name__ == "__main__":
    main()
