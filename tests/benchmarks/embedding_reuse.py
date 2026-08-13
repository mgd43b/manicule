"""What a corpus-wide re-parse costs at the embedder, measured rather than reasoned about.

The numbers in ``docs/parsing.md`` §4.5 and ``docs/ingest.md`` §10.1 come from here. That is
the whole reason this file exists in the repository rather than in somebody's scratch
directory: a figure cited in a design document as an argument, reproducible only by the person
who took it, is a claim whose check is not where its reader is.

Run it::

    .venv/bin/python -m tests.benchmarks.embedding_reuse
    .venv/bin/python -m tests.benchmarks.embedding_reuse --no-reuse
    .venv/bin/python -m tests.benchmarks.embedding_reuse --documents 40 --changed-fraction 0.1

**It measures forward passes, not seconds at a model.** The embedder is a counting stand-in,
because counting what a real model was asked for requires no real model and using one would
make the result depend on a machine. Elapsed time and peak memory are reported for the harness
around it, and are useful for comparing two runs of *this* program — they are not a prediction
of what an accelerator will take, and the constant factor against a real model is unmeasured.

**``--no-reuse`` is the "before" row**, and it is honest about what it does: it swaps in a
vector store that reports every chunk absent, which is precisely the path ingest took before
durable reuse existed — every prepared chunk handed to ``embed_chunks``. It is a reproduction
of the old behaviour rather than a checkout of the old commit, and this note is here so nobody
reads the number as the latter.

Everything is synthetic: generated line documents, temporary directories, no network.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from manicule.cli.render import console
from manicule.config.settings import EmbeddingSettings
from manicule.core.embedding import StoredVector, VectorState
from manicule.ingest.reindex import re_parse_stale
from tests.ingest import fakes
from tests.ingest.test_pipeline import build, parse_versions
from tests.ingest.test_reindex_sweep import MARKER, TrimmingLineParser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk
    from manicule.core.fingerprints import ParseFingerprint

CACHE_CAPACITY = EmbeddingSettings().cache_entries
"""The in-memory cache's configured size, read rather than restated.

A corpus at or below this could have its numbers explained by the LRU instead of by durable
reuse, so the default below is comfortably past it and ``--documents`` warns when it is not.
"""


class ForgetfulVectors(fakes.MemoryVectors):
    """A store that reports every chunk absent, whatever it holds.

    The "before" row. Reuse asks the store what it has; a store that always answers "nothing"
    sends every chunk to the model, which is what ingest did before any of this existed.
    Writing still works, so the run is otherwise identical — only the reuse answer changes.
    """

    @override
    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        return {chunk.id: StoredVector(state=VectorState.ABSENT) for chunk in chunks}


@dataclass(frozen=True)
class Measurement:
    """One pass over the corpus, in the terms the reuse report itself uses."""

    label: str
    documents: int
    chunks: int
    cache_hits: int
    embedded: int
    forward_calls: int
    seconds: float
    peak_mib: float

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("documents", f"{self.documents:,}"),
            ("chunks", f"{self.chunks:,}"),
            ("served by the warm cache", f"{self.cache_hits:,}"),
            ("model inputs embedded", f"{self.embedded:,}"),
            ("model calls", f"{self.forward_calls:,}"),
            ("elapsed", f"{self.seconds:.2f}s"),
            ("peak memory", f"{self.peak_mib:.1f} MiB"),
        ]


def corpus(*, documents: int, chunks_per_document: int, changed_fraction: float) -> dict[str, str]:
    """Documents of distinct lines, a fraction of them holding the marker version two removes.

    Distinct on purpose: duplicate text is what the in-memory cache is good at, so a corpus
    with repeats would let the LRU account for the result and prove nothing about the identity
    this benchmark is about.
    """
    changed = round(documents * changed_fraction)
    pages: dict[str, str] = {}
    for index in range(documents):
        marker = MARKER if index < changed else ""
        pages[f"doc-{index}"] = "\n".join(
            f"distinct line {index * chunks_per_document + line}{marker}"
            for line in range(chunks_per_document)
        )
    return pages


async def measure(
    *, documents: int, chunks_per_document: int, changed_fraction: float, reuse: bool
) -> tuple[Measurement, Measurement]:
    """Index the corpus, then re-parse it under a bumped parser. Returns both passes."""
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors() if reuse else ForgetfulVectors()
    blobs = fakes.MemoryBlobs()
    embedder = fakes.CountingEmbedder()
    await vectors.ensure_ready(embedder.fingerprint)

    pages = corpus(
        documents=documents,
        chunks_per_document=chunks_per_document,
        changed_fraction=changed_fraction,
    )
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parse_fingerprints=parse_versions(lines="1"),
    )

    tracemalloc.start()
    started = time.perf_counter()
    await pipeline.run(fakes.DictConnector(pages))
    first = Measurement(
        label="first ingest",
        documents=documents,
        chunks=sum(embedder.batches),
        cache_hits=0,
        embedded=sum(embedder.batches),
        forward_calls=len(embedder.batches),
        seconds=time.perf_counter() - started,
        peak_mib=tracemalloc.get_traced_memory()[1] / (1024 * 1024),
    )
    tracemalloc.stop()

    indexed_chunks = first.chunks
    embedder.batches.clear()
    upgraded, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parsers={"lines": TrimmingLineParser() if changed_fraction else fakes.LineParser()},
        parse_fingerprints=parse_versions(lines="2"),
    )

    current: tuple[ParseFingerprint, ...] = (
        parse_versions(lines="2")("lines"),  # pyright: ignore[reportAssignmentType]
    )
    tracemalloc.start()
    started = time.perf_counter()
    sweep = await re_parse_stale(
        store=store, pipeline=upgraded, blobs=blobs, parse_fingerprints=current
    )
    second = Measurement(
        label="re-parse" if reuse else "re-parse (no reuse)",
        documents=sweep.reparsed,
        chunks=indexed_chunks,
        cache_hits=sweep.embedding.cache_hits,
        embedded=sum(embedder.batches),
        forward_calls=len(embedder.batches),
        seconds=time.perf_counter() - started,
        peak_mib=tracemalloc.get_traced_memory()[1] / (1024 * 1024),
    )
    tracemalloc.stop()

    if second.embedded != sweep.embedding.embedded:
        msg = (
            f"the report says {sweep.embedding.embedded} chunks were embedded and the embedder "
            f"was handed {second.embedded}. The whole point of this program is that those two "
            f"numbers are the same one, so a disagreement is a defect rather than a result."
        )
        raise AssertionError(msg)
    return first, second


def render(measurements: Sequence[Measurement]) -> str:
    """The passes side by side, so the comparison is the thing on the page.

    Plain text rather than a rich table, and sized from the content rather than the terminal:
    this output is meant to be pasted into a pull request or a design document, and a table
    that reflows to whatever width the author's window happened to be is not.
    """
    labels = [measurement.label for measurement in measurements]
    width = max(len(name) for name, _ in measurements[0].rows())
    columns = max(len(label) for label in labels) + 2
    lines = [f"{'':<{width}}  " + "".join(f"{label:>{columns}}" for label in labels)]
    for index, (name, _) in enumerate(measurements[0].rows()):
        cells = "".join(
            f"{measurement.rows()[index][1]:>{columns}}" for measurement in measurements
        )
        lines.append(f"{name:<{width}}  {cells}")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--chunks-per-document", type=int, default=1_000)
    parser.add_argument(
        "--changed-fraction",
        type=float,
        default=0.0,
        help="fraction of documents whose text version two actually moves",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="suppress durable reuse, reproducing the path ingest took before it existed",
    )
    arguments = parser.parse_args()

    out = console()
    total = arguments.documents * arguments.chunks_per_document
    if total <= CACHE_CAPACITY:
        out.print(
            f"[yellow]note:[/yellow] {total:,} chunks is not more than the embedding cache "
            f"holds ({CACHE_CAPACITY:,}), so a warm process could account for these numbers "
            f"without any durable reuse at all.\n"
        )

    first, second = await measure(
        documents=arguments.documents,
        chunks_per_document=arguments.chunks_per_document,
        changed_fraction=arguments.changed_fraction,
        reuse=not arguments.no_reuse,
    )
    out.print(render([first, second]))


if __name__ == "__main__":
    asyncio.run(main())
