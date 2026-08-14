#!/usr/bin/env python3
"""Measure the staged connector sync against the sequential loop it replaced.

**Synthetic timings, and they are not a throughput claim.** Every delay here is an
``asyncio.sleep`` standing in for a network round trip, a parser and a model, chosen so that the
*shape* of the pipeline is visible: the speedup a real corpus sees depends on a real source's
latency and a real model's batch time, and quoting a number from this as production throughput
would be quoting the sleep. What the numbers are good for is proving that the stages overlap,
that the bounds hold while they do, and that the scaling follows the configuration rather than
the machine.

**The baseline is the loop that was there**, not the staged pipeline configured narrow. It
drives ``IngestPipeline.ingest`` over ``Connector.discover`` one document at a time, which is
what ``run`` did before the stages existed — so the comparison is against real code rather than
against a version of the new code with its knobs turned down.

Three scenarios, because they cost entirely different things and a single number hides it:

``first``
    Nothing stored. Every document is fetched, parsed, chunked and embedded.

``unchanged``
    Everything stored and nothing moved. Level-1 change detection answers before the fetch, so
    the run costs one store read per document and touches neither the source nor the model.

``partial``
    A tenth of the corpus edited. The realistic steady state, and the one where the fetch stage
    still has to visit every document while the embedder only sees a few.

Usage::

    uv run tools/benchmark_ingest_stages.py
    uv run tools/benchmark_ingest_stages.py --documents 200 --fetch-concurrency 8
    uv run tools/benchmark_ingest_stages.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

if __package__ is None:  # pragma: no cover - the script form, run from a checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests.fakes import HashEmbedder
from tests.ingest import fakes

from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline, RunReport
from manicule.ingest.workers import InProcessRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import RawDocument
    from manicule.core.embedding import Vector
    from manicule.core.protocols import Connector
    from manicule.core.sources import DocRef


class DelayedConnector(fakes.ObservedConnector):
    """A source with a round trip, so a fetch is something worth overlapping.

    The delay is the only thing separating this from an instant dictionary, and it is the whole
    reason the benchmark says anything: with an instant source a sequential loop and a staged one
    do the same work in the same order and the difference is scheduling noise.
    """

    def __init__(self, documents: dict[str, str], *, latency_s: float) -> None:
        super().__init__(documents)
        self._latency_s = latency_s

    @override
    async def fetch(self, ref: DocRef) -> RawDocument:
        async with self.fetching.holding():
            await asyncio.sleep(self._latency_s)
            return await fakes.DictConnector.fetch(self, ref)


class DelayedRunner:
    """A parse runner that takes time, standing in for the worker pool.

    A real pool costs a pickle across a pipe and a subprocess's parse; what matters to the shape
    is that an attempt occupies a worker for a while, so several documents in the ingest stage
    keep several workers busy.
    """

    def __init__(self, parsers: dict[str, object], *, parse_s: float) -> None:
        self._inner = InProcessRunner(parsers)
        self._parse_s = parse_s
        self.gate = fakes.Gate(opened=True)

    async def run_attempt(self, name: str, raw: RawDocument) -> object:
        async with self.gate.holding():
            await asyncio.sleep(self._parse_s)
            return await self._inner.run_attempt(name, raw)


class DelayedEmbedder(fakes.ExclusiveEmbedder):
    """A model that takes time per batch and refuses to be called twice at once.

    Per *batch* rather than per chunk, because that is what a forward pass costs and it is why
    batching is worth anything. Refusing overlap keeps the benchmark honest: a run that got its
    speedup by issuing concurrent model calls would fail here rather than post a good number.
    """

    def __init__(self, *, embed_s: float) -> None:
        super().__init__()
        self._embed_s = embed_s
        self.peak_inside = 0
        """The most calls inside the model at once. One, or the benchmark has already raised.

        Measured here rather than read off the run report so that the sequential baseline is
        measured the same way as the staged run — a comparison where one side's number comes
        from a gauge the other side does not have is not a comparison.
        """

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self.inside:
            self.overlaps += 1
            msg = "the benchmark's embedder was called concurrently"
            raise AssertionError(msg)
        self.inside += 1
        self.peak_inside = max(self.peak_inside, self.inside)
        try:
            self.batches.append(len(texts))
            await asyncio.sleep(self._embed_s)
            return await HashEmbedder.embed(self, texts)
        finally:
            self.inside -= 1


@dataclass(frozen=True, slots=True)
class Measurement:
    """One scenario, one strategy, and everything worth reading afterwards."""

    scenario: str
    strategy: str
    seconds: float
    discovered: int
    indexed: int
    skipped: int
    peak_fetches: int
    peak_parses: int
    peak_embeds: int
    fetch_queue_peak: int
    parse_queue_peak: int
    discovery_waits: int
    """Times discovery found a hand-off full and had to wait. Zero for the sequential loop,
    which has no hand-off to fill — and a number above zero is the only direct evidence that
    backpressure reached the source rather than merely being configured."""

    embed_batches: int
    embed_chunks: int
    peak_retained_documents: int
    """Fetched bodies alive at once. The bounded-memory proxy, and a better one than resident
    bytes: it is exactly the quantity the hand-off capacities bound, it is deterministic, and it
    does not move when the garbage collector does."""

    peak_traced_kib: int
    """Python allocations at their high-water mark, from ``tracemalloc``. Kept beside the
    document count rather than instead of it: it is the number an operator recognizes, and it is
    the one that varies with everything except the design."""

    @property
    def batch_utilization(self) -> float:
        """Chunks per forward pass, against the largest batch the configuration allows."""
        return (self.embed_chunks / self.embed_batches) if self.embed_batches else 0.0


TOO_SMALL_TO_COMPARE_S = 0.05
"""Below this, a wall-time ratio says more about task setup than about the pipeline."""


def corpus(count: int) -> dict[str, str]:
    """Invented documents, eight lines each, so a document is eight chunks."""
    return {
        f"page-{number:04d}": "\n".join(
            f"Paragraph {line} of invented page {number}." for line in range(8)
        )
        for number in range(count)
    }


def _pipeline(
    store: fakes.MemoryIngestStore,
    vectors: fakes.MemoryVectors,
    embedder: DelayedEmbedder,
    runner: DelayedRunner,
    *,
    fetch_concurrency: int,
    parse_workers: int,
) -> IngestPipeline:
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=embedder,
        vectors=vectors,
        runner=runner,  # pyright: ignore[reportArgumentType] - satisfies ParseRunner structurally
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=fetch_concurrency,
        parse_workers=parse_workers,
        queue_depth_factor=2,
    )


async def _sequentially(pipeline: IngestPipeline, connector: Connector) -> RunReport:
    """The loop that was there: one document's whole path, then the next.

    Written out rather than approximated by narrowing the staged run's knobs, because the point
    of a baseline is that it is the thing being compared against.
    """
    report = RunReport(connector=connector.name)
    stream = connector.discover(None)
    async for discovered in stream:
        for position, outcome in enumerate(await pipeline.ingest(connector, discovered)):
            report.record(outcome, expanded=position > 0)
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        await closer()
    return report


async def measure(
    scenario: str,
    strategy: str,
    *,
    documents: int,
    fetch_concurrency: int,
    parse_workers: int,
    latency_s: float,
    parse_s: float,
    embed_s: float,
) -> Measurement:
    """Run one scenario one way, and report what happened rather than only how long it took."""
    pages = corpus(documents)
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    embedder = DelayedEmbedder(embed_s=embed_s)
    runner = DelayedRunner({"lines": fakes.LineParser()}, parse_s=parse_s)
    pipeline = _pipeline(
        store,
        vectors,
        embedder,
        runner,
        fetch_concurrency=fetch_concurrency,
        parse_workers=parse_workers,
    )
    connector = DelayedConnector(pages, latency_s=latency_s)

    if scenario in {"unchanged", "partial"}:
        # Warm the index without charging the measured run for it. The staged run is used for
        # the warm-up whichever strategy is being measured, so both start from one corpus.
        await pipeline.run(connector)
        if scenario == "partial":
            for number in range(0, documents, 10):
                source_id = f"page-{number:04d}"
                connector.documents[source_id] += "\nA paragraph somebody added."

    embedder.batches.clear()
    embedder.peak_inside = 0
    runner.gate.peak = 0
    connector.fetching.peak = 0

    tracemalloc.start()
    started = time.perf_counter()
    if strategy == "sequential":
        report = await _sequentially(pipeline, connector)
    else:
        report = await pipeline.run(connector)
    elapsed = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Measurement(
        scenario=scenario,
        strategy=strategy,
        seconds=elapsed,
        discovered=report.discovered,
        indexed=report.indexed,
        skipped=report.skipped_version + report.skipped_hash,
        peak_fetches=connector.fetching.peak,
        peak_parses=runner.gate.peak,
        peak_embeds=embedder.peak_inside,
        fetch_queue_peak=report.stages.fetch_queue.peak_depth,
        parse_queue_peak=report.stages.parse_queue.peak_depth,
        discovery_waits=report.stages.fetch_queue.blocked_puts,
        embed_batches=len(embedder.batches),
        embed_chunks=sum(embedder.batches),
        # The sequential loop holds exactly one fetched body, by construction: it has no queue
        # to put one in. Stated rather than measured, because there is no gauge on that path.
        peak_retained_documents=report.stages.peak_bodies if strategy == "staged" else 1,
        peak_traced_kib=traced_peak // 1024,
    )


def _table(rows: list[Measurement]) -> str:
    header = (
        f"{'scenario':<10} {'strategy':<11} {'seconds':>8} {'fetch':>6} {'parse':>6} "
        f"{'embed':>6} {'fq':>4} {'pq':>4} {'waits':>6} {'batches':>8} {'per':>5} "
        f"{'bodies':>7} {'KiB':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.scenario:<10} {row.strategy:<11} {row.seconds:>8.3f} {row.peak_fetches:>6} "
            f"{row.peak_parses:>6} {row.peak_embeds:>6} {row.fetch_queue_peak:>4} "
            f"{row.parse_queue_peak:>4} {row.discovery_waits:>6} {row.embed_batches:>8} "
            f"{row.batch_utilization:>5.1f} {row.peak_retained_documents:>7} "
            f"{row.peak_traced_kib:>8}"
        )
    speedups: list[str] = []
    for scenario in ("first", "unchanged", "partial"):
        pair = {row.strategy: row.seconds for row in rows if row.scenario == scenario}
        if len(pair) != 2 or pair["staged"] <= 0:  # pragma: no cover - every scenario runs both
            continue
        # A ratio between two millisecond measurements is a measurement of task setup. An
        # unchanged sync does no fetch and no forward pass, so at any ordinary corpus size it
        # lands here — and printing "0.3x" beside the other two without saying so invites
        # exactly the wrong conclusion, which is that the stages made something slower.
        if max(pair.values()) < TOO_SMALL_TO_COMPARE_S:
            speedups.append(
                f"{scenario}: both under {TOO_SMALL_TO_COMPARE_S * 1000:.0f} ms, "
                "so the ratio is scheduling noise"
            )
            continue
        speedups.append(f"{scenario}: {pair['sequential'] / pair['staged']:.1f}x")
    lines.append("")
    lines.append("staged against sequential — " + "; ".join(speedups))
    lines.append(
        "fq/pq: peak depth of the fetch and parse hand-offs. waits: times discovery found one "
        "full.\nper: chunks per forward pass. bodies: fetched documents alive at once."
    )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=60)
    parser.add_argument("--fetch-concurrency", type=int, default=6)
    parser.add_argument("--parse-workers", type=int, default=3)
    parser.add_argument("--latency-ms", type=float, default=10.0, help="One fetch's round trip.")
    parser.add_argument("--parse-ms", type=float, default=4.0, help="One parse attempt.")
    parser.add_argument("--embed-ms", type=float, default=3.0, help="One forward pass.")
    parser.add_argument("--json", action="store_true", help="Machine-readable, for a diff.")
    arguments = parser.parse_args()

    rows: list[Measurement] = []
    for scenario in ("first", "unchanged", "partial"):
        for strategy in ("sequential", "staged"):
            rows.append(
                await measure(
                    scenario,
                    strategy,
                    documents=arguments.documents,
                    fetch_concurrency=arguments.fetch_concurrency,
                    parse_workers=arguments.parse_workers,
                    latency_s=arguments.latency_ms / 1000,
                    parse_s=arguments.parse_ms / 1000,
                    embed_s=arguments.embed_ms / 1000,
                )
            )

    if arguments.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
