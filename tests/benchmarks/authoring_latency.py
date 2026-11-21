"""How long ``document_create`` takes while an unrelated source is syncing.

A program, not a test: ``python -m tests.benchmarks.authoring_latency``. It builds a throwaway
runtime over a temporary directory, with the shipped pipeline, SQLite and the embedded vector
store and a deterministic hash embedder, so what it measures is everything in a write *except*
a model: the derived-index barrier, the pipeline, the stores and the membership write. It
touches nothing outside its own temporary directory and needs no server.

Each case is run against two barriers, **legacy** (every run and every publication takes the
single exclusive side, which is what the process had before source leases) and **leased** (the
shipped behavior), and in three conditions:

* ``idle`` — nothing else is running;
* ``parked`` — an unrelated source is parked inside its fetch for ``--park-seconds`` and then
  released, which is a sync waiting on a server. The legacy barrier makes a write wait out the
  park; the leased one does not;
* ``waiting`` — the same, with a maintenance task queued behind the parked source, which is the
  documented cost: a waiting maintenance task closes admission, so a write queues behind it.

``cold`` is the first write on a fresh runtime (it builds the pipeline and prepares the vector
store); ``warm`` is every write after it. They are reported separately because a served process
pays the first once. Wall-clock numbers describe this machine, this embedder and this synthetic
corpus. They show the barrier's effect and are not a latency estimate for any deployment.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, override

if TYPE_CHECKING:
    from collections.abc import Sequence

AUTHORED: Final = "authored"
OTHER: Final = "mail"
BARRIERS: Final = ("legacy", "leased")
CONDITIONS: Final = ("idle", "parked", "waiting")


@dataclass(frozen=True, slots=True)
class Sample:
    """One write: how long it took in all, and how much of that was waiting for the barrier."""

    total_ms: float
    barrier_wait_ms: float


@dataclass(frozen=True, slots=True)
class Result:
    barrier: str
    condition: str
    cold: tuple[Sample, ...]
    warm: tuple[Sample, ...]


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; ``0.0`` for no values so an empty column prints as zero."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[rank]


def _runtime(directory: Path) -> Any:
    from manicule.app.runtime import Runtime  # noqa: PLC0415
    from manicule.config.settings import (  # noqa: PLC0415
        AuthoringSettings,
        ConnectorSettings,
        Settings,
    )
    from manicule.container import keys  # noqa: PLC0415
    from manicule.plugins.registry import discover  # noqa: PLC0415
    from tests.fakes import HashEmbedder  # noqa: PLC0415
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415

    found = discover()
    bound = found.registry.bind("benchmark")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    root = directory / AUTHORED
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        data_dir=directory / "data",
        workspace="benchmark",
        connectors={AUTHORED: ConnectorSettings(type="filesystem", options={"root": str(root)})},
        authoring=AuthoringSettings(source=AUTHORED, collections=("synthetic",)),
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block"},  # pyright: ignore[reportArgumentType]
        ingest={"shutdown_grace_s": 0.5},  # pyright: ignore[reportArgumentType]
    )
    return Runtime(settings, discovery=found)


def _as_legacy(runtime: Any) -> None:
    """Make the runtime behave as it did before source leases: one exclusive lock for all."""

    def exclusive(_source: str) -> Any:
        return runtime.derived_mutation_guard()

    def never_built(_slot: str) -> None:
        return None

    runtime.derived_source_lease = exclusive
    runtime.derived_publication_guard = exclusive
    runtime._built = never_built


def _timed_waits(runtime: Any, waited: list[float]) -> None:
    """Record how long each lease or exclusive acquisition waited before it was admitted."""
    for name in ("derived_source_lease", "derived_publication_guard"):
        original = getattr(runtime, name)

        def wrapped(source: str, original: Any = original) -> Any:
            return _Timed(original(source), waited)

        setattr(runtime, name, wrapped)
    exclusive = runtime.derived_mutation_guard

    def wrapped_exclusive() -> Any:
        return _Timed(exclusive(), waited)

    runtime.derived_mutation_guard = wrapped_exclusive


class _Timed:
    """An async context manager that records how long entering the wrapped one took."""

    def __init__(self, inner: Any, waited: list[float]) -> None:
        self._inner = inner
        self._waited = waited

    async def __aenter__(self) -> Any:
        start = time.perf_counter()
        try:
            return await self._inner.__aenter__()
        finally:
            self._waited.append((time.perf_counter() - start) * 1000.0)

    async def __aexit__(self, *exc: object) -> Any:
        return await self._inner.__aexit__(*exc)


async def _park(runtime: Any, gate: asyncio.Event, entered: asyncio.Event) -> Any:
    """A sync of an unrelated source, parked in its fetch until ``gate`` is set."""
    from tests.ingest import fakes  # noqa: PLC0415

    class Parked(fakes.DictConnector):
        @override
        async def fetch(self, ref: Any) -> Any:
            raw = await super().fetch(ref)
            entered.set()
            await gate.wait()
            return raw

    connector = Parked({"page": "an unrelated page"}, name=OTHER)
    connector.media_types["page"] = "text/plain"
    async with runtime.leased_pipeline(OTHER) as pipeline:
        return await pipeline.run(connector)


async def _once(
    barrier: str, condition: str, *, writes: int, park_s: float, directory: Path
) -> tuple[Sample, ...]:
    from manicule.app.service import ApplicationService  # noqa: PLC0415

    async with _runtime(directory) as runtime:
        service = ApplicationService(runtime)
        await service.collection_create("synthetic")
        waited: list[float] = []
        if barrier == "legacy":
            _as_legacy(runtime)
        _timed_waits(runtime, waited)

        parked: asyncio.Task[Any] | None = None
        maintenance: asyncio.Task[None] | None = None
        gate, entered = asyncio.Event(), asyncio.Event()
        release_maintenance = asyncio.Event()
        if condition != "idle":
            # Built before it is parked: a source that is parked must already have the pipeline
            # it runs on, as a served sync does, and building it here keeps the build out of
            # the park window that the write is measured against.
            await runtime.pipeline()
            parked = asyncio.create_task(_park(runtime, gate, entered))
            await entered.wait()
            if condition == "waiting":

                async def maintenance_task() -> None:
                    async with runtime.derived_mutation_guard():
                        await release_maintenance.wait()

                maintenance = asyncio.create_task(maintenance_task())
                await asyncio.sleep(0.05)

            async def release() -> None:
                await asyncio.sleep(park_s)
                gate.set()
                release_maintenance.set()

            releasing = asyncio.create_task(release())
        else:
            releasing = None

        samples: list[Sample] = []
        for number in range(writes):
            waited.clear()
            start = time.perf_counter()
            await service.document_create(
                collection="synthetic",
                slug=f"note-{number}",
                body=f"synthetic note number {number} about harbors and tides\n",
            )
            samples.append(Sample((time.perf_counter() - start) * 1000.0, sum(waited)))
        for task in (releasing, parked, maintenance):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        return tuple(samples)


async def measure(
    *, writes: int, cold_runs: int, park_s: float, conditions: Sequence[str] = CONDITIONS
) -> list[Result]:
    """Every barrier in every condition: ``cold_runs`` fresh runtimes of ``writes`` writes each."""
    results: list[Result] = []
    for condition in conditions:
        for barrier in BARRIERS:
            cold: list[Sample] = []
            warm: list[Sample] = []
            for _ in range(cold_runs):
                with tempfile.TemporaryDirectory(prefix="manicule-authoring-") as directory:
                    samples = await _once(
                        barrier, condition, writes=writes, park_s=park_s, directory=Path(directory)
                    )
                cold.append(samples[0])
                warm.extend(samples[1:])
            results.append(Result(barrier, condition, tuple(cold), tuple(warm)))
    return results


def render(results: Sequence[Result]) -> str:
    """A table: one row per barrier and condition, cold and warm columns in milliseconds."""
    header = (
        f"{'condition':<9} {'barrier':<7} | {'cold p50':>9} {'cold p95':>9} "
        f"| {'warm p50':>9} {'warm p95':>9} {'warm wait p50':>14} {'n':>5}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        cold = [sample.total_ms for sample in result.cold]
        warm = [sample.total_ms for sample in result.warm]
        waits = [sample.barrier_wait_ms for sample in result.warm]
        lines.append(
            f"{result.condition:<9} {result.barrier:<7} | "
            f"{statistics.median(cold) if cold else 0.0:>9.1f} {percentile(cold, 0.95):>9.1f} "
            f"| {statistics.median(warm) if warm else 0.0:>9.1f} {percentile(warm, 0.95):>9.1f} "
            f"{statistics.median(waits) if waits else 0.0:>14.1f} {len(warm):>5}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--writes", type=int, default=6, help="writes per fresh runtime")
    parser.add_argument("--cold-runs", type=int, default=5, help="fresh runtimes per case")
    parser.add_argument("--park-seconds", type=float, default=2.0, help="how long the sync parks")
    arguments = parser.parse_args(argv)
    results = asyncio.run(
        measure(
            writes=arguments.writes, cold_runs=arguments.cold_runs, park_s=arguments.park_seconds
        )
    )
    print(render(results))  # noqa: T201 - this program's output is the table
    return 0


if __name__ == "__main__":
    sys.exit(main())
