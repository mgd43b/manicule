"""What counting checksum coverage costs on a Qdrant server, before and after the version index.

Counting coverage is what ``status`` and ``doctor`` ask of a network vector store, and it is an
exact count over a payload filter. A count with a filter on a field the collection has not
indexed reads every point's payload to answer, so its cost follows the corpus and the payload
size rather than the answer — a healthy corpus returns zero and pays for the whole scan. This
program measures that on a synthetic corpus, through three states of one collection, in the
order a real upgrade moves through them:

``baseline``
    No index on the checksum version, and the half-written pair counted as one filter: a
    ``should`` of two arms, each a condition with the other field's condition negated. Three
    count requests per coverage pass. This is every collection an earlier release made, asked
    the question the earlier release asked.

``shipped``
    The version index built, the same single filter. Three count requests per coverage pass.
    This is what the store does now, and the arm is the store's own ``checksum_coverage()``, not
    a copy of it.

``candidate-two-branch``
    The version index built and the half-written pair counted as the sum of two simple,
    disjoint filters. Four count requests per coverage pass, built here with the client
    directly. **It is not shipped.** It was the obvious alternative to the nested filter and was
    measured against it; it is kept so that comparison can be taken again, on another server
    version or another corpus, rather than trusted.

For each arm it reports the exact-count latency (p50 and p95) of the half-written count alone
and of a whole coverage pass, the time the index took to build, and how responsive the server
stays: a vector search and a ``get_collections`` — the request ``health()`` makes — sampled
while the server is idle and again while coverage passes run back to back in the background
(p50 and p99), and the same two requests sampled while the index is being built — the cost to
everyone else of the one prepare that adds it. The counts every arm returns are compared with
what was planted and the program refuses to report an arm that is wrong, so a speedup can never
be a different answer.

**What this does not measure**, and does not try to: CPU and memory. The server's process is
not this one. Each phase announces itself on standard error with an epoch timestamp and the name
of the collection it is using, and the background phase of each arm runs for at least
``--min-busy-seconds``, so a sampler such as ``docker stats`` running beside it can be lined up
with the window it belongs to.

**Everything it touches it made.** The program takes no collection name and has no way to be
given one. Each run makes one collection under the suite's ``manicule_test`` prefix, in a
workspace named with a fresh UUID, and removes what it made — the collection and its
workspace's own record — even when it fails. The shared record collection beside it is the
store's, kept by design, and is neither counted nor dropped. It never writes to or drops any
other collection.

Everything is synthetic: random words, random vectors, no corpus. Payloads are sized like real
chunks (``chunk_json`` of a few kilobytes, stored on disk the way the shipped shape stores them),
because a count that reads payloads costs what the payloads weigh and a benchmark over tiny ones
would flatter every arm.

Run it, against a throwaway server on a port nothing else uses::

    QDRANT_TEST_URL=http://localhost:16333 \\
        .venv/bin/python -m tests.benchmarks.qdrant_checksum_coverage
    QDRANT_TEST_URL=http://localhost:16333 \\
        .venv/bin/python -m tests.benchmarks.qdrant_checksum_coverage \\
        --points 200000 --dimension 1024 --json

Without ``QDRANT_TEST_URL`` it prints how to set it and exits without touching anything.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

from qdrant_client import AsyncQdrantClient, models

from manicule.core.embedding import VECTOR_CHECKSUM_VERSION, EmbedFingerprint, Pooling
from manicule.storage.qdrant import CollectionShape, QdrantVectorStore
from manicule.storage.vector_schema import (
    CHECKSUM_COLUMN,
    CHECKSUM_VERSION_COLUMN,
    CHUNK_COLUMN,
    CHUNK_ID_COLUMN,
    DOCUMENT_ID_COLUMN,
    ID_COLUMN,
    IDENTITY_COLUMN,
    KIND_COLUMN,
    LANG_COLUMN,
    POSITION_COLUMN,
    PUBLICATION_COLUMN,
)
from tests.qdrant_support import QDRANT_URL_ENV, TEST_COLLECTION_PREFIX, server_url

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ARMS: Final = ("baseline", "shipped", "candidate-two-branch")
"""The three states, in the order a collection moves through them."""

ARM_WIDTH: Final = 22
"""The width of the arm column, which has to hold the longest arm name and a space."""

WRITE_PAGE: Final = 250
"""Points per upsert. Setup cost, outside every measurement."""

KIND_PROSE: Final = "prose"
LANGUAGE: Final = "en"
LEGACY: Final = "legacy"
"""The publication a point belongs to when nothing has been published, which is every point here."""

_VOCABULARY: Final = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "sigma",
    "omega",
    "parser",
    "index",
    "vector",
    "chunk",
    "anchor",
    "corpus",
    "ledger",
    "source",
    "record",
    "segment",
    "storage",
    "payload",
    "filter",
    "query",
    "section",
    "heading",
    "table",
    "figure",
    "citation",
    "thread",
    "folder",
    "message",
)


@dataclass(frozen=True, slots=True)
class Truth:
    """What was planted, which is what every correct arm has to report."""

    rows: int
    unrecorded: int
    """Points recording neither half of the pair: the backfill's set."""

    half_written: int
    """Points recording exactly one half: reported as failed."""

    @property
    def recorded(self) -> int:
        return self.rows - self.unrecorded


@dataclass(frozen=True, slots=True)
class Answer:
    """What one coverage pass said, and how it asked."""

    rows: int
    recorded: int
    failed: int
    requests: int

    def matches(self, truth: Truth) -> bool:
        return (self.rows, self.recorded, self.failed) == (
            truth.rows,
            truth.recorded,
            truth.half_written,
        )


@dataclass(frozen=True, slots=True)
class Responsiveness:
    """Latency of two cheap requests, in milliseconds."""

    search_p50_ms: float
    search_p99_ms: float
    ping_p50_ms: float
    ping_p99_ms: float
    samples: int
    """Searches taken. Zero means none ran, and the search figures are then not measurements."""

    ping_samples: int
    """``get_collections`` requests taken, on the same terms."""


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One state of the collection, measured."""

    arm: str
    requests: int
    """Count requests in one coverage pass."""

    half_written_p50_ms: float
    half_written_p95_ms: float
    coverage_p50_ms: float
    coverage_p95_ms: float
    repeats: int
    idle: Responsiveness
    busy: Responsiveness
    background_passes: int
    """Coverage passes the background loop finished while the busy window was sampled."""

    busy_window_epoch: tuple[float, float]
    """Wall-clock start and end of the busy window, for lining a CPU/memory sampler up with it."""


@dataclass(frozen=True, slots=True)
class Run:
    """Everything one run measured."""

    collection: str
    points: int
    dimension: int
    payload_bytes_mean: int
    index_build_seconds: float
    during_index_build: Responsiveness
    """Search and ``get_collections`` latency sampled while the index was being built."""

    truth: Truth
    arms: tuple[ArmResult, ...]


_NO_SAMPLE: Final = Responsiveness(0.0, 0.0, 0.0, 0.0, samples=0, ping_samples=0)
"""What ``during_index_build`` says when no build ran, which cannot happen over the three arms."""


def percentile(values: Sequence[float], fraction: float) -> float:
    """The nearest-rank percentile: the smallest value at least ``fraction`` of them do not exceed.

    Nearest-rank rather than interpolated, so a p99 over a few hundred samples is a sample that
    happened and not a number between two that did.
    """
    if not values:
        raise ValueError("a percentile needs at least one sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def owned(name: str, minted: Sequence[str]) -> str:
    """``name``, if this run made it under the test prefix; otherwise a refusal.

    The one gate in front of anything destructive. A collection is touched only if this program
    minted it, and a name outside the test prefix is refused even if it somehow got on the list.
    """
    if not name.startswith(TEST_COLLECTION_PREFIX) or name not in minted:
        raise ValueError(f"refusing to touch {name!r}: this run did not make it")
    return name


def synthetic_text(generator: random.Random, size: int) -> str:
    """About ``size`` characters of random words, the shape of a chunk's text and no more."""
    words: list[str] = []
    length = 0
    while length < size:
        word = generator.choice(_VOCABULARY)
        words.append(word)
        length += len(word) + 1
    return " ".join(words)[:size]


def payload_for(
    index: int, generator: random.Random, *, kind: str, payload_bytes: tuple[int, int]
) -> dict[str, object]:
    """One point's payload, with ``kind`` deciding which halves of the checksum pair it records.

    ``kind`` is ``recorded`` (both), ``neither``, ``checksum-only`` or ``version-only``. The
    chunk text is sized between the two bounds, and everything else is what the store writes.
    """
    size = generator.randint(*payload_bytes)
    chunk_id = f"bench-chunk-{index:08d}"
    digest = hashlib.sha256(chunk_id.encode()).hexdigest()
    payload: dict[str, object] = {
        ID_COLUMN: f"{LEGACY}/{chunk_id}",
        CHUNK_ID_COLUMN: chunk_id,
        PUBLICATION_COLUMN: LEGACY,
        DOCUMENT_ID_COLUMN: f"bench-document-{index // 40:06d}",
        KIND_COLUMN: KIND_PROSE,
        LANG_COLUMN: LANGUAGE,
        POSITION_COLUMN: index % 40,
        CHUNK_COLUMN: json.dumps({"id": chunk_id, "text": synthetic_text(generator, size)}),
        IDENTITY_COLUMN: digest,
    }
    if kind in {"recorded", "checksum-only"}:
        payload[CHECKSUM_COLUMN] = digest
    if kind in {"recorded", "version-only"}:
        payload[CHECKSUM_VERSION_COLUMN] = VECTOR_CHECKSUM_VERSION
    return payload


def _records_nothing(field: str) -> models.Filter:
    return models.Filter(
        should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key=field)),
            models.FieldCondition(key=field, match=models.MatchValue(value="")),
        ]
    )


def _neither_filter() -> models.Filter:
    """Points recording neither half: what ``recorded`` is subtracted from the rows by."""
    return models.Filter(
        must=[_records_nothing(CHECKSUM_COLUMN), _records_nothing(CHECKSUM_VERSION_COLUMN)]
    )


def _nested_half_written_filter() -> models.Filter:
    """The single filter that counts a half-written pair, as the store builds it.

    Copied here for the ``baseline`` arm, which must run before the version index exists and so
    cannot be the store's own pass without being its own copy of nothing.
    """
    checksum = _records_nothing(CHECKSUM_COLUMN)
    version = _records_nothing(CHECKSUM_VERSION_COLUMN)
    return models.Filter(
        should=[
            models.Filter(must=[checksum], must_not=[version]),
            models.Filter(must=[version], must_not=[checksum]),
        ]
    )


def _branch_filters() -> tuple[models.Filter, models.Filter]:
    """The ``candidate-two-branch`` arm's filters: each kind of half-written pair on its own.

    Disjoint by construction, so the two counts add to what the nested filter counts.
    """
    checksum = _records_nothing(CHECKSUM_COLUMN)
    version = _records_nothing(CHECKSUM_VERSION_COLUMN)
    return (
        models.Filter(must=[checksum], must_not=[version]),
        models.Filter(must=[version], must_not=[checksum]),
    )


class _Timed:
    """Every ``count`` the client makes, timed, passed through to the server.

    Installed on the one client this run owns, so the store's own coverage pass can be timed
    request by request without a copy of what it asks.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        self.seconds: list[float] = []
        original = client.count

        async def timed(*args: Any, **kwargs: Any) -> models.CountResult:
            started = time.perf_counter()
            try:
                return await original(*args, **kwargs)
            finally:
                self.seconds.append(time.perf_counter() - started)

        setattr(client, "count", timed)  # noqa: B010 - a wrapper on this run's own client

    def take(self) -> list[float]:
        taken, self.seconds = self.seconds, []
        return taken


def _fingerprint(dimension: int) -> EmbedFingerprint:
    return EmbedFingerprint(
        model_id="benchmark/qdrant-checksum-coverage",
        dimension=dimension,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="benchmark/tokenizer",
        max_sequence_length=512,
    )


def _unit_vector(generator: random.Random, dimension: int) -> list[float]:
    raw = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


async def _settle(client: AsyncQdrantClient, collection: str, *, seconds: float) -> None:
    """Return once the optimizer is idle, so no arm is measured against a segment being built."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        info = await client.get_collection(collection)
        if info.status is models.CollectionStatus.GREEN:
            return
        await asyncio.sleep(0.25)
    raise TimeoutError(f"{collection} did not settle within {seconds:.0f} seconds")


async def _populate(
    client: AsyncQdrantClient,
    collection: str,
    *,
    points: int,
    dimension: int,
    planted_each: int,
    payload_bytes: tuple[int, int],
    seed: int,
) -> tuple[Truth, int]:
    """Write the corpus: mostly recorded, with a few planted points of each unhealthy kind.

    Mostly healthy because that is the case that was slow — a count that returns nothing and
    still reads everything. Returns what was planted and the mean payload size written.
    """
    generator = random.Random(seed)  # noqa: S311 - synthetic data, not keys
    kinds = ["checksum-only"] * planted_each + ["version-only"] * planted_each
    kinds += ["neither"] * planted_each
    spread = max(1, points // max(1, len(kinds)))
    planted = {index * spread + spread // 2: kind for index, kind in enumerate(kinds)}
    total_bytes = 0
    for start in range(0, points, WRITE_PAGE):
        batch: list[models.PointStruct] = []
        for index in range(start, min(points, start + WRITE_PAGE)):
            payload = payload_for(
                index,
                generator,
                kind=planted.get(index, "recorded"),
                payload_bytes=payload_bytes,
            )
            total_bytes += len(str(payload[CHUNK_COLUMN]))
            batch.append(
                models.PointStruct(
                    id=index + 1, vector=_unit_vector(generator, dimension), payload=payload
                )
            )
        await client.upsert(collection, points=batch, wait=True)
    truth = Truth(
        rows=points,
        unrecorded=planted_each,
        half_written=2 * planted_each,
    )
    return truth, total_bytes // max(1, points)


async def sample_latency(
    client: AsyncQdrantClient,
    collection: str,
    query: list[float],
    *,
    probes: int,
    interval: float,
    floor_seconds: float,
    until: asyncio.Future[Any] | None = None,
) -> Responsiveness:
    """Search and ``get_collections`` latency.

    Takes ``probes`` of each and keeps going for ``floor_seconds``; or, given ``until``, takes
    them for as long as that future is unfinished. With ``until`` it is checked before each
    probe starts, so a probe that began after the work finished is never counted as having run
    during it, and a probe type that never ran reports no sample rather than a figure.
    """
    searches: list[float] = []
    pings: list[float] = []
    started = time.monotonic()
    while True:
        if until is not None and until.done():
            break
        begin = time.perf_counter()
        await client.query_points(collection, query=query, limit=10, with_payload=False)
        searches.append((time.perf_counter() - begin) * 1000)
        if until is not None and until.done():
            break
        begin = time.perf_counter()
        await client.get_collections()
        pings.append((time.perf_counter() - begin) * 1000)
        await asyncio.sleep(interval)
        if (
            until is None
            and len(searches) >= probes
            and time.monotonic() - started >= floor_seconds
        ):
            break
    return Responsiveness(
        search_p50_ms=percentile(searches, 0.50) if searches else 0.0,
        search_p99_ms=percentile(searches, 0.99) if searches else 0.0,
        ping_p50_ms=percentile(pings, 0.50) if pings else 0.0,
        ping_p99_ms=percentile(pings, 0.99) if pings else 0.0,
        samples=len(searches),
        ping_samples=len(pings),
    )


async def _build_version_index(client: AsyncQdrantClient, collection: str) -> tuple[float, float]:
    """Create the version index and return when the request started and when it finished.

    Timed in here, around the one request, so the figure is the build's and not the build's
    plus whatever the sampler did around it.
    """
    started = time.perf_counter()
    await client.create_payload_index(
        collection,
        field_name=CHECKSUM_VERSION_COLUMN,
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    return started, time.perf_counter()


async def _measure_arm(
    arm: str,
    *,
    client: AsyncQdrantClient,
    store: QdrantVectorStore,
    collection: str,
    timer: _Timed,
    truth: Truth,
    query: list[float],
    repeats: int,
    probes: int,
    interval: float,
    min_busy_seconds: float,
    on_phase: Callable[[str, str, str], None],
) -> ArmResult:
    """Time one arm: its exact counts, then the server's responsiveness idle and under counting."""

    async def coverage_pass() -> Answer:
        if arm == "shipped":
            coverage = await store.checksum_coverage()
            return Answer(coverage.rows, coverage.recorded, coverage.failed, requests=3)
        # The two cheap requests the store makes before it counts, so the arms are timed doing
        # the same work around the counts.
        await store.fingerprint()
        await client.collection_exists(collection)
        rows = (await client.count(collection, exact=True)).count
        neither = (await client.count(collection, count_filter=_neither_filter(), exact=True)).count
        if arm == "candidate-two-branch":
            half = 0
            for branch in _branch_filters():
                half += (await client.count(collection, count_filter=branch, exact=True)).count
            return Answer(rows, rows - neither, half, requests=4)
        half = (
            await client.count(collection, count_filter=_nested_half_written_filter(), exact=True)
        ).count
        return Answer(rows, rows - neither, half, requests=3)

    first = await coverage_pass()  # a warm-up the figures below do not include
    timer.take()
    if not first.matches(truth):
        raise AssertionError(f"arm {arm!r} counted {first}, but {truth} was planted")

    on_phase(arm, "counting", collection)
    half_written: list[float] = []
    whole: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        answer = await coverage_pass()
        whole.append((time.perf_counter() - started) * 1000)
        per_request = timer.take()
        # The first two requests are the rows and the pair; the rest count the half-written pair.
        half_written.append(sum(per_request[2:]) * 1000)
        if not answer.matches(truth):
            raise AssertionError(f"arm {arm!r} counted {answer}, but {truth} was planted")

    on_phase(arm, "idle", collection)
    idle = await sample_latency(
        client, collection, query, probes=probes, interval=interval, floor_seconds=0.0
    )

    on_phase(arm, "busy", collection)
    stop = asyncio.Event()
    passes = 0

    async def background() -> None:
        nonlocal passes
        while not stop.is_set():
            await coverage_pass()
            passes += 1
            # A request that completes without suspending would starve the sampler; over HTTP
            # every request suspends, and this is for the in-process engine the tests may use.
            await asyncio.sleep(0)

    task = asyncio.create_task(background())
    window_start = time.time()
    try:
        busy = await sample_latency(
            client,
            collection,
            query,
            probes=probes,
            interval=interval,
            floor_seconds=min_busy_seconds,
        )
    finally:
        stop.set()
        await task
    window_end = time.time()
    timer.take()
    on_phase(arm, "done", collection)
    return ArmResult(
        arm=arm,
        requests=first.requests,
        half_written_p50_ms=percentile(half_written, 0.50),
        half_written_p95_ms=percentile(half_written, 0.95),
        coverage_p50_ms=percentile(whole, 0.50),
        coverage_p95_ms=percentile(whole, 0.95),
        repeats=repeats,
        idle=idle,
        busy=busy,
        background_passes=passes,
        busy_window_epoch=(window_start, window_end),
    )


async def _remove(
    client: AsyncQdrantClient, store: QdrantVectorStore, collection: str, minted: list[str]
) -> None:
    """Remove what this run made and nothing else. Safe to call after any failure."""
    if await client.collection_exists(owned(collection, minted)):
        await client.delete_collection(collection)
    # Its own record in the shared `<prefix>_meta` collection goes with it. The shared collection
    # stays, as the store keeps it by design: it is not this run's to count or drop.
    await store.reset_storage()


async def measure(
    *,
    url: str,
    points: int,
    dimension: int,
    payload_bytes: tuple[int, int] = (1024, 8192),
    planted_each: int = 3,
    repeats: int = 7,
    probes: int = 100,
    interval: float = 0.01,
    min_busy_seconds: float = 5.0,
    request_timeout_s: int = 300,
    settle_seconds: float = 600.0,
    seed: int = 20260101,
    on_phase: Callable[[str, str, str], None] | None = None,
) -> Run:
    """Run the three arms over one synthetic collection and return what was measured.

    ``on_phase(arm, phase, collection)`` is called as each phase starts, which is the hook a
    CPU and memory sampler is lined up with. The collection is created, populated, measured and
    removed in here, and nothing outside this call can name it.
    """
    announce = on_phase or _silent
    if repeats < 1:
        raise ValueError(
            "repeats must be at least one, or there is nothing to take a percentile of"
        )
    if points < 12 * planted_each:
        raise ValueError("too few points to plant every kind of unhealthy point in")
    # Not the suite's shared helper, whose timeout is thirty seconds: a baseline count over a
    # large collection is expected to exceed that, and a benchmark that dies on the very thing
    # it measures reports nothing.
    client = AsyncQdrantClient(url=url, timeout=request_timeout_s, check_compatibility=False)
    workspace = f"benchmark-{uuid.uuid4()}"
    store = QdrantVectorStore(
        client,
        workspace_id=workspace,
        collection_prefix=TEST_COLLECTION_PREFIX,
        shape=CollectionShape(on_disk_payload=True),
    )
    fingerprint = _fingerprint(dimension)
    collection = store.storage_name(fingerprint)
    minted = [collection]
    timer = _Timed(client)
    try:
        await store.ensure_ready(fingerprint)
        announce("setup", "created", collection)
        # The state every existing collection is in: made by a release that did not index the
        # version. `ensure_ready` just indexed it, so the index is taken away again.
        await client.delete_payload_index(collection, field_name=CHECKSUM_VERSION_COLUMN, wait=True)
        truth, mean_bytes = await _populate(
            client,
            collection,
            points=points,
            dimension=dimension,
            planted_each=planted_each,
            payload_bytes=payload_bytes,
            seed=seed,
        )
        await _settle(client, collection, seconds=settle_seconds)
        query = _unit_vector(random.Random(seed + 1), dimension)  # noqa: S311 - synthetic
        schema = (await client.get_collection(collection)).payload_schema or {}
        if CHECKSUM_VERSION_COLUMN in schema:
            raise AssertionError("the baseline collection still has the version index")

        results: list[ArmResult] = []
        build_seconds = 0.0
        during_build: Responsiveness | None = None
        for arm in ARMS:
            if arm == "shipped":
                announce(arm, "index-build", collection)
                # The build is one request that returns when the index exists. Search and
                # `get_collections` are sampled beside it, which is what an upgrade's first
                # prepare does to everyone else using the server.
                build = asyncio.ensure_future(_build_version_index(client, collection))
                try:
                    during_build = await sample_latency(
                        client,
                        collection,
                        query,
                        probes=probes,
                        interval=interval,
                        floor_seconds=0.0,
                        until=build,
                    )
                finally:
                    built_from, built_to = await build
                build_seconds = built_to - built_from
                await _settle(client, collection, seconds=settle_seconds)
            results.append(
                await _measure_arm(
                    arm,
                    client=client,
                    store=store,
                    collection=collection,
                    timer=timer,
                    truth=truth,
                    query=query,
                    repeats=repeats,
                    probes=probes,
                    interval=interval,
                    min_busy_seconds=min_busy_seconds,
                    on_phase=announce,
                )
            )
        return Run(
            collection=collection,
            points=points,
            dimension=dimension,
            payload_bytes_mean=mean_bytes,
            index_build_seconds=build_seconds,
            during_index_build=during_build or _NO_SAMPLE,
            truth=truth,
            arms=tuple(results),
        )
    finally:
        try:
            await _remove(client, store, collection, minted)
        finally:
            await client.close()


def render(run: Run) -> str:
    """A table that fits a terminal, for pasting into a pull request."""
    header = (
        f"{run.points} points, {run.dimension} dimensions, mean chunk_json "
        f"{run.payload_bytes_mean} bytes, index build {run.index_build_seconds:.2f} s"
    )
    build = run.during_index_build
    lines = [
        header,
        f"during the index build ({build.samples} searches, {build.ping_samples} pings), ms: "
        f"search p50 {build.search_p50_ms:.1f} p99 {build.search_p99_ms:.1f}, "
        f"ping p50 {build.ping_p50_ms:.1f} p99 {build.ping_p99_ms:.1f}",
        f"{'arm':<{ARM_WIDTH}}{'req':>4}{'half p50':>10}{'half p95':>10}"
        f"{'pass p50':>10}{'pass p95':>10}{'srch p50':>10}{'srch p99':>10}"
        f"{'ping p50':>10}{'ping p99':>10}",
        f"{'':<{ARM_WIDTH}}{'':>4}{'ms':>10}{'ms':>10}{'ms':>10}{'ms':>10}{'ms':>10}{'ms':>10}{'ms':>10}{'ms':>10}",
    ]
    for result in run.arms:
        for label, sample in (("idle", result.idle), ("busy", result.busy)):
            lines.append(
                f"{result.arm:<{ARM_WIDTH}}{result.requests:>4}"
                f"{result.half_written_p50_ms:>10.1f}{result.half_written_p95_ms:>10.1f}"
                f"{result.coverage_p50_ms:>10.1f}{result.coverage_p95_ms:>10.1f}"
                f"{sample.search_p50_ms:>10.1f}{sample.search_p99_ms:>10.1f}"
                f"{sample.ping_p50_ms:>10.1f}{sample.ping_p99_ms:>10.1f}  {label}"
            )
    baseline, shipped, candidate = run.arms
    if shipped.coverage_p50_ms > 0 and shipped.half_written_p50_ms > 0:
        lines.append(
            f"baseline over shipped, p50: coverage pass "
            f"{baseline.coverage_p50_ms / shipped.coverage_p50_ms:.1f}x, half-written count "
            f"{baseline.half_written_p50_ms / shipped.half_written_p50_ms:.1f}x (this run only)"
        )
    if candidate.coverage_p50_ms > 0 and candidate.half_written_p50_ms > 0:
        lines.append(
            f"shipped over candidate-two-branch, p50: coverage pass "
            f"{shipped.coverage_p50_ms / candidate.coverage_p50_ms:.2f}x, half-written count "
            f"{shipped.half_written_p50_ms / candidate.half_written_p50_ms:.2f}x (this run only)"
        )
    return "\n".join(lines)


def _silent(arm: str, phase: str, collection: str) -> None:
    """The default hook: nothing is announced."""
    del arm, phase, collection


def _announce(arm: str, phase: str, collection: str) -> None:
    print(  # noqa: T201 - benchmark output
        f"PHASE arm={arm} phase={phase} epoch={time.time():.3f} collection={collection}",
        file=sys.stderr,
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    """The command line. There is deliberately no option that names a collection."""
    built = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    built.add_argument("--points", type=int, default=20_000, help="points to write (default 20000)")
    built.add_argument("--dimension", type=int, default=128)
    built.add_argument("--min-payload-bytes", type=int, default=1024)
    built.add_argument("--max-payload-bytes", type=int, default=8192)
    built.add_argument("--planted-each", type=int, default=3)
    built.add_argument("--repeats", type=int, default=7, help="exact-count passes per arm")
    built.add_argument(
        "--probes", type=int, default=100, help="search samples per responsiveness phase"
    )
    built.add_argument("--min-busy-seconds", type=float, default=5.0)
    built.add_argument("--timeout", type=int, default=300, help="client request timeout, seconds")
    built.add_argument("--json", action="store_true")
    return built


async def _main(arguments: argparse.Namespace) -> int:
    url = server_url()
    if url is None:
        print(  # noqa: T201 - benchmark output
            f"{QDRANT_URL_ENV} is not set, so there is no Qdrant server to measure. Point it at a "
            f"throwaway server; this program creates and removes only its own "
            f"{TEST_COLLECTION_PREFIX}-prefixed collection.",
            file=sys.stderr,
        )
        return 2
    run = await measure(
        url=url,
        points=arguments.points,
        dimension=arguments.dimension,
        payload_bytes=(arguments.min_payload_bytes, arguments.max_payload_bytes),
        planted_each=arguments.planted_each,
        repeats=arguments.repeats,
        probes=arguments.probes,
        min_busy_seconds=arguments.min_busy_seconds,
        request_timeout_s=arguments.timeout,
        on_phase=_announce,
    )
    if arguments.json:
        print(json.dumps(asdict(run), sort_keys=True))  # noqa: T201 - benchmark output
    else:
        print(render(run))  # noqa: T201 - benchmark output
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main(parser().parse_args())))


if __name__ == "__main__":
    main()
