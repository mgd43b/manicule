"""When dense search stops being exhaustive, and what it becomes instead.

``docs/storage.md`` §6.2 states the transition: exhaustive cosine search below
``ann_index_threshold``, an IVF-PQ index with ``num_partitions ≈ sqrt(n)`` above it. This
module is the only place that rule is written down, in the same spirit as
:func:`~manicule.core.embedding.classify_stored_vector` — the decision is shared by a store
that acts on it, a maintenance boundary that performs it and a status surface that reports it,
and three copies of a threshold comparison is three chances for them to disagree about what an
index *is*.

Nothing here imports LanceDB. The classification is a pure function of three numbers the
caller already has — the configured threshold, the row count, and what the store reports
about the index it holds — which is what lets both the storage layer and the application
layer speak about the same state without one of them depending on the other.

**Exhaustive is not a degraded mode.** It is exact, and below a few tens of thousands of
vectors it is also fast. An IVF-PQ index built too early trades recall — permanently, and
invisibly — for latency nobody was waiting on. So :attr:`AnnLifecycle.EXHAUSTIVE` is a
healthy state and reads as one on every surface; the state that deserves attention is
:attr:`AnnLifecycle.PENDING`, where the corpus has grown past the point the threshold names
and the scan is still linear.

**The index describes itself.** Everything reported about a built index comes from the index:
its type, metric and row coverage from LanceDB's own statistics, and the two facts LanceDB
does not record — which build produced it, and how many partitions it was trained with —
from the name it was created under (:func:`ann_index_name`). This is the same choice
``docs/storage.md`` §6.5 makes for ``chunks__<fp8>``: a derived artifact that carries its own
identity cannot drift from a second record of what it should be, because there is no second
record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from math import isqrt
from typing import Final

PQ_CODE_BITS: Final = 8
"""Bits per PQ code, passed to every build rather than left to the library's default.

Stated because :data:`PQ_CODEBOOK_ROWS` is derived from it. A default that moved underneath us
would move the number of rows a build needs while the floor this module validates against
stayed put — and the two failures that produces are a threshold that can never be satisfied and
a threshold that lets a doomed build through. Passing it makes them one number.
"""

PQ_CODEBOOK_ROWS: Final = 2**PQ_CODE_BITS
"""Rows a product quantizer needs before it can be trained at all.

A PQ codebook has one centroid per code, so 8-bit codes mean 256 of them and 256 vectors to
place them from. Below this LanceDB refuses the build outright — *"Not enough rows to train
PQ"* — rather than producing a weak index, which is the right refusal and the reason
:data:`MINIMUM_ANN_INDEX_THRESHOLD` exists. It is a property of the encoding rather than a
tunable, so it is derived here and not a setting.
"""

MINIMUM_ANN_INDEX_THRESHOLD: Final = PQ_CODEBOOK_ROWS
"""The smallest threshold that can ever produce an index.

Configuration is validated against this rather than clamped to it. A threshold of 50 is not a
cautious choice that happens to round up — it is a request the storage layer can never honor,
and one that would leave an installation permanently :attr:`AnnLifecycle.PENDING` while every
surface reported a build as due. Refusing it at the configuration boundary is the difference
between a typo caught at startup and a status line nobody can act on.
"""

PQ_SUBVECTOR_WIDTH: Final = 16
"""Dimensions per PQ sub-vector, targeted rather than guaranteed.

The count has to divide the dimension exactly, so this is the width
:func:`sub_vectors_for` aims at and the divisor it settles for.
"""

ANN_INDEX_PREFIX: Final = "manicule_ivfpq"
"""Names an index this project built, and distinguishes it from one an operator did.

An index on the vector column under any other name is reported rather than adopted: it is
somebody's deliberate act, the maintenance boundary does not know what it was for, and
replacing it silently would be this module deciding it knows better.
"""

_ANN_INDEX_NAME: Final = re.compile(
    rf"^{re.escape(ANN_INDEX_PREFIX)}_g(?P<generation>\d+)_p(?P<partitions>\d+)$"
)


class AnnLifecycle(StrEnum):
    """Where dense search is on the path ``docs/storage.md`` §6.2 describes.

    Five states, and the split that matters most is between the two that look alike from the
    outside: :attr:`EXHAUSTIVE` and :attr:`PENDING` are both "no index, every vector scanned",
    and they are opposite findings. One is the design working; the other is the corpus having
    outgrown it.
    """

    DISABLED = "disabled"
    """The threshold is ``0``: this installation has chosen exhaustive search permanently.

    Not the same as :attr:`EXHAUSTIVE`, which is the same behavior arrived at by being small
    enough. An existing index is left alone rather than dropped — turning the threshold off
    stops new builds, and destroying a built index is not something a configuration change
    should do on its way past.
    """

    EXHAUSTIVE = "exhaustive"
    """Below the threshold. Search is exact and no index is wanted."""

    PENDING = "pending"
    """At or above the threshold with no index. A build is due.

    Durable without anything being written to record it: the state is a function of committed
    rows, so a crash between the publication that crossed the threshold and the maintenance
    pass that acts on it loses nothing. There is no queue entry to drop.
    """

    READY = "ready"
    """An index covers the corpus, within the refresh bound."""

    STALE = "stale"
    """The unindexed tail has itself grown past the threshold.

    Results stay correct — LanceDB scans unindexed fragments and merges them into the ranked
    result — so this is a latency finding, not a correctness one. It is the same rule that
    triggered the first build, applied to the rows the index does not cover: a tail big enough
    to deserve an index of its own is a tail big enough to be paying for.
    """


@dataclass(frozen=True, slots=True)
class AnnIndex:
    """What the store holds, as the index itself reports it.

    ``build_generation`` and ``num_partitions`` are ``None`` for an index this project did not
    create. Both are read from the name, LanceDB records neither, and inferring them from the
    row count would produce a number that is right until the day the partition rule changes
    and then confidently wrong.
    """

    name: str
    index_type: str
    distance_type: str | None
    indexed_rows: int
    unindexed_rows: int
    num_sub_vectors: int | None
    build_generation: int | None = None
    num_partitions: int | None = None

    @property
    def recognized(self) -> bool:
        """Whether this project built it, and can therefore replace it."""
        return self.build_generation is not None

    @property
    def covered_rows(self) -> int:
        """Rows the index accounts for, indexed or not."""
        return self.indexed_rows + self.unindexed_rows

    @property
    def coverage(self) -> float:
        """Fraction of accounted rows the index actually covers, in ``[0, 1]``.

        ``1.0`` for an index with nothing appended since it was built, and — deliberately —
        ``1.0`` for an empty table rather than ``0.0``: no rows are uncovered, and reporting a
        coverage gap over a corpus with no vectors in it would be a fault nobody can clear.
        """
        if self.covered_rows == 0:
            return 1.0
        return self.indexed_rows / self.covered_rows


@dataclass(frozen=True, slots=True)
class AnnIndexState:
    """The whole answer to "is dense search indexed, and is it current?".

    Assembled by the storage layer, reported verbatim by
    :meth:`~manicule.app.service.ApplicationService.index_status`. Every field is either
    measured or configured; none is estimated.
    """

    lifecycle: AnnLifecycle
    threshold: int
    rows: int
    generation: str | None = None
    """The published vector generation this index belongs to, when the handle follows one.

    An index lives inside the generation directory it was built in, so a re-embed publishes a
    new generation with no index and the state honestly returns to :attr:`AnnLifecycle.PENDING`
    rather than inheriting the retired generation's coverage.
    """

    index: AnnIndex | None = None
    minimum_rows: int = PQ_CODEBOOK_ROWS
    detail: str = ""
    """Why the state is what it is, when the numbers alone do not say it."""

    @property
    def due(self) -> bool:
        """Whether the maintenance boundary has work to do here."""
        return self.lifecycle in {AnnLifecycle.PENDING, AnnLifecycle.STALE}

    @property
    def buildable(self) -> bool:
        """Whether a build attempted right now could actually train."""
        return self.rows >= self.minimum_rows

    @property
    def exact(self) -> bool:
        """Whether every result is currently an exact nearest neighbor.

        True while no index exists — including :attr:`AnnLifecycle.PENDING`, which is exact and
        slow rather than fast and approximate. Stated as its own property because "no index"
        is the interesting fact for a person reading a status page and "which of the three
        index-free states" is not.
        """
        return self.index is None


@dataclass(frozen=True, slots=True)
class AnnIndexBuild:
    """What one pass of the maintenance boundary did, and what it left behind.

    ``before`` and ``after`` are the same shape the status surface reports, so an operator
    reading the result of a build and an operator reading status a minute later are reading
    the same fields. ``built`` is ``False`` for every outcome that changed nothing — nothing
    was due, the corpus is too small to train, a dry run — and ``detail`` says which.
    """

    before: AnnIndexState
    after: AnnIndexState
    built: bool = False
    dry_run: bool = False
    detail: str = ""

    @property
    def replaced(self) -> str | None:
        """The index this build superseded, if it superseded one."""
        if not self.built or self.before.index is None:
            return None
        return self.before.index.name


def partitions_for(rows: int) -> int:
    """``num_partitions ≈ sqrt(n)``, as ``docs/storage.md`` §6.2 specifies.

    Clamped to at least one, and never above the row count: IVF trains one centroid per
    partition and cannot produce more centroids than it has vectors to place them from.
    """
    return max(1, min(isqrt(rows), rows))


def sub_vectors_for(dimension: int) -> int:
    """How many PQ sub-vectors to split a ``dimension``-wide vector into.

    Targets :data:`PQ_SUBVECTOR_WIDTH` dimensions per sub-vector and settles for the largest
    divisor that does not exceed the target, because the count must divide the dimension
    exactly. The common embedding widths land on the target exactly — 1024 gives 64, 768 gives
    48, 384 gives 24 — and the awkward ones give up width rather than correctness: 100 gives 5
    sub-vectors of 20, not 6 of 16.667.

    Returns ``1`` for a dimension below the target width, which is the honest answer for a
    4-dimension test vector and never reached by a real embedder.
    """
    target = max(1, dimension // PQ_SUBVECTOR_WIDTH)
    return max(candidate for candidate in range(1, target + 1) if dimension % candidate == 0)


def ann_index_name(*, build_generation: int, num_partitions: int) -> str:
    """The name that carries what LanceDB will not record.

    Read back by :func:`parse_ann_index_name` on every status call, which is why the format is
    fixed here rather than composed at the call site.
    """
    return f"{ANN_INDEX_PREFIX}_g{build_generation}_p{num_partitions}"


def parse_ann_index_name(name: str) -> tuple[int, int] | None:
    """``(build_generation, num_partitions)``, or ``None`` if this project did not build it."""
    matched = _ANN_INDEX_NAME.match(name)
    if matched is None:
        return None
    return int(matched["generation"]), int(matched["partitions"])


def classify(*, threshold: int, rows: int, index: AnnIndex | None) -> AnnLifecycle:
    """Which state a store with these numbers is in.

    The refresh policy is one line of it, and it is deliberately the same comparison that
    triggered the first build: an index goes :attr:`AnnLifecycle.STALE` when the rows it does
    not cover would themselves have crossed the threshold. Reusing the number means an
    operator has one dial to reason about rather than two that interact.
    """
    if index is None:
        if threshold <= 0:
            return AnnLifecycle.DISABLED
        return AnnLifecycle.PENDING if rows >= threshold else AnnLifecycle.EXHAUSTIVE
    if threshold > 0 and index.unindexed_rows >= threshold:
        return AnnLifecycle.STALE
    return AnnLifecycle.READY


__all__ = [
    "ANN_INDEX_PREFIX",
    "MINIMUM_ANN_INDEX_THRESHOLD",
    "PQ_CODEBOOK_ROWS",
    "PQ_CODE_BITS",
    "PQ_SUBVECTOR_WIDTH",
    "AnnIndex",
    "AnnIndexBuild",
    "AnnIndexState",
    "AnnLifecycle",
    "ann_index_name",
    "classify",
    "parse_ann_index_name",
    "partitions_for",
    "sub_vectors_for",
]
