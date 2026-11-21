"""What the cheap surfaces may and may not ask of the vector store about checksum coverage.

Counting coverage on a network vector store is a handful of counts over every point's payload.
It is exact, and on a large corpus it is not free, so two surfaces are held to a line here: the
readiness probe never asks for it at all, and ``doctor`` words what it reports faithfully to
both of the ways a count can be non-zero.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import pytest

from manicule.app import service as service_module
from manicule.app.service import ApplicationService
from manicule.core.embedding import VectorChecksumCoverage
from tests.app.fakes import FakeBackend, FakeMaintenance

if TYPE_CHECKING:
    from manicule.app import results as r


@dataclass
class ForbiddenCoverage(FakeMaintenance):
    """A maintenance surface that fails the test the moment anything asks it for coverage."""

    asked: list[bool] = field(default_factory=list[bool])

    @override
    async def vector_checksum_coverage(
        self, *, recompute: bool = False
    ) -> VectorChecksumCoverage | None:
        self.asked.append(recompute)
        raise AssertionError("a surface that must stay cheap asked for checksum coverage")


@dataclass
class StalledCoverage(FakeMaintenance):
    """Coverage that never answers. **Deliberately stalled.**"""

    @override
    async def vector_checksum_coverage(
        self, *, recompute: bool = False
    ) -> VectorChecksumCoverage | None:
        del recompute
        await asyncio.Event().wait()
        return None


async def _check(backend: FakeBackend) -> r.Check:
    diagnosis = await ApplicationService(backend).doctor()
    return next(check for check in diagnosis.checks if check.name == "vector_integrity")


async def test_readiness_never_asks_for_checksum_coverage() -> None:
    """The readiness probe is one constant query, and coverage is a count over every payload.

    A pin rather than a regression test: nothing reaches coverage from the probe today. It
    exists so the day somebody adds "and the vectors are covered" to readiness, the probe that
    an orchestrator restarts things on goes red here instead of going slow in production.
    """
    maintenance = ForbiddenCoverage()
    backend = FakeBackend(maintenance_=maintenance)
    service = ApplicationService(backend)

    assert await service.ready() is True
    assert maintenance.asked == []


async def test_doctor_does_not_pass_a_coverage_check_that_never_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled count is ``unknown`` and unmeasured, and never ``ok``.

    The bound that stops doctor hanging also has to stop it claiming a check passed. Counting
    coverage is the one check here whose cost grows with the corpus, so it is the one a slow
    server stalls.
    """
    monkeypatch.setattr(service_module, "DOCTOR_CHECK_SECONDS", 0.2)
    backend = FakeBackend(maintenance_=StalledCoverage())

    check = await _check(backend)

    assert check.state == "unknown"
    assert check.facts == {"measured": False}
    assert "not measured" in check.detail
    assert "rows" not in check.facts, "a coverage figure was reported for a count that never ran"


async def test_doctor_names_a_half_written_pair_when_a_count_found_one() -> None:
    """In counting mode ``failed`` is a malformed pair, and the wording must say so.

    No vector was read and nothing was hashed, so "no longer match the checksum written beside
    them" would claim a comparison nobody made. What was found is a point carrying a checksum
    without its version, or a version without its checksum.
    """
    backend = FakeBackend(
        maintenance_=FakeMaintenance(
            vector_checksums=VectorChecksumCoverage(
                rows=100, recorded=100, failed=2, failures={"malformed": 2}
            )
        )
    )

    check = await _check(backend)

    assert check.state == "failing"
    assert check.facts["failed"] == 2
    assert check.facts["recomputed"] is False
    assert "version" in check.detail
    assert "half" in check.detail
    assert "no longer match" not in check.detail, "a count was worded as a recomputation"
    assert check.remedy == "manicule vector-checksum --verify"


async def test_doctor_names_a_mismatched_digest_when_a_recompute_found_one() -> None:
    """The other cause, which only a recomputation can establish, keeps its own wording."""
    backend = FakeBackend(
        maintenance_=FakeMaintenance(
            vector_checksums=VectorChecksumCoverage(
                rows=100,
                recorded=100,
                verified=97,
                failed=3,
                failures={"mismatched": 3},
                recomputed=True,
            )
        )
    )

    check = await _check(backend)

    assert check.state == "failing"
    assert "no longer match the checksum written beside them" in check.detail
    assert "half" not in check.detail


async def test_doctor_words_a_recompute_that_found_both_causes_as_both() -> None:
    """A recomputed scan can find a malformed pair and a mismatch in one corpus."""
    backend = FakeBackend(
        maintenance_=FakeMaintenance(
            vector_checksums=VectorChecksumCoverage(
                rows=100,
                recorded=100,
                verified=95,
                failed=5,
                failures={"malformed": 2, "mismatched": 3},
                recomputed=True,
            )
        )
    )

    check = await _check(backend)

    assert check.state == "failing"
    assert "no longer match the checksum written beside them" in check.detail
    assert "version" in check.detail
