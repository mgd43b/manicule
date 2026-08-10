"""Recovery from a dead parent, and the lock that stops there being two.

The allowlist test is the one that matters. A denylist passes the happy case and requeues a
terminal document forever, and nothing about the healthy path would ever reveal it.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from manicule.core.content import IN_FLIGHT, SETTLED, DocumentStatus
from manicule.core.errors import InstanceLockedError
from manicule.ingest.recovery import InstanceLock, requeue_interrupted
from tests.fakes import make_document
from tests.ingest import fakes

ANCIENT = datetime(2020, 1, 1, tzinfo=UTC)


def _stored(store: fakes.MemoryIngestStore, status: DocumentStatus, *, age_s: float) -> str:
    detail = "synthetic" if status in {DocumentStatus.FAILED} else None
    document = make_document().model_copy(
        update={
            "id": f"doc-{status.value}",
            "source_id": f"src-{status.value}",
            "status": status,
            "status_detail": detail
            or (
                "synthetic"
                if status.value in {"no_extractable_text", "unsupported_media_type"}
                else None
            ),
            "failed_stage": None if status is not DocumentStatus.FAILED else "parse",
        }
    )
    store.documents[document.id] = document
    store.updated_at[document.id] = datetime.now(UTC) - timedelta(seconds=age_s)
    return document.id


async def test_documents_caught_in_flight_are_requeued() -> None:
    """A document that is not ``indexed`` is not served, so requeueing is cheap and safe."""
    store = fakes.MemoryIngestStore()
    for status in sorted(IN_FLIGHT):
        _stored(store, status, age_s=7200)

    requeued = await requeue_interrupted(store, stale_after_s=3600)

    assert requeued == len(IN_FLIGHT)
    assert all(d.status is DocumentStatus.PENDING for d in store.documents.values())


@pytest.mark.parametrize(
    "status",
    [
        DocumentStatus.CONTAINER,
        DocumentStatus.NO_EXTRACTABLE_TEXT,
        DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
        DocumentStatus.INDEXED,
        DocumentStatus.FAILED,
    ],
)
async def test_a_terminal_document_is_never_swept(status: DocumentStatus) -> None:
    """The failure a denylist causes, made specific.

    ``container`` has zero chunks by design and ``no_extractable_text`` has zero chunks because
    there was nothing to find. Both look like "stopped before embedding" to a ``WHERE status !=
    'indexed'`` clause, and both would then be re-fetched, re-parsed and re-requeued on every
    run — forever, silently, at full cost.
    """
    store = fakes.MemoryIngestStore()
    document_id = _stored(store, status, age_s=999_999)

    requeued = await requeue_interrupted(store, stale_after_s=1)

    assert requeued == 0
    assert store.documents[document_id].status is status


def test_the_swept_set_and_the_settled_set_do_not_overlap() -> None:
    """Stated as a property rather than left to the two lists staying in step by inspection."""
    assert not (IN_FLIGHT & SETTLED)


async def test_a_document_still_being_worked_on_is_left_alone() -> None:
    """``stale_after`` sits comfortably above any per-document limit for this reason."""
    store = fakes.MemoryIngestStore()
    document_id = _stored(store, DocumentStatus.PARSING, age_s=5)

    requeued = await requeue_interrupted(store, stale_after_s=3600)

    assert requeued == 0
    assert store.documents[document_id].status is DocumentStatus.PARSING


def test_a_second_instance_cannot_take_the_same_data_directory(tmp_path: Path) -> None:
    """WAL permits several writers, so the single-writer assumption is enforced, not hoped for.

    A second instance that started anyway would requeue the first one's in-flight documents
    while it was still working on them.
    """
    with InstanceLock(tmp_path) as held:
        assert held.path.exists()
        with pytest.raises(InstanceLockedError, match=str(os.getpid())):
            InstanceLock(tmp_path).acquire()


def test_the_lock_is_released_when_its_holder_goes(tmp_path: Path) -> None:
    """``flock`` rather than a PID file whose contents are believed.

    A PID file survives a crash and has to be reasoned about. A kernel lock does not: this is
    the difference between a lock and a note.
    """
    with InstanceLock(tmp_path):
        pass

    with InstanceLock(tmp_path) as second:
        assert second.path.exists()


def test_releasing_a_lock_that_was_never_taken_is_harmless(tmp_path: Path) -> None:
    InstanceLock(tmp_path).release()
