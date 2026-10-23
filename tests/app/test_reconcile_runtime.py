"""Deletion detection against a real database, a real pipeline and a real connector.

``tests/app/test_reconcile.py`` proves the service and the scheduler over fakes; the storage
suites prove the guards in isolation. This proves they are the same thing: a filesystem source is
synced into a real SQLite index, files are deleted from disk, and the product's own operations
find them. The embedder is the suite's hash stand-in — what is under test is the wiring, not a
model.

Files, not mocks, are the source on purpose. A connector double that yielded the ids the test
chose would make "a partial enumeration deletes nothing" true by construction.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from manicule.app.dispatch import run_op
from manicule.app.runtime import Runtime
from manicule.app.served import Scheduler, SchedulerConfiguration
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.errors import PolicyError
from manicule.plugins.registry import discover
from manicule.storage.backup_lock import backup_lock
from tests.app.test_scheduler import Clock
from tests.fakes import HashEmbedder
from tests.ingest.fakes import BlockChunker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


def runtime_over(environment: Path, source_root: Path, **connector: object) -> Runtime:
    """A runtime whose one configured source is the directory ``source_root``."""
    found = discover()
    bound = found.registry.bind("test")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    settings = Settings(
        data_dir=environment / "data",
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block", "glossary": {"detect_on_ingest": False}},  # pyright: ignore[reportArgumentType]
        connectors={  # pyright: ignore[reportArgumentType]
            "docs": {"type": "filesystem", "options": {"root": str(source_root)}, **connector}
        },
    )
    return Runtime(settings, discovery=found)


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    for number in range(10):
        (root / f"page-{number}.md").write_text(f"# Page {number}\n\nBody of page number {number}.")
    return root


@pytest.fixture
async def service(
    manicule_environment: Path, source_root: Path
) -> AsyncIterator[ApplicationService]:
    async with runtime_over(manicule_environment, source_root) as opened:
        yield ApplicationService(opened)


def remove(root: Path, *numbers: int) -> None:
    for number in numbers:
        (root / f"page-{number}.md").unlink()


async def live(service: ApplicationService) -> int:
    summary = next(c for c in (await service.connector_list()).connectors if c.name == "docs")
    assert summary.documents is not None
    return summary.documents


async def test_a_first_sync_indexes_the_source_and_records_that_it_was_checked(
    service: ApplicationService,
) -> None:
    report = await service.connector_sync("docs")

    assert report.ingested == 10
    assert report.reconciliation is not None
    assert (report.reconciliation.outcome, report.reconciliation.deleted) == ("applied", 0)
    summary = (await service.connector_list()).connectors[0]
    assert summary.last_reconciled_at is not None, "criterion 1: last_reconcile_at is recorded"
    assert summary.reconcile_overdue is False


async def test_a_clean_pass_soft_deletes_exactly_the_files_that_left(
    service: ApplicationService, source_root: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 3)

    result = await service.connector_reconcile("docs")

    assert (result.outcome, result.deleted, result.missing) == ("applied", 1, 1)
    assert await live(service) == 9
    titles = {hit.title for hit in (await service.document_list(source="docs")).documents}
    assert "page-3.md" not in titles
    assert {"page-0.md", "page-9.md"} <= titles, "the files that remain stay indexed"


async def test_a_sync_removes_a_deleted_file_without_anyone_running_reconcile(
    service: ApplicationService, source_root: Path
) -> None:
    """Incremental sync cannot see a deletion; the pass that follows a full sync can."""
    await service.connector_sync("docs")
    remove(source_root, 7)

    report = await service.connector_sync("docs")

    assert report.reconciliation is not None
    assert report.reconciliation.deleted == 1
    assert await live(service) == 9


async def test_a_dry_run_changes_nothing_and_cannot_be_replayed_as_authority(
    service: ApplicationService, source_root: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 2)

    preview = await service.connector_reconcile("docs", dry_run=True)

    assert (preview.outcome, preview.missing, preview.deleted) == ("previewed", 1, 0)
    assert await live(service) == 10
    applied = await service.connector_reconcile("docs")
    assert applied.deleted == 1, "applying enumerated again rather than replaying the preview"


async def test_a_source_that_fails_midway_deletes_nothing(
    service: ApplicationService, source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 3: the prefix is not the truth, and it is never diffed."""
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2)
    connector = await (await service._backend.ingestion()).connector("docs")  # pyright: ignore[reportPrivateUsage]

    real = connector.reconcile

    async def prefix_then_fail() -> AsyncIterator[str]:
        count = 0
        async for source_id in real():
            yield source_id
            count += 1
            if count == 3:
                msg = "the source dropped the connection"
                raise OSError(msg)

    monkeypatch.setattr(connector, "reconcile", prefix_then_fail)

    envelope = await run_op(
        "connector_reconcile", "default", lambda: service.connector_reconcile("docs")
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "OSError"
    assert "dropped the connection" not in json.dumps(envelope.as_json())
    assert await live(service) == 10, "not one document was removed on a partial enumeration"


async def test_a_pass_over_the_ceiling_proposes_and_waits_for_a_person(
    service: ApplicationService, source_root: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2, 3, 4)

    proposed = await service.connector_reconcile("docs")

    assert (proposed.outcome, proposed.missing, proposed.live) == ("proposed", 5, 10)
    assert proposed.proposal_pending is True
    assert await live(service) == 10, "nothing is deleted until a person answers"
    listing = (await service.connector_list()).connectors[0]
    assert listing.reconcile_proposal_pending is True
    assert (listing.reconcile_proposal_missing, listing.reconcile_proposal_live) == (5, 10)
    diagnosis = await service.doctor()
    waiting = next(c for c in diagnosis.checks if c.name == "reconciliation-proposals")
    assert waiting.state == "degraded"
    assert waiting.facts == {"waiting": ["docs"], "missing": 5}


async def test_a_waiting_proposal_can_be_confirmed(
    service: ApplicationService, source_root: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2, 3, 4)
    await service.connector_reconcile("docs")

    confirmed = await service.connector_reconcile_resolve("docs", confirm=True)

    assert (confirmed.outcome, confirmed.deleted) == ("confirmed", 5)
    assert await live(service) == 5
    listing = (await service.connector_list()).connectors[0]
    assert listing.reconcile_proposal_pending is False
    assert listing.last_reconciled_at is not None
    again = await run_op(
        "connector_reconcile_resolve",
        "default",
        lambda: service.connector_reconcile_resolve("docs", confirm=True),
    )
    assert again.ok is False, "a proposal answers once"


async def test_a_waiting_proposal_can_be_discarded_and_is_not_proposed_again_at_once(
    service: ApplicationService, source_root: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2, 3, 4)
    await service.connector_reconcile("docs")

    discarded = await service.connector_reconcile_resolve("docs", confirm=False)

    assert (discarded.outcome, discarded.deleted) == ("discarded", 0)
    assert await live(service) == 10
    assert await service.reconcile_if_due("docs") is None, "the interval restarted"
    listing = (await service.connector_list()).connectors[0]
    assert listing.reconcile_proposal_pending is False


async def test_a_file_synced_after_a_proposal_survives_its_confirmation(
    service: ApplicationService, source_root: Path
) -> None:
    """The proposal names a reviewed set; evidence that a document exists after it wins."""
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2, 3, 4)
    await service.connector_reconcile("docs")
    # Page 4 comes back, and a sync sees it. Sync would also chain a pass but for the proposal.
    (source_root / "page-4.md").write_text("# Page 4\n\nBack again, and changed.")
    await service.connector_sync("docs")

    confirmed = await service.connector_reconcile_resolve("docs", confirm=True)

    assert confirmed.deleted == 4
    titles = {hit.title for hit in (await service.document_list(source="docs")).documents}
    assert "page-4.md" in titles


async def test_reconciliation_is_refused_for_a_disabled_source(
    manicule_environment: Path, source_root: Path
) -> None:
    async with runtime_over(manicule_environment, source_root, enabled=False) as opened:
        with pytest.raises(PolicyError):
            await ApplicationService(opened).connector_reconcile("docs")


async def test_the_scheduler_reconciles_a_due_source_end_to_end(
    service: ApplicationService, source_root: Path
) -> None:
    """Criterion 1 through the real stack: nobody runs a command, and the pass is recorded."""
    await service.connector_sync("docs")
    # Forget that the sync just checked, as a server that was down for a week would find it.
    store = cast("Any", await service._backend.documents())  # pyright: ignore[reportPrivateUsage]
    await store.record_connector_metadata("docs", {"last_clean_reconcile_at": None})
    remove(source_root, 5)
    clock = Clock()
    scheduler = Scheduler(
        service,
        SchedulerConfiguration(
            {}, lifecycle_interval_s=None, sweep_interval_s=None, reconcile_intervals={"docs": 600}
        ),
        sleep=clock.sleep,
    )

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
    finally:
        await scheduler.aclose()

    assert await live(service) == 9
    assert scheduler.reconciling["docs"].deleted == 1
    summary = (await service.connector_list()).connectors[0]
    assert summary.last_reconciled_at is not None


async def test_a_pass_declines_while_a_backup_holds_the_data_and_runs_after(
    service: ApplicationService, source_root: Path, manicule_environment: Path
) -> None:
    """The issue's 'not during a backup', against the real lock a real backup takes."""
    await service.connector_sync("docs")
    remove(source_root, 3)

    async with backup_lock(manicule_environment / "data", wait_s=5):
        held = await run_op(
            "connector_reconcile", "default", lambda: service.connector_reconcile("docs")
        )
        preview = await service.connector_reconcile("docs", dry_run=True)

    assert held.ok is False
    assert held.error is not None
    assert held.error.type == "ReconciliationBlockedError"
    assert await live(service) == 10, "a backup was copying; nothing was removed"
    assert preview.outcome == "previewed", "a preview removes nothing and never has to wait"
    after = await service.connector_reconcile("docs")
    assert (after.outcome, after.deleted) == ("applied", 1)


async def test_confirming_waits_for_a_backup_but_discarding_does_not(
    service: ApplicationService, source_root: Path, manicule_environment: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2, 3, 4)
    await service.connector_reconcile("docs")

    async with backup_lock(manicule_environment / "data", wait_s=5):
        confirm = await service.connector_reconcile_resolve("docs", confirm=True)
    assert confirm.outcome == "blocked"
    assert await live(service) == 10

    confirmed = await service.connector_reconcile_resolve("docs", confirm=True)
    assert (confirmed.outcome, confirmed.deleted) == ("confirmed", 5)


async def test_discarding_does_not_wait_for_a_backup(
    service: ApplicationService, source_root: Path, manicule_environment: Path
) -> None:
    await service.connector_sync("docs")
    remove(source_root, 0, 1, 2, 3, 4)
    await service.connector_reconcile("docs")

    async with backup_lock(manicule_environment / "data", wait_s=5):
        discarded = await service.connector_reconcile_resolve("docs", confirm=False)

    assert discarded.outcome == "discarded"


async def test_a_backup_waits_for_a_pass_in_its_removal_step_and_not_for_its_enumeration(
    service: ApplicationService, source_root: Path, manicule_environment: Path
) -> None:
    """The window is held around the delete, so a slow source never starves a backup."""
    from manicule.storage.backup_lock import MaintenanceBusyError  # noqa: PLC0415

    await service.connector_sync("docs")
    connector = await (await service._backend.ingestion()).connector("docs")  # pyright: ignore[reportPrivateUsage]
    real = connector.reconcile
    during_enumeration: list[bool] = []

    async def enumerating() -> AsyncIterator[str]:
        try:
            async with backup_lock(manicule_environment / "data", wait_s=0.1):
                during_enumeration.append(True)
        except MaintenanceBusyError:
            during_enumeration.append(False)
        async for source_id in real():
            yield source_id

    object.__setattr__(connector, "reconcile", enumerating)

    await service.connector_reconcile("docs")

    assert during_enumeration == [True], "a backup could take the lock mid-enumeration"
