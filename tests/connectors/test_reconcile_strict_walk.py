"""A directory the walk cannot list is an error for reconciliation, never an empty directory.

Discovery skips a subtree it cannot read and carries on, which is right for indexing. Reconcile
treats absence as deletion, so the same skip would report every file beneath the subtree gone —
and with one unreadable directory in ten that is below the default deletion ceiling, so the
product would delete documents whose files are still there. These tests make a real directory
unreadable rather than substituting a walker that already raises.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from manicule.connectors.confluence_snapshot import ConfluenceSnapshotConnector
from manicule.connectors.filesystem import FilesystemConnector
from tests.connectors.test_confluence_snapshot import snapshot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a mode-000 directory is only unreadable to an unprivileged POSIX user",
)


@pytest.fixture
def unreadable(tmp_path: Path) -> Iterator[Path]:
    """A directory that cannot be listed, restored afterwards so pytest can clean up."""
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    hidden.chmod(0o000)
    yield hidden
    hidden.chmod(0o700)


async def ids(connector: FilesystemConnector | ConfluenceSnapshotConnector) -> list[str]:
    return [source_id async for source_id in connector.reconcile()]


async def test_filesystem_reconcile_fails_rather_than_report_an_unreadable_subtree_gone(
    tmp_path: Path, unreadable: Path
) -> None:
    del unreadable
    (tmp_path / "kept.md").write_text("# kept\n")

    with pytest.raises(PermissionError):
        await ids(FilesystemConnector(tmp_path, name="docs"))


async def test_filesystem_discovery_still_skips_what_it_cannot_read(
    tmp_path: Path, unreadable: Path
) -> None:
    """Indexing is allowed to be partial; only a deletion decision is not."""
    del unreadable
    (tmp_path / "kept.md").write_text("# kept\n")
    connector = FilesystemConnector(tmp_path, name="docs")

    found = [doc.ref.source_id async for doc in connector.discover(None)]

    assert [name.rsplit("/", 1)[-1] for name in found] == ["kept.md"]


async def test_a_missing_filesystem_root_is_not_an_empty_one(tmp_path: Path) -> None:
    """A source directory that vanished must not read as 'every file was deleted'."""
    connector = FilesystemConnector(tmp_path / "gone", name="docs")

    with pytest.raises(FileNotFoundError):
        await ids(connector)


async def test_snapshot_reconcile_fails_rather_than_report_an_unreadable_subtree_gone(
    tmp_path: Path,
) -> None:
    snapshot(tmp_path, at="ENG/123456")
    blocked = tmp_path / "ENG" / "locked"
    blocked.mkdir()
    blocked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            await ids(ConfluenceSnapshotConnector(tmp_path))
    finally:
        blocked.chmod(0o700)


async def test_snapshot_discovery_still_skips_what_it_cannot_read(tmp_path: Path) -> None:
    snapshot(tmp_path, at="ENG/123456")
    blocked = tmp_path / "ENG" / "locked"
    blocked.mkdir()
    blocked.chmod(0o000)
    try:
        found = [doc async for doc in ConfluenceSnapshotConnector(tmp_path).discover(None)]
    finally:
        blocked.chmod(0o700)

    assert len(found) == 1
