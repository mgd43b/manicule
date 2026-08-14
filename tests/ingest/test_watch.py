"""Watching a directory, and the editor behavior a naive watcher gets wrong."""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.ingest.watch import Change, WatchEvent, coalesce, is_ignored, settle


@pytest.mark.parametrize(
    "name",
    [".notes.md.swp", "notes.md~", "notes.md.tmp", ".#notes.md", "4913", ".DS_Store"],
)
def test_editor_scratch_files_are_ignored(name: str) -> None:
    """``4913`` is Vim's write-permission probe.

    Without this list a single ``:w`` produces an ingest attempt for a file that no longer
    exists by the time anybody looks at it.
    """
    assert is_ignored(Path("/corpus") / name)


def test_a_real_document_is_not_ignored() -> None:
    assert not is_ignored(Path("/corpus/notes.md"))


def test_a_burst_collapses_to_one_event_per_path() -> None:
    """A single logical save commonly produces several events."""
    events = coalesce(
        [
            (Change.ADDED, Path("/corpus/a.md")),
            (Change.MODIFIED, Path("/corpus/a.md")),
            (Change.MODIFIED, Path("/corpus/a.md")),
        ]
    )

    assert events == [WatchEvent(path=Path("/corpus/a.md"), change=Change.MODIFIED)]


def test_the_last_change_to_a_path_is_the_one_that_survives() -> None:
    """Only the final state of a path is a fact about the world.

    A rename-based save creates a temporary file and removes it inside the window; keeping the
    first event would leave the watcher holding a creation for a file that is gone.
    """
    events = coalesce(
        [
            (Change.ADDED, Path("/corpus/a.md")),
            (Change.DELETED, Path("/corpus/a.md")),
        ]
    )

    assert events[0].change is Change.DELETED


def test_a_creation_that_did_not_survive_the_window_is_dropped(tmp_path: Path) -> None:
    """The re-``stat`` is what makes debouncing correct rather than merely calmer.

    Without it the pipeline is handed a path that no longer exists and reports a fetch failure
    for a document that never existed.
    """
    vanished = tmp_path / "vanished.md"

    settled = settle([WatchEvent(path=vanished, change=Change.ADDED)])

    assert settled == []


def test_a_creation_that_did_survive_the_window_is_kept(tmp_path: Path) -> None:
    real = tmp_path / "real.md"
    real.write_text("# notes\n")

    settled = settle([WatchEvent(path=real, change=Change.ADDED)])

    assert settled == [WatchEvent(path=real, change=Change.ADDED)]


def test_a_deletion_is_believed_without_re_checking(tmp_path: Path) -> None:
    """The event is the evidence.

    A deletion is the one signal a watcher gets that reconciliation cannot produce, and
    re-checking it would discard exactly the information that makes watch mode able to detect
    removal at all.
    """
    gone = tmp_path / "gone.md"

    settled = settle([WatchEvent(path=gone, change=Change.DELETED)])

    assert settled == [WatchEvent(path=gone, change=Change.DELETED)]


def test_scratch_files_are_dropped_after_the_window_too(tmp_path: Path) -> None:
    scratch = tmp_path / "notes.md.swp"
    scratch.write_text("x")

    assert settle([WatchEvent(path=scratch, change=Change.ADDED)]) == []


def test_a_rename_reads_as_a_delete_and_a_create_by_path(tmp_path: Path) -> None:
    """Which is what it is. Content-hash dedup then collapses it back into one document."""
    target = tmp_path / "renamed.md"
    target.write_text("# notes\n")
    temporary = tmp_path / "notes.md.tmp"

    settled = settle(
        coalesce(
            [
                (Change.ADDED, temporary),
                (Change.DELETED, temporary),
                (Change.MODIFIED, target),
            ]
        )
    )

    assert settled == [WatchEvent(path=target, change=Change.MODIFIED)]
