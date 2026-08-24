"""The local Git source is one immutable object graph, never a checkout walk."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from manicule.connectors.git_reader import (
    GitBlobTooLargeError,
    GitObjectMissingError,
    GitSourceError,
    PinnedGitReader,
)
from manicule.connectors.site_routes import SiteRouteError

_GIT = shutil.which("git") or "git"


def git(repository: Path, *arguments: str) -> bytes:
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(  # noqa: S603 - resolved Git and test-owned repository
        [_GIT, "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        env=environment,
    ).stdout


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo with spaces"
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.name", "Synthetic Author")
    git(root, "config", "user.email", "author@example.test")
    return root


def commit(root: Path, files: dict[str, bytes], message: str) -> str:
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "-m", message)
    return git(root, "rev-parse", "HEAD").decode().strip()


def state(root: Path) -> tuple[bytes, bytes, bytes, bytes]:
    return (
        git(root, "for-each-ref", "--format=%(refname) %(objectname)"),
        git(root, "status", "--porcelain=v1", "--untracked-files=all"),
        git(root, "ls-files", "--stage"),
        git(root, "config", "--local", "--list"),
    )


async def opened_reader(
    root: Path, *, revision: str = "HEAD", content_root: str = ".", max_bytes: int = 1024
) -> PinnedGitReader:
    reader = PinnedGitReader(root, revision=revision, max_blob_bytes=max_bytes)
    await reader.setup(content_root=content_root)
    return reader


async def test_tree_inventory_and_blob_reads_support_hostile_literal_paths(tmp_path: Path) -> None:
    root = repository(tmp_path)
    commit(
        root,
        {
            "docs/--option-like.md": b"option body",
            "docs/space name.md": b"space body",
            "docs/über.md": b"Unicode body",
        },
        "initial",
    )
    before = state(root)
    reader = await opened_reader(root, content_root="docs")
    try:
        assert [entry.path for entry in reader.entries] == [
            "docs/--option-like.md",
            "docs/space name.md",
            "docs/über.md",
        ]
        assert all(entry.ordinary_blob and entry.size is not None for entry in reader.entries)
        bodies = [await reader.read_entry(entry) for entry in reader.entries]
        assert bodies == [b"option body", b"space body", b"Unicode body"]
    finally:
        await reader.aclose()
    assert state(root) == before


async def test_head_and_worktree_movement_cannot_change_a_pinned_reader(tmp_path: Path) -> None:
    root = repository(tmp_path)
    first_commit = commit(root, {"docs/page.md": b"first body"}, "first")
    reader = await opened_reader(root, content_root="docs")
    first_entry = reader.entries[0]

    (root / "docs/page.md").write_bytes(b"second body")
    (root / "docs/untracked.md").write_bytes(b"untracked draft")
    second_commit = commit(root, {"docs/second.md": b"second page"}, "second")
    assert second_commit != first_commit

    try:
        assert reader.commit == first_commit
        assert reader.entries == (first_entry,)
        assert await reader.read_entry(first_entry) == b"first body"
        looked_up = await reader.lookup("docs/page.md")
        assert looked_up == first_entry
    finally:
        await reader.aclose()


async def test_an_option_like_revision_is_data_not_a_git_argument(tmp_path: Path) -> None:
    root = repository(tmp_path)
    expected = commit(root, {"docs/page.md": b"body"}, "tag target")
    git(root, "update-ref", "refs/tags/-pinned", expected)

    reader = await opened_reader(root, revision="-pinned", content_root="docs")
    try:
        assert reader.commit == expected
        assert await reader.read_entry(reader.entries[0]) == b"body"
    finally:
        await reader.aclose()


async def test_force_moving_a_ref_after_setup_does_not_move_the_reader(tmp_path: Path) -> None:
    root = repository(tmp_path)
    pinned = commit(root, {"docs/page.md": b"pinned"}, "pinned")
    git(root, "branch", "site", pinned)
    reader = await opened_reader(root, revision="site", content_root="docs")
    moved = commit(root, {"docs/page.md": b"moved"}, "moved")
    git(root, "branch", "--force", "site", moved)
    try:
        assert reader.commit == pinned
        assert await reader.read_entry(reader.entries[0]) == b"pinned"
    finally:
        await reader.aclose()


async def test_symlinks_and_submodules_are_visible_but_never_ordinary_blobs(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    target_commit = commit(root, {"docs/page.md": b"page"}, "ordinary")
    (root / "docs/link.md").symlink_to("../../outside.md")
    git(root, "add", "docs/link.md")
    git(
        root,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{target_commit},docs/vendor",
    )
    git(root, "commit", "--quiet", "-m", "special entries")

    reader = await opened_reader(root, content_root="docs")
    try:
        by_path = {entry.path: entry for entry in reader.entries}
        assert by_path["docs/link.md"].symlink
        assert by_path["docs/vendor"].submodule
        assert [entry.path for entry in reader.ordinary_entries()] == ["docs/page.md"]
        for path in ("docs/link.md", "docs/vendor"):
            with pytest.raises(GitSourceError, match="not an ordinary blob"):
                await reader.read_entry(by_path[path])
    finally:
        await reader.aclose()


async def test_a_manifest_lookup_is_exact_and_cannot_traverse(tmp_path: Path) -> None:
    root = repository(tmp_path)
    commit(
        root,
        {"docs/page.md": b"page", ".manicule-site.json": b'{"version":1,"pages":[]}'},
        "manifest",
    )
    reader = await opened_reader(root, content_root="docs")
    try:
        manifest = await reader.lookup(".manicule-site.json")
        assert manifest is not None
        assert manifest.ordinary_blob
        assert await reader.read_entry(manifest) == b'{"version":1,"pages":[]}'
        with pytest.raises(SiteRouteError, match="traversal"):
            await reader.lookup("../outside")
    finally:
        await reader.aclose()


async def test_large_blobs_are_refused_before_the_batch_process_starts(tmp_path: Path) -> None:
    root = repository(tmp_path)
    commit(root, {"docs/large.md": b"12345"}, "large")
    reader = await opened_reader(root, content_root="docs", max_bytes=4)
    try:
        with pytest.raises(GitBlobTooLargeError, match="4-byte"):
            await reader.read_entry(reader.entries[0])
        assert reader._batch is None  # pyright: ignore[reportPrivateUsage]
    finally:
        await reader.aclose()


async def test_a_missing_pinned_object_never_falls_back_to_the_worktree(tmp_path: Path) -> None:
    root = repository(tmp_path)
    commit(root, {"docs/page.md": b"committed bytes"}, "object to remove")
    reader = await opened_reader(root, content_root="docs")
    entry = reader.entries[0]
    loose_object = root / ".git/objects" / entry.object_id[:2] / entry.object_id[2:]
    assert loose_object.is_file(), "the fixture unexpectedly packed the object"
    loose_object.unlink()
    (root / "docs/page.md").write_bytes(b"working tree fallback would be wrong")
    try:
        with pytest.raises(GitObjectMissingError, match="no longer available"):
            await reader.read_entry(entry)
    finally:
        await reader.aclose()


async def test_teardown_reaps_the_lazy_batch_process(tmp_path: Path) -> None:
    root = repository(tmp_path)
    commit(root, {"docs/page.md": b"body"}, "batch")
    reader = await opened_reader(root, content_root="docs")
    await reader.read_entry(reader.entries[0])
    process = reader._batch  # pyright: ignore[reportPrivateUsage]
    assert process is not None
    assert process.returncode is None

    await reader.aclose()

    assert process.returncode is not None
    assert reader._batch is None  # pyright: ignore[reportPrivateUsage]


async def test_cancellation_discards_a_batch_that_may_be_mid_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path)
    commit(root, {"docs/page.md": b"body"}, "cancel")
    reader = await opened_reader(root, content_root="docs")
    arrived = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*arguments: Any) -> bytes:
        del arguments
        arrived.set()
        await release.wait()
        return b"unreachable"

    monkeypatch.setattr(reader, "_read_from_batch", blocked)
    task = asyncio.create_task(reader.read_entry(reader.entries[0]))
    await arrived.wait()
    process = reader._batch  # pyright: ignore[reportPrivateUsage]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process is not None
    assert process.returncode is not None
    assert reader._batch is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "content_root",
    ["../outside", "/absolute", "docs/../outside", "--option-like/../outside"],
)
async def test_content_roots_cannot_escape_or_be_reinterpreted(
    tmp_path: Path, content_root: str
) -> None:
    root = repository(tmp_path)
    commit(root, {"docs/page.md": b"body"}, "root")
    reader = PinnedGitReader(root)
    with pytest.raises(SiteRouteError, match=r"traversal|relative"):
        await reader.setup(content_root=content_root)
