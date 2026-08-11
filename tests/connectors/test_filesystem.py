"""A local directory as a source: what it finds, what it refuses, and what it never invents.

The connector behind ``manicule index <path>``. Three of these tests defend properties that
are invisible until they are wrong — an identity that varies with the working directory, a
media type that varies with the machine, and a walk that follows a symlink out of the tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.connectors.errors import NotFoundError
from manicule.connectors.filesystem import (
    OCTET_STREAM,
    FilesystemConnector,
    media_type_for,
    version_token,
)
from manicule.core.protocols import Connector, aclose
from manicule.testing import assert_connector_contract

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small tree with the shapes that matter: nested, hidden, and tool output."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "retry.md").write_text("# Retry\n\nTwice.\n", encoding="utf-8")
    (tmp_path / "docs" / "notes.txt").write_text("plain\n", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("no\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "a.js").write_text("1\n", encoding="utf-8")
    return tmp_path


async def _discovered(connector: FilesystemConnector) -> list[str]:
    return [document.source_id async for document in connector.discover(None)]


@pytest.mark.contract
async def test_the_filesystem_connector_satisfies_the_connector_contract(tree: Path) -> None:
    """The suite every connector passes, run against this one.

    It checks the things that are the same for every source: that discovery is decidable, that
    a watermark reflects a *complete* enumeration, and that reconciliation reports what still
    exists.
    """
    connector = FilesystemConnector(tree, name="local")
    assert isinstance(connector, Connector)
    await assert_connector_contract(connector)


async def test_the_walk_skips_version_control_and_tool_output(tree: Path) -> None:
    """A repository's ``.git`` is larger than the repository, and none of it is a document."""
    found = await _discovered(FilesystemConnector(tree))
    assert any(path.endswith("retry.md") for path in found)
    assert not any(".git" in path for path in found)
    assert not any("node_modules" in path for path in found)
    assert not any(".hidden" in path for path in found)


async def test_hidden_files_are_included_when_asked_for(tree: Path) -> None:
    """The positive control: the skip is a default, not a limitation."""
    found = await _discovered(FilesystemConnector(tree, include_hidden=True))
    assert any(".hidden" in path for path in found)


async def test_identity_does_not_depend_on_where_the_walk_started(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trap this connector exists to avoid.

    A document's id is a digest of ``(workspace, source, source_id)``. If the source id were
    relative to the root, indexing ``~/docs`` and then ``~`` would produce two documents for
    one file — and neither would ever supersede the other.
    """
    absolute = await _discovered(FilesystemConnector(tree / "docs"))
    monkeypatch.chdir(tree)
    relative = await _discovered(FilesystemConnector(tree.parent / tree.name / "docs"))
    assert absolute == relative
    assert all(path.startswith("/") for path in absolute)


async def test_a_file_and_a_directory_are_both_valid_roots(tree: Path) -> None:
    """``manicule index one-file.md`` is the commonest first thing anybody does."""
    single = await _discovered(FilesystemConnector(tree / "docs" / "retry.md"))
    assert len(single) == 1


async def test_the_walk_is_in_a_stable_order(tree: Path) -> None:
    """So that ``--limit 10`` means the same ten documents on two machines."""
    first = await _discovered(FilesystemConnector(tree))
    second = await _discovered(FilesystemConnector(tree))
    assert first == second == sorted(first)


async def test_a_file_larger_than_the_cap_is_refused_before_it_is_read(tree: Path) -> None:
    (tree / "docs" / "big.txt").write_text("x" * 5000, encoding="utf-8")
    found = await _discovered(FilesystemConnector(tree, max_bytes=1000))
    assert not any(path.endswith("big.txt") for path in found)


async def test_a_symlink_is_not_followed(tmp_path: Path) -> None:
    """A symlink out of the tree is an escape; one inside it is an unbounded walk."""
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "a.md").write_text("a\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "b.md").write_text("b\n", encoding="utf-8")
    (inside / "link").symlink_to(outside)

    found = await _discovered(FilesystemConnector(inside))
    assert [path.rsplit("/", 1)[-1] for path in found] == ["a.md"]


async def test_fetching_a_path_outside_the_root_is_refused(tmp_path: Path) -> None:
    """A stored source id must not be a way to read any file this process can open."""
    from manicule.core.sources import DocRef  # noqa: PLC0415 - one assertion needs it

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("a\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_text("secret\n", encoding="utf-8")

    connector = FilesystemConnector(root)
    with pytest.raises(NotFoundError) as caught:
        await connector.fetch(DocRef(source_id=str(elsewhere), uri=elsewhere.as_uri()))
    assert "outside" in str(caught.value)


async def test_fetching_a_file_that_has_gone_is_a_refusal_rather_than_a_crash(
    tree: Path,
) -> None:
    from manicule.core.sources import DocRef  # noqa: PLC0415 - one assertion needs it

    connector = FilesystemConnector(tree)
    missing = tree / "docs" / "gone.md"
    with pytest.raises(NotFoundError):
        await connector.fetch(DocRef(source_id=str(missing), uri=missing.as_uri()))


# --- media types ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.md", "text/markdown"),
        ("a.pdf", "application/pdf"),
        ("a.py", "text/x-python"),
        ("a.yaml", "application/yaml"),
        ("a.unknown", OCTET_STREAM),
        ("noextension", OCTET_STREAM),
    ],
)
def test_the_media_type_comes_from_a_table_in_the_source(
    tmp_path: Path, name: str, expected: str
) -> None:
    """Never from the machine's mime database.

    ``mimetypes.guess_type`` reads ``/etc/mime.types`` and the Windows registry, so the same
    file would route to different parsers on two machines and chunk two different ways. The
    platform may change how fast this runs; it may not change what ends up in the index.
    """
    assert media_type_for(tmp_path / name) == expected


def test_the_change_token_moves_when_the_file_does(tmp_path: Path) -> None:
    """And is ``None`` for a file that cannot be read, rather than a value that never moves."""
    path = tmp_path / "a.md"
    path.write_text("one\n", encoding="utf-8")
    first = version_token(path)
    path.write_text("two and then some more\n", encoding="utf-8")
    assert first is not None
    assert version_token(path) != first
    assert version_token(tmp_path / "gone.md") is None


async def test_the_watermark_appears_only_after_a_complete_walk(tree: Path) -> None:
    """A watermark stored for a partial enumeration loses documents permanently.

    The connector reports ``None`` until ``discover`` has run to the end, so a caller that
    persisted it after an interrupted walk would have nothing to persist.
    """
    connector = FilesystemConnector(tree)
    assert connector.watermark is None
    stream = connector.discover(None)
    await anext(stream)
    assert connector.watermark is None, "a watermark appeared part-way through the walk"
    # Closed through the helper every consumer uses: `discover` promises an `AsyncIterator`,
    # which is a weaker thing than a generator and has no `aclose` of its own.
    await aclose(stream)

    await _discovered(connector)
    assert connector.watermark is not None


async def test_reconciliation_reports_what_still_exists(tree: Path) -> None:
    """Without it the index serves a deleted file forever, and no amount of syncing fixes it."""
    connector = FilesystemConnector(tree)
    before = {source_id async for source_id in connector.reconcile()}
    (tree / "docs" / "notes.txt").unlink()
    after = {source_id async for source_id in connector.reconcile()}
    assert before - after == {str(tree / "docs" / "notes.txt")}
