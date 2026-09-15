"""A local directory as a source: what it finds, what it refuses, and what it never invents.

The connector behind ``manicule index <path>``. Three of these tests defend properties that
are invisible until they are wrong — an identity that varies with the working directory, a
media type that varies with the machine, and a walk that follows a symlink out of the tree.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from manicule.connectors.config import FilesystemConfig
from manicule.connectors.errors import NotFoundError
from manicule.connectors.filesystem import (
    OCTET_STREAM,
    FilesystemConnector,
    media_type_for,
    version_token,
)
from manicule.connectors.plugin import build_filesystem
from manicule.core.protocols import Connector, aclose
from manicule.plugins import BuildContext
from manicule.testing import assert_connector_contract

if TYPE_CHECKING:
    from collections.abc import Iterator


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


# --- configured exclusions ----------------------------------------------------------------------


@pytest.fixture
def archived(tmp_path: Path) -> Path:
    """A corpus with superseded material kept beside what replaced it, and a root file.

    Both shapes from the report this was written for: a subtree that must stay in git because
    it is the record of what was tried, and a ``README.md`` about the corpus rather than in it.
    """
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "current.md").write_text("# Current\n", encoding="utf-8")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "superseded.md").write_text("# Was\n", encoding="utf-8")
    (tmp_path / "archive" / "deep").mkdir()
    (tmp_path / "archive" / "deep" / "older.md").write_text("# Older\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# About\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("pattern", ["archive", "archive/**", "**/archive/**"])
async def test_an_excluded_subtree_is_not_walked_however_it_is_spelled(
    archived: Path, pattern: str
) -> None:
    """Three spellings an operator might reach for, naming one subtree.

    ``archive`` and ``archive/**`` are the same intent written two ways, and refusing to
    understand one of them would be this connector holding out for a syntax nobody documented.
    Nothing beneath the directory survives either, which is the point: the reason the reporter
    deleted the directory from his corpus was that there was no way to say this.
    """
    found = await _discovered(FilesystemConnector(archived, exclude=(pattern,)))

    assert any(path.endswith("current.md") for path in found)
    assert not any("archive" in path for path in found), "nothing under it, at any depth"


async def test_an_excluded_file_at_the_root_is_the_case_a_prefix_cannot_say(
    archived: Path,
) -> None:
    """One file, named exactly, and its namesakes elsewhere left alone.

    A root ``README.md`` describes the corpus rather than belonging to it. Anchoring at the root
    is what keeps the pattern from also taking every ``README.md`` further down, which an
    operator asking for this one would not have meant.
    """
    (archived / "live" / "README.md").write_text("# Live\n", encoding="utf-8")

    found = await _discovered(FilesystemConnector(archived, exclude=("README.md",)))

    assert not any(path.endswith(f"{archived}/README.md") for path in found)
    assert any(path.endswith("live/README.md") for path in found)


async def test_discovery_and_reconciliation_exclude_the_same_paths(archived: Path) -> None:
    """The two enumerations must never disagree about what this source holds.

    ``reconcile`` reports what still exists, so a path ``discover`` skips and ``reconcile``
    still names is a document reported deleted on every sync and re-indexed on every sync. The
    reverse — filtering inside ``discover`` alone — is the one that would have been easy to
    write and is why this is asserted rather than assumed: the walk is the single place both
    read, and that is the whole of why it is enforced there.
    """
    connector = FilesystemConnector(archived, exclude=("archive",))

    discovered = sorted(await _discovered(connector))
    reconciled = sorted([identity async for identity in connector.reconcile()])

    assert discovered == reconciled
    assert not any("archive" in identity for identity in reconciled)


@pytest.mark.parametrize("pattern", ["archive", "archive/**", "**/archive/**"])
async def test_an_excluded_directory_is_pruned_rather_than_walked_and_discarded(
    archived: Path, monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    """Never descended into, because the subtree this exists for is a large one.

    Kept superseded material is exactly the thing that grows without bound, so listing it on
    every sync in order to throw it away is a cost that scales with what was excluded. Asserted
    on the directory listings the walk performs, because "did not descend" is invisible in the
    result either way.

    Over all three spellings, because only the bare one is decided by the same candidate that
    decides a file. ``archive/**`` and ``**/archive/**`` both require a trailing separator, so
    they name the directory *only* through the trailing-slash candidate ``excludes`` builds —
    and with that candidate gone every discovery assertion in this file still passes, because
    the files underneath match on their own. Pruning is the only thing that observes it.
    """
    listed: list[Path] = []
    original = Path.iterdir

    def recording(self: Path) -> Iterator[Path]:
        listed.append(self)
        return original(self)

    monkeypatch.setattr(Path, "iterdir", recording)

    await _discovered(FilesystemConnector(archived, exclude=(pattern,)))

    assert archived in listed, "the root is still walked"
    assert archived / "live" in listed
    assert archived / "archive" not in listed
    assert archived / "archive" / "deep" not in listed


async def test_a_source_with_no_exclusions_walks_everything_it_did_before(archived: Path) -> None:
    """The positive control: the default is empty, and empty excludes nothing.

    A default exclusion would be this module deciding some of a directory somebody pointed it at
    does not count.
    """
    assert FilesystemConnector(archived).exclude == ()
    found = await _discovered(FilesystemConnector(archived))

    assert len(found) == 4


async def test_a_root_that_is_itself_a_file_is_indexed_whatever_is_excluded(
    archived: Path,
) -> None:
    """Patterns are relative to the root, so for a file root there is nothing for one to name.

    ``manicule index <path>`` on one file is the path being the argument rather than a setting.
    A source configured to walk nothing at all reports a clean run over an empty corpus, which
    is the failure this whole connector's docstring is written against.
    """
    found = await _discovered(
        FilesystemConnector(archived / "README.md", exclude=("**", "README.md"))
    )

    assert found == [str(archived / "README.md")]


async def test_the_walk_and_authoring_ask_one_question_about_an_excluded_path(
    archived: Path,
) -> None:
    """``excludes`` is public because it is asked of a path that does not exist yet.

    ``document_create`` writes first and ingests second, so it has to be able to ask before
    there is a file to walk. An excluded *ancestor* has to answer for its descendants too, or
    the walk would keep ``archive/`` out while authoring happily wrote into it.
    """
    connector = FilesystemConnector(archived, exclude=("archive",))

    assert connector.excludes(archived / "archive" / "written-later.md")
    assert connector.excludes(archived / "archive" / "deep")
    assert not connector.excludes(archived / "live" / "written-later.md")
    assert not connector.excludes(archived), "the root is the argument, never excluded"
    assert not connector.excludes(Path("/elsewhere/entirely.md")), "`contains` answers that"


_GIT = shutil.which("git") or "git"

_GITIGNORE_CORPUS = (
    "btctrader/live/a.md",
    "btctrader/archive/old.md",
    "btctrader/archive/deep/older.md",
    "README.md",
    "docs/README.md",
    "notes/keep.md",
    "notes/deep/keep.md",
    "notes/scratch.log",
)

_ROOTED_PATTERNS = [
    "btctrader/archive",
    "btctrader/archive/**",
    "**/archive/**",
    "**/*.log",
    "notes/*",
    "notes/deep",
    "archive/*",
    "**/deep/**",
    "btctrader/**",
]


def _repository(tmp_path: Path, pattern: str) -> Path:
    """The corpus above, in a Git repository ignoring ``pattern``. Sync: it blocks throughout."""
    for relative in _GITIGNORE_CORPUS:
        written = tmp_path / relative
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("x\n", encoding="utf-8")
    root = tmp_path.resolve()
    subprocess.run(  # noqa: S603 - test-owned fixed executable and arguments
        [_GIT, "init", "-q", str(root)], check=True, capture_output=True
    )
    (root / ".gitignore").write_text(f"{pattern}\n", encoding="utf-8")
    return root


def _unignored(root: Path) -> list[str]:
    """What ``git check-ignore`` leaves of the corpus, under the repository's own ignore file."""
    return sorted(
        relative
        for relative in _GITIGNORE_CORPUS
        if subprocess.run(  # noqa: S603 - test-owned fixed executable and arguments
            [_GIT, "-C", str(root), "check-ignore", "-q", relative], check=False
        ).returncode
        != 0
    )


async def _walked(root: Path, pattern: str) -> list[str]:
    return sorted(
        str(Path(source_id).relative_to(root))
        for source_id in await _discovered(FilesystemConnector(root, exclude=(pattern,)))
    )


@pytest.mark.parametrize("pattern", _ROOTED_PATTERNS)
async def test_a_pattern_naming_a_path_excludes_what_gitignore_would(
    tmp_path: Path, pattern: str
) -> None:
    """The semantics are not ours to invent, and this is the oracle that says whether we did.

    Everybody who reaches for this setting already knows one path-pattern language, and it is
    the one used by the repository the corpus is usually kept in — the corpus in #379 is a Git
    checkout. A differential test rather than a table of expected answers, because a table is
    our own opinion written down a second time and then agreeing with itself.

    Every pattern here contains a separator, which is the condition under which ``.gitignore``
    reads a pattern as relative to the directory it sits in. Slash-free patterns are where the
    two deliberately part company, and that is the test below rather than a gap here.
    """
    root = _repository(tmp_path, pattern)

    assert await _walked(root, pattern) == _unignored(root)


async def test_a_pattern_is_root_relative_even_where_gitignore_would_match_at_any_depth(
    tmp_path: Path,
) -> None:
    """The one deliberate divergence, pinned rather than left to be discovered.

    ``.gitignore`` has two rules: a pattern with a separator is relative to the directory, and a
    pattern without one matches at *any* depth. So ``README.md`` there hides every README in the
    tree. Here there is one rule — every pattern is relative to the root, as though it had been
    written with a leading separator — and that is what makes the second case in #379 sayable at
    all: a ``README.md`` *about* the corpus, at the top, with the READMEs inside it left alone.

    Asserted against git in the same breath, because "we differ from git here" is a claim that
    rots the moment either side changes, and a comment saying so would not notice.
    """
    root = _repository(tmp_path, "README.md")

    walked = await _walked(root, "README.md")

    assert "README.md" not in walked, "the one named at the root goes"
    assert "docs/README.md" in walked, "and the one below it stays"
    assert "docs/README.md" not in _unignored(root), "where git would have taken it too"


def test_filesystem_configuration_is_closed_and_canonical(tmp_path: Path) -> None:
    """A setting that appears to be in force and silently is not is worse than one that fails.

    ``exclude`` is the first field on this model that is neither a path nor a number, so it is
    the first that can be written in a way that means nothing — an absolute pattern matches no
    root-relative path, ever, and reads exactly like one that is simply not being hit yet.
    """
    config = FilesystemConfig(root=str(tmp_path), exclude=("archive/**", "archive/**", "*.log"))

    assert config.exclude == ("archive/**", "*.log"), "repeats collapse, order is kept"
    with pytest.raises(ValidationError):
        FilesystemConfig(root=str(tmp_path), exclud=("archive/**",))  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError):
        FilesystemConfig(root=str(tmp_path), exclude=("/absolute/**",))
    with pytest.raises(ValidationError):
        FilesystemConfig(root=str(tmp_path), exclude=("../outside/**",))


def test_a_configured_exclusion_reaches_the_connector_that_was_built_from_it(
    tmp_path: Path,
) -> None:
    """Validation at load is worth nothing if the value never reaches the thing it configures.

    ``[connectors.docs.options]`` reaching no connector is a defect this repository has had
    before (#94), and it is invisible: the setting is accepted, the sync runs, and the archive
    is indexed beside what replaced it as though nothing had been written.

    Read through what the connector *does* rather than off an attribute, so this cannot pass on
    a connector that stored the patterns and then consulted none of them.
    """
    settings = FilesystemConfig(root=str(tmp_path), exclude=("archive/**",))
    built = build_filesystem(
        BuildContext(
            settings=None,  # pyright: ignore[reportArgumentType] - unused on this path
            config=settings,
            data_dir=None,  # pyright: ignore[reportArgumentType] - unused on this path
            cache_dir=None,  # pyright: ignore[reportArgumentType] - unused on this path
            components=None,  # pyright: ignore[reportArgumentType] - unused on this path
            instance="docs",
        )
    )

    assert isinstance(built, FilesystemConnector)
    assert built.exclude == settings.exclude
    assert built.excludes(tmp_path / "archive" / "superseded.md")
    assert not built.excludes(tmp_path / "live" / "current.md")


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
