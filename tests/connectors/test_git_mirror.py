"""The managed mirror: how a remote reaches the machine, and what must not travel with it.

A real fetch is made against a bare repository on disk through the ``file`` transport, which is the
only way to run Git's own protocol machinery without a network. Everything a test cannot reach
that way — the ``https`` allowlist, the credential header — is asserted on the environment the
child process is given, because that environment is exactly what Git reads.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
from pathlib import Path

import pytest

from manicule.connectors.git_mirror import TIP_REF, GitFetchError, GitMirror
from manicule.connectors.git_reader import PinnedGitReader
from manicule.connectors.git_remote import (
    is_remote,
    mirror_directory,
    normalize_remote_url,
    require_fetchable_revision,
)
from manicule.core.errors import ConfigError

_GIT = shutil.which("git") or "git"
TOKEN = "ghp_synthetic_token_value"  # noqa: S105 - a synthetic value, not a credential


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(  # noqa: S603 - resolved Git and test-owned repository
        [_GIT, "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(repository: Path, message: str) -> str:
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=Manicule Tests",
        "-c",
        "user.email=tests@example.test",
        "commit",
        "-m",
        message,
    )
    return git(repository, "rev-parse", "HEAD")


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    """A repository standing in for the remote, with ``uploadpack`` left at its defaults."""
    root = tmp_path / "origin"
    root.mkdir()
    git(root, "init", "--quiet")
    (root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    commit(root, "first")
    return root


def mirror_of(origin: Path, tmp_path: Path, **overrides: object) -> GitMirror:
    options: dict[str, object] = {
        "token": None,
        "allowed_protocols": "file",
        "timeout_s": 60.0,
    }
    options.update(overrides)
    return GitMirror(origin.as_uri(), tmp_path / "mirror", **options)  # pyright: ignore[reportArgumentType]


async def test_a_refresh_brings_the_tip_and_a_second_one_brings_the_next_commit(
    origin: Path, tmp_path: Path
) -> None:
    """The point of a mirror: a push upstream is visible at the next sync, with no checkout."""
    mirror = mirror_of(origin, tmp_path)

    ref = await mirror.refresh("HEAD")
    first = git(mirror.directory, "rev-parse", ref)
    assert ref == TIP_REF
    assert first == git(origin, "rev-parse", "HEAD")

    (origin / "guide.md").write_text("# Guide, revised\n", encoding="utf-8")
    second_commit = commit(origin, "second")
    await mirror.refresh("HEAD")

    assert git(mirror.directory, "rev-parse", TIP_REF) == second_commit != first


async def test_the_reader_pins_what_the_mirror_fetched(origin: Path, tmp_path: Path) -> None:
    """The mirror's contract with the reader is a path and a ref name, and nothing else."""
    mirror = mirror_of(origin, tmp_path)
    ref = await mirror.refresh("HEAD")

    async with PinnedGitReader(mirror.directory, revision=ref) as reader:
        entries = await reader.setup()
        assert [entry.path for entry in entries] == ["guide.md"]
        assert await reader.read_entry(entries[0]) == b"# Guide\n"


async def test_history_is_not_fetched(origin: Path, tmp_path: Path) -> None:
    """A tree at one commit is all an indexer can use, so only that is brought over."""
    (origin / "guide.md").write_text("# Newer\n", encoding="utf-8")
    commit(origin, "second")
    mirror = mirror_of(origin, tmp_path)

    await mirror.refresh("HEAD")

    assert git(mirror.directory, "rev-list", "--count", TIP_REF) == "1"


async def test_the_mirror_holds_no_address_and_no_credential(origin: Path, tmp_path: Path) -> None:
    """The remote is a fetch argument and the token a child-process variable, so neither is on
    disk where a backup, a ``kubectl cp`` or a bug report would carry it away."""
    mirror = mirror_of(origin, tmp_path, token=TOKEN)
    await mirror.refresh("HEAD")

    on_disk = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in mirror.directory.rglob("*")
        if path.is_file() and path.stat().st_size < 1_000_000 and ".pack" not in path.name
    )
    assert TOKEN not in on_disk
    assert base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode() not in on_disk
    assert origin.as_uri() not in (mirror.directory / "config").read_text(encoding="utf-8")


def test_the_credential_is_scoped_to_the_origin_and_travels_only_in_the_environment() -> None:
    """Scoped, so a redirect to another host carries nothing; in the environment, so it is not in
    ``ps`` output either."""
    mirror = GitMirror(
        "https://git.example.org/team/docs", Path("unused"), token=TOKEN, username="bot"
    )
    environment = mirror.environment()

    count = int(environment["GIT_CONFIG_COUNT"])
    entries = {
        environment[f"GIT_CONFIG_KEY_{index}"]: environment[f"GIT_CONFIG_VALUE_{index}"]
        for index in range(count)
    }
    encoded = base64.b64encode(f"bot:{TOKEN}".encode()).decode()
    assert entries["http.https://git.example.org/.extraHeader"] == f"Authorization: Basic {encoded}"
    assert entries["core.hooksPath"] == "/dev/null"
    assert entries["credential.helper"] == ""
    assert environment["GIT_ALLOW_PROTOCOL"] == "https"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


def test_a_mirror_with_no_token_sends_no_authorization_at_all() -> None:
    mirror = GitMirror("https://git.example.org/team/docs", Path("unused"))
    environment = mirror.environment()

    keys = [
        environment[f"GIT_CONFIG_KEY_{index}"]
        for index in range(int(environment["GIT_CONFIG_COUNT"]))
    ]
    assert not any("extraHeader" in key for key in keys)


def test_an_inherited_git_variable_cannot_redirect_a_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GIT_DIR`` and ``GIT_SSH_COMMAND`` in a server's environment would change what a command
    reads and runs. The hardened environment drops them before it adds its own."""
    monkeypatch.setenv("GIT_DIR", "/somewhere/else")
    monkeypatch.setenv("GIT_SSH_COMMAND", "evil")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "99")
    mirror = GitMirror("https://git.example.org/team/docs", Path("unused"))
    environment = mirror.environment()

    assert "GIT_DIR" not in environment
    assert "GIT_SSH_COMMAND" not in environment
    assert environment["GIT_CONFIG_COUNT"] != "99"


async def test_a_failed_fetch_names_the_address_and_never_the_token(
    origin: Path, tmp_path: Path
) -> None:
    """Git's own message is kept, because it is what finds the problem, and is scrubbed first."""
    missing = origin.parent / "no-such-repository"
    mirror = GitMirror(
        missing.as_uri(),
        tmp_path / "mirror",
        token=TOKEN,
        allowed_protocols="file",
        timeout_s=60.0,
    )

    with pytest.raises(GitFetchError) as caught:
        await mirror.refresh("HEAD")

    message = str(caught.value)
    assert missing.as_uri() in message
    assert TOKEN not in message


async def test_a_revision_the_remote_does_not_have_is_an_error_not_an_empty_mirror(
    origin: Path, tmp_path: Path
) -> None:
    mirror = mirror_of(origin, tmp_path)

    with pytest.raises(GitFetchError, match="no-such-branch"):
        await mirror.refresh("no-such-branch")


async def test_a_protocol_outside_the_allowlist_is_refused_by_git_itself(
    origin: Path, tmp_path: Path
) -> None:
    """``https`` is the allowlist for a configured source, so a ``file`` remote (or a redirect to
    one) never gets a connection."""
    mirror = mirror_of(origin, tmp_path, allowed_protocols="https")

    with pytest.raises(GitFetchError):
        await mirror.refresh("HEAD")


async def test_a_refresh_that_outlasts_its_timeout_is_stopped(origin: Path, tmp_path: Path) -> None:
    """A fetch that hangs must not hold a sync, and the scheduler's loop behind it, forever."""
    slow = tmp_path / "slow-git"
    slow.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    slow.chmod(0o755)
    mirror = mirror_of(origin, tmp_path, timeout_s=1.0, executable=str(slow))

    with pytest.raises(GitFetchError, match="timed out"):
        await mirror.refresh("HEAD")


def test_two_sources_naming_one_remote_share_one_directory(tmp_path: Path) -> None:
    """One copy per address, so two sources over one repository fetch it once, not twice."""
    assert mirror_directory(tmp_path, "https://a.example.org/x") == mirror_directory(
        tmp_path, "https://a.example.org/x"
    )
    assert mirror_directory(tmp_path, "https://a.example.org/x") != mirror_directory(
        tmp_path, "https://a.example.org/y"
    )


async def test_two_sources_refreshing_one_mirror_at_once_both_succeed(
    origin: Path, tmp_path: Path
) -> None:
    """They share a directory, and two fetches into one repository fight over its locks."""
    first = mirror_of(origin, tmp_path)
    second = mirror_of(origin, tmp_path)

    await asyncio.gather(*(m.refresh("HEAD") for m in (first, second, first, second)))

    assert git(first.directory, "rev-parse", TIP_REF) == git(origin, "rev-parse", "HEAD")


async def test_a_revision_that_is_not_a_ref_is_refused_before_git_runs(
    origin: Path, tmp_path: Path
) -> None:
    mirror = mirror_of(origin, tmp_path)

    with pytest.raises(ConfigError, match="revision"):
        await mirror.refresh("main:refs/heads/elsewhere")


@pytest.mark.parametrize(
    "revision", ["main:other", "-delete", "a b", "a..b", "x@{1}", "/lead", "trail/", "a//b", ""]
)
def test_a_revision_that_cannot_be_spliced_into_a_refspec_is_refused(revision: str) -> None:
    with pytest.raises(ValueError, match="revision"):
        require_fetchable_revision(revision)


@pytest.mark.parametrize("revision", ["HEAD", "main", "release/1.2", "v1.0.0", "a" * 40])
def test_the_revisions_people_actually_use_are_accepted(revision: str) -> None:
    require_fetchable_revision(revision)


def test_a_remote_address_is_normalized_and_one_that_carries_a_secret_is_refused() -> None:
    assert normalize_remote_url("HTTPS://Git.Example.org:443/team/docs.git/") == (
        "https://git.example.org:443/team/docs.git"
    )
    for bad in (
        "https://user:pw@git.example.org/x",
        "https://token@git.example.org/x",
        "https://git.example.org/x?token=1",
        "https://git.example.org/x#frag",
        "http://git.example.org/x",
        "ssh://git@git.example.org/x",
        "file:///srv/repo",
        "https:///nohost",
    ):
        with pytest.raises(ValueError, match=r"(?s)^(?!.*pw).*$"):
            normalize_remote_url(bad)


def test_is_remote_separates_an_address_from_a_path() -> None:
    assert is_remote("https://git.example.org/x")
    assert is_remote("ssh://git@git.example.org/x")
    assert not is_remote("/srv/checkouts/docs")
    assert not is_remote("~/checkouts/docs")
