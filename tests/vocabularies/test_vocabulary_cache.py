"""Where the vocabularies live, and why it is not where ``tiktoken`` would put them.

``tiktoken``'s default cache is under the system temporary directory. macOS reclaims that on
a schedule, and manicule's query path deliberately cannot re-fetch — so the pairing turns a
successful install into one that refuses every question, weeks later, with nothing having
changed and nothing having said so.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from manicule import vocabularies
from manicule.config.settings import default_cache_dir
from manicule.vocabularies import store


def test_the_default_cache_is_not_where_tiktoken_would_have_put_it() -> None:
    """The whole defect, as one assertion: upstream's default is the reclaimed directory."""
    assert (
        vocabularies.default_cache_directory() != Path(tempfile.gettempdir()) / store.CACHE_DIR_NAME
    )


def test_the_default_cache_is_not_somewhere_the_system_reclaims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a real machine, where ``XDG_CACHE_HOME`` is not itself inside the temp directory.

    Pointed at a directory of this test's own rather than read from the environment the suite
    runs under: pytest's ``tmp_path`` lives under ``$TMPDIR`` on macOS, so a sandboxed cache
    is impermanent by construction and asserting on it would test the harness.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path.anchor + "durable"))
    assert not vocabularies.is_impermanent(vocabularies.default_cache_directory())


def test_the_default_cache_lives_under_maniculets_own_cache_directory(
    manicule_environment: Path,
) -> None:
    """Regenerable artifacts belong with the other regenerable artifacts."""
    del manicule_environment
    assert vocabularies.default_cache_directory().parent == default_cache_dir()


@pytest.mark.parametrize("variable", [store.CACHE_DIR_ENV, store.LEGACY_CACHE_DIR_ENV])
def test_an_operator_who_pointed_tiktoken_somewhere_is_obeyed(
    variable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both of upstream's variables still win. This is a default, not an override."""
    monkeypatch.delenv(store.CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(store.LEGACY_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv(variable, str(tmp_path / "theirs"))

    assert vocabularies.cache_directory() == tmp_path / "theirs"


def test_tiktoken_is_pointed_at_the_directory_this_module_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tiktoken`` is the reader, so returning a durable path is not enough on its own.

    A manicule that merely *answered* differently would seed one directory and leave the
    library looking in another — the silent failure the durable default exists to remove,
    reintroduced by the fix for it.
    """
    monkeypatch.delenv(store.CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(store.LEGACY_CACHE_DIR_ENV, raising=False)

    with store.pointed_at_the_cache():
        assert os.environ[store.CACHE_DIR_ENV] == str(vocabularies.default_cache_directory())
    assert store.CACHE_DIR_ENV not in os.environ


def test_an_operators_own_cache_directory_is_neither_replaced_nor_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring a variable nobody set would be the same bug in the other direction."""
    monkeypatch.delenv(store.LEGACY_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv(store.CACHE_DIR_ENV, str(tmp_path / "theirs"))

    with store.pointed_at_the_cache():
        assert os.environ[store.CACHE_DIR_ENV] == str(tmp_path / "theirs")
    assert os.environ[store.CACHE_DIR_ENV] == str(tmp_path / "theirs")


def test_a_temporary_directory_is_recognized_as_one() -> None:
    """What ``doctor`` asks so it can refuse to call an impermanent cache healthy."""
    assert vocabularies.is_impermanent(Path(tempfile.gettempdir()) / store.CACHE_DIR_NAME)
    assert vocabularies.is_impermanent(Path(tempfile.gettempdir()))


def test_a_durable_directory_is_not(tmp_path: Path) -> None:
    """``tmp_path`` is pytest's, not the system temporary directory, and must not trip it."""
    assert not vocabularies.is_impermanent(Path.home() / ".cache" / "manicule")
    del tmp_path


def test_reporting_on_the_cache_does_not_bring_one_into_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``doctor`` writes to the machine only under ``--fix``, and a read path is a read path.

    The first version of ``pointed_at_the_cache`` created the directory unconditionally, so a
    plain ``doctor`` — a report — left a new directory behind it.
    """
    monkeypatch.delenv(store.CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(store.LEGACY_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with store.pointed_at_the_cache():
        pass

    assert not vocabularies.default_cache_directory().exists()


def test_the_pre_seed_makes_the_directory_it_is_about_to_write_into(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write path, where creating it is the whole point."""
    monkeypatch.delenv(store.CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(store.LEGACY_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with store.pointed_at_the_cache(create=True):
        pass

    assert vocabularies.default_cache_directory().is_dir()
