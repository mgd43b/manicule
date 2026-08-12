"""What the hub module says while it works, and what it declines to say.

Both behaviours here exist because a model fetch is the one operation in manicule that can
take minutes, and the only one whose progress a third-party library reports on its behalf.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
from huggingface_hub.utils.tqdm import are_progress_bars_disabled

from manicule.embedding.runtimes import hub


def _cached(answer: bool) -> Any:
    """A stand-in for :func:`hub.is_cached` that answers ``answer``, whatever it is asked."""

    def probe(*_args: Any, **_kwargs: Any) -> bool:
        return answer

    return probe


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record whether the hub's progress bars were suppressed for each download.

    Asked of ``huggingface_hub`` itself, not of the environment. The first version of this
    fixture read ``PROGRESS_ENV`` — manicule's own lever — and passed against an
    implementation that set that variable and changed nothing, because the hub reads it once
    at import. A test that watches the mechanism instead of the effect is how that shipped.
    """
    seen: list[bool] = []

    def download(*_args: Any, local_files_only: bool = False, **_kwargs: Any) -> str:
        if not local_files_only:
            seen.append(are_progress_bars_disabled())
        return "/nowhere"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return seen


def test_a_warm_cache_does_not_draw_a_progress_bar_for_a_download_it_is_not_doing(
    recorded: list[bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two bars before every search, reporting a fetch that did not happen."""
    monkeypatch.delenv(hub.PROGRESS_ENV, raising=False)
    monkeypatch.setattr(hub, "is_cached", _cached(True))

    hub.snapshot("BAAI/bge-m3", ["*.json"])

    assert recorded == [True], "the bars were left on for a download that was already cached"
    assert not are_progress_bars_disabled(), "the suppression outlived the call that set it"


def test_a_real_download_keeps_the_only_thing_on_screen(
    recorded: list[bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first index moves over a gigabyte, and the bar is how somebody knows it is alive.

    Silencing this case is the mistake that makes a working download indistinguishable from a
    wedged process, which is what the bars were covering all along.
    """
    monkeypatch.delenv(hub.PROGRESS_ENV, raising=False)
    monkeypatch.setattr(hub, "is_cached", _cached(False))

    hub.snapshot("BAAI/bge-m3", ["*.json"])

    assert recorded == [False], "manicule silenced the progress of a real download"


def test_an_operator_who_chose_a_setting_keeps_it(
    recorded: list[bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set by a CI job capturing logs, or a container. Not manicule's to restore."""
    monkeypatch.setenv(hub.PROGRESS_ENV, "0")
    monkeypatch.setattr(hub, "is_cached", _cached(True))

    hub.snapshot("BAAI/bge-m3", ["*.json"])

    assert recorded == [False], "an operator's own setting was overridden"
    assert os.environ[hub.PROGRESS_ENV] == "0", "an operator's own value was replaced"


def test_a_local_directory_is_present_without_asking_the_hub(tmp_path: Path) -> None:
    """A path is not a repository id, and there is nothing to look up."""
    assert hub.is_cached(str(tmp_path), ["*.json"])
