"""A pytest plugin giving tests an isolated manicule environment.

Enable it from a ``conftest.py``::

    pytest_plugins = ["manicule.testing.fixtures"]

Then every test runs against a scratch config directory, a scratch data directory and no
provider credentials. Without that, a test reads whatever the developer happens to have
configured, and passes or fails depending on whose machine it is running on.

Importing this module needs pytest. Nothing else in manicule does, so pytest stays a
development dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from manicule.config.providers import PROVIDER_ALIASES
from manicule.config.settings import ENV_PREFIX, Settings

_CONVENTIONAL_KEYS = tuple(
    sorted(
        {
            f"{provider.upper()}_API_KEY"
            for provider in (
                "openai",
                "anthropic",
                "google",
                "xai",
                "mistral",
                "cohere",
                "groq",
                "azure",
                "together",
                "deepseek",
                "openrouter",
            )
        }
        | {alias for aliases in PROVIDER_ALIASES.values() for alias in aliases}
    )
)


@pytest.fixture(autouse=True)
def manicule_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point configuration at a scratch directory and clear every credential.

    Returns the working directory, which is where a ``.env`` or ``manicule.toml`` written by
    a test should go.
    """
    for name in list(os.environ):
        if name.startswith(ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)
    for name in _CONVENTIONAL_KEYS:
        monkeypatch.delenv(name, raising=False)

    home = tmp_path / "home"
    (home / "config" / "manicule").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def settings() -> Settings:
    """Default settings, built in the isolated environment."""
    return Settings()


__all__ = ["manicule_environment", "settings"]
