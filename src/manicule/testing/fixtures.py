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


@pytest.fixture(scope="session", autouse=True)
def model_cache() -> None:
    """Pin the Hugging Face cache to this machine's real one, for the whole session.

    :func:`manicule_environment` redirects ``XDG_CACHE_HOME`` at every test, and recent
    ``huggingface_hub`` resolves its cache through that variable — so a model sitting on disk
    becomes invisible the moment a test imports the hub lazily, and every embedding suite skips
    on a machine where the weights are right there. Worse in CI, where the pre-seed step would
    download several hundred megabytes into a directory nothing later reads, and the job would
    report green having checked nothing.

    Session-scoped and autouse so it runs while the environment is still the real one. The
    resolved path is written back to ``HF_HUB_CACHE``, which takes precedence over both
    ``HF_HOME`` and the XDG variable, so a later redirection cannot move it.

    **Published here rather than kept in manicule's own conftest**, because an embedding backend
    may ship as its own distribution — ``manicule-mlx`` is the first — and its suite is redirected
    by exactly the same fixture and hidden from exactly the same weights. Keeping this private
    reproduced the documented failure the moment the parity tests moved: armed with
    ``REQUIRE_EMBEDDING_MODELS``, thirteen cases failed reporting weights absent that were
    sitting in the cache.
    """
    try:
        import huggingface_hub.constants as hub  # noqa: PLC0415 - an embeddings extra
    except ImportError:
        return
    os.environ["HF_HUB_CACHE"] = str(hub.HF_HUB_CACHE)


__all__ = ["manicule_environment", "model_cache", "settings"]
