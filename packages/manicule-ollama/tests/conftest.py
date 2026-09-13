"""Everything this package's suite needs from manicule, in one import.

The same shape every plugin's conftest has. ``model_cache`` matters for the same reason it does
in ``manicule-mlx``: ``manicule_environment`` redirects ``XDG_CACHE_HOME`` per test and
``huggingface_hub`` follows it, so the real-server suite — which fetches a ``tokenizer.json``
— would miss a vocabulary already sitting in this machine's cache.

The synthetic server is deliberately *not* exported from here: every ``packages/*/tests/
conftest.py`` imports as plain ``conftest``, so anything this module published would be
reachable only by whichever one pytest happened to load first. It lives in ``ollama_fake.py``,
whose name is unique across the repository for the same reason.
"""

from __future__ import annotations

from manicule.testing.fixtures import manicule_environment, model_cache, settings

__all__ = ["manicule_environment", "model_cache", "settings"]
