"""Everything this package's suite needs from manicule, in one import.

The same shape every plugin's conftest has. ``model_cache`` is not optional here: without it
``manicule_environment`` redirects ``XDG_CACHE_HOME``, ``huggingface_hub`` follows it, and the
parity suite reports weights absent that are sitting in this machine's cache.

The qualification harness is deliberately *not* loaded here: every
``packages/*/tests/conftest.py`` imports as plain ``conftest``, so anything this module
exported would be reachable only by whichever one pytest happened to load first.
``tests/test_memory.py`` loads the harness itself, by path.
"""

from __future__ import annotations

from manicule.testing.fixtures import manicule_environment, model_cache, settings

__all__ = ["manicule_environment", "model_cache", "settings"]
