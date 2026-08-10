"""Test configuration.

The environment isolation and the ``settings`` fixture are shipped in
:mod:`manicule.testing.fixtures` rather than written here, so that a plugin author gets the
same setup with one import and manicule's own suite proves that import works.
"""

from __future__ import annotations

from manicule.testing.fixtures import manicule_environment, settings

__all__ = ["manicule_environment", "settings"]
