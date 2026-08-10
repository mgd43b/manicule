"""Test configuration.

The environment isolation and the ``settings`` fixture are shipped in
:mod:`manicule.testing.fixtures` rather than written here, so that a plugin author gets the
same setup with one import and manicule's own suite proves that import works.

The storage fixtures are local: they build a real migrated database in a temporary directory,
which is a manicule-internal concern rather than something a plugin author needs.
"""

from __future__ import annotations

from manicule.testing.fixtures import manicule_environment, settings
from tests.storage_helpers import data_dir, engine, store

__all__ = ["data_dir", "engine", "manicule_environment", "settings", "store"]
