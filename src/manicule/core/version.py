"""The version plugins declare compatibility against.

Read from installed distribution metadata rather than written as a literal, so it cannot
drift from the version that is actually running.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

DISTRIBUTION_NAME = "manicule"


def _installed_version() -> str:
    try:
        return _distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
        return "0.0.0.dev0"


CORE_VERSION = _installed_version()

__all__ = ["CORE_VERSION", "DISTRIBUTION_NAME"]
