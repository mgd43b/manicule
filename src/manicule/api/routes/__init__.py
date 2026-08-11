"""The eleven route groups, one module each — except where two are one decision.

``collections`` and ``tags`` share :mod:`manicule.api.routes.organisation` because they are
the same design seen twice, and the storage layer already treats them that way. ``health`` and
``websocket chat`` have their own modules for the opposite reason: each does something no
other route does — answering without an envelope, and authenticating without FastAPI's
dependency machinery — and a reader looking for either should find it in a file named for it.
"""

from __future__ import annotations

__all__: list[str] = []
