"""Where a workspace's vectors live on disk, and the pin that keeps a generation there.

Both answers are arithmetic over a path and a lock over a directory. Neither needs a vector
engine to give them, and until this module existed neither could be asked without one: the
workspace digest and the generation pin lived in :mod:`manicule.storage.vectors`, which
imports ``lancedb`` at module scope. Asking "which directory is this workspace's?" therefore
dlopened LanceDB's native extension — and on a CPU without AVX2 that is not a slow import,
it is ``SIGILL``. An installation that configures Qdrant could not start on the hardware the
Qdrant backend exists to serve.

So this is the same rule :mod:`manicule.storage.vector_schema` states about a row's field
names, applied to a row's location: nothing here imports a database, and that is load-bearing
rather than tidy. ``tests/test_import_boundary.py`` holds the line.

:mod:`manicule.storage.vectors` re-exports both names, so the Lance store and its tests
continue to import them from where they have always been.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

_EXCLUSIVE_PINS: ContextVar[frozenset[Path]] = ContextVar(
    "manicule_exclusive_vector_pins", default=frozenset()
)


def workspace_vector_directory(root: Path, workspace_id: str) -> Path:
    """Opaque, stable physical namespace for one workspace's independent vector identity."""
    digest = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()
    return root / "workspaces" / digest


@asynccontextmanager
async def generation_pin(directory: Path, *, exclusive: bool = False) -> AsyncGenerator[None]:
    """Cross-process pin preventing cleanup from deleting a generation during an operation."""
    resolved = await asyncio.to_thread(directory.resolve)
    if resolved in _EXCLUSIVE_PINS.get():
        yield
        return
    pins = directory.parent / ".pins"
    pins.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(pins / f"{directory.name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        while True:
            try:
                fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        token = None
        if exclusive:
            token = _EXCLUSIVE_PINS.set(_EXCLUSIVE_PINS.get() | {resolved})
        try:
            yield
        finally:
            if token is not None:
                _EXCLUSIVE_PINS.reset(token)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
