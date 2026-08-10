"""Bounding a parse worker's memory, identically on every platform manicule runs on.

The obvious mechanism is ``RLIMIT_AS`` in the child before it does any work. On Linux that
works. **On macOS it does not**, which matters because macOS on Apple Silicon is the platform
manicule is built for. Measured on Darwin, under both the system Python 3.9 and Homebrew
Python 3.13::

    RLIMIT_AS:   (9223372036854775807, 9223372036854775807)
    RLIMIT_DATA: (9223372036854775807, 9223372036854775807)
    RLIMIT_RSS:  (9223372036854775807, 9223372036854775807)
      set RLIMIT_AS soft=256MiB -> ValueError: current limit exceeds maximum limit
      allocated 512 MiB anyway -> caps NOT enforced

All three report unlimited, all three refuse to be set, and a child allocates half a gigabyte
under a nominal 256 MiB cap without complaint. A design that says "the worker sets
``RLIMIT_AS``" is correct on CI and inert on the developer's machine — which is the worst
place for a resource limit to be missing, because that is where the malformed PDF gets opened
first.

**So the enforced quantity is resident memory, sampled by the parent, on every platform.**
That is a deliberate departure from a purely platform-split design, and the reason is the
project's own rule that a document must ingest identically wherever it runs. Enforcing
address space on Linux and resident memory on Darwin makes the *mechanism* differ, which is
fine, but it also makes the *outcome* differ: a parser that maps a large arena without
touching it dies on one machine and succeeds on the other, and "it ingested on my laptop" is
exactly the class of answer this project exists not to give.

``RLIMIT_AS`` is still applied in the child wherever the kernel accepts it, at
:data:`ADDRESS_SPACE_HEADROOM` times the resident bound. Loose enough that it cannot fire
before the uniform check in any realistic case, tight enough to catch an allocation so sudden
that sampling misses it — a backstop, not the policy.

Sampling can overshoot between ticks. That is accepted and stated: the goal is to stop a
runaway before it takes the machine down, not to enforce a byte-exact quota.
"""

from __future__ import annotations

import contextlib
import os
import resource
import shutil
import signal
import subprocess

ADDRESS_SPACE_HEADROOM = 4
"""How much looser the kernel backstop is than the resident bound this module enforces.

Not a tuning knob. Address space and resident memory are different quantities — an allocator
reserving a large arena it never touches is normal — so a backstop set *at* the resident
bound would fire on well-behaved parsers, and it would fire only on the platform where it can
be set. Four is enough headroom that it does not.
"""


def limit_address_space(soft_bytes: int) -> bool:
    """Cap this process's address space, and say whether the kernel accepted it.

    Called in the child before it does any work. Returns ``False`` rather than raising where
    the limit is unavailable, because the caller's response is not to fail — it is to rely on
    the sampling the parent performs anyway.
    """
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
    except (OSError, ValueError, AttributeError):  # pragma: no cover - platform dependent
        return False
    ceiling = soft_bytes if hard == resource.RLIM_INFINITY else min(soft_bytes, hard)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
    except (OSError, ValueError):
        # Darwin, measured above: the limit reports unlimited and refuses to be set.
        return False
    return True


def resident_bytes(pid: int) -> int | None:
    """Resident memory of a process, or ``None`` if it cannot be read.

    ``psutil`` when it is installed, and ``ps`` when it is not. ``None`` means "unknown", not
    "zero": a caller must not read a failure to measure as a process using no memory, which
    is the reading that turns a missing dependency into a missing limit.
    """
    from_psutil = _psutil_rss(pid)
    if from_psutil is not None:
        return from_psutil
    return _ps_rss(pid)


def _psutil_rss(pid: int) -> int | None:
    try:
        import psutil  # noqa: PLC0415 - optional, and the fallback below is the reason
    except ImportError:  # pragma: no cover - exercised only where the extra is absent
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except Exception:  # noqa: BLE001 - psutil raises a family of process-gone errors
        return None


def _ps_rss(pid: int) -> int | None:
    """Resident kilobytes from ``ps``, at the cost of a fork per poll.

    The path taken when the optional dependency is absent. It is slower and it is correct,
    which is the right way round: a limit that silently stops applying because a package was
    not installed is worse than one that costs a fork.
    """
    executable = shutil.which("ps")
    if executable is None:  # pragma: no cover - a system without ps
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - absolute path from `which`, fixed arguments
            [executable, "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    reported = completed.stdout.strip()
    if not reported.isdigit():
        return None
    return int(reported) * 1024


def kill(pid: int) -> None:
    """``SIGKILL`` a process, ignoring one that has already gone.

    ``SIGKILL`` rather than ``SIGTERM`` because the process being stopped is, by hypothesis,
    not responding: it is inside a native extension that observes no signal handler and
    returns no control to Python. It works identically on Darwin and Linux — verified, child
    exit code ``-9`` — which is what keeps the outcome the same even where the detection
    differs.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGKILL)


__all__ = ["ADDRESS_SPACE_HEADROOM", "kill", "limit_address_space", "resident_bytes"]
