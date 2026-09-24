"""Who the operation in progress is being performed for.

The surfaces authenticate; the service acts. Most of what the service does is the same whoever
asked, which is why its methods do not take a caller — but three things are not, and each of
them fails silently if a surface forgets to pass one:

* **The audit trail** records who did something. A trail whose rows all say "somebody" is a
  list of events, not an audit.
* **Ownership.** A key a signed-in person mints is theirs: it can hold no more authority than
  they do, and it goes when they do.
* **Security alerts** count what one caller does across requests — how many documents they
  read, how many addresses present one key.

So the surface that authenticated a request says so once, with :func:`acting_as`, and the
service reads it with :func:`current`. It is a context variable rather than a parameter
because it has to reach an audit write three calls deep without every signature in between
growing an argument nobody else reads.

**The default is the operator at this machine**, and that is a decision rather than a fallback.
Everything that does not pass through a network surface — the command line, stdio MCP, a
served process's scheduler — runs with the authority of whoever started the process, which is
the same authority ``security.auth.mode = none`` gives a loopback caller. A network surface
that forgot to call :func:`acting_as` would therefore act with the operator's authority, and
that is why every one of them does it in exactly one place: the middleware or guard that
resolved the credential.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.config.settings import Role

if TYPE_CHECKING:
    from collections.abc import Generator

LOCAL_ACTOR = "local"
"""What the audit trail records for the operator at this machine."""

ANONYMOUS_ACTOR = "anonymous"
"""What it records for a network caller that presented no usable credential."""


@dataclass(frozen=True, slots=True)
class Caller:
    """The authenticated party an operation is being performed for."""

    role: Role | None = None
    """``None`` for the local operator, who holds every authority the process has."""

    key_id: str | None = None
    """The API key presented, when one was."""

    user_id: str | None = None
    """The signed-in person, or the person who minted the key that was presented."""

    address: str = ""
    """The client address, as the proxy policy resolved it. Empty when there was none."""

    rate_limit: int | None = None
    """This caller's own requests-per-minute cap, when a presented key carries one.

    ``None`` means "no override" — the caller is metered at
    ``security.rate_limit.per_minute`` like everyone else. Carried here rather than looked up
    again at the point of charging, because the key row that named it has already been read
    once, by whichever surface resolved this caller; reading it a second time would be a second
    place the two could disagree.
    """

    @property
    def is_local(self) -> bool:
        """Whether this is the operator at this machine rather than a network caller."""
        return self.role is None

    @property
    def actor(self) -> str:
        """Who the audit trail records: the person, else the key, else nobody in particular.

        The person first, because a key a person minted is that person acting.
        """
        if self.is_local:
            return LOCAL_ACTOR
        return self.user_id or self.key_id or ANONYMOUS_ACTOR

    def holds(self, floor: Role) -> bool:
        """Whether this caller's authority reaches ``floor``."""
        if self.role is None:
            return True
        return RANK[self.role] >= RANK[floor]


RANK: dict[Role, int] = {Role.VIEWER: 0, Role.MEMBER: 1, Role.ADMIN: 2}
"""Least authority first. The one ranking every surface and the service compare against."""

LOCAL = Caller()

_CURRENT: ContextVar[Caller] = ContextVar("manicule_caller", default=LOCAL)


def current() -> Caller:
    """The caller of the operation in progress."""
    return _CURRENT.get()


@contextmanager
def acting_as(caller: Caller) -> Generator[Caller]:
    """Perform everything inside this block for ``caller``, and restore the previous one after.

    Restored with the token rather than by setting the default back, so that nesting — a
    surface calling another surface's code — unwinds to exactly what was there.
    """
    token = _CURRENT.set(caller)
    try:
        yield caller
    finally:
        _CURRENT.reset(token)
