"""Running one operation and turning whatever happened into an envelope.

Shared by both surfaces, and that is the whole point. If the command line mapped errors one
way and the MCP tool another, the two would disagree about what a failure *is* — and the tool
is the one an assistant calls unattended, where a stack trace is not an answer and a bare
"error" is not either.

Every failure that a caller could act on becomes ``ok: false`` with a type, a message and,
where there is something specific to say, a hint. Anything else propagates: a bug in manicule
is not a result, and dressing one up as a well-formed envelope is how a broken installation
reports success at being broken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.app.results import Envelope, ErrorInfo, Payload, failed, succeeded
from manicule.app.tenancy import CrossWorkspaceError
from manicule.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    ManiculeError,
    PolicyError,
    UnknownComponentError,
    UnknownEntityError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_HINTS: dict[type[Exception], str] = {
    ConfigError: "Check the setting this names, then run `manicule doctor`.",
    # Covers two shapes, and naming only the first sent readers hunting for a second setting
    # that is not there. `policy_problems` reports settings that are individually valid and
    # jointly wrong — those do list every one. But `require_sharing_enabled`, and a connector
    # configured `enabled = false`, raise this too, and there the whole cause is the single
    # setting already named. The hint has to be true of both, because it is printed under both.
    PolicyError: (
        "Configuration forbids this. Either the setting named above disallows it, or two "
        "settings that are each valid disagree — where it is the second, the message lists "
        "every one of them and they are fixed together."
    ),
    UnknownComponentError: (
        "Nothing installed provides that component. Install the distribution that does, or "
        "run `manicule plugin list` to see what is available."
    ),
    # True of every entity this error covers, which is not only rows in this workspace: the
    # same type is raised for a path that does not exist, where "run the matching `list`
    # command to see what this workspace holds" named an action that does not apply and an
    # object the caller was not asking about.
    UnknownEntityError: (
        "Check the name or path in the message above. The matching `list` command — "
        "`manicule document list`, `connector list`, `workspace list` — shows what exists."
    ),
    FingerprintMismatchError: (
        "The index was built by a different chunker or embedder. Re-index, or point at the "
        "data directory that matches this configuration."
    ),
    CrossWorkspaceError: (
        "Something returned data belonging to another workspace and it was refused. This is a "
        "defect, not a configuration problem — report it with the message above."
    ),
}
"""What to do about each kind of failure, in the words of whoever has to do it.

Keyed by exact type rather than by ``isinstance``, so a subclass added later gets a hint
written for it rather than inheriting one written for its parent.
"""


def error_info(exc: Exception) -> ErrorInfo:
    """Describe a failure in the shape the contract promises."""
    return ErrorInfo(
        type=type(exc).__name__,
        message=str(exc),
        hint=_HINTS.get(type(exc), ""),
    )


async def run_op(op: str, workspace: str, call: Callable[[], Awaitable[Payload]]) -> Envelope:
    """Run one operation and wrap the outcome.

    Args:
        op: The operation's name. The same string on both surfaces, so a log line, a shell
            pipeline and a tool call all name it identically.
        workspace: The tenant it ran in. On the envelope whether it succeeded or not.
        call: A zero-argument coroutine function. A thunk rather than a coroutine, so nothing
            is started until this function is ready to catch what it raises.

    Returns:
        A success envelope carrying the payload, or a failure envelope carrying the error.

    Raises:
        Exception: Anything that is not a :class:`~manicule.core.errors.ManiculeError`, a
            :class:`ValueError` or an :class:`OSError`. Those three are outcomes a caller can
            act on; the rest are defects, and a defect reported as a tidy result is a defect
            nobody fixes.
    """
    try:
        payload = await call()
    except (ManiculeError, ValueError, OSError) as exc:
        return failed(op, workspace, error_info(exc))
    return succeeded(op, workspace, payload)


__all__ = ["error_info", "run_op"]
