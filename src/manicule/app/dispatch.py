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

from manicule.app.results import Envelope, ErrorInfo, IngestReport, Payload, failed, succeeded
from manicule.app.tenancy import CrossWorkspaceError
from manicule.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    ManiculeError,
    PolicyError,
    UnknownComponentError,
    UnknownEntityError,
)
from manicule.ingest.capacity import CapacityRefusedError

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
    CapacityRefusedError: (
        "Free durable ingest capacity or raise the configured limit, then retry."
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
    if isinstance(payload, IngestReport) and payload.retry_required:
        reason = payload.incomplete_reason or ErrorInfo(
            type="IncompleteIngestError",
            message="the ingest run did not complete",
            hint="Run the same ingest operation again; its watermark was not advanced.",
        )
        return failed(op, workspace, reason, payload=payload)
    return succeeded(op, workspace, payload)


READ_ONLY_OPS: frozenset[str] = frozenset(
    {
        # Retrieval. Both of these *do* write one row — `query_logs`, through
        # `_record_query` — and that is the single deliberate exception to the classification
        # rather than an oversight. The write is observability, it is wrapped so that losing
        # the SQLite writer cannot fail the query, and `docs/ingest.md` 8.6 says so: a read
        # made conditional on a write is the failure that code exists to avoid, and refusing
        # a search because an index is being rebuilt would be the same failure one level up.
        "ask",
        "search",
        # Reads of what is stored.
        "document_list",
        "document_get",
        "index_status",
        "reembed_status",
        "stats",
        "collection_list",
        "collection_documents",
        "collection_counts",
        "collection_orphans",
        "connector_list",
        "workspace_list",
        "auth_list_keys",
        # Reads of configuration and of what is installed. These touch no data directory at
        # all; they are named rather than left to the default because the default is "writer",
        # and a command that takes an exclusive lock to print a setting would be absurd.
        "config_get",
        "plugin_list",
        # The diagnostic, and it is the one this classification most has to get right: a lock
        # that stops `doctor` running during the operation somebody wants diagnosed is worse
        # than no lock. It reads; `--fix` writes grammars and vocabularies into the *cache*
        # directory rather than the data directory, so even that stays on this side.
        "doctor",
        # Setting the machine up, on the same rule `doctor --fix` is here for: it writes the
        # configuration file and pre-seeds the cache, and touches no data directory at all.
        # Classifying it as a writer was harmless while a writer meant "takes a lock nobody
        # else wants". It stopped being harmless when write commands started going to a
        # server, because `init` is what somebody runs *before* there is anything to serve —
        # requiring one first is a cycle, and sending it to one would be a running process
        # rewriting configuration it had already read.
        "init",
        # Capturing a Confluence session. It used to write one to the macOS Keychain, which
        # was not the data directory either; now it hands the session to the server over the
        # control socket and writes nothing anywhere. What it still needs is a server to hand
        # it to, and `manicule connector login` refuses on its own when there is not one —
        # which is a different requirement from this lock and is stated where it applies.
        "connector_login",
        # Copies *out*. Both read the data directory and write somewhere else, so neither
        # needs the writer's exclusion — and making them writers would mean no backup could be
        # taken while a server was up, which is when somebody most wants one. The cost is
        # named in `docs/ingest.md` 8.6: a copy taken during active indexing is a copy of a
        # moving directory, which was true before this lock existed and is not fixed by it.
        "backup",
        "export",
    }
)
"""Operations that do not take the data directory's writer lock.

**The default is the other one.** :func:`writes` answers "yes" for any operation not named
here, so an operation added tomorrow and forgotten today is one that takes the lock — refused
while another writer runs, which is the safe direction to be wrong in. The unsafe direction is
a new command that quietly indexes beside a running sweep, and no list anybody maintains by
hand would reliably catch it.

Forgetting is nonetheless caught rather than merely survived:
``tests/app/test_process_exclusion.py`` enumerates every operation the command line can emit
and fails if one is in neither set, so the decision has to be made rather than defaulted into.
"""


def writes(op: str) -> bool:
    """Whether ``op`` needs the exclusive lock on its data directory.

    Args:
        op: The operation's name, as it appears on the envelope.

    Returns:
        ``True`` unless the operation is named in :data:`READ_ONLY_OPS`.
    """
    return op not in READ_ONLY_OPS


__all__ = ["READ_ONLY_OPS", "error_info", "run_op", "writes"]
