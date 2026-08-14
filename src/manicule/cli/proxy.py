"""Where a write command runs, and how it gets there.

Since #126 there is one writer per data directory. A served manicule *is* that writer for its
whole life, so a command that writes has three possible fates and exactly one of them is right
for each case:

============================  ==================================================================
A server is listening         Send it there. It holds the lock, the schedule and the session.
No server is listening        Refuse, and name the command that starts one.
Something else holds the lock Refuse, from :class:`~manicule.ingest.recovery.InstanceLock`.
============================  ==================================================================

**The middle row is a deliberate refusal rather than a fallback**, and it is worth being plain
about the cost: ``manicule connector sync`` on a machine with no server used to work and now
says to start one. What buys that back is the credential. A Confluence session is held in the
server's memory and nowhere else — no keychain, no file, no environment variable — so a sync in
a process that is not the server has no session to use and never will. Falling back to a local
run would mean falling back to a run that cannot authenticate, reported as a sync that failed
rather than as an arrangement nobody set up.

**Nothing here starts a server.** A command line that spawned a background process the operator
did not ask for is a surprise that is hard to undo and harder to notice: it would hold a lock,
outlive the terminal, and be found later by somebody wondering what has the data directory. The
refusal names ``manicule serve`` and stops.

**Progress goes to stderr while the operation runs.** stdout carries the envelope and nothing
else, so ``manicule connector sync handbook --json | jq`` is still one JSON document however
long the sync took and however much it said on the way.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from manicule.app import control
from manicule.app.dispatch import error_info
from manicule.app.results import Envelope, failed
from manicule.config.loader import load_settings
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from manicule.app.commands import Command
    from manicule.connectors.credentials import BrowserSession

__all__ = ["HandoverStore", "NoServerError", "forward", "listening", "refuse", "socket_for"]


class NoServerError(ManiculeError):
    """A command that writes was run and there is no server to run it.

    Its own type rather than a bare :class:`~manicule.core.errors.ConfigError` so that
    :data:`~manicule.app.dispatch._HINTS` can say something true of this and of nothing else,
    and so a script can tell "start a server" from "your configuration is wrong".
    """


def socket_for(overrides: Mapping[str, Any]) -> Path:
    """The control socket for the data directory this invocation would use.

    Configuration is *loaded* rather than a runtime opened, for the reason
    :func:`~manicule.cli.serving.stop_running` gives: all this needs is a path, and discovering
    plugins to find one would make every write command fail on an installation whose plugins
    are the reason somebody is running it.

    Raises:
        ManiculeError: Configuration will not load. Left to the caller, which turns it into the
            same failure envelope every other configuration problem produces.
    """
    return control.socket_path(load_settings(**overrides).data_dir)


def listening(overrides: Mapping[str, Any]) -> Path | None:
    """The socket a server is answering on, or ``None`` when there is not one.

    ``None`` for a configuration that will not load as well as for a server that is not there.
    That is not a swallowed error: the caller's next step either way is to run the command
    locally or refuse, and both of those load configuration themselves and report the failure
    properly. Reporting it from here would report it twice.
    """
    try:
        path = socket_for(overrides)
    except (ManiculeError, ValueError, OSError):
        return None
    return path if control.is_serving(path) else None


def refuse(command: Command, *, workspace: str, overrides: Mapping[str, Any]) -> Envelope:
    """The envelope for a write command with no server, naming what to do about it.

    It names the operation, so ``manicule index`` and ``manicule connector sync`` do not both
    produce a sentence that could be about either.
    """
    try:
        settings = load_settings(**overrides)
        where = f" for {settings.data_dir}"
        # A socket that is there and cannot be used is a different situation with a different
        # fix, and telling somebody to start a server when one is already running would send
        # them to a refusal from the instance lock naming a process they did not know about.
        blocked = control.unusable(control.socket_path(settings.data_dir))
    except (ManiculeError, ValueError, OSError):
        # Configuration that will not load is reported by the caller's own path. Here it only
        # costs the message a clause.
        where, blocked = "", ""
    if blocked:
        return failed(
            command.op,
            workspace,
            error_info(
                NoServerError(
                    f"`{command.op}` writes, and the manicule server's control socket{where} "
                    f"cannot be used: {blocked} A server may well be running — this is about "
                    f"the socket rather than the process. Put the socket back to mode 0600 "
                    f"owned by you, or stop the server with `manicule stop` and start it again."
                )
            ),
        )
    return failed(
        command.op,
        workspace,
        error_info(
            NoServerError(
                f"`{command.op}` writes, and no manicule server is running{where}. A served "
                f"manicule holds the writer lock, runs the schedule and is the only place a "
                f"captured Confluence session lives, so write commands go to it rather than "
                f"opening the data directory themselves. Start one with `manicule serve`, then "
                f"run this again. Nothing has been started for you."
            )
        ),
    )


async def forward(path: Path, command: Command, *, workspace: str) -> Envelope:
    """Run ``command`` in the server listening on ``path``, and return what it answered.

    The envelope comes back exactly as the server built it — same operation, same payload, same
    error, same hint — so the caller prints it with the same function it would have used for a
    local run. Nothing is re-wrapped or re-worded in transit, which is why a proxied command and
    a local one agree: they are one object, not two renderings.

    Args:
        path: The socket, already known to be answering.
        command: What to run.
        workspace: The tenant, for the envelope a failure here has to carry — the server names
            its own on everything it answers.

    Returns:
        The server's envelope, or a failure envelope describing why the socket did not produce
        one.
    """
    try:
        answered = await control.connect(
            path, command.invoke(workspace), on_progress=_show_progress
        )
    except (ManiculeError, ValueError, OSError) as exc:
        # ``ValueError`` covers pydantic's ``ValidationError``, which is what a *response* frame
        # this version does not understand raises out of ``read_response``. Without it a server
        # one version ahead of its client reaches the terminal as a traceback rather than as the
        # failure envelope every other outcome arrives in.
        return failed(command.op, workspace, error_info(exc))
    try:
        return Envelope.model_validate(answered)
    except ValueError as exc:
        # A server one version away from its client. Reported as itself rather than as a crash
        # in a validator, because the operation may well have happened.
        msg = (
            f"the manicule server answered `{command.op}` with something this version does not "
            f"recognize as a result ({exc.__class__.__name__}). The operation may have run; "
            f"check with `manicule connector list` or `manicule document list`. If the server "
            f"and the command line are different versions, make them the same."
        )
        return failed(command.op, workspace, error_info(ManiculeError(msg)))


class HandoverStore:
    """A session store whose writes are hand-offs to the running server.

    ``manicule connector login`` runs in *this* process, and it has to: ``--browser`` opens a
    window, and a window opened by a background server is a window nobody is sitting at. But
    the syncs run in the server, so the credential has to end up there. This is the seam
    between those two facts, and it is a
    :class:`~manicule.connectors.sessions.SessionStore` rather than a special case inside the
    login flow — so a session captured by a paste, by a driven browser or from a state file is
    verified and handed over by one piece of code, exactly as it was stored by one before.

    **:meth:`load` always answers ``None``, and that is not a stub.** A command-line process
    holds no sessions and must not start holding one: a copy kept here would be a live
    credential in a short-lived process nobody thinks of as a credential store, which is most of
    what was wrong with the arrangements this replaces. The verification that decides whether a
    session is worth handing over is a request to the instance, not a lookup.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def describe(self) -> str:
        return "the running manicule server's memory, which does not survive it"

    def load(self, base_url: str) -> BrowserSession | None:
        del base_url
        return None

    def holding(self) -> dict[str, str]:
        """Nothing, for the same reason :meth:`load` answers ``None``.

        Not a stub and not a gap in what this can see: a command-line process holds no session,
        so an empty answer is the true one. ``doctor``'s session check reads an empty store as
        "ask the server" rather than as "there is none", which is what keeps this honest instead
        of merely quiet.
        """
        return {}

    async def save(self, session: BrowserSession) -> None:
        """Hand a verified session to the server.

        Raises:
            ManiculeError: The server did not take it. The session is *not* kept anywhere here
                as a consolation — a credential with nowhere to go is one to capture again, and
                capturing again is a few seconds of clicking.
        """
        await control.connect(
            self._path,
            control.Handover(
                base_url=session.base_url,
                account=session.account,
                captured_at=session.captured_at.isoformat(),
                cookies=dict(session.cookies),
            ),
            on_progress=_show_progress,
        )

    async def forget(self, base_url: str) -> bool:
        """Ask the server to drop the session it holds for this instance."""
        answered = await control.connect(
            self._path, control.Forget(base_url=base_url), on_progress=_show_progress
        )
        data = answered.get("data")
        return bool(data.get("forgotten")) if isinstance(data, dict) else False


def _show_progress(message: str) -> None:
    """One progress line, on stderr, as it arrives.

    Written straight to the stream rather than through Rich. Progress is a running commentary
    on somebody else's process; a console object would want to own the cursor, and a long sync
    sharing a terminal with a shell is not a place to be repainting lines.
    """
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()
