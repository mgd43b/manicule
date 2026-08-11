"""Where a server may listen. Loopback unless three separate things say otherwise.

manicule indexes whatever it was pointed at, and answers questions about it in full. A
process serving that on a routable address with no authentication is an open document index,
readable by anybody who can reach the port — and the failure is silent, because from the
inside it looks exactly like a working install.

So there is one function that decides a bind address, every server goes through it, and it
refuses by default. A wide bind needs **all three** of:

1. a host that is not loopback — and the default is loopback, so this is always something a
   person wrote down;
2. ``allow_public``, which no configuration file can set and no default supplies — the caller
   passes it, and the only caller that does is a command-line flag a person typed;
3. authentication switched on, because a routable address without it is the defect this whole
   module exists to avoid.

Any one missing is a refusal naming which. None of the three can be reached by omission: the
absent value in each case is the safe one.

Stdio transports never come here at all, and that is the point of
:func:`stdio` — a bind decision that is not made cannot be made wrongly, so the MCP server's
default mode has no address to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.config.settings import AuthMode
from manicule.core.errors import PolicyError

if TYPE_CHECKING:
    from manicule.config.settings import Settings

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "127.0.0.1/32", "::ffff:127.0.0.1"})
"""Addresses that reach only this machine.

Matched by name rather than resolved, deliberately: a resolver can be told that ``localhost``
is something else, and a bind decision that depends on ``/etc/hosts`` is not a decision.
"""

EVERY_INTERFACE = frozenset({"0.0.0.0", "::", "*", ""})  # noqa: S104 - named here so it is refusable
"""Hosts that mean "every interface".

Enumerated so the refusal can say so specifically. The empty string is here because a blank
value from a config file or an environment variable reads as "unset" and binds everything.
"""


MIN_PORT = 1
MAX_PORT = 65535
"""The range a TCP port can occupy. Checked here so a bad one is refused before a socket."""


def is_loopback(host: str) -> bool:
    """Whether ``host`` reaches only this machine."""
    return host.strip().lower() in LOOPBACK_HOSTS


def is_every_interface(host: str) -> bool:
    """Whether ``host`` means every interface on the machine."""
    return host.strip().lower() in EVERY_INTERFACE


@dataclass(frozen=True, slots=True)
class Bind:
    """A decided listening address."""

    host: str
    port: int
    loopback: bool
    every_interface: bool = False

    def describe(self) -> str:
        """One line, for a banner or a log."""
        scope = "loopback only" if self.loopback else "REACHABLE FROM THE NETWORK"
        return f"{self.host}:{self.port} ({scope})"


def stdio() -> None:
    """The transport that binds nothing.

    Exists as a named no-op so that "this path opens no socket" is a statement in the code
    rather than an absence a reader has to notice. The MCP server's default transport is
    stdio, and a client that speaks it needs no address at all.
    """
    return


def resolve_bind(
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
) -> Bind:
    """Decide where to listen, refusing anything wide that was not asked for three times.

    Args:
        settings: Configuration. ``security.transport`` supplies the defaults, and its own
            default host is loopback.
        host: An override from the command line. ``None`` means "whatever is configured",
            which is the case that must never be able to widen the bind.
        port: An override from the command line.
        allow_public: The operator's explicit opt-in. **Not a setting.** A file that could
            grant this would make a wide bind reachable by editing configuration, and the
            whole point is that it takes a person at a terminal.

    Returns:
        The decided address.

    Raises:
        PolicyError: The address is not loopback and something required to widen it is
            missing. The message names which, and what to do instead.
    """
    transport = settings.security.transport
    chosen = (host if host is not None else transport.bind_host).strip()
    chosen_port = port if port is not None else transport.port
    if not (MIN_PORT <= chosen_port <= MAX_PORT):
        msg = f"port {chosen_port} is outside {MIN_PORT}-{MAX_PORT}"
        raise PolicyError(msg)

    if is_loopback(chosen):
        return Bind(host=chosen, port=chosen_port, loopback=True)

    everywhere = is_every_interface(chosen)
    where = "every interface on this machine" if everywhere else f"the address {chosen!r}"
    problems: list[str] = []
    if not allow_public:
        problems.append(
            f"binding {where} was not explicitly requested. Pass --allow-public-bind to say "
            f"you mean it; there is no setting that grants this, because a wide bind should "
            f"take a person rather than a file"
        )
    if settings.security.auth.mode is AuthMode.NONE:
        problems.append(
            "security.auth.mode is 'none', so anything that can reach the port could read "
            "the whole index. Set security.auth.mode to 'api_key' or 'oauth' first"
        )
    if problems:
        joined = "\n  - ".join(problems)
        msg = (
            f"refusing to bind {where}:\n  - {joined}\n"
            f"Leave security.transport.bind_host at 127.0.0.1 to serve this machine only."
        )
        raise PolicyError(msg)

    return Bind(host=chosen, port=chosen_port, loopback=False, every_interface=everywhere)


__all__ = [
    "EVERY_INTERFACE",
    "LOOPBACK_HOSTS",
    "MAX_PORT",
    "MIN_PORT",
    "Bind",
    "is_every_interface",
    "is_loopback",
    "resolve_bind",
    "stdio",
]
