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
   module exists to avoid — **or** ``allow_unauthenticated``, which is argv for the same reason
   the second is.

Any one missing is a refusal naming which. None of the three can be reached by omission: the
absent value in each case is the safe one.

**The third condition has a way out and the second does not, which is the asymmetry to
understand.** ``--no-authentication`` exists for the deployment manicule is actually for — one
operator, one corpus, a private network they own — where demanding an API key is ceremony
between a person and their own index. It is on the command line for the reason ``allow_public``
is: a setting that could grant it would make an unauthenticated listener reachable by editing a
file, and the whole point is that it takes a person at a terminal. It does not make the bind
safe and does not pretend to. What it buys is that the choice is recorded in the command that
made it, said out loud at startup, and reported by ``manicule doctor`` as a finding for as long
as it holds.

**And it carries a risk that is accepted rather than mitigated, which is the part to read
twice.** With authentication off every anonymous caller resolves to an administrator — see
:mod:`manicule.api.security` — so a socket serving ``document_create`` unauthenticated is a
corpus that anything able to route to the port may write into, over MCP and over
``POST /api/v1/documents`` alike. That corpus is read back by assistants as standing
instructions, so a write into it places text in front of future sessions on machines that never
touched this deployment.

**That is served anyway, deliberately**, because it is the capability the flag exists for:
assistants writing memories into a corpus from machines running no manicule of their own, and a
read-only surface would make that deployment pointless. The operator passing this flag is
asserting that the network in front of the process is one they own — a private LAN, or a cluster
behind an ingress that authenticates for us. manicule cannot verify that assertion, exactly as it
cannot verify ``allow_public``; what it can do is refuse to let a *file* make it, say so at
startup naming the corpus, and report it from ``manicule doctor`` for as long as it holds.

Stdio transports never come here at all, and that is the point of
:func:`stdio` — a bind decision that is not made cannot be made wrongly, so the MCP server's
default mode has no address to get wrong.

:func:`require_authoring_authentication` is the one rule here that is stricter than the three
above: a socket carrying ``document_create`` needs authentication even on loopback, because
"reachable only from this machine" is a weaker statement about a write into a corpus than it is
about a read out of one. ``allow_unauthenticated`` waives it, and that waiver is the whole
point of the flag rather than a hole in it: the refusal exists so nobody serves authoring
unauthenticated *without meaning to*, and an operator who typed the argument means to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.config.settings import AuthMode
from manicule.core.errors import PolicyError

if TYPE_CHECKING:
    from manicule.config.settings import Settings

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})
"""Addresses that reach only this machine.

Matched by name rather than resolved, deliberately: a resolver can be told that
``localhost`` is something else, and a bind decision that depends on ``/etc/hosts`` is not
a decision.

Bind *addresses* only — no CIDR forms. A netmask is a way of describing a range to a
firewall, not something a socket can be bound to, and admitting one here would pass the
loopback check and then fail at the bind with an error about the wrong thing. Other
addresses in ``127.0.0.0/8`` are absent for a different reason: they are genuinely
loopback and are refused anyway, which is the safe direction to be wrong in.
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


def require_authoring_authentication(
    settings: Settings, *, allow_unauthenticated: bool = False
) -> None:
    """Refuse to serve authoring over a socket without authentication.

    Called by every path that puts a server on a port —
    :func:`manicule.mcp.serve.address_for` for the MCP-only transport and
    :func:`manicule.api.app.build_app` for the application everything else is served from. Not
    called for stdio, which has no port for anything to reach.

    **This is stricter than :func:`resolve_bind`, deliberately.** That one admits a loopback bind
    with no authentication at all, which is right for a surface that reads: the port is reachable
    only from this machine. ``document_create`` writes into a corpus, and on a laptop "only from
    this machine" includes every process on it and every page a browser has open. So the rule for
    authoring is the one :func:`resolve_bind` applies to a wide bind, applied to every bind.

    **The condition is authoring being configured, not the tool existing.** Unconfigured, it is
    published and refuses every call with a :class:`~manicule.core.errors.ConfigError` naming the
    settings it needs — so it carries no authority, and an installation that never wanted it is
    not asked to turn authentication on for a feature it does not use.

    Refusing at startup rather than per call, because the alternative is a server that runs,
    accepts connections and declines the one operation somebody deployed it for — discovered by a
    client, at the far end, after a turn has been spent on it.

    **``--no-authentication`` waives it, and the waiver is the point rather than a hole.** This
    refusal exists so that nobody serves authoring unauthenticated *by omission* — a corpus read
    back as standing instructions is not something to expose by forgetting a setting. An operator
    who typed the argument has not forgotten anything: they are asserting the network in front of
    this process is one they own, which is the deployment authoring was built for. The default is
    unchanged, so every installation that says nothing still gets the refusal.

    The waiver is honest about its size. It opens ``document_create`` on the socket **and**
    ``POST /api/v1/documents`` on the HTTP surface, whose member floor an anonymous administrator
    clears — one decision, both doors, said plainly here and at startup rather than mitigated on
    one surface and quiet about the other.

    Args:
        settings: Configuration. ``authoring.configured`` and ``security.auth.mode`` decide.
        allow_unauthenticated: The operator's explicit opt-in, from the command line. **Not a
            setting**, for the reason :func:`resolve_bind` gives about ``allow_public``.

    Raises:
        PolicyError: Authoring is configured, ``security.auth.mode`` is ``none``, and nobody said
            on the command line that they meant it.
    """
    if not settings.authoring.configured or settings.security.auth.mode is not AuthMode.NONE:
        return
    if allow_unauthenticated:
        return
    msg = (
        f"refusing to serve on a socket with authoring configured and no authentication. "
        f"`authoring.source` is {settings.authoring.source!r}, so document_create can write into "
        f"that corpus, and `security.auth.mode` is 'none', so anything that can reach the port "
        f"can call it — on loopback that is every process and every page on this machine. Set "
        f"security.auth.mode to 'api_key' or 'oauth', or clear `authoring.source` and "
        f"`authoring.collections` to serve this installation read-only. Pass "
        f"--no-authentication to serve it anyway on a network you own: the socket then carries "
        f"document_create, and anything that can route to the port may call it. Authoring over "
        f"stdio needs none of this: a pipe has no port."
    )
    raise PolicyError(msg)


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
    allow_unauthenticated: bool = False,
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
        allow_unauthenticated: The other explicit opt-in, and **not a setting** for exactly
            the same reason: a configuration key granting this would put an unauthenticated
            listener one file edit away. It satisfies the third condition rather than removing
            it, and satisfies only that one — a wide bind still needs ``allow_public`` beside
            it, because "I accept no authentication" and "I meant to bind the network" are two
            statements and neither implies the other.

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
    if settings.security.auth.mode is AuthMode.NONE and not allow_unauthenticated:
        problems.append(
            "security.auth.mode is 'none', so anything that can reach the port could read "
            "the whole index. Set security.auth.mode to 'api_key' or 'oauth' first, or pass "
            "--no-authentication to say you accept that; there is no setting that grants it "
            "either"
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
    "require_authoring_authentication",
    "resolve_bind",
    "stdio",
]
