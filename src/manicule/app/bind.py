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

**Nor does it apply to an installation holding mail.** The index is not permission-aware, so a
source whose content is one person's — a mailbox — makes the whole installation private to
whoever may search it, and "anyone who can reach the port" on loopback is every other local
account and process. :func:`require_sensitive_source_authentication` refuses a socket without
authentication on every address, loopback included, for as long as an enabled connector
instance is of a *sensitive* type or the data directory still holds a document from a source one
has synced. ``--no-authentication`` cannot waive it, for team mode's reason: it extends a
single-operator arrangement to a network the operator owns, and mail is not the operator's to
extend.

**None of that applies in team mode.** Everything above about an anonymous caller rests on
there being one operator for that caller to be. ``mode = "team"`` says there is not, so
:func:`require_team_authentication` refuses a socket without authentication on every address,
loopback included, and refuses ``--no-authentication`` rather than letting it waive anything.

:func:`require_authoring_authentication` is the other rule here that is stricter than the three
above: a socket carrying ``document_create`` needs authentication even on loopback, because
"reachable only from this machine" is a weaker statement about a write into a corpus than it is
about a read out of one. ``allow_unauthenticated`` waives it, and that waiver is the whole
point of the flag rather than a hole in it: the refusal exists so nobody serves authoring
unauthenticated *without meaning to*, and an operator who typed the argument means to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.app.sensitive import exposure_when_unauthenticated
from manicule.config.settings import AuthMode, Mode
from manicule.core.errors import PolicyError

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

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


def require_team_authentication(settings: Settings, *, allow_unauthenticated: bool = False) -> None:
    """Refuse to serve a team installation on a socket without authentication.

    Called where a socket is made — :func:`resolve_bind` — and where an application is —
    :func:`manicule.api.app.build_app` — for the reason each of those refusals is repeated in
    the other: an ASGI application can be started by something other than ``manicule serve``.

    **Stricter than every other rule in this module, and not waivable.** The anonymous caller
    this module otherwise tolerates on loopback is the operator at this machine, holding the
    authority the command line already gives them; ``--no-authentication`` extends that
    arrangement to a network the operator owns. In ``team`` mode there is no such person —
    several people share the installation — so an anonymous caller would be an administrator
    who is nobody in particular, on loopback or anywhere else. The flag is refused rather than
    ignored, because an operator who typed it believes it did something.

    Args:
        settings: Configuration. ``mode`` and ``security.auth.mode`` decide.
        allow_unauthenticated: ``--no-authentication``, which team mode refuses outright.

    Raises:
        PolicyError: ``mode`` is ``team`` and either authentication is off or the command line
            asked to serve without it.
    """
    if settings.mode is not Mode.TEAM:
        return
    problems: list[str] = []
    if settings.security.auth.mode is AuthMode.NONE:
        problems.append(
            "security.auth.mode is 'none'. In team mode every caller must present a credential: "
            "set security.auth.mode to 'api_key' or 'oauth'"
        )
    if allow_unauthenticated:
        problems.append(
            "--no-authentication was passed. It makes every anonymous caller an administrator, "
            "which is a single-operator arrangement by definition and has no meaning when "
            "several people share the installation; drop the flag"
        )
    if problems:
        joined = "\n  - ".join(problems)
        msg = (
            f"refusing to serve workspace {settings.workspace!r} in team mode:\n  - {joined}\n"
            f"Set mode to 'personal' if this installation is one person's."
        )
        raise PolicyError(msg)


def require_sensitive_source_authentication(
    settings: Settings,
    sensitive_types: AbstractSet[str],
    *,
    allow_unauthenticated: bool = False,
) -> None:
    """Refuse to serve a sensitive source on a socket without authentication.

    Called where a socket is made — :func:`resolve_bind` — and where an application is —
    :func:`manicule.api.app.build_app` — beside :func:`require_team_authentication` and for its
    reason: an ASGI application can be started by something other than ``manicule serve``.
    Not called for stdio, which has no port.

    **Not waivable, and ``allow_unauthenticated`` is accepted only to be refused.** Everything
    :func:`resolve_bind` says about an anonymous caller being the operator at this machine is
    false for mail: loopback reaches every local account and process, and ``--no-authentication``
    reaches a network. An operator who typed the flag believes it did something, so the message
    says it did not.

    Two facts decide it (see :mod:`manicule.app.sensitive`): an enabled instance of a sensitive
    connector type, or any document in the data directory, in any state, from a source one has
    synced. The second is why disabling the connector or deleting it from configuration does
    not lift the refusal; only deleting the source's documents does.

    Args:
        settings: Configuration. ``security.auth.mode``, ``connectors`` and ``data_dir`` decide.
        sensitive_types: Connector types the registry marks sensitive. **Required**, with no
            default: an omitted argument would read as "nothing is sensitive", which is the
            failure this function exists to prevent. A caller with a service gets it from
            :meth:`~manicule.app.service.ApplicationService.sensitive_types`.
        allow_unauthenticated: ``--no-authentication``, which this refuses to honor.

    Raises:
        PolicyError: Authentication is off and the installation is sensitive, or the data
            directory's database could not be read to tell.
    """
    exposure = exposure_when_unauthenticated(settings, sensitive_types)
    if not exposure:
        return
    lines: list[str] = []
    if exposure.configured:
        lines.append(
            f"enabled connector instance(s) {', '.join(map(repr, exposure.configured))} read a "
            f"sensitive source (their type is registered as sensitive)"
        )
    if exposure.held:
        lines.append(
            f"the data directory still holds documents, in some state (soft-deleted and purged "
            f"rows count), from {', '.join(map(repr, exposure.held))}, which a sensitive "
            f"connector has synced — disabling or removing the connector does not lift this"
        )
    joined = "\n  - ".join(lines)
    waiver = (
        "\n--no-authentication was passed and does not apply here: it makes every anonymous "
        "caller an administrator, and a sensitive source is not something an operator can "
        "extend to whoever can reach a port."
        if allow_unauthenticated
        else ""
    )
    msg = (
        f"refusing to serve workspace {settings.workspace!r} without authentication, on any "
        f"address (loopback included — another local account or process can reach it):\n"
        f"  - {joined}{waiver}\n"
        f"Set security.auth.mode to 'api_key' or 'oauth'. To serve without authentication "
        f"again, delete the source's documents outright from an authenticated server with "
        f"`manicule document delete-source <name> --yes`"
        f"{' and disable or remove the connector' if exposure.configured else ''}."
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
    sensitive_types: AbstractSet[str],
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
    allow_unauthenticated: bool = False,
) -> Bind:
    """Decide where to listen, refusing anything wide that was not asked for three times.

    Args:
        settings: Configuration. ``security.transport`` supplies the defaults, and its own
            default host is loopback.
        sensitive_types: Connector types the registry marks sensitive. Required, with no
            default — see :func:`require_sensitive_source_authentication`.
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
            missing, or the installation is in team mode without authentication — see
            :func:`require_team_authentication` — or holds a sensitive source without it — see
            :func:`require_sensitive_source_authentication`. The message names which, and what
            to do instead.
    """
    # First, and before the loopback answer below: team mode's refusal holds on every address.
    require_team_authentication(settings, allow_unauthenticated=allow_unauthenticated)
    # Beside it, for the same reason: loopback does not help a mailbox.
    require_sensitive_source_authentication(
        settings, sensitive_types, allow_unauthenticated=allow_unauthenticated
    )
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
    "require_sensitive_source_authentication",
    "require_team_authentication",
    "resolve_bind",
    "stdio",
]
