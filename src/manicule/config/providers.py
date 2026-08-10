"""Provider credentials, and whether a provider's endpoint is on this machine.

Two questions live here, and they are not the same question.

**Does this provider need a credential?** That is a property of the provider's *name*:
``openai`` needs a key wherever it is reached, ``ollama`` needs none. The convention is
``<PROVIDER>_API_KEY``, applied to *every* provider, including ones manicule has never heard
of, so adding a provider needs no code change here. :data:`PROVIDER_ALIASES` exists only for
providers whose community-standard variable does not match their name, or that have more than
one in circulation. Aliases are tried in order, and the conventional name is always tried
first.

Precedence, highest first:

1. An explicit value in configuration — including ``MANICULE_PROVIDERS__OPENAI__API_KEY``,
   which is configuration expressed as an environment variable.
2. The conventional variable, then any aliases.
3. Nothing. A missing key is not an error here: whether it matters depends on whether that
   provider is selected, which :func:`manicule.config.settings.Settings.policy_problems`
   decides once the whole configuration is known.

**Does content leave this machine?** That is a property of the *resolved endpoint*, and
:class:`Egress` is the answer. A provider name is a default, never evidence: ``ollama`` is
the local runtime by convention, but ``base_url`` exists precisely so that "local and hosted
models differ by this and nothing else", and an ``ollama`` pointed at ``gpu-box.lan`` sends
every prompt and every retrieved passage across the network. Deciding egress from the name
would let a local-only data policy report itself satisfied while exactly the thing it forbids
happens on every query, which is worse than having no policy at all.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr

PROVIDER_ALIASES: Mapping[str, tuple[str, ...]] = {
    "google": ("GEMINI_API_KEY",),
    "xai": ("GROK_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY",),
    "together": ("TOGETHERAI_API_KEY",),
}
"""Extra variables to try, after the conventional ``<PROVIDER>_API_KEY``."""

IN_PROCESS_PROVIDERS: frozenset[str] = frozenset({"mlx", "onnx", "local"})
"""Backends that run inside this process.

There is no endpoint to check and no socket to open: the model is loaded into this
interpreter and the text never reaches a network stack. These are the only providers whose
name is sufficient evidence about egress, and the set is closed and manicule's own — adding
to it is a code change, reviewed as one.
"""

SELF_HOSTED_PROVIDERS: frozenset[str] = frozenset({"ollama"})
"""Backends reached over HTTP that are *conventionally* run on this machine.

Conventionally, not necessarily. Membership here settles that no credential is needed and
that an unset ``base_url`` should be read as loopback; it settles nothing about where the
configured endpoint actually is. :func:`egress_for` decides that.
"""

KEYLESS_PROVIDERS: frozenset[str] = IN_PROCESS_PROVIDERS | SELF_HOSTED_PROVIDERS
"""Providers that need no credential.

Deliberately *not* a statement about egress. The two properties were one set until an
``ollama`` on another host was found to satisfy a local-only policy while every prompt
crossed the network.
"""

DEFAULT_BASE_URLS: Mapping[str, str] = {
    "ollama": "http://localhost:11434",
}
"""Base URLs manicule fills in.

Only for endpoints that are conventionally fixed and locally hosted. Hosted providers are
reached through the generation library's own defaults, so that a URL manicule pins today
cannot become a URL that is wrong tomorrow.

Filled in by :func:`resolve_provider_keys`, which means the common case reaches
:func:`egress_for` as an explicit loopback URL rather than as a name to be trusted.
"""

LOOPBACK_HOST_NAMES: frozenset[str] = frozenset({"localhost"})
"""Host *names* that are loopback by specification rather than by resolution.

``localhost`` and anything under ``.localhost`` (RFC 6761 §6.3: resolvers must map them to a
loopback address). No other name qualifies — see :func:`egress_for` for why nothing here
performs a DNS lookup.
"""

LOOPBACK_HOST_SUFFIX = ".localhost"

LOCAL_SCHEMES: frozenset[str] = frozenset({"unix", "unix+http", "http+unix", "file"})
"""URL schemes that address this machine's filesystem rather than a network peer.

A unix domain socket cannot leave the host: it has no address family that reaches one.
"""


class Egress(StrEnum):
    """Where the content sent to an endpoint goes.

    ``docs/generation.md`` §7.1 names two classes, local and remote. This splits local in
    two, because "it never reaches a socket" and "it reaches a socket that loops back" are
    different claims and the same section requires the classification to be recorded on every
    answer and in the trace: the residual limit it states — a loopback proxy forwarding to a
    hosted provider — applies to one of them and not the other, and a record that cannot tell
    them apart cannot support the audit it exists for. :attr:`leaves_machine` is the boolean
    where only the policy decision matters.
    """

    IN_PROCESS = "in_process"
    """No endpoint at all. The model runs in this interpreter."""

    LOOPBACK = "loopback"
    """A socket on this machine.

    **The residual limit, stated rather than hidden.** A proxy listening on loopback may
    forward to a hosted provider, and nothing observable at configuration time distinguishes
    that from a local runtime. This class means "manicule is not the component that sent it
    off this machine", not "it did not leave".
    """

    REMOTE = "remote"
    """Another machine. Content leaves this one."""

    @property
    def leaves_machine(self) -> bool:
        """Whether content sent here crosses the network, as far as this can be known."""
        return self is Egress.REMOTE


class ModelRole(StrEnum):
    """Which model an endpoint serves.

    The same provider can appear in both roles at two different endpoints — an ``ollama``
    embedder on loopback and an ``ollama`` generator on another host is a real configuration —
    so egress is recorded per role, not per provider name.
    """

    LLM = "llm"
    EMBEDDING = "embedding"


def env_var_names(provider: str) -> tuple[str, ...]:
    """Environment variables consulted for ``provider``'s key, in order."""
    conventional = provider.strip().upper().replace("-", "_").replace(".", "_") + "_API_KEY"
    return (conventional, *PROVIDER_ALIASES.get(provider.strip().lower(), ()))


def runs_in_process(provider: str) -> bool:
    """Whether ``provider`` loads its model into this interpreter.

    The one thing a provider's name settles about egress on its own, because a backend with
    no endpoint has nowhere to send anything.
    """
    return provider.strip().lower() in IN_PROCESS_PROVIDERS


def needs_credential(provider: str) -> bool:
    """Whether ``provider`` has to be given an API key to be usable.

    A property of the name, not of the endpoint: an ``ollama`` on another host still needs no
    key, and an ``openai`` on loopback still does.
    """
    return provider.strip().lower() not in KEYLESS_PROVIDERS


def egress_for(provider: str, base_url: str | None = None) -> Egress:
    """Where content sent to ``provider`` at ``base_url`` goes.

    The endpoint decides. The name is consulted only when there is no endpoint to consult:

    - An in-process provider is :attr:`Egress.IN_PROCESS` whatever ``base_url`` says, because
      nothing dials it. A ``base_url`` on one is a setting that is not in force, and
      :meth:`~manicule.config.settings.Settings.policy_problems` says so rather than leaving
      it to look effective.
    - With no ``base_url``, the name supplies the default: a self-hosted provider is
      :attr:`Egress.LOOPBACK` (which is what :data:`DEFAULT_BASE_URLS` would have filled in),
      and anything else is :attr:`Egress.REMOTE`.
    - Otherwise the URL is classified, and a host that is not demonstrably this machine is
      :attr:`Egress.REMOTE`.

    **What counts as this machine.** A loopback literal (``127.0.0.0/8``, ``::1``, and
    ``::ffff:127.0.0.1``, which is the same address written for a dual-stack socket); a unix
    socket or file scheme; ``localhost`` and anything under ``.localhost``, which RFC 6761
    reserves for loopback. Everything else is remote, including two cases that look local and
    are not: ``0.0.0.0`` and ``::`` are addresses you *bind*, and what connecting to one
    reaches is a property of the platform's stack rather than of this configuration; and a
    link-local address (``169.254.0.0/16``, ``fe80::/10``) is another machine on the same
    link — ``169.254.169.254`` most famously of all.

    **No name is resolved, deliberately.** Three reasons, and any one of them is sufficient.
    A resolution is not stable: it can differ between this process and the next request, so a
    name that resolved to loopback at startup constrains nothing about where the bytes
    actually went. A lookup makes a security classification depend on DNS being up and
    answering the same way twice. And a resolver that returns a loopback address for a name
    somebody else controls would launder a remote endpoint into a local classification, which
    is the exact failure this function exists to prevent. A hostname alias for this machine is
    therefore refused: write the address.

    Args:
        provider: Provider name. Case and surrounding whitespace do not matter.
        base_url: The endpoint as configured, or ``None`` if none was.

    Returns:
        The egress class. Unparseable or scheme-less values are :attr:`Egress.REMOTE`: this
        is not the place to guess, and the refusal names the URL it could not vouch for.
    """
    name = provider.strip().lower()
    if name in IN_PROCESS_PROVIDERS:
        return Egress.IN_PROCESS
    endpoint = (base_url or "").strip()
    if not endpoint:
        return Egress.LOOPBACK if name in SELF_HOSTED_PROVIDERS else Egress.REMOTE
    return endpoint_egress(endpoint)


def endpoint_egress(base_url: str) -> Egress:
    """Where a URL points: :attr:`Egress.LOOPBACK` for this machine, else :attr:`Egress.REMOTE`.

    Never :attr:`Egress.IN_PROCESS` — a URL is a socket by construction. See
    :func:`egress_for` for the classification rules and for why no name is resolved.
    """
    try:
        parsed = urlsplit(base_url.strip())
        host = parsed.hostname
    except ValueError:
        # A malformed URL — an unbracketed IPv6 literal, say. Not something to guess about.
        return Egress.REMOTE
    if parsed.scheme.lower() in LOCAL_SCHEMES:
        return Egress.LOOPBACK
    if not host:
        return Egress.REMOTE
    return _host_egress(host)


def _host_egress(host: str) -> Egress:
    """Classify one host, which is either an IP literal or a name."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        name = host.rstrip(".").lower()
        is_reserved = name in LOOPBACK_HOST_NAMES or name.endswith(LOOPBACK_HOST_SUFFIX)
        return Egress.LOOPBACK if is_reserved else Egress.REMOTE
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return Egress.LOOPBACK if address.is_loopback else Egress.REMOTE


class Endpoint(BaseModel):
    """One model endpoint a configuration will open, and where its content goes.

    Recorded per role rather than per provider because the same provider name can appear
    twice at two different addresses, and because #7 puts this classification on every answer
    and in every trace: an answer that cannot say whether its context left the machine is not
    auditable afterwards.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ModelRole
    provider: str
    """As configured, lowercased. Not constrained here: this is a record of what a
    configuration resolved to, and a blank provider is a problem for
    :meth:`~manicule.config.settings.Settings.policy_problems` to *report* rather than one
    for this type to raise in the middle of the report."""

    base_url: str | None = None
    egress: Egress

    @property
    def leaves_machine(self) -> bool:
        return self.egress.leaves_machine

    def describe(self) -> str:
        """A phrase naming this endpoint, for a message somebody has to act on."""
        where = f" at {self.base_url}" if self.base_url else ""
        return f"{self.role.value} provider {self.provider!r}{where}"


class ProviderSettings(BaseModel):
    """Credentials and endpoint for one model provider.

    One shape for every provider. There is no per-vendor settings type, because a per-vendor
    type in configuration becomes a per-vendor branch in every consumer.
    """

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = Field(
        default=None,
        description="Resolved from the environment when not set here. Held as a secret so "
        "that printing configuration cannot leak it.",
    )
    base_url: str | None = Field(
        default=None,
        description="Endpoint override. Local and hosted models differ by this and nothing "
        "else — which is why it, and not the provider's name, decides whether content leaves "
        "this machine.",
    )
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Additional provider parameters passed through untouched, e.g. an "
        "organisation id or an API version.",
    )

    @property
    def has_key(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())


def resolve_provider_keys(
    providers: Mapping[str, ProviderSettings],
    required: frozenset[str] = frozenset(),
    environ: Mapping[str, str] | None = None,
) -> dict[str, ProviderSettings]:
    """Fill in missing credentials and base URLs from the environment.

    Args:
        providers: Provider settings as configuration gave them.
        required: Providers that configuration selects. These get an entry even if
            configuration never mentioned them, so that ``LLM_PROVIDER=openai`` plus
            ``OPENAI_API_KEY`` in the environment is a complete setup with no config file.
        environ: Environment to read. Defaults to the real one.

    Returns:
        A new mapping. Entries that already had a key keep it — an explicit value is never
        overwritten by a conventional one.
    """
    env = os.environ if environ is None else environ
    resolved: dict[str, ProviderSettings] = {}

    for name in sorted(set(providers) | {p.strip().lower() for p in required if p.strip()}):
        settings = providers.get(name, ProviderSettings())
        api_key = settings.api_key
        if api_key is None and needs_credential(name):
            for var in env_var_names(name):
                value = env.get(var)
                if value:
                    api_key = SecretStr(value)
                    break
        base_url = settings.base_url or DEFAULT_BASE_URLS.get(name)
        resolved[name] = settings.model_copy(update={"api_key": api_key, "base_url": base_url})

    return resolved


__all__ = [
    "DEFAULT_BASE_URLS",
    "IN_PROCESS_PROVIDERS",
    "KEYLESS_PROVIDERS",
    "LOCAL_SCHEMES",
    "LOOPBACK_HOST_NAMES",
    "LOOPBACK_HOST_SUFFIX",
    "PROVIDER_ALIASES",
    "SELF_HOSTED_PROVIDERS",
    "Egress",
    "Endpoint",
    "ModelRole",
    "ProviderSettings",
    "egress_for",
    "endpoint_egress",
    "env_var_names",
    "needs_credential",
    "resolve_provider_keys",
    "runs_in_process",
]
