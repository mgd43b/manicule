"""Whose address a request actually has, when something is in front of the server.

``X-Forwarded-For`` is a header. A client sets headers. So on a server that reads it without
qualification, **every IP-based decision becomes a value the caller chose** — the rate limit
buckets by whatever the caller typed, the audit trail records whatever the caller typed, and an
allowlist admits whoever claims to be on it. The failure is silent: the header looks like
infrastructure, the code reads like plumbing, and nothing about a running system says the
addresses in the log are fiction.

manicule therefore states the property positively and enforces it here:

**A forwarding header is believed only from a peer inside
``security.transport.trusted_proxies``, and that list is empty by default.** With no proxies
configured — the ordinary case, because the ordinary bind is loopback — the header is not read
at all. The socket's own peer address is the answer, and a socket peer is not something a
caller can set.

**The client is the right-most address that is not itself a trusted proxy.** Walking from the
right is the only direction that works: a caller can prepend as many fabricated hops as it
likes, and every one of them lands to the *left* of what the real proxy appended. Taking the
left-most entry — the obvious reading of "the original client" — is exactly the spoof.

**A malformed entry is not an address.** Anything that does not parse as an IP is skipped
rather than passed along as a string, because a value that reaches a log or an allowlist
without being an address is the same defect wearing a different hat.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from ipaddress import IPv4Network, IPv6Network

    from manicule.config.settings import Settings

FORWARDED_FOR = "x-forwarded-for"
"""The one forwarding header manicule reads, and only from a trusted peer.

Named as a constant because the *absence* of the others matters: ``Forwarded``,
``X-Real-IP``, ``CF-Connecting-IP`` and friends are not consulted at all. Each is another
header a client can set, and reading several means the effective policy is whichever one the
caller found first.
"""


def _normalise(host: str) -> IPv4Address | IPv6Address | None:
    """One host string as an address, or ``None`` when it is not one.

    An IPv4-mapped IPv6 address — what a dual-stack listener reports for an IPv4 client — is
    folded to its IPv4 form, so ``::ffff:127.0.0.1`` matches a ``127.0.0.0/8`` entry. Without
    that, an allowlist written the obvious way silently matches nothing on a dual-stack bind.
    """
    text = host.strip()
    if not text:
        return None
    # A port, as `host:port` for IPv4 or `[host]:port` for IPv6. Starlette hands back a bare
    # host, but a header written by hand or by another proxy may not.
    if text.startswith("["):
        closing = text.find("]")
        if closing != -1:
            text = text[1:closing]
    elif text.count(":") == 1:
        text = text.split(":", 1)[0]
    try:
        parsed = ip_address(text)
    except ValueError:
        return None
    if isinstance(parsed, IPv6Address) and parsed.ipv4_mapped is not None:
        return parsed.ipv4_mapped
    return parsed


def parse_networks(entries: Iterable[str]) -> tuple[IPv4Network | IPv6Network, ...]:
    """Turn configured CIDR strings into networks.

    Raises:
        ValueError: An entry is not a network. Refused rather than skipped: an allowlist with
            a typo silently dropped is an allowlist nobody can read correctly, and this one
            decides whose address gets believed.
    """
    networks: list[IPv4Network | IPv6Network] = []
    for entry in entries:
        text = entry.strip()
        if not text:
            msg = "an empty string is not a CIDR range; remove it from trusted_proxies"
            raise ValueError(msg)
        # `strict=False` so a bare host address is accepted and read as a single-host range:
        # `10.0.0.7` means `10.0.0.7/32`, which is what somebody writing one address means.
        networks.append(ip_network(text, strict=False))
    return tuple(networks)


@dataclass(frozen=True, slots=True)
class ProxyPolicy:
    """Which peers may speak for somebody else, and the resolution that follows from it."""

    trusted: tuple[IPv4Network | IPv6Network, ...] = ()

    @classmethod
    def of(cls, settings: Settings) -> ProxyPolicy:
        """The policy this installation is configured for.

        The strings are validated by :class:`~manicule.config.settings.TransportSettings`, so
        by the time they reach here they parse. Parsing again rather than caching a parsed
        form on the settings object keeps configuration a description and this a decision.
        """
        return cls(trusted=parse_networks(settings.security.transport.trusted_proxies))

    @property
    def enabled(self) -> bool:
        """Whether any peer at all may forward an address. False by default."""
        return bool(self.trusted)

    def trusts(self, host: str) -> bool:
        """Whether ``host`` is a peer whose forwarding header would be believed."""
        parsed = _normalise(host)
        if parsed is None:
            return False
        return any(parsed in network for network in self.trusted)

    def client_address(self, *, peer: str | None, forwarded_for: str | None = None) -> str:
        """The address to attribute this request to.

        Args:
            peer: The socket's own peer, as the server reports it. ``None`` for a transport
                with no peer, which is reported as an empty address rather than guessed at.
            forwarded_for: The raw ``X-Forwarded-For`` header, if one arrived. **Ignored
                entirely unless ``peer`` is a trusted proxy**, which is the whole point.

        Returns:
            An address, or ``""`` when there is nothing trustworthy to say. Empty is a real
            answer: an audit row that says "unknown" is honest, and one that says ``0.0.0.0``
            or repeats a caller-supplied string is not.
        """
        if peer is None:
            return ""
        if not self.enabled or not self.trusts(peer):
            # Not "fall back to the peer having tried the header" — the header is not read at
            # all. There is no code path here in which an untrusted peer's header influences
            # the result, which is a stronger statement than any amount of validation of it.
            return self._present(peer)
        if not forwarded_for:
            return self._present(peer)
        return self._forwarded(forwarded_for) or self._present(peer)

    def _forwarded(self, header: str) -> str:
        """The client, from right to left, skipping hops that are themselves trusted proxies.

        Right to left because a caller controls the left. Everything it fabricates is
        prepended; what the real proxy appended is at the right-hand end. Skipping trusted
        entries handles a chain of two or more proxies, each of which appended the address it
        saw.

        A chain consisting entirely of trusted proxies yields nothing, and the caller falls
        back to the peer — which is correct, because in that case no untrusted client is
        named anywhere in the header.
        """
        for entry in reversed(self._entries(header)):
            parsed = _normalise(entry)
            if parsed is None:
                # Not an address. Skipped rather than returned: a string that is not an
                # address must never reach a log line or an allowlist looking like one.
                continue
            if any(parsed in network for network in self.trusted):
                continue
            return str(parsed)
        return ""

    @staticmethod
    def _entries(header: str) -> Sequence[str]:
        return [part for part in (piece.strip() for piece in header.split(",")) if part]

    @staticmethod
    def _present(peer: str) -> str:
        """A socket peer, normalised. Unparseable peers are reported as unknown."""
        parsed = _normalise(peer)
        return str(parsed) if parsed is not None else ""


__all__ = ["FORWARDED_FOR", "ProxyPolicy", "parse_networks"]
