"""A forwarded address is believed from an allowlisted peer, and from nobody else.

This is the suite that decides whether every IP-based decision in the system is a fact or a
client-controlled value. The failure it defends against is silent: a server that trusts
``X-Forwarded-For`` unconditionally looks exactly like one that does not, right up until
somebody sets the header.

So the tests come in pairs. For each property there is a **negative** — the header is ignored
— and a **positive** — the header is honored — because a resolver that ignored the header
always would satisfy every negative test on its own, and would also be useless.

The end-to-end cases drive the whole application, through the real middleware, and read the
address off the resolved principal. A unit test of :class:`~manicule.api.proxy.ProxyPolicy`
would not notice a middleware that never called it.
"""

from __future__ import annotations

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from manicule.api.proxy import ProxyPolicy, parse_networks
from manicule.api.security import Principal
from manicule.config.settings import Settings
from tests.api.support import app_for, backend_with_a_document, client_for
from tests.app.fakes import FakeBackend

SPOOFED = "203.0.113.9"
"""What a caller claims to be. TEST-NET-3, so it is not anybody's real address."""

REAL_CLIENT = "198.51.100.4"
"""What a genuine proxy would have appended. TEST-NET-2."""

PROXY = "10.0.0.7"
"""The address the proxy itself connects from."""

TRUSTED = ("10.0.0.0/8",)


def _policy(*ranges: str) -> ProxyPolicy:
    return ProxyPolicy(trusted=parse_networks(ranges))


# --- the resolver -----------------------------------------------------------------------------


def test_with_no_trusted_proxies_the_header_is_not_read_at_all() -> None:
    """The default. An empty allowlist means the socket peer is the answer, full stop.

    Not "the header is validated and rejected" — it is never consulted. That is a stronger
    property than any amount of validation, because there is no code path in which an
    untrusted peer's header can influence the result.
    """
    policy = ProxyPolicy()
    assert not policy.enabled
    assert policy.client_address(peer=PROXY, forwarded_for=SPOOFED) == PROXY


def test_a_spoofed_header_from_a_peer_outside_the_allowlist_is_ignored() -> None:
    """The attack this whole module exists for.

    A caller connecting directly, from an address nobody trusts, sets the header to whatever
    it likes. The answer is the address it actually connected from.
    """
    policy = _policy(*TRUSTED)
    outsider = "192.0.2.50"
    assert policy.client_address(peer=outsider, forwarded_for=SPOOFED) == outsider


def test_a_forwarded_address_from_an_allowlisted_peer_is_honored() -> None:
    """The positive control.

    Without it, a resolver that returned the peer unconditionally would pass every test above
    — and a trusted-proxy implementation that never trusts a proxy is not one.
    """
    policy = _policy(*TRUSTED)
    assert policy.client_address(peer=PROXY, forwarded_for=REAL_CLIENT) == REAL_CLIENT


def test_the_client_is_the_rightmost_entry_that_is_not_itself_a_proxy() -> None:
    """Prepended hops are ignored, which is the only reading that is not spoofable.

    A caller controls the left-hand end of this header: whatever it sends arrives before what
    the real proxy appends. Taking the left-most entry — the obvious reading of "the original
    client" — is exactly the spoof.
    """
    policy = _policy(*TRUSTED)
    header = f"{SPOOFED}, {REAL_CLIENT}"
    assert policy.client_address(peer=PROXY, forwarded_for=header) == REAL_CLIENT


def test_a_chain_of_proxies_is_walked_past_to_the_first_untrusted_address() -> None:
    """Two proxies, each of which appended what it saw. The client is still the client."""
    policy = _policy("10.0.0.0/8", "172.16.0.0/12")
    header = f"{REAL_CLIENT}, 172.16.4.4, 10.0.0.8"
    assert policy.client_address(peer=PROXY, forwarded_for=header) == REAL_CLIENT


def test_a_chain_entirely_of_trusted_proxies_falls_back_to_the_peer() -> None:
    """Nothing in the header names an untrusted client, so there is nothing to believe."""
    policy = _policy(*TRUSTED)
    assert policy.client_address(peer=PROXY, forwarded_for="10.0.0.8, 10.0.0.9") == PROXY


def test_an_entry_that_is_not_an_address_is_skipped_rather_than_passed_along() -> None:
    """A string that is not an address must never reach a log line looking like one."""
    policy = _policy(*TRUSTED)
    header = f"not-an-address, {REAL_CLIENT}, <script>"
    assert policy.client_address(peer=PROXY, forwarded_for=header) == REAL_CLIENT


def test_a_header_of_nothing_but_rubbish_falls_back_to_the_peer() -> None:
    """The peer, not an empty string and not the rubbish."""
    policy = _policy(*TRUSTED)
    assert policy.client_address(peer=PROXY, forwarded_for="nonsense, ../../etc") == PROXY


def test_an_ipv4_mapped_peer_matches_an_ipv4_allowlist() -> None:
    """What a dual-stack listener reports for an IPv4 client.

    Without the folding, an allowlist written the obvious way silently matches nothing on a
    dual-stack bind — the proxy stops being trusted and every request is attributed to it.
    """
    policy = _policy(*TRUSTED)
    assert policy.trusts("::ffff:10.0.0.7")
    assert policy.client_address(peer="::ffff:10.0.0.7", forwarded_for=REAL_CLIENT) == REAL_CLIENT


def test_a_peer_with_a_port_is_still_matched() -> None:
    """Another proxy in the chain may write ``host:port``. It is still that host."""
    policy = _policy(*TRUSTED)
    assert policy.client_address(peer=PROXY, forwarded_for=f"{REAL_CLIENT}:51234") == REAL_CLIENT


def test_a_request_with_no_peer_reports_no_address() -> None:
    """Empty is a real answer. An audit row saying "unknown" is honest; a fabricated one is not."""
    assert _policy(*TRUSTED).client_address(peer=None, forwarded_for=REAL_CLIENT) == ""


# --- configuration ----------------------------------------------------------------------------


def test_a_trusted_proxy_entry_that_is_not_a_network_is_refused_at_startup() -> None:
    """A typo here fails closed and silently: the proxy is never trusted and every request is
    attributed to it, while the operator believes a policy is in force. So it is refused."""
    with pytest.raises(ValidationError, match="not an address or a CIDR range"):
        Settings(security={"transport": {"trusted_proxies": ["10.0.0.0/33"]}})  # pyright: ignore[reportArgumentType]


def test_a_bare_address_is_accepted_and_means_that_one_host() -> None:
    """The positive control for the validator: a valid list still validates."""
    settings = Settings(security={"transport": {"trusted_proxies": ["10.0.0.7"]}})  # pyright: ignore[reportArgumentType]
    policy = ProxyPolicy.of(settings)
    assert policy.trusts("10.0.0.7")
    assert not policy.trusts("10.0.0.8")


# --- end to end, through the real middleware --------------------------------------------------


def _address_seen_by_the_app(backend: FakeBackend, *, peer: str, header: str | None) -> str:
    """Drive a request through the whole application and read the principal it resolved.

    Through the application rather than the resolver, because the property under test is not
    "the resolver is correct" — it is "the address every route sees came from the resolver".
    A middleware that never called it would pass every unit test above.

    The probe route is mounted on a real application and reads ``request.state.principal``,
    which is the same object every production route's dependency reads.
    """
    seen: list[Principal] = []
    app = app_for(backend)

    async def probe(request: Request) -> dict[str, str]:
        principal: Principal = request.state.principal
        seen.append(principal)
        return {"address": principal.address}

    app.get("/_probe")(probe)
    headers = {"X-Forwarded-For": header} if header else {}
    with TestClient(app, client=(peer, 41234)) as client:
        assert client.get("/_probe", headers=headers).status_code == 200
    assert seen, "the middleware never resolved a principal"
    return seen[0].address


def test_the_running_application_ignores_a_spoofed_header_from_an_untrusted_peer() -> None:
    """End to end: the header is set, the peer is not on the list, the address is the peer."""
    backend, _ = backend_with_a_document(security={"transport": {"trusted_proxies": list(TRUSTED)}})
    assert _address_seen_by_the_app(backend, peer="192.0.2.50", header=SPOOFED) == "192.0.2.50"


def test_the_running_application_honors_a_forwarded_header_from_a_trusted_peer() -> None:
    """End to end, positive: the same request from an allowlisted peer is believed."""
    backend, _ = backend_with_a_document(security={"transport": {"trusted_proxies": list(TRUSTED)}})
    assert _address_seen_by_the_app(backend, peer=PROXY, header=REAL_CLIENT) == REAL_CLIENT


def test_the_running_application_reads_no_header_when_no_proxies_are_configured() -> None:
    """The default configuration. The header is present and is not consulted."""
    backend, _ = backend_with_a_document()
    assert _address_seen_by_the_app(backend, peer=PROXY, header=SPOOFED) == PROXY


def test_the_default_configuration_trusts_no_proxy() -> None:
    """Stated on its own, because everything above depends on it.

    A default that trusted anything would make the whole allowlist an opt-*out*.
    """
    assert Settings().security.transport.trusted_proxies == ()
    assert not ProxyPolicy.of(Settings()).enabled


def test_a_client_using_the_test_helper_arrives_from_a_real_address() -> None:
    """The suite's own scaffolding, asserted.

    Starlette's default peer is the string ``testclient``, which is not an address — so a
    trusted-proxy suite that forgot to set one would be testing the unparseable-peer case
    while appearing to test the allowlist.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend, peer="192.0.2.7") as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert _address_seen_by_the_app(backend, peer="192.0.2.7", header=None) == "192.0.2.7"
