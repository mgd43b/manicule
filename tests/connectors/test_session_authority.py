"""Which instances count as one instance, and which must never be confused for each other.

A session is filed under the authority it was captured from, so this one function decides two
opposite things at once. Get it too strict and somebody with two connectors pointed at one wiki
is asked to sign in twice for the same server, once per spelling. Get it too loose and a live
corporate session captured for one site is handed to a connector configured for another.

**The two errors are not equally bad, and the tests are weighted accordingly.** A split costs a
sign-in. A merge is a credential leak. So the isolation section below is the longer one, it
includes the shapes an attacker would choose, and every rule in the sharing section is one where
the protocol itself — not this module's convenience — says the two spellings name one server.

Fixtures are synthetic throughout: `https://wiki.example.test` and invented context paths.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from manicule.connectors.credentials import BrowserSession
from manicule.connectors.sessions import SessionVault, authority_key

SITE = "https://wiki.example.test"
CAPTURED_AT = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
SENTINEL = "not-a-real-cookie-value"


def a_session(base_url: str, *, account: str = "sync.user") -> BrowserSession:
    return BrowserSession(
        base_url=base_url,
        account=account,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr(SENTINEL)},
    )


# --- one authority, however it was spelled ------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "equivalent"),
    [
        (SITE, f"{SITE}/"),
        (SITE, "https://WIKI.example.test"),
        (SITE, "HTTPS://wiki.example.test"),
        (SITE, "https://wiki.example.test:443"),
        (f"{SITE}/confluence", "https://wiki.example.test:443/confluence/"),
        ("http://wiki.example.test", "http://wiki.example.test:80"),
        (f"{SITE}/confluence", f"{SITE}/confluence//"),
    ],
    ids=[
        "trailing-slash",
        "host-case",
        "scheme-case",
        "explicit-default-port",
        "port-and-slash-together",
        "explicit-default-port-http",
        "repeated-trailing-slash",
    ],
)
def test_two_spellings_of_one_server_are_one_entry(configured: str, equivalent: str) -> None:
    """Each of these is a case where the *instance* cannot tell the two apart.

    DNS is case-insensitive, a URL scheme is case-insensitive, and a scheme's default port is
    the port it already meant. Somebody who wrote one connector with the port and one without
    has configured one wiki, and being asked to sign in twice for it is the bug.
    """
    assert authority_key(configured) == authority_key(equivalent)


def test_one_sign_in_satisfies_every_connector_on_that_authority() -> None:
    """The property the key exists for, asserted through the vault rather than the function.

    Two connectors, two spellings, one instance: a session captured through either is found by
    a lookup for the other. Asserted on the store because that is where the sharing actually
    has to happen — a key that normalized correctly and a vault that did not consult it would
    pass the test above and fail this one.
    """
    vault = SessionVault()
    asyncio.run(vault.save(a_session("https://WIKI.example.test:443/confluence")))

    assert vault.load(f"{SITE}/confluence") is not None
    assert vault.load(f"{SITE}/confluence/") is not None


def test_forgetting_through_one_spelling_forgets_the_session_itself() -> None:
    """Otherwise `--forget` is a command that reports success and leaves the credential held."""
    vault = SessionVault()
    asyncio.run(vault.save(a_session(f"{SITE}/confluence")))

    assert asyncio.run(vault.forget("https://WIKI.example.test:443/confluence/")) is True
    assert vault.load(f"{SITE}/confluence") is None


# --- and never one authority that is really two -------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "other"),
    [
        (SITE, "http://wiki.example.test"),
        (f"{SITE}/confluence", f"{SITE}/Confluence"),
        (f"{SITE}/confluence", f"{SITE}/jira"),
        (f"{SITE}/confluence", f"{SITE}/confluence-archive"),
        (f"{SITE}/confluence", SITE),
        (SITE, "https://wiki.example.test:8443"),
        (SITE, "https://notwiki.example.test"),
        (SITE, "https://wiki.example.test.evil.test"),
        (SITE, "https://evil.test/wiki.example.test"),
        (SITE, "https://wiki.example.test@evil.test"),
    ],
    ids=[
        "tls-policy",
        "context-path-case",
        "different-application",
        "prefix-not-a-boundary",
        "root-versus-context-path",
        "non-default-port",
        "suffix-lookalike-host",
        "subdomain-of-attacker",
        "host-in-the-path",
        "host-in-the-userinfo",
    ],
)
def test_two_servers_never_become_one_entry(configured: str, other: str) -> None:
    """The direction that costs a credential rather than a sign-in.

    `tls-policy` is here because a session captured over plaintext is not evidence about the
    TLS-protected instance and the cookie's own `secure` flag depends on it.
    `context-path-case` is here because a Confluence context path is case-sensitive on the
    server, so `/Confluence` may be a different application or nothing at all.
    `host-in-the-userinfo` is the shape that reads as the configured site to a person skimming
    a config file and resolves to the attacker's host in every client that parses it.
    """
    assert authority_key(configured) != authority_key(other)


def test_a_session_is_never_offered_to_another_instance() -> None:
    """The isolation above, asserted through the store for the same reason as the sharing."""
    vault = SessionVault()
    asyncio.run(vault.save(a_session(f"{SITE}/confluence")))

    assert vault.load(f"{SITE}/confluence") is not None
    assert vault.load(f"{SITE}/jira") is None
    assert vault.load("http://wiki.example.test/confluence") is None
    assert vault.load("https://wiki.example.test.evil.test/confluence") is None


def test_two_authorities_are_two_held_sessions_rather_than_one_overwritten() -> None:
    """A vault that merged them would report a session for a site nobody signed in to."""
    vault = SessionVault()
    asyncio.run(vault.save(a_session(f"{SITE}/confluence", account="wiki.user")))
    asyncio.run(vault.save(a_session(f"{SITE}/jira", account="jira.user")))

    assert len(vault) == 2
    held = vault.holding()
    assert held[authority_key(f"{SITE}/confluence")] == "wiki.user"
    assert held[authority_key(f"{SITE}/jira")] == "jira.user"


# --- what must not end up in a key --------------------------------------------------------------


def test_credentials_in_a_configured_url_never_reach_the_key() -> None:
    """`holding()` keys are printed by `doctor --json` and rendered by the Web status panel.

    A `base_url` carrying userinfo is a misconfiguration rather than an attack, but the cost of
    passing it through is that a password appears in a diagnostic somebody pastes into an issue.
    Dropping it is also what `origin_of` already does, so the key and the link check agree.
    """
    key = authority_key("https://sync.user:hunter2@wiki.example.test/confluence")

    assert "hunter2" not in key
    assert "sync.user" not in key
    assert key == authority_key(f"{SITE}/confluence")


def test_a_url_that_will_not_parse_keeps_its_own_entry() -> None:
    """Every unparseable value collapsing to one key would file them under one session.

    Nothing should configure these — `ConfluenceConfig` refuses a `base_url` that is not
    absolute http(s) — so this is about the store staying sound for a value that reached it by
    some other route, rather than about supporting them.
    """
    assert authority_key("not a url") != authority_key("also not a url")
    assert authority_key("") == ""


def test_surrounding_whitespace_is_not_a_second_instance() -> None:
    """A trailing newline out of a config file or a shell is not a different wiki."""
    assert authority_key(f"  {SITE}/confluence \n") == authority_key(f"{SITE}/confluence")
