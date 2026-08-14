"""Replacing a stored session must not be able to lose the one that already worked.

``capture_cookies`` verifies a candidate before it calls the store, which protects an existing
credential from a timeout, a closed browser, a dead cookie and a state file for the wrong site.
That is verify-before-write, and it is necessary. It says nothing about a *write* that fails
part-way, and the store used to delete the old record before writing the replacement — so a
``security`` invocation that failed on the fourth of twenty-three chunks left the operator with
no credential at all, having started with a working one.

Every case here runs against :class:`~tests.connectors.keychain_fake.FakeKeychain` rather than
the real command, for the one reason a fake is the right tool: there is no way to ask
``/usr/bin/security`` to fail on the fourth write. The real-Keychain cases live in
``test_browser_sso.py`` and answer the different question of whether a cookie survives the
Keychain's own encoding.

**A new reader is spelled as a new store.** Each assertion builds a fresh
:class:`~manicule.connectors.sessions.KeychainStore` over the same fake state, because a
guarantee that holds only inside the object that made the write is not the guarantee an operator
needs — theirs is a second process, tomorrow.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from manicule.connectors.credentials import BrowserSession
from manicule.connectors.sessions import CHUNK_BYTES, KeychainStore
from manicule.core.errors import ConfigError
from tests.connectors.keychain_fake import FakeKeychain, Fault, SimulatedTermination
from tests.connectors.support import CAPTURED_AT, SESSION_ACCOUNT

SITE = "https://wiki.example.test/confluence"
OTHER_SITE = "https://intranet.example.test"
SERVICE = "manicule test: confluence session"


def a_session(
    marker: str, *, base_url: str = SITE, padding: int = 0, account: str = SESSION_ACCOUNT
) -> BrowserSession:
    """A synthetic session whose cookie names which one it is.

    ``padding`` grows the record past one chunk and then past many, which is the only way to
    have a first, a middle and a last chunk to fail on.
    """
    cookies = {"JSESSIONID": SecretStr(marker)}
    if padding:
        cookies["MSISAuth"] = SecretStr(marker[0] * padding)
    return BrowserSession(
        base_url=base_url, account=account, captured_at=CAPTURED_AT, cookies=cookies
    )


def marker_of(session: BrowserSession | None) -> str | None:
    """Which synthetic session this is, by the cookie that names it."""
    return None if session is None else session.cookies["JSESSIONID"].get_secret_value()


def chunks_for(session: BrowserSession) -> int:
    """How many keychain items this session's record occupies."""
    import base64  # noqa: PLC0415 - one call, and the store's own encoding is the subject

    payload = base64.b64encode(session.to_json().encode()).decode()
    return -(-len(payload) // CHUNK_BYTES)


@pytest.fixture
def keychain() -> FakeKeychain:
    return FakeKeychain()


def store() -> KeychainStore:
    """A store with no memory of any earlier one. This is the 'new process' in every case."""
    return KeychainStore(SERVICE)


# --- the defect ---------------------------------------------------------------------------------


def test_a_failure_part_way_through_a_replacement_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The reproduction. Save a session, break the replacement, and read as a new process.

    The old store deleted every stored chunk before writing the first replacement chunk, so by
    the time anything could fail the credential was already gone. Nothing about the failure
    announced that: the message said the replacement had not been stored, which was true, and
    the operator discovered the rest on the next sync.
    """
    with keychain.installed():
        store().save(a_session("WORKING", padding=600))
        keychain.fail_on_chunk_write(2)
        with pytest.raises(ConfigError):
            store().save(a_session("REPLACEMENT", padding=600))
        keychain.assert_fired()

        assert marker_of(store().load(SITE)) == "WORKING"


def test_a_failure_on_the_very_first_chunk_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The same defect in its plainest form: nothing new was written, and the old one is gone.

    Kept separate from the middle-chunk case because the two used to fail differently and only
    one of them looked like data loss. Failing on a later chunk left a *decodable-looking*
    prefix, so ``load`` raised "not a session manicule wrote"; failing on the first left
    nothing, so ``load`` returned ``None`` and the next run reported no session was stored.
    """
    with keychain.installed():
        store().save(a_session("WORKING", padding=600))
        keychain.fail_on_chunk_write(1)
        with pytest.raises(ConfigError):
            store().save(a_session("REPLACEMENT", padding=600))
        keychain.assert_fired()

        assert marker_of(store().load(SITE)) == "WORKING"


def test_a_process_killed_part_way_through_a_replacement_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The window no exception handler can close, which is why the store's shape has to close it.

    A failed command at least raises something a caller could react to. A terminated process
    does not, and the state it leaves is whatever the last completed write left. This is the
    case that rules out "delete, write, and undo the delete on error" as a repair.
    """
    with keychain.installed():
        store().save(a_session("WORKING", padding=600))
        keychain.fail_on_chunk_write(2, fault=Fault.CRASH_AFTER_WRITE)
        with pytest.raises(SimulatedTermination):
            store().save(a_session("REPLACEMENT", padding=600))
        keychain.assert_fired()

        assert marker_of(store().load(SITE)) == "WORKING"
