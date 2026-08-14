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

import base64
import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from manicule.connectors import sessions
from manicule.connectors.credentials import BrowserSession
from manicule.connectors.sessions import (
    CHUNK_BYTES,
    DIGEST_HEX,
    GENERATION_HEX,
    MAX_STAGED,
    KeychainStore,
)
from manicule.core.errors import ConfigError
from tests.connectors.keychain_fake import (
    POINTER_SLOTS,
    FakeKeychain,
    Fault,
    SimulatedTermination,
    chunk_account,
    generation_account,
    journal_account,
    pointer_account,
)
from tests.connectors.support import CAPTURED_AT, SESSION_ACCOUNT

SITE = "https://wiki.example.test/confluence"
OTHER_SITE = "https://intranet.example.test"
SERVICE = "manicule test: confluence session"

# The account-naming helpers come from the fake rather than from the store, so that these cases
# assert against the layout as *observed* rather than agreeing with the code under test by
# construction. `test_the_records_a_save_leaves_behind_are_the_documented_ones` is what keeps the
# two in step, and it spells the account names out in full so a reviewer can see the format
# without reading either side.


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


def payload_of(session: BrowserSession) -> str:
    """The record as the store encodes it, which is what gets split across items."""
    return base64.b64encode(session.to_json().encode()).decode()


def chunks_for(session: BrowserSession) -> int:
    """How many keychain items this session's record occupies."""
    return -(-len(payload_of(session)) // CHUNK_BYTES)


def seed_legacy(keychain: FakeKeychain, session: BrowserSession) -> None:
    """Put a record where a version of manicule that predates generations would have put it.

    Written straight into the fake rather than through the store, because the store no longer
    has code that writes this shape — which is the point. What has to keep working is *reading*
    what an installed copy wrote last month.
    """
    payload = payload_of(session)
    site = session.base_url
    for index in range(0, len(payload), CHUNK_BYTES):
        keychain.items[(SERVICE, chunk_account(site, index // CHUNK_BYTES))] = payload[
            index : index + CHUNK_BYTES
        ]


@pytest.fixture(autouse=True)
def keychain() -> Iterator[FakeKeychain]:
    """A fake keychain, installed for the whole of every test in this file.

    Autouse and installed by the fixture rather than by a ``with`` block in each case, so that
    there is no line in this file from which a store could reach ``/usr/bin/security``. An
    assertion that slipped outside such a block would run against the developer's own Keychain
    — which is somebody's real credential store, and on macOS it prompts.
    """
    fake = FakeKeychain()
    with fake.installed():
        yield fake


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
    store().save(a_session("WORKING", padding=600))
    keychain.fail_on_chunk_write(2, fault=Fault.CRASH_AFTER_WRITE)
    with pytest.raises(SimulatedTermination):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_failure_on_the_last_staged_chunk_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The most nearly-successful failure there is: everything written but the final piece.

    Worth its own case because it is the one an implementation is most likely to get wrong by
    committing a moment too early — every chunk but one is there, the record looks the right
    shape, and only the length says otherwise.
    """
    replacement = a_session("REPLACEMENT", padding=600)
    store().save(a_session("WORKING", padding=600))
    keychain.fail_on_chunk_write(chunks_for(replacement))
    with pytest.raises(ConfigError):
        store().save(replacement)
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_replacement_that_reads_back_short_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The silent failure: every write reports success and one of them stored a prefix.

    This is the shape a macOS release with a smaller stdin buffer would take. Nothing raises on
    the way in, so only reading the staged generation back and comparing it catches this — and
    it has to be caught *before* the commit, which is what this asserts by looking at what a
    later reader gets.
    """
    store().save(a_session("WORKING", padding=600))
    keychain.truncate_chunk_write(3, keep=40)
    with pytest.raises(ConfigError, match="nothing has been stored"):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_failure_committing_the_pointer_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The whole replacement is written and verified, and the one write that publishes it fails.

    Everything staged is intact and correct at this moment; the only thing missing is the record
    that says it is current. A reader must still get the old session, because a generation
    nothing points at is not a credential.
    """
    store().save(a_session("WORKING", padding=600))
    keychain.fail_on_pointer_write()
    with pytest.raises(ConfigError):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_failure_recording_the_staged_generation_keeps_the_session_that_worked(
    keychain: FakeKeychain,
) -> None:
    """The first write a replacement makes is the journal, and it can fail like any other."""
    store().save(a_session("WORKING", padding=600))
    keychain.fail_on_journal_write()
    with pytest.raises(ConfigError):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_process_killed_immediately_after_the_commit_leaves_the_new_session_whole(
    keychain: FakeKeychain,
) -> None:
    """The other side of the commit point, and the one that says where the line is.

    Dying here means no confirmation read and no cleanup ran, so the replaced generation is
    still sitting in the keychain. A reader must get the *new* session all the same: the commit
    is what decides, and everything after it is tidying.
    """
    store().save(a_session("WORKING", padding=600))
    keychain.fail_on_pointer_write(fault=Fault.CRASH_AFTER_WRITE)
    with pytest.raises(SimulatedTermination):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    restored = store().load(SITE)
    assert marker_of(restored) == "REPLACEMENT"
    assert restored is not None
    assert restored.cookies["MSISAuth"].get_secret_value() == "R" * 600, "whole, not a prefix"


def test_a_partly_staged_generation_is_still_in_the_keychain_and_is_never_returned(
    keychain: FakeKeychain,
) -> None:
    """That ``load`` ignores a partial generation is only meaningful if one is there to ignore.

    Every case above asserts the old session comes back. This one first proves the abandoned
    chunks genuinely exist, so that a passing result cannot be explained by there being nothing
    written at all — which is how a guard of this kind goes hollow.
    """
    store().save(a_session("WORKING", padding=600))
    settled = set(keychain.accounts(SERVICE))

    keychain.fail_on_chunk_write(3)
    with pytest.raises(ConfigError):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    abandoned = set(keychain.accounts(SERVICE)) - settled
    assert len(abandoned) >= 2, f"the abandoned chunks should be present, found {abandoned}"
    assert marker_of(store().load(SITE)) == "WORKING"


# --- the ordinary paths -------------------------------------------------------------------------


def test_a_session_saved_into_an_empty_keychain_comes_back(keychain: FakeKeychain) -> None:
    store().save(a_session("FIRST"))

    restored = store().load(SITE)
    assert restored is not None
    assert marker_of(restored) == "FIRST"
    assert restored.account == SESSION_ACCOUNT
    assert restored.captured_at == CAPTURED_AT


def test_an_empty_keychain_reports_no_session_rather_than_failing(keychain: FakeKeychain) -> None:
    assert store().load(SITE) is None


def test_a_successful_replacement_leaves_only_the_new_session(keychain: FakeKeychain) -> None:
    """Both halves: the new one is readable, and nothing of the old one is left behind."""
    store().save(a_session("OLD", padding=600))
    store().save(a_session("NEW", padding=600))

    assert marker_of(store().load(SITE)) == "NEW"
    held = " ".join(keychain.items.values())
    assert "O" * 600 not in held, "the replaced session's cookies are gone from the keychain"


def test_nothing_is_deleted_before_the_replacement_has_been_committed(
    keychain: FakeKeychain,
) -> None:
    """The defect, stated as the rule that prevents it rather than as one of its symptoms.

    Every failure case above is an instance of this. Asserting the ordering directly is what
    keeps the property from being re-broken by a change none of those cases happen to cover —
    a delete moved earlier for tidiness would pass all of them and fail this.
    """
    store().save(a_session("OLD", padding=600))
    first_of_the_replacement = len(keychain.calls)
    store().save(a_session("NEW", padding=600))

    replacement = keychain.calls[first_of_the_replacement:]
    commits = [
        n
        for n, call in enumerate(replacement)
        if call.is_pointer and call.subcommand == "add-generic-password"
    ]
    deletes = [
        n for n, call in enumerate(replacement) if call.subcommand == "delete-generic-password"
    ]

    assert commits, "the replacement committed"
    assert deletes, "the replacement cleaned up after itself"
    assert min(deletes) > max(commits), (
        "a delete ran before the commit, which is the window this whole scheme exists to close"
    )


def test_a_long_session_is_split_across_items_and_survives_the_round_trip(
    keychain: FakeKeychain,
) -> None:
    """A SAML instance issues cookies of its own, and the record is kilobytes rather than bytes.

    The fake truncates a secret at 128 bytes exactly as the real command does, so a store that
    stopped chunking would fail here rather than passing and failing on somebody's machine.
    """
    session = a_session("LONG", padding=3000)
    assert chunks_for(session) > 20, "the fixture is long enough to be worth the name"

    store().save(session)

    restored = store().load(SITE)
    assert restored is not None
    assert restored.cookies["MSISAuth"].get_secret_value() == "L" * 3000


def test_a_session_that_shrinks_cannot_inherit_a_tail_of_the_longer_one(
    keychain: FakeKeychain,
) -> None:
    """The old store wrote chunks over chunks, so a shorter record left the tail of a longer one.

    It solved that by deleting everything first, which is exactly what made a failed replacement
    destructive. Generation-addressed chunks solve it without the delete: there is no shared
    numbering for a tail to survive in.
    """
    store().save(a_session("LONG", padding=3000))
    store().save(a_session("SHORT"))

    restored = store().load(SITE)
    assert restored is not None
    assert marker_of(restored) == "SHORT"
    assert "MSISAuth" not in restored.cookies


# --- what a crash leaves for the next run to clear up -------------------------------------------


def test_a_cleanup_failure_leaves_the_new_session_active_and_says_what_was_left_behind(
    keychain: FakeKeychain, caplog: pytest.LogCaptureFixture
) -> None:
    """Cleanup runs after the session is stored, so its failure cannot be a failed save.

    What it can be is untidy, and the leftovers are live cookies — so it says so, without
    naming any of them.
    """
    caplog.set_level(logging.WARNING, logger="manicule.connectors")
    store().save(a_session("OLD", padding=600))
    keychain.fail_on_delete()
    store().save(a_session("NEW", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "NEW"

    assert "--forget" in caplog.text, "the message says how to clear the leftovers"
    assert "O" * 600 not in caplog.text
    assert "N" * 600 not in caplog.text

    # The journal was never rewritten, so it still names what cleanup did not finish and
    # `forget` can still reach it. Leftover secret material has to stay findable.
    assert store().forget(SITE) is True
    assert keychain.accounts(SERVICE) == []


def test_forget_removes_a_generation_that_an_interrupted_replacement_abandoned(
    keychain: FakeKeychain,
) -> None:
    """An operator asking for the session to be gone is asking about live cookies.

    A generation staged by a process that then died is referenced by nothing, so without the
    journal there would be no way to enumerate it — and ``--forget`` would report success while
    leaving a working credential in the keychain.
    """
    store().save(a_session("WORKING", padding=600))
    keychain.fail_on_chunk_write(3, fault=Fault.CRASH_AFTER_WRITE)
    with pytest.raises(SimulatedTermination):
        store().save(a_session("ABANDONED", padding=600))
    keychain.assert_fired()

    assert store().forget(SITE) is True
    assert keychain.accounts(SERVICE) == []


def test_a_run_of_interrupted_replacements_does_not_grow_the_journal_without_bound(
    keychain: FakeKeychain,
) -> None:
    """The journal has to fit one keychain item, so it prunes rather than overflowing.

    Pruning deletes what it drops. An entry removed from the journal without its chunks being
    removed would be exactly the unfindable leftover the journal exists to prevent.
    """
    store().save(a_session("WORKING", padding=600))
    for _ in range(MAX_STAGED * 2):
        keychain.fail_on_pointer_write()
        with pytest.raises(ConfigError):
            store().save(a_session("ABANDONED", padding=600))
        keychain.assert_fired()

    journal = keychain.items[(SERVICE, journal_account(SITE))]
    assert len(journal) <= CHUNK_BYTES, "the journal still fits one item"
    assert len(journal.split(" ")) <= MAX_STAGED

    assert marker_of(store().load(SITE)) == "WORKING"
    assert store().forget(SITE) is True
    assert keychain.accounts(SERVICE) == []


# --- a keychain that has been damaged -----------------------------------------------------------


def test_a_commit_naming_a_generation_that_is_gone_falls_back_to_the_verified_previous_one(
    keychain: FakeKeychain,
) -> None:
    """The deterministic rollback, and the reason there are two pointer slots rather than one.

    The older slot is not believed because it is older. It is believed because its own digest
    still describes the generation it names, which is checked here exactly as it is on the
    ordinary path.
    """
    store().save(a_session("OLD", padding=600))
    keychain.fail_on_delete()  # cleanup leaves the old generation in place
    store().save(a_session("NEW", padding=600))
    keychain.assert_fired()

    newest = max(
        (account for account in keychain.accounts(SERVICE) if account.endswith("#0")),
        key=lambda account: keychain.items[(SERVICE, account)],
    )
    del newest  # chosen below by the store's own pointer instead

    current = store().load(SITE)
    assert marker_of(current) == "NEW"

    # Remove the generation the newest commit names, leaving its pointer behind.
    active = _generation_named_by_the_newest_commit(keychain)
    for account in list(keychain.accounts(SERVICE)):
        if account.startswith(f"{SITE}#{active}#"):
            del keychain.items[(SERVICE, account)]

    assert marker_of(store().load(SITE)) == "OLD", "the previous commit is still verified"


def test_a_commit_naming_a_generation_that_is_gone_with_no_fallback_refuses_to_guess(
    keychain: FakeKeychain,
) -> None:
    """No verified generation anywhere means no credential, said out loud and without secrets.

    Returning ``None`` here would report "no session is stored", which is a different problem
    with a different repair and would send somebody looking in the wrong place.
    """
    session = a_session("ONLY", padding=600)
    store().save(session)
    active = _generation_named_by_the_newest_commit(keychain)
    for account in list(keychain.accounts(SERVICE)):
        if account.startswith(f"{SITE}#{active}#"):
            del keychain.items[(SERVICE, account)]

    with pytest.raises(ConfigError, match="incomplete") as raised:
        store().load(SITE)

    message = str(raised.value)
    assert "ONLY" not in message
    assert "O" * 600 not in message
    assert active not in message, "a generation identifier is metadata about a secret"


def test_a_pointer_slot_that_cannot_be_parsed_is_ignored_rather_than_guessed_at(
    keychain: FakeKeychain,
) -> None:
    """A keychain outlives the versions that wrote to it, so an unreadable slot is inert.

    Loosely parsing a corrupt slot would send ``load`` after a generation that was never
    written; ignoring it lets the other slot answer, which is what the other slot is for.
    """
    store().save(a_session("OLD", padding=600))
    keychain.fail_on_delete()
    store().save(a_session("NEW", padding=600))
    keychain.assert_fired()

    held = {slot: keychain.items[(SERVICE, pointer_account(SITE, slot))] for slot in POINTER_SLOTS}
    newer = max(POINTER_SLOTS, key=lambda slot: int(held[slot].split(" ")[1]))
    keychain.items[(SERVICE, pointer_account(SITE, newer))] = "v9 not a commit at all"

    assert marker_of(store().load(SITE)) == "OLD"


def test_both_pointer_slots_unreadable_reports_no_session_rather_than_an_unverified_one(
    keychain: FakeKeychain,
) -> None:
    """The honest limit of the scheme, pinned so that it is a decision rather than a surprise.

    Generation chunks are still in the keychain here and could be reassembled. They are not,
    because nothing left says which of them is a session or that any is complete, and using an
    unverified record is the one thing this store must never do.
    """
    store().save(a_session("ORPHANED", padding=600))
    for slot in POINTER_SLOTS:
        keychain.items.pop((SERVICE, pointer_account(SITE, slot)), None)

    assert store().load(SITE) is None
    assert any("#" in account for account in keychain.accounts(SERVICE)), (
        "the chunks are still there; the store declines to guess at them"
    )


def test_a_generation_holding_a_chunk_of_another_session_is_refused(
    keychain: FakeKeychain,
) -> None:
    """What the digest is for, with the length deliberately unable to help.

    The two sessions here encode to exactly the same number of bytes, so swapping one chunk
    between them leaves a record of the right length made of the wrong pieces. Only comparing
    the content against what was committed can tell that apart from a session — and a store that
    served it would be authenticating as a cookie assembled from two different captures.

    The older commit is removed afterwards so that the refusal is the digest's doing and not the
    rollback's. With the older slot left in place this passes for the wrong reason: the mixture
    is rejected, the previous session is returned instead, and the case would go green against a
    store that had no digest at all.
    """
    store().save(a_session("ONE", padding=600))
    keychain.fail_on_delete()  # cleanup fails, so the replaced generation is still available
    store().save(a_session("TWO", padding=600))
    keychain.assert_fired()

    active = _generation_named_by_the_newest_commit(keychain)
    stale = next(name for name in _generations_present(keychain) if name != active)
    borrowed = keychain.items[(SERVICE, generation_account(SITE, stale, 1))]
    replaced = keychain.items[(SERVICE, generation_account(SITE, active, 1))]
    assert len(borrowed) == len(replaced), "the swap must not change the length"
    assert borrowed != replaced, "and must actually change the content"
    keychain.items[(SERVICE, generation_account(SITE, active, 1))] = borrowed

    for slot in POINTER_SLOTS:
        held = keychain.items.get((SERVICE, pointer_account(SITE, slot)), "")
        if active not in held:
            del keychain.items[(SERVICE, pointer_account(SITE, slot))]

    with pytest.raises(ConfigError, match="incomplete"):
        store().load(SITE)


def test_a_record_that_is_not_a_session_says_how_to_replace_it(keychain: FakeKeychain) -> None:
    store().save(a_session("FINE"))
    active = _generation_named_by_the_newest_commit(keychain)
    keychain.items[(SERVICE, generation_account(SITE, active, 0))] = "not base64 at all"

    with pytest.raises(ConfigError, match="incomplete"):
        store().load(SITE)


# --- records written by an earlier version ------------------------------------------------------


def test_a_session_stored_before_generations_existed_still_loads(keychain: FakeKeychain) -> None:
    """The compatibility that matters: somebody upgrades and their stored session keeps working."""
    seed_legacy(keychain, a_session("FROM-AN-OLDER-VERSION", padding=600))

    restored = store().load(SITE)
    assert restored is not None
    assert marker_of(restored) == "FROM-AN-OLDER-VERSION"
    assert restored.cookies["MSISAuth"].get_secret_value() == "F" * 600


def test_the_first_save_migrates_a_legacy_record_without_deleting_it_first(
    keychain: FakeKeychain,
) -> None:
    """Migration gets the same guarantee as replacement, which means the same ordering.

    Removing the old record to make room would reopen the defect for precisely the operators
    who are upgrading — the ones with a working session and no reason to expect losing it.
    """
    seed_legacy(keychain, a_session("LEGACY", padding=600))
    before = len(keychain.calls)
    store().save(a_session("MIGRATED", padding=600))

    migration = keychain.calls[before:]
    committed = min(
        n
        for n, call in enumerate(migration)
        if call.is_pointer and call.subcommand == "add-generic-password"
    )
    legacy_deletes = [
        n
        for n, call in enumerate(migration)
        if call.subcommand == "delete-generic-password" and not call.is_pointer
    ]

    assert legacy_deletes, "the legacy record was cleared up"
    assert min(legacy_deletes) > committed, "the legacy record outlived the commit that replaced it"
    assert marker_of(store().load(SITE)) == "MIGRATED"
    assert (SERVICE, chunk_account(SITE, 0)) not in keychain.items


def test_a_failed_migration_leaves_the_legacy_record_readable(keychain: FakeKeychain) -> None:
    """The case the ordering above exists for, asserted from the operator's side."""
    seed_legacy(keychain, a_session("LEGACY", padding=600))
    keychain.fail_on_chunk_write(2)
    with pytest.raises(ConfigError):
        store().save(a_session("MIGRATED", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "LEGACY"


# --- one instance's records are not another's ---------------------------------------------------


def test_two_instances_keep_separate_sessions_and_forgetting_one_leaves_the_other(
    keychain: FakeKeychain,
) -> None:
    """A session is not portable between instances, and one offered to the wrong site
    authenticates as nobody there while looking configured."""
    store().save(a_session("HERE", padding=600))
    store().save(a_session("THERE", base_url=OTHER_SITE, padding=600))

    assert marker_of(store().load(SITE)) == "HERE"
    assert marker_of(store().load(OTHER_SITE)) == "THERE"

    assert store().forget(SITE) is True
    assert store().load(SITE) is None
    assert marker_of(store().load(OTHER_SITE)) == "THERE"


def test_forget_clears_active_abandoned_and_legacy_records_for_one_instance_only(
    keychain: FakeKeychain,
) -> None:
    """All three kinds at once, because ``--forget`` is a promise about all three."""
    seed_legacy(keychain, a_session("LEGACY", padding=600))
    store().save(a_session("THERE", base_url=OTHER_SITE, padding=600))
    keychain.fail_on_delete()  # the legacy record survives the migration's cleanup
    store().save(a_session("ACTIVE", padding=600))
    keychain.assert_fired()

    keychain.fail_on_pointer_write()
    with pytest.raises(ConfigError):
        store().save(a_session("ABANDONED", padding=600))
    keychain.assert_fired()

    assert store().forget(SITE) is True
    assert store().load(SITE) is None
    assert [account for account in keychain.accounts(SERVICE) if account.startswith(SITE)] == []
    assert marker_of(store().load(OTHER_SITE)) == "THERE"


def test_forgetting_a_site_that_has_nothing_stored_reports_that_it_removed_nothing(
    keychain: FakeKeychain,
) -> None:
    store().save(a_session("THERE", base_url=OTHER_SITE))

    assert store().forget(SITE) is False
    assert marker_of(store().load(OTHER_SITE)) == "THERE"


def test_a_trailing_slash_names_the_same_instance(keychain: FakeKeychain) -> None:
    store().save(a_session("HERE"))

    assert marker_of(store().load(f"{SITE}/")) == "HERE"


# --- two writers at once ------------------------------------------------------------------------


def test_a_replacement_staged_by_one_writer_cannot_be_overwritten_by_another(
    keychain: FakeKeychain,
) -> None:
    """Generations are random so that two writers never choose the same name.

    If they could, one would write its chunks over the other's and the loser could commit a
    pointer to a generation holding a mixture of both sessions — which is the one outcome the
    whole scheme is built to make impossible.

    A's session is the longer of the two, and that is load-bearing rather than incidental: if
    both writers used the same generation, B's shorter record would land on top of A's chunks
    and leave A's tail behind it, so B would read back its own staging followed by somebody
    else's. Equal-length sessions would hide that, and this case would then pass against a store
    whose generations were a fixed name — which is how it was written first, and it did.
    """
    store().save(a_session("BEFORE", padding=600))

    # The first writer stages a long replacement and dies before committing it.
    keychain.fail_on_pointer_write(fault=Fault.CRASH)
    with pytest.raises(SimulatedTermination):
        store().save(a_session("WRITER-A", padding=3000))
    keychain.assert_fired()

    # The second writer runs to completion while A's generation is still sitting there.
    store().save(a_session("WRITER-B", padding=600))

    restored = store().load(SITE)
    assert marker_of(restored) == "WRITER-B"
    assert restored is not None
    assert restored.cookies["MSISAuth"].get_secret_value() == "W" * 600, "whole, not a mixture"


def test_two_commits_at_the_same_sequence_resolve_to_one_complete_session(
    keychain: FakeKeychain,
) -> None:
    """The state two simultaneous replacements can produce, and the contract for reading it.

    Two writers that both read the pointers before either committed will choose the same slot
    and the same sequence number, and the second to write wins. The contract is *last writer
    wins*: a replacement that loses the race is silently superseded, and no reader ever sees a
    mixture. This constructs that state directly, because the interleaving that produces it
    cannot be arranged deterministically from outside ``save``.
    """
    store().save(a_session("ONE", padding=600))
    keychain.fail_on_delete()
    store().save(a_session("TWO", padding=600))
    keychain.assert_fired()

    held = {slot: keychain.items[(SERVICE, pointer_account(SITE, slot))] for slot in POINTER_SLOTS}
    older, newer = sorted(POINTER_SLOTS, key=lambda slot: int(held[slot].split(" ")[1]))
    tied = held[older].split(" ")
    tied[1] = held[newer].split(" ")[1]
    keychain.items[(SERVICE, pointer_account(SITE, older))] = " ".join(tied)

    first = store().load(SITE)
    assert marker_of(first) in {"ONE", "TWO"}
    assert first is not None
    assert len(first.cookies["MSISAuth"].get_secret_value()) == 600, "whole, not a mixture"
    assert marker_of(store().load(SITE)) == marker_of(first), "and the same one every time"


# --- the rules that hold whatever else happens --------------------------------------------------


def test_no_cookie_and_no_stored_record_ever_reaches_a_command_line(
    keychain: FakeKeychain,
) -> None:
    """``security`` reads a secret from stdin. An argument would put a live corporate session
    into this process's command line, where anything on the machine can read it."""
    session = a_session("TOP-SECRET-SESSION", padding=600)
    store().save(session)
    store().load(SITE)
    store().forget(SITE)

    flattened = keychain.command_lines()
    assert "TOP-SECRET-SESSION" not in flattened
    assert "T" * 600 not in flattened
    assert payload_of(session)[:CHUNK_BYTES] not in flattened
    assert "-A" not in flattened, "any application reading it silently is not the grant we want"


def test_a_commit_that_did_not_take_is_not_reported_as_a_stored_session(
    keychain: FakeKeychain,
) -> None:
    """The confirmation read, which is the only thing that checks the *pointer* rather than the
    chunks.

    A commit stored short parses as nothing, so the slot reads back empty and the replacement is
    published to nobody — every chunk correct, every write reporting success, and no session
    changed. Comparing the staged chunks cannot see this, because the staged chunks are fine.
    """
    store().save(a_session("WORKING", padding=600))
    keychain.truncate_pointer_write(keep=12)
    with pytest.raises(ConfigError, match="does not reach it"):
        store().save(a_session("REPLACEMENT", padding=600))
    keychain.assert_fired()

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_marker_record_that_no_longer_fits_one_item_is_refused_rather_than_truncated(
    keychain: FakeKeychain,
) -> None:
    """A pointer stored as its first 128 bytes reads back as no commit at all.

    That would lose a session on a *successful* save, which is the worst failure available here:
    every write reports success and the credential is gone. So a marker that has outgrown one
    item stops the save instead, and the session already stored is untouched.
    """
    store().save(a_session("WORKING", padding=600))
    with (
        patch.object(sessions, "CHUNK_BYTES", 8),
        pytest.raises(ConfigError, match="fits in one keychain item"),
    ):
        store().save(a_session("REPLACEMENT", padding=600))

    assert marker_of(store().load(SITE)) == "WORKING"


def test_a_value_containing_a_newline_is_refused_before_it_reaches_the_keychain() -> None:
    """A tripwire on the encoding, reached directly because nothing can reach it through ``save``.

    ``security`` reads a secret from stdin as newline-terminated, so a stored value would
    silently become the part before the newline. Nothing manicule writes today can contain one —
    base64 cannot, and the marker records are hexadecimal and spaces — which is exactly why this
    is worth pinning: the next encoding somebody reaches for might.
    """
    # Reaching past the public surface on purpose: no public call can carry a newline this far,
    # which is what makes the branch worth pinning and also what makes it unreachable from here.
    with pytest.raises(ConfigError, match="newline"):
        store()._write(chunk_account(SITE, 0), "first\nsecond", SITE)  # pyright: ignore[reportPrivateUsage]


def test_the_journal_and_the_pointer_both_fit_in_one_keychain_item() -> None:
    """Neither can be chunked, because a chunked pointer would need a pointer of its own.

    Sizes drift, and a pointer that outgrew one item would be stored as its first 128 bytes,
    read back as a slot this version cannot parse, and therefore treated as no commit at all —
    losing a session on a *successful* save. Arithmetic here rather than a runtime surprise.
    """
    longest_journal = MAX_STAGED * (GENERATION_HEX + 1) - 1
    assert longest_journal <= CHUNK_BYTES

    sequence_digits = 12  # a save a second for thirty thousand years
    longest_commit = (
        len("v1") + 1 + sequence_digits + 1 + GENERATION_HEX + 1 + len("30720") + 1 + DIGEST_HEX
    )
    assert longest_commit <= CHUNK_BYTES


def test_the_records_a_save_leaves_behind_are_the_documented_ones(keychain: FakeKeychain) -> None:
    """The storage layout, written out so a reviewer can read it without reading the code.

    Also the guard that keeps :mod:`tests.connectors.keychain_fake` honest: the fake decides
    what counts as a chunk, a pointer and a journal by the shape of the account name, and every
    fault-injection case above depends on it deciding correctly.
    """
    store().save(a_session("ONE"))

    site = "https://wiki.example.test/confluence"
    accounts = keychain.accounts(SERVICE)
    pointers = [account for account in accounts if account in {f"{site}#p0", f"{site}#p1"}]
    journals = [account for account in accounts if account == f"{site}#staged"]
    chunks = [account for account in accounts if account not in pointers + journals]

    assert pointers == [f"{site}#p0"], "one commit slot, and the first one on a fresh store"
    assert journals == [f"{site}#staged"]
    assert chunks, "the session itself"
    for account in chunks:
        generation = account[len(site) + 1 :].split("#")[0]
        assert len(generation) == GENERATION_HEX
        assert account == f"{site}#{generation}#{chunks.index(account)}"

    # Every item carries the site in its label, which is what somebody sees in Keychain Access
    # when they go looking for what manicule has stored. Asserted because the value passed here
    # is not the value the store operates on, so nothing else in this file would notice it
    # becoming the wrong thing — pyright caught exactly that once, and only because it is typed.
    for call in keychain.calls:
        if call.subcommand != "add-generic-password":
            continue
        label = call.arguments[call.arguments.index("-l") + 1]
        assert label == f"manicule — {site}", f"a keychain item is labeled {label!r}"

    assert [call for call in keychain.calls if call.is_pointer], "the fake recognizes a pointer"
    assert [call for call in keychain.calls if call.is_journal], "the fake recognizes a journal"
    assert [call for call in keychain.calls if call.is_chunk], "the fake recognizes a chunk"


def _generations_present(keychain: FakeKeychain) -> set[str]:
    """Every generation that has chunks in the keychain for :data:`SITE`."""
    prefix = f"{SITE}#"
    found: set[str] = set()
    for account in keychain.accounts(SERVICE):
        tail = account[len(prefix) :]
        if account.startswith(prefix) and tail.count("#") == 1:
            found.add(tail.split("#")[0])
    return found


def _generation_named_by_the_newest_commit(keychain: FakeKeychain) -> str:
    """Which generation the winning pointer slot points at, read the way the store reads it."""
    commits = [
        keychain.items[(SERVICE, pointer_account(SITE, slot))]
        for slot in POINTER_SLOTS
        if (SERVICE, pointer_account(SITE, slot)) in keychain.items
    ]
    newest = max(commits, key=lambda raw: int(raw.split(" ")[1]))
    return newest.split(" ")[2]
