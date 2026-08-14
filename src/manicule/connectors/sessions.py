"""Where a Confluence browser session comes from, and where it lives.

Self-hosted Confluence behind an identity provider commonly has personal access tokens disabled
by policy, so the credential its users can actually obtain is the session they already hold in
their browser. Three decisions shape this module, and each is a refusal of an easier option.

**manicule never asks for the password and has nowhere to put one.** No password, no one-time
code and no device approval passes through this process on any path, and that much *is* a fact
about the code rather than a promise: there is no parameter that could carry one and no branch
that would accept one.

**A browser is now driven, and this paragraph used to say the opposite.** It said that Playwright
was the ergonomic answer, that the license (Apache-2.0) was not the objection, and that the
objection was this: a driven browser is a browser manicule controls the DOM of, and the person is
asked to type a corporate password into it, so "manicule never sees the password" would become a
promise about restraint instead of a fact about capability.

That argument was right and it has been overruled, for a reason the argument did not weigh. An
instance behind an identity provider commonly has personal access tokens disabled by policy. For
those installations the manual paste was not the safer of two options — it was the *only* option,
and it asks somebody to open developer tools, find a live session cookie and paste several
kilobytes of it into a terminal. Refusing to drive a browser did not remove the risk; it moved it
onto the person, by hand, every time their session expired.

So the property has genuinely weakened, and it is worth naming precisely rather than glossing:

*Before:* manicule **cannot** see the password, because there is no browser.
*Now, on ``--browser`` only:* manicule **does not** see the password, because
    :mod:`manicule.connectors.browser` reads no page content — a claim a test enforces over that
    module's source, and a reviewer enforces over its diff.

*On this module's own path, unchanged:* manicule cannot see it, because there is still no
    browser. **The paste is not deprecated and is not a fallback.** It is the option that keeps
    the stronger guarantee, and somebody who wants that guarantee should use it.

The two practical warnings from the original argument stand and are documented rather than
solved: a driven Chromium is a new device to a conditional-access policy and may be refused
outright, and the browser is a heavy dependency — which is why it is an extra
(``manicule[browser-auth]``) that nothing else needs.

**The session lives in the macOS Keychain, and nowhere else.** Not ``config.toml``, even at
``0600``: a session cookie is the sync account's whole identity at that company rather than a
scoped grant, and a configuration file ends up in version control eventually. Not under
``<data_dir>`` either — which is why none of ``docs/storage.md`` §7.1's permission rules apply
to it, because nothing lands there. ``ConfluenceConfig`` forbids unknown keys, so a
``session_cookie`` written into configuration is a startup error rather than a working setting.

On a machine with no Keychain — Linux, a container — the fallback is an environment variable
(``session_env``), which is a per-run credential and never written down by manicule at all.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import SecretStr

from manicule.connectors.config import ConfluenceConfig
from manicule.connectors.credentials import BrowserSession
from manicule.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = [
    "KEYCHAIN_SERVICE",
    "KeychainStore",
    "MemoryStore",
    "SessionStore",
    "capture",
    "capture_cookies",
    "cookies_authenticate",
    "default_store",
    "load_session",
    "parse_cookies",
]

KEYCHAIN_SERVICE = "manicule: confluence session"
"""The keychain service every stored session shares. The account is the instance's base URL."""

SECURITY = "/usr/bin/security"
"""macOS's keychain command, by absolute path so that ``$PATH`` cannot choose a different one."""

_NOT_FOUND = 44
"""``security``'s exit status for an item that is not in the keychain."""

CHUNK_BYTES = 120
"""How much of the stored record goes into one keychain item.

``security`` reads a secret from stdin through a fixed 128-byte buffer and silently keeps the
first 128 bytes of anything longer — measured on macOS 15, and reported as success either way.
120 leaves room for a version whose buffer is a little smaller, and the read-back comparison in
:meth:`KeychainStore.save` catches one whose buffer is smaller still.
"""

MAX_CHUNKS = 256
"""How many pieces one session may occupy — about 30 KB of cookies, and a bound on the walk."""

POINTER_SLOTS: Final = ("p0", "p1")
"""The two keychain items that between them say which generation is the session.

Two rather than one, and this is the whole of why replacement is safe. Updating a single
pointer in place has a moment where it holds neither the old value nor the new — ``security``
does not promise otherwise, and a process that stops in that moment leaves a keychain with a
complete session in it and nothing that says where. Writing the slot that is *not* currently
authoritative means the previous commit is readable at every instant, including during the
write that supersedes it.
"""

JOURNAL_SLOT: Final = "staged"
"""The keychain item naming generations that have been written and not yet cleaned up.

Its only job is to make an abandoned generation *findable*. A generation staged by a process
that then died is unreferenced by either pointer slot, so without this record nothing could
enumerate it and ``forget`` would leave live cookies in the keychain after an operator asked
for them to be gone.
"""

MAX_STAGED: Final = 8
"""How many generations the journal names before it prunes the ones no pointer refers to.

The journal has to fit one keychain item, which is what bounds it. Eight entries of
:data:`GENERATION_HEX` characters and a separator each is 71 characters, comfortably inside
:data:`CHUNK_BYTES`, and ``test_the_journal_and_the_pointer_both_fit_in_one_keychain_item``
is what keeps that true if either size moves.
"""

GENERATION_HEX: Final = 8
"""How long a generation identifier is, in hexadecimal characters — 32 bits.

Unpredictable rather than sequential, because two processes replacing at once must not choose
the same name: if they did, one would write its chunks over the other's and the loser could
commit a pointer to a generation holding a mixture of both. Randomness is what makes the two
writers independent, and 32 bits is far past the collision odds of an interactive command that
one person runs a handful of times a year.
"""

DIGEST_HEX: Final = 32
"""How much of the payload's SHA-256 a commit carries — 128 bits.

The length in a commit already catches truncation. This catches everything else: a generation
whose chunks were written by two different sessions, a stale tail, an item edited by hand. It
is stored beside the secret it describes, under the same keychain protection, and it never
appears in a message.
"""

POINTER_VERSION: Final = "v1"
"""What a commit record starts with. A slot that does not start with this is not read.

A keychain outlives the versions of the software that wrote to it. An unreadable slot is
treated as no commit at all rather than guessed at, which is what lets a future format add
fields without a version of manicule that predates them mistaking one for a session.
"""

_HEX_DIGITS: Final = frozenset("0123456789abcdef")

_COMMIT_FIELDS: Final = 5
"""How many space-separated fields a commit record has: version, sequence, generation, length,
digest. A record with any other count is from a format this version does not know."""

_PROBE_PATH = "/rest/api/user/current"

_TIMEOUT_SECONDS = 20.0

_log = logging.getLogger("manicule.connectors")


@dataclass(frozen=True, slots=True)
class _Commit:
    """What one pointer slot says: which generation is the session, and how to tell it is whole.

    ``length`` and ``digest`` are the difference between "the chunks I could find" and "the
    chunks that were written". Walking numbered items until one is missing cannot tell a
    complete record from one truncated at chunk three of five — both end at a gap — so without
    these a pointer would be a promise the store had no way to check.
    """

    sequence: int
    """Which commit this is. The higher of the two slots is the current session."""

    generation: str
    digest: str
    length: int

    def encode(self) -> str:
        """The one line a pointer slot holds. Space-separated, and never containing a newline.

        The stdin protocol ``security`` uses for a secret is two newline-terminated copies of
        the value, so a newline inside a value would be read as the end of it.
        """
        return " ".join(
            (POINTER_VERSION, str(self.sequence), self.generation, str(self.length), self.digest)
        )

    @classmethod
    def decode(cls, raw: str) -> _Commit | None:
        """The commit in a pointer slot, or ``None`` for anything this version cannot read.

        Every field is checked rather than trusted. The store reads what a keychain gives it,
        and a malformed slot has to be inert — a commit parsed loosely out of a corrupt item
        would send ``load`` looking for a generation that was never written.
        """
        parts = raw.split(" ")
        if len(parts) != _COMMIT_FIELDS or parts[0] != POINTER_VERSION:
            return None
        _, sequence, generation, length, digest = parts
        if not sequence.isdigit() or not length.isdigit():
            return None
        if not _is_hex(generation, GENERATION_HEX) or not _is_hex(digest, DIGEST_HEX):
            return None
        return cls(sequence=int(sequence), generation=generation, length=int(length), digest=digest)

    def describes(self, payload: str) -> bool:
        """Whether ``payload`` is the whole of what this commit was made for."""
        return len(payload) == self.length and _digest(payload) == self.digest


@dataclass(frozen=True, slots=True)
class _Stored:
    """A commit and the slot it was read from, so the other slot can be the one written next."""

    slot: str
    commit: _Commit


@dataclass(frozen=True, slots=True)
class _Current:
    """What following the pointers found.

    The two ways of having no payload are different problems and need different messages: a
    store nobody has saved to is empty and a store whose pointer names an incomplete generation
    is broken. Collapsing them would tell an operator to run ``connector login`` in one case
    and say nothing at all about the other.
    """

    payload: str | None
    committed: bool


class SessionStore(Protocol):
    """Where captured sessions are kept."""

    def load(self, base_url: str) -> BrowserSession | None: ...

    def save(self, session: BrowserSession) -> None: ...

    def forget(self, base_url: str) -> bool:
        """Remove the session for ``base_url``. ``True`` if there was one."""
        ...

    def describe(self) -> str:
        """Where this keeps things, for a message that tells somebody what just happened."""
        ...


class KeychainStore:
    """The macOS Keychain, reached through ``/usr/bin/security``.

    A subprocess rather than a library binding because the alternative binds the Security
    framework through ``ctypes``, and an item created that way trusts *the calling binary* —
    which for manicule is a virtual environment's Python. Recreate the environment or upgrade
    the interpreter and every sync starts raising a Keychain dialog. ``/usr/bin/security`` is a
    path that does not move.

    **The secret never appears in an argument vector.** ``security`` reads it from stdin when
    ``-w`` is given no value, so it is not in ``ps``, not in a process listing and not in
    anything that records command lines. Keeping that property is what forces the next two
    paragraphs, and the property is worth the trouble: a Confluence session is the sync
    account's whole identity at that company.

    **The stdin route truncates at 128 bytes, silently.** Measured, not assumed: 128 bytes are
    stored, 129 are stored as the first 128, and ``security`` reports success either way. A
    session record is several hundred bytes and an instance behind single sign-on often issues
    cookies of its own besides Confluence's, so this is not an edge case — the whole credential
    would be stored broken, and a broken session is one that authenticates as nobody and gets a
    sign-in page back. So the record is written in :data:`CHUNK_BYTES` pieces across numbered
    items, and read back by walking them until one is missing.

    **Every write is read back and compared.** Chunking works around today's limit; the read-back
    is what makes a *different* limit a loud failure rather than a quietly truncated credential.

    Items are created with ``-T /usr/bin/security``, the narrowest grant that still lets an
    unattended sync run: ``security`` may read them without a prompt and any other program
    raises the Keychain's own dialog. ``-A``, which would let anything read them silently, is
    not used.

    **Replacement never deletes the session it is replacing.** This class used to, and the
    consequence was that a ``security`` invocation failing on the fourth of twenty-three chunks
    left an operator with no credential at all, having started with a working one. Verifying a
    candidate before writing it — which :func:`capture_cookies` does — protects against a bad
    *candidate* and does nothing whatever about a bad *write*.

    So a replacement is written somewhere the current reader is not looking, and becomes the
    session by a separate, single, small write. Three records make that work, and each is here
    for one reason:

    - **A generation.** Chunks are filed under ``<site>#<generation>#<n>`` where the generation
      is 32 fresh random bits. The new session therefore cannot overwrite any part of the old
      one, which is what rules out a reader seeing a mixture of the two.
    - **Two pointer slots**, ``<site>#p0`` and ``<site>#p1``, each holding a sequence number,
      a generation, a length and a digest. The higher sequence wins. A replacement writes the
      slot that did *not* win, so the previous commit stays readable throughout — see
      :data:`POINTER_SLOTS`.
    - **A journal**, ``<site>#staged``, naming generations that have been written and not yet
      cleaned up, so that :meth:`forget` can remove one that a crash abandoned. See
      :data:`JOURNAL_SLOT`.

    **The commit point is the pointer write, and it is one keychain item.** Everything before it
    is invisible to a reader; everything after it is cleanup. What this does *not* claim is that
    ``security``'s update of that one item is itself atomic — it may well be a delete and an add
    inside the command, and macOS does not say. The dual slot is what makes that not matter: an
    interrupted pointer write can lose at most the slot being written, and the other slot still
    holds the commit before it.

    **What a reader can see, exhaustively.** Either the complete old session, or the complete
    new one, or — if a pointer names a generation that is not whole — a refusal that says so.
    Never a mixture, never a truncated record, never a session that was not read back and
    compared. ``docs/connectors/confluence.md`` §1.1d states the limits of that in operator
    terms.
    """

    def __init__(self, service: str = KEYCHAIN_SERVICE) -> None:
        self._service = service

    @staticmethod
    def available() -> bool:
        """Whether this machine has the Keychain command at all."""
        return sys.platform == "darwin" and shutil.which(SECURITY) is not None

    def describe(self) -> str:
        return f"the macOS Keychain, under the service {self._service!r}"

    def load(self, base_url: str) -> BrowserSession | None:
        """The stored session for this instance, or ``None`` if there has never been one.

        Raises:
            ConfigError: A commit exists and the generation it names is not whole, or the
                record it names is not a session manicule wrote. Both mean the same thing to
                the person reading it — capture again — and neither is allowed to be silent,
                because the alternative to a message here is a sync that starts and finds out.
        """
        current = self._current(_account(base_url))
        if current.payload is None:
            if current.committed:
                msg = (
                    f"the stored Confluence session for {base_url} is incomplete: the keychain "
                    f"records which session is current, and the pieces it names are not all "
                    f"there. Nothing has been guessed at and no partial credential will be "
                    f"used. Run `manicule connector login <name>` to capture one again."
                )
                raise ConfigError(msg)
            return None
        try:
            return BrowserSession.from_json(base64.b64decode(current.payload).decode())
        except (ValueError, UnicodeDecodeError) as exc:
            msg = (
                f"the keychain item for {base_url} is not a session manicule wrote ({exc}). "
                f"Run `manicule connector login <name>` to replace it."
            )
            raise ConfigError(msg) from exc

    def save(self, session: BrowserSession) -> None:
        """Store ``session``, without putting the one already stored at risk.

        The order is the guarantee, so it is written out rather than inferred: record what is
        about to be staged, stage it under a generation nothing reads yet, compare it byte for
        byte, commit it with one write to the pointer slot that is not authoritative, confirm
        that the ordinary read path now reaches it, and only then remove what it replaced.

        Every step before the commit can fail or be interrupted without consequence: none of
        them touches a record a reader consults. The two steps after it have already succeeded
        at storing the session, so neither is allowed to take it away again.

        Raises:
            ConfigError: The keychain refused a write, or gave back something other than what
                was written. In both cases the session that was already stored is untouched.
        """
        payload = base64.b64encode(session.to_json().encode()).decode()
        account = _account(session.base_url)
        commits = self._commits(account)
        generation = _generation()

        journal = self._record_staged(
            account, generation, commits, self._staged(account), session.base_url
        )
        self._stage(account, generation, payload, session.base_url)
        self._compare_staged(account, generation, payload, session.base_url)

        commit = _Commit(
            sequence=commits[0].commit.sequence + 1 if commits else 1,
            generation=generation,
            length=len(payload),
            digest=_digest(payload),
        )
        # The commit point. Before this line a reader sees the session this one replaces; after
        # it, this one. There is no third thing it can see, because a generation is never
        # written over and the slot being written is not the slot being read.
        self._write_marker(
            _pointer(account, _free_slot(commits)), commit.encode(), session.base_url
        )

        self._confirm(account, payload, session.base_url)

        # Everything this store knew about before the commit, which is exactly what the commit
        # replaced. Taken from what was already read rather than read again: a generation that
        # appeared in between belongs to another writer and is not this call's to remove.
        replaced = set(journal) | {stored.commit.generation for stored in commits}
        self._clean_up_quietly(
            account, doomed=replaced - {generation}, keep=generation, base_url=session.base_url
        )

    def forget(self, base_url: str) -> bool:
        """Remove everything this instance's session occupies. ``True`` if there was anything.

        Everything means: the generation that is current, the generation an interrupted
        replacement abandoned, both pointer slots, the journal, and a record written by a
        version of manicule that predates all of them. An operator who asks for a session to be
        gone is asking about live cookies, and a fragment left behind is live cookies.

        Records belonging to another instance are untouched: every account name this builds is
        prefixed with this instance's own normalized base URL.
        """
        account = _account(base_url)
        generations = set(self._staged(account))
        for stored in self._commits(account):
            generations.add(stored.commit.generation)

        removed = False
        for generation in sorted(generations):
            removed = self._discard(account, generation) or removed
        for slot in POINTER_SLOTS:
            removed = self._delete(_pointer(account, slot)) or removed
        removed = self._delete(_journal(account)) or removed
        return self._forget_legacy(account) or removed

    # --- reading ---------------------------------------------------------------------------

    def _current(self, account: str) -> _Current:
        """What the pointers resolve to: the newest commit whose generation is whole.

        Falling to the older slot is the deterministic rollback. It happens when the newest
        commit names a generation that is missing or short — a replacement whose cleanup ran
        against the wrong generation, or an item removed from outside manicule — and the older
        slot is checked by its own digest before being believed, so this is a *verified*
        previous session rather than a guess that something older is probably fine.

        With no commit in either slot, a record written by a version that predates them is
        read instead. That is the migration path, and it is read-only: nothing here writes.
        """
        commits = self._commits(account)
        if not commits:
            return _Current(payload=self._legacy(account), committed=False)
        for stored in commits:
            payload = self._assemble(
                _staged_chunk(account, stored.commit.generation, index)
                for index in range(MAX_CHUNKS)
            )
            if payload is not None and stored.commit.describes(payload):
                return _Current(payload=payload, committed=True)
        return _Current(payload=None, committed=True)

    def _commits(self, account: str) -> list[_Stored]:
        """Both pointer slots, newest first, with anything unreadable dropped."""
        found: list[_Stored] = []
        for slot in POINTER_SLOTS:
            raw = self._read_one(_pointer(account, slot))
            if raw is None:
                continue
            commit = _Commit.decode(raw)
            if commit is not None:
                found.append(_Stored(slot=slot, commit=commit))
        found.sort(key=lambda stored: (stored.commit.sequence, stored.slot), reverse=True)
        return found

    def _staged(self, account: str) -> list[str]:
        """The generations the journal names. Anything that is not one is ignored."""
        raw = self._read_one(_journal(account))
        if raw is None:
            return []
        return [word for word in raw.split(" ") if _is_hex(word, GENERATION_HEX)]

    def _legacy(self, account: str) -> str | None:
        """A record written before generations existed, under ``<site>#<n>``.

        Read on its own terms and never rewritten in place. Such a record has no length and no
        digest beside it, so it cannot be checked the way a generation is; it is returned as
        found, exactly as the version that wrote it would have returned it. The first
        successful save replaces it with a generation and removes it.
        """
        return self._assemble(_chunk(account, index) for index in range(MAX_CHUNKS))

    def _assemble(self, accounts: Iterable[str]) -> str | None:
        """The pieces at these accounts joined in order, stopping at the first one absent.

        ``None`` when the first is absent, which is how "there is no such record" is spelled.
        Stopping at a gap is complete rather than approximate: chunks are written in order from
        zero, so an interrupted write leaves a prefix and never a hole.
        """
        pieces: list[str] = []
        for account in accounts:
            found = self._read_one(account)
            if found is None:
                break
            pieces.append(found)
        return "".join(pieces) if pieces else None

    def _read_one(self, account: str) -> str | None:
        found = self._run(
            ["find-generic-password", "-a", account, "-s", self._service, "-w"],
            absent_status=_NOT_FOUND,
        )
        return None if found is None else found.strip()

    # --- writing ---------------------------------------------------------------------------

    def _record_staged(
        self,
        account: str,
        generation: str,
        commits: list[_Stored],
        journal: list[str],
        base_url: str,
    ) -> list[str]:
        """Name the generation about to be written, before a byte of it is written.

        The order matters and is the only reason this is a separate step: a generation recorded
        after being staged would be invisible to :meth:`forget` in exactly the case the record
        exists for, which is a process that stopped between the two.

        A journal at :data:`MAX_STAGED` is pruned first, by deleting the generations in it that
        no pointer slot refers to. That is a repair rather than a limit: the entries being
        dropped are abandoned replacements, and dropping them without deleting them is what
        would leave cookies in the keychain that nothing can find.

        Returns:
            The generations the journal now names, including ``generation``.
        """
        if len(journal) >= MAX_STAGED:
            live = {stored.commit.generation for stored in commits}
            for orphan in journal:
                if orphan not in live:
                    self._discard(account, orphan)
            journal = [entry for entry in journal if entry in live]
        recorded = [*journal, generation]
        self._write_marker(_journal(account), " ".join(recorded), base_url)
        return recorded

    def _stage(self, account: str, generation: str, payload: str, base_url: str) -> None:
        """Write the record under a generation no reader is looking at yet."""
        for index in range(0, len(payload), CHUNK_BYTES):
            piece = payload[index : index + CHUNK_BYTES]
            self._write(_staged_chunk(account, generation, index // CHUNK_BYTES), piece, base_url)

    def _compare_staged(self, account: str, generation: str, payload: str, base_url: str) -> None:
        """Read the staged generation back and compare it byte for byte before committing it.

        Chunking works around the truncation limit that exists today; this is what turns a
        *different* limit into a loud failure rather than a quietly halved credential. It runs
        before the commit, so a keychain that gives back something else costs nothing at all.

        Raises:
            ConfigError: What came back is not what went in. The staged generation is removed
                and the session already stored is left exactly as it was.
        """
        written = self._assemble(
            _staged_chunk(account, generation, index) for index in range(MAX_CHUNKS)
        )
        if written == payload:
            return
        self._discard(account, generation)
        msg = (
            f"the macOS Keychain did not give back the session that was just written for "
            f"{base_url}, so nothing has been stored rather than something truncated, and the "
            f"session that was already stored is untouched. {SECURITY} keeps only the first "
            f"128 bytes of a secret read from stdin, which is why the record is written in "
            f"{CHUNK_BYTES}-byte pieces; a version of macOS with a smaller buffer would land "
            f"here. Put the session in the environment variable named by session_env instead."
        )
        raise ConfigError(msg)

    def _confirm(self, account: str, payload: str, base_url: str) -> None:
        """Read through the ordinary path, after the commit, and check it arrives here.

        :meth:`_compare_staged` proves the chunks landed. This proves the *pointer* resolves to
        them — that a new process following the normal route reaches this session rather than
        the one it replaced. Failing here leaves both generations in place and skips cleanup,
        so the older slot still names a session that :meth:`_current` will verify and return.

        Raises:
            ConfigError: The commit did not take. Nothing has been removed.
        """
        if self._current(account).payload == payload:
            return
        msg = (
            f"the session for {base_url} was written to the macOS Keychain and read back "
            f"correctly, but reading it the way a later run will does not reach it, so "
            f"manicule will not report it as stored. Nothing that was already stored has been "
            f"removed. Put the session in the environment variable named by session_env "
            f"instead."
        )
        raise ConfigError(msg)

    def _clean_up_quietly(
        self, account: str, *, doomed: set[str], keep: str, base_url: str
    ) -> None:
        """Remove what the new session replaced, treating failure as untidiness, not as loss.

        By the time this runs the session is stored, active and confirmed. A delete that fails
        here leaves secret material in the keychain that nothing refers to, which is worth
        saying out loud and is not worth turning a successful capture into a failed one — still
        less worth rolling the pointer back to a generation this call is in the middle of
        deleting.
        """
        try:
            self._clean_up(account, doomed=doomed, keep=keep, base_url=base_url)
        except ConfigError as exc:
            _log.warning(
                "the Confluence session for %s is stored and in use, but clearing the records "
                "it replaced did not finish (%s). What is left over is unreferenced secret "
                "material rather than a broken credential; `manicule connector login <name> "
                "--forget` removes it.",
                base_url,
                exc,
            )

    def _clean_up(self, account: str, *, doomed: set[str], keep: str, base_url: str) -> None:
        """Delete the generations the new commit replaced, and the pre-generation record.

        The journal is rewritten last, naming only ``keep``. Doing it last is deliberate: while
        the deletes are running, the journal still names everything they are working through, so
        a process that stops in the middle of cleanup leaves a journal that can still find what
        is left over.
        """
        for generation in sorted(doomed):
            self._discard(account, generation)
        self._forget_legacy(account)
        self._write_marker(_journal(account), keep, base_url)

    def _discard(self, account: str, generation: str) -> bool:
        """Delete a generation's chunks. ``True`` if there were any."""
        return self._delete_run(
            _staged_chunk(account, generation, index) for index in range(MAX_CHUNKS)
        )

    def _forget_legacy(self, account: str) -> bool:
        """Delete a pre-generation record. ``True`` if there was one."""
        return self._delete_run(_chunk(account, index) for index in range(MAX_CHUNKS))

    def _delete_run(self, accounts: Iterable[str]) -> bool:
        """Delete consecutive items, stopping at the first absent one. ``True`` if any went.

        The same reasoning as :meth:`_assemble`: chunks are written from zero upward, so a run
        of them has no holes and stopping at a gap has not left anything behind.
        """
        removed = False
        for account in accounts:
            if not self._delete(account):
                break
            removed = True
        return removed

    def _delete(self, account: str) -> bool:
        gone = self._run(
            ["delete-generic-password", "-a", account, "-s", self._service],
            absent_status=_NOT_FOUND,
        )
        return gone is not None

    def _write_marker(self, account: str, value: str, base_url: str) -> None:
        """Write a value whose whole meaning depends on arriving as one keychain item.

        A pointer or a journal that were chunked would need a pointer of their own. So they are
        single items, and a value that has outgrown one is refused here rather than stored as
        its first 128 bytes — which would read back as a slot this version cannot parse, and
        therefore as no commit at all.

        Raises:
            ConfigError: The value does not fit one item.
        """
        if len(value.encode()) > CHUNK_BYTES:
            msg = (
                f"manicule cannot record which Confluence session for {base_url} is current: "
                f"the record that says so no longer fits in one keychain item, and splitting "
                f"it would reintroduce the partial write it exists to prevent. Put the session "
                f"in the environment variable named by session_env instead."
            )
            raise ConfigError(msg)
        self._write(account, value, base_url)

    def _write(self, account: str, value: str, base_url: str) -> None:
        """Store one value under one account.

        Raises:
            ConfigError: The value contains a newline, or ``security`` refused it.
        """
        if "\n" in value:
            msg = (
                f"manicule will not write a keychain record for {base_url} containing a "
                f"newline: {SECURITY} reads a secret from stdin as newline-terminated, so the "
                f"stored value would silently be the part before it."
            )
            raise ConfigError(msg)
        self._run(
            [
                "add-generic-password",
                "-a",
                account,
                "-s",
                self._service,
                "-l",
                f"manicule — {base_url}",
                "-D",
                "manicule Confluence session",
                "-T",
                SECURITY,
                "-U",
                "-w",
            ],
            # `-w` with no value prompts twice and reads both from stdin when stdin is not
            # a terminal. Passing the value as an argument instead would put a live
            # corporate session into this process's command line.
            stdin=f"{value}\n{value}\n",
        )

    def _run(
        self, arguments: list[str], *, stdin: str = "", absent_status: int | None = None
    ) -> str | None:
        """Run ``security``, returning its stdout, or ``None`` for ``absent_status``.

        Raises:
            ConfigError: The command is unavailable or failed. Its stderr is included and its
                stdout is not, because stdout is where the secret would be.
        """
        if not self.available():
            msg = (
                f"{SECURITY} is not available on this machine, so manicule cannot use the "
                f"Keychain. Put the session cookies in the environment variable named by "
                f"session_env instead."
            )
            raise ConfigError(msg)
        try:
            completed = subprocess.run(  # noqa: S603 - absolute path, fixed argv, no shell
                [SECURITY, *arguments],
                input=stdin,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            msg = f"could not run {SECURITY}: {exc}"
            raise ConfigError(msg) from exc
        if completed.returncode == 0:
            return completed.stdout
        if absent_status is not None and completed.returncode == absent_status:
            return None
        msg = (
            f"{SECURITY} {arguments[0]} failed with status {completed.returncode}: "
            f"{completed.stderr.strip() or 'no detail given'}"
        )
        raise ConfigError(msg)


class MemoryStore:
    """A store that keeps sessions in this process. For tests, and for nothing else.

    Named and shipped rather than left in the suite because the store is the seam the capture
    flow is tested through, and a fake defined beside the tests would let the real flow drift
    from the one that is exercised.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, BrowserSession] = {}

    def describe(self) -> str:
        return "this process's memory, which does not survive it"

    def load(self, base_url: str) -> BrowserSession | None:
        return self.sessions.get(_account(base_url))

    def save(self, session: BrowserSession) -> None:
        self.sessions[_account(session.base_url)] = session

    def forget(self, base_url: str) -> bool:
        return self.sessions.pop(_account(base_url), None) is not None


def default_store() -> SessionStore:
    """The store for this machine: the Keychain on macOS, and nothing anywhere else.

    There is no file-backed fallback on purpose. A session written to a file inherits every
    question ``docs/storage.md`` §7.1 answers for the data directory and answers none of them
    better than an environment variable does, so the platforms without a keychain get the
    environment variable rather than a second-best file.
    """
    return KeychainStore()


def load_session(
    config: ConfluenceConfig,
    *,
    environ: Mapping[str, str] | None = None,
    store: SessionStore | None = None,
    now: datetime | None = None,
) -> BrowserSession | None:
    """The stored session for this instance, or the one the environment carries, or ``None``.

    The keychain is consulted first. An environment variable that shadowed a session captured a
    moment ago would make ``manicule connector login`` look as though it had not worked, and the
    variable exists for machines that have no keychain rather than as an override.

    A session taken from the environment is recorded as captured **now**, because there is
    nothing else true to say: the variable carries a cookie and no history. The consequence is
    that ``session_max_age_hours`` does not constrain it, which is the honest reading of a
    credential supplied fresh for each run.
    """
    import os  # noqa: PLC0415 - only this function reads the environment

    env = os.environ if environ is None else environ
    keychain = store if store is not None else default_store()
    if isinstance(keychain, KeychainStore) and not KeychainStore.available():
        stored = None
    else:
        stored = keychain.load(config.base_url)
    if stored is not None:
        return stored
    raw = env.get(config.session_env, "").strip()
    if not raw:
        return None
    return BrowserSession(
        base_url=config.base_url,
        account="",
        captured_at=now if now is not None else datetime.now(tz=UTC),
        cookies=parse_cookies(raw),
    )


def parse_cookies(text: str) -> dict[str, SecretStr]:
    """The cookies in a pasted ``Cookie`` header, in the order they were given.

    Accepts what a browser's developer tools actually hand over: a whole ``Cookie:`` request
    header, the header's value on its own, or several ``name=value`` pairs on separate lines.
    All three are what somebody will paste, and rejecting two of them would only teach people
    to reformat a secret in a text editor.

    Raises:
        ConfigError: There is no ``name=value`` pair in it. Almost always this means a password
            was pasted, so the message says what to do rather than what was wrong with it.
    """
    body = text.strip()
    if body.lower().startswith("cookie:"):
        body = body.split(":", 1)[1]
    cookies: dict[str, SecretStr] = {}
    for line in body.replace("\n", ";").split(";"):
        pair = line.strip()
        if not pair or "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        if name.strip() and value.strip():
            cookies[name.strip()] = SecretStr(value.strip())
    if not cookies:
        msg = (
            "that is not a session cookie: manicule expected something of the form "
            "'JSESSIONID=...; other=...', copied from a browser that is already signed in. "
            "manicule never asks for a password and cannot use one — sign in to Confluence in "
            "your browser first, then copy the Cookie header from its developer tools."
        )
        raise ConfigError(msg)
    return cookies


async def capture(
    config: ConfluenceConfig,
    cookie_text: str,
    *,
    store: SessionStore | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> BrowserSession:
    """Prove a pasted session works, then store it.

    The manual path, unchanged. Parsing is the only thing it does that
    :func:`capture_cookies` does not, and everything after the parse is that function — so a
    session captured from a paste, from a driven browser and from an imported state file are
    verified and stored by one piece of code rather than by three that could drift.

    Raises:
        ConfigError: The paste carried no cookies, or the instance does not answer the endpoint
            this asks.
        SessionExpiredError: The instance answered, and answered as somebody signed out.
    """
    return await capture_cookies(
        config,
        parse_cookies(cookie_text),
        store=store,
        transport=transport,
        now=now,
    )


async def capture_cookies(
    config: ConfluenceConfig,
    cookies: Mapping[str, SecretStr],
    *,
    store: SessionStore | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> BrowserSession:
    """Prove a set of cookies works, then store it. The only route to the credential store.

    The proof is the point. A cookie that was copied short, taken from the wrong tab, extracted
    from a state file belonging to another instance, or collected from a browser the person had
    not finished signing in to is indistinguishable from a working one until something uses it —
    and "something uses it" would otherwise be the first page of the next sync. So this makes one
    request as the session, reads back who the instance says that is, and stores nothing at all
    if the answer is anybody other than a signed-in user.

    **Verification happens before the store is touched.** A failed login leaves whatever was
    there before exactly as it was, because the write is the last thing and it only happens on
    success. That matters most for the browser flow, where a person re-authenticating a session
    that had merely aged would otherwise be able to lose a working credential by closing the
    window.

    That is half of what it takes, and this docstring used to claim it was all of it — it said
    verifying first "is what makes replacement atomic". It is not. It protects an existing
    credential from a bad *candidate* and says nothing about a bad *write*, and the store
    underneath it used to delete the old record before writing the replacement: a ``security``
    invocation that failed on the fourth of twenty-three chunks lost the working session. The
    other half lives in :class:`KeychainStore`, which stages a replacement where no reader is
    looking and publishes it with one small write.

    Raises:
        ConfigError: No cookies were given, or the instance does not answer the endpoint this
            asks, or it answers without naming a user.
        SessionExpiredError: The instance answered, and answered as somebody signed out — which
            includes the sign-in page served with status 200 that
            :mod:`~manicule.connectors.intercept` exists for.
    """
    from manicule.connectors.client import ConfluenceClient  # noqa: PLC0415 - no HTTP at import
    from manicule.connectors.credentials import BrowserSessionCredential  # noqa: PLC0415
    from manicule.connectors.errors import NotFoundError  # noqa: PLC0415

    if not cookies:
        msg = (
            f"no cookies for {config.base_url} were found, so there is nothing to verify and "
            f"nothing to store. A session with no cookies authenticates as nobody."
        )
        raise ConfigError(msg)

    moment = now if now is not None else datetime.now(tz=UTC)
    candidate = BrowserSession(
        base_url=config.base_url,
        account="",
        captured_at=moment,
        cookies=dict(cookies),
    )
    credential = BrowserSessionCredential(
        session=candidate,
        max_age=timedelta(hours=config.session_max_age_hours),
        now=lambda: moment,
    )
    client = ConfluenceClient(config, credential=credential, transport=transport)
    await client.setup()
    try:
        payload = await client.get_json(client.url(_PROBE_PATH))
    except NotFoundError as exc:
        msg = (
            f"{config.base_url}{_PROBE_PATH} does not exist on this instance, so manicule "
            f"cannot confirm who the session belongs to and will not store it. Check "
            f"base_url names the site root including any context path."
        )
        raise ConfigError(msg) from exc
    finally:
        await client.teardown()

    account = _named(payload)
    if not account:
        msg = (
            f"{config.base_url}{_PROBE_PATH} answered without naming a user, so manicule "
            f"cannot confirm the session is signed in and will not store it."
        )
        raise ConfigError(msg)

    session = BrowserSession(
        base_url=config.base_url,
        account=account,
        captured_at=moment,
        cookies=candidate.cookies,
    )
    keychain = store if store is not None else default_store()
    keychain.save(session)
    return session


async def cookies_authenticate(
    config: ConfluenceConfig,
    cookies: Mapping[str, SecretStr],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether ``cookies`` are signed in yet — the question a wait loop asks, without raising.

    :func:`capture_cookies` answers the same question by raising, which is right for a command
    that has been asked to store something and cannot. It is wrong for the browser flow's poll:
    a person who has not finished signing in yet is the *expected* state there, several times a
    second, and an exception per poll would turn the ordinary case into a stack of handled
    errors.

    So this is the same probe with the verdict as a boolean. It shares the endpoint and the
    client with :func:`capture_cookies` rather than reimplementing the check, because a poll loop
    that believed something the storing path would then refuse would hang until the timeout while
    the browser sat signed in.

    **A ``True`` here is not a decision to store.** The caller hands the cookies to
    :func:`capture_cookies`, which asks again through the same client and is the only thing that
    writes. The double check costs one request and closes the window between "signed in" and
    "stored" — a session that died in between is refused rather than persisted dead.
    """
    if not cookies:
        return False
    from manicule.connectors.errors import ConnectorError  # noqa: PLC0415

    try:
        await capture_cookies(
            config,
            cookies,
            store=_Discarding(),
            transport=transport,
            now=now,
        )
    except (ConfigError, ConnectorError):
        # Every refusal means "not signed in yet", which is what a poll wants to hear. The
        # distinction between a dead session, a sign-in page and a half-finished login matters
        # to the *final* attempt, and that one goes through `capture_cookies` and keeps its
        # message.
        return False
    return True


class _Discarding:
    """A store that keeps nothing, so the poll can reuse the verifying path without writing.

    The alternative was a ``store=None`` sentinel meaning "do not store", which is a second
    meaning for a parameter that already means "use the default" — and the failure mode of
    getting that wrong is a credential written to the keychain by a loop that was only asking.
    """

    def describe(self) -> str:  # pragma: no cover - never shown to anybody
        return "nowhere; this probe stores nothing"

    def load(self, base_url: str) -> BrowserSession | None:
        del base_url
        return None

    def save(self, session: BrowserSession) -> None:
        del session

    def forget(self, base_url: str) -> bool:
        del base_url
        return False


def _named(payload: Mapping[str, object]) -> str:
    """Who a ``user/current`` response says it is, under whichever key this version uses."""
    for key in ("username", "accountId", "userKey", "email"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _chunk(account: str, index: int) -> str:
    """The keychain account one piece of a *pre-generation* session is filed under.

    Kept because keychains outlive the software that wrote to them. Nothing writes this shape
    any more; :meth:`KeychainStore._legacy` reads it and the first successful save removes it.
    """
    return f"{account}#{index}"


def _staged_chunk(account: str, generation: str, index: int) -> str:
    """The keychain account one piece of one generation of a session is filed under.

    Two ``#``-separated tails rather than one, which is what keeps a generation's chunks from
    ever colliding with a pre-generation record: the legacy shape has a single tail and it is
    always digits, so ``<site>#<generation>#<n>`` cannot be mistaken for ``<site>#<n>`` in
    either direction.
    """
    return f"{account}#{generation}#{index}"


def _pointer(account: str, slot: str) -> str:
    """The keychain account one of the two commit slots is filed under."""
    return f"{account}#{slot}"


def _journal(account: str) -> str:
    """The keychain account the record of staged generations is filed under."""
    return f"{account}#{JOURNAL_SLOT}"


def _free_slot(commits: list[_Stored]) -> str:
    """Which pointer slot a new commit goes in: the one not currently authoritative.

    With nothing stored, either slot would do and the first is chosen so that a fresh store
    lays itself out the same way every time.
    """
    if not commits:
        return POINTER_SLOTS[0]
    return POINTER_SLOTS[1] if commits[0].slot == POINTER_SLOTS[0] else POINTER_SLOTS[0]


def _generation() -> str:
    """A fresh generation identifier. See :data:`GENERATION_HEX` for why it is random."""
    return secrets.token_hex(GENERATION_HEX // 2)


def _digest(payload: str) -> str:
    """The part of a payload's SHA-256 that a commit carries. See :data:`DIGEST_HEX`."""
    return hashlib.sha256(payload.encode()).hexdigest()[:DIGEST_HEX]


def _is_hex(value: str, length: int) -> bool:
    """Whether ``value`` is exactly ``length`` lower-case hexadecimal characters."""
    return len(value) == length and all(character in _HEX_DIGITS for character in value)


def _account(base_url: str) -> str:
    """The keychain account a site's session is filed under.

    Normalized so that ``https://wiki.example.com`` and ``https://wiki.example.com/`` are one
    entry rather than two, one of which would be found and the other silently not.
    """
    return base_url.strip().rstrip("/")
