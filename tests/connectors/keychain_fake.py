"""A deterministic stand-in for ``/usr/bin/security``, so a crash can be put where it hurts.

The Keychain cases that already exist run against the real command and are right to: whether a
cookie survives the Keychain's own encoding is not a question a fake can answer. This file
answers the other kind of question, which the real command cannot be made to answer at all —
*what does the store leave behind when a write fails half way through?* There is no way to ask
``/usr/bin/security`` to fail on the fourth of twenty-three writes, and a credential store whose
partial-failure behavior has never been observed is one whose partial-failure behavior is a
guess.

So this emulates the three subcommands the store uses, backed by a dictionary, and lets a test
say exactly where the floor gives way. Two properties make it worth trusting:

**It truncates at 128 bytes, like the real one.** ``security`` reads a secret from stdin through
a fixed buffer and silently keeps the first 128 bytes of anything longer, reporting success
either way. That is the measurement the whole chunking scheme exists for, so a fake that stored
the full string would quietly retire the guard that catches it.

**It refuses a duplicate without ``-U``, like the real one.** Status 45. A store that stopped
passing ``-U`` would otherwise pass here and fail on a machine.

The state lives in the fake rather than in any :class:`~manicule.connectors.sessions.KeychainStore`,
so a test can throw the store away and build a new one — which is how "a new reader, in a new
process" is spelled here. That distinction is the entire subject: an in-object guarantee that
does not survive the object is not the guarantee that was wanted.
"""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final
from unittest.mock import patch

from manicule.connectors import sessions

__all__ = [
    "JOURNAL_SLOT",
    "POINTER_SLOTS",
    "STDIN_BUFFER_BYTES",
    "Call",
    "FakeKeychain",
    "Fault",
    "SimulatedTermination",
    "chunk_account",
    "generation_account",
    "journal_account",
    "pointer_account",
]

STDIN_BUFFER_BYTES: Final = 128
"""What the real command keeps of a secret read from stdin. Measured on macOS 15, not assumed."""

_NOT_FOUND: Final = 44
"""``security``'s exit status for an item that is not there."""

_DUPLICATE: Final = 45
"""``security``'s exit status for adding an item that already exists without ``-U``."""

POINTER_SLOTS: Final = ("p0", "p1")
"""What the store is expected to call its two commit slots.

Named here rather than imported so that this file describes what it *observes* rather than
agreeing with the code under test by construction. ``test_the_fake_recognizes_every_record the
store writes`` is what keeps the two in step, and it does so by running a save and looking at
what actually arrived.
"""

JOURNAL_SLOT: Final = "staged"
"""What the store is expected to call its journal of staged generations.

Named here rather than imported, for the reason given in :data:`POINTER_SLOTS`.
"""


def chunk_account(site: str, index: int) -> str:
    """Where a record written before generations existed keeps one of its pieces."""
    return f"{site}#{index}"


def generation_account(site: str, generation: str, index: int) -> str:
    """Where one generation of a session keeps one of its pieces."""
    return f"{site}#{generation}#{index}"


def pointer_account(site: str, slot: str) -> str:
    """Where one of the two commit slots lives."""
    return f"{site}#{slot}"


def journal_account(site: str) -> str:
    """Where the record of staged generations lives."""
    return f"{site}#{JOURNAL_SLOT}"


class SimulatedTermination(BaseException):
    """The process being killed part-way through a replacement.

    A :class:`BaseException` on purpose. Deriving from :class:`Exception` would let any
    ``except Exception`` in the code under test turn a simulated ``SIGKILL`` into a handled
    error, and the test would then be measuring the handler rather than the crash.
    """


class Fault(Enum):
    """What goes wrong at the injected call, and how much of the call happened first."""

    FAIL = "the command exits nonzero and changes nothing"
    CRASH = "the process dies before the command changes anything"
    CRASH_AFTER_WRITE = "the command lands and then the process dies"


@dataclass(frozen=True, slots=True)
class Call:
    """One invocation of ``security``, as the fake saw it."""

    subcommand: str
    account: str
    service: str
    arguments: tuple[str, ...]

    @property
    def tail(self) -> str:
        """The account name after its last ``#``, which is what says the kind of record."""
        return self.account.rpartition("#")[2]

    @property
    def is_chunk(self) -> bool:
        """Whether this names a piece of a serialized session rather than a marker record."""
        return self.tail.isdigit()

    @property
    def is_pointer(self) -> bool:
        return self.tail in POINTER_SLOTS

    @property
    def is_journal(self) -> bool:
        return self.tail == JOURNAL_SLOT


@dataclass
class _Trigger:
    """An armed fault: which call it waits for, and what it does when it arrives."""

    subcommand: str
    kind: str
    ordinal: int
    fault: Fault | None
    keep: int | None
    seen: int = 0
    fired: bool = False

    def matches(self, call: Call) -> bool:
        if call.subcommand != self.subcommand:
            return False
        if self.kind == "chunk" and not call.is_chunk:
            return False
        if self.kind == "pointer" and not call.is_pointer:
            return False
        if self.kind == "journal" and not call.is_journal:
            return False
        self.seen += 1
        return self.seen == self.ordinal


@dataclass
class FakeKeychain:
    """A keychain that remembers what it was told and breaks exactly where it was asked to."""

    items: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])
    calls: list[Call] = field(default_factory=list[Call])
    secrets_seen: list[str] = field(default_factory=list[str])
    _trigger: _Trigger | None = None

    # --- arming a failure -----------------------------------------------------------------

    def fail_on_chunk_write(self, nth: int, *, fault: Fault = Fault.FAIL) -> None:
        """Break the ``nth`` write of a session chunk, counting from 1 at this moment."""
        self._arm("add-generic-password", "chunk", nth, fault, None)

    def fail_on_pointer_write(self, *, fault: Fault = Fault.FAIL) -> None:
        """Break the write that commits a staged generation."""
        self._arm("add-generic-password", "pointer", 1, fault, None)

    def fail_on_journal_write(self, *, fault: Fault = Fault.FAIL) -> None:
        """Break the write that records which generation is about to be staged."""
        self._arm("add-generic-password", "journal", 1, fault, None)

    def fail_on_delete(self, *, fault: Fault = Fault.FAIL) -> None:
        """Break the first delete, which is how cleanup fails after a successful commit."""
        self._arm("delete-generic-password", "any", 1, fault, None)

    def truncate_pointer_write(self, *, keep: int) -> None:
        """Let the commit land while storing only ``keep`` characters of it.

        A commit stored short parses as nothing at all, so the slot reads back as empty and the
        replacement is published to nobody. Only reading through the ordinary path after the
        commit notices, which is what makes this the case that confirmation step exists for.
        """
        self._arm("add-generic-password", "pointer", 1, None, keep)

    def truncate_chunk_write(self, nth: int, *, keep: int) -> None:
        """Let the ``nth`` chunk write succeed while storing only ``keep`` characters of it.

        This is the shape of the failure the read-back comparison exists for: a version of
        ``security`` whose stdin buffer is smaller than the chunk size stores a prefix and
        reports success, so nothing raises and the credential is silently half of one.
        """
        self._arm("add-generic-password", "chunk", nth, None, keep)

    def _arm(
        self, subcommand: str, kind: str, ordinal: int, fault: Fault | None, keep: int | None
    ) -> None:
        self._trigger = _Trigger(subcommand, kind, ordinal, fault, keep)

    def assert_fired(self) -> None:
        """Fail unless the armed fault was actually reached.

        A fault that never fires turns its test into one that asserts the *ordinary* path — and
        an ordinary path that preserves the old session is not the property being claimed. This
        is what stops a renamed record or a reordered write from hollowing out a guard silently.
        """
        assert self._trigger is not None, "no fault was armed"
        assert self._trigger.fired, (
            f"the armed {self._trigger.subcommand} fault for a "
            f"{self._trigger.kind} record never fired: the store made "
            f"{self._trigger.seen} matching calls and the fault wanted number "
            f"{self._trigger.ordinal}"
        )

    # --- what the store sees --------------------------------------------------------------

    @contextmanager
    def installed(self) -> Generator[FakeKeychain]:
        """Put this fake where :mod:`manicule.connectors.sessions` reaches for ``security``.

        ``available()`` is forced true as well, so these cases run on Linux. The Keychain is
        macOS's, but *what the store does when a write fails* is not, and a guard that only runs
        on one platform is a guard that runs on one developer's machine.
        """
        with (
            patch.object(sessions.subprocess, "run", self.run),
            patch.object(sessions.KeychainStore, "available", staticmethod(lambda: True)),
        ):
            yield self

    def run(
        self,
        command: list[str],
        *,
        input: str = "",  # noqa: A002 - subprocess.run spells it this way
        **_: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Stand in for :func:`subprocess.run` for the one command the store invokes."""
        assert command[0] == sessions.SECURITY, f"only {sessions.SECURITY} is emulated"
        subcommand = command[1]
        flags = _flags(command[2:])
        call = Call(
            subcommand=subcommand,
            account=flags.get("-a", ""),
            service=flags.get("-s", ""),
            arguments=tuple(command),
        )
        self.calls.append(call)

        fault, keep = self._verdict(call)
        if fault is Fault.CRASH:
            raise SimulatedTermination(subcommand)
        if fault is Fault.FAIL:
            return _completed(command, 1, stderr="SecKeychainItemCreateFromContent: simulated")

        result = self._apply(call, command, input, keep)
        if fault is Fault.CRASH_AFTER_WRITE:
            raise SimulatedTermination(subcommand)
        return result

    def _verdict(self, call: Call) -> tuple[Fault | None, int | None]:
        trigger = self._trigger
        if trigger is None or trigger.fired or not trigger.matches(call):
            return None, None
        trigger.fired = True
        return trigger.fault, trigger.keep

    def _apply(
        self, call: Call, command: list[str], stdin: str, keep: int | None
    ) -> subprocess.CompletedProcess[str]:
        key = (call.service, call.account)
        if call.subcommand == "add-generic-password":
            if key in self.items and "-U" not in command:
                return _completed(command, _DUPLICATE, stderr="The specified item already exists")
            self.items[key] = _stored(stdin, keep)
            self.secrets_seen.append(self.items[key])
            return _completed(command, 0)
        if call.subcommand == "find-generic-password":
            if key not in self.items:
                return _completed(
                    command, _NOT_FOUND, stderr="The specified item could not be found"
                )
            return _completed(command, 0, stdout=f"{self.items[key]}\n")
        if call.subcommand == "delete-generic-password":
            if key not in self.items:
                return _completed(
                    command, _NOT_FOUND, stderr="The specified item could not be found"
                )
            del self.items[key]
            return _completed(command, 0)
        message = f"the store invoked a subcommand this fake does not emulate: {call.subcommand}"
        raise AssertionError(message)

    # --- reading the aftermath ------------------------------------------------------------

    def accounts(self, service: str) -> list[str]:
        """Every account holding something under ``service``, in a stable order."""
        return sorted(account for held, account in self.items if held == service)

    def command_lines(self) -> str:
        """Every argument of every call, flattened, for asserting a secret is not among them."""
        return " ".join(" ".join(call.arguments) for call in self.calls)


def _flags(arguments: list[str]) -> dict[str, str]:
    """The ``-x value`` pairs in an argument list. Valueless flags map to ``""``."""
    parsed: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        word = arguments[index]
        if not word.startswith("-"):
            index += 1
            continue
        follows = arguments[index + 1] if index + 1 < len(arguments) else ""
        if follows and not follows.startswith("-"):
            parsed[word] = follows
            index += 2
        else:
            parsed[word] = ""
            index += 1
    return parsed


def _stored(stdin: str, keep: int | None) -> str:
    """What the keychain keeps of a secret offered on stdin.

    ``-w`` with no value prompts twice and reads both, so the store writes the value twice; the
    real command compares them and this one takes the first. The 128-byte ceiling is the
    measured truncation, and ``keep`` is a test asking for a smaller one.
    """
    value = stdin.split("\n", maxsplit=1)[0]
    limit = STDIN_BUFFER_BYTES if keep is None else keep
    return value.encode()[:limit].decode(errors="ignore")


def _completed(
    command: list[str], status: int, *, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=command, returncode=status, stdout=stdout, stderr=stderr
    )
