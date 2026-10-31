"""Passwords: how one is judged acceptable, stored and checked.

A password is the one credential here a person chooses, so it is the one a 256-bit hash of
cannot protect: an API key is random and long, which is why ``api_keys`` stores a plain
SHA-256, and a human-chosen string is neither. It is stored through scrypt, a memory-hard
function from the standard library, so that a copy of the database costs an attacker real
memory and time for every guess and the installation takes on no dependency.

**The stored form names its own cost.** ``scrypt$<n>$<r>$<p>$<salt>$<hash>``, salt and hash in
unpadded URL-safe base64. :func:`verify` reads the parameters out of the stored string rather
than using today's constants, so the cost can be raised later without locking anyone out, and
A test can build a cheap hash the same way production builds an expensive one.

**One rule for normalization, applied on set and on check.** :func:`normalize` is Unicode NFKC
and nothing else — no stripping, because a space is a character a person may have chosen. The
same function runs when a password is set and when one is presented, so a password set from the
command line and typed into a browser form compares equal whatever composed or decomposed form
either surface sent.

**Every check costs the same.** :func:`verify` against a hash that does not exist runs the same
function against :func:`decoy`, so how long a refusal takes does not say whether the handle was
a person, a person with a password, or a person who is disabled.

The function is CPU- and memory-bound, so :func:`hash_password` and :func:`verify` run it off
the event loop, behind a small semaphore: each concurrent attempt holds tens of megabytes, and
the per-address failed-authentication bucket does not bound how many addresses guess at once.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

MIN_LENGTH = 12
"""The shortest password accepted, in characters.

Length is the one property of a password that is both cheap to check and actually predicts
how long guessing takes; composition rules mostly teach people to append ``1!``.
"""

MAX_BYTES = 256
"""The longest password accepted, in UTF-8 bytes.

Bounds what a request can make the server hash. The browser form and the JSON body are both
read through a body cap well above this, so the cap here is what a person is told.
"""

SCHEME = "scrypt"

N = 2**16
R = 8
P = 1
SALT_BYTES = 16
KEY_BYTES = 32
MAXMEM = 128 * 1024 * 1024
"""OpenSSL's ceiling on the work buffer; ``128 * N * R`` is 64 MiB here, so this is headroom."""

CONCURRENCY = 2
"""How many hashes run at once in this process, so a burst cannot hold memory in proportion."""

MAX_COST = 2**20
"""The largest ``n`` :func:`verify` will honor from a stored hash.

A hash read from a database is trusted, but a corrupted or tampered row must not be able to
ask for a gigabyte of memory.
"""


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why a password was not acceptable — a sentence to show to the person who chose it."""

    message: str


def normalize(password: str) -> str:
    """The form a password is hashed and compared in. See the module docstring."""
    return unicodedata.normalize("NFKC", password)


def refusal(password: str) -> Refusal | None:
    """Why ``password`` may not be set, or ``None`` when it may.

    Checked on the normalized form, because that is what is stored: a password that is long
    only before normalization is not long.
    """
    candidate = normalize(password)
    if len(candidate) < MIN_LENGTH:
        return Refusal(f"A password must be at least {MIN_LENGTH} characters.")
    try:
        size = len(candidate.encode())
    except UnicodeEncodeError:
        # A JSON body can carry a lone surrogate, which is not text and cannot be hashed.
        return Refusal("A password must be valid text.")
    if size > MAX_BYTES:
        return Refusal(f"A password may be at most {MAX_BYTES} bytes.")
    if not candidate.strip():
        return Refusal("A password cannot be only whitespace.")
    return None


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _derive(candidate: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        candidate.encode(), salt=salt, n=n, r=r, p=p, maxmem=MAXMEM, dklen=KEY_BYTES
    )


def _encode(candidate: str) -> str:
    n, r, p = N, R, P
    salt = secrets.token_bytes(SALT_BYTES)
    key = _derive(candidate, salt, n, r, p)
    return f"{SCHEME}${n}${r}${p}${_b64(salt)}${_b64(key)}"


def _check(candidate: str, stored: str) -> bool:
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != SCHEME:  # noqa: PLR2004 - the format's field count
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, expected = _unb64(parts[4]), _unb64(parts[5])
    except ValueError:
        return False
    if not (1 < n <= MAX_COST and n & (n - 1) == 0 and 1 <= r <= 32 and 1 <= p <= 16):  # noqa: PLR2004
        return False
    try:
        actual = _derive(candidate, salt, n, r, p)
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


@lru_cache(maxsize=1)
def decoy() -> str:
    """A well-formed hash of a password nobody holds, at the current cost.

    What :func:`verify` is run against when there is no real hash, so a refusal for an unknown
    handle takes as long as a refusal for a wrong password. Built once, lazily.
    """
    return _encode(secrets.token_urlsafe(24))


_slots: asyncio.Semaphore | None = None
_slots_loop: asyncio.AbstractEventLoop | None = None


def _gate() -> asyncio.Semaphore:
    """The semaphore for the running loop.

    A semaphore binds to the loop that first uses it; a test suite and a restarted server each
    run a fresh loop in one interpreter, so the gate is rebuilt when the loop changes.
    """
    global _slots, _slots_loop  # noqa: PLW0603
    loop = asyncio.get_running_loop()
    if _slots is None or _slots_loop is not loop:
        _slots = asyncio.Semaphore(CONCURRENCY)
        _slots_loop = loop
    return _slots


async def hash_password(password: str) -> str:
    """The stored form of ``password`` at the current cost."""
    async with _gate():
        return await asyncio.to_thread(_encode, normalize(password))


async def verify(password: str, stored: str | None) -> bool:
    """Whether ``password`` matches ``stored``, taking the same time when there is no ``stored``.

    ``None`` — no such person, no password set, a disabled person — is checked against
    :func:`decoy` and always answers ``False``.
    """
    candidate = normalize(password)

    def work() -> bool:
        # The decoy is built here, in the worker, so its first construction does not stall
        # the event loop.
        return _check(candidate, stored if stored is not None else decoy())

    async with _gate():
        matched = await asyncio.to_thread(work)
    return matched and stored is not None


__all__ = [
    "MAX_BYTES",
    "MIN_LENGTH",
    "Refusal",
    "decoy",
    "hash_password",
    "normalize",
    "refusal",
    "verify",
]
