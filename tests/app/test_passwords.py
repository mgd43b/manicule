"""Passwords: the stored form, the policy, and that a check costs the same whoever is asked."""

from __future__ import annotations

import pytest

from manicule.app import passwords

GOOD = "correct horse battery"


async def test_a_hash_is_self_describing_and_never_the_password() -> None:
    stored = await passwords.hash_password(GOOD)
    scheme, n, r, p, salt, key = stored.split("$")
    assert (scheme, int(r), int(p)) == ("scrypt", passwords.R, passwords.P)
    assert int(n) == passwords.N
    assert GOOD not in stored
    assert salt
    assert key


async def test_two_hashes_of_one_password_differ_and_both_verify() -> None:
    first = await passwords.hash_password(GOOD)
    second = await passwords.hash_password(GOOD)
    assert first != second
    assert await passwords.verify(GOOD, first)
    assert await passwords.verify(GOOD, second)


async def test_a_wrong_password_and_a_missing_hash_both_fail() -> None:
    stored = await passwords.hash_password(GOOD)
    assert not await passwords.verify(GOOD + "x", stored)
    assert not await passwords.verify(GOOD, None), "nothing stored is never a match"


async def test_the_password_is_judged_after_normalization() -> None:
    composed = "café au lait 123"
    decomposed = "café au lait 123"
    stored = await passwords.hash_password(composed)
    assert await passwords.verify(decomposed, stored), "NFKC makes both one password"


async def test_whitespace_is_part_of_a_password() -> None:
    stored = await passwords.hash_password("  spaces matter  ")
    assert await passwords.verify("  spaces matter  ", stored)
    assert not await passwords.verify("spaces matter", stored)


@pytest.mark.parametrize(
    ("password", "fragment"),
    [
        ("short", "at least"),
        ("x" * 11, "at least"),
        ("x" * (passwords.MAX_BYTES + 1), "at most"),
        (" " * 20, "whitespace"),
    ],
)
async def test_an_unacceptable_password_says_why(password: str, fragment: str) -> None:
    problem = passwords.refusal(password)
    assert problem is not None
    assert fragment in problem.message


async def test_an_acceptable_password_has_no_refusal() -> None:
    assert passwords.refusal(GOOD) is None


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "plain",
        "scrypt$x$8$1$AAAA$AAAA",
        "scrypt$16$8$1$AAAA",
        "bcrypt$16$8$1$AAAA$AAAA",
        f"scrypt${2**30}$8$1$AAAA$AAAA",
        "scrypt$17$8$1$AAAA$AAAA",
    ],
)
async def test_a_malformed_or_hostile_stored_hash_fails_closed_without_allocating(
    stored: str,
) -> None:
    assert not await passwords.verify(GOOD, stored)


async def test_the_production_cost_is_what_the_module_says_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    passwords.decoy.cache_clear()
    stored = await passwords.hash_password(GOOD)
    assert stored.split("$")[1] == str(2**16)
    assert await passwords.verify(GOOD, stored)
    passwords.decoy.cache_clear()


async def test_the_decoy_is_a_real_hash_nobody_can_match() -> None:
    assert passwords.decoy().startswith("scrypt$")
    assert not await passwords.verify(GOOD, None)
