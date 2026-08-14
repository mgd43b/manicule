"""A server binds loopback, and a wide bind cannot happen by accident or by omission.

The property this file defends is narrow and absolute: **there is no sequence of defaults,
omissions or configuration edits that produces an unauthenticated listener on a routable
address.** Three separate things must be true, each of which fails safe when absent, and each
one is asserted here on its own.

The last test is the one that catches the failure nobody sees coming: a future server that
binds a literal address instead of going through :func:`~manicule.app.bind.resolve_bind`. No
amount of testing this module would notice that, so the source tree itself is checked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from manicule.app.bind import (
    EVERY_INTERFACE,
    LOOPBACK_HOSTS,
    is_every_interface,
    is_loopback,
    resolve_bind,
)
from manicule.config.settings import Settings
from manicule.core.errors import PolicyError

if TYPE_CHECKING:
    from collections.abc import Mapping

SOURCE = Path(__file__).resolve().parents[2] / "src" / "manicule"

AUTHENTICATED = {"security": {"auth": {"mode": "api_key"}}}

EVERYWHERE = "0.0.0.0"  # noqa: S104 - the address these tests exist to refuse
"""Named once, so the literal a linter objects to appears once and is explained once."""


def test_the_default_bind_is_loopback() -> None:
    """No configuration, no flags: the address reaches this machine and nowhere else."""
    bind = resolve_bind(Settings())
    assert bind.host == "127.0.0.1"
    assert bind.loopback
    assert not bind.every_interface


def test_the_settings_default_is_itself_loopback() -> None:
    """Checked at the source, not through ``resolve_bind``.

    ``resolve_bind`` could pass the test above by hard-coding loopback while the configuration
    default said something else — and then an operator reading their config file would see a
    default that is not the one in force.
    """
    assert Settings().security.transport.bind_host in LOOPBACK_HOSTS


@pytest.mark.parametrize("host", sorted(EVERY_INTERFACE))
def test_every_host_meaning_all_interfaces_is_recognized_as_such(host: str) -> None:
    """Including the empty string, which is what a blank config value looks like."""
    assert is_every_interface(host)
    assert not is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.0.2.10", "example.invalid"])  # noqa: S104 - the point is refusing these
def test_a_non_loopback_host_is_refused_without_the_explicit_flag(host: str) -> None:
    """Configuration alone cannot widen the bind.

    ``allow_public`` is a parameter no settings source can supply, so the only way to reach a
    wide bind is a person passing a flag. The refusal names the flag.
    """
    settings = Settings(security={"transport": {"bind_host": host}, **AUTHENTICATED["security"]})  # pyright: ignore[reportArgumentType]
    with pytest.raises(PolicyError) as caught:
        resolve_bind(settings)
    assert "--allow-public-bind" in str(caught.value)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10"])  # noqa: S104 - the point is refusing these
def test_a_non_loopback_host_is_refused_without_authentication(host: str) -> None:
    """The flag on its own is not enough. This is the defect the module exists to prevent."""
    settings = Settings(security={"transport": {"bind_host": host}})  # pyright: ignore[reportArgumentType]
    with pytest.raises(PolicyError) as caught:
        resolve_bind(settings, allow_public=True)
    message = str(caught.value)
    assert "security.auth.mode" in message


def test_the_refusal_lists_everything_that_is_missing_at_once() -> None:
    """Fixing one problem only to be told about the next is a poor way to spend an afternoon."""
    settings = Settings(security={"transport": {"bind_host": "0.0.0.0"}})  # pyright: ignore[reportArgumentType]  # noqa: S104 - the subject of the refusal
    with pytest.raises(PolicyError) as caught:
        resolve_bind(settings)
    message = str(caught.value)
    assert "--allow-public-bind" in message
    assert "security.auth.mode" in message


def test_a_wide_bind_is_possible_when_all_three_conditions_hold() -> None:
    """The positive control.

    Without it, a ``resolve_bind`` that refused everything would satisfy every assertion
    above — and a bind decision that can only refuse is not a bind decision.
    """
    settings = Settings(
        security={"transport": {"bind_host": "0.0.0.0"}, **AUTHENTICATED["security"]}  # pyright: ignore[reportArgumentType]  # noqa: S104 - deliberate, and the subject of this test
    )
    bind = resolve_bind(settings, allow_public=True)
    assert bind.host == EVERYWHERE
    assert not bind.loopback
    assert bind.every_interface
    assert "REACHABLE FROM THE NETWORK" in bind.describe()


def test_a_command_line_host_cannot_widen_the_bind_on_its_own() -> None:
    """A flag that named an address would otherwise be a one-word route past the policy."""
    with pytest.raises(PolicyError):
        resolve_bind(Settings(), host="0.0.0.0")  # noqa: S104 - the refusal is the assertion


def test_an_omitted_host_takes_the_configured_one_and_never_widens_it() -> None:
    """``host=None`` means "whatever is configured", and the configured default is loopback."""
    assert resolve_bind(Settings(), host=None).loopback


def test_an_out_of_range_port_is_refused_before_a_socket_exists() -> None:
    with pytest.raises(PolicyError):
        resolve_bind(Settings(), port=0)
    with pytest.raises(PolicyError):
        resolve_bind(Settings(), port=70000)


# --- the check that survives a future server ------------------------------------------------


NOT_AN_ADDRESS: dict[str, frozenset[str]] = {
    "parsers/grammars.py": frozenset({"::"}),
}
"""Literals that look like an all-interfaces address and are not one.

Enumerated per module with the reason, rather than loosening the search. ``::`` in
``parsers/grammars.py`` is C++'s scope separator, used to join a symbol path. An exemption
that has to be written down here is one a reviewer sees; a regex that stopped matching would
not be.

Keys are **posix** relative paths, so the table does not depend on which platform is running
the suite. Every entry is checked against the tree by
:func:`test_every_exemption_is_still_earning_its_place`, so one that stops applying fails
rather than sitting here forever quietly widening the search.
"""

LANDMARKS: frozenset[str] = frozenset(
    {
        "mcp/serve.py",
        "cli/main.py",
        "cli/serving.py",
        "app/runtime.py",
        "core/protocols.py",
    }
)
"""Modules the scan must have read, named individually.

A count alone can be satisfied by scanning the wrong tree. These are the files a wide bind
would actually regress in — the two that resolve an address, the two that start a server, and
one deep in the package to prove the walk recursed — plus ``core/protocols.py`` for a
directory the others do not reach. ``app/bind.py`` is deliberately absent: it is the one
module the scan skips.
"""

MINIMUM_MODULES = 50
"""A floor, far below the real count, on how many modules the scan must have read.

Present to catch a scan that collapsed — a moved test file, a changed layout, a suite running
against an installed wheel rather than the source tree — not to track the size of the package.
A number that tracked the real count would need editing every time a module was added, and a
check nobody can add a file without editing is a check people learn to edit.
"""


@dataclass(frozen=True, slots=True)
class Scan:
    """What one pass over the source tree read, and what it objected to."""

    scanned: frozenset[str]
    """Posix relative paths of every module actually parsed. The evidence the scan ran."""

    offenders: Mapping[str, frozenset[str]]


def _literal_strings(path: Path) -> set[str]:
    """Every string literal in a module, whatever it is used for."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def scan_for_wide_addresses(root: Path) -> Scan:
    """Read every module under ``root`` and report the ones naming an all-interfaces address.

    Returns what it *scanned* as well as what it found, because "found nothing" and "read
    nothing" are the same result otherwise — and the second one is the failure this whole
    check exists to not have.
    """
    wide = {EVERYWHERE, "::"}
    scanned: set[str] = set()
    offenders: dict[str, frozenset[str]] = {}
    for module in sorted(root.rglob("*.py")):
        if module.name == "bind.py" and module.parent.name == "app":
            continue
        relative = module.relative_to(root).as_posix()
        scanned.add(relative)
        found = frozenset(_literal_strings(module) & wide) - NOT_AN_ADDRESS.get(
            relative, frozenset()
        )
        if found:
            offenders[relative] = found
    return Scan(scanned=frozenset(scanned), offenders=offenders)


def test_the_source_tree_this_check_reads_is_where_it_thinks_it_is() -> None:
    """The path is computed from this file's location, so a move silently repoints it.

    Asserted on its own as well as inside the scan, because this is the failure that turns a
    security check into a green tick over an empty set.
    """
    assert SOURCE.is_dir(), (
        f"{SOURCE} is not a directory, so the scan below would read no modules and pass. "
        f"This path is derived from this test file's location — if the file moved, fix the "
        f"derivation rather than the assertion."
    )


def test_no_module_but_the_bind_policy_names_an_all_interfaces_address() -> None:
    """A wide bind cannot arrive by a route that never asks.

    Every test above exercises :func:`resolve_bind`, and none of them would notice a server
    that passed ``"0.0.0.0"`` to a listener directly. So this reads the source: the literal
    lives in :mod:`manicule.app.bind`, where it exists to be *refused*, and nowhere else.

    **The scan proves it ran before it reports what it found.** An empty result from a walk
    over nothing is indistinguishable from a clean tree, and this guards a security property —
    so the coverage assertions come first, in this test rather than beside it, and a scan that
    collapsed fails here instead of passing quietly.

    Deleting this test restores a defect that no other test in the suite can see.
    """
    assert SOURCE.is_dir(), f"{SOURCE} is not a directory; the scan would read nothing"
    scan = scan_for_wide_addresses(SOURCE)

    assert len(scan.scanned) >= MINIMUM_MODULES, (
        f"the scan read {len(scan.scanned)} module(s) under {SOURCE}, below the floor of "
        f"{MINIMUM_MODULES}. It is walking the wrong tree, and an empty walk reports success."
    )
    missing = sorted(LANDMARKS - scan.scanned)
    assert missing == [], (
        f"the scan did not read {missing}, which are modules that must exist under {SOURCE}. "
        f"Whatever it walked, it was not this package."
    )

    assert scan.offenders == {}, (
        f"these modules name an all-interfaces address directly: {dict(scan.offenders)}. "
        f"Every bind goes through manicule.app.bind.resolve_bind, which refuses one that was "
        f"not asked for three separate times."
    )


def test_every_exemption_is_still_earning_its_place() -> None:
    """An exemption that no longer applies is a hole in the search that nothing reports.

    ``NOT_AN_ADDRESS`` widens what the scan tolerates. If ``parsers/grammars.py`` stops using
    ``::`` as a scope separator, the entry keeps excusing a literal nobody is writing — and
    the day somebody writes one there for a different reason, the scan says nothing. So each
    entry has to still be true of the tree.
    """
    assert SOURCE.is_dir(), f"{SOURCE} is not a directory; there is nothing to check against"
    for relative, excused in NOT_AN_ADDRESS.items():
        module = SOURCE / relative
        assert module.is_file(), (
            f"NOT_AN_ADDRESS exempts {relative}, which does not exist. Delete the entry."
        )
        present = _literal_strings(module)
        stale = sorted(excused - present)
        assert stale == [], (
            f"NOT_AN_ADDRESS excuses {stale} in {relative}, which no longer contains "
            f"{'it' if len(stale) == 1 else 'them'}. Delete the entry: an exemption that "
            f"applies to nothing still widens the search."
        )
