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
from pathlib import Path

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
def test_every_host_meaning_all_interfaces_is_recognised_as_such(host: str) -> None:
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
"""


def _literal_strings(path: Path) -> set[str]:
    """Every string literal in a module, whatever it is used for."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_no_module_but_the_bind_policy_names_an_all_interfaces_address() -> None:
    """A wide bind cannot arrive by a route that never asks.

    Every test above exercises :func:`resolve_bind`, and none of them would notice a server
    that passed ``"0.0.0.0"`` to a listener directly. So this reads the source: the literal
    lives in :mod:`manicule.app.bind`, where it exists to be *refused*, and nowhere else.

    Deleting this test restores a defect that no other test in the suite can see.
    """
    wide = {EVERYWHERE, "::"}
    offenders: dict[str, set[str]] = {}
    for module in sorted(SOURCE.rglob("*.py")):
        if module.name == "bind.py" and module.parent.name == "app":
            continue
        relative = str(module.relative_to(SOURCE))
        found = (_literal_strings(module) & wide) - NOT_AN_ADDRESS.get(relative, frozenset())
        if found:
            offenders[relative] = found
    assert offenders == {}, (
        f"these modules name an all-interfaces address directly: {offenders}. Every bind goes "
        f"through manicule.app.bind.resolve_bind, which refuses one that was not asked for "
        f"three separate times."
    )
