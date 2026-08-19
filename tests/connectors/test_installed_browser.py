"""Signing in through the installed Chrome, and choosing which way in a login uses.

Two subjects, and they are in one file because the property that matters spans both: **a login
must never quietly become a different login.** A provider that cannot start has to say so and
stop. It must not reach the paste prompt having silently turned into `manual_cookie`, and it must
not open the bundled Chromium that the operator's identity provider is going to refuse. Half the
tests below are therefore about what does *not* happen.

**No browser is launched here.** Discovery is a file-existence check and is exercised against a
fake filesystem; the provider itself is never asked to authenticate. The one test that would need
a real Chrome is in `test_browser_login.py`, marked, and skipped by default.

The dedicated profile is the other subject with its own section. It holds live session cookies
once signed in, so it is a credential at rest, and the tests treat it like one.

Fixtures are synthetic: `https://wiki.example.test/confluence` and invented paths throughout.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from manicule.app.service import ApplicationService, selected_provider
from manicule.config.settings import BrowserProvider, ConnectorSettings, Settings
from manicule.connectors import browser as browser_module
from manicule.connectors.browser import (
    SUPPORTED_BROWSERS,
    InstalledChromiumProvider,
    PlaywrightProvider,
    private_profile,
    resolve_browser,
)
from manicule.connectors.errors import ProviderRefusedError
from manicule.core.errors import ConfigError
from tests.app.fakes import FakeBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

BASE = "https://wiki.example.test/confluence"
P = BrowserProvider


def service_for(default_provider: str = "manual_cookie", **auth: object) -> ApplicationService:
    """A service with two connectors on one authority and a configured default provider."""
    source = ConnectorSettings(
        type="confluence",
        options={"base_url": BASE, "deployment": "server", "auth": "browser_session"},
    )
    settings = Settings(
        connectors={"handbook": source, "runbooks": source},
        authentication={  # pyright: ignore[reportArgumentType]
            "confluence": {"default_provider": default_provider, **auth}
        },
    )
    return ApplicationService(FakeBackend(settings=settings))


def choose(configured: P = P.MANUAL_COOKIE, **flags: object) -> P:
    """`selected_provider` with every flag defaulted to "not given"."""
    return selected_provider(
        browser=bool(flags.get("browser", False)),
        browser_provider=flags.get("browser_provider"),  # type: ignore[arg-type]
        browser_state=flags.get("browser_state"),  # type: ignore[arg-type]
        manual_cookie=bool(flags.get("manual_cookie", False)),
        configured=configured,
    )


# --- what a bare `connector login` does ---------------------------------------------------------


def test_an_opted_in_workspace_gets_its_configured_browser_with_no_flag() -> None:
    """Acceptance criterion 2, and the whole point of the setting.

    The operator configured `installed_chromium` once; typing the source's name is the whole
    command. A flag required here would leave the setting configuring nothing.
    """
    assert choose(P.INSTALLED_CHROMIUM) is P.INSTALLED_CHROMIUM


def test_a_workspace_that_set_nothing_still_asks_for_a_pasted_header() -> None:
    """Acceptance criterion 3. Every configuration written before this setting existed.

    The default is the paste prompt rather than a browser, so adding this section to the
    settings tree changes nothing for an installation that does not opt in — and nobody's
    `connector login` starts opening a window they did not ask for after an upgrade.
    """
    assert choose(P.MANUAL_COOKIE) is P.MANUAL_COOKIE
    assert Settings().authentication.confluence.default_provider is P.MANUAL_COOKIE


def test_the_browser_flag_means_the_configured_browser_rather_than_the_bundled_one() -> None:
    """On an opted-in workspace `--browser` must not mean "the other browser".

    Somebody who configured installed Chrome and then typed `--browser` asked for a browser, and
    the one this workspace means is theirs. Landing on bundled Chromium would open the build
    their identity provider is most likely to refuse — while looking like it had obeyed them.
    """
    assert choose(P.INSTALLED_CHROMIUM, browser=True) is P.INSTALLED_CHROMIUM


def test_the_browser_flag_still_means_bundled_chromium_where_nothing_is_configured() -> None:
    """What `--browser` has always meant, kept for the workspaces that already type it."""
    assert choose(P.MANUAL_COOKIE, browser=True) is P.BUNDLED_CHROMIUM


def test_a_browser_flag_is_never_answered_with_a_paste_prompt() -> None:
    """The silent substitution the selector exists to prevent, stated as its own test.

    A workspace whose default is a non-browser provider still gets a browser from `--browser`.
    Asking for a browser and being handed a prompt is the failure that looks like success.
    """
    for configured in (P.MANUAL_COOKIE, P.BROWSER_STATE):
        assert choose(configured, browser=True) is not P.MANUAL_COOKIE


# --- every explicit override --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"browser_provider": "installed-chromium"}, P.INSTALLED_CHROMIUM),
        ({"browser_provider": "bundled-chromium"}, P.BUNDLED_CHROMIUM),
        ({"browser_provider": "manual-cookie"}, P.MANUAL_COOKIE),
        ({"browser_provider": "browser-state"}, P.BROWSER_STATE),
        ({"browser_state": Path("state.json")}, P.BROWSER_STATE),
        ({"manual_cookie": True}, P.MANUAL_COOKIE),
    ],
    ids=["installed", "bundled", "manual-flag", "state-flag", "state-path", "manual-cookie"],
)
def test_an_explicit_override_beats_the_workspace_default(
    flags: Mapping[str, object], expected: P
) -> None:
    """Acceptance criterion 15. Each path stays reachable whatever the workspace prefers.

    Parameterized against an opted-in workspace rather than a bare one, because the override
    only means anything where there is something to override.
    """
    assert choose(P.INSTALLED_CHROMIUM, **flags) is expected


@pytest.mark.parametrize(
    "spelling", ["installed_chromium", "installed-chromium", "Installed-Chromium"]
)
def test_a_provider_is_named_the_way_a_terminal_invites(spelling: str) -> None:
    """The flag reads with hyphens and the TOML setting reads with underscores.

    Somebody who typed the other one has made no mistake worth a refusal, and a refusal here
    would be for a spelling this project itself uses in both forms.
    """
    assert choose(browser_provider=spelling) is P.INSTALLED_CHROMIUM


def test_an_unknown_provider_is_refused_and_the_refusal_lists_the_real_ones() -> None:
    """A mistyped enum whose refusal does not say what the options were costs a second run."""
    with pytest.raises(ConfigError) as refusal:
        choose(browser_provider="firefox")

    message = str(refusal.value)
    assert "firefox" in message
    for member in P:
        assert member.value.replace("_", "-") in message


# --- discovery ------------------------------------------------------------------------------


def only(name: str, path: str = "/opt/browser") -> dict[str, Path]:
    return {name: Path(path)}


def test_one_installed_browser_is_chosen_without_being_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser_module, "installed_browsers", lambda: only("chrome"))

    assert resolve_browser("") == Path("/opt/browser")


def test_several_installed_browsers_refuse_rather_than_picking_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal that matters most, because the wrong choice is invisible.

    On a machine with a work Chrome and a personal Brave, which one carries the corporate
    identity is exactly the thing the operator knows and this does not. Picking would send
    somebody through an SSO flow in the wrong browser and fail at the end of it.
    """
    monkeypatch.setattr(
        browser_module,
        "installed_browsers",
        lambda: {"chrome": Path("/a"), "brave": Path("/b")},
    )
    with pytest.raises(ProviderRefusedError) as refusal:
        resolve_browser("")

    message = str(refusal.value)
    assert "chrome" in message
    assert "brave" in message
    assert "installed_browser" in message, "the refusal must name the setting that resolves it"


def test_no_installed_browser_names_the_alternatives_rather_than_taking_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance criterion 4. Nothing is installed, and nothing else is silently used."""
    monkeypatch.setattr(browser_module, "installed_browsers", lambda: dict[str, Path]())

    with pytest.raises(ProviderRefusedError) as refusal:
        resolve_browser("")

    message = str(refusal.value)
    assert "bundled-chromium" in message
    assert "--browser-state" in message
    assert "--manual-cookie" in message


def test_a_configured_browser_that_is_not_installed_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser_module, "installed_browsers", lambda: only("chrome"))

    with pytest.raises(ProviderRefusedError, match="not installed"):
        resolve_browser("edge")


@pytest.mark.parametrize("unsupported", ["firefox", "safari"])
def test_a_browser_this_cannot_drive_is_refused_by_name(
    unsupported: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named absences rather than a generic "unknown browser".

    Playwright can drive Firefox, but a session captured there would need this whole flow
    re-proved against a different cookie jar; Safari cannot be driven with a private profile at
    all. Both are honest absences and the refusal says which alternatives exist.
    """
    monkeypatch.setattr(browser_module, "installed_browsers", lambda: only("chrome"))

    with pytest.raises(ProviderRefusedError) as refusal:
        resolve_browser(unsupported)

    assert unsupported in str(refusal.value)
    assert ", ".join(SUPPORTED_BROWSERS) in str(refusal.value)


def test_an_explicit_executable_path_is_taken_as_given(tmp_path: Path) -> None:
    """An operator with a browser somewhere unusual should not have to argue with discovery."""
    executable = tmp_path / "chrome"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    assert resolve_browser(str(executable)) == executable


def test_an_explicit_path_that_is_not_there_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProviderRefusedError, match="not a file"):
        resolve_browser(str(tmp_path / "absent" / "chrome"))


# --- the dedicated profile, which is a credential at rest -----------------------------------------


def test_a_profile_is_created_reachable_only_by_its_owner(tmp_path: Path) -> None:
    created = private_profile(tmp_path / "chrome-profile")

    assert created.is_dir()
    assert stat.S_IMODE(created.stat().st_mode) == 0o700


@pytest.mark.skipif(
    not hasattr(stat, "S_IRGRP") or __import__("sys").platform == "win32",
    reason="POSIX modes are not the access control Windows enforces",
)
def test_a_profile_others_can_reach_is_refused_rather_than_tightened(tmp_path: Path) -> None:
    """Refused rather than repaired, and the distinction is not pedantry.

    A directory others can already write to may already have been written to. Quietly
    chmod-ing it would hide that this had been true, and the operator would never learn that
    the profile holding their session had been exposed.
    """
    profile = tmp_path / "loose"
    profile.mkdir(mode=0o755)

    with pytest.raises(ProviderRefusedError) as refusal:
        private_profile(profile)

    assert "chmod 700" in str(refusal.value)


def test_a_profile_path_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    """A symlink is somewhere else, and this writes a credential into it."""
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)

    with pytest.raises(ProviderRefusedError, match="symlink"):
        private_profile(link)


def test_the_default_profile_is_manicule_s_own_and_not_the_person_s_browser_profile(
    tmp_path: Path,
) -> None:
    """Acceptance criterion 18, as a test rather than only as a sentence in the documentation.

    The whole scope boundary of this feature is that it drives the installed browser **with a
    profile of manicule's own**. A default that resolved anywhere near the ordinary Chrome
    profile would quietly turn this into the thing the specification refuses — reusing the
    operator's daily-use profile — and it would do it on the machines where discovery worked
    best.
    """
    service = service_for("installed_chromium")
    service.settings.data_dir = tmp_path

    profile = service.settings.authentication.confluence.profile_dir or (
        service.settings.data_dir / "browser-auth" / "default"
    )

    assert tmp_path in profile.parents
    for ordinary in ("Application Support/Google/Chrome", ".config/google-chrome", "User Data"):
        assert ordinary not in str(profile)


# --- the provider itself, without launching one ---------------------------------------------


def test_both_providers_are_constructed_headed(tmp_path: Path) -> None:
    """A headless browser cannot show a sign-in form to a person, on either path."""
    installed = InstalledChromiumProvider(
        executable=tmp_path / "chrome", profile_dir=tmp_path / "profile"
    )

    assert installed._headless is False  # pyright: ignore[reportPrivateUsage]
    assert PlaywrightProvider()._headless is False  # pyright: ignore[reportPrivateUsage]


def test_the_installed_provider_is_built_for_the_configured_browser_and_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, end to end, without a browser: settings in, a provider out."""
    executable = tmp_path / "chrome"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    service = service_for("installed_chromium", installed_browser=str(executable))
    service.settings.data_dir = tmp_path
    del monkeypatch

    driver = service._driver_for(P.INSTALLED_CHROMIUM)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(driver, InstalledChromiumProvider)
    assert driver._executable == executable  # pyright: ignore[reportPrivateUsage]
    assert stat.S_IMODE(driver._profile.stat().st_mode) == 0o700  # pyright: ignore[reportPrivateUsage]


def test_a_provider_that_cannot_be_built_refuses_instead_of_returning_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion 4, at the seam where a fallback would actually be written.

    `_driver_for` is the one function that could quietly hand back a working provider when the
    asked-for one cannot be built, and it would look like helpfulness. It raises instead, and
    what it must never do is return a `PlaywrightProvider`.
    """
    monkeypatch.setattr(browser_module, "installed_browsers", lambda: dict[str, Path]())
    service = service_for("installed_chromium")
    service.settings.data_dir = tmp_path

    with pytest.raises(ProviderRefusedError):
        service._driver_for(P.INSTALLED_CHROMIUM)  # pyright: ignore[reportPrivateUsage]


def test_the_bundled_provider_is_still_reachable_on_its_own_name() -> None:
    """The fallback has to keep working, or the refusal above names something that is not there."""
    service = service_for("installed_chromium")

    driver = service._driver_for(P.BUNDLED_CHROMIUM)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(driver, PlaywrightProvider)
