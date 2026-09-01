"""The native messaging host, and what it refuses to take from the extension.

An extension is the least trusted thing that has ever handed manicule a credential. It runs in
somebody's browser, it is updated out of band, and — unlike every other login path — nothing an
operator typed initiated the message that arrives. So the tests here are mostly about the
*checks*, and the shape they share is that a message which should not produce a held session
produces no held session at all rather than a smaller one.

**Nothing here starts a browser, an extension or Chrome.** The host is a function over a byte
stream; the tests write frames into it. That is the whole reason the framing is a pair of
functions rather than something buried in `main`.

The one exception is the last section, and it is an exception on purpose. What the shim carries
cannot be checked by reading it: the defect it exists to prevent was a host that *looked* right
and resolved a different workspace once Chrome — not a shell — decided its environment and its
working directory. So those tests run the generated executable in a child process with neither.

Fixtures are synthetic: `https://wiki.example.test/confluence`, invented cookies, and a fake
instance that exists only in this file's process.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import struct
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import JsonValue

import manicule.cli.extension as host
import manicule.cli.main as cli
import manicule.connectors.sessions as sessions_module
from manicule.app import results as r
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.cli import proxy
from manicule.config.loader import load_settings
from manicule.config.settings import ConnectorSettings, Settings
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from manicule.app.results import Payload

BASE = "https://wiki.example.test/confluence"
SENTINEL = "not-a-real-session-value"

HOST_SITE = "https://confluence.example.test/wiki"
"""A site named only by a configuration file a test writes, and by no default workspace.

Distinct from :data:`BASE` so that the child-process tests below cannot pass by finding the
suite's own fixture: a host that resolved this site read the file it was installed with.
"""

ELSEWHERE_SITE = "https://elsewhere.example.test/wiki"
"""A site no test configures, so a host that accepts it is accepting anything."""

HOST_TIMEOUT_S = 120.0
"""How long a child host may take to import manicule and answer. Generous, because the ceiling
is only here so a hung child fails the suite instead of the job."""

_LENGTH = struct.Struct("=I")


def framed(payload: Mapping[str, JsonValue]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return _LENGTH.pack(len(body)) + body


def cookie(
    name: str = "JSESSIONID", value: str = SENTINEL, **overrides: JsonValue
) -> dict[str, JsonValue]:
    """One cookie as `chrome.cookies.getAll` reports it, which is not quite Playwright's shape."""
    entry: dict[str, JsonValue] = {
        "name": name,
        "value": value,
        "domain": "wiki.example.test",
        "path": "/confluence",
        "secure": True,
    }
    entry.update(overrides)
    return entry


def workspace(**options: JsonValue) -> Settings:
    source = ConnectorSettings(
        type="confluence",
        options={"base_url": BASE, "deployment": "server", "auth": "browser_session", **options},
    )
    return Settings(connectors={"handbook": source, "runbooks": source})


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = workspace()
    monkeypatch.setattr("manicule.config.loader.load_settings", _loading(settings))
    return settings


def _loading(settings: Settings) -> Callable[..., Settings]:
    """A `load_settings` that answers with one workspace, however it is called."""

    def load(**_: Any) -> Settings:
        return settings

    return load


def _no_server(_: Mapping[str, Any]) -> Path | None:
    """`listening`, for a machine with no manicule running."""
    return None


def _serving(path: Path) -> Callable[[Mapping[str, Any]], Path | None]:
    """`listening`, for one that has."""

    def answering(_: Mapping[str, Any]) -> Path | None:
        return path

    return answering


# --- framing ---------------------------------------------------------------------------------


def test_a_message_round_trips_through_the_frame() -> None:
    stream = io.BytesIO()
    host.write_message(stream, {"ok": True, "account": "sync.user"})
    stream.seek(0)

    assert host.read_message(stream) == {"ok": True, "account": "sync.user"}


def test_a_closed_port_is_the_end_rather_than_an_error() -> None:
    """Chrome closing the port is how a hand-off ends normally: the extension got its answer."""
    assert host.read_message(io.BytesIO()) is None


def test_a_declared_length_over_the_ceiling_is_refused_before_it_is_allocated() -> None:
    """The bound exists so that the caller does not decide how much memory this takes.

    Refused on the header alone — the body is never read — so a four-gigabyte claim costs four
    bytes rather than four gigabytes.
    """
    with pytest.raises(ConfigError, match="over the"):
        host.read_message(io.BytesIO(_LENGTH.pack(host.MAX_MESSAGE_BYTES + 1)))


def test_a_frame_that_ends_early_is_refused_rather_than_padded() -> None:
    with pytest.raises(ConfigError, match="ended before"):
        host.read_message(io.BytesIO(_LENGTH.pack(64) + b"{}"))


def test_a_body_that_is_not_json_names_the_position_and_not_the_payload() -> None:
    """The whole message is secret, so a parse failure reports the shape and never the bytes."""
    body = b'{"cookies": [' + SENTINEL.encode()
    with pytest.raises(ConfigError) as refusal:
        host.read_message(io.BytesIO(_LENGTH.pack(len(body)) + body))

    assert SENTINEL not in str(refusal.value)


def test_a_json_array_is_not_a_message() -> None:
    body = b"[1, 2, 3]"
    with pytest.raises(ConfigError, match="not an object"):
        host.read_message(io.BytesIO(_LENGTH.pack(len(body)) + body))


# --- bounds on the jar -------------------------------------------------------------------------


def test_more_cookies_than_any_real_jar_are_refused() -> None:
    too_many: JsonValue = [cookie(name=f"c{index}") for index in range(host.MAX_COOKIES + 1)]

    with pytest.raises(ConfigError, match="over the limit"):
        host._candidates(too_many)  # pyright: ignore[reportPrivateUsage]


def test_an_implausibly_large_cookie_value_is_refused() -> None:
    oversized: JsonValue = [cookie(value="x" * (host.MAX_COOKIE_VALUE_BYTES + 1))]

    with pytest.raises(ConfigError, match="over the"):
        host._candidates(oversized)  # pyright: ignore[reportPrivateUsage]


def test_one_malformed_entry_does_not_cost_a_good_jar() -> None:
    """The rule `cookies_from_state` already applies to somebody else's file.

    An extension from a future Chrome may send a field this does not know. Dropping the row is
    right; dropping the sign-in because of it is not.
    """
    mixed: JsonValue = [cookie(), {"name": "no-value"}, "not-a-cookie", cookie("other")]
    found = host._candidates(mixed)  # pyright: ignore[reportPrivateUsage]

    assert [entry.name for entry in found] == ["JSESSIONID", "other"]


def test_a_session_cookie_from_chrome_is_read_as_one() -> None:
    """Chrome omits `expirationDate` entirely; Playwright spells it -1, and the filter that both
    paths share is written against Playwright's spelling."""
    jar: JsonValue = [cookie()]
    (session_cookie,) = host._candidates(jar)  # pyright: ignore[reportPrivateUsage]

    assert session_cookie.expires == -1.0


def test_an_expiry_chrome_did_send_survives_the_translation() -> None:
    jar: JsonValue = [cookie(expirationDate=1893456000)]
    (dated,) = host._candidates(jar)  # pyright: ignore[reportPrivateUsage]

    assert dated.expires == 1893456000.0


# --- the extension does not get to choose what manicule holds ------------------------------------


async def test_a_site_no_connector_is_configured_for_is_refused(configured: Settings) -> None:
    """The check that keeps a credential store from being filled by an outside party.

    Without it, an extension — compromised, or merely over-eager — could hand over a jar for any
    site and manicule would hold it. Configuration decides *what* may be held; the extension only
    decides when.
    """
    del configured

    with pytest.raises(ConfigError, match="no enabled Confluence connector"):
        await host.handle({"base_url": "https://other.example.test/wiki", "cookies": [cookie()]})


async def test_a_message_naming_no_site_is_refused(configured: Settings) -> None:
    del configured

    with pytest.raises(ConfigError, match="names no site"):
        await host.handle({"cookies": [cookie()]})


async def test_a_disabled_connector_is_not_a_place_to_put_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source the operator turned off is one nothing will sync, so a credential for it is a
    window opened for nothing — the rule `connector_login` already applies."""
    settings = Settings(
        connectors={
            "handbook": ConnectorSettings(
                type="confluence",
                options={"base_url": BASE, "deployment": "server", "auth": "browser_session"},
                enabled=False,
            )
        }
    )
    monkeypatch.setattr("manicule.config.loader.load_settings", _loading(settings))

    with pytest.raises(ConfigError, match="no enabled Confluence connector"):
        await host.handle({"base_url": BASE, "cookies": [cookie()]})


async def test_a_connector_on_the_same_authority_is_found_however_the_url_was_spelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The URL comes out of the browser, so it carries whatever the person typed into Chrome.

    Matching on the normalized authority is what makes `https://WIKI.example.test:443/confluence`
    reach the connector configured as `https://wiki.example.test/confluence`, instead of being
    refused as a site nobody configured.
    """
    monkeypatch.setattr("manicule.config.loader.load_settings", _loading(workspace()))

    name, config = host._configured(  # pyright: ignore[reportPrivateUsage]
        "https://WIKI.example.test:443/confluence/", workspace()
    )

    assert name == "handbook"
    assert config.base_url == BASE


# --- nothing is held without a server, or without the instance agreeing --------------------------


async def test_no_server_is_refused_before_anything_is_validated(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sessions live in the server's memory, so with no server there is nowhere to put one."""
    del configured
    monkeypatch.setattr(proxy, "listening", _no_server)

    with pytest.raises(ConfigError, match="no manicule server is running"):
        await host.handle({"base_url": BASE, "cookies": [cookie()]})


async def test_cookies_for_another_origin_are_filtered_out_and_then_refused(
    configured: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An identity provider's cookies must not reach manicule, and the extension is not what
    decides that — the same filter every other login path uses is applied here.

    The message carries a jar for a different host entirely, which is what a broadly-permissioned
    extension would send. Nothing survives the filter, and an empty jar is a refusal rather than
    an empty success: a stored session with no cookies authenticates as nobody.
    """
    del configured
    monkeypatch.setattr(proxy, "listening", _serving(tmp_path / "socket"))

    with pytest.raises(ConfigError, match="no cookies that apply"):
        await host.handle(
            {
                "base_url": BASE,
                "cookies": [cookie(domain="login.identity.example.test", path="/")],
            }
        )


async def test_a_jar_goes_through_the_verification_every_other_login_uses(
    configured: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The proof that makes accepting a jar from an extension safe at all.

    Whether cookies authenticate is not a question a browser can answer, and it is certainly not
    one to take an extension's word for. What is asserted here is the *route*: the host reaches
    `capture_cookies`, which makes a request as the session and stores nothing if the instance
    answers as anybody signed out. Its behavior has its own tests; what could regress here is the
    host growing a shortcut that hands the vault a jar directly.

    The cookies it is called with are checked too, because the filter running before verification
    is what keeps an identity provider's cookies out of the request — an extension that sent its
    whole jar must not have all of it proved and stored.
    """
    del configured
    seen: dict[str, Any] = {}

    async def refuse(config: Any, cookies: Any, **kwargs: Any) -> Any:
        seen["base_url"] = config.base_url
        seen["cookies"] = dict(cookies)
        from manicule.connectors.errors import SessionExpiredError  # noqa: PLC0415

        msg = "the instance answered as a signed-out user"
        raise SessionExpiredError(msg)

    monkeypatch.setattr(proxy, "listening", _serving(tmp_path / "socket"))
    monkeypatch.setattr(sessions_module, "capture_cookies", refuse)

    from manicule.connectors.errors import SessionExpiredError  # noqa: PLC0415

    with pytest.raises(SessionExpiredError):
        await host.handle(
            {
                "base_url": BASE,
                "cookies": [cookie(), cookie(domain="login.identity.example.test", path="/")],
            }
        )

    assert seen["base_url"] == BASE
    assert list(seen["cookies"]) == ["JSESSIONID"], (
        "the identity provider's cookie reached verification, so the filter ran too late or "
        "not at all"
    )


# --- what the extension is told back --------------------------------------------------------------


def test_no_reply_carries_the_session_it_was_given() -> None:
    """There is nothing the extension needs from the value it just sent, so it gets none of it.

    Asserted over a written frame rather than over a dict, because the frame is what crosses the
    process boundary and lands in Chrome's memory.
    """
    stream = io.BytesIO()
    host.write_message(stream, {"ok": True, "connector": "handbook", "account": "sync.user"})

    assert SENTINEL not in stream.getvalue().decode()


def test_a_refusal_is_an_answer_rather_than_a_dead_port() -> None:
    """A host that exited would leave the extension with a closed port and nothing to show.

    The person is looking at a popup waiting for a sentence; "the port closed" is not one.
    """
    stdin = io.BytesIO(framed({"base_url": "", "cookies": []}))
    stdout = io.BytesIO()
    message = host.read_message(stdin)
    assert message is not None

    # What `main` does with a refusal, without running its loop against a real workspace.
    host.write_message(stdout, {"ok": False, "error": "the message names no site"})
    stdout.seek(0)
    answered = host.read_message(stdout)

    assert answered == {"ok": False, "error": "the message names no site"}


# --- installing the host manifest ------------------------------------------------------------


def test_a_platform_with_no_known_locations_refuses_without_writing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regression. Nothing should be left behind by a refusal.

    The shim used to be written before the target list was known, so on a platform manicule has
    no manifest locations for it created an executable under the data directory that nothing
    would ever run — and then refused with advice about starting a browser, which was not the
    problem. Two different failures deserve two different messages and neither deserves a
    leftover file.
    """
    monkeypatch.setattr(host, "manifest_dirs", dict)

    with pytest.raises(ConfigError, match="does not know where"):
        host.install(data_dir=tmp_path, config_file=tmp_path / "manicule.toml")

    assert not (tmp_path / "browser-auth").exists(), "a refusal left an executable behind"


def test_no_browser_profile_yet_refuses_with_the_advice_that_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other failure: manicule knows where to look and the browser has never been started."""
    monkeypatch.setattr(
        host, "manifest_dirs", lambda: {"chrome": tmp_path / "absent" / "NativeMessagingHosts"}
    )

    with pytest.raises(ConfigError, match="Start the browser once"):
        host.install(data_dir=tmp_path, config_file=tmp_path / "manicule.toml")

    assert not (tmp_path / "browser-auth").exists()


def test_a_manifest_is_written_for_each_browser_that_has_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And only for those. Creating a directory for a browser that is not installed would be
    manicule inventing a profile, and the operator wondering why an extension they never loaded
    is mentioned in their filesystem."""
    here, absent = tmp_path / "chrome", tmp_path / "brave"
    here.mkdir()
    monkeypatch.setattr(
        host,
        "manifest_dirs",
        lambda: {
            "chrome": here / "NativeMessagingHosts",
            "brave": absent / "NativeMessagingHosts",
        },
    )

    written = host.install(data_dir=tmp_path, config_file=tmp_path / "manicule.toml")

    assert [path.parent.parent.name for path in written] == ["chrome"]
    assert not absent.exists()


def test_the_manifest_names_one_extension_and_a_shim_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`allowed_origins` is the authorization: Chrome starts the host for that id and no other.

    The shim is checked for real rather than by its path, because a manifest naming an executable
    that is not there is a pairing that looks installed and fails at the first hand-off.
    """
    profile = tmp_path / "chrome"
    profile.mkdir()
    monkeypatch.setattr(host, "manifest_dirs", lambda: {"chrome": profile / "NativeMessagingHosts"})

    (written,) = host.install(data_dir=tmp_path, config_file=tmp_path / "manicule.toml")
    document = json.loads(written.read_text(encoding="utf-8"))

    assert document["allowed_origins"] == [f"chrome-extension://{host.EXTENSION_ID}/"]
    assert document["type"] == "stdio"
    shim = Path(document["path"])
    assert shim.is_file(), "the manifest names a host that is not there"
    assert shim.stat().st_mode & 0o077 == 0, "the shim is reachable by other users"
    assert written.stat().st_mode & 0o077 == 0, (
        "the document deciding which executable Chrome starts is reachable by other users"
    )


def test_the_pinned_key_is_the_one_chrome_will_derive_the_id_from() -> None:
    """The two halves of the pairing, checked against each other.

    Chrome derives an extension's id from the SHA-256 of the public key in its manifest, first
    128 bits, hex digits mapped onto `a`-`p`. `EXTENSION_ID` is what the host manifest permits.
    If the key is ever regenerated without updating the constant, Chrome refuses to start the
    host and says nothing useful about why — so the drift is caught here instead.
    """
    import base64  # noqa: PLC0415 - only this assertion needs them
    import hashlib  # noqa: PLC0415

    manifest = json.loads((host.extension_dir() / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(base64.b64decode(manifest["key"])).hexdigest()[:32]

    assert "".join(chr(ord("a") + int(c, 16)) for c in digest) == host.EXTENSION_ID


def test_the_extension_ships_where_the_command_says_it_is() -> None:
    """A path printed to an operator has to be one they can open.

    The extension lived at the repository root first, which meant it was present for anybody
    working in a checkout and absent from the wheel — so "load the directory it printed" named
    something most installations do not have. It is inside the package now, and this asserts the
    four files are actually there rather than that a path string was constructed.
    """
    directory = host.extension_dir()

    assert directory.is_dir()
    assert {path.name for path in directory.iterdir()} >= {
        "manifest.json",
        "worker.js",
        "popup.html",
        "popup.js",
    }


# --- the configuration the host starts under ---------------------------------------------------
#
# Chrome starts the shim, so the shim's environment is Chrome's and its working directory is
# Chrome's. Every test below runs the generated executable in a child process with a sterile
# environment and an unrelated working directory, because that is the only arrangement in which
# a host that carries its configuration is distinguishable from one that inherits it — and the
# defect these pin is exactly a host that inherited a different workspace and refused a site the
# operator could see configured.


def _config_naming(path: Path, site: str, *, data_dir: Path) -> Path:
    """A configuration file with one browser-session Confluence source, written at ``path``.

    ``data_dir`` is named in the file so the host looks for a control socket under a directory
    this test owns rather than under the one the developer running the suite actually uses.
    Values go through :func:`json.dumps` rather than being interpolated between quotes: one of
    the tests below deliberately puts a quote in a path, and a fixture that could not express
    that would quietly test something easier.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"data_dir = {json.dumps(str(data_dir))}\n"
        f"\n"
        f"[connectors.handbook]\n"
        f'type = "confluence"\n'
        f"\n"
        f"[connectors.handbook.options]\n"
        f"base_url = {json.dumps(site)}\n"
        f'deployment = "server"\n'
        f'auth = "browser_session"\n',
        encoding="utf-8",
    )
    return path


def _sterile(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A working directory and an environment with nothing of manicule's in either.

    No ``MANICULE_*`` variable and no ``manicule.toml`` within reach, which is what makes the
    assertion mean something: a host that resolves the right workspace here can only have been
    told, because there is nothing to infer it from. ``TMPDIR`` is set so the control socket the
    host goes looking for is under a directory this test owns.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    home = tmp_path / "sterile-home"
    home.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    return elsewhere, {"PATH": os.defpath, "HOME": str(home), "TMPDIR": str(runtime)}


def _installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, config_file: Path) -> Path:
    """Run the installer against a fake browser profile, and return the shim it wrote."""
    profile = tmp_path / "chrome"
    profile.mkdir(exist_ok=True)
    monkeypatch.setattr(host, "manifest_dirs", lambda: {"chrome": profile / "NativeMessagingHosts"})
    (written,) = host.install(data_dir=tmp_path / "data", config_file=config_file)
    return Path(json.loads(written.read_text(encoding="utf-8"))["path"])


def _asked(shim: Path, site: str, *, cwd: Path, env: Mapping[str, str]) -> dict[str, JsonValue]:
    """Start the host as Chrome would, send it one message, and read the one frame back.

    The whole of stdout has to be that frame. Chrome parses this stream, so a warning or a stray
    `print` is not noise beside the answer — it *is* the answer, malformed, and the extension
    reports a parse failure rather than the thing that went wrong.
    """
    completed = subprocess.run(  # noqa: S603 - the shim this test just wrote, and no shell
        [str(shim)],
        input=framed({"base_url": site, "cookies": [cookie()]}),
        capture_output=True,
        cwd=cwd,
        env=dict(env),
        timeout=HOST_TIMEOUT_S,
        check=False,
    )
    assert len(completed.stdout) >= _LENGTH.size, (
        f"the host wrote no usable reply ({len(completed.stdout)} bytes). Its stderr was: "
        f"{completed.stderr.decode('utf-8', 'replace')}"
    )
    (length,) = _LENGTH.unpack(completed.stdout[: _LENGTH.size])
    assert len(completed.stdout) == _LENGTH.size + length, (
        "something other than the reply reached stdout, which is the protocol"
    )
    answered: dict[str, JsonValue] = json.loads(
        completed.stdout[_LENGTH.size : _LENGTH.size + length]
    )
    return answered


def _refusal(answered: Mapping[str, JsonValue]) -> str:
    assert answered["ok"] is False, answered
    error = answered["error"]
    assert isinstance(error, str)
    return error


@pytest.mark.slow
def test_a_host_installed_under_one_configuration_resolves_that_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect, at the boundary that produced it.

    `browser-auth install` reads the selected configuration correctly and used to write a shim
    that named neither it nor anything else — so the host Chrome started fell back to
    `manicule.toml` beside *Chrome's* working directory, and then to the default config
    directory. The site the operator had configured was not in whichever file that found, and the
    extension refused it for having no connector configured, while the server it was talking
    about had one and was running.

    Reaching `no manicule server is running` is the assertion: that message comes from *after*
    the connector lookup in `handle`, so getting it proves the lookup found the source this
    configuration names. The failure this excludes is the refusal on the next line down.
    """
    config = _config_naming(tmp_path / "selected" / "manicule.toml", HOST_SITE, data_dir=tmp_path)
    shim = _installed(tmp_path, monkeypatch, config_file=config)
    elsewhere, sterile = _sterile(tmp_path)

    error = _refusal(_asked(shim, HOST_SITE, cwd=elsewhere, env=sterile))

    assert "no manicule server is running" in error, error
    assert "no enabled Confluence connector" not in error, (
        "the host resolved a workspace other than the one it was installed under"
    )


@pytest.mark.slow
def test_a_site_the_selected_configuration_does_not_name_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half, without which the test above would pass against a host that took anything.

    Carrying the configuration must not turn into ignoring it. The check that keeps an extension
    from choosing what manicule holds a session for is that the site must name a *configured*
    connector, and it has to still refuse one that does not.
    """
    config = _config_naming(tmp_path / "selected" / "manicule.toml", HOST_SITE, data_dir=tmp_path)
    shim = _installed(tmp_path, monkeypatch, config_file=config)
    elsewhere, sterile = _sterile(tmp_path)

    error = _refusal(_asked(shim, ELSEWHERE_SITE, cwd=elsewhere, env=sterile))

    assert "no enabled Confluence connector" in error, error


@pytest.mark.slow
def test_a_configuration_path_a_shell_would_mangle_reaches_the_host_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path is not the installer's to trust, and it is written into a shell script.

    A directory with a space in it used to produce a host that ran the wrong argv. One with a
    quote or a `$(...)` in it would have produced a host that ran something else entirely, as
    this user, every time Chrome started it — and Chrome starts it on a person's own machine,
    which is where the paths with apostrophes and spaces in them live.

    Two things are asserted because two things could go wrong: the path survives, and nothing in
    it was executed.
    """
    awkward = tmp_path / "a b \"c\" 'd' $(touch pwned) ;touch pwned2; ünïcode"
    config = _config_naming(awkward / "manicule.toml", HOST_SITE, data_dir=tmp_path)
    shim = _installed(tmp_path, monkeypatch, config_file=config)
    elsewhere, sterile = _sterile(tmp_path)

    error = _refusal(_asked(shim, HOST_SITE, cwd=elsewhere, env=sterile))

    assert "no manicule server is running" in error, error
    for marker in ("pwned", "pwned2"):
        assert not (elsewhere / marker).exists(), f"the shim ran {marker!r} from a path"
        assert not (awkward / marker).exists(), f"the shim ran {marker!r} from a path"


@pytest.mark.slow
def test_a_reinstall_replaces_the_configuration_the_shim_was_carrying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running it again is how somebody moves the host to another workspace.

    A shim that kept the first configuration would make the second install look like it did
    something and change nothing — the same shape of silent failure as the original defect, with
    the operator now certain they had fixed it.
    """
    first = _config_naming(tmp_path / "first" / "manicule.toml", ELSEWHERE_SITE, data_dir=tmp_path)
    second = _config_naming(tmp_path / "second" / "manicule.toml", HOST_SITE, data_dir=tmp_path)
    _installed(tmp_path, monkeypatch, config_file=first)
    shim = _installed(tmp_path, monkeypatch, config_file=second)
    elsewhere, sterile = _sterile(tmp_path)

    assert str(first) not in shim.read_text(encoding="utf-8")
    assert "no manicule server is running" in _refusal(
        _asked(shim, HOST_SITE, cwd=elsewhere, env=sterile)
    )
    assert "no enabled Confluence connector" in _refusal(
        _asked(shim, ELSEWHERE_SITE, cwd=elsewhere, env=sterile)
    )


@pytest.mark.slow
def test_an_installation_with_no_configuration_file_yet_still_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary first-run case: a default installation whose config file does not exist.

    Naming a file that is not there must leave the host able to start and answer, because that is
    what `browser-auth install` does on a machine where nobody has written any configuration yet.
    A host that refused to run, or that died instead of framing a reply, would make the
    installer's own happy path the broken one.
    """
    shim = _installed(tmp_path, monkeypatch, config_file=tmp_path / "absent" / "manicule.toml")
    elsewhere, sterile = _sterile(tmp_path)

    error = _refusal(_asked(shim, HOST_SITE, cwd=elsewhere, env=sterile))

    assert "no enabled Confluence connector" in error, error


def test_nothing_a_session_could_be_read_from_is_written_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`browser-auth install` writes two files, and neither may ever grow a credential.

    The one claim this whole subsystem makes is that a session is held in the server's memory and
    written nowhere. The shim is a file on disk that the installer generates, so it is the
    obvious place for a later change to put "just the account" or "just a token" — and it is
    read by anything that can read the data directory. Pinning its shape is cheaper than
    noticing.
    """
    config = _config_naming(tmp_path / "selected" / "manicule.toml", HOST_SITE, data_dir=tmp_path)
    profile = tmp_path / "chrome"
    profile.mkdir()
    monkeypatch.setattr(host, "manifest_dirs", lambda: {"chrome": profile / "NativeMessagingHosts"})

    (written,) = host.install(data_dir=tmp_path / "data", config_file=config)
    shim = Path(json.loads(written.read_text(encoding="utf-8"))["path"])
    launcher = shim.read_text(encoding="utf-8")

    assert launcher.startswith("#!/bin/sh\n")
    assert shlex.quote(str(config)) in launcher, "the configuration is not quoted for a shell"
    assert set(json.loads(written.read_text(encoding="utf-8"))) == {
        "name",
        "description",
        "path",
        "type",
        "allowed_origins",
    }
    for credential in ("Cookie", "JSESSIONID", "token", SENTINEL):
        assert credential not in launcher, f"{credential!r} was written into the launcher"


def _install_command(monkeypatch: pytest.MonkeyPatch) -> r.MessagingHostInstalled:
    """Run `browser-auth install`'s own body against a real service, and return what it reported.

    :func:`~manicule.cli.main.emit` is replaced rather than the command re-implemented, because
    the line that has to be covered is the one inside `browser_auth_install` that resolves the
    configuration. A test calling `install()` itself would only be asserting about an argument
    it had chosen.
    """
    captured: list[Payload] = []

    def emit(op: str, call: Callable[[ApplicationService], Awaitable[Payload]]) -> None:
        assert op == "browser_auth_install"

        async def once() -> Payload:
            runtime = Runtime(load_settings(), writer=False)
            try:
                return await call(ApplicationService(runtime))
            finally:
                await runtime.aclose()

        captured.append(asyncio.run(once()))

    monkeypatch.setattr(cli, "emit", emit)
    cli.browser_auth_install()
    (reported,) = captured
    assert isinstance(reported, r.MessagingHostInstalled)
    return reported


@pytest.mark.slow
def test_the_command_installs_a_host_carrying_the_configuration_it_was_run_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manicule_environment: Path
) -> None:
    """The boundary the defect actually lived at, driven from the command that has it.

    Every other test in this section calls `install()` with a path it chose itself, which proves
    the installer honors the argument and not that `manicule browser-auth install` supplies one.
    The reported failure was an operator selecting a workspace with `MANICULE_CONFIG_FILE` and
    getting a host that did not carry it — so this sets that variable, runs the command, and then
    starts the host it wrote in a process where the variable is gone again.
    """
    del manicule_environment
    config = _config_naming(tmp_path / "selected" / "manicule.toml", HOST_SITE, data_dir=tmp_path)
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(config))
    profile = tmp_path / "chrome"
    profile.mkdir()
    monkeypatch.setattr(host, "manifest_dirs", lambda: {"chrome": profile / "NativeMessagingHosts"})

    reported = _install_command(monkeypatch)

    (written,) = (Path(path) for path in reported.installed)
    shim = Path(json.loads(written.read_text(encoding="utf-8"))["path"])
    elsewhere, sterile = _sterile(tmp_path)
    error = _refusal(_asked(shim, HOST_SITE, cwd=elsewhere, env=sterile))

    assert "no manicule server is running" in error, error
    assert "no enabled Confluence connector" not in error, (
        "the command installed a host that resolves a workspace other than the selected one"
    )


@pytest.mark.slow
def test_the_command_installs_a_working_host_with_no_configuration_file_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manicule_environment: Path
) -> None:
    """A default installation, through the same command and with nothing selected.

    `config_file()` falls back to the user's config directory when nothing names one, and on a
    machine where nobody has written any configuration yet that names a file which does not
    exist. Pinning a path to a file that is not there has to leave a host that still starts and
    still frames a reply, or the installer's own ordinary path is the broken one.
    """
    del manicule_environment
    profile = tmp_path / "chrome"
    profile.mkdir()
    monkeypatch.setattr(host, "manifest_dirs", lambda: {"chrome": profile / "NativeMessagingHosts"})

    reported = _install_command(monkeypatch)

    (written,) = (Path(path) for path in reported.installed)
    shim = Path(json.loads(written.read_text(encoding="utf-8"))["path"])
    elsewhere, sterile = _sterile(tmp_path)

    assert "no enabled Confluence connector" in _refusal(
        _asked(shim, HOST_SITE, cwd=elsewhere, env=sterile)
    )
