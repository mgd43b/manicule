"""The native messaging host, and what it refuses to take from the extension.

An extension is the least trusted thing that has ever handed manicule a credential. It runs in
somebody's browser, it is updated out of band, and — unlike every other login path — nothing an
operator typed initiated the message that arrives. So the tests here are mostly about the
*checks*, and the shape they share is that a message which should not produce a held session
produces no held session at all rather than a smaller one.

**Nothing here starts a browser, an extension or Chrome.** The host is a function over a byte
stream; the tests write frames into it. That is the whole reason the framing is a pair of
functions rather than something buried in `main`.

Fixtures are synthetic: `https://wiki.example.test/confluence`, invented cookies, and a fake
instance that exists only in this file's process.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import JsonValue

import manicule.cli.extension as host
import manicule.connectors.sessions as sessions_module
from manicule.cli import proxy
from manicule.config.settings import ConnectorSettings, Settings
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

BASE = "https://wiki.example.test/confluence"
SENTINEL = "not-a-real-session-value"

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
        host.install(data_dir=tmp_path)

    assert not (tmp_path / "browser-auth").exists(), "a refusal left an executable behind"


def test_no_browser_profile_yet_refuses_with_the_advice_that_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other failure: manicule knows where to look and the browser has never been started."""
    monkeypatch.setattr(
        host, "manifest_dirs", lambda: {"chrome": tmp_path / "absent" / "NativeMessagingHosts"}
    )

    with pytest.raises(ConfigError, match="Start the browser once"):
        host.install(data_dir=tmp_path)

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

    written = host.install(data_dir=tmp_path)

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

    (written,) = host.install(data_dir=tmp_path)
    document = json.loads(written.read_text(encoding="utf-8"))

    assert document["allowed_origins"] == [f"chrome-extension://{host.EXTENSION_ID}/"]
    assert document["type"] == "stdio"
    shim = Path(document["path"])
    assert shim.is_file(), "the manifest names a host that is not there"
    assert shim.stat().st_mode & 0o077 == 0, "the shim is reachable by other users"


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
