"""The native messaging host the browser extension talks to.

Chrome starts this process, writes one framed message to its stdin and reads one back. It is the
whole of what the extension can reach: there is no port, no socket it listens on and no HTTP
surface, and the process exists only for the length of one hand-off.

**Why native messaging rather than a loopback endpoint.** An HTTP server on 127.0.0.1 is
reachable by every process on the machine, and `docs/surfaces.md` §4 already refuses to put
credential capture behind one — binding to loopback is not authorization. Native messaging is
mutual and needs no secret: Chrome will only start this host for an extension whose id is listed
in the host manifest, and the extension will only talk to a host it names. The operating system's
process boundary is the authorization, and neither half has a token that could leak.

**stdout is the protocol.** Chrome reads framed binary from it, so anything else written there —
a stray `print`, a warning, a progress bar — corrupts the stream and the extension sees a parse
failure rather than the thing that went wrong. Every diagnostic here goes to stderr, which Chrome
captures into the extension's log, and :func:`main` installs that discipline before it reads a
byte.

**The extension is an input, not an authority.** What arrives is a list of cookies and a URL, and
neither is trusted:

* the URL must name a Confluence connector this workspace has configured, so the extension cannot
  ask manicule to hold a session for a site the operator never named;
* the cookies are re-filtered to that authority by
  :func:`~manicule.connectors.browser.origin_cookies` — the same function the browser and
  state-file paths use — so an extension that sent its whole jar gets the same treatment as
  a browser that had one; and
* the result is proved against the instance by
  :func:`~manicule.connectors.sessions.capture_cookies` before it is handed over, so a session
  that does not authenticate is never held and a working one is never replaced by it.

No reply carries a cookie value, a length or a digest of one. The account the instance confirmed
is echoed, because that is what the operator needs to see and what every other surface already
says.
"""

from __future__ import annotations

import json
import os
import shlex
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue, ValidationError

from manicule.core.errors import ConfigError, ManiculeError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO

    from manicule.connectors.browser import CandidateCookie

__all__ = [
    "HOST_NAME",
    "MAX_COOKIES",
    "MAX_MESSAGE_BYTES",
    "extension_dir",
    "handle",
    "host_manifest",
    "install",
    "main",
    "manifest_dirs",
    "read_message",
    "write_message",
]

HOST_NAME = "com.manicule.session_handoff"
"""What the extension calls and what the host manifest is filed under. One string, because a
second copy is how the two ends come to disagree and the symptom is a port that will not open."""

EXTENSION_ID = "npogopmanhdalcajblcheekkgfnfadap"
"""The extension's id, fixed by the `key` pinned in its manifest.

An unpacked extension otherwise takes an id derived from its path, which would change when the
directory moved and silently stop Chrome from starting this host. The key is a **public** key and
is checked in for that reason: it buys a stable id without a Web Store listing."""

MAX_MESSAGE_BYTES = 1024 * 1024
"""Largest message this will read from the extension.

Chrome's own ceiling is far higher, which is the wrong bound for a message that is a few hundred
cookies at worst. The value a caller names must not decide how much memory this allocates — the
same reasoning, and roughly the same number, as
:data:`~manicule.connectors.browser.MAX_STATE_BYTES`.
"""

MAX_COOKIES = 500
"""How many cookies one message may carry.

A jar for a single origin is a handful. This is far above any real one and far below a number
that would make the filtering below expensive, and it is checked before any cookie is looked at.
"""

MAX_COOKIE_VALUE_BYTES = 8192
"""Longest single cookie value. A session cookie is tens of bytes; this bounds a hostile one."""

_LENGTH = struct.Struct("=I")
"""Chrome's framing: an unsigned 32-bit length in **native** byte order, then that many UTF-8
bytes of JSON. Native rather than little-endian because that is what the protocol specifies, and
on the one platform where it would differ a hard-coded `<I` would fail in a way that looks like a
corrupt message."""


def read_message(stream: BinaryIO) -> dict[str, JsonValue] | None:
    """One framed message, or ``None`` at end of stream.

    ``None`` rather than an exception for a clean close, because Chrome closing the port is how a
    hand-off ends normally — the extension got its answer and went away.

    Raises:
        ConfigError: The frame is malformed or larger than :data:`MAX_MESSAGE_BYTES`. The message
            never quotes the payload: it is a jar of live session cookies, and a diagnostic
            naming the offending entry would put one in a log.
    """
    header = stream.read(_LENGTH.size)
    if not header:
        return None
    if len(header) < _LENGTH.size:
        msg = "the message ended inside its length header"
        raise ConfigError(msg)
    (length,) = _LENGTH.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        # Checked before the read, so an absurd length is refused rather than allocated.
        msg = f"the message declares {length} bytes, over the {MAX_MESSAGE_BYTES}-byte limit"
        raise ConfigError(msg)
    body = stream.read(length)
    if len(body) < length:
        msg = "the message ended before the length its header declared"
        raise ConfigError(msg)
    try:
        loaded: object = json.loads(body)
    except json.JSONDecodeError as exc:
        msg = f"the message is not valid JSON ({exc.msg} at position {exc.pos})"
        raise ConfigError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"the message is a JSON {type(loaded).__name__}, not an object"
        raise ConfigError(msg)
    return cast("dict[str, JsonValue]", loaded)


def write_message(stream: BinaryIO, payload: Mapping[str, JsonValue]) -> None:
    """Frame one reply and flush it, because the extension is waiting on this read."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    stream.write(_LENGTH.pack(len(body)))
    stream.write(body)
    stream.flush()


def _candidates(entries: JsonValue) -> list[CandidateCookie]:
    """The message's cookies as :class:`~manicule.connectors.browser.CandidateCookie` records.

    Chrome's `chrome.cookies.getAll` and Playwright's `storage_state` describe a cookie with
    almost the same field names and two different ones — `expirationDate` against `expires`, and
    Chrome omits it entirely for a session cookie. Both are translated here, so that everything
    downstream sees the one record type
    :func:`~manicule.connectors.browser.origin_cookies` already knows how to filter.

    An entry that is not a usable cookie is skipped rather than raised over, as
    :func:`~manicule.connectors.browser.cookies_from_state` skips one: a single malformed row
    should not cost a good jar, and a jar of nothing usable is refused by the caller.

    Raises:
        ConfigError: There are more than :data:`MAX_COOKIES`, or one is implausibly large.
    """
    from manicule.connectors.browser import CandidateCookie  # noqa: PLC0415

    if not isinstance(entries, list):
        msg = "the message has no `cookies` array"
        raise ConfigError(msg)
    if len(entries) > MAX_COOKIES:
        msg = f"the message carries {len(entries)} cookies, over the limit of {MAX_COOKIES}"
        raise ConfigError(msg)
    found: list[CandidateCookie] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row = cast("Mapping[str, JsonValue]", entry)
        name, value = row.get("name"), row.get("value")
        domain, path = row.get("domain"), row.get("path", "/")
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        if not isinstance(domain, str) or not isinstance(path, str):
            continue
        if len(value.encode("utf-8", "ignore")) > MAX_COOKIE_VALUE_BYTES:
            msg = f"a cookie value is over the {MAX_COOKIE_VALUE_BYTES}-byte limit"
            raise ConfigError(msg)
        # Chrome omits `expirationDate` for a session cookie; Playwright spells that -1, which is
        # what `CandidateCookie` and the filter below are written against.
        expires = row.get("expirationDate")
        found.append(
            CandidateCookie(
                name=name,
                value=value,
                domain=domain,
                path=path or "/",
                expires=float(expires) if isinstance(expires, (int, float)) else -1.0,
                secure=bool(row.get("secure", False)),
            )
        )
    return found


def _configured(base_url: str, settings: Any) -> tuple[str, Any]:  # noqa: ANN401 - Settings
    """The connector this URL belongs to, refusing one the workspace never named.

    **The check that keeps the extension from choosing what manicule holds a session for.**
    Without it a compromised or merely over-eager extension could hand over a jar for any site
    and manicule would hold it, which is a credential store an outside party decides the contents
    of. Matching against configured connectors means the operator's configuration is still what
    decides, and the extension only decides *when*.

    Matched on the normalized authority rather than on the string, so the URL the extension read
    from the browser — which carries whatever spelling the person typed into Chrome — reaches the
    same connector a `connector login` would.

    Raises:
        ConfigError: No configured Confluence connector shares this authority.
    """
    from manicule.connectors.config import (  # noqa: PLC0415
        CONNECTOR_NAME,
        AuthMethod,
        ConfluenceConfig,
    )
    from manicule.connectors.sessions import authority_key  # noqa: PLC0415

    wanted = authority_key(base_url)
    for name, configured in sorted(settings.connectors.items()):
        if configured.type != CONNECTOR_NAME or not configured.enabled:
            continue
        try:
            config = ConfluenceConfig.model_validate(configured.options)
        except ValidationError:
            # Deliberately silent. Configuration that will not parse is `doctor`'s finding and
            # not this host's; reporting it here would answer "your extension cannot reach
            # manicule" with a complaint about an unrelated connector.
            continue
        if config.auth_method is not AuthMethod.BROWSER_SESSION:
            continue
        if authority_key(config.base_url) == wanted:
            return name, config
    msg = (
        "no enabled Confluence connector in this workspace is configured for that site, so "
        "there is nothing for a session to belong to. Configure one, or check that its "
        "base_url names the same site the extension read the cookies from."
    )
    raise ConfigError(msg)


async def handle(message: Mapping[str, JsonValue], **overrides: Any) -> dict[str, JsonValue]:  # noqa: ANN401 - overrides mirror Settings' own fields
    """Turn one message into a held session, and answer with what happened.

    The three steps a `--browser-state` import already takes, in the same order and through the
    same functions: decide which connector this is for, filter the jar to that authority, prove
    it against the instance and hand it over. Nothing here is a second implementation of any of
    them.

    Returns:
        An envelope with ``ok``, and on success the connector and the account the instance
        confirmed. **Never a cookie, a length or a digest** — there is nothing the extension
        needs from the value it just sent.
    """
    from manicule.cli.proxy import HandoverStore, listening  # noqa: PLC0415
    from manicule.config.loader import load_settings  # noqa: PLC0415
    from manicule.connectors.browser import origin_cookies  # noqa: PLC0415
    from manicule.connectors.sessions import capture_cookies  # noqa: PLC0415

    base_url = message.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        msg = "the message names no site"
        raise ConfigError(msg)

    settings = load_settings(**overrides)
    name, config = _configured(base_url, settings)

    served = listening(overrides)
    if served is None:
        msg = (
            "no manicule server is running, so there is nowhere to put a session. Sessions live "
            "in the server's memory; start one with `manicule serve` and send again."
        )
        raise ConfigError(msg)

    found = origin_cookies(_candidates(message.get("cookies")), base_url=config.base_url)
    if not found:
        msg = (
            "the browser sent no cookies that apply to this site. Sign in to it in Chrome "
            "first, then send again."
        )
        raise ConfigError(msg)

    session = await capture_cookies(config, found, store=HandoverStore(served))
    return {"ok": True, "connector": name, "account": session.account}


def main(argv: list[str] | None = None) -> int:
    """Serve messages on stdin until Chrome closes the port.

    Chrome starts this with the host manifest's path as ``argv[1]`` and the calling extension's
    origin as ``argv[2]``; neither is read, because neither is a thing to trust — the extension
    was already authorized by being listed in the manifest Chrome consulted to find this
    process, and re-checking a string the caller supplied would be checking the caller against
    itself.

    A refusal is an answer rather than an exit: the extension is waiting on a read, and a host
    that died would leave it with a closed port and nothing to show the person.
    """
    del argv
    # Before the first read, and the reason is in the module docstring: stdout is the protocol.
    # Anything that writes to it — a library's warning, a stray print added later — becomes a
    # corrupt frame at the far end rather than a visible mistake here.
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr

    import asyncio  # noqa: PLC0415 - kept out of import time for a process Chrome spawns

    while True:
        try:
            message = read_message(stdin)
        except ManiculeError as refusal:
            write_message(stdout, {"ok": False, "error": str(refusal)})
            return 1
        if message is None:
            return 0
        try:
            answered = asyncio.run(handle(message))
        except (ManiculeError, ValueError, OSError) as refusal:
            # `str(refusal)` and never the traceback: every message these raise is written to be
            # read by a person, and a traceback would carry frames holding the jar.
            write_message(stdout, {"ok": False, "error": str(refusal)})
            continue
        write_message(stdout, answered)


# --- installing the host manifest ---------------------------------------------------------------


def extension_dir() -> Path:
    """Where the extension's source lives on this machine, for the operator to load.

    **Inside the package rather than at the repository root**, so that it ships in the wheel.
    An extension at the root is present for somebody working in a checkout and absent for
    everybody who installed manicule normally — and the instruction "load the extension
    directory" would then name a directory most people do not have.

    Resolved rather than described for the same reason: an absolute path printed by the command
    is right on a checkout, a virtualenv and a system install alike, and none of them has to be
    explained.
    """
    return Path(__file__).parent.parent / "extension"


def manifest_dirs() -> dict[str, Path]:
    """Where each supported browser looks for native messaging host manifests.

    Per browser and per platform, because there is no shared location and no API to ask. This is
    the reason the installer is a command rather than a line in the documentation: four paths
    that differ by operating system is exactly the instruction people get wrong.
    """
    home = Path.home()
    if sys.platform == "darwin":
        support = home / "Library" / "Application Support"
        return {
            "chrome": support / "Google" / "Chrome" / "NativeMessagingHosts",
            "chromium": support / "Chromium" / "NativeMessagingHosts",
            "edge": support / "Microsoft Edge" / "NativeMessagingHosts",
            "brave": support / "BraveSoftware" / "Brave-Browser" / "NativeMessagingHosts",
        }
    if sys.platform == "win32":  # pragma: no cover - documented, not exercised here
        return {}
    config = home / ".config"
    return {
        "chrome": config / "google-chrome" / "NativeMessagingHosts",
        "chromium": config / "chromium" / "NativeMessagingHosts",
        "edge": config / "microsoft-edge" / "NativeMessagingHosts",
        "brave": config / "BraveSoftware" / "Brave-Browser" / "NativeMessagingHosts",
    }


def host_manifest(shim: Path, *, extension_id: str = EXTENSION_ID) -> dict[str, JsonValue]:
    """The document Chrome reads to decide whether to start this host, and for whom.

    ``allowed_origins`` is the authorization. Chrome starts this process for the extensions
    listed here and for nothing else, so a second extension — or a page, or another program —
    cannot reach it however much it would like to. It is the half of the pairing that lives
    outside the extension, which is why it is written by manicule rather than shipped in the
    extension directory.
    """
    return {
        "name": HOST_NAME,
        "description": "Hands a browser Confluence session to a local manicule server.",
        "path": str(shim),
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{extension_id}/"],
    }


def _write_private(path: Path, body: str, *, mode: int) -> None:
    """Write ``body`` to ``path``, at ``mode`` from the moment it exists.

    ``write_text`` followed by ``chmod`` has a window in between where the file exists at
    whatever the umask allowed — which for a shell script this user's browser will execute, and
    for the document that decides *which* executable it starts, is a window somebody else on the
    machine could write through. Opening with the mode closes it for a new file; the ``chmod``
    stays because ``O_CREAT`` does not change the mode of a file that already exists, and
    re-running the installer over a shim written before this did is exactly the case that needs
    narrowing.

    **The umask decides nothing about the result, and that is deliberate.** ``os.open`` applies
    it at creation, but the ``chmod`` then sets the mode exactly — so the file ends at ``mode``
    whichever umask the installing shell had, including one that would have cleared a bit. An
    unusual umask leaving Chrome a host it cannot execute would be a broken install, and both
    modes this is called with grant nothing whatever to group or other, so setting them exactly
    is what makes that guarantee unconditional rather than a matter of how somebody's shell was
    configured.

    **It is not a guarantee about the path, only about the file.** Neither ``os.open`` nor
    ``chmod`` is told to refuse a symbolic link, so a path that is already one is followed to
    its target — as ``write_text`` was. That is left alone rather than hardened because it buys
    nothing here: every path this is given is inside a directory manicule creates ``0700``, and
    anybody who can put a link there can replace the file outright.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(body)
    path.chmod(mode)


def _shim(directory: Path, *, config_file: Path) -> Path:
    """Write the executable Chrome starts, and return it.

    Chrome runs the ``path`` in the manifest directly and appends its own two arguments, so it
    cannot be told to run ``manicule browser-auth host``. A short shim is the usual answer and it
    also keeps the host off Typer: Click writes to stdout on a usage error, and stdout is the
    protocol.

    **It names the configuration, because Chrome cannot be relied on to.** The host is started
    by the browser, with the browser's environment and the browser's working directory — neither
    of which is the terminal the install was run from.
    :func:`~manicule.config.settings.config_file` falls back to ``manicule.toml`` beside the
    working directory and then to the user's config directory, so a host left to discover its own
    configuration finds whichever one Chrome's cwd implies. That is not a missing session or a
    bad cookie: it is a *different workspace*, and the symptom is the extension refusing a site
    the operator can see configured, with a message about no connector being configured for it.
    So ``MANICULE_CONFIG_FILE`` is written into the shim, from the path the install command had
    already resolved, and the discovery never runs.

    ``data_dir`` is deliberately **not** pinned alongside it. The control socket the host looks
    for is derived from ``data_dir``, so it might seem to want the same treatment — but pinning
    it would make an edit to the configuration file's own ``data_dir`` silently not take effect
    here, while leaving it unpinned means the file stays the one thing that decides. The
    remaining gap is narrow and worth stating: an operator who selects a data directory with
    ``$MANICULE_DATA_DIR`` in their shell rather than in the file has selected it for their
    shell, and the shim will not carry it.

    Both interpolated paths go through :func:`shlex.quote`. A path is not this function's to
    trust: a directory with a space in it used to produce a shim that ran the wrong argv, and one
    containing a quote or a ``$(...)`` would have produced a shim that ran something else
    entirely, as the user, every time Chrome started the host.
    """
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    shim = directory / "session-handoff-host"
    _write_private(
        shim,
        "#!/bin/sh\n"
        "# Written by `manicule browser-auth install`. Chrome starts this with its own\n"
        "# environment and working directory, so the configuration is named rather than found.\n"
        f"MANICULE_CONFIG_FILE={shlex.quote(str(config_file))}\n"
        "export MANICULE_CONFIG_FILE\n"
        f'exec {shlex.quote(sys.executable)} -m manicule.cli.extension "$@"\n',
        mode=0o700,
    )
    return shim


def install(
    *,
    data_dir: Path,
    config_file: Path,
    browsers: Sequence[str] | None = None,
    extension_id: str = EXTENSION_ID,
) -> list[Path]:
    """Write the host manifest for each installed browser. Returns what was written.

    Only for browsers whose manifest directory's *parent* already exists, because creating one
    would be manicule inventing a profile for a browser that is not installed — and the operator
    would then wonder why an extension they never loaded was mentioned in their filesystem.

    **Nothing is written until there is somewhere to write it.** The shim is created after the
    target list is known, so a platform this cannot serve, or a machine whose browsers have never
    been started, is refused having left no trace — rather than leaving an executable under the
    data directory that nothing will ever run.

    Args:
        data_dir: The selected workspace's data directory. The shim is written under it.
        config_file: The configuration the install command is running under, already resolved to
            an absolute path by its caller. Required rather than defaulted, because a default
            would be this function calling
            :func:`~manicule.config.settings.config_file` a second time — resolving it against
            *this* process rather than against the one that chose it, which is the same
            "discover it later" mistake the shim exists to avoid, moved one frame up.
        browsers: Which browsers to write for. ``None`` is every one with a profile.
        extension_id: The extension Chrome will start the host for.

    Raises:
        ConfigError: The platform has no known manifest locations, or no supported browser
            profile exists yet. The two are different problems with different answers, so they
            are different messages.
    """
    directories = manifest_dirs()
    if not directories:
        msg = (
            f"manicule does not know where {sys.platform} keeps native messaging host "
            f"manifests, so the browser extension cannot be connected on this machine. The "
            f"other login paths are unaffected: `manicule connector login <name>` still works."
        )
        raise ConfigError(msg)

    wanted = set(browsers) if browsers else None
    targets = [
        directory
        for name, directory in directories.items()
        if (wanted is None or name in wanted) and directory.parent.exists()
    ]
    if not targets:
        known = ", ".join(sorted(directories))
        msg = (
            f"no supported browser profile was found to install the messaging host for. "
            f"manicule looks for: {known}. Start the browser once so it creates its profile "
            f"directory, then run this again."
        )
        raise ConfigError(msg)

    shim = _shim(data_dir / "browser-auth", config_file=config_file)
    document = json.dumps(host_manifest(shim, extension_id=extension_id), indent=2)
    written: list[Path] = []
    for directory in targets:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The shim is 0700 and the directory holding it is; the document naming both is written
        # to match rather than to whatever the umask of the shell that ran the install happened
        # to be. It carries no secret — a path and an extension id — but it decides which
        # executable Chrome starts as this user, and that is worth not leaving group-writable.
        path = directory / f"{HOST_NAME}.json"
        _write_private(path, document + "\n", mode=0o600)
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover - the entry point Chrome starts
    raise SystemExit(main())
