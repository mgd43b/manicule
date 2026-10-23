"""Minting a Gmail refresh token: Google's installed-app flow, and where the result goes.

``manicule connector login <name>`` for a Gmail instance has one job — hand a person a refresh
token for the mailbox the instance indexes, so they can put it in the secret store a headless
server reads it from (``docs/connectors/gmail.md`` §1.2, §1.3). This module is the whole of it
apart from the command-line plumbing.

**The flow is the one Google documents for desktop applications**, and each choice in it is a
refusal of a weaker alternative:

- *A loopback redirect on ``127.0.0.1`` with an ephemeral port.* The authorization code comes
  back to a socket this process opened, on this machine, for this sign-in. Nothing is pasted, so
  nothing is typed into a terminal, and the code never passes through a page manicule does not
  control.
- *PKCE with ``S256``.* The code is useless without the verifier, which never leaves this
  process, so another local program that sees the redirect cannot redeem it.
- *``state`` is checked*, in constant time, before the code is looked at, so a redirect this
  sign-in did not start is refused rather than exchanged.
- *Only ``gmail.readonly``*, with ``access_type=offline`` and ``prompt=consent`` — the pair that
  makes Google issue a refresh token every time rather than only the first. A token Google
  reports as carrying any other scope is refused: it is a larger secret than the connector
  needs.
- *The browser does the sign-in.* manicule never asks for a password, a code or a second
  factor; it opens an address and listens.

**The token is a secret from the moment Google sends it.** It is held as a
:class:`~pydantic.SecretStr`, it appears in no message, log record or result, and the only
places it can go are a file created ``0600`` and a standard output that is not a terminal
(:class:`FileSink`, :class:`StreamSink`). Where it goes is checked *before* the browser opens,
so a sign-in is never completed only to find nowhere safe to put the result.

Nothing here imports an HTTP client at module level, for the reason
:mod:`manicule.connectors.credentials` gives.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import importlib.util
import logging
import os
import re
import secrets
import threading
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol, TextIO, cast
from urllib.parse import parse_qs, urlencode, urlsplit

from pydantic import SecretStr

from manicule.connectors.errors import (
    ConnectorError,
    CredentialError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
)
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    import httpx

    from manicule.connectors.config import GmailConfig

_log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SIGN_IN_TIMEOUT_SECONDS",
    "GMAIL_AUTH_URL",
    "GMAIL_READONLY_SCOPE",
    "FileSink",
    "StreamSink",
    "TokenSink",
    "mint_refresh_token",
]

GMAIL_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
"""Where the person is sent to sign in. A constant for the reason the token endpoint is one."""

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
"""The one scope requested. It cannot send, modify or delete mail (``gmail.md`` §1)."""

DEFAULT_SIGN_IN_TIMEOUT_SECONDS = 300.0
"""How long to wait for the person to finish signing in: long enough for a second factor."""

_CONNECTION_TIMEOUT_SECONDS = 10.0
"""How long one connection to the loopback socket may take to send its request.

A browser that opens a connection and sends nothing (speculative preconnect) must not hold the
listener, or the shutdown that waits for it, for the length of the sign-in.
"""

_REQUEST_HEAD_LIMIT = 16_384
"""The most request-line-and-headers the loopback socket reads. A redirect is a few hundred."""

_ERROR_CODE = re.compile(r"[a-z_]{1,40}")
"""What an OAuth ``error`` code looks like. Anything else is reported as ``unknown``."""

_PAGE = {
    "done": "Signed in. You can close this tab and return to the terminal.",
    "failed": "The sign-in could not be completed. Return to the terminal for the reason.",
}
"""The only two pages the loopback socket serves. Static, so nothing the request carried is
reflected into a page."""


class TokenSink(Protocol):
    """Where a freshly minted refresh token is put, and the checks that make that safe."""

    def describe(self) -> str:
        """Where the token goes, in words fit for the result. Never the token."""
        ...

    def check(self) -> None:
        """Refuse now if the token could not be put here, before anybody signs in.

        Raises:
            ConfigError: The destination is not one a secret may go to, or cannot be written.
        """
        ...

    def write(self, token: SecretStr) -> None:
        """Put the token here. Called once, after the token has been verified."""
        ...


@dataclass(frozen=True, slots=True)
class FileSink:
    """A file created ``0600`` for the token, and nothing else in the file.

    **Created, not truncated.** The file is opened with ``O_EXCL``, so it is a new file this
    process made with the final mode from its first byte: there is no moment at which the token
    sits in a file somebody else could already read, and an existing file — a symbolic link
    included — is refused rather than followed. ``overwrite`` replaces one instead, by writing a
    new ``0600`` file beside it and renaming it into place, so a failure part way never leaves
    half a token where the working one was.
    """

    path: Path
    overwrite: bool = False

    def describe(self) -> str:
        return f"the file {self.path}, readable only by you"

    def check(self) -> None:
        parent = self.path.parent
        if not parent.is_dir():
            msg = f"{str(parent)!r} is not a directory, so there is nowhere to create the file."
            raise ConfigError(msg)
        # Found now rather than after a sign-in: a minted token that cannot be written is a
        # grant on the account with nothing to show for it.
        if not os.access(parent, os.W_OK | os.X_OK):
            msg = f"{str(parent)!r} is not writable by you, so the file cannot be created there."
            raise ConfigError(msg)
        if self.path.is_symlink() or self.path.exists():
            if not self.overwrite:
                msg = (
                    f"{str(self.path)!r} already exists, and a token is never written over a "
                    f"file silently. Choose another path, or pass `--overwrite-token-file` to "
                    f"replace it."
                )
                raise ConfigError(msg)
            if self.path.is_dir():
                msg = f"{str(self.path)!r} is a directory, not a file a token can replace."
                raise ConfigError(msg)

    def write(self, token: SecretStr) -> None:
        data = (token.get_secret_value() + "\n").encode()
        if not self.overwrite:
            _create_private(self.path, data)
            return
        staging = self.path.with_name(f".{self.path.name}.{secrets.token_hex(6)}.tmp")
        try:
            _create_private(staging, data)
            staging.replace(self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                staging.unlink()
            raise


def _create_private(path: Path, data: bytes) -> None:
    """Create ``path`` with mode ``0600`` and exactly ``data``, or leave nothing behind."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        # The mode argument is filtered by the umask, which can only narrow it; this makes the
        # result independent of it in the other direction as well.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(OSError):
            path.unlink()
        raise


@dataclass(frozen=True, slots=True)
class StreamSink:
    """A stream that is not a terminal — a pipe into a secret store, or a redirect.

    A terminal is refused because what is printed to one outlives the command: scrollback, a
    screen recording, a shared session, a transcript an assistant is reading. A pipe is the
    caller's own plumbing, which is where a secret-store command expects to read it from.
    """

    stream: TextIO

    def describe(self) -> str:
        return "standard output"

    def check(self) -> None:
        if self.stream.isatty():
            msg = (
                "standard output is a terminal, and a refresh token printed to one stays in "
                "its scrollback and in anything recording the screen. Pass `--token-file PATH` "
                "to write a file only you can read, or pipe this command's output into your "
                "secret store."
            )
            raise ConfigError(msg)

    def write(self, token: SecretStr) -> None:
        # Checked again at the last moment: the stream was inspected before a sign-in that took
        # minutes, and it is the one place the token could reach a screen.
        self.check()
        self.stream.write(token.get_secret_value() + "\n")
        self.stream.flush()


def _ignore(message: str) -> None:
    del message


def _open(open_browser: Callable[[str], object], url: str) -> None:
    """Start the browser. Failing to is not an error: the address was announced for that case."""
    with contextlib.suppress(Exception):
        open_browser(url)


async def mint_refresh_token(
    config: GmailConfig,
    *,
    announce: Callable[[str], None] = _ignore,
    open_browser: Callable[[str], object] | None = None,
    timeout_seconds: float = DEFAULT_SIGN_IN_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SecretStr:
    """Run the installed-app flow once and return the refresh token Google issued.

    The token is **not verified here** — that it belongs to the instance's mailbox is the
    caller's check, :meth:`~manicule.connectors.credentials.RefreshTokenCredential.confirm_mailbox`,
    made before it is written anywhere.

    Args:
        config: The instance's settings with its client secret resolved. ``account`` is offered
            to Google as a hint for which account to choose; it decides nothing, because the
            mailbox is checked afterward.
        announce: Receives the lines a person needs while this waits, including the address to
            open when no browser could be started. None of them carries a secret: the address
            holds the client id, the PKCE challenge and ``state``, all of which Google sees
            anyway.
        open_browser: Opens an address. ``None`` uses the system's default browser. A parameter
            so the flow can be driven without one.
        timeout_seconds: How long to wait for the sign-in to finish.
        transport: Replaces the network for the code exchange, as it does for the credential.

    Raises:
        ConfigError: The HTTP client is not installed, or no client secret was resolved.
        CredentialError: The person declined, the redirect failed the ``state`` check, the wait
            ran out, or Google refused the exchange or granted something other than what was
            asked. The message never carries the code, the token or Google's free text.
    """
    _require_httpx()
    if config.client_secret is None:
        msg = "the Gmail sign-in needs the OAuth client's secret resolved before it starts"
        raise ConfigError(msg)
    if open_browser is None:
        import webbrowser  # noqa: PLC0415 - only a person at a terminal needs a browser

        open_browser = webbrowser.open

    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(32)
    async with _Loopback(state) as loopback:
        url = f"{GMAIL_AUTH_URL}?" + urlencode(
            {
                "client_id": config.client_id,
                "redirect_uri": loopback.redirect_uri,
                "response_type": "code",
                "scope": GMAIL_READONLY_SCOPE,
                "access_type": "offline",
                "prompt": "consent",
                "code_challenge": _challenge(verifier),
                "code_challenge_method": "S256",
                "state": state,
                "login_hint": config.account,
            }
        )
        announce(
            "Opening your browser to sign in. Choose the account this instance indexes. "
            f"If nothing opens, open this address yourself:\n  {url}"
        )
        # On a thread of its own, not waited for, and a daemon: a console browser blocks until
        # the person quits it, and neither the redirect, the timeout below nor the process's
        # exit may wait on that. The loop's default executor would be joined by `asyncio.run`
        # on the way out, which is exactly the wait this avoids.
        threading.Thread(
            target=_open, args=(open_browser, url), name="gmail-login-browser", daemon=True
        ).start()
        announce(f"Waiting up to {timeout_seconds:g} seconds for the sign-in to finish.")
        try:
            async with asyncio.timeout(timeout_seconds):
                code = await loopback.code()
        except TimeoutError:
            msg = (
                f"no sign-in arrived within {timeout_seconds:g} seconds, so nothing was "
                f"issued. Run the command again; `--timeout` allows longer."
            )
            raise CredentialError(msg) from None
        redirect_uri = loopback.redirect_uri

    return await _exchange(
        config,
        client_secret=config.client_secret,
        code=code,
        verifier=verifier,
        redirect_uri=redirect_uri,
        transport=transport,
    )


def _challenge(verifier: str) -> str:
    """The PKCE ``S256`` challenge: the unpadded URL-safe base64 of the verifier's SHA-256."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _require_httpx() -> None:
    if importlib.util.find_spec("httpx") is None:
        msg = (
            "signing in to Gmail needs the HTTP client, which this installation lacks. Install "
            "it with `pip install 'manicule[connectors]'`."
        )
        raise ConfigError(msg)


async def _exchange(
    config: GmailConfig,
    *,
    client_secret: SecretStr,
    code: SecretStr,
    verifier: str,
    redirect_uri: str,
    transport: httpx.AsyncBaseTransport | None,
) -> SecretStr:
    """Redeem the authorization code for a refresh token, and check what it grants."""
    import httpx  # noqa: PLC0415 - no HTTP stack at plugin discovery

    from manicule.connectors.credentials import GOOGLE_TOKEN_URL  # noqa: PLC0415

    form = {
        "grant_type": "authorization_code",
        "code": code.get_secret_value(),
        "client_id": config.client_id,
        "client_secret": client_secret.get_secret_value(),
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=config.request_timeout_seconds
        ) as http:
            response = await http.post(GOOGLE_TOKEN_URL, data=form)
    except httpx.TimeoutException as exc:
        msg = "Google's token endpoint did not answer in time"
        raise RequestTimeoutError(msg) from exc
    except httpx.HTTPError as exc:
        msg = "Google's token endpoint could not be reached"
        raise ConnectorError(msg) from exc

    body = _json_object(response)
    if response.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED):
        raise _refused(body.get("error"), config)
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        msg = "Google's token endpoint is throttling this client"
        raise RateLimitedError(msg)
    if response.status_code != HTTPStatus.OK:
        msg = f"Google's token endpoint answered HTTP {response.status_code}"
        raise RemoteError(msg, response.status_code)

    granted = body.get("scope")
    if not isinstance(granted, str) or set(granted.split()) != {GMAIL_READONLY_SCOPE}:
        msg = (
            "Google issued a token for something other than exactly the gmail.readonly scope, "
            "so it was discarded. Nothing wider than read-only mail is ever kept."
        )
        raise CredentialError(msg)
    token = body.get("refresh_token")
    if not isinstance(token, str) or not token:
        msg = (
            "Google signed the account in but issued no refresh token. Remove this client's "
            "access under the account's third-party apps and connections, then run the "
            "command again."
        )
        raise CredentialError(msg)
    _log.debug("minted a refresh token for Gmail client %s", config.client_id)
    return SecretStr(token)


def _refused(error: object, config: GmailConfig) -> CredentialError:
    """The typed error for a refused exchange, from Google's error code and nothing else."""
    code = error if isinstance(error, str) and _ERROR_CODE.fullmatch(error) else "unknown"
    if code in {"invalid_client", "unauthorized_client"}:
        msg = (
            f"Google rejected the OAuth client ({code}). Check that client_id and the client "
            f"secret (${config.client_secret_env}) belong to the same Desktop-app client."
        )
    elif code == "invalid_grant":
        msg = (
            "Google refused the authorization code (invalid_grant). It is single use and "
            "short lived; run the command again and finish the sign-in in one go."
        )
    else:
        msg = f"Google refused the token exchange ({code})."
    return CredentialError(msg)


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}


class _Loopback:
    """A listener on ``127.0.0.1`` that accepts exactly one authorization redirect.

    **Bound to the loopback address only**, on a port the operating system picks, so it is not
    reachable from another machine and its address cannot be guessed ahead of time. It answers
    one request that carries the answer and ignores the rest (a favicon, a stray probe), and a
    request that claims to be the answer but carries the wrong ``state`` ends the sign-in rather
    than being skipped: that is a redirect this sign-in did not start.

    A request is also refused unless its ``Host`` is the address this socket was opened on,
    which keeps a web page from a hostile name that resolves here (DNS rebinding) from talking
    to it.
    """

    def __init__(self, state: str) -> None:
        self.state = state
        self.redirect_uri = ""
        self._port = 0
        self._server: asyncio.Server | None = None
        self._result: asyncio.Future[SecretStr] | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def __aenter__(self) -> _Loopback:
        result: asyncio.Future[SecretStr] = asyncio.get_running_loop().create_future()
        self._result = result
        self._server = await asyncio.start_server(
            self._serve, host="127.0.0.1", port=0, limit=_REQUEST_HEAD_LIMIT
        )
        self._port = self._server.sockets[0].getsockname()[1]
        self.redirect_uri = f"http://127.0.0.1:{self._port}"
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info
        if self._server is not None:
            self._server.close()
            for writer in tuple(self._writers):
                writer.close()
            await self._server.wait_closed()
        result = self._result
        if result is not None and result.done() and not result.cancelled():
            result.exception()  # consumed, so an unread failure is not reported twice

    def code(self) -> Awaitable[SecretStr]:
        """The authorization code, or the failure the redirect carried."""
        assert self._result is not None  # noqa: S101 - set by __aenter__
        return self._result

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            try:
                async with asyncio.timeout(_CONNECTION_TIMEOUT_SECONDS):
                    head = await reader.readuntil(b"\r\n\r\n")
            except (TimeoutError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                return
            status, page = self._answer(head)
            body = f"<!doctype html><title>manicule</title><p>{page}</p>".encode() if page else b""
            writer.write(
                (
                    f"HTTP/1.1 {status.value} {status.phrase}\r\n"
                    "Content-Type: text/html; charset=utf-8\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Cache-Control: no-store\r\n"
                    "Referrer-Policy: no-referrer\r\n"
                    "X-Content-Type-Options: nosniff\r\n"
                    "Content-Security-Policy: default-src 'none'\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                + body
            )
            with contextlib.suppress(ConnectionError, OSError):
                await writer.drain()
        finally:
            self._writers.discard(writer)
            writer.close()

    def _answer(self, head: bytes) -> tuple[HTTPStatus, str]:
        """Interpret one request, and say what to show; settles the sign-in if it is the answer."""
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        headers = {
            name.strip().lower(): value.strip()
            for name, _, value in (line.partition(":") for line in lines[1:] if line)
        }
        if len(parts) != 3 or parts[0] != "GET":  # noqa: PLR2004 - method, target, version
            return HTTPStatus.METHOD_NOT_ALLOWED, ""
        if headers.get("host") != f"127.0.0.1:{self._port}":
            return HTTPStatus.BAD_REQUEST, ""
        target = urlsplit(parts[1])
        params = parse_qs(target.query)
        if target.path != "/" or not params.keys() & {"code", "error", "state"}:
            return HTTPStatus.NOT_FOUND, ""
        assert self._result is not None  # noqa: S101 - set by __aenter__
        if self._result.done():
            return HTTPStatus.GONE, _PAGE["failed"]
        return self._settle(self._result, params)

    def _settle(
        self, result: asyncio.Future[SecretStr], params: dict[str, list[str]]
    ) -> tuple[HTTPStatus, str]:
        """Decide the sign-in from a redirect's parameters: ``state`` first, then error or code."""
        given = params.get("state", [""])[0]
        if not secrets.compare_digest(given.encode(), self.state.encode()):
            msg = (
                "the redirect did not carry this sign-in's state, so it was refused: it was "
                "not a response to the address this command opened. Nothing was issued."
            )
            result.set_exception(CredentialError(msg))
            return HTTPStatus.BAD_REQUEST, _PAGE["failed"]
        error = params.get("error", [""])[0]
        code = params.get("code", [""])[0]
        if error or not code:
            reason = error if _ERROR_CODE.fullmatch(error) else "unknown"
            if reason == "access_denied":
                msg = "the sign-in was declined in the browser, so nothing was issued."
            elif error:
                msg = f"Google ended the sign-in with an error ({reason}), so nothing was issued."
            else:
                msg = "the redirect carried neither a code nor an error, so nothing was issued."
            result.set_exception(CredentialError(msg))
            return HTTPStatus.OK, _PAGE["failed"]
        result.set_result(SecretStr(code))
        return HTTPStatus.OK, _PAGE["done"]
