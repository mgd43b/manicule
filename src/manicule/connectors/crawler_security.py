"""Canonical crawler URLs and the SSRF boundary before any network transport exists."""

from __future__ import annotations

import ipaddress
import re
import socket
import unicodedata
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

from manicule.connectors.errors import (
    CrawlerAddressError,
    CrawlerPolicyError,
    CrawlerRedirectError,
    CrawlerScopeError,
)

__all__ = [
    "DEFAULT_SENSITIVE_HEADERS",
    "DEFAULT_TRACKING_PARAMETERS",
    "AddressResolver",
    "CanonicalDecision",
    "ConnectionPlan",
    "CrawlerUrlPolicy",
    "NormalizedUrl",
    "RedirectDecision",
    "TrailingSlashPolicy",
    "bounded_response_headers",
    "resolve_system_addresses",
]

_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_URL_CHARS = 8_192
_MAX_QUERY_FIELDS = 128
_MAX_PATH_SEGMENTS = 256
_MAX_HOST_CHARS = 253
_MAX_QUERY_KEY_CHARS = 256
_MAX_REDIRECTS = 32
_FIRST_PRINTABLE_CODEPOINT = 32
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")
_SINGLETON_RESPONSE_HEADERS = frozenset(
    {"content-encoding", "content-length", "content-type", "location", "retry-after"}
)

DEFAULT_TRACKING_PARAMETERS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "utm_campaign",
        "utm_content",
        "utm_medium",
        "utm_source",
        "utm_term",
    }
)
DEFAULT_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authorization",
        "x-api-key",
    }
)


class TrailingSlashPolicy(StrEnum):
    """How page paths with a non-root trailing slash become one identity."""

    PRESERVE = "preserve"
    DIRECTORY = "directory"
    STRIP = "strip"


@dataclass(frozen=True, slots=True)
class NormalizedUrl:
    """One canonical, in-scope HTTP(S) page identity."""

    url: str
    origin: str
    path: str
    query: tuple[tuple[str, str], ...]

    @property
    def hostname(self) -> str:
        hostname = urlsplit(self.url).hostname
        if hostname is None:  # pragma: no cover - construction validates this invariant
            raise RuntimeError("normalized URL lost its hostname")
        return hostname

    @property
    def port(self) -> int:
        parsed = urlsplit(self.url)
        return parsed.port or _DEFAULT_PORTS[parsed.scheme]

    @property
    def tls_server_name(self) -> str | None:
        return self.hostname if urlsplit(self.url).scheme == "https" else None


@dataclass(frozen=True, slots=True)
class CanonicalDecision:
    """Whether publisher-declared canonical evidence is safe to use."""

    accepted: bool
    value: NormalizedUrl | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RedirectDecision:
    """An authorized redirect and whether origin-bound headers may travel."""

    value: NormalizedUrl
    forward_sensitive_headers: bool


@dataclass(frozen=True, slots=True)
class ConnectionPlan:
    """A request target whose addresses were authorized once and must be dialed directly."""

    target: NormalizedUrl
    addresses: tuple[str, ...]

    @property
    def port(self) -> int:
        return self.target.port

    @property
    def tls_server_name(self) -> str | None:
        return self.target.tls_server_name

    @property
    def verify_tls(self) -> bool:
        """HTTPS plans always require certificate and hostname verification."""
        return self.tls_server_name is not None

    def authorize_peer(self, peer: str) -> str:
        """Validate the actual connected address against the planned numeric targets."""
        normalized = _public_address(peer)
        if normalized not in self.addresses:
            raise CrawlerAddressError("crawler connection peer changed after address planning")
        return normalized


class AddressResolver(Protocol):
    """Resolve a hostname once; transports dial the returned numeric addresses directly."""

    def __call__(self, hostname: str, port: int) -> Awaitable[Sequence[str]]: ...


def _plain(value: str, *, what: str) -> str:
    if not value or len(value) > _MAX_URL_CHARS or value != value.strip() or _CONTROL.search(value):
        raise CrawlerPolicyError(
            f"crawler {what} is empty, oversized, padded, or contains controls"
        )
    if unicodedata.normalize("NFC", value) != value:
        raise CrawlerPolicyError(f"crawler {what} must use Unicode NFC normalization")
    return value


def _canonical_host(hostname: str) -> str:
    if "%" in hostname:
        raise CrawlerPolicyError("crawler URL host must not contain an IPv6 zone identifier")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            host = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise CrawlerPolicyError("crawler URL host is not valid IDNA") from exc
        labels = host.rstrip(".").split(".")
        if (
            not host
            or len(host) > _MAX_HOST_CHARS
            or any(not _HOST_LABEL.fullmatch(label) for label in labels)
        ):
            raise CrawlerPolicyError("crawler URL host is invalid") from None
        return host.rstrip(".")
    return literal.compressed


def _canonical_percent(value: str, *, decode_unreserved: bool) -> str:
    if _BAD_PERCENT.search(value):
        raise CrawlerPolicyError("crawler URL contains malformed percent encoding")

    def replace(match: re.Match[str]) -> str:
        byte = int(match.group(1), 16)
        character = chr(byte)
        if decode_unreserved and character in _UNRESERVED and character != ".":
            return character
        return f"%{byte:02X}"

    return _PERCENT.sub(replace, value)


def _canonical_segment(segment: str) -> str:
    canonical = _canonical_percent(segment, decode_unreserved=True)
    # Percent signs left by `_canonical_percent` are validated escapes and must survive quote.
    return quote(canonical, safe="!$&'()*+,-.:;=@_~%")


def _canonical_path(path: str, trailing_slash: TrailingSlashPolicy) -> str:
    if "\\" in path:
        raise CrawlerPolicyError("crawler URL path must use URL separators")
    segments: list[str] = []
    for raw in path.split("/"):
        if raw in {"", "."}:
            continue
        if raw == "..":
            if segments:
                segments.pop()
            continue
        segments.append(_canonical_segment(raw))
    if len(segments) > _MAX_PATH_SEGMENTS:
        raise CrawlerPolicyError("crawler URL path exceeds the segment limit")
    result = "/" + "/".join(segments)
    if result == "/":
        return result
    if trailing_slash is TrailingSlashPolicy.DIRECTORY:
        return f"{result}/"
    if trailing_slash is TrailingSlashPolicy.STRIP:
        return result
    return f"{result}/" if path.endswith("/") else result


def _origin(scheme: str, hostname: str, port: int | None) -> str:
    host = f"[{hostname}]" if ":" in hostname else hostname
    suffix = "" if port is None or port == _DEFAULT_PORTS[scheme] else f":{port}"
    return f"{scheme}://{host}{suffix}"


def _normalized_origin(value: str) -> str:
    raw = _plain(value, what="allowed origin")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise CrawlerPolicyError("crawler allowed origin is malformed") from exc
    if (
        parsed.scheme.lower() not in _DEFAULT_PORTS
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        raise CrawlerPolicyError("crawler allowed origin must be an HTTP(S) origin only")
    return _origin(parsed.scheme.lower(), _canonical_host(parsed.hostname), port)


def _prefix(value: str) -> str:
    raw = _plain(value, what="allowed path prefix")
    if not raw.startswith("/") or "?" in raw or "#" in raw:
        raise CrawlerPolicyError("crawler path prefix must be an absolute URL path")
    canonical = _canonical_path(raw, TrailingSlashPolicy.PRESERVE)
    return canonical if canonical == "/" or canonical.endswith("/") else f"{canonical}/"


def _under_prefix(path: str, prefix: str) -> bool:
    if prefix == "/":
        return True
    without_slash = prefix.rstrip("/")
    return path == without_slash or path.startswith(prefix)


def _public_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError as exc:
        raise CrawlerAddressError("crawler resolver returned a non-IP destination") from exc
    if not address.is_global or any(
        (
            address.is_loopback,
            address.is_link_local,
            address.is_private,
            address.is_multicast,
            address.is_unspecified,
            address.is_reserved,
        )
    ):
        raise CrawlerAddressError("crawler destination is not globally routable")
    if isinstance(address, ipaddress.IPv6Address):
        embedded = address.ipv4_mapped or address.sixtofour
        if address in _NAT64_WELL_KNOWN:
            embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        if embedded is not None and not embedded.is_global:
            raise CrawlerAddressError("crawler destination embeds a non-public IPv4 address")
        if address.teredo is not None and any(not item.is_global for item in address.teredo):
            raise CrawlerAddressError("crawler destination embeds a non-public Teredo address")
    return address.compressed


class CrawlerUrlPolicy:
    """One normalization and authorization path for every crawler URL source."""

    def __init__(
        self,
        *,
        allowed_origins: Iterable[str],
        allowed_path_prefixes: Iterable[str] = ("/",),
        allowed_query_keys: Iterable[str] = (),
        tracking_parameters: Iterable[str] = DEFAULT_TRACKING_PARAMETERS,
        trailing_slash: TrailingSlashPolicy = TrailingSlashPolicy.PRESERVE,
        max_redirects: int = 8,
    ) -> None:
        origins = tuple(dict.fromkeys(_normalized_origin(value) for value in allowed_origins))
        if not origins:
            raise CrawlerPolicyError("crawler policy requires at least one allowed origin")
        prefixes = tuple(dict.fromkeys(_prefix(value) for value in allowed_path_prefixes))
        if not prefixes:
            raise CrawlerPolicyError("crawler policy requires at least one allowed path prefix")
        if max_redirects < 0 or max_redirects > _MAX_REDIRECTS:
            raise CrawlerPolicyError("crawler redirect limit must be between 0 and 32")
        self.allowed_origins = frozenset(origins)
        self.allowed_path_prefixes = prefixes
        self.allowed_query_keys = frozenset(_query_key(value) for value in allowed_query_keys)
        self.tracking_parameters = frozenset(_query_key(value) for value in tracking_parameters)
        self.trailing_slash = trailing_slash
        self.max_redirects = max_redirects

    def normalize(self, value: str, *, base: str | None = None) -> NormalizedUrl:
        """Normalize and authorize a configured, discovered, redirected or canonical URL."""
        return self._normalize(value, base=base, enforce_path=True)

    def normalize_auxiliary(self, value: str, *, base: str | None = None) -> NormalizedUrl:
        """Normalize an origin-scoped transport resource such as that origin's robots file."""
        return self._normalize(value, base=base, enforce_path=False)

    def _normalize(
        self, value: str, *, base: str | None, enforce_path: bool
    ) -> NormalizedUrl:
        raw = _plain(value, what="URL")
        if base is not None:
            normalized_base = self._normalize(base, base=None, enforce_path=enforce_path)
            raw = urljoin(normalized_base.url, raw)
        try:
            parsed = urlsplit(raw)
            port = parsed.port
        except ValueError as exc:
            raise CrawlerPolicyError("crawler URL is malformed") from exc
        scheme = parsed.scheme.lower()
        if scheme not in _DEFAULT_PORTS or parsed.hostname is None:
            raise CrawlerPolicyError("crawler URL must be absolute HTTP(S)")
        if port == 0:
            raise CrawlerPolicyError("crawler URL port must be positive")
        if parsed.username is not None or parsed.password is not None:
            raise CrawlerPolicyError("crawler URL must not contain user information")
        host = _canonical_host(parsed.hostname)
        origin = _origin(scheme, host, port)
        if origin not in self.allowed_origins:
            raise CrawlerScopeError("crawler URL origin is outside configured scope")
        path = _canonical_path(parsed.path or "/", self.trailing_slash)
        if enforce_path and not any(
            _under_prefix(path, prefix) for prefix in self.allowed_path_prefixes
        ):
            raise CrawlerScopeError("crawler URL path is outside configured scope")
        query = self._query(parsed.query)
        netloc = f"[{host}]" if ":" in host else host
        if port is not None and port != _DEFAULT_PORTS[scheme]:
            netloc = f"{netloc}:{port}"
        url = urlunsplit((scheme, netloc, path, urlencode(query, doseq=True), ""))
        return NormalizedUrl(url=url, origin=origin, path=path, query=query)

    def _query(self, raw: str) -> tuple[tuple[str, str], ...]:
        if not raw:
            return ()
        if raw.count("&") + 1 > _MAX_QUERY_FIELDS or _BAD_PERCENT.search(raw):
            raise CrawlerPolicyError("crawler URL query is malformed or exceeds its field limit")
        try:
            pairs = parse_qsl(
                raw,
                keep_blank_values=True,
                strict_parsing=True,
                errors="strict",
                max_num_fields=_MAX_QUERY_FIELDS,
            )
        except (UnicodeError, ValueError) as exc:
            raise CrawlerPolicyError("crawler URL query is malformed") from exc
        kept: list[tuple[str, str]] = []
        for key, value in pairs:
            normalized_key = unicodedata.normalize("NFC", key)
            normalized_value = unicodedata.normalize("NFC", value)
            if normalized_key in self.tracking_parameters:
                continue
            if normalized_key not in self.allowed_query_keys:
                raise CrawlerScopeError("crawler URL query key is outside configured scope")
            if _CONTROL.search(normalized_key) or _CONTROL.search(normalized_value):
                raise CrawlerPolicyError("crawler URL query contains controls")
            kept.append((normalized_key, normalized_value))
        return tuple(sorted(kept))

    def canonical(self, current: str, declared: str) -> CanonicalDecision:
        """Treat an in-scope canonical as evidence and an out-of-scope one as a diagnostic."""
        try:
            return CanonicalDecision(accepted=True, value=self.normalize(declared, base=current))
        except CrawlerPolicyError as exc:
            return CanonicalDecision(accepted=False, reason=type(exc).__name__)

    def redirect(
        self,
        current: str,
        location: str,
        *,
        hop: int,
        enforce_path: bool = True,
    ) -> RedirectDecision:
        """Authorize one redirect and state whether origin-bound headers may travel."""
        if hop < 1 or hop > self.max_redirects:
            raise CrawlerRedirectError("crawler redirect chain exceeded its configured bound")
        source = self._normalize(current, base=None, enforce_path=enforce_path)
        target = self._normalize(location, base=source.url, enforce_path=enforce_path)
        if urlsplit(source.url).scheme == "https" and urlsplit(target.url).scheme != "https":
            raise CrawlerRedirectError("crawler redirect attempted an HTTPS downgrade")
        return RedirectDecision(
            value=target,
            forward_sensitive_headers=source.origin == target.origin,
        )

    async def connection_plan(
        self, value: str | NormalizedUrl, resolver: AddressResolver
    ) -> ConnectionPlan:
        """Resolve once, validate every answer, and return only numeric dial targets."""
        target = value if isinstance(value, NormalizedUrl) else self.normalize(value)
        try:
            literal = ipaddress.ip_address(target.hostname)
        except ValueError:
            answers = await resolver(target.hostname, target.port)
        else:
            answers = (literal.compressed,)
        if not answers:
            raise CrawlerAddressError("crawler destination did not resolve")
        addresses = tuple(dict.fromkeys(_public_address(answer) for answer in answers))
        return ConnectionPlan(target=target, addresses=addresses)

    @staticmethod
    def redirected_headers(
        headers: Mapping[str, str], decision: RedirectDecision
    ) -> dict[str, str]:
        """Remove origin-bound secrets whenever a permitted redirect changes origin."""
        if decision.forward_sensitive_headers:
            return dict(headers)
        return {
            key: value
            for key, value in headers.items()
            if key.lower() not in DEFAULT_SENSITIVE_HEADERS
        }


def _query_key(value: str) -> str:
    if (
        not value
        or len(value) > _MAX_QUERY_KEY_CHARS
        or value != value.strip()
        or _CONTROL.search(value)
    ):
        raise CrawlerPolicyError("crawler query policy contains an invalid key")
    return unicodedata.normalize("NFC", value)


def bounded_response_headers(
    headers: Sequence[tuple[str, str]], *, max_bytes: int = 64 * 1024
) -> dict[str, str]:
    """Validate a response head before any body is read or redirect is followed."""
    if max_bytes < 1:
        raise ValueError("response header limit must be positive")
    total = 2
    found: dict[str, str] = {}
    for name, value in headers:
        lowered = name.lower()
        if (
            not name
            or _CONTROL.search(name)
            or any(
                ord(character) < _FIRST_PRINTABLE_CODEPOINT and character != "\t"
                for character in value
            )
            or "\x7f" in value
            or (lowered in _SINGLETON_RESPONSE_HEADERS and lowered in found)
        ):
            raise CrawlerPolicyError("crawler response contains malformed headers")
        total += len(name.encode()) + len(value.encode()) + 4
        if total > max_bytes:
            raise CrawlerPolicyError("crawler response headers exceed the configured bound")
        found[lowered] = value.strip()
    return found


async def resolve_system_addresses(hostname: str, port: int) -> Sequence[str]:
    """Bound transports may use this resolver, then dial the returned plan without DNS."""
    import asyncio  # noqa: PLC0415 - no event-loop import on pure policy use

    loop = asyncio.get_running_loop()
    try:
        answers = await loop.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise CrawlerAddressError("crawler destination resolution failed") from exc
    return tuple(str(item[4][0]) for item in answers)
