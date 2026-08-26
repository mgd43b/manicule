"""Stable page identity and canonical routes, independent of how pages are fetched."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import quote, unquote_to_bytes, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DEFAULT_PAGE_SUFFIXES",
    "MAX_MANIFEST_BYTES",
    "MAX_MANIFEST_PAGES",
    "SiteManifest",
    "SiteManifestPage",
    "SiteRouteError",
    "SiteRouteRecord",
    "canonical_site_url",
    "infer_site_records",
    "normalize_base_url",
    "normalize_repository_path",
    "normalize_site_route",
    "parse_site_manifest",
    "records_from_manifest",
]

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_PAGES = 10_000
MAX_ROUTE_DEPTH = 128
MAX_PATH_LENGTH = 4_096
MAX_ID_LENGTH = 512
MAX_TITLE_LENGTH = 2_048
MAX_MEDIA_TYPE_LENGTH = 255
DEFAULT_PAGE_SUFFIXES = (".html", ".md", ".mdx")
_FIRST_CONTROL_CODEPOINT = 32
_DEFAULT_PORTS = {"http": 80, "https": 443}

_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")


class SiteRouteError(ValueError):
    """A route inventory cannot describe one unambiguous website."""


def _plain_text(value: str, *, field: str, maximum: int) -> str:
    if not value or len(value) > maximum:
        raise SiteRouteError(f"{field} must contain between 1 and {maximum} characters")
    if value != value.strip() or any(
        ord(character) < _FIRST_CONTROL_CODEPOINT for character in value
    ):
        raise SiteRouteError(f"{field} must not contain surrounding whitespace or controls")
    normalized = unicodedata.normalize("NFC", value)
    if value != normalized:
        raise SiteRouteError(f"{field} must already use Unicode NFC normalization")
    return value


def normalize_repository_path(value: str, *, allow_root: bool = False) -> str:
    """Validate one literal Git path without turning traversal into a different path."""
    if value == "." and allow_root:
        return value
    value = _plain_text(value, field="source", maximum=MAX_PATH_LENGTH)
    if "\\" in value or value.startswith("/") or value.endswith("/") or "//" in value:
        raise SiteRouteError("source must be a normalized repository-relative POSIX path")
    parts = value.split("/")
    if len(parts) > MAX_ROUTE_DEPTH or any(part in {"", ".", ".."} for part in parts):
        raise SiteRouteError("source must not contain empty, dot, or traversal segments")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise SiteRouteError("source must be a normalized repository-relative POSIX path")
    return value


def _decoded_route_segment(value: str) -> str:
    if _BAD_PERCENT.search(value):
        raise SiteRouteError("route contains malformed percent encoding")
    try:
        decoded = unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SiteRouteError("route percent encoding must contain valid UTF-8") from exc
    decoded = unicodedata.normalize("NFC", decoded)
    if decoded in {"", ".", ".."} or "/" in decoded or "\\" in decoded:
        raise SiteRouteError("route contains an ambiguous or traversal segment")
    if any(ord(character) < _FIRST_CONTROL_CODEPOINT for character in decoded):
        raise SiteRouteError("route must not contain control characters")
    return decoded


def normalize_site_route(value: str) -> str:
    """Return one directory-style, percent-encoded route under the configured site root."""
    value = _plain_text(value, field="route", maximum=MAX_PATH_LENGTH)
    parsed = urlsplit(value)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not value.startswith("/")
    ):
        raise SiteRouteError("route must be an absolute site path without an origin or suffix")
    if "\\" in parsed.path:
        raise SiteRouteError("route must use URL path separators")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) > MAX_ROUTE_DEPTH:
        raise SiteRouteError(f"route may contain at most {MAX_ROUTE_DEPTH} segments")
    encoded = [quote(_decoded_route_segment(segment), safe="-._~") for segment in segments]
    return "/" if not encoded else f"/{'/'.join(encoded)}/"


def normalize_base_url(value: str) -> str:
    """Validate and canonicalize an HTTP(S) site root without credentials or URL suffixes."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise SiteRouteError("base_url must contain a valid host and optional port") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.hostname is None:
        raise SiteRouteError("base_url must be an absolute HTTP(S) URL with a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise SiteRouteError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise SiteRouteError("base_url must not contain a query string or fragment")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise SiteRouteError("base_url hostname is invalid") from exc
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        host = f"{host}:{port}"
    root = normalize_site_route(parsed.path or "/")
    return urlunsplit((scheme, host, root, "", ""))


def canonical_site_url(base_url: str, route: str) -> str:
    """Join a normalized site-relative route without allowing an origin or root escape."""
    base = normalize_base_url(base_url)
    normalized_route = normalize_site_route(route)
    parsed = urlsplit(base)
    root = parsed.path.rstrip("/")
    path = f"{root}{normalized_route}" if root else normalized_route
    joined = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    if (urlsplit(joined).scheme, urlsplit(joined).netloc) != (parsed.scheme, parsed.netloc):
        raise SiteRouteError("route changed the configured site origin")
    return joined


class SiteManifestPage(BaseModel):
    """One declared source page and its public route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=MAX_PATH_LENGTH)
    id: str | None = Field(default=None, min_length=1, max_length=MAX_ID_LENGTH)
    route: str = Field(min_length=1, max_length=MAX_PATH_LENGTH)
    title: str | None = Field(default=None, min_length=1, max_length=MAX_TITLE_LENGTH)
    media_type: str | None = Field(default=None, min_length=1, max_length=MAX_MEDIA_TYPE_LENGTH)

    @field_validator("source")
    @classmethod
    def _source(cls, value: str) -> str:
        return normalize_repository_path(value)

    @field_validator("route")
    @classmethod
    def _route(cls, value: str) -> str:
        return normalize_site_route(value)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str | None) -> str | None:
        return None if value is None else _plain_text(value, field="id", maximum=MAX_ID_LENGTH)

    @field_validator("title")
    @classmethod
    def _title(cls, value: str | None) -> str | None:
        return (
            None if value is None else _plain_text(value, field="title", maximum=MAX_TITLE_LENGTH)
        )

    @field_validator("media_type")
    @classmethod
    def _media_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _plain_text(value, field="media_type", maximum=MAX_MEDIA_TYPE_LENGTH).lower()
        if not _MEDIA_TYPE.fullmatch(value):
            raise SiteRouteError("media_type must be a type/subtype without parameters")
        return value


class SiteManifest(BaseModel):
    """The closed v1 route manifest, validated as one atomic inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    pages: tuple[SiteManifestPage, ...] = Field(max_length=MAX_MANIFEST_PAGES)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        _require_unique(self.pages, "source", lambda page: page.source)
        _require_unique(
            (page for page in self.pages if page.id is not None),
            "id",
            lambda page: page.id,
        )
        _require_unique(self.pages, "route", lambda page: page.route)
        # **The namespace `identity` actually reads, which is neither of the two above.**
        # `SiteRouteRecord.identity` is `self.id or self.route`, so ids and routes share one
        # space — and checking each separately admits a collision between them. A page with
        # `id: "/b/"` and a *different* page whose route is `/b/` both answer `/b/`, both checks
        # pass, and the manifest is accepted. The two then collapse onto one document identity:
        # the inventory keyed by identity keeps whichever was built last and the other page is
        # silently not indexed, with nothing anywhere reporting a document it did not store.
        _require_unique(self.pages, "identity", lambda page: page.id or page.route)
        return self


class SiteRouteRecord(SiteManifestPage):
    """Canonical page identity and every route fact that affects a citation."""

    @classmethod
    def from_page(cls, page: SiteManifestPage) -> SiteRouteRecord:
        return cls.model_validate(page.model_dump())

    @property
    def identity(self) -> str:
        return self.id or self.route

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def _require_unique[T](items: Iterable[T], field: str, value_of: Callable[[T], str | None]) -> None:
    seen: set[str] = set()
    for item in items:
        value = value_of(item)
        if value is None:
            continue
        if value in seen:
            raise SiteRouteError(f"duplicate page {field}: {value!r}")
        seen.add(value)


def _object_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise SiteRouteError(f"manifest contains duplicate JSON key {key!r}")
        found[key] = value
    return found


def parse_site_manifest(data: bytes, *, max_bytes: int = MAX_MANIFEST_BYTES) -> SiteManifest:
    """Decode a bounded manifest while refusing duplicate JSON keys and unknown fields."""
    if len(data) > max_bytes:
        raise SiteRouteError(f"site manifest exceeds the {max_bytes}-byte limit")
    try:
        decoded = json.loads(data, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SiteRouteError("site manifest is not valid UTF-8 JSON") from exc
    return SiteManifest.model_validate(decoded)


def _relative_to_content_root(source: str, content_root: str) -> tuple[str, ...]:
    normalized_source = normalize_repository_path(source)
    normalized_root = normalize_repository_path(content_root, allow_root=True)
    source_parts = tuple(normalized_source.split("/"))
    root_parts = () if normalized_root == "." else tuple(normalized_root.split("/"))
    if source_parts[: len(root_parts)] != root_parts or len(source_parts) == len(root_parts):
        raise SiteRouteError(f"source {source!r} is not a page below content_root")
    return source_parts[len(root_parts) :]


def _inferred_route(source: str, content_root: str, suffixes: Sequence[str]) -> str:
    relative = list(_relative_to_content_root(source, content_root))
    filename = relative[-1]
    suffix = next((candidate for candidate in suffixes if filename.endswith(candidate)), None)
    if suffix is None:
        raise SiteRouteError(f"source {source!r} has no recognized page suffix")
    stem = filename[: -len(suffix)]
    if not stem:
        raise SiteRouteError(f"source {source!r} has an empty page name")
    if stem == "index":
        relative.pop()
    else:
        relative[-1] = stem
    encoded = "/".join(quote(part, safe="-._~") for part in relative)
    return normalize_site_route(f"/{encoded}/" if encoded else "/")


def infer_site_records(
    sources: Iterable[str],
    *,
    content_root: str = ".",
    suffixes: Sequence[str] = DEFAULT_PAGE_SUFFIXES,
) -> tuple[SiteRouteRecord, ...]:
    """Infer and atomically validate routes for one complete admitted inventory."""
    records = tuple(
        SiteRouteRecord(source=source, route=_inferred_route(source, content_root, suffixes))
        for source in sorted(sources)
    )
    _require_unique(records, "source", lambda record: record.source)
    _require_unique(records, "route", lambda record: record.route)
    return records


def records_from_manifest(
    manifest: SiteManifest,
    admitted_sources: Iterable[str],
    *,
    content_root: str = ".",
) -> tuple[SiteRouteRecord, ...]:
    """Bind an authoritative manifest to the admitted tree without inferred fallbacks."""
    admitted = {normalize_repository_path(source) for source in admitted_sources}
    declared = {page.source for page in manifest.pages}
    for source in declared:
        _relative_to_content_root(source, content_root)
    missing = sorted(admitted - declared)
    extra = sorted(declared - admitted)
    if missing or extra:
        detail: list[str] = []
        if missing:
            detail.append(f"omitted admitted sources: {missing!r}")
        if extra:
            detail.append(f"declared non-admitted sources: {extra!r}")
        raise SiteRouteError("authoritative manifest mismatch: " + "; ".join(detail))
    records = (SiteRouteRecord.from_page(page) for page in manifest.pages)
    return tuple(sorted(records, key=lambda record: record.source))
