"""Website routes are stable data contracts, before either connector performs I/O."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from manicule.connectors.site_routes import (
    MAX_MANIFEST_PAGES,
    SiteManifest,
    SiteManifestPage,
    SiteRouteError,
    SiteRouteRecord,
    canonical_site_url,
    infer_site_records,
    normalize_base_url,
    normalize_repository_path,
    normalize_site_route,
    parse_site_manifest,
    records_from_manifest,
)


@pytest.mark.parametrize(
    ("spelling", "normalized"),
    [
        ("/", "/"),
        ("/Guide", "/Guide/"),
        ("/Guide/", "/Guide/"),
        ("/Guide//install", "/Guide/install/"),
        ("/café", "/caf%C3%A9/"),
        ("/caf%c3%a9/", "/caf%C3%A9/"),
        ("/space%20here", "/space%20here/"),
    ],
)
def test_equivalent_route_spellings_have_one_canonical_form(spelling: str, normalized: str) -> None:
    assert normalize_site_route(spelling) == normalized


@pytest.mark.parametrize(
    "route",
    [
        "relative",
        "https://other.example.test/page",
        "//other.example.test/page",
        "/page?draft=true",
        "/page#fragment",
        "/../secret",
        "/%2e%2e/secret",
        "/nested%2Fsecret",
        "/bad%encoding",
        "/bad%FFencoding",
        "/back\\slash",
    ],
)
def test_ambiguous_cross_origin_and_traversal_routes_are_refused(route: str) -> None:
    with pytest.raises(SiteRouteError):
        normalize_site_route(route)


@pytest.mark.parametrize(
    "source",
    ["/absolute.md", "../outside.md", "docs/../outside.md", "docs//page.md", "docs/page.md/"],
)
def test_repository_sources_are_literal_bounded_posix_paths(source: str) -> None:
    with pytest.raises(SiteRouteError):
        normalize_repository_path(source)


def test_site_roots_are_canonical_and_context_paths_are_preserved() -> None:
    assert normalize_base_url("HTTPS://Docs.Example.Test:443/handbook") == (
        "https://docs.example.test/handbook/"
    )
    assert canonical_site_url("https://docs.example.test/handbook", "/Guide/") == (
        "https://docs.example.test/handbook/Guide/"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "docs.example.test",
        "ftp://docs.example.test/",
        "https://user:secret@docs.example.test/",
        "https://docs.example.test/?token=secret",
        "https://docs.example.test/#top",
        "https://docs.example.test:bad/",
    ],
)
def test_a_site_root_cannot_carry_credentials_or_url_suffixes(base_url: str) -> None:
    with pytest.raises(SiteRouteError):
        normalize_base_url(base_url)


def test_inferred_routes_follow_only_the_declared_index_and_suffix_convention() -> None:
    records = infer_site_records(
        [
            "docs/index.md",
            "docs/Guide.md",
            "docs/api/index.html",
            "docs/what's-new.mdx",
            "docs/über.html",
        ],
        content_root="docs",
    )

    assert [(record.source, record.route) for record in records] == [
        ("docs/Guide.md", "/Guide/"),
        ("docs/api/index.html", "/api/"),
        ("docs/index.md", "/"),
        ("docs/what's-new.mdx", "/what%27s-new/"),
        ("docs/über.html", "/%C3%BCber/"),
    ]
    assert [record.identity for record in records] == [record.route for record in records]


def test_inferred_route_collisions_fail_the_complete_inventory() -> None:
    with pytest.raises(SiteRouteError, match="duplicate page route"):
        infer_site_records(["docs/guide.md", "docs/guide/index.md"], content_root="docs")


def _manifest(*pages: dict[str, object]) -> bytes:
    return json.dumps({"version": 1, "pages": pages}).encode()


def test_manifest_parsing_normalizes_routes_and_rejects_extra_fields() -> None:
    parsed = parse_site_manifest(
        _manifest(
            {
                "source": "docs/guide.md",
                "id": "guide",
                "route": "/guide",
                "title": "Guide",
                "media_type": "TEXT/MARKDOWN",
            }
        )
    )
    assert parsed.pages[0].route == "/guide/"
    assert parsed.pages[0].media_type == "text/markdown"

    with pytest.raises(ValidationError, match="extra"):
        parse_site_manifest(b'{"version":1,"pages":[],"framework":"guessed"}')
    with pytest.raises(ValidationError, match="literal_error"):
        parse_site_manifest(b'{"version":2,"pages":[]}')
    with pytest.raises(SiteRouteError, match="duplicate JSON key"):
        parse_site_manifest(b'{"version":1,"version":1,"pages":[]}')


@pytest.mark.parametrize("duplicate", ["source", "id", "route"])
def test_duplicate_manifest_identity_fields_fail_atomically(duplicate: str) -> None:
    first: dict[str, object] = {
        "source": "docs/one.md",
        "id": "one",
        "route": "/one/",
    }
    second: dict[str, object] = {
        "source": "docs/two.md",
        "id": "two",
        "route": "/two/",
    }
    second[duplicate] = first[duplicate]

    with pytest.raises(ValidationError, match=f"duplicate page {duplicate}"):
        parse_site_manifest(_manifest(first, second))


def test_an_id_may_not_collide_with_another_page_s_route() -> None:
    """`identity` is `id or route`, so ids and routes share one namespace.

    The three checks above are per field, and per field they all pass here: the ids are unique
    among the pages that have one, and the routes are unique. What collides is the *namespace*
    `SiteRouteRecord.identity` actually reads — a page carrying `id: "/b/"` and a different page
    whose route is `/b/` both answer `/b/`.

    The consequence is silent and total for one of them: the inventory is keyed by identity, so
    it keeps whichever was built last and the other page is simply not indexed, with nothing
    reporting a document that was discovered and not stored.

    The last case is the one that keeps this from being over-strict — a page whose `id` is its
    own route collides with nothing and must stay legal.
    """
    collide = _manifest(
        {"source": "docs/a.md", "route": "/a/", "id": "/b/"},
        {"source": "docs/b.md", "route": "/b/"},
    )
    with pytest.raises(ValidationError, match="duplicate page identity"):
        parse_site_manifest(collide)

    own = _manifest(
        {"source": "docs/a.md", "route": "/a/", "id": "/a/"},
        {"source": "docs/b.md", "route": "/b/"},
    )
    assert len(parse_site_manifest(own).pages) == 2


def test_manifest_limits_are_enforced_before_connector_discovery() -> None:
    with pytest.raises(SiteRouteError, match="byte limit"):
        parse_site_manifest(b"{}" * 10, max_bytes=4)
    page = {"source": "docs/a.md", "route": "/a/"}
    with pytest.raises(ValidationError, match="at most"):
        SiteManifest.model_validate({"version": 1, "pages": [page] * (MAX_MANIFEST_PAGES + 1)})
    with pytest.raises(ValidationError, match="at most 128 segments"):
        SiteManifestPage(source="docs/a.md", route="/" + "/".join(["a"] * 129))


def test_a_manifest_is_authoritative_over_the_admitted_inventory() -> None:
    manifest = SiteManifest(
        version=1,
        pages=(SiteManifestPage(source="docs/one.md", route="/one/"),),
    )
    with pytest.raises(SiteRouteError, match="omitted admitted sources"):
        records_from_manifest(manifest, ["docs/one.md", "docs/unlisted.md"], content_root="docs")
    with pytest.raises(SiteRouteError, match="declared non-admitted sources"):
        records_from_manifest(manifest, [], content_root="docs")


def test_explicit_ids_survive_moves_while_route_identity_does_not() -> None:
    before = SiteRouteRecord(source="docs/old.md", id="guide", route="/old/")
    after = SiteRouteRecord(source="docs/new.md", id="guide", route="/new/")
    implicit_before = SiteRouteRecord(source="docs/old.md", route="/old/")
    implicit_after = SiteRouteRecord(source="docs/new.md", route="/new/")

    assert before.identity == after.identity == "guide"
    assert implicit_before.identity != implicit_after.identity


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source", "docs/moved.md"),
        ("id", "renamed-id"),
        ("route", "/renamed/"),
        ("title", "Renamed title"),
        ("media_type", "text/html"),
    ],
)
def test_route_digest_covers_every_citation_fact(field: str, replacement: str) -> None:
    record = SiteRouteRecord(
        source="docs/guide.md",
        id="guide",
        route="/guide/",
        title="Guide",
        media_type="text/markdown",
    )
    assert record.model_copy(update={field: replacement}).digest != record.digest


def test_route_digests_ignore_json_formatting_and_manifest_order() -> None:
    compact = parse_site_manifest(
        b'{"version":1,"pages":[{"source":"docs/b.md","route":"/b/"},'
        b'{"source":"docs/a.md","route":"/a/"}]}'
    )
    formatted = parse_site_manifest(
        json.dumps(
            {
                "pages": [
                    {"route": "/a", "source": "docs/a.md"},
                    {"route": "/b", "source": "docs/b.md"},
                ],
                "version": 1,
            },
            indent=4,
        ).encode()
    )
    first = records_from_manifest(compact, ["docs/a.md", "docs/b.md"], content_root="docs")
    second = records_from_manifest(formatted, ["docs/a.md", "docs/b.md"], content_root="docs")

    assert [(record.source, record.digest) for record in first] == [
        (record.source, record.digest) for record in second
    ]
