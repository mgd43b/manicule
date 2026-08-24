"""Adversarial URL, redirect and address policy for every future crawler request."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from manicule.connectors.crawler_security import (
    CrawlerUrlPolicy,
    TrailingSlashPolicy,
    bounded_response_headers,
)
from manicule.connectors.errors import (
    CrawlerAddressError,
    CrawlerPolicyError,
    CrawlerRedirectError,
    CrawlerScopeError,
)


def policy(**changes: object) -> CrawlerUrlPolicy:
    values: dict[str, object] = {
        "allowed_origins": ("https://docs.example.test",),
        "allowed_path_prefixes": ("/docs/",),
    }
    values.update(changes)
    return CrawlerUrlPolicy(**values)  # pyright: ignore[reportArgumentType]


def test_equivalent_safe_url_spellings_have_one_identity() -> None:
    configured = policy(allowed_query_keys=("lang",))

    first = configured.normalize(
        "HTTPS://DOCS.EXAMPLE.TEST:443/docs/one/../Guide/%7eintro?lang=en&utm_source=x#part"
    )
    second = configured.normalize("https://docs.example.test/docs/Guide/~intro?lang=en")

    assert first == second
    assert first.url == "https://docs.example.test/docs/Guide/~intro?lang=en"


def test_unicode_hosts_and_paths_are_canonicalized() -> None:
    configured = policy(
        allowed_origins=("https://bücher.example",),
        allowed_path_prefixes=("/",),
    )

    found = configured.normalize("HTTPS://BÜCHER.EXAMPLE/Überblick")

    assert found.url == "https://xn--bcher-kva.example/%C3%9Cberblick"
    assert found.origin == "https://xn--bcher-kva.example"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://docs.example.test/docs/a",
        "https://user:secret@docs.example.test/docs/a",
        "https://docs.example.test:bad/docs/a",
        "https://docs.example.test:0/docs/a",
        "https://bad_host.example/docs/a",
        "https://docs.example.test/docs/a%2",
        "https://docs.example.test/docs/a\\b",
        " https://docs.example.test/docs/a",
        "https://docs.example.test/docs/a\x00",
        "https://[fe80::1%25en0]/docs/a",
    ],
    ids=(
        "scheme",
        "userinfo",
        "port",
        "zero-port",
        "host-label",
        "percent",
        "backslash",
        "padding",
        "control",
        "ipv6-zone",
    ),
)
def test_malformed_or_credential_bearing_urls_are_refused_without_reflection(url: str) -> None:
    with pytest.raises(CrawlerPolicyError) as raised:
        policy().normalize(url)

    assert "secret" not in str(raised.value)


def test_scope_is_checked_after_dot_segment_normalization() -> None:
    with pytest.raises(CrawlerScopeError):
        policy().normalize("https://docs.example.test/docs/../admin")


def test_encoded_separators_and_traversal_are_not_decoded_into_a_new_path() -> None:
    found = policy().normalize("https://docs.example.test/docs/%2e%2e%2fadmin/%2Fliteral")

    assert found.path == "/docs/%2E%2E%2Fadmin/%2Fliteral"


@pytest.mark.parametrize(
    "url",
    [
        "https://elsewhere.example/docs/a",
        "https://docs.example.test/admin",
        "//elsewhere.example/docs/a",
    ],
)
def test_cross_origin_and_path_escaping_urls_are_refused(url: str) -> None:
    with pytest.raises(CrawlerScopeError):
        policy().normalize(url, base="https://docs.example.test/docs/start")


def test_path_prefix_comparison_uses_segment_boundaries() -> None:
    assert policy().normalize("https://docs.example.test/docs").path == "/docs"
    with pytest.raises(CrawlerScopeError):
        policy().normalize("https://docs.example.test/documents")


@pytest.mark.parametrize(
    ("trailing", "expected"),
    [
        (TrailingSlashPolicy.PRESERVE, "/docs/page"),
        (TrailingSlashPolicy.DIRECTORY, "/docs/page/"),
        (TrailingSlashPolicy.STRIP, "/docs/page"),
    ],
)
def test_trailing_slash_policy_is_explicit(trailing: TrailingSlashPolicy, expected: str) -> None:
    assert (
        policy(trailing_slash=trailing).normalize("https://docs.example.test/docs/page").path
        == expected
    )


def test_declared_tracking_queries_drop_and_allowed_queries_remain_distinct() -> None:
    configured = policy(allowed_query_keys=("lang",))
    english = configured.normalize("https://docs.example.test/docs/a?utm_medium=email&lang=en")
    french = configured.normalize("https://docs.example.test/docs/a?lang=fr")

    assert english.url.endswith("?lang=en")
    assert english != french


@pytest.mark.parametrize(
    "query",
    ["search=x", "lang=en&trap=x", "flag", "=blank"],
)
def test_undeclared_query_keys_are_crawler_traps(query: str) -> None:
    with pytest.raises(CrawlerPolicyError):
        policy(allowed_query_keys=("lang",)).normalize(f"https://docs.example.test/docs/a?{query}")


def test_fragments_never_create_a_second_page_identity() -> None:
    configured = policy()
    assert configured.normalize("https://docs.example.test/docs/a#one") == configured.normalize(
        "https://docs.example.test/docs/a#two"
    )


def test_relative_links_use_the_same_normalization_and_scope_path() -> None:
    configured = policy()
    assert (
        configured.normalize("../next", base="https://docs.example.test/docs/chapter/current").url
        == "https://docs.example.test/docs/next"
    )


def test_an_out_of_scope_canonical_is_diagnostic_evidence_not_authority() -> None:
    decision = policy().canonical(
        "https://docs.example.test/docs/a", "https://elsewhere.example/stolen"
    )

    assert not decision.accepted
    assert decision.value is None
    assert decision.reason == "CrawlerScopeError"
    assert "elsewhere" not in decision.reason


def test_an_in_scope_relative_canonical_is_accepted() -> None:
    decision = policy().canonical("https://docs.example.test/docs/a", "/docs/canonical#section")

    assert decision.accepted
    assert decision.value is not None
    assert decision.value.url == "https://docs.example.test/docs/canonical"


def test_https_downgrade_is_refused_even_when_both_origins_are_allowed() -> None:
    configured = policy(allowed_origins=("https://docs.example.test", "http://docs.example.test"))
    with pytest.raises(CrawlerRedirectError, match="downgrade"):
        configured.redirect(
            "https://docs.example.test/docs/a",
            "http://docs.example.test/docs/a",
            hop=1,
        )


def test_redirect_count_is_bounded_before_a_request_can_be_planned() -> None:
    with pytest.raises(CrawlerRedirectError, match="bound"):
        policy(max_redirects=2).redirect("https://docs.example.test/docs/a", "/docs/b", hop=3)


def test_cross_origin_redirects_strip_secret_headers() -> None:
    configured = policy(
        allowed_origins=("https://docs.example.test", "https://cdn.example.test"),
        allowed_path_prefixes=("/",),
    )
    decision = configured.redirect(
        "https://docs.example.test/docs/a", "https://cdn.example.test/page", hop=1
    )

    assert not decision.forward_sensitive_headers
    assert configured.redirected_headers(
        {
            "Authorization": "sentinel",
            "Cookie": "sentinel",
            "X-Api-Key": "sentinel",
            "Accept": "text/html",
        },
        decision,
    ) == {"Accept": "text/html"}


def test_same_origin_redirects_may_keep_origin_bound_headers() -> None:
    configured = policy()
    decision = configured.redirect("https://docs.example.test/docs/a", "/docs/b", hop=1)
    headers = {"Authorization": "sentinel"}
    assert decision.forward_sensitive_headers
    assert configured.redirected_headers(headers, decision) == headers


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "0.0.0.0",  # noqa: S104 - adversarial destination fixture
        "224.0.0.1",
        "192.0.2.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::",
        "ff02::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
        "2002:7f00:1::",
        "64:ff9b::7f00:1",
    ],
)
async def test_non_public_ipv4_and_ipv6_answers_are_refused(address: str) -> None:
    async def resolve(hostname: str, port: int) -> Sequence[str]:
        del hostname, port
        return (address,)

    with pytest.raises(CrawlerAddressError):
        await policy().connection_plan("https://docs.example.test/docs/a", resolve)


async def test_all_dns_answers_must_be_safe_and_are_deduplicated() -> None:
    async def safe(hostname: str, port: int) -> Sequence[str]:
        assert (hostname, port) == ("docs.example.test", 443)
        return ("8.8.8.8", "2606:4700:4700::1111", "8.8.8.8")

    plan = await policy().connection_plan("https://docs.example.test/docs/a", safe)
    assert plan.addresses == ("8.8.8.8", "2606:4700:4700::1111")
    assert plan.port == 443
    assert plan.tls_server_name == "docs.example.test"
    assert plan.verify_tls

    async def mixed(hostname: str, port: int) -> Sequence[str]:
        del hostname, port
        return ("8.8.8.8", "127.0.0.1")

    with pytest.raises(CrawlerAddressError):
        await policy().connection_plan("https://docs.example.test/docs/a", mixed)


async def test_numeric_host_spellings_are_still_subject_to_address_policy() -> None:
    configured = policy(allowed_origins=("https://127.0.0.1",), allowed_path_prefixes=("/",))

    async def unreachable(hostname: str, port: int) -> Sequence[str]:
        del hostname, port
        raise AssertionError("literal IP hosts must not be resolved again")

    with pytest.raises(CrawlerAddressError):
        await configured.connection_plan("https://127.0.0.1/a", unreachable)


async def test_rebinding_peer_must_equal_one_of_the_prevalidated_dial_targets() -> None:
    async def resolve(hostname: str, port: int) -> Sequence[str]:
        del hostname, port
        return ("8.8.8.8",)

    plan = await policy().connection_plan("https://docs.example.test/docs/a", resolve)
    assert plan.authorize_peer("8.8.8.8") == "8.8.8.8"
    with pytest.raises(CrawlerAddressError, match="globally routable"):
        plan.authorize_peer("127.0.0.1")
    with pytest.raises(CrawlerAddressError, match="changed"):
        plan.authorize_peer("1.1.1.1")


async def test_legacy_numeric_hostname_cannot_bypass_resolution_checks() -> None:
    configured = policy(allowed_origins=("https://2130706433",), allowed_path_prefixes=("/",))

    async def resolve(hostname: str, port: int) -> Sequence[str]:
        assert (hostname, port) == ("2130706433", 443)
        return ("127.0.0.1",)

    with pytest.raises(CrawlerAddressError):
        await configured.connection_plan("https://2130706433/a", resolve)


def test_response_headers_are_bounded_and_cannot_inject_lines() -> None:
    assert bounded_response_headers([("Content-Type", " text/html ")]) == {
        "content-type": "text/html"
    }
    with pytest.raises(CrawlerPolicyError, match="bound"):
        bounded_response_headers([("X-Large", "x" * 100)], max_bytes=32)
    with pytest.raises(CrawlerPolicyError, match="malformed"):
        bounded_response_headers([("Location", "https://safe.example\r\nX-Evil: yes")])
    with pytest.raises(CrawlerPolicyError, match="malformed"):
        bounded_response_headers([("Content-Length", "1"), ("content-length", "2")])
