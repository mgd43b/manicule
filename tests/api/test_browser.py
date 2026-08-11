"""What a browser is told: cross-origin, framing, sniffing, and the widget.

The widget makes this surface cross-origin, which brings in three failure modes that no other
part of manicule has.

**A permissive CORS policy** turns a private document index into something any page a user
visits can read with whatever the browser will attach. So there is no wildcard, the middleware
is absent unless origins are configured, and configuration refuses ``*`` outright.

**Framing** turns a chat box into a clickjacking target. So ``frame-ancestors`` is served on
every response, and it is ``'none'`` until an operator names the pages that may embed it.

**Injection into the widget's rendering path** would make any document in the corpus a script
into every page that embeds it. The widget builds DOM and never markup, and that is asserted
against the served script rather than against the source file — because what a page executes
is what the route returned.
"""

from __future__ import annotations

import pytest

from manicule.api.app import frame_policy
from manicule.api.widget import WIDGET_SCRIPT
from manicule.config.settings import Settings
from tests.api.support import backend_with_a_document, client_for

EMBEDDER = "https://docs.example.com"


# --- cross-origin -----------------------------------------------------------------------------


def test_no_cross_origin_header_is_sent_when_no_origins_are_configured() -> None:
    """The default is same-origin, and it is an *absence* rather than a narrow allowlist.

    A header that echoed the request's origin — the usual accident — would be a wildcard
    written the long way.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/api/v1/documents", headers={"Origin": EMBEDDER})
    assert "access-control-allow-origin" not in {key.lower() for key in response.headers}


def test_a_configured_origin_is_permitted_and_another_one_is_not() -> None:
    """The positive control and its negative, in one test.

    Without the positive half, "sends no CORS header" would pass for a surface that can never
    be embedded at all — which is not the feature.
    """
    backend, _ = backend_with_a_document(security={"transport": {"allowed_origins": [EMBEDDER]}})
    with client_for(backend) as client:
        allowed = client.get("/api/v1/documents", headers={"Origin": EMBEDDER})
        refused = client.get("/api/v1/documents", headers={"Origin": "https://not-listed.example"})
    assert allowed.headers["access-control-allow-origin"] == EMBEDDER
    assert "access-control-allow-origin" not in {key.lower() for key in refused.headers}


def test_a_wildcard_origin_is_refused_by_configuration() -> None:
    """Refused rather than accepted-and-narrowed.

    An operator who writes ``*`` has asked for "any page on the internet may read this index";
    the honest response is to say that is not offered, not to silently do something else.
    """
    from pydantic import ValidationError  # noqa: PLC0415 - only this test names one

    with pytest.raises(ValidationError, match=r"may not contain"):
        Settings(security={"transport": {"allowed_origins": ["*"]}})  # pyright: ignore[reportArgumentType]


def test_credentials_are_never_permitted_cross_origin() -> None:
    """A key is presented on every request; there is no cookie to attach.

    With ``allow-credentials`` on, a browser would attach cookies to cross-origin calls, which
    is the ingredient a CSRF needs.
    """
    backend, _ = backend_with_a_document(security={"transport": {"allowed_origins": [EMBEDDER]}})
    with client_for(backend) as client:
        response = client.get("/api/v1/documents", headers={"Origin": EMBEDDER})
    assert "access-control-allow-credentials" not in {key.lower() for key in response.headers}


@pytest.mark.parametrize(
    "entry",
    [
        "docs.example.com",
        "ftp://docs.example.com",
        "https://docs.example.com/embed",
        "https://docs.example.com/",
        "https://docs.example.com?tenant=1",
        "https://docs.example.com#frag",
        "https://",
    ],
)
def test_something_that_is_not_exactly_an_origin_is_refused(entry: str) -> None:
    """A browser's ``Origin`` header is scheme, host and port. Nothing else can ever match one.

    Each of these would be accepted by a validator that only looked at the *path*, and each
    would then silently disable CORS for the origin the operator meant — configuration that
    reads as in force and is not, with the symptom appearing as an error on somebody else's
    page rather than here.
    """
    from pydantic import ValidationError  # noqa: PLC0415 - only this test names one

    with pytest.raises(ValidationError, match="is not an origin"):
        Settings(security={"transport": {"allowed_origins": [entry]}})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "entry",
    ["https://docs.example.com", "http://localhost:5173", "https://docs.example.com:8443"],
)
def test_a_real_origin_is_accepted(entry: str) -> None:
    """The positive control for the list above, including an explicit port."""
    settings = Settings(security={"transport": {"allowed_origins": [entry]}})  # pyright: ignore[reportArgumentType]
    assert settings.security.transport.allowed_origins == (entry,)


# --- framing and sniffing ---------------------------------------------------------------------


def test_framing_is_refused_by_default() -> None:
    """``frame-ancestors 'none'`` until somebody names the pages that may embed the widget."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/api/v1/documents")
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_a_configured_embedding_domain_is_named_in_the_policy() -> None:
    """The positive control: the refusal is a policy, not a constant."""
    backend, _ = backend_with_a_document(
        security={"transport": {"widget_allowed_domains": [EMBEDDER]}}
    )
    with client_for(backend) as client:
        response = client.get("/api/v1/documents")
    assert f"frame-ancestors {EMBEDDER}" in response.headers["content-security-policy"]
    assert "'none'" not in response.headers["content-security-policy"].split("frame-ancestors")[1]


def test_the_frame_policy_is_a_refusal_when_the_list_is_empty() -> None:
    """Asserted on the function as well, because the default is the whole safety property."""
    assert "frame-ancestors 'none'" in frame_policy(())
    assert frame_policy((EMBEDDER,)).endswith(EMBEDDER)


@pytest.mark.parametrize(
    "path", ["/api/v1/documents", "/healthz", "/shared/nothing", "/widget/widget.js"]
)
def test_every_response_refuses_content_sniffing(path: str) -> None:
    """Applied in middleware, so a route added later cannot be the one that forgot."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get(path)
    assert response.headers["x-content-type-options"] == "nosniff"


def test_no_referrer_is_sent_from_any_response() -> None:
    """A URL here can be a share token. It has no business in the next request's ``Referer``."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/shared/some-token-shaped-thing")
    assert response.headers["referrer-policy"] == "no-referrer"


# --- the widget ---------------------------------------------------------------------------------


def test_the_widget_is_served_as_javascript() -> None:
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/widget/widget.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_the_widget_never_writes_markup() -> None:
    """Asserted against **what the route returned**, not against the source file.

    An answer is model output over indexed documents. Treating it as markup would make any
    document in the corpus a script-injection vector into every page that embeds the widget,
    so the script builds DOM through ``textContent`` and nothing else.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        served = client.get("/widget/widget.js").text
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in served, f"the widget uses {forbidden}"
    assert "textContent" in served, "the widget renders nothing, so this test proves nothing"


def test_the_widget_reflects_nothing_from_a_request() -> None:
    """The script is a constant. There is no request value anywhere in what is executed.

    Driven with a query string, a header and a path fragment that would each be obvious in the
    output if anything were interpolated.
    """
    backend, _ = backend_with_a_document()
    marker = "PROBE-9d1f"
    with client_for(backend) as client:
        response = client.get(
            f"/widget/widget.js?{marker}=1",
            headers={"X-Probe": marker, "Referer": f"https://x.example/{marker}"},
        )
    assert marker not in response.text
    assert response.text == WIDGET_SCRIPT


def test_the_widget_keeps_its_key_out_of_storage() -> None:
    """A widget key is as public as the page. It should not travel further than that page."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        served = client.get("/widget/widget.js").text
    for forbidden in ("localStorage", "sessionStorage", "document.cookie"):
        assert forbidden not in served, f"the widget writes its key to {forbidden}"


def test_the_widget_sends_its_key_as_a_header_rather_than_in_a_url() -> None:
    """A credential in a URL is one in the access log and in every ``Referer`` that follows."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        served = client.get("/widget/widget.js").text
    assert '"X-API-Key"' in served
    assert "?key=" not in served
    assert "api_key=" not in served


def test_the_demo_page_embeds_the_widget_and_reflects_nothing() -> None:
    """Something to look at before an operator starts configuring origins."""
    backend, _ = backend_with_a_document()
    marker = "PROBE-4b2a"
    with client_for(backend) as client:
        response = client.get(f"/widget?{marker}=1", headers={"X-Probe": marker})
    assert response.status_code == 200
    assert "/widget/widget.js" in response.text
    assert marker not in response.text


def test_the_demo_page_may_actually_load_the_script_it_embeds() -> None:
    """The application-wide policy is ``default-src 'none'``, which a browser applies to a
    *document* and which refuses its ``<script src>``.

    So the one route that returns HTML states its own policy. Without this the demo page is a
    page that cannot work, and a test asserting only that the markup mentions the script would
    never see it — which is exactly what happened before this assertion existed.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        policy = client.get("/widget").headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "connect-src 'self'" in policy
    assert "default-src 'none'" in policy


def test_the_demo_page_still_refuses_framing_and_inline_script() -> None:
    """Its own policy is narrower, not looser.

    ``script-src 'self'`` with no ``'unsafe-inline'`` means the page cannot execute anything
    this installation did not serve, and framing is refused exactly as everywhere else.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        policy = client.get("/widget").headers["content-security-policy"]
    assert "frame-ancestors 'none'" in policy
    assert "script-src 'self'" in policy
    script = policy.split("script-src")[1].split(";")[0]
    assert "unsafe-inline" not in script
    assert "unsafe-eval" not in script


def test_the_json_surface_keeps_the_strict_policy() -> None:
    """The exception is one page, not a loosening of the default."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        policy = client.get("/api/v1/documents").headers["content-security-policy"]
    assert policy == "default-src 'none'; frame-ancestors 'none'"
