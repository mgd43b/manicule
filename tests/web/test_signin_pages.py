"""The browser surface where people sign in: the sign-in page, the people page, the frame.

Asserted against what the routes rendered, through the production application, with the
identity provider played by the same fake the API suites use. What the pages must do:

* **The sign-in page offers exactly the sign-ins this workspace has**, and where nobody signs in
  it says how the installation authenticates instead — in the bare frame, because a person
  arriving there has no credential and a navigation would describe the installation to them.
* **The people page is an administrator's**, like the API route it reads, and its controls call
  that API rather than deciding anything.
* **The frame names a signed-in person and offers a sign-out** — a plain form, so it works with
  the script off — and names nobody when there is no session to end.
"""

from __future__ import annotations

from manicule.web.rendering import UI_POLICY
from tests.api.identity_provider import (
    FakeIdentityProvider,
    client_with,
    github,
    google,
    oauth_backend,
    sign_in,
)
from tests.web.support import backend_with_a_document, client_for

OK = 200
FORBIDDEN = 403
SEE_OTHER = 303


def test_the_sign_in_page_links_every_sign_in_this_workspace_has_and_no_other() -> None:
    backend = oauth_backend(google(), github(workspace="elsewhere"))
    with client_with(backend, FakeIdentityProvider()) as client:
        page = client.get("/ui/login")
    assert page.status_code == OK
    assert page.headers["content-security-policy"] == UI_POLICY
    assert 'href="/auth/login/google"' in page.text
    assert "Sign in with Google" in page.text
    assert "/auth/login/github" not in page.text, "offered a sign-in for another workspace"
    assert "data-navigation" not in page.text, "a person with no credential was shown the areas"


def test_where_nobody_signs_in_the_page_says_how_this_installation_authenticates() -> None:
    """Not a sign-in button the login routes would refuse."""
    open_backend, _ = backend_with_a_document()
    keyed, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    with client_for(open_backend) as client:
        open_page = client.get("/ui/login")
    with client_for(keyed) as client:
        keyed_page = client.get("/ui/login")
    assert open_page.status_code == OK
    assert "asks for no credential" in open_page.text
    assert "/auth/login/" not in open_page.text
    assert keyed_page.status_code == OK
    assert "API keys" in keyed_page.text
    assert "/auth/login/" not in keyed_page.text


def test_the_sign_in_page_says_who_is_already_signed_in() -> None:
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        page = client.get("/ui/login")
    assert "You are signed in as" in page.text
    assert "Alice" in page.text


def test_the_frame_names_the_signed_in_person_and_offers_a_sign_out_form() -> None:
    """A plain form posting to ``/auth/logout``, which works with the script off."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        page = client.get("/ui/documents")
        out = client.post(
            "/auth/logout",
            headers={"Accept": "text/html", "Sec-Fetch-Site": "same-origin"},
            follow_redirects=False,
        )
        after = client.get("/ui/documents")
    assert page.status_code == OK
    assert 'title="Signed in as">Alice<' in page.text
    assert '<form class="signout" method="post" action="/auth/logout">' in page.text
    assert page.headers["content-security-policy"] == UI_POLICY
    assert out.status_code == SEE_OTHER
    assert after.status_code == 401, "the page still rendered after signing out"


def test_the_frame_offers_no_sign_out_where_there_is_no_session() -> None:
    """The operator at a loopback socket has nothing to sign out of."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        page = client.get("/ui/documents")
    assert page.status_code == OK
    assert "/auth/logout" not in page.text
    assert "Signed in as" not in page.text


def test_the_people_page_lists_members_with_controls_that_call_the_api() -> None:
    backend = oauth_backend(google(role="admin"))
    backend.users_.add_member("bob", role="member", email="bob@example.org")
    backend.users_.add_member("carol", role="viewer", email="carol@example.org", disabled=True)
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        page = client.get("/ui/users")
    assert page.status_code == OK
    assert page.headers["content-security-policy"] == UI_POLICY
    assert 'data-user-role="bob"' in page.text
    assert 'data-user-disable="bob"' in page.text
    assert 'data-user-sign-out="bob"' in page.text
    assert 'data-user-enable="carol"' in page.text
    assert 'data-user-disable="carol"' not in page.text
    assert ">you<" in page.text, "the reader's own row is not marked"
    assert 'href="/ui/users"' in page.text, "the People area is not in the navigation"
    assert "<script>" not in page.text


def test_a_member_is_refused_the_people_page_as_a_page() -> None:
    """The floor its API route takes, rendered for a browser rather than as an envelope."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        page = client.get("/ui/users")
    assert page.status_code == FORBIDDEN
    assert "text/html" in page.headers["content-type"]
    assert "you have &#39;member&#39;" in page.text, "the page did not carry the API's refusal"
    assert "data-user-role" not in page.text


def test_the_people_page_says_nobody_signs_in_where_nobody_does() -> None:
    backend, _ = backend_with_a_document(security={"auth": {"mode": "none"}})
    with client_for(backend) as client:
        page = client.get("/ui/users")
    assert page.status_code == OK
    assert "nobody signs in here" in page.text
    assert "Nobody has signed in to this workspace yet." in page.text
