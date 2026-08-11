"""The admin group reports things that were actually recorded.

An admin page that always answers ``[]`` looks exactly like a healthy installation nobody has
used yet, so each read here is asserted against a write that produced it. That write is the
**service's**, not a route's: telemetry recorded only by whichever surface remembered to record
it describes that surface's traffic rather than the installation's.

The audit trail additionally has to distinguish "nothing happened" from "nothing was recorded",
because an operator reading an empty one needs to know which they have.
"""

from __future__ import annotations

from tests.api.support import backend_with_a_document, client_for, envelope

AUDITED = {"security": {"audit": {"enabled": True}}}


def test_a_search_is_recorded_and_the_admin_page_reports_it() -> None:
    """The write is the service's, so the row exists whichever surface asked the question."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        assert client.get("/api/v1/search", params={"q": "retry policy"}).status_code == 200
        page = envelope(client.get("/api/v1/admin/query-logs"))["data"]
    assert page["total"] == 1
    assert page["entries"][0]["query"] == "retry policy"
    assert page["entries"][0]["profile"] == "balanced"


def test_an_answered_question_is_recorded_too() -> None:
    """``ask`` retrieves, so it produces a retrieval row exactly as ``search`` does."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        assert client.post("/api/v1/chat", json={"question": "does it retry"}).status_code == 200
        page = envelope(client.get("/api/v1/admin/query-logs"))["data"]
    assert page["total"] == 1
    assert page["entries"][0]["query"] == "does it retry"


def test_query_logs_report_a_chunk_count_rather_than_chunk_ids() -> None:
    """Corpus structure has no business travelling with a telemetry listing."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        client.get("/api/v1/search", params={"q": "retry"})
        response = client.get("/api/v1/admin/query-logs")
    entry = envelope(response)["data"]["entries"][0]
    assert set(entry) == {
        "id",
        "query",
        "profile",
        "chunks",
        "confidence",
        "elapsed_ms",
        "created_at",
    }
    assert isinstance(entry["chunks"], int)


def test_a_page_of_query_logs_is_bounded_and_carries_the_total() -> None:
    """A client cannot page without knowing how many rows there are."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        for number in range(3):
            client.get("/api/v1/search", params={"q": f"query {number}"})
        page = envelope(client.get("/api/v1/admin/query-logs", params={"limit": 2}))["data"]
    assert page["total"] == 3
    assert page["count"] == 2
    # Newest first, so a page of two from three is the last two asked.
    assert page["entries"][0]["query"] == "query 2"


def test_an_audit_trail_that_is_switched_off_says_so() -> None:
    """ "Nothing happened" and "nothing was recorded" are different answers.

    Reported as a field rather than implied by an empty list, because the empty list is what
    both of them look like.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        client.post("/api/v1/conversations", json={"title": "a thread"})
        page = envelope(client.get("/api/v1/admin/audit-logs"))["data"]
    assert page["enabled"] is False
    assert page["entries"] == []


def test_switching_auditing_on_records_the_security_relevant_events() -> None:
    """The positive control, and the events chosen: minting a link, revoking one, deleting.

    Each of those changes who can read something. An audit trail of everything else and not
    these would be a log rather than an audit trail.
    """
    backend, _ = backend_with_a_document(**AUDITED)
    backend.conversations_.seed("conv_1")
    with client_for(backend) as client:
        client.post("/api/v1/conversations/conv_1/share", json={})
        client.delete("/api/v1/conversations/conv_1/share")
        client.delete("/api/v1/conversations/conv_1")
        page = envelope(client.get("/api/v1/admin/audit-logs"))["data"]
    assert page["enabled"] is True
    recorded = [entry["event_type"] for entry in page["entries"]]
    assert recorded == [
        "conversation.deleted",
        "conversation.unshared",
        "conversation.shared",
    ]


def test_an_audit_entry_never_carries_the_token_it_recorded() -> None:
    """The record of a capability being minted must not be a copy of the capability."""
    backend, _ = backend_with_a_document(**AUDITED)
    backend.conversations_.seed("conv_1")
    with client_for(backend) as client:
        minted = envelope(client.post("/api/v1/conversations/conv_1/share", json={}))
        token = str(minted["data"]["token"])
        response = client.get("/api/v1/admin/audit-logs")
    assert token not in response.text


def test_minting_an_api_key_is_audited_without_the_secret() -> None:
    """A key is an identity. Its creation is worth recording; its secret is not."""
    backend, _ = backend_with_a_document(**AUDITED)
    with client_for(backend) as client:
        issued = envelope(client.post("/api/v1/auth/keys", json={"name": "widget"}))
        secret = str(issued["data"]["secret"])
        response = client.get("/api/v1/admin/audit-logs", params={"event_type": "api_key.created"})
    page = envelope(response)["data"]
    assert [entry["event_type"] for entry in page["entries"]] == ["api_key.created"]
    assert page["entries"][0]["details"]["name"] == "widget"
    assert secret not in response.text


def test_search_quality_says_plainly_that_nothing_has_been_measured() -> None:
    """``available: false`` is the truth, and is not the same as a score of zero.

    The route reports what the evaluation harness recorded. With nothing recorded there is no
    number, and inventing one is the failure that whole subsystem exists to prevent.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        page = envelope(client.get("/api/v1/admin/search-quality"))["data"]
    assert page["available"] is False
    assert page["is_evidence"] is False
    assert page["records"] == 0
    assert "judged" in page["caveat"] or "judgements" in page["caveat"]


def test_plugin_health_reports_unknown_for_a_component_nothing_has_built() -> None:
    """A plugin nothing has asked for has not been proved healthy, and saying it is would be a
    diagnostic that measured nothing."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        page = envelope(client.get("/api/v1/admin/plugins"))["data"]
    assert page["count"] == 0
    assert page["plugins"] == []
