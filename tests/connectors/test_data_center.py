"""Server and Data Center, over a source that refuses the field Data Center refuses.

The failure this file exists for is not the HTTP 400. A 400 announces itself, names the field,
and gets fixed the same afternoon. It is that **there are eight places this connector builds a
query**, and fixing the one somebody happened to run leaves the other seven to fail later, one
at a time, against a live wiki — reconciliation next week, an include macro next month.

So the fake here rejects *any* CQL containing ``status``, on every request, and each path below
is driven end to end rather than inspected. A call site nobody remembered fails in the test that
names it rather than at 3am. That is also why these are separate tests instead of one sync: a
single run that happened to exercise four paths would go green with the other four broken.

**The other half is quieter and has no 400 to announce it.** A connector scoped to two spaces
that reads the account's whole space catalog succeeds, reports success, and pays for every
entitlement the credential happens to have. Those tests count requests, because the symptom is
cost and correctness of scope rather than an error.
"""

from __future__ import annotations

import httpx
import pytest

from manicule.connectors import ConnectorError, SessionExpiredError
from manicule.connectors.confluence import ANCESTORS
from manicule.core.provenance import Provenance
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE
from tests.connectors.fake_confluence import (
    FakeAttachment,
    FakeConfluence,
    FakePage,
    storage_include,
)
from tests.connectors.support import cloud_config, connected, drain, ids, server_config

CLOUD = "https://wiki.example.test/wiki"
SERVER = "https://wiki.example.test/confluence"

ROOT = "100100"
CHILD = "100110"


def _pages() -> list[FakePage]:
    return [
        FakePage(id=ROOT, title="Architecture", space="ENG", when="2026-08-09T14:30:00.000+01:00"),
        FakePage(id=CHILD, title="Data Model", space="ENG", parent=ROOT),
        FakePage(id="100300", title="Marketing Plan", space="ENG"),
    ]


def _server(**overrides: object) -> FakeConfluence:
    """A Data Center instance that rejects ``status`` exactly as the real resource does."""
    settings: dict[str, object] = {
        "base_url": SERVER,
        "pages": _pages(),
        "spaces": {"ENG": "Engineering"},
        "page_size": 10,
    }
    settings.update(overrides)
    instance = FakeConfluence(**settings)  # type: ignore[arg-type]
    instance.rejects_status_field = True
    return instance


def _cloud(**overrides: object) -> FakeConfluence:
    settings: dict[str, object] = {
        "base_url": CLOUD,
        "pages": _pages(),
        "spaces": {"ENG": "Engineering"},
        "page_size": 10,
    }
    settings.update(overrides)
    return FakeConfluence(**settings)  # type: ignore[arg-type]


def _queries(instance: FakeConfluence) -> list[str]:
    return instance.queries()


# --- one test per query-building call site ---------------------------------------------------


async def test_whole_space_discovery_runs_against_data_center() -> None:
    """Call site 1: full discovery, pages and attachments in one query."""
    instance = _server()
    connector = await connected(instance, server_config(SERVER, spaces=("ENG",), page_size=10))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert sorted(ids(found)) == [ROOT, CHILD, "100300"]
    assert _queries(instance), "a query must actually have been sent"
    assert all("status" not in query for query in _queries(instance))


async def test_incremental_discovery_runs_against_data_center() -> None:
    """Call site 2: the same query with a watermark clause, which must still omit status."""
    instance = _server()
    config = server_config(SERVER, spaces=("ENG",), page_size=10)
    connector = await connected(instance, config)
    try:
        await drain(connector.discover(None))
        watermark = connector.watermark
        await drain(connector.discover(watermark))
    finally:
        await connector.teardown()

    incremental = [query for query in _queries(instance) if "lastmodified >=" in query]
    assert incremental, "the second run must have carried a watermark"
    assert all("status" not in query for query in incremental)


async def test_page_tree_discovery_runs_against_data_center() -> None:
    """Call site 3: subtree page discovery, which adds an ancestor clause and no status."""
    instance = _server()
    connector = await connected(
        instance, server_config(SERVER, spaces=("ENG",), root_page_ids=(ROOT,), page_size=10)
    )
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert sorted(ids(found)) == [ROOT, CHILD]
    tree = [query for query in _queries(instance) if "ancestor" in query]
    assert tree
    assert all("status" not in query for query in tree)


async def test_subtree_membership_enumeration_runs_against_data_center() -> None:
    """Call site 4: ``Subtree.members``, which builds its own query in another module.

    Reached only with attachments on, and only after page discovery has already succeeded —
    which is exactly the shape that makes a missed call site look like "attachments are
    broken" a fortnight after the rest of the connector was signed off.
    """
    instance = _server(
        attachments=[
            FakeAttachment(
                id="att-1", title="d.pdf", space="ENG", page_id=CHILD, page_title="Data Model"
            )
        ]
    )
    connector = await connected(
        instance,
        server_config(
            SERVER, spaces=("ENG",), root_page_ids=(ROOT,), include_attachments=True, page_size=10
        ),
    )
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert "att-1" in ids(found)
    assert all("status" not in query for query in _queries(instance))


async def test_attachment_discovery_runs_against_data_center() -> None:
    """Call site 5: the space-wide attachment query a scoped run sends alongside its pages."""
    instance = _server(
        attachments=[
            FakeAttachment(
                id="att-1", title="d.pdf", space="ENG", page_id=CHILD, page_title="Data Model"
            )
        ]
    )
    connector = await connected(
        instance,
        server_config(SERVER, spaces=("ENG",), include_attachments=True, page_size=10),
    )
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert "att-1" in ids(found)
    attachments = [query for query in _queries(instance) if "attachment" in query]
    assert attachments
    assert all("status" not in query for query in attachments)


async def test_reconciliation_runs_against_data_center() -> None:
    """Call site 6: the unordered full enumeration deletion detection depends on.

    The one where a missed call site is worst. Reconciliation failing is a pass that reports an
    error rather than a diff, so nothing is deleted — but nothing is ever *detected* as deleted
    either, and the index serves removed pages until somebody notices.
    """
    instance = _server()
    connector = await connected(instance, server_config(SERVER, spaces=("ENG",), page_size=10))
    try:
        existing = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    assert sorted(existing) == [ROOT, CHILD, "100300"]
    assert all("status" not in query for query in _queries(instance))


async def test_attachment_reconciliation_runs_against_data_center() -> None:
    """Call site 7: the attachment half of a scoped reconciliation."""
    instance = _server(
        attachments=[
            FakeAttachment(
                id="att-1", title="d.pdf", space="ENG", page_id=CHILD, page_title="Data Model"
            )
        ]
    )
    connector = await connected(
        instance,
        server_config(
            SERVER, spaces=("ENG",), root_page_ids=(ROOT,), include_attachments=True, page_size=10
        ),
    )
    try:
        existing = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    assert sorted(existing) == [ROOT, CHILD, "att-1"]
    assert all("status" not in query for query in _queries(instance))


async def test_include_macro_title_lookup_runs_against_data_center() -> None:
    """Call site 8: ``title_query``, reached only while expanding an include macro.

    The last one anybody would run by hand, and the one whose failure is invisible from both
    ends — the page indexes, and the content the macro renders is missing from it.
    """
    instance = _server(
        pages=[
            FakePage(
                id=ROOT,
                title="Architecture",
                space="ENG",
                storage=storage_include("intro", title="Data Model"),
            ),
            FakePage(id=CHILD, title="Data Model", space="ENG", storage="<p>the model</p>"),
        ]
    )
    connector = await connected(instance, server_config(SERVER, spaces=("ENG",), page_size=10))
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == ROOT)
        raw = await connector.fetch(ref)
    finally:
        await connector.teardown()

    assert "the model" in raw.as_text(), "the include was expanded"
    titles = [query for query in _queries(instance) if "title =" in query]
    assert titles, "the title lookup must have been sent"
    assert all("status" not in query for query in titles)


async def test_a_forgotten_call_site_would_fail_rather_than_pass_quietly() -> None:
    """The fake's own guard, exercised, so the tests above are known to be able to fail.

    Every assertion in this file rests on the instance rejecting `status`. If that rejection
    stopped working, all eight would pass against a source that accepts anything — which is the
    shape of a suite that guards nothing.
    """
    instance = _server()
    connector = await connected(instance, cloud_config(base_url=SERVER, page_size=10))
    try:
        with pytest.raises(ConnectorError, match="400"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


# --- Cloud is unchanged ------------------------------------------------------------------------


async def test_cloud_still_writes_the_status_clause_on_every_path() -> None:
    """Not assumed. The clause is what stops trashed content reaching reconciliation there."""
    instance = _cloud(
        attachments=[
            FakeAttachment(
                id="att-1", title="d.pdf", space="ENG", page_id=CHILD, page_title="Data Model"
            )
        ]
    )
    connector = await connected(
        instance,
        cloud_config(base_url=CLOUD, spaces=("ENG",), include_attachments=True, page_size=10),
    )
    try:
        await drain(connector.discover(None))
        await drain(connector.discover(connector.watermark))
        await drain(connector.reconcile())
    finally:
        await connector.teardown()

    sent = _queries(instance)
    assert sent
    assert all("status = current" in query for query in sent), sent


async def test_cloud_subtree_and_attachment_paths_keep_the_clause() -> None:
    instance = _cloud(
        attachments=[
            FakeAttachment(
                id="att-1", title="d.pdf", space="ENG", page_id=CHILD, page_title="Data Model"
            )
        ]
    )
    connector = await connected(
        instance,
        cloud_config(
            base_url=CLOUD,
            spaces=("ENG",),
            root_page_ids=(ROOT,),
            include_attachments=True,
            page_size=10,
        ),
    )
    try:
        await drain(connector.discover(None))
        await drain(connector.reconcile())
    finally:
        await connector.teardown()

    sent = _queries(instance)
    assert any("ancestor" in query for query in sent), "the subtree path ran"
    assert all("status = current" in query for query in sent), sent


async def test_a_data_center_page_still_carries_its_full_provenance_record() -> None:
    """The #140 contract, re-proved on the deployment this change exists for.

    A query fix that quietly cost the source record would be a poor trade, and this is the
    deployment whose provenance nothing had exercised end to end against a refusing source.
    """
    instance = _server()
    connector = await connected(instance, server_config(SERVER, spaces=("ENG",), page_size=10))
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == ROOT)
        raw = await connector.fetch(ref)
    finally:
        await connector.teardown()

    record = Provenance.from_metadata(raw.metadata)
    assert record is not None
    source = record.source
    assert source is not None
    assert source.source_id == ROOT
    assert source.version == "1"
    assert source.canonical_uri.startswith(SERVER)
    assert source.content_type == CONFLUENCE_MEDIA_TYPE
    assert source.modified_at is not None, "version.when, the Server spelling"
    assert source.section_path == ("ENG",)
    assert raw.metadata[ANCESTORS] == ["ENG"]


# --- explicit space scoping --------------------------------------------------------------------


def _space_requests(instance: FakeConfluence) -> tuple[list[httpx.Request], list[httpx.Request]]:
    """Catalog listings and direct single-space lookups, kept apart."""
    space = [request for request in instance.requests if "/rest/api/space" in request.url.path]
    listings = [request for request in space if request.url.path.endswith("/rest/api/space")]
    direct = [request for request in space if request not in listings]
    return listings, direct


async def test_an_allowlist_looks_each_key_up_directly_and_never_lists_the_catalog() -> None:
    """One request per configured key, and the catalog endpoint untouched."""
    instance = _server(spaces={"ENG": "Engineering", "OPS": "Operations", "HR": "People"})
    connector = await connected(
        instance, server_config(SERVER, spaces=("ENG", "OPS"), page_size=10)
    )
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    listings, direct = _space_requests(instance)
    assert listings == [], "an allowlist is a scope boundary, not a filter over the catalog"
    assert [request.url.path.rsplit("/", 1)[-1] for request in direct] == ["ENG", "OPS"]


async def test_no_allowlist_still_enumerates_every_visible_space() -> None:
    """Unchanged, and it has to be: only an unscoped source has a use for the catalog."""
    instance = _server(spaces={"ENG": "Engineering", "OPS": "Operations"})
    connector = await connected(instance, server_config(SERVER, page_size=10))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    listings, direct = _space_requests(instance)
    assert listings, "an unscoped source enumerates"
    assert direct == []
    assert ids(found)


async def test_a_scoped_sync_costs_what_it_is_configured_for_not_what_the_account_can_see() -> None:
    """The quiet half of this change, measured rather than argued.

    A service account entitled to hundreds of spaces and pointed at two used to pay for the
    catalog on every run — six requests carrying five hundred space records, to answer a
    question about two keys. Nothing failed; it was simply the wrong amount of work, and the
    scope boundary was being read past for convenience.
    """
    catalog = {f"SP{index:03d}": f"Space {index}" for index in range(500)}
    catalog.update({"ENG": "Engineering", "OPS": "Operations"})
    instance = _server(spaces=catalog, page_size=100)
    connector = await connected(
        instance, server_config(SERVER, spaces=("ENG", "OPS"), page_size=100)
    )
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    listings, direct = _space_requests(instance)
    assert listings == []
    assert len(direct) == 2, "proportional to the configuration, not to the entitlements"


async def test_the_source_own_spelling_of_a_key_is_what_gets_used() -> None:
    """Space keys are case-insensitive to look up and have one canonical casing.

    Echoing the configured string back would build every subsequent CQL literal against a
    spelling the instance does not use — which a case-sensitive comparison somewhere downstream
    would then get wrong, quietly.
    """
    instance = _server(spaces={"ENG": "Engineering"})
    connector = await connected(instance, server_config(SERVER, spaces=("eng",), page_size=10))
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert any('space = "ENG"' in query for query in _queries(instance)), _queries(instance)


async def test_a_key_needing_encoding_addresses_the_space_rather_than_something_else() -> None:
    """A key is one path segment, and interpolating it raw would address another resource."""
    instance = _server(spaces={"A/B?C": "Odd"}, pages=[FakePage(id="1", title="P", space="A/B?C")])
    connector = await connected(instance, server_config(SERVER, spaces=("A/B?C",), page_size=10))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert ids(found) == ["1"]
    _, direct = _space_requests(instance)
    assert len(direct) == 1
    # `str(url)` rather than `.path`, which httpx hands back decoded — so reading the latter
    # would report success for a key that had been interpolated raw.
    assert str(direct[0].url).endswith("/rest/api/space/A%2FB%3FC")


@pytest.mark.parametrize("operation", ["discover", "reconcile"])
async def test_a_missing_space_is_refused_before_any_content_query(operation: str) -> None:
    """Both entry points, because a refusal on one of them is a corpus deleted by the other."""
    instance = _server(spaces={"ENG": "Engineering"})
    connector = await connected(
        instance, server_config(SERVER, spaces=("ENG", "ENGG"), page_size=10)
    )

    async def run() -> None:
        if operation == "discover":
            await drain(connector.discover(None))
        else:
            await drain(connector.reconcile())

    try:
        with pytest.raises(ConnectorError, match="ENGG"):
            await run()
    finally:
        await connector.teardown()

    assert _queries(instance) == [], "nothing was enumerated before the refusal"


async def test_a_refusal_does_not_recite_the_spaces_the_account_can_see() -> None:
    """The allowlist is a scope boundary. Listing what is outside it to improve a message is
    the request this whole path exists to stop making."""
    instance = _server(spaces={"ENG": "Engineering", "SECRET": "Elsewhere"})
    connector = await connected(instance, server_config(SERVER, spaces=("ENGG",), page_size=10))
    try:
        with pytest.raises(ConnectorError) as raised:
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert "SECRET" not in str(raised.value)
    listings, _ = _space_requests(instance)
    assert listings == [], "and it did not go looking, either"


async def test_a_credential_failure_is_not_reported_as_an_unknown_space() -> None:
    """ "Your token expired" and "that key is wrong" need different repairs.

    Folding the first into the second sends somebody hunting for a typo in a key that was
    always right, which is a long afternoon.
    """
    instance = _server(spaces={"ENG": "Engineering"})
    instance.sign_out("/rest/api/space/")
    connector = await connected(instance, server_config(SERVER, spaces=("ENG",), page_size=10))
    try:
        # The type, not merely the absence of the wrong words: a negative assertion passes for
        # any message at all, including one that stopped naming the space by accident.
        with pytest.raises(SessionExpiredError) as raised:
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert "is not there" not in str(raised.value)


async def test_discovery_and_reconciliation_validate_the_same_spaces() -> None:
    """One method answers both, so they cannot come to disagree about what is in scope."""
    instance = _server(spaces={"ENG": "Engineering", "OPS": "Operations"})
    connector = await connected(instance, server_config(SERVER, spaces=("ENG",), page_size=10))
    try:
        discovered = set(ids(await drain(connector.discover(None))))
        reconciled = set(await drain(connector.reconcile()))
    finally:
        await connector.teardown()

    assert discovered == reconciled
    listings, direct = _space_requests(instance)
    assert listings == []
    assert [request.url.path.rsplit("/", 1)[-1] for request in direct] == ["ENG", "ENG"]


async def test_a_key_configured_twice_is_one_space_and_one_lookup() -> None:
    """``ENG`` and ``eng`` are one space, and enumerating it twice yields every page twice.

    Deduplicated on what the source calls it rather than on what configuration spelled, because
    the two spellings only become one after the lookup has answered.
    """
    instance = _server(spaces={"ENG": "Engineering"})
    connector = await connected(
        instance, server_config(SERVER, spaces=("ENG", "eng"), page_size=10)
    )
    try:
        found = ids(await drain(connector.discover(None)))
    finally:
        await connector.teardown()

    assert sorted(found) == [ROOT, CHILD, "100300"], "each page once"
    assert len(found) == len(set(found))
    searches = [query for query in _queries(instance) if "space =" in query]
    assert len(searches) == 1, searches
