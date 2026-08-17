"""Root-page scoping: what a configured page tree contains, and what it must never contain.

A subtree is defined by a traversal, and a traversal over a paginated API with moved pages,
permission-trimmed results and deleted parents is where a scoped sync goes wrong. So these
tests are mostly about the edges rather than the happy path: a root that is not there, a root
this account cannot read, a page that moves out from under one mid-sync, a tree deeper than any
bound, and a deployment that accepts the descendant predicate without applying it.

**Two failure directions, and only one of them is loud.** Indexing too much is visible — the
corpus has pages nobody asked for. Indexing too little is not: the run reports success, the
counts look plausible, and reconciliation then proposes deleting the difference. Every
assertion below that looks over-careful is guarding the quiet direction.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from manicule.connectors import ConnectorError
from manicule.connectors.config import ConfluenceConfig
from manicule.connectors.confluence import ANCESTOR_IDS, ROOT_PAGE_IDS, SCOPE
from manicule.core.sources import Watermark
from manicule.testing import closing
from tests.connectors.fake_confluence import (
    SERVER_BASE,
    FakeAttachment,
    FakeConfluence,
    FakePage,
)
from tests.connectors.support import cloud_config, connected, drain, ids, server_config

ARCHITECTURE = "100100"
"""The root every test that needs one uses. Synthetic, like every id in this repository."""

OPERATIONS = "100200"
RUNBOOKS = "200100"


def _tree() -> list[FakePage]:
    """One space with two independent trees in it, and one page in neither.

    ``ENG`` holds ``Architecture`` (two children, one grandchild), ``Operations`` (one child)
    and ``Marketing Plan``, which hangs off nothing. The last one is the whole point: every
    test that scopes to ``Architecture`` is also a test that ``Marketing Plan`` stayed out.
    """
    return [
        FakePage(id=ARCHITECTURE, title="Architecture", space="ENG"),
        FakePage(id="100110", title="Data Model", space="ENG", parent=ARCHITECTURE),
        FakePage(id="100111", title="Schemas", space="ENG", parent="100110"),
        FakePage(id="100120", title="Interfaces", space="ENG", parent=ARCHITECTURE),
        FakePage(id=OPERATIONS, title="Operations", space="ENG"),
        FakePage(id="100210", title="On Call", space="ENG", parent=OPERATIONS),
        FakePage(id="100300", title="Marketing Plan", space="ENG"),
    ]


def _instance(**overrides: object) -> FakeConfluence:
    settings: dict[str, object] = {
        "pages": _tree(),
        "spaces": {"ENG": "Engineering"},
        "page_size": 10,
    }
    settings.update(overrides)
    return FakeConfluence(**settings)  # type: ignore[arg-type]


def _two_spaces() -> FakeConfluence:
    """``ENG`` as above, plus an ``OPS`` space with a tree of its own."""
    return FakeConfluence(
        pages=[
            *_tree(),
            FakePage(id=RUNBOOKS, title="Runbooks", space="OPS"),
            FakePage(id="200110", title="Restart", space="OPS", parent=RUNBOOKS),
            FakePage(id="200300", title="Rota", space="OPS"),
        ],
        spaces={"ENG": "Engineering", "OPS": "Operations"},
        page_size=10,
    )


def _scoped(instance: FakeConfluence, **overrides: object) -> ConfluenceConfig:
    settings: dict[str, object] = {
        "base_url": instance.base_url,
        "spaces": ("ENG",),
        "root_page_ids": (ARCHITECTURE,),
        "include_attachments": False,
        "page_size": 10,
    }
    settings.update(overrides)
    return cloud_config(**settings)


async def _discovered(instance: FakeConfluence, config: ConfluenceConfig) -> list[str]:
    connector = await connected(instance, config)
    try:
        return sorted(ids(await drain(connector.discover(None))))
    finally:
        await connector.teardown()


async def _reconciled(instance: FakeConfluence, config: ConfluenceConfig) -> list[str]:
    connector = await connected(instance, config)
    try:
        return sorted(await drain(connector.reconcile()))
    finally:
        await connector.teardown()


# --- what a tree contains -----------------------------------------------------------------


async def test_a_root_brings_every_descendant_at_any_depth() -> None:
    """Children and grandchildren, and nothing that merely shares the space."""
    instance = _instance()
    found = await _discovered(instance, _scoped(instance))

    assert found == [ARCHITECTURE, "100110", "100111", "100120"]


async def test_the_narrowing_is_in_the_query_rather_than_in_a_filter_afterwards() -> None:
    """A subtree sync that enumerated the space and discarded most of it is not a subtree sync.

    It returns the same documents, which is why nothing downstream would notice, and it pays
    for every page in the space on every run — the exact cost the setting exists to avoid.
    """
    instance = _instance()
    await _discovered(instance, _scoped(instance))

    pages = [query for query in instance.queries() if "type = page" in query]
    assert pages, instance.queries()
    assert all(f"ancestor = {ARCHITECTURE}" in query for query in pages), pages


async def test_a_root_page_is_indexed_alongside_its_descendants_by_default() -> None:
    """``root_page_ids = ["100100"]`` names page 100100, so 100100 is in the corpus.

    The default is chosen for the direction the mistake fails in: a corpus missing exactly the
    page that was named looks complete from every angle except the one nobody checks.
    """
    instance = _instance()
    found = await _discovered(instance, _scoped(instance))

    assert ARCHITECTURE in found


async def test_a_root_page_can_be_left_out_and_then_the_query_never_asks_for_it() -> None:
    """Turning it off narrows the query rather than dropping a result on arrival.

    Which makes the setting checkable against the wire: what the run is scoped to is what it
    asked for.
    """
    instance = _instance()
    found = await _discovered(instance, _scoped(instance, include_root_pages=False))

    assert found == ["100110", "100111", "100120"]
    pages = [query for query in instance.queries() if "type = page" in query]
    assert all(f"id = {ARCHITECTURE}" not in query for query in pages), pages


async def test_two_independent_roots_in_one_space_bring_both_trees() -> None:
    instance = _instance()
    found = await _discovered(instance, _scoped(instance, root_page_ids=(ARCHITECTURE, OPERATIONS)))

    assert found == [ARCHITECTURE, "100110", "100111", "100120", OPERATIONS, "100210"]
    assert "100300" not in found


async def test_roots_in_two_allowed_spaces_are_both_synced() -> None:
    instance = _two_spaces()
    found = await _discovered(
        instance,
        _scoped(instance, spaces=("ENG", "OPS"), root_page_ids=(ARCHITECTURE, RUNBOOKS)),
    )

    assert found == [ARCHITECTURE, "100110", "100111", "100120", RUNBOOKS, "200110"]
    assert "200300" not in found


async def test_overlapping_roots_produce_one_document_per_page() -> None:
    """A root and one of its own descendants configured together is one scope, not two.

    Two enumerations of the same page would be two documents with one page id, and the second
    would overwrite the first — after being fetched, parsed and embedded a second time.
    """
    instance = _instance()
    connector = await connected(instance, _scoped(instance, root_page_ids=(ARCHITECTURE, "100110")))
    try:
        found = ids(await drain(connector.discover(None)))
        reconciled = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    assert sorted(found) == [ARCHITECTURE, "100110", "100111", "100120"]
    assert len(found) == len(set(found))
    assert len(reconciled) == len(set(reconciled))


async def test_a_page_records_which_configured_roots_put_it_in_scope() -> None:
    """Derived synchronization metadata, kept beside the page's own identity rather than in it.

    A page under two configured roots names both: the question this answers is "why is this
    here", and one answer would be a guess about which root the reader meant.
    """
    instance = _instance()
    connector = await connected(instance, _scoped(instance, root_page_ids=(ARCHITECTURE, "100110")))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    by_id = {document.source_id: document.ref.metadata for document in found}
    assert by_id["100111"][ROOT_PAGE_IDS] == [ARCHITECTURE, "100110"]
    assert by_id["100120"][ROOT_PAGE_IDS] == [ARCHITECTURE]
    assert by_id["100111"][ANCESTOR_IDS] == [ARCHITECTURE, "100110"]


async def test_a_tree_deeper_than_any_bound_this_connector_could_have_chosen_is_one_query() -> None:
    """There is no depth ceiling, because there is no client-side walk to bound.

    ``ancestor`` matches at any depth, so a fortieth-level page costs the same as a first-level
    one. A connector that walked ``child/page`` would need a limit here, and every value for it
    would be wrong for somebody — silently, by leaving the bottom of their tree unindexed.
    """
    deep = [FakePage(id=ARCHITECTURE, title="Architecture", space="ENG")]
    deep.extend(
        FakePage(
            id=f"1002{level:02d}",
            title=f"Level {level}",
            space="ENG",
            parent=ARCHITECTURE if level == 1 else f"1002{level - 1:02d}",
        )
        for level in range(1, 41)
    )
    instance = FakeConfluence(pages=deep, spaces={"ENG": "Engineering"}, page_size=10)
    found = await _discovered(instance, _scoped(instance))

    assert len(found) == 41
    assert "100240" in found


async def test_a_cycle_in_the_page_tree_is_not_a_walk_this_connector_can_hang_on() -> None:
    """Confluence does not permit one, and nothing here would survive it by luck.

    The property is structural rather than defensive: scope is decided from the ancestor list a
    single response carries, and nothing follows a parent link, so there is no loop to enter.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(id=ARCHITECTURE, title="Architecture", space="ENG", parent="100110"),
            FakePage(id="100110", title="Data Model", space="ENG", parent=ARCHITECTURE),
        ],
        spaces={"ENG": "Engineering"},
        page_size=10,
    )
    found = await _discovered(instance, _scoped(instance))

    assert found == [ARCHITECTURE, "100110"]


# --- validating the roots before anything is enumerated ------------------------------------


async def test_a_root_that_does_not_exist_stops_the_run_before_anything_is_enumerated() -> None:
    """And *before* is the word that matters: an empty subtree is a proposal to delete one.

    A run that enumerated first and failed afterwards would be harmless; a run that enumerated
    an empty subtree and completed cleanly would hand reconciliation the whole index.
    """
    instance = _instance()
    connector = await connected(instance, _scoped(instance, root_page_ids=("999999",)))
    try:
        with pytest.raises(ConnectorError, match="999999"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert instance.queries() == [], "no content was enumerated before the refusal"


async def test_a_root_this_account_cannot_read_is_an_access_failure_not_an_empty_subtree() -> None:
    """Losing access to a root and the root's subtree being deleted look identical downstream.

    They are one refusal apart. Without it, the first is reported as the second, and every
    document under that root is proposed for deletion on the strength of a permission change.
    """
    instance = _instance()
    instance.forbidden_pages.add(ARCHITECTURE)
    connector = await connected(instance, _scoped(instance))
    try:
        with pytest.raises(ConnectorError):
            await drain(connector.reconcile())
    finally:
        await connector.teardown()


async def test_a_trashed_root_is_refused_rather_than_treated_as_an_empty_tree() -> None:
    instance = _instance(
        pages=[
            FakePage(id=ARCHITECTURE, title="Architecture", space="ENG", status="trashed"),
            FakePage(id="100110", title="Data Model", space="ENG", parent=ARCHITECTURE),
        ]
    )
    connector = await connected(instance, _scoped(instance))
    try:
        with pytest.raises(ConnectorError, match="trashed"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_root_that_is_not_a_page_is_refused() -> None:
    """A blog post has an id somebody can paste and no descendants to give.

    Left alone it is a root that enumerates an empty subtree, which is the refusal-shaped
    failure wearing a configuration mistake's clothes.
    """
    instance = _instance(
        pages=[
            FakePage(id=ARCHITECTURE, title="Launch", space="ENG", kind="blogpost"),
            FakePage(id="100110", title="Data Model", space="ENG", parent=ARCHITECTURE),
        ]
    )
    connector = await connected(instance, _scoped(instance))
    try:
        with pytest.raises(ConnectorError, match="blogpost"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_root_whose_space_the_source_did_not_name_is_refused() -> None:
    """Without a space there is no query to scope and no allowlist check to make.

    Guessing one would mean querying a space nobody named, and a query against a space that is
    not there returns nothing rather than an error.
    """
    instance = _instance(
        pages=[FakePage(id=ARCHITECTURE, title="Architecture", space="")],
        spaces={"ENG": "Engineering"},
    )
    connector = await connected(instance, _scoped(instance, spaces=()))
    try:
        with pytest.raises(ConnectorError, match="did not say which space"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_root_outside_the_space_allowlist_is_refused() -> None:
    """``spaces`` and ``root_page_ids`` narrow one scope between them.

    Honoring a root outside the allowlist would mean the allowlist had quietly stopped being
    one, which is the single thing a setting by that name may not do.
    """
    instance = _two_spaces()
    connector = await connected(
        instance, _scoped(instance, spaces=("ENG",), root_page_ids=(ARCHITECTURE, RUNBOOKS))
    )
    try:
        with pytest.raises(ConnectorError, match=RUNBOOKS):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_listed_space_with_no_configured_root_is_refused() -> None:
    """The other direction, and it fails just as quietly if it is allowed through.

    A space that is configured and enumerates nothing is indistinguishable from a space that is
    empty, and reconciliation then proposes deleting everything it ever contributed.
    """
    instance = _two_spaces()
    connector = await connected(
        instance, _scoped(instance, spaces=("ENG", "OPS"), root_page_ids=(ARCHITECTURE,))
    )
    try:
        with pytest.raises(ConnectorError, match="OPS"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_root_in_a_space_the_account_cannot_see_is_refused_with_no_allowlist() -> None:
    """With no ``spaces`` there is no allowlist, and the check becomes visibility itself."""
    instance = _two_spaces()
    instance.spaces.pop("OPS")
    connector = await connected(instance, _scoped(instance, spaces=(), root_page_ids=(RUNBOOKS,)))
    try:
        with pytest.raises(ConnectorError, match="OPS"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


# --- a source that does not narrow ----------------------------------------------------------


async def test_a_deployment_that_ignores_the_descendant_predicate_stops_the_run() -> None:
    """The one failure with no symptom, made into a refusal.

    An instance that accepts ``ancestor`` and does not apply it answers with the whole space.
    Every request succeeded, every page is real, and the sync indexes a space while its
    configuration says one page tree — so the check is on the pages that arrive, not on the
    query that went out.
    """
    instance = _instance()
    instance.ancestor_predicate = "ignored"
    connector = await connected(instance, _scoped(instance))
    try:
        with pytest.raises(ConnectorError, match="not apply the narrowing"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_an_empty_answer_for_a_root_with_children_is_refused() -> None:
    """ "This tree has no descendants" is the most dangerous sentence reconciliation can hear.

    A deployment that accepts ``ancestor`` and matches nothing with it answers exactly as one
    whose subtree has been emptied: a successful query, no rows, no error anywhere. Taken at
    face value that is every descendant proposed for deletion. So it is never taken at face
    value — each root is asked, through a different endpoint, whether it has a child.
    """
    instance = _instance()
    instance.ancestor_predicate = "empty"
    connector = await connected(instance, _scoped(instance))
    try:
        with pytest.raises(ConnectorError, match="child pages"):
            await drain(connector.reconcile())
    finally:
        await connector.teardown()


async def test_reconciliation_refuses_a_source_that_answers_with_more_than_the_subtree() -> None:
    """The same check as discovery's, on the pass where getting it wrong is destructive.

    Reconciliation reports what still exists, and the pipeline deletes what it omits — so a
    pass that accepted a space-wide answer would report far too much, agree with nothing
    discovery said, and hide the deployment problem behind a diff that deletes nothing.
    """
    instance = _instance()
    instance.ancestor_predicate = "ignored"
    connector = await connected(instance, _scoped(instance))
    try:
        with pytest.raises(ConnectorError, match="not apply the narrowing"):
            await drain(connector.reconcile())
    finally:
        await connector.teardown()


async def test_a_root_that_really_has_no_children_reconciles_as_itself() -> None:
    """The guard must not make an ordinary one-page tree unsyncable."""
    instance = _instance(pages=[FakePage(id=ARCHITECTURE, title="Architecture", space="ENG")])
    reconciled = await _reconciled(instance, _scoped(instance))

    assert reconciled == [ARCHITECTURE]


# --- moves, renames and identity -------------------------------------------------------------


async def test_a_page_renamed_inside_the_subtree_keeps_its_page_id() -> None:
    """Identity is the Confluence page id and nothing derived from where the page sits."""
    instance = _instance()
    config = _scoped(instance)
    before = await _discovered(instance, config)

    instance.pages["100110"] = FakePage(
        id="100110",
        title="Storage Model",
        space="ENG",
        parent=ARCHITECTURE,
        version=2,
        when="2026-08-10T09:00:00.000+01:00",
    )
    after = await _discovered(instance, config)

    assert before == after
    assert "100110" in after


async def test_a_page_moved_between_two_configured_roots_keeps_its_identity() -> None:
    """One document, re-scoped. Two would be the same page indexed twice under one id."""
    instance = _instance()
    config = _scoped(instance, root_page_ids=(ARCHITECTURE, OPERATIONS))
    connector = await connected(instance, config)
    try:
        before = await drain(connector.discover(None))
        instance.move("100110", OPERATIONS)
        after = await drain(connector.discover(None))
        reconciled = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    moved = next(document for document in after if document.source_id == "100110")
    assert moved.ref.metadata[ROOT_PAGE_IDS] == [OPERATIONS]
    assert next(d for d in before if d.source_id == "100110").ref.metadata[ROOT_PAGE_IDS] == [
        ARCHITECTURE
    ]
    assert reconciled.count("100110") == 1


async def test_a_page_moved_out_of_every_configured_root_stops_being_reconciled() -> None:
    """Which is what removes it: the pipeline soft-deletes what reconciliation does not report.

    Its subtree goes with it, because scope is inherited — the grandchild did not move, and it
    is out of scope all the same.
    """
    instance = _instance()
    config = _scoped(instance)
    before = await _reconciled(instance, config)

    instance.move("100110", "100300")
    after = await _reconciled(instance, config)

    assert set(before) - set(after) == {"100110", "100111"}


async def test_a_page_moved_into_the_subtree_is_discovered() -> None:
    """A move is not an edit, so this is the case a version-token comparison cannot catch."""
    instance = _instance()
    config = _scoped(instance)
    before = await _discovered(instance, config)
    assert "100300" not in before

    instance.move("100300", ARCHITECTURE)
    after = await _discovered(instance, config)

    assert "100300" in after


async def test_a_page_moved_out_mid_enumeration_is_removed_by_the_next_reconciliation() -> None:
    """Discovery may still yield it — it matched when the query ran — and that is not a defect.

    The guarded reconciliation path is what removes it, which is the same path that removes a
    deleted page. Nothing here tries to make a single enumeration atomic against a wiki people
    are editing.
    """
    instance = _instance()
    config = _scoped(instance)
    connector = await connected(instance, config)
    try:
        seen: list[str] = []
        async with closing(connector.discover(None)) as stream:
            async for document in stream:
                seen.append(document.source_id)
                if document.source_id == ARCHITECTURE:
                    instance.move("100120", "100300")
        reconciled = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    assert "100120" not in reconciled
    assert set(reconciled) == {ARCHITECTURE, "100110", "100111"}


# --- attachments -----------------------------------------------------------------------------


def _with_attachments() -> FakeConfluence:
    return FakeConfluence(
        pages=_tree(),
        attachments=[
            FakeAttachment(
                id="att-in",
                title="schema.pdf",
                space="ENG",
                page_id="100110",
                page_title="Data Model",
            ),
            FakeAttachment(
                id="att-out",
                title="campaign.pdf",
                space="ENG",
                page_id="100300",
                page_title="Marketing Plan",
            ),
        ],
        spaces={"ENG": "Engineering"},
        page_size=10,
    )


async def test_an_attachment_is_in_scope_exactly_when_the_page_holding_it_is() -> None:
    """Resolved through the container page rather than through the attachment's own ancestry.

    Confluence exposes an attachment's container reliably and its position in a page tree less
    so, and the container is the authoritative answer in any case.
    """
    instance = _with_attachments()
    found = await _discovered(instance, _scoped(instance, include_attachments=True))

    assert "att-in" in found
    assert "att-out" not in found


async def test_an_attachment_of_a_page_that_has_not_changed_is_still_discovered() -> None:
    """The case a naive implementation loses: the container is not in this run's page results.

    An incremental page query returns only pages that changed, so an attachment added to a
    quiet page has a container the run has never seen. Scope is therefore decided against the
    whole subtree rather than against what this enumeration happened to return.
    """
    instance = _with_attachments()
    # No overlap, so that "did this page change" is the only reason a page could come back.
    # The overlap exists for CQL's minute granularity and is tested where it belongs.
    config = _scoped(instance, include_attachments=True, watermark_overlap_minutes=0)
    connector = await connected(instance, config)
    try:
        await drain(connector.discover(None))
        watermark = connector.watermark
        instance.attachments["att-new"] = FakeAttachment(
            id="att-new",
            title="added.pdf",
            space="ENG",
            page_id="100111",
            page_title="Schemas",
            when="2026-08-20T10:00:00.000+01:00",
        )
        found = ids(await drain(connector.discover(watermark)))
    finally:
        await connector.teardown()

    assert "att-new" in found
    assert "100111" not in found, "the page itself did not change"


async def test_moving_a_page_out_of_scope_takes_its_attachments_with_it() -> None:
    instance = _with_attachments()
    config = _scoped(instance, include_attachments=True)
    before = await _reconciled(instance, config)
    assert "att-in" in before

    instance.move("100110", "100300")
    after = await _reconciled(instance, config)

    assert "att-in" not in after
    assert set(before) - set(after) == {"100110", "100111", "att-in"}


async def test_attachments_stay_out_entirely_when_they_are_turned_off() -> None:
    instance = _with_attachments()
    found = await _discovered(instance, _scoped(instance, include_attachments=False))

    assert found == [ARCHITECTURE, "100110", "100111", "100120"]
    assert all("attachment" not in query for query in instance.queries())


async def test_the_attachment_query_is_space_wide_and_that_is_the_stated_cost() -> None:
    """Pages are narrowed at the source; attachments are not, and the difference is visible.

    Recorded as a test rather than only as a sentence in the documentation, because it is the
    one place a subtree sync still pays per space — and a later change that narrowed it would
    have to be a deliberate one.
    """
    instance = _with_attachments()
    await _discovered(instance, _scoped(instance, include_attachments=True))

    attachments = [query for query in instance.queries() if "type = attachment" in query]
    assert attachments
    assert all("ancestor" not in query for query in attachments), attachments


# --- watermarks ------------------------------------------------------------------------------


async def test_an_abandoned_subtree_enumeration_offers_no_watermark() -> None:
    """A prefix of a subtree is still a prefix, and a watermark from one skips the rest."""
    instance = _instance()
    connector = await connected(instance, _scoped(instance))
    try:
        async with closing(connector.discover(None)) as stream:
            await anext(stream)
        assert connector.watermark is None
    finally:
        await connector.teardown()


async def test_a_run_bounded_by_a_limit_does_not_advance_a_subtree_watermark() -> None:
    """``--limit`` bounds work; it does not define a subtree.

    The pipeline stops consuming discovery, which is an abandoned enumeration by another name.
    Reproduced here at the connector, because the guarantee has to hold wherever the consumer
    stops rather than only where the pipeline does.

    **Two spaces, and the limit falls in the second.** With one space the assertion passes for
    a reason that has nothing to do with the guard: no space finished, so there is no position
    to offer and ``None`` comes back either way. Here the first space *did* finish, so a
    watermark is available to be offered wrongly — and refusing to offer it is the whole
    behavior. A limit is not a scope, and half a corpus recorded as a complete position skips
    the other half forever.
    """
    instance = _two_spaces()
    config = _scoped(instance, spaces=("ENG", "OPS"), root_page_ids=(ARCHITECTURE, RUNBOOKS))
    connector = await connected(instance, config)
    try:
        taken: list[str] = []
        async with closing(connector.discover(None)) as stream:
            async for document in stream:
                taken.append(document.source_id)
                if len(taken) == 5:
                    break
        assert taken[:4] == [ARCHITECTURE, "100110", "100111", "100120"]
        assert taken[4] == RUNBOOKS, "the first space finished before the limit was reached"
        assert connector.watermark is None
    finally:
        await connector.teardown()


async def test_a_completed_subtree_run_records_the_scope_its_position_belongs_to() -> None:
    instance = _instance()
    connector = await connected(instance, _scoped(instance))
    try:
        await drain(connector.discover(None))
        watermark = connector.watermark
    finally:
        await connector.teardown()

    assert watermark is not None
    assert watermark.metadata[SCOPE] == f"roots={ARCHITECTURE} include_roots=true"


async def test_changing_the_configured_roots_discards_the_stored_position() -> None:
    """A watermark is a position **within a scope**, and reusing one across scopes loses pages.

    Every page in the newly configured tree that has not changed since the stored instant is
    already behind it, so an incremental query would never return it — and nothing would ever
    return it again.
    """
    instance = _instance()
    connector = await connected(instance, _scoped(instance, root_page_ids=(OPERATIONS,)))
    stored = Watermark(
        value="2026-09-01T00:00:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={
            "spaces": {"ENG": "2026-09-01T00:00:00+01:00"},
            SCOPE: f"roots={ARCHITECTURE} include_roots=true",
        },
    )
    try:
        found = ids(await drain(connector.discover(stored)))
    finally:
        await connector.teardown()

    assert sorted(found) == [OPERATIONS, "100210"]
    assert all("lastmodified >=" not in query for query in instance.queries())


async def test_changing_the_roots_says_so_where_somebody_will_read_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The scope change is a fact about the index, not only about this run.

    Documents indexed under a root that is no longer configured are still there and are now out
    of scope, so the diagnostic names both scopes and says what removes them.
    """
    instance = _instance()
    connector = await connected(instance, _scoped(instance, root_page_ids=(OPERATIONS,)))
    stored = Watermark(
        value="2026-09-01T00:00:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={
            "spaces": {"ENG": "2026-09-01T00:00:00+01:00"},
            SCOPE: f"roots={ARCHITECTURE} include_roots=true",
        },
    )
    with caplog.at_level(logging.WARNING, logger="manicule.connectors.confluence"):
        try:
            await drain(connector.discover(stored))
        finally:
            await connector.teardown()

    assert caplog.records, "a scope change with a stored position must be reported"
    message = caplog.records[0].getMessage()
    assert ARCHITECTURE in message
    assert OPERATIONS in message
    assert "reconciliation" in message


async def test_an_unchanged_scope_reports_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """A warning on every ordinary run is a warning nobody reads on the run that matters."""
    instance = _instance()
    connector = await connected(instance, _scoped(instance))
    with caplog.at_level(logging.WARNING, logger="manicule.connectors.confluence"):
        try:
            await drain(connector.discover(None))
        finally:
            await connector.teardown()

    assert caplog.records == []


async def test_an_unchanged_scope_resumes_from_the_stored_position() -> None:
    """The other half: a scope that did not change must not force a full re-enumeration."""
    instance = _instance()
    connector = await connected(instance, _scoped(instance))
    stored = Watermark(
        value="2026-08-09T14:30:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={
            "spaces": {"ENG": "2026-08-09T14:30:00+01:00"},
            SCOPE: f"roots={ARCHITECTURE} include_roots=true",
        },
    )
    try:
        await drain(connector.discover(stored))
    finally:
        await connector.teardown()

    pages = [query for query in instance.queries() if "type = page" in query]
    assert any('lastmodified >= "2026/08/09 14:25"' in query for query in pages), pages


async def test_a_watermark_stored_before_scopes_existed_still_resumes_a_whole_space_sync() -> None:
    """Whole-space is the only scope such a watermark can have been recorded in.

    Reading its absence as a mismatch would make every existing installation re-enumerate its
    entire corpus once, for nothing.
    """
    instance = _instance()
    connector = await connected(instance, cloud_config(base_url=instance.base_url, page_size=10))
    stored = Watermark(
        value="2026-08-09T14:30:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={"spaces": {"ENG": "2026-08-09T14:30:00+01:00"}},
    )
    try:
        await drain(connector.discover(stored))
    finally:
        await connector.teardown()

    assert any('lastmodified >= "2026/08/09 14:25"' in q for q in instance.queries())


# --- reconciliation ---------------------------------------------------------------------------


async def test_discovery_and_reconciliation_report_the_same_scope() -> None:
    """Scoped on one side only is a corpus that deletes itself, or one that never shrinks.

    They agree because they ask the same object the same question, rather than because two
    implementations of one rule happen to line up today.
    """
    instance = _with_attachments()
    connector = await connected(instance, _scoped(instance, include_attachments=True))
    try:
        discovered = set(ids(await drain(connector.discover(None))))
        reconciled = set(await drain(connector.reconcile()))
    finally:
        await connector.teardown()

    assert discovered == reconciled


async def test_scoped_reconciliation_preserves_native_search_page_boundaries() -> None:
    """Page two is not requested until page one has been handed to durable reconciliation."""
    instance = _instance(page_size=2)
    connector = await connected(instance, _scoped(instance, page_size=2))
    try:
        async with closing(connector.reconcile_batches()) as batches:
            first = await anext(batches)
            assert len(first) == 2
            assert len(instance.queries()) == 1

            second = await anext(batches)
            assert len(second) == 2
            assert len(instance.queries()) == 2
    finally:
        await connector.teardown()


async def test_a_failure_part_way_through_a_scoped_reconciliation_raises() -> None:
    """The ids seen so far are not the truth, and diffing them soft-deletes the difference.

    The failure is injected after the pages have been reported, so it lands in the attachment
    enumeration — which is the part of a scoped reconciliation that is still streaming when a
    cursor expires or a source starts throttling.
    """
    instance = _with_attachments()
    connector = await connected(instance, _scoped(instance, include_attachments=True))
    seen: list[str] = []

    async def walk() -> None:
        async for source_id in connector.reconcile():
            seen.append(source_id)
            if len(seen) == 1:
                instance.throttles.append({"Retry-After": "999"})

    try:
        with pytest.raises(ConnectorError):
            await walk()
    finally:
        await connector.teardown()

    assert "att-in" not in seen, "the enumeration must not have completed"


async def test_a_subtree_enumeration_that_fails_reports_no_ids_at_all() -> None:
    """A scoped reconciliation resolves the whole page tree before it yields its first id.

    So a source that fails during that resolution produces nothing rather than a prefix, and
    the pipeline's first guard has nothing to discard. Asserted rather than assumed, because it
    is a property of how the enumeration is ordered and a later change could lose it silently.
    """
    instance = _instance()
    connector = await connected(instance, _scoped(instance))
    instance.throttle(times=1, retry_after="999")
    seen: list[str] = []

    async def walk() -> None:
        async for source_id in connector.reconcile():
            seen.append(source_id)

    try:
        with pytest.raises(ConnectorError):
            await walk()
    finally:
        await connector.teardown()

    assert seen == []


async def test_a_scoped_reconciliation_never_reports_a_page_outside_the_tree() -> None:
    """Because everything it does not report is a page the pipeline will delete."""
    instance = _instance()
    reconciled = await _reconciled(instance, _scoped(instance))

    assert "100300" not in reconciled
    assert OPERATIONS not in reconciled


# --- everything that must not have changed ------------------------------------------------------


async def test_whole_space_behavior_is_untouched_when_no_roots_are_configured() -> None:
    """The setting is a narrowing of the existing scope model, not a second one beside it."""
    instance = _instance()
    config = cloud_config(base_url=instance.base_url, spaces=("ENG",), page_size=10)
    found = await _discovered(instance, config)

    assert found == sorted(page.id for page in _tree())
    assert all("ancestor" not in query for query in instance.queries())


async def test_a_whole_space_watermark_still_carries_only_what_it_used_to_plus_its_scope() -> None:
    instance = _instance()
    connector = await connected(instance, cloud_config(base_url=instance.base_url, page_size=10))
    try:
        await drain(connector.discover(None))
        watermark = connector.watermark
    finally:
        await connector.teardown()

    assert watermark is not None
    assert set(watermark.metadata) == {"spaces", SCOPE}
    assert watermark.metadata[SCOPE] == "whole-space"


async def test_scope_does_not_change_how_a_server_body_is_read() -> None:
    """Deployment decides the body format and the URLs; scope decides which pages.

    Kept apart on purpose: a subtree of a Server instance is still storage format, and a whole
    space of a Cloud one is still Atlassian Document Format.
    """
    instance = FakeConfluence(base_url=SERVER_BASE, pages=_tree(), spaces={"ENG": "Engineering"})
    config = server_config(
        instance.base_url,
        spaces=("ENG",),
        root_page_ids=(ARCHITECTURE,),
        include_attachments=False,
        page_size=10,
    )
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        fetched = await connector.fetch(
            next(document for document in found if document.source_id == "100110").ref
        )
    finally:
        await connector.teardown()

    assert sorted(ids(found)) == [ARCHITECTURE, "100110", "100111", "100120"]
    assert fetched.metadata["body_format"] == "storage"
    assert fetched.metadata[ROOT_PAGE_IDS] == [ARCHITECTURE]
    assert fetched.uri.startswith(SERVER_BASE)


async def test_a_fetched_page_keeps_the_root_that_put_it_in_scope() -> None:
    """Provenance has to survive the fetch, because the fetch is what the pipeline stores."""
    instance = _instance()
    connector = await connected(instance, _scoped(instance))
    try:
        found = await drain(connector.discover(None))
        fetched = await connector.fetch(
            next(document for document in found if document.source_id == "100111").ref
        )
    finally:
        await connector.teardown()

    assert fetched.source_id == "100111"
    assert fetched.metadata[ROOT_PAGE_IDS] == [ARCHITECTURE]
    assert fetched.metadata[ANCESTOR_IDS] == [ARCHITECTURE, "100110"]


async def test_a_whole_space_fetch_asserts_no_root_at_all() -> None:
    """``None`` rather than an omitted key, so a source that stopped being scoped clears it."""
    instance = _instance()
    connector = await connected(instance, cloud_config(base_url=instance.base_url, page_size=10))
    try:
        found = await drain(connector.discover(None))
        fetched = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    assert fetched.metadata[ROOT_PAGE_IDS] is None


# --- configuration refusals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "root",
    ["Architecture", "", "100100 OR 1=1", "abc", "100100; DROP"],
    ids=["title", "empty", "injection", "letters", "statement"],
)
def test_a_root_page_id_that_is_not_a_content_id_is_refused_at_startup(root: str) -> None:
    """Ids reach CQL unquoted, so "is it digits" is the whole of the injection defense.

    Not an escaping rule but a shape rule, which is the stronger of the two: a value that could
    end a literal and continue as query syntax never reaches a query at all, so there is no
    escaping to get subtly wrong. Against a search endpoint the result of getting it wrong is
    not an error — it is results, from a query nobody wrote.
    """
    with pytest.raises(ValueError, match="page id"):
        cloud_config(base_url="https://confluence.example.test/x", root_page_ids=(root,))


def test_a_root_page_id_survives_the_whitespace_a_configuration_file_collects() -> None:
    """Surrounding whitespace is a typing artifact, not a different page."""
    config = cloud_config(
        base_url="https://confluence.example.test/x", root_page_ids=(f" {ARCHITECTURE} ",)
    )

    assert config.root_page_ids == (ARCHITECTURE,)


def test_root_page_ids_are_stored_deduplicated_and_in_a_stable_order() -> None:
    """They are half of the scope identity, and a reordered list is the same scope.

    Without this, editing a configuration file's line order would discard every stored position
    and re-enumerate the corpus for no reason anybody could see.
    """
    config = cloud_config(
        base_url="https://confluence.example.test/x",
        root_page_ids=(OPERATIONS, ARCHITECTURE, OPERATIONS),
    )
    reordered = cloud_config(
        base_url="https://confluence.example.test/x",
        root_page_ids=(ARCHITECTURE, OPERATIONS),
    )

    assert config.root_page_ids == (OPERATIONS, ARCHITECTURE)
    assert config.scope_identity == reordered.scope_identity


def test_include_root_pages_without_any_roots_is_refused_rather_than_ignored() -> None:
    """A setting that silently does nothing reads exactly like one that is in force."""
    with pytest.raises(ValueError, match="root_page_ids is empty"):
        cloud_config(base_url="https://confluence.example.test/x", include_root_pages=False)


def test_the_scope_identity_distinguishes_the_things_that_change_what_is_in_scope() -> None:
    def identity(**overrides: object) -> str:
        settings: dict[str, object] = {"base_url": "https://confluence.example.test/x"}
        settings.update(overrides)
        return cloud_config(**settings).scope_identity

    assert identity() == "whole-space"
    assert identity(root_page_ids=(ARCHITECTURE,)) != identity(root_page_ids=(OPERATIONS,))
    assert identity(root_page_ids=(ARCHITECTURE,)) != identity(
        root_page_ids=(ARCHITECTURE,), include_root_pages=False
    )
    assert identity(root_page_ids=(ARCHITECTURE,), spaces=("ENG",)) == identity(
        root_page_ids=(ARCHITECTURE,), spaces=("OPS",)
    )
