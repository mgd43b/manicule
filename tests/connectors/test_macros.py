"""Macros, which are where content hides.

``include`` and ``excerpt-include`` render another page's content inline. A reader sees it; a
body fetched from the API carries only the macro node. Unresolved, the text is missing from the
chunk while appearing present in the UI — invisible from both ends
(``docs/connectors/confluence.md`` §5).

Expansion is bounded by a depth limit and by cycle detection, and both are exercised here
against fixtures that would not terminate without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from manicule.connectors.macros import excerpt_of_storage, storage_macros
from manicule.ingest.reconcile import reconcile
from tests.connectors.fake_confluence import (
    SERVER_BASE,
    FakeConfluence,
    FakePage,
    paragraph,
    storage_include,
    with_include,
)
from tests.connectors.support import cloud_config, connected, drain, server_config
from tests.ingest.test_pipeline import build


async def _body(
    instance: FakeConfluence, page_id: str = "1", **overrides: object
) -> tuple[str, Sequence[object], Sequence[object]]:
    """Fetch one page and return its text, the pages it included, and what it could not."""
    config = cloud_config(base_url=instance.base_url, **overrides)
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == page_id)
        raw = await connector.fetch(ref)
    finally:
        await connector.teardown()
    included = raw.metadata["included_pages"]
    unresolved = raw.metadata["unresolved_macros"]
    assert isinstance(included, list)
    assert isinstance(unresolved, list)
    return raw.as_text(), included, unresolved


async def test_an_included_page_becomes_part_of_the_body() -> None:
    """Without this the chunk is short by exactly the content the macro was there to add."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1", title="Overview", space="ENG", adf=with_include("see", title="Detail")
            ),
            FakePage(id="2", title="Detail", space="ENG", adf=paragraph("the rotation interval")),
        ]
    )
    text, included, unresolved = await _body(instance)

    assert "the rotation interval" in text
    assert included == ["2"]
    assert unresolved == []


async def test_changed_include_forces_an_unchanged_parent_outside_the_overlap() -> None:
    """A child edit re-renders its parent even when CQL only returns the child.

    The assertion is intentionally through the full connector and pipeline: persisted edges,
    forced token bypass, macro expansion, and replacement publication must agree or the old
    included text remains retrievable despite a clean-looking incremental sync.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Overview",
                space="ENG",
                when="2026-08-01T14:30:00.000+01:00",
                adf=with_include("see", title="Detail"),
            ),
            FakePage(id="2", title="Detail", space="ENG", adf=paragraph("before rotation")),
        ]
    )
    connector = await connected(
        instance,
        cloud_config(base_url=instance.base_url, watermark_overlap_minutes=0),
    )
    pipeline, store, _ = build()
    try:
        await pipeline.run(connector)
        parent = await store.find_document(connector.name, "1")
        assert parent is not None
        assert store.source_dependencies[parent.id] == ("2",)

        instance.pages["2"] = replace(
            instance.pages["2"],
            version=2,
            when="2026-08-20T14:30:00.000+01:00",
            adf=paragraph("after rotation"),
        )
        instance.body_calls.clear()
        report = await pipeline.run(connector)
    finally:
        await connector.teardown()

    refreshed = await store.find_document(connector.name, "1")
    assert refreshed is not None
    assert report.discovered == 2
    assert instance.body_calls["1"] == 1
    assert refreshed.version_token == "1"  # noqa: S105 - a source revision, not a credential
    text = "\n".join(chunk.text for chunk in store.chunks[refreshed.id])
    assert "after rotation" in text
    assert "before rotation" not in text


async def test_reconciled_deleted_include_forces_parent_and_removes_its_edge() -> None:
    """A removed child cannot strand stale expanded text or a permanent retry edge."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Overview",
                space="ENG",
                when="2026-08-01T14:30:00.000+01:00",
                adf=with_include("see", title="Detail"),
            ),
            FakePage(id="2", title="Detail", space="ENG", adf=paragraph("retired procedure")),
        ]
    )
    connector = await connected(
        instance,
        cloud_config(base_url=instance.base_url, watermark_overlap_minutes=0),
    )
    pipeline, store, _ = build()
    try:
        await pipeline.run(connector)
        instance.delete("2")
        result = await reconcile(connector, store, max_delete_fraction=1)
        assert result.deleted == ("2",)

        instance.body_calls.clear()
        report = await pipeline.run(connector)
    finally:
        await connector.teardown()

    parent = await store.find_document(connector.name, "1")
    assert parent is not None
    assert report.discovered == 1
    assert instance.body_calls["1"] == 1
    assert store.source_dependencies[parent.id] == ()
    text = "\n".join(chunk.text for chunk in store.chunks[parent.id])
    assert "retired procedure" not in text


async def test_inaccessible_include_forces_parent_and_removes_stale_text() -> None:
    """A 403 child is an invalidation event even though its old row remains servable."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Overview",
                space="ENG",
                when="2026-08-01T14:30:00.000+01:00",
                adf=with_include("see", title="Restricted detail"),
            ),
            FakePage(
                id="2",
                title="Restricted detail",
                space="ENG",
                adf=paragraph("confidential rotation"),
            ),
        ]
    )
    connector = await connected(
        instance,
        cloud_config(base_url=instance.base_url, watermark_overlap_minutes=0),
    )
    pipeline, store, _ = build()
    try:
        await pipeline.run(connector)
        instance.pages["2"] = replace(
            instance.pages["2"],
            version=2,
            when="2026-08-20T14:30:00.000+01:00",
        )
        instance.forbidden_pages.add("2")
        instance.body_calls.clear()
        await pipeline.run(connector)
    finally:
        await connector.teardown()

    parent = await store.find_document(connector.name, "1")
    assert parent is not None
    assert instance.body_calls["1"] == 1
    assert store.source_dependencies[parent.id] == ()
    text = "\n".join(chunk.text for chunk in store.chunks[parent.id])
    assert "confidential rotation" not in text


async def test_reverse_include_cycle_is_bounded_and_forces_each_parent_once() -> None:
    """Persisted reverse edges stop at a cycle instead of repeatedly re-enqueueing it."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="A",
                space="ENG",
                when="2026-08-01T14:30:00.000+01:00",
                adf=with_include("from A", title="B"),
            ),
            FakePage(id="2", title="B", space="ENG", adf=with_include("from B", title="A")),
        ]
    )
    connector = await connected(
        instance,
        cloud_config(base_url=instance.base_url, watermark_overlap_minutes=0),
    )
    pipeline, _, _ = build()
    try:
        await pipeline.run(connector)
        instance.pages["2"] = replace(
            instance.pages["2"], version=2, when="2026-08-20T14:30:00.000+01:00"
        )
        instance.body_calls.clear()
        report = await pipeline.run(connector)
    finally:
        await connector.teardown()

    assert report.discovered == 2


async def test_two_pages_that_include_each_other_do_not_expand_forever() -> None:
    """An overview and a detail page cross-including is an ordinary authoring mistake.

    Without cycle detection the expansion does not terminate, so this fixture is the guard's
    reason for existing rather than an illustration of it.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(id="1", title="A", space="ENG", adf=with_include("a", title="B")),
            FakePage(id="2", title="B", space="ENG", adf=with_include("b", title="A")),
        ]
    )
    text, included, unresolved = await _body(instance)

    assert "b" in text
    assert included == ["2"]
    assert len(unresolved) == 1
    assert "would not terminate" in str(unresolved[0])


async def test_a_page_that_includes_itself_is_caught_before_it_duplicates() -> None:
    """The path starts with the page being fetched, so its own id is already on it."""
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="A", space="ENG", adf=with_include("a", title="A"))]
    )
    text, included, unresolved = await _body(instance)

    assert text.count('"a"') == 1
    assert included == []
    assert len(unresolved) == 1


async def test_expansion_stops_at_the_configured_depth() -> None:
    """A page including a page that includes a page is real; deeper is a template loop."""
    instance = FakeConfluence(
        pages=[
            FakePage(id="1", title="A", space="ENG", adf=with_include("a", title="B")),
            FakePage(id="2", title="B", space="ENG", adf=with_include("b", title="C")),
            FakePage(id="3", title="C", space="ENG", adf=paragraph("deepest")),
        ]
    )
    text, included, unresolved = await _body(instance, macro_depth=1)

    assert "b" in text
    assert "deepest" not in text
    assert included == ["2"]
    assert "depth limit of 1" in str(unresolved[0])


async def test_an_excerpt_include_takes_only_the_excerpt() -> None:
    """``excerpt-include`` renders the target's excerpt, not the whole page.

    Splicing the whole page instead would put content on the including page that a reader of
    it never sees, which is the same defect as the missing content, mirrored.
    """
    excerpt: dict[str, object] = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "bodiedExtension",
                "attrs": {"extensionKey": "excerpt", "parameters": {"macroParams": {}}},
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "the summary"}]}
                ],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "the long version"}]},
        ],
    }
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="A",
                space="ENG",
                adf=with_include("a", title="B", macro="excerpt-include"),
            ),
            FakePage(id="2", title="B", space="ENG", adf=excerpt),
        ]
    )
    text, included, unresolved = await _body(instance)

    assert "the summary" in text
    assert "the long version" not in text
    assert included == ["2"]
    assert unresolved == []


async def test_an_excerpt_include_of_a_page_with_no_excerpt_says_so() -> None:
    """Confluence renders nothing here too — but nothing plus a reason is diagnosable."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="A",
                space="ENG",
                adf=with_include("a", title="B", macro="excerpt-include"),
            ),
            FakePage(id="2", title="B", space="ENG", adf=paragraph("no excerpt here")),
        ]
    )
    _, included, unresolved = await _body(instance)

    assert included == []
    assert "defines no excerpt" in str(unresolved[0])


async def test_a_macro_naming_a_page_nobody_can_see_is_recorded() -> None:
    """Restricted or deleted, the outcome is the same and the index should be able to say so."""
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="A", space="ENG", adf=with_include("a", title="Missing"))]
    )
    _, included, unresolved = await _body(instance)

    assert included == []
    assert "Missing" in str(unresolved[0])


async def test_turning_expansion_off_is_recorded_rather_than_silent() -> None:
    """A legitimate choice that must not be a quiet one.

    The content is still absent from the chunk while a reader sees it on the page; the only
    difference from the accidental case is that somebody asked for it.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(id="1", title="A", space="ENG", adf=with_include("a", title="B")),
            FakePage(id="2", title="B", space="ENG", adf=paragraph("hidden")),
        ]
    )
    text, included, unresolved = await _body(instance, resolve_macros=False)

    assert "hidden" not in text
    assert included == []
    assert "resolve_macros" in str(unresolved[0])


# --- storage format ------------------------------------------------------------------------


async def _server_body(instance: FakeConfluence) -> tuple[str, Sequence[object]]:
    connector = await connected(instance, server_config(instance.base_url))
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == "1")
        raw = await connector.fetch(ref)
    finally:
        await connector.teardown()
    included = raw.metadata["included_pages"]
    assert isinstance(included, list)
    return raw.as_text(), included


async def test_a_storage_format_include_is_expanded_too() -> None:
    """Data Center has the same macros and no ADF, so it needs the same expansion."""
    instance = FakeConfluence(
        base_url=SERVER_BASE,
        pages=[
            FakePage(
                id="1",
                title="A",
                space="OPS",
                storage=storage_include("before", title="B"),
            ),
            FakePage(id="2", title="B", space="OPS", storage="<p>the rotation interval</p>"),
        ],
    )
    text, included = await _server_body(instance)

    assert text.startswith("<p>before</p>")
    assert text.endswith("<p>the rotation interval</p>")
    assert "ac:structured-macro" not in text
    assert included == ["2"]


async def test_everything_that_is_not_a_macro_is_returned_byte_for_byte() -> None:
    """The splice is into the original string, at spans a parser reported.

    Rebuilding the page from a parse tree would rewrite entities, attribute quoting and
    self-closing tags across the whole document in order to change one element — and the
    parser downstream would then be reading markup Confluence never sent.
    """
    original = (
        "<p>keep &amp; this</p><table><tr><td>cell</td></tr></table>"
        + storage_include("x", title="B")
        + "<p>tail</p>"
    )
    instance = FakeConfluence(
        base_url=SERVER_BASE,
        pages=[
            FakePage(id="1", title="A", space="OPS", storage=original),
            FakePage(id="2", title="B", space="OPS", storage="<p>in</p>"),
        ],
    )
    text, _ = await _server_body(instance)

    assert text.startswith("<p>keep &amp; this</p><table><tr><td>cell</td></tr></table><p>x</p>")
    assert text.endswith("<p>tail</p>")


async def test_a_storage_cycle_terminates_as_well() -> None:
    instance = FakeConfluence(
        base_url=SERVER_BASE,
        pages=[
            FakePage(id="1", title="A", space="OPS", storage=storage_include("a", title="B")),
            FakePage(id="2", title="B", space="OPS", storage=storage_include("b", title="A")),
        ],
    )
    text, included = await _server_body(instance)

    assert "<p>b</p>" in text
    assert included == ["2"]


def test_an_include_nested_inside_another_macro_is_still_found() -> None:
    """An include inside an ``info`` panel is still an include.

    A scan that reported only outermost elements would leave it unexpanded, which is the
    failure this module exists to prevent arriving through the scanner instead.
    """
    body = (
        '<ac:structured-macro ac:name="info"><ac:rich-text-body>'
        + storage_include("note", title="Detail")
        + "</ac:rich-text-body></ac:structured-macro>"
    )
    names = [macro.name for macro in storage_macros(body)]

    assert "include" in names
    assert "info" in names


def test_a_macro_body_is_located_without_its_own_tags() -> None:
    """What ``excerpt-include`` splices is the content, not the wrapper that marks it."""
    body = (
        '<ac:structured-macro ac:name="excerpt"><ac:rich-text-body>'
        "<p>summary</p></ac:rich-text-body></ac:structured-macro>"
    )

    assert excerpt_of_storage(body) == "<p>summary</p>"


def test_a_body_with_no_macros_is_left_exactly_as_it_arrived() -> None:
    assert storage_macros("<p>plain</p>") == []


@pytest.mark.parametrize(
    "parameter",
    ["", "page", "pageTitle"],
    ids=["default", "page", "pageTitle"],
)
async def test_the_target_is_read_from_whichever_parameter_carries_it(parameter: str) -> None:
    """Editors and templates disagree about where the target's name goes.

    Reading only the editor's own spelling leaves every templated page short by the content
    its macro was there to add, and nothing says so.
    """
    document: dict[str, object] = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "extension",
                "attrs": {
                    "extensionKey": "include",
                    "parameters": {"macroParams": {parameter: {"value": "B"}}},
                },
            }
        ],
    }
    instance = FakeConfluence(
        pages=[
            FakePage(id="1", title="A", space="ENG", adf=document),
            FakePage(id="2", title="B", space="ENG", adf=paragraph("found it")),
        ]
    )
    text, included, _ = await _body(instance)

    assert "found it" in text
    assert included == ["2"]


async def test_a_title_that_cannot_be_a_cql_literal_leaves_the_page_fetchable() -> None:
    """An unusable macro title is an unresolved macro, not a failed page.

    ``cql.quote`` refuses a line break or a NUL, and it is right to: escaping cannot save a
    literal that would terminate and continue as query syntax. But the refusal reached
    ``_page_id_of`` as a bare ``ValueError`` and failed the **whole page fetch**, not just the
    macro — and ``_included`` calls ``_page_id_of`` *outside* its own ``try``, which catches only
    the three body failures, so there was nowhere else for it to be caught.

    It takes only ordinary content to produce one. ``_StorageScanner.handle_data`` only strips,
    so an interior newline in a title parameter survives, and ``convert_charrefs`` turns a
    ``&#10;`` into one. A wiki page whose include macro has its title split across two lines made
    the page it sits on unfetchable — and the rest of that page is content somebody is looking
    for, which is the whole argument the unresolved path already makes.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Overview",
                space="ENG",
                adf=with_include("see", title="Release\nNotes"),
            )
        ]
    )

    text, included, unresolved = await _body(instance)

    assert "see" in text, "the page's own content still arrives"
    assert included == []
    assert unresolved, "the macro is reported as unresolved rather than disappearing silently"
