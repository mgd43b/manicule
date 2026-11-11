"""The Trilium connector over a synthetic server: ``docs/connectors/trilium.md`` §3-§6.

Every test runs the real client against :class:`tests.connectors.fake_trilium.FakeTrilium`, so the
real retry loop, headers and query encoding are the ones exercised.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from manicule.connectors.config import TriliumConfig
from manicule.connectors.errors import (
    ConnectorError,
    NotFoundError,
    RateLimitedError,
    RemoteError,
    UntrustedLinkError,
)
from manicule.connectors.trilium import (
    HTML,
    INVENTORY_MAX_AGE_SECONDS,
    MARKDOWN,
    PLAIN,
    TriliumConnector,
)
from manicule.connectors.trilium_client import NoteProtectedError, TriliumClient
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.sources import DiscoveredDoc
from manicule.testing import assert_connector_contract, closing
from tests.connectors.fake_trilium import BASE_URL, TOKEN, FakeNote, FakeTrilium, Waits, vault


def config(**overrides: object) -> TriliumConfig:
    settings: dict[str, object] = {"base_url": BASE_URL, "token": SecretStr(TOKEN)}
    settings.update(overrides)
    return TriliumConfig.model_validate(settings)


def connector_for(
    server: FakeTrilium,
    settings: TriliumConfig | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    max_note_bytes: int | None = None,
    clock: Callable[[], float] = lambda: 0.0,
) -> TriliumConnector:
    resolved = settings if settings is not None else config(base_url=server.base_url)
    client = TriliumClient(resolved, transport=server.transport(), sleep=sleep or Waits())
    return TriliumConnector(
        resolved, client, name="notes", max_note_bytes=max_note_bytes, clock=clock
    )


async def drain[T](stream: AsyncIterator[T]) -> list[T]:
    async with closing(stream) as items:
        return [item async for item in items]


async def discovered(connector: TriliumConnector) -> dict[str, DiscoveredDoc]:
    return {doc.source_id: doc for doc in await drain(connector.discover(None))}


async def reconciled(connector: TriliumConnector) -> list[str]:
    return await drain(connector.reconcile())


# --- the contract ---------------------------------------------------------------------------


async def test_the_connector_honors_the_contract() -> None:
    connector = connector_for(vault())
    try:
        await assert_connector_contract(connector)
    finally:
        await connector.teardown()


async def test_every_note_in_scope_is_discovered_with_its_title_and_a_uri() -> None:
    connector = connector_for(vault())
    found = await discovered(connector)

    assert set(found) == {"plan", "garden"}
    assert found["plan"].title == "Plan"
    assert found["plan"].ref.uri == f"{BASE_URL}/#root/plan"
    assert found["plan"].media_type == HTML
    assert found["plan"].version_token


# --- what is a document (§3) ----------------------------------------------------------------


async def test_containers_system_notes_protected_notes_and_unindexed_types_are_not_documents() -> (
    None
):
    server = vault()
    server.add(FakeNote("secret", "Secret", parents=("work",), protected=True))
    server.add(FakeNote("pic", "Picture", parents=("work",), type="image", mime="image/png"))
    server.add(FakeNote("cfg", "Config", parents=("_hidden",)))
    server.add(FakeNote("diagram", "Flow", parents=("work",), type="mermaid", mime="text/mermaid"))

    found = await discovered(connector_for(server))

    assert set(found) == {"plan", "garden", "diagram"}
    assert found["diagram"].media_type == PLAIN


async def test_a_note_that_is_also_cloned_into_the_system_tree_keeps_its_real_place() -> None:
    """Trilium clones a bookmarked note under its launch bar, inside ``_hidden``."""
    server = vault()
    server.notes["plan"].parents = ("work", "_hidden")

    connector = connector_for(server)
    found = await discovered(connector)
    raw = await connector.fetch(found["plan"].ref)

    assert "plan" in found
    provenance = Provenance.from_metadata(raw.metadata)
    assert provenance is not None
    assert provenance.source is not None
    assert provenance.source.section_path == ("Work",)


async def test_markdown_and_code_notes_are_delivered_as_text() -> None:
    server = vault()
    server.add(
        FakeNote(
            "md", "Readme", parents=("work",), type="code", mime="text/x-markdown", content="# Hi"
        )
    )
    server.add(
        FakeNote(
            "py", "Script", parents=("work",), type="code", mime="text/x-python", content="x = 1"
        )
    )
    connector = connector_for(server)
    found = await discovered(connector)

    assert found["md"].media_type == MARKDOWN
    assert found["py"].media_type == PLAIN
    raw = await connector.fetch(found["md"].ref)
    assert raw.as_bytes() == b"# Hi"


async def test_a_text_note_is_wrapped_so_its_title_is_the_first_heading() -> None:
    server = vault()
    server.notes["plan"].title = "Plan <1> & more"
    connector = connector_for(server)
    found = await discovered(connector)

    raw = await connector.fetch(found["plan"].ref)

    text = raw.as_text()
    assert raw.media_type == HTML
    assert "<h1>Plan &lt;1&gt; &amp; more</h1>" in text
    assert text.index("<h1>") < text.index("quarterly plan")


# --- identity and placement (§4) ------------------------------------------------------------


async def test_a_clone_is_one_document_placed_by_its_shortest_path_and_the_smaller_id() -> None:
    server = vault()
    server.add(FakeNote("deep", "Deep", parents=("work",), type="book"))
    server.add(FakeNote("clone", "Shared", parents=("deep", "home", "work")))
    connector = connector_for(server)

    found = await discovered(connector)
    raw = await connector.fetch(found["clone"].ref)

    assert list(found).count("clone") == 1
    provenance = Provenance.from_metadata(raw.metadata)
    assert provenance is not None
    assert provenance.source is not None
    # `work` and `home` are both one step from the top; the smaller id, `home`, places it.
    assert provenance.source.section_path == ("Home",)
    assert raw.metadata["trilium_parent_note_ids"] == ["deep", "home", "work"]


async def test_the_provenance_record_carries_the_note_s_own_address_and_dates() -> None:
    server = vault()
    server.notes["plan"].created = "2025-03-04 05:06:07.000Z"
    server.notes["plan"].modified = "2025-04-05 06:07:08.000Z"
    connector = connector_for(server)
    found = await discovered(connector)

    raw = await connector.fetch(found["plan"].ref)

    provenance = Provenance.from_metadata({PROVENANCE_KEY: raw.metadata[PROVENANCE_KEY]})
    assert provenance is not None
    assert provenance.source is not None
    assert provenance.source.canonical_uri == f"{BASE_URL}/#root/plan"
    assert provenance.source.source_id == "plan"
    assert provenance.source.created_at == datetime(2025, 3, 4, 5, 6, 7, tzinfo=UTC)
    assert provenance.source.modified_at == datetime(2025, 4, 5, 6, 7, 8, tzinfo=UTC)


async def test_a_title_a_record_cannot_hold_is_cleaned_and_the_note_is_still_indexed() -> None:
    server = vault()
    server.notes["plan"].title = "Plan\x1b[31m"
    connector = connector_for(server)

    found = await discovered(connector)
    raw = await connector.fetch(found["plan"].ref)

    assert "\x1b" not in found["plan"].title
    assert Provenance.from_metadata({PROVENANCE_KEY: raw.metadata[PROVENANCE_KEY]}) is not None


# --- scope (§5) -----------------------------------------------------------------------------


async def test_root_note_ids_admit_a_note_when_any_path_reaches_one() -> None:
    server = vault()
    server.notes["garden"].parents = ("home", "work")  # reachable from `work` too

    found = await discovered(connector_for(server, config(root_note_ids=["work"])))

    assert set(found) == {"plan", "garden"}


async def test_root_note_ids_leave_out_everything_elsewhere() -> None:
    found = await discovered(connector_for(vault(), config(root_note_ids=["work"])))

    assert set(found) == {"plan"}


async def test_a_label_excludes_the_note_and_everything_beneath_it() -> None:
    server = vault()
    server.notes["work"].labels = {"manicule_skip": ""}

    found = await discovered(connector_for(server))

    assert set(found) == {"garden"}


async def test_exclusion_is_judged_on_every_path_so_a_clone_cannot_escape_it() -> None:
    server = vault()
    server.notes["work"].labels = {"manicule_skip": ""}
    server.notes["plan"].parents = ("work", "home")  # also under an unmarked parent

    found = await discovered(connector_for(server))

    assert "plan" not in found


async def test_exclusion_wins_over_a_root_note_id() -> None:
    server = vault()
    server.notes["plan"].labels = {"manicule_skip": ""}

    found = await discovered(connector_for(server, config(root_note_ids=["work"])))

    assert found == {}


async def test_archived_notes_are_left_out_unless_asked_for() -> None:
    server = vault()
    server.notes["plan"].labels = {"archived": ""}

    assert "plan" not in await discovered(connector_for(server))
    assert "plan" in await discovered(connector_for(server, config(include_archived=True)))


# --- change detection (§6.2) ----------------------------------------------------------------


async def test_the_version_token_is_stable_until_something_the_document_says_changes() -> None:
    server = vault()
    first = (await discovered(connector_for(server)))["plan"].version_token
    again = (await discovered(connector_for(server)))["plan"].version_token

    server.edit("plan", "<p>new plan</p>")
    edited = (await discovered(connector_for(server)))["plan"].version_token

    assert first == again
    assert edited != first


async def test_renaming_a_note_changes_its_token_though_its_content_did_not() -> None:
    server = vault()
    before = (await discovered(connector_for(server)))["plan"].version_token

    server.rename("plan", "Roadmap")

    assert (await discovered(connector_for(server)))["plan"].version_token != before


async def test_renaming_an_ancestor_changes_every_descendant_s_token() -> None:
    """The descendant's own modification date does not move, but its breadcrumb does."""
    server = vault()
    before = await discovered(connector_for(server))

    server.rename("work", "Job")
    after = await discovered(connector_for(server))

    assert after["plan"].version_token != before["plan"].version_token
    assert after["garden"].version_token == before["garden"].version_token


async def test_the_watermark_is_offered_only_once_a_walk_has_run_to_the_end() -> None:
    connector = connector_for(vault())
    assert connector.watermark is None

    async with closing(connector.discover(None)) as stream:
        await anext(stream)
    assert connector.watermark is None

    await discovered(connector)
    assert connector.watermark is not None


# --- reconciliation and truncation (§6.1, §6.4) ---------------------------------------------


async def test_reconcile_names_every_note_in_scope_and_omits_a_deleted_one() -> None:
    server = vault()
    connector = connector_for(server)
    assert await reconciled(connector) == ["garden", "plan"]

    server.delete("garden")

    assert await reconciled(connector) == ["plan"]


async def test_a_note_that_leaves_scope_leaves_the_inventory() -> None:
    server = vault()
    connector = connector_for(server)
    server.notes["plan"].protected = True

    assert await reconciled(connector) == ["garden"]


async def test_a_response_that_may_have_been_cut_off_is_refused_by_reconcile_and_discover() -> None:
    server = vault()
    server.cap = 3
    connector = connector_for(server, config(base_url=BASE_URL, max_notes=3))

    with pytest.raises(ConnectorError, match="max_notes"):
        await reconciled(connector)
    with pytest.raises(ConnectorError, match="max_notes"):
        await drain(connector.discover(None))
    assert connector.watermark is None


async def test_a_response_just_under_the_limit_is_trusted() -> None:
    server = vault()  # root, _hidden, work, home, plan, garden: six notes
    connector = connector_for(server, config(max_notes=7))

    assert await reconciled(connector) == ["garden", "plan"]


def _entry(**changes: object) -> dict[str, object]:
    """A well-formed ETAPI note entry, with ``changes`` applied (``None`` removes a key)."""
    entry = FakeTrilium().serialize(FakeNote("x", "X"))
    entry["attributes"] = [{"type": "label", "name": "tag", "value": "a"}]
    for key, value in changes.items():
        if value is None:
            entry.pop(key)
        else:
            entry[key] = value
    return entry


MALFORMED: dict[str, dict[str, object]] = {
    "no note id": _entry(noteId=None),
    "empty note id": _entry(noteId=""),
    "no type": _entry(type=None),
    "empty type": _entry(type=""),
    "type is a number": _entry(type=7),
    "no isProtected": _entry(isProtected=None),
    "isProtected is a string": _entry(isProtected="false"),
    "parents is a string": _entry(parentNoteIds="root"),
    "a parent is a number": _entry(parentNoteIds=["root", 5]),
    "a parent is empty": _entry(parentNoteIds=[""]),
    "no attributes": _entry(attributes=None),
    "attributes is a string": _entry(attributes="label"),
    "an attribute is a string": _entry(attributes=["manicule_skip"]),
    "an attribute has no name": _entry(attributes=[{"type": "label", "value": ""}]),
    "an attribute has no type": _entry(attributes=[{"name": "manicule_skip", "value": ""}]),
    "a label has no value": _entry(attributes=[{"type": "label", "name": "manicule_skip"}]),
    "a label value is a number": _entry(
        attributes=[{"type": "label", "name": "manicule_skip", "value": 1}]
    ),
    "title is a number": _entry(title=3),
    "blobId is a number": _entry(blobId=3),
}


@pytest.mark.parametrize("entry", MALFORMED.values(), ids=MALFORMED.keys())
async def test_an_entry_that_is_not_shaped_like_a_note_fails_the_sweep(
    entry: dict[str, object],
) -> None:
    """Each of these would otherwise fail open: a note vanishes from the inventory, or a label
    that should have excluded a subtree is not read."""
    server = vault()
    server.extra_results.append(entry)

    with pytest.raises(ConnectorError, match="not shaped like an ETAPI note"):
        await reconciled(connector_for(server))


async def test_a_well_formed_entry_is_accepted() -> None:
    server = vault()
    server.extra_results.append(_entry())

    assert "x" in await reconciled(connector_for(server))


# --- fetch (§6.3) ---------------------------------------------------------------------------


async def test_fetching_reads_content_only_for_the_notes_asked_for() -> None:
    server = vault()
    connector = connector_for(server)
    found = await discovered(connector)
    assert server.content_requests() == []

    await connector.fetch(found["plan"].ref)

    assert server.content_requests() == ["plan"]


async def test_a_note_outside_scope_is_not_served_even_if_a_ref_names_it() -> None:
    server = vault()
    server.add(FakeNote("private", "Private", parents=("work",), labels={"manicule_skip": ""}))
    connector = connector_for(server)
    found = await discovered(connector)
    ref = found["plan"].ref.model_copy(update={"source_id": "private"})

    with pytest.raises(NotFoundError):
        await connector.fetch(ref)
    assert "private" not in server.content_requests()


async def test_a_note_protected_after_it_was_swept_is_gone_not_a_fault() -> None:
    server = vault()
    connector = connector_for(server)
    found = await discovered(connector)

    server.notes["plan"].protected = True

    with pytest.raises(NoteProtectedError):
        await connector.fetch(found["plan"].ref)


async def test_a_note_deleted_after_it_was_swept_is_gone_not_a_fault() -> None:
    server = vault()
    connector = connector_for(server)
    found = await discovered(connector)
    server.delete("plan")

    with pytest.raises(NotFoundError):
        await connector.fetch(found["plan"].ref)


async def test_a_note_over_the_fetch_cap_is_refused_while_streaming() -> None:
    server = vault()
    server.edit("plan", "x" * 5_000)
    connector = connector_for(server, max_note_bytes=1_000)
    found = await discovered(connector)

    with pytest.raises(ConnectorError, match="fetch cap"):
        await connector.fetch(found["plan"].ref)


# --- the HTTP layer (§2) --------------------------------------------------------------------


async def test_the_token_is_sent_as_a_bearer_credential_and_nothing_else_is() -> None:
    server = vault()
    await discovered(connector_for(server))

    assert set(server.authorizations) == {f"Bearer {TOKEN}"}
    assert all(path.startswith("/etapi/") for path, _ in server.requests)


async def test_a_context_path_is_kept() -> None:
    server = FakeTrilium(base_url="https://example.org/trilium")
    server.add(FakeNote("n", "N"))
    connector = connector_for(server, config(base_url="https://example.org/trilium/"))

    found = await discovered(connector)

    assert set(found) == {"n"}
    assert server.requests[0][0] == "/trilium/etapi/notes"
    assert found["n"].ref.uri == "https://example.org/trilium/#root/n"


async def test_a_throttled_request_waits_as_told_and_then_succeeds() -> None:
    server = vault()
    server.throttle = ["3", None]
    waits = Waits()

    found = await discovered(connector_for(server, sleep=waits))

    assert "plan" in found
    assert waits.seconds[0] == 3.0
    assert len(waits.seconds) == 2


async def test_a_pause_longer_than_the_ceiling_stops_the_run() -> None:
    server = vault()
    server.throttle = ["99999"]

    with pytest.raises(RateLimitedError):
        await discovered(connector_for(server))


async def test_a_rejected_token_is_named_as_such() -> None:
    server = vault()
    connector = connector_for(server, config(token=SecretStr("wrong")))

    with pytest.raises(ConnectorError, match="rejected the token"):
        await discovered(connector)


async def test_a_redirect_to_another_host_is_refused_and_the_token_goes_nowhere() -> None:
    server = vault()
    server.proxy_redirect = "https://login.example.net/sign-in"

    with pytest.raises(UntrustedLinkError, match=r"login\.example\.net"):
        await discovered(connector_for(server))
    assert len(server.requests) == 1


async def test_a_redirect_on_the_same_origin_is_refused_too() -> None:
    server = vault()
    server.proxy_redirect = f"{BASE_URL}/login"

    with pytest.raises(ConnectorError, match="does not redirect"):
        await discovered(connector_for(server))
    assert len(server.requests) == 1


async def test_a_proxy_sign_in_page_served_as_a_note_s_content_is_not_indexed() -> None:
    """A JSON endpoint refuses a sign-in page because it does not parse. A note's content is
    whatever its media type is, so for a note that is not HTML the sign-in page is caught by the
    media type disagreeing with the one the note was swept with."""
    server = vault()
    server.add(FakeNote("script", "Script", parents=("work",), type="code", mime="text/x-python"))
    connector = connector_for(server)
    found = await discovered(connector)
    server.signin_for_content = True

    with pytest.raises(ConnectorError, match="sign-in page"):
        await connector.fetch(found["script"].ref)


# --- review findings -------------------------------------------------------------------------


class Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_a_read_is_judged_on_a_sweep_no_older_than_the_limit() -> None:
    """A label added after discovery must reach a read, within a bounded time."""
    server = vault()
    clock = Clock()
    connector = connector_for(server, clock=clock)
    found = await discovered(connector)

    server.notes["plan"].labels = {"manicule_skip": ""}
    await connector.fetch(found["plan"].ref)  # still inside the age limit: served from the sweep

    clock.now = INVENTORY_MAX_AGE_SECONDS + 1
    with pytest.raises(NotFoundError):
        await connector.fetch(found["plan"].ref)


async def test_concurrent_reads_with_no_sweep_share_one_request() -> None:
    server = vault()
    connector = connector_for(server)
    refs = [(await discovered(connector_for(server)))[n].ref for n in ("plan", "garden")]
    server.requests.clear()

    await asyncio.gather(*(connector.fetch(ref) for ref in refs))

    searches = [path for path, _ in server.requests if path == "/etapi/notes"]
    assert len(searches) == 1


async def test_a_protected_folder_is_not_a_level_in_its_children_s_breadcrumb() -> None:
    server = vault()
    server.add(FakeNote("vault", "Vault", parents=("work",), type="book", protected=True))
    server.add(FakeNote("child", "Child", parents=("vault",)))
    connector = connector_for(server)
    found = await discovered(connector)

    raw = await connector.fetch(found["child"].ref)

    provenance = Provenance.from_metadata({PROVENANCE_KEY: raw.metadata[PROVENANCE_KEY]})
    assert provenance is not None
    assert provenance.source is not None
    assert provenance.source.section_path == ("Work",)


async def test_labels_that_share_a_name_are_all_kept() -> None:
    server = vault()
    server.notes["plan"].labels = {"tag": ["a", "b"]}
    connector = connector_for(server)
    found = await discovered(connector)

    raw = await connector.fetch(found["plan"].ref)

    assert raw.metadata["trilium_labels"] == {"tag": ["a", "b"]}


async def test_a_note_that_cannot_be_reached_from_the_top_is_counted_not_dropped_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = vault()
    server.add(FakeNote("a", "A", parents=("b",)))
    server.add(FakeNote("b", "B", parents=("a",)))  # a cycle: Trilium cannot make one
    with caplog.at_level("INFO", logger="manicule.connectors.trilium"):
        found = await discovered(connector_for(server))

    assert "a" not in found
    assert "2 not reachable" in caplog.text


async def test_a_note_type_the_connector_does_not_know_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = vault()
    server.add(FakeNote("fresh", "Fresh", parents=("work",), type="markdownNext"))
    with caplog.at_level("WARNING", logger="manicule.connectors.trilium"):
        found = await discovered(connector_for(server))

    assert "fresh" not in found
    assert "markdownNext" in caplog.text


# --- retries and the fetch cap, at their edges ------------------------------------------------


async def test_a_server_error_is_retried_and_then_succeeds() -> None:
    server = vault()
    server.failures = [503, 502]
    waits = Waits()

    found = await discovered(connector_for(server, sleep=waits))

    assert "plan" in found
    assert len(waits.seconds) == 2


async def test_a_server_error_on_every_attempt_stops_the_run() -> None:
    server = vault()
    server.failures = [503] * 10

    with pytest.raises(RemoteError, match="503"):
        await discovered(connector_for(server, config(max_retries=2)))
    assert len(server.requests) == 3


async def test_a_dropped_connection_is_retried_and_then_named() -> None:
    server = vault()
    server.failures = [None, None, None]

    with pytest.raises(ConnectorError, match="could not reach"):
        await discovered(connector_for(server, config(max_retries=2)))


async def test_throttling_on_every_attempt_stops_the_run() -> None:
    server = vault()
    server.throttle = ["1"] * 10

    with pytest.raises(RateLimitedError):
        await discovered(connector_for(server, config(max_retries=2)))
    assert len(server.requests) == 3


async def test_a_note_exactly_at_the_fetch_cap_is_read_and_one_byte_over_is_not() -> None:
    server = vault()
    server.edit("plan", "x" * 100)
    connector = connector_for(server, max_note_bytes=100)
    found = await discovered(connector)
    assert await connector.fetch(found["plan"].ref)

    server.edit("plan", "x" * 101)
    found = await discovered(connector)
    with pytest.raises(ConnectorError, match="fetch cap"):
        await connector.fetch(found["plan"].ref)
