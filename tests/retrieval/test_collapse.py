"""Near-duplicate collapse: a passage a conversation repeats is shown once, and nothing else moves.

``docs/retrieval.md`` §5.5 and ``docs/connectors/gmail.md`` §8. The documents are real rows in a
migrated store, because the stage reads each candidate's conversation from document metadata and
a fake store would answer whatever the test assumed it would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from manicule.core.content import BlockKind
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.retrieval.collapse import CollapseStage, shingles
from manicule.retrieval.config import CollapseConfig
from manicule.testing import assert_retrieval_stage_contract
from tests.retrieval.fakes import SCOPE, a_query
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from manicule.core.content import Document
    from manicule.storage.docstore import SqliteDocStore

ORIGINAL = (
    "We agreed the retry budget for the aurora ledger is four attempts with a thirty second "
    "backoff between them."
)
REPLY = (
    "Ownership of the aurora ledger passes to the platform team from next quarter, with Pat "
    "Rivera as the escalation contact."
)


async def message(
    store: SqliteDocStore, source_id: str, metadata: dict[str, Any] | None = None
) -> Document:
    document = make_document("mail", source_id, media_type="message/rfc822").model_copy(
        update={"metadata": metadata or {}}
    )
    await store.upsert_document(document)
    return document


def candidate(document: Document, text: str, score: float, position: int = 0) -> Candidate:
    chunk = make_chunk(document, position, text, heading_path=("Aurora ledger planning",))
    return Candidate(chunk=chunk, score=score, scores={"rrf": score})


def quoted(text: str, depth: int = 1) -> str:
    """``text`` as a plain-text reply quotes it, ``depth`` levels down, wrapped at 40."""
    marker = ">" * depth + " "
    words = text.split()
    lines = [" ".join(words[i : i + 7]) for i in range(0, len(words), 7)]
    return "\n".join(marker + line for line in lines)


async def test_the_stage_contract_holds(store: SqliteDocStore) -> None:
    first = await message(store, "m1", {"gmail_thread_id": "t1"})
    second = await message(store, "m2", {"gmail_thread_id": "t1"})
    await assert_retrieval_stage_contract(
        CollapseStage(docstore=store),
        a_query(),
        [candidate(first, ORIGINAL, 0.9), candidate(second, quoted(ORIGINAL), 0.8)],
    )


async def test_a_quotation_of_text_already_shown_in_the_thread_is_dropped(
    store: SqliteDocStore,
) -> None:
    original = await message(store, "m1", {"gmail_thread_id": "t1"})
    reply = await message(store, "m2", {"gmail_thread_id": "t1"})
    shown = candidate(original, ORIGINAL, 0.9)

    kept = await CollapseStage(docstore=store).run(
        a_query(), [shown, candidate(reply, f"On Monday Pat wrote:\n{quoted(ORIGINAL)}", 0.8)]
    )

    assert kept == [shown]


async def test_whichever_copy_ranked_first_is_kept_even_when_it_is_the_quotation(
    store: SqliteDocStore,
) -> None:
    """The known cost: a reply that outranks the original is kept, and the original goes.

    The passage is then cited to the reply quoting it. ``docs/connectors/gmail.md`` §8 measures
    how often, and why protecting the original was not adopted.
    """
    original = await message(store, "m1", {"gmail_thread_id": "t1"})
    reply = await message(store, "m2", {"gmail_thread_id": "t1"})
    quoting = candidate(reply, quoted(ORIGINAL), 0.9)

    kept = await CollapseStage(docstore=store).run(
        a_query(), [quoting, candidate(original, ORIGINAL, 0.8)]
    )

    assert kept == [quoting]


async def test_one_quotation_removes_a_later_quotation_of_the_same_text(
    store: SqliteDocStore,
) -> None:
    """The original need not be on the page for the copies of it to collapse to one."""
    first = await message(store, "m2", {"gmail_thread_id": "t1"})
    second = await message(store, "m3", {"gmail_thread_id": "t1"})
    shown = candidate(first, quoted(ORIGINAL), 0.9)

    kept = await CollapseStage(docstore=store).run(
        a_query(), [shown, candidate(second, quoted(ORIGINAL, depth=2), 0.8)]
    )

    assert kept == [shown]


@pytest.mark.parametrize(
    "corrected",
    [
        pytest.param(ORIGINAL.replace("four", "five"), id="a changed figure"),
        pytest.param(ORIGINAL.replace("thirty second", "thirty second jittered"), id="inserted"),
        pytest.param(quoted(ORIGINAL.replace("four", "five")), id="quoted, then corrected"),
        pytest.param(ORIGINAL.replace("thirty second ", ""), id="removed"),
        pytest.param(ORIGINAL.replace("four attempts", "six or seven tries"), id="rephrased"),
    ],
)
async def test_a_correction_is_never_dropped_as_a_copy_of_what_it_corrects(
    store: SqliteDocStore, corrected: str
) -> None:
    """One word in the middle differs, most trigrams survive, and the answer is different."""
    original = await message(store, "m1", {"gmail_thread_id": "t1"})
    reply = await message(store, "m2", {"gmail_thread_id": "t1"})
    ranked = [candidate(original, ORIGINAL, 0.9), candidate(reply, corrected, 0.8)]

    assert await CollapseStage(docstore=store).run(a_query(), ranked) == ranked


async def test_unshown_text_at_a_chunk_s_ends_is_a_seam_not_new_text(
    store: SqliteDocStore,
) -> None:
    """A quotation whose chunk began earlier and ended later than what was shown is a repeat."""
    original = await message(store, "m1", {"gmail_thread_id": "t1"})
    reply = await message(store, "m2", {"gmail_thread_id": "t1"})
    shown = candidate(original, f"{ORIGINAL} {REPLY}", 0.9)
    seamed = (
        f"Thanks Sam, on Monday Pat wrote:\n{quoted(ORIGINAL)}\n> {REPLY.split('.', maxsplit=1)[0]}"
    )

    kept = await CollapseStage(docstore=store).run(
        a_query(), [shown, candidate(reply, seamed, 0.8)]
    )

    assert kept == [shown]


async def test_a_seam_between_passages_shown_apart_is_not_mistaken_for_a_correction(
    store: SqliteDocStore,
) -> None:
    """The join of two shown passages is unshown, but no chunk above holds both sides of it."""
    first = await message(store, "m1", {"gmail_thread_id": "t1"})
    second = await message(store, "m2", {"gmail_thread_id": "t1"})
    third = await message(store, "m3", {"gmail_thread_id": "t1"})
    ranked = [
        candidate(first, ORIGINAL, 0.9),
        candidate(second, REPLY, 0.8),
        candidate(third, quoted(f"{REPLY} {ORIGINAL}"), 0.7),
    ]

    kept = await CollapseStage(docstore=store).run(a_query(), ranked)

    assert kept == ranked[:2]


async def test_a_candidate_that_adds_enough_new_text_is_kept(store: SqliteDocStore) -> None:
    """A reply that quotes the original and says something of its own is not a repeat."""
    original = await message(store, "m1", {"gmail_thread_id": "t1"})
    reply = await message(store, "m2", {"gmail_thread_id": "t1"})
    ranked = [
        candidate(original, ORIGINAL, 0.9),
        candidate(reply, f"{REPLY}\n\n{quoted(ORIGINAL)}", 0.8),
    ]

    assert await CollapseStage(docstore=store).run(a_query(), ranked) == ranked


async def test_repetition_is_counted_against_everything_kept_above_not_one_candidate(
    store: SqliteDocStore,
) -> None:
    """Two kept chunks between them can cover a third that neither covers alone."""
    first = await message(store, "m1", {"gmail_thread_id": "t1"})
    second = await message(store, "m2", {"gmail_thread_id": "t1"})
    third = await message(store, "m3", {"gmail_thread_id": "t1"})
    ranked = [
        candidate(first, ORIGINAL, 0.9),
        candidate(second, REPLY, 0.8),
        candidate(third, f"{quoted(REPLY)}\n>\n{quoted(ORIGINAL, depth=2)}", 0.7),
    ]

    kept = await CollapseStage(docstore=store).run(a_query(), ranked)

    assert kept == ranked[:2]


async def test_another_conversation_is_never_compared(store: SqliteDocStore) -> None:
    """The same words in a different thread may be exactly the difference being asked about."""
    here = await message(store, "m1", {"gmail_thread_id": "t1"})
    there = await message(store, "m2", {"gmail_thread_id": "t2"})
    ranked = [candidate(here, ORIGINAL, 0.9), candidate(there, ORIGINAL, 0.8)]

    assert await CollapseStage(docstore=store).run(a_query(), ranked) == ranked


async def test_the_same_thread_id_from_two_sources_is_two_conversations(
    store: SqliteDocStore,
) -> None:
    """A thread id is unique within the mailbox that issued it, not across two mailboxes."""
    mine = await message(store, "m1", {"gmail_thread_id": "t1"})
    theirs = make_document("other-mail", "m1", media_type="message/rfc822").model_copy(
        update={"metadata": {"gmail_thread_id": "t1"}}
    )
    await store.upsert_document(theirs)
    ranked = [candidate(mine, ORIGINAL, 0.9), candidate(theirs, quoted(ORIGINAL), 0.8)]

    assert await CollapseStage(docstore=store).run(a_query(), ranked) == ranked


async def test_a_document_without_a_conversation_passes_through_untouched(
    store: SqliteDocStore,
) -> None:
    """Every source that records no thread sees the ranking it had before the stage existed."""
    page = await message(store, "p1")
    copy = await message(store, "p2")
    threaded = await message(store, "m1", {"gmail_thread_id": "t1"})
    ranked = [
        candidate(page, ORIGINAL, 0.9),
        candidate(copy, ORIGINAL, 0.8),
        candidate(threaded, ORIGINAL, 0.7),
    ]

    assert await CollapseStage(docstore=store).run(a_query(), ranked) == ranked


async def test_survivors_keep_their_order_and_their_scores(store: SqliteDocStore) -> None:
    """The stage removes; it neither reorders nor rescores."""
    original = await message(store, "m1", {"gmail_thread_id": "t1"})
    reply = await message(store, "m2", {"gmail_thread_id": "t1"})
    other = await message(store, "o1")
    ranked = [
        candidate(original, ORIGINAL, 0.9),
        candidate(reply, quoted(ORIGINAL), 0.8),
        candidate(other, REPLY, 0.7),
    ]

    kept = await CollapseStage(docstore=store).run(a_query(), ranked)

    assert kept == [ranked[0], ranked[2]]
    assert [(c.score, c.scores) for c in kept] == [(0.9, {"rrf": 0.9}), (0.7, {"rrf": 0.7})]


async def test_the_field_naming_a_conversation_is_configured(store: SqliteDocStore) -> None:
    first = await message(store, "m1", {"conversation": "c1", "gmail_thread_id": "t1"})
    second = await message(store, "m2", {"conversation": "c1", "gmail_thread_id": "t2"})
    ranked = [candidate(first, ORIGINAL, 0.9), candidate(second, quoted(ORIGINAL), 0.8)]

    default = await CollapseStage(docstore=store).run(a_query(), ranked)
    configured = await CollapseStage(
        docstore=store, config=CollapseConfig(thread_key="conversation")
    ).run(a_query(), ranked)

    assert default == ranked
    assert configured == ranked[:1]


class RecordingStore:
    """The real store, with every document read it is asked for written down."""

    def __init__(self, store: SqliteDocStore) -> None:
        self.store = store
        self.filters: list[Filter] = []

    async def list_documents(self, scope: Filter, **options: Any) -> Any:
        self.filters.append(scope)
        return await self.store.list_documents(scope, **options)


async def test_the_metadata_read_is_no_wider_than_the_query(store: SqliteDocStore) -> None:
    """Narrowed to the candidates' documents, and to whatever the query was already narrowed to."""
    first = await message(store, "m1", {"gmail_thread_id": "t1"})
    second = await message(store, "m2", {"gmail_thread_id": "t1"})
    recording = RecordingStore(store)
    query = Query(
        text="retry budget",
        limit=3,
        filter=Filter(workspace_ids=SCOPE, document_ids=frozenset({first.id})),
    )

    kept = await CollapseStage(docstore=recording).run(  # pyright: ignore[reportArgumentType]
        query, [candidate(first, ORIGINAL, 0.9), candidate(second, quoted(ORIGINAL), 0.8)]
    )

    [read] = recording.filters
    assert read.workspace_ids == SCOPE
    assert read.document_ids == frozenset({first.id})
    assert len(kept) == 2, "a document the query cannot see names no conversation"


@pytest.mark.parametrize(
    "restriction",
    [
        pytest.param({"kinds": frozenset({BlockKind.PROSE})}, id="kinds"),
        pytest.param({"langs": frozenset({"en"})}, id="langs"),
    ],
)
async def test_a_chunk_level_restriction_is_not_sent_to_the_document_read(
    store: SqliteDocStore, restriction: dict[str, Any]
) -> None:
    """A list of documents cannot apply ``kinds`` or ``langs``, and the candidates meet them."""
    first = await message(store, "m1", {"gmail_thread_id": "t1"})
    second = await message(store, "m2", {"gmail_thread_id": "t1"})
    recording = RecordingStore(store)
    query = Query(text="retry budget", limit=3, filter=Filter(workspace_ids=SCOPE, **restriction))

    kept = await CollapseStage(docstore=recording).run(  # pyright: ignore[reportArgumentType]
        query, [candidate(first, ORIGINAL, 0.9), candidate(second, quoted(ORIGINAL), 0.8)]
    )

    [read] = recording.filters
    assert not read.kinds
    assert not read.langs
    assert len(kept) == 1, "the metadata was read, and the quotation dropped"


async def test_nothing_to_compare_reads_nothing(store: SqliteDocStore) -> None:
    recording = RecordingStore(store)
    assert await CollapseStage(docstore=recording).run(a_query(), []) == []  # pyright: ignore[reportArgumentType]
    assert recording.filters == []


@pytest.mark.parametrize(
    "form",
    [
        pytest.param(quoted(ORIGINAL), id="quoted once"),
        pytest.param(quoted(ORIGINAL, depth=3), id="quoted three deep"),
        pytest.param(quoted(ORIGINAL).replace("> ", ">"), id="markers without a space"),
        pytest.param(quoted(ORIGINAL).replace(">", "> >"), id="spaced nested markers"),
        pytest.param(ORIGINAL.upper(), id="case"),
        pytest.param("\n".join(ORIGINAL.split()), id="wrapped differently"),
    ],
)
def test_quote_markers_case_and_wrapping_are_not_text(form: str) -> None:
    """An HTML-only reply arrives with no markers at all, so a quotation must equal its source."""
    assert shingles(form) == shingles(ORIGINAL)


def test_very_short_text_still_has_an_identity() -> None:
    assert shingles("ok") == frozenset({("ok",)})
    assert shingles("> ") == frozenset()
