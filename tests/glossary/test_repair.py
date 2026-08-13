"""``document reindex --stale-glossary``: what it repairs, what it refuses, and what it costs.

Three of the fixtures here are the repository's own history rather than invented cases. #108,
#110 and #111 each landed against a corpus that kept what the previous rules produced, and each
is reproduced by putting that output back into the store under a superseded fingerprint and
asking the sweep to fix it. Where the old behaviour can be reproduced mechanically it is —
patching one rule back out is a stronger fixture than a hand-written row, because a
hand-written row asserts what somebody believed the old detector did.

The assertion the whole command stands on is
:func:`test_the_repair_reaches_no_parser_no_blob_and_no_embedder`. Every other test here would
pass against an implementation that quietly re-ingested each document, and an operator who ran
this on a corpus expecting a text pass would get a GPU-bound afternoon instead.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, override

import pytest

from manicule.core.errors import PolicyError
from manicule.core.glossary import DefinitionForm, GlossaryEntry
from manicule.ingest import reindex
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.reindex import (
    DETECTION_IS_OFF,
    plan_stale_glossary,
    redetect_glossary,
    redetect_stale_glossary,
)
from manicule.storage.docstore import SqliteDocStore
from tests.glossary import system

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk, Document

pytestmark = pytest.mark.usefixtures("store")

EXPANSION = "Network Operations Workspace"

SUPERSEDED = '{"detector":"deterministic","middleware":[],"rules":"sha256:0"}'
"""A fingerprint no installed detector produces. What a corpus indexed yesterday carries."""


def entry(document: Document, chunk: Chunk, acronym: str, expansion: str) -> GlossaryEntry:
    """One row as a superseded detector would have written it."""
    return GlossaryEntry(
        acronym=acronym,
        display=acronym,
        expansion=expansion,
        document_id=document.id,
        chunk_id=chunk.id,
        location=document.title,
        form=DefinitionForm.EM_DASH,
        confidence=0.95,
    )


async def indexed_under_old_rules(
    store: SqliteDocStore,
    source_id: str,
    line: str,
    entries: Sequence[GlossaryEntry] | None = None,
    *,
    title: str = "Glossary of terms",
) -> tuple[Document, list[Chunk]]:
    """A document whose chunks are current and whose glossary is not.

    The state a detector change actually leaves behind: ``parse_fp``, ``chunk_fp`` and
    ``embed_fp`` are whatever ingest wrote and are not touched, and only the glossary is stale.
    """
    document, chunks = await system.index(store, source_id, title, [line])
    rows = list(entries) if entries is not None else []
    await store.replace_glossary_entries(document.id, rows, fingerprint=SUPERSEDED)
    return document, chunks


class BoundedSelects(SqliteDocStore):
    """A store that fails rather than lets the sweep spin, and the reason it has to exist.

    The same double ``tests/ingest/test_reindex_sweep.py`` needs for the parse sweep, over the
    real store rather than a fake, and it is not optional. The loop here is the one thing that
    can go wrong by never *ending*: a cursor that does not advance past the documents a pass
    could not repair reads the same page for ever, and a test written without this bound hangs
    instead of failing — which is the worst available outcome, because CI reports a timeout with
    no indication of what was being asserted.

    Measured: with ``left_behind += 1`` deleted from the failure branch of
    :func:`~manicule.ingest.reindex.redetect_stale_glossary`, the termination test below ran for
    sixty seconds and was killed. With this bound it fails in a tenth of a second, naming the
    cursor.

    The bound is on the query count rather than on a clock, and it is checked synchronously
    inside the call: every ``await`` in that loop is over a local SQLite file and can complete
    without yielding, so ``asyncio.wait_for`` may never get the chance to fire its timer.
    """

    def __init__(self, *args: Any, ceiling: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ceiling = ceiling
        self.selects = 0

    @override
    async def select_documents(self, **kwargs: Any) -> Sequence[Document]:
        self.selects += 1
        if self.selects > self.ceiling:
            msg = (
                f"the sweep asked for the selection {self.selects} times, past the ceiling of "
                f"{self.ceiling}. Its cursor has stopped moving past the documents it could not "
                f"repair, so it is reading the same page for ever."
            )
            raise AssertionError(msg)
        return await super().select_documents(**kwargs)


async def sweep(
    store: SqliteDocStore, *, ceiling: int = 20, **kwargs: Any
) -> reindex.GlossarySweep:
    """Run the sweep against a store that cannot let it spin.

    **Every sweep in this file goes through the bound, not only the one about termination.** A
    cursor bug is not confined to the failure branch: with the fingerprint write removed from
    :meth:`~manicule.storage.glossary.GlossaryMixin.replace_glossary_entries`, no document ever
    leaves the selection and the *success* path loops for ever. Measured — that mutation was run
    against an unbounded ``test_a_second_run_selects_nothing`` and the process was still going
    ten minutes later. A bound only on the test that says "terminates" would have left every
    other test in this file able to hang instead of fail.
    """
    bounded = BoundedSelects(store.engine, workspace_id=store.workspace_id, ceiling=ceiling)
    return await redetect_stale_glossary(
        store=bounded,
        glossary=bounded,
        fingerprint=glossary_fingerprint(),
        **kwargs,
    )


# --- selection ------------------------------------------------------------------------------


async def test_a_current_parse_lineage_does_not_make_a_stale_glossary_current(
    store: SqliteDocStore,
) -> None:
    """Acceptance 1, and the sentence the whole feature exists to make false.

    The document is indexed, its text came out of the installed parser, its chunks came out of
    the installed chunker and its vectors came out of the installed embedder. Every check that
    existed before this change passes. Its definitions came out of rules that have since been
    corrected, and nothing said so.
    """
    document, chunks = await indexed_under_old_rules(store, "glossary", f"NOW — {EXPANSION}")
    await store.replace_glossary_entries(
        document.id,
        [entry(document, chunks[0], "NOW", "Nightly Operations Watchdog")],
        fingerprint=SUPERSEDED,
    )

    selected = await reindex.select(
        store,
        glossary_fingerprint=glossary_fingerprint(),
    )

    assert [found.id for found in selected] == [document.id]


async def test_a_document_the_current_detector_read_is_not_selected(
    store: SqliteDocStore,
) -> None:
    """The control. Without it this suite would pass against a selector that selects everything."""
    await system.index(store, "glossary", "Glossary of terms", [f"NOW — {EXPANSION}"])

    selected = await reindex.select(
        store,
        glossary_fingerprint=glossary_fingerprint(),
    )

    assert selected == []


async def test_a_document_with_no_recorded_lineage_is_selected(
    store: SqliteDocStore,
) -> None:
    """The migration policy, as a query.

    Every row predating the column carries ``NULL``, and the first release treats that as stale
    rather than as trusted. A predicate spelled ``glossary_fp = current`` would leave exactly
    this population out of the selection it exists to find — and SQL's three-valued logic makes
    that the *easy* mistake, because ``glossary_fp <> 'x'`` is unknown rather than true here.
    """
    document, _ = await system.index(store, "glossary", "Glossary of terms", [f"NOW — {EXPANSION}"])
    await _clear_lineage(store, document.id)

    selected = await reindex.select(
        store,
        glossary_fingerprint=glossary_fingerprint(),
    )

    assert [found.id for found in selected] == [document.id]


# --- what the repair fixes --------------------------------------------------------------------


async def test_a_malformed_expansion_is_replaced_by_the_current_detectors_reading(
    store: SqliteDocStore,
) -> None:
    """Acceptance 2, and the fixture is measured rather than imagined.

    Run against the module as it stood at ``ac09da8~1`` — the commit before #111 — the line
    below yields ``RNE = 'Regional Network Edge (e.g'`` at confidence 0.95. Today it yields
    ``'Regional Network Edge'``. The stored row here is the first of those, exactly, and the
    repair has to turn it into the second.
    """
    line = (
        "RNE - Regional Network Edge (e.g., a gateway): Connects a private network to an "
        "upstream network"
    )
    document, chunks = await system.index(store, "glossary", "Glossary of terms", [line])
    await store.replace_glossary_entries(
        document.id,
        [entry(document, chunks[0], "RNE", "Regional Network Edge (e.g")],
        fingerprint=SUPERSEDED,
    )

    report = await sweep(store)

    assert report.changed == 1
    stored = await store.glossary_entries(document.id)
    assert [(found.acronym, found.expansion) for found in stored] == [
        ("RNE", "Regional Network Edge")
    ]


async def test_a_false_heading_derived_entry_is_removed(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 3, with the old rule genuinely put back rather than described.

    Before #110 a heading and the paragraph under it were a definition whenever the page called
    itself a glossary: 0.45 for the form plus 0.15 for the page is exactly the threshold, and it
    cleared by being equal. Measured then, ten of ten heading sections were stored, including
    ``NOW`` meaning ``'The Network Operations Workspace holds the runbooks'``.

    The harm is not the wrong row. Two disagreeing expansions of one term are a conflict, and
    ``resolve_expansion`` correctly refuses to choose — so one page like this takes a *correct*
    definition elsewhere out of the answer entirely.
    """
    lines = "## NOW\nThe Network Operations Workspace holds the runbooks"
    from manicule.ingest import glossary as detector  # noqa: PLC0415

    def always(*_: object, **__: object) -> bool:
        return True

    monkeypatch.setattr(detector, "a_heading_may_define", always)
    document, chunks = await system.index(store, "handbook", "Glossary of terms", [lines])
    monkeypatch.undo()

    stale = await store.glossary_entries(document.id)
    assert [(found.acronym, found.form) for found in stale] == [("NOW", DefinitionForm.HEADING)], (
        "the fixture must reproduce the pre-#110 entry, or the repair has nothing to remove"
    )
    await store.replace_glossary_entries(document.id, stale, fingerprint=SUPERSEDED)
    del chunks

    report = await sweep(store)

    assert report.changed == 1
    assert await store.glossary_entries(document.id) == []


async def test_a_newly_supported_list_definition_is_added(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 4, with the old rule put back the same way.

    Before #108 a parser-written bullet occupied the position every written form anchors its
    term at, so ``- HDR - Hot Draining Router`` yielded nothing at all while the same line
    without the marker yielded one entry at confidence 0.95. A bulleted glossary was undetectable
    rather than partly detectable.
    """
    from manicule.ingest import glossary as detector  # noqa: PLC0415

    monkeypatch.setattr(detector, "_LIST_MARKER_RE", re.compile(r"^(?!x)x"))
    document, _ = await system.index(
        store, "glossary", "Glossary of terms", ["- HDR - Hot Draining Router"]
    )
    monkeypatch.undo()

    assert await store.glossary_entries(document.id) == [], (
        "the fixture must reproduce the pre-#108 gap, or the repair has nothing to add"
    )
    await store.replace_glossary_entries(document.id, [], fingerprint=SUPERSEDED)

    report = await sweep(store)

    assert report.changed == 1
    assert [
        (found.acronym, found.expansion) for found in await store.glossary_entries(document.id)
    ] == [("HDR", "Hot Draining Router")]


async def test_a_change_to_a_field_nothing_resolves_through_still_counts_as_changed(
    store: SqliteDocStore,
) -> None:
    """Every stored field is compared, not the ones a lookup goes through.

    Found by Copilot on this pull request, and it was right: an earlier ``_entry_shape`` omitted
    ``display`` and ``location`` because nothing resolves through them. Both are stored and both
    are shown — ``display`` is the source's own spelling, which a citation quotes *instead of*
    the normalised key, and ``location`` is where in the document the definition was found. A
    detector change that moved either would have rewritten what a reader is served while the
    sweep reported the document unchanged.

    ``SaFeR`` is the case that makes it concrete rather than theoretical: #103 exists so that a
    deliberately mixed-case spelling is readable, the key is ``SAFER`` either way, and the two
    are a different thing on the screen.
    """
    line = "SaFeR — Service Failure Reporter"
    document, chunks = await system.index(store, "glossary", "Glossary of terms", [line])
    detected = await store.glossary_entries(document.id)
    assert [found.display for found in detected] == ["SaFeR"], "the fixture must store a spelling"

    # The same term, the same key, the same expansion, the same everything a lookup reads — and
    # a display and a location the previous detector wrote differently.
    await store.replace_glossary_entries(
        document.id,
        [
            detected[0].model_copy(update={"display": "SAFER", "location": "somewhere else"}),
        ],
        fingerprint=SUPERSEDED,
    )

    report = await sweep(store)

    assert report.changed == 1, (
        "a rewritten display or location is a rewritten row, and reporting it as unchanged "
        "would hide the repair from the operator who ran it"
    )
    assert report.unchanged == 0
    repaired = await store.glossary_entries(document.id)
    assert (repaired[0].display, repaired[0].location) == ("SaFeR", "Glossary of terms")
    del chunks


async def test_a_document_producing_nothing_records_current_lineage(
    store: SqliteDocStore,
) -> None:
    """Acceptance 5. The empty result is a derived result, and the sweep has to say so.

    Without it this document is selected on every run for ever, and the corpus can never
    distinguish "nothing here is a definition" from "nobody has looked".
    """
    document, _ = await indexed_under_old_rules(
        store, "prose", "The scheduler restarts nightly, which is fine.", title="Runbook"
    )

    report = await sweep(store)

    assert (report.redetected, report.unchanged, report.changed) == (1, 1, 0)
    assert await store.glossary_entries(document.id) == []
    assert await store.glossary_lineage(document.id) == glossary_fingerprint().canonical()


# --- what the repair must not do ----------------------------------------------------------------


async def test_a_dry_run_touches_no_row_and_no_lineage(store: SqliteDocStore) -> None:
    """Acceptance 6, checked against every kind of state the real run writes."""
    document, chunks = await indexed_under_old_rules(store, "glossary", f"NOW — {EXPANSION}", [])
    before_chunks = [chunk.id for chunk in await store.document_chunks(document.id)]

    plan = await plan_stale_glossary(
        store=store,
        fingerprint=glossary_fingerprint(),
    )

    assert plan.dry_run
    assert plan.selected == 1
    assert (plan.redetected, plan.changed, plan.unchanged, plan.failed) == (0, 0, 0, 0)
    assert await store.glossary_entries(document.id) == []
    assert await store.glossary_lineage(document.id) == SUPERSEDED
    assert [chunk.id for chunk in await store.document_chunks(document.id)] == before_chunks
    del chunks


async def test_the_repair_reaches_no_parser_no_blob_and_no_embedder(
    store: SqliteDocStore,
) -> None:
    """Acceptance 7, and the assertion the command's whole reason for existing rests on.

    **The cost boundary is the signature.** :func:`redetect_glossary` is handed a store to read
    chunks through and a place to put entries, and nothing else: there is no ``pipeline``, no
    ``blobs``, no ``embedder`` and no ``vectors`` parameter, so it could not fetch a source, open
    a retained blob, run a parser or produce a vector if it tried. That is a stronger statement
    than counting calls on a double, because a double only proves that this particular run did
    not make them.

    It is checked as a signature rather than trusted, because a later change that added one of
    those arguments "just for logging" would pass every other test in this file.
    """
    import inspect  # noqa: PLC0415 - local to this assertion

    for verb in (redetect_glossary, redetect_stale_glossary, plan_stale_glossary):
        names = set(inspect.signature(verb).parameters)
        forbidden = names & {"pipeline", "blobs", "embedder", "vectors", "connector", "runner"}
        assert not forbidden, (
            f"{verb.__name__} takes {sorted(forbidden)}. The whole point of this rung is that "
            f"it cannot reach a parser, a blob, a connector or the model."
        )

    document, _ = await indexed_under_old_rules(store, "glossary", f"NOW — {EXPANSION}", [])
    report = await sweep(store)

    assert report.redetected == 1
    assert [found.acronym for found in await store.glossary_entries(document.id)] == ["NOW"]


async def test_a_detector_failure_keeps_the_last_servable_rows_and_the_stale_lineage(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 8, and requirement 7: fail closed, and be findable afterwards.

    Two things must not happen when detection raises, and both would be the default. The
    previous entries must not be erased — they are the ones a reader is being served, and a bug
    in a rule is not a reason to empty a working glossary. And the fingerprint must not advance,
    because a document stamped current while holding the old detector's rows is precisely the
    state that has no way of ever being noticed.
    """
    survivor = None

    def explode(*_: object, **__: object) -> list[GlossaryEntry]:
        msg = "catastrophic backtracking in a definition pattern"
        raise RuntimeError(msg)

    document, chunks = await system.index(
        store, "glossary", "Glossary of terms", [f"NOW — {EXPANSION}"]
    )
    survivor = await store.glossary_entries(document.id)
    assert survivor, "the fixture must have something to preserve"
    await store.replace_glossary_entries(document.id, survivor, fingerprint=SUPERSEDED)
    del chunks

    from manicule.ingest import glossary as detector  # noqa: PLC0415

    monkeypatch.setattr(detector, "detect_entries", explode)
    report = await sweep(store)
    monkeypatch.undo()

    assert (report.failed, report.redetected) == (1, 0)
    assert report.failures
    assert document.id in report.failures[0]
    assert await store.glossary_entries(document.id) == survivor, "a working glossary was erased"
    assert await store.glossary_lineage(document.id) == SUPERSEDED, (
        "the fingerprint advanced past a detector run that did not happen, so nothing will "
        "ever select this document again"
    )


async def test_a_document_re_ingested_underneath_the_sweep_costs_one_line_not_the_run(
    store: SqliteDocStore,
) -> None:
    """The race this sweep has and the parse sweep does not.

    An entry's ``chunk_id`` is a foreign key. This reads a document's chunks and then writes rows
    citing them, and a sync re-ingesting the same document in between replaces exactly those
    chunks — so the insert violates the key. The parse sweep runs through the pipeline and shares
    its embedding lock; this one takes no lock at all, because never reaching the model is the
    whole point of it.

    Reproduced by replacing the chunks between the read and the write, which is what a concurrent
    sync does. The sweep has to finish: the other documents are repaired, the raced one is named,
    and it is still selected afterwards so the next run picks it up.
    """
    raced, _ = await indexed_under_old_rules(store, "glossary-a", f"NOW — {EXPANSION}", [])
    await indexed_under_old_rules(store, "glossary-b", f"NOW — {EXPANSION}", [])

    original = SqliteDocStore.document_chunks
    seen: list[str] = []

    async def replace_underneath(inner: SqliteDocStore, document_id: str) -> Sequence[Chunk]:
        chunks = await original(inner, document_id)
        if document_id == raced.id and document_id not in seen:
            seen.append(document_id)
            # Exactly what a sync does at its commit point, and it takes with it the chunks the
            # entries about to be written are citing.
            await inner.replace_chunks(document_id, [])
        return chunks

    # Patched on the class rather than on the fixture's instance: the sweep runs against its own
    # bounded handle over the same database, so an instance patch would not be seen by it.
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(SqliteDocStore, "document_chunks", replace_underneath)
        report = await sweep(store)

    assert seen == [raced.id], "the fixture must actually race, or it tests nothing"
    assert report.failed == 1
    assert raced.id in report.failures[0]
    assert report.redetected == 1, "the document beside it was repaired regardless"
    assert await store.glossary_lineage(raced.id) == SUPERSEDED, (
        "a raced document must stay selected rather than be recorded as repaired"
    )


async def test_a_second_run_selects_nothing(store: SqliteDocStore) -> None:
    """Acceptance 9. Idempotence, including for the document that produced no entries.

    The zero-entry document is the one that makes this worth asserting: an implementation
    recording lineage only where there are rows would loop on it for ever, and would look
    perfectly correct on the page that has a definition on it.
    """
    await indexed_under_old_rules(store, "glossary", f"NOW — {EXPANSION}", [])
    await indexed_under_old_rules(store, "prose", "Nothing here defines anything.", title="Runbook")

    first = await sweep(store)
    second = await sweep(store)

    assert first.selected == 2
    assert first.redetected == 2
    assert second.selected == 0
    assert second.redetected == 0


async def test_the_sweep_ends_on_a_corpus_where_every_document_fails(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop's termination argument, exercised where it can actually go wrong.

    A repaired document leaves the selection and a failed one stays in it, so a cursor that did
    not advance past failures would read the same page for ever — and against an in-memory
    fixture that is a hang rather than a failure. The cursor advances by exactly the documents a
    pass left behind, which is what makes this return at all.
    """
    for index in range(3):
        await indexed_under_old_rules(store, f"glossary-{index}", f"NOW — {EXPANSION}", [])

    from manicule.ingest import glossary as detector  # noqa: PLC0415

    def explode(*_: object, **__: object) -> list[GlossaryEntry]:
        msg = "every document trips this rule"
        raise RuntimeError(msg)

    monkeypatch.setattr(detector, "detect_entries", explode)
    # Two pages of three documents, then the empty page that ends the loop: three queries. Four
    # is one more than the sweep needs, so a cursor that has stopped moving fails on its next
    # query rather than on a clock that may never get the chance to run.
    report = await sweep(store, batch=2, ceiling=4)

    assert (report.selected, report.failed, report.redetected) == (3, 3, 0)
    assert len(report.failures) == 3


async def test_a_page_boundary_does_not_skip_a_document(store: SqliteDocStore) -> None:
    """Paging while repairing, which is the shape that loses documents when it is wrong."""
    for index in range(5):
        await indexed_under_old_rules(store, f"glossary-{index}", f"NOW — {EXPANSION}", [])

    report = await sweep(store, batch=2)

    assert (report.selected, report.redetected) == (5, 5)
    assert await plan_stale_glossary(
        store=store,
        fingerprint=glossary_fingerprint(),
    ) == reindex.GlossarySweep(dry_run=True, selected=0)


# --- detection switched off -----------------------------------------------------------------


async def test_the_repair_refuses_while_detection_is_off_rather_than_running_it(
    store: SqliteDocStore,
) -> None:
    """Requirement 8's other half: the disabled state has to mean something on this path too.

    Recomputing would run rules the configuration says not to run. Stamping the disabled
    fingerprint instead would erase the record of which detector produced the entries still
    being served. Neither is a repair, so the command says so and names the setting.

    The plan refuses as well, and that is not an oversight: under a disabled detector the
    installed fingerprint *is* the disabled one, so the selection would be every document a
    detector ever read — reported as outstanding work this command would not do.
    """
    await indexed_under_old_rules(store, "glossary", f"NOW — {EXPANSION}", [])
    off = glossary_fingerprint(enabled=False)

    with pytest.raises(PolicyError, match="detect_on_ingest"):
        await redetect_stale_glossary(
            store=store,
            glossary=store,
            fingerprint=off,
        )
    with pytest.raises(PolicyError, match="detect_on_ingest"):
        await plan_stale_glossary(
            store=store,
            fingerprint=off,
        )
    assert "detect_on_ingest" in DETECTION_IS_OFF


async def _clear_lineage(store: SqliteDocStore, document_id: str) -> None:
    """Put a document back into the state the migration leaves every existing row in.

    Written through the model rather than through :meth:`set_lineage`, which deliberately reads
    ``None`` as "leave it alone" and so cannot clear anything.
    """
    from sqlalchemy import text  # noqa: PLC0415 - a storage extra

    async with store.engine.begin() as connection:
        await connection.execute(
            text("UPDATE documents SET glossary_fp = NULL WHERE id = :id"), {"id": document_id}
        )
