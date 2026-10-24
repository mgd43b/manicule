"""The measured evaluation the ticket asks for, over a labeled corpus rather than the examples.

Two halves, and both are needed for the same reason. **Detection** is measured line by line
against :mod:`tests.glossary.skeleton_corpus`, which labels every line including the ones that
must produce nothing — precision on a corpus of definitions only is a number a detector that
admits everything scores perfectly on. **Retrieval** is measured end to end through the shipped
retriever, because a definition detected and then not reachable by a question is a definition
nobody has.

Every figure in these docstrings was produced by the assertions below and can be reproduced by
running them. The assertions are the *claims* — stated where a regression fails them and stated
loosely enough not to pin an arithmetic accident — and the docstrings carry the values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from manicule.core.retrieval import Query
from manicule.ingest.glossary import MIN_SKELETON_LENGTH, detect_entries
from tests.evaluation.fakes import BagOfWordsEmbedder
from tests.glossary import skeleton_corpus as labeled
from tests.glossary import system
from tests.glossary.corpus import passages
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk
    from manicule.core.glossary import GlossaryEntry
    from manicule.retrieval.retriever import Retriever
    from manicule.storage.docstore import SqliteDocStore

LIMIT = 10


# --- the detection half -------------------------------------------------------------------------


def _detected(line: str) -> list[GlossaryEntry]:
    """Every entry one labeled line yields, **on a page that says it is a glossary**.

    The placement is not incidental. Off a glossary page a spaced hyphen scores 0.45 against a
    0.60 threshold, so every negative in this corpus would be refused by arithmetic that has
    nothing to do with the matcher and the measurement would report a precision it had not
    earned. On one, the context evidence supplies the missing 0.15, the scoring gate admits every
    line, and what refuses the negatives is the expansion rules alone.
    """
    document = make_document(source_id="glossary", title=labeled.GLOSSARY_TITLE)
    chunk = make_chunk(document, 0, line, heading_path=(labeled.GLOSSARY_TITLE,))
    return detect_entries([chunk], title=labeled.GLOSSARY_TITLE)


def test_detection_precision_and_recall_over_the_labeled_corpus() -> None:
    """The headline figures, and the baseline they are an improvement on.

    Over 35 labeled lines — 18 positive, 17 negative — measured against this matcher and against
    ``origin/main``'s:

    =========================  ===============  ==================
    Measure                    ``origin/main``  this change
    =========================  ===============  ==================
    Entries detected           15               18
    Detection precision        15/15 = 1.000    18/18 = **1.000**
    Detection recall           15/18 = 0.833    18/18 = **1.000**
    Expansion-boundary prec.   14/15 = 0.933    18/18 = **1.000**
    False-positive entries     0                **0**
    =========================  ===============  ==================

    The three the narrow matcher missed are one per convention it cannot read — ``SORT — SecOps
    Reliability Toolkit, …`` and ``CIRCA — CloudInfra Retention Capacity Auditor. …`` need the
    compound split, ``AuDiT — Automated Data Trail, …`` needs the skeleton — and each is missed
    rather than mis-cut, because a right-hand side over ``MAX_EXPANSION_WORDS`` with no initials
    agreement produces no entry at all. Its one boundary error is ``SaFeR``, which it recorded
    with all ten words of ``Service Failure Reporter, a component that groups related failures
    together``.

    **The negative count is what makes the precision figure worth reading**, and it was measured
    against this matcher rather than against the old one: seventeen lines, of which eleven are
    prose on a glossary page and six are constructed against the two new rules specifically. Zero
    are admitted.
    """
    positives = [item for item in labeled.LABELED if item.positive]
    detections = 0
    right_term = 0
    right_boundary = 0
    false_positives: list[tuple[str, str, str]] = []

    for item in labeled.LABELED:
        found = _detected(item.line)
        detections += len(found)
        hit = [entry for entry in found if entry.acronym == item.acronym] if item.positive else []
        if hit:
            right_term += 1
            right_boundary += hit[0].expansion == item.expansion
        false_positives.extend(
            (entry.acronym, entry.expansion, item.line)
            for entry in found
            if entry.acronym != item.acronym
        )

    assert not false_positives, f"the widened matcher admitted prose: {false_positives}"
    assert detections == 18
    assert right_term == len(positives) == 18, "recall is 1.000 over the labeled positives"
    assert right_boundary == right_term, "every detected entry stored the labeled core expansion"


@pytest.mark.parametrize(
    "item",
    [*labeled.CONVENTIONAL, *labeled.COMPOUND, *labeled.STYLIZED, *labeled.DESCRIBED],
    ids=lambda item: f"{item.category}-{item.acronym}",
)
def test_each_accepted_representation_is_read(item: labeled.Labeled) -> None:
    """Every positive, one case each, so a convention that stops working is named in the failure.

    The ticket asks for every accepted representation to be documented. This is that list executed
    rather than described: ordinary word initials, compound components, and the uppercase
    skeleton, each on a bare line and each again carrying a description.
    """
    found = _detected(item.line)

    assert [(entry.acronym, entry.expansion) for entry in found] == [(item.acronym, item.expansion)]


@pytest.mark.parametrize(
    "item",
    [*labeled.COMPOUND_NEGATIVES, *labeled.SKELETON_NEGATIVES, *labeled.ORDINARY_MENTIONS],
    ids=lambda item: f"{item.category}-{item.line[:24]}",
)
def test_each_negative_is_refused_by_the_widened_matcher(item: labeled.Labeled) -> None:
    """The other half, and the half the widening had to be measured against.

    Each of these is the negative case that limits one accepted representation, on a glossary page
    where the scoring gate admits it and only the expansion rules can refuse it.
    """
    assert _detected(item.line) == []


@pytest.mark.parametrize(("line", "acronym", "expansion"), labeled.KEPT_WHOLE)
def test_no_initials_agreement_still_means_no_cut(line: str, acronym: str, expansion: str) -> None:
    """The conservative half of the boundary rule, measured against the *wider* matcher.

    Nothing here says where the expansion ends, so the whole right-hand side is kept. The failure
    this catches leaves the entry count unchanged and only the stored text wrong, which is why it
    is asserted on the text.

    **What reddens it, established by running the mutations rather than by reasoning.** Cutting at
    the first description boundary unconditionally — the wrong fix ``core_expansion`` warns about —
    turns all three into ``central processor``, ``MicroObject Storage Index`` and ``Retention
    Capacity``. That is the guard.

    It is **not** sensitive to a free-subsequence matcher, and an earlier version of this docstring
    claimed it was. Replacing :func:`~manicule.ingest.glossary.initials_match` with a scan leaves
    this test green, because ``core_expansion`` asks about the *whole* right-hand side first: the
    scan accepts that, the whole is returned, and the expected value is the whole. The failure is
    real and lands somewhere else. Requirement 3 is held down by
    ``test_a_free_subsequence_scan_would_match_and_this_matcher_refuses``, whose fixtures are built
    so that the whole right-hand side is over the length bound and cannot be returned instead.
    """
    found = _detected(line)

    assert [(entry.acronym, entry.expansion) for entry in found] == [(acronym, expansion)]


# --- the retrieval half -------------------------------------------------------------------------


async def _corpus(store: SqliteDocStore) -> list[Chunk]:
    """Two glossary pages and sixty handbook passages, each page one chunk.

    The handbook is :func:`tests.glossary.corpus.passages` — forty-five ordinary uses of an
    English word and fifteen usages of an acronym — reused rather than rewritten because its whole
    purpose is to be the dilution a definition has to be found through, and that purpose is
    independent of which terms are being defined. It is read, never modified: the two documents
    whose cosines are recorded elsewhere in this suite are not among the ones built here.
    """
    chunks: list[Chunk] = []
    for source_id, title, texts in (
        ("skeleton-glossary", labeled.GLOSSARY_TITLE, [labeled.definitions_page()]),
        ("skeleton-second", labeled.SECOND_TITLE, [labeled.second_page()]),
        (
            "skeleton-handbook",
            labeled.HANDBOOK_TITLE,
            [*passages(), *labeled.HOMOGRAPH_USES],
        ),
    ):
        _, indexed = await system.index(store, source_id, title, texts)
        chunks.extend(indexed)
    return chunks


def _ask(text: str) -> Query:
    return Query(text=text, limit=LIMIT, filter=system.query_filter())


async def _retrievers(
    store: SqliteDocStore, chunks: Sequence[Chunk]
) -> tuple[Retriever, Retriever]:
    """The baseline and the glossary-wired pipeline, identical but for the glossary."""
    baseline = await system.retriever_over(store, BagOfWordsEmbedder(), chunks, glossary=False)
    wired = await system.retriever_over(store, BagOfWordsEmbedder(), chunks)
    return baseline, wired


async def test_hit_rate_for_definitional_questions(store: SqliteDocStore) -> None:
    """Whether a question about a term reaches the passage that defines it.

    Fourteen definitional questions over eleven terms, counting the defining chunk inside the top
    *k* of the assembled context:

    ====  =============  ====================
    k     Baseline       Glossary wired
    ====  =============  ====================
    1     9/14 = 0.643   14/14 = **1.000**
    3     12/14 = 0.857  14/14 = **1.000**
    10    14/14 = 1.000  14/14 = **1.000**
    ====  =============  ====================

    **Both columns moved with the lexical leg's match rule, and the baseline is why.** Before #481
    the baseline was 0, 0 and 2 of 14: ``What is SORT?`` asked the lexical leg for "what" and
    "is" as well, matched nothing, and left the definition to the stand-in embedder's similarity.
    Requiring only the content words (#482) made it 7, 9 and 13; matching any of them (round 3 of
    ``tests/benchmarks/lexical_questions.py``) makes it 9, 12 and 14. The glossary still adds
    five questions at k = 1 and two at k = 3, because a lookup fetches the defining chunk by id
    rather than leaving it to win a ranking.

    Thirteen of the wired fourteen are rank 1 by **lookup**: an alias hit fetches the defining
    chunk by id and promotes it rather than making it win a similarity contest.

    **The fourteenth is the interesting one and it is kept rather than removed.** ``What does
    SecOps Reliability Toolkit stand for?`` names the *expansion* and never writes the term, so no
    alias fires and nothing is promoted: the glossary is keyed by term, and a question that
    already knows the expansion gets no help from it. It reaches rank 1 anyway, because the
    lexical leg now matches the expansion's own words on the glossary page — search doing its
    job, not the glossary. While every word was required it arrived at rank 10, and that limit of
    the design is still there for a question whose words the page does not hold.

    Three of these ask in a case the source never wrote — ``what is sort?``, ``what does safer
    stand for?``, ``what is recap?`` — and they resolve because the *key* is normalized. That is
    requirement 1 measured rather than asserted: the skeleton is a comparison form and resolves
    nothing, so a lookup that had quietly started using it would fail all three.
    """
    chunks = await _corpus(store)
    entries = await system.entries_by_acronym(store, chunks)
    baseline, wired = await _retrievers(store, chunks)

    hits = {1: 0, 3: 0, 10: 0}
    baseline_hits = {1: 0, 3: 0, 10: 0}
    for text, acronym in labeled.DEFINITIONAL_QUERIES:
        defining = entries[acronym].chunk_id
        after = system.rank_of((await wired.retrieve(_ask(text))).context.passages, defining)
        before = system.rank_of((await baseline.retrieve(_ask(text))).context.passages, defining)
        for cutoff in hits:
            hits[cutoff] += after is not None and after <= cutoff
            baseline_hits[cutoff] += before is not None and before <= cutoff

    assert hits == {1: 14, 3: 14, 10: 14}
    assert baseline_hits == {1: 9, 3: 12, 10: 14}, (
        "the baseline moved, so what the glossary adds over search has to be measured again"
    )


async def test_unsupported_questions_are_rejected(store: SqliteDocStore) -> None:
    """Questions this corpus cannot answer must fire nothing. Rejection rate **6/6 = 1.000**.

    Two failure modes on purpose. ``What is ZZQX?`` is a definitional frame around a token that is
    not in the glossary, which is the case a lookup could get wrong by being generous about keys.
    The other five are fluent English about subjects the corpus never mentions — they retrieve
    passages, because retrieval always retrieves something, and what is asserted is that no
    glossary entry fires and no explicit definition is claimed.
    """
    chunks = await _corpus(store)
    _, wired = await _retrievers(store, chunks)

    fired: list[str] = []
    for text in labeled.UNSUPPORTED_QUERIES:
        result = await wired.retrieve(_ask(text))
        assert result.confidence is not None
        if (result.expansion is not None and result.expansion.fired) or (
            result.confidence.explicit_definition
        ):
            fired.append(text)

    assert not fired, f"an unsupported question fired the glossary: {fired}"


async def test_using_a_defined_token_is_not_asking_what_it_means(store: SqliteDocStore) -> None:
    """Requirement 8's negative half: exact lexical overlap is not an explicit definition.

    Four questions that contain a defined term and ask nothing about it. None is classified, and
    the reason is the definitional frame rather than the recorded match reason — ``sort the queue
    by age`` and ``What is SORT?`` can record the same reason, so the reason cannot tell them
    apart.
    """
    chunks = await _corpus(store)
    _, wired = await _retrievers(store, chunks)

    classified: list[str] = []
    for text in labeled.NON_DEFINITIONAL_QUERIES:
        result = await wired.retrieve(_ask(text))
        assert result.confidence is not None
        if result.confidence.explicit_definition:
            classified.append(text)

    assert not classified, f"a use of a term was classified as a definition: {classified}"


async def test_no_confidence_band_moves(store: SqliteDocStore) -> None:
    """The band is the same with the glossary wired as without it, for all 24 queries.

    **Band changes: none. Score changes: one, and it is the expanded search's passages.** Measured
    over every query in the corpus — fourteen definitional, six unsupported, four
    non-definitional. That is requirement 8, and it holds by construction: a promoted passage
    carries no leg score, so it contributes nothing to the arithmetic in either direction, and
    ``explicit_definition`` is a boolean that replaces a *reason* string.

    The one score that moves is ``the audit trail is on the same volume as the index``, 0.610 to
    0.625 and ``medium`` both times. ``audit`` names a glossary term, so the expanded text runs a
    second search, and since the lexical leg matches any content word (``docs/retrieval.md``
    §4.1) that search returns passages the first did not. A different context is different
    evidence; while every word was required, the second search matched nothing lexically here
    and no score moved at all.

    **What is asserted exactly is the substitution this feature is built to avoid.** Every dense
    score in the wired context must be one a search actually measured — the better of the
    original and the expanded pass, per chunk — so a detection confidence quietly added to a
    cosine fails here however the totals move. And a score may move only where the context did.
    """
    chunks = await _corpus(store)
    baseline, wired = await _retrievers(store, chunks)

    queries = [
        *(text for text, _ in labeled.DEFINITIONAL_QUERIES),
        *labeled.UNSUPPORTED_QUERIES,
        *labeled.NON_DEFINITIONAL_QUERIES,
    ]
    bands: list[str] = []
    unexplained: list[str] = []
    invented: list[tuple[str, str]] = []
    moved: list[str] = []
    for text in queries:
        before = await baseline.retrieve(_ask(text))
        after = await wired.retrieve(_ask(text))
        assert before.confidence is not None
        assert after.confidence is not None
        if after.confidence.band is not before.confidence.band:
            bands.append(text)
        measured: dict[str, float] = {}
        searches = [before]
        report = after.trace.glossary
        if report is not None and report.second_pass:
            searches.append(await baseline.retrieve(_ask(report.expanded_query)))
        for search in searches:
            for candidate in search.candidates:
                if "dense" in candidate.scores:
                    measured[candidate.chunk.id] = max(
                        measured.get(candidate.chunk.id, -1.0), candidate.scores["dense"]
                    )
        for passage in after.context.passages:
            if "dense" in passage.scores and passage.scores["dense"] != pytest.approx(
                measured.get(passage.chunk.id)
            ):
                invented.append((text, passage.chunk.id))
        if after.confidence.score != pytest.approx(before.confidence.score):
            moved.append(text)
            shown = [p.chunk.id for p in after.context.passages if "glossary" not in p.scores]
            if shown == [p.chunk.id for p in before.context.passages]:
                unexplained.append(text)

    assert not bands, f"the glossary moved a confidence band: {bands}"
    assert not invented, f"a dense score no search measured: {invented}"
    assert not unexplained, f"a score moved with the same evidence shown: {unexplained}"
    assert moved == ["the audit trail is on the same volume as the index"]


async def test_two_definitions_of_one_key_stay_a_conflict(store: SqliteDocStore) -> None:
    """Requirement 7, measured where the new comparison forms could have broken it.

    ``HALO`` is defined on both pages, by two phrases that each spell it through ordinary word
    initials. Neither display spelling nor skeleton is consulted to break the tie, because there
    is no tie-breaker: the query expands to nothing and reports both sources.
    """
    chunks = await _corpus(store)
    _, wired = await _retrievers(store, chunks)

    result = await wired.retrieve(_ask("What is HALO?"))

    assert result.expansion is not None
    assert not result.expansion.fired, "a contested term must expand to nothing"
    assert [conflict.key for conflict in result.expansion.conflicts] == ["HALO"]
    assert len(result.expansion.conflicts[0].expansions) == 2


async def test_the_skeleton_is_never_a_lookup_key(store: SqliteDocStore) -> None:
    """Requirement 1, at the only place it can actually be violated.

    ``SaFeR`` stores under ``SAFER`` and its skeleton is ``SFR``. Both are well-formed keys, so a
    detector that recorded the skeleton as an alias would produce a glossary that answered ``what
    is SFR?`` — and nothing in the detection tests would notice, because the entry would look
    correct. Asking the retriever is what separates a comparison form from a key.
    """
    chunks = await _corpus(store)
    entries = await system.entries_by_acronym(store, chunks)
    _, wired = await _retrievers(store, chunks)

    assert entries["SAFER"].display == "SaFeR"
    assert "SFR" not in entries
    assert entries["SAFER"].keys == ("SAFER",)

    result = await wired.retrieve(_ask("What is SFR?"))

    assert result.expansion is None or not result.expansion.fired


async def test_the_promoted_passage_still_carries_what_the_expansion_was_trimmed_of(
    store: SqliteDocStore,
) -> None:
    """Requirement 5: only the stored expansion is trimmed, never the passage.

    The entry for ``SORT`` is three words and the page it cites states the whole line. Asserted
    against a passage that has been through storage and retrieval rather than against the fixture,
    because the fixture containing its own text proves nothing about what a reader is shown.
    """
    chunks = await _corpus(store)
    entries = await system.entries_by_acronym(store, chunks)
    _, wired = await _retrievers(store, chunks)

    result = await wired.retrieve(_ask("What is SORT?"))

    assert result.expansion is not None
    match = result.expansion.matches[0]
    assert match.entry.expansion == "SecOps Reliability Toolkit"
    promoted = next(
        passage
        for passage in result.context.passages
        if passage.chunk.id == entries["SORT"].chunk_id
    )
    assert "SORT — SecOps Reliability Toolkit" in promoted.chunk.text


def test_detection_needs_no_embedding_backend_and_repeats_exactly() -> None:
    """Requirement 9, which is two claims and only one of them is about determinism.

    **No model.** Asserted in a subprocess, because by the time this module is imported the suite
    has loaded a good deal and an in-process check would measure the wrong thing. Importing the
    detector loads no MLX, no ONNX runtime, no torch and no tokenizer, so "deterministic across
    embedding backends" is true of it in the strong sense: it cannot observe which backend is
    installed. ``tests/test_import_boundary.py`` makes the same claim about ``manicule.ingest`` as
    a package; this makes it about the module that grew a new comparison form, where a helper
    reaching for a normalizer out of an embedding package would be an easy mistake to make.

    **Same input, same entries.** Detection over the whole labeled corpus twice, compared field
    by field. The rule that could break this is the new one: :func:`initials_forms` returns a
    ``frozenset``, and a set iterated into a result would order entries by hash seed rather than
    by the document. It does not — the sets are intersected and never iterated into output — and
    this is what says so, run under the suite's random-order plugin on every seed it picks.
    """
    probe = (
        "import json, sys; import manicule.ingest.glossary; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in "
        "{'mlx', 'mlx_embeddings', 'onnxruntime', 'torch', 'transformers', "
        "'sentence_transformers', 'tokenizers', 'numpy'})))"
    )
    loaded = subprocess.run(  # noqa: S603 - the interpreter running this suite, no shell
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert json.loads(loaded.stdout) == [], "importing the detector loaded an embedding backend"

    first = [_detected(item.line) for item in labeled.LABELED]
    second = [_detected(item.line) for item in labeled.LABELED]

    assert first == second


def test_the_bound_on_the_skeleton_is_the_value_the_sweep_chose() -> None:
    """The constant, pinned to the measurement in its own docstring rather than to a preference.

    Three is not a round number chosen for comfort. Two admits ``WEB = 'when enabled'`` from this
    corpus and four loses ``AuDiT``; the sweep is in :data:`~manicule.ingest.glossary.
    MIN_SKELETON_LENGTH` and this asserts its conclusion, so a later edit to the constant fails
    here and is sent to read it.
    """
    assert MIN_SKELETON_LENGTH == 3
