"""The check the whole package rests on: a system at chance cannot be reported as anything else.

This module is the reason the rest exists. An evaluation harness whose components have no
semantic content produces well-formed reports about nothing, and every feature justified by
one of those reports is justified by noise. So the first four tests here are, in order:

1. A pipeline whose only retrieval mechanism is an embedder with no semantic content is
   **refused** — measured against what guessing would do, not against a threshold.
2. The same pipeline with a semantically meaningful embedder is **admitted**. Without this,
   a probe that refused everything would pass test 1 while being exactly as useless.
3. A probe too small to have detected a perfect system refuses to run, rather than reporting
   "at chance" for something it could never have cleared.
4. A corpus small enough that guessing scores well refuses too.

Everything is run through the shipped retriever and the shipped dense stage, against a
migrated database. A guard demonstrated on a stub is a guard demonstrated on a stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.evaluation.corpus import CorpusVersion, corpus_version_of
from manicule.evaluation.errors import (
    AtChanceError,
    ProbeUnusableError,
    UnderpoweredProbeError,
)
from manicule.evaluation.probe import DiscriminationProbe, ProbeItem, probe_from_titles
from manicule.evaluation.systems import RetrieverSystem
from tests.evaluation.fakes import (
    BagOfWordsEmbedder,
    FixedSystem,
    LookupSystem,
    MeaninglessEmbedder,
    an_item,
)
from tests.evaluation.pipeline import SCOPE, build_corpus, dense_only_retriever

if TYPE_CHECKING:
    from manicule.evaluation.probe import ProbeOutcome
    from manicule.evaluation.systems import SystemUnderComparison
    from manicule.storage.docstore import SqliteDocStore

PROBE_K = 3
"""How deep the probe looks. Three of sixty documents is a 5% chance rate, which twenty-four
items can distinguish from a working system by a wide margin."""


async def _system(
    store: SqliteDocStore, embedder: BagOfWordsEmbedder | MeaninglessEmbedder, label: str
) -> RetrieverSystem:
    chunks = await build_corpus(store)
    retriever = await dense_only_retriever(store, embedder, chunks)
    version = await corpus_version_of(
        store,
        label="fixture",
        workspace_ids=SCOPE,
        embed_fingerprint=embedder.fingerprint.canonical(),
    )
    return RetrieverSystem(
        retriever,
        config_label=label,
        corpus_version=version,
        workspace_ids=SCOPE,
    )


async def _probe(store: SqliteDocStore) -> DiscriminationProbe:
    items = await probe_from_titles(store, workspace_ids=SCOPE, limit=24)
    return DiscriminationProbe(items, k=PROBE_K)


def _explain(outcome: ProbeOutcome) -> str:
    return f"probe said: {outcome.describe()}"


async def test_an_embedder_with_no_semantic_content_is_refused(store: SqliteDocStore) -> None:
    """The load-bearing check. A meaningless embedder must be impossible to report as working.

    Its vectors are a hash of the whole string: correctly shaped, correctly normalised,
    deterministic, and unrelated to meaning. Nothing about it raises, nothing about it looks
    wrong, and a pipeline built on it returns confident, plausible rankings. The only thing
    that can catch it is a measurement against chance, which is what this asserts.
    """
    system = await _system(store, MeaninglessEmbedder(), "meaningless")
    probe = await _probe(store)

    outcome = await probe.run(system)

    assert not outcome.discriminates, _explain(outcome)
    assert outcome.p_value > outcome.alpha, _explain(outcome)
    # Not merely "failed a threshold": it landed where guessing lands. Four times the chance
    # rate is a generous ceiling — the expected hit rate is exactly ``chance_rate`` — and it is
    # here so that a system doing *something* would show up rather than being lumped in.
    assert outcome.hit_rate <= 4 * outcome.chance_rate, _explain(outcome)

    with pytest.raises(AtChanceError, match="AT CHANCE"):
        await probe.certify(system)


async def test_an_embedder_with_semantic_content_is_admitted(store: SqliteDocStore) -> None:
    """The other half, and it is not a formality.

    A probe that refused every system would pass the test above while being exactly as
    uninformative as one that refused none. This is what makes the refusal mean something.
    """
    system = await _system(store, BagOfWordsEmbedder(), "bag-of-words")
    probe = await _probe(store)

    outcome = await probe.certify(system)

    assert outcome.discriminates, _explain(outcome)
    assert outcome.hit_rate > outcome.chance_rate, _explain(outcome)
    assert outcome.p_value <= outcome.alpha, _explain(outcome)


async def test_the_two_verdicts_come_from_the_same_probe_and_the_same_corpus(
    store: SqliteDocStore,
) -> None:
    """One thing differs between the admitted system and the refused one: the embedder.

    Without this, the two tests above could each be passing for some reason of their own — a
    different corpus, a different probe, a different ``k`` — and the pair would prove nothing
    about embedders at all.
    """
    chunks = await build_corpus(store)
    version = await corpus_version_of(store, label="fixture", workspace_ids=SCOPE)
    probe = await _probe(store)

    systems = {
        "bag-of-words": BagOfWordsEmbedder(),
        "meaningless": MeaninglessEmbedder(),
    }
    outcomes = {
        label: await probe.run(
            RetrieverSystem(
                await dense_only_retriever(store, embedder, chunks),
                config_label=label,
                corpus_version=version,
                workspace_ids=SCOPE,
            )
        )
        for label, embedder in systems.items()
    }

    assert outcomes["bag-of-words"].trials == outcomes["meaningless"].trials
    assert outcomes["bag-of-words"].pool_size == outcomes["meaningless"].pool_size
    assert outcomes["bag-of-words"].chance_rate == outcomes["meaningless"].chance_rate
    assert outcomes["bag-of-words"].discriminates
    assert not outcomes["meaningless"].discriminates


async def test_a_probe_too_small_to_detect_a_perfect_system_refuses_to_run() -> None:
    """A verdict that was never in doubt is not a verdict.

    Three items against twenty documents at ``k = 5`` is a chance rate of 25%, and three
    perfect answers out of three is ``0.25 ** 3 = 0.016`` — short of ``alpha``. So a flawless
    system would be reported as being at chance, which is the mirror image of the failure this
    package exists to prevent and just as uninformative. It refuses instead, and names how many
    items it would need.
    """
    system = FixedSystem(
        [an_item("d1")],
        corpus_version=CorpusVersion(label="tiny", document_count=20),
    )
    probe = DiscriminationProbe(
        [ProbeItem(text=f"q{i}", document_ids=frozenset({"d1"})) for i in range(3)], k=5
    )

    with pytest.raises(UnderpoweredProbeError, match="at least"):
        await probe.run(system)


async def test_a_corpus_small_enough_that_guessing_wins_refuses_to_run() -> None:
    """Five results out of four documents is not retrieval, and every system would pass."""
    system = FixedSystem(
        [an_item("d1")],
        corpus_version=CorpusVersion(label="tiny", document_count=4),
    )
    probe = DiscriminationProbe(
        [ProbeItem(text=f"q{i}", document_ids=frozenset({"d1"})) for i in range(40)], k=5
    )

    with pytest.raises(ProbeUnusableError, match="guessing"):
        await probe.run(system)


async def test_a_system_that_cannot_say_how_large_its_corpus_is_gets_no_verdict() -> None:
    """Chance is ``k / N``. Without ``N`` there is no null hypothesis and any p-value is made up."""
    system = FixedSystem([an_item("d1")], corpus_version=CorpusVersion(label="unknown"))
    probe = DiscriminationProbe(
        [ProbeItem(text=f"q{i}", document_ids=frozenset({"d1"})) for i in range(40)], k=3
    )

    with pytest.raises(ProbeUnusableError, match="document_count"):
        await probe.run(system)


async def test_a_system_that_always_returns_the_same_list_is_refused() -> None:
    """It returns documents, so it looks like it retrieves. It answers no question."""
    items = [an_item(f"d{i}") for i in range(3)]
    system = FixedSystem(items, corpus_version=CorpusVersion(label="c", document_count=400))
    probe = DiscriminationProbe(
        [ProbeItem(text=f"q{i}", document_ids=frozenset({f"target-{i}"})) for i in range(40)],
        k=3,
    )

    with pytest.raises(AtChanceError):
        await probe.certify(system)


async def test_a_probe_matches_on_text_when_the_two_systems_share_no_identifiers() -> None:
    """The cross-system path: an external system's document ids are not manicule's.

    Text containment is the only identifier two independently built systems share, so the probe
    accepts a distinctive span as the known answer. Without it, every external system would
    score zero and be refused for a reason that has nothing to do with retrieval.
    """
    system: SystemUnderComparison = LookupSystem(
        {
            f"question {i}": [an_item(f"their-id-{i}", text=f"the answer is marker-{i} exactly")]
            for i in range(40)
        },
        corpus_version=CorpusVersion(label="c", document_count=400),
    )
    probe = DiscriminationProbe(
        [ProbeItem(text=f"question {i}", contains=f"marker-{i}") for i in range(40)], k=3
    )

    outcome = await probe.certify(system)

    assert outcome.hits == outcome.trials, _explain(outcome)


async def test_a_probe_item_with_no_known_answer_is_refused() -> None:
    """It would score as a miss for every system, dragging a working one towards chance."""
    with pytest.raises(ValueError, match="declares no correct answer"):
        ProbeItem(text="what is the gateway port")


async def test_titles_shared_by_two_documents_are_not_used_as_probes(
    store: SqliteDocStore,
) -> None:
    """A question with two right answers scores a correct system as wrong half the time."""
    from tests.storage_helpers import make_document  # noqa: PLC0415 - only this test needs it

    await build_corpus(store)
    duplicate = make_document(
        source="fixture",
        source_id="aurora-ledger-configuration-copy",
        title="aurora ledger configuration",
        uri="file:///copy.md",
        body=b"copy",
    )
    await store.upsert_document(duplicate)

    items = await probe_from_titles(store, workspace_ids=SCOPE, limit=100)

    assert "aurora ledger configuration" not in {item.text for item in items}
    assert items, "the rest of the corpus should still yield probe items"
