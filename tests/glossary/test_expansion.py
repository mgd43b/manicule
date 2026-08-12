"""When an alias fires, when it refuses to, and what happens when two definitions disagree.

``bugs/bug2.md`` §4 is the hard half of this feature, and it is the half that decides whether
turning it on makes retrieval worse. Every rule in it has a test here that fails when the rule
is removed, and the removals were run.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from manicule.core.glossary import DefinitionForm, GlossaryEntry, MatchReason
from manicule.core.retrieval import Query
from manicule.retrieval.expansion import (
    ExpansionPolicy,
    candidate_surfaces,
    definitional_frame,
    expanded_text,
    resolve_expansion,
)
from manicule.retrieval.homographs import COMMON_ENGLISH_WORDS, is_common_word
from tests.glossary.system import query_filter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.retrieval import Filter

EXPANSION = "Network Operations Workspace"
OTHER = "Nightly Operations Watchdog"


def entry(
    acronym: str = "NOW",
    expansion: str = EXPANSION,
    *,
    display: str | None = None,
    document: str = "doc-1",
    chunk: str = "chunk-1",
    confidence: float = 0.95,
    aliases: tuple[str, ...] = (),
) -> GlossaryEntry:
    return GlossaryEntry(
        acronym=acronym,
        display=display or acronym,
        expansion=expansion,
        document_id=document,
        chunk_id=chunk,
        location="Glossary",
        form=DefinitionForm.EM_DASH,
        confidence=confidence,
        aliases=aliases,
    )


class Vocabulary:
    """A glossary source that answers from a list and records what it was asked.

    Correctly scoped by construction — it holds one workspace's entries and applies no filter —
    which is what makes it the *control* for the leaky one in ``test_storage.py``. A refusal
    seen against that one is a refusal the code produced, rather than one any source would have.
    """

    def __init__(self, *entries: GlossaryEntry) -> None:
        self._entries = entries
        self.asked: list[list[str]] = []

    async def entries_for(
        self,
        keys: Sequence[str],
        filter: Filter,  # noqa: A002 - mirrors the protocol it stands in for
    ) -> Sequence[GlossaryEntry]:
        del filter
        self.asked.append(list(keys))
        wanted = set(keys)
        return [item for item in self._entries if wanted & set(item.keys)]


async def expand(text: str, *entries: GlossaryEntry, policy: ExpansionPolicy | None = None):  # noqa: ANN201
    source = Vocabulary(*entries)
    query = Query(text=text, filter=query_filter())
    return await resolve_expansion(query, source, policy or ExpansionPolicy()), source


# --- when it fires ----------------------------------------------------------------------------


async def test_an_exact_case_match_fires() -> None:
    """``NOW`` written the way the glossary writes it needs no other evidence."""
    result, _ = await expand("What is NOW?", entry())

    assert result.fired
    assert [match.reason for match in result.matches] == [MatchReason.EXACT_CASE]
    assert result.expanded == f"What is {EXPANSION}?"
    assert result.original == "What is NOW?", "the original is retained, never replaced"


@pytest.mark.parametrize(
    "text",
    [
        "what is now?",
        "What is Now?",
        "what does now stand for",
        "define now",
        "the meaning of now",
        "now",
        "What is N.O.W.?",
    ],
)
async def test_a_definitional_question_fires_whatever_the_case(text: str) -> None:
    """A question is not a use, and the difference is visible without a capital letter.

    ``What is N.O.W.?`` is in this list because it found a real defect: every frame ended in
    ``\\b``, which cannot match after a full stop, so the punctuation variant the spec requires
    silently never fired.
    """
    result, _ = await expand(text, entry())

    assert result.fired, f"{text!r} did not fire"
    assert result.matches[0].reason in {
        MatchReason.EXACT_CASE,
        MatchReason.DEFINITIONAL_FRAME,
    }


async def test_a_term_that_is_not_an_english_word_fires_without_any_framing() -> None:
    result, _ = await expand(
        "show me the isotope dashboards", entry("ISOTOPE", "Index Storage Optimisation Endpoint")
    )

    assert result.fired
    assert result.matches[0].reason is MatchReason.UNAMBIGUOUS


async def test_an_alias_resolves_to_the_same_entry() -> None:
    result, _ = await expand("What is NETOPS?", entry(aliases=("NETOPS",)))

    assert result.fired
    assert result.matches[0].key == "NETOPS"
    assert result.matches[0].entry.expansion == EXPANSION


# --- when it refuses --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "should I restart the daemon now or wait for the window",
        "the index is warm now",
        "we now retain the original bytes",
        "by now every worker has reloaded",
    ],
)
async def test_an_ordinary_use_of_a_common_word_does_not_fire(text: str) -> None:
    """``bugs/bug2.md`` §4: do not expand every occurrence of a common English word.

    None of these asks about anything. Expanding them would run a second search for a term
    nobody named and put a glossary page in front of an operational question.
    """
    result, _ = await expand(text, entry())

    assert not result.fired
    assert result.matches == ()
    assert result.expanded == ""


async def test_a_common_word_still_needs_evidence_even_next_to_a_question_word() -> None:
    """The frame has to match *the token*, not merely be in the same sentence."""
    result, _ = await expand("what is the retention window now?", entry())

    assert not result.fired


async def test_a_word_the_operator_adds_to_the_homograph_list_stops_firing() -> None:
    """The escape hatch for a corpus whose terms collide with words the shipped list misses."""
    unguarded, _ = await expand("show me the isotope dashboards", entry("ISOTOPE", "Index Store"))
    guarded, _ = await expand(
        "show me the isotope dashboards",
        entry("ISOTOPE", "Index Store"),
        policy=ExpansionPolicy(homographs=frozenset({"ISOTOPE"})),
    )

    assert unguarded.fired
    assert not guarded.fired, "the list extends the shipped one rather than replacing it"
    assert is_common_word("NOW"), "and the shipped one is still in force"


async def test_a_nonexistent_acronym_produces_nothing_at_all() -> None:
    result, source = await expand("What is ZZQX?", entry())

    assert not result.fired
    assert result.conflicts == ()
    assert source.asked, "the store was still consulted — the query did name a candidate token"


async def test_an_entry_below_the_confidence_floor_is_not_acted_on() -> None:
    """The remedy available to an operator who cannot re-ingest a corpus."""
    weak = entry(confidence=0.61)
    permissive, _ = await expand("What is NOW?", weak)
    strict, _ = await expand("What is NOW?", weak, policy=ExpansionPolicy(min_entry_confidence=0.9))

    assert permissive.fired
    assert not strict.fired


async def test_disabling_expansion_consults_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Off means no lookup, not a lookup whose answer is discarded.

    Asserted on the *source*, not on the result. A disabled feature that still queried a store
    would still be slow and could still fail, and a test that only checked ``fired`` is False
    would pass for either implementation.
    """
    del caplog
    source = Vocabulary(entry())
    query = Query(text="What is NOW?", filter=query_filter())

    result = await resolve_expansion(query, source, ExpansionPolicy(enabled=False))

    assert not result.fired
    assert source.asked == [], "a disabled feature must not reach the store at all"


async def test_no_glossary_source_at_all_is_not_an_error() -> None:
    query = Query(text="What is NOW?", filter=query_filter())

    result = await resolve_expansion(query, None, ExpansionPolicy())

    assert not result.fired
    assert result.original == "What is NOW?"


# --- conflicts ---------------------------------------------------------------------------------


async def test_two_definitions_in_scope_are_reported_and_neither_is_chosen() -> None:
    """``bugs/bug2.md`` §4: do not choose silently between conflicting definitions.

    The strongest temptation in this whole feature is to take the more confident entry. It is
    defensible, it is silent, and it produces an answer that is fluent, cited and about the
    wrong thing.
    """
    result, _ = await expand(
        "What is NOW?",
        entry(document="doc-a", chunk="chunk-a", confidence=0.95),
        entry(expansion=OTHER, document="doc-b", chunk="chunk-b", confidence=0.60),
    )

    assert not result.fired, "a conflicting term expands to nothing"
    assert result.expanded == ""
    assert [conflict.key for conflict in result.conflicts] == ["NOW"]
    assert set(result.conflicts[0].expansions) == {EXPANSION, OTHER}


async def test_every_conflicting_definition_keeps_its_own_provenance() -> None:
    """A conflict a reader cannot go and look at is a warning they cannot act on."""
    result, _ = await expand(
        "What is NOW?",
        entry(document="doc-a", chunk="chunk-a"),
        entry(expansion=OTHER, document="doc-b", chunk="chunk-b"),
    )

    sources = {(item.document_id, item.chunk_id) for item in result.conflicts[0].entries}
    assert sources == {("doc-a", "chunk-a"), ("doc-b", "chunk-b")}


async def test_the_same_definition_stated_twice_is_not_a_conflict() -> None:
    """Two documents agreeing is corroboration, not disagreement."""
    result, _ = await expand(
        "What is NOW?",
        entry(document="doc-a", chunk="chunk-a"),
        entry(expansion="network   operations workspace.", document="doc-b", chunk="chunk-b"),
    )

    assert result.fired
    assert result.conflicts == ()


async def test_a_conflict_is_reported_even_when_the_occurrence_would_not_have_fired() -> None:
    """The one case where a conflict would otherwise be invisible.

    An ordinary lower-case use does not expand — but the disagreement in the corpus is a fact
    about the corpus, and hiding it behind the case rule would mean only the queries that
    already worked ever surfaced it.
    """
    result, _ = await expand(
        "restart the daemon now",
        entry(document="doc-a", chunk="chunk-a"),
        entry(expansion=OTHER, document="doc-b", chunk="chunk-b"),
    )

    assert not result.fired
    assert [conflict.key for conflict in result.conflicts] == ["NOW"]


# --- bounds --------------------------------------------------------------------------------------


async def test_only_so_many_terms_expand_in_one_query() -> None:
    result, _ = await expand(
        "compare ISOTOPE, KRYPTON, LANTERN and MISTRAL",
        entry("ISOTOPE", "Index Storage Optimisation Tooling Endpoint"),
        entry("KRYPTON", "Key Rotation Yield Protocol Token Notary", chunk="chunk-2"),
        entry("LANTERN", "Ledger And Node Telemetry Export Runner Node", chunk="chunk-3"),
        entry("MISTRAL", "Metrics Ingest Stream Transfer And Load", chunk="chunk-4"),
        policy=ExpansionPolicy(max_terms=2),
    )

    assert len(result.matches) == 2


async def test_one_term_written_twice_resolves_once() -> None:
    result, _ = await expand("What is NOW, and who owns NOW?", entry())

    assert len(result.matches) == 1


# --- the pieces, checked on their own ---------------------------------------------------------


def test_a_frame_matches_the_token_rather_than_the_sentence() -> None:
    assert definitional_frame("what is now?", "now")
    assert not definitional_frame("what is the window now?", "now")
    assert not definitional_frame("what is nowhere near ready", "now")


def test_surfaces_are_deduplicated_by_key_keeping_the_first_spelling() -> None:
    """Every plausible token is offered to the store, and each key only once.

    ``and`` is in the list and that is correct: deciding what is a term is the glossary's job,
    not a guess made before consulting it. What must not happen is ``NOW`` and ``now`` being
    two lookups, because they would then be two matches for one occurrence.
    """
    surfaces = candidate_surfaces("NOW and now")

    assert surfaces[0] == ("NOW", "NOW")
    assert [key for _, key in surfaces].count("NOW") == 1


async def test_expansion_substitutes_the_token_and_leaves_the_sentence() -> None:
    """Substitution rather than concatenation, which was measured rather than assumed."""
    result, _ = await expand("Who owns NOW and when did it ship?", entry())

    assert result.expanded == f"Who owns {EXPANSION} and when did it ship?"
    assert expanded_text("What is NOW?", ()) == "What is NOW?", "nothing fired, nothing changes"


def test_the_homograph_list_holds_the_words_the_rule_is_about() -> None:
    """A list that had lost ``now`` would make every rule above it untested and still green."""
    assert "NOW" in COMMON_ENGLISH_WORDS
    assert "ISOTOPE" not in COMMON_ENGLISH_WORDS
    assert all(word.isupper() for word in COMMON_ENGLISH_WORDS)


def test_a_policy_is_a_value_and_can_be_varied_one_field_at_a_time() -> None:
    base = ExpansionPolicy()

    assert replace(base, enabled=False).enabled is False
    assert replace(base, enabled=False).max_terms == base.max_terms
