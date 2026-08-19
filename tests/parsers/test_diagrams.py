"""Diagram readings: what reaches the embedder, and what must never reach a citation.

The defect under test is quiet by construction. A diagram embeds today, produces a vector today,
and answers no query about what it draws — nothing raises, and the only symptom is a search that
does not find an architecture diagram by the relationship it states. So every test here asserts
about the *reading*, and the middleware tests assert about what was left alone.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from manicule.config.settings import Settings
from manicule.container.container import Container
from manicule.core.content import Chunk
from manicule.parsers import grammars
from manicule.parsers.config import (
    DIAGRAM_LANGUAGES,
    DIAGRAM_MIDDLEWARE_NAME,
    DiagramConfig,
)
from manicule.parsers.diagrams import DiagramMiddleware, notations, reading
from manicule.parsers.plugin import PLUGIN
from manicule.plugins import ComponentRegistry
from manicule.testing import assert_middleware_contract
from tests.fakes import make_chunks, make_document


@pytest.fixture(autouse=True)
def pack_on_its_default_cache() -> Iterator[None]:
    """The pack keeps one registry per process, so restore it around every test here.

    A test elsewhere that pointed it at a temporary directory would otherwise decide what this
    suite can see.
    """
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)
    yield
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)


def require_grammars() -> None:
    """Skip precisely, naming what did not run and how to make it run.

    Called by the tests that read a diagram rather than applied to the module, so the ones that
    check a declaration or a registration still run on a machine with no grammars cached — those
    are the tests that catch a reader wired up wrongly, and a blanket skip would take them out
    exactly when something is missing.
    """
    absent = grammars.missing_grammars(sorted(DIAGRAM_LANGUAGES))
    if absent:
        pytest.skip(
            f"no grammar cached for {', '.join(absent)} in {grammars.cache_directory()}, so the "
            f"assertions for {', '.join(absent)} did not run — fetch them with "
            f"`manicule doctor --fix`"
        )


DOT = """digraph G {
  label="Auth architecture";
  rankdir=LR;
  subgraph cluster_cp { label="Control plane"; auth; store; }
  auth  [label="Auth Service", shape=box];
  store [label="Token Store"];
  auth -> store [label="validates against"];
  store -> db;
  audit [label="Audit Log"];
}
"""

FLOWCHART = """flowchart LR
  auth[Auth Service] -->|validates against| store[(Token Store)]
  store --> db
  subgraph cp [Control plane]
    auth
  end
"""

SEQUENCE = """sequenceDiagram
    participant A as Auth Service
    participant B as Token Store
    A->>B: validate token
    B-->>A: ok
"""


def read(language: str, source: str, *, budget: int = 10_000, limit: int = 64) -> str:
    """A reading, asserted to exist. The ``None`` cases have tests of their own."""
    require_grammars()
    result = reading(language, source, budget=budget, max_relations=limit)
    assert result is not None, f"{language} produced no reading"
    return result


# --- the defect ------------------------------------------------------------------------------


def test_a_dot_edge_reads_through_to_the_labels_at_its_ends() -> None:
    """The whole point. ``auth -> store`` says nothing; *Auth Service → Token Store* does.

    Without this the join between an edge and its endpoints' labels exists nowhere in what the
    embedder reads, so the one fact the diagram was drawn to state is not in the vector.
    """
    assert "Auth Service → Token Store: validates against" in read("dot", DOT)


def test_an_identifier_with_no_label_is_used_as_it_is_drawn() -> None:
    """Graphviz draws the identifier when a node declares no label, and so does the reading.

    Skipping the edge instead would silently drop half a diagram whose author never labelled
    anything, which is the common case for a quick sketch.
    """
    assert "Token Store → db" in read("dot", DOT)


def test_an_undirected_edge_is_not_reported_as_a_direction() -> None:
    """``--`` and ``->`` are different claims, and only one of them has a source and a target."""
    result = read("dot", "graph { a -- b; }")

    assert "a — b" in result
    assert "→" not in result


def test_a_cluster_reports_what_it_groups() -> None:
    """A cluster label is often the only place a system's name appears on the page."""
    assert 'group "Control plane": Auth Service, Token Store' in read("dot", DOT)


def test_the_graph_label_becomes_the_first_line() -> None:
    """A diagram's own title is content, and it is the coarsest term it can be retrieved by."""
    assert read("dot", DOT).startswith("Auth architecture")


def test_a_node_no_edge_mentions_is_still_reported() -> None:
    """A diagram of unconnected boxes still states something, and its labels are all of it.

    Reporting only edges would return no reading at all for such a diagram, leaving the raw DOT
    in the vector — the exact outcome this module exists to replace.
    """
    assert "nodes: Audit Log" in read("dot", DOT)


def test_styling_never_reaches_the_reading() -> None:
    """``shape``, ``rankdir`` and the rest change the picture, never what the picture says.

    They are also the token mass that made the raw source a poor embedding input, so carrying
    them through would defeat the change while appearing to implement it.
    """
    result = read("dot", DOT)

    assert "shape" not in result
    assert "rankdir" not in result
    assert "box" not in result


# --- labels, which are untrusted strings -------------------------------------------------------


def test_a_quoted_label_loses_its_quoting_and_its_line_breaks() -> None:
    r"""A label is one fact, and the reading is line-oriented.

    DOT writes a wrapped label as ``\n`` inside a quoted string; carried through, one
    relationship would read as two.
    """
    result = read("dot", 'digraph { a [label="Auth\\nService"]; a -> b; }')

    assert '"' not in result
    assert "Auth" in result
    assert len(result.splitlines()) == 1


def test_an_html_label_contributes_its_text_and_never_its_markup() -> None:
    """Storage-format and DOT bodies are both authored by anyone who can edit the page.

    The markup is stripped by scanning rather than by parsing, because routing untrusted markup
    through an HTML engine is what promotes inert text to a document.
    """
    result = read("dot", "digraph { a [label=<<b>Auth</b> Service>]; a -> b; }")

    assert "Auth Service" in result
    assert "<b>" not in result
    assert "b>" not in result


# --- mermaid ---------------------------------------------------------------------------------


def test_a_flowchart_link_label_joins_the_vertices_it_runs_between() -> None:
    assert "Auth Service → Token Store: validates against" in read("mermaid", FLOWCHART)


def test_a_flowchart_vertex_shape_is_punctuation_rather_than_content() -> None:
    """``[(Token Store)]`` is a cylinder holding the words *Token Store*.

    Read as "whichever child is not the identifier", so the dozen shapes nobody enumerated keep
    working — an enumeration would lose their text silently.
    """
    result = read("mermaid", FLOWCHART)

    assert "Token Store" in result
    assert "[(" not in result


def test_a_sequence_signal_is_a_relationship_between_the_names_drawn() -> None:
    """A sequence diagram is relationships too, and its actors are drawn by their aliases.

    ``A->>B`` between two one-letter actors would otherwise embed as two letters and an arrow.
    """
    result = read("mermaid", SEQUENCE)

    assert "Auth Service → Token Store: validate token" in result
    assert "Token Store → Auth Service: ok" in result


def test_a_mermaid_subgraph_reports_what_it_groups() -> None:
    assert 'group "Control plane": Auth Service' in read("mermaid", FLOWCHART)


# --- every failure keeps today's behavior ------------------------------------------------------


def test_a_notation_with_no_reader_produces_nothing() -> None:
    """PlantUML has a grammar in the pack and no reader here, on purpose (§8.4.1).

    ``None`` rather than a refusal, because the caller's response to it is to leave the chunk
    exactly as it is — which is what every notation did before this module existed.
    """
    assert reading("plantuml", "@startuml\na->b\n@enduml", budget=999, max_relations=8) is None


def test_a_source_that_does_not_parse_produces_nothing_rather_than_guesses() -> None:
    """tree-sitter is error-tolerant, so the risk is a confident reading of rubbish.

    Only named node types are read, so a truncated diagram contributes the part that parsed and
    invents nothing for the part that did not.
    """
    require_grammars()
    assert reading("dot", "digraph {", budget=999, max_relations=8) is None


def test_an_empty_source_produces_nothing() -> None:
    assert reading("dot", "   \n ", budget=999, max_relations=8) is None


# --- bounding --------------------------------------------------------------------------------


def test_a_reading_never_exceeds_the_source_it_replaces() -> None:
    """The budget that makes a chunk-budget regression impossible.

    The source already satisfied the chunk budget and prose tokenizes more densely than
    punctuation, so a reading no longer than its source cannot be larger in tokens. Anything
    that does not fit is dropped rather than truncated.
    """
    budget = 60

    result = read("dot", DOT, budget=budget)

    assert len(result) <= budget


def test_what_a_bound_dropped_is_counted_rather_than_omitted() -> None:
    """A quietly shortened reading is indistinguishable from a small diagram.

    The count is deliberately not called "relationships": groupings and the unconnected-node
    line are droppable too, and naming the wrong thing here would be a small lie in the one
    place whose job is to prevent a silent one.
    """
    result = read("dot", DOT, budget=60)

    assert "… and" in result
    assert "more" in result


def test_max_relations_bounds_a_reading_that_fits_the_character_budget() -> None:
    """The second bound, for the diagram that is many edges between short identifiers.

    Such a reading stays well inside the character budget while becoming a list nobody would
    read, and the first relationships are what the vector should be about.
    """
    source = "digraph {\n" + "\n".join(f"n{i} -> n{i + 1};" for i in range(40)) + "\n}"

    result = read("dot", source, limit=5)

    assert len(result.splitlines()) == 6, result
    assert result.splitlines()[-1].startswith("… and")


def test_a_reading_that_cannot_fit_at_all_is_no_reading() -> None:
    """Below the budget for even one fact, the chunk keeps the embedding input it has."""
    require_grammars()
    assert reading("dot", DOT, budget=3, max_relations=64) is None


# --- the declaration ---------------------------------------------------------------------------


def test_the_reader_table_and_the_declared_set_agree() -> None:
    """A reader added here and not declared in config is a reader nothing ever reaches.

    The set is what registration validates configuration against and what the middleware
    selects on; the table is what dispatches. Two spellings of one answer drift the first time
    a notation is added to one of them.
    """
    assert notations() == DIAGRAM_LANGUAGES


def test_every_declared_notation_has_a_media_type_so_its_grammar_is_seeded() -> None:
    """Declaring the language is what makes ``prefetch`` fetch the grammar and the bundle carry
    it, so a notation missing from ``MEDIA_TYPES`` reads nothing on an air-gapped install and
    everything on a developer's warm cache — one corpus, two chunkings."""
    assert set(grammars.MEDIA_TYPES) >= DIAGRAM_LANGUAGES


# --- the middleware ----------------------------------------------------------------------------


def diagram_chunk(text: str, language: str, *, breadcrumb: str = "ENG > Platform") -> Chunk:
    document = make_document()
    template = make_chunks(document, count=1)[0]
    return template.model_copy(
        update={
            "text": text,
            "embed_text": f"{breadcrumb}\n\n{text}",
            "metadata": {"lang": language},
        }
    )


def middleware() -> DiagramMiddleware:
    return DiagramMiddleware(DiagramConfig())


async def test_the_middleware_leaves_the_cited_text_alone() -> None:
    """The conformance suite, which is the check no parser test can perform.

    A citation into a diagram must keep quoting the DOT the page holds; the reading is for the
    embedder and exists nowhere a reader is shown.
    """
    require_grammars()
    document = make_document()
    chunks = [diagram_chunk(DOT, "dot")]

    returned = await assert_middleware_contract(middleware(), document, chunks)

    assert returned[0].text == DOT


async def test_the_middleware_declares_that_it_rewrites_the_embedding_input() -> None:
    """Undeclared, it would change every vector while both fingerprint refusals passed."""
    assert middleware().mutates_embedded_text is True


async def test_the_breadcrumb_survives_above_the_reading() -> None:
    """``embed_text`` is breadcrumb + text (§5), and only the text half is replaced.

    Dropping the breadcrumb would make a diagram in a section called "Configuration"
    unretrievable by what it configures — the defect §5.1 exists to prevent, arriving through
    a middleware instead of through the chunker.
    """
    require_grammars()
    chunk = diagram_chunk(DOT, "dot")

    rewritten = await middleware().after_chunk(make_document(), [chunk])

    assert rewritten[0].embed_text.startswith("ENG > Platform\n\n")
    assert "Auth Service → Token Store: validates against" in rewritten[0].embed_text
    assert "rankdir" not in rewritten[0].embed_text


async def test_a_chunk_in_a_language_with_no_reader_is_returned_untouched() -> None:
    chunk = diagram_chunk("SELECT 1;", "sql")

    rewritten = await middleware().after_chunk(make_document(), [chunk])

    assert rewritten[0].embed_text == chunk.embed_text


async def test_a_chunk_with_no_language_is_returned_untouched() -> None:
    """Prose merged with a short diagram loses the agreed language, and that is a fail-safe."""
    document = make_document()
    chunk = make_chunks(document, count=1)[0]

    rewritten = await middleware().after_chunk(document, [chunk])

    assert rewritten[0].embed_text == chunk.embed_text


async def test_a_chunk_another_middleware_already_rewrote_is_left_alone() -> None:
    """The breadcrumb is recovered as the prefix ``embed_text`` carries above ``text``.

    A chunk that no longer has that shape has no recoverable prefix, and inventing one would
    put a guessed breadcrumb into the vector. Order in the chain is configuration, so this is
    reachable by configuring a redactor first.
    """
    chunk = diagram_chunk(DOT, "dot").model_copy(update={"embed_text": "[REDACTED]"})

    rewritten = await middleware().after_chunk(make_document(), [chunk])

    assert rewritten[0].embed_text == "[REDACTED]"


async def test_a_diagram_that_yields_no_reading_keeps_its_source_as_the_vector() -> None:
    chunk = diagram_chunk("digraph {", "dot")

    rewritten = await middleware().after_chunk(make_document(), [chunk])

    assert rewritten[0].embed_text == chunk.embed_text


async def test_narrowing_the_configured_notations_turns_one_off() -> None:
    """Configuration is declarative: naming one notation reads that one and leaves the other."""
    narrowed = DiagramMiddleware(DiagramConfig(languages=frozenset({"mermaid"})))
    chunk = diagram_chunk(DOT, "dot")

    rewritten = await narrowed.after_chunk(make_document(), [chunk])

    assert rewritten[0].embed_text == chunk.embed_text


async def test_the_registered_middleware_builds_through_the_container() -> None:
    """Registration is the only route in, so a component that cannot be built is not delivered.

    Through the real plugin and the real container rather than by constructing the class: the
    registration declares a configuration model and a metadata factory, and either of them being
    wrong produces a component that exists in the source and not in the product.
    """
    registry = ComponentRegistry().bind("parsing")
    PLUGIN.register(registry)
    settings = Settings()
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"middleware": (DIAGRAM_MIDDLEWARE_NAME,)}
            )
        }
    )

    chain = await Container(configured, registry).middleware()

    assert [hook.name for hook in chain] == [DIAGRAM_MIDDLEWARE_NAME]
    assert chain[0].mutates_embedded_text, (
        "the fingerprint would not know this middleware rewrites every vector"
    )
