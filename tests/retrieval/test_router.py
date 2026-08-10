"""The router keeps trivial input away from the corpus, and real questions away from itself."""

from __future__ import annotations

import pytest

from manicule.config.settings import RouterSettings
from manicule.core.errors import ConfigError
from manicule.retrieval.router import QueryRouter, Route, UtilityKind, normalise

ALL_KINDS = (UtilityKind.DOCUMENT_COUNT, UtilityKind.DOCUMENT_LIST, UtilityKind.INDEX_STATUS)


def a_router(**settings: object) -> QueryRouter:
    return QueryRouter(RouterSettings(**settings), available=ALL_KINDS)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("text", ["hi", "Hello!", "  thanks  ", "Good morning.", "hey"])
def test_a_whole_greeting_never_reaches_the_corpus(text: str) -> None:
    """Trivial input should not consume a generation call. That is the whole feature."""
    assert a_router().route(text).route is Route.GREETING


@pytest.mark.parametrize(
    "text",
    [
        "yo-yo manufacturing tolerances",
        "thanks for the memory dump, what does it say?",
        "hello world program in rust",
        "hey how do I rotate the signing key",
        "sup command in bash",
    ],
)
def test_a_question_that_begins_with_a_greeting_still_reaches_the_corpus(text: str) -> None:
    """The failure a prefix match produces, and the reason the match is anchored at both ends.

    ``/^(hi|hello|hey|yo)\\b/`` routes every one of these away from the index: ``-`` is a
    non-word character so the boundary matches, and the rest of the sentence is never read.
    Each is an ordinary query against a technical corpus, and each would get an answer that
    never touched a document. A missed greeting costs one retrieval; a false greeting costs a
    wrong answer to a real question.
    """
    assert a_router().route(text).route is Route.RETRIEVE


def test_a_long_input_is_never_a_greeting() -> None:
    """A greeting is short. A sentence that begins with one is a question."""
    padded = "hello" + " " * 60
    assert a_router(max_chars=8).route(padded).route is Route.RETRIEVE


def test_utility_questions_route_to_their_handler() -> None:
    """Each kind names a handler that exists; the phrase is what selects it."""
    router = a_router()
    assert router.route("how many documents?").utility is UtilityKind.DOCUMENT_COUNT
    assert router.route("List Documents").utility is UtilityKind.DOCUMENT_LIST
    assert router.route("index status").utility is UtilityKind.INDEX_STATUS


def test_a_utility_kind_with_no_handler_is_never_emitted() -> None:
    """A route nothing returns is not a route.

    Declaring routes the system cannot answer produces a type that documents a feature the
    function does not have. Here the phrase falls through to retrieval, which answers it badly
    rather than reaching a handler that is not there.
    """
    router = QueryRouter(RouterSettings(), available=[UtilityKind.DOCUMENT_LIST])
    assert router.route("how many documents").route is Route.RETRIEVE
    assert router.route("list documents").utility is UtilityKind.DOCUMENT_LIST


def test_a_handler_the_router_cannot_reach_is_refused() -> None:
    """The other direction: a handler no phrase selects is a feature nothing can invoke."""

    class Unreachable:
        value = "unreachable"

    with pytest.raises(ConfigError, match="no route reaches"):
        QueryRouter(RouterSettings(), available=[Unreachable()])  # pyright: ignore[reportArgumentType]


def test_a_disabled_router_routes_everything_to_the_corpus() -> None:
    assert a_router(enabled=False).route("hello").route is Route.RETRIEVE


def test_the_router_reads_nothing_but_the_text() -> None:
    """Determinism is the property that makes this cheap enough to run before the cache."""
    router = a_router()
    first = router.route("hello")
    second = router.route("hello")
    assert first == second


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello!  ", "hello"),
        ("HOW MANY DOCUMENTS?", "how many documents"),
        ("yo-yo", "yo-yo"),
        ("...", ""),
    ],
)
def test_normalisation_tolerates_punctuation_without_splitting_words(
    raw: str, expected: str
) -> None:
    """Interior punctuation survives, which is what keeps ``yo-yo`` from becoming ``yo``."""
    assert normalise(raw) == expected


def test_a_direct_route_declares_that_it_bypassed_retrieval() -> None:
    """The one path where an answer legitimately has no sources has to say so."""
    router = a_router()
    assert router.route("hi").bypasses_retrieval
    assert router.route("how many documents").bypasses_retrieval
    assert not router.route("how do I rotate the signing key").bypasses_retrieval
