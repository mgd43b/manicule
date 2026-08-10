"""Defects found by review, each with the assertion that would have caught it.

Every test here failed before the fix it names, with one marked exception that exists to pin
a behaviour a fix must *not* change. They are collected rather than scattered
because what they have in common is the shape of the bug: in each case the suite was green,
the code looked right, and the guarantee was not being kept.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, override

import pytest
from litellm.exceptions import APIConnectionError
from pydantic import SecretStr

from manicule.config.profiles import PROFILES
from manicule.config.settings import RedactionMethod, RedactionSettings, Settings
from manicule.core.anchors import HeadingAnchor
from manicule.core.content import RawDocument
from manicule.core.errors import ConfigError, ProviderTimeoutError
from manicule.core.generation import FinishReason, Usage
from manicule.core.retrieval import RetrievalProfile
from manicule.generation.answering import (
    Answerer,
    AnswerRequest,
    AnswerResult,
    accepted_extras,
    answering,
)
from manicule.generation.answers import DropReason, EventKind
from manicule.generation.binder import CitationBinder
from manicule.generation.history import Turn
from manicule.generation.markers import ATTEMPT_PREFIX, MARKER_MAX_LEN, MarkerScanner, ScanEventKind
from manicule.generation.policy import EgressPolicy, filter_context
from manicule.generation.redaction import Redactor
from manicule.generation.verification import (
    CitationVerifier,
    OpenSource,
    UnverifiableSource,
    load_documents,
)
from manicule.testing.normalise import contains_claimed_text
from tests.generation.fakes import (
    FakeDocuments,
    FakeParser,
    ProtocolOnlyGenerator,
    ScriptedGenerator,
    candidate,
    context,
    document,
    query,
    resolver,
    settings,
)
from tests.generation.test_provider_and_budget import FakeStream, chunk, generator

ROLLBACK = "Roll back with `deploy --rollback`."
EMAIL = "oncall@example.invalid"


def _raw() -> RawDocument:
    """Bytes for a resolver that has to return a real :class:`OpenSource`."""
    return RawDocument(source_id="doc-1-src", uri="u", media_type="text/markdown", content=ROLLBACK)


# --- the scanner ---------------------------------------------------------------------------


def canonical(text: str, chunks: int) -> list[tuple[str, str, tuple[int, ...]]]:
    """The event stream for one splitting, with adjacent text runs merged.

    Merging is what makes two splittings comparable at all: where the scanner *cuts* a run of
    ordinary characters is an artefact of arrival, and only the markers and the resulting text
    are observable.
    """
    scanner = MarkerScanner()
    size = max(len(text) // max(chunks, 1), 1)
    events = [
        event
        for piece in (
            [text] if chunks <= 1 else [text[at : at + size] for at in range(0, len(text), size)]
        )
        for event in scanner.feed(piece)
    ]
    events.extend(scanner.finish())
    merged: list[tuple[str, str, tuple[int, ...]]] = []
    for event in events:
        plain = event.kind in {ScanEventKind.TEXT, ScanEventKind.UNTERMINATED}
        if plain and merged and merged[-1][0] == "plain":
            merged[-1] = ("plain", merged[-1][1] + event.text, ())
        elif plain:
            merged.append(("plain", event.text, ()))
        else:
            merged.append((event.kind.value, event.text, event.slots))
    return merged


@pytest.mark.parametrize(
    "sample",
    [
        f"{ATTEMPT_PREFIX}:" + "x" * 70 + f" and {ATTEMPT_PREFIX}:1]] done",
        f"prose {ATTEMPT_PREFIX}:1]] more {ATTEMPT_PREFIX}:2,3]] end",
        f"[[wiki]] {ATTEMPT_PREFIX}:1]] argv[1] items[0]",
        f"{ATTEMPT_PREFIX}" + "y" * (MARKER_MAX_LEN * 2) + "]]",
        f"a{ATTEMPT_PREFIX}:1]]b{ATTEMPT_PREFIX}:oops]]c[[not-a-cite]]d",
        "[" * 40 + f"{ATTEMPT_PREFIX}:2]]",
    ],
)
def test_which_citations_exist_does_not_depend_on_where_the_network_split_the_answer(
    sample: str,
) -> None:
    """The defect: the close search and the overrun release both scanned the whole buffer.

    One splitting produced a verified citation and deleted its marker; another produced no
    citation and left ``[[cite:1]]`` visible as prose — from the same model output. Text was
    never lost either way, which is why nothing failed.
    """
    baseline = canonical(sample, 1)

    for chunks in (2, 3, 5, 7, 11, len(sample)):
        assert canonical(sample, chunks) == baseline, f"split into {chunks} pieces diverged"


def test_a_feed_whose_result_is_discarded_still_advances_the_scanner() -> None:
    """``feed`` was a generator, so the buffer append happened on the first ``next()``.

    The test that first replaced this one *drained* the returned iterable, which is exactly
    what used to trigger the deferred append — so it passed against the broken code and
    asserted nothing. It also claimed a property the fix does not provide: a discarded return
    value still loses the events it carried. What the fix actually guarantees is that the
    scanner's **state** advances, which is what a partial marker spanning two calls depends on.
    """
    scanner = MarkerScanner()
    scanner.feed(f"prose {ATTEMPT_PREFIX}:1")  # discarded on purpose

    assert [event.kind for event in scanner.feed("]]")] == [ScanEventKind.MARKER]


@pytest.mark.parametrize("payload", [":\u0663", ":\uff13"])
def test_non_ascii_digits_are_not_silently_rewritten_as_slots(payload: str) -> None:
    """``\\d`` is Unicode-aware, so Eastern Arabic and full-width digits parsed as slots — and
    the canonical marker then replaced the model's bytes with ASCII, which is a wider edit
    than the whitespace normalisation this syntax authorises."""
    events = canonical(f"{ATTEMPT_PREFIX}{payload}]]", 1)

    assert [kind for kind, _, _ in events] == [ScanEventKind.MALFORMED.value]


# --- verification --------------------------------------------------------------------------


def test_an_empty_claim_is_not_contained_in_everything() -> None:
    """``"" in anything`` is ``True``, so a whitespace-only chunk reached the strongest
    verification level whether or not its anchor pointed anywhere."""
    assert not contains_claimed_text("some resolved text", "")
    assert not contains_claimed_text("some resolved text", "   \n\t ")
    assert contains_claimed_text("some resolved text", "resolved")


async def test_a_slots_verdict_does_not_change_during_one_answer() -> None:
    """The defect: a timeout verdict was synthesised and never recorded.

    Verification finishing after the first marker timed out gave a later marker for the same
    slot a different answer — so the reader was told the citation was dropped for a slow disk
    *and* shown it as resolved, and the accounting counted both.
    """

    class SlowButCorrect:
        """Slow enough to miss the deadline, and then **succeeds**.

        The fixture that first covered this returned ``None``, so the late verdict was also a
        drop and no citation could ever appear — the assertions could not tell the fixed code
        from the broken code, and the test passed against both. Succeeding late is the only
        shape in which an overwritten verdict is observable at all.
        """

        can_resolve = True

        async def open(self, document: object) -> Any:
            await asyncio.sleep(0.2)
            return OpenSource(FakeParser(resolutions={"Rollback": ROLLBACK}), _raw())

    verifier = CitationVerifier(SlowButCorrect(), timeout_s=0.02)
    passages = (candidate(chunk_id="c1", document_id="doc-1", text=ROLLBACK),)
    assembled = context(passages)
    run = verifier.start(assembled, {"doc-1": document()}, started_at=time.monotonic())
    binder = CitationBinder(run=run)

    await binder.feed(f"A{ATTEMPT_PREFIX}:1]]")
    await asyncio.sleep(0.3)
    await binder.feed(f" B{ATTEMPT_PREFIX}:1]]")
    await binder.finish()
    await run.aclose()

    assert binder.citations == (), (
        "the slot timed out for the first marker, so it stays timed out for the second — "
        "otherwise the reader is told the citation was dropped for a slow disk and shown it "
        "as resolved, with the accounting counting both"
    )
    assert [drop.reason for drop in binder.drops] == [DropReason.VERIFICATION_TIMEOUT]
    assert binder.accounting.verified == 0


async def test_closing_a_run_settles_every_slot_rather_than_leaving_it_waiting() -> None:
    """Cancelling a task before its first step throws in *before* the body's ``try``, so the
    ``finally`` that settles the slots never ran — and the next ``verdict`` sat out its whole
    budget with nothing in flight."""
    verifier = CitationVerifier(resolver(FakeParser()), timeout_s=30.0)
    assembled = context((candidate(chunk_id="c1", document_id="doc-1", text=ROLLBACK),))
    run = verifier.start(assembled, {"doc-1": document()})

    await run.aclose()

    began = time.monotonic()
    verdict = await run.verdict(1)

    assert time.monotonic() - began < 1.0, "a closed run answers immediately"
    assert verdict.drop is not None


async def test_a_changed_anchor_is_not_served_from_the_cache_of_the_old_one() -> None:
    """A chunk id covers document, position and text — deliberately not the anchor.

    So a re-parse that moves an anchor while leaving the text alone kept the same id under
    the same version token, and a stale ``resolved`` was replayed for an anchor nothing had
    ever checked.
    """
    parser = FakeParser(
        resolutions={"Rollback": f"## Rollback\n{ROLLBACK}", "Elsewhere": "nothing"}
    )
    verifier = CitationVerifier(resolver(parser))
    documents = {"doc-1": document()}

    good = context((candidate(chunk_id="c1", document_id="doc-1", text=ROLLBACK),))
    run = verifier.start(good, documents)
    assert (await run.verdict(1)).survives
    await run.aclose()

    moved = context(
        (
            candidate(
                chunk_id="c1",
                document_id="doc-1",
                text=ROLLBACK,
                anchor=HeadingAnchor(path=("Ops", "Elsewhere")),
            ),
        )
    )
    run = verifier.start(moved, documents)
    verdict = await run.verdict(1)
    await run.aclose()

    assert not verdict.survives, "the anchor changed, so the old verdict does not apply"


async def test_a_parser_exception_message_never_reaches_the_record() -> None:
    """A parser's message routinely quotes the input that upset it, and this detail travels
    into the trace — which promises to carry no document text."""

    class Leaky:
        can_resolve = True

        async def open(self, document: object) -> Any:
            msg = "invalid token at 'the customer is bob@example.invalid'"
            raise RuntimeError(msg)

    verifier = CitationVerifier(Leaky())
    assembled = context((candidate(chunk_id="c1", document_id="doc-1", text=ROLLBACK),))
    run = verifier.start(assembled, {"doc-1": document()})
    verdict = await run.verdict(1)
    await run.aclose()

    assert verdict.drop is not None
    assert "RuntimeError" in verdict.drop.detail
    assert "bob@example.invalid" not in verdict.drop.detail


# --- the provider -------------------------------------------------------------------------


async def test_a_cancelled_first_token_wait_still_closes_the_provider_connection() -> None:
    """``except GenerationError`` does not catch ``CancelledError``, so a client that
    disconnected during the first-token wait left an open response to a model that kept
    generating — referenced only by a dead local."""

    class Stalling(FakeStream):
        @override
        async def __anext__(self) -> Any:
            await asyncio.sleep(5)
            raise AssertionError  # pragma: no cover

    stream = Stalling([])
    gen, _ = generator(streams=[stream], settings={"first_token_timeout_s": 5, "max_retries": 0})
    tokens = gen.stream([{"role": "user", "content": "hi"}])

    async def consume() -> None:
        async for _ in tokens:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True, (
        "the connection was open and referenced only by the frame that was cancelled"
    )


async def test_a_connection_dropping_after_a_role_only_chunk_is_still_retried() -> None:
    """Every OpenAI-compatible provider opens with a role-only chunk carrying no content.

    Stopping the retry window at the first *chunk* closed it before anything had been
    delivered, so the commonest real failure — a connection dropping just after the stream
    opens — became terminal for a reader who had seen nothing.
    """
    failing = FakeStream([chunk(""), chunk("late")], fail_at=1)
    good = FakeStream([chunk("hello"), chunk(finish="stop")])
    gen, calls = generator(streams=[failing, good], settings={"max_retries": 2})

    tokens = [token async for token in gen.stream([{"role": "user", "content": "hi"}])]

    assert len(calls) == 2, "nothing was delivered, so the retry is invisible and correct"
    assert tokens[0].text == "hello"


async def test_a_first_token_timeout_names_the_setting_that_governs_it() -> None:
    """The message chose its wording by comparing remaining budgets, so a first-token expiry
    told the operator to raise the inter-token gap — a knob with no effect on that interval."""
    gen, _ = generator(
        streams=[FakeStream([])],
        settings={"first_token_timeout_s": 0.05, "stream_idle_timeout_s": 30, "timeout_s": 120},
    )

    class Stalling(FakeStream):
        @override
        async def __anext__(self) -> Any:
            await asyncio.sleep(5)
            raise AssertionError  # pragma: no cover

    gen, _ = generator(
        streams=[Stalling([])],
        settings={
            "first_token_timeout_s": 0.05,
            "stream_idle_timeout_s": 30,
            "timeout_s": 120,
            "max_retries": 0,
        },
    )

    with pytest.raises(ProviderTimeoutError, match="first token"):
        [token async for token in gen.stream([{"role": "user", "content": "hi"}])]


async def test_the_total_wall_clock_is_named_when_it_is_the_shorter_budget() -> None:
    async def never(**kwargs: Any) -> Any:
        await asyncio.sleep(5)

    gen, _ = generator(
        completion=never,
        settings={"timeout_s": 0.05, "first_token_timeout_s": 60, "max_retries": 0},
    )

    with pytest.raises(ProviderTimeoutError, match="total generation time"):
        [token async for token in gen.stream([{"role": "user", "content": "hi"}])]


async def test_retries_do_not_sleep_past_the_total_deadline() -> None:
    """Once the wall clock expired the budget clamped to zero, `_call` raised a retryable
    timeout, and the loop slept and tried again on a budget that could not improve."""

    async def refuse(**kwargs: Any) -> Any:
        raise APIConnectionError(message="refused", llm_provider="ollama_chat", model="m")

    gen, _ = generator(completion=refuse, settings={"timeout_s": 0.05, "max_retries": 4})

    began = time.monotonic()
    with pytest.raises((ProviderTimeoutError, Exception)):
        [token async for token in gen.stream([{"role": "user", "content": "hi"}])]

    assert time.monotonic() - began < 2.0, "the backoff must be clamped to the wall clock"


async def test_usage_seen_before_a_mid_stream_failure_is_not_discarded() -> None:
    """Truncation is the path where cost accounting matters most."""
    stream = FakeStream(
        [chunk("half"), chunk("", usage=Usage(prompt_tokens=90, completion_tokens=2))], fail_at=2
    )
    gen, _ = generator(streams=[stream])

    tokens = [token async for token in gen.stream([{"role": "user", "content": "hi"}])]

    assert tokens[-1].finish_reason is FinishReason.ERROR
    assert tokens[-1].usage == Usage(prompt_tokens=90, completion_tokens=2)


# --- the answer path ----------------------------------------------------------------------


def build(
    generator_: object,
    store: object = None,
    config: Settings | None = None,
    documents: object = None,
) -> Answerer:
    resolved = config or settings()
    return Answerer(
        generator=generator_,  # pyright: ignore[reportArgumentType]
        verifier=CitationVerifier(
            resolver(FakeParser(resolutions={"Rollback": f"## Rollback\n{ROLLBACK}"}))
        ),
        documents=documents or FakeDocuments({"doc-1": document()}),  # pyright: ignore[reportArgumentType]
        settings=resolved,
        policy=EgressPolicy.of(resolved),
        redactor=Redactor(resolved.security.data_policy.auto_redact),
        conversations=store,  # pyright: ignore[reportArgumentType]
    )


def a_request() -> AnswerRequest:
    passages = (
        candidate(
            chunk_id="c1", document_id="doc-1", text=ROLLBACK, heading_path=("Ops", "Rollback")
        ),
    )
    return AnswerRequest(query=query(), context=context(passages), conversation_id="conv-1")


async def test_a_failure_before_the_first_token_still_ends_with_a_final_envelope() -> None:
    """A caller written to the contract reads ``events[-1].envelope``, so a failure that
    emits nothing at all is the one shape it cannot survive."""

    class Broken:
        async def get_document(self, document_id: str) -> Any:
            msg = "the document store is down"
            raise RuntimeError(msg)

    answers = build(ScriptedGenerator(script=["fine"]), documents=Broken())

    events = [event async for event in answers.answer(a_request())]

    assert events[-1].kind is EventKind.FINAL
    assert events[-1].envelope is not None
    assert events[-1].envelope.finish_reason is FinishReason.ERROR
    assert "the document store is down" in (events[-1].envelope.error or "")


async def test_a_slow_conversation_store_does_not_turn_a_good_answer_into_an_exception() -> None:
    """A storage *failure* was swallowed and a storage *stall* was not, and the stall
    surfaced as a ``TimeoutError`` that reads exactly like a provider timeout."""
    module = sys.modules["manicule.generation.answering"]

    class Slow:
        async def append(self, message: object) -> str:
            await asyncio.sleep(1)
            return "msg-1"

    answers = build(ScriptedGenerator(script=["a complete answer"]), store=Slow())
    original = module.PERSIST_DEADLINE_S
    module.PERSIST_DEADLINE_S = 0.02  # pyright: ignore[reportAttributeAccessIssue]
    try:
        result = AnswerResult()
        events = [event async for event in answers.answer(a_request(), result)]
    finally:
        module.PERSIST_DEADLINE_S = original  # pyright: ignore[reportAttributeAccessIssue]

    assert events[-1].envelope is not None
    assert events[-1].envelope.finish_reason is FinishReason.STOP
    assert result.message_id is None, "the id is lost; the answer is not"


async def test_an_abandoned_answer_is_persisted_with_a_finish_reason_and_its_timing() -> None:
    """Stored unmarked, a partial answer reads exactly like a complete one — which is the
    distinction the whole in-band reporting rule exists to preserve."""

    class Recording:
        def __init__(self) -> None:
            self.written: list[Any] = []

        async def append(self, message: Any) -> str:
            self.written.append(message)
            return "msg-1"

    store = Recording()
    generator_ = ScriptedGenerator(script=["one ", "two ", "three "] * 10)
    answers = build(generator_, store=store)

    async def consume() -> None:
        async with answering(answers, a_request()) as events:
            async for event in events:
                if event.kind is EventKind.DELTA:
                    await asyncio.sleep(0.05)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.06)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(store.written) == 1
    assert store.written[0].finish_reason is FinishReason.ERROR
    assert store.written[0].response_time_ms is not None
    assert generator_.closed is True, "the provider stream closes with the answer, not later"


async def test_the_estimate_describes_the_prompt_that_is_actually_sent() -> None:
    """A generator with no history channel had its estimate inflated by a conversation that
    never left the machine — and that count feeds the drift comparison."""

    turns = [
        Turn(role="user", content="word " * 200),
        Turn(role="assistant", content="word " * 200),
    ]

    with_history = AnswerResult()
    answers = build(ScriptedGenerator(script=["fine"]))
    request = a_request()
    async for _ in answers.answer(
        AnswerRequest(query=request.query, context=request.context, history=turns), with_history
    ):
        pass

    without = AnswerResult()
    plugin = build(ProtocolOnlyGenerator(script=["fine"]))
    async for _ in plugin.answer(
        AnswerRequest(query=request.query, context=request.context, history=turns), without
    ):
        pass

    assert with_history.trace.estimated_prompt_tokens > without.trace.estimated_prompt_tokens
    assert without.trace.history_tokens == 0
    assert with_history.trace.history_tokens > 0


async def test_the_history_budget_is_the_one_the_window_check_approved() -> None:
    """A default of 1024 could spend twice what the ``fast`` profile reserved — the prompt
    overflow the startup refusal exists to prevent, arriving from inside."""
    answers = build(ScriptedGenerator(script=["fine"]), config=settings(rag={"profile": "fast"}))

    assert answers.history_budget == PROFILES[RetrievalProfile.FAST].history_tokens


# --- policy and redaction -------------------------------------------------------------------


def test_a_passage_whose_document_vanished_is_not_sent_to_a_remote_model() -> None:
    """The miss is realistic: the document store filters soft-deleted rows while the chunk
    index still returns their chunks. A document deleted between retrieval and generation —
    including one deleted *because* somebody decided it was sensitive — arrived here as an
    absent row and was sent anyway.
    """
    remote = settings(
        llm={"provider": "openai", "model": "gpt-4o-mini"}, providers={"openai": {"api_key": "k"}}
    )
    assembled = context((candidate(chunk_id="c1", document_id="doc-gone"),))

    filtered, drops = filter_context(assembled, {}, EgressPolicy.of(remote))

    assert filtered.passages == ()
    assert "not in the index" in drops[0].reason


def test_a_passage_whose_document_vanished_is_kept_when_nothing_leaves() -> None:
    """There is nothing to fail closed about, so proportionality wins.

    The one test in this file that passed *before* its fix as well as after: it pins the half
    of the fail-closed change that must **not** happen, so that a later tightening cannot
    quietly start refusing passages on a fully local install.
    """
    assembled = context((candidate(chunk_id="c1", document_id="doc-gone"),))

    filtered, drops = filter_context(assembled, {}, EgressPolicy.of(settings()))

    assert len(filtered.passages) == 1
    assert drops == ()


def test_dropping_a_passage_recomputes_the_context_token_count() -> None:
    """``model_copy`` does not re-validate, so a stale total over-reports the prompt."""
    remote = settings(
        llm={"provider": "openai", "model": "gpt-4o-mini"},
        providers={"openai": {"api_key": "k"}},
        security={"data_policy": {"source_restrictions": {"local_only": ["secrets"]}}},
    )
    assembled = context(
        (
            candidate(chunk_id="c1", document_id="doc-1"),
            candidate(chunk_id="c2", document_id="doc-2"),
        )
    )
    documents = {
        "doc-1": document(document_id="doc-1", source="secrets"),
        "doc-2": document(document_id="doc-2", source="public"),
    }

    filtered, _ = filter_context(assembled, documents, EgressPolicy.of(remote))

    assert filtered.token_count == filtered.passages[0].chunk.token_count


def test_a_workspace_override_applies_however_its_key_is_capitalised() -> None:
    """A raw dict lookup made a case-mismatched key an inert restriction that read as in
    force, and startup said nothing."""
    policy = EgressPolicy.of(
        settings(
            workspace="acme",
            llm={"provider": "openai", "model": "gpt-4o-mini"},
            providers={"openai": {"api_key": "k"}},
            security={"data_policy": {"workspace_overrides": {"Acme": {"cloud_allowed": False}}}},
        )
    )

    assert policy.workspace_cloud_allowed is False
    assert policy.refuses("anything")


def test_an_override_naming_no_workspace_is_refused_at_startup() -> None:
    problems = settings(
        workspace="acme",
        security={"data_policy": {"workspace_overrides": {"other-team": {"cloud_allowed": False}}}},
    ).policy_problems()

    assert any("workspace_overrides" in problem for problem in problems)


@pytest.mark.parametrize(
    "sample",
    [
        "(555) 123-4567",
        "call (555) 123-4567 today",
        "+1 (555) 123-4567",
        "555-123-4567",
        "+44 20 7123 4567",
        "+15551234567",
    ],
)
def test_the_phone_detector_still_catches_the_forms_people_actually_write(sample: str) -> None:
    """The negative samples that tightened this pattern removed positives with them.

    ``(555) 123-4567`` — the commonest written US form — stopped matching, and nothing
    noticed: the suite had exactly one positive phone sample, which happened to survive. A
    security control that quietly narrows is the failure this project keeps naming, so both
    directions are pinned now.
    """
    result = Redactor(RedactionSettings(enabled=True, patterns=("phone",))).redact(sample)

    assert result.counts == {"phone": 1}, f"{sample!r} was not detected"


@pytest.mark.parametrize(
    "sample",
    [
        "released on 2026-08-10 after review",
        "ISO 2026-08-10T12:00:00Z",
        "order 12345 shipped",
        "build 20260810 succeeded",
        "US$ 1,234,567.89 total",
        "the meeting is at 12:30:45 UTC",
        "config key a:b:c and x:1:2",
    ],
)
def test_the_detectors_leave_ordinary_numbers_and_timestamps_alone(sample: str) -> None:
    """Not one negative sample existed before. ``phone`` needed only five digits with optional
    separators, so it ate every ISO date and bit into the middle of ``1,234,567.89``, leaving
    ``1,234,[REDACTED]`` — which changes what a figure says rather than hiding it."""
    result = Redactor(
        RedactionSettings(enabled=True, patterns=("email", "phone", "credit-card", "ip-address"))
    ).redact(sample)

    assert result.text == sample
    assert result.counts == {}


def test_a_hashed_replacement_is_not_matched_again_by_a_later_detector() -> None:
    """Detectors ran in sequence over the already-substituted text, so a digest starting with
    thirteen digits was reported as a card — false telemetry, and a mangled co-reference
    token, which is the only reason ``hash`` exists."""
    redactor = Redactor(
        RedactionSettings(
            enabled=True,
            patterns=("email", "credit-card"),
            method=RedactionMethod.HASH,
            hash_salt=SecretStr("a-per-installation-secret"),
        )
    )

    for index in range(400):
        result = redactor.redact(f"user{index}@example.invalid")
        assert result.counts == {"email": 1}, result.text
        assert result.text.count("[REDACTED]") == 1


def test_a_custom_pattern_matching_the_empty_string_is_refused() -> None:
    """Every position in every passage becomes a match, so a 32k-token context becomes a
    prompt several times larger sent to a metered endpoint."""
    with pytest.raises(ConfigError, match="empty string"):
        Redactor(RedactionSettings(enabled=True, patterns=(), custom_patterns=(r"\d*",)))


# --- the second review pass ----------------------------------------------------------------


async def test_the_prompt_that_reaches_the_provider_is_the_redacted_one() -> None:
    """The defect this catches was in the *fix* for a previous one.

    Redacted titles, URIs and heading paths were computed, used for the token estimate, and
    then thrown away: the generator rebuilt its own prompt from the raw arguments and sent
    that. ``envelope.redacted`` was ``True`` and the trace counted the detector firing, so the
    record reported protection that had been discarded — which is worse than not redacting at
    all, because it is protection somebody could rely on.

    The fix hands the generator the prompt rather than the parts, which also makes the
    estimate literally the thing that was sent.
    """
    config = settings(
        llm={"provider": "openai", "model": "gpt-4o-mini"},
        providers={"openai": {"api_key": "k"}},
        security={"data_policy": {"auto_redact": {"enabled": True, "patterns": ["email"]}}},
    )
    leaky = document(document_id="doc-1", title=f"Q3 comp {EMAIL}")
    parser = FakeParser(resolutions={"Rollback": f"## Rollback\n{ROLLBACK}"})
    generator_ = ScriptedGenerator(script=["ok"])
    answers = Answerer(
        generator=generator_,
        verifier=CitationVerifier(resolver(parser)),
        documents=FakeDocuments({"doc-1": leaky}),
        settings=config,
        policy=EgressPolicy.of(config),
        redactor=Redactor(config.security.data_policy.auto_redact),
    )

    request = AnswerRequest(
        query=query(),
        context=context(
            (
                candidate(
                    chunk_id="c1",
                    document_id="doc-1",
                    text=ROLLBACK,
                    heading_path=("Ops", f"owner {EMAIL}"),
                ),
            )
        ),
    )
    async for _ in answers.answer(request):
        pass

    on_the_wire = "\n".join(message["content"] for message in generator_.seen_messages)
    assert on_the_wire, "the generator is handed the prompt, not asked to rebuild it"
    assert EMAIL not in on_the_wire
    assert "[REDACTED]" in on_the_wire
    for sent in generator_.seen_documents:
        for document_ in sent.values():
            assert EMAIL not in document_.title


async def test_a_generator_is_given_the_prompt_so_slot_numbering_is_not_its_to_decide() -> None:
    """The correspondence between "slot 3" and ``context.passages[2]`` is the basis of the
    citation guarantee, and while the prompt was built inside the pluggable component it was
    an unenforced convention. A plugin that reordered passages — a documented
    lost-in-the-middle mitigation — produced citations naming a passage the model never saw at
    that number: mechanically wrong, passing all three levels, and indistinguishable from the
    misattribution the design honestly excludes.
    """
    generator_ = ScriptedGenerator(script=["fine"])
    answers = build(generator_)

    async for _ in answers.answer(a_request()):
        pass

    assert "messages" in accepted_extras(generator_)
    assert generator_.seen_messages, "the prompt is built above the generator and handed over"
    assert "[slot 1]" in "\n".join(m["content"] for m in generator_.seen_messages)


async def test_a_document_the_store_substituted_is_not_turned_into_a_citation() -> None:
    """A store resolving through an alias, a merge or a redirect would otherwise produce a
    citation whose ``document_id`` and ``uri`` name different documents."""

    class Redirecting:
        async def get_document(self, document_id: str) -> Any:
            return document(document_id="doc-OTHER", title="Some Other Document")

    found = await load_documents(
        Redirecting(),
        context((candidate(chunk_id="c1", document_id="doc-1", text=ROLLBACK),)),
    )

    assert found == {}, "a document that is not the one asked for is not an answer"


@pytest.mark.parametrize("blank", ["", "   \n\t "])
async def test_a_passage_with_no_text_produces_no_citation_at_either_ceiling(blank: str) -> None:
    """``contains_claimed_text`` refuses an empty claim on the level-2 path, and that guard has
    to hold at every ceiling — not only the one that happens to call it. Otherwise a
    whitespace-only chunk is cited, the reader gets a blank preview, and "the text at that
    location is the text the model was given" is vacuously true.
    """
    passages = (candidate(chunk_id="c1", document_id="doc-1", text=blank),)
    documents = {"doc-1": document()}

    for source in (
        resolver(FakeParser(resolutions={"Rollback": blank})),
        UnverifiableSource("retention is off"),
    ):
        run = CitationVerifier(source).start(context(passages), documents)
        verdict = await run.verdict(1)
        await run.aclose()

        assert not verdict.survives, f"a blank passage survived under {type(source).__name__}"
