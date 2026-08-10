"""The answer path end to end: policy, prompt, provider, binder, persistence."""

from __future__ import annotations

import asyncio

from manicule.config.providers import Egress
from manicule.config.settings import RedactionScope, Settings
from manicule.core.generation import FinishReason, Usage
from manicule.core.protocols import Generator
from manicule.core.retrieval import Candidate
from manicule.generation.answering import (
    Answerer,
    AnswerRequest,
    AnswerResult,
    accepted_extras,
    answering,
)
from manicule.generation.answers import AnswerEnvelope, AnswerEvent, DropReason, EventKind
from manicule.generation.history import Turn
from manicule.generation.markers import ATTEMPT_PREFIX
from manicule.generation.policy import EgressPolicy
from manicule.generation.ports import StoredMessage
from manicule.generation.redaction import Redactor
from manicule.generation.verification import CitationVerifier
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

ROLLBACK = "Roll back with `deploy --rollback`."
EMAIL = "oncall@example.invalid"


class RecordingStore:
    """A conversation store that records what was written."""

    def __init__(self) -> None:
        self.written: list[StoredMessage] = []
        self.delay = 0.0

    async def append(self, message: StoredMessage) -> str:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.written.append(message)
        return f"msg-{len(self.written)}"


def answerer(
    generator: Generator,
    *,
    store: RecordingStore | None = None,
    config: Settings | None = None,
    parser: FakeParser | None = None,
) -> Answerer:
    resolved = config or settings()
    return Answerer(
        generator=generator,
        verifier=CitationVerifier(
            resolver(parser or FakeParser(resolutions={"Rollback": f"## Rollback\n{ROLLBACK}"}))
        ),
        documents=FakeDocuments({"doc-1": document()}),
        settings=resolved,
        policy=EgressPolicy.of(resolved),
        redactor=Redactor(resolved.security.data_policy.auto_redact),
        conversations=store,  # pyright: ignore[reportArgumentType] - a structural ConversationStore
    )


def passages() -> tuple[Candidate, ...]:
    return (
        candidate(
            chunk_id="c1", document_id="doc-1", text=ROLLBACK, heading_path=("Ops", "Rollback")
        ),
    )


async def run(
    answers: Answerer,
    *,
    conversation_id: str | None = None,
    confidence: float | None = None,
    corpus_consulted: bool = True,
) -> tuple[list[AnswerEvent], AnswerEnvelope]:
    request = AnswerRequest(
        query=query(),
        context=context(passages()),
        conversation_id=conversation_id,
        confidence=confidence,
        corpus_consulted=corpus_consulted,
    )
    events = [event async for event in answers.answer(request)]
    envelope = events[-1].envelope
    assert envelope is not None
    return events, envelope


async def test_a_complete_answer_streams_deltas_then_a_final_envelope() -> None:
    generator = ScriptedGenerator(
        script=["Roll back first.", f"{ATTEMPT_PREFIX}:1]]", " Then verify."],
        usage=Usage(prompt_tokens=1234, completion_tokens=42),
    )

    events, envelope = await run(answerer(generator))

    assert events[-1].kind is EventKind.FINAL
    assert envelope.text == f"Roll back first.{ATTEMPT_PREFIX}:1]] Then verify."
    assert len(envelope.citations) == 1
    assert envelope.finish_reason is FinishReason.STOP
    assert envelope.usage == Usage(prompt_tokens=1234, completion_tokens=42)
    assert generator.closed is True


async def test_a_provider_failure_mid_stream_is_reported_in_band_and_keeps_its_citations() -> None:
    """A truncated answer that simply ends is indistinguishable from a complete one.

    Citations already emitted stand: they were verified when they were emitted, and the
    provider failing afterwards says nothing about them.
    """
    generator = ScriptedGenerator(
        script=[f"Roll back.{ATTEMPT_PREFIX}:1]]", " and then"], fail_after=1
    )

    _, envelope = await run(answerer(generator))

    assert envelope.finish_reason is FinishReason.ERROR
    assert "stopped half-way" in (envelope.error or "")
    assert len(envelope.citations) == 1
    assert envelope.text.startswith("Roll back.")


async def test_a_partial_answer_is_persisted_so_it_can_be_shared_and_rated() -> None:
    store = RecordingStore()
    generator = ScriptedGenerator(script=["half an answer", " and the rest"], fail_after=1)

    await run(answerer(generator, store=store), conversation_id="conv-1")

    assert len(store.written) == 1
    assert store.written[0].content == "half an answer"
    assert store.written[0].finish_reason is FinishReason.ERROR


async def test_a_cancelled_answer_still_writes_what_it_had() -> None:
    """``CancelledError`` is delivered once, so an ``await`` in a ``finally`` reached by
    cancellation completes normally. Tested rather than assumed, because the folklore is
    wrong and the shield exists for a *second* cancellation, not the first."""
    store = RecordingStore()
    generator = ScriptedGenerator(script=["one ", "two ", "three "] * 20)
    answers = answerer(generator, store=store)

    async def consume() -> None:
        request = AnswerRequest(
            query=query(), context=context(passages()), conversation_id="conv-1"
        )
        async with answering(answers, request) as events:
            async for event in events:
                if event.kind is EventKind.DELTA and len(event.text) > 2:
                    await asyncio.sleep(0.05)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.06)
    task.cancel()
    with__cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        with__cancelled = True

    assert with__cancelled is True
    assert len(store.written) == 1, "a partial answer that exists nowhere cannot be rated"
    assert store.written[0].content


async def test_redaction_reaches_the_wire_and_never_the_citation() -> None:
    """The prompt says ``[REDACTED]``; the citation resolves to the original text.

    That is the feature working: redaction controls what leaves the machine, not what is
    hidden from the person who indexed the corpus and already has read access to it.
    """
    config = settings(
        llm={"provider": "openai", "model": "gpt-4o-mini"},
        providers={"openai": {"api_key": "k"}},
        security={
            "data_policy": {
                "auto_redact": {
                    "enabled": True,
                    "patterns": ["email"],
                    "scope": RedactionScope.REMOTE,
                }
            }
        },
    )
    text = f"Roll back and page {EMAIL} first."
    parser = FakeParser(resolutions={"Rollback": f"## Rollback\n{text}"})
    generator = ScriptedGenerator(script=[f"Page the on-call.{ATTEMPT_PREFIX}:1]]"])
    answers = answerer(generator, config=config, parser=parser)

    request = AnswerRequest(
        query=query(f"who is {EMAIL}?"),
        context=context(
            (
                candidate(
                    chunk_id="c1", document_id="doc-1", text=text, heading_path=("Ops", "Rollback")
                ),
            )
        ),
    )
    events = [event async for event in answers.answer(request)]
    envelope = events[-1].envelope
    assert envelope is not None

    sent = generator.seen_context[0].passages[0].chunk.text
    assert EMAIL not in sent
    assert "[REDACTED]" in sent
    assert EMAIL not in generator.seen_query[0].text
    assert envelope.citations[0].quote == text, "the citation quotes the source, unredacted"
    assert envelope.redacted is True


async def test_a_directly_routed_answer_cannot_acquire_citations_and_is_not_ungrounded() -> None:
    """With an empty context there are no slots, so every marker fails at level 0. The flag
    means "the context was non-empty and nothing survived", and this one was empty by design.
    """
    generator = ScriptedGenerator(script=[f"Hello.{ATTEMPT_PREFIX}:1]]"])
    answers = answerer(generator)

    request = AnswerRequest(
        query=query(), context=context(()), corpus_consulted=False, confidence=0.9
    )
    events = [event async for event in answers.answer(request)]
    envelope = events[-1].envelope
    assert envelope is not None

    assert envelope.citations == ()
    assert envelope.ungrounded is False
    assert envelope.corpus_consulted is False
    assert envelope.confidence is None, "not 1.0 and not 0.0 — absent"
    assert envelope.dropped[0].reason is DropReason.OUT_OF_RANGE


async def test_generation_forwards_confidence_verbatim_and_never_writes_to_it() -> None:
    """A high confidence with an ungrounded answer is a meaningful, diagnosable combination —
    the evidence was there and the model did not use it — and it is only visible because the
    two numbers stayed separate."""
    generator = ScriptedGenerator(script=[f"Unsupported.{ATTEMPT_PREFIX}:9]]"])

    _, envelope = await run(answerer(generator), confidence=0.87)

    assert envelope.confidence == 0.87
    assert envelope.ungrounded is True


async def test_history_is_sent_when_the_generator_can_take_it() -> None:
    generator = ScriptedGenerator(script=["fine"])
    answers = answerer(generator)

    request = AnswerRequest(
        query=query(),
        context=context(passages()),
        history=[Turn(role="user", content="earlier"), Turn(role="assistant", content="reply")],
    )
    async for _ in answers.answer(request):
        pass

    assert [message["content"] for message in generator.seen_history] == ["earlier", "reply"]


async def test_a_protocol_only_generator_is_recorded_as_carrying_no_history() -> None:
    """A generator that cannot take a conversation is a recorded fact rather than a
    conversation that quietly vanishes."""
    generator = ProtocolOnlyGenerator(script=["fine"])
    answers = answerer(generator)

    assert accepted_extras(generator) == frozenset()

    request = AnswerRequest(
        query=query(),
        context=context(passages()),
        history=[Turn(role="user", content="earlier"), Turn(role="assistant", content="reply")],
    )
    events = [event async for event in answers.answer(request)]

    assert events[-1].envelope is not None
    assert answers.history_supported is False, "the fact is recorded, not silently lost"


async def test_the_trace_records_which_detectors_fired_and_never_what_they_matched() -> None:
    """Recording the match would turn the trace into the leak the detector existed to
    prevent, so the record carries names and counts and nothing else."""
    config = settings(
        llm={"provider": "openai", "model": "gpt-4o-mini"},
        providers={"openai": {"api_key": "k"}},
        security={"data_policy": {"auto_redact": {"enabled": True, "patterns": ["email"]}}},
    )
    text = f"Page {EMAIL} first."
    parser = FakeParser(resolutions={"Rollback": f"## Rollback\n{text}"})
    answers = answerer(
        ScriptedGenerator(script=[f"Page them.{ATTEMPT_PREFIX}:1]]"]), config=config, parser=parser
    )

    result = AnswerResult()
    request = AnswerRequest(
        query=query(f"who is {EMAIL}?"),
        context=context(
            (
                candidate(
                    chunk_id="c1", document_id="doc-1", text=text, heading_path=("Ops", "Rollback")
                ),
            )
        ),
    )
    async for _ in answers.answer(request, result):
        pass

    assert result.trace.detectors_fired == {"email": 2}, "the query and the passage both matched"
    assert EMAIL not in result.trace.model_dump_json()
    assert result.trace.egress is Egress.REMOTE
    assert result.trace.model == "fake/model", "the trace names the generator that answered"
    assert result.trace.encoding_name == "o200k_base"
    assert result.trace.estimated_prompt_tokens > 0
    assert result.trace.citations_verified == 1
    assert result.trace.history_conditioned_retrieval is False, (
        "a follow-up retrieves on its own text alone, and recording it is what makes a bad "
        "follow-up diagnosable rather than mysterious"
    )
