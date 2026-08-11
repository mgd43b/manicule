"""The citation binder: marker → slot → verified :class:`Citation`.

**This is the boundary, and it sits above :class:`~manicule.core.protocols.Generator` and
outside any protocol.** That placement is the same argument the retrieval design makes for
putting the hydrating join *inside* the dense stage, applied in the opposite direction and
for the identical reason: a boundary a plugin can omit is not a boundary, and a third-party
generator is exactly the thing that would implement ``generate`` and forget to verify
anything. An installed generator plugin can change which model answers, how it is reached,
what it costs and how fast it streams. It cannot change which citations survive.

**The only edit anything downstream of the model may make to the answer is deleting a
marker.** Once you accept that some citation must sometimes be removed, every convenient
repair becomes available — trimming a sentence, rewriting a clause, re-generating the tail —
and each of them changes what the user was told for a reason the user cannot see. So a
failed citation is dropped and the answer is not: its marker goes, the sentence stands
exactly as the model wrote it, and the drop is reported in band and persisted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from manicule.core.retrieval import Candidate
from manicule.generation.answers import (
    AnswerEvent,
    Citation,
    CitationAccounting,
    CitationDrop,
    DropReason,
)
from manicule.generation.markers import MarkerScanner, ScanEvent, ScanEventKind, render_marker
from manicule.generation.verification import SlotVerdict, VerificationRun


@dataclass(slots=True)
class CitationBinder:
    """Binds markers in a streamed answer to verified citations.

    Feed it the model's text as it arrives; it yields
    :class:`~manicule.generation.answers.AnswerEvent` values in stream order, so a consumer
    rendering deltas sees each citation at exactly the position its marker occupied.

    Three transformations and no others, all of them deletions or the normalisation of syntax
    the binder defined itself:

    - a marker whose slots all fail verification is **deleted**;
    - a marker naming ``3,5`` where only 3 verifies becomes ``[[cite:3]]``;
    - ``[[cite: 3]]`` and ``[[cite:3 ]]`` are rendered ``[[cite:3]]``.

    Surviving markers stay in the stored answer text, so a stored answer still says *where*
    its citations were, and the citation records travel beside it carrying the slot each one
    answers to.
    """

    run: VerificationRun

    @property
    def passages(self) -> Sequence[Candidate]:
        """The passages the run is verifying.

        Read through the run rather than held separately. Two copies would let
        :meth:`VerificationRun.verdict` range-check against one list while
        :meth:`_citation_for` indexes another — which yields either an ``IndexError`` or,
        worse, a citation built from the wrong passage.
        """
        return self.run.passages

    _scanner: MarkerScanner = field(default_factory=MarkerScanner, init=False)
    _text: list[str] = field(default_factory=list[str], init=False)
    _citations: dict[int, Citation] = field(default_factory=dict[int, Citation], init=False)
    _drops: list[CitationDrop] = field(default_factory=list[CitationDrop], init=False)
    _dropped_slots: set[int] = field(default_factory=set[int], init=False)
    _markers_seen: int = field(default=0, init=False)
    _unterminated: int = field(default=0, init=False)

    @property
    def text(self) -> str:
        """The answer as it will be stored and shown, markers included."""
        return "".join(self._text)

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Verified citations, in slot order.

        Deduplicated by slot: a model citing the same passage in five sentences produced one
        citation, and reporting five would inflate every count built on this.
        """
        return tuple(self._citations[slot] for slot in sorted(self._citations))

    @property
    def drops(self) -> tuple[CitationDrop, ...]:
        return tuple(self._drops)

    @property
    def accounting(self) -> CitationAccounting:
        return CitationAccounting(
            slots_offered=len(self.passages),
            markers_seen=self._markers_seen,
            verified=len(self._citations),
            dropped=len(self._drops),
            unterminated_markers=self._unterminated,
        )

    @property
    def ungrounded(self) -> bool:
        """Whether every citation this answer offered was dropped.

        Requires that at least one marker was seen. **Zero citations offered is recorded, not
        judged**: an answer with no markers may be the correct answer — "the sources do not
        cover this" — and no mechanism can distinguish that from a model that forgot.
        """
        return bool(self.passages) and self._markers_seen > 0 and not self._citations

    async def feed(self, chunk: str) -> list[AnswerEvent]:
        """Consume a piece of model output and return what it produced, in stream order.

        A list rather than an async generator, deliberately. This is consumed from inside
        another async generator, and an abandoned generator's ``finally`` does not run until
        something finalises it — so a nested one is a resource this layer cannot close on a
        client disconnect. Returning a list makes the nesting disappear: the only generators
        on the answer path are the two that genuinely have to suspend.
        """
        produced: list[AnswerEvent] = []
        for event in self._scanner.feed(chunk):
            produced.extend(await self._bind(event))
        return produced

    async def finish(self) -> list[AnswerEvent]:
        """Flush the scanner. Must run, or a trailing partial marker is lost."""
        produced: list[AnswerEvent] = []
        for event in self._scanner.finish():
            produced.extend(await self._bind(event))
        return produced

    async def _bind(self, event: ScanEvent) -> list[AnswerEvent]:
        if event.kind is ScanEventKind.TEXT:
            return [self._emit(event.text)]
        if event.kind is ScanEventKind.UNTERMINATED:
            # Not a marker, so not the binder's to delete. Released exactly as the model
            # wrote it: the scanner has no evidence about where it was meant to end, and
            # deleting to a guessed boundary is the sentence-level surgery this design refuses.
            self._unterminated += 1
            return [self._emit(event.text)]

        self._markers_seen += 1
        if event.kind is ScanEventKind.MALFORMED:
            drop = CitationDrop(
                reason=DropReason.MALFORMED_MARKER,
                detail=f"{event.text!r} is not a slot reference",
            )
            self._drops.append(drop)
            return [AnswerEvent.dropped(drop)]

        produced: list[AnswerEvent] = []
        survivors: list[int] = []
        for slot in event.slots:
            verdict = await self.run.verdict(slot)
            citation = self._citation_for(slot, verdict)
            if citation is None:
                produced.extend(self._record_drop(slot, verdict))
                continue
            survivors.append(slot)
            if slot not in self._citations:
                self._citations[slot] = citation
                produced.append(AnswerEvent.cited(citation))
        if survivors:
            produced.append(self._emit(render_marker(tuple(survivors))))
        return produced

    def _emit(self, text: str) -> AnswerEvent:
        self._text.append(text)
        return AnswerEvent.delta(text)

    def _record_drop(self, slot: int, verdict: SlotVerdict) -> list[AnswerEvent]:
        """Record a slot's failure once, however many times the model cited it.

        The verdict is a property of the slot and does not change during one answer, so a
        model that cites a bad slot in every sentence produced one failed citation and not
        six. The markers are still deleted every time.
        """
        if slot in self._dropped_slots:
            return []
        self._dropped_slots.add(slot)
        drop = verdict.drop or CitationDrop(
            slot=slot,
            reason=DropReason.UNRESOLVABLE,
            reached=verdict.reached,
            detail="the passage could not be turned into a citation",
        )
        self._drops.append(drop)
        return [AnswerEvent.dropped(drop)]

    def _citation_for(self, slot: int, verdict: SlotVerdict) -> Citation | None:
        """Build the citation from the **context**, or return ``None`` if it failed.

        Not one field comes from the model. The model contributed ``slot``.
        """
        if not verdict.survives or verdict.document is None:
            return None
        chunk = self.passages[slot - 1].chunk
        document = verdict.document
        return Citation(
            slot=slot,
            document_id=chunk.document_id,
            uri=document.uri,
            title=document.title,
            heading_path=chunk.heading_path,
            anchor=chunk.anchor,
            chunk_id=chunk.id,
            kind=chunk.kind,
            quote=chunk.text,
            verification=verdict.reached,
        )


__all__ = ["CitationBinder"]
