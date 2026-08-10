"""The citation marker, and the scanner that finds one in a character stream.

``[[cite:3]]``, and ``[[cite:3,5]]`` for several. The syntax is the whole reason citation
verification is affordable: **the model selects a slot rather than writing a citation**, so a
marker carries one small integer and every field of the resulting
:class:`~manicule.generation.answers.Citation` is built from the context passage that integer
names.

**Why not ``[1]``.** It was the obvious choice and it is unusable. It occurs constantly in
ordinary technical prose, in bibliographies a corpus may contain, and — fatally — in code,
where ``argv[1]`` and ``items[0]`` are everywhere and the corpus is full of code blocks by
design. A binder scanning for ``[1]`` eats the answer. ``[[3]]`` collides with wiki-link
syntax, and ``[^3]`` is a Markdown footnote reference a quoted passage can legitimately
contain. ``[[cite:N]]`` has no plausible collision in prose or code, is cheap to detect on a
character stream, and — the reason for the ``cite:`` prefix rather than a bare ``[[3]]`` — a
malformed attempt is still *recognisable as an attempt*, so it can be counted rather than
mistaken for prose.

**The scanner is not Markdown-aware, deliberately.** Being Markdown-aware means parsing a
partially-received document, which is guesswork, and acting on a guess means the answer a
reader gets is a rewrite nobody reviewed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

MARKER_OPEN = "[["
MARKER_CLOSE = "]]"
MARKER_KEYWORD = "cite"
ATTEMPT_PREFIX = f"{MARKER_OPEN}{MARKER_KEYWORD}"
"""What makes something *recognisable as an attempt* even when it is malformed.

The scanner and :func:`escape_markers` are both defined against this one constant, so a
passage cannot contain something the escaper leaves alone and the scanner then binds.
"""

MARKER_MAX_LEN = 64
"""Characters after the opening ``[[`` within which a marker must close.

Without this bound one unterminated ``[[cite:`` stalls the stream forever, because the
scanner would hold every subsequent character waiting for a ``]]`` that never comes. Past it
the buffered text is released **verbatim** — the binder deletes markers, and something that
is not a marker is not the binder's to delete.
"""

_SLOT_LIST = re.compile(r"^\s*:\s*(\d+)(?:\s*,\s*(\d+))*\s*$")
"""The payload of a well-formed marker: a colon and one or more slot numbers.

Whitespace is tolerated on both sides of the colon and around the separators, which is the
one normalisation the binder performs — of syntax it defined itself. ``[[cite: 3]]`` and
``[[cite:3 ]]`` mean what they obviously mean, and rendering them as ``[[cite:3]]`` is not an
edit to the answer in any sense a reader could observe.
"""

_SLOT = re.compile(r"\d+")

_ESCAPE = re.compile(re.escape(ATTEMPT_PREFIX), re.IGNORECASE)
"""What :func:`escape_markers` neutralises in a passage body."""


def escape_markers(text: str) -> str:
    """Neutralise marker syntax inside text that will be shown to a model.

    Not hypothetical: manicule's own documentation describes this syntax, and manicule's own
    documentation is exactly the sort of thing somebody indexes. A passage containing a
    literal ``[[cite:3]]`` that the model then quotes would otherwise bind — to a real
    passage, so it would even *verify* — producing a citation nobody asked for.

    The escape separates the two opening brackets, which is the minimum that stops the
    scanner recognising an attempt while leaving the text legible to the model. Like
    redaction, it is applied to **a copy on its way into a prompt**: the stored passage, the
    chunk text a citation quotes, and the bytes verification runs against are all untouched.
    """
    return _ESCAPE.sub(lambda match: f"{match.group(0)[0]} {match.group(0)[1:]}", text)


class ScanEventKind(StrEnum):
    """What the scanner found."""

    TEXT = "text"
    """Ordinary characters, to be passed through unchanged."""

    MARKER = "marker"
    """A well-formed marker naming at least one slot."""

    MALFORMED = "malformed"
    """A closed attempt whose payload was not a slot list.

    Deleted and counted. It is not prose — nothing writes ``[[cite:...]]`` by accident — so
    releasing it into the answer would show a reader syntax that means nothing to them.
    """

    UNTERMINATED = "unterminated"
    """An attempt that never closed inside :data:`MARKER_MAX_LEN`.

    Released verbatim, and counted separately from a malformed one: the scanner has no
    evidence about where such a thing was meant to end, so deleting to a guessed boundary
    would be exactly the sentence-level surgery this design refuses.
    """


@dataclass(frozen=True, slots=True)
class ScanEvent:
    """One thing the scanner found, in stream order."""

    kind: ScanEventKind
    text: str
    """The characters this event covers. For :attr:`ScanEventKind.TEXT` and
    :attr:`ScanEventKind.UNTERMINATED` this is what must reach the reader verbatim; for the
    other two it is the raw marker, which does not."""

    slots: tuple[int, ...] = ()
    """Slots named, deduplicated and in the order the model wrote them. Empty unless
    :attr:`kind` is :attr:`ScanEventKind.MARKER`."""


def parse_slots(payload: str) -> tuple[int, ...] | None:
    """Slots named by the text between ``[[cite`` and ``]]``, or ``None`` if malformed.

    Duplicates are collapsed while order is kept: ``[[cite:3,3]]`` is one citation to slot 3,
    and emitting it twice would double-count the accounting for a model that stuttered.
    """
    if _SLOT_LIST.match(payload) is None:
        return None
    seen: dict[int, None] = {}
    for found in _SLOT.finditer(payload):
        seen.setdefault(int(found.group(0)), None)
    return tuple(seen) or None


_KEEP_GOING = ScanEvent(ScanEventKind.TEXT, "")
"""A decided step with nothing to emit. See :meth:`MarkerScanner._drain`."""


@dataclass(slots=True)
class MarkerScanner:
    """Finds markers in text arriving a piece at a time.

    Feed it whatever the provider hands over — a marker split across three network chunks is
    the normal case, not an edge case — and it yields events in stream order. It holds back
    only what could still turn out to be a marker: one character when the text ends mid-``[``,
    and at most :data:`MARKER_MAX_LEN` while an attempt is open.

    :meth:`finish` must be called, or a trailing partial marker is silently lost. That is the
    difference between an answer ending in ``[[cite`` and an answer ending nowhere.
    """

    _pending: str = field(default="", init=False)
    _in_attempt: bool = field(default=False, init=False)

    def feed(self, chunk: str) -> Iterator[ScanEvent]:
        """Consume ``chunk`` and yield everything now decidable."""
        self._pending += chunk
        yield from self._drain(final=False)

    def finish(self) -> Iterator[ScanEvent]:
        """Flush whatever is still held.

        An attempt still open at the end of a stream never closed, so it is released verbatim
        under the same rule as one that overran the bound — the provider stopped, which says
        nothing about what the model meant.
        """
        yield from self._drain(final=True)

    def _drain(self, *, final: bool) -> Iterator[ScanEvent]:
        """Yield every decidable event. ``final`` forbids holding anything back.

        A step returning ``None`` means "not decidable yet"; an event with empty text means
        "decided, nothing to emit, keep going". The two are different and collapsing them
        stalls the stream on the buffer that opens an attempt.
        """
        while self._pending:
            step = (
                self._close_attempt(final=final)
                if self._in_attempt
                else self._seek_attempt(final=final)
            )
            if step is None:
                return
            if step.text:
                yield step

    def _seek_attempt(self, *, final: bool) -> ScanEvent | None:
        """Emit text up to the next ``[[``, or all of it when there is none."""
        opening = self._pending.find(MARKER_OPEN)
        if opening < 0:
            # A trailing "[" could still become "[[", so it waits for the next chunk — unless
            # there will not be one.
            keep = 1 if not final and self._pending.endswith("[") else 0
            return self._take(len(self._pending) - keep)
        if opening > 0:
            return self._take(opening)
        decided = self._is_attempt(final=final)
        if decided is None:
            return None
        if decided:
            self._in_attempt = True
            return _KEEP_GOING
        # A wiki link, an array of arrays, anything else that opens with two brackets. Release
        # the brackets and resume scanning after them, which is what guarantees progress.
        return self._take(len(MARKER_OPEN))

    def _take(self, size: int) -> ScanEvent | None:
        """Emit the first ``size`` characters of the buffer as text."""
        if size <= 0:
            return None
        text, self._pending = self._pending[:size], self._pending[size:]
        return ScanEvent(ScanEventKind.TEXT, text)

    def _is_attempt(self, *, final: bool) -> bool | None:
        """Whether the buffer opens an attempt. ``None`` means not yet decidable."""
        available = self._pending[: len(ATTEMPT_PREFIX)].lower()
        if len(available) == len(ATTEMPT_PREFIX):
            return available == ATTEMPT_PREFIX
        if not ATTEMPT_PREFIX.startswith(available):
            return False
        # Still consistent with an attempt, so it is undecided — unless nothing more is coming.
        return False if final else None

    def _close_attempt(self, *, final: bool) -> ScanEvent | None:
        """Resolve an open attempt once its closing brackets arrive, or its bound expires."""
        end = self._pending.find(MARKER_CLOSE, len(ATTEMPT_PREFIX))
        if end < 0:
            overrun = len(self._pending) - len(MARKER_OPEN) > MARKER_MAX_LEN
            if not (overrun or final):
                return None
            raw, self._pending, self._in_attempt = self._pending, "", False
            return ScanEvent(ScanEventKind.UNTERMINATED, raw)
        raw = self._pending[: end + len(MARKER_CLOSE)]
        if len(raw) - len(MARKER_OPEN) > MARKER_MAX_LEN:
            # It closed, but not within the bound. Released verbatim for the same reason: a
            # marker this long is not one, and the scanner is not entitled to delete prose.
            self._pending, self._in_attempt = self._pending[len(raw) :], False
            return ScanEvent(ScanEventKind.UNTERMINATED, raw)
        self._pending, self._in_attempt = self._pending[len(raw) :], False
        slots = parse_slots(raw[len(ATTEMPT_PREFIX) : -len(MARKER_CLOSE)])
        if slots is None:
            return ScanEvent(ScanEventKind.MALFORMED, raw)
        return ScanEvent(ScanEventKind.MARKER, raw, slots)


def render_marker(slots: tuple[int, ...]) -> str:
    """The canonical spelling of a marker naming ``slots``.

    What survives into the stored answer, so a stored answer still says where its citations
    were and ``messages.sources`` can hold the citation records positionally.
    """
    return f"{ATTEMPT_PREFIX}:{','.join(str(slot) for slot in slots)}{MARKER_CLOSE}"


__all__ = [
    "ATTEMPT_PREFIX",
    "MARKER_CLOSE",
    "MARKER_KEYWORD",
    "MARKER_MAX_LEN",
    "MARKER_OPEN",
    "MarkerScanner",
    "ScanEvent",
    "ScanEventKind",
    "escape_markers",
    "parse_slots",
    "render_marker",
]
