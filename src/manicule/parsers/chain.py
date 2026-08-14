"""Choosing a parser, running the fallback chain, and naming what came out.

Chains rot when "failure" is fuzzy, so this module names four outcomes and says which of them
advance:

**Hard failure — advances.** The parser raised something other than a decline. Recorded per
attempt.

**Declined — advances, tracked separately.** The parser inspected the input and reported that
it is not its kind: the plaintext parser handed a JPEG, the archive parser handed an OOXML
container. That is *information*. If every parser in the chain declined, the document is
``unsupported_media_type``; if any raised, it is ``failed`` at stage ``parse``.

**Empty output — advances, and is remembered.** Zero text-bearing blocks without raising.
Advancing is the entire purpose of putting a layout model behind a fast text extractor. If
*every* parser comes back empty, the document is ``no_extractable_text``, not ``failed``.

**Degraded output — does not advance.** A parser that produced text but only ``Unlocated``
anchors, or only page-level ones, **has succeeded**. Falling back on quality grounds makes
the chain non-deterministic, makes results depend on thresholds nobody tuned, and doubles
parse cost on exactly the documents that are already slow. If a parser's quality is
unacceptable, reorder the chain rather than making the runtime guess.

The pipeline that calls this owns per-attempt time and memory limits and the retry policy
(``docs/ingest.md`` §5). What lives here is which parser runs, what counts as failure, and
what status results.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import JsonValue

from manicule.core.content import (
    BlockKind,
    DocumentStatus,
    Metadata,
    ParsedBlock,
    PipelineStage,
    RawDocument,
)
from manicule.core.errors import ConfigError, ParseError
from manicule.core.protocols import Parser, read_blocks

WILDCARD = "*"
"""The key whose chain is appended to every other chain.

Shipping ``["plaintext"]`` here means an unknown text-ish file is indexed with real line
anchors rather than skipped. It only works because the plaintext parser refuses non-text
bytes — without that refusal a shipped tail would index every unrecognized binary as
mojibake, and ``unsupported_media_type`` would be unreachable because some parser would
always claim every document.
"""

DEFAULT_CHAINS: Mapping[str, tuple[str, ...]] = {
    "application/pdf": ("pdf",),
    "text/html": ("html",),
    WILDCARD: ("plaintext",),
}
"""Chains that ship. Keyed by media type, first entry is the primary.

There is deliberately no separate "primary parser" concept, which removes a whole class of
question about how the two interact.
"""


class Outcome(StrEnum):
    """What one attempt produced."""

    PARSED = "parsed"
    EMPTY = "empty"
    DECLINED = "declined"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One parser's turn, recorded so a result is explicable months later."""

    parser: str
    outcome: Outcome
    reason: str = ""

    def as_metadata(self) -> list[JsonValue]:
        return [self.parser, self.outcome.value, self.reason]


@dataclass(frozen=True, slots=True)
class ChainResult:
    """What the chain produced, and the status that follows from it."""

    blocks: list[ParsedBlock]
    status: DocumentStatus
    status_detail: str
    failed_stage: PipelineStage | None = None
    parser_used: str | None = None
    attempts: tuple[Attempt, ...] = ()

    @property
    def metadata(self) -> Metadata:
        """What the parse stage records on the document.

        Falling back is **not** a status. It is metadata, so ``status`` stays coarse enough
        to filter on while the detail remains available to diagnostics.
        """
        attempted: list[JsonValue] = [attempt.as_metadata() for attempt in self.attempts]
        recorded: Metadata = {"parsers_attempted": attempted}
        if self.parser_used is not None:
            recorded["parser_used"] = self.parser_used
        if self.status_detail:
            recorded["reason"] = self.status_detail
        return recorded


@dataclass(frozen=True, slots=True)
class ParserChain:
    """Resolves a media type to an ordered list of parsers and runs them.

    Args:
        parsers: Every installed parser by registered name.
        chains: Configured chains by media type. User configuration **replaces** a media
            type's chain rather than merging into it — merging produces chains nobody can
            predict from reading the config.
    """

    parsers: Mapping[str, Parser]
    chains: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_CHAINS))

    def __post_init__(self) -> None:
        missing = sorted(
            {name for chain in self.chains.values() for name in chain if name not in self.parsers}
        )
        if missing:
            available = ", ".join(sorted(self.parsers)) or "none installed"
            msg = (
                f"parserFallbacks names {', '.join(missing)}, which no installed plugin "
                f"provides. Available: {available}. A chain whose behavior depends on what "
                f"happens to be installed indexes the same document differently on different "
                f"machines, so this is a startup error rather than a silent skip."
            )
            raise ConfigError(msg)

    def resolve(self, media_type: str) -> tuple[str, ...]:
        """The chain for a media type: its own entries, then the global tail.

        The resolved chain is deterministic and is recorded on the document, so a result can
        be explained long after the run that produced it.
        """
        base = self.chains.get(_bare(media_type), ())
        tail = tuple(name for name in self.chains.get(WILDCARD, ()) if name not in base)
        return (*base, *tail)

    async def run(self, raw: RawDocument) -> ChainResult:
        """Run the chain to completion, in this process, and classify the result."""
        return await run_chain(self.resolve(raw.media_type), raw, self.attempt)

    async def attempt(self, name: str, raw: RawDocument) -> tuple[list[ParsedBlock], Attempt]:
        """Give one parser its turn, in this process, and say what came of it.

        The unit the ingest pipeline dispatches. Per *attempt* rather than per document,
        because ``docs/parsing.md`` §6.3 makes the time and memory limits per-parser: a chain
        of three parsers on a 30-second limit may legitimately take 90 seconds, and a
        per-document limit would make the last parser in a chain fail for reasons belonging
        to the first.
        """
        parser = self.parsers[name]
        try:
            blocks = await read_blocks(parser, raw)
        except ParseError as exc:
            return [], Attempt(parser=name, outcome=Outcome.DECLINED, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - a parser's own bug must not end the batch
            reason = f"{type(exc).__name__}: {exc}"
            return [], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason)
        if not bears_text(blocks):
            return [], Attempt(parser=name, outcome=Outcome.EMPTY, reason="no text-bearing blocks")
        return blocks, Attempt(parser=name, outcome=Outcome.PARSED)


async def run_chain(
    chain: Sequence[str],
    raw: RawDocument,
    attempt: Callable[[str, RawDocument], Awaitable[tuple[list[ParsedBlock], Attempt]]],
) -> ChainResult:
    """Run ``chain`` to completion over ``raw``, and classify what came out.

    The loop is separated from the thing that runs one parser because the ingest pipeline
    runs each attempt in a **worker subprocess**: a deadline is only enforceable across a
    process boundary, since a parser inside a native extension observes no cancellation and
    Python cannot kill a thread. Both callers share this loop, so an attempt that was killed
    and an attempt that raised advance the chain identically — and, crucially, are classified
    identically.
    """
    if not chain:
        return ChainResult(
            blocks=[],
            status=DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
            status_detail=(
                f"no parser chain matched {raw.media_type!r}. Configure one under "
                f"[parserFallbacks], or add a '*' entry to index unknown text files."
            ),
        )

    attempts: list[Attempt] = []
    for name in chain:
        blocks, outcome = await attempt(name, raw)
        attempts.append(outcome)
        if outcome.outcome is Outcome.PARSED:
            return ChainResult(
                blocks=blocks,
                status=DocumentStatus.PARSED,
                status_detail="",
                parser_used=name,
                attempts=tuple(attempts),
            )
    return classify(raw, tuple(attempts))


def classify(raw: RawDocument, attempts: Sequence[Attempt]) -> ChainResult:
    """Turn a chain that produced no text into the one status that describes it.

    **The mixed case is the one an implementation guesses at.** A chain where one parser hard
    failed and another returned empty, with no parser producing text, is ``failed`` — *not*
    ``no_extractable_text``. A parser that broke leaves us genuinely not knowing whether text
    was there, and ``no_extractable_text`` means something specific: the tooling worked and
    there was nothing to find. Widening it to cover "something broke and the rest found
    nothing" would make the scanned-corpus warning fire on library bugs and stop meaning what
    it means.
    """
    outcomes = {attempt.outcome for attempt in attempts}
    tried = ", ".join(f"{attempt.parser} ({attempt.outcome.value})" for attempt in attempts)

    if Outcome.FAILED in outcomes:
        broke = [attempt for attempt in attempts if attempt.outcome is Outcome.FAILED]
        detail = "; ".join(f"{attempt.parser}: {attempt.reason}" for attempt in broke)
        return ChainResult(
            blocks=[],
            status=DocumentStatus.FAILED,
            status_detail=f"every parser failed or found nothing. {detail}",
            failed_stage=PipelineStage.PARSE,
            attempts=tuple(attempts),
        )

    if outcomes == {Outcome.DECLINED}:
        return ChainResult(
            blocks=[],
            status=DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
            status_detail=(
                f"every parser for {raw.media_type!r} declined this document: {tried}. "
                f"Each inspected it and reported that it is not their kind."
            ),
            attempts=tuple(attempts),
        )

    return ChainResult(
        blocks=[],
        status=DocumentStatus.NO_EXTRACTABLE_TEXT,
        status_detail=(
            f"the parser chain ran to completion and found no text ({tried}). The usual "
            f"cause is a scanned or image-only document; optical character recognition is "
            f"out of scope, so this is reported rather than indexed as empty."
        ),
        attempts=tuple(attempts),
    )


def container_result(members: int) -> ChainResult:
    """The result for an archive whose members became documents of their own.

    Distinct from ``no_extractable_text``: nothing failed and nothing was missing. Conflating
    the two would put every archive into the bucket that triggers the scanned-corpus warning,
    which is how a diagnostic stops meaning anything.
    """
    return ChainResult(
        blocks=[],
        status=DocumentStatus.CONTAINER,
        status_detail=f"expanded into {members} member document(s)",
        parser_used="archive",
        attempts=(Attempt(parser="archive", outcome=Outcome.PARSED),),
    )


def bears_text(blocks: Sequence[ParsedBlock]) -> bool:
    """Whether any block carries a character worth indexing.

    Whitespace-only, form-feed-only and empty-string blocks all count as nothing, which is
    what makes "the chain found no text" a statement about content rather than about how many
    objects a parser happened to yield.
    """
    return any(block.kind is not BlockKind.MEDIA and block.text.strip() for block in blocks)


def _bare(media_type: str) -> str:
    """A media type without its parameters, e.g. ``text/html; charset=utf-8``.

    Parameters are kept out of chain keys because a chain that matched only when the charset
    happened to be spelled the same way would look configured and silently not apply.
    """
    return media_type.split(";", maxsplit=1)[0].strip().lower()


__all__ = [
    "DEFAULT_CHAINS",
    "WILDCARD",
    "Attempt",
    "ChainResult",
    "Outcome",
    "ParserChain",
    "bears_text",
    "classify",
    "container_result",
    "run_chain",
]
