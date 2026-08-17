"""The structure-aware chunker.

Boundaries come from blocks, never from prose. The chunker consumes
:class:`~manicule.core.content.ParsedBlock` and treats ``kind`` and ``heading_path`` as
facts. It does not look at the text to decide whether something is a heading, a table or
code — structure was discovered once, by the parser that could still see the markup, and
re-deriving it here both duplicates the work and does it worse: the parser had the ``<h2>``
element, this would have a line that starts with a capital letter.

Two numbers are expensive to change once a corpus is indexed, and both are in
:class:`~manicule.core.fingerprints.ChunkFingerprint` rather than in a comment:

- **512 tokens on ``embed_text``**, of which 64 are reserved for the breadcrumb. The binding
  constraint is the embedder's sequence length: past it every library in the stack truncates
  **silently**, so a 900-token chunk handed to a 512-token model produces a vector describing
  the first 512 tokens while the stored chunk still claims all 900. The chunker reads the
  effective limit from the embedder and refuses to start when its budget exceeds it.
- **64 tokens of overlap**, on prose and lists only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from manicule.chunking import breadcrumb
from manicule.chunking.sentences import paragraphs, sentences
from manicule.chunking.tokens import SupportsTokenCount, TokenCounter
from manicule.core.anchors import Anchor, CellAnchor, LineAnchor, PageAnchor, Rect
from manicule.core.content import BlockKind, Chunk, Document, Metadata, ParsedBlock
from manicule.core.errors import ChunkingError, ContextOverflowError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.ids import chunk_id
from manicule.parsers.config import STRUCTURAL_BREADCRUMB_TOKENS

CHUNKER_NAME = "structural"

CHUNKER_VERSION = "3"
"""This chunker's own version, and what moving it costs.

1 -> 2: an oversized table with no non-header row is split at its rows with no header
repeated, instead of producing nothing. Under version 1 ``_split_table`` built its parts by
iterating the rows *after* the header, so a table whose rows were all header rows returned an
empty list — the block reached no chunk, no vector and no citation, and a document that was
only that table produced zero chunks and was indistinguishable from one with no extractable
text. Every parser that emits ``rows`` could produce it: reproduced through all seven. A
table whose ``rows`` list is empty splits as prose, which is the answer this function already
gives when ``rows`` is absent and there is genuinely no boundary to use.

**What the bump costs an existing index**, since a version is only worth having if somebody
can price it:

- **Chunks that change**: only documents holding a table that is over the budget *and* has no
  non-header row. Everything else re-chunks to byte-identical chunks with byte-identical
  boundaries — this adds chunks where there were none and moves no boundary that exists.
  Measured rather than asserted: all 56 fixtures in the built corpus that these seven parsers
  read were chunked under both bodies and produced identical chunk lists. None of them holds
  the shape, which is why nothing failed while the content was being dropped.
- **Re-chunk and re-embed**: the whole corpus, including the documents whose chunks do not
  move. ``ChunkFingerprint`` records the chunker's version and not the document's content, so
  after the bump it matches for none of them, and ``check_before_run`` refuses the run until
  the operator re-chunks. Nothing can tell in advance which documents hold the shape without
  chunking them to find out, and a fingerprint that could would be a hash of the output rather
  than of the rules — the same reasoning ``PARSERS["confluence"]``'s 1 -> 2 records.
- **Why not leave it**, which is the tempting option because the shape is rare and the bill is
  corpus-wide: ``documents.chunk_fp`` is the only per-document lineage for chunking, and
  :func:`~manicule.ingest.reindex.select` finds stale documents by asking for the ones a
  *different* chunker built. Fixing forward without a bump would leave every already-ingested
  document that hit this permanently invisible to ``reindex`` — the content still missing,
  behind a fingerprint claiming to be current. That is the two-generations corpus
  ``parsers/versions.py`` exists to prevent, one stage along.

No parser ``rules`` version moves with it. What the parsers extract is unchanged: this is a
change to what the chunker does with identical blocks, which is the division
``PARSERS["html"]``'s 2 -> 3 comment draws from the other side.

2 -> 3: ``max_tokens`` became a post-render invariant. Version 2 packed content against the
breadcrumb-adjusted text budget and added overlap afterwards, so a near-full prose or list
group could exceed the fingerprint it claimed. Its code-line fallback also preserved one
oversized line whole. Version 3 repacks against each chunk's exact rendered breadcrumb,
admits only the overlap that still fits, and hard-splits every remaining oversized unit.
Those changes move boundaries, so retained-source offline rebuild is required; relabelling
the existing generation would leave its stored chunks and vectors falsely current.
"""

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
MIN_TOKENS = 64
BREADCRUMB_TOKENS = STRUCTURAL_BREADCRUMB_TOKENS

BLOCK_SEPARATOR = "\n\n"

OVERLAPPING_KINDS = frozenset({BlockKind.PROSE, BlockKind.LIST})
"""Kinds that may overlap.

Never ``code``, ``table``, ``panel``, ``heading`` or ``media``. Overlapping a table means
half a table appears twice with a repeated header and no way to tell the copies apart;
overlapping code emits a fragment whose :class:`LineAnchor` duplicates another chunk's lines,
which is indistinguishable from an anchor that is simply wrong.
"""

ATOMIC_KINDS = frozenset({BlockKind.TABLE, BlockKind.CODE})
"""Kinds that are never *partially* included in a chunk with other blocks.

A paragraph introducing a table belongs with it, so blocks of different kinds may share a
chunk. What may not happen is half a table joining a paragraph and the other half starting
the next chunk.
"""


@dataclass(frozen=True, slots=True)
class _Unit:
    """A piece of a document that is small enough to be placed whole."""

    text: str
    kind: BlockKind
    anchor: Anchor
    heading_path: tuple[str, ...]
    tokens: int
    lang: str | None = None
    """Carried from :attr:`ParsedBlock.lang`, so that a chunk can say what language it is in.

    Travels on the unit rather than in ``metadata`` because splitting and overlap both build
    new units from old ones, and a field survives ``dataclasses.replace`` while a metadata key
    survives only where somebody remembered to copy it.
    """

    metadata: Metadata = field(default_factory=Metadata)
    starts_section: bool = False
    source_ordinal: int = -1
    source_contiguous: bool = False


@dataclass(frozen=True, slots=True)
class _Overlap:
    """An overlap window: the text carried forward, and where it came from.

    The units travel with the text because the next chunk's anchor has to cover them. Text
    alone would leave the caller reconstructing provenance by counting characters, which is
    the kind of arithmetic that is right until a separator changes.
    """

    text: str = ""
    units: tuple[_Unit, ...] = ()


class StructuralChunker:
    """Groups blocks into retrievable chunks, respecting the structure the parser found.

    Args:
        counter: How tokens are counted. Comes from the bound embedder, so the budget is
            measured with the tokenizer that enforces it.
        embedder: The embedder the chunks will be sent to, when one is bound. Resolved as a
            construction dependency so :meth:`setup` can refuse a budget the model cannot
            read — the check has to happen before ingest, not after, because past the limit
            the input is dropped without an error and the chunk is indexed as its opening
            tokens while still claiming all of its text.
        grammars: Grammar version by language, for
            :attr:`~manicule.core.fingerprints.ChunkFingerprint.grammars`. Per language, so a
            Python grammar bump invalidates Python documents and leaves the rest alone.
        version_components: Other pinned versions that move chunk boundaries or anchors — the
            HTML-to-text conversion that email line numbers address, for instance. Folded
            into the fingerprint's ``version``, because a converter upgrade that shifts every
            anchor in every HTML email must not pass unnoticed.
    """

    def __init__(
        self,
        counter: TokenCounter,
        *,
        embedder: SupportsTokenCount | None = None,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
        min_tokens: int = MIN_TOKENS,
        breadcrumb_tokens: int = BREADCRUMB_TOKENS,
        grammars: Mapping[str, str] | None = None,
        version_components: Mapping[str, str] | None = None,
    ) -> None:
        if breadcrumb_tokens >= max_tokens:
            msg = (
                f"the breadcrumb reserve ({breadcrumb_tokens}) leaves no room for text in a "
                f"{max_tokens}-token budget. Lower the reserve or raise the budget."
            )
            raise ChunkingError(msg)
        self._counter = counter
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._min_tokens = min_tokens
        self._breadcrumb_tokens = breadcrumb_tokens
        self._text_budget = max_tokens - breadcrumb_tokens
        components = dict(version_components or {})
        suffix = "".join(f";{name}={value}" for name, value in sorted(components.items()))
        self.fingerprint = ChunkFingerprint(
            chunker=CHUNKER_NAME,
            version=f"{CHUNKER_VERSION}{suffix}",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_id=counter.tokenizer_id,
            grammars=dict(grammars or {}),
        )

    @property
    def provisional(self) -> bool:
        """Whether these chunks were counted without the model that will embed them.

        Provisional chunks are refused by ingest. A count taken with a stand-in vocabulary
        can undercount by an unknown margin, and undercounting is the direction that
        truncates.
        """
        return self._counter.provisional

    async def setup(self) -> None:
        """Refuse a budget the bound embedder cannot read.

        Raises:
            ContextOverflowError: The chunk budget exceeds the embedder's effective sequence
                length. Everything past that limit is dropped without an error, so the check
                belongs here — before a corpus exists — rather than at the first query that
                returns half a passage.
        """
        if self._embedder is None:
            return
        limit = self._embedder.fingerprint.max_sequence_length
        if self._max_tokens > limit:
            msg = (
                f"the chunk budget is {self._max_tokens} tokens but "
                f"{self._embedder.fingerprint.describe()} attends to {limit}. Text past the "
                f"limit is dropped with no error raised, so every oversized chunk would be "
                f"indexed as its opening tokens while still claiming all of its text. "
                f"Set plugins.config.\"chunker.structural\".max_tokens to {limit} or lower, "
                f"or choose a model with a longer sequence length."
            )
            raise ContextOverflowError(msg)

    # --- the algorithm -------------------------------------------------------------------

    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        """Produce chunks in document order."""
        block_list = [block for block in blocks if block.text]
        if not block_list:
            return []
        block_list = _promote_headings_when_that_is_all_there_is(block_list)

        units: list[_Unit] = []
        for source_ordinal, block in enumerate(block_list):
            units.extend(
                replace(unit, source_ordinal=source_ordinal) for unit in self._to_units(block)
            )
        if not units:
            return []

        groups = self._accumulate(units)
        groups = self._merge_short_tail(groups)
        groups = self._fit_final_bases(document, groups)
        return self._render(document, groups)

    def finalize(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Measure middleware's final embedding inputs and enforce this fingerprint's budget.

        ``after_chunk`` may legitimately rewrite ``embed_text``. It may not make a chunk cease
        to satisfy the chunk fingerprint that records its limit, and its inherited
        ``token_count`` is no longer true after any rewrite. The ingest stage calls this after
        the complete middleware chain, immediately before glossary detection and embedding.

        This guard cannot split middleware output: splitting ``embed_text`` alone would make
        it cease to correspond to the immutable, cited ``text``. A growing middleware must
        reserve space or emit a bounded representation instead.
        """
        measured: list[Chunk] = []
        over_budget = 0
        maximum = 0
        for chunk in chunks:
            tokens = self._counter(chunk.embed_text)
            maximum = max(maximum, tokens)
            if tokens > self._max_tokens:
                over_budget += 1
            measured.append(chunk.model_copy(update={"token_count": tokens}))
        if over_budget:
            msg = (
                f"middleware produced {over_budget} chunk(s) above the configured "
                f"{self._max_tokens}-token chunk budget; the maximum final embedding input "
                f"was {maximum} tokens. Reduce the middleware's added context or its input "
                f"before rebuilding."
            )
            raise ChunkingError(msg)
        return measured

    # --- step 1: blocks to units ---------------------------------------------------------

    def _to_units(self, block: ParsedBlock) -> list[_Unit]:
        """One unit per block, unless the block does not fit and has to be split (§4.2)."""
        tokens = self._counter(block.text)
        if block.kind is BlockKind.HEADING:
            # A heading is a boundary and a breadcrumb component, not content.
            return [
                _Unit(
                    text=block.text,
                    kind=block.kind,
                    anchor=block.anchor,
                    heading_path=block.heading_path,
                    tokens=tokens,
                    lang=block.lang,
                    metadata=dict(block.metadata),
                    starts_section=True,
                )
            ]
        if tokens <= self._text_budget:
            return [
                _Unit(
                    text=block.text,
                    kind=block.kind,
                    anchor=block.anchor,
                    heading_path=block.heading_path,
                    tokens=tokens,
                    lang=block.lang,
                    metadata=dict(block.metadata),
                )
            ]
        if block.kind is BlockKind.TABLE:
            return self._split_table(block)
        if block.kind is BlockKind.CODE:
            return self._split_lines(block)
        return self._split_prose(block)

    def _unit(self, block: ParsedBlock, text: str, **extra: object) -> _Unit:
        metadata: Metadata = dict(block.metadata)
        for key, value in extra.items():
            # Metadata is JSON-shaped; every value passed here is a str, int, bool or list.
            metadata[key] = value  # pyright: ignore[reportArgumentType] - JsonValue by construction
        return _Unit(
            text=text,
            kind=block.kind,
            anchor=block.anchor,
            heading_path=block.heading_path,
            tokens=self._counter(text),
            lang=block.lang,
            metadata=metadata,
        )

    def _split_table(self, block: ParsedBlock) -> list[_Unit]:
        """Split by rows, repeating the header into every part.

        A table part without its header is a grid of numbers; with it, each part is
        independently meaningful and independently retrievable. Header rows are known from
        the parser, never guessed from the first row being bold — a guess would silently
        promote a data row on every table that does not have one.
        """
        rows = _string_list(block.metadata.get("rows"))
        if rows is None:
            # The parser did not describe the table's rows, so there is no row boundary to
            # split at that is not a guess about the rendering. Prose splitting keeps the
            # text whole and is honest about having no better structure.
            return self._split_prose(block)
        if not rows:
            # A table whose rows the parser described as none of them. There is no boundary to
            # split at for the same reason as above, and the block still has text: it reaches
            # here only by exceeding the budget.
            return self._split_prose(block)
        header_rows = _non_negative_int(block.metadata.get("header_rows"))
        if header_rows >= len(rows):
            # Every row is a header row, so there is no data row to repeat a header *into*.
            #
            # This returned nothing at all before, because the loop below starts after the
            # header and had nowhere to start: an oversized header-only table reached no
            # chunk, no vector and no citation, and a document that was only that table
            # produced zero chunks and was indistinguishable from one with no extractable
            # text. ``docs/parsing.md`` §4.2 says there is no depth at which splitting gives
            # up, and this was one. Every parser that emits ``rows`` can produce the shape —
            # a Markdown pipe table of a header and its delimiter, a ``<thead>`` with no
            # ``<tbody>``, a one-row sheet under ``header_rows: 1``.
            #
            # Splitting by row with no header repeated, rather than falling back to prose.
            # Prose is what this function gives when ``rows`` is *absent*, and the two are not
            # the same situation: there the row boundaries are unknown, here they are known
            # and merely all header. Prose splitting would discard them and cut the table
            # mid-row and mid-cell — measured on a 60-row all-header table, three rows were
            # left in no chunk intact — which is the defect that emitting ``rows`` exists to
            # prevent. Splitting at the boundaries also keeps each part's ``CellAnchor``
            # narrowed to its own rows instead of every part claiming the whole table.
            header_rows = 0
        refs = _string_list(block.metadata.get("row_refs"))
        header = rows[:header_rows]
        header_text = "\n".join(header)
        header_tokens = self._counter(header_text) if header else 0

        parts: list[list[int]] = []
        current: list[int] = []
        running = header_tokens
        for index in range(header_rows, len(rows)):
            row_tokens = self._counter(rows[index])
            if current and running + row_tokens > self._text_budget:
                parts.append(current)
                current = []
                running = header_tokens
            current.append(index)
            running += row_tokens
        if current:
            parts.append(current)
        # ``parts`` is never empty here: ``rows`` is non-empty and ``header_rows`` is now
        # strictly less than its length, so the loop above ran at least once. Both facts are
        # established directly above, which is what lets everything below index ``parts``
        # without asking whether the table produced anything.

        units: list[_Unit] = []
        for part_index, indices in enumerate(parts, start=1):
            text = "\n".join([*header, *(rows[i] for i in indices)])
            unit = self._unit(
                block,
                text,
                table_part=[part_index, len(parts)],
                rows=[indices[0] + 1, indices[-1] + 1],
            )
            anchor = _narrow_cell_anchor(block.anchor, refs, header_rows, indices)
            units.append(replace(unit, anchor=anchor))
        if any(unit.tokens > self._text_budget for unit in units):
            # A single row does not fit. Splitting it further is a cell-level operation the
            # parser's rendering does not expose, so the row is treated as prose: it is
            # still split, still whole, and still carries the row's own anchor.
            return [
                split
                for unit in units
                for split in (
                    [unit]
                    if unit.tokens <= self._text_budget
                    else self._split_text(unit.text, unit, hard_split_kind="row")
                )
            ]
        return units

    def _split_lines(self, block: ParsedBlock) -> list[_Unit]:
        """Last-resort split for a code block the parser could not divide further.

        Code boundaries belong to the parse tree, which the parser owns; this runs only when
        a single leaf — one very long function, a generated file with no interior structure —
        still exceeds the budget. It cuts at blank-line runs and then at line ends, never
        mid-token, mid-string or mid-comment, and lines mean the same thing on every machine.
        """
        units: list[_Unit] = []
        current: list[tuple[int, str]] = []
        source_lines = block.text.splitlines(keepends=True) or [block.text]
        exact_line_mapping = (
            isinstance(block.anchor, LineAnchor)
            and block.anchor.end - block.anchor.start + 1 == len(source_lines)
        )

        def materialize(selected: Sequence[tuple[int, str]]) -> _Unit:
            text = "".join(value for _, value in selected)
            unit = replace(self._unit(block, text), source_contiguous=True)
            if exact_line_mapping and isinstance(block.anchor, LineAnchor):
                first = block.anchor.start + selected[0][0]
                last = block.anchor.start + selected[-1][0]
                unit = replace(
                    unit,
                    anchor=LineAnchor(start=first, end=last, symbol=block.anchor.symbol),
                )
            return unit

        def flush() -> None:
            if current:
                units.append(materialize(current))
                current.clear()

        # Keep line endings on their source lines. Joining with a synthetic separator would
        # lose blank lines and make the non-overlap payload impossible to reconstruct exactly.
        for offset, line in enumerate(source_lines):
            candidate = "".join(value for _, value in (*current, (offset, line)))
            if current and self._counter(candidate) > self._text_budget:
                flush()
            current.append((offset, line))
            if self._counter(line) > self._text_budget:
                oversized = materialize(current)
                current.clear()
                units.extend(self._hard_split(oversized, "line"))
        flush()
        return units or [self._unit(block, block.text)]

    def _split_prose(self, block: ParsedBlock) -> list[_Unit]:
        """Paragraph, then sentence, then — only for a single oversized sentence — tokens."""
        return self._split_text(block.text, self._unit(block, block.text), hard_split_kind="token")

    def _split_text(self, text: str, template: _Unit, *, hard_split_kind: str) -> list[_Unit]:
        units: list[_Unit] = []
        for paragraph in paragraphs(text) or [text]:
            if self._counter(paragraph) <= self._text_budget:
                units.append(replace(template, text=paragraph, tokens=self._counter(paragraph)))
                continue
            pieces = sentences(paragraph) or [paragraph]
            for packed in self._pack(pieces, " ", None, template=template):
                if packed.tokens <= self._text_budget:
                    units.append(packed)
                    continue
                units.extend(self._hard_split(packed, hard_split_kind))
        return units

    def _pack(
        self,
        pieces: Sequence[str],
        joiner: str,
        block: ParsedBlock | None,
        *,
        template: _Unit | None = None,
    ) -> list[_Unit]:
        """Greedily fill units with whole ``pieces``, never cutting one in half."""
        units: list[_Unit] = []
        current: list[str] = []
        running = 0
        for piece in pieces:
            tokens = self._counter(piece)
            if current and running + tokens > self._text_budget:
                units.append(self._materialize(joiner.join(current), block, template))
                current = []
                running = 0
            current.append(piece)
            running += tokens
        if current:
            units.append(self._materialize(joiner.join(current), block, template))
        return units

    def _materialize(self, text: str, block: ParsedBlock | None, template: _Unit | None) -> _Unit:
        if block is not None:
            return self._unit(block, text)
        if template is None:  # pragma: no cover - one of the two is always supplied
            msg = "a unit needs either a block or a template to inherit from"
            raise ChunkingError(msg)
        return replace(template, text=text, tokens=self._counter(text))

    def _hard_split(self, unit: _Unit, kind: str) -> list[_Unit]:
        """Cut a single oversized piece at a token boundary, and record that it happened.

        Only a sentence longer than the whole budget reaches here — a minified line, a base64
        blob pasted into a page. ``metadata.hard_split`` makes those countable, because a
        document full of them is a document that will retrieve badly and the operator should
        be able to find out.
        """
        pieces: list[_Unit] = []
        remaining = unit.text
        while remaining:
            cut = self._longest_prefix_within_budget(remaining)
            head, remaining = remaining[:cut], remaining[cut:]
            metadata: Metadata = {**unit.metadata, "hard_split": True, "hard_split_at": kind}
            pieces.append(replace(unit, text=head, tokens=self._counter(head), metadata=metadata))
        return pieces

    def _longest_prefix_within_budget(self, text: str) -> int:
        """How many characters of ``text`` fit the budget, by bisection on the counter.

        Bisection rather than decoding token ids back to text: the counter is the only thing
        the embedder guarantees, and a chunker that reconstructed text from token ids would
        produce different cuts under a tokenizer that round-trips imperfectly.
        """
        low, high = 1, len(text)
        best = 1
        while low <= high:
            middle = (low + high) // 2
            if self._counter(text[:middle]) <= self._text_budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _fit_final_bases(
        self, document: Document, groups: Sequence[Sequence[_Unit]]
    ) -> list[list[_Unit]]:
        """Repack groups against the exact breadcrumb-plus-current-text rendering.

        The earlier text budget is still useful: it preserves structural boundaries cheaply
        and keeps most documents on the fast path. This pass is authoritative. It accounts
        for separators and tokenizer behavior at string boundaries rather than assuming token
        counts add.
        """
        hierarchy = _source_hierarchy(document)
        fitted: list[list[_Unit]] = []
        for group in groups:
            heading_path = group[0].heading_path

            def fits_text(value: str, path: Sequence[str] = heading_path) -> bool:
                crumb = self._breadcrumb(document, hierarchy, path, content=value)
                return self._rendered_count(crumb, value) <= self._max_tokens

            current: list[_Unit] = []
            for original in group:
                for unit in self._fit_final_unit(original, fits_text):
                    candidate = [*current, unit]
                    candidate_text = _join_units(candidate)
                    if current and not fits_text(candidate_text):
                        fitted.append(current)
                        current = []
                    if not fits_text(unit.text):  # pragma: no cover - guarded below
                        raise ChunkingError("an indivisible unit cannot fit the final chunk budget")
                    current.append(unit)
            if current:
                fitted.append(current)
        return fitted

    def _fit_final_unit(
        self, unit: _Unit, fits_text: Callable[[str], bool]
    ) -> list[_Unit]:
        if fits_text(unit.text):
            return [unit]

        pieces: list[_Unit] = []
        remaining = unit.text
        while remaining:

            def fits_prefix(prefix: str) -> bool:
                return fits_text(prefix)

            cut = self._longest_prefix_satisfying(remaining, fits_prefix)
            if cut == 0:
                msg = "the rendered breadcrumb leaves no room for even one content character"
                raise ChunkingError(msg)
            head, remaining = remaining[:cut], remaining[cut:]
            metadata: Metadata = {
                **unit.metadata,
                "hard_split": True,
                "hard_split_at": "final_budget",
            }
            pieces.append(replace(unit, text=head, tokens=self._counter(head), metadata=metadata))
        return pieces

    def _rendered_count(self, crumb: str, text: str) -> int:
        embed_text = f"{crumb}{BLOCK_SEPARATOR}{text}" if crumb else text
        return self._counter(embed_text)

    @staticmethod
    def _longest_prefix_satisfying(text: str, fits: Callable[[str], bool]) -> int:
        low, high = 1, len(text)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if fits(text[:middle]):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    # --- step 2: units to groups ---------------------------------------------------------

    def _accumulate(self, units: Sequence[_Unit]) -> list[list[_Unit]]:
        """Fill chunks with consecutive units, closing at headings and at the budget."""
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        running = 0
        opens_section = True

        for original in units:
            if original.kind is BlockKind.HEADING:
                if current:
                    groups.append(current)
                current, running = [], 0
                opens_section = True
                continue
            unit = original
            if not current and opens_section:
                # Remembered on the chunk's first unit so the minimum-size merge below can
                # see a heading boundary it must not cross.
                unit = replace(unit, starts_section=True)
                opens_section = False
            closes = bool(current) and (
                running + unit.tokens > self._text_budget
                or not _mergeable(current[-1].anchor, unit.anchor)
                or (unit.kind in ATOMIC_KINDS and running + unit.tokens > self._text_budget)
            )
            if closes:
                groups.append(current)
                current, running = [], 0
            current.append(unit)
            running += unit.tokens
        if current:
            groups.append(current)
        return groups

    def _merge_short_tail(self, groups: list[list[_Unit]]) -> list[list[_Unit]]:
        """Merge a sub-minimum chunk backwards, when that is possible without loss.

        A trailing eight-token chunk is retrieval noise: a short text produces a vector
        dominated by its few tokens and wins queries it should lose. It is never *dropped* —
        dropping is data loss, and silent data loss is what this design is mostly about — so
        a chunk that cannot merge stands alone.
        """
        merged: list[list[_Unit]] = []
        for group in groups:
            tokens = sum(unit.tokens for unit in group)
            # A heading is a hard boundary. Merging across one would give the combined chunk
            # the earlier section's heading path, so the later section would be embedded and
            # cited under a heading it does not belong to.
            if not merged or tokens >= self._min_tokens or group[0].starts_section:
                merged.append(group)
                continue
            previous = merged[-1]
            same_kind = _dominant_kind(previous) is _dominant_kind(group)
            fits = sum(unit.tokens for unit in previous) + tokens <= self._text_budget
            joins = _mergeable(previous[-1].anchor, group[0].anchor)
            if same_kind and fits and joins:
                merged[-1] = [*previous, *group]
            else:
                merged.append(group)
        return merged

    # --- step 3: groups to chunks --------------------------------------------------------

    def _render(self, document: Document, groups: Sequence[Sequence[_Unit]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        previous: Sequence[_Unit] | None = None
        # Resolved once per document rather than once per chunk. It is a fact about the document,
        # and reading it validates the stored source record — so inside the loop a document of two
        # hundred chunks paid for two hundred identical validations of one JSON blob.
        hierarchy = _source_hierarchy(document)
        for position, group in enumerate(groups):
            current_text = _join_units(group)
            heading_path = group[0].heading_path
            crumb = self._breadcrumb(document, hierarchy, heading_path, content=current_text)

            def overlap_fits(
                value: str,
                current: str = current_text,
                breadcrumb_text: str = crumb,
            ) -> bool:
                text = f"{value}{BLOCK_SEPARATOR}{current}" if value else current
                return self._rendered_count(breadcrumb_text, text) <= self._max_tokens

            overlap = self._overlap_from(previous, group, fits=overlap_fits)
            text = current_text
            if overlap.text:
                text = f"{overlap.text}{BLOCK_SEPARATOR}{text}"
            embed_text = f"{crumb}{BLOCK_SEPARATOR}{text}" if crumb else text
            measured = self._counter(embed_text)
            if measured > self._max_tokens:  # pragma: no cover - direct postcondition above
                raise ChunkingError("the final rendered chunk exceeds its configured token budget")
            # The overlap window extends the anchor with it (docs/parsing.md §4.3). A chunk
            # that opens with the previous chunk's last sentences and names only its own lines
            # quotes from outside the place it cites — and it reads correctly while doing so,
            # which is why only the round-trip check finds it.
            anchor = _merge_anchors([unit.anchor for unit in (*overlap.units, *group)])
            chunks.append(
                Chunk(
                    id=chunk_id(document.id, position, text),
                    document_id=document.id,
                    text=text,
                    embed_text=embed_text,
                    anchor=anchor,
                    heading_path=heading_path,
                    kind=_dominant_kind(group),
                    position=position,
                    token_count=measured,
                    metadata=_group_metadata(group, provisional=self._counter.provisional),
                )
            )
            previous = group
        return chunks

    def _overlap_from(
        self,
        previous: Sequence[_Unit] | None,
        group: Sequence[_Unit],
        *,
        fits: Callable[[str], bool] | None = None,
    ) -> _Overlap:
        """Whole trailing sentences of the previous chunk, and the units they came from.

        Taken in whole sentences, never mid-sentence: a window that cuts mid-sentence produces
        a chunk starting on a fragment, which is what the overlap existed to avoid.

        Capped at **half** the preceding chunk. The overlap window and the minimum chunk size
        are the same number, so an uncapped window would make a minimum-sized chunk's
        successor open with an exact copy of the whole thing — two chunks, one entirely
        contained in the other, both matching the same query.

        The contributing units are returned as well as the text, because the anchor has to
        cover them (``docs/parsing.md`` §4.3). Sentences are taken per unit rather than from
        the joined text so that provenance is exact rather than reconstructed by counting
        characters — a sentence never spans two blocks, so nothing is lost by it.
        """
        if previous is None or self._overlap_tokens <= 0:
            return _Overlap()
        if _dominant_kind(previous) not in OVERLAPPING_KINDS:
            return _Overlap()
        if _dominant_kind(group) not in OVERLAPPING_KINDS:
            return _Overlap()
        if not _mergeable(previous[-1].anchor, group[0].anchor):
            # The overlap would be text the next chunk's anchor does not cover, which is a
            # citation quoting from outside the place it names.
            return _Overlap()
        source = _join_units(previous)
        cap = min(self._overlap_tokens, max(1, self._counter(source) // 2))

        taken: list[str] = []
        used: list[_Unit] = []
        for unit in reversed(previous):
            # A unit the next chunk's anchor already covers may be cut into: taking part of it
            # widens nothing, because the anchor is the same one either way. That is the case
            # whenever an oversized block was split across chunks, which is where overlap
            # matters most. Any other unit is taken whole or not at all — a partial take would
            # leave the anchor covering lines the chunk does not quote, and a line anchor is
            # meant to *be* the text it addresses.
            free = unit.anchor == group[0].anchor
            unit_sentences = list(reversed(sentences(unit.text) or [unit.text]))
            if not free:
                candidate = [*reversed(unit_sentences), *taken]
                candidate_text = " ".join(candidate)
                candidate_tokens = self._counter(candidate_text)
                if candidate_tokens > cap or (fits is not None and not fits(candidate_text)):
                    break
                taken = candidate
                used.insert(0, unit)
                continue
            stopped = False
            contributed = False
            for sentence in unit_sentences:
                candidate = [sentence, *taken]
                candidate_text = " ".join(candidate)
                candidate_tokens = self._counter(candidate_text)
                if candidate_tokens > cap or (fits is not None and not fits(candidate_text)):
                    stopped = True
                    break
                taken = candidate
                contributed = True
            if contributed:
                used.insert(0, unit)
            if stopped:
                break
        return _Overlap(" ".join(taken), tuple(used))

    def _breadcrumb(
        self,
        document: Document,
        hierarchy: Sequence[str],
        heading_path: Sequence[str],
        *,
        content: str | None = None,
    ) -> str:
        """One chunk's breadcrumb. ``hierarchy`` is the document's, resolved by the caller once."""
        parts = breadcrumb.elements(
            _string_list(document.metadata.get("breadcrumb_prefix")) or (),
            hierarchy,
            (document.title,),
            heading_path,
        )
        if content is None:
            return breadcrumb.render(parts, self._counter, self._breadcrumb_tokens)
        # The reserve is an upper bound, not a promise to spend all of it. Separators and
        # tokenizer merges at the breadcrumb/content boundary are part of the final budget.
        for budget in range(self._breadcrumb_tokens, -1, -1):
            rendered = breadcrumb.render(parts, self._counter, budget)
            if self._rendered_count(rendered, content) <= self._max_tokens:
                return rendered
        return ""  # pragma: no cover - content fitting handles the empty-breadcrumb case


def _join_units(units: Sequence[_Unit]) -> str:
    """Render units without inventing separators inside one losslessly split source block."""
    if not units:
        return ""
    rendered = units[0].text
    previous = units[0]
    for unit in units[1:]:
        contiguous = (
            previous.source_contiguous
            and unit.source_contiguous
            and previous.source_ordinal == unit.source_ordinal
        )
        rendered += ("" if contiguous else BLOCK_SEPARATOR) + unit.text
        previous = unit
    return rendered


def _source_hierarchy(document: Document) -> tuple[str, ...]:
    """Where a document sits in its source, coarsest first.

    Two spellings of one fact, in precedence order, because they arrived in that order. A
    validated :class:`~manicule.core.provenance.SourceMetadata` is preferred when the document
    carries one; ``metadata["ancestors"]`` is the older untyped convention that connectors
    without a record still fill in, and it keeps working exactly as it did.

    **The record wins where both are present, and that is the safe way round.** Its
    ``section_path`` has been through depth, length and control-character validation;
    ``ancestors`` has been through none. Preferring the unvalidated spelling would mean a
    connector that supplied both got the weaker of its own two answers embedded into every
    vector — and a breadcrumb is not something anybody reads afterwards to check.

    This is also the whole of what propagates from a document's source record into a chunk, and
    it propagates as *text the embedder reads* rather than as a copy on the chunk row. Nothing
    document-level is duplicated per chunk; ``docs/contracts.md`` §2 fixes what a chunk is, and
    a citation resolves the rest through ``document_id``.
    """
    record = document.provenance
    if record is not None and record.source is not None and record.source.section_path:
        return record.source.section_path
    return tuple(_string_list(document.metadata.get("ancestors")) or ())


# --- anchor merging ----------------------------------------------------------------------


def _mergeable(left: Anchor, right: Anchor) -> bool:
    """Whether two blocks may share a chunk, judged by whether their anchors can combine.

    This is what stops a chunk spanning a page break: a chunk whose anchor names one page
    while half its text is on the next is a citation that reads correctly and points at the
    wrong place. Where the anchors cannot combine, the chunk closes.
    """
    if left == right:
        return True
    if isinstance(left, LineAnchor) and isinstance(right, LineAnchor):
        return True
    if isinstance(left, PageAnchor) and isinstance(right, PageAnchor):
        return left.page == right.page
    if isinstance(left, CellAnchor) and isinstance(right, CellAnchor):
        return left.sheet == right.sheet
    return False


def _merge_anchors(anchors: Sequence[Anchor]) -> Anchor:
    """The one anchor that covers all of ``anchors``.

    Only combinations :func:`_mergeable` admitted can arrive here, so every case below is
    reachable and none of them widens a location beyond what the blocks actually occupied.
    """
    first = anchors[0]
    if all(anchor == first for anchor in anchors):
        return first
    if isinstance(first, LineAnchor):
        lines = [anchor for anchor in anchors if isinstance(anchor, LineAnchor)]
        symbols = {anchor.symbol for anchor in lines}
        return LineAnchor(
            start=min(anchor.start for anchor in lines),
            end=max(anchor.end for anchor in lines),
            # A chunk covering two definitions belongs to neither, and naming one of them
            # would put a wrong symbol into the breadcrumb, which reaches the embedder.
            symbol=symbols.pop() if len(symbols) == 1 else None,
        )
    if isinstance(first, PageAnchor):
        pages = [anchor for anchor in anchors if isinstance(anchor, PageAnchor)]
        # An empty `rects` means "this page, no finer location" — a slide's speaker notes are
        # on the slide but not on its surface. Unioning rectangles across a group containing
        # one of those would produce an anchor covering only the blocks that *had* rectangles,
        # so the chunk would resolve to text that does not contain what it claims. Page-level
        # is the coarser answer and the only correct one: it is what every member of the
        # group can honestly be said to be inside.
        if any(not anchor.rects for anchor in pages):
            return PageAnchor(page=first.page)
        rects: list[Rect] = []
        for anchor in pages:
            rects.extend(anchor.rects)
        return PageAnchor(page=first.page, rects=tuple(rects))
    if isinstance(first, CellAnchor):
        cells = [anchor for anchor in anchors if isinstance(anchor, CellAnchor)]
        # Deduplicated against a set rather than by scanning the list being built: extending a
        # list from a generator that tests membership in that same list happens to work, and
        # is the kind of thing a later reader reasonably rewrites into something that does
        # not. Source order is kept, because a multi-area ref reads as the header rows first.
        areas: list[str] = []
        seen: set[str] = set()
        for anchor in cells:
            for part in anchor.ref.split(","):
                if part not in seen:
                    seen.add(part)
                    areas.append(part)
        return CellAnchor(sheet=first.sheet, ref=",".join(areas))
    return first


def _narrow_cell_anchor(
    anchor: Anchor, refs: Sequence[str] | None, header_rows: int, indices: Sequence[int]
) -> Anchor:
    """A table part's own anchor, covering its rows *and* the header repeated into it.

    Splitting a spreadsheet table improves provenance rather than costing it: each part
    addresses exactly its own rows, where a whole-table anchor would resolve to every row and
    fail the tightness bound.
    """
    if not isinstance(anchor, CellAnchor) or refs is None:
        return anchor
    wanted = [*range(header_rows), *indices]
    areas = [refs[i] for i in wanted if 0 <= i < len(refs)]
    if not areas:
        return anchor
    return CellAnchor(sheet=anchor.sheet, ref=",".join(_collapse_areas(areas)))


def _collapse_areas(areas: Sequence[str]) -> list[str]:
    """Join runs of adjacent single-row areas into ranges, keeping the order given."""
    collapsed: list[str] = []
    for area in areas:
        if collapsed and _adjacent(collapsed[-1], area):
            collapsed[-1] = f"{collapsed[-1].split(':')[0]}:{area.split(':')[-1]}"
            continue
        collapsed.append(area)
    return collapsed


def _adjacent(left: str, right: str) -> bool:
    """Whether ``right`` is the row immediately after ``left`` over the same columns."""
    left_end, right_start = left.rsplit(":", maxsplit=1)[-1], right.split(":", maxsplit=1)[0]
    left_columns, left_row = _split_reference(left_end)
    right_columns, right_row = _split_reference(right_start)
    if left_columns != right_columns or left_row is None or right_row is None:
        return False
    return right_row == left_row + 1


def _split_reference(reference: str) -> tuple[str, int | None]:
    digits = "".join(character for character in reference if character.isdigit())
    letters = reference[: len(reference) - len(digits)]
    return letters, int(digits) if digits else None


# --- small helpers -----------------------------------------------------------------------


def _promote_headings_when_that_is_all_there_is(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Emit headings as prose for a document that is *only* headings.

    A stub page or a table of contents would otherwise produce zero chunks and be
    indistinguishable from a document with no extractable text — which would put it in the
    bucket that triggers the scanned-corpus warning it has nothing to do with.
    """
    if not blocks or any(block.kind is not BlockKind.HEADING for block in blocks):
        return blocks
    return [
        block.model_copy(
            update={"kind": BlockKind.PROSE, "metadata": {**block.metadata, "headings_only": True}}
        )
        for block in blocks
    ]


def _dominant_kind(units: Sequence[_Unit]) -> BlockKind:
    """The kind of the majority of a chunk's tokens, ties going to the first unit."""
    totals: dict[BlockKind, int] = {}
    for unit in units:
        totals[unit.kind] = totals.get(unit.kind, 0) + unit.tokens
    best = max(totals.values())
    for unit in units:
        if totals[unit.kind] == best:
            return unit.kind
    return units[0].kind  # pragma: no cover - the loop above always returns


def _group_metadata(units: Sequence[_Unit], *, provisional: bool) -> Metadata:
    """Metadata a chunk inherits from its units, without the parser's per-block detail."""
    carried: Metadata = {}
    for key in ("table_part", "rows", "hard_split", "hard_split_at", "severity", "cell"):
        for unit in units:
            if key in unit.metadata:
                carried[key] = unit.metadata[key]
                break
    lang = _shared_lang(units)
    if lang is not None:
        # Promoted into a column by both stores, so `Filter.langs` resolves the same way in
        # the lexical leg and the dense one (`docs/retrieval.md` §3.3).
        carried["lang"] = lang
    if provisional:
        # Counted without the model that will embed these, so ingest must refuse them.
        carried["provisional"] = True
    return carried


def _shared_lang(units: Sequence[_Unit]) -> str | None:
    """The language every unit agrees on, or ``None`` when they do not.

    A chunk may hold a paragraph and the code block it introduces, and ``ParsedBlock.lang``
    means a code language on one and a natural language on the other. Naming either as the
    chunk's language would make a ``langs`` filter return the other one, so disagreement is
    undetermined — which is what ``None`` already means everywhere else it appears, rather
    than a stand-in for English.
    """
    langs = {unit.lang for unit in units}
    return langs.pop() if len(langs) == 1 else None


def _string_list(value: object) -> list[str] | None:
    """A metadata value as a list of strings, or ``None`` when it is anything else.

    Metadata arrives as JSON, so a parser can put anything there. A malformed value means
    the chunker has no structure to split on, which is a fallback rather than a crash.
    """
    if not isinstance(value, list):
        return None
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        return None
    return [item for item in items if isinstance(item, str)]


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def finalize_chunks(chunker: object, chunks: Sequence[Chunk]) -> list[Chunk]:
    """Apply a chunker's optional post-middleware finalizer.

    Third-party chunkers keep their existing protocol. The structural chunker supplies this
    hook because its exact bound counter is the authority for its fingerprinted final budget.
    """
    finalizer = getattr(chunker, "finalize", None)
    if not callable(finalizer):
        return list(chunks)
    typed = cast("Callable[[Sequence[Chunk]], list[Chunk]]", finalizer)
    return typed(chunks)


__all__ = [
    "ATOMIC_KINDS",
    "BREADCRUMB_TOKENS",
    "CHUNKER_NAME",
    "CHUNKER_VERSION",
    "MAX_TOKENS",
    "MIN_TOKENS",
    "OVERLAPPING_KINDS",
    "OVERLAP_TOKENS",
    "StructuralChunker",
    "finalize_chunks",
]
