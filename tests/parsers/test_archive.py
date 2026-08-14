"""The zip parser: members are documents, and four limits that are counted, not read."""

from __future__ import annotations

import copy
import io
import zipfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import override

import pytest

from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, DocumentStatus, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.ids import content_hash
from manicule.core.protocols import read_blocks
from manicule.parsers.archive import NOT_AN_ARCHIVE, ArchiveConfig, ArchiveParser
from manicule.parsers.expansion import (
    ExpandedMember,
    MemberFailure,
    MemberOutcome,
    SupportsExpansion,
    aclose_members,
    inner_path,
    read_members,
)
from manicule.testing import assert_round_trip
from tests.corpus.archive import BOMB_DECLARED_SIZE, BOMB_REAL_SIZE
from tests.parsers.support import check_fixture, document_for, raw_from, raw_of

A_MEGABYTE = 1024 * 1024


@pytest.fixture
def parser() -> ArchiveParser:
    return ArchiveParser(ArchiveConfig())


def _raw(corpus: Path, name: str) -> RawDocument:
    return raw_from(corpus / "archive" / name, "application/zip")


async def _expand(parser: ArchiveParser, raw: RawDocument) -> list[MemberOutcome]:
    """Drained through ``read_members``, which closes the member stream in a ``finally``."""
    return await read_members(parser, raw)


def _members(outcomes: Sequence[MemberOutcome]) -> list[ExpandedMember]:
    return [outcome for outcome in outcomes if isinstance(outcome, ExpandedMember)]


def _failures(outcomes: Sequence[MemberOutcome]) -> list[MemberFailure]:
    return [outcome for outcome in outcomes if isinstance(outcome, MemberFailure)]


# --- what a container is -------------------------------------------------------------------


async def test_an_archive_emits_no_blocks_because_its_members_are_documents(
    parser: ArchiveParser, corpus: Path
) -> None:
    """Zero chunks, and not even a manifest.

    A chunk listing filenames is retrieval noise competing with the real content inside the
    archive, and it would win queries against the documents it lists. The container's status
    is ``container`` rather than ``no_extractable_text``: nothing failed, and conflating the
    two would put every archive into the bucket that triggers the scanned-corpus warning.
    """
    report = await check_fixture(parser, _raw(corpus, "typical.zip"))
    assert report.blocks == 0


async def test_resolve_returns_nothing_because_this_parser_produces_no_anchors(
    parser: ArchiveParser, corpus: Path
) -> None:
    """A member's anchors belong to whichever parser read the member.

    Resolving one here would mean this parser claiming a location it did not produce and
    cannot check.
    """
    raw = _raw(corpus, "typical.zip")
    assert await parser.resolve(LineAnchor(start=1, end=1), raw) is None


async def test_the_parser_satisfies_the_expansion_protocol(parser: ArchiveParser) -> None:
    """Members reach the pipeline the same way email attachments do, through one protocol."""
    assert isinstance(parser, SupportsExpansion)


async def test_every_member_becomes_a_document_addressed_through_the_container_separator(
    parser: ArchiveParser, corpus: Path
) -> None:
    """``zip:<container>!/<inner path>``, which survives being pasted into a bug report.

    Identity comes from the inner path and never from the position: a member that moves within
    the archive is the same member, and one inserted ahead of another must not inherit its
    identity — which is what an ordinal would do, silently, on the next sync.
    """
    members = _members(await _expand(parser, _raw(corpus, "typical.zip")))
    assert [member.uri for member in members] == [
        "zip:typical.zip!/readme.txt",
        "zip:typical.zip!/reports/january.txt",
        "zip:typical.zip!/reports/february.md",
    ]
    assert all(member.source_id.endswith(member.uri.split("!/", 1)[1]) for member in members)
    assert [member.raw.media_type for member in members] == [
        "text/plain",
        "text/plain",
        "text/markdown",
    ]


async def test_a_directory_entry_is_skipped_without_being_reported_as_a_failure(
    parser: ArchiveParser, corpus: Path
) -> None:
    """A directory is not a document, so it is neither a member nor a problem.

    Reporting it would fill diagnostics with entries nobody can act on, which is how a
    diagnostic stops being read.
    """
    outcomes = await _expand(parser, _raw(corpus, "typical.zip"))
    assert _failures(outcomes) == []
    assert len(outcomes) == 3


async def test_an_empty_archive_yields_no_members_and_does_not_raise(
    parser: ArchiveParser, corpus: Path
) -> None:
    """An archive with nothing in it is a normal outcome, not a failure."""
    assert await _expand(parser, _raw(corpus, "empty.zip")) == []


async def test_a_nested_archive_is_a_member_and_is_not_descended_into_here(
    parser: ArchiveParser, corpus: Path
) -> None:
    """One level, so the pipeline's queue makes the traversal breadth-first.

    Descending here would let one wide branch of one archive starve every other document in
    the batch, and would put the whole-tree budgets in parser state where a second archive
    would inherit them.
    """
    members = _members(await _expand(parser, _raw(corpus, "nested.zip")))
    inner = next(member for member in members if member.uri.endswith("inner.zip"))
    assert inner.raw.media_type == "application/zip"
    assert inner.depth == 1
    assert inner.raw.metadata["container_depth"] == 1


async def test_expanding_the_same_archive_twice_yields_identical_members(
    parser: ArchiveParser, corpus: Path
) -> None:
    """A container that varies churns every member's document on every re-ingest.

    Directory iteration order is exactly where this breaks, and it breaks quietly: the members
    are all still there, with different identities.
    """
    raw = _raw(corpus, "typical.zip")
    first = await _expand(parser, raw)
    assert [outcome.model_dump_json() for outcome in first] == [
        outcome.model_dump_json() for outcome in await _expand(parser, raw)
    ]


# --- the zip bomb ---------------------------------------------------------------------------


async def test_a_zip_bomb_is_stopped_by_the_counting_wrapper_not_by_its_declared_size(
    parser: ArchiveParser, corpus: Path
) -> None:
    """The header says a kilobyte; the stream is four megabytes.

    Every header-based check passes this member — its declared size is under the per-member
    limit, under the tree limit, and its declared ratio is under the cap. What stops it is the
    counter around the stream, which is the only thing that can, because
    ``ZipInfo.file_size`` is a field inside an archive an attacker wrote.
    """
    raw = _raw(corpus, "bomb.zip")
    with zipfile.ZipFile(io.BytesIO(raw.as_bytes())) as archive:
        declared = archive.getinfo("bomb.bin")
    limits = ArchiveConfig()
    assert declared.file_size == BOMB_DECLARED_SIZE
    assert declared.file_size < limits.max_member_bytes
    assert declared.file_size < limits.max_total_bytes
    assert declared.file_size / declared.compress_size < limits.max_compression_ratio

    failures = _failures(await _expand(parser, raw))
    assert len(failures) == 1
    assert "while streaming" in failures[0].reason
    assert failures[0].status is DocumentStatus.FAILED


async def test_trusting_the_declared_size_does_not_stop_the_bomb(corpus: Path) -> None:
    """The guard disabled, to show it was carrying the weight.

    A reader that lets the archive's own ``file_size`` bound the read never reports the limit
    the member actually hit. :mod:`zipfile` truncates the decompressed stream at the declared
    size and then fails the checksum, so the attack is reported as a corrupt file — and on an
    archive whose declared size happened to match its checksum it would instead be indexed as
    a kilobyte of a four-megabyte member, silently.
    """
    outcomes = await _expand(_HeaderTrustingParser(ArchiveConfig()), _raw(corpus, "bomb.zip"))

    assert outcomes, "the disabled guard produced no outcomes at all, so this measures nothing"
    reasons = [failure.reason for failure in _failures(outcomes)]
    assert reasons, "the disabled guard produced no failure, so there is nothing to compare"

    # The point of the test: whatever went wrong, it was not reported as the limit. Every
    # assertion below would pass vacuously on an empty list, which is why the two above are
    # there — the shape is asserted before the property.
    assert not any("while streaming" in reason for reason in reasons)
    assert any("could not be decompressed" in reason for reason in reasons), reasons
    delivered = [len(member.raw.as_bytes()) for member in _members(outcomes)]
    assert all(size < BOMB_REAL_SIZE for size in delivered)


async def test_the_per_member_ceiling_stops_a_member_whose_stream_outruns_its_header(
    corpus: Path,
) -> None:
    """The second of the four limits, with the ratio cap taken out of the way.

    Any one limit is bypassable, so each has to be shown to work on its own rather than only
    in the company of the others.
    """
    parser = ArchiveParser(ArchiveConfig(max_compression_ratio=1e9, max_member_bytes=A_MEGABYTE))
    failures = _failures(await _expand(parser, _raw(corpus, "bomb.zip")))
    assert len(failures) == 1
    assert "per-member limit while streaming" in failures[0].reason


async def test_the_whole_tree_budget_is_exhausted_while_streaming_and_ends_the_archive(
    corpus: Path,
) -> None:
    """A tree budget is fatal to the archive: every remaining member hits the same wall.

    It fails that archive and never the batch — members already expanded keep their documents,
    and the rest of the run continues.
    """
    parser = ArchiveParser(
        ArchiveConfig(
            max_compression_ratio=1e9, max_member_bytes=BOMB_REAL_SIZE * 2, max_total_bytes=4096
        )
    )
    failures = _failures(await _expand(parser, _raw(corpus, "bomb.zip")))
    assert len(failures) == 1
    assert "archive tree exceeded" in failures[0].reason


async def test_a_wide_archive_stops_at_the_member_count_limit(corpus: Path) -> None:
    """The many-tiny-files variant, which every byte limit passes.

    Sixty members of twenty-seven bytes each is nothing at all by weight, and ten thousand of
    them is still nothing — which is the point: the count is the only limit that catches it.
    """
    parser = ArchiveParser(ArchiveConfig(max_members=10))
    outcomes = await _expand(parser, _raw(corpus, "wide.zip"))
    assert len(_members(outcomes)) == 10
    assert len(_failures(outcomes)) == 1
    assert "member count exceeded" in _failures(outcomes)[0].reason


async def test_a_tree_budget_already_spent_upstream_is_honored(
    parser: ArchiveParser, corpus: Path
) -> None:
    """The budgets are whole-tree, and this parser expands one level.

    They therefore travel with the document rather than living in the parser, so an archive
    nested three deep cannot restart a budget its ancestors already spent.
    """
    raw = _raw(corpus, "typical.zip")
    spent = raw.model_copy(
        update={"metadata": {**raw.metadata, "container_tree_bytes": 1024**3 - 10}}
    )
    failures = _failures(await _expand(parser, spent))
    assert failures
    assert "archive tree exceeded" in failures[0].reason


# --- names, links, secrets and depth ---------------------------------------------------------


async def test_members_whose_names_collide_get_one_address_each(corpus: Path) -> None:
    """An archive contributes as many documents as it has members, or says why not.

    Three of these entries normalize to one name — a literal duplicate, which appending to a
    zip produces, and a ``./``-prefixed spelling — and two hostile names normalize to none.
    Storage reconciles members by ``source_id``, which is derived from the normalized name, so
    a collision is not an error anybody sees: the later member overwrites the earlier one and
    the archive quietly contributes fewer documents than it contains. Counting the addresses
    is the only way that shows up.
    """
    outcomes = await _expand(ArchiveParser(ArchiveConfig()), _raw(corpus, "colliding.zip"))

    assert len(outcomes) == 5, "five members, five outcomes"
    addresses = [outcome.source_id for outcome in outcomes]
    assert len(set(addresses)) == 5, f"members share an address: {addresses}"

    bodies = [member.raw.as_bytes() for member in _members(outcomes)]
    assert len(bodies) == 3
    assert len({bytes(body) for body in bodies}) == 3, "a colliding member's content was lost"


async def test_a_member_named_to_escape_the_archive_root_is_rejected_and_never_rewritten(
    parser: ArchiveParser, corpus: Path
) -> None:
    """Sanitizing ``../escape.txt`` into ``escape.txt`` invents a document.

    Members are parsed in memory and never written to disk, so the traversal cannot touch the
    filesystem — but the name still becomes a ``uri`` shown to users and stored in the index,
    and a citation naming a file the archive never contained is the failure this whole design
    is about.
    """
    outcomes = await _expand(parser, _raw(corpus, "traversal.zip"))
    failures = _failures(outcomes)
    assert len(failures) == 1
    assert "escapes the archive root" in failures[0].reason
    assert failures[0].metadata["member_name"] == "../escape.txt"
    assert ".." not in failures[0].uri
    assert [member.uri for member in _members(outcomes)] == ["zip:traversal.zip!/safe.txt"]


async def test_a_symlink_member_is_skipped_with_a_reason(
    parser: ArchiveParser, corpus: Path
) -> None:
    """A zip can name a file outside itself in the Unix external attributes.

    Nothing follows one today. The defense is against an extraction path being added later
    without remembering, which is exactly the kind of change nobody re-reads this file before
    making.
    """
    failures = _failures(await _expand(parser, _raw(corpus, "symlink.zip")))
    assert len(failures) == 1
    assert "symlink archive member" in failures[0].reason


async def test_an_encrypted_member_is_reported_and_the_archive_keeps_going(
    parser: ArchiveParser, corpus: Path
) -> None:
    """One member nobody can read does not cost the archive its other members."""
    outcomes = await _expand(parser, _raw(corpus, "encrypted.zip"))
    assert [member.uri for member in _members(outcomes)] == ["zip:encrypted.zip!/plain.txt"]
    assert _failures(outcomes)[0].reason.startswith("encrypted archive member")


async def test_a_member_past_the_nesting_limit_is_reported_rather_than_silently_dropped(
    corpus: Path,
) -> None:
    """Three levels is already unusual; deeper is a mistake or an attack.

    Either way the boundary is visible: the member is stored with a reason and re-indexable if
    the limit is raised, rather than absent with no record that it ever existed.
    """
    parser = ArchiveParser(ArchiveConfig(max_depth=2))
    raw = _raw(corpus, "typical.zip")
    deep = raw.model_copy(update={"metadata": {**raw.metadata, "container_depth": 2}})
    failures = _failures(await _expand(parser, deep))
    assert len(failures) == 3
    for failure in failures:
        assert failure.reason.startswith("archive nesting depth exceeded")
        assert failure.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE


async def test_a_member_repeating_a_container_on_its_path_is_stopped_by_content_hash(
    parser: ArchiveParser, corpus: Path
) -> None:
    """A zip cannot literally contain itself; identical content nested inside itself is easy.

    Detecting it costs one hash of bytes already in hand, and not detecting it costs the
    recursion.
    """
    inner = (corpus / "archive" / "inner.zip").read_bytes()
    raw = _raw(corpus, "nested.zip")
    seeded = raw.model_copy(
        update={"metadata": {**raw.metadata, "container_path_hashes": [content_hash(inner)]}}
    )
    failures = _failures(await _expand(parser, seeded))
    assert len(failures) == 1
    assert "cycle detected" in failures[0].reason


async def test_a_member_carries_the_path_hashes_that_will_detect_a_cycle_below_it(
    parser: ArchiveParser, corpus: Path
) -> None:
    """The recursion path travels with the member, because the parser holds no state.

    Without it every level would start its cycle detection over and a self-referential archive
    would descend until the depth limit caught it — which is a limit, not a defense.
    """
    members = _members(await _expand(parser, _raw(corpus, "nested.zip")))
    inner = next(member for member in members if member.uri.endswith("inner.zip"))
    hashes = inner.raw.metadata["container_path_hashes"]
    assert isinstance(hashes, list)
    assert hashes == [content_hash(inner.raw.as_bytes())]


# --- the OOXML trap ---------------------------------------------------------------------------


async def test_an_office_file_with_a_zip_extension_is_declined(
    parser: ArchiveParser, corpus: Path
) -> None:
    """The realistic version of the §9.4 bug, where the extension is wrong too.

    A sniffer looking for ``PK\\x03\\x04`` calls every Office document an archive, and an
    archive parser that believed it would index ``word/document.xml`` as a member — producing
    citations into XML for a document whose text was right there.
    """
    with pytest.raises(ParseError, match=NOT_AN_ARCHIVE):
        await read_blocks(parser, _raw(corpus, "ooxml.zip"))


async def test_an_e_book_or_open_document_container_is_declined_too(
    parser: ArchiveParser, corpus: Path
) -> None:
    """The other signature: an uncompressed ``mimetype`` member first, at offset zero.

    Checking only for ``[Content_Types].xml`` would let every EPUB and ODF file through, and
    those are the ones most likely to arrive with a generic media type.
    """
    with pytest.raises(ParseError, match=NOT_AN_ARCHIVE):
        await read_blocks(parser, _raw(corpus, "ebook.zip"))


async def test_expanding_a_document_container_is_refused_as_well_as_parsing_it(
    parser: ArchiveParser, corpus: Path
) -> None:
    """Both entry points, because either one alone leaves the trap open.

    Whether a document reaches this parser through the chain or through a container walk, the
    answer has to be the same one.
    """
    with pytest.raises(ParseError, match=NOT_AN_ARCHIVE):
        await _expand(parser, _raw(corpus, "ooxml.zip"))


async def test_something_that_is_not_a_zip_at_all_is_declined(parser: ArchiveParser) -> None:
    """Declining lets the next parser in the chain try, and says what was wrong."""
    with pytest.raises(ParseError, match="declining"):
        await read_blocks(parser, raw_of(b"not a zip", "application/zip"))


async def test_zero_bytes_is_declined_rather_than_read_as_an_empty_archive(
    parser: ArchiveParser, corpus: Path
) -> None:
    """An empty file is not an archive with nothing in it, and the two must not merge.

    A real empty archive still has an end-of-central-directory record. Reading zero bytes as
    an empty container would give a truncated download the status ``container`` and no member
    would ever be missed.
    """
    with pytest.raises(ParseError, match="declining"):
        await read_blocks(parser, _raw(corpus, "zero-bytes.zip"))


# --- the guard itself --------------------------------------------------------------------------


async def test_an_archive_parser_that_invents_a_block_is_caught_by_the_round_trip_check(
    corpus: Path,
) -> None:
    """The guard is load-bearing, proved by breaking it.

    A manifest chunk is the tempting mistake here: it looks like content, it is retrievable,
    and its anchor addresses nothing in the archive. The round-trip contract refuses an anchor
    the parser cannot resolve, which is what makes "members are documents, not chunks"
    enforceable rather than a convention.
    """
    raw = _raw(corpus, "typical.zip")
    with pytest.raises(AssertionError):
        await assert_round_trip(
            _ManifestParser(ArchiveConfig()), raw, fixture="manifest", document=document_for(raw)
        )


class _ManifestParser(ArchiveParser):
    """Emits a chunk listing the archive's filenames, with an anchor nobody can resolve."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        with zipfile.ZipFile(io.BytesIO(raw.as_bytes())) as archive:
            names = "\n".join(archive.namelist())
        yield ParsedBlock(kind=BlockKind.PROSE, text=names, anchor=LineAnchor(start=1, end=3))


class _HeaderTrustingParser(ArchiveParser):
    """Reads each member with the size the archive declares, which is the disabled guard.

    Kept beside the parser it contradicts so the two cannot drift apart: if the real reader
    stops setting its own ceiling, the test that uses this class starts passing for the wrong
    reason and the one above it fails.
    """

    @override
    def _read(self, archive: zipfile.ZipFile, info: zipfile.ZipInfo, used_bytes: int) -> bytes:
        del used_bytes
        honest = copy.copy(info)
        with archive.open(honest) as handle:
            return handle.read()


# --- closing the archive on every path -----------------------------------------------------


async def test_a_member_stream_stopped_after_one_member_still_closes_the_archive(
    corpus: Path,
) -> None:
    """The real leak this discipline exists for.

    A consumer that takes one member and stops — a pipeline with a member budget, an assertion
    failing between two members — throws ``GeneratorExit`` in at the ``yield``. Only a
    ``finally`` runs after that, so the ``ZipFile`` and its decompression stream are released
    because a ``with`` encloses every ``yield``, not because the loop ran to the end.
    """
    parser = _RecordingParser(ArchiveConfig())
    stream = parser.expand(_raw(corpus, "typical.zip"))
    async for _ in stream:
        break
    await aclose_members(stream)
    assert parser.opened
    assert all(archive.fp is None for archive in parser.opened)
    assert await _is_closed(stream)


async def test_an_archive_parser_that_closes_after_its_last_member_is_caught_by_that_test(
    corpus: Path,
) -> None:
    """Proof the check above is load-bearing rather than decorative.

    Closing after the loop is correct on a full drain, which is every ordinary run. It is only
    wrong on the path that is hard to notice, and there the consequence is not a warning.
    """
    parser = _LeakyParser(ArchiveConfig())
    stream = parser.expand(_raw(corpus, "typical.zip"))
    async for _ in stream:
        break
    await aclose_members(stream)
    assert parser.opened
    assert any(archive.fp is not None for archive in parser.opened)
    for archive in parser.opened:
        archive.close()


class _RecordingParser(ArchiveParser):
    """The real parser, keeping a handle on every archive it opened so a test can check it."""

    def __init__(self, config: ArchiveConfig) -> None:
        super().__init__(config)
        self.opened: list[zipfile.ZipFile] = []

    @override
    def _opened(self, raw: RawDocument) -> zipfile.ZipFile:
        archive = super()._opened(raw)
        self.opened.append(archive)
        return archive


class _LeakyParser(_RecordingParser):
    """Closes the archive after its last member instead of around every ``yield``."""

    @override
    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        archive = self._opened(raw)
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = inner_path(info.filename)
            if path is None:  # pragma: no cover - typical.zip has no hostile names
                continue
            yield ExpandedMember(
                source_id=path,
                uri=f"zip:{raw.uri}!/{path}",
                raw=raw_of(b"", "text/plain", uri=path),
                depth=1,
            )
        archive.close()


async def _is_closed(stream: AsyncIterator[object]) -> bool:
    """Whether a stream is finished with, rather than merely paused.

    A generator that has been closed raises ``StopAsyncIteration`` at once instead of resuming
    where it left off, which is the observable difference between "released" and "suspended
    holding whatever it had open".
    """
    try:
        await anext(stream)
    except StopAsyncIteration:
        return True
    return False
