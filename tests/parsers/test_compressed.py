"""Gzip and tar members: documents of their own, under the zip parser's limits."""

from __future__ import annotations

import gzip
import io
import tarfile

import pytest

from manicule.core.content import DocumentStatus, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import read_blocks
from manicule.parsers.compressed import CompressedParser
from manicule.parsers.config import ArchiveConfig
from manicule.parsers.expansion import (
    CONTAINER_DEPTH,
    ExpandedMember,
    MemberFailure,
    SupportsExpansion,
    read_members,
)
from tests.parsers.support import raw_of


def _tar(entries: dict[str, bytes], *, extra: list[tarfile.TarInfo] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info in extra or []:
            archive.addfile(info)
    return buffer.getvalue()


def _raw(data: bytes, uri: str, media_type: str, **metadata: object) -> RawDocument:
    document = raw_of(data, media_type, uri=uri)
    return document.model_copy(update={"metadata": metadata}) if metadata else document


@pytest.fixture
def parser() -> CompressedParser:
    return CompressedParser(ArchiveConfig())


async def test_a_gzip_expands_to_one_member_named_without_the_suffix(
    parser: CompressedParser,
) -> None:
    raw = _raw(gzip.compress(b"notes in a file\n"), "mail:m!/notes.txt.gz", "application/gzip")

    (member,) = await read_members(parser, raw)

    assert isinstance(member, ExpandedMember)
    assert member.uri == "gzip:mail:m!/notes.txt.gz!/notes.txt"
    assert member.raw.media_type == "text/plain"
    assert member.raw.as_bytes() == b"notes in a file\n"
    assert member.raw.metadata[CONTAINER_DEPTH] == 1


async def test_a_gzipped_tar_is_a_tar_one_level_down(parser: CompressedParser) -> None:
    raw = _raw(gzip.compress(_tar({"a.txt": b"alpha\n"})), "x!/bundle.tgz", "application/gzip")

    (member,) = await read_members(parser, raw)

    assert isinstance(member, ExpandedMember)
    assert member.raw.media_type == "application/x-tar"
    inner = await read_members(parser, member.raw)
    assert [m.uri.rsplit("!/", 1)[1] for m in inner if isinstance(m, ExpandedMember)] == ["a.txt"]
    assert isinstance(inner[0], ExpandedMember)
    assert inner[0].raw.metadata[CONTAINER_DEPTH] == 2


async def test_a_gzip_with_no_suffix_is_named_content(parser: CompressedParser) -> None:
    raw = _raw(gzip.compress(b"<?xml version='1.0'?><a/>"), "x!/report", "application/gzip")

    (member,) = await read_members(parser, raw)

    assert isinstance(member, ExpandedMember)
    assert member.uri.endswith("!/content")
    assert member.raw.media_type == "application/xml"


async def test_a_tar_yields_regular_files_and_reports_links_and_hostile_names(
    parser: CompressedParser,
) -> None:
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/hostname"
    data = _tar({"dir/a.md": b"# a\n", "../escape.txt": b"x"}, extra=[link])

    outcomes = await read_members(parser, _raw(data, "x!/b.tar", "application/x-tar"))

    members = [o for o in outcomes if isinstance(o, ExpandedMember)]
    failures = [o for o in outcomes if isinstance(o, MemberFailure)]
    assert [m.raw.media_type for m in members] == ["text/markdown"]
    assert len(failures) == 2
    assert all(f.status is DocumentStatus.FAILED for f in failures)
    assert any("escapes the archive root" in f.reason for f in failures)
    assert any("link, device or FIFO" in f.reason for f in failures)
    assert all("../" not in f.uri for f in failures), "the hostile name is not reproduced"


async def test_a_decompression_bomb_is_stopped_by_the_counter() -> None:
    bomb = gzip.compress(b"\x00" * (50 * 1024 * 1024))
    parser = CompressedParser(ArchiveConfig(max_compression_ratio=20))

    (outcome,) = await read_members(parser, _raw(bomb, "x!/b.gz", "application/gzip"))

    assert isinstance(outcome, MemberFailure)
    assert "decompression bomb" in outcome.reason


async def test_the_per_member_ceiling_applies_to_a_gzip() -> None:
    parser = CompressedParser(ArchiveConfig(max_member_bytes=1024, max_compression_ratio=1e9))

    (outcome,) = await read_members(
        parser, _raw(gzip.compress(b"a" * 4096), "x!/b.gz", "application/gzip")
    )

    assert isinstance(outcome, MemberFailure)
    assert "per-member limit" in outcome.reason


async def test_nesting_past_the_depth_limit_is_reported(parser: CompressedParser) -> None:
    raw = _raw(gzip.compress(b"x"), "x!/b.gz", "application/gzip", **{CONTAINER_DEPTH: 3})

    (outcome,) = await read_members(parser, raw)

    assert isinstance(outcome, MemberFailure)
    assert outcome.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE


async def test_a_truncated_gzip_is_a_failed_member_not_a_crash(parser: CompressedParser) -> None:
    broken = gzip.compress(b"hello world " * 100)[:-10]

    (outcome,) = await read_members(parser, _raw(broken, "x!/b.gz", "application/gzip"))

    assert isinstance(outcome, MemberFailure)
    assert outcome.status is DocumentStatus.FAILED


async def test_something_that_is_neither_is_declined(parser: CompressedParser) -> None:
    with pytest.raises(ParseError, match="neither a gzip stream nor a tar"):
        await read_blocks(parser, _raw(b"plain words\n" * 50, "x!/b.gz", "application/gzip"))


async def test_the_parser_emits_no_blocks_and_satisfies_the_expansion_protocol(
    parser: CompressedParser,
) -> None:
    raw = _raw(gzip.compress(b"x"), "x!/b.gz", "application/gzip")

    assert await read_blocks(parser, raw) == []
    assert isinstance(parser, SupportsExpansion)


async def test_repeated_names_get_distinct_addresses_and_keep_their_type(
    parser: CompressedParser,
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("data.csv", "data.csv#1", "data.csv", "data.csv"):
            info = tarfile.TarInfo(name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"a,b\n"))

    outcomes = await read_members(parser, _raw(buffer.getvalue(), "x!/d.tar", "application/x-tar"))

    members = [o for o in outcomes if isinstance(o, ExpandedMember)]
    assert len({m.source_id for m in members}) == 4
    assert members[0].raw.media_type == "text/csv"
    assert members[2].raw.media_type == "text/csv", "a qualified address keeps the file's own type"


async def test_a_literal_name_matching_a_generated_qualifier_gets_its_own_address(
    parser: CompressedParser,
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("a.txt", "a.txt#2", "./a.txt", "a.txt"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    outcomes = await read_members(parser, _raw(buffer.getvalue(), "x!/d.tar", "application/x-tar"))

    assert len({o.source_id for o in outcomes}) == 4


async def test_a_truncated_tar_keeps_the_members_read_before_the_damage(
    parser: CompressedParser,
) -> None:
    whole = _tar({"good.txt": b"readable\n", "bad.txt": b"x" * 4000})
    cut = whole[: 512 + 512 + 512 + 100]

    outcomes = await read_members(parser, _raw(cut, "x!/d.tar", "application/x-tar"))

    members = [o for o in outcomes if isinstance(o, ExpandedMember)]
    failures = [o for o in outcomes if isinstance(o, MemberFailure)]
    assert [m.uri.rsplit("!/", 1)[1] for m in members] == ["good.txt"]
    assert failures
    assert failures[-1].truncates, "the listing is partial and must not reconcile as complete"
