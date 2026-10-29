"""The mbox reader: where a message begins and ends, and what it is.

``docs/connectors/mbox.md`` §3 and §4. These are the properties a reader that handles one dialect
gets silently wrong in the others: a body line that looks like a separator, a ``Content-Length``
nobody checked, a copy that differs from another only in what a mail client recorded.
"""

from __future__ import annotations

import mailbox
import re
from pathlib import Path

import pytest

from manicule.connectors import mbox_reader as reader
from tests.connectors.mbox_support import (
    SEPARATOR,
    Dialect,
    Filing,
    append_mbox,
    message,
    messages,
    with_headers,
    write_mbox,
)

FROM_LINE_BODY = (
    "A paragraph.\n\nFrom the top of the quarter, the figures held.\nThanks: all of it.\n"
)
QUOTED_BODY = "Someone wrote:\n>From the log: restarted\n>>From the log: restarted twice\n"


def scanned(path: Path) -> list[reader.Located]:
    return list(reader.scan(path, path.stat().st_size))


def digests(path: Path) -> list[str]:
    return [located.digest for located in scanned(path)]


def test_a_file_is_cut_into_its_messages_and_each_reads_back(tmp_path: Path) -> None:
    raws = messages(5, attachment=True)
    path = write_mbox(tmp_path / "box", raws)
    found = scanned(path)
    assert len(found) == 5
    for located in found:
        content = reader.read(path, located.offset, located.length, located.digest)
        assert len(content) == located.size
        assert content.startswith(b"From: pat@example.org\n")
        assert b"\r" not in content


def test_the_reader_agrees_with_the_standard_library_where_they_should(tmp_path: Path) -> None:
    """On ordinary files the two cut the same messages; the disagreements are the point of §3.5."""
    path = write_mbox(tmp_path / "box", messages(6))
    ours = [reader.read(path, m.offset, m.length, m.digest) for m in scanned(path)]
    box = mailbox.mbox(path, create=False)
    try:
        theirs = [bytes(m) for m in box]
    finally:
        box.close()
    assert len(ours) == len(theirs) == 6
    for mine, other in zip(ours, theirs, strict=True):
        assert mine.replace(b"\r\n", b"\n").strip() == other.replace(b"\r\n", b"\n").strip()


@pytest.mark.parametrize("dialect", ["mboxo", "mboxrd", "mboxcl"])
def test_every_dialect_files_a_message_with_a_from_line_as_one_message(
    tmp_path: Path, dialect: Dialect
) -> None:
    raws = [message(1, body=FROM_LINE_BODY), message(2)]
    path = write_mbox(tmp_path / "box", raws, Filing(dialect=dialect))
    found = scanned(path)
    assert len(found) == 2
    body = reader.read(path, found[0].offset, found[0].length, found[0].digest)
    assert b"\nFrom the top of the quarter, the figures held.\n" in body


def test_one_message_is_one_identity_in_every_dialect_and_line_ending(tmp_path: Path) -> None:
    raws = [message(1, body=FROM_LINE_BODY), message(2)]
    identities = {
        (dialect, eol): digests(
            write_mbox(tmp_path / f"{dialect}-{eol!r}", raws, Filing(dialect=dialect, eol=eol))
        )
        for dialect in ("mboxo", "mboxrd", "mboxcl")
        for eol in (b"\n", b"\r\n")
    }
    assert len({tuple(value) for value in identities.values()}) == 1


def test_a_quoted_from_line_is_unquoted_once_unless_a_length_is_believed(tmp_path: Path) -> None:
    """§3.3: the rule is mboxrd's, so an mboxo file loses one level of quoting on ``>From``."""
    raws = [message(1, body=QUOTED_BODY)]
    rd = write_mbox(tmp_path / "rd", raws, Filing(dialect="mboxrd"))
    cl = write_mbox(tmp_path / "cl", raws, Filing(dialect="mboxcl"))
    o = write_mbox(tmp_path / "o", raws, Filing(dialect="mboxo"))
    for path in (rd, cl):
        kept = reader.read(path, *_where(path))
        assert b"\n>From the log: restarted\n>>From the log: restarted twice\n" in kept
    lossy = reader.read(o, *_where(o))
    assert b"\nFrom the log: restarted\n>From the log: restarted twice\n" in lossy


def _where(path: Path) -> tuple[int, int, str]:
    (located,) = scanned(path)
    return located.offset, located.length, located.digest


def test_a_content_length_is_believed_only_when_it_lands_on_a_boundary(tmp_path: Path) -> None:
    """A body whose lines look exactly like a message is one message, if its length says so."""
    body = "Intro.\n\nFrom x@example.org Tue Feb  3 09:14:00 2026\nSubject: inner\n\nInner body.\n"
    path = write_mbox(
        tmp_path / "box", [message(1, body=body), message(2)], Filing(dialect="mboxcl")
    )
    assert len(scanned(path)) == 2

    # The same bytes without a length are two messages plus a third the body fakes: nothing in
    # the file says the inner "message" is not one.
    plain = tmp_path / "plain"
    plain.write_bytes(re.sub(rb"Content-Length: \d+\n", b"", path.read_bytes()))
    assert len(scanned(plain)) == 3


def test_a_wrong_content_length_is_ignored(tmp_path: Path) -> None:
    raw = write_mbox(tmp_path / "box", messages(3), Filing(dialect="mboxrd")).read_bytes()
    broken = raw.replace(b"\nSubject:", b"\nContent-Length: 7\nSubject:")
    path = tmp_path / "broken"
    path.write_bytes(broken)
    assert len(scanned(path)) == 3


def test_a_from_line_not_preceded_by_a_blank_line_does_not_start_a_message(
    tmp_path: Path,
) -> None:
    """§3.1 (2): a writer that omits the blank line is not guessed at, and nothing is split."""
    path = tmp_path / "box"
    path.write_bytes(
        SEPARATOR + b"\nSubject: one\n\nbody\n" + SEPARATOR + b"\nSubject: two\n\nbody two\n"
    )
    assert len(scanned(path)) == 1


def test_a_final_message_with_no_trailing_newline_is_delivered(tmp_path: Path) -> None:
    path = tmp_path / "box"
    path.write_bytes(SEPARATOR + b"\nSubject: one\n\nno newline at the end")
    (located,) = scanned(path)
    assert reader.read(path, located.offset, located.length, located.digest).endswith(b"the end")


def test_a_copy_that_differs_only_in_what_a_client_recorded_is_the_same_message(
    tmp_path: Path,
) -> None:
    plain = write_mbox(tmp_path / "a", [message(1)])
    read = write_mbox(
        tmp_path / "b",
        [with_headers(message(1), "X-Mozilla-Status: 0001", "X-Status: R", "Status: RO")],
    )
    assert digests(plain) == digests(read)
    # ...and it is delivered as the file says, headers included.
    content = reader.read(read, *_where(read))
    assert b"X-Mozilla-Status: 0001" in content


def test_a_different_message_is_a_different_identity(tmp_path: Path) -> None:
    one = digests(write_mbox(tmp_path / "a", [message(1)]))
    other = digests(write_mbox(tmp_path / "b", [message(1, body="Different words.")]))
    assert one != other


@pytest.mark.parametrize(
    "flag",
    [("X-Mozilla-Status: 0008",), ("X-Mozilla-Status: 0009",), ("X-Status: D",), ("X-Status: AD",)],
)
def test_a_message_flagged_deleted_is_not_a_message(tmp_path: Path, flag: tuple[str, ...]) -> None:
    path = write_mbox(tmp_path / "box", [with_headers(message(1), *flag), message(2)])
    assert len(scanned(path)) == 1


def test_a_folders_placeholder_message_is_not_a_message(tmp_path: Path) -> None:
    placeholder = message(0).replace(
        b"Archived message 0", b"DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA"
    )
    imap = with_headers(message(9), "X-IMAP: 1700000000 0000000001")
    path = write_mbox(tmp_path / "box", [placeholder, imap, message(2)])
    assert len(scanned(path)) == 1


def test_a_message_is_read_back_only_if_it_is_still_that_message(tmp_path: Path) -> None:
    path = write_mbox(tmp_path / "box", messages(3))
    first, second, _ = scanned(path)
    rewritten = write_mbox(tmp_path / "box", [message(2), message(3)])
    with pytest.raises(reader.MessageChangedError):
        reader.read(rewritten, first.offset, first.length, first.digest)
    with pytest.raises(reader.MessageChangedError):
        reader.read(rewritten, second.offset, second.length, second.digest)
    with pytest.raises(reader.MessageChangedError):
        reader.read(rewritten, 3, 10, first.digest)


def test_nothing_past_the_limit_is_read(tmp_path: Path) -> None:
    """A message appended while a scan runs is wholly in the next scan, never half-read (§5.1)."""
    path = write_mbox(tmp_path / "box", messages(2))
    limit = path.stat().st_size
    append_mbox(path, messages(2, start=2))
    assert len(list(reader.scan(path, limit))) == 2
    assert len(scanned(path)) == 4


def test_the_message_id_is_normalized(tmp_path: Path) -> None:
    path = write_mbox(
        tmp_path / "box",
        [message(1, message_id="<Mixed.Case+tag@example.org>"), message(2, with_id=False)],
    )
    first, second = scanned(path)
    assert first.message_id == "Mixed.Case+tag@example.org"
    assert second.message_id is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (SEPARATOR + b"\nSubject: x\n\nbody\n", True),
        (b"From - Thu Jan 01 00:00:00 2026\nX-Mozilla-Status: 0001\n\nbody\n", True),
        (b"From the desk of nobody\nThanks:\nAll of it.\n", False),
        (b"From alice\nno header here\n", False),
        (b'// <!-- <mdb:mork:z v="1.4"/> -->\n', False),
        (b"", False),
        (b"Subject: starts with a header\n\nbody\n", False),
    ],
)
def test_a_file_is_an_mbox_by_its_first_bytes(
    tmp_path: Path, content: bytes, expected: bool
) -> None:
    path = tmp_path / "file"
    path.write_bytes(content)
    assert reader.is_mbox(path) is expected


def test_a_very_long_line_is_read_in_fragments_and_cannot_start_a_message(tmp_path: Path) -> None:
    long_line = b"A" * (reader.MAX_LINE + 100)
    fake = b"From sender@example.org Tue Feb  3 09:14:00 2026\nSubject: fake\n\n"
    path = tmp_path / "box"
    path.write_bytes(
        SEPARATOR + b"\nSubject: one\n\n" + long_line[:50] + b"\n" + long_line + fake + b"tail\n"
    )
    # The fake separator sits in the middle of a long line's second fragment, not at a line start.
    assert len(scanned(path)) == 1
