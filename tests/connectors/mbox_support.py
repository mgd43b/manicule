"""Synthetic mbox archives: every dialect, large files, duplicates, missing ids, attachments.

Nothing here is real mail or resembles it. Every address is at ``example.org``, every body is
generated, and a message's text is derived from its number so a failure names the message by what
it said. ``docs/connectors/mbox.md`` §8 lists what each variant exists to exercise.

Messages come from :func:`tests.connectors.gmail_support.mail` (the standard library's own
writer), so the connector is tested against messages a real mail program could have written, and
this module only decides how they are *filed*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tests.connectors.gmail_support import mail

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

Dialect = Literal["mboxo", "mboxrd", "mboxcl"]
"""``mboxo`` escapes ``From `` and cannot be undone; ``mboxrd`` escapes ``>*From `` and can;
``mboxcl`` writes ``Content-Length`` and escapes nothing."""

SEPARATOR = b"From sender@example.org Tue Feb  3 09:14:00 2026"


def message(
    number: int,
    *,
    body: str | None = None,
    with_id: bool = True,
    message_id: str | None = None,
    attachment: bool = False,
) -> bytes:
    """Message ``number``: a subject and a body that say which it is, with a ``Message-ID``."""
    return mail(
        f"Archived message {number}",
        body if body is not None else f"Body of archived message {number}, about retry budgets.",
        message_id=(message_id or f"<archive.{number}@example.org>") if with_id else None,
        with_id=with_id,
        attachment=attachment,
    )


def with_headers(raw: bytes, *headers: str) -> bytes:
    """``raw`` with extra header lines added at the top, as a mail client does to its copy."""
    return "".join(f"{header}\r\n" for header in headers).encode() + raw


def messages(
    count: int, *, start: int = 0, with_id: bool = True, attachment: bool = False
) -> list[bytes]:
    return [
        message(number, with_id=with_id, attachment=attachment)
        for number in range(start, start + count)
    ]


@dataclass(frozen=True, slots=True)
class Filing:
    """How a file is written: its dialect, its line endings and its separator line."""

    dialect: Dialect = "mboxrd"
    eol: bytes = b"\n"
    separator: bytes = SEPARATOR


_FROM = re.compile(rb"^(>*)From ")


def _lines(raw: bytes) -> list[bytes]:
    """A message's lines without their terminators, whatever terminator it was written with."""
    text = raw.replace(b"\r\n", b"\n")
    return text.split(b"\n")[: -1 if text.endswith(b"\n") else None]


def encode(raw: bytes, filing: Filing = Filing()) -> bytes:  # noqa: B008 - frozen
    """One message as an mbox writer of this dialect would file it."""
    lines = _lines(raw)
    if filing.dialect == "mboxo":
        lines = [b">" + line if line.startswith(b"From ") else line for line in lines]
    elif filing.dialect == "mboxrd":
        lines = [b">" + line if _FROM.match(line) else line for line in lines]
    elif filing.dialect == "mboxcl":
        divider = lines.index(b"")
        body = filing.eol.join(lines[divider + 1 :]) + filing.eol
        lines = [*lines[:divider], f"Content-Length: {len(body)}".encode(), *lines[divider:]]
    return filing.separator + filing.eol + filing.eol.join(lines) + filing.eol + filing.eol


def write_mbox(
    path: Path,
    raws: Iterable[bytes],
    filing: Filing = Filing(),  # noqa: B008 - frozen
) -> Path:
    """Write ``raws`` to ``path`` as one mbox file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for raw in raws:
            handle.write(encode(raw, filing))
    return path


def append_mbox(path: Path, raws: Iterable[bytes], filing: Filing = Filing()) -> None:  # noqa: B008
    with path.open("ab") as handle:
        for raw in raws:
            handle.write(encode(raw, filing))


def padded(number: int, size: int) -> bytes:
    """Message ``number`` with a body of roughly ``size`` bytes, for files that must be large."""
    line = f"Padding line of archived message {number} for the size of the file.\n"
    count = max(1, size // len(line))
    return message(number, body=line * count)


def archive(
    root: Path,
    files: Sequence[tuple[str, Iterable[bytes], Filing]],
) -> list[Path]:
    """Several files under ``root`` at once; each entry is a relative name, messages, a filing."""
    return [write_mbox(root / name, raws, filing) for name, raws, filing in files]


def thunderbird(root: Path) -> Path:
    """A Thunderbird-shaped profile: extensionless folders, ``.msf`` caches and ``.sbd`` subfolders.

    Returns the root. ``Inbox`` holds two messages, an expunged one and the folder's placeholder;
    ``Inbox.sbd/Receipts`` holds one; the ``.msf`` files are summary caches that must never be
    read as mail.
    """
    placeholder = with_headers(
        message(900, body="Do not delete."),
        "X-Mozilla-Status: 0001",
        "X-Mozilla-Status2: 00800000",
    ).replace(b"Archived message 900", b"DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA")
    write_mbox(
        root / "Inbox",
        [
            placeholder,
            with_headers(message(1), "X-Mozilla-Status: 0001"),
            with_headers(message(2), "X-Mozilla-Status: 0009"),
            message(3),
        ],
        Filing(separator=b"From - Tue Feb 03 09:14:00 2026", eol=b"\r\n"),
    )
    write_mbox(root / "Inbox.sbd" / "Receipts", [message(4)], Filing(eol=b"\r\n"))
    (root / "Inbox.msf").write_bytes(b'// <!-- <mdb:mork:z v="1.4"/> -->\n')
    (root / "Inbox.sbd" / "Receipts.msf").write_bytes(b'// <!-- <mdb:mork:z v="1.4"/> -->\n')
    return root


def apple_export(root: Path, raws: Iterable[bytes]) -> Path:
    """An Apple Mail "Export Mailbox": a ``.mbox`` directory holding a file named ``mbox``."""
    return write_mbox(root / "Exported.mbox" / "mbox", raws)


def not_mbox(root: Path) -> None:
    """Things beside an archive that are not mbox files and must not be mistaken for one."""
    (root / "notes.txt").write_text("From the desk of nobody\nThanks:\nAll of it.\n")
    (root / "empty").write_bytes(b"")
    (root / "maildir" / "cur").mkdir(parents=True)
    (root / "maildir" / "cur" / "1700000000.M1P1.host,S=900:2,S").write_bytes(
        message(950).replace(b"\r\n", b"\n")
    )
    (root / "library" / "Messages").mkdir(parents=True)
    (root / "library" / "Messages" / "7.emlx").write_bytes(
        b"1234\n" + message(951) + b'<?xml version="1.0"?><plist version="1.0"></plist>\n'
    )


def scan_all(path: Path):  # noqa: ANN201
    """Every live message in a file as the reader finds it, for tests that need positions."""
    from manicule.connectors import mbox_reader  # noqa: PLC0415

    return mbox_reader.scan(path, path.stat().st_size)


def reader_digests(path: Path) -> list[str]:
    """The identities of the messages in a file, in file order."""
    return [located.digest for located in scan_all(path)]
