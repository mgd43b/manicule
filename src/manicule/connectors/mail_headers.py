"""What a mail connector reads out of a message it has just fetched, and what it cites.

Shared by the Gmail and Outlook connectors, which both fetch a whole RFC 5322 message and need
the same two things from its front: the ``Subject`` that titles the document and the
``Message-ID`` that links it and lets another mail source recognize the same message
(:mod:`manicule.core.mail_ids`). Both are read from the headers only, and neither is ever logged.
"""

from __future__ import annotations

import email.errors
import email.parser
import email.policy
import re
from datetime import datetime
from typing import Final

from pydantic import JsonValue

from manicule.core.provenance import Provenance, SourceMetadata

__all__ = ["message_provenance", "read_headers"]

_HEADER_END: Final = re.compile(rb"\r?\n\r?\n")
_HEADER_LIMIT: Final = 1 << 20
"""Headers are read from the front of the message and never past this, so a 25 MB message with a
huge attachment costs a header's worth of parsing, not a second copy of the bytes."""


def read_headers(raw: bytes) -> tuple[str, str | None]:
    """The ``Subject`` and the ``Message-ID`` (without its angle brackets) of a message.

    An unreadable or absent header is an empty subject or no id, never an error: the message is
    still indexed.
    """
    end = _HEADER_END.search(raw, 0, _HEADER_LIMIT)
    head = raw[: end.start()] if end is not None else raw[:_HEADER_LIMIT]
    try:
        headers = email.parser.BytesHeaderParser(policy=email.policy.default).parsebytes(head)
        subject = headers.get("Subject")
        message_id = headers.get("Message-ID")
        subject_text = " ".join(str(subject).split()) if subject is not None else ""
        cleaned = (
            str(message_id).strip().removeprefix("<").removesuffix(">").strip()
            if message_id is not None
            else ""
        )
    except (ValueError, TypeError, IndexError, email.errors.MessageError):
        return "", None
    return subject_text, cleaned or None


def message_provenance(
    *,
    source_id: str,
    subject: str,
    link: str,
    version: str,
    media_type: str,
    arrived: datetime | None,
) -> JsonValue:
    """What a citation says about the message: its subject, its link, when it arrived.

    A subject delivered with a control character cannot be a title. The message is still cited by
    its link, and the record keeps that link; only if the link too is refused does it say, in
    words that do not quote the subject, that it could not be described.
    """
    for title in (subject, ""):
        try:
            source = SourceMetadata(
                title=title,
                canonical_uri=link,
                source_id=source_id,
                version=version,
                created_at=arrived,
                content_type=media_type,
            )
        except ValueError:
            continue
        return Provenance(source=source).as_metadata_value()
    return Provenance(
        unavailable_reason="the message could not be described by a title and a link"
    ).as_metadata_value()
