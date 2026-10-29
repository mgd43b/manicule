"""The ``Message-ID`` of a mail document, however its source recorded it.

Two mail connectors can hold one message, and the only thing they share is the header the sender
wrote. This is the one place that header is normalized and the one place it is read back out of
a stored document, so the two connectors cannot disagree about whether two ids are equal
(``docs/connectors/mbox.md`` §6).

**The id is a comparison key and nothing more.** It is never an identity (``mbox.md`` §4): the
sender chooses it, it may be absent, and every copy of a message shares it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "MAIL_MESSAGE_ID_KEY",
    "message_id_from_link",
    "message_id_of",
    "normalize_message_id",
]

MAIL_MESSAGE_ID_KEY: Final = "mail_message_id"
"""The ``documents.metadata`` key a mail document records its normalized ``Message-ID`` under."""

MAX_MESSAGE_ID_LENGTH: Final = 512
"""A longer value is not a message id; RFC 5322 limits a line to 998 characters, id included."""

_GMAIL_SEARCH: Final = "#search/rfc822msgid:"
"""How ``gmail.md`` §2 links a message that has a ``Message-ID``."""


def normalize_message_id(value: str) -> str | None:
    """The id inside the angle brackets, or the trimmed text when there are none.

    ``None`` for an empty value, one that is implausibly long, or one with a control character,
    so a header that is not an id is no id rather than a key that matches nothing but itself.
    """
    text = value.strip()
    start = text.find("<")
    if start != -1:
        end = text.find(">", start + 1)
        text = text[start + 1 : end] if end != -1 else text[start + 1 :]
    text = text.strip()
    if not text or len(text) > MAX_MESSAGE_ID_LENGTH:
        return None
    if any(not character.isprintable() for character in text):
        return None
    return text


def message_id_from_link(link: str) -> str | None:
    """The ``Message-ID`` a Gmail message's link carries, or ``None`` for any other address.

    A Gmail message with a ``Message-ID`` is linked by the search for it, percent-encoded; one
    without is linked by its Gmail id (``#all/…``), which carries nothing to compare.
    """
    start = link.find(_GMAIL_SEARCH)
    if start == -1:
        return None
    return normalize_message_id(unquote(link[start + len(_GMAIL_SEARCH) :]))


def message_id_of(metadata: Mapping[str, object], canonical_uri: str = "") -> str | None:
    """The id recorded in a mail document's metadata, else the one its Gmail link carries."""
    recorded = metadata.get(MAIL_MESSAGE_ID_KEY)
    if isinstance(recorded, str):
        return normalize_message_id(recorded)
    return message_id_from_link(canonical_uri) if canonical_uri else None
