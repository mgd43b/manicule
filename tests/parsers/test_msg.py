"""Outlook ``.msg``: a MAPI property reader, and a shim onto the ``.eml`` parser.

Two things are under test and they are different in kind. One is the reader — compound file,
two string spellings, a binary property that catches people, fixed-width properties that are not
where the variable-width ones are. The other is that the shim really is a shim: what comes out
is a message, read by the parser that reads messages, so anchors and chunking are identical
between the two formats rather than merely similar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.core.anchors import LineAnchor
from manicule.core.errors import ParseError
from manicule.parsers.config import MSG_MEDIA_TYPE, MailConfig, MsgConfig
from manicule.parsers.expansion import ExpandedMember
from manicule.parsers.mail import MailParser
from manicule.parsers.msg import MsgParser
from tests.parsers.support import check_corpus, check_fixture, raw_from, raw_of

pytestmark = pytest.mark.anyio


@pytest.fixture
def parser() -> MsgParser:
    return MsgParser(MsgConfig())


@pytest.fixture
def messages(corpus: Path) -> Path:
    return corpus / "msg"


async def blocks(parser: MsgParser, path: Path):  # noqa: ANN201 - list[ParsedBlock], inferred
    return [block async for block in parser.parse(raw_from(path, MSG_MEDIA_TYPE))]


# --- the reader ------------------------------------------------------------------------------


async def test_a_transported_message_keeps_the_headers_it_arrived_with(
    parser: MsgParser, messages: Path
) -> None:
    """``PidTagTransportMessageHeaders`` is the real header block, and is used when it exists."""
    found = await blocks(parser, messages / "typical.msg")

    assert "ana@example.test" in found[0].text
    assert "ben@example.test" in found[0].text
    assert "Tue, 03 Feb 2026 09:14:00 +0000" in found[0].text
    assert any("netted down" in block.text for block in found)


async def test_the_mime_headers_of_the_original_are_not_carried_across(
    parser: MsgParser, messages: Path
) -> None:
    """The body is rebuilt from MAPI properties, so a header describing the old one is false.

    ``Content-Transfer-Encoding: 7bit`` is in the transport block of the fixture precisely so
    that carrying the whole block wholesale is visible here rather than as mojibake later.
    """
    found = await blocks(parser, messages / "typical.msg")

    assert "Content-Transfer-Encoding" not in found[0].text
    assert "Content-Type" not in found[0].text


async def test_a_draft_has_its_headers_synthesized_from_the_recipient_table(
    parser: MsgParser, messages: Path
) -> None:
    """The trap, and it is most of a real export rather than an edge case.

    Nothing in Sent Items ever traversed a transport, so there is no header block at all and
    the addresses exist only as sender properties and a recipient table.
    """
    found = await blocks(parser, messages / "synthesized-headers.msg")

    assert "ana@example.test" in found[0].text
    assert "platform@example.test" in found[0].text
    assert "Draft: the retry budget" in found[0].text


async def test_a_recipient_type_is_read_from_the_properties_stream(
    parser: MsgParser, messages: Path
) -> None:
    """``PidTagRecipientType`` is ``PT_LONG``, so it is not beside the strings.

    A reader looking for ``__substg1.0_0C150003`` finds nothing, defaults, and silently promotes
    every ``Cc`` to a ``To`` — which is a message asserting who it was addressed to, wrongly,
    with nothing anywhere to notice.
    """
    found = await blocks(parser, messages / "synthesized-headers.msg")

    headers = found[0].text
    assert "To: Platform <platform@example.test>" in headers
    assert "Cc: Ben Ito <ben@example.test>" in headers


async def test_an_html_body_is_found_under_its_binary_property(
    parser: MsgParser, messages: Path
) -> None:
    """``PidTagHtml`` is ``PT_BINARY``, so the stream is ``…10130102``.

    A reader that assumed ``…1013001F`` by analogy with the plain body finds nothing and
    concludes the message has no HTML part — which reads as a message with no body at all.
    """
    found = await blocks(parser, messages / "html-only.msg")

    assert any("Where the retry budget went" in block.text for block in found)
    assert not any("<h1>" in block.text for block in found)


async def test_the_eight_bit_spelling_of_a_string_property_is_read(
    parser: MsgParser, messages: Path
) -> None:
    """A store without ``STORE_UNICODE_OK`` writes ``001E``, and both spellings exist.

    Read rather than deduced from the flag: only one of the two stream names is present, so
    looking for both is determinate — and a file that disagrees with its own flag still reads.
    """
    found = await blocks(parser, messages / "ansi.msg")

    assert "ana@example.test" in found[0].text
    assert any("netted down" in block.text for block in found)


async def test_a_file_that_is_not_a_compound_file_is_declined(parser: MsgParser) -> None:
    """Declining hands it to the next parser; reading it as empty looks like a success."""
    with pytest.raises(ParseError, match="not a compound file"):
        [block async for block in parser.parse(raw_of(b"Subject: nope\n\nhi\n", MSG_MEDIA_TYPE))]


async def test_a_property_past_its_ceiling_is_refused(messages: Path) -> None:
    """A ``.msg`` is a file from the corpus, so its properties are untrusted input."""
    tight = MsgParser(MsgConfig(max_property_bytes=8))

    with pytest.raises(ParseError, match="above the 8-byte ceiling"):
        [block async for block in tight.parse(raw_from(messages / "typical.msg", MSG_MEDIA_TYPE))]


# --- the shim --------------------------------------------------------------------------------


async def test_an_attachment_becomes_a_member_through_the_email_parser(
    parser: MsgParser, messages: Path
) -> None:
    """Reconstituted as a MIME part, so one code path serves both formats.

    Identity, depth budget and refusals are then the ``.eml`` parser's, which is the whole
    argument for the shim: two implementations of container membership would be two answers.
    """
    raw = raw_from(messages / "typical.msg", MSG_MEDIA_TYPE)

    members = [member async for member in parser.expand(raw)]

    assert [member.source_id for member in members] == ["mail:typical.msg!/notes.txt"]
    assert [member.depth for member in members] == [1]


async def test_an_attachment_keeps_the_media_type_the_message_declared(
    parser: MsgParser, messages: Path
) -> None:
    """``PidTagAttachMimeTag`` is what the sender said the bytes were, and it is worth reading.

    The mail parser infers a member's type from its filename only when the part declares none,
    so an attachment with no extension reaches the parser chain as generic bytes and is routed
    nowhere — while the message itself said, in a property this reader had declared and never
    looked at, exactly what it was.
    """
    raw = raw_from(messages / "mime-tagged.msg", MSG_MEDIA_TYPE)

    members = [member async for member in parser.expand(raw)]

    expanded = [member for member in members if isinstance(member, ExpandedMember)]
    assert [member.raw.media_type for member in expanded] == [
        "text/csv",
        "application/octet-stream",
    ], (
        "the well-formed tag is honored and the one carrying a header injection is refused "
        "back to the default rather than passed into a header verbatim"
    )


async def test_a_property_past_its_ceiling_is_refused_without_being_read_whole(
    messages: Path,
) -> None:
    """The ceiling bounds the allocation, not just the verdict.

    Reading a stream whole and measuring afterwards spends exactly the memory the limit exists
    to protect — which would make these settings a report on the allocation rather than a bound
    on it. One byte past the limit is enough to refuse and is all that is taken.
    """
    tight = MsgParser(MsgConfig(max_body_bytes=8))

    with pytest.raises(ParseError, match="above the 8-byte ceiling"):
        [block async for block in tight.parse(raw_from(messages / "typical.msg", MSG_MEDIA_TYPE))]


async def test_the_blocks_are_the_ones_the_email_parser_would_have_produced(
    parser: MsgParser, messages: Path
) -> None:
    """The claim the shim exists to make, stated as an executable comparison.

    Anchors and chunking are identical between ``.eml`` and ``.msg`` *by construction* rather
    than by two implementations agreeing — so the way to check it is to reconstitute the
    message, hand it to the email parser directly, and compare.
    """
    from manicule.parsers.msg import (  # noqa: PLC0415 - the seam under test
        _as_rfc5322,  # pyright: ignore[reportPrivateUsage] - the conversion is what is claimed
    )

    raw = raw_from(messages / "typical.msg", MSG_MEDIA_TYPE)
    converted = raw_of(_as_rfc5322(raw, config=MsgConfig()), "message/rfc822", uri=raw.uri)

    mine = await blocks(parser, messages / "typical.msg")
    theirs = [block async for block in MailParser(MailConfig()).parse(converted)]

    assert [(block.text, block.anchor) for block in mine] == [
        (block.text, block.anchor) for block in theirs
    ]


async def test_an_anchor_resolves_against_the_message_the_file_describes(
    parser: MsgParser, messages: Path
) -> None:
    """Deterministic re-conversion is what makes a line anchor into a ``.msg`` exact."""
    raw = raw_from(messages / "typical.msg", MSG_MEDIA_TYPE)
    found = await blocks(parser, messages / "typical.msg")
    located = next(block for block in found if isinstance(block.anchor, LineAnchor))

    resolved = await parser.resolve(located.anchor, raw)

    assert resolved is not None
    assert located.text in resolved


# --- the shipped obligations -------------------------------------------------------------


async def test_every_fixture_round_trips(parser: MsgParser, messages: Path) -> None:
    readable = ("typical.msg", "synthesized-headers.msg", "html-only.msg", "ansi.msg")

    await check_corpus(
        parser,
        [raw_from(messages / name, MSG_MEDIA_TYPE) for name in readable],
        min_blocks=8,
    )


async def test_the_parser_contract_holds_for_a_synthesized_message(
    parser: MsgParser, messages: Path
) -> None:
    await check_fixture(parser, raw_from(messages / "synthesized-headers.msg", MSG_MEDIA_TYPE))
