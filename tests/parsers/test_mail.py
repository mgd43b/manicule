"""The ``.eml`` parser: one canonical text, one part-selection rule, pinned conversion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from email.message import EmailMessage
from pathlib import Path
from typing import override

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, DocumentStatus, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.ids import content_hash
from manicule.core.protocols import aclose, read_blocks
from manicule.parsers.config import html_text_version
from manicule.parsers.expansion import (
    ExpandedMember,
    MemberFailure,
    SupportsExpansion,
    aclose_members,
    read_members,
)
from manicule.parsers.mail import MailConfig, MailParser
from manicule.testing import assert_parser_contract, assert_round_trip
from tests.corpus.mail import ATTACHMENT_TEXT
from tests.parsers.support import check_corpus, document_for, raw_from, raw_of

READABLE = (
    "typical.eml",
    "multipart.eml",
    "html-only.eml",
    "headers-only.eml",
    "duplicate-attachments.eml",
)


@pytest.fixture
def parser() -> MailParser:
    return MailParser(MailConfig())


def _raw(corpus: Path, name: str) -> RawDocument:
    return raw_from(corpus / "mail" / name, "message/rfc822")


async def _blocks(parser: MailParser, raw: RawDocument) -> list[ParsedBlock]:
    """Drained through ``read_blocks``, which closes the stream in a ``finally``."""
    return await read_blocks(parser, raw)


def _line_start(block: ParsedBlock) -> int:
    assert isinstance(block.anchor, LineAnchor), "a mail block is addressed by lines"
    return block.anchor.start


def _html_message(html: str) -> RawDocument:
    """An HTML-only message built here, so two bodies can differ by one element."""
    message = EmailMessage()
    message["From"] = "sender@example.invalid"
    message["To"] = "recipient@example.invalid"
    message["Subject"] = "Where it went, in one message"
    message.set_content(html, subtype="html")
    return raw_of(message.as_bytes(), "message/rfc822", uri="built.eml")


async def test_every_readable_fixture_round_trips_within_its_declared_location_budget(
    parser: MailParser, corpus: Path, chunker: StructuralChunker
) -> None:
    """Every block cites lines of the canonical text that contain it.

    Both body kinds are in the corpus, which matters: an HTML-only body's line numbers address
    the *converted* text, so this is the assertion that would fail if :meth:`MailParser.resolve`
    reconstructed it any differently from :meth:`MailParser.parse`.
    """
    reports = await check_corpus(
        parser, [_raw(corpus, name) for name in READABLE], chunker=chunker, min_blocks=15
    )
    assert sum(report.unlocated for report in reports) == 0


async def test_the_headers_are_one_block_before_the_body_with_its_own_span(
    parser: MailParser, corpus: Path
) -> None:
    """Five header fields, rendered in a fixed order, addressed as lines one to five.

    A fixed order rather than the source's: the source's order is a property of whichever mail
    transfer agent touched the message last, and every anchor in the message is measured from
    the height of this block.
    """
    blocks = await _blocks(parser, _raw(corpus, "typical.eml"))
    assert blocks[0].anchor == LineAnchor(start=1, end=5)
    assert blocks[0].text.splitlines()[0].startswith("From: ")
    assert blocks[0].text.splitlines()[-1].startswith("Subject: ")


async def test_the_subject_is_the_title_and_the_first_element_of_the_heading_path(
    parser: MailParser, corpus: Path
) -> None:
    """It is the only structure an email has, so it has to serve as both.

    Without it a chunk of a reply embeds with no breadcrumb at all, and the thread it belongs
    to is unrecoverable from the chunk.
    """
    blocks = await _blocks(parser, _raw(corpus, "typical.eml"))
    assert blocks[0].metadata["title"] == "Re: watermark on interrupted syncs"
    assert all(block.heading_path == ("Re: watermark on interrupted syncs",) for block in blocks)


async def test_a_quoted_reply_chain_is_kept_rather_than_trimmed(
    parser: MailParser, corpus: Path
) -> None:
    """Trimming is a retrieval optimisation with a real cost, and not a parsing decision.

    The quoted passage is frequently the only statement of the thing being replied to, so a
    parser that dropped it would index an answer whose question exists nowhere in the corpus.
    """
    blocks = await _blocks(parser, _raw(corpus, "typical.eml"))
    quoted = [block for block in blocks if block.text.startswith(">")]
    assert quoted
    assert "If it is the start of the run" in quoted[0].text


async def test_the_canonical_body_is_the_first_plain_part_and_not_the_richer_one(
    parser: MailParser, corpus: Path
) -> None:
    """The rule that makes a multipart message deterministic (§10).

    ``multipart.eml`` carries a plain body, an HTML alternative and two attachments. Preferring
    the richer part would anchor into converted HTML while a perfectly good plain body sat
    above it, and "richer" is not a rule anyone can reproduce.
    """
    blocks = await _blocks(parser, _raw(corpus, "multipart.eml"))
    assert blocks[0].metadata["body_media_type"] == "text/plain"
    assert blocks[1].text == "The checklist and the counts are attached. Nothing else has changed."


async def test_an_html_only_body_is_converted_and_records_the_version_that_converted_it(
    parser: MailParser, corpus: Path
) -> None:
    """Line numbers address the converted text, so the conversion is part of the anchor.

    A converter upgrade shifts every anchor in every HTML email — round-tripping today and
    pointing at the wrong paragraph after a dependency bump, with no test failing in between.
    Recording the version is what turns that into a fingerprint mismatch instead.
    """
    blocks = await _blocks(parser, _raw(corpus, "html-only.eml"))
    assert blocks[0].metadata["body_media_type"] == "text/html"
    assert blocks[0].metadata["html_to_text_version"] == html_text_version()
    assert "selectolax/" in html_text_version()
    assert any(block.text == "Where the first quarter went" for block in blocks)


async def test_resolving_an_html_body_anchor_applies_the_same_pinned_conversion(
    parser: MailParser, corpus: Path
) -> None:
    """``resolve`` recomposes the canonical text rather than reading the source bytes.

    If it read the HTML source instead, the line numbers would address markup and every
    citation into an HTML email would quote a tag.
    """
    raw = _raw(corpus, "html-only.eml")
    blocks = await _blocks(parser, raw)
    body = next(block for block in blocks if "reconciler" in block.text)
    assert await parser.resolve(body.anchor, raw) == body.text


async def test_an_inline_break_in_an_html_body_moves_every_line_after_it(
    parser: MailParser,
) -> None:
    """Why ``PARSERS["email"].rules`` moved for a change this parser did not make.

    An HTML-only body's line numbers address the text ``_html_to_text`` builds from the web
    parser's blocks, and a ``<br>`` now puts a newline *inside* one of those blocks. So the
    break becomes a line of the canonical text and every anchor below it shifts by one — which
    is a change to what every citation into that message resolves to, with nothing in this
    module changed at all.

    Run rather than reasoned about: the same message is built twice, once with the break and
    once without, and the two anchor sequences are compared.
    """
    body = (
        "<html><body><h1>Where it went</h1>"
        "<p>first clause{divider}second clause</p>"
        "<p>a paragraph below the break</p></body></html>"
    )
    without = await _blocks(parser, _html_message(body.format(divider=" ")))
    with_break = await _blocks(parser, _html_message(body.format(divider="<br/>")))

    assert without[-1].text == "a paragraph below the break"
    assert with_break[-1].text == without[-1].text, "the same paragraph, in both messages"
    assert without[2].text == "first clause second clause"
    assert with_break[2].text == "first clause\nsecond clause"

    moved = _line_start(with_break[-1]) - _line_start(without[-1])
    assert moved == 1, (
        "one line further down, because the break is now a line of the canonical text. "
        "Asserted as the difference rather than as two absolute line numbers, so a change "
        "to which headers this parser renders fails where it happened rather than here"
    )
    assert without[-1].anchor == LineAnchor(start=9, end=9), "and this is where it was"


async def test_resolving_across_an_inline_break_returns_the_lines_it_claims(
    parser: MailParser, corpus: Path
) -> None:
    """``resolve`` recomposes the canonical text by the same rules, break included.

    If the two paths disagreed about what a break contributes, an anchor would resolve to the
    line above or below the one it names — a citation that reads correctly and is wrong.
    """
    raw = _raw(corpus, "html-only.eml")
    blocks = await _blocks(parser, raw)
    # Named rather than found by "has a newline in it": the header block's lines are its
    # fields, and the list block's are its items, so neither is what this is about.
    broken = next(block for block in blocks if block.text.startswith("The reconciler"))

    assert broken.text.splitlines() == [
        "The reconciler now runs on its own schedule,",
        "which removed the long tail of deletions that used to arrive a day late.",
    ]
    assert await parser.resolve(broken.anchor, raw) == broken.text


async def test_a_message_with_headers_and_no_body_yields_the_header_block_alone(
    parser: MailParser, corpus: Path
) -> None:
    """A calendar decline and a bounce stub are both this shape.

    One block rather than none: the headers are real content, and reporting the document as
    having no extractable text would hide a message that says who sent what to whom.
    """
    blocks = await _blocks(parser, _raw(corpus, "headers-only.eml"))
    assert len(blocks) == 1
    assert blocks[0].anchor == LineAnchor(start=1, end=5)


async def test_something_that_is_not_a_message_is_declined_rather_than_failed(
    parser: MailParser, corpus: Path
) -> None:
    """A note somebody renamed to ``.eml`` is text, and the chain should get to say so.

    Declining leaves the document able to end as ``unsupported_media_type`` or to be indexed
    by the plaintext parser; failing would record a breakage that did not happen.
    """
    with pytest.raises(ParseError, match="declining"):
        await _blocks(parser, _raw(corpus, "not-a-message.eml"))


async def test_a_body_that_lies_about_its_character_set_fails_rather_than_guessing(
    parser: MailParser, corpus: Path
) -> None:
    """Guessing another encoding puts mojibake into the index under a citation.

    The message declares UTF-8 and carries Latin-1, which is what a mail client that mislabels
    its output produces. Failing names the header to fix; a second guess would index text no
    message contains.
    """
    with pytest.raises(ParseError, match="does not decode"):
        await _blocks(parser, _raw(corpus, "broken-charset.eml"))


async def test_zero_bytes_is_declined_rather_than_read_as_an_empty_message(
    parser: MailParser, corpus: Path
) -> None:
    """An empty file has no header fields, so it is not a message.

    Declining leaves the chain able to reach ``unsupported_media_type``; returning no blocks
    would claim the tooling read a message and found nothing in it.
    """
    with pytest.raises(ParseError, match="declining"):
        await _blocks(parser, _raw(corpus, "empty.eml"))


# --- attachments ---------------------------------------------------------------------------


async def test_the_parser_satisfies_the_expansion_protocol(parser: MailParser) -> None:
    """Attachments reach the pipeline the same way archive members do.

    One protocol rather than two, so a PDF is a PDF wherever it arrived from and the depth,
    cycle and budget bookkeeping is the same code in both cases.
    """
    assert isinstance(parser, SupportsExpansion)


async def test_each_attachment_becomes_a_document_addressed_by_its_own_filename(
    parser: MailParser, corpus: Path
) -> None:
    """Identity comes from the name, never from the position in the message.

    An attachment added ahead of another must not inherit its identity, which is exactly what
    an ordinal would do — silently, on the next sync, to every citation into the message.
    """
    members = await read_members(parser, _raw(corpus, "multipart.eml"))
    assert all(isinstance(member, ExpandedMember) for member in members)
    assert [member.uri for member in members] == [
        "mail:multipart.eml!/checklist.txt",
        "mail:multipart.eml!/counts.json",
    ]
    assert [member.depth for member in members] == [1, 1]


async def test_an_attachment_keeps_the_media_type_the_message_declared_for_it(
    parser: MailParser, corpus: Path
) -> None:
    """A type the source states beats a filename extension (§6.1), and both beat a guess."""
    members = await read_members(parser, _raw(corpus, "multipart.eml"))
    types = [member.raw.media_type for member in members if isinstance(member, ExpandedMember)]
    assert types == ["text/plain", "application/json"]


async def test_two_attachments_with_one_name_get_two_addresses(
    parser: MailParser, corpus: Path
) -> None:
    """A message publishes no second name for them, so the part path supplies one.

    Positional, and used only where the alternative is two documents sharing one identity —
    which reconciliation would resolve by deleting one of them.
    """
    raw = _raw(corpus, "duplicate-attachments.eml")
    members = await read_members(parser, raw)
    addresses = [member.uri for member in members]
    assert len(addresses) == len(set(addresses)) == 2


async def test_an_enclosed_message_with_no_filename_is_still_expanded(
    parser: MailParser, corpus: Path
) -> None:
    """A ``multipart/digest``, which is where "attachment" stops meaning what it looks like.

    Its parts are bare ``message/rfc822`` with no disposition and no filename, so a check for
    either finds nothing. Before this, such a part satisfied no branch anywhere — not the
    body, because it is not ``text/*``; not the walk, because it is not ``multipart``; not an
    attachment — and a digest of twenty messages indexed as a header block with every enclosed
    message absent and **no failure reported**. That is the omission ``expansion.py`` says
    must never happen, and the only reason it was ever visible is that somebody counted.
    """
    raw = _raw(corpus, "digest.eml")

    members = [
        member for member in await read_members(parser, raw) if isinstance(member, ExpandedMember)
    ]

    assert len(members) == 2, "a two-message digest must expand into two documents"
    assert all(member.raw.media_type == "message/rfc822" for member in members)
    recovered = b"".join(member.raw.as_bytes() for member in members)
    assert b"The retention window moves to ninety days" in recovered
    assert b"The reconciler now runs on its own schedule" in recovered


async def test_the_enclosed_messages_of_a_digest_are_not_also_indexed_as_body(
    parser: MailParser, corpus: Path
) -> None:
    """Expanded once, not twice.

    The complement of the test above, and the reason the fix is "treat it as an attachment"
    rather than "walk into it": a part that becomes a member must not also contribute to the
    containing message's canonical text, or the same sentence is indexed under two documents
    and both cite it.
    """
    raw = _raw(corpus, "digest.eml")

    blocks = await read_blocks(parser, raw)
    text = "\n".join(block.text for block in blocks)

    assert "Weekly digest" in text, "the digest's own headers are still indexed"
    assert "The retention window moves to ninety days" not in text
    assert "The reconciler now runs on its own schedule" not in text


async def test_a_message_with_more_attachments_than_the_limit_stops_and_says_so(
    corpus: Path,
) -> None:
    """A message is a container, so it needs the ceiling every container needs.

    Mail expanded attachments with no member limit at all while the archive parser had counted
    them since it was written, so a message with a hundred thousand parts became a hundred
    thousand documents — and an archive containing a message containing an archive had no
    member ceiling on the middle hop. The limit is set to one here so the fixture corpus does
    not have to contain a hostile message to exercise it.
    """
    parser = MailParser(MailConfig(max_members=1))

    outcomes = await read_members(parser, _raw(corpus, "multipart.eml"))

    expanded = [member for member in outcomes if isinstance(member, ExpandedMember)]
    refused = [member for member in outcomes if isinstance(member, MemberFailure)]
    assert len(expanded) == 1, "the limit is one, so exactly one attachment may expand"
    assert refused, "the attachments past the limit must be reported, never dropped"
    assert "attachment count exceeded" in refused[0].reason
    assert "the limit is 1" in refused[0].reason


async def test_the_body_part_is_never_also_expanded_as_an_attachment(
    parser: MailParser, corpus: Path
) -> None:
    """Otherwise every message would be indexed twice, once as itself and once as a member."""
    members = await read_members(parser, _raw(corpus, "typical.eml"))
    assert members == []


async def test_disabling_expansion_reports_every_attachment_instead_of_dropping_it(
    corpus: Path,
) -> None:
    """A configuration that indexes less must not do it quietly.

    "The message had two attachments and the index has none" is not something anyone would
    otherwise discover, so the setting produces a visible outcome naming itself.
    """
    parser = MailParser(MailConfig(expand_attachments=False))
    members = await read_members(parser, _raw(corpus, "multipart.eml"))
    assert len(members) == 2
    for member in members:
        assert isinstance(member, MemberFailure)
        assert member.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE
        assert "expandAttachments" in member.reason


async def test_an_attachment_past_the_nesting_limit_is_reported_rather_than_dropped(
    corpus: Path,
) -> None:
    """Depth is counted from the top-level document and the boundary has to be visible.

    A member beyond it is stored with a reason, so somebody hitting the limit can find out
    that they did rather than wondering where a file went.
    """
    parser = MailParser(MailConfig(max_depth=2))
    raw = _raw(corpus, "multipart.eml")
    deep = raw.model_copy(update={"metadata": {**raw.metadata, "container_depth": 2}})
    members = await read_members(parser, deep)
    assert members
    for member in members:
        assert isinstance(member, MemberFailure)
        assert member.reason.startswith("archive nesting depth exceeded")


async def test_an_attachment_repeating_a_container_on_its_path_is_stopped(
    parser: MailParser, corpus: Path
) -> None:
    """Cycle detection by content hash, which costs nothing and terminates the recursion.

    Self-referential nesting by identical content is trivial to construct even though a
    message cannot literally contain itself.
    """
    raw = _raw(corpus, "multipart.eml")
    seeded = raw.model_copy(
        update={
            "metadata": {
                **raw.metadata,
                "container_path_hashes": [content_hash(ATTACHMENT_TEXT.encode("utf-8"))],
            }
        }
    )
    members = await read_members(parser, seeded)
    cycles = [member for member in members if isinstance(member, MemberFailure)]
    assert len(cycles) == 1
    assert "cycle detected" in cycles[0].reason


# --- the guard itself ------------------------------------------------------------------


async def test_an_anchor_one_line_out_is_caught_by_the_round_trip_check(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The guard is load-bearing, proved by breaking it.

    An email's canonical text is composed rather than read off disk, so an off-by-one here is
    a header-height mistake — the class of error that resolves to real, adjacent, wrong text.
    """
    raw = _raw(corpus, "typical.eml")
    with pytest.raises(AssertionError):
        await assert_round_trip(
            _OffByOneParser(MailConfig()),
            raw,
            fixture="off-by-one",
            chunker=chunker,
            document=document_for(raw),
        )


class _OffByOneParser(MailParser):
    """Anchors every block one line further down than the text it claims."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            anchor = block.anchor
            if not isinstance(anchor, LineAnchor):  # pragma: no cover - always a LineAnchor
                yield block
                continue
            moved = LineAnchor(start=anchor.start + 1, end=anchor.end + 1, symbol=anchor.symbol)
            yield block.model_copy(update={"anchor": moved})


async def test_a_block_stream_stopped_after_one_block_is_not_left_suspended(
    parser: MailParser, corpus: Path
) -> None:
    """A consumer that stops early must not strand the generator.

    An HTML-only body composes its canonical text through a second parser, so a message is the
    case where a stranded stream holds another stream open behind it.
    """
    stream = parser.parse(_raw(corpus, "html-only.eml"))
    async for _ in stream:
        break
    await aclose(stream)
    assert await _is_closed(stream)


async def test_an_attachment_stream_stopped_after_one_member_is_not_left_suspended(
    parser: MailParser, corpus: Path
) -> None:
    """``expand`` is an async generator too, and gets the same discipline as ``parse``.

    A consumer that takes the first attachment and stops is the ordinary shape of a pipeline
    with a member budget, so it is the path that must not strand anything.
    """
    stream = parser.expand(_raw(corpus, "multipart.eml"))
    async for _ in stream:
        break
    await aclose_members(stream)
    assert await _is_closed(stream)


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


async def test_a_cdata_body_in_an_html_email_is_recovered_and_moves_the_lines_after_it() -> None:
    """Recovering CDATA moves an HTML mail body's line numbers, because they address its blocks.

    **This is the coupling that nearly shipped a defect.** ``_html_to_text`` builds the canonical
    text an email ``LineAnchor`` addresses by joining the blocks :class:`WebParser` yields. The web
    parser now recovers CDATA sections rather than deleting them, so a recovered body becomes a
    block of its own and **every line after it shifts**.

    The HTML parser itself anchors structurally — a heading path and a published fragment — so the
    change is invisible there. Email is the parser that reuses it and anchors by *line*, and the
    only reason this is safe is that ``ParserVersions.rules["email"]`` was bumped alongside the
    html one: without it, a corpus would hold email anchors under the old numbering while newly
    ingested identical messages got the new, and nothing would say so.

    Asserted here rather than in the web parser's suite because the failure is the email parser's
    to have, and a test living beside the change would not have found it.
    """
    body = (
        "<h2>Retry policy</h2>"
        "<ac:plain-text-body><![CDATA[def retry(): ...]]></ac:plain-text-body>"
        "<p>Trailing paragraph.</p>"
    )
    message = (
        "From: a@example.test\r\nTo: b@example.test\r\nSubject: Retry\r\n"
        f"MIME-Version: 1.0\r\nContent-Type: text/html; charset=utf-8\r\n\r\n{body}"
    )
    parser = MailParser(MailConfig())
    raw = raw_of(message, "message/rfc822")
    blocks = await assert_parser_contract(parser, raw)
    texts = [block.text for block in blocks]

    assert "def retry(): ..." in texts, (
        "the recovered macro body is missing from the mail body, so an HTML-only email loses the "
        "same content the HTML parser used to lose"
    )
    # The recovered body occupies its own line, so the paragraph after it is no longer where it
    # would have been. `assert_parser_contract` above is what proves the anchors still resolve to
    # the text their blocks claim — which is the property that would break if the recovery moved
    # the lines without the numbering following.
    assert texts.index("Trailing paragraph.") > texts.index("def retry(): ...")


def test_the_email_parser_records_a_rules_version_that_moved_with_the_conversion() -> None:
    """The bump that re-parses existing emails, asserted as a fingerprint rather than a literal.

    Not ``rules == "2"``, which is a tautology. What matters is that the *fingerprint* an email
    document carries differs from the one a build before this change produced — because that
    difference is what selects those documents for a re-parse from retained bytes. Without it the
    corpus keeps line anchors that address text the parser no longer produces, behind a lineage
    claiming to be current.
    """
    from manicule.core.fingerprints import ParseFingerprint  # noqa: PLC0415
    from manicule.parsers.versions import parse_fingerprint  # noqa: PLC0415

    current = parse_fingerprint("email")
    assert current is not None
    before = ParseFingerprint(parser=current.parser, version="1", libraries=dict(current.libraries))

    assert not current.matches(before), (
        "the email parse fingerprint did not move, so documents parsed before the CDATA recovery "
        "will never be re-parsed and their line anchors address text that is no longer produced"
    )
    assert "version" in current.changed_fields(before)


async def test_an_html_body_is_built_from_block_text_alone() -> None:
    """Why ``PARSERS["email"].rules`` did not move when ``PARSERS["html"].rules`` went 3 -> 4.

    ``email``'s two previous bumps both followed the web parser, and a third would look like the
    pattern. It is not the pattern, and this is the property that decides it: ``_html_to_text``
    joins the blocks' ``text`` and reads nothing else, so a change to what the web parser says
    *about* a block cannot move an email ``LineAnchor``. #109 was exactly such a change — it
    added ``rows`` to table metadata and left every block's text alone — so bumping ``email``
    with it would have re-parsed and re-embedded every email in the corpus to produce identical
    bytes.

    Asserted against the web parser's own blocks rather than against a stored expectation,
    because the claim is about the coupling between the two parsers and not about this fixture:
    the day the conversion starts reading metadata, ``email`` has to move with ``html`` again
    and this is what says so. Both sides run through the parsers' public surfaces, so what is
    compared is what a stored anchor actually addresses.
    """
    from manicule.parsers.config import WebConfig  # noqa: PLC0415 - the conversion's own parser
    from manicule.parsers.web import WebParser  # noqa: PLC0415

    rows = "".join(
        f"<tr><td>T{index:03d}</td><td>Expansion {index}</td></tr>" for index in range(60)
    )
    body = (
        "<html><body><p>Glossary follows.</p>"
        f"<table><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>"
        "<p>Regards.</p></body></html>"
    )

    web = await read_blocks(WebParser(WebConfig()), raw_of(body, "text/html"))
    table = next(block for block in web if block.kind is BlockKind.TABLE)
    assert table.metadata["rows"], "the fixture must carry the metadata whose influence is denied"

    mail = await read_blocks(MailParser(MailConfig()), _html_message(body))

    # The first block is the rendered headers; the rest are the converted body, one per block
    # the web parser yielded, which is what makes their line numbers a function of that text.
    assert [block.text for block in mail[1:]] == [block.text for block in web], (
        "an HTML-only body's blocks are no longer the web parser's blocks, so what that parser "
        "says *about* a block can now move an email LineAnchor — and PARSERS['email'].rules has "
        "to move with PARSERS['html'].rules again"
    )
