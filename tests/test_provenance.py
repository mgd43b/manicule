"""The source-metadata interface: what it refuses, and what it keeps apart.

Every test here is about one of two properties, because those are the two the interface exists
for.

**Canonical identity and snapshot identity are both preserved, and neither can pretend to be the
other.** That is asserted structurally where it can be — a model that has nowhere to put a local
path cannot be made to assert one — and behaviorally where it cannot.

**A record is attacker-controlled input.** It comes out of a file inside the corpus and then out
of a database, so the validation is asserted on both paths: on the way in, where a bad manifest
earns a stated reason, and on the way back out, where a row somebody has edited must not be able
to put a ``javascript:`` link into a rendered citation.

Synthetic hosts only, under ``.test`` (RFC 6761 §6.2 reserves it), so nothing here can resolve.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from manicule.core.content import Document, DocumentStatus
from manicule.core.provenance import (
    CITABLE_SCHEMES,
    MAX_FIELD_CHARS,
    MAX_SECTION_DEPTH,
    PROVENANCE_KEY,
    LocalSnapshot,
    Provenance,
    SourceMetadata,
)
from manicule.parsers.config import ADF_MEDIA_TYPE, CONFLUENCE_MEDIA_TYPE

CANONICAL = "https://docs.example.test/pages/123456/retry-policy"


def a_source(**overrides: object) -> SourceMetadata:
    """A valid record, for tests that vary one field."""
    fields: dict[str, object] = {
        "title": "Retry policy",
        "canonical_uri": CANONICAL,
        "source_id": "123456",
        "version": "7",
        "modified_at": datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC),
    }
    fields.update(overrides)
    return SourceMetadata(**fields)  # pyright: ignore[reportArgumentType] - literals, checked by the model


def a_document(**overrides: object) -> Document:
    """A document carrying whatever metadata a test wants to put in it."""
    fields: dict[str, object] = {
        "id": "d1",
        "source": "local",
        "source_id": "/corpus/mirror/123456.html",
        "uri": CANONICAL,
        "title": "Retry policy",
        "content_hash": "sha256-of-the-bytes",
        "media_type": "text/html",
        "status": DocumentStatus.INDEXED,
    }
    fields.update(overrides)
    return Document(**fields)  # pyright: ignore[reportArgumentType] - literals, checked by the model


# --- the two identities stay apart -----------------------------------------------------------


def test_a_source_record_has_nowhere_to_put_a_local_path() -> None:
    """The structural half of "neither pretends to be the other".

    Asserted as a refusal rather than trusted to review. If ``SourceMetadata`` ever grows a
    snapshot field, a connector that only knows where a file sits on this disk gains the ability
    to assert that the file is the publication — which is precisely the defect the split exists
    to make unrepresentable, and it would be one field addition away with nothing to catch it.
    """
    for field in ("path", "snapshot_path", "checksum", "snapshot_checksum", "local_path"):
        with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
            SourceMetadata.model_validate({"title": "Retry policy", field: "../../etc/passwd"})


def test_a_snapshot_has_nowhere_to_put_a_canonical_address() -> None:
    """And the other direction, for the same reason.

    A snapshot that could carry a canonical URI is a snapshot that can be read as the
    publication's identity, which makes the two models one model with a naming convention.
    """
    for field in ("canonical_uri", "uri", "url", "source_id", "version"):
        with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
            LocalSnapshot.model_validate({"path": "mirror/123456.html", field: CANONICAL})


def test_the_three_timestamps_are_three_separate_fields() -> None:
    """Source modification, snapshot retrieval and local indexing never collapse.

    The failure this prevents is a corpus that looks freshly revised because somebody re-ran an
    import: with one timestamp, "the document changed" and "we fetched it again" are the same
    fact, and there is no way to ask which happened.
    """
    edited = datetime(2026, 1, 1, tzinfo=UTC)
    mirrored = datetime(2026, 6, 1, tzinfo=UTC)
    indexed = datetime(2026, 12, 1, tzinfo=UTC)

    document = a_document(
        indexed_at=indexed,
        metadata={
            PROVENANCE_KEY: Provenance(
                source=a_source(modified_at=edited),
                snapshot=LocalSnapshot(path="mirror/123456.html", retrieved_at=mirrored),
            ).as_metadata_value()
        },
    )

    record = document.provenance
    assert record is not None
    assert record.source is not None
    assert record.snapshot is not None
    assert record.source.modified_at == edited
    assert record.snapshot.retrieved_at == mirrored
    assert document.indexed_at == indexed
    assert len({edited, mirrored, indexed}) == 3, "the fixture must use three distinct moments"


def test_a_provenance_carries_a_source_or_a_reason_and_never_both_or_neither() -> None:
    """``Retention``'s rule, for the same reason: absent with a stated reason, never silent.

    Both halves are refused. A record with neither is a record that says a manifest was
    considered and reports nothing about it; a record with both is two answers to one question,
    and a reader has no rule for which wins.
    """
    with pytest.raises(ValidationError, match="either a source record or a reason"):
        Provenance()
    with pytest.raises(ValidationError, match="either a source record or a reason"):
        Provenance(source=a_source(), unavailable_reason="malformed")

    assert Provenance(source=a_source()).usable
    assert not Provenance(unavailable_reason="malformed").usable


# --- what a record must say ------------------------------------------------------------------


def test_a_record_that_identifies_nothing_is_refused() -> None:
    """A record carrying only a timestamp would take the canonical path and change nothing.

    That is the silent partial success: the manifest is honored, the code path is the new one,
    and the citation renders the filename it always did — so the person who wrote the manifest
    has no way to tell it is doing nothing.
    """
    with pytest.raises(ValidationError, match=r"at least one of title, canonical_uri or source_id"):
        SourceMetadata(modified_at=datetime(2026, 3, 4, tzinfo=UTC))

    assert SourceMetadata(title="Retry policy").title == "Retry policy"
    assert SourceMetadata(source_id="123456").source_id == "123456"
    assert SourceMetadata(canonical_uri=CANONICAL).canonical_uri == CANONICAL


# --- the security cases ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "javascript:window.__owned=1",
        "JavaScript:window.__owned=1",
        "data:text/html;base64,PHNjcmlwdD4=",
        "vbscript:msgbox(1)",
        "file:///etc/passwd",
        "file:///corpus/mirror/123456.html",
    ],
    ids=["javascript", "mixed case", "data", "vbscript", "etc passwd", "the snapshot itself"],
)
def test_a_canonical_uri_may_only_be_a_web_address(hostile: str) -> None:
    """The allowlist, exercised on the schemes a denylist gets wrong.

    Two of these are not executable and are refused anyway, and they are the interesting ones. A
    ``file:`` canonical URI is the local snapshot presented as the publication — the exact
    conflation this whole interface exists to prevent — and it would arrive looking like a
    perfectly well-formed URI, so nothing downstream would question it.

    The mixed-case entry is here because a scheme comparison that forgot to fold case would pass
    every other case in this list and fail only this one.
    """
    with pytest.raises(ValidationError, match=r"scheme|citable address"):
        SourceMetadata(title="Retry policy", canonical_uri=hostile)


def test_a_canonical_uri_must_address_a_host() -> None:
    """``https:///page`` parses, names nothing, and would render as a dead link."""
    with pytest.raises(ValidationError, match="names no host"):
        SourceMetadata(title="Retry policy", canonical_uri="https:///pages/123456")


def test_the_citable_schemes_are_the_two_a_reader_can_open() -> None:
    """Named, so widening the allowlist is a visible diff rather than a passing test."""
    assert {"http", "https"} == CITABLE_SCHEMES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "Retry\x1b[2Jpolicy"),
        ("title", "Retry\x00policy"),
        ("source_id", "123456\x1b]0;owned\x07"),
        ("version", "7\r\n8"),
    ],
    ids=["clear screen", "nul", "set window title", "newline"],
)
def test_a_control_character_is_refused_wherever_it_is_declared(field: str, value: str) -> None:
    """These strings are printed to a terminal, where ``\\x1b`` is not text.

    The browser surface escapes HTML and would not have caught any of these, because none of
    them is markup. ``manicule search`` prints a citation's title straight to stdout, so an
    escape sequence in a manifest is a route to clearing the screen of, or forging output in,
    the terminal of somebody reading a citation. ``NUL`` is refused for a different reason: it
    travels through JSON, SQLite and FTS5, and those three do not agree about it.
    """
    with pytest.raises(ValidationError, match="control character"):
        SourceMetadata(**{"title": "Retry policy", field: value})  # pyright: ignore[reportArgumentType]


def test_an_overlong_declared_string_is_refused_rather_than_truncated() -> None:
    """Truncating would present a *different* title as the document's own.

    Refusing is also the only option that keeps the record honest: a title cut to fit is a title
    manicule wrote, and the whole claim of the field is that the source wrote it.
    """
    with pytest.raises(ValidationError, match=r"over the .*-character limit"):
        SourceMetadata(title="x" * (MAX_FIELD_CHARS + 1))
    assert SourceMetadata(title="x" * MAX_FIELD_CHARS).title == "x" * MAX_FIELD_CHARS


def test_a_hierarchy_deeper_than_the_limit_is_refused() -> None:
    """The breadcrumb elides past its budget anyway, so depth buys nothing and costs a row."""
    with pytest.raises(ValidationError, match=r"over the .*-element limit"):
        SourceMetadata(title="Retry policy", section_path=tuple("abcdefgh" * 8))
    deep = tuple(f"level {index}" for index in range(MAX_SECTION_DEPTH))
    assert len(SourceMetadata(title="Retry policy", section_path=deep).section_path) == (
        MAX_SECTION_DEPTH
    )


def test_a_hierarchy_has_no_anonymous_levels() -> None:
    """An empty element would render as ``Engineering >  > Retry`` and mean nothing."""
    with pytest.raises(ValidationError, match="is empty; a hierarchy has no anonymous levels"):
        SourceMetadata(title="Retry policy", section_path=("Engineering", "   ", "Runbooks"))


@pytest.mark.parametrize("field", ["created_at", "modified_at"], ids=["created_at", "modified_at"])
def test_a_naive_timestamp_is_refused(field: str) -> None:
    """A moment with no offset is wrong by the mirror host's offset, silently.

    ``Watermark`` refuses one for the same reason. The specific harm here is that a source
    modification time is the thing somebody compares to decide which of two versions is newer,
    and being a day out is not visible in the value.
    """
    with pytest.raises(ValidationError, match="must carry a UTC offset"):
        SourceMetadata(**{"title": "Retry policy", field: datetime(2026, 3, 4, 5, 6, 7)})  # pyright: ignore[reportArgumentType]  # noqa: DTZ001 - a naive datetime is the input under test


def test_a_naive_retrieval_timestamp_is_refused_on_the_snapshot_too() -> None:
    """The snapshot half gets the same check, because it is the same class of mistake."""
    with pytest.raises(ValidationError, match="must carry a UTC offset"):
        LocalSnapshot(path="mirror/123456.html", retrieved_at=datetime(2026, 3, 4))  # noqa: DTZ001 - the input under test


def test_a_declared_content_type_must_be_a_well_formed_media_type() -> None:
    """Malformed is refused; carrying parameters is not malformed.

    **This assertion was the other way round, and the reasoning behind it reached the wrong
    rule.** It refused any parameter on the grounds that a value with ``;charset=`` on it "is a
    value somebody expected to be interpreted, and interpreting half of it is worse than refusing
    all of it". True — but the alternative to interpreting half is accepting the whole, not
    refusing the whole. The value is stored and compared here, never split, so nothing is
    half-interpreted either way.

    What is checked is that the whole string is well-formed, so a truncated parameter is still
    refused and a reader is never handed something it must guess at.
    """
    for malformed in ("text/html;charset", "text/html;;", "text/html;", "not a media type"):
        with pytest.raises(ValidationError, match="media type"):
            SourceMetadata(title="Retry policy", content_type=malformed)

    for declared in ("text/html", "text/html; charset=utf-8", 'text/plain;charset="utf-8"'):
        assert SourceMetadata(title="Retry policy", content_type=declared).content_type == declared


def test_a_record_can_state_the_media_type_manicule_itself_routed_by() -> None:
    """The case that forced the rule to change, and it was already broken before this parser.

    Both Confluence body formats are identified by a profile parameter, and the parameter is not
    decoration: it is the whole of what distinguishes storage format from the XHTML underneath it,
    and ADF from any other JSON. A record that cannot state the type a document was routed by
    has to lie about it or say nothing, and both of those are worse than parsing a semicolon.

    ``ADF_MEDIA_TYPE`` is asserted here as well as the new one because it was refused too, and had
    been since the day this validator was written — latent only because nothing had yet put it in
    a record.
    """
    for declared in (ADF_MEDIA_TYPE, CONFLUENCE_MEDIA_TYPE):
        assert SourceMetadata(title="Retry policy", content_type=declared).content_type == declared


# --- reading a record back out of storage ----------------------------------------------------


def test_a_document_with_no_record_reports_none() -> None:
    """The backward-compatible path, asserted rather than assumed.

    An ordinary local file with no manifest must take no new code path at all. ``None`` here is
    what makes every caller fall back to the local title and URI, which is what they used before
    this interface existed.
    """
    assert a_document(metadata={}).provenance is None
    assert a_document(metadata={"ancestors": ["Engineering"]}).provenance is None


@pytest.mark.parametrize(
    "stored",
    [
        "not an object",
        ["not", "an", "object"],
        42,
        {"source": {"canonical_uri": "javascript:window.__owned=1"}},
        {"source": {"title": "Retry policy"}, "unavailable_reason": "both at once"},
        {"source": {}},
        {"source": {"title": "Retry\x1b[2Jpolicy"}},
        {"unexpected_key": "value"},
    ],
    ids=[
        "a string",
        "a list",
        "a number",
        "a javascript canonical uri",
        "a source and a reason",
        "a source that says nothing",
        "a control character",
        "an unknown key",
    ],
)
def test_a_stored_record_is_validated_again_and_a_bad_one_reads_as_absent(
    stored: object,
) -> None:
    """The read path fails closed. This is the guard that makes storage untrusted too.

    The record has been through a database since anyone validated it, so "it was checked on the
    way in" is not a property of the value being read now — a hand-edited row, a restored
    backup, or a future bug in the write path all produce bytes that never passed the check.

    The ``javascript:`` case is the one with teeth: without re-validation it would reach a
    rendered citation on the strength of having once been stored, and the browser surface would
    escape it as *text* while a template that made it an ``href`` would not.

    Failing closed rather than raising is deliberate. Raising would make one corrupt row break
    every listing that touched it; reading as absent degrades that document's citation to the
    local filename, which is exactly where it would have been with no manifest.
    """
    assert a_document(metadata={PROVENANCE_KEY: stored}).provenance is None


def test_a_valid_stored_record_survives_the_round_trip_intact() -> None:
    """The positive control, without which every test above passes for a record that never loads.

    If :meth:`Provenance.from_metadata` returned ``None`` unconditionally, every "reads as
    absent" assertion above would pass and the feature would not work at all. This is the test
    that says the reader reads.
    """
    original = Provenance(
        source=a_source(section_path=("Engineering", "Runbooks")),
        snapshot=LocalSnapshot(
            path="mirror/123456.html", retrieved_at=datetime(2026, 6, 1, tzinfo=UTC)
        ),
    )
    loaded = a_document(metadata={PROVENANCE_KEY: original.as_metadata_value()}).provenance
    assert loaded == original


def test_a_refusal_reason_survives_the_round_trip() -> None:
    """A refused manifest has to stay visible, or it is a silently ignored one.

    The symptom of losing this is a citation that names a file — which is indistinguishable from
    having written no manifest, and is the bug this interface was built to remove.
    """
    original = Provenance(
        snapshot=LocalSnapshot(path="mirror/123456.html"),
        unavailable_reason="123456.html.source.json: is not valid JSON",
    )
    loaded = a_document(metadata={PROVENANCE_KEY: original.as_metadata_value()}).provenance
    assert loaded == original
    assert loaded is not None
    assert not loaded.usable
    assert "is not valid JSON" in loaded.unavailable_reason
