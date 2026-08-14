"""Enriched standalone HTML: reading a page's identity, and never trusting it with anything else.

The input is a file inside somebody's corpus, which makes it a document anyone with write access
to the wiki it was exported from could have authored. So there are two kinds of test here and the
second is the important one: that the metadata is read correctly, and that reading it cannot
cause a write anywhere, a read anywhere, a fetch, or an execution.

Fixtures are synthetic — ``https://docs.example.test/``, invented page ids, temporary roots.
"""

from __future__ import annotations

import io
import json
import socket
from pathlib import Path
from typing import Any, override

import pytest

from manicule.connectors import sidecar
from manicule.connectors.enriched_html import (
    MAX_HTML_BYTES,
    AdapterOutcome,
    UnusablePageError,
    extract,
    manifest_for,
    write_sidecars,
)
from manicule.core.ids import content_hash

CANONICAL = "https://docs.example.test/pages/1002"

ROWS: tuple[tuple[str, str], ...] = (
    ("Page ID", "1002"),
    ("Space", "ENG"),
    ("Version", "7"),
    ("Last modified", "2026-08-12T09:45:00Z"),
    ("Source", f'<a href="{CANONICAL}">canonical page</a>'),
)
"""The specification's own example, as label/value pairs.

Rows rather than a template with substitutions in it: an earlier version of this file did string
surgery on the finished HTML and silently produced malformed markup, so a test asserting the
hierarchy was really asserting what a mangled document parsed to.
"""


def page(
    rows: tuple[tuple[str, str], ...] = ROWS,
    *,
    title: str = "Retry Runbook",
    body: str = "<p>Retry with backoff.</p>",
    sections: int = 1,
) -> str:
    """One enriched page. ``rows`` are written verbatim, so a test can put anything in a value."""
    block = "\n".join(f"      <p><strong>{label}:</strong> {value}</p>" for label, value in rows)
    metadata = "\n".join(
        f"    <section data-source-metadata>\n{block}\n    </section>" for _ in range(sections)
    )
    return (
        f"<!doctype html>\n<html>\n  <head><title>{title}</title></head>\n  <body>\n"
        f"{metadata}\n"
        f'    <main data-document-representation="storage">\n      {body}\n    </main>\n'
        f"  </body>\n</html>\n"
    )


def rows_with(*extra: tuple[str, str], drop: str = "") -> tuple[tuple[str, str], ...]:
    """``ROWS`` plus ``extra``, optionally without the row labeled ``drop``."""
    kept = tuple((label, value) for label, value in ROWS if label != drop)
    return kept + extra


def rows_replacing(label: str, value: str) -> tuple[tuple[str, str], ...]:
    return tuple((name, value if name == label else held) for name, held in ROWS)


def written(root: Path, name: str = "1002.html", body: str | None = None) -> Path:
    target = root / "pages" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body if body is not None else page(), encoding="utf-8")
    return target


# --- what the page says about itself -------------------------------------------------------------


def test_the_specifications_example_yields_every_field_it_states() -> None:
    extracted = extract(page())

    assert extracted.source.source_id == "1002"
    assert extracted.source.title == "Retry Runbook"
    assert extracted.source.canonical_uri == CANONICAL
    assert extracted.source.version == "7"
    assert extracted.source.modified_at is not None
    assert extracted.source.modified_at.isoformat() == "2026-08-12T09:45:00+00:00"
    assert extracted.source.section_path == ("ENG",)


def test_the_canonical_address_comes_from_the_href_not_the_link_text() -> None:
    """``<a href="…">canonical page</a>`` is how these are written at least as often as not.

    Recording the words "canonical page" as the page's address would be a citation pointing at
    nothing, rendered as a link on the browser surface.
    """
    assert extract(page()).source.canonical_uri == CANONICAL


def test_the_hierarchy_is_coarsest_first_and_excludes_the_pages_own_title() -> None:
    """The chunker appends the title itself; a path carrying it twice is unintended emphasis."""
    extracted = extract(page(rows_with(("Ancestors", "Runbooks &gt; On-call"))))

    assert extracted.source.section_path == ("ENG", "Runbooks", "On-call")
    assert "Retry Runbook" not in extracted.source.section_path


def test_a_declared_title_beats_the_head_title() -> None:
    """The metadata section is the machine-addressed statement; ``<title>`` is for a tab."""
    extracted = extract(page(rows_with(("Title", "Retry policy"))))

    assert extracted.source.title == "Retry policy"


def test_a_page_stating_no_title_of_its_own_falls_back_to_the_documents() -> None:
    assert extract(page()).source.title == "Retry Runbook"


def test_a_manifest_claims_only_what_the_page_actually_said() -> None:
    """A field written for something the page never stated is a statement it did not make."""
    minimal = page((("Page ID", "1002"),), title="")

    manifest = manifest_for(extract(minimal), html=minimal.encode())

    assert manifest["source_id"] == "1002"
    for absent in ("version", "modified_at", "created_at", "canonical_uri", "section_path"):
        assert absent not in manifest, (
            f"{absent} was written for a page that said nothing about it, which claims the "
            f"export made a statement it did not make"
        )


# --- refusals name what was wrong ----------------------------------------------------------------


def test_an_ordinary_html_file_matches_no_profile_rather_than_being_a_broken_page() -> None:
    """The outcome a directory of ordinary HTML produces, and it is not a complaint.

    It used to be reported as "no [data-source-metadata] section", which reads as a defect in a
    file that has none because it is not an export and was never meant to be one. Every page in a
    documentation site would have carried that reason.
    """
    with pytest.raises(UnusablePageError, match="matches no configured") as refused:
        extract("<html><body><p>Just a page.</p></body></html>")

    assert refused.value.outcome is AdapterOutcome.NO_PROFILE


def test_two_metadata_sections_are_refused_rather_than_guessed_between() -> None:
    with pytest.raises(UnusablePageError, match=r"2 . \[data-source-metadata\]") as refused:
        extract(page(sections=2))

    assert refused.value.outcome is AdapterOutcome.AMBIGUOUS


def test_one_label_stated_twice_with_two_values_is_refused() -> None:
    """ "The first one wins" is a rule nobody writing an exporter knows about."""
    with pytest.raises(UnusablePageError, match="declares 'page id' twice"):
        extract(page(rows_with(("Page ID", "3003"))))


def test_a_page_declaring_no_id_is_refused_because_identity_is_the_point() -> None:
    """Without it a re-export is a second document rather than a new version of one."""
    with pytest.raises(UnusablePageError, match="declares no page id"):
        extract(page(rows_with(drop="Page ID")))


def test_a_timestamp_with_no_offset_is_refused_rather_than_assumed_to_be_utc() -> None:
    """Read as UTC it is wrong by the exporting host's offset, and it decides which is newer."""
    with pytest.raises(UnusablePageError, match="carries no UTC offset"):
        extract(page(rows_replacing("Last modified", "2026-08-12T09:45:00")))


def test_a_value_that_is_not_a_timestamp_is_refused_naming_the_field() -> None:
    with pytest.raises(UnusablePageError, match="last modified 'last thursday'"):
        extract(page(rows_replacing("Last modified", "last thursday")))


@pytest.mark.parametrize(
    "hostile",
    [
        "javascript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "file:///etc/passwd",
        "vbscript:msgbox",
    ],
)
def test_a_canonical_address_a_browser_would_execute_is_refused(hostile: str) -> None:
    """Refused by ``SourceMetadata`` itself, which is the point of building one.

    This module does not reimplement the scheme allowlist; it constructs the real model and lets
    it refuse. A second copy of that rule would be a second thing to keep in step, and the two
    would disagree the first time either was edited — with the browser surface rendering whichever
    one was wrong.
    """
    with pytest.raises(UnusablePageError, match="will not cite"):
        extract(page(rows_replacing("Source", f'<a href="{hostile}">canonical page</a>')))


def test_a_control_character_in_a_title_is_refused() -> None:
    """These fields are printed to a terminal, where ``\\x1b`` repositions the cursor."""
    with pytest.raises(UnusablePageError, match="will not cite"):
        extract(page(rows_with(("Title", "Retry\x1b[2Jpolicy"))))


# --- the manifest is the one the filesystem connector already reads -------------------------------


def test_the_manifest_round_trips_through_the_existing_sidecar_reader(tmp_path: Path) -> None:
    """The whole argument for this approach, asserted end to end.

    Nothing was added to the ingestion path. The manifest this writes is read back by
    ``sidecar.provenance_for`` — the function the filesystem connector already calls at fetch —
    and produces the canonical identity the page declared, beside the local location manicule
    observed for itself.
    """
    target = written(tmp_path)

    assert [outcome.written for outcome in write_sidecars(tmp_path)] == [True]

    provenance = sidecar.provenance_for(
        target, root=tmp_path, checksum=content_hash(target.read_bytes())
    )
    assert provenance is not None
    assert provenance.usable
    assert provenance.source is not None
    assert provenance.source.source_id == "1002"
    assert provenance.source.canonical_uri == CANONICAL
    assert provenance.source.title == "Retry Runbook"
    assert provenance.snapshot is not None
    assert provenance.snapshot.path == "pages/1002.html"


def test_an_edited_page_whose_manifest_was_not_regenerated_is_refused_with_a_reason(
    tmp_path: Path,
) -> None:
    """What the declared checksum buys, and why it is written.

    The manifest states a version and a modification time. If the page beside it is edited and
    the conversion is not re-run, those describe a revision that is no longer there — so the
    record is refused rather than attached, and the operator is told which file to re-convert.
    """
    target = written(tmp_path)
    write_sidecars(tmp_path)

    target.write_text(page(body="<p>Do not retry.</p>"), encoding="utf-8")

    provenance = sidecar.provenance_for(
        target, root=tmp_path, checksum=content_hash(target.read_bytes())
    )
    assert provenance is not None
    assert not provenance.usable
    assert "not from the same retrieval" in provenance.unavailable_reason


def test_the_manifest_declares_no_snapshot_path(tmp_path: Path) -> None:
    """It is cross-checked against the *ingestion* root, which a conversion cannot know.

    Writing one would make every manifest refusable the moment the connector was pointed at a
    different root than the conversion was — a true statement about the wrong tree.
    """
    written(tmp_path)
    write_sidecars(tmp_path)

    manifest = json.loads(
        (tmp_path / "pages" / "1002.html.source.json").read_text(encoding="utf-8")
    )
    assert sidecar.SNAPSHOT_PATH not in manifest
    assert "snapshot_checksum" in manifest


def test_converting_twice_changes_nothing(tmp_path: Path) -> None:
    """Idempotent, which is what makes re-running it after an edit the obvious remedy."""
    written(tmp_path)
    write_sidecars(tmp_path)
    manifest_path = tmp_path / "pages" / "1002.html.source.json"
    first = manifest_path.read_text(encoding="utf-8")

    write_sidecars(tmp_path, force=True)

    assert manifest_path.read_text(encoding="utf-8") == first


# --- security -------------------------------------------------------------------------------------


def test_a_hostile_page_id_cannot_influence_where_anything_is_written(tmp_path: Path) -> None:
    """Traversal is not refused here; it is unrepresentable.

    The output path is derived from the file the walk reached, so no value read out of a
    document reaches it. The page id below is recorded as a *string* — it is the publisher's
    identifier and manicule does not interpret it — and the manifest still lands beside the page.
    """
    escape = tmp_path / "escape-target"
    escape.mkdir()
    written(tmp_path, body=page(rows_replacing("Page ID", "../../../escape-target/owned")))

    write_sidecars(tmp_path)

    assert (tmp_path / "pages" / "1002.html.source.json").is_file()
    assert list(escape.iterdir()) == [], "a page id reached the filesystem"
    assert not (tmp_path / "escape-target" / "owned").exists()


def test_the_source_file_is_never_modified(tmp_path: Path) -> None:
    """The reason scripts and macro bodies stay inert is that nothing rewrites them."""
    target = written(
        tmp_path,
        body=page(
            body='<script>fetch("https://evil.example.test")</script>'
            "<![CDATA[ macro body with <tags> ]]>"
        ),
    )
    before = target.read_bytes()

    write_sidecars(tmp_path)

    assert target.read_bytes() == before


def test_conversion_makes_no_network_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical URL is metadata. It is read out of an attribute and never dereferenced."""
    written(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        message = "conversion opened a socket"
        raise AssertionError(message)

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert [outcome.written for outcome in write_sidecars(tmp_path)] == [True]


def test_an_existing_manifest_is_not_replaced_without_being_asked(tmp_path: Path) -> None:
    """One already there was most likely written by hand or by another tool."""
    written(tmp_path)
    manifest_path = tmp_path / "pages" / "1002.html.source.json"
    manifest_path.write_text('{"source_id": "written-by-hand"}', encoding="utf-8")

    outcomes = write_sidecars(tmp_path)

    assert manifest_path.read_text(encoding="utf-8") == '{"source_id": "written-by-hand"}'
    assert not outcomes[0].written
    assert "--force" in outcomes[0].skipped_reason


def test_force_replaces_it(tmp_path: Path) -> None:
    written(tmp_path)
    manifest_path = tmp_path / "pages" / "1002.html.source.json"
    manifest_path.write_text('{"source_id": "written-by-hand"}', encoding="utf-8")

    outcomes = write_sidecars(tmp_path, force=True)

    assert outcomes[0].written
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["source_id"] == "1002"


def test_a_symlink_out_of_the_tree_writes_nothing_outside_it(tmp_path: Path) -> None:
    """A write outside the root the operator named.

    Two mechanisms hold this — the walk skips symlinks, and every path is checked against the
    root — and disabling *either* alone leaves it holding. That is defense in depth rather than a
    redundant check, and it is why the test below exists to pin the first one on its own.
    """
    outside = tmp_path.parent / "outside-the-root"
    outside.mkdir(exist_ok=True)
    (outside / "1003.html").write_text(page(), encoding="utf-8")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "link.html").symlink_to(outside / "1003.html")

    outcomes = write_sidecars(root)

    assert outcomes == []
    assert not (outside / "1003.html.source.json").exists()


def test_a_symlink_within_the_tree_is_not_walked_either(tmp_path: Path) -> None:
    """The symlink rule on its own, with root containment unable to help.

    This link resolves *inside* the root, so the containment check passes it. Only the walk's
    refusal to follow symlinks stops the page being converted twice — once under its own name and
    once under the link's — which would put two manifests on disk describing one page, and a
    second document in the corpus citing the same source id.
    """
    target = written(tmp_path)
    (tmp_path / "pages" / "alias.html").symlink_to(target)

    outcomes = write_sidecars(tmp_path)

    assert [outcome.path.name for outcome in outcomes] == ["1002.html"]
    assert not (tmp_path / "pages" / "alias.html.source.json").exists()


def test_an_oversized_file_is_skipped_with_its_size(tmp_path: Path) -> None:
    """Without a bound, a file in the named directory decides how much memory this allocates."""
    target = tmp_path / "huge.html"
    target.write_bytes(b"<html>" + b"x" * (MAX_HTML_BYTES + 1))

    outcomes = write_sidecars(tmp_path)

    assert not outcomes[0].written
    assert "over the" in outcomes[0].skipped_reason


# --- the walk -------------------------------------------------------------------------------------


def test_a_manifest_is_not_itself_treated_as_a_page(tmp_path: Path) -> None:
    written(tmp_path)
    write_sidecars(tmp_path)

    again = write_sidecars(tmp_path, force=True)

    assert [outcome.path.name for outcome in again] == ["1002.html"]


def test_every_page_that_produced_nothing_is_reported_with_its_reason(tmp_path: Path) -> None:
    """A run reporting only successes presents "no metadata anywhere" as a clean conversion."""
    written(tmp_path)
    (tmp_path / "pages" / "plain.html").write_text("<html><body>hi</body></html>", encoding="utf-8")

    outcomes = write_sidecars(tmp_path)

    skipped = {
        outcome.path.name: (outcome.outcome, outcome.skipped_reason)
        for outcome in outcomes
        if not outcome.written
    }
    assert "plain.html" in skipped
    outcome, reason = skipped["plain.html"]
    assert outcome is AdapterOutcome.NO_PROFILE
    assert "matches no configured" in reason


# --- found by self-review ------------------------------------------------------------------------


def test_a_linked_title_is_the_title_and_not_the_link() -> None:
    """The href is preferred for the address field only, never for every field.

    An earlier version preferred an anchor's ``href`` wherever it found one, so a page whose
    title happened to be a link recorded the URL as its title — a citation captioned with its own
    address, on every surface.
    """
    extracted = extract(
        page(
            rows_with(("Title", '<a href="https://docs.example.test/pages/1002">Retry policy</a>'))
        )
    )

    assert extracted.source.title == "Retry policy"
    assert extracted.source.canonical_uri == CANONICAL


def test_two_spellings_of_one_field_are_both_named_in_the_refusal() -> None:
    """ "Source" and "Canonical URL" fill the same field, and a reader needs to be told both.

    Reporting only the second spelling sends them looking for a duplicate row that is not there.
    """
    with pytest.raises(UnusablePageError, match="'source' and 'canonical url'"):
        extract(page(rows_with(("Canonical URL", "https://docs.example.test/pages/9999"))))


def test_an_oversized_page_is_never_read_into_memory(tmp_path: Path) -> None:
    """The limit has to bound the *read*, not measure what came back from an unbounded one.

    The first version called ``read_bytes()`` and checked the length, so the whole file was
    already resident by the time anything decided it was too big — which is exactly the
    allocation the limit exists to prevent. This pins the read itself.
    """
    target = tmp_path / "huge.html"
    target.write_bytes(b"<html>" + b"x" * (MAX_HTML_BYTES + 1024))
    reads: list[int | None] = []

    class Recording(io.BufferedReader):
        """A real file handle that remembers how much each read asked for."""

        @override
        def read(self, size: int | None = -1, /) -> bytes:
            reads.append(size)
            return super().read(size)

    def recording(self: Path, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return Recording(io.FileIO(str(self), "rb"))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", recording)
        outcomes = write_sidecars(tmp_path)

    assert not outcomes[0].written
    assert reads, "the page was never opened, so this asserts nothing about how it was read"
    assert all(size is not None and 0 <= size <= MAX_HTML_BYTES + 1 for size in reads), (
        f"an unbounded read reached the file: {reads}"
    )
