"""Sidecar generation against a *named* source, with that source's own profiles.

``manicule connector sidecar <root>`` converts with the built-in default profile. For a corpus
whose exporter spells its markers differently that is not a partial success, it is a predictable
one: every page reports ``no_profile``, nothing is written, and the configured sync over the very
same directory adapts all of them. Two readings of one profile, and neither report mentions the
other.

``--source docs`` closes that by taking the root and the profiles from the connector a sync would
run. The properties worth testing are therefore about the *seam*, not about the conversion:

* the profiles that reach ``write_sidecars`` are the connector's own tuple, in configured order,
  with nothing appended (:func:`test_the_configured_profiles_reach_the_conversion_unchanged`);
* what the conversion adapts and what the connector adapts are the same set, which is the
  disagreement this exists to make impossible;
* the boundary holds — an unknown, disabled, non-filesystem or adaptation-off source is refused
  by name, and a caller-supplied subdirectory cannot leave the configured root.

Fixtures are synthetic throughout: invented page ids, ``https://docs.example.test/``, temporary
roots, and marker attributes that are **deliberately not** the built-in ones, because a fixture
that matched the default would pass whether or not the source's profiles were used at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import JsonValue

from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors.enriched import DEFAULT_PROFILE, AdapterOutcome
from manicule.connectors.filesystem import ENRICHED_KEY, FilesystemConnector
from manicule.connectors.plugin import ConnectorsPlugin
from manicule.container import Container
from manicule.core.errors import ConfigError, PolicyError, UnknownEntityError
from manicule.core.ids import content_hash
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE
from manicule.plugins.registry import ComponentRegistry
from tests.app.fakes import FakeBackend
from tests.ingest.fakes import MemoryIngestStore

if TYPE_CHECKING:
    from manicule.app.results import SidecarReport
    from manicule.core.content import Document

CANONICAL = "https://docs.example.test/pages/1002"

PROFILE: JsonValue = {
    "name": "custom-storage-export",
    "metadata_selector": "[data-export-metadata]",
    "body_selector": '[data-export-representation="storage"]',
    "representation": CONFLUENCE_MEDIA_TYPE,
    # Label spellings of this exporter's own, and they **replace** the defaults rather than
    # extending them. So a page written with "Page ID" would be refused by this profile and a
    # page written with "Identifier" is refused by the default one, which is what makes every
    # assertion below a statement about which profile actually ran.
    "labels": {"identifier": "source_id", "revision": "version", "source": "canonical_uri"},
}
"""The specification's example profile, with one addition.

``labels`` there maps only ``identifier`` and ``revision``. That is enough to adapt a page and
not enough for the manifest to carry a canonical URI, which the required end-to-end test asks
for in the same breath — so the mapping for the address is added rather than the assertion
dropped. Stated because it is a departure from the example as written.
"""


def page(
    identifier: str = "1002",
    *,
    title: str = "Retry Runbook",
    body: str = "<h1>Retry Runbook</h1><p>The client retries twice.</p>",
    metadata_attribute: str = "data-export-metadata",
    body_attribute: str = 'data-export-representation="storage"',
    sections: int = 1,
) -> str:
    """One enriched page in this exporter's dialect, not the built-in one."""
    block = (
        f"<p><strong>Identifier:</strong> {identifier}</p>\n"
        f"<p><strong>Revision:</strong> 7</p>\n"
        f'<p><strong>Source:</strong> <a href="https://docs.example.test/pages/{identifier}">'
        f"canonical page</a></p>"
    )
    banner = "\n".join(f"<section {metadata_attribute}>{block}</section>" for _ in range(sections))
    return (
        f"<!doctype html><html><head><title>{title}</title></head><body>\n"
        f"<nav>Exported by the wiki mirror. Home | Spaces | Search</nav>\n"
        f"{banner}\n<main {body_attribute}>{body}</main>\n</body></html>\n"
    )


def corpus(tmp_path: Path, *, name: str = "1002.html", body: str | None = None) -> Path:
    """A root holding one page, one directory down, with no manifest yet."""
    root = tmp_path / "export"
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "pages" / name).write_text(body if body is not None else page(), encoding="utf-8")
    return root


def settings_for(
    root: Path,
    *,
    profiles: list[JsonValue] | None = None,
    enabled: bool = True,
    connector_type: str = "filesystem",
    name: str = "docs",
) -> Settings:
    """Configuration declaring one source, exactly as a ``config.toml`` would."""
    options: dict[str, JsonValue] = {"root": str(root)}
    if profiles is not None:
        options["enriched_profiles"] = profiles
    return Settings(
        connectors={name: ConnectorSettings(type=connector_type, options=options, enabled=enabled)}
    )


def container_for(settings: Settings) -> Container:
    """A container holding the **real** connectors plugin.

    The real registry rather than a stub, so ``FilesystemConfig`` validation, the per-instance
    options merge and ``build_filesystem`` are the ones the product runs. A test that built the
    connector itself would be asserting that two lines it wrote agree with each other.
    """
    registry = ComponentRegistry().bind("connectors")
    ConnectorsPlugin().register(registry)
    return Container(settings, registry)


async def service_for(settings: Settings) -> tuple[ApplicationService, FakeBackend]:
    """A service whose ingest surface hands back connectors from a real container.

    Only filesystem sources are built. A Confluence source in these fixtures exists to be
    *refused* by type and would need a credential to construct, and swallowing that failure here
    would let a genuine build error look like a source that was never asked for.
    """
    backend = FakeBackend(settings=settings)
    container = container_for(settings)
    for name, configured in settings.connectors.items():
        if configured.type == "filesystem":
            backend.ingestion_.connectors[name] = await container.connector(name)
    return ApplicationService(backend), backend


async def convert(
    settings: Settings, *, source: str = "docs", root: Path | None = None, force: bool = False
) -> SidecarReport:
    service, _ = await service_for(settings)
    return await service.connector_sidecar(root, source=source, force=force)


# --- the required end-to-end sequence -----------------------------------------------------------
#
# The specification states it as twelve numbered steps. They are separate assertions rather than
# one run with a summary at the end, because a single "it worked" is how a requirement stops being
# checked: whichever step breaks, the failure has to name it.


async def test_a_named_source_writes_exactly_one_manifest(tmp_path: Path) -> None:
    """Steps 1-3. One page under the root, one manifest, and nothing else written."""
    root = corpus(tmp_path)

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.written == 1
    assert report.considered == 1
    assert report.by_outcome == {AdapterOutcome.ADAPTED.value: 1}
    assert [path.name for path in sorted(root.glob("**/*.source.json"))] == [
        "1002.html.source.json"
    ]


async def test_the_manifest_carries_the_declared_identity_address_version_and_type(
    tmp_path: Path,
) -> None:
    """Step 4. Every field the page declared, read back off disk rather than off the report."""
    root = corpus(tmp_path)

    await convert(settings_for(root, profiles=[PROFILE]))

    manifest = json.loads((root / "pages" / "1002.html.source.json").read_text(encoding="utf-8"))
    assert manifest["source_id"] == "1002"
    assert manifest["canonical_uri"] == CANONICAL
    assert manifest["version"] == "7"
    assert manifest["content_type"] == CONFLUENCE_MEDIA_TYPE


async def test_the_report_names_the_source_and_the_profiles_that_ran(tmp_path: Path) -> None:
    """Not a step of its own, and the thing that makes a disappointing run diagnosable.

    ``no_profile`` for every page is the same line whether the run used the default profile or
    the source's own two, and the operator's next move is opposite in the two cases.
    """
    root = corpus(tmp_path)

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.source == "docs"
    assert report.profiles == ("custom-storage-export",)


async def test_the_positional_form_still_uses_the_built_in_default(tmp_path: Path) -> None:
    """Backward compatibility, and the defect stated as a test.

    The same corpus, converted without ``--source``, adapts nothing — because these markers are
    not the default's. That is precisely what an operator saw before this change *while their
    configured sync adapted every one of these pages*.
    """
    root = corpus(tmp_path)
    service, _ = await service_for(settings_for(root, profiles=[PROFILE]))

    report = await service.connector_sidecar(root)

    assert report.written == 0
    assert report.by_outcome == {AdapterOutcome.NO_PROFILE.value: 1}
    assert report.source == ""
    assert report.profiles == (DEFAULT_PROFILE.name,)


# --- steps 5 to 12, through the real pipeline ---------------------------------------------------


async def _synced(settings: Settings, store: MemoryIngestStore | None = None) -> MemoryIngestStore:
    """Sync the configured source through the real pipeline over the real parsers.

    The connector is the configured one — built by the real factory from the real settings — so
    the profiles that adapt at fetch are the same tuple the conversion just used, by construction
    rather than by a test passing the same literal to both.
    """
    from tests.connectors.test_enriched_storage import pipeline  # noqa: PLC0415

    held = MemoryIngestStore() if store is None else store
    await pipeline(held).run(await container_for(settings).connector("docs"))
    return held


def only(store: MemoryIngestStore) -> Document:
    """The single document the store holds, or a failure naming what it holds instead."""
    documents = list(store.documents.values())
    assert len(documents) == 1, f"expected one document, got {[d.source_id for d in documents]}"
    return documents[0]


async def test_the_document_is_keyed_by_the_declared_page_id(tmp_path: Path) -> None:
    """Step 6. The whole point of the manifest: identity off the path and onto the page."""
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    await convert(settings)

    store = await _synced(settings)

    assert [document.source_id for document in store.documents.values()] == ["1002"]


async def test_the_extracted_body_reaches_the_storage_parser(tmp_path: Path) -> None:
    """Step 7. The media type is what routed it, so this is the whole of "it arrived"."""
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    await convert(settings)

    store = await _synced(settings)

    assert only(store).media_type == CONFLUENCE_MEDIA_TYPE
    assert only(store).metadata["parser_used"] == "confluence"


async def test_the_metadata_wrapper_is_absent_from_every_chunk(tmp_path: Path) -> None:
    """Step 8. Not "mostly absent": the banner is what generic ingestion used to index."""
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    await convert(settings)

    store = await _synced(settings)

    texts = [chunk.text for chunks in store.chunks.values() for chunk in chunks]
    assert texts, "nothing was chunked, so the absence below is vacuous"
    for text in texts:
        for banner in ("Identifier", "Revision", "Exported by the wiki mirror", CANONICAL):
            assert banner not in text, f"the wrapper reached a chunk: {text!r}"


async def test_the_complete_source_snapshot_remains_retained(tmp_path: Path) -> None:
    """Step 9. The file on disk is untouched, and the record's digest is the file's.

    Both halves matter. The stored bytes are the *extracted body*, so a check that only compared
    ``content_hash`` would pass against a corpus whose originals had been rewritten.
    """
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    before = (root / "pages" / "1002.html").read_bytes()
    await convert(settings)

    store = await _synced(settings)

    after = (root / "pages" / "1002.html").read_bytes()
    assert after == before, "the conversion or the sync rewrote the page"
    assert b"data-export-metadata" in after, "the wrapper is not on disk any more"
    document = only(store)
    record = document.metadata[ENRICHED_KEY]
    assert isinstance(record, dict)
    assert record["snapshot_checksum"] == content_hash(after)
    assert document.content_hash != content_hash(after), (
        "the stored digest is the file's, so the body was never extracted"
    )


async def test_doctor_reports_healthy_identity_and_content(tmp_path: Path) -> None:
    """Step 10. Both checks green over the documents a real sync produced.

    The documents are carried into the surface's own store rather than invented, so this is a
    statement about what the pipeline wrote and not about a fixture somebody hand-built to be
    healthy.
    """
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    await convert(settings)
    store = await _synced(settings)

    service, backend = await service_for(settings)
    for document in store.documents.values():
        backend.store.add(document)
    diagnosis = await service.doctor()

    checks = {check.name: check for check in diagnosis.checks}
    assert checks["document-identity"].state == "ok", checks["document-identity"].detail
    assert checks["document-content"].state == "ok", checks["document-content"].detail


async def test_moving_the_page_updates_one_document_rather_than_creating_a_second(
    tmp_path: Path,
) -> None:
    """Steps 11 and 12, and the reason identity was moved off the path at all.

    The manifest travels with the page, because it is what the page's identity is written in.
    A mirror reorganised from by-space to by-tree moves both, and the corpus must see one page
    that moved rather than one deletion and one arrival.
    """
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    await convert(settings)
    store = await _synced(settings)
    first = only(store)

    moved = root / "by-tree" / "eng"
    moved.mkdir(parents=True)
    for name in ("1002.html", "1002.html.source.json"):
        (root / "pages" / name).rename(moved / name)
    (root / "pages").rmdir()
    await _synced(settings, store)

    assert len(store.documents) == 1, (
        f"the move created a second document: {[d.source_id for d in store.documents.values()]}"
    )
    assert only(store).id == first.id, "same page, different document id"
    record = only(store).provenance
    assert record is not None
    assert record.snapshot is not None
    assert record.snapshot.path == "by-tree/eng/1002.html", (
        "the document did not learn where the page went"
    )


# --- profile precedence, exactly ----------------------------------------------------------------


async def test_the_configured_profiles_reach_the_conversion_unchanged(tmp_path: Path) -> None:
    """Requirement 9, and it is an ordering and an identity claim rather than a membership one.

    ``in`` would pass for a run that appended the default, reordered the two, or silently
    substituted an equal-looking profile parsed a second time. The connector's own tuple is the
    authority, so the comparison is against that object.
    """
    assert isinstance(PROFILE, dict)
    second: JsonValue = {
        **PROFILE,
        "name": "second-export",
        "metadata_selector": "[data-other-metadata]",
    }
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE, second])
    service, backend = await service_for(settings)

    report = await service.connector_sidecar(None, source="docs")

    connector = backend.ingestion_.connectors["docs"]
    assert isinstance(connector, FilesystemConnector)
    assert report.profiles == tuple(profile.name for profile in connector.profiles)
    assert report.profiles == ("custom-storage-export", "second-export")
    assert DEFAULT_PROFILE.name not in report.profiles, "the default was appended"


async def test_a_source_that_omits_the_default_does_not_get_it_back(tmp_path: Path) -> None:
    """The other half of "configuration already decides whether the default is included".

    A page in the *default* dialect under a source that declares only a custom profile must not
    convert, because the sync over that source will not adapt it either. Appending the default
    here would write a manifest for a page the connector then declines to read.
    """
    root = tmp_path / "export"
    root.mkdir()
    (root / "default-dialect.html").write_text(
        page(
            metadata_attribute="data-source-metadata",
            body_attribute='data-document-representation="storage"',
        ),
        encoding="utf-8",
    )

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.written == 0
    assert report.by_outcome == {AdapterOutcome.NO_PROFILE.value: 1}
    assert not list(root.glob("**/*.source.json"))


async def test_conversion_and_sync_adapt_the_same_pages(tmp_path: Path) -> None:
    """The last negative case, which must be impossible rather than merely absent.

    "A configured profile that matches normal sync but not sidecar generation" was the whole
    defect. It is checked by adapting the same corpus twice — once as the conversion does it and
    once as the connector does at fetch — and requiring the two verdicts to agree file by file.
    """
    from manicule.connectors.enriched import UnusablePageError, adapt  # noqa: PLC0415

    root = corpus(tmp_path)
    (root / "pages" / "plain.html").write_text("<html><body>hi</body></html>", encoding="utf-8")
    (root / "pages" / "1003.html").write_text(page("1003"), encoding="utf-8")
    settings = settings_for(root, profiles=[PROFILE])
    service, backend = await service_for(settings)

    report = await service.connector_sidecar(None, source="docs")

    connector = backend.ingestion_.connectors["docs"]
    assert isinstance(connector, FilesystemConnector)
    converted = {
        Path(skip.path).name: skip.outcome for skip in report.skipped
    } | {  # the ones that produced a manifest are the complement
        path.name.removesuffix(".source.json"): AdapterOutcome.ADAPTED.value
        for path in root.glob("**/*.source.json")
    }
    for page_path in sorted((root / "pages").glob("*.html")):
        try:
            adapt(page_path.read_text(encoding="utf-8"), profiles=connector.profiles)
        except UnusablePageError as refusal:
            at_fetch = refusal.outcome.value
        else:
            at_fetch = AdapterOutcome.ADAPTED.value
        assert converted[page_path.name] == at_fetch, (
            f"{page_path.name} adapts one way at conversion and another at fetch"
        )


# --- the boundary: which sources may be named ---------------------------------------------------


async def test_an_unknown_source_is_refused_by_name(tmp_path: Path) -> None:
    """And the message says which *kind* of name was wanted, because that is the mistake."""
    service, _ = await service_for(settings_for(corpus(tmp_path), profiles=[PROFILE]))

    with pytest.raises(UnknownEntityError) as refusal:
        await service.connector_sidecar(None, source="nope")

    assert "configured instance" in str(refusal.value)
    assert "docs" in str(refusal.value), "the configured names are not listed"


async def test_a_connector_type_is_not_a_source_name(tmp_path: Path) -> None:
    """Requirement 1 stated as its own test: ``--source filesystem`` is not ``--source docs``."""
    service, _ = await service_for(settings_for(corpus(tmp_path), profiles=[PROFILE]))

    with pytest.raises(UnknownEntityError):
        await service.connector_sidecar(None, source="filesystem")


async def test_a_disabled_source_is_refused(tmp_path: Path) -> None:
    """The same refusal ``connector sync`` gives, for the same reason.

    A switch an operator set and checked has to mean something on every surface that reads it.
    Converting a disabled source's corpus is preparing an ingest that will not run.
    """
    root = corpus(tmp_path)
    service, _ = await service_for(settings_for(root, profiles=[PROFILE], enabled=False))

    with pytest.raises(PolicyError) as refusal:
        await service.connector_sidecar(None, source="docs")

    assert "enabled = false" in str(refusal.value)
    assert not list(root.glob("**/*.source.json")), "a refused run still wrote a manifest"


async def test_a_non_filesystem_source_is_refused_by_its_configured_type(tmp_path: Path) -> None:
    """There is no local directory of pages behind a Confluence source to write beside.

    A connector object *is* put on the ingest surface, so that removing the type check does not
    simply produce "nothing was registered under that name" — a failure that would let this pass
    while proving nothing about the check it is named for. With one there, the two refusals are
    told apart by their messages: this one names the configured type, and the construction guard
    behind it names the class that was built instead.
    """
    del tmp_path
    settings = Settings(
        connectors={
            "wiki": ConnectorSettings(
                type="confluence", options={"base_url": "https://docs.example.test/wiki"}
            )
        }
    )
    service, backend = await service_for(settings)
    backend.ingestion_.connectors["wiki"] = object()

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(None, source="wiki")

    assert "'confluence'" in str(refusal.value), (
        "the refusal did not come from the configured-type check"
    )
    assert backend.ingestion_.asked == [], "a connector was built before the type was checked"


async def test_a_filesystem_name_that_builds_something_else_is_refused(tmp_path: Path) -> None:
    """The guard behind the type check, for a plugin that registered another class.

    Reachable only where something has taken over the ``filesystem`` component name. Refused
    rather than duck-typed: the root and the profiles read off that object decide where this
    writes and what it recognises.

    **What this asserts is that the message names the class, not that it is spelled a particular
    way.** It used to read ``"builds a object" in ...``, pinning an article that was wrong for
    the one type it fires on most — and pinning it in the direction that makes the defect cost
    something to fix, because correcting the message turned the test red. A substring assertion
    is a specification of the part it quotes; quoting the ungrammatical part specified the
    ungrammaticality.
    """
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    service, backend = await service_for(settings)
    backend.ingestion_.connectors["docs"] = object()

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(None, source="docs")

    assert "builds an instance of object" in str(refusal.value), (
        "the refusal did not name the class that was built instead"
    )
    assert not list(root.glob("**/*.source.json"))


async def test_a_source_with_an_empty_profile_list_is_refused(tmp_path: Path) -> None:
    """Adaptation is off for that source, so a conversion under it writes nothing at all.

    Refused rather than run, because a run would report ``no_profile`` for every page — which is
    the exact instruction-that-cannot-work this change exists to remove, reintroduced one layer
    down.
    """
    root = corpus(tmp_path)
    service, _ = await service_for(settings_for(root, profiles=[]))

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(None, source="docs")

    assert "empty `enriched_profiles`" in str(refusal.value)
    assert "no_profile" in str(refusal.value), "the refusal does not say what would have happened"
    assert not list(root.glob("**/*.source.json"))


async def test_neither_a_root_nor_a_source_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """There is no sensible default directory, and the working directory is the wrong guess."""
    service, _ = await service_for(settings_for(corpus(tmp_path), profiles=[PROFILE]))

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(None)

    assert "--source" in str(refusal.value)


# --- the boundary: where a named source may be pointed ------------------------------------------


async def test_a_subdirectory_inside_the_root_is_allowed(tmp_path: Path) -> None:
    """The bounded narrowing, and the reason the guard below is a boundary rather than a ban."""
    root = corpus(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "1003.html").write_text(page("1003"), encoding="utf-8")

    report = await convert(settings_for(root, profiles=[PROFILE]), root=Path("pages"))

    assert report.written == 1
    assert (root / "pages" / "1002.html.source.json").is_file()
    assert not (root / "other" / "1003.html.source.json").exists(), "the narrowing did nothing"


async def test_narrowing_to_a_subdirectory_narrows_duplicate_detection_too(
    tmp_path: Path,
) -> None:
    """The cost of the bounded subdirectory, stated rather than discovered.

    Two pages declaring one id are refused *as a pair* — and the pair is only visible to a run
    that walked both. Narrowing to one of their directories writes a manifest for the one it saw,
    because as far as that run is concerned nothing clashes. Converting the whole root refuses
    both, which is why the whole root is the default.

    **The corpus does not silently break, and that is what makes this a trade rather than a
    hole.** The unconverted twin keeps its path identity, and the moment the converted one holds
    the id it declares, ``doctor``'s ``document-identity`` check stops excluding it and reports
    two rows for one page. Pinned here so that a change which lost the later detection would fail
    something, rather than leaving the earlier detection's absence uncovered at both ends.
    """
    root = corpus(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "dup.html").write_text(page(), encoding="utf-8")
    settings = settings_for(root, profiles=[PROFILE])

    narrowed = await convert(settings, root=Path("pages"))

    assert narrowed.by_outcome == {AdapterOutcome.ADAPTED.value: 1}, (
        "the narrowed run saw the clash it could not have seen"
    )
    store = await _synced(settings)
    service, backend = await service_for(settings)
    for document in store.documents.values():
        backend.store.add(document)
    check = {c.name: c for c in (await service.doctor()).checks}["document-identity"]

    assert check.state == "degraded", "the corpus holds two rows for one page and doctor is quiet"
    assert "dup.html" in check.detail


async def test_the_whole_root_refuses_the_pair_the_narrowed_run_could_not_see(
    tmp_path: Path,
) -> None:
    """The other half, so the test above is a statement about narrowing and not about the pair."""
    root = corpus(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "dup.html").write_text(page(), encoding="utf-8")

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.by_outcome == {AdapterOutcome.DUPLICATE_IDENTITY.value: 2}
    assert not list(root.glob("**/*.source.json"))


async def test_a_narrowed_conversion_writes_the_same_manifest_as_a_whole_root_one(
    tmp_path: Path,
) -> None:
    """Why narrowing is safe at all: a manifest says nothing about which root produced it.

    ``snapshot_path`` is deliberately never written (§4), so no field in a manifest is relative
    to the conversion's root. If one were, a narrowed run would emit paths that disagreed with
    the ones the connector walks to and every manifest it wrote would be refused at ingest.
    """
    whole = corpus(tmp_path / "a")
    narrow = corpus(tmp_path / "b")
    await convert(settings_for(whole, profiles=[PROFILE]))
    await convert(settings_for(narrow, profiles=[PROFILE]), root=Path("pages"))

    assert (whole / "pages" / "1002.html.source.json").read_text(encoding="utf-8") == (
        narrow / "pages" / "1002.html.source.json"
    ).read_text(encoding="utf-8")


async def test_a_root_outside_the_configured_root_is_refused(tmp_path: Path) -> None:
    """Requirement 5. A source name is not a licence to write beside somebody else's files."""
    root = corpus(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "1003.html").write_text(page("1003"), encoding="utf-8")
    service, _ = await service_for(settings_for(root, profiles=[PROFILE]))

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(outside, source="docs")

    assert "outside" in str(refusal.value)
    assert not list(outside.glob("*.source.json")), "the refused run wrote into the other tree"


async def test_a_traversal_out_of_the_root_is_refused(tmp_path: Path) -> None:
    """The same guard reached the way an attacker would reach it, through ``..``.

    Containment is checked on the **resolved** path for exactly this: ``pages/../../elsewhere``
    is a path under the root as text and is not one as a location.
    """
    root = corpus(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "1003.html").write_text(page("1003"), encoding="utf-8")
    service, _ = await service_for(settings_for(root, profiles=[PROFILE]))

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(Path("pages/../../elsewhere"), source="docs")

    assert "outside" in str(refusal.value)
    assert not list(outside.glob("*.source.json"))


async def test_a_symlinked_subdirectory_is_refused_rather_than_followed(tmp_path: Path) -> None:
    """Never follow symlinks — and the link here points *inside* the root, deliberately.

    A link out of the tree would be caught by containment alone, so it would not show that this
    guard exists. One pointing at a real directory under the root is refused only because
    following links is refused, which is the rule the walk beneath already keeps.
    """
    root = corpus(tmp_path)
    (root / "shortcut").symlink_to(root / "pages", target_is_directory=True)
    service, _ = await service_for(settings_for(root, profiles=[PROFILE]))

    with pytest.raises(ConfigError) as refusal:
        await service.connector_sidecar(Path("shortcut"), source="docs")

    assert "symbolic link" in str(refusal.value)
    assert not list(root.glob("**/*.source.json"))


async def test_a_symlink_escaping_the_root_is_refused(tmp_path: Path) -> None:
    """The other direction, and it must be refused by *both* guards rather than by luck."""
    root = corpus(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "1003.html").write_text(page("1003"), encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    service, _ = await service_for(settings_for(root, profiles=[PROFILE]))

    with pytest.raises(ConfigError):
        await service.connector_sidecar(Path("escape"), source="docs")

    assert not list(outside.glob("*.source.json"))


async def test_a_symlinked_page_under_the_root_gets_no_manifest(tmp_path: Path) -> None:
    """The walk's own rule, asserted through the ``--source`` path.

    A conversion that followed this would write a manifest beside a file outside the tree — the
    one write primitive this design exists to make unrepresentable.
    """
    root = corpus(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "1003.html").write_text(page("1003"), encoding="utf-8")
    (root / "pages" / "link.html").symlink_to(outside / "1003.html")

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.written == 1, "the symlinked page was converted as well"
    assert not (outside / "1003.html.source.json").exists()
    assert not (root / "pages" / "link.html.source.json").exists()


async def test_a_page_id_that_looks_like_a_path_writes_nowhere_but_beside_the_page(
    tmp_path: Path,
) -> None:
    """Requirement 5's "never derive an output path from document-provided metadata".

    The page declares a traversal as its own id. It is not sanitised and not refused — it is
    simply never used as a path, because the only path that reaches the writer is
    ``manifest_path_for`` of the file the walk reached. The assertion is therefore about where
    the manifest landed, and the declared value travelling into the manifest *as data* is the
    proof that the hostile string was carried rather than dropped.
    """
    root = corpus(tmp_path, body=page("../../../../etc/cron.d/x"))
    victim = tmp_path.parent / "etc"

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.written == 1
    written = sorted(path.relative_to(root).as_posix() for path in root.glob("**/*.source.json"))
    assert written == ["pages/1002.html.source.json"]
    assert not victim.exists(), "a page id was joined to something"
    manifest = json.loads((root / "pages" / "1002.html.source.json").read_text(encoding="utf-8"))
    assert manifest["source_id"] == "../../../../etc/cron.d/x"


# --- the refusals the conversion itself makes, reached through a named source -------------------


async def test_an_ambiguous_page_is_refused_under_a_named_source(tmp_path: Path) -> None:
    """Two metadata sections is a page that cannot say which of them describes it."""
    root = corpus(tmp_path, body=page(sections=2))

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.written == 0
    assert report.by_outcome == {AdapterOutcome.AMBIGUOUS.value: 1}
    assert not list(root.glob("**/*.source.json"))


async def test_two_pages_declaring_one_id_get_no_manifest_either(tmp_path: Path) -> None:
    """Neither, because writing one would make ownership depend on walk order."""
    root = corpus(tmp_path)
    (root / "pages" / "copy.html").write_text(page(), encoding="utf-8")

    report = await convert(settings_for(root, profiles=[PROFILE]))

    assert report.written == 0
    assert report.by_outcome == {AdapterOutcome.DUPLICATE_IDENTITY.value: 2}
    assert not list(root.glob("**/*.source.json"))


# --- the remediation names a command that works, and this is how that is known ------------------


async def test_the_remediation_for_a_custom_source_names_the_source_form(tmp_path: Path) -> None:
    """Requirement 6. The command on the document is the one that resolves *these* profiles.

    Before this, an unapplied identity under a custom-profile source said ``manicule connector
    sidecar <root>`` — a command that reports ``no_profile`` for every page and writes nothing.
    An instruction that was written and never executed.
    """
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])

    record = only(await _synced(settings)).metadata[ENRICHED_KEY]

    assert isinstance(record, dict)
    assert record["outcome"] == AdapterOutcome.IDENTITY_NOT_APPLIED.value
    assert "manicule connector sidecar --source docs" in str(record["reason"])
    assert str(root) not in str(record["reason"]), "the root form is still being offered"


async def test_the_remediation_command_is_run_and_produces_a_manifest(tmp_path: Path) -> None:
    """The whole of requirement 6, and it is deliberately not a string comparison.

    A test asserting the text *contains* ``--source docs`` would have passed for every wrong
    command that happened to contain the right substring. So the command is taken off the
    document, parsed, and **executed** through the same service a person's shell would reach —
    and the assertion is that it wrote the manifest rather than reporting ``no_profile``.

    This is the guard for the defect the specification calls the heart of it: remediation text
    that names a command nobody ran.
    """
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    record = only(await _synced(settings)).metadata[ENRICHED_KEY]
    assert isinstance(record, dict)
    named = _command_in(str(record["reason"]))
    assert named[:3] == ["manicule", "connector", "sidecar"], named

    service, _ = await service_for(settings)
    root_argument, source_argument = _as_arguments(named)
    report = await service.connector_sidecar(root_argument, source=source_argument)

    assert report.written == 1, f"the command the document names wrote nothing: {report.skipped}"
    assert AdapterOutcome.NO_PROFILE.value not in report.by_outcome, (
        "the remediation named a command that reports no_profile, which is the defect itself"
    )
    assert (root / "pages" / "1002.html.source.json").is_file()


async def test_a_default_profile_source_is_told_to_use_the_root_form(tmp_path: Path) -> None:
    """The other branch, and it must not be ``--source`` for everything.

    ``manicule index <path>`` builds a connector with no configured name, so ``--source`` would
    name something no configuration resolves — an error the operator cannot act on at all, which
    is worse than the root form that actually works for the default profile.
    """
    root = corpus(
        tmp_path,
        body=page(
            metadata_attribute="data-source-metadata",
            body_attribute='data-document-representation="storage"',
        ),
    )
    connector = FilesystemConnector(root, name="local")

    assert connector.sidecar_command == f"manicule connector sidecar {root.resolve()}"
    assert "--source" not in connector.sidecar_command


def _command_in(reason: str) -> list[str]:
    """The backtick-quoted command a reason names, as argv.

    Parsed out of the prose rather than reconstructed, because reconstructing it is how a test
    comes to check the command it built instead of the one that shipped.
    """
    import shlex  # noqa: PLC0415

    quoted = [part for part in reason.split("`") if part.startswith("manicule connector sidecar")]
    assert quoted, f"no sidecar command is named in: {reason!r}"
    return shlex.split(quoted[0])


def _as_arguments(argv: list[str]) -> tuple[Path | None, str]:
    """``argv`` for ``manicule connector sidecar`` as the root and source it stands for."""
    rest = argv[3:]
    if rest[:1] == ["--source"]:
        return None, rest[1]
    return Path(rest[0]), ""


async def test_an_existing_manifest_survives_without_force(tmp_path: Path) -> None:
    """And ``--force`` is what replaces it, so the refusal is a choice rather than a wall."""
    root = corpus(tmp_path)
    settings = settings_for(root, profiles=[PROFILE])
    manifest = root / "pages" / "1002.html.source.json"
    manifest.write_text('{"source_id": "written-by-hand"}\n', encoding="utf-8")

    report = await convert(settings)

    assert report.written == 0
    assert report.by_outcome == {AdapterOutcome.ALREADY_PRESENT.value: 1}
    assert json.loads(manifest.read_text(encoding="utf-8"))["source_id"] == "written-by-hand"

    forced = await convert(settings, force=True)

    assert forced.written == 1
    assert json.loads(manifest.read_text(encoding="utf-8"))["source_id"] == "1002"
