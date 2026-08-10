"""Grammar packaging: the declared set, the pre-seed, and the refusal.

The failure this suite exists to prevent is not a crash. It is a repository that chunks one
way on a machine that reached the network and another way on a machine that did not — one
corpus with two chunkings, none of it raising anything. Every test here is a guard on one
step of that path, and each has its negative: a set validated against nothing, a refusal that
turns out to be a fallback, a fingerprint that changes when the cache does.

**Nothing here needs network access.** Every test that must observe an absent grammar points
the cache at an empty directory *and* the manifest at an address that cannot be reached, so
that a test which accidentally starts downloading fails loudly instead of passing slowly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from importlib.metadata import metadata
from pathlib import Path

import pytest

from manicule.core.errors import ConfigError, ParseError
from manicule.parsers import grammars

UNREACHABLE_MANIFEST = "http://127.0.0.1:9/manicule-tests-must-not-download.json"
"""Discard port on the loopback interface. A fetch attempted through this refuses at once
rather than hanging, so a test that starts downloading fails rather than stalling CI."""

PERMISSIVE_LICENCES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"})


@pytest.fixture(autouse=True)
def restore_pack_configuration() -> Iterator[None]:
    """Put the pack back on its default cache and manifest after every test.

    The pack keeps one registry per process, so a test that points it at a temporary
    directory would otherwise decide what every later test can see.
    """
    yield
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)


@pytest.fixture
def empty_cache(tmp_path: Path) -> Path:
    """A cache directory with no grammars in it, and no route to fetch any."""
    cache = tmp_path / "grammars"
    cache.mkdir()
    grammars.configure_pack(
        grammars.DECLARED_LANGUAGES, cache_dir=cache, manifest_url=UNREACHABLE_MANIFEST
    )
    return cache


def test_a_language_key_is_checked_against_the_manifest_rather_than_at_first_use() -> None:
    """A typo has to be a startup error.

    Checked at first use instead, it is a document that mysteriously never parses on one
    machine — and by then a corpus has been indexed without it.
    """
    with pytest.raises(ConfigError) as raised:
        grammars.validate_languages(["python", "c_sharp"])

    assert "c_sharp" in str(raised.value)


def test_the_key_for_c_sharp_is_csharp_and_the_error_says_so() -> None:
    """The one naming trap in the pack, and the only defence is the message.

    ``c_sharp`` is a plausible spelling that is simply not a key, and it fails at lookup with
    nothing to go on unless the error offers the near misses.
    """
    with pytest.raises(ConfigError) as raised:
        grammars.validate_languages(["c_sharp"])

    assert "csharp" in str(raised.value)
    assert grammars.validate_languages(["csharp"]) == ("csharp",)


def test_a_real_grammar_manicule_does_not_declare_is_refused_with_its_own_message() -> None:
    """ "Not a grammar" and "not one of ours" are different mistakes with different fixes.

    Haskell is a real key in the pack's manifest. Declaring it is not a typo; it is asking
    for a language manicule routes no media type to, and saying "unknown key" would send
    somebody looking for a spelling error that is not there.
    """
    with pytest.raises(ConfigError) as raised:
        grammars.validate_languages(["python", "haskell"])

    message = str(raised.value)
    assert "haskell" in message
    assert "media type" in message


def test_declaring_no_languages_at_all_is_refused() -> None:
    """An empty set is a parser nothing routes to, which is a misconfiguration that looks
    exactly like a parser nobody is using."""
    with pytest.raises(ConfigError):
        grammars.validate_languages([])


def test_the_declared_set_is_a_value_rather_than_an_order() -> None:
    """Two configurations naming the same languages describe the same corpus.

    The set feeds ``ChunkFingerprint.grammars``; if the order it was written in survived, a
    reordered configuration file would read as a different chunking process.
    """
    assert grammars.validate_languages(["rust", "python", "rust"]) == ("python", "rust")


def test_every_declared_language_routes_from_exactly_one_media_type() -> None:
    """One language, one media type, and no two languages sharing one.

    A collision would make routing depend on dictionary order, which is the kind of thing
    that works until the table is edited.
    """
    media_types = list(grammars.MEDIA_TYPES.values())

    assert len(set(media_types)) == len(media_types)
    for language, media_type in grammars.MEDIA_TYPES.items():
        assert grammars.language_for_media_type(media_type) == language
    assert grammars.language_for_media_type("application/pdf") is None


def test_the_declared_set_claims_no_media_type_another_parser_owns() -> None:
    """The pack has grammars for HTML, JSON, YAML and Markdown, and all four are somebody
    else's documents (``docs/parsing.md`` §2.4).

    Declaring them here would put two parsers on one media type and let registration order
    decide which one runs.
    """
    claimed = set(grammars.MEDIA_TYPES.values())

    assert claimed.isdisjoint(
        {"text/html", "application/json", "application/yaml", "text/markdown", "text/x-toml"}
    )


def test_an_absent_grammar_is_detected_without_reaching_the_network(empty_cache: Path) -> None:
    """Absence has to be answerable offline, or the check that prevents a download is one.

    The manifest points at an address that refuses connections, so a lookup that decided to
    fetch would fail rather than quietly succeed on a machine with egress.
    """
    assert not list(empty_cache.iterdir())
    assert grammars.missing_grammars(["python", "rust"]) == ("python", "rust")
    assert grammars.is_available("python") is False


def test_a_missing_grammar_refuses_and_names_the_command_that_fixes_it(
    empty_cache: Path,
) -> None:
    """The refusal is the whole mechanism, so its message is part of it.

    ``unsupported_media_type`` with no detail tells an operator nothing to do; the reason
    string names the language and the command, and it is what a document's
    ``status_detail`` carries.
    """
    del empty_cache

    with pytest.raises(grammars.GrammarUnavailableError) as raised:
        grammars.load_parser("python")

    assert raised.value.language == "python"
    assert raised.value.reason == "grammar unavailable: python — run manicule doctor --fix"


def test_a_missing_grammar_is_not_a_parse_error(empty_cache: Path) -> None:
    """This is the guard, and it is one word wide.

    ``ParseError`` means "not my kind of document" and hands the file to the next parser in
    the chain — which for source code is the plain-text parser, which line-splits it. A
    missing grammar that declined would therefore produce exactly the two-machine chunking
    the declared set exists to prevent, while every other test here still passed.
    """
    del empty_cache

    with pytest.raises(grammars.GrammarUnavailableError) as raised:
        grammars.load_parser("rust")

    assert not isinstance(raised.value, ParseError)


def test_tags_queries_resolve_with_no_grammar_and_no_network(empty_cache: Path) -> None:
    """Symbols must not depend on what a machine managed to fetch.

    There are no ``.scm`` files in the wheel — the patterns are compiled into the same native
    library that contains no grammars — which is exactly why they answer here, with an empty
    cache and an unreachable manifest, while ``load_parser`` refuses.
    """
    del empty_cache
    source = grammars.tags_query_source("python")

    assert source is not None
    assert "@definition.function" in source
    assert grammars.tags_query_source("sql") is None


def test_an_unreachable_manifest_is_reported_with_the_url_it_tried(empty_cache: Path) -> None:
    """The air-gapped case, which is the one where a raw download error is useless.

    An operator whose mirror is not configured needs to see that the public manifest was
    tried; "connection refused" alone sends them to their firewall.
    """
    del empty_cache

    with pytest.raises(grammars.GrammarFetchError) as raised:
        grammars.prefetch(["python"])

    assert UNREACHABLE_MANIFEST in str(raised.value)
    assert grammars.MANIFEST_URL_ENV in str(raised.value)


def test_a_prefetch_that_wrote_nothing_is_a_failure_however_it_reported_itself(
    tmp_path: Path,
) -> None:
    """The pre-seed that succeeds and leaves the cache empty.

    The grammar pack keeps a process-global registry, and its own ``prefetch`` returns
    without error for a language already in that registry **even when the configured cache
    directory does not contain it**. A container build that parses one Python file and then
    pre-seeds into the image directory is therefore told it succeeded, and ships an image
    with no grammars in it — which surfaces on an air-gapped host as a refusal to parse code
    that worked on the machine that built the image.

    The load happens first and against the real cache, because that is the order the hazard
    occurs in and the reason it is not caught by the fixtures elsewhere in this file. Nothing
    here reaches the network: the manifest is pointed at the discard port, and the failure
    being guarded against is a *silent success* rather than a download.
    """
    if not grammars.is_available("python"):
        pytest.skip("no python grammar cached, so the pack cannot be warmed without a fetch")
    grammars.load_parser("python")

    empty = tmp_path / "grammars"
    empty.mkdir()
    grammars.configure_pack(
        grammars.DECLARED_LANGUAGES, cache_dir=empty, manifest_url=UNREACHABLE_MANIFEST
    )

    with pytest.raises(grammars.GrammarFetchError) as raised:
        grammars.prefetch(["python"])

    assert "still not in the cache" in str(raised.value)
    assert str(empty) in str(raised.value)
    assert not grammars.is_available("python")


def test_prefetch_does_nothing_when_every_declared_grammar_is_already_cached() -> None:
    """Pre-seeding is meant to be run on every start.

    It is only cheap enough for that if the steady state is a directory listing, so a
    prefetch that fetched something here would mean ``manicule init`` re-downloads the world
    on every container start.
    """
    available = [
        language for language in grammars.DECLARED_LANGUAGES if grammars.is_available(language)
    ]
    if not available:
        pytest.skip(
            "no grammars are cached in this environment, so the already-cached path cannot "
            "be observed; run `manicule doctor --fix` (or prefetch the declared set) first"
        )

    assert grammars.prefetch(available) == ()


def test_grammar_versions_describe_the_declared_set_and_not_the_cache(empty_cache: Path) -> None:
    """``ChunkFingerprint.grammars`` must not depend on what a machine happens to hold.

    If the map shrank when a grammar was absent, the fingerprint would differ between a
    freshly installed machine and a warmed one, and the corpus each built would be declared
    incompatible with the other for no reason at all.
    """
    del empty_cache
    versions = grammars.grammar_versions(["python", "rust"])

    assert versions == {"python": grammars.pack_version(), "rust": grammars.pack_version()}


def test_a_grammar_version_is_not_a_hash_of_a_platform_specific_binary() -> None:
    """Same declared set, same answer, on every platform.

    A hash of the compiled grammar would be genuinely per-language and would also differ
    between macOS and Linux for identical grammar source, so moving a corpus between machines
    would invalidate it. The recorded version is the pack release, which describes the
    bundle every platform's grammars are built from.
    """
    version = grammars.pack_version()

    assert version == metadata("tree-sitter-language-pack")["Version"]
    assert set(grammars.grammar_versions(grammars.DECLARED_LANGUAGES).values()) == {version}


def test_the_manifest_override_is_removed_again_when_it_is_not_asked_for(
    empty_cache: Path,
) -> None:
    """One call must not leave the previous call's mirror in force.

    The override is an environment variable because that is the only hook the pack exposes,
    and an environment variable is exactly the kind of state that outlives the thing that
    set it.
    """
    del empty_cache
    assert os.environ[grammars.MANIFEST_URL_ENV] == UNREACHABLE_MANIFEST

    grammars.configure_pack(grammars.DECLARED_LANGUAGES)

    assert grammars.MANIFEST_URL_ENV not in os.environ


def test_the_grammar_pack_is_still_distributed_under_a_permissive_licence() -> None:
    """Upstream's stated policy is permissive-only; this asserts it rather than trusting it.

    It is the strongest assertion the installed artifact supports. **The wheel enumerates no
    per-grammar licences** — the manifest carries a group and a size per language and nothing
    else, and the CycloneDX SBOM beside it describes the Rust build dependencies of the
    native extension rather than the 371 grammars. A per-grammar audit therefore cannot be
    written against what is installed; it belongs with the offline bundle, which is where the
    grammars themselves would be enumerated.
    """
    declared = metadata("tree-sitter-language-pack")["License-Expression"]

    assert declared in PERMISSIVE_LICENCES
