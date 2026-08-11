"""The offline grammar bundle: an install that parses code with nothing to fetch from.

``tests/parsers/test_grammars.py`` proves the declared set, the pre-seed and the refusal.
Everything there assumes the pre-seed can *reach* something. This suite is the case where it
cannot: no route to the grammar release, no mirror, and a grammar cache that has never been
written to.

Two conditions have to be separable, and they are asserted separately and together, because
the last serious bug in this area came from exercising only one of them. **A cache redirect
hid the grammars on macOS**: the pack resolves its cache through ``platformdirs``, which honours
``XDG_CACHE_HOME`` on Linux and uses ``~/Library/Caches`` on macOS, so a suite that redirected
the XDG variable proved something on the Linux runner and nothing at all on the platform the
project targets. So:

- **An empty cache** is exercised in-process, by configuring the pack at a temporary directory.
- **A cache that resolves somewhere unexpected** is exercised in a subprocess with ``HOME`` and
  ``XDG_CACHE_HOME`` both moved, which is the only way to make ``platformdirs`` answer
  differently on every platform at once. Whichever convention this machine uses, the child's
  cache is a directory that has never existed, and the child prints where it landed so the
  redirect is *shown* to have taken rather than assumed.

Both run with the manifest pointed at the discard port, so a case that quietly starts
downloading fails instead of passing slowly against the real release.

Every bundle here is built by the shipped builder from this machine's own grammars. Nothing
hand-writes a manifest: a reader tested against a fixture's idea of the format is a format with
two definitions and no failing test.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from manicule.core.errors import ConfigError
from manicule.parsers import grammar_bundle, grammars
from tests.grammar_support import (
    BUNDLE_LANGUAGES,
    NAME_TRAP_LANGUAGE,
    UNREACHABLE_MANIFEST,
    build_bundle,
    require_source_grammars,
    source_cache,
)

CODE = b"class Store:\n    def refresh(self):\n        return 1\n"
"""A file every bundled Python grammar must produce the same tree for."""


@pytest.fixture(autouse=True)
def restore_pack_configuration() -> Iterator[None]:
    """Put the pack back on its default cache and manifest after every test.

    The pack keeps one registry per process, so a test that points it at a temporary directory
    would otherwise decide what every later test can see.
    """
    yield
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> grammar_bundle.GrammarBundle:
    """One real bundle for the whole module, built from this machine's grammars.

    Module-scoped because building copies compiled libraries and nothing that reads a bundle
    modifies it. The tests that *do* modify one copy it first, so a corruption case cannot
    decide what a later test sees.
    """
    return build_bundle(tmp_path_factory.mktemp("bundle"))


@pytest.fixture
def empty_cache(tmp_path: Path) -> Path:
    """A cache directory with no grammars in it, and no route to fetch any."""
    cache = tmp_path / "cache"
    cache.mkdir()
    grammars.configure_pack(
        grammars.DECLARED_LANGUAGES, cache_dir=cache, manifest_url=UNREACHABLE_MANIFEST
    )
    return cache


# --- the air-gapped install ---------------------------------------------------------------


def test_a_bundle_seeds_an_empty_cache_with_no_route_to_the_network(
    bundle: grammar_bundle.GrammarBundle, empty_cache: Path
) -> None:
    """The whole feature, in one test: install, no network, no grammars, and code parses.

    The manifest points at a port that refuses connections, so a pre-seed that reached for the
    network here would raise rather than succeed slowly — which means the grammars that end up
    in the cache can only have come out of the bundle.
    """
    assert not list(empty_cache.iterdir())
    assert grammars.missing_grammars(BUNDLE_LANGUAGES) == BUNDLE_LANGUAGES

    seeded = grammars.prefetch(BUNDLE_LANGUAGES, bundle_dir=bundle.root)

    assert seeded == BUNDLE_LANGUAGES
    assert grammars.missing_grammars(BUNDLE_LANGUAGES) == ()
    assert grammars.load_parser("python").parse(CODE).root_node.type == "module"


def test_seeding_is_a_copy_out_of_the_bundle_rather_than_a_move_into_it(
    bundle: grammar_bundle.GrammarBundle, empty_cache: Path
) -> None:
    """A bundle survives being used, and is not the cache.

    A bundle installed under ``site-packages`` is read-only on any sensibly built image, and
    one that got consumed by the first machine to seed from it could not be shared by an image
    layer or a read-only mount.
    """
    grammars.prefetch(BUNDLE_LANGUAGES, bundle_dir=bundle.root)

    for language in BUNDLE_LANGUAGES:
        entry = bundle.grammars[language]
        assert bundle.path_for(language).is_file()
        assert (empty_cache / entry.filename).read_bytes() == bundle.path_for(language).read_bytes()


def test_a_second_pre_seed_copies_nothing(
    bundle: grammar_bundle.GrammarBundle, empty_cache: Path
) -> None:
    """Pre-seeding is meant to be run on every start, which it can only be if it is cheap.

    A bundle that re-copied its libraries on every call would put tens of megabytes of I/O in
    front of every container start, and would look identical from the outside.
    """
    del empty_cache
    grammars.prefetch(BUNDLE_LANGUAGES, bundle_dir=bundle.root)

    assert grammars.prefetch(BUNDLE_LANGUAGES, bundle_dir=bundle.root) == ()


def test_a_bundle_serves_a_cache_that_resolves_somewhere_unexpected(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """The macOS incident, as a test, on whatever platform is running it.

    ``HOME`` and ``XDG_CACHE_HOME`` are both moved to a directory that has never existed, so
    the pack's own cache resolution — ``~/Library/Caches`` here, ``$XDG_CACHE_HOME`` on Linux —
    lands somewhere empty whichever convention it follows. The child reports the path it
    resolved, and this asserts it is under the redirect: a redirect that silently failed would
    otherwise leave the child reading the developer's real cache and passing for the wrong
    reason, which is exactly what happened last time.

    In a subprocess because the redirect has to happen before the pack computes anything, and
    because a process that has already loaded a grammar cannot honestly be asked what a fresh
    install would do.
    """
    home = tmp_path / "elsewhere"
    result = _air_gapped_child(
        home,
        _SEED_AND_PARSE,
        extra={grammar_bundle.BUNDLE_DIR_ENV: str(bundle.root)},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert str(home) in report["cache_dir"]
    assert report["missing_before"] == list(BUNDLE_LANGUAGES)
    assert report["seeded"] == list(BUNDLE_LANGUAGES)
    assert report["missing_after"] == []
    assert report["parsed"] == "module"


def test_the_same_install_without_the_bundle_refuses_loudly(tmp_path: Path) -> None:
    """Disable the bundle and the air-gapped install must fail, not degrade.

    The negative of the test above, and the reason that one is evidence: identical child,
    identical redirect, identical unreachable manifest, and the bundle taken away. If this
    passed, the previous test would be proving that a redirected cache still finds the
    developer's grammars rather than proving anything about a bundle.

    What it must not do is succeed with a line-splitting fallback. The failure names the
    languages, the manifest it tried, and that no bundle was found — the three things an
    operator on an air-gapped host has to know.
    """
    result = _air_gapped_child(tmp_path / "elsewhere", _SEED_AND_PARSE)

    assert result.returncode != 0
    assert "GrammarFetchError" in result.stderr
    assert "no offline grammar bundle is installed" in result.stderr
    assert grammar_bundle.BUNDLE_DIR_ENV in result.stderr
    assert "python" in result.stderr


def test_a_bundle_installed_as_a_distribution_needs_no_configuration(
    tmp_path: Path,
) -> None:
    """The bundle arrives through the install channel rather than as a directory to copy.

    The builder's ``--package`` mode writes an importable distribution around the bundle, and
    the child here is given nothing but that distribution on its path — no environment
    variable, no configured directory. An air-gapped host that can install manicule at all can
    therefore install its grammars, which is the only shape that does not depend on somebody
    remembering to copy a directory to the right place.
    """
    from tools.build_grammar_bundle import main  # noqa: PLC0415 - a build script, not runtime

    require_source_grammars()
    package = tmp_path / "dist"
    exit_code = main(["--output", str(package), "--package", "--languages", *BUNDLE_LANGUAGES])
    assert exit_code == 0

    result = _air_gapped_child(
        tmp_path / "elsewhere",
        _SEED_AND_PARSE,
        extra={"PYTHONPATH": str(package / "src")},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["seeded"] == list(BUNDLE_LANGUAGES)
    assert report["parsed"] == "module"


def test_a_read_only_bundle_is_a_cache_directory_in_its_own_right(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """A container with a read-only filesystem needs no copy at all.

    The bundle's library directory is laid out as the pack's own cache is, so a deployment that
    cannot write anywhere can point the pack straight at it. Asserted with the directory
    actually made read-only, because "we never write to it" is a claim about code and this is a
    claim about the filesystem.
    """
    read_only = tmp_path / "readonly"
    shutil.copytree(bundle.root, read_only)
    _make_read_only(read_only)
    try:
        grammars.configure_pack(
            grammars.DECLARED_LANGUAGES,
            cache_dir=read_only / grammar_bundle.LIBRARY_DIR_NAME,
            manifest_url=UNREACHABLE_MANIFEST,
        )

        assert grammars.missing_grammars(BUNDLE_LANGUAGES) == ()
        assert grammars.load_parser("rust").parse(b"fn main() {}\n").root_node.type == "source_file"
    finally:
        _make_writable(read_only)


# --- what the bundle records ---------------------------------------------------------------


def test_the_bundle_records_the_release_it_was_built_from(
    bundle: grammar_bundle.GrammarBundle,
) -> None:
    """Which grammars built a corpus must be knowable, not inferred from a directory listing.

    ``ChunkFingerprint.grammars`` records the pack release, and an offline install has no
    manifest to consult and no download to infer it from — so the bundle carries it. The
    per-language digest is recorded for the same reason: a bundle is copied between machines by
    hand, and "the same bundle" has to mean the same bytes.
    """
    manifest = json.loads((bundle.root / grammar_bundle.MANIFEST_NAME).read_text())

    assert manifest["pack_version"] == grammars.pack_version()
    assert manifest["platform"] == grammar_bundle.platform_tag()
    assert manifest["schema_version"] == grammar_bundle.SCHEMA_VERSION
    assert sorted(manifest["languages"]) == sorted(BUNDLE_LANGUAGES)
    for language in BUNDLE_LANGUAGES:
        recorded = manifest["languages"][language]
        original = source_cache() / recorded["filename"]
        assert original.read_bytes() == bundle.path_for(language).read_bytes()
        assert recorded["size"] == original.stat().st_size


def test_the_recorded_release_is_the_installed_one_rather_than_a_cache_path(
    bundle: grammar_bundle.GrammarBundle,
) -> None:
    """The version is a fact about the install, and the two sources can disagree.

    The pack's default cache directory happens to contain its release in the path, so a build
    that scraped the version out of it would look right until somebody built from a cache
    directory named anything else — a container image's ``/opt/grammars``, for instance — and
    then a bundle would claim a release nobody has.
    """
    assert bundle.pack_version == grammars.pack_version()
    assert bundle.pack_version not in {"", "unknown"}


def test_a_bundle_built_for_another_pack_release_is_refused(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """Grammars ship as one bundle per release, so mixing releases mixes parse trees.

    Silently accepted, this is the corpus-consistency hazard in its purest form: the fingerprint
    records the installed release while the grammars producing the trees came from another one,
    so two machines agree on the recorded version and disagree on the chunks.
    """
    other = _edited_manifest(bundle, tmp_path, pack_version="0.0.0-not-installed")

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(other)

    assert "0.0.0-not-installed" in str(raised.value)
    assert grammars.pack_version() in str(raised.value)


def test_a_bundle_built_for_another_platform_is_refused(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """Grammar libraries are compiled objects, and the pack reports one as present regardless.

    A bundle copied from an x86 build host to an Apple Silicon one contains files with the right
    names and the wrong machine code. Without this check they are copied into the cache,
    reported as downloaded, and fail at load with "language not found" — a message that reads as
    a routing bug on the wrong machine entirely.
    """
    other = _edited_manifest(bundle, tmp_path, platform="linux-s390x")

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(other)

    assert "linux-s390x" in str(raised.value)
    assert grammar_bundle.platform_tag() in str(raised.value)


def test_a_bundle_from_a_later_manifest_schema_is_refused(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """A manifest this manicule cannot read is refused rather than read optimistically."""
    other = _edited_manifest(bundle, tmp_path, schema_version=grammar_bundle.SCHEMA_VERSION + 1)

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(other)

    assert str(grammar_bundle.SCHEMA_VERSION) in str(raised.value)


def test_a_directory_that_is_not_a_bundle_says_so(tmp_path: Path) -> None:
    """A configured path that is empty, or a typo, must not read as "no bundle configured"."""
    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(tmp_path)

    assert grammar_bundle.MANIFEST_NAME in str(raised.value)
    assert "tools/build_grammar_bundle.py" in str(raised.value)


# --- integrity ------------------------------------------------------------------------------


def test_a_truncated_library_is_refused_when_the_bundle_is_read(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """Half a shared library is reported by the pack as a language it has.

    Caught on read rather than at load, because by load time the file has already been copied
    into the cache and the machine believes it has a grammar. A ``stat`` per language is what
    that costs.
    """
    damaged = _copied_bundle(bundle, tmp_path)
    library = damaged / grammar_bundle.LIBRARY_DIR_NAME / bundle.grammars["python"].filename
    library.write_bytes(library.read_bytes()[:1024])

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(damaged)

    assert "python" in str(raised.value)
    assert "1024" in str(raised.value)


def test_a_library_missing_from_the_bundle_is_refused_when_it_is_read(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path
) -> None:
    """A manifest is a claim about files; the files are checked against it."""
    damaged = _copied_bundle(bundle, tmp_path)
    (damaged / grammar_bundle.LIBRARY_DIR_NAME / bundle.grammars["rust"].filename).unlink()

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(damaged)

    assert "rust" in str(raised.value)
    assert "missing" in str(raised.value)


def test_a_library_edited_without_changing_its_length_is_refused_when_it_is_seeded(
    bundle: grammar_bundle.GrammarBundle, empty_cache: Path, tmp_path: Path
) -> None:
    """The case a size check cannot see, and the reason the digest is recorded.

    A bundle is moved between machines by whatever is available — a USB stick, an rsync, an
    image layer — and a flipped byte in a shared library is not a crash, it is undefined
    behaviour inside a parser. The copy has to read every byte anyway, so hashing it there is
    free and is where the check belongs.
    """
    damaged = _copied_bundle(bundle, tmp_path)
    library = damaged / grammar_bundle.LIBRARY_DIR_NAME / bundle.grammars["python"].filename
    content = bytearray(library.read_bytes())
    content[-1] ^= 0xFF
    library.write_bytes(bytes(content))

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.read(damaged).seed(["python"], empty_cache)

    assert "sha256" in str(raised.value)
    assert not list(empty_cache.iterdir())


def test_a_grammar_that_cannot_be_loaded_refuses_instead_of_declining(tmp_path: Path) -> None:
    """The gap a bundle opens, closed: present, reported as downloaded, and unloadable.

    ``downloaded_languages()`` answers from file names, so anything with the right name is a
    language this machine has. Left unwrapped, the pack then raises ``Language 'python' not
    found`` — which reaches the parser chain as an ordinary exception and **advances it**, so
    the document is handed to whatever comes next. For source code that is the plain-text
    parser, and a line-split file is exactly the outcome the declared language set exists to
    prevent.

    Asserted as *not* a ``ParseError`` and *is* a ``GrammarUnavailableError``, so a caller
    already written to stop on a missing grammar stops on a broken one without being changed.

    **In a subprocess, and the reason is itself the bug being guarded against.** The grammar
    pack keeps one registry per process and ``get_parser`` answers from it, so a process that
    has already loaded Python hands back the working grammar however broken the file on disk
    is. In-process this test passes while checking nothing — the same shape as the pre-seed
    that reports success against an empty cache.
    """
    result = _air_gapped_child(tmp_path / "elsewhere", _PLANT_A_BROKEN_LIBRARY)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["reported_as_present"] is True
    assert report["type"] == "GrammarUnusableError"
    assert report["is_parse_error"] is False
    assert report["is_grammar_unavailable"] is True
    assert report["reason"] == "grammar unusable: python — run manicule doctor --fix"
    assert "another platform" in report["message"]


# --- resolution -----------------------------------------------------------------------------


def test_an_explicitly_named_bundle_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that named a path meant that path.

    The reverse order would make a deployment-wide variable silently override a per-call
    argument, which is how a container ends up seeding from the image's bundle when the
    operator asked for the one they just copied in.
    """
    monkeypatch.setenv(grammar_bundle.BUNDLE_DIR_ENV, str(tmp_path / "from-environment"))

    assert grammar_bundle.locate(tmp_path / "named") == tmp_path / "named"
    assert grammar_bundle.locate() == tmp_path / "from-environment"


def test_no_bundle_anywhere_is_not_an_error_but_says_where_it_looked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An install with no bundle is ordinary; it fetches. The message is what makes it fixable.

    ``None`` rather than an exception, because raising here would make every networked install
    fail for lacking something it does not need.
    """
    monkeypatch.delenv(grammar_bundle.BUNDLE_DIR_ENV, raising=False)
    monkeypatch.setattr(grammar_bundle, "_installed_bundle", lambda: None)

    assert grammar_bundle.resolve() is None
    described = grammar_bundle.describe_search_path()
    assert grammar_bundle.BUNDLE_DIR_ENV in described
    assert grammar_bundle.BUNDLE_MODULE in described


def test_a_bundle_that_is_present_and_unusable_is_never_reported_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one downgrade that must not happen.

    A bundle that is present and wrong, reported as "no bundle", sends an operator to configure
    a mirror they do not have while the answer is sitting on the disk in front of them.
    """
    monkeypatch.setenv(grammar_bundle.BUNDLE_DIR_ENV, str(tmp_path))

    with pytest.raises(grammar_bundle.GrammarBundleError):
        grammar_bundle.resolve()
    with pytest.raises(grammar_bundle.GrammarBundleError):
        grammars.bundle_status()


def test_the_pre_seed_failure_names_the_bundle_as_well_as_the_mirror(
    empty_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Connection refused" alone sends an operator to their firewall.

    On an air-gapped host the actionable fact is almost never the network. It is that there is
    no bundle, or that the bundle does not carry the language being asked for.
    """
    del empty_cache
    monkeypatch.delenv(grammar_bundle.BUNDLE_DIR_ENV, raising=False)
    monkeypatch.setattr(grammar_bundle, "_installed_bundle", lambda: None)

    with pytest.raises(grammars.GrammarFetchError) as raised:
        grammars.prefetch(["python"])

    message = str(raised.value)
    assert UNREACHABLE_MANIFEST in message
    assert "no offline grammar bundle is installed" in message
    assert "docs/parsing.md" in message


def test_a_bundle_missing_the_language_being_asked_for_says_which_it_has(
    bundle: grammar_bundle.GrammarBundle, empty_cache: Path
) -> None:
    """A partial bundle must not read as a total one.

    Seeding what it can and then failing on the rest is right — the languages it carries are
    genuinely available — but the failure has to name the gap, or an operator concludes the
    bundle is broken rather than incomplete.
    """
    del empty_cache

    with pytest.raises(grammars.GrammarFetchError) as raised:
        grammars.prefetch([*BUNDLE_LANGUAGES, "go"], bundle_dir=bundle.root)

    message = str(raised.value)
    assert "'go'" in message
    assert "python" in message
    assert grammars.missing_grammars(BUNDLE_LANGUAGES) == ()


def test_the_status_line_describes_what_an_air_gapped_host_can_do(
    bundle: grammar_bundle.GrammarBundle,
) -> None:
    """What ``doctor`` prints. Three facts, because "no grammars" has three different fixes."""
    status = grammars.bundle_status(bundle.root)

    assert str(bundle.root) in status
    assert grammars.pack_version() in status
    assert grammar_bundle.platform_tag() in status
    for language in BUNDLE_LANGUAGES:
        assert language in status


# --- building -------------------------------------------------------------------------------


def test_the_builder_discovers_a_library_name_it_could_not_have_guessed(
    tmp_path: Path,
) -> None:
    """C#'s grammar is ``libtree_sitter_c_sharp``, and its language key is ``csharp``.

    A builder deriving the file name from the key writes a bundle silently missing C#, and the
    air-gapped host refuses every ``.cs`` file while every other language works — which reads
    as a grammar problem rather than a packaging one. The name is asked of the pack instead:
    each candidate library is offered alone in an empty directory and the pack says which
    language it answers for.
    """
    require_source_grammars((NAME_TRAP_LANGUAGE,))
    built = build_bundle(tmp_path / "bundle", (NAME_TRAP_LANGUAGE,))

    assert built.grammars[NAME_TRAP_LANGUAGE].filename != f"libtree_sitter_{NAME_TRAP_LANGUAGE}.so"
    assert "c_sharp" in built.grammars[NAME_TRAP_LANGUAGE].filename
    assert built.path_for(NAME_TRAP_LANGUAGE).is_file()


def test_a_bundle_cannot_be_built_for_a_language_the_source_lacks(tmp_path: Path) -> None:
    """A build that half-succeeds is a host that refuses one language for no visible reason."""
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(
        source_cache() / f"libtree_sitter_python{grammar_bundle.library_suffix()}",
        source / f"libtree_sitter_python{grammar_bundle.library_suffix()}",
    )

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.build(["python", "rust"], tmp_path / "bundle", source=source)

    assert "rust" in str(raised.value)
    assert not (tmp_path / "bundle" / grammar_bundle.MANIFEST_NAME).exists()


def test_a_bundle_cannot_be_built_from_an_empty_source(tmp_path: Path) -> None:
    """The cache nobody pre-seeded. Named as such, rather than as a missing language."""
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.build(["python"], tmp_path / "bundle", source=source)

    assert "Pre-seed" in str(raised.value)


def test_a_bundle_cannot_be_built_for_a_language_manicule_does_not_declare(
    tmp_path: Path,
) -> None:
    """The declared set governs the bundle too, or a bundle becomes a second declaration."""
    with pytest.raises(ConfigError):
        grammar_bundle.build(["haskell"], tmp_path / "bundle", source=source_cache())


def test_building_a_bundle_leaves_the_pack_where_it_found_it(
    tmp_path: Path, empty_cache: Path
) -> None:
    """Discovery repoints the pack two dozen times, and the pack has one registry per process.

    Restoring *what was in force* rather than the default matters here: a container that
    configured an image-local cache and then built a bundle would otherwise be moved back to
    the per-user cache, and would spend the rest of its life looking for grammars where it has
    never put any.
    """
    shutil.copy2(
        source_cache() / f"libtree_sitter_python{grammar_bundle.library_suffix()}",
        empty_cache / f"libtree_sitter_python{grammar_bundle.library_suffix()}",
    )
    grammars.configure_pack(
        grammars.DECLARED_LANGUAGES, cache_dir=empty_cache, manifest_url=UNREACHABLE_MANIFEST
    )

    grammar_bundle.build(["python"], tmp_path / "bundle", source=empty_cache)

    assert grammars.cache_directory() == empty_cache
    assert os.environ[grammars.MANIFEST_URL_ENV] == UNREACHABLE_MANIFEST


def test_the_bundle_asserts_the_licence_rather_than_trusting_the_policy(
    bundle: grammar_bundle.GrammarBundle,
) -> None:
    """A bundle is redistributed, so its licence is asserted at the moment it is built.

    Upstream states that every grammar it carries is permissive and that copyleft is not
    accepted. The installed pack's own expression is the strongest statement the artifact
    supports — it enumerates no per-grammar licences — and it is checked and recorded rather
    than assumed, so a release that changes it fails a build instead of shipping.
    """
    assert bundle.licence in grammar_bundle.PERMISSIVE_LICENCES
    assert grammar_bundle.check_licence(grammar_bundle.licence_of_installed_pack())


@pytest.mark.parametrize("expression", ["GPL-3.0-only", "MIT AND LGPL-2.1", "AGPL-3.0-or-later"])
def test_a_copyleft_licence_fails_the_build(expression: str) -> None:
    """The guard, disabled, is a bundle nobody may redistribute shipped as if they may.

    ``MIT AND LGPL-2.1`` is in the list because a check that only looked at the first term
    would pass it, and a compound expression is exactly how a copyleft component arrives.
    """
    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.check_licence(expression)

    assert "copyleft" in str(raised.value)


def test_a_licence_nobody_has_assessed_fails_with_a_different_message() -> None:
    """ "Not permitted" and "not yet considered" are different answers.

    Collapsing them invites the fix that makes the check pass — adding the term to the list —
    for a licence that should have been refused.
    """
    with pytest.raises(grammar_bundle.GrammarBundleError) as raised:
        grammar_bundle.check_licence("SomeVendor-Proprietary-1.0")

    message = str(raised.value)
    assert "copyleft" not in message
    assert "assessed" in message


def test_the_permissive_list_is_accepted_term_by_term() -> None:
    """Every licence docs/parsing.md §12 assessed, and the operators between them."""
    for licence in grammar_bundle.PERMISSIVE_LICENCES:
        assert grammar_bundle.check_licence(licence) == licence
    assert grammar_bundle.check_licence("Apache-2.0 OR MIT")


# --- helpers --------------------------------------------------------------------------------


_SEED_AND_PARSE = """
import json
from manicule.parsers import grammars

languages = ["python", "rust"]
report = {"cache_dir": str(grammars.cache_directory())}
report["missing_before"] = list(grammars.missing_grammars(languages))
report["seeded"] = list(grammars.prefetch(languages))
report["missing_after"] = list(grammars.missing_grammars(languages))
source = b"class Store:\\n    def refresh(self):\\n        return 1\\n"
report["parsed"] = grammars.load_parser("python").parse(source).root_node.type
print(json.dumps(report))
"""
"""What the air-gapped child does: report where its cache landed, pre-seed, and parse.

A script rather than an assertion inside the child, so that the parent decides what counts as
success. A child that asserted for itself and exited zero would be trusted about a cache
directory it never printed.
"""

_PLANT_A_BROKEN_LIBRARY = """
import json
from pathlib import Path
from manicule.core.errors import ParseError
from manicule.parsers import grammar_bundle, grammars

cache = Path.home() / "grammars"
cache.mkdir(parents=True)
(cache / f"libtree_sitter_python{grammar_bundle.library_suffix()}").write_bytes(b"not a library")
grammars.configure_pack(grammars.DECLARED_LANGUAGES, cache_dir=cache)

report = {"reported_as_present": grammars.is_available("python")}
try:
    grammars.load_parser("python")
except grammars.GrammarUnavailableError as exc:
    report["type"] = type(exc).__name__
    report["reason"] = exc.reason
    report["message"] = str(exc)
    report["is_parse_error"] = isinstance(exc, ParseError)
    report["is_grammar_unavailable"] = isinstance(exc, grammars.GrammarUnavailableError)
print(json.dumps(report))
"""
"""A cache holding a file with a grammar's name and nothing else in it.

The state a bundle built for another architecture produces, reached without needing one: the
pack reports the language as downloaded either way, and what matters is what happens next.
"""


def _air_gapped_child(
    home: Path, script: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` as if on a fresh host with no route to anything.

    The environment is built rather than inherited: ``HOME`` and ``XDG_CACHE_HOME`` are moved
    so the pack's cache resolves somewhere that has never existed on either platform's
    convention, and the manifest points at the discard port so a fetch refuses at once instead
    of reaching the real release.
    """
    home.mkdir(parents=True, exist_ok=True)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / "cache"),
        "LOCALAPPDATA": str(home / "local"),
        grammars.MANIFEST_URL_ENV: UNREACHABLE_MANIFEST,
        **(extra or {}),
    }
    return subprocess.run(  # noqa: S603 - this interpreter, on a script in this file
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _copied_bundle(bundle: grammar_bundle.GrammarBundle, tmp_path: Path) -> Path:
    """A writable copy, so a corruption case cannot decide what a later test sees."""
    destination = tmp_path / "copy"
    shutil.copytree(bundle.root, destination)
    return destination


def _edited_manifest(
    bundle: grammar_bundle.GrammarBundle, tmp_path: Path, **changes: object
) -> Path:
    """A copy of ``bundle`` whose manifest disagrees with it in exactly one way."""
    destination = _copied_bundle(bundle, tmp_path)
    path = destination / grammar_bundle.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return destination


def _make_read_only(root: Path) -> None:
    """Strip write permission from a tree, deepest first."""
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    root.chmod(root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def _make_writable(root: Path) -> None:
    """Put it back, so the temporary directory can be removed."""
    root.chmod(root.stat().st_mode | stat.S_IWUSR)
    for path in root.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
