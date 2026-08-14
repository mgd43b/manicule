"""The offline vocabulary bundle: an install that answers questions with nothing to fetch.

``tiktoken`` ships no vocabularies in its wheel and downloads them on first use, which put a
blob-store fetch on manicule's query path. This suite is the case where that fetch cannot
succeed: no route to anything, and a ``tiktoken`` cache that has never been written to.

**Two conditions have to be separable, and they are asserted separately and together**,
because the last serious bug in the equivalent grammar work came from exercising only one of
them. So:

- **A cut network** is exercised on its own, in-process, by replacing the single function
  through which ``tiktoken`` reaches the network. That test's cache is the real one, so it
  proves the door is shut rather than that the machine is offline.
- **A redirected cache** is exercised on its own, in-process, by pointing
  ``TIKTOKEN_CACHE_DIR`` at an empty directory. That test's network is intact, so it proves
  the bundle supplied the bytes rather than the machine happening to be air-gapped.
- **Both at once** is the headline test, and then again in a subprocess with ``TMPDIR``,
  ``HOME`` and ``XDG_CACHE_HOME`` all moved and ``TIKTOKEN_CACHE_DIR`` *unset*, so that
  ``tiktoken``'s own default resolution — a directory under the system temporary directory —
  lands somewhere that has never existed. The child reports the cache it resolved and proves
  its own socket cut before doing anything, so neither is assumed.

**The in-process tests clear ``tiktoken``'s encoding registry first, and that is not a
detail.** ``tiktoken.get_encoding`` memoizes every encoding it has ever built in a module
global, so a suite that has already assembled one context would hand the memoized vocabulary
back without touching the cache — and every assertion here would pass against a warm process
while proving nothing whatsoever about a cold host.

Every bundle here is built by the shipped builder from this machine's own cache. Nothing
hand-writes a manifest: a reader tested against a fixture's idea of the format is a format
with two definitions and no failing test.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.outcomes import Failed, OutcomeException, Skipped

from manicule import vocabularies
from manicule.chunking.tokens import TIKTOKEN_ENCODING
from manicule.config.settings import ENV_PREFIX
from manicule.core.errors import ConfigError
from manicule.generation.budget import GENERATION_ENCODING
from manicule.retrieval.tokens import DEFAULT_ENCODING, ContextTokenCounter
from manicule.vocabularies import bundle as bundles
from tests.vocabulary_support import (
    BUNDLE_ENCODINGS,
    build_bundle,
    require_source_vocabularies,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TEXT = "authentication tokens rotate on a schedule"
"""Something to count, so that a test asserts an encoding *works* and not merely that it was
constructed. A vocabulary that loads and cannot encode is a state a digest check would miss."""


@pytest.fixture
def cold_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that has never built an encoding, without starting a new one.

    ``tiktoken`` keeps one ``ENCODINGS`` dict per process and answers from it before doing
    anything else. Left alone, every test below would be handed a vocabulary built by whatever
    ran earlier in the session, and would pass with an empty cache and a severed network while
    proving nothing.
    """
    from tiktoken import registry  # noqa: PLC0415 - a retrieval extra, not core

    monkeypatch.setattr(registry, "ENCODINGS", {})


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Cut ``tiktoken``'s only route to the network, and record any attempt to take it.

    ``read_file`` is where the library opens a socket — ``read_file_cached`` calls it, and
    only when the cache cannot answer. Replacing it is a sharper instrument than unplugging
    the machine: a test that reaches for the network fails naming the URL it reached for,
    rather than passing slowly or failing with a connection error that could mean anything.

    Returns the list of URLs that were reached for, which is what the tests assert is empty.
    """
    import tiktoken.load as loader  # noqa: PLC0415 - a retrieval extra, not core

    attempted: list[str] = []

    def refuse(blobpath: str) -> bytes:
        attempted.append(blobpath)
        msg = f"this test has no route to {blobpath}"
        raise OSError(msg)

    monkeypatch.setattr(loader, "read_file", refuse)
    return attempted


@pytest.fixture
def watched_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Leave the network exactly where it was, and record anything that uses it.

    The counterpart of :func:`no_network`, for the tests whose point is that a download was
    *possible*. Cutting the network there would prove only that the bundle works on a machine
    that could not have downloaded anything, which is the weaker claim and not the one those
    tests make.
    """
    import tiktoken.load as loader  # noqa: PLC0415 - a retrieval extra, not core

    attempted: list[str] = []
    original = loader.read_file

    def watch(blobpath: str) -> bytes:
        attempted.append(blobpath)
        return original(blobpath)

    monkeypatch.setattr(loader, "read_file", watch)
    return attempted


@pytest.fixture
def empty_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``tiktoken`` cache directory with nothing in it."""
    cache = tmp_path / "tiktoken"
    cache.mkdir()
    monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, str(cache))
    return cache


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> bundles.VocabularyBundle:
    """One real bundle for the whole module, built from this machine's cache.

    Module-scoped because building copies 5 MB and nothing that reads a bundle modifies it.
    The tests that *do* modify one copy it first, so a corruption case cannot decide what a
    later test sees.
    """
    return build_bundle(tmp_path_factory.mktemp("bundle"))


# --- the air-gapped install ---------------------------------------------------------------


@pytest.mark.usefixtures("cold_registry")
def test_a_bundle_seeds_an_empty_cache_with_no_route_to_the_network(
    bundle: bundles.VocabularyBundle, empty_cache: Path, no_network: list[str]
) -> None:
    """The whole feature, in one test: install, no network, no cache, and a context is fitted.

    Both conditions at once. The vocabulary that ends up in the cache can only have come out
    of the bundle, because the one function that could have downloaded it records every call
    and is asserted never to have been made.
    """
    assert not list(empty_cache.iterdir())
    assert vocabularies.missing_vocabularies(BUNDLE_ENCODINGS) == BUNDLE_ENCODINGS

    seeded = vocabularies.prefetch(BUNDLE_ENCODINGS, bundle_dir=bundle.root)

    assert seeded == BUNDLE_ENCODINGS
    assert vocabularies.missing_vocabularies(BUNDLE_ENCODINGS) == ()
    assert ContextTokenCounter().count(TEXT) > 0
    assert no_network == [], f"reached for the network: {no_network}"


@pytest.mark.usefixtures("cold_registry")
def test_the_network_alone_being_cut_is_survivable_because_the_cache_is_warm(
    no_network: list[str],
) -> None:
    """The first condition on its own, against this machine's real cache.

    Separating it is what makes the headline test evidence rather than coincidence: this
    proves the door being shut does not break an ordinary install, so a failure there is about
    the bundle and the empty cache rather than about the door.
    """
    require_source_vocabularies()

    assert ContextTokenCounter().count(TEXT) > 0
    assert no_network == []


@pytest.mark.usefixtures("cold_registry")
def test_the_cache_alone_being_empty_is_served_by_the_bundle_with_the_network_intact(
    bundle: bundles.VocabularyBundle, empty_cache: Path, watched_network: list[str]
) -> None:
    """The second condition on its own, with nothing stopping a download.

    The bundle is consulted before the network, so nothing is fetched here even though
    something could be — which is the property that makes the pre-seed cheap on a machine that
    *does* have a network, and the reason the order is that way round.

    **The watcher is the assertion, and comparing the bytes is not.** Written the obvious way
    — seed, then check the cache matches the bundle — this test passed with seeding disabled
    entirely, because the fall-through downloaded the same file and the bytes agreed. That is
    the exact shape of failure this suite is built against, so what is asserted is that the
    door was open and nobody walked through it.
    """
    seeded = vocabularies.prefetch(BUNDLE_ENCODINGS, bundle_dir=bundle.root)

    assert seeded == BUNDLE_ENCODINGS
    assert watched_network == [], f"the bundle was on disk and this downloaded {watched_network}"
    for entry in bundle.vocabularies.values():
        assert (empty_cache / entry.filename).read_bytes() == (
            bundle.vocabulary_dir / entry.filename
        ).read_bytes()


@pytest.mark.usefixtures("cold_registry", "empty_cache")
def test_the_query_path_refuses_rather_than_downloading_what_it_was_not_given(
    no_network: list[str],
) -> None:
    """The door, asserted as a door rather than as a consequence of being offline.

    Nothing here cuts the machine off — ``no_network`` is a *recorder* that would also refuse,
    and the assertion is that it was never reached. A ``load_encoding`` that quietly fetched
    would still produce a working counter on a networked developer's machine and a mystery on
    an air-gapped host, which is precisely the split this test exists to close.
    """
    with pytest.raises(vocabularies.VocabularyUnavailableError) as raised:
        vocabularies.load_encoding("o200k_base")

    message = str(raised.value)
    assert "o200k_base" in message
    assert "o200k_base.tiktoken" in message
    assert vocabularies.CACHE_DIR_ENV in message
    assert bundles.BUNDLE_DIR_ENV in message
    assert "tools/build_vocabulary_bundle.py" in message
    # The command, and it exists: `manicule doctor --fix` seeds vocabularies alongside
    # grammars. A refusal that named a Python one-liner would be describing manicule's
    # internals to somebody who wants their search to work.
    assert "manicule doctor --fix" in message
    assert no_network == []


@pytest.mark.usefixtures("cold_registry")
def test_a_context_counter_refuses_while_retrieval_is_being_assembled(
    empty_cache: Path,
) -> None:
    """Early, and the whole point of #61: the refusal is at construction, not at a question.

    ``build_retriever`` constructs this counter while assembling the retrieval pipeline. A
    counter that resolved its vocabulary at the first ``count`` would put the failure inside
    an answered-looking query, which is where it was and why it read as "search is broken".
    """
    del empty_cache

    with pytest.raises(vocabularies.VocabularyUnavailableError):
        ContextTokenCounter()


@pytest.mark.usefixtures("cold_registry")
def test_a_second_pre_seed_copies_nothing(
    bundle: bundles.VocabularyBundle, empty_cache: Path
) -> None:
    """Pre-seeding is meant to be run on every start, which it can only be if it is cheap.

    The return value alone does not show that, since a copy followed by an empty answer is the
    same value. What shows it is the files: seeding writes a temporary name and renames it
    into place, so a second copy would replace the inode.
    """
    vocabularies.prefetch(BUNDLE_ENCODINGS, bundle_dir=bundle.root)
    before = {
        entry.filename: (empty_cache / entry.filename).stat().st_ino
        for entry in bundle.vocabularies.values()
    }

    assert vocabularies.prefetch(BUNDLE_ENCODINGS, bundle_dir=bundle.root) == ()
    after = {
        entry.filename: (empty_cache / entry.filename).stat().st_ino
        for entry in bundle.vocabularies.values()
    }
    assert after == before


@pytest.mark.usefixtures("cold_registry")
def test_a_read_only_bundle_is_a_cache_directory_in_its_own_right(
    bundle: bundles.VocabularyBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_network: list[str],
) -> None:
    """A container with a read-only filesystem needs no copy at all.

    The bundle's vocabulary directory is laid out as ``tiktoken``'s cache is, file names
    included, so a deployment that cannot write anywhere can point the cache variable straight
    at it. Asserted with the directory actually made read-only, because "we never write to it"
    is a claim about code and this is a claim about the filesystem.
    """
    read_only = tmp_path / "readonly"
    shutil.copytree(bundle.root, read_only)
    _make_read_only(read_only)
    try:
        monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, str(read_only / bundles.VOCABULARY_DIR_NAME))

        assert vocabularies.missing_vocabularies(BUNDLE_ENCODINGS) == ()
        assert ContextTokenCounter().count(TEXT) > 0
        assert no_network == []
    finally:
        _make_writable(read_only)


@pytest.mark.usefixtures("cold_registry")
def test_a_cache_entry_with_the_right_name_and_the_wrong_bytes_is_re_seeded(
    bundle: bundles.VocabularyBundle, empty_cache: Path, no_network: list[str]
) -> None:
    """A pre-seed that counts a corrupt file as present reports success and fixes nothing.

    ``tiktoken`` checks the digest when it reads, and on a mismatch it *deletes the file and
    fetches* — which on the host this exists for is a refusal, arriving after an install step
    said everything was fine. So a cache entry that will not be believed is reported as
    absent, and the pre-seed repairs it like any other gap.

    One flipped byte, which is what a truncated copy or a bad transfer leaves behind.
    """
    blob = vocabularies.blobs_for("o200k_base")[0]
    good = (bundle.vocabulary_dir / bundle.vocabularies[blob.url].filename).read_bytes()
    (empty_cache / blob.key).write_bytes(good[:-1] + bytes([good[-1] ^ 0xFF]))

    assert vocabularies.missing_vocabularies(["o200k_base"]) == ("o200k_base",)
    assert vocabularies.prefetch(["o200k_base"], bundle_dir=bundle.root) == ("o200k_base",)

    assert (empty_cache / blob.key).read_bytes() == good
    assert vocabularies.missing_vocabularies(["o200k_base"]) == ()
    assert no_network == []


def test_the_file_list_for_an_encoding_is_learned_once_and_not_re_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe's expensive shape happens at most once per encoding per process.

    Learning that an encoding needs no *second* file means letting the constructor ask for
    one, and a constructor allowed to get that far builds a 200 000-entry BPE table. The list
    is a property of the installed library rather than of the disk, so it is memoized — and
    the assertion is that the constructor is never run again, because a timing comparison
    would be a flake and a cache that is merely fast is not the claim.

    **The spy is on the constructor and not on the loader**, which the first draft got wrong
    and which passed while checking nothing: the probe *replaces* the loader for the duration
    of its call, so a stand-in installed there is exactly what the probe puts aside. The
    constructor is the thing the probe reaches for and does not touch.
    """
    from tiktoken import registry  # noqa: PLC0415 - a retrieval extra, not core

    require_source_vocabularies(("o200k_base",))
    first = vocabularies.blobs_for("o200k_base")
    assert first

    def never() -> dict[str, object]:
        message = "the file list was already known and this ran the constructor again"
        raise AssertionError(message)

    constructors = dict(registry.ENCODING_CONSTRUCTORS or {})
    constructors["o200k_base"] = never
    monkeypatch.setattr(registry, "ENCODING_CONSTRUCTORS", constructors)

    assert vocabularies.blobs_for("o200k_base") == first


def test_a_pre_seed_that_wrote_nothing_is_a_failure_however_it_reported_itself(
    empty_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tiktoken`` swallows every write error when the cache was not named explicitly.

    ``read_file_cached`` wraps its cache write in ``except OSError: pass`` unless the
    directory came from an environment variable — so a fetch can return a working encoding and
    leave the cache empty, and the next process downloads again or, on the host this exists
    for, refuses. Asking the cache afterwards is what turns that into an error while it can
    still be acted on.
    """
    del empty_cache
    import tiktoken  # noqa: PLC0415 - a retrieval extra, not core

    monkeypatch.setattr(tiktoken, "get_encoding", _a_fetch_that_writes_nothing)

    with pytest.raises(vocabularies.VocabularyFetchError) as raised:
        vocabularies.prefetch(["o200k_base"])

    assert "still without it" in str(raised.value)


# --- a cache that resolves somewhere unexpected --------------------------------------------


def test_a_bundle_serves_a_cache_that_resolves_somewhere_unexpected(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """A real search, on a host with no route to anything and no cache anywhere it looks.

    ``TIKTOKEN_CACHE_DIR`` is **unset** here, so ``tiktoken`` falls back to its own rule — a
    directory under the system temporary directory — and ``TMPDIR``, ``HOME`` and
    ``XDG_CACHE_HOME`` are all moved so that rule lands somewhere that has never existed
    whichever convention the platform follows. The child reports where it landed, so the
    redirect is *shown* to have taken rather than assumed.

    In a subprocess because the redirect has to happen before anything resolves a cache, and
    because a process that has already built an encoding cannot honestly be asked what a fresh
    install would do. And the child runs a real ``Retriever`` over a real migrated database
    rather than a token counter on its own, because the failure this fixes was not "the
    counter raised" — it was ``search`` not working.
    """
    home = tmp_path / "elsewhere"
    result = _air_gapped_child(
        home, _SEED_AND_SEARCH, extra={bundles.BUNDLE_DIR_ENV: str(bundle.root)}
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["network"] != "connected", "the socket cut did not take"
    assert str(home) in report["cache_dir"]
    assert report["missing_before"] == list(BUNDLE_ENCODINGS)
    assert report["seeded"] == list(BUNDLE_ENCODINGS)
    assert report["missing_after"] == []
    assert report["stages"] == ["dense", "lexical", "rrf"]
    assert report["passages"] > 0
    assert report["context_tokens"] > 0
    assert report["tokenizer"].startswith("tiktoken:o200k_base")


def test_the_same_install_without_the_bundle_refuses_loudly(tmp_path: Path) -> None:
    """Take the bundle away and the air-gapped install must fail, not degrade.

    The negative of the test above, and the reason that one is evidence: identical child,
    identical redirect, identical socket cut, and no bundle. If this passed, the previous test
    would be proving that a redirected cache still finds the developer's vocabularies rather
    than proving anything about a bundle.

    The failure names the encoding, the cache it read, and that no bundle was found — the
    three things an operator on an air-gapped host has to know. What it must not do is answer
    the question with an estimate.
    """
    result = _air_gapped_child(tmp_path / "elsewhere", _SEED_AND_SEARCH)

    assert result.returncode != 0
    assert "VocabularyFetchError" in result.stderr
    assert "no offline vocabulary bundle is installed" in result.stderr
    assert bundles.BUNDLE_DIR_ENV in result.stderr
    assert "cl100k_base" in result.stderr


def test_a_bundle_installed_as_a_distribution_needs_no_configuration(tmp_path: Path) -> None:
    """The bundle arrives through the install channel rather than as a directory to copy.

    The builder's ``--package`` mode writes an importable distribution around the bundle, and
    the child here is given nothing but that distribution on its path — no environment
    variable, no configured directory. An air-gapped host that can install manicule at all can
    therefore install its vocabularies, which is the only shape that does not depend on
    somebody remembering to copy a directory to the right place.
    """
    from tools.build_vocabulary_bundle import main  # noqa: PLC0415 - a build script

    require_source_vocabularies()
    package = tmp_path / "dist"
    assert main(["--output", str(package), "--package", "--encodings", *BUNDLE_ENCODINGS]) == 0

    result = _air_gapped_child(
        tmp_path / "elsewhere",
        _SEED_AND_SEARCH,
        extra={"PYTHONPATH": os.pathsep.join([str(package / "src"), str(REPO_ROOT)])},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["seeded"] == list(BUNDLE_ENCODINGS)
    assert report["passages"] > 0


# --- what the bundle records ---------------------------------------------------------------


def test_the_bundle_records_the_url_and_the_digest_tiktoken_declares(
    bundle: bundles.VocabularyBundle,
) -> None:
    """Provenance a redistributor can read, and a digest that is upstream's rather than ours.

    The URL is recorded because manicule does not publish these files and cannot state their
    license terms — so whoever carries a bundle should be able to see exactly what is in it.
    The digest is recorded because a bundle is copied between machines by hand, and "the same
    bundle" has to mean the same bytes; taking it from ``tiktoken``'s own declaration rather
    than from whatever was downloaded is what makes it an assertion instead of a checksum.
    """
    manifest = json.loads((bundle.root / bundles.MANIFEST_NAME).read_text())

    assert manifest["schema_version"] == bundles.SCHEMA_VERSION
    assert sorted(manifest["encodings"]) == sorted(BUNDLE_ENCODINGS)
    for encoding in BUNDLE_ENCODINGS:
        declared = vocabularies.blobs_for(encoding)
        assert manifest["encodings"][encoding] == [blob.url for blob in declared]
        for blob in declared:
            recorded = manifest["vocabularies"][blob.url]
            assert recorded["sha256"] == blob.sha256
            assert recorded["size"] == vocabularies.cache_path(blob.url).stat().st_size


def test_a_bundle_built_against_another_tiktoken_release_is_still_usable(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """The one place this deliberately differs from the grammar bundle, asserted as a choice.

    A grammar is a compiled object and its bundle is refused outright on a pack mismatch. A
    ``.tiktoken`` file is a text table that has not moved across ``tiktoken`` releases, so
    refusing on a version bump would force a rebuild that changes nothing — and on the host
    that cannot rebuild anything. What is checked instead is stronger: the bundle must carry
    the URL the installed library asks for, with the digest it declares.
    """
    other = _edited_manifest(bundle, tmp_path, tiktoken_version="0.0.0-not-installed")

    read_back = bundles.read(other)

    assert read_back.tiktoken_version == "0.0.0-not-installed"
    assert read_back.encoding_names == BUNDLE_ENCODINGS


def test_a_bundle_from_a_later_manifest_schema_is_refused(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """A manifest this manicule cannot read is refused rather than read optimistically."""
    other = _edited_manifest(bundle, tmp_path, schema_version=bundles.SCHEMA_VERSION + 1)

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(other)

    assert str(bundles.SCHEMA_VERSION) in str(raised.value)


def test_a_directory_that_is_not_a_bundle_says_so(tmp_path: Path) -> None:
    """A configured path that is empty, or a typo, must not read as "no bundle configured"."""
    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(tmp_path)

    assert bundles.MANIFEST_NAME in str(raised.value)
    assert "tools/build_vocabulary_bundle.py" in str(raised.value)


@pytest.mark.parametrize("content", ["not json at all", '["a", "list"]'])
def test_a_manifest_that_is_not_a_bundle_manifest_says_which_file(
    tmp_path: Path, content: str
) -> None:
    """A truncated copy and a file that was never a manifest both land here.

    Both are one bad ``scp`` away, and both have to name the file — a ``JSONDecodeError``
    reported from inside a parser sends whoever hits it through manicule instead of to the
    directory they copied.
    """
    (tmp_path / bundles.MANIFEST_NAME).write_text(content)

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(tmp_path)

    assert bundles.MANIFEST_NAME in str(raised.value)


def test_a_manifest_listing_an_encoding_with_no_files_is_refused(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """A bundle carrying nothing for an encoding would report itself present and seed nothing.

    That is the worst of both: the pre-seed sees a bundle, copies nothing, and the failure
    that follows names a bundle as if it had been consulted usefully.
    """
    edited = _edited_manifest(bundle, tmp_path, encodings={"o200k_base": []})

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(edited)

    assert "no vocabulary files" in str(raised.value)


def test_a_manifest_naming_a_file_it_does_not_describe_is_refused(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """An encoding whose file table entry is missing is a bundle that half-exists."""
    edited = _edited_manifest(
        bundle, tmp_path, encodings={"o200k_base": ["https://example.invalid/o200k.tiktoken"]}
    )

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(edited)

    assert "describes no such file" in str(raised.value)


def test_a_manifest_whose_numbers_are_not_numbers_is_refused_as_a_bundle_problem(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """A corrupt manifest must read as a corrupt manifest.

    Left to fail wherever the conversion happens, it surfaces as a bare ``ValueError`` from a
    frame with no file name in it — which sends whoever hits it into manicule's internals
    rather than to the directory they copied.
    """
    edited = _copied_bundle(bundle, tmp_path)
    path = edited / bundles.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    url = next(iter(manifest["vocabularies"]))
    manifest["vocabularies"][url]["size"] = "quite large"
    path.write_text(json.dumps(manifest))

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(edited)

    assert "malformed" in str(raised.value)


# --- integrity ------------------------------------------------------------------------------


def test_a_truncated_vocabulary_is_refused_when_the_bundle_is_read(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """Caught on read rather than at load.

    By load time the file has been copied into the cache and ``tiktoken`` has decided the
    digest is wrong, deleted it and reached for the network — which on the host this exists
    for is a refusal blaming a blob store for a bad ``scp``. A ``stat`` per file is what that
    costs.
    """
    damaged = _copied_bundle(bundle, tmp_path)
    entry = next(iter(bundle.vocabularies.values()))
    library = damaged / bundles.VOCABULARY_DIR_NAME / entry.filename
    library.write_bytes(library.read_bytes()[:1024])

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(damaged)

    assert "1024" in str(raised.value)


def test_a_vocabulary_missing_from_the_bundle_is_refused_when_it_is_read(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """A manifest is a claim about files; the files are checked against it."""
    damaged = _copied_bundle(bundle, tmp_path)
    entry = next(iter(bundle.vocabularies.values()))
    (damaged / bundles.VOCABULARY_DIR_NAME / entry.filename).unlink()

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(damaged)

    assert "missing" in str(raised.value)


def test_a_vocabulary_edited_without_changing_its_length_is_refused_when_it_is_seeded(
    bundle: bundles.VocabularyBundle, empty_cache: Path, tmp_path: Path
) -> None:
    """The case a size check cannot see, and the reason the digest is recorded.

    A bundle moves between machines by whatever is available, and a flipped byte in a BPE
    table is not a crash: it is a different token boundary, on a corpus that will be compared
    against one chunked elsewhere. The copy has to read every byte anyway, so hashing it there
    is free and is where the check belongs.
    """
    damaged = _copied_bundle(bundle, tmp_path)
    entry = next(iter(bundle.vocabularies.values()))
    path = damaged / bundles.VOCABULARY_DIR_NAME / entry.filename
    content = bytearray(path.read_bytes())
    content[-1] ^= 0xFF
    path.write_bytes(bytes(content))

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.read(damaged).seed(BUNDLE_ENCODINGS, empty_cache)

    assert "sha256" in str(raised.value)
    assert not list(empty_cache.iterdir())


def test_verify_reads_every_byte_where_a_read_only_checks_lengths(
    bundle: bundles.VocabularyBundle, tmp_path: Path
) -> None:
    """The deliberate check, for a build step confirming what it just wrote.

    ``read`` deliberately stops at sizes, because it runs on every pre-seed. That split is
    only honest if the expensive half exists and catches what the cheap half cannot, which is
    a file whose length is right and whose content is not.
    """
    damaged = _copied_bundle(bundle, tmp_path)
    entry = next(iter(bundle.vocabularies.values()))
    path = damaged / bundles.VOCABULARY_DIR_NAME / entry.filename
    content = bytearray(path.read_bytes())
    content[0] ^= 0xFF
    path.write_bytes(bytes(content))

    read_back = bundles.read(damaged)
    with pytest.raises(bundles.VocabularyBundleError) as raised:
        read_back.verify()

    assert "sha256" in str(raised.value)


def test_asking_a_bundle_for_a_file_it_does_not_carry_names_what_it_has(
    bundle: bundles.VocabularyBundle, empty_cache: Path
) -> None:
    """The wrong bundle and a broken one are different mistakes.

    Seeding an encoding the bundle does not carry is not an error — the caller still has the
    network to try for it — but *asking* for its file is, and the answer has to be the list,
    not a ``KeyError`` from a dictionary the caller never saw.
    """
    assert bundle.seed(["p50k_base"], empty_cache) == ()
    assert not list(empty_cache.iterdir())

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundle.path_for("https://example.invalid/nothing.tiktoken")

    assert "o200k_base.tiktoken" in str(raised.value)


def test_building_from_a_cache_holding_the_wrong_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache with the right file name and the wrong content must not be bundled.

    Bundled, it would be copied to every air-gapped host and rejected there by a library with
    no idea where the file came from. The digest asserted at build time is ``tiktoken``'s own
    declaration, which is the only thing in the system that knows what these bytes should be.
    """
    require_source_vocabularies(("o200k_base",))
    cache = tmp_path / "wrong"
    cache.mkdir()
    blob = vocabularies.blobs_for("o200k_base")[0]
    real = vocabularies.cache_path(blob.url).read_bytes()
    monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, str(cache))
    (cache / blob.key).write_bytes(real[:-1] + bytes([real[-1] ^ 0xFF]))

    with pytest.raises(bundles.VocabularyBundleError) as raised:
        bundles.build(["o200k_base"], tmp_path / "out")

    assert blob.sha256 is not None
    assert blob.sha256 in str(raised.value)


# --- resolution -----------------------------------------------------------------------------


def test_an_explicitly_named_bundle_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that named a path meant that path.

    The reverse order would make a deployment-wide variable silently override a per-call
    argument, which is how a container ends up seeding from the image's bundle when the
    operator asked for the one they just copied in.
    """
    monkeypatch.setenv(bundles.BUNDLE_DIR_ENV, str(tmp_path / "from-environment"))

    assert bundles.locate(tmp_path / "named") == tmp_path / "named"
    assert bundles.locate() == tmp_path / "from-environment"


def test_no_bundle_anywhere_is_not_an_error_but_says_where_it_looked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An install with no bundle is ordinary; it pre-seeds over the network.

    ``None`` rather than an exception, because raising here would make every networked install
    fail for lacking something it does not need. The message is what makes it fixable.
    """
    monkeypatch.delenv(bundles.BUNDLE_DIR_ENV, raising=False)
    monkeypatch.setattr(bundles, "_installed_bundle", lambda: None)

    assert bundles.resolve() is None
    described = bundles.describe_search_path()
    assert bundles.BUNDLE_DIR_ENV in described
    assert bundles.BUNDLE_MODULE in described


def test_a_bundle_that_is_present_and_unusable_is_never_reported_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one downgrade that must not happen.

    A bundle that is present and wrong, reported as "no bundle", sends an operator to arrange
    network access they do not have while the answer is sitting on the disk in front of them.
    """
    monkeypatch.setenv(bundles.BUNDLE_DIR_ENV, str(tmp_path))

    with pytest.raises(bundles.VocabularyBundleError):
        bundles.resolve()
    with pytest.raises(bundles.VocabularyBundleError):
        vocabularies.bundle_status()


def test_an_installed_distribution_without_a_payload_reads_as_no_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``manicule_vocabularies`` importable but empty is "no bundle", not "a broken bundle".

    It is the state a distribution is in before its payload has been built into it, and it
    must not fail an install that has network access and needs no bundle. The loud failure
    belongs one step later, where the pre-seed cannot complete — and that failure names the
    bundle search path, so nothing is hidden.
    """
    monkeypatch.delenv(bundles.BUNDLE_DIR_ENV, raising=False)
    site = tmp_path / "site"
    module = site / bundles.BUNDLE_MODULE
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])
    monkeypatch.delitem(sys.modules, bundles.BUNDLE_MODULE, raising=False)
    importlib.invalidate_caches()

    assert bundles.locate() is None

    (module / "bundle").mkdir()

    assert bundles.locate() == module / "bundle"


@pytest.mark.usefixtures("cold_registry")
def test_the_pre_seed_failure_names_the_bundle_as_well_as_the_host(
    empty_cache: Path, monkeypatch: pytest.MonkeyPatch, no_network: list[str]
) -> None:
    """ "Connection refused" alone sends an operator to their firewall.

    On an air-gapped host the actionable fact is almost never the network. It is that there is
    no bundle, or that the bundle does not carry the encoding that was configured.

    ``cold_registry`` because this asserts that the pre-seed *did* reach for the network, and
    ``tiktoken`` answers from its process-global registry before it reads anything at all — so
    in a session that has already assembled a context, the fetch would be skipped and this
    would fail while everything under test was correct.
    """
    del empty_cache
    monkeypatch.delenv(bundles.BUNDLE_DIR_ENV, raising=False)
    monkeypatch.setattr(bundles, "_installed_bundle", lambda: None)

    with pytest.raises(vocabularies.VocabularyFetchError) as raised:
        vocabularies.prefetch(["o200k_base"])

    message = str(raised.value)
    assert "no offline vocabulary bundle is installed" in message
    assert "openaipublic.blob.core.windows.net" in message
    assert no_network == [
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    ]


@pytest.mark.usefixtures("cold_registry", "empty_cache")
def test_a_broken_bundle_is_quoted_into_the_refusal_rather_than_thrown_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle that is present and wrong must not replace the failure being reported.

    ``load_encoding`` does not resolve the bundle before it refuses — it has no reason to —
    so the bundle is first touched while the refusal's message is being *assembled*. Left
    unguarded, an unreadable bundle turns "this vocabulary is not on this machine, here is how
    to get it" into a manifest complaint, and whoever hits it has no idea which question
    failed or why.

    The state is reachable with one bad copy: a bundle directory that arrived without its
    manifest.
    """
    broken = tmp_path / "half-copied"
    broken.mkdir()
    monkeypatch.setenv(bundles.BUNDLE_DIR_ENV, str(broken))

    with pytest.raises(vocabularies.VocabularyUnavailableError) as raised:
        vocabularies.load_encoding("o200k_base")

    message = str(raised.value)
    assert "o200k_base" in message
    assert "the offline vocabulary bundle is unusable" in message
    assert bundles.MANIFEST_NAME in message


# --- the switch that decides whether this suite means anything -------------------------------


def test_the_required_switch_survives_the_test_environment() -> None:
    """The mechanism that stops a green job from meaning nothing, and it has failed twice.

    ``manicule_environment`` deletes every variable in the ``MANICULE_`` namespace before each
    test, so a switch named that way is scrubbed before it is ever read and the job reports
    green having skipped everything. That has already happened to the embedding switch and to
    the grammar one. This name is outside the namespace, and the value is read at *import* as
    well, so a future fixture that deletes it cannot disarm the switch either.
    """
    from tests import vocabulary_support  # noqa: PLC0415 - the module under test here

    assert not vocabulary_support.REQUIRE_BUNDLE_ENV.startswith(ENV_PREFIX)
    assert (
        bool(os.environ.get(vocabulary_support.REQUIRE_BUNDLE_ENV, "").strip())
        == vocabulary_support.BUNDLE_REQUIRED
    )


@pytest.mark.parametrize("armed", [True, False])
def test_an_absent_vocabulary_fails_when_the_switch_is_armed_and_skips_when_it_is_not(
    monkeypatch: pytest.MonkeyPatch, empty_cache: Path, *, armed: bool
) -> None:
    """Both directions, exercised rather than assumed.

    The switch has exactly one job — turn this suite's skips into failures — and both halves
    matter for different reasons. Armed and skipping means CI certifies nothing. Unset and
    failing means a first checkout is red for a reason that has nothing to do with the change
    under test, which is how a suite gets marked flaky and then ignored.
    """
    from tests import vocabulary_support  # noqa: PLC0415 - the module under test here

    del empty_cache
    monkeypatch.setattr(vocabulary_support, "BUNDLE_REQUIRED", armed)
    expected = Failed if armed else Skipped

    # `OutcomeException` — the base of both — rather than `expected`, and the reason is the
    # defect this whole file is about. `pytest.raises(Failed)` does not catch `Skipped`, so a
    # switch that had stopped arming would let the skip escape and pytest would mark *this
    # test* skipped: a green run, and a guard that checked nothing. Caught here, the wrong
    # outcome is an assertion failure like any other. Found by disabling the switch and
    # watching this test skip rather than fail.
    with pytest.raises(OutcomeException) as raised:
        vocabulary_support.require_source_vocabularies(("o200k_base",))

    assert type(raised.value) is expected

    assert "o200k_base" in str(raised.value)
    assert (vocabulary_support.REQUIRE_BUNDLE_ENV in str(raised.value)) is armed


# --- what a pre-seed is asked for -----------------------------------------------------------


def test_the_encodings_a_pre_seed_needs_come_from_the_code_that_asks_for_them() -> None:
    """One definition, read by the pre-seed, the builder, the container build and CI.

    Three copies of a list of encoding names is how a host ends up with the chunker's
    vocabulary and not the fitter's — an install that indexes and then cannot answer, which is
    the exact shape of the defect this package closes.
    """
    required = vocabularies.required_encodings()

    assert TIKTOKEN_ENCODING in required
    assert DEFAULT_ENCODING in required
    assert GENERATION_ENCODING in required


def test_a_caller_with_resolved_settings_pre_seeds_the_encoding_it_configured() -> None:
    """A diagnostic that read manicule's defaults would describe a different installation.

    An install that set ``rag.context.encoding`` to something else must pre-seed *that*, or
    the pre-seed reports success over a vocabulary the fitter will never ask for and the
    refusal arrives at the first question anyway. The caller that already resolved settings
    says which one; the pre-seed script, which has none, reads them.
    """
    assert "p50k_base" in vocabularies.required_encodings("p50k_base")
    assert "p50k_base" not in vocabularies.required_encodings("o200k_base")
    # The generation budget's encoding is a constant rather than a setting, so it is in both
    # sets — which is the point of assembling the list here instead of at each caller.
    assert GENERATION_ENCODING in vocabularies.required_encodings("p50k_base")


def test_an_encoding_this_install_does_not_know_is_a_configuration_error() -> None:
    """``rag.context.encoding`` holding a model name is the specific mistake to catch.

    ``encoding_for_model("gpt-4o")`` would make an estimate look authoritative about a model
    that is not being used, so a model name reaching this far must fail naming what it should
    have been.
    """
    with pytest.raises(ConfigError) as raised:
        vocabularies.load_encoding("gpt-4o")

    message = str(raised.value)
    assert "o200k_base" in message
    assert "never a model name" in message


def test_the_cache_path_this_module_computes_is_where_tiktoken_actually_writes(
    empty_cache: Path,
) -> None:
    """The one rule reproduced from upstream, checked against upstream.

    ``tiktoken`` exposes no ``cache_dir()``, so :func:`cache_path` reproduces its resolution
    and its cache-key rule. Getting that wrong is silent in the worst way — a pre-seed writes
    5 MB into a directory nothing reads and reports success — so the reproduction is asserted
    by letting ``tiktoken`` do the writing and checking the file lands where this says it will.
    """
    import tiktoken.load as loader  # noqa: PLC0415 - a retrieval extra, not core

    url = "https://openaipublic.blob.core.windows.net/encodings/manicule-test.tiktoken"
    payload = b"placeholder bytes\n"
    original = loader.read_file
    loader.read_file = lambda _: payload
    try:
        loader.read_file_cached(url)
    finally:
        loader.read_file = original

    assert vocabularies.cache_path(url).read_bytes() == payload
    assert vocabularies.cache_path(url).parent == empty_cache


# --- helpers --------------------------------------------------------------------------------


_SEED_AND_SEARCH = """
import asyncio, json, os, socket, sys
from pathlib import Path


def refuse(*args, **kwargs):
    raise OSError("no route to host: this machine has no network")


# The class itself is left alone — `ssl` subclasses it at import, and replacing it outright
# breaks the standard library before anything under test runs. Cutting the three calls that
# open or resolve a connection is the whole of a socket's route out.
socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse

report = {"network": "connected"}
try:
    socket.create_connection(("1.1.1.1", 443), timeout=1)
except OSError as exc:
    report["network"] = str(exc)

from manicule import vocabularies
from manicule.retrieval.assembly import ContextAssembler
from manicule.retrieval.dense import DenseStage
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.lexical import LexicalStage
from manicule.retrieval.profile import Profiles
from manicule.retrieval.retriever import Retriever
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.tokens import ContextTokenCounter
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.migrator import upgrade
from tests.fakes import HashEmbedder
from tests.retrieval.fakes import ListVectorStore, a_query
from tests.storage_helpers import make_chunk, make_document

ENCODINGS = ["cl100k_base", "o200k_base"]
report["cache_dir"] = str(vocabularies.cache_directory())
report["missing_before"] = list(vocabularies.missing_vocabularies(ENCODINGS))
report["seeded"] = list(vocabularies.prefetch(ENCODINGS))
report["missing_after"] = list(vocabularies.missing_vocabularies(ENCODINGS))


async def search():
    engine = create_engine(Path(os.environ["HOME"]) / "data")
    await upgrade(engine)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    document = make_document(source_id="live")
    await store.upsert_document(document)
    chunks = [
        make_chunk(document, 0, "authentication tokens rotate weekly"),
        make_chunk(document, 1, "authentication is configured in the console"),
        make_chunk(document, 2, "unrelated prose about the weather"),
    ]
    await store.replace_chunks(document.id, chunks)
    profiles = Profiles({"candidates": 3, "final_top_k": 3})
    stages = [
        DenseStage(
            embedder=HashEmbedder(),
            vectors=ListVectorStore(chunks),
            docstore=store,
            profiles=profiles,
        ),
        LexicalStage(docstore=store, profiles=profiles),
        RRFStage(),
    ]
    retriever = Retriever(
        runner=PipelineRunner(stages, docstore=store),
        docstore=store,
        assembler=ContextAssembler(counter=ContextTokenCounter(), profiles=profiles),
        profiles=profiles,
        rrf_k=60,
        embed_fingerprint=HashEmbedder().fingerprint.canonical(),
    )
    try:
        return await retriever.retrieve(a_query("authentication"))
    finally:
        await engine.dispose()


result = asyncio.run(search())
report["stages"] = [span.name for span in result.trace.stages]
report["passages"] = len(result.context.passages)
report["context_tokens"] = result.context.token_count
report["tokenizer"] = result.trace.assembly.tokenizer
print(json.dumps(report))
"""
"""What the air-gapped child does: prove its own network cut, report where its cache landed,
pre-seed, and then run a real ``Retriever`` over a real migrated database.

A script rather than assertions inside the child, so that the parent decides what counts as
success. A child that asserted for itself and exited zero would be trusted about a cache
directory it never printed.
"""


def _a_fetch_that_writes_nothing(encoding: str) -> str:
    """A ``get_encoding`` that reports success and leaves the cache exactly as it found it.

    Not a hypothetical: ``read_file_cached`` swallows every ``OSError`` from its cache write
    when the directory was not named by an environment variable, so this is what an
    unwritable cache looks like from the outside.
    """
    return encoding


def _air_gapped_child(
    home: Path, script: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` as if on a fresh host with no route to anything.

    The environment is built rather than inherited. ``TIKTOKEN_CACHE_DIR`` is deliberately
    *absent*, so ``tiktoken``'s own rule applies; ``TMPDIR`` and the Windows equivalents move
    where that rule lands, and ``HOME`` and ``XDG_CACHE_HOME`` move everything else. Whichever
    convention the platform follows, the child's cache is a directory that has never existed.
    """
    home.mkdir(parents=True, exist_ok=True)
    temporary = home / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "XDG_CACHE_HOME": str(home / "cache"),
        "LOCALAPPDATA": str(home / "local"),
        "PYTHONPATH": str(REPO_ROOT),
        **(extra or {}),
    }
    return subprocess.run(  # noqa: S603 - this interpreter, on a script in this file
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _copied_bundle(bundle: bundles.VocabularyBundle, tmp_path: Path) -> Path:
    """A writable copy, so a corruption case cannot decide what a later test sees."""
    destination = tmp_path / "copy"
    shutil.copytree(bundle.root, destination)
    return destination


def _edited_manifest(bundle: bundles.VocabularyBundle, tmp_path: Path, **changes: object) -> Path:
    """A copy of ``bundle`` whose manifest disagrees with it in exactly one way."""
    destination = _copied_bundle(bundle, tmp_path)
    path = destination / bundles.MANIFEST_NAME
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
