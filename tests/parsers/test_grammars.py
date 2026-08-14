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
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from importlib.metadata import metadata
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.core.errors import ConfigError, ParseError
from manicule.parsers import grammars

UNREACHABLE_MANIFEST = "http://127.0.0.1:9/manicule-tests-must-not-download.json"
"""Discard port on the loopback interface. A fetch attempted through this refuses at once
rather than hanging, so a test that starts downloading fails rather than stalling CI."""

PERMISSIVE_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"})


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
    """The one naming trap in the pack, and the only defense is the message.

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


# --- the retry around the download ----------------------------------------------------------
#
# The grammars come from a third-party GitHub release, and the failure it actually produces is a
# transfer that drops part-way rather than a host that refuses. Until #80 a single one of those
# failed whatever had asked for it, which in this repository meant every open pull request at
# once, because the image build pre-seeds. These three are the whole contract: a dropped transfer
# is retried, a host that is genuinely gone still fails and says so, and the happy path does not
# pay for either.


def _installed_pack() -> Any:
    """The pack module ``_fetch`` reaches for, so a test can stand in for its download."""
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - a parsing extra, not core

    return pack


DROPPED_TRANSFER = (
    "Download error: Failed to download parsers-linux-x86_64.tar.zst: io: Peer disconnected"
)
"""The observed failure, quoted from the build log in #80 rather than invented.

It arrives as a bare ``RuntimeError`` from the native layer, which is why that type is in the
caught tuple at all — an earlier reading of this suite assumed the pack's own exception
hierarchy and would have let this one through as an unhandled crash.
"""


def test_a_dropped_transfer_is_retried_with_backoff_and_the_seed_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The failure that blocked the repository, and the behavior that unblocks it.

    The stand-in fails twice the way the real host does and then *actually writes the library*,
    copied out of this machine's cache — so what is asserted at the end is the real
    post-condition, a grammar the pack can find, rather than a mock reporting that it was
    called. A retry that returned success without landing a file would pass a weaker version of
    this test and ship an image with no grammars in it.

    The waits are recorded rather than taken. That the second is longer than the first is the
    part worth checking: a fixed interval would spend every attempt inside the same rate-limit
    window, which is one of the two transient failures this is for.
    """
    if not grammars.is_available("python"):
        pytest.skip("no python grammar cached to copy, so a successful attempt cannot be staged")
    library = next(iter(grammars.cache_directory().glob("*python*")))

    cache = tmp_path / "grammars"
    cache.mkdir()
    grammars.configure_pack(["python"], cache_dir=cache, manifest_url=UNREACHABLE_MANIFEST)
    slept: list[float] = []
    monkeypatch.setattr(grammars.time, "sleep", slept.append)
    monkeypatch.setattr(grammars, "FETCH_RETRY_DELAYS", (0.25, 0.5))
    attempts: list[list[str]] = []

    def flaky(languages: list[str]) -> None:
        attempts.append(list(languages))
        if len(attempts) < 3:
            raise RuntimeError(DROPPED_TRANSFER)
        shutil.copy2(library, cache / library.name)

    monkeypatch.setattr(_installed_pack(), "prefetch", flaky)

    assert grammars.prefetch(["python"]) == ("python",)
    assert attempts == [["python"], ["python"], ["python"]]
    assert slept == [0.25, 0.5]
    assert grammars.is_available("python")


def test_a_host_that_never_answers_still_fails_and_keeps_the_whole_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retrying must not become a way of tolerating an outage, or of half-seeding one.

    Two things are asserted together on purpose. **Nothing lands**: the cache is empty
    afterwards, so there is no attempt count at which this produces a partial set of grammars
    for a later build step to package and ship — an image carrying three of twenty-four
    languages would build, pass, and silently parse the rest as plain text.

    **The message still says everything it did**, because it is read almost exclusively by
    someone whose host has no route to the release: the bundle, the mirror variable and the
    section that explains building one. The attempt count is the only addition, and it is there
    so that a genuine outage is not read as a single unlucky request.
    """
    cache = tmp_path / "grammars"
    cache.mkdir()
    grammars.configure_pack(["python"], cache_dir=cache, manifest_url=UNREACHABLE_MANIFEST)
    slept: list[float] = []
    monkeypatch.setattr(grammars.time, "sleep", slept.append)
    monkeypatch.setattr(grammars, "FETCH_RETRY_DELAYS", (0.0, 0.0))
    attempts: list[list[str]] = []

    def never_answers(languages: list[str]) -> None:
        attempts.append(list(languages))
        raise RuntimeError(DROPPED_TRANSFER)

    monkeypatch.setattr(_installed_pack(), "prefetch", never_answers)

    with pytest.raises(grammars.GrammarFetchError) as raised:
        grammars.prefetch(["python"])

    message = str(raised.value)
    assert len(attempts) == 3
    # Two waits for three attempts: it gives up rather than sleeping once more first.
    assert len(slept) == 2
    assert "after 3 attempts" in message
    assert "Peer disconnected" in message
    assert grammars.MANIFEST_URL_ENV in message
    assert "docs/parsing.md §8.1" in message
    assert list(cache.iterdir()) == []
    assert not grammars.is_available("python")


def test_a_download_that_works_first_time_neither_retries_nor_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The steady state, and the off-by-one that would make every pre-seed five seconds slower.

    ``prefetch`` is called on every ``manicule init`` and every ``doctor --fix``, so a loop that
    slept before deciding whether it needed to would put a fixed cost on the path that is
    supposed to be free.
    """
    if not grammars.is_available("python"):
        pytest.skip("no python grammar cached to copy, so a successful attempt cannot be staged")
    library = next(iter(grammars.cache_directory().glob("*python*")))

    cache = tmp_path / "grammars"
    cache.mkdir()
    grammars.configure_pack(["python"], cache_dir=cache, manifest_url=UNREACHABLE_MANIFEST)
    slept: list[float] = []
    monkeypatch.setattr(grammars.time, "sleep", slept.append)
    attempts: list[list[str]] = []

    def works(languages: list[str]) -> None:
        attempts.append(list(languages))
        shutil.copy2(library, cache / library.name)

    monkeypatch.setattr(_installed_pack(), "prefetch", works)

    assert grammars.prefetch(["python"]) == ("python",)
    assert len(attempts) == 1
    assert slept == []


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


# --- the commands this module names ---------------------------------------------------------
#
# This module's docstrings name three ``manicule`` commands, and for a while all three claims
# were false: ``prefetch`` had no caller, ``doctor`` reported nothing about grammars, and
# ``doctor --fix`` did not exist at all — so the one string a refused document carries named a
# flag that would have been rejected as unknown. The claims are cheap to check and were never
# checked, which is the whole of it. These tests run them.
#
# The command line is driven here, in a parser suite, deliberately: the claim under test is a
# claim *this module* makes about the command line, so it belongs beside the module that makes
# it rather than in the suite for the surface it happens to name.


def _options_of(argv: Sequence[str]) -> set[str]:
    """Every option ``manicule <argv>`` offers, from the built command tree.

    Built rather than read out of the source, because a command defined and never attached is
    in the file and not in the interface — which is exactly the failure shape being guarded
    against, one layer along.
    """
    import typer.main  # noqa: PLC0415 - only these assertions need the click tree

    command: object = typer.main.get_command(cli.app)
    for name in argv:
        group: dict[str, object] = getattr(command, "commands", {})
        assert name in group, f"`manicule {' '.join(argv)}` names {name!r}, which is not a command"
        command = group[name]
    params: list[object] = getattr(command, "params", [])
    return {str(option) for parameter in params for option in getattr(parameter, "opts", [])}


def _claimed_commands(text: str) -> list[list[str]]:
    """Every ``manicule …`` command a docstring names, as argv.

    Only double-backticked spans count, so prose that happens to contain the word is not a
    claim — ``manicule supports the languages it declares`` is a sentence, not a command.
    """
    return [claim.split()[1:] for claim in re.findall(r"``(manicule [^`]+)``", text)]


def test_every_command_the_parsing_package_names_exists_with_the_flags_it_names() -> None:
    """A docstring that names a command is a claim, and this is the check on it.

    Over the whole package rather than this one module, because the defect was never specific
    to ``grammars.py``: ``grammar_bundle.py`` described what ``manicule init`` does on an
    air-gapped host and ``config.py`` describes what ``manicule doctor`` loads, and a claim is
    as cheap to get wrong in one file as in another.

    ``PRESEED_COMMAND`` is included with the docstrings because it is the same kind of claim
    carried in a different vehicle: it is the ``status_detail`` of every document refused for
    want of a grammar, so it is read by more people than any docstring here.
    """
    package = Path(grammars.__file__).parent
    sources = sorted(package.glob("*.py"))
    claims = [
        *(
            claim
            for source in sources
            for claim in _claimed_commands(source.read_text(encoding="utf-8"))
        ),
        grammars.PRESEED_COMMAND.split()[1:],
    ]
    named = {" ".join(claim) for claim in claims}

    assert len(sources) > 1, f"{package} produced no modules to read"
    assert {"doctor --fix", "init"} <= named, f"the package stopped naming its callers: {named}"
    for claim in claims:
        flags = [part for part in claim if part.startswith("-")]
        offered = _options_of([part for part in claim if not part.startswith("-")])
        assert set(flags) <= offered, (
            f"`manicule {' '.join(claim)}` names {flags}, and that command offers {sorted(offered)}"
        )


def _service_over_an_empty_cache(cache: Path) -> ApplicationService:
    """A service whose *configuration* puts the grammar cache in an empty directory.

    Configured rather than fixtured, and the difference is the point: ``doctor`` applies the
    cache directory the installation is configured with before it asks what is present, so a
    test that redirected the pack directly would be answered about a cache the service does not
    read. This is also the state an image-local cache and an air-gapped host are in.
    """
    from tests.app.fakes import FakeBackend  # noqa: PLC0415 - the suite's service double

    backend = FakeBackend()
    backend.settings = Settings(
        plugins={  # pyright: ignore[reportArgumentType] - validated on the way in
            "config": {
                "parser.sourcecode": {
                    "grammar_cache_dir": str(cache),
                    "grammar_manifest_url": UNREACHABLE_MANIFEST,
                }
            }
        }
    )
    return ApplicationService(backend)


def test_every_command_the_pre_seed_names_as_its_caller_actually_calls_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The claim that had no caller, run rather than read.

    Each command named by :func:`~manicule.parsers.grammars.prefetch` is invoked through the
    real command line — real argument parsing, real dispatch, real service — with the pre-seed
    replaced by a spy and the configured cache pointed at an empty directory, since a command
    that pre-seeds a machine which already has every grammar correctly seeds nothing. A command
    that stopped calling it, or that never did, fails here.
    """
    claims = _claimed_commands(grammars.prefetch.__doc__ or "")
    assert claims, "prefetch stopped naming the commands behind it"

    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(tmp_path / "config.toml"))
    service = _service_over_an_empty_cache(tmp_path / "cache")
    seen: list[tuple[str, ...]] = []

    def spy(languages: Sequence[str], **_: object) -> tuple[str, ...]:
        seen.append(tuple(languages))
        return ()

    monkeypatch.setattr(grammars, "prefetch", spy)

    async def execute(op: str, call: Callable[[ApplicationService], Any]) -> Any:
        return await run_op(op, service.workspace, lambda: call(service))

    monkeypatch.setattr(cli, "_execute", execute)
    for claim in claims:
        seen.clear()
        result = CliRunner().invoke(cli.app, ["--json", *claim])

        assert result.exit_code == 0, result.output
        assert seen, f"`manicule {' '.join(claim)}` claims to pre-seed grammars and does not"
        assert set(seen[0]) == set(grammars.DECLARED_LANGUAGES)


async def test_doctor_reports_a_missing_grammar_the_way_missing_grammars_claims_it_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other absent caller: ``missing_grammars`` said ``doctor`` reported it, and it did not.

    Against a real empty cache and an unreachable manifest, so what is asserted is a diagnostic
    produced from the machine's actual state rather than from a stubbed answer.
    """
    service = _service_over_an_empty_cache(tmp_path / "cache")
    monkeypatch.setattr(grammars, "prefetch", _must_not_fetch)

    diagnosis = await service.doctor()
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert check.state == "degraded"
    assert grammars.PRESEED_COMMAND in check.detail
    assert str(tmp_path / "cache") in check.detail, "the configured cache is not what was checked"
    # Every declared language is missing here, and naming all twenty-four is the paragraph
    # that used to bury the fix. The count is the claim; a sample of the names makes it
    # checkable, and `_a_short_list_is_named_in_full` below covers the case where the whole
    # set fits.
    assert f"{len(grammars.DECLARED_LANGUAGES)} missing grammar(s)" in check.detail
    assert "and 18 more" in check.detail


async def test_doctor_names_a_short_list_of_missing_grammars_in_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Counting is for the paragraph case. Two missing grammars are two names."""
    service = _service_over_an_empty_cache(tmp_path / "cache")
    monkeypatch.setattr(grammars, "prefetch", _must_not_fetch)

    def two_missing(_languages: Sequence[str]) -> tuple[str, ...]:
        return ("python", "rust")

    monkeypatch.setattr(grammars, "missing_grammars", two_missing)

    diagnosis = await service.doctor()
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert "python, rust" in check.detail
    assert "more" not in check.detail


def _must_not_fetch(languages: Sequence[str], **_: object) -> tuple[str, ...]:
    """A pre-seed that fails the test rather than the machine's network.

    ``doctor`` without ``--fix`` must not seed anything: it is a report, and one that quietly
    downloaded 84 MB of grammars would be a diagnostic with a side effect.
    """
    msg = f"doctor pre-seeded {list(languages)} without being asked to"
    raise AssertionError(msg)


def test_the_grammar_pack_is_still_distributed_under_a_permissive_license() -> None:
    """Upstream's stated policy is permissive-only; this asserts it rather than trusting it.

    It is the strongest assertion the installed artifact supports. **The wheel enumerates no
    per-grammar licenses** — the manifest carries a group and a size per language and nothing
    else, and the CycloneDX SBOM beside it describes the Rust build dependencies of the
    native extension rather than the 371 grammars. A per-grammar audit therefore cannot be
    written against what is installed; it belongs with the offline bundle, which is where the
    grammars themselves would be enumerated.
    """
    declared = metadata("tree-sitter-language-pack")["License-Expression"]

    assert declared in PERMISSIVE_LICENSES
