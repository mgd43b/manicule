"""manicule is written in American English, and this is what keeps it that way.

The repository had drifted thoroughly British — around 2,150 occurrences across prose,
docstrings, comments, test names and identifiers alike — and a sweep that is not enforced
drifts back one pull request at a time. Three defects were found in one day where a limit was
documented in prose and quietly stopped matching the code; the lesson taken from them is that a
rule the suite enforces cannot drift, and a rule kept as a habit will.

**What this catches, said plainly, because a check whose name outruns what it verifies is worse
than no check at all.** It catches the families listed in :data:`RULES` — the ``-ise``/``-isation``
verbs, ``-our``, ``-re``, ``-ogue``, ``-ce`` nouns, the doubled ``l`` before a vowel suffix, and a
short tail of individual words. **It is an enumeration, not a specification.** It does not catch
every British spelling in English, and it never will: nobody is going to list them, and a list
that claimed to be complete would be believed. ``kerb``, ``plough``, ``gaol`` and several hundred
others would pass here today. What the enumeration is good for is the drift that actually
happens, which is a contributor writing ``normalise`` beside forty existing ``normalize``\\ s.

**Adding a family is the intended repair.** If a British spelling gets through, put it in
:data:`RULES` rather than treating this file's silence as permission.

**Names defined outside this repository are not this file's business**, and :data:`EXTERNAL`
holds them by the exact text that must survive rather than by file. pydantic-settings spells its
own hook ``settings_customise_sources``; ``asyncio.Task.cancelled()`` and GitHub Actions'
``cancelled()`` are both spelled by their definitions. Overriding a hook under a corrected
spelling does not override it, which is a silent failure this repository has already had once —
``pyright`` caught it during the sweep, and only because the method carried ``@override``.

This file is exempt from its own check: it has to name the spellings it looks for. The cost is
that its own prose is unguarded, which is the smallest gap available and is stated rather than
left to be noticed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SUFFIX = r"(?:e|ed|es|er|ers|ing|ation|ations|able|ability)(?![a-z])"
"""What has to follow an ``-is`` stem for it to be a verb rather than a coincidence.

Load-bearing, and each edge was a real false positive while this was being written:
``optimistic``, ``initialism`` and ``analysis`` all contain a listed stem and none is a verb.
``(?![a-z])`` rather than ``\\b`` because a lower-case letter is what tells them apart, while
``\\b`` would also reject ``NormalizedThing`` — camelCase has no word boundary in it.
"""


def _ise(stem: str) -> str:
    return stem + _SUFFIX


RULES: tuple[tuple[str, str], ...] = (
    # -ise / -isation verbs. The stem list is what this repository actually used; a verb
    # nobody here has written yet is not covered, which is the enumeration being an
    # enumeration.
    (_ise("normalis"), "normaliz-"),
    (_ise("organis"), "organiz-"),
    (_ise("serialis"), "serializ-"),
    (_ise("recognis"), "recogniz-"),
    (_ise("optimis"), "optimiz-"),
    (_ise("initialis"), "initializ-"),
    (_ise("summaris"), "summariz-"),
    (_ise("categoris"), "categoriz-"),
    (_ise("authoris"), "authoriz-"),
    (_ise("customis"), "customiz-"),
    (_ise("minimis"), "minimiz-"),
    (_ise("maximis"), "maximiz-"),
    (_ise("synchronis"), "synchroniz-"),
    (_ise("standardis"), "standardiz-"),
    (_ise("specialis"), "specializ-"),
    (_ise("penalis"), "penaliz-"),
    (_ise("sanitis"), "sanitiz-"),
    (_ise("tokenis"), "tokeniz-"),
    (_ise("canonicalis"), "canonicaliz-"),
    (_ise("materialis"), "materializ-"),
    (_ise("parameteris"), "parameteriz-"),
    (_ise("parametris"), "parametriz-"),
    (_ise("characteris"), "characteriz-"),
    (_ise("centralis"), "centraliz-"),
    (_ise("localis"), "localiz-"),
    (_ise("neutralis"), "neutraliz-"),
    (_ise("theoris"), "theoriz-"),
    (_ise("capitalis"), "capitaliz-"),
    (_ise("finalis"), "finaliz-"),
    (_ise("containeris"), "containeriz-"),
    (_ise("internationalis"), "internationaliz-"),
    (_ise("monopolis"), "monopoliz-"),
    (_ise("criticis"), "criticiz-"),
    (_ise("amortis"), "amortiz-"),
    (_ise("memois"), "memoiz-"),
    (_ise("quantis"), "quantiz-"),
    (_ise("symmetris"), "symmetriz-"),
    (_ise("realis"), "realiz-"),
    (_ise("visualis"), "visualiz-"),
    (_ise("prioritis"), "prioritiz-"),
    (_ise("randomis"), "randomiz-"),
    (_ise("utilis"), "utiliz-"),
    (_ise("generalis"), "generaliz-"),
    (_ise("stabilis"), "stabiliz-"),
    (_ise("vectoris"), "vectoriz-"),
    (_ise("weaponis"), "weaponiz-"),
    # -esis verbs, where the noun keeps its s and only the verb moves: ``emphasis`` stays,
    # ``emphasise`` does not. The suffix guard is what tells them apart.
    (_ise("synthesis"), "synthesiz-"),
    (_ise("parenthesis"), "parenthesiz-"),
    (_ise("emphasis"), "emphasiz-"),
    (_ise("hypothesis"), "hypothesiz-"),
    # -yse verbs. ``analysis`` and ``analyses`` are both correct here and both excluded by
    # the suffix guard; only the verb forms are British.
    (r"analys(?:e|ed|ing|er|ers)(?![a-z])", "analyz-"),
    (r"catalys(?:e|ed|es|ing)(?![a-z])", "catalyz-"),
    (r"paralys(?:e|ed|es|ing)(?![a-z])", "paralyz-"),
    # -our
    (r"behaviour", "behavior"),
    (r"colour", "color"),
    (r"favour", "favor"),
    (r"flavour", "flavor"),
    (r"harbour", "harbor"),
    (r"honour", "honor"),
    (r"labour", "labor"),
    (r"neighbour", "neighbor"),
    (r"rigour", "rigor"),
    (r"valour", "valor"),
    (r"vapour", "vapor"),
    (r"armour", "armor"),
    (r"endeavour", "endeavor"),
    (r"savour", "savor"),
    (r"odour", "odor"),
    (r"rumour", "rumor"),
    (r"splendour", "splendor"),
    (r"demeanour", "demeanor"),
    # -re
    (r"centre(?![a-z])|centres(?![a-z])|centred(?![a-z])|centring(?![a-z])", "center-"),
    (r"theatre", "theater"),
    (r"fibre(?![a-z])|fibres(?![a-z])", "fiber"),
    (r"calibre", "caliber"),
    (r"litre(?![a-z])|litres(?![a-z])", "liter"),
    (r"metre(?![a-z])|metres(?![a-z])", "meter"),
    (r"sombre", "somber"),
    (r"spectre", "specter"),
    (r"lustre", "luster"),
    # -ogue, where American drops the tail. ``dialogue`` is deliberately absent: American
    # English keeps it for a conversation and shortens it only for the UI widget, and no rule
    # here can tell those apart. A family that would misfire on correct prose is worse than an
    # uncovered one, because the repair for a check that cries wolf is to switch it off.
    (r"catalogue", "catalog"),
    (r"analogue", "analog"),
    # -ce nouns
    (r"licence", "license"),
    (r"defence", "defense"),
    (r"offence", "offense"),
    (r"pretence", "pretense"),
    # A doubled l before a vowel suffix, which American English does not double.
    (r"labell(?:ed|ing|er|ers)(?![a-z])", "label-"),
    (r"cancell(?:ed|ing)(?![a-z])", "cancel-"),
    (r"modell(?:ed|ing|er|ers)(?![a-z])", "model-"),
    (r"signall(?:ed|ing)(?![a-z])", "signal-"),
    (r"totall(?:ed|ing)(?![a-z])", "total-"),
    (r"travell(?:ed|ing|er|ers)(?![a-z])", "travel-"),
    (r"diall(?:ed|ing)(?![a-z])", "dial-"),
    (r"fuell(?:ed|ing)(?![a-z])", "fuel-"),
    (r"levell(?:ed|ing)(?![a-z])", "level-"),
    (r"channell(?:ed|ing)(?![a-z])", "channel-"),
    # A single l where American English doubles it.
    (r"fulfil(?![lm])", "fulfill"),
    (r"fulfilment", "fulfillment"),
    # ``(?![lm])`` on both stems, so ``enrolment`` and ``fulfilment`` are reported once by the
    # rule that knows the right ending rather than twice, the second time with wrong advice.
    (r"enrol(?![lm])", "enroll"),
    (r"enrolment", "enrollment"),
    (r"instalment", "installment"),
    (r"skilful", "skillful"),
    (r"wilful", "willful"),
    # Individual words.
    (r"artefact", "artifact"),
    (r"judgement", "judgment"),
    (r"acknowledgement", "acknowledgment"),
    (r"ageing", "aging"),
    (r"grey(?![a-z])|greyscale", "gray"),
    (r"programme(?![a-z])|programmes(?![a-z])", "program"),
    (r"whilst", "while"),
    (r"amongst", "among"),
    (r"speciality", "specialty"),
    (r"sceptic", "skeptic"),
    (r"orientated", "oriented"),
    (r"aluminium", "aluminum"),
    (r"sulphur", "sulfur"),
    (r"encyclopaedia", "encyclopedia"),
    (r"mediaeval", "medieval"),
    (r"manoeuvre", "maneuver"),
    (r"misspelt", "misspelled"),
    (r"storeys", "stories"),
    (r"jewellery", "jewelry"),
    (r"moustache", "mustache"),
    (r"pyjamas", "pajamas"),
    (r"connexion", "connection"),
    (r"inflexion", "inflection"),
    (r"cancelled(?![a-z])", "canceled"),
)
"""Each family as a regular expression, with what to write instead.

Matched case-insensitively, so ``NORMALISER_VERSION`` and ``FakeOrganisation`` are caught
alongside prose. The replacement text is a hint for the failure message, not a rewrite rule —
``normaliz-`` because the right ending depends on the word.
"""

EXTERNAL: tuple[tuple[str, str], ...] = (
    (
        "settings_customise_sources",
        "pydantic-settings names its own hook. Overriding it under a corrected spelling does "
        "not override it — the base method is simply never replaced, and the settings sources "
        "silently revert to the default order.",
    ),
    (
        "CancelledError",
        "asyncio's exception type.",
    ),
    (
        "cancelled()",
        "two functions, spelled this way by both definitions: asyncio.Task.cancelled(), and "
        "GitHub Actions' status function that the CI workflow branches on. Prose quoting "
        "either by name is covered by the same entry, which is why it is the bare call and "
        "not a longer fragment.",
    ),
    (
        'ac:name="colour"',
        "Confluence spells the status macro's parameter this way, so a storage-format fixture "
        "has to carry the real name. The test using it asserts that the value never reaches "
        "the index; respelled, the fixture would stop resembling the markup it stands in for "
        "and the assertion would prove nothing about a real page.",
    ),
)
"""Text that must survive because something outside this repository defines it.

Held as the exact string rather than as a file, so an exemption forgives one name and not
everything that happens to share a file with it. Every entry carries the consequence of getting
it wrong, because "external name" on its own is an assertion a reader cannot check.
"""

QUOTED: tuple[tuple[str, str], ...] = (
    (
        "`normalise_acronym` and\n`normalise_expansion`",
        "docs/ingest.md names both pre-rename functions while explaining why the glossary "
        "fingerprint moved. An operator reading that section is holding a stored digest that "
        "those names produced, so the paragraph is useless without them.",
    ),
    (
        "``licence`` became ``license``",
        "grammar_bundle.SCHEMA_VERSION says which key the bump renamed. A reader looking at a "
        "schema-1 manifest needs the key that manifest actually carries.",
    ),
    (
        "respelled ``Optimisation``",
        "tests/glossary/corpus.py records the one edit its bytes have taken, because every "
        "cosine measured against that chunk was measured before it.",
    ),
)
"""British spellings this repository names on purpose, while explaining that it renamed them.

Separate from :data:`EXTERNAL` because the justification is different and collapsing the two
would hide it: these are *our* former spellings, quoted so a document explaining a rename can
say what the old name was. A migration note that cannot print the name an operator is holding
is not a migration note.

Held as the whole quoting phrase rather than the bare word, so the exemption forgives the
sentence that explains the rename and not the next casual use of it.
"""

SKIP_SUFFIXES = frozenset({".png", ".lock"})
SKIP_NAMES = frozenset({"LICENSE", "uv.lock"})
"""``LICENSE`` is the GPL's text, reproduced verbatim; ``uv.lock`` is generated. Neither is
this project's prose, and editing either to satisfy a spelling check would be a defect."""

SELF = Path(__file__).name

MATCHERS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), advice) for pattern, advice in RULES
)


def _tracked_files() -> list[Path]:
    """Every tracked text file, from git rather than from a walk of the working tree.

    A walk would read `.venv`, `__pycache__` and whatever else a developer left lying about,
    and the check would pass or fail depending on the machine it ran on.
    """
    listing = subprocess.run(  # noqa: S603 - fixed argv, no shell, no caller input
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],  # noqa: S607 - git off PATH, as CI runs it
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in listing.split("\0") if name]


def _forgiven(text: str, start: int, end: int) -> bool:
    """Whether a match sits inside an exempted phrase rather than standing on its own."""
    return any(
        occurrence <= start and occurrence + len(phrase) >= end
        for phrase, _ in (*EXTERNAL, *QUOTED)
        for occurrence in _occurrences(text, phrase)
    )


def _occurrences(text: str, name: str) -> list[int]:
    found: list[int] = []
    index = text.find(name)
    while index >= 0:
        found.append(index)
        index = text.find(name, index + 1)
    return found


def british_spellings(text: str) -> list[tuple[str, str, int]]:
    """Every listed British spelling in ``text``, as ``(written, what to write, offset)``.

    Over the whole text rather than line by line, because an exempted phrase in :data:`QUOTED`
    may be wrapped across a line break and a per-line scan would not see it whole.
    """
    return [
        (match.group(0), advice, match.start())
        for matcher, advice in MATCHERS
        for match in matcher.finditer(text)
        if not _forgiven(text, match.start(), match.end())
    ]


def test_no_british_spelling_reaches_the_tracked_tree() -> None:
    """The sweep, held in place.

    Failure names the file, the line, the word and what to write, because a check that says only
    "British spelling found" makes the reader do the search a second time.
    """
    offenders: list[str] = []

    for path in _tracked_files():
        if path.name == SELF or path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        relative = path.relative_to(REPO_ROOT)
        for written, advice, offset in british_spellings(text):
            line = text.count("\n", 0, offset) + 1
            offenders.append(f"{relative}:{line}: {written!r} — write {advice!r}")

    assert not offenders, (
        "manicule is written in American English. "
        + f"{len(offenders)} British spelling(s) found:\n"
        + "\n".join(offenders[:40])
        + ("\n…" if len(offenders) > 40 else "")
    )


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("the normalised value", "normalised"),
        ("NORMALISER_VERSION", "NORMALISER"),
        ("class FakeOrganisation:", "Organisation"),
        ("def normalise_acronym(", "normalise"),
        ("its behaviour changed", "behaviour"),
        ("the colour scheme", "colour"),
        ("a permissive licence", "licence"),
        ("the labelled entry", "labelled"),
        ("an artefact of the build", "artefact"),
        ("a judgement call", "judgement"),
        ("in self-defence", "defence"),
        ("the catalogue of plugins", "catalogue"),
        ("centred on the page", "centred"),
        ("centring the column", "centring"),
        ("we analysed the corpus", "analysed"),
        ("the run was cancelled", "cancelled"),
        ("travelling between zones", "travelling"),
        ("it emphasised the point", "emphasised"),
    ],
)
def test_the_check_catches_a_spelling_of_each_family(sentence: str, expected: str) -> None:
    """Calibration. A guard that has only ever passed is not evidence that it bites.

    Every family in :data:`RULES` is a claim that something would be caught, and the way that
    claim goes wrong is a regular expression that matches nothing — which looks exactly like a
    clean tree.
    """
    found = [written for written, _, _ in british_spellings(sentence)]
    assert expected in found, f"{sentence!r} was not caught; the rule for it matches nothing"


@pytest.mark.parametrize(
    "sentence",
    [
        "it is optimistic about the shard",  # not ``optimise``
        "an initialism, not an acronym",  # not ``initialise``
        "needs value-flow analysis",  # not ``analyse``
        "the analyses disagree",  # noun plural, correct here
        "emphasis is a field, not a comment",  # noun, correct here
        "a hypothesis nobody tested",  # noun, correct here
        "otherwise the payload says so",  # not an ``-ise`` verb
        "pairwise and clockwise comparisons",  # nor are these
        "the enterprise licence server",  # ``enterprise`` is correct; ``licence`` is
        "advertised, advised and comprised",  # all correct in American English
        "a promise it raised and revised",  # so are these
        "the laboratory result",  # not ``labour``
        "collaborate on the parametrization",  # already American
        "a laboratory result nobody collaborated on",  # not ``labour``
        "fulfilling the request",  # correct under either spelling of the verb
        "a dialogue between two operators",  # American English keeps this one
        "the greyhound slipped its lead",  # ``grey`` is listed; ``greyhound`` is not it
    ],
)
def test_the_check_leaves_american_english_alone(sentence: str) -> None:
    """The other half of calibration: what must *not* trip.

    Each of these was a real false positive while the rules were being written. ``optimistic``,
    ``initialism`` and ``analysis`` are the reason :data:`_SUFFIX` exists at all; a check that
    fails on correct prose gets switched off, which is a slower way of not having one.
    """
    found = [written for written, _, _ in british_spellings(sentence)]
    permitted = {"licence"}  # the one deliberate offender, in the ``enterprise`` case
    assert not (set(found) - permitted), f"{sentence!r} was wrongly flagged: {found}"


def test_every_exemption_is_still_present_and_still_needed() -> None:
    """An exemption for a name nothing uses is a hole nobody is watching.

    Exemptions are how this check gets quietly disabled — one forgiving string at a time — so
    each has to name text the tree actually contains. When a dependency corrects its own
    spelling, or the last caller goes away, the entry fails here rather than lingering as
    permission for a spelling nobody needs any more.
    """
    corpus = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in _tracked_files()
        if path.suffix not in SKIP_SUFFIXES and path.is_file()
    )

    for name, reason in (*EXTERNAL, *QUOTED):
        assert name in corpus, (
            f"{name!r} is exempted as a name defined elsewhere, and nothing in the tree uses "
            f"it any more. Delete the exemption rather than leaving it to forgive something "
            f"it was never written for. The reason given was: {reason}"
        )
        assert reason.endswith("."), f"{name!r} is exempted without a written reason"
