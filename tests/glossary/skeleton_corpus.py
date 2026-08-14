"""A labeled corpus for measuring the widened initials matcher, positives and negatives alike.

Everything here is invented for this suite. There are no real organization names, no product
names taken from anywhere, no URLs and no copied corpus text.

**Separate from :mod:`tests.glossary.corpus` on purpose, and the reason is the same one that put
the description-bearing entries on a supplement page rather than on the twenty-five entry one.**
Every cosine recorded in this suite and in ``docs/retrieval.md`` §8 and §14 is a property of those
exact chunks. A corpus for measuring *detection* needs dozens of labeled lines, and adding them
there would retire those measurements without failing anything — the quietest way a measured
constant goes wrong. What this module does reuse is
:data:`~tests.glossary.corpus.PROSE_ON_THE_GLOSSARY_PAGE`: those three lines are the ticket's own
negatives, they are already placed correctly, and a second copy of them here would be a fixture
that could drift out of step with the one the requirement is written against.

**Every line is labeled, including the ones that must produce nothing.** A detection corpus made
only of definitions measures recall and calls it precision: a detector that admits everything
scores perfectly on it. The negative population is therefore the larger half, and it is chosen
against *this* matcher rather than against the one it replaces — the widening is what has to be
shown not to admit prose, so prose that the narrow matcher already refused for unrelated reasons
proves nothing here.

The categories are the ticket's, one attribute each, so a measurement can be broken down by the
convention it is about rather than reported as a single number that hides which rule moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.glossary.corpus import PROSE_ON_THE_GLOSSARY_PAGE

GLOSSARY_TITLE: Final = "Glossary of platform terminology"
SECOND_TITLE: Final = "Glossary of archive terminology"
HANDBOOK_TITLE: Final = "Operations handbook"


@dataclass(frozen=True)
class Labeled:
    """One source line and what the detector is required to make of it.

    ``acronym`` empty means **no entry at all**, which is a stronger and more useful label than
    "some entry with a different key": a line that must not be a definition must not become one
    under any spelling.
    """

    line: str
    acronym: str
    expansion: str
    category: str

    @property
    def positive(self) -> bool:
        return bool(self.acronym)


# --- conventional acronyms: ordinary word initials, which already worked -----------------------

CONVENTIONAL: Final[tuple[Labeled, ...]] = (
    Labeled(
        "HALO — Health And Latency Observer",
        "HALO",
        "Health And Latency Observer",
        "conventional",
    ),
    Labeled(
        "PRISM — Pipeline Runtime Inspection And Storage Monitor",
        "PRISM",
        "Pipeline Runtime Inspection And Storage Monitor",
        "conventional",
    ),
    Labeled(
        "ORBIT — Operational Retention Backup Index Tool",
        "ORBIT",
        "Operational Retention Backup Index Tool",
        "conventional",
    ),
    Labeled(
        "VECTOR — Vault Export Control Tooling Or Runner",
        "VECTOR",
        "Vault Export Control Tooling Or Runner",
        "conventional",
    ),
    Labeled(
        "N.O.T.E. — Node Observation Trace Export",
        "NOTE",
        "Node Observation Trace Export",
        "conventional",
    ),
)
"""Terms the narrow matcher already read. Present so the widening can be shown not to cost them.

The dotted spelling is here because it is the one case where the *key* is not a substring of the
display, so a skeleton implementation that reached for the display where it wanted the key would
fail on it and on nothing else.
"""

# --- camel-cased compound terms ----------------------------------------------------------------

COMPOUND: Final[tuple[Labeled, ...]] = (
    Labeled("SORT — SecOps Reliability Toolkit", "SORT", "SecOps Reliability Toolkit", "compound"),
    Labeled(
        "HTTP — HyperText Transfer Protocol", "HTTP", "HyperText Transfer Protocol", "compound"
    ),
    Labeled(
        "CIRCA — CloudInfra Retention Capacity Auditor",
        "CIRCA",
        "CloudInfra Retention Capacity Auditor",
        "compound",
    ),
    Labeled(
        "MOSAIC — MicroObject Storage And Index Cache",
        "MOSAIC",
        "MicroObject Storage And Index Cache",
        "compound",
    ),
)
"""Terms whose expansion needs a compound word read as its components.

Each one's *word* initials spell something else — ``SRT``, ``HTP``, ``CRCA``, ``MSAIC`` — so none
of these is admitted by the narrow matcher and none is a case that would have passed anyway.
"""

# --- stylized mixed-case acronyms ---------------------------------------------------------------

STYLIZED: Final[tuple[Labeled, ...]] = (
    Labeled("SaFeR — Service Failure Reporter", "SAFER", "Service Failure Reporter", "stylized"),
    Labeled(
        "ReCAP — Retention Capacity And Planning",
        "RECAP",
        "Retention Capacity And Planning",
        "stylized",
    ),
    Labeled("LiNK — Ledger Node Keeper", "LINK", "Ledger Node Keeper", "stylized"),
    Labeled("AuDiT — Automated Data Trail", "AUDIT", "Automated Data Trail", "stylized"),
)
"""Terms whose expansion spells the uppercase skeleton of the display rather than the key.

``SAFER`` is five characters and ``Service Failure Reporter`` supplies three, which is the whole
convention: the lower-case letters are spelling, not initials. Every skeleton here is exactly
three or four characters, because :data:`~manicule.ingest.glossary.MIN_SKELETON_LENGTH` refuses
two and a writer who capitalizes five letters has written the key.
"""

# --- definitions carrying a trailing description ------------------------------------------------

DESCRIBED: Final[tuple[Labeled, ...]] = (
    Labeled(
        "SORT — SecOps Reliability Toolkit, a package that operations teams install on a host.",
        "SORT",
        "SecOps Reliability Toolkit",
        "described",
    ),
    Labeled(
        "SaFeR — Service Failure Reporter, a component that groups related failures together.",
        "SAFER",
        "Service Failure Reporter",
        "described",
    ),
    Labeled(
        "PRISM — Pipeline Runtime Inspection And Storage Monitor; it samples every queue in turn.",
        "PRISM",
        "Pipeline Runtime Inspection And Storage Monitor",
        "described",
    ),
    Labeled(
        "CIRCA — CloudInfra Retention Capacity Auditor. It reports on space that will not be "
        "reclaimed.",
        "CIRCA",
        "CloudInfra Retention Capacity Auditor",
        "described",
    ),
    Labeled(
        "AuDiT — Automated Data Trail, the record of which process wrote which row and when.",
        "AUDIT",
        "Automated Data Trail",
        "described",
    ),
)
"""The three description boundaries — comma, semicolon, sentence — across all three conventions.

Placed together rather than one per convention because the boundary rule and the initials rule
compose: the cut is awarded by the initials, so a convention that can establish initials on a bare
line and not on a described one would be a bug neither category would catch alone.
"""

# --- lines that must produce nothing --------------------------------------------------------------

COMPOUND_NEGATIVES: Final[tuple[Labeled, ...]] = (
    Labeled(
        "SORT — Secops Reliability Toolkit, a package that operations teams install on a host.",
        "",
        "",
        "compound-negative",
    ),
    Labeled(
        "SORT — Storage Operations Roster, the rota that names who is on call each week.",
        "",
        "",
        "compound-negative",
    ),
    Labeled(
        "MOSAIC — MicroObject Storage Index, the layer that keeps every small object addressable "
        "across zones.",
        "",
        "",
        "compound-negative",
    ),
)
"""What limits the compound rule, one line per way it could have been drawn too wide.

The first is the whole justification for the rule being about *capitalization*: ``Secops`` is the
same letters as ``SecOps`` written without the internal capital, so it is one component, spells
``SRT``, and earns no cut. A rule that split on a dictionary of prefixes would take both and could
not tell a reader why.

The second is the arbitrary-subsequence case. ``SORT``'s letters appear in order inside
``Storage Operations Roster`` — ``S``, then ``O`` and ``t`` inside ``Operations``, then ``R`` — so
a matcher that scanned for them would cut here, and this one must not.

The third is a compound that splits correctly and still spells the wrong term: ``MicroObject
Storage Index`` gives ``MOSI``, not ``MOSAIC``. Splitting is not agreeing.

**Every line here runs past the length bound on purpose**, and an earlier draft of this fixture
did not. Two of these were written short, so refusing the *cut* still left the whole right-hand
side inside :data:`~manicule.ingest.glossary.MAX_EXPANSION_WORDS` and an entry was written from
the fallback rule — which the measurement then counted as a false positive of the widening,
though the widening had nothing to do with it. That confuses two different refusals. Made long,
"no cut" and "no entry" coincide and the label means one thing. The other refusal has its own
population: see :data:`KEPT_WHOLE`.
"""

SKELETON_NEGATIVES: Final[tuple[Labeled, ...]] = (
    Labeled("WEb - when enabled, the process starts automatically.", "", "", "skeleton-negative"),
    Labeled(
        "SaFeR — Service for Escalation Routing, a queue that pages the duty manager.",
        "",
        "",
        "skeleton-negative",
    ),
    Labeled(
        "mDNS — multicast Domain Name System, resolved on the local segment without a server.",
        "",
        "",
        "skeleton-negative",
    ),
)
"""What limits the skeleton rule.

The first is :data:`~manicule.ingest.glossary.MIN_SKELETON_LENGTH` stated as a corpus line rather
than as an argument. ``WEb`` skeletons to ``WE``, and ``when enabled`` — the prefix of the
ticket's own ``API`` negative — spells ``WE``. At a bound of two this line stores ``when enabled``
as the meaning of ``WEB``; at three there is no skeleton, no cut, and the opening-word rule
refuses the whole right-hand side as it always did. Both halves were run.

The second is the arbitrary-subsequence case for the skeleton, which needs its own fixture because
the two widenings fail differently. ``SaFeR`` skeletons to ``SFR``; ``Service for Escalation
Routing`` spells ``SFER`` with its function word and ``SER`` without one, and neither is ``SFR``
— but ``S``, ``f`` and ``r`` do appear in that order in the letters, so a scanning matcher takes
it.

The third is the lower-case initial, refused twice over: ``acronym_shaped`` refuses ``mDNS`` as a
term, and ``initial_skeleton`` independently refuses to read ``DNS`` out of it. Two gates rather
than one, because the skeleton is a public function and a caller reaching it without the shape
gate would otherwise be told that ``mDNS`` is a term spelled ``DNS``.
"""

ORDINARY_MENTIONS: Final[tuple[Labeled, ...]] = (
    *(Labeled(line, "", "", "ordinary") for line in PROSE_ON_THE_GLOSSARY_PAGE),
    Labeled(
        "Escalation - once the second alert fires, the duty manager is paged.", "", "", "ordinary"
    ),
    Labeled(
        "SCOPE - if the collection is empty, the query is answered from the whole workspace.",
        "",
        "",
        "ordinary",
    ),
    Labeled(
        "Retention: these records are kept for two years and then exported.", "", "", "ordinary"
    ),
    Labeled(
        "STATUS - there is no supported route from a stalled run back to a queued one.",
        "",
        "",
        "ordinary",
    ),
    Labeled(
        "LIMIT - unless the operator raises it, the ceiling stays where it was set.",
        "",
        "",
        "ordinary",
    ),
    Labeled(
        "ORDER - whenever two runs finish together, the later identifier wins.", "", "", "ordinary"
    ),
    Labeled(
        "SCHEDULE: while the migration is running, the hourly job is suspended.",
        "",
        "",
        "ordinary",
    ),
    Labeled(
        "RESULT - they are written to the same table and read back by identifier.",
        "",
        "",
        "ordinary",
    ),
)
"""Prose shaped like a definition, on a page that says it is a glossary.

The first three are the ticket's requirement 6, imported from
:data:`~tests.glossary.corpus.PROSE_ON_THE_GLOSSARY_PAGE` rather than copied. The rest are the
same shape — an upper-case token followed by a dash, which a glossary page carries to exactly the
threshold on its own — so every one of them is refused by the expansion rules and by nothing else.
That is the only placement in which refusing them is evidence about anything.

**What this population is and is not evidence for.** Each of these is refused by the opening word
of its right-hand side, and the widened matcher awards no cut on any of them, which is exactly the
claim: widening what may earn a cut must not reach past that rule. It is not evidence that no
widening could — that is what :data:`SKELETON_NEGATIVES`'s first line is for, and it is the one
line here that was watched failing.
"""

KEPT_WHOLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "CPU — central processor, the part that executes instructions",
        "CPU",
        "central processor, the part that executes instructions",
    ),
    (
        "MOSAIC — MicroObject Storage Index, the layer that keeps small objects addressable",
        "MOSAIC",
        "MicroObject Storage Index, the layer that keeps small objects addressable",
    ),
    (
        "ReCAP — Retention Capacity, and the planning that follows from it",
        "RECAP",
        "Retention Capacity, and the planning that follows from it",
    ),
)
"""Lines where an entry is correct and a *cut* is not: source line, key, and the whole expansion.

The conservative half of the boundary rule, measured rather than assumed. None of these has a
boundary prefix that spells its term under any of the four readings, so nothing knows where the
expansion stops; the documented answer is to keep the whole right-hand side and let
``MAX_EXPANSION_WORDS`` bound it.

**Measured separately from the entries-or-not question because the failure is invisible to it.**
A rule that cut at the first description boundary without being convinced would keep all three
entries and store ``central processor``, ``MicroObject Storage Index`` and ``Retention Capacity``
— the count unchanged, only the text wrong. Entry counts cannot see that, which is why the
assertion is on the stored expansion.

Of the three, only ``RECAP`` has a boundary prefix that a free-subsequence scan would accept:
``Retention Capacity`` contains ``R``, ``E``, ``C``, ``A``, ``P`` in order. ``central processor``
has no ``U`` for ``CPU`` and ``MicroObject Storage Index`` no second ``C`` for ``MOSAIC``. Worth
recording because an earlier draft asserted the opposite for all three and was wrong about two of
them: the population is chosen for the conservative fallback, not for requirement 3, and the
fixtures that separate a scanning matcher from this one are ``COMPOUND_NEGATIVES``' second line
and ``SKELETON_NEGATIVES``' second.
"""

LABELED: Final[tuple[Labeled, ...]] = (
    *CONVENTIONAL,
    *COMPOUND,
    *STYLIZED,
    *DESCRIBED,
    *COMPOUND_NEGATIVES,
    *SKELETON_NEGATIVES,
    *ORDINARY_MENTIONS,
)
"""Every labeled line, positives and negatives together, in category order."""


# --- the pages the retrieval half is measured over ------------------------------------------------

DEFINITION_LINES: Final[tuple[str, ...]] = tuple(
    item.line for item in (*CONVENTIONAL, *COMPOUND, *STYLIZED)
)

CONFLICTING_LINE: Final = "HALO — Historical Archive Lookup Overlay"
"""A second definition of ``HALO``, on a second page, spelling the same key by its own initials.

Both readings are legitimate — that is the point. A term with two definitions in scope is a
disagreement the corpus contains, and neither the display spelling nor the skeleton is allowed to
break the tie, which is requirement 7.
"""

SECOND_PAGE_LINES: Final[tuple[str, ...]] = (
    CONFLICTING_LINE,
    "TIDE — Tiered Index Data Export",
    "GaTE — Gateway Transfer Envelope",
)

HOMOGRAPH_USES: Final[tuple[str, ...]] = (
    "Follow the link at the bottom of the runbook to reach the escalation policy.",
    "The orbit of the nightly job drifts by a few minutes each week, which is expected.",
    "Every audit of the export path so far has come back clean, so the schedule stands.",
    "Sort the queue by age before draining it, otherwise the oldest entries starve.",
    "The audit trail is written to the same volume as the index, which is worth changing.",
    "A broken link in the handbook is worth fixing even when the target still resolves.",
)
"""Ordinary English uses of tokens this corpus also defines.

``LINK``, ``ORBIT``, ``AUDIT`` and ``SORT`` are all defined here and all ordinary words, which is
what makes them the population over-expansion is measured on. None of these sentences asks what
the term means, so none may fire an expansion.
"""

# --- the queries ---------------------------------------------------------------------------------

DEFINITIONAL_QUERIES: Final[tuple[tuple[str, str], ...]] = (
    ("What is SORT?", "SORT"),
    ("what is sort?", "SORT"),
    ("What does SecOps Reliability Toolkit stand for?", "SORT"),
    ("What is SaFeR?", "SAFER"),
    ("what does safer stand for?", "SAFER"),
    ("define SAFER", "SAFER"),
    ("What is CIRCA?", "CIRCA"),
    ("What is MOSAIC?", "MOSAIC"),
    ("what is recap?", "RECAP"),
    ("What is AuDiT?", "AUDIT"),
    ("What is LiNK?", "LINK"),
    ("What is PRISM?", "PRISM"),
    ("What is HTTP?", "HTTP"),
    ("What is VECTOR?", "VECTOR"),
)
"""Questions that ask what a defined term means, with the key each must resolve through.

The lower-case and mixed-case spellings are here because requirement 1 is that the *key* resolves
them: ``what is sort?``, ``what does safer stand for?`` and ``What is AuDiT?`` are three
capitalizations of terms whose display spellings are ``SORT``, ``SaFeR`` and ``AuDiT``, and all
three have to arrive at the same normalized key. A suite that only asked in the source's own
spelling would pass on an implementation that looked terms up by display.
"""

UNSUPPORTED_QUERIES: Final[tuple[str, ...]] = (
    "What is ZZQX?",
    "how do I renew a passport at the post office",
    "what is the boiling point of seawater at altitude",
    "which train leaves for the coast on a Sunday morning",
    "what does the third movement of the symphony quote",
    "how long should bread prove in a cold kitchen",
)
"""Questions this corpus cannot answer, mixed between a nonsense token and ordinary English.

The two fail differently and a rejection rate measured on only one of them is measuring the wrong
thing: ``ZZQX`` has no lexical match anywhere, while the rest are fluent questions about subjects
the corpus never mentions and will happily retrieve something.
"""

NON_DEFINITIONAL_QUERIES: Final[tuple[str, ...]] = (
    "sort the queue by age before draining it",
    "follow the link in the runbook to the escalation policy",
    "the audit trail is on the same volume as the index",
    "the orbit of the nightly job drifts each week",
)
"""Questions that *use* a defined token without asking about it.

The requirement they hold down is that exact lexical overlap is not a definitional frame: each of
these contains a term the glossary defines, and none of them may be classified as having been
answered by an explicit definition.
"""


def glossary_page() -> str:
    """The labeled lines as one chunk, which is how 512/64 chunking delivers a glossary page."""
    return f"{GLOSSARY_TITLE}\n\n" + "\n".join(item.line for item in LABELED)


def definitions_page() -> str:
    """The positives only, as one chunk. What the retrieval half is measured over."""
    return f"{GLOSSARY_TITLE}\n\n" + "\n".join(DEFINITION_LINES)


def second_page() -> str:
    return f"{SECOND_TITLE}\n\n" + "\n".join(SECOND_PAGE_LINES)


__all__ = [
    "COMPOUND",
    "COMPOUND_NEGATIVES",
    "CONFLICTING_LINE",
    "CONVENTIONAL",
    "DEFINITIONAL_QUERIES",
    "DEFINITION_LINES",
    "DESCRIBED",
    "GLOSSARY_TITLE",
    "HANDBOOK_TITLE",
    "HOMOGRAPH_USES",
    "KEPT_WHOLE",
    "LABELED",
    "NON_DEFINITIONAL_QUERIES",
    "ORDINARY_MENTIONS",
    "SECOND_PAGE_LINES",
    "SECOND_TITLE",
    "SKELETON_NEGATIVES",
    "STYLIZED",
    "UNSUPPORTED_QUERIES",
    "Labeled",
    "definitions_page",
    "glossary_page",
    "second_page",
]
