"""A synthetic corpus in which the baseline demonstrably fails.

Everything here is invented for this suite. There are no private glossary entries, no
organisation names, no URLs and no copied corpus text.

**Three earlier versions of this fixture did not reproduce the bug, and that is the most
useful thing in this module.** Measured against the real embedder, with the definition ranked
among the passages listed:

===================================================  ===============================
Fixture                                              Rank of the definition
===================================================  ===============================
The glossary line as its own short passage, thirty   **1 of 33** — no failure
ordinary uses of "now" around it
The glossary as one chunk holding 25 entries         **1 of 31**, cosine 0.4655 —
                                                     ranked fine, but below the noise
                                                     floor, so confidence said ``none``
The above, plus fifteen passages that *use* the      **15 of 61** — the failure
acronym in running text
===================================================  ===============================

Those ranks are the historical record of how the fixture was arrived at, measured by a method
this suite no longer performs — attempting to re-derive them produced different figures, so they
are left as the account they are rather than restated as current fact. What *is* current is
:mod:`tests.glossary.test_measured`, which measures the shipped retriever and states the numbers
it produced. The corpus is 62 chunks since the supplement was added, not 61.

So reproducing it needed two ingredients, and a fixture missing either one proves nothing:

1. **The definition is diluted inside a chunk.** Chunking is 512/64, so a glossary page
   arrives as one chunk holding dozens of entries and any single definition is a fortieth of
   the vector. A fixture where the definition is its own one-line passage makes it the trivial
   nearest neighbour of any question naming the term — the definition and the question are
   nearly the same string.
2. **The term is used far more often than it is defined.** This is what a real corpus looks
   like once a term exists: it is spelled out where it is defined and written as the acronym
   everywhere else. Those usage passages are short, on topic, and contain the acronym, so
   similarity ranks every one of them above the definition.

The distractors deliberately **do not** contain the literal expansion. An earlier version gave
every passage the phrase "network operations workspace", which made the expanded query rank the
glossary *last* — a fixture that would have proved expansion harmful, for a reason no corpus
has.

**A second glossary page carries the description-boundary material**, and it is separate from
the first for a reason that outlives this change: the twenty-five entry chunk is what every
cosine in this suite and in ``docs/retrieval.md`` §8 was measured against, so adding a line to
it would retire those measurements without failing anything. See :data:`SUPPLEMENT_ENTRIES`.
What the supplement holds, each a category the extraction rules can get wrong:

* :data:`DESCRIBED_ENTRY` — a definition trailed by nine words of prose, which the detector
  refused wholesale before it could tell an expansion from a description.
* :data:`STYLIZED_ENTRY` — the same shape with a stylized spelling, so display preservation and
  description-trimming are proved *together* rather than on two fixtures each built for one.
* :data:`PROSE_ON_THE_GLOSSARY_PAGE` — three lines that look like definitions and are not, on a
  page whose title supplies the evidence that carries them to exactly the threshold. Two of the
  three were recorded as real entries before this fixture existed.
"""

from __future__ import annotations

from typing import Final

GLOSSARY_TITLE: Final = "Glossary of terms"

ACRONYM: Final = "NOW"
EXPANSION: Final = "Network Operations Workspace"

DESCRIBED_ACRONYM: Final = "NOVA"
DESCRIBED_EXPANSION: Final = "Network Operations Visibility Assistant"
DESCRIBED_ENTRY: Final = (
    f"{DESCRIBED_ACRONYM} — {DESCRIBED_EXPANSION}, a service used to correlate operational "
    "signals across systems."
)
"""A definition trailed by a description: four words of term, nine of prose.

Recorded as no entry at all before ``core_expansion`` existed — thirteen words is over
``MAX_EXPANSION_WORDS``, so the whole line was refused and nothing downstream could help.
"""

STYLIZED_DISPLAY: Final = "ReLAY"
STYLIZED_ACRONYM: Final = "RELAY"
STYLIZED_EXPANSION: Final = "Retention Export Ledger And Yield"
STYLIZED_ENTRY: Final = (
    f"{STYLIZED_DISPLAY} — {STYLIZED_EXPANSION}, the nightly export path between zones."
)
"""A term whose spelling is stylized, carrying a description as well.

Two requirements on one line on purpose. A fixture proving stylized display on a bare definition
and a second proving description-trimming on an ordinary term would leave the case where both
apply untested, and that is the case a real glossary is full of.
"""

PROSE_ON_THE_GLOSSARY_PAGE: Final[tuple[str, ...]] = (
    "NOTE - this paragraph describes an operational consideration, not a term.",
    "Today - the system is operating normally.",
    "API - when enabled, the process starts automatically.",
)
"""Prose that looks like a definition, on a page that says it is a glossary.

**The placement is the whole test.** On an ordinary page a spaced hyphen scores 0.45 against a
0.60 threshold, so every line here is refused by arithmetic that has nothing to do with this
feature and a fixture placed there passes whether the detector is right or wrong. Under
:data:`SUPPLEMENT_TITLE` the context evidence adds the missing 0.15, the total is exactly the
threshold, and the scoring gate admits all three — so this is the only placement where refusing
them proves anything. Do not move them off a glossary page.

Two of the three were recorded as real entries before ``core_expansion`` existed: ``NOTE`` at
nine words and ``API`` at six, both under :data:`~manicule.ingest.glossary.MAX_EXPANSION_WORDS`.
"""

SUPPLEMENT_TITLE: Final = "Glossary supplement: newer terminology"

SUPPLEMENT_ENTRIES: Final[tuple[str, ...]] = (DESCRIBED_ENTRY, STYLIZED_ENTRY)
"""The definitions that carry descriptions, on a page of their own.

**A second document rather than two more lines on the first**, and that is a constraint this
suite has to keep. Every cosine recorded against :data:`GLOSSARY_ENTRIES` is a property of that
exact chunk; growing it would leave ``STRONG_SIMILARITY`` and the §8 tables citing a corpus that
no longer exists, which is the quietest way a measured constant goes wrong. Measured directly:
adding these two lines and the prose to the twenty-five entry page moved its cosine to ``What is
NOW?`` from 0.4432 to 0.4984. Both are still below the 0.54 noise floor, so no claim would have
*broken* — the numbers would just no longer have been the numbers.

A separate page is also the better fixture. A term defined on one page while a question is asked
against a corpus of several is what a real deployment looks like, and it means promotion has to
find the right document rather than the only one.
"""

GLOSSARY_ENTRIES: Final[tuple[str, ...]] = (
    "ATLAS — Automated Transfer Ledger And Scheduler",
    "BRIDGE — Batch Reconciliation Interface Data Gateway Engine",
    "CANOPY — Capacity And Node Observation Pipeline Yard",
    "DELTA — Distributed Event Ledger Transfer Agent",
    "EMBER — Event Metrics Buffer And Export Relay",
    "GRANITE — Graph Retention And Node Index Tooling Environment",
    "ISOTOPE — Index Storage Optimisation Tooling Endpoint",
    "JUNIPER — Job Under Node Inspection Pipeline Runner",
    "KRYPTON — Key Rotation Yield Protocol Token Notary",
    "LANTERN — Ledger And Node Telemetry Export Runner Node",
    "MISTRAL — Metrics Ingest Stream Transfer And Load",
    "NIMBUS — Node Inventory Metrics Bucket Update Service",
    f"{ACRONYM} — {EXPANSION}",
    "OBSIDIAN — Observation Buffer Storage Index And Node",
    "PUMICE — Pipeline Update Metrics Index Collection Engine",
    "QUARTZ — Query Uptime And Retention Tracking Zone",
    "RAVINE — Retention And Vault Index Node Export",
    "SIERRA — Storage Index Export Retention Relay Agent",
    "TUNDRA — Tooling Under Node Data Retention Agent",
    "UMBER — Update Metrics Buffer And Export Relay",
    "VESSEL — Vault Export Storage Service Endpoint Layer",
    "WILLOW — Workload Index Ledger Log Observation Window",
    "XENON — Export Node Observation Notary",
    "YARROW — Yield And Retention Reporting Observation Window",
    "ZEPHYR — Zone Export Pipeline Health Yield Runner",
)
"""Twenty-five terms on one page. The count is the dilution, and the dilution is the point.

**Do not add entries here.** This tuple is the chunk every cosine recorded in this suite and in
``docs/retrieval.md`` §8 was measured against — 0.4655 for ``What is NOW?``, 0.702 and 0.747 for
the two phrasings :data:`~manicule.retrieval.confidence.STRONG_SIMILARITY` is justified from. A
line added here changes the chunk's vector and silently retires all three: the constants would
still cite measurements, and the corpus they name would no longer be the corpus that produced
them. New material goes on :data:`SUPPLEMENT_ENTRIES`, which is a second document and leaves
this one byte-identical.
"""

ORDINARY: Final[tuple[str, ...]] = (
    "The nightly reconciliation job is running now, so the report will lag by one cycle. "
    "Whoever is on call is told before the run starts, and the lag is visible until the next "
    "cycle clears it.",
    "If the queue is empty now, the backlog cleared while the workers were restarting. The "
    "drain is recorded, and whoever asked for it can read the counters afterwards.",
    "Stop the scheduler now and drain the pending batch before applying the upgrade. There is "
    "a checklist for this, and the on-call engineer opens it during an incident.",
    "Right now the cache holds about four thousand entries and evicts the oldest first. A "
    "cache that stops evicting is the first sign of a stalled worker.",
    "By now every worker has picked up the new configuration file. You can see which nodes "
    "have reloaded and which are still serving the previous revision.",
    "The old export format is deprecated; new clients should now use the streaming endpoint. "
    "The change was published a fortnight before it took effect.",
    "Until now the retention window was seven days, and it is fourteen from this release. The "
    "archive is sized against whichever window is in force.",
    "Operators who used to edit the file by hand now run the generator instead. The generated "
    "copy is kept and the hand-edited one is refused.",
    "Now that the migration has finished, the read replica can be promoted. The promotion is "
    "done through the tooling rather than from a shell, so the change is recorded.",
    "You can shut the service down now; nothing is holding a write lock. Confirm that before "
    "the window opens.",
    "The build takes eleven minutes now, down from nineteen before the cache was added. Both "
    "figures are charted so a regression is visible.",
    "It is now safe to remove the temporary directory the installer created. Those are swept "
    "weekly and the reclaimed space is reported.",
    "Rotate the signing key now if it has not been rotated in the last ninety days. The age of "
    "every key is tracked and a warning is raised before expiry.",
    "The dashboard refreshes every thirty seconds, so the figure you see now is nearly live. "
    "It was built in so nobody has to run the query by hand.",
    "We now retain the original bytes for every document that permits it. The storage cost was "
    "reviewed before the setting was turned on.",
    "Previously the timeout was fixed; it is now derived from the observed latency. The "
    "derived value can be pinned if it moves badly.",
    "Now and then the connector reports a document it has already seen, which is harmless. The "
    "repeats are counted so a genuine loop can be told from noise.",
    "Restart the daemon now, then confirm the health endpoint answers within two seconds. The "
    "expected timings are in the runbook.",
    "The index is warm now, so the first query after a deploy is no longer slow. The warm-up "
    "is measured and an alert fires if it exceeds a minute.",
    "That limit was raised, and the ceiling is now two hundred megabytes per file. Every limit "
    "in force is listed with whoever last changed it.",
    "Anyone reading this now should assume the older instructions are out of date. The current "
    "runbook was moved last quarter.",
    "The parser used to guess the encoding; it now reads the declared one and refuses to "
    "guess. The change rejects some older uploads, which was intended.",
    "Now would be a good time to take a backup, before the schema change is applied. The "
    "backup verb and its destination are already configured.",
    "The worker pool is idle now, which is what you want before a rolling restart. Read the "
    "pool depth once rather than host by host.",
    "Numbers that were estimates are now measured, and the two disagree by about a tenth. Both "
    "are recorded so the correction is auditable.",
    "Support for the legacy flag is gone now; the replacement takes the same value. The flags "
    "that survived the cull are documented.",
    "Now, after three runs, the watermark has advanced past the point that used to stall. "
    "Watch the watermark during every sync.",
    "The alert fires immediately now rather than after the second consecutive failure. That "
    "was tuned after a quiet outage went unnoticed.",
    "Everything that was manual is now scripted, which is why the runbook is shorter. The "
    "scripts and the runbook sit side by side.",
    "The team meets every Tuesday and the agenda is published the evening before, now that the "
    "calendar integration works.",
    # Definitional shapes. A question like "What is X?" is nearest to text that is itself a
    # question, so a fixture with none of these understates how hard the real case is.
    "What is the retention window now? It is fourteen days for documents and thirty for the "
    "audit log, and the two were deliberately set apart.",
    "What is the current batch size now? Sixty-four by default, and the scheduler halves it "
    "whenever the previous batch overran its budget.",
    "What is the supported version now? Only the last two minor releases are supported, and "
    "anything older is refused at startup rather than warned about.",
    "What is the right thing to do now that the export has failed twice? Take the third "
    "failure as terminal and raise it rather than retrying a fourth time.",
    "What counts as a healthy node now? One that has answered the health endpoint within two "
    "seconds on each of the last three polls.",
    "What is the default profile now? Balanced, which runs both legs and no reranker, and it "
    "is what a fresh install starts with.",
    "What is stored now that the retention setting has changed? The original bytes, the "
    "extracted text and one version record for each revision seen.",
    "What is the escalation path now? The on-call engineer first, then the duty manager after "
    "fifteen minutes without acknowledgement.",
    "What is the maximum file size now? Two hundred megabytes, above which the connector skips "
    "the file and records the reason.",
    "What is the difference between the two counters now? One counts rows fetched and the "
    "other counts rows that survived the join, and they diverge on a dirty index.",
    "What is the plan now that the grammar bundle ships with the install? Air-gapped machines "
    "no longer need a download step at all.",
    "What is the fallback now if the model cannot be loaded? The process refuses to start "
    "rather than serving a degraded index nobody was told about.",
    "What is the schedule now? Hourly during the working day and every six hours overnight, "
    "which halves the load without lengthening the worst-case lag.",
    "What is the owner of this queue now? The platform team took it over when the previous "
    "group was folded into it.",
    "What is the timeout now for a single fetch? Thirty seconds, after which the document is "
    "recorded as failed and retried on the next run.",
)
"""Forty-five ordinary English uses of "now". None is about the defined term."""

IN_USE: Final[tuple[str, ...]] = (
    "Raise the ticket in NOW and attach the incident timeline; the duty manager reads it from "
    "there rather than from mail.",
    "Access to NOW is granted with the rest of the platform tooling, and it is revoked the "
    "same way when somebody leaves.",
    "The NOW dashboard shows queue depth, worker health and the age of the oldest "
    "unacknowledged alert, refreshed every thirty seconds.",
    "If NOW is unreachable, fall back to the static runbook copy and record the outage against "
    "the tooling component rather than the service.",
    "Every change to a production limit is logged in NOW, so the person who raised it and the "
    "person who approved it are both recorded.",
    "NOW replaced three separate spreadsheets, which is why the migration took a quarter "
    "rather than a fortnight.",
    "The weekly report is generated from NOW and circulated on Monday morning before the "
    "planning meeting.",
    "Do not edit the NOW records by hand; the importer overwrites them on the next run and the "
    "edit is lost without warning.",
    "New joiners are given read access to NOW on their first day and write access once their "
    "on-call shadowing is complete.",
    "The retention policy for NOW records is two years, after which they are exported and the "
    "originals removed.",
    "NOW has an API, and the client library is published alongside the rest of the platform "
    "packages.",
    "Search in NOW is scoped to your own team by default; widening it needs the platform role.",
    "When an incident closes, the timeline in NOW becomes the record of what happened and is "
    "no longer editable.",
    "Latency graphs in NOW are drawn from the same series the alerts read, so the two cannot "
    "disagree.",
    "Archived NOW pages stay searchable but are marked so nobody mistakes them for current "
    "guidance.",
)
"""Fifteen passages that *use* the acronym without defining it.

These are what bury the definition, and they are not adversarial: a term that exists is
referred to far more often than it is explained. A corpus without them is a corpus where the
definition is the only passage containing the token, which no real one is.
"""

# --- the queries the regression cases run -------------------------------------------------

QUERY_ACRONYM: Final = "What is NOW?"
QUERY_LOWER: Final = "what is now?"
QUERY_MIXED_CASE: Final = "What is Now?"
QUERY_PUNCTUATED: Final = "What is N.O.W.?"
QUERY_FULL: Final = f"What is the {EXPANSION}?"
QUERY_ORDINARY_USE: Final = "should I restart the daemon now or wait for the window"
QUERY_ABSENT: Final = "What is ZZQX?"
QUERY_DESCRIBED: Final = f"What is {DESCRIBED_ACRONYM}?"
QUERY_STYLIZED: Final = f"What is {STYLIZED_DISPLAY}?"
QUERY_STYLIZED_LOWER: Final = f"what does {STYLIZED_ACRONYM.lower()} stand for?"
QUERY_UNRELATED: Final = "how do I renew a passport at the post office"
"""A question about nothing in this corpus, phrased as a real question rather than as a
nonsense token — ``QUERY_ABSENT`` already covers a term that does not exist, and the two fail
differently: one has no lexical match anywhere, the other is ordinary English about a subject
the corpus never mentions."""


def glossary_page() -> str:
    """The glossary as one chunk, which is how 512/64 chunking delivers it.

    Unchanged since the measurements in ``docs/retrieval.md`` §8 were taken, and it has to stay
    that way — see :data:`GLOSSARY_ENTRIES`.
    """
    return f"{GLOSSARY_TITLE}\n\n" + "\n".join(GLOSSARY_ENTRIES)


def supplement_page() -> str:
    """The second glossary: two described definitions and three lines of prose, one chunk.

    The prose sits *inside* it rather than on a page of its own, because a definition-shaped
    sentence is only hard to refuse where the page vouches for it.
    """
    return (
        f"{SUPPLEMENT_TITLE}\n\n"
        + "\n".join(SUPPLEMENT_ENTRIES)
        + "\n"
        + "\n".join(PROSE_ON_THE_GLOSSARY_PAGE)
    )


def passages() -> tuple[str, ...]:
    """Every non-glossary passage, in a stable order."""
    return (*ORDINARY, *IN_USE)


__all__ = [
    "ACRONYM",
    "DESCRIBED_ACRONYM",
    "DESCRIBED_ENTRY",
    "DESCRIBED_EXPANSION",
    "EXPANSION",
    "GLOSSARY_ENTRIES",
    "GLOSSARY_TITLE",
    "IN_USE",
    "ORDINARY",
    "PROSE_ON_THE_GLOSSARY_PAGE",
    "QUERY_ABSENT",
    "QUERY_ACRONYM",
    "QUERY_DESCRIBED",
    "QUERY_FULL",
    "QUERY_LOWER",
    "QUERY_MIXED_CASE",
    "QUERY_ORDINARY_USE",
    "QUERY_PUNCTUATED",
    "QUERY_STYLIZED",
    "QUERY_STYLIZED_LOWER",
    "QUERY_UNRELATED",
    "STYLIZED_ACRONYM",
    "STYLIZED_DISPLAY",
    "STYLIZED_ENTRY",
    "STYLIZED_EXPANSION",
    "SUPPLEMENT_ENTRIES",
    "SUPPLEMENT_TITLE",
    "glossary_page",
    "passages",
    "supplement_page",
]
