"""Measuring whether retrieval is any good, against a running baseline.

    query set -> two systems -> blinded pair -> preference -> report, per intent

**The rule this package makes enforceable.** No retrieval feature ships without a measured
improvement. That is a slogan until something can measure, and it is *worse* than a slogan if
the thing that measures cannot itself be shown to work — a harness that reports plausible
numbers while measuring nothing does not merely fail to enforce the rule, it supplies evidence
for whatever anybody wants to build.

**So the load-bearing part is not the comparison. It is the refusal.**
:mod:`manicule.evaluation.probe` puts every system through a known-answer probe before a
single preference is recorded, and a system that cannot be distinguished from guessing gets no
report at all — not a caveat, not a flag, no report. That check is enforced in three places, on
purpose: when the harness runs, when a record is constructed, and when a report is built from
records read back off disk.

**Pairwise, not absolute.** Two ranked lists side by side and a keypress. Absolute relevance
labels and nDCG are a different and far more expensive instrument, and they come later — and
only if preference stops discriminating between candidate configurations.

**Per intent category, never one average.** A change that helps explanatory questions and
ruins exact-identifier lookups reads as a small win in an averaged number.

Nothing here is imported by ``import manicule``, and nothing here needs a numerical stack: the
statistics are exact, in pure Python, so an evaluation harness is never the thing that could
not be run because a dependency was missing.
"""

from __future__ import annotations

from manicule.evaluation.corpus import CorpusVersion, corpus_version_of, digest_of
from manicule.evaluation.errors import (
    AtChanceError,
    ConfigurationDriftError,
    CorpusMismatchError,
    EvaluationError,
    IncomparableRecordsError,
    PreferenceRecordError,
    ProbeUnusableError,
    QuerySetError,
    UnderpoweredProbeError,
)
from manicule.evaluation.harness import PreferenceHarness
from manicule.evaluation.judging import (
    Judge,
    JudgingStoppedError,
    ScriptedJudge,
    SlotJudge,
    StreamJudge,
    render_pairing,
)
from manicule.evaluation.preference import (
    Pairing,
    Preference,
    PreferenceRecord,
    PreferenceStore,
    Side,
    Slot,
    assign_slots,
    build_record,
)
from manicule.evaluation.probe import (
    DiscriminationProbe,
    ProbeItem,
    ProbeOutcome,
    probe_from_titles,
)
from manicule.evaluation.queries import (
    EvalQuery,
    Intent,
    Provenance,
    QuerySet,
    Thumbs,
    dump_query_set,
    load_query_set,
)
from manicule.evaluation.report import IntentSummary, PreferenceReport, build_report
from manicule.evaluation.systems import (
    CallableSystem,
    ResultItem,
    RetrieverSystem,
    StageObservation,
    SystemResult,
    SystemUnderComparison,
)

__all__ = [
    "AtChanceError",
    "CallableSystem",
    "ConfigurationDriftError",
    "CorpusMismatchError",
    "CorpusVersion",
    "DiscriminationProbe",
    "EvalQuery",
    "EvaluationError",
    "IncomparableRecordsError",
    "Intent",
    "IntentSummary",
    "Judge",
    "JudgingStoppedError",
    "Pairing",
    "Preference",
    "PreferenceHarness",
    "PreferenceRecord",
    "PreferenceRecordError",
    "PreferenceReport",
    "PreferenceStore",
    "ProbeItem",
    "ProbeOutcome",
    "ProbeUnusableError",
    "Provenance",
    "QuerySet",
    "QuerySetError",
    "ResultItem",
    "RetrieverSystem",
    "ScriptedJudge",
    "Side",
    "Slot",
    "SlotJudge",
    "StageObservation",
    "StreamJudge",
    "SystemResult",
    "SystemUnderComparison",
    "Thumbs",
    "UnderpoweredProbeError",
    "assign_slots",
    "build_record",
    "build_report",
    "corpus_version_of",
    "digest_of",
    "dump_query_set",
    "load_query_set",
    "probe_from_titles",
    "render_pairing",
]
