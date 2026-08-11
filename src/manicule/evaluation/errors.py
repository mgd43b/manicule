"""What the harness refuses to do, and why each refusal is a raise rather than a warning.

Every error here guards a way of producing a number that reads as a measurement and is not
one. A warning would leave the number in the report, and the number is the thing people
quote — so each of these stops the report existing instead.
"""

from __future__ import annotations

from manicule.core.errors import ManiculeError


class EvaluationError(ManiculeError):
    """Base for everything this package refuses."""


class QuerySetError(EvaluationError):
    """A query set could not be read as one.

    Raised rather than skipping the offending entries, because a loader that drops what it does
    not understand turns "we ran 150 queries" into "we ran the 118 that parsed" with nothing
    saying so.
    """


class PreferenceRecordError(EvaluationError):
    """A stored judgement could not be read back.

    Refused rather than skipped, for the reason :class:`QuerySetError` gives: a reader that
    drops the lines it does not understand reports a rate computed over an unknown subset, and
    the subset it dropped is the one written by whichever version of the format it cannot read.
    """


class CorpusMismatchError(EvaluationError):
    """Two sides reported different corpora.

    A preference measured across different content is a measurement of the content. Same
    documents on both sides is the precondition that makes the difference attributable to
    retrieval at all.
    """


class ProbeUnusableError(EvaluationError):
    """The discrimination probe cannot say anything about this system.

    Not the same as failing the probe. This is the probe declining to produce a verdict it
    could not have justified — too few items, a corpus too small for ``k`` to mean anything, or
    a system that cannot say how many documents it is choosing between.
    """


class UnderpoweredProbeError(ProbeUnusableError):
    """The probe has too few items to distinguish a perfect system from a useless one.

    The specific failure this whole package exists to prevent, inverted: a check that reports
    "indistinguishable from chance" no matter what it is handed is exactly as uninformative as
    one that reports "fine" no matter what it is handed.
    """


class AtChanceError(EvaluationError):
    """A system under comparison retrieves no better than chance.

    Its preferences would be a measurement of noise. This is the load-bearing refusal: a
    retrieval evaluation whose components cannot be shown to retrieve is not evidence about
    retrieval, and every conclusion drawn from it is unfalsifiable.
    """


class ConfigurationDriftError(EvaluationError):
    """A side's configuration changed while it was being measured.

    The record names the configuration that produced each result. If the configuration moved
    mid-run, half the records name something that was not running when they were made, and
    nothing in the file would say which half.
    """


class IncomparableRecordsError(EvaluationError):
    """Records that may not be summarised together were handed to the reporter.

    Different systems, different corpora or different query-set provenance. Averaging across
    any of them produces a figure nobody can attribute to anything, which is the failure mode
    the whole design of the trace was built to make impossible.
    """


__all__ = [
    "AtChanceError",
    "ConfigurationDriftError",
    "CorpusMismatchError",
    "EvaluationError",
    "IncomparableRecordsError",
    "PreferenceRecordError",
    "ProbeUnusableError",
    "QuerySetError",
    "UnderpoweredProbeError",
]
