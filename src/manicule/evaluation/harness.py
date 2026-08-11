"""Running the comparison: certify, pair, judge, record.

The order is the design. Certification comes first and it is not optional, because a
preference between two systems that cannot be distinguished from guessing is a preference
between two random orderings — and once such a record exists on disk, nothing later can tell
it apart from a real one.

Four refusals, all before or during the run rather than at reporting time:

1. **Either side at chance.** Nothing is recorded at all.
2. **Different corpora.** The comparison would measure the content, not the retrieval.
3. **A configuration that moved mid-run.** Half the records would name something that was not
   running when they were made.
4. **A corpus that moved mid-run.** Same failure, one layer out: results recorded against a
   version that is no longer what was searched.

Stages come through on every record, so per-stage attribution is a property of the data rather
than a feature. That is nearly free, and the reason is upstream: a pipeline is a declared list
of uniform stages, so two configurations that differ in one place produce records that differ
in one stage, and the report can name it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.evaluation.errors import (
    ConfigurationDriftError,
    CorpusMismatchError,
)
from manicule.evaluation.judging import JudgingStoppedError
from manicule.evaluation.preference import (
    DEFAULT_BLINDING_SEED,
    Pairing,
    PreferenceRecord,
    assign_slots,
    build_record,
)

if TYPE_CHECKING:
    from manicule.core.content import Metadata
    from manicule.evaluation.corpus import CorpusVersion
    from manicule.evaluation.judging import Judge
    from manicule.evaluation.preference import PreferenceStore
    from manicule.evaluation.probe import DiscriminationProbe, ProbeOutcome
    from manicule.evaluation.queries import QuerySet
    from manicule.evaluation.systems import SystemResult, SystemUnderComparison

DEFAULT_LIMIT = 10
"""Results asked of each side. Ten is what a judge can read; the probe uses a smaller window
for a different question."""


class PreferenceHarness:
    """Two systems, one query set, one file of judgements."""

    def __init__(
        self,
        *,
        left: SystemUnderComparison,
        right: SystemUnderComparison,
        probe: DiscriminationProbe,
        store: PreferenceStore,
        limit: int = DEFAULT_LIMIT,
        seed: str = DEFAULT_BLINDING_SEED,
    ) -> None:
        if left.config_label == right.config_label:
            msg = (
                f"both sides are labelled {left.config_label!r}. The labels are what a report "
                f"and every stored record use to say which configuration won, so two sides "
                f"sharing one produce a file in which the winner cannot be identified"
            )
            raise ValueError(msg)
        self._left = left
        self._right = right
        self._probe = probe
        self._store = store
        self._limit = limit
        self._seed = seed
        self._certified: tuple[ProbeOutcome, ProbeOutcome] | None = None

    async def certify(self) -> tuple[ProbeOutcome, ProbeOutcome]:
        """Run the discrimination probe against both sides, and check the corpora agree.

        Cached for the life of the harness: the probe costs a search per item per side, and the
        corpus is checked for movement on every recorded result anyway.

        Raises:
            AtChanceError: A side retrieves no better than guessing.
            ProbeUnusableError: The probe could not produce an honest verdict.
            CorpusMismatchError: The two sides are not searching the same content.
        """
        if self._certified is None:
            disagreement = self._left.corpus_version.disagreement_with(self._right.corpus_version)
            if disagreement is not None:
                raise CorpusMismatchError(disagreement)
            left = await self._probe.certify(self._left)
            right = await self._probe.certify(self._right)
            self._certified = (left, right)
        return self._certified

    async def compare(self, query_set: QuerySet, judge: Judge) -> tuple[PreferenceRecord, ...]:
        """Run every query through both sides and record what the judge decided.

        Records are appended as they are made, so a session stopped halfway keeps everything
        judged up to that point. Skipped queries record nothing.

        Raises:
            ConfigurationDriftError: A side's configuration or corpus changed mid-run.
        """
        left_probe, right_probe = await self.certify()
        seen_config: dict[str, Metadata] = {}
        seen_corpus: dict[str, CorpusVersion] = {}
        recorded: list[PreferenceRecord] = []

        for query in query_set.queries:
            left = await self._left.search(query.text, limit=self._limit)
            right = await self._right.search(query.text, limit=self._limit)
            for result in (left, right):
                self._require_stable(result, seen_config, seen_corpus)

            pairing = Pairing(
                query=query,
                left=left,
                right=right,
                slots=assign_slots(query.id, seed=self._seed),
                seed=self._seed,
            )
            try:
                decision = await judge.judge(pairing)
            except JudgingStoppedError:
                break
            if decision is None:
                continue
            preference, note = decision
            record = build_record(
                pairing,
                preference=preference,
                query_set=query_set.name,
                provenance=query_set.provenance,
                left_probe=left_probe,
                right_probe=right_probe,
                judge=judge.label,
                note=note,
            )
            self._store.append(record)
            recorded.append(record)
        return tuple(recorded)

    def _require_stable(
        self,
        result: SystemResult,
        seen_config: dict[str, Metadata],
        seen_corpus: dict[str, CorpusVersion],
    ) -> None:
        """Refuse a side that changed underneath the run.

        Both checks compare against the *first* result from that side rather than the previous
        one, so a configuration that flips back and forth is caught as readily as one that
        changes once.
        """
        label = result.config_label
        first_config = seen_config.setdefault(label, result.configuration)
        if first_config != result.configuration:
            msg = (
                f"{label} changed configuration mid-run: started as {first_config} and is now "
                f"{result.configuration}. Every record names the configuration that produced "
                f"it, so continuing would write a file in which some records describe a "
                f"pipeline that is no longer running and nothing says which"
            )
            raise ConfigurationDriftError(msg)
        first_corpus = seen_corpus.setdefault(label, result.corpus_version)
        moved = first_corpus.disagreement_with(result.corpus_version)
        if moved is not None:
            msg = f"{label} changed corpus mid-run: {moved}"
            raise ConfigurationDriftError(msg)


__all__ = ["DEFAULT_LIMIT", "PreferenceHarness"]
