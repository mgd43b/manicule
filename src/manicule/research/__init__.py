"""Multi-step research: several retrievals for one question, then one cited report.

The design in five sentences, from ``docs/research.md``:

**A sub-question is a Query and nothing else. Every cycle's retrieval is an ordinary pipeline
run over it. Nothing a model writes between cycles is ever evidence — it only decides what to
search next. The report is an ordinary answer over one assembled context, so the citation
guarantee is inherited whole rather than re-implemented. Every bound is declared before the
run rather than discovered during it.**

The third is the one that is easy to lose. A loop that summarizes each cycle and hands the
summaries to the report has replaced the corpus with the model's notes about the corpus: the
report then cites passages it never read, through a paraphrase nobody verified. So notes exist,
they are recorded, and they reach exactly one place — the prompt that picks the next
sub-question. The report reads passages.

Nothing here imports a provider library, a database or a tokenizer at module scope;
``tests/test_import_boundary.py`` fails the build if that stops being true.
"""

from __future__ import annotations

from manicule.research.ledger import EvidenceLedger
from manicule.research.loop import ResearchLoop, plan_problem
from manicule.research.models import (
    Evidence,
    ResearchPlan,
    ResearchStep,
    ResearchTrace,
    SubQuestion,
)

__all__ = [
    "Evidence",
    "EvidenceLedger",
    "ResearchLoop",
    "ResearchPlan",
    "ResearchStep",
    "ResearchTrace",
    "SubQuestion",
    "plan_problem",
]
