"""Generation and chat: streamed answers whose citations are verified before they are shown.

The whole design in five sentences. **A model never writes a citation, it selects one. Every
citation is verified against the retained source bytes before it reaches a reader, and an
unverified one is deleted rather than shown. Deleting a marker is the only edit anything
downstream of the model may make to the answer. Redaction is a projection applied on egress
and never touches the artefact a citation is verified against. Confidence and citation
accounting are two different numbers and are never combined into one.**

The third is the one that is easy to lose. Once you accept that some citation must sometimes
be removed, every convenient repair becomes available — trimming a sentence, rewriting a
clause, re-generating the tail — and each of them changes what the user was told for a reason
the user cannot see.

**Importing this package loads no provider library.** The litellm import lives inside the
factory in :mod:`manicule.generation.plugin`, so an installation that never asks a question
never pays for it.
"""

from __future__ import annotations

from manicule.generation.answering import (
    Answerer,
    AnswerRequest,
    AnswerResult,
    SupportsAnswer,
    accepted_extras,
    answering,
)
from manicule.generation.answers import (
    AnswerEnvelope,
    AnswerEvent,
    Citation,
    CitationAccounting,
    CitationDrop,
    DropReason,
    EventKind,
    GenerationTrace,
    PolicyDrop,
    Verification,
)
from manicule.generation.binder import CitationBinder
from manicule.generation.budget import (
    GENERATION_ENCODING,
    TokenEstimator,
    drift_problem,
    usable_prompt_tokens,
)
from manicule.generation.config import GENERATOR_NAME, GeneratorConfig
from manicule.generation.history import HistoryPlan, Turn, fit_history, neutralise_markers
from manicule.generation.markers import (
    ATTEMPT_PREFIX,
    MARKER_MAX_LEN,
    MarkerScanner,
    ScanEvent,
    ScanEventKind,
    escape_markers,
    render_marker,
)
from manicule.generation.policy import EgressPolicy, filter_context
from manicule.generation.ports import (
    ConversationStore,
    Feedback,
    FeedbackReason,
    SharedTurn,
    ShareStore,
    StoredMessage,
)
from manicule.generation.prompt import CITATION_PROTOCOL, SYSTEM_PROMPT, ChatMessage, build_messages
from manicule.generation.redaction import BUILTIN_DETECTORS, Detector, Redactor
from manicule.generation.sharing import (
    CitationLabel,
    ShareLink,
    anonymous_location,
    anonymous_trail,
    hash_token,
    is_live,
    new_share,
    redact_for_anonymous,
    require_sharing_enabled,
    tokens_match,
)
from manicule.generation.verification import (
    AnchorResolver,
    ChainRouter,
    CitationVerifier,
    RetainedBytesResolver,
    UnverifiableSource,
    VerificationRun,
    load_documents,
)

__all__ = [
    "ATTEMPT_PREFIX",
    "BUILTIN_DETECTORS",
    "CITATION_PROTOCOL",
    "GENERATION_ENCODING",
    "GENERATOR_NAME",
    "MARKER_MAX_LEN",
    "SYSTEM_PROMPT",
    "AnchorResolver",
    "AnswerEnvelope",
    "AnswerEvent",
    "AnswerRequest",
    "AnswerResult",
    "Answerer",
    "ChainRouter",
    "ChatMessage",
    "Citation",
    "CitationAccounting",
    "CitationBinder",
    "CitationDrop",
    "CitationLabel",
    "CitationVerifier",
    "ConversationStore",
    "Detector",
    "DropReason",
    "EgressPolicy",
    "EventKind",
    "Feedback",
    "FeedbackReason",
    "GenerationTrace",
    "GeneratorConfig",
    "HistoryPlan",
    "MarkerScanner",
    "PolicyDrop",
    "Redactor",
    "RetainedBytesResolver",
    "ScanEvent",
    "ScanEventKind",
    "ShareLink",
    "ShareStore",
    "SharedTurn",
    "StoredMessage",
    "SupportsAnswer",
    "TokenEstimator",
    "Turn",
    "UnverifiableSource",
    "Verification",
    "VerificationRun",
    "accepted_extras",
    "anonymous_location",
    "anonymous_trail",
    "answering",
    "build_messages",
    "drift_problem",
    "escape_markers",
    "filter_context",
    "fit_history",
    "hash_token",
    "is_live",
    "load_documents",
    "neutralise_markers",
    "new_share",
    "redact_for_anonymous",
    "render_marker",
    "require_sharing_enabled",
    "tokens_match",
    "usable_prompt_tokens",
]
