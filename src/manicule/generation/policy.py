"""Egress policy: what may leave, decided by the endpoint rather than by a name.

The question "is this text leaving the machine?" has an obvious wrong answer, and manicule
once gave it. Classifying by provider name meant ``provider = "ollama"`` with
``base_url = "http://gpu-box.lan:11434"`` and ``cloud_allowed = false`` started cleanly while
every prompt and every retrieved passage crossed the network — the policy reporting itself
satisfied by the configuration it exists to forbid. It erred the other way too: an
OpenAI-compatible endpoint on ``127.0.0.1`` was classified cloud, so the safe configuration
was the one that failed.

So nothing here re-derives locality. :func:`~manicule.config.providers.egress_for` and
:attr:`~manicule.config.settings.Settings.selected_endpoints` decide it from the resolved
endpoint, and this module consumes that decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from manicule.config.providers import Egress, Endpoint, ModelRole
from manicule.config.settings import RedactionScope, Settings
from manicule.core.content import Document
from manicule.core.retrieval import Candidate, Context
from manicule.generation.answers import PolicyDrop
from manicule.generation.budget import TokenEstimator


@dataclass(frozen=True, slots=True)
class EgressPolicy:
    """What this configuration permits for one generation endpoint.

    Built once at startup from :class:`~manicule.config.settings.Settings`, so a query never
    re-reads configuration and two answers under one process cannot disagree about policy.
    """

    endpoint: Endpoint
    local_only_sources: frozenset[str]
    cloud_allowed_sources: frozenset[str]
    workspace_cloud_allowed: bool
    redaction_scope: RedactionScope
    redaction_enabled: bool

    @classmethod
    def of(cls, settings: Settings, workspace: str | None = None) -> EgressPolicy:
        """Resolve the policy for the configured generator and one workspace."""
        endpoint = next(
            point for point in settings.selected_endpoints if point.role is ModelRole.LLM
        )
        policy = settings.security.data_policy
        overrides = {
            name.strip().lower(): value for name, value in policy.workspace_overrides.items()
        }
        override = overrides.get((workspace or settings.workspace).strip().lower())
        workspace_cloud = (
            policy.cloud_allowed
            if override is None
            else (
                policy.cloud_allowed if override.cloud_allowed is None else override.cloud_allowed
            )
        )
        return cls(
            endpoint=endpoint,
            local_only_sources=frozenset(policy.source_restrictions.local_only),
            cloud_allowed_sources=frozenset(policy.source_restrictions.cloud_allowed),
            workspace_cloud_allowed=workspace_cloud,
            redaction_scope=policy.auto_redact.scope,
            redaction_enabled=policy.auto_redact.enabled,
        )

    @property
    def egress(self) -> Egress:
        return self.endpoint.egress

    @property
    def leaves_machine(self) -> bool:
        return self.endpoint.leaves_machine

    @property
    def should_redact(self) -> bool:
        """Whether redaction applies to this request.

        ``remote`` is the point of the feature: what leaves the machine is redacted, what
        stays does not, so a fully local install pays nothing. ``always`` exists for the
        proxy case classification cannot see, and for operators who want the model's input
        uniform regardless of where it runs.
        """
        if not self.redaction_enabled:
            return False
        return self.redaction_scope is RedactionScope.ALWAYS or self.leaves_machine

    def refuses(self, source: str) -> str:
        """Why ``source`` may not be sent to this endpoint, or ``""``.

        Two rules, and the order between them is the whole point. **A source restriction is a
        floor, not a default**: ``local_only`` is not released by a workspace override,
        because the broader rule is the one somebody sets for convenience. The narrower rule
        wins.
        """
        if not self.leaves_machine:
            return ""
        if source in self.local_only_sources:
            return (
                f"source {source!r} is listed in "
                f"security.data_policy.source_restrictions.local_only, and the "
                f"{self.endpoint.describe()} is not on this machine"
            )
        if not self.workspace_cloud_allowed and source not in self.cloud_allowed_sources:
            return (
                f"cloud processing is disabled for this workspace and source {source!r} is "
                f"not exempted, while the {self.endpoint.describe()} is not on this machine"
            )
        return ""


def filter_context(
    context: Context,
    documents: Mapping[str, Document],
    policy: EgressPolicy,
    *,
    counter: TokenEstimator,
) -> tuple[Context, tuple[PolicyDrop, ...]]:
    """Remove passages policy forbids sending, and say which.

    **Only removes.** It never reorders, never adds, and never re-runs assembly. Re-assembling
    to backfill the freed budget would make the context a function of which model you asked,
    and two runs that saw different passages are not comparable.

    A passage whose document is missing from ``documents`` is kept: it is retrieval's output,
    and dropping it here would be a policy decision made on the strength of a missing row.
    Its citations are dropped later, by verification, which is where "we cannot name this
    document" belongs.

    Dropping rather than refusing the query is proportionality: refusing makes the mere
    *existence* of one restricted document break unrelated questions that happened to
    retrieve it at rank 7. Search still shows the document, because search is local and only
    generation crosses the boundary — so a user learns the document exists and that its
    content did not leave. They already had read access; nothing is disclosed that was not.

    Args:
        context: The assembled context, as retrieval produced it.
        documents: The documents its passages point into, by id.
        policy: What this configuration permits for the generation endpoint.
        counter: Measures a surviving passage in the units
            :attr:`~manicule.core.retrieval.Context.token_count` is already in — so it must be
            configured from ``rag.context``, the settings
            :class:`~manicule.retrieval.tokens.ContextTokenCounter` is built from, and not from
            ``llm.token_safety_factor``. Required rather than defaulted, because a default here
            would be this module guessing at another one's tokenizer.
    """
    kept: list[Candidate] = []
    drops: list[PolicyDrop] = []
    for candidate in context.passages:
        document = documents.get(candidate.chunk.document_id)
        reason = _refusal(candidate.chunk.document_id, document, policy)
        if reason:
            drops.append(
                PolicyDrop(
                    document_id=candidate.chunk.document_id,
                    chunk_id=candidate.chunk.id,
                    source=document.source if document else "(unknown)",
                    reason=reason,
                )
            )
        else:
            kept.append(candidate)
    if not drops:
        return context, ()
    # `token_count` is recomputed rather than carried over: `model_copy` does not re-validate,
    # so a stale total would over-report the prompt in the trace and in any budget check
    # downstream of it.
    #
    # **In the generator's tokenizer, because that is the one the number is already in.** This
    # summed `Chunk.token_count` until it was noticed that doing so silently changed units:
    # that field is the *embedder's* SentencePiece count, for a model that is not generating
    # anything, and assembly measured this total with tiktoken inflated by
    # `rag.context.safety_factor`. A filtered context therefore reported a figure comparable
    # with neither the one assembly produced nor the `token_budget` sitting beside it in
    # `Context.metadata` — wrong by an unknown factor that varies with language and content
    # type. Nothing reads this today, which is exactly why it was worth fixing before
    # something does: the failure is a plausible number, not a missing one.
    kept_tokens = sum(counter.count_chunk(candidate.chunk) for candidate in kept)
    return (
        context.model_copy(update={"passages": tuple(kept), "token_count": kept_tokens}),
        tuple(drops),
    )


def _refusal(document_id: str, document: Document | None, policy: EgressPolicy) -> str:
    """Why this passage may not be sent, or ``""``.

    **A document that cannot be found fails closed when anything is leaving.** The miss is
    realistic rather than theoretical: the document store filters soft-deleted rows while the
    chunk index still returns their chunks, so a document deleted between retrieval and
    generation — including one deleted *precisely because* somebody decided it was sensitive
    — arrives here as an absent row. Keeping it would send its full text to a hosted model
    with no policy evaluated at all, which is the irreversible direction.

    When nothing leaves the machine there is nothing to fail closed about, so the passage is
    kept and the proportionality argument for dropping-rather-than-refusing stands.
    """
    if not policy.leaves_machine:
        return ""
    if document is None:
        return (
            f"document {document_id} is not in the index, so the source policy that governs "
            f"it cannot be evaluated, and the {policy.endpoint.describe()} is not on this "
            f"machine"
        )
    return policy.refuses(document.source)


__all__ = ["EgressPolicy", "filter_context"]
