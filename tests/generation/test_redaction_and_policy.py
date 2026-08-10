"""What may leave this machine, and what is rewritten on its way out.

Egress is decided by the resolved **endpoint**, never by a provider's name. That predicate
lives in ``manicule.config.providers``; these suites check that generation consumes it rather
than re-deriving it — which is the bug that let an ``ollama`` on another host satisfy a
local-only policy while every prompt crossed the network.
"""

from __future__ import annotations

import pytest

from manicule.config.providers import Egress
from manicule.config.settings import RedactionMethod, RedactionScope, RedactionSettings
from manicule.core.content import Document
from manicule.core.errors import ConfigError, RedactionError
from manicule.core.retrieval import Context
from manicule.generation.policy import EgressPolicy, filter_context
from manicule.generation.redaction import BUILTIN_DETECTORS, Redactor
from manicule.testing import assert_local_only_policy_is_enforced
from tests.generation.fakes import candidate, context, document, settings

EMAIL = "someone@example.invalid"


def redaction(**kwargs: object) -> RedactionSettings:
    base: dict[str, object] = {"enabled": True, "patterns": ("email",)}
    base.update(kwargs)
    return RedactionSettings(**base)  # pyright: ignore[reportArgumentType] - keyword plumbing


# --- detectors ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("detector", "sample"),
    [
        ("email", f"write to {EMAIL} today"),
        ("phone", "call +44 20 7946 0958 now"),
        ("credit-card", "card 4111 1111 1111 1111 expires"),
        ("ip-address", "host 192.168.1.44 is down"),
    ],
)
def test_each_built_in_detector_removes_what_it_names(detector: str, sample: str) -> None:
    result = Redactor(redaction(patterns=(detector,))).redact(sample)

    assert "[REDACTED]" in result.text
    assert result.counts == {detector: 1}


def test_a_detector_reports_how_often_it_fired_and_never_what_it_matched() -> None:
    """Recording the match turns a diagnostic into the leak the detector existed to prevent."""
    result = Redactor(redaction()).redact(f"{EMAIL} and other@example.invalid")

    assert result.counts == {"email": 2}
    assert EMAIL not in str(result.counts)


def test_hashing_preserves_co_reference_so_two_mentions_are_still_one_person() -> None:
    redactor = Redactor(
        redaction(method=RedactionMethod.HASH, hash_salt="a-per-installation-secret")
    )

    result = redactor.redact(f"{EMAIL} wrote, and {EMAIL} replied, unlike other@example.invalid")

    tokens = {part for part in result.text.split() if part.startswith("[REDACTED]:")}
    assert len(tokens) == 2, "the same address must be the same token; a different one must not"


def test_hashing_without_a_per_installation_salt_is_refused() -> None:
    """An unsalted digest of an email address is reversible with a word list, so it would
    send the value in a costume rather than protect it."""
    with pytest.raises(ConfigError, match="hash_salt"):
        Redactor(redaction(method=RedactionMethod.HASH))


def test_an_unknown_detector_name_is_refused_with_the_available_ones_listed() -> None:
    with pytest.raises(ConfigError, match="not a built-in detector"):
        Redactor(redaction(patterns=("passport-number",)))


def test_a_custom_pattern_that_does_not_compile_is_a_refusal_not_a_silent_drop() -> None:
    """A dropped pattern makes redaction quietly weaker than the configuration says it is."""
    with pytest.raises(ConfigError, match="not a valid regular expression"):
        Redactor(redaction(custom_patterns=("(unclosed",)))

    problems = settings(
        security={
            "data_policy": {"auto_redact": {"enabled": True, "custom_patterns": ["(unclosed"]}}
        }
    ).policy_problems()
    assert any("not a valid regular expression" in problem for problem in problems), (
        "the same refusal has to fire at startup, before anything is constructed"
    )


async def test_exceeding_the_deadline_fails_the_query_rather_than_sending_plaintext() -> None:
    """The fail-safe direction is refuse-to-send. There is no path where a timeout results in
    unredacted text reaching a remote model."""
    redactor = Redactor(redaction(custom_patterns=(r"(?:a+)+b",), patterns=(), timeout_s=0.01))

    with pytest.raises(RedactionError, match="nothing was sent"):
        # Bounded on purpose: enough backtracking to blow a 10ms deadline, few enough that
        # the abandoned thread finishes in about a second rather than outliving the suite.
        await redactor.redact_all(["a" * 24 + "c"])


async def test_redaction_is_a_no_op_when_it_is_switched_off() -> None:
    redactor = Redactor(redaction(enabled=False))

    texts, counts = await redactor.redact_all([f"contact {EMAIL}"])

    assert texts == [f"contact {EMAIL}"]
    assert counts == {}


# --- egress ------------------------------------------------------------------------------


def test_redaction_scope_remote_pays_nothing_on_a_loopback_endpoint() -> None:
    """The whole point of the feature: what leaves is redacted, what stays is not."""
    policy = EgressPolicy.of(
        settings(
            security={"data_policy": {"auto_redact": {"enabled": True, "patterns": ["email"]}}}
        )
    )

    assert policy.egress is Egress.LOOPBACK
    assert policy.should_redact is False


def test_redaction_scope_always_covers_the_loopback_proxy_this_cannot_see_through() -> None:
    """A proxy on 127.0.0.1 forwarding to a hosted provider classifies as loopback and
    nothing in this process can tell. An operator who knows about it can force redaction on.
    """
    policy = EgressPolicy.of(
        settings(
            security={
                "data_policy": {
                    "auto_redact": {
                        "enabled": True,
                        "patterns": ["email"],
                        "scope": RedactionScope.ALWAYS,
                    }
                }
            }
        )
    )

    assert policy.should_redact is True


def test_a_lan_endpoint_is_remote_however_the_provider_is_spelled() -> None:
    """``ollama`` at ``gpu-box.lan`` is another machine on a network, and "the office GPU
    box" is exactly the case where an operator believes otherwise."""
    policy = EgressPolicy.of(
        settings(
            llm={
                "provider": "ollama",
                "model": "qwen2.5:14b",
                "base_url": "http://gpu-box.lan:11434",
            },
            security={"data_policy": {"auto_redact": {"enabled": True, "patterns": ["email"]}}},
        )
    )

    assert policy.egress is Egress.REMOTE
    assert policy.should_redact is True


def test_a_local_only_policy_is_enforced_in_both_directions() -> None:
    assert_local_only_policy_is_enforced(
        settings(
            llm={"provider": "ollama", "model": "m", "base_url": "http://gpu-box.lan:11434"},
            security={"data_policy": {"cloud_allowed": False}},
        )
    )
    assert_local_only_policy_is_enforced(
        settings(
            llm={"provider": "openai", "model": "m", "base_url": "http://127.0.0.1:8080"},
            providers={"openai": {"api_key": "k"}},
            security={"data_policy": {"cloud_allowed": False}},
        )
    )


def remote_policy(**data_policy: object) -> EgressPolicy:
    return EgressPolicy.of(
        settings(
            llm={"provider": "openai", "model": "gpt-4o-mini"},
            providers={"openai": {"api_key": "k"}},
            security={"data_policy": data_policy},
        )
    )


def two_passages_from(sources: tuple[str, str]) -> tuple[Context, dict[str, Document]]:
    passages = (
        candidate(chunk_id="c1", document_id="doc-1"),
        candidate(chunk_id="c2", document_id="doc-2"),
    )
    documents = {
        "doc-1": document(document_id="doc-1", source=sources[0]),
        "doc-2": document(document_id="doc-2", source=sources[1]),
    }
    return context(passages), documents


def test_a_local_only_source_drops_its_passage_and_never_the_query() -> None:
    """Refusing the whole query makes the mere existence of one restricted document break
    unrelated questions that happened to retrieve it at rank 7."""
    assembled, documents = two_passages_from(("secrets", "confluence"))

    filtered, drops = filter_context(
        assembled, documents, remote_policy(source_restrictions={"local_only": ["secrets"]})
    )

    assert len(filtered.passages) == 1
    assert filtered.passages[0].chunk.id == "c2"
    assert [drop.source for drop in drops] == ["secrets"]
    assert "local_only" in drops[0].reason


def test_a_workspace_override_cannot_release_a_local_only_source() -> None:
    """A source restriction is a floor, not a default: the narrower rule wins, because the
    broader one is the one somebody sets for convenience."""
    assembled, documents = two_passages_from(("secrets", "secrets"))

    _, drops = filter_context(
        assembled,
        documents,
        remote_policy(
            source_restrictions={"local_only": ["secrets"]},
            workspace_overrides={"default": {"cloud_allowed": True}},
        ),
    )

    assert len(drops) == 2


def test_a_workspace_that_forbids_cloud_drops_all_but_the_exempted_sources() -> None:
    assembled, documents = two_passages_from(("secrets", "public"))

    filtered, drops = filter_context(
        assembled,
        documents,
        remote_policy(
            source_restrictions={"cloud_allowed": ["public"]},
            workspace_overrides={"default": {"cloud_allowed": False}},
        ),
    )

    assert [c.chunk.id for c in filtered.passages] == ["c2"]
    assert len(drops) == 1


def test_policy_filtering_only_removes_and_never_reorders_or_backfills() -> None:
    """Re-assembling to fill the freed budget would make the context a function of which
    model you asked, and two runs that saw different passages are not comparable."""
    assembled, documents = two_passages_from(("public", "public"))

    filtered, drops = filter_context(assembled, documents, remote_policy())

    assert filtered is assembled
    assert drops == ()


def test_a_source_named_in_both_restriction_lists_is_a_startup_refusal() -> None:
    problems = settings(
        security={
            "data_policy": {
                "source_restrictions": {"local_only": ["secrets"], "cloud_allowed": ["secrets"]}
            }
        }
    ).policy_problems()

    assert any("local_only and cloud_allowed" in problem for problem in problems)


def test_every_named_detector_is_one_the_configuration_can_actually_select() -> None:
    """A detector registry and a configuration that disagree is a policy nobody is running."""
    for name in BUILTIN_DETECTORS:
        assert Redactor(redaction(patterns=(name,))).detectors
