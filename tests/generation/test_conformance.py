"""The generator satisfies the protocol it is loaded through, and so does the plugin.

``@runtime_checkable`` deliberately checks only that attributes exist, never their
signatures, so ``isinstance`` reports success against an implementation a caller cannot call.
"""

from __future__ import annotations

import pytest

from manicule.container import keys
from manicule.core.protocols import Generator, generating
from manicule.generation.config import GENERATOR_NAME, GeneratorConfig
from manicule.generation.plugin import PLUGIN
from manicule.generation.provider import LitellmGenerator
from manicule.plugins import ComponentRegistry
from manicule.testing import assert_protocol_signatures
from tests.generation.fakes import (
    ProtocolOnlyGenerator,
    ScriptedGenerator,
    context,
    query,
    settings,
)


@pytest.mark.contract
def test_the_built_in_generator_matches_the_protocol_signature() -> None:
    """``history`` and ``documents`` are keyword-only with defaults, which the conformance
    check permits: a caller working from the protocol never passes them."""
    generator = LitellmGenerator(settings().llm)

    assert_protocol_signatures(generator, Generator)
    assert isinstance(generator, Generator)
    assert generator.model_id == "ollama_chat/qwen2.5:14b"
    assert isinstance(generator.context_window, int)


@pytest.mark.contract
def test_a_generator_written_strictly_to_the_protocol_also_conforms() -> None:
    assert_protocol_signatures(ProtocolOnlyGenerator(), Generator)


@pytest.mark.contract
def test_the_generation_plugin_registers_a_generator_with_a_config_model() -> None:
    """A component with no declared model has its settings rejected rather than ignored."""
    registry = ComponentRegistry().bind("generation")
    PLUGIN.register(registry)

    record = registry.record(keys.GENERATOR.named(GENERATOR_NAME))

    assert record.config_model is GeneratorConfig
    assert record.summary


async def test_generating_closes_the_stream_on_every_exit_path() -> None:
    """An abandoned generation stream holds an open response to a model that is still
    working — billed tokens nobody reads, or the only local model occupied until it finishes.
    """
    generator = ScriptedGenerator(script=["one", "two", "three"])

    async with generating(generator, query(), context()) as tokens:
        assert (await anext(tokens)).text == "one"

    assert generator.closed is True


async def test_generating_closes_the_stream_when_the_consumer_raises() -> None:
    generator = ScriptedGenerator(script=["one", "two"])

    async def consume_and_fail() -> None:
        async with generating(generator, query(), context()) as tokens:
            async for _ in tokens:
                msg = "the consumer failed"
                raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="the consumer failed"):
        await consume_and_fail()

    assert generator.closed is True


async def test_generating_forwards_only_the_extras_a_generator_declares() -> None:
    generator = ScriptedGenerator(script=["hi"])

    async with generating(
        generator, query(), context(), extra={"history": [{"role": "user", "content": "earlier"}]}
    ) as tokens:
        async for _ in tokens:
            pass

    assert [message["content"] for message in generator.seen_history] == ["earlier"]
