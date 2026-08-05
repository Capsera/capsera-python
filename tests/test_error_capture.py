"""
Tests for SDK error capture — failed provider calls must emit a zero-cost
TokenEvent with is_error=True and re-raise the original exception.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter


class FakeProviderError(Exception):
    """Stand-in for a provider SDK exception (e.g. rate limit / auth error)."""


@pytest.fixture
def config() -> Config:
    # Enforcement off so the test does not attempt a budget-check round-trip.
    return Config(
        api_key="test",
        endpoint="http://localhost:9999",
        env="test",
        flush_interval_ms=99_999,
        enable_budget_enforcement=False,
        debug=True,
    )


@pytest.fixture
def emitter(config: Config) -> CapturingEmitter:
    e = CapturingEmitter(config)
    e.start()
    yield e
    e.shutdown()


def _fake_anthropic_raises(exc: Exception) -> ModuleType:
    fake = ModuleType("anthropic")
    fake.resources = ModuleType("anthropic.resources")
    fake.resources.messages = ModuleType("anthropic.resources.messages")

    class FakeMessages:
        _capsera_patched = False

        def create(self, **kwargs: Any) -> Any:
            raise exc

    class FakeAsyncMessages:
        _capsera_patched = False

        async def create(self, **kwargs: Any) -> Any:
            raise exc

    fake.resources.messages.Messages = FakeMessages
    fake.resources.messages.AsyncMessages = FakeAsyncMessages
    return fake


def _fake_openai_raises(exc: Exception) -> ModuleType:
    fake = ModuleType("openai")
    fake.resources = ModuleType("openai.resources")
    fake.resources.chat = ModuleType("openai.resources.chat")
    fake.resources.chat.completions = ModuleType("openai.resources.chat.completions")

    class FakeCompletions:
        _capsera_patched = False

        def create(self, **kwargs: Any) -> Any:
            raise exc

    class FakeAsyncCompletions:
        _capsera_patched = False

        async def create(self, **kwargs: Any) -> Any:
            raise exc

    fake.resources.chat.completions.Completions = FakeCompletions
    fake.resources.chat.completions.AsyncCompletions = FakeAsyncCompletions
    return fake


def test_anthropic_sync_error_emits_error_event(config: Config, emitter: CapturingEmitter) -> None:
    fake = _fake_anthropic_raises(FakeProviderError("boom"))
    interceptor = Interceptor(config, emitter)
    interceptor._patch_anthropic_sync(fake)

    client = fake.resources.messages.Messages()
    with pytest.raises(FakeProviderError):
        client.create(model="claude-sonnet-4-6", max_tokens=10, messages=[])

    events = emitter.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev.is_error is True
    assert ev.error_type == "FakeProviderError"
    assert ev.cost_usd == 0.0
    assert ev.input_tokens == 0
    assert ev.output_tokens == 0
    assert ev.model == "claude-sonnet-4-6"
    assert ev.provider == "anthropic"


def test_openai_sync_error_emits_error_event(config: Config, emitter: CapturingEmitter) -> None:
    fake = _fake_openai_raises(FakeProviderError("rate limited"))
    interceptor = Interceptor(config, emitter)
    interceptor._patch_openai_sync(fake)

    client = fake.resources.chat.completions.Completions()
    with pytest.raises(FakeProviderError):
        client.create(model="gpt-4o", messages=[])

    events = emitter.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev.is_error is True
    assert ev.error_type == "FakeProviderError"
    assert ev.cost_usd == 0.0
    assert ev.provider == "openai"


@pytest.mark.asyncio
async def test_anthropic_async_error_emits_error_event(config: Config, emitter: CapturingEmitter) -> None:
    fake = _fake_anthropic_raises(FakeProviderError("async boom"))
    interceptor = Interceptor(config, emitter)
    interceptor._patch_anthropic_async(fake)

    client = fake.resources.messages.AsyncMessages()
    with pytest.raises(FakeProviderError):
        await client.create(model="claude-sonnet-4-6", max_tokens=10, messages=[])

    events = emitter.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev.is_error is True
    assert ev.error_type == "FakeProviderError"
    assert ev.cost_usd == 0.0
