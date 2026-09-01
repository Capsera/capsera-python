"""
Tests for the Interceptor — monkey-patch logic for Anthropic and OpenAI.

Strategy
--------
* We do NOT import the real anthropic/openai packages (they may not be
  installed in CI).  Instead we build a minimal mock module that exposes
  the same class/attribute structure that the Interceptor expects.
* We install the mock into ``sys.modules`` before the Interceptor tries to
  import it, so the ``import anthropic`` inside ``_patch_anthropic`` picks
  up our fake module.
* After patching we call the fake ``Messages.create`` and assert that:
    1. The original method was still called (response is returned).
    2. A TokenEvent was enqueued with the correct token counts / cost.
    3. The agent context from the ``@agent`` decorator is reflected.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import capsera
from capsera._config import Config
from capsera._context import AgentContext, _context_stack
from capsera._emitter import Emitter
from capsera._interceptor import Interceptor
from capsera._models import TokenEvent

from .conftest import (
    CapturingEmitter,
    make_anthropic_response,
    make_openai_response,
)

# =========================================================================== #
# Helper — build a fake "anthropic" module
# =========================================================================== #

def _make_fake_anthropic_module(response: Any) -> ModuleType:
    """
    Build a minimal fake anthropic module that mirrors the attribute paths the
    Interceptor traverses:

        anthropic.resources.messages.Messages.create
        anthropic.resources.messages.AsyncMessages.create
    """
    fake_anthropic = ModuleType("anthropic")

    # anthropic.resources
    fake_resources = ModuleType("anthropic.resources")
    fake_anthropic.resources = fake_resources

    # anthropic.resources.messages
    fake_messages_mod = ModuleType("anthropic.resources.messages")
    fake_resources.messages = fake_messages_mod

    # Messages class
    class FakeMessages:
        _capsera_patched = False  # reset between tests

        def create(self, **kwargs):
            return response

    # AsyncMessages class (mirrors sync for now)
    class FakeAsyncMessages:
        _capsera_patched = False

        async def create(self, **kwargs):
            return response

    fake_messages_mod.Messages = FakeMessages
    fake_messages_mod.AsyncMessages = FakeAsyncMessages

    return fake_anthropic


def _make_fake_openai_module(response: Any) -> ModuleType:
    """Build a minimal fake openai module."""
    fake_openai = ModuleType("openai")

    fake_resources = ModuleType("openai.resources")
    fake_openai.resources = fake_resources

    fake_chat = ModuleType("openai.resources.chat")
    fake_resources.chat = fake_chat

    fake_completions_mod = ModuleType("openai.resources.chat.completions")
    fake_chat.completions = fake_completions_mod

    class FakeCompletions:
        _capsera_patched = False

        def create(self, **kwargs):
            return response

    class FakeAsyncCompletions:
        _capsera_patched = False

        async def create(self, **kwargs):
            return response

    fake_completions_mod.Completions = FakeCompletions
    fake_completions_mod.AsyncCompletions = FakeAsyncCompletions

    return fake_openai


def make_google_response(
    model: str = "gemini-2.0-flash",
    input_tokens: int = 120,
    output_tokens: int = 45,
) -> MagicMock:
    response = MagicMock()
    response.model = model
    response.usage_metadata = MagicMock()
    response.usage_metadata.prompt_token_count = input_tokens
    response.usage_metadata.candidates_token_count = output_tokens
    return response


def _make_fake_google_generativeai_module(response: Any) -> tuple[ModuleType, ModuleType]:
    fake_google = ModuleType("google")
    fake_genai = ModuleType("google.generativeai")
    fake_google.generativeai = fake_genai

    class FakeGenerativeModel:
        def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
            self.model_name = model_name

        def generate_content(self, *args: Any, **kwargs: Any) -> Any:
            return response

        async def generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
            return response

    fake_genai.GenerativeModel = FakeGenerativeModel
    return fake_google, fake_genai


def _make_fake_google_genai_module(response: Any) -> tuple[ModuleType, ModuleType, ModuleType]:
    fake_google = ModuleType("google")
    fake_genai = ModuleType("google.genai")
    fake_models_mod = ModuleType("google.genai.models")
    fake_google.genai = fake_genai
    fake_genai.models = fake_models_mod

    # Streaming: only the FINAL chunk carries usage_metadata, mirroring the
    # real SDK's accounting.
    chunk_no_usage = MagicMock()
    chunk_no_usage.usage_metadata = None

    class FakeModels:
        last_kwargs: dict[str, Any] | None = None

        def generate_content(self, **kwargs: Any) -> Any:
            self.__class__.last_kwargs = kwargs
            return response

        def generate_content_stream(self, **kwargs: Any) -> Any:
            self.__class__.last_kwargs = kwargs
            yield chunk_no_usage
            yield response

    class FakeAsyncModels:
        last_kwargs: dict[str, Any] | None = None

        async def generate_content(self, **kwargs: Any) -> Any:
            self.__class__.last_kwargs = kwargs
            return response

        # Real google-genai convention: an `async def` that RETURNS an async
        # iterator (callers `await` it, then `async for`).
        async def generate_content_stream(self, **kwargs: Any) -> Any:
            self.__class__.last_kwargs = kwargs

            async def _iter():
                yield chunk_no_usage
                yield response

            return _iter()

    fake_models_mod.Models = FakeModels
    fake_models_mod.AsyncModels = FakeAsyncModels
    return fake_google, fake_genai, fake_models_mod


# =========================================================================== #
# Fixtures
# =========================================================================== #

@pytest.fixture
def config() -> Config:
    return Config(
        api_key="test",
        endpoint="http://localhost:9999",
        env="test",
        flush_interval_ms=99_999,
        debug=True,
    )


@pytest.fixture
def emitter(config: Config) -> CapturingEmitter:
    e = CapturingEmitter(config)
    e.start()
    yield e
    e.shutdown()


# =========================================================================== #
# Anthropic — sync create
# =========================================================================== #

class TestAnthropicSyncCreate:
    def test_basic_event_emitted(self, config: Config, emitter: CapturingEmitter) -> None:
        """After patching, calling Messages.create emits exactly one TokenEvent."""
        anthropic_response = make_anthropic_response(
            model="claude-sonnet-4-6",
            input_tokens=200,
            output_tokens=75,
        )
        fake_anthropic = _make_fake_anthropic_module(anthropic_response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            client_instance = fake_anthropic.resources.messages.Messages()
            result = client_instance.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Original return value must be preserved
        assert result is anthropic_response

        events = emitter.drain()
        assert len(events) == 1

        ev = events[0]
        assert ev.provider == "anthropic"
        assert ev.model == "claude-sonnet-4-6"
        assert ev.input_tokens == 200
        assert ev.output_tokens == 75
        assert ev.cache_read_tokens == 0
        assert ev.cache_write_tokens == 0
        assert ev.cost_usd > 0
        assert ev.env == "test"

    def test_cost_computation(self, config: Config, emitter: CapturingEmitter) -> None:
        """cost_usd is computed from pricing table, not hardcoded."""
        # claude-sonnet-4-6: $0.003 / 1k input, $0.015 / 1k output
        anthropic_response = make_anthropic_response(
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=1000,
        )
        fake_anthropic = _make_fake_anthropic_module(anthropic_response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            client = fake_anthropic.resources.messages.Messages()
            client.create(model="claude-sonnet-4-6", max_tokens=256, messages=[])

        events = emitter.drain()
        assert len(events) == 1
        expected_cost = Decimal("0.003") + Decimal("0.015")  # $0.018 total
        assert abs(Decimal(str(events[0].cost_usd)) - expected_cost) < Decimal("0.0001")

    def test_cache_tokens_captured(self, config: Config, emitter: CapturingEmitter) -> None:
        """Anthropic prompt cache tokens are extracted and priced."""
        anthropic_response = make_anthropic_response(
            model="claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=100,
            cache_read=200,
            cache_write=50,
        )
        fake_anthropic = _make_fake_anthropic_module(anthropic_response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            client = fake_anthropic.resources.messages.Messages()
            client.create(model="claude-sonnet-4-6", max_tokens=256, messages=[])

        events = emitter.drain()
        assert events[0].cache_read_tokens == 200
        assert events[0].cache_write_tokens == 50

    def test_idempotent_patching(self, config: Config, emitter: CapturingEmitter) -> None:
        """Calling _patch_anthropic_sync twice does not double-wrap the method."""
        response = make_anthropic_response()
        fake_anthropic = _make_fake_anthropic_module(response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)
            interceptor._patch_anthropic_sync(fake_anthropic)  # second call

            client = fake_anthropic.resources.messages.Messages()
            client.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])

        events = emitter.drain()
        # Only one event despite double patch attempt
        assert len(events) == 1

    def test_latency_recorded(self, config: Config, emitter: CapturingEmitter) -> None:
        """latency_ms is a non-negative integer."""
        response = make_anthropic_response()
        fake_anthropic = _make_fake_anthropic_module(response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            client = fake_anthropic.resources.messages.Messages()
            client.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])

        events = emitter.drain()
        assert events[0].latency_ms >= 0


# =========================================================================== #
# Anthropic — agent context propagation
# =========================================================================== #

class TestAnthropicAgentContext:
    def test_agent_decorator_sets_agent_id(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Events emitted inside @agent-decorated function carry the correct agent_id."""
        response = make_anthropic_response()
        fake_anthropic = _make_fake_anthropic_module(response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            @capsera.agent(name="my_test_agent", team="eng", task_type="qa")
            def run_agent():
                client = fake_anthropic.resources.messages.Messages()
                return client.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])

            run_agent()

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].agent_id == "my_test_agent"
        assert events[0].team_id == "eng"
        assert events[0].task_type == "qa"

    def test_tag_context_manager_sets_agent_id(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Events emitted inside a tag() block carry the correct agent_id."""
        response = make_anthropic_response()
        fake_anthropic = _make_fake_anthropic_module(response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            with capsera.tag(agent="tag_agent", team="ops"):
                client = fake_anthropic.resources.messages.Messages()
                client.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])

        events = emitter.drain()
        assert events[0].agent_id == "tag_agent"
        assert events[0].team_id == "ops"

    def test_no_context_uses_config_default(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """When no context is active, default_agent from Config is used."""
        config_with_default = Config(
            api_key="test",
            endpoint="http://localhost:9999",
            env="test",
            flush_interval_ms=99_999,
            default_agent="global_agent",
            default_team="global_team",
        )
        custom_emitter = CapturingEmitter(config_with_default)
        custom_emitter.start()

        response = make_anthropic_response()
        fake_anthropic = _make_fake_anthropic_module(response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config_with_default, custom_emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            client = fake_anthropic.resources.messages.Messages()
            client.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])

        events = custom_emitter.drain()
        assert events[0].agent_id == "global_agent"
        assert events[0].team_id == "global_team"

        custom_emitter.shutdown()

    def test_nested_agent_decorators(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Inner decorator wins; after it returns, outer decorator is restored."""
        response = make_anthropic_response()
        fake_anthropic = _make_fake_anthropic_module(response)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                       "anthropic.resources": fake_anthropic.resources,
                                       "anthropic.resources.messages": fake_anthropic.resources.messages}):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_anthropic_sync(fake_anthropic)

            @capsera.agent(name="outer_agent")
            def outer():
                @capsera.agent(name="inner_agent")
                def inner():
                    client = fake_anthropic.resources.messages.Messages()
                    client.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])

                inner()

            outer()

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].agent_id == "inner_agent"


# =========================================================================== #
# OpenAI — sync create
# =========================================================================== #

class TestOpenAICompatibleProviders:
    def test_groq_base_url_labels_provider_groq(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """A Groq call rides the openai SDK, so only the client's base_url can
        distinguish it from real OpenAI. The event must be labeled 'groq' and
        priced from the Groq table, not the $0.001/$0.003 unknown fallback."""
        oai_response = make_openai_response(
            model="llama-3.3-70b-versatile", prompt_tokens=1000, completion_tokens=1000
        )
        fake_openai = _make_fake_openai_module(oai_response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            client = fake_openai.resources.chat.completions.Completions()
            # Real OpenAI Completions carries self._client (the top-level
            # client) with .base_url.
            client._client = SimpleNamespace(base_url="https://api.groq.com/openai/v1")
            client.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Hi"}],
            )

        ev = emitter.drain()[0]
        assert ev.provider == "groq"
        assert ev.model == "llama-3.3-70b-versatile"
        # Groq price is 0.00059 in / 0.00079 out per 1k → not the unknown estimate.
        assert float(ev.cost_usd) == pytest.approx(0.00138)

    def test_no_base_url_still_labels_openai(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        # Real OpenAI (default base_url or none exposed) stays 'openai'.
        oai_response = make_openai_response(model="gpt-4o", prompt_tokens=10, completion_tokens=5)
        fake_openai = _make_fake_openai_module(oai_response)
        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)
            client = fake_openai.resources.chat.completions.Completions()
            client.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        assert emitter.drain()[0].provider == "openai"


class TestOpenAISyncCreate:
    def test_basic_event_emitted(self, config: Config, emitter: CapturingEmitter) -> None:
        """After patching, calling Completions.create emits exactly one TokenEvent."""
        oai_response = make_openai_response(
            model="gpt-4o",
            prompt_tokens=150,
            completion_tokens=60,
        )
        fake_openai = _make_fake_openai_module(oai_response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            client = fake_openai.resources.chat.completions.Completions()
            result = client.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result is oai_response

        events = emitter.drain()
        assert len(events) == 1

        ev = events[0]
        assert ev.provider == "openai"
        assert ev.model == "gpt-4o"
        assert ev.input_tokens == 150
        assert ev.output_tokens == 60
        assert ev.cost_usd > 0

    def test_cost_computation_openai(self, config: Config, emitter: CapturingEmitter) -> None:
        """gpt-4o pricing: $0.0025/1k input, $0.010/1k output."""
        oai_response = make_openai_response(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=1000,
        )
        fake_openai = _make_fake_openai_module(oai_response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            client = fake_openai.resources.chat.completions.Completions()
            client.create(model="gpt-4o", messages=[])

        events = emitter.drain()
        expected = Decimal("0.0025") + Decimal("0.010")
        assert abs(Decimal(str(events[0].cost_usd)) - expected) < Decimal("0.0001")

    def test_agent_context_openai(self, config: Config, emitter: CapturingEmitter) -> None:
        """Agent context is applied to OpenAI events too."""
        oai_response = make_openai_response()
        fake_openai = _make_fake_openai_module(oai_response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            with capsera.tag(agent="oai_agent", team="product"):
                client = fake_openai.resources.chat.completions.Completions()
                client.create(model="gpt-4o", messages=[])

        events = emitter.drain()
        assert events[0].agent_id == "oai_agent"
        assert events[0].team_id == "product"


# =========================================================================== #
# Google Gemini
# =========================================================================== #

class TestGoogleGemini:
    def test_google_generativeai_event_emitted(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        response = make_google_response(
            model="gemini-2.0-flash",
            input_tokens=222,
            output_tokens=77,
        )
        fake_google, fake_genai = _make_fake_google_generativeai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.generativeai": fake_genai,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_generativeai(fake_genai) is True

            client = fake_genai.GenerativeModel("gemini-2.0-flash")
            result = client.generate_content("hello")

        assert result is response
        events = emitter.drain()
        assert len(events) == 1
        ev = events[0]
        assert ev.provider == "google"
        assert ev.model == "gemini-2.0-flash"
        assert ev.input_tokens == 222
        assert ev.output_tokens == 77
        assert ev.cost_usd > 0

    def test_google_genai_event_emitted(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        response = make_google_response(
            model="gemini-2.0-flash",
            input_tokens=111,
            output_tokens=44,
        )
        fake_google, fake_genai, fake_models_mod = _make_fake_google_genai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.models": fake_models_mod,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_genai(fake_genai) is True

            models = fake_genai.models.Models()
            result = models.generate_content(model="gemini-2.0-flash", contents="hello")

        assert result is response
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].provider == "google"
        assert events[0].model == "gemini-2.0-flash"
        assert events[0].input_tokens == 111
        assert events[0].output_tokens == 44

    def test_google_genai_async_event_emitted(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """`await client.aio.models.generate_content(...)` — the surface an
        async app (e.g. an AutoGen ChatCompletionClient adapter) actually
        calls. Patching only the sync Models class left it invisible."""
        response = make_google_response(
            model="gemini-2.0-flash", input_tokens=88, output_tokens=33
        )
        fake_google, fake_genai, fake_models_mod = _make_fake_google_genai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.models": fake_models_mod,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_genai(fake_genai) is True

            models = fake_genai.models.AsyncModels()
            result = asyncio.run(
                models.generate_content(model="gemini-2.0-flash", contents="hello")
            )

        assert result is response
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].provider == "google"
        assert events[0].input_tokens == 88
        assert events[0].output_tokens == 33
        assert events[0].cost_usd > 0

    def test_google_genai_async_call_carries_agent_context(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """The bi-copilot shape end-to-end: an AutoGen model client wrapped in
        capsera.autogen_agent() makes an async Gemini call — the emitted event
        must be attributed to the wrapped agent, not 'unknown'."""
        import capsera as capsera_pkg

        response = make_google_response(input_tokens=10, output_tokens=5)
        fake_google, fake_genai, fake_models_mod = _make_fake_google_genai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.models": fake_models_mod,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_genai(fake_genai) is True

            class GeminiAdapter:
                """Duck-typed stand-in for an AutoGen ChatCompletionClient."""

                def __init__(self) -> None:
                    self._models = fake_genai.models.AsyncModels()

                async def create(self, messages: Any) -> Any:
                    return await self._models.generate_content(
                        model="gemini-2.0-flash", contents=messages
                    )

            wrapped = capsera_pkg.autogen_agent(GeminiAdapter(), name="critic", team="bi")
            asyncio.run(wrapped.create([{"role": "user", "content": "review"}]))

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].agent_id == "critic"
        assert events[0].team_id == "bi"
        assert events[0].provider == "google"

    def test_google_genai_sync_stream_emits_final_usage(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        response = make_google_response(input_tokens=60, output_tokens=21)
        fake_google, fake_genai, fake_models_mod = _make_fake_google_genai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.models": fake_models_mod,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_genai(fake_genai) is True

            models = fake_genai.models.Models()
            chunks = list(
                models.generate_content_stream(model="gemini-2.0-flash", contents="hi")
            )

        assert chunks[-1] is response
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 60
        assert events[0].output_tokens == 21

    def test_google_genai_async_stream_emits_final_usage(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        response = make_google_response(input_tokens=70, output_tokens=28)
        fake_google, fake_genai, fake_models_mod = _make_fake_google_genai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.models": fake_models_mod,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_genai(fake_genai) is True

            models = fake_genai.models.AsyncModels()

            async def consume() -> list:
                stream = await models.generate_content_stream(
                    model="gemini-2.0-flash", contents="hi"
                )
                return [chunk async for chunk in stream]

            chunks = asyncio.run(consume())

        assert chunks[-1] is response
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 70
        assert events[0].output_tokens == 28

    def test_google_genai_alias_resolves_to_model_version(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Rolling aliases must be billed as the concrete model that served them.

        Real google.genai responses carry no ``.model`` attribute — only
        ``model_version`` (e.g. requesting ``gemini-flash-lite-latest``
        resolves to ``gemini-3.5-flash-lite``). Verified against the live
        API on 2026-08-03.
        """
        response = make_google_response(input_tokens=3, output_tokens=1)
        response.model = None
        response.model_version = "gemini-3.5-flash-lite"
        fake_google, fake_genai, fake_models_mod = _make_fake_google_genai_module(response)

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.models": fake_models_mod,
        }):
            interceptor = Interceptor(config, emitter)
            assert interceptor._patch_google_genai(fake_genai) is True

            models = fake_genai.models.Models()
            models.generate_content(model="gemini-flash-lite-latest", contents="hello")

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].model == "gemini-3.5-flash-lite"
        assert events[0].cost_usd > 0


# =========================================================================== #
# Full init() integration (no real HTTP calls)
# =========================================================================== #

class TestInit:
    def test_init_and_record(self) -> None:
        """
        capsera.init() + record() should enqueue an event.
        We capture it by monkey-patching Emitter._send.
        """
        captured: list[TokenEvent] = []

        def capture_send(self, events):  # type: ignore[override]
            captured.extend(events)

        with patch.object(Emitter, "_send", capture_send):
            capsera.init(
                api_key="test",
                endpoint="http://localhost:9999",
                env="test",
                flush_interval_ms=9999,
            )
            capsera.record(
                input_tokens=300,
                output_tokens=100,
                model="claude-sonnet-4-6",
                provider="anthropic",
                agent_id="record_agent",
            )
            capsera.flush()

        assert len(captured) == 1
        ev = captured[0]
        assert ev.input_tokens == 300
        assert ev.output_tokens == 100
        assert ev.agent_id == "record_agent"
        assert ev.model == "claude-sonnet-4-6"

    def test_record_before_init_does_not_raise(self) -> None:
        """Calling record() before init() logs a warning but never raises."""
        import capsera as tb
        original_emitter = tb._emitter
        tb._emitter = None
        try:
            tb.record(
                input_tokens=1,
                output_tokens=1,
                model="gpt-4o",
                provider="openai",
            )
        finally:
            tb._emitter = original_emitter


# =========================================================================== #
# Pricing unit tests
# =========================================================================== #

class TestPricing:
    def test_known_model_cost(self) -> None:
        from capsera._pricing import compute_cost
        cost = compute_cost(
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=10_000,
            output_tokens=5_000,
        )
        # $0.00015/1k * 10 + $0.0006/1k * 5 = $0.0015 + $0.0030 = $0.0045
        assert abs(cost - Decimal("0.0045")) < Decimal("0.0001")

    def test_unknown_model_uses_fallback(self) -> None:
        from capsera._pricing import compute_cost
        cost = compute_cost(
            model="some-future-model-xyz",
            provider="unknown",
            input_tokens=1000,
            output_tokens=1000,
        )
        # Fallback: $0.001 + $0.003 = $0.004
        assert cost == Decimal("0.004")

    def test_cache_tokens_priced(self) -> None:
        from capsera._pricing import compute_cost
        cost_with_cache = compute_cost(
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=0,
            cache_read_tokens=1000,
            cache_write_tokens=0,
        )
        cost_without_cache = compute_cost(
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=0,
        )
        # Cache read is 10% of input price so cost_with > cost_without
        assert cost_with_cache > cost_without_cache

    def test_haiku_cheaper_than_sonnet(self) -> None:
        from capsera._pricing import compute_cost
        tokens = dict(input_tokens=1000, output_tokens=1000)
        haiku = compute_cost(model="claude-haiku-4-5-20251001", provider="anthropic", **tokens)
        sonnet = compute_cost(model="claude-sonnet-4-6", provider="anthropic", **tokens)
        assert haiku < sonnet


# =========================================================================== #
# Context stack unit tests
# =========================================================================== #

class TestContextStack:
    def test_push_pop(self) -> None:
        assert _context_stack.current() is None or True  # may have leftover state
        _context_stack.push(AgentContext(agent="a1"))
        assert _context_stack.current().agent == "a1"
        _context_stack.pop()

    def test_nesting(self) -> None:
        _context_stack.push(AgentContext(agent="outer"))
        _context_stack.push(AgentContext(agent="inner"))
        assert _context_stack.current().agent == "inner"
        _context_stack.pop()
        assert _context_stack.current().agent == "outer"
        _context_stack.pop()

    def test_thread_isolation(self) -> None:
        results: dict[str, Any] = {}

        def thread_fn(name: str) -> None:
            _context_stack.push(AgentContext(agent=name))
            time.sleep(0.05)  # overlap with other thread
            results[name] = _context_stack.current().agent
            _context_stack.pop()

        t1 = threading.Thread(target=thread_fn, args=("thread_1",))
        t2 = threading.Thread(target=thread_fn, args=("thread_2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["thread_1"] == "thread_1"
        assert results["thread_2"] == "thread_2"


# =========================================================================== #
# TokenEvent serialisation
# =========================================================================== #

class TestTokenEventSerialization:
    def test_to_dict_excludes_none(self) -> None:
        ev = TokenEvent(
            id="123",
            timestamp="2026-01-01T00:00:00+00:00",
            request_id="456",
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_tokens=100,
            output_tokens=50,
            agent_id="agent",
            env="test",
        )
        d = ev.to_dict()
        assert "team_id" not in d
        assert "user_id" not in d
        assert "task_type" not in d
        assert d["model"] == "claude-sonnet-4-6"
        assert d["input_tokens"] == 100

    def test_to_dict_includes_set_optionals(self) -> None:
        ev = TokenEvent(
            id="123",
            timestamp="2026-01-01T00:00:00+00:00",
            request_id="456",
            model="gpt-4o",
            provider="openai",
            input_tokens=10,
            output_tokens=5,
            agent_id="a",
            env="prod",
            team_id="team-x",
            task_type="summarisation",
        )
        d = ev.to_dict()
        assert d["team_id"] == "team-x"
        assert d["task_type"] == "summarisation"


class TestCallerLocationPathIndependence:
    """Caller resolution must not depend on what the customer named their
    directories.

    The SDK's own frames used to be matched by the path fragment "capsera",
    which also matched any customer path *containing* that string — a company
    of that name, a directory called `capsera-integration`, a venv under
    `/opt/capsera/`. Every frame of their code was skipped and `caller_file`
    came back "unknown", silently, with attribution simply absent. It
    reproduced in this repository, whose checkout is named `capsera`, which is
    how the eval pipeline surfaced it (FID-5 at 0%).
    """

    def test_sdk_frames_are_matched_by_module_not_path(self) -> None:
        import importlib

        module = importlib.import_module("capsera._interceptor")
        module._skip_fragments_initialized = False
        fragments = module._detect_frameworks()
        module._skip_fragments_initialized = False

        assert "capsera" not in fragments, (
            "a bare 'capsera' path fragment matches customer directories too"
        )

    def test_caller_resolves_from_a_path_containing_the_sdk_name(self, tmp_path) -> None:
        """Run user code from a directory called `capsera` and demand attribution."""
        import importlib
        import subprocess
        import sys
        import textwrap

        module = importlib.import_module("capsera._interceptor")
        sdk_root = str(Path(module.__file__).resolve().parent.parent)

        app_dir = tmp_path / "capsera" / "customer-project"
        app_dir.mkdir(parents=True)
        app = app_dir / "app.py"
        app.write_text(textwrap.dedent("""
            import sys
            sys.path.insert(0, sys.argv[1])
            from capsera._interceptor import _caller_location
            print(_caller_location())
        """))

        proc = subprocess.run(
            [sys.executable, str(app), sdk_root],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        resolved = proc.stdout.strip()
        assert resolved.startswith("app.py:"), (
            f"expected the customer's own file, got {resolved!r} "
            "(a path-fragment match would return 'unknown')"
        )
