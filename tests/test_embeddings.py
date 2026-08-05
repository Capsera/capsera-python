"""
Tests for OpenAI Embeddings interception.

Verifies that the SDK intercepts openai.resources.embeddings.Embeddings.create
and correctly tracks embedding API calls with proper token counts, cost, and
task_type attribution.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import capsera
from capsera._config import Config
from capsera._interceptor import Interceptor
from capsera._models import TokenEvent

from .conftest import CapturingEmitter


def _make_fake_openai_module_with_embeddings(
    chat_response: MagicMock | None = None,
    embedding_response: MagicMock | None = None,
) -> ModuleType:
    """Build a fake openai module with both chat.completions and embeddings."""
    fake_openai = ModuleType("openai")
    fake_resources = ModuleType("openai.resources")
    fake_openai.resources = fake_resources

    # Chat completions (minimal)
    fake_chat = ModuleType("openai.resources.chat")
    fake_resources.chat = fake_chat
    fake_completions_mod = ModuleType("openai.resources.chat.completions")
    fake_chat.completions = fake_completions_mod

    class FakeCompletions:
        _capsera_patched = False
        def create(self, **kwargs):
            return chat_response

    class FakeAsyncCompletions:
        _capsera_patched = False
        async def create(self, **kwargs):
            return chat_response

    fake_completions_mod.Completions = FakeCompletions
    fake_completions_mod.AsyncCompletions = FakeAsyncCompletions

    # Embeddings
    fake_embeddings_mod = ModuleType("openai.resources.embeddings")
    fake_resources.embeddings = fake_embeddings_mod

    class FakeEmbeddings:
        _capsera_patched = False
        def create(self, **kwargs):
            return embedding_response

    class FakeAsyncEmbeddings:
        _capsera_patched = False
        async def create(self, **kwargs):
            return embedding_response

    fake_embeddings_mod.Embeddings = FakeEmbeddings
    fake_embeddings_mod.AsyncEmbeddings = FakeAsyncEmbeddings

    return fake_openai


def make_embedding_response(
    model: str = "text-embedding-3-small",
    prompt_tokens: int = 50,
    total_tokens: int = 50,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.total_tokens = total_tokens
    response = MagicMock()
    response.model = model
    response.usage = usage
    return response


def make_legacy_embedding_response(
    model: str = "text-embedding-3-small",
    prompt_tokens: int = 50,
    total_tokens: int = 50,
) -> MagicMock:
    """Simulate a LegacyAPIResponse wrapper (no .usage, but has .parse())."""
    inner = make_embedding_response(model, prompt_tokens, total_tokens)
    wrapper = MagicMock(spec=[])
    wrapper.parse = MagicMock(return_value=inner)
    return wrapper


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


class TestEmbeddingSync:
    def test_embedding_event_emitted(self, config: Config, emitter: CapturingEmitter) -> None:
        embedding_resp = make_embedding_response(
            model="text-embedding-3-small",
            prompt_tokens=100,
            total_tokens=100,
        )
        fake_openai = _make_fake_openai_module_with_embeddings(
            embedding_response=embedding_resp,
        )

        modules = {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
            "openai.resources.embeddings": fake_openai.resources.embeddings,
        }
        with patch.dict(sys.modules, modules):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_embeddings_sync(fake_openai)

            client = fake_openai.resources.embeddings.Embeddings()
            result = client.create(
                model="text-embedding-3-small",
                input="test text",
            )

        assert result is embedding_resp
        events = emitter.drain()
        assert len(events) == 1

        ev = events[0]
        assert ev.provider == "openai"
        assert ev.model == "text-embedding-3-small"
        assert ev.input_tokens == 100
        assert ev.output_tokens == 0
        assert ev.task_type == "embedding"

    def test_embedding_cost_correct(self, config: Config, emitter: CapturingEmitter) -> None:
        embedding_resp = make_embedding_response(
            model="text-embedding-3-small",
            prompt_tokens=1000,
            total_tokens=1000,
        )
        fake_openai = _make_fake_openai_module_with_embeddings(
            embedding_response=embedding_resp,
        )

        modules = {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
            "openai.resources.embeddings": fake_openai.resources.embeddings,
        }
        with patch.dict(sys.modules, modules):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_embeddings_sync(fake_openai)

            client = fake_openai.resources.embeddings.Embeddings()
            client.create(model="text-embedding-3-small", input="test")

        events = emitter.drain()
        # text-embedding-3-small: $0.00002/1k input
        expected_cost = Decimal("0.00002")
        assert abs(Decimal(str(events[0].cost_usd)) - expected_cost) < Decimal("0.000001")

    def test_embedding_with_agent_context(self, config: Config, emitter: CapturingEmitter) -> None:
        embedding_resp = make_embedding_response()
        fake_openai = _make_fake_openai_module_with_embeddings(
            embedding_response=embedding_resp,
        )

        modules = {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
            "openai.resources.embeddings": fake_openai.resources.embeddings,
        }
        with patch.dict(sys.modules, modules):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_embeddings_sync(fake_openai)

            with capsera.tag(agent="retrieval_agent", team="search"):
                client = fake_openai.resources.embeddings.Embeddings()
                client.create(model="text-embedding-3-small", input="query")

        events = emitter.drain()
        assert events[0].agent_id == "retrieval_agent"
        assert events[0].team_id == "search"

    def test_embedding_idempotent_patching(self, config: Config, emitter: CapturingEmitter) -> None:
        embedding_resp = make_embedding_response()
        fake_openai = _make_fake_openai_module_with_embeddings(
            embedding_response=embedding_resp,
        )

        modules = {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
            "openai.resources.embeddings": fake_openai.resources.embeddings,
        }
        with patch.dict(sys.modules, modules):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_embeddings_sync(fake_openai)
            interceptor._patch_openai_embeddings_sync(fake_openai)

            client = fake_openai.resources.embeddings.Embeddings()
            client.create(model="text-embedding-3-small", input="test")

        events = emitter.drain()
        assert len(events) == 1

    def test_legacy_response_wrapper(self, config: Config, emitter: CapturingEmitter) -> None:
        """LegacyAPIResponse wrappers are unwrapped before accessing .usage."""
        legacy_resp = make_legacy_embedding_response(
            model="text-embedding-3-small",
            prompt_tokens=75,
        )
        fake_openai = _make_fake_openai_module_with_embeddings(
            embedding_response=legacy_resp,
        )

        modules = {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
            "openai.resources.embeddings": fake_openai.resources.embeddings,
        }
        with patch.dict(sys.modules, modules):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_embeddings_sync(fake_openai)

            client = fake_openai.resources.embeddings.Embeddings()
            client.create(model="text-embedding-3-small", input="test")

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 75


class TestEmbeddingAsync:
    @pytest.mark.asyncio
    async def test_async_embedding_event_emitted(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        embedding_resp = make_embedding_response(
            model="text-embedding-3-large",
            prompt_tokens=200,
            total_tokens=200,
        )
        fake_openai = _make_fake_openai_module_with_embeddings(
            embedding_response=embedding_resp,
        )

        modules = {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
            "openai.resources.embeddings": fake_openai.resources.embeddings,
        }
        with patch.dict(sys.modules, modules):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_embeddings_async(fake_openai)

            client = fake_openai.resources.embeddings.AsyncEmbeddings()
            result = await client.create(
                model="text-embedding-3-large",
                input="async test",
            )

        assert result is embedding_resp
        events = emitter.drain()
        assert len(events) == 1

        ev = events[0]
        assert ev.provider == "openai"
        assert ev.model == "text-embedding-3-large"
        assert ev.input_tokens == 200
        assert ev.output_tokens == 0


class TestEmbeddingPricing:
    def test_embedding_3_small_pricing(self) -> None:
        from capsera._pricing import compute_cost
        cost = compute_cost(
            model="text-embedding-3-small",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        # $0.00002/1k * 1000 = $0.02
        assert abs(cost - Decimal("0.02")) < Decimal("0.001")

    def test_embedding_3_large_pricing(self) -> None:
        from capsera._pricing import compute_cost
        cost = compute_cost(
            model="text-embedding-3-large",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        # $0.00013/1k * 1000 = $0.13
        assert abs(cost - Decimal("0.13")) < Decimal("0.01")

    def test_embedding_ada_pricing(self) -> None:
        from capsera._pricing import compute_cost
        cost = compute_cost(
            model="text-embedding-ada-002",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        # $0.0001/1k * 1000 = $0.10
        assert abs(cost - Decimal("0.10")) < Decimal("0.01")
