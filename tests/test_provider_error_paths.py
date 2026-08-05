"""Error capture on the surfaces that historically had none.

Contract for every patched surface: a provider exception must (a) reach the
app completely unchanged, and (b) leave exactly one zero-cost event with
``is_error=True`` and the exception's type name — a failed call is spend
telemetry too (error-rate per agent/model is a product feature).

Covers: google.genai, OpenAI embeddings (sync + async), Mistral, Cohere,
Vertex, Bedrock. The first-party Anthropic/OpenAI chat paths already had
error capture (tested in test_error_capture.py).
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter


@pytest.fixture
def config() -> Config:
    return Config(api_key="k", endpoint="http://localhost:9999", env="test",
                  flush_interval_ms=99_999, enable_budget_enforcement=False)


@pytest.fixture
def emitter(config: Config):
    e = CapturingEmitter(config)
    e.start()
    yield e
    e.shutdown()


def assert_single_error_event(emitter: CapturingEmitter, provider: str,
                              model: str) -> None:
    events = emitter.drain()
    assert len(events) == 1, f"expected 1 error event, got {events!r}"
    ev = events[0]
    assert ev.is_error is True
    assert ev.error_type == "RuntimeError"
    assert ev.provider == provider
    assert ev.model == model
    assert ev.input_tokens == 0 and ev.output_tokens == 0
    assert float(ev.cost_usd) == 0.0


class _Boom(RuntimeError):
    pass


def test_google_genai_error_is_captured_and_propagates(config, emitter):
    fake_google = ModuleType("google")
    fake_genai = ModuleType("google.genai")
    fake_models_mod = ModuleType("google.genai.models")
    fake_google.genai = fake_genai
    fake_genai.models = fake_models_mod

    class Models:
        def generate_content(self, **kwargs):
            raise RuntimeError("gemini unavailable")

    fake_models_mod.Models = Models
    with patch.dict(sys.modules, {"google": fake_google,
                                  "google.genai": fake_genai,
                                  "google.genai.models": fake_models_mod}):
        Interceptor(config, emitter)._patch_google_genai(fake_genai)
        with pytest.raises(RuntimeError, match="gemini unavailable"):
            Models().generate_content(model="gemini-2.5-flash", contents="hi")

    assert_single_error_event(emitter, "google", "gemini-2.5-flash")


def _fake_openai_embeddings_raising() -> ModuleType:
    fake_openai = ModuleType("openai")
    fake_resources = ModuleType("openai.resources")
    fake_openai.resources = fake_resources
    fake_embeddings_mod = ModuleType("openai.resources.embeddings")
    fake_resources.embeddings = fake_embeddings_mod

    class Embeddings:
        _capsera_patched = False

        def create(self, **kwargs):
            raise RuntimeError("embeddings down")

    class AsyncEmbeddings:
        _capsera_patched = False

        async def create(self, **kwargs):
            raise RuntimeError("embeddings down")

    fake_embeddings_mod.Embeddings = Embeddings
    fake_embeddings_mod.AsyncEmbeddings = AsyncEmbeddings
    return fake_openai


def test_openai_embeddings_sync_error_is_captured(config, emitter):
    fake_openai = _fake_openai_embeddings_raising()
    with patch.dict(sys.modules, {"openai": fake_openai}):
        Interceptor(config, emitter)._patch_openai_embeddings_sync(fake_openai)
        cls = fake_openai.resources.embeddings.Embeddings
        with pytest.raises(RuntimeError, match="embeddings down"):
            cls().create(model="text-embedding-3-small", input="hi")

    assert_single_error_event(emitter, "openai", "text-embedding-3-small")


async def test_openai_embeddings_async_error_is_captured(config, emitter):
    fake_openai = _fake_openai_embeddings_raising()
    with patch.dict(sys.modules, {"openai": fake_openai}):
        Interceptor(config, emitter)._patch_openai_embeddings_async(fake_openai)
        cls = fake_openai.resources.embeddings.AsyncEmbeddings
        with pytest.raises(RuntimeError, match="embeddings down"):
            await cls().create(model="text-embedding-3-small", input="hi")

    assert_single_error_event(emitter, "openai", "text-embedding-3-small")


def test_mistral_error_is_captured(config, emitter):
    mistralai = ModuleType("mistralai")
    chat_mod = ModuleType("mistralai.chat")
    mistralai.chat = chat_mod

    class Chat:
        def complete(self, **kwargs):
            raise RuntimeError("mistral down")

    chat_mod.Chat = Chat
    with patch.dict(sys.modules, {"mistralai": mistralai,
                                  "mistralai.chat": chat_mod}):
        Interceptor(config, emitter)._patch_mistral()
        with pytest.raises(RuntimeError, match="mistral down"):
            Chat().complete(model="mistral-small-latest", messages=[])

    assert_single_error_event(emitter, "mistral", "mistral-small-latest")


def test_cohere_error_is_captured(config, emitter):
    cohere = ModuleType("cohere")

    class ClientV2:
        def chat(self, **kwargs):
            raise RuntimeError("cohere down")

    cohere.ClientV2 = ClientV2
    with patch.dict(sys.modules, {"cohere": cohere}):
        Interceptor(config, emitter)._patch_cohere()
        with pytest.raises(RuntimeError, match="cohere down"):
            ClientV2().chat(model="command-r", messages=[])

    assert_single_error_event(emitter, "cohere", "command-r")


def test_vertex_error_is_captured(config, emitter):
    vertexai = ModuleType("vertexai")
    gm_mod = ModuleType("vertexai.generative_models")
    vertexai.generative_models = gm_mod

    class GenerativeModel:
        def __init__(self, model_name="gemini-2.0-flash-001"):
            self._model_name = model_name

        def generate_content(self, *args, **kwargs):
            raise RuntimeError("vertex down")

    gm_mod.GenerativeModel = GenerativeModel
    with patch.dict(sys.modules, {"vertexai": vertexai,
                                  "vertexai.generative_models": gm_mod}):
        Interceptor(config, emitter)._patch_vertex()
        with pytest.raises(RuntimeError, match="vertex down"):
            GenerativeModel().generate_content("hi")

    assert_single_error_event(emitter, "vertex", "gemini-2.0-flash-001")


def _fake_botocore(service_name: str) -> tuple[ModuleType, ModuleType, type]:
    botocore = ModuleType("botocore")
    client_mod = ModuleType("botocore.client")
    botocore.client = client_mod

    class _ServiceModel:
        def __init__(self, name: str) -> None:
            self.service_name = name

    class _Meta:
        def __init__(self, name: str) -> None:
            self.service_model = _ServiceModel(name)

    class BaseClient:
        def __init__(self) -> None:
            self.meta = _Meta(service_name)

        def _make_api_call(self, operation_name, api_params):
            raise RuntimeError("aws down")

    client_mod.BaseClient = BaseClient
    return botocore, client_mod, BaseClient


def test_bedrock_converse_error_is_captured(config, emitter):
    botocore, client_mod, BaseClient = _fake_botocore("bedrock-runtime")
    with patch.dict(sys.modules, {"botocore": botocore,
                                  "botocore.client": client_mod}):
        Interceptor(config, emitter)._patch_bedrock()
        with pytest.raises(RuntimeError, match="aws down"):
            BaseClient()._make_api_call(
                "Converse", {"modelId": "amazon.nova-micro-v1:0"})

    assert_single_error_event(emitter, "bedrock", "amazon.nova-micro-v1:0")


def test_bedrock_non_bedrock_service_error_passes_through_without_event(
    config, emitter
):
    """The botocore patch intercepts ALL AWS calls; an S3 failure must not
    produce an LLM billing event."""
    botocore, client_mod, BaseClient = _fake_botocore("s3")
    with patch.dict(sys.modules, {"botocore": botocore,
                                  "botocore.client": client_mod}):
        Interceptor(config, emitter)._patch_bedrock()
        with pytest.raises(RuntimeError, match="aws down"):
            BaseClient()._make_api_call("GetObject", {"Bucket": "b"})

    assert emitter.drain() == []
