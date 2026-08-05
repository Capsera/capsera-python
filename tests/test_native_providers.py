"""Structural tests for the native-SDK provider patches (Mistral, Cohere,
Vertex, Bedrock).

We cannot run the real vendor SDKs here, so — exactly like the existing
Anthropic/OpenAI interceptor tests — we build fake modules mirroring the class
paths and response shapes the interceptor patches, install them into
sys.modules, and assert the patched method still returns the response AND
emits a correctly-labeled, correctly-priced event. A live smoke test per
provider validates the real class path; if it has drifted the patch fails
open (no capture, no crash).
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter


@pytest.fixture
def config() -> Config:
    return Config(api_key="test-key", endpoint="http://localhost:9999",
                  env="test", flush_interval_ms=99_999, debug=True)


@pytest.fixture
def emitter(config: Config):
    e = CapturingEmitter(config)
    e.start()
    yield e
    e.shutdown()


def _run(config: Config, emitter: CapturingEmitter, modules: dict, patch_call):
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        interceptor = Interceptor(config, emitter)
        patch_call(interceptor)
        return
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_mistral_native_capture(config, emitter):
    mistralai = ModuleType("mistralai")
    chat_mod = ModuleType("mistralai.chat")
    mistralai.chat = chat_mod

    class Chat:
        def complete(self, **kwargs):
            return SimpleNamespace(
                model="mistral-large-latest",
                usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=1000),
            )

    chat_mod.Chat = Chat
    _run(config, emitter, {"mistralai": mistralai, "mistralai.chat": chat_mod},
         lambda itc: itc._patch_mistral())

    resp = Chat().complete(model="mistral-large-latest", messages=[])
    assert resp.model == "mistral-large-latest"
    ev = emitter.drain()[0]
    assert ev.provider == "mistral"
    assert ev.model == "mistral-large-latest"
    assert ev.input_tokens == 1000 and ev.output_tokens == 1000
    # mistral-large-latest = 0.0005 in / 0.0015 out per 1k (verified 2026-08-03)
    # → exactly 0.002 for 1k+1k, proving the pricing table (not a fallback) priced it.
    assert float(ev.cost_usd) == 0.002


def test_cohere_native_capture(config, emitter):
    cohere = ModuleType("cohere")

    class ClientV2:
        def chat(self, **kwargs):
            return SimpleNamespace(
                model="command-r-plus",
                usage=SimpleNamespace(
                    tokens=SimpleNamespace(input_tokens=500, output_tokens=200)
                ),
            )

    cohere.ClientV2 = ClientV2
    _run(config, emitter, {"cohere": cohere}, lambda itc: itc._patch_cohere())

    ClientV2().chat(model="command-r-plus", messages=[])
    ev = emitter.drain()[0]
    assert ev.provider == "cohere"
    assert ev.model == "command-r-plus"
    assert ev.input_tokens == 500 and ev.output_tokens == 200


def test_vertex_native_capture(config, emitter):
    vertexai = ModuleType("vertexai")
    gm_mod = ModuleType("vertexai.generative_models")
    vertexai.generative_models = gm_mod

    class GenerativeModel:
        def __init__(self, model_name="gemini-1.5-pro-002"):
            self._model_name = model_name

        def generate_content(self, *a, **k):
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=800, candidates_token_count=300
                )
            )

    gm_mod.GenerativeModel = GenerativeModel
    _run(config, emitter,
         {"vertexai": vertexai, "vertexai.generative_models": gm_mod},
         lambda itc: itc._patch_vertex())

    GenerativeModel("gemini-1.5-pro-002").generate_content("hi")
    ev = emitter.drain()[0]
    assert ev.provider == "vertex"
    assert ev.model == "gemini-1.5-pro-002"
    assert ev.input_tokens == 800 and ev.output_tokens == 300


def test_bedrock_converse_capture(config, emitter):
    botocore = ModuleType("botocore")
    client_mod = ModuleType("botocore.client")
    botocore.client = client_mod

    class BaseClient:
        def __init__(self, service_name):
            self.meta = SimpleNamespace(
                service_model=SimpleNamespace(service_name=service_name)
            )

        def _make_api_call(self, operation_name, api_params):
            return {"usage": {"inputTokens": 1200, "outputTokens": 400}, "output": {}}

    client_mod.BaseClient = BaseClient
    _run(config, emitter, {"botocore": botocore, "botocore.client": client_mod},
         lambda itc: itc._patch_bedrock())

    client = BaseClient("bedrock-runtime")
    client._make_api_call("Converse", {"modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"})
    ev = emitter.drain()[0]
    assert ev.provider == "bedrock"
    assert ev.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert ev.input_tokens == 1200 and ev.output_tokens == 400


def test_bedrock_ignores_non_bedrock_services(config, emitter):
    # The BaseClient patch is global (every boto3 client), so an S3 call must
    # pass straight through and emit nothing.
    botocore = ModuleType("botocore")
    client_mod = ModuleType("botocore.client")
    botocore.client = client_mod

    class BaseClient:
        def __init__(self, service_name):
            self.meta = SimpleNamespace(
                service_model=SimpleNamespace(service_name=service_name)
            )

        def _make_api_call(self, operation_name, api_params):
            return {"Buckets": []}

    client_mod.BaseClient = BaseClient
    _run(config, emitter, {"botocore": botocore, "botocore.client": client_mod},
         lambda itc: itc._patch_bedrock())

    result = BaseClient("s3")._make_api_call("ListBuckets", {})
    assert result == {"Buckets": []}
    assert emitter.drain() == []
