"""Tests for the Portkey gateway integration (fake portkey_ai module)."""
from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter

# importlib: the package attribute `capsera._interceptor` is shadowed by the
# Interceptor-instance singleton in capsera/__init__.
_runtime = importlib.import_module("capsera._interceptor")


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


@pytest.fixture(autouse=True)
def _restore_suppression():
    saved = set(_runtime._GATEWAY_SUPPRESS_PACKAGES)
    yield
    _runtime._GATEWAY_SUPPRESS_PACKAGES.clear()
    _runtime._GATEWAY_SUPPRESS_PACKAGES.update(saved)


def _openai_shaped(model="gpt-4o", prompt=100, completion=50):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion),
    )


@pytest.fixture
def fake_portkey():
    portkey = ModuleType("portkey_ai")
    api_resources = ModuleType("portkey_ai.api_resources")
    apis = ModuleType("portkey_ai.api_resources.apis")
    chat_mod = ModuleType("portkey_ai.api_resources.apis.chat_complete")
    embed_mod = ModuleType("portkey_ai.api_resources.apis.embeddings")

    class Completions:
        def create(self, **kwargs):
            if kwargs.get("explode"):
                raise RuntimeError("gateway 502")
            return _openai_shaped(model=kwargs.get("model", "gpt-4o"))

    class AsyncCompletions:
        async def create(self, **kwargs):
            return _openai_shaped(model=kwargs.get("model", "gpt-4o"))

    class Embeddings:
        def create(self, **kwargs):
            return SimpleNamespace(
                model="text-embedding-3-small",
                usage=SimpleNamespace(prompt_tokens=256, total_tokens=256),
            )

    chat_mod.Completions = Completions
    chat_mod.AsyncCompletions = AsyncCompletions
    embed_mod.Embeddings = Embeddings
    apis.chat_complete = chat_mod
    apis.embeddings = embed_mod
    api_resources.apis = apis
    portkey.api_resources = api_resources

    modules = {
        "portkey_ai": portkey,
        "portkey_ai.api_resources": api_resources,
        "portkey_ai.api_resources.apis": apis,
        "portkey_ai.api_resources.apis.chat_complete": chat_mod,
        "portkey_ai.api_resources.apis.embeddings": embed_mod,
    }
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    yield SimpleNamespace(chat=Completions, async_chat=AsyncCompletions,
                          embeddings=Embeddings)
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_portkey_chat_capture(config, emitter, fake_portkey):
    Interceptor(config, emitter)._patch_portkey()

    resp = fake_portkey.chat().create(model="gpt-4o", messages=[])
    assert resp.model == "gpt-4o"

    (ev,) = emitter.drain()
    assert ev.gateway == "portkey"
    assert ev.provider == "openai"        # resolved from the model via catalog
    assert ev.model == "gpt-4o"
    assert ev.input_tokens == 100 and ev.output_tokens == 50


def test_portkey_unknown_model_attributes_to_portkey(config, emitter, fake_portkey):
    Interceptor(config, emitter)._patch_portkey()

    fake_portkey.chat().create(model="my-virtual-key-model", messages=[])
    (ev,) = emitter.drain()
    assert ev.provider == "portkey"       # unresolvable model still attributed
    assert ev.gateway == "portkey"


async def test_portkey_async_chat_capture(config, emitter, fake_portkey):
    Interceptor(config, emitter)._patch_portkey()

    await fake_portkey.async_chat().create(model="claude-sonnet-4-6", messages=[])
    (ev,) = emitter.drain()
    assert ev.provider == "anthropic"
    assert ev.gateway == "portkey"


def test_portkey_embeddings_capture(config, emitter, fake_portkey):
    Interceptor(config, emitter)._patch_portkey()

    fake_portkey.embeddings().create(model="text-embedding-3-small", input="hi")
    (ev,) = emitter.drain()
    assert ev.input_tokens == 256 and ev.output_tokens == 0
    assert ev.task_type == "embedding"
    assert ev.gateway == "portkey"


def test_portkey_error_capture(config, emitter, fake_portkey):
    Interceptor(config, emitter)._patch_portkey()

    with pytest.raises(RuntimeError):
        fake_portkey.chat().create(model="gpt-4o", messages=[], explode=True)

    (ev,) = emitter.drain()
    assert ev.is_error is True
    assert ev.error_type == "RuntimeError"
    assert ev.cost_usd == 0.0


def test_portkey_streaming_passthrough(config, emitter, fake_portkey):
    """Native-SDK streams pass through untouched in v1 — no success event,
    and critically no crash or consumed stream."""
    Interceptor(config, emitter)._patch_portkey()

    fake_portkey.chat().create(model="gpt-4o", messages=[], stream=True)
    assert emitter.drain() == []


def test_portkey_patch_idempotent(config, emitter, fake_portkey):
    interceptor = Interceptor(config, emitter)
    interceptor._patch_portkey()
    first = fake_portkey.chat.create
    interceptor._patch_portkey()
    assert fake_portkey.chat.create is first


def test_missing_portkey_is_silent(config, emitter, monkeypatch):
    monkeypatch.setitem(sys.modules, "portkey_ai", None)
    from capsera.gateways import _portkey

    assert _portkey.install_portkey() is False
