"""Provider responses are untrusted input.

Real SDKs return usage=None on some error paths, omit fields on streaming
finalizers, and (via proxies/gateways) occasionally produce strings where
integers belong. Capture must fail OPEN — the customer's response comes
back untouched — and whatever event does get emitted must still satisfy
the ingest contract.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from ..conftest import CapturingEmitter
from .conftest import backend_event_schema


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


def _patch_mistral_with(config, emitter, usage, model="mistral-large-latest"):
    """Install a fake mistralai module whose complete() returns the given
    usage object, patch it, call it, and return (response, events)."""
    mistralai = ModuleType("mistralai")
    chat_mod = ModuleType("mistralai.chat")
    mistralai.chat = chat_mod

    class Chat:
        def complete(self, **kwargs):
            return SimpleNamespace(model=model, usage=usage)

    chat_mod.Chat = Chat
    saved = {k: sys.modules.get(k) for k in ("mistralai", "mistralai.chat")}
    sys.modules.update({"mistralai": mistralai, "mistralai.chat": chat_mod})
    try:
        interceptor = Interceptor(config, emitter)
        interceptor._patch_mistral()
        resp = Chat().complete(model="mistral-large-latest", messages=[])
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return resp, emitter.drain()


def _assert_contract(event) -> None:
    backend_event_schema().model_validate(event.to_dict())


def test_usage_none_fails_open_with_zero_tokens(config, emitter):
    resp, events = _patch_mistral_with(config, emitter, usage=None)
    assert resp is not None                      # customer response untouched
    if events:                                   # if captured, must be valid
        assert events[0].input_tokens == 0 and events[0].output_tokens == 0
        _assert_contract(events[0])


def test_usage_missing_completion_tokens(config, emitter):
    usage = SimpleNamespace(prompt_tokens=500)   # no completion_tokens at all
    resp, events = _patch_mistral_with(config, emitter, usage=usage)
    assert resp is not None
    assert len(events) == 1
    assert events[0].input_tokens == 500
    assert events[0].output_tokens == 0
    _assert_contract(events[0])


def test_usage_with_string_token_counts(config, emitter):
    # Seen through proxies that re-serialize JSON: numbers become strings.
    usage = SimpleNamespace(prompt_tokens="1000", completion_tokens="250")
    resp, events = _patch_mistral_with(config, emitter, usage=usage)
    assert resp is not None                      # never crash the call
    if events:
        assert isinstance(events[0].input_tokens, int)
        assert isinstance(events[0].output_tokens, int)
        _assert_contract(events[0])


def test_usage_with_negative_tokens_is_floored(config, emitter):
    usage = SimpleNamespace(prompt_tokens=-10, completion_tokens=-5)
    resp, events = _patch_mistral_with(config, emitter, usage=usage)
    assert resp is not None
    assert len(events) == 1
    assert events[0].input_tokens >= 0 and events[0].output_tokens >= 0
    _assert_contract(events[0])


def test_response_model_none_falls_back_to_request_model(config, emitter):
    resp, events = _patch_mistral_with(
        config, emitter,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model=None,
    )
    assert resp is not None
    assert len(events) == 1
    assert events[0].model == "mistral-large-latest"
    _assert_contract(events[0])


def test_booby_trapped_usage_object_fails_open(config, emitter):
    class ExplodingUsage:
        @property
        def prompt_tokens(self):
            raise RuntimeError("SDK internals changed")

        @property
        def completion_tokens(self):
            raise RuntimeError("SDK internals changed")

    resp, events = _patch_mistral_with(config, emitter, usage=ExplodingUsage())
    assert resp is not None                      # the customer call survives
    for ev in events:                            # anything emitted is valid
        _assert_contract(ev)
