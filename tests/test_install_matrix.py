"""Install-time robustness: missing providers degrade gracefully (COM-5)
and repeated patching never double-wraps (COM-4).

Complements tests/validation/test_concurrency.py's concurrent-install test.
"""
from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter, make_anthropic_response, make_openai_response
from .test_interceptor import _make_fake_anthropic_module, _make_fake_openai_module


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


def test_missing_anthropic_warns_and_openai_still_patches(
    config, emitter, caplog
):
    """COM-5: one provider failing to import must not take down capture for
    the others — and must be a visible WARNING, not silence."""
    response = make_openai_response(model="gpt-4o-mini")
    fake_openai = _make_fake_openai_module(response)
    # sys.modules[name] = None makes `import name` raise ImportError; every
    # provider except our fake openai is "not installed".
    blocked = {name: None for name in (
        "anthropic", "google", "google.generativeai", "google.genai",
        "mistralai", "mistralai.chat", "cohere", "vertexai",
        "vertexai.generative_models", "botocore", "botocore.client",
    )}
    with patch.dict(sys.modules, {"openai": fake_openai, **blocked}):
        with caplog.at_level(logging.WARNING, logger="capsera"):
            Interceptor(config, emitter).install()
        client = fake_openai.resources.chat.completions.Completions()
        client.create(model="gpt-4o-mini", messages=[])

    assert any("anthropic" in r.message for r in caplog.records), \
        "missing provider must log a WARNING naming it"
    events = emitter.drain()
    assert len(events) == 1, "openai capture must survive anthropic's absence"
    assert events[0].model == "gpt-4o-mini"


def test_double_patch_emits_single_event_per_call(config, emitter):
    """COM-4: patching the same class twice (double init(), re-import) must
    not double-wrap — a double-wrapped create() bills every call twice."""
    response = make_anthropic_response()
    fake_anthropic = _make_fake_anthropic_module(response)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        interceptor = Interceptor(config, emitter)
        interceptor._patch_anthropic_sync(fake_anthropic)
        interceptor._patch_anthropic_sync(fake_anthropic)  # second install
        client = fake_anthropic.resources.messages.Messages()
        client.create(model="claude-haiku-4-5", messages=[])

    assert len(emitter.drain()) == 1


def test_reinit_updates_emitter_for_already_patched_class(config):
    """Re-init with a fresh emitter must route NEW calls to the NEW emitter —
    patched closures resolve the runtime at call time, not at patch time."""
    response = make_anthropic_response()
    fake_anthropic = _make_fake_anthropic_module(response)
    first = CapturingEmitter(config)
    second = CapturingEmitter(config)
    first.start()
    second.start()
    try:
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            Interceptor(config, first)._patch_anthropic_sync(fake_anthropic)
            Interceptor(config, second)._patch_anthropic_sync(fake_anthropic)
            client = fake_anthropic.resources.messages.Messages()
            client.create(model="claude-haiku-4-5", messages=[])

        assert first.drain() == []
        assert len(second.drain()) == 1
    finally:
        first.shutdown()
        second.shutdown()
