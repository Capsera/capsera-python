"""Async happy paths that previously had zero coverage.

test_error_capture.py awaits AsyncMessages.create only on its ERROR path;
AsyncCompletions.create and generate_content_async were defined by fakes in
test_interceptor.py but never awaited by any test. These are the paths every
asyncio-based agent app rides.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter, make_anthropic_response, make_openai_response
from .test_interceptor import (
    _make_fake_anthropic_module,
    _make_fake_google_generativeai_module,
    _make_fake_openai_module,
    make_google_response,
)


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


async def test_anthropic_async_create_emits_event(config, emitter):
    response = make_anthropic_response(model="claude-haiku-4-5",
                                       input_tokens=120, output_tokens=60,
                                       cache_read=10, cache_write=5)
    fake_anthropic = _make_fake_anthropic_module(response)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        Interceptor(config, emitter)._patch_anthropic_async(fake_anthropic)
        client = fake_anthropic.resources.messages.AsyncMessages()
        result = await client.create(model="claude-haiku-4-5", max_tokens=16,
                                     messages=[])

    assert result is response  # NI: response object untouched
    events = emitter.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev.provider == "anthropic"
    assert ev.input_tokens == 120 and ev.output_tokens == 60
    assert ev.cache_read_tokens == 10 and ev.cache_write_tokens == 5
    assert ev.cost_usd > 0


async def test_openai_async_create_emits_event(config, emitter):
    response = make_openai_response(model="gpt-4o-mini",
                                    prompt_tokens=90, completion_tokens=30)
    fake_openai = _make_fake_openai_module(response)
    with patch.dict(sys.modules, {"openai": fake_openai}):
        Interceptor(config, emitter)._patch_openai_async(fake_openai)
        client = fake_openai.resources.chat.completions.AsyncCompletions()
        result = await client.create(model="gpt-4o-mini", messages=[])

    assert result is response
    events = emitter.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev.provider == "openai"
    assert ev.model == "gpt-4o-mini"
    assert ev.input_tokens == 90 and ev.output_tokens == 30
    assert ev.cost_usd > 0


async def test_google_generativeai_async_emits_event(config, emitter):
    response = make_google_response(model="gemini-2.5-flash",
                                    input_tokens=200, output_tokens=80)
    fake_google, fake_genai = _make_fake_google_generativeai_module(response)
    with patch.dict(sys.modules, {"google": fake_google,
                                  "google.generativeai": fake_genai}):
        Interceptor(config, emitter)._patch_google_generativeai(fake_genai)
        model = fake_genai.GenerativeModel("gemini-2.5-flash")
        result = await model.generate_content_async("hello")

    assert result is response
    events = emitter.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev.provider == "google"
    assert ev.model == "gemini-2.5-flash"
    assert ev.input_tokens == 200 and ev.output_tokens == 80
    assert ev.cost_usd > 0
