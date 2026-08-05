"""Observability: `get_interception_report()` ("is everything I use hooked?")
and the debug tap (one structured log line per intercepted call)."""
from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest

import capsera
from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter, make_anthropic_response
from .test_interceptor import _make_fake_anthropic_module


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


def test_report_distinguishes_patched_unpatched_missing(config, emitter):
    fake_anthropic = _make_fake_anthropic_module(make_anthropic_response())
    blocked = {name: None for name in (
        "openai", "google", "google.generativeai", "google.genai",
        "mistralai", "mistralai.chat", "cohere", "vertexai",
        "vertexai.generative_models", "botocore", "botocore.client",
    )}
    with patch.dict(sys.modules, {"anthropic": fake_anthropic, **blocked}):
        report = capsera.get_interception_report()
        assert report["anthropic.messages"] == "unpatched"
        assert report["openai.chat_completions"] == "not-installed"

        Interceptor(config, emitter).install()
        report = capsera.get_interception_report()

    assert report["anthropic.messages"] == "patched"
    assert report["anthropic.messages_async"] == "patched"
    assert report["openai.chat_completions"] == "not-installed"
    assert report["mistral.chat_complete"] == "not-installed"
    assert report["bedrock.converse"] == "not-installed"


def test_debug_tap_logs_one_line_per_call(config, emitter, caplog):
    debug_config = Config(api_key="k", endpoint="http://localhost:9999",
                          env="test", flush_interval_ms=99_999,
                          enable_budget_enforcement=False, debug=True)
    fake_anthropic = _make_fake_anthropic_module(
        make_anthropic_response(model="claude-haiku-4-5",
                                input_tokens=100, output_tokens=50))
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        Interceptor(debug_config, emitter)._patch_anthropic_sync(fake_anthropic)
        client = fake_anthropic.resources.messages.Messages()
        with caplog.at_level(logging.INFO, logger="capsera"):
            client.create(model="claude-haiku-4-5", messages=[])

    taps = [r for r in caplog.records if "capsera: intercepted" in r.message]
    assert len(taps) == 1
    line = taps[0].message
    for fragment in ("provider=anthropic", "model=claude-haiku-4-5",
                     "tokens=100/50", "error=False"):
        assert fragment in line, f"{fragment!r} missing from {line!r}"


def test_debug_tap_silent_by_default(config, emitter, caplog):
    fake_anthropic = _make_fake_anthropic_module(make_anthropic_response())
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        Interceptor(config, emitter)._patch_anthropic_sync(fake_anthropic)
        client = fake_anthropic.resources.messages.Messages()
        with caplog.at_level(logging.INFO, logger="capsera"):
            client.create(model="claude-haiku-4-5", messages=[])

    assert not any("capsera: intercepted" in r.message for r in caplog.records)


def test_package_exposes_dunder_version():
    """`capsera.__version__` is the first thing a user or bug report reaches
    for. It lived only in capsera._version until the first PyPI release, so a
    clean-venv install had no public version attribute at all."""
    import capsera
    from capsera._version import __version__ as source_version

    assert capsera.__version__ == source_version
    assert "__version__" in capsera.__all__
