"""
Integration tests simulating LangChain behavior.

These tests prevent regressions for:
- Bug 1: LegacyAPIResponse crash (OpenAI SDK via LangChain returns a wrapper)
- Bug 3: caller_file resolving to LangChain internals
- Session propagation via set_session()
- Framework detection adding LangChain to skip list
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import capsera
from capsera._config import Config
from capsera._context import _context_stack
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter, make_openai_response


def _make_fake_openai_module(response):
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


class TestLegacyAPIResponse:
    """Regression tests for Bug 1: LegacyAPIResponse wrapper from LangChain."""

    def test_legacy_response_is_unwrapped(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """When OpenAI returns a LegacyAPIResponse, the SDK unwraps via .parse()."""
        inner_response = make_openai_response(
            model="gpt-4o-mini",
            prompt_tokens=500,
            completion_tokens=200,
        )
        # Simulate LegacyAPIResponse: no .usage attribute, but has .parse()
        legacy_wrapper = MagicMock(spec=[])
        legacy_wrapper.parse = MagicMock(return_value=inner_response)

        fake_openai = _make_fake_openai_module(legacy_wrapper)

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
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result is legacy_wrapper
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 500
        assert events[0].output_tokens == 200
        assert events[0].model == "gpt-4o-mini"

    def test_normal_response_still_works(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Normal responses with .usage attribute still work correctly."""
        response = make_openai_response(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
        )
        fake_openai = _make_fake_openai_module(response)

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
        assert len(events) == 1
        assert events[0].input_tokens == 100


class TestSessionPropagation:
    """Tests for request-level session_id propagation."""

    def test_set_session_flows_to_events(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        response = make_openai_response()
        fake_openai = _make_fake_openai_module(response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            capsera.set_session("session-abc-123")
            try:
                client = fake_openai.resources.chat.completions.Completions()
                client.create(model="gpt-4o", messages=[])
            finally:
                capsera.clear_session()

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].session_id == "session-abc-123"

    def test_tag_session_overrides_thread_session(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """tag(session_id=...) takes precedence over set_session()."""
        response = make_openai_response()
        fake_openai = _make_fake_openai_module(response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            capsera.set_session("thread-session")
            try:
                with capsera.tag(agent="test", session_id="tag-session"):
                    client = fake_openai.resources.chat.completions.Completions()
                    client.create(model="gpt-4o", messages=[])
            finally:
                capsera.clear_session()

        events = emitter.drain()
        assert events[0].session_id == "tag-session"

    def test_clear_session_removes_session(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        response = make_openai_response()
        fake_openai = _make_fake_openai_module(response)

        with patch.dict(sys.modules, {
            "openai": fake_openai,
            "openai.resources": fake_openai.resources,
            "openai.resources.chat": fake_openai.resources.chat,
            "openai.resources.chat.completions": fake_openai.resources.chat.completions,
        }):
            interceptor = Interceptor(config, emitter)
            interceptor._patch_openai_sync(fake_openai)

            capsera.set_session("temp-session")
            capsera.clear_session()

            client = fake_openai.resources.chat.completions.Completions()
            client.create(model="gpt-4o", messages=[])

        events = emitter.drain()
        assert events[0].session_id is None

    def test_session_thread_isolation(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """set_session() is thread-local — doesn't leak across threads."""
        import threading

        results = {}

        def thread_fn(name: str, session: str):
            _context_stack.set_session(session)
            import time
            time.sleep(0.05)
            results[name] = _context_stack.get_session()
            _context_stack.clear_session()

        t1 = threading.Thread(target=thread_fn, args=("t1", "session-1"))
        t2 = threading.Thread(target=thread_fn, args=("t2", "session-2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "session-1"
        assert results["t2"] == "session-2"


class TestFrameworkDetection:
    """Tests for dynamic framework detection in _SKIP_FRAME_FRAGMENTS."""

    def test_detect_frameworks_adds_installed_packages(self) -> None:
        # Reset the initialized flag for testing
        import importlib

        from capsera._interceptor import _detect_frameworks
        _mod = importlib.import_module("capsera._interceptor")
        _mod._skip_fragments_initialized = False

        fragments = _detect_frameworks()

        # Base fragments should always be present
        assert "capsera" in fragments
        assert "site-packages/openai" in fragments

        # Reset for other tests
        _mod._skip_fragments_initialized = False

    def test_detect_frameworks_is_idempotent(self) -> None:
        import importlib

        from capsera._interceptor import _detect_frameworks
        _mod = importlib.import_module("capsera._interceptor")
        _mod._skip_fragments_initialized = False

        result1 = _detect_frameworks()
        result2 = _detect_frameworks()

        assert result1 == result2
        _mod._skip_fragments_initialized = False

    def test_autogen_04_detected_without_top_level_autogen_module(self, monkeypatch) -> None:
        # AutoGen 0.4+ ships as autogen_agentchat/autogen_core/autogen_ext with
        # NO top-level `autogen` module, so detection must fire off autogen_core
        # (the base dependency every 0.4 layout shares).
        import importlib
        import sys
        import types

        _mod = importlib.import_module("capsera._interceptor")
        monkeypatch.setitem(sys.modules, "autogen_core", types.ModuleType("autogen_core"))
        _mod._skip_fragments_initialized = False

        fragments = _mod._detect_frameworks()

        assert "site-packages/autogen" in fragments
        _mod._skip_fragments_initialized = False
