"""The four stream wrappers — previously the largest untested surface.

Anthropic: ``client.messages.stream(...)`` returns a context manager; the
proxy records usage from ``get_final_message()`` on ``__exit__`` (and must do
so BEFORE closing the inner manager). Iteration without ``with`` is a known
gap, tracked here as a strict xfail so a fix flips it loudly.

OpenAI: ``create(stream=True)`` returns an iteration-recording wrapper; usage
arrives on the final chunk because the interceptor force-injects
``stream_options={"include_usage": True}``. ``with``-only usage and abandoned
iteration record nothing — asserted as documentation of current behavior.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from .conftest import CapturingEmitter, make_anthropic_response, make_openai_usage


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


# =========================================================================== #
# Anthropic fakes — mirror MessageStreamManager / MessageStream
# =========================================================================== #

class FakeMessageStream:
    def __init__(self, final_message: Any) -> None:
        self._final = final_message
        self.chunks = ["Hel", "lo"]
        self.final_message_calls = 0

    def get_final_message(self) -> Any:
        self.final_message_calls += 1
        return self._final

    def __iter__(self):
        return iter(self.chunks)


class FakeMessageStreamManager:
    def __init__(self, stream: FakeMessageStream) -> None:
        self._stream = stream
        self.exited = False
        self.record_seen_exited: bool | None = None

    def __enter__(self) -> FakeMessageStream:
        return self._stream

    def __exit__(self, *args: Any) -> bool:
        self.exited = True
        return False

    def __iter__(self):
        return iter(self._stream)


class FakeAsyncMessageStream:
    def __init__(self, final_message: Any) -> None:
        self._final = final_message
        self.chunks = ["Hel", "lo"]

    async def get_final_message(self) -> Any:
        return self._final

    def __aiter__(self):
        async def gen():
            for c in self.chunks:
                yield c
        return gen()


class FakeAsyncMessageStreamManager:
    def __init__(self, stream: FakeAsyncMessageStream) -> None:
        self._stream = stream
        self.exited = False

    async def __aenter__(self) -> FakeAsyncMessageStream:
        return self._stream

    async def __aexit__(self, *args: Any) -> bool:
        self.exited = True
        return False


def _fake_anthropic(stream_factory: Any, async_stream_factory: Any = None) -> ModuleType:
    """Anthropic module tree where Messages.stream returns stream_factory()."""
    fake_anthropic = ModuleType("anthropic")
    fake_resources = ModuleType("anthropic.resources")
    fake_anthropic.resources = fake_resources
    fake_messages_mod = ModuleType("anthropic.resources.messages")
    fake_resources.messages = fake_messages_mod

    class FakeMessages:
        _capsera_patched = False

        def create(self, **kwargs: Any) -> Any:
            raise AssertionError("create() must not be hit by stream tests")

        def stream(self, **kwargs: Any) -> Any:
            return stream_factory(**kwargs)

    class FakeAsyncMessages:
        _capsera_patched = False

        async def create(self, **kwargs: Any) -> Any:
            raise AssertionError("create() must not be hit by stream tests")

        # anthropic's AsyncMessages.stream is a sync method returning an
        # async context manager — the SDK does not await it.
        def stream(self, **kwargs: Any) -> Any:
            return (async_stream_factory or stream_factory)(**kwargs)

    fake_messages_mod.Messages = FakeMessages
    fake_messages_mod.AsyncMessages = FakeAsyncMessages
    return fake_anthropic


class TestAnthropicStreamProxy:
    def test_with_block_emits_one_event_before_manager_close(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        final = make_anthropic_response(model="claude-haiku-4-5",
                                        input_tokens=100, output_tokens=50)
        stream = FakeMessageStream(final)
        mgr = FakeMessageStreamManager(stream)

        def record_order_probe():
            # capture whether __exit__ already ran when usage is read
            mgr.record_seen_exited = mgr.exited
            stream.final_message_calls += 1
            return final
        stream.get_final_message = record_order_probe

        fake_anthropic = _fake_anthropic(lambda **kw: mgr)
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            Interceptor(config, emitter)._patch_anthropic_sync(fake_anthropic)
            client = fake_anthropic.resources.messages.Messages()
            with client.stream(model="claude-haiku-4-5", max_tokens=16,
                               messages=[]) as s:
                consumed = "".join(s)

        assert consumed == "Hello"
        assert mgr.exited, "inner manager must still be closed"
        assert mgr.record_seen_exited is False, \
            "usage must be read BEFORE the inner __exit__ (stream still live)"
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 100
        assert events[0].output_tokens == 50
        assert events[0].provider == "anthropic"
        assert events[0].cost_usd > 0

    @pytest.mark.xfail(
        strict=True,
        reason="known gap: iteration-only anthropic streams (no `with`) are "
               "never recorded — _AnthropicStreamProxy.__iter__ does not "
               "trigger _record_from_stream",
    )
    def test_iteration_without_with_block_is_recorded(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        final = make_anthropic_response()
        mgr = FakeMessageStreamManager(FakeMessageStream(final))
        fake_anthropic = _fake_anthropic(lambda **kw: mgr)
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            Interceptor(config, emitter)._patch_anthropic_sync(fake_anthropic)
            client = fake_anthropic.resources.messages.Messages()
            list(client.stream(model="claude-haiku-4-5", messages=[]))

        assert len(emitter.drain()) == 1

    def test_attribute_access_forwards_to_manager(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        mgr = FakeMessageStreamManager(FakeMessageStream(make_anthropic_response()))
        mgr.custom_marker = "forwarded"
        fake_anthropic = _fake_anthropic(lambda **kw: mgr)
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            Interceptor(config, emitter)._patch_anthropic_sync(fake_anthropic)
            client = fake_anthropic.resources.messages.Messages()
            proxy = client.stream(model="claude-haiku-4-5", messages=[])
        assert proxy.custom_marker == "forwarded"

    def test_error_opening_stream_emits_error_event_and_propagates(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        def boom(**kwargs: Any) -> Any:
            raise RuntimeError("stream open failed")

        fake_anthropic = _fake_anthropic(boom)
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            Interceptor(config, emitter)._patch_anthropic_sync(fake_anthropic)
            client = fake_anthropic.resources.messages.Messages()
            with pytest.raises(RuntimeError, match="stream open failed"):
                client.stream(model="claude-haiku-4-5", messages=[])

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].is_error is True
        assert events[0].error_type == "RuntimeError"

    async def test_async_with_block_emits_one_event(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        final = make_anthropic_response(input_tokens=70, output_tokens=30)
        mgr = FakeAsyncMessageStreamManager(FakeAsyncMessageStream(final))
        fake_anthropic = _fake_anthropic(lambda **kw: mgr, lambda **kw: mgr)
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            Interceptor(config, emitter)._patch_anthropic_async(fake_anthropic)
            client = fake_anthropic.resources.messages.AsyncMessages()
            chunks = []
            async with client.stream(model="claude-haiku-4-5", messages=[]) as s:
                async for part in s:
                    chunks.append(part)

        assert chunks == ["Hel", "lo"]
        assert mgr.exited
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 70
        assert events[0].output_tokens == 30


# =========================================================================== #
# OpenAI fakes — chunk stream, usage only on the final chunk
# =========================================================================== #

def _chunks_with_final_usage(model: str = "gpt-4o-mini"):
    return [
        SimpleNamespace(model=model, usage=None, choices=["delta-1"]),
        SimpleNamespace(model=model, usage=None, choices=["delta-2"]),
        SimpleNamespace(model=model, usage=make_openai_usage(80, 40), choices=[]),
    ]


class FakeOpenAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class FakeAsyncOpenAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


def _fake_openai(sync_result_factory: Any, async_result_factory: Any = None) -> ModuleType:
    fake_openai = ModuleType("openai")
    fake_resources = ModuleType("openai.resources")
    fake_openai.resources = fake_resources
    fake_chat = ModuleType("openai.resources.chat")
    fake_resources.chat = fake_chat
    fake_completions_mod = ModuleType("openai.resources.chat.completions")
    fake_chat.completions = fake_completions_mod

    class FakeCompletions:
        _capsera_patched = False
        last_kwargs: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> Any:
            FakeCompletions.last_kwargs = kwargs
            return sync_result_factory(**kwargs)

    class FakeAsyncCompletions:
        _capsera_patched = False
        last_kwargs: dict[str, Any] | None = None

        async def create(self, **kwargs: Any) -> Any:
            FakeAsyncCompletions.last_kwargs = kwargs
            return (async_result_factory or sync_result_factory)(**kwargs)

    fake_completions_mod.Completions = FakeCompletions
    fake_completions_mod.AsyncCompletions = FakeAsyncCompletions
    return fake_openai


class TestOpenAIStreamWrapper:
    def test_exhausted_iteration_emits_event_with_final_chunk_usage(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        fake_openai = _fake_openai(
            lambda **kw: FakeOpenAIStream(_chunks_with_final_usage()))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            Interceptor(config, emitter)._patch_openai_sync(fake_openai)
            client = fake_openai.resources.chat.completions.Completions()
            chunks = list(client.create(model="gpt-4o-mini", stream=True,
                                        messages=[]))

        assert len(chunks) == 3  # NI: the app sees every chunk unchanged
        sent = fake_openai.resources.chat.completions.Completions.last_kwargs
        assert sent["stream_options"] == {"include_usage": True}, \
            "interceptor must force-inject include_usage"
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 80
        assert events[0].output_tokens == 40
        assert events[0].model == "gpt-4o-mini"
        assert events[0].cost_usd > 0

    def test_caller_stream_options_are_preserved_and_extended(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        fake_openai = _fake_openai(
            lambda **kw: FakeOpenAIStream(_chunks_with_final_usage()))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            Interceptor(config, emitter)._patch_openai_sync(fake_openai)
            client = fake_openai.resources.chat.completions.Completions()
            list(client.create(model="gpt-4o-mini", stream=True, messages=[],
                               stream_options={"custom": "kept"}))

        sent = fake_openai.resources.chat.completions.Completions.last_kwargs
        assert sent["stream_options"] == {"custom": "kept", "include_usage": True}

    def test_abandoned_iteration_emits_nothing(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Documented behavior: usage rides the FINAL chunk, so a consumer
        that stops early never supplies token counts — no event is emitted
        (better silence than a fabricated zero-cost billing row)."""
        fake_openai = _fake_openai(
            lambda **kw: FakeOpenAIStream(_chunks_with_final_usage()))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            Interceptor(config, emitter)._patch_openai_sync(fake_openai)
            client = fake_openai.resources.chat.completions.Completions()
            wrapper = client.create(model="gpt-4o-mini", stream=True, messages=[])
            it = iter(wrapper)
            next(it)
            it.close()  # deterministic generator finalization

        assert emitter.drain() == []

    def test_with_block_without_iteration_emits_nothing(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        """Documented behavior: __enter__/__exit__ are pass-through; only
        iteration observes chunks, so `with` alone records nothing."""
        fake_openai = _fake_openai(
            lambda **kw: FakeOpenAIStream(_chunks_with_final_usage()))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            Interceptor(config, emitter)._patch_openai_sync(fake_openai)
            client = fake_openai.resources.chat.completions.Completions()
            with client.create(model="gpt-4o-mini", stream=True, messages=[]):
                pass

        assert emitter.drain() == []

    def test_error_mid_stream_propagates_without_phantom_event(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        def broken_chunks():
            yield SimpleNamespace(model="gpt-4o-mini", usage=None, choices=["d"])
            raise RuntimeError("connection dropped")

        class BrokenStream:
            def __iter__(self):
                return broken_chunks()

        fake_openai = _fake_openai(lambda **kw: BrokenStream())
        with patch.dict(sys.modules, {"openai": fake_openai}):
            Interceptor(config, emitter)._patch_openai_sync(fake_openai)
            client = fake_openai.resources.chat.completions.Completions()
            with pytest.raises(RuntimeError, match="connection dropped"):
                list(client.create(model="gpt-4o-mini", stream=True, messages=[]))

        assert emitter.drain() == [], \
            "no usage was ever observed — a phantom event would corrupt billing"

    async def test_async_exhausted_iteration_emits_event(
        self, config: Config, emitter: CapturingEmitter
    ) -> None:
        fake_openai = _fake_openai(
            None, lambda **kw: FakeAsyncOpenAIStream(_chunks_with_final_usage()))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            Interceptor(config, emitter)._patch_openai_async(fake_openai)
            client = fake_openai.resources.chat.completions.AsyncCompletions()
            stream = await client.create(model="gpt-4o-mini", stream=True,
                                         messages=[])
            chunks = [c async for c in stream]

        assert len(chunks) == 3
        sent = fake_openai.resources.chat.completions.AsyncCompletions.last_kwargs
        assert sent["stream_options"] == {"include_usage": True}
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].input_tokens == 80
        assert events[0].output_tokens == 40
