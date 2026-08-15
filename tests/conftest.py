"""
Shared pytest fixtures for the Capsera SDK test suite.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from capsera._config import Config
from capsera._emitter import Emitter
from capsera._models import TokenEvent

# --------------------------------------------------------------------------- #
# Config fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def base_config() -> Config:
    """A minimal Config suitable for unit tests (no real backend)."""
    return Config(
        api_key="test-key",
        endpoint="http://localhost:9999",  # non-existent; emitter should handle gracefully
        env="test",
        flush_interval_ms=10_000,  # effectively disable auto-flush in tests
        debug=True,
    )


# --------------------------------------------------------------------------- #
# Emitter fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def capturing_emitter(base_config: Config) -> CapturingEmitter:
    """
    An Emitter subclass that captures events in memory rather than sending them
    over HTTP.  Use ``captured`` to inspect what was enqueued.
    """
    emitter = CapturingEmitter(base_config)
    emitter.start()
    yield emitter
    emitter.shutdown()


class CapturingEmitter(Emitter):
    """Test double that stores events instead of shipping them."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.captured: list[TokenEvent] = []
        self._lock = threading.Lock()

    def _send(self, events: list[TokenEvent]) -> None:  # type: ignore[override]
        with self._lock:
            self.captured.extend(events)

    def drain(self) -> list[TokenEvent]:
        """Flush the queue and return all captured events."""
        self.flush()
        with self._lock:
            events = list(self.captured)
            self.captured.clear()
        return events


# --------------------------------------------------------------------------- #
# Anthropic mock helpers
# --------------------------------------------------------------------------- #

def make_anthropic_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_write: int = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_write
    return usage


def make_anthropic_response(
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_write: int = 0,
) -> MagicMock:
    response = MagicMock()
    response.model = model
    response.usage = make_anthropic_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
    )
    return response


# --------------------------------------------------------------------------- #
# OpenAI mock helpers
# --------------------------------------------------------------------------- #

def make_openai_usage(
    prompt_tokens: int = 80,
    completion_tokens: int = 40,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    return usage


def make_openai_response(
    model: str = "gpt-4o",
    prompt_tokens: int = 80,
    completion_tokens: int = 40,
) -> MagicMock:
    response = MagicMock()
    response.model = model
    response.usage = make_openai_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return response


# --------------------------------------------------------------------------- #
# Global state isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _reset_pooled_budget_clients():
    """Drop the process-wide pooled budget-check clients around every test so
    a client (real or monkeypatched) created in one test never leaks into the
    next."""
    from capsera._interceptor import _reset_budget_clients

    _reset_budget_clients()
    yield
    _reset_budget_clients()
