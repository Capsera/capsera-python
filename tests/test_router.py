"""
Tests for model routing integration via _interceptor._maybe_route.
"""
from __future__ import annotations

import copy

from capsera._config import Config
from capsera._interceptor import _maybe_route
from capsera._router import Router


def _cfg(**overrides) -> Config:
    base = dict(
        api_key="test",
        endpoint="http://localhost:9999",
        env="test",
        flush_interval_ms=9999,
    )
    base.update(overrides)
    return Config(**base)


class TestMaybeRoute:
    def test_passthrough_when_routing_disabled(self) -> None:
        """When enable_routing=False (default), kwargs are returned unchanged."""
        cfg = _cfg(enable_routing=False)
        kwargs = {"model": "claude-opus-4-6", "max_tokens": 100}
        result_kwargs, original = _maybe_route(kwargs, cfg)
        assert result_kwargs == kwargs
        assert original is None

    def test_passthrough_when_no_router(self) -> None:
        """When enable_routing=True but no Router instance, pass through."""
        cfg = _cfg(enable_routing=True)
        kwargs = {"model": "claude-opus-4-6", "max_tokens": 100}
        result_kwargs, original = _maybe_route(kwargs, cfg, router=None)
        assert result_kwargs["model"] == "claude-opus-4-6"
        assert original is None

    def test_passthrough_when_routing_enabled_no_rules(self) -> None:
        """When routing is enabled with a Router but no rules, pass through."""
        cfg = _cfg(enable_routing=True)
        router = Router(cfg)
        kwargs = {"model": "claude-opus-4-6", "max_tokens": 100}
        result_kwargs, original = _maybe_route(kwargs, cfg, router)
        assert result_kwargs["model"] == "claude-opus-4-6"
        assert original is None

    def test_routes_when_rule_matches(self) -> None:
        """When a rule matches, _maybe_route returns modified kwargs."""
        cfg = _cfg(
            enable_routing=True,
            routing_rules=[
                {"type": "model_map", "from_model": "claude-opus-4-6", "target_model": "claude-haiku-4-5"},
            ],
        )
        router = Router(cfg)
        kwargs = {"model": "claude-opus-4-6", "max_tokens": 100}
        result_kwargs, original = _maybe_route(kwargs, cfg, router)
        assert result_kwargs["model"] == "claude-haiku-4-5"
        assert result_kwargs["max_tokens"] == 100
        assert original == "claude-opus-4-6"

    def test_does_not_mutate_original_dict(self) -> None:
        """_maybe_route must never mutate the caller's kwargs dict."""
        cfg = _cfg(
            enable_routing=True,
            routing_rules=[
                {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
            ],
        )
        router = Router(cfg)
        original_kwargs = {"model": "gpt-4o", "temperature": 0.7}
        snapshot = copy.deepcopy(original_kwargs)
        _maybe_route(original_kwargs, cfg, router)
        assert original_kwargs == snapshot
