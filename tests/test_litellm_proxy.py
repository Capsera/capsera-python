"""Tests for the LiteLLM Proxy drop-in handler (capsera.gateways.litellm_proxy).

The module is designed to be referenced from a proxy config.yaml, so importing
it IS the installation: it self-initialises from env vars, registers the
capture callback, and exposes ``capsera_handler`` with the pre-call budget
hook. These tests import it fresh against a fake litellm.
"""
from __future__ import annotations

import importlib
import sys
from datetime import UTC
from types import ModuleType, SimpleNamespace

import pytest

from capsera._models import BudgetExceededError

# importlib: the package attribute `capsera._interceptor` is shadowed by the
# Interceptor-instance singleton in capsera/__init__.
_runtime = importlib.import_module("capsera._interceptor")


@pytest.fixture(autouse=True)
def _restore_gateway_state():
    from capsera.gateways import _litellm as gl

    saved = set(_runtime._GATEWAY_SUPPRESS_PACKAGES)
    yield
    _runtime._GATEWAY_SUPPRESS_PACKAGES.clear()
    _runtime._GATEWAY_SUPPRESS_PACKAGES.update(saved)
    gl._logger_cls = None
    with gl._snapshots_lock:
        gl._snapshots.clear()
    with gl._emitted_lock:
        gl._emitted.clear()


def _fake_litellm_modules() -> dict[str, ModuleType]:
    litellm = ModuleType("litellm")
    integrations = ModuleType("litellm.integrations")
    custom_logger_mod = ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    custom_logger_mod.CustomLogger = CustomLogger
    integrations.custom_logger = custom_logger_mod
    litellm.integrations = integrations
    litellm.callbacks = []
    return {
        "litellm": litellm,
        "litellm.integrations": integrations,
        "litellm.integrations.custom_logger": custom_logger_mod,
    }


@pytest.fixture
def proxy_module(monkeypatch):
    """Import capsera.gateways.litellm_proxy fresh against a fake litellm and
    an offline Capsera endpoint."""
    monkeypatch.setenv("CAPSERA_API_KEY", "cap-test")
    monkeypatch.setenv("CAPSERA_ENDPOINT", "http://localhost:9999")
    monkeypatch.setenv("CAPSERA_ENFORCE_BUDGETS", "true")

    modules = _fake_litellm_modules()
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    sys.modules.pop("capsera.gateways.litellm_proxy", None)
    try:
        module = importlib.import_module("capsera.gateways.litellm_proxy")
        yield module
    finally:
        sys.modules.pop("capsera.gateways.litellm_proxy", None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        import capsera

        capsera.shutdown()


class _FakeAsyncBudgetClient:
    def __init__(self, decision):
        self.decision = decision
        self.payloads = []

    async def post(self, path, json):
        self.payloads.append((path, json))
        return SimpleNamespace(status_code=200, json=lambda: self.decision)


def _key_dict(**overrides):
    defaults = {"key_alias": "svc-checkout", "team_alias": "payments", "end_user_id": None}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_handler_is_built_and_callback_registered(proxy_module):
    assert proxy_module.capsera_handler is not None
    litellm = sys.modules["litellm"]
    # install_litellm registered the capture callback on the (fake) litellm
    assert len(litellm.callbacks) >= 1
    assert getattr(litellm, "_capsera_patched", False)


async def test_pre_call_hook_blocks_on_budget(proxy_module, monkeypatch):
    handler = proxy_module.capsera_handler
    fake_client = _FakeAsyncBudgetClient(
        {"allowed": False, "action": "block", "message": "monthly cap hit"}
    )
    monkeypatch.setattr(_runtime, "_get_async_budget_client", lambda cfg: fake_client)

    data = {"model": "anthropic/claude-sonnet-4-6", "metadata": {"capsera_agent_id": "writer"}}
    with pytest.raises(Exception) as excinfo:
        await handler.async_pre_call_hook(_key_dict(), None, data, "completion")

    exc = excinfo.value
    # fastapi.HTTPException(429) inside a proxy; BudgetExceededError without fastapi
    assert getattr(exc, "status_code", None) == 429 or isinstance(exc, BudgetExceededError)

    (_, payload), = fake_client.payloads
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["provider"] == "anthropic"
    assert payload["agent_id"] == "writer"          # capsera_* metadata wins
    assert payload["team_id"] == "payments"         # falls back to key alias


async def test_pre_call_hook_allows_and_returns_data(proxy_module, monkeypatch):
    handler = proxy_module.capsera_handler
    monkeypatch.setattr(
        _runtime, "_get_async_budget_client",
        lambda cfg: _FakeAsyncBudgetClient({"allowed": True, "action": "allow"}),
    )

    data = {"model": "gpt-4o", "metadata": {}}
    result = await handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert result is data


async def test_pre_call_hook_fails_open(proxy_module, monkeypatch):
    handler = proxy_module.capsera_handler

    def broken(cfg):
        raise ConnectionError("capsera backend down")

    monkeypatch.setattr(_runtime, "_get_async_budget_client", broken)
    data = {"model": "gpt-4o"}
    result = await handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert result is data  # availability preserved


def _allow(monkeypatch):
    monkeypatch.setattr(
        _runtime, "_get_async_budget_client",
        lambda cfg: _FakeAsyncBudgetClient({"allowed": True, "action": "allow"}),
    )


def _seed_routing(proxy_module, monkeypatch, *, rules, served):
    """Seed the proxy router with backend rules + a fixed served-model set,
    and stub out the async refresh so nothing hits the network."""
    monkeypatch.setattr(proxy_module, "_served_models", served)

    async def _noop_refresh(router, cfg):
        return None

    monkeypatch.setattr(proxy_module, "_refresh_routing", _noop_refresh)
    router = proxy_module._get_route_router()
    assert router is not None
    router.set_rules(rules, source="backend")
    return router


async def test_routing_rewrites_to_served_target(proxy_module, monkeypatch):
    _allow(monkeypatch)
    _seed_routing(
        proxy_module, monkeypatch,
        rules=[{"type": "model_map", "from_model": "gpt-4o",
                "target_model": "gpt-4o-mini", "name": "downgrade"}],
        served={"gpt-4o", "gpt-4o-mini"},
    )
    data = {"model": "gpt-4o", "metadata": {}}
    result = await proxy_module.capsera_handler.async_pre_call_hook(
        _key_dict(), None, data, "completion"
    )
    assert result is data
    assert data["model"] == "gpt-4o-mini"                       # rewritten
    assert data["metadata"]["capsera_original_model"] == "gpt-4o"  # stashed for capture


async def test_routing_skips_when_target_not_served(proxy_module, monkeypatch):
    """The safety guard: never rewrite to a model the proxy does not serve."""
    _allow(monkeypatch)
    _seed_routing(
        proxy_module, monkeypatch,
        rules=[{"type": "model_map", "from_model": "gpt-4o",
                "target_model": "gpt-4o-mini", "name": "downgrade"}],
        served={"gpt-4o"},  # target absent from the proxy's model_list
    )
    data = {"model": "gpt-4o", "metadata": {}}
    await proxy_module.capsera_handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert data["model"] == "gpt-4o"                       # untouched
    assert "capsera_original_model" not in data["metadata"]


async def test_routing_skips_when_model_list_undiscoverable(proxy_module, monkeypatch):
    """Cannot prove what the proxy serves → do not route (fail-safe)."""
    _allow(monkeypatch)
    _seed_routing(
        proxy_module, monkeypatch,
        rules=[{"type": "model_map", "from_model": "gpt-4o",
                "target_model": "gpt-4o-mini", "name": "downgrade"}],
        served=None,
    )
    data = {"model": "gpt-4o", "metadata": {}}
    await proxy_module.capsera_handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert data["model"] == "gpt-4o"


async def test_routing_skips_embeddings(proxy_module, monkeypatch):
    _allow(monkeypatch)
    _seed_routing(
        proxy_module, monkeypatch,
        rules=[{"type": "model_map", "from_model": "text-embedding-3-large",
                "target_model": "text-embedding-3-small", "name": "dg"}],
        served={"text-embedding-3-large", "text-embedding-3-small"},
    )
    data = {"model": "text-embedding-3-large", "metadata": {}}
    await proxy_module.capsera_handler.async_pre_call_hook(_key_dict(), None, data, "embedding")
    assert data["model"] == "text-embedding-3-large"


async def test_routing_disabled_by_env(proxy_module, monkeypatch):
    monkeypatch.setenv("CAPSERA_ROUTE_AT_GATEWAY", "false")
    _allow(monkeypatch)
    _seed_routing(
        proxy_module, monkeypatch,
        rules=[{"type": "model_map", "from_model": "gpt-4o",
                "target_model": "gpt-4o-mini", "name": "downgrade"}],
        served={"gpt-4o", "gpt-4o-mini"},
    )
    data = {"model": "gpt-4o", "metadata": {}}
    await proxy_module.capsera_handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert data["model"] == "gpt-4o"


async def test_block_precedes_routing(proxy_module, monkeypatch):
    """A blocked request is rejected before any model rewrite is attempted."""
    monkeypatch.setattr(
        _runtime, "_get_async_budget_client",
        lambda cfg: _FakeAsyncBudgetClient({"allowed": False, "message": "cap hit"}),
    )
    _seed_routing(
        proxy_module, monkeypatch,
        rules=[{"type": "model_map", "from_model": "gpt-4o",
                "target_model": "gpt-4o-mini", "name": "downgrade"}],
        served={"gpt-4o", "gpt-4o-mini"},
    )
    data = {"model": "gpt-4o", "metadata": {}}
    with pytest.raises(Exception):
        await proxy_module.capsera_handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert data["model"] == "gpt-4o"  # never rewritten


async def test_routing_failure_is_fail_open(proxy_module, monkeypatch):
    """A routing bug must never fail the request."""
    _allow(monkeypatch)
    monkeypatch.setattr(proxy_module, "_served_models", {"gpt-4o", "gpt-4o-mini"})

    async def _noop_refresh(router, cfg):
        return None

    monkeypatch.setattr(proxy_module, "_refresh_routing", _noop_refresh)

    def _boom(*a, **k):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(proxy_module, "_get_route_router", _boom)
    data = {"model": "gpt-4o", "metadata": {}}
    result = await proxy_module.capsera_handler.async_pre_call_hook(_key_dict(), None, data, "completion")
    assert result is data
    assert data["model"] == "gpt-4o"


class _FakeModelsClient:
    """Stand-in for httpx.AsyncClient returning a canned /v1/models payload."""

    def __init__(self, status=200, data=None, *a, **k):
        self._status = status
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        payload = {"object": "list", "data": self._data} if self._data is not None else {}
        return SimpleNamespace(status_code=self._status, json=lambda: payload)


async def test_served_models_via_http_parses_v1_models(proxy_module, monkeypatch):
    import httpx

    data = [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"no_id": 1}]
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeModelsClient(200, data))
    names = await proxy_module._served_models_via_http(SimpleNamespace(budget_check_timeout=1.0))
    assert names == {"gpt-4o", "gpt-4o-mini"}   # malformed entry skipped


async def test_served_models_via_http_non_200_returns_none(proxy_module, monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeModelsClient(503, None))
    assert await proxy_module._served_models_via_http(SimpleNamespace(budget_check_timeout=1.0)) is None


async def test_refresh_prefers_http_over_internal(proxy_module, monkeypatch):
    async def _http(cfg):
        return {"from-http"}

    monkeypatch.setattr(proxy_module, "_served_models_via_http", _http)
    monkeypatch.setattr(proxy_module, "_served_models_via_internal", lambda: {"from-internal"})
    monkeypatch.setattr(proxy_module, "_served_models", None)
    await proxy_module._refresh_served_models(object())
    assert proxy_module._served_models == {"from-http"}


async def test_refresh_falls_back_to_internal_when_http_fails(proxy_module, monkeypatch):
    async def _http(cfg):
        return None

    monkeypatch.setattr(proxy_module, "_served_models_via_http", _http)
    monkeypatch.setattr(proxy_module, "_served_models_via_internal", lambda: {"from-internal"})
    monkeypatch.setattr(proxy_module, "_served_models", None)
    await proxy_module._refresh_served_models(object())
    assert proxy_module._served_models == {"from-internal"}


async def test_refresh_keeps_last_good_when_both_fail(proxy_module, monkeypatch):
    async def _http(cfg):
        return None

    monkeypatch.setattr(proxy_module, "_served_models_via_http", _http)
    monkeypatch.setattr(proxy_module, "_served_models_via_internal", lambda: None)
    monkeypatch.setattr(proxy_module, "_served_models", {"kept"})
    await proxy_module._refresh_served_models(object())
    assert proxy_module._served_models == {"kept"}   # not blanked by a transient failure


def test_handler_inherits_capture(proxy_module):
    """The proxy handler IS the litellm logger — success events flow to the
    active emitter."""
    from datetime import datetime, timedelta

    from capsera._config import Config
    from capsera._interceptor import Interceptor

    from .conftest import CapturingEmitter

    config = Config(api_key="t", endpoint="http://localhost:9999",
                    env="test", flush_interval_ms=99_999)
    emitter = CapturingEmitter(config)
    emitter.start()
    try:
        Interceptor(config, emitter)._publish_runtime()
        start = datetime.now(UTC)
        kwargs = {
            "model": "anthropic/claude-sonnet-4-6",
            "litellm_call_id": "proxy-call-1",
            "litellm_params": {
                "custom_llm_provider": "anthropic",
                "metadata": {"user_api_key_alias": "svc-checkout"},
            },
            "response_cost": 0.5,
        }
        response = SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                  prompt_tokens_details=None),
        )
        proxy_module.capsera_handler.log_success_event(
            kwargs, response, start, start + timedelta(milliseconds=90)
        )
        (ev,) = emitter.drain()
        assert ev.gateway == "litellm"
        assert ev.agent_id == "svc-checkout"
        assert ev.cost_usd == 0.5
    finally:
        emitter.shutdown()
