from __future__ import annotations

import importlib
from typing import Any

import pytest

from capsera._config import Config
from capsera._context import AgentContext, _context_stack
from capsera._interceptor import _enforce_budget
from capsera._models import BudgetExceededError, PromptMetadata


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    last_payload: dict[str, Any] | None = None
    response_payload: dict[str, Any] = {
        "allowed": False,
        "action": "block",
        "message": "Budget would be exceeded.",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
        self.__class__.last_payload = json
        return _FakeResponse(self.__class__.response_payload)


def test_enforce_budget_raises_when_backend_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    interceptor = importlib.import_module("capsera._interceptor")

    monkeypatch.setattr(interceptor.httpx, "Client", _FakeClient)
    _FakeClient.response_payload = {
        "allowed": False,
        "action": "block",
        "message": "Budget would be exceeded.",
    }
    config = Config(
        api_key="test",
        endpoint="http://localhost:8000",
        default_agent="agent-1",
        enable_budget_enforcement=True,
    )

    with pytest.raises(BudgetExceededError, match="Budget would be exceeded"):
        _enforce_budget(
            {"model": "claude-sonnet-4-6", "max_tokens": 100},
            "anthropic",
            config,
            PromptMetadata(system_prompt_tokens=50, conversation_tokens=150),
        )

    assert _FakeClient.last_payload is not None
    assert _FakeClient.last_payload["agent_id"] == "agent-1"
    assert _FakeClient.last_payload["estimated_cost_usd"] > 0


def test_enforce_budget_applies_backend_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    interceptor = importlib.import_module("capsera._interceptor")

    monkeypatch.setattr(interceptor.httpx, "Client", _FakeClient)
    _FakeClient.response_payload = {
        "allowed": True,
        "action": "downgrade",
        "target_model": "claude-haiku-4-5",
    }
    config = Config(
        api_key="test",
        endpoint="http://localhost:8000",
        default_agent="agent-1",
        enable_budget_enforcement=True,
    )
    kwargs = {"model": "claude-sonnet-4-6", "max_tokens": 100}

    original = _enforce_budget(kwargs, "anthropic", config, None)

    assert original == "claude-sonnet-4-6"
    assert kwargs["model"] == "claude-haiku-4-5"


def test_enforce_budget_skips_downgrade_for_critical_task(monkeypatch: pytest.MonkeyPatch) -> None:
    interceptor = importlib.import_module("capsera._interceptor")

    monkeypatch.setattr(interceptor.httpx, "Client", _FakeClient)
    _FakeClient.response_payload = {
        "allowed": True,
        "action": "downgrade",
        "target_model": "claude-haiku-4-5",
    }
    config = Config(
        api_key="test",
        endpoint="http://localhost:8000",
        default_agent="agent-1",
        critical_task_types=["reasoning"],
        enable_budget_enforcement=True,
    )
    kwargs = {"model": "claude-sonnet-4-6", "max_tokens": 100}
    _context_stack.push(AgentContext(agent="agent-1", task_type="reasoning", customer_id="customer-a"))
    try:
        original = _enforce_budget(kwargs, "anthropic", config, None)
    finally:
        _context_stack.pop()

    assert original is None
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert _FakeClient.last_payload is not None
    assert _FakeClient.last_payload["customer_id"] == "customer-a"


def test_enforce_budget_disabled_does_not_call_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    interceptor = importlib.import_module("capsera._interceptor")

    def fail_client(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("budget backend should not be called")

    monkeypatch.setattr(interceptor.httpx, "Client", fail_client)
    config = Config(
        api_key="test",
        endpoint="http://localhost:8000",
        enable_budget_enforcement=False,
    )

    _enforce_budget(
        {"model": "claude-sonnet-4-6", "max_tokens": 100},
        "anthropic",
        config,
        None,
    )
