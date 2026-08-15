"""Tests for the LiteLLM gateway integration.

Same approach as the native-provider tests: we cannot run the real litellm
here, so a fake module mirroring the surfaces the integration touches
(``litellm.callbacks``, ``litellm.integrations.custom_logger.CustomLogger``,
the ``completion``/``acompletion`` entry points) is installed into
sys.modules, and the tests drive the callback exactly the way litellm does —
including from a foreign thread, which is how litellm runs sync success
callbacks in production.
"""
from __future__ import annotations

import importlib
import sys
import threading
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor
from capsera._models import BudgetExceededError
from capsera.gateways import _litellm as gl

from .conftest import CapturingEmitter

# importlib, not `import capsera._interceptor as _runtime`: the package
# attribute is shadowed by the Interceptor-instance singleton in
# capsera/__init__ (see tests/validation/test_resilience.py for the same note).
_runtime = importlib.import_module("capsera._interceptor")


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


@pytest.fixture(autouse=True)
def _reset_gateway_state():
    """Roll back every piece of module state a test can leave behind: the
    cached logger class (built against THIS test's fake CustomLogger), the
    snapshot/dedup maps, and the interceptor's suppression registry."""
    saved_suppress = set(_runtime._GATEWAY_SUPPRESS_PACKAGES)
    yield
    _runtime._GATEWAY_SUPPRESS_PACKAGES.clear()
    _runtime._GATEWAY_SUPPRESS_PACKAGES.update(saved_suppress)
    gl._logger_cls = None
    with gl._snapshots_lock:
        gl._snapshots.clear()
    with gl._emitted_lock:
        gl._emitted.clear()


def _fake_litellm_modules() -> dict[str, ModuleType]:
    litellm = ModuleType("litellm")
    integrations = ModuleType("litellm.integrations")
    custom_logger_mod = ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:  # mirrors litellm's base: hooks are optional no-ops
        pass

    custom_logger_mod.CustomLogger = CustomLogger
    integrations.custom_logger = custom_logger_mod
    litellm.integrations = integrations
    litellm.callbacks = []
    litellm.completion = lambda **kwargs: SimpleNamespace(model=kwargs.get("model"))
    litellm.acompletion = None  # exercised in the wrapper test below
    litellm.embedding = lambda **kwargs: SimpleNamespace(model=kwargs.get("model"))
    return {
        "litellm": litellm,
        "litellm.integrations": integrations,
        "litellm.integrations.custom_logger": custom_logger_mod,
    }


@pytest.fixture
def fake_litellm():
    modules = _fake_litellm_modules()
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    yield modules["litellm"]
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _install(config: Config, emitter: CapturingEmitter) -> None:
    Interceptor(config, emitter)._patch_litellm()


def _success_kwargs(**overrides):
    kwargs = {
        "model": "anthropic/claude-sonnet-4-6",
        "litellm_call_id": "call-123",
        "litellm_params": {"custom_llm_provider": "anthropic", "metadata": {}},
        "response_cost": 0.0123,
    }
    kwargs.update(overrides)
    return kwargs


def _success_response(model="claude-sonnet-4-6", prompt=1000, completion=500):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=None,
        ),
    )


def _times():
    start = datetime.now(UTC)
    return start, start + timedelta(milliseconds=250)


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #

def test_install_registers_callback_once(config, emitter, fake_litellm):
    _install(config, emitter)
    _install(config, emitter)  # idempotent

    assert len(fake_litellm.callbacks) == 1
    assert getattr(fake_litellm, "_capsera_patched", False)
    assert "litellm" in _runtime._GATEWAY_SUPPRESS_PACKAGES
    # Entry points got the budget-preflight wrapper
    assert getattr(fake_litellm.completion, "_capsera_patched", False)
    assert getattr(fake_litellm.embedding, "_capsera_patched", False)


def test_missing_litellm_is_silent(config, emitter, monkeypatch):
    # A None sys.modules entry makes `import litellm` raise ImportError even
    # when the real package exists in the environment.
    monkeypatch.setitem(sys.modules, "litellm", None)
    assert gl.install_litellm() is False


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

def test_success_event_capture(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    kwargs = _success_kwargs()
    logger_inst.log_pre_api_call("anthropic/claude-sonnet-4-6", [], kwargs)
    logger_inst.log_success_event(kwargs, _success_response(), start, end)

    (ev,) = emitter.drain()
    assert ev.gateway == "litellm"
    assert ev.provider == "anthropic"
    assert ev.model == "claude-sonnet-4-6"          # vendor prefix stripped
    assert ev.input_tokens == 1000 and ev.output_tokens == 500
    assert ev.cost_usd == 0.0123                    # litellm's own figure wins
    assert ev.latency_ms == 250
    assert ev.is_error is False


def test_cost_falls_back_to_pricing_table(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    kwargs = _success_kwargs(response_cost=None)
    logger_inst.log_success_event(kwargs, _success_response(), start, end)

    (ev,) = emitter.drain()
    # claude-sonnet-4-6: 1000 in × $0.003/1k + 500 out × $0.015/1k = $0.0105 —
    # proves the stripped model ID hit the real pricing row, not the fallback.
    assert ev.cost_usd == pytest.approx(0.0105)


def test_streaming_uses_complete_streaming_response(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    kwargs = _success_kwargs(
        complete_streaming_response=_success_response(prompt=800, completion=300),
        response_cost=None,
    )
    # The raw stream wrapper litellm hands over has no usage of its own.
    logger_inst.log_success_event(kwargs, SimpleNamespace(), start, end)

    (ev,) = emitter.drain()
    assert ev.input_tokens == 800 and ev.output_tokens == 300


def test_cache_tokens_extracted(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    response = SimpleNamespace(
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=100,
            cache_read_input_tokens=600,
            cache_creation_input_tokens=200,
            prompt_tokens_details=None,
        ),
    )
    logger_inst.log_success_event(_success_kwargs(), response, start, end)

    (ev,) = emitter.drain()
    assert ev.cache_read_tokens == 600
    assert ev.cache_write_tokens == 200


def test_cross_thread_attribution_via_snapshot(config, emitter, fake_litellm):
    """litellm runs sync success callbacks on a worker thread, where the
    caller's contextvars are invisible — the pre-call snapshot must carry the
    agent context across."""
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()
    kwargs = _success_kwargs()

    import capsera

    with capsera.tag(agent="planner", team="core", customer_id="acme"):
        logger_inst.log_pre_api_call("m", [], kwargs)

    worker = threading.Thread(
        target=logger_inst.log_success_event,
        args=(kwargs, _success_response(), start, end),
    )
    worker.start()
    worker.join()

    (ev,) = emitter.drain()
    assert ev.agent_id == "planner"
    assert ev.team_id == "core"
    assert ev.customer_id == "acme"


def test_metadata_attribution_fallback(config, emitter, fake_litellm):
    """No capsera context at all (e.g. events logged by a LiteLLM proxy):
    capsera_* metadata keys and proxy key aliases attribute the event."""
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    kwargs = _success_kwargs()
    kwargs["litellm_params"]["metadata"] = {
        "capsera_agent_id": "checkout-service",
        "user_api_key_team_alias": "payments",
        "user_api_key_end_user_id": "cust-42",
    }
    logger_inst.log_success_event(kwargs, _success_response(), start, end)

    (ev,) = emitter.drain()
    assert ev.agent_id == "checkout-service"
    assert ev.team_id == "payments"
    assert ev.customer_id == "cust-42"


def test_routing_attribution_from_metadata(config, emitter, fake_litellm):
    """When the proxy hook downgraded the model, it stashes the pre-rewrite
    model in metadata; capture must record was_routed / original_model so the
    routing-stats dashboard sees a gateway-routed call the same as an SDK one."""
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    # The dispatched (routed) model is gpt-4o-mini; original was gpt-4o.
    kwargs = _success_kwargs(model="gpt-4o-mini")
    kwargs["litellm_params"] = {
        "custom_llm_provider": "openai",
        "metadata": {"capsera_original_model": "gpt-4o"},
    }
    logger_inst.log_success_event(
        kwargs, _success_response(model="gpt-4o-mini"), start, end
    )

    (ev,) = emitter.drain()
    assert ev.model == "gpt-4o-mini"
    assert ev.was_routed is True
    assert ev.original_model == "gpt-4o"


def test_no_routing_attribution_without_metadata(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    logger_inst.log_success_event(_success_kwargs(), _success_response(), start, end)

    (ev,) = emitter.drain()
    assert ev.was_routed is False
    assert ev.original_model is None


def test_failure_event(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    kwargs = _success_kwargs(exception=ValueError("boom"))
    logger_inst.log_failure_event(kwargs, None, start, end)

    (ev,) = emitter.drain()
    assert ev.is_error is True
    assert ev.error_type == "ValueError"
    assert ev.cost_usd == 0.0
    assert ev.gateway == "litellm"


def test_double_fire_dedup(config, emitter, fake_litellm):
    """Same call id reported twice (sync + async handler overlap in some
    litellm versions) must produce exactly one event."""
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()
    kwargs = _success_kwargs()

    logger_inst.log_success_event(kwargs, _success_response(), start, end)
    logger_inst.log_success_event(kwargs, _success_response(), start, end)

    assert len(emitter.drain()) == 1


def test_embedding_call_type_tagged(config, emitter, fake_litellm):
    _install(config, emitter)
    logger_inst = fake_litellm.callbacks[0]
    start, end = _times()

    kwargs = _success_kwargs(
        model="text-embedding-3-small",
        call_type="embedding",
        response_cost=None,
        litellm_params={"custom_llm_provider": "openai", "metadata": {}},
    )
    response = SimpleNamespace(
        model="text-embedding-3-small",
        usage=SimpleNamespace(prompt_tokens=512, completion_tokens=0,
                              prompt_tokens_details=None),
    )
    logger_inst.log_success_event(kwargs, response, start, end)

    (ev,) = emitter.drain()
    assert ev.task_type == "embedding"
    assert ev.provider == "openai"
    assert ev.input_tokens == 512


# --------------------------------------------------------------------------- #
# Entry wrappers: metadata injection + budget preflight
# --------------------------------------------------------------------------- #

class _FakeBudgetClient:
    def __init__(self, decision):
        self.decision = decision
        self.payloads = []

    def post(self, path, json):
        self.payloads.append((path, json))
        return SimpleNamespace(status_code=200, json=lambda: self.decision)


def test_entry_wrapper_blocks_on_budget(config, emitter, fake_litellm, monkeypatch):
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    fake_litellm.completion = original  # recorder must exist BEFORE install wraps it
    _install(config, emitter)

    blocked = {"allowed": False, "action": "block", "message": "budget exhausted"}
    fake_client = _FakeBudgetClient(blocked)
    monkeypatch.setattr(_runtime, "_get_sync_budget_client", lambda cfg: fake_client)

    with pytest.raises(BudgetExceededError):
        fake_litellm.completion(model="anthropic/claude-sonnet-4-6", messages=[])

    # The provider payload was derived from the litellm-prefixed model.
    (_, payload), = fake_client.payloads
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["provider"] == "anthropic"
    assert calls == []  # original never ran — spend prevented before dispatch


def test_entry_wrapper_injects_capsera_metadata(config, emitter, fake_litellm, monkeypatch):
    _install(config, emitter)
    monkeypatch.setattr(
        _runtime, "_get_sync_budget_client",
        lambda cfg: _FakeBudgetClient({"allowed": True, "action": "allow"}),
    )
    seen = {}

    def original(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    # Re-wrap a fresh original so we can observe the kwargs it receives.
    fake_litellm.completion = gl._wrap_entry_sync(original)

    import capsera

    with capsera.tag(agent="researcher", team="ml", session_id="s-1"):
        fake_litellm.completion(model="gpt-4o", messages=[])

    meta = seen["metadata"]
    assert meta["capsera_agent_id"] == "researcher"
    assert meta["capsera_team_id"] == "ml"
    assert meta["capsera_session_id"] == "s-1"


def test_entry_wrapper_fails_open_when_backend_down(config, emitter, fake_litellm, monkeypatch):
    _install(config, emitter)

    def broken_client(cfg):
        raise ConnectionError("backend unreachable")

    monkeypatch.setattr(_runtime, "_get_sync_budget_client", broken_client)
    result = fake_litellm.completion(model="gpt-4o", messages=[])
    assert result is not None  # call passed through


# --------------------------------------------------------------------------- #
# Double-count suppression on the inner provider patch
# --------------------------------------------------------------------------- #

def _fake_openai_namespace():
    class Completions:
        _client = SimpleNamespace(base_url="https://api.openai.com/v1")

        def create(self, **kwargs):
            return SimpleNamespace(
                model="gpt-4o",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    return SimpleNamespace(
        resources=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(Completions=Completions)
            )
        )
    ), Completions


_LITELLM_SHIM_SRC = "def call_via_litellm(fn):\n    return fn()\n"


def _litellm_shim():
    """A callable whose frame claims to live inside site-packages/litellm —
    exactly what the stack looks like when litellm drives the openai SDK."""
    code = compile(_LITELLM_SHIM_SRC, "/x/site-packages/litellm/llms/openai_like.py", "exec")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 — test-only shim
    return ns["call_via_litellm"]


def test_inner_openai_call_suppressed_when_litellm_active(config, emitter, fake_litellm):
    interceptor = Interceptor(config, emitter)
    interceptor._patch_litellm()

    fake_openai, completions_cls = _fake_openai_namespace()
    interceptor._patch_openai_sync(fake_openai)

    client = completions_cls()

    # Call arriving through litellm internals: suppressed (litellm's own
    # callback records it).
    _litellm_shim()(lambda: client.create(model="gpt-4o", messages=[]))
    assert emitter.drain() == []

    # Direct user call: captured as usual.
    client.create(model="gpt-4o", messages=[])
    (ev,) = emitter.drain()
    assert ev.model == "gpt-4o"


def test_inner_openai_call_captured_when_litellm_not_installed(config, emitter):
    """Without the litellm integration active there is no suppression —
    a call from a litellm-looking frame still records (partial capture
    beats silence when the callback is not registered)."""
    interceptor = Interceptor(config, emitter)
    fake_openai, completions_cls = _fake_openai_namespace()
    interceptor._patch_openai_sync(fake_openai)

    _litellm_shim()(lambda: completions_cls().create(model="gpt-4o", messages=[]))
    assert len(emitter.drain()) == 1
