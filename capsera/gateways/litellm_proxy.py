"""Capsera handler for the LiteLLM **Proxy** (AI gateway server).

Gives a LiteLLM proxy deployment org-level Capsera capture AND hard pre-call
budget enforcement at the gateway itself — every service that talks to the
proxy is governed, regardless of language, with zero client changes.

Setup (proxy ``config.yaml``)::

    litellm_settings:
      callbacks: ["capsera.gateways.litellm_proxy.capsera_handler"]

Environment (on the proxy host)::

    CAPSERA_API_KEY=cap-...                       # required
    CAPSERA_ENDPOINT=https://api.capsera.ai       # default shown
    CAPSERA_ENV=production                        # default shown
    CAPSERA_ENFORCE_BUDGETS=true                  # "false" → capture only
    CAPSERA_ROUTE_AT_GATEWAY=true                 # "false" → never rewrite models
    CAPSERA_PROXY_BASE_URL=http://127.0.0.1:4000  # proxy's own URL, for GET /v1/models

And ``pip install capsera`` alongside the proxy.

Attribution through the proxy: callers may put ``capsera_*`` keys in litellm
request ``metadata`` (``capsera_agent_id``, ``capsera_team_id``,
``capsera_task_type``, ``capsera_customer_id``, ``capsera_cost_center``,
``capsera_session_id``) — apps already running the Capsera SDK inject these
automatically. Without them, events fall back to the proxy virtual-key's
``key_alias`` / ``team_alias``, so per-key spend attribution works out of the
box.

Enforcement: ``async_pre_call_hook`` consults Capsera's ``/v1/budgets/check``
before litellm dispatches to a provider. A matching *blocking* budget rejects
the request with HTTP 429 before any provider spend. Fail-open: transport or
backend errors pass the request through.

Routing (closed-loop optimization at the gateway): after the budget check, the
same hook evaluates Capsera's routing rules and may rewrite ``data["model"]``
to a cheaper target — but **only when that target is a public model the proxy
itself serves**. Servability is discovered from the proxy's OpenAI-compatible
``GET /v1/models`` (a stable HTTP contract; the in-process router is only a
fallback). Rewriting to a model the proxy does not know would turn "save money"
into a failed request, so an unservable target, an undiscoverable model list,
or any error leaves the model untouched. The original model is stashed in
``metadata`` so the capture side records ``was_routed`` / ``original_model``
and the routing-stats dashboard can prove the savings. Disable with
``CAPSERA_ROUTE_AT_GATEWAY=false``.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("capsera")


def _ensure_initialized() -> None:
    """Init the SDK from environment variables unless the host app already
    called capsera.init() itself."""
    import capsera  # noqa: PLC0415

    from .._config import _global_config  # noqa: PLC0415

    if _global_config.get() is not None and capsera._emitter is not None:
        return

    api_key = os.environ.get("CAPSERA_API_KEY", "")
    if not api_key:
        logger.warning(
            "capsera: CAPSERA_API_KEY is not set — proxy events cannot be "
            "delivered to the Capsera backend"
        )
    capsera.init(
        api_key=api_key or "dev",
        endpoint=os.environ.get("CAPSERA_ENDPOINT", "https://api.capsera.ai"),
        env=os.environ.get("CAPSERA_ENV", "production"),
        enable_budget_enforcement=(
            os.environ.get("CAPSERA_ENFORCE_BUDGETS", "true").strip().lower() != "false"
        ),
    )


def _blocked_exception(message: str) -> Exception:
    """HTTP 429 for the proxy (fastapi is always present there); a plain
    BudgetExceededError if it somehow is not."""
    try:
        from fastapi import HTTPException  # noqa: PLC0415

        return HTTPException(
            status_code=429,
            detail={"error": "capsera_budget_blocked", "message": message},
        )
    except ImportError:
        from .._models import BudgetExceededError  # noqa: PLC0415

        return BudgetExceededError(message)


async def _budget_decision(user_api_key_dict: Any, data: dict[str, Any]) -> dict[str, Any] | None:
    """POST /v1/budgets/check for a proxy request. Returns the decision dict,
    or None when enforcement is off / unconfigured / the check errored."""
    import importlib  # noqa: PLC0415

    # importlib, not `from .. import`: the package attribute is shadowed by
    # the Interceptor-instance singleton (see _litellm.py).
    _runtime = importlib.import_module("capsera._interceptor")
    from .._catalog import provider_for, strip_vendor_prefix  # noqa: PLC0415

    cfg = _runtime.config
    if cfg is None or not cfg.enable_budget_enforcement:
        return None

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    model = strip_vendor_prefix(str(data.get("model") or ""))
    agent_id = (
        metadata.get("capsera_agent_id")
        or getattr(user_api_key_dict, "key_alias", None)
        or cfg.default_agent
        or "unknown"
    )
    team_id = (
        metadata.get("capsera_team_id")
        or getattr(user_api_key_dict, "team_alias", None)
        or cfg.default_team
    )
    payload = {
        "agent_id": str(agent_id),
        "team_id": str(team_id) if team_id else None,
        "task_type": metadata.get("capsera_task_type"),
        "customer_id": (
            metadata.get("capsera_customer_id")
            or getattr(user_api_key_dict, "end_user_id", None)
        ),
        "cost_center": metadata.get("capsera_cost_center"),
        "model": model or None,
        "provider": provider_for(model),
        "estimated_cost_usd": 0.0,
    }
    client = _runtime._get_async_budget_client(cfg)
    resp = await client.post("/v1/budgets/check", json=payload)
    if resp.status_code == 200:
        return resp.json()
    return None


# --------------------------------------------------------------------------- #
# Gateway routing (closed-loop optimization at the proxy)
# --------------------------------------------------------------------------- #
# One Router instance drives evaluation; refresh is async and TTL-gated so the
# proxy event loop never blocks on the Router's sync HTTP client. The model
# list is read fresh from the live proxy each decision so runtime `/model/new`
# additions are respected.

_route_router: Any = None
_route_router_lock = threading.Lock()
_route_last_refresh: float = 0.0
_route_refresh_lock: asyncio.Lock | None = None
# Cached set of model names the proxy serves — the routing guard's servability
# check. Populated on the routing refresh cycle (see _refresh_served_models),
# so the routing hot path reads it without any I/O.
_served_models: set[str] | None = None


def _routing_enabled() -> bool:
    return os.environ.get("CAPSERA_ROUTE_AT_GATEWAY", "true").strip().lower() != "false"


def _get_route_router() -> Any:
    """The lazily-built routing engine, or None when the SDK is unconfigured."""
    global _route_router
    if _route_router is not None:
        return _route_router
    runtime = importlib.import_module("capsera._interceptor")
    cfg = runtime.config
    if cfg is None:
        return None
    with _route_router_lock:
        if _route_router is None:
            from .._router import Router  # noqa: PLC0415

            _route_router = Router(cfg)
    return _route_router


# Model-list discovery — the models the proxy serves, used by the routing
# guard. Two sources, in order of stability:
#   1. GET /v1/models — the OpenAI-compatible HTTP contract every such proxy
#      exposes. Stable and version-independent: it survives LiteLLM internal
#      refactors that would break an in-process attribute read.
#   2. litellm.proxy.proxy_server.llm_router — an in-process INTERNAL, kept
#      only as a fallback for deployments where (1) is unreachable.
# Either way, discovery failure yields None → the guard treats "unknown" as
# "do not route" (fail-safe).


def _proxy_base_url() -> str:
    """The proxy's own base URL for the /v1/models call. Defaults to LiteLLM's
    documented bind (``127.0.0.1:4000``); override with ``CAPSERA_PROXY_BASE_URL``
    when the proxy listens elsewhere."""
    return os.environ.get("CAPSERA_PROXY_BASE_URL", "http://127.0.0.1:4000").rstrip("/")


async def _served_models_via_http(cfg: Any) -> set[str] | None:
    """Preferred discovery: the proxy's OpenAI-compatible ``GET /v1/models``.

    Auth uses ``LITELLM_MASTER_KEY``/``CAPSERA_PROXY_KEY`` when set (the proxy
    may require a key). Returns None on any non-200 / transport error so the
    caller falls back to the internal read.
    """
    import httpx  # noqa: PLC0415

    headers: dict[str, str] = {}
    key = os.environ.get("LITELLM_MASTER_KEY") or os.environ.get("CAPSERA_PROXY_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timeout = getattr(cfg, "budget_check_timeout", 1.0) or 1.0
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{_proxy_base_url()}/v1/models", headers=headers)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data")
    except Exception:  # noqa: BLE001
        logger.debug("capsera: proxy /v1/models discovery failed", exc_info=True)
        return None
    if not isinstance(data, list):
        return None
    names = {m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)}
    return names or None


def _served_models_via_internal() -> set[str] | None:
    """Fallback discovery: read LiteLLM's in-process router. Internal API —
    used only when the stable HTTP endpoint is unreachable."""
    try:
        import litellm.proxy.proxy_server as ps  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — not inside a proxy, or import side effect
        return None
    router = getattr(ps, "llm_router", None)
    if router is None:
        return None
    names: set[str] = set()
    for name in (getattr(router, "model_names", None) or []):
        if isinstance(name, str):
            names.add(name)
    for entry in (getattr(router, "model_list", None) or []):
        name = entry.get("model_name") if isinstance(entry, dict) else None
        if isinstance(name, str):
            names.add(name)
    return names or None


async def _refresh_served_models(cfg: Any) -> None:
    """Refresh the cached served-model set: stable HTTP endpoint first, the
    in-process router only as a fallback. A transient failure keeps the last
    known-good set rather than blanking it (which would disable routing)."""
    global _served_models
    names = await _served_models_via_http(cfg)
    if names is None:
        names = _served_models_via_internal()
    if names is not None:
        _served_models = names


async def _refresh_routing(router: Any, cfg: Any) -> None:
    """Pull routing rules + budget-status into ``router``, TTL-gated.

    Uses the pooled ASYNC budget client (same base URL + auth as the budget
    check) so refresh never blocks the event loop the way the Router's own
    sync client would. Single-flight via a lock; the timestamp advances before
    the fetch so a slow or failing backend does not stampede.
    """
    global _route_last_refresh, _route_refresh_lock
    interval = getattr(cfg, "routing_refresh_interval", 60) or 60
    if time.monotonic() - _route_last_refresh < interval:
        return
    if _route_refresh_lock is None:
        _route_refresh_lock = asyncio.Lock()
    async with _route_refresh_lock:
        if time.monotonic() - _route_last_refresh < interval:
            return
        _route_last_refresh = time.monotonic()
        runtime = importlib.import_module("capsera._interceptor")
        from .._router import convert_backend_rules, convert_budget_status  # noqa: PLC0415

        try:
            client = runtime._get_async_budget_client(cfg)
        except Exception:  # noqa: BLE001
            logger.debug("capsera: proxy routing refresh — no client", exc_info=True)
            return
        try:
            resp = await client.get("/v1/routing/rules")
            if resp.status_code == 200:
                router.set_rules(convert_backend_rules(resp.json()), source="backend")
        except Exception:  # noqa: BLE001
            logger.debug("capsera: proxy routing-rules refresh failed", exc_info=True)
        try:
            resp = await client.get("/v1/routing/budget-status")
            if resp.status_code == 200:
                router.set_budget_cache(convert_budget_status(resp.json()))
        except Exception:  # noqa: BLE001
            logger.debug("capsera: proxy budget-status refresh failed", exc_info=True)
        # Served-model set shares the routing TTL — refreshed here so the hot
        # path reads it without I/O.
        await _refresh_served_models(cfg)


async def _maybe_route(user_api_key_dict: Any, data: dict[str, Any], call_type: Any) -> None:
    """Rewrite ``data['model']`` to a cheaper served target when a rule matches.

    No-ops (leaving ``data`` untouched) when routing is disabled, the call is
    an embedding, the SDK is unconfigured, no rule matches, the target is not
    a model the proxy serves, or the proxy model list is undiscoverable.
    """
    if not _routing_enabled():
        return
    if isinstance(call_type, str) and "embed" in call_type.lower():
        return
    model = str(data.get("model") or "")
    if not model:
        return
    router = _get_route_router()
    if router is None:
        return
    runtime = importlib.import_module("capsera._interceptor")
    cfg = runtime.config
    if cfg is None:
        return

    await _refresh_routing(router, cfg)

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    from .._context import AgentContext  # noqa: PLC0415

    ctx = AgentContext(
        agent=metadata.get("capsera_agent_id") or getattr(user_api_key_dict, "key_alias", None),
        team=metadata.get("capsera_team_id") or getattr(user_api_key_dict, "team_alias", None),
        task_type=metadata.get("capsera_task_type"),
        customer_id=metadata.get("capsera_customer_id"),
        cost_center=metadata.get("capsera_cost_center"),
    )
    decision = router.evaluate({"model": model}, ctx, refresh=False)
    if not decision.routed or not decision.target_model or decision.target_model == model:
        return

    target = decision.target_model
    served = _served_models  # cached by _refresh_served_models (HTTP-first)
    # Fail-safe: never rewrite to a model the proxy cannot serve, and never
    # rewrite when we could not prove what it serves.
    if served is None or target not in served:
        logger.debug(
            "capsera: proxy routing %s -> %s skipped (target not served / model list unknown)",
            model, target,
        )
        return

    data["model"] = target
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        data["metadata"] = meta
    # The capture side reads this to stamp was_routed / original_model.
    meta.setdefault("capsera_original_model", model)
    if cfg.debug:
        logger.info("capsera: proxy routed %s -> %s (rule '%s')", model, target, decision.rule_name)


def _build_handler_class() -> type:
    from ._litellm import build_logger_class  # noqa: PLC0415

    class CapseraProxyHandler(build_logger_class()):  # type: ignore[misc]
        """Proxy-side handler: inherits full capture from the LiteLLM logger
        and adds hard pre-call budget blocking."""

        async def async_pre_call_hook(  # litellm proxy hook signature
            self,
            user_api_key_dict: Any,
            cache: Any,
            data: dict[str, Any],
            call_type: str,
        ) -> Any:
            payload = data if isinstance(data, dict) else {}
            try:
                decision = await _budget_decision(user_api_key_dict, payload)
            except Exception:  # noqa: BLE001 — fail-open
                logger.debug(
                    "capsera: proxy budget preflight failed — passing through",
                    exc_info=True,
                )
                decision = None
            if decision and not decision.get("allowed", True):
                message = decision.get("message") or "Capsera budget policy blocked this request."
                raise _blocked_exception(message)

            # Routing runs only for a request that cleared the budget gate.
            # Fail-open: a routing error must never fail the request.
            if isinstance(data, dict):
                try:
                    await _maybe_route(user_api_key_dict, data, call_type)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "capsera: proxy routing failed — passing through", exc_info=True
                    )
            return data

    return CapseraProxyHandler


# The instance the proxy config references. None (with a warning) when litellm
# is not importable — which only happens outside a proxy environment.
capsera_handler: Any = None
try:
    import litellm  # noqa: F401  # noqa: PLC0415

    _ensure_initialized()

    # Make sure the callback/suppression side is live even if the host app
    # never called capsera.init() with litellm importable (idempotent).
    from ._litellm import install_litellm  # noqa: PLC0415

    install_litellm()

    capsera_handler = _build_handler_class()()
except ImportError:
    logger.warning(
        "capsera: litellm is not importable — capsera_handler unavailable "
        "(this module is meant to run inside a LiteLLM proxy)"
    )
except Exception:  # noqa: BLE001
    logger.warning("capsera: failed to initialise the LiteLLM proxy handler", exc_info=True)
