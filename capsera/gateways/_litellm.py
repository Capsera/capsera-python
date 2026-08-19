"""LiteLLM integration — capture and budget enforcement for gateway'd calls.

LiteLLM routes calls to 100+ providers through its own HTTP clients, so the
provider monkey-patches in ``_interceptor`` either miss the call entirely
(Anthropic, Gemini ride litellm's raw httpx handlers) or would double-count it
(OpenAI-compatible providers ride the very ``openai`` SDK we patch). This
module closes both gaps:

1. **Capture** — a ``CustomLogger`` callback registered in
   ``litellm.callbacks``. LiteLLM invokes it for sync + async + streaming
   completions and embeddings, with its own computed ``response_cost`` (which
   prices every provider litellm supports, including ones absent from
   Capsera's table). One event per call, tagged ``gateway="litellm"``.
2. **Budget enforcement** — thin wrappers around ``litellm.completion`` /
   ``acompletion`` / ``embedding`` / ``aembedding`` that run Capsera's
   pre-call budget check (block only; margin downgrades are not applied at
   the litellm layer because model rewrites there need provider-prefix
   awareness). Fail-open like every other enforcement path.
3. **Double-count suppression** — on successful install, "litellm" is
   registered with the interceptor so inner provider-SDK calls made *by*
   litellm are passed through unrecorded (the callback records them).

Threading note: litellm runs sync success callbacks on a worker thread, where
Capsera's contextvars-based agent context is invisible. Attribution therefore
rides two rails: ``log_pre_api_call`` snapshots the caller's context keyed by
``litellm_call_id`` (popped in the success/failure handler), and the entry
wrappers additionally inject ``capsera_*`` keys into litellm ``metadata``,
which litellm threads through to the callback regardless of thread — and
which also survives the network hop to a LiteLLM *proxy*.
"""
from __future__ import annotations

import functools
import importlib
import logging
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .._catalog import provider_for, strip_vendor_prefix
from .._context import AgentContext, _context_stack
from .._models import BudgetExceededError, TokenEvent
from .._pricing import compute_cost
from .._usage import cached_is_included, split_cached_input

# The interceptor runtime MODULE (config/emitter globals are republished on
# re-init, so they must be read as module attributes at call time). Resolved
# via importlib because ``capsera._interceptor`` as a *package attribute* is
# shadowed by the Interceptor-instance singleton in capsera/__init__.
_runtime = importlib.import_module("capsera._interceptor")

logger = logging.getLogger("capsera")

GATEWAY_NAME = "litellm"

# litellm ``custom_llm_provider`` values → Capsera canonical provider names.
# Values not listed pass through unchanged (litellm's naming mostly matches).
_PROVIDER_NORMALIZE = {
    "vertex_ai": "vertex",
    "vertex_ai_beta": "vertex",
    "gemini": "google",
    "azure_ai": "azure",
    "together_ai": "together",
    "fireworks_ai": "fireworks",
    "text-completion-openai": "openai",
}

# --------------------------------------------------------------------------- #
# Context snapshots + emit dedup, keyed by litellm_call_id
# --------------------------------------------------------------------------- #
# Bounded FIFO maps: entries for calls that never reach a post-call handler
# (or litellm versions that skip pre-call hooks) age out by insertion order
# instead of leaking.
_SNAPSHOT_MAX = 4096
_snapshots: dict[str, tuple[AgentContext | None, str | None, str]] = {}
_snapshots_lock = threading.Lock()

_EMITTED_MAX = 4096
_emitted: dict[tuple[str, bool], None] = {}
_emitted_lock = threading.Lock()

_logger_cls: type | None = None


def _call_id(kwargs: dict[str, Any]) -> str | None:
    cid = kwargs.get("litellm_call_id")
    if not cid:
        params = kwargs.get("litellm_params")
        if isinstance(params, dict):
            cid = params.get("litellm_call_id")
    return str(cid) if cid else None


def snapshot_context(kwargs: dict[str, Any]) -> None:
    """Record the calling thread's agent context for later attribution.

    Runs inline in the caller's thread/task (litellm pre-call hooks are
    synchronous with the request), which is the only point where the
    contextvars stack is guaranteed visible.
    """
    try:
        call_id = _call_id(kwargs or {})
        if not call_id:
            return
        snap = (
            _context_stack.current(),
            _context_stack.get_session(),
            _runtime._caller_location(stack_start=2),
        )
        with _snapshots_lock:
            while len(_snapshots) >= _SNAPSHOT_MAX:
                _snapshots.pop(next(iter(_snapshots)), None)
            _snapshots[call_id] = snap
    except Exception:  # noqa: BLE001 — never raise into litellm
        pass


def _pop_snapshot(
    call_id: str | None,
) -> tuple[AgentContext | None, str | None, str] | None:
    if not call_id:
        return None
    with _snapshots_lock:
        return _snapshots.pop(call_id, None)


def _mark_if_new(call_id: str | None, is_error: bool) -> bool:
    """Atomically claim (call_id, is_error) for emission.

    Returns False when this exact outcome was already emitted — insurance
    against litellm versions that invoke both the sync and async handler for
    one call. A failure followed by a retry's success are distinct outcomes
    and both emit. Calls without an id always emit.
    """
    if not call_id:
        return True
    key = (call_id, is_error)
    with _emitted_lock:
        if key in _emitted:
            return False
        while len(_emitted) >= _EMITTED_MAX:
            _emitted.pop(next(iter(_emitted)), None)
        _emitted[key] = None
        return True


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #

def _uget(obj: Any, name: str) -> Any:
    """Attribute-or-key lookup — litellm objects are pydantic-ish, but proxy
    payloads and older versions hand plain dicts through the same hooks."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _latency_ms(start_time: Any, end_time: Any) -> int:
    try:
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            return max(int((end_time - start_time).total_seconds() * 1000), 0)
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            return max(int((end_time - start_time) * 1000), 0)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _litellm_cost(kwargs: dict[str, Any]) -> float | None:
    """LiteLLM's own USD figure for the call, when it published one."""
    for candidate in (
        kwargs.get("response_cost"),
        _uget(kwargs.get("standard_logging_object"), "response_cost"),
        _uget(kwargs.get("litellm_params"), "response_cost"),
    ):
        if candidate is not None:
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
    return None


def _resolve_provider(litellm_params: dict[str, Any], raw_model: str, model: str) -> str:
    declared = str(litellm_params.get("custom_llm_provider") or "").lower()
    if declared:
        return _PROVIDER_NORMALIZE.get(declared, declared)
    resolved = provider_for(model)
    if resolved != "unknown":
        return resolved
    head, sep, _ = raw_model.partition("/")
    if sep and head:
        return _PROVIDER_NORMALIZE.get(head.lower(), head.lower())
    return "unknown"


def emit_event(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    *,
    is_error: bool = False,
) -> None:
    """Map a litellm success/failure callback payload to a TokenEvent.

    Resolves the active config/emitter from the interceptor runtime at call
    time (same re-init semantics as every patched provider closure). Never
    raises into litellm.
    """
    try:
        cfg = _runtime.config
        emitter = _runtime.emitter
        if cfg is None or emitter is None:
            return
        kwargs = kwargs or {}
        call_id = _call_id(kwargs)
        if not _mark_if_new(call_id, is_error):
            return
        snap = _pop_snapshot(call_id)

        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
        metadata = litellm_params.get("metadata") or kwargs.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        raw_model = str(kwargs.get("model") or "")

        # Streaming callbacks carry the aggregated response beside the raw
        # stream object — prefer whichever source actually has usage.
        model = ""
        usage = None
        for candidate in (kwargs.get("complete_streaming_response"), response_obj):
            if candidate is None:
                continue
            if usage is None:
                usage = _uget(candidate, "usage")
            if not model:
                found = _uget(candidate, "model")
                if found:
                    model = str(found)
        model = strip_vendor_prefix(model or raw_model) or "unknown"

        in_tok = out_tok = cache_read = cache_write = 0
        anthropic_shaped = False
        if usage is not None and not is_error:
            in_tok = int(_uget(usage, "prompt_tokens") or _uget(usage, "input_tokens") or 0)
            out_tok = int(_uget(usage, "completion_tokens") or _uget(usage, "output_tokens") or 0)
            # litellm passes the upstream usage object through, so which field
            # is present tells us which shape we are looking at — and that
            # decides whether the cached count is already inside in_tok.
            cache_read = int(_uget(usage, "cache_read_input_tokens") or 0)
            anthropic_shaped = cache_read > 0
            if not cache_read:
                details = _uget(usage, "prompt_tokens_details")
                cache_read = int(_uget(details, "cached_tokens") or 0)
            cache_write = int(_uget(usage, "cache_creation_input_tokens") or 0)
            anthropic_shaped = anthropic_shaped or cache_write > 0

        provider = _resolve_provider(litellm_params, raw_model, model)

        # Without this, a cached OpenAI-shaped call bills its cached tokens
        # twice in the fallback cost path: once inside prompt_tokens at the
        # input rate, once beside it at the cache rate. Anthropic-shaped usage
        # reports the buckets disjointly and must be left alone.
        in_tok, cache_read = split_cached_input(
            in_tok,
            cache_read,
            included=not anthropic_shaped and cached_is_included(provider),
        )

        if is_error:
            cost = 0.0
        else:
            litellm_cost = _litellm_cost(kwargs)
            if litellm_cost is not None:
                cost = litellm_cost
            else:
                cost = float(compute_cost(
                    model=model,
                    provider=provider,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                ))

        # Attribution: pre-call snapshot (thread-proof) > current context
        # (async same-task) > capsera_* metadata (wrapper-injected, also the
        # only rail that survives the hop to a litellm PROXY) > proxy key
        # aliases > config defaults.
        ctx = (snap[0] if snap else None) or _context_stack.current()
        session = (snap[1] if snap else None) or _context_stack.get_session()
        caller_file = snap[2] if snap else "unknown"

        agent_id = (
            (ctx.agent if ctx else None)
            or metadata.get("capsera_agent_id")
            or metadata.get("user_api_key_alias")
            or cfg.default_agent
            or "unknown"
        )
        team_id = (
            (ctx.team if ctx else None)
            or metadata.get("capsera_team_id")
            or metadata.get("user_api_key_team_alias")
            or cfg.default_team
        )
        task_type = (ctx.task_type if ctx else None) or metadata.get("capsera_task_type")
        if not task_type and str(kwargs.get("call_type") or "") in ("embedding", "aembedding"):
            task_type = "embedding"
        session_id = (
            (ctx.session_id if ctx else None)
            or metadata.get("capsera_session_id")
            or session
        )
        customer_id = (
            (ctx.customer_id if ctx else None)
            or metadata.get("capsera_customer_id")
            or metadata.get("user_api_key_end_user_id")
        )
        cost_center = (ctx.cost_center if ctx else None) or metadata.get("capsera_cost_center")

        # Routing attribution. The proxy pre-call hook stashes the pre-rewrite
        # model here when it downgrades a call to a cheaper served model, so a
        # gateway-routed event carries the same was_routed/original_model the
        # in-process SDK router records — feeding the routing-stats dashboard.
        routed_from = metadata.get("capsera_original_model")
        original_model = strip_vendor_prefix(str(routed_from)) if routed_from else None
        was_routed = bool(original_model) and original_model != model

        error_type: str | None = None
        if is_error:
            exc = kwargs.get("exception")
            if isinstance(exc, BaseException):
                error_type = type(exc).__name__
            elif exc is not None:
                error_type = str(exc)

        event = TokenEvent(
            id=str(uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            request_id=str(uuid4()),
            model=model,
            provider=provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=max(float(cost), 0.0),
            agent_id=str(agent_id),
            team_id=str(team_id) if team_id else None,
            task_type=str(task_type) if task_type else None,
            session_id=str(session_id) if session_id else None,
            customer_id=str(customer_id) if customer_id else None,
            cost_center=str(cost_center) if cost_center else None,
            env=cfg.env,
            latency_ms=_latency_ms(start_time, end_time),
            caller_file=caller_file,
            gateway=GATEWAY_NAME,
            was_routed=was_routed,
            original_model=original_model,
            is_error=is_error,
            error_type=error_type,
        )
        emitter.enqueue(event)

        if cfg.debug:
            logger.info(
                "capsera: litellm captured provider=%s model=%s tokens=%d/%d "
                "cost_usd=%s agent=%s error=%s",
                provider, model, in_tok, out_tok, event.cost_usd, agent_id, is_error,
            )
    except Exception:  # noqa: BLE001 — never raise into litellm
        try:
            if _runtime.config is not None and _runtime.config.debug:
                logger.exception("capsera: failed to record litellm event")
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Entry-point wrappers: metadata injection + pre-call budget enforcement
# --------------------------------------------------------------------------- #

def _inject_metadata(kwargs: dict[str, Any]) -> None:
    """Copy the current Capsera context into litellm ``metadata`` under
    ``capsera_*`` keys (never clobbering caller-supplied values)."""
    try:
        ctx = _context_stack.current()
        session = _context_stack.get_session()
        if ctx is None and session is None:
            return
        meta = kwargs.get("metadata")
        if meta is None:
            meta = {}
            kwargs["metadata"] = meta
        elif not isinstance(meta, dict):
            return
        pairs = (
            ("capsera_agent_id", ctx.agent if ctx else None),
            ("capsera_team_id", ctx.team if ctx else None),
            ("capsera_task_type", ctx.task_type if ctx else None),
            ("capsera_customer_id", ctx.customer_id if ctx else None),
            ("capsera_cost_center", ctx.cost_center if ctx else None),
            ("capsera_session_id", (ctx.session_id if ctx else None) or session),
        )
        for key, value in pairs:
            if value and key not in meta:
                meta[key] = value
    except Exception:  # noqa: BLE001
        pass


def _preflight_args(kwargs: dict[str, Any]) -> tuple[dict[str, Any], str]:
    model = strip_vendor_prefix(str(kwargs.get("model") or ""))
    raw = str(kwargs.get("model") or "")
    head, sep, _ = raw.partition("/")
    provider = provider_for(model)
    if provider == "unknown" and sep and head:
        provider = _PROVIDER_NORMALIZE.get(head.lower(), head.lower())
    # Preserve max_tokens so the check's cost estimate matches direct-SDK calls.
    check_kwargs = {"model": model}
    for key in ("max_tokens", "max_completion_tokens"):
        if key in kwargs:
            check_kwargs[key] = kwargs[key]
    return check_kwargs, provider


def _budget_preflight_sync(kwargs: dict[str, Any]) -> None:
    """Capsera pre-call budget check for a litellm entry point.

    Delegates to the interceptor's enforcement rather than re-implementing it,
    so a litellm-routed call gets exactly the decision a direct provider call
    would — including the local engine, which means no round trip here either.

    Margin downgrades still do not take effect at this layer, and now by
    construction rather than by omission: enforcement applies a downgrade by
    rewriting ``model`` on the dict it is handed, and what it is handed is the
    derived ``check_kwargs``, not litellm's real kwargs. The vendor-prefixed
    model string the caller passed is left alone. Blocks propagate; everything
    else fails open.
    """
    cfg = _runtime.config
    if cfg is None:
        return
    try:
        check_kwargs, provider = _preflight_args(kwargs)
        _runtime._enforce_budget(check_kwargs, provider, cfg, None)
    except BudgetExceededError:
        raise
    except Exception:  # noqa: BLE001
        logger.debug("capsera: litellm budget preflight failed — passing through", exc_info=True)


async def _budget_preflight_async(kwargs: dict[str, Any]) -> None:
    """Async counterpart. See ``_budget_preflight_sync``."""
    cfg = _runtime.config
    if cfg is None:
        return
    try:
        check_kwargs, provider = _preflight_args(kwargs)
        await _runtime._enforce_budget_async(check_kwargs, provider, cfg, None)
    except BudgetExceededError:
        raise
    except Exception:  # noqa: BLE001
        logger.debug("capsera: litellm budget preflight failed — passing through", exc_info=True)


def _wrap_entry_sync(original: Any) -> Any:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _inject_metadata(kwargs)
        _budget_preflight_sync(kwargs)
        return original(*args, **kwargs)

    wrapper._capsera_patched = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_entry_async(original: Any) -> Any:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        _inject_metadata(kwargs)
        await _budget_preflight_async(kwargs)
        return await original(*args, **kwargs)

    wrapper._capsera_patched = True  # type: ignore[attr-defined]
    return wrapper


# --------------------------------------------------------------------------- #
# Callback class + installer
# --------------------------------------------------------------------------- #

def build_logger_class() -> type:
    """Create (once) the CustomLogger subclass. Requires litellm importable —
    deferred so this module imports cleanly when litellm is absent."""
    global _logger_cls
    if _logger_cls is not None:
        return _logger_cls

    from litellm.integrations.custom_logger import CustomLogger  # noqa: PLC0415

    class CapseraLiteLLMLogger(CustomLogger):  # type: ignore[misc]
        """Maps litellm callback payloads to Capsera TokenEvents."""

        _capsera_logger = True

        def log_pre_api_call(self, model: Any, messages: Any, kwargs: dict[str, Any]) -> None:
            snapshot_context(kwargs)

        async def async_log_pre_api_call(self, model: Any, messages: Any, kwargs: dict[str, Any]) -> None:
            snapshot_context(kwargs)

        def log_success_event(self, kwargs: dict[str, Any], response_obj: Any,
                              start_time: Any, end_time: Any) -> None:
            emit_event(kwargs, response_obj, start_time, end_time)

        async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any,
                                          start_time: Any, end_time: Any) -> None:
            emit_event(kwargs, response_obj, start_time, end_time)

        def log_failure_event(self, kwargs: dict[str, Any], response_obj: Any,
                              start_time: Any, end_time: Any) -> None:
            emit_event(kwargs, response_obj, start_time, end_time, is_error=True)

        async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: Any,
                                          start_time: Any, end_time: Any) -> None:
            emit_event(kwargs, response_obj, start_time, end_time, is_error=True)

    _logger_cls = CapseraLiteLLMLogger
    return _logger_cls


def install_litellm() -> bool:
    """Register the Capsera callback + entry wrappers with litellm.

    Idempotent (guarded by ``litellm._capsera_patched``). Returns False when
    litellm is absent or registration failed — in that case double-count
    suppression is NOT activated, so inner provider-SDK capture keeps working.
    """
    try:
        import litellm  # noqa: PLC0415
    except ImportError:
        return False
    except Exception:  # noqa: BLE001 — a broken litellm install must not kill init()
        logger.warning("capsera: litellm import failed — litellm calls will not be tracked",
                       exc_info=True)
        return False

    if not getattr(litellm, "_capsera_patched", False):
        try:
            instance = build_logger_class()()
            existing = getattr(litellm, "callbacks", None)
            if isinstance(existing, list):
                existing.append(instance)
            else:
                litellm.callbacks = [instance]
            litellm._capsera_patched = True
            litellm._capsera_logger = instance
        except Exception:  # noqa: BLE001
            logger.warning(
                "capsera: could not register LiteLLM callback — litellm calls will not be tracked",
                exc_info=True,
            )
            return False

        # Budget-enforcement wrappers are best-effort: capture works without
        # them, and code holding a pre-init `from litellm import completion`
        # reference bypasses them (the callback still records those calls).
        try:
            main_mod = getattr(litellm, "main", None)
            for name, wrap in (
                ("completion", _wrap_entry_sync),
                ("embedding", _wrap_entry_sync),
                ("acompletion", _wrap_entry_async),
                ("aembedding", _wrap_entry_async),
            ):
                original = getattr(litellm, name, None)
                if original is None or getattr(original, "_capsera_patched", False):
                    continue
                wrapped = wrap(original)
                setattr(litellm, name, wrapped)
                # Keep litellm.main's alias in lockstep when it points at the
                # same function object (litellm.Router resolves through it).
                if main_mod is not None and getattr(main_mod, name, None) is original:
                    setattr(main_mod, name, wrapped)
        except Exception:  # noqa: BLE001
            logger.debug("capsera: litellm entry wrappers not installed", exc_info=True)

        if _runtime.config is not None and _runtime.config.debug:
            logger.debug("capsera: registered LiteLLM callback + entry wrappers")

    # Callback is live — stop the inner provider patches from double-counting
    # litellm-internal SDK calls.
    _runtime.register_gateway_suppression("litellm")
    return True
