"""
Capsera SDK
========================

AI FinOps: per-agent token attribution, budget enforcement, and cost
optimisation for any LLM-powered system.

Quickstart
----------
    import capsera
    capsera.init(api_key="cap-...", endpoint="http://localhost:8000")

    # All Anthropic / OpenAI calls made anywhere in this process are now
    # automatically intercepted and attributed. Budget enforcement is on by
    # default: any blocking budget you create in the dashboard stops calls
    # before provider spend (pass enable_budget_enforcement=False to opt out).

    @capsera.agent(name="my_agent", team="platform-eng")
    def run(query: str) -> str:
        # All LLM calls inside this function are tagged with agent context.
        ...

    # Or use the context manager inline:
    with capsera.tag(agent="my_agent"):
        response = client.messages.create(...)

    # Manual recording for proprietary clients:
    capsera.record(
        input_tokens=1_000,
        output_tokens=250,
        model="claude-sonnet-4-6",
        provider="anthropic",
    )

    # Flush remaining events (called automatically on normal process exit):
    capsera.shutdown()
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from ._budget_engine import BudgetEngine
from ._config import Config, _global_config
from ._context import AgentContext, _context_stack
from ._emitter import Emitter
from ._interceptor import Interceptor, _reset_budget_clients, get_interception_report
from ._models import BudgetExceededError, TokenEvent
from ._pricing import compute_cost
from ._prompt_analyzer import PromptAnalyzer
from ._router import Router
from ._version import __version__
from .frameworks import (
    LangChainCapseraCallback,
    autogen_agent,
    crewai_agent,
    crewai_task,
    framework_unit,
    langgraph_node,
)

__all__ = [
    # Every published package is expected to expose this; users and tooling
    # reach for capsera.__version__ before anything else when reporting a bug.
    "__version__",
    "init",
    "agent",
    "tag",
    "set_session",
    "clear_session",
    "record",
    "flush",
    "shutdown",
    "get_interception_report",
    "framework_unit",
    "langgraph_node",
    "crewai_agent",
    "crewai_task",
    "autogen_agent",
    "LangChainCapseraCallback",
    # Re-export types that users may want to type-hint against
    "Config",
    "TokenEvent",
    "AgentContext",
    "BudgetExceededError",
]

logger = logging.getLogger("capsera")

# Module-level singletons — set by init()
_emitter: Emitter | None = None
_interceptor: Interceptor | None = None
_router: Router | None = None
_analyzer: PromptAnalyzer | None = None
_budget_engine: BudgetEngine | None = None


# --------------------------------------------------------------------------- #
# init()
# --------------------------------------------------------------------------- #

def init(
    api_key: str = "dev",
    endpoint: str = "http://localhost:8000",
    agent_name: str | None = None,
    team: str | None = None,
    gateway: str | None = None,
    env: str = "development",
    flush_interval_ms: int = 500,
    enable_routing: bool = False,
    routing_rules: list[dict[str, Any]] | None = None,
    routing_refresh_interval: int = 60,
    critical_task_types: list[str] | None = None,
    enable_budget_enforcement: bool = True,
    enforcement_mode: str = "local",
    budget_cache_persist: bool = True,
    budget_check_timeout: float = 1.0,
    enable_prompt_analysis: bool = False,
    prompt_size_threshold: int = 4000,
    on_oversized_prompt: Callable[[int, int], None] | None = None,
    debug: bool = False,
    max_retries: int = 3,
    on_error: callable | None = None,
) -> None:
    """
    Initialise the Capsera SDK.

    Call this **once** at application startup, before any LLM client imports
    if possible (though it is safe to call after — we patch both the class and
    any live instances).

    Parameters
    ----------
    api_key:
        Your Capsera API key (``cap-...``).  Set to
        ``"dev"`` to run locally without a backend.
    endpoint:
        Base URL of your Capsera backend.  Defaults to ``http://localhost:8000``.
    agent_name:
        Default agent name applied to all calls that are not inside an
        ``@agent`` decorator or ``tag()`` context manager.
    team:
        Default team name for the same purpose.
    gateway:
        Declare the AI gateway all traffic rides through (``"bifrost"``,
        ``"litellm"``, ``"kong"``, ...). Only needed for self-hosted gateways
        whose base_url Capsera cannot recognise — well-known gateway hosts
        (OpenRouter, Portkey, Helicone, Cloudflare AI Gateway, a local
        LiteLLM proxy) and the LiteLLM/Portkey client libraries are detected
        and tagged automatically.
    env:
        Deployment environment tag (``"production"``, ``"staging"``, etc.).
    flush_interval_ms:
        How often (in milliseconds) the background thread ships events to the
        backend.  Lower values reduce data loss on crash; higher values reduce
        network overhead.
    enable_routing:
        Enable the model routing rules engine.  When ``True``, routing rules
        fetched from the backend may downgrade expensive models to cheaper
        ones before the API call is made.
    enforcement_mode:
        How budgets are enforced.

        ``"local"`` (default) evaluates in-process against a snapshot
        refreshed in the background, so no network call sits in front of the
        provider request. One snapshot fetch happens at init() time; after
        that, refreshes ride on the event flush the SDK already makes.

        ``"strict"`` asks the backend per call (``POST /v1/budgets/check``)
        and waits. Zero overshoot, at the cost of a round trip on every LLM
        request — appropriate when a single call over the cap is genuinely
        unacceptable.

        ``"off"`` records spend and enforces nothing. Passing
        ``enable_budget_enforcement=False`` forces this regardless of mode.
    budget_cache_persist:
        Whether the budget snapshot is cached on disk (system temp dir, keyed
        by a hash of endpoint + API key). Keeps enforcement working from the
        first call in short-lived processes. Set False for read-only or
        shared filesystems where the write is unwanted.
    enable_budget_enforcement:
        Enable a pre-call budget check against the backend. **On by default.**
        Blocking budgets stop calls before provider spend occurs; throttle and
        margin-downgrade actions are applied before the call as well. When no
        budget matches the call, the check returns "allowed" and the call
        proceeds. Network/backend errors fail open so application availability
        is preserved. Set ``enable_budget_enforcement=False`` to opt out (e.g.
        if the one extra round-trip per call is unacceptable for your latency
        budget and you do not rely on hard blocking).
    debug:
        Log detailed SDK activity to the ``capsera`` logger at INFO level.
    """
    global _emitter, _interceptor, _router, _analyzer, _budget_engine

    config = Config(
        api_key=api_key,
        endpoint=endpoint,
        default_agent=agent_name,
        default_team=team,
        default_gateway=gateway,
        env=env,
        flush_interval_ms=flush_interval_ms,
        enable_routing=enable_routing,
        routing_rules=routing_rules or [],
        routing_refresh_interval=routing_refresh_interval,
        critical_task_types=critical_task_types or [],
        enable_budget_enforcement=enable_budget_enforcement,
        enforcement_mode=enforcement_mode,
        budget_cache_persist=budget_cache_persist,
        budget_check_timeout=budget_check_timeout,
        enable_prompt_analysis=enable_prompt_analysis,
        prompt_size_threshold=prompt_size_threshold,
        on_oversized_prompt=on_oversized_prompt,
        debug=debug,
    )
    _global_config.set(config)

    # Drop any pooled budget-check clients from a previous init() so the new
    # endpoint/api_key/timeout take effect immediately.
    _reset_budget_clients()

    if _emitter is not None:
        # Re-init: flush and stop the previous emitter so events recorded
        # under the old configuration are delivered, not stranded. Patched
        # provider methods resolve the active emitter at call time (see
        # _interceptor's module runtime), so new calls use the new emitter.
        _emitter.shutdown()
    if _budget_engine is not None:
        # Re-init: the old engine holds a pooled client bound to the
        # previous endpoint/key. Its snapshot is stale by definition.
        _budget_engine.close()
    _budget_engine = (
        BudgetEngine(config) if config.enforcement_mode == "local" else None
    )
    _emitter = Emitter(
        config,
        max_retries=max_retries,
        on_error=on_error,
        budget_engine=_budget_engine,
    )
    _emitter.start()

    _router = Router(config) if enable_routing else None
    _analyzer = PromptAnalyzer(config) if enable_prompt_analysis else None

    if _budget_engine is not None:
        # Before the first patched call, not after: a short-lived process
        # would otherwise make every call it will ever make before the
        # emitter's first flush ever fetched a snapshot.
        _budget_engine.prime()

    _interceptor = Interceptor(config, _emitter, _router, _analyzer, _budget_engine)
    _interceptor.install()

    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.info(
            "capsera: initialised (endpoint=%s, env=%s, flush=%dms)",
            endpoint,
            env,
            flush_interval_ms,
        )


# --------------------------------------------------------------------------- #
# agent() — decorator
# --------------------------------------------------------------------------- #

def agent(
    name: str,
    team: str | None = None,
    task_type: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
):
    """
    Decorator that tags all LLM calls made within the decorated function.

    Works with both regular and async functions.  Nesting is supported — the
    innermost decorator wins for ``agent`` / ``team`` / ``task_type``.

    Parameters
    ----------
    name:
        Logical agent identifier (e.g. ``"reasoning_agent"``).
    team:
        Team name (e.g. ``"platform-eng"``).
    task_type:
        Task category (e.g. ``"forensic_query"``).
    customer_id:
        End-customer, account, or tenant identifier for gross-margin analysis.
    cost_center:
        Finance cost center or business unit identifier.

    Usage::

        @capsera.agent(name="summariser", team="nlp", task_type="summarisation")
        def summarise(text: str) -> str:
            return llm.invoke(text)

        # Async:
        @capsera.agent(name="classifier")
        async def classify(text: str) -> str:
            return await async_llm.ainvoke(text)
    """
    def decorator(func):  # type: ignore[return]
        ctx = AgentContext(
            agent=name,
            team=team,
            task_type=task_type,
            customer_id=customer_id,
            cost_center=cost_center,
        )

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):  # type: ignore[return]
                _context_stack.push(ctx)
                try:
                    return await func(*args, **kwargs)
                finally:
                    _context_stack.pop()
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):  # type: ignore[return]
            _context_stack.push(ctx)
            try:
                return func(*args, **kwargs)
            finally:
                _context_stack.pop()

        return sync_wrapper

    return decorator


# --------------------------------------------------------------------------- #
# tag() — context manager
# --------------------------------------------------------------------------- #

@contextmanager
def tag(
    agent: str,
    team: str | None = None,
    task_type: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that tags all LLM calls within the ``with`` block.

    Usage::

        with capsera.tag(agent="planner", task_type="planning"):
            result = llm.invoke(prompt)

    Parameters
    ----------
    agent:
        Logical agent identifier.
    team:
        Team name.
    task_type:
        Task category.
    session_id:
        End-user session identifier (optional; never contains prompt content).
    customer_id:
        End-customer, account, or tenant identifier for margin analysis.
    cost_center:
        Finance cost center or business unit identifier.
    """
    ctx = AgentContext(
        agent=agent,
        team=team,
        task_type=task_type,
        session_id=session_id,
        customer_id=customer_id,
        cost_center=cost_center,
    )
    _context_stack.push(ctx)
    try:
        yield
    finally:
        _context_stack.pop()


# --------------------------------------------------------------------------- #
# set_session() / clear_session() — request-level session propagation
# --------------------------------------------------------------------------- #

def set_session(session_id: str) -> None:
    """
    Set a thread-level session ID that flows to all LLM calls in this thread.

    Call this in your API middleware or request handler to associate all LLM
    calls within a request with the end-user's session, without needing to
    wrap every call in ``tag(session_id=...)``.

    Usage::

        # In FastAPI middleware or route handler:
        capsera.set_session(request.session_id)
        try:
            result = await process_query(...)
        finally:
            capsera.clear_session()
    """
    _context_stack.set_session(session_id)


def clear_session() -> None:
    """Clear the thread-level session ID set by ``set_session()``."""
    _context_stack.clear_session()


# --------------------------------------------------------------------------- #
# record() — manual recording
# --------------------------------------------------------------------------- #

def record(
    *,
    input_tokens: int,
    output_tokens: int,
    model: str,
    provider: str,
    cost_usd: float | None = None,
    agent_id: str | None = None,
    task_type: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    latency_ms: int = 0,
    gateway: str | None = None,
) -> None:
    """
    Manually record a token usage event.

    Use this for custom or proprietary LLM clients that are not automatically
    intercepted by the SDK.

    Parameters
    ----------
    input_tokens:
        Number of input / prompt tokens consumed.
    output_tokens:
        Number of output / completion tokens generated.
    model:
        Model identifier as returned by the provider (e.g. ``"claude-sonnet-4-6"``).
    provider:
        Provider name: ``"anthropic"``, ``"openai"``, ``"google"``, etc.
    cost_usd:
        If you know the exact cost, pass it here.  Otherwise the SDK computes
        it from the built-in pricing table.
    agent_id:
        Agent identifier.  Falls back to the current context stack / config default.
    task_type:
        Task category.
    session_id:
        End-user session identifier.
    customer_id:
        End-customer, account, or tenant identifier for margin analysis.
    cost_center:
        Finance cost center or business unit identifier.
    cache_read_tokens:
        Anthropic prompt cache read tokens (optional).
    cache_write_tokens:
        Anthropic prompt cache write tokens (optional).
    latency_ms:
        End-to-end request latency in milliseconds (optional).
    gateway:
        AI gateway the call rode through (``"litellm"``, ``"portkey"``,
        ``"bifrost"``, ...). Falls back to the ``init(gateway=...)`` default.
    """
    if _emitter is None:
        logger.warning(
            "capsera.record() called before capsera.init() — event dropped"
        )
        return

    cost: Decimal
    if cost_usd is not None:
        cost = Decimal(str(cost_usd))
    else:
        cost = compute_cost(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    ctx = _context_stack.current()
    cfg = _global_config.get()

    resolved_agent_id = (
        agent_id
        or (ctx.agent if ctx else None)
        or (cfg.default_agent if cfg else None)
        or "unknown"
    )
    resolved_team = (ctx.team if ctx else None) or (cfg.default_team if cfg else None)
    resolved_task_type = task_type or (ctx.task_type if ctx else None)
    resolved_session_id = session_id or (ctx.session_id if ctx else None) or _context_stack.get_session()
    resolved_customer_id = customer_id or (ctx.customer_id if ctx else None)
    resolved_cost_center = cost_center or (ctx.cost_center if ctx else None)

    event = TokenEvent(
        id=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
        request_id=str(uuid4()),
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=float(cost),
        agent_id=resolved_agent_id,
        team_id=resolved_team,
        task_type=resolved_task_type,
        session_id=resolved_session_id,
        customer_id=resolved_customer_id,
        cost_center=resolved_cost_center,
        env=(cfg.env if cfg else "development"),
        latency_ms=latency_ms,
        gateway=gateway or (cfg.default_gateway if cfg else None),
    )
    _emitter.enqueue(event)


# --------------------------------------------------------------------------- #
# flush() / shutdown()
# --------------------------------------------------------------------------- #

def flush() -> None:
    """
    Force-flush all pending events to the backend immediately.

    Useful in short-lived scripts or test teardown to ensure no events are
    lost before the process exits.
    """
    if _emitter:
        _emitter.flush()


def shutdown() -> None:
    """
    Flush all pending events and gracefully stop the background thread.

    This is called automatically via ``atexit`` if you prefer not to manage
    lifecycle manually.
    """
    if _emitter:
        _emitter.shutdown()


# Register a best-effort shutdown hook so events are flushed on normal exit.
import atexit  # noqa: E402

atexit.register(shutdown)
