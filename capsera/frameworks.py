"""Lightweight helpers for agent frameworks.

These helpers intentionally avoid hard dependencies on LangGraph, CrewAI, or
LangChain. They provide clean attribution context around framework units while
the provider-level interceptor still captures the actual LLM calls.
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from ._context import AgentContext, _context_stack


def _context(
    *,
    agent: str,
    team: str | None = None,
    task_type: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> AgentContext:
    return AgentContext(
        agent=agent,
        team=team,
        task_type=task_type,
        session_id=session_id,
        customer_id=customer_id,
        cost_center=cost_center,
    )


def framework_unit(
    *,
    agent: str,
    team: str | None = None,
    task_type: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a framework node/task so LLM calls inside it are attributed."""

    ctx = _context(
        agent=agent,
        team=team,
        task_type=task_type,
        customer_id=customer_id,
        cost_center=cost_center,
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _context_stack.push(ctx)
                try:
                    return await func(*args, **kwargs)
                finally:
                    _context_stack.pop()

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _context_stack.push(ctx)
            try:
                return func(*args, **kwargs)
            finally:
                _context_stack.pop()

        return sync_wrapper

    return decorator


def langgraph_node(
    name: str,
    *,
    team: str | None = None,
    task_type: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for LangGraph node functions."""

    return framework_unit(
        agent=name,
        team=team,
        task_type=task_type,
        customer_id=customer_id,
        cost_center=cost_center,
    )


def crewai_agent(
    name: str,
    *,
    team: str | None = None,
    task_type: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for CrewAI agent entrypoints."""

    return framework_unit(
        agent=name,
        team=team,
        task_type=task_type,
        customer_id=customer_id,
        cost_center=cost_center,
    )


def crewai_task(
    name: str,
    *,
    team: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for CrewAI task functions."""

    return framework_unit(
        agent=name,
        team=team,
        task_type="crewai_task",
        customer_id=customer_id,
        cost_center=cost_center,
    )


class _AttributedChatClient:
    """Duck-typed proxy around an AutoGen ``ChatCompletionClient``.

    Pushes the agent's context for exactly the span of each ``create`` /
    ``create_stream`` call and passes every other attribute straight through,
    so one shared underlying client (or factory-built clients) can serve many
    agents while each agent's completions are attributed to its own name.
    Deliberately not a ``ChatCompletionClient`` subclass: autogen stays an
    optional dependency, and the ABC's abstract surface differs across
    versions — duck typing tracks the real contract (AssistantAgent only ever
    calls methods on the client, it never isinstance-checks it).
    """

    def __init__(self, inner: Any, ctx: AgentContext) -> None:
        self._capsera_inner = inner
        self._capsera_ctx = ctx

    def __getattr__(self, item: str) -> Any:
        return getattr(self._capsera_inner, item)

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return f"AutogenAgentClient({self._capsera_inner!r})"

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        _context_stack.push(self._capsera_ctx)
        try:
            result = self._capsera_inner.create(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            _context_stack.pop()

    async def create_stream(self, *args: Any, **kwargs: Any) -> Any:
        # Async generator: the context must span the whole iteration, not just
        # the call that constructs the stream — the provider request happens
        # while the caller iterates. The ``finally`` pop also runs when the
        # consumer abandons the stream (aclose/GC), and pop() on an already-
        # unwound stack is a safe no-op.
        _context_stack.push(self._capsera_ctx)
        try:
            stream = self._capsera_inner.create_stream(*args, **kwargs)
            if inspect.isawaitable(stream):
                stream = await stream
            async for item in stream:
                yield item
        finally:
            _context_stack.pop()


def autogen_agent(
    model_client: Any,
    *,
    name: str,
    team: str | None = None,
    task_type: str | None = None,
    customer_id: str | None = None,
    cost_center: str | None = None,
) -> Any:
    """Wrap an AutoGen ``ChatCompletionClient`` so its completions are
    attributed to ``name``.

    AutoGen agents (``AssistantAgent`` et al.) are declared as objects, not
    functions, so there is nothing to decorate — attribution instead rides the
    model client each agent owns::

        planner = AssistantAgent(
            name="planner",
            model_client=capsera.autogen_agent(build_client(), name="planner"),
        )

    Context isolation is ``contextvars``-based, so concurrently-running agents
    on one event loop each keep their own attribution. The wrapper is inert
    when ``capsera.init()`` was never called — exactly like the decorators.
    """

    return _AttributedChatClient(
        model_client,
        _context(
            agent=name,
            team=team,
            task_type=task_type,
            customer_id=customer_id,
            cost_center=cost_center,
        ),
    )


class LangChainCapseraCallback:
    """Minimal LangChain-compatible callback handler.

    LangChain callback managers call `on_llm_start` before provider execution
    and `on_llm_end` / `on_llm_error` afterward. Keeping context here lets the
    provider interceptor attribute the underlying OpenAI/Anthropic/Gemini call
    to the LangChain chain or runnable.
    """

    def __init__(
        self,
        *,
        agent: str,
        team: str | None = None,
        task_type: str | None = "langchain",
        session_id: str | None = None,
        customer_id: str | None = None,
        cost_center: str | None = None,
    ) -> None:
        self._ctx = _context(
            agent=agent,
            team=team,
            task_type=task_type,
            session_id=session_id,
            customer_id=customer_id,
            cost_center=cost_center,
        )
        self._depth = 0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        _context_stack.push(self._ctx)
        self._depth += 1

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._pop()

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        self._pop()

    def _pop(self) -> None:
        if self._depth <= 0:
            return
        _context_stack.pop()
        self._depth -= 1
