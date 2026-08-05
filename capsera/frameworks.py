"""Lightweight helpers for agent frameworks.

These helpers intentionally avoid hard dependencies on LangGraph, CrewAI, or
LangChain. They provide clean attribution context around framework units while
the provider-level interceptor still captures the actual LLM calls.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from ._context import AgentContext, _context_stack


def _context(
    *,
    agent: str,
    team: Optional[str] = None,
    task_type: Optional[str] = None,
    session_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
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
    team: Optional[str] = None,
    task_type: Optional[str] = None,
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
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
    team: Optional[str] = None,
    task_type: Optional[str] = None,
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
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
    team: Optional[str] = None,
    task_type: Optional[str] = None,
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
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
    team: Optional[str] = None,
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for CrewAI task functions."""

    return framework_unit(
        agent=name,
        team=team,
        task_type="crewai_task",
        customer_id=customer_id,
        cost_center=cost_center,
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
        team: Optional[str] = None,
        task_type: Optional[str] = "langchain",
        session_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
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
