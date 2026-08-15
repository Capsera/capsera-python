"""
Context stack for agent tagging.

The @agent decorator and the tag() context manager push/pop AgentContext
entries; the interceptor reads the top entry to attribute LLM calls to the
correct agent, team, and task_type.

Isolation is provided by ``contextvars`` (not ``threading.local``): every
thread still gets its own independent stack, and — crucially — every asyncio
task does too. With a plain thread-local stack, two agents awaited
concurrently on the same event loop (``asyncio.gather``) interleave their
push/pop operations on one shared list and steal each other's attribution;
ContextVar snapshots the stack per task, so each coroutine only ever sees its
own scopes. The stack is stored as an immutable tuple so a ``set()`` in one
task can never mutate state visible to a sibling task.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass
class AgentContext:
    """Carries attribution metadata for a single logical scope."""

    agent: str | None = None
    team: str | None = None
    task_type: str | None = None
    session_id: str | None = None
    customer_id: str | None = None
    cost_center: str | None = None


_stack_var: contextvars.ContextVar[tuple[AgentContext, ...]] = contextvars.ContextVar(
    "capsera_context_stack", default=()
)
_session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "capsera_session_id", default=None
)


class ContextStack:
    """
    Stack of AgentContext objects, isolated per thread AND per asyncio task.

    Methods are safe to call from any thread or task; each execution context
    sees its own independent stack courtesy of contextvars.
    """

    def push(self, ctx: AgentContext) -> None:
        _stack_var.set(_stack_var.get() + (ctx,))

    def pop(self) -> None:
        stack = _stack_var.get()
        if stack:
            _stack_var.set(stack[:-1])

    def current(self) -> AgentContext | None:
        """Return the innermost context, or None if the stack is empty."""
        stack = _stack_var.get()
        return stack[-1] if stack else None

    def depth(self) -> int:
        return len(_stack_var.get())

    def set_session(self, session_id: str | None) -> None:
        """Set a request-level session_id that applies to all calls in this
        thread (and, in async servers, this task)."""
        _session_var.set(session_id)

    def get_session(self) -> str | None:
        """Get the ambient session_id, if set."""
        return _session_var.get()

    def clear_session(self) -> None:
        """Remove the ambient session_id."""
        _session_var.set(None)


# Module-level singleton — all submodules share this instance
_context_stack = ContextStack()
