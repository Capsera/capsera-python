"""
Tests for context stack (the "enricher" layer) — agent tagging and
context propagation across decorators, context managers, and threads.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

import capsera
from capsera._context import AgentContext, ContextStack, _context_stack


class TestContextStack:
    """Unit tests for the ContextStack class in isolation."""

    def setup_method(self) -> None:
        """Ensure stack is clean before each test."""
        # Drain any leftover state from previous tests
        stack = ContextStack()
        self._stack = stack

    def test_empty_stack_returns_none(self) -> None:
        assert self._stack.current() is None

    def test_single_push(self) -> None:
        self._stack.push(AgentContext(agent="a"))
        assert self._stack.current().agent == "a"
        self._stack.pop()

    def test_pop_from_empty_is_safe(self) -> None:
        self._stack.pop()  # must not raise

    def test_depth_tracking(self) -> None:
        assert self._stack.depth() == 0
        self._stack.push(AgentContext(agent="x"))
        assert self._stack.depth() == 1
        self._stack.push(AgentContext(agent="y"))
        assert self._stack.depth() == 2
        self._stack.pop()
        assert self._stack.depth() == 1
        self._stack.pop()
        assert self._stack.depth() == 0

    def test_nesting_returns_innermost(self) -> None:
        self._stack.push(AgentContext(agent="outer", team="team-a"))
        self._stack.push(AgentContext(agent="inner", team="team-b"))
        ctx = self._stack.current()
        assert ctx.agent == "inner"
        assert ctx.team == "team-b"
        self._stack.pop()
        ctx2 = self._stack.current()
        assert ctx2.agent == "outer"
        self._stack.pop()

    def test_thread_isolation(self) -> None:
        """Each thread sees its own stack."""
        seen: dict[str, str | None] = {}

        def worker(name: str) -> None:
            self._stack.push(AgentContext(agent=name))
            time.sleep(0.05)
            seen[name] = self._stack.current().agent  # type: ignore[union-attr]
            self._stack.pop()

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(4):
            assert seen[f"t{i}"] == f"t{i}"


class TestAgentDecorator:
    """Tests for @capsera.agent()."""

    def test_sync_function_tagged(self) -> None:
        captured: list[str | None] = []

        @capsera.agent(name="dec_agent", team="dec_team", task_type="dec_task")
        def my_func() -> None:
            captured.append(_context_stack.current().agent)  # type: ignore[union-attr]
            captured.append(_context_stack.current().team)  # type: ignore[union-attr]
            captured.append(_context_stack.current().task_type)  # type: ignore[union-attr]

        my_func()
        assert captured == ["dec_agent", "dec_team", "dec_task"]

    def test_context_restored_after_function(self) -> None:
        @capsera.agent(name="temp")
        def my_func() -> None:
            pass

        my_func()
        # Stack should be empty (or back to what it was before)
        ctx = _context_stack.current()
        # We can't assert None because outer tests may have their own context,
        # but temp should not be on top.
        assert ctx is None or ctx.agent != "temp"

    def test_exception_still_pops(self) -> None:
        @capsera.agent(name="boom_agent")
        def my_func() -> None:
            raise ValueError("intentional")

        with pytest.raises(ValueError):
            my_func()

        # Agent must be popped even after exception
        ctx = _context_stack.current()
        assert ctx is None or ctx.agent != "boom_agent"

    def test_async_function_tagged(self) -> None:
        captured: list[str | None] = []

        @capsera.agent(name="async_agent")
        async def my_async_func() -> None:
            captured.append(_context_stack.current().agent)  # type: ignore[union-attr]

        asyncio.run(my_async_func())
        assert captured == ["async_agent"]

    def test_return_value_preserved(self) -> None:
        @capsera.agent(name="ret_agent")
        def my_func() -> int:
            return 42

        assert my_func() == 42

    def test_async_return_value_preserved(self) -> None:
        @capsera.agent(name="async_ret")
        async def my_func() -> str:
            return "hello"

        result = asyncio.run(my_func())
        assert result == "hello"

    def test_functools_wraps_preserves_name(self) -> None:
        @capsera.agent(name="named_agent")
        def original_function() -> None:
            """Original docstring."""

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."


class TestTagContextManager:
    """Tests for the capsera.tag() context manager."""

    def test_basic_tagging(self) -> None:
        with capsera.tag(agent="ctx_agent", team="ctx_team"):
            assert _context_stack.current().agent == "ctx_agent"  # type: ignore[union-attr]
            assert _context_stack.current().team == "ctx_team"  # type: ignore[union-attr]

    def test_enterprise_attribution_tags(self) -> None:
        with capsera.tag(
            agent="ctx_agent",
            customer_id="customer-1",
            cost_center="support-ai",
        ):
            ctx = _context_stack.current()
            assert ctx is not None
            assert ctx.customer_id == "customer-1"
            assert ctx.cost_center == "support-ai"

    def test_session_id_propagated(self) -> None:
        with capsera.tag(agent="a", session_id="sess-123"):
            assert _context_stack.current().session_id == "sess-123"  # type: ignore[union-attr]

    def test_context_restored_after_exit(self) -> None:
        with capsera.tag(agent="inner_ctx"):
            pass
        ctx = _context_stack.current()
        assert ctx is None or ctx.agent != "inner_ctx"

    def test_exception_in_block_still_pops(self) -> None:
        with pytest.raises(RuntimeError):
            with capsera.tag(agent="err_agent"):
                raise RuntimeError("test error")

        ctx = _context_stack.current()
        assert ctx is None or ctx.agent != "err_agent"

    def test_nested_tags(self) -> None:
        with capsera.tag(agent="outer"):
            with capsera.tag(agent="inner"):
                assert _context_stack.current().agent == "inner"  # type: ignore[union-attr]
            assert _context_stack.current().agent == "outer"  # type: ignore[union-attr]
