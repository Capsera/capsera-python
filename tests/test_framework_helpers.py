from __future__ import annotations

import asyncio

import capsera
from capsera._context import _context_stack


def test_langgraph_node_decorator_sets_context() -> None:
    captured = []

    @capsera.langgraph_node(
        "planner-node",
        team="agents",
        task_type="planning",
        customer_id="customer-a",
        cost_center="product-ai",
    )
    def node(state):
        ctx = _context_stack.current()
        captured.append(ctx)
        return state

    assert node({"ok": True}) == {"ok": True}
    ctx = captured[0]
    assert ctx.agent == "planner-node"
    assert ctx.team == "agents"
    assert ctx.task_type == "planning"
    assert ctx.customer_id == "customer-a"
    assert ctx.cost_center == "product-ai"
    assert _context_stack.current() is None


def test_crewai_task_decorator_sets_task_context() -> None:
    captured = []

    @capsera.crewai_task("research-task", team="research", customer_id="customer-b")
    def task():
        captured.append(_context_stack.current())

    task()
    assert captured[0].agent == "research-task"
    assert captured[0].task_type == "crewai_task"
    assert captured[0].team == "research"
    assert captured[0].customer_id == "customer-b"


def test_framework_unit_supports_async_functions() -> None:
    captured = []

    @capsera.framework_unit(agent="async-node", task_type="retrieval")
    async def node():
        captured.append(_context_stack.current())
        return "done"

    assert asyncio.run(node()) == "done"
    assert captured[0].agent == "async-node"
    assert captured[0].task_type == "retrieval"
    assert _context_stack.current() is None


def test_langchain_callback_pushes_and_pops_context() -> None:
    callback = capsera.LangChainCapseraCallback(
        agent="chain",
        team="platform",
        task_type="rag",
        session_id="session-1",
        customer_id="customer-c",
        cost_center="support-ai",
    )

    callback.on_llm_start({}, ["hello"])
    ctx = _context_stack.current()
    assert ctx.agent == "chain"
    assert ctx.team == "platform"
    assert ctx.task_type == "rag"
    assert ctx.session_id == "session-1"
    assert ctx.customer_id == "customer-c"
    assert ctx.cost_center == "support-ai"

    callback.on_llm_end({})
    assert _context_stack.current() is None


def test_langchain_callback_error_pops_context() -> None:
    callback = capsera.LangChainCapseraCallback(agent="chain")
    callback.on_llm_start({}, ["hello"])
    callback.on_llm_error(RuntimeError("boom"))
    assert _context_stack.current() is None
