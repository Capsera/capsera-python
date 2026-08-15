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


class _FakeChatClient:
    """Stands in for an AutoGen ChatCompletionClient (duck-typed)."""

    def __init__(self) -> None:
        self.seen_contexts: list = []
        self.model_info = {"family": "fake", "vision": False}

    async def create(self, messages, **kwargs):
        self.seen_contexts.append(_context_stack.current())
        return {"content": "ok", "messages": messages}

    async def create_stream(self, messages, **kwargs):
        for chunk in ("a", "b"):
            self.seen_contexts.append(_context_stack.current())
            yield chunk

    def count_tokens(self, messages) -> int:
        return 7


def test_autogen_agent_attributes_create_calls() -> None:
    inner = _FakeChatClient()
    client = capsera.autogen_agent(
        inner, name="planner", team="bi", task_type="planning"
    )

    result = asyncio.run(client.create([{"role": "user", "content": "hi"}]))
    assert result["content"] == "ok"
    ctx = inner.seen_contexts[0]
    assert ctx.agent == "planner"
    assert ctx.team == "bi"
    assert ctx.task_type == "planning"
    assert _context_stack.current() is None


def test_autogen_agent_attributes_whole_stream() -> None:
    inner = _FakeChatClient()
    client = capsera.autogen_agent(inner, name="analyst")

    async def consume():
        return [chunk async for chunk in client.create_stream([])]

    assert asyncio.run(consume()) == ["a", "b"]
    assert [c.agent for c in inner.seen_contexts] == ["analyst", "analyst"]
    assert _context_stack.current() is None


def test_autogen_agent_pops_context_on_error() -> None:
    class _Boom:
        async def create(self, messages):
            raise RuntimeError("provider down")

    client = capsera.autogen_agent(_Boom(), name="critic")
    try:
        asyncio.run(client.create([]))
    except RuntimeError:
        pass
    assert _context_stack.current() is None


def test_autogen_agent_passes_other_attributes_through() -> None:
    inner = _FakeChatClient()
    client = capsera.autogen_agent(inner, name="chart-writer")
    assert client.count_tokens([]) == 7
    assert client.model_info == {"family": "fake", "vision": False}


def test_autogen_agent_isolates_concurrent_agents() -> None:
    # Two agents sharing one event loop must never steal each other's
    # attribution — the contextvars-based stack keeps each task's pushes
    # invisible to its sibling.
    inner_a, inner_b = _FakeChatClient(), _FakeChatClient()
    a = capsera.autogen_agent(inner_a, name="agent-a")
    b = capsera.autogen_agent(inner_b, name="agent-b")

    async def run_both():
        await asyncio.gather(a.create([]), b.create([]))

    asyncio.run(run_both())
    assert inner_a.seen_contexts[0].agent == "agent-a"
    assert inner_b.seen_contexts[0].agent == "agent-b"
