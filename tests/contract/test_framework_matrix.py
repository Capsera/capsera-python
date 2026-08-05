"""Framework contract tests — real frameworks, real clients, local mock.

Each cell drives a minimal real-framework pipeline and asserts the capture
contract (ledger == events, exact tokens), attribution through capsera's
context helpers, and caller-location resolution to USER code rather than
framework internals (FID-5 — the skip-list's whole job).

Clients are configured via ANTHROPIC_BASE_URL / OPENAI_BASE_URL, the same
mechanism the eval corpus uses. Frameworks not installed here skip cleanly
(CrewAI/AutoGen/Haystack/DSPy cells arrive with `pip install -e
"sdk[test-frameworks]"`).
"""
from __future__ import annotations

import pytest

langchain_openai = pytest.importorskip("langchain_openai")
langchain_anthropic = pytest.importorskip("langchain_anthropic")

from capsera import tag  # noqa: E402
from capsera.frameworks import langgraph_node  # noqa: E402

PROMPT = "Say OK"


@pytest.fixture
def provider_env(monkeypatch, mock_llm_server):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_llm_server.base_url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"{mock_llm_server.base_url}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")


class TestLangChainContract:
    def test_chat_openai_capture_attribution_and_caller(
        self, provider_env, mock_llm_server, capsera_capture
    ):
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", max_retries=0)
        with tag("rag-drafter", team="rag"):
            result = llm.invoke(PROMPT)

        assert result.content.startswith("echo:")  # NI through the framework
        events = capsera_capture.drain()
        ledger = mock_llm_server.ledger
        assert len(ledger) == 1 and len(events) == 1
        ev = events[0]
        assert ev.input_tokens == ledger[0]["usage"]["input_tokens"]
        assert ev.output_tokens == ledger[0]["usage"]["output_tokens"]
        assert ev.agent_id == "rag-drafter"
        assert ev.team_id == "rag"
        assert "test_framework_matrix" in (ev.caller_file or ""), (
            f"caller must resolve through langchain internals to user code, "
            f"got {ev.caller_file!r}")

    def test_chat_anthropic_capture(
        self, provider_env, mock_llm_server, capsera_capture
    ):
        llm = langchain_anthropic.ChatAnthropic(model="claude-haiku-4-5",
                                                max_retries=0)
        with tag("reviewer"):
            result = llm.invoke(PROMPT)

        assert result.content.startswith("echo:")
        events = capsera_capture.drain()
        ledger = mock_llm_server.ledger
        assert len(ledger) == 1 and len(events) == 1
        assert events[0].provider == "anthropic"
        assert events[0].agent_id == "reviewer"
        assert events[0].input_tokens == ledger[0]["usage"]["input_tokens"]

    def test_openai_embeddings_capture(
        self, provider_env, mock_llm_server, capsera_capture
    ):
        embedder = langchain_openai.OpenAIEmbeddings(
            model="text-embedding-3-small", max_retries=0,
            check_embedding_ctx_length=False)
        with tag("retriever", team="rag", task_type="embedding"):
            vector = embedder.embed_query(PROMPT)

        assert vector  # NI: app still gets its embedding
        events = capsera_capture.drain()
        ledger = mock_llm_server.ledger
        assert len(ledger) == 1 and len(events) == 1
        assert events[0].agent_id == "retriever"
        assert events[0].task_type == "embedding"
        assert events[0].input_tokens == ledger[0]["usage"]["input_tokens"]


class TestLangGraphContract:
    def test_two_node_graph_attributes_per_node(
        self, provider_env, mock_llm_server, capsera_capture
    ):
        langgraph_graph = pytest.importorskip("langgraph.graph")
        from typing_extensions import TypedDict

        planner = langchain_anthropic.ChatAnthropic(model="claude-haiku-4-5",
                                                    max_retries=0)
        worker = langchain_openai.ChatOpenAI(model="gpt-4o-mini",
                                             max_retries=0)

        class State(TypedDict):
            plan: str
            report: str

        @langgraph_node("planner", team="incident")
        def plan_node(state: State) -> State:
            return {**state, "plan": planner.invoke(PROMPT).content}

        @langgraph_node("reporter", team="incident")
        def report_node(state: State) -> State:
            return {**state, "report": worker.invoke(state["plan"]).content}

        graph = langgraph_graph.StateGraph(State)
        graph.add_node("plan", plan_node)
        graph.add_node("report", report_node)
        graph.add_edge(langgraph_graph.START, "plan")
        graph.add_edge("plan", "report")
        graph.add_edge("report", langgraph_graph.END)
        out = graph.compile().invoke({"plan": "", "report": ""})

        assert out["report"].startswith("echo:")
        events = capsera_capture.drain()
        ledger = mock_llm_server.ledger
        assert len(ledger) == 2 and len(events) == 2
        by_agent = {ev.agent_id: ev for ev in events}
        assert set(by_agent) == {"planner", "reporter"}
        assert by_agent["planner"].provider == "anthropic"
        assert by_agent["reporter"].provider == "openai"
        assert all(ev.team_id == "incident" for ev in events)
        # exact token join: match each event to its ledger row by provider
        rows = {r["provider"]: r for r in ledger}
        assert by_agent["planner"].input_tokens == \
            rows["anthropic"]["usage"]["input_tokens"]
        assert by_agent["reporter"].input_tokens == \
            rows["openai"]["usage"]["input_tokens"]
